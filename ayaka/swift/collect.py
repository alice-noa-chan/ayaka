"""Adapt datasets and collect uncalibrated T=1 letter reads."""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ayaka.eval.read_artifact import fingerprint

from .binding import SwiftReadIndex, make_swift_binding
from .grouping import read_question, validate_option_count
from .losses import target_distribution
from .policy import FALSE_LABELS
from .prompt import Question, parse_question, validate_prompt_variant
from .readers import LetterReader, add_reader_arguments, reader_from_args


@dataclass(frozen=True)
class DatasetItem:
    id: str
    source: str
    tier: str
    state: object
    question: Question
    gold: str | dict[str, float]
    public: bool
    gold_distribution: dict[str, float] | None = None
    family: str | None = None
    case_id: str | None = None
    split: str | None = None
    cluster_id: str | None = None
    lineage_ids: tuple[str, ...] = ()
    adapter: str = "jevbench"


def infer_tier(record: dict, source: str) -> str:
    metadata = record.get("metadata", {})
    hints = [str(record.get(key, "")) for key in ("tier", "family", "split", "id")]
    hints += [str(metadata.get(key, "")) for key in ("tier", "task_family", "split")]
    hints.append(Path(source).stem)
    if any("judge" in hint.lower() for hint in hints):
        return "judge"
    for hint in hints:
        hint = hint.lower()
        for prefix, tier in (
            ("easy", "easy"),
            ("original", "standard"),
            ("standard", "standard"),
            ("hard", "hard"),
        ):
            if hint == prefix or hint.startswith((prefix + "-", prefix + "_", prefix + "/")):
                return tier
    return "standard"


def _is_public(record: dict, source: str) -> bool:
    split = record.get("split", record.get("metadata", {}).get("split", ""))
    parts = Path(source.replace("\\", "/")).parts
    return (
        bool(record.get("public"))
        or split == "public"
        or any(part.lower() in ("public", "jevbench_public") for part in parts)
    )


def adapt_jevbench(record: dict, source: str = "inline") -> DatasetItem:
    _recorded_split(record)
    question = dict(record["question"])
    kind = question["type"]
    original_labels = [str(label) for label in record["labels"]]
    label_map = {label: label for label in original_labels}
    if kind == "choice":
        criteria = question.get("criteria") or {}
        question["criteria"] = {
            label: criteria.get(label, label) if isinstance(criteria, dict) else criteria[i]
            for i, label in enumerate(original_labels)
        }
    elif kind == "noul":
        if len(original_labels) != 2:
            raise ValueError("noul dataset items need two labels")
        false = next((label for label in original_labels if label.lower() in FALSE_LABELS), None)
        if false is None:
            raise ValueError("noul dataset lacks a false/no label")
        label_map = {label: "false" if label == false else "true" for label in original_labels}
    parsed = parse_question(question)
    expected = record["expected"]
    if isinstance(expected, dict):
        gold: str | dict[str, float] = {label_map[str(label)]: p for label, p in expected.items()}
    else:
        gold = label_map[str(expected)]
    distribution = (record.get("provenance") or {}).get("gold_probs")
    if distribution is None:
        distribution = record.get("target_distribution") or record.get("gold_distribution")
    if distribution is not None:
        distribution = {label_map[str(label)]: p for label, p in distribution.items()}
    _validate_gold(parsed.labels, gold, distribution)
    cygnet = "cygnet" in str(record.get("source", source)).lower()
    public = _is_public(record, source)
    identity = str(
        record["id"]
        if cygnet
        else record.get("group")
        or (
            record["id"]
            if public
            else record.get("case_id")
            or (record.get("metadata") or {}).get("case_id")
            or record["id"]
        )
    )
    return DatasetItem(
        str(record["id"]),
        str(record.get("source") or source),
        infer_tier(record, source),
        record.get("state", ""),
        parsed,
        gold,
        public,
        distribution,
        record.get("family"),
        identity,
        record.get("split", (record.get("metadata") or {}).get("split")),
        identity,
        tuple(record.get("lineage_ids") or (record.get("metadata") or {}).get("lineage_ids") or ()),
        "cygnet" if cygnet else "jevbench",
    )


def _validate_gold(
    labels: list[str], gold: str | dict[str, float], distribution: dict[str, float] | None
) -> None:
    target_distribution(
        {
            "labels": labels,
            "raw_probs": dict.fromkeys(labels, 1 / len(labels)),
            "gold": gold,
            "gold_distribution": distribution,
        }
    )


def adapt_canonical(
    record: dict, source: str = "inline", line_number: int = 1
) -> list[DatasetItem]:
    _recorded_split(record)
    metadata = record.get("metadata", {})
    if is_image_record(record):
        # Swift reads text only; an image row's text is just a placeholder such as
        # "Use only the attached document.", so its read would measure nothing and can
        # collide with other image rows across calibration/dev.
        return []
    base_id = str(
        record.get("id") or metadata.get("source_example_id") or f"{source}:{line_number}"
    )
    case_id = str(
        metadata.get("source_lineage")
        or metadata.get("case_facts_sha256")
        or record.get("case_id")
        or metadata.get("case_id")
        or base_id
    )
    lineages = metadata.get("lineage_ids") or []
    if isinstance(lineages, str):
        lineages = [lineages]
    lineages = tuple(
        sorted(
            {
                str(value)
                for value in [
                    *lineages,
                    metadata.get("source_lineage"),
                    metadata.get("case_facts_sha256"),
                    case_id,
                ]
                if value is not None
            }
        )
    )
    items = []
    for index, question in enumerate(record["questions"]):
        # Canonical calibration datasets only have tiers when explicitly recorded.
        tier = question.get("tier") or record.get("tier") or metadata.get("tier") or "standard"
        tier = "standard" if tier == "original" else tier
        kind = question["type"]
        candidates = question["candidates"]
        labels = [str(candidate["id"]) for candidate in candidates]
        mapping = dict(zip(labels, labels, strict=True))
        if kind == "noul":
            mapping = {
                label: "false" if label.lower() in FALSE_LABELS else "true" for label in labels
            }
        elif kind == "score":
            if any(candidate.get("ordinal") is None for candidate in candidates):
                raise ValueError("canonical score candidates need ordinals")
            mapping = {str(candidate["id"]): str(candidate["ordinal"]) for candidate in candidates}
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("candidate labels or ordinals must be unique")
        criteria = {
            mapping[str(candidate["id"])]: candidate.get("description") for candidate in candidates
        }
        parsed = parse_question(
            {"type": kind, "instructions": question.get("instruction", ""), "criteria": criteria}
        )
        # Canonical partition generators serialize some one-hot values as bool.
        # Convert the source representation here; saved-row validation is strict.
        target = {
            mapping[str(label)]: float(p) if type(p) in (bool, int, float) else p
            for label, p in question["target_distribution"].items()
        }
        gold = max(target, key=target.__getitem__)
        _validate_gold(parsed.labels, gold, target)
        items.append(
            DatasetItem(
                f"{base_id}/{question.get('id', index)}",
                str(metadata.get("source") or source),
                tier,
                record.get("state", ""),
                parsed,
                gold,
                _is_public(record, source),
                target,
                question.get("family") or record.get("family") or metadata.get("task_family"),
                case_id,
                record.get("split", metadata.get("split")),
                case_id,
                lineages,
                "canonical",
            )
        )
    return items


def is_image_record(record: dict) -> bool:
    metadata = record.get("metadata") or {}
    return (
        metadata.get("modality") == "image"
        or bool(metadata.get("media"))
        or bool(record.get("media"))
    )


def _recorded_split(record):
    metadata = record.get("metadata") or {}
    if "split" in record and "split" in metadata and record["split"] != metadata["split"]:
        raise ValueError("row and metadata split conflict")


def iter_dataset(paths: list[str | Path]) -> Iterator[DatasetItem]:
    for requested in paths:
        path = Path(requested)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            with file.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "questions" in record:
                        yield from adapt_canonical(record, str(file), number)
                    else:
                        yield adapt_jevbench(record, str(file))


def collect(
    items: Iterator[DatasetItem],
    reader: LetterReader,
    output: str | Path,
    *,
    limit: int | None = None,
    group_size: int = 20,
    state_format: str = "pretty",
    concurrency: int = 1,
    model: str | None = None,
    revision: str | None = None,
    prompt_variant: str = "min",
    diagnostic: bool = False,
    reasoned: bool = False,
    direct_reads: list[dict] | None = None,
    trace_max_tokens: int = 384,
) -> int:
    """Append reads in input order, with at most concurrency decisions in flight.

    Reuse exact bindings only; limit counts new decisions. Bulk latency_s is
    backend timing under load, not a serial HTTP latency measurement.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    validate_prompt_variant(prompt_variant)
    validate_option_count(2, group_size)
    if state_format not in ("pretty", "compact"):
        raise ValueError("state_format must be pretty or compact")
    model = model or getattr(reader, "model", getattr(reader, "model_id_or_path", "fake"))
    diagnostic = diagnostic or getattr(reader, "backend", None) == "fixture"
    output = Path(output)
    revision = revision or getattr(reader, "revision", None)
    if revision is None and getattr(reader, "backend", None) == "fixture":
        revision = "fixture-model-v1"
    if getattr(reader, "backend", None) in ("hf", "vllm") and (
        not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision)
    ):
        raise ValueError(
            "production collection requires --revision as an immutable model/tokenizer commit"
        )
    index = SwiftReadIndex(load_reads([output]) if output.exists() else [])
    direct_index = SwiftReadIndex(direct_reads or [])
    if reasoned:
        from .reasoning import reasoning_recipe

        recipe = reasoning_recipe(trace_max_tokens)
        if not direct_reads:
            raise ValueError("--reasoned requires bound --direct-reads")
    if limit == 0:
        return 0
    # Preflight the entire request before launching even one read. A mismatch
    # later in the input must not spend inference on earlier unseen ids.
    planned = []
    seen = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate question id in requested dataset: {item.id}")
        seen.add(item.id)
        binding = make_swift_binding(
            item,
            reader,
            model=model,
            revision=revision,
            prompt_variant=prompt_variant,
            state_format=state_format,
            group_size=group_size,
            diagnostic=diagnostic,
        )
        original = None
        if reasoned:
            from .router import candidate

            original = direct_index.get(binding)
            if original is None:
                raise ValueError("reasoned collection requires an exact direct read for every item")
            if not candidate(original["raw_probs"], item.state, item.question.instruction):
                continue
        existing = index.get(binding)
        if existing is not None and reasoned:
            nested = existing.get("reasoned_read", {})
            if (
                nested.get("recipe") != recipe
                or nested.get("direct_record_sha256") != original["record_sha256"]
            ):
                raise ValueError("cached reasoned read differs from budget/recipe/direct binding")
        if existing is None:
            planned.append((item, binding, original))
    count = 0
    submitted = 0
    items = iter(planned)
    pending = deque()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("a", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):

        def fill() -> None:
            nonlocal submitted
            while len(pending) < concurrency and (limit is None or submitted < limit):
                planned_item = next(items, None)
                if planned_item is None:
                    break
                item, binding, original = planned_item
                if reasoned:
                    from .reasoning import reasoned_read

                    future = pool.submit(
                        reasoned_read,
                        reader,
                        item.state,
                        item.question,
                        state_format=state_format,
                        prompt_variant=prompt_variant,
                        max_tokens=trace_max_tokens,
                    )
                else:
                    future = pool.submit(
                        read_question,
                        reader,
                        item.state,
                        item.question,
                        group_size=group_size,
                        state_format=state_format,
                        prompt_variant=prompt_variant,
                    )
                pending.append((item, binding, original, future))
                submitted += 1

        fill()
        while pending:
            item, binding, original, future = pending.popleft()
            try:
                result = future.result()
            except BaseException:
                for _, _, _, queued in pending:
                    queued.cancel()
                raise
            if reasoned:
                result, metadata = result
            row = {
                "id": item.id,
                "model": model,
                "revision": revision,
                "tokenizer_revision": binding["tokenizer_revision"],
                "binding_sha256": binding["binding_sha256"],
                "diagnostic": diagnostic,
                "rendered_input_sha256": binding["rendered_input_sha256"],
                "prompt_variant": prompt_variant,
                "source": item.source,
                "tier": item.tier,
                "type": item.question.type,
                "labels": item.question.labels,
                "gold": item.gold,
                "public": item.public,
                "split": item.split,
                "case_id": item.case_id or item.cluster_id or item.id,
                "cluster_id": item.cluster_id or item.case_id or item.id,
                "lineage_ids": list(item.lineage_ids),
                "adapter": item.adapter,
                "binding": binding,
                "readout": result.readout,
                "passes": result.passes,
                "pass_bindings": [
                    {
                        **pass_input,
                        "binding_sha256": fingerprint(
                            {"parent_binding_sha256": binding["binding_sha256"], **pass_input}
                        ),
                    }
                    for pass_input in result.pass_inputs
                ],
                "raw_probs": result.raw_probs,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_s": result.latency_s,
            }
            if item.gold_distribution is not None:
                row["gold_distribution"] = item.gold_distribution
            if item.family is not None:
                row["family"] = item.family
            if item.case_id is not None:
                row["case_id"] = item.case_id
            if result.candidate_log_masses is not None:
                row["candidate_log_masses"] = result.candidate_log_masses
            row["record_sha256"] = fingerprint(row)
            if reasoned:
                nested = {
                    **asdict(result),
                    **metadata,
                    "direct_binding_sha256": original["binding_sha256"],
                    "direct_record_sha256": original["record_sha256"],
                }
                row = {k: v for k, v in original.items() if k != "record_sha256"}
                row["reasoned_read"] = nested
                row["record_sha256"] = fingerprint(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            count += 1
            fill()
    return count


def load_reads(paths: list[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="+", help="JSONL files or directories")
    parser.add_argument("--output", "--out", default="reads.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--reasoned", action="store_true")
    parser.add_argument(
        "--direct-reads", nargs="+", help="exact direct reads for candidate selection/binding"
    )
    parser.add_argument("--trace-max-tokens", type=int, default=384)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="mark reads ineligible for strict fitting/selection",
    )
    add_reader_arguments(parser)
    args = parser.parse_args(argv)
    count = collect(
        iter_dataset(args.dataset),
        reader_from_args(args),
        args.output,
        limit=args.limit,
        group_size=args.group_size,
        state_format=args.state_format,
        concurrency=args.concurrency,
        model=args.hf_model if args.backend == "hf" else args.model,
        revision=args.revision,
        prompt_variant=args.prompt_variant,
        diagnostic=args.diagnostic,
        reasoned=args.reasoned,
        direct_reads=load_reads(args.direct_reads) if args.direct_reads else None,
        trace_max_tokens=args.trace_max_tokens,
    )
    print(f"Wrote {count} new reads to {args.output}")


if __name__ == "__main__":
    main()
