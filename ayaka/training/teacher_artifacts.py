"""Convert anchored Swift train observations into direct-student teachers.

No backend is contacted. Recompute the original and final serving tokens, bind
candidate/gold meaning, and preserve all observed usage (including excluded
empty or capped traces). Neither hashes nor fixtures attest backend execution.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

from ..config import ElectraConfig
from ..data.schema import Sample
from ..eval.read_artifact import fingerprint
from ..swift.binding import SwiftReadIndex, validate_bound_reads
from ..swift.collect import DatasetItem
from ..swift.readers import READOUT, token_input
from ..swift.reasoning import FINAL_INSTRUCTION, TRACE_INSTRUCTION
from .direct_distillation import _sha, make_teacher_read
from .prepare_v2 import canonical, sha256
from .swift_direct import (
    direct_readout_binding,
    encode_direct_sample,
    normalize_input_encoding,
    swift_question,
)
from .tokenizer_identity import scoped_tokenizer_preparation

VERSION = "ayaka-swift-teacher-observations-1"


def teacher_dataset_items(samples):
    """Canonical train IDs and integer Score wire labels, without reading holdouts."""
    result = []
    for sample in samples:
        metadata = sample.metadata
        if metadata.get("split") != "train" or metadata.get("public", False) is not False:
            raise ValueError("teacher observations require non-public train samples only")
        source, lineage = metadata.get("source_example_id"), metadata.get("source_lineage")
        if not isinstance(source, str) or not source or not isinstance(lineage, str) or not lineage:
            raise ValueError("teacher samples require explicit source IDs and lineage")
        extra_lineages = metadata.get("lineage_ids", [])
        if not isinstance(extra_lineages, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in extra_lineages
        ):
            raise ValueError("teacher lineage IDs must be a sequence of nonempty strings")
        for original in sample.questions:
            q, wire, semantics = swift_question(original)
            target = {
                label: float(q.target_distribution.get(semantics[label], 0.0))
                for label in wire.labels
            }
            result.append(
                DatasetItem(
                    source + "/" + q.id,
                    metadata.get("source", "direct-teacher-train"),
                    metadata.get("tier", "standard"),
                    copy.deepcopy(sample.state),
                    wire,
                    max(target, key=target.__getitem__),
                    False,
                    target,
                    metadata.get("task_family"),
                    lineage,
                    "train",
                    lineage,
                    tuple(sorted({lineage, *extra_lineages})),
                    "canonical",
                )
            )
    if not result or len({item.id for item in result}) != len(result):
        raise ValueError("teacher train cohort must be nonempty with unique question IDs")
    return result


def _identity(value, cfg, mechanics_only):
    required = {
        "model",
        "revision",
        "tokenizer_revision",
        "native_weights_sha256",
        "adapter_sha256",
    }
    if not isinstance(value, dict) or set(value) != required or value["model"] != cfg.backbone:
        raise ValueError(
            "teacher identity must bind the exact model, revision and native/adapter bytes"
        )
    if cfg.backbone == "tiny" and not mechanics_only:
        raise ValueError("random tiny observations require explicit mechanics-only mode")
    for key in ("revision", "tokenizer_revision"):
        if (
            not isinstance(value[key], str)
            or len(value[key]) != 40
            or any(c not in "0123456789abcdef" for c in value[key])
        ):
            raise ValueError("teacher identity requires immutable model/tokenizer revisions")
    if cfg.backbone != "tiny" and (
        value["revision"] != cfg.backbone_revision
        or value["tokenizer_revision"] != cfg.backbone_revision
    ):
        raise ValueError("teacher must match the student's pinned native model and tokenizer")
    _sha(value["native_weights_sha256"], "native_weights_sha256")
    if value["adapter_sha256"] is not None:
        _sha(value["adapter_sha256"], "adapter_sha256")


def _anchor(rows, expected, name):
    _sha(expected, name)
    if not isinstance(rows, list) or fingerprint(rows) != expected:
        raise ValueError(f"{name} differs from the externally anchored ordered read contents")


def _scope(rows, *, mechanics_only):
    if mechanics_only:
        SwiftReadIndex(rows)
    else:
        validate_bound_reads(rows)
    if any(
        row.get("split") != "train"
        or row.get("public") is not False
        or row.get("readout") != READOUT
        for row in rows
    ):
        raise ValueError("teacher artifacts must contain only canonical non-public train reads")


def _wire(native, messages, letters, kwargs, observed, context):
    if (
        not isinstance(observed, dict)
        or not isinstance(observed.get("canonical_token_ids"), dict)
        or list(observed["canonical_token_ids"]) != letters
    ):
        raise ValueError("teacher canonical tokens must follow the actual displayed letter order")
    actual = token_input(native, messages, letters, {**kwargs, "return_dict": False})
    if actual != observed or len(actual["input_token_ids"]) > context:
        raise ValueError(
            "observed teacher tokens differ from actual native serving input or context"
        )
    return actual


def _reasoned_observation(original, paired, native, kwargs, context):
    nested = paired.get("reasoned_read")
    if not isinstance(nested, dict) or {
        k: v for k, v in paired.items() if k not in {"record_sha256", "reasoned_read"}
    } != {k: v for k, v in original.items() if k != "record_sha256"}:
        raise ValueError("paired teacher must retain its exact original direct read")
    if (
        nested.get("direct_record_sha256") != original["record_sha256"]
        or nested.get("direct_binding_sha256") != original["binding_sha256"]
    ):
        raise ValueError("paired teacher direct observation binding mismatch")
    recipe = nested.get("recipe", {})
    budget = recipe.get("max_tokens")
    implementation = recipe.get("implementation_sha256")
    _sha(implementation, "reasoning implementation")
    if (
        set(recipe)
        != {
            "contract",
            "instruction",
            "final_instruction",
            "max_tokens",
            "temperature",
            "enable_thinking",
            "stop",
            "readout",
            "implementation_sha256",
        }
        or recipe.get("contract") != "swift_reasoned_read_v1"
        or recipe.get("instruction") != TRACE_INSTRUCTION
        or recipe.get("final_instruction") != FINAL_INSTRUCTION
        or type(budget) is not int
        or not 1 <= budget <= 1024
        or recipe.get("temperature") != 0
        or recipe.get("enable_thinking") is not False
        or recipe.get("stop") != "eos"
        or recipe.get("readout") != READOUT
        or nested.get("recipe_sha256") != fingerprint(recipe)
    ):
        raise ValueError("teacher reasoning recipe must be complete and bounded to 1..1024")
    generation_messages = [
        {
            "role": "user",
            "content": original["binding"]["messages"][-1]["content"] + "\n\n" + TRACE_INSTRUCTION,
        }
    ]
    if nested.get("generation_messages") != generation_messages:
        raise ValueError("teacher generation must use the original question and options")
    letters = [chr(65 + i) for i in range(len(original["labels"]))]
    generation = _wire(
        native, generation_messages, letters, kwargs, nested.get("generation_input"), context
    )
    if len(generation["input_token_ids"]) + budget > context:
        raise ValueError("teacher declared generation budget cannot fit the native context")
    passes = nested.get("pass_inputs")
    if not isinstance(passes, list) or len(passes) != 1:
        raise ValueError("teacher requires one complete final canonical gathered pass")
    final = passes[0].get("messages")
    if (
        not isinstance(final, list)
        or len(final) != 3
        or final[0] != generation_messages[0]
        or not isinstance(final[1], dict)
        or final[1].get("role") != "assistant"
        or not isinstance(final[1].get("content"), str)
        or final[2] != {"role": "user", "content": FINAL_INSTRUCTION}
    ):
        raise ValueError("teacher final read must contain its exact three-turn trace context")
    final_input = {key: passes[0].get(key) for key in ("input_token_ids", "canonical_token_ids")}
    final_input.update(
        {key + "_sha256": fingerprint(value) for key, value in list(final_input.items())}
    )
    _wire(native, final, letters, kwargs, final_input, context)
    # Gather validation checks every raw canonical logit and the probabilities
    # recomputed from them. Only the prompt changes to the actual final prefix.
    binding = copy.deepcopy(original["binding"])
    binding.update(
        messages=final,
        rendered_input_sha256=fingerprint(final),
        token_inputs=[final_input],
        canonical_token_ids_sha256=fingerprint([final_input["canonical_token_ids"]]),
    )
    binding.pop("binding_sha256")
    binding["binding_sha256"] = fingerprint(binding)
    synthetic = {
        **original,
        **{
            key: nested.get(key)
            for key in (
                "raw_probs",
                "candidate_log_masses",
                "input_tokens",
                "output_tokens",
                "latency_s",
                "readout",
                "passes",
            )
        },
        "binding": binding,
        "binding_sha256": binding["binding_sha256"],
        "pass_bindings": passes,
        "rendered_input_sha256": fingerprint(final),
    }
    synthetic.pop("record_sha256")
    synthetic["record_sha256"] = fingerprint(synthetic)
    SwiftReadIndex([synthetic])
    tokens = nested.get("trace_tokens")
    integer_fields = (
        "trace_input_tokens",
        "read_input_tokens",
        "read_output_tokens",
        "input_tokens",
        "output_tokens",
    )
    if (
        type(tokens) is not int
        or not 0 <= tokens <= budget
        or nested.get("finish_reason") not in {"eos", "length"}
        or nested.get("length_capped") is not (nested["finish_reason"] == "length")
        or any(type(nested.get(key)) is not int or nested[key] < 1 for key in integer_fields)
        or nested["trace_input_tokens"] != len(generation["input_token_ids"])
        or nested["read_input_tokens"] != len(final_input["input_token_ids"])
        or nested["read_output_tokens"] != 1
        or nested["input_tokens"] != nested["trace_input_tokens"] + nested["read_input_tokens"]
        or nested["output_tokens"] != tokens + 1
        or nested.get("readout") != READOUT
        or type(nested.get("passes")) is not int
        or nested.get("passes") != 2
    ):
        raise ValueError("teacher usage, route or termination provenance mismatch")
    if any(
        type(nested.get(key)) not in (int, float)
        or not math.isfinite(nested[key])
        or nested[key] <= 0
        for key in ("trace_latency_s", "read_latency_s", "latency_s")
    ) or not math.isclose(
        nested["latency_s"],
        nested["trace_latency_s"] + nested["read_latency_s"],
        abs_tol=1e-9,
        rel_tol=1e-9,
    ):
        raise ValueError("teacher latency provenance mismatch")
    exclusion = (
        "empty_trace"
        if not final[1]["content"].strip() or not tokens
        else "incomplete_trace"
        if nested["finish_reason"] != "eos"
        else None
    )
    return nested, final[1]["content"], exclusion


@scoped_tokenizer_preparation
def export_swift_teachers(
    samples,
    tok,
    cfg,
    direct_reads,
    paired_reads,
    *,
    input_encoding,
    teacher_identity,
    expected_direct_sha256,
    expected_paired_sha256,
    verify_gold,
    mechanics_only=False,
):
    """Export verified-content observations; gold-benefit filtering remains downstream.

    Sparse paired reads are allowed for any train questions, including forced
    high-confidence/no-number reads. No auto-router eligibility is imposed.
    """
    if type(mechanics_only) is not bool or not callable(verify_gold):
        raise ValueError("require explicit mechanics scope and an independent gold verifier")
    _identity(teacher_identity, cfg, mechanics_only)
    _anchor(direct_reads, expected_direct_sha256, "direct artifact")
    _anchor(paired_reads, expected_paired_sha256, "paired artifact")
    if not paired_reads:
        raise ValueError("teacher export requires at least one actual paired observation")
    _scope(direct_reads, mechanics_only=mechanics_only)
    _scope(paired_reads, mechanics_only=mechanics_only)
    cohort = teacher_dataset_items(samples)
    expected = {item.id: item for item in cohort}
    if {row["id"] for row in direct_reads} != set(expected) or {
        row["id"] for row in paired_reads
    } - set(expected):
        raise ValueError(
            "teacher artifacts must match the complete direct train cohort and paired subset"
        )
    encoding = normalize_input_encoding(input_encoding)
    if encoding["encoder"] != "swift_canonical":
        raise ValueError("Swift teacher conversion requires the explicit Swift direct encoder")
    native = getattr(tok, "hf", tok)
    lookup = {row["id"]: row for row in direct_reads}
    pairs = {row["id"]: row for row in paired_reads}
    teachers, observations, budgets = {}, [], set()
    total_input = total_output = reasoning_tokens = 0
    for sample in samples:
        encoded = encode_direct_sample(sample, tok, cfg, input_encoding=encoding)
        for original, item in zip(sample.questions, encoded, strict=True):
            q, wire, semantics = swift_question(original)
            identity = sample.metadata["source_example_id"] + "/" + q.id
            row, declared = lookup[identity], expected[identity]
            bound = row["binding"]
            if (
                any(
                    bound.get(key) != teacher_identity[key]
                    for key in ("model", "revision", "tokenizer_revision")
                )
                or bound["runtime"].get("adapter_sha256") != teacher_identity["adapter_sha256"]
                or bound["runtime"].get("state_format") != encoding["state_format"]
                or bound["runtime"].get("chat_template_kwargs") != encoding["chat_template_kwargs"]
                or row["prompt_variant"] != encoding["prompt_variant"]
            ):
                raise ValueError("teacher model/tokenizer/adapter or serving input recipe mismatch")
            if (
                fingerprint(bound.get("state")) != fingerprint(sample.state)
                or fingerprint(bound.get("question")) != fingerprint(asdict(wire))
                or row["labels"] != wire.labels
                or row.get("case_id") != declared.case_id
                or row.get("cluster_id") != declared.cluster_id
                or row.get("lineage_ids") != list(declared.lineage_ids)
                or row.get("source") != declared.source
                or row.get("adapter") != "canonical"
                or row.get("gold") != declared.gold
                or row.get("gold_distribution") != declared.gold_distribution
            ):
                raise ValueError(
                    "teacher original evidence, candidates, soft gold or lineage mismatch"
                )
            independently_verified = verify_gold(copy.deepcopy(sample), copy.deepcopy(q))
            if not isinstance(independently_verified, dict) or independently_verified != {
                c.id: q.target_distribution.get(c.id, 0.0) for c in q.candidates
            }:
                raise ValueError("teacher corpus gold disagrees with the independent verifier")
            actual = direct_readout_binding(item)
            direct_input = bound["token_inputs"][0]
            letters = [chr(65 + i) for i in range(len(wire.labels))]
            if list(direct_input["canonical_token_ids"]) != letters:
                raise ValueError(
                    "direct teacher canonical tokens must retain displayed letter order"
                )
            if (
                fingerprint(bound["messages"]) != actual["messages_sha256"]
                or direct_input["input_token_ids_sha256"] != actual["input_token_ids_sha256"]
                or direct_input["canonical_token_ids_sha256"]
                != actual["canonical_token_ids_sha256"]
            ):
                raise ValueError(
                    "teacher direct observations differ from actual student native input"
                )
            if (
                type(row.get("input_tokens")) is not int
                or row["input_tokens"] != item.length
                or type(row.get("output_tokens")) is not int
                or row.get("output_tokens") != 1
                or type(row.get("passes")) is not int
                or row["passes"] != 1
                or type(row.get("latency_s")) not in (int, float)
                or not math.isfinite(row["latency_s"])
                or row["latency_s"] <= 0
            ):
                raise ValueError("direct teacher observation usage mismatch")
            total_input += row["input_tokens"]
            total_output += row["output_tokens"]
            pair = pairs.get(identity)
            if pair is None:
                observations.append(
                    {"id": identity, "status": "no_saved_paired_read", "trace_tokens": 0}
                )
                continue
            nested, trace, exclusion = _reasoned_observation(
                row,
                pair,
                native,
                encoding["chat_template_kwargs"],
                max(cfg.max_seq_len, cfg.serve_max_seq_len),
            )
            budgets.add(nested["recipe"]["max_tokens"])
            total_input += nested["input_tokens"]
            total_output += nested["output_tokens"]
            reasoning_tokens += nested["trace_tokens"]
            observations.append(
                {
                    "id": identity,
                    "status": exclusion or "completed_reasoned_observation",
                    "trace_tokens": nested["trace_tokens"],
                    "finish_reason": nested["finish_reason"],
                    "direct_record_sha256": row["record_sha256"],
                    "paired_record_sha256": pair["record_sha256"],
                }
            )
            if exclusion:
                continue
            by_original_id = {candidate_id: label for label, candidate_id in semantics.items()}
            teachers[identity] = make_teacher_read(
                sample,
                original,
                probs=[nested["raw_probs"][by_original_id[c.id]] for c in original.candidates],
                direct_probs=[row["raw_probs"][by_original_id[c.id]] for c in original.candidates],
                model_sha256=fingerprint(teacher_identity),
                recipe_sha256=fingerprint(
                    {
                        "identity": teacher_identity,
                        "runtime": bound["runtime"],
                        "reasoning": nested["recipe"],
                    }
                ),
                trace_sha256=fingerprint(trace),
                generated_tokens=nested["trace_tokens"],
                direct_readout=actual,
            )
            teachers[identity]["observation_provenance"] = {
                "contract": VERSION,
                "direct_artifact_sha256": expected_direct_sha256,
                "paired_artifact_sha256": expected_paired_sha256,
                "direct_record_sha256": row["record_sha256"],
                "paired_record_sha256": pair["record_sha256"],
                "teacher_identity": copy.deepcopy(teacher_identity),
                "input_tokens": row["input_tokens"] + nested["input_tokens"],
                "output_tokens": 1 + nested["output_tokens"],
                "reasoning_tokens": nested["trace_tokens"],
                "backend_execution_attested": False,
            }
    if len(budgets) != 1:
        raise ValueError("teacher cohort requires one predeclared effort budget")
    report = {
        "version": VERSION,
        "teacher_identity": copy.deepcopy(teacher_identity),
        "input_encoding": encoding,
        "direct_artifact_sha256": expected_direct_sha256,
        "paired_artifact_sha256": expected_paired_sha256,
        "teacher_reads_sha256": fingerprint(teachers),
        "observations": observations,
        "available_teacher_questions": len(teachers),
        "direct_questions": len(direct_reads),
        "paired_questions": len(paired_reads),
        "max_reasoning_tokens": next(iter(budgets)),
        "usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "reasoning_tokens": reasoning_tokens,
            "backend_calls": len(direct_reads) + 2 * len(paired_reads),
        },
        "usage_scope": "saved observations only; missing failed upstream attempts require the collection runner usage log",
        "mechanics_only": mechanics_only,
        "promotable": False,
        "execution_attested": False,
        "scope": "anchored observed-content conversion; gold benefit is filtered during preparation; no backend/weight attestation",
    }
    return teachers, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train",
        "config",
        "input-encoding",
        "teacher-identity",
        "direct-reads",
        "paired-reads",
        "out",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument(
        "--expected-direct-sha256",
        required=True,
        help="fingerprint of the ordered JSON read contents",
    )
    parser.add_argument(
        "--expected-paired-sha256",
        required=True,
        help="fingerprint of the ordered JSON read contents",
    )
    parser.add_argument("--mechanics-only", action="store_true")
    parser.add_argument("--contractnli-train", type=Path)
    parser.add_argument(
        "--native-path", type=Path, help="complete offline native model/tokenizer package"
    )
    parser.add_argument(
        "--mechanics-tokenizer",
        type=Path,
        help="saved local fast tokenizer, required only for random tiny mechanics; never a model override",
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("teacher export requires a fresh directory")
    raw = args.train.read_bytes()
    _sha(args.expected_train_sha256, "train corpus")
    if sha256(raw) != args.expected_train_sha256:
        raise ValueError("train corpus differs from its external byte anchor")
    samples = [Sample.from_json(json.loads(line)) for line in raw.splitlines() if line.strip()]
    teacher_dataset_items(samples)
    from .direct_bundle import _gold_verifier, local_tokenizer

    cfg = ElectraConfig(**json.loads(args.config.read_bytes()))
    from .native_metadata import inspect_metadata, verify_metadata

    if args.native_path is not None and args.mechanics_tokenizer is not None:
        raise ValueError("native-path and mechanics-tokenizer cannot be combined")
    metadata, native_root = inspect_metadata(
        cfg.backbone, cfg.backbone_revision, path=args.native_path
    )
    if args.mechanics_tokenizer is not None:
        if not args.mechanics_only or cfg.backbone != "tiny":
            raise ValueError(
                "a mechanics tokenizer is restricted to explicit random tiny mechanics"
            )
        if not args.mechanics_tokenizer.is_dir():
            raise ValueError("mechanics tokenizer must be a saved local directory")
        from transformers import AutoTokenizer

        from ..tokenization import HFTokenizer

        tok = HFTokenizer(
            AutoTokenizer.from_pretrained(
                args.mechanics_tokenizer, local_files_only=True, trust_remote_code=False
            ),
            "tiny",
        )
    else:
        if cfg.backbone == "tiny" and args.native_path is None:
            raise ValueError(
                "Swift tiny mechanics requires --mechanics-only and --mechanics-tokenizer"
            )
        tok = local_tokenizer(
            cfg, allow_tiny=args.mechanics_only, native_path=native_root, expected_metadata=metadata
        )
    from ..data.direct_natural import explicit_policy_registry

    verify, sources = _gold_verifier(
        {"train": samples},
        explicit_policy_registry(args.contractnli_train),
        allow_tiny=args.mechanics_only,
    )

    def reads(path):
        return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]

    teachers, report = export_swift_teachers(
        samples,
        tok,
        cfg,
        reads(args.direct_reads),
        reads(args.paired_reads),
        input_encoding=json.loads(args.input_encoding.read_bytes()),
        teacher_identity=json.loads(args.teacher_identity.read_bytes()),
        expected_direct_sha256=args.expected_direct_sha256,
        expected_paired_sha256=args.expected_paired_sha256,
        verify_gold=verify,
        mechanics_only=args.mechanics_only,
    )
    report.update(
        train_corpus_sha256=args.expected_train_sha256,
        gold_sources=sources,
        paid_execution_started=False,
        model_loaded=False,
        optimizer_steps=0,
        native_metadata=metadata,
    )
    payloads = {
        "teacher_reads.json": canonical(teachers) + b"\n",
        "receipt.json": canonical(report) + b"\n",
    }
    verify_metadata(metadata, cfg.backbone, cfg.backbone_revision, path=native_root)
    if args.train.read_bytes() != raw:
        raise ValueError("train corpus changed during teacher export")
    verify.verify_files()
    args.out.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (args.out / name).write_bytes(payload)
    print(
        json.dumps(
            {
                "available_teacher_questions": len(teachers),
                "usage": report["usage"],
                "paid_execution_started": False,
                "execution_attested": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
