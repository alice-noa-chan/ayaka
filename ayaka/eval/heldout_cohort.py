"""Held-out development cohort that neither published v1 nor trained v2 saw.

Published v1 trained on nearly every row of the HelpSteer2, CommonsenseQA and
MASSIVE train files, and trained v2 drew its corpus from the same files, so a
development cohort taken from them favors v1. This builder reads natural
questions only from original validation/test files, drops any item whose
text also appears in the matching train file, and adds repository-authored
verified questions after the indices that earlier cohorts already used.

Natural questions use the exact task wording of training preparation
(``direct_natural.task_view`` and ``contract_nli.document_questions``) or of
the v1 loaders (HotpotQA, StrategyQA). Every sample records whether it is
natural or synthetic, its original split, and a source lineage; Korean and
Japanese translations of one MASSIVE utterance share a lineage, so they count
as one independent case. Nothing is fitted, and no model is loaded.

    python -m ayaka.eval.heldout_cohort --settings docs/experiments/heldout_dev_settings.json \\
        --contractnli-archive contract-nli.zip --checkpoint V2_CHECKPOINT \\
        --tokenizer NATIVE_TOKENIZER_DIR --out heldout-dev
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ..data import contract_nli
from ..data.decontam import Decontaminator
from ..data.direct_natural import PINNED_RAW_SHA256, task_view
from ..data.natural_training_v2 import SOURCES as TRAIN_SOURCES
from ..data.natural_training_v2 import digest
from ..data.reasoning_v2 import curriculum
from ..data.schema import Sample
from ..data.source_groups import evidence
from ..data.transforms import hotpot_decision, strategyqa_noul
from .quality_hierarchy import READ_SPLITS
from .read_artifact import fingerprint

VERSION = "ayaka-heldout-dev-cohort-1"
NATURAL_SOURCES = (
    "helpsteer2",
    "commonsense_qa",
    "massive",
    "hotpotqa",
    "strategyqa",
    "contract_nli",
)


@dataclass(frozen=True)
class PinnedFile:
    repo: str
    revision: str
    filename: str
    sha256: str

    def path(self):
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                self.repo,
                self.filename,
                repo_type="dataset",
                revision=self.revision,
                local_files_only=True,
            )
        )
        if _sha256(path.read_bytes()) != self.sha256:
            raise ValueError(f"held-out source differs from its pinned bytes: {self.filename}")
        return path

    def provenance(self):
        return {
            "repo": self.repo,
            "revision": self.revision,
            "file": self.filename,
            "sha256": self.sha256,
        }


def _train_file(source):
    repo, revision, filename, _ = TRAIN_SOURCES[source]
    return PinnedFile(repo, revision, filename, PINNED_RAW_SHA256[source])


HOTPOT = ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab")
STRATEGY = ("ChilleD/StrategyQA", "705562638fe1d8ca6bb98c66fc8f94d45fda8c83")
MASSIVE_REVISION = TRAIN_SOURCES["massive_ko"][1]
HELDOUT_FILES = {
    "helpsteer2": PinnedFile(
        "nvidia/HelpSteer2",
        TRAIN_SOURCES["helpsteer2"][1],
        "validation.jsonl.gz",
        "610eeb5289494d613c4c0f70aade2df8df0b499f3a24e76d232f74e6909d010a",
    ),
    "commonsense_qa": PinnedFile(
        "tau/commonsense_qa",
        TRAIN_SOURCES["commonsense_qa"][1],
        "data/validation-00000-of-00001.parquet",
        "bdbd9bf9cc4d2349b24901038b2ab2f58e10e4e507ad2fd425dca55cd3cb6660",
    ),
    "massive_ko": PinnedFile(
        "AmazonScience/massive",
        MASSIVE_REVISION,
        "ko-KR/validation/0000.parquet",
        "cce429518bfd53056bb83ef49a21222a2581fbeb0eed874bf9aeb426d5756d07",
    ),
    "massive_ja": PinnedFile(
        "AmazonScience/massive",
        MASSIVE_REVISION,
        "ja-JP/validation/0000.parquet",
        "c98ddb8e6f3c172fdc42c5a0653adc2bf8e2160ae13d4a23fef67b84bdff437e",
    ),
    "hotpotqa": PinnedFile(
        *HOTPOT,
        "distractor/validation-00000-of-00001.parquet",
        "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
    ),
    "strategyqa": PinnedFile(
        *STRATEGY,
        "data/test-00000-of-00001-bae602f3ee37f4ca.parquet",
        "d45b64bac89eea93147731ba4d757459e2d1cb8d7e6e73c59c6501e4e0a0dc61",
    ),
}
# Train files whose texts must not reappear in the held-out cohort.
TRAIN_FILES = {
    "helpsteer2": [_train_file("helpsteer2")],
    "commonsense_qa": [_train_file("commonsense_qa")],
    "massive_ko": [_train_file("massive_ko")],
    "massive_ja": [_train_file("massive_ja")],
    "hotpotqa": [
        PinnedFile(
            *HOTPOT,
            "distractor/train-00000-of-00002.parquet",
            "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
        ),
        PinnedFile(
            *HOTPOT,
            "distractor/train-00001-of-00002.parquet",
            "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f",
        ),
    ],
    "strategyqa": [
        PinnedFile(
            *STRATEGY,
            "data/train-00000-of-00001-506370352f622815.parquet",
            "82262246f9f669cc69128c64d96e81860d232482565ecb9d9c63e7ee2c6574a7",
        )
    ],
}
CONTRACT_DEV_MEMBER = "contract-nli/dev.json"
CONTRACT_DEV_SHA256 = "310af7d661d2ab50ee3700169cef524c75f39fb296bbf5a515c229eb0f42e68e"
CONTRACT_TRAIN_MEMBER = "contract-nli/train.json"
LICENSES = {
    "helpsteer2": "CC-BY-4.0",
    "commonsense_qa": "MIT",
    "massive_ko": "CC-BY-4.0",
    "massive_ja": "CC-BY-4.0",
    "hotpotqa": "CC-BY-SA-4.0",
    "strategyqa": "MIT",
    "contract_nli": "CC-BY-4.0",
}
ORIGINAL_SPLITS = {
    "helpsteer2": "validation",
    "commonsense_qa": "validation",
    "massive_ko": "validation",
    "massive_ja": "validation",
    "hotpotqa": "validation",
    "strategyqa": "test",
    "contract_nli": "dev",
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    """Casefolded NFKC text with collapsed whitespace, for train-overlap checks."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def read_rows(path):
    """Rows and Hugging Face feature metadata of a .jsonl.gz or .parquet file."""
    path = Path(path)
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()], {}
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    return table.to_pylist(), json.loads(metadata.get(b"huggingface", b"{}"))


def _intent_names(features):
    names = features.get("info", {}).get("features", {}).get("intent", {}).get("names")
    if not names:
        raise ValueError("MASSIVE file lacks its intent ontology")
    return names


def validate_settings(settings):
    if not isinstance(settings, dict) or settings.get("version") != VERSION:
        raise ValueError(f"settings must declare version {VERSION}")
    keys = {"version", "split", "seed", "max_input_tokens", "natural", "synthetic"}
    if set(settings) != keys:
        raise ValueError(f"settings need exactly {', '.join(sorted(keys))}")
    if settings["split"] not in READ_SPLITS:
        raise ValueError(f"split must be one of {READ_SPLITS}; test is never built here")
    natural, synthetic = settings["natural"], settings["synthetic"]
    if not isinstance(natural, dict) or set(natural) != set(NATURAL_SOURCES):
        raise ValueError(f"natural quotas must name exactly {', '.join(NATURAL_SOURCES)}")
    counts = [settings["seed"], settings["max_input_tokens"], *natural.values()]
    if not isinstance(synthetic, dict) or set(synthetic) != {"start", "per_type"}:
        raise ValueError("synthetic settings need start and per_type")
    counts += [synthetic["start"], synthetic["per_type"]]
    if any(type(n) is not int or n < 0 for n in counts) or settings["max_input_tokens"] < 1:
        raise ValueError("seed, token limit and quotas must be nonnegative integers")
    return settings


def _metadata(source, original_split, key, lineage, row, file_sha256, language, view):
    return {
        "source": source,
        "language": language,
        "license": LICENSES[source],
        "label_source": "human",
        "data_kind": "natural",
        "original_split": original_split,
        "modality": "text",
        "tier": "standard",
        "heldout_cohort_version": VERSION,
        "raw_row_sha256": fingerprint(row),
        "raw_file_sha256": file_sha256,
        "source_lineage": lineage,
        "source_example_id": f"heldout/{source}/{original_split}/{key}",
        **view,
    }


def natural_units(rows, files, split="dev"):
    """Candidate units per source: lists of samples that share one lineage.

    ``rows`` maps each held-out source (and ``contract_nli``) to its parsed
    rows; ``files`` maps it to the file SHA-256. MASSIVE needs ``features``
    under ``rows["massive_ko"]["features"]`` and the Japanese equivalent.
    Every sample is labeled with ``split``, the cohort's evaluation split.
    """
    units = defaultdict(list)
    by_prompt = {}
    for row in rows["helpsteer2"]["rows"]:
        lineage = "natural/" + digest(
            ["helpsteer2", evidence(Sample({"prompt": row["prompt"]}, []))]
        )
        # One response per prompt keeps every unit an independent case.
        if lineage not in by_prompt or fingerprint(row) < fingerprint(by_prompt[lineage]):
            by_prompt[lineage] = row
    for lineage, row in by_prompt.items():
        state, questions, view = task_view("helpsteer2", row)
        meta = _metadata(
            "helpsteer2",
            "validation",
            fingerprint(row),
            lineage,
            row,
            files["helpsteer2"],
            "en",
            view,
        )
        units["helpsteer2"].append(([Sample(state, questions, meta)], normalized(row["prompt"])))
    for row in rows["commonsense_qa"]["rows"]:
        lineage = "natural/" + digest(["commonsense_qa", evidence(Sample(row["question"], []))])
        state, questions, view = task_view("commonsense_qa", row)
        meta = _metadata(
            "commonsense_qa",
            "validation",
            row["id"],
            lineage,
            row,
            files["commonsense_qa"],
            "en",
            view,
        )
        units["commonsense_qa"].append(
            ([Sample(state, questions, meta)], normalized(row["question"]))
        )
    by_id = defaultdict(dict)
    for source, language in (("massive_ko", "ko"), ("massive_ja", "ja")):
        names = _intent_names(rows[source]["features"])
        for row in rows[source]["rows"]:
            by_id[str(row["id"])][source] = (row, language, names)
    for utterance_id, pair in by_id.items():
        if set(pair) != {"massive_ko", "massive_ja"}:
            continue
        lineage = "natural/" + digest(["massive", utterance_id])
        samples, texts = [], []
        for source in ("massive_ko", "massive_ja"):
            row, language, names = pair[source]
            state, questions, view = task_view(source, row, names)
            meta = _metadata(
                source, "validation", utterance_id, lineage, row, files[source], language, view
            )
            samples.append(Sample(state, questions, meta))
            texts.append((source, normalized(row["utt"])))
        units["massive"].append((samples, texts))
    for row in rows["hotpotqa"]["rows"]:
        converted = hotpot_decision(row)
        if not converted or converted[0].questions[0].type != "noul":
            continue  # yes/no questions only; comparisons and spans are skipped
        sample = converted[0]
        view = {"task_family": "multi_hop_yes_no", "task_view": "hotpot-distractor-yes-no-v1"}
        sample.metadata = _metadata(
            "hotpotqa",
            "validation",
            row["id"],
            f"heldout/hotpotqa/{row['id']}",
            row,
            files["hotpotqa"],
            "en",
            view,
        )
        units["hotpotqa"].append(([sample], normalized(row["question"])))
    for row in rows["strategyqa"]["rows"]:
        sample = strategyqa_noul(row)[0]
        view = {"task_family": "implicit_yes_no", "task_view": "strategyqa-facts-yes-no-v1"}
        sample.metadata = _metadata(
            "strategyqa",
            "test",
            row["qid"],
            f"heldout/strategyqa/{row['qid']}",
            row,
            files["strategyqa"],
            "en",
            view,
        )
        units["strategyqa"].append(([sample], normalized(row["question"])))
    labels = rows["contract_nli"]["labels"]
    for doc in rows["contract_nli"]["documents"]:
        lineage = "contract-nli/document/" + fingerprint(contract_nli._text(doc["text"]))
        view = {
            "task_family": "contract_policy",
            "task_view": "original-full-document-17-hypotheses-three-way-v1",
        }
        meta = _metadata(
            "contract_nli", "dev", doc["id"], lineage, doc, files["contract_nli"], "en", view
        )
        sample = Sample(doc["text"], contract_nli.document_questions(doc, labels), meta)
        units["contract_nli"].append(([sample], normalized(doc["text"])))
    for source_units in units.values():
        for samples, _ in source_units:
            for sample in samples:
                sample.metadata["split"] = split
    return units


def train_texts(rows):
    """Normalized train texts per held-out source, for overlap exclusion."""
    return {
        "helpsteer2": {normalized(r["prompt"]) for r in rows["helpsteer2"]},
        "commonsense_qa": {normalized(r["question"]) for r in rows["commonsense_qa"]},
        "massive_ko": {normalized(r["utt"]) for r in rows["massive_ko"]},
        "massive_ja": {normalized(r["utt"]) for r in rows["massive_ja"]},
        "hotpotqa": {normalized(r["question"]) for r in rows["hotpotqa"]},
        "strategyqa": {normalized(r["question"]) for r in rows["strategyqa"]},
        "contract_nli": {normalized(d["text"]) for d in rows["contract_nli"]},
    }


def _overlaps_train(source, key, train):
    if source == "massive":
        return any(text in train[name] for name, text in key)
    return key in train[source]


def synthetic_samples(start, per_type, split="dev"):
    """Verified authored questions after the first ``start`` indices of each type.

    The generator writes each split in its own document voice and labels the
    samples with that split.
    """
    if per_type == 0:
        return []
    generated = curriculum(split, start + per_type)
    total = start + per_type
    result = []
    for position, (sample, traces) in enumerate(generated):
        if position % total < start:
            continue
        sample.metadata.update(
            source_lineage=sample.metadata["case_facts_sha256"],
            modality="text",
            tier="standard",
            data_kind="synthetic",
            label_source="verified_procedure",
            original_split=split,
            heldout_cohort_version=VERSION,
            verified_traces=traces,
        )
        result.append(sample)
    return result


def build_cohort(settings, units, train, *, fits=None, reserved=(), public=None):
    """Select whole units deterministically and report every exclusion.

    ``fits(sample)`` returns whether a sample's complete input fits the token
    limit for every system that will read it; ``None`` skips that check.
    ``reserved`` samples (earlier cohorts, private tests) exclude any unit that
    shares a lineage or normalized state with them.
    """
    validate_settings(settings)
    if any(
        sample.metadata.get("split") != settings["split"]
        for source_units in units.values()
        for samples, _ in source_units
        for sample in samples
    ):
        raise ValueError("candidate units were built for a different split than the settings")
    blocked_lineages = {s.metadata.get("source_lineage") for s in reserved}
    blocked_states = {evidence(s) for s in reserved}
    public = Decontaminator.from_jevbench() if public is None else public
    selected, excluded = [], defaultdict(Counter)
    for source in NATURAL_SOURCES:
        quota = settings["natural"][source]
        ordered = sorted(
            units.get(source, []),
            key=lambda unit: fingerprint(
                [settings["seed"], source, unit[0][0].metadata["source_lineage"]]
            ),
        )
        taken = 0
        for samples, key in ordered:
            if taken == quota:
                break
            if _overlaps_train(source, key, train):
                excluded[source]["train_text_overlap"] += 1
            elif any(
                s.metadata["source_lineage"] in blocked_lineages or evidence(s) in blocked_states
                for s in samples
            ):
                excluded[source]["reserved_overlap"] += 1
            elif any(public.sample_hit(s) for s in samples):
                excluded[source]["public_benchmark_overlap"] += 1
            elif fits is not None and not all(fits(s) for s in samples):
                excluded[source]["context_limit"] += 1
            else:
                selected.extend(samples)
                taken += 1
        if taken < quota:
            raise ValueError(f"{source}: only {taken} eligible units for a quota of {quota}")
    synthetic = synthetic_samples(
        settings["synthetic"]["start"], settings["synthetic"]["per_type"], settings["split"]
    )
    if any(public.sample_hit(s) for s in synthetic):
        raise ValueError("authored questions overlap public benchmark text; choose new indices")
    if fits is not None and not all(fits(s) for s in synthetic):
        raise ValueError("authored questions exceed the token limit")
    return selected + synthetic, {k: dict(v) for k, v in excluded.items()}


def summary(samples):
    def counts(key):
        groups = defaultdict(
            lambda: {"samples": 0, "questions": 0, "cases": set(), "types": Counter()}
        )
        for s in samples:
            group = groups[s.metadata[key]]
            group["samples"] += 1
            group["questions"] += len(s.questions)
            group["cases"].add(s.metadata["source_lineage"])
            group["types"].update(q.type for q in s.questions)
        return {
            name: {
                "samples": g["samples"],
                "questions": g["questions"],
                "independent_cases": len(g["cases"]),
                "types": dict(sorted(g["types"].items())),
            }
            for name, g in sorted(groups.items())
        }

    return {
        "questions": sum(len(s.questions) for s in samples),
        "independent_cases": len({s.metadata["source_lineage"] for s in samples}),
        "types": dict(sorted(Counter(q.type for s in samples for q in s.questions).items())),
        "by_source": counts("source"),
        "by_data_kind": counts("data_kind"),
        "by_language": counts("language"),
    }


def load_sources(contract_archive):
    """Read every pinned held-out and train file; returns (held-out rows, train texts, file hashes)."""
    archive_path = Path(contract_archive)
    if _sha256(archive_path.read_bytes()) != contract_nli.ARCHIVE_SHA256:
        raise ValueError("ContractNLI archive differs from its pinned bytes")
    with zipfile.ZipFile(archive_path) as archive:
        dev_bytes = archive.read(CONTRACT_DEV_MEMBER)
        train_bytes = archive.read(CONTRACT_TRAIN_MEMBER)
    if (
        _sha256(dev_bytes) != CONTRACT_DEV_SHA256
        or _sha256(train_bytes) != contract_nli.TRAIN_SHA256
    ):
        raise ValueError("ContractNLI members differ from their pinned bytes")
    contract_dev = json.loads(dev_bytes)
    contract_train = json.loads(train_bytes)
    if contract_dev["labels"] != contract_train["labels"]:
        raise ValueError("ContractNLI dev hypotheses differ from the training hypotheses")
    heldout, files = {}, {}
    for source, pinned in HELDOUT_FILES.items():
        data, features = read_rows(pinned.path())
        heldout[source] = {"rows": data, "features": features}
        files[source] = pinned.sha256
    heldout["contract_nli"] = contract_dev
    files["contract_nli"] = CONTRACT_DEV_SHA256
    for source in ("massive_ko", "massive_ja"):
        _, train_features = read_rows(_train_file(source).path())
        if _intent_names(heldout[source]["features"]) != _intent_names(train_features):
            raise ValueError(f"{source}: validation intent ontology differs from training")
    train_rows = {
        source: [row for pinned in pinned_files for row in read_rows(pinned.path())[0]]
        for source, pinned_files in TRAIN_FILES.items()
    }
    train_rows["contract_nli"] = contract_train["documents"]
    provenance = {source: pinned.provenance() for source, pinned in HELDOUT_FILES.items()}
    provenance["contract_nli"] = {
        "archive_sha256": contract_nli.ARCHIVE_SHA256,
        "member": CONTRACT_DEV_MEMBER,
        "sha256": CONTRACT_DEV_SHA256,
    }
    return heldout, train_texts(train_rows), files, provenance


def context_check(checkpoint, tokenizer_path, limit):
    """Whole-input fit for trained v2 (Swift contract) and published v1 (native format)."""
    from transformers import AutoTokenizer

    from ..checkpoint import load_config
    from ..input_contract import read_contract
    from ..input_errors import ContextLimitError
    from ..primitives import QuestionSpec
    from ..tokenization import HFTokenizer
    from ..training.batching import _noul_canonical
    from ..training.swift_direct import encode_direct_sample
    from ..training.tokenizer_identity import TokenizerPin
    from .checkpoint_comparison import _json_ordinal
    from .matched_execution import full_decision_context

    cfg = load_config(str(checkpoint))
    contract = read_contract(str(checkpoint), cfg)
    if contract is None:
        raise ValueError("context check needs a checkpoint with a Swift input contract")
    tok = HFTokenizer(
        AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True), cfg.backbone
    )
    pin = TokenizerPin(tok)

    def fits(sample):
        with pin.scope():
            try:
                encode_direct_sample(
                    sample, tok, cfg, input_encoding=contract["input_encoding"], context_limit=limit
                )
            except ContextLimitError:
                return False
            # The same canonical question view the matched comparison gives v1.
            specs = []
            for original in sample.questions:
                q = _noul_canonical(original)
                ordinals = (
                    [_json_ordinal(c.ordinal) for c in q.candidates] if q.type == "score" else None
                )
                specs.append(
                    QuestionSpec(
                        q.type, q.instruction, [c.description for c in q.candidates], ordinals
                    )
                )
            try:
                full_decision_context(sample.state, specs, tok, limit=limit)
            except ValueError:
                return False
            return True

    return fits, {"checkpoint_config_sha256": fingerprint(asdict(cfg)), "limit": limit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--contractnli-archive", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="Swift-contract checkpoint for the context check"
    )
    parser.add_argument(
        "--tokenizer", type=Path, help="native tokenizer directory for the context check"
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="Sample JSONL whose cases to exclude",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("the cohort output must be a new directory")
    if (args.checkpoint is None) != (args.tokenizer is None):
        raise ValueError("the context check needs both --checkpoint and --tokenizer")
    settings = validate_settings(json.loads(args.settings.read_bytes()))
    heldout, train, files, provenance = load_sources(args.contractnli_archive)
    units = natural_units(heldout, files, settings["split"])
    reserved = [
        Sample.from_json(json.loads(line))
        for path in args.exclude
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]
    fits, context = (None, None)
    if args.checkpoint is not None:
        fits, context = context_check(args.checkpoint, args.tokenizer, settings["max_input_tokens"])
    samples, excluded = build_cohort(settings, units, train, fits=fits, reserved=reserved)
    from .pretraining_v2 import cohort_fingerprint

    lines = b"".join(
        json.dumps(s.to_json(), ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for s in samples
    )
    manifest = {
        "version": VERSION,
        "settings": settings,
        "sources": provenance,
        "excluded_cohorts": [
            {"file": path.name, "sha256": _sha256(path.read_bytes())} for path in args.exclude
        ],
        "context_check": context,
        "exclusions": excluded,
        "summary": summary(samples),
        "cohort_sha256": cohort_fingerprint(samples),
        "dev_jsonl_sha256": _sha256(lines),
        "fitted": False,
        "scope": "development cohort; neither published v1 nor trained v2 trained on these splits",
    }
    args.out.mkdir(parents=True)
    (args.out / "dev.jsonl").write_bytes(lines)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
