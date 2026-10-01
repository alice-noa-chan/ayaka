"""Pinned local human-labelled rehearsal, with source-grouped reserved splits."""

import gzip
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from .decontam import Decontaminator
from .schema import Candidate, Question, Sample, one_hot
from .transforms import helpsteer2_scores, intent_choice

SOURCES = {
    "helpsteer2": (
        "nvidia/HelpSteer2",
        "990b2711a36180dd19d9c94b8627844866f8982a",
        "train.jsonl.gz",
        "en",
    ),
    "commonsense_qa": (
        "tau/commonsense_qa",
        "94630fe30dad47192a8546eb75f094926d47e155",
        "data/train-00000-of-00001.parquet",
        "en",
    ),
    "massive_ko": (
        "AmazonScience/massive",
        "ed58ac423a2f4121720918bf5301577edce4ffd3",
        "ko-KR/train/0000.parquet",
        "ko",
    ),
    "massive_ja": (
        "AmazonScience/massive",
        "ed58ac423a2f4121720918bf5301577edce4ffd3",
        "ja-JP/train/0000.parquet",
        "ja",
    ),
}
SPLITS = ("train", "router_train", "dev", "calibration", "test")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def evidence(sample):
    state = sample.state
    if isinstance(state, dict) and "prompt" in state:
        state = state["prompt"]  # all responses to the same HelpSteer prompt stay together
    return unicodedata.normalize(
        "NFKC", json.dumps(state, ensure_ascii=False, sort_keys=True)
    ).casefold()


def valid_provenance(metadata):
    source = SOURCES.get(metadata.get("source"))
    return bool(
        source
        and metadata.get("revision") == source[1]
        and metadata.get("license") == "CC-BY-4.0"
        and metadata.get("label_source") == "human"
        and metadata.get("original_split") == "train"
    )


def local_sources():
    """Read pinned cached files only. Never resolve mutable main or download data."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    sources, provenance = {}, []
    for name, (repo, revision, filename, language) in SOURCES.items():
        path = Path(
            hf_hub_download(
                repo, filename, repo_type="dataset", revision=revision, local_files_only=True
            )
        )
        if filename.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            features = {}
        else:
            table = pq.read_table(path)
            rows = table.to_pylist()
            features = json.loads(table.schema.metadata.get(b"huggingface", b"{}"))
        samples = []
        for row in rows:
            metadata = {
                "source": name,
                "revision": revision,
                "language": language,
                "license": "CC-BY-4.0",
                "label_source": "human",
                "original_split": "train",
                "data_kind": "natural",
                "modality": "text",
            }
            if name == "helpsteer2":
                converted = helpsteer2_scores(row, metadata={**metadata, "task_family": "judge"})
            elif name == "commonsense_qa":
                choices = row["choices"]
                candidates = [
                    Candidate(label, text)
                    for label, text in zip(choices["label"], choices["text"], strict=True)
                ]
                converted = [
                    Sample(
                        row["question"],
                        [
                            Question(
                                "q",
                                "choice",
                                "Choose the most appropriate answer.",
                                candidates,
                                one_hot(candidates, row["answerKey"]),
                            )
                        ],
                        {**metadata, "task_family": "choice"},
                    )
                ]
            else:
                if row["partition"] != "train":
                    raise ValueError("natural rehearsal must read the original train partition")
                names = features["info"]["features"]["intent"]["names"]
                instruction = {
                    "ko": "이 요청에 가장 적합한 의도를 선택하세요.",
                    "ja": "この依頼に最も適した意図を選んでください。",
                }[language]
                converted = intent_choice(
                    row,
                    text_key="utt",
                    label_key="intent",
                    ontology={str(i): label.replace("_", " ") for i, label in enumerate(names)},
                    instruction=instruction,
                    metadata={**metadata, "task_family": "choice"},
                )
            for sample in converted:
                # MASSIVE translations/localizations share the original SLURP ID.
                parent = (
                    ("massive", str(row["id"]))
                    if name.startswith("massive_")
                    else (name, evidence(sample))
                )
                lineage = digest(parent)
                sample.metadata.update(
                    source_lineage=f"natural/{lineage}",
                    source_example_id=f"natural/{name}/{digest(sample.state)}",
                )
                samples.append(sample)
        sources[name] = samples
        provenance.append(
            {
                "source": name,
                "repo": repo,
                "revision": revision,
                "file": filename,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "license": "CC-BY-4.0",
                "original_split": "train",
                "available_samples": len(samples),
                "card_url": f"https://huggingface.co/datasets/{repo}",
            }
        )
    return sources, provenance


def partition_sources(sources, reserved, *, fits, train_limit=512, heldout_limit=64):
    """Remove public/previous evaluation evidence before assigning any training row."""
    if any(type(n) is not int or n < 1 for n in (train_limit, heldout_limit)):
        raise ValueError("natural source limits must be positive")
    blocker = Decontaminator.from_jevbench()
    blocked = {evidence(sample) for sample in reserved}
    blocked_lineages = {sample.metadata.get("source_lineage") for sample in reserved}
    result = {split: [] for split in SPLITS}
    counts, removed, seen = Counter(), Counter(), set()
    for source, samples in sources.items():
        for sample in sorted(samples, key=lambda s: digest(s.to_json())):
            if not valid_provenance(sample.metadata):
                raise ValueError(
                    "natural source has unapproved revision, split or human-label provenance"
                )
            content = evidence(sample)
            if (
                content in blocked
                or sample.metadata["source_lineage"] in blocked_lineages
                or blocker.sample_hit(sample)
            ):
                removed["reserved_or_public_overlap"] += 1
                continue
            # Identical evidence across IDs must not become independent train/test cases.
            identity = (source, content)
            if identity in seen:
                removed["duplicate_evidence"] += 1
                continue
            seen.add(identity)
            bucket = int(digest(sample.metadata["source_lineage"])[:8], 16) % 100
            split = (
                "train"
                if bucket < 72
                else "router_train"
                if bucket < 79
                else "dev"
                if bucket < 86
                else "calibration"
                if bucket < 93
                else "test"
            )
            limit = train_limit if split == "train" else heldout_limit
            if source in {"helpsteer2", "commonsense_qa"} and split == "train":
                limit = max(1, train_limit // 2)
            if counts[source, split] >= limit:
                continue
            if not fits(sample):
                removed["context_overflow_whole_sample"] += 1
                continue
            sample.metadata.update(split=split, split_policy="source_group_hash_v1")
            result[split].append(sample)
            counts[source, split] += 1
    expected = {(source, split) for source in sources for split in SPLITS}
    if set(counts) != expected:
        raise ValueError("each natural source requires nonempty independent splits")
    return result, {
        "counts": {f"{source}/{split}": n for (source, split), n in counts.items()},
        "removed": dict(removed),
        "reserved_samples": len(reserved),
        "split_policy": "source groups; formats and ontologies shared across splits",
        "target_provenance": "human labels; no generated rationale added",
    }
