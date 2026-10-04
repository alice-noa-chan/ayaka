"""Pinned local human-labelled rehearsal, with source-grouped reserved splits."""

import gzip
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .contract_nli import SOURCE as CONTRACT_SOURCE
from .contract_nli import SOURCE_POLICY as CONTRACT_POLICY
from .decontam import Decontaminator
from .schema import Candidate, Question, Sample, one_hot
from .source_groups import connected_groups, evidence
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
POLICY_SOURCES = {**SOURCES, CONTRACT_SOURCE: CONTRACT_POLICY}
SPLITS = ("train", "router_train", "dev", "calibration", "test")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def valid_provenance(metadata):
    source = POLICY_SOURCES.get(metadata.get("source"))
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


def partition_sources(
    sources,
    reserved,
    *,
    fits,
    train_limit=512,
    heldout_limit=64,
    quotas=None,
    fits_by_split=None,
    reserved_evidence=None,
):
    """Remove public/previous evaluation evidence before assigning any training row."""
    if any(type(n) is not int or n < 1 for n in (train_limit, heldout_limit)):
        raise ValueError("natural source limits must be positive")
    expected = {(source, split) for source in sources for split in SPLITS}
    if quotas is not None and (
        set(quotas) != set(sources)
        or any(set(limits) != set(SPLITS) for limits in quotas.values())
        or any(type(n) is not int or n < 1 for limits in quotas.values() for n in limits.values())
    ):
        raise ValueError("natural quotas require positive sample limits for every source/split")
    blocker = Decontaminator.from_jevbench()
    rows = [(source, sample) for source, samples in sources.items() for sample in samples]
    for source, sample in rows:
        if sample.metadata.get("source") != source or not valid_provenance(sample.metadata):
            raise ValueError(
                "natural source has unapproved revision, split or human-label provenance"
            )
        if not sample.metadata.get("source_lineage"):
            raise ValueError("natural source requires a primary source_lineage")
    all_samples = [sample for _, sample in rows] + list(reserved)
    row_groups, groups = connected_groups(all_samples)
    blocked = set(row_groups[len(rows) :])
    blocked.update(
        row_groups[i] for i, (_, sample) in enumerate(rows) if blocker.sample_hit(sample)
    )
    if reserved_evidence is not None:
        from .reserved_evidence import ReservedEvidenceBlocker

        if not isinstance(reserved_evidence, ReservedEvidenceBlocker):
            raise ValueError("reserved evidence requires its declared state-only overlap policy")
        blocked.update(
            row_groups[i]
            for i, (_, sample) in enumerate(rows)
            if reserved_evidence.state_hit(sample)
        )
    assignment = {id(sample): groups[row_groups[i]] for i, (_, sample) in enumerate(rows)}
    excluded = {id(sample) for i, (_, sample) in enumerate(rows) if row_groups[i] in blocked}
    result = {split: [] for split in SPLITS}
    counts, removed, seen = Counter(), Counter(), set()
    removed_by_source = {source: Counter() for source in sources}

    def remove(source, reason):
        removed[reason] += 1
        removed_by_source[source][reason] += 1

    for source, samples in sources.items():
        for sample in sorted(samples, key=lambda s: digest(s.to_json())):
            if id(sample) in excluded:
                remove(source, "reserved_or_public_overlap")
                continue
            # Prompt groups determine splits; only complete identical annotations
            # are duplicates. Different responses and fractional ratings survive.
            questions = sample.to_json()["questions"]
            for question in questions:
                question.pop("id")
                question["candidates"].sort(key=lambda candidate: candidate["id"])
            identity = (
                source,
                digest({"state": sample.state, "questions": sorted(questions, key=digest)}),
            )
            if identity in seen:
                remove(source, "duplicate_evidence")
                continue
            seen.add(identity)
            group = assignment[id(sample)]
            bucket = int(digest(group["key"])[:8], 16) % 100
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
            if quotas is not None:
                limit = quotas[source][split]
            if counts[source, split] >= limit:
                remove(source, "quota_excluded")
                continue
            if not (fits_by_split(sample, split) if fits_by_split is not None else fits(sample)):
                remove(source, "context_overflow_whole_sample")
                continue
            sample = deepcopy(sample)
            sample.metadata.update(
                split=split,
                split_policy="source_group_closure_v2",
                source_group_id=group["id"],
                lineage_ids=group["identities"],
            )
            result[split].append(sample)
            counts[source, split] += 1
    if set(counts) != expected:
        raise ValueError("each natural source requires nonempty independent splits")
    if quotas is not None and any(
        counts[source, split] != quotas[source][split] for source, split in expected
    ):
        raise ValueError("natural source quota cannot be filled within its independent split")
    return result, {
        "counts": {f"{source}/{split}": n for (source, split), n in counts.items()},
        "removed": dict(removed),
        "removed_by_source": {source: dict(counts) for source, counts in removed_by_source.items()},
        "source_groups": {
            source: {
                "groups": len(group_sizes),
                "largest_group_samples": max(group_sizes.values(), default=0),
            }
            for source in sources
            for group_sizes in [
                Counter(row_groups[i] for i, (name, _) in enumerate(rows) if name == source)
            ]
        },
        "reserved_samples": len(reserved),
        "split_policy": "source groups with transitive aliases and evidence; formats and ontologies shared across splits",
        "dedup_policy": "identical state, question contracts and gold only; prompt siblings retained",
        "quota_unit": "whole samples; every question retained; group members remain in one split",
        "target_provenance": "human labels; no generated rationale added",
    }
