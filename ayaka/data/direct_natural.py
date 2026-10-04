"""Pinned human-label direct rehearsal with raw-source gold verification.

MASSIVE is explicitly transformed into balanced binary intent propositions,
not a shortened 60-way Choice task. HelpSteer human mean ratings are encoded
on adjacent ordinal levels without truncating fractional labels. These sources
preserve general capabilities; they are not natural policy reasoning evidence.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import asdict
from pathlib import Path

from ..eval.read_artifact import fingerprint
from .contract_nli import SOURCE as CONTRACT_SOURCE
from .contract_nli import ContractGoldRegistry
from .natural_training_v2 import SOURCES, digest, evidence, valid_provenance
from .schema import Candidate, Question, Sample
from .transforms import HELPSTEER_LEVELS

VERSION = "ayaka-raw-human-direct-gold-1"
POLICY_VERSION = "ayaka-raw-human-policy-direct-gold-2"
PINNED_RAW_SHA256 = {
    "helpsteer2": "c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713",
    "commonsense_qa": "b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b",
    "massive_ko": "3b5e11303fb75aa42d70b5be42b5052192c17911363388c8c15749f61a9c6d00",
    "massive_ja": "90fb4b86ba1b6ff76d07deb21c9b4d08a5fba329e09f79c14ae64a5c01fb1c6e",
}


def local_raw_sources():
    """Original immutable training files only; no model API or network requests."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ..training.direct_state import file_digest

    result = {}
    for source, (repo, revision, filename, _) in SOURCES.items():
        path = Path(
            hf_hub_download(
                repo, filename, repo_type="dataset", revision=revision, local_files_only=True
            )
        )
        expected = PINNED_RAW_SHA256[source]
        if file_digest(path) != expected:
            raise ValueError(f"raw human source differs from its pinned byte anchor: {source}")
        if filename.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            features = {}
        else:
            table = pq.read_table(path)
            rows = table.to_pylist()
            features = json.loads(table.schema.metadata.get(b"huggingface", b"{}"))
        result[source] = {
            "rows": rows,
            "features": features,
            "local_path": path,
            "provenance": {
                "repo": repo,
                "revision": revision,
                "file": filename,
                "sha256": expected,
            },
        }
        if file_digest(path) != expected:
            raise ValueError(f"raw human source changed during parsing: {source}")
    for source, data in result.items():
        if file_digest(data["local_path"]) != PINNED_RAW_SHA256[source]:
            raise ValueError(f"raw human source changed during collection: {source}")
    return result


def _ordinal_target(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 4:
        raise ValueError("human ordinal rating must be finite in 0..4")
    low, high = math.floor(value), math.ceil(value)
    return {
        f"s{i}": float(i == low)
        if low == high
        else (high - value if i == low else value - low if i == high else 0.0)
        for i in range(5)
    }


def verify_raw_binding(binding, *, registry=None):
    """Recheck bound cached files without reparsing the complete human corpus."""
    if binding is None:
        return
    if binding.get("version") == POLICY_VERSION:
        if (
            not isinstance(registry, NaturalGoldRegistry)
            or registry.policy_registry is None
            or binding != registry.binding
        ):
            raise ValueError("policy raw binding requires its explicit pinned composite registry")
        registry.verify_files()
        return
    if binding.get("version") != VERSION or type(binding.get("local_files_verified")) is not bool:
        raise ValueError("unsupported raw human file binding")
    if binding["local_files_verified"] is False:
        if binding.get("scope") != "injected CPU fixture only":
            raise ValueError("raw fixture binding must retain its explicit mechanics scope")
        return
    if binding.get("local_files_verified") is not True or set(binding.get("sources", {})) != set(
        SOURCES
    ):
        raise ValueError("raw human source inventory changed")
    from huggingface_hub import hf_hub_download

    from ..training.direct_state import file_digest

    for source, (repo, revision, filename, _) in SOURCES.items():
        expected = {
            "repo": repo,
            "revision": revision,
            "file": filename,
            "sha256": PINNED_RAW_SHA256[source],
        }
        if binding["sources"][source] != expected:
            raise ValueError(f"raw human file binding differs from pinned policy: {source}")
        path = hf_hub_download(
            repo, filename, repo_type="dataset", revision=revision, local_files_only=True
        )
        if file_digest(path) != expected["sha256"]:
            raise ValueError(f"raw human source changed after validation: {source}")


def _intent_description(ontology, index, language):
    description = ontology[index].replace("_", " ")
    return {
        "ko": f"이 요청의 의도는 '{description}'이다.",
        "ja": f"この依頼の意図は「{description}」である。",
    }[language]


def _question_contract(q):
    return {
        "id": q.id,
        "type": q.type,
        "instruction": q.instruction,
        "candidates": sorted((asdict(c) for c in q.candidates), key=lambda c: c["id"]),
    }


class NaturalGoldRegistry:
    """Verify labels from original raw rows, never from converted stored targets.

    Raw fixture injection is restricted to tiny CPU preparation by the bundle.
    Published rubric wording is shared schema; human labels and input evidence
    are checked independently of converter output and teacher observations.
    """

    def __init__(self, raw_sources=None, *, policy_registry=None):
        if policy_registry is not None and not isinstance(policy_registry, ContractGoldRegistry):
            raise ValueError("policy gold requires the pinned ContractNLI registry")
        self.policy_registry = policy_registry
        self._raw_local_files_verified = raw_sources is None
        self.local_files_verified = self._raw_local_files_verified and (
            policy_registry is None or policy_registry.local_files_verified
        )
        self.raw = local_raw_sources() if raw_sources is None else raw_sources
        if not self.raw or set(self.raw) - set(SOURCES):
            raise ValueError("raw human sources must belong to the pinned source policy")
        for source, data in self.raw.items():
            repo, revision, filename, _ = SOURCES[source]
            p = data["provenance"]
            if (p.get("repo"), p.get("revision"), p.get("file")) != (repo, revision, filename) or (
                not isinstance(p.get("sha256"), str)
                or len(p["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in p["sha256"])
            ):
                raise ValueError("raw source provenance must match the immutable file policy")
        self.binding = {
            "version": POLICY_VERSION if policy_registry is not None else VERSION,
            "local_files_verified": self.local_files_verified,
            "sources": {name: data["provenance"] for name, data in self.raw.items()},
            "scope": "cached raw human training labels"
            if self.local_files_verified
            else "injected CPU fixture only",
        }
        if policy_registry is not None:
            self.binding["policy_sources"] = {CONTRACT_SOURCE: policy_registry.binding}
            self.binding["scope"] = (
                "cached raw human rehearsal and public full-contract training annotations"
                if self.local_files_verified
                else "injected CPU fixture only"
            )
        self._memory_sha256 = self._memory_identity()

    @property
    def source_names(self):
        return set(self.raw) | ({CONTRACT_SOURCE} if self.policy_registry is not None else set())

    def _memory_identity(self):
        return fingerprint(
            {
                "local_files_verified": self.local_files_verified,
                "raw_local_files_verified": self._raw_local_files_verified,
                "binding": self.binding,
                "policy": self.policy_registry.binding
                if self.policy_registry is not None
                else None,
                "sources": {
                    source: {key: data[key] for key in ("rows", "features", "provenance")}
                    for source, data in self.raw.items()
                },
            }
        )

    def verify_files(self):
        """Recheck actual anchored raw bytes before committing prepared outputs."""
        if self._memory_identity() != self._memory_sha256:
            raise ValueError("parsed human sources or bindings changed after validation")
        if self.policy_registry is not None:
            self.policy_registry.verify_files()
        if not self._raw_local_files_verified:
            return
        from ..training.direct_state import file_digest

        if set(self.raw) != set(SOURCES):
            raise ValueError("raw human source inventory changed")
        for source, data in self.raw.items():
            expected = PINNED_RAW_SHA256[source]
            if (
                data["provenance"]["sha256"] != expected
                or file_digest(data["local_path"]) != expected
            ):
                raise ValueError(f"raw human source changed after validation: {source}")

    def _row(self, source, index):
        if (
            source not in self.raw
            or type(index) is not int
            or not 0 <= index < len(self.raw[source]["rows"])
        ):
            raise ValueError("natural sample must bind an existing original source row")
        return self.raw[source]["rows"][index]

    def _identity(self, source, index, row):
        language = SOURCES[source][3]
        if source.startswith("massive_"):
            parent = ["massive", str(row["id"])]
        elif source == "helpsteer2":
            parent = [source, evidence(Sample({"prompt": row["prompt"]}, []))]
        else:
            parent = [source, evidence(Sample(row["question"], []))]
        return {
            "source": source,
            "revision": SOURCES[source][1],
            "language": language,
            "license": "CC-BY-4.0",
            "label_source": "human",
            "original_split": "train",
            "data_kind": "natural",
            "modality": "text",
            "direct_natural_version": VERSION,
            "raw_row_index": index,
            "raw_row_sha256": fingerprint(row),
            "raw_file_sha256": self.raw[source]["provenance"]["sha256"],
            "source_lineage": "natural/" + digest(parent),
            "source_example_id": f"natural-direct/{source}/{index}/{fingerprint(row)}",
        }

    def sample(self, source, index):
        row = self._row(source, index)
        metadata = self._identity(source, index, row)
        if source == "helpsteer2":
            state = {"prompt": row["prompt"], "response": row["response"]}
            questions = [
                Question(
                    attr,
                    "score",
                    instruction,
                    [
                        Candidate(f"s{i}", f"{i}: {desc}", ordinal=i)
                        for i, desc in enumerate(levels)
                    ],
                    _ordinal_target(row[attr]),
                )
                for attr, (instruction, levels) in HELPSTEER_LEVELS.items()
            ]
            metadata.update(task_family="judge", task_view="human-ordinal-mean-adjacent-v1")
        elif source == "commonsense_qa":
            state = row["question"]
            candidates = [
                Candidate(label, text)
                for label, text in zip(row["choices"]["label"], row["choices"]["text"], strict=True)
            ]
            if row["answerKey"] not in {c.id for c in candidates}:
                raise ValueError("raw human answer must be an original candidate")
            questions = [
                Question(
                    "q",
                    "choice",
                    "Choose the most appropriate answer.",
                    candidates,
                    {c.id: float(c.id == row["answerKey"]) for c in candidates},
                )
            ]
            metadata.update(task_family="choice", task_view="original-five-way-choice-v1")
        else:
            if row["partition"] != "train":
                raise ValueError("direct natural preparation reads original train rows only")
            names = self.raw[source]["features"]["info"]["features"]["intent"]["names"]
            label = row["intent"]
            if type(label) is not int or not 0 <= label < len(names) or len(names) < 2:
                raise ValueError("raw intent must belong to the complete original ontology")
            seed = int(fingerprint(row)[:16], 16)
            negative = (label + 1 + seed % (len(names) - 1)) % len(names)
            probes = [label, negative]
            if seed % 2:
                probes.reverse()
            state = row["utt"]
            questions = [
                Question.noul(
                    f"intent/{i}",
                    _intent_description(names, i, metadata["language"]),
                    float(i == label),
                )
                for i in probes
            ]
            metadata.update(
                task_family="intent_rehearsal",
                task_view="massive-balanced-binary-probes-v1",
                original_ontology_size=len(names),
            )
        return Sample(state, questions, metadata)

    def sources(self):
        result = {
            source: [self.sample(source, i) for i in range(len(data["rows"]))]
            for source, data in self.raw.items()
        }
        if self.policy_registry is not None:
            result.update(self.policy_registry.sources())
        return result

    def __call__(self, sample, q):
        m = sample.metadata
        if m.get("source") == CONTRACT_SOURCE:
            if self.policy_registry is None:
                raise ValueError("contract source requires the explicit pinned policy registry")
            return self.policy_registry(sample, q)
        if not valid_provenance(m) or m.get("direct_natural_version") != VERSION:
            raise ValueError("unsupported raw human natural provenance")
        source, index = m["source"], m.get("raw_row_index")
        row = self._row(source, index)
        identity = self._identity(source, index, row)
        if any(m.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "natural raw evidence, source file or translation lineage binding changed"
            )
        # Reconstruct approved wording/options from raw evidence; ignore every
        # stored target, teacher, trace and proposal in the converted sample.
        expected = self.sample(source, index)
        by_id = {question.id: question for question in expected.questions}
        if (
            sample.state != expected.state
            or q.id not in by_id
            or (
                _question_contract(q) != _question_contract(by_id[q.id])
                or {question.id for question in sample.questions} != set(by_id)
                or any(m.get(key) != expected.metadata[key] for key in ("task_family", "task_view"))
            )
        ):
            raise ValueError(
                "natural state/question/candidates differ from the approved original task view"
            )
        if source == "helpsteer2":
            return _ordinal_target(row[q.id])
        if source == "commonsense_qa":
            return {c.id: float(c.id == row["answerKey"]) for c in q.candidates}
        positive = int(q.id.split("/")[1]) == row["intent"]
        return {"false": float(not positive), "true": float(positive)}


def explicit_policy_registry(train_path):
    """CLI opt-in; absent paths keep the original four-source default."""
    return (
        NaturalGoldRegistry(policy_registry=ContractGoldRegistry(train_path))
        if train_path is not None
        else None
    )
