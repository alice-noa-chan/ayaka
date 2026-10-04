"""Pinned, full-document ContractNLI training annotations for policy decisions.

The 17 original hypotheses and three-way labels remain intact. Human evidence
spans stay in the raw registry, never in model inputs. The public train corpus
is useful supervision, not a private benchmark or a proof of teacher benefit.
No download, original development/test access, model or optimizer is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from urllib.parse import urldefrag

from ..eval.read_artifact import fingerprint
from .schema import Candidate, Question, Sample

SOURCE = "contract_nli"
REPO = "stanfordnlp/contract-nli"
REVISION = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
VERSION = "ayaka-contract-nli-raw-gold-1"
TRAIN_SHA256 = "dbceb356cd6203b35b27be94a5fa85e499a81c34c42c89ad53060b39f0257ba5"
LICENSE_SHA256 = "9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94"
ARCHIVE_SHA256 = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
ARCHIVE_BYTES = 65_362_913
ARCHIVE_GIT_BLOB = "757fd1dafd29a997fba00c60c6d40b2930a36159"
SOURCE_POLICY = (REPO, REVISION, "contract-nli/train.json", "en")
HYPOTHESIS_IDS = tuple(
    f"nda-{i}" for i in (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20)
)
LABELS = (
    ("Entailment", "The contract entails the hypothesis."),
    ("Contradiction", "The contract contradicts the hypothesis."),
    ("NotMentioned", "The contract neither entails nor contradicts the hypothesis."),
)
INSTRUCTION = (
    "Classify the following hypothesis using the entire contract, including its exceptions. "
    "Choose Entailment, Contradiction, or NotMentioned. "
    "NotMentioned means neither entailed nor contradicted; it is distinct from Contradiction.\n"
    "Hypothesis: "
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in original contract JSON")
        result[key] = value
    return result


def _text(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _question_contract(question):
    return {
        "id": question.id,
        "type": question.type,
        "instruction": question.instruction,
        "candidates": sorted((asdict(c) for c in question.candidates), key=lambda c: c["id"]),
    }


class ContractGoldRegistry:
    """Offline original train bytes and annotations, with immutable exit guards.

    Pass the extracted official train.json and its sibling LICENSE explicitly.
    Fixture injection is identified as mechanics-only and must never reach a
    native-model preparation. No fallback to a mutable repository is possible.
    """

    def __init__(self, train_path=None, *, fixture=None):
        if (train_path is None) == (fixture is None):
            raise ValueError("supply the pinned ContractNLI train file or an explicit CPU fixture")
        self.local_files_verified = fixture is None
        self.path = Path(train_path).resolve() if train_path is not None else None
        if self.local_files_verified:
            train_bytes = self.path.read_bytes()
            if (
                self.path.name != "train.json"
                or hashlib.sha256(train_bytes).hexdigest() != TRAIN_SHA256
            ):
                raise ValueError("ContractNLI requires the pinned original train.json bytes")
            if _sha(self.path.with_name("LICENSE")) != LICENSE_SHA256:
                raise ValueError("ContractNLI requires its pinned CC-BY-4.0 license")
            self.raw = json.loads(train_bytes, object_pairs_hook=_unique_object)
        else:
            self.raw = fixture
        self._validate()
        self.binding = {
            "version": VERSION,
            "local_files_verified": self.local_files_verified,
            "repo": REPO,
            "revision": REVISION,
            "archive_file": "resources/contract-nli.zip",
            "archive_sha256": ARCHIVE_SHA256,
            "file": SOURCE_POLICY[2],
            "sha256": TRAIN_SHA256 if self.local_files_verified else fingerprint(self.raw),
            "license": "CC-BY-4.0",
            "license_sha256": LICENSE_SHA256 if self.local_files_verified else None,
            "original_split": "train",
            "documents": len(self.raw["documents"]),
            "questions_per_document": len(HYPOTHESIS_IDS),
            "labels_sha256": fingerprint(self.raw["labels"]),
            "scope": "public original full-contract human training annotations"
            if self.local_files_verified
            else "injected CPU fixture only",
        }
        self._memory_sha256 = self._memory_identity()
        self._path_identity = str(self.path)
        self.verify_files()

    def _validate(self):
        if (
            not isinstance(self.raw, dict)
            or set(self.raw) != {"documents", "labels"}
            or not isinstance(self.raw["documents"], list)
            or not self.raw["documents"]
            or not isinstance(self.raw["labels"], dict)
            or set(self.raw["labels"]) != set(HYPOTHESIS_IDS)
        ):
            raise ValueError("ContractNLI requires its complete documents and 17 hypotheses")
        if self.local_files_verified and len(self.raw["documents"]) != 423:
            raise ValueError("ContractNLI original train document count differs")
        for label in self.raw["labels"].values():
            if (
                not isinstance(label, dict)
                or set(label) != {"hypothesis", "short_description"}
                or any(not isinstance(v, str) or not v.strip() for v in label.values())
            ):
                raise ValueError("ContractNLI hypothesis definitions must remain complete")
        ids = set()
        for doc in self.raw["documents"]:
            if (
                not isinstance(doc, dict)
                or set(doc)
                != {"id", "file_name", "text", "spans", "annotation_sets", "document_type", "url"}
                or type(doc["id"]) is not int
                or doc["id"] in ids
                or any(
                    not isinstance(doc[k], str) or not doc[k].strip()
                    for k in ("text", "file_name", "url")
                )
                or doc["document_type"] not in {"search-pdf", "sec-text", "sec-html"}
                or not isinstance(doc["spans"], list)
                or not isinstance(doc["annotation_sets"], list)
                or len(doc["annotation_sets"]) != 1
            ):
                raise ValueError("ContractNLI original document identity/annotation schema differs")
            ids.add(doc["id"])
            for span in doc["spans"]:
                if (
                    not isinstance(span, list)
                    or len(span) != 2
                    or any(type(n) is not int for n in span)
                    or not 0 <= span[0] < span[1] <= len(doc["text"])
                ):
                    raise ValueError(
                        "ContractNLI evidence offsets exceed the original full document"
                    )
            annotation_set = doc["annotation_sets"][0]
            if (
                not isinstance(annotation_set, dict)
                or set(annotation_set) != {"annotations"}
                or not isinstance(annotation_set["annotations"], dict)
                or set(annotation_set["annotations"]) != set(HYPOTHESIS_IDS)
            ):
                raise ValueError("ContractNLI must retain all 17 original annotations per document")
            for annotation in annotation_set["annotations"].values():
                if (
                    not isinstance(annotation, dict)
                    or set(annotation) != {"choice", "spans"}
                    or annotation["choice"] not in dict(LABELS)
                    or not isinstance(annotation["spans"], list)
                    or any(
                        type(i) is not int or not 0 <= i < len(doc["spans"])
                        for i in annotation["spans"]
                    )
                    or len(set(annotation["spans"])) != len(annotation["spans"])
                    or bool(annotation["spans"]) == (annotation["choice"] == "NotMentioned")
                ):
                    raise ValueError("ContractNLI original label/evidence annotation differs")

    def _memory_identity(self):
        return fingerprint(
            {"raw": self.raw, "binding": self.binding, "verified": self.local_files_verified}
        )

    def verify_files(self):
        if self._memory_identity() != self._memory_sha256 or str(self.path) != self._path_identity:
            raise ValueError("parsed contract annotations/binding/path changed after validation")
        if self.local_files_verified and (
            _sha(self.path) != TRAIN_SHA256
            or _sha(self.path.with_name("LICENSE")) != LICENSE_SHA256
        ):
            raise ValueError("ContractNLI train/license bytes changed after validation")

    def _document(self, index):
        if type(index) is not int or not 0 <= index < len(self.raw["documents"]):
            raise ValueError("contract sample must name an existing original train document")
        return self.raw["documents"][index]

    def _identity(self, index, doc):
        return {
            "source": SOURCE,
            "revision": REVISION,
            "language": "en",
            "license": "CC-BY-4.0",
            "label_source": "human",
            "original_split": "train",
            "data_kind": "natural",
            "modality": "text",
            "direct_policy_version": VERSION,
            "task_family": "contract_policy",
            "task_view": "original-full-document-17-hypotheses-three-way-v1",
            "raw_row_index": index,
            "raw_row_sha256": fingerprint(doc),
            "raw_file_sha256": self.binding["sha256"],
            "hypotheses_sha256": self.binding["labels_sha256"],
            "source_lineage": "contract-nli/document/" + fingerprint(_text(doc["text"])),
            "source_example_id": f"contract-nli/train/{doc['id']}/{fingerprint(doc)}",
            "lineage_ids": [
                f"contract-nli/id/{doc['id']}",
                "contract-nli/url/" + fingerprint(urldefrag(doc["url"])[0]),
                "contract-nli/file/" + fingerprint(doc["file_name"]),
            ],
        }

    def sample(self, index):
        doc = self._document(index)
        annotations = doc["annotation_sets"][0]["annotations"]
        return Sample(
            doc["text"],
            [
                Question(
                    key,
                    "choice",
                    INSTRUCTION + self.raw["labels"][key]["hypothesis"],
                    [Candidate(label, description) for label, description in LABELS],
                    {label: float(label == annotations[key]["choice"]) for label, _ in LABELS},
                )
                for key in HYPOTHESIS_IDS
            ],
            self._identity(index, doc),
        )

    def sources(self):
        return {SOURCE: [self.sample(i) for i in range(len(self.raw["documents"]))]}

    def __call__(self, sample, question):
        index = sample.metadata.get("raw_row_index")
        doc = self._document(index)
        identity = self._identity(index, doc)
        # Split preparation augments lineage_ids with the transitive closure.
        # Require every original alias while allowing those audited additions.
        aliases = sample.metadata.get("lineage_ids")
        if (
            not isinstance(aliases, list)
            or any(not isinstance(alias, str) for alias in aliases)
            or not set(identity["lineage_ids"]).issubset(aliases)
            or any(sample.metadata.get(k) != v for k, v in identity.items() if k != "lineage_ids")
        ):
            raise ValueError("contract raw evidence/identity/hypothesis binding changed")
        expected = self.sample(index)
        questions = {q.id: q for q in expected.questions}
        if (
            sample.state != doc["text"]
            or len(sample.questions) != len(HYPOTHESIS_IDS)
            or {q.id for q in sample.questions} != set(HYPOTHESIS_IDS)
            or question.id not in questions
            or _question_contract(question) != _question_contract(questions[question.id])
        ):
            raise ValueError("contract full text/hypotheses/three-way candidates differ")
        label = doc["annotation_sets"][0]["annotations"][question.id]["choice"]
        return {candidate.id: float(candidate.id == label) for candidate in question.candidates}


def extract_training(archive, out):
    """Extract exactly train.json and LICENSE from the anchored official ZIP.

    Other names are inspected for archive ambiguity, but their contents are
    never read. No basename fallback, extractall or original dev/test parsing.
    """
    path, destination = Path(archive).resolve(), Path(out).resolve()
    if destination.exists():
        raise ValueError("preserve existing dataset records; choose a new output")
    raw = path.read_bytes()
    if (
        len(raw) != ARCHIVE_BYTES
        or hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256
        or hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != ARCHIVE_GIT_BLOB
    ):
        raise ValueError("ContractNLI archive differs from pinned revision/blob/byte anchors")
    with zipfile.ZipFile(path) as stream:
        # filename normalizes platform separators and NULs. Inspect the
        # original member spelling before accepting that interpretation.
        names = [info.orig_filename for info in stream.infolist()]
        if len(names) != len(set(names)) or any(
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or "\\" in name
            or "\0" in name
            for name in names
        ):
            raise ValueError("ContractNLI archive has duplicate or unsafe member paths")
        contents = {
            "train.json": stream.read(SOURCE_POLICY[2]),
            "LICENSE": stream.read("contract-nli/LICENSE"),
        }
    if (
        hashlib.sha256(contents["train.json"]).hexdigest() != TRAIN_SHA256
        or hashlib.sha256(contents["LICENSE"]).hexdigest() != LICENSE_SHA256
    ):
        raise ValueError("ContractNLI train/license members differ from pinned byte anchors")
    contract = ContractGoldRegistry(
        fixture=json.loads(contents["train.json"], object_pairs_hook=_unique_object)
    )
    if len(contract.raw["documents"]) != 423:
        raise ValueError("ContractNLI original train must contain 423 complete documents")
    report = {
        "version": "ayaka-contract-nli-extraction-1",
        "repo": REPO,
        "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_git_blob": ARCHIVE_GIT_BLOB,
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
        "opened_members": [SOURCE_POLICY[2], "contract-nli/LICENSE"],
        "original_dev_test_opened": False,
        "documents": 423,
        "questions": 423 * len(HYPOTHESIS_IDS),
        "optimizer_steps": 0,
        "model_weights_loaded": False,
        "paid_execution_started": False,
        "promotable": False,
    }
    contract.verify_files()
    if _sha(path) != ARCHIVE_SHA256:
        raise ValueError("ContractNLI archive changed before extraction publication")
    destination.mkdir(parents=True)
    for name, content in contents.items():
        with (destination / name).open("xb") as handle:
            handle.write(content)
    with (destination / "extraction.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(extract_training(args.archive, args.out), indent=2))


if __name__ == "__main__":
    main()
