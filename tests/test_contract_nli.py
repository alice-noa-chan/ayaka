import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from ayaka.data import contract_nli as module
from ayaka.data.schema import Sample
from ayaka.data.source_groups import connected_groups
from ayaka.eval.read_artifact import fingerprint


def contract_fixture(count=180):
    labels = {
        key: {
            "hypothesis": f"The contract requires original obligation {key}.",
            "short_description": key,
        }
        for key in module.HYPOTHESIS_IDS
    }
    documents = []
    for i in range(count):
        text = (
            f"Original agreement {i}. All material must be returned except archival legal copies."
        )
        documents.append(
            {
                "id": i,
                "file_name": f"agreement-{i}.txt",
                "text": text,
                "spans": [[0, len(text)]],
                "annotation_sets": [
                    {
                        "annotations": {
                            key: {
                                "choice": module.LABELS[(i + j) % 3][0],
                                "spans": [] if (i + j) % 3 == 2 else [0],
                            }
                            for j, key in enumerate(module.HYPOTHESIS_IDS)
                        }
                    }
                ],
                "document_type": "sec-text",
                "url": f"https://example.org/agreement/{i}",
            }
        )
    return {"documents": documents, "labels": labels}


def contract_registry(count=180):
    return module.ContractGoldRegistry(fixture=contract_fixture(count))


def test_full_original_document_hypotheses_and_neutral_class_reach_inputs_without_gold_spans():
    registry = contract_registry(3)
    sample = registry.sample(0)
    assert sample.state == registry.raw["documents"][0]["text"]
    assert [q.id for q in sample.questions] == list(module.HYPOTHESIS_IDS)
    assert len(sample.questions) == 17
    assert {q.type for q in sample.questions} == {"choice"}
    assert "spans" not in json.dumps(sample.metadata)
    assert "file_name" not in sample.state and "https://" not in sample.state
    targets = {tuple(registry(sample, q).values()) for q in sample.questions}
    assert targets == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    assert "distinct from Contradiction" in sample.questions[2].instruction
    assert registry.binding["scope"] == "injected CPU fixture only"


def test_gold_is_read_from_original_annotation_even_after_stored_target_and_option_order_change():
    registry = contract_registry()
    sample = registry.sample(0)
    question = sample.questions[0]
    question.target_distribution = {"NotMentioned": 1}
    question.candidates.reverse()
    assert registry(sample, question) == {"NotMentioned": 0, "Contradiction": 0, "Entailment": 1}


@pytest.mark.parametrize(
    "damage",
    [
        "truncate",
        "drop_hypothesis",
        "replace_hypothesis",
        "label",
        "instruction",
        "raw_hash",
        "lineage",
        "remove_alias",
        "split",
        "index_bool",
    ],
)
def test_raw_gold_rejects_changed_evidence_hypotheses_and_original_identity(damage):
    registry = contract_registry()
    sample = registry.sample(0)
    if damage == "truncate":
        sample.state = sample.state[:30]
    elif damage == "drop_hypothesis":
        sample.questions.pop()
    elif damage == "replace_hypothesis":
        sample.questions[-1] = copy.deepcopy(sample.questions[0])
    elif damage == "label":
        sample.questions[0].candidates[2].description = "False"
    elif damage == "instruction":
        sample.questions[0].instruction += " Answer yes or no."
    else:
        key = {
            "raw_hash": "raw_row_sha256",
            "lineage": "source_lineage",
            "remove_alias": "lineage_ids",
            "split": "original_split",
            "index_bool": "raw_row_index",
        }[damage]
        sample.metadata[key] = (
            [] if damage == "remove_alias" else True if damage == "index_bool" else "changed"
        )
    with pytest.raises(ValueError):
        registry(sample, sample.questions[0])


@pytest.mark.parametrize("damage", ["raw", "labels", "binding", "verified", "path"])
def test_registry_exit_rejects_memory_and_binding_mutation(damage):
    registry = contract_registry()
    if damage == "raw":
        registry.raw["documents"][0]["annotation_sets"][0]["annotations"]["nda-1"]["choice"] = (
            "NotMentioned"
        )
    elif damage == "labels":
        registry.raw["labels"]["nda-1"]["hypothesis"] = "Changed obligation."
    elif damage == "binding":
        registry.binding["sha256"] = "0" * 64
    elif damage == "verified":
        registry.local_files_verified = True
    else:
        registry.path = "elsewhere/train.json"
    with pytest.raises(ValueError, match="changed after validation"):
        registry.verify_files()


@pytest.mark.parametrize(
    "damage",
    [
        "partial_labels",
        "partial_annotations",
        "duplicate_id",
        "multiple_annotations",
        "evidence_offset",
        "evidence_index",
        "neutral_evidence",
        "no_positive_evidence",
    ],
)
def test_raw_schema_requires_complete_original_task_and_valid_evidence(damage):
    raw = contract_fixture(2)
    doc = raw["documents"][0]
    if damage == "partial_labels":
        raw["labels"].pop("nda-1")
    elif damage == "partial_annotations":
        doc["annotation_sets"][0]["annotations"].pop("nda-1")
    elif damage == "duplicate_id":
        raw["documents"][1]["id"] = doc["id"]
    elif damage == "multiple_annotations":
        doc["annotation_sets"].append(copy.deepcopy(doc["annotation_sets"][0]))
    elif damage == "evidence_offset":
        doc["spans"][0][1] += 1
    else:
        annotation = doc["annotation_sets"][0]["annotations"]["nda-1"]
        if damage == "evidence_index":
            annotation["spans"] = [3]
        elif damage == "neutral_evidence":
            annotation["choice"] = "NotMentioned"
        else:
            annotation["spans"] = []
    with pytest.raises(ValueError):
        module.ContractGoldRegistry(fixture=raw)


def test_document_url_and_filename_aliases_close_siblings_and_reserved_excerpts():
    raw = contract_fixture(3)
    raw["documents"][1]["url"] = raw["documents"][0]["url"] + "#clause-4"
    raw["documents"][2]["file_name"] = raw["documents"][1]["file_name"]
    registry = module.ContractGoldRegistry(fixture=raw)
    samples = registry.sources()[module.SOURCE]
    reserved = Sample(
        "An excerpt with a restored source alias.",
        [],
        {"derived_from": samples[0].metadata["lineage_ids"][1]},
    )
    groups, _ = connected_groups(samples + [reserved])
    assert len(set(groups)) == 1


def test_original_train_and_license_are_pinned_before_json_parse_without_test_access(
    tmp_path, monkeypatch
):
    raw = contract_fixture(423)
    train = tmp_path / "train.json"
    train.write_bytes(json.dumps(raw).encode())
    license_file = tmp_path / "LICENSE"
    license_file.write_bytes(b"test-only license fixture")
    monkeypatch.setattr(module, "TRAIN_SHA256", module._sha(train))
    monkeypatch.setattr(module, "LICENSE_SHA256", module._sha(license_file))
    original = module.json.loads

    def guarded_load(value, **kwargs):
        assert module._sha(train) == module.TRAIN_SHA256
        assert module._sha(license_file) == module.LICENSE_SHA256
        return original(value, **kwargs)

    monkeypatch.setattr(module.json, "loads", guarded_load)
    registry = module.ContractGoldRegistry(train)
    assert registry.local_files_verified
    assert registry.binding["documents"] == 423
    assert registry.binding["labels_sha256"] == fingerprint(raw["labels"])
    train.write_bytes(train.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bytes changed"):
        registry.verify_files()
    with pytest.raises(ValueError, match="pinned original"):
        module.ContractGoldRegistry(train)


def test_duplicate_json_keys_are_rejected_instead_of_silently_losing_original_annotations():
    with pytest.raises(ValueError, match="duplicate key"):
        json.loads('{"nda-1": 1, "nda-1": 2}', object_pairs_hook=module._unique_object)


def test_parser_consumes_the_exact_anchored_bytes_even_if_a_later_read_temporarily_changes(
    tmp_path, monkeypatch
):
    raw = contract_fixture(423)
    train = tmp_path / "train.json"
    approved = json.dumps(raw).encode()
    train.write_bytes(approved)
    license_file = tmp_path / "LICENSE"
    license_file.write_bytes(b"fixture license")
    changed = copy.deepcopy(raw)
    changed["documents"][0]["annotation_sets"][0]["annotations"]["nda-1"]["choice"] = (
        "Contradiction"
    )
    injected = json.dumps(changed).encode()
    monkeypatch.setattr(module, "TRAIN_SHA256", module._sha(train))
    monkeypatch.setattr(module, "LICENSE_SHA256", module._sha(license_file))
    original = Path.read_bytes
    reads = []

    def swap_second_read(path):
        if path.resolve() == train.resolve():
            reads.append(path)
            if len(reads) == 2:
                return injected
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", swap_second_read)
    with pytest.raises(ValueError, match="bytes changed"):
        module.ContractGoldRegistry(train)


def archive_fixture(tmp_path, monkeypatch, *, extra=()):
    train = json.dumps(contract_fixture(423)).encode()
    license_bytes = b"fixture license"
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(module.SOURCE_POLICY[2], train)
        stream.writestr("contract-nli/LICENSE", license_bytes)
        stream.writestr("contract-nli/dev.json", b"original dev must remain unopened")
        stream.writestr("contract-nli/test.json", b"original test must remain unopened")
        for name in extra:
            info = zipfile.ZipInfo(name)
            # Windows ZipInfo normalizes backslashes; emulate bytes from an
            # archive made elsewhere to exercise the reader's actual boundary.
            info.filename = name
            stream.writestr(info, b"ambiguous or unsafe member")
    raw = archive.read_bytes()
    monkeypatch.setattr(module, "ARCHIVE_SHA256", module._sha(archive))
    monkeypatch.setattr(module, "ARCHIVE_BYTES", len(raw))
    monkeypatch.setattr(
        module, "ARCHIVE_GIT_BLOB", hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    )
    monkeypatch.setattr(module, "TRAIN_SHA256", hashlib.sha256(train).hexdigest())
    monkeypatch.setattr(module, "LICENSE_SHA256", hashlib.sha256(license_bytes).hexdigest())
    return archive


def test_extraction_opens_only_exact_train_and_license_members(tmp_path, monkeypatch):
    archive = archive_fixture(tmp_path, monkeypatch)
    opened = []
    original = zipfile.ZipFile.read

    def guarded_read(self, name, *args, **kwargs):
        assert name in {module.SOURCE_POLICY[2], "contract-nli/LICENSE"}
        opened.append(name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    out = tmp_path / "training-only"
    report = module.extract_training(archive, out)
    assert opened == report["opened_members"]
    assert report["original_dev_test_opened"] is False
    assert report["questions"] == 7191
    assert {p.name for p in out.iterdir()} == {"train.json", "LICENSE", "extraction.json"}
    assert module.ContractGoldRegistry(out / "train.json").local_files_verified


@pytest.mark.parametrize(
    "extra", [[module.SOURCE_POLICY[2]], ["../train.json"], ["/train.json"], [r"other\train.json"]]
)
def test_extraction_rejects_duplicate_or_unsafe_members_without_publishing(
    tmp_path, monkeypatch, extra
):
    archive = archive_fixture(tmp_path, monkeypatch, extra=extra)
    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="duplicate or unsafe"):
        module.extract_training(archive, out)
    assert not out.exists()


def test_extraction_rechecks_archive_after_parsing_before_output_creation(tmp_path, monkeypatch):
    archive = archive_fixture(tmp_path, monkeypatch)
    original = module.ContractGoldRegistry._validate

    def mutate_after_parse(self):
        original(self)
        archive.write_bytes(archive.read_bytes() + b" ")

    monkeypatch.setattr(module.ContractGoldRegistry, "_validate", mutate_after_parse)
    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="changed before"):
        module.extract_training(archive, out)
    assert not out.exists()
