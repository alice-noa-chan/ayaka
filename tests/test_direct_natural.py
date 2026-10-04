import copy

import pytest

from ayaka.config import tiny_config
from ayaka.data.direct_natural import NaturalGoldRegistry, _ordinal_target
from ayaka.data.natural_training_v2 import SOURCES, digest, evidence, partition_sources
from ayaka.eval.read_artifact import fingerprint
from ayaka.tokenization import ToyTokenizer
from ayaka.training.direct_bundle import audit_bundle, prepare_bundle
from ayaka.training.prepare_v2 import audit_splits


def raw_registry():
    raw = {}
    for source, (repo, revision, filename, language) in SOURCES.items():
        rows = []
        for i in range(160):
            if source == "helpsteer2":
                row = {
                    "prompt": f"Write a response to original request {i}.",
                    "response": f"Human reviewed response {i}.",
                    "helpfulness": 3.5,
                    "correctness": 4,
                    "coherence": 2.5,
                    "complexity": 1.5,
                    "verbosity": 2,
                }
            elif source == "commonsense_qa":
                row = {
                    "question": f"Original human labelled five-way question {i}?",
                    "choices": {
                        "label": list("ABCDE"),
                        "text": [f"answer {j} for original case {i}" for j in range(5)],
                    },
                    "answerKey": "C",
                }
            else:
                row = {
                    "id": i,
                    "partition": "train",
                    "utt": f"{language} original distinct human request {i}",
                    "intent": i % 60,
                }
            rows.append(row)
        raw[source] = {
            "rows": rows,
            "features": {
                "info": {"features": {"intent": {"names": [f"intent_{i}" for i in range(60)]}}}
            },
            "provenance": {
                "repo": repo,
                "revision": revision,
                "file": filename,
                "sha256": fingerprint(rows),
            },
        }
    return NaturalGoldRegistry(raw)


def test_original_fractional_human_ratings_are_not_truncated_or_called_vote_counts():
    registry = raw_registry()
    sample = registry.sample("helpsteer2", 0)
    target = registry(sample, sample.questions[0])
    assert target == {"s0": 0, "s1": 0, "s2": 0, "s3": 0.5, "s4": 0.5}
    assert sum(int(key[1:]) * value for key, value in target.items()) == 3.5
    assert sample.metadata["task_view"] == "human-ordinal-mean-adjacent-v1"
    assert not registry.local_files_verified
    for value in (-1, 5, True, float("nan")):
        with pytest.raises(ValueError):
            _ordinal_target(value)


def test_massive_binary_task_is_balanced_order_varied_and_translation_grouped():
    registry = raw_registry()
    first_targets = set()
    for i in range(25):
        ko, ja = registry.sample("massive_ko", i), registry.sample("massive_ja", i)
        assert ko.metadata["source_lineage"] == ja.metadata["source_lineage"]
        assert ko.metadata["task_view"] == "massive-balanced-binary-probes-v1"
        assert ko.metadata["original_ontology_size"] == 60
        assert len(ko.questions) == 2
        assert [q.type for q in ko.questions] == ["noul", "noul"]
        positive = [registry(ko, q)["true"] for q in ko.questions]
        assert sum(positive) == 1
        first_targets.add(positive[0])
    assert first_targets == {0, 1}


def test_raw_registry_keeps_previous_source_group_hashes_for_reserved_split_stability():
    registry = raw_registry()
    for source in SOURCES:
        sample = registry.sample(source, 3)
        parent = ["massive", "3"] if source.startswith("massive_") else [source, evidence(sample)]
        assert sample.metadata["source_lineage"] == "natural/" + digest(parent)


@pytest.mark.parametrize(
    "damage",
    [
        "target",
        "state",
        "instruction",
        "description",
        "row_hash",
        "file_hash",
        "lineage",
        "revision",
    ],
)
def test_gold_checks_raw_human_labels_and_rejects_changed_original_evidence(damage):
    registry = raw_registry()
    sample = registry.sample("commonsense_qa", 3)
    q = sample.questions[0]
    if damage == "target":
        q.target_distribution = {"A": 1}
        assert registry(sample, q)["C"] == 1
        return
    if damage == "state":
        sample.state = "another input"
    elif damage == "instruction":
        q.instruction += " Changed task."
    elif damage == "description":
        q.candidates[0].description = "changed candidate"
    else:
        field = {
            "row_hash": "raw_row_sha256",
            "file_hash": "raw_file_sha256",
            "lineage": "source_lineage",
            "revision": "revision",
        }[damage]
        sample.metadata[field] = "changed"
    with pytest.raises(ValueError):
        registry(sample, q)


def test_raw_source_natural_bundle_roundtrip_retains_five_split_gold_and_file_binding(tmp_path):
    registry = raw_registry()
    splits, report = partition_sources(
        registry.sources(), [], fits=lambda _: True, train_limit=4, heldout_limit=2
    )
    audit_splits(splits)
    assert all(report["counts"][f"{source}/{split}"] > 0 for source in SOURCES for split in splits)
    prepare_bundle(
        tmp_path / "bundle",
        splits,
        ToyTokenizer(),
        tiny_config(readout="lm", max_seq_len=2048),
        {},
        steps=2,
        rows_per_step=8,
        allow_tiny=True,
        natural_registry=registry,
    )
    _, recipe, items, _, _ = audit_bundle(
        tmp_path / "bundle", allow_tiny=True, natural_registry=registry
    )
    assert recipe["gold_sources"] == registry.binding
    assert {it.type for it in items} == {"score", "choice", "noul"}
    assert all(it.direct_distillation and not it.reasoning_labels for it in items)
    changed = copy.deepcopy(registry.raw)
    changed["helpsteer2"]["provenance"]["sha256"] = fingerprint("other source file")
    with pytest.raises(ValueError, match="binding changed"):
        audit_bundle(
            tmp_path / "bundle", allow_tiny=True, natural_registry=NaturalGoldRegistry(changed)
        )


def test_corrupted_converted_human_gold_cannot_enter_native_direct_training(tmp_path):
    registry = raw_registry()
    splits, _ = partition_sources(
        registry.sources(), [], fits=lambda _: True, train_limit=4, heldout_limit=2
    )
    q = next(s for s in splits["train"] if s.metadata["source"] == "commonsense_qa").questions[0]
    q.target_distribution = {"A": 1}
    with pytest.raises(ValueError, match="gold disagrees"):
        prepare_bundle(
            tmp_path / "absent",
            splits,
            ToyTokenizer(),
            tiny_config(readout="lm", max_seq_len=2048),
            {},
            steps=2,
            rows_per_step=8,
            allow_tiny=True,
            natural_registry=registry,
        )
    assert not (tmp_path / "absent").exists()
