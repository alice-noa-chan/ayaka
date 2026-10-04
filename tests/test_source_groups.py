import copy

import pytest

from ayaka.data.decontam import Decontaminator
from ayaka.data.natural_training_v2 import SPLITS, partition_sources
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.data.source_groups import connected_groups, group_evidence
from ayaka.training.prepare_v2 import audit_splits, build_splits
from tests.test_natural_training_v2 import sources


def test_alias_closure_links_fields_and_is_order_independent():
    rows = [
        Sample("one", [], {"source_lineage": "a", "source_example_id": "example-a"}),
        Sample("two", [], {"source_lineage": "b", "derived_from": "example-a"}),
        Sample("three", [], {"source_lineage": "c", "translation_of": ["b"]}),
        Sample("three", [], {"source_lineage": "d"}),
        Sample("independent", [], {"source_lineage": "e"}),
    ]
    for order in (rows, rows[::-1], [rows[i] for i in (4, 2, 0, 3, 1)]):
        indices, groups = connected_groups(order)
        linked = {indices[i] for i, row in enumerate(order) if row.state != "independent"}
        assert len(linked) == 1
        linked_id = linked.pop()
        group = groups[linked_id]
        assert group["key"] == "a"
        assert group["identities"] == ["a", "b", "c", "d", "example-a"]
        assert indices[order.index(rows[-1])] != linked_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("lineage_ids", "x"),
        ("derived_from", [1]),
        ("source_example_id", ""),
        ("translation_of", {}),
    ],
)
def test_aliases_fail_closed_on_invalid_identity(field, value):
    with pytest.raises(ValueError, match="source group"):
        connected_groups([Sample("s", [], {field: value})])


def test_public_candidate_hit_removes_translated_group_before_quota(monkeypatch):
    original = sources()
    public = "this candidate contains the entire reserved public benchmark explanation"
    original["massive_ko"][0].questions[0].candidates[1].description = public
    monkeypatch.setattr(Decontaminator, "from_jevbench", lambda: Decontaminator([public]))
    splits, report = partition_sources(
        original, [], fits=lambda _: True, train_limit=200, heldout_limit=100
    )
    assert report["removed"]["reserved_or_public_overlap"] == 2
    assert all(
        s.metadata["source_lineage"] != "translation-parent-0"
        for rows in splits.values()
        for s in rows
    )
    assert all("split" not in s.metadata for rows in original.values() for s in rows)


def test_helpsteer_prompt_siblings_preserve_responses_and_fractional_targets():
    original = sources()
    first = original["massive_ko"][0]
    first.state = {"prompt": "shared prompt", "response": "first response"}
    second = copy.deepcopy(first)
    second.metadata["source_lineage"] = "different-parent"
    second.state["response"] = "second response"
    second.questions[0].target_distribution = {"false": 0.5, "true": 0.5}
    duplicate = copy.deepcopy(second)
    duplicate.metadata["source_example_id"] = "duplicate-example"
    original["massive_ko"].extend([second, duplicate])
    splits, report = partition_sources(
        original, [], fits=lambda _: True, train_limit=200, heldout_limit=100
    )
    selected = [
        (split, s) for split, rows in splits.items() for s in rows if isinstance(s.state, dict)
    ]
    assert len(selected) == 2
    assert len({split for split, _ in selected}) == 1
    assert {s.state["response"] for _, s in selected} == {"first response", "second response"}
    assert any(s.questions[0].target_distribution["true"] == 0.5 for _, s in selected)
    assert report["removed"]["duplicate_evidence"] == 1
    audit_splits(splits)


def test_reserved_alias_exclusion_propagates_through_evidence_bridge():
    original = sources()
    bridge = copy.deepcopy(original["massive_ko"][0])
    bridge.metadata.update(source_lineage="bridge-parent", source_example_id="bridge-example")
    original["massive_ko"].append(bridge)
    reserved = Sample("reserved original wording", [], {"derived_from": "bridge-example"})
    splits, report = partition_sources(
        original, [reserved], fits=lambda _: True, train_limit=200, heldout_limit=100
    )
    assert report["removed"]["reserved_or_public_overlap"] == 3
    assert not any(
        s.metadata["source_lineage"] in {"bridge-parent", "translation-parent-0"}
        for rows in splits.values()
        for s in rows
    )


def test_whitespace_variant_prompt_and_translation_are_reserved_as_one_group():
    original = sources()
    original["massive_ko"][0].state = {"prompt": "Same  prompt\nwith four words", "response": "new"}
    reserved = Sample({"prompt": "same prompt with four words", "response": "old"}, [])
    splits, report = partition_sources(
        original, [reserved], fits=lambda _: True, train_limit=200, heldout_limit=100
    )
    assert report["removed"]["reserved_or_public_overlap"] == 2
    assert not any(
        s.metadata["source_lineage"] == "translation-parent-0"
        for rows in splits.values()
        for s in rows
    )


def test_text_group_normalization_does_not_fold_media_bytes_or_paths():
    left = Sample("Use the image", [], {"media": {"data": "QQ==", "path": "image/A.png"}})
    right = copy.deepcopy(left)
    right.metadata["media"]["data"] = "qQ=="
    assert group_evidence(left) != group_evidence(right)
    right = copy.deepcopy(left)
    right.metadata["media"]["path"] = "image/a.png"
    assert group_evidence(left) != group_evidence(right)


def test_split_audit_rejects_alias_and_prompt_leakage_after_partition():
    splits, _ = partition_sources(
        sources(), [], fits=lambda _: True, train_limit=8, heldout_limit=3
    )
    splits["dev"][0].metadata["derived_from"] = splits["train"][0].metadata["source_lineage"]
    with pytest.raises(ValueError, match="source evidence/aliases"):
        audit_splits(splits)
    splits["dev"][0].metadata.pop("derived_from")
    splits["dev"][0].state = {"prompt": "same prompt", "response": "different"}
    splits["train"][0].state = {"prompt": "same prompt", "response": "original"}
    with pytest.raises(ValueError, match="source evidence/aliases"):
        audit_splits(splits)


def test_natural_alias_cannot_hide_behind_authored_source_kind():
    natural, _ = partition_sources(
        sources(), [], fits=lambda _: True, train_limit=8, heldout_limit=3
    )
    splits = build_splits(1, 1, 1)
    natural["train"][0].metadata["derived_from"] = splits["dev"][0].metadata["source_example_id"]
    for split in splits:
        splits[split] += natural[split]
    with pytest.raises(ValueError, match="source evidence/aliases"):
        audit_splits(splits)


def test_explicit_quotas_are_sample_counts_and_require_complete_fill():
    quotas = {source: dict.fromkeys(SPLITS, 2) for source in sources()}
    calls = []
    splits, report = partition_sources(
        sources(),
        [],
        fits=lambda _: False,
        quotas=quotas,
        fits_by_split=lambda sample, split: calls.append(split) or True,
    )
    assert len(calls) == 20
    assert set(calls) == set(SPLITS)
    assert all(len(rows) == 4 for rows in splits.values())
    assert all(n == 2 for n in report["counts"].values())
    quotas["massive_ko"]["dev"] = 1000
    with pytest.raises(ValueError, match="quota cannot be filled"):
        partition_sources(sources(), [], fits=lambda _: True, quotas=quotas)


@pytest.mark.parametrize(
    "text",
    [
        "日本語の評価文を選択肢だけに置いて検査する",
        "日本語の評価文を公表された長い文章の中から一致させるために十分な文字数を用意する",
    ],
)
def test_cjk_without_spaces_and_structured_fields_are_decontaminated(text):
    blocker = Decontaminator([text])
    sample = Sample(
        "unrelated",
        [
            Question(
                "q",
                "choice",
                "Choose",
                [Candidate("a", text), Candidate("b", "other")],
                {"a": 1.0, "b": 0.0},
            )
        ],
    )
    assert blocker.sample_hit(sample)
    assert blocker.text_hit("候補：" + text + "。")
    assert blocker.sample_hit(Sample({"prompt": text, "response": "other"}, []))
    assert not Decontaminator(["はい", "true", "短い共通語"]).text_hit("短い共通語")
