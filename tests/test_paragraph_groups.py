import copy
import itertools
from dataclasses import FrozenInstanceError, replace

import pytest

import ayaka.data.paragraph_groups as paragraph_module
from ayaka.data.paragraph_groups import (
    ParagraphIndex,
    audit_paragraph_roles,
    bind_paragraph_sample,
    build_paragraph_index,
    paragraph_inventory_digest,
)
from ayaka.data.source_groups import connected_groups
from ayaka.data.transforms import hotpot_decision
from ayaka.eval.read_artifact import fingerprint

NAMESPACE = "hotpotqa/article"


def row(row_id, titles, *, kind="noul", source="hotpot_val"):
    return {
        "id": row_id,
        "source": source,
        "type": kind,
        "context": {
            "title": titles,
            "sentences": [[f"{title} evidence."] for title in titles],
        },
    }


def index(rows):
    return build_paragraph_index(
        rows,
        namespace=NAMESPACE,
        expected_inventory_sha256=paragraph_inventory_digest(rows, namespace=NAMESPACE),
    )


def sample(raw):
    return hotpot_decision(
        {"context": raw["context"], "question": "Is this stated?", "answer": "yes"},
        metadata={
            "source": raw["source"],
            "source_example_id": raw["id"],
            "source_lineage": f"legacy/{raw['id']}",
            "label_source": "human",
        },
    )[0]


def test_unselected_bridge_closes_whole_inventory_and_bound_selected_samples():
    rows = [row("a", ["A", "B"]), row("b", ["B", "C"], kind=None), row("c", ["C", "D"])]
    components = index(rows)
    assert len({r.component_id for r in components.rows}) == 1
    # The quota excluded the middle row; selected whole contexts differ.
    selections = {"calibration": ["a"], "dev": ["c"]}
    minima = {(role, "hotpot_val", "noul"): 1 for role in selections}
    with pytest.raises(ValueError, match="transitive paragraph component"):
        audit_paragraph_roles(components, selections, minimum_counts=minima)
    bound = [
        bind_paragraph_sample(components, raw["id"], sample(raw)) for raw in (rows[0], rows[2])
    ]
    groups, _ = connected_groups(bound)
    assert groups[0] == groups[1]
    assert rows[1]["type"] is None


def test_large_shared_component_is_hashed_once_instead_of_once_per_member(monkeypatch):
    rows = [row(str(i), ["Shared", f"Article {i}"]) for i in range(80)]
    anchor = paragraph_inventory_digest(rows, namespace=NAMESPACE)
    original = paragraph_module.fingerprint
    component_hash_calls = []

    def counted(value):
        if (
            isinstance(value, list)
            and value
            and all(isinstance(alias, str) and alias.startswith("paragraph-") for alias in value)
        ):
            component_hash_calls.append(len(value))
        return original(value)

    monkeypatch.setattr(paragraph_module, "fingerprint", counted)
    result = build_paragraph_index(rows, namespace=NAMESPACE, expected_inventory_sha256=anchor)
    assert len({r.component_id for r in result.rows}) == 1
    assert len(component_hash_calls) == 1
    assert component_hash_calls[0] > len(rows)


def test_digest_and_components_are_order_independent_and_accept_one_shot_input():
    rows = [row("a", ["A", "B"]), row("b", ["B"]), row("c", ["C"])]
    expected = index(rows)
    for permutation in itertools.permutations(rows):
        assert index(permutation) == expected
    assert (
        build_paragraph_index(
            iter(rows),
            namespace=NAMESPACE,
            expected_inventory_sha256=expected.inventory_sha256,
        )
        == expected
    )


def test_relabelled_titles_copied_text_and_unicode_whitespace_close_conservatively():
    left, right, changed = row("a", ["Ａ"]), row("b", ["Other"]), row("c", ["a"])
    left["context"]["sentences"] = [["Shared  paragraph\ntext."]]
    right["context"]["sentences"] = [["shared paragraph text."]]
    changed["context"]["sentences"] = [["Different version of that article."]]
    assert len({r.component_id for r in index([left, right, changed]).rows}) == 1


def test_distinct_paragraphs_can_fill_predeclared_roles_without_attestation():
    rows = [row("a", ["A"]), row("b", ["B"], kind="choice")]
    report = audit_paragraph_roles(
        index(rows),
        {"calibration": ["a"], "dev": ["b"]},
        minimum_counts={
            ("calibration", "hotpot_val", "noul"): 1,
            ("dev", "hotpot_val", "choice"): 1,
        },
    )
    assert report["inventory_rows"] == report["inventory_components"] == 2
    assert report["selected_decisions"] == report["selected_components"] == 2
    assert report["counts"] == {"calibration/hotpot_val/noul": 1, "dev/hotpot_val/choice": 1}
    assert report["closure_before_selection"] is True
    assert report["label_provenance_attested"] is False
    assert report["historical_or_public_decontamination_attested"] is False
    assert report["promotable"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows[0].update(type=[]),
        lambda rows: rows[0].update(source=""),
        lambda rows: rows[0].update(id=""),
        lambda rows: rows[0].update(extra="unsupported"),
        lambda rows: rows[0].update(context={"title": [], "sentences": []}),
        lambda rows: rows[0]["context"].update(title=["A", "B"]),
        lambda rows: rows[0]["context"].update(title=[False]),
        lambda rows: rows[0]["context"].update(sentences=[[]]),
        lambda rows: rows[0]["context"].update(sentences=[[5]]),
        lambda rows: rows[0]["context"].update(sentences=[["  "]]),
        lambda rows: rows[0]["context"].update(sentences=["A evidence."]),
    ],
)
def test_malformed_context_or_inventory_fails_closed(mutate):
    rows = [row("a", ["A"])]
    mutate(rows)
    with pytest.raises(ValueError, match="paragraph"):
        index(rows)


def test_wrong_external_inventory_anchor_is_rejected():
    with pytest.raises(ValueError, match="logical anchor"):
        build_paragraph_index(
            [row("a", ["A"])], namespace=NAMESPACE, expected_inventory_sha256="0" * 64
        )


def test_index_and_bound_sample_are_isolated_from_caller_mutation():
    rows = [row("a", ["A"])]
    components = index(rows)
    original = sample(rows[0])
    snapshot = copy.deepcopy(original)
    bound = bind_paragraph_sample(components, "a", original)
    assert original == snapshot
    assert bound.state == original.state
    assert bound.questions == original.questions
    assert "legacy/a" in bound.metadata["lineage_ids"]
    rows[0]["context"]["sentences"][0][0] = "Replaced evidence."
    assert components.rows[0].state == original.state
    with pytest.raises(FrozenInstanceError):
        components.rows[0].component_id = "replacement"
    bound.metadata["lineage_ids"].clear()
    assert bind_paragraph_sample(components, "a", original).metadata["lineage_ids"]


def test_constructor_and_replace_cannot_detach_components_from_the_anchored_inventory():
    rows = [row("a", ["A", "B"]), row("bridge", ["B", "C"], kind=None), row("c", ["C", "D"])]
    components = index(rows)
    forged_rows = tuple(replace(r, component_id=f"forged/{r.id}") for r in components.rows)
    with pytest.raises(ValueError, match="init=False"):
        replace(components, rows=forged_rows)
    with pytest.raises(ValueError, match="init=False"):
        replace(components, namespace="forged")
    with pytest.raises(TypeError):
        ParagraphIndex(
            namespace=components.namespace,
            inventory_sha256=components.inventory_sha256,
            rows=forged_rows,
        )
    # The supported raw constructor must recompute exactly the same closure.
    rebuilt = ParagraphIndex(
        rows,
        namespace=components.namespace,
        expected_inventory_sha256=components.inventory_sha256,
    )
    assert rebuilt == components
    with pytest.raises(ValueError, match="transitive paragraph"):
        audit_paragraph_roles(
            rebuilt,
            {"calibration": ["a"], "dev": ["c"]},
            minimum_counts={(r, "hotpot_val", "noul"): 1 for r in ("calibration", "dev")},
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda s: setattr(s, "state", s.state + " Extra evidence."),
        lambda s: s.metadata.update(source="different"),
        lambda s: s.metadata.update(source_example_id="loop-index"),
        lambda s: s.questions.clear(),
        lambda s: setattr(s.questions[0], "type", "choice"),
    ],
)
def test_binding_rejects_changed_context_or_converter_mapping(change):
    raw = row("a", ["A"])
    original = sample(raw)
    change(original)
    with pytest.raises(ValueError, match="converted paragraph"):
        bind_paragraph_sample(index([raw]), "a", original)


@pytest.mark.parametrize(
    "selections,minima,match",
    [
        ({"dev": ["missing"]}, {("dev", "hotpot_val", "noul"): 1}, "unique existing"),
        ({"dev": ["a", "a"]}, {("dev", "hotpot_val", "noul"): 1}, "unique existing"),
        (
            {"dev": ["a"], "test": ["a"]},
            {(r, "hotpot_val", "noul"): 1 for r in ("dev", "test")},
            "unique existing",
        ),
        ({"dev": ["bridge"]}, {("dev", "hotpot_val", "noul"): 1}, "bridge"),
        ({"dev": ["a"]}, {("dev", "hotpot_val", "noul"): 2}, "below"),
        ({"dev": ["a"]}, {("dev", "different", "noul"): 1}, "lacks"),
        ({"dev": ["a"]}, {("dev", "hotpot_val", "noul"): True}, "positive int"),
        ({"dev": ["a"], "test": []}, {("dev", "hotpot_val", "noul"): 1}, "every selected role"),
        ({"unknown": ["a"]}, {("unknown", "hotpot_val", "noul"): 1}, "explicit roles"),
        ({"dev": "a"}, {("dev", "hotpot_val", "noul"): 1}, "explicit sequence"),
    ],
)
def test_role_and_minimum_contract_fails_closed(selections, minima, match):
    components = index([row("a", ["A"]), row("bridge", ["B"], kind=None)])
    with pytest.raises(ValueError, match=match):
        audit_paragraph_roles(components, selections, minimum_counts=minima)


def test_many_questions_from_one_component_cannot_fill_a_component_minimum():
    rows = [row(str(i), ["Shared", f"Article {i}"]) for i in range(8)]
    selections = {"dev": [r["id"] for r in rows]}
    minima = {("dev", "hotpot_val", "noul"): 8}
    report = audit_paragraph_roles(index(rows), selections, minimum_counts=minima)
    assert report["selected_decisions"] == 8 and report["selected_components"] == 1
    with pytest.raises(ValueError, match="below a declared component minimum"):
        audit_paragraph_roles(
            index(rows),
            selections,
            minimum_counts=minima,
            minimum_components={("dev", "hotpot_val", "noul"): 2},
        )


def test_component_mapping_preserves_omitted_bridges_and_counts_distinct_groups():
    rows = [
        row("a", ["A", "B"]),
        row("bridge", ["B", "C"], kind=None),
        row("c", ["C", "D"]),
        row("other", ["Other"]),
    ]
    selected = {"dev": ["a", "c", "other"]}
    minima = {("dev", "hotpot_val", "noul"): 3}
    report = audit_paragraph_roles(
        index(rows),
        selected,
        minimum_counts=minima,
        minimum_components={("dev", "hotpot_val", "noul"): 2},
    )
    audit = report["component_audit"]
    mapping = audit["membership"]["dev"]
    assert mapping["a"] == mapping["c"] != mapping["other"]
    assert "bridge" not in mapping
    assert audit["counts"] == {"dev/hotpot_val/noul": 2}
    assert audit["minimum_counts"] == {"dev/hotpot_val/noul": 2}
    assert audit["components_per_role"] == {"dev": 2}
    assert audit["membership_sha256"] == fingerprint(audit["membership"])
    assert audit["independence_attested"] is False
    assert audit["effective_sample_size_estimated"] is False
    with pytest.raises(ValueError, match="component minimum"):
        audit_paragraph_roles(
            index(rows),
            selected,
            minimum_counts=minima,
            minimum_components={("dev", "hotpot_val", "noul"): 3},
        )


def test_component_cell_counts_are_not_added_up_as_independent_role_counts():
    rows = [
        row("noul", ["Shared"]),
        row("choice", ["Shared"], kind="choice"),
        row("other_source", ["Shared"], source="other"),
    ]
    minima = {("dev", r["source"], r["type"]): 1 for r in rows}
    report = audit_paragraph_roles(
        index(rows),
        {"dev": [r["id"] for r in rows]},
        minimum_counts=minima,
        minimum_components=minima,
    )
    audit = report["component_audit"]
    assert len(audit["counts"]) == 3 and set(audit["counts"].values()) == {1}
    assert audit["components_per_role"] == {"dev": 1}
    assert report["selected_components"] == 1


def test_component_minima_are_opt_in_and_preserve_legacy_receipt_shape():
    components = index([row("a", ["A"])])
    kwargs = {"minimum_counts": {("dev", "hotpot_val", "noul"): 1}}
    original = audit_paragraph_roles(components, {"dev": ["a"]}, **kwargs)
    assert "component_audit" not in original
    assert (
        audit_paragraph_roles(components, {"dev": ["a"]}, minimum_components=None, **kwargs)
        == original
    )


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2", None])
def test_component_minima_require_positive_integer_counts(value):
    with pytest.raises(ValueError, match="positive int"):
        audit_paragraph_roles(
            index([row("a", ["A"])]),
            {"dev": ["a"]},
            minimum_counts={("dev", "hotpot_val", "noul"): 1},
            minimum_components={("dev", "hotpot_val", "noul"): value},
        )


@pytest.mark.parametrize(
    "minima",
    [
        {},
        [],
        {("dev", "other", "noul"): 1},
        {("dev", "hotpot_val", "noul"): 1, ("dev", "hotpot_val", "choice"): 1},
    ],
)
def test_component_minima_cover_exactly_the_declared_decision_cells(minima):
    with pytest.raises(ValueError, match="every declared decision cell"):
        audit_paragraph_roles(
            index([row("a", ["A"])]),
            {"dev": ["a"]},
            minimum_counts={("dev", "hotpot_val", "noul"): 1},
            minimum_components=minima,
        )


def test_component_membership_and_anchor_are_order_stable_and_isolated():
    rows = [row("a", ["A"]), row("b", ["B"]), row("c", ["C"], kind="choice")]
    minima = {("dev", "hotpot_val", "noul"): 2, ("test", "hotpot_val", "choice"): 1}

    def audit(components, selections):
        return audit_paragraph_roles(
            components,
            selections,
            minimum_counts=minima,
            minimum_components=minima,
        )

    components = index(rows)
    original = audit(components, {"dev": ["a", "b"], "test": ["c"]})
    reversed_order = audit(index(list(reversed(rows))), {"test": ["c"], "dev": ["b", "a"]})
    assert original == reversed_order
    snapshot = copy.deepcopy(original)
    original["component_audit"]["membership"]["dev"].clear()
    assert audit(components, {"dev": ["a", "b"], "test": ["c"]}) == snapshot


def test_component_minima_cannot_bypass_existing_cross_role_rejection():
    rows = [row("a", ["A", "B"]), row("bridge", ["B", "C"], kind=None), row("c", ["C"])]
    minima = {(role, "hotpot_val", "noul"): 1 for role in ("dev", "test")}
    with pytest.raises(ValueError, match="transitive paragraph component"):
        audit_paragraph_roles(
            index(rows),
            {"dev": ["a"], "test": ["c"]},
            minimum_counts=minima,
            minimum_components=minima,
        )
