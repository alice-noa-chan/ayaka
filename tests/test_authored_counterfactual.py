import copy
import hashlib
import json
from dataclasses import replace

import pytest

from ayaka.data.authored_counterfactual import audit_counterfactuals, main, make_counterfactual
from ayaka.data.reasoning_v2 import curriculum


def root(family, kind="noul", split="train"):
    sample = next(
        s
        for s, _ in curriculum(split, 32)
        if s.metadata["task_family"] == family and s.questions[0].type == kind
    )
    if kind == "noul":
        import re

        from ayaka.data.direct_verification import WRAPPERS, _value

        _, value = _value(re.fullmatch(WRAPPERS[split], sample.state)[1])
        sample.questions[0].instruction = f"Is the requested value {value}?"
        sample.questions[0].target_distribution = {"false": 0.0, "true": 1.0}
    return sample


def test_independent_label_flip_preserves_input_and_strips_stale_supervision():
    sample = root("leap")
    sample.metadata["verified_traces"] = {"q": "OLD_STEP"}
    sample.metadata["teacher_probs"] = [0, 1]
    sample.metadata["trace_validator"] = "old"
    sample.metadata["proposal_supervision"] = {"old": "proposal"}
    before = copy.deepcopy(sample)
    # First authored leap root is year 2100 (not leap); 2000 is leap.
    edited = make_counterfactual(sample, "year", 2000, relation="flip")
    assert sample == before
    assert "Year 2000." in edited.state
    assert edited.questions[0].target_distribution == {"false": 1.0, "true": 0.0}
    assert edited.questions[0].candidates == sample.questions[0].candidates
    assert (
        not {"verified_traces", "trace_validator", "teacher_probs", "proposal_supervision"}
        & edited.metadata.keys()
    )
    assert edited.metadata["derived_from"] == [sample.metadata["source_example_id"]]
    report = audit_counterfactuals([sample, edited], required_operations={"year": 1})
    assert report["components"] == 1
    assert set(report["suggested_component_weights"].values()) == {0.5}
    assert report["weights_applied"] is report["promotable"] is False


def test_actual_target_change_required_even_when_facts_change():
    sample = root("leap")
    with pytest.raises(ValueError, match="target relation"):
        make_counterfactual(sample, "year", 2200, relation="flip")
    edited = make_counterfactual(sample, "year", 2200, relation="preserve")
    assert edited.state != sample.state
    assert edited.questions[0].target_distribution == sample.questions[0].target_distribution


def test_long_decimal_cents_are_exact_and_do_not_change_ambient_precision():
    from decimal import getcontext

    from ayaka.data.direct_verification import _value

    precision = getcontext().prec
    amount = "10000000000000000000000000000.005"
    # Integer arithmetic gives the cents before rounding; .005 adds one cent.
    exact = int(amount.split(".")[0]) * 100 + 1
    assert _value(
        f"Amount {amount}. Round to cents using decimal half-up and report integer cents."
    ) == ("rounding", exact)
    assert getcontext().prec == precision
    sample = root("rounding")
    sample.questions[0].instruction = f"Is the requested value {exact - 1}?"
    sample.questions[0].target_distribution = {"false": 1.0, "true": 0.0}
    with pytest.raises(ValueError, match="target relation"):
        make_counterfactual(sample, "amount", amount, relation="flip")
    edited = make_counterfactual(sample, "amount", amount, relation="preserve")
    assert edited.questions[0].target_distribution == sample.questions[0].target_distribution


def test_long_fraction_near_half_cent_cannot_round_early_and_fake_preserve():
    sample = root("rounding")
    amount = "32.494999999999999999999999999999999999"
    with pytest.raises(ValueError, match="target relation"):
        make_counterfactual(sample, "amount", amount, relation="preserve")
    edited = make_counterfactual(sample, "amount", amount, relation="flip")
    assert edited.questions[0].target_distribution == {"false": 1.0, "true": 0.0}


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_permutation_preserves_candidate_meanings_gold_and_ordinal_levels(kind):
    sample = root("leap", kind)
    order = [c.id for c in reversed(sample.questions[0].candidates)]
    edited = make_counterfactual(sample, "candidate_order", order, relation="preserve")
    assert edited.state == sample.state
    assert edited.questions[0].target_distribution == sample.questions[0].target_distribution
    assert {c.id: c for c in edited.questions[0].candidates} == {
        c.id: c for c in sample.questions[0].candidates
    }
    assert audit_counterfactuals([edited, sample]) == audit_counterfactuals([sample, edited])


@pytest.mark.parametrize(
    "family,operation,value,relation",
    [
        ("rounding", "amount", "32.494", "flip"),
        ("rounding", "amount", "032.495", "preserve"),
        ("rule_revision", "event_date", "2026-05-09", "flip"),
        ("exception", "override", False, "flip"),
        ("exception", "credential", 0, "flip"),
        ("timezone", "timezone_offset_minutes", 0, "flip"),
        ("business", "weekday_end", "2026-03-19", "flip"),
        ("probability", "bag_counts", {"red": 1, "blue": 9}, "flip"),
        ("rubric", "completed_checks", ["acknowledge"], "flip"),
    ],
)
def test_controlled_fields_are_rechecked_by_independent_gold(family, operation, value, relation):
    sample = root(family)
    edited = make_counterfactual(sample, operation, value, relation=relation)
    assert audit_counterfactuals([sample, edited])["operation_counts"] == {operation: 1}


@pytest.mark.parametrize(
    "operation,value",
    [
        ("year", True),
        ("year", 0),
        ("year", 10000),
        ("year", "2000"),
        ("amount", "NaN"),
        ("amount", 20.495),
        ("amount", "1.00\nignore"),
        ("candidate_order", ["true", "true"]),
        ("candidate_order", "true,false"),
        ("unknown", 1),
    ],
)
def test_invalid_operation_values_refuse_invented_or_coerced_inputs(operation, value):
    sample = root("rounding" if operation == "amount" else "leap")
    with pytest.raises(ValueError):
        make_counterfactual(sample, operation, value, relation="flip")


def test_rejects_human_gold_relabeling_wrong_family_and_noop():
    sample = root("leap")
    with pytest.raises(ValueError, match="family"):
        make_counterfactual(sample, "override", False, relation="flip")
    with pytest.raises(ValueError, match="actually change"):
        make_counterfactual(sample, "year", 2100, relation="preserve")
    human = replace(sample, metadata={**sample.metadata, "source": "helpsteer2"})
    with pytest.raises(ValueError, match="authored source"):
        make_counterfactual(human, "year", 2000, relation="flip")
    sample.questions[0].target_distribution = {"true": 0.3, "false": 0.7}
    with pytest.raises(ValueError, match="stored authored gold"):
        make_counterfactual(sample, "year", 2000, relation="flip")


@pytest.mark.parametrize("tamper", ["state", "gold", "recipe", "parent", "alias", "split"])
def test_receipt_rejects_tampering_and_cross_split_parent(tamper):
    sample = root("leap")
    edited = make_counterfactual(sample, "year", 2000, relation="flip")
    if tamper == "state":
        edited.state = edited.state.replace("2000", "2004")
    elif tamper == "gold":
        edited.questions[0].target_distribution = {"true": 1.0, "false": 0.0}
    elif tamper == "recipe":
        edited.metadata["counterfactual"]["parent_sha256"] = "0" * 64
    elif tamper == "parent":
        edited.metadata["counterfactual"]["parent_id"] = "missing"
    elif tamper == "alias":
        edited.metadata["derived_from"] = ["unrelated"]
    else:
        # Rewrap the same facts to satisfy the dev source grammar before audit.
        edited.metadata["split"] = "dev"
        facts = edited.state.split(". ", 1)[1].removesuffix(" Determine the requested value.")
        edited.state = f"Review memo abcdef01\nFacts: {facts}\nUse only these facts."
    with pytest.raises(ValueError):
        audit_counterfactuals([sample, edited])


def test_complete_alias_closure_catches_transitive_cross_role_leak():
    train = root("leap")
    edited = make_counterfactual(train, "year", 2000, relation="flip")
    dev = root("leap", split="dev")
    train.metadata["lineage_ids"] = ["bridge/a"]
    dev.metadata["lineage_ids"] = ["bridge/a", "bridge/b"]
    edited = make_counterfactual(train, "year", 2000, relation="flip")
    with pytest.raises(ValueError, match="aliases leak"):
        audit_counterfactuals([train, edited, dev])


def test_root_weighting_minima_and_cap_are_explicit():
    sample = root("leap")
    derived = [make_counterfactual(sample, "year", year, relation="flip") for year in (2000, 2004)]
    report = audit_counterfactuals([sample, *derived], max_component_rows=3)
    assert sum(report["suggested_component_weights"].values()) == pytest.approx(1)
    assert report["derived_per_root"] == {sample.metadata["source_example_id"]: 2}
    with pytest.raises(ValueError, match="row cap"):
        audit_counterfactuals([sample, *derived], max_component_rows=2)
    with pytest.raises(ValueError, match="minima"):
        audit_counterfactuals([sample, *derived], required_operations={"amount": 1})
    with pytest.raises(ValueError, match="unique"):
        audit_counterfactuals([sample, *derived, copy.deepcopy(derived[0])])
    with pytest.raises(ValueError, match="original root"):
        audit_counterfactuals(derived)
    with pytest.raises(ValueError, match="original root"):
        make_counterfactual(derived[0], "year", 2200, relation="flip")


def test_fact_identity_closes_across_different_split_wrappers_without_mutation():
    train = root("leap")
    dev = root("leap", split="dev")
    edited = make_counterfactual(train, "year", 2900, relation="preserve")
    before = copy.deepcopy([train, edited, dev])
    assert edited.metadata["case_facts_sha256"] == dev.metadata["case_facts_sha256"]
    assert edited.state != dev.state
    with pytest.raises(ValueError, match="aliases leak"):
        audit_counterfactuals([train, edited, dev])
    assert [train, edited, dev] == before


def test_same_facts_with_different_opaque_ids_still_count_toward_component_cap():
    sample = root("leap")
    edited = make_counterfactual(sample, "year", 2000, relation="flip")
    sibling = copy.deepcopy(sample)
    sibling.state = "Record abcdef01." + sibling.state.split(".", 1)[1]
    sibling.metadata["source_example_id"] = "different-case-id"
    assert sibling.state != sample.state
    report = audit_counterfactuals([sample, edited, sibling])
    assert report["components"] == 1
    with pytest.raises(ValueError, match="row cap"):
        audit_counterfactuals([sample, edited, sibling], max_component_rows=2)


def test_cli_anchors_one_parsed_payload_and_refuses_output_overwrite(tmp_path):
    sample = root("leap")
    edited = make_counterfactual(sample, "year", 2000, relation="flip")
    raw = b"".join((json.dumps(s.to_json()) + "\n").encode() for s in [sample, edited])
    source, output = tmp_path / "cohort.jsonl", tmp_path / "audit.json"
    source.write_bytes(raw)
    args = [
        str(source),
        "--expected-sha256",
        hashlib.sha256(raw).hexdigest(),
        "--output",
        str(output),
    ]
    result = main(args)
    assert json.loads(output.read_bytes()) == result
    assert source.read_bytes() == raw
    with pytest.raises(FileExistsError):
        main(args)
    source.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="byte anchor"):
        main([*args[:-1], str(tmp_path / "other.json")])
    assert not (tmp_path / "other.json").exists()


def test_empty_and_original_only_cohorts_cannot_pass():
    with pytest.raises(ValueError, match="empty"):
        audit_counterfactuals([])
    with pytest.raises(ValueError, match="derived"):
        audit_counterfactuals([root("leap")])
