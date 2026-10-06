"""Synthetic native receipts and adverse contracts; no cached model or private corpus."""

import copy
import json
from dataclasses import asdict, replace

import pytest

from ayaka.config import model_config
from ayaka.eval import matched_contract as contract
from ayaka.eval import matched_execution as execution
from ayaka.eval.read_artifact import fingerprint
from ayaka.swift.collect import adapt_jevbench, collect, load_reads
from ayaka.swift.policy import Policy
from ayaka.swift.readers import FakeReader, logmass_probs
from ayaka.tokenization import ToyTokenizer
from scripts.direct_v2.matched_compare import checked_compare


class ReceiptReader(FakeReader):
    backend = "hf"
    logprobs_mode = "raw_logits"
    tokenizer_revision = contract.BACKBONE_REVISION
    chat_template_kwargs = {"enable_thinking": False}

    def read(self, messages, letters):
        result = super().read(messages, letters)
        return replace(result, input_tokens=len(result.input_token_ids))


def resign(row):
    b = row["binding"]
    b["runtime_sha256"] = fingerprint(b["runtime"])
    b["rendered_input_sha256"] = fingerprint(b["messages"])
    b["binding_sha256"] = fingerprint({k: v for k, v in b.items() if k != "binding_sha256"})
    row["binding_sha256"] = b["binding_sha256"]
    for gathered in row["pass_bindings"]:
        gathered["binding_sha256"] = fingerprint(
            {
                "parent_binding_sha256": b["binding_sha256"],
                **{k: v for k, v in gathered.items() if k != "binding_sha256"},
            }
        )
    row["record_sha256"] = fingerprint({k: v for k, v in row.items() if k != "record_sha256"})


@pytest.fixture
def matched(tmp_path):
    items = []
    for kind, labels in (
        ("choice", ["a", "b"]),
        ("noul", ["false", "true"]),
        ("score", ["0", "1"]),
    ):
        items.append(
            adapt_jevbench(
                {
                    "id": kind,
                    "state": f"Evidence for {kind}",
                    "source": "fixture",
                    "tier": "standard",
                    "public": False,
                    "split": "dev",
                    "labels": labels,
                    "expected": labels[0],
                    "question": {
                        "type": kind,
                        "instructions": "Use the evidence.",
                        "criteria": dict.fromkeys(labels, "option"),
                    },
                }
            )
        )
    path = tmp_path / "reads.jsonl"
    collect(
        items,
        ReceiptReader(),
        path,
        model="google/gemma-4-12B-it",
        revision=contract.BACKBONE_REVISION,
        prompt_variant="min",
    )
    v2 = load_reads([path])
    for row in v2:
        row["binding"]["runtime"]["implementation_sha256"] = dict(contract.SWIFT_IMPLEMENTATION)
        resign(row)
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps(asdict(Policy(fitted_on="calibration-fixture"))), encoding="utf-8"
    )
    protocol = contract.make_protocol(
        checkpoint_hashes={
            "ayaka_config.json": "c" * 64,
            "head.safetensors": "a" * 64,
            "adapter/adapter_config.json": "b" * 64,
            "adapter/adapter_model.safetensors": "d" * 64,
        },
        policy_sha256=contract.digest(policy_file.read_bytes()),
        fit_input_hashes={"calibration.jsonl": "e" * 64},
        fit_source_hashes={"fit.py": "f" * 64},
    )
    v1 = []
    for item in items:
        b = contract.item_binding(item)
        v1.append(
            {
                "id": item.id,
                **{
                    k: b[k]
                    for k in (
                        "gold",
                        "gold_distribution",
                        "source",
                        "tier",
                        "public",
                        "case_id",
                        "cluster_id",
                    )
                },
                "type": item.question.type,
                "labels": item.question.labels.copy(),
                "binding": b,
                "run": copy.deepcopy(protocol["v1_run"]),
                "readout": "v1_native_route",
                "raw_probs": dict.fromkeys(item.question.labels, 0.5),
                "baseline_probs": dict.fromkeys(item.question.labels, 0.5),
                "route": "baseline",
            }
        )
    return items, v2, v1, protocol, policy_file


def test_checked_comparison_uses_native_receipts_and_complete_exact_scoring_fields(
    matched, monkeypatch
):
    from scripts.direct_v2 import matched_compare

    monkeypatch.setattr(matched_compare.v1v2_compare, "B", 20)
    items, v2, v1, protocol, path = matched
    receipt = execution_fixture(items, v1, protocol)
    original = copy.deepcopy((v2, v1, protocol))
    report = checked_compare(v2, v1, items, protocol, path.read_bytes(), receipt)
    assert report["validation"]["native_reads_validated"] is True
    assert report["validation"]["fresh_independence_attested"] is False
    assert report["systems"]["v2_off"]["n"] == 3
    assert (v2, v1, protocol) == original


@pytest.mark.parametrize("system", ["v1", "v2"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("gold", "1"),
        ("gold_distribution", {"0": 0.2, "1": 0.8}),
        ("tier", "hard"),
        ("source", "other"),
        ("case_id", "other"),
        ("cluster_id", "other"),
        ("type", "choice"),
        ("labels", ["1", "0"]),
        ("public", True),
    ],
)
def test_top_level_mutation_cannot_score_with_unchanged_nested_binding(
    matched, system, field, value
):
    items, v2, v1, protocol, _ = matched
    rows = v1 if system == "v1" else v2
    rows[-1][field] = value
    if system == "v2":
        resign(rows[-1])
    with pytest.raises(ValueError):
        contract.validate_rows(rows, items, protocol, system=system)


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy", {"gate": "calculation"}),
        ("checkpoint_files_sha256", "9" * 64),
        ("max_seq_len", 4096),
        ("max_new_tokens", 128),
    ],
)
def test_even_uniform_v1_recipe_mutation_is_rejected(matched, field, value):
    items, _, rows, protocol, _ = matched
    for row in rows:
        row["run"][field] = value
    with pytest.raises(ValueError, match="exact frozen"):
        contract.validate_rows(rows, items, protocol, system="v1")


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "nested", "baseline", "mass"])
def test_resume_rows_are_checked_before_model_work(matched, kind):
    items, _, rows, protocol, _ = matched
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "extra":
        rows[0]["id"] = "other"
    elif kind == "nested":
        rows[0]["binding"]["question"]["instruction"] = "altered"
    elif kind == "baseline":
        rows[0]["baseline_probs"] = {"a": float("nan"), "b": 0.5}
    else:
        rows[0]["candidate_log_masses"] = {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        contract.validate_rows(rows, items, protocol, system="v1")
    if kind == "missing":
        assert len(contract.validate_rows(rows, items, protocol, system="v1", complete=False)) == 2


@pytest.mark.parametrize(
    "kind",
    [
        "stale_record",
        "mass",
        "runtime",
        "pass",
        "messages",
        "input_count",
        "output_count",
        "reasoned",
        "thinking",
    ],
)
def test_native_corruption_rejected_even_when_outer_hashes_are_resigned(matched, kind):
    items, rows, _, protocol, _ = matched
    row = rows[0]
    if kind in {"stale_record", "mass"}:
        row["candidate_log_masses"]["a"] = 20.0
    elif kind == "runtime":
        row["binding"]["runtime"]["logprobs_mode"] = "normalized"
    elif kind == "pass":
        row["pass_bindings"][0]["token_logits"]["65"] = 20.0
    elif kind == "messages":
        row["binding"]["messages"][0]["content"] += " SECRET GOLD"
        row["pass_bindings"][0]["messages"] = copy.deepcopy(row["binding"]["messages"])
    elif kind == "input_count":
        row["input_tokens"] += 1
    elif kind == "output_count":
        row["output_tokens"] = 0
    elif kind == "reasoned":
        row["reasoned_read"] = {}
    else:
        row["binding"]["runtime"]["chat_template_kwargs"]["enable_thinking"] = True
    if kind != "stale_record":
        resign(row)
    with pytest.raises(ValueError):
        contract.validate_rows(rows, items, protocol, system="v2")


def test_policy_and_protocol_parse_only_verified_bytes_and_record_fit_anchors(matched, tmp_path):
    _, _, _, protocol, policy_path = matched
    path = tmp_path / "protocol.json"
    data = json.dumps(protocol).encode()
    path.write_bytes(data)
    assert contract.load_protocol(path, contract.digest(data)) == protocol
    assert contract.load_policy(policy_path, protocol).reasoning_route is None
    policy_path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="external anchor"):
        contract.load_policy(policy_path, protocol)
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="external anchor"):
        contract.load_protocol(path, contract.digest(data))


@pytest.mark.parametrize(
    "field",
    [
        "v1_run",
        "cohort_sha256",
        "policy_fit_inputs_sha256",
        "policy_fit_source_sha256",
        "scope",
        "extra",
    ],
)
def test_protocol_cannot_silently_change_frozen_recipe(matched, field):
    protocol = matched[3]
    protocol[field] = {} if field.endswith("sha256") else "altered"
    with pytest.raises(ValueError):
        contract.validate_protocol(protocol)


def test_comparison_api_cannot_substitute_an_unpinned_policy(matched):
    items, v2, v1, protocol, _ = matched
    with pytest.raises(ValueError, match="external anchor"):
        checked_compare(v2, v1, items, protocol, b"{}", {})


def test_exact_policy_contract_distinguishes_boolean_from_float(matched):
    items, _, v1, protocol, _ = matched
    for row in v1:
        row["run"]["policy"]["weight"] = True
    with pytest.raises(ValueError, match="exact frozen"):
        contract.validate_rows(v1, items, protocol, system="v1")


def test_resigned_reversed_token_map_cannot_swap_scored_candidate_coordinates(matched):
    items, rows, _, protocol, _ = matched
    row = rows[0]
    b, gathered = row["binding"], row["pass_bindings"][0]
    inputs = b["token_inputs"][0]
    ids = dict(reversed(list(inputs["canonical_token_ids"].items())))
    inputs["canonical_token_ids"] = copy.deepcopy(ids)
    inputs["canonical_token_ids_sha256"] = fingerprint(ids)
    b["canonical_token_ids_sha256"] = fingerprint([ids])
    gathered["canonical_token_ids"] = copy.deepcopy(ids)
    gathered["token_logits"] = {"65": 4.0, "66": -4.0}
    row["candidate_log_masses"] = {
        label: gathered["token_logits"][str(ids[letter][0])]
        for letter, label in zip(ids, row["labels"], strict=True)
    }
    row["raw_probs"] = logmass_probs(row["candidate_log_masses"])
    resign(row)
    with pytest.raises(ValueError, match="candidate coordinates"):
        contract.validate_rows(rows, items, protocol, system="v2")


def execution_fixture(items, rows, protocol):
    counts = execution.context_preflight(items, ToyTokenizer(), model_config("electra-large"))
    for row in rows:
        c = {
            "version": execution.CONTEXT_VERSION,
            "decisions": [copy.deepcopy(counts[row["id"]]["direct"])],
            "extraction_input_tokens": counts[row["id"]]["extraction_input_tokens"],
        }
        c["sha256"] = fingerprint(c)
        row["checked_context"] = c
    return {
        "version": "ayaka-checked-v1-execution-1",
        "complete": True,
        "preflight_only": False,
        "output_sha256": fingerprint(rows),
        "protocol": protocol,
        "context_preflight": counts,
    }


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "wrong_question",
        "wrong_tokens",
        "clipped_count",
        "wrong_extraction",
        "wrong_rows",
        "preflight_only",
    ],
)
def test_actual_scoring_cannot_bypass_checked_execution_context(matched, damage):
    items, v2, v1, protocol, path = matched
    receipt = execution_fixture(items, v1, protocol)
    c = v1[0]["checked_context"]
    if damage == "missing":
        v1[0].pop("checked_context")
    elif damage == "wrong_question":
        c["decisions"][0]["questions_sha256"] = "a" * 64
    elif damage == "wrong_tokens":
        c["decisions"][0]["input_token_ids_sha256"] = ["b" * 64]
    elif damage == "clipped_count":
        c["decisions"][0]["input_tokens"] = [1]
    elif damage == "wrong_extraction":
        c["extraction_input_tokens"] = 1
    elif damage == "wrong_rows":
        receipt["output_sha256"] = "c" * 64
    else:
        receipt["preflight_only"] = True
    c["sha256"] = fingerprint({k: v for k, v in c.items() if k != "sha256"})
    if damage != "wrong_rows":
        receipt["output_sha256"] = fingerprint(v1)
    with pytest.raises(ValueError):
        checked_compare(v2, v1, items, protocol, path.read_bytes(), receipt)
