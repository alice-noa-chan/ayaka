"""Temp bytes, toy tokenizer and fake route; never load published weights."""

import json
import os
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_matched_contract import execution_fixture
from test_matched_contract import matched as matched

from ayaka.config import model_config
from ayaka.eval import matched_contract as contract
from ayaka.eval import matched_execution as execution
from ayaka.primitives import QuestionSpec
from ayaka.tokenization import ToyTokenizer
from scripts.direct_v2 import matched_v1


@pytest.fixture
def assets(matched, tmp_path, monkeypatch):
    items, _, _, protocol, _ = matched
    root = tmp_path / "checkpoint"
    root.mkdir()
    files = {
        "electra_config.json": json.dumps(asdict(model_config("electra-large"))).encode(),
        "meta.json": b"{}",
        "head.safetensors": b"synthetic head bytes only",
        "adapter/adapter_config.json": json.dumps(
            {
                "base_model_name_or_path": "google/gemma-4-12B-it",
                "revision": contract.BACKBONE_REVISION,
                "r": 64,
            }
        ).encode(),
        "adapter/adapter_model.safetensors": b"synthetic adapter bytes only",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    protocol = contract.make_protocol(
        checkpoint_hashes={k: contract.digest(v) for k, v in files.items()},
        policy_sha256=protocol["policy_sha256"],
        fit_input_hashes=protocol["policy_fit_inputs_sha256"],
        fit_source_hashes=protocol["policy_fit_source_sha256"],
    )
    receipt = {
        "checkpoint_repo": contract.CHECKPOINT_REPO,
        "checkpoint_revision": contract.CHECKPOINT_REVISION,
        "base_revision": contract.BACKBONE_REVISION,
        "checkpoint_path": str(root),
        "source_sha256": protocol["checkpoint_source_sha256"],
    }
    p = tmp_path / "protocol.json"
    data = json.dumps(protocol).encode()
    p.write_bytes(data)
    ck = tmp_path / "checkpoint.json"
    ck.write_text(json.dumps(receipt), encoding="utf-8")
    args = SimpleNamespace(
        protocol=p,
        protocol_sha256=contract.digest(data),
        procedural=tmp_path / "p.jsonl",
        hard_calibration=tmp_path / "c.jsonl",
        hard_dev=tmp_path / "d.jsonl",
        checkpoint_receipt=ck,
        output=tmp_path / "v1.jsonl",
        receipt=tmp_path / "execution.json",
        device="cpu",
        preflight_only=True,
    )
    monkeypatch.setattr(matched_v1, "cohort_items", lambda *a: items)
    return items, protocol, receipt, root, args


def test_snapshot_consumes_copied_verified_config_head_adapter_and_meta(assets):
    _, protocol, receipt, root, _ = assets
    with execution.checkpoint_snapshot(receipt, protocol) as (snapshot, cfg):
        assert cfg.readout == "hybrid"
        assert snapshot != root
        assert set(protocol["checkpoint_source_sha256"]) == execution.consumed_files(snapshot)
        assert all(
            not (snapshot / name).is_symlink() for name in protocol["checkpoint_source_sha256"]
        )
        assert (snapshot / "head.safetensors").read_bytes() == b"synthetic head bytes only"
    assert not snapshot.exists()


@pytest.mark.parametrize(
    "damage", ["head", "adapter", "meta", "preferred", "remove", "extra", "receipt", "traversal"]
)
def test_checkpoint_changes_fail_before_loader_is_called(assets, damage):
    _, protocol, receipt, root, _ = assets
    if damage in {"head", "adapter", "meta"}:
        target = {
            "head": "head.safetensors",
            "adapter": "adapter/adapter_model.safetensors",
            "meta": "meta.json",
        }[damage]
        (root / target).write_bytes(b"changed")
    elif damage == "preferred":
        (root / "ayaka_config.json").write_bytes((root / "electra_config.json").read_bytes())
    elif damage == "remove":
        (root / "head.safetensors").unlink()
    elif damage == "extra":
        (root / "adapter/extra.json").write_bytes(b"{}")
    elif damage == "receipt":
        receipt["source_sha256"] = {"head.safetensors": "a" * 64}
    else:
        hashes = dict(protocol["checkpoint_source_sha256"])
        hashes["../escape"] = "a" * 64
        protocol = contract.make_protocol(
            checkpoint_hashes=hashes,
            policy_sha256=protocol["policy_sha256"],
            fit_input_hashes=protocol["policy_fit_inputs_sha256"],
            fit_source_hashes=protocol["policy_fit_source_sha256"],
        )
        receipt["source_sha256"] = hashes
    with pytest.raises(ValueError), execution.checkpoint_snapshot(receipt, protocol):
        pytest.fail("loader boundary reached")


@pytest.mark.parametrize("which", ["snapshot", "original", "new_preferred"])
def test_mutation_during_execution_prevents_success_receipt(assets, which):
    _, protocol, receipt, root, _ = assets
    with (
        pytest.raises(ValueError),
        execution.checkpoint_snapshot(receipt, protocol) as (snapshot, _),
    ):
        if which == "new_preferred":
            (root / "ayaka_config.json").write_bytes((root / "electra_config.json").read_bytes())
        else:
            target = snapshot if which == "snapshot" else root
            (target / "head.safetensors").write_bytes(b"changed during run")


def test_offline_preflight_never_calls_model_factory_or_writes_model_rows(assets):
    items, _, _, _, args = assets

    def forbidden(*a):
        pytest.fail("model loader called by preflight")

    report = matched_v1.execute(
        args, tokenizer_factory=lambda cfg: ToyTokenizer(), model_factory=forbidden
    )
    assert report["preflight_only"] is True and report["complete"] is False
    assert set(report["context_preflight"]) == {item.id for item in items}
    assert args.receipt.is_file() and not args.output.exists()


def test_existing_bad_rows_are_rejected_before_tokenizer_or_model_load(assets):
    _, _, _, _, args = assets
    args.output.write_text(json.dumps({"id": "other"}) + "\n", encoding="utf-8")

    def forbidden(*a):
        pytest.fail("tokenizer/model boundary reached")

    with pytest.raises(ValueError, match="frozen cohort"):
        matched_v1.execute(args, tokenizer_factory=forbidden, model_factory=forbidden)
    assert not args.receipt.exists()


def test_original_overflow_is_rejected_before_model_load(assets, monkeypatch):
    items, _, _, _, args = assets
    items = [replace(items[0], state="x" * 9000)]
    monkeypatch.setattr(matched_v1, "cohort_items", lambda *a: items)
    args.preflight_only = False
    with pytest.raises(ValueError, match="refuse truncation"):
        matched_v1.execute(
            args,
            tokenizer_factory=lambda cfg: ToyTokenizer(),
            model_factory=lambda *a: pytest.fail("model loaded"),
        )
    assert not args.output.exists() and not args.receipt.exists()


def test_extraction_budget_has_its_own_context_limit(assets, monkeypatch):
    items, _, _, _, args = assets
    monkeypatch.setattr(execution, "chat_ids", lambda *a: [1] * (12288 - 383))
    with pytest.raises(ValueError, match="reserve frozen"):
        matched_v1.execute(
            args,
            tokenizer_factory=lambda cfg: ToyTokenizer(),
            model_factory=lambda *a: pytest.fail("model loaded"),
        )
    assert items and not args.receipt.exists()


class Inner:
    tok = ToyTokenizer()
    max_seq_len = 8192
    max_labels = 26

    def __init__(self):
        self.calls = 0

    def decide(self, state, specs, **kw):
        self.calls += 1
        return [SimpleNamespace(probs=[0.5, 0.5], extras={"evidence": {"route": "baseline"}})]


class Route:
    def __init__(self, overflow=False):
        self.original = Inner()
        self.overflow = overflow

    def decide(self, state, specs, **kw):
        result = self.original.decide(state, specs, **kw)
        if self.overflow:
            return self.original.decide(state + "x" * 9000, specs, **kw)
        return result


def test_row_publication_and_resume_require_checked_context(assets):
    items, protocol, _, _, args = assets
    tok = ToyTokenizer()
    counts = execution.context_preflight(items, tok, execution.checkpoint_config(assets[3]))
    route = Route()
    assert matched_v1.checked_run(route, items, args.output, protocol, counts) == 3
    rows = matched_v1.load_reads([args.output])
    execution.validate_context_rows(rows)
    assert matched_v1.checked_run(route, items, args.output, protocol, counts) == 0
    rows[0].pop("checked_context")
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="checked full-context"):
        matched_v1.checked_run(route, items, args.output, protocol, counts)


def test_actual_worked_steps_overflow_never_reaches_second_model_forward(assets):
    items, protocol, _, _, args = assets
    counts = execution.context_preflight(
        items, ToyTokenizer(), execution.checkpoint_config(assets[3])
    )
    route = Route(overflow=True)
    inner = route.original
    with pytest.raises(ValueError, match="refuse truncation"):
        matched_v1.checked_run(route, [items[0]], args.output, protocol, counts)
    assert inner.calls == 1
    assert args.output.read_bytes() == b""


def test_full_input_check_uses_untruncated_tokenization_at_exact_boundary():
    tok = ToyTokenizer()
    specs = [QuestionSpec("choice", "q", ["one", "two"])]
    counted = execution.full_decision_context("state", specs, tok)
    (n,) = counted["input_tokens"]
    assert execution.full_decision_context("state", specs, tok, limit=n)["input_tokens"] == [n]
    with pytest.raises(ValueError, match="refuse truncation"):
        execution.full_decision_context("state", specs, tok, limit=n - 1)


@pytest.mark.parametrize(
    "field",
    ["questions_sha256", "input_token_ids_sha256", "input_tokens", "extraction_input_tokens"],
)
def test_resigned_resume_context_must_match_fresh_tokenizer_preflight(assets, field):
    items, protocol, _, _, args = assets
    rows = []
    for item in items:
        b = contract.item_binding(item)
        rows.append(
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
                "labels": item.question.labels,
                "binding": b,
                "run": protocol["v1_run"],
                "readout": "v1_native_route",
                "raw_probs": dict.fromkeys(item.question.labels, 0.5),
                "baseline_probs": dict.fromkeys(item.question.labels, 0.5),
                "route": "baseline",
            }
        )
    execution_fixture(items, rows, protocol)
    c = rows[0]["checked_context"]
    if field == "extraction_input_tokens":
        c[field] = 1
    else:
        c["decisions"][0][field] = (
            [1]
            if field == "input_tokens"
            else ["a" * 64]
            if field == "input_token_ids_sha256"
            else "a" * 64
        )
    c["sha256"] = contract.fingerprint({k: v for k, v in c.items() if k != "sha256"})
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    args.preflight_only = False
    with pytest.raises(ValueError):
        matched_v1.execute(
            args,
            tokenizer_factory=lambda cfg: ToyTokenizer(),
            model_factory=lambda *a: pytest.fail("model loaded"),
        )
    assert not args.receipt.exists()


@pytest.mark.parametrize("destination", ["output", "receipt"])
def test_output_paths_cannot_publish_inside_consumed_checkpoint(assets, destination):
    _, _, _, root, args = assets
    setattr(args, destination, root / "adapter/execution.json")
    with pytest.raises(ValueError, match="must not modify checkpoint"):
        matched_v1.execute(args, tokenizer_factory=lambda cfg: pytest.fail("tokenizer loaded"))
    assert not (root / "adapter/execution.json").exists()


def test_existing_output_hardlink_cannot_mutate_consumed_checkpoint(assets):
    _, _, _, root, args = assets
    source = root / "head.safetensors"
    os.link(source, args.output)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="must not modify checkpoint"):
        matched_v1.execute(args, tokenizer_factory=lambda cfg: pytest.fail("tokenizer loaded"))
    assert source.read_bytes() == before


def test_complete_fake_execution_receipt_is_required_by_real_scoring_entrypoint(
    assets, matched, monkeypatch
):
    import ayaka.evidence_pipeline
    from scripts.direct_v2.matched_compare import checked_compare, v1v2_compare

    items, protocol, _, root, args = assets
    args.preflight_only = False
    loaded = []

    def fake_loader(snapshot, cfg):
        assert snapshot != root
        execution.verify_checkpoint_files(snapshot, protocol["checkpoint_source_sha256"])
        loaded.append(snapshot)
        return object()

    monkeypatch.setattr(ayaka.evidence_pipeline, "reasoning_decision", lambda *a: Route())
    monkeypatch.setattr(v1v2_compare, "B", 20)
    report = matched_v1.execute(
        args, tokenizer_factory=lambda cfg: ToyTokenizer(), model_factory=fake_loader
    )
    assert len(loaded) == 1 and not loaded[0].exists()
    assert report["complete"] is True
    rows = matched_v1.load_reads([args.output])
    comparison = checked_compare(matched[1], rows, items, protocol, matched[4].read_bytes(), report)
    assert comparison["systems"]["v1_on"]["n"] == 3
