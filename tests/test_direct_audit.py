"""Externally audited prepared rows, actual trainer parity and mutation boundaries."""

import copy
import json
import subprocess
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import pytest
import torch
from test_direct_bundle import bundle as legacy_bundle
from test_direct_bundle import dataset, resign
from test_direct_natural import raw_registry
from test_evidence_swift_bridge import tokenizer

from ayaka.checkpoint import apply_lora
from ayaka.config import tiny_config
from ayaka.data.natural_training_v2 import partition_sources
from ayaka.eval.read_artifact import fingerprint
from ayaka.model.electra import ElectraDecisionModel
from ayaka.tokenization import HFTokenizer, ToyTokenizer
from ayaka.training import direct_audit, direct_bundle
from ayaka.training.direct_audit import audit_snapshot, create_audit_receipt, load_audited_bundle
from ayaka.training.direct_bundle import prepare_bundle, training_batches
from ayaka.training.direct_distillation import make_teacher_read
from ayaka.training.prepare_v2 import sha256
from ayaka.training.run_direct import training_config
from ayaka.training.trainer import Trainer


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prepared(tmp_path, swift=False, registry=None):
    root, receipt = tmp_path / "bundle", tmp_path / "cpu-audit.json"
    tok = HFTokenizer(tokenizer(), "tiny") if swift else ToyTokenizer()
    cfg = tiny_config(readout="lm", max_seq_len=2048, lora_dropout=0)
    splits = (
        partition_sources(
            registry.sources(), [], fits=lambda _: True, train_limit=4, heldout_limit=2
        )[0]
        if registry is not None
        else dataset()
    )
    prepare_bundle(
        root,
        splits,
        tok,
        cfg,
        {},
        steps=2,
        rows_per_step=4,
        seed=19,
        allow_tiny=True,
        input_encoding={"encoder": "swift_canonical", "prompt_variant": "labeled"}
        if swift
        else None,
        natural_registry=registry,
    )
    anchor = sha256((root / "manifest.json").read_bytes())
    result = create_audit_receipt(
        root,
        receipt,
        tok,
        expected_manifest_sha256=anchor,
        allow_tiny=True,
        natural_registry=registry,
    )
    kwargs = {
        "expected_manifest_sha256": anchor,
        "expected_receipt_sha256": result["audit_receipt_sha256"],
        "allow_tiny": True,
    }
    return root, receipt, tok, cfg, kwargs


@pytest.mark.parametrize("swift", [False, True])
def test_exact_rows_prefix_aliases_schedule_and_actual_trainer_gradient_parity(
    tmp_path, monkeypatch, swift
):
    root, receipt, tok, cfg, kwargs = prepared(tmp_path, swift)
    full = audit_snapshot(
        root, tok, allow_tiny=True, expected_manifest_sha256=kwargs["expected_manifest_sha256"]
    )

    def forbidden(*args, **kw):
        pytest.fail("frozen load must not regenerate, tokenize, load raw gold or weights")

    monkeypatch.setattr(direct_bundle, "_prepare", forbidden)
    monkeypatch.setattr(direct_bundle, "_gold_verifier", forbidden)
    if swift:
        monkeypatch.setattr(tok.hf, "encode", forbidden)
        monkeypatch.setattr(tok.hf, "apply_chat_template", forbidden)
    cached = load_audited_bundle(root, receipt, tok, **kwargs)
    assert cached.tokenizer is tok and full.tokenizer is tok
    assert direct_bundle._item_bytes(full.items) == direct_bundle._item_bytes(cached.items)
    assert full.inventory == cached.inventory and full.splits == cached.splits
    assert direct_audit._aliases(full.groups) == direct_audit._aliases(cached.groups)
    assert cached.groups[0][0].enc.prefix_ids is not cached.groups[1][0].enc.prefix_ids
    actual = list(training_batches(cached.recipe, cached.inventory, cached.groups))
    expected = list(training_batches(full.recipe, full.inventory, full.groups))
    assert [[asdict(x) for x in b] for b in actual] == [[asdict(x) for x in b] for b in expected]
    model = ElectraDecisionModel.from_config(cfg, dtype=torch.float32)
    model.backbone.requires_grad_(False)
    apply_lora(model)
    trainer = Trainer(
        model, tok, training_config(full.recipe, {"bf16": False, "log_every": 0}), "cpu"
    )
    a, b = full.groups[0], cached.groups[0]
    assert trainer.predict(a) == trainer.predict(b)
    trainer.backward_step(a)
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    trainer.opt.zero_grad(set_to_none=True)
    trainer.backward_step(b)
    assert gradients
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], atol=0, rtol=0)
    assert trainer.step_i == 0 and not trainer.opt.state


@pytest.mark.parametrize("swift", [False, True])
def test_natural_multiquestion_prefix_identity_and_fractional_soft_gold_survive_without_raw_sources(
    tmp_path, monkeypatch, swift
):
    registry = raw_registry()
    root, receipt, tok, _, kwargs = prepared(tmp_path, swift, registry)
    full = audit_snapshot(root, tok, allow_tiny=True, natural_registry=registry)
    monkeypatch.setattr(
        direct_bundle, "_gold_verifier", lambda *a, **k: pytest.fail("no raw sources")
    )
    cached = load_audited_bundle(root, receipt, tok, **kwargs)
    assert direct_bundle._item_bytes(full.items) == direct_bundle._item_bytes(cached.items)
    assert any(len(group) > 1 for group in cached.groups)
    assert any(any(0 < y < 1 for y in item.target) for item in cached.items)
    for a, b in zip(full.groups, cached.groups, strict=True):
        for i in range(len(a)):
            for j in range(len(a)):
                assert (a[i].enc.prefix_ids is a[j].enc.prefix_ids) == (
                    b[i].enc.prefix_ids is b[j].enc.prefix_ids
                )


@pytest.mark.parametrize("field", ["expected_manifest_sha256", "expected_receipt_sha256"])
@pytest.mark.parametrize("damage", [None, "0" * 64, "bad"])
def test_missing_or_wrong_external_anchors_fail_before_architecture(
    tmp_path, monkeypatch, field, damage
):
    root, receipt, tok, _, kwargs = prepared(tmp_path)
    kwargs[field] = damage
    monkeypatch.setattr(
        direct_bundle, "inspect_direct_model", lambda *a, **k: pytest.fail("early reject")
    )
    with pytest.raises(ValueError, match="pinned|digest|anchor|unbound"):
        load_audited_bundle(root, receipt, tok, **kwargs)


@pytest.mark.parametrize(
    "name",
    [
        "train.jsonl",
        "dev.jsonl",
        "teacher_reads.json",
        "train_items.jsonl",
        "test_commitment.json",
        "recipe.json",
    ],
)
@pytest.mark.parametrize("resigned", [False, True])
def test_changed_payloads_cannot_reuse_external_audit_even_with_new_local_checksums(
    tmp_path, name, resigned
):
    root, receipt, tok, _, kwargs = prepared(tmp_path)
    raw = (root / name).read_bytes() + b" "
    if resigned:
        resign(root, name, raw)
    else:
        (root / name).write_bytes(raw)
    with pytest.raises(ValueError, match="anchor"):
        load_audited_bundle(root, receipt, tok, **kwargs)


@pytest.mark.parametrize(
    "drift", ["source", "dependencies", "architecture", "tokenizer", "configuration", "survey"]
)
def test_current_execution_environment_drift_cannot_reuse_cpu_receipt(tmp_path, monkeypatch, drift):
    root, receipt, tok, _, kwargs = prepared(tmp_path, swift=True)
    if drift == "source":
        monkeypatch.setattr(direct_bundle, "_source_hashes", lambda: {"different.py": "f" * 64})
    elif drift == "dependencies":
        monkeypatch.setattr(direct_audit, "_dependencies", lambda: {"transformers": "different"})
    elif drift == "architecture":
        monkeypatch.setattr(
            direct_bundle, "inspect_direct_model", lambda *a, **k: {"changed": True}
        )
    elif drift == "tokenizer":
        tok.hf.add_tokens(["changed-vocabulary"])
    elif drift == "configuration":
        tok.hf.clean_up_tokenization_spaces = not tok.hf.clean_up_tokenization_spaces
    else:
        monkeypatch.setattr(direct_bundle, "_model_policy", lambda *a, **k: {"changed": "survey"})
    with pytest.raises(ValueError, match="stale|architecture|tokenizer|policy"):
        load_audited_bundle(root, receipt, tok, **kwargs)


@pytest.mark.parametrize("drift", ["payload", "source", "tokenizer", "dependencies"])
def test_mutation_during_regeneration_prevents_receipt_publication(tmp_path, monkeypatch, drift):
    root, _, tok, _, kwargs = prepared(tmp_path, swift=True)
    original = direct_bundle.audit_bundle

    def changed(*args, **kw):
        result = original(*args, **kw)
        if drift == "payload":
            (root / "dev.jsonl").write_bytes(b"replacement")
        elif drift == "source":
            monkeypatch.setattr(direct_bundle, "_source_hashes", lambda: {"replacement": "f" * 64})
        elif drift == "tokenizer":
            tok.hf.split_special_tokens = not tok.hf.split_special_tokens
        else:
            monkeypatch.setattr(direct_audit, "_dependencies", lambda: {"replacement": "1"})
        return result

    monkeypatch.setattr(direct_bundle, "audit_bundle", changed)
    out = tmp_path / "absent.json"
    with pytest.raises(ValueError, match="changed|anchor"):
        create_audit_receipt(
            root,
            out,
            tok,
            expected_manifest_sha256=kwargs["expected_manifest_sha256"],
            allow_tiny=True,
        )
    assert not out.exists()


def test_exact_decimal_ordinals_without_pickle_or_numeric_rounding(tmp_path):
    root, receipt, tok, _, kwargs = prepared(tmp_path)
    frozen = load_audited_bundle(root, receipt, tok, **kwargs)
    item = copy.deepcopy(frozen.items[0])
    item.ordinals[0] = Decimal("1.2500000000000000001")
    raw = direct_bundle._item_bytes([item])
    restored = direct_audit._items(raw)[0]
    assert restored.ordinals[0] == item.ordinals[0]
    assert isinstance(restored.ordinals[0], Decimal)


def test_verified_teacher_distribution_is_retained_but_trace_input_stays_absent(tmp_path):
    sample = dataset()["train"][0]
    q = sample.questions[0]
    probs = [
        0.9 if q.target_distribution[c.id] else 0.1 / (len(q.candidates) - 1) for c in q.candidates
    ]
    teacher = make_teacher_read(
        sample,
        q,
        probs=probs,
        direct_probs=[1 / len(probs)] * len(probs),
        model_sha256="a" * 64,
        recipe_sha256="b" * 64,
        trace_sha256=fingerprint("completed mechanics trace"),
        generated_tokens=12,
    )
    root, receipt = tmp_path / "bundle", tmp_path / "audit.json"
    legacy_bundle(root, teachers={sample.metadata["source_example_id"] + "/" + q.id: teacher})
    anchor = sha256((root / "manifest.json").read_bytes())
    result = create_audit_receipt(root, receipt, expected_manifest_sha256=anchor, allow_tiny=True)
    cached = load_audited_bundle(
        root,
        receipt,
        expected_manifest_sha256=anchor,
        expected_receipt_sha256=result["audit_receipt_sha256"],
        allow_tiny=True,
    )
    assert cached.items[0].teacher == probs
    assert cached.items[0].reasoning_labels is None and cached.items[0].base_probs is None


def test_holdout_originals_never_opened_and_new_test_file_refused(tmp_path, monkeypatch):
    root, receipt, tok, _, kwargs = prepared(tmp_path)
    original = Path.read_bytes

    def protected(path):
        if path.name == "test.jsonl":
            pytest.fail("original holdout must not be opened")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", protected)
    load_audited_bundle(root, receipt, tok, **kwargs)
    (root / "test.jsonl").write_text("original-test", encoding="utf-8")
    with pytest.raises(ValueError, match="original test"):
        load_audited_bundle(root, receipt, tok, **kwargs)


@pytest.mark.parametrize("where", ["bundle", "holdout", "existing"])
def test_receipt_requires_new_external_output(tmp_path, where):
    root, receipt, tok, _, kwargs = prepared(tmp_path)
    destination = {
        "bundle": root / "new.json",
        "holdout": root.with_name("bundle-holdout") / "new.json",
        "existing": receipt,
    }[where]
    with pytest.raises(ValueError, match="new file outside"):
        create_audit_receipt(
            root,
            destination,
            tok,
            expected_manifest_sha256=kwargs["expected_manifest_sha256"],
            allow_tiny=True,
        )


def test_actual_offline_cli_receipt_load(tmp_path):
    root, _, tok, _, kwargs = prepared(tmp_path)
    out = tmp_path / "cli-audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ayaka.training.direct_audit",
            "--bundle",
            str(root),
            "--out",
            str(out),
            "--expected-manifest-sha256",
            kwargs["expected_manifest_sha256"],
            "--mechanics-only",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    kwargs["expected_receipt_sha256"] = printed["audit_receipt_sha256"]
    cached = load_audited_bundle(root, out, tok, **kwargs)
    assert cached.binding["regeneration_complete"] and not cached.binding["execution_attested"]
