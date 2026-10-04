"""Opt-in native-input bundle -> teacher -> train/calibrate/dev/export binding."""

import copy
import json
from dataclasses import asdict, replace

import pytest
import torch
from test_direct_bundle import dataset, resign
from test_direct_distillation import dataset as teacher_dataset
from test_direct_distillation import reads, verify
from test_direct_swift_inputs import sample
from test_evidence_swift_bridge import tokenizer

from ayaka.config import tiny_config
from ayaka.data.schema import Sample
from ayaka.eval.read_artifact import fingerprint
from ayaka.tokenization import HFTokenizer
from ayaka.training import direct_bundle, run_direct
from ayaka.training.direct_bundle import audit_bundle, prepare_bundle, training_batches
from ayaka.training.direct_distillation import make_teacher_read, prepare_direct_distillation
from ayaka.training.frozen_replay import attach_base_replay
from ayaka.training.prepare_v2 import canonical, sha256
from ayaka.training.swift_direct import (
    direct_readout_binding,
    encode_direct_sample,
    normalize_input_encoding,
)

ENCODING = {"encoder": "swift_canonical", "prompt_variant": "labeled", "state_format": "compact"}


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prepared(root, *, steps=2):
    tok = HFTokenizer(tokenizer(), "tiny")
    cfg = tiny_config(readout="lm", max_seq_len=2048, lora_dropout=0.05)
    manifest = prepare_bundle(
        root,
        dataset(),
        tok,
        cfg,
        {},
        steps=steps,
        rows_per_step=3,
        seed=19,
        allow_tiny=True,
        input_encoding=ENCODING,
    )
    return tok, cfg, manifest


def test_swift_bundle_roundtrip_binds_all_splits_and_original_score_meaning(tmp_path):
    root = tmp_path / "bundle"
    tok, cfg, manifest = prepared(root)
    anchor = sha256((root / "manifest.json").read_bytes())
    actual, recipe, items, inventory, groups = audit_bundle(
        root, tok=tok, allow_tiny=True, expected_manifest_sha256=anchor
    )
    assert actual == manifest and recipe["version"] == "ayaka-direct-bundle-5"
    assert recipe["input_encoding"] == normalize_input_encoding(ENCODING)
    assert all(item.direct_input_binding["recipe"] == recipe["input_recipe"] for item in items)
    assert not (root / "test.jsonl").exists()
    assert (tmp_path / "bundle-holdout/test.jsonl").is_file()
    report = json.loads((root / "preparation.json").read_bytes())
    assert report["input_encoding"] == recipe["input_encoding"]
    assert set(report["context_audit"]) == {"train", "router_train", "dev", "calibration", "test"}
    scheduled = list(training_batches(recipe, inventory, groups))
    assert len(scheduled) == 2 and all(len(batch) == 3 for batch in scheduled)
    for original, group in zip(dataset()["train"], groups, strict=True):
        expected = encode_direct_sample(original, tok, cfg, input_encoding=ENCODING)
        assert [asdict(item) for item in group] == [asdict(item) for item in expected]


@pytest.mark.parametrize("damage", ["settings", "recipe", "template", "old_version"])
def test_recipe_and_tokenizer_changes_rejected_even_after_file_checksums_are_resigned(
    tmp_path, damage
):
    root = tmp_path / "bundle"
    tok, _, _ = prepared(root)
    recipe = json.loads((root / "recipe.json").read_bytes())
    if damage == "settings":
        recipe["input_encoding"]["prompt_variant"] = "rules"
    elif damage == "recipe":
        recipe["input_recipe"]["tokenizer_sha256"] = "f" * 64
    elif damage == "template":
        tok.hf.chat_template += "different-prefix"
    else:
        recipe["version"] = "ayaka-direct-bundle-4"
    resign(root, "recipe.json", canonical(recipe) + b"\n")
    with pytest.raises(ValueError, match="recipe|tokenizer|chat template"):
        audit_bundle(root, tok=tok, allow_tiny=True)


def test_native_input_teacher_direct_observation_must_match_exact_prompt(tmp_path):
    splits = teacher_dataset()
    tok, cfg = HFTokenizer(tokenizer(), "tiny"), tiny_config(readout="lm", max_seq_len=2048)
    unbound_reads = reads(splits)
    with pytest.raises(ValueError, match="exact Swift input/readout binding"):
        prepare_direct_distillation(
            splits, tok, cfg, unbound_reads, verify, input_encoding=ENCODING
        )
    teacher_sample = splits["train"][0]
    encoded = encode_direct_sample(teacher_sample, tok, cfg, input_encoding=ENCODING)
    bound_reads = {}
    for q, item in zip(teacher_sample.questions, encoded, strict=True):
        old = unbound_reads["train/" + q.id]
        bound_reads["train/" + q.id] = make_teacher_read(
            teacher_sample,
            q,
            probs=old["probs"],
            direct_probs=old["direct_probs"],
            model_sha256=old["model_sha256"],
            recipe_sha256=old["recipe_sha256"],
            trace_sha256=old["trace_sha256"],
            generated_tokens=old["generated_tokens"],
            direct_readout=direct_readout_binding(item),
        )
    items, report = prepare_direct_distillation(
        splits, tok, cfg, bound_reads, verify, input_encoding=ENCODING
    )
    assert report["accepted_teacher_questions"] == 3
    assert [item.target for item in items] == [item.target for item in encoded]
    assert [item.teacher for item in items] == [
        unbound_reads["train/" + q.id]["probs"] for q in teacher_sample.questions
    ]
    damaged = copy.deepcopy(bound_reads)
    damaged["train/pick"]["direct_readout_binding"]["canonical_token_ids_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="exact Swift input/readout binding"):
        prepare_direct_distillation(splits, tok, cfg, damaged, verify, input_encoding=ENCODING)


def test_fixed_schedule_swift_training_calibration_dev_export_and_resume(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    tok, _, _ = prepared(root)
    monkeypatch.setattr(direct_bundle, "local_tokenizer", lambda *a, **kw: tok)
    monkeypatch.setattr(run_direct, "local_tokenizer", lambda *a, **kw: tok)
    training = {"bf16": False, "log_every": 0, "micro_batch_tokens": 8192}
    result = run_direct.run_pipeline(
        root,
        tmp_path / "full",
        action="train",
        mechanics_only=True,
        training=training,
        checkpoint_every=1,
    )
    assert result["optimizer_steps"] == 2 and result["reload_probability_parity"]
    assert result["input_encoding"] == normalize_input_encoding(ENCODING)
    assert result["input_recipe_sha256"] == fingerprint(result["input_recipe"])
    assert result["independent_test_required"] and not result["promotable"]
    for name in ("calibration_raw", "dev"):
        evaluation = json.loads((tmp_path / "full" / f"{name}.json").read_bytes())
        assert all(
            row["direct_readout_binding"] is not None and row["reasoning_tokens"] == 0
            for row in evaluation["rows"]
        )
    resumed = run_direct.run_pipeline(
        root,
        tmp_path / "resumed",
        action="train",
        mechanics_only=True,
        training=training,
        resume=tmp_path / "full/state-00000001",
        checkpoint_every=1,
    )
    assert (
        resumed["reload_probability_parity"]
        and resumed["input_encoding"] == result["input_encoding"]
    )
    from safetensors.torch import load_file

    full = load_file(tmp_path / "full/state-00000002/trainable.safetensors")
    restored = load_file(tmp_path / "resumed/state-00000002/trainable.safetensors")
    assert full.keys() == restored.keys()
    for name in full:
        torch.testing.assert_close(full[name], restored[name], atol=0, rtol=0)


def test_swift_frozen_replay_records_canonical_tokens_and_refuses_changed_recipe():
    from test_frozen_replay import setup

    trainer, sources, _ = setup()
    trainer.tok = HFTokenizer(tokenizer(), "tiny")
    groups = [
        encode_direct_sample(src, trainer.tok, trainer.model.cfg, input_encoding=ENCODING)
        for src in sources
    ]
    saved = attach_base_replay(trainer, sources, groups, "a" * 64)
    assert all(row["direct_readout_binding"] is not None for row in saved["header"]["schema"])
    changed = [
        encode_direct_sample(
            src,
            trainer.tok,
            trainer.model.cfg,
            input_encoding={"encoder": "swift_canonical", "prompt_variant": "rules"},
        )
        for src in sources
    ]
    with pytest.raises(ValueError, match="another native model/input/split/order"):
        attach_base_replay(trainer, sources, changed, "a" * 64, saved=saved)


def test_serving_context_limit_preserves_long_original_inputs_and_refuses_overflow():
    from test_direct_swift_inputs import setup

    tok, _, trainer, _, _ = setup("gemma")
    trainer.model.cfg = replace(trainer.model.cfg, max_seq_len=64, serve_max_seq_len=1024)
    src = sample()
    src.questions = src.questions[:1]
    src.metadata.update(split="dev", source_lineage="same-document", language="en")
    result = run_direct.evaluate_direct(trainer, [src], "dev", input_encoding=ENCODING)
    assert result["rows"][0]["tokens"] > 64
    direct = encode_direct_sample(
        Sample(src.state, src.questions, {"source_example_id": "test-fixture"}),
        tok,
        trainer.model.cfg,
        input_encoding=ENCODING,
        context_limit=1024,
    )[0]
    assert result["rows"][0]["direct_readout_binding"] == direct_readout_binding(direct)
    trainer.model.cfg = replace(trainer.model.cfg, serve_max_seq_len=64)
    with pytest.raises(ValueError, match="context"):
        run_direct.evaluate_direct(trainer, [src], "dev", input_encoding=ENCODING)


@pytest.mark.parametrize(
    "value",
    [
        {"encoder": "invalid"},
        {"encoder": "ayaka_segmented", "prompt_variant": "min"},
        {"encoder": "swift_canonical", "state_format": "invalid"},
        {"encoder": "swift_canonical", "chat_template_kwargs": {"enable_thinking": True}},
        {"encoder": "swift_canonical", "chat_template_kwargs": {"return_tensors": "pt"}},
    ],
)
def test_invalid_input_settings_fail_explicitly(value):
    with pytest.raises(ValueError):
        normalize_input_encoding(value)


def test_cli_prepares_and_audits_the_explicit_swift_encoder_without_loading_weights(
    tmp_path, monkeypatch, capsys
):
    tok = HFTokenizer(tokenizer(), "tiny")
    monkeypatch.setattr(direct_bundle, "local_tokenizer", lambda *a, **kw: tok)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for split, sources in dataset().items():
        (corpus / f"{split}.jsonl").write_bytes(
            b"\n".join(canonical(src.to_json()) for src in sources) + b"\n"
        )
    config = tmp_path / "config.json"
    config.write_bytes(canonical(asdict(tiny_config(readout="lm", max_seq_len=2048))))
    root = tmp_path / "bundle"
    args = [
        "prepare",
        "--corpus",
        str(corpus),
        "--config",
        str(config),
        "--out",
        str(root),
        "--steps",
        "1",
        "--rows-per-step",
        "3",
        "--mechanics-only",
        "--prompt-variant",
        "labeled",
        "--state-format",
        "compact",
    ]
    with pytest.raises(ValueError, match="encoder|encoding"):
        direct_bundle.main(args)
    assert not root.exists()
    direct_bundle.main([*args, "--input-encoder", "swift_canonical"])
    anchor = sha256((root / "manifest.json").read_bytes())
    direct_bundle.main(
        ["audit", "--bundle", str(root), "--mechanics-only", "--expected-manifest-sha256", anchor]
    )
    recipe = json.loads((root / "recipe.json").read_bytes())
    assert recipe["input_encoding"] == normalize_input_encoding(ENCODING)
    reports = capsys.readouterr().out
    assert reports.count('"optimizer_steps_executed": 0') == 2
    assert reports.count('"paid_execution_started": false') == 2
