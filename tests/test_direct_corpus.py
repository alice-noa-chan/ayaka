import copy
import json
from dataclasses import asdict

import pytest
from test_direct_corpus_plan import planned_inputs

from ayaka.data.reasoning_v2 import SPLITS
from ayaka.eval.read_artifact import fingerprint
from ayaka.input_errors import ContextLimitError
from ayaka.training import direct_audit, direct_bundle
from ayaka.training import direct_corpus as module
from ayaka.training.prepare_v2 import canonical, sha256
from ayaka.training.swift_direct import normalize_input_encoding


def inputs():
    _, cfg, tok, registry, original = planned_inputs()
    reserved = module.ReservedInventory(draft=True)
    encoding = normalize_input_encoding(None)
    plan = module.create_plan(original["settings"], cfg, tok, registry, encoding, reserved)
    return plan, cfg, tok, registry, encoding, reserved


def private_manifest(tmp_path, samples):
    source = tmp_path / "previous-evaluation.jsonl"
    source.write_bytes(b"\n".join(canonical(s.to_json()) for s in samples) + b"\n")
    manifest = tmp_path / "private-manifest.json"
    value = {
        "version": module.RESERVED_VERSION,
        "files": [{"path": source.name, "sha256": sha256(source.read_bytes())}],
    }
    manifest.write_bytes(canonical(value) + b"\n")
    return source, manifest, sha256(manifest.read_bytes())


def test_actual_plan_prepare_cli_uses_cached_only_inputs_and_publishes_portable_receipt(
    tmp_path, monkeypatch, capsys
):
    plan, cfg, _, registry, _, _ = inputs()
    config = tmp_path / "config.json"
    config.write_bytes(canonical(asdict(cfg)))
    settings = tmp_path / "settings.json"
    settings.write_bytes(canonical(plan["settings"]))
    monkeypatch.setattr(module, "NaturalGoldRegistry", lambda: registry)
    plan_path = tmp_path / "input-plan.json"
    common = [
        "--config",
        str(config),
        "--mechanics-only",
        "--draft-without-prior-reserved",
        "--input-encoder",
        "ayaka_segmented",
    ]
    module.main(["plan", *common, "--settings", str(settings), "--out", str(plan_path)])
    declared = json.loads(plan_path.read_bytes())
    assert "raw_verified_questions" not in declared
    output = tmp_path / "prepared"
    module.main(
        [
            "prepare",
            *common,
            "--plan",
            str(plan_path),
            "--expected-plan-file-sha256",
            sha256(plan_path.read_bytes()),
            "--out",
            str(output),
        ]
    )
    result = json.loads((output / "selection.json").read_bytes())
    assert result["raw_verified_questions"] == {
        "helpsteer2": 800,
        "commonsense_qa": 160,
        "massive_ko": 320,
        "massive_ja": 320,
    }
    assert result["corpus_contract"]["whole_epochs"]["rows"] == 40
    assert (
        result["corpus_contract"]["plan"]["assets"]["reserved"]["scope"]
        == "draft_without_prior_private_inventory"
    )
    monkeypatch.setattr(
        "ayaka.data.direct_natural.local_raw_sources",
        lambda: pytest.fail("portable load read raw data"),
    )
    loaded = direct_audit.load_audited_bundle(
        output / "bundle",
        output / "cpu-audit.json",
        allow_tiny=True,
        expected_manifest_sha256=result["bundle_manifest_sha256"],
        expected_receipt_sha256=result["audit_receipt_sha256"],
    )
    assert len(loaded.items) == 40
    assert (
        len(list(direct_bundle.training_batches(loaded.recipe, loaded.inventory, loaded.groups)))
        == 10
    )
    assert result["model_weights_loaded"] is False and result["paid_execution_started"] is False
    assert "plan_file_sha256" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["missing", "wrong"])
def test_cli_requires_exact_plan_file_anchor_before_loading_any_tokenizer(
    tmp_path, monkeypatch, bad
):
    plan, cfg, _, _, _, _ = inputs()
    config, plan_path = tmp_path / "config.json", tmp_path / "plan.json"
    config.write_bytes(canonical(asdict(cfg)))
    plan_path.write_bytes(canonical(plan) + b"\n")
    monkeypatch.setattr(
        module, "local_tokenizer", lambda *a, **k: pytest.fail("must validate plan first")
    )
    args = [
        "prepare",
        "--config",
        str(config),
        "--plan",
        str(plan_path),
        "--out",
        str(tmp_path / "absent"),
        "--draft-without-prior-reserved",
    ]
    if bad == "wrong":
        args += ["--expected-plan-file-sha256", "0" * 64]
    with pytest.raises((ValueError, SystemExit)):
        module.main(args)


def test_private_manifest_requires_external_anchor_and_binding_excludes_paths(tmp_path):
    plan, _, _, registry, _, _ = inputs()
    sample = registry.sample("commonsense_qa", 9)
    source, manifest, anchor = private_manifest(tmp_path, [sample])
    inventory = module.ReservedInventory(manifest, anchor)
    assert inventory.binding["files"] == {sha256(source.read_bytes()): 1}
    assert str(tmp_path) not in json.dumps(inventory.binding)
    moved_manifest = tmp_path / "same-files-new-manifest.json"
    value = json.loads(manifest.read_bytes())
    value["files"][0]["path"] = str(source.resolve())
    moved_manifest.write_bytes(canonical(value))
    relocated = module.ReservedInventory(moved_manifest, sha256(moved_manifest.read_bytes()))
    assert relocated.binding == inventory.binding
    assert plan["assets"]["reserved"]["scope"].startswith("draft")
    for kwargs in (
        {},
        {"path": manifest},
        {"path": manifest, "expected_sha256": "0" * 64},
        {"path": manifest, "expected_sha256": anchor, "draft": True},
    ):
        with pytest.raises(ValueError):
            module.ReservedInventory(**kwargs)


@pytest.mark.parametrize("damage", ["bytes", "list_addition", "guard_removal", "sample", "binding"])
def test_reserved_inventory_exit_catches_changed_list_bytes_and_memory(tmp_path, damage):
    _, _, _, registry, _, _ = inputs()
    source, manifest, anchor = private_manifest(tmp_path, [registry.sample("commonsense_qa", 9)])
    inventory = module.ReservedInventory(manifest, anchor)
    if damage == "bytes":
        source.write_bytes(source.read_bytes() + b"\n")
    elif damage == "list_addition":
        value = json.loads(manifest.read_bytes())
        value["files"].append(value["files"][0])
        manifest.write_bytes(canonical(value))
    elif damage == "guard_removal":
        inventory.files.clear()
    elif damage == "sample":
        inventory.samples[0].state = "changed original input"
    else:
        inventory.binding["scope"] = "changed"
    with pytest.raises(ValueError):
        inventory.verify()


def test_context_selection_uses_final_permutation_and_only_catches_typed_overflow(monkeypatch):
    plan, cfg, tok, registry, encoding, reserved = inputs()
    first, _ = module.select_corpus(plan, cfg, tok, registry, encoding, reserved)
    excluded = next(s for s in first["train"] if s.metadata["source"] == "commonsense_qa")
    seen = []
    original = module.encode_direct_sample

    def checked(sample, *args, **kwargs):
        assert sample.metadata[module.MARKER] == module.plan_sha256(plan)
        if sample.metadata["source"] == "commonsense_qa":
            raw = registry.sample("commonsense_qa", sample.metadata["raw_row_index"])
            expected = module.permuted_sample(raw, plan)
            assert [c.id for c in sample.questions[0].candidates] == [
                c.id for c in expected.questions[0].candidates
            ]
            seen.append(sample.metadata["source_example_id"])
        if sample.metadata["source_example_id"] == excluded.metadata["source_example_id"]:
            raise ContextLimitError("only the complete shuffled input overflows")
        return original(sample, *args, **kwargs)

    monkeypatch.setattr(module, "encode_direct_sample", checked)
    selected, report = module.select_corpus(plan, cfg, tok, registry, encoding, reserved)
    assert excluded.metadata["source_example_id"] not in {
        s.metadata["source_example_id"] for rows in selected.values() for s in rows
    }
    assert report["selection"]["removed"]["context_overflow_whole_sample"] == 1
    assert len(seen) >= len(SPLITS)
    monkeypatch.setattr(
        module,
        "encode_direct_sample",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("malformed native input")),
    )
    with pytest.raises(ValueError, match="malformed native"):
        module.select_corpus(plan, cfg, tok, registry, encoding, reserved)


@pytest.mark.parametrize(
    "damage", ["public_addition", "private_bytes", "private_list", "input_bytes"]
)
def test_late_changes_prevent_bundle_publication(tmp_path, monkeypatch, damage):
    plan, cfg, tok, registry, encoding, _ = inputs()
    source, manifest, anchor = private_manifest(tmp_path, [registry.sample("commonsense_qa", 9)])
    reserved = module.ReservedInventory(manifest, anchor)
    plan = module.create_plan(plan["settings"], cfg, tok, registry, encoding, reserved)
    input_file = tmp_path / "input.json"
    input_file.write_bytes(canonical(plan))
    input_guard = module.FileGuard(input_file, sha256(input_file.read_bytes()))
    original, public = direct_bundle._prepare, module.public_inventory()

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if damage == "public_addition":
            monkeypatch.setattr(
                module,
                "public_inventory",
                lambda: {**public, "new.jsonl": fingerprint("new benchmark")},
            )
        elif damage == "private_bytes":
            source.write_bytes(source.read_bytes() + b"\n")
        elif damage == "private_list":
            manifest.write_bytes(manifest.read_bytes() + b"\n")
        else:
            input_file.write_bytes(input_file.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(direct_bundle, "_prepare", mutate)
    root = tmp_path / "absent"
    with pytest.raises(ValueError):
        module.prepare_corpus(
            root,
            plan,
            cfg,
            tok,
            registry,
            encoding,
            reserved,
            allow_tiny=True,
            input_guards=(input_guard,),
        )
    assert not root.exists()


def test_private_change_during_full_audit_prevents_receipt_publication(tmp_path, monkeypatch):
    plan, cfg, tok, registry, encoding, _ = inputs()
    source, manifest, anchor = private_manifest(tmp_path, [registry.sample("commonsense_qa", 9)])
    reserved = module.ReservedInventory(manifest, anchor)
    plan = module.create_plan(plan["settings"], cfg, tok, registry, encoding, reserved)
    original = direct_audit.audit_snapshot

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(direct_audit, "audit_snapshot", mutate)
    root = tmp_path / "prepared"
    with pytest.raises(ValueError, match="changed before publication"):
        module.prepare_corpus(root, plan, cfg, tok, registry, encoding, reserved, allow_tiny=True)
    assert (root / "bundle/manifest.json").exists()
    assert not (root / "cpu-audit.json").exists() and not (root / "selection.json").exists()


def test_authored_reserved_overlap_and_false_binding_cannot_be_approved(tmp_path):
    plan, cfg, tok, registry, encoding, reserved = inputs()
    splits, _ = module.select_corpus(plan, cfg, tok, registry, encoding, reserved)
    _, manifest, anchor = private_manifest(tmp_path, [splits["train"][0]])
    reserved = module.ReservedInventory(manifest, anchor)
    plan = module.create_plan(plan["settings"], cfg, tok, registry, encoding, reserved)
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="overlaps reserved/public"):
        module.prepare_corpus(root, plan, cfg, tok, registry, encoding, reserved, allow_tiny=True)
    assert not root.exists()
    changed = copy.deepcopy(plan)
    changed["assets"]["reserved"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="inventory changed"):
        module.prepare_corpus(
            root, changed, cfg, tok, registry, encoding, reserved, allow_tiny=True
        )


def fast_inputs():
    from test_evidence_swift_bridge import tokenizer

    from ayaka.tokenization import HFTokenizer

    old_plan, cfg, _, registry, encoding, reserved = inputs()
    tok = HFTokenizer(tokenizer(), cfg.backbone)
    encoding = normalize_input_encoding(
        {"encoder": "swift_canonical", "prompt_variant": "labeled", "state_format": "compact"}
    )
    plan = module.create_plan(old_plan["settings"], cfg, tok, registry, encoding, reserved)
    return plan, cfg, tok, registry, encoding, reserved


def test_tokenizer_drift_after_selection_cannot_publish_any_bundle_or_receipt(
    tmp_path, monkeypatch
):
    plan, cfg, tok, registry, encoding, reserved = fast_inputs()
    original = module.select_corpus

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        tok.hf.backend_tokenizer.add_tokens(["scope-drift-token"])
        return result

    monkeypatch.setattr(module, "select_corpus", mutate)
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="tokenizer changed"):
        module.prepare_corpus(root, plan, cfg, tok, registry, encoding, reserved, allow_tiny=True)
    assert not root.exists()


@pytest.mark.parametrize("boundary", ["create", "publish"])
def test_input_plan_checks_actual_tokenizer_at_exit_and_before_cli_write(
    tmp_path, monkeypatch, boundary
):
    plan, cfg, tok, registry, encoding, reserved = fast_inputs()
    if boundary == "create":
        original, calls = registry.verify_files, []

        def mutate():
            original()
            calls.append(1)
            if len(calls) == 2:
                tok.hf.backend_tokenizer.add_tokens(["plan-drift-token"])

        monkeypatch.setattr(registry, "verify_files", mutate)
        with pytest.raises(ValueError, match="actual model/tokenizer"):
            module.create_plan(plan["settings"], cfg, tok, registry, encoding, reserved)
    else:
        config, settings = tmp_path / "config.json", tmp_path / "settings.json"
        config.write_bytes(canonical(asdict(cfg)))
        settings.write_bytes(canonical(plan["settings"]))
        monkeypatch.setattr(module, "local_tokenizer", lambda *a, **k: tok)
        monkeypatch.setattr(module, "NaturalGoldRegistry", lambda: registry)
        original = module.create_plan

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            tok.hf.backend_tokenizer.add_tokens(["publication-drift-token"])
            return result

        monkeypatch.setattr(module, "create_plan", mutate)
        out = tmp_path / "absent.json"
        with pytest.raises(ValueError, match="actual model/tokenizer"):
            module.main(
                [
                    "plan",
                    "--config",
                    str(config),
                    "--settings",
                    str(settings),
                    "--out",
                    str(out),
                    "--mechanics-only",
                    "--draft-without-prior-reserved",
                    "--input-encoder",
                    "ayaka_segmented",
                ]
            )
        assert not out.exists()
