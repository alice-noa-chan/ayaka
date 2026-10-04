import copy
import json
from dataclasses import asdict

import pytest

from ayaka.config import ELECTRA_LARGE, tiny_config
from ayaka.data.reasoning_v2 import SPLITS, curriculum
from ayaka.eval.read_artifact import fingerprint
from ayaka.tokenization import ToyTokenizer
from ayaka.training import direct_bundle
from ayaka.training.direct_bundle import audit_bundle, main, prepare_bundle, training_batches
from ayaka.training.direct_distillation import make_teacher_read
from ayaka.training.optimization import OptimizationConfig
from ayaka.training.prepare_v2 import canonical, sha256


def dataset():
    result = {}
    for split in SPLITS:
        result[split] = []
        for sample, traces in curriculum(split, 10):
            sample.metadata.update(
                source_lineage=sample.metadata["case_facts_sha256"],
                modality="text",
                verified_traces=traces,
            )
            result[split].append(sample)
    return result


def bundle(root, *, teachers=None):
    return prepare_bundle(
        root,
        dataset(),
        ToyTokenizer(),
        tiny_config(readout="lm", max_seq_len=2048),
        teachers or {},
        steps=2,
        rows_per_step=16,
        allow_tiny=True,
    )


def resign(root, name, raw):
    (root / name).write_bytes(raw)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["files"][name] = sha256(raw)
    (root / "manifest.json").write_bytes(canonical(manifest))


def test_bundle_roundtrip_finite_schedule_and_deterministic_resume(tmp_path):
    original = bundle(tmp_path / "bundle")
    manifest, recipe, items, inventory, groups = audit_bundle(tmp_path / "bundle", allow_tiny=True)
    assert original == manifest
    assert manifest["promotable"] is manifest["execution_attested"] is False
    assert len(items) == 30 and all(it.direct_distillation for it in items)
    full = list(training_batches(recipe, inventory, groups))
    resumed = list(training_batches(recipe, inventory, groups, start_step=1))
    assert len(full) == 2 and all(len(batch) == 16 for batch in full)
    assert resumed == full[1:]
    assert list(training_batches(recipe, inventory, groups, start_step=2)) == []
    report = json.loads((tmp_path / "bundle/preparation.json").read_bytes())
    assert report["workload"]["total_rows"] == 32
    assert report["workload"]["unique_prepared_rows"] == 30
    assert report["workload"]["counts"]["route"] == {"direct": 32}
    assert report["workload"]["tokens"]["trace_tokens"] == 0
    assert set(report["context_audit"]) == set(SPLITS)
    assert recipe["native_architecture"]["weights_loaded"] is False
    assert recipe["optimizations"]["attention"] == "native"
    assert recipe["native_architecture"]["conservative_total_parameters"] < 14_000_000_000
    with pytest.raises(ValueError, match="resume step"):
        list(training_batches(recipe, inventory, groups, start_step=3))


def test_liger_topology_is_prepared_and_audited_without_importing_external_kernels(
    tmp_path, monkeypatch
):
    from ayaka.training import optimization

    monkeypatch.setattr(optimization, "_liger_functions", lambda: pytest.fail("no kernel import"))
    prepare_bundle(
        tmp_path / "liger",
        dataset(),
        ToyTokenizer(),
        tiny_config(readout="lm", max_seq_len=2048),
        {},
        steps=2,
        rows_per_step=16,
        allow_tiny=True,
        optimizations=OptimizationConfig(liger=True),
    )
    _, recipe, _, _, _ = audit_bundle(tmp_path / "liger", allow_tiny=True)
    plan = recipe["native_architecture"]["kernel_plan"]
    assert recipe["optimizations"]["liger"] is True
    assert len(plan["liger_modules"]["geglu"]) == 6
    assert plan["execution_verified"] is False
    recipe["optimizations"]["liger"] = False
    resign(tmp_path / "liger", "recipe.json", canonical(recipe) + b"\n")
    with pytest.raises(ValueError, match="actual native architecture"):
        audit_bundle(tmp_path / "liger", allow_tiny=True)


def test_unsupported_flash_architecture_fails_before_creating_bundle(tmp_path):
    with pytest.raises(ValueError, match="unified causal text"):
        prepare_bundle(
            tmp_path / "unsupported",
            dataset(),
            ToyTokenizer(),
            tiny_config(readout="lm"),
            {},
            steps=2,
            rows_per_step=16,
            allow_tiny=True,
            optimizations=OptimizationConfig(attention="flash_attention_2"),
        )
    assert not (tmp_path / "unsupported").exists()


def test_accepted_teacher_tokens_and_soft_target_alignment_survive_full_regeneration(tmp_path):
    sample = dataset()["train"][0]
    q = sample.questions[0]
    target = [q.target_distribution[c.id] for c in q.candidates]
    p = [0.9 if y else 0.1 / (len(target) - 1) for y in target]
    d = [1 / len(target)] * len(target)
    teacher = {
        sample.metadata["source_example_id"] + "/" + q.id: make_teacher_read(
            sample,
            q,
            probs=p,
            direct_probs=d,
            model_sha256="a" * 64,
            recipe_sha256="b" * 64,
            trace_sha256=fingerprint("a completed trace"),
            generated_tokens=12,
        )
    }
    bundle(tmp_path / "bundle", teachers=teacher)
    _, _, items, _, _ = audit_bundle(tmp_path / "bundle", allow_tiny=True)
    assert items[0].teacher == pytest.approx(p)
    assert items[0].target == target
    assert items[0].reasoning_labels is None


@pytest.mark.parametrize(
    "file", ["train.jsonl", "teacher_reads.json", "train_items.jsonl", "recipe.json"]
)
def test_rejects_checksum_changes_before_any_tokenizer_load(tmp_path, monkeypatch, file):
    root = tmp_path / "bundle"
    bundle(root)
    (root / file).write_bytes(b"corrupt")
    monkeypatch.setattr(
        direct_bundle, "local_tokenizer", lambda *a, **k: pytest.fail("must fail before tokenizer")
    )
    with pytest.raises(ValueError, match="checksum"):
        audit_bundle(root, allow_tiny=True)


@pytest.mark.parametrize("file", ["train_items.jsonl", "preparation.json"])
def test_resigned_saved_token_or_report_tampering_fails_regeneration(tmp_path, file):
    root = tmp_path / "bundle"
    bundle(root)
    if file == "train_items.jsonl":
        rows = [json.loads(line) for line in (root / file).read_bytes().splitlines()]
        rows[0]["enc"]["prefix_ids"][0] += 1
        raw = b"\n".join(canonical(row) for row in rows) + b"\n"
    else:
        report = json.loads((root / file).read_bytes())
        report["workload"]["tokens"]["text_tokens"] += 1
        raw = canonical(report) + b"\n"
    resign(root, file, raw)
    with pytest.raises(ValueError, match="regenerated"):
        audit_bundle(root, allow_tiny=True)


def test_rejects_path_injection_source_changes_and_tokenizer_changes(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    bundle(root)
    with pytest.raises(ValueError, match="tokenizer"):
        audit_bundle(root, ToyTokenizer(vocab_size=1024), allow_tiny=True)
    source = direct_bundle._source_hashes()
    monkeypatch.setattr(direct_bundle, "_source_hashes", lambda: {**source, "changed.py": "a" * 64})
    with pytest.raises(ValueError, match="source snapshot"):
        audit_bundle(root, allow_tiny=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["files"]["../../outside.json"] = "a" * 64
    (root / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError, match="exactly"):
        audit_bundle(root, allow_tiny=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("inference_mode", "on"),
        ("initialization", "failed_pilot"),
        ("reasoning_training_tokens", True),
        ("schedule", {"steps": 2, "rows_per_step": 16, "seed": True}),
        ("loss_weights", {"gold_nll_with_teacher": False, "pointer_aux": 0}),
    ],
)
def test_resigned_recipe_cannot_disable_direct_contract(tmp_path, field, value):
    root = tmp_path / "bundle"
    bundle(root)
    recipe = json.loads((root / "recipe.json").read_bytes())
    recipe[field] = value
    resign(root, "recipe.json", canonical(recipe) + b"\n")
    with pytest.raises(ValueError):
        audit_bundle(root, allow_tiny=True)


def test_checks_holdout_gold_and_no_overwrite_before_creating_directory(tmp_path):
    splits = dataset()
    bad = copy.deepcopy(splits)
    q = bad["test"][0].questions[0]
    q.target_distribution = {c.id: 1 / len(q.candidates) for c in q.candidates}
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="reserved gold"):
        prepare_bundle(
            out,
            bad,
            ToyTokenizer(),
            tiny_config(readout="lm", max_seq_len=2048),
            {},
            steps=2,
            rows_per_step=16,
            allow_tiny=True,
        )
    assert not out.exists()
    root = tmp_path / "bundle"
    before = bundle(root)
    with pytest.raises(ValueError, match="new directory"):
        bundle(root)
    assert json.loads((root / "manifest.json").read_bytes()) == before


def test_real_models_require_pinned_approved_native_readout_before_tokenizer(tmp_path):
    from dataclasses import replace

    for cfg in (
        ELECTRA_LARGE,
        replace(ELECTRA_LARGE, readout="lm", backbone_revision="main"),
        replace(ELECTRA_LARGE, readout="lm", backbone="unknown/model"),
    ):
        with pytest.raises(ValueError):
            prepare_bundle(tmp_path / "invalid", {}, None, cfg, {}, steps=1, rows_per_step=16)
    assert not (tmp_path / "invalid").exists()


def test_cli_prepare_and_audit_are_cpu_only_and_tiny_is_explicit(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for split, samples in dataset().items():
        (corpus / f"{split}.jsonl").write_bytes(
            b"\n".join(canonical(s.to_json()) for s in samples) + b"\n"
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
        "2",
        "--rows-per-step",
        "16",
    ]
    with pytest.raises(ValueError, match="mechanics-only"):
        main(args)
    main([*args, "--mechanics-only"])
    main(["audit", "--bundle", str(root), "--mechanics-only"])
    reports = capsys.readouterr().out
    assert reports.count('"paid_execution_started": false') == 2
    assert reports.count('"optimizer_steps_executed": 0') == 2


def test_resigned_architecture_and_native_vocabulary_mismatch_are_rejected(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    bundle(root)
    recipe = json.loads((root / "recipe.json").read_bytes())
    recipe["native_architecture"]["conservative_total_parameters"] += 1
    resign(root, "recipe.json", canonical(recipe) + b"\n")
    with pytest.raises(ValueError, match="actual native architecture"):
        audit_bundle(root, allow_tiny=True)
    real = direct_bundle.inspect_direct_model

    def small_vocabulary(*a, **k):
        report = real(*a, **k)
        report["native_output_shape"][0] = 1
        return report

    monkeypatch.setattr(direct_bundle, "inspect_direct_model", small_vocabulary)
    with pytest.raises(ValueError, match="vocabulary"):
        bundle(tmp_path / "bad_vocabulary")
    assert not (tmp_path / "bad_vocabulary").exists()
