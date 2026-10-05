"""Offline native gathers and adversarial prompt-only preparation; no GPU."""

import copy
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, replace

import pytest
import torch
from test_direct_distillation import dataset, verify
from test_evidence_swift_bridge import native_model, tokenizer
from test_teacher_artifacts import IDENTITY, REVISION

from ayaka.config import tiny_config
from ayaka.eval.read_artifact import fingerprint
from ayaka.losses import LossWeights
from ayaka.model.electra import ElectraDecisionModel
from ayaka.swift.collect import collect, load_reads
from ayaka.swift.readers import HFReader, logmass_probs, token_input
from ayaka.tokenization import HFTokenizer
from ayaka.training.direct_distillation import prepare_direct_distillation
from ayaka.training.prepare_v2 import canonical, sha256
from ayaka.training.prompt_teachers import export_prompt_teachers
from ayaka.training.swift_direct import encode_direct_sample
from ayaka.training.teacher_artifacts import teacher_dataset_items
from ayaka.training.trainer import TrainConfig, Trainer

STUDENT = {"encoder": "swift_canonical", "prompt_variant": "min", "state_format": "compact"}
TEACHER = {**STUDENT, "prompt_variant": "cygnet"}


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class PromptReader(HFReader):
    def describe(self, messages, letters):
        return token_input(
            self.tokenizer, messages, letters, {**self.chat_template_kwargs, "return_dict": False}
        )

    def generate_trace(self, *args, **kwargs):
        pytest.fail("prompt-only collection must never generate a rationale")


@pytest.fixture(scope="module")
def observations(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("prompt-observations")
    samples = dataset()["train"]
    samples[0].questions[1].candidates.reverse()
    samples[0].questions[2].candidates.reverse()
    return collect_pair(tmp_path, samples)


def collect_pair(tmp_path, samples):
    tok = HFTokenizer(tokenizer(), "tiny")
    lm, _ = native_model("granite")
    reader = PromptReader("tiny", revision=REVISION, device="cpu", dtype="float32")
    reader.tokenizer, reader.model = tok.hf, lm
    cfg = tiny_config(readout="lm", max_seq_len=2048, serve_max_seq_len=2048)
    rows = []
    for encoding in (STUDENT, TEACHER):
        path = tmp_path / (encoding["prompt_variant"] + ".jsonl")
        collect(
            teacher_dataset_items(samples),
            reader,
            path,
            model="tiny",
            revision=REVISION,
            prompt_variant=encoding["prompt_variant"],
            state_format="compact",
        )
        rows.append(load_reads([path]))
    return samples, tok, cfg, *rows


def exported(values, **overrides):
    samples, tok, cfg, direct, prompt = values
    args = {
        "input_encoding": STUDENT,
        "teacher_input_encoding": TEACHER,
        "teacher_identity": IDENTITY,
        "expected_direct_sha256": fingerprint(direct),
        "expected_prompt_sha256": fingerprint(prompt),
        "verify_gold": verify,
        "mechanics_only": True,
    }
    return export_prompt_teachers(samples, tok, cfg, direct, prompt, **{**args, **overrides})


def test_actual_native_gathers_bind_both_inputs_and_short_student_stays_trace_free(observations):
    before = copy.deepcopy([observations[0], observations[3], observations[4]])
    teachers, report = exported(observations)
    samples, tok, cfg, direct, prompt = observations
    assert report["usage"]["backend_calls"] == 6
    assert report["usage"]["reasoning_tokens"] == 0
    assert report["usage"]["input_tokens"] == sum(r["input_tokens"] for r in [*direct, *prompt])
    assert report["execution_attested"] is report["promotable"] is False
    assert all(row["generated_tokens"] == 0 for row in teachers.values())
    assert teachers["train/known"]["candidate_ids"] == ["false", "true"]
    assert teachers["train/rate"]["candidate_ids"] == ["three", "two", "one"]
    assert "SECRET_TEACHER_TRACE" not in json.dumps(teachers)
    splits = dataset()
    splits["train"] = samples
    items, prepared = prepare_direct_distillation(
        splits, tok, cfg, teachers, verify, input_encoding=STUDENT
    )
    expected_items = encode_direct_sample(samples[0], tok, cfg, input_encoding=STUDENT)
    assert prepared["prompt_context_teacher_questions"] == 3
    assert prepared["reasoning_training_tokens"] == 0
    for item, original, row in zip(items, expected_items, prompt, strict=True):
        assert item.enc == original.enc
        assert item.length < row["input_tokens"]
        assert item.reasoning_labels is item.reasoning_positions is None
    assert [samples, direct, prompt] == before


def resign(row):
    bound = row["binding"]
    bound["runtime_sha256"] = fingerprint(bound["runtime"])
    bound["rendered_input_sha256"] = fingerprint(bound["messages"])
    bound.pop("binding_sha256")
    bound["binding_sha256"] = fingerprint(bound)
    row["binding_sha256"] = bound["binding_sha256"]
    row["rendered_input_sha256"] = bound["rendered_input_sha256"]
    row.pop("record_sha256")
    row["record_sha256"] = fingerprint(row)


@pytest.mark.parametrize(
    "damage", ["state", "question", "lineage", "messages", "runtime", "trace", "usage", "probs"]
)
def test_resigned_saved_reads_cannot_change_content_native_context_or_route(observations, damage):
    values = (*observations[:3], copy.deepcopy(observations[3]), copy.deepcopy(observations[4]))
    row = values[4][0]
    if damage == "state":
        row["binding"]["state"]["answer"] = "y"
    elif damage == "question":
        row["binding"]["question"]["instruction"] += " use the supplied answer"
    elif damage == "lineage":
        row["lineage_ids"] = row["binding"]["lineage_ids"] = ["unrelated"]
    elif damage == "messages":
        row["binding"]["messages"][0]["content"] += " PRIVATE GOLD"
    elif damage == "runtime":
        row["binding"]["runtime"]["dtype"] = "changed"
    elif damage == "trace":
        row["trace_tokens"] = 1
    elif damage == "usage":
        row["input_tokens"] += 1
    else:
        row["raw_probs"] = {label: 1 / len(row["labels"]) for label in row["labels"]}
    resign(row)
    with pytest.raises(ValueError):
        exported(values)


@pytest.mark.parametrize(
    "field",
    [
        "probs",
        "candidate_ids",
        "teacher_readout_binding",
        "route",
        "generated_tokens",
        "trace_sha256",
        "execution_attested",
    ],
)
def test_preparation_reconstructs_exact_envelope_instead_of_trusting_declared_probs(
    observations, field
):
    teachers, _ = exported(observations)
    row = teachers["train/pick"]
    if field in {"probs", "candidate_ids"}:
        row[field].reverse()
    elif field == "teacher_readout_binding":
        row[field]["input_token_ids_sha256"] = "0" * 64
    elif field == "route":
        row[field] = "reasoned"
    elif field == "trace_sha256":
        row[field] = "a" * 64
    else:
        row[field] = 1 if field == "generated_tokens" else True
    splits = dataset()
    splits["train"] = observations[0]
    with pytest.raises(ValueError):
        prepare_direct_distillation(
            splits, observations[1], observations[2], teachers, verify, input_encoding=STUDENT
        )


@pytest.mark.parametrize("field,value", [("generated_tokens", False), ("execution_attested", 0)])
def test_boolean_integer_aliases_cannot_pass_exact_prompt_contract(observations, field, value):
    teachers, _ = exported(observations)
    teachers["train/pick"][field] = value
    splits = dataset()
    splits["train"] = observations[0]
    with pytest.raises(ValueError, match="payload"):
        prepare_direct_distillation(
            splits, observations[1], observations[2], teachers, verify, input_encoding=STUDENT
        )


def test_coupled_version_route_downgrade_cannot_smuggle_a_fake_reasoned_trace(observations):
    from ayaka.training.direct_distillation import VERSION

    teachers, _ = exported(observations)
    teachers["train/pick"].update(
        version=VERSION,
        route="reasoned",
        generated_tokens=1,
        finish_reason="eos",
        trace_sha256="a" * 64,
    )
    splits = dataset()
    splits["train"] = observations[0]
    with pytest.raises(ValueError, match="prompt-only teacher fields"):
        prepare_direct_distillation(
            splits, observations[1], observations[2], teachers, verify, input_encoding=STUDENT
        )


def test_external_anchors_coverage_transport_and_context_are_enforced(observations):
    with pytest.raises(ValueError, match="externally anchored"):
        exported(observations, expected_prompt_sha256="0" * 64)
    with pytest.raises(ValueError, match="identical transport"):
        exported(observations, teacher_input_encoding={**TEACHER, "state_format": "pretty"})
    with pytest.raises(ValueError, match="tiny"):
        exported(observations, mechanics_only=False)
    with pytest.raises(ValueError, match="complete declared"):
        exported((*observations[:4], observations[4][:-1]))
    with pytest.raises(ValueError, match="context"):
        exported(
            (
                observations[0],
                observations[1],
                replace(observations[2], max_seq_len=32, serve_max_seq_len=32),
                *observations[3:],
            )
        )
    short_cap = max(row["input_tokens"] for row in observations[3])
    assert short_cap < min(row["input_tokens"] for row in observations[4])
    # All student rows fit; only the longer teacher exceeds this limit.
    with pytest.raises(ValueError, match="context"):
        exported(
            (
                observations[0],
                observations[1],
                replace(observations[2], max_seq_len=short_cap, serve_max_seq_len=short_cap),
                *observations[3:],
            )
        )
    bad = copy.deepcopy(observations[0])
    bad[0].metadata["split"] = "dev"
    with pytest.raises(ValueError, match="train"):
        exported((bad, *observations[1:]))
    with pytest.raises(ValueError, match="independent verifier"):
        exported(observations, verify_gold=lambda s, q: {})


def fixture_probabilities(row, probabilities):
    """Declared mechanics gathers; never claim these were observed model output."""
    masses = dict(zip(row["labels"], map(math.log, probabilities), strict=True))
    row["candidate_log_masses"] = masses
    row["raw_probs"] = logmass_probs(masses)
    canonical_ids = row["binding"]["token_inputs"][0]["canonical_token_ids"]
    row["pass_bindings"][0]["token_logits"] = {
        str(ids[0]): masses[label]
        for ids, label in zip(canonical_ids.values(), row["labels"], strict=True)
    }
    resign(row)


def test_gold_gain_filter_and_actual_direct_backward_keep_the_gold_and_short_context(observations):
    values = (*observations[:3], copy.deepcopy(observations[3]), copy.deepcopy(observations[4]))
    for direct, prompt, dp, tp in zip(
        values[3],
        values[4],
        ([0.2, 0.8], [0.3, 0.7], [0.6, 0.1, 0.3]),
        ([0.9, 0.1], [0.5, 0.5], [0.05, 0.9, 0.05]),
        strict=True,
    ):
        fixture_probabilities(direct, dp)
        fixture_probabilities(prompt, tp)
    teachers, _ = exported(values)
    splits = dataset()
    splits["train"] = values[0]
    items, report = prepare_direct_distillation(
        splits, values[1], values[2], teachers, verify, input_encoding=STUDENT
    )
    assert report["accepted_teacher_questions"] == 3, [
        (row["id"], row["reasons"]) for row in report["items"]
    ]
    original_teacher = copy.deepcopy(items[0].teacher)
    original_binding = copy.deepcopy(report["items"][0]["teacher_readout_binding"])
    teachers["train/pick"]["probs"][:] = [0.1, 0.9]
    teachers["train/pick"]["teacher_readout_binding"]["messages_sha256"] = "0" * 64
    assert items[0].teacher == original_teacher
    assert report["items"][0]["teacher_readout_binding"] == original_binding
    _, text = native_model("granite")
    model = ElectraDecisionModel(values[2], text, text.config)
    trainer = Trainer(
        model,
        values[1],
        TrainConfig(
            steps=1, bf16=False, loss_weights=LossWeights(gold_nll_with_teacher=True, pointer_aux=0)
        ),
        "cpu",
    )
    loss = trainer.backward_step(items)
    assert math.isfinite(float(loss["total"]))
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert trainer.step_i == 0 and not trainer.opt.state
    # A worse teacher is discarded, retaining independently verified gold.
    fixture_probabilities(values[4][0], [0.1, 0.9])
    teachers, _ = exported(values)
    items, report = prepare_direct_distillation(
        splits, values[1], values[2], teachers, verify, input_encoding=STUDENT
    )
    assert items[0].teacher is None and items[0].target == [1.0, 0.0]
    assert report["gold_replay_questions"] == 1


def test_asymmetric_soft_score_preserves_original_ids_and_ordinal_coordinates(tmp_path):
    sample = dataset()["train"][0]
    sample.questions = [sample.questions[-1]]
    sample.questions[0].candidates.reverse()
    sample.questions[0].instruction = "Report the empirical distribution of the observed levels."
    sample.state["ratings"] = [1, 3, 3, 2, 3]
    sample.questions[0].target_distribution = {"one": 0.2, "two": 0.2, "three": 0.6}
    values = collect_pair(tmp_path, [sample])
    # Wire ordinal order is 1,2,3; original candidate IDs are three,two,one.
    assert values[3][0]["labels"] == ["1", "2", "3"]
    fixture_probabilities(values[3][0], [0.45, 0.3, 0.25])
    fixture_probabilities(values[4][0], [0.1, 0.2, 0.7])

    def fresh_gold(s, q):
        return {
            c.id: s.state["ratings"].count(c.ordinal) / len(s.state["ratings"])
            for c in q.candidates
        }

    teachers, _ = exported(values, verify_gold=fresh_gold)
    teacher = teachers["train/rate"]
    assert teacher["candidate_ids"] == ["three", "two", "one"]
    assert teacher["probs"] == pytest.approx([0.7, 0.2, 0.1])
    assert teacher["direct_probs"] == pytest.approx([0.25, 0.3, 0.45])
    splits = dataset()
    splits["train"] = [sample]
    items, report = prepare_direct_distillation(
        splits,
        values[1],
        values[2],
        teachers,
        lambda s, q: fresh_gold(s, q) if q.type == "score" else verify(s, q),
        input_encoding=STUDENT,
    )
    assert report["accepted_teacher_questions"] == 1
    assert items[0].ordinals == [3, 2, 1]
    assert items[0].target == pytest.approx([0.6, 0.2, 0.2])
    assert items[0].teacher == pytest.approx([0.7, 0.2, 0.1])


@pytest.mark.parametrize("side", [3, 4])
def test_resigned_canonical_map_reorder_rejected_on_each_prompt_side(observations, side):
    values = (*observations[:3], copy.deepcopy(observations[3]), copy.deepcopy(observations[4]))
    row = values[side][0]
    bound = row["binding"]
    inputs = bound["token_inputs"][0]
    inputs["canonical_token_ids"] = dict(reversed(list(inputs["canonical_token_ids"].items())))
    inputs["canonical_token_ids_sha256"] = fingerprint(inputs["canonical_token_ids"])
    bound["canonical_token_ids_sha256"] = fingerprint([inputs["canonical_token_ids"]])
    row["pass_bindings"][0]["canonical_token_ids"] = copy.deepcopy(inputs["canonical_token_ids"])
    fixture_probabilities(row, [0.2, 0.8])
    with pytest.raises(ValueError, match="actual native encoding"):
        exported(values)


def test_prompt_teacher_survives_real_bundle_regeneration_without_opening_holdout(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from test_direct_bundle import dataset as authored_dataset

    from ayaka.data.direct_verification import verify_authored_gold
    from ayaka.training.direct_bundle import audit_bundle, prepare_bundle

    splits = authored_dataset()
    # Observe a declared train subset; unobserved train rows remain gold-only.
    values = collect_pair(tmp_path, [splits["train"][0]])
    teachers, _ = exported(values, verify_gold=verify_authored_gold)
    bundle = tmp_path / "bundle"
    manifest = prepare_bundle(
        bundle,
        splits,
        values[1],
        values[2],
        teachers,
        steps=1,
        rows_per_step=3,
        allow_tiny=True,
        input_encoding=STUDENT,
    )
    anchor = sha256((bundle / "manifest.json").read_bytes())
    holdout = tmp_path / "bundle-holdout/test.jsonl"
    original_read = Path.read_bytes

    def do_not_open_holdout(path):
        if path == holdout:
            pytest.fail("bundle re-audit must use the opaque test commitment")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", do_not_open_holdout)
    actual, recipe, items, _, _ = audit_bundle(
        bundle, tok=values[1], allow_tiny=True, expected_manifest_sha256=anchor
    )
    report = json.loads((bundle / "preparation.json").read_bytes())
    assert actual == manifest and manifest["promotable"] is False
    assert recipe["input_encoding"]["prompt_variant"] == "min"
    assert report["prompt_context_teacher_questions"] == 1
    assert len(items) == 30 and all(item.reasoning_positions is None for item in items)
    assert report["reasoning_training_tokens"] == 0


def test_cli_prompt_aliases_export_only_after_local_validation(tmp_path, observations):
    from ayaka.data.reasoning_v2 import curriculum

    # CLI gold verification uses a generated authored case, never the loose
    # injected verifier in the transport fixture above.
    sample, _ = next(iter(curriculum("train", 1)))
    sample.metadata["source_lineage"] = sample.metadata["case_facts_sha256"]
    tok, cfg = observations[1:3]
    lm, _ = native_model("granite")
    reader = PromptReader("tiny", revision=REVISION, dtype="float32", device="cpu")
    reader.tokenizer, reader.model = tok.hf, lm
    paths = {}
    for name, raw in {
        "train": canonical(sample.to_json()) + b"\n",
        "config": canonical(asdict(cfg)),
        "input-encoding": canonical(STUDENT),
        "teacher-input-encoding": canonical(TEACHER),
        "teacher-identity": canonical(IDENTITY),
    }.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(raw)
    for name, encoding in (("direct-reads", STUDENT), ("prompt-reads", TEACHER)):
        paths[name] = tmp_path / (name + ".jsonl")
        collect(
            teacher_dataset_items([sample]),
            reader,
            paths[name],
            model="tiny",
            revision=REVISION,
            prompt_variant=encoding["prompt_variant"],
            state_format="compact",
        )
    tok.hf.save_pretrained(tmp_path / "tokenizer")
    command = [
        sys.executable,
        "-m",
        "ayaka.training.teacher_artifacts",
        "--teacher-kind",
        "prompt_context",
    ]
    for name, path in paths.items():
        command.extend(["--" + name, str(path)])
    command.extend(
        [
            "--expected-train-sha256",
            sha256(paths["train"].read_bytes()),
            "--expected-direct-sha256",
            fingerprint(load_reads([paths["direct-reads"]])),
            "--expected-prompt-sha256",
            fingerprint(load_reads([paths["prompt-reads"]])),
            "--mechanics-only",
            "--mechanics-tokenizer",
            str(tmp_path / "tokenizer"),
            "--out",
            str(tmp_path / "export"),
        ]
    )
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "empty-cache"),
    }
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "export/receipt.json").read_bytes())
    assert report["usage"]["reasoning_tokens"] == 0
    assert report["model_loaded"] is report["paid_execution_started"] is False
    assert report["optimizer_steps"] == 0 and report["teacher_kind"] == "prompt_context"
    repeated = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
    assert repeated.returncode != 0 and "fresh directory" in repeated.stderr
