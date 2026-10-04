"""Actual offline HF raw gathers with deterministic trace fixtures; no GPU."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch
from test_direct_distillation import dataset, verify
from test_evidence_swift_bridge import native_model, tokenizer

from ayaka.config import tiny_config
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.eval.read_artifact import fingerprint
from ayaka.swift.collect import collect, load_reads
from ayaka.swift.readers import HFReader, logmass_probs, token_input
from ayaka.swift.reasoning import TraceResult, reasoned_read
from ayaka.swift.router import eligible
from ayaka.tokenization import HFTokenizer
from ayaka.training.direct_distillation import prepare_direct_distillation
from ayaka.training.prepare_v2 import canonical, sha256
from ayaka.training.teacher_artifacts import export_swift_teachers, teacher_dataset_items

REVISION = "a" * 40
ENCODING = {"encoder": "swift_canonical", "prompt_variant": "labeled", "state_format": "compact"}
IDENTITY = {
    "model": "tiny",
    "revision": REVISION,
    "tokenizer_revision": REVISION,
    "native_weights_sha256": "b" * 64,
    "adapter_sha256": None,
}
TRACE = "PRIVATE STEPS: apply the rule and its exception."


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class TraceHFReader(HFReader):
    """Generation is a fixture; final probabilities come from a real tiny LM."""

    def __init__(self, tok, lm, *, empty=False, capped=False):
        super().__init__("tiny", revision=REVISION, dtype="float32", device="cpu")
        self.tokenizer, self.model = tok, lm
        self.empty, self.capped, self.trace_calls = empty, capped, []

    def describe(self, messages, letters):
        # Existing Swift transport bug is owned by Claude. Preserve actual IDs
        # with the already validated Transformers-5 envelope workaround.
        return token_input(
            self.tokenizer, messages, letters, {**self.chat_template_kwargs, "return_dict": False}
        )

    def generate_trace(self, messages, max_tokens):
        self.trace_calls.append((copy.deepcopy(messages), max_tokens))
        text = "" if self.empty else "x" * max_tokens if self.capped else TRACE
        count = (
            max_tokens
            if self.capped
            else len(self.tokenizer.encode(text, add_special_tokens=False)) + 1
        )
        return TraceResult(
            text,
            len(self.describe(messages, ["A"])["input_token_ids"]),
            count,
            "length" if self.capped else "eos",
            0.001,
        )


def observations(tmp_path, *, samples=None, empty=False, capped=False, budget=128, confident=False):
    if samples is None:
        samples = dataset()["train"]
        samples[0].questions[1].candidates.reverse()  # canonical Noul remapping
        samples[0].questions[2].candidates.reverse()  # sorted wire ordinal != original IDs/order
    tok = HFTokenizer(tokenizer(), "tiny")
    lm, _ = native_model("granite")
    if confident:
        with torch.no_grad():
            lm.lm_head.bias[tok.hf.encode("A", add_special_tokens=False)[0]] = 40
            lm.lm_head.bias[tok.hf.encode("B", add_special_tokens=False)[0]] = -40
    reader = TraceHFReader(tok.hf, lm, empty=empty, capped=capped)
    cfg = tiny_config(readout="lm", max_seq_len=2048, serve_max_seq_len=2048)
    items = teacher_dataset_items(samples)
    path = tmp_path / "direct.jsonl"
    collect(
        items,
        reader,
        path,
        model="tiny",
        revision=REVISION,
        prompt_variant="labeled",
        state_format="compact",
    )
    direct = load_reads([path])
    paired = []
    for item, row in zip(items, direct, strict=True):
        result, metadata = reasoned_read(
            reader,
            item.state,
            item.question,
            prompt_variant="labeled",
            state_format="compact",
            max_tokens=budget,
        )
        nested = {
            **asdict(result),
            **metadata,
            "direct_record_sha256": row["record_sha256"],
            "direct_binding_sha256": row["binding_sha256"],
        }
        retained = {key: value for key, value in row.items() if key != "record_sha256"}
        retained["reasoned_read"] = nested
        retained["record_sha256"] = fingerprint(retained)
        paired.append(retained)
    return samples, tok, cfg, direct, paired, reader


def exported(values, *, gold_verifier=verify, **overrides):
    samples, tok, cfg, direct, paired, _ = values
    return export_swift_teachers(
        samples,
        tok,
        cfg,
        direct,
        paired,
        input_encoding=ENCODING,
        teacher_identity=IDENTITY,
        expected_direct_sha256=fingerprint(direct),
        expected_paired_sha256=fingerprint(paired),
        verify_gold=gold_verifier,
        mechanics_only=True,
        **overrides,
    )


def test_actual_native_gather_conversion_preserves_original_soft_gold_and_usage(tmp_path):
    values = observations(tmp_path)
    originals = copy.deepcopy(values[:5])
    teachers, report = exported(values)
    samples, tok, cfg, direct, paired, reader = values
    assert len(teachers) == report["available_teacher_questions"] == 3
    assert report["usage"]["backend_calls"] == 9
    assert report["usage"]["reasoning_tokens"] == sum(
        row["reasoned_read"]["trace_tokens"] for row in paired
    )
    assert report["usage"]["input_tokens"] == sum(
        a["input_tokens"] + b["reasoned_read"]["input_tokens"]
        for a, b in zip(direct, paired, strict=True)
    )
    assert report["usage"]["output_tokens"] == sum(
        a["output_tokens"] + b["reasoned_read"]["output_tokens"]
        for a, b in zip(direct, paired, strict=True)
    )
    assert "PRIVATE STEPS" not in json.dumps([teachers, report])
    assert (
        report["mechanics_only"] and report["promotable"] is report["execution_attested"] is False
    )
    score = teachers["train/rate"]
    assert score["candidate_ids"] == ["three", "two", "one"]
    assert score["probs"] == [paired[2]["reasoned_read"]["raw_probs"][str(i)] for i in (3, 2, 1)]
    assert teachers["train/known"]["candidate_ids"] == ["false", "true"]
    splits = dataset()
    splits["train"] = samples
    # Use the same original Noul/Score order when comparing the teacher binding.
    items, prepared = prepare_direct_distillation(
        splits, tok, cfg, teachers, verify, input_encoding=ENCODING
    )
    assert items and prepared["reasoning_training_tokens"] == 0
    assert len(reader.trace_calls) == 3
    assert samples == originals[0] and direct == originals[3] and paired == originals[4]


@pytest.mark.parametrize("kind", ["empty", "capped"])
def test_excluded_traces_keep_all_observed_generated_tokens(tmp_path, kind):
    values = observations(tmp_path, empty=kind == "empty", capped=kind == "capped")
    teachers, report = exported(values)
    assert teachers == {} and report["available_teacher_questions"] == 0
    assert {row["status"] for row in report["observations"]} == (
        {"empty_trace"} if kind == "empty" else {"incomplete_trace"}
    )
    assert report["usage"]["reasoning_tokens"] == (3 if kind == "empty" else 3 * 128)
    assert report["usage"]["backend_calls"] == 9


def test_forced_high_confident_no_number_train_observation_is_not_auto_filtered(tmp_path):
    src = Sample(
        "Rule: approve each request.",
        [
            Question(
                "q",
                "choice",
                "Apply the rule.",
                [Candidate("approve", "approve"), Candidate("reject", "reject")],
                {"approve": 1.0},
            )
        ],
        {"split": "train", "source_example_id": "plain", "source_lineage": "plain-case"},
    )
    values = observations(tmp_path, samples=[src], confident=True, budget=1024)
    assert not eligible(values[3][0])
    teachers, report = exported(values, gold_verifier=lambda s, q: {"approve": 1.0, "reject": 0.0})
    assert len(teachers) == 1 and report["max_reasoning_tokens"] == 1024


def resign(values):
    # Re-sign internally consistent files so semantic/native checks, rather
    # than trivial record corruption, exercise the relevant trust boundaries.
    direct, paired = values[3], values[4]
    for a, b in zip(direct, paired, strict=True):
        a["binding"]["runtime_sha256"] = fingerprint(a["binding"]["runtime"])
        a["binding"].pop("binding_sha256")
        a["binding"]["binding_sha256"] = fingerprint(a["binding"])
        a["binding_sha256"] = a["binding"]["binding_sha256"]
        a.pop("record_sha256")
        a["record_sha256"] = fingerprint(a)
        nested = b["reasoned_read"]
        b.clear()
        b.update({key: copy.deepcopy(value) for key, value in a.items() if key != "record_sha256"})
        nested["direct_record_sha256"], nested["direct_binding_sha256"] = (
            a["record_sha256"],
            a["binding_sha256"],
        )
        nested["recipe_sha256"] = fingerprint(nested["recipe"])
        b["reasoned_read"] = nested
        b["record_sha256"] = fingerprint(b)


@pytest.mark.parametrize(
    "damage",
    [
        "state",
        "question",
        "gold",
        "model",
        "adapter",
        "lineage",
        "trace",
        "final_tokens",
        "usage",
        "budget",
    ],
)
def test_semantic_native_usage_and_identity_tampering_rejected_after_resigning(tmp_path, damage):
    values = observations(tmp_path)
    row, paired = values[3][0], values[4][0]
    bound = row["binding"]
    if damage == "state":
        bound["state"] = {"different": "evidence"}
    elif damage == "question":
        bound["question"]["instruction"] += " changed"
    elif damage == "gold":
        bound["gold"] = row["gold"] = "y"
    elif damage == "model":
        bound["model"] = row["model"] = "different-native"
    elif damage == "adapter":
        bound["runtime"]["adapter_sha256"] = "c" * 64
    elif damage == "lineage":
        bound["case_id"] = row["case_id"] = "different-case"
    elif damage == "trace":
        paired["reasoned_read"]["pass_inputs"][0]["messages"][1]["content"] = "different trace"
    elif damage == "final_tokens":
        paired["reasoned_read"]["pass_inputs"][0]["canonical_token_ids"]["A"][0] += 10
    elif damage == "usage":
        paired["reasoned_read"]["trace_input_tokens"] += 1
    else:
        paired["reasoned_read"]["recipe"]["max_tokens"] = 1025
    resign(values)
    with pytest.raises(ValueError):
        exported(values)


def test_sparse_pairs_are_reported_and_other_split_public_duplicate_or_unanchored_reads_refused(
    tmp_path,
):
    values = observations(tmp_path)
    samples, tok, cfg, direct, paired, _ = values
    teachers, report = exported((*values[:4], paired[:1], values[5]))
    assert len(teachers) == 1 and report["usage"]["backend_calls"] == 5
    assert sum(row["status"] == "no_saved_paired_read" for row in report["observations"]) == 2
    with pytest.raises(ValueError, match="externally anchored"):
        export_swift_teachers(
            samples,
            tok,
            cfg,
            direct,
            paired,
            input_encoding=ENCODING,
            teacher_identity=IDENTITY,
            expected_direct_sha256="f" * 64,
            expected_paired_sha256=fingerprint(paired),
            verify_gold=verify,
            mechanics_only=True,
        )
    for change in ("split", "public"):
        altered = copy.deepcopy(samples)
        altered[0].metadata[change] = "dev" if change == "split" else True
        with pytest.raises(ValueError, match="train samples"):
            teacher_dataset_items(altered)
    with pytest.raises(ValueError, match="unique question"):
        teacher_dataset_items([samples[0], samples[0]])
    with pytest.raises(ValueError, match="independent verifier"):
        exported(values, gold_verifier=lambda s, q: dict.fromkeys(q.target_distribution, 0.0))
    with pytest.raises(ValueError, match="mechanics-only"):
        export_swift_teachers(
            samples,
            tok,
            cfg,
            direct,
            paired,
            input_encoding=ENCODING,
            teacher_identity=IDENTITY,
            expected_direct_sha256=fingerprint(direct),
            expected_paired_sha256=fingerprint(paired),
            verify_gold=verify,
        )


@pytest.mark.parametrize("native_upload", [False, True])
def test_real_offline_cli_exports_authored_gold_with_saved_fast_tokenizer(tmp_path, native_upload):
    from ayaka.backbone import tiny_text_config
    from ayaka.data.reasoning_v2 import curriculum

    sample, _ = next(iter(curriculum("train", 1)))
    sample.metadata["source_lineage"] = sample.metadata["case_facts_sha256"]
    samples, tok, cfg, direct, paired, _ = observations(tmp_path, samples=[sample])
    paths = {}
    for name, raw in {
        "train": b"\n".join(canonical(s.to_json()) for s in samples) + b"\n",
        "config": canonical(asdict(cfg)),
        "input-encoding": canonical(ENCODING),
        "teacher-identity": canonical(IDENTITY),
        "direct-reads": b"\n".join(canonical(row) for row in direct) + b"\n",
        "paired-reads": b"\n".join(canonical(row) for row in paired) + b"\n",
    }.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(raw)
    tok.hf.save_pretrained(tmp_path / "tokenizer")
    if native_upload:
        tiny_text_config().save_pretrained(tmp_path / "tokenizer")
    command = [sys.executable, "-m", "ayaka.training.teacher_artifacts"]
    for name, path in paths.items():
        command.extend(["--" + name, str(path)])
    command.extend(
        [
            "--expected-train-sha256",
            sha256(paths["train"].read_bytes()),
            "--expected-direct-sha256",
            fingerprint(direct),
            "--expected-paired-sha256",
            fingerprint(paired),
            "--mechanics-only",
            "--native-path" if native_upload else "--mechanics-tokenizer",
            str(tmp_path / "tokenizer"),
            "--out",
            str(tmp_path / "export"),
        ]
    )
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "empty-cache"),
    }
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "export/receipt.json").read_bytes())
    teachers = json.loads((tmp_path / "export/teacher_reads.json").read_bytes())
    assert len(teachers) == len(sample.questions) == report["available_teacher_questions"]
    assert report["optimizer_steps"] == 0
    assert report["model_loaded"] is report["paid_execution_started"] is False
    assert report["mechanics_only"] and report["promotable"] is False
    assert "PRIVATE STEPS" not in json.dumps([teachers, report])
    if native_upload:
        assert set(report["native_metadata"]["files"]) >= {"config.json", "tokenizer.json"}
        assert not (tmp_path / "empty-cache/hub").exists()
    repeated = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
    assert repeated.returncode != 0 and "fresh directory" in repeated.stderr
    command[command.index(str(tmp_path / "export"))] = str(tmp_path / "rejected")
    command.remove("--mechanics-only")
    rejected = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
    assert rejected.returncode != 0
    assert (
        "tiny" in rejected.stderr
        if native_upload
        else "restricted to explicit random tiny" in rejected.stderr
    )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("place", ["direct", "generation", "final"])
def test_canonical_token_dictionary_order_cannot_reverse_candidate_meaning(tmp_path, place):
    values = observations(tmp_path)
    row, pair = values[3][0], values[4][0]
    nested = pair["reasoned_read"]
    if place == "direct":
        wire = row["binding"]["token_inputs"][0]
        # Existing Swift validator maps letters by iteration order. Hashes and
        # dict equality alone cannot bind order; re-sign the swapped masses too.
        wire["canonical_token_ids"] = dict(reversed(list(wire["canonical_token_ids"].items())))
        row["pass_bindings"][0]["canonical_token_ids"] = copy.deepcopy(wire["canonical_token_ids"])
        logits = row["pass_bindings"][0]["token_logits"]
        row["candidate_log_masses"] = {
            label: logits[str(wire["canonical_token_ids"][letter][0])]
            for letter, label in zip(wire["canonical_token_ids"], row["labels"], strict=True)
        }
        row["raw_probs"] = logmass_probs(row["candidate_log_masses"])
    elif place == "generation":
        wire = nested["generation_input"]
    else:
        wire = nested["pass_inputs"][0]
    if place != "direct":
        wire["canonical_token_ids"] = dict(reversed(list(wire["canonical_token_ids"].items())))
    resign(values)
    with pytest.raises(ValueError):
        exported(values)


@pytest.mark.parametrize(
    "field,value", [("passes", 2), ("output_tokens", True), ("latency_s", 0.0)]
)
def test_direct_usage_cannot_claim_extra_calls_or_non_integer_tokens(tmp_path, field, value):
    values = observations(tmp_path)
    values[3][0][field] = value
    resign(values)
    with pytest.raises(ValueError, match="direct teacher observation usage"):
        exported(values)
