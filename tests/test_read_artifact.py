import copy
import math

import pytest

from ayaka.eval.read_artifact import (
    ReadIndex,
    assert_read_splits_isolated,
    calibration_rows,
    fingerprint,
    logspace_nll,
    make_binding,
    make_record,
    resolve_target,
    validate_binding,
    validate_record,
)


def parameters(split="calibration", **overrides):
    values = {
        "question_id": f"{split}/q",
        "split": split,
        "case_id": f"source/{split}",
        "lineage_ids": [f"source/{split}"],
        "model_sha256": "a" * 64,
        "runtime": {
            "backend": "cpu-fixture",
            "model_revision": "b" * 40,
            "tokenizer_revision": "c" * 40,
            "tokenizer_sha256": "d" * 64,
            "chat_template_sha256": "e" * 64,
            "reader_sha256": "f" * 64,
            "prompt_recipe_sha256": "1" * 64,
            "alias_vocabulary_sha256": "2" * 64,
            "dtype": "float32",
            "logits_mode": "raw",
            "decision_path": "native_logits",
            "modality": "text",
            "context_limit": 64,
            "vocab_size": 64,
        },
        "kind": "choice",
        "state": {"fact": split},
        "candidates": [{"id": "a", "description": "first"}, {"id": "b", "description": "second"}],
        "messages": [{"role": "user", "content": f"Which answer, {split}?"}],
        "input_token_ids": [10, 11],
        "target_distribution": {"a": 0.4, "b": 0.6},
        "candidate_token_ids": {"a": [0], "b": [1]},
    }
    values.update(overrides)
    return values


def record(split="calibration", **overrides):
    return make_record(make_binding(**parameters(split, **overrides)), {0: 0.0, 1: 0.0})


def test_preserves_soft_target_over_hard_argmax_hint_and_hand_calculated_nll():
    target = resolve_target(["a", "b"], "b", {"a": 0.4, "b": 0.6})
    binding = make_binding(**parameters(target_distribution=target))
    read = make_record(binding, {0: math.log(0.4), 1: math.log(0.6)})
    assert logspace_nll(read) == pytest.approx(-0.4 * math.log(0.4) - 0.6 * math.log(0.6))
    assert logspace_nll(read) != pytest.approx(-math.log(0.6))
    assert read["raw_probs"] == pytest.approx({"a": 0.4, "b": 0.6})


def test_zero_serialized_probability_retains_unsaturated_soft_target_loss():
    binding = make_binding(**parameters(target_distribution={"a": 0.5, "b": 0.5}))
    read = make_record(binding, {0: 0, 1: -1000})
    assert read["raw_probs"] == {"a": 1, "b": 0}
    assert logspace_nll(read) == 500
    assert validate_record(read) == binding


def test_temperature_loss_uses_log_masses_without_losing_soft_targets():
    binding = make_binding(**parameters(target_distribution={"a": 0.75, "b": 0.25}))
    read = make_record(binding, {0: math.log(0.9), 1: math.log(0.1)})
    # sqrt(.9)/(sqrt(.9)+sqrt(.1)) = .75, with .75/.25 gold mass.
    assert logspace_nll(read, 2) == pytest.approx(-0.75 * math.log(0.75) - 0.25 * math.log(0.25))
    for temperature in (0, -1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="temperature"):
            logspace_nll(read, temperature)


def test_alias_counts_and_large_common_offset_are_preserved():
    binding = make_binding(**parameters(candidate_token_ids={"a": [0, 20], "b": [1]}))
    read = make_record(binding, {0: 1e30, 20: 1e30, 1: 1e30})
    assert read["raw_probs"] == pytest.approx({"a": 2 / 3, "b": 1 / 3})
    assert logspace_nll(read) == pytest.approx(-0.4 * math.log(2 / 3) - 0.6 * math.log(1 / 3))
    with pytest.raises(ValueError, match="every declared raw alias"):
        make_record(binding, {0: 0, 1: 0})


def test_exact_cache_roundtrip_and_return_value_isolation():
    source = record()
    original = copy.deepcopy(source)
    index = ReadIndex([source])
    source["raw_probs"]["a"] = 0.9
    cached = index.get(original["binding"])
    assert cached == original
    cached["raw_probs"]["b"] = 0.9
    assert index.get(original["binding"]) == original
    assert index.get(make_binding(**parameters(question_id="new/q"))) is None
    with pytest.raises(ValueError, match="duplicate"):
        ReadIndex([original, original])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(model_sha256="3" * 64),
        lambda p: p["runtime"].update(model_revision="4" * 40),
        lambda p: p["runtime"].update(dtype="bfloat16"),
        lambda p: p["runtime"].update(reader_sha256="5" * 64),
        lambda p: p["runtime"].update(chat_template_sha256="6" * 64),
        lambda p: p["runtime"].update(tokenizer_revision="7" * 40),
        lambda p: p["runtime"].update(prompt_recipe_sha256="8" * 64),
        lambda p: p.update(state={"fact": "changed"}),
        lambda p: p["messages"][0].update(content="different actual prompt"),
        lambda p: p.update(input_token_ids=[10, 12]),
        lambda p: p["candidates"][0].update(description="different candidate text"),
        lambda p: p["candidates"].reverse(),
        lambda p: p.update(candidate_token_ids={"a": [1], "b": [0]}),
        lambda p: p.update(target_distribution={"a": 0.6, "b": 0.4}),
        lambda p: p.update(split="dev"),
        lambda p: p.update(lineage_ids=["different/source"]),
    ],
)
def test_cache_refuses_changed_model_recipe_or_actual_question_inputs(mutation):
    source = record()
    requested = parameters()
    mutation(requested)
    with pytest.raises(ValueError, match="cached read differs"):
        ReadIndex([source]).get(make_binding(**requested))


def test_json_object_key_order_is_stable_but_candidate_order_is_not():
    left = parameters(state={"a": 1, "b": 2})
    right = parameters(state={"b": 2, "a": 1})
    assert make_binding(**left) == make_binding(**right)
    right["candidates"].reverse()
    assert make_binding(**left)["input_sha256"] != make_binding(**right)["input_sha256"]


@pytest.mark.parametrize("split", ["dev", "test", "public", "train"])
def test_calibration_refuses_every_non_calibration_partition(split):
    with pytest.raises(ValueError, match="calibration only"):
        calibration_rows([record(), record(split)])


def test_calibration_binds_one_model_and_recipe_without_assuming_one_candidate_layout():
    base = record()
    options = parameters(
        question_id="calibration/noul",
        kind="noul",
        candidates=[{"id": "false"}, {"id": "true"}],
        candidate_token_ids={"false": [0], "true": [1]},
        target_distribution={"false": 0.5, "true": 0.5},
    )
    noul = make_record(make_binding(**options), {0: 0, 1: 0})
    assert len(calibration_rows([base, noul])) == 2
    changed = parameters(question_id="calibration/other")
    changed["runtime"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="different model/runtime"):
        calibration_rows([base, make_record(make_binding(**changed), {0: 0, 1: 0})])
    with pytest.raises(ValueError, match="nonempty"):
        calibration_rows([])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"question_id": "calibration/q"}, "question"),
        ({"case_id": "source/calibration"}, "lineage"),
        ({"lineage_ids": ["source/calibration"]}, "lineage"),
        ({"state": {"fact": "calibration"}}, "state"),
    ],
)
def test_isolation_catches_renamed_and_translated_cases(overrides, message):
    with pytest.raises(ValueError, match=message + " overlap"):
        assert_read_splits_isolated(
            {"calibration": [record()], "dev": [record("dev", **overrides)]}
        )


def test_disjoint_partitions_and_correct_partition_names():
    assert_read_splits_isolated({"calibration": [record()], "dev": [record("dev")]})
    with pytest.raises(ValueError, match="declared partition"):
        assert_read_splits_isolated({"dev": [record()]})


@pytest.mark.parametrize("mode", ["processed", "processed_logprobs", None])
def test_processed_or_undeclared_logits_mode_is_rejected(mode):
    options = parameters()
    options["runtime"]["logits_mode"] = mode
    with pytest.raises(ValueError, match="raw, native"):
        make_binding(**options)


@pytest.mark.parametrize(
    "aliases",
    [
        {"a": [0]},
        {"a": [0], "b": []},
        {"a": [0, 0], "b": [1]},
        {"a": [0], "b": [0]},
        {"a": [-1], "b": [1]},
    ],
)
def test_missing_invalid_or_overlapping_aliases_are_rejected(aliases):
    with pytest.raises(ValueError, match="alias"):
        make_binding(**parameters(candidate_token_ids=aliases))


@pytest.mark.parametrize(
    "logits", [{0: math.nan, 1: 0}, {0: math.inf, 1: 0}, {0: 0}, {False: 0, 1: 0}]
)
def test_nonfinite_missing_or_boolean_native_logits_are_rejected(logits):
    with pytest.raises(ValueError):
        make_record(make_binding(**parameters()), logits)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda r: r.update(version=2), "versioned"),
        (lambda r: r["binding"].update(model_sha256="9" * 64), "binding fingerprint"),
        (lambda r: r["raw_probs"].update(a=0.9), "probabilities disagree"),
        (lambda r: r.update(logit_offset=math.nan), "offset"),
        (lambda r: r.update(record_sha256="0" * 64), "record fingerprint"),
    ],
)
def test_corrupt_records_are_not_reused(mutation, message):
    source = record()
    mutation(source)
    with pytest.raises(ValueError, match=message):
        validate_record(source)


def test_input_context_and_canonical_noul_order_are_explicit():
    options = parameters(input_token_ids=[0] * 64)
    with pytest.raises(ValueError, match="context"):
        make_binding(**options)
    with pytest.raises(ValueError, match="false/true"):
        make_binding(**parameters(kind="noul"))
    with pytest.raises(ValueError, match="2–26"):
        make_binding(**parameters(candidates=[{"id": str(i)} for i in range(27)]))
    binding = record()["binding"]
    binding["runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recipe fingerprint"):
        validate_binding(binding)


def test_target_validation_and_record_identity_cover_soft_targets():
    assert resolve_target(["a", "b"], "a") == {"a": 1, "b": 0}
    assert resolve_target(["a", "b"], {"a": 1}) == {"a": 1, "b": 0}
    for target in ({"a": 0.6}, {"unknown": 1}, {"a": math.nan, "b": 1}, {"a": True, "b": 0}):
        with pytest.raises(ValueError):
            resolve_target(["a", "b"], target)
    with pytest.raises(ValueError, match="disagree"):
        resolve_target(["a", "b"], {"a": 1}, {"b": 1})
    assert fingerprint(record()) == fingerprint(copy.deepcopy(record()))
