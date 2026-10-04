"""Exact artifact identity with bounded serialization and mutation rejection."""

import copy
import hashlib
from contextvars import copy_context
from dataclasses import asdict

import pytest
from test_direct_distillation import dataset, verify
from test_evidence_swift_bridge import tokenizer

from ayaka.config import tiny_config
from ayaka.eval.read_artifact import fingerprint
from ayaka.tokenization import HFTokenizer, ToyTokenizer
from ayaka.training import tokenizer_identity as identity
from ayaka.training.direct_bundle import audit_bundle, prepare_bundle
from ayaka.training.direct_distillation import prepare_direct_distillation
from ayaka.training.swift_direct import (
    encode_direct_sample,
    input_serving_recipe,
    normalize_input_encoding,
)

ENCODING = {"encoder": "swift_canonical", "prompt_variant": "labeled", "state_format": "compact"}


def counter(monkeypatch):
    calls = []
    original = identity._serialize

    def measured(backend):
        calls.append(backend)
        return original(backend)

    monkeypatch.setattr(identity, "_serialize", measured)
    return calls


def test_exact_existing_hash_conventions_scope_reset_and_copy_isolation(monkeypatch):
    native = tokenizer()
    serialized = native.backend_tokenizer.to_str()
    expected = {
        "raw_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "json_string_sha256": fingerprint(serialized),
    }
    calls = counter(monkeypatch)
    with identity.tokenizer_identity_scope(native):
        for _ in range(20):
            assert identity.backend_fingerprints(native) == expected
        result = identity.backend_fingerprints(native)
        result["raw_sha256"] = "wrong"
        assert identity.backend_fingerprints(native) == expected
    assert len(calls) == 2
    assert identity.backend_fingerprints(native) == expected
    assert identity.backend_fingerprints(native) == expected
    assert len(calls) == 4  # No identity cache survives the preparation scope.


def test_nested_same_object_shared_other_object_isolated_and_exception_reset(monkeypatch):
    first, second = tokenizer(), tokenizer()
    calls = counter(monkeypatch)
    with identity.tokenizer_identity_scope(first):
        with identity.tokenizer_identity_scope(HFTokenizer(first, "tiny")):
            identity.backend_fingerprints(first)
        with identity.tokenizer_identity_scope(second):
            identity.backend_fingerprints(first)
            identity.backend_fingerprints(second)
    assert len(calls) == 4
    assert calls.count(first.backend_tokenizer) == calls.count(second.backend_tokenizer) == 2
    with pytest.raises(RuntimeError), identity.tokenizer_identity_scope(first):
        raise RuntimeError("caller failed")
    before = len(calls)
    identity.backend_fingerprints(first)
    assert len(calls) == before + 1


@pytest.mark.parametrize("mutation", ["backend", "template", "padding", "replacement"])
def test_mutation_discards_results_and_cannot_poison_next_scope(mutation):
    native = tokenizer()
    old = identity.backend_fingerprints(native)
    with (
        pytest.raises(ValueError, match="tokenizer changed"),
        identity.tokenizer_identity_scope(native),
    ):
        if mutation == "backend":
            native.backend_tokenizer.add_tokens(["NEW_OBSERVED_TOKEN"])
        elif mutation == "template":
            native.chat_template += " changed"
        elif mutation == "padding":
            native.padding_side = "left"
        else:
            native._tokenizer = copy.deepcopy(native.backend_tokenizer)
        # Exit checks full backend bytes, even without another cached read.
    with identity.tokenizer_identity_scope(native):
        new = identity.backend_fingerprints(native)
    if mutation == "backend":
        assert new != old
    else:
        assert new == old


def test_decorated_pure_preparation_checks_before_return_and_rejects_unsupported_signature():
    native = tokenizer()

    @identity.scoped_tokenizer_preparation
    def prepare(value, tok):
        if value:
            tok.backend_tokenizer.add_tokens(["MUTATED"])
        return identity.backend_fingerprints(tok)

    with pytest.raises(ValueError, match="tokenizer changed"):
        prepare(True, native)
    assert prepare(False, tok=native) == identity.backend_fingerprints(native)
    with identity.tokenizer_identity_scope(ToyTokenizer()):
        pass
    with pytest.raises(ValueError, match="requires a tokenizer argument"):
        identity.scoped_tokenizer_preparation(lambda other: other)


def test_direct_preparation_uses_two_serializations_with_identical_items_and_recipes(monkeypatch):
    tok = HFTokenizer(tokenizer(), "tiny")
    cfg = tiny_config(readout="lm", max_seq_len=2048)
    samples = dataset()
    expected = [
        asdict(item)
        for sample in samples["train"]
        for item in encode_direct_sample(sample, tok, cfg, input_encoding=ENCODING)
    ]
    recipe = input_serving_recipe(tok, ENCODING)
    calls = counter(monkeypatch)
    items, report = prepare_direct_distillation(
        samples, tok, cfg, {}, verify, input_encoding=ENCODING
    )
    assert len(calls) == 2
    assert [asdict(item) for item in items] == expected
    assert report["input_encoding"] == normalize_input_encoding(ENCODING)
    assert all(item.direct_input_binding["recipe"] == recipe for item in items)


def test_complete_bundle_rechecks_rows_gold_and_bytes_without_per_sample_serialization(
    tmp_path, monkeypatch
):
    from test_direct_bundle import dataset as authored_splits

    tok = HFTokenizer(tokenizer(), "tiny")
    cfg = tiny_config(readout="lm", max_seq_len=2048)
    calls = counter(monkeypatch)
    root = tmp_path / "bundle"
    prepare_bundle(
        root,
        authored_splits(),
        tok,
        cfg,
        {},
        steps=2,
        rows_per_step=3,
        allow_tiny=True,
        input_encoding=ENCODING,
    )
    # Two standalone recipe hashes + one test scope + one complete development scope.
    assert len(calls) == 6
    before = len(calls)
    audit_bundle(root, tok=tok, allow_tiny=True)
    assert len(calls) - before == 4


def test_gold_callback_mutation_fails_before_direct_preparation_returns():
    tok = HFTokenizer(tokenizer(), "tiny")
    cfg = tiny_config(readout="lm", max_seq_len=2048)

    def mutate_then_verify(sample, question):
        tok.hf.backend_tokenizer.add_tokens(["CHANGED_DURING_GOLD_VERIFICATION"])
        return verify(sample, question)

    with pytest.raises(ValueError, match="tokenizer changed"):
        prepare_direct_distillation(
            dataset(), tok, cfg, {}, mutate_then_verify, input_encoding=ENCODING
        )


def test_copied_context_cannot_reuse_a_snapshot_after_its_scope_exits(monkeypatch):
    native = tokenizer()
    calls = counter(monkeypatch)
    with identity.tokenizer_identity_scope(native):
        previous = identity.backend_fingerprints(native)
        copied = copy_context()
    native.backend_tokenizer.add_tokens(["AFTER_SCOPE_EXIT"])
    actual = copied.run(identity.backend_fingerprints, native)
    assert actual != previous and len(calls) == 3

    def new_scope():
        with identity.tokenizer_identity_scope(native):
            assert identity.backend_fingerprints(native) == actual

    copied.run(new_scope)
    assert len(calls) == 5
