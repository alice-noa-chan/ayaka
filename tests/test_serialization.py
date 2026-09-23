from ayaka.serialization import (
    canonical_state,
    dedup_key,
    serialize_state,
    serialize_typed,
    state_cache_key,
)
from ayaka.special_tokens import NUM_SPECIAL_TOKENS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS


def test_typed_serialization_distinguishes_types():
    # 42 != "42" != true at the token level (docs.md section 10)
    num = serialize_typed(42)
    string = serialize_typed("42")
    boolean = serialize_typed(True)
    assert num == "<num> 42"
    assert string == "<str> 42"
    assert boolean == "<bool_true>"
    assert len({num, string, boolean}) == 3
    assert serialize_typed(None) == "<null>"
    assert serialize_typed(False) == "<bool_false>"


def test_object_keys_canonical_order_arrays_preserved():
    value = {"b": 1, "a": 2}
    out = serialize_typed(value)
    # keys sorted canonically
    assert out.index("<key> a") < out.index("<key> b")
    arr = serialize_typed([3, 1, 2])
    # array order preserved
    assert arr.index("<num> 3") < arr.index("<num> 1") < arr.index("<num> 2")


def test_nested_structure_matches_doc_shape():
    state = {"user": {"age": 31, "verified": True}, "tags": ["premium", "beta"]}
    out = serialize_state(state)
    assert out.startswith("<state> <obj>")
    assert out.endswith("<end_obj> </state>")
    assert "<key> user <obj>" in out
    assert "<key> age <num> 31" in out
    assert "<key> verified <bool_true>" in out
    assert "<key> tags <array> <str> premium <str> beta <end_array>" in out


def test_text_state_uses_doc_markers():
    assert serialize_state("hello world") == "<doc> hello world </doc>"


def test_canonical_state_normalizes():
    a = canonical_state({"A ": [1.0, "  X  y"], "b": True})
    b = canonical_state({"b": True, "a": [1, "x Y"]})
    assert a == b


def test_dedup_key_ignores_candidate_order_and_case():
    state = {"msg": "hello"}
    k1 = dedup_key(state, "Pick one", ["Billing", "Tech"])
    k2 = dedup_key(state, " pick  one ", ["tech", "billing"])
    assert k1 == k2
    k3 = dedup_key(state, "Pick one", ["billing", "sales"])
    assert k3 != k1


def test_state_cache_key_versions():
    state = {"x": 1}
    k1 = state_cache_key(state, model_version="m1", tokenizer_version="t1")
    k2 = state_cache_key(state, model_version="m1", tokenizer_version="t2")
    k3 = state_cache_key(state, model_version="m1", tokenizer_version="t1")
    assert k1 == k3 and k1 != k2
    assert len(k1) == 64


def test_special_token_inventory_unique_and_ordered():
    assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))
    assert len(SPECIAL_TOKENS) == NUM_SPECIAL_TOKENS
    assert SPECIAL_TOKEN_IDS["<pad>"] == 0
