import random

import pytest

from ayaka.prompt import LETTERS, QuestionView, canonical_order, render_prefix, render_question
from ayaka.tokenization import ToyTokenizer

TOK = ToyTokenizer()


def _decode_span(ids, span):
    # toy ids are reversible enough for assertions via re-encoding
    return ids[span[0] : span[1]]


def test_canonical_order_is_permutation_invariant():
    descs = ["refund", "Cancel order", "track package", "billing question"]
    base = [descs[i] for i in canonical_order(descs)]
    for seed in range(5):
        perm = descs[:]
        random.Random(seed).shuffle(perm)
        assert [perm[i] for i in canonical_order(perm)] == base


def test_choice_render_identical_under_permutation():
    descs = ["refund", "cancel", "track", "billing"]
    q = render_question(QuestionView("choice", "What does the user want?", descs), TOK)
    perm = [2, 0, 3, 1]
    qp = render_question(
        QuestionView("choice", "What does the user want?", [descs[i] for i in perm]), TOK
    )
    assert q.suffix_ids == qp.suffix_ids
    # label/span of each candidate follow the candidate, not its input slot
    for new_pos, old in enumerate(perm):
        assert qp.label_ids[new_pos] == q.label_ids[old]
        assert qp.option_spans[new_pos] == q.option_spans[old]


def test_choice_labels_and_spans():
    descs = ["beta option", "alpha option"]
    r = render_question(QuestionView("choice", "Pick", descs), TOK)
    # canonical order puts alpha first -> label A
    assert r.label_ids[1] == TOK.single_token_id("A")
    assert r.label_ids[0] == TOK.single_token_id("B")
    for s, e in r.option_spans:
        assert 0 <= s < e <= len(r.suffix_ids)
    assert r.option_spans[1][0] < r.option_spans[0][0]
    assert _decode_span(r.suffix_ids, r.option_spans[1]) == TOK.encode("(A) alpha option\n")


def test_noul_yes_no_readout():
    r = render_question(
        QuestionView("noul", "Is it permitted?", ["a condition is missing", "all conditions hold"]),
        TOK,
    )
    assert r.label_ids == [TOK.single_token_id(" No"), TOK.single_token_id(" Yes")]
    assert r.suffix_ids[-1] == TOK.encode(":")[0]  # cue "Answer:" ends the suffix
    trivial = render_question(QuestionView("noul", "Is it?", ["false", "true"]), TOK)
    assert TOK.encode("the statement holds")[0] in trivial.suffix_ids


def test_score_keeps_ordinal_order_and_digit_labels():
    r = render_question(
        QuestionView("score", "Rate", ["high", "low", "mid"], ordinals=[2, 0, 1]), TOK
    )
    assert r.display_order == [1, 2, 0]
    assert r.label_ids == [
        TOK.single_token_id("2"),
        TOK.single_token_id("0"),
        TOK.single_token_id("1"),
    ]
    wide = render_question(
        QuestionView("score", "Rate", [f"l{i}" for i in range(12)], ordinals=list(range(12))), TOK
    )
    assert wide.label_ids[0] == TOK.single_token_id("A")  # >9 levels -> letters


def test_large_sets_are_pointer_only():
    descs = [f"intent number {i}" for i in range(40)]
    r = render_question(QuestionView("choice", "Which intent?", descs), TOK)
    assert r.label_ids is None
    assert len(r.option_spans) == 40 and all(e > s for s, e in r.option_spans)
    assert len(LETTERS) == 26


def test_prefix_truncation_keeps_both_ends():
    state = "HEAD " + "filler " * 400 + " TAIL"
    ids = render_prefix(state, TOK, max_state_tokens=100)
    full = render_prefix(state, TOK)
    assert len(ids) < len(full)
    assert ids[0] == TOK.bos_id
    from ayaka.prompt import SYSTEM, USER_OPEN

    head_len = 1 + len(TOK.encode(USER_OPEN + SYSTEM + "\n\n<state>\n"))
    tail = TOK.encode("\n</state>\n\n")
    assert ids[head_len : head_len + 4] == TOK.encode("HEAD")
    assert ids[len(ids) - len(tail) - 4 : len(ids) - len(tail)] == TOK.encode("TAIL")
    assert ids[-len(tail) :] == tail


def test_single_candidate_rejected():
    with pytest.raises(ValueError):
        render_question(QuestionView("choice", "q", ["only"]), TOK)
