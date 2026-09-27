from ayaka.evidence import validate_program
from ayaka.evidence_retrieval import retrieve


def long_record():
    return {
        "state": "Policy: current request cap is 5000 USD.\n\n"
        + "Archive: stationery and office paint were recorded.\n\n" * 350
        + "Current request: 4321 USD; applicable cap 5000 USD.",
        "question": {"type": "noul", "instructions": "Is the current request below its cap?"},
        "labels": ["no", "yes"],
    }


def test_retrieval_keeps_verbatim_spans_and_request_facts():
    record = long_record()
    view, info = retrieve(record)
    assert info["active"]
    assert info["chars_after"] < info["chars_before"]
    assert "Current request: 4321 USD" in view["state"]
    quotes = [record["state"][s["start"] : s["end"]].strip() for s in info["selected_spans"]]
    for quote in quotes:
        validate_program(record["state"], {"quotes": [quote], "calculations": {}})


def test_retrieval_cannot_use_hidden_labels_or_family():
    record = long_record()
    original = retrieve(record)
    record.update(expected="SECRET", family="SECRET", oracle={"answer": "SECRET"})
    assert retrieve(record) == original
    assert "SECRET" not in str(original)


def test_structured_unicode_input_is_not_rewritten():
    record = {"state": {"memo": "보험 한도 — 12 kg"}, "question": "Check weight", "labels": []}
    view, info = retrieve(record)
    assert view["state"] == record["state"]
    assert not info["active"]
