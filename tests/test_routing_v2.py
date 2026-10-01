import pytest

from ayaka.routing import BenefitRouter, fit_router, paired_training_rows


def rows(split, gain=0.4):
    return [
        {
            "id": f"{split}/{i}",
            "split": split,
            "features": [0.6, 0.2, 0.67, 5.0, 2.0, 1.0, 0, 0, 0.375],
            "gain": gain,
            "tokens": 100,
        }
        for i in range(32)
    ]


def test_router_promotes_only_dev_validated_gain_and_roundtrips(tmp_path):
    positive = fit_router(rows("router_train"), rows("dev"))
    assert positive.promoted and positive.validation["nll_gain_95ci"][0] > 0
    gain, tokens = positive.predict(rows("dev")[0]["features"])
    assert gain == pytest.approx(0.4) and tokens == pytest.approx(100)
    path = tmp_path / "router.json"
    positive.save(path)
    assert BenefitRouter.load(path).predict(rows("dev")[0]["features"]) == pytest.approx(
        (gain, tokens)
    )
    negative = fit_router(rows("router_train", gain=-0.4), rows("dev", gain=-0.4))
    assert not negative.promoted
    negative.save(path)
    with pytest.raises(ValueError, match="promotion"):
        BenefitRouter.load(path)


def test_router_refuses_test_or_unpaired_results():
    with pytest.raises(ValueError, match="independent"):
        fit_router(rows("test"), rows("dev"))
    with pytest.raises(ValueError, match="aligned"):
        paired_training_rows([{"id": "a"}], [{"id": "b"}], "router_train")


def test_single_budget_validation_cannot_activate_other_efforts():
    from ayaka.primitives import DecisionResult, QuestionSpec
    from ayaka.tokenization import ToyTokenizer

    router = fit_router(rows("router_train"), rows("dev"))
    q = QuestionSpec("noul", "Question?", ["no", "yes"])
    p = DecisionResult("noul", [0.6, 0.4], {})
    assert router.should_reason("State", q, p, ToyTokenizer(), budget=384)
    assert not router.should_reason("State", q, p, ToyTokenizer(), budget=1024)
    assert router.validation["validated_budgets"] == [384]


def test_multiple_efforts_do_not_inflate_independent_sample_count():
    repeated = [dict(row, id="same") for row in rows("router_train")]
    with pytest.raises(ValueError, match="independent"):
        fit_router(repeated, rows("dev"))
    train, dev = [], []
    for budget in (128, 384, 1024):
        for split, destination in (("router_train", train), ("dev", dev)):
            for row in rows(split):
                row["features"][-1] = budget / 1024
                destination.append(row)
    router = fit_router(train, dev)
    assert router.validation["dev_n"] == 96
    assert router.validation["dev_independent_questions"] == 32
    assert router.validation["validated_budgets"] == [128, 384, 1024]
    with pytest.raises(ValueError, match="one paired"):
        fit_router(train + [train[0]], dev)
    assert router.predict(rows("dev")[0]["features"])[1] <= 384


def test_repeated_cases_cannot_be_counted_as_independent_or_cross_splits():
    train = [dict(row, cluster_id="repeated") for row in rows("router_train")]
    with pytest.raises(ValueError, match="independent"):
        fit_router(train, rows("dev"))
    train = [dict(row, cluster_id=f"case/{i}") for i, row in enumerate(rows("router_train"))]
    dev = [dict(row, cluster_id=f"case/{i}") for i, row in enumerate(rows("dev"))]
    with pytest.raises(ValueError, match="underlying case"):
        fit_router(train, dev)
