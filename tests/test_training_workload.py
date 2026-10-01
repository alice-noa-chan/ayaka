import itertools

import pytest

from ayaka.training.workload import (
    finite_workload,
    profile_schedule,
    sample_indices,
    scheduled_batches,
    stress_indices,
)


def inventory():
    return [
        {
            "language": language,
            "source_lineage": f"case-{i}",
            "data_kind": "authored",
            "rows": [
                {
                    "type": "choice",
                    "length": 20 + i,
                    "trace_tokens": i * 2,
                    "proposal_tokens": 0,
                    "image": i == 2,
                    "flagged": False,
                }
            ]
            * (i + 1),
        }
        for i, language in enumerate(("en", "ko", "ja"))
    ]


def test_finite_workload_matches_production_sample_order_and_pending_rows():
    import hashlib

    rows = inventory()
    order = sample_indices([r["language"] for r in rows], 42, {"en": 0.6, "ko": 0.2, "ja": 0.2})
    actual = []
    while len(actual) < 20:
        index = next(order)
        actual.extend((index, position) for position in range(len(rows[index]["rows"])))
    digest = hashlib.sha256("".join(f"{i}:{j};" for i, j in actual[:20]).encode()).hexdigest()
    result = finite_workload(rows, 5, 4, 42, {"en": 0.6, "ko": 0.2, "ja": 0.2})
    assert result["schedule_sha256"] == digest and result["total_rows"] == 20
    assert sum(result["counts"]["route"].values()) == 20
    assert result == finite_workload(rows, 5, 4, 42, {"en": 0.6, "ko": 0.2, "ja": 0.2})
    assert result["schedule_sha256"] != finite_workload(rows, 5, 4, 43)["schedule_sha256"]


def test_sampler_retains_shuffle_epochs_and_language_validation():
    order = list(itertools.islice(sample_indices(["en"] * 3, 7), 9))
    assert all(set(order[n : n + 3]) == {0, 1, 2} for n in (0, 3, 6))
    with pytest.raises(ValueError, match="positive"):
        next(sample_indices(["en", "ja"], 7, {"en": 1}))
    with pytest.raises(ValueError, match="positive"):
        finite_workload(inventory(), 0, 64, 7)


def test_stress_selection_covers_largest_rows_in_every_stratum():
    rows = inventory()
    rows.append({**rows[0], "rows": [{**rows[0]["rows"][0], "length": 900}]})
    selected = stress_indices(rows)
    assert selected == [1, 2, 3]


def test_profile_schedule_visits_full_plan_and_uses_real_batches_with_leftovers():
    rows = inventory()
    planned = list(scheduled_batches(rows, 1200, 4, 7))
    selected = profile_schedule(rows, 1200, 4, 7)
    indices = [index for index, _ in selected]
    assert indices[0] == 0 and indices[-1] == 1199
    assert 20 <= len(indices) <= 26
    assert all(batch == planned[index] and len(batch) == 4 for index, batch in selected)
    assert max(sum(rows[i]["rows"][j]["length"] for i, j in b) for _, b in selected) == max(
        sum(rows[i]["rows"][j]["length"] for i, j in b) for b in planned
    )
    assert profile_schedule(rows, 2, 4, 7) == list(enumerate(planned[:2]))
