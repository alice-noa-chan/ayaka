import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/swift"))
SPEC = importlib.util.spec_from_file_location(
    "v1v2_compare", ROOT / "scripts/swift/v1v2_compare.py"
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)

from ayaka.swift.policy import Policy  # noqa: E402

RUN = {**compare.V1_RUN, "checkpoint_files_sha256": "c" * 64, "policy": {"gate": "calculation"}}


def item(identifier, kind, state, case, source="synth", gold="b"):
    labels = ["false", "true"] if kind == "noul" else ["a", "b"]
    gold = "true" if kind == "noul" else gold
    return {
        "state": state,
        "question": {"type": kind, "instruction": "q", "labels": labels, "descriptions": labels},
        "gold": gold,
        "gold_distribution": None,
        "source": source,
        "tier": "standard",
        "public": False,
        "case_id": case,
        "cluster_id": case,
    }


def cohort():
    return {
        "c1": item("c1", "choice", "s1", "k1"),
        "n1": item("n1", "noul", "s2", "k2"),
        "s1": item("s1", "score", "s3", "k3", gold="b"),
        "c2": item("c2", "choice", "s4", "k4"),
    }


def v2_row(identifier, binding, probs):
    return {
        "id": identifier,
        "type": binding["question"]["type"],
        "labels": binding["question"]["labels"],
        "gold": binding["gold"],
        "gold_distribution": None,
        "tier": "standard",
        "source": binding["source"],
        "case_id": binding["case_id"],
        "cluster_id": binding["cluster_id"],
        "public": False,
        "diagnostic": False,
        "prompt_variant": "min",
        "model": "google/gemma-4-12B-it",
        "revision": compare.BACKBONE_REVISION,
        "readout": "canonical_letter_raw",
        "passes": 1,
        "raw_probs": dict(zip(binding["question"]["labels"], probs, strict=True)),
        "binding": {
            **binding,
            "runtime": {
                "prompt_variant": "min",
                "adapter_sha256": None,
                "implementation_sha256": compare.SWIFT_IMPLEMENTATION,
            },
        },
    }


def v1_row(identifier, binding, probs):
    row = v2_row(identifier, binding, probs)
    for key in ("prompt_variant", "model", "revision", "passes", "diagnostic"):
        row.pop(key)
    return {
        **row,
        "readout": "v1_native_route",
        "run": RUN,
        "binding": binding,
        "baseline_probs": row["raw_probs"],
        "route": "baseline",
    }


def systems():
    expected = cohort()
    v2 = [v2_row(k, b, [0.1, 0.9]) for k, b in expected.items()]
    v1 = [v1_row(k, b, [0.6, 0.4]) for k, b in expected.items()]
    return expected, v2, v1


def test_complete_matched_rows_are_scored():
    expected, v2, v1 = systems()
    report = compare.compare(v2, v1, Policy(noul_commit=False), expected)
    assert report["delta_v2off_minus_v1on"]["credit"] > 0
    assert report["bootstrap"]["components"] == 4
    assert "not a fresh independent confirmation" in report["scope"]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda v2, v1: v2.pop(), "exactly the frozen cohort"),
        (lambda v2, v1: v1.append(copy.deepcopy(v1[0])), "exactly the frozen cohort"),
        (lambda v2, v1: v2[1].update(prompt_variant="cygnet"), "frozen Swift min"),
        (lambda v2, v1: v2[1].update(reasoned_read={}), "frozen Swift min"),
        (lambda v2, v1: v2[2]["binding"].update(state="other"), "does not bind"),
        (lambda v2, v1: v1[2]["binding"].update(tier="hard"), "does not bind"),
        (lambda v2, v1: v1[3].update(run={**RUN, "max_seq_len": 4096}), "pinned v1 route"),
        (
            lambda v2, v1: v1[3].update(run={**RUN, "checkpoint_files_sha256": "d" * 64}),
            "more than one run",
        ),
    ],
)
def test_mismatched_or_incomplete_rows_are_refused(mutate, match):
    expected, v2, v1 = systems()
    mutate(v2, v1)
    with pytest.raises(ValueError, match=match):
        compare.compare(v2, v1, Policy(noul_commit=False), expected)


def test_shared_hotpot_paragraphs_join_one_component():
    rows = [
        {"id": "a", "case_id": "1", "source": "hotpot_val", "binding": {"state": "X: x\nY: y"}},
        {"id": "b", "case_id": "2", "source": "hotpot_val", "binding": {"state": "Y: y2\nZ: z"}},
        {"id": "c", "case_id": "3", "source": "hotpot_val", "binding": {"state": "W: w"}},
        {"id": "d", "case_id": "1", "source": "quality_val", "binding": {"state": "Y: other"}},
    ]
    groups = compare.components(rows)
    assert sorted(map(len, groups)) == [1, 3]


def test_cohort_files_must_match_frozen_hashes(tmp_path):
    paths = {name: tmp_path / f"{name}.jsonl" for name in compare.COHORT}
    for path in paths.values():
        path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen cohort hash"):
        compare.load_cohort(paths)
