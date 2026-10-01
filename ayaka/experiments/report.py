"""CPU-only summaries of measured v2 artifacts, preserving their provenance."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..data.schema import Sample
from ..eval.reasoning_v2 import dataset_signature
from ..eval.v2 import paired_report, summarize


def diagnostic_summary(rows):
    result = summarize(rows)
    score = [r for r in rows if r["type"] == "score"]
    if score:
        result["by_type"]["score"].update(
            nmae=sum(r["nmae"] for r in score) / len(score),
            chance_nmae=sum(r["nmae_chance"] for r in score) / len(score),
        )
    return result


def compact_evaluation(report, samples=None, *, legacy_curriculum_verified=False):
    """Recompute diagnostics from stored rows; never generate or alter raw files."""
    rows = copy.deepcopy(report.get("rows", {}))
    verification = "recorded artifact; no prepared-split comparison"
    if samples is not None:
        signature = dataset_signature(samples)
        verification = "measured artifact signature matches prepared split"
        unsigned = report.get("dataset_signature") is None
        if report.get("dataset_signature") != signature and not (
            unsigned and legacy_curriculum_verified
        ):
            raise ValueError("measured artifact does not match the prepared split")
        if unsigned:
            verification = "legacy curriculum code hash and row IDs/types/targets; original input signature unavailable"
        from ..training.batching import _noul_canonical

        questions = {
            f"{s.metadata.get('source_example_id')}/{q.id}": _noul_canonical(q)
            for s in samples
            for q in s.questions
        }
        cases = {
            f"{s.metadata.get('source_example_id')}/{q.id}": s.metadata.get(
                "case_facts_sha256", f"{s.metadata.get('source_example_id')}/{q.id}"
            )
            for s in samples
            for q in s.questions
        }
        for rs in rows.values():
            for row in rs:
                if row["id"] not in cases:
                    raise ValueError("measured question is absent from the prepared split")
                q = questions[row["id"]]
                target = [q.target_distribution.get(c.id, 0) for c in q.candidates]
                if row["type"] != q.type or row["target"] != target:
                    raise ValueError("measured targets or types differ from preparation")
                if row.get("cluster_id", cases[row["id"]]) != cases[row["id"]]:
                    raise ValueError("measured case identity changed")
                row["cluster_id"] = cases[row["id"]]
    summaries = {}
    for mode, rs in rows.items():
        if not rs:
            continue
        summaries[mode] = {
            **diagnostic_summary(rs),
            "independent_cases": len({r.get("cluster_id", r["id"]) for r in rs}),
            "routes": dict(Counter(r.get("route", "unknown") for r in rs)),
            "finish_reasons": dict(Counter(r.get("finish_reason", "unknown") for r in rs)),
            "errors": dict(Counter(r["error"] for r in rs if r.get("error"))),
            "requested_budgets": dict(Counter(str(r["budget"]) for r in rs)),
            "input_tokens": sum(r.get("input_tokens", 0) for r in rs),
            "by_family": {
                family: diagnostic_summary([r for r in rs if r.get("family", "unknown") == family])
                for family in sorted({r.get("family", "unknown") for r in rs})
            },
            "by_language": {
                language: diagnostic_summary(
                    [r for r in rs if r.get("language", "unknown") == language]
                )
                for language in sorted({r.get("language", "unknown") for r in rs})
            },
        }
    pairs = {
        mode: paired_report(rows["off"], rs)
        for mode, rs in rows.items()
        if mode != "off"
        and rs
        and rows.get("off")
        and [r["id"] for r in rs] == [r["id"] for r in rows["off"]]
    }
    return {
        "status": report["status"],
        "dataset_signature": report.get("dataset_signature"),
        "dataset_verification": verification,
        "reports": summaries,
        "paired": pairs,
        "diagnostic_recomputation": "stored probabilities, latency and raw tokens; case-clustered CI",
    }


def collect(run):
    root = Path(run)
    sources = {}

    def read(path):
        relative = str(path.relative_to(root)).replace("\\", "/")
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def split_samples(split):
        path = root / f"{split}.json"
        if not path.exists():
            raise ValueError(f"prepared {split} split is missing")
        return [Sample.from_json(r["sample"]) for r in read(path)]

    result = {"official_composite": None, "sealed": False, "screen": {}, "evaluation": {}}
    for name in (
        "preparation",
        "selection",
        "budget",
        "head_selection",
        "natural_provenance",
        "recover_screen_status",
        "screen_status",
        "heads_status",
        "sft_status",
        "evaluate_status",
        "screen_summary",
    ):
        path = root / f"{name}.json"
        if path.exists():
            result[name] = read(path)
    for path in sorted(root.glob("screen/*.json")):
        report = read(path)
        # Interrupted files that never became a measured screen are excluded.
        if "reports" in report:
            original = next(
                (r for r in result.get("screen_summary", []) if r.get("name") == path.stem), {}
            )
            curriculum_hash = hashlib.sha256(
                (Path(__file__).parents[1] / "data/reasoning_v2.py").read_bytes()
            ).hexdigest()
            legacy_verified = (
                original.get("code_sha256", {}).get("data/reasoning_v2.py") == curriculum_hash
            )
            if report.get("dataset_signature") is None and not legacy_verified:
                result["screen"][path.stem] = {
                    "status": report["status"],
                    "dataset_signature": None,
                    "dataset_verification": "legacy input signature and original curriculum hash unavailable",
                    "reports": report["reports"],
                    "paired": {},
                    "diagnostic_recomputation": "unavailable; original point measurements retained without inferred case intervals",
                }
                continue
            result["screen"][path.stem] = compact_evaluation(
                report,
                split_samples("dev"),
                legacy_curriculum_verified=legacy_verified,
            )
    if (root / "heads.json").exists():
        result["heads"] = [
            {
                **{k: v for k, v in r.items() if k != "evaluation"},
                "evaluation": compact_evaluation(r["evaluation"], split_samples("dev")),
            }
            for r in read(root / "heads.json")
        ]
    result["sft"] = {}
    for path in sorted(root.glob("sft/*/meta.json")):
        result["sft"][path.parent.name] = {
            "metadata": read(path),
            "config": read(path.parent / "ayaka_config.json"),
            "checkpoint_files": {
                str(p.relative_to(path.parent)).replace("\\", "/"): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sorted(path.parent.rglob("*"))
                if p.is_file() and p.suffix == ".safetensors"
            },
        }
    for path in sorted(root.glob("eval/*/*.json")):
        name, kind = path.parent.name, path.stem
        raw = read(path)
        candidate = result["evaluation"].setdefault(name, {})
        if "rows" not in raw:
            candidate[kind] = raw
            continue
        split = "calibration" if kind == "calibration_measurements" else kind
        samples = (
            split_samples(split)
            if split in ("router_train", "dev", "calibration", "test")
            else None
        )
        if kind == "router_train" and dataset_signature(samples) != raw.get("dataset_signature"):
            from .v2 import case_representatives

            samples = case_representatives(samples)
        candidate[kind] = compact_evaluation(raw, samples)
    result["failures"] = {path.name: read(path) for path in sorted(root.glob("*-failure.json"))}
    result["source_artifact_sha256"] = sources
    result["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = collect(args.run)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
