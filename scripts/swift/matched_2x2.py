"""Matched public-item diagnostic; previously seen items explain a gap, select nothing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC = REPO / "ayaka/eval/data/jevbench_public"
DEFAULT_HISTORICAL = REPO / "runs/runpod-large-v1-20260927/runs/large-v1/jevbench_report.json"
DEFAULT_CYGNET = Path(
    "C:/Users/solso/AppData/Local/Temp/claude/D--models-ayaka/"
    "bb598320-f421-4cb7-b377-a1922da1a193/scratchpad/"
    "cygnet-recipe/runs/l40s-pinned/results.jsonl"
)
CELLS = ("frozen_native", "frozen_swift", "lora_native", "lora_swift")
NOTICE = (
    "DIAGNOSTIC ONLY: these 231 public items were seen before. This explains a gap; "
    "it selects nothing and is never used for fitting, selection or gates."
)
LIMITATIONS = (
    "The native/Swift factor combines input serialization and readout; it cannot separate them. "
    "Native frozen hybrid has zero gates (effective LM for labeled public items). "
    "Native checkpoint loads LoRA, pointer head, gates and temperatures; its weights contrast "
    "is the complete v1 checkpoint effect, while Swift's is adapter-only. "
    "Current native cells use 8192 tokens; historical v1 predates c7f10cb. "
    "Exact McNemar tests compare paired binary correctness, are two-sided and unadjusted. "
    "An interaction or averaged marginal effect is not a binary pair, so McNemar does not apply."
)


def records(path: Path) -> list[dict]:
    """Accept native reports, flat result arrays and JSONL reads/results."""
    source = path.read_text(encoding="utf-8")
    try:
        document = json.loads(source)
    except json.JSONDecodeError:
        return [json.loads(line) for line in source.splitlines() if line.strip()]
    if isinstance(document, list):
        return document
    if "tiers" in document:
        return [
            {**row, "tier": tier}
            for tier, report in document["tiers"].items()
            for row in report["results"]
        ]
    if "results" in document:
        return document["results"]
    return [document]


def public_items(directory: Path) -> dict[str, dict]:
    result = {}
    for tier in ("easy", "original", "hard"):
        for row in records(directory / (tier + ".jsonl")):
            if row["id"] in result:
                raise ValueError(f"duplicate public id: {row['id']}")
            result[row["id"]] = {
                "tier": "standard" if tier == "original" else tier,
                "type": row["question"]["type"],
                "family": row["family"],
                "expected": str(row["expected"]),
                "labels": [str(label) for label in row["labels"]],
            }
    return result


def label(value: object, kind: str) -> str:
    value = str(value)
    if kind == "noul":
        return {"no": "false", "yes": "true"}.get(value.lower(), value.lower())
    return value


def load_cell(
    path: Path, items: dict[str, dict], *, cell_name: str | None = None
) -> dict[str, dict]:
    result = {}
    # A Swift factor must use one direct prompt variant/base/tokenizer recipe.
    swift_recipe = None
    for row in records(path):
        item_id = row.get("id", row.get("task_id"))
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"missing id in {path}")
        if item_id in result:
            raise ValueError(f"duplicate id in {path}: {item_id}")
        if item_id not in items:
            raise ValueError(f"id-set mismatch in {path}: unexpected {item_id}")
        item = items[item_id]
        if cell_name is not None and "cell" in row and row["cell"] != cell_name:
            raise ValueError(f"wrong native cell in {path}: {item_id}")
        if "raw_probs" in row:
            if row.get("reasoned_read") or row.get("readout") not in (None, "canonical_letter_raw"):
                raise ValueError(f"matched Swift cell requires direct letter reads: {item_id}")
            recipe = tuple(
                row.get(key)
                for key in ("model", "revision", "tokenizer_revision", "prompt_variant", "readout")
            )
            if swift_recipe is not None and recipe != swift_recipe:
                raise ValueError(f"mixed Swift recipes in {path}")
            swift_recipe = recipe
        kind = item["type"]
        gold = label(item["expected"], kind)
        if "type" in row and row["type"] != kind:
            raise ValueError(f"type mismatch: {item_id}")
        if "tier" in row and row["tier"].replace("original", "standard") != item["tier"]:
            # Swift reads collected before the infer_tier fix labelled hard.jsonl
            # judge_hard items as "judge"; the public data's tier is authoritative.
            legacy_judge = (
                "raw_probs" in row
                and row["tier"] == "judge"
                and item["tier"] == "hard"
                and "judge" in str(item.get("family", ""))
            )
            if not legacy_judge:
                raise ValueError(f"tier mismatch: {item_id}")
        if "family" in row and row["family"] != item["family"]:
            raise ValueError(f"family mismatch: {item_id}")
        for key in ("gold", "expected"):
            if key in row and label(row[key], kind) != gold:
                raise ValueError(f"gold mismatch: {item_id}")
        probs = row.get("raw_probs", row.get("probs"))
        predicted = row.get("pred", row.get("predicted"))
        if probs is not None:
            if not isinstance(probs, dict) or not probs:
                raise ValueError(f"probabilities must be a nonempty object: {item_id}")
            if any(
                not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                for p in probs.values()
            ) or not math.isclose(sum(probs.values()), 1, abs_tol=1e-5):
                raise ValueError(f"invalid probabilities: {item_id}")
            allowed = {label(lb, kind) for lb in item["labels"]}
            if {label(lb, kind) for lb in probs} != allowed:
                raise ValueError(f"probability labels mismatch: {item_id}")
            if predicted is None:
                predicted = max(probs, key=probs.get)
        if predicted is None:
            raise ValueError(f"missing prediction/probabilities: {item_id}")
        predicted = label(predicted, kind)
        if predicted not in {label(lb, kind) for lb in item["labels"]}:
            raise ValueError(f"unknown prediction label: {item_id}")
        correct = predicted == gold
        # Cygnet's `ok` is request success, not correctness. Native `ok` is correctness.
        recorded = row.get("correct", row.get("ok") if "task_id" not in row else None)
        if recorded is not None and recorded != correct:
            raise ValueError(f"correctness disagrees with prediction: {item_id}")
        result[item_id] = {
            "pred": predicted,
            "correct": correct,
            "probs": probs,
            **{key: row[key] for key in ("prompt_variant", "tokenizer_revision") if key in row},
        }
    if set(result) != set(items):
        raise ValueError(
            f"id-set mismatch in {path}: missing {sorted(set(items) - set(result))[:5]}"
        )
    return result


def mcnemar_exact(left_only: int, right_only: int) -> float:
    """Exact two-sided binomial test conditional on the discordant count."""
    if min(left_only, right_only) < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / 2**n)


def contrast(left, right, ids, items) -> dict:
    def counts(cohort):
        a = sum(left[i]["correct"] and not right[i]["correct"] for i in cohort)
        b = sum(right[i]["correct"] and not left[i]["correct"] for i in cohort)
        return {
            "n": len(cohort),
            "left_only": a,
            "right_only": b,
            "discordant": a + b,
            "delta_accuracy": (b - a) / len(cohort),
            "mcnemar_exact_p": mcnemar_exact(a, b),
        }

    return {
        **counts(ids),
        "by_family": {
            family: counts([i for i in ids if items[i]["family"] == family])
            for family in sorted({items[i]["family"] for i in ids})
        },
        "discordant_ids": {
            "left_only": [i for i in ids if left[i]["correct"] and not right[i]["correct"]],
            "right_only": [i for i in ids if right[i]["correct"] and not left[i]["correct"]],
        },
    }


def analyze(cells: dict[str, dict], items: dict[str, dict]) -> dict:
    if not items or not set(CELLS) <= set(cells):
        raise ValueError("four nonempty matched cells are required")
    for name, cell in cells.items():
        if set(cell) != set(items):
            raise ValueError(f"id-set mismatch: {name}")
    first_id = next(iter(items))
    frozen_swift, lora_swift = (cells[name][first_id] for name in ("frozen_swift", "lora_swift"))
    for key in ("prompt_variant", "tokenizer_revision"):
        if key in frozen_swift and key in lora_swift and frozen_swift[key] != lora_swift[key]:
            raise ValueError(f"Swift cells differ in {key}")
    groups = {"all": sorted(items)}
    for field in ("tier", "type"):
        for value in sorted({row[field] for row in items.values()}):
            groups[f"{field}/{value}"] = sorted(i for i in items if items[i][field] == value)
    for tier in sorted({row["tier"] for row in items.values()}):
        for kind in sorted({row["type"] for row in items.values()}):
            ids = sorted(i for i in items if items[i]["tier"] == tier and items[i]["type"] == kind)
            if ids:
                groups[f"tier_type/{tier}/{kind}"] = ids
    output = {}
    pairs = {
        "weights_native": ("frozen_native", "lora_native"),
        "weights_swift": ("frozen_swift", "lora_swift"),
        "format_readout_frozen": ("frozen_native", "frozen_swift"),
        "format_readout_lora": ("lora_native", "lora_swift"),
        "diagonal_frozen_native_to_lora_swift": ("frozen_native", "lora_swift"),
        "diagonal_lora_native_to_frozen_swift": ("lora_native", "frozen_swift"),
    }
    if "cygnet_reference" in cells and "historical_v1" in cells:
        pairs["historical_v1_to_cygnet"] = ("historical_v1", "cygnet_reference")
    if "cygnet_reference" in cells:
        pairs["cygnet_to_frozen_swift"] = ("cygnet_reference", "frozen_swift")
    if "historical_v1" in cells:
        pairs["historical_to_current_v1"] = ("historical_v1", "lora_native")
    for group, ids in groups.items():
        accuracy = {
            name: {
                "correct": sum(cell[i]["correct"] for i in ids),
                "accuracy": sum(cell[i]["correct"] for i in ids) / len(ids),
            }
            for name, cell in cells.items()
        }
        a, b, c, d = (accuracy[name]["accuracy"] for name in CELLS)
        output[group] = {
            "n": len(ids),
            "accuracy": accuracy,
            "effects": {
                "weights_native": c - a,
                "weights_swift": d - b,
                "format_readout_frozen": b - a,
                "format_readout_lora": d - c,
                "weights_marginal": ((c - a) + (d - b)) / 2,
                "format_readout_marginal": ((b - a) + (d - c)) / 2,
                "interaction": (d - b) - (c - a),
            },
            "contrasts": {
                name: {
                    "left": left,
                    "right": right,
                    **contrast(cells[left], cells[right], ids, items),
                }
                for name, (left, right) in pairs.items()
            },
        }
    return {
        "role": "diagnostic",
        "used_for_fit_or_selection": False,
        "used_for_gates": False,
        "notice": NOTICE,
        "limitations": LIMITATIONS,
        "effect_units": "accuracy difference (multiply by 100 for percentage points)",
        "interaction_formula": "(lora_swift - frozen_swift) - (lora_native - frozen_native)",
        "groups": output,
    }


def markdown(report: dict) -> str:
    groups = report["groups"]
    names = list(groups["all"]["accuracy"])
    lines = [NOTICE, "", LIMITATIONS, "", "Accuracy (correct/n; %)", ""]
    lines += [
        "| Cohort | n | " + " | ".join(names) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in names) + " |",
    ]
    for group, data in groups.items():
        scores = [
            f"{data['accuracy'][name]['correct']}/{data['n']}; "
            f"{100 * data['accuracy'][name]['accuracy']:.2f}"
            for name in names
        ]
        lines.append(f"| {group} | {data['n']} | " + " | ".join(scores) + " |")
    lines += ["", "Effects (percentage points; positive favors LoRA or Swift)", ""]
    effects = list(groups["all"]["effects"])
    lines += [
        "| Cohort | " + " | ".join(effects) + " |",
        "| --- | " + " | ".join("---:" for _ in effects) + " |",
    ]
    for group, data in groups.items():
        lines.append(
            f"| {group} | " + " | ".join(f"{100 * data['effects'][e]:+.2f}" for e in effects) + " |"
        )
    lines += [
        "",
        "Paired contrasts (delta = right minus left; exact two-sided McNemar)",
        "",
        "| Cohort | Contrast | Left only | Right only | Delta pp | p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for group, data in groups.items():
        for name, row in data["contrasts"].items():
            lines.append(
                f"| {group} | {name} | {row['left_only']} | {row['right_only']} | "
                f"{100 * row['delta_accuracy']:+.2f} | {row['mcnemar_exact_p']:.6g} |"
            )
    lines += [
        "",
        "Per-family discordance (all items; left/right as in the named contrast)",
        "",
        "| Contrast | Family | n | Left only | Right only |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, row in groups["all"]["contrasts"].items():
        for family, counts in row["by_family"].items():
            lines.append(
                f"| {name} | {family} | {counts['n']} | "
                f"{counts['left_only']} | {counts['right_only']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in CELLS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--cygnet-reference", type=Path, default=DEFAULT_CYGNET)
    parser.add_argument("--historical-v1", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--public-data-dir", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--output", type=Path, required=True, help="JSON report; Markdown sibling")
    args = parser.parse_args(argv)
    items = public_items(args.public_data_dir)
    if len(items) != 231:
        parser.error(f"expected the same 231 public items, found {len(items)}")
    paths = {name: getattr(args, name) for name in (*CELLS, "cygnet_reference", "historical_v1")}
    report = analyze(
        {name: load_cell(path, items, cell_name=name) for name, path in paths.items()}, items
    )
    report["inputs"] = {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items()
    }
    report["public_ids_sha256"] = hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(NOTICE)
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")
    return report


if __name__ == "__main__":
    main()
