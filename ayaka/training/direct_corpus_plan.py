"""Whole-corpus contracts for offline human rehearsal and direct decisions.

An input plan declares selection settings and asset identities before selection.
The computed contract freezes selected membership, not proof of a deterministic
selection replay, runtime performance, unseen evaluation or model quality.
Validation here never opens raw/private data or native weights.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from decimal import Decimal

from ..data.contract_nli import SOURCE as CONTRACT_SOURCE
from ..data.decontam import POLICY
from ..data.natural_training_v2 import FINAL_SOURCES, POLICY_SOURCES, SOURCES
from ..data.reasoning_v2 import SPLITS
from ..data.reserved_evidence import POLICY as RESERVED_POLICY
from ..eval.read_artifact import fingerprint
from .workload import scheduled_batches

VERSION = "ayaka-direct-corpus-plan-1"
POLICY_VERSION = "ayaka-direct-corpus-plan-2"
# Plan2 plus the opt-in extra natural sources (StrategyQA).
FINAL_VERSION = "ayaka-direct-corpus-plan-3"
MARKER = "corpus_plan_sha256"
AUTHORED = "ayaka-v2-verified"
TYPES = ("choice", "noul", "score")
SELECTION = {
    "split": "source_group_closure_v2",
    "permutation": "sha256(seed,source_example_id,question_id)-shuffle-v1",
    "context": "final-native-input-whole-sample-no-truncation-v1",
    "decontamination": POLICY,
    "membership": "externally-audited-frozen-selection; no selection-replay claim",
}
POLICY_SELECTION = {**SELECTION, "reserved_evidence": RESERVED_POLICY}
SOURCE_WIDTHS = {
    "helpsteer2": (5, "score"),
    "commonsense_qa": (1, "choice"),
    "massive_ko": (2, "noul"),
    "massive_ja": (2, "noul"),
    CONTRACT_SOURCE: (17, "choice"),
    "strategyqa": (1, "noul"),
}
SETTINGS = {
    "natural_sample_quotas",
    "authored_per_type",
    "epochs",
    "rows_per_step",
    "seed",
    "minimum_english_question_fraction",
}
ASSETS = {
    "model_sha256",
    "tokenizer_sha256",
    "native_metadata_sha256",
    "input_encoding_sha256",
    "gold_sources_sha256",
    "public_files",
    "reserved",
}


def digest(value, name="digest"):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"corpus {name} must be a SHA256")
    return value


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"corpus {name} requires exactly the known fields")


def validate_settings(settings):
    _exact(settings, SETTINGS, "settings")
    quotas = settings["natural_sample_quotas"]
    if not isinstance(quotas, dict) or set(quotas) not in (
        set(SOURCES),
        set(POLICY_SOURCES),
        set(FINAL_SOURCES),
    ):
        raise ValueError(
            "corpus natural sources require the four-source, policy or final inventory"
        )
    for limits in quotas.values():
        _exact(limits, SPLITS, "sample quotas")
        if any(type(n) is not int or n < 1 for n in limits.values()):
            raise ValueError("corpus sample quotas must be positive exact integers")
    _exact(settings["authored_per_type"], SPLITS, "authored counts")
    if any(type(n) is not int or not 1 <= n <= 256 for n in settings["authored_per_type"].values()):
        raise ValueError("corpus authored per-type counts must be integers in 1..256")
    if any(type(settings[k]) is not int or settings[k] < 1 for k in ("epochs", "rows_per_step")):
        raise ValueError("corpus epochs and rows_per_step must be positive exact integers")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("corpus seed must be a nonnegative exact integer")
    fraction = settings["minimum_english_question_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("corpus English question fraction must be finite in 0..1")
    return settings


def validate_plan(plan):
    _exact(plan, {"version", "settings", "assets", "selection_policy"}, "plan")
    expected_sources, selection = (
        (SOURCES, SELECTION)
        if plan["version"] == VERSION
        else (POLICY_SOURCES, POLICY_SELECTION)
        if plan["version"] == POLICY_VERSION
        else (FINAL_SOURCES, POLICY_SELECTION)
        if plan["version"] == FINAL_VERSION
        else (None, None)
    )
    if expected_sources is None or plan["selection_policy"] != selection:
        raise ValueError("unsupported corpus plan or selection policy")
    validate_settings(plan["settings"])
    _exact(plan["settings"]["natural_sample_quotas"], expected_sources, "versioned natural sources")
    assets = plan["assets"]
    _exact(assets, ASSETS, "asset binding")
    for key in ASSETS - {"public_files", "reserved"}:
        digest(assets[key], key)
    public = assets["public_files"]
    if not isinstance(public, dict) or not public:
        raise ValueError("corpus plan requires a public benchmark inventory")
    for name, value in public.items():
        if not isinstance(name, str) or not name.endswith(".jsonl") or "/" in name or "\\" in name:
            raise ValueError("corpus public inventory requires literal JSONL names")
        digest(value, "public file")
    reserved = assets["reserved"]
    _exact(reserved, {"scope", "manifest_sha256", "files"}, "reserved inventory")
    if reserved["scope"] == "draft_without_prior_private_inventory":
        if reserved["manifest_sha256"] is not None or reserved["files"] != {}:
            raise ValueError("draft corpus cannot claim a verified private inventory")
    elif reserved["scope"] == "externally_anchored_private_inventory":
        digest(reserved["manifest_sha256"], "reserved manifest")
        if not isinstance(reserved["files"], dict) or not reserved["files"]:
            raise ValueError("reserved inventory must contain original evaluation files")
        for key, value in reserved["files"].items():
            digest(key, "reserved file")
            if type(value) is not int or value < 1:
                raise ValueError("reserved file sample counts must be positive exact integers")
    else:
        raise ValueError("unknown corpus private inventory scope")
    return plan


def plan_sha256(plan):
    return fingerprint(validate_plan(plan))


def validate_recipe(plan, recipe):
    validate_plan(plan)
    assets = plan["assets"]
    actual = {
        "model_sha256": fingerprint(recipe["model"]),
        "tokenizer_sha256": recipe["tokenizer_sha256"],
        "native_metadata_sha256": fingerprint(recipe["native_metadata"]),
        "input_encoding_sha256": fingerprint(recipe["input_encoding"]),
        "gold_sources_sha256": fingerprint(recipe["gold_sources"]),
    }
    if any(assets[key] != value for key, value in actual.items()):
        raise ValueError("corpus plan differs from actual model/tokenizer/input/raw bindings")
    if recipe.get("corpus_plan_sha256") != plan_sha256(plan):
        raise ValueError("recipe corpus plan digest differs")


def split_summary(samples, expected_plan_sha256):
    digest(expected_plan_sha256, "plan")
    sources, languages = {}, Counter()
    for sample in samples:
        if sample.metadata.get(MARKER) != expected_plan_sha256:
            raise ValueError("every corpus sample must retain its exact plan marker")
        source = sample.metadata.get("source")
        if source not in {*FINAL_SOURCES, AUTHORED}:
            raise ValueError("corpus contains an unplanned source")
        row = sources.setdefault(source, {"samples": 0, "questions": 0, "types": Counter()})
        row["samples"] += 1
        row["questions"] += len(sample.questions)
        row["types"].update(q.type for q in sample.questions)
        languages[sample.metadata["language"]] += len(sample.questions)
    return {
        "sources": {name: {**row, "types": dict(row["types"])} for name, row in sources.items()},
        "language_questions": dict(languages),
        "membership_sha256": fingerprint([s.to_json() for s in samples]),
    }


def validate_summary(summary):
    _exact(summary, {"sources", "language_questions", "membership_sha256"}, "split summary")
    digest(summary["membership_sha256"], "selected membership")
    if not isinstance(summary["sources"], dict) or set(summary["sources"]) not in (
        {*SOURCES, AUTHORED},
        {*POLICY_SOURCES, AUTHORED},
        {*FINAL_SOURCES, AUTHORED},
    ):
        raise ValueError("corpus selected sources differ from the approved inventory")
    for row in summary["sources"].values():
        _exact(row, {"samples", "questions", "types"}, "source summary")
        if any(type(row[k]) is not int or row[k] < 1 for k in ("samples", "questions")):
            raise ValueError("corpus source counts must be positive exact integers")
        types = row["types"]
        if (
            not isinstance(types, dict)
            or not types
            or set(types) - set(TYPES)
            or any(type(n) is not int or n < 1 for n in types.values())
            or sum(types.values()) != row["questions"]
        ):
            raise ValueError("corpus source type/question counts disagree")
    languages = summary["language_questions"]
    if (
        not isinstance(languages, dict)
        or not languages
        or set(languages) - {"en", "ko", "ja"}
        or any(type(n) is not int or n < 1 for n in languages.values())
        or sum(languages.values()) != sum(r["questions"] for r in summary["sources"].values())
    ):
        raise ValueError("corpus language question counts disagree")
    return summary


def _validate_quota(plan, summary, split):
    validate_summary(summary)
    settings = plan["settings"]
    expected = {
        source: {
            "samples": settings["natural_sample_quotas"][source][split],
            "questions": settings["natural_sample_quotas"][source][split] * width,
            "types": {kind: settings["natural_sample_quotas"][source][split] * width},
        }
        for source in settings["natural_sample_quotas"]
        for width, kind in [SOURCE_WIDTHS[source]]
    }
    n = settings["authored_per_type"][split]
    expected[AUTHORED] = {"samples": n * 3, "questions": n * 3, "types": dict.fromkeys(TYPES, n)}
    if summary["sources"] != expected:
        raise ValueError(f"corpus whole-sample/source/type quotas differ: {split}")
    langs = {
        "en": sum(
            expected[source]["questions"]
            for source in expected
            if source not in {"massive_ko", "massive_ja"}
        ),
        "ko": expected["massive_ko"]["questions"],
        "ja": expected["massive_ja"]["questions"],
    }
    if summary["language_questions"] != langs:
        raise ValueError(f"corpus source/language question quotas differ: {split}")
    if (
        split == "train"
        and langs["en"] / sum(langs.values()) < settings["minimum_english_question_fraction"]
    ):
        raise ValueError("corpus English question fraction is below its declared minimum")


def whole_epochs(plan, inventory, schedule, *, weights=None):
    validate_plan(plan)
    _exact(schedule, {"steps", "rows_per_step", "seed"}, "epoch schedule")
    if any(type(n) is not int for n in schedule.values()):
        raise ValueError("corpus epoch schedule requires exact integers")
    settings = plan["settings"]
    rows = sum(len(row["rows"]) for row in inventory)
    epochs, batch = settings["epochs"], settings["rows_per_step"]
    if not rows or rows * epochs % batch:
        raise ValueError("whole epochs require integer batches without a repeated tail")
    if weights is not None or schedule != {
        "steps": rows * epochs // batch,
        "rows_per_step": batch,
        "seed": settings["seed"],
    }:
        raise ValueError("corpus requires its exact whole-epoch schedule without weighted sampling")
    visits = Counter(pair for group in scheduled_batches(inventory, **schedule) for pair in group)
    expected = {(i, j) for i, row in enumerate(inventory) for j in range(len(row["rows"]))}
    if set(visits) != expected or set(visits.values()) != {epochs}:
        raise ValueError("every prepared row must be visited exactly once per declared epoch")
    return {
        "epochs": epochs,
        "rows": rows,
        "min_visits": epochs,
        "max_visits": epochs,
        "visits_sha256": fingerprint([[i, j, visits[i, j]] for i, j in sorted(visits)]),
    }


def prepared_groups_sha256(groups):
    """Bind CPU row fields; runtime frozen-base reads have a separate binding."""

    def portable(value):
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, dict):
            return {key: portable(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(v) for v in value]
        return value

    def prepared(item):
        value = asdict(item)
        # Only this runtime attachment is excluded. The runner binds the
        # validated frozen reads separately in its optimizer-state identity.
        value["base_probs"] = None
        return fingerprint(portable(value))

    return fingerprint([[prepared(item) for item in group] for group in groups])


def validate_runtime_replay(recipe, inventory, groups):
    weight = recipe["loss_weights"]["base_replay"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("planned runtime base replay weight must be finite and nonnegative")
    for sample, group in zip(inventory, groups, strict=True):
        for item in group:
            values = item.base_probs
            required = weight > 0 and sample["data_kind"] == "natural"
            if not required:
                if values is not None:
                    raise ValueError(
                        "runtime base replay cannot be attached to unplanned/authored rows"
                    )
            elif (
                not isinstance(values, list)
                or len(values) != len(item.target)
                or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in values)
                or not math.isclose(math.fsum(values), 1, rel_tol=0, abs_tol=1e-6)
            ):
                raise ValueError(
                    "planned natural rows require aligned normalized frozen-base reads"
                )


def validate_contract(plan, splits, commitment, inventory, groups, schedule):
    """Shared prepare/full/fast validator; original test remains opaque."""
    markers = any(MARKER in s.metadata for samples in splits.values() for s in samples)
    extension = commitment.get("corpus_contract")
    if plan is None:
        if markers or extension is not None:
            raise ValueError("partial corpus plan markers require a complete planned contract")
        return None
    anchor = plan_sha256(plan)
    _exact(extension, {"plan_sha256", "summary"}, "opaque test contract")
    if extension["plan_sha256"] != anchor:
        raise ValueError("opaque test corpus plan digest differs")
    summaries = {split: split_summary(samples, anchor) for split, samples in splits.items()}
    if set(summaries) != set(SPLITS) - {"test"}:
        raise ValueError("corpus contract requires all development splits")
    summaries["test"] = extension["summary"]
    for split, summary in summaries.items():
        _validate_quota(plan, summary, split)
    test = summaries["test"]
    counts = commitment["counts"]
    source_samples = test["sources"]
    sample_languages = {
        "en": sum(
            row["samples"]
            for key, row in source_samples.items()
            if key not in {"massive_ko", "massive_ja"}
        ),
        "ko": source_samples["massive_ko"]["samples"],
        "ja": source_samples["massive_ja"]["samples"],
    }
    if (
        sum(r["samples"] for r in test["sources"].values()) != counts["samples"]
        or sum(r["questions"] for r in test["sources"].values()) != counts["questions"]
        or test["membership_sha256"] != commitment["split_sha256"]
        or dict(sum((Counter(r["types"]) for r in test["sources"].values()), Counter()))
        != counts["types"]
        or counts["languages"] != sample_languages
        or counts["modalities"] != {"text": counts["samples"]}
    ):
        raise ValueError("opaque test corpus counts/membership differ from the holdout")
    if len(groups) != len(inventory) or any(
        len(group) != len(row["rows"]) for group, row in zip(groups, inventory, strict=True)
    ):
        raise ValueError("corpus prepared group widths differ from the whole-question inventory")
    if any(item.base_probs is not None for group in groups for item in group):
        raise ValueError("CPU corpus preparation must exclude runtime frozen-base probabilities")
    if len(inventory) != len(splits["train"]) or any(
        row["content_sha256"] != fingerprint(sample.to_json())
        or row["language"] != sample.metadata["language"]
        or [q.type for q in sample.questions] != [q["type"] for q in row["rows"]]
        for sample, row in zip(splits["train"], inventory, strict=True)
    ):
        raise ValueError("corpus inventory differs from its original whole questions")
    return {
        "plan": plan,
        "plan_sha256": anchor,
        "splits": summaries,
        "whole_epochs": whole_epochs(plan, inventory, schedule),
        "inventory_sha256": fingerprint(inventory),
        "prepared_groups_sha256": prepared_groups_sha256(groups),
    }


def recipe_plan(recipe):
    plan = recipe.get("corpus_plan")
    if plan is None:
        if "corpus_plan" in recipe or "corpus_plan_sha256" in recipe or "corpus_contract" in recipe:
            raise ValueError("partial recipe corpus plan fields")
    else:
        validate_recipe(plan, recipe)
        contract = recipe.get("corpus_contract")
        if (
            not isinstance(contract, dict)
            or contract.get("plan") != plan
            or contract.get("plan_sha256") != plan_sha256(plan)
        ):
            raise ValueError("recipe must retain its complete corpus contract")
    return plan


def preparation_binding(plan, cfg, tokenizer_sha256, native_metadata, input_encoding, gold_sources):
    """Match actual preparation assets without filesystem reads."""
    if plan is not None:
        validate_recipe(
            plan,
            {
                "model": asdict(cfg),
                "tokenizer_sha256": tokenizer_sha256,
                "native_metadata": native_metadata,
                "input_encoding": input_encoding,
                "gold_sources": gold_sources,
                "corpus_plan_sha256": plan_sha256(plan),
            },
        )
