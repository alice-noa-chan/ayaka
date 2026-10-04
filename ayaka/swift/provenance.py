"""Shared strict role, pairing and leakage checks for Swift fitting/adoption."""

from __future__ import annotations

import math

from ayaka.eval.read_artifact import fingerprint

from .binding import validate_bound_reads
from .prompt import validate_prompt_variant

MATCH_FIELDS = (
    "type",
    "tier",
    "labels",
    "gold",
    "gold_distribution",
    "case_id",
    "cluster_id",
    "split",
    "readout",
    "metadata",
    "family",
)


def row_key(row: dict) -> tuple[str, str]:
    return str(row.get("source", "")), str(row["id"])


def group_reads(
    rows: list[dict], split: str, *, exploratory=False, require_variants=True
) -> dict[str, list[dict]]:
    """Validate provenance and align by source/id, independently of file order."""
    if not rows:
        raise ValueError(f"empty {split} reads")
    if split not in ("calibration", "dev"):
        raise ValueError("selector role must be calibration or dev")
    groups: dict[str, dict[tuple[str, str], dict]] = {}
    for row in rows:
        source_parts = str(row.get("source", "")).replace("\\", "/").lower().split("/")
        metadata = row.get("metadata") or {}
        if (
            row.get("public") is not False
            or row.get("split", metadata.get("split")) == "public"
            or any(part in ("public", "jevbench_public") for part in source_parts)
        ):
            raise ValueError("REFUSING public or unmarked reads; require explicit public=False")
        if "split" in row and "split" in metadata and row["split"] != metadata["split"]:
            raise ValueError("row and metadata split conflict")
        if row.get("split", metadata.get("split")) != split:
            raise ValueError(f"selector role {split} must match recorded split exactly")
        if row.get("readout") == "grouped_approx":
            if not exploratory:
                raise ValueError("grouped diagnostic reads require --exploratory")
            continue
        variant = row.get("prompt_variant")
        validate_prompt_variant(variant)
        if not row.get("model") or not row.get("revision"):
            raise ValueError("reads must record model and revision")
        tokens = row.get("input_tokens", 0)
        if not isinstance(tokens, (int, float)) or not math.isfinite(tokens) or tokens <= 0:
            raise ValueError("reads need finite positive input_tokens")
        key = row_key(row)
        group = groups.setdefault(variant, {})
        if key in group:
            raise ValueError(f"duplicate {split} source/id for {variant}: {key}")
        group[key] = row
    if require_variants and ("min" not in groups or len(groups) < 2):
        raise ValueError(f"{split} requires min and at least one other variant")
    reference = "min" if "min" in groups else next(iter(groups))
    keys = sorted(groups[reference])
    for variant, group in groups.items():
        if set(group) != set(keys):
            raise ValueError(f"{split} variants must have the SAME source/id rows: {variant}")
        for key in keys:
            if any(
                group[key].get(field) != groups[reference][key].get(field) for field in MATCH_FIELDS
            ):
                raise ValueError(f"{split} row metadata differs across variants: {key}")
    aligned = {variant: [group[key] for key in keys] for variant, group in sorted(groups.items())}
    if not exploratory:
        for group in aligned.values():
            validate_bound_reads(group)
    return aligned


def assert_roles_isolated(calibration, dev):
    """Check every variant's global source lineage and rendered input before fitting."""

    def keys(rows):
        found = set()
        for row in rows:
            source = str(row.get("source", ""))
            if not source:
                raise ValueError("selector needs an explicit source namespace")
            for value in (row.get("case_id"), row.get("cluster_id"), *row.get("lineage_ids", [])):
                if value:
                    value = str(value)
                    found.add(("lineage", value if ":" in value else f"{source}:{value}"))
            found.add(("question", source, row["id"]))
            binding = row.get("binding") or {}
            input_hash = binding.get("rendered_input_sha256") or row.get("rendered_input_sha256")
            if not input_hash:
                raise ValueError("selector requires rendered-input hash for overlap checks")
            found.add(("input", input_hash))
            for value in binding.get("token_inputs", []):
                found.add(("tokens", value["input_token_ids_sha256"]))
            for value in row.get("pass_bindings", []):
                if value.get("messages"):
                    found.add(("input", fingerprint(value["messages"])))
                if value.get("input_token_ids"):
                    found.add(("tokens", fingerprint(value["input_token_ids"])))
        return found

    overlap = keys(calibration) & keys(dev)
    if overlap:
        raise ValueError(
            f"calibration/dev case, lineage or rendered-input overlap: {sorted(overlap)[0]}"
        )
