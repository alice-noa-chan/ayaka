"""Offline Hotpot builder: full raw closure precedes roles, filters and quotas.

Produces canonical rows consumable by the existing direct preparation pipeline.
Only supplied inventories are closed; no corpus-history independence attestation
is inferred from a successful build or from a caller's exclusion list.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from ..config import model_config
from ..eval.matched_contract import digest, pinned_bytes
from ..eval.read_artifact import fingerprint
from ..input_errors import ContextLimitError
from ..training.swift_direct import encode_direct_sample, input_serving_recipe
from ..training.tokenizer_identity import scoped_tokenizer_preparation
from .decontam import POLICY, Decontaminator, _record_texts
from .paragraph_groups import (
    ROLES,
    audit_paragraph_roles,
    bind_paragraph_sample,
    build_paragraph_index,
    paragraph_inventory_digest,
)
from .schema import Candidate, Question, Sample
from .transforms import hotpot_decision

VERSION = "ayaka-raw-hotpot-splits-1"


def _cells(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("builder requires predeclared nonempty quota cells")
    cells = {}
    for name, n in values.items():
        cell = tuple(name.split("/")) if isinstance(name, str) else ()
        if (
            len(cell) != 3
            or cell[0] not in ROLES
            or not cell[1]
            or cell[2] not in {"choice", "noul"}
            or type(n) is not int
            or n < 1
        ):
            raise ValueError("quota cells require role/source/type and positive integers")
        cells[cell] = n
    return cells


def _validate_plan(plan):
    keys = {
        "version",
        "seed",
        "namespace",
        "source",
        "hf_split",
        "license",
        "roles",
        "minimum_counts",
        "minimum_components",
        "input_encoding",
        "context_limit",
        "blocked_raw_ids",
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != keys
        or plan["version"] != VERSION
        or type(plan["seed"]) is not int
    ):
        raise ValueError("builder needs an exact predeclared plan")
    for key in ("namespace", "source", "hf_split", "license"):
        if not isinstance(plan[key], str) or not plan[key].strip():
            raise ValueError("builder source namespace/split/license must be explicit")
    roles = plan["roles"]
    if (
        not isinstance(roles, dict)
        or not roles
        or set(roles) - ROLES
        or any(type(n) is not int or n < 1 for n in roles.values())
    ):
        raise ValueError("builder requires positive predeclared role weights")
    minima, components = _cells(plan["minimum_counts"]), _cells(plan["minimum_components"])
    if (
        set(minima) != set(components)
        or {c[0] for c in minima} != set(roles)
        or any(c[1] != plan["source"] or components[c] > minima[c] for c in minima)
    ):
        raise ValueError("decision/component minima must cover the same planned roles/source/cells")
    if (
        type(plan["context_limit"]) is not int
        or plan["context_limit"] < 2
        or not isinstance(plan["blocked_raw_ids"], list)
        or any(not isinstance(v, str) or not v for v in plan["blocked_raw_ids"])
        or len(set(plan["blocked_raw_ids"])) != len(plan["blocked_raw_ids"])
    ):
        raise ValueError("builder needs a context limit and unique explicit blocked raw IDs")
    return minima, components


def raw_digest(raw):
    rows = copy.deepcopy(list(raw))
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("raw builder requires original explicit string IDs")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("raw inventory must be nonempty with unique IDs")
    return fingerprint(sorted(rows, key=lambda r: r["id"]))


def _role(component, plan):
    draw = int(fingerprint([VERSION, plan["seed"], component]), 16) % sum(plan["roles"].values())
    for name, weight in sorted(plan["roles"].items()):
        if draw < weight:
            return name
        draw -= weight
    raise AssertionError("role assignment exhausted")


@scoped_tokenizer_preparation
def build_hotpot_splits(raw, plan, tok, cfg, *, expected_raw_sha256, decontaminator):
    """Verify raw input, close ALL contexts, then filter without model predictions."""
    plan, raw = copy.deepcopy(plan), copy.deepcopy(list(raw))
    minima, component_minima = _validate_plan(plan)
    anchor = raw_digest(raw)
    if anchor != expected_raw_sha256:
        raise ValueError("raw inventory differs from its predeclared logical anchor")
    if not set(plan["blocked_raw_ids"]) <= {row["id"] for row in raw}:
        raise ValueError("blocked inventory IDs must exist; include historical bridge contexts")
    if decontaminator is None or not callable(getattr(decontaminator, "sample_hit", None)):
        raise ValueError("builder requires an explicit public decontamination check")
    entry_recipe = input_serving_recipe(tok, plan["input_encoding"])
    inventory, converted = [], {}
    for row in sorted(raw, key=lambda r: r["id"]):
        samples = hotpot_decision(
            row,
            metadata={
                "source": plan["source"],
                "source_example_id": row["id"],
                "label_source": "human",
                "hf_split": plan["hf_split"],
                "license": plan["license"],
                "modality": "text",
                "tier": "hard",
            },
        )
        if len(samples) > 1:
            raise ValueError("builder expects one original decision per raw row")
        kind = samples[0].questions[0].type if samples else None
        inventory.append(
            {"id": row["id"], "source": plan["source"], "type": kind, "context": row["context"]}
        )
        if samples:
            converted[row["id"]] = samples[0]
    index = build_paragraph_index(
        inventory,
        namespace=plan["namespace"],
        expected_inventory_sha256=paragraph_inventory_digest(
            inventory, namespace=plan["namespace"]
        ),
    )
    component_roles = {row.component_id: _role(row.component_id, plan) for row in index.rows}
    raw_by_id = {row["id"]: row for row in raw}
    blocked_components = set()
    for row in index.rows:
        original = raw_by_id[row.id]
        question = Question(
            "raw-public-probe",
            "choice",
            original["question"],
            [Candidate(str(i), title) for i, title in enumerate(original["context"]["title"])],
            {},
        )
        if row.id in plan["blocked_raw_ids"] or decontaminator.sample_hit(
            Sample(row.state, [question])
        ):
            blocked_components.add(row.component_id)
    pools, dropped = defaultdict(lambda: defaultdict(list)), Counter()
    for row in index.rows:
        if row.type is None:
            dropped["unsupported_bridge"] += 1
            continue
        if row.component_id in blocked_components:
            dropped["blocked_or_public_component"] += 1
            continue
        sample = bind_paragraph_sample(index, row.id, converted[row.id])
        if decontaminator.sample_hit(sample):
            # A question-only overlap must also block all siblings, not one row.
            blocked_components.add(row.component_id)
            dropped["public_question_component"] += 1
            continue
        role = component_roles[row.component_id]
        cell = (role, row.source, row.type)
        if cell not in minima:
            dropped["unplanned_cell"] += 1
            continue
        try:
            items = encode_direct_sample(
                sample,
                tok,
                cfg,
                input_encoding=plan["input_encoding"],
                context_limit=plan["context_limit"],
            )
        except ContextLimitError:
            dropped["complete_input_overflow"] += 1
            continue
        if not items:
            raise ValueError("direct encoder produced no decision")
        sample.metadata.update(split=role, source_lineage=row.component_id)
        pools[cell][row.component_id].append((row.id, sample))
    selections = {role: [] for role in sorted(plan["roles"])}
    outputs = {role: [] for role in selections}
    for cell, target in sorted(minima.items()):
        groups = pools[cell]
        queues = [
            deque(sorted(groups[c], key=lambda x: x[0]))
            for c in sorted(groups, key=lambda c: fingerprint([plan["seed"], c]))
            if c not in blocked_components
        ]
        chosen = []
        while queues and len(chosen) < target:
            remaining = []
            for queue in queues:
                if len(chosen) < target:
                    chosen.append(queue.popleft())
                if queue:
                    remaining.append(queue)
            queues = remaining
        if len(chosen) < target:
            raise ValueError(f"{cell}: eligible raw inventory cannot fill predeclared quota")
        for row_id, sample in chosen:
            selections[cell[0]].append(row_id)
            outputs[cell[0]].append({"id": row_id, **sample.to_json()})
    audit = audit_paragraph_roles(
        index, selections, minimum_counts=minima, minimum_components=component_minima
    )
    for rows in outputs.values():
        rows.sort(key=lambda r: r["id"])
    receipt = {
        "version": VERSION,
        "plan": plan,
        "plan_sha256": fingerprint(plan),
        "raw_inventory_sha256": anchor,
        "paragraph_audit": audit,
        "role_assignment_sha256": fingerprint(component_roles),
        "input_recipe": entry_recipe,
        "dropped": dict(sorted(dropped.items())),
        "promotable": False,
        "historical_inventory_complete": False,
        "fresh_independence_attested": False,
        "scope": "supplied raw Hotpot contexts, source-label conversion, public check and full-input eligibility",
    }
    if input_serving_recipe(tok, plan["input_encoding"]) != entry_recipe:
        raise ValueError("input recipe changed during raw preparation; discard outputs")
    return outputs, receipt


def pinned_public_checker(manifest_bytes, root):
    """Consume exactly externally pinned nonempty public JSONL bytes."""
    manifest = json.loads(manifest_bytes)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "format", "policy", "files_sha256"}
        or manifest["version"] != "ayaka-public-exclusion-1"
        or manifest["format"] != "jevbench-singular-question"
        or fingerprint(manifest["policy"]) != fingerprint(POLICY)
        or not isinstance(manifest["files_sha256"], dict)
        or not manifest["files_sha256"]
    ):
        raise ValueError("production builder requires a pinned public inventory and exact policy")
    root = Path(root).resolve()
    texts, count = [], 0
    for name, sha in sorted(manifest["files_sha256"].items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.suffix != ".jsonl":
            raise ValueError("public inventory files must be JSONL inside the declared root")
        data = pinned_bytes(path, sha)
        records = [json.loads(line) for line in data.splitlines() if line.strip()]
        if not records or any(not isinstance(row, dict) for row in records):
            raise ValueError("each public inventory file must contain records")
        count += len(records)
        for row in records:
            question = row.get("question")
            if (
                "questions" in row
                or "state" not in row
                or not isinstance(question, dict)
                or not isinstance(question.get("instructions", question.get("instruction")), str)
                or not question.get("instructions", question.get("instruction", "")).strip()
            ):
                raise ValueError("public inventory requires declared singular JevBench questions")
            criteria = question.get("criteria", {})
            values = criteria.values() if isinstance(criteria, dict) else criteria
            if not isinstance(criteria, (dict, list)) or any(
                not isinstance(v, str) for v in values
            ):
                raise ValueError("public inventory requires text candidate descriptions")
            texts.extend(_record_texts(row))
    checker = Decontaminator(
        texts, n=POLICY["word_ngram"], min_exact_words=POLICY["min_exact_words"]
    )
    if not (checker.grams or checker.exact or checker.cjk_grams or checker.cjk_exact):
        raise ValueError("public inventory supplies no effective exclusion text")
    return checker, {
        "manifest": manifest,
        "manifest_sha256": digest(manifest_bytes),
        "records": count,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "plan", "public-manifest"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--public-root", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("raw split outputs must be fresh; no old split is replaced")
    raw = [
        json.loads(line)
        for line in pinned_bytes(args.raw, args.raw_sha256).splitlines()
        if line.strip()
    ]
    plan = json.loads(pinned_bytes(args.plan, args.plan_sha256))
    _validate_plan(plan)
    checker, public_receipt = pinned_public_checker(
        pinned_bytes(args.public_manifest, args.public_manifest_sha256), args.public_root
    )
    from transformers import AutoTokenizer

    from ..tokenization import HFTokenizer

    cfg = model_config("electra-large")
    if plan["input_encoding"].get("encoder") == "swift_canonical":
        from dataclasses import replace

        cfg = replace(cfg, readout="lm")
    tok = HFTokenizer(
        AutoTokenizer.from_pretrained(
            cfg.backbone, revision=cfg.backbone_revision, local_files_only=True
        ),
        cfg.backbone,
    )
    outputs, receipt = build_hotpot_splits(
        raw,
        plan,
        tok,
        cfg,
        expected_raw_sha256=raw_digest(raw),
        decontaminator=checker,
    )
    files = {
        f"{role}.jsonl": (
            "".join(
                json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                for r in rows
            )
        ).encode()
        for role, rows in outputs.items()
    }
    receipt.update(
        raw_file_sha256=args.raw_sha256,
        plan_file_sha256=args.plan_sha256,
        public_exclusion=public_receipt,
        files_sha256={name: digest(data) for name, data in files.items()},
    )
    args.output.mkdir()
    for name, data in files.items():
        (args.output / name).write_bytes(data)
    (args.output / "manifest.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files_sha256": receipt["files_sha256"],
                "promotable": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
