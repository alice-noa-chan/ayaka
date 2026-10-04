"""Externally anchored CPU regeneration receipts and portable prepared-row loads.

The receipt moves expensive rendering/gold regeneration before paid allocation.
It requires a separately trusted digest; it is not execution attestation or
isolation against concurrent mutation. Actual native kernel parity still runs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from ..collate import EncodedQuestion
from ..config import ElectraConfig
from ..data.direct_natural import verify_raw_binding
from ..data.schema import Sample
from ..eval.read_artifact import fingerprint
from ..prompt import RenderedQuestion
from . import direct_bundle as bundles
from .batching import TrainItem, _noul_canonical
from .direct_corpus_plan import recipe_plan, validate_contract
from .direct_holdout import DEVELOPMENT_SPLITS, validate_commitment
from .optimization import OptimizationConfig
from .prepare_v2 import audit_splits, canonical, sha256
from .swift_direct import input_serving_recipe, validate_direct_input_items
from .tokenizer_identity import configuration_fingerprint
from .workload import describe_rows, finite_workload

VERSION = "ayaka-direct-cpu-audit-1"


def _digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be an externally pinned SHA256")
    return value


def _payloads(path, expected_manifest_sha256=None):
    root = Path(path)
    raw = (root / "manifest.json").read_bytes()
    if expected_manifest_sha256 is not None and sha256(raw) != _digest(
        expected_manifest_sha256, "bundle manifest"
    ):
        raise ValueError("bundle manifest differs from the external audit anchor")
    manifest = json.loads(raw)
    if (
        manifest.get("version") != bundles.VERSION
        or manifest.get("status") != bundles.STATUS
        or manifest.get("promotable") is not False
        or manifest.get("execution_attested") is not False
        or type(manifest.get("optimizer_steps_executed")) is not int
        or manifest["optimizer_steps_executed"] != 0
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != bundles.FILES
        or (root / "test.jsonl").exists()
    ):
        raise ValueError("require an unpromoted development bundle without original test inputs")
    payloads = {name: (root / name).read_bytes() for name in sorted(bundles.FILES)}
    if {name: sha256(value) for name, value in payloads.items()} != manifest["files"]:
        raise ValueError("direct bundle payload differs from its audit anchor")
    return raw, manifest, payloads


def _dependencies():
    # CPU and CUDA torch build suffixes differ; kernel/runtime parity is separate.
    return {
        name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "peft")
    }


def _tokenizer_binding(tok, cfg):
    return {
        "tokenizer_sha256": bundles._tokenizer_identity(tok, cfg),
        "configuration_sha256": configuration_fingerprint(tok),
    }


def _splits(payloads):
    return {
        split: [
            Sample.from_json(json.loads(line))
            for line in payloads[f"{split}.jsonl"].splitlines()
            if line.strip()
        ]
        for split in DEVELOPMENT_SPLITS
    }


def _aliases(groups):
    result = []
    for group in groups:
        seen = {}
        result.append([seen.setdefault(id(item.enc.prefix_ids), i) for i, item in enumerate(group)])
    return result


@dataclass
class AuditedBundle:
    manifest: dict
    recipe: dict
    items: list
    inventory: list
    groups: list
    splits: dict
    binding: dict
    tokenizer: object


def audit_snapshot(
    path,
    tok=None,
    *,
    allow_tiny=False,
    natural_registry=None,
    expected_manifest_sha256=None,
    native_path=None,
):
    """Full regeneration with stable payload/source/tokenizer snapshots, CPU only."""
    before = _payloads(path, expected_manifest_sha256)
    recipe = json.loads(before[2]["recipe.json"])
    cfg = ElectraConfig(**recipe["model"])
    bundles._model_policy(cfg, allow_tiny=allow_tiny)
    native_root = bundles.bound_native_root(cfg, recipe, native_path=native_path)
    tok = (
        tok
        if tok is not None
        else bundles.local_tokenizer(
            cfg,
            allow_tiny=allow_tiny,
            native_path=native_root,
            expected_metadata=recipe["native_metadata"],
        )
    )
    sources, dependencies = bundles._source_hashes(), _dependencies()
    tokenizer = _tokenizer_binding(tok, cfg)
    manifest, recipe, items, inventory, groups = bundles.audit_bundle(
        path,
        tok,
        allow_tiny=allow_tiny,
        natural_registry=natural_registry,
        expected_manifest_sha256=sha256(before[0]),
        native_path=native_root,
    )
    if (
        _payloads(path, sha256(before[0])) != before
        or bundles._source_hashes() != sources
        or _dependencies() != dependencies
        or _tokenizer_binding(tok, cfg) != tokenizer
        or bundles._item_bytes(items) != before[2]["train_items.jsonl"]
    ):
        raise ValueError("bundle/source/tokenizer changed during CPU audit; discard receipt")
    bundles.bound_native_root(cfg, recipe, native_path=native_root)
    if natural_registry is not None:
        natural_registry.verify_files()
    verify_raw_binding(recipe["gold_sources"], registry=natural_registry)
    binding = {
        "version": VERSION,
        "bundle_manifest_sha256": sha256(before[0]),
        "source_sha256": sources,
        "dependencies": dependencies,
        "tokenizer": tokenizer,
        "prefix_aliases": _aliases(groups),
        "inventory_sha256": fingerprint(inventory),
        "workload": finite_workload(inventory, **recipe["schedule"]),
        "rows": len(items),
        "regeneration_complete": True,
        "optimizer_steps": 0,
        "model_weights_loaded": False,
        "execution_attested": False,
        "promotable": False,
    }
    preparation = json.loads(before[2]["preparation.json"])
    if "corpus_contract" in preparation:
        binding["corpus_contract"] = preparation["corpus_contract"]
    return AuditedBundle(
        manifest, recipe, items, inventory, groups, _splits(before[2]), binding, tok
    )


def create_audit_receipt(
    path,
    out,
    tok=None,
    *,
    expected_manifest_sha256,
    allow_tiny=False,
    natural_registry=None,
    native_path=None,
    publication_guard=None,
):
    """Write after full CPU verification; pin the returned digest outside the bundle."""
    _digest(expected_manifest_sha256, "bundle manifest")
    destination, root = Path(out).resolve(), Path(path).resolve()
    holdout = root.with_name(root.name + "-holdout")
    if destination.exists() or any(
        directory == destination or directory in destination.parents
        for directory in (root, holdout)
    ):
        raise ValueError("audit receipt must be a new file outside bundle and default holdout")
    snapshot = audit_snapshot(
        path,
        tok,
        allow_tiny=allow_tiny,
        natural_registry=natural_registry,
        expected_manifest_sha256=expected_manifest_sha256,
        native_path=native_path,
    )
    cfg = ElectraConfig(**snapshot.recipe["model"])
    native_root = bundles.bound_native_root(cfg, snapshot.recipe, native_path=native_path)
    if native_root is not None and (
        destination == native_root or native_root in destination.parents
    ):
        raise ValueError("audit receipt must be outside the native loader directory")
    raw = canonical(snapshot.binding) + b"\n"
    if natural_registry is not None:
        natural_registry.verify_files()
    verify_raw_binding(snapshot.recipe["gold_sources"], registry=natural_registry)
    if publication_guard is not None:
        publication_guard()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)
    return {"audit_receipt_sha256": sha256(raw), **snapshot.binding}


def _items(raw):
    items = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        encoded = value.pop("enc")
        rendered = encoded.pop("rendered")
        rendered["option_spans"] = [tuple(span) for span in rendered["option_spans"]]
        value["enc"] = EncodedQuestion(rendered=RenderedQuestion(**rendered), **encoded)
        value["ordinals"] = [Decimal(x) if isinstance(x, str) else x for x in value["ordinals"]]
        item = TrainItem(**value)
        if (
            not item.direct_distillation
            or item.base_probs is not None
            or any(
                getattr(item, key) is not None
                for key in (
                    "reasoning_positions",
                    "reasoning_labels",
                    "proposal_input_ids",
                    "proposal_positions",
                    "proposal_labels",
                    "native_inputs",
                )
            )
        ):
            raise ValueError(
                "audited direct rows must exclude replay/trace/proposal/image payloads"
            )
        items.append(item)
    if not items or bundles._item_bytes(items) != raw:
        raise ValueError("prepared direct rows must round-trip exactly without coercion")
    validate_direct_input_items(items)
    return items


def load_audited_bundle(
    path,
    receipt,
    tok=None,
    *,
    expected_manifest_sha256,
    expected_receipt_sha256,
    allow_tiny=False,
    native_path=None,
):
    """Read only exact externally audited buffers, without rendering or raw-source loads."""
    _digest(expected_manifest_sha256, "bundle manifest")
    _digest(expected_receipt_sha256, "audit receipt")
    raw_receipt = Path(receipt).read_bytes()
    if sha256(raw_receipt) != expected_receipt_sha256:
        raise ValueError("CPU audit receipt differs from the externally pinned digest")
    binding = json.loads(raw_receipt)
    if (
        binding.get("version") != VERSION
        or binding.get("regeneration_complete") is not True
        or type(binding.get("optimizer_steps")) is not int
        or binding["optimizer_steps"] != 0
        or binding.get("model_weights_loaded") is not False
        or binding.get("execution_attested") is not False
        or binding.get("promotable") is not False
        or binding.get("bundle_manifest_sha256") != expected_manifest_sha256
        or binding.get("source_sha256") != bundles._source_hashes()
        or binding.get("dependencies") != _dependencies()
    ):
        raise ValueError("CPU audit receipt is stale, unbound or falsely promoted")
    _, manifest, payloads = _payloads(path, expected_manifest_sha256)
    if manifest["source_sha256"] != binding["source_sha256"]:
        raise ValueError("bundle training source differs from the CPU audit")
    recipe = json.loads(payloads["recipe.json"])
    corpus_plan = recipe_plan(recipe)
    cfg = ElectraConfig(**recipe["model"])
    if recipe["allow_tiny"] and not allow_tiny:
        raise ValueError("audited tiny bundles require explicit mechanics-only mode")
    if recipe["model_policy"] != bundles._model_policy(cfg, allow_tiny=allow_tiny):
        raise ValueError("model size/license policy differs from the CPU audit")
    native_root = bundles.bound_native_root(cfg, recipe, native_path=native_path)
    architecture = bundles.inspect_direct_model(
        cfg,
        official_weight_elements=recipe["model_policy"].get("backbone_weight_elements"),
        optimizations=OptimizationConfig(**recipe["optimizations"]),
        native_path=native_root,
        expected_config=recipe["native_architecture"]["native_config"],
    )
    if architecture != recipe["native_architecture"]:
        raise ValueError("actual native architecture differs from the CPU audit")
    tok = (
        tok
        if tok is not None
        else bundles.local_tokenizer(
            cfg,
            allow_tiny=allow_tiny,
            native_path=native_root,
            expected_metadata=recipe["native_metadata"],
        )
    )
    if _tokenizer_binding(tok, cfg) != binding["tokenizer"] or (
        input_serving_recipe(tok, recipe["input_encoding"]) != recipe["input_recipe"]
    ):
        raise ValueError("actual tokenizer or serving recipe differs from the CPU audit")
    splits, items = _splits(payloads), _items(payloads["train_items.jsonl"])
    audit_splits(splits, development_only=True)
    commitment = validate_commitment(json.loads(payloads["test_commitment.json"]), splits)
    inventory, groups, offset = [], [], 0
    for sample, aliases in zip(splits["train"], binding["prefix_aliases"], strict=True):
        group = items[offset : offset + len(sample.questions)]
        if len(aliases) != len(group) or len(group) != len(sample.questions):
            raise ValueError("audited direct group/alias counts differ")
        for original, item in zip(sample.questions, group, strict=True):
            q = _noul_canonical(original)
            if (
                item.sample_id != sample.metadata["source_example_id"]
                or item.type != q.type
                or item.target != [q.target_distribution.get(c.id, 0.0) for c in q.candidates]
                or item.ordinals
                != [c.ordinal if c.ordinal is not None else i for i, c in enumerate(q.candidates)]
                or recipe["input_encoding"]["encoder"] == "swift_canonical"
                and (
                    item.direct_input_binding is None
                    or item.direct_input_binding["recipe"] != recipe["input_recipe"]
                    or item.direct_input_binding["question_id"] != q.id
                    or item.direct_input_binding["candidate_ids"] != [c.id for c in q.candidates]
                )
            ):
                raise ValueError("audited direct candidate/gold/recipe mapping differs")
        for i, alias in enumerate(aliases):
            if type(alias) is not int or not 0 <= alias <= i or aliases[alias] != alias:
                raise ValueError("invalid audited prefix alias")
            if group[i].enc.prefix_ids != group[alias].enc.prefix_ids:
                raise ValueError("audited prefix alias changed input tokens")
            group[i].enc.prefix_ids = group[alias].enc.prefix_ids
        inventory.append(describe_rows(sample, group))
        groups.append(group)
        offset += len(group)
    if (
        offset != len(items)
        or binding["rows"] != len(items)
        or fingerprint(inventory) != binding["inventory_sha256"]
        or finite_workload(inventory, **recipe["schedule"]) != binding["workload"]
        or json.loads(payloads["preparation.json"])["workload"] != binding["workload"]
    ):
        raise ValueError("audited rows, inventory or complete schedule differ")
    contract = validate_contract(
        corpus_plan, splits, commitment, inventory, groups, recipe["schedule"]
    )
    preparation = json.loads(payloads["preparation.json"])
    if contract is None:
        if "corpus_contract" in binding or "corpus_contract" in preparation:
            raise ValueError("partial audited corpus contract")
    elif (
        binding.get("corpus_contract") != contract
        or preparation.get("corpus_contract") != contract
        or recipe.get("corpus_contract") != contract
    ):
        raise ValueError("audited corpus contract differs from original sources and whole epochs")
    bundles.bound_native_root(cfg, recipe, native_path=native_root)
    return AuditedBundle(manifest, recipe, items, inventory, groups, splits, binding, tok)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--mechanics-only", action="store_true")
    parser.add_argument("--native-path", type=Path)
    parser.add_argument("--contractnli-train", type=Path)
    args = parser.parse_args(argv)
    from ..data.direct_natural import explicit_policy_registry

    result = create_audit_receipt(
        args.bundle,
        args.out,
        expected_manifest_sha256=args.expected_manifest_sha256,
        allow_tiny=args.mechanics_only,
        native_path=args.native_path,
        natural_registry=explicit_policy_registry(args.contractnli_train),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
