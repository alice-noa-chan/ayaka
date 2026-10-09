"""Plan and prepare a whole-epoch direct corpus from cached human annotations.

CPU/offline only. No model weights, generator, optimizer, GPU allocation or
download is available here. Optional public ContractNLI train annotations add
full-document policy supervision; teacher benefit and v2 quality are unproven.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..config import AyakaConfig
from ..data.contract_nli import SOURCE as CONTRACT_SOURCE
from ..data.contract_nli import ContractGoldRegistry
from ..data.decontam import Decontaminator, jevbench_public_dir
from ..data.direct_natural import NaturalGoldRegistry
from ..data.natural_training_v2 import EXTRA_SOURCES, partition_sources
from ..data.reasoning_v2 import SPLITS, curriculum
from ..data.reserved_evidence import ReservedEvidenceBlocker
from ..data.schema import Sample
from ..data.source_groups import connected_groups
from ..eval.read_artifact import fingerprint
from ..input_errors import ContextLimitError
from ..losses import LossWeights
from .direct_audit import create_audit_receipt
from .direct_bundle import _tokenizer_identity, local_tokenizer, prepare_bundle
from .direct_corpus_plan import (
    FINAL_VERSION,
    MARKER,
    POLICY_SELECTION,
    POLICY_VERSION,
    SELECTION,
    VERSION,
    digest,
    plan_sha256,
    preparation_binding,
    validate_plan,
    validate_settings,
)
from .native_metadata import inspect_metadata, verify_metadata
from .optimization import OptimizationConfig
from .prepare_v2 import canonical, sha256
from .swift_direct import encode_direct_sample, normalize_input_encoding
from .tokenizer_identity import tokenizer_identity_scope

RESERVED_VERSION = "ayaka-direct-reserved-inputs-1"


class FileGuard:
    """Bind literal file bytes at the entrance and before every publication."""

    def __init__(self, path, expected_sha256=None):
        self.path = Path(path).resolve()
        self.raw = self.path.read_bytes()
        self.sha256 = sha256(self.raw)
        self._identity = fingerprint([str(self.path), self.sha256])
        if expected_sha256 is not None and digest(expected_sha256, "external file") != self.sha256:
            raise ValueError("corpus input file differs from its externally pinned byte digest")

    def verify(self):
        if (
            fingerprint([str(self.path), self.sha256]) != self._identity
            or sha256(self.path.read_bytes()) != self.sha256
        ):
            raise ValueError("corpus input file changed before publication")


class ReservedInventory:
    """Externally anchored, explicit private evaluation file list; no discovery.

    The entrance SHA pins literal manifest bytes. The plan's manifest_sha256
    hashes its logical file/count inventory without paths, allowing relocation.
    Source IDs or original evaluation texts never enter the plan/receipt.
    """

    def __init__(self, path=None, expected_sha256=None, *, draft=False):
        if type(draft) is not bool or (
            draft
            and (path is not None or expected_sha256 is not None)
            or not draft
            and (path is None or expected_sha256 is None)
        ):
            raise ValueError("supply an anchored private manifest or explicit draft scope")
        self.manifest, self.files, self.samples = None, [], []
        if draft:
            self.binding = {
                "scope": "draft_without_prior_private_inventory",
                "manifest_sha256": None,
                "files": {},
            }
        else:
            self.manifest = FileGuard(path, expected_sha256)
            value = json.loads(self.manifest.raw)
            if (
                not isinstance(value, dict)
                or set(value) != {"version", "files"}
                or value["version"] != RESERVED_VERSION
                or not isinstance(value["files"], list)
                or not value["files"]
            ):
                raise ValueError("reserved manifest requires a nonempty explicit file inventory")
            counts, paths = {}, set()
            for row in value["files"]:
                if (
                    not isinstance(row, dict)
                    or set(row) != {"path", "sha256"}
                    or not isinstance(row["path"], str)
                    or not row["path"].strip()
                ):
                    raise ValueError("reserved inventory requires literal path and byte SHA256")
                candidate = Path(row["path"])
                if not candidate.is_absolute():
                    candidate = self.manifest.path.parent / candidate
                guard = FileGuard(candidate, row["sha256"])
                if guard.path in paths or guard.sha256 in counts:
                    raise ValueError("reserved inventory cannot repeat a path or file payload")
                paths.add(guard.path)
                samples = [
                    Sample.from_json(json.loads(line))
                    for line in guard.raw.splitlines()
                    if line.strip()
                ]
                if not samples:
                    raise ValueError("reserved evaluation files must contain original samples")
                counts[guard.sha256] = len(samples)
                self.files.append(guard)
                self.samples.extend(samples)
            self.binding = {
                "scope": "externally_anchored_private_inventory",
                "manifest_sha256": fingerprint({"version": RESERVED_VERSION, "files": counts}),
                "files": counts,
            }
        self._samples_sha256 = fingerprint([s.to_json() for s in self.samples])
        self._binding_sha256 = fingerprint(self.binding)
        self._guards_sha256 = self._guards_identity()
        self.verify()

    def _guards_identity(self):
        return fingerprint(
            {
                "manifest": [str(self.manifest.path), self.manifest.sha256]
                if self.manifest
                else None,
                "files": [[str(guard.path), guard.sha256] for guard in self.files],
            }
        )

    def verify(self):
        if self._guards_identity() != self._guards_sha256:
            raise ValueError("reserved file inventory changed before publication")
        if self.manifest is not None:
            self.manifest.verify()
        for guard in self.files:
            guard.verify()
        if (
            fingerprint([s.to_json() for s in self.samples]) != self._samples_sha256
            or fingerprint(self.binding) != self._binding_sha256
        ):
            raise ValueError("parsed reserved inventory changed before publication")


def public_inventory():
    result = {
        p.name: sha256(p.read_bytes()) for p in sorted(Path(jevbench_public_dir()).glob("*.jsonl"))
    }
    if not result:
        raise ValueError("offline public benchmark inventory is missing")
    return result


def create_plan(settings, cfg, tok, registry, input_encoding, reserved, *, native_path=None):
    """Bind inputs before any source selection; computed results stay outside."""
    validate_settings(settings)
    encoding = normalize_input_encoding(input_encoding)
    metadata, root = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=native_path)
    registry.verify_files()
    if registry.source_names != set(settings["natural_sample_quotas"]):
        raise ValueError("corpus plan requires all pinned human source inventories")
    reserved.verify()
    plan = {
        "version": FINAL_VERSION
        if set(registry.source_names) & set(EXTRA_SOURCES)
        else POLICY_VERSION
        if CONTRACT_SOURCE in registry.source_names
        else VERSION,
        "settings": copy.deepcopy(settings),
        "selection_policy": copy.deepcopy(
            POLICY_SELECTION if CONTRACT_SOURCE in registry.source_names else SELECTION
        ),
        "assets": {
            "model_sha256": fingerprint(asdict(cfg)),
            "tokenizer_sha256": _tokenizer_identity(tok, cfg),
            "native_metadata_sha256": fingerprint(metadata),
            "input_encoding_sha256": fingerprint(encoding),
            "gold_sources_sha256": fingerprint(registry.binding),
            "public_files": public_inventory(),
            "reserved": copy.deepcopy(reserved.binding),
        },
    }
    verify_metadata(metadata, cfg.backbone, cfg.backbone_revision, path=root)
    verify_plan_assets(plan, cfg, tok, registry, encoding, reserved, native_path=root)
    return validate_plan(plan)


def verify_plan_assets(plan, cfg, tok, registry, input_encoding, reserved, *, native_path=None):
    registry.verify_files()
    reserved.verify()
    if (
        reserved.binding != plan["assets"]["reserved"]
        or public_inventory() != plan["assets"]["public_files"]
    ):
        raise ValueError("public/reserved corpus inventory changed from its input plan")
    metadata, root = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=native_path)
    preparation_binding(
        plan, cfg, _tokenizer_identity(tok, cfg), metadata, input_encoding, registry.binding
    )
    verify_metadata(metadata, cfg.backbone, cfg.backbone_revision, path=root)


def permuted_sample(sample, plan):
    result = copy.deepcopy(sample)
    for q in result.questions:
        seed = fingerprint([plan["settings"]["seed"], result.metadata["source_example_id"], q.id])
        random.Random(seed).shuffle(q.candidates)
    result.metadata[MARKER] = plan_sha256(plan)
    return result


def select_corpus(plan, cfg, tok, registry, input_encoding, reserved):
    """Verify human labels, then select whole samples on final native inputs."""
    validate_plan(plan)
    registry.verify_files()
    if registry.source_names != set(plan["settings"]["natural_sample_quotas"]):
        raise ValueError("corpus registry sources differ from the declared input plan")
    sources = registry.sources()
    verified, soft = Counter(), Counter()
    for source, samples in sources.items():
        for sample in samples:
            for q in sample.questions:
                if registry(sample, q) != q.target_distribution:
                    raise ValueError("converted target differs from its original human annotation")
                verified[source] += 1
                soft[source] += int(any(0 < p < 1 for p in q.target_distribution.values()))
    registry.verify_files()
    splits = {}
    for split in SPLITS:
        splits[split] = []
        for sample, traces in curriculum(split, plan["settings"]["authored_per_type"][split]):
            sample.metadata.update(
                source_lineage=sample.metadata["case_facts_sha256"],
                modality="text",
                verified_traces=traces,
            )
            splits[split].append(permuted_sample(sample, plan))
    authored = [s for rows in splits.values() for s in rows]
    indices, _ = connected_groups(authored + reserved.samples)
    blocked = set(indices[len(authored) :])
    public = Decontaminator.from_jevbench()
    private = (
        ReservedEvidenceBlocker(reserved.samples)
        if plan["version"] in (POLICY_VERSION, FINAL_VERSION)
        else None
    )
    if any(
        index in blocked
        or public.sample_hit(sample)
        or (private is not None and private.state_hit(sample))
        for sample, index in zip(authored, indices[: len(authored)], strict=True)
    ):
        raise ValueError("authored control overlaps reserved/public evidence; choose a new corpus")

    def fits(sample, split):
        try:
            encode_direct_sample(
                sample,
                tok,
                cfg,
                input_encoding=input_encoding,
                context_limit=cfg.max_seq_len
                if split == "train"
                else max(cfg.max_seq_len, cfg.serve_max_seq_len),
            )
        except ContextLimitError:
            return False
        return True

    with tokenizer_identity_scope(tok):
        if any(not fits(s, split) for split, rows in splits.items() for s in rows):
            raise ValueError("authored control does not fit whole native input; refuse truncation")
        natural, selection = partition_sources(
            sources,
            authored + reserved.samples,
            fits=lambda _: True,
            fits_by_split=lambda s, split: fits(permuted_sample(s, plan), split),
            quotas=plan["settings"]["natural_sample_quotas"],
            reserved_evidence=private,
        )
    for split in SPLITS:
        splits[split].extend(permuted_sample(s, plan) for s in natural[split])
    registry.verify_files()
    return splits, {
        "selection": selection,
        "raw_verified_questions": dict(verified),
        "raw_soft_label_questions": dict(soft),
        "selected_policy_labels": {
            split: dict(
                Counter(
                    label
                    for s in rows
                    if s.metadata["source"] == CONTRACT_SOURCE
                    for q in s.questions
                    for label, value in q.target_distribution.items()
                    if value == 1
                )
            )
            for split, rows in splits.items()
            if split != "test"
        }
        if CONTRACT_SOURCE in registry.source_names
        else {},
    }


def prepare_corpus(
    out,
    plan,
    cfg,
    tok,
    registry,
    input_encoding,
    reserved,
    *,
    native_path=None,
    allow_tiny=False,
    optimizations=None,
    input_guards=(),
):
    root = Path(out)
    if root.exists():
        raise ValueError("preserve existing records; corpus output must be a new directory")
    encoding = normalize_input_encoding(input_encoding)
    anchor = plan_sha256(plan)
    metadata, native_root = inspect_metadata(cfg.backbone, cfg.backbone_revision, path=native_path)

    def guard():
        if plan_sha256(plan) != anchor:
            raise ValueError("input corpus plan changed during preparation")
        for value in input_guards:
            value.verify()
        verify_plan_assets(plan, cfg, tok, registry, encoding, reserved, native_path=native_root)
        verify_metadata(metadata, cfg.backbone, cfg.backbone_revision, path=native_root)

    guard()
    with tokenizer_identity_scope(tok):
        splits, selection = select_corpus(plan, cfg, tok, registry, encoding, reserved)
    # End serialization reuse before publishing any bundle or receipt.
    guard()
    settings = plan["settings"]
    questions = sum(len(s.questions) for s in splits["train"])
    if questions * settings["epochs"] % settings["rows_per_step"]:
        raise ValueError("whole corpus epochs must fit integer batches")
    manifest = prepare_bundle(
        root / "bundle",
        splits,
        tok,
        cfg,
        {},
        steps=questions * settings["epochs"] // settings["rows_per_step"],
        rows_per_step=settings["rows_per_step"],
        seed=settings["seed"],
        weights=LossWeights(gold_nll_with_teacher=True, distill=0, pointer_aux=0),
        allow_tiny=allow_tiny,
        optimizations=optimizations,
        natural_registry=registry,
        input_encoding=encoding,
        native_path=native_root,
        corpus_plan=plan,
        preparation_guard=guard,
    )
    bundle_anchor = sha256((root / "bundle/manifest.json").read_bytes())
    guard()
    audit = create_audit_receipt(
        root / "bundle",
        root / "cpu-audit.json",
        tok,
        expected_manifest_sha256=bundle_anchor,
        allow_tiny=allow_tiny,
        natural_registry=registry,
        native_path=native_root,
        publication_guard=guard,
    )
    report = {
        "version": "ayaka-direct-corpus-selection-1",
        "plan_sha256": anchor,
        **selection,
        "corpus_contract": audit["corpus_contract"],
        "bundle_manifest_sha256": bundle_anchor,
        "audit_receipt_sha256": audit["audit_receipt_sha256"],
        "holdout_manifest_sha256": sha256((root / "bundle-holdout/manifest.json").read_bytes()),
        "source_sha256": manifest["source_sha256"],
        "optimizer_steps": 0,
        "model_weights_loaded": False,
        "gpu_job_started": False,
        "paid_execution_started": False,
        "promotable": False,
        "scope": __doc__,
    }
    raw = canonical(report) + b"\n"
    guard()
    (root / "selection.json").write_bytes(raw)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--out", required=True, type=Path)
        sub.add_argument("--native-path", type=Path)
        sub.add_argument(
            "--contractnli-train",
            type=Path,
            help="pinned original train.json with sibling LICENSE; enables corpus plan2",
        )
        sub.add_argument(
            "--extra-natural",
            action="append",
            choices=sorted(EXTRA_SOURCES),
            default=[],
            help="opt-in extra human source; with --contractnli-train enables corpus plan3",
        )
        sub.add_argument("--mechanics-only", action="store_true")
        private = sub.add_mutually_exclusive_group(required=True)
        private.add_argument("--reserved-manifest", type=Path)
        private.add_argument("--draft-without-prior-reserved", action="store_true")
        sub.add_argument("--expected-reserved-manifest-sha256")
        sub.add_argument(
            "--input-encoder",
            choices=("ayaka_segmented", "swift_canonical"),
            default="swift_canonical",
        )
        sub.add_argument(
            "--prompt-variant", choices=("min", "cygnet", "rules", "labeled"), default="labeled"
        )
        sub.add_argument("--state-format", choices=("pretty", "compact"), default="compact")
        if command == "plan":
            sub.add_argument("--settings", required=True, type=Path)
        else:
            sub.add_argument("--plan", required=True, type=Path)
            sub.add_argument("--expected-plan-file-sha256", required=True)
            sub.add_argument(
                "--attention", choices=("native", "sdpa", "flash_attention_2"), default="native"
            )
            sub.add_argument("--liger", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("choose a new output; existing corpus records must remain intact")
    cfg_file = FileGuard(args.config)
    cfg = AyakaConfig(**json.loads(cfg_file.raw))
    if args.command == "prepare":
        plan_file = FileGuard(args.plan, args.expected_plan_file_sha256)
        plan = validate_plan(json.loads(plan_file.raw))
    else:
        settings_file = FileGuard(args.settings)
        settings = validate_settings(json.loads(settings_file.raw))
    reserved = ReservedInventory(
        args.reserved_manifest,
        args.expected_reserved_manifest_sha256,
        draft=args.draft_without_prior_reserved,
    )
    encoding = {"encoder": args.input_encoder}
    if args.input_encoder == "swift_canonical":
        encoding.update(prompt_variant=args.prompt_variant, state_format=args.state_format)
    encoding = normalize_input_encoding(encoding)
    tok = local_tokenizer(cfg, allow_tiny=args.mechanics_only, native_path=args.native_path)
    registry = NaturalGoldRegistry(
        policy_registry=ContractGoldRegistry(args.contractnli_train)
        if args.contractnli_train is not None
        else None,
        extra_sources=args.extra_natural,
    )
    if args.command == "plan":
        plan = create_plan(
            settings, cfg, tok, registry, encoding, reserved, native_path=args.native_path
        )
        raw = canonical(plan) + b"\n"
        cfg_file.verify()
        settings_file.verify()
        verify_plan_assets(
            plan, cfg, tok, registry, encoding, reserved, native_path=args.native_path
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            stream.write(raw)
        result = {"plan_file_sha256": sha256(raw), "plan_sha256": plan_sha256(plan)}
    else:
        result = prepare_corpus(
            args.out,
            plan,
            cfg,
            tok,
            registry,
            encoding,
            reserved,
            native_path=args.native_path,
            allow_tiny=args.mechanics_only,
            optimizations=OptimizationConfig(attention=args.attention, liger=args.liger),
            input_guards=(cfg_file, plan_file),
        )
        result = {
            key: result[key]
            for key in (
                "plan_sha256",
                "bundle_manifest_sha256",
                "audit_receipt_sha256",
                "holdout_manifest_sha256",
            )
        }
    print(json.dumps({**result, "paid_execution_started": False, "promotable": False}, indent=2))


if __name__ == "__main__":
    main()
