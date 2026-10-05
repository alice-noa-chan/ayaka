"""Build the predeclared non-public hard calibration/dev splits for Swift prompt gating.

Sources are validation splits only (QuALITY, HotpotQA distractor, HelpSteer2), so they stay
disjoint from the train splits used elsewhere. Rows that share a source document never cross
calibration/dev. Anything sharing a 13-gram with the JevBench public items is dropped. The
output is deterministic for a fixed seed and is hashed in ``manifest.json`` before any read.

    python scripts/swift/build_hard_dev.py --output runs/swift-hard-dev-20261005
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.data import loaders  # noqa: E402
from ayaka.data.decontam import Decontaminator  # noqa: E402

SEED = 20261005
# Per split (calibration and dev each). Tier follows JevBench's families: long documents and
# multi-hop are hard-tier; answer-quality judgement is judge-tier.
TARGETS = {
    ("quality_val", "choice"): 120,
    ("hotpot_val", "choice"): 40,
    ("hotpot_val", "noul"): 120,
    ("helpsteer2_val", "score"): 120,
}
TIERS = {"quality_val": "hard", "hotpot_val": "hard", "helpsteer2_val": "judge"}
LICENSES = {
    "quality_val": "CC BY 4.0 (QuALITY)",
    "hotpot_val": "CC BY-SA 4.0 (HotpotQA)",
    "helpsteer2_val": "CC BY 4.0 (HelpSteer2)",
}
# HelpSteer2 asks five attributes per response; keep the two closest to JevBench judging.
HELPSTEER_INSTRUCTIONS = ("helpful", "correct")


def register_specs() -> None:
    specs = loaders.DATASET_SPECS
    specs["quality_val"] = copy.deepcopy(specs["quality_dev"])
    specs["hotpot_val"] = {
        **copy.deepcopy(specs["hotpot_decisions"]),
        "hf": ("hotpotqa/hotpot_qa", "distractor", "validation"),
        "oversample": 1,
    }
    specs["helpsteer2_val"] = {
        **copy.deepcopy(specs["helpsteer2"]),
        "hf": ("nvidia/HelpSteer2", None, "validation"),
    }


def text_of(state) -> str:
    return (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )


def cluster_of(source: str, sample) -> str:
    """Rows sharing a document (QuALITY article, HelpSteer2 prompt) share a cluster."""
    state = sample.state
    if source == "helpsteer2_val" and isinstance(state, dict):
        key = json.dumps(state.get("prompt", state), ensure_ascii=False, sort_keys=True)
    else:
        key = text_of(state)
    return source + ":" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def role_of(cluster: str) -> str:
    digest = hashlib.sha256(f"{SEED}:{cluster}".encode()).digest()
    return "calibration" if digest[0] % 2 == 0 else "dev"


def keep_question(source: str, question) -> bool:
    if source != "helpsteer2_val":
        return True
    return any(word in question.instruction.lower() for word in HELPSTEER_INSTRUCTIONS)


def build(output: Path, limit: int) -> dict:
    register_specs()
    decontam = Decontaminator.from_jevbench()
    pools: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    dropped = Counter()
    for source in sorted({source for source, _ in TARGETS}):
        samples, manifest = loaders.load_spec_samples(source, limit=limit, seed=SEED)
        for sample in samples:
            if decontam.sample_hit(sample):
                dropped["jevbench_13gram"] += 1
                continue
            record = sample.to_json()
            cluster = cluster_of(source, sample)
            role = role_of(cluster)
            for question in record["questions"]:
                q = next(x for x in sample.questions if x.id == question["id"])
                if not keep_question(source, q):
                    continue
                row = {
                    # A HelpSteer2 prompt has several responses; the source example id keeps
                    # their rows distinct inside one document cluster.
                    "id": "/".join(
                        (
                            source,
                            cluster.split(":")[1],
                            str(record.get("metadata", {}).get("source_example_id", "")),
                            question["id"],
                        )
                    ),
                    "state": record["state"],
                    "split": role,
                    "tier": TIERS[source],
                    "questions": [question],
                    "metadata": {
                        **record.get("metadata", {}),
                        "split": role,
                        "tier": TIERS[source],
                        "source": source,
                        "source_lineage": cluster,
                        "license": LICENSES[source],
                        "modality": "text",
                        "hf_split": manifest.split,
                    },
                }
                pools[(role, source, question["type"])].append(row)
    rng = random.Random(SEED)
    outputs: dict[str, list[dict]] = {"calibration": [], "dev": []}
    shortfall = {}
    for role in outputs:
        for (source, kind), target in sorted(TARGETS.items()):
            pool = sorted(pools[(role, source, kind)], key=lambda r: r["id"])
            rng.shuffle(pool)
            chosen = pool[:target]
            if len(chosen) < target:
                shortfall[f"{role}/{source}/{kind}"] = f"{len(chosen)}/{target}"
            outputs[role].extend(chosen)
    for role, rows in outputs.items():
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{role}: duplicate row ids")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for role, rows in outputs.items():
        rows.sort(key=lambda r: r["id"])
        path = output / f"{role}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
            newline="\n",  # LF on every platform, so the frozen hashes reproduce anywhere
        )
        files[role] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "decisions": len(rows),
            "by_source_type": dict(
                Counter(f"{r['metadata']['source']}/{r['questions'][0]['type']}" for r in rows)
            ),
            "clusters": len({r["metadata"]["source_lineage"] for r in rows}),
        }
    cal_clusters = {r["metadata"]["source_lineage"] for r in outputs["calibration"]}
    dev_clusters = {r["metadata"]["source_lineage"] for r in outputs["dev"]}
    if cal_clusters & dev_clusters:
        raise ValueError("calibration/dev share a source document")
    manifest = {
        "seed": SEED,
        "targets_per_split": {f"{s}/{k}": n for (s, k), n in TARGETS.items()},
        "tiers": TIERS,
        "licenses": LICENSES,
        "files": files,
        "dropped": dict(dropped),
        "shortfall": shortfall,
        "jevbench_public_13gram_decontaminated": True,
        "note": "Non-public hard/judge protocol for Swift prompt gating; never used for training.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4000, help="source rows loaded per source")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, args.limit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
