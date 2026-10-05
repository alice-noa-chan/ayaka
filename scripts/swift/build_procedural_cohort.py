"""Build the fresh procedural cohort for the v1-on vs v2-off comparison.

Six repository generators (temporal, numeric, policy, long_rules, calendar, probability)
with seed 20261005. v1 large trained on the same generators with seed 0, so these items are
in-distribution for v1: a conservative cohort for a "v2 off beats v1 on" claim. Any state
that also appears in the seed-0 training draw (each generator's full default size) is
dropped. So is anything sharing a 13-gram with the JevBench public items. All questions of
one sample share a cluster. Output is LF-terminated and deterministic.

    python scripts/swift/build_procedural_cohort.py --output runs/v1v2-cohort-20261005
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ayaka.data import loaders  # noqa: E402
from ayaka.data.decontam import Decontaminator  # noqa: E402

SEED = 20261005
TRAIN_SEED = 0
SAMPLES = {
    "synth_temporal": 20,
    "synth_numeric": 20,
    "synth_policy": 20,
    "synth_long_rules": 20,
    "synth_calendar": 20,
    "synth_probability": 60,
}
# Per-type floors over the whole cohort; the build refuses to emit less.
MINIMUM = {"choice": 120, "noul": 120, "score": 40}


def state_key(state) -> str:
    text = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(output: Path) -> dict:
    decontam = Decontaminator.from_jevbench()
    rows, dropped = [], Counter()
    for spec, count in SAMPLES.items():
        trained = {
            state_key(s.state)
            for s in loaders.load_spec_samples(
                spec, limit=loaders.DATASET_SPECS[spec]["default_n"], seed=TRAIN_SEED
            )[0]
        }
        # Draw extra so drops do not shrink the cohort; keep the first `count` survivors.
        fresh, _ = loaders.load_spec_samples(spec, limit=count * 3, seed=SEED)
        kept = 0
        for index, sample in enumerate(fresh):
            if kept == count:
                break
            if state_key(sample.state) in trained:
                dropped["seed0_training_state"] += 1
                continue
            if decontam.sample_hit(sample):
                dropped["jevbench_13gram"] += 1
                continue
            kept += 1
            record = sample.to_json()
            cluster = f"{spec}:{SEED}:{index}"
            for question in record["questions"]:
                rows.append(
                    {
                        "id": f"{spec}/{SEED}/{index}/{question['id']}",
                        "state": record["state"],
                        "split": "test",
                        "tier": "standard",
                        "questions": [question],
                        "metadata": {
                            **record.get("metadata", {}),
                            "split": "test",
                            "tier": "standard",
                            "source": spec,
                            "source_lineage": cluster,
                            "modality": "text",
                            "generator_seed": SEED,
                        },
                    }
                )
        if kept < count:
            raise ValueError(f"{spec}: only {kept} of {count} samples survived")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row ids")
    types = Counter(row["questions"][0]["type"] for row in rows)
    short = {k: f"{types[k]}/{v}" for k, v in MINIMUM.items() if types[k] < v}
    if short:
        raise ValueError(f"per-type minimum not met: {short}")
    rows.sort(key=lambda row: row["id"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "procedural.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "seed": SEED,
        "train_seed_checked": TRAIN_SEED,
        "samples_per_generator": SAMPLES,
        "minimum_per_type": MINIMUM,
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decisions": len(rows),
        "clusters": len({r["metadata"]["source_lineage"] for r in rows}),
        "by_source_type": dict(
            Counter(f"{r['metadata']['source']}/{r['questions'][0]['type']}" for r in rows)
        ),
        "dropped": dict(dropped),
    }
    (output / "procedural_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
