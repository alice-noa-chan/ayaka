"""Hugging Face model card for an Electra export folder.

All metrics in the card are read from run artifacts: the checkpoint's
meta.json (steps, run config, held-out metrics,
reference held-out set, temperatures), jevbench_report.json (public tiers with
item-for-item reference comparisons), dataset_manifest.jsonl (sources,
licenses, decontamination counts) and export_meta.json (int8 parity). The Speed section states the
architecture's measured speed-ups, which do not depend on training.

    python -m ayaka.modelcard --export runs/exports/electra-small --code-url https://github.com/<you>/ayaka
"""

from __future__ import annotations

import json
import os

JEV_KEY = "jev-1.13.0"
TIER_NAMES = {"easy": "easy", "original": "standard (public)", "hard": "hard"}


def _load(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _num(x, fmt="{:.3f}") -> str:
    return "—" if x is None else fmt.format(x)


def gather(export_dir: str) -> dict:
    meta = _load(os.path.join(export_dir, "export_meta.json"), {})
    ecfg = _load(os.path.join(export_dir, "electra_config.json"), {})
    src = meta.get("source", "")  # training checkpoint dir
    run_dir = os.path.dirname(os.path.normpath(src)) if src else ""
    return {
        "name": os.path.basename(os.path.normpath(export_dir)),
        "export_meta": meta,
        "electra_config": ecfg,
        "ckpt_meta": _load(os.path.join(src, "meta.json"), {}) if src else {},
        "jevbench": _load(os.path.join(run_dir, "jevbench_report.json"), {}) if run_dir else {},
        "manifest": _load(os.path.join(run_dir, "dataset_manifest.jsonl"), []) if run_dir else [],
        "run_config": _load(os.path.join(run_dir, "run_config.json"), {}) if run_dir else {},
    }


def _base_model(info: dict) -> str:
    run = info["run_config"] or info["ckpt_meta"].get("run", {})
    from .config import model_config

    try:
        return model_config(run.get("model_size", "")).backbone
    except KeyError:
        return info["electra_config"].get("backbone", "")


def _jevbench_section(jb: dict) -> list[str]:
    tiers = jb.get("tiers", {})
    summary = jb.get("summary", {})
    if not tiers:
        return ["_No JevBench report found for this run._", ""]
    refs = summary.get("references", {})
    names = {k: v.get("display", k) for k, v in _load_refs().items()}
    lines = [
        "Public JevBench items, compared item-for-item with the published per-task outcomes "
        "of reference systems on the same items (JevBench v1.2 per-task file). The Intelligence "
        "proxy is JevBench's chance-corrected formula with the official tier weights renormalized "
        "over the public tiers; the official score also uses a sealed held-out set this card cannot "
        "include.",
        "",
        "| System | " + " | ".join(TIER_NAMES.get(t, t) for t in tiers) + " | Intelligence proxy |",
        "|---|" + "---|" * (len(tiers) + 1),
        "| **this model** | "
        + " | ".join(_pct(tiers[t]["accuracy"]) for t in tiers)
        + f" | **{_num(summary.get('intelligence_proxy'), '{:.1f}')}** |",
    ]
    for key in [JEV_KEY] + [k for k in refs if k != JEV_KEY]:
        if key not in refs:
            continue
        acc = refs[key]["accuracy"]
        lines.append(
            f"| {names.get(key, key)} | "
            + " | ".join(_pct(acc.get(t)) for t in tiers)
            + f" | {_num(refs[key].get('intelligence_proxy'), '{:.1f}')} |"
        )
    lines += ["", "| Tier | n | Brier | ECE | p50 latency |", "|---|---|---|---|---|"]
    for t, m in tiers.items():
        lines.append(
            f"| {TIER_NAMES.get(t, t)} | {m['n']} | {_num(m.get('brier'))} | {_num(m.get('ece'))} | "
            f"{_num(m.get('latency_p50_s'), '{:.2f} s')} |"
        )
    return lines + [""]


def _load_refs() -> dict:
    from .eval.jevbench import reference_outcomes

    try:
        return reference_outcomes()
    except FileNotFoundError:
        return {}


def _metrics_row(m: dict) -> str:
    return (
        f"| {m.get('n', '—')} | {_pct(m.get('accuracy'))} | {_num(m.get('kl'))} | "
        f"{_num(m.get('brier'))} | {_num(m.get('ece'))} |"
    )


def _data_section(manifest: list[dict]) -> list[str]:
    if not manifest:
        return ["_No dataset manifest found for this run._", ""]
    lines = ["| Source | Split | Language | License | Notes |", "|---|---|---|---|---|"]
    for r in manifest:
        lines.append(
            f"| {r.get('source_url') or r.get('dataset_id')} | {r.get('split', '')} | {r.get('language', '')} | "
            f"{r.get('license', '')} | {r.get('notes', '')} |"
        )
    return lines + [""]


def _hf_dataset_id(url: str) -> str | None:
    """hf://datasets/<org>/<name>/... or hf://<org>/<name>/... -> <org>/<name>."""
    if not url.startswith("hf://"):
        return None
    parts = url.removeprefix("hf://").removeprefix("datasets/").split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 and parts[0] and parts[1] else None


def build_card(export_dir: str, code_url: str = "<code-url>") -> str:
    info = gather(export_dir)
    meta, ck = info["export_meta"], info["ckpt_meta"]
    run = info["run_config"] or ck.get("run", {})
    base = _base_model(info)
    quantized = bool(meta.get("quantized"))
    datasets = sorted(
        {d for r in info["manifest"] if (d := _hf_dataset_id(r.get("source_url", "")))}
    )
    front = [
        "---",
        "license: apache-2.0",
        f"base_model: {base}",
        "base_model_relation: finetune" if not quantized else "base_model_relation: quantized",
        "library_name: ayaka",
        "pipeline_tag: text-classification",
        "language: [en, ko, ja]",
        "tags: [decision-model, jev, system-one, calibration, gemma4]",
    ]
    if datasets:
        front.append("datasets:")
        front += [f"  - {d}" for d in datasets]
    front.append("---")

    fid = ck.get("reference_eval") or ck.get("jev_fidelity") or {}
    fid_spec = fid.get("spec", "jev_distill_test30k" if "jev_fidelity" in ck else "—")
    held = ck.get("heldout", {})
    uses_jev_output = any(r.get("dataset_id") == "jev_distill" for r in info["manifest"])
    temps = ck.get("temperatures", {})
    parity = meta.get("parity_vs_bf16")
    body = [
        f"# {info['name']}",
        "",
        f"An **Electra** decision model: give it a state (text or JSON) and typed questions — "
        f"`noul` (yes/no), `choice` (runtime options) or `score` (ordered levels) — and it returns a "
        f"calibrated probability distribution per question in one forward pass, with no text "
        f"generation. Backbone: [`{base}`](https://huggingface.co/{base}) with a LoRA adapter "
        f"(merged) and a permutation-equivariant pointer head.",
        "",
        "Weights: "
        + (
            "int8 (per-row symmetric), half the size of the bf16 release; embeddings stay int8 in "
            "memory and Linear layers are dequantized to bf16 at load."
            if quantized
            else "bf16, LoRA merged."
        ),
        "",
        "## Use",
        "",
        "```bash",
        f'pip install "ayaka @ git+{code_url}"',
        f"hf download <this-repo> --local-dir {info['name']}",
        f"python -m ayaka.serve --model {info['name']} --device cuda   # TypeSafe-compatible POST /v1/systemone",
        "```",
        "",
        "```python",
        "from ayaka.export import load_exported",
        "from ayaka.primitives import Decision, QuestionSpec",
        f'model, tok = load_exported("{info["name"]}", device="cuda")',
        "d = Decision(model, tok)",
        'd.choice({"message": "where is my parcel?"}, "What does the user want?", ["track order", "refund"])',
        "```",
        "",
        "Several questions about one state share a single encoding of that state and never see "
        "each other. Choice options are rendered in a content-derived canonical order, so "
        "reordering the options permutes the output exactly.",
        "",
        "## Evaluation",
        "",
        "### JevBench (public tiers)",
        "",
        *_jevbench_section(info["jevbench"]),
        "### Held-out reference set and held-out training mixture",
        "",
        "| Set | n | Accuracy vs reference argmax | KL(ref ‖ model) | Brier | ECE |",
        "|---|---|---|---|---|---|",
        f"| `{fid_spec}` " + _metrics_row(fid) if fid else f"| `{fid_spec}` | — | — | — | — | — |",
        "| Held-out mix " + _metrics_row(held) if held else "| Held-out mix | — | — | — | — | — |",
        "",
        f"Temperatures per primitive and prompt-length bucket (`prim@long` = at least "
        f"{run.get('long_prompt_tokens', 1024)} prompt tokens; fitted on "
        f"`{run.get('calibration_spec', 'jev_open_calibration')}`, topped up from a reserved "
        f"training slice for buckets it lacks): `{json.dumps(temps)}`",
        "",
    ]
    if quantized:
        body += [
            "### int8 vs bf16 parity",
            "",
            (
                f"On {parity['n']} held-out Jev items: argmax agreement "
                f"**{_pct(parity['argmax_agreement'])}**, mean KL {_num(parity['mean_kl'], '{:.2e}')}, "
                f"max |Δp| {_num(parity['max_abs_prob_diff'])}."
                if parity
                else "_Parity not measured for this export._"
            ),
            "",
        ]
    body += [
        "## Speed",
        "",
        "Gemma 4 E2B/E4B end in KV-shared layers; Electra runs only the answer position through "
        "them, encodes the constant prompt head once and computes sliding-window attention "
        "block-wise. All three are exact (tests match the stock forward). Measured on zero-shot "
        "E2B, 8-core x86 CPU, bf16: JevBench easy p50 2.44 s → 0.69 s, hard 42.3 s → 11.8 s. "
        "Serve on a GPU for production latency; `--dtype float32` tracks reference probabilities "
        "more closely on CPU at ~1.3–1.5× latency.",
        "",
        "## Training",
        "",
        f"- steps: {ck.get('steps', '—')}, questions/step: {run.get('questions_per_step', '—')}, "
        f"LoRA lr: {run.get('lr', '—')}, restricted data included: {run.get('include_restricted', False)}",
        f"- distilled from a larger Electra teacher: {bool(run.get('teacher_labels'))}",
        "- losses: soft-target NLL (KL to teacher when distilling), Brier, RPS (score), "
        "missing-evidence overconfidence penalty, auxiliary pointer NLL",
        "- every sample sharing a 13-gram with a JevBench public item was removed before training "
        "(counts in the table below); JevBench items were used for evaluation only",
        "",
        *_data_section(info["manifest"]),
        "## Limitations",
        "",
        "- JevBench numbers above cover the public tiers only; the official leaderboard adds sealed "
        "items that were not available here.",
        "- Long documents: training states average ~190 tokens (plus QuALITY articles up to ~3.7K); "
        "inputs are capped at 4,096 tokens (the middle of longer states is elided).",
        "- English-centric supervision; Korean/Japanese come from smaller NLU sets.",
        "- Probabilities are calibrated on Jev-distilled labels, so they inherit that teacher's "
        "judgments and biases.",
        "",
        "## License and attribution",
        "",
        f"- Model weights: Apache-2.0, derived from [`{base}`](https://huggingface.co/{base}) (Apache-2.0).",
        "- JevBench public items and per-task reference outcomes (bundled with the code for "
        "evaluation and decontamination): MIT, © Florian Standhartinger and contributors.",
        "- Training data licenses are listed per source above. "
        + (
            "This run includes the full `SargeDev/jev-distill-corpus-v3`, whose `yuri_v3` stream "
            "holds outputs of TypeSafe's Jev API."
            if uses_jev_output
            else "From `SargeDev/jev-distill-corpus-v3` only the `openjev_v2` stream is used "
            "(Open-Jev, CC0-1.0, labels made without any commercial API); no commercial-API "
            "outputs were used as training labels."
        ),
        "- Not affiliated with or endorsed by TypeSafe AI or Google.",
        "",
    ]
    return "\n".join(front + [""] + body)


def write_card(export_dir: str, code_url: str = "<code-url>") -> str:
    path = os.path.join(export_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_card(export_dir, code_url))
    return path


def main(argv: list[str] | None = None) -> str:
    import argparse

    ap = argparse.ArgumentParser(description="Write README.md model card for an export folder")
    ap.add_argument("--export", required=True)
    ap.add_argument("--code-url", default="<code-url>", help="git URL of this code (pip install)")
    args = ap.parse_args(argv)
    path = write_card(args.export, args.code_url)
    print(f"[modelcard] {path}")
    return path


if __name__ == "__main__":
    main()
