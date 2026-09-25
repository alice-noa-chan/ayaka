"""Dataset loading: HF datasets -> canonical Samples (sec 29/42/49).

``datasets`` is an optional dependency — it is imported lazily inside
the loaders so tests and CPU-only installs never require it. Specs
pin the exact HF path/config/split plus transform kwargs; a manifest
record is emitted per loaded spec (sec 37).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable

from .dedup import dedup_samples
from .manifest import DatasetManifest
from .schema import Sample
from .transforms import TRANSFORMS

# --------------------------------------------------------- spec registry
#
# Each spec: hf = (path, config, split) | jsonl = path
#            transform -> TRANSFORMS key + kwargs
#            family -> task family quota bucket (sec 41)
#            lang, license, label_schema for the manifest
#
# Feature-resolution kwargs handled by _apply/_rows_to_samples:
#   ontology_from_features: ClassLabel names -> intent ontology
#   ontology_from_column:  (label,label_text) row pairs -> ontology
#   options_from_columns:  choice0..N columns -> options list
#   label_names_from_features: Sequence(ClassLabel) -> multilabel names
#   labels_from_features: ClassLabel names -> nli `labels` tuple
#   label_list_to_dict: list[int] labels -> {id: 0/1} for multilabel
#   score_from_labels: klue sts labels dict -> real-label score
# Specs with parquet=True load script-era datasets through HF's
# auto-converted parquet branch via a data_files glob (datasets 5.x
# removed loading-script support).

JEV_DISTILL_REPO = "SargeDev/jev-distill-corpus-v3"

DATASET_SPECS: dict[str, dict] = {
    # SargeDev/jev-distill-corpus-v3 mixes three label streams:
    #   yuri_v3    Jev 1.13 outputs via OpenRouter. TypeSafe's Master
    #              Customer Agreement s2.3(b) forbids using Output to
    #              distill / train a model imitating the service, so this
    #              stream is not used by default.
    #   yuri_v1    a "32B teacher" of unstated identity/license: not used
    #              by default.
    #   openjev_v2 Open-Jev release-v2-redistributable: synthetic labels
    #              made without any commercial API, CC0-1.0.
    # The jev_open* specs keep only openjev_v2 ("where" filters rows before
    # sampling). The full jev_distill* specs stay available opt-in.
    # hf_jsonl specs download one file and sample rows uniformly (the
    # corpus is ordered by stream, so a head slice would be biased).
    "jev_open": {
        "hf_jsonl": (JEV_DISTILL_REPO, "train.jsonl"),
        "where": {"source": ["openjev_v2"]},
        "transform": "jev_distill",
        "kwargs": {},
        "family": "direct_jev",
        "lang": "en",
        "license": "CC0-1.0 (Open-Jev release-v2-redistributable stream)",
        "label_schema": "Open-Jev synthetic reference distributions (noul/choice)",
    },
    "jev_open_calibration": {
        "hf_jsonl": (JEV_DISTILL_REPO, "calibration.jsonl"),
        "where": {"source": ["openjev_v2"]},
        "transform": "jev_distill",
        "kwargs": {},
        "family": "direct_jev",
        "lang": "en",
        "license": "CC0-1.0 (Open-Jev stream)",
        "label_schema": "held-out split reserved for temperature fitting",
    },
    "jev_open_test": {
        "hf_jsonl": (JEV_DISTILL_REPO, "test_set_30k.jsonl"),
        "where": {"source": ["openjev_v2"]},
        "transform": "jev_distill",
        "kwargs": {},
        "family": "direct_jev",
        "lang": "en",
        "license": "CC0-1.0 (Open-Jev stream)",
        "label_schema": "held-out evaluation split",
    },
    "jev_distill": {
        "hf_jsonl": (JEV_DISTILL_REPO, "train.jsonl"),
        "transform": "jev_distill",
        "kwargs": {},
        "family": "direct_jev",
        "lang": "en",
        "license": "Apache-2.0 as published; yuri_v3 labels are Jev API Output (TypeSafe MCA s2.3(b))",
        "label_schema": "Jev 1.13 / 32B-teacher / Open-Jev distributions",
    },
    "jev_distill_calibration": {
        "hf_jsonl": (JEV_DISTILL_REPO, "calibration.jsonl"),
        "transform": "jev_distill",
        "kwargs": {},
        "family": "direct_jev",
        "lang": "en",
        "license": "Apache-2.0 as published; includes Jev API Output",
        "label_schema": "held-out split reserved for temperature fitting",
    },
    "jev_distill_test30k": {
        "hf_jsonl": (JEV_DISTILL_REPO, "test_set_30k.jsonl"),
        "transform": "jev_distill",
        "kwargs": {},
        "family": "direct_jev",
        "lang": "en",
        "license": "Apache-2.0 as published; includes Jev API Output",
        "label_schema": "held-out Jev-fidelity evaluation split",
    },
    "snli": {
        "hf": ("stanfordnlp/snli", None, "train"),
        "transform": "nli",
        "kwargs": {"labels_from_features": "label"},
        "family": "nli",
        "lang": "en",
        "license": "CC BY-SA 3.0",
        "label_schema": "entailment/neutral/contradiction",
        "source_family": "snli_mnli",
    },
    "multi_nli": {
        "hf": ("nyu-mll/multi_nli", None, "train"),
        "transform": "nli",
        "kwargs": {"labels_from_features": "label"},
        "family": "nli",
        "lang": "en",
        "license": "see dataset card",
        "label_schema": "entailment/neutral/contradiction",
        "source_family": "snli_mnli",
    },
    "anli_r1": {
        "hf": ("facebook/anli", None, "train_r1"),
        "transform": "nli",
        "kwargs": {"labels_from_features": "label"},
        "family": "nli",
        "lang": "en",
        "license": "non-commercial (review before commercial use)",
        "label_schema": "entailment/neutral/contradiction",
        "source_family": "anli",
    },
    "boolq": {
        "hf": ("google/boolq", None, "train"),
        "transform": "boolq",
        "kwargs": {},
        "family": "noul",
        "lang": "en",
        "license": "CC BY-SA 3.0",
        "label_schema": "bool",
    },
    "banking77": {
        "hf": ("mteb/banking77", "default", "train"),
        "parquet": True,
        "transform": "intent",
        "kwargs": {
            "text_key": "text",
            "label_key": "label",
            "ontology_from_column": "label_text",
        },
        "family": "choice",
        "lang": "en",
        "license": "CC BY 4.0",
        "label_schema": "77 intent classes",
    },
    "clinc_oos": {
        "hf": ("clinc/clinc_oos", "plus", "train"),
        "parquet": True,
        "transform": "intent",
        "kwargs": {
            "text_key": "text",
            "label_key": "intent",
            "ontology_from_features": "intent",
            "oos_label": "oos",
        },
        "family": "hard_adversarial",
        "lang": "en",
        "license": "CC BY 3.0",
        "label_schema": "150 intents + oos",
    },
    "super_glue_multirc": {
        "hf": ("aps/super_glue", "multirc", "train"),
        "transform": "multirc_grouped",
        "kwargs": {},
        "family": "noul",
        "lang": "en",
        "license": "see super_glue card",
        "label_schema": "per-answer bool",
    },
    "klue_nli": {
        "hf": ("klue/klue", "nli", "train"),
        "transform": "nli",
        "kwargs": {"labels_from_features": "label"},
        "family": "nli",
        "lang": "ko",
        "license": "CC BY-SA 4.0",
        "label_schema": "entailment/neutral/contradiction",
        "source_family": "klue_nli",
    },
    "klue_sts": {
        "hf": ("klue/klue", "sts", "train"),
        "transform": "sts",
        "kwargs": {
            "sent1_key": "sentence1",
            "sent2_key": "sentence2",
            "score_key": "labels",
            "score_from_labels": "real-score",
            "n_levels": 6,
        },
        "family": "score",
        "lang": "ko",
        "license": "CC BY-SA 4.0",
        "label_schema": "0..5 real",
    },
    "klue_ynat": {
        "hf": ("klue/klue", "ynat", "train"),
        "transform": "intent",
        "kwargs": {
            "text_key": "title",
            "label_key": "label",
            "ontology_from_features": "label",
            "instruction": "Which topic does this news headline belong to?",
        },
        "family": "choice",
        "lang": "ko",
        "license": "CC BY-SA 4.0",
        "label_schema": "7 topics",
    },
    "kor_nli_multi": {
        "hf": ("kakaobrain/kor_nli", "multi_nli", "train"),
        "transform": "nli",
        "kwargs": {"labels_from_features": "label"},
        "family": "nli",
        "lang": "ko",
        "license": "CC BY-SA 4.0",
        "label_schema": "entailment/neutral/contradiction",
        "source_family": "kornli",
    },
    "massive_ja": {
        "hf": ("AmazonScience/massive", "ja-JP", "train"),
        "parquet": True,
        "transform": "intent",
        "kwargs": {
            "text_key": "utt",
            "label_key": "intent",
            "ontology_from_features": "intent",
        },
        "family": "choice",
        "lang": "ja",
        "license": "CC BY 4.0",
        "label_schema": "60 intents",
    },
    "massive_ko": {
        "hf": ("AmazonScience/massive", "ko-KR", "train"),
        "parquet": True,
        "transform": "intent",
        "kwargs": {
            "text_key": "utt",
            "label_key": "intent",
            "ontology_from_features": "intent",
        },
        "family": "choice",
        "lang": "ko",
        "license": "CC BY 4.0",
        "label_schema": "60 intents",
    },
    "jglue_jnli": {
        "hf": ("shunk031/JGLUE", "JNLI", "train"),
        "parquet": True,
        "transform": "nli",
        "kwargs": {
            "premise_key": "sentence1",
            "hypothesis_key": "sentence2",
            "labels_from_features": "label",
        },
        "family": "nli",
        "lang": "ja",
        "license": "CC BY-SA 4.0 (JGLUE)",
        "label_schema": "entailment/neutral/contradiction",
        "source_family": "jnli",
    },
    "jglue_jsts": {
        "hf": ("shunk031/JGLUE", "JSTS", "train"),
        "parquet": True,
        "transform": "sts",
        "kwargs": {
            "sent1_key": "sentence1",
            "sent2_key": "sentence2",
            "score_key": "label",
            "n_levels": 6,
        },
        "family": "score",
        "lang": "ja",
        "license": "CC BY-SA 4.0 (JGLUE)",
        "label_schema": "0..5 real",
    },
    "jglue_commonsense": {
        "hf": ("shunk031/JGLUE", "JCommonsenseQA", "train"),
        "parquet": True,
        "transform": "mc",
        "kwargs": {
            "context_key": "question",
            "options_key": "choices",
            "options_from_columns": ["choice0", "choice1", "choice2", "choice3", "choice4"],
            "label_key": "label",
            "instruction": "Choose the most appropriate answer.",
        },
        "family": "choice",
        "lang": "ja",
        "license": "CC BY-SA 4.0 (JGLUE)",
        "label_schema": "5 options",
    },
    # Long-document reading (docs.md sec 19): JevBench hard states average
    # ~1.2K tokens while jev-distill averages ~190, so long evidence needs
    # its own source. Gemma 4 averages ~4.6 chars/token on these articles,
    # so 17K chars keeps a state under ~3.7K tokens (max_seq_len 4096).
    "quality": {
        "hf": ("emozilla/quality", None, "train"),
        "transform": "reading_mc",
        "kwargs": {
            "article_key": "article",
            "question_key": "question",
            "options_key": "options",
            "label_key": "answer",
            "max_chars": 17_000,
        },
        "family": "long_context",
        "lang": "en",
        "license": "CC BY 4.0 (QuALITY)",
        "label_schema": "4 options, gold index",
    },
    "quality_dev": {
        "hf": ("emozilla/quality", None, "validation"),
        "transform": "reading_mc",
        "kwargs": {
            "article_key": "article",
            "question_key": "question",
            "options_key": "options",
            "label_key": "answer",
            "max_chars": 17_000,
        },
        "family": "long_context",
        "lang": "en",
        "license": "CC BY 4.0 (QuALITY)",
        "label_schema": "4 options, gold index",
    },
    # ---- license-clean decision data (labels by people or documented
    # procedures, never commercial-LLM outputs). Families map onto the
    # mixture quotas in data/mixture.py.
    "open_jev_bde": {
        "hf": ("ZefanCai/Open-Jev", "browser-drone-expansion-v1-redistributable", "train"),
        "group_by": "group_id",
        "transform": "open_jev",
        "kwargs": {},
        "family": "direct_jev",
        "lang": "en",
        "license": "CC0-1.0 (Open-Jev redistributable config)",
        "label_schema": "typed noul/choice/score, several questions per state",
    },
    "vitaminc": {
        "hf": ("tals/vitaminc", None, "train"),
        "transform": "vitaminc",
        "kwargs": {},
        "family": "fact_check",
        "lang": "en",
        "license": "CC BY-SA 3.0",
        "label_schema": "supports / refutes / not enough info (train split only)",
    },
    "helpsteer2": {
        "hf": ("nvidia/HelpSteer2", None, "train"),
        "transform": "helpsteer2",
        "kwargs": {},
        "family": "judge",
        "lang": "en",
        "license": "CC BY 4.0",
        "label_schema": "five human 0-4 ratings per response",
    },
    "hh_rlhf": {
        "hf": ("Anthropic/hh-rlhf", None, "train"),
        "transform": "hh_rlhf",
        "kwargs": {},
        "family": "judge",
        "lang": "en",
        "license": "MIT",
        "label_schema": "human preference between two replies",
    },
    "aegis_safety": {
        "hf": ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, "train"),
        "transform": "aegis",
        "kwargs": {},
        "oversample": 2,
        "family": "judge",
        "lang": "en",
        "license": "CC BY 4.0",
        "label_schema": "human safe/unsafe for prompt and response",
    },
    "aqua_rat": {
        "hf": ("deepmind/aqua_rat", "raw", "train"),
        "transform": "aqua_rat",
        "kwargs": {},
        "family": "reasoning",
        "lang": "en",
        "license": "Apache-2.0",
        "label_schema": "5-option algebra word problems",
    },
    "hotpot_decisions": {
        "hf": ("hotpotqa/hotpot_qa", "distractor", "train"),
        "transform": "hotpot",
        "kwargs": {},
        "oversample": 6,
        "family": "reasoning",
        "lang": "en",
        "license": "CC BY-SA 4.0",
        "label_schema": "multi-hop yes/no and two-entity comparisons over 10 paragraphs",
    },
    "squad_v2_answerable": {
        "hf": ("rajpurkar/squad_v2", None, "train"),
        "transform": "squad_v2",
        "kwargs": {},
        "family": "noul",
        "lang": "en",
        "license": "CC BY-SA 4.0",
        "label_schema": "passage answers the question or not",
    },
    "strategyqa": {
        "hf": ("ChilleD/StrategyQA", None, "train"),
        "transform": "strategyqa",
        "kwargs": {},
        "family": "reasoning",
        "lang": "en",
        "license": "MIT",
        "label_schema": "implicit multi-step yes/no",
    },
    "arc_challenge": {
        "hf": ("allenai/ai2_arc", "ARC-Challenge", "train"),
        "transform": "labelled_mc",
        "kwargs": {},
        "family": "reasoning",
        "lang": "en",
        "license": "CC BY-SA 4.0",
        "label_schema": "4-option science questions",
    },
    "commonsense_qa": {
        "hf": ("tau/commonsense_qa", None, "train"),
        "transform": "labelled_mc",
        "kwargs": {},
        "family": "reasoning",
        "lang": "en",
        "license": "MIT",
        "label_schema": "5-option commonsense questions",
    },
    # generated by this repository with exactly computed labels
    # (data/synthetic.py): temporal and numeric reasoning, long policies
    "synth_temporal": {
        "custom": "synthetic",
        "kind": "temporal",
        "default_n": 20_000,
        "transform": "synthetic",
        "kwargs": {},
        "family": "temporal_numeric",
        "lang": "en",
        "license": "generated by this repository (Apache-2.0)",
        "label_schema": "computed: day counts, windows, weekdays, lateness, event order",
    },
    "synth_numeric": {
        "custom": "synthetic",
        "kind": "numeric",
        "default_n": 20_000,
        "transform": "synthetic",
        "kwargs": {},
        "family": "temporal_numeric",
        "lang": "en",
        "license": "generated by this repository (Apache-2.0)",
        "label_schema": "computed: totals with discount/tax/shipping, budget gaps, maxima",
    },
    "synth_policy": {
        "custom": "synthetic",
        "kind": "policy",
        "default_n": 15_000,
        "transform": "synthetic",
        "kwargs": {},
        "family": "policy",
        "lang": "en",
        "license": "generated by this repository (Apache-2.0)",
        "label_schema": "computed: long reimbursement policy, permitted / first failed clause",
    },
    "legalbench": {
        "custom": "legalbench",
        "transform": "legalbench",
        "kwargs": {},
        "family": "policy",
        "lang": "en",
        "license": "CC BY 4.0 (only tasks whose README states CC BY 4.0)",
        "label_schema": "rule/policy application: Yes/No -> noul, else choice over task labels",
    },
    "go_emotions": {
        "hf": ("google-research-datasets/go_emotions", "simplified", "train"),
        "transform": "multilabel",
        "kwargs": {
            "text_key": "text",
            "labels_key": "labels",
            "label_names_from_features": "labels",
            "label_list_to_dict": True,
            "instruction_template": "Does this text express {label}?",
        },
        "family": "human_soft_label",
        "lang": "en",
        "license": "Apache 2.0",
        "label_schema": "27+1 emotions multi-label",
    },
    "amazon_reviews": {
        "hf": ("mteb/amazon_reviews_multi", "en", "train"),
        "parquet": True,
        "transform": "ordinal",
        "kwargs": {
            "text_key": "text",
            "label_key": "label",
            "n_levels": 5,
            "instruction": "What star rating does this review give?",
        },
        "family": "score",
        "lang": "en",
        "license": "Apache 2.0",
        "label_schema": "0..4 stars",
    },
}


def _humanize(label: str) -> str:
    return label.replace("_", " ")


def _apply(transform: str, kwargs: dict, row: dict, ds, metadata: dict) -> list[Sample]:
    kw = dict(kwargs)
    if "options_from_columns" in kw:
        cols = kw.pop("options_from_columns")
        row = dict(row)
        row[kw["options_key"]] = [row[c] for c in cols]
    if "labels_from_features" in kw:
        kw["labels"] = tuple(ds.features[kw.pop("labels_from_features")].names)
    if "ontology_from_features" in kw:
        feat = kw.pop("ontology_from_features")
        names = ds.features[feat].names
        kw["ontology"] = {str(i): _humanize(n) for i, n in enumerate(names)}
        if kw.get("oos_label") in names:
            kw["oos_label"] = str(names.index(kw["oos_label"]))
    if "label_names_from_features" in kw:
        feat = kw.pop("label_names_from_features")
        names = ds.features[feat].feature.names
        kw["label_names"] = {str(i): _humanize(n) for i, n in enumerate(names)}
        if kw.pop("label_list_to_dict", False):
            row = dict(row)
            active = set(row[kw["labels_key"]])
            row[kw["labels_key"]] = {str(i): int(i in active) for i in range(len(names))}
    else:
        kw.pop("label_list_to_dict", None)
    if kw.pop("score_from_labels", None) == "real-score":
        # klue sts rows carry labels dict {'label': x, 'real-label': y}
        row = dict(row)
        row[kw["score_key"]] = row[kw["score_key"]].get("real-label", 0.0)
    return TRANSFORMS[transform](row, metadata=metadata, **kw)


def _group_multirc(rows: Iterable[dict]) -> Iterable[dict]:
    """Group super_glue multirc rows into per-(paragraph,question)
    rows with answers lists for multirc_nouls."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        key = (r["idx"]["paragraph"], r["idx"]["question"])
        g = groups.setdefault(
            key, {"paragraph": r["paragraph"], "question": r["question"], "answers": []}
        )
        g["answers"].append((r["answer"], bool(r["label"])))
    return groups.values()


def load_spec_samples(
    spec_name: str,
    limit: int | None = None,
    dedup: bool = True,
    seed: int = 0,
) -> tuple[list[Sample], DatasetManifest]:
    """Load one spec into canonical Samples + its manifest record."""
    spec = DATASET_SPECS[spec_name]
    metadata = {
        "language": spec["lang"],
        "source": spec_name,
        "task_family": spec["family"],
        "evidence_state": "intact",
    }
    if spec.get("custom") == "synthetic":
        from .synthetic import generate

        samples = generate(spec["kind"], limit or spec["default_n"], seed, metadata)
        manifest = DatasetManifest(
            dataset_id=spec_name,
            source_url="generated: ayaka/data/synthetic.py",
            revision="v1",
            config=spec["kind"],
            split="generated",
            license=spec["license"],
            language=spec["lang"],
            task_family=spec["family"],
            primitive_mapping=spec["transform"],
            original_label_schema=spec["label_schema"],
            notes=f"{len(samples)} states, seed {seed}",
        )
        return samples, manifest
    if spec.get("custom") == "legalbench":
        from .legalbench import REPO, allowed_tasks, load_legalbench

        samples = load_legalbench(limit, seed, metadata)
        if dedup:
            samples = dedup_samples(samples)
        tasks = allowed_tasks()
        manifest = DatasetManifest(
            dataset_id=spec_name,
            source_url=f"hf://datasets/{REPO}",
            revision="main",
            config=f"{len(tasks)} CC BY 4.0 tasks",
            split="train+test",
            license=spec["license"],
            language=spec["lang"],
            task_family=spec["family"],
            primitive_mapping=spec["transform"],
            original_label_schema=spec["label_schema"],
            notes=f"tasks: {', '.join(sorted(tasks))}",
        )
        return samples, manifest
    if "jsonl" in spec:
        ds = None
        with open(spec["jsonl"], encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        source = spec["jsonl"]
        revision = spec.get("revision", "local")
        config, split = "", ""
    elif "hf_jsonl" in spec:
        from huggingface_hub import hf_hub_download

        repo, filename = spec["hf_jsonl"]
        path = hf_hub_download(repo, filename, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        where = spec.get("where")
        if where:
            lines = [
                ln
                for ln in lines
                if all(json.loads(ln).get(k) in allowed for k, allowed in where.items())
            ]
        if limit is not None and limit < len(lines):
            lines = random.Random(seed).sample(lines, limit)
        rows = [json.loads(line) for line in lines]
        ds = None
        source = f"hf://datasets/{repo}/{filename}"
        revision = "main"
        config, split = "", filename.removesuffix(".jsonl")
    else:
        from datasets import load_dataset  # optional dep (remote image)

        path, config, split = spec["hf"]
        if spec.get("parquet"):
            # datasets 5.x dropped loading scripts; script-era datasets
            # load via the auto-converted parquet branch. The branch has
            # one 'default' BuilderConfig — per-config data is selected
            # through a data_files glob instead.
            cfg_dir = config or "default"
            url = f"hf://datasets/{path}@refs/convert/parquet/{cfg_dir}/{split}/*.parquet"
            ds = load_dataset("parquet", data_files=url, split="train")
        else:
            ds = load_dataset(path, config, split=split)
        if spec.get("group_by"):
            rows = ds  # whole groups only: limit applies to grouped samples below
        else:
            # shuffle before slicing: many sources are ordered (by label, topic,
            # stream), so a head slice would be biased. `oversample` scans
            # more rows for transforms that keep only some of them.
            n = len(ds) if limit is None else min(len(ds), limit * spec.get("oversample", 1))
            rows = ds.shuffle(seed=seed).select(range(n)) if n < len(ds) else ds
        source = f"hf://{path}/{config}/{split}"
        revision = (
            "refs/convert/parquet"
            if spec.get("parquet")
            else str(getattr(getattr(ds, "info", None), "version", "") or "")
        )
    samples = _rows_to_samples(spec, rows, ds, metadata, limit, seed)
    if dedup:
        samples = dedup_samples(samples)
    manifest = DatasetManifest(
        dataset_id=spec_name,
        source_url=source,
        revision=revision,
        config=config or "",
        split=split or "",
        license=spec["license"],
        language=spec["lang"],
        task_family=spec["family"],
        primitive_mapping=spec["transform"],
        original_label_schema=spec["label_schema"],
        notes=spec.get("notes", ""),
    )
    return samples, manifest


def _group_rows(rows, key: str, limit: int | None, seed: int) -> list[dict]:
    """Rows sharing ``key`` -> [{"rows": [...], key: value}], a seeded sample
    of ``limit`` whole groups when limited."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(r[key], []).append(dict(r))
    keys = sorted(groups)
    if limit is not None and limit < len(keys):
        keys = random.Random(seed).sample(keys, limit)
    return [{"rows": groups[k], key: k} for k in keys]


def _rows_to_samples(spec, rows, ds, metadata, limit, seed: int = 0) -> list[Sample]:
    transform = spec["transform"]
    kwargs = dict(spec["kwargs"])
    if transform == "multirc_grouped":
        transform = "multirc"
        rows = _group_multirc(rows)
    if spec.get("group_by"):
        rows = _group_rows(rows, spec["group_by"], limit, seed)
    if "ontology_from_column" in kwargs:
        # build {label_id: humanized description} from row pairs
        col = kwargs.pop("ontology_from_column")
        label_key = kwargs.get("label_key", "label")
        kwargs["ontology"] = {
            str(r[label_key]): _humanize(str(r[col])) for r in rows if r.get(col) is not None
        }
    out: list[Sample] = []
    for i, row in enumerate(rows):
        if limit is not None and len(out) >= limit:
            break
        md = dict(metadata)
        idx = row.get("idx", row.get("id", i))
        md["source_example_id"] = str(idx)
        md["source_family"] = spec.get("source_family", "")
        out.extend(_apply(transform, kwargs, dict(row), ds, md))
    return out


def load_pools(
    spec_names: list[str],
    limit_per_spec: int | None = None,
    dedup: bool = True,
    limits: dict[str, int] | None = None,
    seed: int = 0,
    decontaminator=None,
) -> tuple[dict[tuple[str, str], list[Sample]], list[DatasetManifest]]:
    """Load specs into (family, language) pools for the MixtureSampler.

    ``limits`` overrides ``limit_per_spec`` per spec name; a
    ``decontaminator`` drops samples overlapping evaluation items."""
    pools: dict[tuple[str, str], list[Sample]] = {}
    manifests: list[DatasetManifest] = []
    for name in spec_names:
        limit = (limits or {}).get(name, limit_per_spec)
        try:
            samples, manifest = load_spec_samples(name, limit, dedup, seed)
            if decontaminator is not None:
                samples, dropped = decontaminator.filter(samples)
                manifest.notes = (manifest.notes + f" decontam_dropped={dropped}").strip()
        except Exception as e:  # a broken spec must not kill the run
            print(f"[loaders] spec {name!r} failed, skipping: {e}")
            manifests.append(
                DatasetManifest(
                    dataset_id=name,
                    source_url="",
                    revision="",
                    config="",
                    split="",
                    license=DATASET_SPECS.get(name, {}).get("license", ""),
                    language=DATASET_SPECS.get(name, {}).get("lang", ""),
                    task_family=DATASET_SPECS.get(name, {}).get("family", ""),
                    primitive_mapping=DATASET_SPECS.get(name, {}).get("transform", ""),
                    original_label_schema="",
                    notes=f"LOAD FAILED: {e}",
                )
            )
            continue
        pools.setdefault((manifest.task_family, manifest.language), []).extend(samples)
        manifests.append(manifest)
    return pools, manifests
