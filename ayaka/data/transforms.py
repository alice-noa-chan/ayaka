"""Dataset -> canonical schema transforms (docs.md sections 29/42/43).

Each function consumes one raw row (a plain dict, e.g. from a HF
dataset) and emits Sample objects in the canonical schema. Raw dataset
formats never reach the model.

Mapping summary (sec 29):
  NLI family / COPA / HellaSwag / YNAT / Banking77 / MASSIVE /
  CLINC150 / JCommonsenseQA / relation extraction  -> Choice
  BoolQ / JCoLA / WiC / morality / MultiRC-answers / toxicity /
  GoEmotions labels                                  -> Noul
  JSTS / KLUE-STS / Amazon stars                     -> Score
"""

from __future__ import annotations

from .schema import Candidate, Question, Sample, one_hot

# ---------------------------------------------------------------- Choice


def nli_choice(
    row: dict,
    *,
    premise_key: str = "premise",
    hypothesis_key: str = "hypothesis",
    label_key: str = "label",
    labels: tuple[str, ...] = ("entailment", "neutral", "contradiction"),
    metadata: dict | None = None,
) -> list[Sample]:
    """premise/hypothesis -> Choice[entailment, neutral, contradiction].

    Covers SNLI, MultiNLI, ANLI, JNLI, KLUE-NLI, KorNLI. Rows with
    label -1 / unknown are skipped (SNLI no-gold rows).
    """
    label = row[label_key]
    if isinstance(label, str):
        label = labels.index(label) if label in labels else -1
    if label < 0 or label >= len(labels):
        return []
    cands = [Candidate(f"c{i}", labels[i]) for i in range(len(labels))]
    q = Question(
        id="q0",
        type="choice",
        instruction="Does the hypothesis follow from the premise?",
        candidates=cands,
        target_distribution=one_hot(cands, f"c{label}"),
    )
    state = {"premise": row[premise_key], "hypothesis": row[hypothesis_key]}
    return [Sample(state=state, questions=[q], metadata=dict(metadata or {}))]


def chaos_nli(
    row: dict,
    *,
    labels: tuple[str, ...] = ("entailment", "neutral", "contradiction"),
    metadata: dict | None = None,
) -> list[Sample]:
    """ChaosNLI: target = human annotation distribution (sec 42.1).

    Expects row fields: premise, hypothesis, plus either
    ``label_dist`` ([p_e, p_n, p_c]) or ``label_count`` ([n_e, n_n, n_c]).
    """
    if "label_dist" in row:
        dist = [float(x) for x in row["label_dist"]]
    else:
        counts = [float(x) for x in row["label_count"]]
        total = sum(counts)
        dist = [c / total for c in counts]
    cands = [Candidate(f"c{i}", labels[i]) for i in range(len(labels))]
    q = Question(
        id="q0",
        type="choice",
        instruction="Does the hypothesis follow from the premise?",
        candidates=cands,
        target_distribution={f"c{i}": dist[i] for i in range(len(labels))},
    )
    state = {"premise": row["premise"], "hypothesis": row["hypothesis"]}
    md = dict(metadata or {})
    md.setdefault("task_family", "human_soft_label")
    return [Sample(state=state, questions=[q], metadata=md)]


def intent_choice(
    row: dict,
    *,
    text_key: str,
    label_key: str,
    ontology: dict[str, str],
    instruction: str = "Which intent best describes this request?",
    oos_label: str | None = None,
    nota_description: str = "none of the above intents",
    metadata: dict | None = None,
) -> list[Sample]:
    """Intent classification -> Choice over described candidates (sec 43.4).

    ``ontology`` maps label id -> candidate description. ``oos_label``
    (e.g. CLINC150's out-of-scope class) routes mass to an explicit
    none-of-the-above candidate (A5).
    """
    label = str(row[label_key])
    cands = [Candidate(k, desc, is_nota=False) for k, desc in ontology.items()]
    target = one_hot(cands, label) if label in ontology else None
    if label == oos_label or target is None and oos_label is not None:
        nota = Candidate("__nota__", nota_description, is_nota=True)
        cands = cands + [nota]
        target = one_hot(cands, "__nota__")
    if target is None:
        return []
    q = Question(
        id="q0",
        type="choice",
        instruction=instruction,
        candidates=cands,
        target_distribution=target,
    )
    return [Sample(state=row[text_key], questions=[q], metadata=dict(metadata or {}))]


def mc_choice(
    row: dict,
    *,
    context_key: str,
    options_key: str,
    label_key: str,
    instruction: str,
    metadata: dict | None = None,
) -> list[Sample]:
    """Generic multiple choice (COPA, HellaSwag, JCommonsenseQA)."""
    options = list(row[options_key])
    label = int(row[label_key])
    if label < 0 or label >= len(options):
        return []
    cands = [Candidate(f"c{i}", str(opt)) for i, opt in enumerate(options)]
    q = Question(
        id="q0",
        type="choice",
        instruction=instruction,
        candidates=cands,
        target_distribution=one_hot(cands, f"c{label}"),
    )
    return [Sample(state=row[context_key], questions=[q], metadata=dict(metadata or {}))]


def reading_mc(
    row: dict,
    *,
    article_key: str,
    question_key: str,
    options_key: str,
    label_key: str,
    max_chars: int | None = None,
    metadata: dict | None = None,
) -> list[Sample]:
    """Long-document multiple choice (QuALITY): the article is the state,
    the row's own question is the instruction. Articles longer than
    ``max_chars`` are skipped rather than truncated — cutting the middle
    could remove the evidence and silently corrupt the label."""
    article = str(row[article_key])
    if max_chars is not None and len(article) > max_chars:
        return []
    options = [str(o) for o in row[options_key]]
    label = int(row[label_key])
    if not 0 <= label < len(options) or len(set(options)) != len(options):
        return []
    cands = [Candidate(f"c{i}", opt) for i, opt in enumerate(options)]
    q = Question(
        id="q0",
        type="choice",
        instruction=str(row[question_key]),
        candidates=cands,
        target_distribution=one_hot(cands, f"c{label}"),
    )
    md = dict(metadata or {})
    md["task_family"] = "long_context"
    return [Sample(state=article, questions=[q], metadata=md)]


# ------------------------------------------------------------------ Noul


def boolq_noul(
    row: dict,
    *,
    passage_key: str = "passage",
    question_key: str = "question",
    answer_key: str = "answer",
    metadata: dict | None = None,
) -> list[Sample]:
    """Passage evidence -> boolean proposition (BoolQ)."""
    p_true = 1.0 if row[answer_key] else 0.0
    q = Question.noul(
        "q0",
        proposition=str(row[question_key]),
        p_true=p_true,
        false_desc="The answer is no",
        true_desc="The answer is yes",
    )
    return [Sample(state=row[passage_key], questions=[q], metadata=dict(metadata or {}))]


def acceptability_noul(
    row: dict,
    *,
    text_key: str,
    label_key: str,
    instruction: str,
    metadata: dict | None = None,
) -> list[Sample]:
    """Binary judgment -> Noul (JCoLA acceptability, WiC, morality)."""
    q = Question.noul("q0", instruction.format(text=row[text_key]), float(row[label_key]))
    return [Sample(state=row[text_key], questions=[q], metadata=dict(metadata or {}))]


def multirc_nouls(
    row: dict,
    *,
    paragraph_key: str = "paragraph",
    question_key: str = "question",
    answers_key: str = "answers",  # list[(answer_text, is_correct)]
    metadata: dict | None = None,
) -> list[Sample]:
    """MultiRC: one paragraph state + per-answer independent Nouls
    grouped into a single shared-state sample (sec 42.3)."""
    questions = [
        Question.noul(
            f"q{i}",
            proposition=f"{row[question_key]} — is '{ans}' a valid answer?",
            p_true=1.0 if ok else 0.0,
            false_desc="not a valid answer",
            true_desc="a valid answer",
        )
        for i, (ans, ok) in enumerate(row[answers_key])
    ]
    return [Sample(state=row[paragraph_key], questions=questions, metadata=dict(metadata or {}))]


def multilabel_nouls(
    row: dict,
    *,
    text_key: str,
    labels_key: str,
    label_names: dict[str, str],
    instruction_template: str,
    fractional: bool = False,
    metadata: dict | None = None,
) -> list[Sample]:
    """Multi-label -> one Noul per label under a shared state (sec 43.3).

    ``labels_key`` holds either binary/int labels or fractional
    annotator fractions (Jigsaw unintended bias, sec 42.4).
    """
    questions = []
    for i, (lid, desc) in enumerate(label_names.items()):
        val = row[labels_key][lid]
        p_true = float(val) if fractional else float(bool(val))
        questions.append(Question.noul(f"q{i}", instruction_template.format(label=desc), p_true))
    return [Sample(state=row[text_key], questions=questions, metadata=dict(metadata or {}))]


# ----------------------------------------------------------------- Score


def sts_score(
    row: dict,
    *,
    sent1_key: str = "sentence1",
    sent2_key: str = "sentence2",
    score_key: str = "score",
    n_levels: int = 6,
    instruction: str = "How semantically similar are the two sentences?",
    metadata: dict | None = None,
) -> list[Sample]:
    """Continuous STS -> ordinal anchors with adjacent mass (sec 43.2).

    score 3.4 over levels 0..5 -> [0,0,0,0.6,0.4,0].
    """
    score = float(row[score_key])
    lo = int(score)
    hi = min(lo + 1, n_levels - 1)
    frac = score - lo
    dist = [0.0] * n_levels
    dist[lo] += 1.0 - frac
    dist[hi] += frac
    cands = [
        Candidate(f"l{i}", f"similarity level {i} of {n_levels - 1}", ordinal=i)
        for i in range(n_levels)
    ]
    q = Question(
        id="q0",
        type="score",
        instruction=instruction,
        candidates=cands,
        target_distribution={f"l{i}": dist[i] for i in range(n_levels)},
    )
    state = {"sentence1": row[sent1_key], "sentence2": row[sent2_key]}
    return [Sample(state=state, questions=[q], metadata=dict(metadata or {}))]


def ordinal_score(
    row: dict,
    *,
    text_key: str,
    label_key: str,
    n_levels: int,
    instruction: str,
    level_descriptions: list[str] | None = None,
    metadata: dict | None = None,
) -> list[Sample]:
    """Ordinal labels (e.g. 1-5 stars) -> Score with one-hot target."""
    label = int(row[label_key])
    descs = level_descriptions or [f"level {i}" for i in range(n_levels)]
    cands = [Candidate(f"l{i}", descs[i], ordinal=i) for i in range(n_levels)]
    q = Question(
        id="q0",
        type="score",
        instruction=instruction,
        candidates=cands,
        target_distribution=one_hot(cands, f"l{label}"),
    )
    return [Sample(state=row[text_key], questions=[q], metadata=dict(metadata or {}))]


# -------------------------------------------------------------- Jev direct


def jev_direct(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """Already-typed Jev rows -> canonical validation passthrough.

    Expected row shape mirrors the canonical schema: state plus
    questions[] each with type/instruction/candidates[{id,description}]
    and target_distribution.
    """
    questions = []
    for i, q in enumerate(row["questions"]):
        cands = [
            Candidate(
                id=str(c["id"]),
                description=str(c["description"]),
                ordinal=c.get("ordinal"),
                is_nota=bool(c.get("is_nota", False)),
            )
            for c in q["candidates"]
        ]
        questions.append(
            Question(
                id=str(q.get("id", f"q{i}")),
                type=q["type"],
                instruction=q["instruction"],
                candidates=cands,
                target_distribution={str(k): float(v) for k, v in q["target_distribution"].items()},
            )
        )
    md = dict(metadata or {})
    md.setdefault("task_family", "direct_jev")
    return [Sample(state=row["state"], questions=questions, metadata=md)]


def jev_distill(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """SargeDev/jev-distill-corpus-v3 row -> Sample.

    Row: {kind, options[], target[], state, question, domain, family,
    source}. Targets are Jev 1.13 (or 32B-teacher) distributions aligned
    with ``options``; noul options are ["false", "true"].
    """
    kind = row["kind"]
    opts = [str(o) for o in row["options"]]
    tgt = [max(float(x), 0.0) for x in row["target"]]
    z = sum(tgt) or 1.0
    tgt = [x / z for x in tgt]
    if kind == "noul":
        cands = [Candidate("false", "false"), Candidate("true", "true")]
        dist = {"false": tgt[0], "true": tgt[1]}
    elif kind == "score":
        cands = [
            Candidate(f"s{i}", o, ordinal=int(o) if o.lstrip("-").isdigit() else i)
            for i, o in enumerate(opts)
        ]
        if len({c.ordinal for c in cands}) != len(cands):
            cands = [Candidate(f"s{i}", o, ordinal=i) for i, o in enumerate(opts)]
        dist = {c.id: t for c, t in zip(cands, tgt, strict=True)}
    else:
        cands = [Candidate(f"o{i}", o) for i, o in enumerate(opts)]
        dist = {c.id: t for c, t in zip(cands, tgt, strict=True)}
    q = Question(
        id="q0", type=kind, instruction=row["question"], candidates=cands, target_distribution=dist
    )
    md = dict(metadata or {})
    md.update(
        {
            "task_family": "direct_jev",
            "domain": row.get("domain", ""),
            "jev_family": row.get("family", ""),
            "teacher_stream": row.get("source", ""),
            "source_example_id": row.get("id", ""),
        }
    )
    return [Sample(state=row["state"], questions=[q], metadata=md)]


TRANSFORMS = {
    "nli": nli_choice,
    "chaos_nli": chaos_nli,
    "intent": intent_choice,
    "mc": mc_choice,
    "boolq": boolq_noul,
    "acceptability": acceptability_noul,
    "multirc": multirc_nouls,
    "multilabel": multilabel_nouls,
    "sts": sts_score,
    "ordinal": ordinal_score,
    "jev": jev_direct,
    "jev_distill": jev_distill,
    "reading_mc": reading_mc,
}


# ------------------------------------------------- license-clean decision sets
#
# Each source below is openly licensed and labelled by people or by a
# documented procedure (no commercial-LLM outputs as labels). Provenance
# and licenses are recorded in the dataset specs (data/loaders.py).

NOUL_NO = {"no", "false"}


def open_jev_group(group: dict, *, metadata: dict | None = None) -> list[Sample]:
    """ZefanCai/Open-Jev redistributable config -> one multi-question Sample.

    ``group`` = {"rows": [...]} sharing a group_id (one state, several typed
    questions). Rows: kind, question, options[], target[], state_json.
    """
    import contextlib
    import json

    rows = group["rows"]
    state = json.loads(rows[0]["state_json"])
    if isinstance(state, str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            state = json.loads(state)  # some states are double-encoded JSON strings
    questions = []
    for r in rows:
        opts = [str(o) for o in r["options"]]
        tgt = [max(float(x), 0.0) for x in r["target"]]
        z = sum(tgt)
        if len(opts) != len(tgt) or len(opts) < 2 or z <= 0:
            continue
        tgt = [x / z for x in tgt]
        qid = str(r["id"]).rsplit(":", 1)[-1]
        if r["kind"] == "noul":
            p_false = sum(t for o, t in zip(opts, tgt, strict=True) if o.lower() in NOUL_NO)
            questions.append(Question.noul(qid, r["question"], 1.0 - p_false))
        elif r["kind"] == "score":
            cands = [Candidate(f"s{i}", o, ordinal=i) for i, o in enumerate(opts)]
            dist = {c.id: t for c, t in zip(cands, tgt, strict=True)}
            questions.append(Question(qid, "score", r["question"], cands, dist))
        else:
            cands = [Candidate(f"o{i}", o) for i, o in enumerate(opts)]
            if len({c.description for c in cands}) != len(cands):
                continue
            dist = {c.id: t for c, t in zip(cands, tgt, strict=True)}
            questions.append(Question(qid, "choice", r["question"], cands, dist))
    if not questions:
        return []
    md = dict(metadata or {})
    md.update(
        {
            "task_family": "direct_jev",
            "source_example_id": rows[0]["group_id"],
            "open_jev_source": rows[0].get("source", ""),
        }
    )
    return [Sample(state=state, questions=questions, metadata=md)]


VITAMINC = {
    "SUPPORTS": "The evidence supports the claim",
    "REFUTES": "The evidence contradicts the claim",
    "NOT ENOUGH INFO": "The evidence neither supports nor contradicts the claim",
}


def vitaminc_choice(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """tals/vitaminc: contrastive fact verification with explicit NEI."""
    if row.get("label") not in VITAMINC:
        return []
    cands = [Candidate(k.lower().replace(" ", "_"), v) for k, v in VITAMINC.items()]
    gold = row["label"].lower().replace(" ", "_")
    instruction = f"Claim: {row['claim']}\nWhat does the evidence say about this claim?"
    q = Question("q0", "choice", instruction, cands, one_hot(cands, gold))
    return [Sample(state=row["evidence"], questions=[q], metadata=dict(metadata or {}))]


HELPSTEER_LEVELS = {
    "helpfulness": (
        "How helpful is the response to the prompt overall?",
        [
            "not helpful at all",
            "slightly helpful",
            "partially helpful",
            "mostly helpful",
            "fully helpful",
        ],
    ),
    "correctness": (
        "How correct and factual is the response?",
        [
            "mostly wrong or hallucinated",
            "several errors",
            "some errors",
            "minor errors",
            "fully correct",
        ],
    ),
    "coherence": (
        "How clear and coherent is the response?",
        ["incoherent", "hard to follow", "somewhat clear", "mostly clear", "perfectly clear"],
    ),
    "complexity": (
        "How much expertise does writing the response require?",
        [
            "anyone could write it",
            "basic education",
            "some expertise",
            "strong expertise",
            "deep domain expertise",
        ],
    ),
    "verbosity": (
        "How verbose is the response relative to what was asked?",
        ["very terse", "short", "balanced", "long", "very verbose"],
    ),
}


def helpsteer2_scores(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """nvidia/HelpSteer2: one prompt/response, five human 0-4 ratings."""
    questions = []
    for attr, (instruction, levels) in HELPSTEER_LEVELS.items():
        v = row.get(attr)
        if v is None or not 0 <= int(v) <= 4:
            continue
        cands = [Candidate(f"s{i}", f"{i}: {d}", ordinal=i) for i, d in enumerate(levels)]
        questions.append(Question(attr, "score", instruction, cands, one_hot(cands, f"s{int(v)}")))
    if not questions:
        return []
    state = {"prompt": row["prompt"], "response": row["response"]}
    return [Sample(state=state, questions=questions, metadata=dict(metadata or {}))]


def hh_rlhf_choice(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """Anthropic/hh-rlhf: pick the human-preferred final assistant reply."""
    marker = "\n\nAssistant:"
    chosen, rejected = row["chosen"], row["rejected"]
    ci, ri = chosen.rfind(marker), rejected.rfind(marker)
    if ci < 0 or ri < 0 or chosen[:ci] != rejected[:ri]:
        return []
    a, b = chosen[ci + len(marker) :].strip(), rejected[ri + len(marker) :].strip()
    if not a or not b or a == b:
        return []
    cands = [Candidate("chosen", a), Candidate("rejected", b)]
    instruction = "Which assistant reply is more helpful while staying harmless?"
    q = Question("q0", "choice", instruction, cands, one_hot(cands, "chosen"))
    return [Sample(state=chosen[:ci].strip(), questions=[q], metadata=dict(metadata or {}))]


def aqua_rat_choice(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """deepmind/aqua_rat: algebra word problems with five options."""
    import re

    opts, labels = [], []
    for o in row["options"]:
        m = re.match(r"\s*([A-E])\s*\)\s*(.*)", str(o))
        if not m:
            return []
        labels.append(m.group(1))
        opts.append(m.group(2).strip())
    if row.get("correct") not in labels or len(set(opts)) != len(opts):
        return []
    cands = [Candidate(lb, o) for lb, o in zip(labels, opts, strict=True)]
    instruction = "Which option is the correct answer to the problem?"
    q = Question("q0", "choice", instruction, cands, one_hot(cands, row["correct"]))
    return [Sample(state=row["question"], questions=[q], metadata=dict(metadata or {}))]


def aegis_nouls(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """nvidia/Aegis 2.0: human safety labels for prompt and response."""
    if not row.get("prompt") or row["prompt"] == "REDACTED":
        return []
    questions = []
    if row.get("prompt_label") in ("safe", "unsafe") and row.get("prompt_label_source") == "human":
        questions.append(
            Question.noul(
                "prompt_unsafe",
                "Is the user's request unsafe (harmful, abusive or seeking dangerous help)?",
                float(row["prompt_label"] == "unsafe"),
            )
        )
    resp = row.get("response")
    if (
        resp
        and row.get("response_label") in ("safe", "unsafe")
        and row.get("response_label_source") == "human"
    ):
        questions.append(
            Question.noul(
                "response_unsafe",
                "Is the assistant's response unsafe?",
                float(row["response_label"] == "unsafe"),
            )
        )
    if not questions:
        return []
    state = {"user": row["prompt"], "assistant": resp} if resp else {"user": row["prompt"]}
    return [Sample(state=state, questions=questions, metadata=dict(metadata or {}))]


def hotpot_decision(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """hotpotqa (distractor): multi-hop over 10 paragraphs. Yes/no answers
    become a noul; comparison answers that name one of the two compared
    entities become a two-way choice. Span answers are skipped."""
    ctx = row["context"]
    state = "\n\n".join(
        f"{t}: {''.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"], strict=True)
    )
    ans = str(row["answer"]).strip()
    if ans.lower() in ("yes", "no"):
        q = Question.noul("q0", row["question"], float(ans.lower() == "yes"))
    elif row.get("type") == "comparison":
        titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
        if len(titles) != 2 or ans not in titles:
            return []
        cands = [Candidate(f"e{i}", t) for i, t in enumerate(titles)]
        q = Question(
            "q0", "choice", row["question"], cands, one_hot(cands, f"e{titles.index(ans)}")
        )
    else:
        return []
    return [Sample(state=state, questions=[q], metadata=dict(metadata or {}))]


def squad_v2_answerable(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """rajpurkar/squad_v2: can the passage answer the question? (abstention)"""
    q = Question.noul(
        "q0",
        f"Does the passage contain the answer to this question: {row['question']}",
        float(bool(row["answers"]["text"])),
        false_desc="the passage does not answer it",
        true_desc="the passage answers it",
    )
    return [Sample(state=row["context"], questions=[q], metadata=dict(metadata or {}))]


def strategyqa_noul(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """StrategyQA: implicit multi-step yes/no questions with supporting facts."""
    q = Question.noul("q0", row["question"], float(bool(row["answer"])))
    state = row.get("facts") or row.get("description", "")
    return [Sample(state=state, questions=[q], metadata=dict(metadata or {}))]


def labelled_mc(row: dict, *, metadata: dict | None = None) -> list[Sample]:
    """{question, choices: {label[], text[]}, answerKey} (ARC, CommonsenseQA)."""
    labels, texts = row["choices"]["label"], [str(t) for t in row["choices"]["text"]]
    if row.get("answerKey") not in labels or len(set(texts)) != len(texts):
        return []
    cands = [Candidate(str(lb), t) for lb, t in zip(labels, texts, strict=True)]
    instruction = "Which option best answers the question?"
    q = Question("q0", "choice", instruction, cands, one_hot(cands, row["answerKey"]))
    return [Sample(state=row["question"], questions=[q], metadata=dict(metadata or {}))]


TRANSFORMS.update(
    {
        "open_jev": open_jev_group,
        "vitaminc": vitaminc_choice,
        "helpsteer2": helpsteer2_scores,
        "hh_rlhf": hh_rlhf_choice,
        "aqua_rat": aqua_rat_choice,
        "aegis": aegis_nouls,
        "hotpot": hotpot_decision,
        "squad_v2": squad_v2_answerable,
        "strategyqa": strategyqa_noul,
        "labelled_mc": labelled_mc,
        "legalbench": None,  # loaded by data/legalbench.py (per-task label sets)
    }
)
