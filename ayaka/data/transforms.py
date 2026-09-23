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
}
