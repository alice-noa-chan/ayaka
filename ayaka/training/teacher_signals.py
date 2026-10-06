"""Unweighted diagnostics of already validated prompt-only teacher pairs.

These raw candidate-distribution summaries never select teachers, change loss
weights, fit a serving threshold, or establish a distillation benefit. Source
metadata is for reporting only. Undefined endpoint/tie statistics are explicit.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

VERSION = "ayaka-prompt-teacher-signals-2"


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _robust(values):
    """Unclipped linear quantiles at (n-1)q; finite-only populations are explicit."""
    ordered = sorted(values)

    def quantile(q):
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        left = math.floor(position)
        right = math.ceil(position)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {
        "finite_pairs": len(ordered),
        "min": ordered[0] if ordered else None,
        "p10": quantile(0.1),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "max": ordered[-1] if ordered else None,
    }


def _entropy(probs):
    return -math.fsum(p * math.log(p) for p in probs if p > 0)


def _probs(probs, count):
    if (
        not isinstance(probs, (list, tuple))
        or len(probs) != count
        or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs
        )
        or abs(math.fsum(probs) - 1) > 1e-6
    ):
        raise ValueError("teacher signal requires valid aligned probability distributions")
    return tuple(probs)


def _logit(probs, labels):
    if any(p == 0 for p in probs):
        return None
    return math.log(probs[labels.index("true")]) - math.log(probs[labels.index("false")])


def _slice(rows):
    entropy_changes, teacher_entropies, student_entropies, finite_kl = [], [], [], []
    student_logits, teacher_logits = [], []
    kl_nonfinite, logit_endpoints, ties, flips, untied = 0, 0, 0, 0, 0
    for row in rows:
        labels, student, teacher = row["candidate_ids"], row["direct_probs"], row["teacher_probs"]
        a, b = _entropy(student), _entropy(teacher)
        student_entropies.append(a)
        teacher_entropies.append(b)
        entropy_changes.append(b - a)
        if any(t > 0 and s == 0 for t, s in zip(teacher, student, strict=True)):
            kl_nonfinite += 1
        else:
            finite_kl.append(
                math.fsum(
                    t * (math.log(t) - math.log(s))
                    for t, s in zip(teacher, student, strict=True)
                    if t > 0
                )
            )
        winners = [
            {i for i, p in enumerate(probs) if p == max(probs)} for probs in (student, teacher)
        ]
        if any(len(winner) != 1 for winner in winners):
            ties += 1
        else:
            untied += 1
            flips += winners[0] != winners[1]
        if row["type"] == "noul":
            x, y = _logit(student, labels), _logit(teacher, labels)
            if x is None or y is None:
                logit_endpoints += 1
            else:
                student_logits.append(x)
                teacher_logits.append(y)
    accepted = sum(row["teacher_accepted"] for row in rows)
    result = {
        "observed_pairs": len(rows),
        "accepted_teacher_questions": accepted,
        "rejected_teacher_questions": len(rows) - accepted,
        "acceptance_fraction": accepted / len(rows),
        "mean_student_entropy_nats": _mean(student_entropies),
        "mean_teacher_entropy_nats": _mean(teacher_entropies),
        "mean_entropy_change_nats": _mean(entropy_changes),
        "mean_forward_kl_finite_nats": _mean(finite_kl),
        "forward_kl_finite_nats": _robust(finite_kl),
        "entropy_change_nats": _robust(entropy_changes),
        "forward_kl_finite_pairs": len(finite_kl),
        "forward_kl_nonfinite_pairs": kl_nonfinite,
        "argmax_tied_pairs": ties,
        "argmax_untied_pairs": untied,
        "argmax_flips": flips,
        "argmax_flip_fraction_of_untied": flips / untied if untied else None,
    }
    if rows[0]["type"] == "noul":
        delta = [y - x for x, y in zip(student_logits, teacher_logits, strict=True)]
        center = _mean(delta)
        residual_rmse = math.sqrt(_mean([(d - center) ** 2 for d in delta])) if delta else None
        xmean, ymean = _mean(student_logits), _mean(teacher_logits)
        if student_logits:
            xdev = [x - xmean for x in student_logits]
            ydev = [y - ymean for y in teacher_logits]
            xx, yy = math.fsum(x * x for x in xdev), math.fsum(y * y for y in ydev)
            xy = math.fsum(x * y for x, y in zip(xdev, ydev, strict=True))
        else:
            xx = yy = xy = 0
        result["noul_true_logit"] = {
            "finite_pairs": len(delta),
            "endpoint_pairs": logit_endpoints,
            "mean_teacher_minus_student": center,
            "student": _robust(student_logits),
            "teacher": _robust(teacher_logits),
            "teacher_minus_student": _robust(delta),
            "residual_rmse_after_slice_constant": residual_rmse,
            "teacher_on_student_ols_slope": xy / xx if xx > 0 else None,
            "pearson_correlation": xy / math.sqrt(xx * yy) if xx > 0 and yy > 0 else None,
        }
    return result


def prompt_teacher_signal_report(records):
    """Report source×type pairs after exact binding and existing gold filtering.

    Missing-teacher rows may be included for replay coverage. Their candidate
    distributions are absent, so they do not enter the pair statistics.
    """
    pairs, missing, accepted_sources = defaultdict(list), Counter(), Counter()
    for original in records:
        if not isinstance(original, dict):
            raise ValueError("teacher signal records must be objects")
        row = dict(original)
        source, kind = row.get("source"), row.get("type")
        if (
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(kind, str)
            or kind not in {"choice", "noul", "score"}
        ):
            raise ValueError("teacher signal requires an explicit source/type reporting cell")
        if (
            type(row.get("teacher_present")) is not bool
            or type(row.get("teacher_accepted")) is not bool
        ):
            raise ValueError("teacher signal coverage/acceptance flags must be bools")
        if not row["teacher_present"]:
            if row["teacher_accepted"]:
                raise ValueError("a missing teacher cannot be accepted")
            missing[source, kind] += 1
            continue
        labels = row.get("candidate_ids")
        if (
            not isinstance(labels, (list, tuple))
            or len(labels) < 2
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)
            or (kind == "noul" and (len(labels) != 2 or set(labels) != {"false", "true"}))
        ):
            raise ValueError("teacher signal candidate IDs must retain canonical semantics")
        row["candidate_ids"] = tuple(labels)
        row["direct_probs"] = _probs(row.get("direct_probs"), len(labels))
        row["teacher_probs"] = _probs(row.get("teacher_probs"), len(labels))
        pairs[source, kind].append(row)
        if row["teacher_accepted"]:
            accepted_sources[source] += 1
    slices = []
    for source, kind in sorted(pairs.keys() | missing.keys()):
        rows = pairs[source, kind]
        accepted = [r for r in rows if r["teacher_accepted"]]
        rejected = [r for r in rows if not r["teacher_accepted"]]
        summary = (
            _slice(rows)
            if rows
            else {
                "observed_pairs": 0,
                "accepted_teacher_questions": 0,
                "rejected_teacher_questions": 0,
                "acceptance_fraction": None,
            }
        )
        slices.append(
            {
                "source": source,
                "type": kind,
                **summary,
                "no_saved_teacher_gold_replay_questions": missing[source, kind],
                "accepted_pair_statistics": _slice(accepted) if accepted else None,
                "rejected_pair_statistics": _slice(rejected) if rejected else None,
            }
        )
    accepted_total = sum(accepted_sources.values())
    return {
        "version": VERSION,
        "by_source_type": slices,
        "accepted_source_mix": [
            {"source": source, "questions": n, "fraction": n / accepted_total}
            for source, n in sorted(accepted_sources.items())
        ],
        "weighting": "unweighted questions; not component weights or training exposure",
        "probability_space": "raw candidate probabilities before fitted serving policy",
        "probability_clipping_applied": False,
        "robust_summary_method": "unclipped linear quantiles at (n-1)q; means and OLS retained",
        "scope": "diagnostic only; centered residuals do not establish gold improvement",
        "policy_refit_required_before_trained_comparison": True,
        "eligibility_or_authored_cap_applied": False,
        "execution_attested": False,
        "promotable": False,
    }
