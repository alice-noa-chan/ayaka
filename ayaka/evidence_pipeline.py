"""Opt-in evidence route around the existing Decision API.

The original model is retained. An invalid plan falls back to its exact
probabilities. An auxiliary adapter shared with that model requires an explicit
readout context that disables only that auxiliary adapter, after merging the
original decision adapter. Independent extractors need no such context.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from time import perf_counter

from .evidence import EvidenceError, augmented_state, needs_evidence
from .evidence_ids import id_messages, parse_id_plan, recover_id_evidence, validate_id_plan
from .evidence_policy import EvidencePolicy, fuse_probabilities


class EvidenceDecision:
    def __init__(
        self,
        original,
        extractor,
        policy: EvidencePolicy,
        *,
        readout_context=nullcontext,
        native=None,
    ):
        if policy.readout == "native" and native is None:
            raise ValueError("native policy requires a candidate verifier")
        self.original = original
        self.extractor = extractor
        self.policy = policy
        self.readout_context = readout_context
        self.native = native

    def decide(self, state, questions, device=None):
        with self.readout_context():
            baselines = self.original.decide(state, questions, device=device)
        results = []
        for spec, baseline in zip(questions, baselines, strict=True):
            labels = [str(i) for i in range(len(spec.candidates))]
            criteria = dict(zip(labels, spec.candidates, strict=True))
            if spec.type == "noul":
                criteria = {"false": spec.candidates[0], "true": spec.candidates[1]}
            request = {
                "state": state,
                "question": {
                    "type": spec.type,
                    "instructions": spec.instruction,
                    "criteria": criteria,
                },
                "labels": labels,
            }
            baseline_p = dict(zip(labels, baseline.probs, strict=True))
            extra = {"route": "baseline", "error": None, "extraction_s": 0.0, "readout_s": 0.0}
            auxiliary = baseline_p
            verified = None
            computed = None
            if (
                self.policy.readout != "baseline"
                and max(baseline.probs) <= self.policy.baseline_cutoff
                and needs_evidence(request)
            ):
                tic = perf_counter()
                raw = ""
                try:
                    raw = self.extractor(id_messages(request))
                    verified = validate_id_plan(state, parse_id_plan(raw))
                except (EvidenceError, ValueError) as exc:
                    extra["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
                    if self.policy.recover_ids:
                        try:
                            verified = recover_id_evidence(state, raw)
                            extra["recovered_ids_only"] = True
                        except EvidenceError:
                            pass
                finally:
                    extra["extraction_s"] = perf_counter() - tic
                if verified:
                    if spec.type == "noul":
                        value = verified["calculations"].get("question_holds", {}).get("result")
                        if isinstance(value, bool):
                            computed = labels[1 if value else 0]
                    tic = perf_counter()
                    with self.readout_context():
                        if self.policy.readout == "native":
                            probabilities = self.native(request, spec, verified)
                        else:
                            revised = augmented_state(
                                state, verified, calculations=self.policy.readout == "executed"
                            )
                            probabilities = self.original.decide(revised, [spec], device=device)[
                                0
                            ].probs
                    extra["readout_s"] = perf_counter() - tic
                    auxiliary = dict(zip(labels, probabilities, strict=True))
                    extra["source_ids"] = verified["source_ids"]
                    extra["executed_calculations"] = verified["calculations"]
            probabilities, extra["route"] = fuse_probabilities(
                baseline_p,
                auxiliary,
                self.policy,
                computed_label=computed,
                usable=verified is not None,
            )
            p = [probabilities[lb] for lb in labels]
            extras = dict(baseline.extras, evidence=extra)
            if spec.type == "noul":
                extras["p_true"] = p[1]
            expected = (
                sum(o * pi for o, pi in zip(spec.ordinals, p, strict=True))
                if spec.type == "score"
                else None
            )
            results.append(
                replace(
                    baseline,
                    probs=p,
                    distribution=dict(zip(spec.candidates, p, strict=True)),
                    expected=expected,
                    extras=extras,
                )
            )
        return results
