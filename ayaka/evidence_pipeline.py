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

from .evidence import (
    EvidenceError,
    augmented_state,
    needs_calculation,
    needs_evidence,
    reasoned_state,
    reasoning_messages,
)
from .evidence_ids import id_messages, parse_id_plan, recover_id_evidence, validate_id_plan
from .evidence_policy import EvidencePolicy, fuse_probabilities


def decision_request(state, spec):
    """The extractor's view of one question: positional labels, no gold."""
    labels = [str(i) for i in range(len(spec.candidates))]
    criteria = dict(zip(labels, spec.candidates, strict=True))
    if spec.type == "noul":
        criteria = {"false": spec.candidates[0], "true": spec.candidates[1]}
    return {
        "state": state,
        "question": {"type": spec.type, "instructions": spec.instruction, "criteria": criteria},
        "labels": labels,
    }


def gate_passes(policy: EvidencePolicy, request) -> bool:
    return needs_calculation(request) if policy.gate == "calculation" else needs_evidence(request)


# Selected on the procedural dev set (500) and Open-Jev test (300) before public
# or independent test scoring (runs/evidence-eval-20260928/frozen_policy.json).
FROZEN_REASONING_POLICY = EvidencePolicy(
    readout="reasoned", weight=1.0, baseline_cutoff=0.9, gate="calculation"
)


def reasoning_decision(
    model, tok, max_seq_len=None, policy=FROZEN_REASONING_POLICY, max_new_tokens=384
):
    """Decision wrapped with the gated worked-steps route.

    Needs the decision LoRA unmerged (``load_checkpoint(..., merge=False)``):
    worked steps come from the base model with the adapter disabled, the
    readout from the trained adapter.
    """
    from .evidence_generation import PlanGenerator
    from .primitives import Decision

    if not hasattr(model.backbone, "disable_adapter"):
        raise ValueError("reasoning route needs an unmerged adapter checkpoint (merge=False)")
    reasoner = PlanGenerator(model, tok, max_new_tokens=max_new_tokens, stop_when=None)
    return EvidenceDecision(Decision(model, tok, max_seq_len), reasoner, policy)


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

    # serve.DecisionService reads these for token accounting
    @property
    def tok(self):
        return self.original.tok

    @property
    def max_seq_len(self):
        return self.original.max_seq_len

    def _reasoned(self, state, spec, request, extra, device):
        """Worked steps from the extractor, then the normal readout on them."""
        tic = perf_counter()
        try:
            notes = self.extractor(reasoning_messages(request))
        except (EvidenceError, ValueError) as exc:
            extra["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
            return None
        finally:
            extra["extraction_s"] = perf_counter() - tic
        if not str(notes).strip():
            extra["error"] = "empty worked steps"
            return None
        extra["worked_steps"] = str(notes)[:4000]
        tic = perf_counter()
        with self.readout_context():
            probs = self.original.decide(reasoned_state(state, notes), [spec], device=device)
        extra["readout_s"] = perf_counter() - tic
        return probs[0].probs

    def decide(self, state, questions, device=None):
        with self.readout_context():
            baselines = self.original.decide(state, questions, device=device)
        results = []
        for spec, baseline in zip(questions, baselines, strict=True):
            request = decision_request(state, spec)
            labels = request["labels"]
            baseline_p = dict(zip(labels, baseline.probs, strict=True))
            extra = {"route": "baseline", "error": None, "extraction_s": 0.0, "readout_s": 0.0}
            auxiliary = baseline_p
            verified = None
            computed = None
            if (
                self.policy.readout != "baseline"
                and max(baseline.probs) <= self.policy.baseline_cutoff
                and gate_passes(self.policy, request)
                and self.policy.readout == "reasoned"
            ):
                probabilities = self._reasoned(state, spec, request, extra, device)
                if probabilities is not None:
                    auxiliary = dict(zip(labels, probabilities, strict=True))
            elif (
                self.policy.readout != "baseline"
                and max(baseline.probs) <= self.policy.baseline_cutoff
                and gate_passes(self.policy, request)
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
                usable=verified is not None or auxiliary is not baseline_p,
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
