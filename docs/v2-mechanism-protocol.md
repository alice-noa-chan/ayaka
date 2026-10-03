# Frozen Ayaka / Jeeves mechanism diagnosis

The completed v1-only [results and execution ledger](v2-mechanism-results-20261004.md)
are recorded separately, including incomplete attempts and control limitations.

**Execution scope amendment:** the paired checkpoint forecast did not fit its
full-cap time allowance, even after correcting fixed overhead. Neither attempt
completed the cohort. The final admitted scope is **published v1 only**, with
all 12 preselected cases, 36 questions, five contexts and three readouts
unchanged. No pilot comparison is claimed. The failed receipts remain separate.
The v1-only task has a 3,400-second backend limit: $2.35 maximum conservative
compute plus $0.10 margin, admitted after both earlier actual charges. This
reduces checkpoint scope, not the reasoning budget or completed-case requirement.

This is a diagnostic on English **dev**, not a new training run, release gate,
JevBench score, or reproduction of Jeeves. The previous clean pilot's main
comparison used reasoning **off**; three on/high probes cannot establish the
cause of that pilot's regression.

## Source audit

Official Jeeves source was inspected at
[`3f948dec68187ed3ced9152ed3d84b73e498665c`](https://github.com/PostHog/jeeves/tree/3f948dec68187ed3ced9152ed3d84b73e498665c).
Its [head](https://github.com/PostHog/jeeves/blob/3f948dec68187ed3ced9152ed3d84b73e498665c/model/head.py)
is a learned 256-dimensional query/key pointer, without Ayaka's Set Mixer or LM
hybrid. Its [trainer](https://github.com/PostHog/jeeves/blob/3f948dec68187ed3ced9152ed3d84b73e498665c/trainer.py)
uses Qwen3.5-9B, half-trace SFT, then CISPO reinforcement learning. Questions and
options are repeated after a closed thinking region. The diffusion drafter is a
separate latency optimization. Ayaka's failed continuation used Gemma 4 E4B,
the inherited hybrid head, and 200 joint-SFT steps, without CISPO.
Ayaka's `chat_ids` explicitly sets `enable_thinking=False` and requests ordinary
worked steps. Gemma's pinned native template instead injects `<|think|>` when
its thinking mode is enabled. Native thinking is not evaluated by this protocol.
The SFT builder appends tokenizer EOS (1), while the first observed v1 trace
ended with native turn EOS (106). Controls match the actual generated closer;
the causal effect of the SFT/inference closer difference is not isolated here.

The authors report no-thinking 0.804 and thinking 0.840 on the same 2,962-question
test in their [README](https://github.com/PostHog/jeeves/blob/3f948dec68187ed3ced9152ed3d84b73e498665c/README.md).
These are their measurements, not an Ayaka replication or an official composite.

## Predeclared comparison

Freeze published v1 and the delivered clean 200-step continuation. Use seed
20261003 and the lowest SHA256(seed/lineage) **two independent English dev
cases per each of six rules**: 12 cases, 36 typed questions per checkpoint.
This reduction from the initial three-case design is made before any new model
output, to leave time for complete evaluation and delivery within existing credit.
Translations do not inflate the independent sample count. Calibration and test
are never opened. Choose cases before looking at any model outputs.

Each question has five contexts: production direct; an empty, matched reasoning
context; a greedy model-generated trace with 512 maximum tokens; a complete
reference derivation; and a cyclic derivation from another selected case of the
same rule. Reference notes are independently recomputed from the visible rule
and facts and checked against every prepared target. They are **privileged
interventions**, not inference capabilities. Identical oracle/distractor notes
are flagged rather than replaced after evaluation.
Generate first, then use its exact EOS token to close all three teacher-forced
controls. When generation reaches the cap, leave every control unclosed as well.
This avoids confusing different native EOS/turn markers with trace content.

Read the same hidden states through native LM labels, the **existing trained
Ayaka Set Mixer pointer**, and their inherited hybrid gate. This pointer is not
Jeeves's plain pointer and has not been separately optimized as a standalone
model. Weight files stay fixed, dropout is disabled, no temperatures are fit,
and no optimizer exists. The diagnostic's direct hybrid must reproduce the
production cached readout within 1e-6. Context truncation, empty generation and
generation errors fail explicitly instead of silently substituting direct
results. Each question has its own trace cache. Save raw distributions, trace
text/hash, EOS/cap termination, token counts, and checkpoint/cohort identities.

Report typed chance-corrected competence, NLL and Score RPS. Compare reasoning
against **both direct and empty**: the latter controls for prompt/suffix changes
and termination token. Content lengths still differ and remain a limitation.
Compute paired 2,000-replicate underlying-case bootstrap intervals and the
difference between LM and hybrid gains, relative to empty, on identical cases.
The small, previously used dev domain makes these diagnostic intervals
exploratory; multiple head/context comparisons do not establish a release win.

## What can be concluded

- Oracle gains but weak generated gains suggest a trace-generation/integration
  bottleneck; they do not demonstrate a practical accuracy improvement.
- A positive LM-versus-hybrid interaction suggests a readout interaction on
  these checkpoints. It does not show that a retrained plain pointer would win.
- Empty-context changes expose prompt/closure effects. Distractor changes test
  reliance on supplied notes; identical distractors are uninformative.
- No gain even with oracle notes means this intervention was ineffective for
  this checkpoint/format. It does not prove that reasoning, Jeeves's recipe, or
  all Gemma models are intrinsically unsuitable.

Distinguishing the full Jeeves recipe from backbone, training and head choices
would require matched training ablations. This low-cost inference experiment
cannot isolate those training causes. Do not promote or retrain based on a
favorable small-cohort result.

## Resource boundary

Reuse the immutable offline kit plus a checksummed overlay containing only this
diagnostic, its frozen plan and the delivered pilot adapter/head. Prepare and
upload before GPU allocation. Admit **one** RTX5090 task, two CPU cores and
32GiB RAM, with a 4,800-second backend limit, zero retries or extension. At the
GPU-attached published rates this costs at most $3.31, plus $0.10 reserved margin.
Recheck live credit, remaining account allowance, hardware and rates before
invoking. No top-up, training, marketplace substitution or paid automatic retry.
Return an incomplete receipt if the whole cohort cannot finish. Download and
verify every result byte, confirm compute stops, then remove only this run's
temporary volume. Preserve the public v1 and unrelated resources.

### Timing correction, before any completed comparison

The initial diagnostic produced only one v1 question before its admission
estimate rejected further work. That estimate incorrectly divided prefill and
readout costs by seven generated tokens and multiplied them by 512. The failed
receipt and all 15 first-question rows are preserved; no aggregate comparison
was completed. Separate fixed prefill/readout latency from synchronized decode
latency and extrapolate only the latter. A deliberate corrected run keeps the
same cases, weights, contexts and budget; no outcome-dependent selection is
made. Its separately named receipt preserves the failure, and fresh admission
must subtract the first run's actual charge. The backend still has zero automatic
retries and the original balance remains the total spending limit.
