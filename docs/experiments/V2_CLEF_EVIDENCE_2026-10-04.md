# Clef ideas for the Ayaka v2 candidate

Goal: JevBench headline A #1, subject to the existing <=14B parameter limit,
license policy and prepaid GPU budget. Work remains on `ayaka-v2-experiments`.
This change prepares experimental text-feature components, not a newly trained
or promoted model. Published v1 stays the reference.

## What the primary sources establish

Cloudflare describes supervised CE+Brier, permutation augmentation and a later
RL stage with reference preservation and ordinal partial credit. Its advertised
leadership is on **Decision Index**, a separate evaluation. The public description
does not supply the complete RL algorithm, data or coefficients. CE+Brier already
exist in Ayaka v1. [Cloudflare announcement](https://blog.cloudflare.com/clef-decision-models/)

The Flash release uses Qwen3.5-9B and Apache-2.0; full Clef uses Qwen3.8-27B.
The latter exceeds our parameter limit, so it contributes ideas only. The released
model has multimodal input support; the new Ayaka feature prototype below does
not implement a vision processor or claim that support.
[Flash model card](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/README.md),
[Clef model card](https://huggingface.co/Cloudflare/clef/blob/2f3de3dd85f379784083b0814d997ab627200f0c/README.md)

In the released head, candidate queries route into full hidden-state memory,
then field vectors mix through decoder layers. Lexical features use **output-head
rows**. Its prior is a cosine proxy, with a sigmoid residual gate initialized
at 0 (therefore coefficient 0.5), rather than exact native LM logits. Field mixing
has no explicit caller group boundary. The prompt encoder can shorten state
to fit a schema. These details motivate our changes below; they do not establish
a cause of Ayaka's earlier regression.
[Pinned head implementation](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/joint_schema_model.py)

The Flash head configuration is width 1024, two routing layers, four field layers,
16 attention heads and feedforward width 4096. We do not copy this capacity or
assume it fits our latency budget.
[Pinned head configuration](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/joint_head_config.json)

Observed 2026-10-04, JevBench **v1.5.6**, 110 systems:

| System | Headline A | Rank | Intelligence | Calibration | Speed | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Clef Flash | 55.112 | 26 | 53.1 | 87.6 | 83.6 | 46.8 |
| Clef | 16.992 | 63 | 67.9 | 86.7 | 79.9 | 28.1 |

These pages use adjusted latency and estimated cost, not our measured serving
times or bills. Their positions show why importing an architecture alone cannot
justify a JevBench #1 forecast.
[Flash result](https://www.benchmarkheaven.com/jev-models/clef-flash),
[Clef result](https://www.benchmarkheaven.com/jev-models/clef)

The evaluation's typed competence, abstention, calibration, speed/cost treatment
and paired significance matter together. Public-only diagnostics cannot be called
official sealed-inclusive results.
[JevBench v1.5 methodology](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5.md)

## Implemented components and deliberate differences

`ayaka/model/evidence.py` provides `EvidenceResidualHead`, consuming precomputed
native logits, question/candidate states and record evidence. It does not load
weights, generate reasoning or make backend calls.

- The final correction projection starts at zero. Initial valid logits and
  distributions equal the supplied native readout. Only that projection is
  zero-initialized, avoiding a zero-gate/zero-output dead gradient. A two-step
  optimizer test verifies learning reaches the evidence projection.
- Default width 128, four heads, one evidence-routing layer and one field layer:
  **1,904,512 parameters at hidden=4096**, or 2,297,728 at hidden=5120, without
  lexical projection. Count total model+head parameters before selecting a base.
- `group_ids=None` isolates questions. Explicit nonnegative group ids allow
  mixing only within that group and record. `field_layers=0` removes field mixing;
  `routing_layers=0` permits the candidate/query-only ablation.
- Shared memory must be **state-only evidence**. `memory_scope[B,Q,S]` can restrict
  each question's trace bank. It cannot undo contamination already present in
  backbone features. Head-level permutations do not prove causal prompt invariance.
- Optional lexical features require actual output-head features supplied explicitly;
  neither weight tying nor input/output embedding equality is assumed.
- Masks handle padded candidates/questions/records. Default caps are 8,192 memory
  tokens, 64 questions, 128 candidates/question and 2,048 total padded candidates
  per call. Oversize/invalid inputs fail; no silent truncation or invented mass.
  Tensor support above 26 candidates does not supply a new native alphabet readout;
  that adapter must provide legitimate candidate logits with a verified definition.

`ayaka/training/evidence_objective.py` provides `evidence_loss`:

```
gold CE + 0.5 * gold Brier + 0.35 * Score RPS
        + optional beta * KL(frozen reference || student)
```

CE stays enabled. Beta defaults to 0; a reference-preservation experiment must
set it explicitly and supply a matching frozen reference. Soft targets remain
distributions; smoothing defaults to 0 and only changes CE when requested.
Reference/targets are detached and extreme student tails remain in log space.
RPS sorts distinct explicit levels and uses normalized rank CDF error, matching
the existing loss. It is not a numeric-distance metric for irregular intervals.
The caller must establish sample/recipe/candidate-order binding before tensor loss.

Both modules are independently implemented; no downloaded Clef code is vendored
or executed, and the repository's existing license is unchanged. Existing
model/config/checkpoint/API defaults and Claude-owned Swift files are untouched.
In particular, these modules do not alter reasoning `on + high`, effort precedence,
fallback reporting or unsupported-backend validation.

## Experiments that can identify a useful change

Use the same base revision, input recipe, native candidate definition and paired
cohort. Preserve a native frozen arm and a matched published-v1 readout control;
different prompts/backends cannot identify whether fine-tuning caused a regression.

| Arm | Residual features | Cross-question mixing | Reference beta |
|---|---|---|---:|
| B | native logits only | none | n/a |
| S | candidate + question, routing=0, fields=0 | none | 0 |
| E | S + evidence routing=1, fields=0 | none | 0 |
| G | E + fields=1 | explicit related groups only | 0 |
| R | G | same groups as G | 0.05 |

This nested comparison separates learned candidate correction, evidence access,
group context and reference regularization. Beta 0.05 is a proposed single ablation,
not an empirically selected optimum. Lexical features and smoothing stay disabled
in the first comparison; optional arms require a new predeclared workload.

Training, router training, calibration, dev and test remain separate by case/lineage
and template/rule family. Fit policy on calibration; choose architectures on dev;
open independent test only after freezing the choice. Reuse paired rows and bootstrap
whole cases, not correlated questions. Report typed chance-corrected competence,
NLL/Brier/ECE, Score expected value/nMAE/RPS, improvements and regressions, and p95
latency plus measured input/output usage and cost. Corrected argmax alone is insufficient.

Promotion requires paired dev evidence of a headline improvement with independently
measured latency/cost and acceptable typed probability losses. Declare noninferiority
tolerances before fitting, with Claude's independent review. If the interval cannot
distinguish an arm from B, retain the cheaper native arm. Local success does not
establish leaderboard #1; an official evaluation is a separate step.

Regression probes include date boundaries/leap years/month ends/business days/timezones,
rounding, exception precedence/amendments, insufficient information, English plus basic
Korean/Japanese, candidate order/large sets, unrelated extra questions, and prompt
injection inside evidence. Compare grouped versus standalone questions using the
**complete actual prompt/backbone path**, beyond the feature-level tests here.

## Cost and integration boundaries

GPU/API inference used for this work: **0**. Downloads: source/config/license only,
never model weights. No new model survey or paid training arm is launched automatically.

A full bf16 memory bank at 8,192 tokens and hidden=4096 is **64 MiB per state**;
1,000 such states would be 62.5 GiB before other tensors. A trainable projection
cannot simply be pre-applied and cached permanently without changing the training
experiment. Compression savings are unmeasured. vLLM logprob artifacts contain no
full hidden states, and the existing Gemma fastpath does not expose this memory
contract. Therefore feature extraction, layer identity, cache layout and native
logit parity need a separate adapter audit before any real head experiment.

First establish the cheapest correct native baseline with the existing reader/policy
work. Do not expand Claude's current collection job by these arms. If evidence heads
remain worth testing, collect frozen state evidence once and reuse compatible question
features/native references across head-only arms, where the backend actually supports
it. Estimate extraction, storage, optimizer, evaluation and transfer time together.
Before GPU launch, fix datasets/token lengths/steps/epochs/arms and a complete schedule
that fits remaining prepaid credit, within the existing overall maximum of 8 GPU-hours.
The time cap is emergency protection, not a plan to stop an unfinished epoch.

No production integration or full Clef compatibility is claimed. CPU optimizer tests
check mechanics; they do not prove real-model accuracy, speed or JevBench rank.
Clef review requests and ownership are in `.dev/codex-clef-to-claude-20261004.md`;
independent acceptance has not yet been received.

## Reproduce the CPU verification

```powershell
.venv/Scripts/ruff.exe check ayaka/model/evidence.py ayaka/training/evidence_objective.py tests/test_evidence_head.py tests/test_evidence_objective.py
.venv/Scripts/ruff.exe format --check ayaka/model/evidence.py ayaka/training/evidence_objective.py tests/test_evidence_head.py tests/test_evidence_objective.py
.venv/Scripts/python.exe -m pytest tests/test_evidence_head.py tests/test_evidence_objective.py
```

Observed: **80 passed in 10.10s** (44 head + 36 objective); lint/format pass.
The whole-tree lint and format check also passed (220 Python files). Full CPU
integration: **939 passed, 1 skipped in 281.07s**, with no source changes during
execution. The test process explicitly used existing Git Bash; the system environment
was not modified. Its snapshot-bound receipt/log are recorded locally under
`.dev/codex-clef-full-cpu-20261004.*`. Passing existing Swift tests does not resolve
the separately reported R12/R13/R14 preflight/parity limitations by itself.

Pinned source survey: Clef `2f3de3dd85f379784083b0814d997ab627200f0c`, Flash
`17f0b0ad64efb65d273590632833508766b2aae6`; identical head-source SHA256
`0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
The local `runs/clef-audit-20261004/manifest.json` records per-file URLs/hashes.
Source-only survey data and shared `.dev` artifacts remain outside Git.
