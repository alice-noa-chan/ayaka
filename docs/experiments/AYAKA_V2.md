# Ayaka v2 exploration

v2 is an experimental implementation, separate from the published v1 weights.
Its objective is a future JevBench composite Top 5 entry; no ranking improvement
is claimed merely from implementation or procedural training.

## Controls

New v2 checkpoints (`ElectraConfig.version=2`) default to `auto`/`medium`.
Old checkpoints remain direct by default. The old `--reasoning` CLI switch
retains its frozen v1 worked-steps route. New controls start the v2 backend:

```sh
python -m ayaka.serve --ckpt CHECKPOINT --reasoning-mode auto --reasoning-effort medium
python -m ayaka.serve --ckpt CHECKPOINT --reasoning-mode on --reasoning-effort high
```

`--max-reasoning-tokens` overrides the effort limit (128/384/1024). Acceptable
explicit limits are integers from 0 through 1024. Off or zero never generates.
Legacy `--reasoning` cannot be combined with v2 controls. Unsupported generation
requests, invalid modes/efforts, unknown fields and null settings are errors.

```json
{
  "state": "Hello!",
  "options": {"reasoning": {"mode": "on", "effort": "high"}},
  "questions": {
    "polite": {"type": "noul", "instructions": "Is the greeting polite?"},
    "direct": {"type": "noul", "instructions": "Is it a greeting?", "reasoning": {"mode": "off"}}
  }
}
```

Individual fields inherit in checkpoint → server → request → question order.
`on/high` begins reasoning without a speculative direct readout; neither high
confidence, easy input, lack of numbers nor the auto router changes its budget.
Natural EOS may finish before the limit. Context shortages never cause a lower
effort: the response reports fallback and the failure. Empty traces, generation
errors and readout failures similarly return a direct result and diagnostics.

Responses contain `reasoning[question]`: resolved settings, requested budget,
actual route, raw generated count, finish reason, error and timings. Usage counts
raw reasoning tokens including EOS and generation that failed before fallback.
v2 input accounting includes direct prefill, trace prefill and repeated question
readout; v1 input accounting remains the existing external prompt convention.
Worked steps are an internal inference mechanism and are not returned by default.

## Architecture and training

Native output heads are retained even when untied; bias, scaling and softcap are
applied. Non-Gemma chat turns derive from their native templates. Dense caches
can batch branch; recurrent caches branch individually when a layer cannot expand.
v2 reasoning and readout share one active adapter/cache per question. Large
candidate reranking copies the trace cache instead of sharing a sibling readout.

`readout=lm|pointer|hybrid` provides controlled ablations; a pointer with zero
Set Mixer layers is the simple variant. LM-only rejects pointer-only large sets.
Adapter checkpoints support native backbones; standalone export supports Gemma
bf16, with int8 limited to the existing E-series layout.

The offline curriculum is repository-authored MIT data. Calendar, business-day,
time-zone, decimal, probability and rule answers/traces come from deterministic
validators. Calculators never run in inference. Joint SFT mixes direct examples
and complete traces, applies CE only to trace prediction positions, and combines
it with typed decision losses. Empty/overbudget solutions are rejected whole.
No commercial LLM or Jev API output is added to training. Preparation audits
public 13-gram overlap, state identity and template/rule/document/lineage splits.

The curriculum uses different document templates and parameter/rule variants per
split. It shares underlying procedural algorithms and therefore does not replace
an independent natural-language holdout. The objective rubric and three basic
EN/KO/JA shipping regressions are initial mechanics checks, not broad judge or
multilingual capability claims. CPU preparation also reserves evaluation-only
Open-Jev test, HelpSteer2 human ratings, KLUE NLI and JGLUE JNLI examples with
source/license manifests. These never enter training, router fitting, calibration
or ranking. Download failures are reported separately. Further broad evaluation
is needed before publishing v2 weights or raising the training budget.

The auto router predicts paired NLL reduction and generated tokens from observable
direct-result/input features. It compares only λ=0/0.0005/0.001/0.002. Dev selection
uses NLL gain, then fewer tokens within 0.01 nats, and requires the paired bootstrap
lower bound to exceed zero before promotion. This small cost-tie rule makes the
otherwise underspecified λ selection reproducible. No promoted router is shipped
without real paired outcomes; `auto` stays direct with `no_validated_router` when
none is loaded. The legacy v1 reasoning flag keeps its original control. `on`
bypasses routing entirely. Path calibration fits only the
reserved calibration split by primitive/actual route/budget band, with scalar
fallbacks. Load artifacts using `--reasoning-router` and `--reasoning-calibration`.
Router fitting can combine all three effort pairs, but bootstraps independent
questions rather than treating repeated efforts as new samples. Each budget also
requires a positive dev confidence bound; unvalidated budgets stay direct.
Older single-budget artifacts apply only at their measured budget. Forced `on`
requests continue to run at every supported explicit budget.
The independent test also measures actual `auto/medium` decisions alongside off
and forced efforts. Router features use the uncalibrated direct distribution;
output calibration cannot feed back into the routing inputs. Direct outputs use
the zero-generation calibration band even when their requested auto budget is 384.

## Survey and evaluation

[Candidate manifest](v2_candidates.json) pins all 12 official revisions, total
safetensors parameter counts (including multimodal weights), licenses and LoRA
targets. E4B's total size is approximately 8B, not its effective 4B label. Domyn's
card says `license: other`, but its linked [official terms](https://www.domyn.com/legal/software-licenses/domyn-small)
identify MIT; the manifest preserves that evidence. Spark and Nanbeige require
custom-code review and are not silently substituted with another architecture.

The [live JevBench board](https://benchmarkheaven.com/jev-models),
[Arena text board](https://arena.ai/leaderboard/text) and
[Artificial Analysis](https://artificialanalysis.ai/models) were inspected on
2026-10-01. They measure different tasks/modes. Arena coverage is sparse for this
small-model roster; AA v4.3.2 has many values unavailable in the rendered public
view. Missing values are unknown. Older AA indices and effective/active parameter
counts are not used as comparable current measurements.

Local metrics follow the [v1.5 typed method](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5.md)
and [headline-A equal-type amendment](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md):
Noul abstention counts wrong, Score uses expected positions/nMAE/RPS, and task types
are equally weighted. Legacy v1 metrics are versioned separately. Local reports
have `official_composite: null` and `sealed: false`. They cannot establish an
official rank. Unknown cost remains unknown; reference-token estimates require a
source/date and include generated reasoning.

Each candidate uses the same 96 dev questions (32 per type) for direct and forced
128-token screening. Only complete results enter selection. A ≤1 CC-point tie is
resolved by typed proper loss, p95 serial latency, then sourced cost. Head ablations
use the same frozen backbone, training samples and seed; the LM arm is its direct
baseline. Up to two candidates receive joint SFT, followed by effort/route/
calibration and an independent test split. Test never selects the model or λ.
Reports include both repaired and damaged answers and seed-15 paired bootstrap
intervals. Deadline-incomplete runs are recorded and cannot win screening.

## Bounded execution

```sh
python -m ayaka.experiments.v2 prepare --manifest docs/experiments/v2_candidates.json --out NEW_RUN --download-weights
python -m ayaka.experiments.v2 run --manifest docs/experiments/v2_candidates.json --out NEW_RUN
# Hosted equivalent, CPU preparation precedes GPU allocation:
modal run modal_v2.py
```

The run requires exactly one H100 80GB. Stage limits are screen 2h, heads 1h, SFT
2h, evaluation 2h, reproduction/recovery 1h. Child-process deadlines include
loading and report writes. A persistent ledger reserves each allocation before
launch and never refunds it after a crash or resume; a 120-second startup/flush
margin is retained. The Modal GPU function itself has an eight-hour timeout,
one container and no retries. The dedicated `ayaka-v2-exploration` volume stores
prepared data, cached pinned weights, reports, budget ledger and checkpoints.
Use a fresh volume/output directory for a separately authorized exploration.

After the stopped Decimal-ordinal screen, `modal run modal_v2.py --recover-screen`
reuses the prepared volume and original ledger. It retains the interrupted 2h
reservation, spends at most 1h on a replacement screen from the recovery allowance,
and omits the separate reproduction stage. Other stages retain their caps, subject
to the remaining total budget. Reserved allocations are not measured GPU runtime.

Curriculum version 2 also hashes the underlying facts separately from their
document voice. Identical facts cannot cross splits even under paraphrased
wrappers. Timezone parameters, override authorization, rubric weights, bag sizes
and missing-evidence domains now differ across splits. The component algorithms
remain shared; this is a mechanics/generalization probe, not a new natural benchmark.
CPU preparation records the curriculum version and stale preparation is rejected.
The interrupted version-1 screen is archived separately and cannot select a v2
candidate. `--refresh-curriculum` repeats CPU preparation and caps the new screen
at 50 minutes and head ablations at 10 minutes, preserving both prior reservations
and the SFT/evaluation caps. Small candidates run first to leave unused time for
larger candidates; every candidate retains the same 96-question comparison.

## Status on 2026-10-01

- Implementation tests and native CPU cache/logit parity checks have passed.
- Remote CPU preparation downloaded all ten built-in candidates. The two custom
  architectures remain explicit survey-only entries pending code review.
- H100 screening/training results will be recorded separately when measured.
- No v2 weights, official JevBench score, final backbone recommendation or larger
  training budget is implied by this preparation status.
