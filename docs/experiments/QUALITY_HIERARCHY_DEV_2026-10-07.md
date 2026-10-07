# Three-system development quality screen

`ayaka-quality-hierarchy-dev-1` is a new screen for future development runs.
It does not amend the frozen October 5 matched-comparison experiment.

The required order is:

1. v2 direct beats v1 with forced reasoning by at least 5 equal-type CC points
   and 5 thresholded Choice/Noul credit points.
2. v2 with forced reasoning beats v2 direct in equal-type CC.
3. Both gains have a positive lower bound in the paired 95% bootstrap interval.
4. Neither comparison worsens any type's CC, NLL, or Brier. Score normalized
   RPS and nMAE must also be nonworse. Source/type and language/type CC cannot
   regress. Numerical regression tolerance is 1e-8.
5. The cohort contains all three types and at least 200 independent source
   lineages. v2 reasoning must complete at least one reasoned read; an entirely
   fallback run cannot establish a reasoning improvement.

Each system must cover the same canonical development questions exactly once.
The checker validates prepared evidence/menu/target fingerprints, candidate
order, targets, ordinals, source lineage, source, language, tier, modality,
partition, split, external checkpoint identity, and selected reasoning budget.
It recomputes every metric from stored probabilities rather than trusting
cached summaries. Failures and fallback distributions remain part of the
served-system measurement.

Produce current `ayaka.eval.pretraining_v2` track reports for both checkpoints,
including `off` and the chosen forced effort. Use a canonical `Sample` JSONL
development cohort and independently recorded checkpoint fingerprints:

```bash
python -m ayaka.eval.quality_hierarchy \
  --v1-report v1-dev.json --v2-report v2-dev.json \
  --samples dev.jsonl --v1-model-id "$V1_CHECKPOINT_SHA256" \
  --v2-model-id "$V2_CHECKPOINT_SHA256" \
  --v1-mode high --v2-mode high --out hierarchy-dev.json
```

The command returns 0 for a passing screen and 1 for a failed screen. Reports
are written once and record the bytes consumed from every input artifact.
Older reports missing explicit candidate identities/source fields must be
recollected under the declared new recipe. No metrics are fitted by this command.

The bootstrap resamples whole source lineages within type/tier coverage strata.
Declare shared evidence and translations under the same source lineage during
preparation. The interval is conditional on the supplied cohort and selected
models; it does not correct for repeated model or prompt searches.

A passing result remains experimental and non-promotable. This checker does
not attest actual model execution or fresh data independence. Production
promotion requires separately checked execution evidence and an independent
final holdout after development choices are frozen. Speed and cost require
matched execution hardware and are not decided by this quality screen.

## Known limitations (added 2026-10-07)

The [gap analysis](V2_GAP_ANALYSIS_2026-10-07.md) of the first matched run
found two limits of this screen as declared. They are recorded here; the
rule itself is unchanged.

- Rule 1 compares v2 with reasoning off against v1 with reasoning. On the
  calculation questions this compares a single direct read with worked
  steps, so it mixes model quality with inference budget. A matched
  direct-versus-direct and routed-versus-routed pair would separate them.
- Rule 5 needs at least 200 independent cases. The first development cohort
  had 99, so it could not pass whatever the result.

The per-source and per-language non-regression checks stay blocking: in
that run reasoning gains came from the synthetic source while natural
sources were flat or worse, which is the case these checks exist to catch.
