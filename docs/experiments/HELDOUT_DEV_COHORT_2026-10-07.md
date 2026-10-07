# Held-out development cohort and training-exposure check, 2026-10-07

The October 7 development cohort was drawn from the HelpSteer2,
CommonsenseQA and MASSIVE **train** files. Published v1 was trained on almost
every row of those same files, so the v1/v2 comparison partly measured
memorization. This report records that finding and the replacement cohort,
built only from splits that neither model trained on. CPU only: no model was
run and nothing was fitted.

## The training-exposure problem

From v1's `data_summary.json` and `dataset_manifest.jsonl` (all `train`
splits) and the 10-07 bundle:

| 10-07 dev source | Rows it was drawn from | Published v1 trained on | Trained v2 trained on |
|---|---|---|---|
| HelpSteer2 | train | 101,175 of 101,620 train questions | 320 questions |
| CommonsenseQA | train | 9,609 of 9,741 train rows | 384 |
| MASSIVE ko / ja | train | 11,085 / 11,006 of 11,514 rows (as 60-way intent Choice) | 96 each |
| ContractNLI | train | none | 1,088 |
| ayaka-v2-verified | repository generator | none | 384 (other indices) |

The exact v1 row selection was not saved, but with these coverage rates
nearly every HelpSteer2, CommonsenseQA and MASSIVE dev item was very likely
in v1's training data. Each question was a disjoint item for v2.

### Per-source comparison on directly read questions (C3)

Only the questions both systems read without worked steps (v1's baseline
route), so reasoning does not confound the comparison. Local CC:

| Source | Type | n | v1 | v2 off | v2 − v1 | v2 training share | v1 exposure |
|---|---|---:|---:|---:|---:|---:|---|
| ContractNLI | Choice | 130 | 67.7 | 82.7 | **+15.0** | 46% | none |
| ayaka-v2-verified | Choice | 21 | 86.0 | 100.0 | +14.0 | 16% | none |
| ayaka-v2-verified | Noul | 22 | 81.8 | 54.5 | −27.3 | 16% | none |
| ayaka-v2-verified | Score | 32 | 68.3 | 77.3 | +9.0 | 16% | none |
| CommonsenseQA | Choice | 32 | 72.7 | 60.9 | **−11.7** | 16% | ≈99% of train rows |
| HelpSteer2 | Score | 80 | 60.4 | 58.8 | −1.6 | 14% | ≈99.6% of train questions |
| MASSIVE ko | Noul | 16 | 25.0 | 25.0 | 0.0 | 4% | ≈96% |
| MASSIVE ja | Noul | 16 | 50.0 | 62.5 | +12.5 | 4% | ≈96% |

v2 leads where v1 had no exposure and v2 had substantial data
(ContractNLI); v1's clearest lead is CommonsenseQA, where it very likely
trained on the evaluated questions. Group sizes are small, so this is
consistent with memorization, not proof of it. The verified Noul drop is
the calculation abstention analysed in the
[CPU cause checks](CAUSE_CHECKS_CPU_2026-10-07.md).

## The held-out cohort

`python -m ayaka.eval.heldout_cohort` builds it from pinned files
([settings](heldout_dev_settings.json),
[manifest](results/heldout-dev-cohort-20261007-manifest.json)).

| Source | Original split | Questions | Independent cases | Type |
|---|---|---:|---:|---|
| HelpSteer2 | validation | 120 | 24 | Score |
| CommonsenseQA | validation | 96 | 96 | Choice |
| ContractNLI | dev | 68 | 4 | Choice |
| MASSIVE ko + ja | validation | 128 | 32 | Noul |
| HotpotQA (yes/no) | validation | 96 | 96 | Noul |
| StrategyQA | test | 64 | 64 | Noul |
| ayaka-v2-verified (synthetic) | dev generator, indices 32–63 | 96 | 26 | all three |
| **Total** | | **668** | **342** | Choice 196, Noul 320, Score 152 |

Natural questions: 572 in 316 cases; synthetic: 96 in 26 cases. The 10-07
cohort had 376 questions in 99 cases with 64 Noul questions.

Rules the builder enforces:

- **Held-out splits only.** No natural question comes from a train file. The
  MASSIVE validation files were downloaded at the training revision
  (`ed58ac4`); every other file was already cached. Each file is pinned by
  SHA-256.
- **Train-text exclusion.** An item whose normalized text appears in the
  matching train file is dropped: HelpSteer2 prompt, CommonsenseQA question,
  MASSIVE utterance in either language, HotpotQA and StrategyQA question,
  ContractNLI document. The validation files contain such duplicates (116
  Korean and 134 Japanese MASSIVE utterances, one HelpSteer2 prompt, one
  StrategyQA question); four MASSIVE candidates were excluded during
  selection.
- **Same task wording as training.** HelpSteer2, CommonsenseQA, MASSIVE and
  ContractNLI questions come from the functions training preparation uses
  (`direct_natural.task_view`, `contract_nli.document_questions`); the
  refactor that shared them left the complete training conversion
  byte-identical (fingerprints of all five sources and of gold verification
  match). HotpotQA and StrategyQA use the v1 loader views.
- **Independent cases.** One HelpSteer2 response per prompt; Korean and
  Japanese translations of one MASSIVE utterance share a lineage and count
  as one case.
- **Fresh synthetic items.** Verified questions use generator indices 32–63,
  after the 0–31 that every earlier dev cohort used.
- **Whole inputs.** Every sample fits 7,680 tokens both through trained v2's
  Swift contract and through v1's untruncated native format, leaving room
  for 384 reasoning tokens.
- **Other exclusions.** JevBench public 13-gram overlap, and any lineage or
  state shared with the 10-07 dev and calibration files. The private
  holdout test was not opened: it comes from train files, which the
  train-text rule already excludes.
- **Deterministic.** A rebuild from the same pinned files produced a
  byte-identical `dev.jsonl`.

Gold balance on Noul is 45–50% true for every source.

## Limits

- HotpotQA and StrategyQA are new task types for v2, while v1 trained on
  their train splits; format familiarity still favors v1 on those two.
- StrategyQA includes its supporting facts in the state, as in v1's
  training view, which makes it easier than the original task.
- ContractNLI contributes only 4 documents (68 questions); its CC moves in
  steps of one document's worth of questions.
- The synthetic part is in-distribution for v2's training generator. Report
  it separately from natural sources, as the manifest's `by_data_kind` does.
- The cohort data is not committed: HotpotQA is CC BY-SA 4.0. The builder
  regenerates it byte-for-byte from the pinned files.

## Reporting change (B4)

`ayaka.eval.v2.summarize` now also reports `cc_question_weighted`, each
type's CC weighted by its question count, next to the equal-type headline.
On the 10-07 cohort one Noul question moved the equal-type headline by about
1 point; the new field shows how much of a difference comes from that
weighting. The declared headline is unchanged.

## Reproduce

```bash
python -m ayaka.eval.heldout_cohort \
  --settings docs/experiments/heldout_dev_settings.json \
  --contractnli-archive .dev/contract-nli-source-20261005/contract-nli.zip \
  --checkpoint .dev/v1v2-compare-20261007/payload/v2 \
  --tokenizer .dev/native-gemma4-12b-20261005 \
  --exclude .dev/vast-train-20261007/bundle/dev.jsonl \
  --exclude .dev/vast-train-20261007/bundle/calibration.jsonl \
  --out runs/heldout-dev-20261007
```

The expected `dev_jsonl_sha256` is
`c4454d6b829bfe9fdd18f0d4676f812a89bd9349b8b88a641a8241e5ee23fbcb`.
