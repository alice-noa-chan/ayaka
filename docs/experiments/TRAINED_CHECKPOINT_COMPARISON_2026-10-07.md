# Published v1 and 74-step trained v2 comparison

The trained v2 checkpoint does **not** meet the requested quality hierarchy.
On the same 376 development questions, v2 reasoning improves over v2 direct,
but v2 direct fails to beat published v1 with its existing reasoning policy.
The observed CC order is **v2 on > v1 on > v2 off**.

The machine-readable report is
[trained-checkpoint-comparison-20261007.json](results/trained-checkpoint-comparison-20261007.json).
These are local development CC scores, not official JevBench scores or ranks.

## Complete matched results

CC adjusts classification credit for chance and Score error for a chance
reference. The headline averages the three type scores equally; it is not raw
accuracy. Noul retains its fixed 0.2/0.8 abstention thresholds, and an abstention
receives zero credit. Existing difficulty-tier weights remain unchanged.

| System | Equal-type CC | Choice CC, 200 questions | Noul CC, 64 questions | Score CC, 112 questions |
| --- | ---: | ---: | ---: | ---: |
| Published v1, existing reasoning policy | 63.09 | 70.24 | 56.25 | 62.78 |
| Trained v2, reasoning off | 57.93 | 78.22 | 31.25 | 64.31 |
| Trained v2, reasoning on | 77.87 | 81.13 | 81.25 | 71.24 |

| Declared comparison | CC difference | Paired 95% interval | Classification credit difference |
| --- | ---: | ---: | ---: |
| v2 off minus v1 on | -5.16 | [-13.93, +3.73] | +1.14 percentage points |
| v2 on minus v2 off | +19.94 | [+11.23, +29.12] | +7.58 percentage points |

Intervals use 2,000 bootstrap repetitions, seed 15, and whole underlying cases
within type/tier coverage strata. The 376 questions represent 99 independent
cases. The intervals are conditional on this development cohort and the frozen
systems; they do not establish an official benchmark rank.

## What failed

Direct v2 improves aggregate Choice and Score CC, but loses 25.00 Noul CC points.
That Noul regression outweighs the improvements in the equal-type headline.
The verified source's Noul delta is -56.25 CC; CommonsenseQA Choice also loses
11.72 CC. ContractNLI Choice improves by 17.65 CC. The required direct-over-v1
gain of at least 5 CC and 5 classification credit points is not achieved, and
its paired CC interval crosses zero.

Reasoning improves all three aggregate type CC scores and has a positive paired
interval. Classification credit increases on 28 questions and decreases on 8; Score nMAE
improves on 71 questions and worsens on 41. However, its probability losses
worsen, and ContractNLI Choice loses 4.41 CC relative to direct v2.

Direct Noul loses credit primarily through the fixed abstention rule:

| Noul diagnostic | v1 on | v2 off | v2 on |
| --- | ---: | ---: | ---: |
| Abstained questions | 7 | 18 | 0 |
| Served classification credit | 78.125% | 65.625% | 90.625% |
| Argmax credit without abstention, diagnostic only | 85.938% | 87.500% | 90.625% |

Direct v2's diagnostic binary argmax credit is slightly higher than v1's,
while its served credit falls because more predictions are withheld. This
diagnostic preserves the primary result: the declared CC still uses the frozen
abstention thresholds. Changing those thresholds on these questions would
change the experiment.

| Probability metric, lower is better | v2 off | v2 on |
| --- | ---: | ---: |
| Choice NLL | 0.35708 | 0.52841 |
| Choice Brier | 0.19672 | 0.23109 |
| Noul NLL | 0.35971 | 0.53919 |
| Score NLL | 0.87505 | 1.40768 |
| Score Brier | 0.48241 | 0.65346 |
| Score normalized RPS | 0.08528 | 0.09722 |

The existing strict development screen fails for both comparisons. It also
requires at least 200 independent cases; this cohort has 99. No acceptance
thresholds, prompts, budgets, weights, or temperatures were changed to improve
the result. The original private test remains unopened and the checkpoint is
not promoted.

The next development priorities are direct Noul confidence under the declared
abstention policy and source-specific regressions, followed by reasoned
probability calibration and the ContractNLI regression. Further training should
distinguish withheld correct predictions from incorrect assertions. Reasoned
temperatures need independent validation: this checkpoint
was trained with gold-only direct reads, and this comparison reuses its saved
temperatures without fitting on the development questions. This experiment
does not establish a causal explanation for the observed regressions.

## Frozen systems and inputs

- Backbone: `google/gemma-4-12B-it`, revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Published v1: `alice-noa-chan/ayaka-large`, revision
  `267605ee22f2b5f934d81e5fbee691952d2e6f55`; rank-64, alpha-128 LoRA.
  Its unchanged calculation gate selects 27 reasoned and 349 baseline reads.
  Worked steps use the backbone with the v1 adapter disabled; decision reads
  retain the v1 adapter. This is the published policy, not forced reasoning
  on every v1 question.
- Trained v2: rank-32, alpha-64 LoRA; 74 optimizer updates over 2,368 gold-only
  training questions. Both modes use the same complete saved checkpoint and
  Swift input contract. Direct reads complete on all 376 questions.
- v2 on: forced medium reasoning, maximum 384 generated tokens, saved
  temperatures. All 376 questions use the reasoning route, with zero errors;
  361 generations finish at EOS and 15 reach the declared token limit.
- Source mix: verified 96, HelpSteer2 80, CommonsenseQA 32, MASSIVE Korean 16,
  MASSIVE Japanese 16, ContractNLI 136. Languages: English 344, Korean 16,
  Japanese 16. Candidate identities, ordering, targets, and source lineages
  match across all systems.
- All 376 direct-v2 token bindings match the original training evaluation
  inputs. No optimizer updates occur during comparison.

## Native continuation repair and verification

The first v2-on attempt exposed a real native-template boundary bug. Gemma 4's
generation-only empty thought channel is omitted when the assistant becomes
chat history. The old Swift continuation rejected that changed generation
prefix. Seven initial observations are preserved in
`failed-v2_on-initial.jsonl`, separately from the complete corrected arm.

Commit `ffb222a13abf9d9a809b254e562f173e89217675` validates the original
conversation independently of the generation cue. It rejects rewritten
evidence and rereads the complete canonical final history when the cache prefix
changes. Exact compatible prefixes keep cache reuse. All 376 actual Gemma 4
reasoned reads use the explicit `canonical_full_read` path; its additional
input cost is recorded rather than claiming cache reuse. Ruff lint/format
checks and 116 relevant tests pass; native-tokenizer preflight also passes for
all 376 questions.

The valid v1/off observations were collected under
`662658abfba4e81cbe3dcd2e90e48715bd0a4cc6`. Only the complete on arm was
recollected under `ffb222a`; checkpoint weights and declared model settings
were unchanged. CPU scoring uses `ffb222a` and recomputes metrics from stored
probabilities and canonical targets.

The recovered archive contains 22 indexed files. Every file size and SHA-256,
all three complete question sets, input bindings, checkpoint identities, and
repair provenance were verified locally before rental destruction.

| Retained artifact | SHA-256 |
| --- | --- |
| Development JSONL | `f4a1b8d124a5c80c737ed4b5f7a2f2463c8e1d18eefd37220efa600c538b296b` |
| Canonical protocol | `03be4934b197b02b8fb304f4d6d50f93311bc231422305dffd7be69b641ffd19` |
| v1 checkpoint | `84a69af538f9ec5e9e9061dd7862a8b5a53dcf7d0c9267404425fb83747e55a3` |
| v2 checkpoint | `752534912fd83f3753b9b8d25e97cd66bc68a686faf371e321417818f20e3f01` |
| Repaired source ZIP | `d1c2b2f4f6055117b9703424fb68da9453734f0343018b2b50a60867e51e9ed1` |
| Recovered result archive | `90e65400389a54775818c7ddc3b9303cccdc3ed4519f41480d5f7a95abeb3830` |

## Rental and retained results

The comparison ran on one RTX PRO 6000 Blackwell Server Edition, 96 GB,
500 W. Instance `54582949` was destroyed after successful local recovery,
and its absence was independently confirmed. CPU bootstrap scoring ran after
GPU release. The accepted rate was $1.444533/hour including 180 GiB storage;
the 3.12-hour rental implies an estimated $4.50 compute/storage cost. Transfer
is separate, and this estimate is not a final invoice.

The operational archive, checkpoints, raw observations, receipts, and scripts
are retained in `.dev/v1v2-compare-20261007/` and
`.dev/vast-train-20261007/`. The CLI's initial destroy command aborted at its
confirmation prompt; recovery explicitly used `--yes` only after local result
verification, and the retained rental helper now includes that flag for
authorized destruction.

Recompute the quality summary from the repository root:

```bash
python -m ayaka.eval.checkpoint_comparison compare \
  --samples .dev/v1v2-compare-20261007/retrieved/results/dev.jsonl \
  --protocol .dev/v1v2-compare-20261007/retrieved/results/protocol.json \
  --v1-on .dev/v1v2-compare-20261007/retrieved/results/v1_on.jsonl \
  --v2-off .dev/v1v2-compare-20261007/retrieved/results/v2_off.jsonl \
  --v2-on .dev/v1v2-compare-20261007/retrieved/results/v2_on.jsonl \
  --replicates 2000 --out comparison-recomputed.json
```

The command requires a new output path. Raw results retain the failed initial
on attempt; the report uses the complete repaired on arm only. No published
checkpoint or private test was modified.
