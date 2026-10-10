---
license: mit
base_model: google/gemma-4-12B-it
base_model_relation: adapter
library_name: ayaka
pipeline_tag: text-classification
language: [en, ko, ja]
tags: [decision-model, jev, jevbench, system-one, calibration, gemma4, lora]
---

# Ayaka v2 large

<p align="center"><img src="https://huggingface.co/alice-noa-chan/ayaka-v2-large/resolve/main/ayaka.png" width="256" alt="Ayaka"></p>

Ayaka v2 large is an open, MIT-licensed decision model for Jev-style
structured decisions. You give it a **state** and typed questions (`noul` /
`choice` / `score`), and it returns a calibrated probability for every
candidate label.

It is a LoRA adapter plus a small decision head on
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)
(text stack only, base revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`).
It ships with its frozen serving policy (`frozen_tuning.json`):

- **Noul questions:** always reasoned (at most 384 tokens of worked steps, then a re-read).
- **Choice and Score questions:** reasoned only when a learned router predicts a benefit.
- **All questions:** per-path temperatures.

It is trained only on license-clean data.

The v1 model, [`alice-noa-chan/ayaka-large`](https://huggingface.co/alice-noa-chan/ayaka-large),
is unchanged and remains available.

Code, training pipeline and documentation:
<https://github.com/alice-noa-chan/ayaka>.

## What's new in v2

**Measured on unseen held-out data, against v1:**

- **+10.8 equal-type CC** over v1 (81.0 vs 70.2, paired 95% CI [+7.2, +14.2]).
- **Choice:** CC 75.4 → 82.6, NLL 0.507 → 0.376.
- **Noul:** CC 67.2 → 93.2, NLL 0.197 → 0.118.
- **Score:** within noise of v1 overall (67.2 vs 68.0, CI [−5.1, +3.9]). See the known weakness below.
- **Far fewer abstentions:** Noul answers inside the 0.2–0.8 band drop from 50 to 2 of 384.
- **Learned reasoning router:** v1 reasons only on calculation questions picked by fixed rules. v2 always reasons on Noul and uses a router, trained and promoted on held-out data, for Choice and Score.

**New capabilities v1 does not have (experimental, not part of the measured results above):**

- **Choice without a full option list.** A Choice question can ask the model to propose candidates. The setting goes in the question itself, or in `ayaka.questions.<name>` when you use the TypeSafe SDK:
  - `"candidate_generation": {"mode": "open", "experimental": true, "scope": "..."}` with no `criteria`: the model writes 2–8 mutually exclusive options and keeps a reserved **Other** residual for anything unlisted.
  - `"mode": "expand"`: splits an existing "other" option of your list into new options.

  Every option, generated or given, is then scored by the decision model.
- **Image inputs.** `serve --images` accepts base64 images in the request's `ayaka.media` field and reads them through Gemma 4's native vision encoder. URLs are never fetched.
  - Text calibration and the router are not applied to image questions.
  - The frozen policy is not applied with `--images`.
  - Image quality has not been measured for this checkpoint.

## Held-out results against v1

The final_test cohort has 836 questions and 431 independent cases. Neither model trained on it, and no fit or choice saw it. It was read once, after the serving policy was frozen. Hardware: one RTX PRO 6000, one request at a time.

| system | equal-type CC | Choice CC | Noul CC | Noul abstentions | Score CC | Speed axis | mean latency |
|---|---|---|---|---|---|---|---|
| v1 `ayaka-large` (its frozen reasoning policy) | 70.2 | 75.4 | 67.2 | 50 | **68.0** | 82.4 | 0.41 s |
| v2, single pass | 73.7 | 79.0 | 75.0 | 35 | 67.2 | 85.9 | 0.13 s |
| **v2 with its frozen policy (default)** | **81.0** | **82.6** | **93.2** | **2** | 67.2 | 58.9 | 3.46 s |

- v2 gains **+10.8 CC** over v1 (paired 95% CI [+7.2, +14.2], whole cases resampled).
- NLL and Brier also improve over v1:

  | type | NLL v1 → v2 | Brier v1 → v2 |
  |---|---|---|
  | Choice | 0.507 → 0.376 | 0.273 → 0.192 |
  | Noul | 0.197 → 0.118 | 0.054 → 0.029 |

### Known weakness: Score on HelpSteer2

**v2 is not better than v1 on Score questions.**

| Score subset | v1 CC | v2 CC | v2 − v1, 95% CI |
|---|---|---|---|
| All Score questions | 68.0 | 67.2 | [−5.1, +3.9] |
| HelpSteer2 rubric questions | 66.6 | 62.3 | [−8.6, −0.1] |

- Across all Score questions the difference is within noise. NLL is 0.792 for v1 and 0.812 for v2.
- On the HelpSteer2 rubric questions v2 is slightly worse. Two reasons:
  - v1 trained on about 101k HelpSteer2 questions, v2 on 15k.
  - The router sends some of these questions to reasoning, where they lose accuracy.

  If your workload is mostly rubric scoring, compare the two models on your own data.

The project predeclared a release gate before this run: no per-type regression against v1, with zero tolerance. **v2 failed that gate on the Score checks** and was therefore not adopted as a replacement for v1. It is published as a separate model, with this result stated openly. The full report:
`docs/experiments/V2_FINAL_RUN_RESULT_2026-10-10.md` in the repository.

JevBench public tiers have not been measured for v2.

## Use (pinned)

```bash
pip install -c https://raw.githubusercontent.com/alice-noa-chan/ayaka/064174dce15145563c12f245af3b93ed9428d01d/constraints.txt \
  "git+https://github.com/alice-noa-chan/ayaka@064174dce15145563c12f245af3b93ed9428d01d"
python -m ayaka.serve --ckpt alice-noa-chan/ayaka-v2-large --device cuda --port 8000
```

The server finds `frozen_tuning.json` in the repository, loads the adapter merged, and applies the frozen policy, router and temperatures. These are the settings the held-out numbers above were measured with.

- **Single pass:** add `--frozen-tuning off --reasoning-mode off`. Every question then takes one forward pass, which is faster (Speed axis 85.9, mean 0.13 s) but less accurate.
- **API:** the server speaks TypeSafe's `POST /v1/systemone` format and returns a probability for every label:
  - `noul`: P(true);
  - `choice`: per-label probabilities;
  - `score`: per-level probabilities and the expected score.

  `usage.input_tokens` and `usage.output_tokens` are reported per request.
- **Files:** this repository holds the adapter (fp32 safetensors, 1.05 GB), the head (`head.safetensors`), the config, the Noul calibration and the frozen policy. The base model is fetched from Google's repository at the pinned revision.

> **Naming.** `electra_config.json` is a legacy alias of `ayaka_config.json`;
> both hold the same config. `adapter/adapter_config.json` records the path
> of the base model on the training machine. It is left byte-identical
> because the Noul calibration is bound to its hash. Ayaka always loads the
> base from `ayaka_config.json`.

## Training

- **Initialization:** a fresh LoRA on the pinned base (not continued from v1). Gold labels only, with no distillation and no teacher outputs.
- **Adapter:** LoRA r 64, alpha 128, dropout 0.05, on q/k/v/o and gate/up/down projections.
- **Schedule:** 2 epochs of 37,040 rows, 32 rows per step (2,315 scheduled steps), cosine with 3% warmup. LoRA LR 3e-5, head LR 1.5e-4.
- **Losses:** NLL 1.0, Brier 0.5, RPS 0.35, missing-label 0.25.
- **Checkpoint selection:** every 250 steps on a reserved split, with early stopping after three reads without improvement. Training stopped at step 2,000; the released weights are the step-1,250 state.
- **Serving fit:**
  - per-path temperatures fitted on a reserved calibration cohort;
  - the router fitted on router_train and promoted on dev;
  - the policy chosen on router_train.

  All of it was frozen and hash-sealed before the final_test read.
- **Hardware:** one RTX PRO 6000 (96 GB); the whole run took 7.6 h.

### Training data

No commercial-LLM outputs and no Jev API labels were used. Held-out cohort items were excluded by source group, lineage and 13-word overlap.

| source | dataset | licence | train rows |
|---|---|---|---|
| helpsteer2 | nvidia/HelpSteer2 | CC BY 4.0 | 3,000 responses (15,000 Score questions) |
| commonsense_qa | tau/commonsense_qa | MIT | 6,400 |
| massive_ko | AmazonScience/massive (ko-KR) | CC BY 4.0 | 3,000 utterances |
| massive_ja | AmazonScience/massive (ja-JP) | CC BY 4.0 | 3,000 utterances |
| contract_nli | ContractNLI | CC BY 4.0 | 108 documents |
| strategyqa | ChilleD/StrategyQA | MIT | 1,036 |
| authored | generated and verified by this repository | MIT | 256 per type |

CC BY sources were used as training data only. Their licence and attribution are listed here.

## Benchmark disclosure

- **Non-public data only:** training, checkpoint selection, calibration, router fitting, policy choice and the release gate all used non-public held-out cohorts drawn from validation and test splits.
- **No JevBench in training:** no JevBench items or labels are in the training corpus.
- **No public results in this release:** public JevBench results were not used for any choice in it.

## License

MIT for code and fine-tuned weights. The base model
`google/gemma-4-12B-it` is Apache-2.0, and its notices apply to the base
portion.
