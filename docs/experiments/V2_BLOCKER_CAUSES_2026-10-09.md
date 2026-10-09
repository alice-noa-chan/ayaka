# Causes of the three remaining v2 gate blockers (2026-10-09)

**Scope:** a CPU-only diagnosis of why the best v2 state (merged `noul_always+router`, temperatures fitted on the 334-question splits, dev CC 75.62) still fails three checks of the v1 gate. No GPU was rented, nothing was fitted on dev, nothing is adopted and no serving default changes. Only aggregates are recorded here, never dataset rows.

- **Inputs:** the merged reads of `V2_HELDOUT_MERGED_2026-10-09.md`, the 10-08 `dev/v1_on` read, and the 1,002-question router_train extension of `V2_TUNING_EXTENSION_RESULT_2026-10-09.md`.
- **Method:** the policy rows were rebuilt with the functions of `ayaka.eval.heldout_tuning` (`load_run`, `calibrated`, `fit_router`, `compose`, `policy_rule`). They reproduce the reported dev CC of 75.62 exactly. Per-source intervals use a cluster bootstrap (2,000 replicates, seed 15) over each source's `cluster_id`.
- **Dev use:** dev was only broken down, never used to choose anything. The one remedy that was tested (section 3) used only the calibration and router_train splits.

## Summary

| Blocker | v1 | v2 | v2 − v1, 95% cluster CI | Real effect? | Main cause |
|---|---|---|---|---|---|
| commonsense_qa Choice CC | 81.77 | 79.17 | −2.60 [−11.72, +6.51], 96 clusters | not distinguishable from 0 | net 2 questions; v1 trained on ~9.6k CommonsenseQA rows, v2 on 256 |
| helpsteer2 Score CC | 55.65 | 53.99 | −1.66 [−5.72, +3.31], 24 clusters | not distinguishable from 0 | v1 trained on ~101k HelpSteer2 questions, v2 on 256 samples; per-attribute label-prior bias |
| Noul NLL | 0.223 | 0.232 | MASSIVE ko +0.140 [−0.001, +0.315]; StrategyQA +0.112 [+0.027, +0.229] | **yes, reproducible** | reasoning on MASSIVE produces confident wrong answers; v1 trained on StrategyQA and MASSIVE, v2 had no StrategyQA |

Two of the three blockers are within sampling noise of a zero-tolerance check. The third is a real calibration trade-off introduced by the reasoned route on MASSIVE, together with training-data coverage.

## 1. Training scale explains most of the per-source gaps

v2 and v1 were trained at very different scales (`TRAINED_CHECKPOINT_COMPARISON_2026-10-07.md`, `HELDOUT_DEV_COHORT_2026-10-07.md`):

| | Published v1 (large) | Trained v2 |
|---|---|---|
| LoRA | rank 64, alpha 128 | rank 32, alpha 64 |
| Optimizer updates | 860 | 74 |
| Training questions | ~35 sources, up to 20k rows each | 2,368 gold-only questions |
| CommonsenseQA train | 9,609 of 9,741 rows | 256 samples |
| HelpSteer2 train | 101,175 of 101,620 questions | 256 samples (5 questions each) |
| StrategyQA train | yes (`ChilleD/StrategyQA` train) | none |
| MASSIVE ko / ja train | ~11k utterances each | 512 samples each |

The held-out cohort uses the validation or test splits of these sources, so neither model saw the dev questions. v1 still learned each source's format, label ontology and label prior from thousands of in-distribution rows. v2 did not.

[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (Biderman et al., TMLR 2024) finds that LoRA needs higher ranks and longer training than usual to approach full fine-tuning. v2 is at the low end on both counts.

## 2. commonsense_qa and helpsteer2: below the resolution of the check

### commonsense_qa (96 questions, 96 clusters)

- v1 and v2 agree on 84 of 96 questions. v2 loses 7 questions and wins 5, so the CC gap is a net of 2 questions.
- The bootstrap probability that v2 is truly worse is 0.65.
- v2 is overconfident when wrong: its mean confidence on wrong answers is 0.81, against 0.62 for v1. This drives Choice NLL to 0.547, against 0.394 for v1.
- Reasoning does not resolve it. On router_train (144 questions), reasoned CC is higher (83.5 vs 77.4) but NLL is worse (0.523 vs 0.505). This matches [To CoT or not to CoT?](https://arxiv.org/abs/2409.12183) (Sprague et al., ICLR 2025): chain of thought helps mainly on math and symbolic reasoning, with little gain on commonsense.
- CommonsenseQA has measurable label noise and ambiguous items ([noise audit](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11016068/), Scientific Reports 2024; [Plausibly Problematic Questions](https://arxiv.org/abs/2410.10854), 2024). On 96 items a two-question gap is within that noise.

### helpsteer2 (120 questions, but only 24 responses)

Each response contributes five attribute questions, so the source has only 24 independent clusters. The bootstrap probability that v2 is truly worse is 0.76. The router never reasons on Score, so this is the direct read alone.

| Attribute | v1 CC | v2 CC | v1 mean bias | v2 mean bias | v1 sd | v2 sd |
|---|---|---|---|---|---|---|
| coherence | 59.8 | 57.7 | +0.21 | −0.21 | 0.67 | 0.88 |
| complexity | 64.9 | 58.8 | −0.04 | +0.29 | 0.65 | 0.86 |
| correctness | 49.4 | 45.7 | +0.18 | −0.09 | 1.07 | 1.14 |
| helpfulness | 50.5 | 51.3 | +0.02 | +0.10 | 1.07 | 1.03 |
| verbosity | 54.6 | 57.3 | −0.27 | −0.08 | 0.64 | 0.74 |

Bias is the expected score minus gold on the 0–4 scale. sd is the mean standard deviation of the predicted distribution.

- v2 wins two attributes and loses three.
- Its losses come from per-attribute shifts of the predicted mean (coherence too low, complexity too high) and from wider distributions. Both are signs of a label prior learned from too little data. v1's sharper per-attribute priors come from ~101k training questions.
- Reasoning makes Score worse on router_train (CC 56.5 vs 60.9), which is why the router never selects it.
- Score CC is computed from the expected position, which is the Bayes-optimal point prediction for squared error. [Regression-aware fine-tuning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ae3bb0adcdce28afb5f7e86a79b6cffa-Abstract-Conference.html) (Lukasik et al., ICLR 2025) trains on that expected value directly rather than with cross-entropy alone. Ayaka already trains Score with Brier and RPS, so the data scale is the more likely gap.

## 3. Noul NLL: a real trade-off from the reasoned route

The policy reasons on every Noul question (`noul_always`). Summed over sources, the change in NLL against v1 is:

| Source | n | v1 NLL | v2 policy NLL | v2 direct NLL | Summed NLL change |
|---|---|---|---|---|---|
| hotpotqa | 96 | 0.225 | 0.204 | 0.328 | −2.04 |
| massive_ja | 64 | 0.216 | 0.230 | 0.231 | +0.84 |
| massive_ko | 64 | 0.260 | 0.400 | 0.254 | **+8.95** |
| strategyqa | 64 | 0.074 | 0.186 | 0.201 | **+7.16** |
| ayaka-v2-verified | 32 | 0.456 | 0.083 | 0.627 | −11.96 |

- **The loss is concentrated in a few confident errors.** Twenty-one questions with |p − y| > 0.8 carry 72% of v2's Noul NLL; v1 has 14 such questions, carrying 50%. v2 removes the uncertain middle almost completely: 7 questions with 0.2 < |p − y| ≤ 0.8, against 41 for v1. That is exactly what removes the abstentions (41 → 7), and it costs NLL wherever the decision is wrong.
- **The MASSIVE effect is reproducible on the tuning splits.** On router_train (334 + 1,002 extension), reasoning raises MASSIVE CC strongly (ko 64.6 → 81.2, ja 62.5 → 83.3). It also raises NLL (ko 0.154 → 0.242, ja 0.193 → 0.257). This is not dev noise.
- **A per-language temperature does not fix it** (fitted on calibration, measured on router_train only):

  | Language | Reasoned, global T 2.85 | Reasoned, per-language T | Direct, T = 1 |
  |---|---|---|---|
  | ko | 0.242 | 0.249 (T 3.45) | 0.151 |
  | ja | 0.256 | 0.256 (T 3.15) | 0.191 |

  On router_train, 16 of 192 reasoned MASSIVE reads are confidently wrong, and they carry 75% of the source's NLL. Softening every read enough to hide them would give back the CC gain.
- **This matches the literature on reasoning and confidence.**
  - [Calibration Drift Under Reasoning](https://arxiv.org/abs/2606.11211) (2026) reports that longer reasoning can inflate confidence on wrong answers.
  - [Reasoning Models Better Express Their Confidence](https://arxiv.org/abs/2505.14489) (2025) finds that reasoning models are better calibrated overall. The effect therefore depends on the task, which fits the split here: hotpotqa and the verified items improve, MASSIVE worsens.
  - [Multicalibration for Confidence Scoring in LLMs](https://proceedings.mlr.press/v235/detommaso24a.html) (ICML 2024) explains why one global temperature cannot be calibrated for every group at once. The per-language result above shows that even a group temperature is not enough when the errors are confident rather than uniformly sharp.
- **StrategyQA is a coverage gap.** v1 trained on the StrategyQA train split in the same facts-plus-question format and reaches NLL 0.074. v2 never saw StrategyQA. Its reasoned read improves on its own direct read (0.201 → 0.186), but it cannot reach v1's in-distribution sharpness.

## 4. The gate is close to its statistical resolution

Against v1, the screen applies about 26 zero-tolerance checks:

- per type: CC, NLL, Brier, RPS and nMAE;
- per source and type;
- per language and type.

Some per-source groups have only 24–96 clusters. Two systems of equal true quality would each fail such a check about half the time. A model that is clearly better overall (+8.15 CC, CI +3.73 to +12.44) can still fail a few checks by chance.

This is the standard subgroup-analysis problem: too many small subgroups give false negatives ([MJA, 2004](https://www.mja.com.au/journal/2004/180/6/subgroup-analysis-clinical-trials); [Burke et al., BMJ 2015](https://www.bmj.com/content/351/bmj.h5651.abstract)). [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (Miller, 2024) recommends clustered standard errors for exactly this kind of correlated item set, such as the five HelpSteer2 attributes of one response.

Any change to the rule is the project owner's decision. It must be made on principle, not to pass this result. Because the outcome on this dev is now known, the confirmation has to use the unopened private test, once.

## What would actually move each blocker

| Blocker | Free (CPU) | Needs GPU training |
|---|---|---|
| commonsense_qa Choice | none needed if the rule uses clustered non-inferiority; otherwise nothing post-hoc helps | more CommonsenseQA train data (MIT), as v1 had |
| helpsteer2 Score | same as above | more HelpSteer2 train data (CC BY 4.0) to learn the per-attribute priors |
| Noul NLL | not fixable by temperature (shown above); reading MASSIVE directly instead would be a new post-hoc policy on a dev used four times | StrategyQA train data (MIT); more MASSIVE ko/ja data (CC BY 4.0); proper-score supervision on reasoned natural Noul reads |

Every source in the right-hand column is already in v1's license-clean training mix. The data side of the training option therefore needs no new licensing review. Scaling v2 towards v1's data and step counts is a GPU decision for the project owner. It needs its own estimate and a predeclared gate.
