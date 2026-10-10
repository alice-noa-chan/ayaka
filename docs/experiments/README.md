# Experiment index

One line per report: the question it asked, what it found, and whether the
result is still current. Reports are append-only records; a later report
supersedes an earlier one instead of editing it. Status values:

- **current**: the latest word on its question.
- **adopted**: the result changed a default or a serving policy.
- **rejected**: a candidate that failed its predeclared rule.
- **superseded**: replaced by a later report (linked).
- **reference**: a protocol, contract or design note rather than a measurement.

All scores are local development measurements unless a row says otherwise;
none is an official JevBench score or rank.

## Start here

| Report | What it says | Status |
|---|---|---|
| [Final v2 run result, 10-10](V2_FINAL_RUN_RESULT_2026-10-10.md) | Last predeclared v2 run: on the unseen final_test cohort the frozen noul_always+router policy gains +10.8 CC over v1 (CI +7.2…+14.2), Speed 58.9, but fails the zero-tolerance Score checks and the clustered HelpSteer2 check, so it is not adopted. The owner published it separately as ayaka-v2-large with the result disclosed. | current |
| [Final v2 run predeclaration, 10-10](V2_FINAL_RUN_PREDECLARATION_2026-10-10.md) | Corpus plan 3, training, selection, tuning and the once-only final_test decision rule, with two amendments made before results. | reference |
| [V2 blocker causes, 10-09](V2_BLOCKER_CAUSES_2026-10-09.md) | Two of three gate blockers were noise-level; Noul NLL came from reasoned MASSIVE reads and missing StrategyQA; v2 was under-trained. | superseded by the final run |
| [V2 gap analysis, 10-07](V2_GAP_ANALYSIS_2026-10-07.md) | The v2 direct deficit comes from the 27 questions where v1 reasons; on directly read questions v2 already leads. Remaining abstentions are calculation and ko/ja intent questions. | current |
| [Held-out dev cohort, 10-07](HELDOUT_DEV_COHORT_2026-10-07.md) | Published v1 trained on ~96–99.6% of the train rows the 10-07 dev drew from; a new 668-question, 342-case cohort uses only validation/test splits neither model trained on. | current |
| [CPU cause checks, 10-07](CAUSE_CHECKS_CPU_2026-10-07.md) | Test 1's Noul regression is a lost "false" bias on calculation items with no ranking skill (AUC 0.50–0.55); direct reads are under-confident on gold-true answers; tokenizer reserialization causes the 2.1 s v2 latency. | current |
| [Trained checkpoint comparison, 10-07](TRAINED_CHECKPOINT_COMPARISON_2026-10-07.md) | v2 on 77.87 > v1 on 63.09 > v2 off 57.93 CC on 376 dev questions. v2 not promoted. | current |
| [Ayaka v2 controls](AYAKA_V2.md) | Reasoning modes, efforts and API fields for v2 checkpoints. | reference |

## Trained v2 checkpoint and calibration

| Report | What it says | Status |
|---|---|---|
| [Noul calibration trial, 10-07](NOUL_CALIBRATION_2026-10-07.md) | Affine Noul correction improved NLL/ECE but raised abstentions 18 → 19; automatically rejected. | rejected |
| [Quality hierarchy screen, 10-07](QUALITY_HIERARCHY_DEV_2026-10-07.md) | Predeclared three-system dev screen (v1 on < v2 off < v2 on). | reference |
| [JevBench version transfer, 10-07](JEVBENCH_VERSION_TRANSFER_2026-10-07.md) | v1.5.7 → v1.6.1 Intelligence transfer from 38 paired systems; Ayaka still needs an old-scale anchor. | current |
| [RPS optimization, 10-07](RPS_OPTIMIZATION_2026-10-07.md) | Device-local ragged ordinal RPS loss with matched gradients. | adopted |

## Swift readout on frozen Gemma 4 12B

| Report | What it says | Status |
|---|---|---|
| [Swift plan, 10-04](AYAKA_V3_SWIFT.md) | Frozen 12B letter readout plus per-type temperatures and gated levers. | reference |
| [JevBench v1.5 board, 10-04](JEVBENCH_V15_BOARD_2026-10-04.md) | Scoring rules and board observations behind the Swift design. | superseded by [version transfer](JEVBENCH_VERSION_TRANSFER_2026-10-07.md) for v1.6.1 |
| [Swift GPU results, 10-05](SWIFT_GPU_RESULTS_2026-10-05.md) | `cygnet` prompt raised Intelligence but failed the composite gate; gated reasoning route adopted; v1 adapter neutral under Swift readout. | current; route adopted |
| [Swift hard dev protocol, 10-05](SWIFT_HARD_DEV_PROTOCOL_2026-10-05.md) | Frozen protocol for rerunning the prompt gate on new hard/judge data. | reference |
| [Swift Noul decomposition, 10-05](SWIFT_NOUL_DECOMPOSITION_2026-10-05.md) | `cygnet`'s gain is Noul on in-house authored items; a threshold lever does not transfer. | current |
| [v1 on vs v2 off protocol, 10-05](V1ON_V2OFF_PROTOCOL_2026-10-05.md) | Fixed comparison rule for frozen v2 off against v1 on; no measurement recorded. | reference |

## Saved-read diagnostics

| Report | What it says | Status |
|---|---|---|
| [Paired teacher diagnostic, 10-05](V2_PAIRED_TEACHER_DIAGNOSTIC_2026-10-05.md) | Raw worked steps help synthetic/verified items and hurt natural HelpSteer2. | current |
| [Paired calibration diagnostic, 10-05](V2_PAIRED_CALIBRATION_DIAGNOSTIC_2026-10-05.md) | Temperature-only correction reduces overconfidence; the HelpSteer2 regression remains. | current |
| [Continuation audit, 10-04](V2_CONTINUATION_AUDIT_2026-10-04.md) | The Beam pilot's regression is mostly extra Noul abstentions, not wrong argmax directions. | current |
| [Saved policy probe, 10-04](V2_SAVED_POLICY_PROBE_2026-10-04.md) | Noul commitment raises thresholded competence but worsens probability quality. | rejected |
| [Noul commitment bands, 10-04](V2_NOUL_COMMIT_BANDS_2026-10-04.md) | Abstaining groups do not support 0.801 confidence. | rejected |
| [v1 mechanism results, 10-04](../v2-mechanism-results-20261004.md) | v1 gains from generated worked steps; generated reasoning worsens Noul probabilities. | current |
| [v1 mechanism protocol](../v2-mechanism-protocol.md) | Protocol for the mechanism diagnosis above. | reference |

## Training runs and pilots

| Report | What it says | Status |
|---|---|---|
| [Clean Beam pilot, 10-03](V2_CLEAN_BEAM_2026-10-03.md) | 200-step continuation from ayaka-base regressed (CC 9.93 → 2.81). | rejected |
| [v2 findings, 10-01](V2_FINDINGS_2026-10-01.md) | First bounded H100 exploration; measured gains and multilingual regressions. | superseded by [checkpoint comparison](TRAINED_CHECKPOINT_COMPARISON_2026-10-07.md) |
| [Direct over v1 reasoning plan, 10-04](V2_DIRECT_OVER_V1_REASONING_2026-10-04.md) | Training hypotheses for v2 direct beating v1 with reasoning. | reference; see [gap analysis](V2_GAP_ANALYSIS_2026-10-07.md) |
| [Literature review, 10-05](V2_LITERATURE_REVIEW_2026-10-05.md) | Prioritized research directions before another paid run. | reference |
| [HelpSteer target fix, 10-05](V2_HELPSTEER_TARGET_FIX_2026-10-05.md) | Fractional HelpSteer2 ratings now split across adjacent levels. | adopted |

## Data and training preparation

| Report | What it says | Status |
|---|---|---|
| [Direct corpus preparation](DIRECT_CORPUS_PREPARATION.md) | Offline, byte-pinned direct training corpus builder. | reference |
| [Direct upload preparation](DIRECT_UPLOAD_PREPARATION.md) | Moving verified direct inputs to a GPU host; native weight checks. | reference |
| [Checked pretraining conditions, 10-06](V2_CHECKED_PRETRAINING_CONDITIONS_2026-10-06.md) | Review conditions wired into the real execution path. | reference |
| [Component coverage and robust teachers, 10-06](V2_COMPONENT_COVERAGE_AND_ROBUST_TEACHERS_2026-10-06.md) | Paragraph component coverage audit and teacher tail reporting. | reference |
| [Paragraph protocol preparation, 10-05](V2_PARAGRAPH_PROTOCOL_PREPARATION_2026-10-05.md) | Paragraph isolation and teacher signal preparation. | reference |
| [Offline augmentation and prompt teachers, 10-05](V2_OFFLINE_AUGMENTATION_AND_PROMPT_TEACHERS_2026-10-05.md) | Counterfactual data checks and optional prompt teachers. | reference |
| [Clef evidence ideas, 10-04](V2_CLEF_EVIDENCE_2026-10-04.md) | Experimental text-feature components inspired by published systems. | reference |
| [Evidence feature adapter, 10-04](V2_EVIDENCE_FEATURE_ADAPTER_2026-10-04.md) | Real backbone feature extraction for the evidence experiment. | reference |
| [Pretraining readiness, 10-01](V2_PRETRAINING_2026-10-01.md) | Prepared image/language/proposal data and zero-step entry point. | superseded by [training ready](V2_TRAINING_READY_2026-10-02.md) |
| [Training optimization, 10-02](V2_TRAINING_OPTIMIZATION_2026-10-02.md) | Equal-work CPU timing, batched losses and preparation caches. | adopted |
| [Training ready, 10-02](V2_TRAINING_READY_2026-10-02.md) | Natural EN/KO/JA rehearsal and H100 profiling without optimizer updates. | reference |
| [RunPod ready, 10-02](V2_RUNPOD_READY_2026-10-02.md) | Offline RunPod archive and pinned runtime. | reference |
| [GPU comparison, 10-02](V2_GPU_COMPARISON_2026-10-02.md) | RTX PRO 6000 is the preferred measured GPU for the native recipe. | current |

## Serving and read contracts

| Report | What it says | Status |
|---|---|---|
| [Jev API compatibility, 10-04](JEV_API_COMPAT_2026-10-04.md) | Both servers follow the TypeSafe Jev API and SDK. | adopted |
| [Native read contract](V2_NATIVE_READ_CONTRACT.md) | Opt-in CPU contract for native text reads. | reference |
| [vLLM read audit, 10-04](V2_VLLM_READ_AUDIT_2026-10-04.md) | Pinned vLLM source and reader contract audit. | reference |
| [Images and generated candidates](../MULTIMODAL_AND_CANDIDATES.md) | Native image inputs and opt-in Choice generation. | reference |

Machine-readable results live in [`results/`](results/).
