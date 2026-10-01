# Ayaka v2 readiness before training — 2026-10-01

The native-image, reasoning and generated-Choice training paths are implemented
and an immutable local bundle is prepared. **No pretrained optimizer update or
new GPU job was executed.** The machine has PyTorch `2.14.0+cpu`; a real H100
backward check remains necessary before starting training. These preparation
results do not establish image accuracy, multilingual preservation, generated
partition calibration, or a JevBench rank.

The measured readiness record is
[v2-pretraining-20261001.json](results/v2-pretraining-20261001.json).
The previous H100 findings and language regressions remain in
[V2 findings](V2_FINDINGS_2026-10-01.md); this preparation does not supersede them.

## Prepared model and data

The bundle is `runs/v2-pretraining-20261001-e4b-validated`. Its manifest SHA-256 is
`49900660f761c6509d9123ef3ba14f7e6c6f3666f60c28fc699029bd057b4330`.
It records implementation revision
`69cce1160283d20fceaafd8c0e3e8c04a5478d02`, package-source hashes, dependency
versions, configuration, license checks and all five JSONL hashes. The earlier
small prototype bundle was retained separately.

The pinned base is
[Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2),
revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, under Apache 2.0.
Its native monolithic `model.safetensors` is cached locally: 15,992,595,884 bytes,
SHA-256 `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`.
The cache audit is bound to this bundle. Both indexed shards and monolithic
checkpoints are supported; cache preparation does not allocate GPU memory.

Actual architecture construction on the meta device counts 7,979,000,870 unique
parameters including the adapter and decision head, with 37,900,038 trainable.
The conservative official tensor-element count plus adapter/head is
8,034,056,528. Both are below 14B. Native context is 131,072; the training
configuration uses 4,096. The real pinned tokenizer and image processor checked
every row without loading pretrained tensors into a model.

| Per split | Count |
|---|---:|
| Samples | 1,216 |
| Questions | 1,312 |
| Choice / Noul / Score questions | 544 / 448 / 320 |
| English / Korean / Japanese samples | 512 / 352 / 352 |
| Text / native-image samples | 832 / 384 |
| Prepared direct, trace and proposal rows | 2,464 |

Across `train`, `router_train`, `dev`, `calibration`, `test`, there are 6,080
samples, 6,560 questions and 12,320 prepared rows. Maximum row length is 483
tokens in the first four splits and 504 in test; no state or expanded media
prefix was truncated. Independent source-cluster counts are respectively
270 / 269 / 271 / 271 / 271, much smaller than the number of derived rows.

The repository-authored MIT curriculum includes verified calendar/numeric/rule
traces, native-language direct decisions, paired images and explicit transcripts
for receipts/charts/calendars/rule exceptions, unreadable evidence, finite Choice
proposal targets and separate overlap/coverage/sufficiency/parent diagnostics.
EN/KO/JA translations and image/transcript pairs share their underlying source
lineage; repeated unreadable images and primitive variants are clustered too.
Cross-split source, template, rule-combination, document-voice and evidence-content
checks reject leakage, including copies with changed gold targets.

These are mechanics examples with authored targets, not a large natural-language
training corpus. The existing Open-Jev/HelpSteer2/KLUE/JGLUE regression probes
remain evaluation-only. Passing the new native-language rehearsal does not prove
that those previous regressions are solved. Main training and release still need
license-clean natural training data and new independent natural measurements.

## Training behavior

Fresh LoRA and a fresh decision head are initialized from the pinned base. The
experimental SFT adapter that regressed Korean/Japanese is not reused. Only the
text adapter and decision head are trainable; vision/projector/audio components
stay frozen because the checkpoint saves the shared text adapter and typed head.
Native media passes through the full differentiable model, so image decision and
trace losses reach the text adapter. Image rows are isolated; no training KV cache
is retained or reused across questions.

Training combines existing typed decision losses with concise trace CE (weight
0.3) and isolated proposal CE (weight 0.2). Direct rows are mixed with traces.
Proposal JSON uses the serving prompt, but its target and disposable rationale
never enter the final teacher-partition readout. Verified finite-domain labels
include explicit uncertainty priors and residual Other; normalization is not
treated as proof that arbitrary real-world outcomes are exhaustive.

The lazy stream processes media one sample at a time, samples English/Korean/
Japanese with weights 0.6/0.2/0.2, and uses only `train` for optimizer gradients.
Before training, representative modality/family/language/primitive rows must
produce finite losses and finite nonzero gradients while all trainable weights
and optimizer state remain unchanged. The execution wrapper caps loading,
backward preflight, training and saving together. Progress checkpoints include
optimizer/scheduler/RNG state and a completion marker written last; interrupted
unmarked checkpoints must not be used.

To repeat preparation into a **new directory** without loading model weights:

```bash
python -m ayaka.training.prepare_v2 \
  --checkpoint runs/v2-exploration-20261001/sft/gemma4-e4b \
  --out runs/my-new-v2-bundle
python -m ayaka.training.cache_v2 \
  --bundle runs/my-new-v2-bundle --out runs/my-new-weight-audit.json
```

Default execution only audits the prepared bundle:

```bash
python -m ayaka.training.run_v2 \
  --bundle runs/v2-pretraining-20261001-e4b-validated
```

It returned `bundle_verified_no_training`, `source_code_matches: true`, and
`optimizer_steps: 0`. The local result is `readiness.json` inside the bundle.

On a CUDA host, with the same source files and pinned cached weights, the next
check can run backward only (this command has **not** been executed on H100):

```bash
python -m ayaka.training.run_v2 \
  --bundle runs/v2-pretraining-20261001-e4b-validated \
  --backward-only --device cuda --max-train-seconds 900 \
  --out runs/v2-cuda-backward-check
```

This mode rejects `--steps` and returns before training. CUDA execution requires
a compatible CUDA build of PyTorch/torchvision; the current CPU environment
cannot satisfy this gate. Transfer the immutable bundle and pinned Hugging Face
cache to the GPU host, or use explicit weight downloads there. Package versions
are recorded in the manifest; source drift requires preparing a new bundle.

Actual optimization additionally requires `--execute`, a positive explicit
`--steps`, a positive `--max-train-seconds` at most 28,800, and a new `--out`.
Neither the step count nor the main-training budget was chosen in this request.
The process timer bounds one invocation, not repeated jobs or cloud billing.
Keep a cumulative external ledger: the earlier exploration already has a
13,627-second conservative attached-GPU bound within its original 28,800-second
ceiling. No additional GPU time was used by this preparation.

## Evaluation, calibration and routing

`ayaka.eval.pretraining_v2` evaluates each modality/language/task and generated
teacher partition separately, records actual routes/budgets/tokens/latency, and
reports Noul abstention, Score nMAE/RPS, NLL, reasoning corrections and regressions.
Image/transcript differences are paired diagnostics, not causal attribution of
perception versus reasoning. Optional live proposals are exactly audited only
when they match the declared finite status grammar; free-form paraphrases remain
explicitly unverified.

For a future trained checkpoint, evaluate `router_train`, `dev`, and reserved
`calibration` before touching test. Each invocation needs a new report file and
an explicit full-process time budget, including model loading and in-flight
generation. For example:

```bash
python -m ayaka.eval.pretraining_v2 \
  --bundle runs/v2-pretraining-20261001-e4b-validated \
  --checkpoint runs/my-trained-checkpoint --split dev \
  --modes off low medium high --proposals --device cuda \
  --max-evaluation-seconds 1800 --out runs/my-dev-report.json
```

A hard deadline exits with code 124; an incomplete/missing report cannot promote
an artifact. Smaller evaluation budgets must be accounted for cumulatively too.

Model fingerprints cover config, saved head and adapter bytes. Image calibration
requires reserved calibration rows from that exact model, modality and fixed
partition, with at least 20 independent source clusters for each actual
type/route/budget path. Unfitted paths receive no broad temperature fallback:

```bash
python -m ayaka.training.scoped_calibration \
  --report runs/my-calibration-report.json --checkpoint runs/my-trained-checkpoint \
  --modality image --partition fixed --out runs/my-image-calibration.json
python -m ayaka.training.scoped_router \
  --train-report runs/my-router-train-report.json --dev-report runs/my-dev-report.json \
  --checkpoint runs/my-trained-checkpoint --modality image \
  --out runs/my-image-router.json
```

The router requires paired off/low/medium/high measurements and positive
clustered dev confidence bounds, using only the agreed lambda grid. No verified
gain means no promoted artifact. Attach valid artifacts using `--images
--image-calibration ... --image-router ...`; mismatched model/domain fails before
full weight loading. Without an image router, `auto` stays direct and reports
`no_validated_image_router`. **`on + high` bypasses both routing and speculative
direct classification and preserves its 1,024-token budget**, including simple
and confident tasks; native EOS may terminate normally.

No image calibration or router was fitted/promoted in this preparation. The
generated candidate serving API continues to label arbitrary partitions
uncalibrated: the finite audit and teacher targets are not a validated free-form
coverage estimator. Audio/video/PDF ingestion and generated Score/Noul schemas
remain outside this interface.

## Verification and transport

Ruff lint and format checks passed for 123 Python files. The complete CPU suite
passed **395 tests**, with one existing Beam-SDK skip, in 41.40 seconds. New checks
cover native full/cache logits, both image architectures, differentiable LoRA
gradients, frozen vision, unchanged zero-step weights, proposal isolation,
source-aware preparation, scoped calibration/routing, forced high execution,
candidate conservation and retry usage.

Windows IPv4 loopback occasionally reset even model-free health requests.
Keyed POST retries now replay the same cached response and usage instead of
spending inference twice. The process-local cache retains up to 8,192 keys for
one hour with a 64 MiB body budget and 2 MiB per-response ceiling. Admission
fails before inference if full; live keys are not evicted. Oversized results
retain actual usage in their cached error. This bounds memory and tolerates
response loss; it does not establish an OS fix or cross-process/restart
idempotency. IPv6 is supported explicitly (`--host ::1`).
