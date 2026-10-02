# Offline Ayaka v2 kit for Runpod

The prepared archive includes Linux CPython 3.11.14, the hash-locked Torch
2.8.0/CUDA 12.8 environment, pinned native E4B weights and processor/tokenizer,
exact Ayaka source, and all five audited data splits. Startup never runs pip,
downloads model/dataset files, or rebuilds supervision. A network connection is
not needed for the audit or native training data/model reads.

The preferred transport is `ayaka-v2-runpod-ready.tar.zst`, compressed with
Zstandard **level 3, four threads**: 15.62GB instead of 23.31GB (32.98% smaller).
The entire decompressed tar's SHA-256 matches the previously verified original.
A small static Linux x86_64 extractor is provided separately, so the Pod needs
no zstd installation. The native training kit and its manifest are unchanged.

## Deploy

Use one **RTX PRO 6000 Blackwell 96GB**, Linux x86_64 (Ubuntu 22.04/24.04 is a
suitable host environment), an NVIDIA driver compatible with CUDA 12.8 and
at least **100GB `/workspace` volume capacity** while the archive and extracted
kit coexist. The runtime uses four CPU threads. The complete checkpoint schedule
needs at least 20GiB free after extraction; the launcher checks this before
model loading. Use a persistent volume for both the kit and outputs.

Prestage the compressed archive and checksum on the attached volume before GPU startup
where possible. The package is a local artifact; creating it does not upload it
to Runpod. Also upload `runpod-zstd-linux-x86_64.tar.gz` and its `.sha256`
sidecar when the host does not already have zstd. With all four files on
`/workspace`, use the included extractor:

```bash
set -euo pipefail
cd /workspace
sha256sum -c ayaka-v2-runpod-ready.tar.zst.sha256
sha256sum -c runpod-zstd-linux-x86_64.tar.gz.sha256
tar -xzf runpod-zstd-linux-x86_64.tar.gz
./runpod-zstd-linux-x86_64/zstd -dc ayaka-v2-runpod-ready.tar.zst | tar -xf -
bash /workspace/ayaka-v2/scripts/runpod_v2/start.sh
```

If `zstd` is already installed, `tar --zstd -xf ayaka-v2-runpod-ready.tar.zst`
replaces the extractor steps. Streaming extraction does not create an extra
23GB intermediate tar. Shell `pipefail` prevents training after a failed decode.

If the kit has already been extracted on the attached volume, use this as the
Pod template start command:

```bash
bash /workspace/ayaka-v2/scripts/runpod_v2/start.sh
```

The default action starts the entire **1,200-step × 64-row** schedule after
integrity checks and the production finite-gradient, stress/memory, throughput
and complete-budget checks. The four-hour emergency process cap does not
define the intended training endpoint. If the complete plan fails its budget
check, no optimizer update starts. No rows or steps are silently removed.

Results are stored under `/workspace/ayaka-v2/outputs/v2-main-1200/`, including
the final `checkpoint/complete.json`, training history, throughput and budget
reports. A complete checkpoint must contain `complete: true` and `steps: 1200`.
Step checkpoints include optimizer/LR/RNG state. The launcher refuses to
restart a used run directory; automatic restart does not silently create a
second fresh training run. This launcher does not implement checkpoint resume.

Console output is also saved under `outputs/logs/`. Stop the Pod after the
requested work finishes: an exited Python process does not stop Pod billing.
The start script does not delete outputs or stop the Pod automatically.

## Audit and subsequent evaluation

An audit does not load the pretrained weight tensors or execute any optimizer:

```bash
bash /workspace/ayaka-v2/scripts/runpod_v2/start.sh audit
```

Evaluation uses the same offline model cache and requires a complete checkpoint,
an explicit split and an explicit time allowance. It does not happen implicitly
after training. Set the allowance before starting an evaluation; no unmeasured
evaluation cost is included in the previous $4.05/$6.68 training projections.

```bash
# Replace SECONDS with the evaluation budget chosen for this run.
bash /workspace/ayaka-v2/scripts/runpod_v2/start.sh evaluate \
  --split dev --modes off low medium high --max-evaluation-seconds SECONDS
```

Use `router_train`, `dev` and `calibration` for their declared purposes. Finish
model/router/calibration selection before the single independent `test` run.
The existing scoped router/calibration tools remain included in `ayaka.training`;
promotion requires complete reports bound to the trained checkpoint. Neither
quality nor the eventual complete evaluation runtime has yet been measured.

## Rebuild on CPU

Prepare the relocatable runtime with installed Astral uv on a Linux CPU machine:

```bash
bash scripts/runpod_v2/prepare_runtime.sh /absolute/new/runtime
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" /absolute/new/runtime/bin/python3.11 \
  scripts/runpod_v2/package.py \
  --bundle runs/v2-pretraining-20261002-ready \
  --snapshot /absolute/pinned/huggingface/snapshot \
  --runtime /absolute/new/runtime --out /absolute/new/kit \
  --archive /absolute/ayaka-v2-runpod-ready.tar
# On the CPU preparation host with zstd available:
zstd -3 -T4 --check /absolute/ayaka-v2-runpod-ready.tar \
  -o /absolute/ayaka-v2-runpod-ready.tar.zst
cd /absolute
sha256sum ayaka-v2-runpod-ready.tar.zst > ayaka-v2-runpod-ready.tar.zst.sha256
```

Runtime preparation needs network access on the CPU machine; deployed training
does not. `requirements.lock.txt` includes every resolved dependency's hashes,
including NVIDIA CUDA libraries and Triton, and allows only wheels. The archive
stores regular model files and internal relative runtime symlinks; Hugging Face
credentials, unrelated runs and user artifacts are excluded.

Runpod references: [persistent storage](https://docs.runpod.io/pods/storage/types),
[file transfer](https://docs.runpod.io/pods/storage/transfer-files),
[GPU Pod rates](https://www.runpod.io/pricing).

## Corrected recovery preparation

The failed 2026-10-02 pilot is experimental. Its original invoice menus and
Noul labels admitted shortcuts; its control did not establish improvement.
Use a newly prepared `--clean` bundle and the original pinned v1 checkpoint.
Old evaluations, shape-only timing profiles, datasets and launch defaults must
not be reused as clean evidence.

```bash
python -m ayaka.training.prepare_recovery --clean \
  --source-bundle runs/v2-pretraining-20261002-training-ready-final \
  --checkpoint /absolute/pinned/v1-checkpoint \
  --training-window-seconds 3600 --out /absolute/new/clean-bundle
python -m scripts.runpod_v2.clean_package \
  --base-tar runs/ayaka-v2-runpod-ready.tar \
  --bundle /absolute/new/clean-bundle --parent /absolute/pinned/v1-checkpoint \
  --out /absolute/new/ayaka-v2-clean-ready.tar.zst
```

Packaging requires the `runpod` optional dependency. It reads verified local
Linux runtime and pinned model files from the old archive, excludes its old
source/data and private cache files, adds current code/clean data/v1 weights,
and uses zstd level 3. It verifies the resulting frame and every member hash.
No model download or dependency install is needed at GPU startup. Upload or
prestage this archive and its SHA-256 sidecar before starting the paid GPU.

Extract into a fresh directory with `pipefail` and `--no-same-owner`. The existing
zstd extractor archive can be used if the machine has no zstd binary.

```bash
set -euo pipefail
sha256sum -c ayaka-v2-clean-ready.tar.zst.sha256
mkdir /workspace/clean-kit
zstd -dc ayaka-v2-clean-ready.tar.zst | tar --no-same-owner -xf - -C /workspace/clean-kit
bash /workspace/clean-kit/ayaka-v2/scripts/runpod_v2/start_clean.sh plan \
  --out /workspace/clean-kit/ayaka-v2/outputs/clean-plan.json
# Execute only with an admitted whole-Pod budget, including QA and delivery.
bash /workspace/clean-kit/ayaka-v2/scripts/runpod_v2/start_clean.sh execute \
  --out /workspace/clean-kit/ayaka-v2/outputs/clean-pilot \
  --training-window-seconds 3600 --evaluation-window-seconds 1800
```

These allowances are limits, not measured runtime forecasts. New data invalidates
old profiling. The fresh production preflight must admit the entire 200-step
schedule before any optimizer update. Parent evaluation, loading, packaging,
receipt verification and Pod deletion also need an external billing envelope.
The runner does not allocate, stop or delete Pods. This full archive is larger
than the previous thin package; prestaging trades local bandwidth/storage for
less paid GPU setup.

Clean training uses natural rehearsal and new corrected authored cases. Separate
authored dev/test generators cover rule combinations absent from this training
curriculum. Dev/test still share some formulas; final test is authored-only and
does not prove natural, image or official sealed-inclusive JevBench quality.
Raw improvement, calibrated improvement and published-parent comparison must
all pass before opening test. Full training and release remain separate decisions.
