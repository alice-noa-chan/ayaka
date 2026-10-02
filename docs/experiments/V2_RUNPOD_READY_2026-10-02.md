# Runpod native v2 preparation — 2026-10-02

The deployable offline kit is ready for one **RTX PRO 6000 Blackwell 96GB**.
Preparation ran locally on CPU. No Runpod Pod was created, no paid GPU was used
for this preparation, and no pretrained optimizer update was performed.

## Actual artifact

The preferred local archive is `runs/ayaka-v2-runpod-ready.tar.zst`; its sidecar is
`runs/ayaka-v2-runpod-ready.tar.zst.sha256`. The original uncompressed tar remains
available. The machine-readable
[preparation result](results/v2-runpod-ready-20261002.json) records the final
archive size/SHA-256 and kit-manifest SHA-256. These are hashes of the actual
assembled files, rather than a planned build.

The archive contains all five audited data splits, exact training source,
the pinned native model/processor/tokenizer, a standalone Linux CPython runtime,
the complete installed CUDA/Python dependency stack, license notices and the
startup scripts. It packages only these explicit inputs. Unrelated user
artifacts and authentication caches are not part of the kit. Large binaries and
temporary preparation logs stay outside Git.

| Prepared item | Actual value |
| --- | --- |
| Preferred Zstandard archive size | 15,624,451,321 bytes (15.62GB / 14.55GiB), 32.98% smaller |
| Compression | Level 3, four threads, frame checksum enabled, libzstd 1.5.7 |
| Compressed archive SHA-256 | `48e862818043a1ff463554b8dc169b0dbb5afddb11b3d0937b74b2623ca616af` |
| Portable extractor archive | 554,970 bytes, static Linux x86_64, zstd 1.5.7 / musl 1.2.5 |
| Extractor archive SHA-256 | `6e2d1bbc6ca77507b2bb76a2040bcc06c2fc50a60d6e427dab4a687ac2fad4de` |
| Archive size | 23,314,565,120 bytes (23.31GB / 21.71GiB) |
| Archive SHA-256 | `3db50bf77cae01eb7e5575aee41a1df7cd1a8602d2dc6948484cfa5eee89d701` |
| Kit file records | 28,877, independently verified inside the archive |
| Kit-manifest SHA-256 | `c8363ba710fb790d1b0452597fe49fceaa8393376131506e4cc0dd0aeca0830b` |
| Base model | `google/gemma-4-E4B-it` |
| Model revision | `ee0ef6023621cff504d758262d4e04895a5af4a2` |
| Native weight size | 15,992,595,884 bytes |
| Native weight SHA-256 | `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503` |
| Dataset manifest SHA-256 | `7b11be347d7a319663102b26aa6ec91de5947f23ae9efc6d6ed3aaabad9cad7e` |
| Training source revision | `41194b4ddd5f96fa32a7eabf031d5232c2eeef79` |
| Python / Torch / CUDA runtime | 3.11.14 / 2.8.0+cu128 / 12.8 |
| Torchvision / Transformers / PEFT | 0.23.0+cu128 / 5.17.0 / 0.21.0 |
| Resolved, hash-locked distributions | 79 |
| CPU threads | 4 |
| Planned optimizer steps / rows per step | 1,200 / 64 |
| Scheduled row exposures | 76,800 |
| Schedule SHA-256 | `0854216412c77fad34cdacf36c46eff46c9a60d308a388edd8284e2ebc220105` |

The datasets were already constructed, native-tokenized and audited in the
[training readiness report](V2_TRAINING_READY_2026-10-02.md). This kit copies
that immutable bundle without regenerating it. Train has 2,752 state samples,
3,872 questions and 5,024 training rows. Each of router_train, dev, calibration
and test has 1,472 state samples, 1,824 questions and 2,976 rows. Scheduled
exposures repeat rows; they are not 76,800 independent examples. Language,
native image, concise trace, direct decision and proposal supervision remain
in the prepared schedule with their recorded source provenance.

## Verification

The copied runtime executed `launch.py audit` in a Linux network namespace with
external networking disabled and CUDA devices hidden. It verified every file
record, original pinned weight hashes, all five dataset splits, source binding,
native processor/tokenizer loading, dependency versions and the entire finite
1,200-step schedule. It returned `offline_kit_verified_no_training` and
`optimizer_steps: 0`. Model weight tensors were not loaded by this audit.

An independent archive reader also verifies the complete tar SHA-256 and every
contained file's hash/size or relative symlink against the archived manifest.
This catches packaging corruption in addition to checking the source kit.

The additional Zstandard transport compression took 36.63 seconds locally;
complete decompression/hash verification and compressed-file hashing took
55.94 seconds. All 23,314,565,120 restored bytes hash to the original tar's
SHA-256. The sealed kit is unchanged, including all 28,877 verified records.
The compressed frame includes its original content size and checksum.
This CPU operation does not load model weights or invoke the training pipeline.

Windows full suite: **447 passed, 1 existing optional Beam skip**. Linux full
suite in the pinned CUDA runtime, with external networking disabled and GPU
hidden: **446 passed, the same skip**; the subsequently added pinned-weight
copy-binding test brings the focused Linux kit suite to **7 passed**. The
Windows full run includes that seventh test. Ruff lint/format and both Bash
syntax checks pass. All 79 installed distribution versions match the lock;
no installed wheel declares a manylinux tag newer than 2.28.

Two existing CPU parity tests needed runtime compatibility adjustments. The
Qwen hybrid-cache test explicitly selects Transformers' pure Torch reference
functions because installed FLA otherwise dispatches a CPU tensor to Triton.
The Gemma image batch-gradient test permits a 2e-5 per-element absolute
rounding difference while retaining its strict gradient-norm, loss and logits
checks. Production kernel dispatch and model code were not changed.

## Launch

Use a Linux x86_64 Pod with a CUDA 12.8-compatible NVIDIA driver and **100GB
persistent `/workspace` capacity**, leaving at least 20GiB free for checkpoint
outputs after extracting the kit. Prestage the archive and sidecar on the
volume before starting the paid GPU where possible. Local preparation does
not upload these files or configure a Runpod volume/template. A static
Linux x86_64 zstd extractor is supplied in `runs/runpod-zstd-linux-x86_64.tar.gz`
with its own checksum and license notices. It needs no shared libraries or
Python installation; prepare all four transport files before GPU startup.

```bash
set -euo pipefail
cd /workspace
sha256sum -c ayaka-v2-runpod-ready.tar.zst.sha256
sha256sum -c runpod-zstd-linux-x86_64.tar.gz.sha256
tar -xzf runpod-zstd-linux-x86_64.tar.gz
./runpod-zstd-linux-x86_64/zstd -dc ayaka-v2-runpod-ready.tar.zst | tar -xf -
bash /workspace/ayaka-v2/scripts/runpod_v2/start.sh
```

An existing host zstd can instead use `tar --zstd -xf`. Streaming extraction
avoids storing a second, uncompressed intermediate tar, and `pipefail` stops
the command sequence on a decompression failure. The decoder is built from
the [official Zstandard 1.5.7 source](https://github.com/facebook/zstd/releases/tag/v1.5.7),
whose release SHA-256 is checked before compiling with musl 1.2.5. The delivered
binary's static linkage, version, round-trip decode, tar permissions and failed
decode exit status are verified locally; the machine report records its digest.

Once extracted, the last line is the Pod start command. No pip installation,
dataset generation or model download occurs. Integrity checks, native backward
preflight, memory/stress and complete-budget profiling still run on startup
before the first optimizer update. The launcher then executes all 1,200
planned steps. The four-hour emergency cap is not a reduced training target.

Completion is recorded in
`/workspace/ayaka-v2/outputs/v2-main-1200/checkpoint/complete.json` with
`complete: true` and `steps: 1200`. Console logs, step checkpoints and optimizer,
LR/RNG state are retained. A used run directory is rejected to prevent an
automatic Pod restart from silently initiating a second fresh run. This
launcher does not add checkpoint resume or automatic Pod termination.

## Costs and remaining validation

The earlier matched native GPU comparison suggested about **1h56m typical or
3h12m conservative** for complete RTX PRO training, and **$4.05/$6.68** at
$2.09/hour. Those are forecasts based on the prior benchmark, not timings
measured on the user's Runpod host. The new archive upload/extraction and full
integrity audit, storage and evaluation are additional; preloading the volume
keeps transfer/install work out of the paid GPU window. The actual Runpod
credit balance and driver/hardware availability have not been verified.

The live Pod's GPU preflight, main pretrained training and quality evaluation
remain. Evaluation is a separate command with an explicit split and time
budget; it requires the complete checkpoint. Router/calibration promotion must
use complete checkpoint-bound reports and the independent test remains outside
model selection. Training completion does not imply measured model quality or
JevBench ranking. A stopped training process does not stop Pod billing.

See the [deployment guide](../../scripts/runpod_v2/README.md) for audit,
evaluation and CPU rebuild commands. Runtime rebuilding is version/hash locked;
the shipped kit itself needs no network to read its training inputs.
