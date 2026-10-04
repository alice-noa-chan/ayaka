# Swift GPU read collection

This runner targets Ubuntu, CUDA, Python 3.11, and one 48 GB GPU. It installs
`vllm==0.30.0`, then `pip install --no-deps -e .`: Swift's serving and collection
paths use the standard library, and vLLM supplies torch/transformers. Ayaka's
training extras and PEFT are unnecessary here. Run inside a fresh virtualenv.
Nothing in a dry run installs packages, contacts the Hub, or loads a GPU model.

## Prepare inputs locally

From the repository root, with no downloads:

```powershell
python scripts/swift/pack_inputs.py
python scripts/swift/gpu_runner.py --dry-run
```

The packer writes `scripts/swift/inputs.tar.gz` and its `.sha256`. Its explicit
[manifest](manifest.json) includes only v2 calibration/dev, the vendored Cygnet
calibration data and notices, and all 231 JevBench public items and their LICENSE.
The v2 `test.jsonl` stays unopened; the packer refuses any `test.jsonl` input.
It stores normalized tar metadata, gzip mtime=0, sorted entries, input hashes,
and an exact decision/read inventory. Identical inputs produce identical bytes.

If the repository's JevBench copy is incomplete, pass
`--jevbench-dir PATH_TO_AUDIT_CLONE_PUBLIC_DIRECTORY`; that directory must contain
`easy.jsonl`, `hard.jsonl`, `original.jsonl`, and `LICENSE`, totaling exactly 231
items. The packer places them at the canonical repository paths in the archive.
It never scans other audit-clone datasets. Cygnet's 241 generated items are copied
unchanged from commit `3cf591c692dec649f7c134449814610307c7bb3a`; see the vendored
[attribution](../../ayaka/swift/data/cygnet_ATTRIBUTION.md) and
[MIT license](../../ayaka/swift/data/cygnet_LICENSE).

Transfer the repository code and this input archive to the GPU host yourself.
The input archive contains data and a generated manifest, not repository source,
model weights, or the test set. Extract it **at the repository root**:

```bash
sha256sum -c scripts/swift/inputs.tar.gz.sha256
tar -xzf scripts/swift/inputs.tar.gz
python3.11 -m venv .venv
source .venv/bin/activate
```

## Run on the GPU host

Obtain access to any gated models beforehand and set `HF_TOKEN` on the GPU host
if needed. Secrets are not written to `env.txt`. Supply the E4B revision
explicitly; there is no silent `main` fallback:

```bash
export SWIFT_GEMMA_E4B_REVISION=FULL_COMMIT_SHA
export SWIFT_MAX_MINUTES=75
bash scripts/swift/collect_gpu.sh --dry-run
bash scripts/swift/collect_gpu.sh --prompt-variants min,cygnet,rules
```

The default models are:

| Model | Requested revision |
| --- | --- |
| `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| `google/gemma-4-E4B-it` | `SWIFT_GEMMA_E4B_REVISION` |

Qwen3.5-4B is already measured on the board as frozen SemIf and is excluded
from the defaults. It can still be requested explicitly with `--model`.

Repeat `--model MODEL@REVISION` to replace this list. SHA, tag, or branch inputs
are resolved once through the Hub on the GPU host. Both weights and tokenizer
are then served at the resolved immutable commit; each read row records `model`
and `revision`, plus `prompt_variant`. Dry runs leave unspecified pins visibly unresolved and require
no Hub access.

Each model runs alone with bf16, context length 16,384, GPU memory utilization
0.90, and prefix caching enabled. The server must use
`--logprobs-mode processed_logprobs --max-logprobs 26`; the runner supplies
these flags and refuses overrides. The supervisor waits for `/health`, runs
HF/vLLM parity for every selected prompt variant, then collects
every manifest dataset once per prompt variant with
`python -m ayaka.swift.collect --concurrency 16 --prompt-variant VARIANT`.
It reuses that vLLM server and its prefix cache across all bulk collections,
then starts a Swift HTTP server for each variant's serial probe (200 requests
each). Each Swift server stops before the next probe; the vLLM process group,
including GPU workers, stops before the next model.
See [vLLM's pinned serving arguments](https://docs.vllm.ai/en/v0.30.0/cli/serve/).

Gemma and Qwen default to `chat_template_kwargs={"enable_thinking":false}`.
[Qwen3.5-4B's template](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja)
uses that flag to place an empty, closed `<think>` block in the generation
prefix. The runner verifies this behavior against the **resolved revision's**
tokenizer before serving Qwen and records the rendered prefix/verification in
`revision.json`. For other models, `--model-options options.json` supports
per-model switches and additional vLLM options:

```json
{
  "Qwen/Qwen3.5-4B": {
    "chat_template_kwargs": {"enable_thinking": false},
    "vllm_args": []
  }
}
```

Additional arguments cannot override the required revision, dtype, context,
memory, prefix-cache, address, or model-name settings. Other useful arguments
are `--prompt-variants min,cygnet,rules` (default; any distinct subset allowed),
`--concurrency N`, `--vllm-port`, `--swift-port`, `--health-timeout SECONDS`,
`--manifest PATH`, `--output PATH`, and `--archive PATH`. Live output must be an
empty directory. The lower-level collector resumes only exact bindings and
preflights the entire requested input before making any reader call. Changed
inputs, candidate order, targets, splits, model/revision or runtime recipe are
errors; duplicate IDs and legacy rows without bindings are refused.

## Canonical readout and parity gate

`canonical_letter` means softmax over one canonical token per option letter at
the assistant answer position. HF derives each ID by tokenizing the actual
chat-template-rendered prompt plus that letter; it requires exactly one appended
token. Vocabulary alias scanning is available only with HF `--readout alias_sum`
for diagnostics. vLLM accepts only exact letter token strings and records/rejects
letter aliases. It requests `top_logprobs=min(20, number_of_letters)` and requires
every letter to be present. Missing mass is an error, including 21–26 letter
reads whose top-20 response cannot cover all candidates.

`scripts/swift/parity.py` reads the same rendered messages with HF and vLLM,
reporting maximum/mean absolute probability difference and argmax agreement.
The runner uses 50 single-pass items per variant, maximum difference ≤0.02 and
agreement ≥0.98. Configure `--parity-n`, `--parity-max-abs`,
`--parity-min-agreement` and `--parity-hf-device`; failure aborts that model's
collection and still packs the partial results, including `parity.json` and
`parity.log`. HF defaults to CPU/bf16 so it does not compete with the active
vLLM GPU allocation; the host needs RAM for the reference model. This gate
requires actual GPU-host verification; CPU fixtures do not establish parity.

Swift uses the shared `read_artifact.fingerprint` and `resolve_target`, but its
`swift_letter_messages_v1` binding is intentionally distinct from the native
contract. `make_binding`/`ReadIndex` require raw logits, actual input token IDs,
and tokenizer fingerprints, which the processed-logprob vLLM HTTP API does not
provide; grouped reads also exceed the native contract's 26-candidate limit.
The Swift equivalent binds model ID/revision, backend/readout/logprobs recipe,
prompt variant/state format, rendered messages, ordered labels, state, target,
split and case lineage. It also hashes the reader/prompt/grouping implementations.
Grouped bindings include the deterministic initial messages; each actual pass,
including the adaptive winner pass, records a message/label binding derived from
the parent recipe. These hashes bind declared content and do not attest server
execution or hash the checkpoint weights.

## Policy fitting and reports

`Policy()` defaults to no commitment (`commit_margin=None`). Legacy JSON with
`noul_commit=true` and no margin loads explicitly as margin 0. Fit accepts only
explicit non-public `split="calibration"` rows. `--diagnostic` (or legacy
`--allow-public`) permits exploratory fitting and saves `promotable:false`.
Collection preserves the original split, including Cygnet's `private` split.
Canonical clusters use metadata source lineage/case-facts hash; public items
preserve `group` or item ID, and Cygnet preserves item ID.

Explicit `gold_distribution` targets take precedence over hard argmax hints,
with strict label/mass validation; hard and distribution-target counts are
recorded. Canonical source booleans are converted to numeric 0/1 at adaptation.
Swift retains candidate log masses for stable temperature scaling and CE with
the same semantics as `read_artifact.logspace_nll`; that helper itself requires
a native raw receipt. Legacy probability zeros with positive target mass have
infinite NLL, reported as `NLL:null` plus `nll_infinite_n`, without a floor.

Evaluation reports raw / temperatures / temperatures+commit(margin 0) / fitted
arms, each with typed competence and emitted-probability NLL, class-summed Brier
and ECE per primitive. These all-item probability diagnostics are separate from
JevBench's tier/scoring calibration axes. Score ordinal-value nMAE/chance/CC
diagnostics are separate from JevBench's position metric; the HTTP API requires
contiguous integer ordinals. Positive effective reasoning budgets receive 422
before inference; omitted settings, off and explicit zero budgets are accepted.

Hierarchical reads record `readout="grouped_approx"`, pass counts and aggregate
token usage. Evaluation and fitting report them separately and exclude them
from single-pass metrics by default. Bootstrap helpers and the prompt selector
resample complete `cluster_id` groups across types/languages, stratifying by
primitive coverage; public items are independent singletons.

## Counts and latency

The current default input inventory, **per model per variant**, is:

| Dataset | JSONL rows | Decisions | One-position model reads |
| --- | ---: | ---: | ---: |
| v2 calibration | 1,472 | 1,824 | 2,208 |
| v2 dev | 1,472 | 1,824 | 2,208 |
| Cygnet calibration | 241 | 241 | 241 |
| JevBench public | 231 | 231 | 231 |
| Bulk total | 3,416 | 4,120 | 4,888 |

Canonical rows can contain multiple questions; option sets above 26 require
hierarchical reads, so model-read counts exceed decision counts. Three variants
give **12,360 bulk decisions / 14,664 model reads + 600 serial requests per model**.
Both default models total **24,720 bulk decisions / 29,328 model reads**, plus
**1,200 serial Swift HTTP requests** (200 extra model reads per model/variant
with these public items), or **30,528 model reads including probes**. The default
parity gate adds **600 forwards** (300 HF, 300 vLLM), for 31,128 total forwards.
Dry runs recompute these counts from the manifest, rather than hard-coding them.

The probe deterministically shuffles the 231 public items with seed 20261004 and
uses 200 distinct items, one HTTP request and one question at a time, through
`/v1/systemone`. No bulk requests are active then. It includes Swift HTTP,
prompt rendering, vLLM, and response transfer in the measured seconds. It runs
after bulk collection, with prefix caching enabled and no artificial warmups;
this is a warm-cache measurement, and that context matters when comparing Speed.

Each variant's `latency.json` contains its variant, each raw sample, raw `p50_s`/`p95_s`, completed count,
and `complete`. Quantiles use the same order-statistic convention as the local
JevBench evaluator. For JevBench's self-hosted Speed estimate, first adjust each
quantile with `seconds * 2 + 0.15`, then average the scores
`100 - 20 * log10(adjusted_seconds / 0.1)`. Bulk row `latency_s` measures backend
timing under load and must **not** be used for Speed. A partial probe's quantiles
are clearly marked incomplete.

## Choose the prompt on non-public dev

`min` preserves the original messages byte for byte and remains the serving
default. `cygnet` reproduces Cygnet's MIT system scaffold and measured user
layout (state, instructions, Options, letter-only answer suffix); Noul remains
false-first. `rules` adds a short system scaffold for governing definitions,
exceptions/amendments, effective dates, arithmetic and chained inference. The
CPU whitespace proxy measures 42 extra tokens (budget: 80); real model
token counts come from collected `input_tokens`, including grouped passes.

Run separately for each model after copying results back:

```bash
M=out/google_gemma-4-12B-it
python scripts/swift/select_variant.py \
  --calibration "$M"/{min,cygnet,rules}/v2_calibration.reads.jsonl \
  --dev "$M"/{min,cygnet,rules}/{v2_dev,cygnet_calibration}.reads.jsonl \
  --latency "$M"/{min,cygnet,rules}/latency.json \
  --output "$M/variant_selection.json" --policy-dir "$M/policies"
```

The existing fit runs separately on each variant's **v2 calibration** reads.
Evaluation uses the SAME **v2 dev + 241 Cygnet generated items** for every
variant (2,065 input decisions per variant, before grouped-read exclusion).
The selector excludes grouped approximations from these single-pass comparisons.
It refuses public or unmarked
reads, duplicate/missing rows, changed gold/type/case metadata, overlapping
calibration/dev IDs and mixed models/revisions. Do not pass public accuracy
reads to it. Freeze the selected prompt/policy before opening public diagnostics.

Reports include I, C, local A, Speed and Cost. Cost uses mean input tokens only
with `--usd-in-per-m 0.0403` by default. Speed uses each variant's complete raw
serial latency probe; probes must match model/revision and sampled items.
Without probes, all variants use the explicitly marked `--speed-axis 91`
estimate; bulk `latency_s` never supplies Speed.

The paired case bootstrap uses B=2,000, seed=15, resamples whole cases within
primitive-coverage strata, and reports 95% CIs for A and I deltas against `min`.
Every draw keeps paired variants and questions in each case together, recomputes
I/C/input cost, and holds fitted policies and Speed fixed. Select the highest A;
if its A-delta CI against the runner-up includes zero, prefer the one with fewer
mean input tokens (equal tokens retain higher A). The report includes that
comparison and decision, per-variant policies and a selected `policies/policy.json`.

Serve the selected variant explicitly:

```bash
python -m ayaka.swift.server --policy "$M/policies/policy.json" --prompt-variant SELECTED_VARIANT
```

Policies record `prompt_variant`; old policies default to `min`. A server rejects
a different `--prompt-variant` unless `--force-variant` is explicitly supplied.

## Deadline and outputs

`SWIFT_MAX_MINUTES` defaults to 75; `--max-minutes` overrides it. The clock starts
at runner entry, including input checks, installation, Hub resolution, and server
startup. Up to 30 seconds (20% for shorter caps) are reserved for cleanup and
packing. All blocking subprocesses, health waits, and final compression are
bounded by the remaining budget. Process groups receive TERM, then KILL after
a shared grace period of at most three seconds. SIGINT/SIGTERM also clean up
and pack. `progress.json` identifies finished steps, resolved revisions, errors,
and `complete`, `partial_failure`, `time_cap`, or `interrupted` status.

Each `out/<model_slug>/<variant>/` contains four dataset `*.reads.jsonl` files,
`latency.json`, `swift.log`, and collection/probe logs. Shared `vllm.log`, `revision.json`,
and `env.txt` are under `out/<model_slug>/`, with GPU name, driver, memory, package versions, requested/resolved
revisions, and model options. Steps that never started may have no output file.
Each successful bulk row is flushed; each latency sample is saved atomically.
Installation logs, the input manifest, and the progress marker are under `out/`.

Finally, the runner produces **`out.tar.zst` and `out.tar.zst.sha256`**, including
partial results after a cap or failure. Compression uses installed Python
`zstandard`, or Ubuntu's `libzstd` without requiring a successful pip install.
Ensure one of these is available before starting. The reserved packing phase
must fit the remaining cap; an unavailable compressor or stalled filesystem is
reported as a packing failure. Exit codes: 0 complete, 1 failure/partial failure,
124 time cap, 130 interrupted. Verify with `sha256sum -c out.tar.zst.sha256`.
No cloud deployment or paid instance launch is performed by these scripts.
