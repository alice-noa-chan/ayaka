# Swift collection and exact read receipts

The live runner targets Linux, Python 3.11 and one GPU. Dependencies, model
weights, tokenizers and any optional adapter must already be available. It
checks installed `vllm==0.30.0`; it never runs pip or downloads a tokenizer.
`--dry-run` inventories inputs and prints the fixed plan without GPU, model
loading, subprocesses or network calls.

```powershell
python scripts/swift/gpu_runner.py --dry-run
python scripts/swift/pack_inputs.py
```

The input packer uses the explicit [manifest](manifest.json), refuses
`test.jsonl`, and preserves input hashes and deterministic archive metadata.
Transfer code and the input archive to the intended host yourself. The stored
[parity cohort](parity_cohort.jsonl) travels with the repository code.

## Fixed priorities and budgets

```bash
export SWIFT_GEMMA_E4B_REVISION=FULL_COMMIT_SHA
bash scripts/swift/collect_gpu.sh --dry-run
bash scripts/swift/collect_gpu.sh --max-minutes 75
```

| Priority | Minutes | Work |
| --- | ---: | --- |
| P0 | 10 | 12B HF GPU bf16 reference; exit/free GPU memory; vLLM load and exact-ID parity |
| P1 | 35 | gemma-4-12B-it x min,cygnet,rules: calibration, dev, Cygnet; select on dev; public diagnostic last |
| P2 | 5 | 200 serial HTTP requests for each of P1's best two variants |
| P3 | 20 | E4B x the same variants/datasets, with its own sequential HF/vLLM parity |
| P4 | 15 | Optional ayaka-large LoRA arm on the identical pinned 12B base/tokenizer |
| Setup/pack | 5 | Environment preflight and result archive |

These are planning estimates, not measured throughput. The default enabled
plan totals 75 minutes; enabling P4 gives 90 minutes. The global cap is checked
at priority boundaries: finish the current priority, then mark later priorities
`skipped`. A stuck priority has a separate emergency timeout equal to its table
budget. That timeout keeps diagnostic partials and aborts the run. Cleanup and
packing have a separate bounded 30-second reserve, so the cap is a boundary
stop rule rather than a promise to kill an active priority at that instant.

The primary pin is `google/gemma-4-12B-it` at
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; E4B needs an explicit revision.
The variant set is fixed to `min,cygnet,rules`. Models run alone in bf16 with
context 16384, GPU utilization 0.90 and prefix caching. Extra server arguments
cannot override the pinned tokenizer, revisions, dtype, context, logits mode,
server name or address. Chat kwargs default to `{"enable_thinking": false}`.

P4 is enabled only by `--with-lora-arm`, with a cached `--lora-path` and pinned
`--lora-revision`. The runner checks adapter base/revision metadata, hashes local
adapter files, reads an HF+PEFT reference, and serves the adapter with vLLM LoRA.
PEFT must already be installed for that arm. The receipt records adapter and
base-tokenizer provenance. This compares a native letter weight arm on the same
serving base; it does not reproduce ayaka-large's published hybrid system.

## Canonical raw readout

`canonical_letter_raw` is the raw logit of one predeclared canonical token per
letter at the first assistant position. The client renders the pinned model's
chat template, then requires appending each letter to add exactly one distinct
token without changing the prefix. It lazily loads only a tokenizer through
transformers with `local_files_only=True`, cached per model/revision.

The grammar-free vLLM request uses `logprobs=true`, `logprob_token_ids`,
`return_tokens_as_token_ids=true`, `return_token_ids=true`, `max_tokens=1` and
`temperature=0`. The server runs `--logprobs-mode raw_logits`. Field names and
`token_id:N` strings were checked against the pinned source described in the
[source audit](../../docs/experiments/V2_VLLM_READ_AUDIT_2026-10-04.md).

The wire key `logprob` contains logits in this mode. Never exponentiate an
unshifted value as a probability. Normalize the complete candidate logits with
a shifted softmax, then apply policy. Every requested token must appear exactly
once, be finite and strictly exceed -9999; floor hits are clipped reads and are
rejected. Only an extra sampled token can be ignored. The client also verifies
returned server prompt token IDs against its own input. All 26 letters fit one
request; grouping begins only above 26 options.

`swift_canonical_tokens_v2` receipts bind model/revision, tokenizer revision,
runtime recipe, actual prompt IDs/hashes, canonical ID map/hash, rendered input,
ordered labels, targets, original split and case/lineage provenance. The Swift
read index validates record and binding hashes and gathered logits/probabilities.
It uses the shared fingerprint convention; grouped diagnostics keep a separate
contract from `read_artifact.make_binding`. Hashes detect changed receipts and
inputs; they do not attest checkpoint execution. Legacy receipts require
recollection for strict use. Cache reuse preflights the complete requested input
before starting model reads.

## Parity gate

The [deterministic builder](build_parity_cohort.py) derives seven stored fixtures
from non-public calibration data only: 2, 20 and 26 options, an explicitly skewed
case, a typed score case, and two permutations of the same 26-option item.
Added distractors are marked synthetic; this cohort tests tokenization/kernel
agreement and is not an accuracy benchmark. Rebuild explicitly with
`python scripts/swift/build_parity_cohort.py`; the runner never regenerates it.

Predeclared bf16 thresholds in [parity.py](parity.py) are P max-abs <=0.02,
argmax agreement >=0.98 and centered log-mass max-abs <=0.05 nats. For each item,
compare identical prompt token IDs/hashes and canonical IDs, complete finite
raw gathers, and max abs of `(logit_i - mean(logit))` differences. Centering
ignores a common offset while retaining errors in tiny-probability tails.
The CPU R14 fixture (B=1e-40 versus 1e-9) fails this gate despite negligible P
error and identical argmax. This tolerance is fixed before GPU observations.

The HF reference runs first on the same GPU in bf16 in a subprocess. Only after
that child exits, releasing model/allocator/context memory, does vLLM start.
`load_times.json` and `parity.json` record HF load time and vLLM startup-to-health
time. Any parity failure aborts bulk and preserves an artifact with
`comparison_valid:false`, plus per-item samples/error and logs. CPU fakes do
not establish real model/kernel parity.

Reference rows and comparison samples are saved atomically during the run,
with `comparison_valid:false` until every variant passes. Missing wire values,
nonfinite gathers, interruption and model/server startup failures retain the
completed samples and an invalid diagnostic. Failed startup attempts record
elapsed load time when available; vLLM is never started after an HF failure.

## Fit and select guards

Strict fitting accepts bound, non-public `split="calibration"` reads. Selector
roles must match the recorded calibration/dev split exactly; public, train,
test, unknown and conflicting metadata splits are rejected. Source-namespaced
case/lineage and rendered-message/token hash overlap across all variants are
checked before fitting. Cygnet retains its original split and is collected
separately as a diagnostic; it is never renamed dev or used in selection.
Public and Cygnet rows are explicitly marked `diagnostic:true` in their receipts.
Grouped approximation rows are reported separately and excluded from strict
single-pass selection after the full-role overlap check.

Unbound, legacy or diagnostic fitting inputs require `--exploratory`; policies
and selector reports then carry `promotable:false`. Legacy `--diagnostic` and
`--allow-public` fitting overrides also require that explicit flag. Selector
exploration still enforces role/public/isolation rules and needs rendered input
hashes to check overlap. Fresh bound reads need nonempty model/revisions,
tokenizer revision, runtime, readout and prompt variant with valid Swift hashes.

P1 fits each variant on v2 calibration and ranks it on matching v2 dev. It uses
local composite A, input-only cost and a declared estimated speed axis, with
paired whole-case bootstrap (B=2000, seed=15). A confidence interval containing
zero favors fewer input tokens. P2 probes only the two highest-A variants;
latency reports do not retroactively refit the policies. The public accuracy
arm runs after the policy/variant selection is frozen.

## Outputs and counts

The manifest currently gives 4120 decisions / 4888 model reads per model/variant,
including grouped reads. Both models and three variants total 24720 bulk
decisions / 29328 model reads. P2 adds 400 serial requests; parity adds 84
forwards (7 items x 3 variants x HF/vLLM x 2 models). Optional P4 adds its own
bulk/parity work. Dry-run output recomputes the inventory and prints priorities.

The latency probe shuffles the public items deterministically, requests one
question at a time through Swift HTTP, and records raw p50/p95 seconds with no
bulk traffic. Prefix caching remains enabled after collection, so these are
warm-cache measurements. Bulk `latency_s` must not supply the Speed axis.

`progress.json` records priority statuses, completed steps, errors and elapsed
time. Model directories hold references, parity diagnostics, load times,
resolved revisions, environment metadata, selection and per-variant reads.
Only the best two primary variants have latency artifacts. Successful rows are
flushed as collected. Final `out.tar.zst` and its SHA256 include partial results;
Python zstandard or system libzstd must already be available. Exit codes are
0 complete, 1 failure, 124 boundary cap/emergency timeout, and 130 interrupted.
These scripts do not launch a paid instance or push/deploy anything.
