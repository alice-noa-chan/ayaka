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
bash scripts/swift/collect_gpu.sh --max-minutes 85
# Or explicitly admit a prefix within the default 75-minute cap:
bash scripts/swift/collect_gpu.sh --max-minutes 75 --allow-partial
```

| Priority | Work minutes | Load minutes | Work |
| --- | ---: | ---: | --- |
| P0 | 10 | 5 | 12B HF GPU bf16 reference; exit/free GPU memory; vLLM load and exact-ID parity |
| P1 | 35 | 0 | gemma-4-12B-it x min,cygnet,rules,labeled: calibration, dev, Cygnet; select on dev; public diagnostic last |
| P2 | 5 | 0 | 200 serial HTTP requests for each of P1's best two variants |
| P3 | 20 | 5 | E4B x the same variants/datasets, with its own sequential HF/vLLM parity |
| P4 | 15 | 5 | Optional ayaka-large LoRA arm on the identical pinned 12B base/tokenizer |
| Environment prep | 3 | 0 | Environment preflight |
| Cleanup/pack | 2 | 0 | Fixed 120-second reserve inside the cap |

These are planning estimates, not measured throughput. Each new model/adapter
adds 2 minutes for HF loading and 3 for vLLM loading. The enabled plan totals
85 minutes; enabling P4 gives 105 minutes. Before environment prep or model
loading, the runner refuses an over-budget plan. The default 75-minute cap
therefore needs a larger cap or `--allow-partial`. With that flag, only the
priority prefix whose full estimated total fits is admitted; at 75 minutes this
is P0/P1/P2. All remaining enabled priorities are marked `skipped` with
`skip_reason: not_admitted` up front. Dry runs print the admission table, including
refusals, without launching anything. If no priority fits, no environment or
model work starts.

The monotonic hard deadline is fixed at runner start plus `max_minutes * 60`.
Environment and priority work deadlines are the smaller of their own budget
deadline and the hard deadline minus 120 seconds. Priority budgets include the
load estimates above. A priority that reaches its deadline is stopped and marked
`timeout`, and later admitted priorities are skipped. Variant fitting also runs
in a supervised child so it cannot leave vLLM running through unbounded CPU work.
Cleanup and packing use only the time remaining before the original hard
deadline; they never receive a fresh budget. `status: complete` requires all
enabled priorities to finish within their deadlines and packing to succeed;
admitted prefixes and timeouts produce `status: partial`.

The primary pin is `google/gemma-4-12B-it` at
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; E4B needs an explicit revision.
The variant set is fixed to `min,cygnet,rules,labeled`. Models run alone in bf16 with
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

P1 fits each variant on v2 calibration and chooses a single prompt proposal by
calibration A, breaking ties in the fixed order min, cygnet, rules, labeled. It
compares that proposal with the existing fitted min policy on matching non-public
dev. The historical dev ranking and token fallback remain diagnostic reports.
The final promotable selection always uses the adoption gate. P2 probes the two
highest-dev-A variants with diagnostic serving; public latency is never fed back
into fitting or gates. The public accuracy arm runs after the selection is frozen.

### Offline adoption

All new levers default off. Supply the already fitted min policy and bound,
non-public calibration/dev reads to request optional candidates:

```sh
python scripts/swift/adopt.py --calibration cal/*.reads.jsonl --dev dev/*.reads.jsonl \
  --baseline-policy baseline.json --levers variant bias \
  --output adoption.json --policy policy.json
```

The fixed sequence is variant then bias; reasoning routing is reserved and refused.
A variant is chosen on calibration only, then tested once on dev. Bias fits only
calibration rows for the accepted variant, with frozen temperatures. Its objective
is mean pre-commit NLL + 0.001/2 times squared offsets. Choice and Score use one
vector per option count; Noul uses one true-logit intercept. Buckets below 30
questions have no bias. Bias uses raw canonical letter masses and applies before
temperature; it requires no additional inference.

For each requested lever, paired whole-case bootstrap uses the existing primitive
coverage strata, B=2000 and seed=15. Adopt only when the 95% lower bound of delta A
is strictly positive, delta I's upper bound is at least zero, point delta I is at
least -0.5, and every primitive's point CC delta is at least -2. An accepted lever
becomes the next baseline. Rejection leaves it off; there are no dev retries.
These finite-sample gates provide evidence rather than certainty of future gains.

Token Cost is computed from the measured read counts and declared prices. Speed
uses complete serial non-public dev probes supplied with --latency, otherwise
--speed-axis (91) is recorded as assumed. --assume-cost records the fixed
--cost-axis (56.4) as assumed. Bulk concurrency timings do not supply Speed.
Public/test reads, diagnostic reads, invalid bindings, role overlap and mismatched
recipes are refused before fitting. There is no exploratory override in adoption.

adoption.json records constants, fitted candidates, deltas, CIs, reasons and read
hashes. policy.json contains only accepted levers. If none pass, its parameters
are exactly the input baseline. Accepted policies carry a receipt bound to their
parameters and data hashes. Serving refuses optional levers without matching
passing receipts; --diagnostic/--force-variant explicitly serves unpromotable
candidates. Candidate-only fits of non-min prompts are never promotable.

## Outputs and counts

The manifest currently gives 4120 decisions / 4888 model reads per model/variant,
including grouped reads. Both models and four variants total 32960 bulk
decisions / 39104 model reads. P2 adds 400 serial requests; parity adds 112
forwards (7 items x 4 variants x HF/vLLM x 2 models). Optional P4 adds its own
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
0 complete or an explicitly allowed partial prefix, 1 failure, 124 work/cleanup
timeout, and 130 interrupted. Packing may be incomplete if the hard deadline is
reached; the local progress receipt records its outcome.
These scripts do not launch a paid instance or push/deploy anything.
