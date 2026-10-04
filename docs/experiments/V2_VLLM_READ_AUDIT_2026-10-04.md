# v2 native letter reads: pinned vLLM source and CPU audit

This audit resolves parts of the reader contract before paid GPU use. It does
not measure model accuracy, HF/vLLM engine parity, latency or training benefit.
Claude owns Swift implementation. Codex reviewed the proposed backend contract
and shared the findings through `.dev` without editing Swift files.

## Version and evidence

The Swift runner currently pins vLLM **0.30.0**. The inspected source is tag
`v0.30.0`, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
Nine primary source files were downloaded to the ignored local directory
`runs/vllm-audit-20261004`; no package installation or model download occurred.
The source manifest SHA-256 is
`c6c3312dd6cb65fe151cf0123378593f0f5035ab59944c53b68868c1db1c59af`.

Local evidence is `.dev/codex-vllm-source-audit-20261004-r2.json`, produced by
`.dev/codex_vllm_source_audit.py`. The helper extracts selected source methods
with AST and runs CPU toy tensors. It substitutes a simple output container
and equivalent CPU rank counter; it does not import a vLLM engine or exercise
GPU kernels. The original first receipt is preserved separately.

## Findings

| Question | Source and CPU observation | Consequence |
| --- | --- | --- |
| Does temperature zero collapse reported scores? | The classic all-greedy branch returns unscaled log-softmax; the newer sampling state skips temperature processing for a zero/one batch. The extracted zero-temperature paths preserved scores. | Temperature zero alone is not grounds to reject the proposed reader. |
| Can candidate tokens outside vocabulary top-20 be read? | Explicit token-id gathering returned both requested low-ranked tokens. Their normalized probabilities matched direct canonical softmax. | Use the pinned explicit-ID API instead of inventing missing mass. |
| Does sampler “raw” guarantee grammar-free model logits? | The classic runner applies a grammar bitmask before calling the sampler. | Avoid structured-choice masking for the raw native contract. |
| Is the chat response lossless for extreme scores? | Serving clips score fields below -9999. CPU scores -10000/-10002 serialized identically. | Reject floor values or use an unclipped native engine path. |

For the explicit-ID fixture, 30 other tokens outranked both candidates. IDs
30/31 at logits -100/-102 yielded probabilities **.880797/.119203** through the
gathered log-probabilities, matching direct candidate normalization. Their
absence from top-20 therefore need not prevent a complete read.

For the response-clipping fixture, native scores -10000/-10002 likewise yield
**.880797/.119203**; the serialized scores -9999/-9999 yield **.5/.5**.
This is an arithmetic counterexample, not an estimate of how frequently a
real checkpoint produces these scores. A strict client must conservatively
reject candidate values at or below the wire floor.

Primary sources:

- [Classic sampler: greedy and explicit-ID gathering](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/sample/sampler.py)
- [New sampling state: zero-temperature handling](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/sample/states.py)
- [Classic runner: grammar application order](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py)
- [Chat protocol: token-ID logprobs and token-ID response names](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/openai/chat_completion/protocol.py)
- [Sampling validation: explicit-ID limits](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/sampling_params.py)
- [Chat serving: response score clipping](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/openai/chat_completion/serving.py)

## Proposed Swift integration

Define one canonical answer token per candidate at the actual rendered assistant
prefix. Verify appending each letter adds exactly one distinct token without
changing prefix tokenization. Bind the same model/tokenizer revisions, template,
chat kwargs, actual input tokens and candidate map on HF and vLLM.

For vLLM 0.30.0, request `logprobs=true`, `logprob_token_ids` for those exact
candidate ids, and `return_tokens_as_token_ids=true`, with `max_tokens=1` and
no structured output grammar. Explicit ids take precedence over natural top-k;
the version accepts at most 128 requested ids, sufficient for 2–26 canonical
candidates. Pin server `raw_logits` and forbid conflicting overrides.

The wire field is named `logprob` even when the server returns raw logits.
Parse by token id, require all requested ids exactly once, check finite values
and reject score-floor hits. An extra sampled-token entry can be excluded after
validation. Feed the gathered raw logits to `make_record`, with one alias id per
candidate; never exponentiate unshifted logits or mark processed scores as raw.

This approach is a source-supported proposal, not a completed Swift adapter.
CPU fake-HTTP tests should cover explicit ids, missing/duplicate ids, the floor,
21–26 candidates, skewed scores and incompatible server configuration.

Actual HF/vLLM model/kernel parity remains necessary before adopting this
backend. Predeclare input-token equality, dtype, tolerances and a small parity
cohort, including 2/20/26 candidates and permutations. Freeze the total eligible
calibration/dev workload and an affordable complete schedule before starting a
paid instance. A timeout remains an emergency guard, not the normal completion
plan. No new GPU experiment was authorized or started by this audit.

## Related selector issue

The new prompt-variant selector accepted `split="test", public=false` rows as
both calibration and dev in `group_reads`, because it checked public status
without requiring the declared partition to match the requested role.
This counterexample invoked grouping only: no policy fitting, test-data scoring
or model inference was performed. Receipt:
`.dev/codex-variant-selector-counterexample-20261004.json`; inspected source SHA
`60a63404ffdf3e22f9ca4cc7454dba78c1d62ad60a596d9b0d918ef19368d387`.

The finding was sent to Claude as R12. Partition and case/lineage/input checks
must run before fitting or prompt selection; source-path/question-id matching
alone does not establish isolation. A proposed fix and a verified fix remain
different states. The existing independent test partition was not opened.

A later synthetic preflight also called the selector with valid calibration
rows and test-marked dev rows. A sentinel intercepted `fit_policy` at entry:
the selector reached fitting before rejecting the dev/test role mismatch.
No fit was computed. A separate sentinel showed that calibration-marked rows
without any read binding reached the temperature-fit entry too (R13).
Strict record validation must therefore cover fitting and selection inputs,
as well as collection resume. Legacy unbound rows need an explicit diagnostic
path that cannot produce a promotable policy.

Receipt: `.dev/codex-swift-preflight-20261004.json`. Five reviewed Swift source
hashes were unchanged during this short probe. It also confirmed that the new
native-only service rejects `on + high` with zero reader calls, and the default
Noul commitment margin is `None`. These are CPU control-path observations;
HTTP status and complete configuration precedence still need related tests.
