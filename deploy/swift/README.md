# Ayaka Swift — evaluator package (draft)

**Status: not yet measured.** The policy file, measured figures and the identity check below are filled in only after
the GPU collection described in [`docs/experiments/AYAKA_V3_SWIFT.md`](../../docs/experiments/AYAKA_V3_SWIFT.md).
Do not submit this package until every `TBD` is replaced by a measured value.

Ayaka Swift answers JevBench's `typesafe` wire format (`POST /v1/systemone`) with **frozen**
`google/gemma-4-12B-it` (no fine-tuning) served by unmodified vLLM 0.30.0. For a fixed list of at most
26 candidates, each direct question uses one forward pass:
the options are shown as letters, and the raw logit of one predeclared canonical token per letter is read at the
first answer position. A fitted policy then applies per-primitive temperatures and, only if the calibration split
selected it, a Noul commitment band. The default fixed-candidate path uses one output token per direct read.
The experimental candidate-generation extension below is outside the fixed-candidate JevBench path.

## Run

One 48 GB GPU (TBD: cards measured). The model must be readable from the Hugging Face cache or the Hub.

```bash
docker build -f deploy/swift/Dockerfile -t ayaka-swift .
docker run --gpus all -p 8009:8009 -e HOST=0.0.0.0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" ayaka-swift
```

Without Docker: `pip install vllm==0.30.0 && pip install --no-deps -e .`, then
`POLICY=deploy/swift/policy.json bash deploy/swift/serve.sh`.

Warm up until this returns HTTP 200:

```bash
curl -s http://127.0.0.1:8009/v1/systemone -H 'Content-Type: application/json' -d '{"state": "warm-up",
  "questions": {"decision": {"type": "noul", "instructions": "Is this a warm-up?",
  "criteria": {"false": "no", "true": "yes"}}}}'
```

Then run the harness:

```bash
python3 -m jevbench.cli run \
  --tasks <JEVBENCH>/datasets/public/easy.jsonl,<JEVBENCH>/datasets/public/original.jsonl,<JEVBENCH>/datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8009 --key-env '' \
  --model ayaka-swift --cost-basis self_hosted_gpu --reserve-usd 0 \
  --results <OUT>/results.jsonl --raw-dir <OUT>/raw --ledger <OUT>/ledger.jsonl --manifest <OUT>/manifest.json \
  --run-label ayaka-swift --delay-s 0
```

**Identity check:** TBD (public correct count and tier split from our own runs).

## Status codes

422 for an input the system does not take (over the context limit, an unknown question type, a reasoning budget,
non-uniform Score levels); 502 when vLLM fails; 400 for a body that is not JSON.

## Experimental Choice candidate generation

**Generation quality and calibration are unmeasured. This is not part of JevBench;
the benchmark always uses fixed candidates.** The Swift implementation follows the
text proposal/partition rules in [the v2 API](../../docs/MULTIMODAL_AND_CANDIDATES.md),
with the namespace and reasoning differences described here.

No policy, or `{"mode":"fixed"}`, preserves the ordinary response bytes and makes
zero extra backend calls. Generation is opt-in for each Choice question. Prefer
top-level `ayaka.questions.<question id>.candidate_generation`; the official TypeSafe
SDK can send this object through `extra_body`. The legacy per-question
`candidate_generation` key remains an alias. If both are present for one question,
they must match or the server returns 422 before inference.

Open mode requires absent criteria, explicit instructions, `experimental: true`,
and a nonempty scope of 1–4,000 characters. It proposes at least two buckets and
appends the reserved `__other__` residual. For example:

```json
{
  "state": "The customer asks where their package is.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the one primary customer request."
    }
  },
  "ayaka": {
    "questions": {
      "intent": {
        "candidate_generation": {
          "mode": "open", "experimental": true,
          "scope": "The single primary customer request",
          "max_new": 4, "max_tokens": 384
        }
      }
    }
  }
}
```

For expand mode, supply original string criteria on the question, then use this
policy in the same namespace:

```json
{
  "mode": "expand", "experimental": true,
  "scope": "The single primary customer request",
  "other_id": "other", "max_new": 4, "max_tokens": 384
}
```

`other_id` must identify an existing criterion (for example, `"other": "A request
other than a refund"`). The original distribution is scored first with the normal
fixed policy. Proposals refine only that parent. The child judge sees its parent
condition and original sibling exclusions; every original sibling probability is
retained exactly. Children receive `P(parent) * P(child | parent)`. With parent
mass 0.30 and conditionals 0.70/0.20/0.10, their masses are 0.21/0.06/0.03. The last
child is residual Other and retains the original parent ID. Original lists above
26 options use Swift's existing grouping approximation; the small child list is
judged separately, without rescoring the final combined list or its siblings.

Both modes make one bounded proposal attempt per question, without recursion.
`max_new` is 1–8 (default 4; at least 2 for open); `max_tokens` is 1–1,024 (default
384). Swift uses the same frozen model through vLLM chat, with temperature 0,
`enable_thinking=false`, and EOS stopping. HF serving does not support this
extension. The proposal has its own budget and **does not require a reasoning
mode**; reasoning off/zero does not disable proposals. Explicit positive client
reasoning requests still return 422. Noul/Score generation and media input are
also rejected with 422 before inference.

Proposals must be a JSON array of objects containing exactly `id`, `description`,
and `excludes`. Each value must be a nonempty string of at most 2,000 characters;
IDs are limited to 80. Duplicate IDs/descriptions, including Unicode NFKC,
casefolded and whitespace-normalized duplicates, reserved IDs, replacement of
original IDs/descriptions, duplicate object fields, and oversized lists are rejected.
Each frozen description includes the proposal's exclusion definition. Semantic
synonym/overlap detection, coverage and information sufficiency remain unmeasured.

The judge starts from the **original state**, with the frozen list and relevant
scope/parent instructions. It does not see proposal JSON, rationale, or a proposal
continuation/cache. Generated partitions use only the Choice temperature; fitted
position/K bias and the fixed-list reasoning router are disabled. Diagnostics
mark `calibration: "unvalidated_generated_partition"`. Normalization is not evidence
of exhaustive coverage or of sufficient information.

Jev answer fields retain their usual placement. Extensions live only under each
generated answer's `ayaka` object:

```text
answers.intent.ayaka.candidates:
  items: [{id, description}, ...] (ordered final frozen list, residual last)
  hash: SHA-256 of canonical JSON containing that list, scope and original criteria
  mode: open | expand
  parent_id: null for open, original other_id for expand
  residual_id: __other__ for open, original other_id for expand
  status: completed | expansion_failed
answers.intent.ayaka.diagnostics:
  proposal_tokens: {input_tokens, output_tokens}
  proposal_budget, finish_reason, validation, validation_outcome, calibration
  stages: proposal / generated_partition (plus original for expand)
  scope, probability_semantics, coverage, information_sufficiency
  parent_candidates, parent_probabilities, conditional_probabilities (on expansion)
```

Failed expansion returns the original fixed answer and frozen list, explicitly
marked `candidates.status: "expansion_failed"` with the error and validation outcome.
Failed open generation returns 422 with a clear error, spent usage and failure
details under `ayaka.questions.<question id>`; it never claims a successful answer.

Standard `usage.input_tokens` and `usage.output_tokens` include proposals as well as
every scoring stage, including rejected proposals when the backend reports usage.
Top-level `ayaka.usage` breaks these totals down into `proposal_input_tokens`,
`proposal_output_tokens`, `scoring_input_tokens` and `scoring_output_tokens`. No
generation metadata or usage extension is emitted for an entirely fixed request.

CPU verification: `python -m pytest -p no:cacheprovider tests/test_swift_candidates.py`
uses fake readers/generators and a mocked vLLM wire response; no weights, GPU,
downloads or generation-quality measurements are needed.

## Disclosures

- Frozen weights: `google/gemma-4-12B-it` revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, Apache-2.0 with
  Google's Gemma Prohibited Use Policy. Swift code: MIT.
- The one-pass option-letter readout is NInfer's approach, as used by Cygnet; the readout definition, decision policy,
  fitting and packaging are ours.
- Options are presented as letters in the benchmark's label order with criteria text verbatim; Noul shows `false` first.
- Policy parameters were fitted only on our non-public calibration split; JevBench public items were never used for
  fitting or selection (they are reported as a diagnostic). TBD: fitted values and selected prompt variant.
- Price basis: Google's gemma-4-12B-it at bf16, input tokens only. TBD: mean input tokens on the public set.
- Context limit 16384 tokens.

## Wire-format check without a GPU

`python deploy/swift/harness_smoke.py --jevbench <JEVBENCH>` runs JevBench's own `typesafe` adapter and scorer
against the Swift server with a fake reader. All 231 public items return valid answers (scores are meaningless).
