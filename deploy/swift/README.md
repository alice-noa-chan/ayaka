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
Set `AYAKA_API_KEY` to the local key used by clients before exposing the container.

```bash
docker build -f deploy/swift/Dockerfile -t ayaka-swift .
docker run --gpus all -p 8009:8009 -e HOST=0.0.0.0 -e AYAKA_API_KEY \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" ayaka-swift
```

Without Docker: `pip install vllm==0.30.0 && pip install --no-deps -e .`, then
`POLICY=deploy/swift/policy.json bash deploy/swift/serve.sh`.

Warm up until this returns HTTP 200:

```bash
curl -s http://127.0.0.1:8009/v1/systemone -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${AYAKA_API_KEY:-}" -d '{"state": "warm-up",
  "questions": {"decision": {"type": "noul", "instructions": "Is this a warm-up?",
  "criteria": {"false": "no", "true": "yes"}}}}'
```

Then run the harness:

```bash
python3 -m jevbench.cli run \
  --tasks <JEVBENCH>/datasets/public/easy.jsonl,<JEVBENCH>/datasets/public/original.jsonl,<JEVBENCH>/datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8009 --key-env AYAKA_API_KEY \
  --model jev-latest --cost-basis self_hosted_gpu --reserve-usd 0 \
  --results <OUT>/results.jsonl --raw-dir <OUT>/raw --ledger <OUT>/ledger.jsonl --manifest <OUT>/manifest.json \
  --run-label ayaka-swift --delay-s 0
```

**Identity check:** TBD (public correct count and tier split from our own runs).

## Status codes

422 for an invalid body or field (an unknown model/question type, a reasoning budget,
non-uniform Score levels), including malformed JSON. Missing/invalid keys return 401;
full request admission returns 429 with `retry-after`; backend queue overload returns 529;
backend failures return 502; deadlines return 504. Every response includes a fresh
`x-typesafe-request-id` UUID.

## Reasoning controls

Fixed text questions with at most 26 candidates accept `options.reasoning` and
per-question `reasoning` overrides. Fields follow checkpoint, request, then
question precedence; each override replaces only its supplied fields.

- `mode: "off"` or `max_tokens: 0` makes a direct read and disables the saved router.
- `mode: "on"` forces worked steps, even without a router. `effort: "high"` reserves
  1,024 generated tokens; low and medium reserve 128 and 384. An explicit
  `max_tokens` overrides effort. Forced reads skip the redundant baseline read.
- `mode: "auto"` uses the saved router, if present. Its trace budget is bounded
  by both the requested budget and the saved router's budget.
- Omitting reasoning controls preserves the saved policy's existing behavior.

Unsupported readers, image inputs, generated candidate lists, and grouped lists
reject positive reasoning requests with 422 before any question starts inference.
Generated tokens remain in usage; worked steps stay private. Measurements of the
saved router apply to its frozen budget, not to explicit budget overrides.

## Experimental Choice candidate generation

**Generation quality and calibration are unmeasured. This is not part of JevBench;
the benchmark always uses fixed candidates.** The Swift implementation follows the
text proposal/partition rules in [the v2 API](../../docs/MULTIMODAL_AND_CANDIDATES.md),
with the namespace and reasoning differences described here.

No policy, or `{"mode":"fixed"}`, preserves the ordinary inference path and makes
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
mode**; reasoning off/zero does not disable proposals. Positive client reasoning
requests for generated lists return 422. Noul/Score generation and media input are
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


## Official SDK compatibility and service controls

Both `ayaka.swift.server` and `ayaka.serve` accept `jev-latest`, `jev-preview`, the
served name, and the configured versioned `--model-id`. These aliases enable SDK
drop-in use; Ayaka is an independent model and does not claim to be Jev. Set
`--model-id ayaka-swift-<release>` for a deployed version, with
`--model-description` and `--model-release-date` for `/v1/models`. The response
always identifies the actual served version, regardless of the requested alias.
`GET /v1/models` returns `{"models":[{"name":...,"description":...,"release_date":...}]}`.

Install the optional test/client extra with `pip install -e ".[sdk-compat]"`.
The ordinary official SDK retains its standard fields. `ayaka.client` subclasses
its response models to retain top-level and per-answer `ayaka` metadata:

```python
from typesafe_sdk import Choice, TypeSafeClient
from ayaka.client import AyakaResponse, system_one

with TypeSafeClient(api_key="local-key", base_url="http://127.0.0.1:8009") as client:
    result = system_one(
        client,
        "An invoice",
        {"kind": Choice(criteria={"invoice": None, "other": None})},
        ayaka={"reasoning": {"mode": "off"}},
    )
    print(result.answers["kind"].confidence, result.answers["kind"].ayaka)
    # Equivalent official call:
    result = client.system_one(
        "An invoice",
        {"kind": Choice(criteria={"invoice": None, "other": None})},
        extra_body={"ayaka": {"reasoning": {"mode": "off"}}},
        response_model=AyakaResponse,
    )
```

Instructions and criteria accept strings, objects and arrays. Structured prompt
values are compact JSON; Score legends preserve each supplied level description.
Choice supports 2–255 options and Score 2–10 levels. Choice confidence measures
excess above uniform probability; Score confidence measures spread about the first
modal level against the uniform mean absolute deviation. Noul has no separate
confidence field. These formulas live only in `ayaka.jev_api`, including after
candidate expansion merges its final joint distribution.

All HTTP extension diagnostics and usage breakdowns live under `ayaka`. Preferred
request settings are `ayaka.reasoning`, `ayaka.media`, and
`ayaka.questions.<name>.{reasoning,candidate_generation}`. Historical request aliases
remain accepted; conflicting duplicates return 422. Internal v1 Python callers
retain their existing diagnostic aliases. Per-answer calibration is `fitted`,
`unfitted`, `unvalidated_generated_partition`, or `unvalidated_image`.

`--api-key-env NAME` defaults to `AYAKA_API_KEY`. A nonempty key requires Bearer
authentication on POST and `/v1/models`. `/health` remains open. Non-loopback binds
require a key unless `--allow-no-key` is explicit. `--max-inflight` defaults to 32
and `--request-timeout-s` to 60. Admission remains occupied until timed-out backend
work finishes; a deadline cannot cancel an already-running model call. `/metrics`
returns stdlib Prometheus counters, latency buckets and token totals, with bounded
labels and no request content or IDs. A timed-out inference's actual token spend
is counted when it finishes. v1 idempotency replay continues to avoid repeated
inference and token accounting; cache conflicts remain 409.

## Experimental Swift image input

Jev itself is text-only. Swift additionally accepts the image contract in
[the multimodal API](../../docs/MULTIMODAL_AND_CANDIDATES.md) through `ayaka.media`
or its `media` alias: 1–4 base64 PNG/JPEG/WebP images, 8 MiB each, 16 MiB total,
single frames with at most 16 million pixels each, and a 24 MiB HTTP limit.
Pillow validates MIME, decoding and orientation before inference. Swift forwards
images to the same vLLM backend as data-URI image content parts. Remote URLs and
local paths are rejected. The configured backend/model must support images;
Swift's local HF reader does not support this extension.

Image answers report `ayaka.calibration="unvalidated_image"`. Text temperatures,
position bias, commitment and reasoning routing are disabled for the image request.
Image-expanded token counts come from vLLM; the text-only prefix equality check
remains in force for text requests. Image perception, canonical token behavior and
calibration on the pretrained backend remain unmeasured; CPU tests verify transport
and bookkeeping only. Generated partitions remain text-only.
