# Images and generated Choice partitions (experimental)

These APIs extend v2 inference. They do not change the published v1 weights or
the fixed-candidate JevBench contract. Native image accuracy, generated partition
quality, and calibration have **not** been measured on pretrained checkpoints.
The new tests verify execution and probability bookkeeping on random CPU models.

## Native image input

Install the optional image dependencies and use an adapter checkpoint:

```bash
uv pip install -e ".[vision]"
python -m ayaka.serve --ckpt runs/my-checkpoint --images --device cuda
```

The checkpoint's pinned base must have native Gemma 4 or Gemma 4 Unified image
components. Text-only and int8 text exports cannot reconstruct missing media
weights. The loader retains the native processor and media modules, shares the
language backbone with the existing decision adapter/head, and checks the unique
total parameter count including media components and adapter against 14B.
Existing base-model licenses continue to apply.

Send explicit image payloads alongside the ordinary state and fixed questions:

```json
{
  "state": "Read the attached invoice. Use only visible evidence.",
  "media": [
    {"type": "image", "mime_type": "image/png", "data": "BASE64_ENCODED_IMAGE"}
  ],
  "options": {"reasoning": {"mode": "on", "effort": "high"}},
  "questions": {
    "paid": {"type": "noul", "instructions": "Does the invoice explicitly say paid?"},
    "document": {
      "type": "choice", "instructions": "What type of document is shown?",
      "criteria": {"invoice": "Invoice", "receipt": "Receipt", "letter": "Letter"}
    }
  }
}
```

PNG, JPEG, and WebP are accepted: 1–4 images, at most 8 MiB per image, 16 MiB
total decoded bytes, and 16 million pixels per single-frame image. The HTTP body
limit is 24 MiB. MIME types must match decoded contents; EXIF orientation is
applied. URLs, server-local paths, audio, video, PDFs, and animated images are
outside this interface. A document page can be supplied as an image.

Processor-expanded image tokens, patch positions and modality masks reach the
native model. Direct questions and large-choice reranking branch off one image
prefix cache per direct group. A reasoning question encodes its own media/chat
prefix, then generates and reads the typed decision using the same active adapter
and cache. Generated traces and readouts never cross question boundaries.

Reasoning settings keep their existing precedence. `off` and `max_tokens: 0`
generate zero tokens. `on + high` starts reasoning without a speculative direct
judgment and retains its 1,024-token budget even for simple questions. Native EOS
may finish early. Context failure does not silently truncate an image or downgrade
the requested budget; the response reports the fallback/error. Expanded media
tokens and large-choice reranking count toward input usage.

Image `auto` currently returns direct judgments with
`finish_reason: "no_validated_image_router"`. Text router, path calibration and
head temperatures are ineligible for images. Diagnostics include
`modality: "image"` and `calibration: "unvalidated"`. These markers describe a
validation gap, rather than an estimate of the probability that evidence is missing.

## Generate or expand text Choice candidates

Generation is opt-in per question through `candidate_generation`. No field means
the existing fixed-candidate behavior. Explicit `{"mode": "fixed"}` also keeps
the ordinary batched inference path. `expand` and `open` require
`experimental: true`, explicit instructions and a `scope` of 1–4,000 characters.
Only text states and Choice are supported. Generated Noul truth values and Score
levels/rubrics are rejected.

The default proposal budget is 384 tokens; `max_tokens` accepts 1–1,024.
`max_new` accepts 1–8 (default 4); open needs at least two proposed candidates.
These are candidate proposal controls, **separate** from the judge's reasoning
budget. Global reasoning `off` or an effective reasoning budget of zero conflicts
with generation and fails before any inference. Supply a fixed list instead when
generation must be disabled.

### Expand an existing Other outcome

```json
{
  "state": "The customer asks where their package is.",
  "options": {"reasoning": {"mode": "on", "effort": "high"}},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the one primary customer request.",
      "criteria": {
        "refund": "The primary request is for a refund.",
        "other": "A primary customer request other than a refund."
      },
      "candidate_generation": {
        "mode": "expand", "experimental": true,
        "scope": "The single primary customer request",
        "other_id": "other", "max_new": 4, "max_tokens": 384
      }
    }
  }
}
```

The original distribution is judged first. Proposals must contain an `id`,
`description`, and explicit `excludes` definition. IDs cannot replace original
siblings. Malformed, empty, oversized or lexically duplicate proposals are
rejected. The final list is frozen and hashed with its scope and parent definitions.
Original parent definitions and both judgment stages remain in diagnostics.

Proposals are generated in a disposable trace. The child decision starts from the
original state, sees the parent condition and all original sibling exclusions, and
does not consume the proposal rationale/cache. Fixed-list router/calibration and
head temperatures are disabled for generated partitions without changing server
defaults. Forced `on + high` still uses 1,024 tokens for each judgment stage;
proposal generation retains its separately requested budget.

For parent `Other = 0.30` and child conditional probabilities `0.70 / 0.20 / 0.10`,
the expanded probabilities are `0.21 / 0.06 / 0.03`. Original sibling probabilities
remain exactly unchanged. The last child is residual Other and retains the original
parent ID. No recursive generation is performed; one bounded proposal is attempted
per generated question per request.

### Generate when choices are absent

```json
{
  "state": "The customer asks where their package is.",
  "options": {"reasoning": {"mode": "on", "effort": "medium"}},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the one primary customer request.",
      "candidate_generation": {
        "mode": "open", "experimental": true,
        "scope": "The single primary customer request", "max_new": 4
      }
    }
  }
}
```

Open mode requires absent criteria and always adds the reserved `__other__` residual
for remaining outcomes. It cannot infer an arbitrary task or a reliable taxonomy
from an empty instruction. Both modes report
`probability_semantics: "conditional_on_frozen_candidate_partition"`,
`validation: "schema_and_lexical_only"`, and `calibration: "unvalidated"`.
Semantic synonym/overlap detection, coverage and information sufficiency are not
established: `coverage` and `information_sufficiency` are `"not_estimated"`.
Probabilities summing to one do not establish exhaustive world coverage. Residual
Other is an outcome bucket; it is not a replacement for unknown evidence.

Inspect `candidate_generation[name].status` before accepting a result. Failed
expansion preserves the original answer. Failed open generation has no answer
for that question and reports the error, actual finish reason and spent tokens.
`usage.candidate_tokens` counts proposal generation, including failures and EOS;
`usage.reasoning_tokens` counts judgment reasoning across all stages. Output usage
includes both. Stage diagnostics preserve budgets, routes and usage separately.

## Validation and promotion

CPU checks cover native logits/full-cache parity for both image architectures,
saved native processors and LoRA loading, media-dependent outputs, question
isolation, candidate permutation, more than 26 choices, actual input usage,
forced budgets, disabled generation, malformed media/proposals, mass conservation,
calibration isolation and request configuration leakage.

Before promoting either feature, collect independent image and generated-partition
data with source/template-separated train/dev/calibration/test splits. Compare
OCR-derived text with native images, attribute perception versus reasoning errors,
test unreadable and contradictory evidence, and measure NLL/calibration and latency
by modality. For generated candidates, annotate semantic equivalence, overlap,
missing outcomes and coverage separately from final choice accuracy. Calibrate
coverage only with a separately validated estimator; do not derive it from softmax
normalization. Keep fixed JevBench evaluation unchanged and use an opt-in evaluation
track for candidate generation. No new GPU training or public release accompanies
this implementation.

The final integrated CPU suite passed 345 tests with one existing Beam-SDK skip,
and Ruff checked/formatted 104 Python files. During validation, Windows loopback
HTTP intermittently reset connections, including a model-free health handler.
The final suite rerun passed; the transport intermittency remains unresolved and
is not a measured model or calibration failure.
