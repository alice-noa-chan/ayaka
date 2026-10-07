# Native feature adapter for the v2 evidence experiment

`ayaka/training/evidence_features.py` closes the synthetic-feature-only gap by
extracting real backbone tensors through an opt-in text adapter. It accepts an
already loaded **bare, eval-mode** text backbone, without loading weights or
changing production defaults. No paid GPU, API inference or model download was
used for this implementation.

## Paths and input definition

`prepare_evidence_inputs` renders the complete state using the existing Ayaka
segmented-token recipe, with independent suffixes. It creates immutable token ids,
candidate spans, readout token ids, primitive types and display permutations.
It records state and full input fingerprints. It refuses overlength state/question
pairs; it does not call the legacy `encode_decision` truncation path. One answer
token position is reserved. Noul has two candidates; the native adapter accepts
2–26 unique single-token readout ids. The feature head's larger tensor capacity
does not make a larger native alphabet available.

This renderer is distinct from Swift's messages/Cygnet recipe. A matched comparison
must keep the same recipe on both sides. The adapter extracts **direct-read** features;
it does not accept or implement reasoning mode/effort, generation or multimodal input.
API callers still need to honor requested reasoning through the appropriate backend.

| Mode | Backbone calls | Forward token workload | Evidence memory |
|---|---|---|---|
| `prefix_cache` (default) | 1 prefix + Q suffixes | P + sum(Sq) | prefix only, once |
| `full_rows` (reference) | Q complete inputs | Q × P + sum(Sq) | prefix of first row |

P is prefix length and Sq is one question's suffix length. Each cached suffix
receives its own deep copy of the native cache. It cannot update a sibling's cache.
This includes the tested recurrent Qwen cache, and avoids assuming every model can
batch-expand KV tensors. The adapter releases each suffix output after copying its
question/candidate features. It does not retry an unsupported cache using full rows.

State memory and question states are final normalized backbone output. Candidate
features are local means of final suffix option states. Native logits use the
existing restricted output-head evaluator, including actual untied rows, bias,
scaling and supported softcap. Lexical features are optional means of actual
output rows at option token ids. Input embeddings are not a substitute.

The restricted read now preserves float64 when supplied, with minimum fp32
accumulation for fp16/bf16/fp32. Previously a fp64 dot product with hidden=1e-200
and rows=1e200 returned 2 through the full head but NaN through selected rows.
This is a numerical correctness fix, not evidence of improved benchmark accuracy.
Real-checkpoint low-precision native readout parity remains a separate audit.

## Work accounting and training boundary

Context/vocabulary/spans/candidate types, eval mode, total forward-token cap and
estimated stored-feature-byte cap are checked before the first model call.
Defaults: 65,536 forward tokens and 256 MiB stored features per extraction.
Output device defaults to CPU. Metadata records planned/actual calls and tokens,
prefix/suffix lengths, cache type, layer definition, ids and actual tensor bytes.
The conservative byte-width estimate includes all floating parameter/buffer
metadata, so a higher-precision output head cannot evade preflight by following
an fp32 first parameter. Lexical padding preserves output-row dtype independently
of candidate dtype. Pooled candidate and lexical values must be finite; overflow
or nonfinite output rows fail with attempted work recorded.

`FeatureExtractionError.progress` records attempted calls/tokens and the failure
reason if a runtime operation fails, without retrying or inventing a native prior.
Attempted work is not a completed-token measurement or a GPU billing measurement.

Inference outputs are cloned under `inference_mode(False)` to ordinary detached
tensors. This is necessary because head training needs to save its frozen inputs
for weight gradients; an inference tensor alone cannot provide that boundary.
The tests execute actual head backward even when extraction has an outer
inference-mode caller, and verify backbone parameters receive no gradients.

Feature memory is stored once per record rather than once per question. However,
**the prefix cache and one deep-copied branch can coexist on the source device**.
Stored feature bytes exclude weights, KV/recurrent caches, temporary copies,
attention workspace and optimizer state. They are not a peak VRAM guarantee.
Measure those on the intended pinned model/dtype/cache kernel before fixing a GPU
schedule; do not convert avoided forward tokens directly into a cost forecast.

Bundles explicitly carry `artifact_kind=unbound_evidence_features_v1` and
`promotable=False`. They do not bind actual weights/LoRA, runtime implementation,
dataset split/lineage, original targets or ordinal schema for persisted training.
That artifact contract and its resume/cross-split validation remain necessary.
Do not mix such bundles into the existing bound Swift saved-read schema.

## CPU smoke usage

This example creates random tiny weights only; it makes no model-quality claim.

```python
import torch

from ayaka.config import tiny_config
from ayaka.model.decision import AyakaDecisionModel
from ayaka.model.evidence import EvidenceResidualHead
from ayaka.prompt import QuestionView
from ayaka.tokenization import ToyTokenizer
from ayaka.training.evidence_features import (
    extract_evidence_features,
    prepare_evidence_inputs,
)

model = AyakaDecisionModel.from_config(tiny_config(), dtype=torch.float32).eval()
inputs = prepare_evidence_inputs(
    "The request is approved.",
    [QuestionView("noul", "Approved?", ["no", "yes"])],
    ToyTokenizer(),
    context_limit=512,
)
features = extract_evidence_features(
    model.text_model(),
    inputs,
    max_forward_tokens=4096,
    max_feature_bytes=16 * 1024 * 1024,
)
head = EvidenceResidualHead(model.text_config.hidden_size)
output = head(**features.head_inputs())
assert torch.equal(output.logits, features.tensors["native_logits"])
assert features.metadata["promotable"] is False
```

Do not fit calibration before head training. For an admitted head experiment:
train raw head against verified gold/reference → freeze checkpoint → fit policy
on calibration only → compare paired dev → freeze selection → independent test.
Raw reference KL can retain base overconfidence, so report probability losses
before and after fitting. Per Claude's independent review, this head experiment
stays **P5**, after the cheaper native/policy path has a measured result. Group
mixing stays deferred; measure serial head-inclusive latency for any promoted arm.

## Verification

```powershell
.venv/Scripts/ruff.exe check ayaka/training/evidence_features.py tests/test_evidence_features.py
.venv/Scripts/ruff.exe format --check ayaka/training/evidence_features.py tests/test_evidence_features.py
.venv/Scripts/python.exe -m pytest tests/test_evidence_features.py tests/test_evidence_head.py tests/test_evidence_objective.py tests/test_native_backbone.py
```

Initial adapter verification: **121 passed in 11.90s**, including 29 adapter tests.
Follow-up mixed-dtype/pooling boundary verification: **124 passed in 5.86s**,
including 32 adapter tests. Tiny actual HF
Gemma4 with shared KV, untied Granite with bias/scaling, fp64 Granite, and Qwen3.5
with recurrent attention match full native LM logits and full-row/cache features.
Additional tests cover whole-prompt candidate order, siblings/question order,
frozen backward, changed-record independence, zero-forward preflight failures,
missing cache, failed suffix work accounting and a nonfinite output head.
The tiny offline CPU example above was executed successfully as written.

The first combined CPU run had **1011 passed, 2 failed, 1 skipped in 126.87s**.
The two failures were Swift fixture expectations for its new pinned revision and
log-space probability normalization. Three Swift sources changed during that run,
so it is not recorded as stable integration success. After the fixture fixes,
the integrated snapshot passed **1016 tests, 1 skipped in 144.52s**, with no
source changes. The three additional mixed-dtype/pooling regression tests and
their implementation changes were then committed separately. The final integrated
snapshot includes them: **1019 passed, 1 skipped in 121.85s**, with no source
changes. Whole-tree Ruff lint/format checks pass for 225 Python files.

Claude's head/objective review was received and incorporated; the new feature
adapter was sent separately for independent review. Actual CUDA kernels, full
checkpoints, model accuracy and monetized throughput have not been tested.
Local receipts are `.dev/codex-evidence-adapter-final-20261004.xml` and
`.dev/codex-evidence-feature-boundary-final-20261004.xml`, with integration logs
under `.dev/codex-evidence-swift-integrated-20261004.*`. Swift preflight/parity
corrections are owned by Claude and independently reviewed through `.dev`.
The final full-suite receipt/log are `.dev/codex-evidence-swift-final-20261004.*`.
These passing tests do not discharge the independently reproduced runner deadline
expansion: a mocked 60-second global cap became a 600-second priority deadline
before its HF-reference boundary. The budget-complete admission/absolute-deadline
fix and its independent recheck are required before a paid collection job.
