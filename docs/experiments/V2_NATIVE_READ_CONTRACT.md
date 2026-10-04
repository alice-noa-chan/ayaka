# Native saved-read contract for v2

`ayaka.eval.read_artifact` provides an opt-in CPU contract for newly collected
native text reads. It prevents exact-input cache mismatches, discarded soft
targets and probability underflow from being silently accepted in this module.
Claude owns the Swift integration; the current Swift paths are not changed by
adding this library. Existing legacy reads keep their exploratory status.

## Bind inputs before inference

Call `make_binding` with explicit question/split/case/lineage identities, the
exact checkpoint fingerprint, actual rendered messages and input token ids,
full ordered candidate descriptors, native candidate alias ids and the target
distribution. Obtain soft targets with `resolve_target(labels, gold,
gold_distribution)`; an explicit canonical soft target takes precedence over
its hard argmax hint. Conflicting explicit distributions are rejected.

The runtime descriptor requires pinned model/tokenizer revisions and hashes
for the tokenizer, chat template, reader implementation, prompt recipe and
whole alias vocabulary. It includes backend, dtype, context limit and vocabulary
size, with `logits_mode="raw"`, `decision_path="native_logits"`, `modality="text"`.
Candidate-specific alias maps belong to each binding, so Choice/Noul/Score can
share one runtime descriptor without pretending they have identical candidates.

Binding hashes include target, split and lineage too. Actual prompt/candidate
strings and token ids are hashed in their original order; large prompts are not
duplicated in every saved row. JSON object key order is canonicalized, while
candidate and message order remains significant. Metadata is not inferred from
paths, names or existing read ids.

`ReadIndex(records).get(requested_binding)` returns a copy only for an exact
match. An unseen id returns `None`; an existing id with a different binding is
an error. Duplicate ids and corrupt records are rejected. Build the requested
binding and perform lookup before spending inference time.

## Read complete native candidate mass

Enumerate aliases from the pinned tokenizer once and gather their raw logits
at the same actual answer position. Each token id may belong to only one
candidate. Use `make_record(binding, {token_id: raw_logit, ...})` with every
declared alias id. Missing values, NaN, processed-logit modes or duplicate
alias assignments are rejected; missing tokens are never assigned fake mass.

For a local HF adapter, use the model's actual output logits and gather all
required candidate ids in one operation before one CPU transfer. Do not copy
the full vocabulary to CPU on every read or assume the output head equals the
input embedding. Native read construction here performs no model forward.

The record stores shifted candidate logsumexp masses, emitted probabilities
and binding/record fingerprints. A shared offset is removed before aggregation:
even identical logits of `1e30` retain the correct contribution of two aliases
against one. `logspace_nll(record, temperature=...)` uses masses and the full
target distribution, preserving CE when an emitted probability becomes zero.

Two CPU counterexamples verified in tests are:

| Fixture | Contract result |
| --- | --- |
| Equal logits, A has two aliases and B one | A=2/3, B=1/3; omitted alias is an error |
| Logits 0/-1000 with .5/.5 target | Emitted probabilities 1/0; retained NLL=500 |

These are numerical checks, not model capability measurements.

## Fit and isolate

`calibration_rows(records)` accepts only the explicit calibration partition and
one model/runtime recipe. It rejects dev/test/public/train rows, including
mixed inputs. It validates the records but performs no fitting.

`assert_read_splits_isolated({split: records, ...})` checks partition bindings
and cross-split question ids, underlying cases, source lineages, exact state
and actual-input fingerprints. Its identical-state check is conservative and
can reject shared empty or generic states; callers must review dataset semantics
rather than rename cases to evade it. Template/rule/document-voice isolation
still belongs to corpus preparation. Preserve actual global lineage ids before
the dataset adapter discards metadata; do not infer independent cases from
question suffixes or file names.

## Limits before Swift promotion

- Supported path: 2–26 candidates, one native answer position, text only.
  Grouped/hierarchical, multimodal or pointer paths need their own explicit
  contract and must not be labeled native reads by this version.
- Hashes bind declared content; they do not attest backend execution or prove
  that tokenizer alias enumeration is complete. HF/vLLM answer-position and
  raw-logit equivalence still need independent verification.
- Old saturated probabilities cannot recover missing logits. This module does
  not manufacture provenance or upgrade legacy receipts into strict reads.
- The module does not choose Noul commitment, fit temperatures, implement
  reasoning, or change the API. Swift must still explicitly reject unsupported
  forced reasoning rather than quietly accept `on + high` as a direct read.

Implementation: `07dfbf1`. Related evaluator tests: **86 passed** (6.31 seconds),
including 51 contract tests. Lint/format rechecks passed. No GPU, new model
inference or fitting was used for this contract.

The subsequent whole CPU snapshot reported **754 passed, 8 failed, 1 skipped**
(147.70 seconds in pytest). All eight failures were Swift prompt-variant or
runner-fixture expectations. `scripts/swift/select_variant.py` changed during
execution; this is not a stable final integration pass. The source-bound local
receipt and log are `.dev/codex-cpu-integration-20261004-read-artifact.{json,log}`.
The last earlier stable pass, before the contract and prompt-variant changes,
was 694 passed, 1 skipped; it does not cover these later changes.

Claude's shared-document reply accepts reuse of this module for resume binding
and soft-target loss. Code-level review and completed Swift integration remain
pending. Acceptance of a proposed fix is separate from executing and verifying
it. In particular, processed vLLM scores cannot be wrapped as raw native records.
