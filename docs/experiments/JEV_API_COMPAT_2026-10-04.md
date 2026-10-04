# TypeSafe Jev API / SDK compatibility — design and status (2026-10-04)

**Status: design fixed, implementation in progress on `ayaka-v2-experiments`. Not yet verified with the official
SDK.** This page records why the work is needed and the decisions it follows.

## Why

Ayaka's servers (`ayaka.serve` for v1/v2 checkpoints and `ayaka.swift.server`) answer `POST /v1/systemone` in the
TypeSafe wire format, and JevBench's `typesafe` adapter accepts them. The official TypeSafe Python SDK
(`typesafe-sdk` 0.7.2, MIT) is stricter: its response models use `strict=True` and require fields that our servers
do not send yet. Sources are the TypeSafe [API reference](https://docs.typesafe.ai/api.md),
[Confidence](https://docs.typesafe.ai/confidence.md), [Models](https://docs.typesafe.ai/models.md) and the SDK type
reference, all read on 2026-10-04.

| Item | Jev API / SDK | Ayaka servers before this work |
|---|---|---|
| Choice and Score `confidence` | required | missing |
| Score `legend` (level index → description) | required | missing |
| `GET /v1/models` | `{"models": [{name, description, release_date}]}` | Swift: `{"data": [{id}]}` |
| `model: "jev-latest"` (the SDK default) | accepted alias | not resolved |
| structured `instructions` / `criteria` | string, object or array (Choice values may be null) | Swift: strings only |
| limits | Choice ≤ 255 options, Score 2–10 levels | inconsistent |
| overload | 429 with `retry-after`, 529 | not implemented |
| request id | optional `x-typesafe-request-id` header | none |

The confidence formulas follow the documentation exactly:

- **Choice:** (p_max − 1/n)/(1 − 1/n).
- **Score:** max(0, 1 − Σ p_i·|i − m| / MAD_unif), where m is the most likely level and
  MAD_unif = (1/n)·Σ|i − (n − 1)/2|.
- **Noul:** answers carry no confidence field.

## Decisions

1. **Strict superset.** A Jev-shaped request gets a Jev-shaped response with the same required fields and status
   codes. Shared logic moves into one module, `ayaka/jev_api.py`, which both servers use. It covers validation,
   structured rendering, limits, confidence, legend, model listing, aliases and errors. Confidence is implemented
   once, in that module.
2. **Model names.** `jev-latest` and `jev-preview` are accepted so that the SDK works with its default settings.
   The response's `model` field always reports Ayaka's own versioned id. Accepting the aliases is for drop-in client
   use; it is not a claim to be Jev.
3. **One namespace for Ayaka-only features.** Requests carry them in a top-level `"ayaka": {...}` object. The
   official SDK sends it unchanged through `extra_body`. Responses add top-level and per-answer `"ayaka"` objects,
   holding route, calibration status, generated candidates, diagnostics and extension usage. The SDK's response
   models ignore unknown fields, so it parses these responses without error. Existing extension fields
   (`options.reasoning`, per-question `reasoning`, `candidate_generation`, `media`) remain accepted as aliases;
   conflicting duplicates are rejected with 422.
4. **No SDK fork.** `ayaka/client.py` is a thin optional helper. It builds `extra_body={"ayaka": ...}` and provides
   SDK `response_model` subclasses that keep the `ayaka` fields.
5. **Tests use the real SDK.** Each server runs with a CPU fake reader and is called through
   `TypeSafeClient(base_url=...)`. When the SDK is not installed, those tests are skipped. JevBench's own adapter
   smoke test (`deploy/swift/harness_smoke.py`, 231 public items) must keep passing.

## Ayaka-only features covered by the namespace

| Feature | Where | Status |
|---|---|---|
| Choice candidate generation (`open` / `expand`) | `ayaka.questions.<id>.candidate_generation` | v2 server: experimental. Swift: being ported. |
| Image input | `ayaka.media` | v2 server: experimental. Swift: planned, unmeasured. Jev itself is text-only. |
| Reasoning controls | `ayaka.reasoning` / `options.reasoning` | v2 server only. Swift rejects explicit requests and uses only a gated internal route if an adoption gate admits it. |
| Calibration and route diagnostics | response `answers.<id>.ayaka` | Swift |

## v1 (`main`)

Published v1 weights stay frozen, and the JevBench request pins code commit `475bec3`. After the changes are
verified on this branch, only the serving-code compatibility fix is proposed for `main`, as a separate commit, with
the user's confirmation. It does not change model weights or the pinned benchmark request.
