# JevBench benchmark request (GitHub issue draft)

Fill the `TODO` fields after the Hugging Face upload, then open an issue at
<https://github.com/fstandhartinger/jevbench/issues>.

---

**Title:** Benchmark request: Ayaka large (Gemma 4 12B decision model, MIT)

### Model

- Hugging Face: `https://huggingface.co/alice-noa-chan/TODO` at revision `TODO`
- Code: <https://github.com/alice-noa-chan/ayaka> at commit `TODO`
- License: MIT for code and fine-tuned weights. The base model
  `google/gemma-4-12B-it` is Apache-2.0 and its notices apply to the base
  portion. Training data licences are listed per source in the model card.
  No commercial-LLM or Jev API outputs are used.

### How to run

```bash
pip install "git+https://github.com/alice-noa-chan/ayaka@TODO-commit"
# weights: TODO (checkpoint with unmerged LoRA, or merged export: see below)
python -m ayaka.serve --ckpt <checkpoint-dir> --reasoning --device cuda --port 8000
```

The server is `POST /v1/systemone` in TypeSafe's format. Health:
`GET /health`. It serves one request at a time, and all questions of a
request share one state encoding.

### Input and output

The request and response use TypeSafe's format. Full reference:
[`ayaka/serve.py`](../ayaka/serve.py).

```json
{"state": "...", "questions": {
  "q1": {"type": "noul", "instructions": "...", "criteria": {"false": "...", "true": "..."}},
  "q2": {"type": "choice", "instructions": "...", "criteria": {"A": "...", "B": "..."}},
  "q3": {"type": "score", "instructions": "...", "criteria": ["level 0", "level 1", "level 2"]}}}
```

```json
{"answers": {
  "q1": {"type": "noul", "noul": 0.83},
  "q2": {"type": "choice", "choice": "B", "probabilities": {"A": 0.12, "B": 0.88}},
  "q3": {"type": "score", "score": 1.4, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}}},
 "usage": {"input_tokens": 1234, "output_tokens": 0}}
```

Probabilities come from label-token logits (plus a pointer head, whose
learned gate is about 0) divided by a fitted temperature. They are a full
distribution over the given labels. Options are shown to the model in a
content-derived canonical order, so their order in the request does not
change the answer.

### Exact configuration

- **Base:** `google/gemma-4-12B-it`, revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Only the text stack is loaded.
- **LoRA:** r 64, alpha 128, dropout 0.05, on q/k/v/o/gate/up/down
  projections.
- **Training:** 860 steps × 64 questions, LR 1e-4 (head 5e-4), seed 0, one
  H100, 35 license-clean sources.
- **Pointer head:** width 512, 3 set-mixer layers. The gate is learned per
  question type and length and ended at 0.001-0.04.
- **Temperatures:** fitted on held-out data. Long means ≥ 1024 prompt tokens.

  | type | short | long |
  |---|---|---|
  | noul | 0.838 | 1.409 |
  | choice | 1.270 | 1.401 |
  | score | 1.088 | 1.187 |

- **Context:** prompts of up to 8192 tokens at inference; training used 4096.
- **Worked-steps route** (`--reasoning`):
  - A question is routed when all three hold: the question text or options
    ask about a quantity, the state has at least three numbers, and the
    single-pass top probability is at most 0.9.
  - The base model (LoRA disabled) writes at most 384 greedy tokens of
    worked steps.
  - The trained model re-reads the state with those notes, and that
    distribution is returned.
  - On Open-Jev decisions, 4% of questions are routed.
- **Precision:** BF16 weights and activations.
- **Hardware:** tested on NVIDIA A100 80GB and H100 80GB. Weights take
  about 24 GB in BF16. Peak serving memory: TODO. Latency, one request at a
  time:
  - single pass: p50 0.15 s (H100) / 0.29 s (A100 over HTTP);
  - routed hard questions: p50 about 10 s (A100).

### Use of the JevBench public set

- **Not used as training data.** Every training sample sharing a 13-gram
  with a public item is dropped.
- **Used for evaluation and diagnosis during development.** Model size, the
  choice to add the worked-steps route, and error analysis (including
  reading failed outputs on public items) followed public aggregate results.
- **Route hyperparameters** (gate, confidence cutoff, fusion weight) were
  selected only on a procedural dev set and Open-Jev test, and frozen before
  public scoring.
- **An early prompt** contained a worked example structurally mirroring one
  public hard item. It was removed before the reported runs.

Our public-tier numbers, for reference (the sealed tier will likely be
lower): easy 48/48, original 72/72, hard 88/111 through the HTTP server.
