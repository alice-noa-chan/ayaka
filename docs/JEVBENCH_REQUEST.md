# JevBench benchmark request (GitHub issue draft)

Ready to post as an issue at
<https://github.com/fstandhartinger/jevbench/issues>.

---

**Title:** Benchmark request: Ayaka large (Gemma 4 12B decision model, MIT)

### Model

- Hugging Face: <https://huggingface.co/alice-noa-chan/ayaka-large> at revision
  `018642a6eff74f84219df2d026fc01fa22ecd246`
- Code: <https://github.com/alice-noa-chan/ayaka> at commit
  `7727e9ec5cec062c05de34ad40ade6fc1b6866d3`
- License: MIT for code and fine-tuned weights. The base model
  `google/gemma-4-12B-it` is Apache-2.0 and its notices apply to the base
  portion. The model card lists all 35 training sources with their
  licences. No commercial-LLM or Jev API outputs are used.
- **Naming:** this is a Gemma 4 model. "Electra" is the project's earlier
  model name and remains in internal identifiers (`electra-large`,
  `electra_config.json`, which is a legacy alias of `ayaka_config.json`).

### How to run

```bash
C=7727e9ec5cec062c05de34ad40ade6fc1b6866d3
pip install -c https://raw.githubusercontent.com/alice-noa-chan/ayaka/$C/constraints.txt \
  "git+https://github.com/alice-noa-chan/ayaka@$C"
# downloads the adapter and head (525 MB) plus the pinned Gemma 4 12B base
python -m ayaka.serve --ckpt alice-noa-chan/ayaka-large \
  --revision 018642a6eff74f84219df2d026fc01fa22ecd246 --reasoning --device cuda --port 8000
```

A Dockerfile is in the repository. It builds from the exact base image,
pinned by digest, with the same constraints.

The server is `POST /v1/systemone` in TypeSafe's format. Health:
`GET /health`. It serves one request at a time, and all questions of a
request share one state encoding.

**Verified environment** (used for every reported number):

- Python 3.11.11 and torch `2.8.0.dev20250319+cu128`, from
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (digest
  `sha256:cb154fcc…`).
- transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, safetensors 0.8.0,
  tokenizers 0.23.2.
- NVIDIA driver 570 or newer.

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
change the answer. `usage.output_tokens` counts the generated worked-steps
tokens.

### Exact configuration

- **Base:** `google/gemma-4-12B-it`, revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Only the text stack is loaded.
- **LoRA:** r 64, alpha 128, dropout 0.05, on q/k/v/o/gate/up/down
  projections. It is stored as bf16 safetensors. Parity against the fp32
  training copy gives the same accuracy, with a median probability
  difference of 0.001.
- **Training:** 860 steps × 64 questions, LR 1e-4 (head 5e-4), seed 0, one
  H100, 35 license-clean sources.
- **Pointer head:** width 512, 3 set-mixer layers (`head.safetensors`). The
  gate is learned per question type and length and ended at 0.001-0.04.
- **Temperatures:** fitted on held-out data. Long means ≥ 1024 prompt tokens.

  | type | short | long |
  |---|---|---|
  | noul | 0.838 | 1.409 |
  | choice | 1.270 | 1.401 |
  | score | 1.088 | 1.187 |

- **Context:** prompts of up to 8192 tokens at inference; training used 4096.
- **Worked-steps route** (`--reasoning`):
  - A question is routed when all three hold: the question text or options
    ask about a quantity (rubric level numbers do not count), the state has
    at least three numbers, and the single-pass top probability is at most
    0.9.
  - The base model (LoRA disabled) writes at most 384 greedy tokens of
    worked steps.
  - The trained model re-reads the state with those notes, and that
    distribution is returned.
- **Precision:** BF16 weights and activations.
- **Hardware:** tested on NVIDIA A100 80GB and H100 80GB. Weights take
  about 24 GB in BF16. Peak serving memory has not been measured separately;
  it ran with room to spare on 80 GB cards.

### Latency

A100 SXM 80GB, one request at a time. Single-pass numbers were measured
in-process. `--reasoning` numbers went through the HTTP server, which adds
about 0.1 s.

| tier | single pass p50 / p95 | with `--reasoning` p50 / p95 |
|---|---|---|
| easy | 0.18 / 0.19 s | 0.29 / 0.30 s |
| original | 0.18 / 0.18 s | 0.29 / 0.30 s |
| hard | 0.19 / 0.80 s | 9.9 / 31.4 s |

On H100 the single-pass p50 is 0.15 s. The rubric-aware gate routes 0 of
400 HelpSteer2-style rating questions, 18 of 399 hh-rlhf comparisons before
the confidence cutoff, and 4% of Open-Jev decisions.

### Tokens per decision

| cohort | mean input | mean output | $ / 1,000 at $0.05/M input |
|---|---|---|---|
| Open-Jev test (300) | 348 | 10 | 0.017 |
| public original (72) | 146 | 0 | 0.007 |
| public hard (111) | 1,242 | 160 | 0.062 |

### Use of the JevBench public set

No JevBench items or labels were included in the training corpus. All 35
training sources were checked against the public items for shared 13-grams,
and 0 samples matched. However, public JevBench results were used as a
development signal. They helped identify weak task families and informed
synthetic-data design, model-size selection, and the decision to develop a
reasoning path. Public-item outputs were also read to diagnose failures (a
zero-shot calculation-plan format, a calibration bug). One prompt example
that structurally mirrored a public item was removed before the reported
runs.

**Not selected on public results:**

- **Reasoning-route threshold, gating and fusion.** This covers the
  calculation gate rules, the confidence cutoff of 0.9 and the fusion
  weight. They were selected on a procedural dev set (500 questions) and
  Open-Jev test (300 questions) only. The policy was frozen and
  hash-recorded before any public scoring.
- **Worked-steps writer.** Using the base model with the adapter off,
  rather than with the adapter on, was decided on the dev set.
- **Checkpoint.** Each model is the final checkpoint of its run. No
  intermediate checkpoint was compared on public items.
- **Training length and hyperparameters.** Steps were set by budget (860
  for large, 1,200 for small and base). Learning rate, LoRA rank and the
  data-mixture quotas were fixed before training and not re-tuned on public
  scores. The synthetic sources themselves were designed as described
  above.
- **Calibration temperatures.** They were fitted on held-out
  training-distribution data.
- **Freeze.** The released models are frozen for this submission, and no
  further changes will be made in response to public results.

Our public-tier numbers, for reference (the sealed tier will likely be
lower): easy 48/48, original 72/72, hard 88/111 through the HTTP server.
