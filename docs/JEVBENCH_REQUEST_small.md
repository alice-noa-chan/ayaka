# JevBench benchmark request: Ayaka small (GitHub issue draft)

---

**Title:** Benchmark request: Ayaka small (Gemma 4 E2B decision model, MIT)

### Model

- Hugging Face: <https://huggingface.co/alice-noa-chan/ayaka-small> at
  revision `8ef97557668d17e2455be322ac8cbbb59b08ece1`
- Code: <https://github.com/alice-noa-chan/ayaka> at commit `df010826b1807e94b7f1fc8d09eeacc7046ae372`
- License: MIT for code and fine-tuned weights. The base model
  `google/gemma-4-E2B-it` is Apache-2.0. The model card lists all 35
  training sources with their licences. No commercial-LLM or Jev API
  outputs are used.
- **Naming:** a Gemma 4 model. `electra-small` and `electra_config.json`
  are legacy internal names.
- **Family:** the lightweight member of the Ayaka family. Ayaka large is
  submitted separately.

### How to run

```bash
C=df010826b1807e94b7f1fc8d09eeacc7046ae372
pip install -c https://raw.githubusercontent.com/alice-noa-chan/ayaka/$C/constraints.txt   "git+https://github.com/alice-noa-chan/ayaka@$C"
python -m ayaka.serve --ckpt alice-noa-chan/ayaka-small --revision 8ef97557668d17e2455be322ac8cbbb59b08ece1   --device cuda --port 8000
```

Single pass only: no `--reasoning`, and no tokens are generated. The server
and its input/output format are the same as for Ayaka large: TypeSafe
`POST /v1/systemone` with per-label probabilities. The verified
environment and the Dockerfile are also the same.

### Exact configuration

- **Base:** `google/gemma-4-E2B-it`, revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, text stack only.
- **LoRA:** r 32, alpha 64, dropout 0.05, on q/k/v/o/gate/up/down. Stored
  as bf16 safetensors.
- **Training:** 1200 steps × 64 questions, LR 1e-4 (head 5e-4), seed
  0, one A100 80GB, 1.4 h. Same 35 sources and code as
  large, trained directly (no distillation).
- **Temperatures** (long means ≥ 1,024 prompt tokens):

  | type | short | long |
  |---|---|---|
  | noul | 0.963 | 1.247 |
  | choice | 1.303 | 1.125 |
  | score | 1.157 | 1.134 |

- **Precision:** BF16. Weights take about 10 GB.
- **Latency** (A100 80GB, one request at a time, in-process, p50 / p95):
  - easy: 0.072 / 0.074 s;
  - original: 0.072 / 0.074 s;
  - hard: 0.074 / 0.169 s.
- **Tokens per decision:** mean input 348 on Open-Jev test, 146 on public
  original and 1,242 on public hard. Output is 0.

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
lower): easy 48/48, original 65/72, hard 55/111.
