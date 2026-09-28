# JevBench benchmark request: Ayaka base (GitHub issue draft)

---

**Title:** Benchmark request: Ayaka base (Gemma 4 E4B decision model, MIT)

### Model

- Hugging Face: <https://huggingface.co/alice-noa-chan/ayaka-base> at
  revision `742b7565d54ef7091af5e5966d90d26080a472f4`
- Code: <https://github.com/alice-noa-chan/ayaka> at commit `df010826b1807e94b7f1fc8d09eeacc7046ae372`
- License: MIT for code and fine-tuned weights. The base model
  `google/gemma-4-E4B-it` is Apache-2.0. The model card lists all 35
  training sources with their licences. No commercial-LLM or Jev API
  outputs are used.
- **Naming:** a Gemma 4 model. `electra-base` and `electra_config.json`
  are legacy internal names.
- **Family:** the mid-size member of the Ayaka family. Ayaka large and
  Ayaka small are submitted separately.

### How to run

```bash
C=df010826b1807e94b7f1fc8d09eeacc7046ae372
pip install -c https://raw.githubusercontent.com/alice-noa-chan/ayaka/$C/constraints.txt   "git+https://github.com/alice-noa-chan/ayaka@$C"
python -m ayaka.serve --ckpt alice-noa-chan/ayaka-base --revision 742b7565d54ef7091af5e5966d90d26080a472f4   --device cuda --port 8000
```

Single pass only: no `--reasoning`, and no tokens are generated. The server
and its input/output format are the same as for Ayaka large: TypeSafe
`POST /v1/systemone` with per-label probabilities. The verified
environment and the Dockerfile are also the same.

### Exact configuration

- **Base:** `google/gemma-4-E4B-it`, revision
  `ee0ef6023621cff504d758262d4e04895a5af4a2`, text stack only.
- **LoRA:** r 32, alpha 64, dropout 0.05, on q/k/v/o/gate/up/down. Stored
  as bf16 safetensors.
- **Training:** 1200 steps × 64 questions, LR 1e-4 (head 5e-4), seed
  0, one A100 80GB, 3.1 h. Same 35 sources and code as
  large, trained directly (no distillation).
- **Temperatures** (long means ≥ 1,024 prompt tokens):

  | type | short | long |
  |---|---|---|
  | noul | 0.809 | 1.758 |
  | choice | 1.158 | 1.086 |
  | score | 1.189 | 1.048 |

- **Precision:** BF16. Weights take about 16 GB.
- **Latency** (A100 80GB, one request at a time, in-process, p50 / p95):
  - easy: 0.194 / 0.201 s;
  - original: 0.180 / 0.196 s;
  - hard: 0.190 / 0.401 s.
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
lower): easy 48/48, original 68/72, hard 61/111.
