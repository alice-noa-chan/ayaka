# Swift GPU measurement results — 2026-10-05

Measured on vast.ai with one RTX 6000 Ada 48 GB, vLLM 0.30.0, transformers 5.17,
`google/gemma-4-12B-it` revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. The results are local measurements on
our own non-public splits; they are not JevBench scores. The JevBench public items appear only as labelled
diagnostics and were never used for fitting, selection or gates. Local archives (not committed):

- `runs/swift-gpu-20261005/attempt7/out.tar.zst`, sha256 matches its sidecar;
- `runs/swift-gpu-20261005/m2x2.tgz`, sha256 `47035eeb…99bf3`.

E4B (P3) was not measured.

## Bugs that only the GPU run exposed (all fixed and pushed)

| Commit | Problem | Fix |
|---|---|---|
| `a01141f` | transformers 5 `apply_chat_template(tokenize=True)` returns a BatchEncoding; vLLM input ids never matched | read `input_ids` (list or mapping) |
| `fa46fe1` | vLLM renders Gemma 4 string messages as content parts and adds a space before `<turn|>` | `--chat-template-content-format string`; server ids then equalled client ids (44/44) |
| `6e2d0ae` | the 0.05-nat centred log-mass parity gate was below bf16 resolution (ulp 0.125 at \|logit\| 16–32) | gate on log-odds of letters with p ≥ 1e-3 within 4 bf16 ulps (user-approved revision, disclosed) |
| `c0b7cf2` | 384 image rows per v2 split carry only a placeholder text; 192 calibration/dev inputs collided | skip `modality=image` rows (1,440 decisions per split) |
| `693d747`, `04f6669` | P1R needed more than 20 minutes; the latency probe sent Score criteria as a map (422) | 40-minute P1R; Jev-shaped Score arrays; 3 unmeasured warm-ups |
| `c2bafe1` | adapter key mapping did not accept `gemma4_unified` | treat it like `gemma4` (`model.language_model.`) |
| `6ed8c9f` | `judge_hard` items in `hard.jsonl` were tiered as judge | explicit tier/split and file name decide first |


The parity gate passed for all four prompt variants after these fixes: identical prompt and canonical token ids,
argmax agreement 1.0, probability max-abs ≤ 0.0125.

## Prompt variants on non-public dev (1,440 decisions, 12B frozen, fitted policy)

| Variant | I | C | Noul CC | Mean input tokens | Local A |
|---|---:|---:|---:|---:|---:|
| min | 64.1 | 85.6 | 70.5 | 231 | 76.37 |
| cygnet | 68.6 | 86.0 | 81.8 | 300 | 76.92 |
| labeled | 64.2 | 85.5 | 70.5 | 233 | 76.38 |
| rules | 63.7 | 84.0 | 73.3 | 284 | 75.11 |

**Gate decision:** cygnet versus min gave ΔI +4.54 [1.84, 7.33] and ΔA +0.56 [−0.46, 2.14]. The higher token
cost lowers the Cost axis, so the composite lower bound does not exceed 0. Under the predeclared rule cygnet was
**not adopted**, and the policy stays on `min`.

## Gated reasoning route

The route was recomputed on the same Linux image; on Windows the router refit differed at the 1e-16 level and failed
fingerprint binding. It was **adopted**: every predeclared gate passed.

- ΔA +0.46 [0.33, 0.65]; ΔI +0.60 [0.002, 1.40]; Choice +0.65, Noul +1.14, Score 0.
- It routed 1.2 % of non-public standard/judge decisions.
- Projected p95 rose from 0.796 s to 0.809 s (+1.6 %, within the 10 % guard).

The router was fitted on calibration only (n 1,198, positives 765).

## Serial latency (raw, before the ×2 + 0.15 s adjustment)

| System | p50 | p95 |
|---|---:|---:|
| non-public dev, direct | 0.044 s | 0.151 s |
| non-public dev, routed | 0.041 s | 0.055 s |
| public diagnostic, min | 0.047 s | 0.607 s |
| public diagnostic, cygnet | 0.049 s | 0.595 s |

## Why Cygnet scored higher than v1: matched 2×2 on public items (diagnostic only)

All cells use the same base revision and the same 231 items. "Native" is v1's own serialization and readout at an
8,192-token context; "Swift" is the letter readout with the `min` prompt.

| Cell | Correct | Hard |
|---|---:|---:|
| frozen, native | 179 | 66 |
| frozen, Swift `min` | 193 | 76 |
| v1 LoRA checkpoint, native | 191 | 71 |
| v1 LoRA adapter, Swift `min` | 193 | 73 |
| Cygnet (published reads) | 203 | 85 |
| historical v1 report (4,096-token budget) | 192 | 72 |

Public accuracy by Swift prompt variant on frozen 12B: min 193, labeled 198, **cygnet 201 (hard 83)**, rules 190.

What this shows, within the resolution of 231 items:

1. **Fine-tuning did not destroy base ability.** Under v1's own format, the v1 checkpoint gains 5.2 points over
   frozen (13 vs 25 discordant, p = 0.073). Under the Swift readout, the adapter is neutral (12 vs 12). This rejects
   the earlier hypothesis and agrees with Codex's caution that it had not been shown.
2. **Format and readout matter for the frozen model:** native to Swift adds 6.1 points (12 vs 26, p = 0.034).
3. **The remaining gap to Cygnet is the prompt.** Swift `min` trails Cygnet by 10 items (2 vs 12, p = 0.013), while
   Swift with the `cygnet` prompt reaches 201 against Cygnet's 203. Two small differences remain: the content-format
   rendering and calibration.
4. The 4,096-token truncation suspicion is ruled out: the current v1 at 8,192 tokens scores 191, against 192
   historically.

## Open decision

On non-public dev, the cygnet prompt raised Intelligence significantly but failed the composite gate because it
adds about 30 % input tokens. On the public diagnostic, it closes most of the gap to Cygnet (+8 items, hard +7).
Our dev split is procedural and has no hard tier, so it may understate hard-tier gains. Using public results to
choose the prompt would break the rule that public items are diagnostic only. Any change of prompt policy is
therefore a user decision, to be made with this caveat recorded.

## Cost

Credit fell from $49.60 to $40.28 during the session. That includes an unrelated instance
(`nuri-fbb34b7-20261004-v3`, $0.844/h) running on the same account throughout. The Swift instances themselves were
roughly 4.6 h at $0.62–0.64/h, plus bandwidth and storage.
