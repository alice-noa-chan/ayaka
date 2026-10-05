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

## Prompt variants on non-public dev (1,312 single-pass decisions, 12B frozen, fitted policy)

Values from the stored `variant_selection.json`. The 128 grouped MASSIVE rows are excluded, as in the gate.

| Variant | I | C | Mean input tokens | Cost | Local A |
|---|---:|---:|---:|---:|---:|
| min | 64.41 | 85.34 | 231 | 70.93 | 76.44 |
| cygnet | 69.21 | 86.64 | 300 | 67.52 | 77.24 |
| labeled | 64.61 | 84.95 | 233 | 70.82 | 76.40 |
| rules | 63.35 | 83.70 | 284 | 68.24 | 74.94 |

**Gate decision:** the stored `adoption.json` (variant lever) gives cygnet versus min:
- ΔA +0.803, 95% CI [−0.240, 2.243]
- ΔI +4.80, 95% CI [1.99, 7.63]

The selection report's own bootstrap agrees: ΔA +0.800, CI [−0.243, 2.241]. The higher token cost lowers the Cost
axis, so the composite lower bound does not exceed 0. Under the predeclared rule cygnet was **not adopted**, and
the policy stays on `min`.

*Correction (2026-10-05):* an earlier version of this table reported 1,440 decisions, ΔI +4.54 [1.84, 7.33] and
ΔA +0.56 [−0.46, 2.14]. Those figures came from a local recomputation that included the 128 grouped rows. That
script was not kept, and the figures were not a stored gate artifact. The decision is the same either way.

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

1. **No support for the earlier "fine-tuning destroyed base ability" hypothesis in this cohort.** Under v1's own
   format, the whole v1 checkpoint (adapter, pointer head, gates and temperatures together) scores 5.2 points above
   frozen (13 vs 25 discordant, p = 0.073). That is not significant, so it does not establish a general
   fine-tuning gain. Under the Swift readout, the adapter alone is neutral (12 vs 12). Codex had cautioned that the
   hypothesis was never shown.
2. **The native-versus-Swift factor matters for the frozen model:** it adds 6.1 points (12 vs 26, p = 0.034). This
   factor combines input serialization and readout and cannot separate them.
3. **Most of the remaining gap to Cygnet follows the prompt.** Swift `min` trails Cygnet by 10 items (2 vs 12,
   p = 0.013). Swift with the `cygnet` prompt reaches 201 against Cygnet's 203. Content-format rendering and
   calibration still differ.
4. **No large context-recovery gain was observed.** The current v1 at an 8,192-token context scores 191; the
   historical report, run with a 4,096-token budget, scores 192. The two runs used different implementations, so
   this is not a context-only ablation, and truncation is not ruled out for every task.

**Routing actually adopted.** The adopted router sent 16 of 1,312 dev decisions to reasoning, all from the
verified procedural source. It routed none of the HelpSteer2 (natural), CommonsenseQA or repository-authored rows.
Codex's paired diagnostic shows that raw worked steps worsen the natural HelpSteer2 rows
(`.dev/codex-actual-paired-source-diagnostic-20261005.json`). The adopted policy therefore avoids that regression on
this dev cohort. It is a selective serving gain, not evidence for distilling reasoned outputs on natural documents.

## Open decision

On non-public dev, the cygnet prompt raised Intelligence significantly but failed the composite gate because it
adds about 30 % input tokens. On the public diagnostic, it closes most of the gap to Cygnet (+8 items, hard +7).
Our dev split is procedural and has no hard tier, so it may understate hard-tier gains. Using public results to
choose the prompt would break the rule that public items are diagnostic only. Any change of prompt policy is
therefore a user decision, to be made with this caveat recorded.

## Cost

The Swift instances ran about 4.6 GPU-hours at $0.62–0.64/h, about $3. Bandwidth (about 70 GB of model and image
downloads at $0.004–0.01/GB) and storage add well under $1. The account balance is not a direct measure of this
work: another project's instance ran on the same account during the session.
