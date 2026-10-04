# Swift readout for the v2 candidate — plan (2026-10-04)

Goal: JevBench v1.5 headline A #1. Swift is a readout/decision-policy component of the v2 candidate on
`ayaka-v2-experiments`, not a separate release line (the file name is kept because it is referenced).
Published v1 weights stay frozen.

## Why change direction

- Live board v1.5.6 (snapshot `runs/jevbench-audit-20261004/live-v1.5.6-20261004.json`):
  #1 Cygnet 73.70 = **frozen** `google/gemma-4-12B-it`, options shown as letters, the model's own probability
  for each letter read at one answer position, one temperature (T = 3.4). Axes: I 71.1, C 87.0, S 91.0, Cost 56.4.
- On the v1.2 public set that frozen readout scored hard 85/111; our trained v1 large scored 72/111 single pass.
  This is an observed underperformance across different systems (prompt, readout and runtime differ); whether
  fine-tuning caused it stays unconfirmed until a matched readout comparison (same base/tokenizer revision,
  prompt and letter readout, with and without the v1 adapter). The v2 continuation's dev regression is largely
  explained by more Noul abstentions under unchanged thresholds
  ([V2_CONTINUATION_AUDIT_2026-10-04](V2_CONTINUATION_AUDIT_2026-10-04.md)). Swift therefore starts from the frozen readout
  (approach credited to [NInfer](https://github.com/igorls/ninfer) and
  [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe)) and adds a decision policy.

## The decision policy

1. **Per-primitive temperatures** (Choice / Noul / Score), fitted by NLL on non-public data only
   (v2 calibration split; never JevBench public or sealed items).
2. **Noul commit band.** After temperature, probabilities strictly inside `(0.2, 0.8)` move to
   `0.199 / 0.801` only when `|P(yes) − 0.5| >= commit_margin`; `None` disables commitment and remaining
   in-band values abstain. The margin and nearby Noul temperatures are fitted jointly by local composite A
   on non-public calibration reads with fixed Speed/Cost axes, because the
   [saved-policy probe](V2_SAVED_POLICY_PROBE_2026-10-04.md) found reader-dependent calibration costs.
   Report a case-bootstrap CI and runner-up, preferring no commitment within 0.25 points of the best and
   then the qualifying temperature closest to its NLL fit; assess the selected policy on independent dev reads.
3. **Measured prompt.** Keep `min` (the original one-line system prompt) as the default until
   non-public dev measurements select a variant. Compare it with `cygnet`, the MIT scaffold/layout
   from `blockbrain-ai/cygnet-recipe/shim/cygnet_shim.py`, and our short `rules` scaffold. The latter
   covers governing definitions, exceptions/amendments, effective dates, arithmetic and chained
   inference, without benchmark names or item text. Its CPU whitespace-token overhead must stay
   within 80 tokens; actual input-token cost is measured from reads.

## Expected composite (estimate, not a measurement)

With Cygnet's other axes unchanged and Noul CC +16 on both splits, I ≈ 71.1 + 5.3 ≈ 76.4 and
A ≈ 75.3. A bootstrap #1 claim needs about 1.5–2 points of margin over #2, so further Intelligence work
is still wanted (prompt variants evaluated on our dev split only).

## Steps

1. `ayaka/swift/`: prompt, letter readers (vLLM / HF), policy, TypeSafe server, collect, fit, local v1.5 scorer.
2. Base-model comparison inside the same job. The cost axis is gated below 50, so bases priced above about
   12B-class (Qwen 27B / Flash-Next ≈ 42, 26B-A4B ≈ 49, 9B ≈ 45) are out. Cheaper bases are measured alongside
   12B: `gemma-4-E4B-it` (cost ≈ 63–68). With the commit policy a 4B-class base needs I ≈ 63–66 to match
   12B; the choice is made on dev reads plus the composite estimate. Frozen `Qwen3.5-4B` read the same way is
   already on the board as SemIf (I 51.3; 187/231 public), so it is not re-measured: +5 from the commit
   policy would still leave it far short.
   Reference: [Strands Decider](https://github.com/strands-labs/strands-decider) (Apache-2.0, 2B, 167/231)
   independently found that fine-tuning an instruct torso loses compositional skills (their v8) and that
   frozen readouts of a stronger torso beat training a weaker one. Its research log also shows that an
   internal held-out set did not rank models and that 231 public items cannot resolve differences below
   about 4 points. So base and policy choices need a dev set that tracks JevBench, plus a paired CI.
3. One short GPU job: collect T = 1 reads for the v2 calibration/dev splits, Cygnet's MIT calibration items
   and the 231 public items, once per `min,cygnet,rules` for each of the two Gemma models. Reuse one
   pinned vLLM server per model and its prefix cache, finish all bulk reads, then run 200 serial Swift
   HTTP latency requests per variant. Cost estimate goes to the user first. Default dry-run counts:
   4,120 bulk decisions / 4,888 model reads per model/variant; 24,720 bulk decisions / 29,328 model reads
   across two models and three variants, plus 1,200 serial requests (30,528 total model reads).
4. `scripts/swift/select_variant.py` fits each variant's policy on v2 calibration only and compares the
   SAME non-public v2 dev + Cygnet generated items (2,065 decisions per variant). It refuses public
   reads, missing/duplicate pairs, changed gold/type/case metadata, overlapping calibration/dev IDs
   and mixed model revisions. Report I, C and local composite A: Speed from that variant's complete
   serial probe if provided (otherwise explicitly estimated at 91), Cost from mean input tokens only
   at `--usd-in-per-m` (default 0.0403). Resample paired whole cases within primitive-coverage strata
   (B=2,000, seed=15) for 95% CIs of A/I deltas versus `min`; policies and Speed remain fixed, input
   cost is recomputed per draw. Pick highest A; if its A CI versus the runner-up includes 0, prefer
   the one with fewer mean input tokens. Equal tokens retain higher A. This is a local estimate,
   conditional on policy fitting and Speed, not a board claim. Freeze the choice before opening
   public accuracy diagnostics; public latency samples supply timing only. See the
   [runner/selection commands](../../scripts/swift/README.md). Policy JSON records its variant;
   serving with a different `--prompt-variant` is refused unless `--force-variant` is supplied.
5. Package (pinned weights revision, vLLM version, policy.json, server) and file the JevBench request.
