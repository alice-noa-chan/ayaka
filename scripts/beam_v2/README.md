# Beam clean pilot

This wrapper preserves the verified 2026-10-02 clean archive and original v1
parent. The only recipe change is a recorded 3600 → 5400 second completion
allowance. Dataset bytes, optimizer settings, the fixed 200
steps and the dev/calibration/test separation remain identical.

The live A100 80GB SXM offer was $1.518/hour, but reservation was rejected:
managed-compute-eligible credit was **$0**, with a **$25 minimum**. Existing
serverless credit does not apply. Do not confuse a quote with credit eligibility.

Serverless RTX5090 was ready. GPU-attached rates checked on 2026-10-02 are
$0.000303/sec for GPU, $0.000105/sec per CPU core and $0.0000055/sec per GiB RAM.
Two cores and 32GiB cost $2.4804/hour. A 9700-second task costs at most $6.69
rounded up, plus a $0.10 margin. Completed CPU preparation used $0.056027,
leaving $6.976841 credit. The account cap is $6.85, within available credit.
These are upper cost allowances, not a measured RTX5090 runtime guarantee.

Validate live inventory and remaining account allowance with
`budget.admit_serverless` before invoking `job.py serverless`. This explicit
route uses only RTX5090, two cores and 32GiB, a 9700-second backend timeout,
zero retries and marketplace disabled. Check 32GB/BF16 hardware and run fresh
production loss/gradient, stress and complete-schedule profiling before updates.
Do not lower precision or truncate the fixed 200-step schedule.

Upload the immutable archive with `python transfer.py cp` to the private
`ayaka-clean-20261002` volume. From **this directory**, use the already installed
Beam-client Python interpreter to run `job.py build`, then `job.py prepare`.
The transfer adapter fixes the installed SDK's Windows remote-path separator
without changing native local paths or credentials. The CPU task verifies the archive and every extracted file and runs the bundled
interpreter's plan. CPU and GPU stage independent archive ranges concurrently to
local disk and verify all checksums there. Preparation requires 60GiB free local
disk, avoiding slow small-file operations on the object-store volume. Tasks run
headless so a lost client log connection cannot cancel otherwise healthy work;
backend timeouts and reservation expiry remain mandatory. Poll owned task status
and durable receipts rather than resubmitting. The immutable kit manifest is checked before applying the separately
recorded time allowance. No GPU should be reserved until preparation succeeds.
The first non-headless CPU task was cancelled after its client log connection
failed; it allocated no GPU. The corrected CPU task completed in 499.67s.

The reserved route is allowed only with confirmed managed credit through
`budget.admit_offer`. Create the isolated pool, then reserve exactly the admitted provider/region/offer
with `beam pool scale`, `--ttl 3h` and `--max-spend 4.56`. Never extend it, enable
automatic retries, or use a serverless fallback. `job.py pilot --reservation-expires-at EPOCH` runs only on that
pool. A100 80GB/BF16 is checked before model work. The complete schedule requires
fresh production profiling before any optimizer update; old timing is not reused.
Read the expiry from the actual reservation. The worker reserves ten minutes
before it for artifact delivery and refuses insufficient complete-run envelopes.
Durable phase files remain readable when the SDK's client log stream disconnects.

Results, including failures, are compressed with zstd level 3, checksum verified
on the persistent volume, and described by a receipt. Download and verify the
receipt and every archive member before releasing this pool's machine. Verify
checkpoint tensors are finite locally. Preserve diagnostics; an incomplete run
does not authorize promotion or full training. Release early when receipt is
complete; the three-hour backend expiry remains a backup for client failure.
For serverless execution, confirm the owned task/container stops after delivery;
there is no reserved machine to keep billing. Never top up or retry automatically.

References: [pricing](https://www.beam.cloud/pricing),
[reserved pools](https://docs.beam.cloud/v2/scaling/pools),
[storage and billing](https://docs.beam.cloud/v2/resources/pricing-and-billing).
