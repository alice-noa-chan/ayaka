# Beam clean pilot

This wrapper preserves the verified 2026-10-02 clean archive and original v1
parent. The only recipe change is a recorded 3600 → 5400 second completion
allowance for A100 hardware. Dataset bytes, optimizer settings, the fixed 200
steps and the dev/calibration/test separation remain identical.

Use live account credit and offers with `budget.admit_offer` before allocation.
The observed single A100 80GB SXM quote was $1.518/hour. A three-hour reservation
costs at most $4.56 after upward rounding; $0.30 is reserved for CPU preparation.
These are cost allowances, not a measured A100 runtime guarantee. An account
spend cap is additional protection and does not replace reservation expiry.

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

Create the isolated pool, then reserve exactly the admitted provider/region/offer
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

References: [reserved pools](https://docs.beam.cloud/v2/scaling/pools),
[storage and billing](https://docs.beam.cloud/v2/resources/pricing-and-billing).
