# Offline direct corpus preparation

Use `python -m ayaka.training.direct_corpus` on the v2 experiment branch.
The CLI uses cached, revision- and byte-pinned human training files and native
tokenizer/configuration assets. It cannot load pretrained weights, generate
teacher traces, run an optimizer, allocate a GPU or download missing data.

The example `direct_corpus_settings.json` declares 1,280 train questions:
896 human rehearsal questions and 384 repository-authored verified control
questions. English contributes 1,088 questions (85%). Each reserved split
contains 240 questions. One complete train epoch is 40 batches of 32 rows.
These are selected-corpus counts, not a pass through the entire original
53,093-row human source collection. More epochs require an explicit new plan.

General human rehearsal preserves capabilities. It does not replace natural
policy reasoning data, independently observed teacher improvements or a
matched v1-reasoning versus v2-off quality evaluation. Preparing this control
does not establish that a paid experiment or model promotion is justified.

## 1. Declare settings and configuration

Copy the example settings and use the intended pinned LM-readout configuration
with explicit context limits. The plan binds the entire configuration,
including adapter shape, context limits, model revision and version. The input
encoder, actual tokenizer serialization and native metadata are bound too.

Example configuration creation, using the repository's pinned large model:

```python
import json
from dataclasses import asdict, replace
from pathlib import Path
from ayaka.config import ELECTRA_LARGE

cfg = replace(ELECTRA_LARGE, name="ayaka-v2-direct-control", version=2,
              readout="lm", lora_r=32, lora_alpha=64)
Path("direct-config.json").write_text(json.dumps(asdict(cfg)), encoding="utf-8")
```

## 2. Anchor existing private evaluation inputs

Create an explicit manifest of the original evaluation JSONL files to exclude:

```json
{
  "version": "ayaka-direct-reserved-inputs-1",
  "files": [{"path": "previous-evaluation.jsonl", "sha256": "<file byte SHA256>"}]
}
```

Relative paths resolve against the manifest directory. Every listed file must
contain nonempty valid `Sample` JSONL. Pass the externally trusted SHA256 of
the literal manifest bytes. The plan stores a path-free logical inventory
hash and file digest/counts; private evidence and original IDs are excluded.
Relocating the same files can change the external manifest byte SHA without
changing the logical plan identity.

For an explicitly incomplete CPU control, replace `--reserved-manifest` and
its digest with `--draft-without-prior-reserved`. The draft scope is preserved
in the bundle, preparation report and audit receipt. This flag cannot claim
that an earlier private evaluation inventory was checked.

## 3. Freeze the input plan before selection

```bash
python -m ayaka.training.direct_corpus plan \
  --config direct-config.json \
  --settings docs/experiments/direct_corpus_settings.json \
  --reserved-manifest private-manifest.json \
  --expected-reserved-manifest-sha256 MANIFEST_BYTE_SHA256 \
  --out input-plan.json
```

The command prints two different anchors: `plan_file_sha256` covers exact file
bytes including the terminal newline; `plan_sha256` covers canonical plan
contents and becomes each selected sample's marker. Pin the file anchor
outside the mutable plan file. The input plan contains settings and asset
bindings; verified, excluded and selected counts belong to the result report.

An uploaded native tokenizer/configuration directory can be supplied with
`--native-path` to both commands. No mutable repository/cache fallback is
used for that explicit path. Complete native weights and the deployable
snapshot are separate requirements of the GPU runner.

## 4. Prepare and fully audit before paid allocation

```bash
python -m ayaka.training.direct_corpus prepare \
  --config direct-config.json \
  --plan input-plan.json --expected-plan-file-sha256 PLAN_FILE_BYTE_SHA256 \
  --reserved-manifest private-manifest.json \
  --expected-reserved-manifest-sha256 MANIFEST_BYTE_SHA256 \
  --attention flash_attention_2 --liger \
  --out prepared-direct
```

The default encoding is Swift canonical, labeled, compact, nonthinking.
Alternate supported encoders must be declared consistently in both commands.
Human gold is checked against all original cached source annotations before
selection. Candidate permutations are applied before context checks, every
question in a selected sample is retained, and only typed context overflow
can exclude a whole natural sample. Malformed inputs fail preparation.
Authored/public/private overlap or authored overflow fails rather than
silently changing its declared quota. Transitive prompt/translation/source
aliases are excluded before natural split selection and quota admission.

Outputs are a development-only `bundle/`, separate `bundle-holdout/`, a fully
regenerated `cpu-audit.json`, and a computed `selection.json`. Bind the emitted
bundle, audit and holdout byte anchors externally. Public/private inventories,
input files, raw annotations and native assets are rechecked before output
publication. Tokenizer serialization reuse ends before artifacts are written.

The common planned contract validates prepare, full audit, portable load and
resume. It fixes source sample/type quotas, language question ratios, actual
prepared-row/group digests, derived steps/seed and exactly E visits to every
row. Weighted language replacement, a repeated batch tail, missing questions
or swapped groups fail. Fast loading uses the externally pinned CPU receipt
and exact frozen buffers without reopening raw/public/private source files.

The computed membership digest freezes the CPU-verified selection. It is not
proof that an arbitrary later corpus was selected by replaying the complete
raw-source algorithm, execution attestation or evidence of unseen evaluation.
FA2/Liger here are declared/preflight settings; actual CUDA parity, throughput,
whole-workflow credit admission, real policy teachers and quality measurements
remain separate work. Existing unplanned bundle6/holdout2 controls remain
readable when all corpus-plan markers and extensions are absent.
