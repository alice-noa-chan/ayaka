#!/bin/bash
# The unchecked runner cannot produce receipts accepted by the checked comparator.
set -euo pipefail
echo "This job is retired. Prepare pinned inputs with scripts/direct_v2/prepare_matched.py" >&2
echo "and use scripts/direct_v2/matched_job.sh with its CPU admission checks." >&2
exit 2
