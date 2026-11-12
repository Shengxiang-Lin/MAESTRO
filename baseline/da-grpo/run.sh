#!/usr/bin/env bash
# New Unsloth + TRL GRPOTrainer entrypoint; runs in the foreground.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "${HERE}/train.py" "$@"
