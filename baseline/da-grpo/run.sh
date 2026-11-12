#!/usr/bin/env bash
# Single-card O1-style GRPO. The Python trainer handles all settings.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/train.py" "$@"
