#!/usr/bin/env bash
# DA-SFT foreground launcher. Run from any working directory.
set -euo pipefail
exec python3 -u "$(dirname "$0")/train.py" "$@"
