#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source scripts/sim.sh >/dev/null 2>&1

python -u -m excavator_policy.analyze_predictions "$@"
