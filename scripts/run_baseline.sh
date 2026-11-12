#!/bin/bash
# Run the original GroundingPI-4B profile (fallback / baseline).
#
#   scripts/run_baseline.sh --image <jpg> --query "the dog"
#   scripts/run_baseline.sh --no-llm --image <jpg> --query "the dog"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

BIN="$GPI_BUILD_DIR/groundingpi"
[ -x "$BIN" ] || { echo "groundingpi not built; run scripts/build.sh" >&2; exit 3; }

ASSETS="${GPI_BASE_ASSETS:-$GPI_MODEL_DIR/llm-assets}"
LLM="${GPI_BASE_LLM:-$GPI_MODEL_DIR/groundingpi-4b-w8a8.rkllm}"
MODELS="${GPI_BASE_MODELS:-$GPI_MODEL_DIR/v2-stage00.rknn,$GPI_MODEL_DIR/v2-stage09.rknn,$GPI_MODEL_DIR/v2-stage18.rknn}"

exec "$BIN" --assets "$ASSETS" --llm "$LLM" --models "$MODELS" --prep "${GPI_BASE_PREP:-cpu}" "$@"
