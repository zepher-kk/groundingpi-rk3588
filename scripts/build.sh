#!/bin/bash
# Build the production apps (and optionally the experiments).
#
#   scripts/build.sh                     # apps only
#   scripts/build.sh --experiments       # + probes / batch tools
#   scripts/build.sh --experiments --opencv   # + OpenCV tools (prep/pair/rga)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

EXP=OFF
OPENCV=OFF
for a in "$@"; do
  case "$a" in
    --experiments) EXP=ON ;;
    --opencv) OPENCV=ON ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

cmake -S "$REPO_ROOT" -B "$GPI_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DRKLLM_SDK_DIR="$RKLLM_SDK_DIR" \
  -DRKNN_SDK_DIR="$RKNN_SDK_DIR" \
  -DENABLE_EXPERIMENTS="$EXP" \
  -DENABLE_OPENCV_EXPERIMENTS="$OPENCV"

cmake --build "$GPI_BUILD_DIR" -j"$(nproc)"
echo "built into $GPI_BUILD_DIR"
