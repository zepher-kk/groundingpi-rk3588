#!/bin/bash
# Shared environment for this repository. Source it, or copy to .env.local.
# Nothing here is machine-specific: override any value from your shell.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Where the converted assets live (weights are NOT in this repo).
export GPI_ASSET_DIR="${GPI_ASSET_DIR:-$HOME/gpi-assets}"
export GPI_STUDENT_DIR="${GPI_STUDENT_DIR:-$GPI_ASSET_DIR/student}"
export GPI_MODEL_DIR="${GPI_MODEL_DIR:-$GPI_ASSET_DIR/baseline}"

# Rockchip SDK copies (not redistributed). Example layout:
#   $RKLLM_SDK_DIR/include/rkllm.h, $RKLLM_SDK_DIR/lib/librkllmrt.so
#   $RKNN_SDK_DIR/include/rknn_api.h, $RKNN_SDK_DIR/lib/librknnrt.so
export RKLLM_SDK_DIR="${RKLLM_SDK_DIR:-$HOME/rk-sdk/rkllm}"
export RKNN_SDK_DIR="${RKNN_SDK_DIR:-$HOME/rk-sdk/rknn}"

# Optional: override the runtime search path baked at link time.
# export GPI_RPATH="$RKLLM_SDK_DIR/lib:$RKNN_SDK_DIR/lib"

export GPI_BUILD_DIR="${GPI_BUILD_DIR:-$REPO_ROOT/build}"
