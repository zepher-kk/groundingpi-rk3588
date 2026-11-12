# Provenance and licensing notes

This repository contains only **source code, configuration and documentation** produced for the
GroundingPI RK3588 project. **No model weights, SDK binaries or SDK headers are included.**
Weights are published separately as GitHub Release assets (see `docs/release-assets.md`).

## License of this repository's own code

MIT (see `LICENSE`), © 2026 zepher-kk. This covers only the files in this repository:
`src/`, `include/`, `apps/`, `experiments/`, `scripts/`, `configs/`, `docs/`.

If you prefer an explicit patent grant instead of MIT, Apache-2.0 is a drop-in replacement for
this code base; replace `LICENSE` and the "License" section of `README.md`.

## What is deliberately not covered / not redistributed

- **RKLLM / RKNN SDK** (`rkllm.h`, `rknn_api.h`, `librkllmrt.so`, `librknnrt.so`) are Rockchip
  proprietary (`rknn_api.h` states "This is proprietary information owned by Rockchip").
  Obtain them from the official SDK; point `RKLLM_SDK_DIR` / `RKNN_SDK_DIR` at them.
  The MIT license above does **not** apply to them, and they are not shipped here.
- **Model weights** (`*.rkllm`, `*.rknn`, embedding tables, tokenizer binaries) are governed by
  their own upstream terms. The student's language base is Qwen3 (Alibaba Qwen license), the
  visual backbone derives from TinyViT (Microsoft, MIT for the reference implementation), and
  the trained adapter/projector/task rows come from this project. Check upstream terms before
  redistributing converted weights.
- Historical debug scripts, machine logs, board diagnostics and anything containing credentials
  remain private and are not part of this repository.

## Data and results

The evaluation numbers in `docs/performance.md` come from a 64-request set that is a **subset of
the training samples**. They are deployment-regression numbers, not generalization accuracy.
