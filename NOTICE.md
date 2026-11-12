# Provenance and licensing notes

This repository contains only **source code, configuration and documentation** produced for the
GroundingPI RK3588 project. **No model weights, SDK binaries or SDK headers are included.**

## What is in here

| Path | Origin | Notes |
| --- | --- | --- |
| `src/`, `include/`, `apps/` | this project | inference plumbing, C++ entries for the two profiles |
| `experiments/` | this project | probes and batch experiments; **not built by default** |
| `scripts/`, `configs/`, `models/manifest.json` | this project | build/run helpers, templates, asset identity |
| `docs/` | this project | architecture, deployment, performance, limitations |

## What is deliberately not in here

- **RKLLM / RKNN SDK** (`rkllm.h`, `rknn_api.h`, `librkllmrt.so`, `librknnrt.so`).
  These are Rockchip proprietary files (`rknn_api.h` states "This is proprietary information
  owned by Rockchip"). Obtain them from the official SDK for your runtime version and point
  `RKLLM_SDK_DIR` / `RKNN_SDK_DIR` at them. Do not redistribute them from here.
- **Model weights** (`*.rkllm`, `*.rknn`, embedding tables, tokenizer binaries). They are
  large and are governed by their own upstream terms (Qwen3 model license, TinyViT/MoonViT
  weights, plus whatever your own training adds). `models/manifest.json` records identity
  (size/SHA256) so a local copy can be verified.
- Historical debug scripts, machine logs, board diagnostics and anything containing
  credentials. They stay in the private working directory.

## Licensing of this repository's own code

No open-source license has been applied yet. Until one is added, the code is
"all rights reserved" by default. Before publishing:

1. Decide the license for **this project's own code** only.
2. Keep SDK headers/binaries and weights out of the repository (as above), so no
   third-party license is implied over them.
3. If you vendor any upstream code later (e.g. TinyViT reference implementation),
   add per-file attribution and keep its original license text.

## Data and results

The evaluation numbers in `docs/performance.md` come from a 64-request set that is a
**subset of the training samples**. They are deployment-regression numbers, not
generalization accuracy.
