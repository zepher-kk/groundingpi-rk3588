# GroundingPI on RK3588

C++ inference stack for running GroundingPI-style grounding models on a Rockchip RK3588 board
(LubanCat 4, 8 GB). Two profiles share one build and one entry:

| Profile | Vision | Language | Measured single request |
| --- | --- | --- | --- |
| `student` (3s student) | TinyViT-11M + trained projector, FP16, 384², 144 tokens | Qwen3-1.7B **FP16** | **2.26 s** (P50), 2.43 s (P95) |
| `baseline` (original) | MoonViT-27, 3 × FP16 RKNN stages, 448², 256 tokens | Qwen3-4B **W8A8** | ≈ 6.8–7.1 s |

Both feed vision embeddings into the language model (`RKLLM_INPUT_EMBED`), one request at a time,
KV cleared per request. The two language models cannot be co-resident in 8 GB
(4.09 GB + 4.84 GB), so you run **one profile per process**.

> The 3s student is an **experiment baseline**, trained on 64 samples. On that set it reaches
> **45/64 valid and 45/64 hit at IoU≥0.5** under the documented strict parser, mean IoU 0.9937
> on valid outputs. It does **not** generalize: `dev32` is 0/32, and refusals (`none`) are not
> trained. Do not read these numbers as product accuracy.

## Repository layout

```
groundingpi-rk3588/
├── CMakeLists.txt          # production apps by default, experiments opt-in
├── src/ include/           # shared inference code (tokenizer, embed table, vision, llm, protocol)
├── apps/
│   ├── baseline/main.cc    # original GroundingPI-4B entry
│   └── student/main.cc     # 3s student entry (one | bench | pix | emb | pre | vraw | ids | ptest)
├── configs/                # profile templates (paths come from the environment)
├── scripts/                # build.sh, run_student.sh, run_baseline.sh, verify_assets.py
├── experiments/            # probes + batch tools (not built unless asked)
├── models/manifest.json    # expected asset names, sizes, SHA256 (no weights)
├── docs/                   # architecture, deployment, performance, protocol, limitations
└── NOTICE.md               # provenance / licensing
```

## Requirements

- RK3588 board (aarch64), Linux, g++ ≥ 11, CMake ≥ 3.16
- **RKLLM runtime 1.3.1** and **RKNN runtime 2.5.0** from the Rockchip SDK (not bundled)
- **OpenCV** (core, imgproc, imgcodecs): both apps decode JPEGs with it
- **librga** headers (`im2d.hpp`): needed by the baseline app's preprocessing path
- Experiments with `--opencv` reuse the same OpenCV; the RGA probes need librga

## Build

```bash
export RKLLM_SDK_DIR=$HOME/rk-sdk/rkllm      # include/rkllm.h + lib/librkllmrt.so
export RKNN_SDK_DIR=$HOME/rk-sdk/rknn        # include/rknn_api.h + lib/librknnrt.so
scripts/build.sh                             # apps only
scripts/build.sh --experiments --opencv      # + probes and batch tools
```

Nothing is hard-coded to a machine path: SDK locations come from CMake variables, model
locations from the environment (see `scripts/env.sh`).

## Assets

Weights are not in this repository. Put your converted assets somewhere and point the
environment at them:

```bash
export GPI_ASSET_DIR=$HOME/gpi-assets
export GPI_STUDENT_DIR=$GPI_ASSET_DIR/student     # student profile files
export GPI_MODEL_DIR=$GPI_ASSET_DIR/baseline      # original GroundingPI files

python3 scripts/verify_assets.py --profile student --sha
python3 scripts/verify_assets.py --profile baseline
```

`models/manifest.json` lists the exact filenames, sizes and SHA256 values.
How to produce them from a trained checkpoint (merge LoRA, replace task rows, export the
vision ONNX → RKNN, build the W8A8/FP16 RKLLM, export tokenizer bins and the embedding table)
is in `docs/deployment.md`.

## Run

```bash
# 3s student, single image + prompt
scripts/run_student.sh one /path/to/image.jpg "the dog on the left"

# 3s student, batch regression (one line per request)
scripts/run_student.sh bench manifests/bench64.tsv /tmp/out.tsv

# original GroundingPI-4B
scripts/run_baseline.sh --image /path/to/image.jpg --query "the dog"
scripts/run_baseline.sh --no-llm --image /path/to/image.jpg --query "the dog"   # vision only
```

Output fields, statuses and exit codes are documented in `docs/protocol.md`.

## Results (see docs/performance.md)

| Metric | Student (FP16) | Baseline (4B W8A8) |
| --- | ---: | ---: |
| Vision | 90.5 ms | 3157 ms (3 stages) |
| Language TTFT | 1469 ms | 2961 ms |
| Request P50 / P95 | 2262 / 2428 ms | ≈ 7.1 s (single sample) |
| Peak RSS | 3.82 GB | ≈ 5.5 GB |
| Valid / hit (64 train-set requests) | **45/64 / 45/64** | not measured on this set |

Optional `batch=2` (same model, one `rkllm_run` with two embedding inputs) improves language
throughput by ~11–12% with identical tokens, but single-request completion grows to ~3.86 s.
It is **not** the default; see `docs/batch-experiment.md`.

## Known limitations

See `docs/known-limitations.md`. Short version: 45/64 is a training-set overfit score, the
student never emits `<|box_end|>` (and occasionally emits a stray `</tool_response>` token),
refusal is untrained, W8A8 loses accuracy for this model, and there is no online queueing or
concurrency support.

## License

No license applied yet; SDK binaries and weights are not distributed here. Read `NOTICE.md`
before publishing or reusing.
