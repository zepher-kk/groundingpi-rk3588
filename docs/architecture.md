# Architecture

One process, one profile, two stages: **vision RKNN → embedding injection → RKLLM**.

```
image bytes ──► preprocess (letterbox + normalize, CPU)
                    │
                    ▼
            vision RKNN (NPU core 0)
                    │  144 × 2048 (student)  |  256 × 2560 (baseline)
                    ▼
        prompt ids (tokenizer) + text embedding table
                    │  replace image_pad slots with vision rows
                    ▼
            RKLLM (INPUT_EMBED, 3 NPU cores)
                    │  greedy, stop at <|im_end|>
                    ▼
        answer parser (deploy-parse-v1) → xyxy in original image
```

## Components (`src/`, `include/`)

| File | Role |
| --- | --- |
| `tokenizer.cc/.h` | byte-level BPE matching the Qwen tokenizer; reads binary `vocab.bin` / `merges.bin` |
| `embed.cc/.h` | row-addressable bf16 embedding table (`embed_tokens_*.bf16`) |
| `vision.cc/.h` | RKNN stage wrapper: FP16 input, FP32 output, optional shared buffers between stages |
| `llm.cc/.h` | RKLLM handle: init, per-request KV clear, blocking `rkllm_run` with embedding input |
| `protocol.cc/.h` | baseline prompt (256 pad slots) + GAM box decoding |
| `preprocess.cc/.h` | letterbox to a square canvas, normalize (baseline: `v*2/255-1`) |
| `npy.cc/.h` | `.npy` reader for pixel inputs |
| `common.cc/.h` | file IO, JSON escaping, timers |

The student app carries its own prompt builder and the `deploy-parse-v1` parser, because the
3s protocol differs (144 slots, 384 canvas, coordinate ids 151670–152669, PIL-compatible
bicubic resize with ImageNet normalization).

## Apps

| App | Notes |
| --- | --- |
| `apps/student/main.cc` (`student_v1`) | modes: `one`, `bench`, `pix`, `emb`, `pre`, `vraw`, `ids`, `ptest`; strict per-request accounting, exit codes 0/3/4/5/6 |
| `apps/baseline/main.cc` (`groundingpi`) | `--image/--pixels/--batch`, `--assets/--llm/--models`, dump modes; preserves the original behavior |

## Scheduling facts

- Vision always runs on **NPU core 0**; the language model is built with 3 cores. A single
  vision graph cannot be split across cores (sequential graph), so multi-core only helps
  **across images**, not within one request.
- Vision and language never overlap: the 144/256 vision rows must exist before the prompt
  can be submitted.
- The student keeps the vision context resident for the whole process; the baseline releases
  it before the language stage by default (`--keep-vision` keeps it).
- `n_batch` is an init-time parameter; see `docs/batch-experiment.md`.

## Why the two profiles cannot share a process

`student` needs a 4.09 GB FP16 RKLLM, `baseline` a 4.84 GB W8A8 RKLLM. With 8 GB RAM plus the
vision context, they do not co-reside. `scripts/run_*.sh` therefore start one profile per
process. A unified launcher can dispatch by profile name, but it still runs one at a time.
