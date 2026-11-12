# Measured performance

Board: LubanCat 4 (RK3588, 8 GB), rkllm-runtime 1.3.1, rknn 2.5.0.
All numbers are from this repository's own runs; the 64-request set is a **training subset**.

## Student profile (`gpi3s-student-fp16-v1`)

64 requests, one process, vision re-run per request, KV cleared per request:

| Stage | P50 | P95 | max |
| --- | ---: | ---: | ---: |
| JPEG decode | 4.7 ms | 6.5 ms | 7.1 ms |
| letterbox + normalize | 7.9 ms | 9.3 ms | 11.3 ms |
| vision RKNN (144×2048) | 90.5 ms | 90.6 ms | 90.7 ms |
| language TTFT | 1468.8 ms | 1499.7 ms | 1612.7 ms |
| language total | 2150.0 ms | 2313.3 ms | 2442.2 ms |
| **request total** | **2261.7 ms** | **2427.6 ms** | 2550.7 ms |

Init 5.2 s (single, not part of the request), peak RSS 3.82 GB, 64 requests in 154 s wall.

Quality on the same set (deployment rule): **45/64 valid, 45/64 hit at IoU≥0.5**, mean IoU
0.9937 over valid outputs (0.6987 with invalid counted as 0). Original strict protocol: 0/64.

## Baseline profile (`groundingpi-4b-w8a8`)

Single request, measured once on the same board:

| Stage | Time |
| --- | ---: |
| vision (3 RKNN stages, 448²) | 3157 ms (1019 / 1016 / 1040) |
| language TTFT | 2961 ms |
| language total | 3948 ms |
| request (excluding 20.5 s init) | ≈ 7.1 s |

Historical note in the project: about 6.84 s per request with the same assets; the 4B language
stage alone is ~3.9 s per request with cleared KV (a 1.17 s figure that appears in older notes
is a warm, repeated-prompt measurement and is not comparable).

The baseline was **not** evaluated on the 64-request set; only single-image probes exist.

## Optional batch=2 (language stage)

| Metric | batch=1 | batch=2 |
| --- | ---: | ---: |
| 12 requests, total | 25.8–26.1 s | 23.1–23.4 s (**+11.3–12.1%**) |
| TTFT | 1462 ms | 3000 ms |
| single request completion | 2133 ms | 3864 ms |
| amortized per request | 2133 ms | 1921 ms |
| peak RSS | 3.629 GB | 3.632 GB |

Tokens were identical to batch=1 in every comparison (including swapped slots and mixed
lengths). Because single-request completion exceeds 3 s, batch=2 stays an opt-in experiment.

## What these numbers are not

- Not generalization accuracy: the 64 requests are training images.
- Not a throughput guarantee for online traffic: there is no queue, no concurrency, no
  cross-request image cache.
- Not a 3 s claim for arbitrary images; it is a regression set result (P95 2.43 s).
