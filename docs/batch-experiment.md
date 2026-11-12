# Experiment: native batch=2 for the language stage

Status: **verified, not enabled by default**. Tools live in `experiments/` and are not built
unless you pass `--experiments`.

## Why

Single-request latency is dominated by the language model (2.15 s of a 2.26 s request). RKLLM
1.3.1 supports `n_batch > 1`, which processes several independent embedding requests in one
`rkllm_run`. If it worked, offline batches could be ~10% faster.

## API contract (verified on this board)

- `param.extend_param.n_batch = N` at **init** time only.
- Input is an **array**: `RKLLMInput in[N]; rkllm_run(h, &in[0], ...)`.
- The callback receives an **array** `RKLLMResult result[N]`; slot index = array index.
  There is no batch-id field, so never infer ownership from callback order.
- A slot is finished when its `token_id < 0`; the run ends when all slots finish.
- `rkllm_clear_kv_cache(h, 0, nullptr, nullptr)` covers all slots.
- Equal and **unequal** embedding lengths both work (tested 190+184).

## Result

12 requests (6 pairs, equal-length, 4 known-anomalous samples):

| Round | batch=1 | batch=2 | gain |
| --- | ---: | ---: | ---: |
| 1 | 26.06 s | 23.38 s | +11.48% |
| 2 | 25.99 s | 23.36 s | +11.26% |
| 3 | 25.83 s | 23.05 s | +12.05% |

- Tokens identical to batch=1 in **all** comparisons; swapping slots keeps results attached to
  the sample; re-running after a KV clear is identical.
- TTFT 1.46 s → 3.00 s; single request completes in ~3.86 s (it waits for its partner).
- Peak RSS +3.6 MB.
- Full pipeline (vision + language, two images per batch): +11.9% and +14.9%, same tokens.

## Why it is not the default

The 3 s single-request bound is part of this baseline. batch=2 trades latency for throughput,
so the default path stays `batch=1`; use batch=2 only for offline batches where every request
is already queued.

## Run it

```bash
scripts/build.sh --experiments --opencv
D=$GPI_STUDENT_DIR
B=$GPI_BUILD_DIR
# 1) embeddings for a fixed list (vision once per request)
$B/student_batch_prep "$D" "$D/llm-batch/manifest12.tsv" "$D/llm-batch/emb"
# 2) language comparison
$B/student_batch_probe "$D/llm-batch/emb" s "$D/llm-batch/order12.tsv" /tmp/s.tsv "$D/student_qwen3-1.7b-fp16.rkllm"
$B/student_batch_probe "$D/llm-batch/emb" b "$D/llm-batch/order12.tsv" /tmp/b.tsv "$D/student_qwen3-1.7b-fp16.rkllm" 2
# 3) full pipeline for one pair
$B/student_batch_pair "$D" "$D/student_qwen3-1.7b-fp16.rkllm" \
   img_a.jpg "the dog on the left" img_b.jpg "the dog on the right" 2
```

`experiments/` also contains older probes (`protocol_probe`, `coexist_probe`, `pipeline_bench`,
`resident_service`) kept for reproduction; they may contain example paths from the authoring
machine and are not part of the supported entry points.
