# Known limitations

## Accuracy (student, `gpi3s-student-fp16-v1`)

- **45/64 valid and 45/64 hit (IoU≥0.5)** on the 64-request set under the deployment rule.
  This is a **training-set overfit result**, not generalization.
- **Original protocol: 0/64.** The model does not emit `<|box_end|>` in free generation.
  19 of 64 outputs contain an extra token before `<|im_end|>`; 17 of those are the stray
  `</tool_response>` (id 151666). The parser reports this instead of hiding it.
- Earlier reports quoted "64/64 valid"; that used a loose rule (read the first four
  coordinates and ignore the rest). It is superseded.
- `dev32` (disjoint images): **0/32 hits**. No generalization has been demonstrated.
- Refusal/unanswerable (`<|none_box|>`) was not trained; the parser can recognize the syntax
  but there is no measured refusal accuracy.

## Numerics

- W8A8 for this model is **not usable**: 9/64 valid, 8/64 hit, mean IoU 0.147. The token
  decisions sit on small margins, and int8 activation quantization flips them. FP16 is the
  delivered precision.
- Vision RKNN FP16 vs PyTorch: cosine 0.998, relative L2 ≈ 5.5–6.2%. Two of six probe images
  changed their token sequence because of this, without changing the first four coordinates.

## Runtime and scheduling

- Vision runs on **one NPU core** per image; the graph is sequential, so extra cores only help
  across images. No multi-image vision pipeline is implemented in these apps.
- Vision and language never overlap (the prompt depends on the vision tokens).
- The two profiles cannot co-reside (4.09 GB + 4.84 GB VS 8 GB), so one profile per process.
- `batch=2` improves throughput ~11–12% but pushes single-request completion to 3.86 s;
  it is opt-in only (`docs/batch-experiment.md`).

## Not implemented

- Online queueing / concurrent requests / cross-request image cache.
- Long-run stability, thermal throttling behavior, memory-leak testing.
- RGA or zero-copy preprocessing (CPU path only in the reference configuration).
- Negative/refusal training, larger data, distillation (out of scope for this repository).

## Environment assumptions

- RKLLM 1.3.1 and RKNN 2.5.0; other versions are untested.
- The tokenizer must be re-serialized by a transformers 4.x-era writer; a transformers 5.x
  `tokenizer_config.json` breaks `load_huggingface` (see `docs/deployment.md` §2).
- Board disk: the student assets are ~4.7 GB, the baseline ~5.3 GB; do not plan on both plus
  build output on a small partition.
