# 3s student FP16 baseline (`gpi3s-student-fp16-v1`)

Language Qwen3-1.7B **FP16** + vision TinyViT-11M with the trained projector, for RK3588
(LubanCat 4, 8 GB).

## Files

Splits must be concatenated before use:

```bash
cat student_qwen3-1.7b-fp16.rkllm.part-* > student_qwen3-1.7b-fp16.rkllm
sha256sum -c SHA256SUMS
```

## Measured (64-request training subset, deployment parser)

- Valid 45/64, hit at IoU≥0.5 45/64, mean IoU 0.9937 on valid outputs.
- Original strict protocol: 0/64 (the model does not emit `<|box_end|>`; 17 outputs contain a
  stray `</tool_response>` token).
- Latency: P50 2.26 s, P95 2.43 s (vision 90 ms, language 2.15 s), peak RSS 3.82 GB.

## Not included / not claimed

- Generalization: `dev32` is 0/32. This is an experiment baseline, not a product.
- Refusal (`none`) is untrained.
- W8A8 of the same model is rejected (9/64 valid) — do not substitute it.
- Weights are provided as-is; check upstream licenses (Qwen3, TinyViT) before redistribution.

Use the repository's `scripts/verify_assets.py` and `docs/deployment.md` for setup.
