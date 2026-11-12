# Producing the assets (deployment pipeline)

This document records how the files in `models/manifest.json` were produced. It assumes you
already have a trained student checkpoint (`projector` + task rows `delta` + LoRA adapter) and
the frozen visual backbone. Everything below runs on **CPU**; no GPU is required for
conversion.

> Paths are examples. Nothing in the code depends on them; point `GPI_STUDENT_DIR` /
> `GPI_MODEL_DIR` at wherever you put the outputs.

## 1. Merge the student into a standard HF model

The training code replaces rows `>= 151669` with the learned task rows (it is a replacement,
not an addition) and merges LoRA into `q/k/v/o`:

```python
W = base_q_proj + (lora_alpha / r) * (B @ A)          # FP32 math, store bf16
embed[:151669]  = base rows
embed[151669:]  = checkpoint['delta']                 # 1001 rows
lm_head         = same replacement (model is tied)
```

Then write a standard `Qwen3ForCausalLM` directory (`config.json` with `vocab_size=152670`,
tokenizer, `generation_config.json`, sharded safetensors). Verify with a reload: the task-row
embeddings must be bit-identical and 4–8 reference prompts must generate the same token ids as
the training-time wrapper.

## 2. Re-serialize the tokenizer (important)

Transformers 5.x writes `tokenizer_config.json` with `extra_special_tokens` as a **list**.
RKLLM 1.3.1's bundled transformers (4.5x) calls `.keys()` on it and fails with

```
AttributeError: 'list' object has no attribute 'keys'
```

Fix by rebuilding the tokenizer in a 4.x environment (`AutoTokenizer.from_pretrained(base)` +
`add_special_tokens({...})` + `save_pretrained(merged)`), then confirm
`load_huggingface` returns 0. Do not patch version strings.

## 3. Convert the language model

```python
from rkllm.api import RKLLM
sdk = RKLLM()
sdk.load_huggingface(model=MERGED_DIR, device='cpu', dtype='float16')
# FP16 (accuracy-preserving, 4.09 GB):
sdk.build(do_quantization=False, target_platform='rk3588', num_npu_core=3, max_context=512)
# or W8A8 (2.37 GB, NOT accurate enough for this model — see docs/known-limitations.md):
# sdk.build(do_quantization=True, quantized_dtype='w8a8', dataset=CALIB_JSON, ...)
sdk.export_rkllm('student_qwen3-1.7b-fp16.rkllm')
```

Calibration (only for W8A8) is plain text `{"input": ..., "target": ...}`; text-only
calibration cannot represent the vision embedding distribution, which is one reason W8A8
degrades this model.

## 4. Export the vision tower

```python
student = Student(frozen_tinyvit, trained_projector).eval()
torch.onnx.export(student, (torch.zeros(1, 3, 384, 384),), 'vision.onnx',
                  input_names=['pixels'], output_names=['tokens'], opset_version=13)
```

Then convert ONNX → RKNN (fp16, no quantization) with `rknn-toolkit2`, `target_platform=rk3588`.
Check torch ↔ ONNX Runtime (should be ~1e-5) and RKNN ↔ torch (cosine ≈ 0.998) before trusting it.

## 5. Board-side helper files

| File | How |
| --- | --- |
| `vocab.bin`, `merges.bin` | dump `tokenizer.json` (`model.vocab` + `added_tokens`, `model.merges`) into the binary layout `gp::Tokenizer` reads: `uint32 count`, then `(uint32 id, uint32 len, bytes)`; merges: `uint32 count`, then `(len1, bytes1, len2, bytes2)` |
| `embed_tokens_151669x2048.bf16` | raw bf16 rows `0..151668` of the merged `embed_tokens.weight` (prompts never use coordinate ids) |
| `bench64.tsv` | `sample_id, jpg, query, width, height, gt..., prompt_ids` |

## 6. Verify

```bash
python3 scripts/verify_assets.py --profile student --sha
scripts/run_student.sh ids "the dog"            # prompt ids, no model needed
scripts/run_student.sh one image.jpg "the dog"  # end-to-end
```

Expected: the tokenizer ids match the Python tokenizer for the same query, and a valid image
produces a box. Any change in model SHA or parse rule is a new baseline version.
