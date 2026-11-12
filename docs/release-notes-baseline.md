# Original GroundingPI-4B baseline (`groundingpi-4b-w8a8`)

MoonViT-27 vision (3 FP16 RKNN stages, 448×448, 256 tokens) + Qwen3-4B **W8A8**, for RK3588
(LubanCat 4, 8 GB). This is the original deployment profile kept as the fallback entry.

## Files

```bash
cat groundingpi-4b-w8a8.rkllm.part-* > groundingpi-4b-w8a8.rkllm
sha256sum -c SHA256SUMS
```

| File | Role |
| --- | --- |
| `groundingpi-4b-w8a8.rkllm` (split) | language model |
| `v2-stage00/09/18.rknn` | vision, three sequential stages |
| `vocab.bin`, `merges.bin`, `embed.bf16` | tokenizer + text embedding table (2560 dims) |

## Run

```bash
scripts/run_baseline.sh --image <jpg> --query "the dog"
scripts/run_baseline.sh --no-llm --image <jpg> --query "the dog"    # vision only
```

## Measured

- Vision 3 stages: ~3.16 s; language: ~3.95 s (TTFT ~2.96 s); request ≈ 7.1 s, plus ~20.5 s init.
- Outputs `4 coords + <|box_end|> + <|im_end|>` (the protocol the student does **not** produce).
- Not evaluated on the 64-request student set; only single-image probes exist.

## License / provenance

Weights derive from the upstream GroundingPI teacher (Qwen3-4B text tower + MoonViT-27 vision).
They are published for reproducibility of this deployment; check upstream terms before reuse.
