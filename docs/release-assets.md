# Publishing weights as GitHub Release assets

Weights are not in git. GitHub release assets have a **2 GiB per-file** limit, so the two
language models must be split. This page defines the naming, splitting and verification
convention, plus the exact commands.

## What to publish

### Release `student-fp16-v1` (recommended first)

| Asset | Size | Notes |
| --- | ---: | --- |
| `student_qwen3-1.7b-fp16.rkllm.part-aa` | ~1.9 GB | split, `student-llm.sha256` for the whole file |
| `student_qwen3-1.7b-fp16.rkllm.part-ab` | ~1.9 GB | |
| `student_qwen3-1.7b-fp16.rkllm.part-ac` | ~0.3 GB | |
| `student_vision_trained_fp16.rknn` | 48 MB | single file |
| `embed_tokens_151669x2048.bf16.part-aa` | ~0.6 GB | |
| `vocab.bin` / `merges.bin` | 2.5 MB each | |
| `SHA256SUMS` | small | checksums for every original (unsplit) file |

### Release `baseline-4b-w8a8` (optional)

`groundingpi-4b-w8a8.rkllm` split into 3 parts, `v2-stage00/09/18.rknn`,
`llm-assets/{vocab.bin,merges.bin,embed.bf16}`, `SHA256SUMS`.

## Packaging script

```bash
scripts/package_release_assets.sh <asset-dir> <out-dir> [--split-mb 1900]
```

It copies small files, splits files above the limit into `.part-aa`, `.part-ab`, …,
and writes `SHA256SUMS` containing the checksum of the **original** files (not the parts).
Run it where the weights live (the board, or a copy on disk).

## Publish

```bash
# from a machine with the packaged files and gh authenticated
gh release create student-fp16-v1 \
  --repo zepher-kk/groundingpi-rk3588 \
  --title "3s student FP16 baseline (gpi3s-student-fp16-v1)" \
  --notes-file docs/release-notes-student.md \
  out/*.part-* out/*.rknn out/vocab.bin out/merges.bin out/SHA256SUMS
```

## Consume

```bash
mkdir -p ~/gpi-assets/student && cd ~/gpi-assets/student
gh release download student-fp16-v1 --repo zepher-kk/groundingpi-rk3588
cat student_qwen3-1.7b-fp16.rkllm.part-* > student_qwen3-1.7b-fp16.rkllm
sha256sum -c SHA256SUMS
export GPI_STUDENT_DIR=$PWD
scripts/verify_assets.py --profile student --sha
```

Release notes must repeat the boundaries: 45/64 on the training set, strict protocol 0/64,
no generalization, no refusal capability, FP16 only (W8A8 rejected).
