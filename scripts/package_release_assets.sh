#!/bin/bash
# Package weight files for GitHub Release (2 GiB per-asset limit).
#
#   scripts/package_release_assets.sh <asset-dir> <out-dir> [--split-mb 1900]
#
# Output: copies/splits known file names, writes SHA256SUMS for the ORIGINAL files.
set -euo pipefail

SRC=${1:?usage: $0 <asset-dir> <out-dir> [--split-mb N]}
OUT=${2:?usage: $0 <asset-dir> <out-dir> [--split-mb N]}
SPLIT_MB=1900
[ "${3:-}" = "--split-mb" ] && SPLIT_MB=${4:?missing value}

mkdir -p "$OUT"
: > "$OUT/SHA256SUMS"

FILES=(
  student_qwen3-1.7b-fp16.rkllm
  student_vision_trained_fp16.rknn
  vocab.bin
  merges.bin
  embed_tokens_151669x2048.bf16
  groundingpi-4b-w8a8.rkllm
  v2-stage00.rknn
  v2-stage09.rknn
  v2-stage18.rknn
  llm-assets/vocab.bin
  llm-assets/merges.bin
  llm-assets/embed.bf16
)

split_one() {
  local src=$1 base=$2
  local bytes
  bytes=$(stat -c %s "$src")
  if [ "$bytes" -le $((SPLIT_MB * 1024 * 1024)) ]; then
    cp -f "$src" "$OUT/$base"
    echo "copy  $base ($bytes bytes)"
  else
    rm -f "$OUT/$base".part-*
    split -b "${SPLIT_MB}M" -d -a 2 "$src" "$OUT/$base.part-"
    # rename numeric suffix to letters (aa, ab, ...) for clarity
    local i=0
    for p in "$OUT/$base".part-[0-9][0-9]; do
      local suffix
      suffix=$(printf "%s" "$(printf '\\%03o%03o' $((97 + i / 26)) $((97 + i % 26)))")
      mv -f "$p" "$OUT/$base.part-$suffix"
      i=$((i + 1))
    done
    echo "split $base ($bytes bytes) -> $i parts of <= ${SPLIT_MB} MiB"
  fi
  ( cd "$(dirname "$src")" && sha256sum "$(basename "$src")" ) >> "$OUT/SHA256SUMS"
}

for f in "${FILES[@]}"; do
  if [ -f "$SRC/$f" ]; then
    split_one "$SRC/$f" "$(basename "$f")"
  fi
done

echo
echo "wrote $OUT/SHA256SUMS"
echo "publish with:"
echo "  gh release create student-fp16-v1 --repo zepher-kk/groundingpi-rk3588 \\"
echo "     --title '3s student FP16 baseline' --notes-file docs/release-notes-student.md \\"
echo "     $OUT/*"
