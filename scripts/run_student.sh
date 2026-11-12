#!/bin/bash
# Run the 3s student. Works from any directory; assets come from the environment.
#
#   scripts/run_student.sh one <jpg> "the dog on the left"
#   scripts/run_student.sh bench <manifest.tsv> <out.tsv>
#   scripts/run_student.sh ids "the dog on the left"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

BIN="$GPI_BUILD_DIR/student_v1"
[ -x "$BIN" ] || { echo "student_v1 not built; run scripts/build.sh" >&2; exit 3; }
[ -d "$GPI_STUDENT_DIR" ] || { echo "GPI_STUDENT_DIR=$GPI_STUDENT_DIR not found" >&2; exit 3; }

exec "$BIN" "$GPI_STUDENT_DIR" "$@"
