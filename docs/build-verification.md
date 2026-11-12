# Clean-checkout build verification

Verified on 2026-11-12 from a `git archive` of this repository (no working-tree state, no
cached build), on the target board:

| Item | Value |
| --- | --- |
| Board | LubanCat 4, RK3588, aarch64, 8 GB |
| OS / toolchain | Ubuntu 22.04 userland, g++ 11.4, CMake 3.22.1 |
| Runtime | rkllm 1.3.1 (`/usr/lib/librkllmrt.so`), rknn 2.5.0 (`/usr/lib/librknnrt.so`) |
| SDK dirs | `/tmp/rk-sdk/rkllm` (rkllm.h + lib), `/tmp/rk-sdk/rknn` (rknn_api.h + lib) |

Procedure (as a user would do it):

```bash
unzip repo.zip -d /tmp/gpi-rel && cd /tmp/gpi-rel
export RKLLM_SDK_DIR=/tmp/rk-sdk/rkllm
export RKNN_SDK_DIR=/tmp/rk-sdk/rknn
scripts/build.sh                          # production apps only
scripts/build.sh --experiments --opencv   # + probes / batch / RGA tools
```

Results:

| Step | Result |
| --- | --- |
| `scripts/build.sh` | `groundingpi` 103,192 B and `student_v1` 108,080 B built |
| `scripts/build.sh --experiments --opencv` | 13 targets built (`student_batch_probe`, `student_batch_prep`, `student_batch_pair`, `protocol_probe`, `coexist_probe`, `pipeline_bench`, `resident_service`, `probe`, `rga_probe*`, …) |
| `scripts/verify_assets.py --profile student` | 5/5 files, sizes match, 0 problems |
| `scripts/run_student.sh ids "the dog on the left"` from `/tmp` | 190 token ids, identical to the Python reference |

Notes:

- The repository ships **no** SDK headers, libraries or weights; the paths above point at local
  copies. `CMakeLists.txt` fails fast with a clear message when `RKLLM_SDK_DIR`/`RKNN_SDK_DIR`
  are missing.
- Build RPATH is derived from the SDK dirs (`GPI_RPATH` overrides), so no machine path is baked
  into the binaries.
- Experiments are opt-in; the default build produces exactly the two supported entries.
