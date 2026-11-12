# Experiments (unsupported, opt-in)

These tools are kept for reproducing the measurements in `docs/performance.md` and
`docs/batch-experiment.md`. They are **not built by default** and are not part of the
supported entry points:

```bash
scripts/build.sh --experiments            # probes (no OpenCV needed)
scripts/build.sh --experiments --opencv   # + embedding prep / batch pair / RGA probes
```

They take paths from the environment instead of hard-coded machine paths:

| Variable | Used by |
| --- | --- |
| `GPI_STUDENT_DIR` | check_ids / check_l12 / check_vision / parser_check / tok_lookup |
| `GPI_STUDENT_BIN` | same scripts (defaults to `$GPI_BUILD_DIR/student_v1`) |
| `GPI_EXPERIMENT_*` | older probes that previously defaulted to author paths — they now default to empty and must be given paths on the command line |

| File | Purpose |
| --- | --- |
| `protocol_probe.cc` | run the 190-token protocol prompt with a known embedding layout |
| `student_coexist.cc`, `coexist_probe.cc` | co-residency probes (vision + language in one process) |
| `pipeline_bench.cc`, `resident_service.cc` | earlier multi-image / resident-service benchmarks |
| `probe.cc`, `rga_probe*.cc` | RKNN and RGA smoke probes |
| `student_batch_probe.cc` | language-only batch=1 vs batch=2 |
| `student_batch_prep.cc` | build prompt embeddings for a manifest (vision once) |
| `student_batch_pair.cc` | full pipeline (vision + language) for a pair of images |
| `protocol_parse.py` | Python mirror of the C++ parser (`deploy-parse-v1`) |
| `parser_check.py` | 17-case C++/Python parser equality check |
| `check_ids.py`, `check_l12.py`, `check_vision.py` | tokenizer / L1-L2 / vision parity checks |
| `tok_lookup.py` | print token strings for selected ids from `vocab.bin` |

Notes:

- The older probes (`protocol_probe`, `coexist_probe`, `pipeline_bench`, `resident_service`)
  contain 256-slot / 448-canvas assumptions from the original GroundingPI profile. They are
  not part of the student path.
- No experiments run in CI; there is no CI.
