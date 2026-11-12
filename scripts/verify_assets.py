#!/usr/bin/env python3
"""Verify that the local asset directory matches models/manifest.json.

Usage:
  scripts/verify_assets.py [--profile student|baseline] [--dir PATH] [--sha]

Default dir: $GPI_STUDENT_DIR for student, $GPI_MODEL_DIR for baseline.
Without --sha only name/size are checked (fast, works on 4 GB files).
Exit code 0 = all good, 1 = missing/mismatched, 2 = usage error.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["student", "baseline"], default="student")
    ap.add_argument("--dir", default="")
    ap.add_argument("--sha", action="store_true", help="also verify SHA256 (slow for GB files)")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(ROOT, "models", "manifest.json"), encoding="utf-8"))
    entry = manifest[args.profile]
    base = args.dir or os.environ.get(
        "GPI_STUDENT_DIR" if args.profile == "student" else "GPI_MODEL_DIR", "")
    if not base:
        print("no --dir and no GPI_STUDENT_DIR/GPI_MODEL_DIR set", file=sys.stderr)
        return 2

    bad = 0
    for f in entry["files"]:
        path = os.path.join(base, f["name"])
        if not os.path.exists(path):
            print("MISSING  %-40s %s" % (f["name"], path))
            bad += 1
            continue
        size = os.path.getsize(path)
        ok_size = size == f.get("bytes", size)
        line = "%-8s %-40s %12d bytes" % ("ok" if ok_size else "SIZE", f["name"], size)
        if ok_size and args.sha and "sha256" in f:
            ok = sha256(path) == f["sha256"]
            line += "  sha %s" % ("ok" if ok else "MISMATCH")
            bad += 0 if ok else 1
        elif not ok_size:
            line += "  expected %d" % f["bytes"]
            bad += 1
        print(line)

    print("\n%s: %d problem(s)" % (args.profile, bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
