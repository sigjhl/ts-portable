#!/usr/bin/env python3
"""Stock TotalSegmentator wrapper that times only the compute call (excludes import).
Run with the TS venv python. Prints COMPUTE_S=<seconds> for bench_case.py to parse,
and writes a multilabel mask to <out>.

Usage: run_ts.py <ct.nii.gz> <out.nii.gz> <tier: total|fast|fastest> <device: gpu|cpu>
"""
import sys
import time

from totalsegmentator.python_api import totalsegmentator

FLAGS = {"total": {}, "fast": {"fast": True}, "fastest": {"fastest": True}}


def main():
    ct, out, tier, device = sys.argv[1:5]
    t = time.perf_counter()
    totalsegmentator(ct, out, ml=True, quiet=True, device=device, **FLAGS[tier])
    print(f"COMPUTE_S={time.perf_counter() - t:.3f}", flush=True)


if __name__ == "__main__":
    main()
