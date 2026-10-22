#!/usr/bin/env python3
"""TotalSegmentator wrapper for the version experiment (works with 2.4.0 and 2.12.0 python_api).
Prints COMPUTE_S=<seconds> for bench_case.py and writes a multilabel mask to <out>.
Usage: run_ts_roi.py <ct.nii.gz> <out.nii.gz> <mode: full|spleen> <device: gpu|cpu>
"""
import sys
import time

from totalsegmentator.python_api import totalsegmentator


def main():
    ct, out, mode, device = sys.argv[1:5]
    kwargs = {"ml": True, "quiet": True, "device": device}
    if mode == "spleen":
        kwargs["roi_subset"] = ["spleen"]
    elif mode != "full":
        raise SystemExit(f"bad mode {mode}")
    t = time.perf_counter()
    totalsegmentator(ct, out, **kwargs)
    print(f"COMPUTE_S={time.perf_counter() - t:.3f}", flush=True)


if __name__ == "__main__":
    main()
