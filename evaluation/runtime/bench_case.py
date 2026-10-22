#!/usr/bin/env python3
"""Run a command under a process-tree peak-RSS sampler and report timing.

Fair memory measurement for both tools: stock TS spawns nnU-Net subprocesses, so
sampling only the direct child under-counts it. We poll RSS of the whole process tree
(psutil) and keep the peak. Reports:
  - proc_wall_s : wall time of the whole subprocess (cold, includes import/model-load)
  - peak_pss_mib: peak summed PSS (proportional set size) across the process tree.
                  PSS apportions shared pages, so summing across a multi-process tool (TS
                  forks nnU-Net workers sharing torch/CUDA libs) gives the true unique
                  physical footprint — summing RSS would double-count shared pages and
                  unfairly inflate multi-process tools. Falls back to RSS if PSS is
                  unavailable (non-Linux).
  - compute_s   : compute-only time parsed from the child's stdout (COMPUTE_S=, or the
                  Rust binary's `predict=<n>s`), i.e. excluding interpreter/import — the
                  fair cross-tool timing number
  - infer_s     : inference-lap only, from the Rust binary's `sliding_window_ort: <n>s`
                  (summed if multiple submodels); None for TS

Run with a Python that has psutil (requirements.txt). The child inherits the
parent env, so set ORT_DYLIB_PATH etc. before calling.

Usage: bench_case.py <label> -- <command...>
"""
import json
import re
import subprocess
import sys
import tempfile
import time

import psutil


def main():
    label = sys.argv[1]
    assert sys.argv[2] == "--", "usage: bench_case.py <label> -- <command...>"
    cmd = sys.argv[3:]
    interval = 0.2  # PSS reads smaps (heavier); 0.2s is ample for 10-500s runs

    def tree_mem(proc):
        """Summed PSS (fair; apportions shared pages) over the process tree; RSS fallback."""
        total_pss = 0
        used_pss = True
        for pr in [proc, *proc.children(recursive=True)]:
            try:
                fi = pr.memory_full_info()
                pss = getattr(fi, "pss", None)
                if pss is None:
                    total_pss += fi.rss
                    used_pss = False
                else:
                    total_pss += pss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_pss, used_pss

    logf = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    try:
        proc = psutil.Process(p.pid)
    except psutil.NoSuchProcess:
        proc = None
    peak = 0
    pss_ok = True
    while p.poll() is None:
        if proc is not None:
            try:
                mem, used_pss = tree_mem(proc)
                peak = max(peak, mem)
                pss_ok = pss_ok and used_pss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(interval)
    rc = p.returncode
    proc_wall = time.perf_counter() - t0

    logf.seek(0)
    out = logf.read()

    m = re.search(r"COMPUTE_S=([0-9.]+)", out)
    compute = float(m.group(1)) if m else None
    if compute is None:
        m = re.search(r"predict=([0-9.]+)s", out)
        compute = float(m.group(1)) if m else None
    infers = [float(x) for x in re.findall(r"sliding_window_ort:\s*([0-9.]+)s", out)]
    infer = round(sum(infers), 3) if infers else None

    print(json.dumps({
        "label": label, "rc": rc,
        "proc_wall_s": round(proc_wall, 2),
        "peak_pss_mib": round(peak / 1048576, 1),
        "mem_metric": "pss" if pss_ok else "rss_fallback",
        "compute_s": round(compute, 2) if compute is not None else None,
        "infer_s": infer,
        "log": logf.name,
    }))
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
