"""Runtime and memory summaries (Table 3 medians and memory, Table 4, Table S13 timing).

    python3 runtime/summarize_runtime.py [DATA_DIR] [OUT_DIR]

Inputs in DATA_DIR (default data/runtime):
- workstation_timing_amos.jsonl: one line per case and configuration from runtime/bench_case.py,
  label "<configuration>|<full|fast>|<case>", proc_wall_s = cold end-to-end wall time,
  peak_pss_mib = peak proportional set size of the process tree;
- station_amos100_spleen.csv: PACS reading station, spleen-only mode, one warm-up case then 100
  cases per configuration (runtime/station/Run-Experiment.ps1);
- station_duke_spleen.csv: the same for the Duke series from DICOM and from NIfTI.
Writes runtime_summary.json. Standard library only.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "runtime"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "results" / "runtime"
WS_NAMES = {"ts_gpu": "Reference implementation, GPU", "ts_cpu": "Reference implementation, CPU",
            "ours_fp32": "TS-Portable fp32 ONNX, OpenVINO CPU", "ours_fp32_cuda": "TS-Portable fp32 ONNX, CUDA GPU",
            "ours_int8_nncf": "TS-Portable INT8, OpenVINO CPU"}


def pct(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def workstation():
    rows = [json.loads(line) for line in open(DATA / "workstation_timing_amos.jsonl") if line.strip()]
    by = {}
    for r in rows:
        cond, tier, case = r["label"].split("|")
        if r["rc"] == 0:
            by.setdefault((cond, tier), []).append(r)
    out = {}
    for tier in ("full", "fast"):
        ref = st.median(r["proc_wall_s"] for r in by[("ours_int8_nncf", tier)])
        for cond, name in WS_NAMES.items():
            rs = by[(cond, tier)]
            med = st.median(r["proc_wall_s"] for r in rs)
            out[f"{cond}|{tier}"] = {
                "configuration": name, "tier": "1.5 mm" if tier == "full" else "3 mm", "n": len(rs),
                "median_wall_s": med, "ratio_to_int8": med / ref,
                "median_peak_pss_gib": st.median(r["peak_pss_mib"] for r in rs) / 1024}
    return out


def station(name, warm_col="warmup"):
    rows = list(csv.DictReader(open(DATA / name)))
    # DSDS_0021 is excluded from the Duke analyses (reference mask not aligned with the images)
    rows = [r for r in rows if r["status"] == "ok" and r[warm_col].lower() == "false" and r["case"] != "DSDS_0021"]
    arms = sorted({r["arm"] for r in rows})
    out = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        w = [float(r["wall_seconds"]) for r in rs]
        d = {"n": len(rs), "median_wall_s": st.median(w), "p90_wall_s": pct(w, 90),
             "q1_wall_s": pct(w, 25), "q3_wall_s": pct(w, 75),
             "median_peak_working_set_gib": st.median(float(r["peak_tree_working_set_mb"]) for r in rs) / 1024}
        if rs[0].get("peak_gpu_mb"):
            g = [float(r["peak_gpu_mb"]) for r in rs if r["peak_gpu_mb"]]
            if g:
                d["max_peak_gpu_gib"] = max(g) / 1024
            d["cpu_fallback_detected"] = sum(r.get("cpu_fallback_detected", "").lower() == "true" for r in rs)
        out[arm] = d
    base = next((a for a in ("int8_openvino",) if a in out), None)
    if base:
        for a in out:
            out[a]["ratio_to_tsportable"] = out[a]["median_wall_s"] / out[base]["median_wall_s"]
    return out


def main():
    out = {"workstation_amos": workstation(),
           "station_amos_spleen_only": station("station_amos100_spleen.csv"),
           "station_duke_spleen_only": station("station_duke_spleen.csv")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runtime_summary.json").write_text(json.dumps(out, indent=1))
    print("Workstation (Table 3): median wall s, ratio to INT8, median peak PSS GiB")
    for k, v in out["workstation_amos"].items():
        print(f"  {v['configuration']:40s} {v['tier']:6s} {v['median_wall_s']:7.1f} {v['ratio_to_int8']:5.2f} {v['median_peak_pss_gib']:4.1f}")
    for key, title in (("station_amos_spleen_only", "Reading station, AMOS (Table 4)"),
                       ("station_duke_spleen_only", "Reading station, Duke (Table S13)")):
        print(title + ": median, p90, ratio, median peak working set GiB")
        for arm, v in out[key].items():
            extra = f"  peak GPU {v['max_peak_gpu_gib']:.2f} GiB" if "max_peak_gpu_gib" in v else ""
            print(f"  {arm:16s} n={v['n']:3d} {v['median_wall_s']:6.2f} {v['p90_wall_s']:6.2f} "
                  f"{v.get('ratio_to_tsportable', float('nan')):5.2f} {v['median_peak_working_set_gib']:5.2f}{extra}")


if __name__ == "__main__":
    main()
