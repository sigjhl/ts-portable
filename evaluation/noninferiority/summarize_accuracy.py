"""Descriptive accuracy summaries from the per-case records.

    python3 noninferiority/summarize_accuracy.py [DATA_DIR] [OUT_DIR]

Reads data/accuracy/percase_{amos,ts}_{full,fast}.json (scoring/score_accuracy.py) and the fp32
records in data/accuracy/fp32/, and, if present, the bootstrap results written by
bootstrap_endpoints.R in OUT_DIR. Writes accuracy_summary.json with:

- empty predictions per implementation (Results, "Noninferiority");
- direct mask-to-mask DSC between the implementations (Supplemental Table S6);
- the per-case and per-organ-case distribution of the paired difference (Supplemental Table S7);
- fp32 ONNX conversion fidelity (Supplemental Table S5);
- Holm adjustment across the six secondary endpoint tests (Results, "Noninferiority").

Differences are reference implementation minus TS-Portable; positive values favor the reference.
Standard library only.
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "accuracy"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "results" / "noninferiority"
KEYS = ("amos_full", "amos_fast", "ts_full", "ts_fast")


def load(name):
    return json.loads((DATA / name).read_text())


def pct(values, q):
    """Percentile with linear interpolation (numpy's default)."""
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def empties(rows):
    i8 = sum(o["int8_empty"] for r in rows for o in r["organs"])
    ref = sum(o["stock_ts_empty"] for r in rows for o in r["organs"])
    only_i8 = [(r["case"], o["organ"]) for r in rows for o in r["organs"] if o["int8_empty"] and not o["stock_ts_empty"]]
    only_ref = [(r["case"], o["organ"]) for r in rows for o in r["organs"] if o["stock_ts_empty"] and not o["int8_empty"]]
    pairs = sum(len(r["organs"]) for r in rows)
    return {"pairs": pairs, "tsportable": i8, "reference": ref,
            "tsportable_only": only_i8, "reference_only": only_ref}


def direct(rows):
    by, all_pairs, both_nonempty = {}, [], []
    for r in rows:
        for o in r["organs"]:
            v = o["dsc_int8_vs_stock_ts"]
            if v is None:  # both masks empty
                continue
            by.setdefault(o["organ"], []).append((v, r["case"]))
            all_pairs.append((v, r["case"], o["organ"]))
            if not o["int8_empty"] and not o["stock_ts_empty"]:
                both_nonempty.append((v, r["case"], o["organ"]))
    organs = {k: {"n": len(v), "mean": st.mean(x for x, _ in v), "lowest": min(v)[0], "lowest_case": min(v)[1]}
              for k, v in by.items()}
    low_label = min(organs.items(), key=lambda kv: kv[1]["mean"])
    return {
        "mean_over_structures": st.mean(o["mean"] for o in organs.values()),
        "lowest_structure_mean": {"structure": low_label[0], "mean": low_label[1]["mean"]},
        "p5_all_pairs": pct([x for x, *_ in all_pairs], 5),
        "lowest_pair": dict(zip(("dsc", "case", "structure"), min(all_pairs))),
        "lowest_pair_both_nonempty": dict(zip(("dsc", "case", "structure"), min(both_nonempty))),
        "per_structure": organs,
    }


def case_tails(rows, metric):
    d = [r[f"mean_{metric}_stock_ts"] - r[f"mean_{metric}_int8"] for r in rows]
    return {"n": len(d), "median": st.median(d), "p90": pct(d, 90), "p95": pct(d, 95), "max": max(d),
            "cases_gt_0.005": sum(x > 0.005 for x in d), "cases_gt_0.01": sum(x > 0.01 for x in d),
            "cases_gt_0.01_ids": [r["case"] for r, x in zip(rows, d) if x > 0.01]}


def organ_case(rows, metric):
    fi, fr = ("dsc_int8_vs_manual", "dsc_stock_ts_vs_manual") if metric == "dsc" else ("nsd3_int8", "nsd3_stock_ts")
    d = [(o[fr] - o[fi], r["case"], o["organ"]) for r in rows for o in r["organs"]]
    x = [v for v, *_ in d]
    worst = max(d)
    best = min(d)
    return {"n": len(x), "median": st.median(x), "p95": pct(x, 95), "p99": pct(x, 99),
            "max_loss": worst[0], "max_loss_pair": [worst[1], worst[2]],
            "max_gain": -best[0], "max_gain_pair": [best[1], best[2]],
            "losses_gt_0.01": sum(v > 0.01 for v in x), "losses_gt_0.05": sum(v > 0.05 for v in x),
            "losses_gt_0.10": sum(v > 0.10 for v in x), "gains_gt_0.01": sum(v < -0.01 for v in x)}


def fp32(key, rows):
    cohort, tier = key.split("_")
    dsc = load(f"fp32/dsc_percase_{key}.json")
    hist = load("fp32/nsd_amos_percase.json")[tier] if cohort == "amos" else load(f"fp32/nsd_ts_percase_{tier}.json")
    nsd_fp32 = st.mean(st.mean(o["f3"] for o in r["organs"]) for r in hist)
    nsd_ref = st.mean(r["mean_nsd3_stock_ts"] for r in rows)
    dsc_fp32 = st.mean(r["mean_dsc_onnx_fp32"] for r in dsc)
    dsc_ref = st.mean(r["mean_dsc_original_ts"] for r in dsc)
    return {"n": len(rows), "nsd3_fp32": nsd_fp32, "nsd3_reference": nsd_ref,
            "nsd3_reference_minus_fp32": nsd_ref - nsd_fp32,
            "dsc_fp32": dsc_fp32, "dsc_reference": dsc_ref, "dsc_reference_minus_fp32": dsc_ref - dsc_fp32}


def holm(pvals):
    order = sorted(pvals, key=pvals.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def main():
    out = {"empty_predictions": {}, "direct_dsc": {}, "per_case_tails": {}, "per_organ_case": {}, "fp32_fidelity": {}}
    for key in KEYS:
        rows = load(f"percase_{key}.json")
        out["empty_predictions"][key] = empties(rows)
        out["direct_dsc"][key] = direct(rows)
        out["per_case_tails"][key] = {m: case_tails(rows, m) for m in ("dsc", "nsd3")}
        out["per_organ_case"][key] = {m: organ_case(rows, m) for m in ("dsc", "nsd3")}
        out["fp32_fidelity"][key] = fp32(key, rows)

    res = {"dsc": OUT / "dsc_bootstrap_results.json", "nsd3": OUT / "r_bootstrap_results.json"}
    if all(p.exists() for p in res.values()):
        p = {f"{k}_{m}": json.loads(res[m].read_text())["primary"][k]["p_value_one_sided_bca"]
             for k in ("amos_fast", "ts_full", "ts_fast") for m in res}
        out["holm_secondary"] = {"unadjusted": p, "adjusted": holm(p)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "accuracy_summary.json").write_text(json.dumps(out, indent=1))

    print("Empty predictions (TS-Portable / reference, of pairs):")
    for k, v in out["empty_predictions"].items():
        print(f"  {k:10s} {v['tsportable']:4d} / {v['reference']:4d} of {v['pairs']}; TS-Portable only: {v['tsportable_only']}")
    print("Direct DSC, mean over structures / 5th percentile / lowest pair:")
    for k, v in out["direct_dsc"].items():
        print(f"  {k:10s} {v['mean_over_structures']:.4f} / {v['p5_all_pairs']:.4f} / {v['lowest_pair']}")
    print("Per-case tails (median, p90, p95, max, >0.005, >0.01):")
    for k, v in out["per_case_tails"].items():
        for m, t in v.items():
            print(f"  {k:10s} {m:5s} {t['median']:.4f} {t['p90']:.4f} {t['p95']:.4f} {t['max']:.4f} {t['cases_gt_0.005']} {t['cases_gt_0.01']}")
    print("Organ-case (n, median, p95, p99, max loss, max gain, >0.01, >0.05, >0.10, gains>0.01):")
    for k, v in out["per_organ_case"].items():
        for m, t in v.items():
            print(f"  {k:10s} {m:5s} {t['n']} {t['median']:.4f} {t['p95']:.4f} {t['p99']:.4f} {t['max_loss']:.3f} {t['max_gain']:.3f} "
                  f"{t['losses_gt_0.01']} {t['losses_gt_0.05']} {t['losses_gt_0.10']} {t['gains_gt_0.01']}")
    print("fp32 fidelity (NSD@3 fp32, reference, diff; DSC diff):")
    for k, v in out["fp32_fidelity"].items():
        print(f"  {k:10s} {v['nsd3_fp32']:.4f} {v['nsd3_reference']:.4f} {v['nsd3_reference_minus_fp32']:+.5f}; {v['dsc_reference_minus_fp32']:+.5f}")
    if "holm_secondary" in out:
        print("Holm-adjusted secondary p values:", {k: round(v, 4) for k, v in out["holm_secondary"]["adjusted"].items()})


if __name__ == "__main__":
    main()
