"""Bootstrap CIs for the Bland-Altman relative bias and 95% limits of agreement, 100 external AMOS
cases (Table 5, AMOS rows), and the limits without the smallest spleen (amos_0167).

    python spleen_volumetry/amos_bland_altman_ci.py [VOLUMES_CSV] [OUT_DIR]

Relative difference = first minus second volume as a percentage of the pair mean; limits = mean
+/- 1.96 SD. Percentile CIs from 10,000 case resamples (NumPy default_rng, seed 20260718, one stream
drawn in the order of the comparisons below). The CCC and its BCa interval come from
amos_spleen_agreement.R.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOLUMES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spleen" / "amos_spleen_volumes_percase.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "results" / "spleen_volumetry"
PAIRS = [
    ("PACS spleen-only: TS-Portable vs manual", "pacs_int8_openvino_ml", "manual_ml"),
    ("PACS spleen-only: reference 2.4.0 vs manual", "pacs_stock_ts_cpu_ml", "manual_ml"),
    ("PACS spleen-only: TS-Portable vs reference 2.4.0", "pacs_int8_openvino_ml", "pacs_stock_ts_cpu_ml"),
    ("Workstation full mode: TS-Portable vs manual", "int8_full_ml", "manual_ml"),
    ("Workstation full mode: reference 2.12.0 vs manual", "reference_full_ml", "manual_ml"),
    ("Workstation full mode: TS-Portable vs reference 2.12.0", "int8_full_ml", "reference_full_ml"),
]

rows = list(csv.DictReader(open(VOLUMES)))
cases = np.array([r["case"] for r in rows])
rng = np.random.default_rng(20260718)
out = {}
for label, a, b in PAIRS:
    x = np.array([float(r[a]) for r in rows])
    y = np.array([float(r[b]) for r in rows])
    rel = 100 * (x - y) / ((x + y) / 2)
    bias, sd = rel.mean(), rel.std(ddof=1)
    rb = rel[rng.integers(0, len(rows), (10000, len(rows)))]
    bb, sb = rb.mean(1), rb.std(1, ddof=1)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    keep = cases != "amos_0167"
    out[label] = {
        "bias": float(bias), "bias_ci": ci(bb),
        "loa_low": float(bias - 1.96 * sd), "loa_low_ci": ci(bb - 1.96 * sb),
        "loa_high": float(bias + 1.96 * sd), "loa_high_ci": ci(bb + 1.96 * sb),
        "within_loa_frac": float(np.mean(np.abs(rel - bias) <= 1.96 * sd)),
        "without_amos_0167": {"bias": float(rel[keep].mean()),
                              "loa": [float(rel[keep].mean() - 1.96 * rel[keep].std(ddof=1)),
                                      float(rel[keep].mean() + 1.96 * rel[keep].std(ddof=1))]},
    }
    o = out[label]
    print(f"{label:55s} bias {o['bias']:+.1f}% ({o['bias_ci'][0]:+.1f}, {o['bias_ci'][1]:+.1f}); "
          f"LoA {o['loa_low']:+.1f} ({o['loa_low_ci'][0]:+.1f}, {o['loa_low_ci'][1]:+.1f}) to "
          f"{o['loa_high']:+.1f} ({o['loa_high_ci'][0]:+.1f}, {o['loa_high_ci'][1]:+.1f})")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "amos_spleen_ba_ci.json").write_text(json.dumps(out, indent=1))
