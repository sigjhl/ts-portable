"""Compare regenerated result files with the reported ones in data/reported/.

    python3 compare_results.py [RESULTS_DIR]

Walks every reported JSON file that has a regenerated counterpart (same relative path) and prints
the largest absolute difference over all numeric fields. Differences in the last digits come from
floating-point arithmetic on another platform or R version; the reported values are rounded to
five decimals or fewer.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTED = HERE / "data" / "reported"
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
SKIP_KEYS = {"r_version", "source", "rng", "boot_version", "jsonlite_version", "warnings", "implementation",
             "environment", "input", "output"}


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in SKIP_KEYS:
                continue
            if k not in b:
                out["missing"].append(f"{path}/{k}")
                continue
            walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["missing"].append(f"{path} (length {len(a)} vs {len(b)})")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            out["mismatch"].append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = abs(a - b)
        if d > out["max"][0]:
            out["max"] = (d, path)
    elif a != b:
        out["mismatch"].append(f"{path}: {a!r} vs {b!r}")


def main():
    found = 0
    for rep in sorted(REPORTED.rglob("*.json")):
        new = RESULTS / rep.relative_to(REPORTED)
        if not new.exists():
            continue
        found += 1
        out = {"max": (0.0, ""), "missing": [], "mismatch": []}
        walk(json.loads(rep.read_text()), json.loads(new.read_text()), "", out)
        status = "OK" if out["max"][0] < 1e-6 and not out["mismatch"] else "CHECK"
        print(f"{status}  {rep.relative_to(REPORTED)}  max |diff| = {out['max'][0]:.2e}  {out['max'][1]}")
        for m in out["mismatch"][:5]:
            print("      mismatch:", m)
    if not found:
        print("no regenerated files found under", RESULTS)


if __name__ == "__main__":
    main()
