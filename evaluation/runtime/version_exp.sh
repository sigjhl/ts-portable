#!/usr/bin/env bash
# Reference implementation version 2.4.0 (reading station) versus 2.12.0 (workstation), both on the
# workstation GPU (Supplemental Table S4). Each configuration runs over the AMOS scans through the
# same timing harness (bench_case.py) and writes one multilabel mask per case to
# $OUT/masks/<version>_<mode>_<device>/<case>.nii.gz; spleen_volumetry/version_comparison.py
# compares the masks.
#
#   runtime/version_exp.sh <amos-images> <out>          (amos-images: imagesVa, CT ids < 500)
#
# Configuration (environment variables):
#   PY2120   Python with TotalSegmentator 2.12.0 and psutil
#   PY240    Python with TotalSegmentator 2.4.0 (PyTorch 2.0.1); TS240_WEIGHTS its weights folder
set -uo pipefail
IMAGES=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$HERE/bench_case.py; RUNTS=$HERE/run_ts_roi.py
JSONL=$OUT/timing.jsonl
mkdir -p "$OUT"

run_arm() {   # version mode device
  local ver=$1 mode=$2 device=$3 maskdir=$OUT/masks/${1}_${2}_${3}
  mkdir -p "$maskdir"
  for ct in "$IMAGES"/amos_*.nii.gz; do
    [ -f "$ct" ] || continue
    local case; case=$(basename "$ct" .nii.gz)
    [ "${case#amos_}" \< "0500" ] || continue
    local key="$ver|$mode|$device|$case"
    grep -q "\"$key\"" "$JSONL" 2>/dev/null && continue
    case $ver in
      ts240)  TOTALSEG_WEIGHTS_PATH=$TS240_WEIGHTS "$PY2120" "$BENCH" "$key" -- "$PY240" "$RUNTS" "$ct" "$maskdir/$case.nii.gz" "$mode" "$device" ;;
      ts2120) "$PY2120" "$BENCH" "$key" -- "$PY2120" "$RUNTS" "$ct" "$maskdir/$case.nii.gz" "$mode" "$device" ;;
    esac >> "$JSONL" 2>/dev/null
  done
  echo "[$ver/$mode/$device] done"
}

run_arm ts240 spleen gpu
run_arm ts2120 spleen gpu
run_arm ts240 full gpu
