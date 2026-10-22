#!/usr/bin/env bash
# Workstation timing and memory driver (Table 3): runs one configuration at one resolution over a
# folder of CT scans, one case per process, and appends one JSON line per case (runtime/bench_case.py:
# cold end-to-end wall time and peak proportional set size of the process tree) to a JSONL file.
# Resumable: cases already in the JSONL are skipped. Masks are written to <maskdir> and can be scored
# with scoring/score_accuracy.py.
#
#   runtime/run_bench.sh <config> <tier> <inputs> <jsonl> <maskdir>
#
#   config : ours_int8_nncf  TS-Portable, INT8 models, ONNX Runtime with OpenVINO, CPU
#            ours_fp32       TS-Portable, fp32 ONNX models, ONNX Runtime with OpenVINO, CPU
#            ours_fp32_cuda  TS-Portable, fp32 ONNX models, ONNX Runtime with CUDA, GPU
#            ts_cpu | ts_gpu reference implementation (TotalSegmentator 2.12.0), CPU or GPU
#   tier   : full (1.5 mm) | fast (3 mm)
#   inputs : folder of CT scans, either <case>.nii.gz (AMOS imagesVa) or <case>/ct.nii.gz
#
# Configuration (environment variables; defaults in brackets):
#   PY             Python with psutil and, for ts_*, TotalSegmentator 2.12.0        [python3]
#   BIN            TS-Portable command-line program (cargo build --release)       [target/release/onnx-totalseg]
#   ORT_OPENVINO   onnxruntime-openvino 1.24.1 shared library (libonnxruntime.so.1.24.1)   [required for ours_*]
#   INT8_MODELS    models folder of the release (INT8 models)                     [models]
#   FP32_MODELS    folder of the fp32 ONNX models                                  [required for ours_fp32*]
#   ORT_CUDA_DIR   folder of the onnxruntime-gpu 1.22.0 shared libraries           [required for ours_fp32_cuda]
#   CUDNN_DIR      folder of the cuDNN 9 libraries                                 [required for ours_fp32_cuda]
#   CASELIST       optional file of case ids to restrict the run
#   NIGHT_ONLY=1   stop between 08:00 and 19:00 (the reported runs used a quiet machine)
set -uo pipefail
cfg=$1; tier=$2; inputs=$3; jsonl=$4; maskdir=$5
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
BIN=${BIN:-target/release/onnx-totalseg}
INT8_MODELS=${INT8_MODELS:-models}
BENCH=$HERE/bench_case.py
RUNTS=$HERE/run_ts.py
mkdir -p "$maskdir" "$(dirname "$jsonl")"

if [ -n "${CASELIST:-}" ]; then
  mapfile -t CASES < <(grep -v '^$' "$CASELIST")
elif compgen -G "$inputs/*/ct.nii.gz" > /dev/null; then
  mapfile -t CASES < <(for d in "$inputs"/*/; do [ -f "$d/ct.nii.gz" ] && basename "$d"; done)
else
  mapfile -t CASES < <(for f in "$inputs"/*.nii.gz; do basename "$f" .nii.gz; done)
fi
ct_of() { if [ -f "$inputs/$1/ct.nii.gz" ]; then echo "$inputs/$1/ct.nii.gz"; else echo "$inputs/$1.nii.gz"; fi; }

ours_flag=""; [ "$tier" = fast ] && ours_flag="--fast"
ts_tier="total"; [ "$tier" = fast ] && ts_tier="fast"

echo "[$cfg/$tier] ${#CASES[@]} cases -> $jsonl"
for case in "${CASES[@]}"; do
  if [ "${NIGHT_ONLY:-0}" = 1 ]; then
    h=$(date +%H)
    if [ "$h" -ge 8 ] && [ "$h" -lt 19 ]; then echo "[$cfg/$tier] stopping (hour $h)"; exit 7; fi
  fi
  ct=$(ct_of "$case"); [ -f "$ct" ] || continue
  key="$cfg|$tier|$case"
  grep -q "\"$key\"" "$jsonl" 2>/dev/null && continue
  mask="$maskdir/$case.nii.gz"
  case $cfg in
    ours_int8_nncf) ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION=f32 ORT_DYLIB_PATH=$ORT_OPENVINO $PY "$BENCH" "$key" -- \
                      "$BIN" segment -i "$ct" -o "$mask" $ours_flag --int8 --device openvino --models-dir "$INT8_MODELS" --profile ;;
    ours_fp32)      ORT_DYLIB_PATH=$ORT_OPENVINO $PY "$BENCH" "$key" -- \
                      "$BIN" segment -i "$ct" -o "$mask" $ours_flag --device openvino --models-dir "$FP32_MODELS" --profile ;;
    ours_fp32_cuda) LD_LIBRARY_PATH=$ORT_CUDA_DIR:$CUDNN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} $PY "$BENCH" "$key" -- \
                      "$BIN" segment -i "$ct" -o "$mask" $ours_flag --fp32 --device cuda --models-dir "$FP32_MODELS" \
                      --ort-lib "$ORT_CUDA_DIR/libonnxruntime.so.1.22.0" --profile ;;
    ts_gpu)         $PY "$BENCH" "$key" -- $PY "$RUNTS" "$ct" "$mask" "$ts_tier" gpu ;;
    ts_cpu)         $PY "$BENCH" "$key" -- $PY "$RUNTS" "$ct" "$mask" "$ts_tier" cpu ;;
    *) echo "unknown configuration $cfg"; exit 2 ;;
  esac >> "$jsonl" 2>/dev/null
  tail -1 "$jsonl" | sed 's/, "log":[^}]*//'
done
echo "[$cfg/$tier] done"
