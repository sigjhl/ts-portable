# Windows INT8 deployment — build, stage, run, benchmark

Goal: build a **self-contained OpenVINO INT8 CPU package** on an online Windows machine and copy it
to a target Windows computer that has no Python, no GPU requirement, and possibly no internet access.

---

## 1. Build and stage (online machine)

The `ort` crate is `load-dynamic` (it loads the ONNX Runtime DLL at runtime via `--ort-lib`), so the
build needs **no OpenVINO or CUDA SDK** — just Rust.

```powershell
# 1. Rust toolchain (once): https://rustup.rs  -> default stable MSVC
# 2. Build from the repository root:
cargo build --release
#    -> target\release\onnx-totalseg.exe and target\release\onnx-totalseg-gui.exe

# 3. Stage the self-contained OpenVINO package (downloads the onnxruntime-openvino 1.24.1 and
#    openvino 2025.4.1 wheels, bundles their DLLs, copies .\models):
powershell -ExecutionPolicy Bypass -File deployment\prepare_windows_openvino.ps1 -Force
#    -> dist\onnx-totalseg-openvino\  (onnx-totalseg.exe + onnxruntime\*.dll + models\ + README)
```

Copy `onnx-totalseg-gui.exe` next to `onnx-totalseg.exe` in the staged folder if you want the window.

### (Optional) trim the model payload

`prepare_windows_openvino.ps1` copies the whole `models\` folder. An INT8 delivery needs only the INT8
weights:

```powershell
$m = "dist\onnx-totalseg-openvino\models"
# INT8, all tiers/ROIs (about 210 MB unpacked): keep *_int8.onnx + cards + label map, drop fp32/fp16
Get-ChildItem $m -File | Where-Object { $_.Name -notmatch '(_int8\.onnx$|\.model_card\.json$|^total_label_map\.json$)' } | Remove-Item
# --- OR spleen-only (~48 MB): keep just the 6-mm localizer + the organs submodel (contains spleen).
#   Model weights: total_6mm_298_int8.onnx, total_1p5mm_291_organs_int8.onnx.
#   Metadata: total_6mm_298.model_card.json, every total_1p5mm_29*.model_card.json,
#             and total_label_map.json. ROI routing reads all five 1.5-mm cards before
#             selecting the organs model, but no other 1.5-mm weights are needed.
```

Then copy `dist\onnx-totalseg-openvino\`, `deployment\run_openvino.ps1`, and
`deployment\bench_openvino.ps1` to the target computer.

---

## 2. Run (target computer)

Smoke test (INT8, spleen-only mode, one scan):

```powershell
.\run_openvino.ps1 -InputPath data\scan.nii.gz -OutputPath out\scan_spleen.nii.gz
# confirms the OpenVINO CPU plugin loads and writes a mask.
```

`run_openvino.ps1` defaults to INT8, the 1.5-mm tier, and `-Roi spleen` (6-mm localization, crop,
1.5-mm inference — the equivalent of the reference implementation's ROI-subset option). Pass `-Fp32`
for the fp32 OpenVINO baseline, `-Tier fast` for the 3-mm model, and `-Statistics` for the volume and
HU table.

To reproduce the manuscript's precision setting, set
`$env:ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION = "f32"` before running (see
[`docs/METHOD.md`](../docs/METHOD.md)).

---

## 3. Benchmark

`bench_openvino.ps1` wraps any command and records wall time and peak working-set / private memory of
the process tree, appending one JSON record per run to `benchmark_results.jsonl`:

```powershell
Get-ChildItem data\*.nii.gz | ForEach-Object {
  .\bench_openvino.ps1 -Label "int8_$($_.BaseName)" .\run_openvino.ps1 `
    -InputPath $_.FullName -OutputPath "out\i8_$($_.Name)"
}
```

Run one case per process so every run pays its own startup, session creation, and input/output.

### Manuscript PACS reading-station demonstration

The manuscript ran this package on a PACS reading station (Intel Core i5-10400 six-core CPU, 32 GiB
RAM, NVIDIA Quadro P400 with 2 GiB) over all 100 AMOS validation scans in spleen-only mode, timed
against the reference implementation of TotalSegmentator on the station's CPU and GPU (stock
TotalSegmentator run without `--force_split`, which is a whole-volume VRAM option unnecessary for an
ROI crop). Median per-case runtime was **5.8 s** for TS-Portable on the CPU, 43.4 s for the reference
implementation on the CPU, and 43.5 s on the Quadro P400.

Spleen-volume agreement with the AMOS reference masks (Lin's concordance correlation coefficient,
Bland–Altman limits, splenomegaly threshold 314.5 mL) is computed offline from the output masks, not
on the station.
