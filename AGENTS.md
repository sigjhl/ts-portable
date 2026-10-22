# Notes for coding agents

Facts about this repository that a script or an agent needs. The README is the
human introduction; this file is the reference.

## What is here

- `onnx-totalseg` (CLI, `src/main.rs`): segments one CT scan with the
  TotalSegmentator `total` task models (TotalSegmentator 2.12.0) through ONNX
  Runtime. The INT8 models run on the CPU through the OpenVINO execution
  provider; the fp32 models run through the CPU, CUDA or DirectML providers.
- `onnx-totalseg-gui` (desktop window, `src/gui_main.rs`): a front end that
  runs the CLI binary found next to it, always with `--int8 --device openvino
  --roi_subset <organs>`, and shows the volume table. It adds nothing the CLI
  cannot do.
- `onnx-totalseg serve`: an HTTP server that keeps the models loaded for
  repeated runs on one machine (`--help` for its options). Batch work does not
  need it; one process per scan is how the manuscript's timings were measured.
- `scripts/quantize_*.py`, `docs/METHOD.md`: how the INT8 models were made.
  Not needed to use the tool.
- `deployment/`: Windows staging and run scripts, and `models.sha256`.
- `.github/workflows/windows-bundle.yml`: builds and publishes the Windows zip
  on a `v*` tag.

## Build

Rust stable (edition 2024). No ONNX Runtime, OpenVINO or CUDA SDK is needed to
build; the binaries load the ONNX Runtime library at run time.

```
cargo build --release --locked          # target/release/onnx-totalseg, onnx-totalseg-gui
cargo test --release --locked
```

## Run-time pieces and where they are looked up

1. **Models**: `--models-dir DIR`, else `$ONNX_TOTALSEG_MODELS`, else `./models`
   relative to the current working directory. The GUI always uses `models/`
   next to its executable. The INT8 set is `ts-portable-models.zip` from
   https://huggingface.co/sigjhl/ts-portable (172 MB zipped, 212 MB
   unpacked); unzipping it yields `models/` holding 7 `*_int8.onnx`, 7
   `*.model_card.json`, `total_label_map.json` and `SHA256SUMS.txt`. The
   checksums are also in `deployment/models.sha256`. The same repository holds
   the fp32 models.
2. **ONNX Runtime library**: `--ort-lib PATH`, else `$ORT_DYLIB_PATH`, else the
   system library. For INT8 it must be the `onnxruntime-openvino` 1.24.1 build:
   - Windows: the release zip has it in `onnxruntime\onnxruntime.dll` with the
     OpenVINO DLLs beside it; the `onnxruntime` folder must be on `PATH` when
     calling the CLI directly (`deployment\run_openvino.ps1` does this). To
     build the folder yourself: `deployment\prepare_windows_openvino.ps1`.
   - Linux x86-64: `pip install onnxruntime-openvino==1.24.1` and point
     `--ort-lib` at `site-packages/onnxruntime/capi/libonnxruntime.so.1.24.1`.
   - macOS: no OpenVINO build exists, so no INT8. Use `pip install onnxruntime`,
     `--device cpu`, and the fp32 models.
3. **Execution provider**: `--device openvino` is required for `--int8`
   (the CLI refuses otherwise). `--device auto` picks DirectML when present,
   else CPU, and never OpenVINO.

Minimal INT8 run (Linux shown; on Windows use the `.exe` and `.dll` paths):

```
onnx-totalseg -i scan.nii.gz -o out/ --device openvino --int8 \
    --ort-lib /path/to/libonnxruntime.so.1.24.1 --models-dir /path/to/models \
    --roi_subset spleen,liver --statistics --quiet
```

## Inputs

- `-i` takes a NIfTI file (`.nii`, `.nii.gz`) or a DICOM directory. A DICOM
  folder with several series prompts for one when stdin is a terminal;
  otherwise the largest series is used. Pass `--dicom-series <SeriesInstanceUID>`
  to choose explicitly. JPEG 2000 transfer syntaxes are decoded.
- `--roi_subset name[,name...]` (repeatable) restricts the run to those
  structures and uses the crop cascade (6-mm localizer, crop, segment); names are
  the keys of `models/total_label_map.json` (`spleen`, `liver`, `kidney_left`,
  ...). Without it the full 117-label segmentation is written.
- `--fast` uses the 3-mm model instead of the five 1.5-mm models.

## Outputs, exactly as written

With `-o DIR` (default mode): `DIR/<structure>.nii.gz`, one binary mask per
structure present in the result (restricted to `--roi_subset` when given), uint8,
in the input geometry. With `-o DIR --ml`: `DIR/total.nii.gz`, one multilabel
file whose values are the ids in `total_label_map.json`. With `-o file.nii.gz`:
that single multilabel file.

`--statistics` is the machine-readable result. `--statistics -` prints it to
stdout (add `--quiet`, since progress lines also go to stdout); `--statistics`
alone writes `statistics.json` in `DIR` (or beside the single output file);
`--statistics PATH` writes to PATH. One JSON object:

```
{
  "input": "scan.nii.gz", "output": "out", "tier": "full", "model": "full",
  "int8": true, "structures_requested": ["spleen", "liver"],
  "voxel_ml": 0.003375, "n_structures": 2,
  "structures": {
    "spleen": { "label": 1, "mask": "out/spleen.nii.gz", "voxel_count": 71234,
                "volume_ml": 240.415, "intensity_mean": 98.21, "intensity_median": 99.0 },
    "liver":  { ... }
  }
}
```

Only structures with at least one voxel appear. `volume_ml` is voxel count times
voxel volume (`voxel_ml`); `intensity_*` are mean and median HU of the input
inside the mask. `mask` is null when no file was written for that structure.
There is no CSV output.

`--preview` writes `preview_<task>.png` (a contact sheet with all labels
overlaid) in the same place as `statistics.json`, or to `--preview PATH`.
`--detections-json` writes `detections.json` when component filtering
(`--min-component-volume-mm3`) is used.

The GUI writes `output/<scan name>/` next to its executable with the masks and
`preview_total.png`; its volume table is shown in the window only, not saved.

Progress lines go to stdout (silenced by `--quiet`); errors go to stderr.

## Exit codes

- 0: success.
- 2: usage error (missing `--in` or `--out`, input path not found, invalid
  option values); message on stderr starts with `error:`.
- 1: any other failure (model or library not found, ONNX Runtime error, bad
  input file); message starts with `Error:`.

## Environment variables that change results

- `ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION=f32` keeps the non-quantized
  operations in fp32; the manuscript's numbers were produced with it. Unset, a
  BF16-capable CPU may compute them in BF16. Set it when reproducing the paper.
- `ONNX_TOTALSEG_OPENVINO_THREADS=N` limits OpenVINO inference threads;
  `--threads N` sets ONNX Runtime's intra-op threads.
- `ONNX_TOTALSEG_OPENVINO_CACHE_DIR=DIR` caches compiled models between runs.
- `ONNX_TOTALSEG_MODELS`, `ORT_DYLIB_PATH`: see above.

## Do not

- Do not look for a GPU. The INT8 path is CPU-only by design; `--device cuda`
  and `--device directml` exist for the fp32 models only.
- Do not re-quantize or re-export models to use the tool; download the released
  INT8 set. `scripts/` reproduce the manuscript's quantization and need CT-ORG
  volumes and the fp32 models.
- Do not pass `--int8` without `--device openvino`, and do not expect
  `--device auto` to use the INT8 models.
- Do not commit models, scans or masks; `.gitignore` excludes `models/`,
  `*.onnx`, `*.nii`, `*.nii.gz`, `*.dcm` and `*.zip`.
- Do not time the GUI; it adds process start-up around the CLI. Time the CLI,
  one process per scan.
