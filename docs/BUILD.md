# Building and running ts-portable

## Build

```
cargo build --release
```

This produces two binaries in `target/release/`: `onnx-totalseg` (command line) and
`onnx-totalseg-gui` (the window). Both load ONNX Runtime at run time, so the build needs only Rust.
The INT8 models need the OpenVINO execution provider (`--device openvino --int8`); where its library
comes from depends on the platform. The models are the same everywhere.

### Windows

`deployment\prepare_windows_openvino.ps1` downloads the `onnxruntime-openvino` 1.24.1 and `openvino`
2025.4.1 wheels and stages their DLLs in `dist\onnx-totalseg-openvino\onnxruntime\` next to the
binary. The [`Windows bundle`](../.github/workflows/windows-bundle.yml) workflow does the same on
GitHub Actions when a `v*` tag is pushed and publishes the release zip (binaries and OpenVINO
libraries; models are downloaded separately). See
[`deployment/WINDOWS_INT8_PROTOCOL.md`](../deployment/WINDOWS_INT8_PROTOCOL.md).

### Linux (x86-64)

The `onnxruntime-openvino` wheel bundles the OpenVINO runtime, so the library can be taken from a
Python environment:

```
python3 -m venv .ovenv && .ovenv/bin/pip install onnxruntime-openvino==1.24.1
ORT=$(ls -d .ovenv/lib/python3.*/site-packages/onnxruntime/capi)
./target/release/onnx-totalseg -i scan.nii.gz -o output/ --device openvino --int8 \
    --ort-lib "$ORT/libonnxruntime.so.1.24.1" --models-dir models
```

This is the configuration behind the manuscript's workstation CPU results. The window looks for
`onnxruntime/libonnxruntime.so` next to its executable, so copy the `capi` libraries there to use it
on Linux.

### macOS

Intel publishes no `onnxruntime-openvino` wheel for macOS, so the INT8 models cannot run there
(`--int8` requires `--device openvino`). The fp32 ONNX models (the `fp32/` folder of the [models repository](https://huggingface.co/sigjhl/ts-portable)) run through
the plain CPU provider: `pip install onnxruntime`, then `--device cpu` with
`--ort-lib <site-packages>/onnxruntime/capi/libonnxruntime.<version>.dylib` and the fp32 models in
`--models-dir`.

### GPU (optional)

The fp32 models also run through CUDA (`--device cuda` with the `onnxruntime-gpu` library; the
manuscript's GPU arm) and through DirectML on Windows (`--device directml`). The INT8 path needs no
GPU.

## Running

### Where things are found

- **Models:** `--models-dir`, else the `ONNX_TOTALSEG_MODELS` environment variable, else a `models`
  folder under the current working directory. The Windows app always uses the `models` folder next
  to its executable.
- **ONNX Runtime:** `--ort-lib`, else `ORT_DYLIB_PATH`, else the system library path.

### DICOM input

When a folder holds more than one series and you run in a terminal, the CLI lists the series
(patient, protocol, description) and lets you pick one by number. Pass
`--dicom-series <SeriesInstanceUID>` to choose non-interactively; scripts and piped runs fall back to
the largest series. The window asks you to choose.

### Outputs

`--roi_subset <organ>` localizes first, then segments only those regions; use commas or repeat the
option for several organs. `--statistics -` prints, per structure, the mask file, voxel count, volume
in mL and mean and median HU, plus the input, output and tier of the run. `--preview` writes
`preview_<task>.png` beside the masks, with all labels in one image. [`AGENTS.md`](../AGENTS.md)
documents every output path and field.

### Reproducing the paper's numbers

The manuscript's results were produced with `ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION=f32`, which
keeps the non-quantized operations in fp32. Without it, OpenVINO may run them in BF16 on CPUs that
support it (recent Xeon and Core Ultra processors), which can change the output slightly.

## Verifying the download

Each release lists the zip's SHA-256 checksum (`.sha256` file); the models' checksums are in
[`deployment/models.sha256`](../deployment/models.sha256) and, inside the models zip,
`models/SHA256SUMS.txt`. Releases also carry a GitHub build-provenance attestation, which proves the
zip was built by the `Windows bundle` workflow from the tagged source. With the
[GitHub CLI](https://cli.github.com):

```
gh attestation verify ts-portable-windows.zip --repo sigjhl/ts-portable
```
