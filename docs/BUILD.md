# Building and running ts-portable

## Build

```
cargo build --release
```

This produces two binaries in `target/release/`: `ts-portable-cli` (command line) and
`ts-portable` (the window). Both load ONNX Runtime at run time, so the build needs only Rust.
The INT8 models need the OpenVINO execution provider (`--device openvino --int8`); where its library
comes from depends on the platform. The models are the same everywhere.

### Windows

`deployment\prepare_windows_openvino.ps1` downloads the `onnxruntime-openvino` 1.24.1 and `openvino`
2025.4.1 wheels and stages their DLLs in `dist\ts-portable-openvino\onnxruntime\` next to the
binary. The [`Windows bundle`](../.github/workflows/windows-bundle.yml) workflow does the same on
GitHub Actions when a `v*` tag is pushed and publishes the release zip (binaries, OpenVINO
libraries and the INT8 models). See
[`deployment/WINDOWS_INT8_PROTOCOL.md`](../deployment/WINDOWS_INT8_PROTOCOL.md).

### Linux (x86-64)

The `onnxruntime-openvino` wheel bundles the OpenVINO runtime, so the library can be taken from a
Python environment:

```
python3 -m venv .ovenv && .ovenv/bin/pip install onnxruntime-openvino==1.24.1
ORT=$(ls -d .ovenv/lib/python3.*/site-packages/onnxruntime/capi)
./target/release/ts-portable-cli -i scan.nii.gz -o output/ --device openvino --int8 \
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

### NVIDIA GPU (CUDA)

The fp32 models run on NVIDIA GPUs through ONNX Runtime's CUDA provider (`--device cuda`). Besides the
NVIDIA driver, it needs the GPU build of ONNX Runtime and the CUDA 12 and cuDNN 9 libraries; all of
them come from pip, so no CUDA toolkit install is needed:

```
python -m venv cuda-env
cuda-env/bin/pip install onnxruntime-gpu==1.22.0 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
    nvidia-cufft-cu12 nvidia-cudnn-cu12
```

On Linux, put the NVIDIA `lib` folders on the library path and point `--ort-lib` at the GPU build:

```
SP=$(echo cuda-env/lib/python3.*/site-packages)
export LD_LIBRARY_PATH=$(echo $SP/nvidia/*/lib | tr ' ' :)
ts-portable-cli -i scan.nii.gz -o output/ --models-dir fp32 --device cuda \
    --ort-lib $SP/onnxruntime/capi/libonnxruntime.so.1.22.0
```

On Windows (PowerShell; the pip command is `cuda-env\Scripts\pip`), the DLLs are in `bin` folders:

```
$sp = "cuda-env\Lib\site-packages"
$env:PATH = ((Get-ChildItem "$sp\nvidia\*\bin").FullName -join ';') + ";$env:PATH"
.\ts-portable-cli.exe -i scan.nii.gz -o output --models-dir fp32 --device cuda --ort-lib "$sp\onnxruntime\capi\onnxruntime.dll"
```

DirectML (`--device directml`, with the `onnxruntime-directml` library) runs the fp32 models on other
Windows GPUs. The INT8 path needs no GPU.

## Running

### Where things are found

- **Models:** `--models-dir`, else the `TS_PORTABLE_MODELS` environment variable, else a `models`
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

The manuscript's results were produced with `TS_PORTABLE_OPENVINO_INFERENCE_PRECISION=f32`, which
keeps the non-quantized operations in fp32. Without it, OpenVINO may run them in BF16 on CPUs that
support it (recent Xeon and Core Ultra processors), which can change the output slightly.

## Verifying the download

Each release lists the zip's SHA-256 checksum (`.sha256` file); the models' checksums are in
[`deployment/models.sha256`](../deployment/models.sha256) and, inside the release and models zips,
`models/SHA256SUMS.txt`. Releases also carry a GitHub build-provenance attestation, which proves the
zip was built by the `Windows bundle` workflow from the tagged source. With the
[GitHub CLI](https://cli.github.com):

```
gh attestation verify ts-portable-windows.zip --repo sigjhl/ts-portable
```
