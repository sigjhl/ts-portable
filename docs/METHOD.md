# Quantization method

This repository quantizes the released TotalSegmentator ONNX models to 8-bit
integer precision for CPU execution through the OpenVINO execution provider. Two
scripts are provided.

## `scripts/quantize_nncf_sequential.py` — the paper's method

Accuracy-controlled post-training quantization using NNCF
`quantize_with_accuracy_control`. This is the quantizer used to produce the
TS-Portable INT8 models evaluated in the manuscript.

- **Models:** seven — the five 1.5-mm submodels, the 3-mm model, and the 6-mm
  localization model of the TotalSegmentator 2.12.0 `total` task.
- **Precision:** per-channel INT8 weights, UINT8 activations, with any operation
  whose quantization would exceed the accuracy budget restored to fp32.
- **NNCF settings:** `QuantizationPreset.MIXED`, `TargetDevice.CPU`, fast bias
  correction.
- **Accuracy control:** the maximum permitted patch-fidelity drop is `0.005`
  (`--max-drop`), with up to four accuracy-restoration iterations
  (`--max-iterations 4`).
- **CT-ORG data splits** (first 28 volumes), fixed before the final models were
  generated:
  - calibration: `volume-0`–`volume-7`, up to 20 foreground patches per volume
    (160 patches per 1.5-mm model, 66 for the 3-mm model, 44 for the 6-mm model)
  - selection: `volume-8`–`volume-17`, 2 foreground-rich patches per volume
    (20 per model)
  - test: `volume-18`–`volume-27`, 2 patches per volume, held out, never seen by
    NNCF
- **Objective:** mean foreground-label Dice between the candidate and the fp32
  model on the selection patches, background excluded. This is a
  model-selection fidelity objective, not a clinical endpoint or a whole-volume
  estimate. CT-ORG supports no accuracy claim; accuracy was evaluated on AMOS and
  the TotalSegmentator development set.
- **UINT8 re-encoding:** the NNCF loop runs on mainline ONNX Runtime, whose
  candidate graphs use signed INT8 activation zero points that OpenVINO's CPU
  plugin cannot run. The script re-encodes them as equivalent UINT8 zero points
  before writing `<name>_int8.onnx`; the released models were re-checked under
  `onnxruntime-openvino` on the selection and held-out patches.
- **Software:** NNCF 3.2.0, ONNX 1.22.0, ONNX Runtime 1.27.0, Python 3.12.11
  for quantization (see `requirements.txt`); `onnxruntime-openvino` 1.24.1
  with OpenVINO 2025.4.1 for deployed inference.

```
python scripts/quantize_nncf_sequential.py <model_card.json> \
    --vol-dir <CT-ORG/volumes> --max-drop 0.005 --max-iterations 4
```

Run from the repository root so `from scripts.quantize_int8 import ...` resolves.

## `scripts/quantize_int8.py` — simple PTQ baseline and calibration machinery

A plain ORT static-quantization recipe (QDQ, per-channel INT8 weights, UINT8
activations, MinMax calibration). It also provides the calibration and
exclusion helpers (`build_calib`, `extract_patches`, `detect_exclusions`,
`foreground_frac`) imported by the NNCF script.

The recipe, common to both:

- **QDQ format, per-channel INT8 weights, UINT8 activations.** UINT8 activations
  are required by oneDNN's `u8s8` INT8 GEMM; signed activations do not lower and
  produce incorrect output.
- **Excluded operations:** the input-touching first encoder convolution and the
  final segmentation-head convolution are kept in fp32. They are small compute
  but protect thin-margin argmax on many-class heads; excluding them improves
  fidelity at negligible speed cost.
- **Bounded-memory chunked calibration** so large 3D patches do not exhaust RAM.
- Calibration uses each model's own preprocessing on shared CT volumes, so
  activation ranges match deployment.

## Why OpenVINO

The QDQ graph is portable, but only OpenVINO's oneDNN-backed CPU execution
provider lowers it to INT8 convolution kernels for 3D convolutions. Plain
ORT/MLAS keeps the convolutions in fp32 wrapped in quantize/dequantize nodes and
gains nothing from INT8. The Rust runtime loads the quantized graph with
`--device openvino`.

The manuscript's accuracy and runtime results were produced with
`ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION=f32`, which keeps the non-quantized
operations in fp32. Without it, OpenVINO may run them in BF16 on CPUs that
support it; that configuration is not part of the manuscript's results.

## Rerunning the pipeline

1. Install the vendored PyTorch-free preprocessing library
   ([`vendor/onnx_totalseg/`](../vendor/onnx_totalseg/README.md)): `pip install -e vendor/onnx_totalseg`.
2. Install the pipeline dependencies: `pip install -r requirements.txt`.
3. Get the calibration data: the CT-ORG volumes (Rister et al., *Scientific Data* 2020) and the fp32
   ONNX models and cards from the [models repository](https://huggingface.co/sigjhl/ts-portable).
4. From the repository root, for each of the seven model cards:

   ```
   python scripts/quantize_nncf_sequential.py <model_card.json> \
       --vol-dir <path/to/CT-ORG/volumes> --max-drop 0.005 --max-iterations 4
   ```
