---
license: apache-2.0
library_name: onnx
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - radiology
  - computed-tomography
  - segmentation
  - totalsegmentator
  - nnu-net
  - onnx
  - int8
  - quantization
  - openvino
---

# TS-Portable models

ONNX models for [TS-Portable](https://github.com/sigjhl/ts-portable), a self-contained program that runs
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) on the CPU. They are the seven models of
the `total` task of TotalSegmentator version 2.12.0 (the five 1.5-mm submodels, the 3-mm model, and the
6-mm localization model), converted to ONNX, and an 8-bit integer (INT8) version of each.

These files are the weights only. To segment a CT scan, use the TS-Portable program from the GitHub
repository; its README walks through the Windows download and the command line.

## Files

| File | Contents | Size |
|---|---|---|
| `ts-portable-models.zip` | INT8 models (`*_int8.onnx`), model cards, label map, `SHA256SUMS.txt`, in a `models/` folder | 172 MB (212 MB unpacked) |
| `fp32/` | the same seven models unquantized (`*.onnx`), model cards, label map, `SHA256SUMS.txt` | 757 MB |

- **INT8 (`ts-portable-models.zip`)** is what TS-Portable uses by default (`--device openvino --int8`).
  It needs the OpenVINO execution provider of ONNX Runtime (`onnxruntime-openvino` 1.24.1) on an x86-64
  CPU; it does not run on macOS. Unzip it next to the program, so that `models/` sits beside the
  executable.
- **fp32 (`fp32/`)** runs through any ONNX Runtime provider: CPU (including macOS), CUDA on NVIDIA GPUs
  (`--device cuda`), or DirectML on Windows (`--device directml`). Point TS-Portable at it with
  `--models-dir fp32`.

Each model card (`*.model_card.json`) records the nnU-Net configuration the program needs: target
spacing, patch size, axis order, intensity normalization, resampling settings, and labels.
`total_label_map.json` maps structure names to TotalSegmentator's label numbers.

## How the INT8 models were made

Accuracy-controlled post-training quantization with NNCF 3.2.0 (`quantize_with_accuracy_control`):
per-channel INT8 weights and UINT8 activations, with any operation whose quantization would cost more
than the accuracy budget restored to fp32 (maximum patch-fidelity drop 0.005, at most four restoration
iterations). The first and last convolution of each model stay in fp32. Calibration, selection and
held-out patch testing used CT-ORG volumes 0–27, which are disjoint from the evaluation data. The script
and the exact settings are in the GitHub repository (`scripts/quantize_nncf_sequential.py`,
`docs/METHOD.md`).

## Evaluation

In the accompanying study, TS-Portable with these INT8 models was compared with the distributed PyTorch
TotalSegmentator package on 100 external CT scans of the AMOS validation set, 15 abdominal organs, at
1.5 mm:

- **Accuracy:** noninferior within a 0.01 margin for per-case mean Dice (difference 0.00017; 95% CI,
  0.00003–0.00031) and 3-mm normalized surface distance (difference 0.00015; 95% CI, −0.00002 to
  0.00031).
- **Speed:** on a 12-core workstation CPU (AMD Ryzen Threadripper PRO 7945WX), the median time per scan
  was 48.0 s, against 342.5 s for the reference implementation on the same CPU and 61.5 s on an NVIDIA
  RTX A6000 GPU.
- **fp32 models:** the unquantized ONNX models run by the same program differed from the reference
  implementation by at most 0.00012 in mean Dice or surface distance.

The per-case results and the code behind every number are in the repository's `evaluation/` folder.

## Intended use and limitations

- For research. TS-Portable is not a medical device and has no regulatory clearance.
- The models are TotalSegmentator's: they reproduce its correct segmentations and its errors. Check
  masks on the source images when a result looks implausible.
- INT8 results depend on the CPU and runtime. The paper's results used
  `ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION=f32`; without it, CPUs with BF16 support may give slightly
  different output.

## Checksums

```
6132bca2025353bb182fc69cbc6baeb526cd254ce242cca50185edb39f0af68d  ts-portable-models.zip
```

Per-file checksums are in `models/SHA256SUMS.txt` inside the zip and in `fp32/SHA256SUMS.txt`; the INT8
ones are also in the GitHub repository (`deployment/models.sha256`).

## License and attribution

Apache License 2.0. The weights are converted and quantized from the `total` task of TotalSegmentator
(Apache-2.0), which is built on nnU-Net (Apache-2.0). Please cite TotalSegmentator:

> Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: robust segmentation of 104 anatomic
> structures in CT images. *Radiol Artif Intell* 2023; 5:e230024.

and the TS-Portable paper (Lee JH, *Quantized TotalSegmentator for Fast Abdominal Organ Segmentation on
CPUs*; the preprint link will be added here when it is posted).
