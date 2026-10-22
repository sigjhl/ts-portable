# ts-portable

Segment organs on a CT scan on an ordinary computer, in seconds. No GPU, no Python, no setup.

ts-portable is an 8-bit integer build of
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) that runs on the CPU. It
accompanies the manuscript *Quantized TotalSegmentator for Fast Abdominal Organ Segmentation on
CPUs*, where it is called **TS-Portable**.

| Abdominopelvic CT, all structures | ts-portable, CPU | TotalSegmentator, CPU | TotalSegmentator, GPU |
|---|---|---|---|
| 1.5 mm | **48 s** | 343 s | 62 s |
| 3 mm | **3.9 s** | 27 s | 16 s |

Median time per scan over 100 abdominopelvic CT scans, including model loading. CPU: AMD Ryzen
Threadripper PRO 7945WX (12 cores); GPU: NVIDIA RTX A6000. On a six-core office PC (Intel Core
i5-10400), segmenting the spleen alone took a median of 5.8 s. Accuracy was noninferior to the
reference implementation (mean Dice difference 0.00017).

## Try it (Windows)

1. Download **[the app](https://github.com/sigjhl/ts-portable/releases/latest/download/ts-portable-windows.zip)**,
   right-click it, choose **Extract All…** and extract it anywhere (the Desktop is fine). Open the
   extracted folders until you see `onnx-totalseg-gui.exe`. The folder that holds it is the
   **app folder**.
2. Download **[the models](https://huggingface.co/sigjhl/ts-portable/resolve/main/ts-portable-models.zip)**
   (172 MB), right-click it and choose **Extract All…**. When Windows asks where to put the files,
   click **Browse…** and choose the app folder. If Windows says some files already exist, choose
   **Replace**. The app folder should now look like this:

   ```
   ts-portable-windows\                      ← the app folder
   ├── onnx-totalseg-gui.exe                 ← the app: double-click this
   ├── onnx-totalseg.exe
   ├── onnxruntime\
   ├── licenses\
   └── models\                               ← from the models download
       ├── total_1p5mm_291_organs_int8.onnx
       ├── ...                               (seven .onnx files in all)
       └── total_label_map.json
   ```

   If `models` ended up one level too deep (for example `ts-portable-models\models\`), drag the
   `models` folder into the app folder.
3. Double-click **`onnx-totalseg-gui.exe`**. If Windows warns about an unrecognized app, click
   **More info → Run anyway** (the app is not code-signed).
4. Open a NIfTI file or a DICOM folder, pick the organs, click **Run**.

Each scan gets an `output/<scan name>/` folder next to the app with the organ masks, an optional
`preview_total.png`, and a table of volume (mL) and mean and median HU.

On macOS or Linux, [build from source](docs/BUILD.md).

## Command line

```
onnx-totalseg -i scan.nii.gz -o output/ --device openvino --int8 --roi_subset spleen,liver --statistics -
```

| Option | Does |
|---|---|
| `--roi_subset a,b` | segment only these organs (faster) |
| `--fast` | 3 mm models instead of 1.5 mm |
| `--statistics` | write volumes and HU to `statistics.json` next to the masks; `--statistics -` prints them to the terminal instead |
| `--preview` | write a PNG overlay next to the masks |

The input can be a NIfTI file or a DICOM folder. `onnx-totalseg --help` lists everything;
[docs/BUILD.md](docs/BUILD.md) covers DICOM series selection, where the models and ONNX Runtime are
found, and the setting that reproduces the paper.

## More

- **[docs/BUILD.md](docs/BUILD.md)**: building on Windows, Linux and macOS; GPU; verifying the download
- **[Models](https://huggingface.co/sigjhl/ts-portable)**: fp32 and INT8 ONNX models,
  model cards and label maps (checksums in [`deployment/models.sha256`](deployment/models.sha256))
- **[docs/METHOD.md](docs/METHOD.md)**: how the INT8 models were made, and how to rerun it
- **[evaluation/](evaluation/README.md)**: the code and per-case results behind every table and figure
- **[AGENTS.md](AGENTS.md)**: every command, output path and field, for scripts and coding agents

## Citation and license

[`evaluation/`](evaluation/README.md) reproduces the paper's results.

See [`CITATION.cff`](CITATION.cff); the full reference will be added on publication.

Apache License 2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Derived from TotalSegmentator
(Wasserthal et al., *Radiology: Artificial Intelligence* 2023) and nnU-Net, both Apache-2.0. The INT8
weights are quantized from the `total` task of TotalSegmentator 2.12.0 and carry the same license.
