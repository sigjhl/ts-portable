# vendor/onnx_totalseg — vendored dependency

`onnx_totalseg` is the PyTorch-free Python preprocessing library for the
TotalSegmentator ONNX models. The quantization scripts in `../../scripts/`
import from it:

```python
from onnx_totalseg import io_nibabel          # NIfTI read/preprocess
from onnx_totalseg._nnunet_ops import ...      # nnU-Net resample/normalize/crop/pad
from onnx_totalseg.segmentator import Segmentator
```

It is vendored here as an installable subtree.

## Provenance

A copy of the `onnx_totalseg` package (the package and its `pyproject.toml`
only; the export toolkit, models, and validation harness are not included). The
commit it was taken from is recorded in [`.source-commit`](.source-commit).
This vendored copy is the distributed version.

## Install

From the repository root:

```
pip install -e vendor/onnx_totalseg
```

This pulls the package's runtime dependencies (numpy, scipy, scikit-image,
nibabel, onnxruntime, pydicom, dicom2nifti, isal). The full quantizer import
chain has been verified to resolve against this tree.

## License

Apache License 2.0, the same license as the rest of this repository (see the
`LICENSE` and `NOTICE` files at the repository root). `_nnunet_ops.py` and
`io_nibabel.py` are ports of nnU-Net code (Apache-2.0, DKFZ), and the package
is derived from TotalSegmentator (Apache-2.0).
