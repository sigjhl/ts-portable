#!/usr/bin/env python3
"""Offline INT8 quantizer for the TotalSegmentator ONNX models.

Produces a `<name>_int8.onnx` QDQ variant that the Rust runtime loads with
`--int8 --device openvino` (OpenVINO's oneDNN backend lowers the QDQ graph to int8
convolution kernels; plain ORT/MLAS gains nothing from int8 for 3-D convs).

Locked recipe (empirically the best simple PTQ for these 3-D U-Nets — see
docs/design/int8-openvino-handoff.md):
  * QDQ format, per-channel int8 weights, **uint8** activations
    (uint8 is required by oneDNN's u8s8 int8 GEMM; signed activations do NOT lower
    and produce garbage).
  * MinMax calibration (percentile is ~10x slower to calibrate and, on CT inputs
    which are already clip-normalized, only redistributes error — no net win).
  * Exclude the input-touching first Conv and the seg-head Conv (feeds a graph
    output). These are tiny compute but protect thin-margin argmax on many-class
    heads; excluding them improves fidelity at ~no speed cost.
  * Bounded-memory chunked calibration (CalibStridedMinMax) so large 3-D patches
    never blow up RAM.

Calibration uses each model's OWN preprocessing on shared CT volumes, so activation
ranges match deployment. Run in an env with `onnx_totalseg`, `nibabel`, `onnxruntime`
(MLAS ORT is fine; the QDQ output is portable and runs under OpenVINO at inference).

Usage:
  quantize_int8.py <model_card.json> [--out PATH] [--vol-dir DIR]
     [--calib-vols v1,v2,...] [--per-vol 20] [--max-calib 160] [--stride 4]

Example:
  python scripts/quantize_int8.py models/total_1p5mm_291_organs.model_card.json \
      --vol-dir /data/CT/volumes --calib-vols volume-28,volume-63,volume-10,volume-45
"""
import argparse
import re
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

from onnx_totalseg import io_nibabel
from onnx_totalseg._nnunet_ops import (change_spacing_to_spacing,
                                       compute_steps_for_sliding_window,
                                       crop_to_nonzero, ct_normalize, pad_nd_image)
from onnx_totalseg.segmentator import Segmentator

# Deterministic, auditable CT-ORG calibration cohort. Keep this sequential split
# stable so regeneration cannot be mistaken for case selection by observed quality.
DEFAULT_VOLS = [f"volume-{i}" for i in range(8)]


# ---- calibration patch extraction (each model's own preprocessing) --------------
def extract_patches(seg, vol_path):
    data, props = io_nibabel.read_image(str(vol_path))
    img = change_spacing_to_spacing(data[0], props["spacing"], seg.target_spacing, order=3)
    img = img.astype(np.int32).astype(np.float32)[None]
    d, _, _ = crop_to_nonzero(img)
    mean, std, lo, hi = seg.norm
    d[0] = ct_normalize(d[0], mean, std, lo, hi)
    dp, _ = pad_nd_image(d, seg.patch_size, "constant", {"constant_values": 0}, True)
    pz, py, px = seg.patch_size
    steps = compute_steps_for_sliding_window(dp.shape[1:], seg.patch_size, 0.5)
    out = []
    for z in steps[0]:
        for y in steps[1]:
            for x in steps[2]:
                out.append(dp[:, z:z + pz, y:y + py, x:x + px][None].astype(np.float32))
    return out


def foreground_frac(p):
    return float((p > (p.min() + 1e-3)).mean())


def build_calib(card, vol_dir, vols, per_vol, max_calib):
    seg = Segmentator(str(card), device="cpu")
    patches = []
    for v in vols:
        vp = Path(vol_dir) / f"{v}.nii.gz"
        if not vp.exists():
            print(f"  [calib] {v}: missing, skip")
            continue
        ps = [p for p in extract_patches(seg, vp) if foreground_frac(p) > 0.03]
        ps.sort(key=foreground_frac, reverse=True)
        patches.extend(ps[:per_vol])
        print(f"  [calib] {v}: kept {min(len(ps), per_vol)}")
    return np.concatenate(patches[:max_calib], 0), seg.input_name


# ---- node selection + quantization ----------------------------------------------
def detect_exclusions(onnx_path):
    g = onnx.load(onnx_path).graph
    gin = {i.name for i in g.input}
    gout = {o.name for o in g.output}
    first = [n.name for n in g.node if n.op_type == "Conv" and any(x in gin for x in n.input)]
    head = [n.name for n in g.node if n.op_type == "Conv" and any(x in gout for x in n.output)]
    return sorted(set(first) | set(head))


class StridedReader(CalibrationDataReader):
    """Yields one patch at a time; set_range enables chunked (bounded-memory) collection."""

    def __init__(self, arr, name):
        self.arr, self.name = arr, name
        self.s, self.e, self.i = 0, len(arr), 0

    def set_range(self, start_index, end_index):
        self.s, self.e, self.i = start_index, min(end_index, len(self.arr)), start_index

    def get_next(self):
        if self.i >= self.e:
            return None
        out = {self.name: self.arr[self.i:self.i + 1]}
        self.i += 1
        return out

    def __len__(self):
        return len(self.arr)


def quantize(fp32, out, calib, name, stride=4):
    exclude = detect_exclusions(fp32)
    n = max(stride, (len(calib) // stride) * stride)
    calib = calib[:n].astype(np.float32)
    quantize_static(
        fp32, out, StridedReader(calib, name),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude,
        extra_options={"CalibStridedMinMax": stride},
    )
    return exclude, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("card", help="path to <name>.model_card.json")
    ap.add_argument("--out", help="output onnx (default <fp32-stem>_int8.onnx next to fp32)")
    ap.add_argument("--vol-dir", default="CT-ORG/volumes")
    ap.add_argument("--calib-vols", default=",".join(DEFAULT_VOLS))
    ap.add_argument("--per-vol", type=int, default=20)
    ap.add_argument("--max-calib", type=int, default=160)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()

    import json
    card = Path(a.card)
    fp32 = card.parent / json.loads(card.read_text())["onnx_file"]
    out = Path(a.out) if a.out else fp32.with_name(fp32.stem + "_int8.onnx")
    vols = [v for v in a.calib_vols.split(",") if v]

    print(f"[quantize_int8] {fp32.name} -> {out.name}")
    calib, name = build_calib(card, a.vol_dir, vols, a.per_vol, a.max_calib)
    exclude, n = quantize(str(fp32), str(out), calib, name, a.stride)
    mb = out.stat().st_size / 1e6
    print(f"[quantize_int8] done: calib={n} excluded={len(exclude)} "
          f"({', '.join(exclude)}) size={mb:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
