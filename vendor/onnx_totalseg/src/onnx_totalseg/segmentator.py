"""Segmentator: dependency-light ONNX runtime reproduction of TotalSegmentator inference.

Lifecycle is split so a future persistent wrapper avoids repeated cold start:
  seg = Segmentator(model_card)         # cold start: ORT session + weights
  labels = seg.predict_file(in, out)    # warm: callable many times
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import onnxruntime as ort

from . import io_nibabel
from .ep import resolve_providers
from ._nnunet_ops import (
    change_spacing_to_shape,
    change_spacing_to_spacing,
    compute_gaussian,
    compute_new_shape,
    compute_steps_for_sliding_window,
    crop_to_nonzero,
    ct_normalize,
    insert_crop_into_image,
    pad_nd_image,
    resample_data_or_seg_to_shape,
    resample_label_to_shape,
)


class Segmentator:
    def __init__(self, model_card_path: str,
                 providers: Optional[Sequence[str]] = None,
                 device: str = "auto",
                 intra_op_threads: Optional[int] = None,
                 resample_order: Optional[int] = None,
                 fp16: bool = False,
                 device_id: Optional[int] = None):
        card = json.loads(Path(model_card_path).read_text())
        self.card = card
        self.root = Path(model_card_path).parent
        self.patch_size: List[int] = [int(x) for x in card["patch_size"]]
        self.target_spacing: List[float] = [float(x) for x in card["target_spacing"]]
        self.transpose_forward = list(card["transpose_forward"])
        self.transpose_backward = list(card["transpose_backward"])
        self.num_seg_heads = int(card["num_seg_heads"])
        n = card["normalization"]
        self.norm = (float(n["mean"]), float(n["std"]),
                     float(n["lower_bound"]), float(n["upper_bound"]))
        rk = card["resampling_data_kwargs"]
        self.data_order = int(rk["order"])
        self.data_order_z = int(rk["order_z"])
        self.data_force_sep_z = rk["force_separate_z"]
        # image->target downsample order: card default (3, exact/stock) unless overridden.
        # order=1 (trilinear) is ~300x cheaper for downsampling; tiny train/test shift.
        self.image_resample_order = resample_order if resample_order is not None else self.data_order
        inf = card["inference"]
        self.tile_step_size = float(inf["tile_step_size"])
        self.use_gaussian = bool(inf["use_gaussian"])
        self.sigma_scale = float(inf["gaussian_sigma_scale"])
        # label id -> name, for optional per-structure export
        self.labels = {int(v): k for k, v in card["labels"].items()}

        so = ort.SessionOptions()
        if intra_op_threads is not None:
            so.intra_op_num_threads = intra_op_threads
        # providers wins if given explicitly; otherwise resolve from friendly device name
        if providers is None:
            providers = resolve_providers(device, device_id)
        onnx_path = self.root / card["onnx_file"]
        if fp16:
            fp16_path = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
            if fp16_path.exists():
                onnx_path = fp16_path
            else:
                print(f"[fp16] variant {fp16_path.name} not found; using fp32")
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=list(providers))
        self.requested_providers = list(providers)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

    # ---- core pipeline ----------------------------------------------------------
    def _sliding_window(self, data: np.ndarray) -> np.ndarray:
        patch = self.patch_size
        data, slicer_revert = pad_nd_image(
            data, patch, "constant", {"constant_values": 0}, return_slicer=True)
        steps = compute_steps_for_sliding_window(data.shape[1:], patch, self.tile_step_size)
        if self.use_gaussian:
            gaussian = compute_gaussian(tuple(patch), self.sigma_scale, value_scaling_factor=1.0)
        else:
            gaussian = 1.0
        logits = np.zeros((self.num_seg_heads, *data.shape[1:]), dtype=np.float32)
        counts = np.zeros(data.shape[1:], dtype=np.float32)
        pz, py, px = patch
        for z in steps[0]:
            for y in steps[1]:
                for x in steps[2]:
                    sl = (slice(z, z + pz), slice(y, y + py), slice(x, x + px))
                    patch_in = data[(slice(None), *sl)][None].astype(np.float32, copy=False)
                    pred = self.session.run(["logits"], {self.input_name: patch_in})[0][0]
                    if self.use_gaussian:
                        pred = pred * gaussian
                    logits[(slice(None), *sl)] += pred
                    counts[sl] += gaussian
        logits /= counts
        return logits[(slice(None), *slicer_revert[1:])]

    def predict_array(self, data: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
        """data: (1, x, y, z) in sitk-style order; spacing: matching 3-tuple."""
        tf = self.transpose_forward
        data = data.astype(np.float32, copy=True)
        data = data.transpose([0, *[i + 1 for i in tf]])
        original_spacing = [spacing[i] for i in tf]

        shape_before_cropping = data.shape[1:]
        data, _seg, bbox = crop_to_nonzero(data)
        shape_after_crop = data.shape[1:]

        new_shape = compute_new_shape(data.shape[1:], original_spacing, self.target_spacing)
        # normalize (before resampling, per nnU-Net), then resample data (order 3)
        mean, std, lower, upper = self.norm
        data[0] = ct_normalize(data[0], mean, std, lower, upper)
        data = resample_data_or_seg_to_shape(
            data, new_shape, original_spacing, self.target_spacing,
            is_seg=False, order=self.data_order, order_z=self.data_order_z,
            force_separate_z=self.data_force_sep_z)

        logits = self._sliding_window(data)

        # Argmax at network (3 mm) resolution, THEN nearest-resample the label map back to
        # the cropped-original shape. This matches stock TS (nnunet_resampling=False) and
        # avoids materializing full-resolution 118-channel logits (tens of GiB).
        seg_net = logits.argmax(0).astype(np.uint8)
        seg = resample_label_to_shape(seg_net, shape_after_crop)

        full = np.zeros(shape_before_cropping, dtype=np.uint8)
        full = insert_crop_into_image(full, seg, bbox)
        full = full.transpose(self.transpose_backward)
        return full

    def predict_loaded(self, data: np.ndarray, props: dict, lowres: bool = False,
                       restrict_ids=None):
        """Predict from an already-loaded (data[1,z,y,x], props). Returns
        (seg[z,y,x] uint8, canonical_affine|None). Resample image -> target spacing (scipy
        zoom, int32 cast like TS), run the nnU-Net core, then upsample -> original shape
        (default) or keep at the model grid (lowres, the --ultrafast path).

        restrict_ids is accepted for API parity with MultiModelSegmentator (which uses it to
        skip sub-models) but ignored here: a single model has nothing to skip."""
        spacing = props["spacing"]
        orig_shape = data.shape[1:]
        img_t = change_spacing_to_spacing(
            data[0], spacing, self.target_spacing, order=self.image_resample_order)
        img_t = img_t.astype(np.int32).astype(np.float32)
        seg_t = self.predict_array(img_t[None], self.target_spacing)
        if lowres:
            return seg_t.astype(np.uint8), io_nibabel.lowres_affine(props, self.target_spacing)
        seg = change_spacing_to_shape(seg_t, orig_shape, order=0).astype(np.uint8)
        return seg, None

    def close(self):
        """Release the ORT session (and its GPU arena). Used to free the one-shot crop-cascade
        localizer once the ROI bbox is known, so peak VRAM = one target model instead of
        localizer + target. Do NOT call on a cached/shared session (e.g. the warm server)."""
        self.session = None
        gc.collect()

    def predict_file(self, in_path: str, out_path: Optional[str] = None,
                     lowres: bool = False) -> np.ndarray:
        data, props = io_nibabel.read_image(in_path)
        seg, canonical_affine = self.predict_loaded(data, props, lowres=lowres)
        if out_path is not None:
            io_nibabel.write_seg(seg, out_path, props, canonical_affine=canonical_affine)
        return seg
