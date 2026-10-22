"""Numpy-only ports of the exact nnU-Net v2.6.4 preprocessing / postprocessing ops.

These are faithful reimplementations (copied + de-torched) of nnU-Net and acvl_utils
functions so the runtime reproduces stock TotalSegmentator output WITHOUT importing
torch or nnunetv2. Vendored from:
  - nnunetv2.preprocessing.resampling.default_resampling
  - nnunetv2.preprocessing.normalization.default_normalization_schemes (CTNormalization)
  - nnunetv2.preprocessing.cropping.cropping
  - nnunetv2.inference.sliding_window_prediction (compute_gaussian, steps)
  - nnunetv2.imageio.nibabel_reader_writer (NibabelIOWithReorient)
  - acvl_utils.cropping_and_padding.{bounding_boxes,padding}
Original code: Apache License 2.0, DKFZ / HIP Applied Computer Vision Lab.

Only the inference paths are ported (is_seg=False resampling). Segmentation resampling
(is_seg=True, needs batchgenerators) is intentionally omitted.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, map_coordinates
from scipy.ndimage import zoom as ndi_zoom
from skimage.transform import resize

ANISO_THRESHOLD = 3  # nnunetv2.configuration.ANISO_THRESHOLD


# ----------------------------------------------------------------------------------
# TS-style outer resampling (totalsegmentator.resampling.change_spacing -> resample_img)
# scipy.ndimage.zoom(mode="nearest"), NOT skimage.resize. The grid convention differs and
# must match stock TS exactly, otherwise thin structures shift by ~1-2 voxels.
# ----------------------------------------------------------------------------------
def change_spacing_to_spacing(data3d: np.ndarray, current_spacing, target_spacing,
                              order: int) -> np.ndarray:
    """Resample a 3D array from current_spacing to target_spacing (TS change_spacing,
    no target_shape branch: zoom = current/target)."""
    zoom = np.array(current_spacing, float) / np.array(target_spacing, float)
    return ndi_zoom(data3d, zoom, order=order, mode="nearest")


def change_spacing_to_shape(data3d: np.ndarray, target_shape, order: int) -> np.ndarray:
    """Resample a 3D array to an exact target_shape (TS change_spacing target_shape branch:
    zoom = target_shape / old_shape)."""
    target_shape = tuple(int(x) for x in target_shape)
    zoom = np.array(target_shape, float) / np.array(data3d.shape, float)
    out = ndi_zoom(data3d, zoom, order=order, mode="nearest")
    if out.shape != target_shape:  # guard against zoom rounding (±1)
        fixed = np.zeros(target_shape, dtype=out.dtype)
        sl = tuple(slice(0, min(o, t)) for o, t in zip(out.shape, target_shape))
        fixed[sl] = out[sl]
        out = fixed
    return out


# ----------------------------------------------------------------------------------
# Resampling (default_resampling.py) — inference paths only (is_seg=False)
# ----------------------------------------------------------------------------------
def get_do_separate_z(spacing, anisotropy_threshold=ANISO_THRESHOLD) -> bool:
    return (np.max(spacing) / np.min(spacing)) > anisotropy_threshold


def get_lowres_axis(new_spacing):
    return np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]


def compute_new_shape(old_shape, old_spacing, new_spacing) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    return np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])


def determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        axis = get_lowres_axis(current_spacing) if force_separate_z else None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z, axis = False, None
        elif len(axis) == 2:
            do_separate_z, axis = False, None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_data_or_seg(data: np.ndarray, new_shape, is_seg: bool = False, axis=None,
                         order: int = 3, do_separate_z: bool = False, order_z: int = 0,
                         dtype_out=None) -> np.ndarray:
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1
    if is_seg:
        raise NotImplementedError("is_seg=True resampling not needed for inference")
    resize_fn = resize
    kwargs = {"mode": "edge", "anti_aliasing": False}

    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if np.any(shape != new_shape):
        data = data.astype(float, copy=False)
        if do_separate_z:
            assert axis is not None
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                if shape[axis] != new_shape[axis]:
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim
                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5
                    coord_map = np.array([map_rows, map_cols, map_dims])
                    reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode="nearest")[None]
                else:
                    reshaped_final[c] = reshaped_here
        else:
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        return reshaped_final
    return data


def resample_label_to_shape(seg: np.ndarray, new_shape) -> np.ndarray:
    """Nearest-neighbour resample of an integer label map to new_shape.

    Matches stock TS's default fast upsample (change_spacing order=0) used to bring the
    3 mm segmentation back to original resolution. order=0 resize assigns exact label
    values (no interpolation), so the round-to-int is lossless.
    """
    if tuple(seg.shape) == tuple(new_shape):
        return seg
    out = resize(seg.astype(np.float32), tuple(int(x) for x in new_shape),
                 order=0, mode="edge", anti_aliasing=False)
    return np.rint(out).astype(np.uint8)


def resample_data_or_seg_to_shape(data: np.ndarray, new_shape, current_spacing, new_spacing,
                                  is_seg: bool = False, order: int = 3, order_z: int = 0,
                                  force_separate_z=False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD) -> np.ndarray:
    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold)
    assert data.ndim == 4, "data must be c x y z"
    return resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)


# ----------------------------------------------------------------------------------
# CT normalization (default_normalization_schemes.CTNormalization)
# ----------------------------------------------------------------------------------
def ct_normalize(image: np.ndarray, mean: float, std: float, lower: float, upper: float,
                 target_dtype=np.float32) -> np.ndarray:
    image = image.astype(target_dtype, copy=False)
    np.clip(image, lower, upper, out=image)
    image -= mean
    image /= max(std, 1e-8)
    return image


# ----------------------------------------------------------------------------------
# Cropping (cropping.py + acvl_utils.bounding_boxes)
# ----------------------------------------------------------------------------------
def bounding_box_to_slice(bounding_box: List[List[int]]):
    return tuple(slice(*i) for i in bounding_box)


def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    for z in range(Z):
        if np.any(mask[z]):
            minzidx = z
            break
    for z in range(Z - 1, -1, -1):
        if np.any(mask[z]):
            maxzidx = z + 1
            break
    for x in range(X):
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in range(X - 1, -1, -1):
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break
    for y in range(Y):
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in range(Y - 1, -1, -1):
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    assert data.ndim in (3, 4)
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data: np.ndarray, nonzero_label: int = -1):
    """Returns (cropped_data, cropped_nonzero_seg, bbox)."""
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    slicer = (slice(None),) + slicer
    data = data[slicer]
    seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox


def _int_bbox(bbox):
    return [[int(i) for i in row] for row in bbox]


def insert_crop_into_image(image: np.ndarray, crop: np.ndarray, bbox: List[List[int]]) -> np.ndarray:
    bbox = _int_bbox(bbox)
    num_dims = image.ndim
    bbox_dims = len(bbox)
    leading = num_dims - bbox_dims
    for i in range(bbox_dims):
        mn, mx = bbox[i]
        if mx <= 0 or mn >= image.shape[leading + i]:
            return image
    image_slices, crop_slices = [], []
    for i in range(num_dims):
        if i < leading:
            image_slices.append(slice(None))
            crop_slices.append(slice(None))
        else:
            mn, mx = bbox[i - leading]
            crop_start = max(0, -mn)
            image_start = max(0, mn)
            image_end = min(mx, image.shape[i])
            crop_end = crop_start + (image_end - image_start)
            image_slices.append(slice(image_start, image_end))
            crop_slices.append(slice(crop_start, crop_end))
    image[tuple(image_slices)] = crop[tuple(crop_slices)]
    return image


# ----------------------------------------------------------------------------------
# Sliding window helpers (sliding_window_prediction.py + acvl_utils.padding)
# ----------------------------------------------------------------------------------
def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 1.0,
                     dtype=np.float32) -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    g = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    g = g / (np.max(g) / value_scaling_factor)
    g = g.astype(dtype)
    g[g == 0] = np.min(g[g != 0])
    return g


def compute_steps_for_sliding_window(image_size, tile_size, tile_step_size: float) -> List[List[int]]:
    assert all(i >= j for i, j in zip(image_size, tile_size)), \
        "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1
                 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]
    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        actual = max_step_value / (num_steps[dim] - 1) if num_steps[dim] > 1 else 99999999999
        steps.append([int(np.round(actual * i)) for i in range(num_steps[dim])])
    return steps


def pad_nd_image(image: np.ndarray, new_shape=None, mode: str = "constant", kwargs: dict = None,
                 return_slicer: bool = False):
    if kwargs is None:
        kwargs = {}
    old_shape = np.array(image.shape)
    if new_shape is not None and len(new_shape) < len(image.shape):
        new_shape = list(image.shape[:len(image.shape) - len(new_shape)]) + list(new_shape)
    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]
    difference = np.array(new_shape) - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]
    if not (all(i == 0 for i in pad_below) and all(i == 0 for i in pad_above)):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image
    if not return_slicer:
        return res
    pad_arr = np.array(pad_list)
    pad_arr[:, 1] = np.array(res.shape) - pad_arr[:, 1]
    slicer = tuple(slice(*i) for i in pad_arr)
    return res, slicer
