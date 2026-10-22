"""Execution-provider selection for 'any GPU' portability.

The exported .onnx is EP-agnostic: the SAME file runs on whichever ORT execution provider
the deployment's onnxruntime build exposes. "Any GPU" is therefore a packaging + selection
concern, not a re-export:

    NVIDIA discrete .......... CUDAExecutionProvider     (pip onnxruntime-gpu;   Linux/Win)
    AMD discrete ............. ROCMExecutionProvider      (pip onnxruntime-rocm;  Linux)
    ANY DX12 GPU on Windows .. DmlExecutionProvider       (pip onnxruntime-directml)
        -> NVIDIA + AMD + Intel/AMD iGPUs, vendor-agnostic
    Intel CPU/iGPU/GPU ....... OpenVINOExecutionProvider  (pip onnxruntime-openvino)
    Apple .................... CoreMLExecutionProvider     (pip onnxruntime-silicon)
    fallback ................. CPUExecutionProvider        (always present)
"""
from __future__ import annotations

from typing import List, Optional

import onnxruntime as ort

# friendly device name -> ORT execution provider
_DEVICE_TO_EP = {
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

# auto preference: discrete GPU, then vendor-agnostic (DirectML), then accelerators, then Apple
_AUTO_ORDER = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
]


# EPs that accept a {"device_id": N} option to pick a specific adapter. Needed on
# multi-GPU boxes (e.g. Intel iGPU + NVIDIA T400) so DirectML targets the right card
# instead of defaulting to adapter 0 (often the iGPU).
_DEVICE_ID_EPS = {"CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider"}

# Memory-frugal EP options. arena_extend_strategy=kSameAsRequested stops the CUDA/ROCm BFC
# arena from doubling its reservation (kNextPowerOfTwo default), cutting a full 1.5 mm
# sub-model's peak VRAM ~2.34 GB -> ~1.79 GB (-24%) at NO measured speed cost — so ours fits
# a small card with room to spare and stays flat regardless of ROI size. DirectML has its own
# allocator (no arena knob); its Windows VRAM is measured separately (E13).
_MEM_FRUGAL_OPTS = {
    "CUDAExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"},
    "ROCMExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"},
}


def _ep_opts(ep, device_id):
    """Wrap a GPU EP as (name, opts) with memory-frugal + optional device_id options; pass
    EPs with no options (e.g. CPU) through unchanged."""
    opts = dict(_MEM_FRUGAL_OPTS.get(ep, {}))
    if device_id is not None and ep in _DEVICE_ID_EPS:
        opts["device_id"] = int(device_id)
    return (ep, opts) if opts else ep


def resolve_providers(device: str = "auto", device_id: Optional[int] = None):
    """Return an ordered ORT providers list for the requested device.

    device: 'auto' (best available GPU EP, else CPU), 'cpu', or a vendor name
    (cuda/rocm/directml/openvino/coreml). A GPU EP is always followed by CPU fallback.
    device_id: adapter index for the GPU EP (DirectML/CUDA/ROCm). None = default adapter.
    When set, GPU EPs are returned as (name, {'device_id': N}) tuples for InferenceSession.
    """
    avail = set(ort.get_available_providers())
    device = (device or "auto").lower()
    if device == "mps":          # TS name for Apple GPU
        device = "coreml"
    if device == "gpu":          # TS name -> best available GPU, else CPU
        device = "auto"
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "auto":
        chosen = [ep for ep in _AUTO_ORDER if ep in avail]
        return [_ep_opts(ep, device_id) for ep in chosen] + ["CPUExecutionProvider"]
    ep = _DEVICE_TO_EP.get(device)
    if ep is None:
        raise ValueError(
            f"unknown device '{device}'. Use auto/cpu/cuda/rocm/directml/openvino/coreml.")
    if ep == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    if ep not in avail:
        raise RuntimeError(
            f"requested device '{device}' -> {ep} not available in this onnxruntime build. "
            f"Available: {sorted(avail)}. Install the matching onnxruntime package "
            f"(e.g. onnxruntime-gpu for cuda, onnxruntime-directml for directml).")
    return [_ep_opts(ep, device_id), "CPUExecutionProvider"]
