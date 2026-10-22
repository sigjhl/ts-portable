"""One-shot CLI / plugin entry point — TS-compatible for the CT `total` workflow.

Speed tiers (default = full 1.5 mm): --fast (3 mm), --fastest (6 mm, = stock TS --fastest),
--ultrafast (6 mm, order-1, no upsample; region/volume only, NOT HU).

TS-familiar usage works as-is: `-i` NIfTI / DICOM dir / DICOM .zip; `-o file.nii.gz` is a
multilabel file (our default), `-o dir/` writes per-structure masks (TS-style) + `--ml`
forces multilabel; `--roi_subset`, `--statistics` (volume + intensity HU), `--device
gpu/cpu/mps`, `--nr_thr_resamp/--nr_thr_saving`, `--task total`, `--quiet`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from . import io_nibabel, registry, roi as roi_mod
from .ep import resolve_providers
from .full_segmentator import MultiModelSegmentator
from .report import compute_statistics
from .segmentator import Segmentator

TIERS = {"full": ("full", None, False), "fast": ("fast", None, False),
         "fastest": ("fastest", None, False), "ultrafast": ("fastest", 1, True)}
CASCADE_TIERS = {"full", "fast"}


def _build(model, order, providers, device, threads, fp16=False, device_id=None):
    if registry.is_multimodel(model):
        return MultiModelSegmentator(registry.full_cards(), registry.label_map(),
                                     providers=providers, device=device,
                                     intra_op_threads=threads, fp16=fp16, device_id=device_id)
    card = os.environ.get("ONNX_TOTALSEG_CARD") or registry.resolve_card(model)
    return Segmentator(card, providers=providers, device=device,
                       intra_op_threads=threads, resample_order=order, fp16=fp16,
                       device_id=device_id)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="onnx-totalseg",
                                 description="ONNX Runtime TotalSegmentator (CT total)")
    ap.add_argument("-i", "--in", dest="inp", required=True,
                    help="CT NIfTI, or a directory / .zip of DICOM slices")
    ap.add_argument("-o", "--out", dest="out", required=True,
                    help="output: .nii.gz = multilabel file; a directory = per-structure masks")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", dest="tier", action="store_const", const="fast", help="3 mm model")
    g.add_argument("--fastest", dest="tier", action="store_const", const="fastest",
                   help="6 mm model (= stock TS --fastest)")
    g.add_argument("--ultrafast", dest="tier", action="store_const", const="ultrafast",
                   help="6 mm, order-1, no upsample: region/volume only, NOT HU")
    ap.set_defaults(tier="full")
    ap.add_argument("--ml", action="store_true", help="force a single multilabel file")
    ap.add_argument("--roi_subset", "--roi", dest="roi", default=None,
                    help="comma-separated structures to keep (crop-cascade on full/fast)")
    ap.add_argument("--statistics", "--report", dest="statistics", nargs="?", const="<auto>",
                    default=None, help="write per-organ volume (mL) + intensity (HU) JSON")
    ap.add_argument("--task", default="total", help="only 'total' (CT) is supported")
    ap.add_argument("--device", default="auto",
                    help="auto (default) | gpu | cpu | mps | cuda | directml | openvino | coreml")
    ap.add_argument("--providers", default=None, help="explicit comma-separated ORT EPs")
    ap.add_argument("--gpu-id", "--device_id", dest="gpu_id", type=int, default=None,
                    help="GPU adapter index for DirectML/CUDA (multi-GPU boxes, e.g. iGPU+dGPU); default adapter 0")
    ap.add_argument("--threads", "--nr_thr_resamp", dest="threads", type=int, default=None,
                    help="inference (ORT) threads; default ORT auto")
    ap.add_argument("--nr_thr_saving", dest="save_threads", type=int,
                    default=min(8, os.cpu_count() or 8), help="threads for per-structure saving")
    ap.add_argument("--fp16", action="store_true",
                    help="force fp16 (default already on GPU); ignored on CPU")
    ap.add_argument("--fp32", action="store_true",
                    help="force fp32 even on GPU (default is fp16 on GPU, fp32 on CPU)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--resample-order", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    def say(*a):
        if not args.quiet:
            print(*a)

    if args.task != "total":
        print(f"error: task '{args.task}' not supported — this build is CT 'total' only "
              f"(tiers: default/--fast/--fastest/--ultrafast).", file=sys.stderr)
        return 2
    if not Path(args.inp).exists():
        print(f"error: input not found: {args.inp}", file=sys.stderr)
        return 2

    # DICOM .zip -> extract to a temp dir
    tmp = None
    inp = args.inp
    if inp.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(inp) as zf:
            zf.extractall(tmp)
        inp = tmp

    model, order, lowres = TIERS[args.tier]
    if args.model:
        model = args.model
    if args.resample_order is not None:
        order = args.resample_order
    providers = args.providers.split(",") if args.providers else None
    try:
        roi_ids = roi_mod.resolve_roi(args.roi.split(",")) if args.roi else None
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Precision is device-matched by default (matches stock TS): fp16 on GPU, fp32 on CPU.
    # --fp32 forces fp32 even on GPU; --fp16 is explicit (already the GPU default), ignored on CPU.
    effective = providers or resolve_providers(args.device)
    is_gpu = bool(effective) and effective[0] != "CPUExecutionProvider"
    fp16 = is_gpu and not args.fp32
    if args.fp16 and not is_gpu:
        print("[fp16] no GPU provider active — running fp32 (fp16 regresses on CPU)", file=sys.stderr)

    t0 = time.time()
    seg_model = _build(model, order, providers, args.device, args.threads, fp16=fp16, device_id=args.gpu_id)
    use_cascade = roi_ids is not None and args.tier in CASCADE_TIERS
    localizer = _build("fastest", 1, providers, args.device, args.threads, fp16=fp16,
                       device_id=args.gpu_id) if use_cascade else None
    t_cold = time.time() - t0

    say(f"tier={args.tier}  provider={seg_model.active_provider}"
        + (f"  roi={args.roi}" if roi_ids else "")
        + ("  (6mm output)" if lowres else "") + ("  (crop-cascade)" if use_cascade else ""))

    t1 = time.time()
    data, props = io_nibabel.read_image(inp)
    if use_cascade:
        # one-shot: free the localizer once the bbox is known so peak VRAM = one target model.
        seg, caff = roi_mod.crop_cascade(data, props, seg_model, localizer, roi_ids,
                                         free_localizer=True), None
    else:
        seg, caff = seg_model.predict_loaded(data, props, lowres=lowres)
        if roi_ids is not None:
            seg = roi_mod.filter_roi(seg, roi_ids)

    _, id2name = roi_mod.total_label_maps()
    out = args.out
    is_nii = out.endswith(".nii") or out.endswith(".nii.gz")
    if args.ml or is_nii:  # multilabel single file (default for a .nii path)
        ml_path = out if is_nii else os.path.join(out, "total.nii.gz")
        os.makedirs(os.path.dirname(ml_path) or ".", exist_ok=True)
        io_nibabel.write_seg(seg, ml_path, props, canonical_affine=caff)
        out_loc = ml_path
    else:  # per-structure masks in a directory (TS-style)
        io_nibabel.write_label_masks(seg, out, props, id2name,
                                     restrict=set(roi_ids) if roi_ids else None,
                                     canonical_affine=caff, threads=args.save_threads)
        out_loc = out
    say(f"wrote {out_loc}  (cold_start={t_cold:.2f}s, predict={time.time()-t1:.2f}s)")

    if args.statistics is not None:
        voxel_ml = float(np.prod([float(s) for s in
                                  (seg_model.target_spacing if lowres else props["spacing"])])) / 1000.0
        stats = json.dumps(compute_statistics(data[0], seg, voxel_ml, id2name), indent=2)
        if args.statistics == "-":
            print(stats)
        else:
            base = out_loc if os.path.isdir(out_loc) else os.path.dirname(out_loc) or "."
            path = base + "/statistics.json" if args.statistics == "<auto>" else args.statistics
            Path(path).write_text(stats)
            say(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
