#!/usr/bin/env python3
"""Generate one accuracy-controlled UINT8/INT8 ONNX model on fixed CT-ORG splits.

The split is declared independently of observed model behavior:
  calibration: volume-0..7, up to 20 foreground patches per volume
  selection:   volume-8..17, 2 foreground patches per volume
  test:        volume-18..27, 2 foreground patches per volume (saved, never scored here)
"""

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import nncf
import numpy as np
import onnx
import onnxruntime as ort
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters

from scripts.quantize_int8 import build_calib, detect_exclusions, extract_patches, foreground_frac
from onnx_totalseg.segmentator import Segmentator


CALIBRATION_VOLUMES = [f"volume-{i}" for i in range(0, 8)]
SELECTION_VOLUMES = [f"volume-{i}" for i in range(8, 18)]
TEST_VOLUMES = [f"volume-{i}" for i in range(18, 28)]


def patch_dice(pred, ref):
    scores = []
    for label in np.union1d(np.unique(pred), np.unique(ref)):
        if label == 0:
            continue
        p = pred == label
        r = ref == label
        denominator = p.sum() + r.sum()
        if denominator:
            scores.append(float(2 * np.logical_and(p, r).sum() / denominator))
    return float(np.mean(scores)) if scores else float(np.mean(pred == ref))


def make_session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    payload = model.SerializeToString() if isinstance(model, onnx.ModelProto) else str(model)
    return ort.InferenceSession(payload, sess_options=options, providers=["CPUExecutionProvider"])


def validate(model, dataset):
    session = make_session(model)
    input_name = session.get_inputs()[0].name
    values = []
    for inputs, reference in dataset:
        logits = session.run(None, {input_name: inputs[input_name]})[0]
        values.append(patch_dice(logits.argmax(1), reference))
    return float(np.mean(values)), values


def select_patches(segmentator, volume_dir, volumes, per_volume):
    selected = []
    identifiers = []
    counts = {}
    for volume in volumes:
        candidates = extract_patches(segmentator, volume_dir / f"{volume}.nii.gz")
        candidates = [patch for patch in candidates if foreground_frac(patch) > 0.03]
        candidates.sort(key=foreground_frac, reverse=True)
        chosen = candidates[:per_volume]
        counts[volume] = len(chosen)
        for index, patch in enumerate(chosen):
            selected.append(patch)
            identifiers.append(f"{volume}:{index}")
        print(f"  [{volumes[0]}..{volumes[-1]}] {volume}: kept {len(chosen)}", flush=True)
    return selected, identifiers, counts


def quantizer_dtypes(model):
    initializers = {tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    activation_zero_points = []
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and len(node.input) > 2:
            activation_zero_points.append(initializers[node.input[2]].dtype.name)
    return sorted(set(activation_zero_points)), len(activation_zero_points)


def convert_activation_quantizers_to_uint8(model):
    """Re-encode asymmetric INT8 activations as equivalent UINT8 QDQ tensors."""
    initializer_indices = {tensor.name: index for index, tensor in enumerate(model.graph.initializer)}
    quantize_nodes = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    zero_point_names = {node.input[2] for node in quantize_nodes if len(node.input) > 2}
    converted = []
    for name in sorted(zero_point_names):
        index = initializer_indices[name]
        array = onnx.numpy_helper.to_array(model.graph.initializer[index])
        if array.dtype == np.int8:
            shifted = (array.astype(np.int16) + 128).astype(np.uint8)
            model.graph.initializer[index].CopyFrom(onnx.numpy_helper.from_array(shifted, name))
            converted.append(name)

    quantized_outputs = {node.output[0] for node in quantize_nodes}
    for value_info in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
        if (
            value_info.name in quantized_outputs
            and value_info.type.tensor_type.elem_type == onnx.TensorProto.INT8
        ):
            value_info.type.tensor_type.elem_type = onnx.TensorProto.UINT8
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("card", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/tmp/totalseg_nncf_sequential"))
    parser.add_argument("--vol-dir", type=Path,
                        default=Path("CT-ORG/volumes"))
    parser.add_argument("--max-drop", type=float, default=0.005)
    parser.add_argument("--max-iterations", type=int, default=4)
    args = parser.parse_args()

    card = args.card.resolve()
    card_data = json.loads(card.read_text())
    fp32_path = card.parent / card_data["onnx_file"]
    run_root = args.root / fp32_path.stem
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    calibration, input_name = build_calib(
        card, args.vol_dir, CALIBRATION_VOLUMES, per_vol=20, max_calib=160
    )
    calibration_items = [{input_name: calibration[i:i + 1]} for i in range(len(calibration))]

    segmentator = Segmentator(str(card), device="cpu")
    selection_inputs, selection_ids, selection_counts = select_patches(
        segmentator, args.vol_dir, SELECTION_VOLUMES, 2
    )
    fp32 = onnx.load(fp32_path)
    reference_session = make_session(fp32)
    reference_name = reference_session.get_inputs()[0].name
    selection_items = []
    for patch in selection_inputs:
        inputs = {input_name: patch}
        reference = reference_session.run(None, {reference_name: patch})[0].argmax(1)
        selection_items.append((inputs, reference))
    np.save(run_root / "selection_patches.npy", np.concatenate(selection_inputs, axis=0))
    del reference_session, selection_inputs
    gc.collect()

    exclusions = detect_exclusions(fp32_path)
    intermediate = run_root / "intermediate"
    intermediate.mkdir(exist_ok=True)
    accuracy_parameters = AdvancedAccuracyRestorerParameters(
        max_num_iterations=args.max_iterations,
        tune_hyperparams=False,
        ranking_subset_size=min(10, len(selection_items)),
        num_ranking_workers=1,
        intermediate_model_dir=str(intermediate),
    )
    quantized = nncf.quantize_with_accuracy_control(
        fp32,
        nncf.Dataset(calibration_items),
        nncf.Dataset(selection_items),
        validate,
        max_drop=args.max_drop,
        preset=nncf.QuantizationPreset.MIXED,
        target_device=nncf.TargetDevice.CPU,
        subset_size=len(calibration_items),
        fast_bias_correction=True,
        ignored_scope=nncf.IgnoredScope(names=exclusions),
        advanced_accuracy_restorer_parameters=accuracy_parameters,
    )

    converted_zero_points = convert_activation_quantizers_to_uint8(quantized)
    output_model = run_root / f"{fp32_path.stem}_int8.onnx"
    onnx.checker.check_model(quantized)
    onnx.save(quantized, output_model)
    dtypes, quantizer_count = quantizer_dtypes(quantized)
    if dtypes != ["uint8"]:
        raise RuntimeError(f"expected UINT8 activation quantizers, found {dtypes}")

    final_metric, final_values = validate(quantized, selection_items)

    # The test split is materialized only after model selection has finished. It is
    # saved for the separate OpenVINO evaluator and is never passed to NNCF.
    test_inputs, test_ids, test_counts = select_patches(
        segmentator, args.vol_dir, TEST_VOLUMES, 2
    )
    test_array = np.concatenate(test_inputs, axis=0)
    np.save(run_root / "independent_patches.npy", test_array)
    del test_array, test_inputs, segmentator, calibration
    gc.collect()

    domains = sorted({node.domain or "ai.onnx" for node in quantized.graph.node})
    summary = {
        "nncf_version": nncf.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "python": platform.python_version(),
        "card": str(card),
        "fp32_model": str(fp32_path),
        "output_model": str(output_model),
        "split_policy": "sequential CT-ORG volumes declared before model generation",
        "calibration_volumes": CALIBRATION_VOLUMES,
        "calibration_patches": len(calibration_items),
        "selection_volumes": SELECTION_VOLUMES,
        "selection_patch_ids": selection_ids,
        "selection_counts": selection_counts,
        "test_volumes": TEST_VOLUMES,
        "test_patch_ids": test_ids,
        "test_counts": test_counts,
        "metric": "mean patch-level mean foreground-label Dice versus fp32",
        "max_drop": args.max_drop,
        "max_restore_iterations": args.max_iterations,
        "ignored_nodes": exclusions,
        "selection_metric": final_metric,
        "selection_per_patch": final_values,
        "activation_quantizer_dtypes": dtypes,
        "activation_quantizer_count": quantizer_count,
        "converted_activation_zero_points": len(converted_zero_points),
        "domains": domains,
        "model_size_bytes": output_model.stat().st_size,
        "elapsed_s": time.time() - started,
    }
    (run_root / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
