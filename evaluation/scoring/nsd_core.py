"""Surface-area-weighted normalized surface distance (NSD) and HD95.

Requires the Google DeepMind ``surface-distance`` package (version 0.1 for the
paper rerun). That implementation represents surfaces as surfels and weights
each distance by its corresponding surfel area.
"""

from __future__ import annotations

import numpy as np

try:
    from surface_distance import metrics as surface_distance_metrics
except ImportError as exc:  # pragma: no cover - exercised only in a misconfigured environment
    raise ImportError(
        "Install the validated NSD dependency with `python -m pip install surface-distance==0.1`."
    ) from exc


def surface_dice(mask_pred, mask_gt, spacing_mm, tolerance_mm):
    """Return surface-area-weighted surface Dice at ``tolerance_mm``.

    The reference mask must be nonempty. A reference-present, empty prediction
    is assigned 0, matching the manuscript estimand.
    """

    pred = np.asarray(mask_pred, dtype=bool)
    gt = np.asarray(mask_gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape}, reference {gt.shape}")
    if not gt.any():
        raise ValueError("surface_dice requires a nonempty reference mask")
    if not pred.any():
        return 0.0
    distances = surface_distance_metrics.compute_surface_distances(
        mask_gt=gt,
        mask_pred=pred,
        spacing_mm=tuple(float(value) for value in spacing_mm),
    )
    return float(
        surface_distance_metrics.compute_surface_dice_at_tolerance(
            distances,
            tolerance_mm=float(tolerance_mm),
        )
    )


def hd95(mask_pred, mask_gt, spacing_mm):
    """Return the surfel-area-weighted 95th-percentile robust Hausdorff distance (mm).

    Uses the same DeepMind surface representation as :func:`surface_dice`. The
    reference mask must be nonempty. HD95 is undefined for a reference-present,
    empty prediction, so ``None`` is returned (the caller excludes it from the
    HD95 denominator and counts the empty prediction separately).
    """

    pred = np.asarray(mask_pred, dtype=bool)
    gt = np.asarray(mask_gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape}, reference {gt.shape}")
    if not gt.any():
        raise ValueError("hd95 requires a nonempty reference mask")
    if not pred.any():
        return None
    distances = surface_distance_metrics.compute_surface_distances(
        mask_gt=gt,
        mask_pred=pred,
        spacing_mm=tuple(float(value) for value in spacing_mm),
    )
    return float(surface_distance_metrics.compute_robust_hausdorff(distances, 95.0))


def organ_surface_metrics(mask_pred, mask_gt, spacing_mm, tolerances=(1.0, 2.0, 3.0)):
    """Return NSD at each tolerance and HD95 from a single surface computation.

    All metrics share the DeepMind surfel-area representation and physical
    spacing. The reference must be nonempty. A reference-present, empty
    prediction is scored NSD 0 at every tolerance (matching the estimand) with
    HD95 ``None``; ``pred_empty`` flags that case for empty-prediction tallies.
    Returns ``{"nsd": {tolerance: value}, "hd95": value_or_None, "pred_empty": bool}``.
    """

    pred = np.asarray(mask_pred, dtype=bool)
    gt = np.asarray(mask_gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape}, reference {gt.shape}")
    if not gt.any():
        raise ValueError("organ_surface_metrics requires a nonempty reference mask")
    tolerances = tuple(float(value) for value in tolerances)
    if not pred.any():
        return {"nsd": {tol: 0.0 for tol in tolerances}, "hd95": None, "pred_empty": True}
    distances = surface_distance_metrics.compute_surface_distances(
        mask_gt=gt,
        mask_pred=pred,
        spacing_mm=tuple(float(value) for value in spacing_mm),
    )
    nsd = {
        tol: float(surface_distance_metrics.compute_surface_dice_at_tolerance(distances, tol))
        for tol in tolerances
    }
    hausdorff = float(surface_distance_metrics.compute_robust_hausdorff(distances, 95.0))
    return {"nsd": nsd, "hd95": hausdorff, "pred_empty": False}
