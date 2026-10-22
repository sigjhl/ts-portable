"""Focused tests for the manuscript's surface-area-weighted NSD implementation."""

import unittest

import numpy as np

from nsd_core import hd95, organ_surface_metrics, surface_dice


class SurfaceDiceTests(unittest.TestCase):
    def test_identical_masks_score_one_with_anisotropic_spacing(self):
        mask = np.zeros((8, 9, 10), dtype=bool)
        mask[2:6, 2:7, 3:8] = True
        self.assertAlmostEqual(surface_dice(mask, mask, (0.7, 0.8, 3.0), 3.0), 1.0)

    def test_reference_present_empty_prediction_scores_zero(self):
        gt = np.zeros((5, 5, 5), dtype=bool)
        gt[2, 2, 2] = True
        self.assertEqual(surface_dice(np.zeros_like(gt), gt, (1.0, 1.0, 1.0), 3.0), 0.0)

    def test_reference_must_be_present(self):
        with self.assertRaises(ValueError):
            surface_dice(np.zeros((3, 3, 3)), np.zeros((3, 3, 3)), (1, 1, 1), 3)


class Hd95Tests(unittest.TestCase):
    def _cube(self):
        mask = np.zeros((10, 11, 12), dtype=bool)
        mask[3:7, 3:8, 4:9] = True
        return mask

    def test_identical_masks_hd95_zero_with_anisotropic_spacing(self):
        mask = self._cube()
        self.assertAlmostEqual(hd95(mask, mask, (0.7, 0.8, 3.0)), 0.0)

    def test_reference_present_empty_prediction_hd95_undefined(self):
        gt = self._cube()
        self.assertIsNone(hd95(np.zeros_like(gt), gt, (1.0, 1.0, 1.0)))

    def test_reference_must_be_present_for_hd95(self):
        with self.assertRaises(ValueError):
            hd95(np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), (1, 1, 1))

    def test_displaced_prediction_hd95_positive_and_spacing_scaled(self):
        gt = self._cube()
        pred = np.zeros_like(gt)
        pred[3:7, 3:8, 5:10] = True  # shifted one voxel along the last (3.0 mm) axis
        value = hd95(pred, gt, (0.7, 0.8, 3.0))
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 3.0 + 1e-6)  # bounded by the coarse-axis spacing


class OrganSurfaceMetricsTests(unittest.TestCase):
    def test_identical_masks_all_nsd_one_hd95_zero(self):
        mask = np.zeros((9, 10, 11), dtype=bool)
        mask[2:6, 2:7, 3:8] = True
        result = organ_surface_metrics(mask, mask, (0.7, 0.8, 3.0))
        self.assertFalse(result["pred_empty"])
        for tol in (1.0, 2.0, 3.0):
            self.assertAlmostEqual(result["nsd"][tol], 1.0)
        self.assertAlmostEqual(result["hd95"], 0.0)

    def test_empty_prediction_nsd_zero_hd95_none(self):
        gt = np.zeros((5, 5, 5), dtype=bool)
        gt[2, 2, 2] = True
        result = organ_surface_metrics(np.zeros_like(gt), gt, (1.0, 1.0, 1.0))
        self.assertTrue(result["pred_empty"])
        self.assertEqual(set(result["nsd"].values()), {0.0})
        self.assertIsNone(result["hd95"])

    def test_matches_standalone_helpers(self):
        gt = np.zeros((10, 11, 12), dtype=bool)
        gt[3:7, 3:8, 4:9] = True
        pred = np.zeros_like(gt)
        pred[3:7, 3:8, 5:10] = True
        spacing = (0.9, 1.1, 2.5)
        combined = organ_surface_metrics(pred, gt, spacing)
        self.assertAlmostEqual(combined["nsd"][3.0], surface_dice(pred, gt, spacing, 3.0))
        self.assertAlmostEqual(combined["hd95"], hd95(pred, gt, spacing))

if __name__ == "__main__":
    unittest.main()
