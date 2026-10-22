"""Model registry: friendly name -> exported model-card file.

Both models share the identical pipeline (Segmentator reads everything it needs from the
card: target spacing, patch, normalization, label map). Adding a model is a registry row,
not new code.
"""
from __future__ import annotations

import os
from pathlib import Path

# single-model friendly name -> <tag> (card <tag>.model_card.json, onnx <tag>.onnx)
_MODELS = {
    "fast": "total_3mm_297",     # 3 mm, 117 classes  (TS --fast)
    "fastest": "total_6mm_298",  # 6 mm, 117 classes  (TS --fastest, overview)
}

# 'full' is the 1.5 mm `total` (TS default): 5 disjoint sub-models, MERGE ORDER below
# (earlier overwritten by later, matching stock TS).
_FULL_PARTS = [
    "total_1p5mm_291_organs",
    "total_1p5mm_292_vertebrae",
    "total_1p5mm_293_cardiac",
    "total_1p5mm_294_muscles",
    "total_1p5mm_295_ribs",
]


def models_dir() -> Path:
    env = os.environ.get("ONNX_TOTALSEG_MODELS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "models"


def available() -> list[str]:
    return sorted(list(_MODELS) + ["full"])


def is_multimodel(model: str) -> bool:
    return model == "full"


def resolve_card(model: str = "fast") -> str:
    if model not in _MODELS:
        raise ValueError(f"unknown single model '{model}'. Choices: {sorted(_MODELS)}")
    return str(models_dir() / f"{_MODELS[model]}.model_card.json")


def full_cards() -> list[str]:
    return [str(models_dir() / f"{t}.model_card.json") for t in _FULL_PARTS]


def label_map() -> str:
    return str(models_dir() / "total_label_map.json")
