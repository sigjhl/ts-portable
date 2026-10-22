"""MultiModelSegmentator: the full 1.5 mm `total` path (5 disjoint sub-models merged).

Mirrors stock TS `total` (default): resample the image to 1.5 mm ONCE, run each sub-model
(each with its OWN CT-normalization), remap each model's LOCAL label ids to GLOBAL total
ids, and merge by sequential overwrite (later models win at overlaps), then upsample the
merged label map back to the original geometry. Memory stays low because each model is
argmaxed at 1.5 mm before merge (never 118-channel logits at full res).

The 5 sub-models run STRICTLY SEQUENTIALLY and never need to be resident together, so we
build each session, run it, and free it before the next. Peak GPU memory is therefore ONE
sub-model (~2.3 GB fp16), not five (~10 GB) — which is what lets full fit a 4 GB card. The
cost is ~one session-build per sub-model per call (~seconds on the already-slow full tier).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import io_nibabel
from ._nnunet_ops import change_spacing_to_shape, change_spacing_to_spacing
from .ep import resolve_providers
from .segmentator import Segmentator


class MultiModelSegmentator:
    def __init__(self, card_paths: Sequence[str], label_map_path: str,
                 providers: Optional[Sequence[str]] = None, device: str = "auto",
                 intra_op_threads: Optional[int] = None, fp16: bool = False,
                 device_id: Optional[int] = None):
        # card_paths MUST be in merge order (earlier overwritten by later, like stock TS).
        self.card_paths = list(card_paths)
        self.providers = list(providers) if providers else resolve_providers(device, device_id)
        self.intra_op_threads = intra_op_threads
        self.fp16 = fp16
        # canonical name -> global total id (e.g. heart -> 51)
        self.global_map = {k: int(v) for k, v in json.loads(Path(label_map_path).read_text()).items()}
        # per-card set of GLOBAL ids it can emit (from the card labels; no session needed).
        # Lets --roi_subset run ONLY the sub-model(s) that own the requested labels.
        self.card_gids = []
        for cp in self.card_paths:
            names = json.loads(Path(cp).read_text())["labels"].keys()  # {name: local_id}
            gids = {self.global_map.get(n, 0) for n in names if n != "background"}
            gids.discard(0)
            self.card_gids.append(gids)
        # metadata from the cards (no session needed); all parts share target spacing
        card0 = json.loads(Path(self.card_paths[0]).read_text())
        self.target_spacing = [float(x) for x in card0["target_spacing"]]
        self.data_order = int(card0["resampling_data_kwargs"]["order"])
        # providers[0] may be (name, {device_id}); expose the EP name for display
        p0 = self.providers[0]
        self.active_provider = p0[0] if isinstance(p0, tuple) else p0
        self.requested_providers = self.providers

    def _build_part(self, card_path: str) -> Segmentator:
        return Segmentator(card_path, providers=self.providers,
                           intra_op_threads=self.intra_op_threads, fp16=self.fp16)

    def predict_loaded(self, data: np.ndarray, props: dict, lowres: bool = False,
                       restrict_ids=None):
        """Predict from an already-loaded (data, props). Returns (seg[z,y,x] uint8, None).
        full is always full-res (lowres ignored).

        restrict_ids: if given (global total ids, e.g. from --roi_subset), run ONLY the
        sub-model(s) whose labels intersect it and skip the rest. Used by the crop-cascade so
        a single-organ ROI (spleen -> organs; iliopsoas/autochthon -> muscles) pays one
        sub-model build+inference instead of five, at full 1.5 mm quality."""
        spacing = props["spacing"]
        orig_shape = data.shape[1:]

        cards = self.card_paths
        if restrict_ids:
            rid = {int(x) for x in restrict_ids}
            cards = [cp for cp, g in zip(self.card_paths, self.card_gids) if g & rid]

        # resample image -> target spacing ONCE (shared by all parts); TS int32 cast
        img_t = change_spacing_to_spacing(data[0], spacing, self.target_spacing, order=self.data_order)
        img_t = img_t.astype(np.int32).astype(np.float32)

        combined = np.zeros(img_t.shape, dtype=np.uint8)
        for card_path in cards:  # merge order: later parts overwrite earlier at overlaps
            part = self._build_part(card_path)        # one session resident at a time
            seg_local = part.predict_array(img_t[None], self.target_spacing)  # at img_t shape
            for local_id, name in part.labels.items():
                if local_id == 0:
                    continue
                gid = self.global_map.get(name, 0)
                if gid:
                    combined[seg_local == local_id] = gid
            del part, seg_local                       # free the session (+ its GPU arena)
            gc.collect()

        seg = change_spacing_to_shape(combined, orig_shape, order=0).astype(np.uint8)
        return seg, None

    def predict_file(self, in_path: str, out_path: Optional[str] = None,
                     lowres: bool = False) -> np.ndarray:
        data, props = io_nibabel.read_image(in_path)
        seg, canonical_affine = self.predict_loaded(data, props)
        if out_path is not None:
            io_nibabel.write_seg(seg, out_path, props, canonical_affine=canonical_affine)
        return seg
