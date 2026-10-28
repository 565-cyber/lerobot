"""Dataset wrapper that recolors the tracked interaction object blue -> green.

Two modes of operation:

* **Offline cache (recommended, multi-view)**: run
  ``scripts/build_multiview_cache.py`` once to freeze the multi-view tracker's
  per-frame target masks to ``outputs/cache/episode_<ep>/up_mask.npy`` and
  ``front_mask.npy``. This wrapper then *reads* those masks (memory-mapped) in
  ``__init__`` and recolors inside ``__getitem__``. The tracker never runs inside
  DataLoader workers, and the up-identity/front-local association is baked in.

* **Legacy in-init tracking (single-view)**: if no cache is given, the wrapper
  runs :class:`InteractionObjectTracker` once per episode during ``__init__`` on
  a single camera and caches per-frame target bboxes (random-access safe).

``camera_mode`` selects which cached masks to apply:

* ``"up_only"``    -> recolor the ``observation.images.up`` frame only.
* ``"front_only"`` -> recolor the ``observation.images.front`` frame only.
* ``"multiview"``  -> recolor both up and front frames.

Safety (spec sections 9 / 14):
  * Only the tracked target mask is recolored. Frames where the target is
    occluded or terminally lost are returned **unchanged** (宁可漏增强，不要错增强).
  * Only the selected camera image tensors are replaced; ``state`` / ``action``
    (and every other key) are returned exactly as-is — this is asserted with
    ``torch.equal`` on a sample frame at the end of ``__init__``.
  * The underlying LeRobotDataset is never mutated.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from object_generalization.masks.blue_object_mask import BlueMaskConfig, blue_threshold_mask
from object_generalization.masks.interaction_object_tracker import InteractionObjectTracker, TrackerConfig
from object_generalization.transforms.object_recolor import BLUE_HUE, GREEN_HUE, recolor_to_green

DEFAULT_AUGMENT_CAMERA = "observation.images.up"

# camera_mode -> cameras whose cached mask is recolored
CAMERA_KEYS = {
    "up_only": ["observation.images.up"],
    "front_only": ["observation.images.front"],
    "multiview": ["observation.images.up", "observation.images.front"],
}

# camera key -> cache file stem
CAM_SHORT = {
    "observation.images.up": "up",
    "observation.images.front": "front",
}


def _recolor_at_bbox(image, bbox, mask_config, recolor_fn):
    """Recolor the blue target inside ``bbox`` (padded), leaving everything else identical."""
    C, H, W = image.shape
    x0, y0, x1, y1 = bbox
    pad = max(16, int(0.5 * max(x1 - x0, y1 - y0)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)

    crop = image[:, y0:y1, x0:x1]  # (C, ch, cw)
    m = blue_threshold_mask(crop, mask_config)  # uint8 (ch, cw) 0/255
    if not m.any():
        return image

    full = torch.zeros((1, H, W), dtype=image.dtype, device=image.device)
    full[0, y0:y1, x0:x1] = torch.from_numpy(m).to(image.device) / 255.0
    return recolor_fn(image, full)


class ObjectColorAugmentedDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        camera_mode: str = "up_only",
        augment_cameras: list[str] | None = None,
        tracker_config: TrackerConfig | None = None,
        mask_cache_dir: str | None = None,
        source_hue: float | None = None,
        target_hue: float | None = None,
    ):
        self.base_dataset = base_dataset
        if augment_cameras is not None:
            # explicit list takes precedence over camera_mode (backward compat)
            self.augment_cameras = list(augment_cameras)
        elif camera_mode in CAMERA_KEYS:
            self.augment_cameras = list(CAMERA_KEYS[camera_mode])
        else:
            raise ValueError(f"unknown camera_mode {camera_mode!r}; expected {sorted(CAMERA_KEYS)}")
        self.camera_mode = camera_mode
        self.tracker_config = tracker_config or TrackerConfig()
        self.source_hue = BLUE_HUE if source_hue is None else source_hue
        self.target_hue = GREEN_HUE if target_hue is None else target_hue

        # Recolor-time mask config: same HSV, but target-appropriate area bounds.
        mcfg = self.tracker_config.mask
        self.recolor_mask_config = BlueMaskConfig(
            hsv_lower=mcfg.hsv_lower,
            hsv_upper=mcfg.hsv_upper,
            open_kernel=mcfg.open_kernel,
            close_kernel=mcfg.close_kernel,
            min_area=self.tracker_config.min_area,
            max_area=self.tracker_config.max_area,
        )

        self._cache_masks = self._load_masks(mask_cache_dir) if mask_cache_dir else None
        # Legacy single-view bbox tracking (only used when no cache is given).
        self.tracks = None if self._cache_masks is not None else self._build_tracks()

        self._assert_state_action_unchanged()

    # ---- cache loading ----------------------------------------------------
    def _load_masks(self, cache_dir):
        """Memory-map the offline per-episode mask sequences.

        Returns {camera_key: {episode_index: memmap (T, H, W) uint8 0/255}}.
        Missing episodes / cameras are simply absent (those frames stay unchanged).
        """
        masks = {cam: {} for cam in self.augment_cameras}
        if not os.path.isdir(cache_dir):
            return masks
        for entry in sorted(os.listdir(cache_dir)):
            if not entry.startswith("episode_"):
                continue
            ep = int(entry[len("episode_"):])
            ep_dir = os.path.join(cache_dir, entry)
            for cam in self.augment_cameras:
                path = os.path.join(ep_dir, f"{CAM_SHORT[cam]}_mask.npy")
                if os.path.exists(path):
                    masks[cam][ep] = np.load(path, mmap_mode="r")
        return masks

    # ---- legacy single-view tracking --------------------------------------
    def _build_tracks(self):
        tracks = {}
        cur_ep = None
        tracker = InteractionObjectTracker(self.tracker_config)
        cam = self.augment_cameras[0]
        for idx in range(len(self.base_dataset)):
            item = self.base_dataset[idx]
            ep = int(item["episode_index"])
            fr = int(item["frame_index"])
            if ep != cur_ep:
                tracker.reset()
                cur_ep = ep
                tracks[ep] = {}
                tracker.initialize(item[cam])
            else:
                tracker.update(item[cam])
            info = tracker.get_debug_info()
            tracks[ep][fr] = info["bbox"] if info["state"] == "tracking" else None
        return tracks

    # ---- safety -----------------------------------------------------------
    def _assert_state_action_unchanged(self):
        """Recolor must never alter state/action: assert torch.equal on a sample frame."""
        item = dict(self.base_dataset[0])
        before = {k: item[k].clone() for k in item if k in ("observation.state", "action")}
        rec = self._augment(item)
        for k in before:
            if not torch.equal(before[k], rec[k]):
                raise RuntimeError(f"augmentation changed {k} — aborting")

    # ---- augmentation -----------------------------------------------------
    def _augment(self, item):
        ep = int(item["episode_index"])
        fr = int(item["frame_index"])
        for cam in self.augment_cameras:
            if cam not in item:
                continue
            if self._cache_masks is not None:
                masks_by_ep = self._cache_masks.get(cam, {})
                arr = masks_by_ep.get(ep)
                if arr is None or fr >= arr.shape[0]:
                    continue
                m = arr[fr]
                if not (m > 0).any():
                    continue  # target occluded / lost -> leave unchanged
                item[cam] = recolor_to_green(
                    item[cam], m, source_hue=self.source_hue, target_hue=self.target_hue
                )
            else:
                bbox = self.tracks.get(ep, {}).get(fr)
                if bbox is None:
                    continue  # target occluded / lost -> leave unchanged
                item[cam] = _recolor_at_bbox(
                    item[cam],
                    bbox,
                    self.recolor_mask_config,
                    lambda img, m: recolor_to_green(
                        img, m, source_hue=self.source_hue, target_hue=self.target_hue
                    ),
                )
        return item

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self._augment(dict(self.base_dataset[idx]))

    # Forward a few convenient attributes so downstream code (ACT) can query them.
    @property
    def hf_dataset(self):
        return self.base_dataset.hf_dataset

    @property
    def meta(self):
        return self.base_dataset.meta

    @property
    def delta_timestamps(self):
        return self.base_dataset.delta_timestamps
