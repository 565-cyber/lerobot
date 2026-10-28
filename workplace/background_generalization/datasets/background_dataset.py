"""BackgroundAugmentedDataset: wraps a LeRobotDataset and replaces image backgrounds.

Stage 2. Only the camera images are modified; state and action are left untouched
(verified by the check script). The background is sampled deterministically per
episode so all frames of one episode share the same background (no flicker)::

    background_id = episode_index % len(backgrounds)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from ..masks.mask_utils import load_episode_mask
from ..transforms.background_replacement import apply_background_replacement

_SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resize3d(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if t.shape[-2] == h and t.shape[-1] == w:
        return t
    return torch.nn.functional.interpolate(t[None], size=(h, w), mode="bilinear", align_corners=False)[0]


def _solid_color(episode_index: int, h: int, w: int) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.00, 0.00, 0.00], [0.20, 0.20, 0.20], [0.40, 0.40, 0.40],
            [0.70, 0.60, 0.50], [0.30, 0.20, 0.10], [0.60, 0.70, 0.60],
            [0.80, 0.80, 0.80], [0.10, 0.15, 0.20],
        ],
        dtype=torch.float32,
    )
    rgb = palette[episode_index % len(palette)]
    return rgb[:, None, None].expand(3, h, w).clone()


class _BackgroundStore:
    """Loads and samples backgrounds, deterministic per episode."""

    def __init__(self, background_root: str | Path | None, seed: int = 42):
        self.paths: list[Path] = []
        self.tensors: dict[Path, torch.Tensor] = {}
        if background_root is not None:
            root = Path(background_root)
            if root.is_dir():
                self.paths = sorted(
                    p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXT
                )
        rng = random.Random(seed)
        rng.shuffle(self.paths)

    def get(self, episode_index: int, h: int, w: int) -> torch.Tensor:
        if not self.paths:
            return _solid_color(episode_index, h, w)
        path = self.paths[episode_index % len(self.paths)]
        if path not in self.tensors:
            self.tensors[path] = read_image(str(path)).float() / 255.0  # (3, H, W) float [0, 1]
        return _resize3d(self.tensors[path], h, w)


class BackgroundAugmentedDataset(Dataset):
    """Wraps a LeRobotDataset and replaces camera-image backgrounds.

    Args:
        base_dataset: an instance of ``LeRobotDataset``.
        mask_root: directory of precomputed foreground masks (see
            ``masks/mask_utils.py`` for the path convention). If None, no mask is
            applied and images are returned unchanged.
        background_root: directory of background images (jpg/png/...). If None or
            empty, a deterministic solid color is used per episode.
        probability: fraction of episodes to augment (per-episode, deterministic).
        temporal_mode: "episode" (only supported in v1).
        seed: reproducibility of episode -> background assignment.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        mask_root: str | Path | None = None,
        background_root: str | Path | None = None,
        probability: float = 1.0,
        temporal_mode: str = "episode",
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.mask_root = None if mask_root is None else Path(mask_root)
        self.probability = float(probability)
        self.temporal_mode = temporal_mode
        self.seed = int(seed)
        if self.temporal_mode != "episode":
            raise NotImplementedError("v1 only supports temporal_mode='episode'.")

        self._backgrounds = _BackgroundStore(background_root, seed=self.seed)
        self._mask_cache: dict[tuple[str, int], np.ndarray | None] = {}
        self.camera_keys: list[str] = list(self.base_dataset.meta.camera_keys)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _should_augment(self, episode_index: int) -> bool:
        if self.probability >= 1.0:
            return True
        if self.probability <= 0.0:
            return False
        rng = random.Random(self.seed * 1000003 + episode_index)
        return rng.random() < self.probability

    def _get_mask(self, camera_key: str, episode_index: int, frame_index: int) -> np.ndarray | None:
        if self.mask_root is None:
            return None
        key = (camera_key, episode_index)
        if key not in self._mask_cache:
            self._mask_cache[key] = load_episode_mask(self.mask_root, camera_key, episode_index)
        arr = self._mask_cache[key]
        if arr is None or frame_index >= len(arr):
            return None
        return arr[frame_index]

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]

        episode_index = int(item["episode_index"].item())
        frame_index = int(item["frame_index"].item()) if "frame_index" in item else int(item["index"].item())

        if not self._should_augment(episode_index):
            return item

        for cam in self.camera_keys:
            mask = self._get_mask(cam, episode_index, frame_index)
            if mask is None:
                continue
            img = item[cam]  # (C, H, W) float [0, 1]
            h, w = img.shape[-2], img.shape[-1]
            background = self._backgrounds.get(episode_index, h, w)
            item[cam] = apply_background_replacement(img, mask, background)

        return item
