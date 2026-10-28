"""Mask storage convention shared between mask building and dataset loading.

Vestigial for Stage 2B. Its ``build_masks.py`` producer has been deleted -- that
approach needed a hand-captured empty-workspace reference image, which does not
exist, and could not have survived the UP camera's ±7% auto-exposure drift
anyway (see ``build_plate.py``'s module docstring). What is left is the file
format, still used by ``BackgroundAugmentedDataset``, the Stage 2A in-process
wrapper.

Format::

    path:    {mask_root}/{camera_key}/episode_{episode_index:06d}.npy
    content: np.uint8 array of shape (T, H, W), values 0 (background) or 1 (foreground).
             T is the number of frames in the episode, indexed by frame_index (0-based).

Note the polarity is the OPPOSITE of Stage 2B's ``placement.npy``: there, 1 means
"a distractor may go here" (background), here 1 means "keep this pixel"
(foreground). Two arrays with the same dtype and shape and opposite meanings is
exactly the kind of thing that gets silently swapped, which is why
``build_plate`` deliberately does not name its output ``up_mask.npy``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def episode_mask_path(mask_root: str | Path, camera_key: str, episode_index: int) -> Path:
    return Path(mask_root) / camera_key / f"episode_{episode_index:06d}.npy"


def save_episode_mask(
    mask_root: str | Path, camera_key: str, episode_index: int, mask: np.ndarray
) -> Path:
    """Save a per-episode foreground mask as uint8 (T, H, W)."""
    mask = np.asarray(mask).astype(np.uint8)
    path = episode_mask_path(mask_root, camera_key, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask)
    return path


def load_episode_mask(
    mask_root: str | Path, camera_key: str, episode_index: int
) -> np.ndarray | None:
    """Load a per-episode foreground mask as uint8 (T, H, W), or None if missing."""
    path = episode_mask_path(mask_root, camera_key, episode_index)
    if not path.exists():
        return None
    return np.load(path).astype(np.uint8)
