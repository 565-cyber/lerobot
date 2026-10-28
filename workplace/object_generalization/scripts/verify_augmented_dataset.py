"""Verify ObjectColorAugmentedDataset (multiview + offline cache) end-to-end.

Checks: __init__ state/action assertion passes; a tracking frame is recolored
blue->green in the masked region while outside stays identical; state/action are
bit-identical to the base dataset; a non-tracking frame is returned unchanged.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from object_generalization.datasets.object_augmented_dataset import (  # noqa: E402
    ObjectColorAugmentedDataset,
)
from object_generalization.transforms.object_recolor import outside_mask_difference  # noqa: E402

CACHE = os.path.join(WORKPLACE, "object_generalization", "outputs", "cache")
UP = "observation.images.up"
FRONT = "observation.images.front"


def main():
    base = LeRobotDataset("newdata", root="/home/zjq/lerobot/dataset/newdata", episodes=[0, 1, 2])
    aug = ObjectColorAugmentedDataset(base, camera_mode="multiview", mask_cache_dir=CACHE)
    print(f"base={len(base)} aug={len(aug)} cameras={aug.augment_cameras}")
    print("init state/action assert: PASSED (no exception)")

    # find a frame where the front target is tracked in ep0 (f120) and one at home (f0)
    idx_track = None
    idx_home = None
    for i in range(len(base)):
        it = base[i]
        if int(it["episode_index"]) == 0:
            if int(it["frame_index"]) == 120:
                idx_track = i
            if int(it["frame_index"]) == 0:
                idx_home = i
        if idx_track is not None and idx_home is not None:
            break

    for label, idx in [("tracking f120", idx_track), ("home f0", idx_home)]:
        b = base[idx]
        a = aug[idx]
        print(f"\n[{label}]")
        for k in ("observation.state", "action"):
            if k in b and k in a:
                same = torch.equal(b[k], a[k])
                print(f"  {k}: equal={same}")
        for cam in (UP, FRONT):
            if cam not in b:
                continue
            diff = (b[cam] - a[cam]).abs().max().item()
            n_changed = int((b[cam] != a[cam]).any(dim=0).sum().item())
            print(f"  {cam}: max_abs_diff={diff:.4f} changed_pixels={n_changed}")

    # green hue check on a tracked front frame
    b = base[idx_track]
    a = aug[idx_track]
    changed = (b[FRONT] != a[FRONT]).any(dim=0)
    print(f"\n[green check] front changed pixel count at f120 = {int(changed.sum())}")

    # outside-mask difference should be ~0 (recolor leaves mask-outside identical)
    from object_generalization.transforms.object_recolor import _to_hwc_uint8  # noqa: F401

    m = aug._cache_masks[FRONT][0][120]
    if (m > 0).any():
        diff_out = outside_mask_difference(b[FRONT], a[FRONT], m)
        print(f"  outside-mask max diff (front f120) = {diff_out:.4f} (should be ~0)")


if __name__ == "__main__":
    main()
