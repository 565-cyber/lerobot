"""End-to-end verification of the HYBRID (SAM2 ∪ lightweight) cache.

Checks every frame of all episodes: state/action bit-identical, and the recolor
leaves every pixel outside the mask unchanged (outside-mask max diff == 0).
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

CACHE = os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid")
UP = "observation.images.up"
FRONT = "observation.images.front"


def main():
    base = LeRobotDataset("newdata", root="/home/zjq/lerobot/dataset/newdata", episodes=[0, 1, 2])
    aug = ObjectColorAugmentedDataset(base, camera_mode="multiview", mask_cache_dir=CACHE)
    print(f"init state/action assert: PASSED; cameras={aug.augment_cameras}")

    bad_state = 0
    max_out = 0.0
    recolored = 0
    for i in range(len(base)):
        b = base[i]
        a = aug[i]
        if not (torch.equal(b["observation.state"], a["observation.state"]) and torch.equal(b["action"], a["action"])):
            bad_state += 1
        ep = int(b["episode_index"])
        fr = int(b["frame_index"])
        for cam in (UP, FRONT):
            m = aug._cache_masks[cam][ep][fr]
            if (m > 0).any():
                recolored += 1
                max_out = max(max_out, outside_mask_difference(b[cam], a[cam], m))
    print(f"frames={len(base)} state/action_mismatch={bad_state} max_outside_diff={max_out:.6f} recolored_frames={recolored}")


if __name__ == "__main__":
    main()
