"""Compare SAM2 (blue-intersected) masks vs the lightweight multiview tracker.

Loads the SAM2 offline cache and re-runs the lightweight MultiViewInteractionTracker
live, then reports per-camera, per-phase coverage (fraction of frames with a
non-empty mask) and IoU where both methods agree the cube is visible.

This is the coverage/agreement report for the "上 SAM2 替换 mask" decision — it
answers whether SAM2 tracks the cube on more interaction frames than the
lightweight tracker, and whether the two agree on *where* the cube is.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from object_generalization.masks.multiview_object_tracker import (  # noqa: E402
    MultiViewInteractionTracker,
    MultiViewTrackerConfig,
)

UP = "observation.images.up"
FRONT = "observation.images.front"
CACHE = os.path.join(WORKPLACE, "object_generalization", "outputs", "cache")


def _iou(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _cov(a):
    return (a > 0).any()


def main():
    phases = ["home", "approaching", "grasping", "moving", "placing", "retracting"]
    # accumulate per-phase, per-camera stats across episodes
    acc = {
        "up": {p: {"sam2": 0, "lw": 0, "both": 0, "iou_sum": 0.0, "n_iou": 0, "n": 0} for p in phases},
        "front": {p: {"sam2": 0, "lw": 0, "both": 0, "iou_sum": 0.0, "n_iou": 0, "n": 0} for p in phases},
    }

    for ep in (0, 1, 2):
        ds = LeRobotDataset("newdata", root="/home/zjq/lerobot/dataset/newdata", episodes=[ep])
        up_sam2 = np.load(os.path.join(CACHE, f"episode_{ep}", "up_mask.npy"))
        front_sam2 = np.load(os.path.join(CACHE, f"episode_{ep}", "front_mask.npy"))

        mv = MultiViewInteractionTracker(MultiViewTrackerConfig())
        for idx in range(len(ds)):
            item = ds[idx]
            fr = int(item["frame_index"])
            if fr == 0:
                res = mv.initialize(item[UP], item[FRONT], item["observation.state"])
            else:
                res = mv.update(item[UP], item[FRONT], item["observation.state"])
            phase = mv.get_debug_info()["phase"] or "home"
            lw_up = res["up_mask"]
            lw_front = res["front_mask"]

            for cam, sam2, lw in (
                ("up", up_sam2[fr], lw_up),
                ("front", front_sam2[fr], lw_front),
            ):
                s = acc[cam][phase]
                s["n"] += 1
                s["sam2"] += int(_cov(sam2))
                s["lw"] += int(_cov(lw))
                both = _cov(sam2) and _cov(lw)
                s["both"] += int(both)
                if both:
                    s["iou_sum"] += _iou(sam2, lw)
                    s["n_iou"] += 1

    print(f"{'phase':<12} {'cam':<6} {'n':>4} {'SAM2%':>7} {'LW%':>6} {'agree%':>7} {'IoU':>6}")
    for p in phases:
        for cam in ("up", "front"):
            s = acc[cam][p]
            if s["n"] == 0:
                continue
            sam2p = 100.0 * s["sam2"] / s["n"]
            lwp = 100.0 * s["lw"] / s["n"]
            agreep = 100.0 * s["both"] / s["n"]
            iou = s["iou_sum"] / s["n_iou"] if s["n_iou"] else 0.0
            print(f"{p:<12} {cam:<6} {s['n']:>4} {sam2p:>6.1f}% {lwp:>5.1f}% {agreep:>6.1f}% {iou:>6.2f}")


if __name__ == "__main__":
    main()
