"""Smoke test: run SAM2 video tracking on the ep0 UP stream.

Prompts the cube on frame 0 with the bbox from the existing lightweight up
tracker (no manual click), propagates, and saves a few visualization panels.
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# SAM2 repo on the path (editable install already adds it, but be explicit)
SAM2_ROOT = os.path.join(WORKPLACE, "object_generalization", "third_party", "sam2")
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402
from object_generalization.masks.interaction_object_tracker import (  # noqa: E402
    InteractionObjectTracker,
    TrackerConfig,
)

UP_KEY = "observation.images.up"
CKPT = os.path.join(WORKPLACE, "object_generalization", "checkpoints", "sam2.1_hiera_base_plus.pt")
# config name is relative to the sam2 package (Hydra search path pkg://sam2)
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def main():
    ds = LeRobotDataset("newdata", root="/home/zjq/lerobot/dataset/newdata", episodes=[0])

    # 1. prompt bbox from the lightweight up tracker on frame 0
    up0 = ds[0][UP_KEY]
    tr = InteractionObjectTracker(TrackerConfig())
    tr.initialize(up0)
    info = tr.get_debug_info()
    assert info["bbox"] is not None, "up tracker failed to init on f0"
    x0, y0, x1, y1 = info["bbox"]
    print(f"up f0 prompt bbox = ({x0},{y0},{x1},{y1})")

    # 2. write up frames to a temp JPEG dir (SAM2 reads frames from disk)
    tmp = tempfile.mkdtemp(prefix="sam2_frames_")
    for idx in range(len(ds)):
        img = _to_hwc_uint8(ds[idx][UP_KEY])  # (H, W, 3) RGB uint8
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(tmp, f"{idx:05d}.jpg"), bgr)
    print(f"wrote {len(ds)} frames to {tmp}")

    # 3. build predictor + track
    from sam2.build_sam import build_sam2_video_predictor

    torch.cuda.empty_cache()
    predictor = build_sam2_video_predictor(CFG, CKPT, device="cuda:0")
    state = predictor.init_state(video_path=tmp)

    _, out_obj_ids, _ = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=0,
        box=np.array([x0, y0, x1, y1], dtype=np.float32),
    )
    print("obj ids:", out_obj_ids)

    masks = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        masks[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze(0)  # (H, W) bool

    n_nonzero = sum(1 for m in masks.values() if m.any())
    print(f"propagated {len(masks)} frames, {n_nonzero} with non-empty mask")
    for fr in [0, 120, 180, 220, 260, 300, 420]:
        m = masks.get(fr)
        if m is None:
            continue
        print(f"  f{fr}: mask area = {int(m.sum())} px, bbox = {_bbox(m)}")

    # 4. visualize a few frames
    out_dir = os.path.join(WORKPLACE, "object_generalization", "outputs", "debug")
    os.makedirs(out_dir, exist_ok=True)
    for fr in [0, 120, 180, 220, 260, 300, 420]:
        if fr not in masks:
            continue
        img = _to_hwc_uint8(ds[fr][UP_KEY])
        m = masks[fr]
        overlay = img.copy()
        overlay[m] = (overlay[m] * 0.5 + np.array([0, 255, 0], np.uint8) * 0.5).astype(np.uint8)
        panel = np.hstack([img, overlay])
        cv2.imwrite(
            os.path.join(out_dir, f"sam2_up_ep0_f{fr}.png"),
            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
        )
    print("saved panels to outputs/debug/sam2_up_ep0_f*.png")


def _bbox(m):
    if not m.any():
        return None
    ys, xs = np.where(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


if __name__ == "__main__":
    main()
