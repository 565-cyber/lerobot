"""DEPRECATED: superseded by ``scripts/inspect_robot_state_tracking.py``.

The front tracker is now robot-state-aware (``RobotStatePrior``) and no longer
exposes the single-joint ``use_state_prior`` / ``z_active_threshold`` gate this
script referenced. Run ``inspect_robot_state_tracking.py`` instead; it also
reproduces the old elbow<20 gate as an in-script baseline for the A/B comparison.

Inspect / validate the FRONT (wrist) camera interaction-object tracker (v1).

Does NOT train anything. Runs FrontInteractionObjectTracker sequentially per
episode and reports:
  * initialization success, tracking / awaiting / occluded / lost frame counts,
  * target-switch count (large center jump between consecutive tracking frames),
  * per-stage table + 5-panel figures (Original / Blue candidates / Selected
    target / Green recolor / Tracking bbox).

Usage:
    python scripts/inspect_front_tracking.py --root /home/zjq/lerobot/dataset/newdata
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from object_generalization.masks.blue_object_mask import blue_threshold_mask  # noqa: E402
from object_generalization.masks.front_interaction_tracker import (  # noqa: E402
    FrontInteractionObjectTracker,
    FrontTrackerConfig,
)
from object_generalization.transforms.object_recolor import recolor_to_green  # noqa: E402

FRONT_KEY = "observation.images.front"
FIGURE_FRAMES = {0, 90, 150, 180, 210, 240, 300, 360, 420, 480, 540}
SWITCH_THRESHOLD = 60.0  # px; the gripper-fixed cube moves ~2px/frame, so >60px = switch


def _to_uint8_hwc(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu()
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
    return arr


def _label(img, text):
    return cv2.putText(img.copy(), text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def save_figure(original, all_blue, selected_mask, recolored, bbox, state, out_path):
    orig = _to_uint8_hwc(original)
    rec = _to_uint8_hwc(recolored)
    cand = (np.asarray(all_blue).squeeze() > 0).astype(np.uint8) * 255
    cand = cv2.cvtColor(cand, cv2.COLOR_GRAY2BGR)
    sel = (np.asarray(selected_mask).squeeze() > 0).astype(np.uint8) * 255
    sel = cv2.cvtColor(sel, cv2.COLOR_GRAY2BGR)
    track = orig.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(track, (x0, y0), (x1, y1), (0, 0, 255), 2)
    row = np.hstack(
        [
            _label(orig, "Original"),
            _label(cand, "Blue candidates"),
            _label(sel, "Selected target"),
            _label(rec, "Green recolor"),
            _label(track, f"Tracking bbox [{state}]"),
        ]
    )
    row_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, row_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(WORKPLACE, "object_generalization", "outputs"))
    ap.add_argument("--no-state-prior", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print(f"Loading LeRobotDataset('{args.dataset}', root={args.root}, episodes={args.episodes})")
    ds = LeRobotDataset(args.dataset, root=args.root, episodes=args.episodes)
    print(f"  num_frames   = {len(ds)}")
    print(f"  front key    = {FRONT_KEY}")

    cfg = FrontTrackerConfig(use_state_prior=not args.no_state_prior)
    print(f"  state prior   = {'ON' if cfg.use_state_prior else 'OFF'} (z joint < {cfg.z_active_threshold})")

    print("=" * 72)
    print("Front tracking pass (sequential per episode):")
    tracker = FrontInteractionObjectTracker(cfg)
    cur_ep = None
    global_state = Counter()
    ep_init = {}          # ep -> bool (locked a target at some point)
    ep_init_frame = {}    # ep -> first frame it locked a target
    ep_state = Counter()
    stage_rows = []
    switches = Counter()  # ep -> switch count
    prev_center = {}
    prev_tracking = {}
    n_frames = 0

    for idx in range(len(ds)):
        item = ds[idx]
        ep = int(item["episode_index"])
        fr = int(item["frame_index"])
        img = item[FRONT_KEY]
        state = item["observation.state"]
        n_frames += 1

        if ep != cur_ep:
            tracker.reset()
            cur_ep = ep
            ep_state[ep] = Counter()
            prev_center[ep] = None
            prev_tracking[ep] = False
            mask = tracker.initialize(img, state)
        else:
            mask = tracker.update(img, state)

        info = tracker.get_debug_info()
        st = info["state"]
        global_state[st] += 1
        ep_state[ep][st] += 1

        if info["center"] is not None and st == "tracking":
            if ep not in ep_init:
                ep_init[ep] = True
                ep_init_frame[ep] = fr
            if prev_tracking[ep] and prev_center[ep] is not None:
                d = ((info["center"][0] - prev_center[ep][0]) ** 2 + (info["center"][1] - prev_center[ep][1]) ** 2) ** 0.5
                if d > SWITCH_THRESHOLD:
                    switches[ep] += 1
            prev_center[ep] = info["center"]
            prev_tracking[ep] = True
        else:
            prev_tracking[ep] = False

        if fr in FIGURE_FRAMES:
            stage_rows.append((ep, fr, st, info["area"], info["extent"], info["bbox"], info["reason"]))
            all_blue = blue_threshold_mask(img, cfg.mask)
            rec = recolor_to_green(img, mask) if st == "tracking" else img
            save_figure(img, all_blue, mask, rec, info["bbox"], st, os.path.join(args.out, "debug", f"front_cam{ep}_f{fr}.png"))

    # ---- aggregate stats ----
    print("=" * 72)
    print("Aggregate stats:")
    print(f"  frames tracked      = {n_frames}")
    for st in ["tracking", "awaiting", "occluded", "lost"]:
        if global_state[st]:
            print(f"    {st:<12} {global_state[st]:6d}  ({100.0 * global_state[st] / n_frames:5.1f}%)")
    n_init = sum(1 for v in ep_init.values() if v)
    print(f"  init success        = {n_init}/{len(ep_state)}")
    for ep in sorted(ep_init_frame):
        print(f"      ep{ep} locked target at frame {ep_init_frame[ep]}")
    print(f"  target switches     = {dict(switches)}")
    lost_eps = [ep for ep in ep_state if ep_state[ep]["lost"] > 0]
    print(f"  lost episodes       = {lost_eps}")

    print("=" * 72)
    print("Per-stage table (ep, frame, state, area, extent, bbox, reason):")
    for (ep, fr, st, area, ext, bbox, reason) in stage_rows:
        print(f"  ep{ep} f{fr:<4} {st:<9} area={str(area):<6} ext={str(round(ext,2) if ext else ext):<6} bbox={bbox} reason={reason}")

    print("=" * 72)
    print("DONE")


if __name__ == "__main__":
    main()
