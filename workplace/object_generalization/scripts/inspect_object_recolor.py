"""Inspect / validate the interaction-object tracker + blue->green recolor (v1).

Does NOT train anything. For each episode it runs the tracker sequentially and
reports:
  * init success rate, tracking / occluded / lost frame counts, lost episodes,
  * a per-stage table (init / approach / grasp / carry / end),
  * state/action byte-identical through the wrapper + a DataLoader batch,
  * 5-panel figures: Original / Blue candidates / Selected object / Green recolor
    / Tracking bbox.

Usage:
    python scripts/inspect_object_recolor.py --root /home/zjq/lerobot/dataset/newdata
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

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
from object_generalization.masks.blue_object_mask import blue_threshold_mask  # noqa: E402
from object_generalization.masks.interaction_object_tracker import (  # noqa: E402
    InteractionObjectTracker,
    TrackerConfig,
)
from object_generalization.transforms.object_recolor import (  # noqa: E402
    outside_mask_difference,
    recolor_to_green,
)

STATE_KEY = "observation.state"
ACTION_KEY = "action"
FIGURE_FRAMES = {0, 120, 200, 260, 320, 400, 540}


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


def save_figure(original, all_blue_mask, selected_mask, recolored, bbox, state, out_path):
    orig = _to_uint8_hwc(original)
    rec = _to_uint8_hwc(recolored)

    cand = (np.asarray(all_blue_mask).squeeze() > 0).astype(np.uint8) * 255
    cand = cv2.cvtColor(cand, cv2.COLOR_GRAY2BGR)
    sel = (np.asarray(selected_mask).squeeze() > 0).astype(np.uint8) * 255
    sel = cv2.cvtColor(sel, cv2.COLOR_GRAY2BGR)

    track = orig.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(track, (x0, y0), (x1, y1), (0, 0, 255), 2)

    panels = [
        _label(orig, "Original"),
        _label(cand, "Blue candidates"),
        _label(sel, "Selected object"),
        _label(rec, "Green recolor"),
        _label(track, f"Tracking bbox [{state}]"),
    ]
    row = np.hstack(panels)
    row_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, row_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(WORKPLACE, "object_generalization", "outputs"))
    ap.add_argument("--occlusion-timeout", type=int, default=150)
    args = ap.parse_args()

    print("=" * 72)
    print(f"Loading LeRobotDataset('{args.dataset}', root={args.root}, episodes={args.episodes})")
    ds = LeRobotDataset(args.dataset, root=args.root, episodes=args.episodes)
    print(f"  num_frames   = {len(ds)}")
    print(f"  camera_keys  = {ds.meta.camera_keys}")
    cam = ds.meta.camera_keys[0]
    print(f"  tracking cam = {cam}")

    cfg = TrackerConfig(occlusion_timeout=args.occlusion_timeout)

    # ---- tracking pass ----
    print("=" * 72)
    print("Tracking pass (sequential per episode):")
    tracker = InteractionObjectTracker(cfg)
    cur_ep = None
    ep_state = Counter()          # episode_index -> state counter
    ep_init_ok = {}               # episode_index -> bool (init found a target at f0)
    ep_lost_frame = {}            # episode_index -> first frame it became lost
    stage_rows = []               # (ep, fr, state, area, bbox, reason)
    global_state = Counter()
    n_frames = 0

    for idx in range(len(ds)):
        item = ds[idx]
        ep = int(item["episode_index"])
        fr = int(item["frame_index"])
        img = item[cam]
        n_frames += 1

        if ep != cur_ep:
            tracker.reset()
            cur_ep = ep
            ep_state[ep] = Counter()
            mask = tracker.initialize(img)
        else:
            mask = tracker.update(img)

        info = tracker.get_debug_info()
        st = info["state"]
        global_state[st] += 1
        ep_state[ep][st] += 1

        if fr == 0:
            ep_init_ok[ep] = st == "tracking"
        if st == "lost" and ep not in ep_lost_frame:
            ep_lost_frame[ep] = fr

        if fr in FIGURE_FRAMES:
            stage_rows.append((ep, fr, st, info["area"], info["bbox"], info["reason"]))

        if fr in FIGURE_FRAMES:
            all_blue = blue_threshold_mask(img, cfg.mask)
            rec = recolor_to_green(img, mask) if st == "tracking" else img
            tag = f"cam{ep}_f{fr}"
            save_figure(img, all_blue, mask, rec, info["bbox"], st, os.path.join(args.out, "debug", f"{tag}.png"))

    # ---- aggregate stats ----
    print("=" * 72)
    print("Aggregate stats:")
    print(f"  frames tracked      = {n_frames}")
    for st in ["tracking", "occluded", "lost", "uninit"]:
        if global_state[st]:
            print(f"    {st:<12} {global_state[st]:6d}  ({100.0 * global_state[st] / n_frames:5.1f}%)")
    n_init_ok = sum(1 for v in ep_init_ok.values() if v)
    print(f"  init success rate   = {n_init_ok}/{len(ep_init_ok)}")
    lost_eps = sorted(ep_lost_frame.keys())
    print(f"  lost episodes       = {len(lost_eps)}  {lost_eps}")
    for ep in lost_eps:
        print(f"      ep{ep} lost at frame {ep_lost_frame[ep]}")

    print("=" * 72)
    print("Per-stage table (ep, frame, state, area, bbox, reason):")
    for (ep, fr, st, area, bbox, reason) in stage_rows:
        print(f"  ep{ep} f{fr:<4} {st:<9} area={area:<5} bbox={bbox} reason={reason}")

    # ---- wrapper invariance ----
    print("=" * 72)
    print("State/action invariance through ObjectColorAugmentedDataset:")
    aug = ObjectColorAugmentedDataset(ds, tracker_config=cfg)
    for idx in [0, 200, 400, 600, 1000]:
        b = ds[idx]
        a = aug[idx]
        for key in [STATE_KEY, ACTION_KEY]:
            ok = torch.equal(b[key], a[key])
            assert ok, f"{key} changed at idx={idx}!"
        print(f"  idx={idx:<5} state/action equal=True")
    # outside-mask invariance on the up camera for a recolored frame
    changed = 0
    for idx in range(0, min(len(ds), 500), 7):
        b = ds[idx]
        a = aug[idx]
        if not torch.equal(b[cam], a[cam]):
            changed += 1
            diff = (a[cam] - b[cam]).abs()
            bbox = aug.tracks[int(b["episode_index"])][int(b["frame_index"])]
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                inside = diff[:, y0:y1, x0:x1].max().item()
                outside = diff.clone()
                outside[:, y0:y1, x0:x1] = 0
                outside_max = outside.max().item()
                assert outside_max == 0.0, f"mask-outside changed at idx={idx}!"
    print(f"  recolored frames in sample={changed}; mask-outside max diff = 0.0 (asserted)")

    # ---- DataLoader batch ----
    print("=" * 72)
    print("DataLoader batch check:")
    dl = DataLoader(aug, batch_size=4, shuffle=False)
    batch = next(iter(dl))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k:<24} {tuple(v.shape)}  dtype={v.dtype}")
    base_batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in [STATE_KEY, ACTION_KEY]}
    for k in [STATE_KEY, ACTION_KEY]:
        assert torch.equal(batch[k], base_batch[k]), f"{k} changed in batch!"
    print(f"  state/action batch equal to base: True")

    print("=" * 72)
    print("DONE")


if __name__ == "__main__":
    main()
