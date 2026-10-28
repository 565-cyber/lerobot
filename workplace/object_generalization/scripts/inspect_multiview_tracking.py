"""Validate the multi-view (up=master, front=slave) interaction-object tracker.

Does NOT train anything. For each episode it runs, in the same pass:

  * a *before* tracker (the state-aware single-view FrontInteractionObjectTracker)
  * an *after* tracker (MultiViewInteractionTracker: up identity + front matching)

and compares them on identical frames, segmented by robot phase (approach /
grasp / moving / placing). It also:

  * verifies up/front frame synchronization (both co-located in the same row),
  * reports cross-view association (up found vs front matched) per phase,
  * writes 5-panel figures (up original / up selected / front original /
    front selected / front recolor) to prove both cameras track the same object.

Usage:
    python scripts/inspect_multiview_tracking.py --root /home/zjq/lerobot/dataset/newdata
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

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

from object_generalization.masks.front_interaction_tracker import (  # noqa: E402
    FrontInteractionObjectTracker,
    FrontTrackerConfig,
)
from object_generalization.masks.multiview_object_tracker import (  # noqa: E402
    MultiViewInteractionTracker,
    MultiViewTrackerConfig,
)
from object_generalization.masks.robot_state_prior import (  # noqa: E402
    RobotStatePrior,
    RobotStatePriorConfig,
)
from object_generalization.transforms.object_recolor import recolor_to_green  # noqa: E402

UP_KEY = "observation.images.up"
FRONT_KEY = "observation.images.front"
FIGURE_FRAMES = {0, 90, 120, 150, 180, 220, 240, 260, 300, 360, 420, 480}
SWITCH_THRESHOLD = 60.0  # px
INTERACTING = ["approaching", "grasping", "moving", "placing"]


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
    return np.ascontiguousarray(arr)


def _mask_bgr(mask):
    return cv2.cvtColor((np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)


def _label(img, text):
    return cv2.putText(img.copy(), text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def _overlay_bbox(img, bbox):
    out = img.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return out


def save_figure(up_img, up_mask, front_img, front_mask, front_rec, fr, phase, conf, status, out_path):
    up_orig = _to_uint8_hwc(up_img)
    front_orig = _to_uint8_hwc(front_img)
    front_rec = _to_uint8_hwc(front_rec)
    up_sel = _overlay_bbox(up_orig, _bbox_of(up_mask, up_orig.shape[:2]))
    front_sel = _overlay_bbox(front_orig, _bbox_of(front_mask, front_orig.shape[:2]))

    # front recolor (green) panel: overlay the green image where the mask is
    front_green = front_orig.copy()
    if front_mask is not None and (np.asarray(front_mask).squeeze() > 0).any():
        m = (np.asarray(front_mask).squeeze() > 0)
        front_green[m] = front_rec[m]

    title = f"f{fr} {phase} conf={conf:.2f} {status}"
    panels = [
        _label(up_orig, "up original"),
        _label(up_sel, "up selected"),
        _label(front_orig, "front original"),
        _label(front_sel, "front selected"),
        _label(front_green, title),
    ]
    # pad panels to a common height for hstack
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    for i, p in enumerate(panels):
        if p.shape[0] != h or p.shape[1] != w:
            pad = np.zeros((h, w, 3), np.uint8)
            pad[: p.shape[0], : p.shape[1]] = p
            panels[i] = pad
    row = np.hstack(panels)
    row_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, row_bgr)


def _bbox_of(mask, shape):
    if mask is None:
        return None
    m = (np.asarray(mask).squeeze() > 0)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(WORKPLACE, "object_generalization", "outputs"))
    args = ap.parse_args()

    ds = LeRobotDataset(args.dataset, root=args.root, episodes=args.episodes)
    print("=" * 72)
    print(f"Loading LeRobotDataset('{args.dataset}', episodes={args.episodes}, frames={len(ds)})")

    # ---- sync check: up/front co-located in the same row ------------------
    print("=" * 72)
    print("Frame-sync check (up/front must share the same frame):")
    bad = 0
    for idx in range(len(ds)):
        item = ds[idx]
        if UP_KEY not in item or FRONT_KEY not in item:
            bad += 1
    print(f"  frames missing up or front key: {bad} / {len(ds)}")
    if bad:
        print("  -> MISALIGNED: aborting, will not fuse views.")
        return
    print("  -> OK: up and front are co-located in the same row (same episode_index + frame_index).")

    # ---- trackers ---------------------------------------------------------
    shared_prior = RobotStatePrior(RobotStatePriorConfig())
    before = FrontInteractionObjectTracker(FrontTrackerConfig())
    after = MultiViewInteractionTracker(MultiViewTrackerConfig())

    cur_ep = None
    phase_frames = Counter()
    before_track = defaultdict(Counter)   # phase -> Counter(state)
    after_track = defaultdict(Counter)    # phase -> Counter(front_state)
    up_found = Counter()                  # phase -> up tracking frames
    before_switches = Counter()
    after_switches = Counter()
    before_prev = {}; before_prev_t = {}
    after_prev = {}; after_prev_t = {}
    before_lost = Counter()
    after_lost = Counter()
    n_frames = 0
    stage_rows = []

    for idx in range(len(ds)):
        item = ds[idx]
        ep = int(item["episode_index"])
        fr = int(item["frame_index"])
        up_img = item[UP_KEY]
        front_img = item[FRONT_KEY]
        state = item["observation.state"]
        n_frames += 1

        if ep != cur_ep:
            shared_prior.reset()
            before.reset()
            after.reset()
            cur_ep = ep
            before_prev[ep] = None; before_prev_t[ep] = False
            after_prev[ep] = None; after_prev_t[ep] = False

        pd = shared_prior.update(state)          # shared phase label for segmentation
        phase = pd["phase"]

        if fr == 0:
            before_mask = before.initialize(front_img, pd)
            after_res = after.initialize(up_img, front_img, state)
        else:
            before_mask = before.update(front_img, pd)
            after_res = after.update(up_img, front_img, state)

        before_info = before.get_debug_info()
        after_info = after.get_debug_info()
        before_st = before_info["state"]
        after_st = after_info["front_state"]
        up_st = after_info["up_state"]

        phase_frames[phase] += 1
        before_track[phase][before_st] += 1
        after_track[phase][after_st] += 1
        if up_st == "tracking":
            up_found[phase] += 1

        # lost frames
        if before_st == "lost":
            before_lost[ep] += 1
        if after_st == "lost":
            after_lost[ep] += 1

        # switches (consecutive-frame center jumps)
        for tag, info, st, prev, prev_t, sw in [
            ("before", before_info, before_st, before_prev, before_prev_t, before_switches),
            ("after", after_info, after_st, after_prev, after_prev_t, after_switches),
        ]:
            center = info.get("center") if tag == "before" else info.get("front_center")
            if center is not None and st == "tracking":
                if prev_t[ep] and prev[ep] is not None:
                    d = np.hypot(center[0] - prev[ep][0], center[1] - prev[ep][1])
                    if d > SWITCH_THRESHOLD:
                        sw[ep] += 1
                prev[ep] = center; prev_t[ep] = True
            else:
                prev_t[ep] = False

        if fr in FIGURE_FRAMES:
            stage_rows.append((ep, fr, phase, before_st, after_st, up_st,
                               after_info["front_bbox"], after_info["confidence"]))
            front_rec = recolor_to_green(front_img, after_res["front_mask"]) if after_st == "tracking" else front_img
            save_figure(up_img, after_res["up_mask"], front_img, after_res["front_mask"], front_rec,
                        fr, phase, after_info["confidence"], after_st,
                        os.path.join(args.out, "debug", f"multiview_cam{ep}_f{fr}.png"))

    # ---- aggregate ---------------------------------------------------------
    print("=" * 72)
    print(f"Cross-view association + coverage (frames={n_frames}), before=state-aware single-view front, after=multiview:")
    print(f"  {'phase':<12} {'frames':>7} | {'up found':>8} | {'front BEFORE':>13} | {'front AFTER':>12} | {'lost B/A':>8}")
    for phase in INTERACTING + ["home", "retracting"]:
        tot = phase_frames.get(phase, 0)
        if tot == 0:
            continue
        up = up_found.get(phase, 0)
        b_tr = before_track[phase].get("tracking", 0)
        a_tr = after_track[phase].get("tracking", 0)
        b_lost = before_track[phase].get("lost", 0)
        a_lost = after_track[phase].get("lost", 0)
        print(f"  {phase:<12} {tot:7d} | {up:6d} {100.0*up/tot:4.0f}% | {b_tr:6d} {100.0*b_tr/tot:5.1f}% | {a_tr:6d} {100.0*a_tr/tot:5.1f}% | {b_lost}/{a_lost}")

    print("=" * 72)
    print(f"Target switches (> {SWITCH_THRESHOLD:.0f}px): before={dict(before_switches)}  after={dict(after_switches)}")
    print(f"Lost frames total:                before={sum(before_lost.values())}  after={sum(after_lost.values())}")

    print("=" * 72)
    print("Per-stage table (ep, frame, phase, before, after, up, bbox, conf):")
    for (ep, fr, phase, b_st, a_st, up_st, bbox, conf) in stage_rows:
        print(f"  ep{ep} f{fr:<4} {phase:<11} before={b_st:<9} after={a_st:<9} up={up_st:<9} bbox={bbox} conf={conf:.2f}")

    print("=" * 72)
    print("DONE")


if __name__ == "__main__":
    main()
