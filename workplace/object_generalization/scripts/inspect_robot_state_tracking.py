"""Inspect the robot-state-aware FRONT tracker and compare it to a baseline.

Does NOT train anything. For each episode it runs the state-aware tracker
(``RobotStatePrior`` + ``FrontInteractionObjectTracker``) and, in the same
pass, a *baseline* that reproduces the old single-joint elbow gate, so the two
are compared on identical frames.

Reports:
  * overall tracking success (tracking frames / total),
  * per-phase tracking success (approach / grasp / move / place), using the
    state-aware phase as the frame segmentation for BOTH trackers,
  * target-switch counts (large center jump between consecutive tracking frames),
  * 5-panel debug figures: Original / Blue candidates / Selected target /
    Green recolor / phase+confidence+bbox.

Usage:
    python scripts/inspect_robot_state_tracking.py --root /home/zjq/lerobot/dataset/newdata
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

from object_generalization.masks.blue_object_mask import blue_threshold_mask  # noqa: E402
from object_generalization.masks.front_interaction_tracker import (  # noqa: E402
    FrontInteractionObjectTracker,
    FrontTrackerConfig,
)
from object_generalization.masks.robot_state_prior import (  # noqa: E402
    RobotStatePrior,
    RobotStatePriorConfig,
)
from object_generalization.transforms.object_recolor import recolor_to_green  # noqa: E402

FRONT_KEY = "observation.images.front"
FIGURE_FRAMES = {0, 90, 150, 180, 220, 240, 260, 300, 360, 420, 460, 500}
SWITCH_THRESHOLD = 60.0  # px
INTERACTING = ["approaching", "grasping", "moving", "placing"]


class BaselinePrior:
    """Reproduces the old single-joint elbow gate: track iff elbow_flex < thr."""

    def __init__(self, thr=20.0):
        self.thr = thr

    def reset(self):
        pass

    def update(self, state):
        s = np.asarray(state.detach().cpu().numpy() if hasattr(state, "detach") else state).reshape(-1)
        elbow = float(s[2])
        active = elbow < self.thr
        return {
            "phase": "moving" if active else "home",
            "confidence": 1.0,
            "grasping": False,
            "releasing": False,
            "arm_height": "down" if active else "up",
            "gripper_state": "holding",
        }


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


def save_figure(original, all_blue, selected_mask, recolored, bbox, phase, conf, st, out_path):
    orig = _to_uint8_hwc(original)
    rec = _to_uint8_hwc(recolored)
    cand = cv2.cvtColor((np.asarray(all_blue).squeeze() > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    sel = cv2.cvtColor((np.asarray(selected_mask).squeeze() > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    track = orig.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(track, (x0, y0), (x1, y1), (0, 0, 255), 2)
    info = np.full(orig.shape, 40, np.uint8)
    cv2.putText(info, f"phase: {phase}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info, f"conf: {conf:.2f}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info, f"tracker: {st}", (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    row = np.hstack([_label(orig, "Original"), _label(cand, "Blue candidates"), _label(sel, "Selected target"), _label(rec, "Green recolor"), _label(info, "State prior")])
    row_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, row_bgr)


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

    # state-aware tracker: full phase prior
    prior = RobotStatePrior(RobotStatePriorConfig())
    sa = FrontInteractionObjectTracker(FrontTrackerConfig())
    # baseline: neutral search/score (no phase multipliers) + elbow gate
    base_cfg = FrontTrackerConfig(
        search_scale_approach=1.0, search_scale_grasp=1.0, search_scale_moving=1.0, search_scale_placing=1.0,
        w_dist_approach=1.0, w_dist_grasp=1.0, w_dist_moving=1.0, w_dist_placing=1.0,
    )
    base = FrontInteractionObjectTracker(base_cfg)
    base_prior = BaselinePrior()

    cur_ep = None
    # per-phase aggregates (phase from the state-aware prior)
    phase_frames = Counter()          # phase -> total frames
    phase_track = defaultdict(Counter)  # phase -> {sa|base -> tracking frames}
    sa_global = Counter()
    base_global = Counter()
    sa_switches = Counter()           # ep -> switch count
    base_switches = Counter()
    sa_lock = {}                      # ep -> first locked frame
    base_lock = {}
    sa_prev = {}
    base_prev = {}
    sa_prev_t = {}
    base_prev_t = {}
    n_frames = 0
    phase_seq = []                    # (ep, fr, phase, conf, sa_state, base_state)

    for idx in range(len(ds)):
        item = ds[idx]
        ep = int(item["episode_index"])
        fr = int(item["frame_index"])
        img = item[FRONT_KEY]
        state = item["observation.state"]
        n_frames += 1

        if ep != cur_ep:
            prior.reset(); sa.reset(); base_prior.reset(); base.reset()
            cur_ep = ep
            sa_prev[ep] = None; sa_prev_t[ep] = False
            base_prev[ep] = None; base_prev_t[ep] = False

        pd = prior.update(state)
        bd = base_prior.update(state)

        if fr == 0:
            sa_mask = sa.initialize(img, pd)
            base_mask = base.initialize(img, bd)
        else:
            sa_mask = sa.update(img, pd)
            base_mask = base.update(img, bd)

        sa_info = sa.get_debug_info()
        base_info = base.get_debug_info()
        phase = pd["phase"]
        sa_st = sa_info["state"]
        base_st = base_info["state"]
        sa_global[sa_st] += 1
        base_global[base_st] += 1
        phase_frames[phase] += 1
        if sa_st == "tracking":
            phase_track[phase]["sa"] += 1
        if base_st == "tracking":
            phase_track[phase]["base"] += 1

        # lock frames
        if sa_info["center"] is not None and sa_st == "tracking" and ep not in sa_lock:
            sa_lock[ep] = fr
        if base_info["center"] is not None and base_st == "tracking" and ep not in base_lock:
            base_lock[ep] = fr

        # switches (large center jump between consecutive tracking frames)
        for tag, info, prev, prev_t, sw in [
            ("sa", sa_info, sa_prev, sa_prev_t, sa_switches),
            ("base", base_info, base_prev, base_prev_t, base_switches),
        ]:
            if info["center"] is not None and info["state"] == "tracking":
                if prev_t[ep] and prev[ep] is not None:
                    d = np.hypot(info["center"][0] - prev[ep][0], info["center"][1] - prev[ep][1])
                    if d > SWITCH_THRESHOLD:
                        sw[ep] += 1
                prev[ep] = info["center"]; prev_t[ep] = True
            else:
                prev_t[ep] = False

        if fr in FIGURE_FRAMES:
            phase_seq.append((ep, fr, phase, pd["confidence"], sa_st, base_st, sa_info["bbox"]))
            all_blue = blue_threshold_mask(img, sa.config.mask)
            rec = recolor_to_green(img, sa_mask) if sa_st == "tracking" else img
            save_figure(img, all_blue, sa_mask, rec, sa_info["bbox"], phase, pd["confidence"], sa_st,
                        os.path.join(args.out, "debug", f"state_aware_cam{ep}_f{fr}.png"))

    # ---- aggregate ----
    print("=" * 72)
    print(f"Overall tracking success (frames={n_frames}):")
    for tag, g in [("state-aware", sa_global), ("baseline(elbow<20)", base_global)]:
        tr = g.get("tracking", 0)
        print(f"  {tag:<22} tracking={tr:5d} ({100.0*tr/n_frames:5.1f}%)  "
              f"awaiting={g.get('awaiting',0):4d}  occluded={g.get('occluded',0):4d}  lost={g.get('lost',0):4d}")

    print("=" * 72)
    print("Per-phase tracking success (segmentation = state-aware phase):")
    print(f"  {'phase':<12} {'frames':>7} | {'state-aware':>20} | {'baseline':>20}")
    for phase in INTERACTING:
        tot = phase_frames.get(phase, 0)
        if tot == 0:
            continue
        sa_tr = phase_track[phase].get("sa", 0)
        base_tr = phase_track[phase].get("base", 0)
        print(f"  {phase:<12} {tot:7d} | {sa_tr:6d} ({100.0*sa_tr/tot:5.1f}%)       | {base_tr:6d} ({100.0*base_tr/tot:5.1f}%)")
    # non-interacting phases
    for phase in ["home", "retracting"]:
        tot = phase_frames.get(phase, 0)
        sa_tr = phase_track[phase].get("sa", 0)
        base_tr = phase_track[phase].get("base", 0)
        print(f"  {phase:<12} {tot:7d} | {sa_tr:6d} ({100.0*sa_tr/tot:5.1f}%)       | {base_tr:6d} ({100.0*base_tr/tot:5.1f}%)  [should be ~0]")

    print("=" * 72)
    print(f"Locked-target frame per episode:  state-aware={sa_lock}  baseline={base_lock}")
    print(f"Target switches (center jump > {SWITCH_THRESHOLD:.0f}px):  state-aware={dict(sa_switches)}  baseline={dict(base_switches)}")

    print("=" * 72)
    print("Per-stage table (ep, frame, phase, conf, sa_state, base_state, bbox):")
    for (ep, fr, phase, conf, sa_st, base_st, bbox) in phase_seq:
        print(f"  ep{ep} f{fr:<4} {phase:<11} conf={conf:.2f}  sa={sa_st:<9} base={base_st:<9} bbox={bbox}")

    print("=" * 72)
    print("DONE")


if __name__ == "__main__":
    main()
