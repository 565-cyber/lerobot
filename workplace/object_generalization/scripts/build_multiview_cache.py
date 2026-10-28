"""Offline multi-view target-mask cache builder.

Runs :class:`MultiViewInteractionTracker` once per episode (sequential, in order)
and writes the per-frame up / front target masks to disk:

    outputs/cache/episode_<ep>/up_mask.npy      # uint8 (T, H, W), 0/255
    outputs/cache/episode_<ep>/front_mask.npy   # uint8 (T, H, W), 0/255
    outputs/cache/episode_<ep>/meta.json        # shapes + per-phase coverage

``ObjectColorAugmentedDataset(camera_mode=..., mask_cache_dir=...)`` then reads
these masks in ``__init__`` (memory-mapped) and recolors the target cube
blue -> green inside ``__getitem__`` — the tracker never runs inside DataLoader
workers, and the multi-view association (up identity + front local matching) is
frozen into the cached masks.

Safety: state/action are untouched; frames with no target mask are returned
unchanged (宁可漏增强，不要错增强).

Usage:
    python scripts/build_multiview_cache.py --root /home/zjq/lerobot/dataset/newdata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

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
from object_generalization.masks.robot_state_prior import (  # noqa: E402
    RobotStatePrior,
    RobotStatePriorConfig,
)

UP_KEY = "observation.images.up"
FRONT_KEY = "observation.images.front"
CAM_SHORT = {UP_KEY: "up", FRONT_KEY: "front"}
INTERACTING = ["approaching", "grasping", "moving", "placing"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache"))
    args = ap.parse_args()

    ds = LeRobotDataset(args.dataset, root=args.root, episodes=args.episodes)
    print(f"Loaded LeRobotDataset('{args.dataset}', episodes={args.episodes}, frames={len(ds)})")

    tracker = MultiViewInteractionTracker(MultiViewTrackerConfig())
    prior = RobotStatePrior(RobotStatePriorConfig())

    cur_ep = None
    up_buf = []
    front_buf = []
    phase_cov = defaultdict(Counter)          # phase -> Counter(status)
    phase_frames = Counter()
    n_recolored = Counter()                   # camera short -> frames with a mask

    def flush(ep):
        if ep is None or not up_buf:
            return
        ep_dir = os.path.join(args.out, f"episode_{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        up = np.stack(up_buf)                 # (T, H, W) uint8
        front = np.stack(front_buf)
        np.save(os.path.join(ep_dir, "up_mask.npy"), up)
        np.save(os.path.join(ep_dir, "front_mask.npy"), front)
        meta = {
            "episode": ep,
            "frames": int(up.shape[0]),
            "height": int(up.shape[1]),
            "width": int(up.shape[2]),
            "camera_mode": "multiview",
            "up_recolored_frames": int(n_recolored["up"]),
            "front_recolored_frames": int(n_recolored["front"]),
        }
        with open(os.path.join(ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  episode_{ep}: {up.shape[0]} frames -> {ep_dir} "
              f"(up {meta['up_recolored_frames']} / front {meta['front_recolored_frames']} recolored)")
        up_buf.clear()
        front_buf.clear()
        n_recolored.clear()

    for idx in range(len(ds)):
        item = ds[idx]
        ep = int(item["episode_index"])
        fr = int(item["frame_index"])
        up_img = item[UP_KEY]
        front_img = item[FRONT_KEY]
        state = item["observation.state"]

        if ep != cur_ep:
            flush(cur_ep)
            tracker.reset()
            prior.reset()
            cur_ep = ep

        pd = prior.update(state)
        phase = pd["phase"]
        phase_frames[phase] += 1

        if fr == 0:
            res = tracker.initialize(up_img, front_img, state)
        else:
            res = tracker.update(up_img, front_img, state)

        up_mask = np.asarray(res["up_mask"]).squeeze()
        front_mask = np.asarray(res["front_mask"]).squeeze()
        up_buf.append((up_mask > 0).astype(np.uint8) * 255)
        front_buf.append((front_mask > 0).astype(np.uint8) * 255)

        phase_cov[phase][res["status"]] += 1
        if (up_mask > 0).any():
            n_recolored["up"] += 1
        if (front_mask > 0).any():
            n_recolored["front"] += 1

    flush(cur_ep)

    # ---- summary ----------------------------------------------------------
    print("=" * 72)
    print("Per-phase front coverage (multiview):")
    print(f"  {'phase':<12} {'frames':>7} {'tracking':>9} {'coverage':>9}")
    for phase in INTERACTING + ["home", "retracting"]:
        tot = phase_frames.get(phase, 0)
        if tot == 0:
            continue
        tr = phase_cov[phase].get("tracking", 0)
        print(f"  {phase:<12} {tot:7d} {tr:9d} {100.0 * tr / tot:8.1f}%")
    print("DONE")


if __name__ == "__main__":
    main()
