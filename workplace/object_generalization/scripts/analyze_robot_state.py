"""Analyze ``observation.state`` from the real ``newdata`` dataset (v1).

Does NOT train anything. Prints the state's shape/dtype, the authoritative
feature names (from the parquet metadata), per-joint min/max/mean/std, and a
per-episode trajectory aligned with the gripper open/close events so we can
infer what each of the 6 dims means *from the data*, not by guessing.

Usage:
    python scripts/analyze_robot_state.py --root /home/zjq/lerobot/dataset/newdata
"""

from __future__ import annotations

import argparse
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

GRIP_EVENT_DELTA = 1.5  # deg/frame: a gripper move larger than this is open/close


def _state_names(ds) -> list[str]:
    feats = ds.meta.features
    st = feats.get("observation.state", {})
    return list(st.get("names", [f"joint[{i}]" for i in range(6)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    ds = LeRobotDataset(args.dataset, root=args.root, episodes=args.episodes)
    names = _state_names(ds)
    it = ds[0]
    s = it["observation.state"]

    print("=" * 72)
    print(f"Dataset: '{args.dataset}'  episodes={args.episodes}  frames={len(ds)}")
    print(f"  state:  shape={tuple(s.shape)}  dtype={s.dtype}")
    print(f"  action: shape={tuple(it['action'].shape)}  dtype={it['action'].dtype}")
    print(f"  feature names (from parquet meta): {names}")

    # ---- per-dim range over the whole dataset ----
    all_s = np.stack([ds[i]["observation.state"].numpy() for i in range(0, len(ds), 2)])
    print("=" * 72)
    print(f"Per-joint statistics (n={all_s.shape[0]} sampled frames):")
    print(f"  {'joint':<16} {'min':>9} {'max':>9} {'mean':>9} {'std':>8}")
    for j, name in enumerate(names):
        col = all_s[:, j]
        print(f"  {name:<16} {col.min():9.2f} {col.max():9.2f} {col.mean():9.2f} {col.std():8.2f}")

    # ---- per-episode trajectory + gripper events ----
    for ep in args.episodes:
        eps = LeRobotDataset(args.dataset, root=args.root, episodes=[ep])
        N = len(eps)
        print("=" * 72)
        print(f"Episode {ep} ({N} frames):")
        print(f"  frame  " + "  ".join(f"{n:<12}" for n in names))
        key_frames = set(range(0, N, 60)) | {N - 1}
        for fr in sorted(key_frames):
            sv = eps[fr]["observation.state"].numpy()
            print(f"  f{fr:<5} " + "  ".join(f"{v:12.2f}" for v in sv))

        print(f"  gripper ({names[-1]}) events (delta > {GRIP_EVENT_DELTA}):")
        prev = None
        for fr in range(N):
            g = float(eps[fr]["observation.state"].numpy()[-1])
            if prev is not None and abs(g - prev) > GRIP_EVENT_DELTA:
                print(f"    f{fr:<5} {prev:6.2f} -> {g:6.2f}  {'OPEN ' if g > prev else 'CLOSE'}")
            prev = g

    print("=" * 72)
    print("DONE")


if __name__ == "__main__":
    main()
