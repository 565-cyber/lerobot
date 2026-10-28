"""Generate the augmented LeRobot dataset (blue cube -> green cube) to disk.

Reads the base dataset episodes, recolors the tracked cube in the up+front frames
using the offline hybrid mask cache (SAM2 ∪ lightweight, blue-restricted), and
writes a NEW dataset via the official ``LeRobotDataset.create`` / ``add_frame`` /
``save_episode`` write path. State / action / timestamps are copied bit-for-bit;
only the masked blue pixels change (hue rotation blue->green).

A preflight pass checks that every requested episode has a complete hybrid mask of
the right length *before* any directory is created: otherwise a gap in the cache
surfaces as a KeyError after ~20 minutes of encoding, or (worse) is skipped
silently. ``LeRobotDataset.create`` cannot resume into an existing directory
(``LeRobotDatasetMetadata.create`` uses ``mkdir(exist_ok=False)``), so the output
root must not exist -- move the old one aside first.

Output: /home/zjq/lerobot/dataset/newdata_green (same codec AV1, same fps, same
features schema). Load it back with
    LeRobotDataset("newdata_green", root="/home/zjq/lerobot/dataset/newdata_green")

Usage:
    python scripts/generate_augmented_dataset.py                          # all episodes
    python scripts/generate_augmented_dataset.py --episodes 0 1 2
    python scripts/generate_augmented_dataset.py --limit 5 --out /tmp/smoke_green
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.video_utils import VideoEncodingManager  # noqa: E402
from object_generalization.scripts import _cache_io  # noqa: E402
from object_generalization.transforms.object_recolor import recolor_to_green  # noqa: E402

UP = "observation.images.up"
FRONT = "observation.images.front"


def _to_hwc_uint8(tensor):
    """(C,H,W) float [0,1] torch -> (H,W,3) uint8 numpy (the record write format)."""
    arr = tensor.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()
    return (arr * 255.0).astype(np.uint8)


def _clean_features(base_features):
    """Strip the recorded video `info` from the feature schema so the new dataset
    re-derives its own codec info on first video encode."""
    out = {}
    for k, v in base_features.items():
        v2 = dict(v)
        if v2.get("dtype") in ("image", "video"):
            v2.pop("info", None)
        out[k] = v2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--repo-id", default="newdata_green")
    ap.add_argument("--out", default="/home/zjq/lerobot/dataset/newdata_green")
    ap.add_argument("--episodes", nargs="+", default=["all"],
                    help="episode indices, or 'all' (= every episode of the dataset)")
    ap.add_argument("--mask-cache", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid"))
    ap.add_argument("--limit", type=int, default=None, help="frames/episode cap (smoke test)")
    ap.add_argument("--report", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "reports", "generate_status.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"output dir already exists: {args.out} (move it aside first)")

    total = _cache_io.total_episodes(args.root)
    episodes = _cache_io.parse_episodes(args.episodes[0] if len(args.episodes) == 1 else args.episodes, total)
    lengths = _cache_io.episode_lengths(args.root)

    # preflight: refuse to start on an incomplete cache rather than fail mid-encode
    incomplete = []
    for ep in episodes:
        status, _ = _cache_io.entry_status(os.path.join(args.mask_cache, f"episode_{ep}"), lengths[ep])
        if not _cache_io.is_done(status):
            incomplete.append((ep, status))
    if incomplete:
        raise SystemExit(f"incomplete mask cache for {len(incomplete)} episode(s): {incomplete}")

    print(f"{len(episodes)} episode(s), {sum(lengths[e] for e in episodes)} frames -> {args.out}")
    if args.dry_run:
        return

    base = LeRobotDataset(args.dataset, root=args.root, episodes=episodes)
    up_masks = {
        ep: np.load(os.path.join(args.mask_cache, f"episode_{ep}", "up_mask.npy"), mmap_mode="r")
        for ep in episodes
    }
    front_masks = {
        ep: np.load(os.path.join(args.mask_cache, f"episode_{ep}", "front_mask.npy"), mmap_mode="r")
        for ep in episodes
    }

    out = LeRobotDataset.create(
        args.repo_id,
        fps=base.fps,
        features=_clean_features(base.features),
        root=args.out,
        robot_type=base.meta.info.get("robot_type"),
        use_videos=True,
    )

    n_added = 0
    per_ep = {}
    cur_ep = None
    t_ep = time.time()

    with VideoEncodingManager(out):
        for idx in range(len(base)):
            item = base[idx]
            ep = int(item["episode_index"])
            fr = int(item["frame_index"])
            if args.limit is not None and fr >= args.limit:
                continue
            if cur_ep is not None and ep != cur_ep:
                out.save_episode()
                print(f"  episode {cur_ep}: {per_ep[cur_ep]}", flush=True)
                cur_ep = None
            if cur_ep is None:
                cur_ep = ep
                per_ep[ep] = {"frames": 0, "recolored_up": 0, "recolored_front": 0, "seconds": None}
                t_ep = time.time()

            # timestamp is auto-computed by add_frame as frame_index/fps (== base,
            # since the base timestamps are exactly frame_index/30 in float32).
            frame = {
                "observation.state": item["observation.state"],
                "action": item["action"],
                "task": item["task"],
            }
            for cam, masks, key in ((UP, up_masks, "recolored_up"), (FRONT, front_masks, "recolored_front")):
                m = masks[ep][fr]
                if (m > 0).any():
                    frame[cam] = _to_hwc_uint8(recolor_to_green(item[cam], m))
                    per_ep[ep][key] += 1
                else:
                    frame[cam] = _to_hwc_uint8(item[cam])
            out.add_frame(frame)
            n_added += 1
            per_ep[ep]["frames"] += 1
            per_ep[ep]["seconds"] = round(time.time() - t_ep, 1)

        if cur_ep is not None:
            out.save_episode()
            print(f"  episode {cur_ep}: {per_ep[cur_ep]}", flush=True)

    print(f"wrote {n_added} frames -> {args.out}")

    n_rec = sum(v["recolored_up"] + v["recolored_front"] for v in per_ep.values())
    print(f"recolored camera-frames: {n_rec}")
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump({"out": args.out, "episodes": episodes, "total_frames": n_added,
                   "recolored_camera_frames": n_rec, "per_episode": per_ep}, f, indent=2, sort_keys=True)
    print(f"report -> {args.report}")

    # sanity reload
    check = LeRobotDataset(args.repo_id, root=args.out)
    print(f"reloaded: episodes={check.num_episodes} frames={check.num_frames} tasks={list(check.meta.tasks.index)}")


if __name__ == "__main__":
    main()
