"""Render base-vs-generated spot-check sheets for the augmented dataset.

The 3-episode version hardcoded the frames ``{(0,0),(0,180),(0,300),(1,260),
(2,120)}``, which for the other 47 episodes would render panels where nothing was
recolored -- i.e. the one artifact meant to catch a bad mask would show nothing.

So the frames are chosen from the mask cache instead: for each episode, frame 0
plus the first and last frame the front mask actually covers (roughly "approach"
and "placing"). Episodes with no coverage at all are reported as such rather than
rendered blank.

Frames are reached by absolute index (episode start + frame), so a 50-episode
spot check costs ~200 video decodes instead of a full 25538-frame scan.

Outputs: one ``ep<ep>_sheet.png`` per episode, one ``overview_front.png`` contact
sheet covering all episodes at a glance, and ``spotcheck_report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from object_generalization.scripts import _cache_io  # noqa: E402
from object_generalization.scripts.verify_generated_dataset import (  # noqa: E402
    _episode_from_index,
    _hue_stats,
)

UP = "observation.images.up"
FRONT = "observation.images.front"
CAMS = ((UP, "up"), (FRONT, "front"))
LEGACY_FRAMES = {(0, 0), (0, 180), (0, 300), (1, 260), (2, 120)}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _to_bgr(img_tensor):
    arr = img_tensor.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()
    return cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _label(img, text, h=.0):
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (32, 32, 32), -1)
    cv2.putText(canvas, text, (6, int(h) or 19), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _pick_frames(mask, n_frames, n_pick):
    """Frame 0 plus up to ``n_pick`` frames spread over the mask's non-empty runs."""
    present = np.flatnonzero((mask > 0).any(axis=(1, 2)))
    if present.size == 0:
        return [0]
    if n_pick <= 1:
        picks = [int(present[0])]
    else:
        idx = np.linspace(0, present.size - 1, n_pick).round().astype(int)
        picks = [int(present[i]) for i in idx]
    return sorted({0, *picks})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--gen-root", default="/home/zjq/lerobot/dataset/newdata_green")
    ap.add_argument("--repo-id", default="newdata_green")
    ap.add_argument("--cache", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid"))
    ap.add_argument("--episodes", nargs="+", default=["all"])
    ap.add_argument("--frames-per-episode", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "generated_check", "spotcheck"))
    ap.add_argument("--legacy-frames", action="store_true", help="use the old hardcoded frame set")
    ap.add_argument("--overview-cols", type=int, default=10)
    args = ap.parse_args()

    total = _cache_io.total_episodes(args.base_root)
    episodes = _cache_io.parse_episodes(args.episodes[0] if len(args.episodes) == 1 else args.episodes, total)
    lengths = _cache_io.episode_lengths(args.base_root)
    os.makedirs(args.out, exist_ok=True)

    # Unfiltered, each addressed through its own metadata -- see the note in
    # verify_generated_dataset.main about `episodes=` not filtering meta.episodes.
    base = LeRobotDataset(args.dataset, root=args.base_root)
    gen = LeRobotDataset(args.repo_id, root=args.gen_root)
    if gen.num_episodes != len(episodes):
        raise SystemExit(
            f"generated dataset has {gen.num_episodes} episode(s), expected {len(episodes)}; "
            f"was it generated from the same --episodes list?"
        )
    base_from = _episode_from_index(base)
    gen_from = _episode_from_index(gen)
    gen_pos = _cache_io.gen_positions(episodes)

    report, overview_tiles, no_coverage = {}, [], []
    for ep in episodes:
        masks = {
            short: np.load(os.path.join(args.cache, f"episode_{ep}", f"{short}_mask.npy"), mmap_mode="r")
            for _, short in CAMS
        }
        if args.legacy_frames:
            frames = sorted(fr for e, fr in LEGACY_FRAMES if e == ep)
        else:
            # frame choice driven by the UP mask union -- either camera having
            # coverage means the episode has something to show
            any_mask = np.maximum(masks["up"], masks["front"])
            frames = _pick_frames(any_mask, lengths[ep], args.frames_per_episode)
        if not any((masks[s][f] > 0).any() for f in frames for _, s in CAMS):
            no_coverage.append(ep)

        rows = []
        ep_rec = {"frames": {}, "coverage": {}}
        for _, short in CAMS:
            ep_rec["coverage"][short] = round(float((masks[short] > 0).any(axis=(1, 2)).mean()), 4)
        for fr in frames:
            i, j = base_from[ep] + fr, gen_from[gen_pos[ep]] + fr
            b, g = base[i], gen[j]
            cols = []
            fr_rec = {}
            for cam, short in CAMS:
                m = masks[short][fr]
                bb, gg = _to_bgr(b[cam]), _to_bgr(g[cam])
                if (m > 0).any():
                    cnts, _ = cv2.findContours((np.asarray(m) > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    for img in (bb, gg):
                        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
                    hb, _ = _hue_stats(b[cam], m)
                    hg, _ = _hue_stats(g[cam], m)
                    fr_rec[short] = {"mask_px": int((m > 0).sum()),
                                     "hue_base": round(hb, 1), "hue_gen": round(hg, 1)}
                else:
                    fr_rec[short] = {"mask_px": 0}
                cols += [_label(bb, f"ep{ep} f{fr} {short} base"),
                         _label(gg, f"ep{ep} f{fr} {short} gen")]
            rows.append(np.hstack(cols))
            ep_rec["frames"][str(fr)] = fr_rec

        sheet = np.vstack(rows)
        cv2.imwrite(os.path.join(args.out, f"ep{ep:02d}_sheet.png"), sheet)

        # One small front base|gen pair per episode for the all-episode overview.
        # Frame 0 is the legible choice: the cube sits alone on the table before
        # the arm arrives. The last covered frame shows the cube already inside
        # the container, and the largest-mask frame is the close-up grasp, where
        # the gripper fills the image -- both make a 50-tile overview unscannable.
        f_areas = (masks["front"] > 0).sum(axis=(1, 2))
        covered = np.flatnonzero(f_areas > 0)
        fr0 = 0 if f_areas[0] > 0 else (int(covered[0]) if covered.size else frames[-1])
        i, j = base_from[ep] + fr0, gen_from[gen_pos[ep]] + fr0
        pair = np.hstack([cv2.resize(_to_bgr(base[i][FRONT]), (240, 180)),
                          cv2.resize(_to_bgr(gen[j][FRONT]), (240, 180))])
        overview_tiles.append(_label(pair, f"ep{ep} f{fr0}: front base | gen"))
        report[str(ep)] = ep_rec

    # 10 x N overview, one row per group of episodes
    cols = max(1, args.overview_cols)
    while len(overview_tiles) % cols:
        overview_tiles.append(np.zeros_like(overview_tiles[0]))
    grid = np.vstack([np.hstack(overview_tiles[r * cols:(r + 1) * cols])
                      for r in range(len(overview_tiles) // cols)])
    cv2.imwrite(os.path.join(args.out, "overview_front.png"), grid)

    report["_no_coverage_episodes"] = no_coverage
    with open(os.path.join(args.out, "spotcheck_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"sheets -> {args.out}")
    print(f"no coverage in any camera (nothing to show): {no_coverage}")
    thin = [ep for ep in episodes if report[str(ep)]["coverage"]["front"] < 0.2]
    print(f"episodes with front coverage < 0.2: {thin}")


if __name__ == "__main__":
    main()
