"""Verify the generated augmented dataset (newdata_green) against the base.

Three phases, because at 25538 frames "decode every frame twice and count the
mismatches" is both slow (~15-30 min) and weaker than it looks -- the old version
printed mismatch COUNTS instead of failing.

  phase 1  exhaustive, parquet-level, seconds: state / action / timestamp / index
           bit-identity over EVERY frame. No video decode, so this is the cheap
           assertion that must never be weakened.
  phase 2  strided pixel analysis: hue inside the mask (blue -> green), drift
           outside the mask (AV1 re-encode), and a dataset-level re-proof of the
           "never recolor a non-blue pixel" invariant. Per episode, per camera.
  phase 3  ``--full-frames``: the same pixel checks on every frame (paranoid mode).

Exit status is non-zero on any hard failure.
"""

from __future__ import annotations

import argparse
import glob
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
from object_generalization.masks.blue_object_mask import (  # noqa: E402
    BlueMaskConfig,
    blue_threshold_mask,
)
from object_generalization.scripts import _cache_io  # noqa: E402

UP = "observation.images.up"
FRONT = "observation.images.front"
CAMS = ((UP, "up"), (FRONT, "front"))

# must match build_sam2_cache.SAFE_MAX_AREA -- the mask area gate
SAFE_MAX_AREA = 15000


def _hue_stats(img_tensor, mask):
    """(mean, count) of HSV hue (0..179) under a bool mask."""
    arr = img_tensor.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    m = np.asarray(mask).squeeze() > 0
    if not m.any():
        return None, 0
    return float(hsv[..., 0][m].mean()), int(m.sum())


def _read_table(root, columns):
    import pandas as pd

    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    if not files:
        raise SystemExit(f"no data parquet under {root}/data")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def _stack(col):
    """Column -> (n_rows, n_dims) float32. ``timestamp`` is scalar, so the reshape
    keeps ``(bb != gg).any(axis=1)`` valid for vector and scalar columns alike."""
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in col])
    return arr.reshape(arr.shape[0], -1)


def phase1(base_root, gen_root, episodes, report):
    """Exhaustive parquet-level identity check, restricted to ``episodes``.

    The generated root legitimately holds a subset during a smoke run, and its
    episodes were renumbered from 0 by the writer, so they are mapped back to
    base episode indices before the row-for-row join (see
    ``_cache_io.gen_positions``). For a full run both mappings are the identity,
    which is why the 3-episode version got away with ignoring them.
    """
    cols = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    b = _read_table(base_root, cols).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    g = _read_table(gen_root, cols).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    fails = []
    base_total_rows = len(b)

    pos = _cache_io.gen_positions(episodes)
    gen2base = {v: k for k, v in pos.items()}
    unexpected = sorted(set(int(v) for v in g["episode_index"].unique()) - set(gen2base))
    if unexpected:
        fails.append(f"generated dataset has unexpected episode_index {unexpected}")

    b = b[b["episode_index"].isin(episodes)].reset_index(drop=True)
    g = g[g["episode_index"].isin(gen2base)].reset_index(drop=True)
    g = g.assign(episode_index=g["episode_index"].map(gen2base))
    g = g.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    if len(b) != len(g):
        # name the offending episodes rather than only the total
        bl = b.groupby("episode_index").size().to_dict()
        gl = g.groupby("episode_index").size().to_dict()
        diff = {ep: (bl.get(ep), gl.get(ep)) for ep in sorted(episodes) if bl.get(ep) != gl.get(ep)}
        fails.append(f"row count {len(b)} (base) != {len(g)} (generated); per-episode (base, gen) = {diff}")
        return fails, {}

    for key in ("episode_index", "frame_index"):
        if not np.array_equal(b[key].to_numpy(), g[key].to_numpy()):
            bad = np.flatnonzero(b[key].to_numpy() != g[key].to_numpy())[:5]
            fails.append(f"{key} misaligned at rows {bad.tolist()}")

    for col in ("observation.state", "action", "timestamp"):
        bb, gg = _stack(b[col]), _stack(g[col])
        neq = np.flatnonzero((bb != gg).any(axis=1))
        if neq.size:
            r = int(neq[0])
            fails.append(
                f"{col}: {neq.size} mismatching rows, first at "
                f"(episode={int(b['episode_index'][r])}, frame={int(b['frame_index'][r])})"
            )

    summary = {
        "frames": int(len(b)),
        "episodes": int(b["episode_index"].nunique()),
        "base_total_rows": int(base_total_rows),
        "state_action_timestamp_identical": not fails,
    }
    # Aggregate confirmation: state/action stats must not move. Only meaningful
    # when the generated file covers the whole base dataset -- a subset run's
    # stats.json describes the subset and is expected to differ.
    if len(episodes) == _cache_io.total_episodes(base_root):
        for name, root in (("base", base_root), ("generated", gen_root)):
            with open(os.path.join(root, "meta", "stats.json")) as f:
                stats = json.load(f)
            summary[f"{name}_state_mean"] = stats["observation.state"]["mean"]
        if summary["base_state_mean"] != summary["generated_state_mean"]:
            fails.append("meta/stats.json observation.state mean changed")
    else:
        summary["stats_comparison"] = "skipped (subset run)"
    return fails, summary


def phase2(base, gen, cache, episodes, lengths, stride, full_frames, report):
    """Strided (or full) pixel checks. Returns (failures, per-episode stats)."""
    masks = {
        ep: {short: np.load(os.path.join(cache, f"episode_{ep}", f"{short}_mask.npy"), mmap_mode="r")
             for _, short in CAMS}
        for ep in episodes
    }
    # absolute dataset index of each episode's frame 0, so pixel frames can be
    # reached directly -- the old loop decoded all 25538 frames before testing
    # `fr % stride == 0`, which cost more than everything else combined.
    # Each dataset is addressed through ITS OWN metadata (they are loaded
    # unfiltered), plus the base->generated episode renumbering.
    base_from = _episode_from_index(base)
    gen_from = _episode_from_index(gen)
    gen_pos = _cache_io.gen_positions(episodes)
    blue_cfg = BlueMaskConfig()

    per_ep, fails = {}, []
    for ep in episodes:
        st = {
            "frames_sampled": 0, "recolored_up": 0, "recolored_front": 0,
            "hue_base": {"up": [], "front": []}, "hue_gen": {"up": [], "front": []},
            "drift_mean": {"up": [], "front": []}, "drift_p99": {"up": [], "front": []},
            "nonblue_pixels": 0, "over_area_masks": 0,
        }
        frames = range(0, lengths[ep]) if full_frames else range(0, lengths[ep], stride)
        for fr in frames:
            i, j = base_from[ep] + fr, gen_from[gen_pos[ep]] + fr
            b, g = base[i], gen[j]
            if int(b["episode_index"]) != ep or int(b["frame_index"]) != fr:
                fails.append(f"ep{ep} f{fr}: base index {i} resolved to the wrong frame")
                continue
            if int(g["episode_index"]) != gen_pos[ep] or int(g["frame_index"]) != fr:
                fails.append(f"ep{ep} f{fr}: generated index {j} resolved to the wrong frame")
                continue
            st["frames_sampled"] += 1
            for cam, short in CAMS:
                m = masks[ep][short][fr]
                msum = int((m > 0).sum())
                if msum == 0:
                    continue
                st[f"recolored_{short}"] += 1
                if msum > SAFE_MAX_AREA:
                    st["over_area_masks"] += 1

                # re-proof of the safety invariant at the dataset level: every
                # recolored pixel must be blue in the ORIGINAL frame
                blue = blue_threshold_mask(b[cam], blue_cfg)
                st["nonblue_pixels"] += int(((m > 0) & (blue == 0)).sum())

                hb, _ = _hue_stats(b[cam], m)
                hg, _ = _hue_stats(g[cam], m)
                if hb is not None and hg is not None:
                    st["hue_base"][short].append(hb)
                    st["hue_gen"][short].append(hg)

                # outside-mask drift is AV1 re-encode only (recolor is strictly inside)
                ba = (b[cam].permute(1, 2, 0).contiguous().cpu().numpy() * 255.0)
                ga = (g[cam].permute(1, 2, 0).contiguous().cpu().numpy() * 255.0)
                diff = np.abs(ba - ga)[~(np.asarray(m).squeeze() > 0)]
                if diff.size:
                    st["drift_mean"][short].append(float(diff.mean()))
                    st["drift_p99"][short].append(float(np.percentile(diff, 99)))

        row = {"episode": ep, "frames_sampled": st["frames_sampled"]}
        for short in ("up", "front"):
            if st["hue_base"][short]:
                hb, hg = float(np.mean(st["hue_base"][short])), float(np.mean(st["hue_gen"][short]))
                row[f"{short}_hue_base"] = round(hb, 1)
                row[f"{short}_hue_gen"] = round(hg, 1)
                row[f"{short}_hue_shift"] = round(hb - hg, 1)
                row[f"{short}_recolored_frames"] = st[f"recolored_{short}"]
                row[f"{short}_drift_mean"] = round(float(np.mean(st["drift_mean"][short])), 3)
                row[f"{short}_drift_p99"] = round(float(np.mean(st["drift_p99"][short])), 2)
            else:
                # no coverage: expected for the episodes the ledger marks degraded
                row[f"{short}_recolored_frames"] = 0
        row["nonblue_pixels"] = st["nonblue_pixels"]
        row["over_area_masks"] = st["over_area_masks"]
        per_ep[ep] = row

        if st["nonblue_pixels"]:
            fails.append(f"ep{ep}: {st['nonblue_pixels']} recolored pixels were NOT blue in the base frame")
        if st["over_area_masks"]:
            fails.append(f"ep{ep}: {st['over_area_masks']} masks exceed the {SAFE_MAX_AREA}px area gate")
        for short in ("up", "front"):
            if st["hue_base"][short] and np.mean(st["hue_base"][short]) - np.mean(st["hue_gen"][short]) < 40:
                fails.append(
                    f"ep{ep} {short}: hue shift "
                    f"{np.mean(st['hue_base'][short]) - np.mean(st['hue_gen'][short]):.1f} < 40"
                )
            if st["drift_mean"][short] and np.mean(st["drift_mean"][short]) > 5.0:
                fails.append(f"ep{ep} {short}: outside-mask mean drift {np.mean(st['drift_mean'][short]):.2f}/255 > 5")
    return fails, per_ep


def _episode_from_index(ds):
    """{episode_index: absolute index of its frame 0} from the episode metadata."""
    import pandas as pd

    eps = ds.meta.episodes
    if hasattr(eps, "to_pandas"):
        eps = eps.to_pandas()
    if isinstance(eps, pd.DataFrame):
        return {int(r.episode_index): int(r.dataset_from_index) for r in eps.itertuples()}
    return {int(k): int(v["dataset_from_index"]) for k, v in eps.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--gen-root", default="/home/zjq/lerobot/dataset/newdata_green")
    ap.add_argument("--repo-id", default="newdata_green")
    ap.add_argument("--cache", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid"))
    ap.add_argument("--episodes", nargs="+", default=["all"])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--full-frames", action="store_true")
    ap.add_argument("--min-hue-shift", type=float, default=40.0)
    ap.add_argument("--report", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "reports", "verify_report.json"))
    args = ap.parse_args()

    total = _cache_io.total_episodes(args.base_root)
    episodes = _cache_io.parse_episodes(args.episodes[0] if len(args.episodes) == 1 else args.episodes, total)
    lengths = _cache_io.episode_lengths(args.base_root)

    print(f"=== phase 1: parquet-level identity over {len(episodes)} episode(s) "
          f"({args.base_root} vs {args.gen_root})")
    fails, summary = phase1(args.base_root, args.gen_root, episodes, args.report)
    print(f"  frames={summary.get('frames')} episodes={summary.get('episodes')} "
          f"state/action/timestamp identical={summary.get('state_action_timestamp_identical')}")
    for f in fails:
        print(f"  FAIL {f}")

    print(f"=== phase 2: pixel checks ({'all frames' if args.full_frames else f'stride {args.stride}'})")
    # loaded WITHOUT an episode filter: `meta.episodes` is never filtered by the
    # `episodes=` argument, so its `dataset_from_index` describes the unfiltered
    # store -- pairing it with a filtered dataset is what makes absolute indexing
    # land on the wrong frame (or out of range) for any subset run.
    base = LeRobotDataset(args.dataset, root=args.base_root)
    gen = LeRobotDataset(args.repo_id, root=args.gen_root)
    if gen.num_episodes != len(episodes):
        raise SystemExit(
            f"generated dataset has {gen.num_episodes} episode(s), expected {len(episodes)}; "
            f"was it generated from the same --episodes list?"
        )
    px_fails, per_ep = phase2(base, gen, args.cache, episodes, lengths, args.stride, args.full_frames, args.report)
    degraded = [ep for ep, r in per_ep.items()
                if r.get("up_recolored_frames", 0) == 0 and r.get("front_recolored_frames", 0) == 0]
    for ep in episodes:
        r = per_ep[ep]
        bits = " ".join(f"{c}={r.get(f'{c}_recolored_frames', 0)}f"
                        + (f"/hue {r[f'{c}_hue_base']}->{r[f'{c}_hue_gen']}" if f"{c}_hue_base" in r else "")
                        for c in ("up", "front"))
        print(f"  ep{ep}: {bits} nonblue={r['nonblue_pixels']}")
    for f in px_fails:
        print(f"  FAIL {f}")
    if degraded:
        print(f"  NOTE {len(degraded)} episode(s) with no coverage at all (expected for degraded ones): {degraded}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    all_fails = fails + px_fails
    with open(args.report, "w") as f:
        json.dump({"pass": not all_fails, "failures": all_fails, "phase1": summary, "per_episode": per_ep},
                  f, indent=2, sort_keys=True)
    print(f"report -> {args.report}")
    print("RESULT:", "PASS" if not all_fails else f"FAIL ({len(all_fails)} problem(s))")
    if all_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
