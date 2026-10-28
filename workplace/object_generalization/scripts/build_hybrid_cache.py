"""Hybrid mask cache: union of SAM2-safe and lightweight multiview masks.

The comparison (compare_sam2_lightweight.py) shows the two methods are
complementary and agree exactly where they both track (IoU = 1.00):

  * FRONT camera: SAM2 strictly dominates (approach 100% vs 83.7%, moving
    88.3% vs 73.1%, home 65% vs 26%) — the learned propagation resolves the
    background-blue ambiguity far better than the geometry gate.
  * UP camera: lightweight strictly dominates (home 100% vs 84.8%, retract
    97.9% vs 69.1%) — a single blue object lives there (no ambiguity), and the
    lightweight tracker re-acquires after grasp occlusion while SAM2 does not.

Both masks are blue-restricted (never a non-blue pixel), so their union can only
recolor the cube — it cannot introduce a wrong region. This script writes the
union to the SAME layout the dataset wrapper reads, under outputs/cache_hybrid/.

Usage:
    python scripts/build_hybrid_cache.py --root /home/zjq/lerobot/dataset/newdata
    python scripts/build_hybrid_cache.py --episodes all --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

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
from object_generalization.masks.multiview_object_tracker import (  # noqa: E402
    MultiViewInteractionTracker,
    MultiViewTrackerConfig,
)
from object_generalization.scripts import _cache_io  # noqa: E402

UP = "observation.images.up"
FRONT = "observation.images.front"


def _blue_only(img, mask):
    """Restrict a tracked mask to pixels that pass the blue threshold.

    The SAM2 masks are already intersected in ``build_sam2_cache._safe_mask``,
    but the lightweight tracker's output is a *tracked* mask, not a per-pixel
    threshold map: it carries a rim of pixels just outside the threshold. That
    is not hypothetical -- episode 39 frame 100 (front) had 210 such pixels, all
    8-adjacent to blue, hue 114.7 (blue) with saturation 71.2 straddling the
    ``sat >= 70`` floor.

    Intersecting here is what makes the union blue-restricted *by construction*
    rather than by assumption, which is the "never recolor a non-blue pixel"
    invariant. It can only shrink the mask (宁可漏增强，不要错增强).
    """
    if mask is None or not np.asarray(mask).any():
        return np.zeros((480, 640), np.uint8)
    blue = blue_threshold_mask(img, BlueMaskConfig())  # (H, W) 0/255
    return (((np.asarray(mask).squeeze() > 0) & (blue > 0)).astype(np.uint8) * 255)


def _sam2_signature(cache_dir, ep):
    """Identity of the SAM2 masks this hybrid entry was built from.

    Stored in the hybrid ``done.json`` and re-checked on skip: a hybrid entry
    built before SAM2 existed (or before SAM2 was re-run) must be rebuilt, or
    ``--skip-existing`` would silently freeze an SAM2-less union.
    """
    sig = {}
    for cam in ("up", "front"):
        p = os.path.join(cache_dir, f"episode_{ep}", f"{cam}_mask.npy")
        if os.path.isfile(p):
            st = os.stat(p)
            sig[cam] = [st.st_size, st.st_mtime_ns]
        else:
            sig[cam] = None
    return sig


def _load_sam2(cache_dir, ep, n_frames):
    """SAM2 masks for one episode, or None when absent/malformed.

    Missing SAM2 data is not fatal: the union stays blue-restricted, so falling
    back to the lightweight mask alone can only *miss* augmentation, never
    introduce a wrong one.
    """
    out = {}
    for cam in ("up", "front"):
        p = os.path.join(cache_dir, f"episode_{ep}", f"{cam}_mask.npy")
        if not os.path.isfile(p):
            return None
        arr = np.load(p, mmap_mode="r")
        if tuple(arr.shape) != (n_frames, 480, 640):
            return None
        out[cam] = np.asarray(arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--episodes", nargs="+", default=["all"],
                    help="episode indices, or 'all' (= every episode of the dataset)")
    ap.add_argument("--sam2-cache", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache"))
    ap.add_argument("--out", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid"))
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true", help="list what would be built, then exit")
    ap.add_argument("--report", default=None, help="JSONL ledger (default <out>/reports/hybrid_status.jsonl)")
    args = ap.parse_args()

    report = args.report or os.path.join(args.out, "reports", "hybrid_status.jsonl")
    total = _cache_io.total_episodes(args.root)
    episodes = _cache_io.parse_episodes(args.episodes[0] if len(args.episodes) == 1 else args.episodes, total)

    for ep in episodes:
        ep_dir = os.path.join(args.out, f"episode_{ep}")
        t0 = time.time()
        ds = LeRobotDataset(args.dataset, root=args.root, episodes=[ep])
        n = len(ds)
        status, prev = _cache_io.entry_status(ep_dir, n)
        sam2_sig = _sam2_signature(args.sam2_cache, ep)
        if args.skip_existing and _cache_io.is_done(status) and (prev or {}).get("sam2_sig") == sam2_sig:
            print(f"skip episode {ep} ({status}, n={n})", flush=True)
            continue
        if args.dry_run:
            has_sam2 = any(v is not None for v in sam2_sig.values())
            print(f"build episode {ep} (n={n}, sam2={'yes' if has_sam2 else 'MISSING'})", flush=True)
            continue

        rec = {"episode": ep, "n_frames": n, "sam2_sig": sam2_sig}
        try:
            sam2 = _load_sam2(args.sam2_cache, ep, n)
            rec["sam2_missing"] = sam2 is None

            mv = MultiViewInteractionTracker(MultiViewTrackerConfig())
            up_lw = np.zeros((n, 480, 640), np.uint8)
            front_lw = np.zeros((n, 480, 640), np.uint8)
            for idx in range(n):
                item = ds[idx]
                fr = int(item["frame_index"])
                if fr == 0:
                    res = mv.initialize(item[UP], item[FRONT], item["observation.state"])
                else:
                    res = mv.update(item[UP], item[FRONT], item["observation.state"])
                up_lw[fr] = _blue_only(item[UP], res["up_mask"])
                front_lw[fr] = _blue_only(item[FRONT], res["front_mask"])

            # union of the two blue-restricted masks (max is safe: both are 0/255)
            up_union = np.maximum(sam2["up"], up_lw) if sam2 else up_lw
            front_union = np.maximum(sam2["front"], front_lw) if sam2 else front_lw

            os.makedirs(ep_dir, exist_ok=True)
            _cache_io.atomic_save_npy(os.path.join(ep_dir, "up_mask.npy"), up_union)
            _cache_io.atomic_save_npy(os.path.join(ep_dir, "front_mask.npy"), front_union)

            for name, u, l in (("up", up_union, up_lw), ("front", front_union, front_lw)):
                cov = float((u > 0).any(axis=(1, 2)).mean())
                cov_lw = float((l > 0).any(axis=(1, 2)).mean())
                cov_sam2 = float((sam2[name] > 0).any(axis=(1, 2)).mean()) if sam2 else 0.0
                rec[f"{name}_cov"] = round(cov, 4)
                rec[f"{name}_cov_lw"] = round(cov_lw, 4)
                rec[f"{name}_cov_sam2"] = round(cov_sam2, 4)
                print(f"ep{ep} {name}: union={cov:.3f} sam2={cov_sam2:.3f} lw={cov_lw:.3f}", flush=True)

            rec.update(status="ok", seconds=round(time.time() - t0, 1))
            _cache_io.write_done(ep_dir, rec)
        except Exception as exc:  # noqa: BLE001 - never abort the batch on one episode
            rec.update(
                status="error", error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            print(f"ERROR episode {ep}: {rec['error']}", flush=True)
            os.makedirs(ep_dir, exist_ok=True)
            _cache_io.atomic_save_npy(os.path.join(ep_dir, "up_mask.npy"), np.zeros((n, 480, 640), np.uint8))
            _cache_io.atomic_save_npy(os.path.join(ep_dir, "front_mask.npy"), np.zeros((n, 480, 640), np.uint8))
            _cache_io.write_done(ep_dir, rec)
        _cache_io.append_jsonl(report, rec)

    print(f"done; ledger -> {report}")


if __name__ == "__main__":
    main()
