"""Offline SAM2 video-tracking mask cache (replaces the lightweight tracker).

For each episode, runs SAM 2 (Segment Anything 2) video propagation on the UP and
FRONT streams and writes the per-frame target masks to the SAME cache layout the
dataset wrapper already reads:

    outputs/cache/episode_<ep>/up_mask.npy      # uint8 (T, H, W) 0/255
    outputs/cache/episode_<ep>/front_mask.npy   # uint8 (T, H, W) 0/255

Prompting (no manual clicks):
  * UP:    box prompt on the first frame where the up tracker holds the target.
  * FRONT: box prompt on the first frame where the multi-view tracker confidently
           sees the cube (front == "tracking"); SAM2 then propagates FORWARD and
           BACKWARD to cover the whole episode.

The lightweight tracker is used ONLY to auto-generate the prompt frame; the
per-frame masks come entirely from SAM2's learned propagation.

Memory: the predictor is built once and shared; video frames + inference state are
offloaded to CPU to fit a ~10GB GPU on ~660-frame videos.

Safety: state/action are never touched; the masks are passed to the existing
recolor path which is bit-identical outside the mask.

Scaling to all 50 episodes (the 3-episode version aborted the whole run on a
single un-promptable episode):
  * ``--skip-existing`` resumes: an episode is skipped only when BOTH masks are
    present with the right shape, so the pre-existing episodes 0-2 caches
    (valid masks, no ``done.json``) are reused rather than recomputed;
  * a view that cannot be prompted is recorded and left as zeros for that view
    only -- never a stale/guessed prompt;
  * ``--device`` lets the driver put one worker on each GPU via
    ``CUDA_VISIBLE_DEVICES``.

Usage:
    python scripts/build_sam2_cache.py --root /home/zjq/lerobot/dataset/newdata \
        --episodes 0 --front
    python scripts/build_sam2_cache.py --episodes all --front --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SAM2_ROOT = os.path.join(WORKPLACE, "object_generalization", "third_party", "sam2")
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from object_generalization.masks.blue_object_mask import (  # noqa: E402
    BlueMaskConfig,
    _to_hwc_uint8,
    blue_threshold_mask,
    detect_blue_components,
)
from object_generalization.masks.multiview_object_tracker import (  # noqa: E402
    MultiViewInteractionTracker,
    MultiViewTrackerConfig,
)
from object_generalization.scripts import _cache_io  # noqa: E402

UP_KEY = "observation.images.up"
FRONT_KEY = "observation.images.front"
CKPT = os.path.join(WORKPLACE, "object_generalization", "checkpoints", "sam2.1_hiera_base_plus.pt")
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def _write_frames(ds, cam, tmp_dir):
    for idx in range(len(ds)):
        rgb = _to_hwc_uint8(ds[idx][cam])
        cv2.imwrite(os.path.join(tmp_dir, f"{idx:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _blue_fraction(img_tensor, mask):
    if mask is None or not mask.any():
        return 0.0
    rgb = _to_hwc_uint8(img_tensor)
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    px = hsv[mask]
    return float(((px[:, 0] >= 90) & (px[:, 0] <= 140) & (px[:, 1] >= 50) & (px[:, 2] >= 40)).mean())


# SAFETY: SAM2 tracks the *instance* (and can drift off the cube onto the gripper
# or red container, e.g. during grasp/placing). The recolor hue-rotates whatever
# is under the mask, so a drifted mask would corrupt a non-blue region. We
# therefore intersect SAM2's instance mask with the blue threshold mask and keep
# it only while the intersection stays within the cube's area range:
#   cube ~1.3k px (up) / ~7.5k px (front); background blue 20-46k px (front).
# This guarantees we never recolor a non-blue pixel (宁可漏增强，不要错增强).
SAFE_MIN_AREA = 20      # reject noise specks (a grasp sliver may be <100 px)
SAFE_MAX_AREA = 15000   # reject the large background blue region (>=20k px)


def _safe_mask(img_tensor, sam2_mask):
    """SAM2 instance mask restricted to blue pixels, area-gated. uint8 0/255."""
    if sam2_mask is None or not np.asarray(sam2_mask).any():
        return np.zeros((480, 640), np.uint8)
    blue = blue_threshold_mask(img_tensor, BlueMaskConfig())  # (H, W) 0/255
    inter = (np.asarray(sam2_mask) > 0) & (blue > 0)
    a = int(inter.sum())
    if a < SAFE_MIN_AREA or a > SAFE_MAX_AREA:
        return np.zeros((480, 640), np.uint8)
    return (inter.astype(np.uint8) * 255)


def _front_geometry_ok(c, cfg) -> bool:
    """The multiview tracker's own front gate (compact, small, square)."""
    return (
        cfg.front_min_area <= c["area"] <= cfg.front_max_area
        and cfg.front_aspect_min <= c["aspect"] <= cfg.front_aspect_max
        and c["extent"] >= cfg.front_extent_min
    )


def _prompts(ds):
    """Auto-derive the SAM2 box prompts for one episode. Never raises.

    Returns ``{"up": p|None, "front": p|None}`` where each prompt is
    ``{"frame": int, "bbox": [x0,y0,x1,y1], "tier": str}``.

    Two traps the 3-episode version fell into:

      * it ``assert``-ed on both prompts, so a single un-promptable episode
        aborted the whole run. Measured over all 50 episodes, the up view always
        prompts on frame 0, but the front view never *locks* in episodes 3 and 36;
      * ``MultiViewInteractionTracker`` keeps the last ``front_target`` when it
        goes to ``lost`` (``multiview_object_tracker.py:_occlude``), so its bbox
        can be stale -- the *state* must be checked, never the bbox alone.

    For the front view the tracker is authoritative; a geometry-gate candidate is
    only a fallback, used when the tracker never locks. It is safe because the
    background blue box in the front view is ~39k px and fails ``front_max_area``,
    while the cube is 2-7k px and near-square: verified visually on both episodes
    that need it (ep3 f200, ep36 f200 -- the box lands squarely on the cube).
    """
    cfg = MultiViewTrackerConfig()
    mv = MultiViewInteractionTracker(cfg)
    up_prompt = front_prompt = geo_prompt = None
    for idx in range(len(ds)):
        item = ds[idx]
        fr = int(item["frame_index"])
        state = item["observation.state"]
        if fr == 0:
            mv.initialize(item[UP_KEY], item[FRONT_KEY], state)
        else:
            mv.update(item[UP_KEY], item[FRONT_KEY], state)
        dbg = mv.get_debug_info()
        if up_prompt is None and dbg["up_state"] == "tracking" and dbg["up_obs"] is not None:
            up_prompt = {"frame": fr, "bbox": [int(v) for v in dbg["up_obs"]["bbox"]], "tier": "tracker"}
        if front_prompt is None and dbg["front_state"] == "tracking" and dbg["front_bbox"] is not None:
            front_prompt = {"frame": fr, "bbox": [int(v) for v in dbg["front_bbox"]], "tier": "tracker"}
        if geo_prompt is None:
            for c in detect_blue_components(item[FRONT_KEY], cfg.up.mask):
                if _front_geometry_ok(c, cfg):
                    geo_prompt = {"frame": fr, "bbox": [int(v) for v in c["bbox"]], "tier": "geometry"}
                    break
        if up_prompt is not None and front_prompt is not None:
            break  # early exit: the tracker beat the fallback
    return {"up": up_prompt, "front": front_prompt if front_prompt is not None else geo_prompt}


def _propagate(predictor, tmp_dir, prompt_frame, bbox, reverse=False):
    # try/finally: on the happy path reset_state ran anyway, but a mid-propagation
    # exception leaked a ~6.4 GB CPU inference state, so the NEXT episode would
    # allocate a second one -- 50 episodes turned that into an OOM/swap death.
    state = predictor.init_state(
        video_path=tmp_dir, offload_video_to_cpu=True, offload_state_to_cpu=True
    )
    try:
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=prompt_frame,
            obj_id=0,
            box=np.array(list(bbox), dtype=np.float32),
        )
        masks = {}
        for fr, obj_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
            masks[fr] = (logits[0] > 0.0).cpu().numpy().squeeze(0)
        return masks
    finally:
        predictor.reset_state(state)


def _propagate_episode(predictor, tmp_dir, prompt_frame, bbox, n_frames):
    """Bidirectional propagation covering every frame of the episode."""
    fw = _propagate(predictor, tmp_dir, prompt_frame, bbox, reverse=False)
    if prompt_frame == 0:
        # forward-only: byte-identical to the 3-episode runs' up stream
        return {fr: fw.get(fr) for fr in range(n_frames)}
    rv = _propagate(predictor, tmp_dir, prompt_frame, bbox, reverse=True)
    out = {}
    for fr in range(n_frames):
        m = fw.get(fr) if fr >= prompt_frame else rv.get(fr)
        out[fr] = m if m is not None else np.zeros((480, 640), np.uint8)
    return out


def _coverage(masks: dict, n_frames: int) -> float:
    if not masks:
        return 0.0
    present = sum(1 for fr in range(n_frames) if masks.get(fr) is not None and np.asarray(masks[fr]).any())
    return present / max(1, n_frames)


def _process_episode(predictor, args, ep, n_frames):
    """Build both masks for one episode. Returns (masks_dict, ledger_record)."""
    t0 = time.time()
    ds = LeRobotDataset(args.dataset, root=args.root, episodes=[ep])
    prompts = _prompts(ds)
    rec = {
        "episode": ep,
        "n_frames": n_frames,
        "up_prompt": prompts["up"],
        "front_prompt": prompts["front"],
        "device": args.device,
    }

    up_masks = None
    if prompts["up"] is not None:
        with tempfile.TemporaryDirectory(prefix="sam2_up_") as td:
            _write_frames(ds, UP_KEY, td)
            up_masks = _propagate_episode(
                predictor, td, prompts["up"]["frame"], prompts["up"]["bbox"], n_frames
            )
        up_safe = {fr: _safe_mask(ds[fr][UP_KEY], m) for fr, m in up_masks.items()}
    else:
        up_safe = {}

    front_safe = None
    if args.front and prompts["front"] is not None:
        with tempfile.TemporaryDirectory(prefix="sam2_front_") as td:
            _write_frames(ds, FRONT_KEY, td)
            front_masks = _propagate_episode(
                predictor, td, prompts["front"]["frame"], prompts["front"]["bbox"], n_frames
            )
        front_safe = {fr: _safe_mask(ds[fr][FRONT_KEY], m) for fr, m in front_masks.items()}

    # report at the same probe frames the 3-episode runs printed
    if up_masks is not None:
        for fr in [0, 90, 120, 180, 220, 260, 300, 420]:
            if fr >= n_frames:
                continue
            upm = up_masks.get(fr)
            upa = int(np.asarray(upm).sum()) if upm is not None else 0
            upb = _blue_fraction(ds[fr][UP_KEY], upm)
            up_s = int(up_safe[fr].sum()) // 255 if up_safe.get(fr) is not None else 0
            print(f"  f{fr}: up raw={upa} blue={upb:.2f} safe={up_s}", flush=True)

    zeros = np.zeros((n_frames, 480, 640), np.uint8)
    up_arr = (
        np.stack([up_safe.get(fr, np.zeros((480, 640), np.uint8)) for fr in range(n_frames)])
        if up_safe else zeros
    )
    front_arr = (
        np.stack([front_safe.get(fr, np.zeros((480, 640), np.uint8)) for fr in range(n_frames)])
        if front_safe is not None else zeros
    )

    # a view that produced nothing is not an error: the hybrid stage still has the
    # lightweight tracker's mask for it (宁可漏增强，不要错增强)
    if prompts["up"] is None:
        status = "up_prompt_missing"
    elif front_safe is None and args.front:
        status = "front_prompt_missing"
    else:
        status = "ok"
    rec.update(
        status=status,
        up_safe_cov=round(float((up_arr > 0).any(axis=(1, 2)).mean()), 4),
        front_safe_cov=round(float((front_arr > 0).any(axis=(1, 2)).mean()), 4),
        seconds=round(time.time() - t0, 1),
    )
    return {"up": up_arr, "front": front_arr}, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--episodes", nargs="+", default=["all"],
                    help="episode indices, or 'all' (= every episode of the dataset)")
    ap.add_argument("--out", default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache"))
    ap.add_argument("--front", action="store_true", help="also build the front mask (bidirectional)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                    help="skip episodes whose masks are already complete (default: on)")
    ap.add_argument("--max-consecutive-errors", type=int, default=3,
                    help="abort after this many consecutive failed episodes (systemic-bug guard)")
    ap.add_argument("--report", default=None, help="JSONL ledger (default <out>/reports/sam2_status.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="list what would run, then exit")
    args = ap.parse_args()

    report = args.report or os.path.join(args.out, "reports", "sam2_status.jsonl")
    total = _cache_io.total_episodes(args.root)
    episodes = _cache_io.parse_episodes(args.episodes[0] if len(args.episodes) == 1 else args.episodes, total)

    # shape check needs each episode's length; cheap (metadata only, no video decode)
    lengths = _cache_io.episode_lengths(args.root)

    todo = []
    for ep in episodes:
        ep_dir = os.path.join(args.out, f"episode_{ep}")
        status, _ = _cache_io.entry_status(ep_dir, lengths[ep])
        if args.skip_existing and _cache_io.is_done(status):
            print(f"skip episode {ep} ({status}, n={lengths[ep]})")
            continue
        todo.append(ep)

    print(f"{len(episodes)} episode(s) requested, {len(todo)} to build: {todo}", flush=True)
    if args.dry_run or not todo:
        return

    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(CFG, CKPT, device=args.device)
    consecutive_errors = 0

    for ep in todo:
        n_frames = lengths[ep]
        ep_dir = os.path.join(args.out, f"episode_{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        try:
            masks, rec = _process_episode(predictor, args, ep, n_frames)
            _cache_io.atomic_save_npy(os.path.join(ep_dir, "up_mask.npy"), masks["up"])
            _cache_io.atomic_save_npy(os.path.join(ep_dir, "front_mask.npy"), masks["front"])
            # done.json is written LAST: it is the commit point for this episode
            _cache_io.write_done(ep_dir, rec)
            _cache_io.append_jsonl(report, rec)
            consecutive_errors = 0
            print(
                f"  saved episode_{ep}: up{masks['up'].shape} front{masks['front'].shape} "
                f"status={rec['status']} up_cov={rec['up_safe_cov']} front_cov={rec['front_safe_cov']} "
                f"({rec['seconds']}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
            consecutive_errors += 1
            rec = {
                "episode": ep, "n_frames": n_frames, "device": args.device, "status": "error",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            }
            print(f"  ERROR episode {ep}: {rec['error']}", flush=True)
            _cache_io.atomic_save_npy(
                os.path.join(ep_dir, "up_mask.npy"), np.zeros((n_frames, 480, 640), np.uint8)
            )
            _cache_io.atomic_save_npy(
                os.path.join(ep_dir, "front_mask.npy"), np.zeros((n_frames, 480, 640), np.uint8)
            )
            _cache_io.write_done(ep_dir, rec)
            _cache_io.append_jsonl(report, rec)
            if consecutive_errors >= args.max_consecutive_errors:
                print(f"aborting: {consecutive_errors} consecutive errors", file=sys.stderr)
                sys.exit(3)
        finally:
            torch.cuda.empty_cache()

    print(f"done; ledger -> {report}")


if __name__ == "__main__":
    main()
