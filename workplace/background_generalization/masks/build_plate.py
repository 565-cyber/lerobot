"""Per-episode background plate, temporal-difference frequency, and placement map.

Replaces the v1 ``build_masks.py`` (grayscale absolute difference against a
hand-captured "empty workspace" reference image), which could never run here:

  * that reference image does not exist,
  * the UP camera has auto-exposure on (measured gain 0.933-1.071 within one
    episode), so a fixed gray threshold fires on the whole frame at once,
  * static robot parts sit in the reference and get classified as *background*,
    which would have them painted over -- an outright wrong augmentation,
  * the FRONT camera rides the arm, so reference subtraction is meaningless
    there.

What this builds instead, for the UP camera only (measured bolted: homography to
frame 0 stays within 5.6 px / 0.995-1.019 scale / 0.5 deg over 568 frames):

    gain[t]     per-frame brightness gain (60th percentile -> frame 0's level)
    warp[t]     euclidean transform frame-0 coords -> frame-t coords
    plate       per-pixel temporal median of the aligned, gain-normalized frames
    freq        fraction of frames in which a pixel is far from the plate
    placement   where a distractor MAY be pasted: `freq <= place_freq_max`,
                minus the cube's trajectory, minus the red container, eroded by
                a safety margin

``placement`` is the only authority on distractor sites. It is deliberately
conservative: a pixel only qualifies if the arm *and its shadow* barely touch it
across the whole episode, so a pasted object can never land where the recorded
trajectory would have collided with it.

The pipeline never re-colours the whole background -- it pastes distractors and
leaves every other pixel bit-identical. ``placement`` is what makes that safe.

Cache contract mirrors ``build_sam2_cache.py``: atomic writes, a ``done.json``
committed LAST, a JSONL ledger, and ``--skip-existing`` so a re-run resumes.

Note the file is named ``placement.npy``, NOT ``up_mask.npy``: in the sibling
``object_generalization/`` pipeline that name means "the blue cube", and the same
name meaning "allowed background" one directory over is a trap for the reader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402
from object_generalization.scripts import _cache_io  # noqa: E402

UP = "observation.images.up"


@dataclass
class PlateConfig:
    """Every threshold lives here -- no magic numbers in the code below."""

    # --- photometric normalization -----------------------------------------
    brightness_percentile: float = 60.0
    """Percentile used as the frame's brightness level. 60 sits safely inside
    the table surface; the arm and its shadow are darker and do not drag it."""

    # --- what counts as "not background" in a frame -------------------------
    diff_threshold: float = 18.0
    """Gray-level distance from the plate above which a pixel is 'foreground'
    for that frame. Includes the arm AND its shadow, so it over-covers -- the
    safe direction (it can only hide a distractor, never paint over the arm)."""

    # --- placement map ------------------------------------------------------
    place_freq_max: float = 0.10
    """A pixel is a legal distractor site only if it is far-from-plate in at
    most this fraction of frames. The arm and its shadow sweep ~69% of the
    frame, so the free region is confined to the left/right margins and the top
    strip of the UP view; the container's exclusion and the margin below cut it
    further -- see the README table.

    It was 0.05. Two separate things bound this and only one is a correctness
    floor:

      * Hard, at 0.5: the plate is a per-pixel temporal MEDIAN. Past the halfway
        point the arm is there in a majority of frames and the median IS the arm
        -- the plate becomes a picture of the arm, and a distractor composited
        from it is an outright wrong augmentation.
      * Soft, well below that: occlusion means a distractor is simply not drawn
        while the arm covers it, so what decides whether an object looks stable
        is how often the pixel it stands on differs from the plate. At 0.05 the
        pixel matches the plate in >=95% of frames and the only thing that ever
        hides the object is an unambiguous visit by the arm. Raise it and objects
        start living on the arm's shadow GRADIENT, where the residual hovers
        around diff_threshold, `fg` flips, and the object blinks mid-episode --
        the same defect as the removed ECC warp, arriving by another door.

    Measured by sweeping it on episodes 0/1/5 with everything else fixed
    (mid-episode hidden runs / legal area on ep1):

        0.05   0 0 0  10.0%      0.20   5 4 0  32.0%
        0.10   0 0 0  20.8%      0.30   9 5 1  42.8%
        0.50   8 5 5  59.9%      0.80   6 7 7  70.8%

    0.05 and 0.10 are indistinguishable on the metric that matters and 0.10 is
    twice the area on the tightest episode; 0.20 is where it breaks. So 0.10,
    and the raw free area is 26.9% (ep0) / 21.2% (ep1)."""

    place_erode_px: int = 5
    """Safety margin, as a RADIUS in pixels (eroded as a disk).

    Explicitly a radius, not a kernel size: `np.ones((9,9))` reaches 5.7 px at
    its corners, so a "9" reads as 9 px but acts as 4-5. This is a safety
    parameter, so it is written in the units it is reasoned about in.
    Cost of raising it: the largest free blob shrinks fast (ep0 largest blob
    12115 px at r=4 -> 8962 px at r=9), because the right-hand strip is only
    ~75 px wide."""

    exclude_dilate_px: int = 8
    """Radius (px) by which the cube and container footprints are grown before
    being subtracted from the free region.

    Must be at least the compositing feather radius, or a feathered distractor
    edge can bleed back onto the container. Measured: this term is the one that
    matters -- the container INTERIOR is static, so `freq` there is ~0 and it
    would otherwise be classified as a legal site; excluding it removes 16%
    (ep0) / 28% (ep1) of the raw free area. The cube term is nearly inert
    (0.1221 -> 0.1214 on ep0) because the cube's footprint already sits inside
    the high-frequency arm swath; it is kept as a safety net for episodes where
    it does not."""

    min_placement_area: int = 3000
    """Below this many legal pixels the episode cannot host a distractor."""

    # --- red container ------------------------------------------------------
    red_hsv_a: tuple = (0, 90, 70)
    red_hsv_b: tuple = (10, 255, 255)
    red_hsv_c: tuple = (170, 90, 70)
    red_hsv_d: tuple = (179, 255, 255)
    red_min_area: int = 5000

    # --- per-frame alignment: deliberately NONE -----------------------------
    #
    # No warp is estimated. Every frame is used as-is. This is not an omission,
    # and the earlier version of this file did estimate one -- it was removed
    # because it measured worse than doing nothing on every episode tried.
    #
    # The UP camera is static: <= 5.6 px of homography to frame 0 over a whole
    # episode. ECC maximises correlation over the WHOLE frame, and the arm
    # covers most of it (freq > 0.1 over ~70% of pixels), so on many frames the
    # best correlation is found by following the ARM. Seeding each frame from
    # the previous estimate lets that walk: measured over 50 episodes, 19 hit
    # the 25 px clamp and the median episode had 21% of frames displaced by
    # more than 5 px -- against <= 5.6 px of real motion.
    #
    # Two things followed from that, and both are fixed by not warping:
    #
    #   * the distractor is composited at to_frame_t(alpha, warp), so a bad
    #     warp DISPLACES it -- by up to 58 px, more than the object is wide.
    #     That is the stepping/flicker visible in the generated video. With no
    #     warp the object sits on the same pixels in every frame.
    #   * "aligned" frames were less aligned than raw ones, so the median plate
    #     came out blurred (Laplacian variance 46-71, against 100-120 unwarped)
    #     and freq was inflated, shrinking the legal area (ep0: 6.1% warped,
    #     19.2% unwarped; ep1 9.5%/13.6%; ep5 12.9%/20.0%).
    #
    # Leaving the residual <= 5.6 px of real motion uncorrected is the cheaper
    # error by a wide margin. `max_camera_shift_px` is the bound the decision
    # rests on, and the verifier checks the recorded (identity) warp against it.
    #
    # A masked second pass was tried and is worse, not better: restricting ECC
    # to the static background leaves only smooth low-texture table, where the
    # objective is ill-conditioned, and the drift grew (t95 25 px -> 50 px).
    max_camera_shift_px: float = 6.0
    """Largest frame-0 -> frame-t shift the UP camera is allowed to have.

    Measured, not chosen for comfort: <= 5.6 px homography over a whole episode
    on episode 0. Nothing estimates a warp any more, so this is not a cutoff --
    it is the number the "no alignment" decision rests on."""

    # --- plate estimation ---------------------------------------------------
    plate_subsample: int = 150
    """Frames used for the median. The plate only needs a robust per-pixel
    centre; 150 evenly-spaced frames give that while keeping the float64
    median copy well under a gigabyte. ``freq`` still uses every frame."""

    # --- misc ---------------------------------------------------------------
    shape_hw: tuple = (480, 640)
    min_frames: int = 30
    """Episodes shorter than this are not worth a temporal median."""


def _frame_gain(gray: np.ndarray, cfg: PlateConfig) -> np.ndarray:
    return float(np.percentile(gray, cfg.brightness_percentile))


def _static_warps(frames_gray, cfg: PlateConfig):
    """Identity warps: frame-0 coords == frame-t coords, by construction.

    The UP camera does not move (<= ``max_camera_shift_px`` over a whole
    episode), so the identity is not an approximation of a large quantity --
    it is within a few pixels of the true answer, and it is the *same* few
    pixels on every frame. See ``PlateConfig.max_camera_shift_px`` for why no
    warp is estimated instead.

    The ``(T, 2, 3)`` array is still returned, and still written to
    ``warp.npy``, rather than the callers being taught to skip warping: every
    consumer (the compositor's ``to_frame0``/``to_frame_t``, the verifier's
    ``_residual``) then keeps working unchanged, and the "is the scene static"
    assumption stays a value that is read and checked rather than a special
    case spread across four files.
    """
    n = len(frames_gray)
    return (np.tile(np.eye(2, 3, dtype=np.float32), (n, 1, 1)),
            np.zeros(n, np.float32))


def _disk(radius: int) -> np.ndarray:
    """Elliptical structuring element of the given RADIUS (not kernel size)."""
    r = max(0, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _red_container(plate: np.ndarray, cfg: PlateConfig) -> np.ndarray:
    """Static red container footprint, taken from the plate (it is in the median).

    The HSV bounds go through ``np.asarray`` because a config reconstructed from
    ``done.json`` (which the generator and verifier both do) has lists where the
    dataclass default had tuples, and ``cv2.inRange`` rejects a plain list.
    """
    def _hsv(t):
        return np.asarray(t, np.uint8)

    hsv = cv2.cvtColor(cv2.cvtColor(plate, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, _hsv(cfg.red_hsv_a), _hsv(cfg.red_hsv_b))
         | cv2.inRange(hsv, _hsv(cfg.red_hsv_c), _hsv(cfg.red_hsv_d)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n_lbl <= 1:
        return np.zeros(cfg.shape_hw, np.uint8)
    keep = np.zeros_like(m)
    for j in range(1, n_lbl):
        if stats[j, 4] >= cfg.red_min_area:
            keep[lbl == j] = 255
    return keep


def build_one(ds, cube_union: np.ndarray | None, cfg: PlateConfig) -> dict:
    """Compute plate / freq / placement for one episode already loaded as ``ds``."""
    n = len(ds)
    h, w = cfg.shape_hw

    color_sub, gray_all, gains = [], np.empty((n, h, w), np.uint8), np.empty(n, np.float32)
    for i in range(n):
        rgb = _to_hwc_uint8(ds[i][UP])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        g = _frame_gain(gray, cfg)
        gains[i] = g
        gray_all[i] = gray

    ref_gain = gains[0]
    scale = ref_gain / np.maximum(gains, 1.0)

    idx = np.unique(np.linspace(0, n - 1, min(cfg.plate_subsample, n)).round().astype(int))
    for i in idx:
        rgb = _to_hwc_uint8(ds[int(i)][UP])
        color_sub.append(np.clip(rgb.astype(np.float32) * scale[i], 0, 255).astype(np.uint8))
    color_sub = np.stack(color_sub)

    gray_norm = np.clip(gray_all.astype(np.float32) * scale[:, None, None], 0, 255).astype(np.uint8)
    warps, translations = _static_warps(gray_norm, cfg)

    # WARP_INVERSE_MAP: dst(p) = src(M . p), putting everything in frame-0 coords
    align = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    gray_aligned = np.empty_like(gray_norm)
    color_aligned = np.empty_like(color_sub)
    for i in range(n):
        gray_aligned[i] = cv2.warpAffine(gray_norm[i], warps[i], (w, h),
                                         flags=align, borderMode=cv2.BORDER_REPLICATE)
    for k, i in enumerate(idx):
        color_aligned[k] = cv2.warpAffine(color_sub[k], warps[int(i)], (w, h),
                                          flags=align, borderMode=cv2.BORDER_REPLICATE)

    plate = np.empty((h, w, 3), np.uint8)
    for c in range(3):
        plate[..., c] = np.median(color_aligned[..., c], axis=0).astype(np.uint8)
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)

    counts = np.zeros((h, w), np.uint16)
    for i in range(n):
        counts += (np.abs(gray_aligned[i].astype(np.float32) - plate_gray) > cfg.diff_threshold)
    freq = counts.astype(np.float32) / float(n)

    free = (freq <= cfg.place_freq_max).astype(np.uint8)
    free = cv2.erode(free, _disk(cfg.place_erode_px))
    for name, excl in (("cube", cube_union), ("red", _red_container(plate, cfg))):
        if excl is None:
            continue
        grown = cv2.dilate((np.asarray(excl).squeeze() > 0).astype(np.uint8),
                           _disk(cfg.exclude_dilate_px))
        free[grown > 0] = 0

    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(free, 8)
    blobs = sorted((int(stats[j, 4]) for j in range(1, n_lbl)), reverse=True)
    return {
        "plate": plate,
        "freq": (np.clip(freq, 0, 1) * 255).astype(np.uint8),
        "placement": (free * 255).astype(np.uint8),
        "gain": gains.astype(np.float32),
        "warp": warps.astype(np.float32),
        "placement_frac": round(float((free > 0).mean()), 4),
        "placement_largest_blob": blobs[0] if blobs else 0,
        "placement_blobs": len([b for b in blobs if b >= 400]),
        # Kept in the ledger, always 0.0 now: the consumers that read this field
        # are checking that the scene was treated as static, and a nonzero value
        # would mean a warp had been reintroduced without this comment moving.
        "camera_shift_px": round(float(translations.max()), 2),
        "brightness_gain_range": [round(float(gains.min()), 3), round(float(gains.max()), 3)],
        "diff_far_frac_gt_0.1": round(float((freq > 0.1).mean()), 4),
    }


def _config_key(config) -> str:
    """Canonical JSON for comparing a stored config against a live one."""
    return json.dumps(config, sort_keys=True, default=str)


def _entry_status(ep_dir, n_frames, cfg: PlateConfig):
    """``complete`` | ``stale`` | ``missing`` | ``error`` for this file set.

    Deliberately NOT ``_cache_io.entry_status``: that helper validates
    ``{cam}_mask.npy``, and this pipeline's mask has the opposite polarity and a
    different name on purpose.
    """
    for name, shape in (("plate.npy", (*cfg.shape_hw, 3)), ("placement.npy", cfg.shape_hw),
                        ("freq.npy", cfg.shape_hw), ("gain.npy", (n_frames,)),
                        ("warp.npy", (n_frames, 2, 3))):
        p = os.path.join(ep_dir, name)
        if not os.path.isfile(p):
            return "missing", None
        try:
            got = tuple(np.load(p, mmap_mode="r").shape)
        except (ValueError, OSError):
            return "missing", None
        if got != shape:
            return "missing", None
    rec = _cache_io.read_done(ep_dir)
    if rec is None:
        return "missing", None
    if rec.get("status") == "error":
        return "error", rec
    # A cache built with different parameters is not a cache. Without this the
    # `--skip-existing` path reuses it silently: every file exists and has the
    # right shape, so the entry reads as "complete" while holding numbers that
    # the current config would not produce. This is not hypothetical -- it is
    # how the ECC-warped plates would have survived the removal of the warp.
    #
    # Round-tripped through JSON before comparing because asdict() yields tuples
    # where the stored copy has lists, so a direct == would never match and
    # every entry would read as stale.
    if _config_key(rec.get("config")) != _config_key(asdict(cfg)):
        return "stale", rec
    return "complete", rec


def _cube_union(hybrid_cache: str | None, ep: int, n_frames: int):
    """Union over the episode of the blue cube's up-camera mask, or None.

    The cube is a moving foreground object, so its *trajectory* must be excluded
    from the placement map -- a distractor dropped on the grasp site would make
    the recorded demonstration physically impossible.
    """
    if not hybrid_cache:
        return None
    p = os.path.join(hybrid_cache, f"episode_{ep}", "up_mask.npy")
    if not os.path.isfile(p):
        return None
    arr = np.load(p, mmap_mode="r")
    if tuple(arr.shape) != (n_frames, 480, 640):
        return None
    return (np.asarray(arr) > 0).any(axis=0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the UP-camera background plate and placement map")
    ap.add_argument("--root", default=os.path.join(REPO, "dataset", "newdata"))
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--episodes", nargs="+", default=["all"])
    ap.add_argument("--out", default=os.path.join(WORKPLACE, "background_generalization", "outputs", "plate"))
    ap.add_argument("--hybrid-cache",
                    default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid"),
                    help="object_generalization cache holding up_mask.npy (the cube), for trajectory exclusion")
    ap.add_argument("--report", default=None)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-consecutive-errors", type=int, default=3)
    args = ap.parse_args()

    cfg = PlateConfig()
    report = args.report or os.path.join(args.out, "reports", "plate_status.jsonl")
    total = _cache_io.total_episodes(args.root)
    episodes = _cache_io.parse_episodes(args.episodes[0] if len(args.episodes) == 1 else args.episodes, total)

    errors = 0
    for ep in episodes:
        ep_dir = os.path.join(args.out, f"episode_{ep}")
        ds = LeRobotDataset(args.dataset, root=args.root, episodes=[ep])
        n = len(ds)
        status, _ = _entry_status(ep_dir, n, cfg)
        if args.skip_existing and status == "complete":
            print(f"skip episode {ep} (n={n})", flush=True)
            continue
        if status == "stale":
            print(f"rebuild episode {ep}: cached with a different config", flush=True)
        if args.dry_run:
            print(f"build episode {ep} (n={n})", flush=True)
            continue

        # The config goes in the marker because a downstream consumer has to
        # honour `diff_threshold`: the compositor classifies foreground with its
        # own threshold, and if that one is *higher* it will call pixels
        # background that this map called foreground -- and can then paint a
        # distractor over the arm. Recording it here is what turns "the two
        # constants agree" from an assumption into a checkable fact.
        rec = {"episode": ep, "n_frames": n, "config": asdict(cfg)}
        try:
            if n < cfg.min_frames:
                raise ValueError(f"only {n} frames (min_frames={cfg.min_frames})")
            t0 = time.time()
            out = build_one(ds, _cube_union(args.hybrid_cache, ep, n), cfg)
            os.makedirs(ep_dir, exist_ok=True)
            for name in ("plate", "freq", "placement", "gain", "warp"):
                _cache_io.atomic_save_npy(os.path.join(ep_dir, f"{name}.npy"), out[name])
            rec.update({k: v for k, v in out.items() if k not in ("plate", "freq", "placement", "gain", "warp")})
            rec.update(status="ok", seconds=round(time.time() - t0, 1))
            print(f"ep{ep}: n={n} placement={rec['placement_frac']:.3f} "
                  f"(blobs={rec['placement_blobs']}, largest={rec['placement_largest_blob']}) "
                  f"gain={rec['brightness_gain_range']} "
                  f"cam_shift={rec['camera_shift_px']}px {rec['seconds']}s", flush=True)
            errors = 0
            _cache_io.write_done(ep_dir, rec)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
            errors += 1
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            os.makedirs(ep_dir, exist_ok=True)
            _cache_io.write_done(ep_dir, rec)
            print(f"ERROR episode {ep}: {rec['error']}", flush=True)
            if errors >= args.max_consecutive_errors:
                print(f"{errors} consecutive errors -- aborting (systemic bug?)", file=sys.stderr)
                _cache_io.append_jsonl(report, rec)
                sys.exit(3)
        _cache_io.append_jsonl(report, rec)

    print(f"done; ledger -> {report}")


if __name__ == "__main__":
    main()
