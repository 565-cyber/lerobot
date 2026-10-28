"""Human-eye check of the distractor-augmented dataset (Stage 2B).

The verifier proves things numerically; this produces the three pictures that
catch what a number cannot -- a distractor that landed somewhere absurd, a
silhouette that reads as a flat sticker, a shadow pointing the wrong way.

  sheets/     per (base episode, variant): base | expected | written |
              |written - expected|, for frames spread over the episode. The
              objects are static, so a handful of frames is a fair sample; the
              diff column is what shows the re-encode drift is uniform dust
              rather than a second, unnoticed edit.
  placement/  per episode: the plate with `placement` tinted green and every
              variant's footprints (solid) and shadows (thin) outlined. One
              picture answers both "was there anywhere legal to put this" and
              "did it stay inside".
  overview.png  one written frame per generated episode, in a grid.

``_pick_frames``, ``_label`` and ``_episode_from_index`` are imported from the
Stage 2A inspectors rather than reimplemented -- frame selection and dataset
addressing are exactly the two things that go silently wrong when duplicated.

Usage:
    python .../inspect_distractor_dataset.py --episodes 0 1 --out /tmp/vis
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BG = os.path.dirname(HERE)
WORKPLACE = os.path.dirname(BG)
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from background_generalization.distractors.procedural import (  # noqa: E402
    ProceduralConfig,
    ProceduralCutoutProvider,
)
from background_generalization.masks.build_plate import PlateConfig  # noqa: E402
from background_generalization.transforms.distractor_layout import (  # noqa: E402
    FrameCompositor,
    LayoutConfig,
    _paste_max,
    render_background_layer,
    sample_layout,
)
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402
from object_generalization.scripts import _cache_io  # noqa: E402
from object_generalization.scripts.inspect_generated_dataset import (  # noqa: E402
    _label,
    _pick_frames,
)
from object_generalization.scripts.verify_generated_dataset import _episode_from_index  # noqa: E402

UP = "observation.images.up"
DIFF_CEIL = 40.0
"""|written - expected| saturates here. The display scale is FIXED rather than
per-panel: a per-panel max would make a frame with 1/255 of drift look identical
to one where the whole arm moved."""


def _bgr(tensor) -> np.ndarray:
    return cv2.cvtColor(_to_hwc_uint8(tensor), cv2.COLOR_RGB2BGR)


def _scale(img: np.ndarray, f: float) -> np.ndarray:
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def _heat(diff: np.ndarray) -> np.ndarray:
    d = np.clip(diff, 0, DIFF_CEIL) * (255.0 / DIFF_CEIL)
    return cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_TURBO)


def _overlay_placement(plate_rgb: np.ndarray, placement: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2BGR).copy()
    m = np.asarray(placement).squeeze() > 0
    img[m] = (0.55 * img[m] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    return img


def _draw_layout(img: np.ndarray, layout, colour) -> None:
    """Outline each footprint (thick) and its shadow (thin), in place."""
    h, w = img.shape[:2]
    for p in layout.placements:
        for arr, thr, thick in ((p.cutout.alpha, 0.0, 2), (p.cutout.shadow, 0.05, 1)):
            m = np.zeros((h, w), np.float32)
            _paste_max(m, (np.asarray(arr) > thr).astype(np.float32), p.top)
            cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, colour, thick)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visual inspection of the distractor-augmented dataset")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--report-dir", default=os.path.join(BG, "outputs", "distract"))
    ap.add_argument("--out", default=os.path.join(BG, "outputs", "distract_check"))
    ap.add_argument("--episodes", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", type=int, default=None,
                    help="default: every variant")
    ap.add_argument("--frames-per-episode", type=int, default=4)
    ap.add_argument("--zoom", type=float, default=0.5,
                    help="sheet panel scale. At 1.0 a sheet is a 2560x1920 / ~6.5 MB PNG, "
                         "which over K*50 episodes is not a contact sheet but an archive")
    ap.add_argument("--no-overview", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.report_dir, "spec.json")) as f:
        spec = json.load(f)
    requested = args.episodes or spec["episodes"]
    episodes = _cache_io.parse_episodes(requested[0] if len(requested) == 1 else requested,
                                        _cache_io.total_episodes(spec["base_root"]))
    lengths = _cache_io.episode_lengths(spec["base_root"])
    K = int(spec["variants"])
    variants = args.variants if args.variants is not None else list(range(K))
    pos = _cache_io.gen_positions(episodes)
    layout_cfg = LayoutConfig(**spec["layout_config"])
    proc_cfg = ProceduralConfig(**spec["procedural_config"])
    plate_cfg = PlateConfig(**spec["plate_config"])

    base = LeRobotDataset(args.dataset, root=spec["base_root"])
    gen = LeRobotDataset(spec["repo_id"], root=spec["gen_root"])
    base_from, gen_from = _episode_from_index(base), _episode_from_index(gen)
    for d in ("sheets", "placement"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    overview = []
    for ep in episodes:
        pd = os.path.join(spec["plate_root"], f"episode_{ep}")
        plate = np.load(os.path.join(pd, "plate.npy"))
        placement = np.load(os.path.join(pd, "placement.npy"))
        gains = np.load(os.path.join(pd, "gain.npy"))
        warps = np.load(os.path.join(pd, "warp.npy"))
        n = lengths[ep]

        overlay_plate = _overlay_placement(plate, placement)
        thumbnails = []
        for k in variants:
            gi = pos[ep] * K + k
            rng = np.random.default_rng([int(spec["seed"]), int(ep), int(k)])
            layout = sample_layout(placement, ProceduralCutoutProvider(proc_cfg), layout_cfg, rng)
            bg = render_background_layer(plate, layout, layout_cfg, placement)
            comp = FrameCompositor(plate, bg, layout.paint_union(plate.shape[:2]), warps, gains,
                                   cfg=layout_cfg, plate_diff_threshold=spec["plate_diff_threshold"],
                                   layout=layout)
            _draw_layout(overlay_plate, layout, (0, 165, 255))

            rows, last = [], None
            for fr in _pick_frames(np.ones((n, 1, 1), np.uint8), n, args.frames_per_episode):
                ub = _to_hwc_uint8(base[base_from[ep] + fr][UP])
                exp = comp.composite(ub, fr)
                got = last = _to_hwc_uint8(gen[gen_from[gi] + fr][UP])
                diff = np.abs(got.astype(np.float32) - exp.astype(np.float32)).max(axis=2)
                rows.append(np.hstack([
                    _label(cv2.cvtColor(ub, cv2.COLOR_RGB2BGR), f"base ep{ep} f{fr}"),
                    _label(cv2.cvtColor(exp, cv2.COLOR_RGB2BGR), f"expected gen{gi}"),
                    _label(cv2.cvtColor(got, cv2.COLOR_RGB2BGR), f"written gen{gi}"),
                    _label(_heat(diff), f"|written-expected| max {diff.max():.0f}/{int(DIFF_CEIL)}"),
                ]))
            sheet = np.vstack(rows)
            if args.zoom != 1.0:
                sheet = _scale(sheet, args.zoom)
            cv2.imwrite(os.path.join(args.out, "sheets", f"ep{ep}_v{k}.png"), sheet)

            # the LAST sampled frame, which is what the overview tile shows
            thumb = _scale(cv2.cvtColor(last, cv2.COLOR_RGB2BGR), 0.5)
            thumbnails.append(_label(thumb, f"ep{ep} v{k} obj={len(layout)}"))
            if k == 0:
                overview.append(thumbnails[-1])

        # Two files, not one: the safety picture needs full resolution to judge a
        # 24 px object against a 5 px margin, and the variant comparison needs
        # the frames small enough to read side by side.
        cv2.imwrite(os.path.join(args.out, "placement", f"ep{ep}_overlay.png"),
                    _label(overlay_plate,
                           f"ep{ep}: placement=green, footprints=orange (thick), shadows=thin"))
        cv2.imwrite(os.path.join(args.out, "placement", f"ep{ep}_variants.png"),
                    _label(np.hstack(thumbnails), f"ep{ep}: every variant at 0.5x"))
        print(f"ep{ep}: {len(variants)} sheet(s) + placement overlay", flush=True)

    if overview and not args.no_overview:
        tile_h, tile_w = overview[0].shape[:2]
        per_row = min(10, len(overview))
        rows, row = [], []
        for i, t in enumerate(overview):
            row.append(t)
            if len(row) == per_row or i == len(overview) - 1:
                while len(row) < per_row:
                    row.append(np.zeros((tile_h, tile_w, 3), np.uint8))
                rows.append(np.hstack(row))
                row = []
        out = os.path.join(args.out, "overview.png")
        cv2.imwrite(out, np.vstack(rows))
        print(f"overview: {len(overview)} tiles of {tile_w}x{tile_h} -> {out}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
