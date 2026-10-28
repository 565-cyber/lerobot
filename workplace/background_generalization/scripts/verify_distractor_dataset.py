"""Verify the distractor-augmented dataset (Stage 2B) against the base.

Two phases, split the same way as Stage 2A's verifier and for the same reason --
"decode 3x25538 frames twice and count mismatches" is slower *and* weaker than
an exhaustive parquet check plus a strided pixel check.

  phase 1  exhaustive, parquet-level, seconds. ``state`` / ``action`` /
           ``timestamp`` / ``frame_index`` bit-identity over EVERY generated
           frame. This is the assertion that must never be weakened.
  phase 2  strided pixel work, per (base episode, variant):
             a. rebuild the layout from its recorded seed and require it to
                reproduce ``layouts.jsonl`` exactly -- reproducibility is
                checked, not assumed;
             b. placement re-proof, using only the plate cache: every footprint
                inside ``placement``, and ``placement`` clear of the container
                and the cube's trajectory under the same dilation build_plate
                applied;
             c. in-memory identity -- composite the BASE frame and require the
                result to be bit-identical everywhere the distractor alpha is
                zero (HARD);
             d. the distractor must not be drawn over frame foreground, measured
                against an independently recomputed frame-vs-plate residual;
             e. ``front`` carries the gain ``build_plate`` measured for that
                frame, and nothing else;
             f. AV1 re-encode drift, reported as mean/p99 and NOT gated on max
                (the same lesson as Stage 2A).

Two of these are easy to get subtly wrong in a way that produces a permanent,
believable false alarm, so the shape of each is deliberate:

* (d) tests the residual AFTER the area floor the compositor applies. Against
  the raw residual it flags 1-4 px components -- exactly the speckle
  ``min_fg_area`` exists to drop -- in a handful of frames, which reads as a
  safety failure and is not one. Those pixels are counted separately as
  ``speckle_px`` so the cost of that design choice stays visible.
* (e) gates the median RATIO ``front/base``, not ``|front - base*scale|``. The
  generated frame is AV1-decoded, so a per-pixel difference reaches ~40/255 at a
  sharp edge with the gain exactly right, and gating on max would fail every
  time. The median ratio is ~0 when the gain was applied and O(0.1) when it was
  not, which is the distinction actually worth asserting.

Why phase 1 is not a literal reuse of ``verify_generated_dataset.phase1``
------------------------------------------------------------------------
That function joins base and generated rows one-to-one. Here the generated set
holds ``K`` rows for every base row, and its episode numbering is
``p*K + k`` rather than ``p`` -- neither of which ``_cache_io.gen_positions``
describes. The parquet plumbing (``_read_table`` / ``_stack``) is imported from
that module; the join itself is variant-aware and lives here.

The gates this deliberately does NOT carry over from Stage 2A: ``hue_shift >=
40`` (specific to hue rotation) and ``drift_mean <= 5/255`` outside the mask
(this pipeline changes the background on purpose). Outside-object identity is
asserted in memory, where it is exact, rather than on disk, where AV1 is not.

Exit status is non-zero on any hard failure.
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
from background_generalization.masks.build_plate import (  # noqa: E402
    PlateConfig,
    _red_container,
)
from background_generalization.transforms.distractor_layout import (  # noqa: E402
    FrameCompositor,
    LayoutConfig,
    _drop_small,
    _paste_max,
    render_background_layer,
    sample_layout,
    to_frame0,
    to_frame_t,
)
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402
from object_generalization.scripts import _cache_io  # noqa: E402
from object_generalization.scripts.verify_generated_dataset import (  # noqa: E402
    _episode_from_index,
    _read_table,
    _stack,
)

UP = "observation.images.up"
FRONT = "observation.images.front"


def gen_episode_index(p: int, k: int, variants: int) -> int:
    return p * variants + k


def variant_rng(seed: int, ep: int, k: int) -> np.random.Generator:
    """MUST mirror the generator's; see its docstring for why this is a check."""
    return np.random.default_rng([int(seed), int(ep), int(k)])


def front_scale(gains: np.ndarray, t: int) -> float:
    return float(gains[0]) / max(float(gains[t]), 1.0)


def load_assets(plate_root: str, ep: int) -> dict:
    d = os.path.join(plate_root, f"episode_{ep}")
    return {
        "plate": np.load(os.path.join(d, "plate.npy")),
        "placement": np.load(os.path.join(d, "placement.npy")),
        "freq": np.load(os.path.join(d, "freq.npy")),
        "gains": np.load(os.path.join(d, "gain.npy")),
        "warps": np.load(os.path.join(d, "warp.npy")),
    }


def rebuild_compositor(assets, spec, ep, k, cube):
    """Regenerate the variant from its seed. Returns ``(layout, compositor, bg_layer)``."""
    layout_cfg = LayoutConfig(**spec["layout_config"])
    proc_cfg = ProceduralConfig(**spec["procedural_config"])
    rng = variant_rng(spec["seed"], ep, k)
    layout = sample_layout(assets["placement"], ProceduralCutoutProvider(proc_cfg),
                           layout_cfg, rng)
    bg = render_background_layer(assets["plate"], layout, layout_cfg, assets["placement"])
    comp = FrameCompositor(assets["plate"], bg, layout.paint_union(assets["plate"].shape[:2]),
                           assets["warps"], assets["gains"], cube_masks=cube,
                           cfg=layout_cfg, plate_diff_threshold=spec["plate_diff_threshold"],
                           layout=layout)
    return layout, comp, bg


def _independent_fg(frame_rgb, assets, cube, t, layout_cfg, thr):
    """Frame foreground, rebuilt from the cache by a second implementation.

    The compositor's own ``foreground`` is not evidence that the alpha is
    safely masked -- it is the thing being tested. This re-derives the same
    quantity from ``plate.npy`` / ``gain.npy`` / ``warp.npy`` with its own
    threshold, area floor, dilation and cube union, so that "the distractor is
    never drawn over the arm" becomes a statement about the cache rather than
    about one function's internal consistency.
    """
    fg = _residual(frame_rgb, assets, t, thr).astype(np.uint8)
    if cube is not None:
        fg = np.maximum(fg, (np.asarray(cube[t]).squeeze() > 0).astype(np.uint8))
    if layout_cfg.min_fg_area > 0:
        fg = _drop_small(fg, layout_cfg.min_fg_area)
    r = int(layout_cfg.fg_dilate_px)
    if r > 0:
        fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return fg > 0


def _independent_illum(frame_rgb, assets, t, layout_cfg):
    """Frame illumination ``(k, shadow)`` in frame-t coords, from the cache alone.

    Deliberately not ``FrameCompositor.illumination``: that is the function whose
    decision is under test. This re-derives the ratio from ``plate.npy`` /
    ``gain.npy`` / ``warp.npy`` with its own arithmetic and its own masks, so
    "the shadow was shaded and the arm was erased" stays a claim about the pixels
    rather than about one implementation agreeing with itself.
    """
    f = frame_rgb.astype(np.float32) * front_scale(assets["gains"], t)
    f0 = to_frame0(np.clip(f, 0, 255), assets["warps"][t])
    plate = assets["plate"].astype(np.float32)
    ratio = f0 / np.where(plate < 1.0, 1.0, plate)
    lo, hi = ratio.min(axis=2), ratio.max(axis=2)
    lit = (hi - lo <= layout_cfg.shadow_dev_max)
    lit &= (lo + hi) / 2.0 > layout_cfg.shadow_k_lo
    lit &= (lo + hi) / 2.0 < layout_cfg.shadow_k_hi
    k = to_frame_t(np.where(lit, (lo + hi) / 2.0, 1.0).astype(np.float32), assets["warps"][t])
    sh = to_frame_t(lit.astype(np.uint8), assets["warps"][t]) > 0.5
    return k.astype(np.float32), sh


def _independent_ratio(frame_rgb, assets, t):
    """Unconditional ``frame/plate`` in frame-t coords: ``(ratio, spread)``.

    The same arithmetic as ``_independent_illum`` but WITHOUT the shadow band
    applied, so a pixel is reported even when it falls outside the band. That
    distinction is the point: `_independent_illum` sets every out-of-band pixel
    to exactly 1.0, which erases the evidence needed to ask *why* a run was
    hidden. A core darkened to 0.4 on all three channels leaves the band (below
    `shadow_k_lo`) and would otherwise be indistinguishable from a core that was
    never touched at all.

    ``ratio`` is the channel mean, ``spread`` the channel max minus min; a
    multiplicative shadow keeps `spread` small whatever its depth.
    """
    f = frame_rgb.astype(np.float32) * front_scale(assets["gains"], t)
    f0 = to_frame0(np.clip(f, 0, 255), assets["warps"][t])
    plate = assets["plate"].astype(np.float32)
    q = f0 / np.where(plate < 1.0, 1.0, plate)
    warp = assets["warps"][t]
    ratio = to_frame_t(q.mean(axis=2).astype(np.float32), warp)
    spread = to_frame_t((q.max(axis=2) - q.min(axis=2)).astype(np.float32), warp)
    return ratio.astype(np.float32), spread.astype(np.float32)


def _shadowed_union(frame_rgb, assets, t, layout, layout_cfg):
    """Union of the objects this frame is shaded rather than erased.

    The per-object gate, rebuilt: an object qualifies when most of its core has a
    channel-consistent ratio below 1. Rebuilt from the layout record and the
    cache, not read off the compositor.
    """
    k, sh = _independent_illum(frame_rgb, assets, t, layout_cfg)
    h, w = assets["plate"].shape[:2]
    warp = assets["warps"][t]
    base = to_frame_t(layout.paint_union((h, w)), warp)
    out = np.zeros((h, w), bool)
    for p in layout.placements:
        m = np.zeros((h, w), np.float32)
        _paste_max(m, (p.cutout.alpha > 0).astype(np.float32), p.top)
        core = (to_frame_t(m, warp) > 0.5) & (base >= 0.5)
        if not core.any():
            continue
        lit = sh[core]
        if lit.mean() < layout_cfg.shadow_core_frac:
            continue
        if float(np.median(k[core][lit])) >= 1.0:
            continue
        out |= core
    return out


def _independent_alpha(frame_rgb, assets, cube, t, layout, layout_cfg, thr):
    """The full alpha, from the cache: objects minus the arm, shadow excepted."""
    erase = _independent_fg(frame_rgb, assets, cube, t, layout_cfg, thr)
    if layout.placements:
        _, sh = _independent_illum(frame_rgb, assets, t, layout_cfg)
        r = int(layout_cfg.fg_dilate_px)
        if r > 0:
            sh = cv2.dilate(sh.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        erase = erase & ~(sh & _shadowed_union(frame_rgb, assets, t, layout, layout_cfg))
    h, w = assets["plate"].shape[:2]
    a_base = to_frame_t(layout.paint_union((h, w)), assets["warps"][t])
    return np.clip(a_base * (1.0 - erase.astype(np.float32)), 0.0, 1.0)


def _residual(frame_rgb, assets, t, thr):
    """Frame-vs-plate residual in frame-t coords, recomputed from the cache.

    Deliberately not ``FrameCompositor.foreground``: this is the independent
    second implementation that turns "the alpha is masked against foreground"
    from a claim about the code into a claim about the pixels.
    """
    f = frame_rgb.astype(np.float32) * front_scale(assets["gains"], t)
    gray = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    plate_gray = cv2.cvtColor(assets["plate"], cv2.COLOR_RGB2GRAY).astype(np.float32)
    r0 = (np.abs(to_frame0(gray, assets["warps"][t]) - plate_gray) > thr).astype(np.uint8)
    return to_frame_t(r0, assets["warps"][t]) > 0


def phase1_variants(base_root, gen_root, episodes, variants):
    """Exhaustive parquet identity, variant-aware. Returns ``(fails, summary)``."""
    cols = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    b = _read_table(base_root, cols).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    g = _read_table(gen_root, cols).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    fails = []

    pos = _cache_io.gen_positions(episodes)
    gen2base = {gen_episode_index(p, k, variants): (ep, k)
                for ep, p in pos.items() for k in range(variants)}

    unexpected = sorted(set(int(v) for v in g["episode_index"].unique()) - set(gen2base))
    if unexpected:
        fails.append(f"generated dataset has unexpected episode_index {unexpected}")

    b = b[b["episode_index"].isin(episodes)].reset_index(drop=True)
    g = g[g["episode_index"].isin(gen2base)].reset_index(drop=True)
    g = g.assign(base_episode=g["episode_index"].map(lambda i: gen2base[int(i)][0]),
                 variant=g["episode_index"].map(lambda i: gen2base[int(i)][1]))

    if len(g) != len(b) * variants:
        fails.append(f"row count: base {len(b)} x {variants} = {len(b) * variants} != generated {len(g)}")
        return fails, {}

    n_by_variant = {}
    for k in range(variants):
        bk = b.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        gk = g[g["variant"] == k].sort_values(["base_episode", "frame_index"]).reset_index(drop=True)
        n_by_variant[k] = int(len(gk))
        if len(gk) != len(bk):
            fails.append(f"variant {k}: {len(gk)} rows != base {len(bk)}")
            continue
        # the join relies on both sides being in the same episode order; assert
        # it rather than trusting that "sorted the same way" is enough
        if not np.array_equal(bk["episode_index"].to_numpy(), gk["base_episode"].to_numpy()):
            fails.append(f"variant {k}: episode ordering does not line up with the base")
            continue
        if not np.array_equal(bk["frame_index"].to_numpy(), gk["frame_index"].to_numpy()):
            bad = np.flatnonzero(bk["frame_index"].to_numpy() != gk["frame_index"].to_numpy())[:5]
            fails.append(f"variant {k}: frame_index misaligned at rows {bad.tolist()}")
        for col in ("observation.state", "action", "timestamp"):
            bb, gg = _stack(bk[col]), _stack(gk[col])
            neq = np.flatnonzero((bb != gg).any(axis=1))
            if neq.size:
                r = int(neq[0])
                fails.append(
                    f"variant {k}: {col}: {neq.size} mismatching rows, first at "
                    f"(base episode={int(bk['episode_index'][r])}, frame={int(bk['frame_index'][r])})"
                )
    return fails, {
        "frames": int(len(b)),
        "episodes": int(b["episode_index"].nunique()),
        "variants": int(variants),
        "rows_per_variant": n_by_variant,
        "state_action_timestamp_identical": not fails,
    }


def _layout_matches(record: dict, layout) -> bool:
    """Compare a re-derived layout against ``layouts.jsonl`` entry for entry."""
    if int(record["n"]) != len(layout):
        return False
    for a, b in zip(record["objects"], layout.to_record()["objects"]):
        if a["top"] != b["top"] or a["canvas"] != b["canvas"]:
            return False
        if abs(float(a["anchor"][0]) - b["anchor"][0]) > 1e-6 or \
           abs(float(a["anchor"][1]) - b["anchor"][1]) > 1e-6:
            return False
    return True


def phase2(args, spec, layout_recs, episodes):
    """Strided pixel checks. Returns ``(fails, per_key_stats)``."""
    base = LeRobotDataset(args.dataset, root=spec["base_root"])
    gen = LeRobotDataset(spec["repo_id"], root=spec["gen_root"])
    K = int(spec["variants"])
    pos = _cache_io.gen_positions(episodes)
    base_from, gen_from = _episode_from_index(base), _episode_from_index(gen)

    if gen.num_episodes != len(episodes) * K:
        raise SystemExit(f"generated dataset has {gen.num_episodes} episodes, "
                         f"expected {len(episodes) * K}")

    layouts = {(int(r["base_episode"]), int(r["variant"])): r["layout"] for r in layout_recs}
    layout_cfg = LayoutConfig(**spec["layout_config"])
    thr = float(spec["plate_diff_threshold"])

    per_key, fails = {}, []
    for ep in episodes:
        assets = load_assets(spec["plate_root"], ep)
        container = _red_container(assets["plate"], args.plate_cfg)
        plate_gray = cv2.cvtColor(assets["plate"], cv2.COLOR_RGB2GRAY)
        n = args.lengths[ep]

        # --- placement re-proof: facts about the cache, no compositing --------
        place_b = assets["placement"].squeeze() > 0
        disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
        grown_red = cv2.dilate((container > 0).astype(np.uint8), disk(args.plate_cfg.exclude_dilate_px)) > 0
        if np.any(place_b & grown_red):
            fails.append(f"ep{ep}: placement map overlaps the dilated red container "
                         f"({int((place_b & grown_red).sum())} px)")
        cube_path = os.path.join(args.hybrid_cache, f"episode_{ep}", "up_mask.npy")
        if os.path.isfile(cube_path):
            cu = np.load(cube_path, mmap_mode="r")
            if tuple(cu.shape) == (n, 480, 640):
                grown_cube = cv2.dilate((np.asarray(cu) > 0).any(axis=0).astype(np.uint8),
                                        disk(args.plate_cfg.exclude_dilate_px)) > 0
                if np.any(place_b & grown_cube):
                    fails.append(f"ep{ep}: placement map overlaps the cube's trajectory "
                                 f"({int((place_b & grown_cube).sum())} px)")
        freq_frac = assets["freq"].astype(np.float32) / 255.0
        if place_b.any():
            fmax = float(freq_frac[place_b].max())
            if fmax > args.plate_cfg.place_freq_max + 1e-6:
                fails.append(f"ep{ep}: placement holds a pixel with freq={fmax:.3f} > "
                             f"{args.plate_cfg.place_freq_max}")

        cube = None
        if os.path.isfile(cube_path):
            arr = np.load(cube_path, mmap_mode="r")
            cube = arr if tuple(arr.shape) == (n, 480, 640) else None

        for k in range(K):
            gi = gen_episode_index(pos[ep], k, K)
            rec = layouts.get((ep, k))
            if rec is None:
                fails.append(f"ep{ep} v{k}: no entry in layouts.jsonl")
                continue
            layout, comp, bg = rebuild_compositor(assets, spec, ep, k, cube)
            st = {
                "gen_episode": gi, "base_episode": ep, "variant": k,
                "n_objects": len(layout), "frames_sampled": 0,
                "layout_reproduced": _layout_matches(rec, layout),
                "outside_alpha_diff": 0, "alpha_mask_diff": 0, "speckle_px": 0,
                "dropped_paint_px": 0, "paint_outside_placement": 0, "shadow_px": 0,
                "shadowed_px": 0,
                "drift_mean": [], "drift_p99": [], "container_mean": [], "container_p99": [],
                "front_drift": [], "front_ratio_err": 0.0, "placed_px": 0,
            }
            if not st["layout_reproduced"]:
                fails.append(f"ep{ep} v{k}: re-derived layout does not match layouts.jsonl "
                             f"(recorded {rec['objects']}, got {layout.to_record()['objects']})")

            # --- what the background layer changed, and whether it can be seen --
            # Both in frame-0 coordinates, so no warp is involved and the two
            # masks are directly comparable. This is the pair of facts that
            # makes the layer safe (inside placement) AND complete (nothing the
            # layer baked is dropped by the compositor's alpha). The second one
            # is what a silhouette-only alpha silently violates: the shadows are
            # baked into `bg` and then never blended, and every frame still looks
            # plausible on its own.
            hw = assets["plate"].shape[:2]
            changed0 = np.abs(bg.astype(np.int16) - assets["plate"].astype(np.int16)).max(axis=2) > 0
            paint0 = layout.paint_union(hw) > 0
            shadow0 = layout.shadow_union(hw) > 0
            st["shadow_px"] = int(shadow0.sum())
            st["paint_outside_placement"] = int((changed0 & ~place_b).sum())
            st["dropped_paint_px"] = int((changed0 & ~paint0).sum())
            if st["paint_outside_placement"]:
                fails.append(f"ep{ep} v{k}: the background layer changes "
                             f"{st['paint_outside_placement']} px outside `placement`")
            if st["dropped_paint_px"]:
                fails.append(f"ep{ep} v{k}: the background layer changed "
                             f"{st['dropped_paint_px']} px that the compositor's alpha never "
                             f"blends -- that content is baked and then thrown away")
            if st["shadow_px"] and not np.any(shadow0 & changed0):
                fails.append(f"ep{ep} v{k}: {st['shadow_px']} px of contact shadow were baked but "
                             f"none of them reach the background layer")

            # --- footprints inside the legal region ---------------------------
            for p in layout.placements:
                foot = np.zeros(assets["placement"].shape[:2], bool)
                _paste_max(foot, (p.cutout.alpha > 0).astype(np.float32), p.top)
                st["placed_px"] += int(foot.sum())
                if np.any(foot & ~place_b):
                    fails.append(f"ep{ep} v{k}: a footprint leaves `placement` "
                                 f"({int((foot & ~place_b).sum())} px)")
                # Independent of the placement map: compare what was pasted with
                # the source silhouette. A shortfall means the frame border ate
                # part of the object, which the placement check cannot see.
                raw = int((p.cutout.alpha > 0).sum())
                if int(foot.sum()) != raw:
                    fails.append(f"ep{ep} v{k}: an object is clipped by the frame border "
                                 f"({int(foot.sum())} of {raw} px survive)")

            frames = range(0, n) if args.full_frames else range(0, n, args.stride)
            isum = jsum = 0
            for fr in frames:
                bi = base_from[ep] + fr
                gj = gen_from[gi] + fr
                bit = base[bi]
                if int(bit["episode_index"]) != ep or int(bit["frame_index"]) != fr:
                    fails.append(f"ep{ep} f{fr}: base index {bi} resolved to the wrong frame")
                    continue
                git = gen[gj]
                if int(git["episode_index"]) != gi or int(git["frame_index"]) != fr:
                    fails.append(f"ep{ep} v{k} f{fr}: generated index {gj} resolved to the wrong frame")
                    continue
                st["frames_sampled"] += 1
                ub = _to_hwc_uint8(bit[UP])
                a = comp.alpha(ub, fr)
                exp = comp.composite(ub, fr, a=a)

                # (c) in-memory identity outside the objects -- HARD
                outside = a <= 0
                if np.any(exp[outside] != ub[outside]):
                    bad = int((exp[outside] != ub[outside]).sum())
                    st["outside_alpha_diff"] += bad
                    fails.append(f"ep{ep} v{k} f{fr}: {bad} px outside the alpha are not "
                                 f"bit-identical after compositing")

                # (d) the alpha's masking against frame foreground, cross-checked
                # against an independent rebuild.
                #
                # The obvious phrasing -- "no alpha pixel may sit on frame
                # foreground" -- cannot fail: the compositor defines
                # a = paint * (1 - erase), so testing `(a > 0) & fg` tests a
                # property its own definition guarantees, and reports 0 forever.
                # What is worth testing is whether `erase` is RIGHT, so this
                # rebuilds it from the cache and compares the resulting masks.
                # A disagreement means the gain normalization, the area floor,
                # the dilation, the cube union, the shadow gate or the warp
                # convention has drifted -- any of which could hide the arm from
                # the compositor, or un-hide it.
                #
                # Note `erase`, not `fg`: since the shadow fix the two differ, and
                # the difference is exactly the thing that must not silently
                # regress. An object standing in the arm's shadow is now DRAWN,
                # darkened by the frame's illumination, instead of being deleted
                # (see the module docstring of distractor_layout). So a nonzero
                # `(a > 0) & fg` count is expected and is not a failure; what is
                # asserted is that the arm's body is still erased, by way of the
                # two implementations agreeing on WHICH pixels those are.
                #
                # READ THE SCOPE, though, before citing a 0 here as evidence.
                # The two implementations share the 18.0 threshold by
                # construction (fg_threshold == plate_diff_threshold, and the
                # FrameCompositor guard enforces <=), so this cannot catch a
                # threshold error. And the difference can only be non-zero where
                # the residual comes within `fg_dilate_px` of the paint region:
                # measured over 57 sampled frames of ep0 v0, that opportunity
                # set is 0 px -- on this camera the arm simply never comes close
                # to the legal strip. The real guarantee that no distractor
                # covers the arm is `freq[placement] <= place_freq_max` checked
                # above (every placement pixel was background in >= 95% of
                # frames); this is a consistency check layered on top of it.
                #
                # The shadow gate's own safety case is separate and is NOT a
                # consistency check: it is that `shadow_core_frac` of a core must
                # be channel-neutrally darker than the plate. `shadowed_px` below
                # reports how much of the paint region that ever excuses.
                if layout.placements:
                    resid = _residual(ub, assets, fr, thr)
                    real = _drop_small(resid.astype(np.uint8), layout_cfg.min_fg_area) > 0
                    st["speckle_px"] += int(((a > 0) & resid & ~real).sum())
                    shaded = _shadowed_union(ub, assets, fr, layout, layout_cfg)
                    st["shadowed_px"] += int((shaded & (a > 0)).sum())
                    a_ind = _independent_alpha(ub, assets, cube, fr, layout, layout_cfg, thr)
                    bad = int(((a > 0) != (a_ind > 0)).sum())
                    if bad:
                        st["alpha_mask_diff"] += bad
                        fails.append(f"ep{ep} v{k} f{fr}: the compositor's alpha mask differs from "
                                     f"an independently rebuilt one on {bad} px")

                # (e) front is the base frame times the measured gain. Gated on
                # the RATIO, not on |gen - base*scale|: the generated frame is
                # AV1-decoded, so a per-pixel difference maxes out at ~40/255 on
                # a sharp edge even when the gain is exactly right. The median
                # ratio is robust to that and is what actually distinguishes
                # "gain applied" from "gain wrong or missing".
                fb = _to_hwc_uint8(bit[FRONT])
                fg = _to_hwc_uint8(git[FRONT])
                want = np.clip(fb.astype(np.float32) * front_scale(assets["gains"], fr), 0, 255)
                st["front_drift"].append(float(np.abs(fg.astype(np.float32) - want).mean()))
                # only mid-exposure pixels: near 0 the ratio is unstable, and
                # near 255 the gain saturates and truncates it
                mid = (fb >= 30) & (fb <= 200)
                if mid.any():
                    ratio = float(np.median(fg[mid].astype(np.float32) / fb[mid].astype(np.float32)))
                    st["front_ratio_err"] = max(st["front_ratio_err"], abs(ratio - front_scale(assets["gains"], fr)))

                # (f) AV1 drift, and the container specifically
                gu = _to_hwc_uint8(git[UP])
                d = np.abs(gu.astype(np.float32) - exp.astype(np.float32))
                outside_alpha = ~(a > 0)
                if outside_alpha.any():
                    st["drift_mean"].append(float(d[outside_alpha].mean()))
                    st["drift_p99"].append(float(np.percentile(d[outside_alpha], 99)))
                dc = np.abs(gu.astype(np.float32) - ub.astype(np.float32))[container > 0]
                if dc.size:
                    st["container_mean"].append(float(dc.mean()))
                    st["container_p99"].append(float(np.percentile(dc, 99)))

            def _m(key):
                return round(float(np.mean(st[key])), 3) if st[key] else None

            def _p(key):
                return round(float(np.mean(st[key])), 2) if st[key] else None

            st["drift_mean"], st["drift_p99"] = _m("drift_mean"), _p("drift_p99")
            st["container_mean"], st["container_p99"] = _m("container_mean"), _p("container_p99")
            st["front_drift"] = _m("front_drift")
            st["front_ratio_err"] = round(float(st["front_ratio_err"]), 4)

            if st["alpha_mask_diff"]:
                fails.append(f"ep{ep} v{k}: {st['alpha_mask_diff']} px where the compositor's "
                             f"alpha mask disagrees with an independently rebuilt one")
            if st["front_ratio_err"] > args.front_ratio_max:
                fails.append(f"ep{ep} v{k}: front gain ratio is off by {st['front_ratio_err']} > "
                             f"{args.front_ratio_max} -- was the gain applied at all?")
            if st["container_mean"] is not None and st["container_mean"] > args.container_mean_max:
                fails.append(f"ep{ep} v{k}: red container changed by mean "
                             f"{st['container_mean']}/255 > {args.container_mean_max} "
                             f"(AV1 drift alone is ~1.5)")
            if st["drift_mean"] is not None and st["drift_mean"] > args.drift_mean_max:
                fails.append(f"ep{ep} v{k}: outside-object AV1 drift mean "
                             f"{st['drift_mean']}/255 > {args.drift_mean_max}")
            per_key[gi] = st
    return fails, per_key


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the distractor-augmented dataset (Stage 2B)")
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--report-dir", default=os.path.join(BG, "outputs", "distract"),
                    help="where the generator wrote spec.json / layouts.jsonl")
    ap.add_argument("--hybrid-cache",
                    default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid"))
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="default: whatever spec.json recorded")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--full-frames", action="store_true")
    ap.add_argument("--front-ratio-max", type=float, default=0.035,
                    help="max |median(front/base) - scale|. Measured on the smoke set: 0.0139 "
                         "with the gain applied (a systematic ~1.1%% luma bias from AV1 encoding "
                         "the front video), 0.067 at the smallest scale if the gain were skipped "
                         "altogether. 0.035 sits between them with ~2x margin either way.")
    ap.add_argument("--container-mean-max", type=float, default=3.0,
                    help="mean |gen-base| on the red container, /255 (AV1 drift alone is ~1.5)")
    ap.add_argument("--drift-mean-max", type=float, default=5.0)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.report_dir, "spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(args.report_dir, "layouts.jsonl")) as f:
        layout_recs = [json.loads(line) for line in f if line.strip()]

    # Configs come from spec.json, never from this file's defaults: a verifier
    # that re-derives them from its own constants would silently stop checking
    # anything the moment a default changed.
    args.plate_cfg = PlateConfig(**spec["plate_config"])
    requested = args.episodes or spec["episodes"]
    episodes = _cache_io.parse_episodes(requested[0] if len(requested) == 1 else requested,
                                        _cache_io.total_episodes(spec["base_root"]))
    args.lengths = _cache_io.episode_lengths(spec["base_root"])
    if args.full_frames and spec.get("limit"):
        raise SystemExit(f"--full-frames is meaningless here: the dataset was generated with "
                         f"--limit {spec['limit']}, so only {spec['limit']} frames/episode exist")

    print(f"=== phase 1: parquet identity over {len(episodes)} base episode(s) "
          f"x {spec['variants']} variant(s)")
    fails, summary = phase1_variants(spec["base_root"], spec["gen_root"], episodes, spec["variants"])
    print(f"  frames={summary.get('frames')} episodes={summary.get('episodes')} "
          f"rows/variant={summary.get('rows_per_variant')} "
          f"state/action/timestamp identical={summary.get('state_action_timestamp_identical')}")
    for f in fails:
        print(f"  FAIL {f}")

    print(f"=== phase 2: pixel checks ({'all frames' if args.full_frames else f'stride {args.stride}'})")
    px_fails, per_key = phase2(args, spec, layout_recs, episodes)
    for gi in sorted(per_key):
        r = per_key[gi]
        print(f"  gen{gi} (base {r['base_episode']} v{r['variant']}): "
              f"obj={r['n_objects']} placed={r['placed_px']}px shadow={r['shadow_px']}px "
              f"shaded_not_erased={r['shadowed_px']}px "
              f"repro={r['layout_reproduced']} paint_dropped={r['dropped_paint_px']} "
              f"paint_outside={r['paint_outside_placement']} mask_diff={r['alpha_mask_diff']} "
              f"speckle={r['speckle_px']} front:ratio_err={r['front_ratio_err']} "
              f"drift={r['front_drift']} | up drift={r['drift_mean']}/{r['drift_p99']} "
              f"container={r['container_mean']}/{r['container_p99']}")
    for f in px_fails:
        print(f"  FAIL {f}")

    os.makedirs(os.path.dirname(args.report or os.path.join(args.report_dir, "x")), exist_ok=True)
    report = args.report or os.path.join(args.report_dir, "verify_report.json")
    all_fails = fails + px_fails
    with open(report, "w") as f:
        json.dump({"pass": not all_fails, "failures": all_fails, "phase1": summary,
                   "per_episode": {str(k): v for k, v in per_key.items()}}, f, indent=2, sort_keys=True)
    print(f"report -> {report}")
    print("RESULT:", "PASS" if not all_fails else f"FAIL ({len(all_fails)} problem(s))")
    if all_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
