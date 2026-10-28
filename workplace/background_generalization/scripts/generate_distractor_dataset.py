"""Generate the distractor-augmented LeRobot dataset (Stage 2B) to disk.

RoboAgent's second augmentation axis: keep the scene, add clutter. Each base
episode is rendered ``--variants K`` times, each with its own static distractor
layout drawn from the episode's ``placement`` map (built by ``build_plate.py``).
The K copies carry *identical* actions -- only the visual context differs -- so
"50 base episodes vs 50xK augmented" is a clean contrast that does not confound
"more data" with "more diversity".

What changes and what does not
------------------------------
    up    : distractors composited in, and nothing else. Outside the objects'
            alpha the output is BIT-IDENTICAL to the base frame -- the
            compositor is built for exactly that invariant, and ``verify_``
            re-proves it per frame.
    front : content untouched; only the photometric gain the UP camera was
            normalized with is applied (``--no-front-gain`` disables). This is a
            global scalar, so it cannot move or recolour anything, but it does
            mean the two views disagree about the scene -- see the README's
            known-limitations list, which is required reading before rollout.
    state / action / task : copied through untouched.

Seeds and reproducibility
-------------------------
Variant (ep, k) is drawn from ``default_rng([seed, ep, k])``. The generator
records the resulting layout metadata in ``layouts.jsonl`` but NOT the cutout
pixels; the verifier re-runs the same seed and asserts it reproduces the
recorded layout exactly. That way reproducibility is itself checked rather than
assumed, and the sidecar stays a few kB.

Episode numbering
-----------------
``LeRobotDataset.create`` numbers episodes 0..n-1 in write order, so base
episode ``p`` (the p-th requested episode) becomes generated episodes
``p*K .. p*K+K-1``, variant k at ``p*K + k``. ``_cache_io.gen_positions`` cannot
express that multiplication -- the verifier has its own mapping, and so does
:func:`gen_episode_index` here.

``LeRobotDataset.create`` cannot resume into an existing directory
(``LeRobotDatasetMetadata.create`` uses ``mkdir(exist_ok=False)``), so ``--out``
must not exist. Rollback is ``rm -rf`` it plus the plate cache.

Usage:
    python .../generate_distractor_dataset.py --episodes 0 1 --out /tmp/smoke
    python .../generate_distractor_dataset.py --variants 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

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
from lerobot.datasets.video_utils import VideoEncodingManager  # noqa: E402
from background_generalization.distractors.procedural import (  # noqa: E402
    ProceduralConfig,
    ProceduralCutoutProvider,
)
from background_generalization.masks.build_plate import (  # noqa: E402
    PlateConfig,
    _entry_status,
)
from background_generalization.transforms.distractor_layout import (  # noqa: E402
    FrameCompositor,
    LayoutConfig,
    render_background_layer,
    sample_layout,
)
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402
from object_generalization.scripts import _cache_io  # noqa: E402
from object_generalization.scripts.generate_augmented_dataset import _clean_features  # noqa: E402

UP = "observation.images.up"
FRONT = "observation.images.front"


def gen_episode_index(p: int, k: int, variants: int) -> int:
    """Generated episode index of variant ``k`` of the ``p``-th requested episode."""
    return p * variants + k


def variant_rng(seed: int, ep: int, k: int) -> np.random.Generator:
    """The reproducible stream for one (episode, variant).

    Bit-stream stability for a given seed is what lets the verifier re-derive
    layouts instead of the generator shipping cutout pixels, and the seed is
    derived from the episode number rather than a running counter so that
    ``--episodes 7`` alone reproduces episode 7 exactly.
    """
    return np.random.default_rng([int(seed), int(ep), int(k)])


def front_scale(gains: np.ndarray, t: int) -> float:
    """UP's frame-t normalization factor: ``gains[0] / gains[t]``.

    Exactly the scalar ``build_plate`` normalized the plate with, reused here so
    both cameras receive the same photometric treatment.
    """
    return float(gains[0]) / max(float(gains[t]), 1.0)


def layout_census(episodes, variants, seed, plate_root, layout_cfg, proc_cfg) -> int:
    """Sample every layout without rendering anything. Returns the failure count.

    This exists because "0 objects placed" is the one failure this pipeline has
    that is both silent and fatal to the experiment. The variant is written, it
    verifies (there is no alpha, so every in-memory identity assertion holds
    vacuously, and the placement re-proof has no footprint to check), it looks
    like a normal episode in the overview -- and it is a bit-exact duplicate of
    its base episode wearing an augmented episode number. Spend 2.5 h generating
    150 episodes and a few of them are not augmented at all, while the report
    says 150/150.

    ``sample_layout`` gives up rather than loosening a constraint, so nothing in
    the per-episode log distinguishes "placed 0 because the episode is tight"
    from "placed 0 because the placement map is broken". The census separates
    them: no blobs at all above ``min_blob_area`` is a broken map (and leaves
    ``rejected`` at 0, which is why the rejected counter alone cannot see it),
    whereas blobs that exist but never fit is a tight episode.
    """
    counts, no_blob, no_room = [], [], []
    for ep in episodes:
        try:
            placement = np.load(os.path.join(plate_root, f"episode_{ep}", "placement.npy"))
        except OSError as exc:
            raise SystemExit(f"episode {ep}: cannot read placement.npy ({exc}); "
                             f"run build_plate first") from exc
        for k in range(variants):
            layout = sample_layout(placement,
                                   ProceduralCutoutProvider(proc_cfg),
                                   layout_cfg, variant_rng(seed, ep, k))
            counts.append(len(layout))
            if len(layout) == 0:
                n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(
                    np.asarray(placement).squeeze().astype(np.uint8), 8)
                biggest = max([int(stats[j, 4]) for j in range(1, n_lbl)], default=0)
                (no_blob if biggest < layout_cfg.min_blob_area else no_room).append(
                    (ep, k, layout.rejected, biggest))

    c = np.asarray(counts)
    print(f"  layout census: {len(counts)} layout(s), "
          f"objects min {c.min()} median {int(np.median(c))} max {c.max()} mean {c.mean():.2f}")
    print(f"  objects per layout: " +
          " ".join(f"{v}:{int((c == v).sum())}" for v in range(c.max() + 1)))
    if no_blob:
        print(f"  FAIL {len(no_blob)} layout(s) on a placement map with no blob >= "
              f"{layout_cfg.min_blob_area}px -- these are broken, not tight: {no_blob[:5]}")
    if no_room:
        print(f"  FAIL {len(no_room)} layout(s) had blobs but placed nothing "
              f"(episode, variant, rejects, largest blob): {no_room[:5]}")
    if no_blob or no_room:
        print("  A 0-object variant is a bit-exact duplicate of its base episode: it "
              "still verifies and still lands in the overview, so it would ship as "
              "augmentation that is not augmentation. Do not generate past this.")
    return len(no_blob) + len(no_room)


def load_assets(plate_root: str, ep: int):
    """Plate artifacts plus the config they were built with."""
    d = os.path.join(plate_root, f"episode_{ep}")
    rec = _cache_io.read_done(d) or {}
    return {
        "plate": np.load(os.path.join(d, "plate.npy")),
        "placement": np.load(os.path.join(d, "placement.npy")),
        "gains": np.load(os.path.join(d, "gain.npy")),
        "warps": np.load(os.path.join(d, "warp.npy")),
        "config": rec.get("config"),
    }


def load_cube_masks(hybrid_cache: str | None, ep: int, n_frames: int):
    """Per-frame blue-cube mask, unioned into the compositor's foreground.

    Redundant with ``placement`` (which already excludes the cube's trajectory)
    and kept anyway: it is the one foreground source that does not depend on a
    brightness threshold, so it still protects the cube on a frame where the
    residual happens to be quiet.
    """
    if not hybrid_cache:
        return None
    p = os.path.join(hybrid_cache, f"episode_{ep}", "up_mask.npy")
    if not os.path.isfile(p):
        return None
    arr = np.load(p, mmap_mode="r")
    if tuple(arr.shape) != (n_frames, 480, 640):
        return None
    return arr


def build_variant(assets, layout_cfg, proc_cfg, seed, ep, k, cube, diff_threshold):
    """Sample one layout and the compositor that renders it."""
    rng = variant_rng(seed, ep, k)
    layout = sample_layout(assets["placement"], ProceduralCutoutProvider(proc_cfg),
                           layout_cfg, rng)
    bg_layer = render_background_layer(assets["plate"], layout, layout_cfg, assets["placement"])
    comp = FrameCompositor(
        assets["plate"], bg_layer, layout.paint_union(assets["plate"].shape[:2]),
        assets["warps"], assets["gains"], cube_masks=cube, cfg=layout_cfg,
        plate_diff_threshold=diff_threshold, layout=layout,
    )
    return layout, comp


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the distractor-augmented dataset (Stage 2B)")
    ap.add_argument("--root", default=os.path.join(REPO, "dataset", "newdata"))
    ap.add_argument("--dataset", default="newdata")
    ap.add_argument("--repo-id", default="newdata_distract")
    ap.add_argument("--out", default=os.path.join(REPO, "dataset", "newdata_distract"))
    ap.add_argument("--plate-root",
                    default=os.path.join(BG, "outputs", "plate"))
    ap.add_argument("--hybrid-cache",
                    default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid"),
                    help="object_generalization cache holding up_mask.npy (the cube); omit with '' to skip")
    ap.add_argument("--episodes", nargs="+", default=["all"])
    ap.add_argument("--variants", type=int, default=3,
                    help="copies per base episode (K); the augmented set is 50*K episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="frames/episode cap (smoke test)")
    ap.add_argument("--front-gain", action=argparse.BooleanOptionalAction, default=True,
                    help="apply the UP camera's photometric gain to the front frames too")
    ap.add_argument("--report-dir", default=os.path.join(BG, "outputs", "distract"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.variants < 1:
        raise SystemExit(f"--variants must be >= 1, got {args.variants}")
    cfg = PlateConfig()
    layout_cfg = LayoutConfig()
    proc_cfg = ProceduralConfig()

    total = _cache_io.total_episodes(args.root)
    episodes = _cache_io.parse_episodes(args.episodes[0] if len(args.episodes) == 1 else args.episodes, total)
    lengths = _cache_io.episode_lengths(args.root)

    # Preflight: a gap in the plate cache must surface now, not as a KeyError
    # after an hour of encoding.
    incomplete = []
    for ep in episodes:
        status, _ = _entry_status(os.path.join(args.plate_root, f"episode_{ep}"),
                                  lengths[ep], cfg)
        if not _cache_io.is_done(status):
            incomplete.append((ep, status))
    if incomplete:
        raise SystemExit(f"incomplete plate cache for {len(incomplete)} episode(s): {incomplete}")

    # The compositor's fg_threshold must not exceed the threshold the placement
    # map was built with, or it would call pixels background that placement
    # called foreground and could paint a distractor over the arm. build_plate
    # records its config, so read it instead of trusting the in-code default.
    configs = {ep: load_assets(args.plate_root, ep)["config"] for ep in episodes}
    missing_cfg = [ep for ep, c in configs.items() if not c]
    if missing_cfg:
        print(f"NOTE: plate cache has no recorded config for {missing_cfg} "
              f"(built before configs were recorded); falling back to PlateConfig() defaults")
        diff_threshold = cfg.diff_threshold
    else:
        thresholds = {round(float(c["diff_threshold"]), 6) for c in configs.values()}
        if len(thresholds) != 1:
            raise SystemExit(f"plate cache mixes diff_threshold values {sorted(thresholds)}; "
                             f"rebuild it so the compositor guard is meaningful")
        diff_threshold = thresholds.pop()
    if layout_cfg.fg_threshold > diff_threshold:
        raise SystemExit(f"LayoutConfig.fg_threshold={layout_cfg.fg_threshold} exceeds the plate's "
                         f"diff_threshold={diff_threshold}; the placement map would be unsound")

    n_frames = sum(min(lengths[e], args.limit) if args.limit else lengths[e] for e in episodes)
    print(f"{len(episodes)} episode(s) x {args.variants} variant(s) = "
          f"{len(episodes) * args.variants} generated episode(s), {n_frames * args.variants} frames")
    print(f"  plate={args.plate_root}  diff_threshold={diff_threshold}  front_gain={args.front_gain}")
    print(f"  -> {args.out}")
    if args.dry_run:
        raise SystemExit(layout_census(episodes, args.variants, args.seed, args.plate_root,
                                       layout_cfg, proc_cfg))

    # Checked after the dry run, not before: the census renders nothing, and
    # needing a non-existent --out to ask "what would this place?" is backwards
    # -- the question is most useful precisely when the output already exists.
    if os.path.exists(args.out):
        raise SystemExit(f"output dir already exists: {args.out} (move it aside first)")

    base = LeRobotDataset(args.dataset, root=args.root)
    out = LeRobotDataset.create(
        args.repo_id,
        fps=base.fps,
        features=_clean_features(base.features),
        root=args.out,
        robot_type=base.meta.info.get("robot_type"),
        use_videos=True,
    )

    n_added, cur_ep = 0, None
    per_ep, layout_recs = {}, []
    t_all = time.time()

    with VideoEncodingManager(out):
        for p, ep in enumerate(episodes):
            ds = LeRobotDataset(args.dataset, root=args.root, episodes=[ep])
            n_ep = len(ds) if args.limit is None else min(len(ds), args.limit)
            assets = load_assets(args.plate_root, ep)
            cube = load_cube_masks(args.hybrid_cache, ep, len(ds))

            for k in range(args.variants):
                gi = gen_episode_index(p, k, args.variants)
                t0 = time.time()
                layout, comp = build_variant(assets, layout_cfg, proc_cfg, args.seed,
                                             ep, k, cube, diff_threshold)
                occ, occ_max = [], 0.0
                for fr in range(n_ep):
                    item = ds[fr]
                    up = _to_hwc_uint8(item[UP])
                    a = comp.alpha(up, fr)
                    frame = {
                        "observation.state": item["observation.state"],
                        "action": item["action"],
                        "task": item["task"],
                    }
                    frame[UP] = comp.composite(up, fr, a=a)
                    front = _to_hwc_uint8(item[FRONT])
                    if args.front_gain:
                        s = front_scale(assets["gains"], fr)
                        if s != 1.0:
                            front = np.clip(front.astype(np.float32) * s, 0, 255).astype(np.uint8)
                    frame[FRONT] = front
                    out.add_frame(frame)
                    n_added += 1
                    if layout.placements:
                        o = comp.occlusion_from_alpha(a, fr)
                        occ.append(float(np.mean(o)))
                        occ_max = max(occ_max, max(o))

                out.save_episode()
                rec = {
                    "gen_episode": gi, "base_episode": ep, "variant": k,
                    "seed": args.seed, "frames": n_ep,
                    "n_objects": len(layout), "rejected": layout.rejected,
                    "mean_hidden": round(float(np.mean(occ)), 4) if occ else None,
                    "max_hidden": round(occ_max, 4) if occ else None,
                    "seconds": round(time.time() - t0, 1),
                }
                per_ep[gi] = rec
                layout_recs.append({**rec, "layout": layout.to_record()})
                print(f"  gen{gi} (base {ep}, variant {k}): objects={len(layout)} "
                      f"rejected={layout.rejected} hidden<= {rec['max_hidden']} "
                      f"{rec['seconds']}s", flush=True)
                cur_ep = gi

    print(f"wrote {n_added} frames / {out.num_episodes} episodes -> {args.out} "
          f"({round(time.time() - t_all, 1)}s)")

    os.makedirs(args.report_dir, exist_ok=True)
    spec = {
        "seed": args.seed, "variants": args.variants, "episodes": episodes,
        "base_root": args.root, "gen_root": args.out, "repo_id": args.repo_id,
        "plate_root": args.plate_root, "front_gain": bool(args.front_gain),
        "limit": args.limit, "plate_diff_threshold": diff_threshold,
        "plate_config": asdict(cfg), "layout_config": asdict(layout_cfg),
        "procedural_config": asdict(proc_cfg),
    }
    for name, payload in (("spec.json", spec),
                          ("generate_status.json", {"spec": spec, "per_episode": per_ep,
                                                    "total_frames": n_added})):
        with open(os.path.join(args.report_dir, name), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    # The layout sidecar is written LAST of the three: a half-written dump is
    # worse than none, because a verifier would read it as a short run.
    tmp = os.path.join(args.report_dir, "layouts.jsonl.part")
    with open(tmp, "w") as f:
        for r in layout_recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, os.path.join(args.report_dir, "layouts.jsonl"))
    print(f"reports -> {args.report_dir}")

    check = LeRobotDataset(args.repo_id, root=args.out)
    print(f"reloaded: episodes={check.num_episodes} frames={check.num_frames} "
          f"(expected {len(episodes) * args.variants} / {n_added})")
    if check.num_episodes != len(episodes) * args.variants:
        raise SystemExit(f"episode count {check.num_episodes} != {len(episodes) * args.variants}")


if __name__ == "__main__":
    main()
