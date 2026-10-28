"""Audit every distractor occlusion: is the arm in front of it, or only a shadow?

`mean_hidden` / `max_hidden` (from the generator) and the verifier's checks say
whether the pipeline is self-consistent. They do not say whether hiding an
object was the RIGHT thing to do -- and for a long time it was not: the
compositor erased any distractor the arm's *cast shadow* crossed, because
`foreground` over-covers by design and includes the shadow. That made every
distractor blink out together as the arm entered and left (see the "Shade,
don't erase" section of `transforms/distractor_layout.py` and of `CLAUDE.md`).

That defect was invisible to every gate that existed, because it is not a
consistency failure -- the frames were exactly what the code intended. What
catches it is asking, per occlusion run, WHICH mechanism caused it. So this is
an audit, not a verifier: it changes no pass/fail on the dataset, and it exists
to be re-run after any change to the shadow gate.

The classification, per run, over the object's core. It is deliberately NOT a
shadow/arm split, because those two are not the only possibilities and the
interesting failures live in between. `lit` -- the fraction of the core that is
a band-legal shadow -- cannot explain a hidden run on its own: the band test
sets every out-of-band pixel to exactly 1.0, so a core darkened *below*
`shadow_k_lo` reads identically to one that was never touched. Hence `r`, the
median unconditional ``frame/plate`` the core actually sits at, and `dev`, its
channel spread:

  * SHADOW : ``lit >= shadow_core_frac``. A shadow the gate should have caught.
    Appearing together with ``shadowed_by_gate`` is the hard failure.
  * LIGHT  : ``r >= 0.97``. The core is at or above plate level, so nothing is
    in front of it and it is not in shadow -- yet it was erased. Measured cause,
    and it is not a lighting change: ``PlateConfig.brightness_percentile`` is 60,
    and the arm is dark and takes up to 32% of the frame, so once the arm covers
    enough of the image the 60th percentile lands *inside the arm* rather than on
    the table. ``gains[t]`` therefore tracks the arm's AREA, not the light --
    ``corr(table_frac, gains)`` is -0.80 (ep0) and -0.92 (ep3), and on ep1 at
    t=568..570 the table's raw ``frame/plate`` never leaves 0.96..1.006 while
    ``gains`` falls 195 -> 162. ``foreground`` then compares an unchanged table
    against a correction that inflates it by up to 17%, over ``fg_threshold`` 18,
    so the static table reads as an occluder and every distractor on it is
    erased. This is the interesting class: an exposure artefact wearing an
    occlusion's clothes.
  * DARK   : darkened channel-neutrally but outside the band. A very deep cast
    shadow (should have been shaded) and a dark achromatic occluder (correctly
    erased) are the *same pixels*, so this is reported, never auto-resolved.
  * ARM    : neither -- non-neutral content in front of the plate.

It uses the verifier's INDEPENDENT rebuild (`_independent_illum`,
`_shadowed_union`) rather than the compositor's own `illumination`, so the
verdict is not the compositor agreeing with itself.

`--gate-override` exists to prove this tool can fail, which is the only thing
that makes a clean result worth anything. Passing `--gate-override 2.0` rebuilds
with the shadow gate disabled -- restoring the old erase-on-shadow behaviour --
while the CLASSIFIER keeps the real threshold. On the same six smoke variants:

    gate ON  (real spec)   ->  4 runs   0 SHADOW           max  3 frames
    gate OFF (2.0)         -> 40 runs  36 SHADOW           max 42 frames

The `36 SHADOW / 42 frames` reproduce the pre-fix census recorded in
`NEXT_STEPS.md` section 5 (measured on the rendered dataset, by different code),
so the classifier is detecting the real defect and not simply never firing.
Every SHADOW run reports `shadow_frac` 0.995-1.000, in the band measured for
real shadows. Note also that all 36 sit at a boundary: the shadow sweeps the
table as the arm enters and leaves, which is why the flicker was every
distractor at once, twice per episode.

The remaining four runs in each case are the same four, and the gate does not
change them: they are LIGHT/DARK, i.e. the exposure artefact described above,
not shadows. That is why the gate cannot be the whole fix.

Usage:
    HF_HUB_OFFLINE=1 python scripts/audit_occlusion.py --report-dir outputs/distract
    ... --out outputs/distract/audit.json
    ... --gate-override 2.0          # self-test: must light up with SHADOW runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

WORKPLACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(WORKPLACE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, WORKPLACE)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from background_generalization.scripts.verify_distractor_dataset import (  # noqa: E402
    _independent_illum,
    _independent_ratio,
    _shadowed_union,
    load_assets,
    rebuild_compositor,
)
from background_generalization.transforms.distractor_layout import LayoutConfig  # noqa: E402
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402
from object_generalization.scripts.verify_generated_dataset import _episode_from_index  # noqa: E402

BG = os.path.join(WORKPLACE, "background_generalization")
UP = "observation.images.up"


def runs_of(hidden: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``(start, length)`` spans where ``hidden`` is True."""
    out, start = [], None
    for t, h in enumerate(hidden):
        if h and start is None:
            start = t
        elif not h and start is not None:
            out.append((start, t - start))
            start = None
    if start is not None:
        out.append((start, int(hidden.sum()) - start))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report-dir", default=os.path.join(BG, "outputs", "distract"))
    ap.add_argument("--plate-root", default=None, help="default: spec.json's plate_root")
    ap.add_argument("--hybrid-cache",
                    default=os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid"))
    ap.add_argument("--episodes", nargs="+", default=None, help="default: all in layouts.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boundary", type=int, default=30,
                    help="frames from either end still reported as 'boundary'; report-only, "
                         "never a filter -- that mistake once hid the one run it existed to catch")
    ap.add_argument("--gate-override", type=float, default=None,
                    help="self-test: rebuild with this shadow_core_frac instead of the spec's. "
                         "Pass 2.0 to disable the gate, which forces the old erase-on-shadow "
                         "behaviour back. The CLASSIFIER keeps the spec's real threshold, so a "
                         "working audit then lights up with SHADOW runs -- and a broken one "
                         "stays silent. Use it to prove this tool can fail.")
    args = ap.parse_args()

    spec = json.load(open(os.path.join(args.report_dir, "spec.json")))
    recs = [json.loads(l) for l in open(os.path.join(args.report_dir, "layouts.jsonl"))]
    if args.episodes:
        want = {int(e) for e in args.episodes}
        recs = [r for r in recs if r["gen_episode"] in want]

    plate_root = args.plate_root or spec["plate_root"]
    if not os.path.isabs(plate_root):
        plate_root = os.path.join(REPO, plate_root)
    base = LeRobotDataset(spec["repo_id"].replace("_distract", ""), root=spec["base_root"])
    base_from = _episode_from_index(base)
    # Two configs on purpose. `cls_frac` is what the classifier calls a shadow,
    # and it must stay at the spec's real value even when --gate-override is in
    # play -- otherwise disabling the gate would also disable the detector and
    # the self-test would report the same silence as a broken tool.
    layout_cfg = LayoutConfig(**spec["layout_config"])
    cls_frac = layout_cfg.shadow_core_frac
    gate_cfg, gate_spec = layout_cfg, spec
    if args.gate_override is not None:
        gate_spec = {**spec, "layout_config": {**spec["layout_config"],
                                               "shadow_core_frac": args.gate_override}}
        gate_cfg = LayoutConfig(**gate_spec["layout_config"])

    print(f"spec: place_freq_max={spec['plate_config']['place_freq_max']} "
          f"shadow_core_frac={cls_frac} "
          f"k_band=({layout_cfg.shadow_k_lo}, {layout_cfg.shadow_k_hi}) "
          f"dev_max={layout_cfg.shadow_dev_max}")
    if args.gate_override is not None:
        print(f"  GATE OVERRIDE in effect: rebuilding with shadow_core_frac="
              f"{args.gate_override} (labeling still at {cls_frac})")

    runs, no_cube = [], 0
    for rec in recs:
        g, ep, k, n = rec["gen_episode"], rec["base_episode"], rec["variant"], rec["frames"]
        assets = load_assets(plate_root, ep)
        cp = os.path.join(args.hybrid_cache, f"episode_{ep}", "up_mask.npy")
        cube = None
        if os.path.isfile(cp):
            arr = np.load(cp, mmap_mode="r")
            cube = arr if tuple(arr.shape) == (n, 480, 640) else None
        else:
            no_cube += 1
        layout, comp, _ = rebuild_compositor(assets, gate_spec, ep, k, cube)
        bi = base_from[ep]

        frames = [_to_hwc_uint8(base[bi + t][UP]) for t in range(n)]
        cores = [comp.object_cores(t) for t in range(n)]
        alphas = [comp.alpha(frames[t], t) for t in range(n)]

        for oi in range(len(layout.placements)):
            # Same rule the metric uses: the fraction of the CORE left without
            # alpha. >0.5 is the generator's "this object is hidden" threshold.
            hid = np.array([float((alphas[t][cores[t][oi]] < 0.5).mean()) > 0.5
                            for t in range(n)])
            if not hid.any():
                continue
            for s, L in runs_of(hid):
                core = cores[s][oi]
                if not core.any():
                    continue
                # Only at run starts: this is the expensive independent rebuild
                # (an ECC warp plus a channel-wise ratio) and runs are rare. The
                # whole-frame version cost ~100x more for the same verdict.
                kk, sh = _independent_illum(frames[s], assets, s, layout_cfg)
                lit = float(sh[core].mean())
                # The gate's own criterion, recomputed: was this object shaded at
                # the moment it was hidden?
                gated = bool((_shadowed_union(frames[s], assets, s, layout, gate_cfg)
                              & cores[s][oi]).any())
                atb = s <= args.boundary or (s + L) >= (n - args.boundary)
                # Sub-classify on the UNCONDITIONAL ratio, not on the band.
                # `lit` alone cannot explain a hidden run: `_independent_illum`
                # forces every out-of-band pixel to exactly 1.0, so a core
                # darkened below `shadow_k_lo` looks identical to one that was
                # never touched. What separates the cases is `r`, the median
                # ratio the core actually sits at, and `dev`, its channel spread.
                ratio, spread = _independent_ratio(frames[s], assets, s)
                r_med = float(np.median(ratio[core]))
                dev_med = float(np.median(spread[core]))
                if lit >= cls_frac:
                    verdict = "SHADOW"
                elif r_med >= 0.97:
                    # Not darker than the plate: nothing in front, no shadow. The
                    # erasure therefore came from the illumination residual -- see
                    # the LIGHT case in the module docstring. Kept distinct from
                    # ARM because the fix is completely different.
                    verdict = "LIGHT"
                elif dev_med <= layout_cfg.shadow_dev_max:
                    # Darkened channel-neutrally but outside the band (deeper than
                    # shadow_k_lo, or spreadier than shadow_dev_max). Ambiguous by
                    # construction: a very deep cast shadow, which SHOULD have been
                    # shaded, and a dark achromatic occluder, which should indeed be
                    # erased, are the same pixels. Reported, never auto-resolved --
                    # deciding needs a look at the frames, not a threshold.
                    verdict = "DARK"
                else:
                    verdict = "ARM"
                runs.append(dict(
                    gen=g, base_episode=ep, variant=k, obj=oi, start=s, length=L,
                    shadow_frac=round(lit, 4), k_med=round(float(np.median(kk[core])), 4),
                    ratio_med=round(r_med, 4), spread_med=round(dev_med, 4),
                    shadowed_by_gate=gated, verdict=verdict, boundary=bool(atb)))
                print(f"  {'FAIL ' if (verdict == 'SHADOW' and gated) else '     '}"
                      f"gen{g:>3} ep{ep:>2} obj{oi} t={s:>4} len={L:>3}  "
                      f"shadow={lit * 100:5.1f}%  ratio={r_med:.3f} dev={dev_med:.3f}  "
                      f"gate={'on ' if gated else 'off'}  -> {verdict:<7} "
                      f"{'[boundary]' if atb else '[MID-SEQ]'}")

    n = len(runs)
    by = {v: [r for r in runs if r["verdict"] == v]
          for v in ("SHADOW", "LIGHT", "DARK", "ARM")}
    print("\n" + "=" * 78)
    for v, sel in by.items():
        mid = [r for r in sel if not r["boundary"]]
        print(f"  {v:<7} {len(sel):>3} runs   mid-sequence: {len(mid)}")
    print(f"  total   {n} runs"
          + (f"   ({no_cube} episode(s) audited without a cube mask)" if no_cube else ""))
    if runs:
        print(f"  longest run {max(r['length'] for r in runs)} frames, "
              f"max hidden fraction {max(r['length'] / recs[0]['frames'] for r in runs):.3%} of an episode")

    # The hard failure: the shadow gate exists precisely so that a shadow does
    # not erase a distractor. A run the gate marked as shaded, yet still hidden,
    # means the gate did not fire -- and this is the assertion that can fail.
    deep = [r for r in runs if r["verdict"] == "DARK"]
    if deep:
        print(f"\n  NOTE {len(deep)} run(s) were darkened channel-neutrally but outside the band:")
        for r in deep[:10]:
            print(f"    gen{r['gen']} obj{r['obj']} t={r['start']} len={r['length']} "
                  f"ratio={r['ratio_med']} dev={r['spread_med']}")
        print(f"  A shadow deeper than shadow_k_lo={layout_cfg.shadow_k_lo} and a dark achromatic "
              f"occluder are the same pixels here, so this is not a verdict. If the frames show a "
              f"shadow, the object should have been shaded and the band is too narrow.")

    violations = [r for r in runs if r["verdict"] == "SHADOW" and r["shadowed_by_gate"]]
    if violations:
        print(f"\n  FAIL {len(violations)} run(s) are shadow-explained AND were marked shaded by "
              f"the gate, yet the object was still hidden:")
        for r in violations[:10]:
            print(f"    gen{r['gen']} obj{r['obj']} t={r['start']} len={r['length']}")
        print("  The gate fired but the alpha did not follow. Check `alpha()`'s erase mask "
              "and the shadow dilation against the composite.")
    unfired = [r for r in runs if r["verdict"] == "SHADOW" and not r["shadowed_by_gate"]]
    if unfired:
        print(f"\n  NOTE {len(unfired)} run(s) are shadow-explained but the gate did NOT fire "
              f"(shadow_frac between {cls_frac} and 1.0, or k out of band).")
        print("  These are the residual erasures; inspect before calling the gate sound.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(runs=runs, summary={v: len(s) for v, s in by.items()},
                           violations=violations, dark_ambiguous=deep), f, indent=1)
        print(f"\n-> {args.out}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
