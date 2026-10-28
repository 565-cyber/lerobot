"""Where distractors go, and the per-frame mask that draws them.

Three pieces, all pure (no LeRobot, no dataset):

1. :func:`sample_layout` picks sites from the plate's ``placement`` map. The
   map is the ONLY authority on where an object may rest -- it already excludes
   the arm's swept area, the cube's trajectory and the red container.
2. :func:`render_background_layer` bakes the objects (and their contact
   shadows) into a copy of the plate, producing the single image that every
   frame of the episode composites against. Because the objects do not move
   during an episode, this is computed once per episode, not per frame.
3. :func:`alpha_for_frame` produces that frame's blend weight: the objects'
   footprint, minus wherever the real arm/cube currently is.

Minimal-change compositing
-------------------------
    out = (1 - a_t) * frame + a_t * background_layer * L_t
    a_t = paint_union * (1 - occluder_t)
    occluder_t = fg_t AND NOT (shadow_t AND shaded_object)

so every pixel outside the distractors -- silhouettes AND their shadows, which
is what ``paint_union`` covers -- is returned BIT-IDENTICAL to the input frame.
That is the hard, checkable invariant of this pipeline, and the dual of Stage
2A's "unchanged outside the mask". ``L_t`` is 1 outside the arm's shadow, so it
does not weaken that invariant.

Shade, don't erase
------------------
``fg_t`` is ``|frame - plate| > fg_threshold``, which over-covers on purpose: it
includes the arm's *cast shadow* on the table, because the safe direction is to
hide a distractor rather than risk painting one onto the arm. Erasing on that
over-cover is wrong, though, in a way that is very visible: the arm's shadow
sweeps the whole resting area at once as the arm enters and leaves, so EVERY
distractor blinks out together -- measured, 36 of 40 occlusion runs in the smoke
set, at t<=42 and t>=542, up to 1.4 s.

A shadow is not an occluder. It is the plate under less light, so the distractor
under it is still there and should be *darkened*, not removed. The two are
separable per pixel with no extra calibration, because a shadow is multiplicative:
``frame_c / plate_c`` takes the same value on R, G and B. Something else in front
of the plate does not.

So a pixel whose ratio is channel-consistent and below 1 is shaded; everything
else ``fg`` marks is treated as an occluder and erased, exactly as before. The
per-pixel test is then gated per object: an object is shaded only when >=
``shadow_core_frac`` of its core passes, which is what rejects a black arm link
lying across a distractor (its ratio is far darker than any shadow) and a partly
covered one.

The residual risk is a uniformly mid-gray occluder covering >= 90% of an object's
core while matching the plate's channel ratios, which this test cannot rule out;
``shadow_k_lo`` is set well above the darkest occluder the arm can present
(measured shadows span 0.73-0.95, so the band is narrow) rather than merely below
the measured shadows. The placement map is the primary defence regardless: every
site is real background in >= 95% of frames, so the arm's BODY is rarely there --
it is specifically its shadow that the map cannot exclude.

Two invariants are enforced here rather than merely intended:

* An object's whole footprint must lie inside ``placement``. Pasting by the
  anchor alone does not give this -- a 50 px object anchored at the centre of a
  75 px strip can hang over the edge, onto the arm's path or the container.
* The contact shadow is clipped to ``placement`` too. The shadow reaches ~0.9 x
  the object's size beyond its footprint, which would otherwise be free to
  darken the red container -- a task-relevant object this pipeline must never
  touch. The consequence is the strong invariant: **the background layer
  differs from the plate only inside ``placement``.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..distractors.procedural import Cutout, CutoutProvider


@dataclass
class LayoutConfig:
    n_per_episode: tuple = (2, 4)
    """How many distractors to *attempt* per episode, inclusive.

    Self-limiting: the inner loop gives up after ``max_site_attempts`` and moves
    on, so an episode with no room simply gets fewer and the run never fails for
    want of space. That is why this can be set by what the scene should look
    like rather than by the tightest episode.

    It was ``(1, 2)``, from the 5.5-8% free area measured on episodes 0 and 1 --
    which turn out to be the two tightest of all 50. Over the full set the
    median is 16.8% (max 28.5%), and ``(1, 2)`` placed only 1.64 objects on
    average, leaving most of the legal area unused. ``(2, 4)`` places 3.12,
    including 2.3 on episode 0 and 3.3 on episode 1. More clutter is both closer
    to RoboAgent's L2 condition and closer to a real desk with objects pushed to
    the edges; every placement is still verified individually."""

    min_blob_area: int = 3000
    """A connected placement region smaller than this cannot host an object."""

    max_site_attempts: int = 80
    """Give up and place fewer objects rather than relax a safety check."""

    min_gap_px: int = 8
    """Minimum clearance between two distractors' footprints."""

    require_object_in_frame: bool = True
    """Reject a site whose object would be sliced by the frame border.

    The footprint-inside-``placement`` check does NOT catch this: the legal strip
    along the right edge runs to the last column, so an object half off-frame
    still tests clean -- ``_paste_max`` clips the canvas, and the surviving half
    is all legal. Measured on episodes 0/1 x 3 variants: 4 of 11 objects were
    clipped, worst case 1234 px surviving of 2224.

    Worth rejecting rather than tolerating. The cut is a hard straight edge
    exactly at the image border with nothing occluding the object, which no real
    scene produces and a policy can latch onto. The object's SHADOW is allowed
    off-frame; only the silhouette is constrained.
    """

    fg_threshold: float = 18.0
    """Gray distance from the plate above which a pixel counts as foreground.

    MUST stay <= ``PlateConfig.diff_threshold``, the threshold ``freq`` -- and
    therefore ``placement`` -- was built with. Raising it above that would flag
    FEWER pixels as foreground than the placement map assumed, and a distractor
    could then be drawn over the arm. Lowering it is safe (it only over-covers).
    ``FrameCompositor`` enforces this against the real plate config."""

    min_fg_area: int = 20
    """Drop residual components smaller than this before growing them.

    The residual is dominated by speckle: 272 components per frame, median area
    2 px, from table texture under the few px of remaining misalignment. They
    are only ~1% of the fg pixels (the arm is a single component of thousands),
    so removing them costs essentially no protection, but without this the
    objects pick up a shimmering dust of one-pixel occlusions."""

    fg_dilate_px: int = 4
    """Grow the residual foreground mask, AFTER the area filter. Over-covering
    is the safe direction: it can only hide a distractor, never paint over the
    arm. Measured effect on the occlusion budget on episode 0 (1000 px site):
    0 px -> 3.8% of frames mostly occluded, 3 px -> 4.7%, 6 px -> 5.4%. This is
    the knob that spends the <=5% budget."""

    shadow_strength: float = 0.43
    """Plate multiplier in a shadow core is ``1 - shadow_strength`` = 0.57
    (measured on the blue cube). Must match the provider's value."""

    constrain_shadow_to_placement: bool = True
    """Keep the shadow from darkening anything outside ``placement``."""

    shadow_k_lo: float = 0.55
    """Darkest ``frame/plate`` ratio still read as a *cast shadow*.

    Measured shadows on these episodes span 0.73-0.95, so 0.55 keeps every one
    of them with margin. Set well ABOVE the darkest thing the arm can present
    rather than merely below the measured shadows, because the cost of the two
    mistakes is not symmetric: too high and a real shadow is erased (the
    distractor blinks, the defect this exists to fix), too low and a mid-gray
    arm part is shaded instead of erased, i.e. a distractor drawn ON the arm.
    See the "Shade, don't erase" note in the module docstring."""

    shadow_k_hi: float = 1.00
    """Brightest ``frame/plate`` ratio still read as a cast shadow. A shadow only
    removes light, so a pixel brighter than the plate is not one."""

    shadow_dev_max: float = 0.08
    """Channel spread of ``frame/plate`` still read as multiplicative.

    A shadow scales R, G and B by the same factor, so the three ratios agree.
    Measured: 0.007-0.040 on the shadow runs, i.e. set by sensor noise."""

    shadow_core_frac: float = 0.90
    """Fraction of an object's CORE that must be shadow-explained before the
    object is shaded rather than erased and reshaded at all.

    Per object, not per pixel, and deliberately so: an arm partly across a
    distractor still leaves a majority of the core shadow-like, which a
    per-pixel rule would keep, painting the object over the arm. Measured on the
    smoke set: 36 runs score 99.5-100% here, the 3 real arm runs 0%, and the one
    genuinely mixed run (a black link across ~34% of the core) 65.7% -- so the
    threshold sits in an empty band, not at a tuned boundary."""


@dataclass(frozen=True)
class PlacedDistractor:
    cutout: Cutout
    top: tuple[int, int]  # (y, x) of the cutout canvas in frame coords


@dataclass
class EpisodeLayout:
    placements: list = field(default_factory=list)
    rejected: int = 0
    """Sites tried and refused because the footprint would not fit. Reported so
    a layout that silently degrades to zero objects is visible."""

    def __len__(self) -> int:
        return len(self.placements)

    def alpha_union(self, shape_hw: tuple[int, int]) -> np.ndarray:
        """Union of every object's silhouette, in frame coordinates."""
        out = np.zeros(shape_hw, np.float32)
        for p in self.placements:
            _paste_max(out, p.cutout.alpha, p.top)
        return out

    def shadow_union(self, shape_hw: tuple[int, int]) -> np.ndarray:
        """Union of every object's contact shadow, in frame coordinates."""
        out = np.zeros(shape_hw, np.float32)
        for p in self.placements:
            _paste_max(out, p.cutout.shadow, p.top)
        return out

    def paint_union(self, shape_hw: tuple[int, int]) -> np.ndarray:
        """Silhouettes AND shadows: what the compositor must be handed.

        Not ``alpha_union``. ``render_background_layer`` bakes the shadows into
        the background layer, so a compositor restricted to the silhouettes
        blends that layer only under the objects and drops the shadow entirely:
        each object reads as a flat sticker with a dark rim, and nothing in a
        scene where the cube, the container and the arm all cast one. Measured
        on the smoke set: the layer changed 4482 px (1884 object + 2608 shadow)
        and only the 1884 object pixels reached the frame.

        ``alpha_union`` keeps its silhouette-only meaning -- the occlusion
        metric and the footprint assertions talk about the object, not its
        shadow. ``FrameCompositor`` rejects an alpha that does not cover this.
        """
        return np.maximum(self.alpha_union(shape_hw), self.shadow_union(shape_hw))

    def footprint_union(self, shape_hw: tuple[int, int]) -> np.ndarray:
        """Hard (unfeathered) coverage, used for the safety assertions."""
        out = np.zeros(shape_hw, bool)
        for p in self.placements:
            _paste_max(out, (p.cutout.alpha > 0).astype(np.float32), p.top)
        return out > 0

    def to_record(self) -> dict:
        return {
            "n": len(self.placements),
            "rejected": self.rejected,
            "objects": [
                {"top": [int(p.top[0]), int(p.top[1])],
                 "canvas": [int(p.cutout.size[0]), int(p.cutout.size[1])],
                 "anchor": [round(float(p.cutout.anchor[0]), 2),
                            round(float(p.cutout.anchor[1]), 2)]}
                for p in self.placements
            ],
        }


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Zero out connected components smaller than ``min_area`` pixels."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n_lbl <= 1:
        return m
    keep = np.zeros(n_lbl, bool)
    keep[1:] = stats[1:, 4] >= min_area
    return keep[lbl].astype(np.uint8)


def _alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """Half-open ``[y0, y1) x [x0, x1)`` box of the non-zero alpha."""
    ys, xs = np.nonzero(np.asarray(alpha) > 0)
    if ys.size == 0:
        return 0, 0, 0, 0
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _paste_max(dst: np.ndarray, src: np.ndarray, top: tuple[int, int]) -> None:
    """dst = max(dst, src) with ``src``'s top-left at ``top``, clipped to dst."""
    h, w = dst.shape[:2]
    sh, sw = src.shape[:2]
    y0, x0 = int(top[0]), int(top[1])
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(h, y0 + sh), min(w, x0 + sw)
    if ys1 <= ys0 or xs1 <= xs0:
        return
    dst[ys0:ys1, xs0:xs1] = np.maximum(
        dst[ys0:ys1, xs0:xs1], src[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    )


def sample_layout(
    placement: np.ndarray,
    provider: CutoutProvider,
    cfg: LayoutConfig,
    rng: np.random.Generator,
) -> EpisodeLayout:
    """Choose distractor sites from ``placement``; never relax a check."""
    placement_b = np.asarray(placement).squeeze() > 0
    h, w = placement_b.shape
    layout = EpisodeLayout()

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(placement_b.astype(np.uint8), 8)
    blobs = [j for j in range(1, n_lbl) if stats[j, 4] >= cfg.min_blob_area]
    if not blobs:
        return layout

    weights = np.array([stats[j, 4] for j in blobs], np.float64)
    weights /= weights.sum()

    want = int(rng.integers(cfg.n_per_episode[0], cfg.n_per_episode[1] + 1))
    taken = np.zeros((h, w), bool)

    for _ in range(want):
        placed = False
        for _ in range(cfg.max_site_attempts):
            j = blobs[int(rng.choice(len(blobs), p=weights))]
            ys, xs = np.nonzero(lbl == j)
            k = int(rng.integers(len(ys)))
            site_y, site_x = int(ys[k]), int(xs[k])

            cut = provider.sample(rng)
            ay, ax = cut.anchor
            top = (int(round(site_y - ay)), int(round(site_x - ax)))

            # (a0) the silhouette must fit in the image, shadow excepted
            if cfg.require_object_in_frame:
                oy0, ox0, oy1, ox1 = _alpha_bbox(cut.alpha)
                if (top[0] + oy0 < 0 or top[1] + ox0 < 0
                        or top[0] + oy1 > h or top[1] + ox1 > w):
                    layout.rejected += 1
                    continue

            foot = np.zeros((h, w), bool)
            _paste_max(foot, (cut.alpha > 0).astype(np.float32), top)
            grow = cv2.dilate(foot.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0

            # (a) the object must fit entirely inside the legal region
            if np.any(foot & ~placement_b):
                layout.rejected += 1
                continue
            # (b) and keep clear of objects already placed
            if np.any(grow & taken):
                layout.rejected += 1
                continue

            layout.placements.append(PlacedDistractor(cut, top))
            taken |= foot
            placed = True
            break
        if not placed:
            break  # fewer objects rather than a looser constraint

    return layout


def render_background_layer(
    plate_rgb: np.ndarray,
    layout: EpisodeLayout,
    cfg: LayoutConfig,
    placement: np.ndarray | None = None,
) -> np.ndarray:
    """Bake the objects and their shadows into a copy of the plate.

    Returns uint8 RGB, same shape/order as ``plate_rgb``. Only pixels inside
    ``placement`` may differ from the plate (asserted by the caller).
    """
    h, w = plate_rgb.shape[:2]
    layer = plate_rgb.astype(np.float32)

    for p in layout.placements:
        shadow = np.zeros((h, w), np.float32)
        _paste_max(shadow, p.cutout.shadow, p.top)
        if cfg.constrain_shadow_to_placement and placement is not None:
            shadow = shadow * (np.asarray(placement).squeeze() > 0)
        layer *= (1.0 - cfg.shadow_strength * shadow)[..., None]

        alpha = np.zeros((h, w), np.float32)
        rgb = np.zeros((h, w, 3), np.float32)
        _paste_max(alpha, p.cutout.alpha, p.top)
        _paste_max(rgb, p.cutout.rgb.astype(np.float32), p.top)
        layer = layer * (1.0 - alpha)[..., None] + rgb * alpha[..., None]

    return np.clip(layer, 0, 255).astype(np.uint8)


def to_frame0(arr: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Resample ``arr`` (frame-t coords) into frame-0 coords: dst(p)=src(M.p)."""
    h, w = arr.shape[:2]
    return cv2.warpAffine(arr, np.asarray(warp, np.float32), (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def to_frame_t(arr0: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Resample ``arr0`` (frame-0 coords) into frame t: dst(p)=src(M^-1.p)."""
    h, w = arr0.shape[:2]
    return cv2.warpAffine(arr0, np.asarray(warp, np.float32), (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class FrameCompositor:
    """Augment one episode's UP frames against a fixed distractor layout.

    All three coordinate conventions live here so they cannot drift apart:

    * the plate, the background layer and the objects' alpha are in **frame-0**
      coordinates (that is what ``build_plate`` produced);
    * each input frame is in **frame-t** coordinates;
    * ``warp[t]`` maps frame-0 -> frame-t, and the frame is never resampled --
      only the auxiliary arrays are, which is what keeps every pixel outside
      the objects bit-identical to the input.

    Alignment is not optional here. The residual test below is exactly the one
    ``build_plate`` used to build ``freq`` (and therefore ``placement``): frame
    resampled into frame-0 coords, minuend the un-resampled plate. Skipping it
    and differencing the raw frame instead makes the whole right-hand edge of
    the frame read as "foreground" in 100% of frames, and every distractor
    pasted there is invisible.
    """

    def __init__(self, plate_rgb, bg_layer, alpha_union, warps, gains,
                 cube_masks=None, cfg=None, plate_diff_threshold=None, layout=None):
        self.plate = plate_rgb
        self.bg_layer = bg_layer
        self.alpha_union = alpha_union
        self.warps = warps
        self.gains = gains
        self.cube = cube_masks
        self.cfg = cfg or LayoutConfig()
        self.layout = layout
        if plate_diff_threshold is not None and self.cfg.fg_threshold > float(plate_diff_threshold):
            raise ValueError(
                f"fg_threshold={self.cfg.fg_threshold} exceeds the plate's diff_threshold="
                f"{plate_diff_threshold}, which built `placement`. The compositor would then treat "
                f"pixels as background that the placement map considered foreground, and could "
                f"paint a distractor over the arm."
            )
        # The alpha must cover the shadows, not just the silhouettes. Passing
        # `layout.alpha_union` here silences every cast shadow in the dataset --
        # a silent regression, since the frames still look plausible one at a
        # time. Checked rather than documented: the cost of the wrong argument
        # is invisible in the output and invisible in the tests that existed.
        if layout is not None and not np.array_equal(
                np.asarray(alpha_union) > 0,
                np.maximum(layout.alpha_union(self.shape_hw),
                           layout.shadow_union(self.shape_hw)) > 0):
            raise ValueError(
                "alpha_union does not match layout.paint_union(): the compositor would drop the "
                "contact shadows that the background layer has already baked in. Pass "
                "Layout.paint_union(...) unless you have a reason not to."
            )
        self.plate_gray = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        self._bg_cache: dict[int, np.ndarray] = {}
        self._a_cache: dict[int, np.ndarray] = {}
        self._ill_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._shaded_cache: dict[int, np.ndarray | None] = {}
        self._core_cache: dict[int, list[np.ndarray]] = {}

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self.plate.shape[:2]

    def background(self, t: int) -> np.ndarray:
        if t not in self._bg_cache:
            self._bg_cache[t] = to_frame_t(self.bg_layer, self.warps[t])
        return self._bg_cache[t]

    def alpha_base(self, t: int) -> np.ndarray:
        if t not in self._a_cache:
            self._a_cache[t] = to_frame_t(self.alpha_union, self.warps[t])
        return self._a_cache[t]

    def foreground(self, frame_rgb: np.ndarray, t: int) -> np.ndarray:
        """Residual-vs-plate mask in frame-t coords (over-covering by design)."""
        f = frame_rgb.astype(np.float32)
        g = float(self.gains[t]) if self.gains is not None else 1.0
        if self.gains is not None:
            # build_plate normalizes frame t to frame-0 level via gains[0]/gains[t]
            f = f * (float(self.gains[0]) / max(g, 1.0))
        gray = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        resid0 = (np.abs(to_frame0(gray, self.warps[t]) - self.plate_gray) > self.cfg.fg_threshold)
        fg = to_frame_t(resid0.astype(np.uint8), self.warps[t])
        if self.cube is not None:
            fg = np.maximum(fg, (np.asarray(self.cube[t]).squeeze() > 0).astype(np.uint8))
        if self.cfg.min_fg_area > 0:
            fg = _drop_small(fg, self.cfg.min_fg_area)
        if self.cfg.fg_dilate_px > 0:
            r = int(self.cfg.fg_dilate_px)
            fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return fg

    def illumination(self, frame_rgb: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        """How much light fell on each pixel this frame, and whether that is all.

        Returns ``(k, shadow)`` in frame-t coords:

        * ``k``      : ``frame/plate``, averaged over R, G and B; 1.0 wherever
          the pixel is not a cast shadow (so multiplying by it is a no-op).
        * ``shadow`` : the pixel's ratio is the SAME on all three channels and
          below 1 -- the plate under less light rather than something in front
          of it.

        Computed in frame-0 coords, like every other plate comparison here, then
        warped. The ratio is only meaningful where ``plate > 0``; dividing by
        ``max(plate, 1)`` makes a near-black plate pixel read as a huge ratio,
        which fails ``k_hi`` and is then treated as an occluder -- the safe
        direction.
        """
        if t in self._ill_cache:
            return self._ill_cache[t]
        f = frame_rgb.astype(np.float32)
        if self.gains is not None:
            f = f * (float(self.gains[0]) / max(float(self.gains[t]), 1.0))
        f0 = to_frame0(np.clip(f, 0, 255), self.warps[t])
        q = f0 / np.maximum(self.plate.astype(np.float32), 1.0)      # (H, W, 3)
        k = q.mean(axis=2)
        dev = q.max(axis=2) - q.min(axis=2)
        ok = ((dev <= self.cfg.shadow_dev_max) & (k > self.cfg.shadow_k_lo)
              & (k < self.cfg.shadow_k_hi))
        k0 = np.where(ok, k, 1.0).astype(np.float32)
        k_t = to_frame_t(k0, self.warps[t]).astype(np.float32)
        # Warping a 0/1 mask through INTER_LINEAR leaves partial values along the
        # boundary; >0.5 makes a pixel have to be *mostly* shadow, so the ring
        # where the warp is uncertain counts as an occluder. Over-covering again.
        sh_t = to_frame_t(ok.astype(np.uint8), self.warps[t]) > 0.5
        self._ill_cache[t] = (k_t, sh_t)
        return k_t, sh_t

    def object_cores(self, t: int) -> list[np.ndarray]:
        """Per object, its CORE in frame-t coords.

        The hard footprint (``alpha > 0``), further restricted to where the base
        alpha is already ``>= 0.5``. The feathered rim is excluded on purpose:
        ``alpha_base < 0.5`` holds there by construction even with nothing
        occluding it, so including it would blend every per-object fraction with
        a constant ~22%.
        """
        if t in self._core_cache:
            return self._core_cache[t]
        base = self.alpha_base(t)
        cores = []
        for p in self.layout.placements:
            m = np.zeros(self.shape_hw, np.float32)
            _paste_max(m, (p.cutout.alpha > 0).astype(np.float32), p.top)
            cores.append((to_frame_t(m, self.warps[t]) > 0.5) & (base >= 0.5))
        self._core_cache[t] = cores
        return cores

    def shaded_objects(self, frame_rgb: np.ndarray, t: int) -> np.ndarray | None:
        """Union of the objects that are under a cast shadow this frame.

        ``None`` when the compositor was built without a layout, which is the
        only case where the per-object gate cannot be evaluated; the caller then
        falls back to erasing, i.e. the pre-existing behaviour.

        The gate is per OBJECT, not per pixel, and that is the whole safety
        argument: an arm lying across a distractor is still a shadow-like
        majority per pixel, so a per-pixel rule would happily keep the object
        and paint it onto the arm. Requiring ``shadow_core_frac`` of the core
        rejects that (measured: the one genuinely mixed run scores 65.7%).
        """
        if t in self._shaded_cache:
            return self._shaded_cache[t]
        if self.layout is None:
            self._shaded_cache[t] = None
            return None
        k, sh = self.illumination(frame_rgb, t)
        out = np.zeros(self.shape_hw, bool)
        for core in self.object_cores(t):
            if not core.any():
                continue
            lit = sh[core]
            if float(lit.mean()) < self.cfg.shadow_core_frac:
                continue
            # A shadow only removes light, so its median ratio must be below 1.
            # Redundant against k_hi per pixel, but it costs nothing and it is
            # the statement the physics actually makes.
            if float(np.median(k[core][lit])) >= 1.0:
                continue
            out |= core
        self._shaded_cache[t] = out
        return out

    def alpha(self, frame_rgb: np.ndarray, t: int) -> np.ndarray:
        """Blend weight for this frame: objects, minus wherever the arm sits.

        Not "minus everywhere ``foreground`` is set": ``foreground`` over-covers
        by design and includes the arm's cast shadow, which must shade a
        distractor rather than erase it. See the module docstring.
        """
        erase = self.foreground(frame_rgb, t) > 0
        shaded = self.shaded_objects(frame_rgb, t)
        if shaded is not None:
            _, sh = self.illumination(frame_rgb, t)
            # Grow the shadow by the same amount `foreground` was grown, or the
            # over-cover bites a ring out of the object along the shadow's edge
            # -- a shimmering crescent, the same class of defect as the dust
            # `min_fg_area` exists to remove.
            r = int(self.cfg.fg_dilate_px)
            if r > 0:
                sh = cv2.dilate(sh.astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            erase = erase & ~(sh & shaded)
        return np.clip(self.alpha_base(t) * (1.0 - erase.astype(np.float32)), 0.0, 1.0)

    def composite(self, frame_rgb: np.ndarray, t: int, a: np.ndarray | None = None) -> np.ndarray:
        """(1-a)*frame + a*background*k. Bit-identical to ``frame_rgb`` where a==0.

        ``k`` is the frame's illumination (see :meth:`illumination`), so an
        object standing in the arm's shadow is drawn darkened rather than
        removed. It is 1.0 outside the shadow, and the term is multiplied by
        ``a``, so it cannot perturb the bit-identity outside the objects.

        ``a`` may be passed in by a caller that already needs this frame's
        :meth:`alpha` (the generator does, for the occlusion metric). It MUST be
        the alpha of this same frame and time -- passing anything else composites
        against the wrong mask. ``None`` (the default) computes it, so the
        ordinary call is unchanged.
        """
        if a is None:
            a = self.alpha(frame_rgb, t)
        k, _ = self.illumination(frame_rgb, t)
        aw = a[..., None]
        return np.clip(frame_rgb.astype(np.float32) * (1.0 - aw)
                       + self.background(t).astype(np.float32) * k[..., None] * aw,
                       0, 255).astype(np.uint8)

    def footprint_occlusion(self, frame_rgb: np.ndarray, t: int) -> list[float]:
        """Per object, the fraction of its CORE hidden in this frame.

        Two traps this avoids, both of which inflate the number:

        * "is any pixel hidden" reports every object as occluded in ~94% of
          frames, because the residual carries a dust of one-pixel components;
        * masking by ``alpha > 0`` sweeps in the feathered rim, where
          ``alpha < 0.5`` holds by construction even with nothing occluding it
          -- a steady, phantom ~22% for every object.

        So the denominator is the object's core (``alpha >= 0.5``), where
        ``composite alpha < 0.5`` means exactly one thing: residual foreground
        is covering it.

        "Covering" now means the arm, not its shadow: a shadowed object keeps
        its alpha and is darkened instead (see the module docstring), so it no
        longer counts here. This is the change that makes ``mean_hidden`` an
        occlusion budget rather than an occlusion-plus-lighting budget.
        """
        return self.occlusion_from_alpha(self.alpha(frame_rgb, t), t)

    def occlusion_from_alpha(self, a: np.ndarray, t: int) -> list[float]:
        """The metric, given an alpha already computed for this frame.

        The core definition lives in :meth:`object_cores` so the metric and the
        shading gate cannot drift apart -- they are two readings of the same
        region.
        """
        cores = self.object_cores(t)
        out = []
        for core in cores:
            if not core.any():
                out.append(0.0)
                continue
            out.append(float((a[core] < 0.5).mean()))
        return out
