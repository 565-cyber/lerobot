"""Procedurally generated distractor cutouts -- no photography required.

The provider draws a small object (box / cylinder / sphere / blob) as it would
look lying on the workspace table, seen from the UP camera: an RGB patch, a
silhouette alpha, and the contact shadow it casts.

Physical consistency is not invented here, it is measured from the dataset. The
shadow geometry defaults come from ``build_plate``'s plate and the blue cube's
silhouette on episode 0, where the cube sits alone on the table and the arm is
at home (measured over frames 0/10/20/30/40):

    shadow azimuth      275 deg (image coords, +y down) -> toward the top of the
                        frame, i.e. the light comes from the NEAR side
    shadow core         table brightness x 0.57, at the contact edge
    shadow reach        ~1.0 x the silhouette's radius beyond its edge, by which
                        point it is back to the table brightness
    shadow profile      a gradient, not a plateau: 0.55 -> 0.77 -> 0.93 -> 1.00
                        at 1.25 / 1.50 / 1.75 / 2.00 radii from the centre
    table baseline      gray 181, RGB ~ (176, 161, 171)

"shadow length ~0.9 x the object's height" used to be here and was measuring a
different thing -- the shadow's total extent rather than where its mass sits.
It made the shadow a rigid offset copy at uniform darkness, which renders as a
second grey object. See ``shadow_spread_ratio`` for what replaced it.

An earlier attempt to measure this around the red container was WRONG and is
worth not repeating: the container is ringed by the arm's diffuse shadow in the
plate, so the scan reported the darkest direction as 240 deg with a huge delta.
The cube is the only clean probe, because its shadow leaves with it (it is gone
by frame 300 and the plate removes it).

Colour sampling deliberately excludes the blue band (the interaction cube) and
the red band (the container): a distractor that looks like the target would make
the task ambiguous rather than merely harder, which is not the augmentation
being studied. The check is applied to the shaded object, in the target
detector's own HSV box -- see ``forbidden_boxes`` for why the drawn base hue is
not a sufficient guarantee.

The canvas convention: ``rgb``/``alpha``/``shadow`` share one local frame, and
``anchor`` is the pixel in that frame where the object's ground contact point
sits. Layout code pastes by aligning ``anchor`` to a chosen site, so the shadow
offset is entirely the provider's business.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@dataclass(frozen=True)
class Cutout:
    """One placeable object sprite.

    Attributes
    ----------
    rgb : (h, w, 3) uint8, RGB order (matches LeRobot's decoded frames).
    alpha : (h, w) float32 in [0, 1]; the object's silhouette (soft edge).
    shadow : (h, w) float32 in [0, 1]; per-pixel darkening to apply to the
        plate *under* the object (shadow = 1 -> multiply the plate by
        ``1 - shadow_strength``).
    anchor : (y, x) float; where the object rests, in this canvas's pixels.
    """

    rgb: np.ndarray
    alpha: np.ndarray
    shadow: np.ndarray
    anchor: tuple[float, float]

    def __post_init__(self) -> None:
        h, w = self.alpha.shape
        if self.rgb.shape != (h, w, 3):
            raise ValueError(f"rgb {self.rgb.shape} != ({h}, {w}, 3)")
        if self.shadow.shape != (h, w):
            raise ValueError(f"shadow {self.shadow.shape} != ({h}, {w})")
        if self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {self.rgb.dtype}")
        if not (0 <= self.anchor[0] < h and 0 <= self.anchor[1] < w):
            raise ValueError(f"anchor {self.anchor} outside canvas ({h}, {w})")

    @property
    def size(self) -> tuple[int, int]:
        return self.alpha.shape


@runtime_checkable
class CutoutProvider(Protocol):
    """Anything that can draw one distractor.

    The only contract: given a numpy ``Generator``, return a :class:`Cutout`
    with a valid anchor. A photographed object (segmented once, off-line) can
    satisfy this without any other change to the pipeline.
    """

    def sample(self, rng: np.random.Generator) -> Cutout: ...


@dataclass
class ProceduralConfig:
    """Every knob, including the ones measured from the dataset."""

    size_px_min: int = 24
    """Object size on the table, in pixels. The blue cube is ~36 px across, so
    this spans "smaller than the target" to "clearly bigger"."""

    size_px_max: int = 56

    shapes: tuple = ("box", "cylinder", "sphere", "blob")

    feather_px: int = 2
    """Soft edge width of the silhouette. MUST stay <= the layout's
    ``exclude_dilate_px``, else a feathered edge can bleed onto the red
    container that the placement map excluded."""

    forbidden_hues: tuple = ((95, 140), (0, 10), (170, 180))
    """OpenCV hue bands the *base* colour avoids: the blue cube (the target) and
    the red container. Sampling retries a bounded number of times, then snaps to
    the nearest legal hue.

    Deliberately not the whole safety story -- see ``forbidden_boxes``."""

    hue_retries: int = 24
    sat_range: tuple = (40, 190)
    val_range: tuple = (55, 235)

    forbidden_boxes: tuple = (
        ("blue-target", (95, 70, 50), (140, 255, 255)),
        ("red-container", (0, 90, 70), (10, 255, 255)),
        ("red-container", (170, 90, 70), (179, 255, 255)),
    )
    """(label, lower, upper) HSV boxes checked against the SHADED object.

    A base hue outside ``forbidden_hues`` is not enough: ``_shade`` multiplies
    the channels, so a highlight clips at 255 and a near-boundary hue slides
    back inside. Measured on 300 cutouts, that put 2 (~0.7%) almost entirely
    inside the blue box. The first box is therefore copied verbatim from
    ``BlueMaskConfig`` -- "would the target's own detector fire on this
    distractor" is the question that matters, so it is asked in the detector's
    own units rather than in a hue band that merely resembles them."""

    forbidden_frac_max: float = 0.02
    """At most this fraction of the object's core pixels may land in any one
    forbidden box. Not zero: a handful of anti-aliased rim pixels is invisible,
    and demanding zero would make the fallback fire on ordinary objects."""

    color_retries: int = 12
    """Shaded colours drawn before falling back to a neutral grey. The fallback
    is guaranteed safe (saturation 0 cannot satisfy boxes whose S floor is 70)
    and keeps the Lambertian shading, so it degrades to a grey object rather
    than to no object."""

    # --- lighting (measured; see module docstring) --------------------------
    light_elev_deg: float = 45.0
    shadow_azimuth_deg: float = 275.0
    shadow_len_ratio: float = 0.30
    """Mean offset of the silhouette copy, as a fraction of the canvas size.

    Not the length of the shadow, and not 0.9 -- an earlier reading of the same
    cube that measured the shadow's *extent* rather than where its mass sits.
    Fitted, not guessed: see ``shadow_spread_ratio``."""

    shadow_strength: float = 0.43
    """Plate multiplier in the shadow core is ``1 - shadow_strength`` = 0.57."""

    shadow_spread_ratio: float = 0.40
    """Spread of the offsets around ``shadow_len_ratio``, as a fraction of the
    canvas size. This is the penumbra.

    A single offset copy is a *rigid duplicate* of the object at uniform
    strength: no penumbra, and as dark at the tip as at the contact point, so it
    reads as a second grey object rather than a shadow. The real one is a
    gradient. Measured on the blue cube in episode 0, walking outward from its
    centre along the shadow azimuth, the table goes

        1.25 r -> 0.55    1.50 r -> 0.77    1.75 r -> 0.93    2.00 r -> 1.00

    (x the plate, on a cube of radius r; it is gone by ~2 r). ``(0.30, 0.40,
    blur 1.0)`` reproduces those four points to within 0.05 -- fitted by grid
    search over a disc, not by eye. The spread is doing most of the work: the
    shadow is a broad diffuse blob just past the object, which is what a 45 deg
    source and an object resting on the table actually produce.

    The four-point fit only makes sense for a disc: ``sqrt(area/pi)`` is an
    equivalent-area radius, so sampling a box or a blob at "1.25 r" lands
    somewhere different relative to its real edge and the residuals read as
    0.19 even though the model is fine. The shape-agnostic form of the same
    measurement is the distance beyond the silhouette at which the shadow falls
    below 10% of its edge value, in units of that radius: the cube gives ~1.0,
    this gives 0.98 (0.93-1.08 over 14 draws).
    """

    shadow_blur_px: float = 1.2
    shadow_steps: int = 10
    """How many offset copies of the silhouette are summed to make the shadow.
    Fewer than ~8 leaves visible banding in the penumbra."""
    shade_contrast: float = 0.35
    """How strongly the surface normal modulates the base colour. Kept modest:
    a top-down view mostly shows the object's top face."""

    ambient_rim: float = 0.22
    """Darkening at the silhouette edge (cheap contact occlusion)."""

    edge_soft_px: float = 3.0
    """Edge rounding of the height field, purely so the shading gradient stays
    finite at the silhouette (a perfectly sharp step gives an infinite one)."""

    table_gray: float = 181.0
    """Measured table baseline; used to keep rendered objects in the scene's
    brightness range."""

    def __post_init__(self) -> None:
        if self.size_px_min < 8 or self.size_px_max <= self.size_px_min:
            raise ValueError("bad size range")
        if self.feather_px < 0:
            raise ValueError("feather_px must be >= 0")


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    return y - c, x - c


def _height_field(shape: str, n: int, rng: np.random.Generator, cfg: ProceduralConfig):
    """Height above the table for each pixel of an ``n x n`` object canvas.

    Returns ``h`` in [0, 1]; ``h > 0`` is the silhouette. Smooth at the rim so
    the shading gradient is finite.
    """
    y, x = _grid(n)
    r = n / 2.0
    soft = max(1e-3, cfg.edge_soft_px)

    if shape == "sphere":
        d2 = (x * x + y * y) / (r * r)
        return np.sqrt(np.clip(1.0 - d2, 0.0, 1.0)).astype(np.float32)

    if shape == "cylinder":
        # flat top, rounded rim
        d = np.sqrt(x * x + y * y)
        return _smoothstep((r - d) / soft).astype(np.float32)

    if shape == "blob":
        # jittered radius -> a lumpy, irregular object
        ang = np.arctan2(y, x)
        k = rng.integers(3, 6)
        wob = np.ones_like(ang)
        for i in range(1, k + 1):
            ph = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.06, 0.20) / i
            wob += amp * np.sin(i * ang + ph)
        d = np.sqrt(x * x + y * y) / np.maximum(wob, 0.35)
        return _smoothstep((r - d) / soft).astype(np.float32)

    # box: plateau with rounded edges, slightly rotated
    th = rng.uniform(0, np.pi / 2)
    xr = x * np.cos(th) + y * np.sin(th)
    yr = -x * np.sin(th) + y * np.cos(th)
    half = r * rng.uniform(0.78, 0.95)
    d = np.minimum(half - np.abs(xr), half - np.abs(yr))
    return _smoothstep(d / soft).astype(np.float32)


def _shade(h: np.ndarray, base_rgb: np.ndarray, cfg: ProceduralConfig) -> np.ndarray:
    """Lambertian shade of a height field, lit from the measured direction."""
    gy, gx = np.gradient(h)
    # Slope-to-normal scale. Must be the object's radius in pixels, not an O(1)
    # constant: the height field rises over a length scale of r, so dh/dx is
    # ~1/r and a small bump leaves every normal within a few degrees of +z --
    # the object then renders perfectly flat. Choosing bump = r makes a
    # hemisphere's rim normals approach horizontal, as they should.
    bump = max(1.0, h.shape[0] / 2.0)
    nz = np.ones_like(h)
    norm = np.sqrt(gx * gx * bump * bump + gy * gy * bump * bump + nz * nz)
    nx, ny, nz = -gx * bump / norm, -gy * bump / norm, nz / norm

    az = np.deg2rad(cfg.shadow_azimuth_deg + 180.0)  # light is opposite the shadow
    el = np.deg2rad(cfg.light_elev_deg)
    lx, ly, lz = np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)

    ndl = nx * lx + ny * ly + nz * lz
    # a flat top face (N = +z) maps to exactly 1.0
    shade = np.clip(ndl / max(1e-6, np.cos(el)), 0.0, 1.4)

    # contact occlusion at the silhouette: darken where the object meets the table
    rim = 1.0 - cfg.ambient_rim * (1.0 - _smoothstep(h / max(1e-3, cfg.edge_soft_px)))
    fac = (1.0 - cfg.shade_contrast * (1.0 - shade)) * rim
    rgb = base_rgb[None, None, :] * fac[..., None]
    return np.clip(rgb, 0, 255).astype(np.float32)


def _sample_color(rng: np.random.Generator, cfg: ProceduralConfig) -> np.ndarray:
    """A base colour that is not the cube's blue and not the container's red."""
    def forbidden(hue: float) -> bool:
        return any(lo <= hue <= hi for lo, hi in cfg.forbidden_hues)

    hue = float(rng.uniform(0, 180))
    for _ in range(cfg.hue_retries):
        if not forbidden(hue):
            break
        hue = float(rng.uniform(0, 180))
    else:
        # bounded retries exhausted (tiny forbidden bands, but be deterministic)
        hue = next(h for h in np.arange(0, 180, 0.5) if not forbidden(float(h)))

    hsv = np.array([[[hue, rng.uniform(*cfg.sat_range), rng.uniform(*cfg.val_range)]]], np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)


def forbidden_fraction(rgb_obj: np.ndarray, h: np.ndarray, cfg: ProceduralConfig):
    """Worst fraction of the object's core pixels inside any forbidden box.

    Returns ``(fraction, label)``. The core is ``h > 0.5`` so the feathered rim
    does not vote -- those pixels are blended with the table anyway.
    """
    core = h > 0.5
    if not core.any():
        return 0.0, ""
    arr = np.clip(rgb_obj, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    worst, label = 0.0, ""
    n_core = float(core.sum())
    for name, lo, hi in cfg.forbidden_boxes:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
        frac = float((m & core).sum()) / n_core
        if frac > worst:
            worst, label = frac, name
    return worst, label


def _neutralise(rgb_obj: np.ndarray) -> np.ndarray:
    """Drop the hue but keep the shading: a safe last resort.

    Saturation 0 cannot satisfy any forbidden box (all of them have an S floor
    of 70), so this is a guarantee, not a hope.
    """
    g = np.clip(rgb_obj, 0, 255) @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.repeat(g[..., None], 3, axis=2).astype(np.float32)


class ProceduralCutoutProvider:
    """Default :class:`CutoutProvider`: draws a random small object."""

    def __init__(self, cfg: ProceduralConfig | None = None, seed: int | None = None):
        self.cfg = cfg or ProceduralConfig()
        self._rng = np.random.default_rng(seed)

    def sample(self, rng: np.random.Generator | None = None) -> Cutout:
        cfg = self.cfg
        rng = rng if rng is not None else self._rng

        n = int(rng.integers(cfg.size_px_min, cfg.size_px_max + 1))
        shape = str(rng.choice(cfg.shapes))
        h = _height_field(shape, n, rng, cfg)
        alpha_hard = (h > 0).astype(np.float32)

        # Colour, checked on the SHADED result rather than on the drawn hue --
        # channel clipping during shading moves the hue, so a base colour just
        # outside the band can render inside it.
        rgb_obj, worst = None, None
        for _ in range(cfg.color_retries):
            cand = _shade(h, _sample_color(rng, cfg), cfg)
            frac, label = forbidden_fraction(cand, h, cfg)
            if frac <= cfg.forbidden_frac_max:
                rgb_obj = cand
                break
            rgb_obj, worst = cand, (frac, label)
        else:
            # Every draw landed in a forbidden box. Ship a grey object rather
            # than one the target's own detector would fire on.
            rgb_obj = _neutralise(rgb_obj)

        # --- canvas: object plus wherever its shadow lands ------------------
        # shadow azimuth is measured in image coords with +y pointing DOWN
        az = np.deg2rad(cfg.shadow_azimuth_deg)
        ux, uy = float(np.cos(az)), float(np.sin(az))
        length = cfg.shadow_len_ratio * n          # mean offset of the copy
        spread = cfg.shadow_spread_ratio * n       # penumbra width
        far = length + spread / 2.0                # the furthest copy's offset
        pad = int(np.ceil(max(0.0, cfg.shadow_blur_px) * 3)) + 1

        # The canvas must hold the FURTHEST copy, not the mean one: sizing it by
        # `length` clips the penumbra at the far end and leaves a straight edge.
        fdx, fdy = far * ux, far * uy
        x0, y0 = min(0, int(np.floor(fdx))) - pad, min(0, int(np.floor(fdy))) - pad
        x1 = max(n, int(np.ceil(n + fdx))) + pad
        y1 = max(n, int(np.ceil(n + fdy))) + pad
        W, H = x1 - x0, y1 - y0
        ox, oy = -x0, -y0  # the object's top-left inside the canvas

        rgb = np.zeros((H, W, 3), np.uint8)
        alpha = np.zeros((H, W), np.float32)
        rgb[oy:oy + n, ox:ox + n] = np.clip(rgb_obj, 0, 255).astype(np.uint8)
        alpha[oy:oy + n, ox:ox + n] = alpha_hard

        if cfg.feather_px > 0:
            k = 2 * int(cfg.feather_px) + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), cfg.feather_px / 2.0)

        # --- contact shadow: the silhouette spread along the light ----------
        # The warp must be applied to the silhouette ALREADY placed in canvas
        # coordinates. Warping the raw n x n object mask instead translates it
        # from y ~ 0, which throws it off-canvas and leaves a 5 px sliver.
        #
        # The offsets are spread around `shadow_len_ratio`, weighted by a
        # Gaussian, which is what gives the penumbra; the geometry and its fit
        # to the measurement are documented on `shadow_spread_ratio`.
        hard = (alpha > 0).astype(np.float32)
        S = max(1, int(cfg.shadow_steps))
        sigma = max(spread / 2.4, 1e-6)
        acc = np.zeros((H, W), np.float32)
        wsum = 0.0
        for u in np.linspace(0.0, 1.0, S):
            off = length + (float(u) - 0.5) * spread
            w = float(np.exp(-0.5 * ((off - length) / sigma) ** 2))
            if w <= 1e-6:
                continue
            acc += w * cv2.warpAffine(
                hard, np.float32([[1, 0, off * ux], [0, 1, off * uy]]), (W, H),
                flags=cv2.INTER_LINEAR)
            wsum += w
        if wsum > 0:
            acc /= wsum
        if cfg.shadow_blur_px > 0:
            k = 2 * int(np.ceil(cfg.shadow_blur_px)) + 1
            acc = cv2.GaussianBlur(acc, (k, k), cfg.shadow_blur_px / 2.0)

        # the object hides its own shadow
        shadow = np.clip(acc - _smoothstep(np.clip(alpha * 1.5, 0, 1)), 0.0, 1.0)

        # Re-normalise so the value at the contact edge -- the first table pixel
        # outside the silhouette, which is the "shadow core" the measurement
        # refers to -- is 1.0, leaving `shadow_strength` to set its darkness.
        # Without this the stack's peak sits under the object and the visible
        # part never reaches the measured 0.57 x plate.
        edge = (alpha <= 0.02) & (shadow > 0)
        if edge.any():
            peak = float(shadow[edge].max())
            if peak > 1e-6:
                shadow = np.clip(shadow / peak, 0.0, 1.0).astype(np.float32)

        return Cutout(rgb=rgb, alpha=alpha, shadow=shadow,
                      anchor=(oy + (n - 1) / 2.0, ox + (n - 1) / 2.0))
