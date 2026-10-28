"""Front (wrist) camera interaction-object tracker (RoboAgent-style, v2).

The front camera moves with the arm and sees large *irregular* blue background
regions (20k-46k px, extent 0.5-0.7) in addition to the *compact* target cube
(~7.5k px, extent 0.8-0.9) that is only clearly visible while approached/grasped.

v2 is *robot-state-aware*: instead of a single-joint elbow threshold it consumes
the output of :class:`RobotStatePrior` (``{"phase", "confidence", ...}``) and
uses the phase to (1) gate whether tracking is even allowed, (2) widen/tighten
the search region, and (3) weight previous-target consistency. It never uses the
state to localize the cube in pixels (no FK / no camera projection).

Discriminators (priority order):
  1. geometry: the target is compact (extent >= extent_min), small (area <=
     max_area, below the background's >=13k), and near-square.
  2. interaction prior (phase): track only while the arm is engaging the object
     (approaching / grasping / moving / placing); suppress in home / retracting.

States: awaiting -> tracking <-> occluded -> lost (terminal, no re-acquire).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from object_generalization.masks.blue_object_mask import (
    BlueMaskConfig,
    _to_hwc_uint8,
    detect_blue_components,
)

# phases in which the robot is actively manipulating the object
INTERACTING_PHASES = {"approaching", "grasping", "moving", "placing"}
NON_INTERACTING_PHASES = {"home", "retracting"}


@dataclass
class FrontTrackerConfig:
    """Front-camera tracker thresholds. Extends the HSV config in ``mask``."""

    mask: BlueMaskConfig = field(default_factory=BlueMaskConfig)

    # geometry gates (target: compact + small + square; background: huge + irregular)
    min_area: int = 800
    max_area: int = 12000
    aspect_min: float = 0.5
    aspect_max: float = 2.0
    extent_min: float = 0.75
    init_area_prior: float = 7000.0

    # tracking search region = search_scale * max(w,h) + search_pad (px), grows while occluded
    search_scale: float = 4.0
    search_pad: int = 80
    occlusion_grow: float = 1.0
    max_search_pad: int = 300

    occlusion_timeout: int = 150
    w_area: float = 1.0
    w_color: float = 1.0

    # ---- robot-state-aware search / scoring ------------------------------
    # per-phase search-radius multiplier (approach: wrist moves fast toward table)
    search_scale_approach: float = 1.6
    search_scale_grasp: float = 0.8
    search_scale_moving: float = 0.7
    search_scale_placing: float = 1.3

    # per-phase weight on distance-to-previous-target (higher = trust prev bbox,
    # fewer target switches)
    w_dist_approach: float = 0.6
    w_dist_grasp: float = 1.5
    w_dist_moving: float = 2.0
    w_dist_placing: float = 1.0

    # if the prior is this uncertain, prefer "occluded" over adopting a candidate
    min_prior_confidence: float = 0.5


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class FrontInteractionObjectTracker:
    def __init__(self, config: FrontTrackerConfig | None = None):
        self.config = config or FrontTrackerConfig()
        self.reset()

    def reset(self):
        self.state = "awaiting"  # awaiting | tracking | occluded | lost
        self.target = None
        self.occluded_frames = 0
        self.last_reason = None
        self.n_candidates = 0
        self.last_phase = None
        self.prior_confidence = 0.0

    # ---- public API -------------------------------------------------------
    def initialize(self, image, prior: dict | None = None) -> np.ndarray:
        """Try to lock the target on the first frame. Returns its mask (or zeros).

        ``prior`` is the output dict of :class:`RobotStatePrior`.
        """
        self._note_prior(prior)
        if not self._interaction_active(prior):
            self.state = "awaiting"
            self.last_reason = f"phase_{self.last_phase or 'none'}"
            return self._zero(image)
        # releasing: the object is leaving the gripper; never (re)initialize here
        if prior is not None and prior.get("releasing", False):
            self.state = "awaiting"
            self.last_reason = "releasing"
            return self._zero(image)
        target = self._find_target(image)
        if target is None:
            self.state = "awaiting"
            self.last_reason = "no_compact_candidate"
            return self._zero(image)
        self._adopt(target, confidence=1.0 / (1.0 + self._init_cost(target)))
        self.state = "tracking"
        self.last_reason = "init"
        return target["mask"]

    def update(self, image, prior: dict | None = None) -> np.ndarray:
        """Track the target in the next frame. Returns its mask (or zeros)."""
        if self.state == "lost":
            return self._zero(image)
        if self.target is None:
            # appearance-conditional init: keep scanning until a valid cube appears
            return self.initialize(image, prior)

        self._note_prior(prior)
        if not self._interaction_active(prior):
            # arm retracted / home -> the gripper no longer holds the cube
            self.occluded_frames += 1
            if self.occluded_frames > self.config.occlusion_timeout:
                self.state = "lost"
                self.last_reason = f"lost_phase_{self.last_phase or 'none'}"
            else:
                self.state = "occluded"
                self.last_reason = f"phase_{self.last_phase or 'none'}"
            return self._zero(image)

        comps = self._candidates(image)
        region = self._search_region(image, prior)
        cands = [c for c in comps if self._in_region(c, region)]

        if not cands:
            self.occluded_frames += 1
            if self.occluded_frames > self.config.occlusion_timeout:
                self.state = "lost"
                self.last_reason = "lost_occluded"
            else:
                self.state = "occluded"
                self.last_reason = "occluded"
            return self._zero(image)

        best = min(cands, key=lambda c: self._track_cost(c, prior))
        # anti-switch guard: refuse to jump to a candidate when the prior is
        # uncertain (宁可漏增强, 不要错增强 -> stay occluded, never switch to bg)
        if self._switch_risk(best, prior):
            self.occluded_frames += 1
            self.state = "occluded"
            self.last_reason = "low_confidence_guard"
            return self._zero(image)

        self._adopt(best, confidence=1.0 / (1.0 + self._track_cost(best, prior) / 50.0))
        self.occluded_frames = 0
        self.state = "tracking"
        self.last_reason = "track"
        return best["mask"]

    def get_debug_info(self) -> dict:
        base = {
            "state": self.state,
            "bbox": None,
            "center": None,
            "area": 0,
            "extent": 0.0,
            "confidence": 0.0,
            "occluded_frames": self.occluded_frames,
            "reason": self.last_reason,
            "n_candidates": self.n_candidates,
            "phase": self.last_phase,
            "prior_confidence": self.prior_confidence,
        }
        if self.target is not None:
            base.update(
                {
                    "bbox": self.target["bbox"],
                    "center": self.target["center"],
                    "area": self.target["area"],
                    "extent": self.target["extent"],
                    "confidence": self.target["confidence"],
                }
            )
        return base

    # ---- internals --------------------------------------------------------
    def _note_prior(self, prior):
        if prior is not None:
            self.last_phase = prior.get("phase", None)
            self.prior_confidence = float(prior.get("confidence", 0.0))

    def _interaction_active(self, prior) -> bool:
        if prior is None:
            return True
        phase = prior.get("phase", "moving")
        return phase in INTERACTING_PHASES

    def _switch_risk(self, cand, prior) -> bool:
        # a "switch" is a large center jump away from the current target
        jump = _dist(cand["center"], self.target["center"])
        if jump < 60.0:  # normal frame-to-frame motion (gripper-fixed cube)
            return False
        # only allow a large jump when the prior is confident we are mid-interaction
        if prior is None:
            return True
        return prior.get("confidence", 0.0) < self.config.min_prior_confidence

    def _zero(self, image) -> np.ndarray:
        h, w = _to_hwc_uint8(image).shape[:2]
        return np.zeros((h, w), np.uint8)

    def _candidates(self, image):
        comps = detect_blue_components(image, self.config.mask)
        comps = [
            c
            for c in comps
            if self.config.min_area <= c["area"] <= self.config.max_area
            and self.config.aspect_min <= c["aspect"] <= self.config.aspect_max
            and c["extent"] >= self.config.extent_min
        ]
        self.n_candidates = len(comps)
        return comps

    def _find_target(self, image):
        comps = self._candidates(image)
        if not comps:
            return None
        return min(comps, key=self._init_cost)

    def _search_region(self, image, prior):
        h, w = _to_hwc_uint8(image).shape[:2]
        cx, cy = self.target["center"]
        pad = self.config.search_pad + min(
            self.occluded_frames * self.config.occlusion_grow, self.config.max_search_pad
        )
        scale = self.config.search_scale * self._phase_search_mult(prior)
        r = scale * max(self.target["w"], self.target["h"]) + pad
        x0 = int(max(0, cx - r))
        x1 = int(min(w, cx + r))
        y0 = int(max(0, cy - r))
        y1 = int(min(h, cy + r))
        return (x0, y0, x1, y1)

    def _phase_search_mult(self, prior) -> float:
        phase = self.last_phase
        return {
            "approaching": self.config.search_scale_approach,
            "grasping": self.config.search_scale_grasp,
            "moving": self.config.search_scale_moving,
            "placing": self.config.search_scale_placing,
        }.get(phase, 1.0)

    def _in_region(self, c, region):
        cx, cy = c["center"]
        x0, y0, x1, y1 = region
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def _init_cost(self, c) -> float:
        return (
            abs(math.log(c["area"] / max(self.config.init_area_prior, 1.0)))
            + 2.0 * abs(c["aspect"] - 1.0)
            + 3.0 * (1.0 - c["extent"])
        )

    def _track_cost(self, c, prior) -> float:
        d = _dist(c["center"], self.target["center"])
        area_diff = abs(c["area"] - self.target["area"]) / max(self.target["area"], 1.0)
        color_diff = math.dist(c["color_mean"], self.target["color_mean"]) / 255.0
        w_dist = self._phase_dist_weight(prior)
        return w_dist * d + self.config.w_area * 30.0 * area_diff + self.config.w_color * 100.0 * color_diff

    def _phase_dist_weight(self, prior) -> float:
        phase = self.last_phase
        return {
            "approaching": self.config.w_dist_approach,
            "grasping": self.config.w_dist_grasp,
            "moving": self.config.w_dist_moving,
            "placing": self.config.w_dist_placing,
        }.get(phase, 1.0)

    def _adopt(self, c, confidence: float):
        self.target = {
            "center": c["center"],
            "bbox": c["bbox"],
            "w": c["w"],
            "h": c["h"],
            "area": c["area"],
            "extent": c["extent"],
            "color_mean": c["color_mean"],
            "confidence": float(confidence),
        }
