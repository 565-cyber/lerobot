"""Episode-level interaction-object tracker (RoboAgent-style idea, v1).

Instead of re-detecting "all blue" every frame, this tracker first *initializes*
on the interaction object (frame 0 of an episode) and then *tracks its identity*
across frames by searching near the previous location. This is what lets the
augmentation recolor only the robot's target object and leave background blue
objects / robot parts untouched.

State machine::

    uninit --initialize--> tracking <--update--> occluded --timeout--> lost (terminal)

``lost`` is terminal: once the target is gone (e.g. dropped into the trash) we
never re-acquire a *different* blue object (宁可漏增强，不要错增强).

Robot state is accepted (``state=None``) but not used in v1: observation.state is
6 joint angles (deg), and an end-effector image prior would need SO101 FK + an
uncalibrated camera projection, which is out of scope for v1.
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


@dataclass
class TrackerConfig:
    """Tracking / init thresholds. Extends the HSV config in ``mask``."""

    mask: BlueMaskConfig = field(default_factory=BlueMaskConfig)

    # candidate area bounds for the target (up camera cube: ~55..1300 px)
    min_area: int = 40
    max_area: int = 4000

    # init geometry gates (target is a small, roughly square, compact block)
    aspect_min: float = 0.4
    aspect_max: float = 2.5
    extent_min: float = 0.35
    init_area_prior: float = 1000.0

    # tracking search region = search_scale * max(w,h) + search_pad (px)
    search_scale: float = 4.0
    search_pad: int = 80
    # while occluded, grow the search region by occlusion_grow px/frame (capped),
    # so a target that moved while hidden (e.g. carried in the gripper) is recovered.
    occlusion_grow: float = 1.0
    max_search_pad: int = 300

    # frames of no-detection before declaring the target terminally lost
    # (bridges the ~90-130 frame grasp occlusion while still terminating after drop)
    occlusion_timeout: int = 150

    # tracking score weights (distance is in px, area/color terms scaled to px)
    w_area: float = 1.0
    w_color: float = 1.0

    # optional workspace ROI (x0, y0, x1, y1); None = full frame
    roi: tuple[int, int, int, int] | None = None


def _center_in_roi(center, roi):
    cx, cy = center
    x0, y0, x1, y1 = roi
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class InteractionObjectTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.reset()

    def reset(self):
        self.state = "uninit"  # uninit | tracking | occluded | lost
        self.target = None      # dict: center, bbox, w, h, area, color_mean, confidence
        self.occluded_frames = 0
        self.last_reason = None
        self.n_candidates = 0

    # ---- public API -------------------------------------------------------
    def initialize(self, image, state=None) -> np.ndarray:
        """Initialize the target on the first frame. Returns its mask (or zeros)."""
        comps = self._candidates(image)
        cands = [c for c in comps if self._passes_geometry(c)]
        if self.config.roi is not None:
            cands = [c for c in cands if _center_in_roi(c["center"], self.config.roi)]

        if not cands:
            self.state = "lost"
            self.last_reason = "init_no_candidate"
            return self._zero(image)

        best = min(cands, key=lambda c: self._init_cost(c))
        self._adopt(best, confidence=1.0 / (1.0 + self._init_cost(best)))
        self.state = "tracking"
        self.last_reason = "init"
        return best["mask"]

    def update(self, image, state=None) -> np.ndarray:
        """Track the target in the next frame. Returns its mask (or zeros)."""
        if self.state == "lost":
            return self._zero(image)  # terminal, never re-acquire
        if self.target is None:
            return self.initialize(image, state)

        comps = self._candidates(image)
        region = self._search_region(image)
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

        best = min(cands, key=lambda c: self._track_cost(c))
        self._adopt(best, confidence=1.0 / (1.0 + self._track_cost(best) / 50.0))
        self.occluded_frames = 0
        self.state = "tracking"
        self.last_reason = "track"
        return best["mask"]

    def get_target_observation(self) -> dict | None:
        """Return the tracked target's identity/shape observation (for multi-view fusion).

        None if no target is currently held. Includes the fields the front
        (slave) tracker needs to associate its own candidates to the same object.
        """
        if self.target is None:
            return None
        return {
            "bbox": self.target["bbox"],
            "center": self.target["center"],
            "area": self.target["area"],
            "aspect": self.target["aspect"],
            "extent": self.target["extent"],
            "color_mean": self.target["color_mean"],
        }

    def get_debug_info(self) -> dict:
        if self.target is None:
            return {
                "state": self.state,
                "bbox": None,
                "center": None,
                "area": 0,
                "confidence": 0.0,
                "occluded_frames": self.occluded_frames,
                "reason": self.last_reason,
                "n_candidates": self.n_candidates,
            }
        return {
            "state": self.state,
            "bbox": self.target["bbox"],
            "center": self.target["center"],
            "area": self.target["area"],
            "confidence": self.target["confidence"],
            "occluded_frames": self.occluded_frames,
            "reason": self.last_reason,
            "n_candidates": self.n_candidates,
        }

    # ---- internals --------------------------------------------------------
    def _zero(self, image) -> np.ndarray:
        h, w = _to_hwc_uint8(image).shape[:2]
        return np.zeros((h, w), np.uint8)

    def _candidates(self, image):
        comps = detect_blue_components(image, self.config.mask)
        comps = [c for c in comps if self.config.min_area <= c["area"] <= self.config.max_area]
        self.n_candidates = len(comps)
        return comps

    def _passes_geometry(self, c) -> bool:
        return (
            self.config.aspect_min <= c["aspect"] <= self.config.aspect_max
            and c["extent"] >= self.config.extent_min
        )

    def _search_region(self, image):
        h, w = _to_hwc_uint8(image).shape[:2]
        cx, cy = self.target["center"]
        pad = self.config.search_pad + min(
            self.occluded_frames * self.config.occlusion_grow, self.config.max_search_pad
        )
        r = self.config.search_scale * max(self.target["w"], self.target["h"]) + pad
        x0 = int(max(0, cx - r))
        x1 = int(min(w, cx + r))
        y0 = int(max(0, cy - r))
        y1 = int(min(h, cy + r))
        return (x0, y0, x1, y1)

    def _in_region(self, c, region):
        cx, cy = c["center"]
        x0, y0, x1, y1 = region
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def _init_cost(self, c) -> float:
        cost = abs(math.log(c["area"] / max(self.config.init_area_prior, 1.0)))
        cost += 2.0 * abs(c["aspect"] - 1.0)  # prefer square blocks
        if self.config.roi is not None:
            x0, y0, x1, y1 = self.config.roi
            cost += 0.001 * _dist(c["center"], ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        return cost

    def _track_cost(self, c) -> float:
        d = _dist(c["center"], self.target["center"])
        area_diff = abs(c["area"] - self.target["area"]) / max(self.target["area"], 1.0)
        color_diff = math.dist(c["color_mean"], self.target["color_mean"]) / 255.0
        return d + self.config.w_area * 30.0 * area_diff + self.config.w_color * 100.0 * color_diff

    def _adopt(self, c, confidence: float):
        self.target = {
            "center": c["center"],
            "bbox": c["bbox"],
            "w": c["w"],
            "h": c["h"],
            "area": c["area"],
            "aspect": c["aspect"],
            "extent": c["extent"],
            "color_mean": c["color_mean"],
            "confidence": float(confidence),
        }
