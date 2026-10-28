"""Multi-view interaction-object tracker (v1).

``up`` camera = master -> object *identity* (a single blue cube, seen from above).
``front`` camera = slave -> local *observation tracking* (associate its own blue
candidates to the up-identified object).

The two views have different viewpoints and lighting, so:

  * the up tracker is never force-projected onto the front frame (no per-frame
    bbox override); instead its target is summarized into a :class:`TargetTemplate`
    (shape + scale + appearance) that the front uses to *associate* its own
    candidates,
  * cross-view color is unreliable, so it is the weakest cue; temporal
    consistency (the front's own previous target) and shape (square, compact)
    dominate,
  * up keeps its own temporal state, front keeps its own temporal state.

If the up tracker loses the target, the template is frozen (never re-acquired
randomly); the front keeps matching with the frozen template + temporal prior.
A candidate that fails shape / area-ratio / temporal hard gates is rejected and
the frame is reported ``occluded`` (never switch to a background blue object).

No FK, no camera projection, no SAM/VLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from object_generalization.masks.blue_object_mask import (
    _to_hwc_uint8,
    detect_blue_components,
)
from object_generalization.masks.interaction_object_tracker import (
    InteractionObjectTracker,
    TrackerConfig,
)
from object_generalization.masks.robot_state_prior import (
    RobotStatePrior,
    RobotStatePriorConfig,
)


@dataclass
class TargetTemplate:
    """The up-identified object's identity summary (slowly EMA-updated)."""

    aspect: float
    area: float
    color_mean: tuple
    crop_embedding: np.ndarray  # (S, S, 3) normalized HSV patch statistics
    valid: bool = True
    updated_at: int = 0


@dataclass
class MultiViewTrackerConfig:
    """Multi-view thresholds. Reuses the up tracker; defines the front matcher."""

    up: TrackerConfig = field(default_factory=TrackerConfig)

    # front geometry gate (same discriminators as the state-aware front tracker:
    # compact + small + square; background is huge + irregular)
    front_min_area: int = 800
    front_max_area: int = 12000
    front_aspect_min: float = 0.5
    front_aspect_max: float = 2.0
    front_extent_min: float = 0.75

    # matching weights (priority: temporal > shape > appearance > color).
    # The up-template *area* is NOT a soft score term: the two views have a
    # scale ratio that varies 1.7x..7.5x across the episode (wrist approaches the
    # cube), so absolute area is only a loose hard band below. Area consistency
    # is instead measured against the front's own previous target (temporal).
    w_temporal: float = 2.0
    w_shape: float = 1.0
    w_appearance: float = 0.4
    w_color: float = 0.2

    # front search region around the front's own previous target; grows while the
    # front is occluded so the cube can be re-found after grasp/carry (mirrors the
    # state-aware front tracker's proven occlusion recovery).
    front_search_scale: float = 4.0
    front_search_pad: int = 80
    front_occlusion_grow: float = 1.5
    front_max_search_pad: int = 320

    # hard rejection gates (anti-background-switch). Deliberately NO cross-view
    # aspect gate: the up view sees a square (top face) while the front sees an
    # oblique square (aspect 0.5..2.0), so |aspect_front - aspect_up| legitimately
    # reaches ~1.0 — the front geometry gate (front_aspect_min/max) already bounds
    # aspect, and background is excluded by area/extent. Only the loose area-ratio
    # band survives as a sanity check.
    area_ratio_min: float = 0.2        # front_area / up_area lower bound (very loose)
    area_ratio_max: float = 40.0       # front_area / up_area upper bound (very loose)
    match_score_thr: float = 3.0       # total score above this -> occluded

    # template EMA update rate
    template_alpha: float = 0.1
    crop_size: int = 8

    # up view health floor: when the up target collapses below this area (the
    # gripper occluding the cube from above turns a ~1000px cube into a 50px
    # sliver), the template is FROZEN — its area must not shrink, or the front's
    # area-ratio hard gate (front/up) would explode and reject the real cube.
    up_min_template_area: float = 300.0

    # front: frames of no-match before terminally lost
    occlusion_timeout: int = 150

    # optional robot-state prior (phase output only; not used for matching)
    state_prior: RobotStatePriorConfig = field(default_factory=RobotStatePriorConfig)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class MultiViewInteractionTracker:
    def __init__(self, config: MultiViewTrackerConfig | None = None):
        self.config = config or MultiViewTrackerConfig()
        self.up_tracker = InteractionObjectTracker(self.config.up)
        self._prior = RobotStatePrior(self.config.state_prior)
        self.reset()

    def reset(self):
        self.up_tracker.reset()
        self._prior.reset()
        self.template = None
        self.front_target = None
        self.front_state = "uninit"   # uninit | tracking | occluded | lost
        self.front_occluded = 0
        self.up_state = "uninit"
        self.last_confidence = 0.0
        self.last_phase = None
        self.last_reason = None

    # ---- public API -------------------------------------------------------
    def initialize(self, up_image, front_image, robot_state=None) -> dict:
        up_mask = self.up_tracker.initialize(up_image)
        self.up_state = self.up_tracker.get_debug_info()["state"]
        obs = self.up_tracker.get_target_observation()
        if obs is not None and self._up_view_healthy(obs):
            self.template = self._build_template(up_image, obs)
        front_mask = self._match_front(front_image)
        return self._result(up_mask, front_mask, robot_state)

    def update(self, up_image, front_image, robot_state=None) -> dict:
        up_mask = self.up_tracker.update(up_image)
        self.up_state = self.up_tracker.get_debug_info()["state"]
        obs = self.up_tracker.get_target_observation()
        # only (re)build/update the template from a healthy up view; when the up
        # target is occluded or collapsing (gripper occluding the cube from above)
        # the template is frozen so its area/shape/appearance stay canonical.
        if self.up_state == "tracking" and obs is not None and self._up_view_healthy(obs):
            if self.template is None:
                self.template = self._build_template(up_image, obs)
            else:
                self._update_template(up_image, obs)
        # up occluded/lost -> template frozen (kept as-is), front keeps matching
        front_mask = self._match_front(front_image)
        return self._result(up_mask, front_mask, robot_state)

    def _up_view_healthy(self, obs) -> bool:
        return float(obs["area"]) >= self.config.up_min_template_area

    def get_debug_info(self) -> dict:
        return {
            "up_state": self.up_state,
            "front_state": self.front_state,
            "front_bbox": self.front_target["bbox"] if self.front_target else None,
            "front_center": self.front_target["center"] if self.front_target else None,
            "front_area": self.front_target["area"] if self.front_target else 0,
            "front_aspect": self.front_target["aspect"] if self.front_target else 0.0,
            "confidence": self.last_confidence,
            "phase": self.last_phase,
            "reason": self.last_reason,
            "template_valid": self.template is not None and self.template.valid,
            "up_obs": self.up_tracker.get_target_observation(),
        }

    # ---- internals --------------------------------------------------------
    def _result(self, up_mask, front_mask, robot_state=None) -> dict:
        if robot_state is not None:
            self.last_phase = self._prior.update(robot_state)["phase"]
        return {
            "up_mask": up_mask,
            "front_mask": front_mask,
            "confidence": self.last_confidence,
            "phase": self.last_phase,
            "status": self.front_state,
        }

    def _build_template(self, up_image, obs) -> TargetTemplate:
        return TargetTemplate(
            aspect=float(obs["aspect"]),
            area=float(obs["area"]),
            color_mean=tuple(float(v) for v in obs["color_mean"]),
            crop_embedding=self._crop_embedding(up_image, obs["bbox"]),
            valid=True,
            updated_at=0,
        )

    def _update_template(self, up_image, obs):
        a = self.config.template_alpha
        t = self.template
        t.aspect = (1.0 - a) * t.aspect + a * float(obs["aspect"])
        t.area = (1.0 - a) * t.area + a * float(obs["area"])
        t.color_mean = tuple((1.0 - a) * t.color_mean[i] + a * float(obs["color_mean"][i]) for i in range(3))
        emb = self._crop_embedding(up_image, obs["bbox"])
        t.crop_embedding = (1.0 - a) * t.crop_embedding + a * emb
        t.updated_at += 1

    def _crop_embedding(self, image, bbox) -> np.ndarray:
        S = self.config.crop_size
        rgb = _to_hwc_uint8(image)
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((S, S, 3), np.float32)
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        hsv = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV).astype(np.float32)
        small = cv2.resize(hsv, (S, S), interpolation=cv2.INTER_AREA)
        small[..., 0] /= 180.0   # H
        small[..., 1:] /= 255.0  # S, V
        return small

    def _front_geometry_ok(self, c) -> bool:
        cfg = self.config
        return (
            cfg.front_min_area <= c["area"] <= cfg.front_max_area
            and cfg.front_aspect_min <= c["aspect"] <= cfg.front_aspect_max
            and c["extent"] >= cfg.front_extent_min
        )

    def _match_front(self, front_image) -> np.ndarray:
        cfg = self.config
        if self.front_state == "lost":
            return self._zero(front_image)
        if self.template is None or not self.template.valid:
            self.front_state = "uninit"
            self.last_reason = "no_up_identity"
            return self._zero(front_image)

        comps = detect_blue_components(front_image, cfg.up.mask)
        cands = [c for c in comps if self._front_geometry_ok(c)]
        if not cands:
            return self._occlude(front_image)

        # restrict to a search region around the front's own previous target; the
        # region grows while occluded so the cube is re-found after grasp/carry.
        # (If we have a previous target, drop candidates far outside it — this is
        # the temporal prior that prevents a jump to a background blue object.)
        if self.front_target is not None:
            region = self._front_search_region(front_image)
            local = [c for c in cands if self._in_region(c, region)]
            if local:
                cands = local

        for c in cands:
            c["_emb"] = self._crop_embedding(front_image, c["bbox"])

        best, best_score = None, float("inf")
        for c in cands:
            score, hard = self._match_score(c)
            if hard:
                continue
            if score < best_score:
                best_score, best = score, c

        if best is None or best_score > cfg.match_score_thr:
            self.last_reason = "no_match" if best is None else "score_high"
            return self._occlude(front_image)

        self.front_target = self._to_front_target(best, best_score)
        self.front_state = "tracking"
        self.front_occluded = 0
        self.last_confidence = 1.0 / (1.0 + best_score)
        self.last_reason = "match"
        return best["mask"]

    def _front_search_region(self, front_image):
        h, w = _to_hwc_uint8(front_image).shape[:2]
        cx, cy = self.front_target["center"]
        pad = self.config.front_search_pad + min(
            self.front_occluded * self.config.front_occlusion_grow,
            self.config.front_max_search_pad,
        )
        bbox = self.front_target["bbox"]
        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))
        r = self.config.front_search_scale * max(bw, bh) + pad
        x0 = int(max(0, cx - r))
        x1 = int(min(w, cx + r))
        y0 = int(max(0, cy - r))
        y1 = int(min(h, cy + r))
        return (x0, y0, x1, y1)

    def _in_region(self, c, region):
        cx, cy = c["center"]
        x0, y0, x1, y1 = region
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def _match_score(self, c):
        cfg = self.config
        t = self.template

        # shape: prefer a near-square cube (up sees square, front sees oblique square)
        shape_cost = abs(c["aspect"] - 1.0)
        # appearance: cross-view texture (weak; lighting/viewpoint differ)
        app_cost = float(np.abs(c["_emb"] - t.crop_embedding).mean())
        # color: weakest cue (lighting differs between views)
        color_cost = math.dist(c["color_mean"], t.color_mean) / 255.0

        if self.front_target is not None:
            # temporal = the front's OWN previous target: center + area consistency.
            # Dominant cue, and scale-invariant (front's own area history, not the up
            # template's absolute area, which varies 1.7x..7.5x). After a long
            # occlusion the previous position is stale (the cube may have moved in
            # the gripper), so the temporal prior is scaled down toward zero.
            jump_px = _dist(c["center"], self.front_target["center"])
            d_center = jump_px / 100.0
            d_area = abs(math.log((c["area"] + 1e-6) / max(self.front_target["area"], 1e-6)))
            staleness = 1.0 / (1.0 + self.front_occluded)
            temporal_cost = staleness * (d_center + d_area)
        else:
            # first match: no front history -> loose scale prior against the up template
            temporal_cost = abs(math.log((c["area"] + 1e-6) / max(t.area, 1e-6)))

        score = (
            cfg.w_temporal * temporal_cost
            + cfg.w_shape * shape_cost
            + cfg.w_appearance * app_cost
            + cfg.w_color * color_cost
        )

        ratio = (c["area"] + 1e-6) / max(t.area, 1e-6)
        hard = ratio < cfg.area_ratio_min or ratio > cfg.area_ratio_max
        return score, hard

    def _to_front_target(self, c, score):
        return {
            "center": c["center"],
            "bbox": c["bbox"],
            "area": c["area"],
            "aspect": c["aspect"],
            "color_mean": c["color_mean"],
            "confidence": 1.0 / (1.0 + score),
        }

    def _occlude(self, front_image) -> np.ndarray:
        self.front_occluded += 1
        if self.front_occluded > self.config.occlusion_timeout:
            self.front_state = "lost"
            self.last_reason = "lost_occluded"
        else:
            self.front_state = "occluded"
            self.last_reason = "occluded"
        self.last_confidence = 0.0
        return self._zero(front_image)

    def _zero(self, image) -> np.ndarray:
        h, w = _to_hwc_uint8(image).shape[:2]
        return np.zeros((h, w), np.uint8)
