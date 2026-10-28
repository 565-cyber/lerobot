"""Lightweight robot-state-aware interaction prior (RoboAgent-style, no FK).

Turns the 6-d ``observation.state`` (5 SO-101 arm joints + gripper opening
width) into an interaction *phase* plus booleans for the grasp/release events.
It does **not** localize anything in pixels: there is no FK and no camera
projection here. It only tells the tracker *whether* the robot is manipulating
an object and *which* manipulation sub-phase it is in.

Joint layout (from the parquet feature names, verified in
``scripts/analyze_robot_state.py``):

    state[0] shoulder_pan   (base yaw)
    state[1] shoulder_lift
    state[2] elbow_flex     (dominant vertical-reach signal: 99 home -> -15 down)
    state[3] wrist_flex
    state[4] wrist_roll
    state[5] gripper        (opening width: 13.5 closed / 17 holding / 26-31 open)

Phase = f(elbow_flex, shoulder_lift, gripper, d(elbow_flex), d(gripper)):
    home        arm folded + gripper closed            -> no interaction
    approaching arm descending / gripper pre-opening   -> target still on table
    grasping    gripper closing while arm down         -> grasp moment
    moving      arm down, gripper holding              -> carry
    placing     gripper opening while arm down         -> release moment
    retracting  arm ascending back home                -> no interaction

``grasping`` / ``releasing`` are the transient close/open events (windowed
gripper delta), so they fire over the short window when the gripper actually
moves, independent of the other joints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotStatePriorConfig:
    """All thresholds live here (no magic numbers in the classifier)."""

    # joint indices (SO-101 layout)
    idx_shoulder_pan: int = 0
    idx_shoulder_lift: int = 1
    idx_elbow_flex: int = 2
    idx_wrist_flex: int = 3
    idx_wrist_roll: int = 4
    idx_gripper: int = 5

    # arm height (elbow_flex dominates vertical reach)
    elbow_up_thr: float = 70.0    # above = arm folded up
    elbow_down_thr: float = 0.0   # below = arm at the table
    lift_home_thr: float = -60.0  # shoulder_lift below = folded

    # gripper opening-width states
    grip_closed_thr: float = 16.0  # below = fully closed (home / no object)
    grip_open_thr: float = 22.0    # above = fully open (pre-grasp); holding lives between

    # windowed event detection: compare joint value to K frames ago
    window: int = 15
    close_delta: float = 2.0    # gripper falls more than this over the window = grasp
    open_delta: float = 1.5     # gripper rises more than this over the window = release
    rise_delta: float = 2.0     # elbow rises more than this over the window = retract
    descend_delta: float = 2.0  # elbow falls more than this over the window = descend

    # confidence bounds
    conf_min: float = 0.5
    conf_max: float = 0.95


def _to_flat(state) -> np.ndarray:
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    return np.asarray(state, dtype=np.float32).reshape(-1)


class RobotStatePrior:
    """Stateful phase estimator: remembers the previous states to compute windowed
    joint deltas for the grasp/release events."""

    def __init__(self, config: RobotStatePriorConfig | None = None):
        self.config = config or RobotStatePriorConfig()
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.config.window + 1)
        self.phase = "home"
        self.confidence = self.config.conf_min

    # ---- public API -------------------------------------------------------
    def update(self, state) -> dict:
        s = _to_flat(state)
        self.history.append(s)
        cfg = self.config

        elbow = float(s[cfg.idx_elbow_flex])
        lift = float(s[cfg.idx_shoulder_lift])
        grip = float(s[cfg.idx_gripper])

        # windowed deltas (0 until enough history has accumulated)
        if len(self.history) >= cfg.window + 1:
            ref = self.history[-cfg.window - 1]
            d_elbow = elbow - float(ref[cfg.idx_elbow_flex])
            d_grip = grip - float(ref[cfg.idx_gripper])
        else:
            d_elbow = 0.0
            d_grip = 0.0

        # ---- interpretable sub-signals -----------------------------------
        if elbow > cfg.elbow_up_thr:
            arm_height = "up"
        elif elbow < cfg.elbow_down_thr:
            arm_height = "down"
        else:
            arm_height = "mid"

        if grip < cfg.grip_closed_thr:
            gripper_state = "closed"
        elif grip > cfg.grip_open_thr:
            gripper_state = "open"
        else:
            gripper_state = "holding"

        # transient events (arm down only, to separate grasp/release from the
        # pre-grasp open and the retract close)
        grasping = (d_grip < -cfg.close_delta) and (elbow < cfg.elbow_down_thr)
        releasing = (d_grip > cfg.open_delta) and (grip < cfg.grip_open_thr) and (elbow < cfg.elbow_down_thr)

        # ---- phase (priority order) --------------------------------------
        if elbow > cfg.elbow_up_thr and grip < cfg.grip_closed_thr:
            phase = "home"
        elif (elbow > cfg.elbow_up_thr) or (d_elbow > cfg.rise_delta and grip < cfg.grip_open_thr):
            phase = "retracting"
        elif releasing:
            phase = "placing"
        elif grasping:
            phase = "grasping"
        elif elbow >= cfg.elbow_down_thr or grip >= cfg.grip_open_thr:
            phase = "approaching"
        else:
            phase = "moving"

        self.phase = phase
        self.confidence = self._confidence(phase, elbow, lift, grip, d_elbow, d_grip)

        return {
            "phase": phase,
            "confidence": self.confidence,
            "arm_height": arm_height,
            "gripper_state": gripper_state,
            "grasping": grasping,
            "releasing": releasing,
        }

    # ---- internals --------------------------------------------------------
    def _confidence(self, phase, elbow, lift, grip, d_elbow, d_grip) -> float:
        cfg = self.config
        c = cfg.conf_min
        if phase == "home":
            c += 0.20 * min(1.0, (elbow - cfg.elbow_up_thr) / 30.0)
            c += 0.15 * min(1.0, (cfg.grip_closed_thr - grip) / 5.0)
        elif phase == "retracting":
            c += 0.25 * min(1.0, abs(d_elbow) / 10.0)
        elif phase == "grasping":
            c += 0.35 * min(1.0, abs(d_grip) / 5.0)
        elif phase == "placing":
            c += 0.30 * min(1.0, d_grip / 5.0)
        elif phase == "approaching":
            c += 0.20 * min(1.0, abs(d_elbow) / 20.0)
            c += 0.10 * min(1.0, (grip - cfg.grip_closed_thr) / 10.0)
        else:  # moving
            c += 0.20 * min(1.0, -elbow / 20.0)
        return float(min(cfg.conf_max, max(cfg.conf_min, c)))
