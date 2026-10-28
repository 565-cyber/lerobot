"""Blue target object detection via HSV thresholding (v1, traditional CV).

Provides three layers, reused by the interaction-object tracker:

    blue_threshold_mask    -> raw blue binary mask (all blue pixels, morphology cleaned)
    detect_blue_components -> list of connected blue components (candidates)
    detect_blue_mask       -> single "best" component with conservative gating

No SAM / no segmentation model. All thresholds are config parameters.
The default HSV range is the one calibrated for `newdata` in
``workplace/annotate_target.py`` (COLOR_HSV["blue"]).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class BlueMaskConfig:
    """HSV / morphology thresholds. Keep all magic numbers here."""

    hsv_lower: tuple[int, int, int] = (95, 70, 50)      # (H, S, V) lower bound
    hsv_upper: tuple[int, int, int] = (140, 255, 255)   # (H, S, V) upper bound
    open_kernel: int = 5       # morphological opening kernel size (px)
    close_kernel: int = 7      # morphological closing kernel size (px)
    min_area: int = 150        # reject noise / tiny specks
    max_area: int = 8000       # reject huge non-cube blue regions (e.g. wrist cam false positives)
    dominance_ratio: float = 2.0  # if >=2 components and largest < ratio*second -> ambiguous


def _to_hwc_uint8(image) -> np.ndarray:
    """Coerce an image to a (H, W, 3) uint8 numpy array.

    Accepts (C, H, W) torch tensor in [0, 1] (LeRobot frames) or (H, W, 3) arrays.
    """
    if isinstance(image, torch.Tensor):
        t = image.detach().float().cpu()
        if t.dim() == 3 and t.shape[-1] == 3:
            arr = t.numpy()
        else:
            arr = t.permute(1, 2, 0).numpy()
    else:
        arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
    return arr


def blue_threshold_mask(image, config: BlueMaskConfig | None = None) -> np.ndarray:
    """Return the raw blue binary mask (all blue pixels, morphology cleaned).

    Returns np.uint8 (H, W) with 255 = blue pixel.
    """
    config = config or BlueMaskConfig()
    rgb = _to_hwc_uint8(image)
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(config.hsv_lower), np.array(config.hsv_upper))
    if config.open_kernel > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((config.open_kernel, config.open_kernel), np.uint8))
    if config.close_kernel > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((config.close_kernel, config.close_kernel), np.uint8))
    return m


def detect_blue_components(image, config: BlueMaskConfig | None = None):
    """Return all blue connected components as a list of dicts, sorted by area desc.

    Each dict: area, bbox=(x0,y0,x1,y1), center=(cx,cy), w, h, aspect, extent,
    mask (uint8 HxW, 1=fg), color_mean (mean RGB over the component).
    """
    config = config or BlueMaskConfig()
    rgb = _to_hwc_uint8(image)
    h, w = rgb.shape[:2]
    m = blue_threshold_mask(rgb, config)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 1.0:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        mm = cv2.moments(c)
        cx = int(mm["m10"] / mm["m00"]) if mm["m00"] else x + bw // 2
        cy = int(mm["m01"] / mm["m00"]) if mm["m00"] else y + bh // 2
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c], -1, 1, thickness=cv2.FILLED)
        color_mean = tuple(float(v) for v in rgb[mask > 0].mean(axis=0)) if mask.any() else (0.0, 0.0, 0.0)
        comps.append(
            {
                "area": int(area),
                "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
                "center": (cx, cy),
                "w": int(bw),
                "h": int(bh),
                "aspect": float(bw) / float(bh) if bh else 0.0,
                "extent": float(area) / float(bw * bh) if bw * bh else 0.0,
                "mask": mask,
                "color_mean": color_mean,
            }
        )
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps


def detect_blue_mask(image, config: BlueMaskConfig | None = None):
    """Detect the blue target object and return its foreground mask.

    Args:
        image: (C, H, W) torch float [0, 1] (or (H, W, 3) uint8/float).
        config: optional :class:`BlueMaskConfig`.

    Returns:
        (mask, info) where ``mask`` is ``np.uint8 (H, W)`` (1 = target) and
        ``info`` is a dict with ``valid`` plus debug fields (area, bbox,
        pixel_count, mask_ratio, n_components, reason).
    """
    config = config or BlueMaskConfig()
    rgb = _to_hwc_uint8(image)
    h, w = rgb.shape[:2]
    comps = detect_blue_components(rgb, config)

    info = {
        "valid": False,
        "reason": None,
        "area": 0,
        "bbox": None,
        "pixel_count": 0,
        "mask_ratio": 0.0,
        "n_components": len(comps),
    }
    zero = np.zeros((h, w), np.uint8)

    if not comps:
        info["reason"] = "no_detection"
        return zero, info

    largest = comps[0]["area"]
    second = comps[1]["area"] if len(comps) > 1 else 0

    if len(comps) > 1 and largest < config.dominance_ratio * second:
        info["reason"] = "ambiguous_multiple"
        return zero, info
    if largest < config.min_area:
        info["reason"] = "too_small"
        return zero, info
    if largest > config.max_area:
        info["reason"] = "too_large"
        return zero, info

    out = comps[0]["mask"]
    info.update(
        {
            "valid": True,
            "area": int(largest),
            "bbox": comps[0]["bbox"],
            "pixel_count": int(out.sum()),
            "mask_ratio": float(out.mean()),
            "reason": None,
        }
    )
    return out, info
