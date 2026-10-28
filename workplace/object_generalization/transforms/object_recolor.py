"""Recolor a masked region from blue to green while preserving Saturation / Value.

Approach (spec section 5): convert to HSV, rotate only the Hue of the masked
pixels (blue -> green), keep S and V untouched, convert back to RGB. This keeps
the original bright/dark faces, shadows and 3-D structure instead of filling a
flat green patch.

Mask-outside pixels are bit-for-bit identical to the input (guaranteed by
overlaying only the masked pixels onto a copy of the original).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

# OpenCV HSV hue centers (H in [0, 179]).
BLUE_HUE = 120.0
GREEN_HUE = 60.0


def _to_hwc_uint8(image) -> np.ndarray:
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


def _to_torch_chw(arr: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    """Convert (H, W, 3) uint8 -> (C, H, W) tensor matching ref's dtype/device/range."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)  # (C,H,W) uint8
    t = t.to(device=ref.device, dtype=ref.dtype)
    if ref.dtype.is_floating_point:
        t = t.float() / 255.0
    return t


def recolor_to_green(
    image,
    mask,
    source_hue: float = BLUE_HUE,
    target_hue: float = GREEN_HUE,
):
    """Rotate the hue of masked pixels from blue to green.

    Args:
        image: (C, H, W) torch float in [0, 1] (LeRobot frame).
        mask: (H, W) or (1, H, W) 0/1 foreground mask.
        source_hue: hue center to map away from (default blue).
        target_hue: hue center to map to (default green).

    Returns:
        (C, H, W) torch tensor, same shape/dtype/device/range as ``image``.
    """
    original = _to_hwc_uint8(image)  # (H, W, 3) uint8
    m = np.asarray(mask).squeeze().astype(bool)

    hsv = cv2.cvtColor(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int32)
    delta = int(round(target_hue - source_hue))  # blue(120) -> green(60) => -60
    hue[m] = (hue[m] + delta) % 180
    hsv[..., 0] = hue.astype(np.uint8)

    recolored_rgb = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)

    out = original.copy()
    out[m] = recolored_rgb[m]  # only masked pixels change; outside is bit-identical

    if isinstance(image, torch.Tensor):
        return _to_torch_chw(out, image)
    return out


def outside_mask_difference(original, recolored, mask) -> float:
    """Max absolute pixel difference outside the mask (should be ~0)."""
    o = _to_hwc_uint8(original).astype(np.float32)
    r = _to_hwc_uint8(recolored).astype(np.float32)
    m = np.asarray(mask).squeeze().astype(bool)
    diff = np.abs(o - r)[~m]
    return float(diff.max()) if diff.size else 0.0
