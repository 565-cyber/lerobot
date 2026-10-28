"""Background Replacement (pure function — no dataset / policy / training logic).

Stage 2 of the background-generalization study: replace only the task-irrelevant
background of a camera image while keeping the foreground (robot arm, gripper,
target object) intact.

Core formula::

    augmented = mask * image + (1 - mask) * background

The function is self-contained: it accepts plain tensors/arrays and does not
import LeRobot, so it can be unit-tested in isolation.

Conventions
-----------
- ``image``     : (C, H, W) float, assumed already in [0, 1] (LeRobot dataset
                  invariant — frames are decoded to float32 [0, 1]).
- ``mask``      : (H, W) or (1, H, W), 1 = keep (foreground), 0 = replace.
                  uint8 0/255 or 0/1 are normalized automatically.
- ``background``: (3, H, W) | (H, W, 3) | (H, W), float or uint8.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _as_float(x) -> Tensor:
    if isinstance(x, Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x)).float()


def _to_01(x) -> Tensor:
    """Normalize a numeric tensor/array to float32 in [0, 1].

    Handles uint8 in [0, 255] (divides by 255) as well as values already in [0, 1].
    """
    t = _as_float(x)
    if t.numel() > 0 and float(t.max()) > 1.5:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def _resize2d(t: Tensor, h: int, w: int, mode: str = "nearest") -> Tensor:
    """Resize a (H, W) tensor to (h, w)."""
    if t.shape[-2] == h and t.shape[-1] == w:
        return t
    align = None if mode == "nearest" else False
    return F.interpolate(t[None, None], size=(h, w), mode=mode, align_corners=align)[0, 0]


def _resize3d(t: Tensor, h: int, w: int, mode: str = "bilinear") -> Tensor:
    """Resize a (C, H, W) tensor to (C, h, w)."""
    if t.shape[-2] == h and t.shape[-1] == w:
        return t
    align = None if mode == "nearest" else False
    return F.interpolate(t[None], size=(h, w), mode=mode, align_corners=align)[0]


def _rgb(t: Tensor, c: int, h: int, w: int) -> Tensor:
    """Coerce a background tensor to (c, h, w) float in [0, 1]."""
    if t.dim() == 2:  # (H, W) grayscale
        return _resize2d(t, h, w)[None].expand(c, -1, -1).clone()
    if t.dim() == 3:
        if t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = _resize3d(t, h, w)  # (C, H, W)
        else:
            t = _resize3d(t.permute(2, 0, 1), h, w)  # (H, W, C) -> (C, H, W)
        if t.shape[0] == c:
            return t
        if t.shape[0] == 1:
            return t.expand(c, -1, -1).clone()
        return t[:c]
    raise ValueError(f"unexpected background shape {tuple(t.shape)}")


def _mask2d(m: Tensor, h: int, w: int) -> Tensor:
    """Coerce a mask tensor to a (1, h, w) float in [0, 1]."""
    m = _to_01(m)
    if m.dim() == 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] in (1, 3, 4) and m.shape[0] not in (1, 3, 4):
            m = m[..., 0]  # (H, W, C)
        else:
            m = m[0]  # (C, H, W) with C > 1 -> first channel
    return _resize2d(m, h, w)[None]


def apply_background_replacement(
    image: Tensor | np.ndarray,
    mask: Tensor | np.ndarray | None = None,
    background: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Replace the background of ``image`` outside the foreground ``mask``.

    Args:
        image: (C, H, W) float in [0, 1].
        mask: (H, W) or (1, H, W) foreground mask (1 = keep, 0 = replace).
              If None, the image is returned unchanged.
        background: (3, H, W) | (H, W, 3) | (H, W). If None, black is used.

    Returns:
        augmented image (C, H, W) float32 in [0, 1].
    """
    img = _to_01(image)
    c, h, w = img.shape
    if mask is None:
        return img
    m = _mask2d(mask, h, w)
    bg = torch.zeros_like(img) if background is None else _rgb(_to_01(background), c, h, w)
    return m * img + (1.0 - m) * bg


class BackgroundReplacement:
    """Thin callable wrapper around :func:`apply_background_replacement`."""

    def __call__(
        self,
        image: Tensor,
        mask: Tensor | None = None,
        background: Tensor | None = None,
    ) -> Tensor:
        return apply_background_replacement(image, mask, background)
