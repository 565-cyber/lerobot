"""Distractor cutout sources for the Stage 2B background/clutter augmentation.

A *cutout* is a self-contained RGBA-ish sprite: an RGB patch, a soft alpha, and
the contact shadow it casts. Nothing here knows about LeRobot, the dataset, or
the plate -- a provider only draws one object -- so a real photographed object
can be dropped in later by implementing the same protocol
(:class:`CutoutProvider`) without touching the compositing or verification code.

See :mod:`.procedural` for the default, procedurally generated implementation.
"""

from .procedural import (  # noqa: F401
    Cutout,
    CutoutProvider,
    ProceduralConfig,
    ProceduralCutoutProvider,
)

__all__ = ["Cutout", "CutoutProvider", "ProceduralConfig", "ProceduralCutoutProvider"]
