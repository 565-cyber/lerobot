"""Minimal check script for BackgroundAugmentedDataset.

Verifies, against the real ``newdata1`` dataset:

  1. real shapes / dtypes of the wrapped ``dataset[0]`` (camera image, action, state),
  2. action and state are byte-for-byte unchanged vs the base dataset,
  3. a DataLoader batch collates correctly,
  4. (optional) saves a 4-panel figure: original / mask / background / augmented.

Run from the repo root (conda env ``lerobot``)::

    python -m background_generalization.scripts.inspect_bg_dataset

Note on ``--root``: this local LeRobotDataset uses ``root`` directly as the dataset
directory (see ``lerobot_dataset.py``: ``self.root = Path(root)``), so it must point
at ``dataset/newdata1`` — not the parent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# allow `python scripts/inspect_bg_dataset.py` to import the local package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from background_generalization.datasets.background_dataset import BackgroundAugmentedDataset
from background_generalization.transforms.background_replacement import apply_background_replacement


def _synthetic_mask(h: int, w: int) -> np.ndarray:
    """Center-box placeholder mask, display only.

    There is no producer for real ones any more: Stage 2B pastes distractors
    under their own alpha instead of replacing a masked-out background, so this
    wrapper is only exercised by this smoke test.
    """
    m = np.zeros((h, w), dtype=np.uint8)
    mh, mw = int(h * 0.15), int(w * 0.15)
    m[mh : h - mh, mw : w - mw] = 1
    return m


def _to_hwc(t) -> np.ndarray:
    t = torch.as_tensor(t).detach().float()
    if t.dim() == 3:
        t = t.permute(1, 2, 0)
    return t.clamp(0, 1).cpu().numpy()


def _save_figure(original, mask, background, augmented, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"skip figure (matplotlib unavailable): {e}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(_to_hwc(original))
    axes[0].set_title("original")
    axes[1].imshow(np.asarray(mask).squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("mask")
    axes[2].imshow(_to_hwc(background))
    axes[2].set_title("background")
    axes[3].imshow(_to_hwc(augmented))
    axes[3].set_title("augmented")
    for ax in axes:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="newdata1")
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata1")
    ap.add_argument("--mask-root", default=None)
    ap.add_argument("--background-root", default=None)
    ap.add_argument("--probability", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--camera", default=None, help="camera key; default = first camera")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-dir", default="/home/zjq/lerobot/workplace/background_generalization/outputs")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    base_ds = LeRobotDataset(args.repo_id, root=args.root)
    wrapped = BackgroundAugmentedDataset(
        base_ds,
        mask_root=args.mask_root,
        background_root=args.background_root,
        probability=args.probability,
        seed=args.seed,
    )

    idx = args.index
    base_item = base_ds[idx]
    item = wrapped[idx]
    cam = args.camera or wrapped.camera_keys[0]

    # ---- Level 1: real shapes/dtypes ----
    print("camera_keys:", wrapped.camera_keys)
    print(f"[{cam}] dtype={item[cam].dtype} shape={tuple(item[cam].shape)} "
          f"min={item[cam].min().item():.3f} max={item[cam].max().item():.3f}")
    print("action:", tuple(item["action"].shape), item["action"].dtype)
    print("observation.state:", tuple(item["observation.state"].shape), item["observation.state"].dtype)
    print("episode_index:", item["episode_index"].item(), "frame_index:", item["frame_index"].item())

    # ---- Level 2: state/action unchanged ----
    assert torch.equal(base_item["action"], item["action"]), "action changed!"
    assert torch.equal(base_item["observation.state"], item["observation.state"]), "state changed!"
    print("OK: action/state byte-for-byte unchanged")

    # ---- Level 3: DataLoader batch ----
    dl = DataLoader(wrapped, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(dl))
    print(f"batch[{cam}]:", tuple(batch[cam].shape), batch[cam].dtype)
    print("batch[action]:", tuple(batch["action"].shape), "batch[observation.state]:",
          tuple(batch["observation.state"].shape))
    print("OK: DataLoader batch collates")

    # ---- Level 4: visual (4-panel) ----
    ep = int(item["episode_index"].item())
    frame = int(item["frame_index"].item())
    h, w = item[cam].shape[-2], item[cam].shape[-1]
    mask = wrapped._get_mask(cam, ep, frame)
    if mask is None:
        mask = _synthetic_mask(h, w)
        print("mask: synthetic (no --mask-root); no mask builder exists for this wrapper")
    else:
        print(f"mask: real ({mask.shape})")
    background = wrapped._backgrounds.get(ep, h, w)
    augmented = apply_background_replacement(base_item[cam], mask, background)
    out_path = Path(args.out_dir) / f"{args.repo_id}_{cam.replace('.', '_')}_idx{idx}.png"
    _save_figure(base_item[cam], mask, background, augmented, out_path)


if __name__ == "__main__":
    main()
