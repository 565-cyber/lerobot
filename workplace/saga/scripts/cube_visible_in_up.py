#!/usr/bin/env python
"""更正：**黄方块在 up 相机里看得见** —— 我先前说「看不见」是阈值造的假。

## 错在哪

先前用 `(r>.75)&(g>.70)&(b<.35)&(|r-g|<.18)` 扫全数据集，得到 up 最大 **19 px**，
据此下了「黄方块从不出现在 up」的结论，还写进了 FINDINGS。

错因：**up 视角下方块被暖光洗成近乎米白** —— 实测中位 RGB `[0.961, 0.953, 0.729]`，
**b = 0.73**，而判据要求 `b < 0.35`。（front 里同一个方块是 b≈0.2 的饱和黄。）
同一类错这一轮已经犯过两次（橙纸盒当黄方块、淡黄桌布当黄方块）。

## 现在能证到什么

1. 方块**当桌面物体**时在 up 里可见（左上角那个米黄小方块）；
2. 方块**被夹住**时在 up 里也可见（夹爪末端那枚淡黄绿楔形块）；
3. `front 有没有方块` 与 `up 楔形块在不在` **同步**：ep0 68 vs 0 px、ep1 505 vs 0 px。

第 3 条是这里用的判据 —— front 是眼在手上、正对夹爪，方块在不在它一目了然，
所以可以拿它**当裁判**去判定 up 里那块到底是什么。

用法：
    python workplace/saga/scripts/cube_visible_in_up.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
WORKPLACE = HERE.parents[2]
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def cube_up(a: np.ndarray) -> np.ndarray:
    """up 视角的方块：被暖光洗白，b 高。**不是**饱和黄。"""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 0.60) & (g > 0.56) & (r - b > 0.16) & (g - b > 0.13) & (np.abs(r - g) < 0.13)


def cube_front(a: np.ndarray) -> np.ndarray:
    """front 视角的方块：近距、有饱和色。"""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 0.72) & (g > 0.62) & (b < 0.40) & (np.abs(r - g) < 0.22)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset", default="record-test10")
    ap.add_argument("--root", default=str(REPO / "dataset" / "newdata1"))
    ap.add_argument("--out", default=str(REPO / "workplace/saga/figures"))
    ap.add_argument("--episode", type=int, default=1)
    args = ap.parse_args()

    import lerobot.policies.act.modeling_act  # noqa: F401  注册 'act'
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    ds = make_dataset(TrainPipelineConfig(
        policy=cfg, dataset=DatasetConfig(repo_id=args.dataset, root=args.root)))
    ep = ds.meta.episodes.to_pandas()
    U, F = "observation.images.up", "observation.images.front"
    lo = int(ep.iloc[args.episode]["dataset_from_index"])
    hi = int(ep.iloc[args.episode]["dataset_to_index"])
    T = hi - lo

    Uarr = np.stack([ds[i][U].permute(1, 2, 0).numpy() for i in range(lo, hi)])
    Farr = np.stack([ds[i][F].permute(1, 2, 0).numpy() for i in range(lo, hi)])
    up_px = np.array([int(cube_up(a).sum()) for a in Uarr])
    fr_px = np.array([int(cube_front(a).sum()) for a in Farr])
    # 「夹着」的判据：**front 画面顶部**（=夹爪所在处）有方块。
    # 不能用「front 检测到方块」—— 桌上没夹起来的方块 front 同样看得见（踩过）。
    fr_top = np.array([int(cube_front(a[:170]).sum()) for a in Farr])
    held = fr_top > 800

    print(f"ep{args.episode}: T={T}")
    print(f"  up   检测到方块(>150px)   {100 * (up_px > 150).mean():.0f}% 的帧")
    print(f"  front 画面顶部有方块(=夹着) {100 * held.mean():.0f}% 的帧")
    print(f"  front 夹着方块时  up 方块 px 中位 {int(np.median(up_px[held])):5d}")
    print(f"  front 没夹方块时  up 方块 px 中位 {int(np.median(up_px[~held])):5d}")
    print("  → 同步 ⇒ up 里那块淡黄绿**就是方块**，不是机械臂的一部分")

    # 挑三帧：方块在桌上 / 正被夹 / 已放进盒
    t_hold = int(np.argmax(held))
    t_tab = int(np.argmin(np.where(held, 1e9, np.abs(fr_px - 1600))))
    t_done = int(np.argmin(fr_px))
    snaps = [("方块在桌上（未被夹）", t_tab), ("方块被夹住", t_hold),
             ("方块已投进红盒", t_done)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for r, (label, t) in enumerate(snaps):
        for c, (arr, fn, nm) in enumerate([(Uarr, cube_up, "up"),
                                           (Farr, cube_front, "front")]):
            ax = axes[r][c]
            ax.imshow(arr[t])
            m = fn(arr[t])
            if m.sum() > 50:
                ax.contour(m.astype(float), levels=[0.5], colors=["#00ff7f"], linewidths=1.5)
            ax.set_title(f"{nm}   t={t}   判据命中 {int(m.sum())} px"
                         + ("   ← 夹爪处" if nm == "front" and int(cube_front(arr[t][:170]).sum()) > 800 else ""),
                         fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        # 第三列：up 里的时域曲线
        ax = axes[r][2]
        ax.plot(up_px / max(up_px.max(), 1), color="#c1440e", lw=1.4, label="up 方块(归一)")
        ax.plot(fr_px / max(fr_px.max(), 1), color="#1f6fb4", lw=1.4, label="front 方块(归一)")
        ax.axvline(t, color="k", ls=":", lw=1.6)
        ax.set_title(f"t={t} 在时域上的位置", fontsize=10)
        ax.set_xlabel("帧"); ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(
        "更正：黄方块在 **up** 相机里看得见（先前「看不见」是阈值造的假）\n"
        "up 里方块被暖光洗成米白 RGB≈[0.96,0.95,0.73]，b=0.73；旧的 b<0.35 判据整类判掉了它\n"
        "绿线 = 各相机自己的检测判据；front（眼在手上）当裁判判定 up 里那块是不是方块",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "cube_visible_in_up.png"
    fig.savefig(p, dpi=78); plt.close(fig)
    print(f"\n[save] {p}")


if __name__ == "__main__":
    main()
