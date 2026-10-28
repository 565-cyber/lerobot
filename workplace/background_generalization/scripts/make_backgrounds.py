#!/usr/bin/env python
"""生成背景图池，丢进 `backgrounds/` 目录。

    python scripts/make_backgrounds.py --mode solid --n 8 --out backgrounds/solid8
    python scripts/make_backgrounds.py --mode noise --n 8 --out backgrounds/noise8
    python scripts/make_backgrounds.py --mode gradient --n 8 --out backgrounds/grad8

`backgrounds/README.md` 定的接口就是「一个装着图的目录」，`_BackgroundStore`
按 `episode_index % len(images)` 取 —— 一个 episode 全程用同一张（时间一致，
不闪）。所以**模式 = 你往目录里放什么图**，代码侧不需要知道"模式"这个概念。

## 为什么先做这三种合成背景，而不是直接上真实背景

本阶段要回答的是「policy 有多依赖背景，两个相机一样吗」。这是个**因果**问题，
需要受控变量：合成背景知道**确切**改了什么，改了多少，可以逐档调（纯色 →
噪声 → 渐变），于是"背景变了多少"是个可读的量。真实背景池（RoboSaGA 原意）
更适合**训练**，不适合**归因** —— 两张真实背景之间的差异是无界的、没法分档。

两条路不冲突：诊断先跑，真实池接 `--mode` 之外，直接把图丢进目录就行。

## 尺寸和内容

存的是 **64x64** 的小图，`_BackgroundStore.get` 会双线性放大到相机分辨率。
放大噪点会得到大块的平滑斑 —— 想要逐像素噪声就不要放大，见 `--size`。
存小图是为了让目录轻，也是因为放大本来就是 `_BackgroundStore` 的行为。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MODES = ("solid", "noise", "gradient")


def _solid(n: int, rng: np.random.Generator, side: int) -> list[np.ndarray]:
    """`_solid_color` 里那 8 个灰阶/土色，外加几个明显有色相的。

    加有色相的是因为**背景色的色相**是这里唯一能独立于"纹理"控制的变量，
    而 up 和 front 的差异如果要归因，最可能就是归在色相和纹理上。
    """
    base = [(0, 0, 0), (51, 51, 51), (102, 102, 102), (179, 153, 128),
            (77, 51, 26), (153, 179, 153), (204, 204, 204), (26, 38, 51),
            (200, 60, 60), (60, 200, 90), (60, 90, 200), (220, 200, 60)]
    out = []
    for i in range(n):
        c = base[i % len(base)]
        out.append(np.full((side, side, 3), c, np.uint8))
    return out


def _noise(n: int, rng: np.random.Generator, side: int) -> list[np.ndarray]:
    """均匀噪声，逐通道独立。放大之后是大块斑，不是逐像素噪声 —— 见模块文档。"""
    return [rng.integers(0, 256, (side, side, 3), dtype=np.uint8) for _ in range(n)]


def _gradient(n: int, rng: np.random.Generator, side: int) -> list[np.ndarray]:
    """线性渐变，方向/端点随机。比纯色多一个"空间结构"，但结构简单到可描述。"""
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    yy /= max(side - 1, 1); xx /= max(side - 1, 1)
    out = []
    for _ in range(n):
        a = rng.integers(0, 256, 3).astype(np.float32)
        b = rng.integers(0, 256, 3).astype(np.float32)
        t = rng.random()
        # 随机方向：t 在"横"和"纵"之间的插值权重
        w = t * xx + (1 - t) * yy
        img = a[None, None, :] * (1 - w[..., None]) + b[None, None, :] * w[..., None]
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=64, help="存下来的边长（会被放大到相机分辨率）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in out.iterdir() if p.suffix.lower() in (".jpg", ".png", ".bmp", ".webp"))
    if existing:
        raise SystemExit(
            f"{out} 里已经有 {len(existing)} 张图了。`_BackgroundStore` 取的是整个目录，"
            f"混着放会让「这次用的是哪一档」说不清。换个目录，或先清空。")

    rng = np.random.default_rng(args.seed)
    imgs = {"solid": _solid, "noise": _noise, "gradient": _gradient}[args.mode](
        args.n, rng, args.size)

    import cv2
    for i, im in enumerate(imgs):
        p = out / f"{args.mode}_{i:03d}.png"
        # 存 PNG：JPEG 会给纯色和渐变引入块效应，那本身就是一种"纹理"，
        # 会把纯色这一档污染成"有一点纹理"，正好是要排除的变量。
        cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    print(f"[save] {out}  {len(imgs)} 张 {args.size}x{args.size} {args.mode}")


if __name__ == "__main__":
    main()
