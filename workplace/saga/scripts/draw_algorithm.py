#!/usr/bin/env python
"""画 SaGA 的**论文式算法图** —— 只有算法逻辑，没有工程细节。

    python workplace/saga/scripts/draw_algorithm.py --out workplace/saga

面板 (a) 是按 `zhuang25b.pdf` 里 Algorithm 1 的体例写的伪代码，
符号沿用原文（α、β、λ、M、M_v、I_β、I_α、S）；
面板 (b) 是同一套逻辑的流程示意。

**刻意不画进图里的**（那些属于实现，不属于算法）：
文件路径、preprocessor / autocast 这类框架概念、步时开销、batch_size 下限、
缓冲区分辨率等折衷。要那些看 `saga_framework.png`。

伪代码与原文的两处差异**是照参考实现改的**，不是笔误：
原文第 14 行写 `min(Fullgrad(...), λ)`（在**存**的时候裁），
源码是在**取**的时候 `torch.clip(smaps, 0, λ)`，**之后**再 `smaps ** 2`。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# 论文观感：衬线体 + Computer Modern 数学
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK = "#1a1a1a"
GREY = "#6b6b6b"
C_UPD = "#2b4c7e"   # 更新支路
C_AUG = "#a33a3a"   # 增强支路
C_BUF = "#2e6b4f"   # 缓冲区


def cbox(ax, x, y, w, h, text, edge, fc="white", fs=9.5, lw=1.1, txt=INK):
    """直角矩形（论文里不用圆角）。"""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                           linewidth=lw, facecolor=fc, edgecolor=edge, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=txt,
            zorder=3, linespacing=1.5)


def arr(ax, p, q, color=INK, lw=1.0, dashed=False, rad=0.0, style="-|>", ms=10):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color,
        zorder=1, shrinkA=1.5, shrinkB=1.5,
        linestyle=(0, (4, 3)) if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


# --------------------------------------------------------------------------- #
#                          面板 (a)：Algorithm 1                              #
# --------------------------------------------------------------------------- #
# (缩进层级, 代码, 行尾注释 or None)   "---" 表示一条分隔线
ALGO = [
    (0, r"$\mathbf{Input:}$ batch $M$ of $m$ trajectories;  visual encoders $\{v\}$;  OOD image pool $X_O$", None),
    (2, r"$\alpha$ augmentation ratio;  $\beta$ buffer-update ratio;  $\lambda$ saliency clipping factor;  $T_{\mathrm{warm}}$ warm-up epochs", None),
    (0, r"$\mathbf{Output:}$ augmented batch $M^{*}$", None),
    ("rule",),
    (0, r"Initialize buffer $S$ with all-one saliency maps", None),
    (0, r"$\mathbf{for}$ each training iteration $t$ $\mathbf{do}$", None),
    (1, r"$\mathbf{if}$ $t < T_{\mathrm{warm}}$ $\mathbf{then}$ return $M$", "no augmentation"),
    (1, r"$\mathbf{for}$ each visual encoder $v$ in the policy $\mathbf{do}$", "one lane per camera"),
    (2, r"$M_v \leftarrow$ images processed by encoder $v$", None),
    (2, "", None),
    (2, r"(a) buffer update", None),
    (2, r"sort batch indices $I$ by buffer update count $c_i$", None),
    (2, r"$I_\beta \leftarrow$ the $\beta|I|$ least recently updated indices", None),
    (2, r"$g \leftarrow \mathrm{FullGrad}\left(v(M_v[I_\beta])\right)$", "encoder-level saliency"),
    (2, r"$\hat{g} \leftarrow \left(g - \min g\right) / \left(\max g - \min g\right)$", "per-sample min-max, to $[0,1]$"),
    (2, r"$S.\mathrm{set}\left(\hat{g},\ \mathrm{gid}(I_\beta)\right)$ ;  $c_{I_\beta} \leftarrow c_{I_\beta} + 1$", None),
    (2, "", None),
    (2, r"(b) augmentation", None),
    (2, r"$I_\alpha \leftarrow \alpha|I|$ indices sampled uniformly", None),
    (2, r"$s \leftarrow \mathrm{clip}\left(S.\mathrm{get}\left(\mathrm{gid}(I_\alpha)\right),\ 0,\ \lambda\right)^2$", "retrieve, clip, then square"),
    (2, r"$M_v[I_\alpha] \leftarrow s \odot M_v[I_\alpha] + (1-s) \odot x_O$,   $x_O \sim X_O$", None),
    (1, r"$\mathbf{end}$ $\mathbf{for}$", None),
    (1, r"Optimise the policy on the augmented batch $M^{*}$", None),
    (0, r"$\mathbf{end}$ $\mathbf{for}$", None),
]


def draw_algo(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 99.2, "(a)", fontsize=13, fontweight="bold", color=INK, va="top")

    # 外框 + 标题栏（论文 Algorithm 的体例：标题下面一条横线）
    ax.add_patch(Rectangle((0.5, 1.0), 99.0, 94.5, linewidth=1.2,
                           facecolor="#fcfcfa", edgecolor=INK, zorder=0))
    ax.text(3.0, 92.6, "Algorithm 1", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(22.0, 92.6, "SaGA: Saliency-Guided Augmentation",
            fontsize=11.5, fontstyle="italic", color=INK)
    ax.plot([3.0, 97.0], [90.6, 90.6], color=INK, lw=1.0, zorder=1)

    y = 87.0
    dy = 3.72
    for item in ALGO:
        if item[0] == "rule":
            ax.plot([2.0, 98.0], [y + 0.6, y + 0.6], color="#bbbbbb", lw=0.8,
                    ls=(0, (3, 3)), zorder=1)
            y -= dy * 0.85
            continue
        ind, code, note = item
        if code == "":
            y -= dy * 0.55
            continue
        if code in ("(a) buffer update", "(b) augmentation"):
            ax.text(4.0 + ind * 2.6, y, code, fontsize=9.2, color=GREY,
                    style="italic", va="center")
            y -= dy * 0.95
            continue
        ax.text(4.0 + ind * 2.6, y, code, fontsize=9.6, color=INK, va="center")
        if note:
            ax.text(99.0, y, note, fontsize=8.3, color=GREY, style="italic",
                    ha="right", va="center")
        y -= dy

    # 行号在左边槽里单独画，保证右对齐
    y = 87.0
    ln = 1
    for item in ALGO:
        if item[0] == "rule":
            y -= dy * 0.85
            continue
        code = item[1]
        if code == "":
            y -= dy * 0.55
            continue
        if code in ("(a) buffer update", "(b) augmentation"):
            y -= dy * 0.95
            continue
        ax.text(3.0, y, f"{ln:2d}", fontsize=8.6, color=GREY, ha="right",
                va="center", family="DejaVu Sans Mono")
        ln += 1
        y -= dy


# --------------------------------------------------------------------------- #
#                     面板 (b)：同一套逻辑的流程示意                          #
# --------------------------------------------------------------------------- #
def draw_flow(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 99.4, "(b)", fontsize=13, fontweight="bold", color=INK, va="top")

    LW, RW = 27.0, 73.0          # 左列（更新） / 右列（增强 + 缓冲区）
    BW, BH = 34.0, 6.4           # 列内方块尺寸

    # ---- 输入，分叉到两列 ----
    cbox(ax, 50, 95.0, 56, 5.6, r"$M_v$   images of camera $v$", INK, "#f7f7f7", 10)
    arr(ax, (50, 92.2), (50, 88.5), style="-[")
    ax.plot([LW, RW], [88.5, 88.5], color=INK, lw=1.0, zorder=1)
    arr(ax, (LW, 88.5), (LW, 84.9))
    arr(ax, (RW, 88.5), (RW, 66.9))
    # 支路标签放在竖箭头两侧，避免被箭头划穿
    # 不加粗：mathtext 不继承 fontweight，加粗会让两截字形不一致
    ax.text(LW - 2.0, 90.6, r"(a) buffer update   $\beta|I|$", fontsize=9.0,
            color=C_UPD, ha="right", va="center")
    ax.text(RW + 2.0, 90.6, r"(b) augmentation   $\alpha|I|$", fontsize=9.0,
            color=C_AUG, ha="left", va="center")

    # ---- 左列：更新支路 ----
    cbox(ax, LW, 81.5, BW, BH, r"$v(\cdot)$   visual encoder", C_UPD, "#f4f7fb", 9.0)
    cbox(ax, LW, 72.5, BW, BH, r"$\mathrm{FullGrad}(\cdot)$", C_UPD, "#f4f7fb", 9.0)
    cbox(ax, LW, 63.5, BW, BH, r"min–max  $\rightarrow$  $\hat{g} \in [0,1]$", C_UPD, "#f4f7fb", 8.6)
    arr(ax, (LW, 78.3), (LW, 75.7), color=C_UPD)
    arr(ax, (LW, 69.3), (LW, 66.7), color=C_UPD)

    # ---- 缓冲区（左右两列在此汇合）----
    cbox(ax, RW, 63.5, BW, BH, r"buffer  $S_v$", C_BUF, "#f2f8f4", 9.6)
    arr(ax, (LW + BW / 2, 63.5), (RW - BW / 2, 63.5), color=C_BUF,
        style="-|>", lw=1.2)
    ax.text((LW + RW) / 2, 65.6, r"$S.\mathrm{set}$", fontsize=8.2, color=C_BUF,
            ha="center", va="bottom")

    # ---- 右列：增强支路 ----
    cbox(ax, RW, 54.5, BW, BH, r"retrieve  $S.\mathrm{get}(\cdot)$", C_AUG, "#fbf4f4", 9.0)
    cbox(ax, RW, 45.5, BW, BH, r"$\mathrm{clip}(\cdot,\ 0,\ \lambda)$", C_AUG, "#fbf4f4", 9.0)
    cbox(ax, RW, 36.5, BW, BH, r"$(\cdot)^2$   weight  $s$", C_AUG, "#fbf4f4", 9.0)
    arr(ax, (RW, 60.3), (RW, 57.7), color=C_BUF)
    arr(ax, (RW, 51.3), (RW, 48.7), color=C_AUG)
    arr(ax, (RW, 42.3), (RW, 39.7), color=C_AUG)

    # ---- 混合 ----
    cbox(ax, 51, 22.0, 56, 7.6,
         r"$\tilde{M}_v[I_\alpha] \;=\; s \odot M_v[I_\alpha] \;+\; (1-s) \odot x_O$",
         INK, "#f7f7f7", 10.5)
    cbox(ax, 9, 22.0, 16, 6.2, r"$x_O \sim X_O$", INK, "#f7f7f7", 9.2)
    arr(ax, (RW, 33.3), (RW, 25.8), color=C_AUG)
    ax.text(RW + 1.6, 29.5, r"$s$", fontsize=9.2, color=C_AUG, ha="left", va="center")
    arr(ax, (17.0, 22.0), (23.0, 22.0), color=INK)
    ax.text(20.0, 25.4, r"$1-s$", fontsize=8.2, color=GREY, ha="center", va="bottom")

    # ---- 输出 ----
    cbox(ax, 50, 10.0, 36, 5.8, r"augmented  $M^{*}_v$", INK, "#f7f7f7", 9.8)
    arr(ax, (50, 18.2), (50, 12.9))

    ax.text(50, 2.6,
            r"repeated for every visual encoder $v$:  each camera keeps its own $S_v$",
            fontsize=8.8, color=GREY, ha="center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workplace/saga")
    args = ap.parse_args()

    fig = plt.figure(figsize=(14.0, 8.6))
    axA = fig.add_axes([0.015, 0.02, 0.52, 0.96]); axA.axis("off")
    axB = fig.add_axes([0.565, 0.02, 0.425, 0.96]); axB.axis("off")
    draw_algo(axA)
    draw_flow(axB)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        p = out / f"saga_algorithm.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"[save] {p}")


if __name__ == "__main__":
    main()
