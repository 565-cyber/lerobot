#!/usr/bin/env python
"""画 SaGA 的框架图（给人看的，不参与训练）。

    python workplace/saga/scripts/draw_framework.py --out workplace/saga

内容全部对应 `workplace/saga/` 里的实现，不是论文的复述：
每一条都能指到具体文件与行（关键几处标在图的注释里）。

刻意画出来的两件事：
1. **逐相机**：Algorithm 1 对每个视觉模态各跑一次，每个相机一份 extractor + 一份缓冲区。
2. **两处注入讲究**：必须在 `preprocessor` 之后（ImageNet 归一化空间）、
   必须在 autocast 之外（fp32 算显著度）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配色：浅填充 + 深描边，低饱和，打印/投影都能看清
C = {
    "data":   ("#e8eef5", "#4a6fa5"),
    "prep":   ("#e4eef0", "#3f7d8c"),
    "gate":   ("#f5efe0", "#a8853a"),
    "update": ("#e3ecf7", "#3d6fb4"),
    "buffer": ("#ece6f5", "#6b4fa8"),
    "aug":    ("#e4f2e6", "#3f8a4f"),
    "ood":    ("#f7ece2", "#b06a2c"),
    "policy": ("#eeeeee", "#555555"),
    "note":   ("#fbeaea", "#a33a3a"),
    "frame":  ("#fffdf6", "#c9a227"),
}


def box(ax, x, y, w, h, text, kind, fs=9, bold=False, align="center"):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.4, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.55)


def arrow(ax, p, q, text=None, color="#444444", style="-|>", rad=0.0, fs=8,
          lx=0.0, ly=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=13, linewidth=1.3,
        color=color, zorder=1, shrinkA=1, shrinkB=1,
        connectionstyle=f"arc3,rad={rad}",
        linestyle="--" if dashed else "-"))
    if text:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, text,
                ha="center", va="center", fontsize=fs, color=color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workplace/saga")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(17.5, 12.2))
    ax.set_xlim(0, 100); ax.set_ylim(-6.5, 100)
    ax.axis("off")

    # ================= 1. 数据与预处理 =================
    ax.text(2, 97.4, "① 数据与预处理", fontsize=12.5, fontweight="bold", color="#333333")
    box(ax, 17, 92.6, 30, 6.2,
        "LeRobotDataset  record-test10\n21 998 帧 · 50 集 · 相机 up + front", "data", 9.5)
    box(ax, 50, 92.6, 26, 6.2,
        "DataLoader\nbatch = 16 · 4 workers", "data", 9.5)
    box(ax, 84, 92.6, 30, 6.2,
        "preprocessor\nrename → to_batch → device → normalizer", "prep", 9.5)
    arrow(ax, (32, 92.6), (37, 92.6))
    arrow(ax, (63, 92.6), (69, 92.6))
    ax.text(84, 88.1, "ImageNet 归一化（0.485/0.229…）", ha="center", fontsize=8.8,
            color="#3f7d8c", style="italic")

    # 注入点说明
    ax.annotate("", xy=(84, 84.0), xytext=(84, 89.2),
                arrowprops=dict(arrowstyle="-|>", color="#a33a3a", lw=2.0))
    ax.text(85.5, 86.6, "唯一注入点", fontsize=9.5, color="#a33a3a", fontweight="bold")
    ax.text(85.5, 84.2,
            "必须在 preprocessor 之后（图像已归一化）\n"
            "必须在 autocast 之外（fp32 算显著度）",
            fontsize=8.4, color="#a33a3a", va="center", linespacing=1.5)

    # ================= 2. SaGA 大框 =================
    ax.add_patch(FancyBboxPatch(
        (4.6, 26.0), 93.2, 55.5,
        boxstyle="round,pad=0.6,rounding_size=1.6",
        linewidth=2.2, facecolor=C["frame"][0], edgecolor=C["frame"][1], zorder=0))
    ax.text(3.4, 79.6, "② RoboSaGA  ——  workplace/saga/", fontsize=12.5,
            fontweight="bold", color="#8a6d1a")
    ax.text(3.4, 76.9, "seed 的 batch 进来，返回增强后的 batch（只换相机图像键）",
            fontsize=8.8, color="#8a6d1a")

    # 闸门
    box(ax, 50, 74.2, 62, 4.6,
        "闸门：step < warmup(13 750 步 = 10 epoch)  或  α = 0  →  原样返回，逐位不变", "gate", 9.2)
    arrow(ax, (84, 83.8), (50, 76.7), rad=0.0, color="#a33a3a")

    # ---- 两条相机泳道 ----
    for i, (cam, cx) in enumerate([("up", 26.5), ("front", 73.5)]):
        ax.text(cx, 70.4, f"相机 {cam}", fontsize=11, fontweight="bold",
                ha="center", color="#333333")

        # 更新支路
        box(ax, cx, 65.0, 40, 7.0,
            "【更新支路】β = 0.1 → 每步 1 张\n"
            "按 update_counter 选「最久未更新」的帧 id", "update", 8.8)
        box(ax, cx, 55.1, 40, 8.0,
            "FullGrad（在 ACT backbone 上算）\n"
            "Σ |∂f/∂x · x|  +  Σ |∂f/∂z ⊙ b|      逐样本 min-max → [0,1]\n"
            "即论文的 FullGrad；第二项是 grad⊙bias，不是 grad⊙activation", "update", 8.2)
        arrow(ax, (cx, 61.5), (cx, 59.3))

        # 缓冲区
        box(ax, cx, 45.2, 40, 6.6,
            "显著度缓冲区（本相机专属）\n"
            "21 999 × 120 × 160 uint8 · 初值 255(=1.0) · 全局帧 id 索引", "buffer", 8.4)
        arrow(ax, (cx, 51.0), (cx, 48.6), text="buffer.set", fs=7.8, lx=8.6)

        # 增强支路
        box(ax, cx, 34.8, 40, 8.6,
            "【增强支路】α = 0.5 → 每步前 8 帧\n"
            "取回 s → clip(s, 0, λ=0.8) → s ← s²（源码有、论文没写）\n"
            "x ← s ⊙ x  +  (1 − s) ⊙ x_OOD          权重上限 λ² = 0.64", "aug", 8.2)
        arrow(ax, (cx, 41.8), (cx, 39.2), text="buffer.get", fs=7.8, lx=8.6)

    # 相机间的关系说明
    ax.text(50, 24.5,
            "两个相机共用一组 backbone 权重，但显著度、缓冲区、混合**各算各的** —— "
            "这正是论文 Algorithm 1 的结构（参考实现 for obs_key in visual_modalities）",
            ha="center", fontsize=8.8, color="#666666", style="italic")

    # OOD 背景源
    box(ax, 50, 68.0, 22, 5.4,
        "OOD 背景池\nsolid8 纯色 × 8（ImageNet 归一化）", "ood", 8.4)
    arrow(ax, (50, 70.7), (46.5, 34.8), rad=0.22, color="#b06a2c", dashed=True)
    arrow(ax, (50, 70.7), (53.5, 34.8), rad=-0.22, color="#b06a2c", dashed=True)
    ax.text(50, 30.6, "x_OOD", fontsize=8, color="#b06a2c", ha="center")

    # ================= 3. 策略与优化 =================
    box(ax, 26, 15.0, 44, 6.8,
        "ACT  policy.forward\n"
        "backbone → 15×20 特征 → transformer → 动作 chunk 100×6", "policy", 8.8)
    box(ax, 74, 15.0, 44, 6.8,
        "loss → backward → optimizer\n"
        "显著度 .detach()：梯度**不**回传策略", "policy", 8.8)
    arrow(ax, (26, 30.5), (26, 18.5))
    arrow(ax, (48, 15.0), (52, 15.0))

    # 直通说明
    ax.text(50, 21.0, "非图像键（state / action / timestamp / index）逐位不变，直接透传",
            ha="center", fontsize=8.8, color="#666666")
    # 旁路：非图像键 / warmup 期间整块 SaGA 都不经过，从预处理直达策略
    ax.plot([19.0, 2.4], [72.6, 72.6], color="#888888", lw=1.3, ls="--", zorder=1)
    ax.plot([2.4, 2.4], [72.6, 15.0], color="#888888", lw=1.3, ls="--", zorder=1)
    ax.add_patch(FancyArrowPatch(
        (2.4, 15.0), (3.9, 15.0), arrowstyle="-|>", mutation_scale=13, lw=1.3,
        color="#888888", linestyle="--", zorder=1))
    ax.text(1.3, 44.0, "旁路（非图像键 / warmup）", fontsize=8.2, color="#888888",
            rotation=90, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    # ================= 4. 红线与事实 =================
    box(ax, 50, 7.2, 96, 9.6,
        "不放松的红线：不重着色/移动机械臂·夹爪·蓝方块·红盒·桌面   |   "
        "state / action / timestamp 逐帧 bit-identical   |   宁可漏增强，不要错增强\n"
        "在线增强只改**训练时的张量**，数据集落盘内容一个字节都不动 —— 后两条自动满足\n"
        "⚠  batch_size ≥ 1/β = 10：int(16 × 0.1) = 1 才更新；batch 8/9 会静默退化成固定权重叠加",
        "note", 8.8)

    # ================= 5. 实测开销角注 =================
    ax.text(99.4, -3.6,
            "实测开销：基线 452 ms/步（基线 run 的 checkpoint mtime）· 开 SaGA ≈ 1.10×"
            "      |      验证：workplace/saga/scripts/verify_saga.py 31/31 通过（含 FullGrad 完备性）",
            ha="right", va="center", fontsize=8.4, color="#888888")

    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = out / f"saga_framework.{ext}"
        fig.savefig(p, dpi=110, bbox_inches="tight", facecolor="white")
        print(f"[save] {p}")


if __name__ == "__main__":
    main()
