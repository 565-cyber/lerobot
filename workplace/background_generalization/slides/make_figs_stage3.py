"""Slide figure: the place_freq_max sweep that set the 0.10 default.

Numbers are the ones recorded in PlateConfig.place_freq_max's docstring,
measured by sweeping the parameter on episodes 0/1/5 with everything else fixed.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

THR = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
AREA = [10.0, 20.8, 32.0, 42.8, 59.9, 70.8]          # ep1 legal area, %
RUNS = [(0, 0, 0), (0, 0, 0), (5, 4, 0), (9, 5, 1), (8, 5, 5), (6, 7, 7)]

fig, ax = plt.subplots(figsize=(11.5, 5.4))
x = np.arange(len(THR))
colors = ["#4c78a8"] * 2 + ["#f58518"] * 4
bars = ax.bar(x, AREA, color=colors, width=0.6, zorder=3)
for b, v in zip(bars, AREA):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center",
            fontsize=10.5, fontweight="bold", zorder=4)

for i, r in enumerate(RUNS):
    tot = sum(r)
    ax.text(i, 4, f"{tot}", ha="center", fontsize=12, fontweight="bold",
            color="#c0392b" if tot else "#1a7f37", zorder=4)
ax.text(-0.75, 4, "中段遮挡\n帧数 (ep0/1/5 合计)", ha="right", va="center",
        fontsize=9.5, color="#333")

ax.axvline(1.5, color="#333", ls="--", lw=1.5, zorder=2)
ax.text(1.42, 63, "选 0.10", ha="right", fontsize=11, fontweight="bold", color="#4c78a8")
ax.text(1.58, 63, "开始出现中段闪烁", ha="left", fontsize=11, color="#c0392b")

ax.axvspan(3.5, 5.5, color="#c0392b", alpha=0.07, zorder=0)
ax.text(4.5, 47, "硬上限 0.5：\n中位数=机械臂", ha="center", fontsize=10, color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c0392b", lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in THR], fontsize=11)
ax.set_xlabel("place_freq_max —— 一个像素被判为「合法落点」所需的、与底板匹配的帧比例下限",
              fontsize=10.5)
ax.set_ylabel("ep1 合法面积 (%)", fontsize=10.5)
ax.set_title("放宽阈值买面积，但代价是干扰物在片段中途开始闪烁 —— 0.10 是实测的拐点",
             fontsize=12)
ax.set_ylim(0, 78)
ax.grid(axis="y", alpha=0.25, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
p = os.path.join(OUT, "fig08_freq_sweep.png")
fig.savefig(p, bbox_inches="tight", facecolor="white")
print("wrote", p)
