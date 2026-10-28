"""Slide figures part 1: plate / freq / placement, straight from the cache.

Everything here reads the real Stage-1 artifacts -- nothing is mocked or
hand-drawn. Run from the repo root.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BG = os.path.dirname(HERE)
WORKPLACE = os.path.dirname(BG)
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402

OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# The .ttc ships as one collection; matplotlib indexes it under the JP face,
# but it carries the unified ideographs, so Chinese renders fine.
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

PLATE = os.path.join(BG, "outputs", "plate")
HYBRID = os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid")
UP = "observation.images.up"
EP = 0


def load_ep(ep):
    d = os.path.join(PLATE, f"episode_{ep}")
    return {
        "plate": np.load(os.path.join(d, "plate.npy")),
        "freq": np.load(os.path.join(d, "freq.npy")).astype(np.float32) / 255.0,
        "placement": np.load(os.path.join(d, "placement.npy")),
        "gain": np.load(os.path.join(d, "gain.npy")),
        "rec": json.load(open(os.path.join(d, "done.json"))),
    }


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def panel(ax, img, title, cmap=None):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------- fig 01
def fig_plate_vs_frame():
    """A raw frame vs the temporal median, and what the difference looks like."""
    a = load_ep(EP)
    ds = LeRobotDataset("newdata", root=os.path.join(REPO, "dataset", "newdata"), episodes=[EP])
    fr = 220
    raw = _to_hwc_uint8(ds[fr][UP])
    plate = a["plate"]
    diff = np.abs(raw.astype(np.float32) - plate.astype(np.float32)).max(axis=2)

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    panel(axes[0], raw, f"原始帧 frame {fr}")
    panel(axes[1], plate, "plate = 逐像素时序中位数")
    im = axes[2].imshow(diff, cmap="inferno", vmin=0, vmax=120)
    axes[2].set_title("|frame − plate|  (灰度, max over RGB)")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    # freq: how often each pixel is far from the plate
    im2 = axes[3].imshow(a["freq"], cmap="magma", vmin=0, vmax=1)
    axes[3].set_title(f"freq  (>0.10 的像素占 {float((a['freq']>0.1).mean())*100:.1f}%)")
    axes[3].set_xticks([]); axes[3].set_yticks([])
    fig.colorbar(im2, ax=axes[3], fraction=0.046)
    fig.suptitle("Stage 1: 用「时间」而不是「参考图」定义背景", fontsize=13, y=1.06)
    save(fig, "fig01_plate_vs_frame.png")


# ---------------------------------------------------------------- fig 02
def fig_placement():
    """The placement formula, and the area each term removes."""
    a = load_ep(EP)
    plate, freq, placement = a["plate"], a["freq"], a["placement"]
    place_b = placement > 0

    # reconstruct the terms exactly as build_plate does
    cfg_dilate, cfg_erode, freq_max = 8, 5, 0.10
    raw_free = freq <= freq_max

    def disk(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    cube = np.load(os.path.join(HYBRID, f"episode_{EP}", "up_mask.npy"), mmap_mode="r")
    cube_u = (np.asarray(cube) > 0).any(axis=0).astype(np.uint8)
    grown_cube = cv2.dilate(cube_u, disk(cfg_dilate)) > 0

    hsv = cv2.cvtColor(cv2.cvtColor(plate, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, np.array((0, 90, 70), np.uint8), np.array((10, 255, 255), np.uint8))
           | cv2.inRange(hsv, np.array((170, 90, 70), np.uint8), np.array((179, 255, 255), np.uint8)))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats((red > 0).astype(np.uint8), 8)
    keep = np.zeros_like(red)
    for j in range(1, n_lbl):
        if stats[j, 4] >= 5000:
            keep[lbl == j] = 255
    grown_red = cv2.dilate((keep > 0).astype(np.uint8), disk(cfg_dilate)) > 0

    eroded = cv2.erode(raw_free.astype(np.uint8), disk(cfg_erode)) > 0
    n = plate.shape[0] * plate.shape[1]

    fig = plt.figure(figsize=(17, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 2.0], hspace=0.28, wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, plate, "(a) plate")
    for m, c, lab in ((grown_cube, (0, 0.6, 1.0), "cube trajectory"),
                      (grown_red, (1.0, 0.1, 0.1), "red container")):
        o = np.zeros((*m.shape, 4))
        o[m] = (*c, 0.45)
        ax.imshow(o)
    ax.plot([], [], color=(0, 0.6, 1.0), lw=6, label="excluded: cube trajectory")
    ax.plot([], [], color=(1.0, 0.1, 0.1), lw=6, label="excluded: red container")
    ax.plot([], [], color=(0, 0.75, 0.2), lw=6, label="placement (legal site)")
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    overlay = plate.copy().astype(np.float32)
    overlay[place_b] = 0.35 * overlay[place_b] + 0.65 * np.array([0, 220, 60])
    panel(ax, np.clip(overlay, 0, 255).astype(np.uint8), "(b) placement 叠加")

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, freq, "(c) freq", cmap="magma")
    ax.contour(place_b, levels=[0.5], colors=["#00e060"], linewidths=1.2)

    ax = fig.add_subplot(gs[0, 3])
    panel(ax, placement, "(d) placement 二值图")

    # --- what each term costs, in percent of the frame ---
    # Order follows build_plate exactly: threshold -> erode -> subtract the two
    # exclusion footprints. Eroding first and subtracting after is not the same
    # set, so a chart drawn in the other order would not describe the code.
    ax = fig.add_subplot(gs[1, :])
    steps = [
        ("freq ≤ 0.10\n(原始合法像素)", float(raw_free.mean())),
        ("⊖ 腐蚀 5 px 安全边距", float(eroded.mean())),
        ("∩ ¬dilate(红盒)", float((eroded & ~grown_red).mean())),
        ("∩ ¬dilate(方块轨迹)", float((eroded & ~grown_red & ~grown_cube).mean())),
        ("= placement\n(最终合法区)", float(place_b.mean())),
    ]
    labels = [s[0] for s in steps]
    vals = [s[1] * 100 for s in steps]
    colors = ["#4c78a8"] * 4 + ["#f58518"]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}%", ha="center", fontsize=10,
                fontweight="bold")
    for i in range(1, len(vals)):
        d = vals[i - 1] - vals[i]
        ax.annotate(f"−{d:.2f}", xy=(i - 0.5, max(vals[i - 1], vals[i]) + 2.6), ha="center",
                    fontsize=9, color="#c0392b")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("占画面比例 (%)")
    ax.set_ylim(0, 38)
    ax.set_title(f"placement = (freq ≤ 0.10) ∩ ¬dilate(方块轨迹) ∩ ¬dilate(红盒) ⊖ erode(5 px)"
                 f"   ——  episode {EP}, {n} px 的画面里只剩 {int(place_b.sum())} px",
                 fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig02_placement.png")


# ---------------------------------------------------------------- fig 03
def fig_placement_stats():
    """Legal area and largest blob across all 50 episodes -- the budget."""
    frs, blobs = [], []
    for p in sorted(glob.glob(os.path.join(PLATE, "episode_*", "done.json"))):
        r = json.load(open(p))
        frs.append(r["placement_frac"])
        blobs.append(r["placement_largest_blob"])
    frs = np.array(frs) * 100
    blobs = np.array(blobs) / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
    ax = axes[0]
    ax.hist(frs, bins=16, color="#4c78a8", edgecolor="white")
    ax.axvline(np.median(frs), color="#f58518", lw=2,
               label=f"median {np.median(frs):.1f}%")
    ax.axvline(frs.min(), color="#c0392b", lw=2, ls="--",
               label=f"min {frs.min():.1f}%  (ep{int(np.argmin(frs))})")
    ax.set_xlabel("合法面积占画面比例 (%)")
    ax.set_ylabel("episode 数")
    ax.set_title("50 集的合法区大小", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.hist(blobs, bins=16, color="#54a24b", edgecolor="white")
    ax.axvline(np.median(blobs), color="#f58518", lw=2,
               label=f"median {np.median(blobs):.0f}k px")
    ax.axvline(blobs.min(), color="#c0392b", lw=2, ls="--",
               label=f"min {blobs.min():.0f}k px")
    ax.set_xlabel("最大连通合法区 (千像素)")
    ax.set_ylabel("episode 数")
    ax.set_title("最大的一块能放下几个 56 px 的物体", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig03_placement_stats.png")


# ---------------------------------------------------------------- fig 04
def fig_pipeline():
    """The four stages, and what crosses each boundary."""
    fig, ax = plt.subplots(figsize=(15.5, 4.3))
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 4.3); ax.axis("off")

    stages = [
        ("Stage 1", "build_plate.py", "每集 UP 相机",
         "gain / plate / freq\nplacement", "~15 min，可断点续跑", "#4c78a8"),
        ("Stage 2", "generate_...py", "渲染 K 份",
         "newdata_distract\n50 × K 集", "~2 h，不可续跑", "#f58518"),
        ("Stage 3", "verify_...py", "逐帧对拍",
         "verify_report.json\nPASS / FAIL", "parquet + 像素两级", "#54a24b"),
        ("Stage 4", "inspect_...py", "人眼过一遍",
         "sheets / overlay\noverview", "S7 闸门", "#b279a2"),
    ]
    x = 0.35
    for i, (tag, script, what, out, note, c) in enumerate(stages):
        box = FancyBboxPatch((x, 1.32), 3.2, 2.42, boxstyle="round,pad=0.12",
                             linewidth=2, edgecolor=c, facecolor=c + "22")
        ax.add_patch(box)
        ax.text(x + 1.6, 3.47, f"{tag}   {what}", ha="center", fontsize=11.5,
                fontweight="bold", color=c)
        ax.text(x + 1.6, 3.02, script, ha="center", fontsize=11.5, family="monospace")
        ax.text(x + 1.6, 2.42, out, ha="center", fontsize=9, color="#222")
        ax.text(x + 1.6, 1.72, note, ha="center", fontsize=8.5, color="#666")
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 3.24, 2.53), (x + 3.70, 2.53),
                                         arrowstyle="-|>", mutation_scale=22, color="#555", lw=2))
        x += 3.8

    ax.text(0.35, 0.72, "每个阶段都能单独重跑；跨阶段边界只传文件（.npy 缓存 + .jsonl 账本 + LeRobot 数据集）。",
            fontsize=10.5, color="#333")
    ax.text(0.35, 0.22, "上游是 object_generalization（Stage 2A）：复用了它的蓝方块掩码（排除抓取轨迹）和缓存 IO。",
            fontsize=9.5, color="#777")
    save(fig, "fig04_pipeline.png")


if __name__ == "__main__":
    fig_plate_vs_frame()
    fig_placement()
    fig_placement_stats()
    fig_pipeline()
