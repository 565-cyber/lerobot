"""The contact shadow: what the model renders, and what the frames actually show.

Two panels, deliberately NOT merged into one "they agree" claim:

(a) the provider's own shadow, sampled radially -- checked against the four
    numbers README.md records (0.55 / 0.77 / 0.93 / 1.00 at 1.25/1.5/1.75/2.0 r).
    Those four are quoted from the docs; this panel shows the model reproduces
    them, not that the measurement was redone.

(b) an independent walk outward from the blue cube on frame 0, in absolute
    pixels, with the silhouette boundary marked. This is a fresh measurement and
    its absolute values differ from the quoted four -- see the printed table.
"""
from __future__ import annotations

import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BG = os.path.dirname(HERE)
WORKPLACE = os.path.dirname(BG)
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from background_generalization.distractors.procedural import (  # noqa: E402
    ProceduralConfig,
    ProceduralCutoutProvider,
)
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

OUT = os.path.join(HERE, "figs")
PLATE = os.path.join(BG, "outputs", "plate")
HYBRID = os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid")
UP = "observation.images.up"
EP = 0
DOC_R = [1.25, 1.50, 1.75, 2.00]
DOC_V = [0.55, 0.77, 0.93, 1.00]


def model_profile(radii, n_draws=14):
    """The provider's shadow, sampled radially -- radius from the real silhouette."""
    cfg = ProceduralConfig(shapes=("sphere",))
    rows = []
    for seed in range(n_draws):
        cut = ProceduralCutoutProvider(cfg).sample(np.random.default_rng(seed))
        ay, ax = cut.anchor
        # The object's radius, NOT the canvas half-height: the canvas is padded
        # to hold the shadow and is asymmetric, so size[0] / 2 is not the object.
        r = float(np.sqrt((cut.alpha > 0).sum() / np.pi))
        az = np.deg2rad(cfg.shadow_azimuth_deg)
        ux, uy = float(np.cos(az)), float(np.sin(az))
        row = []
        for d in radii:
            y, x = int(round(ay + uy * d * r)), int(round(ax + ux * d * r))
            s = 0.0
            if 0 <= y < cut.shadow.shape[0] and 0 <= x < cut.shadow.shape[1]:
                s = float(cut.shadow[y, x])
            row.append(1.0 - cfg.shadow_strength * s)
        rows.append(row)
    return np.array(rows)


def measure_cube():
    """Walk outward along the shadow azimuth from the blue cube, in absolute px."""
    plate = np.load(os.path.join(PLATE, f"episode_{EP}", "plate.npy"))
    plate_g = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ds = LeRobotDataset("newdata", root=os.path.join(REPO, "dataset", "newdata"),
                        episodes=[EP])
    cube = np.load(os.path.join(HYBRID, f"episode_{EP}", "up_mask.npy"), mmap_mode="r")
    rgb = _to_hwc_uint8(ds[0][UP])
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = np.asarray(cube[0]).squeeze() > 0
    ys, xs = np.nonzero(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    r_eq = float(np.sqrt(m.sum() / np.pi))

    az = np.deg2rad(275.0)
    ux, uy = float(np.cos(az)), float(np.sin(az))
    # a small fan, median across it, so one noisy pixel does not set the curve
    fan = np.deg2rad(np.arange(-10, 10.1, 2.5))
    ds_px = np.arange(0, 46)
    prof, inside = [], []
    for d in ds_px:
        vals = []
        for a in fan:
            y = int(round(cy + np.sin(az + a) * d))
            x = int(round(cx + np.cos(az + a) * d))
            if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] and plate_g[y, x] > 20:
                vals.append(g[y, x] / plate_g[y, x])
        prof.append(float(np.median(vals)) if vals else np.nan)
        inside.append(bool(m[int(round(cy + uy * d)), int(round(cx + ux * d))])
                      if 0 <= int(round(cy + uy * d)) < 480 else False)
    return ds_px, np.array(prof), np.array(inside), r_eq, (cy, cx)


def main():
    radii = np.arange(0.9, 2.5, 0.04)
    mod = model_profile(radii)
    mod_med = mod.mean(axis=0)

    print("model vs the four numbers README records:")
    for d, v in zip(DOC_R, DOC_V):
        i = int(np.argmin(np.abs(radii - d)))
        print("   %.2f r: model %.3f   README %.2f   diff %+.3f" % (d, mod_med[i], v, mod_med[i] - v))

    ds_px, prof, inside, r_eq, centre = measure_cube()
    boundary = int(np.max(ds_px[inside]))
    print(f"\nindependent walk on ep0 frame 0: equivalent radius r={r_eq:.1f} px, "
          f"silhouette ends at d={boundary} px along 275 deg")
    for k in range(boundary - 2, min(boundary + 16, len(ds_px))):
        print("   d=%2d px  ratio=%.3f %s" % (ds_px[k], prof[k], "<- outside" if k > boundary else ""))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))

    ax = axes[0]
    ax.plot(radii, mod_med, lw=3, color="#4c78a8",
            label=f"模型渲染（len={ProceduralConfig().shadow_len_ratio}, "
                  f"spread={ProceduralConfig().shadow_spread_ratio}, "
                  f"strength={ProceduralConfig().shadow_strength}）")
    ax.plot(DOC_R, DOC_V, "o", ms=11, color="#f58518", zorder=5,
            label="README 记录的实测点（蓝方块）")
    for x, y in zip(DOC_R, DOC_V):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(8, -14),
                    fontsize=9.5, color="#a85a00")
    ax.axhline(1.0, color="#999", ls=":", lw=1)
    ax.axhline(1 - ProceduralConfig().shadow_strength, color="#c0392b", ls="--", lw=1.2)
    ax.text(2.44, 1 - ProceduralConfig().shadow_strength + 0.012, "阴影核心 = 1 − 0.43 = 0.57",
            fontsize=9, color="#c0392b", ha="right")
    ax.set_xlabel("离轮廓中心距离（以等效面积半径 r 为单位）")
    ax.set_ylabel("亮度 ÷ 桌面（1.0 = 无阴影）")
    ax.set_title("(a) 模型复现文档记录的四个点，最大偏差 0.066", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_ylim(0.45, 1.06); ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.plot(ds_px, prof, lw=3, color="#f58518", marker="o", ms=4,
            label="独立复测：ep0 帧 0，沿 275° 扇形中位数")
    ax.axvspan(boundary + 0.5, ds_px[-1], color="#4c78a8", alpha=0.07)
    ax.axvline(boundary, color="#c0392b", ls="--", lw=1.5)
    ax.text(boundary + 0.6, 1.04, f"轮廓边界 d={boundary} px\n(等效面积半径 r={r_eq:.1f} px)",
            fontsize=9.5, color="#c0392b")
    ax.axhline(1.0, color="#999", ls=":", lw=1)
    ax.set_xlabel("离方块轮廓中心的距离（绝对像素）")
    ax.set_ylabel("亮度 ÷ 桌面")
    ax.set_title("(b) 独立复测：阴影在轮廓外约 12 px 内衰减掉", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_ylim(0.0, 1.15); ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("接触阴影的参数是从数据里量出来的 —— 而「怎么量」本身也是要小心的",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "fig07_shadow_measured.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
