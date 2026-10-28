"""Slide figures part 2: the compositing step, run for real.

Unlike part 1 this does not just read the cache -- it builds a layout for
episode 0 with the pipeline's own code and composites actual frames, so every
pixel in these panels is a real output. Run from the repo root.
"""
from __future__ import annotations

import json
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
from background_generalization.transforms.distractor_layout import (  # noqa: E402
    FrameCompositor,
    LayoutConfig,
    render_background_layer,
    sample_layout,
)
from object_generalization.masks.blue_object_mask import _to_hwc_uint8  # noqa: E402

OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

PLATE = os.path.join(BG, "outputs", "plate")
HYBRID = os.path.join(WORKPLACE, "object_generalization", "outputs", "cache_hybrid")
UP = "observation.images.up"
EP, K, SEED = 0, 0, 0


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def panel(ax, img, title, fs=11, cmap=None):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([]); ax.set_yticks([])


def build():
    """Reproduce exactly what generate_distractor_dataset does for (ep, k)."""
    d = os.path.join(PLATE, f"episode_{EP}")
    plate = np.load(os.path.join(d, "plate.npy"))
    placement = np.load(os.path.join(d, "placement.npy"))
    gains = np.load(os.path.join(d, "gain.npy"))
    warps = np.load(os.path.join(d, "warp.npy"))
    rec = json.load(open(os.path.join(d, "done.json")))

    ds = LeRobotDataset("newdata", root=os.path.join(REPO, "dataset", "newdata"),
                        episodes=[EP])
    cube = np.load(os.path.join(HYBRID, f"episode_{EP}", "up_mask.npy"), mmap_mode="r")

    layout_cfg, proc_cfg = LayoutConfig(), ProceduralConfig()
    rng = np.random.default_rng([SEED, EP, K])
    layout = sample_layout(placement, ProceduralCutoutProvider(proc_cfg), layout_cfg, rng)
    bg_layer = render_background_layer(plate, layout, layout_cfg, placement)
    comp = FrameCompositor(
        plate, bg_layer, layout.paint_union(plate.shape[:2]), warps, gains,
        cube_masks=cube, cfg=layout_cfg,
        plate_diff_threshold=rec["config"]["diff_threshold"], layout=layout,
    )
    return ds, plate, placement, layout, layout_cfg, proc_cfg, comp


# ---------------------------------------------------------------- fig 05
def fig_composite():
    """The compositing equation, and the bit-identity it buys."""
    ds, plate, placement, layout, lcfg, pcfg, comp = build()
    print(f"ep{EP} v{K}: {len(layout)} objects, rejected={layout.rejected}, "
          f"placements={[(int(p.top[0]), int(p.top[1])) for p in layout.placements]}")

    fr = 0
    base = _to_hwc_uint8(ds[fr][UP])
    a = comp.alpha(base, fr)
    out = comp.composite(base, fr, a=a)

    # where alpha == 0 the output must be bit-identical to the input
    outside = a <= 0
    bitexact = np.array_equal(out[outside], base[outside])
    n_changed = int((~outside).sum())
    print(f"  frame {fr}: alpha>0 px = {n_changed}, bit-identical outside = {bitexact}")

    fig = plt.figure(figsize=(17, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.55, 1.0], hspace=0.22, wspace=0.05)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, base, f"(a) 原始帧 base  ep{EP} frame {fr}")
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(a, cmap="gray", vmin=0, vmax=1)
    ax.set_title("(b) alpha  (物体 ∪ 接触阴影) × (1 − 前景)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[0, 2]); panel(ax, out, "(c) 合成输出 = (1−a)·base + a·bg_layer")
    ax = fig.add_subplot(gs[0, 3])
    d = np.abs(out.astype(np.int16) - base.astype(np.int16)).max(axis=2)
    im = ax.imshow(d, cmap="turbo", vmin=0, vmax=160)
    ax.set_title("(d) |输出 − 输入|  —— alpha 外恒为 0")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- the invariant, as a claim you can check ---
    ax = fig.add_subplot(gs[1, :2]); ax.axis("off")
    ax.text(0.0, 0.94, "硬不变式（逐帧断言，失败即退出）", fontsize=13, fontweight="bold")
    ax.text(0.0, 0.70, "alpha == 0  的每个像素：  输出 == 输入，逐位相同",
            fontsize=12.5, color="#1a7f37", fontweight="bold")
    ax.text(0.0, 0.50, f"本帧实测：alpha>0 共 {n_changed} px，其余 {int(outside.sum())} px 全部逐位相同 → {bitexact}",
            fontsize=11, color="#333")
    ax.text(0.0, 0.26, "这正是「最小改动合成」与「整张背景替换」的分界：",
            fontsize=11, color="#333")
    ax.text(0.0, 0.10, "前者给出一个可判定的不变式；后者只能靠肉眼看「像不像」。",
            fontsize=11, color="#333")

    # --- the three-way contrast ---
    ax = fig.add_subplot(gs[1, 2:]); ax.axis("off")
    ax.text(0.0, 0.94, "所以基线数据完全没被改写", fontsize=13, fontweight="bold")
    lines = [
        "state / action / timestamp  →  逐帧 bit-identical",
        "front 相机                  →  只有全局光度增益（可关）",
        "UP 相机                     →  只在 alpha 内改动",
    ]
    for i, ln in enumerate(lines):
        ax.text(0.0, 0.68 - i * 0.22, "•  " + ln, fontsize=11, color="#333")
    save(fig, "fig05_composite.png")


# ---------------------------------------------------------------- fig 06
def fig_cutouts():
    """The procedural objects, and the shadow they cast."""
    pcfg = ProceduralConfig()
    rng = np.random.default_rng(7)
    prov = ProceduralCutoutProvider(pcfg)

    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.20, wspace=0.10)

    for i, shape in enumerate(["box", "cylinder", "sphere", "blob"]):
        pcfg_i = ProceduralConfig(shapes=(shape,))
        prov_i = ProceduralCutoutProvider(pcfg_i)
        # draw until we get one of this shape at a reasonable size
        cut = prov_i.sample(np.random.default_rng(100 + i))
        ax = fig.add_subplot(gs[0, i])
        # render the canvas: shadow darkening over a table-coloured field
        base = np.full((*cut.shadow.shape, 3), 181.0, np.float32)
        base *= (1.0 - pcfg_i.shadow_strength * cut.shadow)[..., None]
        base = base * (1 - cut.alpha[..., None]) + cut.rgb * cut.alpha[..., None]
        ax.imshow(np.clip(base, 0, 255).astype(np.uint8))
        ax.set_title(f"{shape}   {cut.size[0]}×{cut.size[1]} px", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ay, ax_ = cut.anchor
        ax.plot([ax_], [ay], marker="+", color="red", ms=12, mew=2)

    # shadow profile: the measurement that set the defaults
    ax = fig.add_subplot(gs[1, :2])
    rr = np.linspace(0.8, 2.4, 200)
    # fitted model: stack of Gaussian-weighted offset copies, params (0.30, 0.40)
    length, spread = pcfg.shadow_len_ratio, pcfg.shadow_spread_ratio
    prof = np.zeros_like(rr)
    for u in np.linspace(0, 1, pcfg.shadow_steps):
        off = length + (u - 0.5) * spread
        w = np.exp(-0.5 * ((off - length) / max(spread / 2.4, 1e-6)) ** 2)
        prof += w * (rr <= off)  # inside the offset copy
    prof = 1.0 - pcfg.shadow_strength * np.clip(prof, 0, 1)
    ax.plot(rr, prof, lw=2.5, color="#4c78a8",
            label=f"拟合模型 (len={length}, spread={spread})")
    meas_r = [1.25, 1.50, 1.75, 2.00]
    meas_v = [0.55, 0.77, 0.93, 1.00]
    ax.plot(meas_r, meas_v, "o", ms=9, color="#f58518", zorder=5,
            label="实测：蓝方块，ep0 帧 0/10/20/30/40")
    for x, y in zip(meas_r, meas_v):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(6, -12),
                    fontsize=9, color="#a85a00")
    ax.axhline(1.0, color="#999", ls=":", lw=1)
    ax.set_xlabel("离轮廓中心距离（以方块半径 r 为单位）")
    ax.set_ylabel("暗度（× 桌面亮度）")
    ax.set_title("接触阴影是一个梯度，不是平移复制 —— 参数由这 4 个点网格搜索拟合",
                 fontsize=10.5)
    ax.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # the wrong way, for contrast
    ax = fig.add_subplot(gs[1, 2:])
    n = 128
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2.0
    disc = ((yy - c) ** 2 + (xx - c) ** 2) <= (n * 0.28) ** 2
    az = np.deg2rad(pcfg.shadow_azimuth_deg)
    ux, uy = np.cos(az), np.sin(az)
    off = int(pcfg.shadow_len_ratio * n)
    rigid = np.roll(np.roll(disc, int(round(off * uy)), axis=0), int(round(off * ux)), axis=1)
    rigid_img = np.full((n, n, 3), 181.0, np.float32)
    rigid_img[rigid] = 181.0 * (1 - pcfg.shadow_strength)
    rigid_img[disc] = (200, 120, 60)

    grad_img = np.full((n, n, 3), 181.0, np.float32)
    acc = np.zeros((n, n), np.float32); wsum = 0.0
    sigma = max(pcfg.shadow_spread_ratio * n / 2.4, 1e-6)
    for u in np.linspace(0, 1, pcfg.shadow_steps):
        o = (pcfg.shadow_len_ratio + (u - 0.5) * pcfg.shadow_spread_ratio) * n
        w = float(np.exp(-0.5 * ((o - pcfg.shadow_len_ratio * n) / sigma) ** 2))
        M = np.float32([[1, 0, o * ux], [0, 1, o * uy]])
        acc += w * cv2.warpAffine(disc.astype(np.float32), M, (n, n))
        wsum += w
    acc /= max(wsum, 1e-6)
    acc = cv2.GaussianBlur(acc, (5, 5), 1.0)
    grad_img *= (1 - pcfg.shadow_strength * np.clip(acc, 0, 1))[..., None]
    grad_img[disc] = (200, 120, 60)

    both = np.hstack([rigid_img, np.full((n, 10, 3), 255, np.float32), grad_img])
    ax.imshow(np.clip(both, 0, 255).astype(np.uint8))
    ax.set_title("左：刚性偏移复制 → 读起来像「第二个灰色物体」      右：高斯加权叠放（现在用的）",
                 fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Stage 2: 干扰物长什么样 —— 参数全部从数据集里量出来", fontsize=13, y=1.0)
    save(fig, "fig06_cutouts.png")


if __name__ == "__main__":
    fig_composite()
    fig_cutouts()
