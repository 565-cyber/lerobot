#!/usr/bin/env python3
"""探针：从 newdata1 的 up 视图中离线解 up 相机的**高度模型**。

原理（见日记）：
  俯视相机在高度 D、天底点 C*（相机正下方那点在图像里的位置）。桌面上 (X,Y,0)
  的像素 u0 由 z=0 单应 H 精确给出。同一 (X,Y) 在高度 h 处的点成像为
      u_h = C* + k·(u0 − C*),   k = D/(D−h)
  于是"底面方块的像素" = "顶面方块的像素" 绕 C* 缩小 1/k 倍。等价地：

      底面角点 i 相对顶面角点 i 的位移  δ_i = B_i − T_i = −(h/D)·(T_i − C*)

  对整块的支持函数取差（Δ(u) = h_comp(u) − h_top(u)）得到一个"铰链"函数：

      Δ(u) = (h/D)·max(0, (C* − T̄)·u)          ← u 指向 C* 时为正、最大

  所以每帧测出 Δ(u) 就能定出一个向量 a_j = (h/D)·(C* − T̄_j)，而
      a_j = g − m·T̄_j,   g = (h/D)·C*,  m = h/D
  对 50 个 episode（方块位置不同）做**线性最小二乘**即可解出 (g, m)，
  于是 C* = g/m，D = h/m（h = 方块高度，用 H 在 z=0 上实测的面积开方）。

用法： python _up_height_probe.py [--h-mm 21.5] [--frames N] [--fig]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import ball_swap as bs                                    # noqa: E402


def unit_dirs(n=72):
    th = np.arange(n) * (2 * np.pi / n)
    return np.stack([np.cos(th), np.sin(th)], 1)


def support(hull_pts, dirs):
    """按方向取支撑函数（hull_pts: (m,2) 凸包顶点）。"""
    return (hull_pts @ dirs.T).max(0)


def top_and_comp(bgr, seed_px, win=42, verbose=False):
    """在 seed 附近取「顶面」与「顶面∪侧面」两块掩码。

    顶面 = 复合块里较亮的连通部分（局部阈值），复合 = 暖色高饱和块。
    返回 (top_hull, comp_hull, info)。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    Hc, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    y0, y1 = int(seed_px[1] - win), int(seed_px[1] + win)
    x0, x1 = int(seed_px[0] - win), int(seed_px[0] + win)
    if x0 < 0 or y0 < 0 or x1 > bgr.shape[1] or y1 > bgr.shape[0]:
        return None, None, {"why": "窗口越界"}
    win_m = ((Hc >= 8) & (Hc <= 48) & (S >= 55) & (V >= 50)).astype(np.uint8)
    m = np.zeros_like(win_m)
    m[y0:y1, x0:x1] = win_m[y0:y1, x0:x1]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None, None, {"why": "无连通块"}
    # 选离 seed 最近、且面积合理的块
    best, bestd = 0, 1e9
    for j in range(1, n):
        if not (150 <= st[j, 4] <= 4000):
            continue
        d = np.hypot(ce[j][0] - seed_px[0], ce[j][1] - seed_px[1])
        if d < bestd:
            best, bestd = j, d
    if best == 0:
        return None, None, {"why": f"面积都不合适 {[int(st[j,4]) for j in range(1,n)]}"}
    comp = (lab == best).astype(np.uint8)
    v = V[comp > 0].astype(np.uint8)
    if v.size < 150:
        return None, None, {"why": "像素太少"}
    # 局部 Otsu 分「亮顶面 / 暗侧面」
    thr, _ = cv2.threshold(v.reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    top = (comp > 0) & (V > thr)
    top = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    nt, labt, stt, _ = cv2.connectedComponentsWithStats(top, 8)
    if nt < 2:
        return None, None, {"why": "顶面为空"}
    top = (labt == int(np.argmax(stt[1:, 4])) + 1).astype(np.uint8)
    if top.sum() < 120:
        return None, None, {"why": f"顶面太小 {int(top.sum())}"}

    def hull(mask):
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None
        return cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)

    ht, hc = hull(top), hull(comp)
    if ht is None or hc is None:
        return None, None, {"why": "凸包失败"}
    info = {"area_comp": int(comp.sum()), "area_top": int(top.sum()),
            "thr": float(thr), "comp_c": tuple(np.round(ce[best], 1)),
            "top_c": tuple(np.round(ht.mean(0), 1))}
    return ht, hc, info


def fit_hinge(dirs, dlt):
    """拟合 Δ(u) = max(0, a·u)，返回 a。迭代剔除负半边。"""
    a = np.zeros(2)
    pos = dlt > 0.5
    for _ in range(6):
        if pos.sum() < 3:
            break
        # 最小二乘 a·u = Δ(u)（只用正半边）
        U, y = dirs[pos], dlt[pos]
        a, *_ = np.linalg.lstsq(U, y, rcond=None)
        pos = (dirs @ a) > 0.5
    return a, pos.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-mm", type=float, default=21.5, help="方块高度 (mm)")
    ap.add_argument("--seed-offset", type=int, default=40)
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("newdata1", root=str(HERE.parents[1] / "dataset/newdata1"))
    g = bs.load_geom()
    mfn, _, _, _ = bs.cube_mask_fns(g, "up", "yellow")

    def up(i):
        x = ds[i]["observation.images.up"].permute(1, 2, 0).numpy()
        if x.max() <= 1.01:
            x = x * 255
        return np.clip(x, 0, 255).astype(np.uint8)

    dirs = unit_dirs(72)
    eps = ds.meta.episodes
    rows = []
    for e in range(len(eps)):
        ia = int(eps[e]["dataset_from_index"])
        i = ia + a.seed_offset
        bgr = cv2.cvtColor(up(i), cv2.COLOR_RGB2BGR)
        m = mfn(bgr)
        n, lab, st, ce = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        if n < 2:
            print(f"ep{e:2d} f{i:5d}: 标定掩码空")
            continue
        j = int(np.argmax(st[1:, 4])) + 1
        seed = ce[j]
        ht, hc, info = top_and_comp(bgr, seed)
        if ht is None:
            print(f"ep{e:2d} f{i:5d}: {info['why']}")
            continue
        dlt = support(hc, dirs) - support(ht, dirs)
        av, npos = fit_hinge(dirs, dlt)
        Tbar = ht.mean(0)
        rows.append({"ep": e, "f": i, "T": Tbar.tolist(), "a": av.tolist(),
                     "npos": int(npos), "maxdlt": float(dlt.max()),
                     "info": info})
        print(f"ep{e:2d} f{i:5d}: T̄=({Tbar[0]:5.1f},{Tbar[1]:5.1f}) "
              f"a=({av[0]:+6.2f},{av[1]:+6.2f}) |a|={np.linalg.norm(av):5.2f} "
              f"Δmax={dlt.max():5.1f} npos={npos:2d} "
              f"areaC={info['area_comp']:5d} areaT={info['area_top']:4d} thr={info['thr']:.0f}")

    print(f"\n有效帧 {len(rows)}/{len(eps)}")
    if len(rows) < 5:
        return 1
    A = np.concatenate([np.stack([np.ones(len(rows)), -np.array([r["T"] for r in rows])[:, 0]], 1),
                        np.stack([np.zeros(len(rows)), -np.array([r["T"] for r in rows])[:, 1]], 1)], 0)
    hmm = a.h_mm
    # a_j = g − m·T̄_j  →  [I, −T_x]·[gx, m]  … 两行分别解 gx/m, gy/m（共用 m）⇒ 联合最小二乘
    T = np.array([r["T"] for r in rows])
    A = np.zeros((2 * len(rows), 4))
    b = np.zeros(2 * len(rows))
    for k, r in enumerate(rows):
        A[2 * k, 0] = 1.0
        A[2 * k, 2] = -T[k, 0]
        b[2 * k] = r["a"][0]
        A[2 * k + 1, 1] = 1.0
        A[2 * k + 1, 2] = -T[k, 1]
        b[2 * k + 1] = r["a"][1]
    sol, res, *_ = np.linalg.lstsq(A, b, rcond=None)
    gxy, m = sol[:2], sol[2]
    print(f"\n解出:  g=(h/D)·C* = ({gxy[0]:.1f},{gxy[1]:.1f})   m = h/D = {m:.6f}")
    if m <= 0:
        print("  m≤0 —— 方向可疑，检查数据")
        return 1
    C = gxy / m
    D = hmm / m
    print(f"  ⇒ 天底点 C* = ({C[0]:.1f},{C[1]:.1f}) px   相机高度 D = {D:.0f} mm")
    print(f"  ⇒ 搬运高度 h 时的放大系数 D/(D−h)：h=100→{D/(D-100):.3f}  "
          f"h=150→{D/(D-150):.3f}  h=200→{D/(D-200):.3f}")
    # 每帧残差
    pred = gxy[None, :] - m * T
    err = np.linalg.norm(np.array([r["a"] for r in rows]) - pred, axis=1)
    print(f"  残差 |a_j − (g − m·T̄_j)|: 均值 {err.mean():.2f}px 中位 {np.median(err):.2f}px "
          f"最大 {err.max():.2f}px（a 的量级 {np.mean([np.linalg.norm(r['a']) for r in rows]):.1f}px）")

    out = HERE / "calib" / "up_height_model.json"
    out.write_text(json.dumps({
        "method": "cube top/base support-function homothety (offline, no ruler)",
        "h_mm": hmm,
        "C_px": C.tolist(), "D_mm": float(D), "m_over_h": float(m),
        "n_frames": len(rows),
        "resid_px_mean": float(err.mean()), "resid_px_max": float(err.max()),
        "per_frame": rows,
    }, indent=2))
    print(f"[save] {out}")
    if a.fig:
        make_fig(rows, dirs, C, D, hmm, m)
    return 0


def make_fig(rows, dirs, C, D, hmm, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.array([r["T"] for r in rows])
    A = np.array([r["a"] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.scatter(T[:, 0], T[:, 1], c=np.linalg.norm(A, axis=1), cmap="inferno", s=30)
    for k in range(len(rows)):
        u = A[k] / (np.linalg.norm(A[k]) + 1e-9)
        ax.plot([T[k, 0], C[0]], [T[k, 1], C[1]], "-", color="0.7", lw=0.6)
    ax.plot(*C, "r*", ms=18, label=f"C*=({C[0]:.0f},{C[1]:.0f})")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("每帧 a_j 方向都指向 C*")
    ax.legend()
    ax = axs[1]
    O = np.zeros(2)
    for k in range(len(rows)):
        u = A[k] / (np.linalg.norm(A[k]) + 1e-9)
        ax.plot([T[k, 0], T[k, 0] + 8 * u[0]], [T[k, 1], T[k, 1] + 8 * u[1]],
                "-", color="C0", lw=1)
    ax.scatter(T[:, 0], T[:, 1], s=8, color="k")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("a_j 方向场（放大 8×）")
    ax = axs[2]
    mag = np.linalg.norm(A, axis=1)
    dist = np.linalg.norm(T - C, axis=1)
    ax.scatter(dist, mag, s=12)
    xs = np.linspace(dist.min(), dist.max(), 50)
    ax.plot(xs, m * xs, "r-", label=f"|a| = (h/D)|C*−T̄|, h/D={m:.5f}")
    ax.set_xlabel("|C* − T̄| (px)")
    ax.set_ylabel("|a| (px)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle(f"up 高度模型：D={D:.0f}mm  h={hmm}mm  N={len(rows)}")
    fig.tight_layout()
    p = HERE / "figures" / "up_height_model.png"
    fig.savefig(p, dpi=130)
    print(f"[save] {p}")


if __name__ == "__main__":
    sys.exit(main())
