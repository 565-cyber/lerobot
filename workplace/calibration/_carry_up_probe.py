#!/usr/bin/env python3
"""探针：夹持期 up 图里那坨「暖色块」到底能不能量出方块的顶面/剪影。

目的：**让 up 视图在夹持期真的能画球**。已知（见 memory）up 相机只有 z=0 单应，
缺高度标定；但球半径可以完全绕开高度 —— 只要拿方块自己当尺子：

    r_ball_px = (R/cube_h) · sqrt(顶面像素面积)

深度（含 k=D/(D−h) 和高度未知带来的放大）在分子分母里约掉。所以问题只剩
「顶面那块的面积/形心量得准不准」，也就是**手指遮挡**。

这个探针按帧打印：暖色块 bbox/面积、Otsu 分出的亮块（=顶面）面积/形心、
换算出的等效边长，以及按 z=0 单应推出的「若方块平放在台面上」应有的边长。

用法：
    python _carry_up_probe.py --ep 0 [--ascii 330-345]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import ball_swap as bs                                    # noqa: E402


def up_bgr(ds, i):
    fr = ds[i]
    return cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255,
                                0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def local_sqrt_detJ(g, xy_m):
    """z=0 单应在该点的局部面积因子 sqrt(det J)，单位 px/mm（1mm→几像素）。"""
    H = np.asarray(g["H"], float)
    p = np.array([[[xy_m[0], xy_m[1]]]], float)
    o = cv2.perspectiveTransform(p, H)[0, 0]
    d = 1e-3  # 1mm
    px = cv2.perspectiveTransform(p + [[[d, 0.0]]], H)[0, 0]
    py = cv2.perspectiveTransform(p + [[[0.0, d]]], H)[0, 0]
    J = np.stack([(px - o) / d, (py - o) / d], 1)       # px per mm, 2x2
    return float(np.sqrt(abs(np.linalg.det(J)))), J


def warm_blob(bgr, seed_px, win=46, loose=(8, 48, 45, 55)):
    """seed 附近的「暖色块」（顶面+侧面，含暗的那半）。返回 (comp_mask, hsv) 或 (None, hsv)。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    Hc, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    x0, y0 = int(seed_px[0] - win), int(seed_px[1] - win)
    x1, y1 = int(seed_px[0] + win), int(seed_px[1] + win)
    if x0 < 0 or y0 < 0 or x1 > bgr.shape[1] or y1 > bgr.shape[0]:
        return None, hsv
    h0, h1, s0, v0 = loose
    m = np.zeros(bgr.shape[:2], np.uint8)
    m[y0:y1, x0:x1] = (((Hc >= h0) & (Hc <= h1) & (S >= s0) & (V >= v0))
                       [y0:y1, x0:x1]).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None, hsv
    best, bd = 0, 1e9
    for j in range(1, n):
        if st[j, 4] < 60:
            continue
        # 加权：离 seed 近 + 面积大
        d = np.hypot(*(ce[j] - np.asarray(seed_px, float)))
        if d < bd:
            best, bd = j, d
    if best == 0:
        return None, hsv
    return (lab == best).astype(np.uint8), hsv


def split_top(comp, hsv, vthr=None):
    """comp 里按 V 分「亮顶面 / 暗侧面」。返回 (top_mask, thr)。"""
    V = hsv[:, :, 2]
    v = V[comp > 0].astype(np.uint8)
    if v.size < 60:
        return None, None
    if vthr is None:
        thr, _ = cv2.threshold(v.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr = vthr
    top = ((comp > 0) & (V > thr)).astype(np.uint8)
    top = cv2.morphologyEx(top, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return top, float(thr)


def stats(m):
    """掩码 → (面积, 形心, bbox, 外接矩形最短边)。"""
    if m is None or m.sum() == 0:
        return None
    ys, xs = np.nonzero(m)
    return dict(area=int(m.sum()), c=[float(xs.mean()), float(ys.mean())],
                bbox=(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                      int(ys.max() - ys.min() + 1)))


def ascii_map(m, m2, x0, x1, y0, y1):
    print(f"     {''.join(str((x // 10) % 10) for x in range(x0, x1))}")
    for y in range(y0, y1):
        row = ""
        for x in range(x0, x1):
            a = m is not None and m[y, x] > 0
            b = m2 is not None and m2[y, x] > 0
            row += "T" if b else ("o" if a else ".")
        print(f"{y:4d} {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--ascii", default="", help="如 330-345 打印这几帧的掩码 ASCII")
    ap.add_argument("--cube-mm", type=float, default=21.5)
    ap.add_argument("--ball-mm", type=float, default=24.0)
    ap.add_argument("--loose-v", type=float, default=45.0)
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("newdata1", root=str(HERE.parents[1] / "dataset/newdata1"))
    g = bs.load_geom()
    mfn, _, _, _ = bs.cube_mask_fns(g, "up", "yellow")
    eps = ds.meta.episodes
    ia = int(eps[a.ep]["dataset_from_index"])
    ib = int(eps[a.ep]["dataset_to_index"])
    ig, ir = bs.episode_bounds(ds, ia)
    print(f"ep{a.ep}: 帧 [{ia},{ib})  夹取 {ig}  结束 {ir}")

    seed, kseed = bs.last_up_px(ds, ig, mfn)
    print(f"种子（夹取前最后一帧 up 标定掩码）: {np.round(seed,1)} @ f{kseed}")
    trk = bs.carry_wedge_track(ds, ig, ir, seed)

    doascii = set()
    if a.ascii:
        lo, hi = a.ascii.split("-")
        doascii = set(range(int(lo), int(hi) + 1))

    print(f"\n{'帧':>5} {'追踪c':>14} {'暖块面积':>7} {'暖块bbox':>18} {'顶面面积':>7} "
          f"{'顶面形心':>14} {'顶bbox':>16} {'√顶面':>6} {'thr':>4} {'k=D/(D-h)@D=1000':>6}")
    rows = []
    for k in sorted(trk):
        bgr = up_bgr(ds, k)
        c = trk[k]["c"]
        comp, hsv = warm_blob(bgr, c, loose=(8, 48, 45, a.loose_v))
        if comp is None:
            print(f"{k:5d} {np.round(c,1)}  暖块空")
            continue
        top, thr = split_top(comp, hsv)
        sc, st = stats(comp), stats(top)
        if st is None:
            print(f"{k:5d} 顶面空")
            continue
        sq = float(np.sqrt(st["area"]))
        rows.append(dict(k=k, c=c, comp=sc, top=st, sq=sq, thr=thr,
                         bbox=trk[k].get("bbox")))
        print(f"{k:5d} ({c[0]:5.1f},{c[1]:5.1f}) {sc['area']:7d} {str(sc['bbox']):>18} "
              f"{st['area']:7d} ({st['c'][0]:5.1f},{st['c'][1]:5.1f}) "
              f"{str(st['bbox']):>16} {sq:6.1f} {thr:4.0f}")
        if k in doascii:
            x0, y0, w, h = sc["bbox"]
            x0a, x1a = max(x0 - 12, 0), min(x0 + w + 12, bgr.shape[1])
            y0a, y1a = max(y0 - 12, 0), min(y0 + h + 12, bgr.shape[0])
            print(f"--- f{k} ASCII (o=暖块 T=顶面) ---")
            ascii_map(comp, top, x0a, x1a, y0a, y1a)

    if not rows:
        print("无有效帧")
        return 1
    sq = np.array([r["sq"] for r in rows])
    print(f"\n√顶面面积: 均值 {sq.mean():.1f}px  中位 {np.median(sq):.1f}  "
          f"范围 {sq.min():.1f}~{sq.max():.1f}  变异 {100*sq.std()/sq.mean():.1f}%")
    print(f"⇒ 球半径 = ({a.ball_mm/2:.1f}/{a.cube_mm})·√顶面 = "
          f"{a.ball_mm/2/a.cube_mm*sq.mean():.1f}px（均值）")
    print(f"   球直径 = {a.ball_mm/a.cube_mm*sq.mean():.1f}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
