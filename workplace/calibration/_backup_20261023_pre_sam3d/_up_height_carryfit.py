#!/usr/bin/env python3
"""探针：**把夹持期的方块当杠杆**，和台面期的 49 帧一起联立求 up 相机高度 (C*, D)。

为什么夹持期有用：铰链关系 Δ(u) = (h_cube/D)·max(0,(C*−T̄)·u) 里的 h_cube 是**方块
自己的高度**（21.5mm），与搬运高度无关；而同时它给出 a_j = (21.5/D)(C* − T̄_j)。
台面那 49 帧的 T̄ 只跨 ~100px ⇒ 方向可信、距离不可信（残差对 D 几乎不敏感）。
夹持期搬运时 T̄ 跨 ~340px，两个量就分得开了 ⇒ 能定出 D。

用法： python _up_height_carryfit.py --ep 0 [--fig]
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

import ball_swap as bs                                          # noqa: E402
import _up_height_probe as P                                    # noqa: E402


def up_bgr(ds, i):
    fr = ds[i]
    return cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255,
                                0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def sat_seed(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    Hc, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((Hc >= 15) & (Hc <= 40) & (S >= 60) & (V >= 80)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    idx = [j for j in range(1, n) if st[j, 4] >= 40]
    if not idx:
        return None
    j = max(idx, key=lambda j: st[j, 4])
    return ce[j], int(st[j, 4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--i0", type=int, default=300)
    ap.add_argument("--i1", type=int, default=370)
    ap.add_argument("--min-area", type=int, default=600)
    ap.add_argument("--h-mm", type=float, default=21.5)
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("newdata1", root=str(HERE.parents[1] / "dataset/newdata1"))
    g = bs.load_geom()
    dirs = P.unit_dirs(72)

    # ---- 台面期 49 帧（上一步存的） ----
    prev = json.loads((HERE / "calib" / "up_height_model.json").read_text())
    desk = [dict(T=np.array(r["T"]), a=np.array(r["a"]), src="desk", f=r["f"])
            for r in prev["per_frame"]]
    print(f"台面期 {len(desk)} 帧（T̄ 跨 "
          f"{np.ptp([d['T'][0] for d in desk]):.0f}×{np.ptp([d['T'][1] for d in desk]):.0f}px）")

    # ---- 夹持期 ----
    carry = []
    for k in range(a.i0, a.i1 + 1):
        bgr = up_bgr(ds, k)
        s = sat_seed(bgr)
        if s is None or s[1] < a.min_area:
            continue
        seed, area = s
        ht, hc, info = P.top_and_comp(bgr, seed, win=52)
        if ht is None:
            print(f"f{k:4d} area={area:5d} — {info['why']}")
            continue
        dlt = P.support(hc, dirs) - P.support(ht, dirs)
        av, npos = P.fit_hinge(dirs, dlt)
        T = ht.mean(0)
        carry.append(dict(T=T, a=av, src="carry", f=k))
        print(f"f{k:4d} area={area:5d} T̄=({T[0]:6.1f},{T[1]:6.1f}) "
              f"a=({av[0]:+6.2f},{av[1]:+6.2f}) |a|={np.linalg.norm(av):5.2f} "
              f"Δmax={dlt.max():4.0f} npos={npos:2d} areaC={info['area_comp']:5d} "
              f"areaT={info['area_top']:4d} thr={info['thr']:.0f} fill={info['area_top']/info['area_comp']:.2f}")

    rows = desk + carry
    if len(carry) < 4:
        print("夹持期有效帧太少")
        return 1
    T = np.array([r["T"] for r in rows])
    A = np.array([r["a"] for r in rows])
    M = np.zeros((2 * len(rows), 4))
    b = np.zeros(2 * len(rows))
    for k, r in enumerate(rows):
        M[2 * k, 0] = 1.0
        M[2 * k, 2] = -T[k, 0]
        b[2 * k] = r["a"][0]
        M[2 * k + 1, 1] = 1.0
        M[2 * k + 1, 2] = -T[k, 1]
        b[2 * k + 1] = r["a"][1]
    sol, *_ = np.linalg.lstsq(M, b, rcond=None)
    gxy, m = sol[:2], sol[2]
    C = gxy / m
    D = a.h_mm / m
    pred = gxy[None, :] - m * T
    err = np.linalg.norm(A - pred, axis=1)
    print(f"\n联立 {len(rows)} 帧（台面 {len(desk)} + 夹持 {len(carry)}）:")
    print(f"  C* = ({C[0]:.1f},{C[1]:.1f})px   D = {D:.0f}mm   (h_cube={a.h_mm}mm)")
    print(f"  残差：台面 {err[:len(desk)].mean():.2f}px  夹持 {err[len(desk):].mean():.2f}px  "
          f"全体 均值 {err.mean():.2f} 最大 {err.max():.2f}px")

    # 只夹持期
    for tag, sub in (("只台面", desk), ("只夹持", carry)):
        if len(sub) < 3:
            continue
        T2 = np.array([r["T"] for r in sub]); A2 = np.array([r["a"] for r in sub])
        M2 = np.zeros((2 * len(sub), 4)); b2 = np.zeros(2 * len(sub))
        for k, _ in enumerate(sub):
            M2[2 * k, 0] = 1.0; M2[2 * k, 2] = -T2[k, 0]; b2[2 * k] = A2[k, 0]
            M2[2 * k + 1, 1] = 1.0; M2[2 * k + 1, 2] = -T2[k, 1]; b2[2 * k + 1] = A2[k, 1]
        s2, *_ = np.linalg.lstsq(M2, b2, rcond=None)
        if s2[2] > 0:
            C2 = s2[:2] / s2[2]; D2 = a.h_mm / s2[2]
            e2 = np.linalg.norm(A2 - (s2[None, :2] - s2[2] * T2), axis=1)
            print(f"  {tag}: C*=({C2[0]:7.1f},{C2[1]:8.1f})  D={D2:7.0f}mm  "
                  f"残差 {e2.mean():.2f}px（|C*−T̄| 跨度 "
                  f"{np.ptp(np.linalg.norm(T2-C2,axis=1)):.0f}px）")

    # ---- D 扫描：固定 D 反解最优 C*，看残差形状 ----
    print(f"\nD 扫描（固定 D，对 C* 做最小二乘，报残差）：")
    for Dt in (200, 300, 400, 500, 600, 700, 800, 1000, 1300, 1700, 2500, 4000, 8000):
        mt = a.h_mm / Dt
        # a_j = mt·(C* − T̄_j)  ⇒  C* 的最小二乘
        Cs = T + A / mt
        Cfit = Cs.mean(0)
        e = np.linalg.norm(A - mt * (Cfit[None, :] - T), axis=1)
        print(f"  D={Dt:5d}mm  C*=({Cfit[0]:8.1f},{Cfit[1]:9.1f})  残差 均值 {e.mean():5.2f} "
              f"最大 {e.max():6.2f}px   只夹持 {e[len(desk):].mean():5.2f}px")

    (HERE / "calib" / "up_height_model2.json").write_text(json.dumps({
        "method": "support-function hinge, desktop(49) + carry-phase joint fit",
        "h_mm": a.h_mm, "C_px": C.tolist(), "D_mm": float(D), "m_over_h": float(m),
        "n_desk": len(desk), "n_carry": len(carry),
        "resid_desk_px": float(err[:len(desk)].mean()),
        "resid_carry_px": float(err[len(desk):].mean()),
        "carry_frames": [r["f"] for r in carry],
    }, indent=2))
    print(f"[save] {HERE/'calib'/'up_height_model2.json'}")
    if a.fig:
        make_fig(rows, len(desk), C, D, a.h_mm, m)
    return 0


def make_fig(rows, nd, C, D, hmm, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.array([r["T"] for r in rows]); A = np.array([r["a"] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.scatter(T[:nd, 0], T[:nd, 1], c="0.6", s=18, label="台面 49 帧")
    ax.scatter(T[nd:, 0], T[nd:, 1], c="C3", s=30, label="夹持期")
    ax.plot(*C, "r*", ms=18)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend()
    ax.set_title(f"C*=({C[0]:.0f},{C[1]:.0f})")
    ax = axs[1]
    for k in range(len(rows)):
        u = A[k] / (np.linalg.norm(A[k]) + 1e-9)
        sc = 0.02  # 方向场：画成从 T̄ 指向 C* 的短线
        ax.plot([T[k, 0], C[0]], [T[k, 1], C[1]], "-", color="C3" if k >= nd else "0.7", lw=0.6)
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_title("每帧 T̄ → C*（灰=台面, 红=夹持）")
    ax = axs[2]
    dist = np.linalg.norm(T - C, axis=1); mag = np.linalg.norm(A, axis=1)
    ax.scatter(dist[:nd], mag[:nd], c="0.6", s=18, label="台面")
    ax.scatter(dist[nd:], mag[nd:], c="C3", s=30, label="夹持")
    xs = np.linspace(dist.min(), dist.max(), 50)
    ax.plot(xs, m * xs, "r-", label=f"|a|=(h/D)|C*−T̄|, h/D={m:.5f}")
    ax.set_xlabel("|C* − T̄| px"); ax.set_ylabel("|a| px"); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle(f"up 高度模型（联立）：D={D:.0f}mm")
    fig.tight_layout()
    p = HERE / "figures" / "up_height_model2.png"
    fig.savefig(p, dpi=130); print(f"[save] {p}")


if __name__ == "__main__":
    sys.exit(main())
