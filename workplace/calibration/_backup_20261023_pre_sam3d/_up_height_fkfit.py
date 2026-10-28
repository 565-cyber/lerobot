#!/usr/bin/env python3
"""探针：用**夹持期**的观测把 up 相机的高度模型 (C*, D) 标出来。

模型（俯视相机，高度 D、天底点 C*）：

    u0 = H · (x_b, y_b, 1)          # 台面 z=0 的像素（精确）
    u_h = C* + k·(u0 − C*),  k = D/(D − h)

    h = 方块中心离台面的高度（米）。h=0 时 u_h=u0。

待拟合 6 个参数：C*_x, C*_y, D, 以及 base 系的常量偏移 (ax, ay, az)
（把 FK/手眼那套系统性误差吸收掉一部分——它主要就是个常量偏移 + 小姿态偏差）。

观测：夹持期 up 图里方块的**饱和黄掩码形心**（H15-40/S≥60/V≥80），按面积加权。
遮挡会让形心偏，所以权重 = 面积，且只信面积大的帧。

输出：残差、D 的置信区间（扫描）、每帧预测 vs 观测。

用法： python _up_height_fkfit.py --ep 0 [--fig]
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
import arm as A                                           # noqa: E402
from fk_check import Urdf, TCP_LINK                        # noqa: E402


def up_bgr(ds, i):
    fr = ds[i]
    return cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255,
                                0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def observe(bgr):
    """饱和黄掩码的最大块 → (形心, 面积, bbox) 或 None。"""
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
    return dict(c=ce[j].astype(float), area=int(st[j, 4]),
                bbox=tuple(int(v) for v in st[j, :4]))


def project_up(g, P_base_m, p):
    """base 系点 → 台面像素 u0（z=0 单应）。"""
    pb = g["T_base_board"] @ np.append(P_base_m, 1.0)
    q = g["H"] @ np.array([pb[0], pb[1], 1.0])
    return q[:2] / q[2]


def model_px(u0, h_mm, p):
    Cx, Cy, D = p[0], p[1], p[2]
    k = D / (D - h_mm)
    C = np.array([Cx, Cy])
    return C + k * (u0 - C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--i0", type=int, default=238)
    ap.add_argument("--i1", type=int, default=412)
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("newdata1", root=str(HERE.parents[1] / "dataset/newdata1"))
    g = bs.load_geom()
    u = g["urdf"]
    eps = ds.meta.episodes
    ia = int(eps[a.ep]["dataset_from_index"])

    rows = []
    for k in range(a.i0, a.i1 + 1):
        fr = ds[k]
        stg = fr["observation.state"].numpy().astype(float)
        normg = {m: float(v) for m, v in zip(A.MOTORS, stg)}
        T = u.fk(A.norm_to_q(normg, u), TCP_LINK)
        P = T[:3, 3]                                  # TCP(=两指中点≈方块中心)
        ob = observe(up_bgr(ds, k))
        rows.append(dict(k=k, P=P, grip=float(normg.get("gripper", np.nan)), ob=ob))
    print(f"ep{a.ep} 帧 {a.i0}~{a.i1}：有观测 {sum(1 for r in rows if r['ob'])}/{len(rows)}")
    for r in rows[::12]:
        P = r["P"] * 1000
        ob = r["ob"]
        print(f"  f{r['k']:4d} TCP=({P[0]:7.1f},{P[1]:7.1f},{P[2]:6.1f})mm grip={r['grip']:5.1f} "
              f"obs={None if ob is None else (np.round(ob['c'],1).tolist(), ob['area'])}")

    # ---- 拟合 ----
    obs = [r for r in rows if r["ob"] is not None and r["ob"]["area"] >= a.min_area]
    u0 = np.array([project_up(g, r["P"], None) for r in obs])
    ypix = np.array([r["ob"]["c"] for r in obs])
    Pz = np.array([r["P"][2] for r in obs])
    wgt = np.array([r["ob"]["area"] for r in obs], float)
    wgt = np.sqrt(wgt / wgt.max())

    def resid(p):
        Cx, Cy, D, ax, ay, az = p
        n = len(u0)
        out = np.zeros(2 * n)
        for i in range(n):
            # 偏移加在 base 系：重投 u0
            Q = obs[i]["P"] + np.array([ax, ay, az])
            uu = project_up(g, Q, None)
            pred = model_px(uu, (Pz[i] + az) * 1000.0, (Cx, Cy, D))
            out[2 * i:2 * i + 2] = (pred - ypix[i]) * wgt[i]
        return out

    from scipy.optimize import least_squares
    best = None
    for D0 in (400.0, 700.0, 1000.0, 1600.0, 3000.0):
        p0 = np.array([430.0, -685.0, D0, 0.0, 0.0, 0.0])
        try:
            r = least_squares(resid, p0, method="lm", max_nfev=4000)
        except Exception as e:
            print(f"  D0={D0} 拟合失败 {e}")
            continue
        res = np.linalg.norm(r.fun.reshape(-1, 2) / wgt[:, None], axis=1)
        print(f"  D0={D0:6.0f} → D={r.x[2]:7.0f}  C*=({r.x[0]:7.1f},{r.x[1]:8.1f})  "
              f"offset=({r.x[3]*1000:6.1f},{r.x[4]*1000:6.1f},{r.x[5]*1000:6.1f})mm  "
              f"RMSE={res.mean():5.2f}px  最大={res.max():5.2f}")
        if best is None or res.mean() < best[0]:
            best = (res.mean(), r.x, res)

    rms, x, res = best
    Cx, Cy, D, ax, ay, az = x
    print(f"\n最优: C*=({Cx:.1f},{Cy:.1f})px  D={D:.0f}mm  "
          f"offset=({ax*1000:.1f},{ay*1000:.1f},{az*1000:.1f})mm  RMSE={rms:.2f}px")
    hcar = np.mean([r["P"][2] for r in obs]) * 1000 + az * 1000
    print(f"  搬运平均高度 h≈{hcar:.0f}mm → 放大系数 k=D/(D−h)={D/(D-hcar):.3f}")
    print(f"  ⇒ 球半径（R=12mm）在 up 图里 ≈ 12·√detJ·k px"
          f"（√detJ 在该处约 1.5 px/mm）⇒ ≈ {12*1.5*D/(D-hcar):.1f}px")

    errs = []
    for i, r in enumerate(obs):
        P = r["P"] + np.array([ax, ay, az])
        pred = model_px(project_up(g, P, None), (P[2]) * 1000.0, (Cx, Cy, D))
        errs.append((r["k"], pred, r["ob"]["c"], np.linalg.norm(pred - r["ob"]["c"]),
                     r["ob"]["area"]))
    print(f"\n{'帧':>5} {'观测':>16} {'模型':>16} {'差':>6} {'面积':>5}")
    for k, pr, ob, d, ar in errs[::6]:
        print(f"{k:5d} ({ob[0]:6.1f},{ob[1]:6.1f}) ({pr[0]:6.1f},{pr[1]:6.1f}) {d:6.1f} {ar:5d}")

    out = HERE / "calib" / "up_height_fkfit.json"
    out.write_text(json.dumps({
        "method": "FK(TCP)+H(z=0)+homothety model fitted to carry-phase observations",
        "C_px": [Cx, Cy], "D_mm": float(D),
        "offset_base_mm": [ax * 1000, ay * 1000, az * 1000],
        "rmse_px": float(rms), "n_obs": len(obs),
        "note": "offset 吸收 FK/手眼系统偏差；D 仍受 FK 绝对精度限制，见 D 扫描",
    }, indent=2))
    print(f"[save] {out}")
    if a.fig:
        make_fig(rows, errs, x)
    return 0


def make_fig(rows, errs, x):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    ax = axs[0]
    ax.plot([e[1][0] for e in errs], [e[1][1] for e in errs], "-", color="C1", lw=1,
            label="模型")
    ax.plot([e[2][0] for e in errs], [e[2][1] for e in errs], ".", color="C0", ms=6,
            label="观测")
    for e in errs:
        ax.plot([e[1][0], e[2][0]], [e[1][1], e[2][1]], "-", color="0.7", lw=0.5)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend()
    ax.set_title("up 视图：模型 vs 观测（夹持期）")
    ax = axs[1]
    ar = np.array([e[4] for e in errs]); dd = np.array([e[3] for e in errs])
    ax.scatter(ar, dd, s=12)
    ax.set_xlabel("观测面积 px"); ax.set_ylabel("残差 px"); ax.grid(alpha=0.3)
    ax.set_title("残差 vs 面积（面积大=没被挡）")
    fig.suptitle(f"D={x[2]:.0f}mm  C*=({x[0]:.0f},{x[1]:.0f})")
    fig.tight_layout()
    p = HERE / "figures" / "up_height_fkfit.png"
    fig.savefig(p, dpi=130)
    print(f"[save] {p}")


if __name__ == "__main__":
    sys.exit(main())
