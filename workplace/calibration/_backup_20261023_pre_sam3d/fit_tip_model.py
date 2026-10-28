#!/usr/bin/env python3
"""离线：用已有 9 帧把「TCP 位置 -> up 图上的像素」这个模型拟合出来。

动机：`anchor_base.py` 的 9 个位姿里只有 1 个（#4）的夹爪真的在画面内，其余 8 个
尖端被画幅下边缘切掉了，自动分出来的「尖端」是臂的轮廓端点，是垃圾。与其再盲开
一轮臂，不如先用这 9 对 (TCP, 像素) 把相机模型定出来 —— 有了它就能**预测**某个位姿
的尖端会不会落在画面里，再据此挑位姿。

模型（俯视 + 视差）：

    M1 = B + s·r·(cos(az+φ), sin(az+φ))      # TCP 在**桌面高度**上的像点
    pred = C + k·(M1 − C),  k = Z/(Z−h)        # 离桌 h 高 → 绕光轴脚 C 放大 k 倍

未知 7 个：B(2)（底座在桌面高度上的像点）、s（px/m）、φ（图像方位 = FK 方位 + φ）、
C(2)（光轴脚的像点）、Z（相机离桌面高）。

只用 #4 是解不出来的（1 个点 2 个方程，7 个未知）。9 个点里 8 个是垃圾 —— 所以用
soft_l1 抗差，并**逐个报残差**：#4 必须小，其余应该是「预测落在画面外」。

用法：
    python workplace/calibration/fit_tip_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAL = HERE / "calib"
FIG = HERE / "figures"
W, H = 640, 480


def model(p, tcp_mm, h_m):
    Bx, By, s, phi, Cx, Cy, Z = p
    xy = np.asarray(tcp_mm)[:, :2] / 1000.0
    r = np.hypot(xy[:, 0], xy[:, 1])
    az = np.arctan2(xy[:, 1], xy[:, 0])
    M1 = np.stack([Bx + s * r * np.cos(az + phi),
                   By + s * r * np.sin(az + phi)], 1)
    k = Z / (Z - h_m)
    C = np.array([Cx, Cy])
    return C + (k * (M1 - C).T).T, M1, k


def main() -> int:
    z = np.load(CAL / "anchor_frames.npz")
    tcp, h, tips = z["tcp"], z["h"], z["tips"]
    n = len(tcp)
    print(f"[数据] {n} 帧   tcp (N,3) mm    h (N,) m    tips (N,2) px")

    xy = tcp[:, :2] / 1000.0
    r = np.hypot(xy[:, 0], xy[:, 1])
    az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    print(f"  {'#':>2} {'r mm':>7} {'az deg':>7} {'h mm':>7}   {'tip px':>14}   在图内")
    for i in range(n):
        inside = (5 <= tips[i, 0] <= W - 5) and (5 <= tips[i, 1] <= H - 5)
        print(f"  {i:>2} {r[i]*1000:>7.1f} {az[i]:>7.1f} {h[i]*1000:>7.1f}   "
              f"({tips[i,0]:>6.1f},{tips[i,1]:>6.1f})   {'是' if inside else '否'}")

    from scipy.optimize import least_squares

    # 起点：从 #4 反推（唯一可信的点）。az≈0 时尖端几乎竖直向上 ⇒ φ ≈ -90°
    good = int(np.argmin(h))  # 占位，下面直接给常数起点
    p0 = [332.0, 541.0, 1433.0, np.radians(-86.0), 320.0, 240.0, 0.30]

    def resid(p):
        pred, _, _ = model(p, tcp, h)
        return (pred - tips).ravel()

    print("\n[拟合] soft_l1，f_scale=8 px")
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=8.0, xtol=1e-14, ftol=1e-14)
    p = sol.x
    pred, M1, k = model(p, tcp, h)
    res = np.linalg.norm(pred - tips, axis=1)
    Bx, By, s, phi, Cx, Cy, Z = p
    print(f"  底座像点(桌面高度) B=({Bx:.1f},{By:.1f})px   s={s:.1f} px/m")
    print(f"  图像方位 = FK方位 {np.degrees(phi):+.1f}°")
    print(f"  光轴脚像点 C=({Cx:.1f},{Cy:.1f})px   相机高 Z={Z*1000:.0f} mm")
    print(f"\n  {'#':>2} {'实测 tip':>16} {'预测 tip':>16} {'残差 px':>8}  预测在图内")
    for i in range(n):
        pi = (5 <= pred[i, 0] <= W - 5) and (5 <= pred[i, 1] <= H - 5)
        print(f"  {i:>2} ({tips[i,0]:>6.1f},{tips[i,1]:>6.1f}) ({pred[i,0]:>6.1f},{pred[i,1]:>6.1f})"
              f" {res[i]:>8.1f}  {'是' if pi else '否'}")
    print(f"\n  残差中位数 {np.median(res):.1f} px   均值 {res.mean():.1f} px")

    # 只看「预测在图内」的帧 —— 这些才是分割可能给出真尖端的帧
    inm = (pred[:, 0] > 5) & (pred[:, 0] < W - 5) & (pred[:, 1] > 5) & (pred[:, 1] < H - 5)
    print(f"  其中预测在图内的帧：{list(np.nonzero(inm)[0])}"
          f"  这些帧残差 {np.round(res[inm], 1).tolist()}")

    (CAL / "tip_camera_model.json").write_text(json.dumps({
        "note": "pred = C + k*(B + s*r*(cos(az+phi), sin(az+phi)) - C); k = Z/(Z-h)",
        "B_px": [Bx, By], "s_px_per_m": s, "phi_deg": float(np.degrees(phi)),
        "C_px": [Cx, Cy], "Z_mm": float(Z * 1000),
        "res_px": res.tolist(), "n": n,
    }, indent=2, ensure_ascii=False))
    print(f"[save] {CAL/'tip_camera_model.json'}")

    # 回看图：把预测点画在原始帧上
    import cv2
    FIG.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i in range(n):
        v = z[f"f{i}"].copy()
        cv2.drawMarker(v, tuple(np.round(tips[i]).astype(int)), (0, 0, 255),
                       cv2.MARKER_CROSS, 16, 2)                    # 红 = 自动分割
        ppx = tuple(np.round(pred[i]).astype(int))
        if 0 <= ppx[0] < W and 0 <= ppx[1] < H:
            cv2.drawMarker(v, ppx, (255, 120, 0), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(v, f"#{i} h={h[i]*1000:.0f} r={r[i]*1000:.0f} res={res[i]:.0f}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tiles.append(cv2.resize(v, (320, 240)))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, n, 3)]
    wand = max(r_.shape[1] for r_ in rows)
    rows = [np.pad(r_, ((0, 0), (0, wand - r_.shape[1]), (0, 0))) for r_ in rows]
    cv2.imwrite(str(FIG / "tip_model_check.png"), np.vstack(rows))
    print(f"[save] {FIG/'tip_model_check.png'}  ← 回看：蓝叉(预测) 是否压住夹爪尖端")
    return 0


if __name__ == "__main__":
    sys.exit(main())
