#!/usr/bin/env python3
"""**不碰手眼**，只问一个问题：FK 和 PnP 的姿态不一致，能不能用「关节零位偏置」解释掉？

为什么这么问：板↔base 求解现在 rms 26mm，主要误差源不是条件数（雅可比条件数只有 7.3），
而是一个**X 吸收不掉**的姿态不一致 —— 用「X-无关不变量」量出来：任意两位姿的
`angle(R_ee_iᵀ R_ee_j)` 与 `angle(R_cb_iᵀ R_cb_j)` 差中位 2.25°、最大 6.5°。
X 是**同一个**常数矩阵，它对每个位姿是同一次左乘，所以**任何 X 都不可能**制造出
随位姿变化的差 —— 差只能来自 T_ee 一侧（FK）或 T_cb 一侧（PnP）。

PnP 那侧已经用三种算法 + jackknife + 内参扰动证明是稳的（0.2° 级）。
所以嫌疑落在 **norm → q** 这条映射上：我们刚用「标定实测行程」换掉了 URDF 限位
（修的是**增益**），但**零位**（norm=0 到底对应 URDF 的哪个角度）还是照抄
`mid(URDF限位)`。若 URDF 的零位与机械零位差一个常数，就产生一个**常量偏置**。

关键：常量偏置只有在**末关节**才被 X 吸收（那个等于重装相机）；中间关节的偏置
会随位姿改变整条链的姿态，**X 吸收不掉** —— 正是我们看到的这种不一致。

所以这里做纯诊断：拿 10 个位姿对的「角度差」当残差，拟合 5 个本体关节的零位偏置，
看能不能把中位 2.25° 压下去。**函数里没有 X、没有板、没有最小二乘手眼** ——
如果它能压到 0.2°，那结论是干净且无歧义的：偏置是真因。

只读离线，不碰臂。

用法：
    python workplace/calibration/joint_offset_probe.py
    python workplace/calibration/joint_offset_probe.py --gain    # 同时拟合增益
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                      # noqa: E402
import front_handeye as FH                                           # noqa: E402
from fk_check import Urdf, CAM_LINK                                  # noqa: E402

CAL = HERE / "calib"
BODY = A.BODY                                                        # 5 个本体关节


def load_pairs(source: str = "manual_capture.json"):
    """返回 (norms, 位姿对的 PnP 相对转角[度])。

    PnP 那侧照样用 `T_base_ee` 存的那份**不重要** —— 只用它的旋转，
    而我们这里刻意**不用**它，全从 `corners/ids` 重算 PnP，避免任何串味。
    """
    import cv2

    Kd = json.loads((CAL / "front_intrinsics.json").read_text())
    K, dist = np.array(Kd["K"]), np.array(Kd["dist"])
    cb, _, _ = FH.board_setup()
    objp = FH.board_object_points(cb)

    recs = [r for r in json.loads((CAL / source).read_text()) if r["n_corners"] >= 6]
    norms, Rcs, tags = [], [], []
    for r in recs:
        ids = np.array(r["ids"], int)
        o = objp[ids].reshape(-1, 1, 3).astype(np.float32)
        im = np.array(r["corners"], np.float32).reshape(-1, 1, 2)
        _, rv, _ = cv2.solvePnP(o, im, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        norms.append({m: float(r["norm"][m]) for m in A.MOTORS})
        Rcs.append(cv2.Rodrigues(rv)[0])
        tags.append(r["tag"])
    pairs = [(i, j) for i in range(len(recs)) for j in range(i + 1, len(recs))]
    ang = np.array([Rotation.from_matrix(Rcs[i].T @ Rcs[j]).magnitude()
                    for i, j in pairs])
    return norms, Rcs, tags, pairs, np.degrees(ang)


def fk_rots(u, norms, off_deg, gain):
    """按 (偏置, 增益) 重算每个位姿的 R_ee。gain=None 表示不动增益。"""
    out = []
    lim = u.limits()
    hc = A.calibrated_half_deg()
    for n in norms:
        q = {}
        for k, m in enumerate(BODY):
            lo, hi = (math.degrees(x) for x in lim[m])
            half = (hc[m] if (hc and m in hc) else (hi - lo) / 2)
            if gain is not None:
                half *= gain[k]
            deg = (lo + hi) / 2 + n[m] / 100.0 * half + off_deg[k]
            q[m] = math.radians(deg)
        q["gripper"] = A.norm_to_q(n, u)["gripper"]   # 不在链上，照抄
        out.append(u.fk(q, CAM_LINK)[:3, :3])
    return out


def residuals(p, u, norms, pairs, ang_pnp, n_off, gain):
    off = p[:n_off]
    g = None if gain is None else p[n_off:]
    Rs = fk_rots(u, norms, off, g)
    return np.array([math.degrees(Rotation.from_matrix(Rs[i].T @ Rs[j]).magnitude())
                     - ang_pnp[k] for k, (i, j) in enumerate(pairs)])


def fk_angles(u, norms, pairs, off, gain=None):
    """按给定 (偏置, 增益) 算出各位姿对的 FK 相对转角（度）。"""
    Rs = fk_rots(u, norms, off, gain)
    return np.degrees([Rotation.from_matrix(Rs[i].T @ Rs[j]).magnitude()
                       for i, j in pairs])


def run(u, norms, pairs, ang_pnp, use_gain: bool):
    n_off = len(BODY)
    p0 = np.r_[np.zeros(n_off), np.ones(n_off)] if use_gain else np.zeros(n_off)
    r0 = residuals(p0, u, norms, pairs, ang_pnp, n_off, np.ones(n_off) if use_gain else None)
    s = least_squares(residuals, p0, args=(u, norms, pairs, ang_pnp, n_off,
                                           np.ones(n_off) if use_gain else None),
                      method="lm", max_nfev=20000)
    r1 = residuals(s.x, u, norms, pairs, ang_pnp, n_off,
                   np.ones(n_off) if use_gain else None)
    tag = "偏置+增益" if use_gain else "只偏置"
    print(f"  [{tag}] 残差 rms {np.sqrt((r0**2).mean()):.2f}° → {np.sqrt((r1**2).mean()):.2f}°"
          f"   最大 {np.abs(r0).max():.2f}° → {np.abs(r1).max():.2f}°")
    print("     偏置 deg：" + "  ".join(f"{m}={v:+.2f}" for m, v in zip(BODY, s.x[:n_off])))
    if use_gain:
        print("     增益 ×：" + "  ".join(f"{m}={v:.4f}" for m, v in zip(BODY, s.x[n_off:])))
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="manual_capture.json")
    ap.add_argument("--gain", action="store_true", help="同时拟合各关节增益")
    a = ap.parse_args()

    u = Urdf(A.URDF_PATH)
    norms, _, tags, pairs, ang_pnp = load_pairs(a.source)
    print(f"[数据] {len(norms)} 个位姿（{', '.join(tags)}）⇒ {len(pairs)} 个位姿对")
    base = fk_angles(u, norms, pairs, np.zeros(5)) - ang_pnp
    print(f"[基线] 不加任何偏置：转角逐对之差的 rms "
          f"{np.sqrt((base**2).mean()):.2f}°  最大 {np.abs(base).max():.2f}°\n")

    run(u, norms, pairs, ang_pnp, use_gain=False)
    if a.gain or True:
        print()
        run(u, norms, pairs, ang_pnp, use_gain=True)
    print("\n判读：若残差被压到 ~0.2°（PnP 自身的噪声级），说明不一致**全部**来自这条映射，"
          "\n      而不在相机、不在 PnP、不在手眼。那就把这 5 个偏置落进 arm.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
