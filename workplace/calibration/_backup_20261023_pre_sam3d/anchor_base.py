#!/usr/bin/env python3
"""把**板坐标系**对到**机械臂 base 系** —— 走 A：动臂，用夹爪尖端当标定物。

原理（为什么这能行）：两套坐标各自都已知一部分，缺的是它们之间的刚体变换。

    板系  --H-->  像素        （up 单应，已有：calib/up_homography.json）
    像素  <--投影--  桌面上的点
    base 系 --FK-->  夹爪尖端     （URDF + 标定，已有：so101_urdf/）

把夹爪尖端支到桌面上的若干位置，尖端**就是**同一个物理点：base 侧由 FK 给出，
板系侧由 up 单应给出，两两配对就能解出 板系→base 的刚体变换。

⚠️ **视差**：尖端离桌面高 `h` 时，单应把它投到的**桌面点**比它正下方的点偏出
`X·h/Z`（X = 离光轴脚的距离，Z = 相机离桌面高）。所以用带视差的模型：

    X_obs = C + k·(R(θ)·P_base + T − C),   k = Z/(Z−h)

未知 6 个（θ、T(2)、C(2)、Z）。**h 必须有 ≥2 个不同取值**才能把 C 和 Z 分开，
否则只能定出「绕 C 的相似变换」，尺度是错的。所以位姿集要**故意打散高度**。

tip detection：夹爪合拢 → 尖端是一个点。背景 = 所有帧的**逐像素中位数**（臂在
动、别的都不动，所以中位数 = 背景）。帧间自动曝光会漂，先按中位数做**增益对齐**。

安全：每个位姿先跑 `link_clearance`，任何真实连杆离桌面 < `--min-clear` 就中止；
`--dry` 只算不走。

用法：
    python workplace/calibration/anchor_base.py --dry
    python workplace/calibration/anchor_base.py --poses 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                    # noqa: E402
from cam_id import resolve                                         # noqa: E402
from fk_check import TCP_LINK                                      # noqa: E402

FIG = HERE / "figures"
CAL = HERE / "calib"
HOMO = CAL / "up_homography.json"

CHAIN = ["shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link",
         "gripper_link", "wrist_camera_mount_link", "wrist_camera_link"]

H_MIN, H_MAX = 18.0, 145.0          # 尖端高度窗口（mm，相对桌面）
REC = 8                             # 每个位姿记几帧取中位数


# ------------------------------------------------------------------ 工具

def warmup(cap, n=40):
    for _ in range(n):
        cap.read()


def grab(cap, n=REC):
    for _ in range(3):
        cap.read()
    fs = []
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            fs.append(f.astype(np.float32))
    return np.median(np.stack(fs), 0)          # (480,640,3) float


def gain_correct(f: np.ndarray, ref_med: float) -> np.ndarray:
    m = float(np.median(f))
    return f * (ref_med / m) if m > 1 else f


# ------------------------------------------------------------------ 位姿规划

def plan_poses(u, seed, n_want, min_clear) -> list[dict]:
    """在 (pan, elbow, wrist_flex) 上扫出「尖端够低、连杆够高」的位姿，
    再用最远点贪心挑一组**在 (x,y,h) 上铺得最开**的 —— 高度必须打散，见模块注释。"""
    cand = []
    for pan, el, wf in itertools.product(range(-100, 101, 10),
                                         range(-30, 111, 10),
                                         range(-100, 101, 20)):
        n = dict(seed, shoulder_pan=float(pan), elbow_flex=float(el),
                 wrist_flex=float(wf))
        q = A.norm_to_q(n, u)
        tcp = u.fk(q, TCP_LINK)[:3, 3]
        h = (tcp[2] - A.TABLE_Z) * 1000
        zs = [(u.fk(q, l)[2, 3] - A.TABLE_Z) * 1000 for l in CHAIN]
        low, high = min(zs), max(zs)
        # 上界是**安全约束**：up 相机就吊在桌面上方不远，臂抬太高会撞到
        if not (H_MIN <= h <= H_MAX and low >= min_clear and high <= 150.0):
            continue
        r = float(np.hypot(tcp[0], tcp[1]))
        # 半径窗口：太小尖端被臂自己挡住；太大跑到画面外、且离标定板覆盖区太远
        # （单应是外推的，越远越不可信）
        if not (0.14 <= r <= 0.24):
            continue
        cand.append({"norm": n, "tcp_mm": tcp * 1000, "h_mm": h, "min_mm": low})
    if not cand:
        raise SystemExit("扫不出满足高度/间隙的位姿 —— 放宽 H_MIN/H_MAX 或 min-clear")

    # 最远点贪心：特征 = (x, y, h/3) —— 高度权重调低但非零
    F = np.array([[c["tcp_mm"][0] / 300, c["tcp_mm"][1] / 300, c["h_mm"] / 100]
                  for c in cand])
    pick = [int(np.argmax(np.linalg.norm(F - F.mean(0), axis=1)))]
    while len(pick) < min(n_want, len(cand)):
        d = np.min(np.linalg.norm(F[:, None, :] - F[pick][None, :, :], axis=2), axis=1)
        d[pick] = -1
        pick.append(int(np.argmax(d)))
    return [cand[i] for i in pick]


# ------------------------------------------------------------------ 尖端分割

def arm_mask(f: np.ndarray, bg: np.ndarray, thr: float) -> np.ndarray:
    d = np.abs(f - bg).max(-1)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


def pick_arm_component(m: np.ndarray) -> np.ndarray:
    """臂总是从画面**下边缘**进来 —— 取下边缘上最大的那个连通域。"""
    nl, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    best, best_area = 0, -1
    for k in range(1, nl):
        x, y, w, h, area = st[k]
        if y + h >= m.shape[0] - 1 and area > best_area:
            best, best_area = k, area
    return (lab == best).astype(np.uint8) if best else np.zeros_like(m)


# ------------------------------------------------------------------ 求解

def umeyama_rigid(P: np.ndarray, X: np.ndarray):
    """无尺度的 2D 刚体拟合：X ≈ R P + T。返回 (θ, T)。"""
    cp, cx = P.mean(0), X.mean(0)
    H = (P - cp).T @ (X - cx)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return float(np.arctan2(R[1, 0], R[0, 0])), cx - R @ cp


def fit_parallax(P: np.ndarray, h: np.ndarray, X: np.ndarray, init):
    """解 6 参数模型 X_obs = C + k(R(θ)P + T − C)，k = Z/(Z−h)。"""
    from scipy.optimize import least_squares
    th0, T0 = init
    Z0 = 0.25

    def unpack(p):
        th, tx, ty, cx, cy, Z = p
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return R, np.array([tx, ty]), np.array([cx, cy]), Z

    def resid(p):
        R, T, C, Z = unpack(p)
        k = Z / (Z - h)
        return ((C + (k * ((P @ R.T) + T - C).T).T) - X).ravel()

    p0 = [th0, T0[0], T0[1], X.mean(0)[0], X.mean(0)[1], Z0]
    sol = least_squares(resid, p0, xtol=1e-14, ftol=1e-14)
    R, T, C, Z = unpack(sol.x)
    pred = C + (Z / (Z - h) * ((P @ R.T) + T - C).T).T
    return sol.x, pred, np.linalg.norm(pred - X, axis=1)


# ------------------------------------------------------------------ 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--poses", type=int, default=9)
    ap.add_argument("--min-clear", type=float, default=25.0)
    ap.add_argument("--thr", type=float, default=35.0, help="背景差分阈值（0-255）")
    a = ap.parse_args()

    cal = json.loads(HOMO.read_text())
    Hmat = np.array(cal["H_board_to_px"])
    Hinv = np.array(cal["Hinv_px_to_board"])
    print(f"[单应] {HOMO.name}  重投影 {cal['mean_reproj_px']:.2f} px  "
          f"尺度 {cal['square_m']*1000:.3f} mm/格")

    am = A.Arm().connect()
    cap = None
    try:
        u = am.u
        seed = am.read_norm()
        print("[起点] " + "  ".join(f"{m}={seed[m]:+.1f}" for m in A.MOTORS))

        poses = plan_poses(u, dict(seed, gripper=A.GRIP_MIN), a.poses, a.min_clear)
        poses.sort(key=lambda p: np.degrees(np.arctan2(p["tcp_mm"][1], p["tcp_mm"][0])))
        print(f"\n[位姿计划] {len(poses)} 个（高度故意打散）：")
        print(f"  {'pan':>6}{'elbow':>7}{'wflex':>7}   {'TCP xy (mm)':>22} {'高':>7} {'最低':>7}")
        for p in poses:
            n, t = p["norm"], p["tcp_mm"]
            print(f"  {n['shoulder_pan']:>6.0f}{n['elbow_flex']:>7.0f}{n['wrist_flex']:>7.0f}"
                  f"   ({t[0]:>8.1f},{t[1]:>8.1f}) {p['h_mm']:>6.1f} {p['min_mm']:>6.1f}")
        hs = [p["h_mm"] for p in poses]
        print(f"  高度跨度 {min(hs):.0f} – {max(hs):.0f} mm"
              f"  ({'✓ 能把 C 和 Z 分开' if max(hs)-min(hs) > 40 else '⚠ 太窄，尺度可能定不住'})")
        if a.dry:
            print("\n--dry：只算不动。")
            return 0

        devs = resolve()
        cap = cv2.VideoCapture(str(devs["camera_A"]))
        if not cap.isOpened():
            raise SystemExit("打不开 up 相机")
        print("\n[相机] 预热…")
        warmup(cap)
        ref_med = float(np.median(grab(cap, 6)))

        # 夹爪收到**几乎**全闭（不是 0：会堵转 → Overload → 电机掉线，实测踩过）
        # 尖端位置由 gripper_frame_link 决定，与夹爪开合无关；只想让尖端形状是个点
        am.goto(dict(seed, gripper=A.GRIP_MIN), seconds=2.0)
        time.sleep(0.5)

        shots = []
        for i, p in enumerate(poses):
            am.goto(p["norm"], seconds=2.5)
            time.sleep(0.8)
            f = grab(cap)
            q = A.norm_to_q(am.read_norm(), u)
            tcp = u.fk(q, TCP_LINK)[:3, 3]
            shots.append({"f": gain_correct(f, ref_med), "raw": f.astype(np.uint8),
                          "tcp_mm": tcp * 1000, "h_mm": (tcp[2] - A.TABLE_Z) * 1000,
                          "norm": p["norm"]})
            print(f"   {i+1}/{len(poses)} pan={p['norm']['shoulder_pan']:+5.0f} "
                  f"实测 TCP=({tcp[0]*1000:+7.1f},{tcp[1]*1000:+7.1f},{tcp[2]*1000:+6.1f})mm"
                  f"  h={(tcp[2]-A.TABLE_Z)*1000:5.1f}mm")

        # ---- 背景 = 逐像素中位数（臂在动）----
        bg = np.median(np.stack([s["f"] for s in shots]), 0)

        # ---- 每个位姿分割出臂，取尖端 ----
        masks = [pick_arm_component(arm_mask(s["f"], bg, a.thr)) for s in shots]
        inter = np.logical_and.reduce([m > 0 for m in masks])
        ys, xs = np.nonzero(inter)
        base_ref = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([320., 470.])
        print(f"\n[分割] 各帧臂面积 " + " ".join(f"{int(m.sum())}" for m in masks))
        print(f"  恒定覆盖区质心（≈底座方向）({base_ref[0]:.0f},{base_ref[1]:.0f})")

        tips, vis_list = [], []
        for s, m in zip(shots, masks):
            yy, xx = np.nonzero(m)
            if len(xx) < 50:
                print("  ⚠ 某个位姿分割出的臂太小，跳过")
                continue
            d = np.hypot(xx - base_ref[0], yy - base_ref[1])
            k = int(np.argmax(d))
            tip_px = np.array([xx[k], yy[k]], float)
            tip_bd = cv2.perspectiveTransform(
                tip_px.reshape(1, 1, 2).astype(np.float32), Hinv)[0, 0]
            tips.append({"tip_px": tip_px, "tip_board_m": tip_bd.astype(float),
                         "tcp_mm": s["tcp_mm"], "h_mm": s["h_mm"], "mask": m})
            v = s["raw"].copy()
            v[m > 0] = (0.5 * v[m > 0] + 0.5 * np.array([0, 180, 0])).astype(np.uint8)
            cv2.drawMarker(v, tuple(np.round(tip_px).astype(int)), (0, 0, 255),
                           cv2.MARKER_CROSS, 18, 2)
            vis_list.append(v)

        if len(tips) < 4:
            raise SystemExit(f"只分出了 {len(tips)} 个尖端 —— 不够解，看 figures/anchor_tips.png")

        P = np.array([t["tcp_mm"][:2] for t in tips]) / 1000.0     # (N,2) base 米
        h = np.array([t["h_mm"] for t in tips]) / 1000.0           # (N,) 米
        X = np.array([t["tip_board_m"] for t in tips])             # (N,2) 板系米

        th, T = umeyama_rigid(P, X)
        rigid_res = np.linalg.norm((P @ np.array([[np.cos(th), np.sin(th)],
                                                  [-np.sin(th), np.cos(th)]]).T + T) - X,
                                   axis=1)
        print(f"\n[无视差刚体拟合] θ={np.degrees(th):+.2f}°  T=({T[0]*1000:+.1f},{T[1]*1000:+.1f}) mm")
        print(f"  残差：平均 {rigid_res.mean()*1000:.2f} mm，最大 {rigid_res.max()*1000:.2f} mm")

        np.savez_compressed(
            CAL / "anchor_frames.npz", bg=bg.astype(np.uint8),
            masks=np.stack(masks), tcp=np.array([t["tcp_mm"] for t in tips]),
            h=np.array([h_ for h_ in h]), tips=np.array([t["tip_px"] for t in tips]),
            **{f"f{i}": st["raw"] for i, st in enumerate(shots)})
        print(f"[save] {CAL/'anchor_frames.npz'}（帧+掩码，供离线调分割）")

        p6, pred, res6 = fit_parallax(P, h, X, (th, T))
        th6, tx, ty, cx, cy, Z = p6
        print(f"\n[带视差 6 参数] θ={np.degrees(th6):+.2f}°  T=({tx*1000:+.1f},{ty*1000:+.1f}) mm")
        print(f"  光轴脚 C=({cx*1000:+.1f},{cy*1000:+.1f}) mm  相机高 Z={Z*1000:.1f} mm")
        print(f"  残差：平均 {res6.mean()*1000:.2f} mm，最大 {res6.max()*1000:.2f} mm")

        use = p6 if res6.mean() < rigid_res.mean() else np.array([th, *T, 0, 0, 1e9])
        tag = "parallax" if res6.mean() < rigid_res.mean() else "rigid"
        print(f"  ⇒ 采用 **{tag}**（残差更小）")

        R6 = np.array([[np.cos(use[0]), -np.sin(use[0])], [np.sin(use[0]), np.cos(use[0])]])
        out = {
            "model": tag, "theta_deg": float(np.degrees(use[0])),
            "T_mm": [float(use[1] * 1000), float(use[2] * 1000)],
            "C_mm": [float(use[3] * 1000), float(use[4] * 1000)],
            "Z_mm": float(use[5] * 1000) if np.isfinite(use[5]) else None,
            "R_board_to_base": R6.tolist(),
            "mean_res_mm": float(res6.mean() * 1000 if tag == "parallax" else rigid_res.mean() * 1000),
            "max_res_mm": float(res6.max() * 1000 if tag == "parallax" else rigid_res.max() * 1000),
            "n_points": len(tips),
            "points": [{"tcp_mm": t["tcp_mm"], "h_mm": t["h_mm"],
                        "tip_px": [float(v) for v in t["tip_px"]],
                        "tip_board_m": t["tip_board_m"]} for t in tips],
            "homography": str(HOMO),
        }
        CAL.mkdir(parents=True, exist_ok=True)
        (CAL / "tip_anchor.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

        # ---- 回看图 ----
        vis = np.vstack([np.hstack(vis_list[i:i + 3])
                         for i in range(0, len(vis_list), 3)]) if vis_list else None
        if vis is not None:
            FIG.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(FIG / "anchor_tips.png"), vis)
            print(f"\n[save] {FIG/'anchor_tips.png'}  ← **必须肉眼回看**：红十字要落在夹爪尖端上")
        print(f"[save] {CAL/'tip_anchor.json'}")
    finally:
        if cap is not None:
            cap.release()
        am.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
