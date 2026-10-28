#!/usr/bin/env python3
"""拿**腕部相机**看标定板 —— 为手眼标定（板系 ↔ base 系）采数据。

为什么换这条路：A 路（用夹爪尖端 + up 相机）卡在**分割**上 —— 白色的臂压在浅色桌面上，
背景差分出来的掩码到处是洞，9 帧里只有 1 帧（臂竖直朝上那次）能可靠地分出尖端；
而且方位角到图像方向的映射自相矛盾（#0 与 #2 方位差 40°，图像里却差 167°），
说明掩码本身不可信。相比之下 **ChArUco 检测在 up 相机上已经做到 0.19 px 重投影**。

腕相机看板的好处：
- 检测的是**板**（高对比、结构化），不是臂的轮廓；
- 每个位姿直接给出 板→相机 的位姿（solvePnP），配合 FK 的 base→相机，
  就能解出 **板系 ↔ base 系**（`cv2.calibrateHandEye`）——**不需要任何视差模型**；
- 顺带把腕相机内参和**相机安装变换**解出来，后面都要用。

本轮只做第一步：**把腕相机转向桌面、看看能不能拍见板**。位姿由 IK 求：
把腕相机的**光轴（link 的 -z）指向桌面**，相机高度 ~250–350 mm，覆盖可达工作区。

安全：每个位姿先跑 `link_clearance`；夹爪命令夹在 `[GRIP_MIN, GRIP_MAX]`；
`--dry` 只算不走。

用法：
    python workplace/calibration/wrist_board_scan.py --dry
    python workplace/calibration/wrist_board_scan.py
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
from fk_check import TCP_LINK, CAM_LINK                            # noqa: E402

CAL = HERE / "calib"
FIG = HERE / "figures"
SHOTS = CAL / "wrist_shots"

# 腕部相机 = 带序列号那台（见 cam_id.py；up 是**不带**序列号那台）
WRIST_CAM = "camera_B"

H_TARGET = 0.30            # 相机离桌面目标高度（米）
MIN_CLEAR = 25.0           # 任何真实连杆离桌面的下限（mm）
MIN_TCP = 45.0             # 夹爪尖端离桌面下限（mm，米制换算见下）


def board_detectors():
    board = json.loads((HERE / "board" / "charuco_board.json").read_text())
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    sq = board["square_mm_nominal"] / 1000.0
    mk = board["marker_mm_nominal"] / 1000.0
    cb = cv2.aruco.CharucoBoard((board["squares_x"], board["squares_y"]), sq, mk, d)
    return cb, cv2.aruco.CharucoDetector(cb), board


def chain_min_mm(am: A.Arm, norm) -> float:
    return min(A.link_clearance(am.u, A.norm_to_q(norm, am.u)).values())


def plan(am: A.Arm, seed, n_want: int) -> list[dict]:
    """IK 扫一批「相机朝下、高度够」的位姿，再按位置/朝向**尽量散开**地挑。"""
    u = am.u
    cand: list[dict] = []
    grid = []
    for az in range(-180, 180, 45):
        for r in (0.10, 0.16, 0.22):
            for hz in (H_TARGET, H_TARGET + 0.06):
                grid.append((az, r, hz))
    # 朝向：竖直向下 + 四个方向各倾 30°（手眼标定**必须有转动多样性**）
    tilts = [(0.0, 0.0)]
    for taz in (0, 90, 180, 270):
        tilts.append((30.0, taz))

    for (az, r, hz), (tilt, taz) in itertools.product(grid, tilts):
        a = np.radians(az)
        p = np.array([r * np.cos(a), r * np.sin(a), hz + A.TABLE_Z])
        ta = np.radians(taz)
        # 光轴 = -z 绕水平轴倾斜 tilt 度，倒向 taz
        ax = np.array([np.sin(np.radians(tilt)) * np.cos(ta),
                       np.sin(np.radians(tilt)) * np.sin(ta),
                       -np.cos(np.radians(tilt))])
        try:
            s = A.solve_ik(u, seed, p, ax, tcp_min_z=MIN_TCP / 1000.0)
        except Exception:                                          # noqa: BLE001
            continue
        if s["cam_err_mm"] > 12 or s["axis_err_deg"] > 8:
            continue
        n = s["norm"]
        if chain_min_mm(am, n) < MIN_CLEAR:
            continue
        tcp = u.fk(A.norm_to_q(n, u), TCP_LINK)[:3, 3]
        if (tcp[2] - A.TABLE_Z) * 1000 < MIN_TCP:
            continue
        cand.append({"norm": n, "cam_mm": s["cam_pos"] * 1000, "axis": ax,
                     "tcp_mm": tcp * 1000, "min_mm": chain_min_mm(am, n),
                     "cam_err_mm": s["cam_err_mm"], "axis_err_deg": s["axis_err_deg"]})

    if not cand:
        raise SystemExit("IK 扫不出位姿 —— 放宽高度/倾角，或检查 seed")

    # 最远点贪心：(相机位置, 朝向) 一起散开
    F = np.array([[*c["cam_mm"] / 200.0, *c["axis"] * 1.5] for c in cand])
    pick = [int(np.argmax(np.linalg.norm(F - F.mean(0), axis=1)))]
    while len(pick) < min(n_want, len(cand)):
        d = np.min(np.linalg.norm(F[:, None, :] - F[pick][None, :, :], axis=2), axis=1)
        d[pick] = -1
        pick.append(int(np.argmax(d)))
    return [cand[i] for i in pick]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--poses", type=int, default=16)
    a = ap.parse_args()

    cb, det, _ = board_detectors()
    devs = resolve()
    if WRIST_CAM not in devs:
        raise SystemExit(f"找不到腕部相机 {WRIST_CAM} —— 没插？")
    print(f"[腕相机] {WRIST_CAM} → {devs[WRIST_CAM]}")

    am = A.Arm().connect()
    cap = None
    try:
        u = am.u
        seed = am.read_norm()
        print("[起点] " + "  ".join(f"{m}={seed[m]:+.1f}" for m in A.MOTORS))
        seed = dict(seed, gripper=float(np.clip(seed["gripper"], A.GRIP_MIN, A.GRIP_MAX)))

        poses = plan(am, seed, a.poses)
        print(f"\n[位姿计划] {len(poses)} 个（相机高度 {H_TARGET*1000:.0f}–"
              f"{(H_TARGET+0.06)*1000:.0f} mm，朝向含 0°/30° 倾角）")
        print(f"  {'相机 xyz (mm)':>26} {'光轴':>26} {'TCP 高':>7} {'最低':>6}")
        for p in poses:
            c, x = p["cam_mm"], p["axis"]
            print(f"  ({c[0]:>7.1f},{c[1]:>7.1f},{c[2]:>7.1f}) "
                  f"({x[0]:>+6.2f},{x[1]:>+6.2f},{x[2]:>+6.2f}) "
                  f"{p['tcp_mm'][2]:>6.1f} {p['min_mm']:>6.1f}")
        if a.dry:
            print("\n--dry：只算不动。")
            return 0

        cap = cv2.VideoCapture(str(devs[WRIST_CAM]))
        if not cap.isOpened():
            raise SystemExit("打不开腕部相机")
        SHOTS.mkdir(parents=True, exist_ok=True)
        for _ in range(20):
            cap.read()

        am.goto(seed, seconds=2.0)
        recs, hits = [], []
        for i, p in enumerate(poses):
            am.goto(p["norm"], seconds=2.5)
            time.sleep(0.8)
            fs = []
            for _ in range(6):
                ok, f = cap.read()
                if ok:
                    fs.append(f)
            if not fs:
                print(f"  {i+1:>2}/{len(poses)} 读不到帧"); continue
            img = fs[len(fs) // 2]
            ch, ci, m, mi = det.detectBoard(img)
            n = 0 if ch is None else len(ch)
            q = A.norm_to_q(am.read_norm(), u)
            cam = u.fk(q, CAM_LINK)
            f = SHOTS / f"pose{i:02d}_n{n:02d}.png"
            cv2.imwrite(str(f), img)
            recs.append({"i": i, "n_corners": n, "cam_mm": (cam[:3, 3] * 1000).tolist(),
                         "axis": (cam[:3, 2] * -1).tolist(),
                         "tcp_mm": (u.fk(q, TCP_LINK)[:3, 3] * 1000).tolist(),
                         "norm": {k: float(v) for k, v in p["norm"].items()},
                         "shot": f.name})
            if n >= 6:
                hits.append(i)
            print(f"  {i+1:>2}/{len(poses)} 角点 {n:>3}  "
                  f"相机({cam[0,3]*1000:>7.1f},{cam[1,3]*1000:>7.1f},{cam[2,3]*1000:>7.1f})mm"
                  f"  {'✓ 看见板' if n >= 6 else ''}")
        cap.release()

        (CAL / "wrist_scan.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False))
        print(f"\n[结果] {len(hits)}/{len(recs)} 个位姿看见板（≥6 角点）：{hits}")
        print(f"[save] {CAL/'wrist_scan.json'}   帧在 {SHOTS}/")

        # 总览图：肉眼看相机到底拍到了什么
        tiles = []
        for r in recs:
            v = cv2.imread(str(SHOTS / r["shot"]))
            cv2.putText(v, f"#{r['i']} corners={r['n_corners']}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 0) if r["n_corners"] >= 6 else (0, 0, 255), 2)
            tiles.append(cv2.resize(v, (320, 240)))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        w = max(r_.shape[1] for r_ in rows)
        rows = [np.pad(r_, ((0, 0), (0, w - r_.shape[1]), (0, 0))) for r_ in rows]
        cv2.imwrite(str(FIG / "wrist_scan.png"), np.vstack(rows))
        print(f"[save] {FIG/'wrist_scan.png'}  ← 回看")
    finally:
        if cap is not None:
            cap.release()
        am.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
