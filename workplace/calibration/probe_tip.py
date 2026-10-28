#!/usr/bin/env python3
"""探针：把夹爪尖端扫过桌面，看它在 `up` 图里**能不能自动分出来**。

为什么先探针再写锚定脚本：锚定要的是「图像里夹爪尖端的像素」↔「base 系里
TCP 的位置」这一对对应对。base 侧由 FK 白送，图像侧要靠分割。分不出来的话
整个 A 路（动臂锚定）就不成立，得换路。

安全：**只动 shoulder_pan**（绕竖直轴转），所有连杆高度完全不变；并且对每个
位姿先跑一遍 `link_clearance`，任何一个真实连杆离桌面 < `--min-clear` 就中止。

用法：
    python workplace/calibration/probe_tip.py --dry        # 只算，不动
    python workplace/calibration/probe_tip.py              # 真走
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                # noqa: E402
from cam_id import CAMS, resolve                               # noqa: E402
from fk_check import TCP_LINK, CAM_LINK                        # noqa: E402

CHAIN = ["shoulder_link", "upper_arm_link", "lower_arm_link",
         "wrist_link", "gripper_link", "wrist_camera_mount_link", "wrist_camera_link"]
FIG = HERE / "figures"


def chain_min_mm(u, q) -> float:
    return min((u.fk(q, l)[2, 3] - A.TABLE_Z) * 1000 for l in CHAIN)


def grab(cap, n=6):
    for _ in range(3):
        cap.read()
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            out.append(f)
    return np.median(np.stack(out), 0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--min-clear", type=float, default=25.0, help="最低连杆离桌面下限 mm")
    ap.add_argument("--pans", type=float, nargs="*", default=[80, 40, 0, -40, -80])
    a = ap.parse_args()

    devs = resolve()
    arm = A.Arm().connect()
    try:
        u = arm.u
        cur = arm.read_norm()
        print(f"\n[起点] " + "  ".join(f"{m}={cur[m]:+.1f}" for m in A.MOTORS))
        print(f"  {'pan':>7} {'TCP xyz (mm)':>28} {'链最低':>8}  安全")
        plan = []
        for pan in a.pans:
            n = dict(cur, shoulder_pan=float(pan))
            q = A.norm_to_q(n, u)
            tcp = u.fk(q, TCP_LINK)[:3, 3]
            low = chain_min_mm(u, q)
            ok = low >= a.min_clear
            print(f"  {pan:>7.1f} ({tcp[0]*1000:>7.1f},{tcp[1]*1000:>7.1f},{tcp[2]*1000:>7.1f}) "
                  f"{low:>7.1f}mm  {'✓' if ok else '✗ 中止'}")
            if not ok:
                raise SystemExit(f"pan={pan} 会让连杆离桌面只剩 {low:.1f} mm，中止")
            plan.append((pan, n))

        if a.dry:
            print("\n--dry：只算不动。")
            return 0

        cap = cv2.VideoCapture(str(devs["camera_A"]))
        if not cap.isOpened():
            raise SystemExit("打不开 up 相机")
        FIG.mkdir(parents=True, exist_ok=True)
        shots = []
        for pan, n in plan:
            arm.goto(n, seconds=2.5)
            got = arm.read_norm()
            q = A.norm_to_q(got, u)
            tcp = u.fk(q, TCP_LINK)[:3, 3]
            time.sleep(0.5)
            img = grab(cap)
            p = FIG / f"probe_tip_pan{pan:+.0f}.png"
            cv2.imwrite(str(p), img)
            shots.append((pan, tcp, img, p))
            print(f"  pan={pan:+6.1f}  实测 TCP=({tcp[0]*1000:+7.1f},"
                  f"{tcp[1]*1000:+7.1f},{tcp[2]*1000:+7.1f})mm  -> {p.name}")
        cap.release()

        # 拼一张总览图，供肉眼回看
        tiles = []
        for pan, tcp, img, _ in shots:
            t = img.copy()
            cv2.putText(t, f"pan {pan:+.0f}  TCP({tcp[0]*1000:+.0f},{tcp[1]*1000:+.0f})",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            tiles.append(cv2.resize(t, (320, 240)))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        montage = np.vstack(rows)
        cv2.imwrite(str(FIG / "probe_tip_montage.png"), montage)
        print(f"\n[save] {FIG/'probe_tip_montage.png'}  ← 回看：夹爪尖端在哪、能不能分出来")

        # 顺手记一下 FK，供下一步用
        (HERE / "probe_tip.json").write_text(__import__("json").dumps(
            [{"pan": pan, "tcp_mm": (tcp * 1000).tolist()} for pan, tcp, _, _ in shots],
            indent=2))
    finally:
        arm.disconnect(keep_torque=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
