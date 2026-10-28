#!/usr/bin/env python3
"""**腕相机底座刚度测试** —— 回答「相机到底是刚性固定在腕 link 上，还是被一按就晃」。

为什么要问这个：板↔base 的手眼解卡在一个 ~4.8° 的、**随位姿变化**的姿态不一致上。
任何常数 X 都吸收不掉它（X 是对每个位姿的**同一次左乘**，制造不出随位姿变化的差），
所以要么是 X 本身不是常数，要么是 FK 那条链不对。

已经排除的（都实测过，写在 `joint_offset_probe.py` / 内参扰动里）：
  - 关节零位 / 增益：拟合出 +245°、−461° 这种非物理偏置，仍只从 4.78° 压到 2.15°；
  - 内参：焦距 ±2%、畸变 k1 ±20% 对这个 4.78° 几乎无影响（dist=0 反而更差）。

剩下两个活口：
    A. **相机在腕 link 上没固定牢** —— X 随位姿变，手眼方程根本无解；
    B. **FK 那条链的关节轴方向与真机不符**。

这个脚本一击分开：把臂**用扭矩冻住**（FK 在这段时间里是死的、一点不动），
然后用手去**按/推/掰腕相机** —— 只动相机、不动臂：
    图动了（>2px，且松开后不一定回原位）⇒ A 成立：先紧螺丝，标定别做了；
    图纹丝不动                                  ⇒ A 排除：问题在 FK（URDF 关节轴）。

⚠️ 冻住只为覆盖这十几秒：`hold_here` 之后**退出前一律松开扭矩**（长时间顶着重力
会把 shoulder_pan 顶到 Overheat，随后从总线掉，要断电重启）。

用法：
    python workplace/calibration/cam_mount_test.py            # 臂对着板，按提示操作
    python workplace/calibration/cam_mount_test.py --seconds 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                     # noqa: E402
import front_handeye as FH                                          # noqa: E402
import manual_capture as MC                                         # noqa: E402

CAL = HERE / "calib"
MIN_CORNERS = 8
THRESH_PX = 2.0        # 超过这个就是「相机真被推动了」；曝光抖动只有 ~0.7px


def detect(det, img):
    ch, ci, _, _ = det.detectBoard(img)
    if ch is None or len(ch) < MIN_CORNERS:
        return None
    return np.array(ci, int).reshape(-1), np.array(ch, float).reshape(-1, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0, help="冻住并持续拍摄的秒数")
    ap.add_argument("--countdown", type=float, default=8.0, help="摆姿势的倒数秒数")
    a = ap.parse_args()

    cb, det, _ = FH.board_setup()
    am = A.Arm().connect()
    cap = None
    try:
        am.robot.bus.disable_torque()
        print("[扭矩] 已关 —— 臂是软的。脚本**不发任何运动指令**，不会自己动。")
        # 不能用 input()：这个脚本是后台跑的（没有 TTY）。改成倒数，留出摆姿势的时间。
        print(f"[准备] {a.countdown:.0f} 秒后把**当前**姿势冻住 —— "
              f"请把相机对准标定板，摆好、扶稳（手别停在相机上，等倒数结束再按）。")
        for k in range(int(a.countdown), 0, -1):
            print(f"        {k}…"); sys.stdout.flush()
            time.sleep(1.0)

        norm = MC.hold_here(am)
        print("[冻住] 姿势已锁定，这次 FK 在下面的测试里是**常数**。")

        cap, dev = MC.open_cam()
        if cap is None:
            raise SystemExit(f"打不开腕相机 {FH.FRONT_CAM}")
        fs = FH.grab(cap)
        if not fs:
            raise SystemExit("腕相机读不到帧")
        ref = detect(det, fs[len(fs) // 2])
        if ref is None:
            raise SystemExit(f"腕相机里看不到板（要 ≥{MIN_CORNERS} 个角点）—— 请重摆姿势")

        ids0, c0 = ref
        print(f"[基准] {len(ids0)} 个角点   质心 ({c0[:,0].mean():.1f},{c0[:,1].mean():.1f}) px")
        print(f"\n>>> 现在请用手去**按 / 推 / 掰腕相机**（只动相机，别动臂），持续 "
              f"{a.seconds:.0f} 秒 <<<\n")
        sys.stdout.flush()

        t0 = time.time()
        worst = 0.0
        n = 0
        while time.time() - t0 < a.seconds:
            fs = FH.grab(cap)
            if not fs:
                continue
            got = detect(det, fs[len(fs) // 2])
            if got is None:
                print("  （这一帧看不到板 —— 相机被推歪出视野了？）"); sys.stdout.flush()
                continue
            ids, c = got
            m = {i: p for i, p in zip(ids, c)}
            both = [i for i in ids0 if i in m]
            if len(both) < MIN_CORNERS:
                continue
            d = np.array([m[i] for i in both]) - np.array(
                [c0[list(ids0).index(i)] for i in both])
            sh = np.median(d, axis=0)
            mag = float(np.linalg.norm(sh))
            worst = max(worst, mag)
            n += 1
            bar = "█" * int(min(mag, 40))
            print(f"  t={time.time()-t0:5.1f}s  质心偏移 ({sh[0]:+7.2f},{sh[1]:+7.2f}) px"
                  f"  |{mag:6.2f}| px  {bar}")
            sys.stdout.flush()

        print(f"\n⇒ {n} 帧，最大质心偏移 {worst:.2f} px")
        if worst > THRESH_PX:
            print("   判读：**相机被推动了** —— A 成立。相机在腕 link 上没固定牢，"
                  "X 不是常数，手眼方程本身无解。先把它锁死/加支架，再重走标定。")
        else:
            print("   判读：纹丝不动 —— A 排除。相机是刚性固定的，那 4.8° 只能来自 FK "
                  "(B)：URDF 的关节轴方向与真机不符。往那边查。")
        print("   （另外：松手后如果偏移没回到 0，说明底座有回差/滑移，同样是 A。）")
    finally:
        if cap is not None:
            cap.release()
        try:
            am.robot.bus.disable_torque()
        except Exception:                                          # noqa: BLE001
            pass
        am.disconnect(keep_torque=True)
        print("[松开] 扭矩已关 —— 臂变软了，扶住。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
