#!/usr/bin/env python3
"""**冻住姿势、用手推 —— 看「编码器说转了多少」和「相机看到转了多少」对不对得上。**

背景：板↔base 的手眼解卡在一个 ~4.8° 的、**随位姿变化**的 FK↔视觉不一致上，
常数 X 吸收不掉（X 是对各位姿的同一次左乘，制造不出随位姿变化的差）。已排除的：
  - 我代码的约定错（合成数据喂同一套残差 → 1e-14°，机器精度）；
  - 关节零位 / 增益（拟合出 +245°、−461° 这种非物理值也压不到 2.1°）；
  - 内参（焦距 ±2%、畸变 k1 ±20%、dist=0 都对这 4.78° 无影响）；
  - 单个关节的轴方向歪（最好 1.55°，且要 36° 的轴倾才够得着）；
  - 上相机里点底座（**上相机看不到底座**，视野 ~400mm，板就占满了）。

原理（不含 X、不含板）：手眼成立时，任意两帧的
    angle(R_ee_0ᵀ R_ee_t)  ≡  angle(R_cb_0ᵀ R_cb_t)
与 X 无关（共轭不改变转角）。这两个角**本来就必须相等**（静止时实测差 0.03°），
所以不相等的那一刻，就是下面某一条在发生：

  **A. 相机在腕 link 上没固定牢** —— 只推相机、别碰臂，此时 FK 应当纹丝不动：
       相机转 ≫ FK转（且松手不回零）⇒ A 成立。

  **B. 编码器读不到机构柔度 / 回差**（读到的角度 ≠ 真实输出角度）—— 只推臂：
       相机转 明显 ≫ FK转（>1°，且松手不回零）⇒ B 成立。

⚠️ **一次只测一件事**（`--what cam` 或 `--what arm`）。上一版把两段塞在一次运行里，
结果你还没等到横幅就错过了整段 —— 所以现在只有**一个开始信号**：
倒数结束、屏幕上出现「### 开始 ###」的那一刻起，就**持续用力**推，
直到出现「### 停 ###」。早推晚推都算数，但整段 15 秒都要在推。

⚠️ 冻住只覆盖这一二十秒，**退出前一律松开扭矩**（长时间顶着重力会把 shoulder_pan
顶到 Overheat，随后从总线掉，要断电重启）。推力以「看得见一点点位移」为准，别顶到堵转。
全程**不碰夹爪**。

用法：
    python workplace/calibration/push_test.py --what cam     # 只推相机
    python workplace/calibration/push_test.py --what arm     # 只推臂
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "so101_urdf"))
import arm as A                                                     # noqa: E402
import front_handeye as FH                                          # noqa: E402
import manual_capture as MC                                         # noqa: E402
from fk_check import Urdf, CAM_LINK                                 # noqa: E402

CAL = HERE / "calib"
MIN_CORNERS = 8
URDF = Urdf(A.URDF_PATH)
KD = json.loads((CAL / "front_intrinsics.json").read_text())
K, DIST = np.array(KD["K"]), np.array(KD["dist"])


def state(am, det, objp, cap):
    """采一帧：返回 (norm, R_ee, R_cb)。R_cb 为 None 表示这一帧看不到板。"""
    norm = am.read_norm()
    R_ee = URDF.fk(A.norm_to_q(norm, URDF), CAM_LINK)[:3, :3]
    fs = FH.grab(cap)
    if not fs:
        return norm, R_ee, None
    ch, ci, _, _ = det.detectBoard(fs[len(fs) // 2])
    if ch is None or len(ch) < MIN_CORNERS:
        return norm, R_ee, None
    ids = np.array(ci, int).reshape(-1)
    o = objp[ids].reshape(-1, 1, 3).astype(np.float32)
    im = np.array(ch, np.float32).reshape(-1, 1, 2)
    _, rv, _ = cv2.solvePnP(o, im, K, DIST, flags=cv2.SOLVEPNP_ITERATIVE)
    return norm, R_ee, cv2.Rodrigues(rv)[0]


def ang(Ref, Rn):
    return math.degrees(Rot.from_matrix(Ref.T @ Rn).magnitude())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["cam", "arm"],
                    help="cam=只推相机（测 A）；arm=只推臂（测 B）")
    ap.add_argument("--countdown", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=15.0)
    a = ap.parse_args()
    target = ("腕相机本身（别碰臂）" if a.what == "cam" else
              "臂 —— 就是【相机所在那截白色连杆】，或它下方/后方的小臂、肘部。"
              "**绝对别推夹爪**（夹爪电机顶到堵转会从总线上掉，要断电重启）")

    cb, det, _ = FH.board_setup()
    objp = FH.board_object_points(cb)
    am = A.Arm().connect()
    cap = None
    try:
        am.robot.bus.disable_torque()
        print("[扭矩] 已关 —— 臂是软的。脚本**不发任何运动指令**。")
        print(f"[准备] {a.countdown:.0f} 秒后冻住**当前**姿势。请先把相机对准标定板"
              f"（要 ≥{MIN_CORNERS} 个角点），摆好扶稳，数完倒数。")
        for k in range(int(a.countdown), 0, -1):
            print(f"        {k}…"); sys.stdout.flush(); time.sleep(1.0)

        MC.hold_here(am)
        cap, dev = MC.open_cam()
        if cap is None:
            raise SystemExit(f"打不开腕相机 {FH.FRONT_CAM}")

        n0, E0, B0 = state(am, det, objp, cap)
        if B0 is None:
            raise SystemExit("腕相机里看不到板 —— 请重摆姿势再跑")

        print("\n" + "#" * 78)
        print(f"### 开始 ###  现在就**持续用力**推 / 掰 / 按【{target}】，"
              f"一直推到看见「停」（{a.seconds:.0f} 秒）")
        print("#" * 78); sys.stdout.flush()

        t0 = time.time()
        worst = 0.0
        n_seen = 0
        while time.time() - t0 < a.seconds:
            n, E, B = state(am, det, objp, cap)
            if B is None:
                print("  （看不到板 —— 推得太狠，相机偏出视野了）"); sys.stdout.flush()
                continue
            dfk, dpnp = ang(E0, E), ang(B0, B)
            per = {m: n[m] - n0[m] for m in A.MOTORS}
            wj = max(per, key=lambda m: abs(per[m]))
            worst = max(worst, dpnp - dfk)
            n_seen += 1
            print(f"  t={time.time()-t0:5.1f}s  FK转 {dfk:6.2f}°   相机转 {dpnp:6.2f}°"
                  f"   差 {dpnp-dfk:+6.2f}°   (动最多 {wj[:11]} {per[wj]:+5.2f})"
                  + ("   <<<< 还在推吗？继续推、别松手 >>>>" if n_seen % 5 == 0 else ""))
            sys.stdout.flush()

        print("\n" + "#" * 78)
        print("### 停 ###  松手")
        print("#" * 78)
        print(f"\n⇒ {n_seen} 帧有效，最大「相机转 − FK转」= {worst:+.2f}°")
        if a.what == "cam":
            print("   判读（推相机段）：FK 本该纹丝不动。")
            print("     > 1°  ⇒ **A 成立：相机在腕 link 上没固定牢**，先锁死再标定。")
            print("     ≲ 0.3° ⇒ A 排除，相机是刚性固定的。")
        else:
            print("   判读（推臂段）：FK 和相机本该同步。")
            print("     > 1°  ⇒ **B 成立：编码器读不到机构柔度/回差**（读到 ≠ 真实输出角），"
                  "而且 FK 的误差会随姿态变 —— 这正是那 4.8° 的候选来源。")
            print("     ≲ 0.3° ⇒ B 排除。")
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
