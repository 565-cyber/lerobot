#!/usr/bin/env python3
"""**手动摆位 + 自动采集** —— 给手眼标定（板 ↔ base）攒数据，臂全程不需要自己动。

为什么换成手动摆：脚本驱动的 16 个位姿里，视点半径最大到 391 mm，臂会伸得很直；
而且转运守卫正好把**唯一带大倾角的那 3 个位姿**砍掉了（倾角 48.7°），剩下 16 个
倾角只有 2.8–11.9° —— 相对转轴几乎全平行于竖直轴，手眼方程 AX=XB **退化**，白跑。
手动摆就没这个问题：摆的是「朝向多样」，而**位姿精度根本不重要** —— 精度由 FK 读
实际关节角保证，臂是怎么到那儿的无所谓。

关键机制（这是它安全的原因）：
- STS3215 是**绝对编码器**，`Present_Position` **在扭矩关闭时照样能读**。
- 所以：连上后**显式关掉扭矩**，臂是软的、你用手摆；我只读关节 + 抓图。
- 默认模式**一条运动指令都不发** —— 臂在物理上不可能自己动，和 `goto` 那条路
  彻底分开，不存在「伸太直」的风险。

`--freeze`（软臂扶不住时必须开）：
臂是软的又伸在外面时，自重会让它**持续下沉**（实测 `elbow_flex` 0.28s 内走 1.9 个
归一化单位 ≈ 6.8/s），手扶着也压不住，采集窗口里的漂移就污染 FK。开了 `--freeze`
就在采集瞬间用扭矩把姿势定住（先写 Goal_Position=当前位置再开扭矩，见 `hold_here`），
漂移归零、也能松手。
⚠️ 冻住之后臂是**硬的**，要动它得先发 `soft` 松开 —— 松开前**先把手放上去**。

跑法（脚本常驻，靠一个命令文件触发采集）：
    python workplace/calibration/manual_capture.py --freeze &
    echo v1   > /tmp/mcap_cmd     # 采一张，标签 v1
    echo soft > /tmp/mcap_cmd     # 松开扭矩（臂变软，先扶住）
    echo quit > /tmp/mcap_cmd     # 收工

每采一张都会：读 6 个关节 → FK 出 相机/TCP 位姿 → 抓腕相机帧 → 检测板 →
报「看到几个角点 / 相机倾角 / 离桌面多高」，并存帧 + 增量写 JSON。

安全：不发力矩、不发指令；夹爪一律不碰（不命令它）。

**越界闸（每次采集前，无条件开）**：读**原始计数**，任一关节落在标定区间外就
**拒绝采集**并提示断电重上电。没有它的话 lerobot 会把越界读数**静默夹成 ±100**，
`hold_here()` 再把这个假值当目标写回去 → 上扭矩的瞬间关节被甩向区间端点
（实测 `wrist_roll` 被甩 199°、顶死发热，STS3215 会 Overload 甚至掉线）。
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                    # noqa: E402
import front_handeye as FH                                         # noqa: E402
from cam_id import resolve                                         # noqa: E402
from fk_check import TCP_LINK, CAM_LINK                            # noqa: E402

CMD_FILE = Path("/tmp/mcap_cmd")
FREEZE = False                 # main() 里按 --freeze 覆盖
OUT = HERE / "calib" / "manual_capture.json"
SHOTS = HERE / "calib" / "manual_shots"

# 标定 json 里的**原始计数**区间 —— 判越界只认真实数值，不认 norm
CAL_RANGES = {m: (int(c["range_min"]), int(c["range_max"]))
              for m, c in json.loads(A.FOLLOWER_CAL.read_text()).items()}


def raw_out_of_range(am) -> dict[str, tuple[int, int, int]]:
    """读**原始计数**，挑出落在标定区间外的关节。返回 {关节: (当前, 下界, 上界)}。

    ⚠️ 为什么不能拿 `read_norm()` 判：lerobot 的 `_normalize` 会把越界读数
    **静默夹成 ±100**（`bounded_val = min(max_, max(min_, val))`），几度到几百度的
    误差一起消失，而且**不报错**。只有原始计数是可信的。

    为什么非得判在前面：`hold_here()` 会把夹过的 norm 当目标写回去，一
    `enable_torque()` 关节就朝区间端点冲。实测 `wrist_roll` 被甩 199°、顶死发热，
    STS3215 持续堵转 → Overload → **掉线**（同 `arm.py:goto` 里夹爪那次）。

    判据是**完备**的：区间宽 3879 < 4096，所以「多转/少转一整圈」的偏移必定落在
    区间外（`[181,4060] ± 4096` 与自身完全不相交），这个故障模式一个都不会漏。
    """
    raw = am.robot.bus.sync_read("Present_Position", normalize=False)
    return {m: (int(raw[m]), *CAL_RANGES[m])
            for m in A.MOTORS
            if not (CAL_RANGES[m][0] <= int(raw[m]) <= CAL_RANGES[m][1])}


def hold_here(am, settle: float = 1.0):
    """把**当前**姿势冻住，返回稳住之后的关节读数。

    ⚠️ 顺序很重要：必须先 `send_action(当前位置)` 把 Goal_Position 写成**现在**的位置，
    再 `enable_torque()`。反过来的话，舵机一上扭矩就会朝 EEPROM 里残留的**上次目标**
    冲过去 —— 那是别的姿势，会突然甩一大下。

    冻住之后：位姿稳、可以松手，采集窗口里的漂移归零；FK 用的是**稳住之后**重读的值，
    所以「冻住时往回拉的那几度」也不会污染标定。
    """
    norm = am.read_norm()
    am.robot.send_action({f"{m}.pos": norm[m] for m in A.MOTORS})
    am.robot.bus.enable_torque()
    time.sleep(settle)
    return am.read_norm()


def rot_axis(R):
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(v)
    return (None, 0.0) if s < 1e-9 else (v / s, math.degrees(math.atan2(s, np.trace(R) - 1)))


def diversity(recs):
    """已采位姿的**转动多样性** —— 手眼标定能不能解出来就看这个。

    判据（见 front_handeye.py 顶部的推导）：两两相对转动的**转轴**要张成三维。
    全都只绕竖直轴转（即只换方位角、不换倾角）⇒ 退化成一条线，解不出来。
    返回 (转轴各向最小占比, 转轴离竖直的最大夹角, 有效对数)。经验阈值：(>0.02, >25°)。
    ⚠️ 对数 <3 时占比**没有意义**（1 个轴张不成三维），调用方必须按对数分别措辞，
    否则会出现「v1 明明有 56° 大倾角、却被报成缺大倾角」这种误导。
    """
    Rs = [np.array(r["T_base_ee"])[:3, :3] for r in recs]
    axes = []
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            ax, an = rot_axis(Rs[i].T @ Rs[j])
            if ax is not None and an > 5.0:
                axes.append(ax)
    if not axes:
        return 0.0, 0.0, 0
    axes = np.array(axes)
    ang_z = max(math.degrees(math.acos(min(1, abs(a[2])))) for a in axes)
    if len(axes) < 3:
        return 0.0, ang_z, len(axes)
    ev = np.linalg.eigvalsh(axes.T @ axes / len(axes))
    return ev[0] / ev.sum(), ang_z, len(axes)


def main() -> int:
    global FREEZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true",
                    help="采集时先把当前姿势用扭矩冻住（默认关：只读不发力）")
    ap.add_argument("--settle", type=float, default=1.0, help="冻住后等稳的秒数")
    a = ap.parse_args()
    FREEZE = a.freeze

    cb, det, _ = FH.board_setup()
    devs = resolve()
    if FH.FRONT_CAM not in devs:
        raise SystemExit(f"找不到腕相机 {FH.FRONT_CAM}")
    print(f"[腕相机] {FH.FRONT_CAM} → {devs[FH.FRONT_CAM]}")

    am = A.Arm().connect()
    cap = None
    try:
        u = am.u
        # 显式关扭矩：臂变软，但关节仍可读（绝对编码器）。这是「一条指令都不发」的前提。
        am.robot.bus.disable_torque()
        print("[扭矩] 已关闭 —— 臂是软的，用手摆；本脚本不会发任何运动指令")
        print(f"[模式] FREEZE={'开（采集瞬间冻住姿势，务必先把手放上去）' if FREEZE else '关（全程只读，臂一直是软的）'}")

        cap = cv2.VideoCapture(str(devs[FH.FRONT_CAM]))
        if not cap.isOpened():
            raise SystemExit("打不开腕相机")
        SHOTS.mkdir(parents=True, exist_ok=True)
        for _ in range(15):
            cap.read()

        recs = json.loads(OUT.read_text()) if OUT.exists() else []
        print(f"[已有] {len(recs)} 条记录（{OUT.name}）")
        print("[就绪] 等命令文件 /tmp/mcap_cmd：写标签=采一张，写 quit=收工")
        sys.stdout.flush()

        while True:
            if not CMD_FILE.exists():
                time.sleep(0.2)
                continue
            cmd = CMD_FILE.read_text().strip()
            CMD_FILE.unlink()
            if cmd in ("quit", "q", "exit"):
                print("[收工] 退出（扭矩保持关闭）")
                break
            if cmd == "soft":
                am.robot.bus.disable_torque()
                print("[松开] 扭矩已关 —— 臂又变软了，扶住再动")
                sys.stdout.flush()
                continue

            # ---- 越界闸：原始计数不在标定区间就**拒绝采集** ----
            # 必须做在 hold_here() **之前** —— 它一 enable_torque() 就会朝夹过的目标冲。
            oor = raw_out_of_range(am)
            if oor:
                for m, (p, lo, hi) in oor.items():
                    off = p - hi if p > hi else p - lo
                    print(f"  [{cmd}] ✗ 拒绝采集：{m} 原始计数 {p}，区间 [{lo},{hi}]"
                          f"（越界 {off:+d}）")
                # 判据取 **mod 一整圈**：物理姿势 = 原始计数 mod 4096，这是实测出来的
                # （用户眼看腕在行程中间、计数 6322，只有 6322-4096=2226 对得上）。
                # 所以：
                #   mod 之后**在**区间里 ⇒ 只是个计数器多背了整圈，姿势没越界，
                #                          断电重上电复位即可（关节本身没坏）；
                #   mod 之后**也不在**   ⇒ 姿势本身停在标定区间外，断电没用，得掰回去。
                # 两类处置相反，别搞混 —— `wrist_roll` 越界 99% 是前一类。
                wind = [m for m, (p, lo, hi) in oor.items() if lo <= p % 4096 <= hi]
                near = [m for m in oor if m not in wind]
                if wind:
                    print(f"        · {'/'.join(wind)}：mod 一圈 = "
                          + ", ".join(f"{m} {oor[m][0] % 4096}" for m in wind)
                          + " 落在区间内 ⇒ 只是计数器多背了整圈，**姿势没越界**，"
                          "**断电重上电**即可复位（关节本身没坏）")
                if near:
                    print(f"        · {'/'.join(near)}：mod 一圈 = "
                          + ", ".join(f"{m} {oor[m][0] % 4096}" for m in near)
                          + " 也在区间外 ⇒ **姿势本身**停在标定区间外，断电没用。"
                          "扭矩已关，用手把它掰回区间内，再重发这条命令")
                sys.stdout.flush()
                continue

            # ---- 采集：先把姿势冻住（可选），再读关节 + 抓图 ----
            t0 = time.time()
            norm = hold_here(am) if FREEZE else am.read_norm()
            q = A.norm_to_q(norm, u)
            Tc = u.fk(q, CAM_LINK)
            c = Tc[:3, 3] * 1000
            axis = FH.cam_axis_of(Tc, +1.0)             # 光轴 = link 的 +z
            tilt = math.degrees(math.acos(np.clip(-axis[2], -1, 1)))
            cam_h = c[2] - A.TABLE_Z * 1000
            tcp_h = (u.fk(q, TCP_LINK)[2, 3] - A.TABLE_Z) * 1000

            fs = FH.grab(cap)
            if not fs:
                print(f"  [{cmd}] 腕相机读不到帧"); sys.stdout.flush(); continue
            img = fs[len(fs) // 2]
            # 抓完图**再读一次关节**：臂是软的，你手会抖。两次读数之差就是这次采集的
            # 位姿漂移量 —— Δ小才说明「图」和「FK」讲的是同一个位姿。
            t1 = time.time()
            norm2 = am.read_norm()
            t2 = time.time()
            per = {m: norm2[m] - norm[m] for m in A.MOTORS}
            worst = max(per, key=lambda m: abs(per[m]))
            drift = abs(per[worst])
            ch, ci, _, _ = det.detectBoard(img)
            n = 0 if ch is None else len(ch)
            shot = SHOTS / f"{cmd}.png"
            cv2.imwrite(str(shot), img)

            rec = {"tag": cmd, "n_corners": n, "shot": str(shot.relative_to(HERE)),
                   "norm": {k: float(v) for k, v in norm.items()},
                   "T_base_ee": Tc.tolist(),
                   "cam_mm": c.tolist(), "tilt_deg": tilt,
                   "cam_h_mm": float(cam_h), "tcp_h_mm": float(tcp_h),
                   "corners": None if ch is None else ch.reshape(-1, 2).tolist(),
                   "ids": None if ci is None else ci.reshape(-1).tolist(),
                   "drift_norm": float(drift), "drift_joint": worst,
                   "drift_per_joint": {k: float(v) for k, v in per.items()},
                   "norm2": {k: float(v) for k, v in norm2.items()},
                   "grab_s": round(t1 - t0, 3), "read_s": round(t2 - t1, 3)}
            recs = [r for r in recs if r.get("tag") != cmd] + [rec]
            OUT.write_text(json.dumps(recs, indent=2, ensure_ascii=False))

            seen = [r for r in recs if r["n_corners"] >= 6]
            d0, dz, npair = diversity(seen)
            print(f"  [{cmd}] 角点 {n:>3} / {len(ch) if ch is not None else 0}"
                  f"{'  ✓ 看见板' if n >= 6 else '  ✗ 没看见/太少'}"
                  f"   相机高 {cam_h:6.1f}mm  倾角 {tilt:4.1f}°  离 base 竖轴 "
                  f"{math.hypot(c[0], c[1]):5.0f}mm  TCP高 {tcp_h:6.1f}mm"
                  f"  漂移 {drift:.1f} ({worst[:9]}, 窗口 {t2-t0:.2f}s)"
                  + ("  ⚠ 手抖了，重采" if drift > 1.5 else ""))
            if npair < 3:
                judge = (f"对数只有 {npair}，还判不了退化（≥3 对才行）；"
                         f"目前转轴离竖直最大 {dz:.0f}°"
                         f"{'，已经有非竖直转轴了 ✓' if dz > 25 else '，还都是竖直轴'}")
            else:
                ok = d0 > 0.02 and dz > 25.0
                judge = (f"各向最小占比 {d0:.3f}（要>0.02）  转轴离竖直最大 {dz:.0f}°（要>25°）"
                         f"  ⇒ {'够用 ✓' if ok else '还不够：朝向太集中，缺大倾角 ✗'}")
            print(f"        已采 {len(seen)} 个有效位姿 | 转动多样性 {judge}")
            sys.stdout.flush()
    finally:
        if cap is not None:
            cap.release()
        try:
            am.robot.bus.disable_torque()          # 走之前确保臂是软的
        except Exception:                          # noqa: BLE001
            pass
        am.disconnect(keep_torque=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
