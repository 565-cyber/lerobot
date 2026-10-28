#!/usr/bin/env python3
"""SO101 跟随臂的最小封装：认口、读状态、正运动学、逆运动学、平滑走位。

为什么单独一个文件：后面**每个**标定脚本都要「把臂开到某个位姿、记下 FK、
再抓一张图」。位姿规划错了会把夹爪压到桌面上，所以走位必须收口在一处，
慢速插值 + 限速 + 回到起点。

约定（与 `workplace/so101_urdf/fk_check.py` 一致，不要另发明）：

- 关节顺序 `MOTORS`；归一化 `norm ∈ [-100,100]`（夹爪 `[0,100]`）。
- `norm → URDF 角`：`q = mid(限位) + (norm/100) * half(限位)`。
  `homing_offset` 在这个差分里约掉了，所以做 FK **只需要 range_min/max**。
- **桌面在 base_link 的 z = -2.4 mm**（底座 STL 下沿），见 `so101_urdf/FINDINGS.md`。
- `gripper_frame_link` = 两点夹爪的**尖端中点**（TCP）；`wrist_camera_link` = 腕部相机。
  相机**光轴 = 该 link 的 -z**（见 `FINDINGS.md` 与本次实测朝向）。

用法：
    python workplace/calibration/arm.py --check          # 只读，不动
    python workplace/calibration/arm.py --check --fk
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workplace" / "so101_urdf"))
from fk_check import Urdf, TCP_LINK, CAM_LINK          # noqa: E402

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
BODY = MOTORS[:5]

FOLLOWER_ID = "my_awesome_follower_arm"
FOLLOWER_CAL = (Path.home() / ".cache/huggingface/lerobot/calibration/robots"
                / f"so101_follower/{FOLLOWER_ID}.json")

URDF_PATH = REPO / "workplace" / "so101_urdf" / "so101_new_calib_camera.urdf"

TABLE_Z = -0.0024            # 桌面在 base_link 系里的 z（米）
CAM_AXIS = ("z", -1)         # 相机光轴 = wrist_camera_link 的 -z

# 夹爪命令的安全余量（归一化）。**不要**碰两端的 0/100 —— 会堵转 → Overload → 掉线
GRIP_MIN, GRIP_MAX = 8.0, 92.0


# --------------------------------------------------------------- 串口：认口

def find_follower_port(verbose: bool = True) -> str:
    """自动认出**跟随臂**在哪个 /dev/ttyACM*。

    不看编号（插拔会换号，见 CAMERAS.md 同样的坑），而是比较电机 EEPROM 里的
    `Min/Max_Position_Limit` 与标定 json 是否逐关节一致 —— 这正是
    `FeetechMotorsBus.is_calibrated` 用的判据。
    """
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal = json.loads(FOLLOWER_CAL.read_text())
    motors = {m: Motor(i + 1, "sts3215",
                       MotorNormMode.RANGE_0_100 if m == "gripper"
                       else MotorNormMode.RANGE_M100_100)
              for i, m in enumerate(MOTORS)}
    for port in sorted(str(p) for p in Path("/dev").glob("ttyACM*")):
        bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)
        try:
            bus.connect(handshake=True)
        except Exception as e:                                    # noqa: BLE001
            if verbose:
                print(f"  {port}: 连不上（{type(e).__name__}）")
            continue
        try:
            ok = all(
                bus.read("Min_Position_Limit", m, normalize=False) == cal[m]["range_min"]
                and bus.read("Max_Position_Limit", m, normalize=False) == cal[m]["range_max"]
                for m in MOTORS)
            if verbose:
                print(f"  {port}: 与跟随臂标定{'一致 ✓' if ok else '不一致'}")
            if ok:
                return port
        finally:
            bus.disconnect(disable_torque=False)
    raise SystemExit("没找到跟随臂 —— 电源/线?或标定 json 与电机对不上")


# --------------------------------------------------------------- 归一化映射

def limits_norm() -> dict[str, tuple[float, float]]:
    return {m: (0.0, 100.0) if m == "gripper" else (-100.0, 100.0) for m in MOTORS}


COUNTS_PER_DEG = 4096 / 360.0        # STS3215 编码器分辨率


def calibrated_half_deg() -> dict[str, float] | None:
    """标定文件里各关节**实测行程**的一半（度）。读不到就返回 None（退回 URDF 限位）。"""
    try:
        cal = json.loads(FOLLOWER_CAL.read_text())
    except Exception:                                          # noqa: BLE001
        return None
    return {m: (cal[m]["range_max"] - cal[m]["range_min"]) / 2 / COUNTS_PER_DEG
            for m in MOTORS if m in cal}


def norm_to_q(norm: dict[str, float], u: Urdf) -> dict[str, float]:
    """归一化 → URDF 关节角（弧度）。

    ⚠️ **本体关节的半行程取「标定实测行程」，不是 URDF 的 limit。**
    这里原来用 `half(URDF限位)`，隐含前提「关节实测行程 = URDF 限位行程」——
    实测**不成立**，URDF 的 limit 比机械行程**窄** 2–6%：

        shoulder_lift  207.9° vs 200.0°     wrist_flex  200.0° vs 190.0°
        wrist_roll     340.9° vs 320.0°     （elbow_flex 193.9 vs 193.7，几乎相等）

    这是个**增益**误差，随关节偏离中点线性放大，且**不会被手眼外参吸收**
    （常量偏置会，增益不会）。实测后果：改用标定行程后，板↔base 的留一预测误差
    从中位 32mm 降到 17mm，姿态残差 3.5°→2.5°
    （见 `workplace/calibration/solve_flat_board.py`）。
    读不到标定文件时自动退回 URDF 限位，行为与旧版一致。
    """
    lim = u.limits()
    hc = calibrated_half_deg()
    q = {}
    for m in MOTORS:
        lo, hi = (math.degrees(x) for x in lim[m])
        if m == "gripper":                       # 夹爪不在相机的运动链上，照旧
            deg = lo + norm[m] / 100.0 * (hi - lo)
        else:
            half = hc[m] if (hc and m in hc) else (hi - lo) / 2
            deg = (lo + hi) / 2 + norm[m] / 100.0 * half
        q[m] = math.radians(deg)
    return q


def q_to_norm(q: dict[str, float], u: Urdf) -> dict[str, float]:
    """`norm_to_q` 的逆（半行程同样取标定实测）。"""
    lim = u.limits()
    hc = calibrated_half_deg()
    out = {}
    for m in MOTORS:
        lo, hi = (math.degrees(x) for x in lim[m])
        deg = math.degrees(q[m])
        if m == "gripper":
            out[m] = (deg - lo) / (hi - lo) * 100.0
        else:
            half = hc[m] if (hc and m in hc) else (hi - lo) / 2
            out[m] = (deg - (lo + hi) / 2) / half * 100.0
    return out


# --------------------------------------------------------------- 逆运动学

def solve_ik(u: Urdf, seed: dict[str, float], p_target: np.ndarray,
             axis_dir: np.ndarray, cam_axis=CAM_AXIS,
             w_pos: float = 1.0, w_dir: float = 60.0,
             w_reg: float = 0.002, w_tcp: float = 4.0,
             tcp_min_z: float = 0.06):
    """数值 IK：把**腕部相机**放到 p_target（base 系，米），光轴指向 axis_dir。

    p_target / axis_dir 用「相机 link 原点」和「link 的 cam_axis 方向」，都是
    base_link 系。加了三条软约束：

    - `w_reg`：别离 seed 太远（解不唯一时挑最近的那个，也避免反关节）；
    - `w_tcp`：TCP（夹爪尖端）离桌面**至少** `tcp_min_z` —— 这是**安全约束**，
      规划出来的位姿绝不能让夹爪戳到桌面。
    """
    from scipy.optimize import least_squares

    lim = limits_norm()
    lo = np.array([lim[m][0] for m in MOTORS])
    hi = np.array([lim[m][1] for m in MOTORS])
    x0 = np.array([seed[m] for m in MOTORS], float)

    ax_i = "xyz".index(cam_axis[0])
    ax_s = cam_axis[1]

    def unpack(x):
        return {m: float(x[i]) for i, m in enumerate(MOTORS)}

    def resid(x):
        q = norm_to_q(unpack(x), u)
        T = u.fk(q, CAM_LINK)
        r_pos = (T[:3, 3] - p_target) * w_pos
        v = ax_s * T[:3, ax_i]
        r_dir = (v - axis_dir) * w_dir
        r_reg = (x - x0) * w_reg
        z_tcp = u.fk(q, TCP_LINK)[2, 3]
        r_tcp = np.array([w_tcp * max(0.0, tcp_min_z - z_tcp)])
        return np.concatenate([r_pos, r_dir, r_reg, r_tcp])

    sol = least_squares(resid, x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
    x = sol.x
    q = norm_to_q(unpack(x), u)
    T = u.fk(q, CAM_LINK)
    return {
        "norm": unpack(x),
        "q": q,
        "cam_pos": T[:3, 3],
        "cam_err_mm": float(np.linalg.norm(T[:3, 3] - p_target) * 1000),
        "axis": ax_s * T[:3, ax_i],
        "axis_err_deg": float(np.degrees(np.arccos(np.clip(
            (ax_s * T[:3, ax_i]) @ axis_dir, -1, 1)))),
        "tcp": u.fk(q, TCP_LINK)[:3, 3],
    }


def link_clearance(u: Urdf, q: dict[str, float]) -> dict[str, float]:
    """所有 link 原点离桌面的高度（mm）—— 走位前的安全检查。"""
    out = {}
    for child in u.by_child:
        try:
            z = u.fk(q, child)[2, 3]
        except Exception:                                          # noqa: BLE001
            continue
        out[child] = (z - TABLE_Z) * 1000
    return out


# --------------------------------------------------------------- 硬件

class Arm:
    """跟随臂。`connect()` 后可以 `read_norm()` / `goto()`；`goto` 是唯一会动的。"""

    def __init__(self, port: str | None = None, verbose: bool = True):
        self.port = port
        self.verbose = verbose
        self.robot = None
        self.u = Urdf(URDF_PATH)

    def connect(self):
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        if self.port is None:
            self.port = find_follower_port(self.verbose)
        self.robot = SO101Follower(SO101FollowerConfig(
            port=self.port, id=FOLLOWER_ID, use_degrees=False, cameras={}))
        self.robot.connect(calibrate=False)          # 只连，不写 EEPROM
        if self.verbose:
            print(f"[臂] 连上 {self.port}")
        return self

    def read_norm(self) -> dict[str, float]:
        obs = self.robot.bus.sync_read("Present_Position")
        return {m: float(obs[m]) for m in MOTORS}

    def fk(self, link: str, norm: dict[str, float] | None = None) -> np.ndarray:
        return self.u.fk(norm_to_q(norm or self.read_norm(), self.u), link)

    def goto(self, target: dict[str, float], seconds: float = 3.0,
             hz: float = 60.0, tol: float = 1.0) -> dict[str, float]:
        """平滑插值走到 target（归一化空间）。返回实际读到的终值。

        **永远不走直线突然跳**：按 `seconds` 插值，每帧只发当前点，
        `max_relative_target` 还会在底层再限一次速。

        ⚠️ 夹爪被**夹到限位**时会持续输出堵转扭矩，STS3215 会报 Overload 并且
        **掉线**（实测踩过一次：命令夹爪 norm=0 = 全闭 → 过载 → 电机 6 从总线上消失，
        要断电重启）。所以夹爪命令一律夹到 `[GRIP_MIN, GRIP_MAX]`，两头各留余量。
        """
        target = dict(target)
        if "gripper" in target:
            target["gripper"] = float(np.clip(target["gripper"], GRIP_MIN, GRIP_MAX))
        self.robot.bus.enable_torque()
        start = self.read_norm()
        n = max(1, int(seconds * hz))
        for k in range(1, n + 1):
            a = k / n
            cmd = {f"{m}.pos": start[m] + (target[m] - start[m]) * a for m in MOTORS}
            self.robot.send_action(cmd)
            time.sleep(1.0 / hz)
        # 收敛：再发几次直到到位或用完时间
        for _ in range(int(2.0 * hz)):
            now = self.read_norm()
            err = max(abs(now[m] - target[m]) for m in MOTORS)
            if err < tol:
                break
            self.robot.send_action({f"{m}.pos": target[m] for m in MOTORS})
            time.sleep(1.0 / hz)
        return self.read_norm()

    def disconnect(self, keep_torque: bool = False):
        if self.robot is None:
            return
        try:
            if not keep_torque:
                self.robot.bus.disable_torque()
        finally:
            self.robot.disconnect()
            self.robot = None


# --------------------------------------------------------------- CLI

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只读状态，不动臂")
    ap.add_argument("--port", default=None)
    a = ap.parse_args()

    arm = Arm(port=a.port).connect()
    try:
        n = arm.read_norm()
        print("\n[当前归一化读数]")
        for m in MOTORS:
            print(f"  {m:14s} {n[m]:+8.2f}")
        q = norm_to_q(n, arm.u)
        print("\n[URDF 关节角 deg]  " +
              "  ".join(f"{m}={math.degrees(q[m]):+.1f}" for m in MOTORS))
        for name, link in (("TCP", TCP_LINK), ("腕相机", CAM_LINK)):
            p = arm.fk(link, n)[:3, 3]
            print(f"  {name:6s} xyz=({p[0]*1000:+7.1f},{p[1]*1000:+7.1f},{p[2]*1000:+7.1f}) mm"
                  f"  离桌面 {(p[2]-TABLE_Z)*1000:+6.1f} mm")
        T = arm.fk(CAM_LINK, n)
        ax_i, ax_s = "xyz".index(CAM_AXIS[0]), CAM_AXIS[1]
        v = ax_s * T[:3, ax_i]
        print(f"  腕相机光轴(-z) -> ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})  "
              f"偏离竖直 {math.degrees(math.acos(np.clip(-v[2],-1,1))):.1f}°")
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
