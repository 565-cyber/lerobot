#!/usr/bin/env python
"""SO101 运动学自检 —— 只读，不连硬件，不需要 mesh。

回答两个问题：

  1. **零位约定是 `new_calib` 还是 `old_calib`？**
     判据（照 README）：
       new_calib — 每个关节的虚拟零位 = 该关节行程的**中点**
       old_calib — 虚拟零位 = 机械臂**水平完全伸展**的姿态
     本脚本算出 URDF 在 `q = 0` 时的姿态并打印出来。你把真机各关节
     置成 LeRobot 归一化 **0**，肉眼看它像不像打印出来的那个姿态。

  2. **LeRobot 归一化空间 ↔ URDF 关节角**怎么换。
     读自 `src/lerobot/motors/motors_bus.py:770`（`_normalize`）：
         norm = (tick - range_min) / (range_max - range_min) * 200 - 100
     即**归一化 0 = range_min/range_max 的中点**（`MotorNormMode.RANGE_M100_100`）。
     注意 `homing_offset` 在这个差分里**约掉了** —— 所以做 FK **不需要**它，
     只需要 `range_min` / `range_max` 两个端点。

     于是（在「用户记的是全行程」这个前提下）：
         q_urdf = mid(限位) + (norm / 100) * half(限位)

用法：
    python workplace/so101_urdf/fk_check.py                      # 默认带相机的那个
    python workplace/so101_urdf/fk_check.py --urdf so101_old_calib.urdf
    python workplace/so101_urdf/fk_check.py --pose 0 0 0 0 0 0   # 六个关节角(度)
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

TCP_LINK = "gripper_frame_link"
CAM_LINK = "wrist_camera_link"

# 六关节的标准顺序（LeRobot / LeIsaac / URDF 三者一致）
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


# --------------------------------------------------------------------------- #
#  SE(3)。URDF 的 origin rpy 是 extrinsic xyz，即 R = Rz(y) @ Ry(p) @ Rx(r)
# --------------------------------------------------------------------------- #

def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def origin_T(el):
    T = np.eye(4)
    if el is None:
        return T
    xyz = np.array([float(v) for v in (el.get("xyz") or "0 0 0").split()])
    rpy = np.array([float(v) for v in (el.get("rpy") or "0 0 0").split()])
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = xyz
    return T


def axis_angle_T(axis, theta):
    T = np.eye(4)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return T
    k = axis / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T[:3, :3] = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return T


# --------------------------------------------------------------------------- #

class Urdf:
    """按**关节名**索引；内部按子连杆名走链。"""

    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.by_child: dict[str, ET.Element] = {}   # 子连杆名 -> joint
        self.jname: dict[str, str] = {}             # 子连杆名 -> 关节名
        self.child_of: dict[str, str] = {}          # 关节名 -> 子连杆名
        self.parent_link: dict[str, str] = {}       # 子连杆名 -> 父连杆名
        for j in root.findall("joint"):
            child = j.find("child").get("link")
            self.by_child[child] = j
            self.jname[child] = j.get("name")
            self.child_of[j.get("name")] = child
            self.parent_link[child] = j.find("parent").get("link")

        self.revolute = [n for n in JOINT_ORDER if n in self.child_of]
        missing = [n for n in JOINT_ORDER if n not in self.child_of]
        self.extra_fixed = [self.jname[c] for c in self.by_child
                            if self.by_child[c].get("type") == "fixed"]
        self.urdf_path = path
        self.missing = missing

    # -- 关节限位（弧度），按关节名 -------------------------------------- #
    def limits(self) -> dict[str, tuple[float, float]]:
        out = {}
        for n in self.revolute:
            lim = self.by_child[self.child_of[n]].find("limit")
            out[n] = (float(lim.get("lower")), float(lim.get("upper")))
        return out

    # -- FK：base_link -> target_link，q 按关节名 ------------------------- #
    def fk(self, q: dict[str, float], target: str) -> np.ndarray:
        chain, link = [], target
        while link in self.by_child:
            chain.append(link)
            link = self.parent_link[link]
        T = np.eye(4)
        for child in reversed(chain):
            j = self.by_child[child]
            T = T @ origin_T(j.find("origin"))
            if j.get("type") == "revolute":
                ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                T = T @ axis_angle_T(ax, q.get(self.jname[child], 0.0))
        return T

    def link_pos(self, q, joint_name, base="base_link"):
        return self.fk(q, self.child_of[joint_name])[:3, 3]


def fmt(v):
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


# --------------------------------------------------------------------------- #

def report(path: Path, pose_deg=None):
    u = Urdf(path)
    lim = u.limits()
    print(f"\n{'=' * 70}\nURDF: {path.name}")
    if u.missing:
        print(f"  ⚠ 缺关节: {u.missing}")
    print(f"  可动关节 {len(u.revolute)}: {', '.join(u.revolute)}")
    print(f"  固定关节: {', '.join(u.extra_fixed) if u.extra_fixed else '(无)'}")

    print(f"\n--- 连杆参数（origin 是相对**父连杆**的，不是相对基座）---")
    for n in u.revolute:
        o = u.by_child[u.child_of[n]].find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()])
        lo, up = lim[n]
        print(f"  {n:15} origin={fmt(xyz)} m   "
              f"行程 [{math.degrees(lo):+7.2f}, {math.degrees(up):+7.2f}]° "
              f"({math.degrees(up - lo):.1f}°)")

    # ---- q = 0 的姿态：这是判零位约定的关键 ----
    zero = {n: 0.0 for n in u.revolute}
    print(f"\n--- ★ URDF 在 q = 0（全部关节角 0）时的姿态 ---")
    for n in u.revolute:
        p = u.link_pos(zero, n)
        print(f"    {n:15} -> x={p[0]:+.4f} y={p[1]:+.4f} z={p[2]:+.4f}  "
              f"水平={math.hypot(p[0], p[1]):.4f}")
    links = {n: u.link_pos(zero, n) for n in u.revolute}

    # 判据的核心：大臂/小臂与水平面的夹角。old_calib 两者都接近 0（平躺），
    # new_calib 大臂陡升。这是**不依赖坐标系朝向**的量，真机比对方便。
    print(f"\n--- ★ 判据：各段与水平面的夹角（q=0）---")
    for a, b in [("shoulder_lift", "elbow_flex"), ("elbow_flex", "wrist_flex"),
                 ("wrist_flex", "wrist_roll")]:
        d = links[b] - links[a]
        hor = math.hypot(d[0], d[1])
        print(f"    {a:14} -> {b:14}  {math.degrees(math.atan2(d[2], hor)):+6.1f}°  "
              f"(水平投影 {hor:.4f} m)")

    if TCP_LINK in u.parent_link:
        p = u.fk(zero, TCP_LINK)[:3, 3]
        print(f"    {'TCP':15} -> x={p[0]:+.4f} y={p[1]:+.4f} z={p[2]:+.4f}  "
              f"水平={math.hypot(p[0], p[1]):.4f}")
    else:
        print(f"    （本 URDF 无 {TCP_LINK}，TCP 位置不可算）")

    if any(j.find("child").get("link") == CAM_LINK for j in u.by_child.values()):
        c = u.fk(zero, CAM_LINK)[:3, 3]
        print(f"    腕部相机 @ q=0: x={c[0]:+.4f} y={c[1]:+.4f} z={c[2]:+.4f}")
    else:
        print("    （本 URDF 没有腕部相机 link）")

    # ---- 到 TCP / 相机 的手眼链 ----
    print(f"\n--- 固定关节链 ---")
    for child, j in u.by_child.items():
        if j.get("type") == "fixed":
            o = j.find("origin")
            print(f"  {u.jname[child]:26} {u.parent_link[child]:24} -> {child:28} "
                  f"xyz={o.get('xyz')}  rpy={o.get('rpy')}")

    # ---- 归一化映射 ----
    print(f"\n--- LeRobot 归一化 ±100 ↔ URDF 关节角 (rad) ---")
    for n in u.revolute:
        lo, up = lim[n]
        mid, half = (lo + up) / 2, (up - lo) / 2
        print(f"  {n:15} mid={mid:+.4f} half={half:.4f}  "
              f"norm=-100:{lo:+.3f}  0:{mid:+.3f}  +100:{up:+.3f}")

    if pose_deg:
        q = {n: math.radians(v) for n, v in zip(u.revolute, pose_deg)}
        T = u.fk(q, TCP_LINK)
        print(f"\n--- 指定姿态(度) {pose_deg} ---")
        print(f"  TCP: x={T[0,3]:+.4f} y={T[1,3]:+.4f} z={T[2,3]:+.4f}")
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(HERE / "so101_new_calib_camera.urdf"))
    ap.add_argument("--also", action="append", default=[],
                    help="额外的 URDF 一起对比")
    ap.add_argument("--pose", nargs=6, type=float)
    a = ap.parse_args()

    report(Path(a.urdf), a.pose)
    for extra in a.also:
        p = Path(extra)
        if p.exists():
            report(p, a.pose)
        else:
            print(f"\n(跳过 {p.name}：文件不在)")

    print(f"\n{'=' * 70}")
    print("判读方法：把真机六个关节都置成 LeRobot 归一化 0，看它像哪个 URDF 的 q=0 姿态。")
    print("  new_calib 的 q=0 = 行程中点姿态；old_calib 的 q=0 = 水平完全伸展姿态。")
    print("注意：夹爪在 LeRobot 里是**线性**(0闭合/100张开)，URDF 里是 revolute，")
    print("      README 明说该映射『not yet reflected in the URDF』—— 不要直接套用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
