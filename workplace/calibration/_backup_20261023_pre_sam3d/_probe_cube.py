"""用「已知被夹住的 21.5mm 方块」验闸门：它在 g=17.4 下必须**零相交**。

顺便验两件事：
  · 闸门本身（体素 + 距离场）在真实尺度上能不能用；
  · `gripper_frame_link` 的 URDF 约定和 `estimate_x_tcp_cube` 实测的夹持点是否自洽 ——
    若这里报出相交，就是**可量化的** URDF↔真机 TCP 不一致（与已记录的 FK 天花板同源）。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/zjq/lerobot/workplace/calibration")


def L(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


GC = L("_pc_gc", HERE / "grasp_check.py")
A = L("_pc_arm", HERE / "arm.py")
MFV = GC.MFV
MR = MFV.MR
from fk_check import Urdf                                       # noqa: E402

U = Urdf(GC.DEFAULT_URDF)
ARM = {m: 0.0 for m in A.MOTORS}


def box_tris(side_m, centre_base, R=np.eye(3)):
    h = side_m / 2
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1)
                        for k in (-1, 1)], float) * h
    F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    tris = np.array([[corners[a], corners[b], corners[c]] for a, b, c in F])
    return tris @ R.T + centre_base


print("被夹住的方块（实测 21.5mm，夹取帧 gripper=17.4）—— 闸门必须报 0 相交")
print(f"{'命令 g':>8} {'相交体积(mm³)':>14} {'到静颚(mm)':>11} {'到动颚(mm)':>11}")
for g in (8.0, 17.4, 30.0, 50.0, 70.0, 92.0):
    q = A.norm_to_q({**ARM, "gripper": g}, U)
    T = U.fk(q, GC.GRASP_LINK)
    p = (T @ np.append(np.array(GC.X_GRASP_MM) * 1e-3, 1.0))[:3]
    R = T[:3, :3]                       # 方块轴向对齐夹爪系（held_pose 的约定）
    tris = box_tris(21.5e-3, p, R)
    r = GC.object_vs_jaws(tris, g, h=5e-4)
    print(f"{g:8.1f} {r['intersect_mm3']:14.2f} {r['clearance_mm']:11.2f} "
          f"{'—' if r['intersect_mm3'] > 0 else '':>11}", end="")
    # 分颚看
    print(f"  （静颚 {r['intersect_fixed_mm3']:.1f} / 动颚 {r['intersect_moving_mm3']:.1f}）")
