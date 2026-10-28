"""判决实验：把**实测的**方块位姿（腕相机系，T_cam_cube）和 URDF 的两颚放进同一个坐标系。

这一步**完全不碰** X_GRASP_MM —— 如果方块落进夹持槽，说明 URDF 的颚几何 + FK + 相机
连杆全对，唯一错的是 X_GRASP_MM 那个数（它来自 `estimate_x_tcp_cube`，跨了一个没验过的
`gripper_frame_link` 约定）。如果方块落在槽外，那是更深层的 URDF↔真机不一致。
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path("/home/zjq/lerobot/workplace/calibration")


def L(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


GC = L("_pf_gc", HERE / "grasp_check.py")
A = L("_pf_arm", HERE / "arm.py")
from fk_check import Urdf, CAM_LINK                              # noqa: E402

U = Urdf(GC.DEFAULT_URDF)
ARM = {m: 0.0 for m in A.MOTORS}
G_HELD = 17.4

d = json.load(open(HERE / "calib" / "held_pose_wrist.json"))
T_cam_cube = np.array(d["T_cam_cube"], float)
edge = d["cube_edge_mm"] * 1e-3
print(f"实测方块：边长 {edge*1e3:.2f}mm，位姿在腕相机系（T_cam_cube），"
      f"平移 {np.round(T_cam_cube[:3,3]*1e3,2)}mm")


def box_tris(side, T):
    h = side / 2
    cor = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1)
                    for k in (-1, 1)], float) * h
    F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    tris = np.array([[cor[i], cor[j], cor[k]] for i, j, k in F])
    return tris @ T[:3, :3].T + T[:3, 3]


# 两颚在腕相机系（与构型无关：两者都刚性挂在腕上）
q = A.norm_to_q({**ARM, "gripper": G_HELD}, U)
T_cam_base = np.linalg.inv(U.fk(q, CAM_LINK))
tf_b, tm_b = GC.jaws_in_base(G_HELD)
def to_cam(T):                                   # 点集 (N,3) → 相机系
    return T @ T_cam_base[:3, :3].T.T if False else (T @ T_cam_base[:3, :3].T
                                                    + T_cam_base[:3, 3])
tf, tm = to_cam(tf_b.reshape(-1, 3)), to_cam(tm_b.reshape(-1, 3))
cube = box_tris(edge, T_cam_cube)

# 闭合轴，也搬到相机系
c0 = to_cam(GC.jaws_in_base(0.0)[1].reshape(-1, 3)).mean(0)
c1 = to_cam(GC.jaws_in_base(100.0)[1].reshape(-1, 3)).mean(0)
ux = (c1 - c0) / np.linalg.norm(c1 - c0)
print(f"闭合轴（腕相机系）u = {np.round(ux,3)}")

cc = cube.reshape(-1, 3).mean(0)
print(f"\n{'命令 g':>8} {'方块质心到静颚':>13} {'到动颚':>10} {'在 u 上间隙':>12}")
for g in (0.0, 8.0, 17.4, 30.0, 50.0, 92.0, 100.0):
    qg = A.norm_to_q({**ARM, "gripper": g}, U)
    Tcb = np.linalg.inv(U.fk(qg, CAM_LINK))
    a = GC.jaws_in_base(g)[0].reshape(-1, 3) @ Tcb[:3, :3].T + Tcb[:3, 3]
    b = GC.jaws_in_base(g)[1].reshape(-1, 3) @ Tcb[:3, :3].T + Tcb[:3, 3]
    # 方块顶点到两颚最近点
    da = cKDTree(a[::20]).query(cube.reshape(-1, 3))[0].min() * 1e3
    db = cKDTree(b[::20]).query(cube.reshape(-1, 3))[0].min() * 1e3
    # 两颚沿 u 的「面」：取各自离方块质心最近的 5% 点，投影到 u
    pa = (a - cc) @ ux * 1e3
    pb = (b - cc) @ ux * 1e3
    ka = pa[np.argsort(np.abs(pa))[:max(1, len(pa)//20)]]
    kb = pb[np.argsort(np.abs(pb))[:max(1, len(pb)//20)]]
    gap = (kb.mean() - ka.mean())
    print(f"{g:8.1f} {da:13.2f} {db:10.2f} {gap:12.2f}")

print(f"\n方块自身沿 u 的半宽 = {np.abs((cube.reshape(-1,3)-cc)@ux).max()*1e3:.2f}mm"
      f"（边长 {edge*1e3:.1f}mm，理论半宽 {edge*1e3/2:.2f}mm）")
print(f"两颚质心间距 = {np.linalg.norm(c1-c0)*1e3:.2f}mm（g=0→100 的行程）")
