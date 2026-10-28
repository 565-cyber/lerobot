"""夹爪开口 vs gripper 命令，以及「被夹物体在闭合轴上装不装得进去」。

起因（用户 2026-10-15）：Pose6DAug 的 R/t/s 不是抓取规划，物体会和手指相交或伸出可抓
区域。这条不能靠掩码修，要量：① 手指开口；② 物体在**闭合轴**上的尺寸；③ 相交量。

做法全部数据驱动，不查手册：
  · 闭合轴 u：两个 gripper 命令下「动颚」网格质心的位移方向；
  · 开口：两颚网格之间的**最小距离**（平行颚的最接近面就是夹持面）；
  · 反解 21.5mm 方块对应的命令，和已知的经验锚（搬运段 gripper=17.4）对表。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path("/home/zjq/lerobot/workplace/calibration")
REPO = HERE.parent.parent


def L(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


sys.path.insert(0, str(REPO / "workplace" / "so101_urdf"))
MR_ = L("_pg_meshrender", HERE / "meshrender.py")
A = L("_pg_arm", HERE / "arm.py")
from fk_check import Urdf                                    # noqa: E402

URDF = REPO / "workplace" / "so101_urdf" / "so101_new_calib.urdf"
if not URDF.exists():
    cands = sorted((REPO / "workplace" / "so101_urdf").glob("*.urdf"))
    print("可用 URDF:", [c.name for c in cands])
    URDF = cands[0]
print(f"URDF: {URDF.name}")

u = Urdf(URDF)
meshes = MR_.urdf_link_meshes(URDF)
JAWS = ["gripper_link", "moving_jaw_so101_v1_link"]
for j in JAWS:
    print(f"  {j}: {'有网格' if j in meshes else '**没有网格**'} "
          f"({len(meshes.get(j, []))} 个 visual)")
print("  链上连杆:", " ".join(sorted(set(meshes) & set(u.parent_link))))

# 手臂关节随便取一个固定构型（开口与手臂姿态无关，只看相对关系）
ARM = {m: 0.0 for m in A.MOTORS}


def jaw_pts(g: float, link: str) -> np.ndarray:
    q = A.norm_to_q({**ARM, "gripper": g}, u)
    T_link = u.fk(q, link)
    out = []
    for T_vis, p in meshes[link]:
        v = MR_.stl_cached(p).reshape(-1, 3)
        out.append(v @ (T_link @ T_vis)[:3, :3].T + (T_link @ T_vis)[:3, 3])
    return np.concatenate(out)


def mesh_gap(P: np.ndarray, Q: np.ndarray) -> float:
    """两颚之间的最小距离（米）。点云降采样到 4000 点，够用且快。"""
    rng = np.random.default_rng(0)
    s = lambda X: X[rng.choice(len(X), min(4000, len(X)), replace=False)]  # noqa: E731
    return float(cKDTree(s(Q)).query(s(P))[0].min())


# 闭合轴：动颚在 g=0 与 g=100 时的质心位移
c0 = jaw_pts(0.0, JAWS[1]).mean(0)
c1 = jaw_pts(100.0, JAWS[1]).mean(0)
ux = c1 - c0
ux /= np.linalg.norm(ux)
print(f"\n闭合轴（base 系）u = [{ux[0]:+.3f} {ux[1]:+.3f} {ux[2]:+.3f}]"
      f"   动颚位移 {np.linalg.norm(c1-c0)*1e3:.2f}mm (g=0→100)")

print(f"\n{'命令 g':>8} {'开口(mm)':>10} {'开口/命令':>10}   两颚在 u 上的投影范围(mm)")
rows = []
for g in (0.0, 8.0, 17.4, 30.0, 50.0, 70.0, 92.0, 100.0):
    P, Q = jaw_pts(g, JAWS[0]), jaw_pts(g, JAWS[1])
    gap = mesh_gap(P, Q)
    rows.append((g, gap))
    pr = P @ ux * 1e3
    qr = Q @ ux * 1e3
    print(f"{g:8.1f} {gap*1e3:10.2f} {gap*1e3/max(g,1e-9):10.4f}   "
          f"静颚[{pr.min():7.1f},{pr.max():7.1f}] 动颚[{qr.min():7.1f},{qr.max():7.1f}]")

gs = np.array([r[0] for r in rows])
gp = np.array([r[1] for r in rows]) * 1e3
print(f"\n反解开口（线性插值）：")
for w, lab in ((21.5, "搬运段夹的 21.5mm 方块（经验锚：命令 ≈17.4）"),
               (24.0, "当前对照臂：直径 24mm 的球（r=12mm）"),
               (24.8, "舵机按 24.8mm 那一维夹")):
    if gp.min() <= w <= gp.max():
        g = float(np.interp(w, gp, gs))
        print(f"  {w:5.1f}mm → 命令 {g:5.1f}   {lab}")
    else:
        print(f"  {w:5.1f}mm → **超出开口范围**（{gp.min():.1f}~{gp.max():.1f}mm）   {lab}")

print(f"\n最大开口（g=100）= {gp.max():.1f}mm；命令被夹在 [8,92] ⇒ 可用到 "
      f"{np.interp(92.0, gs, gp):.1f}mm")
