"""抽查：`_covering_radius` 测出的 theta_c 是否含「极冠空洞」伪影。"""
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


MFV = L("_pt_mfv", HERE / "mesh_from_views.py")

gt, _ = MFV.load_part("sts3215_03a_v1")
gtc = gt.reshape(-1, 3)
lo, hi = gtc.min(0), gtc.max(0)
ctr = 0.5 * (lo + hi)
MR = MFV.MR
gtz = MR.transform(gt, np.block([[np.eye(3), (-ctr)[:, None]], [np.zeros(3), 1.0]]))
K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
dist = np.zeros(5)

for n in (4, 8, 20, 50, 100):
    views, cen = MFV.synthetic_views(gtz, K, dist, n, cam_dist_m=0.25)
    D = np.array([v["T_world_cam"][:3, 3] for v in views])
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    # 探针：与视图同向的法向方向
    Q = D.copy()
    cos = Q @ D.T
    nn_max = np.arccos(np.clip(cos.max(1), -1, 1))          # 每视图到最近邻的角度
    # 全网随机探针
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20000, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    cosP = P @ D.T
    angP = np.arccos(np.clip(cosP.max(1), -1, 1))
    cz = D[:, 2]
    m = (P[:, 2] >= cz.min()) & (P[:, 2] <= cz.max())
    core = P[:, 2] >= cz.min() + 0.2
    print(f"N={n:>3d}  z范围[{cz.min():+.2f},{cz.max():+.2f}]")
    print(f"      视图到最近邻角：中位 {np.median(nn_max):.3f}  max {nn_max.max():.3f}")
    print(f"      随机探针 max-min角：{angP.max():.3f}（全 z 带）"
          f"  {angP[m].max():.3f}（限同带）  {angP[core].max():.3f}（限 z>zmin+0.2）")
    print(f"      2.2/sqrtN = {2.2 / n ** 0.5:.3f}")
