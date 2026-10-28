"""决定哪个「内缩估计量」的 margin 增益真的是 1.00 px/px。

三个估计量在同一个重建网格上比：
  R = 等效半径差 √(A_gt/π) − √(A_rec/π)
  P = (漏−多)/P_gt            ← 只看真值周长
  M = (漏−多)/((P_gt+P_rec)/2) ← Minkowski 一阶关系要的平均周长

只跑舵机的 margin ∈ {0,1,2}（N=50、无噪声），几分钟。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/zjq/lerobot/workplace/calibration")
VC = None


def L(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


VC = L("_pg_vc", HERE / "validate_carve.py")
MFV = VC.MFV
MR = VC.MR

PART = "sts3215_03a_v1"
K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
DIST = np.zeros(5)

gt, pth = MFV.load_part(PART)
lo, hi = gt.reshape(-1, 3).min(0), gt.reshape(-1, 3).max(0)
ctr = 0.5 * (lo + hi)
gtz = MR.transform(gt, np.block([[np.eye(3), (-ctr)[:, None]], [np.zeros(3), 1.0]]))

print("%5s %9s %9s %9s %9s %9s" % ("margin", "inset_R", "inset_P", "inset_M",
                                   "P_gt", "P_rec"))
rows = []
for m in (0.0, 1.0, 2.0):
    views, cen = MFV.synthetic_views(gtz, K, DIST, 50, cam_dist_m=0.25, margin=m)
    ext = 0.5 * float((hi - lo).max()) + 0.015
    f, org, hh, _ = MFV.carve_auto(views, 1.0e-3, cen, hint_half=ext, coarse=4)
    tris = VC.MFV.marching_tets(f, org, hh, verbose=False)
    a = VC.silhouette_audit(tris, views, verbose=False)
    rows.append((m, a["aud_inset_px"], a["aud_inset_perim_px"], a["aud_inset_mid_px"]))
    # 单独量两个周长（全视角平均），看平均周长到底差多少
    from scipy import ndimage
    pg, pr = [], []
    for v in views:
        cam = MR.Cam(v["K"], v["dist"], v["T_world_cam"], v["W"], v["H"])
        rec = MR.rasterize(tris, cam, cull=False)["mask"]
        g = v["mask"]
        st = np.ones((3, 3), bool)
        pg.append(float((g & ~ndimage.binary_erosion(g, structure=st,
                                                     border_value=0)).sum()))
        pr.append(float((rec & ~ndimage.binary_erosion(rec, structure=st,
                                                       border_value=0)).sum()))
    print("%5.1f %9.4f %9.4f %9.4f %9.1f %9.1f"
          % (m, a["aud_inset_px"], a["aud_inset_perim_px"], a["aud_inset_mid_px"],
             np.mean(pg), np.mean(pr)))

x = np.array([r[0] for r in rows])
print()
for k, lab in ((1, "等效半径 R"), (2, "周长 P（只用真值）"), (3, "周长 M（平均）")):
    y = np.array([r[k] for r in rows])
    m_, b_ = np.polyfit(x, y, 1)
    print("  %-18s 增益 %.3f px/px   截距 %+.3f px" % (lab, m_, b_))
