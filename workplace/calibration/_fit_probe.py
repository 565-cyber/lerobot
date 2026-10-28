"""拟合前的吞吐测量：STL 载入/减面后的面数、rasterize 在不同面数×分辨率下的耗时。

只为定 Stage C 的调度（粗/细两级各留多少次重启动、每次几 ms），不改任何东西、不出图。
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("/home/zjq/lerobot/workplace/calibration")


def L(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


MR = L("_fp_mr", HERE / "meshrender.py")

t0 = time.time()
tris = MR.stl_cached(HERE / "calib/recon/sts3215_03a_v1_v50_h0.5_m1.stl")
print(f"载入 {len(tris)} 面，{time.time() - t0:.1f}s")
P = tris.reshape(-1, 3)
print(f"  bbox = {np.round(P.max(0) - P.min(0), 4)} m（应为 45.4/24.8/39.6mm 量级）")
print(f"  质心 = {np.round(P.mean(0), 4)} m")

for cell in (4.0e-3, 3.0e-3, 2.0e-3, 1.5e-3, 1.0e-3, 0.7e-3):
    t0 = time.time()
    d = MR.decimate(tris, cell)
    print(f"  减面 cell={cell*1e3:.1f}mm → {len(d):6d} 面（{time.time()-t0:.1f}s）")

# 腕视图相机的量级：f=409.57，夹持深度 ~90mm
K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
T = np.eye(4)
T[:3, :3] = np.diag([1.0, -1.0, -1.0])          # 光轴 = base 的 −z（与 up_local_cam 同约定）
T[:3, 3] = [0.0, 0.0, 0.09]                     # 相机在 base 原点（=物体中心）正上方 90mm
print(f"\nrasterize 计时（物体在 z=90mm，f=409.6 ⇒ 45mm 物体约 "
      f"{45.4e-3 * 409.57 / 0.09:.0f}px 宽）")
for cell, W, H in ((3.0e-3, 640, 480), (1.5e-3, 640, 480), (1.0e-3, 640, 480),
                   (3.0e-3, 320, 240), (3.0e-3, 160, 120),
                   (1.5e-3, 320, 240), (3.0e-3, 1280, 960)):
    d = MR.decimate(tris, cell)
    s = W / 640.0
    Ks = np.diag([s, s, 1.0]) @ K                # 内参等比缩放 ⇒ 画面就是原图的 s 倍
    cam = MR.Cam(Ks, np.zeros(5), T, W=W, H=H)
    tt = MR.transform(d, np.eye(4))
    MR.rasterize(tt, cam)                        # 预热（cv2 线程池）
    n = 5
    t0 = time.time()
    for _ in range(n):
        ras = MR.rasterize(tt, cam)
    dt = (time.time() - t0) / n
    print(f"  {len(d):6d} 面 {W:4d}x{H:<4d} → {dt*1e3:7.1f} ms/次"
          f"（覆盖 {ras['mask'].mean()*100:5.2f}%）")
