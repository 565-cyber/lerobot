"""把两颚 + 夹持点 + 闭合轴画出来，用眼睛定「接触面在哪、开口往哪量」。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/zjq/lerobot/workplace/calibration")
FIGS = HERE / "figures"


def L(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


GC = L("_pj_gc", HERE / "grasp_check.py")
A = L("_pj_arm", HERE / "arm.py")
MFV = GC.MFV
MR = MFV.MR
from fk_check import Urdf                                      # noqa: E402

U = Urdf(GC.DEFAULT_URDF)
q = A.norm_to_q({**{m: 0.0 for m in A.MOTORS}, "gripper": 17.4}, U)

tf, tm = GC.jaws_in_base(17.4)
T_gr = U.fk(q, GC.GRASP_LINK)
print(f"gripper_frame_link 原点(base,mm) {np.round(T_gr[:3,3]*1e3,1)}")
print(f"夹持点 X_GRASP={GC.X_GRASP_MM}mm → base "
      f"{np.round((T_gr @ np.append(np.array(GC.X_GRASP_MM)*1e-3,1))[:3]*1e3,1)}")

# 闭合轴
c0 = GC.jaws_in_base(0.0)[1].reshape(-1, 3).mean(0)
c1 = GC.jaws_in_base(100.0)[1].reshape(-1, 3).mean(0)
ux = (c1 - c0) / np.linalg.norm(c1 - c0)
print(f"闭合轴 u = {np.round(ux,3)}")

# 每颚在 u 上的投影直方图（看哪一段是「指尖」）
for lab, T in (("静颚(舵机壳)", tf), ("动颚", tm)):
    pr = T.reshape(-1, 3) @ ux * 1e3
    print(f"  {lab} u 投影 [{pr.min():7.1f},{pr.max():7.1f}]mm "
          f"中位 {np.median(pr):7.1f}")

# 沿接近方向看：把 base 系转到「以夹持点为原点、u 为 x 轴」的视角再投影
P = T_gr @ np.append(np.array(GC.X_GRASP_MM) * 1e-3, 1.0)
P = P[:3]
# 接近方向取 gripper_frame 的 z（本机前相机光轴 = link 的 +z 那个约定）
az = T_gr[:3, 2] / np.linalg.norm(T_gr[:3, 2])
ax = ux - (ux @ az) * az
ax /= np.linalg.norm(ax)
ay = np.cross(az, ax)


import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
SURF, INK1, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=SURF)
views = ((ax, ay, "沿接近方向看（az 朝纸外）"),
         (ax, az, "沿 -u 侧看"),
         (ay, az, "沿 u 正对看（闭合轴朝纸外）"))
for k, (e1, e2, ttl) in enumerate(views):
    a = axes[k]
    a.set_facecolor(SURF)
    for lab, T, col in (("静颚", tf, S1), ("动颚", tm, S2)):
        v = T.reshape(-1, 3) - P
        a.plot(v @ e1 * 1e3, v @ e2 * 1e3, ".", color=col, ms=0.6,
               label=lab, alpha=0.55)
    a.plot(0, 0, "x", color=INK1, ms=11, mew=2.4, zorder=6)
    a.annotate("夹持点", (0, 0), textcoords="offset points", xytext=(7, 5),
               color=INK1, fontsize=8.5)
    # 闭合轴投影
    a.annotate("", xy=(ux @ e1 * 20e3 if False else 0, 0), xytext=(0, 0))
    a.set_title(ttl, color=INK1, fontsize=10)
    a.set_aspect("equal")
    a.grid(color="#e1e0d9", lw=0.7)
    a.tick_params(colors=INK2, labelsize=8)
    a.set_xlabel(f"({['ax','ay','ay'][k]}, mm)", color=INK2, fontsize=8.5)
    a.set_ylabel(f"({['ay','az','az'][k]}, mm)", color=INK2, fontsize=8.5)
    if k == 0:
        a.legend(frameon=False, labelcolor=INK2, fontsize=9, markerscale=14)
axes[1].annotate("u", xy=(ux @ ax * 25e3, ux @ az * 25e3),
                 color=INK1, fontsize=10, fontweight="bold")
# 在图 1 上画出 u 的投影方向
for a, (e1, e2) in zip(axes, ((ax, ay), (ax, az), (ay, az))):
    a.annotate("", xy=(ux @ e1 * 22e3, ux @ e2 * 22e3), xytext=(0, 0),
               arrowprops=dict(arrowstyle="->", color=INK1, lw=1.8))
fig.suptitle("SO101 夹爪 @g=17.4：静颚 = 舵机壳，动颚 = 铰链颚；黑叉 = 实测夹持点，"
             "箭头 = 闭合轴 u", color=INK1, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = FIGS / "grasp_jaws_geometry.png"
fig.savefig(out, dpi=125, facecolor=SURF)
print("[图]", out)
