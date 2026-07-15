"""
从真实机械臂录制数据绘制关节位置/速度/加速度曲线

数据来源: eval_record_planner1 (788 frames, 30fps, 6 joints)
"""
import sys
sys.path.append("/home/zjq/lerobot")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── 加载数据 ──
ds_path = "/home/zjq/lerobot/outputs/eval/eval_record_planner1"
ds = LeRobotDataset(ds_path)
print(f"Dataset: {ds.num_frames} frames, {ds.fps} fps, {ds.num_frames/ds.fps:.1f}s")
print(f"Joints: {ds.features['action']['names']}")

# 读取所有 action
n_frames = ds.num_frames
n_joints = ds.features['action']['shape'][0]
joint_names = ds.features['action']['names']

actions = np.zeros((n_frames, n_joints), dtype=np.float64)
for i in range(n_frames):
    actions[i] = ds[i]['action'].numpy()

# ── 计算速度和加速度 (有限差分) ──
FPS = ds.fps
dt = 1.0 / FPS

velocities = np.gradient(actions, dt, axis=0)     # rad/s
accelerations = np.gradient(velocities, dt, axis=0)  # rad/s²
jerk = np.gradient(accelerations, dt, axis=0)      # rad/s³

# ── 绘图 ──
time = np.arange(n_frames) / FPS

# 颜色方案: 6 关节各一个颜色
colors = ['#1565C0', '#E65100', '#2E7D32', '#C62828', '#6A1B9A', '#00838F']
labels = [n.replace('.pos','') for n in joint_names]

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

# --- Position ---
ax = axes[0]
for j in range(n_joints):
    ax.plot(time, actions[:, j], color=colors[j], linewidth=1.2, label=labels[j], alpha=0.85)
ax.set_ylabel("Position (rad)", fontsize=12)
ax.set_title("Joint Positions (Real Robot, eval_record_planner1)", fontsize=13, fontweight="bold", loc="left")
ax.legend(ncol=6, fontsize=9, loc="upper right", framealpha=0.8)
ax.grid(True, alpha=0.25)
ax.yaxis.set_major_locator(MaxNLocator(6))

# --- Velocity ---
ax = axes[1]
for j in range(n_joints):
    ax.plot(time, velocities[:, j], color=colors[j], linewidth=1.0, alpha=0.75)
ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
ax.set_ylabel("Velocity (rad/s)", fontsize=12)
ax.set_title("Joint Velocities (finite difference)", fontsize=13, fontweight="bold", loc="left")
ax.grid(True, alpha=0.25)
ax.yaxis.set_major_locator(MaxNLocator(6))

# --- Acceleration ---
ax = axes[2]
for j in range(n_joints):
    ax.plot(time, accelerations[:, j], color=colors[j], linewidth=0.8, alpha=0.65)
ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
ax.set_ylabel("Acceleration (rad/s²)", fontsize=12)
ax.set_title("Joint Accelerations (finite difference)", fontsize=13, fontweight="bold", loc="left")
ax.grid(True, alpha=0.25)
ax.yaxis.set_major_locator(MaxNLocator(6))

# --- Jerk ---
ax = axes[3]
for j in range(n_joints):
    ax.plot(time, jerk[:, j], color=colors[j], linewidth=0.6, alpha=0.5)
ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Jerk (rad/s³)", fontsize=12)
ax.set_title("Joint Jerk (finite difference)", fontsize=13, fontweight="bold", loc="left")
ax.grid(True, alpha=0.25)
ax.yaxis.set_major_locator(MaxNLocator(6))

fig.suptitle(
    f"Real Robot Joint Trajectories — {ds_path.split('/')[-1]}\n"
    f"{n_frames} frames @ {FPS} Hz, {n_frames/FPS:.1f}s total",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
out_path = "/home/zjq/lerobot/workplace/outputs/real_robot_trajectory.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")

# ── 定量指标 ──
print(f"\n{'Joint':<25} {'RMS Vel':>10} {'Max |Vel|':>10} {'RMS Acc':>10} {'Max |Acc|':>10} {'RMS Jerk':>10}")
print("-" * 75)
for j in range(n_joints):
    print(f"{labels[j]:<25} {np.sqrt(np.mean(velocities[:,j]**2)):>10.4f} "
          f"{np.max(np.abs(velocities[:,j])):>10.4f} "
          f"{np.sqrt(np.mean(accelerations[:,j]**2)):>10.4f} "
          f"{np.max(np.abs(accelerations[:,j])):>10.4f} "
          f"{np.sqrt(np.mean(jerk[:,j]**2)):>10.4f}")

print("\nThese values can serve as baseline for comparison when quintic blending is active.")
print("After recording a session WITH FSM transitions, re-run to see blend improvement.")
