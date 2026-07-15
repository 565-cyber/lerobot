"""
从 FSM 录制数据 (eval_record_planner_3step) 绘制带任务阶段标注的完整轨迹
"""
import sys
sys.path.append("/home/zjq/lerobot")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset('/home/zjq/lerobot/outputs/eval/eval_record_planner_3step')
FPS, n = ds.fps, ds.num_frames
joint_names = [n.replace('.pos','') for n in ds.features['action']['names']]

actions = np.zeros((n, 6))
for i in range(n):
    actions[i] = ds[i]['action'].numpy()

dt = 1.0 / FPS
time = np.arange(n) / FPS
velocities = np.gradient(actions, dt, axis=0)
accelerations = np.gradient(velocities, dt, axis=0)

# 推测任务阶段 (基于 shoulder_pan 趋势 + gripper 事件)
# Phase 1: frames 0-200 (夹黄块, SP~-28°)
# Phase 2: frames 200-600 (夹蓝块, SP~-13° to -25°)
# Phase 3: frames 600-1192 (关盖子, SP~+5°)
phases = [
    (0, 200, "State 1: Pick Yellow", "#FFF9C4"),
    (200, 600, "State 2: Pick Blue", "#E3F2FD"),
    (600, 1192, "State 3: Close Lid", "#E8F5E9"),
]

# Gripper events
grip_events = [80, 352, 552, 1013, 1029]

colors = ['#1565C0', '#E65100', '#2E7D32', '#C62828', '#6A1B9A', '#00838F']

fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

# --- Position ---
ax = axes[0]
for j in range(6):
    ax.plot(time, actions[:, j], color=colors[j], linewidth=1.0, alpha=0.8, label=joint_names[j])
# Phase backgrounds
for ps, pe, label, color in phases:
    ax.axvspan(ps/FPS, pe/FPS, alpha=0.25, color=color)
    ax.text((ps+pe)/(2*FPS), ax.get_ylim()[1]*0.95, label, ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
# Gripper markers
for ge in grip_events:
    ax.axvline(x=ge/FPS, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylabel("Position (rad)", fontsize=12)
ax.set_title("Joint Positions — eval_record_planner_3step (FSM 3-stage recording)", fontsize=14, fontweight="bold")
ax.legend(ncol=6, fontsize=9, loc='upper right', framealpha=0.8)
ax.grid(True, alpha=0.2)
ax.yaxis.set_major_locator(MaxNLocator(6))

# --- Velocity ---
ax = axes[1]
for j in range(6):
    ax.plot(time, velocities[:, j], color=colors[j], linewidth=0.8, alpha=0.65)
for ps, pe, label, color in phases:
    ax.axvspan(ps/FPS, pe/FPS, alpha=0.25, color=color)
for ge in grip_events:
    ax.axvline(x=ge/FPS, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_ylabel("Velocity (rad/s)", fontsize=12)
ax.set_title("Joint Velocities", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.yaxis.set_major_locator(MaxNLocator(6))

# --- Acceleration ---
ax = axes[2]
for j in range(6):
    ax.plot(time, accelerations[:, j], color=colors[j], linewidth=0.5, alpha=0.5)
for ps, pe, label, color in phases:
    ax.axvspan(ps/FPS, pe/FPS, alpha=0.25, color=color)
for ge in grip_events:
    ax.axvline(x=ge/FPS, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Acceleration (rad/s²)", fontsize=12)
ax.set_title("Joint Accelerations (red dashes = gripper events)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.yaxis.set_major_locator(MaxNLocator(6))

plt.tight_layout()
out = "/home/zjq/lerobot/workplace/outputs/fsm_3step_full.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
print(f"\nData: {n} frames @ {FPS}Hz = {n/FPS:.1f}s")
print(f"Phases inferred from shoulder_pan trend + gripper events:")
for ps, pe, label, _ in phases:
    print(f"  {label}: frames {ps}-{pe} ({(pe-ps)/FPS:.1f}s)")
print(f"Gripper events at frames: {grip_events}")
print(f"\nNote: To get precise blend boundaries, add logging to lerobot_record.py")
print(f"      that records fsm_state['blending'] flag per frame.")
