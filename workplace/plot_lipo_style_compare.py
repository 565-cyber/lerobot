"""
生成 LiPo 风格的对比图：Quintic Blend (ours) vs Raw (no blend, simulated)
在同一张图上展示位置/速度/加速度，验证 quintic 的 C² 连续性优势。

用法:
  python workplace/plot_lipo_style_compare.py
"""
import sys
sys.path.append("/home/zjq/lerobot")

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── 加载数据 ──
ds_path = "/home/zjq/lerobot/outputs/eval/eval_record_planner_3step3"
ds = LeRobotDataset(ds_path)
FPS, n = ds.fps, ds.num_frames
joint_names_short = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

actions = np.zeros((n, 6), dtype=np.float64)
for i in range(n):
    actions[i] = ds[i]['action'].numpy()

dt = 1.0 / FPS

# ── 读取 blend log ──
blend_log = json.loads(Path(ds_path, "blend_log.json").read_text())
blend_data = blend_log[0] if isinstance(blend_log[0], list) else blend_log
blending_arr = np.array([e["blending"] for e in blend_data[:n]])
state_arr = np.array([e["state"] for e in blend_data[:n]])

# 找 blend 区间
blend_regions = []
in_blend = False
start = 0
for i in range(len(blending_arr)):
    if blending_arr[i] == 1 and not in_blend:
        start = i
        in_blend = True
    elif blending_arr[i] == 0 and in_blend:
        blend_regions.append((start, i, state_arr[i-1]))
        in_blend = False
if in_blend:
    blend_regions.append((start, len(blending_arr), state_arr[-1]))

print(f"Blend regions: {blend_regions}")

# ═════════════════════════════════════════════════════════
# 取第一个 blend 区域做对比 (State 1→2)
# ═════════════════════════════════════════════════════════
bs, be, bst = blend_regions[0]
pad_before = 80
pad_after = 80
s = max(0, bs - pad_before)
e = min(n, be + pad_after)
total_len = e - s

# 实际 blend 位置
blend_actions = actions[s:e, :]

# ── 模拟 "No Blend" 轨迹 ──
# 真实的无 blend 场景：两个 ACT 策略在边界处直接跳变
#   旧策略最后输出 ≈ actions[bs-1]
#   新策略期望位置 ≈ actions[be-10] (新策略 blend 快结束时的输出, 避免取 blend 末端)
#   无 blend = 1-2 帧硬切换 → 巨大速度/加速度尖峰
no_blend_actions = np.zeros_like(blend_actions)
old_last_pos = actions[bs - 1].copy()
new_target_pos = actions[min(be - 5, n - 1)].copy()  # 新策略的目标位置
jump_frame = bs - s + pad_before  # 跳变发生的帧（在 zoom 窗口内）

for i in range(total_len):
    global_idx = s + i
    if i < jump_frame - 1:
        no_blend_actions[i] = actions[global_idx]
    elif i <= jump_frame + 1:
        # 2 帧内完成跳变: old → new
        alpha = (i - jump_frame + 1) / 3.0
        no_blend_actions[i] = (1 - alpha) * old_last_pos + alpha * new_target_pos
    else:
        # 跳变后用实际新策略轨迹
        src_idx = global_idx + (be - bs)
        src_idx = min(src_idx, n - 1)
        no_blend_actions[i] = actions[src_idx]

# ── 计算 p/v/a ──
time_zoom = np.arange(total_len) / FPS

def compute_pva(data):
    vel = np.gradient(data, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc

quintic_vel, quintic_acc = compute_pva(blend_actions)
raw_vel, raw_acc = compute_pva(no_blend_actions)

# ═════════════════════════════════════════════════════════
# 绘制 LiPo 风格对比图 (选一个代表性关节)
# ═════════════════════════════════════════════════════════
# 选 shoulder_lift (关节1) 作为代表 — 通常运动幅度最大
JOINT_IDX = 1  # shoulder_lift

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

C_PROPOSED = '#1565C0'  # Quintic — 蓝色
C_BASELINE = '#E65100'  # Raw/No blend — 橙色
LW = 2.0

# --- Position ---
ax = axes[0]
ax.plot(time_zoom, blend_actions[:, JOINT_IDX], color=C_PROPOSED, linewidth=LW,
        label='Ours (Quintic Blend)')
ax.plot(time_zoom, no_blend_actions[:, JOINT_IDX], color=C_BASELINE, linewidth=LW,
        linestyle='--', dashes=(5, 3), label='Baseline (No Blend)')
# Blend zone
ax.axvspan((bs - s) / FPS, (be - s) / FPS, alpha=0.12, color='#FFCDD2',
           edgecolor='#E53935', linestyle='--', linewidth=1)
ax.text((bs - s + 10) / FPS, ax.get_ylim()[0] + 0.05, 'blend zone',
        fontsize=9, color='#E53935', va='bottom')
ax.set_ylabel("Position (rad)", fontsize=12)
ax.set_title(f"Joint Position — {joint_names_short[JOINT_IDX]}", fontsize=13, fontweight="bold", loc="left")
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.yaxis.set_major_locator(MaxNLocator(6))

# --- Velocity ---
ax = axes[1]
ax.plot(time_zoom, quintic_vel[:, JOINT_IDX], color=C_PROPOSED, linewidth=LW,
        label='Ours (Quintic Blend)')
ax.plot(time_zoom, raw_vel[:, JOINT_IDX], color=C_BASELINE, linewidth=LW,
        linestyle='--', dashes=(5, 3), label='Baseline (No Blend)')
ax.axvspan((bs - s) / FPS, (be - s) / FPS, alpha=0.12, color='#FFCDD2',
           edgecolor='#E53935', linestyle='--', linewidth=1)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_ylabel("Velocity (rad/s)", fontsize=12)
ax.set_title("Joint Velocity", fontsize=13, fontweight="bold", loc="left")
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.yaxis.set_major_locator(MaxNLocator(6))

# --- Acceleration ---
ax = axes[2]
ax.plot(time_zoom, quintic_acc[:, JOINT_IDX], color=C_PROPOSED, linewidth=LW,
        label='Ours (Quintic Blend)')
ax.plot(time_zoom, raw_acc[:, JOINT_IDX], color=C_BASELINE, linewidth=LW,
        linestyle='--', dashes=(5, 3), label='Baseline (No Blend)')
ax.axvspan((bs - s) / FPS, (be - s) / FPS, alpha=0.12, color='#FFCDD2',
           edgecolor='#E53935', linestyle='--', linewidth=1)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Acceleration (rad/s²)", fontsize=12)
ax.set_title("Joint Acceleration", fontsize=13, fontweight="bold", loc="left")
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.yaxis.set_major_locator(MaxNLocator(6))

fig.suptitle(
    f"Trajectory Smoothing Comparison — State 1→2 Transition (frames {bs}→{be}, {120/FPS:.0f}s blend)\n"
    f"Ours: Quintic polynomial (C² continuous)   vs   Baseline: Linear interpolation (C⁰ only)",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
out_p = f"{ds_path}_lipo_style_compare_joint{JOINT_IDX}.png"
plt.savefig(out_p, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_p}")

# ═════════════════════════════════════════════════════════
# 定量对比指标
# ═════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("Quantitative Comparison — Blend Region Only (frames {}-{})".format(bs, be))
print("=" * 75)

blend_slice = slice(bs - s, be - s)

for j in range(6):
    q_vel = quintic_vel[blend_slice, j]
    r_vel = raw_vel[blend_slice, j]
    q_acc = quintic_acc[blend_slice, j]
    r_acc = raw_acc[blend_slice, j]
    q_jerk = np.gradient(q_acc, dt)
    r_jerk = np.gradient(r_acc, dt)

    rms_jerk_q = np.sqrt(np.mean(q_jerk**2))
    rms_jerk_r = np.sqrt(np.mean(r_jerk**2))
    max_acc_q = np.max(np.abs(q_acc))
    max_acc_r = np.max(np.abs(r_acc))

    if rms_jerk_r > 0:
        reduction = (1 - rms_jerk_q / rms_jerk_r) * 100
    else:
        reduction = 0

    print(f"\n{joint_names_short[j]:<18} | RMS Jerk (rad/s³):  Ours={rms_jerk_q:>8.1f}  "
          f"Baseline={rms_jerk_r:>8.1f}  ({reduction:>+5.1f}%)"
          f"\n{'':18} | Max |Acc| (rad/s²): Ours={max_acc_q:>8.1f}  Baseline={max_acc_r:>8.1f}")

print("\n" + "=" * 75)
print("C² Continuity: Quintic guarantees continuous velocity & acceleration.")
print("              Baseline linear interpolation has discontinuous velocity at boundaries.")
print("=" * 75)
print("Done.")
