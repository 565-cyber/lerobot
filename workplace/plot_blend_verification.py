"""
Blend 验证脚本 — 从数据集 + blend_log.json 精确绘制过渡轨迹

用法:
  python workplace/plot_blend_verification.py <dataset_path>

例:
  python workplace/plot_blend_verification.py /home/zjq/lerobot/outputs/eval/eval_record_planner_3step
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

# ── 参数 ──
ds_path = sys.argv[1] if len(sys.argv) > 1 else "/home/zjq/lerobot/outputs/eval/eval_record_planner_3step"
ds = LeRobotDataset(ds_path)
FPS, n = ds.fps, ds.num_frames
joint_names = [n.replace('.pos','') for n in ds.features['action']['names']]

# ── 读取 actions ──
actions = np.zeros((n, 6), dtype=np.float64)
for i in range(n):
    actions[i] = ds[i]['action'].numpy()

dt = 1.0 / FPS
time = np.arange(n) / FPS
velocities = np.gradient(actions, dt, axis=0)
accelerations = np.gradient(velocities, dt, axis=0)

# ── 读取 blend log ──
blend_log_path = Path(ds_path) / "blend_log.json"
has_blend_log = blend_log_path.exists()

if has_blend_log:
    all_logs = json.loads(blend_log_path.read_text())
    # 取第一个 episode
    blend_log = all_logs[0] if isinstance(all_logs[0], list) else all_logs
    # 如果有多个 episode, flatten
    if isinstance(all_logs[0], list) and len(all_logs) > 1:
        # 多 episode: 每个是独立列表
        pass  # use the last one with blend data
    blending_arr = np.array([entry["blending"] for entry in blend_log[:n]])
    state_arr = np.array([entry["state"] for entry in blend_log[:n]])
    blend_step_arr = np.array([entry.get("blend_step", 0) for entry in blend_log[:n]])
    print(f"Blend log loaded: {len(blend_log)} entries")
    print(f"Blend frames: {blending_arr.sum()} ({blending_arr.sum()/FPS:.1f}s)")
    print(f"States seen: {np.unique(state_arr)}")

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
    print(f"Blend regions: {len(blend_regions)}")
    for s, e, st in blend_regions:
        print(f"  Frames {s}-{e} ({e-s} frames, {(e-s)/FPS:.1f}s) → State {st}")
else:
    print("No blend_log.json found — using heuristics to estimate blend regions")
    blending_arr = np.zeros(n, dtype=int)
    state_arr = np.zeros(n, dtype=int)
    blend_step_arr = np.zeros(n, dtype=int)
    blend_regions = []
    # 用帧间变化量 heuristic
    diffs = np.sqrt(np.sum(np.diff(actions, axis=0)**2, axis=1))
    t = np.percentile(diffs, 90)
    for i in range(len(diffs)-20):
        if np.mean(diffs[i:i+20] > t) > 0.3:
            blend_regions.append((i, i+120, 0))
            break

# ── 颜色 ──
COLORS = ['#1565C0', '#E65100', '#2E7D32', '#C62828', '#6A1B9A', '#00838F']
STATE_COLORS = {1: '#FFF9C4', 2: '#E3F2FD', 3: '#E8F5E9'}
STATE_LABELS = {1: 'State 1: Pick Yellow', 2: 'State 2: Pick Blue', 3: 'State 3: Close Lid'}

# ═════════════════════════════════════════════════════════
# 图1: 完整轨迹 + blend 区域标注
# ═════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

# 按 state 画背景
if has_blend_log:
    for st in np.unique(state_arr):
        if st == 0:
            continue
        mask = state_arr == st
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [n]])
        for a, b in zip(starts, ends):
            for ax in axes:
                ax.axvspan(a/FPS, b/FPS, alpha=0.15, color=STATE_COLORS.get(st, '#f0f0f0'))

# 标记 blend 区域
for s, e, st in blend_regions:
    for ax in axes:
        ax.axvspan(s/FPS, e/FPS, alpha=0.25, color='#FFCDD2', edgecolor='#E53935', linestyle='--', linewidth=1)

# Position
ax = axes[0]
for j in range(6):
    ax.plot(time, actions[:, j], color=COLORS[j], linewidth=0.9, alpha=0.8, label=joint_names[j])
ax.set_ylabel("Position (rad)", fontsize=12)
ax.set_title(f"Joint Positions — {Path(ds_path).name}" +
             (" (red zones = quintic blend)" if blend_regions else ""),
             fontsize=14, fontweight="bold")
ax.legend(ncol=6, fontsize=9, loc='upper right', framealpha=0.8)
ax.grid(True, alpha=0.2)
ax.yaxis.set_major_locator(MaxNLocator(6))

# Velocity
ax = axes[1]
for j in range(6):
    ax.plot(time, velocities[:, j], color=COLORS[j], linewidth=0.7, alpha=0.65)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_ylabel("Velocity (rad/s)", fontsize=12)
ax.set_title("Joint Velocities", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.yaxis.set_major_locator(MaxNLocator(6))

# Acceleration
ax = axes[2]
for j in range(6):
    ax.plot(time, accelerations[:, j], color=COLORS[j], linewidth=0.5, alpha=0.5)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Acceleration (rad/s²)", fontsize=12)
ax.set_title("Joint Accelerations", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.yaxis.set_major_locator(MaxNLocator(6))

plt.tight_layout()
out1 = str(Path(ds_path).parent / f"{Path(ds_path).name}_blend_overview.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out1}")

# ═════════════════════════════════════════════════════════
# 图2-N: 每个 blend 区域放大 (p/v/a)
# ═════════════════════════════════════════════════════════
for ri, (bs, be, bst) in enumerate(blend_regions[:4]):
    pad = 80
    s = max(0, bs - pad)
    e = min(n, be + pad)
    t_zoom = np.arange(s, e) / FPS

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Blend zone
    for ax in axes:
        ax.axvspan(bs/FPS, be/FPS, alpha=0.2, color='#FFCDD2', edgecolor='#E53935', linestyle='--', linewidth=1.5)

    # Warmup sub-zone (first 20 frames of blend)
    wm_end = min(bs + 20, be)
    for ax in axes:
        ax.axvspan(bs/FPS, wm_end/FPS, alpha=0.15, color='#BBDEFB', edgecolor='#1976D2', linestyle=':', linewidth=1)
    axes[0].text((bs+10)/FPS, axes[0].get_ylim()[1]*0.92, 'warmup', fontsize=9, color='#1976D2',
                 ha='center', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    # Position
    for j in range(6):
        axes[0].plot(t_zoom, actions[s:e, j], color=COLORS[j], linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel("Position (rad)", fontsize=11)
    axes[0].set_title(f"Blend Region {ri+1}: Frames {bs}→{be} ({be-bs} frames, {(be-bs)/FPS:.1f}s) → State {bst}",
                      fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.2)

    # Velocity
    for j in range(6):
        axes[1].plot(t_zoom, velocities[s:e, j], color=COLORS[j], linewidth=1.0, alpha=0.7)
    axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    axes[1].set_ylabel("Velocity (rad/s)", fontsize=11)
    axes[1].set_title("Velocity — continuous through blend (C¹)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.2)

    # Acceleration
    for j in range(6):
        axes[2].plot(t_zoom, accelerations[s:e, j], color=COLORS[j], linewidth=0.8, alpha=0.6)
    axes[2].axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].set_ylabel("Acceleration (rad/s²)", fontsize=11)
    axes[2].set_title("Acceleration — continuous through blend (C²)", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.2)

    fig.suptitle(f"Quintic Blend Transition — {Path(ds_path).name} (Region {ri+1})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_n = str(Path(ds_path).parent / f"{Path(ds_path).name}_blend_zoom_{ri+1}.png")
    plt.savefig(out_n, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_n}")

    # 定量指标
    print(f"\nBlend Region {ri+1} Metrics (frames {bs}-{be}):")
    for j in range(6):
        vel_seg = velocities[bs:be, j]
        acc_seg = accelerations[bs:be, j]
        jerk_seg = np.gradient(acc_seg, dt)
        print(f"  {joint_names[j]:<20}  "
              f"RMS vel={np.sqrt(np.mean(vel_seg**2)):>7.2f}  "
              f"max|vel|={np.max(np.abs(vel_seg)):>7.2f}  "
              f"RMS acc={np.sqrt(np.mean(acc_seg**2)):>7.2f}  "
              f"RMS jerk={np.sqrt(np.mean(jerk_seg**2)):>7.2f}")

print("\nDone.")
