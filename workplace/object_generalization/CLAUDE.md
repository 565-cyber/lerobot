# CLAUDE.md

目标颜色泛化：研究只改变抓取目标颜色（蓝方块 → 绿方块）是否会让 vanilla ACT
掉成功率，以及用蓝→绿增强的演示数据训练能否减小掉点。本质是一种标签保持的视觉
增强 `(T_color(I), s, a)`：只在 mask 内旋转被跟踪方块的色相（蓝 H≈120 → 绿
H≈60），位置/形状/阴影/状态/动作都不变。细节、覆盖率数据和生成的数据集都记录在
本目录的 `README.md`。

## 运行脚本

```bash
cd /home/zjq/lerobot
/home/zjq/miniconda3/envs/lerobot/bin/python workplace/object_generalization/scripts/<script>.py
```

## 踩坑记录

- `datasets` 模块遮蔽：本目录的 `datasets/` 会在 `workplace/` 加入 `sys.path` 时
  遮蔽 huggingface 的 `datasets` 包。脚本里要把仓库根目录和 `workplace/` 插到
  `sys.path` **最前面**，并先 import `lerobot.datasets.lerobot_dataset`（参考
  `scripts/` 里任意脚本的文件头）。
- 数据集写入格式：传给 `LeRobotDataset.add_frame` 的图像必须是 **(H, W, 3)
  uint8**（不是 C,H,W float），对应 `(480, 640, 3)` 的相机 feature。
- `add_frame` 会自动算 `frame_index` 和 `timestamp`；frame 字典里不要再放
  `timestamp` 键（`validate_frame` 会报 "Extra feature"）。
- SAM2 config 路径是包相对路径：`"configs/sam2.1/sam2.1_hiera_b+.yaml"`，
  权重 `checkpoints/sam2.1_hiera_base_plus.pt`。用
  `offload_video_to_cpu=True` / `offload_state_to_cpu=True` 和
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 避免 2080 Ti 上 OOM。
