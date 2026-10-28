# CLAUDE.md

LeRobot v0.4.2 检出仓库，用于 SO101 真机 + ACT 策略的机器人研究。源码在
`src/lerobot/` 下；自定义/实验代码放在 `workplace/` 下。

子项目各自的约定放在对应目录的 CLAUDE.md 里，例如当前的目标颜色泛化工作见
`workplace/object_generalization/CLAUDE.md`。

## 环境

- Python：`/home/zjq/miniconda3/envs/lerobot/bin/python`（py 3.10.19）
- torch 2.7.1+cu126，CUDA 可用；GPU：2× NVIDIA RTX 2080 Ti
- 数据集放在 `dataset/` 下，用 `LeRobotDataset("<repo_id>", root=...)` 加载。
