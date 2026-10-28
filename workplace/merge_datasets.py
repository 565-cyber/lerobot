#!/usr/bin/env python
"""合并多个本地 LeRobot 数据集为一个（本地路径版）。

官方 `lerobot_edit_dataset.py --operation.type merge` 对本地数据集有 bug：它给所有
源数据集共用一个 --root，而 LeRobot 里 root 应该是「每个数据集自己的完整目录」。
本脚本用底层 merge_datasets 直接给每个数据集传各自的 root，规避这个 bug。

用法:
  python workplace/merge_datasets.py \
      --inputs dataset/newdata1 dataset/newdata1_prompted \
      --output dataset/newdata1_merged

要求：各数据集 features 完全一致（fps / robot_type / 特征键与 shape）。
      注意：info.json 里 video.codec 字段也必须一致，否则 validate_all_metadata 会报
      「Same features is expected」。（annotate_target.py 已保证不改 codec 字段，且
      用与源视频相同的编码器输出，所以原版与 prompted 版可以直接合并。）
"""

import argparse
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser(description="合并多个本地 LeRobot 数据集")
    ap.add_argument("--inputs", nargs="+", required=True, help="源数据集目录（可多个）")
    ap.add_argument("--output", required=True, help="合并后的输出目录")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    output = Path(args.output)

    if output.exists():
        raise FileExistsError(f"{output} 已存在，请换 --output 或先删除")

    datasets = [LeRobotDataset(p.name, root=str(p)) for p in inputs]
    merged = merge_datasets(datasets, output_repo_id=output.name, output_dir=str(output))

    print(f"合并完成: {output}")
    print(f"  episodes = {merged.meta.total_episodes}, frames = {merged.meta.total_frames}")


if __name__ == "__main__":
    main()
