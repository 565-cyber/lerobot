#!/usr/bin/env python
"""构建「双路（dual-stream）」视觉提示数据集，对应论文 Figure 3 的架构。

论文 Figure 3：ACT 同时接收「原始 RGB」和「带框 RGB」两路输入，各自提特征后融合。
本脚本把这份数据集做出来，最终每个 frame 有 3 个相机键：

    observation.images.up           = 原始（无框）
    observation.images.up_prompted  = 带框（目标绿色框，来自 prompted 版的 up）
    observation.images.front        = 原始（无框，不变）

关键点（已在代码里核实）：
  * ACT 天然支持多个 observation.images.* 键：每个键 = 一路相机，共享 ResNet backbone
    + 独立相机位置编码，最后拼接进 transformer（modeling_act.py forward 里
    `for img in batch[OBS_IMAGES]`）。所以「加一路输入」不需要改任何模型/配置代码。
  * 训练时 input_features 是从数据集 features 自动派生的（factory.py: `if not
    cfg.input_features: ... dataset_to_policy_features(...)`），dtype=="video" -> VISUAL。
  * 数据帧（action/state/timestamp 等）在 data/chunk-*.parquet 里，跟视频无关，原样
    复制即可。视频引用只在 meta/episodes 里（chunk_index/file_index/timestamps）。
  * load_episodes 会丢弃 stats/* 列，所以 episodes 里的 per-episode 统计非必需；
    但 meta/stats.json 里的全局统计是归一化用的，必须给 up_prompted 补一份。

本脚本【只读】源数据集，只往新目录写，绝不改 newdata1 / newdata1_prompted。

用法:
  python workplace/make_dual_stream.py \
      --raw dataset/newdata1 \
      --prompted dataset/newdata1_prompted \
      --out dataset/newdata1_dual
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

UP_KEY = "observation.images.up"
UP_PROMPTED_KEY = "observation.images.up_prompted"


def patch_info_json(out: Path) -> None:
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert UP_KEY in info["features"], f"{UP_KEY} 不在 info.json features 里"
    if UP_PROMPTED_KEY in info["features"]:
        raise RuntimeError(f"{UP_PROMPTED_KEY} 已存在于 info.json，无需重复添加")
    info["features"][UP_PROMPTED_KEY] = copy.deepcopy(info["features"][UP_KEY])
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[info.json] 已添加 {UP_PROMPTED_KEY}")


def patch_episodes_parquet(out: Path) -> None:
    """给每集的 meta/episodes/*.parquet 增加 up_prompted 的 4 个视频引用列。

    值与 up 完全一致（同一帧、同一 chunk/file、同一时间戳），因为 up_prompted
    只是 up 的带框重编码，帧数和时长相同。
    """
    cols = ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]
    for f in sorted((out / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(f)
        for suffix in cols:
            src = f"videos/{UP_KEY}/{suffix}"
            dst = f"videos/{UP_PROMPTED_KEY}/{suffix}"
            if dst in table.column_names:
                raise RuntimeError(f"{dst} 已存在于 {f.name}，无需重复添加")
            assert src in table.column_names, f"{src} 不在 {f.name} 里"
            table = table.append_column(dst, table.column(src))
        pq.write_table(table, f)
        print(f"[episodes] {f.name} 已增加 up_prompted 的 4 个视频引用列")


def patch_stats_json(out: Path) -> None:
    stats_path = out / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    assert UP_KEY in stats, f"{UP_KEY} 不在 stats.json 里"
    if UP_PROMPTED_KEY in stats:
        raise RuntimeError(f"{UP_PROMPTED_KEY} 已存在于 stats.json")
    # 归一化用 up 的统计即可：up_prompted 只是 up + 少量绿色框像素，分布几乎一致，
    # 用同一套 mean/std 归一化反而更利于 raw/prompted 两路特征对齐。
    stats[UP_PROMPTED_KEY] = copy.deepcopy(stats[UP_KEY])
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[stats.json] 已添加 {UP_PROMPTED_KEY}")


def main() -> None:
    ap = argparse.ArgumentParser(description="构建双路视觉提示数据集（up raw + up_prompted + front raw）")
    ap.add_argument("--raw", required=True, help="原始无框数据集目录")
    ap.add_argument("--prompted", required=True, help="已带框的 prompted 数据集目录")
    ap.add_argument("--out", required=True, help="输出数据集目录（必须不存在）")
    args = ap.parse_args()

    raw = Path(args.raw)
    prompted = Path(args.prompted)
    out = Path(args.out)

    if out.exists():
        raise FileExistsError(f"{out} 已存在，请换 --out 或先删除")
    if not (raw / "meta" / "info.json").exists():
        raise FileNotFoundError(f"{raw} 不是有效的 LeRobot 数据集（缺 meta/info.json）")
    if not (prompted / "meta" / "info.json").exists():
        raise FileNotFoundError(f"{prompted} 不是有效的 LeRobot 数据集（缺 meta/info.json）")

    src_up = prompted / "videos" / UP_KEY
    if not src_up.exists():
        raise FileNotFoundError(f"{prompted} 缺 {UP_KEY} 视频，请先确认 prompted 版已标注 up 相机")

    # 1) 复制原始数据集（data / meta / up / front 全部）
    print(f"复制 {raw} -> {out} ...")
    shutil.copytree(raw, out)

    # 2) 把 prompted 版的 up 视频复制为 out 的 up_prompted 视频
    dst_up_prompted = out / "videos" / UP_PROMPTED_KEY
    print(f"复制 prompted 的 up 视频 -> {dst_up_prompted} ...")
    shutil.copytree(src_up, dst_up_prompted)

    # 3) 元信息补齐
    patch_info_json(out)
    patch_episodes_parquet(out)
    patch_stats_json(out)

    print(f"\n完成。双路数据集已生成: {out}")
    print(f"  视频键: {UP_KEY} (raw) + {UP_PROMPTED_KEY} (boxed) + observation.images.front (raw)")
    print("  训练时把 --dataset.repo_id 指向它即可，无需改模型/配置。")


if __name__ == "__main__":
    main()
