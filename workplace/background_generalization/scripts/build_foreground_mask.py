#!/usr/bin/env python
"""合成逐集逐相机的「前景（keep）掩码」，供 RoboSaGA 式背景替换用。

    # 先有机械臂掩码（SAM2，慢）
    python scripts/build_robot_mask_cache.py --episodes 0 1 --vis
    # 再合成前景（HSV，快）
    python scripts/build_foreground_mask.py --episodes 0 1 --vis

输出（uint8 (T,H,W)，**0/1，1 = 保留**）：

    {out}/{observation.images.up}/episode_{ep:06d}.npy
    {out}/{observation.images.front}/episode_{ep:06d}.npy

这个是 `BackgroundAugmentedDataset` 的 `mask_root`。路径和 0/1 约定都走
`masks/mask_utils.py`，所以那边不用改一行就能读。

自检不过的相机会**记进报告并跳过落盘**（`--force` 可覆盖）—— 拿一份
`ok=False` 的掩码去合成，等于让机械臂被替换掉，而且不会有任何报错。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
BG = HERE.parents[1]                # workplace/background_generalization
WORKPLACE = HERE.parents[2]
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from background_generalization.masks.foreground_mask import (  # noqa: E402
    ForegroundMaskConfig,
    build_foreground_mask,
    save_foreground_mask,
)
from background_generalization.masks.robot_mask import _to_hwc_uint8  # noqa: E402

CAM_KEYS = ["observation.images.up", "observation.images.front"]


def episode_span(ds, ep: int) -> tuple[int, int]:
    row = ds.meta.episodes.to_pandas().iloc[ep]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def render(frames: np.ndarray, mask: np.ndarray, report: dict, out_png: Path, ep: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    T = len(frames)
    show = np.linspace(0, T - 1, 6).astype(int).tolist()
    fig, axes = plt.subplots(2, len(show), figsize=(3.1 * len(show), 6.4), squeeze=False)
    c = report["camera"].split(".")[-1]
    for j, k in enumerate(show):
        img = np.clip(_to_hwc_uint8(frames[k]) / 255.0, 0, 1)
        m = mask[k] > 0
        axes[0][j].imshow(img)
        axes[0][j].set_title(f"t={k}", fontsize=9)
        axes[1][j].imshow(img)
        # 半透明绿=保留。要看的是「机械臂、夹爪、黄方块、红桶、蓝 有没有漏在掩码外」
        axes[1][j].imshow(np.ma.masked_where(~m, m), cmap="summer", alpha=0.45)
        axes[1][j].set_title(f"保留 {m.mean():.1%}", fontsize=9)
        for r in (0, 1):
            axes[r][j].set_xticks([]); axes[r][j].set_yticks([])
    tag = "OK" if report["ok"] else "自检不过: " + "; ".join(report["problems"])
    comp = report["components"]
    fig.suptitle(f"前景掩码 ep{ep} {c} —— 绿=1(保留)　{tag}\n"
                 f"机械臂 {comp['robot']:.1%} / 黄 {comp.get('yellow', 0):.2%} / "
                 f"红 {comp.get('red', 0):.1%} / 蓝 {comp.get('blue', 0):.1%}"
                 f"　外扩多盖 {report['grow_overhead']:.1%}", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=95)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--episodes", nargs="+", default=["0"])
    # 机械臂掩码缓存（build_robot_mask_cache.py --out 的那个）
    ap.add_argument("--robot-root", default=str(BG / "outputs" / "robot_mask"))
    ap.add_argument("--out", default=str(BG / "outputs" / "foreground"))
    ap.add_argument("--dilate-px", type=int, default=24)
    ap.add_argument("--task-dilate-px", type=int, default=3)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force", action="store_true", help="自检不过也落盘（默认跳过并报警）")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    if args.repo_id is None or args.root is None:
        tc = json.loads((Path(args.ckpt) / "train_config.json").read_text())
        args.repo_id = args.repo_id or tc["dataset"]["repo_id"]
        args.root = args.root or tc["dataset"]["root"]

    ds = LeRobotDataset(args.repo_id, root=args.root)
    n_ep = ds.meta.total_episodes
    eps = list(range(n_ep)) if args.episodes == ["all"] else [int(e) for e in args.episodes]
    report_path = Path(args.report or (Path(args.out) / "reports" / "foreground_mask.jsonl"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ForegroundMaskConfig(robot_root=args.robot_root,
                               dilate_px=args.dilate_px,
                               task_dilate_px=args.task_dilate_px)
    print(f"[cfg] {args.repo_id} @ {args.root}  共 {n_ep} 集")
    print(f"[cfg] {json.dumps(cfg.as_dict(), ensure_ascii=False)}")

    n_ok = n_bad = 0
    for ep in eps:
        lo, hi = episode_span(ds, ep)
        for c in CAM_KEYS:
            dst = Path(args.out) / c / f"episode_{ep:06d}.npy"
            if args.skip_existing and dst.exists():
                print(f"[skip] ep{ep} {c.split('.')[-1]} 已存在")
                continue

            t0 = time.time()
            # 一次只把一台相机的帧留在内存里：一盏 480x640 float32 的 511 帧
            # 是 1.9 GB，两路一起就是 3.8 GB。前面那个脚本两路一起扛，
            # 这里没必要。
            items = [ds[i] for i in range(lo, hi)]
            frames = np.stack([it[c].numpy() for it in items])
            del items

            mask, report = build_foreground_mask(frames, c, ep, cfg)
            report["seconds"] = round(time.time() - t0, 1)
            comp = report["components"]
            flag = "OK" if report["ok"] else "!! " + " | ".join(report["problems"])
            print(f"[ep{ep}] {c.split('.')[-1]}: 面积中位={report['area_median']:.1%} "
                  f"范围={report['area_min']:.1%}-{report['area_max']:.1%} "
                  f"覆盖帧={report['valid_frac']:.0%} | 机械臂={comp['robot']:.1%} "
                  f"黄={comp.get('yellow', 0):.2%} 红={comp.get('red', 0):.1%} "
                  f"蓝={comp.get('blue', 0):.1%} | 外扩多盖={report['grow_overhead']:.1%} "
                  f"| {flag}")

            if report["ok"] or args.force:
                save_foreground_mask(args.out, c, ep, mask)
                n_ok += 1
            else:
                n_bad += 1
                print(f"       ** 没落盘 **（加 --force 可强制）。这份掩码会让不该被替换的"
                      f"像素进入背景区，不报错，所以宁可停下。")

            with open(report_path, "a") as f:
                f.write(json.dumps(report, ensure_ascii=False) + "\n")
            if args.vis:
                render(frames, mask, report, Path(args.out) / "vis" / f"ep{ep:03d}_{c.split('.')[-1]}.png", ep)
            del frames, mask

    print(f"\n[save] {args.out}   落盘 {n_ok}，自检不过跳过 {n_bad}")
    print(f"[save] {report_path}")
    if n_bad:
        # 非零退出，和 `build_robot_mask_cache.py` 同一个契约。以前这里只打印一行
        # 就 0 退出，于是"跑完了"和"跑完但少了几集"在下游看起来一样：接链条的
        # 时候会直接往下走去做数据集，而 `generate_background_dataset.py` 要到
        # **建好输出目录之后**才在缺掩码的那一集上 SystemExit —— 那个目录还不能
        # 续跑（`mkdir(exist_ok=False)`），只能挪走重来。在这里断掉便宜得多。
        print(f"[!!] 有 {n_bad} 份没落盘。先修，再往下走。")
        sys.exit(1)


if __name__ == "__main__":
    main()
