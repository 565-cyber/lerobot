#!/usr/bin/env python
"""逐集生成「机械臂/夹爪」掩码（SAM2），供 action saliency 用。

    python scripts/build_robot_mask_cache.py --episodes 0 --vis
    python scripts/build_robot_mask_cache.py --episodes all --skip-existing

为什么需要它、为什么不能用时域统计，见 `masks/robot_mask.py` 的模块文档。
一句话：**front 相机装在手臂上**，机械臂和场景在它画面里**都在动**，任何
时域/运动统计量都分不开，只能显式分割。

输出（uint8 (T,H,W)，**255 = 机械臂/夹爪**）：

    {out}/{observation.images.up}/episode_{ep:06d}.npy
    {out}/{observation.images.front}/episode_{ep:06d}.npy

自检不过的相机会**记进报告并画出来**，不会静默产出坏掩码。
"""

from __future__ import annotations

import argparse
import json
import os
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
from background_generalization.masks.robot_mask import (  # noqa: E402
    RobotMaskConfig,
    build_robot_mask,
    save_robot_mask,
    _to_hwc_uint8,
)

CAM_KEYS = ["observation.images.up", "observation.images.front"]


def episode_span(ds, ep: int) -> tuple[int, int]:
    row = ds.meta.episodes.to_pandas().iloc[ep]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def render(frames: dict, masks: dict, report: dict, out_png: Path, ep: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    cams = list(frames.keys())
    T = len(frames[cams[0]])
    show = np.linspace(0, T - 1, 6).astype(int).tolist()
    fig, axes = plt.subplots(len(cams) * 2, len(show), figsize=(3.1 * len(show), 9.4),
                             squeeze=False)
    for ci, c in enumerate(cams):
        r = report[c]
        for j, k in enumerate(show):
            img = np.clip(_to_hwc_uint8(frames[c][k]) / 255.0, 0, 1)
            m = masks[c][k]
            for row, paint in ((0, False), (1, True)):
                ax = axes[ci * 2 + row][j]
                ax.imshow(img)
                if paint:
                    ax.imshow(np.ma.masked_where(~m, m), cmap="winter", alpha=0.5)
                    ax.set_title(f"面积 {m.mean():.1%}", fontsize=9)
                else:
                    ax.set_title(f"{c.split('.')[-1]}  t={k}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        p = r["prompt"]
        tag = "OK" if r["ok"] else "自检不过: " + "; ".join(r["problems"])
        axes[ci * 2][0].set_ylabel(f"{c.split('.')[-1]}\n{tag}", fontsize=9)
    fig.suptitle(f"robot mask ep{ep} —— 蓝 = SAM2 分割出的机械臂/夹爪（提示："
                 f"up={report[CAM_KEYS[0]]['prompt']['tier']}, "
                 f"front={report[CAM_KEYS[1]]['prompt']['tier']}）", fontsize=12)
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
    ap.add_argument("--out", default=str(BG / "outputs" / "robot_mask"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force", action="store_true", help="自检不过也落盘（默认跳过并报警）")
    # 默认 0 = 关。开了以后第一档提示自检不过会换提示重跑（最多再试 --retries 档），
    # 每档都过同一个自检，所以最坏结果不变、只是多花 GPU 时间。
    ap.add_argument("--retries", type=int, default=0,
                    help="自检不过时最多再换几档提示重试（0=不重试，与原来逐位相同）")
    ap.add_argument("--vis", action="store_true", help="画验证图")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    if args.repo_id is None or args.root is None:
        tc = json.loads((Path(args.ckpt) / "train_config.json").read_text())
        args.repo_id = args.repo_id or tc["dataset"]["repo_id"]
        args.root = args.root or tc["dataset"]["root"]

    ds = LeRobotDataset(args.repo_id, root=args.root)
    n_ep = ds.meta.total_episodes
    eps = list(range(n_ep)) if args.episodes == ["all"] else [int(e) for e in args.episodes]
    report_path = Path(args.report or (Path(args.out) / "reports" / "robot_mask.jsonl"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = RobotMaskConfig()
    print(f"[cfg] {args.repo_id} @ {args.root}  共 {n_ep} 集")

    n_bad = 0

    for ep in eps:
        lo, hi = episode_span(ds, ep)
        todo = [c for c in CAM_KEYS
                if not (args.skip_existing
                        and (Path(args.out) / c / f"episode_{ep:06d}.npy").exists())]
        if not todo:
            print(f"[skip] ep{ep} 已存在")
            continue

        t0 = time.time()
        items = [ds[i] for i in range(lo, hi)]
        frames = {c: np.stack([it[c].numpy() for it in items]) for c in CAM_KEYS}
        masks, report = {}, {}
        for c in CAM_KEYS:
            if c not in todo and args.skip_existing:
                continue
            m, r = build_robot_mask(frames[c], c, cfg, device=args.device,
                                    retries=args.retries)
            masks[c] = m > 0
            report[c] = r
            flag = "OK" if r["ok"] else "!! " + " | ".join(r["problems"])
            retry_tag = f" [重试 {r['n_attempts']} 档]" if r.get("retried") else ""
            print(f"[ep{ep}] {c}: 提示={r['prompt']['tier']} 帧={r['prompt']['frame']} "
                  f"框={r['prompt']['box']} 面积中位={r['area_median']:.1%} "
                  f"覆盖帧={r['valid_frac']:.0%}{retry_tag}  {flag}")
            # 自检不过**不落盘**。这里以前是无条件 save，而一份空掩码能一路骗过
            # 下游：`build_foreground_mask` 只查 union 的面积和覆盖率，而任务物
            # （红桶 ~14%、蓝 ~0.6%）自己就超过它的 `min_area_frac=0.015`、
            # valid_frac 也是 100%，于是**union 自检通过**、掩码里却没有机械臂
            # —— 合成时把机械臂换成背景色，不报任何错。实测 ep5 front：
            # 覆盖帧 0.4%、面积中位 0.00%。这正是「宁可漏增强，不要错增强」
            # 要挡住的方向，和 `build_foreground_mask.py` 同一个处理。
            if r["ok"] or args.force:
                save_robot_mask(args.out, c, ep, m)
            else:
                n_bad += 1
                print("       ** 没落盘 **（加 --force 可强制）。这份掩码会让"
                      "机械臂进入背景区，不报错，所以宁可停下。")

        rec = {"episode": ep, "report": report, "seconds": round(time.time() - t0, 1)}
        with open(report_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if args.vis:
            render(frames, masks, report, Path(args.out) / "vis" / f"episode_{ep:06d}.png", ep)

    print(f"\n[save] {args.out}\n[save] {report_path}")
    if n_bad:
        print(f"[!!] 有 {n_bad} 份自检不过、**没有落盘**。这些 episode 走不到下游，"
              f"修好再跑（`--skip-existing` 不会补它们，因为文件根本不存在）。")
        sys.exit(1)


if __name__ == "__main__":
    main()
