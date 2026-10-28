#!/usr/bin/env python
"""把基线数据集写成 RoboSaGA 式背景替换版（离线落盘，训练脚本直接读）。

    # 先备好背景池
    python scripts/make_backgrounds.py --mode solid --n 8 --out backgrounds/solid8
    # 再生成
    python scripts/generate_background_dataset.py --episodes 0 1 \
        --background-root backgrounds/solid8 \
        --mask-root workplace/background_generalization/outputs/foreground \
        --out /tmp/bg_ds --report-dir /tmp/bg_reports

公式就是 `transforms/background_replacement.py` 那一条：

    augmented = mask * image + (1 - mask) * background

## `--cameras` 是本脚本存在的主要理由

用户要回答的命题是「up 和 front 需要的数据增强不一样」。那是个**对照**问题，
所以三种条件必须都能单独跑：`--cameras up` / `--cameras front` / 两个都给。
Stage 2B 的干扰物那一路做不到这件事 —— 它靠 up→front 的单应把同一批干扰物
投到 front，而那个单应**实测解不出来**（见本目录 `CLAUDE.md`），所以那边只能
改 up、front 原样复制。这里每个相机在**自己的图像空间**里替换背景，不需要
跨相机的几何关系，所以 front 可以独立增强。

## 和基线的关系

`observation.state` / `action` / `task` **逐位照抄**，只有图像变。核对交给
`verify_background_dataset.py`，本脚本自己不宣称这一点。
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
BG = HERE.parents[1]
WORKPLACE = HERE.parents[2]
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.video_utils import VideoEncodingManager  # noqa: E402
from object_generalization.scripts.generate_augmented_dataset import _clean_features  # noqa: E402
from background_generalization.datasets.background_dataset import _BackgroundStore  # noqa: E402
from background_generalization.masks.foreground_mask import load_foreground_mask  # noqa: E402
from background_generalization.masks.robot_mask import _to_hwc_uint8  # noqa: E402
from background_generalization.transforms.background_replacement import (  # noqa: E402
    apply_background_replacement,
)

CAM = {"up": "observation.images.up", "front": "observation.images.front"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--repo-id", default=None, help="输出的 repo_id，默认 <base>_bg")
    ap.add_argument("--episodes", nargs="+", default=["0"])
    ap.add_argument("--cameras", nargs="+", choices=["up", "front"], default=["up", "front"],
                    help="哪些相机做替换。没点名的相机逐位复制 —— 这是用户命题的对照组。")
    ap.add_argument("--background-root", default=None,
                    help=f"背景图目录。不给就用 `_BackgroundStore` 的纯色兜底（{BG}/backgrounds）")
    ap.add_argument("--mask-root", default=str(BG / "outputs" / "foreground"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="每集只用前 N 帧（冒烟用）")
    ap.add_argument("--no-mask-policy", choices=["fail", "passthrough"], default="fail",
                    help="某集某相机没有前景掩码时怎么办：fail=当场停（默认）；"
                         "passthrough=这一档不增强、原样复制，逐档记进报告并被下游 verify "
                         "单列成豁免。方向上是红线第三条「宁可漏增强，不要错增强」")
    ap.add_argument("--dry-run", action="store_true", help="只报「哪个 episode 会配到哪张背景」")
    args = ap.parse_args()

    if args.dataset is None or args.root is None:
        tc = json.loads((Path(args.ckpt) / "train_config.json").read_text())
        args.dataset = args.dataset or tc["dataset"]["repo_id"]
        args.root = args.root or tc["dataset"]["root"]
    args.repo_id = args.repo_id or f"{args.dataset}_bg"

    base = LeRobotDataset(args.dataset, root=args.root)
    n_ep = base.meta.total_episodes
    eps = list(range(n_ep)) if args.episodes == ["all"] else [int(e) for e in args.episodes]
    cams = [CAM[c] for c in args.cameras]

    store = _BackgroundStore(args.background_root, seed=args.seed)
    if not store.paths:
        print(f"[warn] {args.background_root or '(未给)'} 里没有图 —— 会退化成逐集纯色。"
              f"纯色是合法的受控背景，但请确认这是你要的那一档。")

    print(f"[cfg] 基线 {args.dataset} @ {args.root}")
    print(f"[cfg] 替换相机 {args.cameras}；掩码 {args.mask_root}")
    print(f"[cfg] 背景池 {len(store.paths)} 张 @ {args.background_root}")
    print(f"[cfg] 输出 {args.out}  repo_id={args.repo_id}")

    if args.dry_run:
        for ep in eps:
            got = store.paths[ep % len(store.paths)].name if store.paths else f"纯色#{ep % 8}"
            print(f"  ep{ep:3d} -> {got}")
        got_eps = {ep % len(store.paths) if store.paths else ep % 8 for ep in eps}
        print(f"\n{len(eps)} 集用到 {len(got_eps)} 种背景")
        return

    if os.path.exists(args.out):
        raise SystemExit(f"输出目录已存在：{args.out}（LeRobotDataset.create 不能续跑，先挪走）")

    out = LeRobotDataset.create(
        args.repo_id,
        fps=base.fps,
        features=_clean_features(base.features),
        root=args.out,
        robot_type=base.meta.info.get("robot_type"),
        use_videos=True,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    pass_all = []                      # 逐 (集, 相机) 记录"这一档没增强"，给下游 verify 用
    t_all = time.time()

    with VideoEncodingManager(out):
        for gi, ep in enumerate(eps):
            t0 = time.time()
            ds = LeRobotDataset(args.dataset, root=args.root, episodes=[ep])
            n_fr = len(ds) if args.limit is None else min(len(ds), args.limit)

            # 掩码在这里读**一次**，缺了就当场停 —— **静默**跳过会让这个 episode
            # 变成基线的逐位复制品，却占着"已增强"的编号（NEXT_STEPS 记过这个坑）。
            #
            # `--no-mask-policy passthrough` 做的就是同一件事，但**不静默**：没掩码的
            # 那一档不增强、原样复制，逐档写进报告；下游 `verify_background_dataset.py`
            # 会把它单列成"豁免"（那条 `背景只变了 N%` 的检查不允许它默默过关）。
            # 方向上这是红线第三条「宁可漏增强，不要错增强」——不增强属于漏，属于安全侧。
            masks, pass_cams = {}, []
            for c in cams:
                m = load_foreground_mask(args.mask_root, c, ep)
                if m is None:
                    if args.no_mask_policy == "passthrough":
                        pass_cams.append(c)
                        print(f"  [passthrough] ep{ep} {c.split('.')[-1]}: 没有前景掩码 "
                              f"→ 这一档**不增强**（原样复制），会记进报告")
                        continue
                    raise SystemExit(
                        f"没有前景掩码 {args.mask_root}/{c}/episode_{ep:06d}.npy\n"
                        f"先跑：scripts/build_robot_mask_cache.py --episodes {ep}\n"
                        f"      scripts/build_foreground_mask.py --episodes {ep}\n"
                        f"（确实想让这一档不增强，加 --no-mask-policy passthrough）")
                if len(m) < n_fr:
                    # 掩码**存在但短了**是另一回事（真bug），不归 passthrough 管
                    raise SystemExit(f"{c} ep{ep} 掩码只有 {len(m)} 帧，要 {n_fr} 帧")
                masks[c] = m
            pass_set = set(pass_cams)
            for c in pass_cams:
                pass_all.append({"episode": ep, "camera": c.split(".")[-1]})

            changed = {c: [] for c in cams if c not in pass_set}
            for fr in range(n_fr):
                item = ds[fr]
                frame = {
                    "observation.state": item["observation.state"],
                    "action": item["action"],
                    "task": item["task"],
                }
                for c in ds.meta.camera_keys:
                    img = item[c]
                    h, w = img.shape[-2], img.shape[-1]
                    if c in cams and c not in pass_set:
                        bgc = store.get(ep, h, w)
                        res = apply_background_replacement(img, masks[c][fr], bgc)
                        changed[c].append(
                            float(((res - img.float()).abs().max(0).values > 1e-6).float().mean()))
                        frame[c] = _to_hwc_uint8(res)
                    else:
                        frame[c] = _to_hwc_uint8(img)
                out.add_frame(frame)

            out.save_episode()
            bg_name = store.paths[ep % len(store.paths)].name if store.paths else f"solid#{ep % 8}"
            rec = {
                "gen_episode": gi, "base_episode": ep, "frames": n_fr,
                "background": bg_name,
                "changed_frac": {c.split(".")[-1]: round(float(np.mean(v)), 4)
                                 for c, v in changed.items()},
                "passthrough": [c.split(".")[-1] for c in pass_cams],
                "cameras": args.cameras,
                "seconds": round(time.time() - t0, 1),
            }
            recs.append(rec)
            print(f"  gen{gi} (base {ep}): 背景={bg_name} 改了 "
                  + " ".join(f"{k}={v:.1%}" for k, v in rec["changed_frac"].items())
                  + (f"  不增强={','.join(rec['passthrough'])}" if rec["passthrough"] else "")
                  + f"  {rec['seconds']}s")
            del ds

    (report_dir / "generate.json").write_text(json.dumps(
        {"args": vars(args), "episodes": recs, "passthrough": pass_all,
         "total_seconds": round(time.time() - t_all, 1)}, ensure_ascii=False, indent=2))
    if pass_all:
        n_arm = len(eps) * len(cams)
        print(f"\n[!!] {len(pass_all)}/{n_arm} 档（集×相机）**没有增强**："
              + ", ".join(f"ep{p['episode']}{p['camera'][0]}" for p in pass_all))
        print(f"     这些档把**原背景**喂进了训练集 → 处理被稀释 "
              f"{len(pass_all)/n_arm:.1%}。对照臂之间必须一致（要么都 passthrough、"
              f"要么都删同一批集），否则跨臂比较的 episode 集合不同。")
    print(f"\n[save] {args.out}")
    print(f"[save] {report_dir / 'generate.json'}")
    print(f"[next] verify_background_dataset.py --report-dir {report_dir} --base-root {args.root}")


if __name__ == "__main__":
    main()
