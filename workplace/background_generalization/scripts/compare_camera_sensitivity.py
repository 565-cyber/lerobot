#!/usr/bin/env python
"""把 `action_saliency.py --fg-mode union` 的多集结果汇成「up vs front」的对比表。

    # 先生成（两个相机一起，同一套前景构造）
    for ep in 0 1 2 3 4 5; do
      python scripts/action_saliency.py --episode $ep --frames 32 --fg-mode union \
          --out outputs/saliency/t15_union_ep$ep.json
    done
    # 再汇总
    python scripts/compare_camera_sensitivity.py --glob 'outputs/saliency/t15_union_ep*.json'

## 这个脚本要回答的问题

「up 和 front 需不需要**不同**的数据增强」—— 具体到背景这一轴，就是
「front 对背景是不是没那么敏感」。所以它算两个东西：

1. **整路消融的跨相机比**（每个试验一行）。`bg_const` 是把背景换成一块纯色的那一档，
   也就是增强**本身**做的操作（`make_backgrounds.py --mode solid`），
   它比 `bg_noise` / `bg_swap_ep` 都更贴题。
2. **逐格等面积对比**（每个格子 = 画面的 1/48，同一噪声操作）。
   这一档的价值在于**等面积**是构造出来的，不是碰巧凑上的。

## 读这张表时必须记住的四条

* **面积要先看。** `union` 档让两路的背景池接近（实测 ep0 68.0% / 67.5%），
  但**不保证相等**（实测 ep1 57.8% / 67.5%）。比值只有在面积接近时才可比，
  脚本把两列面积印出来就是为了让你当场核这一条。
* **逐格对比的位置不匹配。** 两路相机之间没有单应（`CLAUDE.md` 记了四条独立证据
  证明它解不出来），所以「up 的第 (1,2) 格」和「front 的第 (1,2) 格」看的
  不是同一块桌面。逐格对比能控住**面积**，控不住**内容**。
* **只有 `*_noise_matched` 依赖 seed。** `bg_const` / `fg_const` / `bg_swap_ep` /
  `fg_swap_ep` / `full_*` 都是确定性的，不受 seed 影响 —— 结论应当主要落在它们上面。
* **均值会平掉尾部。** 同时看 `frac_gt_nat_step`（Δ 超过一个自然步长的样本占比）。
  对抓取策略来说"偶尔错得厉害"才是要命的，而均值看不见它。
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

UP = "observation.images.up"
FRONT = "observation.images.front"

# 这几档是增强的直接代理（确定性，不随 seed 变），结论主要看它们
PRIMARY = ["bg_const", "fg_const"]
OTHER = ["bg_swap_ep", "fg_swap_ep", "bg_noise_matched", "fg_noise_matched",
         "full_const", "full_noise", "cross_ep"]


def load(paths):
    out = {}
    for p in sorted(paths):
        d = json.loads(Path(p).read_text())
        if UP not in d.get("ablations", {}):
            continue
        out[(d["episode"], p)] = d
    return out


def trial_row(d, name):
    a = d["ablations"]
    if name not in a[UP] or name not in a[FRONT]:
        return None
    u, f = a[UP][name], a[FRONT][name]
    return {"up": u, "front": f, "ratio": u["mean"] / f["mean"] if f["mean"] else float("nan")}


def cell_contrast(npz_path):
    """逐格等面积：纯背景格 vs 纯前景格的 Δ 均值，两路各一份。

    阈值 0.10 / 0.90 是「这个格子基本只有一类像素」。中间地带故意不要 ——
    交界格在**两个相机里都是最敏感的**（实测），混进来会把两边的对比冲淡。
    """
    z = np.load(npz_path)
    res = {}
    for c, lab in ((UP, "up"), (FRONT, "front")):
        d = z["delta_" + c]                       # (rows, cols, B)
        fg = z["fg_" + c]                         # (B, H, W)
        R, C, _ = d.shape
        H, W = fg.shape[1:]
        hs = np.linspace(0, H, R + 1).astype(int)
        ws = np.linspace(0, W, C + 1).astype(int)
        frac = np.array([[fg[:, hs[r]:hs[r + 1], ws[cc]:ws[cc + 1]].mean()
                          for cc in range(C)] for r in range(R)]).ravel()
        dm = d.mean(-1).ravel()
        is_bg, is_fg = frac < 0.10, frac > 0.90
        res[lab] = {
            "bg_cells": int(is_bg.sum()), "fg_cells": int(is_fg.sum()),
            "bg_delta": float(dm[is_bg].mean()) if is_bg.any() else float("nan"),
            "fg_delta": float(dm[is_fg].mean()) if is_fg.any() else float("nan"),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="例如 outputs/saliency/t15_union_ep*.json")
    ap.add_argument("--out", default=None, help="可选：把汇总写一份 json")
    args = ap.parse_args()

    runs = load(glob.glob(args.glob))
    if not runs:
        raise SystemExit(f"{args.glob} 没匹配到结果（里面要有 up 和 front 两路）")
    eps = sorted({k[0] for k in runs})
    print(f"[cfg] {len(runs)} 个结果，episodes={eps}")

    # ---- 1. 整路消融 ----
    print("\n" + "=" * 96)
    print("整路消融：每个格子是「up Δ / front Δ」，括号里是该集两路的面积")
    print("=" * 96)
    print(f"{'试验':>18} " + " ".join(f"{'ep' + str(e):>22}" for e in eps) + f" {'比值中位':>10}")
    summary = {}
    for name in PRIMARY + OTHER:
        cells, ratios = [], []
        for e in eps:
            d = next(v for k, v in runs.items() if k[0] == e)
            r = trial_row(d, name)
            if r is None:
                cells.append(f"{'--':>22}")
                continue
            au, af = r["up"]["area_frac"], r["front"]["area_frac"]
            cells.append(f"{r['up']['mean']:6.2f}/{r['front']['mean']:5.2f} ({au:.0%}/{af:.0%})")
            ratios.append(r["ratio"])
        med = float(np.median(ratios)) if ratios else float("nan")
        summary[name] = {"ratios": ratios, "median": med}
        mark = " <<" if name in PRIMARY else ""
        print(f"{name:>18} " + " ".join(f"{c:>22}" for c in cells) + f" {med:9.2f}x{mark}")
    print("\n  << = 增强的直接代理（把背景换成纯色），结论主要看这两行")

    print("\n" + "-" * 96)
    print("背景敏感性 up/front 的比值（>1 = up 更依赖背景，即支持「front 没那么敏感」）")
    print("-" * 96)
    for name in PRIMARY + ["bg_swap_ep", "bg_noise_matched"]:
        s = summary.get(name)
        if not s or not s["ratios"]:
            continue
        rs = s["ratios"]
        print(f"  {name:>18}  中位 {s['median']:.2f}x   逐集 {['%.2f' % r for r in rs]}"
              f"   {'一致 >1' if all(r > 1 for r in rs) else '**有集数 <=1，方向不稳**'}")

    print("\n" + "-" * 96)
    print("对照：up 在**非背景**扰动上的比值。这几个数不小于上面的话，")
    print("说明 up 只是「整体更敏感」，而不是「更依赖背景」。")
    print("-" * 96)
    for name in ("full_const", "full_noise", "cross_ep"):
        s = summary.get(name)
        if s and s["ratios"]:
            print(f"  {name:>18}  中位 {s['median']:.2f}x   逐集 {['%.2f' % r for r in s['ratios']]}")

    # ---- 2. 逐格等面积 ----
    print("\n" + "=" * 96)
    print("逐格等面积（每格 = 画面 1/48）：纯背景格与纯前景格各自的 Δ")
    print("注意：面积可比，**位置不可比**（两路之间没有单应）")
    print("=" * 96)
    print(f"{'ep':>4} {'相机':>6} {'纯背景格':>9} {'背景格Δ':>9} {'纯前景格':>9} {'前景格Δ':>9} {'背景/前景':>10}")
    cell_sum = []
    for e in eps:
        key = next(k for k in runs if k[0] == e)
        npz = Path(key[1]).with_suffix(".npz")
        if not npz.exists():
            continue
        cc = cell_contrast(npz)
        for lab in ("up", "front"):
            r = cc[lab]
            rat = r["bg_delta"] / r["fg_delta"] if r["fg_delta"] else float("nan")
            print(f"{e:>4} {lab:>6} {r['bg_cells']:>9} {r['bg_delta']:>9.4f} "
                  f"{r['fg_cells']:>9} {r['fg_delta']:>9.4f} {rat:>10.3f}")
        u, f = cc["up"], cc["front"]
        cell_sum.append((e, u["bg_delta"] / f["bg_delta"], u["fg_delta"] / f["fg_delta"]))
    if cell_sum:
        print(f"\n  等面积下 up/front：纯背景格 "
              + " ".join(f"ep{e}={b:.2f}x" for e, b, _ in cell_sum)
              + f"（中位 {np.median([b for _, b, _ in cell_sum]):.2f}x）")
        print(f"  等面积下 up/front：纯前景格 "
              + " ".join(f"ep{e}={g:.2f}x" for e, _, g in cell_sum)
              + f"（中位 {np.median([g for _, _, g in cell_sum]):.2f}x）")
        print("  两个中位若一个 ~1、一个明显 >1，则 up 的优势在**前景**上，不在背景上。")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"episodes": eps, "trials": summary,
             "cell_equal_area": [{"episode": e, "bg_ratio": b, "fg_ratio": g}
                                 for e, b, g in cell_sum]},
            ensure_ascii=False, indent=2))
        print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
