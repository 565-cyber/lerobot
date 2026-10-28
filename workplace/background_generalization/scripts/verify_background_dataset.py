#!/usr/bin/env python
"""校验 RoboSaGA 式背景替换数据集。退出码非零 = 有硬失败。

    python scripts/verify_background_dataset.py \
        --report-dir /tmp/bg_reports --base-root <base> --stride 10

分两段，分法和 Stage 2A/2B 的校验器一样，理由也一样：`state`/`action` 的
逐位一致用 parquet 就能**穷尽**地查，比重解码三遍视频既快又强；像素部分抽样。

    phase 1  parquet 层穷尽：`state` / `action` / `timestamp` / `frame_index`
             逐位一致，**每一帧**。这是不许放松的那一条。
    phase 2  抽样像素，逐 (base episode, 相机)。

## 哪几条在内存里断，哪几条只能带容差

这是本脚本里最容易做错的地方。生成集是 AV1 编码的，**解码回来的帧不是逐位的**
—— 在锐边上单像素差能到 ~40/255（`verify_distractor_dataset.py` 记了这个教训，
Stage 2A 也踩过一次，当时报出 43/255 的假"失败"）。所以：

* **内存里**（拿基线的张量直接重算一遍合成）：断**逐位相等**。合成器是不是真的
  没动前景，在这里是精确的，没有借口放水。
* **磁盘上**（读生成集解码后的帧）：只断**中位差**，并且**不 gate max**。
  max 会被编码噪声顶穿，中位差才反映"这件事到底做没做"。

## 每一条断言在什么输入下会失败

按 `CLAUDE.md` 的要求逐条写清楚，答不上来的断言就是恒真断言，不该留：

| 断言 | 什么输入下会失败 |
|---|---|
| a1 合成器不动前景 | 掩码极性反了 / `_mask2d` 把掩码转置或缩错了尺寸 |
| a2 掩码是 0/1 且对齐 | 落盘的掩码是 0/255 / 帧数对不上 / 是别的 episode 的 |
| b1 前景处没变（磁盘） | 同上，外加"掩码整体错位一帧" |
| b2 背景处变成了池子里那张 | 配错了背景图 / 背景没被 resize / 索引用了 `gi` 而不是 `base_episode` |
| b3 背景处确实变了 | 掩码几乎全是 1 / 背景恰好等于原图 / 生成器静默跳过了这一集 |
| b4 没点名的相机没变 | `--cameras` 的过滤写错，把不该动的相机也换了 |
| b5 被改像素 ∩ SAM2 机械臂 = ∅ | 前景掩码用了旧版本 / 错的 episode / 极性反（**跨工件**） |
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from object_generalization.scripts import _cache_io  # noqa: E402
from object_generalization.scripts.verify_generated_dataset import _read_table, _stack  # noqa: E402
from background_generalization.masks.robot_mask import _to_hwc_uint8  # noqa: E402
from background_generalization.transforms.background_replacement import (  # noqa: E402
    apply_background_replacement,
)

# 磁盘上的阈值。AV1 在锐边上的单像素差能到 ~40/255，所以这两个数不能往紧里卡：
# 15/255 是"中位差这么大" —— 前景被替换掉的话中位差会是几十到上百，和它差一个量级。
TOL_FG_MEDIAN = 15.0      # 前景处 |gen - base| 的中位上限
TOL_BG_MEDIAN = 25.0      # 背景处 |gen - bg| 的中位上限（还叠了 resize 的差异）
MIN_BG_CHANGED = 0.50     # 背景处至少这么大比例的像素要变了
# 标了"不增强"的那些档：gen 与 base 的中位差**只该有码流噪声**。AV1 在硬边上的
# 单像素差能到 ~40，但中位数量级是 2/255；这里给到 8 是留足余量，同时任何真增强
# （背景被换掉）都会把整片背景推到远超 8。所以它能抓住"标了不增强其实改了"。
TOL_PASS_DRIFT = 8.0

# 「这个像素变了没有」的门槛，**不能是 0 或者 1e-6**。
# 生成集是 AV1 编码的，解码回来**每个**像素都变了几个灰阶（实测中位 ~2，
# 锐边上到 ~40）—— 用 `> 1e-6` 判"变了"会让每一条相关断言恒真/恒假，
# 看着像证据其实是在测编码器。8 落在"编码噪声（≤5）"和"真的换掉了（几十到上百）"
# 之间的空带里。这条踩过一次：b3/b5 一开始都用 1e-6，结果报 100% 变化、
# 14 万 px「压到机械臂」，全是假的。
CHANGE_PX = 8.0

# b5 的内容判据：「生成后的像素贴着背景色」且「离原图很远」= 真泄漏。
# 20 比 AV1 噪声底（~2-5）宽，比"真的换成了背景"（几十到上百）紧得多。
LEAK_NEAR_BG = 20.0
LEAK_FAR_BASE = 40.0
# 低于这个面积的泄漏连通块按编码噪声处理，单独计入 speckle_px，不算失败。
LEAK_MIN_PX = 20


def phase1(base_root, gen_root, episodes, report, limit_of=None):
    cols = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    b = _read_table(base_root, cols).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    g = _read_table(gen_root, cols).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    fails = []

    pos = _cache_io.gen_positions(episodes)
    gen2base = {v: k for k, v in pos.items()}
    unexpected = sorted(set(int(v) for v in g["episode_index"].unique()) - set(gen2base))
    if unexpected:
        fails.append(f"生成集里有没预期的 episode_index {unexpected}")

    b = b[b["episode_index"].isin(episodes)].reset_index(drop=True)
    # 冒烟时 `--limit N` 只写了每集前 N 帧，基线那边也要截到 N，否则行数永远对不上
    # ——那是把"我用的是子集"报成"数据错了"。
    if limit_of:
        keep = np.ones(len(b), bool)
        for ep, n in limit_of.items():
            keep &= ~((b["episode_index"].to_numpy() == ep) & (b["frame_index"].to_numpy() >= n))
        b = b[keep].reset_index(drop=True)

    g = g[g["episode_index"].isin(gen2base)].reset_index(drop=True)
    g = g.assign(episode_index=g["episode_index"].map(gen2base))
    g = g.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    if len(b) != len(g):
        bl = b.groupby("episode_index").size().to_dict()
        gl = g.groupby("episode_index").size().to_dict()
        diff = {ep: (bl.get(ep), gl.get(ep)) for ep in sorted(episodes) if bl.get(ep) != gl.get(ep)}
        fails.append(f"行数 {len(b)}(base) != {len(g)}(gen)；逐集 (base, gen) = {diff}")
        return fails, {"frames": int(len(b)), "episodes": int(len(episodes))}

    for key in ("episode_index", "frame_index"):
        if not np.array_equal(b[key].to_numpy(), g[key].to_numpy()):
            bad = np.flatnonzero(b[key].to_numpy() != g[key].to_numpy())[:5]
            fails.append(f"{key} 在第 {bad.tolist()} 行错位")

    for col in ("observation.state", "action", "timestamp"):
        bb, gg = _stack(b[col]), _stack(g[col])
        neq = np.flatnonzero((bb != gg).any(axis=1))
        if neq.size:
            r = int(neq[0])
            fails.append(f"{col}: {neq.size} 行不一致，首个在 "
                         f"(episode={int(b['episode_index'][r])}, frame={int(b['frame_index'][r])})")

    return fails, {"frames": int(len(b)), "episodes": int(len(episodes)),
                   "state_action_timestamp_identical": not fails}


def _bg_png(path: Path, h: int, w: int) -> np.ndarray:
    """独立地把背景 PNG 读进来 —— **不**用 `_BackgroundStore`。

    故意走另一条实现（cv2 vs torchvision、INTER_LINEAR vs bilinear）：复用
    `_BackgroundStore` 的话，"背景配对了没有"就退化成"它等于它自己"。
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不了背景图 {path}")
    img = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--base-root", required=True)
    ap.add_argument("--base-repo", default="record-test10")
    ap.add_argument("--gen-repo", default=None)
    ap.add_argument("--mask-root", default=str(BG / "outputs" / "foreground"))
    ap.add_argument("--robot-root", default=str(BG / "outputs" / "robot_mask"))
    ap.add_argument("--background-root", default=None, help="默认读 generate.json 里记的")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--full-frames", action="store_true")
    args = ap.parse_args()

    gen_rep = Path(args.report_dir) / "generate.json"
    if not gen_rep.exists():
        raise SystemExit(f"找不到 {gen_rep}（生成时 --report-dir 指到别处了？）")
    meta = json.loads(gen_rep.read_text())
    garg = meta["args"]
    gen_root = garg["out"]
    args.gen_repo = args.gen_repo or garg["repo_id"]
    args.background_root = args.background_root or garg["background_root"]
    cams = [{"up": "observation.images.up", "front": "observation.images.front"}[c]
            for c in garg["cameras"]]
    episodes = [r["base_episode"] for r in meta["episodes"]]
    # 生成集里的 episode 编号是**按写入顺序从 0 重排**的，不等于 base episode
    # （`_cache_io.gen_positions` 的 docstring 记了这件事）。phase 1 用了那个映射，
    # phase 2 漏了：它拿 base 号去 `LeRobotDataset(gen_repo, episodes=[ep])`，
    # 子集生成时那一集的本地数据不存在 → 走进"从 hub 拉"的分支 → **挂在
    # huggingface 连接超时上**（不是报错！`connect timeout=None`，实测能挂几分钟
    # 到无限）。全量生成时两者恰好相等，所以一直没暴露。
    # 这里直接用 generate.json 记下的真实编号，顺带不依赖"请求列表是升序"这个假设。
    gen_of = {r["base_episode"]: r["gen_episode"] for r in meta["episodes"]}
    bg_of = {r["base_episode"]: r["background"] for r in meta["episodes"]}
    limit_of = ({r["base_episode"]: r["frames"] for r in meta["episodes"]}
                if garg.get("limit") else None)

    report = {"args": vars(args), "generate_args": garg}
    print(f"=== phase 1：parquet 层穷尽一致（{len(episodes)} 集"
          + (f"，截到每集前 {garg['limit']} 帧" if limit_of else "") + "）===")
    fails1, s1 = phase1(args.base_root, gen_root, episodes, report, limit_of)
    report["phase1"] = s1
    for f in fails1:
        print(f"  FAIL  {f}")
    if not fails1:
        print(f"  PASS  {s1['frames']} 帧 state/action/timestamp 逐位一致")

    # 生成时就没增强的那些档（`--no-mask-policy passthrough`）。它们**不能**走下面
    # 那套检查：没有掩码（`np.load` 会直接抛）、背景本来就不该是池子里那张、
    # `背景只变了 N%` 也必然是 0%。所以逐档排除，但**单列出来报**，不许静默放过 ——
    # 静默的 pass-through 正是这条路径最危险的形态（占着"已增强"的编号）。
    pass_set = {(p["episode"], p["camera"]) for p in meta.get("passthrough", [])}
    if pass_set:
        print(f"\n[豁免] {len(pass_set)} 档在生成时就没增强（generate.json 里的 passthrough）："
              + ", ".join(f"ep{e}{c[0]}" for e, c in sorted(pass_set)))
        print("       这些档的检查换成「gen 应当 ≈ base（只有编码噪声）」，见下表"
              "「不增强漂移」那一列")

    print(f"\n=== phase 2：抽样像素（stride={args.stride}）===")
    fails2, per_ep = [], {}
    rows = []
    for ep in episodes:
        base = LeRobotDataset(args.base_repo, root=args.base_root, episodes=[ep])
        gen = LeRobotDataset(args.gen_repo, root=gen_root, episodes=[gen_of[ep]])
        T = min(len(base), len(gen))
        idx = range(T) if args.full_frames else range(0, T, args.stride)
        bgp = Path(args.background_root) / bg_of[ep] if args.background_root else None
        if bgp is not None and not bgp.exists():
            # 没给 --background-root 时 generate.json 里存的是文件名，路径拼不上
            bgp = None

        acc = {c: {"fg": [], "bg": [], "bgchg": [], "drift": []} for c in cams}
        arm_hit = {c: 0 for c in cams}
        speckle = {c: 0 for c in cams}
        drift_pass = {}                # 不增强档：gen 与 base 的漂移（应当只有编码噪声）
        for fr in idx:
            bi, gi_ = base[fr], gen[fr]
            for c in cams:
                bimg = _to_hwc_uint8(bi[c]).astype(np.float32)
                gimg = _to_hwc_uint8(gi_[c]).astype(np.float32)
                if (ep, c.split(".")[-1]) in pass_set:
                    # 不增强档：不查掩码（没有）、不查背景（本来就该是原图），
                    # 只查一件事 —— 它**真的没被改**。码流噪声之外还有差就是标错了。
                    drift_pass.setdefault(c, []).append(
                        float(np.median(np.abs(gimg - bimg).max(2))))
                    continue
                h, w = bimg.shape[:2]
                m = np.load(Path(args.mask_root) / c / f"episode_{ep:06d}.npy",
                            mmap_mode="r")[fr]
                if m.dtype != np.uint8 or m.max() > 1:
                    fails2.append(f"ep{ep} {c} 帧{fr}: 掩码不是 0/1（max={m.max()}）")
                    m = (m > 0)
                m = m.astype(bool)
                if m.shape != (h, w):
                    fails2.append(f"ep{ep} {c} 帧{fr}: 掩码形状 {m.shape} != {(h, w)}")
                    continue

                # a1：内存里精确 —— 合成器不动前景
                if bgp is not None:
                    bg = _bg_png(bgp, h, w)
                    from torch import from_numpy
                    bimg_t = from_numpy(bimg.transpose(2, 0, 1) / 255.0)
                    res = apply_background_replacement(
                        bimg_t, from_numpy(np.broadcast_to(m[None], (1, h, w)).copy()),
                        from_numpy(bg.transpose(2, 0, 1) / 255.0))
                    res_u8 = _to_hwc_uint8(res).astype(np.float32)
                    if not np.array_equal(res_u8[m], bimg[m]):
                        d = np.abs(res_u8[m] - bimg[m]).max()
                        fails2.append(f"ep{ep} {c} 帧{fr}: 合成器**改了前景**（最大差 {d:.0f}）")

                d = np.abs(gimg - bimg).max(2)
                if m.any():
                    acc[c]["fg"].append(float(np.median(d[m])))
                if (~m).any():
                    acc[c]["bgchg"].append(float((d[~m] > CHANGE_PX).mean()))
                if bgp is not None and (~m).any():
                    acc[c]["bg"].append(float(np.median(np.abs(gimg - bg).max(2)[~m])))
                acc[c]["drift"].append(float(np.median(d)))

                # b5：跨工件 —— 机械臂像素不能被换成背景。
                #
                # 这里**不能**用「arm 且 |gen-base|>8」来判：机械臂内部有高对比边
                # （黑线缆压在白臂上、臂压在深色背景上），AV1 在那些边上单像素差
                # 能到 ~40，于是每帧都会报几百 px 的假泄漏。踩过一次。
                #
                # 改成按**内容**判：这个像素是不是"更像背景、而且明显不像原图"。
                # 真泄漏必然满足（原来几十上百的差、现在贴着背景色）；
                # 编码噪声不满足（它仍贴着原图，只是抖了几个灰阶）。
                rp = Path(args.robot_root) / c / f"episode_{ep:06d}.npy"
                if rp.exists() and bgp is not None:
                    arm = np.load(rp, mmap_mode="r")[fr] > 0
                    near_bg = np.abs(gimg - bg).max(2) < LEAK_NEAR_BG
                    far_base = np.abs(gimg - bimg).max(2) > LEAK_FAR_BASE
                    leak = near_bg & far_base & arm
                    if leak.any():
                        # 真泄漏是**成片**的（机械臂整块被背景盖住）。孤立单像素
                        # 是编码器把某个暗像元推到了恰好接近背景色 —— 它满足内容
                        # 判据纯属巧合。所以按连通块筛，去掉的单独记进 speckle_px
                        # 而不是静默丢掉（和 compositor 的 speckle 同一套处理）。
                        from scipy import ndimage
                        lab, n = ndimage.label(leak)
                        if n:
                            sizes = ndimage.sum(leak, lab, range(1, n + 1))
                            big = [i + 1 for i in range(n) if sizes[i] >= LEAK_MIN_PX]
                            speckle[c] += int(sum(sizes[i - 1] for i in range(1, n + 1)
                                                  if sizes[i - 1] < LEAK_MIN_PX))
                            arm_hit[c] += int(sum(sizes[i - 1] for i in big))

        for c in cams:
            if (ep, c.split(".")[-1]) in pass_set:
                # 不增强档：只报漂移。判据是**它真的没被增强**（码流噪声量级），
                # 而不是"背景变了多少"。
                dp = float(np.median(drift_pass.get(c, [float("nan")])))
                per_ep[f"ep{ep}_{c.split('.')[-1]}"] = {"passthrough": True, "drift": dp}
                rows.append((ep, c.split(".")[-1], None, None, None, None, dp))
                if dp == dp and dp > TOL_PASS_DRIFT:
                    fails2.append(
                        f"ep{ep} {c.split('.')[-1]}: 报告里标着「不增强」，但 gen 与 base "
                        f"的中位差 {dp:.1f}/255 > {TOL_PASS_DRIFT} —— 它其实被改了？")
                continue
            a = acc[c]
            tag = c.split(".")[-1]
            fg_med = float(np.median(a["fg"])) if a["fg"] else float("nan")
            bg_med = float(np.median(a["bg"])) if a["bg"] else float("nan")
            chg = float(np.median(a["bgchg"])) if a["bgchg"] else 0.0
            per_ep[f"ep{ep}_{tag}"] = {"fg_median": fg_med, "bg_median": bg_med,
                                       "bg_changed": chg, "arm_hit_px": arm_hit[c],
                                       "speckle_px": speckle[c], "n": len(a["fg"])}
            if fg_med > TOL_FG_MEDIAN:
                fails2.append(f"ep{ep} {tag}: 前景处中位差 {fg_med:.1f}/255 > {TOL_FG_MEDIAN} "
                              f"—— 前景被改了？")
            if bgp is not None and bg_med == bg_med and bg_med > TOL_BG_MEDIAN:
                fails2.append(f"ep{ep} {tag}: 背景处与池子里那张图的中位差 {bg_med:.1f}/255 "
                              f"> {TOL_BG_MEDIAN} —— 配错背景了？")
            if chg < MIN_BG_CHANGED:
                fails2.append(f"ep{ep} {tag}: 背景处只有 {chg:.1%} 的像素变了，"
                              f"低于 {MIN_BG_CHANGED:.0%} —— 这一集其实没增强？")
            if arm_hit[c]:
                fails2.append(f"ep{ep} {tag}: {arm_hit[c]} px 的机械臂像素被改了 "
                              f"（跨工件比对 SAM2 原始掩码）")
            rows.append((ep, tag, fg_med, bg_med, chg, arm_hit[c], None))

    print(f"\n  {'ep':>4} {'cam':>6} {'前景中位差':>10} {'背景对不上':>10} {'背景变了':>9} "
          f"{'压到机械臂':>10} {'不增强漂移':>10}")
    for ep, tag, fg_med, bg_med, chg, hit, dp in rows:
        def f2(v):
            return "     —" if v is None else f"{v:>10.1f}"
        def p2(v):
            return "        —" if v is None else f"{v:>8.1%}"
        def i2(v):
            return "         —" if v is None else f"{v:>10d}"
        print(f"  {ep:>4} {tag:>6} {f2(fg_med)} {f2(bg_med)} {p2(chg)} {i2(hit)} {f2(dp)}")
    for f in fails2[:20]:
        print(f"  FAIL  {f}")
    if len(fails2) > 20:
        print(f"  ... 另有 {len(fails2) - 20} 条")

    report["phase2"] = per_ep
    report["fails"] = fails1 + fails2
    out = Path(args.report_dir) / "verify.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    n_fail = len(fails1) + len(fails2)
    # 豁免档数要跟着 RESULT 一起印：不然"PASS"看起来和"全都验过了"一样，
    # 而实际上有几档根本没走那套检查。
    n_pass = len(pass_set)
    print(f"\nRESULT: {'FAIL' if n_fail else 'PASS'}  ({n_fail} 条失败"
          + (f"，{n_pass} 档豁免未验（生成时就没增强）" if n_pass else "") + ")")
    print(f"[save] {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
