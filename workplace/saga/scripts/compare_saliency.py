#!/usr/bin/env python
"""对比两个 checkpoint 的显著度：**它们到底差多少**。

    python workplace/saga/scripts/compare_saliency.py

起因：把基线和 SaGA 训练后的显著图并排一放，用户说「是一样的啊」。
先确认过两个 checkpoint 的权重确实是不同的（234 个张量里 153 个有差异，
backbone 最大权重差 0.023）—— 所以不是加载错了，是显著图真的几乎没变。
这个脚本把「几乎」量化出来，并出一张能看出差异的对比图。

三件事要说清楚：

1. **差异有多小**。逐像素算基线 vs SaGA 的 |Δs|，报 Pearson r 和平均/最大 |Δ|。
2. **Δ 图必须标量程**。差异小的时候，用 0–1 的固定标尺画出来是一片黑，
   看起来像「没有差异」；用自动标尺又会被误读成「差异很大」。
   所以 Δ 图的色标**显式标出实际量程**，读者一眼看到范围是 0.00–0.0x。
3. **别把这张图当成「SaGA 没效果」的证据**。它只说明**编码器的显著度**
   在训练前后变化很小，而 SaGA 的作用是拿这个显著度去做数据增强、
   改变**策略**的行为。策略有没有变好，只能靠真机评测回答，不是这张图。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(REPO / "workplace"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saga.fullgrad import FullGrad, EncoderOnly            # noqa: E402
from saga.saga import normalize_smaps                       # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CMAP = "inferno"


def collect(ckpt: str, ds, picks: list[str], cameras: list[str], dev, cache: Path):
    """对一个 checkpoint 算出各 (相机, 帧) 的显著图。

    结果按 checkpoint 目录名缓存成 npz —— FullGrad 是 CPU 上跑的单帧前向+反向，
    一行一帧几秒，调版式时不该反复重算。换 checkpoint 会自动落到另一个文件。
    """
    import hashlib
    p = Path(ckpt).resolve()
    # ckpt 形如 <run>/checkpoints/last/pretrained_model，取 <run> 的名字当可读前缀
    name = p.parents[2].name if len(p.parts) >= 4 else "ckpt"
    key = hashlib.md5(str(p).encode()).hexdigest()[:8]
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / f"heat_{name}_{key}.npz"
    if npz.exists():
        z = np.load(npz)
        print(f"[cache] {npz.name}")
        return {tuple(k.split("|")): z[k] for k in z.files}

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = str(dev)
    policy = ACTPolicy.from_pretrained(ckpt, config=cfg)
    policy.eval()

    spans = ds.meta.episodes.to_pandas()
    out = {}
    for cam in cameras:
        fg = FullGrad(policy.model.backbone, obs_key=None, Extractor=EncoderOnly)
        for pick in picks:
            ep, fr = (int(v) for v in pick.split(":"))
            lo = int(spans.iloc[ep]["dataset_from_index"])
            img = ds[lo + fr][f"observation.images.{cam}"].float()
            xn = (img.unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD
            heat = normalize_smaps(fg(xn.clone()))[0].detach().numpy()
            out[(cam, pick)] = heat
        fg.unregister_hooks()
        del fg
    del policy
    np.savez_compressed(npz, **{f"{c}|{k}": v for (c, k), v in out.items()})
    print(f"[cache] 写入 {npz.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--saga", default=str(REPO / "outputs/train/act_record_test10_saga/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset", default="record-test10")
    ap.add_argument("--root", default=str(REPO / "dataset" / "newdata1"))
    ap.add_argument("--out", default=str(REPO / "workplace/saga/figures/saliency"))
    # 显著图缓存放这儿，不是交付物 —— 但也不放 /tmp，重跑要复用
    ap.add_argument("--cache", default=str(REPO / "workplace/saga/figures/saliency/.cache"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--picks", nargs="+", default=["0:5", "0:100", "0:300", "3:150"])
    args = ap.parse_args()

    dev = torch.device(args.device)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.dataset, root=args.root)
    cameras = ["up", "front"]

    cache = Path(args.cache)
    print(f"[基线] {args.base}")
    base = collect(args.base, ds, args.picks, cameras, dev, cache)
    print(f"[SaGA] {args.saga}")
    saga = collect(args.saga, ds, args.picks, cameras, dev, cache)

    # ---------------- 差异到底有多小 ----------------
    print("\n" + "=" * 78)
    print("逐像素差异（s 已 min-max 归一化到 [0,1]，与 SaGA 实际消费的一致）")
    print("=" * 78)
    print(f"{'相机':<6}{'帧':<10}{'Pearson r':>11}{'平均|Δ|':>10}{'最大|Δ|':>10}"
          f"{'Δ占量程':>10}")
    stats = {}
    for cam in cameras:
        for pick in args.picks:
            a, b = base[(cam, pick)], saga[(cam, pick)]
            d = np.abs(a - b)
            r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            stats[(cam, pick)] = (a, b, d, r)
            print(f"{cam:<6}{pick:<10}{r:>11.4f}{d.mean():>10.4f}{d.max():>10.4f}"
                  f"{d.mean()*100:>9.1f}%")

    allr = [v[3] for v in stats.values()]
    alld = np.concatenate([v[2].ravel() for v in stats.values()])
    print("-" * 78)
    print(f"Pearson r: 均值 {np.mean(allr):.4f}  最小 {np.min(allr):.4f}")
    print(f"平均 |Δ| = {alld.mean():.4f}   最大 |Δ| = {alld.max():.4f}"
          f"   （s 的取值范围是 0–1）")

    # ---------------- 出图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    dmax = float(alld.max())
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        n = len(args.picks)
        fig, axes = plt.subplots(n, 4, figsize=(15.5, 3.5 * n + 0.7), squeeze=False)
        for i, pick in enumerate(args.picks):
            a, b, d, r = stats[(cam, pick)]
            ep = int(pick.split(":")[0])
            lo = int(ds.meta.episodes.to_pandas().iloc[ep]["dataset_from_index"])
            fr = int(pick.split(":")[1])
            img = np.clip(ds[lo + fr][f"observation.images.{cam}"].permute(1, 2, 0).numpy(), 0, 1)
            cols = [
                (img, "原图", None, None),
                (a, f"基线 $s$（集中度 {_conc(a):.2f}）", CMAP, (0, 1)),
                (b, f"SaGA 训练后 $s$（集中度 {_conc(b):.2f}）", CMAP, (0, 1)),
                (d, f"$|\\Delta s|$   r={r:.4f}", CMAP, (0, dmax)),
            ]
            for j, (im, title, cmap, lim) in enumerate(cols):
                ax = axes[i][j]
                if cmap is None:
                    ax.imshow(im)
                else:
                    ax.imshow(im, cmap=cmap, vmin=lim[0], vmax=lim[1])
                ax.set_title(title, fontsize=9.5)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"基线 vs SaGA 训练后 —— 相机 {cam}："
                     f"显著度几乎没变（平均 $|\\Delta s|$={alld.mean():.3f}）",
                     fontsize=12.5)
        # 色标横放在底部预留的条带里。不用 fig.colorbar(ax=...) —— 那会从子图里
        # 抠空间、把色标插到列与列之间压住图（第一版就是这样）。
        fig.tight_layout(rect=[0, 0.075, 1, 0.96])
        b_s1 = axes[-1][1].get_position()
        b_s2 = axes[-1][2].get_position()
        b_d = axes[-1][3].get_position()
        cax_s = fig.add_axes([b_s1.x0, 0.035, b_s2.x1 - b_s1.x0, 0.014])
        cb1 = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                           cax=cax_s, orientation="horizontal")
        cb1.set_label("$s$（量程 0–1）", labelpad=4)
        cax_d = fig.add_axes([b_d.x0, 0.035, b_d.width, 0.014])
        cb2 = fig.colorbar(ScalarMappable(norm=Normalize(0, dmax), cmap=CMAP),
                           cax=cax_d, orientation="horizontal")
        # 量程必须写进标签里 —— Δ 图用 0–1 画是一片黑（像「毫无差异」），
        # 用自动量程又不写量程的话会被误读成「差异很大」。写出来才诚实。
        cb2.set_label(f"$|\\Delta s|$（量程 0–{dmax:.2f}）", labelpad=4)
        p = outdir / f"saliency_diff_{cam}.png"
        fig.savefig(p, dpi=95)
        plt.close(fig)
        print(f"[save] {p}")

    print("\n注意：这张图只说明**编码器的显著度**在训练前后变化很小 —— "
          "不等于 SaGA 没效果。\n"
          "SaGA 的作用是拿显著度做数据增强、改变**策略**的行为；"
          "策略有没有变好，只能靠真机评测回答。")


def _conc(h: np.ndarray) -> float:
    """最强的 10% 像素占总质量的比例。均匀 = 0.10。"""
    f = h.ravel().astype(np.float64)
    thr = np.quantile(f, 0.9)
    tot = f.sum()
    return float(f[f >= thr].sum() / tot) if tot > 0 else float("nan")


if __name__ == "__main__":
    main()
