#!/usr/bin/env python
"""看显著图：ACT 的视觉编码器上，SaGA 实际会拿到什么样的空间权重。

这是**用真算法**（`workplace/saga/fullgrad.py`，逐行对参考实现）出图，
不是我自己按论文正文猜的那版。

## 它是什么、不是什么

SaGA 对**每个相机各做一次**（论文 Algorithm 1；参考实现里就是
`for obs_key in obs_meta["visual_modalities"]`，每个 key 一份 extractor + 一份缓冲区）。
所以两个相机都要出图 —— 这是**确认增强本身在正常工作**，
**不是**拿两个相机做对比。

看三件事：

1. 显著图是「有结构的热图」还是「一坨糊」。`layer4` 在 480×640 的输入上只有
   15×20 个格子，上采样 32 倍。糊成一片的话，`s = clip(g, λ)` 的逐像素裁剪、
   以及 `x·s + x_O·(1−s)` 的逐像素叠加，在这个分辨率上就名存实亡。
2. 两个相机的显著度**都**不是常数 —— 是常数说明 FullGrad 根本没在算。
3. 热点落在哪。这决定 SaGA 到底在**保护**什么、**替换**什么。

## 上一条旧结论要作废

先前那版探针用的是 `grad ⊙ activation`，而参考实现是 `grad ⊙ bias`
（`fullgrad.py:59`）。算法都不是同一个，所以旧图上
"up 热点在机械臂、front 热点在物块" 那个观察**不能直接沿用**，必须重看。

用法：

    python workplace/saga/saliency_probe.py --camera both --out workplace/saga/figures/saliency
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
WORKPLACE = HERE.parents[1]
REPO = HERE.parents[2]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saga.fullgrad import FullGrad, EncoderOnly            # noqa: E402
from saga.saga import normalize_smaps                       # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def concentration(heat: torch.Tensor) -> float:
    """显著度的集中度：最强的 10% 像素占了多少总质量。

    均匀分布 → 恰好 0.10。越集中越大。

    （前一版探针里那个 "最强 10% 像素占比" 是**恒等于 10%** 的恒等式 ——
    它是按 90 分位取的阈值，占比当然是 10%。那个指标没有任何信息量，已删。）
    """
    h = heat.flatten().float()
    q = torch.quantile(h, 0.9)
    tot = h.sum()
    if tot <= 0:
        return float("nan")
    return float(h[h >= q].sum() / tot)


def _norm_pct(t: torch.Tensor) -> np.ndarray:
    """仅供**看图**：按 1/99.5 分位裁离群点。SaGA 实际用的是 min-max，见下。"""
    a = t.abs().float().numpy()
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.5)
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


CMAP = "inferno"
"""显著度的色标。**不用 jet**：jet 是彩虹色标，用在「量级」数据上会制造出
数据里并不存在的亮暗边界，且对红绿色盲不友好。inferno 是感知均匀、亮度单调、
色盲安全的单色系热图色标，正好也是「热度」的语义。"""


def _overlay(img_hwc: np.ndarray, heat: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """把显著度叠到原图上：**不透明度随权重走** —— 低权重处保留原图，高权重处才上色。

    固定不透明度（早期写法）会把整张图压暗，低显著度区域也被涂了一层，
    反而看不清「策略在看哪里」。
    """
    import matplotlib.cm as cm
    rgb = cm.get_cmap(CMAP)(heat)[..., :3]
    a = (alpha * heat)[..., None]
    return np.clip(img_hwc * (1 - a) + rgb * a, 0, 1)


def load_dataset_and_policy(args, dev):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy

    # 用 config 覆写 device，避免先占一份 GPU 显存再搬走（GPU 可能被掩码任务占着）
    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = str(dev)
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg)
    policy.eval()
    ds = LeRobotDataset(args.dataset, root=args.root)
    return policy, ds


def probe_camera(args, policy, ds, camera, dev):
    backbone = policy.model.backbone
    fg = FullGrad(backbone, obs_key=None, Extractor=EncoderOnly)
    cam_key = f"observation.images.{camera}"
    spans = ds.meta.episodes.to_pandas()

    rows = []
    for pick in args.picks:
        ep, fr = (int(v) for v in pick.split(":"))
        lo = int(spans.iloc[ep]["dataset_from_index"])
        item = ds[lo + fr]
        img = item[cam_key].float()                      # (3,480,640) in [0,1]
        xn = (img.unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD

        t0 = time.time()
        # FullGrad 内部会就地 requires_grad_()，所以传 clone
        heat = fg(xn.clone())
        dt = time.time() - t0

        # SaGA 实际消费的是「逐样本 min-max 归一化到 [0,1]」的那张。
        # 注意 normalize_smaps 会把 (B,1,H,W) squeeze 成 (B,H,W)，所以索引是 [0]。
        heat01 = normalize_smaps(heat)[0].detach()

        raw = img.permute(1, 2, 0).numpy()
        rows.append({
            "tag": f"ep{ep}f{fr}",
            "img": raw,
            "heat01": heat01.numpy(),
            "heat_pct": _norm_pct(heat[0, 0].detach()),
            "conc": concentration(heat01),
            "seconds": dt,
        })
        r = rows[-1]
        print(f"  [{camera:5s} {r['tag']:9s}] 特征图 {tuple(heat.shape[-2:])} "
              f"{dt:5.2f}s  集中度 {r['conc']:.3f} "
              f"(均匀=0.100)  min={heat01.min():.3f} max={heat01.max():.3f}")

    fg.unregister_hooks()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset", default="record-test10")
    ap.add_argument("--root", default=str(REPO / "dataset" / "newdata1"))
    # 默认写进仓库：这些图是交付物的一部分，放 /tmp 重启就没了
    ap.add_argument("--out", default=str(REPO / "workplace/saga/figures/saliency"))
    ap.add_argument("--device", default="cpu", help="默认 cpu —— 两个 GPU 可能被掩码任务占着")
    ap.add_argument("--camera", default="both", choices=["up", "front", "both"])
    # 输出文件名后缀。要对多个 checkpoint 出图（基线 / SaGA 训练后）时用它区分，
    # 否则后一次会直接覆盖前一次。
    ap.add_argument("--tag", default="")
    # (episode, frame) 对。刻意挑不同阶段：ep0 开头/中段，ep3 抓取附近。
    ap.add_argument("--picks", nargs="+", default=["0:5", "0:100", "0:300", "3:150"],
                    help="episode:frame，frame 是**集内**帧号")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(f"[cfg] ckpt {args.ckpt}\n[cfg] device {dev}  camera {args.camera}")
    policy, ds = load_dataset_and_policy(args, dev)

    cameras = ["up", "front"] if args.camera == "both" else [args.camera]
    all_rows = {}
    for cam in cameras:
        print(f"\n[{cam}]")
        all_rows[cam] = probe_camera(args, policy, ds, cam, dev)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n_row = len(args.picks)
    n_col = len(cameras)
    fig, axes = plt.subplots(2 * n_row, n_col, figsize=(4.6 * n_col, 2.6 * 2 * n_row),
                             squeeze=False)
    for j, cam in enumerate(cameras):
        for i, r in enumerate(all_rows[cam]):
            axes[2 * i][j].imshow(np.clip(r["img"], 0, 1))
            axes[2 * i][j].set_title(f"{cam}  {r['tag']}  原图", fontsize=10)
            axes[2 * i + 1][j].imshow(_overlay(r["img"], r["heat01"]))
            axes[2 * i + 1][j].set_title(
                f"SaGA 权重（min-max 到 [0,1]，再 clip λ=0.8）集中度 {r['conc']:.2f}",
                fontsize=9)
            for a in (axes[2 * i][j], axes[2 * i + 1][j]):
                a.set_xticks([]); a.set_yticks([])
    # 色标图例：叠加图既用颜色又用不透明度编码 s，没有图例读者无法定量
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                 ax=list(axes[1::2, -1]), fraction=0.03, pad=0.02, shrink=0.55,
                 label="SaGA 权重 $s$")
    fig.suptitle("FullGrad on ACT backbone —— 真算法（grad⊙bias），480×640 输入", fontsize=13)
    # rect 给 suptitle 留出顶部空间，否则它会盖掉第一行的子图标题
    fig.tight_layout(rect=[0, 0, 0.98, 0.965])
    png = out / f"saliency_{args.camera}{args.tag}.png"
    fig.savefig(png, dpi=88)
    plt.close(fig)
    print(f"\n[save] {png}")

    # 只出热图，方便看结构。**上下两行是同一个 FullGrad 输出的两种归一化，
    # 不是两个不同的量**：上行 heat01 是 SaGA 真正消费的（逐样本 min-max），
    # 下行 heat_pct 额外取了绝对值、按 1/99.5 分位裁剪 —— SaGA 不消费它，
    # 纯粹为了让肉眼看到被离群亮点压扁的中间结构。别拿下行当结果展示。
    fig, axes = plt.subplots(n_col * 2, n_row, figsize=(4.2 * n_row, 4.0 * n_col + 0.7),
                             squeeze=False)
    for j, cam in enumerate(cameras):
        for i, r in enumerate(all_rows[cam]):
            axes[2 * j][i].imshow(r["heat01"], cmap=CMAP, vmin=0, vmax=1)
            axes[2 * j][i].set_title(f"{cam} {r['tag']}  (SaGA 用的图)", fontsize=9)
            axes[2 * j + 1][i].imshow(r["heat_pct"], cmap=CMAP, vmin=0, vmax=1)
            axes[2 * j + 1][i].set_title("分位裁剪+取绝对值（只为肉眼，SaGA 不用）",
                                         fontsize=9)
            for a in (axes[2 * j][i], axes[2 * j + 1][i]):
                a.set_xticks([]); a.set_yticks([])
    # 色标横放到底部预留的条带。不用 fig.colorbar(ax=...) —— 它会从子图里抠空间，
    # 把色标插到最后一列上盖住图（第一版就是这样）
    fig.tight_layout(rect=[0, 0.075, 1, 1.0])
    b_l = axes[-1][0].get_position()
    b_r = axes[-1][-1].get_position()
    cax = fig.add_axes([b_l.x0, 0.035, b_r.x1 - b_l.x0, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                      cax=cax, orientation="horizontal")
    cb.set_label("归一化显著度", labelpad=4)
    png2 = out / f"saliency_{args.camera}{args.tag}_raw.png"
    fig.savefig(png2, dpi=88)
    plt.close(fig)
    print(f"[save] {png2}")


if __name__ == "__main__":
    main()
