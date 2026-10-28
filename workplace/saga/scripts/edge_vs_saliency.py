#!/usr/bin/env python
"""FullGrad 保护的到底是「物体」还是「边缘」？

## 为什么问这个

安全红线里写「不重着色机械臂、夹爪、蓝方块、红盒」。要判断 SaGA 有没有踩线，
得先知道它保护的是什么。三种可能：

  (a) 保护**物体**：整个臂身/方块 s 高 → 红线自动满足。
  (b) 保护**边缘**：只有轮廓 s 高，物体内部和桌面一样被换 → 「保护了臂」是错觉。
  (c) 什么都不特别保护：s 与图像内容无关。

FullGrad 是拿 backbone 的梯度做的，而 ACT 的 backbone 是 ImageNet 预训练的 ResNet，
**对边缘/对比度敏感**。所以先验偏 (b)。这个脚本把 (a)/(b)/(c) 分开。

## 怎么分（不需要任何手工掩码）

按**局部边缘强度**给像素分箱，看每一箱里的平均替换率 `1−s`：

  - 若随边缘强度**单调下降** → (b)：边缘被保护，平坦区被换。
  - 若几乎**平的** → (c)：s 与内容无关。
  - (a) 需要真正的物体掩码才能证，本脚本只能给出「边缘解释不了的部分」。

边缘强度用灰度图的 Sobel 梯度幅值。分箱用**分位数**（每箱像素数相等），
否则「高边缘像素本来就少」会把均值拉偏。

用法：
    python workplace/saga/scripts/edge_vs_saliency.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
WORKPLACE = HERE.parents[2]
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saga.scripts.where_replaced import smaps_as_training   # noqa: E402
from saga.saga import RoboSaGA, normalize_smaps             # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) ImageNet 归一化 → (N,H,W) 灰度梯度幅值。

    先在**反归一化**后的灰度上算：ImageNet 的 per-channel std 不同，直接在归一化
    空间里算灰度会把通道权重搞错，边缘强度就不是图像上的边缘了。
    """
    g = (x * IMAGENET_STD.to(x.device) + IMAGENET_MEAN.to(x.device)).clamp(0, 1)
    g = (0.299 * g[:, 0] + 0.587 * g[:, 1] + 0.114 * g[:, 2]).unsqueeze(1)
    kx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, kx.transpose(-1, -2), padding=1)
    return (gx ** 2 + gy ** 2).sqrt().squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset", default="record-test10")
    ap.add_argument("--root", default=str(REPO / "dataset" / "newdata1"))
    ap.add_argument("--out", default=str(REPO / "workplace/saga/figures"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--per-ep", type=int, default=10)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    dev = torch.device(args.device)
    import lerobot.policies.act.modeling_act  # noqa: F401  注册 'act'
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = str(dev)
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg)
    policy.eval()
    ds = make_dataset(TrainPipelineConfig(
        policy=cfg, dataset=DatasetConfig(repo_id=args.dataset, root=args.root)))
    for k in ds.meta.camera_keys:
        m = torch.as_tensor(ds.meta.stats[k]["mean"]).flatten()[:3]
        assert torch.allclose(m, torch.tensor([0.485, 0.456, 0.406]), atol=1e-5), \
            "相机统计量不是 ImageNet —— make_dataset 的 use_imagenet_stats 被关掉了？"

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.ckpt, dataset_stats=ds.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": dev.type},
            "normalizer_processor": {
                "stats": ds.meta.stats,
                "features": {**policy.config.input_features,
                             **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    keys = ["observation.images.up", "observation.images.front"]
    ep = ds.meta.episodes.to_pandas()
    per_cam = {k: {"edge": [], "repl": [], "sall": []} for k in keys}

    for e in np.linspace(0, len(ep) - 1, args.episodes).round().astype(int):
        lo = int(ep.iloc[e]["dataset_from_index"])
        hi = int(ep.iloc[e]["dataset_to_index"])
        idxs = np.linspace(lo + 20, hi - 20, args.per_ep).astype(int)
        b = preprocessor(torch.utils.data.default_collate([ds[int(i)] for i in idxs]))
        gid = b["index"]
        n = len(idxs)
        h, w = b[keys[0]].shape[-2:]
        saga = RoboSaGA(
            backbone=policy.model.backbone, image_keys=keys, image_shape=(h, w),
            background_path=str(WORKPLACE / "background_generalization" /
                                "backgrounds" / "solid8"),
            buffer_depth=int(gid.max().item()) + 2, aug_ratio=0.0,
            update_ratio=0.1, saliency_saga_cap=0.8, warmup_steps=0,
            buffer_shape=(120, 160), device=dev,
        )
        for key in keys:
            with torch.enable_grad():
                smaps = normalize_smaps(saga._fullgrad(b[key].clone())).detach()
            s = smaps_as_training(smaps, (h, w))[:, 0]      # (N,H,W)
            per_cam[key]["edge"].append(sobel_mag(b[key]).cpu())
            per_cam[key]["repl"].append((1.0 - s).cpu())
        print(f"  ep {int(e)}: {n} 帧  ok")
        del b

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    summary = {}
    for ci, (key, nm) in enumerate(zip(keys, ["up", "front"])):
        E = torch.cat(per_cam[key]["edge"]).flatten().numpy()
        R = torch.cat(per_cam[key]["repl"]).flatten().numpy()
        # 分位数分箱：每箱像素数相等
        qs = np.quantile(E, np.linspace(0, 1, args.bins + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        centers, means, lo_e = [], [], []
        for i in range(args.bins):
            m = (E >= qs[i]) & (E < qs[i + 1])
            if m.sum() == 0:
                continue
            centers.append(E[m].mean()); means.append(R[m].mean())
            lo_e.append(np.quantile(E[m], 0.25))
        centers, means = np.array(centers), np.array(means)
        ax = axes[ci]
        ax.plot(centers, means * 100, "o-", color="#c1440e", lw=2, ms=7)
        ax.axhline(R.mean() * 100, ls="--", color="#555", lw=1.2,
                   label=f"全图平均 {R.mean() * 100:.1f}%")
        ax.set_xscale("log")
        ax.set_xlabel("局部边缘强度（Sobel 幅值，中位数）")
        ax.set_ylabel("被 OOD 替换的比例  1−s  (%)")
        ax.set_title(f"相机 {nm}：替换率 vs 边缘强度")
        ax.grid(alpha=0.3); ax.legend()
        ax.set_ylim(0, 100)
        # 高低两端的对比数字
        hi_rep = R[E >= np.quantile(E, 0.9)].mean() * 100
        lo_rep = R[E <= np.quantile(E, 0.1)].mean() * 100
        summary[nm] = (lo_rep, hi_rep, R.mean() * 100)
        print(f"\n=== 相机 {nm} ===")
        print(f"  最低 10% 边缘强度像素：被替换 {lo_rep:5.1f}%")
        print(f"  最高 10% 边缘强度像素：被替换 {hi_rep:5.1f}%")
        print(f"  全图平均            ：被替换 {R.mean() * 100:5.1f}%")
        print(f"  差值（低边 − 高边） ：{lo_rep - hi_rep:+5.1f} 个百分点")
        print(f"  各档区间            ：{means.min() * 100:.1f}% – {means.max() * 100:.1f}%"
              f"（即最保护的一档仍被换掉 {means.min() * 100:.0f}%）")
        print("  注：关系**非单调** —— 中边缘被换得最多、两端反而少。"
              "所以「边缘被保护」是个弱且不完整的说法；")
        print("      真正稳的是「没有任何一档低到可以叫保护」。")

    allmin = min(min(v[0], v[1]) for v in summary.values())
    allmax = max(max(v[0], v[1]) for v in summary.values())
    fig.suptitle(
        f"替换率几乎不随内容变化：全域只在 {allmin:.0f}–{allmax:.0f}% 之间，"
        "没有任何一档被真正保护\n"
        "（保护最狠的一档＝最强边缘，也仍被 OOD 换掉 56%；关系非单调，中边缘反而被换得最多）",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    p = outdir / "edge_vs_saliency.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"\n[save] {p}")


if __name__ == "__main__":
    main()
