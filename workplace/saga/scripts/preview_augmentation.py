#!/usr/bin/env python
"""把 SaGA 增强实际作用到真实数据上 —— 两个相机各出一组对比图。

**这是交付物的直接证据**：论文的方法被同时施加到 up 和 front 上，
每个相机各自一份显著度缓冲区、各自逐像素混合。

走的是与训练**完全同一条**路径：

    LeRobotDataset → default_collate → make_pre_post_processors 造出的 preprocessor
        → saga(batch, step)

所以图上看到的就是训练时策略真正吃进去的图，不是另写一遍的演示代码。

## 缓冲区是**预热过的** —— 这不是作弊

训练里 warmup 是前 10 个 epoch 完全不做增强，等增强开始时缓冲区里已经有真实
显著图了。本脚本为了让一张图就看到真实行为，先对要展示的这批帧算一遍 FullGrad
写进缓冲区，再调 `saga`。

不预热会看到什么：缓冲区初值全 1.0 → `s = clip(1, 0, λ)² = 0.64` 是**常数** →
图上是「fixed-weight random overlay」，不是 SaGA。那正是 `verify_saga.py`
第 8、9 组测的那个退化情形，所以这里必须预热才能展示对。

用法：

    python workplace/saga/scripts/preview_augmentation.py --out workplace/saga/figures/augmentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
SAGA = HERE.parents[1]
WORKPLACE = HERE.parents[2]
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saga.saga import RoboSaGA, normalize_smaps              # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm(xn: torch.Tensor) -> np.ndarray:
    """归一化空间 → 可显示的 [0,1] HWC。"""
    x = xn.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x.permute(0, 2, 3, 1).numpy(), 0, 1)


def build_batch(ds, ep: int, n: int, camera_keys: list[str],
                start: int = 0, stride: int = 1):
    """从一集里取 n 帧，用**与训练相同的 default_collate** 组成 batch。

    `stride > 1` 时跨时段取样。SaGA 对帧的**连续性没有任何要求** ——
    缓冲区按全局帧 id 索引，α/β 选的是 batch 内的位置。所以跨时段取样合法，
    而且能看出增强在整个操作过程中的行为（连续 16 帧几乎长得一样，看不出什么）。
    """
    spans = ds.meta.episodes.to_pandas()
    lo = int(spans.iloc[ep]["dataset_from_index"])
    items = [ds[lo + start + i * stride] for i in range(n)]
    batch = torch.utils.data.default_collate(items)
    print(f"[batch] ep{ep} 起点 {start} 步长 {stride} 取 {n} 帧，全局 id "
          f"{int(batch['index'][0])}..{int(batch['index'][-1])}，"
          f"相机 {[k.split('.')[-1] for k in camera_keys]}")
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset", default="record-test10")
    ap.add_argument("--root", default=str(REPO / "dataset" / "newdata1"))
    # 默认写进仓库：这些图是交付物的一部分，放 /tmp 重启就没了
    ap.add_argument("--out", default=str(REPO / "workplace/saga/figures/augmentation"))
    ap.add_argument("--device", default="cpu",
                    help="默认 cpu —— 两个 GPU 常被掩码任务占着，不抢")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="从本集第几帧开始取")
    ap.add_argument("--stride", type=int, default=1,
                    help="取样步长。>1 则跨时段取样（SaGA 不要求帧连续）")
    ap.add_argument("--n", type=int, default=16,
                    help="batch 大小。必须 ≥ 1/β=10，否则 int(n×β)=0，SaGA 退化")
    ap.add_argument("--show", type=int, default=4, help="图上展示前几帧")
    ap.add_argument("--background", default=None,
                    help="默认用 ../background_generalization/backgrounds/solid8")
    args = ap.parse_args()

    dev = torch.device(args.device)
    bg_path = args.background or str(
        WORKPLACE / "background_generalization" / "backgrounds" / "solid8")

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = str(dev)          # 先覆写，免得先在 GPU 上占一份显存
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg)
    policy.eval()

    # ⚠️ 必须走 `make_dataset`，不能直接 `LeRobotDataset(...)`。
    # `make_dataset` 在 `use_imagenet_stats=True`（默认）下会把 `meta.stats` 里
    # **相机**的 mean/std **换成 ImageNet 常数**（`datasets/factory.py:126`）。
    # 直接构造 LeRobotDataset 拿到的是数据集自己的统计量（实测 up: mean 0.742 /
    # std 0.0137），归一化空间跟训练时**不是同一个** —— 显存里看不出，图会花。
    ds = make_dataset(TrainPipelineConfig(
        policy=cfg, dataset=DatasetConfig(repo_id=args.dataset, root=args.root)))

    # 硬断言：相机统计量必须是 ImageNet。这不是形式，是唯一能挡住"归一化空间不一致"
    # 这类静默错误的检查 —— 空间错了不会报错，只会让图变花、让 SaGA 的混合变成垃圾。
    for k in ds.meta.camera_keys:
        m = torch.as_tensor(ds.meta.stats[k]["mean"]).flatten()[:3]
        s = torch.as_tensor(ds.meta.stats[k]["std"]).flatten()[:3]
        assert torch.allclose(m, torch.tensor([0.485, 0.456, 0.406]), atol=1e-5), \
            f"{k} mean={m.tolist()} 不是 ImageNet —— 归一化空间与训练不一致"
        assert torch.allclose(s, torch.tensor([0.229, 0.224, 0.225]), atol=1e-5), \
            f"{k} std={s.tolist()} 不是 ImageNet —— 归一化空间与训练不一致"
    print("[cfg] 相机统计量 = ImageNet（与训练一致），SaGA 的 OOD 池用同一套常数")

    # --- 与 lerobot_train.py 同样的构造方式，保证是同一条路径 ---
    camera_keys = [k for k in policy.config.image_features]
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.ckpt,
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": dev.type},
            "normalizer_processor": {
                "stats": ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    batch = build_batch(ds, args.episode, args.n, camera_keys,
                        args.start, args.stride)
    batch = preprocessor(batch)
    assert "index" in batch, "preprocessor 把 index 丢了 —— 缓冲区索引不到帧"
    gid = batch["index"]

    h, w = batch[camera_keys[0]].shape[-2:]
    saga = RoboSaGA(
        backbone=policy.model.backbone,
        image_keys=camera_keys,
        image_shape=(h, w),
        background_path=bg_path,
        buffer_depth=int(gid.max().item()) + 2,
        aug_ratio=0.5, update_ratio=0.1, saliency_saga_cap=0.8,
        warmup_steps=0,                 # 本脚本要看的正是「warmup 之后」的行为
        buffer_shape=(120, 160),
        device=dev,
    )

    # ---- 预热：把要展示的这批帧的显著度写进缓冲区（= 训练里 warmup 之后的状态）----
    print("\n[预热] 对这 %d 帧算 FullGrad 并写入缓冲区……" % len(gid))
    k = int(args.n * 0.5)               # 只有前 α 比例会被增强
    for key in camera_keys:
        imgs = batch[key]
        maps = normalize_smaps(saga._fullgrad(imgs[:k].clone())).detach()
        saga.buffers[key].set(maps, gid[:k])
    print("[预热] 完成")

    # ---- 真正的调用：与训练循环里那一行完全相同 ----
    out = saga(batch, step=10_000_000)   # step 大，越过 warmup

    # ---- 统计 ----
    print("\n[结果] 每个相机各自一份显著度、各自混合：")
    stats = {}
    for key in camera_keys:
        cam = key.split(".")[-1]
        k = int(args.n * saga.aug_ratio)
        a, b = out[key][:k], batch[key][:k]
        delta = (a - b).abs()
        s = saga.buffers[key].get(gid[:k])
        s = torch.clip(s, 0, saga.saliency_saga_cap) ** 2
        untouched = torch.equal(out[key][k:], batch[key][k:])
        stats[cam] = dict(s_mean=s.mean().item(), s_min=s.min().item(),
                          s_max=s.max().item(),
                          changed_frac=(delta.amax(dim=1) > 1 / 255).float().mean().item(),
                          mean_delta=delta.mean().item(), head=k, untouched=untouched)
        st = stats[cam]
        print(f"  {cam:5s} 被增强 {k}/{args.n} 帧（前 α 比例），后 {args.n - k} 帧逐位不变={st['untouched']}")
        print(f"        权重 s ∈ [{st['s_min']:.3f}, {st['s_max']:.3f}]  均值 {st['s_mean']:.3f}"
              f"（常数 0.64 = 退化，逐像素变化 = SaGA 生效）")
        print(f"        被改动的像素占比 {st['changed_frac']*100:.1f}%  平均 |Δ| {st['mean_delta']:.4f}")

    # ---- 红线：非图像键必须没动 ----
    for kk in ("observation.state", "action", "index"):
        if kk in batch:
            same = torch.equal(out[kk], batch[kk])
            print(f"  [红线] {kk} 逐位不变: {same}")
            assert same, f"{kk} 被改了 —— 违反红线"

    # ---- 出图 ----
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 色标用 inferno，不用 jet。jet 是彩虹色标，用在「量级」（这里是 0..λ²=0.64
    # 的连续权重）上会凭空造出亮暗分界，读者容易把色带当成阈值；也对红绿色盲不友好。
    # inferno 感知均匀、亮度单调、色盲安全，且就是「热度」语义。
    CMAP = "inferno"

    ns = min(args.show, args.n // 2)
    # 展示的帧**在被增强的前 α 帧里均匀取**，不要取连续几帧 ——
    # 连续帧长得几乎一样，四行看下来等于一行。均匀取才能看出跨时段的稳定行为。
    k = int(args.n * saga.aug_ratio)
    idxs = np.linspace(0, k - 1, ns).round().astype(int) if ns > 1 else np.array([0])
    print(f"\n[出图] 展示前 α={k} 帧里的第 {idxs.tolist()} 帧")
    for key in camera_keys:
        cam = key.split(".")[-1]
        s_all = torch.clip(saga.buffers[key].get(gid[:k]), 0, saga.saliency_saga_cap) ** 2
        fig, axes = plt.subplots(ns, 4, figsize=(17, 3.6 * ns), squeeze=False)
        for i, fi in enumerate(idxs):
            rows = [
                # 真实帧号 = 起点 + batch 内位置 × 步长；别写 fi，fi 只是取样序号
                (denorm(batch[key][fi:fi + 1])[0],
                 f"原图（集内第 {args.start + fi * args.stride} 帧）"),
                (denorm(out[key][fi:fi + 1])[0], "SaGA 增强后 ← 策略实际吃到的"),
                (s_all[fi, 0].cpu().numpy(), "权重 s（clip λ=0.8 后平方）"),
                (1 - s_all[fi, 0].cpu().numpy(), "被 OOD 背景替换的比例 1−s"),
            ]
            for j, (im, title) in enumerate(rows):
                ax = axes[i][j]
                if j < 2:
                    ax.imshow(im)
                else:
                    ax.imshow(im, cmap=CMAP, vmin=0, vmax=1)
                ax.set_title(f"{cam}  {title}" if j == 0 else title, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        # 一根共用色标就够：s 与 1−s 同为逐像素权重，画在同一 0..1 标尺上。
        # （早先给两列各放一根，标签在窄缝里叠成一团 —— 别那么干。）
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                          ax=list(axes[:, -1]), fraction=0.022, pad=0.02, shrink=0.7)
        # labelpad 只能建完之后设（不是 fig.colorbar 的参数）；
        # 不设的话标签会贴着刻度数字，像这次一样糊在一起
        cb.set_label("逐像素权重", labelpad=10)
        fig.suptitle(f"RoboSaGA 增强 —— 相机 {cam}（论文 Algorithm 1，OOD 池 = solid8 纯色）",
                     fontsize=13)
        # rect 给 suptitle 留顶部空间，否则它会盖掉第一行的子图标题
        fig.tight_layout(rect=[0, 0, 0.965, 0.955])
        png = outdir / f"augmentation_{cam}.png"
        fig.savefig(png, dpi=80)
        plt.close(fig)
        print(f"\n[save] {png}")


if __name__ == "__main__":
    main()
