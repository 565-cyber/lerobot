#!/usr/bin/env python
"""OOD 背景**替换落在画面哪里** —— 零几何、零分割模型。

回答的是那条一直没验的问题：SaGA 把像素换成 OOD 背景时，换掉的是**哪里**？
如果换掉的正是策略要用的东西（夹爪、被操作物），那「两个相机各自独立增强」就会让
同一个物理表面在一个视角里是真实颜色、在另一个视角里是 OOD 颜色 —— 那才是真问题。
如果换掉的是边角背景，独立增强就基本无害。

## 凭什么不用几何就能定位「不能改的地方」

一张**时序标准差图** σ(x,y) 就够了，而且两个相机各有各的含义：

| 相机 | σ 低 | σ 高 |
|---|---|---|
| `up`（静止） | 桌子、墙面等不动的背景 | **动的机械臂、被搬运的物体** |
| `front`（装在夹爪上） | **夹爪自身** —— 相机与夹爪刚性固连，夹爪的投影位置**逐帧不动** | 身后掠过的世界（视差） |

所以 σ 在两个相机上都直接给出「改了就错」的区域，且不需要标定、不需要 URDF。

## 必须复现训练的两步量化，否则量的是另一个东西

`BufferManager.set/get`（`buffer.py:75-91`）把显著图**先缩到 120×160、量化成 uint8**，
取回时**再双线性放回 480×640**。本脚本照做，不图省事直接在原分辨率算 ——
分辨率不同，s 的分布就不是训练时那个分布。

用法：
    python workplace/saga/scripts/where_replaced.py --ckpt <ckpt> --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
SAGA = HERE.parents[1]
WORKPLACE = HERE.parents[2]
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saga.saga import RoboSaGA, normalize_smaps                    # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BUFFER_SHAPE = (120, 160)          # 与训练一致（buffer.py 的默认）
LAMBDA = 0.8                       # 与训练一致


def smaps_as_training(smaps: torch.Tensor, out_hw) -> torch.Tensor:
    """把原始显著图变成**训练时真正用的** s。逐行对应 buffer.set + buffer.get +
    saga.py 的 clip + 平方，一步不少。

    顺序：bilinear → 120×160 → ×255 取 uint8 → /255 → bilinear → 480×640
          → clip(0, λ) → 平方
    中间那次 uint8 量化不是形式，它把 s 的取值限制到 256 档，会真实地改变分布。
    """
    x = smaps
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if tuple(x.shape[-2:]) != BUFFER_SHAPE:
        x = F.interpolate(x.float(), size=BUFFER_SHAPE, mode="bilinear",
                          align_corners=False)
    x = (x * 255).to(torch.uint8)                     # <- buffer.set 的最后一步
    x = x.float() / 255.0                             # <- buffer.get 的第一步
    if tuple(x.shape[-2:]) != tuple(out_hw):
        x = F.interpolate(x, size=tuple(out_hw), mode="bilinear",
                          align_corners=False)
    x = torch.clip(x, 0, LAMBDA)                      # <- saga.py:210
    return x ** 2                                     # <- saga.py:216


def pick_frames(ds, n_ep: int, per_ep: int, min_len: int):
    """跨集均匀取样。SaGA 的缓冲区按全局帧 id 索引，不要求帧连续。"""
    spans = ds.meta.episodes.to_pandas()
    lengths = (spans["dataset_to_index"] - spans["dataset_from_index"]).to_numpy()
    ok = np.nonzero(lengths >= min_len + per_ep)[0]
    if len(ok) == 0:
        raise SystemExit(f"没有长度 ≥ {min_len + per_ep} 的集")
    eps = ok[np.linspace(0, len(ok) - 1, min(n_ep, len(ok))).round().astype(int)]
    out = []
    for e in eps:
        lo = int(spans.iloc[e]["dataset_from_index"])
        org = int(spans.iloc[e]["dataset_to_index"])
        usable = org - lo - min_len
        offs = (np.linspace(0, max(usable - per_ep, 0), per_ep)
                .round().astype(int))
        out += [lo + int(o) for o in offs]
    return sorted(set(out)), eps


def contiguous_runs(ds, eps, n_ep: int, T: int):
    """每个集取一段**连续**帧，用来算时序标准差。"""
    spans = ds.meta.episodes.to_pandas()
    runs = []
    for e in eps[:n_ep]:
        lo = int(spans.iloc[e]["dataset_from_index"])
        org = int(spans.iloc[e]["dataset_to_index"])
        T_ = min(T, org - lo)
        runs.append([ds[lo + i] for i in range(T_)])
    return runs


def temporal_std(runs, key):
    """多段连续帧的逐像素时序标准差（对通道取均值），返回 (H,W) 与帧数。"""
    acc, n = None, 0
    for run in runs:
        # ds[i][key] 已经是 float32 的 [0,1]，不是 uint8 —— 别再除 255
        x = torch.stack([r[key] for r in run]).float()            # (T,3,H,W)
        if acc is None:
            # 注意是 x[0, 0] 不是 x[0] —— x[0] 是 (3,H,W)，zeros_like 会把 σ 弄成
            # 三维，然后靠广播悄悄算出一个错的 (3,H,W)（踩过）
            acc = torch.zeros_like(x[0, 0])
        acc += x.std(dim=0).mean(dim=0)                            # (H,W)
        n += 1
    return (acc / n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset", default="record-test10")
    ap.add_argument("--root", default=str(REPO / "dataset" / "newdata1"))
    ap.add_argument("--out", default=str(REPO / "workplace/saga/figures"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=8, help="取样覆盖几集")
    ap.add_argument("--per-ep", type=int, default=24, help="每集取几帧")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--std-episodes", type=int, default=4)
    ap.add_argument("--std-frames", type=int, default=120, help="每段连续多少帧")
    ap.add_argument("--tail", type=int, default=60,
                    help="连续段跳过开头这么多帧（开头是静止的对准阶段）")
    args = ap.parse_args()

    dev = torch.device(args.device)
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

    # 与 preview_augmentation.py 同样的硬断言：归一化空间必须与训练一致
    for k in ds.meta.camera_keys:
        m = torch.as_tensor(ds.meta.stats[k]["mean"]).flatten()[:3]
        assert torch.allclose(m, torch.tensor([0.485, 0.456, 0.406]), atol=1e-5), \
            f"{k} 的统计量不是 ImageNet —— 归一化空间与训练不一致"

    camera_keys = list(policy.config.image_features)
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

    # FullGrad 只需要 backbone；这里不建 BufferManager，权重由 smaps_as_training 复现
    saga = RoboSaGA(
        backbone=policy.model.backbone, image_keys=camera_keys,
        image_shape=(ds.meta.features[camera_keys[0]]["shape"][0],
                     ds.meta.features[camera_keys[0]]["shape"][1]),
        background_path=str(WORKPLACE / "background_generalization" /
                            "backgrounds" / "solid8"),
        buffer_depth=1, disable_buffer=True, device=dev, warmup_steps=0,
    )

    idxs, eps = pick_frames(ds, args.episodes, args.per_ep, args.tail)
    print(f"\n[取样] {len(idxs)} 帧，覆盖 {len(eps)} 集：{eps.tolist()}")

    H, W = saga.image_shape
    acc = {k: dict(repl=torch.zeros(H, W, device=dev),
                   grad=torch.zeros(H, W, device=dev), n=0)
           for k in camera_keys}

    with torch.no_grad():
        for b0 in range(0, len(idxs), args.batch):
            chunk = idxs[b0:b0 + args.batch]
            batch = torch.utils.data.default_collate([ds[i] for i in chunk])
            batch = preprocessor(batch)
            for key in camera_keys:
                imgs = batch[key]
                # 外面那层 no_grad 会让 FullGrad 建不出图（它靠 requires_grad_()，
                # 而 no_grad 下算子不记录 grad_fn，于是 hook 报
                # "cannot register a hook on a tensor that doesn't require gradient"）。
                # 这里局部把梯度重新打开。
                with torch.enable_grad():
                    maps = normalize_smaps(saga._fullgrad(imgs.clone())).detach()
                s = smaps_as_training(maps, (H, W))
                a = acc[key]
                a["repl"] += (1 - s[:, 0]).sum(0)
                # 结构密度：|∇I|（在**归一化空间**里算，与显著度同一个空间）
                g = imgs.mean(1)                                  # (B,H,W)
                gx = (g[:, :, 1:] - g[:, :, :-1]).abs()           # (B,H,W-1)
                gy = (g[:, 1:, :] - g[:, :-1, :]).abs()           # (B,H-1,W)
                # 分开累加，别先相加 —— 两者形状差一列，直接加会广播错误
                a["grad"][:, :-1] += gx.sum(0)
                a["grad"][:-1, :] += gy.sum(0)
                a["n"] += len(chunk)
            print(f"  [{b0 + len(chunk)}/{len(idxs)}]", end="\r", flush=True)
    print()

    # 均值图重算一遍（上面那次是为了并行，这里干净地重来一次代价很小）
    runs = contiguous_runs(ds, eps, args.std_episodes, args.std_frames)
    print(f"[时序] {len(runs)} 段连续帧，每段 {args.std_frames} 帧")
    std_maps, mean_imgs = {}, {}
    for key in camera_keys:
        std_maps[key], _ = temporal_std(runs, key)
        mean_imgs[key] = torch.stack(
            [r[key].float() for run in runs for r in run]).mean(0)   # 已是 [0,1]
        a = acc[key]
        a["repl"] /= a["n"]
        a["grad"] /= a["n"]

    # ---------------- 出图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    CMAP = "inferno"          # 感知均匀、亮度单调、色盲安全；jet 会凭空造分界
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for key in camera_keys:
        cam = key.split(".")[-1]
        repl = acc[key]["repl"].cpu().numpy()
        tau = std_maps[key].numpy()
        grad = acc[key]["grad"].cpu().numpy()
        # 反归一化后才能显示 —— 归一化空间的均图直接 imshow 出来是全黑的。
        # 常数是 (1,3,1,1) 四维，得先拍成 (3,1,1)，否则广播出个 (1,3,H,W)。
        img = (mean_imgs[key] * IMAGENET_STD[0] + IMAGENET_MEAN[0])
        img = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)

        # 替换质量的集中度：按像素均匀分布时 Gini=0
        v = np.sort(repl.ravel())
        n = len(v)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))

        # τ 分位：低 τ = 不该改的地方
        qs = np.quantile(tau, [0.2, 0.5, 0.8])
        stable = tau <= qs[0]
        moving = tau >= qs[2]
        mass_stable = float(repl[stable].sum() / repl.sum())
        mass_moving = float(repl[moving].sum() / repl.sum())
        summary[cam] = dict(
            repl_mean=float(repl.mean()), gini=gini,
            repl_q=[float(x) for x in np.quantile(repl, [0.1, 0.5, 0.9])],
            frac_over_half=float((repl > 0.5).mean()),
            area_stable=float(stable.mean()), area_moving=float(moving.mean()),
            mass_stable=mass_stable, mass_moving=mass_moving,
            grad_stable=float(grad[stable].mean()), grad_all=float(grad.mean()),
            tau_med=float(np.median(tau)))

        # 每根色标进自己的 gridspec 列 —— 用 fig.colorbar(ax=[...]) 会和相邻面板叠住
        fig = plt.figure(figsize=(23, 4.6), constrained_layout=True)
        gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.035, 1, 1, 0.035])
        ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
              fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4])]
        cax = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 5])]

        ax[0].imshow(img); ax[0].set_title(f"{cam}  平均画面", fontsize=10)
        im1 = ax[1].imshow(tau, cmap=CMAP)
        ax[1].set_title("时序标准差 σ —— 亮的地方在动", fontsize=10)
        cb1 = fig.colorbar(im1, cax=cax[0]); cb1.set_label("σ（满量程归一）", labelpad=8)
        im2 = ax[2].imshow(repl, cmap=CMAP, vmin=0, vmax=1)
        ax[2].set_title("平均被替换比例 1−s", fontsize=10)
        # σ 最低 20% 的轮廓叠在替换图上：圈内 = 「改了就错」的区域
        ax[3].contour(tau, levels=[qs[0]], colors="w", linewidths=1.6)
        ax[3].imshow(repl, cmap=CMAP, vmin=0, vmax=1)
        ax[3].set_title("同上 + 白线 = σ 最低 20%（静止 / 夹爪自身）", fontsize=10)
        cb2 = fig.colorbar(im2, cax=cax[1]); cb2.set_label("1−s（0–1）", labelpad=8)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"OOD 替换落在哪里 —— 相机 {cam}"
                     f"（{acc[key]['n']} 帧取样，σ 来自 {len(runs)} 段连续帧）",
                     fontsize=13)
        p = outdir / f"replaced_where_{cam}.png"
        fig.savefig(p, dpi=80); plt.close(fig)
        print(f"[save] {p}")

    # ---------------- 定量图：替换质量按 σ 分位分布 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for key in camera_keys:
        cam = key.split(".")[-1]
        tau = std_maps[key].numpy().ravel()
        repl = acc[key]["repl"].cpu().numpy().ravel()
        edges = np.quantile(tau, np.linspace(0, 1, 11))
        edges[-1] += 1e-9
        which = np.clip(np.digitize(tau, edges) - 1, 0, 9)
        fr = np.array([repl[which == i].mean() for i in range(10)])
        share = np.array([repl[which == i].sum() for i in range(10)])
        share = share / share.sum()
        ax[0].plot(np.arange(10), fr, "o-", label=cam)
        ax[1].plot(np.arange(10), share, "o-", label=cam)
    ax[0].set_xlabel("σ 十分位（0 = 最静止，9 = 最活跃）")
    ax[0].set_ylabel("平均被替换比例 1−s")
    ax[0].set_title("「改了就错」的区域被改得更多还是更少")
    ax[1].set_xlabel("σ 十分位")
    ax[1].set_ylabel("占全部替换质量的比例")
    ax[1].set_title("替换质量花在哪一档上")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout()
    p = outdir / "replaced_where_summary.png"
    fig.savefig(p, dpi=80); plt.close(fig)
    print(f"[save] {p}")

    print("\n=== 数字 ===")
    for cam, s in summary.items():
        print(f"\n[{cam}]")
        print(f"  平均被替换比例         {s['repl_mean']*100:5.1f}%  "
              f"（= 1−s 的均值；s 是权重，s=1 表示保留原图）")
        print(f"  逐像素 1−s 的 10/50/90 分位  "
              f"{s['repl_q'][0]:.2f} / {s['repl_q'][1]:.2f} / {s['repl_q'][2]:.2f}"
              f"   （三档挤在一起 = 权重几乎是个常数，逐像素引导没起作用）")
        print(f"  被替换过半（1−s>0.5）的像素  {s['frac_over_half']*100:5.1f}%")
        print(f"  替换质量的集中度 Gini  {s['gini']:.3f}   （0=均匀铺满全画面，1=集中在一点）")
        print(f"  σ 最低 20% 的面积占比  {s['area_stable']*100:.1f}%")
        print(f"  → 这部分吃掉了替换质量的 {s['mass_stable']*100:.1f}%；"
              f"若替换是均匀的，应该是 {s['area_stable']*100:.1f}%")
        print(f"  σ 最高 20% 吃掉         {s['mass_moving']*100:.1f}%")
        print(f"  结构密度 |∇I|：静止区 {s['grad_stable']:.4f} vs 全图 {s['grad_all']:.4f}")
    print("\n读法：mass_stable 明显**高于** area_stable = 替换偏向了静止/夹爪区（坏）；"
          "\n      明显**低于** = 替换避开了它（好，说明显著度在起作用）。")


if __name__ == "__main__":
    main()
