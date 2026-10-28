#!/usr/bin/env python
"""同一时刻、两个视角，各自独立抽 OOD 背景 —— 看图，不是看数字。

## 这个脚本**原本想**证的事，以及一次更正

原本想展示「同一个物理物体，在两个视角里被换成两种不同的颜色」。**这个目标是成立的**
—— 被夹住的黄方块确实同时出现在两个相机里（证据见 `cube_visible_in_up.py`）。

⚠️ **中途我报告过「黄方块从不出现在 `up`」，那是错的，已撤回。** 错因是用了统一的
「饱和黄」阈值 `b<0.35`，而方块在 `up` 里被暖光洗成**米白**（实测 RGB
`[0.961,0.953,0.729]`，**b=0.73**），被整类判掉；我把 `up` 里那块误认成机械臂的暖光斜面。
**两个相机必须各用各的判据。**（同类错这一轮还犯过两次：橙纸盒、淡黄桌布都被当成过黄方块。）

**按背景物体**的百分比仍然不报（颜色掩码对橙纸盒/淡黄桌布分不干净）；
但**黄方块**本身两边都分得出，各用各的阈值即可。

## 数字部分

只报**全图平均 `1−s`**（不需要任何掩码，因而不可争辩）。按物体的分解见
`edge_vs_saliency.py` —— 那份用边缘强度分箱，同样不需要掩码。

用法：
    python workplace/saga/scripts/two_views_independent_bg.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
WORKPLACE = HERE.parents[2]
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saga.saga import RoboSaGA, normalize_smaps                # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm(x):
    x = x.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x.permute(0, 2, 3, 1).numpy(), 0, 1)


def cube_px(a: np.ndarray, cam: str) -> int:
    """该相机自己的方块判据。**两个相机不能用同一套阈值** ——
    up 里方块被暖光洗成米白（b≈0.73），front 里是饱和黄（b≈0.2）。
    先前用统一的「饱和黄」判据得出「方块不在 up」，是阈值造的假，
    见 `cube_visible_in_up.py`。"""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if cam == "up":
        m = (r > 0.60) & (g > 0.56) & (r - b > 0.16) & (g - b > 0.13) & (np.abs(r - g) < 0.13)
    else:
        m = (r > 0.72) & (g > 0.62) & (b < 0.40) & (np.abs(r - g) < 0.22)
    return int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset", default="record-test10")
    ap.add_argument("--root", default=str(REPO / "dataset" / "newdata1"))
    ap.add_argument("--out", default=str(REPO / "workplace/saga/figures"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=16, help="batch 大小，须 ≥ 1/β = 10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=3, help="展示几个时刻")
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
        assert torch.allclose(m, torch.tensor([0.485, 0.456, 0.406]), atol=1e-5)

    U, F = "observation.images.up", "observation.images.front"
    keys = [U, F]
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

    # 选帧：要**两个相机同时看得见方块**，否则图证明不了「方块是共享的」。
    # （取每集中点帧会选到方块正好被夹爪挡住的那一帧，up 检出 0 px，和标题矛盾。）
    spans = ds.meta.episodes.to_pandas()
    cand = []
    for e in np.linspace(0, len(spans) - 1, 10).round().astype(int):
        lo = int(spans.iloc[e]["dataset_from_index"])
        hi = int(spans.iloc[e]["dataset_to_index"])
        for t in np.linspace(10, hi - lo - 10, 24).astype(int):
            i = lo + int(t)
            cu = cube_px(ds[i][U].permute(1, 2, 0).numpy(), "up")
            cf = cube_px(ds[i][F].permute(1, 2, 0).numpy(), "front")
            if cu > 200 and cf > 2000:
                cand.append((min(cu / 400.0, cf / 4000.0), int(e), i, cu, cf))
    if not cand:
        raise SystemExit("没有找到两个相机同时看得见黄方块的帧")
    cand.sort(reverse=True)
    picks, seen = [], set()
    for sc, e, i, cu, cf in cand:            # 每集最多取一帧，避免三帧同集
        if e in seen:
            continue
        seen.add(e); picks.append(i)
        if len(picks) >= args.frames:
            break
    print("[选帧] 两个相机同时看得见黄方块：")
    for sc, e, i, cu, cf in cand[:6]:
        print(f"    ep{e} frame {i}: up {cu} px / front {cf} px{'  ←选' if i in picks else ''}")

    k = args.n // 2
    idxs = picks + [picks[0]] * (args.n - len(picks))
    batch = preprocessor(torch.utils.data.default_collate([ds[i] for i in idxs]))
    gid = batch["index"]
    h, w = batch[U].shape[-2:]

    saga = RoboSaGA(
        backbone=policy.model.backbone, image_keys=keys, image_shape=(h, w),
        background_path=str(WORKPLACE / "background_generalization" /
                            "backgrounds" / "solid8"),
        buffer_depth=int(gid.max().item()) + 2, aug_ratio=0.5, update_ratio=0.1,
        saliency_saga_cap=0.8, warmup_steps=0, buffer_shape=(120, 160), device=dev,
    )
    for key in keys:            # 预热缓冲区，否则 s 是常数、展示的不是 SaGA
        maps = normalize_smaps(saga._fullgrad(batch[key][:k].clone())).detach()
        saga.buffers[key].set(maps, gid[:k])

    torch.manual_seed(args.seed)
    out = saga(batch, step=10_000_000)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    m = len(picks)
    fig, ax = plt.subplots(4, m, figsize=(5.4 * m, 4.3 * 4))
    if m == 1:
        ax = ax[:, None]
    print("[每帧：饱和黄像素数 / 全图平均被替换]")
    for j in range(m):
        origs, augs, means = {}, {}, {}
        for key in keys:
            origs[key] = denorm(batch[key][j:j + 1])[0]
            augs[key] = denorm(out[key][j:j + 1])[0]
            s = torch.clip(saga.buffers[key].get(gid[j:j + 1]), 0, 0.8) ** 2
            means[key] = float((1 - s[0, 0]).mean().cpu())
        # 两个相机各自的方块像素数 —— 印在图上，说明方块**两边都看得见**
        nyu = cube_px(origs[U], "up")
        nyf = cube_px(origs[F], "front")
        for r, (key, nm) in enumerate([(U, "up"), (F, "front")]):
            ax[2 * r][j].imshow(origs[key]); ax[2 * r][j].set_title(f"{nm} 原始", fontsize=10)
            ax[2 * r + 1][j].imshow(augs[key])
            ax[2 * r + 1][j].set_title(f"{nm} 增强后", fontsize=10)
            for rr in (2 * r, 2 * r + 1):
                ax[rr][j].set_xticks([]); ax[rr][j].set_yticks([])
            if j == 0:
                ax[2 * r][j].set_ylabel(
                    f"{nm}\n（方块 {nyu if nm == 'up' else nyf} px）", fontsize=9)
        print(f"  帧 {idxs[j]:6d}:  up 方块 {nyu:5d} px 全图被替换 {means[U] * 100:5.1f}%"
              f"   |   front 方块 {nyf:5d} px 全图被替换 {means[F] * 100:5.1f}%")
    fig.suptitle(
        "同一时刻、两个视角，各自独立抽 OOD 背景\n"
        "共享内容＝桌面、机械臂、**以及被夹住的黄方块**（方块在两个视角都可见：\n"
        "up 里被暖光洗成米白 b≈0.73，front 里是饱和黄 b≈0.2 —— 不能用同一套阈值找它）",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "two_views_independent_bg.png"
    fig.savefig(p, dpi=80); plt.close(fig)
    print(f"\n[save] {p}")


if __name__ == "__main__":
    main()
