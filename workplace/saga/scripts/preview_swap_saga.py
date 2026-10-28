#!/usr/bin/env python
"""把 SaGA 增强施加到「**物体已被换掉**」的图上 —— 三个来源共用一个驱动。

    --source distractor   桌上干扰物（蓝块）→ 重建网格   `pose6d_distractor`，双视角，连续帧
    --source up_fit       up 视图里被夹的目标（黄块）→ 重建网格   `pose6d_up_fit`，只用 up
    --source wrist_fit    腕视图里被夹的目标（黄块）→ 重建网格    `pose6d_wrist_fit`，只用 front

链条（每一步都是现成的，这个脚本只负责接起来、不另写一遍）：

    替换图（哪个来源都行）
        ↓  把它**当成数据集里的那一帧**塞回 batch
    preprocessor        ImageNet 归一化（和训练完全同一条路）
        ↓
    RoboSaGA            FullGrad 显著度 → 缓冲区 → 逐像素混合 OOD 背景
        ↓
    图

**为什么要把替换图塞回 batch，而不是拿现成的 PNG 拼**：SaGA 的显著度是**策略对当前
这张图**算出来的（论文 Algorithm 1 在线那一步）。拿原帧的显著度去叠替换图，
权重和它下面那张图就对不上，成了两件事拼在一起。所以先把替换图写回 batch，再照常走 FullGrad。

**batch 的摆法**：前 m 个位置放「要替换、要展示」的帧，后面用另一集里的真实帧填空
（SaGA 要求 batch ≥ 1/β = 10）。α=0.5 只增强前一半，所以目标帧必须摆在前半段。
填空帧不替换 —— 它们只是为了凑够 batch，不影响被展示那几帧的显著度
（显著度逐帧独立、按全局帧 id 索引，跨帧不串）。

**这张图要回答的问题**：注入的舵机是**策略从没见过的物体**（训练集只有黄块、蓝块、红盒），
SaGA 会保护它吗？显著度来自策略自己 ⇒ 概率上**不保护**。报的数按「语义区内的 s 均值
vs 全图 s 均值」，不报「被改动像素占比」（纯色 OOD 池下它恒为 100%，没有分辨力）。

用法：

    python workplace/saga/scripts/preview_swap_saga.py --source up_fit
    python workplace/saga/scripts/preview_swap_saga.py --source wrist_fit
    python workplace/saga/scripts/preview_swap_saga.py --source distractor
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
CAL = WORKPLACE / "calibration"
for _p in (str(REPO / "src"), str(WORKPLACE), str(CAL), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saga.saga import RoboSaGA, normalize_smaps              # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# 每个来源：用哪几个相机、在哪几帧（**绝对**数据集帧号；这一集从 0 开始）。
# 帧号取自各自脚本 main() 的默认值，和已出的那几张图是同一组，方便并排看。
# 干扰物那组要**跨集散布**：连着几帧姿态几乎一样，四行会看着像同一张。
# 0/20/100/340 是原 `preview_distractor_swap.py` 用的那一组，保持一致好并排对照。
DISTRACTOR_FRAMES = [0, 20, 100, 340]

SOURCES = {
    "distractor": dict(cams=("up", "front"), frames=None,
                       title="干扰物（桌上蓝块）→ 重建网格"),
    "up_fit": dict(cams=("up",), frames=(338, 246, 350, 402),
                   title="up 视图：被夹的目标（黄块）→ 重建网格"),
    "wrist_fit": dict(cams=("front",), frames=(300, 336, 400),
                      title="腕视图：被夹的目标（黄块）→ 重建网格"),
}


def denorm(xn: torch.Tensor) -> np.ndarray:
    """归一化空间 → 可显示的 [0,1] HWC。"""
    x = xn.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x.permute(0, 2, 3, 1).numpy(), 0, 1)


def bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """合成图（BGR uint8 HWC）→ 数据集同款张量。

    数据集里的相机键是 **RGB、CHW、float32、[0,1]**，所以这里必须换通道序 ——
    少了这个 `::-1` 不报错，只会让红蓝对调。
    """
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)


# ------------------------------------------------------------ 三个来源的装配

def ctx_distractor(ds, args):
    import ball_swap as BS
    import pose6d_distractor as PD
    g = BS.load_geom()
    mesh, albedo, half_h, ext, name = PD.build_mesh(args.mesh)
    P_anchor, c0, dev = PD.anchor(g, ds)
    if P_anchor is None:
        raise SystemExit("[锚] up 图里没找到蓝块")
    print(f"[网格] {name}  {ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f}mm  h={half_h*1000:.2f}mm"
          f"｜锚点 {np.round(P_anchor*1000, 2)}mm（重复性 max {dev.max():.2f}mm）")
    return dict(g=g, BS=BS, PD=PD, mesh=mesh, albedo=albedo, P_anchor=P_anchor, c0=c0)


def compose_distractor(ds, i, ctx, args):
    u = ctx["PD"].up_frame(ds, i, ctx["g"], ctx["P_anchor"], ctx["c0"], ctx["mesh"],
                           args.yaw_deg, ctx["albedo"], args.draw_cell_mm / 1000.0,
                           ctx["BS"].UP_CAM_H_MM, verbose=False)
    w = ctx["PD"].wrist_frame(ds, i, ctx["g"], ctx["P_anchor"], ctx["mesh"],
                              args.yaw_deg, ctx["albedo"], args.draw_cell_mm / 1000.0,
                              verbose=False)
    out = {}
    if u is not None:
        out["observation.images.up"] = (u["out"], u["ras"]["mask"])
    if w is not None:
        out["observation.images.front"] = (w["out"], w["ras"]["mask"])
    return out


def ctx_up_fit(ds, g, args):
    import pose6d_up_fit as UF
    ut, seed = UF.P6.up_track_ctx(ds, g, args.grasp, args.release)
    if not ut:
        raise SystemExit("[up] 夹取前没在 up 图里找到方块")
    mesh = UF.F.load_mesh(args.mesh)
    print(f"[网格] {mesh.name}  面数 {len(mesh.tris0)}｜up 跟踪 {len(ut)} 帧")
    return dict(g=g, UF=UF, ut=ut, mesh=mesh, albedo=UF.PW.SERVO_ALBEDO,
                x_cube=ctx_x_cube(ds, g, args, UF.BS))


def compose_up_fit(ds, i, ctx, args):
    r = ctx["UF"].up_fit_frame(ds, i, ctx["g"], ctx["ut"], ctx["x_cube"], ctx["mesh"],
                               albedo=ctx["albedo"], fix_s=1.0,
                               cell_fit=args.fit_cell_mm / 1000.0,
                               cell_draw=args.draw_cell_mm / 1000.0, verbose=False)
    return {} if r is None else {"observation.images.up": (r["out"], r["ras"]["mask"])}


def ctx_wrist_fit(ds, g, args):
    import pose6d_wrist_fit as WF
    trk = WF.BS.carry_track(ds, args.grasp, args.release)
    if not trk:
        raise SystemExit("[腕] 夹持段里没有方块观测")
    mesh = WF.F.load_mesh(args.mesh)
    print(f"[网格] {mesh.name}  面数 {len(mesh.tris0)}｜腕跟踪 {len(trk)} 帧")
    return dict(g=g, WF=WF, trk=trk, mesh=mesh, albedo=WF.SERVO_ALBEDO,
                x_cube=ctx_x_cube(ds, g, args, WF.BS))


def compose_wrist_fit(ds, i, ctx, args):
    r = ctx["WF"].wrist_fit_frame(ds, i, ctx["g"], ctx["trk"], ctx["mesh"],
                                  x_cube=ctx["x_cube"], fix_s=1.0,
                                  albedo=ctx["albedo"], verbose=False)
    return {} if r is None else {"observation.images.front": (r["out"], r["ras"]["mask"])}


def ctx_x_cube(ds, g, args, BS):
    """`X_tcp_cube` 只在**夹取帧**上算一次（搬运期一帧一算会解出 300mm 的假偏移）。
    和 `pose6d_swap.main` 用同一个锚点、同一条 up 掩码路。"""
    X, note = BS.estimate_x_tcp_cube(
        ds, args.grasp, g, BS.UP_CAM_H_MM / 1000.0, args.cube_h_mm / 1000.0,
        BS.cube_mask_fns(g, "up", "yellow")[0])
    print(f"[常量] X_tcp_cube = {np.round(np.asarray(X)*1000, 2)}mm  ({note})")
    return X


CTX = {"distractor": lambda ds, g, a: ctx_distractor(ds, a),
       "up_fit": ctx_up_fit, "wrist_fit": ctx_wrist_fit}
COMPOSE = {"distractor": compose_distractor, "up_fit": compose_up_fit,
           "wrist_fit": compose_wrist_fit}


# ------------------------------------------------------------ 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--ckpt", default=str(REPO / "outputs/train/act_record_test10/checkpoints/last/pretrained_model"),
                    help="默认基线 ACT（与 preview_augmentation.py 同一个），几张图才可比")
    ap.add_argument("--dataset", default="record-test10")
    ap.add_argument("--root", default=str(REPO / "dataset" / "newdata1"))
    ap.add_argument("--out", default=str(REPO / "workplace/saga/figures/swap_saga"))
    ap.add_argument("--device", default="cpu",
                    help="默认 cpu —— 两个 GPU 常被掩码任务占着，不抢")
    ap.add_argument("--n", type=int, default=16, help="batch 大小，必须 ≥ 1/β=10")
    ap.add_argument("--show", type=int, default=4, help="替换并展示几帧（会截到来源的帧数）")
    ap.add_argument("--frames", default=None,
                    help="覆盖来源的帧列表，逗号分隔的**绝对**数据集帧号。"
                         "distractor 要跨集散布（默认 0,20,100,340）才看得出不同姿态；"
                         "up_fit / wrist_fit 的默认值是各自脚本 main() 里那几帧。")
    ap.add_argument("--filler-episode", type=int, default=1, help="拿哪一集填满 batch")
    ap.add_argument("--mesh", default="sts3215_03a_v1_v50_h0.5_m1")
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    ap.add_argument("--draw-cell-mm", type=float, default=1.5)
    ap.add_argument("--fit-cell-mm", type=float, default=3.0)
    ap.add_argument("--cube-h-mm", type=float, default=21.5)
    ap.add_argument("--grasp", type=int, default=246)
    ap.add_argument("--release", type=int, default=409)
    ap.add_argument("--background", default=None)
    args = ap.parse_args()

    src = SOURCES[args.source]
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
    cfg.device = str(dev)
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg)
    policy.eval()

    # 必须走 make_dataset：use_imagenet_stats 会把相机 mean/std 换成 ImageNet 常数。
    # 直接 LeRobotDataset 拿到的是数据集自己的统计量，归一化空间差 ~17 倍，混合结果
    # 就是垃圾 —— 不报错，只是图花。见 workplace/saga/CLAUDE.md 坑 4。
    ds = make_dataset(TrainPipelineConfig(
        policy=cfg, dataset=DatasetConfig(repo_id=args.dataset, root=args.root)))
    for k in ds.meta.camera_keys:
        m = torch.as_tensor(ds.meta.stats[k]["mean"]).flatten()[:3]
        s = torch.as_tensor(ds.meta.stats[k]["std"]).flatten()[:3]
        assert torch.allclose(m, torch.tensor([0.485, 0.456, 0.406]), atol=1e-5), \
            f"{k} mean={m.tolist()} 不是 ImageNet —— 归一化空间与训练不一致"
        assert torch.allclose(s, torch.tensor([0.229, 0.224, 0.225]), atol=1e-5), \
            f"{k} std={s.tolist()} 不是 ImageNet —— 归一化空间与训练不一致"
    print("[cfg] 相机统计量 = ImageNet（与训练一致）")

    camera_keys = list(policy.config.image_features)

    # ---- batch 的帧号：前 m 个是目标（要替换、要展示），后面拿另一集填空 ----
    spans = ds.meta.episodes.to_pandas()
    lo_fill = int(spans.iloc[args.filler_episode]["dataset_from_index"])
    if args.frames:
        want = [int(x) for x in args.frames.split(",") if x.strip()]
    elif src["frames"] is None:
        want = DISTRACTOR_FRAMES
    else:
        want = list(src["frames"])
    tgt = want[:min(args.show, len(want))]
    fill = [lo_fill + j for j in range(args.n - len(tgt))]
    gids = tgt + fill
    print(f"[batch] {args.source}：目标帧 {tgt}（要替换）+ 填空 {len(fill)} 帧来自 ep{args.filler_episode}")

    # ---- 只对目标帧跑替换；填空帧保持真实数据 ----
    g = None
    ctx = CTX[args.source](ds, None, args) if args.source == "distractor" else None
    if ctx is None:
        import ball_swap as BS
        g = BS.load_geom()
        ctx = CTX[args.source](ds, g, args)
    else:
        g = ctx["g"]

    items, masks, notes = [], {}, []
    for pos, i in enumerate(gids):
        it = ds[i]
        if pos < len(tgt):
            got = COMPOSE[args.source](ds, i, ctx, args)
            if got:
                it = dict(it)                    # 别原地改数据集返回的对象
                for key, (bgr, m) in got.items():
                    it[key] = bgr_to_tensor(bgr)
                    masks[(pos, key.split(".")[-1])] = m
            notes.append((i, sorted(k.split(".")[-1] for k in got)))
            print(f"  f{i}: 替换 {'+'.join(notes[-1][1]) if got else '无（原图保留）'}")
        items.append(it)
    n_ok = sum(1 for _, c in notes if c)
    if not n_ok:
        raise SystemExit("一帧都没替换成功 —— 检查帧号/跟踪器")
    print(f"[替换] {n_ok}/{len(tgt)} 个目标帧至少替换了一个相机")

    batch = torch.utils.data.default_collate(items)
    orig_batch = torch.utils.data.default_collate([ds[i] for i in gids])

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.ckpt, dataset_stats=ds.meta.stats,
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
    batch = preprocessor(batch)
    orig_batch = preprocessor(orig_batch)
    assert "index" in batch, "preprocessor 把 index 丢了 —— 缓冲区索引不到帧"
    gid = batch["index"]

    h, w = batch[camera_keys[0]].shape[-2:]
    saga = RoboSaGA(
        backbone=policy.model.backbone, image_keys=camera_keys, image_shape=(h, w),
        background_path=bg_path, buffer_depth=int(gid.max().item()) + 2,
        aug_ratio=0.5, update_ratio=0.1, saliency_saga_cap=0.8,
        warmup_steps=0,                 # 要看的正是 warmup 之后的行为
        buffer_shape=(120, 160), device=dev,
    )

    # 预热：把这批帧的显著度写进缓冲区（= 训练里 warmup 之后的状态）。不预热的话
    # 缓冲区初值全 1 → s = λ² = 0.64 是常数，图上就是 fixed-weight random overlay。
    k = int(args.n * 0.5)
    print(f"\n[预热] 对前 {k} 帧算 FullGrad 并写入缓冲区……")
    for key in camera_keys:
        maps = normalize_smaps(saga._fullgrad(batch[key][:k].clone())).detach()
        saga.buffers[key].set(maps, gid[:k])
    print("[预热] 完成")

    out = saga(batch, step=10_000_000)

    # ---- 红线 ----
    for kk in ("observation.state", "action", "index"):
        if kk in batch:
            assert torch.equal(out[kk], batch[kk]), f"{kk} 被改了 —— 违反红线"
    print("[红线] state / action / index 逐位不变 ✓")

    # ---- 数字：s 是逐像素权重，越大越被保护 ----
    print("\n[结果] 每个相机各自一份显著度、各自混合：")
    m = len(tgt)
    for key in camera_keys:
        cam = key.split(".")[-1]
        s = torch.clip(saga.buffers[key].get(gid[:k]), 0, saga.saliency_saga_cap) ** 2
        inj = np.stack([masks[(p, cam)] if (p, cam) in masks else np.zeros((h, w), bool)
                        for p in range(k)])
        inj_t = torch.from_numpy(inj)
        s_inj = float(s[:, 0][inj_t].mean()) if int(inj_t.sum()) else float("nan")
        print(f"  {cam:5s} 被增强 {k}/{args.n} 帧，后 {args.n-k} 帧逐位不变="
              f"{torch.equal(out[key][k:], batch[key][k:])}")
        print(f"        全图 s 均值 {s[:m].mean():.3f}（常数 0.64 = 退化）"
              f"｜**注入区({int(inj_t.sum())}px) s = {s_inj:.3f}**"
              f"  ← 策略没见过的东西被保护了吗")

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
    CMAP = "inferno"          # 不用 jet：逐像素权重用彩虹色标会凭空造出阈值感

    # 只画**这一来源真正用的相机**，免得出现一整列空白
    used = [k for k in camera_keys if k.split(".")[-1] in src["cams"]]
    cam_label = {"up": "俯视 up", "front": "腕 front"}
    print(f"\n[出图] {m} 帧 × {len(used)} 相机")

    fig, axes = plt.subplots(m * len(used), 4, figsize=(17, 3.4 * m * len(used)),
                             squeeze=False)
    r = 0
    for pos in range(m):
        for key in used:
            cam = key.split(".")[-1]
            s_img = torch.clip(saga.buffers[key].get(gid[pos:pos + 1]),
                               0, saga.saliency_saga_cap)[0, 0] ** 2
            inj = masks.get((pos, cam))
            s_txt = ""
            if inj is not None:
                s_txt = (f"  注入区 {float(s_img.cpu().numpy()[inj].mean()):.2f}"
                         f" vs 全图 {float(s_img.mean()):.2f}")
            rows = [
                (denorm(orig_batch[key][pos:pos + 1])[0],
                 f"① 原图（真实数据，第 {gids[pos]} 帧）"),
                (denorm(batch[key][pos:pos + 1])[0],
                 "② 物体替换后（黄块/蓝块 → 重建网格）"),
                (s_img.cpu().numpy(), f"③ SaGA 权重 s = clip(λ)²，在②上算{s_txt}"),
                (denorm(out[key][pos:pos + 1])[0], "④ SaGA 增强后 ← 策略实际吃到的"),
            ]
            for j, (im, title) in enumerate(rows):
                ax = axes[r][j]
                ax.imshow(im, cmap=CMAP, vmin=0, vmax=1) if j == 2 else ax.imshow(im)
                ax.set_title(f"{cam_label.get(cam, cam)}   {title}" if j == 0 else title,
                             fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            r += 1

    shown = "、".join(cam_label.get(k.split(".")[-1], k) for k in used)
    scope = "两个相机同时增强" if len(used) == 2 else f"只增强 {shown}"
    fig.suptitle(f"{src['title']} + RoboSaGA（{scope}）"
                 f"——OOD 池 = solid8 纯色，权重在**替换后**的图上算", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.945, 0.975])   # 给 suptitle 和色标留出空间
    # 色标用**专用 axes** 摆在最右边。别用 ax=list(axes[:, 2]) —— 那样 matplotlib 会从
    # 那一列中间几行的位置切一条缝，色标正好压在子图标题上（第一版就是这样）。
    cax = fig.add_axes([0.958, 0.30, 0.011, 0.42])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), cax=cax)
    cb.set_label("逐像素权重 s", labelpad=8)      # labelpad 只能建完 colorbar 后再设
    png = outdir / f"{args.source}_saga.png"
    fig.savefig(png, dpi=80)
    plt.close(fig)
    print(f"\n[save] {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
