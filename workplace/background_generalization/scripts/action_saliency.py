#!/usr/bin/env python
"""Action saliency —— policy 到底在读图像里的哪些像素？

## 测量

A. 整路消融：整路图 / 它的前景 / 它的背景 换成噪声，量 ||Δaction||
   `full_noise` `full_mean` `full_const` `cross_ep` `fg_noise` `bg_noise`
B. 逐格热力图：切成 RxC 格，逐格换噪声，不需要任何分割

**填充值用噪声，不能用 plate。** plate 是时域中值图，对本来就是背景的格子，
用 plate 填等于什么都没改，Δ 恒为 0 —— "背景不重要"就成了被构造出来的结论。

## 「前景/背景」怎么定 —— 每个相机用对它成立的判据

判据由**实测的相机自身运动**选定（`--ego-max`，默认 2 px）：

* **相机静止**（实测 up：相邻帧整体位移中位 **0.26 px**）→ 「画面里在动 = 机器人」
  成立。用**时域判据**：`border_connected(时域 std 低)` = 桌面/墙 = 背景。
* **相机随臂运动**（实测 front：**17.2 px**，机械臂最快那帧 **45.7 px**）→
  时域判据**前提不成立**，必须换成**显式分割**（SAM2，见
  `masks/robot_mask.py`）。原因（都是实测）：

  1. **夹爪被判成背景** —— 它相对相机几乎不动，低 std；
  2. **有纹理的场景物被判成前景** —— 蓝盒/绿垫/线缆在画面里大幅移动，高 std；
  3. 均匀桌面被判成背景**纯属巧合** —— 它其实以 37-40 px/帧在动，只是没纹理，
     逐像元看不出变化（DIS 光流在"纹理最弱 20%"的像元上中位仍有 36.8 px）。

  光流累积、中值图纹理、单应残差都试过，全被测量排除（细节见 `robot_mask.py`）。

## 前景 = 机械臂/夹爪 ∪ 任务物

任务串是 `Pick up yellow block and put it in trash can`。把黄方块或红桶换成噪声
policy 当然会反应，但那是"任务物重要"，不是"背景重要" —— 不排除掉会把结论往
"背景有用"的方向推。所以两者都算前景（`--no-task-objects` 可关）。

## `--fg-mode union`：要跨相机比较时必须用这一档

上面 `auto` 给两个相机挑的是**不同**的判据（up=motion、front=sam2），于是"背景"
是两套不同的构造。实测 ep0：up 的前景 64.4%（其中"机械臂"53.3% —— 俯视图里一条
SO101 不可能占一半，时域判据把机械臂扫过的桌面也算进去了），front 14.2%。
两边的背景池因此是 35.6% vs 85.8% —— **面积和成分都不同**，`bg_swap_ep` 直接
比大小是在比"我的背景池和政治无关地更大"。

`union` 档读的是**增强本身用的那份掩码**（`foreground_mask.py`：
SAM2 机械臂 ∪ 黄/红/蓝，含 24 px 外扩），两个相机同一套构造，面积也接近
（实测 up 32.1% / front 32.4%），于是 `bg_swap_ep` 的面积差天然被压小，
"哪一路更依赖背景"才问得出来。**回答"up 和 front 要不要不同的增强"时必须用这档。**

## 两个必须守住的量

* **等面积**：up 的背景比 front 大，光看 Δ 大小不可比。两路各替换**同样多**的
  像素（`--matched-frac`，默认 25%）。
* **面积收缩**：池子小于目标面积时 `np.random.choice` 会静默取满整个池子，
  两边的面积就又不一样了。所以先按最小池子收缩（`f_eff`）。

不修改 src/lerobot/ 下任何文件，只做推理。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BEYOND = Path(__file__).resolve().parents[1]          # workplace/background_generalization
WORKPLACE = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "src"), str(WORKPLACE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from background_generalization.masks.foreground_mask import (  # noqa: E402
    load_foreground_mask,
)
from background_generalization.masks.task_object_mask import (  # noqa: E402
    TaskObjectConfig,
    task_object_masks_detailed,
    task_object_report,
)
from background_generalization.masks.robot_mask import (  # noqa: E402
    _to_hwc_uint8,
    load_robot_mask,
)

DEFAULT_ROBOT_MASK_ROOT = BEYOND / "outputs" / "robot_mask"


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
def load_policy(ckpt: str, device: torch.device):
    policy = ACTPolicy.from_pretrained(ckpt).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt)
    return policy, pre, post


@torch.no_grad()
def predict(policy, pre, post, batch: dict) -> torch.Tensor:
    """返回**真实单位**的动作块 (B, chunk, action_dim)。"""
    return post(policy.predict_action_chunk(pre(dict(batch))))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def episode_span(ds: LeRobotDataset, episode: int) -> tuple[int, int]:
    row = ds.meta.episodes.to_pandas().iloc[episode]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def sample_frames(lo: int, hi: int, n: int) -> list[int]:
    pad = max(1, int(0.05 * (hi - lo)))
    return np.linspace(lo + pad, hi - 1 - pad, n).astype(int).tolist()


# ---------------------------------------------------------------------------
# 相机自身运动：决定用哪套判据
# ---------------------------------------------------------------------------
def ego_motion(frames: np.ndarray) -> dict:
    """相邻采样帧之间的整体平移（相位相关）。相机随臂走时这个值很大。

    单位像素。深度不变时它只反映**相机自己**动了多少 —— 场景里的物体动
    不产生整体平移，所以这是"相机装在哪"的直接测量。
    """
    g = [cv2.cvtColor(_to_hwc_uint8(f), cv2.COLOR_RGB2GRAY) for f in frames]
    win = cv2.createHanningWindow((g[0].shape[1], g[0].shape[0]), cv2.CV_32F)
    mags = []
    for i in range(len(g) - 1):
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(g[i]), np.float32(g[i + 1]), win)
        mags.append(float(np.hypot(dx, dy)))
    mags = np.asarray(mags)
    return {"median": float(np.median(mags)), "p90": float(np.percentile(mags, 90)),
            "max": float(mags.max())}


# ---------------------------------------------------------------------------
# 前景掩码
# ---------------------------------------------------------------------------
def border_connected(mask: np.ndarray) -> np.ndarray:
    """只保留与画面边缘连通的连通分量（固定相机用）。"""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    border = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    border = border[border != 0]
    return np.isin(lab, border)


def motion_foreground(frames: np.ndarray, std_map: np.ndarray, bg_pct: float) -> np.ndarray:
    """固定相机：低 std 且接边的低 std 区 = 场景 = 背景，其余是前景。"""
    bg = border_connected(std_map <= float(np.percentile(std_map, bg_pct)))
    return ~bg


def cell_slices(h: int, w: int, rows: int, cols: int):
    hs = np.linspace(0, h, rows + 1).astype(int)
    ws = np.linspace(0, w, cols + 1).astype(int)
    for r in range(rows):
        for c in range(cols):
            yield r, c, slice(hs[r], hs[r + 1]), slice(ws[c], ws[c + 1])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cross-episode", type=int, default=None)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--bg-pct", type=float, default=50.0,
                    help="固定相机：逐像素时域 std 的低百分位，低于它算背景")
    ap.add_argument("--ego-max", type=float, default=2.0,
                    help="相机自身运动超过它(px)就判定相机随臂，该相机改用 SAM2 分割")
    ap.add_argument("--fg-mode", choices=["auto", "motion", "sam2", "union"], default="auto",
                    help="union = 直接用增强用的那份前景掩码，两个相机同一套构造。"
                         "跨相机比较背景敏感度时用这档（见模块文档）")
    ap.add_argument("--foreground-root", default=str(BEYOND / "outputs" / "foreground"))
    ap.add_argument("--robot-mask-root", default=str(DEFAULT_ROBOT_MASK_ROOT))
    ap.add_argument("--robot-dilate-px", type=int, default=24,
                    help="SAM2 掩码外扩；吃掉它漏掉的腕部连杆/线缆（实测见 robot_mask.load_robot_mask）")
    ap.add_argument("--no-task-objects", dest="task_objects", action="store_false", default=True)
    ap.add_argument("--sam2-union", dest="sam2_union", action="store_true", default=True,
                    help="motion 判据下也并上 SAM2 掩码兜底（关掉=忠实复现旧时域判据）")
    ap.add_argument("--no-sam2-union", dest="sam2_union", action="store_false")
    ap.add_argument("--matched-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    if args.repo_id is None or args.root is None:
        tc = json.loads((Path(args.ckpt) / "train_config.json").read_text())
        args.repo_id = args.repo_id or tc["dataset"]["repo_id"]
        args.root = args.root or tc["dataset"]["root"]

    policy, pre, post = load_policy(args.ckpt, device)
    cams = list(policy.config.image_features)
    _, H, W = policy.config.image_features[cams[0]].shape
    print(f"[cfg] {args.ckpt}\n[cfg] repo_id={args.repo_id} root={args.root}")
    print(f"[cfg] cameras={cams}  H={H} W={W}  chunk={policy.config.chunk_size}")

    ds = LeRobotDataset(args.repo_id, root=args.root)
    lo, hi = episode_span(ds, args.episode)
    sel = sample_frames(lo, hi, args.frames)
    cross = args.cross_episode if args.cross_episode is not None else args.episode + 1
    lo2, hi2 = episode_span(ds, cross)
    sel2 = sample_frames(lo2, hi2, args.frames)
    print(f"[data] ep{args.episode} frames[{lo},{hi}) 采样 {len(sel)} 帧；cross_ep = ep{cross}")

    items = [ds[i] for i in sel]
    items2 = [ds[i] for i in sel2]
    frames = {c: np.stack([it[c].numpy() for it in items]).astype(np.float32) for c in cams}

    acts = np.stack([it["action"].numpy() for it in items])
    nat_step = float(np.abs(np.diff(acts, axis=0)).mean())
    act_range = float((acts.max(0) - acts.min(0)).mean())
    print(f"[ref] 相邻帧 |Δaction| = {nat_step:.3f}；全程动作动态范围 = {act_range:.3f}")

    # ---- 判据选择：实测相机自身运动 ----
    print(f"\n=== 判据选择（--ego-max {args.ego_max} px）===")
    fg_frames, mode_of, diag = {}, {}, {}
    tocfg = TaskObjectConfig()
    for c in cams:
        ego = ego_motion(frames[c])
        mode = args.fg_mode
        if mode == "auto":
            mode = "sam2" if ego["median"] > args.ego_max else "motion"
        mode_of[c] = mode
        diag[c] = {"ego_motion": ego, "mode": mode}

        std_map = frames[c].std(0).mean(0)
        # 掩码存的是**集内**下标 0..T-1，sel 存的是**数据集全局**下标，必须减 lo。
        loc = [s - lo for s in sel]
        # 只有 sam2 判据需要外扩：motion 判据本来就把整条臂算进去了。
        rm = load_robot_mask(args.robot_mask_root, c, args.episode,
                             dilate_px=args.robot_dilate_px if mode == "sam2" else 0)
        if mode == "motion":
            fg = np.repeat(motion_foreground(frames[c], std_map, args.bg_pct)[None], len(sel), 0)
            if args.fg_mode == "motion" and ego["median"] > args.ego_max:
                why = (f"**强制**用时域判据，但实测相机自身位移 {ego['median']:.1f} px > "
                       f"{args.ego_max} px —— 该判据在这台相机上前提不成立，"
                       f"结果只应作为「旧做法」的对照读，不能当结论")
            else:
                why = (f"相机静止（整体位移中位 {ego['median']:.2f} px）→ "
                       f"「画面里在动 = 机器人」成立，用时域判据")
        elif mode == "union":
            # 增强本身用的那份掩码：外扩和任务物并集**都已经在里面**，所以这里
            # 不再 `_grow`、也不再并 yellow/red —— 重复做只会让两边看起来一致，
            # 其实把"两套定义"这个真问题盖掉了。
            fm = load_foreground_mask(args.foreground_root, c, args.episode)
            if fm is None:
                raise SystemExit(
                    f"{c} 指定 --fg-mode union（= 增强用的那份前景掩码），但没有找到：\n"
                    f"  {args.foreground_root}/{c}/episode_{args.episode:06d}.npy\n"
                    f"先跑：scripts/build_foreground_mask.py --episodes {args.episode}")
            fg = fm[loc] > 0
            why = ("用**增强本身**的那份前景掩码（SAM2 机械臂 ∪ 黄/红/蓝，含外扩）"
                   "→ 两个相机同一套构造，背景池面积和成分可比")
        else:
            if rm is None:
                raise SystemExit(
                    f"{c} 判定为随臂相机（整体位移中位 {ego['median']:.1f} px），"
                    f"时域判据不成立，需要 SAM2 掩码，但没有找到：\n"
                    f"  {args.robot_mask_root}/{c}/episode_{args.episode:06d}.npy\n"
                    f"先跑：scripts/build_robot_mask_cache.py --episodes {args.episode}")
            fg = rm[loc] > 0
            why = (f"相机随臂（整体位移中位 {ego['median']:.1f} px，最大 {ego['max']:.1f} px）→ "
                   f"时域判据前提不成立，用 SAM2 显式分割机械臂")
        if mode == "motion" and rm is not None and args.sam2_union:
            fg = fg | (rm[loc] > 0)   # 静止相机也并上 SAM2 掩码兜底夹爪
        diag[c]["robot_mask"] = ("有" if rm is not None else "无") + \
            ("" if (mode != "motion" or args.sam2_union) else "（未并入）")
        diag[c]["fg_before_task"] = float(fg.mean())

        # union 档里任务物**已经在掩码里**了，再并一次是空操作，而且会把
        # "我没在这一档重新检测任务物"这件事藏起来。所以显式跳过。
        add_task_objs = args.task_objects and mode != "union"
        if add_task_objs:
            det = [task_object_masks_detailed(_to_hwc_uint8(frames[c][k]), tocfg, camera_key=c)
                   for k in range(len(sel))]
            y = np.stack([d["yellow"]["mask"] for d in det])
            r = np.stack([d["red"]["mask"] for d in det])
            fg = fg | y | r
            agg = {k: {"dropped_frac": float(np.mean([d[k]["dropped_frac"] for d in det])),
                       "dropped_too_big_frac": float(np.mean([d[k]["too_big"] for d in det])) / float(H * W),
                       "biggest_dropped_frac": float(np.max([d[k]["biggest"] for d in det]))}
                   for k in ("yellow", "red")}
            diag[c]["task_object_report"] = task_object_report({"yellow": y, "red": r}, tocfg,
                                                               dropped=agg)
        fg_frames[c] = fg
        diag[c]["fg_frac"] = float(fg.mean())
        diag[c]["add_task_objs"] = bool(add_task_objs)
        print(f"  {c}: {why}")
        if mode == "union":
            print(f"     前景占画面 {fg.mean():.1%}（掩码自带，背景池 "
                  f"{(~fg).mean():.1%}）")
        else:
            print(f"     前景占画面 {fg.mean():.1%}（机械臂部分 {diag[c]['fg_before_task']:.1%}"
                  + (f"；黄 {y.mean():.2%} 红 {r.mean():.1%}" if add_task_objs else "") + "）")
        if add_task_objs:
            tr = diag[c]["task_object_report"]
            for k in ("yellow", "red"):
                d = tr[k]
                print(f"       {k}: 面积={d['area_frac']:.2%} 丢掉的块={d.get('dropped_frac', 0):.2%}"
                      f"（其中超尺寸 {d.get('dropped_too_big_frac', 0):.2%}，最大一块 "
                      f"{d.get('biggest_dropped_frac', 0):.1%}）")

    # ---- 基线 ----
    def mk_batch(items_):
        b = {"observation.state": torch.stack([it["observation.state"] for it in items_]).to(device),
             "task": [it.get("task", "") for it in items_]}
        for c in cams:
            b[c] = torch.stack([it[c] for it in items_]).to(device)
        return b

    base_batch = mk_batch(items)
    a_base = predict(policy, pre, post, base_batch).float().cpu()
    print(f"\n[base] chunk {tuple(a_base.shape)}，chunk 内 |a| 均值 {a_base.abs().mean():.3f}")

    def delta_of(batch) -> np.ndarray:
        a = predict(policy, pre, post, batch).float().cpu()
        return (a - a_base).abs().mean(dim=(1, 2)).numpy()  # (B,)

    # ---- 等面积比例：先收缩到最小池子供得起的量 ----
    pools = {c: {"fg": float(fg_frames[c].mean()), "bg": float((~fg_frames[c]).mean())}
             for c in cams}
    min_pool = min(pools[c][t] for c in cams for t in ("fg", "bg"))
    f_eff = min(args.matched_frac, 0.8 * min_pool)
    if f_eff < args.matched_frac:
        print(f"[warn] 最小池子 {min_pool:.1%}，等面积比例 {args.matched_frac:.0%} "
              f"收缩到 {f_eff:.1%} 以保证两路可比")

    # ================= A. 整路消融 =================
    abla = {c: {} for c in cams}
    for c in cams:
        B = len(sel)
        trials = {}
        b = dict(base_batch); b[c] = torch.rand_like(base_batch[c]); trials["full_noise"] = (b, 1.0)
        m = frames[c].mean(0)
        b = dict(base_batch)
        b[c] = torch.from_numpy(m).float().unsqueeze(0).repeat(B, 1, 1, 1).to(device)
        trials["full_mean"] = (b, 1.0)
        b = dict(base_batch); b[c] = torch.full_like(base_batch[c], 0.5); trials["full_const"] = (b, 1.0)
        cr = mk_batch(items2)
        b = dict(base_batch); b[c] = cr[c]; trials["cross_ep"] = (b, 1.0)

        # 换**真实**背景（不是噪声）：前景保留，背景换成另一集的对应帧。
        # 噪声是散点，CNN 可能只是被"输入分布变了"触发；换真实背景没有这个混淆，
        # 直接问「背景的**内容**有没有被用」。fg_swap 是反向对照，两者面积互补。
        fg_t = torch.from_numpy(fg_frames[c]).to(device)[:, None]
        b = dict(base_batch); b[c] = torch.where(fg_t, base_batch[c], cr[c])
        trials["bg_swap_ep"] = (b, float((~fg_frames[c]).mean()))
        b = dict(base_batch); b[c] = torch.where(fg_t, cr[c], base_batch[c])
        trials["fg_swap_ep"] = (b, float(fg_frames[c].mean()))

        # 等面积消融：两路替换**同样多**的像素
        rng = np.random.default_rng(args.seed)
        k = int(f_eff * H * W)
        for tag in ("fg", "bg"):
            src = fg_frames[c] if tag == "fg" else ~fg_frames[c]
            for name, matched in ((f"{tag}_noise", False), (f"{tag}_noise_matched", True)):
                if matched:
                    keep = np.zeros((B, H * W), dtype=bool)
                    for t in range(B):
                        pool = np.flatnonzero(src[t].ravel())
                        if pool.size == 0:
                            continue
                        pick = rng.choice(pool, size=min(k, pool.size), replace=False)
                        keep[t, pick] = True
                    keep = keep.reshape(B, H, W)
                else:
                    keep = np.repeat(src[:1], B, 0) if not src.shape[0] == B else src
                    keep = np.broadcast_to(keep, (B, H, W))
                mt = torch.from_numpy(keep).to(device)
                b = dict(base_batch)
                b[c] = torch.where(mt[:, None], torch.rand_like(base_batch[c]), base_batch[c])
                trials[name] = (b, float(keep.mean()))

        # 换成**一块纯色** —— 这才是增强**本身**做的那个操作
        # （`make_backgrounds.py --mode solid`），上面那些噪声/换集都是它的替代品。
        # 实测 ep0：两路的背景池本来就是 68.0% / 67.6%，所以这一档天然等面积，
        # 不需要再抽样。
        const = torch.full_like(base_batch[c], 0.5)
        for tag in ("fg", "bg"):
            src = fg_frames[c] if tag == "fg" else ~fg_frames[c]
            mt = torch.from_numpy(np.ascontiguousarray(src)).to(device)
            b = dict(base_batch)
            b[c] = torch.where(mt[:, None], const, base_batch[c])
            trials[f"{tag}_const"] = (b, float(src.mean()))

        print(f"\n=== {c} 整路消融（判据={mode_of[c]}；Δ 单位=真实动作单位，"
              f"全程动态范围={act_range:.2f}）===")
        for name, (tb, area) in trials.items():
            d = delta_of(tb)
            # **尾部**。均值小、max 大（实测 front 的 bg_noise_matched 均值 2.44 / max 19.9）
            # 意味着"大多数样本没事，少数样本错得厉害"。对抓取策略来说要命的是后者：
            # 一次失败就废一个 episode，而均值会把它平掉。所以报 p90 和
            # "Δ 超过一个自然步长的样本占比"。
            abla[c][name] = {"mean": float(d.mean()), "max": float(d.max()),
                             "p50": float(np.median(d)),
                             "p90": float(np.percentile(d, 90)),
                             "frac_gt_nat_step": float((d > nat_step).mean()) if nat_step else None,
                             "frac_gt_2nat_step": float((d > 2 * nat_step).mean()) if nat_step else None,
                             "area_frac": area,
                             "frac_of_range": float(d.mean() / act_range) if act_range else None,
                             "frac_of_nat_step": float(d.mean() / nat_step) if nat_step else None}
            print(f"  {name:26s} 面积={area:6.1%}  mean={d.mean():7.3f}  p90={np.percentile(d,90):7.3f}  "
                  f"max={d.max():7.3f}  Δ>1步 {float((d > nat_step).mean()):5.1%}  "
                  f"= 动态范围的 {d.mean()/act_range:6.2%}")

    # ================= B. 逐格热力图 =================
    results, std_cell = {}, {}
    for c in cams:
        d = np.zeros((args.rows, args.cols, len(sel)), dtype=np.float32)
        sc = np.zeros((args.rows, args.cols), dtype=np.float32)
        std_map = frames[c].std(0).mean(0)
        for r, cc, hs, ws in cell_slices(H, W, args.rows, args.cols):
            sc[r, cc] = std_map[hs, ws].mean()
            b = dict(base_batch)
            p = base_batch[c].clone(); p[:, :, hs, ws] = torch.rand_like(p[:, :, hs, ws])
            b[c] = p
            d[r, cc] = delta_of(b)
        results[c], std_cell[c] = d, sc
        m = d.mean(-1)
        corr = float(np.corrcoef(m.ravel(), sc.ravel())[0, 1])
        print(f"\n=== {c} 逐格热力图（不需要分割，对相机漂移免疫）===")
        print(f"  Δ 均值 {m.mean():.3f}  最大 {m.max():.3f}（{np.unravel_index(m.argmax(), m.shape)}）")
        print(f"  corr(格子Δ, 格子时域std) = {corr:+.3f}")
        abla[c]["cell_std_corr"] = corr

    # ---- 输出 ----
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"ckpt": args.ckpt, "episode": args.episode, "cross_episode": cross,
               "act_range": act_range, "natural_step": nat_step, "bg_pct": args.bg_pct,
               "ego_max": args.ego_max, "fg_mode": args.fg_mode,
               "diagnostics": diag, "ablations": abla}
    np.savez(out.with_suffix(".npz"),
             **{f"delta_{c}": results[c] for c in cams},
             **{f"std_{c}": std_cell[c] for c in cams},
             **{f"mean_{c}": frames[c].mean(0) for c in cams},
             **{f"fg_{c}": fg_frames[c] for c in cams},
             summary=json.dumps(summary, ensure_ascii=False))
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n[save] {out.with_suffix('.npz')}\n[save] {out.with_suffix('.json')}")
    render(out, cams, results, fg_frames, frames, act_range, mode_of)


def render(out, cams, results, fg_frames, frames, act_range, mode_of):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(cams)
    mid = fg_frames[cams[0]].shape[0] // 2
    fig, axes = plt.subplots(n, 4, figsize=(17, 3.9 * n), squeeze=False)
    for i, c in enumerate(cams):
        img = np.clip(_to_hwc_uint8(frames[c][mid]) / 255.0, 0, 1)
        ax = axes[i][0]
        ax.imshow(img)
        ax.set_title(f"{c}\n实帧 t={mid}")

        ax = axes[i][1]
        ax.imshow(img)
        m = fg_frames[c][mid]
        ax.imshow(np.ma.masked_where(~m, m), cmap="winter", alpha=0.5)
        ax.set_title(f"{c}\n绿蓝 = 前景（机械臂/夹爪 ∪ 任务物）\n判据 = {mode_of[c]}")

        ax = axes[i][2]
        ax.imshow(~fg_frames[c][mid], cmap="gray")
        ax.set_title(f"{c}\n判定的「背景」像素\n（占画面 {float((~fg_frames[c]).mean()):.1%}）")

        ax = axes[i][3]
        d = results[c].mean(-1)
        im = ax.imshow(d, cmap="inferno", vmin=0, vmax=max(1e-9, float(d.max())))
        ax.set_title(f"{c}\n逐格敏感度 Δaction（噪声替换）")
        plt.colorbar(im, ax=ax, fraction=0.046)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("ACT action saliency —— Δ 单位 = 真实动作单位；"
                 f"参考尺度 = 全程动作动态范围 {act_range:.2f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"[save] {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
