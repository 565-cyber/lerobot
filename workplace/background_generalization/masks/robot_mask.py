"""机械臂/夹爪掩码 —— 用 SAM2 显式分割，而不是靠时域统计去猜。

## 为什么不能用「时域 std 低 = 背景」

原有的做法（`scripts/action_saliency.py` 里那套）是：整集里逐像素时域 std
低的像元判成背景。它默认了「相机静止」，而 **front 恰恰不是静止相机**。

实测（newdata1 ep0，`observation.state` 找出机械臂最快的一帧 t*，相位相关
量相邻帧的整体平移）：

| 相机 | 整体平移 中位 | p90 | t* 处 |
|---|---|---|---|
| `up`    | **0.26 px** | 5.1 px  | **0.09 px** |
| `front` | **17.2 px** | 75.3 px | **45.7 px** |

front 装在手臂上，相机自己跟着手臂走。于是同一套判据在它身上同时犯两个错：

1. **夹爪被当成背景。** 夹爪离相机最近、相对相机的运动最小，低 std → 判成背景。
2. **有纹理的场景物体被当成前景。** 蓝盒、绿垫、线缆在画面里大幅移动，高 std
   → 判成前景。而均匀桌面判成背景**纯属巧合** —— 它其实在以 37-40 px/帧移动，
   只是没有纹理，逐像元看不出变化（实测：DIS 光流在「纹理最弱 20%」的像元上
   中位 36.8 px，说明低 std 是"没纹理"造成的假象，不是"没动"）。

## 为什么不能用光流/中值/单应去补

都试过，都被测量排除：

* 逐像元光流的中位（整集累积）：up 0.12 px、front 2.39 px —— 机械臂只在少数
  帧占住某个像元，时间中位把它抹平了；用分位数又回到 std 的老问题。
* 中值图纹理（"整集里一直存在的结构"）：front 上确实点亮了夹爪，但也点亮了被
  中值糊掉的**场景边缘**，因为糊掉的边缘仍有对比。
* 「中值图有结构」∧「时域 std 低」：交集只剩 0.6%，去掉碎块后 **0.0%** ——
  因为夹爪在 front 画面里**自己也在动**（腕关节在动），std 并不低。
* 单应拟合相机自身运动再做残差：front 的内点率在 6.7%-76% 之间跳，残差中位
  1.9-44.7 px，不稳到不能当交付。

结论：**front 里机械臂和场景都在动**，没有任何时域/运动统计量能把它们分开。
这不是阈值没调好，是判据的前提不成立。所以改成**显式分割**：让 SAM2 去找
「那个东西」，而不是让统计量去找「哪块没动」。

## 提示（prompt）怎么来

SAM2 只负责把提示扩成逐帧实例掩码，提示本身按相机各自的物理事实自动生成：

* `up`（**固定**相机）：机械臂是画面里唯一自走的物体。提示帧取**相邻帧变化最大**
  的那一对（机械臂最快），掩码取 `|frame_t - frame_{t-1}|` 的最大连通块。
  不依赖 plate，因此不受「机械臂在家时也进中值」的影响。
* `front`（**随臂**相机）：机械臂是整集里**一直在那儿**的东西。提示取时域中值图
  的结构强度 `T` 在**画面上半部**的最大连通块（相机装在腕后方，夹爪按安装几何
  必然出现在上方；这一条每集都会被面积自检复核），提示帧取该块内 `|frame - 中值|`
  最小的那一帧（夹爪最贴合中值位置时下笔）。

两个提示都**可覆盖**（`--up-box/--front-box/--frame`），并且脚本会把用到的提示
和自检结果打印出来、画出来，不静默。

## 存储

`{root}/{camera_key}/episode_{ep:06d}.npy`，uint8 **(T, H, W)，0/255**。
注意极性：**255 = 机械臂/夹爪（要保留、算前景）**。

这和本目录 `mask_utils.py` 的 0/1 约定不同（那个是"keep"掩码），也和数据集的
0/255 不同。故意另起目录名 `robot_mask`，避免同名同 dtype 反语义被静默搞混。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# SAM2 依赖：third_party 与权重都在 object_generalization 下
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
WORKPLACE = _HERE.parents[2]
REPO = _HERE.parents[3]
OBJECT_GEN = WORKPLACE / "object_generalization"
SAM2_ROOT = OBJECT_GEN / "third_party" / "sam2"
SAM2_CKPT = OBJECT_GEN / "checkpoints" / "sam2.1_hiera_base_plus.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


@dataclass
class RobotMaskConfig:
    """所有阈值都在这里，函数里不放魔数。"""

    # ---- 提示帧的候选集 ----
    prompt_frames: int = 150       # 用于算中值/结构的采样帧数
    pad_frac: float = 0.05         # 采样时两端各留掉的比例

    # ---- up（固定相机）：取相邻帧差异的最大连通块 ----
    up_diff_pts: int = 60          # 用前多少个候选差值图去选"机械臂最快"那一对
    # |frame_t - frame_{t-1}| 的分位阈值，从紧到松逐个试。
    # 单取高位会失败：差值集中在机械臂的高对比边缘上（棱、线缆、夹爪齿），
    # 阈值一高连通块就碎成边线，`_largest_component` 直接返回空。
    # 实测 ep0 最快那一对（t=322）：q99 → 0.95% 且熔断，q95 → 4.95% → 3.03% 一块。
    up_diff_pcts: tuple[float, ...] = (95.0, 92.0, 88.0)
    up_min_area_frac: float = 0.004

    # ---- front（随臂相机）：取中值图结构的上半部最大连通块 ----
    front_struct_pct: float = 92.0
    front_blur_sigma: float = 3.0
    front_min_area_frac: float = 0.010
    front_top_frac: float = 0.45   # 块质心必须落在画面上方这个比例内

    # ---- 自检 ----
    min_area_frac: float = 0.015   # 机械臂掩码面积下限（占画面）
    max_area_frac: float = 0.70    # 上限：超过就说明 SAM2 把桌面一起吞了
    min_valid_frac: float = 0.90   # 至少有这么大比例的帧要有非空掩码

    box_pad: int = 12              # 提示框外扩（像素）

    # ---- 重试阶梯 ----
    # 第一档自检不过时，后续档把提示框再外扩这么多像素（夹到画面内）。
    # 只在失败时才付这份代价，成功即停。
    retry_box_pad: int = 40


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _to_hwc_uint8(img) -> np.ndarray:
    """(3,H,W) float[0,1] 的 torch tensor / ndarray → (H,W,3) uint8。"""
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(a)


def gradmag(gray: np.ndarray) -> np.ndarray:
    return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))


def _largest_component(mask: np.ndarray, min_area: float) -> np.ndarray:
    from scipy import ndimage

    mask = ndimage.binary_closing(mask, np.ones((9, 9)))
    mask = ndimage.binary_opening(mask, np.ones((5, 5)))
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] < min_area:
        return np.zeros_like(mask)
    return lab == best


def _bbox_of(mask: np.ndarray, pad: int, shape) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = shape
    return [int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)),
            int(min(w - 1, xs.max() + pad)), int(min(h - 1, ys.max() + pad))]


def _gray(arr: np.ndarray, k: int) -> np.ndarray:
    return cv2.cvtColor(_to_hwc_uint8(arr[k]), cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# 提示生成
# ---------------------------------------------------------------------------
def up_prompt(frames: np.ndarray, cfg: RobotMaskConfig) -> dict:
    """固定相机：机械臂 = 画面里唯一自走的东西。

    取相邻帧整体变化最大的那一对（= 机械臂动得最快），差异图的最大连通块
    就是机械臂。**不用 plate** —— 机械臂在家时也会进中值，用 plate 做差会把
    家的位置一起点亮。
    """
    T = len(frames)
    idx = np.linspace(0, T - 2, min(cfg.up_diff_pts, T - 1)).astype(int)
    best, best_val = None, -1.0
    for k in idx:
        d = np.abs(_gray(frames, k + 1).astype(np.float32) - _gray(frames, k).astype(np.float32))
        v = float(d.mean())
        if v > best_val:
            best, best_val = k, v
    d = np.abs(_gray(frames, best + 1).astype(np.float32) - _gray(frames, best).astype(np.float32))
    seed, used = np.zeros_like(d, dtype=bool), None
    for q in cfg.up_diff_pcts:
        thr = float(np.percentile(d, q))
        seed = _largest_component(d > thr, cfg.up_min_area_frac * d.size)
        if seed.any():
            used = q
            break
    if not seed.any():                       # 退化：整幅图，交给 SAM2 自己找
        h, w = d.shape
        return {"camera": "up", "frame": int(best + 1), "box": [0, 0, w - 1, h - 1],
                "tier": "fallback-full-frame", "seed_frac": 0.0, "diff_mean": best_val}
    return {"camera": "up", "frame": int(best + 1), "tier": f"frame-diff(p{used:g})",
            "box": _bbox_of(seed, cfg.box_pad, d.shape),
            "seed_frac": float(seed.mean()), "diff_mean": best_val}


def front_prompt(frames: np.ndarray, cfg: RobotMaskConfig, rank: int = 0) -> dict:
    """随臂相机：机械臂 = 整集里**一直在那儿**的东西。

    时域中值图保留的是"一直存在的结构"：对随臂相机那就是机械臂（场景被糊掉）。
    取中值图结构强度 `T > p{front_struct_pct}` 的连通块，限制质心在画面上半部
    （相机装在腕后方，夹爪按安装几何必然在上方；每集都会被面积自检复核），
    取第 `rank` 大的那块当种子；提示帧取该块内 `|frame - 中值|` 最小的那一帧。

    注意：光靠"低 std"拿不到这个 —— 夹爪在 front 画面里自己也在动，std 不低。
    这条判据靠的是"中值图里还有结构"，不是"没动"。

    `rank` 是给重试阶梯用的：ep5 front 上"上半部结构块"有两个，
    最大的是 5197 px（列 50–176）、第二是 3932 px（列 289–396），
    **夹爪是第二块** —— 取最大就取了错的那块，`rank=1` 能拿回来。
    默认 `rank=0`（原来的行为）保证既有 27 集的掩码不受影响。
    """
    arr = frames.astype(np.float32)
    med = np.median(arr, 0)
    std_map = arr.std(0).mean(0)
    gm = cv2.GaussianBlur(gradmag(med.mean(0)), (0, 0), cfg.front_blur_sigma)

    from scipy import ndimage

    m = gm > np.percentile(gm, cfg.front_struct_pct)
    m = ndimage.binary_closing(m, np.ones((9, 9)))
    m = ndimage.binary_opening(m, np.ones((5, 5)))
    lab, n = ndimage.label(m)
    seed = np.zeros_like(m)
    H = m.shape[0]
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        cand = []
        for i in range(n):
            if sizes[i] < cfg.front_min_area_frac * m.size:
                continue
            cy = np.where(lab == i + 1)[0].mean()
            if cy <= cfg.front_top_frac * H:
                cand.append((sizes[i], i + 1))
        if cand:
            # 按面积从大到小，取第 rank 块（rank 越界就退回最大那块）
            cand.sort(key=lambda c: -c[0])
            seed = lab == cand[min(rank, len(cand) - 1)][1]

    if not seed.any():                      # 退化：整幅图当种子，交给 SAM2 自己找
        box = [0, 0, m.shape[1] - 1, int(cfg.front_top_frac * H)]
        best = int(np.argmin([np.abs(arr[t] - med).mean() for t in range(len(arr))]))
        return {"camera": "front", "frame": best, "box": box, "tier": "fallback-full-width",
                "seed_frac": float(seed.mean()), "std_med": float(np.median(std_map))}

    # 提示帧：夹爪最贴合中值位置的那一帧
    cost = [float(np.abs(arr[t] - med)[:, seed].mean()) for t in range(len(arr))]
    best = int(np.argmin(cost))
    box = _bbox_of(seed, cfg.box_pad, m.shape)
    return {"camera": "front", "frame": best, "box": box, "tier": "median-structure",
            "seed_frac": float(seed.mean()), "std_med": float(np.median(std_map))}


def _pad_box(box, pad: int, shape) -> list[int]:
    """把提示框四边各外扩 `pad` 像素，夹到画面内。"""
    h, w = shape
    x0, y0, x1, y1 = (int(v) for v in box)
    return [max(0, x0 - pad), max(0, y0 - pad),
            min(w - 1, x1 + pad), min(h - 1, y1 + pad)]


def prompt_candidates(frames: np.ndarray, camera_key: str, cfg: RobotMaskConfig,
                      retries: int = 0) -> list[dict]:
    """按「最该成功的提示 → 最宽松的兜底」排出一列候选提示。

    每一档都过 `_self_check` 里**同一个**自检，阶梯只多试、不降标准 ——
    所以全不过时的结果和没有阶梯完全一样，多花的是 GPU 时间。

    为什么要阶梯：自检挡下的那 5 集（ep5 front、ep9/10/11 up、ep10 front）
    不是"掩码质量差一点"，是**提示本身选错了目标**（ep5 front 取到的是
    上半部最大那块 5197 px，夹爪其实是第二块 3932 px）。这种错用同一份提示
    重跑一百次也是同一个结果，必须换提示。而 up 那几例的框底永远贴着
    画面下缘（y1=479）—— 框把机械臂裁掉了，所以第二档就把框放大。

    `retries=0` 时只返回第 0 档，行为与没有阶梯时逐位相同。

    代价（实测 ep3 front，T=465）：`front_prompt` 一次约 9.5-18 s，这里**提前**把
    rank=1 也算出来（不管用不用得上）→ 每集 front 多花约 9.5 s，50 集约 8 min；
    而一次 SAM2 传播约 1 min。所以没有为它做缓存 —— 那份复杂度换不回这 8 分钟。
    up 一侧几乎免费（`up_prompt` 0.4 s）。真嫌慢的话，把 `front_prompt` 拆成
    "算块列表"和"按 rank 取一块"两半，块列表只算一次即可。
    """
    is_up = camera_key.endswith(".up")
    base = up_prompt(frames, cfg) if is_up else front_prompt(frames, cfg, rank=0)
    h, w = frames.shape[-2:]
    cands = [base]
    if is_up:
        cands.append({**base, "box": _pad_box(base["box"], cfg.retry_box_pad, (h, w)),
                      "tier": f"{base['tier']}+pad{cfg.retry_box_pad}"})
        # 最后一档：整幅图，全凭 SAM2 自己的先验。可能把桌面一起吞了，
        # 但这一档照样要过面积**上限**自检，吞了就是不落盘。
        cands.append({**base, "box": [0, 0, w - 1, h - 1], "tier": "fallback-full-frame"})
    else:
        cands.append(front_prompt(frames, cfg, rank=1))
        cands.append({**base, "box": _pad_box(base["box"], cfg.retry_box_pad, (h, w)),
                      "tier": f"{base['tier']}+pad{cfg.retry_box_pad}"})
        cands.append({**base, "box": [0, 0, w - 1, int(cfg.front_top_frac * h)],
                      "tier": "fallback-full-width"})

    # 去重：同一帧、同一个框，跑两遍 SAM2 是纯浪费
    # （front 上只有一个上半部块时 rank=0 和 rank=1 就是同一个提示）。
    uniq, seen = [], set()
    for c in cands:
        key = (int(c["frame"]), tuple(int(v) for v in c["box"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq[:1 + retries]


# ---------------------------------------------------------------------------
# SAM2 传播
# ---------------------------------------------------------------------------
def sam2_propagate(frames: np.ndarray, prompt: dict, cfg: RobotMaskConfig,
                   device: str = "cuda") -> np.ndarray:
    """按提示跑 SAM2 视频传播，返回 (T, H, W) bool。"""
    import sys

    for p in (str(SAM2_ROOT), str(REPO / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from sam2.build_sam import build_sam2_video_predictor

    T = len(frames)
    masks = np.zeros((T,) + frames.shape[2:], dtype=bool)
    with tempfile.TemporaryDirectory(prefix="robotmask_") as td:
        for k in range(T):
            cv2.imwrite(os.path.join(td, f"{k:05d}.jpg"),
                        cv2.cvtColor(_to_hwc_uint8(frames[k]), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        pred = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)
        st = pred.init_state(video_path=td, offload_video_to_cpu=True,
                             offload_state_to_cpu=True)
        try:
            pred.add_new_points_or_box(inference_state=st, frame_idx=int(prompt["frame"]),
                                       obj_id=0, box=np.array(prompt["box"], dtype=np.float32))
            for rev in (True, False):
                for fr, _oid, logits in pred.propagate_in_video(st, reverse=rev):
                    masks[fr] = (logits[0] > 0.0).cpu().numpy().squeeze(0)
        finally:
            pred.reset_state(st)
            del pred
            import torch

            torch.cuda.empty_cache()
    return masks


# ---------------------------------------------------------------------------
# 对外主入口
# ---------------------------------------------------------------------------
def _self_check(m: np.ndarray, camera_key: str, cfg: RobotMaskConfig, prompt: dict) -> dict:
    """给一份掩码跑自检，返回含 `problems` / `ok` 的报告。

    抽出来是为了让重试阶梯的**每一档都走这里** —— 阶梯只多试提示，不放松标准。
    """
    T = m.shape[0]
    area = m.reshape(T, -1).mean(1)
    report = {"camera": camera_key, "prompt": prompt,
              "area_median": float(np.median(area)), "area_min": float(area.min()),
              "area_max": float(area.max()),
              "valid_frac": float((area > 0).mean())}
    problems = []
    if report["area_median"] < cfg.min_area_frac:
        problems.append(f"掩码中位面积 {report['area_median']:.1%} < 下限 {cfg.min_area_frac:.1%}")
    if report["area_median"] > cfg.max_area_frac:
        problems.append(f"掩码中位面积 {report['area_median']:.1%} > 上限 {cfg.max_area_frac:.1%}")
    if report["valid_frac"] < cfg.min_valid_frac:
        problems.append(f"只有 {report['valid_frac']:.1%} 的帧有掩码 < {cfg.min_valid_frac:.1%}")
    report["problems"] = problems
    report["ok"] = not problems
    return report


def build_robot_mask(frames: np.ndarray, camera_key: str, cfg: RobotMaskConfig,
                     device: str = "cuda", override: dict | None = None,
                     retries: int = 0) -> tuple[np.ndarray, dict]:
    """返回 (掩码 uint8 0/255 (T,H,W), 报告 dict)。提示与自检结果都在报告里。

    `retries > 0` 打开重试阶梯：第 0 档提示自检不过，就换下一档提示（见
    `prompt_candidates`）重跑 SAM2，直到某一档过自检。**自检标准全程不变**，
    所以最坏情况（全不过）和 `retries=0` 是同一个结论，多花的只是 GPU 时间；
    报告里 `attempts` 逐档记下结果，`n_attempts` 是实际跑了几档。

    `override` 给的手动框**不参与阶梯**：那是人明确指定的目标，
    自检不过就该报出来让人看，而不是偷偷换成别的框。
    """
    if override and override.get("box"):
        prompts = [{"camera": camera_key.split(".")[-1],
                    "box": [int(v) for v in override["box"]],
                    "frame": int(override["frame"]) if override.get("frame") is not None
                    else len(frames) // 2,
                    "tier": "manual", "seed_frac": None}]
    else:
        prompts = prompt_candidates(frames, camera_key, cfg, retries=retries)

    attempts, reports, masks = [], [], []
    for i, prompt in enumerate(prompts):
        m = sam2_propagate(frames, prompt, cfg, device=device)
        r = _self_check(m, camera_key, cfg, prompt)
        reports.append(r)
        masks.append(m)
        attempts.append({"prompt": prompt, "area_median": r["area_median"],
                         "valid_frac": r["valid_frac"],
                         "problems": r["problems"], "ok": r["ok"]})
        if r["ok"]:
            break
        if i + 1 < len(prompts):
            print(f"       [retry] 第 {i} 档不过（{'; '.join(r['problems'])}），"
                  f"换第 {i + 1} 档：{prompts[i + 1]['tier']}")

    # 全都没过时返回**第 0 档**的掩码和报告，不是最后那档：后面的档一档比一档宽松
    # （末尾是整幅图），`--force` 拿到它会得到一份"整幅图都是前景、什么都不替换"的
    # 废掩码，看着像成功。返回第 0 档还有个好处 —— 阶梯开着但全不过时，
    # `--force` 的行为和没有阶梯时逐位相同；这次试过哪几档记在 `attempts` 里。
    k = next((i for i, r in enumerate(reports) if r["ok"]), 0)
    report = reports[k]
    report["attempts"] = attempts
    report["n_attempts"] = len(attempts)
    report["retried"] = len(attempts) > 1
    return (masks[k].astype(np.uint8) * 255), report


def episode_robot_mask_path(root: str | Path, camera_key: str, episode_index: int) -> Path:
    return Path(root) / camera_key / f"episode_{episode_index:06d}.npy"


def save_robot_mask(root: str | Path, camera_key: str, episode_index: int,
                    mask: np.ndarray) -> Path:
    path = episode_robot_mask_path(root, camera_key, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 名字必须以 .npy 结尾：np.save 对其它结尾会**自动补 .npy**，
    # 于是 os.replace 找不到刚写的那个临时文件（踩过一次）。
    tmp = path.parent / (path.name + ".tmp.npy")
    np.save(tmp, np.asarray(mask).astype(np.uint8))
    os.replace(tmp, path)
    return path


def load_robot_mask(root: str | Path, camera_key: str, episode_index: int,
                    dilate_px: int = 0) -> np.ndarray | None:
    """读掩码。`dilate_px > 0` 时向外扩一圈，用来吃掉 SAM2 漏掉的机械臂残部。

    为什么要这个：实测（ep0，把「背景」池按距机械臂/任务物的距离分层消融）——
    front 的背景 Δ 在排除半径 0/8/16/24/40 px 下是
    **2.33 / 2.38 / 2.01 / 1.54 / 1.19**，一半的效应来自机械臂周围 24-40 px 内。
    SAM2 在 front 上稳定跟住的是**刚体夹爪**，腕部连杆和线缆没进去，
    它们就留在"背景"池里 —— 噪声打上去等于毁掉机械臂。

    方向是**宁可把背景误当机器人**：漏掉残部的代价（把机械臂消融掉，得出
    "背景很重要"的假结论）远大于多盖一圈桌面的代价。多盖了多少会写进报告。
    """
    path = episode_robot_mask_path(root, camera_key, episode_index)
    if not path.exists():
        return None
    m = np.load(path).astype(np.uint8)
    if dilate_px > 0:
        from scipy import ndimage

        m = (ndimage.binary_dilation(m > 0, structure=np.ones((1, 3, 3), bool),
                                     iterations=int(dilate_px)) * 255).astype(np.uint8)
    return m
