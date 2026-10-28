"""前景（keep）掩码 = 机械臂/夹爪（SAM2） ∪ 任务物（黄方块 / 红桶 / 蓝）。

给 **RoboSaGA 式背景替换**当 `mask` 用（`transforms/background_replacement.py`，
约定 1 = 保留、0 = 替换），也给它当安全闸门用。存储走 `mask_utils.py` 的约定：

    {root}/{camera_key}/episode_{ep:06d}.npy    uint8 (T, H, W)，0 / 1

**存 0/1 而不是 0/255。** `apply_background_replacement._to_01` 靠 `max > 1.5`
判断要不要除 255，0/1 的 max 恰好是 1，不会触发那条分支 —— 两种格式它都能吃，
但 0/1 少一次除法、也少一个"这到底是哪种量纲"的疑问。`robot_mask.py` 存的是
0/255，那是它自己的格式，别混。

## 为什么是这三样并起来，而不是"时域低 std"

`action_saliency.py` 那套「整集里不大动的像元 = 背景」在 front 上前提不成立
（相机随臂，见 `robot_mask.py` 的模块文档，有实测表），所以机械臂必须是
**显式分割**出来的，不能靠统计量。任务物（黄/红/蓝）是固定颜色的刚体，
HSV 比实例分割稳，也不需要提示 —— 两条路各取所长。

## 为什么外扩，而且外扩得挺大

`load_robot_mask` 的 docstring 记了实测：front 上把背景池按"距机械臂/任务物的
距离"分层消融，Δ 在排除半径 0/8/16/24/40 px 下是 2.33 / 2.38 / 2.01 / 1.54 / 1.19
—— **一半的效应来自机械臂周围 24–40 px 内**。SAM2 在 front 上稳定跟住的是刚体
夹爪，腕部连杆和线缆没进去，它们留在"背景"里。

背景替换比消融更狠：消融是"这块别算了"，替换是"这块换成别的东西"。把机械臂
残部替换掉 = 毁掉机械臂。所以外扩的方向是**宁可把背景误当机器人**，代价是多盖
一圈桌面、增强幅度小一点 —— 正对本目录红线第三条「宁可漏增强，不要错增强」。
多盖了多少会写进报告，不静默。

## 红线的落点

`CLAUDE.md` 的安全红线第一条点名：机械臂、夹爪、**蓝方块**、红盒、桌面
不可重着色/移动。这份掩码覆盖前四样里的前四样（机械臂、夹爪、蓝、红盒），
所以它落在"保留"那一侧。**桌面不在掩码里** —— 桌面本来就是要被替换的背景，
这正是本阶段的目的，红线约束的是"重着色"这个动作本身，见 README 里的讨论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mask_utils import load_episode_mask, save_episode_mask
from .robot_mask import _to_hwc_uint8, load_robot_mask
from .task_object_mask import TaskObjectConfig, task_object_masks_detailed

# 这一路要保留的物体。加/减这个集合会直接改变「什么会被替换掉」，
# 所以它是显式的、写在这里，而不是散在调用点。
KEEP_KINDS: tuple[str, ...] = ("yellow", "red", "blue")


@dataclass
class ForegroundMaskConfig:
    """所有阈值在这里，函数里不放魔数。"""

    # 机械臂掩码缓存的根（`build_robot_mask_cache.py --out` 的那个）。
    robot_root: str | None = None

    # 并集之后整体外扩的半径（px）。24 是 `action_saliency.py` 的默认值，
    # 由上面那段分层消融的实测支持。
    dilate_px: int = 24

    # 任务物掩码单独的小外扩（px）。HSV 阈值在物体边缘会因为抗锯齿/高光
    # 漏掉一圈，那一圈颜色介于物体和桌面之间 —— 不补上就会在合成图上留下一
    # 道原色的细边。3 px 是够盖住抗锯齿带的最小值，不再往上堆，因为任务物
    # 边界本来就准（固定颜色的刚体）。
    task_dilate_px: int = 3

    task_cfg: TaskObjectConfig = field(default_factory=TaskObjectConfig)

    # 允许哪几类进入 keep 集合。空 = 只保留机械臂。
    keep_kinds: tuple[str, ...] = KEEP_KINDS

    # ---- 自检 ----
    min_valid_frac: float = 0.90   # 至少有这么大比例的帧要有非空掩码
    max_area_frac: float = 0.90    # 超过就说明并集把整幅画面吞了
    min_area_frac: float = 0.015   # 下限：低于它说明哪一路都没接上

    def as_dict(self) -> dict:
        return {
            "robot_root": str(self.robot_root) if self.robot_root else None,
            "dilate_px": self.dilate_px,
            "task_dilate_px": self.task_dilate_px,
            "keep_kinds": list(self.keep_kinds),
            "task_cfg": self.task_cfg.as_dict(),
            "min_valid_frac": self.min_valid_frac,
            "max_area_frac": self.max_area_frac,
            "min_area_frac": self.min_area_frac,
        }


def _grow(mask: np.ndarray, radius: int) -> np.ndarray:
    """对 (T, H, W) 的 bool 逐帧外扩，半径 `radius` 的**圆盘**。

    用距离变换，不用 `binary_dilation`。对**非空**帧两者逐位等价
    （`edt(~mask) <= r` 就是「用半径 r 的圆盘做二值膨胀」），但代价差一个量级：
    `binary_dilation` 配一个 (2r+1)² 的结构元是 O(n·k²)，r=24 时结构元 49×49，
    实测 **0.21 s/帧** —— 一集 531 帧要 111 s，50 集 × 2 相机就是约 2.5 小时，
    纯浪费。EDT 是 O(n log n)，实测 **0.013 s/帧**（16x）。

    ## 空帧必须单独挡掉，否则 EDT 会**凭空造出前景**

    全 False 的输入里 `~mask` 没有零点，而 scipy 的 EDT 在"一个零点都没有"时
    把**数组外面**当成背景，于是每个像素的距离变成"到画面边界的距离"，
    `<= r` 就返回画面**左上角一个半径 r 的圆盘象限** —— 一块不存在的前景。

    这不是理论：`task_dilate_px=3` 先在一帧空的任务物掩码上造出一个 7 px 的
    角落种子，并进 union，`dilate_px=24` 再把它长成 **556 px**。实测 up ep0
    有 **164 帧**至少一种任务物为空（黄方块常被夹爪挡住），于是左上角 27×27
    被判成"要保留"，合成时那块真背景**不替换** —— 增强里留下一小块原背景。
    `binary_dilation` 对全 False 返回全 False，在这一档上是对的。

    我最初宣称两者等价时只测了**非空**真实掩码（40 帧一致），所以没看见它。
    要复现这个坑，必须拿一帧全空的掩码去测。

    顺带一提，EDT 给的是**欧氏**距离，所以外扩出来的天然是圆盘而不是方块 ——
    拿 `np.ones((k,k))` 当结构元会往角上多盖 ~sqrt(2) 倍面积，这里不存在这个问题。
    """
    if radius <= 0:
        return mask
    from scipy import ndimage

    out = np.empty_like(mask, dtype=bool)
    for t in range(mask.shape[0]):
        f = mask[t]
        if not f.any():
            out[t] = False
            continue
        out[t] = ndimage.distance_transform_edt(~f) <= radius
    return out


def _check_no_phantom(grown: np.ndarray, src: np.ndarray, what: str) -> None:
    """不变量：**外扩不能把一整帧空掩码长出东西来**。

    这条断言的价值在于它**会对错误实现失败** —— `_grow` 最初那版（直接 EDT、
    不挡空帧）在 up ep0 上触发它 **164 帧**、每帧 556 px。所以它不是
    "测自己的定义"那种恒真断言：将来谁再动 `_grow`，这里会当场报错，
    而不是让一块假前景悄悄进数据集（合成时那块真背景不替换，等于增强里
    留了一小片原背景 —— 正是红线第三条要挡的方向）。

    空帧变非空是**结构性错误**，不是统计异常，所以 raise 而不是记进 report。
    """
    gained = (grown.reshape(len(grown), -1).any(1)
              & ~src.reshape(len(src), -1).any(1))
    if gained.any():
        idx = np.nonzero(gained)[0]
        raise ValueError(
            f"{what}：有 {int(gained.sum())} 帧本来是**空**掩码，外扩之后却有了前景"
            f"（首帧 t={int(idx[0])}，多出 {int(grown[idx[0]].sum())} px）。"
            f"外扩只能长大、不能无中生有 —— 见 `_grow` 的 docstring。"
        )


def build_foreground_mask(
    frames: np.ndarray,
    camera_key: str,
    episode_index: int,
    cfg: ForegroundMaskConfig,
) -> tuple[np.ndarray, dict]:
    """合成一集一相机的前景掩码。

    Args:
        frames: (T, 3, H, W)，float[0,1] 的 torch tensor 或 ndarray（数据集里的原样）。
        camera_key: 例如 `observation.images.front`。决定用哪一档面积窗口。
        episode_index: 用来找机械臂掩码缓存。
        cfg: 见 `ForegroundMaskConfig`。

    Returns:
        (mask, report)。mask 是 uint8 0/1 (T, H, W)，1 = 保留。
        report["ok"] 为 False 时**不要**拿这份掩码去合成 —— 调用方应当跳过或报错。
    """
    arr = frames.numpy() if hasattr(frames, "numpy") else np.asarray(frames)
    T = len(arr)
    H, W = arr.shape[-2], arr.shape[-1]

    # ---- 机械臂：读缓存，缺了就报错而不是静默降级 ----
    robot = None
    robot_problem = None
    if cfg.robot_root is None:
        robot_problem = "没有给 robot_root，机械臂这一路整个没接上"
    else:
        robot = load_robot_mask(cfg.robot_root, camera_key, episode_index,
                                dilate_px=0)  # 外扩统一在并集之后做，这里不做
        if robot is None:
            robot_problem = (f"找不到机械臂掩码 {cfg.robot_root}/{camera_key}/"
                             f"episode_{episode_index:06d}.npy，"
                             f"先跑 build_robot_mask_cache.py")
    if robot is not None:
        robot_b = robot > 0
        if len(robot_b) != T:
            raise ValueError(f"机械臂掩码 T={len(robot_b)} 和 frames T={T} 对不上")
    else:
        robot_b = np.zeros((T, H, W), bool)

    # ---- 任务物：逐帧 HSV ----
    kinds = tuple(cfg.keep_kinds)
    obj_b = {k: np.zeros((T, H, W), bool) for k in kinds}
    drops = {k: {"dropped_frac": [], "biggest_dropped_frac": []} for k in kinds}
    for t in range(T):
        det = task_object_masks_detailed(_to_hwc_uint8(arr[t]), cfg.task_cfg,
                                         camera_key=camera_key)
        for k in kinds:
            obj_b[k][t] = det[k]["mask"]
            drops[k]["dropped_frac"].append(det[k]["dropped_frac"])
            drops[k]["biggest_dropped_frac"].append(det[k]["biggest"])

    if cfg.task_dilate_px:
        for k in kinds:
            grown = _grow(obj_b[k], cfg.task_dilate_px)
            _check_no_phantom(grown, obj_b[k], f"任务物 {k} 外扩 {cfg.task_dilate_px}px")
            obj_b[k] = grown

    union = robot_b.copy()
    for k in kinds:
        union |= obj_b[k]
    pre = union.copy()
    union = _grow(union, cfg.dilate_px)
    _check_no_phantom(union, pre, "union 外扩")

    # ---- 报告：三个分量各自的贡献，以及外扩"多盖"了多少 ----
    before_grow = robot_b | np.logical_or.reduce([obj_b[k] for k in kinds]) if kinds else robot_b
    comp = {k: float(v.mean()) for k, v in obj_b.items()}
    comp["robot"] = float(robot_b.mean())
    area = union.reshape(T, -1).mean(1)
    report = {
        "camera": camera_key,
        "episode": int(episode_index),
        "T": int(T),
        "area_median": float(np.median(area)),
        "area_min": float(area.min()),
        "area_max": float(area.max()),
        "valid_frac": float((area > 0).mean()),
        "components": comp,
        "grow_overhead": float(union.mean() - before_grow.mean()),
        "dropped": {k: {"dropped_frac": float(np.mean(drops[k]["dropped_frac"])),
                        "biggest_dropped_frac": float(np.max(drops[k]["biggest_dropped_frac"]))}
                    for k in kinds},
        "robot_problem": robot_problem,
    }

    problems = []
    if robot_problem:
        problems.append(robot_problem)
    if report["area_median"] < cfg.min_area_frac:
        problems.append(f"掩码中位面积 {report['area_median']:.1%} < 下限 "
                        f"{cfg.min_area_frac:.1%}（哪一路没接上？）")
    if report["area_median"] > cfg.max_area_frac:
        problems.append(f"掩码中位面积 {report['area_median']:.1%} > 上限 "
                        f"{cfg.max_area_frac:.1%}（并集把画面吞了）")
    if report["valid_frac"] < cfg.min_valid_frac:
        problems.append(f"只有 {report['valid_frac']:.1%} 的帧有掩码，"
                        f"低于 {cfg.min_valid_frac:.1%}")
    report["problems"] = problems
    report["ok"] = not problems
    return union.astype(np.uint8), report


def save_foreground_mask(root: str | Path, camera_key: str, episode_index: int,
                         mask: np.ndarray) -> Path:
    """存 0/1。走 `mask_utils` 的路径约定，所以 `BackgroundAugmentedDataset`
    直接就能读（它调的就是那个 `load_episode_mask`）。"""
    m = np.asarray(mask)
    if m.max() > 1:
        raise ValueError(
            f"前景掩码应当是 0/1，拿到 max={m.max()}。0/255 的掩码请先除以 255。\n"
            f"（更正：这条闸门以前给的理由是「存 0/255 会让 "
            f"apply_background_replacement 把整幅图判成前景、什么都不替换，而且不报错」"
            f"—— **那是错的**。`_to_01` 在 `max > 1.5` 时会自己除以 255，实测两种格式"
            f"它都吃，所以合成那一步不会坏。留这条闸门是为了**磁盘上只有一种格式**："
            f"这份掩码有四个脚本在读，谁都不该先问一句「这个文件是哪种量纲」。"
            f"但别再拿一个不存在的后果去论证它。）")
    return save_episode_mask(root, camera_key, episode_index, m)


def load_foreground_mask(root: str | Path, camera_key: str,
                         episode_index: int) -> np.ndarray | None:
    return load_episode_mask(root, camera_key, episode_index)
