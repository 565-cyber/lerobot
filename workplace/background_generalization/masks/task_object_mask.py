"""任务物掩码：黄方块（要抓的）和红桶（要放进去的）。

任务串是 `Pick up yellow block and put it in trash can`（实测自数据集
`meta`）。做 action saliency 时「背景」必须把**任务相关**的东西排除掉：
把黄方块或红桶换成噪声，policy 当然会反应，但那是"任务物重要"，不是
"背景重要"。混进去会把结论往"背景有用"的方向推。

所以前景 = 机械臂/夹爪（SAM2，见 `robot_mask.py`）∪ 黄方块 ∪ 红桶；
背景 = 其余像素（桌面、墙、绿垫、线缆、电源砖……），那才是环境泛化
真正关心的东西。

## 蓝：从「背景」挪到「前景」（2026-09-22 改）

原来蓝（蓝方块、蓝纸盒）是算在背景里的。改的原因是用途变了：这份掩码现在还要
当 RoboSaGA 式背景替换的 **keep 掩码**用，而 `background_generalization/CLAUDE.md`
的安全红线第一条点名了**蓝方块**——它落在背景里就会被替换掉，直接破线。

实测（newdata1 ep0/ep1，每集 12 帧）最大蓝块占画面：up 0.49%（蓝方块）、
front 9.2-19.0%（蓝纸盒）。**每帧都有**，没有空帧——所以这不是偶发误检，
是画面里一直在的物体。阈值直接搬 Stage 2A 追蓝方块那一组，没有重新标。

方向说明：这个改动让 action saliency 的**背景池少了一块**（蓝纸盒不再算环境）。
对那个实验是保守方向——少测一点环境，好过把"蓝纸盒重要"混进"背景重要"。
`scripts/action_saliency.py` 只点名读 `yellow`/`red`，不遍历 `values()`，
所以这个改动**不会**静默改变它的结果（复核过调用点）。

用 HSV 阈值而不是再跑一遍 SAM2：这两个物体是**固定颜色**的刚体，颜色比
实例分割稳，也不需要提示。

## 实测的边界：黄色分不干净，红色分得干净

复核图（ep0/ep1，每集 4 帧）显示**红色两路都对**：up 里 13.5% 就是俯视的红桶，
front 里最大 22% 就是近距掠视的红桶内壁。红色没问题。

**黄色在 front 里被同色纸盒污染**（橄榄/橙色纸箱）。实测各连通块的中位 HSV：

| 物体 | H | S | V | 面积(占画面) |
|---|---|---|---|---|
| 黄方块（up，真值） | 23–28 | 100–133 | 139–248 | 0.05–0.77% |
| 黄方块（front，近距） | 23–25 | 100–135 | 139–194 | 1.0–1.9% |
| 橄榄纸盒（front） | 25–30 | 111–168 | 114–255 | 3.5–22.1% |

**H/S/V 完全重叠，没有任何色阈值能把方块和纸盒分开。** 这不是阈值没调好，
是两件东西在同一盏暖光下就是同一个颜色。唯一可分的量是**尺寸** ——
所以加了逐相机的面积窗口 `max_area_frac`：超过上限的连通块被丢掉，
丢掉的总面积记进 `report`（不静默），复核时能看见"这次丢了多少"。

注意这是**保守方向**：宁可把纸盒误当方块（把环境从背景池里挪走），
也不要让纸盒留在背景池里 —— 纸盒又大又显眼，留在池子里会把
"背景重要"的结论往上推。两害相权，取前者，并且在报告里写明。

面积窗口按相机给：红桶在 up 里是俯视的大矩形，在 front 里是近距的一块，
差很多；黄方块在两路里也差一个量级（俯视 0.05%、近距可达 1.9%）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# OpenCV 的 H 是 0-179
YELLOW_LO = (20, 90, 90)
YELLOW_HI = (38, 255, 255)
RED_LO_1 = (0, 90, 60)
RED_HI_1 = (10, 255, 255)
RED_LO_2 = (168, 90, 60)
RED_HI_2 = (179, 255, 255)
# 蓝：阈值照搬 Stage 2A 的标定（`object_generalization` 里追蓝方块用的那一组），
# 没有重新标。那边追的是**任务目标**蓝方块，这边要的是**所有蓝**，但阈值可以直接用：
# 越界只会多收，而多收在这个用途里是安全方向（见下）。
BLUE_LO = (95, 70, 50)
BLUE_HI = (140, 255, 255)


# 逐相机的单连通块面积窗口（占画面），(下限, 上限)。
# 上限是**实测**出来的，不是拍的：front 的黄方块 1.0-1.9%、橄榄纸盒 3.5-22.1%，
# 中间是空的，取 0.025 落在空带里。up 的黄方块 ≤0.77%、红桶俯视约 13.5%。
#
# 蓝**故意没有上限**（1.0 = 永不触发 too_big）。两个原因：
#   1. 实测（newdata1 ep0/ep1 每集 12 帧）最大蓝块在 up 是 0.49%（蓝方块），
#      在 front 是 9.2-19.0%（蓝纸盒，近距掠视所以很大）。front 那个是**真物体**，
#      不是阈值失效，用面积把它丢掉就是错的。
#   2. 更要紧的是方向：`_clean` 丢掉超尺寸块 = 这块**不进 keep 掩码** = 做背景替换时
#      **被替换掉**。那是安全红线的反面。所以这里宁可永不丢，把「蓝是不是太大了」
#      交给 `task_object_report` 去**报**（`max_area_frac`），而不是静默地丢。
AREA_WINDOW = {
    "up":    {"yellow": (0.0002, 0.020), "red": (0.0002, 0.30), "blue": (0.0002, 1.0)},
    "front": {"yellow": (0.0002, 0.025), "red": (0.0002, 0.35), "blue": (0.0002, 1.0)},
}
DEFAULT_WINDOW = (0.0002, 0.45)


@dataclass
class TaskObjectConfig:
    yellow_lo: tuple = YELLOW_LO
    yellow_hi: tuple = YELLOW_HI
    red_lo_1: tuple = RED_LO_1
    red_hi_1: tuple = RED_HI_1
    red_lo_2: tuple = RED_LO_2
    red_hi_2: tuple = RED_HI_2
    blue_lo: tuple = BLUE_LO
    blue_hi: tuple = BLUE_HI

    open_px: int = 3               # 去掉孤立噪点
    close_px: int = 7              # 补上镜面高光造成的小洞
    min_area_frac: float = 0.0004  # 碎块下限（占画面）
    max_area_frac: float = 0.45    # 超过就是"整幅都算上了"，判定失败
    area_window: dict = field(default_factory=lambda: dict(AREA_WINDOW))

    def window(self, kind: str, camera_key: str | None) -> tuple[float, float]:
        cam = (camera_key or "").split(".")[-1]
        return self.area_window.get(cam, {}).get(kind, DEFAULT_WINDOW)

    def as_dict(self) -> dict:
        return {"yellow": [self.yellow_lo, self.yellow_hi],
                "red": [self.red_lo_1, self.red_hi_1, self.red_lo_2, self.red_hi_2],
                "blue": [self.blue_lo, self.blue_hi],
                "open_px": self.open_px, "close_px": self.close_px,
                "min_area_frac": self.min_area_frac, "max_area_frac": self.max_area_frac,
                "area_window": self.area_window}


def _clean(m: np.ndarray, cfg: TaskObjectConfig, kind: str,
           camera_key: str | None) -> tuple[np.ndarray, dict]:
    """按面积窗口筛连通块。返回 (掩码, 丢弃统计)。丢弃量会进报告，不静默。"""
    from scipy import ndimage

    if cfg.close_px:
        m = ndimage.binary_closing(m, np.ones((cfg.close_px, cfg.close_px)))
    if cfg.open_px:
        m = ndimage.binary_opening(m, np.ones((cfg.open_px, cfg.open_px)))
    lo, hi = cfg.window(kind, camera_key)
    lab, n = ndimage.label(m)
    drop = {"too_small": 0, "too_big": 0, "biggest": 0.0, "dropped_frac": 0.0}
    if n == 0:
        return m, drop
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    total = m.size
    keep = []
    for i in range(n):
        f = sizes[i] / total
        if f < max(lo, cfg.min_area_frac):
            drop["too_small"] += int(sizes[i])
        elif f > hi:
            drop["too_big"] += int(sizes[i])
            drop["biggest"] = max(drop["biggest"], float(f))
        else:
            keep.append(i + 1)
    out = np.isin(lab, keep) if keep else np.zeros_like(m)
    drop["dropped_frac"] = (drop["too_small"] + drop["too_big"]) / total
    return out, drop


def task_object_masks(rgb_hwc_uint8: np.ndarray, cfg: TaskObjectConfig | None = None,
                      camera_key: str | None = None) -> dict[str, np.ndarray]:
    """返回 {"yellow": bool(H,W), "red": bool(H,W)}。

    `camera_key` 决定用哪一档面积窗口（`AREA_WINDOW`）。不传就用最宽的一档，
    只适合快速看一眼 —— 逐帧跑请传，否则 front 上橄榄纸盒会混进黄色掩码。

    先转 BGR 再转 HSV：调用方给的是 RGB（数据集里就是这个顺序），
    cv2 默认按 BGR 解释，顺序搞反会得到零个红色像素 —— 看着像"红的漏了"。
    """
    cfg = cfg or TaskObjectConfig()
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb_hwc_uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    y = cv2.inRange(hsv, np.array(cfg.yellow_lo, np.uint8), np.array(cfg.yellow_hi, np.uint8)) > 0
    r = (cv2.inRange(hsv, np.array(cfg.red_lo_1, np.uint8), np.array(cfg.red_hi_1, np.uint8)) > 0) | \
        (cv2.inRange(hsv, np.array(cfg.red_lo_2, np.uint8), np.array(cfg.red_hi_2, np.uint8)) > 0)
    b = cv2.inRange(hsv, np.array(cfg.blue_lo, np.uint8), np.array(cfg.blue_hi, np.uint8)) > 0
    ym, _ = _clean(y, cfg, "yellow", camera_key)
    rm, _ = _clean(r, cfg, "red", camera_key)
    bm, _ = _clean(b, cfg, "blue", camera_key)
    return {"yellow": ym, "red": rm, "blue": bm}


def task_object_masks_detailed(rgb_hwc_uint8: np.ndarray, cfg: TaskObjectConfig | None = None,
                               camera_key: str | None = None) -> dict:
    """同上，但把每个通道丢掉了多少连通块一起返回（复核用）。"""
    cfg = cfg or TaskObjectConfig()
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb_hwc_uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raw = {
        "yellow": cv2.inRange(hsv, np.array(cfg.yellow_lo, np.uint8),
                              np.array(cfg.yellow_hi, np.uint8)) > 0,
        "red": (cv2.inRange(hsv, np.array(cfg.red_lo_1, np.uint8),
                            np.array(cfg.red_hi_1, np.uint8)) > 0) |
               (cv2.inRange(hsv, np.array(cfg.red_lo_2, np.uint8),
                            np.array(cfg.red_hi_2, np.uint8)) > 0),
        "blue": cv2.inRange(hsv, np.array(cfg.blue_lo, np.uint8),
                            np.array(cfg.blue_hi, np.uint8)) > 0,
    }
    out = {}
    for k, m in raw.items():
        kept, drop = _clean(m, cfg, k, camera_key)
        out[k] = {"mask": kept, "window": cfg.window(k, camera_key), **drop}
    return out


def task_object_report(masks: dict[str, np.ndarray], cfg: TaskObjectConfig,
                       dropped: dict | None = None) -> dict:
    """`dropped` 是**已经算成比例**的汇总（{kind: {dropped_frac, dropped_too_big_frac,
    biggest_dropped_frac}}），按帧汇总由调用方负责 —— 这里不再做像素/比例换算，
    免得两种量纲在同一个 dict 里混着（踩过一次）。"""
    out, problems = {}, []
    for k, m in masks.items():
        f = float(m.mean())
        out[k] = {"area_frac": f}
        if f > cfg.max_area_frac:
            problems.append(f"{k} 占画面 {f:.1%} > 上限 {cfg.max_area_frac:.1%}，阈值大概率失效")
        if dropped and k in dropped:
            out[k].update(dropped[k])
    out["problems"] = problems
    out["ok"] = not problems
    return out
