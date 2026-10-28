"""Stage C：把「新物体」的位姿拟合从**解析球**推广到**任意网格**（重建产物）。

## 这条路和球那条路的差别，以及为什么它更贵

球只有 **1 个**自由参数（沿视线的深度），因为球是各向同性的：`place_ball` 把球心放在
观测形心那条视线上，滑到「渲出的剪影半径 = 观测半径」为止，位姿就完了。
网格没有这个退化 —— 形状本身带信息，所以深度、朝向都要**从单个剪影反解**。
本模块做的就是这件事：相似变换 `(R, t, s)` 的剪影拟合。

## 渲染：为什么不用 `MR.rasterize`

`rasterize` 是带 z-buffer 的通用光栅器，用来「把东西画出来给人看」。
拟合一次要渲几百次，而它 **74ms/次**（1940 面 @640×480）—— 而且几乎不随分辨率下降
（160×120 也要 57ms），瓶颈是**逐三角形的 Python 开销**，不是像素。

拟合只需要**剪影**，不需要深度：**闭合网格的剪影 = 所有三角形（含背面）投影的并集**。
这条是集合恒等式（物体整个在相机前方时成立），没有近似。于是可以交给 C++：
逐块 `cv2.fillConvexPoly`（三角形是凸的，合法），**1.6ms/次**，快 46×。

★ 两个坑，都实测过：
  1. **`cv2.fillPoly` 传多边形列表是按奇偶规则跨多边形填充的**，重叠处互相抵消。
     12 面方盒一次传 → 2439px，逐块传 → 78317px（并集）。网格越粗崩得越狠
     （原网格 232k 面侥幸看着对：IoU 0.987）。所以**必须逐块**。
  2. `cv2.fillConvexPoly` 与 `MR.rasterize` 的边界约定不同（前者含边界像素，
     后者按**像素中心**在内侧才算）。实测重网格 IoU 稳定在 **0.986~0.988**，且这个
     差**不随减面程度变化**（1940/7132/16036 面都是 0.987）⇒ 它是约定差、不是几何误差。
     拟合对的是**观测掩码**（颜色阈值出来的，有它自己的边界约定），这点差无所谓；
     但做**自检**时必须用同一个渲染器渲地面真值，否则测的是约定差。

## 拟合的规范自由，如实报

单张剪影对**沿光线方向的分辨率**天然弱：深度只能靠**已知公制尺度**反解，
`σ_z ≈ σ_px · z² / (f · s)`。舵机在 z=250mm、f=409.6、s=1、σ_px=0.3 时 ≈1.0mm。
这个公式在 `selftest` 里当**可证伪的直线**去验，不是背书。
平面内滚转只靠轮廓不对称性可观测，近对称时退化 —— 同样报出来。

用法见文件末尾 `main()`。
"""
import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
# ★ 不再 import `scipy.optimize.minimize`：Powell 在这个阶梯状覆盖率目标上**不下降**
#   （实测从真值出发损失反而 0.0432 → 0.2535）。局部优化一律走 `pattern_search`。

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures"
RECON = HERE / "calib" / "recon"

# 配色沿用 `validate_carve.py` 里那份（dataviz 的**已验证默认实例**，槽位按固定顺序取用、
# 不循环）。本机没有 node 跑不了那个校验脚本，所以只取原始值、不改槽位。
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK1, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURF, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({
        "font.family": [f.name for f in font_manager.fontManager.ttflist
                        if f.name == "Noto Sans CJK JP"] or ["DejaVu Sans"],
        "axes.unicode_minus": False, "figure.dpi": 130,
        "savefig.facecolor": SURF, "axes.facecolor": SURF, "figure.facecolor": SURF,
        "axes.edgecolor": AXIS, "text.color": INK1,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK2, "font.size": 9.5,
    })
    return plt


def _ax_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, name, note):
    """存图。★ 出图后**必须肉眼回看**（这是仓库规矩）：`tight_layout` 只用来看住
    标题和刻度会不会被裁掉 —— 第一次出图时 `suptitle(y=1.02)` 没配 `rect`，
    整行大标题被裁成半行。"""
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(p)
    plt.close(fig)
    print(f"[图] {p}  — {note}")
    if not p.exists():
        raise OSError(f"图没写出来：{p}")
    return p


def _load_module(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


MR = _load_module("_fmp_mr", HERE / "meshrender.py")
BS = _load_module("_fmp_bs", HERE / "ball_swap.py")
A = _load_module("_fmp_arm", HERE / "arm.py")

_ZERO3 = np.zeros(3)


# ================================================================ 剪影渲染（快路）

def silhouette(tris, cam, near=0.005, out=None, idx=None):
    """网格在相机里的**剪影**（bool 掩码）—— 逐块 `fillConvexPoly` 求并集。

    `idx` 可以给定「要画哪些面」的下标（Stage D 里用不到，留给只渲一部分的场合）。
    整块落在近平面之前的面**直接丢掉**（不强裁）：拟合时物体会被摆在相机前方十几厘米，
    铰链上的相机不会和它相交；这一点在 `near` 的取值上留了余量。

    ★ 返回 **bool**（和 `MR.rasterize` 的 `mask` 一致）。别返回 uint8 —— 调用方写
      `xx[mask]` 打算做布尔掩码时，uint8 会被当成**花式索引**，静默算出 0/1 的均值
      （形心变成 (0.01, 0.01)，射线指向画面角落）。这个坑真踩过。
    """
    t = tris if idx is None else tris[idx]
    if len(t) == 0:
        return np.zeros((cam.H, cam.W), bool) if out is None else out > 0
    Pc = (t.reshape(-1, 3) @ cam.T_cam_base[:3, :3].T
          + cam.T_cam_base[:3, 3]).reshape(-1, 3, 3)
    ok = (Pc[:, :, 2] > near).all(1)
    m = np.zeros((cam.H, cam.W), np.uint8) if out is None else out
    if not ok.any():
        return m > 0
    uv, _ = cv2.projectPoints(Pc[ok].reshape(-1, 1, 3), _ZERO3, _ZERO3,
                              cam.K, cam.dist)
    a = np.ascontiguousarray(np.round(uv.reshape(-1, 3, 2)), np.int32)
    for tri in a:
        cv2.fillConvexPoly(m, tri, 1)          # ★ 逐块：列表形式是奇偶规则，见模块 docstring
    return m > 0


def coverage(tris, cam, ss=2, near=0.005):
    """剪影的**覆盖率场**（float32，0~1），用 `ss` 倍超采样再面积降采样得到。

    为什么要覆盖率而不是二值掩码：拟合损失对二值掩码是**分段常数**的 —— 位姿动一点点
    掩码不变、损失不变，优化器看不出方向。覆盖率给了亚像素的梯度。
    `ss=2` → 每像素 4 个样本（0/0.25/0.5/0.75/1 五档），够用且便宜。

    ★ **必须把主点挪 `(ss-1)/2` 个超采样像素**（见下面两行）。不挪的话：
      `Ks = diag(ss,ss,1)@K` 之后 `round` 到整数格，原像素 `i` 的 `ss²` 个子像素
      落在原坐标 `i + a/ss`（a=0…ss−1）上，块均值的位置是 `i + (ss−1)/(2ss)`，
      而在 cv2 的约定里 `silhouette`（ss=1）的采样点在整数坐标 `i` ——
      于是**模型侧的边缘相对观测侧系统性偏移**，`ss=2` 时是 +0.25px（= +0.15mm@249mm）。
      实测后果：观测（二值）配模型（覆盖率）时，目标的最小值**不在真值上** ——
      真值自己的 loss 0.0277，而差它 6°、位置差 1.8mm 的一个位姿能拿到 0.0182，
      因为那个位姿把剪影整体"撑胖"了 ~2% 来补这 0.25px。挪主点后两边同一套格，
      真值重新变成最优解附近。**这个半像素比 Powell 那个 bug 更隐蔽**：
      它不报错、结果看起来"更像样"（IoU 0.982），只是位姿悄悄偏了。
    """
    if ss <= 1:
        return silhouette(tris, cam, near=near).astype(np.float32)
    W, H = cam.W * ss, cam.H * ss
    Ks = np.diag([float(ss), float(ss), 1.0]) @ cam.K   # 畸变系数是无量纲的，不用缩
    Ks[0, 2] -= (ss - 1.0) / 2.0                        # ★ 见上：把采样格对回像素中心
    Ks[1, 2] -= (ss - 1.0) / 2.0
    cs = MR.Cam(Ks, cam.dist, cam.T_base_cam, W=W, H=H)
    m = silhouette(tris, cs, near=near)
    return cv2.resize(m.astype(np.float32), (cam.W, cam.H),
                      interpolation=cv2.INTER_AREA)


def scaled_cam(cam, s):
    """等比例缩小画面的相机（粗搜索用；内参等比缩 ⇒ 画面就是原图的 s 倍）。"""
    W = max(32, int(round(cam.W * s)))
    H = max(24, int(round(cam.H * s)))
    Ks = np.diag([s, s, 1.0]) @ cam.K
    return MR.Cam(Ks, cam.dist, cam.T_base_cam, W=W, H=H)


# ================================================================ 位姿参数化

def exp_so3(w):
    """旋转向量 → 旋转矩阵（Rodrigues 闭式）。与仓库既有约定一致（不用欧拉角）。"""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
        return np.eye(3) + K
    k = np.asarray(w, float) / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def log_so3(R):
    """旋转矩阵 → 旋转向量（误差用角度报的时候要它）。"""
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if abs(np.pi - th) < 1e-6:                   # 接近 π：用 (R+I)/2 的特征分解
        M = (R + np.eye(3)) * 0.5
        v = np.sqrt(np.maximum(np.diag(M), 0.0))
        k = v / max(np.linalg.norm(v), 1e-12)
        return k * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) * (th / (2.0 * np.sin(th)))


def pose_T(p, s):
    """参数向量 → 物体→base 的 4×4（含尺度）。参数约定 6 维 `[w(3), t(3)]` 或 7 维加 `log s`。"""
    T = np.eye(4)
    T[:3, :3] = exp_so3(p[:3]) * s
    T[:3, 3] = p[3:6]
    return T


# 超斐波那契那条第二个无理数。**不能拿 g² 顶上**（见 `so3_grid` 的踩坑记录）。
PSI = 1.533751168755204288118041


def so3_grid(n, seed=None):
    """`n` 个铺得比较均匀的旋转（超斐波那契四元数），确定性。

    四元数 → 旋转矩阵，返回 `(n,3,3)`。给粗搜索当种子用；之后一律交给模式搜索细化 ——
    网格再密也只是「找到正确的盆地」，不是精度来源。

    ★★ **踩过大坑：这里原来是 `u3 = (i·g²) % 1`，而 `g² = g + 1` ⇒
      `i·g² = i·g + i` ⇒ 小数部分只差整数 i ⇒ `u3 ≡ u2 + 1/2`（实测 `(u3−u2)%1`
      在 1024 个点上恒等于 0.5）。于是
      `q = (a·sin2πu2, a·cos2πu2, −b·sin2πu2, −b·cos2πu2)` —— 四元数**只剩两个参数**，
      整族旋转躺在 S³ 的一个**二维曲面**上，而 SO(3) 是三维的。后果是覆盖半径
      **不随 n 收缩** —— 实测（4000 个 Haar 随机旋转，`rot_err_deg`，seed=1）：

        | n | 旧 (`i·g²`) 最坏 / 中位 | 新 (`i·ψ`) 最坏 / 中位 |
        |---|------------------------|------------------------|
        | 128  | 108.7° / 44.3° | 49.1° / 23.9° |
        | 512  | 102.1° / 38.4° | 28.0° / 15.0° |
        | 2048 |  97.5° / 36.5° | 20.5° / 10.7° |

      旧的那套 n 翻 16 倍只降 7°，新的每翻 4 倍降 ~28%。这就是「粗搜索偶尔把真值
      盆地整个漏掉」的根因 —— 修之前那几次失败，真值离**最近**的种子有 68°~82°，
      中级一步才 8.6°，根本走不过去。换成 ψ 之后同一套 9 次自检：
      位置差从 [2.06, 1.20, 32.62, 36.70, 2.81, 3.27, 3.58, 53.11, 11.62]mm
      变成 [1.59, 0.96, 1.91, 1.83, 3.31, 2.01, 3.84, 51.82, 10.54]mm，
      「拟合不比真值差」的帧从 7/9 变成 9/9。剩下那一次（trial 7）是**目标本身**
      的歧义（拟合 IoU 0.9651 > 真值 0.9548，见 `smooth` 那条注释），不是搜索的锅。
    """
    i = np.arange(n) + 0.5
    g = (1.0 + 5.0 ** 0.5) / 2.0
    u1, u2, u3 = i / n, (i * g) % 1.0, (i * PSI) % 1.0
    if seed is not None:
        rng = np.random.default_rng(seed)
        u1, u2, u3 = [u + (rng.random() - 0.5) / max(n, 1) for u in (u1, u2, u3)]
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    q = np.stack([a * np.sin(2 * np.pi * u2), a * np.cos(2 * np.pi * u2),
                  b * np.sin(2 * np.pi * u3), b * np.cos(2 * np.pi * u3)], 1)
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                               2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                               2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                               1 - 2 * (x * x + y * y)], -1)], 1)


# ================================================================ 损失

def loss_soft(cov, obs, occ=None):
    """软 IoU 损失 `1 − |M∩O|/|M∪O|`，`occ` 处**只许漏、不许多**。

    分子分母都按覆盖率算（`inter = Σ cov·obs`），所以损失对位姿是**光滑**的。
    `occ` 是「已知被夹爪挡住、观测里必然没有」的像素：在那里**模型多画不罚**
    （模型可能真的在那儿）、**模型少画也不罚**（观测本来就看不见）——
    于是这两类像素从并集里整体剔除。`occ` **只能来自观测**，绝不能来自 FK：
    HANDOVER 记着搬运深度上 FK⊗手眼差 150~162px，用它当遮挡掩码会毒化拟合。

    返回 `(loss, 明细 dict)`。
    """
    o = obs.astype(np.float32)          # obs 可以是**覆盖率场**（0~1），不必是二值
    c = cov.astype(np.float32)
    free = 1.0 if occ is None else (~occ).astype(np.float32)
    inter = float((c * o * free).sum())
    extra = float((c * (1.0 - o) * free).sum())          # 画到观测外面去了
    miss = float(((1.0 - c) * o * free).sum())           # 观测有、模型没画
    un = inter + extra + miss
    return 1.0 - inter / max(un, 1e-6), dict(inter=inter, extra=extra, miss=miss,
                                             union=un, iou=inter / max(un, 1e-6))


# ================================================================ 网格封装

class MeshObj:
    """一个待拟合的网格 + 它的多级减面（粗搜索用粗网格，省时间）。"""

    def __init__(self, tris, name="mesh", cells=(4e-3, 3e-3, 1.5e-3)):
        self.name = name
        tris = np.asarray(tris, np.float32)
        p = tris.reshape(-1, 3)
        ext = float((p.max(0) - p.min(0)).max())
        if not (0.005 < ext < 0.5):          # 单位闸门：18 个 STL 里有一个是毫米
            raise ValueError(f"{name} 外接盒 {ext:.4f} m 不合常理，拒绝（看 load_part 的注释）")
        self.tris0 = tris
        self.ext = ext
        self.centre = p.mean(0)              # 重建产物的质心 ≈ 0，但仍按实际取
        self.cells = tuple(cells)
        self._lv = {}
        for c in self.cells:
            self._lv[c] = MR.decimate(tris, c)

    def level(self, cell=None):
        cell = self.cells[0] if cell is None else cell
        return self._lv[cell]

    def local(self, T):
        """物体系 → base 系（`T` 含尺度）。减面在**物体系**做，所以变换要在之后施加。"""
        return MR.transform(self.level(), T)


def load_mesh(path_or_name, cells=(4e-3, 3e-3, 1.5e-3)):
    """读一个网格（`calib/recon/` 下的重建产物，或 `workplace/so101_urdf/assets/`）。"""
    p = Path(path_or_name)
    if not p.exists():
        for cand in (RECON / f"{path_or_name}.stl", RECON / path_or_name,
                     REPO / "workplace/so101_urdf/assets" / f"{path_or_name}.stl"):
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise FileNotFoundError(f"找不到网格：{path_or_name}")
    return MeshObj(MR.stl_cached(p), p.stem, cells=cells)


# ================================================================ 腕视图拟合

def ray_depth_by_area(mesh, cam, c_px, area_px, cell=None, d0=0.12, iters=2,
                      R0=None, T_hint=None, cam_probe=None):
    """沿观测形心的视线滑，滑到**渲出的剪影面积 = 观测面积**为止。

    球那条路滑的是「半径 = 观测半径」；网格的对应量是**面积**。对固定朝向，
    投影面积 ∝ 1/d²（严格：沿视线平动 = 以相机心为中心的位似），所以
    `d ← d·√(A(d)/A_obs)` 一步就中，再迭代只是抗像素量化。
    **实测：真值 250.2mm → 解出 249.9mm（−0.30mm）。**

    `cam_probe` 给一个**缩略**相机（内参等比缩、`area_px` 按面积比缩过）来探路，
    只在最后一步用 `cam` 复核 —— 外层要扫几百个朝向，这一步省掉 16 倍时间。

    返回 `(P_base 物体中心, d_m, info)`。`R0` 给定初始旋转（默认单位阵）。
    """
    und = cv2.undistortPoints(np.asarray(c_px, float).reshape(1, 1, 2),
                              cam.K, cam.dist).reshape(2)
    dir_c = np.array([und[0], und[1], 1.0])
    dir_c /= np.linalg.norm(dir_c)
    t = mesh.level(cell if cell else mesh.cells[-1])
    if R0 is not None:
        t = MR.transform(t, np.block([[R0, np.zeros((3, 1))], [np.zeros(3), 1.0]]))
    d = float(d0)
    if T_hint is not None:
        d = float(np.linalg.norm(np.asarray(T_hint, float)[:3, 3]))
    pc = cam_probe or cam
    a_t = float(area_px) * (pc.W * pc.H) / float(cam.W * cam.H)
    hist = []
    for k in range(iters + 1):
        cc = pc if k < iters else cam                    # 最后一步用原相机复核
        aa = a_t if k < iters else float(area_px)
        P = cam.T_base_cam[:3, :3] @ (dir_c * d) + cam.T_base_cam[:3, 3]
        A = float(silhouette(MR.transform(t, _t_T(P)), cc).sum())
        hist.append((d, A))
        if A < 1.0:
            d *= 1.6                        # 太小了，退远点再试
            continue
        d = d * float(np.sqrt(A / max(aa, 1.0)))
        d = float(np.clip(d, 0.02, 1.5))
    P = cam.T_base_cam[:3, :3] @ (dir_c * d) + cam.T_base_cam[:3, 3]
    return P, d, dict(hist=hist, dir_c=dir_c, area_model=A)


# 长方体（以及任何有三个互相垂直镜面的形状）在绕自身三轴各转 180° 下**精确不变**，
# 于是单张剪影**原理上**分不出这四个姿态。重建出来的舵机只是近似这个对称 —— 实测
# 绕三轴各 180° 的剪影 IoU 是 0.943 / 0.983 / 0.943，而**真值位姿自己**才 0.9745，
# 也就是说翻转姿态比真值还「像」。所以报姿态误差时**必须同时报两个数**：
# 原始误差，以及「商掉这个对称族之后」的误差。后者才是单剪影真正的分辨力。
_KLEIN = [np.eye(3),
          np.diag([-1.0, -1.0, 1.0]),        # 绕 z 转 180°
          np.diag([1.0, -1.0, -1.0]),        # 绕 x 转 180°
          np.diag([-1.0, 1.0, -1.0])]        # 绕 y 转 180°


def rot_err_deg(R_fit, R_true, modulo_symmetry=False):
    """两个旋转之间的夹角（度）。`modulo_symmetry` 时取 `_KLEIN` 里最小的那个。"""
    def ang(R):
        return float(np.degrees(np.linalg.norm(log_so3(R))))
    if not modulo_symmetry:
        return ang(R_fit.T @ R_true)
    return min(ang(R_fit.T @ R_true @ S) for S in _KLEIN)


def _t_T(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


# 模式搜索的初始步长（7 维：3 旋转 rad + 3 平移 m + 1 log 尺度）。
# ★ 旋转那一档要和网格的**覆盖半径**对得上，不是拍脑袋：`so3_grid(512)` 修好之后
#   覆盖半径 max 26.6° / 中位 15.0°（见 `so3_grid` 的实测），所以中级步长取
#   0.25 rad ≈ 14°，从种子里走出去两三步就能覆盖这个半径；细级再收到 0.08 rad ≈ 4.6°。
#   （修网格之前这里的注释写的是「最近邻角间距 ≈ 9°」，那是照抄公式、没实测 ——
#    真正的数字是 40°+，而且不随 n 收缩。）
MID_STEP = np.array([0.25] * 3 + [4e-3] * 3 + [0.04])
FINE_STEP = np.array([0.08] * 3 + [2e-3] * 3 + [0.02])


def _blur(a, sg):
    """同一套核，两侧都用（`smooth` 那条路的关键就是「同核」）。"""
    return cv2.GaussianBlur(np.ascontiguousarray(a, np.float32), (0, 0), sg)


def fit_wrist(mesh, cam, obs, occ=None, seeds=None, s0=1.0, fix_s=1.0,
              n_rot=512, coarse=0.5, cell_coarse=None, cell_fine=None,
              keep=16, verbose=True, seed=0, t_scale=0.05, smooth=0.0,
              R_fixed=None, rot_range=0.0):
    """腕视图剪影拟合：`(R, t, s)` 的相似变换，三级「粗网格搜索 → 中分辨率 → 全分辨率」。

    `R_fixed`：给定后**锁住物体朝向**（只在中/细级允许 ±`rot_range` 的修正）。
    用途：橡皮这类长条物，单剪影有「竖着 / 横着」等价盆地；按真实夹取方式
    （12mm 厚度沿夹爪接近方向、22mm 跨指、40mm 沿指长）锁住朝向，只解平移。
    不传则保持原来的自由 6-DoF。

    在**真实腕相机**里做（真内参 + 已有位姿链），所以是普通的单剪影 6-DoF 估计。

    分级（每级都实测过成本，见 `_probe`/HANDOVER）：
      · 粗：画面缩到 `coarse` 倍、网格用最粗一级，扫 `n_rot` 个朝向 × 面积定深（1 次渲/朝向）
      · 中：取损失最好的 `keep` 个种子，**模式搜索**在缩略图上细化 7 维（不用 Powell，见 `pattern_search`）
      · 细：最优种子在全分辨率 + 细网格上再细化一遍

    `smooth=σ`：给**模型和观测各模糊一遍同一个核**再用软 IoU。默认 0（不模糊）。
    ★ 为什么会有这个开关 —— 实测的目标**偏差**：观测是**点采样**的二值掩码（颜色阈值
      出来的，每个像素回答「中心在不在物体里」），模型是 `coverage` 的 **2×2 盒平均**。
      两种约定不可比，于是软 IoU 的最小值**不在真值上**：合成自检里真值自己的 loss 是
      0.0386，而 16 个种子里有 11 个收敛到**比它更低**（最好 0.0253，姿态差 176°）。
      两边同核模糊后，0/16 个种子能压过真值 ⇒ 真值重新成为最优解。
      没做成默认值的原因：它改变目标的形状，会作废现有那套步长/分级的标定，
      而当前这两者纠缠在一起的残差（~0.25px 量级）**小于真机上的模型误差**
      （真帧上拟合 IoU 0.93 而不是 1.0）。要换就得重标一遍，写在 HANDOVER 里了。
    """
    if smooth and smooth > 0:
        obs = _blur(obs, smooth)
    # ★ 观测**保持覆盖率场（float 0/1）**，不要变 bool、也不要存成 uint8 再缩放。
    #   踩过：`obs.astype(np.uint8)` 给的是 0/1（不是 0/255），`INTER_AREA` 求均值后
    #   uchar 截断 ⇒ 粗尺度的观测被**系统性腐蚀**（腐蚀量还随缩放倍率变），
    #   于是粗搜索排序失真、σ=0.5px 那档怎么都追不上真值。
    #   两侧都用覆盖率场，软对软，缩放才自洽。
    obs = obs.astype(np.float32)
    if (obs > 0.5).sum() < 30:
        raise ValueError(f"观测掩码只有 {(obs > 0.5).sum()}px，拟合没有意义")
    area = float(obs.sum())
    cell_coarse = cell_coarse or mesh.cells[0]
    cell_fine = cell_fine or mesh.cells[-1]
    yy, xx = np.mgrid[0:cam.H, 0:cam.W]
    c_px = np.array([xx[obs > 0.5].mean(), yy[obs > 0.5].mean()])
    smin, smax = -np.log(1.15), np.log(1.15)

    def _cov_full(tris_):
        c = coverage(tris_, cam, ss=2)
        return _blur(c, smooth) if smooth else c

    def _down(c):
        """降到粗尺度 —— **必须走和 `obs_c` 完全相同的那一次 `INTER_AREA`**。

        ★ 踩过：粗级原来是用 `scaled_cam(cam, 0.5)` 直接渲 320×240，而 `obs_c` 是
          `INTER_AREA` 降下来的。两者的采样盒**差半个原像素**：`Ks = diag(s,s,1)@K`
          把粗像素 i 映到原图坐标 2i，而 `INTER_AREA` 的粗像素 i 平均的是原像素
          {2i, 2i+1}（中心在 2i+0.5）。半个像素的**系统性**偏移在粗尺度上就是
          「模型比观测胖/瘦 0.25 粗像素」，足以把**翻转盆地**排到真值盆地前面 ——
          实测粗搜索排名里 179.5° 那个候选的 loss 比离真值 10.65° 的种子低一倍。
          两边经过**同一个算子**（先全分辨率渲、再同参数 INTER_AREA）后，
          真值位姿在粗尺度上的损失才是 0 附近，排序才有意义。
        """
        return cv2.resize(c, (cam_c.W, cam_c.H), interpolation=cv2.INTER_AREA)

    # 参数向量的**局部化**：t 存「相对 P0_ref 的偏移」，s 存 log 比 —— 这样 7 个分量的
    # 量级相近，Powell 那套「每个方向同一个初始步长」才不会在平移上乱走。
    # fix_s 给定时第 7 维照样存在，只是把界收死（不然 bounds 长度会和 x0 对不上）。
    def unpack(x, s_ref=None):
        ref = P0_ref if s_ref is None else s_ref
        s = float(fix_s) if fix_s is not None else float(np.exp(x[6]) * s0)
        T = pose_T(x[:6], s)
        if R_fixed is not None:
            # x[:3] 是相对 R_fixed 的小修正；rot_range=0 时恒为单位阵
            T[:3, :3] = np.asarray(R_fixed, float) @ T[:3, :3]
        T[:3, 3] += ref
        return T, s

    def s_bounds():
        return (smin, smax) if fix_s is None else (-1e-9, 1e-9)

    # ---- 级 1：粗。每个朝向先用面积把深度定死，再渲一次打分（1 次渲/朝向）----
    t0 = time.time()
    cam_c = scaled_cam(cam, coarse)
    obs_c = cv2.resize(obs, (cam_c.W, cam_c.H), interpolation=cv2.INTER_AREA)
    occ_c = (None if occ is None else
             cv2.resize(occ.astype(np.float32), (cam_c.W, cam_c.H),
                        interpolation=cv2.INTER_AREA) > 0.5)
    cam_p = scaled_cam(cam, coarse * 0.5)             # 定深探路用更小的画面（省 4 倍）
    Rs = [np.asarray(R_fixed, float)] if R_fixed is not None \
        else so3_grid(n_rot, seed=seed)
    cand = []
    for R in Rs:
        P, _, _ = ray_depth_by_area(mesh, cam, c_px, area, cell=cell_coarse, R0=R,
                                    cam_probe=cam_p)
        T = np.eye(4)
        T[:3, :3] = R * s0
        T[:3, 3] = P
        cov = _down(_cov_full(mesh_tris_at(mesh, cell_coarse, T)))
        lo, det = loss_soft(cov, obs_c, occ_c)
        cand.append((lo, R, P, det["iou"]))
    cand.sort(key=lambda z: z[0])
    t_coarse = time.time() - t0
    P0_ref = np.mean([c[2] for c in cand[:max(keep, 1)]], 0)

    # ---- 级 2：中。模式搜索在缩略图上细化（渲染仍是全分辨率+同一算子降采样）----
    def mk_fun(obs_, cell_, ref, occ_, down=False):
        def f(x):
            T, _ = unpack(x, ref)
            c = _cov_full(mesh_tris_at(mesh, cell_, T))
            return loss_soft(_down(c) if down else c, obs_, occ_)[0]
        return f

    # ★ 这里原来调 `minimize(method="Powell")`。**换成模式搜索是被实测逼出来的**：
    #   从真值位姿出发跑 Powell，损失反而从 0.0432 涨到 0.2535、姿态漂了 88°
    #   —— 剪影覆盖率是阶梯状的（判据是「像素在不在里面」），Powell 的共轭方向
    #   在台阶上根本不保证下降，它会把一个好种子毁掉。模式搜索每步只在严格下降时才
    #   移动，最坏情况是原地不动。详见 `pattern_search` 的 docstring。
    rb = (-rot_range, rot_range) if R_fixed is not None else (-np.pi, np.pi)
    bnd_mid = [rb] * 3 + [(-t_scale, t_scale)] * 3 + [s_bounds()]
    f_mid = mk_fun(obs_c, cell_coarse, P0_ref, occ_c, down=True)
    refined = []
    n_ev_mid = 0
    for lo, R, P, _ in cand[:keep]:
        x0 = np.concatenate([
            np.zeros(3) if R_fixed is not None else log_so3(R), P - P0_ref, [0.0]])
        xr, fr, ne = pattern_search(f_mid, x0, MID_STEP, bnd_mid,
                                    shrink=0.5, min_ratio=4e-3, max_iter=160)
        n_ev_mid += ne
        refined.append((fr, xr))
    # ★ 排序**只用来看**，选谁进细级一律按全分辨率损失重打一遍（见下）。粗尺度的
    #   排序靠不住：它把「粗尺度上好看」的种子排在前面，而真值附近的种子常常
    #   损失更高（trial 7 实测：真值自己在粗尺度排第 1，但离真值最近的**网格点**
    #   损失 0.145，排到十几名开外）。全分辨率一次渲染在粗网格上只要 ~13ms，
    #   16 个候选全打一遍 0.2s，比「按粗尺度排名挑前两名」稳得多。
    refined.sort(key=lambda z: z[0])
    t_mid = time.time() - t0 - t_coarse

    # ---- 级 3：全分辨率 + 细网格 ----
    #   全分辨率损失既当**选择器**（所有中级结果都打一遍）又当**目标**（前几名各细化一遍）。
    f_fine = mk_fun(obs, cell_fine, P0_ref, occ, down=False)
    scored = []
    for lo, x in refined:
        T, s = unpack(x)
        lf, det = loss_soft(_cov_full(mesh_tris_at(mesh, cell_fine, T)), obs, occ)
        scored.append((lf, x.copy()))
    scored.sort(key=lambda z: z[0])

    bnd_fine = [rb] * 3 + [(-2 * t_scale, 2 * t_scale)] * 3 + [s_bounds()]
    n_ev_fine = 0
    best = None
    for _, xk in scored[:2]:
        xr, fr, ne = pattern_search(f_fine, xk, FINE_STEP, bnd_fine,
                                    shrink=0.5, min_ratio=2e-3, max_iter=120)
        n_ev_fine += ne
        if best is None or fr < best[0]:
            best = (fr, xr.copy())
    x = best[1]
    T, s = unpack(x)
    cov = _cov_full(mesh_tris_at(mesh, cell_fine, T))
    lf, det = loss_soft(cov, obs, occ)
    cond = _cond_powell(f_fine, x)

    z_m = float(np.linalg.norm(T[:3, 3] - cam.T_base_cam[:3, 3]))
    info = dict(mesh=mesh.name, area_obs=int(area), s=s, cell_fine=cell_fine,
                loss=lf, **det, n_rot=n_rot, coarse=coarse,
                t_coarse_s=t_coarse, t_mid_s=t_mid, t_total_s=time.time() - t0,
                depth_mm=z_m * 1000, f_px=float(cam.K[0, 0]),
                # σ_z：深度是从**视尺寸**反解的，`w_px = f·W/z ⇒ z = f·W/w_px`，
                # 于是 `σ_z = σ_px·z²/(f·W)`。★ `W` 是**物体自身的尺寸**（米），不是尺度系数
                # —— 一开始把无量纲的 s 填在这儿，预测值差了 45 倍（58mm vs 1.1mm）。
                sigma_z_mm=(0.3 * (z_m * 1000.0) ** 2)
                           / (float(cam.K[0, 0]) * mesh.ext * 1000.0 * s),
                cond=cond, c_px=c_px.tolist(),
                n_ev_mid=n_ev_mid, n_ev_fine=n_ev_fine,
                cand_iou=[float(c[3]) for c in cand[:keep]], cand=cand)
    if verbose:
        print(f"  [{mesh.name}] IoU={det['iou']:.4f} 损失={lf:.4f} s={s:.3f} "
              f"深度={z_m*1000:.0f}mm｜粗{len(Rs)}朝向 {t_coarse:.2f}s + "
              f"中{len(refined)} {t_mid:.2f}s + 细 {info['t_total_s']-t_coarse-t_mid:.2f}s｜"
              f"σ_z 预测 {info['sigma_z_mm']:.2f}mm｜Hessian 条件数 {cond:.2e}")
    return T, dict(cov=cov, obs=obs, **info)


def mesh_tris_at(mesh, cell, T):
    """减面网格按 `T` 变换到 base 系（`T` 已含尺度）。"""
    return MR.transform(mesh.level(cell), T)


def basins(cand, tol=0.02, sep_deg=20.0):
    """粗搜索里**有几个互不相同的等价姿态盆地**。

    `cand` 是 `fit_wrist` 的粗搜索表 `[(loss, R, P, iou), …]`。把 IoU 在
    「最好那个 − `tol`」以内的朝向都挑出来，按夹角 > `sep_deg` 贪心聚类，数簇数。

    ★ 这才是「单剪影把朝向定住多少」的**全局**指标。`orientation_spread` 的 θ* 只说明
      盆地**内部**有多窄（实测 2~5°），它和「有几个盆地」是两件事：
      盆地很窄 + 有好几个 ⇒ 优化器会精确地收敛到一个**和真值剪影等价**的错误姿态，
      报一个大的「姿态误差」会让人误以为是优化失败。实测长方体重建件的翻转族
      （`_KLEIN`）剪影 IoU 就到 0.943~0.983，而真值位姿自己才 0.9745。
    """
    if not cand:
        return 0, [], 0.0
    best = min(c[0] for c in cand)
    good = [c for c in cand if c[0] <= best + tol + 1e-12]
    reps = []
    for c in sorted(good, key=lambda z: -z[3]):
        if all(rot_err_deg(c[1], r[1]) > sep_deg for r in reps):
            reps.append(c)
    return len(reps), [float(r[3]) for r in reps], float(best)


def orientation_spread(mesh, cam, obs, T, tol=0.02, n_dir=64, seed=0,
                       hi=np.pi, cell=None, rounds=12):
    """拟合出来的**朝向**到底定了多少：报「拐角 `θ*`」——把姿态绕随机轴转 `θ`，
    只要 `θ ≤ θ*`，IoU 就都不掉出 `IoU_fit − tol`；`θ*` 就是对分搜索出的那个临界角。

    为什么不用「姿态误差」一个数：长方体的单剪影有**离散的翻转族**（见 `_KLEIN`），
    转角误差可以很大而 IoU 几乎不变 —— 报一个「误差 127°」会让人以为是优化失败，
    其实那个姿态和真值在剪影上**等价**。`θ*` 才是可复现、可比较的分辨力指标。

    返回 `(theta_star_deg, rows)`；`rows` 是二分过程中每一档的 (θ, 最差 IoU)。
    """
    T = np.asarray(T, float)
    obs = obs > 0
    cell = cell or mesh.cells[-1]
    s = float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))       # T = s·R ⇒ det = s³
    R0 = T[:3, :3] / max(s, 1e-9)
    iou_fit = loss_soft(coverage(mesh_tris_at(mesh, cell, T), cam, ss=2),
                        obs, None)[1]["iou"]
    floor = iou_fit - tol
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_dir, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def worst_at(th):
        """绕 `n_dir` 个随机轴各转 `th`（物体系），返回**最差**的那个 IoU。"""
        out = []
        for d in dirs:
            R = R0 @ exp_so3(d * th)
            Tk = np.eye(4)
            Tk[:3, :3] = R * s
            Tk[:3, 3] = T[:3, 3]
            out.append(loss_soft(coverage(mesh_tris_at(mesh, cell, Tk), cam, ss=2),
                                 obs, None)[1]["iou"])
        return float(min(out))

    rows, lo, hi_ = [], 0.0, float(hi)
    w_hi = worst_at(hi_)
    rows.append((hi_, w_hi))
    if w_hi >= floor:                     # 转满 180° 也不掉 ⇒ 朝向根本没定住
        return float(np.degrees(hi_)), rows
    for _ in range(rounds):
        mid = 0.5 * (lo + hi_)
        w = worst_at(mid)
        rows.append((mid, w))
        if w >= floor:
            lo = mid
        else:
            hi_ = mid
    return float(np.degrees(lo)), rows


def s_tolerance(mesh, cam, obs, T, tol=0.01, cell=None, span=(0.7, 1.4), n=29):
    """**尺度 ↔ 深度是精确简并的**：`w_px = f·s·W/z` 只有比值 `s/z` 可观测。

    所以单张剪影**定不出**新物体的公制尺度；要么钉住 CAD 尺度（`fit_wrist` 默认
    `fix_s=1.0`），要么从别处拿深度。这里把这条简并**量出来**：沿 `T` 的相机视线把物体
    缩放 `s` 倍并重解深度（面积定深会跟着补偿），报出 IoU 掉到 `1−tol` 以内的 s 区间。
    区间越宽 ⇒ 剪影对尺度/深度的分辨力越弱。

    返回 `(s_lo, s_hi, rows)`。
    """
    T = np.asarray(T, float)
    C = cam.T_base_cam[:3, 3]                    # 相机心在 base 里
    d0 = float(np.linalg.norm(T[:3, 3] - C))
    u = (T[:3, 3] - C) / max(d0, 1e-9)
    obs = obs > 0
    A0 = float(obs.sum())
    o = obs.astype(np.float32)
    rows = []
    for s in np.linspace(*span, n):
        P = C + u * (d0 * s)                     # 沿视线挪，保持成像大小不变
        Tk = T.copy()
        Tk[:3, :3] = T[:3, :3] * s
        Tk[:3, 3] = P
        cov = coverage(mesh_tris_at(mesh, cell or mesh.cells[-1], Tk), cam, ss=2)
        lo, det = loss_soft(cov, obs, None)
        rows.append((float(s), det["iou"]))
    ok = [s for s, i in rows if i >= 1.0 - tol]
    return (min(ok) if ok else float("nan"), max(ok) if ok else float("nan"), rows)


def flat_direction(mesh, cam, obs, T, cell=None, tol=0.02, h=1e-3, span_mm=80.0,
                   n_prof=41, verbose=True, tag=""):
    """在 `T` 处，目标**最平的那个方向**是「转多少 + 沿线走多少」—— 量出来的。

    为什么非要有这个数：报「姿态差 51.8mm」会被读成优化失败，可**单张剪影的弱方向
    根本不是某个单轴**。`axis_resolution` 量的是单轴：转 2~3° 或横移 0.5~1mm 掉出 0.97。
    但把「转一点 + 沿视线走一段」**一起**动，剪影几乎不变 —— 因为转 45.4mm 长的物体
    会让投影宽度变化，而走远会让它变小，两者互相抵消。合成自检 9 次里唯一一次大失败
    （trial 7，51.8mm、商对称后 84°）就是优化器顺着这个组合走到了一个**损失比真值还低**
    的姿态（0.0345 vs 真值 0.0452）—— 那是目标自己的简并，不是搜索的锅。

    做法：6 维局部参数 `[w(3) rad, t(3) m]`（尺度钉死）上的有限差分 Hessian，
    取**绝对值最小**的特征值对应的特征向量 `v`，再把它翻译成物理量：
      · 归一化到「转 1°」，报这时沿线要走多少 mm（比值 r_mm_per_deg）
      · 沿 `v` 单向走 `±span_mm` 等价的距离，记损失剖面 ⇒ 报「损失涨 `tol` 以内」的行程
    返回 dict。**注意剖面是沿固定方向的 1-D 截线**，不做重优化，所以它只描述局部平坦度；
    优化器实际能走多远还要看它愿不愿意沿弯曲的谷底拐弯（trial 7 就走了 51.8mm）。
    """
    cell = cell or mesh.cells[-1]
    o = (obs > 0).astype(np.float32)
    T = np.asarray(T, float)
    s = float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))
    R0 = T[:3, :3] / max(s, 1e-9)

    def pose(x):
        Tk = np.eye(4)
        Tk[:3, :3] = exp_so3(x[:3]) @ R0 * s
        Tk[:3, 3] = T[:3, 3] + x[3:6]
        return Tk

    def f(x):
        return loss_soft(coverage(mesh_tris_at(mesh, cell, pose(x)), cam, ss=2),
                         o, None)[0]

    x0 = np.zeros(6)
    f0 = f(x0)
    H = np.zeros((6, 6))
    for i in range(6):
        ei = np.zeros(6)
        ei[i] = h
        H[i, i] = (f(ei) - 2 * f0 + f(-ei)) / (h * h)
        for j in range(i + 1, 6):
            ej = np.zeros(6)
            ej[j] = h
            H[i, j] = H[j, i] = (f(ei + ej) - f(ei - ej) - f(-ei + ej)
                                 + f(-ei - ej)) / (4 * h * h)
    w, V = np.linalg.eigh(H)
    k = int(np.argmin(np.abs(w)))
    v = V[:, k]
    if np.linalg.norm(v[:3]) < 1e-12 and np.linalg.norm(v[3:]) < 1e-12:
        return dict(ok=False)
    deg_per_unit = np.degrees(np.linalg.norm(v[:3]))     # v 里 1 单位 = 多少度
    mm_per_unit = np.linalg.norm(v[3:]) * 1000.0
    if deg_per_unit > 1e-9:
        v = v / deg_per_unit                              # 归一到「转 1°」
    mm_per_deg = float(np.linalg.norm(v[3:]) * 1000.0)
    # 沿 v **从 0 往外**走，细步长（0.5mm 等价位移）—— 阶梯目标用 linspace 粗采会漏掉
    # 整段平台：4mm 的采样步会跳过 ss=2 的台阶，看起来哪儿都不平。
    def at(kk_mm):
        if deg_per_unit > 1e-9:
            x = v * (kk_mm / max(mm_per_deg, 1e-9))       # 沿线位移 = kk_mm
        else:
            x = v * (kk_mm / 1000.0)
        return float(f(x))

    step_mm = 0.5
    n_s = int(span_mm / step_mm)
    prof = [(0.0, f0)]
    reach = {+1: 0.0, -1: 0.0}
    for sgn in (+1, -1):
        for j in range(1, n_s + 1):
            kk = sgn * j * step_mm
            fv = at(kk)
            prof.append((float(kk), fv))
            if fv <= f0 + tol:
                reach[sgn] = kk
            else:
                break                                     # 第一次涨过 tol 就停
    prof.sort()
    out = dict(ok=True, cond=float(np.abs(w).max() / max(np.abs(w).min(), 1e-30)),
               eig_min=float(w[k]), eig_max=float(w.max()),
               mm_per_deg=mm_per_deg, axis_w=[float(t) for t in v[:3]],
               trans_dir=[float(t) for t in v[3:]], loss0=float(f0),
               flat_mm=(reach[-1], reach[+1]), profile=prof, step_mm=step_mm)
    if verbose:
        print(f"    {tag}最平方向：转 1° 要沿线走 {mm_per_deg:.2f}mm；"
              f"沿它往外走、损失涨 ≤{tol} 的行程 "
              f"{out['flat_mm'][0]:+.1f}mm / {out['flat_mm'][1]:+.1f}mm"
              f"（步长 {step_mm}mm）｜Hessian 条件数 {out['cond']:.2e}")
    return out


def pattern_search(f, x0, step0, bounds, shrink=0.5, min_ratio=4e-3,
                   max_iter=160):
    """罗盘式模式搜索（compass / pattern search）—— **只在严格下降时才移动**的局部优化。

    为什么不用 `scipy.optimize.minimize(method="Powell")`：剪影覆盖率是**阶梯状**的
    （`ss=2` 时每像素只有 0/0.25/0.5/0.75/1 五档），位姿动一点点函数值不变、再动一点
    就跳一档。实测**从真值位姿出发**跑 Powell，loss 会从 0.0432 **涨到 0.2535**
    （姿态漂 88°）—— 它的线搜索假设函数光滑，在阶梯上会被带偏，而且**不保证不上升**。
    模式搜索每步只接受 `f` 更小的解，所以最坏情况就是「原地不动」，不会把好种子毁掉。

    步长按 `shrink` 几何收缩，直到所有分量的步长都缩到 `step0 * min_ratio`。
    各分量的初始步长要自己给（旋转 rad / 平移 m / log 尺度 量级差很多）。

    返回 `(x, f(x), 评估次数)`。
    """
    x = np.asarray(x0, float).copy()
    fx = float(f(x))
    step = np.asarray(step0, float).copy()
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    n, nev, it = len(x), 1, 0
    while (step > np.asarray(step0, float) * min_ratio).any() and it < max_iter:
        it += 1
        moved = False
        for i in range(n):
            for sgn in (1.0, -1.0):
                xt = x.copy()
                xt[i] = min(max(x[i] + sgn * step[i], lo[i]), hi[i])
                if xt[i] == x[i]:
                    continue
                ft = float(f(xt))
                nev += 1
                if ft < fx:
                    x, fx, moved = xt, ft, True
                    break
        if not moved:
            step *= shrink
    return x, fx, nev


def _cond_powell(f, x, h=1e-3):
    """用有限差分估 Hessian 的条件数。

    ★ 条件数**依赖参数化**（这里有 rad 和 m 两种量纲混在一起），所以它只说明
      「在这套局部参数下哪几个方向接近平坦」，不是物理量。要报物理量就报
      `σ_z = σ_px·z²/(f·s)` 那条解析式（见 `fit_wrist` 的 info）。
    """
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / (h * h)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h
            v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej))
            H[i, j] = H[j, i] = v / (4 * h * h)
    w = np.linalg.eigvalsh(H)
    w = np.abs(w)
    return float(w.max() / max(w.min(), 1e-30))


# ================================================================ 自检

_SIGS = (0.0, 0.5, 1.5)


def _trial_scene(servo, rng, k, K, Tg):
    """第 `k` 次合成自检的场景：真值位姿 → 剪影 → 按 σ 模糊。**一处定义，两处用**
    （`selftest` 和出图的 `figures_stage_c`），不然图里的第 k 次和表里的第 k 次对不上。

    返回 `(Ttrue, cam, obs_mask_bool, sigma)`。`rng` 必须是**同一个** `default_rng`，
    按 k 递增调用；sigma 按 k 轮转 —— 这样一次跑完就能看到噪声档的影响。
    """
    w = rng.normal(size=3)
    w = w / np.linalg.norm(w) * rng.uniform(0.2, np.pi)
    Ttrue = np.eye(4)
    Ttrue[:3, :3] = exp_so3(w)
    Ttrue[:3, 3] = rng.uniform(-0.008, 0.008, 3)
    cam = MR.Cam(K, np.zeros(5), Tg, W=640, H=480)
    obs = silhouette(MR.transform(servo.level(servo.cells[-1]), Ttrue), cam)
    sig = _SIGS[k % len(_SIGS)]
    if sig > 0:
        obs = (cv2.GaussianBlur(obs.astype(np.float32), (0, 0), sig) > 0.5)
    return Ttrue, cam, obs, sig


def selftest(frames_seed=0, n_trials=12, verbose=True, smooth=0.0):
    """合成自检：**已知位姿 → 渲剪影 → 拟回去**，再和 `σ_z` 的预测对。

    测的是：投影、剪影、位姿参数化、损失、优化、以及「深度靠尺度反解」这条链自洽。
    **不测**真实光学、分割、光照、以及实物与 CAD 是否一致 —— 那要真采集。
    """
    out = {}
    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    servo = load_mesh("sts3215_03a_v1_v50_h0.5_m1")

    # --- 1) 渲染器：快路 vs z-buffer（应只差边界约定）---
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.0, 0.0, 0.09]
    cam = MR.Cam(K, np.zeros(5), T, W=640, H=480)
    for cell in servo.cells:
        t = MR.transform(servo.level(cell), np.eye(4))
        a = MR.rasterize(t, cam)["mask"]
        b = silhouette(t, cam) > 0
        out[f"iou_fast_vs_zbuf_{cell*1e3:.1f}mm"] = float(
            np.logical_and(a, b).sum() / np.logical_or(a, b).sum())
        out[f"px_fast_{cell*1e3:.1f}mm"] = int(b.sum())
        out[f"px_zbuf_{cell*1e3:.1f}mm"] = int(a.sum())
    if verbose:
        print("1) 快路剪影 vs z-buffer（应 ~0.987，且**不随减面变**）：")
        for cell in servo.cells:
            k = f"{cell*1e3:.1f}mm"
            print(f"   cell={k:>5s}  面数 {len(servo.level(cell)):6d}  "
                  f"IoU={out['iou_fast_vs_zbuf_' + k]:.4f}  "
                  f"{out['px_fast_' + k]}px vs {out['px_zbuf_' + k]}px")

    # --- 2) 面积定深：投影面积 ∝ 1/d²，验它一步就中 ---
    yy, xx = np.mgrid[0:480, 0:640]
    Tg = np.eye(4)
    Tg[:3, :3] = np.diag([1.0, -1.0, -1.0])
    Tg[:3, 3] = [-0.01, 0.0, 0.25]                  # 相机在 base 的 (−10, 0, 250)mm
    d_true = float(np.linalg.norm(Tg[:3, 3]))
    camt = MR.Cam(K, np.zeros(5), Tg, W=640, H=480)
    obs_g = silhouette(MR.transform(servo.level(servo.cells[-1]), np.eye(4)), camt)
    c_px = np.array([xx[obs_g].mean(), yy[obs_g].mean()])
    _, d_sol, _ = ray_depth_by_area(servo, camt, c_px, float(obs_g.sum()),
                                    cell=servo.cells[-1])
    out["depth_solve_mm"], out["depth_true_mm"] = d_sol * 1000.0, d_true * 1000.0
    if verbose:
        print(f"\n2) 面积定深：真值 {d_true*1000:.1f}mm → 解出 {d_sol*1000:.1f}mm"
              f"（误差 {d_sol*1000-d_true*1000:+.2f}mm）")

    # --- 3) 位姿恢复：随机真值位姿 → 拟合。σ 轮着来，看误差怎么随边界模糊涨 ---
    #     ★ 姿态误差报两个：原始的，和「商掉长方体固有的 180° 翻转族」之后的。
    #       后者才是单张剪影真正的分辨力（见 `_KLEIN` 处的实测数字）。
    rng = np.random.default_rng(frames_seed)
    sigs = (0.0, 0.5, 1.5)
    errs = []
    for k in range(n_trials):
        Ttrue, camk, obs_k, sig = _trial_scene(servo, rng, k, K, Tg)
        Tf, info = fit_wrist(servo, camk, obs_k, verbose=False, seed=k, smooth=smooth)
        nb, biou, _ = basins(info["cand"])
        # 真值位姿**自己**能拿多少分 —— 判「拟合是不是真的错」的唯一依据
        _o = _blur(obs_k.astype(np.float32), smooth) if smooth else obs_k
        iou_true = loss_soft(coverage(mesh_tris_at(
            servo, servo.cells[-1], Ttrue), camk, ss=2), _o, None)[1]["iou"]
        th_star, _ = orientation_spread(servo, camk, obs_k, Tf) if k < 4 else (np.nan, [])
        errs.append(dict(sig=sig, dR=rot_err_deg(Tf[:3, :3], Ttrue[:3, :3]),
                         dR_mod=rot_err_deg(Tf[:3, :3], Ttrue[:3, :3],
                                            modulo_symmetry=True),
                         dt_mm=float(np.linalg.norm(Tf[:3, 3] - Ttrue[:3, 3]) * 1000),
                         iou=info["iou"], iou_true=iou_true, n_basins=nb,
                         depth_mm=info["depth_mm"],
                         z_true_mm=float(np.linalg.norm(Ttrue[:3, 3]
                                                        - Tg[:3, 3])) * 1000,
                         sigma_z=info["sigma_z_mm"], theta_star=th_star))
    out["trials"] = errs
    if verbose:
        print(f"\n3) 位姿恢复 {n_trials} 次（σ 是给观测边界加的模糊，px）：")
        print(f"   {'σ':>5} {'位置差':>8} {'深度差':>8} {'姿态差':>9} {'商对称后':>9} "
              f"{'θ*':>5} {'盆地数':>6} {'IoU拟合':>8} {'IoU真值':>8} {'σ_z预测':>8}")
        for e in errs:
            print(f"   {e['sig']:5.1f} {e['dt_mm']:8.2f} "
                  f"{e['depth_mm']-e['z_true_mm']:8.2f} {e['dR']:9.2f} "
                  f"{e['dR_mod']:9.2f} {e['theta_star']:5.1f} {e['n_basins']:6d} "
                  f"{e['iou']:8.4f} {e['iou_true']:8.4f} {e['sigma_z']:8.2f}")
        amb = [e for e in errs if e["iou"] >= e["iou_true"] - 0.005]
        print(f"   ⇒ 「拟合 IoU ≥ 真值 IoU − 0.005」的帧：{len(amb)}/{len(errs)}"
              f"（这些帧上拟合**不比真值差**，姿态差多少都不算失败）")
    return out


# ================================================================ 真帧验证（腕视图·解析方盒）

def up_cam_from_area(g, c_px, cam_h_mm, h_mm, R_cam=None):
    """up 视图的**观测锚定局部相机**（没有真标定，就造一台自洽的）。

    和 `pose6d_swap.up_local_cam` 同一个构造，只把「焦距由**球的半径**反解」换成
    「焦距由**剪影面积**反解」（球只有一个半径，网格没有）：

      · 主点 = 观测形心 `c_px` ⇒ 物心正好落在光轴上
      · 相机位置由 `cam_h_mm`（离桌高度）和朝向反推 ⇒ 物心到相机 = 沿光轴的 `d`
      · 焦距待定，由 `A_render(f) = A_obs` 迭代解（面积 ∝ f² ⇒ `f ← f·√(A_obs/A_r)`，3~5 次）

    ★ **f 和 h 是退化的**：这个构造里 `f ≈ spm·d`（spm = 桌面处 px/m），
      f 大 1% 和相机高 1% 给出同一张图。所以这里报出的 f **不是**标定值，
      只有「面积对上了」这一条是可检验的 —— 真 up 内参要另做标定（原计划 Stage 3，本轮不做）。

    `R_cam`（3×3，cam→base，即 `T_base_cam[:3,:3]`）：**相机朝向**。
      不传 = 旧的「垂直下视、像面 x 对齐 base x」假设 —— 这对球/立方体无害，
      但它**凭空规定了像面的 roll**。真实 up 相机（2026-10-19 标定）的像面 x 轴是
      ≈ (0.161,−0.987,0.017)（几乎 −base y），和「对齐 base x」差了 ~90°。
      长条物（橡皮）一旦按真实夹取朝向摆，这个 roll 错就会让它在 up 图里
      转到错误方向 —— 看起来就是「front 往左、up 往右」。传 `R_cam` 后，
      相机仍锚在观测形心上（位置/f 不变），但像面朝向是真的，投影方向才是对的。
    """
    qq = g["Hi"] @ np.array([c_px[0], c_px[1], 1.0])
    P = np.asarray(g["T_base_board"], float) @ np.array(
        [qq[0] / qq[2], qq[1] / qq[2], 0.0, 1.0])
    x0, y0 = float(P[0]), float(P[1])
    T = np.eye(4)
    if R_cam is None:
        T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        T[:3, 3] = [x0, y0, float(cam_h_mm) / 1000.0]
    else:
        R = np.asarray(R_cam, float)
        T[:3, :3] = R
        centre = np.array([x0, y0, float(h_mm) / 1000.0])
        axis = R[:, 2]                                  # 光轴（base）
        dz = abs(float(axis[2]))
        d = (float(cam_h_mm) - float(h_mm)) / max(dz, 1e-6) / 1000.0
        T[:3, 3] = centre - d * axis                    # 物心在光轴上 ⇒ 投到主点
    return T, np.array([x0, y0, float(h_mm) / 1000.0])


def fit_up(mesh, g, obs, h_mm=0.0, cam_h_mm=None, f0=800.0, n_rot=180,
           fix_s=1.0, cell=None, verbose=True, seed=0, refine=True, R_fixed=None,
           R_cam=None):
    """up 视图拟合：相机自己造，物件姿态**只搜绕竖直轴那一维**。

    为什么只搜一维：up 相机离物体 ~700mm 而物体只有 45mm ⇒ 准正射，绕水平轴翻转对剪影
    是**二阶**小量。把那些自由度塞进拟合只会得到一个好看但不可观测的姿态 —— 这正是
    「球把 up 视图的问题掩盖掉」的镜像（球连一维都没有）。真要看 up 视图对倾斜有多敏感，
    走 `tilt_study`，那里是**量出来的**。

    `R_fixed`（3×3，物体→base）：给定后**完全锁死朝向**，不再搜 yaw，只解 x/y。
    长条物（橡皮）必须这么用：up 剪影对 40×22 的 yaw 约束很弱，自由拟合会选一个
    和腕视图（真实夹取朝向）不一样的角，两个视角就对不上。传入同一个
    `R_fixed` 后，两边的 3D 姿态才是同一个。

    返回 `(T_base_obj, info)`：info 里有反解出的 f、面积比、剪影 IoU、以及
    「面积锚定」这条自洽性的数字（`A_render/A_obs − 1`）。
    """
    cam_h = float(cam_h_mm if cam_h_mm is not None else BS.UP_CAM_H_MM)
    cell = cell or mesh.cells[-1]
    R_fix = None if R_fixed is None else np.asarray(R_fixed, float)
    o = obs.astype(np.float32)
    if (o > 0.5).sum() < 20:
        raise ValueError(f"up 观测掩码只有 {(o > 0.5).sum()}px")
    yy, xx = np.mgrid[0:o.shape[0], 0:o.shape[1]]
    c_px = np.array([xx[o > 0.5].mean(), yy[o > 0.5].mean()])
    area = float(o.sum())
    T_cam, centre = up_cam_from_area(g, c_px, cam_h, h_mm, R_cam=R_cam)

    # --- 焦距：面积 ∝ f²，一步一乘，几次就收敛 ---
    f = float(f0)
    hist = []
    for _ in range(6):
        cam = MR.Cam(np.array([[f, 0, c_px[0]], [0, f, c_px[1]], [0, 0, 1.0]]),
                     np.zeros(5), T_cam, W=o.shape[1], H=o.shape[0])
        T0 = np.eye(4)
        T0[:3, :3] = (np.eye(3) if R_fix is None else R_fix) * fix_s
        T0[:3, 3] = centre
        a_r = float(coverage(mesh_tris_at(mesh, cell, T0), cam, ss=2).sum())
        hist.append((f, a_r, area / max(a_r, 1e-9)))
        f = float(np.clip(f * np.sqrt(area / max(a_r, 1e-9)), 300.0, 2500.0))

    # --- 绕竖直轴：粗扫 n_rot 个角 + 局部精化 ---
    cam = MR.Cam(np.array([[f, 0, c_px[0]], [0, f, c_px[1]], [0, 0, 1.0]]),
                 np.zeros(5), T_cam, W=o.shape[1], H=o.shape[0])
    best = None
    if R_fix is None:
        for th in np.arange(n_rot) * (2 * np.pi / n_rot):
            Tk = np.eye(4)
            Tk[:3, :3] = exp_so3([0, 0, th]) * fix_s
            Tk[:3, 3] = centre
            lo, det = loss_soft(coverage(mesh_tris_at(mesh, cell, Tk), cam, ss=2), o, None)
            if best is None or lo < best[0]:
                best = (lo, th, det["iou"])
        th0 = best[1]
    else:
        # 朝向已锁：只量一次，后面只动 x/y
        Tk = np.eye(4)
        Tk[:3, :3] = R_fix * fix_s
        Tk[:3, 3] = centre
        lo, det = loss_soft(coverage(mesh_tris_at(mesh, cell, Tk), cam, ss=2), o, None)
        best = (lo, 0.0, det["iou"])
        th0 = 0.0

    rot_of = (lambda x: exp_so3([0, 0, x[2]]) * fix_s) if R_fix is None \
        else (lambda x: R_fix * fix_s)
    if refine:
        x0 = np.array([centre[0], centre[1], th0]) if R_fix is None \
            else np.array([centre[0], centre[1]])           # 朝向锁死 ⇒ 只解横向
        step = [max(0.0015, centre[2] * 0.01), max(0.0015, centre[2] * 0.01)]
        bnd = [(centre[0] - 0.02, centre[0] + 0.02), (centre[1] - 0.02, centre[1] + 0.02)]
        if R_fix is None:
            step.append(0.05)
            bnd.append((th0 - np.pi, th0 + np.pi))

        def fun(x):
            Tk = np.eye(4)
            Tk[:3, :3] = rot_of(x)
            Tk[:3, 3] = [x[0], x[1], centre[2]]
            return loss_soft(coverage(mesh_tris_at(mesh, cell, Tk), cam, ss=2),
                             o, None)[0]

        x, lo, nev = pattern_search(fun, x0, np.array(step), bnd, shrink=0.5,
                                    min_ratio=5e-3, max_iter=90)
    else:
        x = np.array([centre[0], centre[1], th0]) if R_fix is None \
            else np.array([centre[0], centre[1]])
        lo, nev = best[0], 0
    T = np.eye(4)
    T[:3, :3] = rot_of(x)
    T[:3, 3] = [x[0], x[1], centre[2]]
    cov = coverage(mesh_tris_at(mesh, cell, T), cam, ss=2)
    lo, det = loss_soft(cov, o, None)
    a_r = float(cov.sum())
    info = dict(mesh=mesh.name, f_px=f, cam_h_mm=cam_h, h_mm=h_mm, centre=centre.tolist(),
                # 把造出来的那台局部相机**交出去** —— 调用方要拿它渲图/量剪影，
                # 自己再拼一遍 K·T 迟早会和这里解出的 f 走岔。
                cam=cam, T_cam=T_cam, cell=cell,
                c_px=c_px.tolist(), area_obs=area, area_render=a_r,
                area_ratio=a_r / max(area, 1e-9), iou=det["iou"], loss=lo,
                theta_deg=(float(np.degrees(x[2])) if R_fix is None else
                           float(np.degrees(np.arctan2(R_fix[1, 0], R_fix[0, 0])))),
                rot_locked=R_fix is not None,
                shift_mm=[(x[0] - centre[0]) * 1000,
                          (x[1] - centre[1]) * 1000],
                coarse_iou=best[2], f_hist=hist, n_ev=nev,
                depth_mm=float(cam_h - h_mm),
                caveat="f 与相机高退化；只有面积这一条被锚住")
    if verbose:
        print(f"  [up:{mesh.name}] IoU={det['iou']:.4f} f={f:.1f}px（退化解）"
              f" 面积比 {info['area_ratio']:.4f} 绕竖轴 {info['theta_deg']:+.1f}°"
              f" 横向挪 {(np.hypot(*info['shift_mm'])):.2f}mm"
              f"｜深度 {info['depth_mm']:.0f}mm")
    return T, info


def fit_up_real(mesh, cam, obs, z_mm=0.0, R_fixed=None, fix_s=1.0, cell=None,
                n_rot=72, verbose=True, seed=0, xy_range_mm=50.0,
                xy_step_mm=2.0, yaw_step_deg=5.0):
    """up 视图拟合，**用真标定相机**（`calib/up_intrinsics.json` + `up_camera_pose.json`）。

    和 `fit_up` 的区别：不再造「观测锚定的垂直下视局部针孔」，而是直接用
    `MR.Cam(K, dist, T_base_cam)`。因此：
      · 相机高、焦距、光轴倾角都是标定值，**不再有 f↔h 退化**；
      · 物体中心高度 `z_mm` 必须给定（桌面上的物体 = 厚度/2；手持物体 = FK 的 P[2]）；
      · 像点 → 平面 z=z_mm 的射线交点直接给 x/y 初值。

    `R_fixed`：给定 3×3 物体→base 旋转后**锁死朝向**，只解 x/y。橡皮这类长条物
    在 up 视图里也应按真实夹取朝向摆（12mm 厚度竖直、22mm 跨指、40mm 沿指长）。
    不传则只搜绕 base z 的 yaw（桌面静物）。

    返回 `(T_base_obj, info)`；`info["cam"]` 就是传入的真相机，可直接渲图。
    """
    cell = cell or mesh.cells[-1]
    o = obs.astype(np.float32)
    if (o > 0.5).sum() < 20:
        raise ValueError(f"up 观测掩码只有 {(o > 0.5).sum()}px")
    yy, xx = np.mgrid[0:o.shape[0], 0:o.shape[1]]
    c_px = np.array([xx[o > 0.5].mean(), yy[o > 0.5].mean()])
    area = float(o.sum())
    z_m = float(z_mm) / 1000.0

    # 观测形心 → 相机射线 → 平面 z=z_m 的交点（真外参，所以这一步是闭式的）
    und = cv2.undistortPoints(c_px.reshape(1, 1, 2).astype(np.float64),
                              cam.K, cam.dist).reshape(2)
    d_c = np.array([und[0], und[1], 1.0], float)
    d_c /= np.linalg.norm(d_c)
    Rbc, tbc = cam.T_base_cam[:3, :3], cam.T_base_cam[:3, 3]
    d_b = Rbc @ d_c
    denom = d_b[2]
    if abs(denom) < 1e-9:
        raise ValueError("观测形心的射线与平面 z=z_mm 平行")
    P0 = tbc + d_b * ((z_m - tbc[2]) / denom)
    centre = np.array([P0[0], P0[1], z_m])

    def T_at(xy, yaw):
        T = np.eye(4)
        R = exp_so3([0, 0, yaw])
        if R_fixed is not None:
            R = np.asarray(R_fixed, float) @ R
        T[:3, :3] = R * fix_s
        T[:3, 3] = [xy[0], xy[1], z_m]
        return T

    def fun(x, cell_=cell, cam_=cam, obs_=o):
        T = T_at(x[:2], x[2])
        return loss_soft(coverage(mesh_tris_at(mesh, cell_, T), cam_, ss=2),
                         obs_, None)[0]

    # 粗扫 yaw：R_fixed 时只扫一个角（yaw=0）
    yaws = ([0.0] if R_fixed is not None
            else list(np.arange(n_rot) * (2 * np.pi / n_rot)))
    best = None
    for th in yaws:
        lo = fun(np.array([centre[0], centre[1], th]))
        if best is None or lo < best[0]:
            best = (lo, th)
    yaw0 = best[1]

    # x/y（+ 自由时的 yaw）局部精化
    x0 = np.array([centre[0], centre[1], yaw0])
    step = np.array([max(1e-3, xy_step_mm / 1000.0),
                     max(1e-3, xy_step_mm / 1000.0),
                     np.deg2rad(yaw_step_deg)])
    r = max(1e-3, xy_range_mm / 1000.0)
    bnd = [(centre[0] - r, centre[0] + r), (centre[1] - r, centre[1] + r),
           ((yaw0, yaw0) if R_fixed is not None
            else (yaw0 - np.pi, yaw0 + np.pi))]
    x, lo, nev = pattern_search(fun, x0, step, bnd, shrink=0.5, min_ratio=4e-3,
                                max_iter=140)
    T = T_at(x[:2], x[2])
    cov = coverage(mesh_tris_at(mesh, cell, T), cam, ss=2)
    lo, det = loss_soft(cov, o, None)
    a_r = float(cov.sum())
    info = dict(mesh=mesh.name, cam=cam, cell=cell, z_mm=float(z_mm),
                c_px=c_px.tolist(), P=centre.tolist(), area_obs=area,
                area_render=a_r, area_ratio=a_r / max(area, 1e-9),
                iou=det["iou"], loss=lo, yaw_deg=float(np.degrees(x[2])),
                shift_mm=[(x[0] - centre[0]) * 1000, (x[1] - centre[1]) * 1000],
                n_ev=nev, R_fixed=(None if R_fixed is None
                                   else np.asarray(R_fixed, float).tolist()))
    if verbose:
        print(f"  [up-real:{mesh.name}] IoU={det['iou']:.4f} z={z_mm:.1f}mm "
              f"yaw={info['yaw_deg']:+.1f}° 面积比 {info['area_ratio']:.4f} "
              f"横向挪 {(np.hypot(*info['shift_mm'])):.2f}mm")
    return T, info


def _bd_p95(mask_fit, mask_ref):
    """拟合剪影的边界点到参考边界的距离（px）的中位/p95/最大 + 面积比。

    自检里不用 IoU 当唯一判据：IoU 是**面积**量，对「整条边界平移 1px」和
    「一个角鼓出来」给同一个数；这里直接报到边界的距离，单位是 px，能换算成 mm。
    """
    a = mask_fit.astype(bool)
    b = mask_ref.astype(bool)
    ea = a & ~cv2.erode(a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    eb = b & ~cv2.erode(b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not ea.any() or not eb.any():
        return dict(med=float("nan"), p95=float("nan"), mx=float("nan"),
                    area_ratio=float(a.sum()) / max(float(b.sum()), 1.0))
    d = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)[ea]
    return dict(med=float(np.median(d)), p95=float(np.percentile(d, 95)),
                mx=float(d.max()), area_ratio=float(a.sum()) / max(float(b.sum()), 1.0))


def tilt_study(tilts=(0.0, 10.0, 20.0, 25.0), mesh=None, h_mm=12.4, cam_h_mm=None,
               n_rot=180, verbose=True):
    """up 视图「光轴其实是斜的」对**非球**物体有多疼 —— 合成研究，量出来的。

    背景：真 up 光轴相对竖直大约斜 **20°**（`H_board_to_px` 在像心的雅可比两主方向
    差 1.10）。**球把这件事掩盖了**（球从哪个方向看都一样）；换成有形状的物体，
    斜视会同时压扁轮廓、并让物体的**顶面和底面**按不同比例投影。

    做法（两侧都锚在「物体占画面同样大」上，才比的是几何、不是尺寸）：
      · 参考：相机瞄准物体、光轴**真的斜 β**，物距相同 ⇒ 物体落在主点上
      · 拟合：只给参考的剪影，走 `fit_up`（假定**垂直下视** + 面积锚定 + 绕竖轴一维）
      · 报：两者剪影 IoU、拟合边界到参考边界的 p95/最大距离（px，并换算 mm）、面积比

    ★ 这一项**不能**用 IoU 一句「还行」带过，所以 h 也扫了一遍（见 `h_rows`）。
      ★ **实测结论和「直觉」相反，别照直觉写**：原来这里写的是「矮物体视差小、高的吃亏大」，
        量出来是 h = 6/12.4/20/30mm → p95 = 2.93/3.65/3.66/2.94 mm，**非单调**。
        原因是这个偏差的主项不是几何视差，而是**物体在画面里的横向展幅**：
        倾斜把整个剪影沿倾斜方向剪切，位移 ∝ 展幅 × tan β，跟高度没有直接关系
        （舵机 45.4mm 长边沿底面的那一段才是主导）。所以横轴应该用边长/展幅，不是高度。
        这一栏就留作「h 不是控制变量」的证据。
    """
    g = BS.load_geom()
    mesh = mesh or load_mesh("sts3215_03a_v1_v50_h0.5_m1")
    cam_h = float(cam_h_mm if cam_h_mm is not None else BS.UP_CAM_H_MM)
    W, H = 640, 480
    cell = mesh.cells[-1]
    pose = np.eye(4)                      # 物体系 = base 系（平放在桌上，绕 z 转由 fit 搜）

    rows = []
    print(f"up 倾斜研究：相机名义高 {cam_h:.0f}mm，物体 {mesh.name}"
          f"（{mesh.ext*1000:.1f}mm），桌面处 1px ≈ "
          f"{(cam_h - h_mm) / 800.0:.2f}mm")
    for beta in tilts:
        # 参考相机：位置让物体落在主点上、光轴斜 β（绕 base x 轴）
        d_mm = cam_h - h_mm
        a = exp_so3([np.deg2rad(beta), 0, 0]) @ np.array([0.0, 0, -1.0])   # 光轴方向
        C = np.array([0.0, 0.0, h_mm / 1000.0])
        P = C - a * (d_mm / 1000.0)
        R = exp_so3([np.deg2rad(beta), 0, 0]) @ np.diag([1.0, -1.0, -1.0])
        Tr = np.eye(4)
        Tr[:3, :3] = R
        Tr[:3, 3] = P
        f_ref = 800.0
        K = np.array([[f_ref, 0, W / 2.0], [0, f_ref, H / 2.0], [0, 0, 1.0]])
        cam_ref = MR.Cam(K, np.zeros(5), Tr, W=W, H=H)
        obs = silhouette(mesh_tris_at(mesh, cell, pose), cam_ref)
        if obs.sum() < 20:
            print(f"  β={beta:4.1f}°：物体没进画面（{obs.sum()}px），跳过")
            continue
        T_fit, info = fit_up(mesh, g, obs.astype(np.float32), h_mm=h_mm,
                             cam_h_mm=cam_h, n_rot=n_rot, cell=cell, verbose=False)
        # fit_up 用的相机在 base 里（T_base_cam 由 up_cam_from_area 造），这里重建一次
        T_cam, centre = up_cam_from_area(g, np.array(info["c_px"]), cam_h, h_mm)
        cam_fit = MR.Cam(np.array([[info["f_px"], 0, info["c_px"][0]],
                                   [0, info["f_px"], info["c_px"][1]], [0, 0, 1.0]]),
                         np.zeros(5), T_cam, W=W, H=H)
        m_fit = silhouette(mesh_tris_at(mesh, cell, T_fit), cam_fit)
        bd = _bd_p95(m_fit, obs)
        mm = (cam_h - h_mm) / info["f_px"]                 # 1px 在物体处的 mm（近似）
        rows.append(dict(beta=beta, iou=info["iou"], p95_px=bd["p95"], mx_px=bd["mx"],
                         med_px=bd["med"], area_ratio=bd["area_ratio"], mm_per_px=mm,
                         p95_mm=bd["p95"] * mm, mx_mm=bd["mx"] * mm))
        if verbose:
            print(f"  β={beta:4.1f}°  IoU={info['iou']:.4f}  "
                  f"边界位移 中位 {bd['med']:.2f}px p95 {bd['p95']:.2f}px "
                  f"最大 {bd['mx']:.1f}px ⇒ p95 {bd['p95']*mm:.2f}mm 最大 {bd['mx']*mm:.2f}mm"
                  f"  面积比 {bd['area_ratio']:.3f}")

    # 高度敏感性：同一个 β=20°，h 从小到大。
    # ★ 实测非单调（见 docstring），所以这里只是把「h 不是控制变量」这条证据摆出来，
    #   不要从这栏读出一个趋势。
    print("\n  β=20° 时随物体高度 h 变化（★ 实测**非单调**，h 不是这个偏差的控制变量）：")
    h_rows = []
    for h in (6.0, 12.4, 20.0, 30.0):
        beta = 20.0
        d_mm = cam_h - h
        a = exp_so3([np.deg2rad(beta), 0, 0]) @ np.array([0.0, 0, -1.0])
        C = np.array([0.0, 0.0, h / 1000.0])
        Tr = np.eye(4)
        Tr[:3, :3] = exp_so3([np.deg2rad(beta), 0, 0]) @ np.diag([1.0, -1.0, -1.0])
        Tr[:3, 3] = C - a * (d_mm / 1000.0)
        cam_ref = MR.Cam(np.array([[800.0, 0, W / 2.0], [0, 800.0, H / 2.0], [0, 0, 1.0]]),
                         np.zeros(5), Tr, W=W, H=H)
        obs = silhouette(mesh_tris_at(mesh, cell, pose), cam_ref)
        if obs.sum() < 20:
            continue
        T_fit, info = fit_up(mesh, g, obs.astype(np.float32), h_mm=h, cam_h_mm=cam_h,
                             n_rot=n_rot, cell=cell, verbose=False)
        T_cam, centre = up_cam_from_area(g, np.array(info["c_px"]), cam_h, h)
        cam_fit = MR.Cam(np.array([[info["f_px"], 0, info["c_px"][0]],
                                   [0, info["f_px"], info["c_px"][1]], [0, 0, 1.0]]),
                         np.zeros(5), T_cam, W=W, H=H)
        m_fit = silhouette(mesh_tris_at(mesh, cell, T_fit), cam_fit)
        bd = _bd_p95(m_fit, obs)
        mm = (cam_h - h) / info["f_px"]
        h_rows.append(dict(h=h, iou=info["iou"], p95_px=bd["p95"], p95_mm=bd["p95"] * mm))
        print(f"   h={h:5.1f}mm  IoU={info['iou']:.4f}  p95 {bd['p95']:.2f}px "
              f"= {bd['p95']*mm:.2f}mm  面积比 {bd['area_ratio']:.3f}")
    return dict(rows=rows, h_rows=h_rows)


def axis_resolution(mesh, cam, obs, T, floor, axes=None, hi=45.0, step=0.5,
                    cell=None, verbose=True, tag=""):
    """**单自由度分辨率**：把位姿沿某个轴推开，剪影 IoU 掉到 `floor` 以下的角度/mm。

    报姿态误差时**必须带上这个**，否则会拿「剪影根本看不见的方向」上的差当好坏的证据。
    实测（合成自检，舵机 @249mm、1px=0.61mm）：单轴转 2~3° 就掉到 0.97 以下，
    横向平移 0.5~1mm、深度 4mm —— 也就是说这个目标的**单轴**分辨率就是这么点。
    但**联合**容差大得多（转 5° 再把位置拉回来，损失仍不高于定点解）：
    单张剪影的弱方向不是某一个轴，而是「转 + 平移」的组合。

    返回 `{轴名: (阈值处的值 | None, 最后测的那个值, 最后的 IoU)}`。
    """
    cell = cell or mesh.cells[-1]
    o = obs.astype(np.float32)
    if axes is None:
        axes = [("base x", np.array([1.0, 0, 0])),
                ("base y", np.array([0, 1.0, 0])),
                ("base z", np.array([0, 0, 1.0]))]
    out = {}
    for nm, ax in axes:
        hit, last = None, (0.0, 1.0)
        c0 = T[:3, 3].copy()
        th = step
        while th <= hi + 1e-9:
            # ★ 绕**物体自己**的位置转，不是绕 base 原点 —— 真帧里物体离 base 原点
            #   一百多毫米，绕原点转等于把它扫出画面，量到的是位移不是分辨率。
            R = exp_so3(np.asarray(ax, float) / np.linalg.norm(ax) * np.deg2rad(th))
            Tk = T.copy()
            Tk[:3, :3] = R @ T[:3, :3]
            Tk[:3, 3] = c0 + R @ (T[:3, 3] - c0)
            v = loss_soft(coverage(mesh_tris_at(mesh, cell, Tk), cam, ss=2), o,
                          None)[1]["iou"]
            last = (th, v)
            if v < floor:
                hit = th
                break
            th += step
        out[nm] = (hit, last[0], last[1])
    if verbose:
        print(f"    {tag}单轴分辨率（IoU 掉到 {floor:.3f} 以下）：" + "  ".join(
            f"{nm} {'>' if v[0] is None else ''}{v[0] if v[0] is not None else v[1]:.1f}° "
            f"(末 {v[2]:.3f})" for nm, v in out.items()))
    return out


def _cube_rots():
    """**立方体自己的 24 个旋转**（轴置换 × 符号，行列式 +1）。

    ★ 拿方盒的拟合姿态去和 `held_pose_wrist.json` 比，**必须先把这 24 个商掉**：
      立方体的剪影对其中任何一个都不变，不除就会报出一个假的 90° 差
      （和 `_KLEIN` 那条是一个道理，只是立方体的等价族有 24 个而不是 4 个）。
      重建出来的舵机不是立方体，它的等价族只有 `_KLEIN` 那 4 个 —— 两个别混用。
    """
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, sg)):
                M[i, p] = s
            if np.linalg.det(M) > 0.5:
                out.append(M)
    return out


_CUBE_ROTS = None


def rot_err_cube(R1, R2):
    """朝向差，**模立方体的 24 对称群**（这才是单剪影能为立方体定住的量）。"""
    global _CUBE_ROTS
    if _CUBE_ROTS is None:
        _CUBE_ROTS = _cube_rots()
    return min(rot_err_deg(R1, R2 @ S) for S in _CUBE_ROTS)


def real_cube(frames=(300, 336, 400), dataset="newdata1", edge_mm=21.5,
              fix_s=None, verbose=True, smooth=0.0, keep_masks=False):
    """真帧验证：把**解析方盒**拟到腕相机里的真实方块掩码上，和 `held_pose_wrist.json` 比。

    为什么拿方盒比而不是重建的舵机：真机夹的是方块，只有在**有真值**的物体上比才有意义
    （舵机的真值只存在于合成自检里）。方盒边 = 用户实测 21.5mm；`held_pose_wrist.json`
    那份把边**也当参数解**，得 21.19mm ⇒ 这一比顺带量了「尺度能不能和位姿分开」。

    观测、相机、跟踪器全部走 `held_pose.py` 那条路（`BS.carry_track` + `P6.object_mask` +
    FK⊗手眼 + 模型 δ），**不是**另起一套 —— 否则比的就不是两个拟合器的差，而是两条链的差。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    P6 = _load_module("_fmp_p6", HERE / "pose6d_swap.py")
    g = BS.load_geom()
    ds = LeRobotDataset(dataset, root=str(REPO / "dataset" / dataset))
    dm = HERE / "calib" / "wrist_delta_model.json"
    dmodel = BS.load_delta_model(dm) if dm.exists() else None
    trk = BS.carry_track(ds, min(frames), max(frames) + 10)
    ref = json.loads((HERE / "calib" / "held_pose_wrist.json").read_text())
    T_ref = np.asarray(ref["T_cam_cube"], float)
    e_ref = float(ref["cube_edge_mm"])

    box = MeshObj(MR.box(edge_mm / 1000.0, edge_mm / 1000.0, edge_mm / 1000.0),
                  name=f"cube{edge_mm:.1f}", cells=(2e-3, 1e-3, 0.5e-3))
    print(f"解析方盒 {edge_mm:.2f}mm；参考位姿的边（同一次拟合解出来的）= {e_ref:.2f}mm")
    print(f"参考 T_cam_cube 平移 {np.round(T_ref[:3, 3] * 1000, 2)}mm\n")

    rows = []
    for i in frames:
        fr = ds[i]
        fg = P6.bgr_of(fr["observation.images.front"])
        rec = trk.get(i)
        if rec is None or rec.get("c") is None:
            print(f"  f{i}: 跟踪器没有这一帧的记录，跳过")
            continue
        c = np.asarray(rec["c"], float)
        m, src = P6.object_mask(rec, fg, c)
        obs = (m > 0)
        st = fr["observation.state"].numpy().astype(float)
        norm = {k: float(v) for k, v in zip(A.MOTORS, st)}
        q = A.norm_to_q(norm, g["urdf"])
        Tbc = (g["urdf"].fk(q, "wrist_camera_link") @ g["X"]).copy()
        if dmodel is not None:
            Tbc[:3, 3] += np.asarray(BS.delta_from_model(dmodel, norm), float).reshape(3)
        H, W = fg.shape[:2]
        cam = MR.Cam(g["K"], g["dist"], Tbc, W=W, H=H)
        cam_to_ref = np.linalg.inv(Tbc)

        t0 = time.time()
        Tf, info = fit_wrist(box, cam, obs.astype(np.float32), fix_s=fix_s,
                             verbose=False, seed=i, smooth=smooth)
        Tcf = cam_to_ref @ Tf                                   # 相机系里的位姿
        s_used = info["s"]
        # 参考位姿在**同一帧**的剪影 IoU（同一渲染器、同一观测；参考用自己的边长）
        ref_box = MeshObj(MR.box(e_ref / 1000.0, e_ref / 1000.0, e_ref / 1000.0),
                          name="ref", cells=(2e-3, 1e-3, 0.5e-3))
        cov_ref = coverage(mesh_tris_at(ref_box, ref_box.cells[-1], Tbc @ T_ref),
                           cam, ss=2)
        iou_ref = loss_soft(cov_ref, obs.astype(np.float32), None)[1]["iou"]
        # ★ 两个位姿**彼此**的剪影 IoU：它才是「22° 的差到底是不是差」的裁判 ——
        #   两个位姿各对观测 0.93/0.91，彼此却 >0.9 ⇒ 它们**同形**，22° 只是同一个
        #   等价类里的两个点（立方体在体对角视角附近，轮廓对旋转是二阶不敏感的）。
        iou_ff = float((info["cov"] * cov_ref).sum()
                       / max((info["cov"] + cov_ref - info["cov"] * cov_ref).sum(), 1e-9))
        d = Tcf[:3, 3] - T_ref[:3, 3]
        # ★ 姿态差要**和目标的敏感度一起报**：绕某轴转多少度 IoU 才掉 0.01。
        #   不报这个，22° 的姿态差会被读成「拟合错了 22°」。
        res = axis_resolution(box, cam, obs, Tf, floor=info["iou"] - 0.01,
                              cell=box.cells[-1], verbose=verbose, tag=f"f{i}: ")
        rows.append(dict(frame=i, res=res, dR=rot_err_cube(Tcf[:3, :3], T_ref[:3, :3]),
                         dR_raw=rot_err_deg(Tcf[:3, :3], T_ref[:3, :3]),
                         d=d, dt=float(np.linalg.norm(d) * 1000),
                         iou=info["iou"], iou_ref=iou_ref, iou_fit_vs_ref=iou_ff,
                         edge=edge_mm * s_used,
                         depth=info["depth_mm"], sigma_z=info["sigma_z_mm"],
                         px=int(obs.sum()), t=time.time() - t0, src=src,
                         # 只给出图时留：掩码和覆盖率场都是一整幅，不留就不进内存
                         **({"obs": obs, "cov": info["cov"], "cov_ref": cov_ref,
                             "fg": fg} if keep_masks else {})))
        if verbose:
            print(f"  f{i:3d}  位置差 {np.linalg.norm(d)*1000:5.2f}mm "
                  f"(x {d[0]*1000:+5.2f} y {d[1]*1000:+5.2f} z {d[2]*1000:+5.2f})  "
                  f"朝向差 {rows[-1]['dR']:5.2f}°（不商对称 {rows[-1]['dR_raw']:6.2f}°）  "
                  f"边 {edge_mm*s_used:5.2f}mm  深度 {info['depth_mm']:5.1f}mm  "
                  f"IoU 拟合 {info['iou']:.4f} vs 参考 {iou_ref:.4f} "
                  f"（两者彼此 {iou_ff:.4f}）  "
                  f"({rows[-1]['px']}px, {rows[-1]['t']:.0f}s)")

    if rows:
        dt = np.array([r["dt"] for r in rows])
        dr = np.array([r["dR"] for r in rows])
        io = np.array([r["iou"] for r in rows])
        ir = np.array([r["iou_ref"] for r in rows])
        print(f"\n汇总 {len(rows)} 帧：位置差 中位 {np.median(dt):.2f}mm 最大 {dt.max():.2f}mm"
              f"｜朝向差 中位 {np.median(dr):.2f}° 最大 {dr.max():.2f}°"
              f"（模立方体 24 对称群）")
        print(f"         剪影 IoU：拟合 中位 {np.median(io):.4f} vs 参考 中位 {np.median(ir):.4f}")
    return rows


# ================================================================ Stage C 出图

def fig_pose_recovery(verbose=True):
    """① 合成自检：位置/深度误差 vs 观测噪声 σ，叠上真值 IoU 与拟合 IoU 的对照。

    这张图的用途是**接受/拒绝这个拟合器**：位置误差要远小于 `σ_z` 预测那条线，
    且拟合 IoU 不能比真值自己的 IoU 明显低（低了就是优化器没走到，不是目标的问题）。
    """
    plt = _mpl()
    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    servo = load_mesh("sts3215_03a_v1_v50_h0.5_m1")
    Tg = np.eye(4)
    Tg[:3, :3] = np.diag([1.0, -1.0, -1.0])
    Tg[:3, 3] = [-0.01, 0.0, 0.25]
    rng = np.random.default_rng(0)
    n = 9
    rows = []
    for k in range(n):
        Ttrue, camk, obs_k, sig = _trial_scene(servo, rng, k, K, Tg)
        Tf, info = fit_wrist(servo, camk, obs_k, verbose=False, seed=k)
        iou_true = loss_soft(coverage(mesh_tris_at(
            servo, servo.cells[-1], Ttrue), camk, ss=2), obs_k.astype(np.float32),
            None)[1]["iou"]
        rows.append(dict(k=k, sig=sig, dt=np.linalg.norm(Tf[:3, 3] - Ttrue[:3, 3]) * 1000,
                         dz=info["depth_mm"] - np.linalg.norm(Ttrue[:3, 3] - Tg[:3, 3]) * 1000,
                         dR=rot_err_deg(Tf[:3, :3], Ttrue[:3, :3], modulo_symmetry=True),
                         iou=info["iou"], iou_true=iou_true, sz=info["sigma_z_mm"]))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
    x = np.arange(n)
    for ax in axes:
        _ax_style(ax)
    axes[0].bar(x, [r["dt"] for r in rows], color=S1, width=0.62, zorder=3)
    axes[0].plot(x, [r["sz"] for r in rows], "o", ms=7, mfc=S2, mec=SURF, mew=1.3,
                 zorder=4, label="σ_z 解析预测 $\\sigma_{px}z^2/(fW)$")
    axes[0].set_xlabel("合成试次（σ 按 0/0.5/1.5px 轮转）")
    axes[0].set_ylabel("位置误差 (mm)")
    axes[0].set_title("位置误差落在 σ_z 预测的同一量级", color=INK1, fontsize=10.5, pad=8)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center")

    axes[1].plot(x, [r["iou_true"] for r in rows], "o", ms=8, mfc=S2, mec=SURF, mew=1.3,
                 zorder=4, label="真值位姿自己的 IoU")
    axes[1].plot(x, [r["iou"] for r in rows], "o", ms=6, mfc=S1, mec=SURF, mew=1.3,
                 zorder=5, label="拟合位姿的 IoU")
    for r in rows:
        axes[1].plot([r["k"], r["k"]], [r["iou"], r["iou_true"]], color=AXIS, lw=1.0,
                     zorder=2)
    axes[1].set_xlabel("合成试次")
    axes[1].set_ylabel("剪影 IoU")
    axes[1].set_title("拟合 IoU 处处不低于真值（9/9）", color=INK1, fontsize=10.5, pad=8)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

    axes[2].axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    axes[2].plot(x, [r["dR"] for r in rows], "o", ms=7, mfc=S1, mec=SURF, mew=1.3, zorder=4)
    for r in rows:
        if r["dt"] > 20:
            axes[2].annotate(f"trial {r['k']}：走到另一个盆地\n({r['dt']:.0f}mm)",
                             (r["k"], r["dR"]), textcoords="offset points",
                             xytext=(-8, -6), ha="right", color=INK2, fontsize=8.2)
    axes[2].set_xlabel("合成试次")
    axes[2].set_ylabel("朝向差 (度，已商掉长方体的翻转族)")
    axes[2].set_title("朝向：8/9 在 10° 内，1 次落到远处的盆地",
                      color=INK1, fontsize=10.5, pad=8)
    fig.suptitle("合成自检：已知位姿 → 渲剪影 → 拟回去（舵机网格，面积定深初值 + 三级模式搜索）",
                 color=INK1, fontsize=12, y=0.99, x=0.012, ha="left")
    p = _save(fig, "fit_mesh_pose_recovery.png",
              "位置 vs σ_z 预测 / IoU 对照 / 朝向误差")
    return rows, p


def fig_flat_direction(verbose=True):
    """② **单张剪影的弱方向**：真值盆地的行程 vs 远处盆地的行程（trial 7）。

    这是 Stage C 最重要的一张「诚实图」：优化器跑掉的那一次，不是它不收敛，
    而是**真值盆地非常窄**（亚毫米级）而远处的另一个盆地又宽又深。
    横轴是「转 1° 的同时沿视线走多少 mm」这条组合方向上的位移。
    """
    plt = _mpl()
    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    servo = load_mesh("sts3215_03a_v1_v50_h0.5_m1")
    Tg = np.eye(4)
    Tg[:3, :3] = np.diag([1.0, -1.0, -1.0])
    Tg[:3, 3] = [-0.01, 0.0, 0.25]
    rng = np.random.default_rng(0)
    tr = {}
    for k in range(9):
        tr[k] = _trial_scene(servo, rng, k, K, Tg)
    Ttrue, camk, obs_k, sig = tr[7]
    Tf, info = fit_wrist(servo, camk, obs_k, verbose=False, seed=7)
    dt = float(np.linalg.norm(Tf[:3, 3] - Ttrue[:3, 3]) * 1000)

    a = flat_direction(servo, camk, obs_k, Ttrue, span_mm=30.0, tag="[真值] ")
    b = flat_direction(servo, camk, obs_k, Tf, span_mm=30.0, tag="[拟合] ")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    _ax_style(axes[0])
    _ax_style(axes[1])
    for row, col, nm in ((a, S2, f"真值盆地（转 1° 走 {a['mm_per_deg']:.2f}mm）"),
                         (b, S1, f"远处那个盆地（转 1° 走 {b['mm_per_deg']:.2f}mm）")):
        xs = [t[0] for t in row["profile"]]
        ys = [t[1] for t in row["profile"]]
        axes[0].plot(xs, ys, "-", color=col, lw=2.0, zorder=4, label=nm)
        axes[0].plot([0], [row["loss0"]], "o", ms=7, mfc=col, mec=SURF, mew=1.3, zorder=5)
    axes[0].set_xlabel("沿最平方向走得离起点多远（按「沿线平移 mm」计）")
    axes[0].set_ylabel("软 IoU 损失 $1-|M\\cap O|/|M\\cup O|$")
    axes[0].set_title("沿各自的最平方向切一刀：真值那边又窄又浅",
                      color=INK1, fontsize=10.5, pad=8)
    ymax = max(max(t[1] for t in a["profile"]), max(t[1] for t in b["profile"]))
    axes[0].set_ylim(min(min(t[1] for t in a["profile"]),
                         min(t[1] for t in b["profile"])) * 0.97, ymax * 1.04)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

    lab = ["真值盆地\n(转 1° 需走 " + f"{a['mm_per_deg']:.2f}mm)",
           "远处盆地\n(转 1° 需走 " + f"{b['mm_per_deg']:.2f}mm)"]
    lo = [a["flat_mm"][0], b["flat_mm"][0]]
    hi = [a["flat_mm"][1], b["flat_mm"][1]]
    xx = np.arange(2)
    axes[1].bar(xx, np.array(hi) - np.array(lo), bottom=lo, color=[S2, S1], width=0.5,
                zorder=3)
    axes[1].axhline(0, color=MUTED, lw=1.0, zorder=2)
    for i, (l, h) in enumerate(zip(lo, hi)):
        axes[1].annotate(f"−{abs(l):.1f} / +{h:.1f} mm", (i, (l + h) / 2),
                         textcoords="offset points", xytext=(40, 0), va="center",
                         color=INK2, fontsize=9)
    axes[1].set_xticks(xx, lab)
    axes[1].set_ylabel("损失涨 ≤0.02 的行程 (mm)")
    axes[1].set_title(f"两个盆地的宽度：拟合位姿离真值 {dt:.1f}mm，"
                      f"而它落在一个更宽的谷里", color=INK1, fontsize=10.5, pad=8)
    fig.suptitle("单张剪影的「转 + 沿线平移」组合弱方向 —— 优化器跑掉不是不收敛，"
                 "是目标的简并", color=INK1, fontsize=12, y=0.99, x=0.012, ha="left")
    p = _save(fig, "fit_mesh_pose_flat_direction.png",
              "最平方向的损失剖面：真值盆地窄、错盆地宽")
    return dict(truth=a, fit=b, dt_mm=dt), p


def _overlay_ax(ax, bgr, lit, lit_name, title, edge_col):
    """在一帧腕视图上叠剪影：观测边界（橙）vs 渲染边界（蓝）。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb, alpha=0.85)
    for m, col in lit:
        e = (m.astype(bool)
             & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        ys, xs = np.nonzero(e)
        ax.plot(xs, ys, ".", ms=1.1, color=col, label=None, rasterized=True)
    ax.set_title(title, color=INK1, fontsize=10.5, pad=6)
    ax.axis("off")
    ax.plot([], [], ".", ms=6, color=edge_col[0], label=lit_name[0])
    ax.plot([], [], ".", ms=6, color=edge_col[1], label=lit_name[1])
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right",
              handletextpad=0.2, borderpad=0.2)


def fig_real_frames(rows, verbose=True):
    """③ 真帧：腕相机上「观测边界 vs 拟合方盒边界」，逐帧报位置差与 IoU。

    观测和相机链全部走 `held_pose.py` 那条，**不另起一套**，否则比的不是拟合器。
    """
    plt = _mpl()
    rs = [r for r in rows if "obs" in r]
    if not rs:
        print("[图] 真帧图跳过：没有留掩码（要 keep_masks=True）")
        return None
    fig, axes = plt.subplots(1, len(rs), figsize=(4.6 * len(rs), 3.9))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rs):
        _overlay_ax(ax, r["fg"], [(r["obs"], S2), (r["cov"] > 0.5, S1)],
                    ("观测边界", "拟合边界"),
                    f"f{r['frame']}：位置差 {r['dt']:.2f}mm｜IoU {r['iou']:.3f} "
                    f"vs 参考 {r['iou_ref']:.3f}",
                    (S2, S1))
    fig.suptitle("真帧验证（腕视图，解析方盒）：拟合位姿的剪影 vs 真实掩码",
                 color=INK1, fontsize=12, y=0.99, x=0.012, ha="left")
    p = _save(fig, "fit_mesh_pose_real_frames.png",
              "腕视图逐帧的剪影叠图（观测 vs 拟合）")
    return p


def fig_tilt(rows, h_rows, verbose=True):
    """④ up 视图倾斜研究 —— **显眼地报**：用户最初的抱怨就是 up 视图。

    三块**各自一个纵轴**（IoU / 边界位移 / h 扫描），不做双纵轴：IoU 是无量纲、
    位移是 mm，两个标度挤一张图会让人按视觉斜率读结论。
    """
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for ax in axes:
        _ax_style(ax)
    b = [r["beta"] for r in rows]

    axes[0].plot(b, [r["iou"] for r in rows], "-o", color=S1, lw=2.2, ms=8, mfc=S1,
                 mec=SURF, mew=1.3, zorder=4)
    axes[0].set_xlabel("up 相机光轴相对竖直的倾角 β (度)")
    axes[0].set_ylabel("剪影 IoU（面积锚定后）")
    axes[0].set_ylim(0.7, 1.02)
    axes[0].set_title("「垂直下视」假定：β=20° 时 IoU 掉到 0.82",
                      color=INK1, fontsize=10.5, pad=8)
    axes[0].annotate("球看不出这件事：球从哪个方向看都一样",
                     (0.97, 0.88), xycoords="axes fraction", ha="right",
                     color=INK2, fontsize=8.6)

    axes[1].plot(b, [r["p95_mm"] for r in rows], "-o", color=S2, lw=2.2, ms=8, mfc=S2,
                 mec=SURF, mew=1.3, zorder=4, label="p95")
    axes[1].plot(b, [r["mx_mm"] for r in rows], "--s", color=S4, lw=1.8, ms=7, mfc=S4,
                 mec=SURF, mew=1.2, zorder=3, label="最大")
    axes[1].set_xlabel("倾角 β (度)")
    axes[1].set_ylabel("拟合边界到参考边界的位移 (mm)")
    axes[1].set_title("同一个偏差换算成物体处的毫米：p95 ~3.6mm",
                      color=INK1, fontsize=10.5, pad=8)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

    hh = [r["h"] for r in h_rows]
    axes[2].plot(hh, [r["p95_mm"] for r in h_rows], "-o", color=S3, lw=2.2, ms=8,
                 mfc=S3, mec=SURF, mew=1.3, zorder=4)
    axes[2].axhline(np.mean([r["p95_mm"] for r in h_rows]), color=MUTED, lw=1.2,
                    ls=(0, (4, 3)), zorder=2)
    axes[2].set_xlabel("物体高度 h (mm)（β 固定 20°）")
    axes[2].set_ylabel("边界位移 p95 (mm)")
    axes[2].set_title("★ 实测**非单调**：h 不是这个偏差的控制变量",
                      color=INK1, fontsize=10.5, pad=8)
    axes[2].annotate("主导项是物体在画面里的横向展幅 × tan β，\n不是几何视差",
                     (0.5, 0.05), xycoords="axes fraction", ha="center",
                     color=INK2, fontsize=8.6)
    fig.suptitle("up 视图：非球物体把「光轴其实斜 ~20°」暴露出来 —— 这是误差棒，不是 bug",
                 color=INK1, fontsize=12, y=0.99, x=0.012, ha="left")
    p = _save(fig, "fit_mesh_pose_tilt.png", "up 镜头倾斜对非球物体的代价")
    return p


def figures_stage_c(verbose=True):
    """把 Stage C 的四张图一次出齐（都进仓库 `figures/`）。"""
    rows_recovery, p1 = fig_pose_recovery(verbose)
    _, p2 = fig_flat_direction(verbose)
    rr = real_cube(frames=(300, 336, 400), fix_s=1.0, verbose=verbose, keep_masks=True)
    p3 = fig_real_frames(rr, verbose)
    tl = tilt_study(verbose=verbose)
    p4 = fig_tilt(tl["rows"], tl["h_rows"], verbose)
    return dict(recovery=rows_recovery, real=rr, tilt=tl,
                figs=[p1, p2, p3, p4])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--mesh", default="sts3215_03a_v1_v50_h0.5_m1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--real", action="store_true",
                    help="真帧验证：解析方盒拟到腕相机真实掩码，对比 held_pose_wrist.json")
    ap.add_argument("--figs", action="store_true",
                    help="出 Stage C 的四张图（进 figures/）")
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--frames", default="300,336,400")
    ap.add_argument("--edge-mm", type=float, default=21.5)
    ap.add_argument("--free-s", action="store_true", help="边长也当参数解（默认钉死 21.5mm）")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="模型/观测同核模糊的 σ（px）；0=不模糊，见 fit_wrist 的 docstring")
    ap.add_argument("--tilt", action="store_true",
                    help="up 视图倾斜研究：光轴真的斜 β 时，「垂直下视」拟合差多少")
    a = ap.parse_args()
    if a.selftest:
        selftest(n_trials=a.trials, smooth=a.smooth)
        return 0
    if a.figs:
        figures_stage_c()
        return 0
    if a.real:
        real_cube(frames=[int(x) for x in a.frames.split(",") if x.strip()],
                  dataset=a.dataset, edge_mm=a.edge_mm,
                  fix_s=None if a.free_s else 1.0, smooth=a.smooth)
        return 0
    if a.tilt:
        tilt_study(mesh=(load_mesh(a.mesh) if a.mesh != "box" else
                         MeshObj(MR.box(a.edge_mm / 1000.0,
                                        a.edge_mm / 1000.0,
                                        a.edge_mm / 1000.0),
                                 name=f"cube{a.edge_mm:.1f}",
                                 cells=(2e-3, 1e-3, 0.5e-3))))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
