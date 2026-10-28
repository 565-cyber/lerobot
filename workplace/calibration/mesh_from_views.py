#!/usr/bin/env python
"""剪影雕刻重建（visual hull）—— 从**多视角剪影**恢复出物体的三角网格。

为什么要有它：`pose6d_swap.py` 里注入的那个物体一直是 `MR.uv_sphere(12mm)` ——
一个**参数方程的解析球**，半径来自卡尺。也就是说「图像→三维模型」这一步根本没发生，
图像只在**位姿**估计里参与过（`place_ball` 的剪影拟合、`up_local_cam` 的反解焦距）。
对球而言这步本来也退化（球网格没有自由参数，重建只能恢复半径，而卡尺比 24px 剪影准），
所以要把注入物换成形状不可解析的物体（这里是 SO101 的舵机 `sts3215_03a_v1.stl`），
真的走一遍重建，并且**拿它的 CAD 当真值**报绝对误差。

## 判据：为什么是「逐视角符号距离取 min」

每个视角 i 有一张剪影掩码。定义

    Φ_i(x) = +d_in(proj_i x)   若 proj_i x 落在剪影内   （d_in  = 到剪影边界的内距）
             -d_out(proj_i x)  若落在剪影外            （d_out = 到剪影边界的外距）

    f(x) = min_i Φ_i(x)      ； 里面 ⟺ f(x) ≥ 0

`{f ≥ 0} = ∩_i cone_i` **恰好**是 visual hull —— 这是**集合恒等式，不是近似**。
理由：`Φ_i ≥ 0 ⟺ proj_i(x) ∈ 剪影_i`，不管 Φ_i 是不是度量距离，起作用的只有它的
**符号**和**零集**，而这两样是精确的。

**必须写明的两条**（否则会误用它）：

1. `f` **不是**三维欧氏距离场。`|∇Φ_i| ≈ z/f`（随深度与方向变），所以
   （a）`f` 只有 C⁰（两个锥面边界相交的脊上是折的）—— marching 只用符号 + 线性插值，
   够用；（b）网格边上的线性插值交点**不等于**精确表面点，偏差 `O(h²)`（体素 h=0.5mm
   时远小于信息极限），见 `--voxel-mm` 的收敛性扫描。
2. `margin` 是**角度/像素**意义上的腐蚀，不是度量腐蚀：在 z=250mm、f=409.6 时
   **1px ≡ 0.61mm**。也就是说「把剪影内缩 margin 像素」等于「把表面沿视线退回
   `margin·z/f` 米」。这正好是我们想要的（像素级剪影的置信度大致与深度无关），
   但报 `margin` 时必须同时报它的度量当量。`validate_carve.py` 会把这条预测当
   **可证伪的直线**去验证。

## 抽取：marching **tetrahedra**，而且只有 3 类

每个立方体切成 6 个四面体。**拓扑只有 3 类**，看 4 个顶点里有几个在里面：

    0 或 4 个 → 0 个面 ； 1 或 3 个 → 1 个面 ； 2 个 → 2 个面（四边形一分为二）

所以**不需要 16 项表，更不需要 marching cubes 那 256 项**。

真正的坑不是表，是**共形性**：每个立方体都用同一条体对角线，会让共享面上两侧诱导出的
对角线**方向相反**，鞍点处两边对「面是否被穿」判断不一致 ⇒ 开裂。解法是
**按奇偶交替的 Kuhn 分解**：立方体 (i,j,k) 在 (i+j+k) 为偶时用对角线 000–111、
为奇时用 111–000。相邻立方体奇偶必反 ⇒ 共享面诱导的对角线重合 ⇒ 全局水密。

**winding 不靠推导、靠场定向**：每个三角面算出来后，在 `质心 ± ε·n` 处插值 `f`。
本文件的约定是 `f` **内正外负**，所以朝外的法线必须满足 `f(+εn) < f(−εn)`；
不满足就把顶点 1、2 对调。这条规则直接就是「外法线」的定义，不需要逐类推导绕序。
它自身不自证，所以另有**两条独立的**检查（见 `mesh_checks`）：

* **有向边计数**：一致定向的闭合流形上，每条有向边恰好出现一次。局部绕序翻掉、
  或者有裂缝，都会让这条计数出现异常 —— 与场定向规则**无关**，能真正抓住错误。
* **有符号体积** 必须 > 0，且与体素计数体积在 1% 内吻合。

（打印里的 `winding_ok` 是场定向规则自身的通过率，**由构造保证**，只当提示看，
不当证据 —— 证据是上面两条。）

## 已知极限（要如实写进交付）

visual hull 永远**只多不少**：`真物体 ⊆ VH ⊆ 凸包`，误差是**单边**的（只会高估）。
体积比的上限是 `1/凹度`，其中 `凹度 = V_真/V_凸包`（实测 SO101 零件从 0.19 到 0.90）。
所以体积比这个指标必须**和凹度一起报**，否则一条好管线在凹件上会显得很差。

跑法：
    python workplace/calibration/mesh_from_views.py --selftest
    python workplace/calibration/mesh_from_views.py --part sts3215_03a_v1 --views 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures"
OUT = HERE / "calib" / "recon"
ASSETS = REPO / "workplace/so101_urdf/assets"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_module(name: str, path: Path):
    """按**文件路径**导入（与 meshrender/pose6d_swap 同一惯例）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MR = _load_module("_mfv_mr", HERE / "meshrender.py")


# --------------------------------------------------------------- 符号距离图

def sdf_view(mask, margin=0.0, pad=4):
    """剪影掩码 → 相机系符号距离图 `(H,W) float32`，**正 = 掩码内**，单位 px。

    `margin`（px）：把判据整体减掉，等价于把剪影内缩 margin 像素 —— 亦即把表面沿视线
    退回 `margin·z/f` 米（见模块 docstring 的第 2 条）。

    先给掩码四周垫一圈 0 再算距离：`cv2.distanceTransform` 不把图像外的像素算进去，
    掩码贴到画面边时内距会被低估。垫完再裁掉，坐标原点不变。
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim != 2:
        raise ValueError(f"mask 必须是 2 维，收到 {m.shape}")
    mp = np.pad(m, int(pad), constant_values=0)
    din = cv2.distanceTransform(mp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dout = cv2.distanceTransform(1 - mp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    phi = np.where(mp > 0, din, -dout).astype(np.float32) - float(margin)
    return phi[int(pad):-int(pad), int(pad):-int(pad)]


def touches_border(mask):
    """剪影是否贴到画面边（贴=被画幅裁过，那一侧的锥面是假的）。"""
    m = np.asarray(mask) > 0
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


# ------------------------------------------------------------------ 视角容器

def make_view(T_world_cam, K, dist, mask, W, H, margin=0.0, near=0.02, name="",
              phi_noise=None):
    """一个视角。`T_world_cam` = **相机在世界系里的位姿**（cam→world），与
    `MR.Cam.T_base_cam` 同义，这里把 `base` 叫 `world` 是因为采集时世界系是**标定板系**。

    `phi_noise`（px，`(H,W)` 或可广播）：**直接加到符号距离场上**。因为 `sdf_view` 返回的
    是真的像素距离场，加一个随机场 = 把零水平集沿法向连续平移，所以能给出**亚像素**的边界
    扰动，而且是低频连续的（不是逐像素白噪声）。见 `_boundary_noise`。

    返回 dict（不用 `MR.Cam`，因为雕刻只用到「世界→相机」和 K/dist，不需要 z-buffer）。
    """
    Twc = np.asarray(T_world_cam, float)
    m = np.asarray(mask) > 0
    phi = sdf_view(m, margin=margin)
    if phi_noise is not None:
        phi = phi + np.asarray(phi_noise, np.float32)
    return dict(name=name, T_world_cam=Twc, T_cam_world=np.linalg.inv(Twc),
                K=np.asarray(K, float), dist=np.asarray(dist, float).reshape(-1),
                phi=phi, W=int(W), H=int(H),
                near=float(near), clipped=touches_border(m),
                # `mask` 是**没加噪声场**的真值剪影，留着给验证脚本做图像域审计用
                # （`phi` 已经被 margin 和 phi_noise 改过，反推不回原始掩码）。
                mask=m,
                border_px=int(m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum()))


def _boundary_noise(H, W, sigma_px, rng, smooth_px=6.0):
    """平滑的低频随机场，单位 px，标准差 = `sigma_px`。

    为什么不用「按 σ 抖动腐蚀半径」那种做法（早期实现就是，已废弃）：腐蚀核的半径必须是
    **整数**，σ=0.5px 时 `round(|a|)` 有 ~68% 的概率取 0 ⇒ **三分之二的视角什么都没发生**，
    剩下三分之一内缩/外扩整整 1px。那是一族双峰的、非亚像素的抖动，量出来的「σ=0.5px 敏感度」
    其实主要在测「1px 的 32% 概率发生」。
    正确做法是扰动距离场：`φ += N(0,σ)`（先做空间平滑 ⇒ 沿边界连续，符合「随机采样的
    分割边界」这个物理图像）。零水平集沿法向平移 σ px，亚像素、连续。
    """
    if sigma_px <= 0:
        return None
    n = rng.standard_normal((int(H), int(W))).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), float(smooth_px))
    s = float(n.std())
    return n * (float(sigma_px) / s if s > 1e-9 else 0.0)


# ---------------------------------------------------------------------- 雕刻

def _grid_axes(bbox_min, bbox_max, h):
    lo = np.asarray(bbox_min, float)
    hi = np.asarray(bbox_max, float)
    if np.any(hi <= lo):
        raise ValueError(f"包围盒空：lo={lo} hi={hi}")
    n = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    return lo, [lo[i] + h * np.arange(int(n[i])) for i in range(3)]


def carve(views, bbox_min, bbox_max, h, planes=(), chunk=2_000_000, verbose=False):
    """逐视角取 min，得到占据场 `f`（内正外负）。

    `planes`: 可选的**世界系半空间**约束序列 `(n, d)`，含义「内部满足 `dot(n, x) <= d`」。
      它的符号距离（米）是 `d - dot(n, x)`，与像素制的 Φ 一起取 min —— 混合单位对
      **符号与零集**没有影响（那才是判据），但 `margin` 对它就是**米**而不是像素。

      实物摆在板子上时**底面永远拍不到**，VH 会把它往下挤成柱体；板面是一个已知平面，
      把它加进来就把这个缺口补掉了 —— 这是采集协议里免费的一条先验，见 CAPTURE_OBJ.md。

    返回 `(f, origin, axes)`，`f` 形状 `(nx,ny,nz)`，`origin` 是 (0,0,0) 格点的世界坐标。
    """
    lo, ax = _grid_axes(bbox_min, bbox_max, h)
    shape = (len(ax[0]), len(ax[1]), len(ax[2]))
    f = np.full(shape, np.inf, np.float32)
    X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)

    for pl in planes:
        n, d = np.asarray(pl[0], float), float(pl[1])
        n = n / max(np.linalg.norm(n), 1e-12)
        np.minimum(f, (d - X @ n).reshape(shape).astype(np.float32), out=f)

    for v in views:
        if v["clipped"] and verbose:
            print(f"    ⚠ 视角 {v['name']} 的剪影贴到画面边（{v['border_px']}px）"
                  f" —— 被裁过的那一侧锥面是假的")
        phi = np.full(len(X), -1e4, np.float32)      # 默认「在外」（投影跑到画幅外就是在外）
        Tcw = v["T_cam_world"]
        for s in range(0, len(X), chunk):
            Xs = X[s:s + chunk]
            Pc = Xs @ Tcw[:3, :3].T + Tcw[:3, 3]
            ok = Pc[:, 2] > v["near"]                # 相机背后的点直接判在外
            if not ok.any():
                continue
            uvp, _ = cv2.projectPoints(Pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                       v["K"], v["dist"])
            uv = uvp.reshape(-1, 2)
            samp = ndimage.map_coordinates(v["phi"], [uv[:, 1], uv[:, 0]], order=1,
                                           mode="constant", cval=-1e4)
            phi[s:s + chunk][ok] = samp
        np.minimum(f, phi.reshape(shape), out=f)

    return f, lo, np.asarray(h, float)


def carve_auto(views, h, centre, hint_half, coarse=4, pad_coarse=1, pad_vox=3,
               planes=(), grow=1.6, max_rounds=3, verbose=False):
    """粗到细雕刻：先粗格找占据体，再用细格只雕它的包围盒。返回 `(f, origin, h, info)`。

    **为什么可以这么裁**（这是集合包含，不是近似）：`visual hull ⊆ bbox(S)`。
    代价是显著的，`carve` 的开销 ∝ 体素数 × 视角数：舵机 177³ = 5.5M 格 → 约 100³ = 1.0M。

    ⚠️ **更正**（`validate_carve.py` 实测后改的，别再把旧理由抄回来）：这里原先写的理由是
    `VH ⊆ conv(S) ⊆ bbox(S)`，其中第二项对、**第一项是错的**。`VH ⊄ conv(S)`：
    取两个正交视角看一个球，点 `(0.7r, 0.7r, 0.7r)` 到两条轴的距都 = 0.99r < r
    （在两个圆柱里）但模长 1.21r > r（在球外）。正确的事实是
      · `VH ⊇ S` 恒成立；`VH = VH(conv(S))`（只依赖凸包）；
      · `VH ⊇ conv(S)` 只在**剪影为凸**时成立（凸锥才含凸包）；
      · `VH ⊆ conv(S)` **一般不成**，因为切锥是**外切**于凸包的，稀疏视角下交出来的
        多面体在角上会拱出凸包外面。
    实测（舵机，N=4/8/50）：`V_R/V_G` = 1.395 / 1.133 / 1.015，而 `1/凹度` = 1.109
    —— 视角少时**越过了**凸包上限，视角密时才落回下面。所以「VH 不会跑出凸包」不是定理，
    而是**稠密视角下的极限**。
    本项目真正保证正确性的是下面那个**重试闸门**（细格若贴到自己的边走就放大重来），
    它是经验检查、不依赖上面任何一条包含关系。裁到凸包外的风险因此被闸门兜住，
    但**推导不能写成不含凸包那一项**。

    **这个裁法的正确性依赖「粗格不漏掉任何占据区域」**，所以：
      · 粗格做一次 3×3×3 的**膨胀**再取包围盒（宁可多给一格，不可少给）；
      · 粗格若贴到提示框的边，说明提示框太小 → 按 `grow` 放大重来；
      · 细格若贴到自己的边，同样重来（正常情况下 `pad_coarse + pad_vox` 有 ~5mm 余量）。
    粗格的体素边长是 `coarse·h`：舵机 2mm 远小于它的最小尺寸 24.8mm，所以漏检风险可忽略；
    **薄片状物体要自己把 `coarse` 调小**，这个函数不会替你发现（粗格漏掉的薄片不会让细格贴边，
    因而逃过检查）。`info` 里带回粗/细格形状与是否发生过放大，便于事后核对。
    """
    centre = np.asarray(centre, float)
    half = float(hint_half)
    hc = h * int(coarse)
    info = dict(coarse_factor=int(coarse), coarse_vox_mm=hc * 1e3, rounds=0,
                grew=False, coarse_shape=None, fine_shape=None)
    for rnd in range(int(max_rounds)):
        info["rounds"] = rnd + 1
        fc, oc, _ = carve(views, centre - half, centre + half, hc,
                          planes=planes, verbose=False)
        occ = ndimage.binary_dilation(fc >= 0, np.ones((3, 3, 3), bool))
        info["coarse_shape"] = tuple(int(s) for s in occ.shape)
        if not occ.any():
            half *= grow
            info["grew"] = True
            if verbose:
                print(f"    [carve_auto] 第{rnd+1}轮：提示框里没有占据体，放大到 "
                      f"±{half*1e3:.0f}mm")
            continue
        if touches_border(occ):
            half *= grow
            info["grew"] = True
            if verbose:
                print(f"    [carve_auto] 第{rnd+1}轮：粗占体贴到提示框边，放大到 "
                      f"±{half*1e3:.0f}mm")
            continue
        idx = np.argwhere(occ)
        m = (pad_coarse * hc + pad_vox * h)
        lo = oc + hc * idx.min(0) - m
        hi = oc + hc * idx.max(0) + m
        f, o, _ = carve(views, lo, hi, h, planes=planes, verbose=False)
        info["fine_shape"] = tuple(int(s) for s in f.shape)
        if touches_border(f >= 0):
            m *= 2.0
            lo = oc + hc * idx.min(0) - m
            hi = oc + hc * idx.max(0) + m
            f, o, _ = carve(views, lo, hi, h, planes=planes, verbose=False)
            info["fine_shape"] = tuple(int(s) for s in f.shape)
            info["grew"] = True
            if verbose:
                print(f"    [carve_auto] 第{rnd+1}轮：细占体贴边，余量加倍后重雕")
            if touches_border(f >= 0):
                raise RuntimeError(
                    "carve_auto：细格占据体仍然贴边 —— 粗格漏检了区域（薄片物体请调小 coarse）")
        if verbose:
            n_c = int(np.prod(info["coarse_shape"]))
            n_f = int(np.prod(info["fine_shape"]))
            print(f"    [carve_auto] 粗 {info['coarse_shape']} ({hc*1e3:g}mm, {n_c/1e6:.2f}M 格)"
                  f" → 细 {info['fine_shape']} ({h*1e3:g}mm, {n_f/1e6:.2f}M 格)，"
                  f"{info['rounds']} 轮")
        return f, o, np.asarray(h, float), info
    raise RuntimeError(f"carve_auto：{max_rounds} 轮都放不下提示框（最后 ±{half*1e3:.0f}mm）")


# ------------------------------------------------------------------ 等值面抽取

# Kuhn 分解：6 个四面体，全部共用体对角线 000–111
_TET_E = np.array([[(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
                   [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)],
                   [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)],
                   [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)],
                   [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)],
                   [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]], np.int32)
_TET_O = 1 - _TET_E                     # 奇偶交替：镜像成对角线 111–000
_TET_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], np.int32)

# 退化面判据：三角面「两倍面积」的上限（`|cross| = 2·面积`，单位 m²）。
# **必须是 0.0，不能放宽** —— 这条踩过坑，实测数字一并记在这：
#   · `0.0`：只丢**逐位重合**（两个交点完全相同）的零面积面。丢它们安全：三角形
#     (a,a,c) 的三条有向边是 (a,a)、(a,c)、(c,a)，(a,c) 与 (c,a) 成对消失，
#     所以丢完仍然配对；不丢反而留下自环 (a,a) 让「每条有向边恰好一次」失败。
#   · `1e-12`（曾经的写法，想「顺手清掉纳米级的碎片」）：**会真的开洞**。
#     实测解析球上多出 328 条边界边、聚成 67 个 5~7 顶点的小环。原因：格点场值
#     |f| 微小但**非零**时（不是恰好 0），同一顶点引出的两条棱交点相距约 1e-9 m
#     —— 面积确实小到 1e-12 量级，但它们在拓扑上是**必需**的，因为相邻三角形的
#     边指着它们配对。丢掉一侧，另一侧就成了开口边。
#   ⇒ 结论：小面积 ≠ 可丢。只有**零面积**才可丢。
_DEGEN_AREA2 = 0.0


def _field_at(tris, f, origin, h, eps):
    """在三角面 `质心 ± eps·n` 处三线性插值 `f`，返回 `(f_plus, f_minus)`。"""
    cen = tris.mean(1)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-15)
    idx = lambda p: (p - origin) / h                                   # noqa: E731
    fp = ndimage.map_coordinates(f, idx(cen + eps * nrm).T, order=1,
                                 mode="constant", cval=-1e4)
    fm = ndimage.map_coordinates(f, idx(cen - eps * nrm).T, order=1,
                                 mode="constant", cval=-1e4)
    return fp, fm


def marching_tets(f, origin, h, eps_frac=0.25, verbose=False):
    """占据场 → 三角汤 `(M,3,3) float32`（世界坐标，米）。

    场定向：`f` 内正外负，所以外法线满足 `f(+εn) < f(−εn)`；不满足就把顶点 1、2 换位。
    """
    f = np.ascontiguousarray(f, np.float32)
    nx, ny, nz = f.shape
    if nx < 2 or ny < 2 or nz < 2:
        return np.zeros((0, 3, 3), np.float32)

    occ = f >= 0.0
    code = np.zeros((nx - 1, ny - 1, nz - 1), np.uint8)
    for b, (di, dj, dk) in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                      (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]):
        code |= occ[di:di + nx - 1, dj:dj + ny - 1, dk:dk + nz - 1].astype(np.uint8) << b
    idx = np.argwhere((code != 0) & (code != 255))
    if not len(idx):
        return np.zeros((0, 3, 3), np.float32)

    even = (idx.sum(1) % 2) == 0
    off = np.where(even[:, None, None, None], _TET_E[None], _TET_O[None])
    corner = (idx[:, None, None, :] + off).reshape(-1, 4, 3)
    V = f[corner[..., 0], corner[..., 1], corner[..., 2]]
    ins = V >= 0
    nin = ins.sum(1)
    keep = (nin > 0) & (nin < 4)
    corner, V = corner[keep], V[keep]
    if not len(corner):
        return np.zeros((0, 3, 3), np.float32)

    # 6 条棱上的交点（在**格点索引空间**里插值，线性所以与转到世界系可交换）
    #
    # ★ 插值方向必须**规范化**：一条共享棱在两个相邻四面体里可能以 (a,b) 或 (b,a) 出现，
    #   两种写法算出的点在浮点上差几个 ulp；而 `np.unique` 是按位比较的，这点差会让同一个
    #   几何顶点被当成两个 —— 于是网格里出现 0.02% 量级的假「裂缝」（实测：解析球上
    #   86 条边界边、27 条有向边违规）。所以统一**从扁平化索引小的那个格点出发**插值，
    #   让交点成为「无序棱」的纯函数，逐位可复现。
    ea, eb = _TET_EDGES[:, 0], _TET_EDGES[:, 1]
    Va, Vb = V[:, ea], V[:, eb]                       # (T,6)
    cross = (Va >= 0) != (Vb >= 0)
    N1, N2 = nx * ny, nx
    ia, ib = corner[:, ea, :], corner[:, eb, :]       # (T,6,3) 全局格点索引
    fa = ia[..., 0] * N1 + ia[..., 1] * N2 + ia[..., 2]
    fb = ib[..., 0] * N1 + ib[..., 1] * N2 + ib[..., 2]
    swap = fa > fb                                    # 规范成 lo → hi
    lo_i = np.where(swap[..., None], ib, ia)
    hi_i = np.where(swap[..., None], ia, ib)
    lo_v = np.where(swap, Vb, Va)
    hi_v = np.where(swap, Va, Vb)
    den = lo_v - hi_v
    t = np.where(np.abs(den) > 1e-30, lo_v / np.where(den == 0, 1.0, den), 0.5)
    CP = lo_i + t[..., None] * (hi_i - lo_i)          # (T,6,3)

    ncr = cross.sum(1)
    order = np.argsort(~cross, axis=1, kind="stable")          # 交点排到前 ncr 个
    G = np.take_along_axis(CP, order[..., None], axis=1)

    outs = []
    s3 = ncr == 3
    if s3.any():
        outs.append(G[s3][:, :3])                              # 1 类：1 或 3 个在内
    s4 = ncr == 4
    if s4.any():                                               # 2 类：四边形 → 2 个面
        q = G[s4][:, :4]
        c0 = q.mean(1, keepdims=True)
        r = q - c0
        nrm = np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-15)
        e1 = r[:, 0] / np.maximum(np.linalg.norm(r[:, 0], axis=1, keepdims=True), 1e-15)
        e2 = np.cross(nrm, e1)
        ang = np.arctan2((r * e2[:, None, :]).sum(-1), (r * e1[:, None, :]).sum(-1))
        q = np.take_along_axis(q, np.argsort(ang, axis=1)[..., None], axis=1)
        outs.append(q[:, [0, 1, 2]])
        outs.append(q[:, [0, 2, 3]])
    tris = np.concatenate(outs, 0).astype(np.float64)
    tris = origin + h * tris                                   # → 世界坐标（米）

    # 丢**零面积**面（判据是 `_DEGEN_AREA2 = 0.0`，即逐位重合，**不可放宽** —— 见该常量
    # 的说明：放宽到 1e-12 会真的在网格上开出 300+ 条边界边）。零面积面来自格点场值
    # **恰好**为 0：`V >= 0` 把它算作内点，于是它到每个外顶点的交点都落在 t=0，同一个点
    # 被记了多次。丢掉它们反而让配对更干净（否则留下自环 (a,a) 破坏「每条有向边恰好一次」）。
    _e1, _e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    area2 = np.linalg.norm(np.cross(_e1, _e2), axis=1)         # = 2*面积
    degenerate = area2 <= _DEGEN_AREA2                          # 见模块常量说明
    n_deg = int(degenerate.sum())
    if n_deg:
        tris = tris[~degenerate]
    if not len(tris):
        return np.zeros((0, 3, 3), np.float32)

    fp, fm = _field_at(tris, f, origin, h, eps_frac * h)
    tris, n_flip, n_disagree = _orient(tris, fp, fm)
    if verbose:
        print(f"    [mtet] {len(idx)} 个跨面格 → {len(tris)} 个三角面"
              f"（丢掉退化面 {n_deg} 个；翻正 {n_flip} 个，"
              f"其中 {n_disagree} 个与场的判定**相左**）")
    return tris.astype(np.float32)


def _orient(tris, fp, fm):
    """定向两段式：**场**给可靠面定朝向，**拓扑**把场判不准的面兜住。

    为什么需要两段（实测依据，别再退回单段）：`f = min_i Φ_i` 在 VH 的**尖脊**处有折点，
    跨脊的三角面质心 ± εn 落在折点两侧 ⇒ `fp ≈ fm`。舵机上实测：全部 239916 个面里
    `|fp−fm|` 的中位数是 0.73，而 27 个「朝向不一致」的面全部低于**全场 p1 = 0.435**
    （最小 0.0019），且全部落在 x≈13.6mm 一条 2.3mm 厚的**脊线**上。这些面里 `fp−fm` 的
    **符号是噪声**，场内定向在那里等于抛硬币，于是相邻面互相矛盾（17 条有向边同向重复）。

    做法：把置信度（`|fp−fm|`）从高到低排序，先到先得地给每个连通分量播种，再沿
    「共享棱」把朝向传播出去 —— 共享棱要求两个面**反向**使用它，所以邻居的翻法由拓扑
    唯一确定，不需要再问场。返回 `(tris, 换位面数, 与场相左的面数)`；
    最后一个数就是「场的判定在多少个面上不可信」的**实测量**（实测舵机 ≈ 27）。
    """
    M = len(tris)
    if M == 0:
        return tris, 0, 0, 0
    v = np.asarray(tris, float).reshape(-1, 3)
    _, inv = np.unique(v, axis=0, return_inverse=True)
    fc = inv.reshape(-1, 3)

    want_flip = fp > fm                                    # 场的独立判定：法线朝内 → 换位
    conf = np.abs(fp - fm)                                 # 该判定的置信度

    # 半棱：三条有向边。`sgn = +1` 表示这条边按 (小,大) 走
    src = np.concatenate([fc[:, 0], fc[:, 1], fc[:, 2]])
    dst = np.concatenate([fc[:, 1], fc[:, 2], fc[:, 0]])
    fid = np.tile(np.arange(M), 3)
    sgn = np.where(src < dst, 1, -1)
    key = np.minimum(src, dst).astype(np.int64) * np.int64(len(inv)) + np.maximum(src, dst)
    o = np.argsort(key, kind="stable")
    key, fid, sgn = key[o], fid[o], sgn[o]

    # 每条无向棱的连续段；长度 2 = 正常的流形内部棱（长度 1 是开口边 → 本函数管不了）
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    lens = np.diff(np.r_[starts, len(key)])
    pair = starts[lens == 2]
    fa, fb = fid[pair].tolist(), fid[pair + 1].tolist()
    # 两个面以**同向**用同一条棱 ⇒ 其中一个必须翻。一致性条件：
    #   sgn_a·(−1)^flip_a = −sgn_b·(−1)^flip_b  ⇒  flip_b = flip_a XOR (sgn_a == sgn_b)
    same = (sgn[pair] == sgn[pair + 1]).tolist()

    adj = [[] for _ in range(M)]
    for a, b, sm in zip(fa, fb, same):
        adj[a].append((b, sm))
        adj[b].append((a, sm))

    # 播种 + 传播：按置信度从高到低走，遇到还没定过的面就用**场**给它定（此时它是
    # 剩余面里最可信的），再沿棱把朝向传出去 —— 传播段只信拓扑，不再问场。
    flip = np.zeros(M, bool)
    state = np.zeros(M, bool)
    q = deque()
    for s in np.argsort(-conf, kind="stable").tolist():
        if state[s]:
            continue
        state[s] = True
        flip[s] = bool(want_flip[s])
        q.append(s)
        while q:
            i = q.popleft()
            fi = flip[i]
            for j, sm in adj[i]:
                if state[j]:
                    continue
                state[j] = True
                flip[j] = fi ^ sm
                q.append(j)

    n_by_field = int(flip.sum())                           # 最终有多少面换了位
    n_disagree = int(np.sum(flip != want_flip))            # 其中与场自己的判定相左的
    out = tris.copy() if flip.any() else tris
    if flip.any():
        out[flip] = out[flip][:, [0, 2, 1]]
    return out, n_by_field, n_disagree


def surface_nets(f, origin, h):
    """备用抽取：**尚未实现**，先明确挡在这里。

    计划里它是 marching tets 自检没过时的退路（格心一个顶点、构造性闭合流形、不需要
    任何分类逻辑，代价是把 VH 的尖脊削圆）。`marching_tets` 现在的实测状态：
      · 解析球：边界边 = 0、有向边违规 = 0、有符号体积与体素吻合 0.05%；
      · 舵机：边界边 = 0、有向边违规 = 0、有符号体积与体素吻合 0.01%；
        代价是 11/239916 个面的朝向是**拓扑**定的而非场定的（见 `_orient`）。
    两条都过了，所以这条退路**没有触发**。**与其留一份没验证过的实现，不如留一个会响的桩。**
    """
    raise NotImplementedError(
        "surface_nets 没有实现 —— marching_tets 目前不需要这条退路。"
        "真要用时再写（格心顶点 + 逐格四边形，注意顶点要在跨面格之间共享）。")


# ------------------------------------------------------------------ 网格自检

def mesh_checks(tris, f=None, origin=None, h=None, quant=None):
    """三条独立检查。**证据是前两条**（有向边、边界边），第三条只是诊断。

    ⚠️ `winding_ok` 的含义在改成「两段式定向」（`_orient`）之后**变了**：以前定向完全由
    场决定，于是这一项恒等于 1.0，是自己验自己、没有信息量。现在朝向由**拓扑**传播确定、
    场只用来给每个分量播种，所以 `winding_ok = 1 − (拓扑推翻场的面数)/总面数` 成了一个
    真实的量：**场定向在多少比例的面上下不了决心**。舵机上 ≈ 1 − 27/239916 = 0.99989。

    返回 dict。`quant`：顶点量化步长（米），`None` = **按位精确合并**（默认）。

    ⚠️ 这里踩过一个坑，记下来免得再犯：`marching_tets` 的交点是按「无序棱」的纯函数算的
    （永远从扁平化索引小的格点出发插值），所以同一个几何顶点在不同四面体里**逐位相同**，
    `quant=None` 是干净的。**不要**为了「保险」去量化：量化是**分箱**，两个只差 1e-17 的点
    正好落在箱边界两侧时会分到不同的箱，凭空造出边界边。早期版本用 `quant=1e-7` 就是
    这么在解析球上量出 86 条假边界边、27 条假有向边违规的。
    """
    out = dict(n_tri=int(len(tris)), boundary_edges=None, directed_edge_viol=None,
               signed_volume_mm3=None, winding_ok=None)
    if not len(tris):
        return out
    v = np.asarray(tris, float).reshape(-1, 3)
    if quant is not None:
        v = np.round(v / quant).astype(np.int64)
    _, inv = np.unique(v, axis=0, return_inverse=True)
    fc = inv.reshape(-1, 3)

    # ① 有向边：一致定向的闭合流形上每条有向边恰好一次
    de = np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]], 0)
    uniq, cnt = np.unique(de, axis=0, return_counts=True)
    out["directed_edge_viol"] = int((cnt != 1).sum())
    # ② 无向边：每条恰好两次（抓裂缝）
    ue = np.sort(np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]], 0), axis=1)
    _, cu = np.unique(ue, axis=0, return_counts=True)
    out["boundary_edges"] = int((cu == 1).sum()) + int(2 * (cu > 2).sum())

    # ③ 有符号体积
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    out["signed_volume_mm3"] = float((np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
                                     * 1e9)
    if f is not None and origin is not None and h is not None:
        fp, fm = _field_at(np.asarray(tris, float), f, origin, h, 0.25 * h)
        out["winding_ok"] = float((fp < fm).mean())
    return out


def voxel_volume_mm3(f, h):
    return float((f >= 0).sum()) * (float(h) * 1000.0) ** 3


# ------------------------------------------------------------- 合成多视角（自检）

def fibonacci_sphere(n, seed=None):
    """Fibonacci 球面方向 `(n,3)`，单位向量。"""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    d = np.stack([r * np.cos(th), r * np.sin(th), z], 1)
    if seed is not None:
        rng = np.random.default_rng(seed)
        R = _rand_rot(rng)
        d = d @ R.T
    return d


def _rand_rot(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def synthetic_views(tris_gt, K, dist, n_views, cam_dist_m=0.25, W=640, H=480,
                    elev_max_deg=None, centre=(0.0, 0.0, 0.0), seed=0, mask_noise_px=0.0,
                    pose_noise_px=0.0, margin=0.0, jitter_centre=None):
    """把 GT 网格放进一个虚拟场景，绕它摆 `n_views` 台相机，渲出剪影。

    `elev_max_deg`：限制仰角范围。实物摆在板子上时**只能拍到上半球**，
      所以真协议要设成 90（即含天顶、不含底下）。`None` = 整球（理想化对照）。

    扰动在**生成端**施加，并记录真实幅度，这样验证时能算**预测效应**而不只是事后误差：
      · `mask_noise_px`：剪影边界噪声，实现成给距离场加一个空间平滑的随机场
        （`_boundary_noise`），所以边界位移是**亚像素且沿边界连续**的 —— 这正是
        「随机采样的分割边界」的物理图像。早期版本按整数腐蚀核实现，σ=0.5px 时
        68% 的视角是空操作，已废弃（理由见 `_boundary_noise` 的 docstring）。
      · `pose_noise_px`：相机位姿误差，按重投影残差 `pose_noise_px` 折算成
        转角 `δθ = px/f` 与横向平移 `δt = px·z/f`，**六个自由度都扰**。

    返回 `(views, cen)`；`cen` 是物体的**实际**中心（`jitter_centre` 会把它挪开）。
    """
    rng = np.random.default_rng(seed)
    K = np.asarray(K, float)
    f_px = 0.5 * (K[0, 0] + K[1, 1])
    cen = np.asarray(jitter_centre if jitter_centre is not None else centre, float)
    T_o = np.eye(4)
    T_o[:3, 3] = cen
    gt = MR.transform(tris_gt, T_o)

    dirs = fibonacci_sphere(n_views) if elev_max_deg is None else None
    if dirs is None:
        em = np.deg2rad(float(elev_max_deg))
        # 上半球（含天顶、含地平线一点），均匀铺在立体角上
        u = rng.random(int(n_views) * 40)
        v = rng.random(int(n_views) * 40)
        cz = u                                     # cos(elevation) 在 [cos(em),1] 上均匀
        cz = 1.0 - cz * (1.0 - np.cos(em))
        sz = np.sqrt(np.maximum(0.0, 1.0 - cz * cz))
        th = 2 * np.pi * v
        d = np.stack([sz * np.cos(th), sz * np.sin(th), cz], 1)
        # 取最均匀的 n 个：贪心最大化最小夹角（简单、够用）
        pick = [0]
        for _ in range(1, int(n_views)):
            cosang = d @ d[pick].T
            pick.append(int(np.argmin(cosang.max(1))))
        dirs = d[pick]

    views = []
    for i, d in enumerate(dirs):
        eye = cen + cam_dist_m * d
        T_world_cam = MR.look_at(eye, cen, up_world=(0, 0, -1) if abs(d[2]) < 0.99
                                 else (0, 1, 0))
        if pose_noise_px > 0:
            dth = pose_noise_px / f_px
            R = cv2.Rodrigues(rng.normal(size=3) * dth / np.sqrt(3))[0]
            dt = rng.normal(size=3) * (pose_noise_px * cam_dist_m / f_px) / np.sqrt(3)
            T_world_cam = T_world_cam.copy()
            T_world_cam[:3, :3] = T_world_cam[:3, :3] @ R
            T_world_cam[:3, 3] = T_world_cam[:3, 3] + dt
        cam = MR.Cam(K, dist, T_world_cam, W, H)
        ras = MR.rasterize(gt, cam, cull=False)
        m = ras["mask"]
        views.append(make_view(T_world_cam, K, dist, m, W, H, margin=margin,
                               name=f"v{i:02d}",
                               phi_noise=_boundary_noise(H, W, mask_noise_px, rng)))
    return views, cen


# --------------------------------------------------------------- 物体坐标系

def object_frame(tris, spread_thresh=1.5):
    """占据体质心 + PCA 主轴 → 4×4（carve 系 → 物体系）。

    **只在特征值展布 > spread_thresh 时才真转主轴**：近各向同性的件 PCA 轴不稳定，
    可复现比好看重要。返回值始终是一个 4×4，调用方不用管走的是哪条分支。
    """
    P = np.asarray(tris, float).reshape(-1, 3)
    c = P.mean(0)
    T = np.eye(4)
    T[:3, 3] = -c
    X = P - c
    cov = X.T @ X / max(len(X), 1)
    w, V = np.linalg.eigh(cov)
    if w[-1] > spread_thresh ** 2 * max(w[0], 1e-30):
        V = V[:, ::-1]                                  # 大 → 小
        if np.linalg.det(V) < 0:
            V[:, 2] *= -1
        R = np.eye(4)
        R[:3, :3] = V.T
        T = R @ T
    return T


# ------------------------------------------------------------------- 文件输出

def write_stl(path: Path, tris, name="recon"):
    """ASCII STL（可被 `MR.stl_cached` 原样读回 —— 这也是一个往返自检）。"""
    tris = np.asarray(tris, np.float32)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-20)
    out = [f"solid {name}"]
    for t, n in zip(tris, nrm):
        out.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        out.append("    outer loop")
        for v in t:
            out.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        out.append("    endloop")
        out.append("  endfacet")
    out.append(f"endsolid {name}")
    Path(path).write_text("\n".join(out) + "\n")


def load_part(name_or_path, expect_m=True):
    """读一个 STL 当**真值**，带单位闸门。

    18 个 SO101 STL 里 17 个是**米**，而 `wrist_camera_mount_so101_v1.stl` 的数是 ~66
    即**毫米**；URDF 里**没有** `scale` 属性，`render_urdf.load_stl` 也不缩放。
    所以这里按外接盒做合理性检查，不合常理就明确报错，而不是悄悄渲出一颗 66 米的巨物。
    """
    p = Path(name_or_path)
    if not p.exists():
        p = ASSETS / (str(name_or_path) if str(name_or_path).endswith(".stl")
                      else f"{name_or_path}.stl")
    if not p.exists():
        raise FileNotFoundError(f"找不到网格：{name_or_path}")
    t = np.asarray(MR.stl_cached(p), np.float32)
    ext = float((t.reshape(-1, 3).max(0) - t.reshape(-1, 3).min(0)).max())
    if ext > 0.5:
        raise ValueError(f"{p.name} 外接盒 {ext:.3f} —— 明显不是米制（其余 STL 是米，"
                         f"这个文件的数是毫米）。要么显式缩放，要么别用它当真值。")
    if ext < 0.005:
        raise ValueError(f"{p.name} 外接盒 {ext * 1000:.2f}mm —— 尺寸不合常理，拒绝。")
    return t, p


# ---------------------------------------------------------------------- 自检

def _sph_metrics(tris_rec, r_true_m):
    """解析球的自检：体积半径、剪影半径、表面距离。"""
    a, b, c = tris_rec[:, 0], tris_rec[:, 1], tris_rec[:, 2]
    V = abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))
    r_v = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    return r_v


# 剪影像素化导致的**单边内缩**，实测值（解析球 R=20mm、z=250mm、f=409.57 ⇒ 1px = 0.61mm，此
# 处按半径算 1px ≈ 250/409.57 = 0.61mm 沿视线；下面统一折算成「剪影半径的像素当量」）：
#   · 完美解析圆盘掩码 → 体积半径 −0.33mm ≈ **−0.53px**
#   · 光栅化掩码     → 体积半径 −0.26mm ≈ **−0.42px**
# 机制（和教科书结论并不矛盾，教科书假设剪影是**精确**的）：
#   ① 像素化剪影的边界，其**最小**径向半径 ≈ r_true − 1px（不是平均，是最小）；
#   ② `sdf_view` 的零集比「最外圈内像素心」再外扩 0.5px；
#   ③ visual hull 是**取交**，于是取了那个最小值。
#   ⇒ 有效剪影半径 ≈ r_true − 0.5px，投影回物空间 0.5px·z/f = 0.31mm（与实测 −0.33mm 吻合）。
# 这是**分辨率**效应：换更细的像素或更近的相机就变小；**不是**代码错，也不是可以内缩外扩
# 糊过去的常数偏差（它随 θ_c、随掩码形状变）。所以自检只断言「有界」，并把数值打出来。
_SPH_BIAS_NOTE = ("剪影像素化 ⇒ VH 单边**内缩**（交点取最小），本尺度下约 −0.5px；"
                  "教科书「VH ⊇ 物体」只在剪影精确时成立")


def selftest(verbose=True):
    """两段自检：

    A. **解析球**（真值不用渲染器，是解析的）—— 依次验投影/雕刻/抽取/坐标系；
       重建半径与真半径之差、以及把重建网格**再渲回**剪影与解析剪影的 IoU。
    B. **舵机 STL**（真值是 CAD）—— 验网格自检三条（有向边、边界边、有符号体积），
       以及体积与体素计数是否一致。
    """
    ok = True
    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    dist = np.zeros(5)

    # ---------- A：解析球 ----------
    R_true = 0.020
    gt_sph = MR.uv_sphere(R_true, 48, 96)
    gt_sph = gt_sph + np.array([0.0, 0.0, 0.0], np.float32)
    views, cen = synthetic_views(gt_sph, K, dist, 40, cam_dist_m=0.25)
    h = 0.5e-3
    f, origin, h, _iA = carve_auto(views, h, cen, hint_half=0.040, verbose=verbose)
    tris = marching_tets(f, origin, h, verbose=verbose)
    chk = mesh_checks(tris, f, origin, h)
    r_v = _sph_metrics(tris, R_true)
    if verbose:
        print(f"  [A 解析球 r={R_true * 1000:.1f}mm] 重建 {len(tris)} 面，"
              f"体积半径 {r_v * 1000:.3f}mm（差 {(r_v - R_true) * 1000:+.3f}mm）")
        print(f"      有向边违规 {chk['directed_edge_viol']}｜边界边 {chk['boundary_edges']}｜"
              f"有符号体积 {chk['signed_volume_mm3']:.1f}mm³ "
              f"vs 体素 {voxel_volume_mm3(f, h):.1f}mm³｜场定向自洽率 {chk['winding_ok']:.6f}")
    a_ok = (chk["directed_edge_viol"] == 0 and chk["boundary_edges"] == 0
            and abs(r_v - R_true) < 1.5 * h)
    ok &= a_ok
    if verbose:
        px_equiv = (r_v - R_true) / (h * 1000.0)
        print(f"      → {'通过' if a_ok else '**不通过**'}："
              f"球壳闭合、朝向一致、半径误差 {px_equiv:+.2f} 体素 < 1.5")
        print(f"        （{_SPH_BIAS_NOTE}）")

    # ---------- B：舵机 STL（真值 = CAD）----------
    gt, pth = load_part("sts3215_03a_v1")
    gtc = gt.reshape(-1, 3)
    lo, hi = gtc.min(0), gtc.max(0)
    ctr = 0.5 * (lo + hi)
    gtz = MR.transform(gt, np.block([[np.eye(3), (-ctr)[:, None]], [np.zeros(3), 1.0]]))
    views, cen = synthetic_views(gtz, K, dist, 50, cam_dist_m=0.25)
    h = 0.5e-3
    f2, o2, h2, _iB = carve_auto(views, h, cen, hint_half=0.040, verbose=verbose)
    t2 = marching_tets(f2, o2, h2, verbose=verbose)
    c2 = mesh_checks(t2, f2, o2, h2)
    if verbose:
        print(f"  [B {pth.name}] 重建 {len(t2)} 面；"
              f"有向边违规 {c2['directed_edge_viol']}｜边界边 {c2['boundary_edges']}｜"
              f"有符号体积 {c2['signed_volume_mm3']:.1f}mm³ "
              f"vs 体素 {voxel_volume_mm3(f2, h):.1f}mm³｜"
              f"场定向自洽率 {c2['winding_ok']:.6f}")
    b_ok = (c2["directed_edge_viol"] == 0 and c2["boundary_edges"] == 0
            and c2["signed_volume_mm3"] > 0)
    ok &= b_ok
    if verbose:
        print(f"      → {'通过' if b_ok else '**不通过**'}：网格闭合、朝向一致、体积为正")

    if verbose:
        print(f"\n{'自检通过' if ok else '**自检不通过**'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------ CLI

def _recon_part(name, n_views=50, h=0.5e-3, margin=0.0, cam_dist_m=0.25,
                elev_max_deg=None, planes_board=False, seed=0, verbose=True):
    """按零件名跑一遍完整重建。返回 dict。"""
    gt, pth = load_part(name)
    gtc = gt.reshape(-1, 3)
    lo, hi = gtc.min(0), gtc.max(0)
    ctr = 0.5 * (lo + hi)
    gtz = MR.transform(gt, np.block([[np.eye(3), (-ctr)[:, None]], [np.zeros(3), 1.0]]))
    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    views, cen = synthetic_views(gtz, K, np.zeros(5), n_views, cam_dist_m=cam_dist_m,
                                 elev_max_deg=elev_max_deg, seed=seed, margin=margin)
    hint_half = 0.5 * float((hi - lo).max()) + 0.015
    planes = []
    if planes_board:
        # 板面：物体摆在 z=0 上（虚拟场景里物体中心在 cen，底面在 cen_z - 半高）
        zb = cen[2] - 0.5 * float((hi - lo)[2])
        planes.append(((0.0, 0.0, -1.0), -zb))
    f, origin, h, info = carve_auto(views, h, cen, hint_half=hint_half,
                                    planes=planes, verbose=verbose)
    tris = marching_tets(f, origin, h, verbose=verbose)
    chk = mesh_checks(tris, f, origin, h)
    chk["carve_auto"] = info
    chk["voxel_volume_mm3"] = voxel_volume_mm3(f, h)
    chk["n_views"] = n_views
    chk["margin_px"] = margin
    chk["voxel_mm"] = h * 1000.0
    chk["elev_max_deg"] = elev_max_deg
    chk["gt_name"] = pth.name
    chk["gt_tris"] = int(len(gt))
    chk["gt_bbox_mm"] = [round(float(v) * 1000, 2) for v in (hi - lo)]
    chk["clipped_views"] = int(sum(v["clipped"] for v in views))
    return dict(tris=tris, gt=gtz, f=f, origin=origin, h=h, chk=chk, views=views, cen=cen)


def _lab(img, text, scale=0.55):
    """cv2 的 Hershey 字体渲染不了中文 —— 图上的字一律 ASCII。

    `text` 可以是 list/tuple = 多行（cv2.putText **不认 `\\n`**，会渲成乱码方块，所以自己分行）。
    另有 640px 宽的画幅限制：单行太长会**被裁掉**（踩过一次，标题后半截跑到画幅外，
    图看着「没问题」，其实数字根本没显示）。
    """
    lines = text if isinstance(text, (list, tuple)) else [text]
    for k, ln in enumerate(lines):
        y = 22 + k * int(round(22 * scale / 0.55))
        cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
        w = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if 8 + w > img.shape[1]:
            raise ValueError(f"图上文字超宽会被裁：{w + 8}px > {img.shape[1]}px —— {ln!r}")
    return img


def _slice_segments(mesh, ax, level):
    """网格与平面 `x_ax = level` 的交线段 → `(K,2,3)`。

    直接按三角面插值，而不是拿顶点去筛（顶点筛选只能得到稀疏的点，看着像散点图）。
    """
    seg = []
    for tri in np.asarray(mesh, float):
        d = tri[:, ax] - level
        if (d > 0).all() or (d < 0).all():
            continue
        pts = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            da, db = a[ax] - level, b[ax] - level
            if (da > 0) != (db > 0):
                pts.append(a + (level - a[ax]) / (b[ax] - a[ax]) * (b - a))
        if len(pts) == 2:
            seg.append(np.stack(pts))
    return np.array(seg) if seg else np.zeros((0, 2, 3))


def _figure(part, res, out_path):
    """六联图：CAD | 重建 | 剪影叠加（两视角）| 两个中间截面。"""
    cen = np.asarray(res["cen"], float)
    T0 = np.block([[np.eye(3), (-cen)[:, None]], [np.zeros(3), 1.0]])
    gt = MR.transform(res["gt"], T0)
    rec = MR.transform(res["tris"], T0)
    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    light = MR.Light(direction=(0.3, -0.4, 0.87), ambient=0.45, diffuse=0.75)
    # 视距：物体只有 45mm，太远就只剩几个像素 —— 拿 ~130mm 看，占到画幅约 1/3。
    EYE_A = np.array([0.100, 0.072, 0.062])
    EYE_B = np.array([0.018, 0.113, 0.023])

    panels = []
    for lbl, mesh in ((f"CAD ground truth  ({len(gt)} tri)", gt),
                      (f"reconstructed  ({len(rec)} tri)", rec)):
        cam = MR.Cam(K, np.zeros(5), MR.look_at(EYE_A, np.zeros(3)), 640, 480)
        ras = MR.rasterize(mesh, cam, cull=False)
        img = (MR.shade(ras, cam, light, (0.55, 0.58, 0.62)) * 255).astype(np.uint8)
        panels.append(_lab(img, lbl))

    for eye, tag in ((EYE_A, "view A"), (EYE_B, "view B")):
        cam = MR.Cam(K, np.zeros(5), MR.look_at(eye, np.zeros(3)), 640, 480)
        ov = np.zeros((480, 640, 3), np.uint8)
        ms = {}
        for key, mesh, col in (("gt", gt, (0, 255, 255)), ("rec", rec, (255, 0, 255))):
            m = MR.rasterize(mesh, cam, cull=False)["mask"]
            ms[key] = m > 0
            cnt = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(ov, cnt, -1, col, 1)
        inter = int((ms["gt"] & ms["rec"]).sum())
        union = int((ms["gt"] | ms["rec"]).sum())
        iou = inter / max(union, 1)
        # IoU 与两向差额都打出来：只看 IoU 会把「一边包住另一边」和「两边各错一半」混同
        panels.append(_lab(ov, [
            f"silhouette {tag}: CAD yellow / recon magenta",
            f"IoU {iou:.3f}   GT-only {int((ms['gt'] & ~ms['rec']).sum())}px"
            f"   recon-only {int((ms['rec'] & ~ms['gt']).sum())}px"]))

    for ax, tag in ((1, "y=0"), (2, "z=0")):
        cam = MR.Cam(K, np.zeros(5), MR.look_at(EYE_A, np.zeros(3)), 640, 480)
        sl = np.zeros((480, 640, 3), np.uint8)
        for mesh, col in ((gt, (0, 255, 255)), (rec, (255, 0, 255))):
            for seg in _slice_segments(mesh, ax, 0.0):
                # 沿线段取 8 个点再投（有畸变时三维直线并不投成二维直线）
                ts = np.linspace(0, 1, 8)[:, None]
                P = seg[0] + ts * (seg[1] - seg[0])
                uv, z = cam.project(P)
                k = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
                for i in range(len(uv[k]) - 1):
                    cv2.line(sl, tuple(np.round(uv[k][i]).astype(int)),
                             tuple(np.round(uv[k][i + 1]).astype(int)), col, 1)
        panels.append(_lab(sl, f"slice {tag}: CAD yellow / recon magenta"))

    big = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    # ⚠ `cv2.imwrite` 目录不存在时**不抛异常、只返回 False** —— 踩过一次：脚本照样打印了
    # 「→ 图」，其实一个字节都没写。所以这里显式建目录 + 检查返回值。
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), big):
        raise OSError(f"cv2.imwrite 写不出去：{out_path}")
    return big


def main():
    ap = argparse.ArgumentParser(description="剪影雕刻重建（visual hull）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--part", type=str, default="sts3215_03a_v1")
    ap.add_argument("--views", type=int, default=50)
    ap.add_argument("--voxel-mm", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.0, help="剪影内缩，单位 px")
    ap.add_argument("--cam-dist-mm", type=float, default=250.0)
    ap.add_argument("--elev-max-deg", type=float, default=None,
                    help="限制仰角（实物摆在板上时只有上半球，给 90）")
    ap.add_argument("--board-plane", action="store_true",
                    help="加板面半空间约束（实物摆板上时底面拍不到，靠这条补）")
    ap.add_argument("--extract", choices=("mtet", "snets"), default="mtet")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    res = _recon_part(a.part, n_views=a.views, h=a.voxel_mm / 1000.0, margin=a.margin,
                      cam_dist_m=a.cam_dist_mm / 1000.0, elev_max_deg=a.elev_max_deg,
                      planes_board=a.board_plane)
    c = res["chk"]
    print(f"\n  {c['gt_name']}  CAD 外接盒 {c['gt_bbox_mm']}mm，{c['gt_tris']} 面")
    print(f"  视角 {c['n_views']}（贴边 {c['clipped_views']}）｜体素 {c['voxel_mm']}mm｜"
          f"margin {c['margin_px']}px")
    print(f"  重建 {c['n_tri']} 面｜有向边违规 {c['directed_edge_viol']}｜"
          f"边界边 {c['boundary_edges']}｜场定向自洽率 {c['winding_ok']:.6f}")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(c['gt_name']).stem}_v{c['n_views']}_h{c['voxel_mm']:g}_m{c['margin_px']:g}"
    stl = OUT / f"{tag}.stl"
    write_stl(stl, res["tris"], name=tag)
    (OUT / f"{tag}.json").write_text(json.dumps(c, indent=2, ensure_ascii=False))
    print(f"  → {stl}")
    if a.out:
        # 相对路径按 `HERE`（workplace/calibration）解析，与 `pose6d_swap.py` / `ball_swap.py`
        # 一致：`--out figures/x.png` 就是 workplace/calibration/figures/x.png。
        p = Path(a.out)
        p = p if p.is_absolute() else (HERE / p)
        _figure(a.part, res, p)
        print(f"  → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
