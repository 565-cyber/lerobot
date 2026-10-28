#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""validate_carve.py —— 拿 CAD 真值验证剪影雕刻：管线对不对、VH 的理论极限成不成立。

**这个脚本测什么、不测什么**（先看这段，别误读数字）：

  测：
    ① **雕刻管线本身的几何正确性** —— 投影、SDF 符号、网格、抽取、winding、坐标系。
       真值是 CAD 网格（与重建产物是**两种不同的表示**），所以这一条不是自己验自己。
    ② `visual hull ⊆ conv(S) ⊆ bbox(S)` 给出的**上限**在数字上成立：
       体积比的上限是 `V_G/V_凸包`，体积 IoU 的上限也是 `V_G/V_凸包`。
    ③ 分辨率 / margin / 剪影噪声 / 位姿噪声 / 视角数 的敏感度是否符合预测。
  不测：
    分割质量、真实光学（反光/阴影/曝光）、光照、遮挡、**实物与 CAD 是否一致**。
    那几项只有真采集能补（见 CAPTURE_OBJ.md）。

**核心指标是「单边性」**：VISUAL HULL 的教科书结论是「只会高估、不会低估」，即
`|∂G \\ R| = 0`。但 `mesh_from_views.py` 的实测结论是：**剪影像素化之后这条不成立**，
重建是**单边内缩**的（解析球上 −0.26mm ≈ −0.42px；解析圆盘 −0.11mm ≈ −0.53px）。
所以这个脚本不追求「0」，而是**报出破了多少、并说明和预测吻合**。

用法：
    $PY -I workplace/calibration/validate_carve.py --selftest
    $PY -I workplace/calibration/validate_carve.py --quick
    $PY -I workplace/calibration/validate_carve.py            # 完整扫描（慢）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "calib" / "recon" / "validate"

# 4 个件的凹度（`V_gt/V_凸包`）跨度大，是计划里选的：舵机近凸、motor_holder 极凹。
# 具体数值不写死在这里 —— 由 `hull_stats()` 实测，避免抄错。
DEFAULT_PARTS = ["sts3215_03a_v1", "moving_jaw_so101_v1", "Trigger_SO101",
                 "motor_holder_so101_wrist_v1"]

# 剪影像素化的「内缩偏置」（margin=0 时仍存在的系统性内缩），单位 px，h=1mm 下实测：
#
#   · 0.590  舵机，**等效半径**口径（`aud_inset_px`）—— 预测用它
#   · 0.42   §8.2 的**解析球**，同一个口径（球上三个口径重合）
#
# 选等效半径口径当主仪器是**量出来**的，不是挑的：三个估计量在 margin∈{0,1,2} 上拟合
# 增益，只有它等于定义值 1.00 ——
#     等效半径 R  增益 0.999 px/px   截距 +0.590px
#     周长 P      增益 0.744 px/px   截距 +0.459px     （分母只用真值周长）
#     周长 M      增益 0.774 px/px   截距 +0.460px     （分母用两侧平均周长）
# 两个周长口径都被低估 ~25%，**不是** Minkowski 关系错，而是**数字化周长被高估**：边界
# 像素计数（8 邻域差）比几何周长大约 1.3×，分母一大 δ 就小。改用平均周长只补回 4 个点，
# 说明主因就是这个。等效半径口径不需要周长，所以它干净。
#
# 球 0.42 < 舵机 0.59 —— 方向是「球更乐观」，差 ~40%，所以 0.42px **不是**能到处搬的
# 普适常数（它随掩码形状、随 θ_c 变）。这个比较是**同口径**的：§8.2 的球值也是从
# 体积半径折算的等效半径。
PIXEL_BIAS_PX = 0.59


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MFV = _load_module("_vc_mfv", HERE / "mesh_from_views.py")
MR = MFV.MR


# ------------------------------------------------------------------ 真值统计

def mesh_volume_mm3(tris):
    """闭合网格的有符号体积（绝对值），mm³。"""
    a, b, c = (np.asarray(tris, float)[:, i] for i in range(3))
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)) * 1e9


def mesh_area_mm2(tris):
    """网格表面积，mm²（用于把「边界位移」换算成「体积损失比例」）。"""
    t = np.asarray(tris, float)
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                                      axis=1).sum()) * 1e6


def hull_stats(tris):
    """`(V_gt, V_凸包, 凹度)`，单位 mm³。

    凹度 = `V_gt/V_凸包` ∈ (0,1]。**它是所有「超集型」近似的误差上限的来源**：
    `V_凸包/V_gt = 1/凹度`，也就是体积比能到多大；体积 IoU 的上限直接就是凹度本身。
    所以任何「重建质量」的横轴都该是凹度，否则凹件上的好管线看着很差。

    ⚠️ 这个上限是**连续体积**之间的不等式 `V_VH ≤ V_凸包 ≤ (1/凹度)·V_gt`。所以拿它
    对比时必须两边同源：`R` 和 `G` 都按**同一套格点**数体素（`vol_ratio`），不能拿
    「格点上的 R」比「解析的 V_gt」—— 格点计数本身有 ~h·A/2 的偏置，舵机在 h=1mm 时
    是 +1.9%，足够把 1.104 抬成 1.125 而看着像把集合包含关系违反了。
    """
    from scipy.spatial import ConvexHull
    v_gt = mesh_volume_mm3(tris)
    P = np.asarray(tris, float).reshape(-1, 3)
    try:
        v_hull = float(ConvexHull(P).volume) * 1e9
    except Exception:                                       # 退化点集
        v_hull = v_gt
    return v_gt, v_hull, v_gt / max(v_hull, 1e-12)


# ------------------------------------------------------------- 网格体素化

def voxelize_mesh(tris, origin, h, shape, stab=2, jitter=1e-4, verbose=False):
    """射线穿刺体素化 → `bool (nx,ny,nz)`，格点与 `carve` 的**同一套**（含半格抖动）。

    为什么用扫描线而不是「逐格点 × 逐三角形」：后者要 `n_vox·n_tri ≈ 1e10` 次。
    扫描线把每个三角形投影到垂直于 `stab` 的平面，算出它盖住哪些**列**以及每列的
    穿点坐标，再按列归并排序、**逐列奇偶填充**。代价 ≈ 表面列数 × 层数。

    垂直于 `stab` 的三角形投影退化成线段（测度零、射线只会擦过），用 `jitter` 给列心
    加一点抖动避开精确共面即可；抖动 `1e-4·h` 远小于体素，不影响结果。
    """
    tris = np.asarray(tris, float)
    shape = [int(s) for s in shape]
    u, v = [i for i in range(3) if i != stab]
    nu, nv, nz_ax = shape[u], shape[v], shape[stab]

    rng = np.random.default_rng(0)
    gu = origin[u] + h * (np.arange(nu) + 0.5 * 0)          # 与 carve 同一套格点
    gv = origin[v] + h * np.arange(nv)
    U, V = np.meshgrid(gu, gv, indexing="ij")
    U = U + jitter * h * rng.standard_normal(U.shape)
    V = V + jitter * h * rng.standard_normal(V.shape)

    p0 = tris[:, 0]
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nn = np.linalg.norm(nrm, axis=1)
    keep = (nn > 0) & (np.abs(nrm[:, stab]) > 1e-9 * nn)
    idx = np.flatnonzero(keep)

    cols, zs = [], []
    for t in idx:
        p = tris[t]
        a = p[:, u]
        b = p[:, v]
        # 投影三角形的包围盒 → 列索引范围
        i0 = max(int(np.floor((a.min() - gu[0]) / h)), 0)
        i1 = min(int(np.ceil((a.max() - gu[0]) / h)), nu - 1)
        j0 = max(int(np.floor((b.min() - gv[0]) / h)), 0)
        j1 = min(int(np.ceil((b.max() - gv[0]) / h)), nv - 1)
        if i1 < i0 or j1 < j0:
            continue
        ss = slice(i0, i1 + 1)
        tt = slice(j0, j1 + 1)
        pu = U[ss, tt].ravel()
        pv = V[ss, tt].ravel()
        # 二维重心坐标：投影三角形 (a0,b0),(a1,b1),(a2,b2)
        d = ((b[1] - b[2]) * (a[0] - a[2]) + (a[2] - a[1]) * (b[0] - b[2]))
        if abs(d) < 1e-18:
            continue
        w0 = ((b[1] - b[2]) * (pu - a[2]) + (a[2] - a[1]) * (pv - b[2])) / d
        w1 = ((b[2] - b[0]) * (pu - a[2]) + (a[0] - a[2]) * (pv - b[2])) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        col = (np.repeat(np.arange(i0, i1 + 1), j1 - j0 + 1) * nv
               + np.tile(np.arange(j0, j1 + 1), i1 - i0 + 1))
        col = col[inside]
        # 平面方程：z = A·u + B·v + C
        nz_ = nrm[t, stab]
        A = -nrm[t, u] / nz_
        B = -nrm[t, v] / nz_
        C = float(nrm[t] @ p0[t]) / nz_
        cols.append(col)
        zs.append(A * gu[0] + B * gv[0] + C + (A * (pu[inside] - gu[0])
                                               + B * (pv[inside] - gv[0])))
    if not cols:
        return np.zeros(shape, bool)
    col = np.concatenate(cols).astype(np.int64)
    z = np.concatenate(zs).astype(np.float64)

    order = np.lexsort((z, col))
    col, z = col[order], z[order]
    newrun = np.r_[True, col[1:] != col[:-1]]
    # 每条列内按顺序两两配对：第 0,1 个穿点之间是内部，2,3 之间是内部……
    pos = np.arange(len(col)) - np.repeat(np.flatnonzero(newrun), np.diff(np.r_[np.flatnonzero(newrun), len(col)]))
    lower = (pos % 2 == 0) & ~np.r_[newrun[1:], True]       # 只取「不是该列最后一个」的下端点
    lo = np.flatnonzero(lower)
    hi = lo + 1
    ok = (hi < len(col)) & (col[hi] == col[lo])
    lo, hi = lo[ok], hi[ok]

    zlat = origin[stab] + h * np.arange(nz_ax)
    k0 = np.searchsorted(zlat, z[lo], side="right")
    k1 = np.searchsorted(zlat, z[hi], side="left") - 1
    out = np.zeros(shape, bool)
    flat = out.reshape(-1)
    base = col[lo] * nz_ax
    for a_, b_, bi in zip(k0, k1, base):
        if b_ >= a_:
            sl = slice(int(bi + a_), int(bi + b_) + 1)
            flat[sl] = True
    if verbose:
        print(f"    [voxelize] {len(idx)} 个可用三角形 → {len(lo)} 个内部区间 → "
              f"{int(out.sum())} 格")
    return out


# ------------------------------------------------------------------ 指标

def grid_metrics(R, G, h, cam_dist_m, f_px, margin_px, mask_noise_px, pose_noise_px,
                 v_gt_mm3, v_hull_mm3, area_gt_mm2):
    """在**同一套格点**上的真值 `G` 与重建 `R` 上算全部指标。长度单位 mm。

    「单边性」有两个读法，都报：
      · `miss_vol = |G \\ R|/|G|` —— 真值里**没被盖住**的那部分体积占比。VH 的保证是
        「只会高估」，所以这个数**理论上是 0**（在 margin=0、无噪声、剪影精确时）。
      · `miss_surf = |∂G \\ R|/|∂G|` —— 真值**表面**落在重建之外的比例。这是计划里写的
        那个 `|∂G \\ R|/|∂G|`，比体积版更敏感（体积版被内部格点稀释）。
      · `false_pos = |R \\ G|/|R|` —— 反方向（高估）的比例，VH 天生就大。

    **可证伪的预测**（都换算到物空间，`z/f` mm/px）：
      · 显式腐蚀 `pred_inset_mm = margin·z/f` —— 这一项是定义，没有近似。
      · 像素化单边内缩 `pred_pixel_mm = 0.42·z/f` —— §8.2 在解析球上**实测**的等效半径差
        （−0.26mm @ z=250mm）折算成像素就是 0.42px。这是数据不是假设。
      · 随机项 `s = sqrt(σ_mask² + σ_pose²)·z/f` ⇒ p95 加 `1.96s`。
      · 合计 `pred_p95_mm = pred_inset_mm + pred_pixel_mm + 1.96s`。
      · 体积损失的**薄壳上界** `ub_miss_vol = (inset+pixel)·(A_gt/V_gt)`。⚠️ 它是**上界**
        而不是预测：推导假设「所有表面都是剪影的约束边」，而凹面在**任何**视角都不构成
        剪影边界，所以那些面积不内缩。真要用它当预测，必须先把 A 换成「约束面积」A_bind，
        而 A_bind 事先不知道 —— 所以这里的做法是**反过来**：从 margin 扫描里拟合出
        `d(miss_vol)/d(margin)` 的斜率，再除以 `(A_gt/V)·z/f` 得到 `A_bind/A_gt`，
        把它当成**测出来的**量报出来。图里画的是上界（用虚线标「上界」）。
    """
    h3 = (h * 1e3) ** 3
    nR, nG = int(R.sum()), int(G.sum())
    inter = int((R & G).sum())
    union = int((R | G).sum())

    def surface(X):
        # 6 邻域腐蚀的差：贴着边界的那些格
        st = ndimage.generate_binary_structure(3, 1)
        return X & ~ndimage.binary_erosion(X, structure=st, border_value=0)

    Rs, Gs = surface(R), surface(G)
    dt_g = ndimage.distance_transform_edt(~Gs, sampling=(h * 1e3,) * 3)
    dt_r = ndimage.distance_transform_edt(~Rs, sampling=(h * 1e3,) * 3)
    d_rg = dt_g[Rs] if Rs.any() else np.zeros(1)
    d_gr = dt_r[Gs] if Gs.any() else np.zeros(1)
    # ⚠️ d_rg / d_gr 的 **中位数和 p95 都不是「内缩量」**，因为分布是混合的：
    #   · 大部分表面格是两边重合的（距离 0，中位被它钉在 0）；
    #   · 一个长尾来自 VH **填凹**（R 的表面在凹坑里离 G 的面很远）—— 这是
    #     `V_凸包` 那一类误差，不是管线误差，且**和 margin 反向**（margin 越大填得越浅）。
    # 所以分方向、且**只取单边的那一侧**才是干净的：
    #   · protrude：真值表面**戳出** R 的那部分（`∂G \\ R`）到 R 的距离 —— 这才是「单边内缩」，
    #     应当跟着 `pred_p95_mm`；
    #   · fill：R 的表面**鼓出** G 的那部分（`∂R \\ G`）到 G 的距离 —— 这是填凹/假阳的深度。
    prot = dt_r[Gs & ~R] if (Gs & ~R).any() else np.zeros(1)
    fill = dt_g[Rs & ~G] if (Rs & ~G).any() else np.zeros(1)

    vR, vG = nR * h3, nG * h3
    mmpp = cam_dist_m / f_px * 1e3
    inset = float(margin_px) * mmpp                 # 显式腐蚀（定义）
    pixel = 0.42 * mmpp                            # §8.2 实测的像素化单边内缩
    sig = float(np.hypot(mask_noise_px, pose_noise_px)) * mmpp
    m = dict(
        n_R=nR, n_G=nG, v_R_mm3=vR, v_G_mm3=vG, area_gt_mm2=area_gt_mm2,
        # 两边**同源**：都按格点数。拿格点 R 比解析 V_gt 会把 ~h·A/2 的量化偏置算成误差。
        vol_ratio=vR / max(vG, 1e-12),
        vol_ceiling=v_hull_mm3 / max(v_gt_mm3, 1e-12),
        # 绝对精度另算一栏（对 CAD），这才该跟 1.0 比
        vol_ratio_cad=vR / max(v_gt_mm3, 1e-12),
        vol_vox_bias=(vG - v_gt_mm3) / max(v_gt_mm3, 1e-12),
        iou=inter / max(union, 1),
        iou_ceiling=v_gt_mm3 / max(v_hull_mm3, 1e-12),
        miss_vol=(nG - inter) / max(nG, 1),
        false_pos=(nR - inter) / max(nR, 1),
        miss_surf=float((Gs & ~R).sum()) / max(int(Gs.sum()), 1),
        p95_R_to_G_mm=float(np.percentile(d_rg, 95)),
        p95_G_to_R_mm=float(np.percentile(d_gr, 95)),
        med_R_to_G_mm=float(np.median(d_rg)),
        p95_protrude_mm=float(np.percentile(prot, 95)),
        med_protrude_mm=float(np.median(prot)),
        p95_fill_mm=float(np.percentile(fill, 95)),
        med_fill_mm=float(np.median(fill)),
        n_protrude=int(prot.size), n_fill=int(fill.size),
        mm_per_px_obj=mmpp,
        pred_inset_mm=inset,
        pred_pixel_mm=pixel,
        pred_sigma_mm=sig,
        pred_med_mm=inset + pixel,
        pred_p95_mm=inset + pixel + 1.96 * sig,
        ub_miss_vol=min(1.0, (inset + pixel + 1.96 * sig) * area_gt_mm2
                        / max(v_gt_mm3, 1e-12)),
        surf_R_px=int(Rs.sum()),
        surf_G_px=int(Gs.sum()),
    )
    m["p95_sym_mm"] = max(m["p95_R_to_G_mm"], m["p95_G_to_R_mm"])
    return m


# ------------------------------------------------------- 图像域审计（主仪器）

def silhouette_audit(tris_rec, views, max_views=None, verbose=False):
    """把重建网格在**每个视角自己的相机**下渲出来，与该视角的真值剪影逐像素比。

    **为什么必须有这一条**：体素域的表面距离在 h=0.5mm 时被量化到 0.5mm 的整数倍
    （实测中位数恒等于 1.0·h，因为「戳出」的最小可能距离就是一个体素），而 §8.2 预测的
    内缩只有 0.26mm —— **仪器的分辨率比被测效应还大**，那种数字只能说明「大约一个体素」。
    图像域完全不一样：真值剪影面积 ~1000px、边界长 ~150px，0.1px 的内缩就改动 15px 的
    面积（1.5%），而面积是整数计数、信噪比极高。所以「内缩了几个像素」这件事**只能在
    图像域测**。这也正好是 §8.2 结论（等效半径 −0.42px）的同一把尺子，可以直接对表。

    `inset_px` 用的是**等效半径差** `√(A_gt/π) − √(A_rec/π)`（对凸形是严格的；凹形的面积
    与半径换算有形状偏置，所以同时也报 `(漏−多)/周长` 这个不依赖形状的估计，两者应当同号）。
    """
    idx = list(range(len(views)))
    if max_views is not None and len(idx) > max_views:
        step = len(idx) / float(max_views)
        idx = [idx[int(i * step)] for i in range(int(max_views))]
    ious, miss, extra, inset_r, inset_p, inset_m = [], [], [], [], [], []
    for i in idx:
        v = views[i]
        cam = MR.Cam(v["K"], v["dist"], v["T_world_cam"], v["W"], v["H"])
        rec = MR.rasterize(tris_rec, cam, cull=False)["mask"]
        gt = v["mask"]
        union = int((rec | gt).sum())
        inter = int((rec & gt).sum())
        ious.append(inter / max(union, 1))
        m_ = int((gt & ~rec).sum())
        e_ = int((rec & ~gt).sum())
        miss.append(m_)
        extra.append(e_)
        r_gt = np.sqrt(max(int(gt.sum()), 1) / np.pi)
        r_rc = np.sqrt(max(int(rec.sum()), 1) / np.pi)
        inset_r.append(r_gt - r_rc)
        # 周长用真值剪影的边界像素数近似（8 邻域差的计数，比 4 邻域更接近真实周长）
        per = float((gt & ~ndimage.binary_erosion(
            gt, structure=np.ones((3, 3), bool), border_value=0)).sum())
        inset_p.append((m_ - e_) / max(per, 1.0))
        # Minkowski 一阶关系是 `ΔA = δ·⟨P⟩`，`⟨P⟩` 取**两侧边界周长的平均**。只用 `P_gt`
        # 会在 margin 大时把 δ 系统性压小（腐蚀后周长变短，实测增益掉到 0.74 而不是 1.00）
        per_r = float((rec & ~ndimage.binary_erosion(
            rec, structure=np.ones((3, 3), bool), border_value=0)).sum())
        inset_m.append((m_ - e_) / max(0.5 * (per + per_r), 1.0))
    out = dict(audit_n=len(idx),
               aud_iou=float(np.mean(ious)),
               aud_inset_px=float(np.mean(inset_r)),
               aud_inset_px_iqr=[float(np.percentile(inset_r, 25)),
                                 float(np.percentile(inset_r, 75))],
               aud_inset_perim_px=float(np.mean(inset_p)),
               aud_inset_mid_px=float(np.mean(inset_m)),
               aud_miss_px=float(np.mean(miss)),
               aud_extra_px=float(np.mean(extra)))
    if verbose:
        print(f"    [图像域] {len(idx)} 视角：IoU {out['aud_iou']:.4f}｜"
              f"内缩(等效半径) {out['aud_inset_px']:+.3f}px "
              f"IQR[{out['aud_inset_px_iqr'][0]:+.3f},{out['aud_inset_px_iqr'][1]:+.3f}]｜"
              f"内缩(面积/周长) {out['aud_inset_perim_px']:+.3f}px｜"
              f"平均漏 {out['aud_miss_px']:.0f}px 多 {out['aud_extra_px']:.0f}px")
    return out


# ------------------------------------------------------------------ 单次实验

def _covering_radius(views, cen, elev_max_deg=None, n_probe=6000, seed=0):
    """**实测**覆盖半径 θ_c：在（半）球面上任取探针点，取「到最近视角方向的最大夹角」。

    为什么不用 `2.2/√N` 这个模型式：它是「最优覆盖」的渐近估计，而 Fibonacci 球的常数
    与之不同，仰角受限时更不一样。既然视角方向本来就在 `T_world_cam` 里，直接量出来
    更省事也更诚实 —— 图上的横轴就成了数据而不是公式。返回 `(θ_c, 探针数)`；
    探针点按视角集实际覆盖的极角范围取（否则物体下方没相机的那半边会把答案钉在 π/2）。
    """
    dirs = np.stack([v["T_world_cam"][:3, 3] - np.asarray(cen, float) for v in views])
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n_probe, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    # 只保留与「视角极角范围」一致的探针：cos(极角) 落在视角实测的 [min,max] 内
    cz = dirs[:, 2]
    lo, hi = float(cz.min()), float(cz.max())
    keep = (P[:, 2] >= lo - 1e-9) & (P[:, 2] <= hi + 1e-9)
    if keep.sum() < 100:                                     # 视角太集中时退回全球
        keep = np.ones(n_probe, bool)
    P = P[keep]
    best = np.full(len(P), -1.0)
    for d in dirs:                                           # 逐个视角更新最近夹角
        np.maximum(best, P @ d, out=best)
    return float(np.arccos(np.clip(best, -1, 1)).max()), int(len(P))

def run_once(part, n_views=50, h=1.0e-3, margin=0.0, cam_dist_m=0.25,
             mask_noise_px=0.0, pose_noise_px=0.0, seed=0, elev_max_deg=None,
             aim_offset_m=None, k1=0.0, board_plane=False, coarse=4, audit_views=12,
             stl_out=None, verbose=False):
    """一整套：建真值 → 造视角（在**生成端**施加扰动）→ 雕刻 → 抽取 → 体素化真值 → 指标。

    返回 dict（含真值统计、扰动幅度、全部指标、耗时）。

    `aim_offset_m` 把物体推离光轴（`jitter_centre`），`k1` 给相机加径向畸变 —— 这两个
    一起用才是在考「畸变下 margin 还是不是各向同性的像素腐蚀」。注意雕刻侧用的是**同一
    组** `K/dist`（`carve` 里 `cv2.projectPoints(..., v["K"], v["dist"])`），所以这不是
    在测「有没有做去畸变」，而是在测**像素域的 margin 在畸变+离轴下会不会失真**。
    """
    t0 = time.time()
    gt, pth = MFV.load_part(part)
    gtc = gt.reshape(-1, 3)
    lo, hi = gtc.min(0), gtc.max(0)
    ctr = 0.5 * (lo + hi)
    gtz = MR.transform(gt, np.block([[np.eye(3), (-ctr)[:, None]], [np.zeros(3), 1.0]]))
    v_gt, v_hull, conc = hull_stats(gtz)
    a_gt = mesh_area_mm2(gtz)

    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    dist = np.array([float(k1), 0.0, 0.0, 0.0, 0.0])
    f_px = 0.5 * (K[0, 0] + K[1, 1])
    views, cen = MFV.synthetic_views(
        gtz, K, dist, n_views, cam_dist_m=cam_dist_m, elev_max_deg=elev_max_deg,
        seed=seed, mask_noise_px=mask_noise_px, pose_noise_px=pose_noise_px,
        margin=margin, jitter_centre=(np.asarray(aim_offset_m, float)
                                      if aim_offset_m is not None else None))
    planes = []
    if board_plane:
        zb = cen[2] - 0.5 * float((hi - lo)[2])
        planes.append(((0.0, 0.0, -1.0), -zb))

    hint = 0.5 * float((hi - lo).max()) + 0.015
    f, origin, hh, info = MFV.carve_auto(views, h, cen, hint_half=hint, coarse=coarse,
                                         planes=planes, verbose=False)
    tris = MFV.marching_tets(f, origin, hh, verbose=False)
    chk = MFV.mesh_checks(tris, f, origin, hh)
    if stl_out is not None:
        # 网格单独落盘成 STL：返回值要 JSON 当缓存，塞 24 万个三角形会把缓存撑爆
        MFV.write_stl(Path(stl_out), tris, name=Path(stl_out).stem)

    R = f >= 0
    # 真值放在**同一套格点**上（同一个世界系：`synthetic_views` 已把 GT 平移到 cen）
    G = voxelize_mesh(MR.transform(gtz, np.block([[np.eye(3), (cen)[:, None]],
                                                 [np.zeros(3), 1.0]])),
                      origin, hh, R.shape, verbose=False)
    met = grid_metrics(R, G, hh, cam_dist_m, f_px, margin, mask_noise_px, pose_noise_px,
                       v_gt, v_hull, a_gt)

    # 覆盖半径：横轴用它而不是 N（N 是平台相关的量，θ_c 才是几何量）。**实测**。
    theta_c, n_probe = _covering_radius(views, cen, elev_max_deg)
    out = dict(part=pth.name, n_views=int(n_views), theta_c=float(theta_c),
               theta_c_model=2.2 / np.sqrt(n_views), theta_c_probes=n_probe,
               voxel_mm=h * 1e3, margin_px=float(margin),
               mask_noise_px=float(mask_noise_px), pose_noise_px=float(pose_noise_px),
               cam_dist_mm=cam_dist_m * 1e3, seed=int(seed),
               elev_max_deg=elev_max_deg,
               aim_offset_mm=(None if aim_offset_m is None
                              else [round(float(x) * 1e3, 1) for x in aim_offset_m]),
               k1=float(k1), board_plane=bool(board_plane),
               v_gt_mm3=v_gt, v_hull_mm3=v_hull, concavity=conc,
               n_tri=int(len(tris)), directed_edge_viol=chk["directed_edge_viol"],
               boundary_edges=chk["boundary_edges"], winding_ok=chk["winding_ok"],
               carve_auto=info, sec=round(time.time() - t0, 1))
    out.update(met)
    # 图像域审计：预测的内缩是 `margin + 0.42` px（§8.2 的实测像素化偏置）
    out.update(silhouette_audit(tris, views, max_views=audit_views, verbose=verbose))
    out["aud_pred_inset_px"] = float(margin) + PIXEL_BIAS_PX
    out["sec"] = round(time.time() - t0, 1)
    if verbose:
        print(f"    {pth.stem:24s} N={n_views:<4d} h={h*1e3:.2f} m={margin:.2f}px "
              f"σm={mask_noise_px:.1f} σp={pose_noise_px:.1f}  "
              f"IoU {met['iou']:.3f}/{met['iou_ceiling']:.3f}  "
              f"V_R/V_G {met['vol_ratio']:.3f}≤{met['vol_ceiling']:.3f}  "
              f"漏(体) {met['miss_vol']*100:5.2f}% [上界 {met['ub_miss_vol']*100:5.2f}%] "
              f"漏(面) {met['miss_surf']*100:5.2f}%  假阳 {met['false_pos']*100:5.2f}%  "
              f"戳出 p95 {met['p95_protrude_mm']:.3f}/中位 {met['med_protrude_mm']:.3f}mm "
              f"[预测 {met['pred_p95_mm']:.3f}/{met['pred_med_mm']:.3f}]  {out['sec']}s")
    return out


# ------------------------------------------------------------------ 自检

def selftest():
    """先验「体素化真值」这条新代码，否则后面所有指标都是空中楼阁。

    验三件事：
      ① 体素化出的体积 ≈ CAD 的解析体积（差在体素量化尺度内）；
      ② 体素化的**剪影**（沿视线投影）与渲染器的掩码一致 —— 两条互不依赖的路
         （一条射线穿刺、一条 z-buffer 光栅）给出同一个剪影，这才叫不循环；
      ③ 格点对齐（体素化的 `origin`/`h` 必须就是 `carve` 的那一套）。
    """
    ok = True
    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    f_px = 0.5 * (K[0, 0] + K[1, 1])
    for part in ("sts3215_03a_v1", "moving_jaw_so101_v1", "Trigger_SO101"):
        gt, pth = MFV.load_part(part)
        gtc = gt.reshape(-1, 3)
        lo, hi = gtc.min(0), gtc.max(0)
        ctr = 0.5 * (lo + hi)
        gtz = MR.transform(gt, np.block([[np.eye(3), (-ctr)[:, None]], [np.zeros(3), 1.0]]))
        v_gt, v_hull, conc = hull_stats(gtz)
        h = 0.25e-3
        pad = 4 * h
        origin = np.asarray(gtz.reshape(-1, 3).min(0) - pad, float)
        shape = [int(np.ceil((gtz.reshape(-1, 3).max(0)[i] + pad - origin[i]) / h)) + 1
                 for i in range(3)]
        G = voxelize_mesh(gtz, origin, h, shape)
        v_vox = float(G.sum()) * (h * 1e3) ** 3
        rel = (v_vox - v_gt) / v_gt
        # ① 体积
        ok1 = abs(rel) < 0.02
        # ② 剪影：射线穿刺出的占据体沿视线投影，**换算成物理面积**后与光栅器的掩码比。
        #    两条实现完全独立（一条扫描线奇偶填充、一条 z-buffer 光栅），所以这不是自证。
        #    比的是面积(mm²)而不是像素数 —— 两者的像素网格尺寸不同，没法比像素。
        dist_m = 1.0
        cam = MR.Cam(K, np.zeros(5),
                     MR.look_at(np.array([0, 0, dist_m]), np.zeros(3), up_world=(0, 1, 0)),
                     640, 480)
        ras = MR.rasterize(gtz, cam, cull=False)["mask"]
        A_vox_mm2 = float(G.any(axis=2).sum()) * (h * 1e3) ** 2
        mmpp = dist_m / f_px * 1e3                              # 物面每个像素几 mm
        A_ras_mm2 = int(ras.sum()) * mmpp ** 2
        relA = A_vox_mm2 / max(A_ras_mm2, 1e-9) - 1
        # 容差 15%：光栅器的掩码边界有 ~0.5px 的单边内缩（§8.2），换算到物面约 1.2mm/px，
        # 对 ~25px 宽的剪影就是百分之几的面积差。这是**已知偏差**，不是要抹掉的误差。
        ok2 = abs(relA) < 0.15
        print(f"  [selftest] {pth.stem:26s} 凹度 {conc:.3f}｜解析 V {v_gt:.1f}mm³｜"
              f"体素 V {v_vox:.1f}mm³（{rel*100:+.2f}%）→ {'通过' if ok1 else '**不通过**'}"
              f"｜投影面积 {A_vox_mm2:.0f}mm² vs 光栅掩码 {A_ras_mm2:.0f}mm²"
              f"（{relA*100:+.1f}%）→ {'通过' if ok2 else '**不通过**'}")
        ok &= ok1 & ok2
    print("\n体素化自检通过" if ok else "\n**体素化自检不通过**")
    return 0 if ok else 1


# ------------------------------------------------------------------ 实验编排

# 标称点：N=50、margin=1px、边界噪声 0.5px、位姿噪声 0.3px（charuco 重投影残差量级）。
#
# **体素取 1.0mm 而不是 0.5mm**，理由是实测的：`carve` 的开销 ∝ 体素数 × 视角数，而凹件的
# 细格包围盒被 VH 撑到接近凸包（`motor_holder` 的 V_H/V_G 上限是 3.8），h=0.5mm 一个 run 要
# 十几分钟；而**主指标已经换成图像域**（内缩在 h=0.5 与 h=1.0 下测得 0.638 vs 0.647px，
# 差 1.4%），体素域只用来做「集合包含」和「上限对比」这类**比值**判断，1mm 足够。
# h 的收敛性单独由 `vox|*` 三个点报出来，不混进扫描里当混淆因子。
NOMINAL = dict(n_views=50, h=1.0e-3, margin=1.0, mask_noise_px=0.5, pose_noise_px=0.3)

# 舵机在画面里的 0.38 半幅偏移（位姿误差与畸变都在离轴时才真正被考到）
OFFAXIS = (0.060, -0.045, 0.0)
K1 = -0.345                                             # 本机 front 相机实测的 k1


def plan(quick=False):
    """把全部配置**在开跑前**枚举出来 —— 图里画的每个点都对应一个具名的 job。

    返回 `[(tag, part, kwargs)]`。`tag` 也是缓存键，所以重跑会跳过已完成的。
    """
    P4 = list(DEFAULT_PARTS)
    # 视角数扫**不含 50**：`marg|{p}|0.0`（N=50 / margin=0，其余全默认）与 `N|{p}|50` 逐参数
    # 相同，所以复用那一格当视角扫的密端（图①要的正是「密端」），避免同一配置跑两遍。
    Ns = (4, 8, 20) if quick else (4, 8, 20, 100)
    ms = (0.0, 1.0, 2.0) if quick else (0.0, 0.5, 1.0, 2.0, 4.0)
    hs = (1.0e-3, 0.5e-3) if quick else (1.0e-3, 0.5e-3, 0.25e-3)
    mc = 4 if quick else 16
    jobs = []
    # ① 视角数扫（4 个件）—— 横轴用覆盖半径 θ_c 而不是 N
    for p in P4:
        for n in Ns:
            jobs.append((f"N|{p}|{n}", p, dict(n_views=n, margin=0.0)))
    # ② 标称点（4 个件）—— 主图就是这 4 个点
    for p in P4:
        jobs.append((f"nom|{p}", p, dict(**NOMINAL)))
    # ③ margin 扫（4 个件）—— 验「增益恒为 1」与「体积损失斜率 → A_bind/A_gt」
    for p in P4:
        for mm in ms:
            jobs.append((f"marg|{p}|{mm}", p, dict(n_views=50, margin=mm)))
    # ④ 噪声网格（舵机）—— 边界噪声 × 位姿噪声
    for sm in (0.0, 0.5, 1.5):
        for sp in (0.0, 0.3):
            jobs.append((f"noise|{sm}|{sp}", "sts3215_03a_v1",
                         dict(n_views=50, margin=1.0, mask_noise_px=sm, pose_noise_px=sp)))
    # ⑤ 体素收敛（舵机）—— 这三个点是**唯一**改动 h 的地方，用来证明结论不是分辨率造出来的
    for h in hs:
        jobs.append((f"vox|{h:g}", "sts3215_03a_v1", dict(n_views=50, margin=0.0, h=h)))
    # ⑥ 离轴 + 畸变（舵机）—— margin 是**像素域**腐蚀，离轴+畸变时还各向同性吗
    jobs.append(("offaxis", "sts3215_03a_v1",
                 dict(n_views=50, margin=1.0, aim_offset_m=OFFAXIS, k1=K1)))
    # ⑦ 上半球 + 板面先验（舵机）—— 真采集的真实约束，免费补掉底面缺口
    jobs.append(("hemi", "sts3215_03a_v1",
                 dict(n_views=50, margin=1.0, elev_max_deg=90.0, board_plane=True)))
    # ⑧ 标称点的蒙特卡洛 —— 给误差棒（报中位与 IQR，不报单个种子）
    for s in range(mc):
        jobs.append((f"mc|{s}", "sts3215_03a_v1", dict(**NOMINAL, seed=1000 + s)))
    return jobs


# 这些 job 的网格要落盘，给图⑤（剪影叠加）和其它离线图用
SAVE_MESH_TAG = ("nom|", "offaxis", "hemi", "vox|0.0005")


def run_all(quick=False, force=False, mesh_dir=None, verbose=True):
    """跑完全部 job。**逐条落盘**（跑一半崩了不丢已完成的），且可续跑。"""
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ("runs_quick.json" if quick else "runs.json")
    done = {}
    if cache.exists() and not force:
        try:
            done = {r["tag"]: r for r in json.loads(cache.read_text())["runs"]}
            if verbose:
                print(f"[缓存] {cache.name} 里已有 {len(done)} 条，跳过它们")
        except Exception as e:                                  # 缓存坏了就重跑，但要说出来
            print(f"[缓存] 读不了 {cache.name}（{e}）→ 全部重跑")
            done = {}
    if mesh_dir:
        Path(mesh_dir).mkdir(parents=True, exist_ok=True)

    def stl_of(tag):
        if not mesh_dir or not tag.startswith(SAVE_MESH_TAG):
            return None
        return Path(mesh_dir) / (tag.replace("|", "_").replace(".", "p") + ".stl")

    jobs = plan(quick)
    runs = []
    t0 = time.time()
    for i, (tag, part, kw) in enumerate(jobs, 1):
        stl = stl_of(tag)
        # 缓存的条目里没有网**格**（网格额外存 STL），所以要么网格已在盘上，要么重跑
        if tag in done and not (stl is not None and not stl.exists()):
            runs.append(done[tag])
            continue
        print(f"[{i}/{len(jobs)}] {tag}", flush=True)
        r = run_once(part, verbose=verbose, audit_views=(8 if quick else 16),
                     stl_out=stl, **kw)
        r["tag"] = tag
        runs.append(r)
        cache.write_text(json.dumps(dict(quick=quick, runs=runs), indent=1, default=str))
    print(f"\n[完成] {len(runs)} 条，墙钟 {(time.time()-t0)/60:.1f} 分钟 → {cache}")
    return runs


# ------------------------------------------------------------------ 出图

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import font_manager                              # noqa: E402

# 配色取自 dataviz 的**已验证默认实例**（`references/palette.md` 的分类槽位，
# 按固定顺序取用、不循环）。本机没有 node，跑不了那个校验脚本，所以不去改动槽位值，
# 只用它的原始值 —— 那个文件本身就是校验过的产物。
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK1, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURF, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


def _mpl_init():
    plt.rcParams.update({
        "font.family": [f.name for f in font_manager.fontManager.ttflist
                        if f.name == "Noto Sans CJK JP"] or ["DejaVu Sans"],
        "axes.unicode_minus": False, "figure.dpi": 130,
        "savefig.facecolor": SURF, "axes.facecolor": SURF, "figure.facecolor": SURF,
        "axes.edgecolor": AXIS, "text.color": INK1,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK2, "font.size": 9.5,
    })


def _ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, name, note):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    print(f"[图] {p}  — {note}")
    if not p.exists():
        raise OSError(f"图没写出来：{p}")


def _by(runs, tag):
    for r in runs:
        if r["tag"] == tag:
            return r
    return None


def _bypref(runs, pref):
    return [r for r in runs if r["tag"].startswith(pref)]


def fig_sparse_dense(runs):
    """① 视角数：单边性**翻号**，以及假阳怎么随视角变密而消失（舵机）。

    刻意不画剪影 IoU 当主指标：实测它**非单调**（N=4/8/20/50 分别 0.977/0.978/0.972/0.963），
    在中间取最大。原因是 IoU 对称地罚两个方向，而 VH 在视角少时是「外扩」、视角密时是
    「内缩」，两头都掉 IoU。会随视角数单调变的量是**假阳体积**，而「单边往哪边偏」要看
    带符号的内缩。所以这一张图画这两个，IoU 只进表。

    密端（N=50）取自 `marg|…|0.0`：它与 `N|…|50` 逐参数相同，所以视角扫**不另跑 50**。
    """
    rs = [r for r in _bypref(runs, "N|sts3215_03a_v1|") if r]
    # 密端（N=50 / margin=0）复用 margin 扫里的那一格：逐参数与 `N|…|50` 相同
    p50 = _by(runs, "marg|sts3215_03a_v1|0.0")
    if p50 is not None:
        rs.append(p50)
    assert all(r["margin_px"] == 0.0 for r in rs), "视角扫必须全在 margin=0 上"
    rs.sort(key=lambda r: r["theta_c"])
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    x = [r["theta_c"] for r in rs]
    ax = axes[0]
    _ax(ax)
    y = [r["aud_inset_px"] for r in rs]
    ax.axhline(0.0, color=INK2, lw=1.4, zorder=2)
    ax.plot(x, y, "-o", color=S1, lw=2.2, ms=8, mfc=S1, mec=SURF, mew=1.3, zorder=4)
    # 翻号点：线性插值出内缩穿过 0 的位置；视角数直接用**夹住它的那两个 run** 报，
    # 不去套 `N≈(2.2/θ_c)²` 那个模型式（θ_c 是实测的，且实测值是 1.21× 这个估计式）
    cross = None
    for i in range(len(x) - 1):
        if y[i] * y[i + 1] < 0:
            t = y[i] / (y[i] - y[i + 1])
            cross = x[i] + t * (x[i + 1] - x[i])
            n_dense, n_sparse = rs[i]["n_views"], rs[i + 1]["n_views"]
    if cross is not None:
        ax.axvline(cross, color=S2, ls="--", lw=1.5, zorder=3)
        ax.annotate(f"翻号 θ_c≈{cross:.2f}\n（夹在 N={n_dense} 与 {n_sparse} 之间）", (cross, 0),
                    textcoords="offset points", xytext=(9, 13), color=S2,
                    fontsize=9, fontweight="bold")
    # 注意方向：数组按 θ_c **升序**，所以 x[0] 是**最密**的一端。内缩为正 = 重建偏小。
    # 两条说明用**引线**画到头顶的空白区（x=1.31 是曲线最右端，竖直引线不会穿过曲线）
    lead = dict(arrowstyle="-", color=MUTED, lw=1.0, shrinkA=2, shrinkB=3)
    ax.set_ylim(min(y) - 0.16, max(y) + 0.34)
    ax.annotate("视角密：外壳贴紧，像素化偏置主导\n⇒ 净内缩（只会低估）",
                xy=(x[0], y[0]), xytext=(x[0], max(y) + 0.32), ha="left", va="top",
                color=INK2, fontsize=9, arrowprops=lead)
    ax.annotate("视角少：剪影像素化把切锥**往外**拱\n⇒ 净外扩（还会高估）",
                xy=(x[-1], y[-1]), xytext=(x[-1], max(y) + 0.32), ha="right", va="top",
                color=INK2, fontsize=9, arrowprops=lead)
    ax.set_xscale("log")
    ax.set_xlabel("实测覆盖半径 θ_c（弧度，log；左密右疏）")
    ax.set_ylabel("单边内缩（px，正 = 重建比真值小）")
    ax.set_title("单边性会翻号：只有视角密的时候才「只会低估」", color=INK1,
                 fontsize=10.5, pad=9)

    ax = axes[1]
    _ax(ax)
    fa = [r["false_pos"] * 100 for r in rs]
    mi = [r["miss_vol"] * 100 for r in rs]
    ax.plot(x, fa, "-o", color=S2, lw=2.2, ms=8, mfc=S2, mec=SURF, mew=1.3, zorder=4)
    ax.plot(x, mi, "-o", color=S1, lw=2.2, ms=8, mfc=S1, mec=SURF, mew=1.3, zorder=4)
    ax.set_ylim(min(mi + fa) * 0.55, max(fa) * 2.0)
    ax.annotate("假阳（多出来的体积）", xy=(x[-1], fa[-1]), xytext=(x[-1], max(fa) * 1.7),
                ha="right", va="top", color=S2, fontsize=9, fontweight="bold",
                arrowprops=lead)
    ax.annotate("漏（真值没被盖住）", xy=(x[0], mi[0]), xytext=(x[0], max(fa) * 0.95),
                ha="left", va="top", color=S1, fontsize=9, fontweight="bold",
                arrowprops=lead)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("实测覆盖半径 θ_c（弧度，log）")
    ax.set_ylabel("体积占比（%，log）")
    ax.set_title("视角变密的代价：假阳掉一个量级，漏开始抬头", color=INK1,
                 fontsize=10.5, pad=9)
    ax.legend(frameon=False, labelcolor=INK2, loc="center right")
    fig.suptitle("舵机，margin=0（隔离纯像素化效应）：视角数与单边性的关系",
                 color=INK1, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "validate_sparse_vs_dense.png", "视角数与单边性翻号")


def _abind_ratio(runs, part):
    """从 margin 扫描反解 `A_bind/A_gt`：实测体积漏的斜率 ÷ 薄壳上界斜率。

    上界斜率假设**整个表面**都是剪影的约束边（`A=A_gt`）；实测斜率更小，因为凹面在任何
    视角都不构成剪影边界、不跟着内缩。两者的比值就是「真正起约束作用的表面占比」——
    这是**量出来的**，不是设出来的。
    """
    rs = sorted([r for r in _bypref(runs, f"marg|{part}|")
                 if r["margin_px"] <= 2.0], key=lambda r: r["margin_px"])
    if len(rs) < 2:
        return None
    x = np.array([r["margin_px"] for r in rs])
    sl = float(np.polyfit(x, [r["miss_vol"] for r in rs], 1)[0])
    ub = rs[0]["area_gt_mm2"] / rs[0]["v_gt_mm3"] * rs[0]["mm_per_px_obj"]
    return sl / ub if ub > 0 else None


def fig_concavity(runs):
    """② 主图（1×3）：凹度决定「上限」，也决定「表面有多少真的在起约束」。"""
    rs = [r for r in (_by(runs, f"nom|{p}") for p in DEFAULT_PARTS) if r]
    if not rs:
        return
    rs = sorted(rs, key=lambda r: r["concavity"])
    conc = [r["concavity"] for r in rs]
    # 预测要**和被测点同条件**：标称点开着两项噪声，所以要加上噪声贡献（舵机实测的那一份）。
    # 用舵机的常数去预测另外三个件，正是这一格要考的推广性。
    n_clean = _by(runs, "noise|0.0|0.0")
    n_nom = _by(runs, "nom|sts3215_03a_v1")
    dz = (n_nom["aud_inset_px"] - n_clean["aud_inset_px"]) if (n_clean and n_nom) else 0.0
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    ax = axes[0]
    _ax(ax)
    pred = [r["aud_pred_inset_px"] + dz for r in rs]
    ax.plot(conc, [r["aud_inset_px"] for r in rs], "-o", color=S1, lw=2.0, ms=8,
            mfc=S1, mec=SURF, mew=1.3, zorder=4)
    ax.plot(conc, [r["aud_inset_perim_px"] for r in rs], "-s", color=S3, lw=1.8,
            ms=7, mfc=S3, mec=SURF, mew=1.2, zorder=3)
    ax.plot(conc, pred, "--", color=MUTED, lw=1.5, zorder=2)
    ax.annotate("等效半径差（主）", (conc[0], rs[0]["aud_inset_px"]),
                textcoords="offset points", xytext=(6, -14), color=S1, fontsize=9,
                fontweight="bold")
    ax.annotate("面积/周长（分母偏大 ⇒ 低估）", (conc[-1], rs[-1]["aud_inset_perim_px"]),
                textcoords="offset points", xytext=(-6, -15), ha="right", color=S3,
                fontsize=9)
    ax.annotate(f"预测 = margin {1.0:+.0f} + 像素化 {PIXEL_BIAS_PX:.2f}（舵机）"
                f" + 噪声 {dz:.2f}（舵机实测）",
                (conc[-1], pred[-1]), textcoords="offset points", xytext=(-6, 8),
                ha="right", color=MUTED, fontsize=8.5)
    ax.set_xlabel("凹度 = V_gt/V_凸包（越小越凹）")
    ax.set_ylabel("单边内缩（px）")
    ax.set_title("内缩量：两个估计量一起画\n（差 = 等效半径对非凸形状的形状偏置）",
                 color=INK1, fontsize=10, pad=8)

    ax = axes[1]
    _ax(ax)
    ax.plot(conc, [r["miss_vol"] * 100 for r in rs], "-o", color=S1, lw=2.0, ms=8,
            mfc=S1, mec=SURF, mew=1.3, zorder=4)
    ax.plot(conc, [r["false_pos"] * 100 for r in rs], "-o", color=S2, lw=2.0, ms=8,
            mfc=S2, mec=SURF, mew=1.3, zorder=4)
    for r in rs:
        ax.annotate(f"{r['miss_vol']*100:.1f}", (r["concavity"], r["miss_vol"] * 100),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    color=S1, fontsize=8.5)
        ax.annotate(f"{r['false_pos']*100:.1f}", (r["concavity"],
                                                 r["false_pos"] * 100),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    color=S2, fontsize=8.5)
    ax.annotate("漏（体积）", (conc[-1], rs[-1]["miss_vol"] * 100),
                textcoords="offset points", xytext=(-6, 6), ha="right", color=S1,
                fontsize=9, fontweight="bold")
    ax.annotate("假阳（体积）", (conc[0], rs[0]["false_pos"] * 100),
                textcoords="offset points", xytext=(8, 6), color=S2, fontsize=9,
                fontweight="bold")
    ax.set_xlabel("凹度 = V_gt/V_凸包（越小越凹）")
    ax.set_ylabel("占真值/重建的体积比例（%）")
    ax.set_title("标称点：凹件上「漏」反而更小\n（凹面本来就不在剪影边界上）",
                 color=INK1, fontsize=10, pad=8)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")

    ax = axes[2]
    _ax(ax)
    rat = [(r["concavity"], _abind_ratio(runs, r["part"][:-4])) for r in rs]
    rat = [(c, v) for c, v in rat if v is not None]
    if rat:
        ax.plot([c for c, _ in rat], [v for _, v in rat], "-o", color=S1, lw=2.0,
                ms=8, mfc=S1, mec=SURF, mew=1.3, zorder=4)
        for c, v in rat:
            ax.annotate(f"{v:.2f}", (c, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", color=S1, fontsize=8.5)
        ax.axhline(1.0, color=MUTED, ls="--", lw=1.5, zorder=2)
        ax.annotate("若整个表面都是剪影约束边", (np.mean(conc), 1.0),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    color=MUTED, fontsize=9)
        ax.set_ylim(0, 1.25)
    ax.set_xlabel("凹度 = V_gt/V_凸包（越小越凹）")
    ax.set_ylabel("A_bind/A_gt（反解实测斜率）")
    ax.set_title("真正起约束的表面占比\n（从 margin 斜率反解，不是设出来的）",
                 color=INK1, fontsize=10, pad=8)
    fig.suptitle("标称点 N=50 / margin=1px / σ_mask=0.5 / σ_pose=0.3，体素 1.0mm "
                 "（四件按凹度 0.90→0.26 排开）", color=INK1, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "validate_concavity.png", "凹度：上限 / 单边性 / 约束表面占比")


def fig_inset(runs):
    """③ 图像域内缩 vs margin —— 验「增益恒为 1px/px」和像素化偏置的大小。

    **主仪器是等效半径口径**：它的增益实测 0.999（定义值 1.000），截距 0.590px 就是像素化
    偏置。两个周长口径画出来当对照 —— 它们的增益只有 0.744/0.774，因为它们的分母是**数字化
    周长**（边界像素计数，比几何周长大约 1.3×）。这条差异是这一轮抓出来的：要是没有
    「增益必须等于 1」这把尺子，那个 0.74 会被当成「理论不成立」。
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _ax(ax)
    rs = sorted(_bypref(runs, "marg|sts3215_03a_v1|"), key=lambda r: r["margin_px"])
    x = np.array([r["margin_px"] for r in rs])
    if not len(x):
        plt.close(fig)
        return
    y = np.array([r["aud_inset_px"] for r in rs])
    yp = np.array([r["aud_inset_perim_px"] for r in rs])
    gain = float(np.polyfit(x, y, 1)[0])
    bias = float(np.interp(0.0, x, y))
    gain_p = float(np.polyfit(x, yp, 1)[0])
    bias_p = float(np.interp(0.0, x, yp))
    ax.plot(x, y, "-o", color=S1, lw=2.0, ms=8, mfc=S1, mec=SURF, mew=1.3, zorder=4)
    ax.plot(x, yp, "-s", color=S3, lw=1.8, ms=7, mfc=S3, mec=SURF, mew=1.2, zorder=3)
    ax.plot(x, x + bias, "--", color=MUTED, lw=1.5, zorder=2)
    ax.annotate("等效半径差（主仪器：增益 = 1.00）", (x[-1], y[-1]),
                textcoords="offset points", xytext=(-8, 10), ha="right", color=S1,
                fontsize=9, fontweight="bold")
    ax.annotate("面积/周长（增益只 0.74：数字化周长偏大）", (x[-1], yp[-1]),
                textcoords="offset points", xytext=(-8, -20), ha="right", color=S3,
                fontsize=9)
    ax.annotate(f"三口径的增益（定义值 1.000）\n"
                f"等效半径 {gain:.3f} → 截距 {bias:+.2f}px\n"
                f"面积/周长 {gain_p:.3f} → 截距 {bias_p:+.2f}px",
                (x[0], yp[0]), textcoords="offset points", xytext=(12, -46),
                color=INK2, fontsize=8.5)
    ax.set_xlabel("margin（图像域腐蚀，px）")
    ax.set_ylabel("重建相对真值的单边内缩（px）")
    ax.set_title(f"margin 的增益恒为 1（定义使然）；截距 = 像素化偏置 {bias:.2f}px"
                 f"（球是 0.42px）",
                 color=INK1, fontsize=10.5, pad=10)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    _save(fig, "validate_inset_vs_margin.png", "内缩增益与偏置")


def fig_onesided(runs):
    """④ 单边性：margin 与边界噪声各贡献多少（舵机）。"""
    rs = sorted(_bypref(runs, "marg|sts3215_03a_v1|"), key=lambda r: r["margin_px"])
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    ax = axes[0]
    _ax(ax)
    x = np.array([r["margin_px"] for r in rs])
    miss = np.array([r["miss_vol"] for r in rs]) * 100
    false = np.array([r["false_pos"] for r in rs]) * 100
    ub = np.array([r["ub_miss_vol"] for r in rs]) * 100
    ax.plot(x, ub, "--", color=MUTED, lw=1.5, zorder=2)
    ax.plot(x, miss, "-o", color=S1, lw=2.2, ms=8, mfc=S1, mec=SURF, mew=1.3, zorder=4)
    ax.plot(x, false, "-o", color=S2, lw=2.2, ms=8, mfc=S2, mec=SURF, mew=1.3, zorder=4)
    sl = float(np.polyfit(x, miss, 1)[0])
    ub_sl = rs[0]["area_gt_mm2"] / rs[0]["v_gt_mm3"] * rs[0]["mm_per_px_obj"] * 100
    ax.annotate("漏（真值没被盖住）", (x[-1], miss[-1]), textcoords="offset points",
                xytext=(-8, 12), ha="right", color=S1, fontsize=9, fontweight="bold")
    ax.annotate("假阳（多出来的）", (x[-1], false[-1]), textcoords="offset points",
                xytext=(-8, -16), ha="right", color=S2, fontsize=9, fontweight="bold")
    ax.annotate("薄壳上界（A=A_gt）", (x[-1], ub[-1]), textcoords="offset points",
                xytext=(-8, 8), ha="right", color=MUTED, fontsize=9)
    ax.annotate(f"实测 {sl:.1f}%/px vs 上界 {ub_sl:.1f}%/px\n⇒ A_bind/A_gt ≈ {sl/ub_sl:.2f}",
                (x[0], miss[0]), textcoords="offset points", xytext=(8, 26),
                color=S1, fontsize=9)
    ax.set_xlabel("margin（px）")
    ax.set_ylabel("体积比例（%）")
    ax.set_title("margin 的代价：漏线性上涨（外推到 0 仍有 3%=像素化底噪）",
                 color=INK1, fontsize=10, pad=9)
    ax.legend(frameon=False, labelcolor=INK2, loc="center left")

    ax = axes[1]
    _ax(ax)
    ax.set_title("边界噪声：σ_mask 的代价是「多出来的体积」\n降得快、"
                 "「漏」涨得慢（VH 取交 ⇒ 单边）", color=INK1, fontsize=10, pad=9)
    xs = [0.0, 0.5, 1.5]
    for sp, c, lab in ((0.0, S2, "σ_pose 0"), (0.3, S1, "σ_pose 0.3px")):
        rr = [_by(runs, f"noise|{sm}|{sp}") for sm in xs]
        rr = [r for r in rr if r]
        if len(rr) >= 2:
            ax.plot([r["mask_noise_px"] for r in rr],
                    [r["miss_vol"] * 100 for r in rr], "-o", color=c, lw=2.2, ms=8,
                    mfc=c, mec=SURF, mew=1.3, zorder=4, label=f"漏 {lab}")
            ax.plot([r["mask_noise_px"] for r in rr],
                    [r["false_pos"] * 100 for r in rr], "--o", color=c, lw=1.6, ms=7,
                    mfc=SURF, mec=c, mew=1.4, zorder=3, label=f"假阳 {lab}")
    ax.set_xlabel("σ_mask（剪影边界噪声，px）")
    ax.set_ylabel("体积比例（%）")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left", ncols=2,
              fontsize=8.5)
    fig.suptitle("舵机：单边内缩的两个来源（margin 是可控腐蚀，噪声是不可控的观测误差）",
                 color=INK1, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "validate_onesided.png", "漏 vs 假阳（margin 与噪声）")


def fig_silhouette_overlay(runs):
    """⑤ 剪影叠加：肉眼看「单边内缩」长什么样（网格从 `calib/recon/validate/meshes/` 读）。"""
    cases = [f"nom|{p}" for p in DEFAULT_PARTS] + ["offaxis", "hemi"]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6))
    K = np.array([[409.57, 0, 321.71], [0, 409.57, 209.72], [0, 0, 1.0]])
    for ax, tag in zip(axes.ravel(), cases):
        stl = OUT / "meshes" / (tag.replace("|", "_") + ".stl")
        r = _by(runs, tag)
        if not stl.exists() or r is None:
            ax.axis("off")
            ax.set_title(f"{tag}：网格或记录缺", color=MUTED, fontsize=9)
            continue
        tris = np.asarray(MFV.MR.stl_cached(stl), float)
        part = tag.split("|")[1] if "|" in tag else "sts3215_03a_v1"
        gt, _ = MFV.load_part(part)
        c = 0.5 * (gt.reshape(-1, 3).min(0) + gt.reshape(-1, 3).max(0))
        gtz = MR.transform(gt, np.block([[np.eye(3), (-c)[:, None]], [np.zeros(3), 1.0]]))
        # 相机**另取一台**（不是 job 用过的那几台）：相当于留出视角，比拿训练视角自评更强。
        # 该 job 特有的约束（畸变、离轴、仰角限制）照搬，否则就不是同一个问题了。
        kw = dict(cam_dist_m=r["cam_dist_mm"] / 1e3, seed=7, n_views=1,
                  elev_max_deg=(r["elev_max_deg"] if r["elev_max_deg"] is not None
                                else 30.0))
        dist = np.array([r["k1"], 0, 0, 0, 0.0])
        if r["aim_offset_mm"] is not None:
            kw["jitter_centre"] = np.array(r["aim_offset_mm"]) / 1e3
        views, _ = MFV.synthetic_views(gtz, K, dist, **kw)
        v = views[0]
        cam = MR.Cam(v["K"], v["dist"], v["T_world_cam"], v["W"], v["H"])
        rec = MR.rasterize(tris, cam, cull=False)["mask"]
        m = v["mask"]
        rgb = np.repeat((m.astype(np.uint8) * 55 + 45)[..., None], 3, -1)
        st3 = np.ones((3, 3), bool)
        rgb[m & ~ndimage.binary_erosion(m, st3, border_value=0)] = (43, 120, 214)
        rgb[rec & ~ndimage.binary_erosion(rec, st3, border_value=0)] = (235, 104, 52)
        ys, xs = np.nonzero(m | rec)
        if len(ys):
            rgb = rgb[max(ys.min() - 12, 0):ys.max() + 13,
                      max(xs.min() - 12, 0):xs.max() + 13]
        ax.imshow(rgb.astype(np.uint8), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{tag.replace('|', ' ')}   IoU {r['aud_iou']:.3f}\n"
                     f"漏 {r['aud_miss_px']:.0f}px  多 {r['aud_extra_px']:.0f}px",
                     color=INK1, fontsize=8.5)
    fig.suptitle("蓝 = 真值剪影轮廓，橙 = 重建剪影轮廓；橙线整体落在蓝线内侧 = 单边内缩",
                 color=INK1, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "validate_silhouette_overlay.png", "剪影叠加（肉眼回看）")


def fig_sensitivity(runs):
    """⑥ 敏感度：每项扰动**只改一个参数**、与一个配对参考点比，量它对 IoU 的效应。

    ⚠️ 这一张图的第一版是错的：它把每个 job 都与**标称点**比，而 `vox|*` 两个 job 跑在
    `margin=0` 且无噪声上 —— 于是「体素 1.0mm」那根 +6pp 的柱子量的其实是 margin
    （1→0）加噪声关掉，不是体素。现在每根柱子都是**配对差**：参考点除该项外与它逐参数相同。
    """
    def pair(ref_tag, tag, lab):
        a, b = _by(runs, ref_tag), _by(runs, tag)
        if not a or not b:
            return None
        return (lab, (b["aud_iou"] - a["aud_iou"]) * 100,
                b["aud_inset_px"] - a["aud_inset_px"])

    # 配对参考点：`n00` = N50/margin1/无噪声；`v1` = N50/margin1/无噪声（体素 1.0mm）
    items = [x for x in (
        pair("vox|0.001", "vox|0.0005", "体素 1.0→0.5mm"),
        pair("marg|sts3215_03a_v1|0.0", "marg|sts3215_03a_v1|1.0", "margin 0→1px"),
        pair("marg|sts3215_03a_v1|1.0", "marg|sts3215_03a_v1|2.0", "margin 1→2px"),
        pair("noise|0.0|0.0", "noise|0.5|0.0", "σ_mask 0→0.5px"),
        pair("noise|0.5|0.0", "noise|1.5|0.0", "σ_mask 0.5→1.5px"),
        pair("noise|0.0|0.0", "noise|0.0|0.3", "σ_pose 0→0.3px"),
        pair("noise|0.0|0.0", "offaxis", "离轴 0.38 半幅 + k1"),
        pair("noise|0.0|0.0", "hemi", "上半球 + 板面"),
        pair("noise|0.0|0.0", "nom|sts3215_03a_v1", "两项噪声都打开"),
    ) if x]
    mc = _bypref(runs, "mc|")
    if len(mc) >= 4:
        io = np.array([r["aud_iou"] for r in mc]) * 100
        items.append(("蒙特卡洛 IQR（标称点）",
                      0.5 * (np.percentile(io, 75) - np.percentile(io, 25)), 0.0))
    if not items:
        return
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    _ax(ax)
    y = np.arange(len(items))[::-1]
    vals = np.array([v for _, v, _ in items])
    ax.barh(y, vals, height=0.62,
            color=[S1 if v >= 0 else S2 for v in vals], zorder=3)
    ax.axvline(0.0, color=INK2, lw=1.3, zorder=4)
    span = max(abs(vals).max(), 1e-9)
    ax.set_xlim(-1.45 * span, 1.45 * span)
    for yy, (lab, v, dv) in zip(y, items):
        txt = f"{v:+.2f}pp" + (f"  内缩 {dv:+.2f}px" if abs(dv) > 1e-9 else "")
        ax.annotate(txt, (v, yy), textcoords="offset points",
                    xytext=(7 if v >= 0 else -7, 0), va="center",
                    ha="left" if v >= 0 else "right", color=INK2, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _, _ in items], color=INK2, fontsize=9)
    ax.set_xlabel("剪影 IoU 变化（百分点；每根柱子都是「配对差」）")
    ax.set_title("敏感度：按配对差排 —— 视角/余量/边界噪声是主项，体素与畸变是次要项",
                 color=INK1, fontsize=10.5, pad=10)
    _save(fig, "validate_sensitivity.png", "敏感度（配对差，非对单点）")


def make_figures(runs, quick=False):
    _mpl_init()
    fig_sparse_dense(runs)
    fig_concavity(runs)
    fig_inset(runs)
    fig_onesided(runs)
    fig_silhouette_overlay(runs)
    fig_sensitivity(runs)


# ------------------------------------------------------------------------ CLI

def report(runs, quick=False):
    """把关键数字打成一张表（可直接抄进 HANDOVER.md）。"""
    print("\n" + "=" * 108)
    print(f"{'job':26s} {'凹度':>6s} {'θ_c':>6s} {'IoU':>7s} {'上限':>7s} "
          f"{'V/V_G':>7s} {'≤':>7s} {'漏体%':>7s} {'假阳%':>7s} {'内缩px':>7s} "
          f"{'预':>6s} {'漏px':>6s} {'多px':>5s}")
    print("-" * 108)
    for r in runs:
        print(f"{r['tag']:26s} {r['concavity']:6.3f} {r['theta_c']:6.3f} "
              f"{r['aud_iou']:7.4f} {r['iou_ceiling']:7.4f} {r['vol_ratio']:7.4f} "
              f"{r['vol_ceiling']:7.4f} {r['miss_vol']*100:7.2f} "
              f"{r['false_pos']*100:7.2f} {r['aud_inset_px']:+7.3f} "
              f"{r['aud_pred_inset_px']:6.2f} {r['aud_miss_px']:6.0f} "
              f"{r['aud_extra_px']:5.0f}")
    print("-" * 108)

    # 「VH 不会跑出凸包」这条**不是定理**，只是个稠密视角下的极限。实测它会被越过，
    # 所以这里报的是**越过的位置**，而不是一句「通过/不通过」。
    over = [r for r in runs if r["vol_ratio"] > r["vol_ceiling"] + 1e-9]
    print(f"[凸包上限] V_R/V_G 越过 1/凹度的 job：{len(over)}/{len(runs)}"
          f"{'（都在视角少的那些档）' if over else ''}")
    if over:
        for r in sorted(over, key=lambda r: r["n_views"])[:4]:
            print(f"    {r['tag']:26s} N={r['n_views']:<4d} "
                  f"V_R/V_G {r['vol_ratio']:.3f} > 上限 {r['vol_ceiling']:.3f}"
                  f"  （假阳 {r['false_pos']*100:.1f}%）")
        print("    机制：切锥是**外切**于凸包的，稀疏视角下交出的多面体在角上拱出凸包；"
              "视角密了才落回下面。")
    bad = []
    # margin 增益：应当恒为 1.00 px/px（定义使然），斜率不对就是 margin 用错了
    rs = sorted(_bypref(runs, "marg|sts3215_03a_v1|"), key=lambda r: r["margin_px"])
    if len(rs) >= 2:
        x = np.array([r["margin_px"] for r in rs])
        y = np.array([r["aud_inset_px"] for r in rs])
        yp = np.array([r["aud_inset_perim_px"] for r in rs])
        ym = np.array([r["aud_inset_mid_px"] for r in rs])
        g, b = np.polyfit(x, y, 1)
        gp, bp = np.polyfit(x, yp, 1)
        gm, bm = np.polyfit(x, ym, 1)
        yv = np.array([r["miss_vol"] for r in rs])
        sl = float(np.polyfit(x, yv, 1)[0])
        ub = rs[0]["area_gt_mm2"] / rs[0]["v_gt_mm3"] * rs[0]["mm_per_px_obj"]
        print(f"[margin 增益] 斜率（定义值 1.000）：等效半径 {g:.3f} / 周长(真值) {gp:.3f} "
              f"/ 周长(平均) {gm:.3f} px/px")
        print(f"              截距：等效半径 {b:+.3f}（= 像素化偏置）/ 周长口径 "
              f"{bp:+.3f} / {bm:+.3f} px")
        print(f"              主仪器 = 等效半径：增益 {g:.3f} ≈ 1，预测常量 "
              f"PIXEL_BIAS_PX={PIXEL_BIAS_PX:.2f}px；§8.2 同口径的解析球是 0.42px"
              f" ⇒ 球偏乐观 ~40%。两个周长口径增益只有 ~0.75，"
              f"因为数字化周长（边界像素计数）比几何周长大 ~1.3x。")
        print(f"[A_bind/A_gt] 体积漏的斜率 {sl*100:.2f} %/px；"
              f"若整个表面都是剪影约束边则为 {ub*100:.2f} %/px ⇒ "
              f"反解 A_bind/A_gt = {sl/ub:.3f}")
    rs = [r for r in _bypref(runs, "N|sts3215_03a_v1|") if r]
    p50 = _by(runs, "marg|sts3215_03a_v1|0.0")     # 密端复用（逐参数同 N|…|50）
    if p50 is not None:
        rs.append(p50)
    rs.sort(key=lambda r: r["theta_c"])
    if rs:
        # IoU 在这里**非单调**（两头都掉），所以把它和方向性的量一起印出来
        print(f"[视角数] θ_c {rs[0]['theta_c']:.3f}（密）→{rs[-1]['theta_c']:.3f}（疏）rad："
              f"IoU {rs[0]['aud_iou']:.3f}→{rs[-1]['aud_iou']:.3f}"
              f"（上限 {rs[-1]['iou_ceiling']:.3f}）；"
              f"内缩 {rs[0]['aud_inset_px']:+.3f}→{rs[-1]['aud_inset_px']:+.3f}px"
              f"（正=重建偏小 ⇒ 密端内缩、疏端外扩）；"
              f"假阳 {rs[-1]['false_pos']*100:.1f}%→{rs[0]['false_pos']*100:.1f}%")
        for r in rs:
            print(f"               N={r['n_views']:>4d} θ_c={r['theta_c']:.3f} "
                  f"IoU={r['aud_iou']:.4f} 内缩={r['aud_inset_px']:+.3f}px "
                  f"（面积周长 {r['aud_inset_perim_px']:+.3f}px）"
                  f" 假阳={r['false_pos']*100:5.2f}% 漏={r['miss_vol']*100:5.2f}% "
                  f"V_R/V_G={r['vol_ratio']:.4f}")
    mc = _bypref(runs, "mc|")
    if len(mc) >= 4:
        for k, lab in (("aud_iou", "剪影 IoU"), ("miss_vol", "漏体比"),
                       ("vol_ratio", "V_R/V_G"), ("aud_inset_px", "内缩 px")):
            v = np.array([r[k] for r in mc], float)
            q1, q2 = np.percentile(v, [25, 75])
            print(f"[蒙特卡洛 {len(mc)} 次] {lab:10s} 中位 {np.median(v):.4f}  "
                  f"IQR [{q1:.4f}, {q2:.4f}]")
    print("=" * 108)
    return bad


def main():
    ap = argparse.ArgumentParser(description="拿 CAD 真值验证剪影雕刻")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="跑完整扫描（慢，可续跑）")
    ap.add_argument("--quick", action="store_true",
                    help="粗体素、少视角、少蒙特卡洛 —— 只为快速迭代")
    ap.add_argument("--force", action="store_true", help="忽略缓存全部重跑")
    ap.add_argument("--figs", action="store_true", help="只用已有缓存出图")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cache = OUT / ("runs_quick.json" if a.quick else "runs.json")
    if a.figs:
        runs = json.loads(cache.read_text())["runs"]
        make_figures(runs, a.quick)
        return 0
    if not a.run:
        ap.error("给 --run（跑扫描）或 --figs（只出图）或 --selftest")
    runs = run_all(quick=a.quick, force=a.force, mesh_dir=OUT / "meshes")
    bad = report(runs, a.quick)
    make_figures(runs, a.quick)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
