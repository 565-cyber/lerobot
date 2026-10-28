#!/usr/bin/env python
"""pose6d_swap.py —— 按 Pose6DAug 的**分层**做法换物体：抹物体 → 渲新物体 → 恢复机器人前景。

用户 2026-10-13 的原话（这版就是照它写的）：

> Front 相机被夹爪遮挡时，最关键的问题很可能不是不知道新物体应该在哪里，而是不知道
> **哪些原始像素应该删除、哪些机械臂像素必须保留在新物体前面**。Pose6DAug 的关键设计
> 恰好包含这一步：先移除原物体并修复背景，再渲染新物体，最后**恢复原视频中机器人和
> 夹爪的前景区域**。

所以这里不再「画个圆叠上去」，而是严格三层（painter's algorithm，从远到近）：

    L0 底板   = 原帧，物体区域抹掉 + 背景修补        ← ball_swap.erase_cube / fill_hole
    L1 新物体 = **球网格**按标定内参 + 5 参数畸变渲上去 ← meshrender
    L2 前景   = **原帧的像素**，只在前景掩码处贴回去   ← 机械臂一根像素都不重画

L2 是整个设计的关键，也是它和「贴图」的分水岭：**新物体永远画在机械臂下面**，
机器人的像素逐位来自原视频（红线：不许重画机械臂）。

L2 的掩码两路来源，逐帧分别报数（不拍脑袋说「有遮挡」）：

  (a) **观测**：物体掩码的**凸包减掩码**（方块剪影是凸的），再卡一道「不是物体色」。
      手指啃掉方块边角时，那一块既在剪影里、又不是方块色 —— 纯观测，不碰 FK。
  (b) **几何**：FK⊗手眼（含每帧 δ 修正，`ball_swap.solve_cam_delta`）渲出的夹爪剪影，
      与新物体 footprint 相交。**只在 (a) 沉默的地方补**：新增物体可能伸到原方块
      剪影之外，那里没有观测证据，只能靠几何。

两路各自的像素数都打出来，谁在起作用一目了然。

**2026-10-14 实测：腕视图（f246–f408）里这两路都是 0px —— 这是数据的事实，不是省事。**

  * **观测路：0px，且有两个互相独立的判据都指向 0**。
    (i) 把跟踪掩码的轮廓画在原帧上放大看，红线**严丝合缝**贴着可见黄面；
    (ii) 不限离边界距离、改用颜色卡（S≥100 & H∈12–48 算「还是黄的」），rim=0 时
        剩 517/571/704px，偏黄 95%/70%/31%，而这些像素用「离边界 ≥2px」一卡
        **全没了** ⇒ 它们整体贴在物体边界 2px 以内，是**抗锯齿混色**，不是手指。
    物理上说得通：腕相机是从两个手指的**中间**看过去的，手指夹在方块的左右侧面上、
    正好落在剪影**边界**上，所以方块一个像素都没被挡住。（凸包减掩码这条路本来也
    只对「啃在剪影内部」的遮挡有效。）
  * **几何路：原始 1990~2235px，其中 97~99% 是原帧的黄色方块像素**（逐像素查过）。
    FK⊗手眼在搬运深度（~90mm）错 ~150px，δ 只修平移、修不了旋转，渲出来的夹爪剪影
    **压在了方块脸上**。照它贴回去 = 把旧方块又贴回新球上面，正撞红线「不要错增强」。
    所以 geometry 路一律先 `& ~物体`；上门之后只剩 **0~14px**，等于没用。

  ⇒ 腕视图的 L2 是**空层**，合成 = L0+L1（**这是正确的结果，不是妥协**）。
    真要看见手指横穿物体，得看 **up 视图**（俯视时手指是**横跨方块顶面**的，
    实测 18% 的帧方块被切成 ≥2 块）。几何路要真正可用，得做 Stage 2
    （拿观测到的手指剪影精修夹爪在腕相机系的位姿）。

用法：
    python workplace/calibration/pose6d_swap.py --frames 246,300,336,400 --report
    python workplace/calibration/pose6d_swap.py --frames 246,336 --out figures/pose6d_v6.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
FIG = HERE / "figures"


def _load_module(name: str, path: Path):
    """按文件路径导入（加 sys.path 会让 object_generalization/datasets 顶掉 HF 的）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load_module("_p6_arm", HERE / "arm.py")
BS = _load_module("_p6_bs", HERE / "ball_swap.py")
MR = _load_module("_p6_mr", HERE / "meshrender.py")


# ------------------------------------------------------------------ 小工具

def bgr_of(t):
    return cv2.cvtColor(np.clip(t.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def cam_from_state(g, norm_q, delta=None):
    """腕相机在 base 系的位姿（FK⊗手眼），delta 是每帧解出来的平移修正（米）。"""
    T = g["urdf"].fk(norm_q, "wrist_camera_link") @ g["X"]
    if delta is not None:
        T = T.copy()
        T[:3, 3] += np.asarray(delta, float)
    return T


# 球是**新物体**，它的材质（albedo + 镜面）是**设计选择**，不是从数据拟合出来的。
# 从数据实测出来的是**光照**（`light_fit.py`：用 Stage 2 拟合的方块位姿把网格渲回原帧，
# 逐像素取法线，再拟合 Phong；平均|ΔBGR| 52.5 → 9.7）：光方向、ambient/diffuse 配比。
#
# ★ 但**镜面反射率是物性，不是光的属性** —— 拟合里那个 specular 属于**方块**（黄塑料确实亮）。
#   一开始照抄到球上：`shade` 的镜面项是**三通道等量相加**，于是球面 27.8% 的像素
#   （`/tmp/_zo.py` 实测：max 通道被 clip 到 ≥250，且全落在三通道都 ≥200 的纯白里）
#   直接烧成白斑 —— 图上一眼就是「红圈 + 一块白」。红球是另一种材质，镜面率得单独定。
#
# 判据不是「看着顺眼」，是「镜面项还能不能把像素洗成白」：`shade` 里该项的上限就是 specular，
# 所以 specular 本身就是「高光能白到什么程度」。取 0.25 ⇒ 高光仍被球的红染色，全白像素 0%。
BALL_ALBEDO = (0.18, 0.24, 0.88)                                   # BGR：偏红（设计选择）
BALL_SPECULAR = 0.25                                               # 缎面塑料（设计选择）
BALL_SHININESS = 48.0


def load_lighting(path=None):
    """优先读 Stage 4 实测的光照；没有就退回占位值（并在 docstring/图上标明）。

    ★ 返回的 `specular/shininess` 是**拟合用的那个方块**的材质，别直接拿去渲球；
      球的材质走 `ball_material()`。
    """
    p = Path(path) if path else (HERE / "calib" / "wrist_lighting.json")
    if p.exists():
        d = json.loads(p.read_text())
        return MR.Light(direction=d["direction_base"], ambient=d["ambient"],
                        diffuse=d["diffuse"], specular=d["specular"],
                        shininess=d["shininess"])
    return MR.Light(direction=(0.35, -0.45, 0.82), ambient=0.42,
                    diffuse=0.62, specular=0.28, shininess=24.0)


def ball_material(lighting, specular=BALL_SPECULAR, shininess=BALL_SHININESS):
    """把「实测的光」套上「球的材质」：方向与 ambient/diffuse 保留，镜面项换成球的。"""
    return MR.Light(direction=lighting.dir, ambient=lighting.ambient,
                    diffuse=lighting.diffuse, specular=specular, shininess=shininess)


def place_ball(cam: MR.Cam, c_px, R_m, r_target_px):
    """由「观测像素形心 + 观测半径」反解球心的 3D 位置（度量 6D 位姿的位置部分）。

    没有深度传感器，所以深度只能从**已知半径**反解：把球心放在穿过观测像素的那条
    视线上，滑到渲染出的剪影半径等于观测半径为止。视线方向用 `undistortPoints`
    取（含畸变），滑的是距离。球是各向同性的，所以没有姿态自由度 —— 这一点如实报。
    """
    und = cv2.undistortPoints(np.asarray(c_px, float).reshape(1, 1, 2),
                              cam.K, cam.dist).reshape(2)
    dir_c = np.array([und[0], und[1], 1.0])
    dir_c /= np.linalg.norm(dir_c)
    mesh_lo = MR.uv_sphere(R_m, 12, 24)            # 搜索阶段用粗网格：要渲 ~58 次
    mesh_hi = MR.uv_sphere(R_m, 36, 72)            # 最终那一张才用细的，不然球上全是棱
    # (12,24) vs (20,40) 解出来的深度只差 0.4~0.5mm（~90mm 上 0.5%，半径 ~0.25px），
    # 单帧却从 4.1s 掉到 1.7s —— 整个夹持段 163 帧就从 ~35min 到 ~5min。

    def render_at(d, mesh):
        P_base = cam.T_base_cam[:3, :3] @ (dir_c * d) + cam.T_base_cam[:3, 3]
        return MR.rasterize(MR.transform(mesh, _T(P_base)), cam, cull=False)["mask"]

    lo, hi = 0.03, 0.6
    Ts = np.linspace(np.log(lo), np.log(hi), 40)
    best, best_err = None, 1e9
    for lt in Ts:                                  # 半径对深度单调，粗扫再细化都够
        d = float(np.exp(lt))
        r = np.sqrt(render_at(d, mesh_lo).sum() / np.pi)
        e = abs(r - r_target_px)
        if e < best_err:
            best, best_err = d, e
    for _ in range(18):                            # 黄金分割细一档
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        r1 = np.sqrt(render_at(m1, mesh_lo).sum() / np.pi)
        r2 = np.sqrt(render_at(m2, mesh_lo).sum() / np.pi)
        if abs(r1 - r_target_px) < abs(r2 - r_target_px):
            hi = m2
        else:
            lo = m1
    d = 0.5 * (lo + hi)
    P_base = cam.T_base_cam[:3, :3] @ (dir_c * d) + cam.T_base_cam[:3, 3]
    T = _T(P_base)
    ras = MR.rasterize(MR.transform(mesh_hi, T), cam, cull=False)
    return T, ras, float(d)


def _T(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def _bgr_of_mask(bgr, mask):
    return np.where(mask[..., None], bgr, 0).astype(np.uint8)


# --------------------------------------------------- L2 掩码 (a)：纯观测

def front_mask_observational(fg, obj_mask, loose_fn, rim=2):
    """物体剪影的**凸包减掩码**里、又不是物体色的那些像素 = 骑在物体前面的东西。

    纯观测，不含任何 FK。返回 (hull, front_mask, 各步像素数说明)。

    为什么用凸包：方块的剪影（正对一个面 + 两个侧面）是个凸六边形，手指从边上啃进来
    时会把它挖出缺口 —— `fill_holes` 抓不到这种（缺口通到边界），凸包能。

    ★ 两处必须收口，否则这条路出来的是**一圈脏边**而不是遮挡（2026-10-14 实测：
    照下面的写法跑，f300/336/400 报 568~1109px，画出来是贴在球右缘的一条**锯齿
    月牙**，接回去等于把旧方块那圈抗锯齿边贴回新球上）：
      1. **凸包先用「腐蚀过」的掩码算**。可见掩码边上常有几像素的小凸起（手指缝里
         透出来的那点黄、混色边），凸包一拉就把它们和主体之间连成一大片楔形，
         那片楔形**不是**遮挡，是几何假象。先 erode `rim` 像素再取凸包，
         小凸起就参与不进来了。
      2. **减掉掩码外扩 `rim` 像素的那一圈**。凸包减掩码里绝大多数本来就是**抗锯齿
         混色边**（实测中位 578px 里大头是它），它既不是物体色（S 被背景拉低）也
         不是背景，靠颜色卡不住，只能靠「离物体边界 ≥ rim 像素」这个几何条件卡。
    两处都是**保守**方向：宁可少判一点遮挡（漏增强），也不能把旧物体贴回去（错增强）。

    ★★ 加了 `rim` 之后这条路**在腕视图全帧归零**（f246/300/336/400 都是 0px）。
      这不是收口收过头，而是它本来就是零 —— 拿**另一个完全独立的判据**复核过：
      不限离边界距离、改用颜色卡（S≥100 & H∈12-48 算「还是黄的」），
      rim=0 时剩 517/571/704px，其中偏黄 95%/70%/31%，且**这些像素用 rim=2 一卡
      也全没了** —— 说明它们整体就贴在物体边界 2px 以内，是抗锯齿混色，不是手指。
      再加目视：把跟踪掩码的轮廓画在原帧上放大看，红线严丝合缝贴着可见黄面，
      腕相机是从两个手指**中间**看过去的，手指落在剪影**边界**上，方块没被挡住。
      ⇒ 腕视图 L2 为空是**结论**，不是失败。要看见手指横穿物体得看 up 视图。
    """
    u = (obj_mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1, 2 * rim + 1))
    inner = cv2.erode(u, k)
    cnts, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        z = np.zeros_like(u)
        return z, z, dict(union=int(u.sum()), hull=0, hull_minus=0, front=0)
    pts = np.vstack([c.reshape(-1, 2) for c in cnts])
    hull = cv2.fillConvexPoly(np.zeros_like(u), cv2.convexHull(pts), 1)
    hm = (hull & (1 - cv2.dilate(u, k))).astype(np.uint8)          # 见上面第 2 点
    hm = cv2.morphologyEx(hm * 255, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    # 再卡一道颜色（搬运档黄，S≥140；桌面 S 只有几十，进不来）：留下「确实不是物体」的
    objmask = (obj_mask > 0) | (loose_fn(fg) > 0)
    ok = (hm & ~objmask)
    return hull, ok, dict(union=int(u.sum()), hull=int(hull.sum()),
                          hull_minus=int(hm.sum()), front=int(ok.sum()))


# --------------------------------------------------- 原物体的可见剪影

def object_mask(rec, fg, c):
    """本帧「原物体」的**可见剪影**掩码 —— 优先用跟踪器自带的那个。

    ★ 踩过的坑（今天才发现，之前 L0 一直是坏的）：**别在搬运段现算**
      `ball_swap.yellow_mask(fg, "wrist")`。那套阈值（H16–45 / S≥68）是按**桌面帧**
      标的；搬运段的桌面受光不同、偏亮偏黄，整张桌面 30–70% 的像素都过筛，
      方块于是和桌面连成**同一个**连通域 —— `keep_nearest_comp`（按面积 ≤12000 筛）
      和 `_loose_near`（按形心距离 ≤60px 筛）双双返回空。实测 f246/f300/f336 三帧
      都是 0px，于是 L0 的擦除掩码只剩球自己那块圆盘，方块整圈黄边留在球外面
      （f336/f400 合成图上那圈黄「月牙」就是这么来的，当时误判成「暗面没擦干净」）。
    `carry_track` 用的是**搬运档**（`CARRY_S_MIN=140`），和桌面对比度够，掩码是完整的
    可见剪影：f246–f400 实测 9012–9882px、bbox ~110×110、bbox 填充率 0.79–0.84，
    163 帧全是 fresh、没有 stale。所以直接用它，只在 stale 时才退回合成方块。

    返回 (mask_uint8, src)，src ∈ {"tracker", "synth"}。
    """
    m = rec.get("mask")
    if m is not None and int((m > 0).sum()) > 250:
        return m, "tracker"
    s = float(rec.get("s") or np.sqrt(max(rec.get("area") or 0.0, 1.0)))
    z = np.zeros(fg.shape[:2], np.uint8)
    if s >= 1.0:                                   # stale：按最后看到的边长合成一块方的
        h = int(round(s / 2))
        cv2.rectangle(z, (int(round(c[0])) - h, int(round(c[1])) - h),
                      (int(round(c[0])) + h, int(round(c[1])) + h), 255, -1)
    return z, "synth"


# --------------------------------------------------- L2 掩码 (b)：FK⊗手眼几何

def front_mask_geometric(ds, i, g, cam, ball_ras, obj, links=("gripper_link",
                                                             "moving_jaw_so101_v1_link")):
    """把夹爪那两块网格按 FK⊗手眼（**含每帧 δ 修正**）渲进腕图，做深度比较。

    δ 是必须的：不加 δ 时 FK 把方块投到 (357,251)、观测在 (269,129)，差 150px
    （`meshrender --armcheck` 复现的数字）。不加 δ 时这条路是纯噪声。

    但**光有 δ 还不够**（今天实测）：δ 只修平移、修不了旋转，150px 的残差还在，
    渲出来的夹爪剪影会**压在原方块脸上** —— 实测 f336/f400 命中的 2235/1990px 里
    97–99% 是原帧的黄色方块像素。照它贴回去就是把旧方块又贴回新球上面。
    所以这里加一道硬门：**L2 恢复的永远不许是物体像素**（`& ~物体`）。被门掉的
    数量单独返回，好如实报「几何路现在还剩多少是有用的」。

    返回 (arm_ras, front_guarded, front_raw)：前两个是相机系掩码，第三个是没上门的。
    """
    fr = ds[i]
    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    ras = MR.rasterize(MR.gather(A.URDF_PATH, q, links=set(links))[0], cam, cull=True)
    bm = ball_ras["mask"]
    raw = ras["mask"] & bm & (ras["depth"] < ball_ras["depth"])
    guard = cv2.dilate((obj > 0).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    return ras, (raw & ~guard), raw


# ================================================================ 腕视图主流程

def wrist_compose(ds, i, g, trk, radius_m, cube_h_m, x_cube=None, lighting=None,
                  ball_mat=None, verbose=True):
    """一帧的腕视图换球，严格按 L0/L1/L2 三层。返回 dict（含各种数字，供报告和图）。

    ★ `x_cube` 必须**在夹取帧上算一次**再传进来，不能一帧一算：
      `estimate_x_tcp_cube` 拿的是「方块还躺在桌上时最后一次干净观测」的桌面位置
      P_last，配**当前帧**的 T_tcp 反解偏移。方块被举起来以后 T_tcp 已经抬到半空，
      再拿它去反解 P_last，出来的不是「方块相对夹爪的偏移」而是噪声 ——
      实测 f336 会解出 δ=(−237,194,−45)mm（300mm！）、f400 的 FK 误差报到 1280px。
      夹持期方块和夹爪是刚性的，所以这个偏移是**常量**，只该算一次。
    """
    fr = ds[i]
    fg = bgr_of(fr["observation.images.front"])
    rec = trk.get(i)
    if rec is None or rec.get("c") is None:
        return None

    c = np.asarray(rec["c"], float)
    s_px = (rec.get("s") or rec.get("s_px")
            or float(np.sqrt(max(rec.get("area") or 1.0, 1.0))))
    r_px = float(np.clip(s_px * radius_m / cube_h_m, 20.0, 110.0))    # 方块当尺子，与深度无关

    # 相机位姿：先按观测解每帧 δ，再渲 —— **这一步就是「把 FK 用起来」的地方**
    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    T0 = cam_from_state(g, q)
    T_base_cam = T0
    delta = np.zeros(3)
    fk_err = None
    if x_cube is not None:
        P = (g["urdf"].fk(q, "gripper_frame_link") @ np.append(x_cube, 1.0))[:3]
        uv, _ = MR.Cam(g["K"], g["dist"], T0).project(P.reshape(1, 3))
        fk_err = float(np.linalg.norm(uv[0] - c))
        delta, _ = BS.solve_cam_delta(g["K"], g["dist"], T0, [(P, c)])
        T_base_cam = cam_from_state(g, q, delta)
    cam = MR.Cam(g["K"], g["dist"], T_base_cam)

    # ---- L1 新物体：球网格按度量位姿渲上去 ----
    T_ball, ball_ras, depth_m = place_ball(cam, c, radius_m, r_px)
    if lighting is None:
        lighting = load_lighting()      # 有没有 calib/wrist_lighting.json 决定实测/占位
    # 光用了实测的那支，镜面项必须换成**球的**材质 —— 别把方块的镜面率串到球上（见常量处注释）
    ball_rgb = MR.shade(ball_ras, cam, ball_mat or ball_material(lighting), BALL_ALBEDO)

    # ---- L0 底板：抹物体 + 背景修补 ----
    # 物体掩码用跟踪器自带的（搬运档阈值，完整可见剪影）；理由见 object_mask 的 docstring
    # —— 在搬运段现算 yellow_mask 会把整张桌面吃进来，掩码恒为空、L0 只擦掉球那块圆盘。
    obj, obj_src = object_mask(rec, fg, c)
    cm = cv2.bitwise_or(obj, (ball_ras["mask"].astype(np.uint8) * 255))
    plate = BS.erase_cube(fg.copy(), cm, grow=3, shadow=True, fill="diffuse")[0]

    # ---- L2 前景掩码：两路来源 ----
    hull, front_obs, cnt = front_mask_observational(fg, obj, _carry_yellow)
    arm_ras, front_geo, front_raw = front_mask_geometric(ds, i, g, cam, ball_ras, obj)
    front = front_obs | front_geo

    # ---- 合成（远 → 近）----
    out = plate.copy()
    out[ball_ras["mask"]] = (ball_rgb * 255)[ball_ras["mask"]]   # shade 出的是 0~1，别忘乘 255
    l1 = out.copy()                              # ★ L1 到这里的快照：**还没有**贴回前景。
    out[front] = fg[front]                       # ★ 前景=原帧像素，机械臂一根都不重画

    res = dict(frame=i, c=c, r_px=r_px, s_px=s_px, depth_mm=depth_m * 1000,
               delta_mm=(delta * 1000).round(1).tolist(), fk_err_px=fk_err,
               obs=cnt, front_obs=int(front_obs.sum()), front_geo=int(front_geo.sum()),
               geo_raw=int(front_raw.sum()), obj_src=obj_src, obj_px=int((obj > 0).sum()),
               front=int(front.sum()), arm_cov=float(arm_ras["mask"].mean()),
               n_overlap=int((front_geo & ball_ras["mask"]).sum()))
    if verbose:
        print(f"  f{i}: 球心={np.round(c,1)} r={r_px:.1f}px 深度={depth_m*1000:.0f}mm"
              f" δ={np.round(delta*1000,1)}mm（FK 原差 {fk_err:.0f}px）｜"
              f"物体 {res['obj_px']}px({obj_src})｜"
              f"L2 观测 {res['front_obs']}px + 几何 {res['front_geo']}px"
              f"（几何上门前 {res['geo_raw']}px）｜夹爪覆盖 {100*res['arm_cov']:.1f}%")
    return dict(out=out, l1=l1, fg=fg, plate=plate, ball_ras=ball_ras, ball_rgb=ball_rgb,
                arm_ras=arm_ras, obj=obj, front_mask=front, **res)


def _carry_yellow(b):
    """搬运段的「这是方块黄」判据 —— 直接复用跟踪器那套阈值（`carry_cube` 用的
    `CARRY_H/CARRY_S_MIN/CARRY_V_MIN`，S≥140）。**别用 `yellow_mask("wrist")`**：
    它按桌面帧标定（S≥68），搬运段桌面整片过筛，当「不是物体色」的门会失效 ——
    这也正是原来观测路恒为 0px 的第二个原因。
    """
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    Hc, S, V = (hsv[:, :, j].astype(np.float32) for j in range(3))
    m = ((Hc >= BS.CARRY_H[0]) & (Hc <= BS.CARRY_H[1])
         & (S >= BS.CARRY_S_MIN) & (V >= BS.CARRY_V_MIN)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


# ================================================================ up 视图（遮挡真正发生的地方）

def up_track_ctx(ds, g, ig, ir):
    """up 视图夹持期的楔形追踪 + 球心路径（复用 ball_swap 那套，不改它）。

    返回 `{帧: dict(mask=…, _allc=…, ball_px=…, ball_obs=…)}`；夹取前找不到方块就返回 {}。
    """
    seed, seed_f = BS.last_up_px(ds, ig, BS.cube_mask_fns(g, "up", "yellow")[0])
    if seed is None:
        return {}, None
    ut = BS.carry_wedge_track(ds, ig, ir, seed)
    path = BS.up_carry_path(ds, ig, ir, ut)
    for k, v in path.items():
        if k in ut:
            ut[k]["ball_px"] = v["px"]
            ut[k]["ball_obs"] = v["obs"]
    return ut, (seed, seed_f)


def up_front_mask(up, c, r, rim=2, min_area=150):
    """**旧**的 up 视图 L2 判据：球盘范围内、**原帧里的亮白像素**（`V≥218 & S≤40`）。

    ⚠️ 2026-10-15 起**不再当 L2 掩码用**，只保留作**诊断对照列** —— L2 现在用的是
    `up_erase_mask` 的擦除掩码（见 `up_compose` 里 L2 段的说明），任何颜色判据都不用。
    报「盘内亮白px」是为了让人看清新旧判据差多少：这一条只抓得到手指受光的那条芯，
    每帧漏 700~1450px，漏掉的正是「球盖在夹爪上」的那部分。
    （`up_finger_mask` 的 `S<60` 是 2026-10-15 第一轮试过的替代判据，同样已降为诊断列。）

    保留原判据的说明：亮白是 `up_erase_mask` 已经在用的那一条（它靠这个把指头从擦除掩码
    里剔掉），在 up 视图里手指是画面里最亮的低饱和东西，和暖色桌面分得开（腕视图里分不开，
    见 `front_mask_observational` 的 docstring）。

    两道闸门，都是防「把桌面高光当手指贴回去」：
      · **只看球盘内**（`r+rim`）：球盘外手指本来就没被动过，不必贴。
      · **连通域 ≥ `min_area`**（默认 150px）：手指是个几千像素的长条（实测 f348
        5913px / f360 3358px），而桌面高光是几十像素的碎斑 —— f348 盘内就有 2 个
        <150px 的碎斑，会被这一刀切掉。

    返回 `(front_bool, white_bool)`；`white` 是外扩前的原始亮白掩码，用来报「盘内到底
    有多少手指像素」，两个数一起看才知道是闸门拦掉的还是本来就没有。
    """
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    V, S = (hsv[:, :, j].astype(np.float32) for j in (2, 1))
    m = ((V >= 218) & (S <= 40)).astype(np.uint8)
    disk = np.zeros(m.shape, np.uint8)
    cv2.circle(disk, (int(round(c[0])), int(round(c[1]))), int(round(r)) + rim, 1, -1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = [j for j in range(1, n)
            if st[j, cv2.CC_STAT_AREA] >= min_area and (lab[disk > 0] == j).any()]
    if not keep:
        return np.zeros(m.shape, bool), (m > 0)
    fm = np.isin(lab, keep) & (disk > 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1,) * 2)
    return cv2.dilate(fm.astype(np.uint8), k) > 0, (m > 0)


UP_FINGER_S_MAX = 60.0


def up_finger_mask(up, c, r, rim=2, s_max=UP_FINGER_S_MAX, grow=1):
    """⚠️ **诊断列，不参与合成**（2026-10-15 第二轮）：球盘内的**低饱和像素**
    = 手指（白塑料）+ 被机械臂阴影压暗的桌面。

    曾经拿它当 L2 的贴回判据（「低饱和的都不许被球盖」），量出来确实把手指抓全了，但
    **桌面阴影也是低饱和**（S≈23）：f246 盘 1298px 里贴回 723px，其中只有 ~222px 是手指，
    其余是把**球后面的桌面**糊到球上 —— 球被啃出一块灰缺口，换了个方向难看。
    另外它太贴着色：物体在 up 里被暖光洗成米白（S≈68~72），离 `s_max=60` 只有 8 个单位，
    光照一变就可能把球上啃出洞。

    所以最终没用颜色判据，改用**旧物体自己在哪儿露出来**（`up_erase_mask` 的 `cm`）——
    和「夹爪在哪挡着」是同一件事，见 `up_compose` 的 L2 段。这个函数留着只为了在
    报告里对照「低饱和判据会贴回多少 px」，说明这条路量过、放弃的原因。

    顺带记下第一条死路：亮白 `V≥218 & S≤40`（`up_front_mask`）只认手指**受光的那条芯**，
    球盘里每帧漏 700~1450px —— 那正是用户看到的「球盖在夹爪上」。
    """
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    disk = np.zeros(S.shape, np.uint8)
    cv2.circle(disk, (int(round(c[0])), int(round(c[1]))), int(round(r)) + rim, 1, -1)
    fm = (S < s_max) & (disk > 0)
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        fm = cv2.dilate(fm.astype(np.uint8), k) > 0
    return fm & (disk > 0)


def up_local_cam(g, c_px, r_up_px, h_mm, radius_m, cam_h_mm=None):
    """以**观测为锚**造一个局部针孔，好把球**网格**渲进 up 视图。

    为什么不直接用真 up 相机：单张平面单应只能把内参定到「焦距 ↔ 高度」的退化族里
    （`/tmp/_zn.py` 实测：扫焦距找正交性，最好的 fx≈1600 处两列范数仍差 3%、
    fx=409 更把相机解到桌面**以下** 240mm）。所以这里不假装知道真 up 相机，而是造一个
    **自洽的局部相机**：球心正好落在光轴上、主点就取观测形心 c，焦距由
    「渲出的剪影半径 = 观测半径 r_up」反解 ⇒ 它的**剪影与观测逐位一致**，
    球面上的法线图也自洽。

    代价写在明处（也要标在图上）：真实 up 光轴是**斜**的（`H_board_to_px` 在像心的
    雅可比两主方向比 1.10 ⇒ ~20° 量级），本相机假定**垂直下视** ⇒ 球的**法线图**
    会差同量级的一个旋转，**着色**跟着差一点。剪影不受影响（那是锚死的）。

    几何：球心在桌面以上 h，相机名义高 `cam_h_mm`（默认 `BS.UP_CAM_H_MM`）⇒ 球心距
    d = cam_h − h。半径 R 的球在光轴上：剪影半角 θ 满足 sinθ = R/d，
    像半径 = f·tanθ = f·R/√(d²−R²) ⇒ `f = r_up·√(d²−R²)/R`。
    把它展开回一阶就是 `f ≈ spm·cam_h`（`spm` 是桌面处 px/m）—— 也就是说，
    这个局部相机**正是 `BS.UP_CAM_H_MM` 与 `spm` 早就隐含的那台相机**，不是临时凑的。

    返回 `(Cam, 球心 base 坐标, f, d_mm)`。
    """
    cam_h = float(cam_h_mm if cam_h_mm is not None else BS.UP_CAM_H_MM)
    d = max(cam_h - float(h_mm), 0.15 * cam_h)          # 别让球心贴到相机上
    R_mm = float(radius_m) * 1000.0
    f = float(r_up_px) * float(np.sqrt(max(d * d - R_mm * R_mm, 1e-6))) / R_mm
    # 球心的水平位置：观测像素打到**桌面**上的那个点（位置由观测锚定，高度用 FK 的 h）
    qq = g["Hi"] @ np.array([c_px[0], c_px[1], 1.0])
    P_bb = np.asarray(g["T_base_board"], float) @ np.array(
        [qq[0] / qq[2], qq[1] / qq[2], 0.0, 1.0])
    x0, y0 = float(P_bb[0]), float(P_bb[1])
    # 相机系：光轴 = base −z（垂直下视），x_cam = x_base，y_cam = −y_base（保右手系）
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    T[:3, 3] = [x0, y0, cam_h / 1000.0]
    K = np.array([[f, 0, float(c_px[0])], [0, f, float(c_px[1])], [0, 0, 1.0]])
    return (MR.Cam(K, np.zeros(5), T),
            np.array([x0, y0, float(h_mm) / 1000.0]), f, d)


def up_compose(ds, i, g, ut, x_cube, radius_m, cube_h_m, lighting=None, verbose=True):
    """up 视图一帧的三层合成 —— 用户那句话在**遮挡真的发生**的视角上的实现。

    和腕视图同一套 L0/L1/L2，两处按 up 的实际差异调整：

    * **L1 是球网格光栅**（2026-10-14 从解析圆换过来的）。up 相机没有可用的投影矩阵
      （单平面退化，见 `up_local_cam` 的 docstring），所以球走**观测锚定的局部针孔**：
      球心在光轴上、主点取观测形心、焦距由「剪影半径 = r_up」反解 ⇒ 剪影与观测一致。
      法线图假定垂直下视，真 up 光轴斜 ~20°，着色会差这一点 —— 图上标明了。
    * **L2 = 「球只画在旧物体露出来的地方」**（2026-10-15 定稿，见 L2 段的长注释）：
      手指横跨方块顶面（f348 实测 649px、f360 975px 落在球盘里），这些都是**手指在挡
      方块**，所以球在那儿的正确行为是**被挡住**。实现上不靠颜色判据，而是用擦除掩码
      本身把球的可见范围限住 —— 擦除掩码已经剔掉受光的手指、也抓不到手指的暗面，
      于是「球在哪能画」与「夹爪在哪挡着」是同一件事的两种说法，光照变了也不会失配。

    红线照旧：改动只允许发生在「擦除掩码 ∪ 球盘」里，报 `outside_px`（现在因为
    `out[~sel] = up[~sel]` 而**恒为 0**，是构造性保证；`changed` 也因此恰好 = 球的可见像素）。
    """
    fr = ds[i]
    up = BS.up_bgr(ds, i)
    uw = ut.get(i) or {}
    if uw.get("mask") is None:
        return None

    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    P = (g["urdf"].fk(q, BS.TCP_LINK) @ np.append(x_cube, 1.0))[:3]     # 方块几何中心

    cm = BS.up_erase_mask(up, uw["mask"], uw.get("_allc"))
    c, src = uw.get("ball_px"), "path"
    if c is None:
        ys, xs = np.nonzero(cm)
        if not len(xs):
            return None
        c, src = (float(xs.mean()), float(ys.mean())), "frame"
    c = np.asarray(c, float)

    qq = g["Hi"] @ np.array([c[0], c[1], 1.0])
    spm, _ = BS.up_scale_px_per_m(g, qq[:2] / qq[2])
    h_mm = max(0.0, float(P[2]) * 1000.0)
    kk = BS.UP_CAM_H_MM / max(BS.UP_CAM_H_MM - h_mm, 0.25 * BS.UP_CAM_H_MM)
    r_up = float(radius_m * spm * kk)
    nsub, _, sst, _ = cv2.connectedComponentsWithStats((uw["mask"] > 0).astype(np.uint8), 8)
    ncomp = int(sum(1 for j in range(1, nsub) if sst[j, 4] >= 40))

    # ---- 诊断列（不再参与合成，见 L2 段的说明）：球盘内的低饱和像素 = 手指 + 被
    #      阴影压暗的桌面。`front` 是 2026-10-15 试过又弃用的判据，`white` 是更旧的一条。
    front = up_finger_mask(up, c, r_up)
    _, white = up_front_mask(up, c, r_up)

    # ---- L0 底板：抹掉旧物体 + 补背景。**只抹物体本身，不抹球盘** —— 2026-10-15 改。
    #      以前把整个球盘也并进擦除掩码，是因为球要铺满整盘；现在球只铺「物体露出来的
    #      地方」，盘内其余像素要**逐位回原帧**（见 L2），擦了再贴回来等于没擦，还白花
    #      补洞的功夫 —— 更要紧的是补洞会把盘内那片真桌面换成环上中位色，成一块糊斑。
    plate = BS.erase_cube(up, cm, grow=0, shadow=False, fill="diffuse")[0]

    # ---- L1 新物体：球**网格**，从「观测锚定的局部针孔」渲进去（理由见 docstring）
    cam_up, P_ball, f_up, d_mm = up_local_cam(g, c, r_up, h_mm, radius_m)
    if lighting is None:
        lighting = load_lighting()
    ball_ras = MR.rasterize(MR.transform(MR.uv_sphere(radius_m, 36, 72), _T(P_ball)),
                            cam_up, cull=False)
    ball_rgb = MR.shade(ball_ras, cam_up, ball_material(lighting), BALL_ALBEDO)
    bm = ball_ras["mask"]                      # 球剪影（圆盘，半径 ≈ r_up）
    ball_bgr = (ball_rgb * 255)

    # ---- L0/L1/L2 的最终形式（2026-10-15 定稿）
    #
    # 用户原话：_「上边一定是夹爪在挡住小球，而不是小球挡住夹爪」_。物理事实是：方块被
    # 夹爪挡住的那一大块，球**同样**该被挡住。所以「球能画在哪」不该由颜色判据定（手指的
    # 暗面、桌面的阴影都是低饱和，颜色分不开），而该由「**旧物体在哪儿露出来**」定 ——
    # 这件事我们本来就知道，就是擦除掩码 `cm`：
    #
    #       · `cm` 已剔掉受光的手指（`up_erase_mask` 内部 `& ~dilate(亮白,3)`）⇒ 手指那侧
    #         球自动停手；手指的**暗面**本来也进不了 `cm`（S≈25 < 45 的松门限）；
    #       · `cm` 自带 5×5 椭圆外扩（±2px），正好当「球 r=12mm 比方块半宽 10.75mm 宽出来的
    #         那 1.25mm ≈ 2.4px」，于是球的剪影沿物体轮廓自然外沿一点点，不会啃进手指。
    #
    # 合成因此收敛成一句可检验的话：**改动只发生在 `cm` 里**（擦除掩码既是「旧物体在哪」
    # 也是「允许改哪」），于是
    #       · `bm ∩ cm`      → 球（旧物体露出来、且球剪影盖得到的地方）
    #       · `cm − bm`      → 补好的背景（方块露出来、球盖不到的那部分：方块比球大，
    #                          它的角伸在球剪影外 —— 按 L0 的语义就该是补好的背景，而不是
    #                          留一圈旧黄）
    #       · `~cm`          → **原帧逐位**（夹爪、桌面、红盒、盘外……一律不动）
    # `outside_px` 因此是构造性 0，`changed` 恰好 = 球可见像素 ∪ 补背景的像素。
    #
    # 走过又弃掉的两条路（都量过，记在这里免得再试）：
    #       A) 按低饱和（`S<60`）在盘内贴回 → 桌面阴影也是低饱和，球上被啃出一块灰缺口
    #          （f246：盘 1298px 里贴回 723px，其中只有 ~222px 是手指）；
    #       B) 按亮白（`V≥218 & S≤40`）在盘内贴回 → 只认手指受光那条芯，每帧漏 700~1450px，
    #          就是用户看到的「球盖在夹爪上」。
    sel = bm & (cm > 0)
    out = plate.copy()
    out[bm] = ball_bgr[bm]             # L1：先整盘画上（= 不做遮挡的错版本，留作对照）
    l1 = out.copy()                    # ★ L1 快照
    out[cm == 0] = up[cm == 0]         # ★ L2：擦除掩码外逐位回原帧 ⇒ 夹爪永远在球前面

    # 两块自检：网格剪影半径 vs 观测半径（应该几乎相等）；球上有没有被镜面项洗白的像素
    r_mesh = float(np.sqrt(int(ball_ras["mask"].sum()) / np.pi))
    bp = ball_rgb[ball_ras["mask"]] * 255.0
    wash = float((bp.min(1) >= 200).mean()) if len(bp) else 0.0

    # 红线：改动只允许在擦除掩码 ∪ 它外扩 3px 里。`out[cm==0] = up[cm==0]` 已经把
    # 这条变成了构造性等式，这里量出来是**回归检查**（防止以后有人加了别的写法又把
    # 改动漏到掩码外）。
    allowed = cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    yy, xx = np.mgrid[0:up.shape[0], 0:up.shape[1]]
    dd = np.hypot(xx - c[0], yy - c[1])
    d = np.abs(out.astype(np.int16) - up.astype(np.int16)).sum(2)
    changed = d > 0
    # L0 漏没漏：**整个擦除掩码**里还剩多少**真方块黄**（严门限，跟腕视图 sweep 的
    # `left_cube` 同一把尺）。按 `cm` 量而不是按球盘量 —— 楔形那部分也是方块，只扫
    # 球盘会把楔形的漏擦漏掉。要 ~0：`cm − bm` 那块现在是补好的背景，不该再有黄。
    hsv_o = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    Ho, So, Vo = (hsv_o[:, :, j].astype(np.float32) for j in range(3))
    strict = (Ho >= 12) & (Ho <= 48) & (So >= 150) & (Vo >= 110)
    left_cube = int((strict & (cm > 0)).sum())

    # ★★ 用户报的那个毛病的**直接量化**（2026-10-15 加）：球的剪影里还剩多少像素
    #    **盖住了「本来不是物体」的东西**。判据取原帧自己（松暖色 = 物体可见部分，和
    #    `up_erase_mask` 同一把尺）：原帧那一位是物体 ⇒ 球盖上去是对的；不是物体 ⇒
    #    球在挡夹爪/背景，就是错的那一类。
    #    换到 L2 = `bm & cm` 之后这一列的含义变成**球越出旧物体轮廓的量**：应当只剩
    #    「球半径 12mm 比方块半宽 10.75mm 大出来的那一圈」（≈2px 环，`cm` 自带的外扩），
    #    量级从旧判据的 700~1450px/帧 掉到百 px 级以下就算过关。
    hsv_u = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    Hu, Su, Vu = (hsv_u[:, :, j].astype(np.float32) for j in range(3))
    obj_obs = (Hu >= 8) & (Hu <= 48) & (Su >= 45) & (Vu >= 55)
    cover_ng = int((bm & changed & ~obj_obs).sum())
    cover_obj = int((bm & changed & obj_obs).sum())

    res = dict(frame=i, c=c, r_up=r_up, r_mesh=r_mesh, f_up=f_up, d_mm=d_mm,
               left_cube=left_cube, cover_ng=cover_ng, cover_obj=cover_obj,
               ball_px=int(bm.sum()), sel_px=int(sel.sum()),
               occl_px=int(bm.sum()) - int(sel.sum()),
               # 补出来的背景露在外面的 px：方块（对角 30.4mm）比球（直径 24mm）**大**，
               # 它伸在球剪影外的部分（角、被手指隔开的另一条带）按 L0 的语义就是「擦掉
               # 旧物体后补好的背景」。那一片像素是**补的，不是真的** —— 这是本方案唯一
               # 一处「编内容」，必须报出来、不能藏（量 `erase_px − ball_px`，只算正的那部分）。
               patch_px=max(0, int((cm > 0).sum()) - int(bm.sum())),
               wash_frac=wash, h_mm=h_mm, k=kk, src=src, ncomp=ncomp,
               obj_px=int((uw["mask"] > 0).sum()), erase_px=int((cm > 0).sum()),
               white_px=int((white & ((dd <= r_up + 2))).sum()),
               front=int(front.sum()),
               on_ball=int((front & (dd <= r_up)).sum()),
               outside_px=int((changed & ~allowed).sum()),
               maxdist_px=float(dd[changed].max()) if changed.any() else 0.0)
    if verbose:
        print(f"  f{i}: 球心={np.round(c,1)}({src}) r={r_up:.1f}px h={h_mm:.0f}mm"
              f"｜方块 {res['obj_px']}px 切成 {ncomp} 块｜擦除 {res['erase_px']}px｜"
              f"盘内残留方块黄 {left_cube}px｜"
              f"红线外改动 {res['outside_px']}px，最远 {res['maxdist_px']:.0f}px")
        print(f"        球剪影 {res['ball_px']}px：**露出 {res['sel_px']}px**，"
              f"被原帧挡回 {res['occl_px']}px（夹爪/背景在前）｜"
              f"越出旧物体轮廓 {cover_ng}px（= 球比方块宽出来的那一圈）｜"
              f"补背景露出 {res['patch_px']}px｜"
              f"网格 r={r_mesh:.1f}px（观测 {r_up:.1f}，差 {r_mesh - r_up:+.2f}px）"
              f" f={f_up:.0f}px d={d_mm:.0f}mm｜球上被洗白 {100 * wash:.1f}%")
    return dict(out=out, l1=l1, up=up, plate=plate, front_mask=front, white=white,
                erase=cm, ball_ras=ball_ras, ball_rgb=ball_rgb, **res)


def figure_up(rows, out_path):
    """up 视图 6 联图。第 3 栏是 L1（整盘画上，**没做遮挡**），第 5 栏是合成；
    3↔5 并排就是「夹爪把球挡回去多少」的肉眼核对。"""
    panels = []
    for r in rows:
        if r is None:
            continue
        ov = r["l1"].copy()               # ★ 用 L1 的快照，不是合成图（否则第 3 栏与第 5 栏重复）
        cv2.drawContours(ov, cv2.findContours(r["ball_ras"]["mask"].astype(np.uint8),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                         (0, 255, 255), 1)          # 画**网格**剪影：和观测圆重合（差 <0.05px）
        d = r["out"].astype(np.int16) - r["up"].astype(np.int16)
        dv = cv2.applyColorMap(np.clip(np.abs(d).sum(2) * 2, 0, 255).astype(np.uint8),
                               cv2.COLORMAP_INFERNO)
        # L2 的判据图：擦除掩码(绿) vs 球剪影(黄) —— 球只能画在两者交里，一眼可核
        mk = r["out"].copy()
        for msk, col in ((r["erase"], (0, 255, 0)),
                         (r["ball_ras"]["mask"], (0, 255, 255))):
            cv2.drawContours(mk, cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)[0],
                             -1, col, 1)
        row = [MR.put_lines(r["up"], [f"original up f{r['frame']}  c=({r['c'][0]:.0f},"
                                      f"{r['c'][1]:.0f})"], org=(8, 22), scale=0.5),
               MR.put_lines(r["plate"], [f"L0  erase cube + ball disk ({r['erase_px']}px)"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(ov, [f"L1 ball MESH r={r['r_mesh']:.0f}px (obs {r['r_up']:.0f})"
                                 f" f={r['f_up']:.0f} d={r['d_mm']:.0f}mm",
                                 "    local pinhole anchored to observation; up axis ~20deg off"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(mk, [f"L2 mask: erase mask (green) x ball silhouette (yellow)",
                                 f"    ball visible {r['sel_px']}px, kept back "
                                 f"{r['occl_px']}px (gripper in front)"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(r["out"], [f"composite   over old-object contour "
                                       f"{r['cover_ng']}px (= the +2px rim)",
                                       f"    changed px stay inside the erase mask"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(dv, ["|diff| vs original"], org=(8, 22), scale=0.5)]
        panels.append(np.hstack(row))
    img = np.vstack(panels)
    cv2.imwrite(str(out_path), img)
    print("→", out_path.relative_to(REPO))
    return img


# ================================================================ 出图

def figure(frames, rows, out_path):
    """6 联图（论文体例：一层一栏，肉眼能核对每层对不对）。"""
    panels = []
    for r in rows:
        if r is None:
            continue
        m = r["ball_ras"]["mask"]
        ov = r["l1"].copy()               # ★ 用 L1 的快照，不是合成图（否则第 3 栏与第 5 栏重复）
        cv2.drawContours(ov, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                         (0, 255, 255), 1)
        d = r["out"].astype(np.int16) - r["fg"].astype(np.int16)
        dv = np.clip(np.abs(d).sum(2) * 2, 0, 255).astype(np.uint8)
        dv = cv2.applyColorMap(dv, cv2.COLORMAP_INFERNO)
        # 文字全用 ASCII：cv2 的 Hershey 字体不认中文，写中文出来是一排方块（踩过）
        row = [MR.put_lines(r["fg"], [f"original f{r['frame']}  obs c=({r['c'][0]:.0f},"
                                      f"{r['c'][1]:.0f})"], org=(8, 22), scale=0.5),
               MR.put_lines(r["plate"], ["L0  erase object + inpaint bg"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(ov, [f"L1  ball mesh  r={r['r_px']:.0f}px  d={r['depth_mm']:.0f}mm"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(MR.overlay(r["out"], r["front_mask"], (0, 255, 0), alpha=0.0),
                            [f"L2  robot fg {r['front']}px (obs {r['front_obs']} / "
                             f"geom {r['front_geo']} of raw {r['geo_raw']})"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(r["out"], [f"composite  dFK={r['fk_err_px']:.0f}px -> "
                                       f"|delta|={np.linalg.norm(r['delta_mm']):.0f}mm"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(dv, ["|diff| vs original"], org=(8, 22), scale=0.5)]
        panels.append(np.hstack(row))
    img = np.vstack(panels)
    cv2.imwrite(str(out_path), img)
    print("→", out_path.relative_to(REPO))
    return img


def sweep(ds, g, trk, x_cube, frames, radius_m, cube_h_m, near_px=95.0, stride=1):
    """把整段夹持期逐帧跑一遍，报**段级**数字（4 帧的样例图说明不了「这段稳不稳」）。

    三个必报的量：
      · `obj_px`     物体掩码大小（应该跟着方块大小走，突然掉就是掩码塌了）
      · `left_*`     擦完之后**还留在擦除区里的偏黄像素** = L0 漏没漏（球+方块那个邻域）
      · `outside_px` 离球心 `near_px` 以外被动过的像素数 = **红线检查**，
                     机械臂/蓝块/红盒/桌面一个像素都不许变，这个数必须是 0

    ★ 「漏擦」这两列必须**成对**看，单看松门限会误报（2026-10-14 实测查过）：
      松门限 `left_obj`（S≥100）在夹持段中位 410px，看着像方块没擦干净 —— 其实不是。
      拿 f246/300/336/400 逐像素比过：这些像素的 HSV 中位 **S≈114~138**，而**真方块中心
      S≈186~189**；位置全落在方块 footprint 的**四个角**（方块 110×110、球盘 r≈55 内切，
      角上本来就盖不住）。它们是被补出来的**暖色桌面**（夹持段桌面本来就偏黄，邻近环带
      的「偏黄」基线就有 8%），跟方块不是一个东西。所以另报一个**严门限** `left_cube`
      （S≥150，真方块的饱和度量级）：这一列才是「有没有留下方块」。

    `stride>1` 用来抽帧（这段单帧成本在 `place_ball` 的 58 次渲染上，很贵）。
    """
    rows = []
    for n, i in enumerate(frames[::stride]):
        r = wrist_compose(ds, i, g, trk, radius_m, cube_h_m, x_cube=x_cube, verbose=False)
        if r is None:
            continue
        fg, out = r["fg"], r["out"]
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        Hc, S, V = (hsv[:, :, j].astype(np.float32) for j in range(3))
        softy = (Hc >= 12) & (Hc <= 48) & (S >= 100) & (V >= 110)
        strict = (Hc >= 12) & (Hc <= 48) & (S >= 150) & (V >= 110)     # 真方块的饱和度量级
        stuck = softy & ((r["obj"] > 0) | (r["ball_ras"]["mask"] > 0))
        c = np.asarray(r["c"], float)
        yy, xx = np.mgrid[0:fg.shape[0], 0:fg.shape[1]]
        far = np.hypot(xx - c[0], yy - c[1]) > near_px
        d = np.abs(out.astype(np.int16) - fg.astype(np.int16)).sum(2)
        rows.append(dict(frame=i, obj_px=int((r["obj"] > 0).sum()),
                         left_px=int(stuck.sum()),
                         left_obj=int((softy & (r["obj"] > 0)).sum()),   # 松（含暖桌面）
                         left_cube=int((strict & (r["obj"] > 0)).sum()),  # 严：留下的方块
                         left_ball=int((softy & (r["ball_ras"]["mask"] > 0)).sum()),
                         l2_px=r["front"],
                         geo_raw=r["geo_raw"], geo_px=r["front_geo"],
                         outside_px=int(((d > 0) & far).sum())))
        if (n + 1) % 20 == 0:
            print(f"    {n+1}/{len(frames[::stride])} ...", flush=True)
    return rows


def report_sweep(rows):
    if not rows:
        print("[段级] 没有可用帧")
        return
    print(f"\n[段级] {len(rows)} 帧（f{rows[0]['frame']}~f{rows[-1]['frame']}）")
    for key, name in (("obj_px", "物体掩码px"), ("left_obj", "偏黄(松S100)"),
                      ("left_cube", "漏擦方块(严S150)"),
                      ("left_ball", "球盘内偏黄px"), ("l2_px", "L2 总px"),
                      ("geo_raw", "几何原始px"), ("geo_px", "几何上门后px"),
                      ("outside_px", "远处改动px")):
        v = np.array([r[key] for r in rows], float)
        print(f"  {name:<12} min {v.min():8.0f}  中位 {np.median(v):8.0f}  max {v.max():8.0f}")
    top = [r for r in sorted(rows, key=lambda x: -x["l2_px"])[:5] if r["l2_px"] > 0]
    print("  L2 最多的 5 帧：" + ("  ".join(f"f{r['frame']}={r['l2_px']}px" for r in top)
                                 if top else "全程为 0"))
    bad = [r["frame"] for r in rows if r["outside_px"] > 0]
    print(f"  红线：远处被动过的帧 {len(bad)}/{len(rows)}"
          + (f"  {bad[:8]}{'...' if len(bad) > 8 else ''}" if bad else "  ✔"))


def report_up(rows):
    """up 视图的段级数字。核心两列是 `sel_px` / `occl_px` —— 球露出来多少、被原帧挡回
    多少。夹爪横跨方块顶面的那些帧（`ncomp≥2`）就是 `occl_px` 大的帧，这就是「遮挡发生
    在 up 视图」的量化证据。"""
    rows = [r for r in rows if r is not None]
    if not rows:
        print("[up 段级] 没有可用帧")
        return
    print(f"\n[up 段级] {len(rows)} 帧（f{rows[0]['frame']}~f{rows[-1]['frame']}）")
    for key, name in (("obj_px", "楔形掩码px"), ("erase_px", "擦除px"),
                      ("left_cube", "擦除区残留方块黄px"),
                      ("ball_px", "球剪影px"), ("sel_px", "球露出px"),
                      ("occl_px", "被原帧挡回px"),
                      ("cover_ng", "越出旧轮廓px"),
                      ("patch_px", "补背景露出px"),
                      ("front", "盘内低饱和px"), ("white_px", "旧亮白判据px")):
        v = np.array([r[key] for r in rows], float)
        print(f"  {name:<12} min {v.min():8.0f}  中位 {np.median(v):8.0f}  max {v.max():8.0f}")
    dr = np.array([r["r_mesh"] - r["r_up"] for r in rows], float)
    wv = np.array([r["wash_frac"] for r in rows], float)
    print(f"  L1 网格剪影半径 − 观测半径：中位 {np.median(dr):+.2f}px，"
          f"max|·| {np.abs(dr).max():.2f}px（贴合观测；锚定法的自检）")
    print(f"  球上被镜面项洗白：中位 {100 * np.median(wv):.1f}%，max {100 * wv.max():.1f}%"
          "（0 附近才对 —— 方块的镜面率串到球上会飙到几十 %）")
    # 手指横跨方块（ncomp≥2）的帧，球必须**大部分被挡回**；这是本改动的正面证据
    spl = [r for r in rows if r["ncomp"] >= 2]
    if spl:
        oc = np.array([r["occl_px"] / max(r["ball_px"], 1) for r in spl], float)
        print(f"  方块被切成 ≥2 块的帧 {len(spl)}/{len(rows)}：球被挡回的比例"
              f" 中位 {100 * np.median(oc):.0f}%，max {100 * oc.max():.0f}%"
              "（夹爪横跨 ⇒ 球就该被挡掉这么多）")
    # ★ 用户报的毛病：球盖在夹爪上。现在这一列只剩「球比方块宽出来的那一圈」
    cov = np.array([r["cover_ng"] for r in rows], float)
    print(f"  **球越出旧物体轮廓**：min {cov.min():.0f}  中位 {np.median(cov):.0f}"
          f"  max {cov.max():.0f}px（= 球 r=12mm vs 方块半宽 10.75mm 的 ±2px 环；"
          "旧判据这里是 700~1450px）")
    # 唯一一处「编内容」：方块（对角 30.4mm）比球（直径 24mm）大，它伸在球剪影外的部分
    # （角、被手指隔开的另一条带）只能靠 L0 的补背景顶上 —— 那一片是补的，不是真的。
    pv = np.array([r["patch_px"] for r in rows], float)
    print(f"  补背景露在外面：{int((pv > 0).sum())}/{len(rows)} 帧非零，中位 {np.median(pv):.0f}"
          f"  max {pv.max():.0f}px（球把整个擦除区盖住时为 0；非零的那些帧是一片补出来的背景）")
    bad = [r["frame"] for r in rows if r["outside_px"] > 0]
    print(f"  红线：掩码外被动过的帧 {len(bad)}/{len(rows)}"
          + (f"  {bad[:8]}" if bad else "  ✔")
          + f"；改动最远 {max(r['maxdist_px'] for r in rows):.0f}px"
          "（球心到擦除掩码远端的距离，方块 footprint 本身就有 ~78px）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--frames", default="246,300,336,400")
    ap.add_argument("--radius-mm", type=float, default=12.0)
    ap.add_argument("--cube-h-mm", type=float, default=21.5)
    ap.add_argument("--grasp", type=int, default=246)
    ap.add_argument("--release", type=int, default=409)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="整段夹持期逐帧跑，报段级数字 + 红线检查（没有 --frames 时用它）")
    ap.add_argument("--stride", type=int, default=1, help="sweep 抽帧步长（单帧很贵）")
    ap.add_argument("--up", action="store_true",
                    help="跑 up 视图的三层合成（遮挡真正发生的那个视角）")
    ap.add_argument("--sweep-up", action="store_true",
                    help="up 视图整段逐帧跑，报段级数字 + 红线检查")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    FIG.mkdir(parents=True, exist_ok=True)
    g = BS.load_geom()
    ds = LeRobotDataset(a.dataset, root=str(REPO / "dataset" / a.dataset))

    # 方块相对夹爪的常量偏移：**只在夹取帧算一次**（理由见 wrist_compose 的 docstring）
    X_cube, note = BS.estimate_x_tcp_cube(
        ds, a.grasp, g, BS.UP_CAM_H_MM / 1000.0, a.cube_h_mm / 1000.0,
        BS.cube_mask_fns(g, "up", "yellow")[0])
    print(f"[常量] X_tcp_cube = {np.round(np.asarray(X_cube)*1000,2)}mm  ({note})")
    R_m, H_m = a.radius_mm / 1000.0, a.cube_h_mm / 1000.0
    frames = [int(x) for x in a.frames.split(",") if x.strip()]
    p_out = None
    if a.out:
        p_out = Path(a.out)
        if not p_out.is_absolute():
            p_out = HERE / p_out          # `--out figures/x.png` ⇒ workplace/calibration/figures/
        p_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- up 视图分支（遮挡真正发生的地方；用户那句「恢复机器人前景」在这里才非空）----
    if a.up or a.sweep_up:
        ut, seed = up_track_ctx(ds, g, a.grasp, a.release)
        if not ut:
            print("[up] 夹取前没在 up 图里找到方块 → 退出")
            return 1
        hit = [v for v in ut.values() if v.get("ball_px") is not None]
        print(f"[up 跟踪] 种子 f{seed[1]} px={np.round(seed[0],1)}；"
              f"夹持段 {len(ut)} 帧，其中 {len(hit)} 帧有球心路径")
        seg = sorted(ut)
        if a.sweep_up:
            print(f"[up 段级] 逐帧跑 {len(seg)} 帧（stride={a.stride}）")
            report_up([up_compose(ds, i, g, ut, X_cube, R_m, H_m, verbose=False)
                       for i in seg[::a.stride]])
            return 0
        rows = []
        for i in frames:
            print(f"[up 帧] f{i}")
            r = up_compose(ds, i, g, ut, X_cube, R_m, H_m)
            if r is None:
                print("  跳过（up 图里没跟到方块）")
            rows.append(r)
        if p_out:
            figure_up(rows, p_out)
        return 0

    trk = BS.carry_track(ds, a.grasp, a.release)
    print(f"[跟踪] 夹持段 f{a.grasp}~f{a.release} 共 {len(trk)} 帧有腕图方块观测")

    if a.sweep:
        seg = sorted(trk)
        print(f"[段级] 逐帧跑 {len(seg)} 帧（stride={a.stride}），单帧成本在球网格的深度反解上，慢")
        report_sweep(sweep(ds, g, trk, X_cube, seg, a.radius_mm / 1000.0,
                           a.cube_h_mm / 1000.0, stride=a.stride))
        return 0

    rows = []
    for i in frames:
        print(f"[帧] f{i}")
        r = wrist_compose(ds, i, g, trk, R_m, H_m, x_cube=X_cube)
        if r is None:
            print("  跳过（腕图里没跟到方块）")
        rows.append(r)
    if p_out:
        figure(frames, rows, p_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

