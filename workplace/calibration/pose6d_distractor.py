#!/usr/bin/env python
"""干扰物（桌上那个**没人动过的蓝块**）：把重建网格替换掉它，**up + 腕 双视角**一张图。

和已经做好的那两支的关系（**那两支一个字都没动**）：

    · `pose6d_wrist_fit.py` / `pose6d_up_fit.py` 换的是**被夹起来的那个物体**
      （黄块 → 注入物）。它的位姿由夹爪给（FK），形状未知 ⇒ 剪影拟合；
      L2 是 `out[cm==0] = fg[cm==0]`（夹爪**永远**在注入物前面）。
    · 这一支换的是**桌上静止的干扰物**（蓝块 → 注入物）。没有夹取、没有夹持姿态，
      而且它的**位置是量出来的、尺度是 CAD 给的** —— 于是**不需要拟合**（见下）。

★★ 为什么这条路**不做剪影拟合** —— 位置和尺度都已知，只有朝向不可观测：

    位置：蓝块静止在桌上，up 单应 `g["Hi"] ⊗ g["T_base_board"]` 直接给出桌面点。
          ep0 每 10 帧量一次，44 次的**中位**就是 (239.8, 117.4, 0)mm，
          **最大偏离 1.17mm** —— 这是量出来的，不是拟合出来的。
    尺度：`sts3215_03a_v1` 有 CAD 真值，`fix_s=1`。
    朝向：**蓝块携带不了舵机朝向的任何信息**。所以它是一个**设计选择**
          （最大面朝下静置 + 指定的偏航角），不是估计量 —— 这一点写在图注里。

    于是整条路是**前向渲染**：一个 base 系的物点 + 一个 CAD 尺度 + 一个选定朝向
    → 两台相机（up 的、腕的）各自的投影。两个视角画的是**同一个** 3D 摆放，
    这才是「双视角」而不是「两次各自拟合」。

★★ 为什么**不能**照抄抓取那支的「面积锚定深度」：
    `fit_wrist` 的初始深度是 `ray_depth_by_area` —— 沿视线滑到「渲出面积 = 观测面积」。
    对**抓取**那支，这是把新物体**缩到旧物体的像**里（舵机 45.1mm 被放到 167mm 处，
    去顶替 85mm 处那个 21.5mm 的方块），语义自洽、IoU 才有意义。
    但干扰物**躺在桌上**：照抄这一步，舵机被推到 339mm 处（真实距离 ~218mm），
    **物心掉到桌面以下 55mm、最低点 −68mm** —— 实测过，是这一支第一个版本的行为。
    桌上物体的位置是被单应钉死的，不该为了凑面积把它挪走。

用法：
    PY=/home/zjq/miniconda3/envs/lerobot/bin/python
    $PY -I workplace/calibration/pose6d_distractor.py --frames 0,20,100,340
    $PY -I workplace/calibration/pose6d_distractor.py --mesh cube21.5 --frames 0,20
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures"


def _load(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


MR = _load("_pd_mr", HERE / "meshrender.py")
F = _load("_pd_fit", HERE / "fit_mesh_pose.py")
P6 = _load("_pd_p6", HERE / "pose6d_swap.py")
PW = _load("_pd_pw", HERE / "pose6d_wrist_fit.py")   # 材质 + 限角顶点法线
BS = P6.BS
A = P6.A

# 方块档面积。★ 上界不是「书的面积」—— 书被夹爪挡住时会碎进这个区间（踩过），
#   真正分开书和方块的是下面的几何锚，不是这个区间。
BLUE_AREA = (400.0, 9000.0)
UP_ANCHOR_TOL = 40.0         # up 图上离锚形心多少像素内算「还是那块」（实测抖 <2px）
WRIST_ANCHOR_TOL = 220.0     # FK⊗手眼在桌面这段深度上实测差 21~102px
ARM_CELL_M = 2e-3            # 只用来渲机械臂剪影做深度比较，不参与位姿


# ------------------------------------------------------------------ 蓝块选择
#
# **颜色不管用**：蓝块和腕图左上角那本蓝书是**同一种蓝**。逐块实测（f0/20/100/150/200/300/340）：
#
#     蓝书   H≈116.9  S≈132.5  V≈106.3   面积 13194~45794px
#     蓝块   H≈113.0  S≈104.8  V≈161.8   面积  1030~2921px
#
# 面积也不是单一门槛能分开的：书被夹爪挡住时会碎成 ~8500px 的块、正落在方块档 400~9000px 里
# （踩过：光按面积取「离某点最近」，f200 挑到的是书角）。所以用**几何锚**，而且锚在 base 系
# —— 两个视角用的是同一个物点。


def blue_mask(bgr, g):
    """蓝块检测器。`mask_fn` 要 **RGB** 输入，`BS.blue_mask_bgr` 已经包过这一层。"""
    return BS.blue_mask_bgr(g)(bgr)


def cube_candidates(mask):
    """「面积 + 形状」双门之后的候选：[(面积, 形心, stats)]。

    形状门是必需的：书碎出来的块往往又长又扁（fill 低、aspect 大），方块接近方形。
    但**门不干净**（书角也可能近似方形），所以候选还得靠几何锚来定。
    """
    n, lab, st, ce = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for k in range(1, n):
        a = int(st[k, 4])
        if not (BLUE_AREA[0] <= a <= BLUE_AREA[1]):
            continue
        w, h = int(st[k, 2]), int(st[k, 3])
        if 0.55 <= w / max(h, 1) <= 1.80 and 0.45 <= a / max(w * h, 1) <= 0.97:
            out.append((a, np.asarray(ce[k], float), st[k]))
    return out


def comp_at(bgr, g, c_px):
    """取形心离 `c_px` 最近的**蓝块连通块**（bool (H,W)）。"""
    m = blue_mask(bgr, g).astype(np.uint8)
    n, lab, _, ce = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = min(range(1, n), key=lambda k: np.hypot(ce[k][0] - c_px[0], ce[k][1] - c_px[1]))
    return lab == k


def up_pick(g, up, anchor_px):
    """up 图上的蓝块 → (单块 bool, 形心, 面积)。`anchor_px=None` 时取最大的一块。"""
    cands = cube_candidates(blue_mask(up, g))
    if not cands:
        return None
    if anchor_px is None:
        comp = max(cands, key=lambda c: c[0])
    else:
        comp = min(cands, key=lambda c: np.hypot(*(c[1] - np.asarray(anchor_px))))
        if np.hypot(*(comp[1] - np.asarray(anchor_px))) > UP_ANCHOR_TOL:
            return None
    m = comp_at(up, g, comp[1])
    return None if m is None else (m, comp[1], comp[0])


def board_point(g, c_px):
    """up 图像素 → base 系桌面点（走 up 单应 `Hi` ⊗ `T_base_board`）。

    ★ 精度上的两句实话：
      ① **稳定性 ≠ 准度**。上面量到的 1.17mm 是重复性；单应本身是在棋盘上标出来的，
         板如果挪过就作废（踩过）。
      ② **蓝块的形心不在接触面上**，它是顶面+侧面的像心、离桌面约 h/2≈12mm。
         单应只对 z=0 精确，所以这点有个**视差**：抬高 h 的物体会沿「背离像心」的方向
         外移约 `(h/D)·|q − 像心|`。蓝块在像素 (144,50)、像心约 (320,240)，
         `|q−像心| ≈ 104mm`，取 D 的名义区间 450~700mm ⇒ 视差 **2~3mm**。
         不修正它：修正要用真 up 相机高 D，而 D 恰恰是定不出来的那个自由度
         （见 `pose6d_swap` 末尾的注释）。当误差棒报出来。
    """
    q = g["Hi"] @ np.array([c_px[0], c_px[1], 1.0])
    return (np.asarray(g["T_base_board"], float)
            @ np.array([q[0] / q[2], q[1] / q[2], 0.0, 1.0]))[:3]


# ------------------------------------------------------------------ 静置姿态

def rest_pose(mesh):
    """把网格摆成**最大面朝下**的静置姿态，并把物心挪到原点。

    重建产物的物体系本来就是 PCA 系，所以这里只做一个**带符号的置换**：把外接盒
    **最短的那一维**转到 +Z。不引入任何新的自由度。

    返回 `(tris_new, half_h_m, ext_mm)`。`half_h_m` 是静置时物心离桌面多高 —— 也就是
    「位置」里那个已知的高度。`tris_new` **已经在静置系里**，调用方只需再叠一个偏航，
    不要再乘一次这里的 `R`（踩过：转两遍 ⇒ 底面沉到桌面以下 6.3mm）。
    """
    p = mesh.tris0.reshape(-1, 3).astype(np.float64)
    # ★ 物心取**外接盒中点**，不取顶点均值：重建网格的顶点在面上不是均匀分布的，
    #   顶点均值会偏离盒心，静置后底面就不在 z=0 上。
    c = (p.min(0) + p.max(0)) / 2.0
    e = np.ptp(p, axis=0)
    j = int(np.argmin(e))                       # 最短维 → z
    k = sorted([a for a in (0, 1, 2) if a != j], key=lambda a: -e[a])
    R = np.zeros((3, 3))
    R[j, 2] = 1.0
    R[k[0], 0] = 1.0
    R[k[1], 1] = 1.0
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    tris = ((p - c) @ R.T).reshape(-1, 3).astype(np.float32)
    return tris, float(np.ptp(tris, axis=0)[2]) / 2.0, np.round(np.ptp(tris, axis=0) * 1000, 1)


def pose_T(P_center, yaw_deg):
    """选定的 3D 摆放：位置 = 量出来的物点，朝向 = 静置姿态再绕竖直轴转 `yaw_deg`。

    朝向是**设计选择**（蓝块不携带舵机朝向的信息），所以就是一行显式的构造，不藏在
    任何优化器里。
    ★ 这里的旋转**只有偏航** —— 静置姿态已经被 `rest_pose` 烘进网格顶点里了。
      踩过：这儿再乘一次 `R_rest` 就是**转两遍**，底面沉到桌面以下 6.3mm。"""
    th = np.deg2rad(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(th), -np.sin(th), 0],
                          [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    T[:3, 3] = np.asarray(P_center, float)
    return T


# ------------------------------------------------------------------ 两台相机

def up_metric_cam(g, c_px, cam_h_mm, W, H, h_mm):
    """up 视图的**量尺正确**的局部针孔。

    位置/朝向沿用 `F.up_cam_from_area`（光轴垂直下视、主点 = 观测形心、物体落在光轴上），
    但焦距**不用「面积反解」** —— 那是把新物体缩到旧物体的像里（抓取那支的语义）。
    这里要的是**真实尺寸**，所以 f 由桌面处实测尺度 spm 定：`spm = f/cam_h ⇒ f = spm·cam_h`。

    ★ 这条式子和 `up_cam_from_area` 的 docstring 里写的 `f ≈ spm·cam_h` 是同一条。
      实测：蓝块处 spm = 1688 px/m、名义 cam_h = 700mm ⇒ f = 1182px；
      而面积反解那条路在同一个位置给 597px —— **比值 1.98，正好是 45.1/24 那一档**。
      两个 f 都对，问的问题不同：595 问「怎么让舵机占住蓝块的像」，1182 问「舵机真实多大」。

    ★ f 与 cam_h 退化（同一个 spm 配任何 cam_h 都给同一条 `f = spm·cam_h`），所以
      渲出的尺寸 ≈ `L·spm·(1 + h/cam_h)` —— 对名义 cam_h 只有 ~2% 的依赖。报 spm 就够了。
    """
    T_cam, centre = F.up_cam_from_area(g, c_px, cam_h_mm, h_mm)
    q = g["Hi"] @ np.array([c_px[0], c_px[1], 1.0])
    spm, _ = BS.up_scale_px_per_m(g, q[:2] / q[2])
    f = spm * cam_h_mm / 1000.0
    K = np.array([[f, 0, c_px[0]], [0, f, c_px[1]], [0, 0, 1.0]])
    return MR.Cam(K, np.zeros(5), T_cam, W=W, H=H), dict(f_px=f, spm=spm, T_cam=T_cam,
                                                         centre=centre)


def wrist_cam(ds, i, g, P_center, c_obs):
    """腕相机：FK⊗手眼 + **单点 δ 修正**（和抓取那支同一个工具 `BS.solve_cam_delta`）。

    抓取那支用的是「夹持中方块中心 ↔ 观测形心」，这里用的是「up 单应定出来的蓝块桌面点
    + 静置半高 ↔ 观测形心」—— 同一个物理物点，不同的物。
    单点 δ 沿视线欠定，那一段保持 0（`solve_cam_delta` 取最小范数解）。"""
    fr = ds[i]
    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    T0 = P6.cam_from_state(g, q)
    delta, _ = BS.solve_cam_delta(g["K"], g["dist"], T0, [(P_center, c_obs)])
    return MR.Cam(g["K"], g["dist"], P6.cam_from_state(g, q, delta)), q, delta, T0


# ------------------------------------------------------------------ 合成

def render(mesh, T, cam, cell, albedo, smooth=True):
    tris_b = MR.transform(mesh.level(cell), T)
    ras = MR.rasterize(tris_b, cam)
    if smooth:
        ras = PW.smooth_normals(ras, tris_b, cam, PW.smooth_for(mesh, cell, tris_b))
    rgb = MR.shade(ras, cam, PW.mesh_material(P6.load_lighting()), albedo)
    return ras, rgb, tris_b


def composite(orig, obsm, bm, rgb, front):
    """L0/L1/L2。

    ★★ 这里的 L2 **不是**抓取那支的 `out[cm==0] = fg[cm==0]` —— 那条等式的前提是
      「夹爪永远在注入物前面」，而干扰物这边**没有那个前景**；而且注入的舵机比蓝块大一圈，
      剪影会伸到擦除掩码外面，照抄那条红线会把舵机削掉一块。
      这里的红线换成：**「被擦掉的那块」∪「网格剪影」之外逐位等于原帧**（同样是构造性等式）。

    ★ 被擦掉的那块**从 `plate` 自己量**（`plate != orig`），不要拿 `obsm` 去猜 ——
      `erase_cube` 自己会按 `grow`/`shadow` 外扩（阴影也算），猜边界只会漏掉那一圈。
      踩过：拿未外扩的掩码当 allowed，红线外虚报 1105px，全是那圈羽化。"""
    plate = BS.erase_cube(orig.copy(), obsm.astype(np.uint8) * 255,
                          grow=3, shadow=True, fill="diffuse")[0]
    erased = (plate != orig).any(2)
    out = plate.copy()
    out[bm] = (rgb * 255.0)[bm]
    l1 = out.copy()
    out[front] = orig[front]                      # L2：真前景摆回前面
    changed = np.abs(out.astype(np.int16) - orig.astype(np.int16)).sum(2) > 0
    return dict(plate=plate, l1=l1, out=out, cm=(obsm.astype(np.uint8) * 255),
                erase_px=int(erased.sum()),
                outside_px=int((changed & ~(bm | erased)).sum()))


# ------------------------------------------------------------------ 一帧

def up_frame(ds, i, g, P_anchor, anchor_px, mesh, yaw_deg, albedo,
             cell, cam_h_mm, smooth=True, verbose=True):
    up = BS.up_bgr(ds, i)
    got = up_pick(g, up, anchor_px)
    if got is None:
        if verbose:
            print(f"  f{i}: up 图里没有方块档的蓝块，跳过")
        return None
    obs, c_up, area = got

    # 位置：**up 单应给的桌面点**（量出来的），不是这一帧的形心反算 —— 蓝块静止，
    #       逐帧反算只会把掩码噪声放进来。
    P_center = np.asarray(P_anchor, float) + np.array([0.0, 0.0, HALF_H["h"]])
    cam, cinfo = up_metric_cam(g, c_up, cam_h_mm, up.shape[1], up.shape[0], HALF_H["h"] * 1000)
    T = pose_T(P_center, yaw_deg)
    ras, rgb, tris_b = render(mesh, T, cam, cell, albedo, smooth)
    bm = ras["mask"]

    st = ds[i]["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    arm = MR.rasterize(MR.gather(A.URDF_PATH, q, cell=ARM_CELL_M)[0], cam, cull=True)
    front = arm["mask"] & bm & (arm["depth"] < ras["depth"])

    r = composite(up, obs, bm, rgb, front)
    return dict(view="up", frame=i, orig=up, obs=obs, mesh_px=int(bm.sum()),
                obj_px=int(obs.sum()), ras=ras, rgb=rgb, arm_mask=arm["mask"],
                # 覆盖：舵机把蓝块盖住了多少（单边性）。舵机更大 ⇒ 这个数应该 ≈1。
                cover=float((bm & obs).sum()) / max(int(obs.sum()), 1),
                size_ratio=float(bm.sum()) / max(int(obs.sum()), 1),
                z_min_mm=float(tris_b[:, :, 2].min() * 1000.0),
                arm_front_px=int(front.sum()), c_up=c_up.tolist(),
                spm=cinfo["spm"], f_px=cinfo["f_px"], area=area, T=T, **r)


def wrist_frame(ds, i, g, P_anchor, mesh, yaw_deg, albedo, cell,
                smooth=True, verbose=True):
    fr = ds[i]
    fg = P6.bgr_of(fr["observation.images.front"])
    P_center = np.asarray(P_anchor, float) + np.array([0.0, 0.0, HALF_H["h"]])

    # 先用**未修正**的 FK 相机把锚投出来，在候选里挑最近的那块
    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    cam_fk = MR.Cam(g["K"], g["dist"], P6.cam_from_state(g, q))
    uv0, dep0 = cam_fk.project(P_center.reshape(1, 3))
    cands = cube_candidates(blue_mask(fg, g))
    if not cands or dep0[0] <= 0:
        if verbose:
            print(f"  f{i}: 腕图里没有方块档的蓝块，跳过")
        return None
    j = int(np.argmin([np.hypot(*(c[1] - uv0[0])) for c in cands]))
    d0 = float(np.hypot(*(cands[j][1] - uv0[0])))
    if d0 > WRIST_ANCHOR_TOL:
        if verbose:
            print(f"  f{i}: 腕图蓝块离 FK 投影 {d0:.0f}px > {WRIST_ANCHOR_TOL:.0f}px，跳过")
        return None
    obs = comp_at(fg, g, cands[j][1])
    if obs is None:
        return None
    c_obs = np.array([np.nonzero(obs)[1].mean() + 0.5, np.nonzero(obs)[0].mean() + 0.5])

    cam, q, delta, T0 = wrist_cam(ds, i, g, P_center, c_obs)
    T = pose_T(P_center, yaw_deg)
    ras, rgb, tris_b = render(mesh, T, cam, cell, albedo, smooth)
    bm = ras["mask"]

    # 前景：几何路（URDF 剪影 × 深度比较，FK⊗手眼含 δ）。干扰物这边它应当 ≈0，
    # 量出来就是「臂到底有没有挡在蓝块前面」这个事实。
    arm_ras, front_geo, front_raw = P6.front_mask_geometric(ds, i, g, cam, ras, obs)

    r = composite(fg, obs, bm, rgb, front_geo)
    return dict(view="wrist", frame=i, orig=fg, obs=obs, mesh_px=int(bm.sum()),
                obj_px=int(obs.sum()), ras=ras, rgb=rgb, arm_mask=arm_ras["mask"],
                cover=float((bm & obs).sum()) / max(int(obs.sum()), 1),
                size_ratio=float(bm.sum()) / max(int(obs.sum()), 1),
                z_min_mm=float(tris_b[:, :, 2].min() * 1000.0),
                arm_front_px=int(front_geo.sum()), front_raw_px=int(front_raw.sum()),
                anchor_d_px=d0, n_cand=len(cands),
                fk_delta_mm=(np.asarray(delta) * 1000).round(1).tolist(),
                # 互检：δ 是把**锚点**按到观测形心上的，所以像心对齐是构造性的；
                # 但**剪影覆盖**（cover）不是 —— 它由尺度、朝向、以及两台相机的几何一起决定。
                P_center_mm=(P_center * 1000).round(1).tolist(), T=T, **r)


# ------------------------------------------------------------------ 出图

def figure(rows, out_path, title="", mesh_name=""):
    """双视角：每帧**两行**（上 = up 相机，下 = 腕相机），每行 5 栏。

    栏：原帧 / L0 擦除修补 / L1 网格整剪影 / 合成 / |差分|。
    两行画的是**同一个** base 系摆放（`T` 逐位相同），只是两台相机不同。
    """
    panels = []
    for r in rows:
        for v in ("up", "wrist"):
            d = r.get(v)
            if d is None:
                continue
            bm = d["ras"]["mask"]
            ov = d["l1"].copy()
            cv2.drawContours(ov, cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                             (0, 255, 255), 1)
            d_ = d["out"].astype(np.int16) - d["orig"].astype(np.int16)
            dv = cv2.applyColorMap(np.clip(np.abs(d_).sum(2) * 2, 0, 255).astype(np.uint8),
                                   cv2.COLORMAP_INFERNO)
            # 文字全用 ASCII：cv2 的 Hershey 字体不认中文（踩过）
            if v == "up":
                sub = [f"spm={d['spm']:.0f}px/m  f={d['f_px']:.0f}px  real scale",
                       f"arm in front {d['arm_front_px']}px"]
            else:
                sub = [f"delta {d['fk_delta_mm']}mm (single-point)",
                       f"arm in front {d['arm_front_px']}px"]
            panels.append(np.hstack([
                MR.put_lines(d["orig"], [f"original  f{d['frame']}  {v}"],
                             org=(8, 22), scale=0.5),
                MR.put_lines(d["plate"], ["L0  erase distractor + inpaint bg"],
                             org=(8, 22), scale=0.5),
                MR.put_lines(ov, [f"L1  {mesh_name}  {d['mesh_px']}px"] + sub,
                             org=(8, 22), scale=0.5),
                MR.put_lines(d["out"], [f"composite  covers block {100*d['cover']:.1f}%  "
                                        f"size x{d['size_ratio']:.2f}",
                                        f"foot z {d['z_min_mm']:+.1f}mm  "
                                        f"red-line outside {d['outside_px']}px"],
                             org=(8, 22), scale=0.5),
                MR.put_lines(dv, ["|diff| vs original"], org=(8, 22), scale=0.5)]))
    img = np.vstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[图] {out_path}  — {title}")
    return img


HALF_H = {}


def anchor(g, ds, span=440, step=10, verbose=True):
    """蓝块在 base 系的桌面锚点：ep0 上每 `step` 帧量一次，取**中位**。

    不用某一帧的形心，免得把掩码噪声当成物点。返回 `(P_anchor, c0, dev_mm)`。
    抽成函数是为了让别的脚本（SaGA 那条）能复用**同一个**锚点，不然两份拷贝会漂。
    """
    P_list, c0 = [], None
    for i in range(0, span, step):
        got = up_pick(g, BS.up_bgr(ds, i), None if c0 is None else c0)
        if got is None:
            continue
        if c0 is None:
            c0 = got[1]
        P_list.append(board_point(g, got[1]))
    if not P_list:
        return None, None, None
    P_list = np.asarray(P_list)
    P_anchor = np.median(P_list, axis=0)
    dev = np.linalg.norm(P_list - P_anchor, axis=1) * 1000
    if verbose:
        print(f"[锚] up 单应给蓝块桌面点 = {np.round(P_anchor*1000, 2)}mm"
              f"（ep0 {len(P_list)} 次采样的中位）"
              f"｜重复性：最大偏离 {dev.max():.2f}mm、中位 {np.median(dev):.2f}mm")
    return P_anchor, c0, dev


def build_mesh(mesh_name):
    """按 `--mesh` 的约定造注入网格：(mesh, albedo, half_h_mm, ext_mm, 打印行)。"""
    if mesh_name.startswith("cube"):
        e = float(mesh_name.replace("cube", "")) / 1000.0
        base = F.MeshObj(MR.box(e, e, e), name=mesh_name, cells=(2e-3, 1e-3, 0.5e-3))
        albedo = PW.CUBE_ALBEDO
    else:
        base = F.load_mesh(mesh_name)
        albedo = PW.SERVO_ALBEDO
    tris, half_h, ext = rest_pose(base)
    HALF_H["h"] = half_h
    mesh = F.MeshObj(tris, name=base.name + "_rest", cells=base.cells)
    return mesh, albedo, half_h, ext, base.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--frames", default="0,20,100,340",
                    help="腕图上蓝块清楚可见的帧（ep0 的 f345 之后腕图就看不到它了）")
    ap.add_argument("--mesh", default="sts3215_03a_v1_v50_h0.5_m1")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="静置朝向的偏航角（**设计选择**，不是估计量）。0 = 长边朝 base +x")
    ap.add_argument("--draw-cell-mm", type=float, default=1.5)
    ap.add_argument("--cam-h-mm", type=float, default=None,
                    help="up 局部相机高度（名义值，只影响 ~2%%；默认 BS.UP_CAM_H_MM）")
    ap.add_argument("--flat", action="store_true", help="关掉顶点法线插值（棱纹会明显）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = [int(x) for x in a.frames.split(",") if x.strip()]
    g = BS.load_geom()
    cam_h = float(a.cam_h_mm if a.cam_h_mm is not None else BS.UP_CAM_H_MM)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(a.dataset, root=str(REPO / "dataset" / a.dataset))

    mesh, albedo, half_h, ext, base_name = build_mesh(a.mesh)
    print(f"注入网格：{base_name}  静置外接盒 {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} mm "
          f"（z = 竖直）⇒ 物心离桌面 h = {half_h*1000:.2f}mm  面数 {len(mesh.tris0)}")

    P_anchor, c0, dev = anchor(g, ds)
    if P_anchor is None:
        print("[锚] up 图里没找到蓝块 → 退出")
        return 1
    print(f"       ★ 准度另说：形心离接触面约 {half_h*1000:.0f}mm 高，单应只对 z=0 精确，"
          f"视差估 2~3mm")

    rows = []
    for i in frames:
        u = up_frame(ds, i, g, P_anchor, c0, mesh, a.yaw_deg, albedo,
                     a.draw_cell_mm / 1000.0, cam_h, smooth=not a.flat)
        w = wrist_frame(ds, i, g, P_anchor, mesh, a.yaw_deg, albedo,
                        a.draw_cell_mm / 1000.0, smooth=not a.flat)
        rows.append(dict(frame=i, up=u, wrist=w))
        if u is not None:
            print(f"  up  f{i}: 蓝块 {u['obj_px']}px → 舵机 {u['mesh_px']}px  "
                  f"盖住蓝块 {100*u['cover']:.1f}%  面积 x{u['size_ratio']:.2f}  "
                  f"底面 z {u['z_min_mm']:+.1f}mm  臂挡前面 {u['arm_front_px']}px  "
                  f"红线外 {u['outside_px']}px")
        if w is not None:
            print(f"  腕  f{i}: 蓝块 {w['obj_px']}px → 舵机 {w['mesh_px']}px  "
                  f"盖住蓝块 {100*w['cover']:.1f}%  面积 x{w['size_ratio']:.2f}  "
                  f"底面 z {w['z_min_mm']:+.1f}mm  锚偏 {w['anchor_d_px']:.0f}px/"
                  f"{w['n_cand']}候选  δ={w['fk_delta_mm']}mm  "
                  f"臂挡前面 {w['arm_front_px']}px  红线外 {w['outside_px']}px")

    if not any(r["up"] or r["wrist"] for r in rows):
        print("没有一帧成功")
        return 1

    for v in ("up", "wrist"):
        ds_ = [r[v] for r in rows if r[v] is not None]
        if not ds_:
            continue
        print(f"汇总 {v:5s}：盖住蓝块 {100*min(d['cover'] for d in ds_):.1f}~"
              f"{100*max(d['cover'] for d in ds_):.1f}%｜面积 x"
              f"{min(d['size_ratio'] for d in ds_):.2f}~{max(d['size_ratio'] for d in ds_):.2f}"
              f"｜底面 z {min(d['z_min_mm'] for d in ds_):+.1f}~"
              f"{max(d['z_min_mm'] for d in ds_):+.1f}mm"
              f"｜红线外 max {max(d['outside_px'] for d in ds_)}px"
              f"｜臂挡前面 max {max(d['arm_front_px'] for d in ds_)}px")

    out = Path(a.out) if a.out else FIG / f"pose6d_distractor_{mesh.name}.png"
    figure(rows, out, mesh_name=mesh.name,
           title=f"干扰物（桌上蓝块）换成重建网格 {base_name}：up + 腕 双视角")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
