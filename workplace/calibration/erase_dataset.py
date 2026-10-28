#!/usr/bin/env python
"""把「夹黄色方块 → 丢红盒子」的演示数据集整段换成**橡皮**，再写成一个新的 LeRobot 数据集。

为什么要按 ep 分三段，而不是把 `pose6d_dual_fit.dual_frame` 一帧帧套到底：

  · **夹取前**（episode 开头 → 夹爪合上）：方块躺在桌上一动不动，但腕相机在下降、
    手指在靠近。`dual_frame` 的位姿是「物体中心 = TCP + t_tool」，夹取前这样渲出来的
    橡皮会跟着夹爪在天上飞。所以这一段用**桌面静态位姿**：XY 由 up 图的方块观测
    （`ball_swap.contact_point`，含顶面高度视差修正）给出，Z 取橡皮半厚（贴桌），
    朝向取夹取那一刻的真实夹取朝向（换成橡皮后姿态不跳）。
  · **夹取 → 放下**：直接用 `dual_frame` —— 那段是已经验过 163 帧的定稿链路
    （L0 擦除 + L1 渲染 + L2 机器人前景）。
  · **放下之后**：方块已经掉进红盒子里，位姿不可观测，再渲 3D 会穿模。这一段只做
    **颜色替换**：把落点附近的黄色像素按橡皮的色相/饱和度重映射，形状不动。
    盒口那个黄色锁扣（~86px）靠面积下限挡掉。

分段边界全部由夹爪位置通道自动判（不需要逐集手标）：
  rest → open（张开到最大）→ close（往回合）就当作夹取开始，之后夹爪读数明显回升
  并保持 5 帧就是放下。50 集全部验证过能判出来。

输出分两步（可断点续跑）：
  1. `--render`：每集一个 shard 数据集（`<tmp>/ep000` …），完成后写 `done.json`；
     已完成的 shard 直接跳过。
  2. `--aggregate`：`lerobot.datasets.aggregate.aggregate_datasets` 把 shard 拼成
     最终数据集（视频是 ffmpeg concat 流拷贝，不重编码）。

用法：
    # 先看几张效果图（原图 | 换完）
    python workplace/calibration/erase_dataset.py --probe --episodes 0 \
        --probe-frames 5,120,240,300,360,420

    # 全量渲染（8 进程）
    python workplace/calibration/erase_dataset.py --render --episodes all --workers 8

    # 拼成最终数据集
    python workplace/calibration/erase_dataset.py --aggregate

阶段划分的实测（ep0）：夹爪 f259 开始合上、f380 明显张开；up 图里方块 f290 才离开
桌面、f400 之后躺在盒子里只露 ~280px。
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKPLACE = HERE.parent
REPO = WORKPLACE.parent
for _p in (REPO, WORKPLACE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


PD = _load("_ed_pd", HERE / "pose6d_dual_fit.py")
BS, P6, PW, MR, F, A = PD.BS, PD.P6, PD.PW, PD.MR, PD.F, PD.A

UP_KEY = "observation.images.up"
FRONT_KEY = "observation.images.front"

# 桌面/搬运两档黄色阈值：桌面用标定的 `yellow_mask`，夹爪靠近后方块被压暗、
# 标定阈值会整块漏掉（ep0 f240 起就是 0px），退回宽松档（S≥45/V≥55）。
LOOSE = dict(h_lo=8, h_hi=48, s_min=45, v_min=55)

# 放下后的颜色替换：实测渲染出来的橡皮（up 视角、顶面）中位 HSV = (13, 187, 206)，
# 方块黄约 (22, 200~255, 200+)。所以只把色相拨到 13、饱和度压到 ~0.9 倍，
# 亮度原样保留（盒子里的明暗关系不能丢）。
# 用 `--eraser-albedo` 换橡皮颜色时，这个色相会跟着反照率重算（见 _hue_of），
# 否则盒子里那颗会跟新橡皮不同色。
RECOLOR_H, RECOLOR_S = 13, 0.9


def _hue_of(albedo_bgr):
    """BGR 反照率 -> OpenCV 色相（0..179），给放下后的颜色替换对齐用。"""
    px = np.uint8([[np.clip(np.asarray(albedo_bgr, float), 0, 1) * 255]])
    return int(PD.cv2.cvtColor(px, PD.cv2.COLOR_BGR2HSV)[0, 0, 0])


# ================================================================ 小工具

def _loose_mask(bgr, h_lo=8, h_hi=48, s_min=45, v_min=55):
    hsv = PD.cv2.cvtColor(bgr, PD.cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[:, :, j].astype(np.float32) for j in range(3))
    m = ((H >= h_lo) & (H <= h_hi) & (S >= s_min) & (V >= v_min)).astype(np.uint8) * 255
    return PD.cv2.morphologyEx(m, PD.cv2.MORPH_OPEN,
                               PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (5, 5)))


def _near_components(mask, centre, max_d, area_lo, area_hi):
    """只留形心落在 centre 半径 max_d 内、面积在区间里的连通域（uint8 255 掩码）。"""
    n, lab, st, ce = PD.cv2.connectedComponentsWithStats(mask, 8)
    keep = []
    for j in range(1, n):
        a = int(st[j, PD.cv2.CC_STAT_AREA])
        if not (area_lo <= a <= area_hi):
            continue
        if np.hypot(ce[j][0] - centre[0], ce[j][1] - centre[1]) <= max_d:
            keep.append(j)
    if not keep:
        return None
    return np.isin(lab, keep).astype(np.uint8) * 255


def _wrist_cube_px(S, fg, T_base_cam):
    """静态段：在腕图里找**桌上那块**方块，返回 (观测像素 or None, 模型投影像素)。

    不能直接用「离模型投影最近」：接近时桌上会冒出一堆几百像素的暖色反光
    （ep0 f240 最近的一块只有 310px，离投影 68px，比真方块还近），而真方块随
    相机下降会长到几万像素。所以按**模型给的深度**算期望面积，再挑面积比最接近
    的那个 —— 这是唯一随构型变化的先验，恰好能把反光筛掉。
    """
    K, dist = S["K"], S["dist"]
    P = np.array([S["rest_base"][0], S["rest_base"][1], S["CUBE_C_H"]])
    uv, z = MR.Cam(K, dist, T_base_cam).project(P[None])
    p, d = uv[0], float(z[0])
    if d <= 0.02:
        return None, p, 0.0, None
    side = 0.0215 * K[0, 0] / d
    a_exp = 1.3 * side ** 2
    # 允许「模型投影 vs 观测」差多少像素：腕链误差实测 ~3cm，换成像素就是
    # 0.03·f/d。给 2 倍余量。不能用一个固定大半径 —— ep19 f120 顶部边缘那块
    # 暖色（y=12，被画幅裁掉一半）离投影 206px，会把真方块顶掉。
    max_d = max(90.0, 0.06 * K[0, 0] / d)

    def _pick(mask, lo, hi, max_d):
        n, lab, st, ce = PD.cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for j in range(1, n):
            a = float(st[j, PD.cv2.CC_STAT_AREA])
            if a < 200 or not (lo * a_exp <= a <= hi * a_exp):
                continue
            if np.hypot(ce[j][0] - p[0], ce[j][1] - p[1]) > max_d:
                continue
            r = abs(np.log(a / a_exp))
            if best is None or r < best[0]:
                best = (r, np.array(ce[j], float), a, (lab == j).astype(np.uint8) * 255)
        return None if best is None else best[1:]

    got = _pick(BS.yellow_mask(fg, "wrist"), 0.25, 4.0, max_d)
    if got is not None:
        return got[0], p, got[1], got[2]
    # 贴近时（f180+）标定档会把**机械臂阴影里的桌面**整片圈进来（ep0 f240：53k px，
    # 是模型的 4.6 倍），面积门把它挡掉之后就没有候选了。抬高饱和度再找一次：
    # 实测桌 S≈50~85 / 方块 S≈130~185，这一刀正好切在中间，那几帧里方块是唯一的大块。
    hsv = PD.cv2.cvtColor(fg, PD.cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 40) &
         (hsv[:, :, 1] >= 130) & (hsv[:, :, 2] >= 120)).astype(np.uint8) * 255
    m = PD.cv2.morphologyEx(m, PD.cv2.MORPH_OPEN,
                            PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (5, 5)))
    got = _pick(m, 0.2, 5.0, max_d * 1.15)
    return (None, p, 0.0, None) if got is None else (got[0], p, got[1], got[2])


def _wrist_cam_corrected(S, T_base_cam, c_obs, a_obs):
    """腕链（FK⊗手眼）在数据集构型上差 2~3cm，而且随构型变 —— 常量手眼吸不住。

    静态段方块在 base 里是**不动且已知**的（up 相机标定过），所以可以直接拿
    「模型投影 vs 观测像素」解一个相机平移 δ（`ball_swap.solve_cam_delta`），
    不是循环论证。同一套修正就是 `pose6d_swap` 抓取链在用的那支。

    ★ 单点 δ 沿视线欠定，min-norm 解**不修深度**；而这条链的深度误差恰恰是随
      构型变的（ep0：模型 136mm / 观测 ~95mm，橡皮会被画小 ~30%）。所以再沿
      「方块→相机」这条视线把距离拉到观测值 —— 沿视线移动不改变投影像素，
      只改尺度。深度由观测面积反推：方块边长 21.5mm，面积开方当边长。
    """
    K = S["K"]
    P = np.array([S["rest_base"][0], S["rest_base"][1], S["CUBE_C_H"]])
    delta, _ = BS.solve_cam_delta(K, S["dist"], T_base_cam, [(P, c_obs)])
    T = T_base_cam.copy()
    T[:3, 3] += np.asarray(delta, float).reshape(3)
    if a_obs and a_obs > 200:
        d_obs = 0.0215 * K[0, 0] / np.sqrt(float(a_obs))
        v = T[:3, 3] - P
        n = float(np.linalg.norm(v))
        # 面积反推的深度要是离谱（选的连通域不是方块），只保留平面 δ
        if n > 1e-6 and 0.55 <= d_obs / max(n, 1e-6) <= 1.8:
            T[:3, 3] = P + v / n * d_obs
    return T


def _med5(x):
    x = np.asarray(x, float)
    pad = np.pad(x, 2, mode="edge")
    return np.array([np.median(pad[i:i + 5]) for i in range(len(x))])


def _bgr_of(t):
    return PD.cv2.cvtColor(
        np.clip(t.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8),
        PD.cv2.COLOR_RGB2BGR)


def _to_hwc_uint8(t):
    arr = t.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()
    return (arr * 255.0).astype(np.uint8)


def _clean_features(base_features):
    out = {}
    for k, v in base_features.items():
        v2 = dict(v)
        if v2.get("dtype") in ("image", "video"):
            v2.pop("info", None)
        out[k] = v2
    return out


# ================================================================ 阶段判定

def detect_phases(gr):
    """夹爪位置通道 → (rest_level, open_max, i_open, i_close, i_rel)。

    夹爪约定（实测）：数值小 = 合上（~12–20），大 = 张开（~30–42）。
    一个 episode 里夹爪只张开一次、合上一次、再张开一次（把方块丢进盒子）。
    """
    gr = np.asarray(gr, float)
    L = len(gr)
    rest = float(np.median(gr[:min(30, L)]))
    om = float(gr.max())
    i_open = int(np.argmax(gr))
    if om - rest < 5.0:
        raise ValueError(f"夹爪没有明显的张开相位：rest={rest:.1f} max={om:.1f}")
    thr = 0.5 * (rest + om)
    sm = _med5(gr)
    i_close = next((i for i in range(i_open, L) if sm[i] < thr), None)
    if i_close is None:
        raise ValueError("找不到夹取（夹爪合上）那一帧")
    closed = float(np.median(gr[min(i_close + 12, L - 1):min(i_close + 32, L)]))
    i_rel = next((i for i in range(i_close + 20, L - 5)
                  if all(sm[i + j] > closed + 1.5 for j in range(5))), L - 1)
    return rest, om, i_open, i_close, i_rel


# ================================================================ 每集一次：桌面位置 + 标定常量

def find_rest_px(ds, g, i0, i1, step=3):
    """在 [i0, i1) 里找**不动**的那块黄：多次检测做聚类，取命中最多的位置。

    夹爪自己的黄色颚、暖光下的机械臂也会进掩码，但它们跟着臂动 —— 逐帧位置对不上，
    聚不成簇。方块是唯一在桌面期**静止**的目标（ep15/19/37 开头几帧检测不到，
    但后面会补上，所以扫整个夹取前窗口而不是只扫开头）。
    """
    mfn = BS.cube_mask_fns(g, "up", "yellow")[0]
    dets = []
    for i in range(i0, i1, step):
        up = BS.up_bgr(ds, i)
        m = mfn(up)
        n, lab, st, ce = PD.cv2.connectedComponentsWithStats(m, 8)
        for j in range(1, n):
            a = int(st[j, PD.cv2.CC_STAT_AREA])
            w, h = int(st[j, 2]), int(st[j, 3])
            if 250 <= a <= 4000 and 18 <= w <= 60 and 18 <= h <= 60:
                dets.append((float(ce[j][0]), float(ce[j][1]), a))
    if not dets:
        return None
    P = np.array([[d[0], d[1]] for d in dets])
    best, best_n = None, -1
    for k in range(len(P)):
        n = int((np.hypot(*(P - P[k]).T) <= 12.0).sum())
        if n > best_n:
            best, best_n = k, n
    sel = np.hypot(*(P - P[best]).T) <= 12.0
    c = np.median(P[sel], axis=0)
    return c, int(sel.sum()), len(dets)


def find_bin_px(ds, g, i0, i1, step=3):
    """放下之后方块躺在盒子里的**静止**位置（up 图）。

    别拿「放开那一帧的 FK 投影」当落点：方块是**掉下去**的，up 图里还会往盒底滑
    3~4 cm（ep0 f380 投影 (365,176)，f420 实测 (438,65)，差 120px）。同样用多帧聚类：
    盒子里的方块不东，机械臂的黄色颚在动、红盒锁扣只有 ~86px（面积下限挡掉）。
    """
    mfn = BS.cube_mask_fns(g, "up", "yellow")[0]
    dets = []
    for i in range(i0, i1, step):
        up = BS.up_bgr(ds, i)
        # 盒子里灯暗、还有红盒反光：标定掩码在这只剩 60~150px，单用会被面积下限
        # 挡掉；此时机械臂已撤走，放宽掩码不会淹，两者取并集最稳。
        m = PD.cv2.bitwise_or(mfn(up), _loose_mask(up))
        n, lab, st, ce = PD.cv2.connectedComponentsWithStats(m, 8)
        for j in range(1, n):
            a = int(st[j, PD.cv2.CC_STAT_AREA])
            w, h = int(st[j, 2]), int(st[j, 3])
            if 120 <= a <= 3000 and 8 <= w <= 70 and 8 <= h <= 70:
                dets.append((float(ce[j][0]), float(ce[j][1]), a))
    if not dets:
        return None
    P = np.array([[d[0], d[1]] for d in dets])
    best, best_n = None, -1
    for k in range(len(P)):
        n = int((np.hypot(*(P - P[k]).T) <= 15.0).sum())
        if n > best_n:
            best, best_n = k, n
    sel = np.hypot(*(P - P[best]).T) <= 15.0
    if int(sel.sum()) < 4:
        return None
    return np.median(P[sel], axis=0), int(sel.sum()), len(dets)


def px_to_base_xy(g, px, h_m=0.03):
    """up 图上一像素 → base 系 XY（含「顶面在高处」的视差修正，同 contact_point）。"""
    H, W = 480, 640
    q = g["Hi"] @ np.array([px[0], px[1], 1.0])
    x0 = q[:2] / q[2]
    C = (g["Hi"] @ np.array([W / 2.0, H / 2.0, 1.0]))
    C = C[:2] / C[2]
    xy = C + (1.0 - h_m / (BS.UP_CAM_H_MM / 1000.0)) * (x0 - C)
    pb = g["T_base_board"] @ np.array([xy[0], xy[1], 0.0, 1.0])
    return pb[:2]


def rest_base_xy(ds, g, rest_px, i0, i1, step=4):
    """up 观测 → 方块**底面中心**在 base 系的 (x,y)，含顶面高度视差修正。"""
    cal = BS.cube_mask_fns(g, "up", "yellow")[0]

    def mfn(bgr):
        m = cal(bgr)
        nm = _near_components(m, rest_px, 14.0, 200, 5000)
        return np.zeros_like(m) if nm is None else nm

    pts = []
    for i in range(i0, i1, step):
        cp = BS.contact_point(BS.up_bgr(ds, i), g, BS.UP_CAM_H_MM / 1000.0, 0.0215,
                              mask_fn=mfn, min_area=120.0)
        if cp is not None:
            pts.append(cp["base_xy"])
    if not pts:
        return None
    return np.median(np.asarray(pts), axis=0)


def setup_episode(ds, g, upc, ep, i0, L, states_ep):
    """一集的全部常量。返回 dict；失败抛异常（调用方跳过这一集）。"""
    gr = states_ep[:, 5]
    rest, om, i_open, i_close, i_rel = detect_phases(gr)
    ig = i0 + i_close
    ir = min(i0 + L - 1, i0 + i_rel + 20)

    t0 = time.time()
    trk = BS.carry_track(ds, ig, ir)
    ut, seed = P6.up_track_ctx(ds, g, ig, ir)
    if not trk or not ut:
        raise RuntimeError(f"ep{ep}: 搬运段跟踪失败 (front {len(trk)} / up {len(ut)})")

    p0, note = BS.estimate_x_tcp_cube(ds, ig, g, BS.UP_CAM_H_MM / 1000.0, 0.0215,
                                      BS.cube_mask_fns(g, "up", "yellow")[0])
    if p0 is None:
        p0 = np.array([0.0, 0.0, -0.035])
    t_tool, t_info = PD.fit_t_tool(ds, upc, ut, range(ig, ir), g, p0)

    cell = 0.0005
    cube = F.MeshObj(MR.box(0.0215, 0.0215, 0.0215), name="cube21.5",
                     cells=(2 * cell, cell, cell / 2))
    # 标定帧必须落在**搬运段**里：ir 后面留的 20 帧余量是给跟踪的，那时方块已经
    # 进盒子了，拿它拟合会把 t_cam 拉飞（实测 ep0 f390 物心距离 222mm）。
    fit_frames = np.linspace(ig + 15, i0 + i_rel - 10, 3).round().astype(int).tolist()
    t_cam, _ = PD.fit_t_cam_obj(ds, trk, g, fit_frames, cube, cell)
    q_ref = A.norm_to_q(
        {m: float(v) for m, v in zip(A.MOTORS, ds[(ig + ir) // 2]["observation.state"]
                                     .numpy().astype(float))}, g["urdf"])
    Xc, delta, _ = PD.handeye_delta(t_tool, t_cam, g, q_ref)

    # 桌面静态位姿：XY 来自 up 观测，Z = 橡皮半厚（贴桌），朝向取夹取那一刻
    found = find_rest_px(ds, g, i0 + 2, ig - 2)
    if found is None:
        raise RuntimeError(f"ep{ep}: 桌面期找不到方块")
    rest_px, n_hit, n_tot = found
    xy = rest_base_xy(ds, g, rest_px, i0 + 2, min(ig - 2, i0 + 140))
    if xy is None:
        raise RuntimeError(f"ep{ep}: 桌面方块位置反投影失败")

    q_close = A.norm_to_q(
        {m: float(v) for m, v in zip(A.MOTORS, ds[ig]["observation.state"]
                                     .numpy().astype(float))}, g["urdf"])
    T_e_close, _ = PD.eraser_pose(q_close, g, t_tool)
    T_static = np.eye(4)
    T_static[:3, :3] = T_e_close[:3, :3]
    T_static[:3, 3] = [float(xy[0]), float(xy[1]), 0.006]

    # 落点：放下那一帧的 FK 位姿（兜底用），以及盒子里方块实际停住的位置（观测）
    q_rel = A.norm_to_q(
        {m: float(v) for m, v in zip(A.MOTORS, ds[i0 + i_rel]["observation.state"]
                                     .numpy().astype(float))}, g["urdf"])
    T_e_rel, _ = PD.eraser_pose(q_rel, g, t_tool)
    up_rel_px = BS.predict_px(g["K"], g["dist"], upc.T_base_cam, T_e_rel[:3, 3])

    found_bin = find_bin_px(ds, g, i0 + i_rel + 20, i0 + L)
    if found_bin is not None:
        bin_px, n_bin, n_tot_bin = found_bin
        _xy = px_to_base_xy(g, bin_px, h_m=0.03)
        bin_base_xy = np.array([_xy[0], _xy[1], 0.03])  # z 按盒底 ~3cm 估
        bin_note = f"盒内 {np.round(bin_px, 1)} ({n_bin}/{n_tot_bin}) → base {np.round(bin_base_xy * 1000, 1)}mm"
    else:
        bin_px, bin_base_xy = up_rel_px, None
        bin_note = "盒内位置未检出，用放开帧投影兜底"

    print(f"[ep{ep:02d}] 夹爪 rest={rest:.1f}/open={om:.1f} 合上+{i_close} 放开+{i_rel}  "
          f"t_tool={np.round(t_tool * 1000, 1)}mm  Δ={np.round(delta * 1000, 1)}mm  "
          f"桌面 {np.round(rest_px, 1)} ({n_hit}/{n_tot} 命中) → base {np.round(xy * 1000, 1)}mm  "
          f"{bin_note}  合用 {time.time() - t0:.0f}s")

    return dict(ep=ep, i0=i0, L=L, ig=ig, ir=ir, i_rel=i0 + i_rel,
                trk=trk, ut=ut, t_tool=t_tool, Xc=Xc, cell=cell,
                T_static=T_static, T_e_rel=T_e_rel, rest_px=rest_px,
                rest_base=np.asarray(xy, float), CUBE_C_H=0.01075,
                up_rel_px=up_rel_px, bin_px=bin_px, bin_base_xy=bin_base_xy,
                # 静态相位的参考帧（手臂在原点、桌面干净）：用来找「这一帧才有的
                # 夹爪像素」—— 它和参考帧的差是唯一可靠的夹爪线索，颜色区分不了
                # （手指 V≈215、桌面 V≈217）。
                up_ref=BS.up_bgr(ds, i0),
                note=note, t_info={k: float(v) for k, v in t_info.items()})


# ================================================================ 逐帧：三个相位

def _up_fore(up, cm, bm_u):
    """up 图 L2：亮白/暖白的手指（和 `dual_frame` 里那条完全一致的判据）。"""
    hsv = PD.cv2.cvtColor(up, PD.cv2.COLOR_BGR2HSV)
    kk = PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (121, 121))
    near = PD.cv2.dilate(((cm > 0) | bm_u).astype(np.uint8), kk) > 0
    white = (((hsv[:, :, 1] <= 20) & (hsv[:, :, 2] >= 165)) |
             ((hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 180))) & near
    seed = PD.cv2.dilate((white & (hsv[:, :, 1] <= 20)).astype(np.uint8),
                         np.ones((9, 9), np.uint8)) > 0
    _, lab = PD.cv2.connectedComponents(white.astype(np.uint8), 8)
    keep = np.unique(lab[seed])
    keep = keep[keep > 0]
    if len(keep):
        white = np.isin(lab, keep)
    k = PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (5, 5))
    return PD.cv2.dilate(white.astype(np.uint8), k) > 0


def static_frame(S, up, fg, T_base_cam, fg_obs=None):
    """夹取前：桌面静态位姿，两视角各渲一次。

    `fg_obs`：腕图里**实测到的方块掩码**（`_wrist_cube_px` 选的连通域）。给了就
    直接拿它当腕视角的擦除掩码 —— 比再按投影位置搜一遍准得多（贴近时标定档的
    黄色掩码会把阴影里的桌面整片圈进来）。
    """
    mesh, cell = S["mesh"], S["cell"]
    cam_f = MR.Cam(S["K"], S["dist"], T_base_cam)
    # ---- up 视角：相机固定、橡皮躺在固定位姿 —— 整段静态相位渲出来的东西一模一样，
    #      只在第一帧渲一次缓存起来（渲一次要几百毫秒，占每帧开销的大头）。
    #      腕视角的相机是随手臂动的，每帧都得重渲。
    ras_f, bgr_f = PD.render(mesh, cell, S["T_static"], cam_f, S["light"], S["mat"],
                             PW.ERASER_ALBEDO, True)
    bm_f = ras_f["mask"]

    cache = S.get("_up_cache")
    if cache is None:
        ras_u, bgr_u = PD.render(mesh, cell, S["T_static"], S["upc"], S["light"],
                                 S["mat"], PW.ERASER_ALBEDO, True)
        bm_u = ras_u["mask"]
        # up 的擦除掩码：标定档 ∪ 宽松档（都限制在桌面位置附近）。
        # 标定档只认亮黄顶面；方块的**背光侧面是 H26/S122/V144 的暗橄榄色**，
        # 标定档圈不到，于是橡皮盖住顶面、侧面那一条还留在外面 —— 看着就是
        # 「方块上抹了一块」。宽松档（S>=45、V>=55）正好把这一条圈进来，
        # 桌面阴影是灰的（S 低）不会误入。
        # 整个静态相位只用第一帧量一次：相机固定、方块也没动。逐帧重算的话，夹爪
        # 一压上来掩码就塌成一条缝 —— 橡皮「补满」的范围跟着缩，看上去像橡皮自己
        # 在方块上忽大忽小，方块被压暗的那半边还会漏出来。
        cm = _near_components(BS.yellow_mask(up, "up"), S["rest_px"], 42.0, 200, 6000)
        lo = _near_components(_loose_mask(up), S["rest_px"], 42.0, 200, 6000)
        if cm is None:
            cm = lo
        elif lo is not None:
            cm = PD.cv2.bitwise_or(cm, lo)
        if cm is None:                   # 最后兜底：方块位置画个圆盘（宁可多擦一点桌面）
            cm = np.zeros(up.shape[:2], np.uint8)
            PD.cv2.circle(cm, (int(round(S["rest_px"][0])), int(round(S["rest_px"][1]))),
                          22, 255, -1)
        cm = np.ascontiguousarray(cm)
        obs_u = cm > 0

        # 把渲染剪影平移到**实测方块**的中心上：方块顶面在 21.5mm 高、橡皮躺在 6mm
        # 高，视差让两者的投影中心差 4~7px；橡皮只有 22mm 宽（方块 21.5mm），偏一点
        # 就在窄边露一条 —— 于是「盖不满」。平移量只算一次（相机和方块都不动）。
        sh = np.zeros(2, int)
        if obs_u.any() and bm_u.any():
            cy, cx = np.nonzero(obs_u); my, mx = np.nonzero(bm_u)
            sh = np.round([cx.mean() - mx.mean(), cy.mean() - my.mean()]).astype(int)
        if sh.any():
            M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
            h, w = up.shape[:2]
            bm_u = PD.cv2.warpAffine(bm_u.astype(np.uint8), M, (w, h),
                                     flags=PD.cv2.INTER_NEAREST) > 0
            bgr_u = PD.cv2.warpAffine(bgr_u, M, (w, h))

        # 静态阶段：橡皮就躺在方块原位、以方块为中心向两边展开（40mm 长轴 vs 方块
        # 21.5mm）。补满时把静态掩码胀 7px：方块近侧那条亮边会在橡皮外沿露一线
        # （橡皮 22mm 宽只比方块 21.5mm 宽 0.5mm），胀一下连抗锯齿边一起吃掉。
        _r = 7
        need_u = PD.cv2.dilate(obs_u.astype(np.uint8),
                               PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE,
                                                            (_r, _r))) > 0
        bm_u_vis, bgr_u_vis = PD.cover_obj(bm_u, bm_u, need_u, bgr_u)
        # 用户 2026-10-22：橡皮**上下两边**多出来的部分减掉（横着的长轴照旧可以
        # 超出去），左右一点不剪。按静态方块掩码的 y 范围裁 —— 掩码是第一帧量的
        # 整块方块（含背光侧面），范围内一定盖得满；外扩 2px 保住抗锯齿边。
        _ku = PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (5, 5))
        bm_u_vis = PD.clip_height(bm_u_vis,
                                  PD.cv2.dilate(obs_u.astype(np.uint8), _ku) > 0)
        bm_u_vis = np.ascontiguousarray(bm_u_vis)
        bgr_u_vis = np.ascontiguousarray(bgr_u_vis)
        cache = dict(cm=cm, obs_u=obs_u, bm_u_vis=bm_u_vis, bgr_u_vis=bgr_u_vis)
        S["_up_cache"] = cache
    cm, obs_u = cache["cm"], cache["obs_u"]
    bm_u_vis, bgr_u_vis0 = cache["bm_u_vis"], cache["bgr_u_vis"]

    # ---- 光照：手臂靠近时整片桌面被压暗（实测桌面 V 从 217 掉到 125~155），橡皮
    #      不跟着暗就是一张发光的贴纸。在方块外圈量「当前帧 / 参考帧」的亮度比，
    #      乘到渲染出来的橡皮上。
    gain = 1.0
    ref = S.get("up_ref")
    if ref is not None:
        k_big = PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (41, 41))
        k_sml = PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (9, 9))
        ring = (PD.cv2.dilate(obs_u.astype(np.uint8), k_big) > 0) & \
               (PD.cv2.dilate(obs_u.astype(np.uint8), k_sml) == 0)
        if int(ring.sum()) > 100:
            g_now = float(np.median(PD.cv2.cvtColor(up, PD.cv2.COLOR_BGR2GRAY)[ring]))
            g_ref = float(np.median(PD.cv2.cvtColor(ref, PD.cv2.COLOR_BGR2GRAY)[ring]))
            if g_ref > 1.0:
                gain = float(np.clip(g_now / g_ref, 0.55, 1.10))
    if abs(gain - 1.0) > 0.02:
        bgr_u_vis = np.clip(bgr_u_vis0.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    else:
        bgr_u_vis = bgr_u_vis0

    # ---- 遮挡：只有**真盖在方块位置上**的东西才盖在橡皮前面。判据是「静态方块
    #      掩码里、这一帧却看不见黄块」 —— 那块地方只可能是夹爪/舵机/线缆压上来了。
    #      不能再拿「和参考帧差多少」当判据：手臂的影子比手指本身的差还大（实测
    #      影子 65~100、手指 35），按差分算会把影子当成遮挡，把橡皮左右整片还原掉
    #      （f150/f200 实测橡皮被削窄 20px，就是影子干的）。影子由上面的 `gain` 处理。
    vis = PD.cv2.bitwise_or(BS.yellow_mask(up, "up"), _loose_mask(up))
    vis = _near_components(vis, S["rest_px"], 42.0, 100, 12000)
    occ = PD.cv2.dilate(obs_u.astype(np.uint8),
                        PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (5, 5))) > 0
    if vis is not None:
        occ &= PD.cv2.dilate(vis, PD.cv2.getStructuringElement(
            PD.cv2.MORPH_ELLIPSE, (9, 9))) == 0
    # 用户 2026-10-22：**左右两边除了遮挡一律不剪** —— 就算方块被夹爪挡住大半，
    # 橡皮的长轴也照旧伸到方块脚印外面，只把遮挡那部分还原。
    # 黑夹爪/黑排线（V<=70）横在橡皮伸出去的那截前面时，颜色判据够硬，单独补一条。
    # 方块脚印**外面**那截橡皮（40mm 长轴伸出来的两头）没有遮挡判据，但手指往往正
    # 好贴在那里。用手指比周围亮这个局部对比来抓：手掌/手指是光滑塑料，比被压暗的
    # 桌面亮 40~60 灰阶；桌面是大片平的，局部中值跟它自己差不多，不会亮起来。
    # 绝对阈值没用 —— 手指 V≈178 反而比没打光的参考桌面(217)暗。
    hsv_u = PD.cv2.cvtColor(up, PD.cv2.COLOR_BGR2HSV)
    _V = hsv_u[:, :, 2]
    _med = PD.cv2.medianBlur(_V, 31).astype(np.int16)
    _dv = _V.astype(np.int16) - _med
    blob = ((_dv > 28) | (_dv < -40)).astype(np.uint8)
    blob = PD.cv2.morphologyEx(
        blob, PD.cv2.MORPH_OPEN,
        PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (3, 3))) > 0
    n_bl, lab_bl, st_bl, _ = PD.cv2.connectedComponentsWithStats(blob.astype(np.uint8), 8)
    if n_bl > 1:
        keep_bl = [j for j in range(1, n_bl) if st_bl[j, PD.cv2.CC_STAT_AREA] >= 60]
        blob = np.isin(lab_bl, keep_bl)
    else:
        blob = np.zeros_like(blob)
    # 黄块自己的亮面/暗面也是「局部亮/暗块」，剔掉
    if vis is not None:
        blob &= PD.cv2.dilate(vis, PD.cv2.getStructuringElement(
            PD.cv2.MORPH_ELLIPSE, (5, 5))) == 0
    fore_u = (occ | blob | (_V <= 70)) & bm_u_vis
    # 遮挡边界是逐像素判出来的，直接拿去剪橡皮会是一排锯齿（f230/f255 右边那道）。
    # 开运算削掉 1px 台阶、闭运算补上 2px 缺口，再把两边对齐。
    _k3 = PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (3, 3))
    _k5 = PD.cv2.getStructuringElement(PD.cv2.MORPH_ELLIPSE, (5, 5))
    _dr = PD.cv2.morphologyEx((bm_u_vis & ~fore_u).astype(np.uint8),
                              PD.cv2.MORPH_OPEN, _k3) > 0
    _dr = PD.cv2.morphologyEx(_dr.astype(np.uint8), PD.cv2.MORPH_CLOSE, _k5) > 0
    _dr &= bm_u_vis
    fore_u = bm_u_vis & ~_dr
    # 和橡皮主体断开、面积很小的碎块直接丢掉（手指边缘没被判成遮挡时留下的橙渣）。
    n_fr, lab_fr, st_fr, _ = PD.cv2.connectedComponentsWithStats(
        (bm_u_vis & ~fore_u).astype(np.uint8), 8)
    if n_fr > 2:
        _a = st_fr[1:, PD.cv2.CC_STAT_AREA]
        _drop = [j + 1 for j, a in enumerate(_a) if a < max(60, 0.15 * _a.max())]
        if _drop and len(_drop) < n_fr - 1:
            bm_u_vis = bm_u_vis & ~np.isin(lab_fr, _drop)
            fore_u = fore_u & bm_u_vis
    out_u, _, _ = PD.compose_up(up, cm, bm_u_vis, bgr_u_vis, fore_u)

    # front 的物体掩码：优先用实测的那一块；没有就按投影位置找黄色连通域
    c_f = BS.predict_px(S["K"], S["dist"], T_base_cam, S["T_static"][:3, 3])
    obj = fg_obs
    if obj is None:
        obj = _near_components(BS.yellow_mask(fg, "wrist"), c_f, 70.0, 150, 12000)
    if obj is None:
        obj = _near_components(_loose_mask(fg), c_f, 70.0, 150, 12000)
    if obj is None:
        obj = np.zeros(fg.shape[:2], np.uint8)
    _, front, _ = P6.front_mask_observational(
        fg, obj, lambda b: BS.yellow_mask(b, "wrist"))
    out_f, _, _ = PD.compose_front(fg, obj, bm_f, bgr_f, front)
    return out_u, out_f


def recolor_frame(up, fg, S, T_base_cam):
    """放下之后：方块已经进盒子，位姿不可观测 —— 只按橡皮色相重映射盒子里的黄。

    up 视角用**盒内静止位置**（`find_bin_px` 的观测）；腕视角用它的 base 系 XY 逐帧
    投影（腕相机在动），半径给得宽一点，因为 z 是估的（方块躺在盒底，按 3cm 算）。
    """
    if S.get("bin_base_xy") is not None:
        c_f = BS.predict_px(S["K"], S["dist"], T_base_cam, S["bin_base_xy"])
        r_f = 110.0
    else:
        c_f = BS.predict_px(S["K"], S["dist"], T_base_cam, S["T_e_rel"][:3, 3])
        r_f = 170.0
    out = []
    for bgr, cam, centre, rad in ((up, "up", S["bin_px"], 45.0),
                                  (fg, "wrist", c_f, r_f)):
        m = PD.cv2.bitwise_or(BS.yellow_mask(bgr, cam), _loose_mask(bgr))
        m = _near_components(m, centre, rad, 120, 8000)
        img = bgr
        if m is not None:
            hsv = PD.cv2.cvtColor(img, PD.cv2.COLOR_BGR2HSV)
            sel = m > 0
            hsv[sel, 0] = RECOLOR_H
            hsv[sel, 1] = np.clip(hsv[sel, 1].astype(np.float32) * RECOLOR_S,
                                  0, 255).astype(np.uint8)
            img = PD.cv2.cvtColor(hsv, PD.cv2.COLOR_HSV2BGR)
        out.append(img)
    return out[0], out[1]


# ================================================================ 一集：渲染 + 写 shard

def process_episode(S):
    """在子进程里跑：逐帧渲染，写一个单集 shard 数据集。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager

    ds = S["ds"]
    i0, L = S["i0"], S["L"]
    ig, ir, i_cut = S["ig"], S["ir"], S["i_rel"] + 12
    upc, g, Xc = S["upc"], S["g"], S["Xc"]
    mesh, cell, light, mat = S["mesh"], S["cell"], S["light"], S["mat"]

    out = LeRobotDataset.create(
        S["shard_repo"], fps=S["fps"], features=S["features"], root=S["shard_root"],
        robot_type=S["robot_type"], use_videos=True)

    def tbase_cam_at(st):
        q = A.norm_to_q({m: float(v) for m, v in zip(
            A.MOTORS, np.asarray(st, float))}, g["urdf"])
        return g["urdf"].fk(q, "wrist_camera_link") @ Xc

    n_static = n_carry = n_post = 0
    t0 = time.time()
    with VideoEncodingManager(out):
        for k in range(L):
            i = i0 + k
            fr = ds[i]
            up = BS.up_bgr(ds, i)
            fg = _bgr_of(fr[FRONT_KEY])
            st = fr["observation.state"].numpy()
            T_base_cam = tbase_cam_at(st)
            if i < ig:
                c_obs, _, a_obs, m_obs = _wrist_cube_px(S, fg, T_base_cam)
                if c_obs is not None:
                    T_base_cam = _wrist_cam_corrected(S, T_base_cam, c_obs, a_obs)
                out_u, out_f = static_frame(S, up, fg, T_base_cam, m_obs)
                n_static += 1
            elif i < i_cut:
                r = PD.dual_frame(ds, i, g, upc, S["t_tool"], Xc, mesh,
                                  S["cube"], cell, light, mat, S["trk"], S["ut"],
                                  smooth=True, verbose=False)
                if r is None:                       # 跟踪断了：这一帧退回颜色替换
                    out_u, out_f = recolor_frame(up, fg, S, T_base_cam)
                    n_post += 1
                else:
                    out_u, out_f = r["out_u"], r["out_f"]
                    n_carry += 1
            else:
                out_u, out_f = recolor_frame(up, fg, S, T_base_cam)
                n_post += 1
            # 数据集写的是 RGB；整条渲染链在 BGR（OpenCV）里跑
            out.add_frame({
                "observation.state": fr["observation.state"],
                "action": fr["action"],
                "task": S["task"],
                UP_KEY: np.ascontiguousarray(out_u[:, :, ::-1]),
                FRONT_KEY: np.ascontiguousarray(out_f[:, :, ::-1]),
            })
            if (k + 1) % 50 == 0:
                print(f"  [ep{S['ep']:02d}] {k + 1}/{L}  {time.time() - t0:.0f}s", flush=True)
        out.save_episode()
    # done.json 最后写：中途崩掉的 shard 下次会被整个删掉重来
    (Path(S["shard_root"]) / "done.json").write_text(json.dumps(
        {"episode": S["ep"], "frames": L, "static": n_static,
         "carry": n_carry, "recolor": n_post}, indent=2))
    print(f"[ep{S['ep']:02d}] 写完 {L} 帧  static {n_static} / carry {n_carry} / recolor {n_post}"
          f"  {time.time() - t0:.0f}s", flush=True)
    return dict(ep=S["ep"], frames=L, static=n_static, carry=n_carry, recolor=n_post,
                seconds=round(time.time() - t0, 1))


# ================================================================ 主流程

def _base_state(ds, path):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(path, "data", "*", "*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "action",
                                                "episode_index", "frame_index"]) for f in files])
    df = df.sort_values(["episode_index", "frame_index"])
    out = {}
    for e, x in df.groupby("episode_index"):
        out[int(e)] = np.stack(x["observation.state"].values).astype(float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="/data1/lerobot/dataset/newdata_yellow50")
    ap.add_argument("--src-repo", default="newdata_yellow50")
    ap.add_argument("--dst-root", default="/data1/lerobot/dataset/newdata_eraser50")
    ap.add_argument("--dst-repo", default="newdata_eraser50")
    ap.add_argument("--tmp", default="/data1/lerobot/dataset/_eraser_shards")
    ap.add_argument("--episodes", nargs="+", default=["all"])
    ap.add_argument("--task", default="Pick up eraser and put it in trash can")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-fail", action="store_true",
                    help="某集失败只记下来，不中断其它集")
    ap.add_argument("--eraser-mesh", default="eraser_40x22x12_r3n80",
                    help="calib/recon/ 下的网格名（或绝对路径）。"
                         "SAM3D 重建出来的橡皮用 eraser_sam3d_40x22x12")
    ap.add_argument("--eraser-albedo", default="",
                    help="BGR 三通道反照率，逗号分隔，覆盖 PW.ERASER_ALBEDO。"
                         "黄色橡皮用 0.10,0.78,0.90")
    ap.add_argument("--probe", action="store_true", help="只渲几张对比图")
    ap.add_argument("--probe-frames", default="")
    ap.add_argument("--probe-out", default=str(HERE / "figures"))
    ap.add_argument("--render", action="store_true", help="渲染所有 shard")
    ap.add_argument("--aggregate", action="store_true", help="把 shard 拼成最终数据集")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    sys.path.insert(0, str(WORKPLACE / "object_generalization" / "scripts"))
    from _cache_io import episode_lengths, parse_episodes  # noqa: E402

    lengths = episode_lengths(a.src_root)
    eps = parse_episodes(a.episodes[0] if len(a.episodes) == 1 else a.episodes,
                         len(lengths))
    ds = LeRobotDataset(a.src_repo, root=a.src_root, video_backend="pyav")
    ep_tbl = ds.meta.episodes.to_pandas()
    from_idx = {int(r.episode_index): int(r.dataset_from_index) for r in ep_tbl.itertuples()}
    states = _base_state(ds, a.src_root)

    if a.probe:
        os.makedirs(a.probe_out, exist_ok=True)
        for ep in eps:
            S = _setup_common(ds, a, ep, from_idx, lengths, states)
            frames = [int(x) for x in a.probe_frames.split(",") if x.strip()]
            _probe(S, frames, Path(a.probe_out))
        return 0

    if a.render:
        os.makedirs(a.tmp, exist_ok=True)
        # dst_root 别提前建：aggregate 要求它不存在（防止覆盖已有数据集）
        jobs = [ep for ep in eps if not _shard_done(a.tmp, ep)]
        print(f"待渲染 {len(jobs)}/{len(eps)} 集（已完成 {len(eps) - len(jobs)}）→ {a.tmp}")
        if jobs:
            ctx = mp.get_context("fork")
            with ctx.Pool(min(a.workers, len(jobs))) as pool:
                for r in pool.imap_unordered(_worker_entry, [a.__dict__ | {"ep": ep}
                                                             for ep in jobs]):
                    if r is not None:
                        print(f"[完成] {r}", flush=True)
        _write_shard_index(a.tmp, eps)
        return 0

    if a.aggregate:
        _aggregate(a, eps)
        return 0

    ap.print_help()
    return 0


def _worker_entry(job):
    t = time.time()
    try:
        return _worker_inner(job)
    except Exception:
        print(f"[ep{job['ep']:02d}] 失败：\n{traceback.format_exc()}", flush=True)
        if job.get("allow_fail"):
            return None
        raise


def _worker_inner(job):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ep = job["ep"]
    # create 不接受已存在的目录：上一次崩在这里的残留先清掉
    shard = Path(job["tmp"]) / f"ep{ep:03d}"
    if shard.exists() and not (shard / "done.json").exists():
        shutil.rmtree(shard)
    ds = LeRobotDataset(job["src_repo"], root=job["src_root"], video_backend="pyav")
    lengths = _ep_lengths(job["src_root"])
    ep_tbl = ds.meta.episodes.to_pandas()
    from_idx = {int(r.episode_index): int(r.dataset_from_index) for r in ep_tbl.itertuples()}
    states = _base_state(ds, job["src_root"])
    S = _setup_common(ds, argparse.Namespace(**job), ep, from_idx, lengths, states)
    return process_episode(S)


_EPLEN_CACHE = {}


def _ep_lengths(root):
    if root not in _EPLEN_CACHE:
        sys.path.insert(0, str(WORKPLACE / "object_generalization" / "scripts"))
        from _cache_io import episode_lengths
        _EPLEN_CACHE[root] = episode_lengths(root)
    return _EPLEN_CACHE[root]


def _setup_common(ds, a, ep, from_idx, lengths, states):
    g = BS.load_geom()
    upc = PD.PU.load_up_cam()
    light = P6.load_lighting()
    cell = 0.0005
    if getattr(a, "eraser_albedo", ""):
        PW.ERASER_ALBEDO = np.array([float(x) for x in a.eraser_albedo.split(",")], float)
        globals()["RECOLOR_H"] = _hue_of(PW.ERASER_ALBEDO)
        print(f"[橡皮] albedo(BGR)={np.round(PW.ERASER_ALBEDO, 3).tolist()} "
              f"-> 重映射色相 H={RECOLOR_H}")
    mesh = F.load_mesh(getattr(a, "eraser_mesh", "eraser_40x22x12_r3n80"),
                       cells=(2 * cell, cell, cell / 2))
    mat = MR.Light(direction=light.dir, ambient=PW.ERASER_AMBIENT,
                   diffuse=PW.ERASER_DIFFUSE, specular=PW.ERASER_SPECULAR,
                   shininess=PW.ERASER_SHININESS)
    i0 = from_idx[ep]
    S = setup_episode(ds, g, upc, ep, i0, lengths[ep], states[ep])
    S.update(ds=ds, g=g, upc=upc, light=light, mat=mat, mesh=mesh,
             cube=F.MeshObj(MR.box(0.0215, 0.0215, 0.0215), name="cube21.5",
                            cells=(2 * cell, cell, cell / 2)))
    S["features"] = _clean_features(ds.meta.features)
    S["fps"] = ds.fps
    S["robot_type"] = ds.meta.info.get("robot_type")
    S["task"] = getattr(a, "task", "Pick up eraser and put it in trash can")
    S["src_repo"], S["src_root"] = a.src_repo, a.src_root
    S["tmp"] = getattr(a, "tmp", "/data1/lerobot/dataset/_eraser_shards")
    S["shard_root"] = str(Path(S["tmp"]) / f"ep{ep:03d}")
    S["shard_repo"] = f"{getattr(a, 'dst_repo', 'newdata_eraser50')}_ep{ep:03d}"
    S["K"], S["dist"] = g["K"], g["dist"]
    S["up_rel_px"] = BS.predict_px(g["K"], g["dist"], upc.T_base_cam,
                                   S["T_e_rel"][:3, 3])
    return S


def _probe(S, frames, out_dir):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    ds, i0, L = S["ds"], S["i0"], S["L"]
    rows = []
    for k in frames:
        k = int(np.clip(k, 0, L - 1))
        i = i0 + k
        q = A.norm_to_q({m: float(v) for m, v in zip(
            A.MOTORS, ds[i]["observation.state"].numpy().astype(float))}, S["g"]["urdf"])
        T_base_cam = S["g"]["urdf"].fk(q, "wrist_camera_link") @ S["Xc"]
        up = BS.up_bgr(ds, i)
        fg = _bgr_of(ds[i][FRONT_KEY])
        if i < S["ig"]:
            phase = "static"
            c_obs, p_mod, a_obs, m_obs = _wrist_cube_px(S, fg, T_base_cam)
            if c_obs is not None:
                T_base_cam = _wrist_cam_corrected(S, T_base_cam, c_obs, a_obs)
            print(f"    f{k} wrist obs={None if c_obs is None else np.round(c_obs, 1)}"
                  f" a={a_obs:.0f} model={np.round(p_mod, 1)}")
            out_u, out_f = static_frame(S, up, fg, T_base_cam, m_obs)
        elif i < S["i_rel"] + 12:
            phase = "carry"
            r = PD.dual_frame(ds, i, S["g"], S["upc"], S["t_tool"], S["Xc"],
                              S["mesh"], S["cube"], S["cell"], S["light"], S["mat"],
                              S["trk"], S["ut"], smooth=True, verbose=False)
            if r is None:
                out_u, out_f = recolor_frame(up, fg, S, T_base_cam)
                phase = "carry(recolor)"
            else:
                out_u, out_f = r["out_u"], r["out_f"]
        else:
            phase = "recolor"
            out_u, out_f = recolor_frame(up, fg, S, T_base_cam)
        rows.append((k, phase, up, out_u, fg, out_f))
        print(f"  ep{S['ep']:02d} f{k} [{phase}] Δup={np.abs(out_u.astype(int) - up).mean():.2f}"
              f" Δfront={np.abs(out_f.astype(int) - fg).mean():.2f}")
    _probe_sheet(rows, out_dir / f"eraser_probe_ep{S['ep']:02d}.png")


def _probe_sheet(rows, path):
    panels = []
    for k, phase, up, out_u, fg, out_f in rows:
        for img, tag in ((up, f"up f{k} orig"), (out_u, f"up f{k} {phase}"),
                         (fg, f"front f{k} orig"), (out_f, f"front f{k} {phase}")):
            im = MR.put_lines(img.copy(), [tag], org=(8, 22), scale=0.5)
            panels.append(PD.cv2.resize(im, (320, 240)))
    per_row = 4
    rows_img = [np.hstack(panels[j:j + per_row]) for j in range(0, len(panels), per_row)]
    PD.cv2.imwrite(str(path), np.vstack(rows_img))
    print(f"[探针] {path}")


def _shard_done(tmp, ep):
    d = Path(tmp) / f"ep{ep:03d}"
    return (d / "done.json").exists() and (d / "meta" / "info.json").exists()


def _write_shard_index(tmp, eps):
    for ep in eps:
        d = Path(tmp) / f"ep{ep:03d}"
        if (d / "meta" / "info.json").exists():
            (d / "done.json").write_text(json.dumps({"episode": ep}, indent=2))


def _aggregate(a, eps):
    from lerobot.datasets.aggregate import aggregate_datasets
    roots, repos, keep = [], [], []
    for ep in eps:
        d = Path(a.tmp) / f"ep{ep:03d}"
        # 必须认 done.json：只有 info.json 的目录可能正在写（episodes/tasks.parquet
        # 还没落盘），aggregate 读 meta 时会当成「本地没有」去连 huggingface 拉取，
        # 断网机器上直接卡到超时。
        if (d / "done.json").exists():
            roots.append(d)
            repos.append(f"{a.dst_repo}_ep{ep:03d}")
            keep.append(ep)
    if Path(a.dst_root).exists():
        raise SystemExit(f"输出目录已存在：{a.dst_root}（先挪走或删掉）")
    print(f"合并 {len(roots)} 个 shard → {a.dst_root}")
    aggregate_datasets(repos, a.dst_repo, roots=roots, aggr_root=Path(a.dst_root))
    print(f"完成：{a.dst_root}（episodes {keep[0]}..{keep[-1]}）")


if __name__ == "__main__":
    raise SystemExit(main())
