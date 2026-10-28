#!/usr/bin/env python
"""Stage 4 —— 从**实测像素**拟合光照（Phong），而不是拿一个占位方向糊弄。

## 数据从哪来

Stage 2（`held_pose.py`）已经把方块在腕相机系里的 6D 位姿拟合出来了（剪影 IoU 中位
0.87）。有了位姿就能把方块的网格渲回原帧，**逐像素得到法线**；再拿同一像素在真实
视频里的颜色去比 —— 这就成了一组「法线 → 亮度」的观测。

## 为什么这组观测能定出光的方向

方块在相机系里是**常量位姿**，但相机随臂在整个场景里转。所以同一块面的法线在
**base 系**里逐帧不同 —— 一个方向的光照在不同帧里照到不同的面上，这个变化就把
光的方向（base 系，3 自由度里的 2 个）钉住了。这比只看桌面亮度强得多：桌面只有一个
法线，永远定不出方向，只能定出「有多亮」。

## 模型与自由度

跟 `meshrender.shade` 完全一致：
    color = albedo ⊙ (ambient + diffuse·max(0, n·l)) + specular·max(0, n·h)^shininess
把 (ambient, diffuse) 归一化成 `ambient = a, diffuse = 1−a`（否则 albedo 和光强
可以互相缩放，不可辨识）。于是：
  · 外层 4 个非线性参数：方位角 az、仰角 el、环境占比 a、高光指数 shininess
  · 内层是**线性**的：albedo(BGR 三通道) 与 specular 一起用最小二乘直接解
每换一组外层参数，法线/视线不用重算（它们与光照无关）—— 所以内层求解是毫秒级的。

## 内层必须带物理边界（2026-10-14 改）

无约束最小二乘会把**模型误差**灌进 specular：它解出 specular=11.01 —— 镜面反射率
大于 1 是不物理的。后果在**方块自己身上看不出来**（拟合残差 9.72 还很漂亮），可一旦把
同一支光外推到**红球**就露馅：`shade` 里镜面项是**三通道等量相加**，11×0.88¹⁹ ≈ 4.2
被 clip 成 1 ⇒ 球面上凡 `n·h>0.88` 的整片像素直接变成纯白，球成了「红圈 + 一大块白斑」
（`figures/pose6d_wrist_fit.png` 的 f336/f400 一眼就是）。

所以内层解改成带界的线性最小二乘：`albedo ∈ [0,1]`（漫反射率）、`specular ∈ [0,1]`
（镜面反射率）。这不是调参讨好观感，是把**物理上不可能的参数**挡在解空间外；
如果挡住之后残差明显变差，那本身就是「模型对不上」的证据，也照实报。

用法：
    python -I workplace/calibration/light_fit.py --fit
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIG = HERE / "figures"
CAL = HERE / "calib"
CUBE_MM = 21.5


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load_module("_lf_arm", HERE / "arm.py")
BS = _load_module("_lf_bs", HERE / "ball_swap.py")
MR = _load_module("_lf_mr", HERE / "meshrender.py")
P6 = _load_module("_lf_p6", HERE / "pose6d_swap.py")
HP = _load_module("_lf_hp", HERE / "held_pose.py")


def gather_samples(ds, g, dmodel, trk, T_init, p_best, frames, min_common=200):
    """逐帧渲方块 → 取「渲染 ∩ 观测(腐蚀 2px)」里的法线、视线、真实颜色。

    腐蚀是为了丢掉轮廓上的抗锯齿混色（那些像素一半是方块、一半是背景，模型描不出来）。
    """
    mesh_flat = MR.box(CUBE_MM / 1000.0, CUBE_MM / 1000.0, CUBE_MM / 1000.0).reshape(-1, 3, 3)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    N, V, C, fid = [], [], [], []
    for i in frames:
        o = HP.obs_of(ds, i, trk)
        if o is None:
            continue
        cam = HP.cam_at(ds, g, dmodel, i)[0]
        ras, _ = HP.render_cube(cam, T_init, p_best, CUBE_MM / 1000.0, mesh_flat)
        core = cv2.erode(o["mask"].astype(np.uint8), k) > 0
        sel = ras["mask"] & core
        if int(sel.sum()) < min_common:
            continue
        n = ras["normal"][sel]
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        ys, xs = np.nonzero(sel)
        Pc = np.stack([(xs - cam.K[0, 2]) / cam.K[0, 0],
                       (ys - cam.K[1, 2]) / cam.K[1, 1],
                       np.ones_like(xs, float)], 1)
        v = -((Pc / np.linalg.norm(Pc, axis=1, keepdims=True)) @ cam.T_base_cam[:3, :3].T)
        # ★ 法线按「朝向观察者」定向，和 shade 里一模一样；否则 max(0,n·l) 会算到反面
        n = n * np.where((n * v).sum(1, keepdims=True) < 0, -1.0, 1.0)
        N.append(n)
        V.append(v)
        C.append(o["fg"][sel].astype(np.float64) / 255.0)
        fid.append(np.full(len(xs), i))
    return (np.concatenate(N), np.concatenate(V), np.concatenate(C),
            np.concatenate(fid))


def dir_from_angles(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _design(b, sp):
    """线性模型的设计矩阵：未知量 = [albedo_B, albedo_G, albedo_R, specular]。

    每个「像素×通道」是一条方程：`pred[i,c] = albedo[c]*b[i] + specular*sp[i]`。
    行序取 i*3+c（和 `C.ravel()` 一致）。
    """
    n = len(b)
    M = np.zeros((3 * n, 4))
    M[0::3, 0] = b
    M[1::3, 1] = b
    M[2::3, 2] = b
    M[:, 3] = np.repeat(sp, 3)
    return M


def _inner(N, V, C, az, el, a, sh):
    l = dir_from_angles(az, el)
    b = a + (1 - a) * np.clip(N @ l, 0, None)
    h = V + l
    h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    sp = np.clip((N * h).sum(1), 0, None) ** sh
    M = _design(b, sp)
    # 带界最小二乘：未知量 = [albedo_B, albedo_G, albedo_R, specular]，全部 ∈ [0,1]。
    # 无约束 lsq 会解出 specular≈11 去吸收模型误差 —— 外推到红球就是一片纯白。
    sol = lsq_linear(M, C.ravel(), bounds=(0.0, 1.0),
                     lsq_solver="exact").x
    pred = (M @ sol).reshape(C.shape)
    return float(np.abs(pred - C).mean()), sol, pred


def fit_light(N, V, C, n_restart=40, seed=0):
    """外层 4 参数（az, el, a, shininess）非线性；内层 (albedo BGR, specular) 线性最小二乘。"""
    rng = np.random.default_rng(seed)
    best = None
    # 网格 + 随机多起点：亮度对方向是非凸的，单起点容易掉进「光在相机背后」那种解
    cands = [(az, el) for az in np.arange(0, 360, 30) for el in (-60, -30, 0, 30, 60)]
    cands += [tuple(rng.uniform([0, -80], [360, 80])) for _ in range(n_restart)]
    for az, el in cands:
        for a in (0.2, 0.35, 0.5, 0.65):
            for sh in (8.0, 24.0, 64.0):
                e, sol, pred = _inner(N, V, C, az, el, a, sh)
                if best is None or e < best[0]:
                    best = (e, az, el, a, sh, sol, pred)
    from scipy.optimize import minimize
    r = minimize(lambda x: _inner(N, V, C, *x)[0], np.array(best[1:5], float),
                 method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-7))
    if r.fun < best[0]:
        e, sol, pred = _inner(N, V, C, *r.x)
        best = (e, *r.x, sol, pred)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--grasp", type=int, default=246)
    ap.add_argument("--release", type=int, default=409)
    ap.add_argument("--frames", default=",".join(str(i) for i in range(298, 409, 3)))
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=60000)
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    g = BS.load_geom()
    ds = LeRobotDataset(a.dataset, root=str(REPO / "dataset" / a.dataset))
    dmodel = BS.load_delta_model(CAL / "wrist_delta_model.json")
    trk = BS.carry_track(ds, a.grasp, a.release)
    j = json.loads((CAL / "held_pose_wrist.json").read_text())
    p_best = np.array(j["p_perturb"])
    print(f"[位姿] 用 held_pose_wrist.json 的常量位姿：平移 "
          f"{np.round(np.array(j['T_cam_cube'])[:3, 3] * 1000, 2)}mm")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa
    x_cube, _ = BS.estimate_x_tcp_cube(
        ds, a.grasp, g, BS.UP_CAM_H_MM / 1000.0, CUBE_MM / 1000.0,
        BS.cube_mask_fns(g, "up", "yellow")[0])
    cam0, q0, T0 = HP.cam_at(ds, g, dmodel, a.grasp)
    T_init = np.linalg.inv(T0) @ HP.fk_cube_pose(g, q0, x_cube)

    frames = [int(x) for x in a.frames.split(",") if x.strip()]
    N, V, C, fid = gather_samples(ds, g, dmodel, trk, T_init, p_best, frames)
    # 抽稀：法线/颜色分布已经很密，60k 像素足够定 4 个参数，内层 lstsq 从 50ms 降到 10ms
    if len(N) > a.max_pixels:
        sub = np.random.default_rng(0).choice(len(N), a.max_pixels, replace=False)
        N, V, C, fid = N[sub], V[sub], C[sub], fid[sub]
    print(f"[样本] {len(N)} 个像素（抽稀后），来自 {len(set(fid.tolist()))} 帧")

    # 占位光照做对照（同一个像素集、同一个网格，只换光照参数）
    lp = HP.LIGHT_PLACEHOLDER
    L = np.asarray(lp["direction"], float)
    L = L / np.linalg.norm(L)
    lam = np.clip(N @ L, 0, None)
    hp = V + L
    hp = hp / np.maximum(np.linalg.norm(hp, axis=1, keepdims=True), 1e-12)
    sp = np.clip((N * hp).sum(1), 0, None) ** lp["shininess"]
    err_p = float(np.abs(np.clip(
        np.array([0.16, 0.72, 0.90]) * (lp["ambient"] + lp["diffuse"] * lam)[:, None]
        + lp["specular"] * sp[:, None], 0, 1) - C).mean() * 255)

    e, az, el, amb, sh, sol, pred = fit_light(N, V, C)
    l = dir_from_angles(az, el)
    alb = sol[:3]
    print(f"[拟合] 平均|ΔBGR| = {e * 255:.2f}   占位光照同像素集 = {err_p:.2f}"
          f"（改善 {(1 - e * 255 / err_p) * 100:.0f}%）")
    print(f"  光方向（base 系，指向光源）= {np.round(l, 4)}"
          f"  az={az % 360:.1f}° el={el:.1f}°")
    print(f"  ambient={amb:.3f}  diffuse={1-amb:.3f}  specular={sol[3]:.4f}  "
          f"shininess={sh:.1f}")
    print(f"  albedo(BGR, 0~1) = {np.round(alb, 4)}")

    out = dict(note="Stage 4：从「方块网格按 Stage 2 拟合位姿渲回原帧」的逐像素法线拟合 Phong",
               method="外层 4 参数 Nelder-Mead + 网格多起点；"
                      "内层 albedo/specular **带界** [0,1] 线性最小二乘（无约束会解出 specular=11，"
                      "外推到红球就是一片纯白；见模块 docstring）",
               direction_base=l.tolist(), az_deg=float(az % 360), el_deg=float(el),
               ambient=float(amb), diffuse=float(1 - amb),
               specular=float(sol[3]), shininess=float(sh),
               cube_albedo_bgr=alb.tolist(),
               mean_abs_bgr_fit=float(e * 255),
               mean_abs_bgr_placeholder=err_p,
               n_pixels=int(len(N)), n_frames=int(len(set(fid.tolist()))),
               frames=sorted(set(int(x) for x in fid.tolist())),
               caveat="只在「渲染∩观测(腐蚀2px)」的方块像素上拟合；绿/蓝通道弱激励，"
                      "球的红色分量靠同一支光外推",
               material_caveat="specular/shininess 是**方块**这一材质的镜面率，不是光的属性；"
                               "换物体（球）渲图时只继承 direction/ambient/diffuse，"
                               "镜面项必须换成新物体的 —— 直接照抄会在球上洗出纯白斑"
                               "（实测 27.8% 像素被 clip，见 pose6d_swap.BALL_SPECULAR）")
    (CAL / "wrist_lighting.json").write_text(json.dumps(out, indent=2))
    print(f"→ {CAL / 'wrist_lighting.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
