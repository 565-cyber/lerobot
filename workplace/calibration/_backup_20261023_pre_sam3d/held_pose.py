#!/usr/bin/env python
"""Stage 2 —— 方块在**腕相机系**里的 6D 位姿（一个常量），以及「重渲回原样」的验收数字。

## 为什么是「一个常量」

夹持期方块被夹爪刚性夹住，腕相机长在同一个手腕上 ⇒ 方块在相机系里**不动**。
这一条可以被观测检验（下面会逐帧报 IoU）：如果哪一段 IoU 明显掉下去，那就是真滑动了，
或者「手指把方块切裂、形心被带偏」。夹持初期 f246→f292 图像上确实动了 ~48px。

## 为什么这条路不需要手眼

位姿直接**拟合到观测的剪影**上：相机 K 已标定（`front_intrinsics.json`），方块边长
实测 21.5mm ⇒ 度量可解。FK⊗手眼只用来给一个初值（真的只当初值用，见 `--no-fk-init`）。
FK 在搬运深度（~90mm）上错 ~150px，所以拿它当结果是不行的 —— 但拿它当起点可以。

## 验收（这就是「含金量」的硬证据）

把方块网格按拟合出的位姿**渲回原帧**，跟真实像素比：
  · 剪影 IoU（几何对不对）
  · mask 内 mean|ΔBGR|（颜色/光照对不对 —— 光照还是占位值，Stage 4 才拟合）
两张图 + 两组数字，不再靠形容词。

用法：
    python -I workplace/calibration/held_pose.py --fit
    python -I workplace/calibration/held_pose.py --figure
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
REPO = HERE.parent.parent
FIG = HERE / "figures"
CAL = HERE / "calib"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load_module("_hp_arm", HERE / "arm.py")
BS = _load_module("_hp_bs", HERE / "ball_swap.py")
MR = _load_module("_hp_mr", HERE / "meshrender.py")
P6 = _load_module("_hp_p6", HERE / "pose6d_swap.py")

CUBE_MM = 21.5                     # 用户实测边长
LIGHT_PLACEHOLDER = dict(direction=(0.35, -0.45, 0.82), ambient=0.42,
                         diffuse=0.62, specular=0.28, shininess=24.0)


# ------------------------------------------------------------------ 相机 / 观测

def cam_at(ds, g, dmodel, i):
    """第 i 帧的腕相机（FK⊗手眼 + **模型预测的** δ）。

    δ 用模型预测而不是逐帧从观测解（`BS.solve_cam_delta`）：后者要拿「方块在
    FK 下的位置」当已知点，而那个位置正是这里要拟合的东西 —— 会循环论证。
    模型 δ 在 newdata1 上留出中位 4.44mm，够用；它的误差会落进拟合残差里如实报出。
    """
    fr = ds[i]
    st = fr["observation.state"].numpy().astype(float)
    norm = {m: float(v) for m, v in zip(A.MOTORS, st)}
    q = A.norm_to_q(norm, g["urdf"])
    T0 = g["urdf"].fk(q, "wrist_camera_link") @ g["X"]
    delta = BS.delta_from_model(dmodel, norm) if dmodel is not None else np.zeros(3)
    T = T0.copy()
    T[:3, 3] += np.asarray(delta, float).reshape(3)              # 模型已给米
    return MR.Cam(g["K"], g["dist"], T), q, T


def fk_cube_pose(g, q, x_cube):
    """FK⊗手眼给出的方块位姿初值：位置 = FK(TCP)·x_cube，姿态 = TCP 系的姿态。"""
    T = g["urdf"].fk(q, BS.TCP_LINK).copy()
    T[:3, 3] = (g["urdf"].fk(q, BS.TCP_LINK) @ np.append(x_cube, 1.0))[:3]
    return T


def obs_of(ds, i, trk):
    fg = P6.bgr_of(ds[i]["observation.images.front"])
    rec = trk.get(i)
    if rec is None or rec.get("c") is None:
        return None
    c = np.asarray(rec["c"], float)
    m, src = P6.object_mask(rec, fg, c)
    return dict(fg=fg, c=c, mask=(m > 0), src=src, rec=rec)


# ------------------------------------------------------------------ 位姿参数化

def _exp_so3(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def pose_of(p, T_init):
    """p = [rotvec(3, rad), trans(3, m), log_scale(1)] → (T_cam_cube, 边长倍率)。

    扰动是**左乘**在相机系里做的：T = ΔT · T_init。这样 rotvec 的三轴就是相机的
    x/y/z，解释起来直接（绕光轴转 = 剪影原地转，最不可观测的那一维）。
    """
    T = np.eye(4)
    T[:3, :3] = _exp_so3(p[:3])
    T[:3, 3] = p[3:6]
    return T @ T_init, float(np.exp(p[6]))


def render_cube(cam, T_init, p, edge_m, mesh_flat):
    """T_init 在**相机系**里；光栅化要 base 系的三角形，所以左乘 T_base_cam。"""
    T, s = pose_of(p, T_init)
    tris = MR.transform(mesh_flat * s, cam.T_base_cam @ T)
    return MR.rasterize(tris, cam, cull=False), s


# ------------------------------------------------------------------ 拟合

def dt_of(mask):
    """~mask 的距离变换：mask 内为 0，外面是到 mask 的距离（px）。"""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def make_cost(samples, edge_m, mesh_flat):
    """对称 DT 代价：渲染跑到观测外、观测没被渲染盖住，两边都罚，各自按到对方的距离。

    用距离变换而不是对称差面积 —— 面积是像素量化的阶梯函数，Powell 会在平台上卡住；
    DT 给出长程梯度，而且**亚像素**：轮廓差 1px 和差 5px 的代价差 5 倍。
    """
    def cost(p):
        tot = 0.0
        for s in samples:
            ras, _ = render_cube(s["cam"], s["T_init"], p, edge_m, mesh_flat)
            m = ras["mask"]
            out = m & ~s["mask"]
            inn = s["mask"] & ~m
            v = 0.0
            if out.any():
                v += float(s["dt"][out].sum())
            if inn.any():
                v += float(dt_of(m)[inn].sum())
            tot += v
        return tot / max(len(samples), 1)
    return cost


def iou(a, b):
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


# ------------------------------------------------------------------ 出图（IoU 曲线）

# 配色取 dataviz 参考调色板的光模式值：单序列 ⇒ 分类槽 1（蓝），没有图例，
# 阈值线用中性 muted、滑移区用极浅中性 —— 都是**注记**，不是第二条序列。
INK1, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURF, GRID, AXIS, SERIES1 = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#2a78d6"


def plot_iou(per_frame, out_path, fit_frames=()):
    """剪影 IoU 随帧变化。**单序列**：一条线 + 一条阈值 + 一段注记带，没有图例。

    这张图要回答的问题只有一个：那个「常量位姿」到底在哪些帧成立。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({
        "font.family": [f.name for f in font_manager.fontManager.ttflist
                        if f.name == "Noto Sans CJK JP"] or ["DejaVu Sans"],
        "axes.unicode_minus": False, "figure.dpi": 140,
        "savefig.facecolor": SURF, "axes.facecolor": SURF, "figure.facecolor": SURF,
    })
    fr = np.array([r["frame"] for r in per_frame], float)
    io = np.array([r["iou"] for r in per_frame], float)
    med = float(np.median(io))

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    # 滑移/未稳定段：连续 IoU<0.5 的那一段（注记，用极浅中性，不是第二条序列）
    low = io < 0.5
    if low.any():
        k = 0
        while k < len(io) and low[k]:
            k += 1
        ax.axvspan(fr[0] - 3, fr[max(k - 1, 0)] + 3, color="#f0efec", zorder=0)
        ax.annotate(f"初期未稳定（IoU<0.5，{int(low.sum())}/{len(io)} 帧）",
                    xy=(fr[max(k - 1, 0)], 0.06), xytext=(fr[0] + 6, 0.10),
                    color=INK2, fontsize=9.5, va="bottom")
        ax.axvline(fr[max(k - 1, 0)] + 3, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)

    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.annotate("IoU 0.5", xy=(fr[-1], 0.5), xytext=(-4, 4),
                textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    ax.plot(fr, io, color=SERIES1, lw=2.0, solid_capstyle="round", zorder=3)
    ax.axhline(med, color=SERIES1, lw=1.0, alpha=0.35, zorder=2)
    ax.annotate(f"中位 {med:.3f}", xy=(fr[-1], med), xytext=(-4, 5),
                textcoords="offset points", ha="right", color=INK2, fontsize=9.5)
    if len(fit_frames):
        ff = np.array(sorted(fit_frames), float)
        ax.plot(ff, np.interp(ff, fr, io), "o", ms=3.4, color=SERIES1,
                markeredgecolor=SURF, markeredgewidth=0.8, zorder=4)

    ax.set_xlim(fr[0] - 3, fr[-1] + 3)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("帧号（newdata1 · ep0 · 夹持段）", color=INK2, fontsize=10)
    ax.set_ylabel("剪影 IoU（方块网格 vs 观测掩码）", color=INK2, fontsize=10)
    ax.set_title("常量 6D 位姿的适用范围：稳定段 IoU≈0.87，抓取初期掉到 0.25",
                 color=INK1, fontsize=12.5, pad=12, loc="left")
    ax.tick_params(colors=MUTED, labelsize=9.5, length=3)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK2)
    fig.text(0.005, 0.008,
             "点 = 参与拟合的帧；同一套常量位姿（相机系）渲到全部 163 帧。"
             "初期偏低 = 方块在夹爪里滑动 / 被手指切裂带偏形心。",
             color=MUTED, fontsize=8.2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ 主流程

def build(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    g = BS.load_geom()
    ds = LeRobotDataset(args.dataset, root=str(REPO / "dataset" / args.dataset))
    dmodel = BS.load_delta_model(Path(args.delta_model)) if args.delta_model else None
    trk = BS.carry_track(ds, args.grasp, args.release)
    x_cube, note = BS.estimate_x_tcp_cube(
        ds, args.grasp, g, BS.UP_CAM_H_MM / 1000.0, CUBE_MM / 1000.0,
        BS.cube_mask_fns(g, "up", "yellow")[0])
    print(f"[常量] X_tcp_cube = {np.round(np.asarray(x_cube) * 1000, 2)}mm  ({note})")
    print(f"[跟踪] 夹持段 {len(trk)} 帧")
    cam, q, T = cam_at(ds, g, dmodel, args.grasp)
    T_base_cube = fk_cube_pose(g, q, x_cube)
    T_init = np.linalg.inv(T) @ T_base_cube
    print(f"[初值] FK⊗手眼给的 T_cam_cube：平移 "
          f"{np.round(T_init[:3, 3] * 1000, 1)}mm")
    return g, ds, trk, dmodel, x_cube, T_init


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--grasp", type=int, default=246)
    ap.add_argument("--release", type=int, default=409)
    ap.add_argument("--delta-model", default=str(CAL / "wrist_delta_model.json"))
    ap.add_argument("--fit-frames", default="300,305,310,315,320,325,330,335,340,345,"
                                           "350,355,360,365,370,375,380,385,390,395,400,405")
    ap.add_argument("--fit", action="store_true", help="拟合 + 落盘 calib/held_pose_wrist.json")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--frames", default="246,300,336,400")
    ap.add_argument("--out", default=str(FIG / "pose6d_cube_refit.png"))
    a = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    g, ds, trk, dmodel, x_cube, T_init = build(a)
    mesh_flat = MR.box(CUBE_MM / 1000.0, CUBE_MM / 1000.0, CUBE_MM / 1000.0).reshape(-1, 3, 3)
    p_out = Path(a.out)

    if not (a.fit or a.figure):
        a.fit = a.figure = True

    # ---------------- 拟合 ----------------
    if a.fit:
        fit_ids = [int(x) for x in a.fit_frames.split(",") if x.strip()]
        samples = []
        for i in fit_ids:
            o = obs_of(ds, i, trk)
            if o is None:
                continue
            samples.append(dict(i=i, cam=cam_at(ds, g, dmodel, i)[0],
                                T_init=T_init, mask=o["mask"], dt=dt_of(o["mask"])))
        print(f"[拟合] {len(samples)} 帧（f{fit_ids[0]}~f{fit_ids[-1]}）")

        from scipy.optimize import minimize
        cost = make_cost(samples, CUBE_MM / 1000.0, mesh_flat)
        p0 = np.zeros(7)
        print(f"[初始代价] {cost(p0):.2f}px/帧")
        res = minimize(cost, p0, method="Powell",
                       bounds=[(-0.5, 0.5)] * 3 + [(-0.03, 0.03)] * 3 + [(-0.3, 0.3)],
                       options=dict(maxiter=4000, xtol=1e-4, ftol=1e-4))
        print(f"[拟合完] 代价 {res.fun:.2f}px/帧（降 {(1 - res.fun / max(cost(p0), 1e-9)):.0%}）")
        T_fit, s_fit = pose_of(res.x, T_init)
        print(f"  旋转扰动 {np.round(np.degrees(res.x[:3]), 2)}°  "
              f"平移扰动 {np.round(res.x[3:6] * 1000, 2)}mm  "
              f"边长 {CUBE_MM * s_fit:.2f}mm")
        print(f"  T_cam_cube 平移 {np.round(T_fit[:3, 3] * 1000, 2)}mm")

        # 逐帧 IoU（含没参与拟合的帧：初期滑动会在这里露出来）
        rows = []
        for i in sorted(trk):
            o = obs_of(ds, i, trk)
            if o is None:
                continue
            cam = cam_at(ds, g, dmodel, i)[0]
            ras, _ = render_cube(cam, T_init, res.x, CUBE_MM / 1000.0, mesh_flat)
            rows.append(dict(frame=i, iou=iou(ras["mask"], o["mask"]),
                             px=int(o["mask"].sum()),
                             obs_px=int(ras["mask"].sum()), src=o["src"]))
        v = np.array([r["iou"] for r in rows], float)
        print(f"\n[逐帧 IaU] {len(rows)} 帧：中位 {np.median(v):.3f}  "
              f"min {v.min():.3f}  max {v.max():.3f}")
        lo = [r for r in rows if r["iou"] < 0.5]
        print(f"  IoU<0.5 的帧 {len(lo)}/{len(rows)}"
              + ("：" + "  ".join(f"f{r['frame']}={r['iou']:.2f}"
                                 for r in lo[:12]) if lo else " ✔"))
        fit_ids_set = set(fit_ids)
        vf = np.array([r["iou"] for r in rows if r["frame"] in fit_ids_set], float)
        print(f"  只看参与拟合的 {len(vf)} 帧：中位 {np.median(vf):.3f} min {vf.min():.3f}")

        # 颜色/光照验收（几何之外）：mask 内 mean|ΔBGR|
        light = MR.Light(**LIGHT_PLACEHOLDER)
        gcol = []
        for i in [int(x) for x in a.frames.split(",") if x.strip()]:
            o = obs_of(ds, i, trk)
            if o is None:
                continue
            cam = cam_at(ds, g, dmodel, i)[0]
            ras, _ = render_cube(cam, T_init, res.x, CUBE_MM / 1000.0, mesh_flat)
            rgb = (MR.shade(ras, cam, light, (0.16, 0.72, 0.90)) * 255)
            common = ras["mask"] & o["mask"]
            if common.sum() < 100:
                continue
            d = np.abs(rgb[common] - o["fg"][common].astype(float)).mean()
            gcol.append(float(d))
        print(f"\n[验收·颜色] mask 交集内 mean|ΔBGR| 中位 {np.median(gcol):.1f}"
              "（光照是占位值，Stage 4 才拟合；这个数现在必然大）")

        out = dict(note="Stage 2：方块在腕相机系里的常量 6D 位姿（拟合到观测剪影）",
                   T_cam_cube=T_fit.tolist(),
                   cube_edge_mm=CUBE_MM * s_fit,
                   p_perturb=list(map(float, res.x)),
                   rot_deg=list(map(float, np.degrees(res.x[:3]))),
                   trans_mm=list(map(float, res.x[3:6] * 1000)),
                   fit_frames=fit_ids,
                   cost_px_per_frame=float(res.fun),
                   iou_median=float(np.median(v)), iou_min=float(v.min()),
                   iou_max=float(v.max()),
                   per_frame=[dict(frame=r["frame"], iou=round(r["iou"], 4))
                              for r in rows],
                   color_mean_abs_bgr_placeholder_light=float(np.median(gcol)),
                   caveat="δ 用模型预测（非逐帧解），避免循环论证；光照未拟合")
        CAL.mkdir(parents=True, exist_ok=True)
        (CAL / "held_pose_wrist.json").write_text(json.dumps(out, indent=2))
        print(f"→ {CAL / 'held_pose_wrist.json'}")
        p_plot = FIG / "pose6d_iou_vs_frame.png"
        plot_iou(out["per_frame"], p_plot, fit_frames=fit_ids)
        print(f"→ {p_plot}")

    # ---------------- 出图 ----------------
    if a.figure:
        j = json.loads((CAL / "held_pose_wrist.json").read_text())
        p_best = np.array(j["p_perturb"])
        light = MR.Light(**LIGHT_PLACEHOLDER)
        rows_img = []
        for i in [int(x) for x in a.frames.split(",") if x.strip()]:
            o = obs_of(ds, i, trk)
            if o is None:
                continue
            cam = cam_at(ds, g, dmodel, i)[0]
            ras, _ = render_cube(cam, T_init, p_best, CUBE_MM / 1000.0, mesh_flat)
            rgb = np.clip(MR.shade(ras, cam, light, (0.16, 0.72, 0.90)) * 255,
                          0, 255).astype(np.uint8)
            re_ren = o["fg"].copy()
            re_ren[ras["mask"]] = rgb[ras["mask"]]
            ov = o["fg"].copy()
            cv2.drawContours(ov, cv2.findContours(o["mask"].astype(np.uint8),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_NONE)[0],
                             -1, (0, 255, 0), 1)
            cv2.drawContours(ov, cv2.findContours(ras["mask"].astype(np.uint8),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_NONE)[0],
                             -1, (0, 0, 255), 1)
            d = np.abs(re_ren.astype(np.int16) - o["fg"].astype(np.int16))
            dv = cv2.applyColorMap(np.clip(d.sum(2), 0, 255).astype(np.uint8),
                                   cv2.COLORMAP_INFERNO)
            io = iou(ras["mask"], o["mask"])
            cells = [o["fg"], ov, re_ren, dv]
            lbl = [f"original f{i}", f"obs(green)/rend(red)  IoU={io:.3f}",
                   "mesh re-render (cube @ fitted pose)",
                   f"|delta| mean={d[o['mask']].mean():.1f}"]
            rows_img.append(np.hstack([MR.put_lines(c, [s], org=(8, 22), scale=0.55)
                                       for c, s in zip(cells, lbl)]))
            print(f"  f{i}: IoU={io:.3f}  重渲 mean|ΔBGR|={d[o['mask']].mean():.1f}")
        img = np.vstack(rows_img)
        cv2.imwrite(str(p_out), img)
        print(f"→ {p_out} {img.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
