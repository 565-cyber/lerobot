#!/usr/bin/env python
"""腕视图：把**任意重建网格**按拟合出来的 6D 位姿渲进真帧（Pose6DAug 的腕视图那一半）。

和 `pose6d_swap.py` 的关系：
  · `pose6d_swap.py --up` 是 **up 视图**那一路，注入物是**解析球**（`MR.uv_sphere`，
    半径来自卡尺，一个像素都没参与）——**那条路一个字都没动**，继续当对照臂。
  · 这个是**腕视图**这一路，注入物是 `calib/recon/` 里的**重建网格**，
    位姿由 `fit_mesh_pose.fit_wrist` 从腕相机的真掩码里**拟合**出来。
    「球的网格是先验解析的；舵机的网格是重建的」—— 这句要写进产出文档。

L0/L1/L2 三层的合成核心**与形状无关**（纯掩码写 + 像素拷回），所以整段照抄
`pose6d_swap.wrist_compose`；真正换掉的只有三处：
  ① 位姿来源：`place_ball`（半径反解深度）→ `fit_wrist`（面积定深 + 三级模式搜索）
  ② 渲染：球用细网格 `uv_sphere` → 网格用它自己的**最细一级减面**
  ③ 报的量：球专属的 `r_px` → 剪影**面积、IoU、单边性**

用法：
    PY=/home/zjq/miniconda3/envs/lerobot/bin/python
    $PY -I workplace/calibration/pose6d_wrist_fit.py --frames 300,336,400
    $PY -I workplace/calibration/pose6d_wrist_fit.py --mesh cube21.5 --frames 300,336,400
"""
import argparse
import importlib.util
import json
import sys
import time
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


MR = _load("_pwf_mr", HERE / "meshrender.py")
F = _load("_pwf_fit", HERE / "fit_mesh_pose.py")
P6 = _load("_pwf_p6", HERE / "pose6d_swap.py")
BS = P6.BS
A = P6.A

# 舵机的塑料材质：光照方向沿用腕视图实测那支，镜面项换成塑料的（不是方块、也不是球）。
# ★★ **albedo 是 BGR 不是 RGB**（`MR.shade` 的约定，和 `BALL_ALBEDO=(0.18,0.24,0.88)`
#    那个「偏红」的注释同源）。踩过：这里一开始按 RGB 写成 (0.72,0.30,0.22)，
#    渲出来的舵机是**深蓝色** —— 还以为是重建网格的法线反了，查了半天。
SERVO_ALBEDO = np.array([0.20, 0.28, 0.66])      # BGR：舵机壳那种暗砖红
SERVO_SPECULAR, SERVO_SHININESS = 0.35, 40.0
CUBE_ALBEDO = np.array([0.28, 0.72, 0.90])       # BGR：方块黄（和真方块同色，便于比掩码）
# BGR。**用户 2026-10-21 把橡皮换成了黄色**，颜色是当场从相机读的，不是拍脑袋：
#   camera_A(up) 整块中位 BGR=(42,100,194) / HSV=(11,199,194)；亮顶面 (71,122,215)、暗侧棱 (14,61,126)
#   camera_B(front) 中位 BGR=(65,146,216) / HSV=(16,182,216)（离得近、光更足，整体更亮一点）
# 两边一致 ⇒ 就是一块饱和的橙黄。反推 albedo 时除以 `ambient+diffuse·lam`（≈0.62~1.20）；
# 第一次取 [0.24,0.46,0.80] 渲出来是 (60,113,193)、S≈0.69，比实物「偏橙、不够黄」，
# 所以按两台相机观测的均值 (53,121,204) 再除一次渲染值，落到下面这个：
# 渲出来中位 ≈(53,121,204)，HSV≈(H12,S0.74) —— 对上 up 的 (11,199) 和 front 的 (16,182)。
# （旧的暖白 [0.82,0.86,0.93] 和深暖灰 [0.34,0.36,0.40] 都作废，那是照 `calib/eraser_on_table.png`
#  那块白橡皮配的，实物已经不是它了。）
ERASER_ALBEDO = np.array([0.21, 0.49, 0.85])
# 橡皮是哑光橡胶/塑料，不是舵机那种亮塑料：镜面项压很低，别在顶面洗出白斑
ERASER_SPECULAR, ERASER_SHININESS = 0.06, 8.0
# 实测光（wrist_lighting.json）是给黄色方块拟合的。橡皮的明暗比实测约 1.7
# （顶面 V≈215 / 侧棱 V≈126），所以用「低 ambient + 高 diffuse」把动态范围留在
# 0.62~1.2 倍这一段，暗面不会被 ambient 抬平。
ERASER_AMBIENT, ERASER_DIFFUSE = 0.62, 0.58

# 真实夹取朝向（夹爪系）。列 = 物体局部 x/y/z 在夹爪系里的方向：
#   local x (40mm 长) → -夹爪 y（沿指长）
#   local y (22mm 宽) → +夹爪 x（跨在两指之间）
#   local z (12mm 厚) → +夹爪 z（接近方向，≈竖直向下）
# 这是绕夹爪 z 的 +90° 旋转（det=+1）。
GRASP_R_OFFSET = np.array([[0.0, -1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]], float)


def mesh_material(lighting, specular=SERVO_SPECULAR, shininess=SERVO_SHININESS):
    return MR.Light(direction=lighting.dir, ambient=lighting.ambient,
                    diffuse=lighting.diffuse, specular=specular,
                    shininess=shininess)


_VN_CACHE = {}


def vertex_normals(tris, tol=1e-5, crease_deg=15.0):
    """**按折痕限角**的顶点法线（每个角一个法线，尺寸 (F,3,3)）。

    重建网格是 marching tets 出来的：每个体素沿固定的体对角线切、相邻体素按奇偶
    换向，所以**水密**（`boundary_edges=0`）但近平面区域上留着一层**对角棱纹** ——
    逐面着色时法线沿对角线交替，渲出来是一条条斜带（`raw` 那 232428 面比 1.5mm
    减面还明显，说明**棱纹在几何里，不在减面里**）。换成插值顶点法线就能把它抹掉，
    **几何一个像素都不动**（剪影还是那个剪影）。

    ★★ 但**不能无条件平均**：无条件平均会把舵机中段那条亮带糊成**锯齿**（踩过，
       `/tmp/_smooth_cmp.png` 右边那张），所以每个角只在「和本面夹角 < `crease_deg`」
       的那些面里平均。面积权重用 `|n|`（= 2×面积）。

    `crease_deg` 实测扫过 {8,12,16,22,35}（`/tmp/_sweep.png`）：**抹掉棱纹所需的角度
    很低**，12° 以上就已经干净，再放宽只是把中段那条带子也一起抹平。而那条亮带在
    **所有**角度下都会淡成一个斑驳的浅痕 ⇒ 它本身也是被棱纹放大的，不是一道真棱
    （两面的夹角 < 8°）。取 15° 是「够抹棱纹、尽量不多动」那一档。"""

    F = len(tris)
    q = np.round(np.asarray(tris).reshape(-1, 3) / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    # numpy 2.0 起 `return_inverse` 的形状跟输入走，ravel 一下两种版本都稳
    inv = np.asarray(inv).ravel().reshape(F, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])   # |n| = 2×面积
    area = np.linalg.norm(n, axis=1)
    fn = n / np.maximum(area[:, None], 1e-12)
    cos_t = float(np.cos(np.deg2rad(crease_deg)))

    cn = np.empty((F, 3, 3), np.float64)
    flat_v = inv.ravel()
    flat_f = np.repeat(np.arange(F), 3)
    order = np.argsort(flat_v, kind="stable")
    sv, sf = flat_v[order], flat_f[order]
    edges = np.searchsorted(sv, np.arange(int(inv.max()) + 2))
    for v in range(len(edges) - 1):
        lo, hi = edges[v], edges[v + 1]
        if hi <= lo:
            continue
        fs = np.unique(sf[lo:hi])                     # 共享该顶点的面
        fa, fna = area[fs], fn[fs]
        for f in fs:
            for k in np.nonzero(inv[f] == v)[0]:
                m = fna @ fn[f] > cos_t
                s = (fa[m, None] * fna[m]).sum(0)
                cn[f, k] = s / max(np.linalg.norm(s), 1e-12)
    return inv, cn.astype(np.float32)


def smooth_normals(ras, tris, cam, cn):
    """把光栅化出的**面**法线换成按重心插值的**角**法线。`ras["tri"]` 已经是 `tris`
    的下标（`rasterize` 里 `src` 映回原三角形），所以插值只要按像素重心来。"""
    tid = ras["tri"]
    ys, xs = np.nonzero(ras["mask"] & (tid >= 0))
    if not len(xs):
        return ras
    tp = tid[ys, xs]
    T = np.asarray(tris, np.float64)[tp]
    uv, _ = cv2.projectPoints(T.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                              cam.K, cam.dist)
    uv = uv.reshape(-1, 3, 2)
    ax, ay = uv[:, 0, 0], uv[:, 0, 1]
    bx, by = uv[:, 1, 0], uv[:, 1, 1]
    cx, cy = uv[:, 2, 0], uv[:, 2, 1]
    px, py = xs + 0.5, ys + 0.5
    d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / d
    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / d
    W = np.clip(np.stack([w0, w1, 1.0 - w0 - w1], 1), 0, None)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    nrb = ras["normal"].copy()
    nrb[ys, xs] = (W[:, :, None] * cn[tp]).sum(1)
    out = dict(ras)
    out["normal"] = nrb
    return out


def smooth_for(mesh, cell, tris):
    """按（网格名，减面尺度）缓存角法线。"""
    key = (mesh.name, float(cell))
    if key not in _VN_CACHE:
        _VN_CACHE[key] = vertex_normals(tris)[1]
    return _VN_CACHE[key]


def wrist_fit_frame(ds, i, g, trk, mesh, x_cube=None, lighting=None, mat=None,
                    albedo=SERVO_ALBEDO, fix_s=1.0, cell=None, seed=None,
                    smooth=True, verbose=True, flat_grasp=False,
                    rot_range_deg=6.0):
    """一帧：真掩码 → `fit_wrist` 位姿 → 网格渲进图。返回 dict（数字 + 供出图的图）。

    **观测、相机、跟踪器全部走 `pose6d_swap.wrist_compose` 那条路**（`BS.carry_track`
    + `P6.object_mask` + FK⊗手眼 δ），一个字都不另起 —— 否则两条路的差异会混进
    「球 vs 网格」的比较里。
    """
    fr = ds[i]
    fg = P6.bgr_of(fr["observation.images.front"])
    rec = trk.get(i)
    if rec is None or rec.get("c") is None:
        return None
    c = np.asarray(rec["c"], float)

    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    T0 = P6.cam_from_state(g, q)
    delta = np.zeros(3)
    fk_err = None
    if x_cube is not None:
        P = (g["urdf"].fk(q, "gripper_frame_link") @ np.append(x_cube, 1.0))[:3]
        uv, _ = MR.Cam(g["K"], g["dist"], T0).project(P.reshape(1, 3))
        fk_err = float(np.linalg.norm(uv[0] - c))
        delta, _ = BS.solve_cam_delta(g["K"], g["dist"], T0, [(P, c)])
    cam = MR.Cam(g["K"], g["dist"], P6.cam_from_state(g, q, delta))

    # 观测掩码：跟踪器自带的搬运档阈值（别用桌面档的 yellow_mask，见 object_mask 的注释）
    obj, obj_src = P6.object_mask(rec, fg, c)
    obs = (obj > 0)

    # ---- L1 的位姿：**这一行就是和球那条路的唯一实质差别** ----
    # 橡皮/长条物：按真实夹取方式锁住朝向（12mm 厚度沿夹爪接近方向，22mm 跨指，
    # 40mm 沿指长），只解平移。自由 6-DoF 的等价竖立解在剪影上同样好，但不合理。
    R_fixed = None
    if flat_grasp:
        T_gr = g["urdf"].fk(q, "gripper_frame_link")
        R_fixed = T_gr[:3, :3] @ GRASP_R_OFFSET
    t0 = time.time()
    T, info = F.fit_wrist(mesh, cam, obs.astype(np.float32), fix_s=fix_s,
                          cell_fine=cell, verbose=False,
                          seed=i if seed is None else seed,
                          R_fixed=R_fixed,
                          rot_range=np.deg2rad(rot_range_deg))
    t_fit = time.time() - t0

    if lighting is None:
        lighting = P6.load_lighting()
    if mat is None:
        if mesh.name.startswith("eraser"):
            mat = MR.Light(direction=lighting.dir, ambient=ERASER_AMBIENT,
                           diffuse=ERASER_DIFFUSE, specular=ERASER_SPECULAR,
                           shininess=ERASER_SHININESS)
        else:
            mat = mesh_material(lighting)
    cell_used = cell or mesh.cells[-1]
    tris_b = MR.transform(mesh.level(cell_used), T)
    ras = MR.rasterize(tris_b, cam)
    if smooth:
        ras = smooth_normals(ras, tris_b, cam, smooth_for(mesh, cell_used, tris_b))
    rgb = MR.shade(ras, cam, mat, albedo)

    # ---- L0 底板：抹物体 + 网格那块，再背景修补 ----
    cm = cv2.bitwise_or(obj, (ras["mask"].astype(np.uint8) * 255))
    plate = BS.erase_cube(fg.copy(), cm, grow=3, shadow=True, fill="diffuse")[0]

    # ---- L2 前景：观测路 + 几何路（几何路拿网格的栅格当「物体在哪」）----
    hull, front_obs, cnt = P6.front_mask_observational(fg, obj, P6._carry_yellow)
    arm_ras, front_geo, front_raw = P6.front_mask_geometric(ds, i, g, cam, ras, obj)
    front = front_obs | front_geo

    # ---- 合成（远 → 近）----
    out = plate.copy()
    out[ras["mask"]] = (rgb * 255)[ras["mask"]]
    l1 = out.copy()
    out[front] = fg[front]

    # 客观量：拟合剪影 vs 观测剪影。★ 单边性是 visual hull 的**理论保证**，
    # 这里量的是「位姿拟合这一步」的偏差，不是重建那一步的。
    cov = info["cov"]
    inter = float((cov * obs).sum())
    uni = float(((cov > 0) | obs).sum())
    res = dict(frame=i, c=c, obj_src=obj_src, obj_px=int(obs.sum()),
               mesh_px=int(ras["mask"].sum()), cov_px=int((cov > 0).sum()),
               iou=inter / max(uni, 1e-9), front_obs=int(front_obs.sum()),
               front_geo=int(front_geo.sum()), geo_raw=int(front_raw.sum()),
               front=int(front.sum()), plate_px=int((plate != fg).any(2).sum()) + 0,
               depth_mm=info["depth_mm"], fk_err_px=fk_err,
               delta_mm=(delta * 1000).round(1).tolist(), t_fit=t_fit,
               loss=info["loss"], s=info["s"],
               # 单边性：拟合剪影里**观测外面**那一块占总边界长度的比例
               extra_px=int(((cov > 0.5) & ~obs).sum()),
               miss_px=int((~(cov > 0.5) & obs).sum()))
    if verbose:
        print(f"  f{i}: 观测 {res['obj_px']}px({obj_src}) → 网格 {res['mesh_px']}px  "
              f"IoU {res['iou']:.4f}  深度 {res['depth_mm']:.1f}mm  "
              f"多出 {res['extra_px']}px / 漏 {res['miss_px']}px  "
              f"δ={np.round(delta*1000,1)}mm(FK 原差 "
              f"{'—' if fk_err is None else f'{fk_err:.0f}px'})  "
              f"拟合 {t_fit:.0f}s")
    return dict(out=out, l1=l1, fg=fg, plate=plate, ras=ras, rgb=rgb, arm_ras=arm_ras,
                obj=obj, front_mask=front, T=T, info=info, mesh=mesh, **res)


def figure(frames, rows, out_path, title=""):
    """6 联图，论文体例：一层一栏，肉眼能核对每层对不对（和 `pose6d_swap.figure` 同构）。"""
    panels = []
    for r in rows:
        if r is None:
            continue
        m = r["ras"]["mask"]
        ov = r["l1"].copy()
        cv2.drawContours(ov, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                         (0, 255, 255), 1)
        d = r["out"].astype(np.int16) - r["fg"].astype(np.int16)
        dv = np.clip(np.abs(d).sum(2) * 2, 0, 255).astype(np.uint8)
        dv = cv2.applyColorMap(dv, cv2.COLORMAP_INFERNO)
        # 文字全用 ASCII：cv2 的 Hershey 字体不认中文（踩过）
        row = [MR.put_lines(r["fg"], [f"original f{r['frame']}"], org=(8, 22), scale=0.5),
               MR.put_lines(r["plate"], ["L0  erase object + inpaint bg"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(ov, [f"L1  {r['mesh'].name}  d={r['depth_mm']:.0f}mm  "
                                 f"area={r['mesh_px']}px"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(MR.overlay(r["out"], r["front_mask"], (0, 255, 0), alpha=0.0),
                            [f"L2  robot fg {r['front']}px (obs {r['front_obs']} / "
                             f"geom {r['front_geo']} of raw {r['geo_raw']})"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(r["out"], [f"composite  silhouette IoU={r['iou']:.3f}  "
                                       f"extra {r['extra_px']}px / miss {r['miss_px']}px"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(dv, ["|diff| vs original"], org=(8, 22), scale=0.5)]
        panels.append(np.hstack(row))
    img = np.vstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[图] {out_path}  — {title}")
    return img


def figure_compare(pairs, out_path, title=""):
    """对照图：**上行=球那条路，下行=网格这条路**，同一帧同一掩码，只有注入物不同。"""
    rows = []
    for a, b in pairs:
        if a is None or b is None:
            continue
        m = b["ras"]["mask"]
        ov = b["l1"].copy()
        cv2.drawContours(ov, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                         (0, 255, 255), 1)
        cm = a["l1"].copy()
        ma = a["ball_ras"]["mask"]
        cv2.drawContours(cm, cv2.findContours(ma.astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                         (0, 255, 255), 1)
        f = b["frame"]
        rows.append(np.hstack([
            MR.put_lines(b["fg"], [f"original f{f}"], org=(8, 22), scale=0.5),
            MR.put_lines(cm, [f"ball  r={a['r_px']:.0f}px  area={int(ma.sum())}px"],
                         org=(8, 22), scale=0.5),
            MR.put_lines(a["out"], ["composite (ball)"], org=(8, 22), scale=0.5),
            MR.put_lines(ov, [f"mesh {b['mesh'].name}  area={b['mesh_px']}px  "
                              f"IoU={b['iou']:.3f}"], org=(8, 22), scale=0.5),
            MR.put_lines(b["out"], ["composite (mesh)"], org=(8, 22), scale=0.5)]))
    img = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[图] {out_path}  — {title}")
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--video-backend", default=None,
                    help="torchcodec / pyav；torchcodec 的 FFmpeg 库坏了时用 pyav")
    ap.add_argument("--frames", default="300,336,400")
    ap.add_argument("--mesh", default="sts3215_03a_v1_v50_h0.5_m1",
                    help="calib/recon/ 下的重建网格名（或 .stl 路径）；"
                         "`cube21.5` 走解析方盒，用来和球/参考位姿对照")
    ap.add_argument("--radius-mm", type=float, default=12.0,
                    help="球那条路（对照臂）的半径")
    ap.add_argument("--cube-h-mm", type=float, default=21.5)
    ap.add_argument("--grasp", type=int, default=246)
    ap.add_argument("--release", type=int, default=409)
    ap.add_argument("--free-s", action="store_true", help="尺度也当参数解（默认钉死 CAD 尺度）")
    ap.add_argument("--albedo", type=float, nargs=3, default=None,
                    metavar=("B", "G", "R"),
                    help="注入物颜色（BGR，0~1）；默认 cube→黄、eraser→白、其他→舵机")
    ap.add_argument("--cell-mm", type=float, default=None,
                    help="最终拟合/渲染的减面格子（mm）。圆角小物体建议 0.5；"
                         "默认沿用网格自带的三级格子（最细 1.5mm，圆角会出现台阶）")
    ap.add_argument("--grasp-mode", choices=("free", "flat"), default=None,
                    help="free=自由 6-DoF 剪影拟合（旧行为）；flat=按真实夹取锁朝向，"
                         "只解平移（橡皮默认 flat）")
    ap.add_argument("--rot-range-deg", type=float, default=6.0,
                    help="flat 模式下仍允许的朝向修正角（度，默认 6）")
    ap.add_argument("--no-ball", action="store_true", help="不出对照臂那张图")
    ap.add_argument("--flat", action="store_true",
                    help="关掉顶点法线插值，回到逐面着色（棱纹会明显；用来对照）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    frames = [int(x) for x in a.frames.split(",") if x.strip()]
    g = BS.load_geom()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(a.dataset, root=str(REPO / "dataset" / a.dataset),
                        video_backend=a.video_backend)
    trk = BS.carry_track(ds, a.grasp, a.release)
    print(f"[跟踪] 夹持段 f{a.grasp}~f{a.release} 共 {len(trk)} 帧有腕图方块观测")

    if a.mesh.startswith("cube"):
        e = float(a.mesh.replace("cube", "")) / 1000.0
        mesh = F.MeshObj(MR.box(e, e, e), name=a.mesh, cells=(2e-3, 1e-3, 0.5e-3))
    else:
        cells = (4e-3, 3e-3, 1.5e-3)
        if a.cell_mm is not None:
            fine = a.cell_mm / 1000.0
            cells = (max(2e-3, 2 * fine), max(1e-3, fine), fine)
        mesh = F.load_mesh(a.mesh, cells=cells)
    cell_fit = a.cell_mm / 1000.0 if a.cell_mm is not None else None
    grasp = a.grasp_mode or ("flat" if a.mesh.startswith("eraser") else "free")
    print(f"位姿模式：{grasp}" + ("（锁朝向，只解平移）" if grasp == "flat" else
                                  "（自由 6-DoF）"))
    if a.albedo is not None:
        albedo = np.asarray(a.albedo, float)
    elif a.mesh.startswith("cube"):
        albedo = CUBE_ALBEDO
    elif a.mesh.startswith("eraser"):
        albedo = ERASER_ALBEDO
    else:
        albedo = SERVO_ALBEDO
    print(f"注入网格：{mesh.name}  尺寸 "
          f"{np.round(np.ptp(mesh.tris0.reshape(-1, 3), axis=0) * 1000, 1)}mm（长×宽×高），"
          f"面数 {len(mesh.tris0)}")

    # ★ `x_cube` 只在**夹取帧**上算一次（见 wrist_compose 的注释：搬运期一帧一算会解出
    #   300mm 的假偏移）。跟 `pose6d_swap.main` 用**同一个**锚点和**同一条** up 掩码路，
    #   这样两条路的相机 δ 完全一致，差异只剩「注入物」。
    X_cube, note = BS.estimate_x_tcp_cube(
        ds, a.grasp, g, BS.UP_CAM_H_MM / 1000.0, a.cube_h_mm / 1000.0,
        BS.cube_mask_fns(g, "up", "yellow")[0])
    print(f"[常量] X_tcp_cube = {np.round(np.asarray(X_cube) * 1000, 2)}mm  ({note})")

    rows = []
    for i in frames:
        for k in range(a.stride):
            j = i + k
            if j >= len(ds):
                break
            r = wrist_fit_frame(ds, j, g, trk, mesh, x_cube=X_cube,
                                fix_s=None if a.free_s else 1.0, albedo=albedo,
                                cell=cell_fit, smooth=not a.flat,
                                flat_grasp=(grasp == "flat"),
                                rot_range_deg=a.rot_range_deg)
            if r is not None:
                rows.append(r)

    if not rows:
        print("没有一帧拟合成功")
        return 1
    io = np.array([r["iou"] for r in rows])
    dp = np.array([r["depth_mm"] for r in rows])
    print(f"\n汇总 {len(rows)} 帧：剪影 IoU 中位 {np.median(io):.4f} 最小 {io.min():.4f}"
          f"｜深度 {dp.min():.0f}~{dp.max():.0f}mm"
          f"｜平均 {np.mean([r['t_fit'] for r in rows]):.0f}s/帧")

    n = len(rows)
    tag = mesh.name.replace("/", "_")
    out = Path(a.out) if a.out else FIG / f"pose6d_wrist_fit_{tag}.png"
    figure(frames, rows, out, title=f"腕视图：重建网格 {mesh.name} 按拟合位姿渲回真帧")

    if not a.no_ball:
        print("\n对照臂：球那条路（pose6d_swap.wrist_compose，一个字没改）")
        pairs = []
        for r in rows:
            b = P6.wrist_compose(ds, r["frame"], g, trk, a.radius_mm / 1000.0,
                                 a.cube_h_mm / 1000.0, x_cube=X_cube)
            pairs.append((b, r))
        figure_compare(pairs, FIG / f"pose6d_wrist_fit_vs_ball_{tag}.png",
                       title="同一帧同一掩码，只有注入物不同（上：球 / 下：网格）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
