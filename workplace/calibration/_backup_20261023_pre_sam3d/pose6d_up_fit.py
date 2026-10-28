#!/usr/bin/env python
"""up 视图：把**任意重建网格**按拟合出来的位姿渲回真帧（对应球那条路的 `pose6d_up_l2`）。

和 `pose6d_wrist_fit.py` 是**一对**，共用 `pose6d_wrist_fit` 里的着色套件
（材质、限角顶点法线）和 `fit_mesh_pose.fit_up`：

    · 腕视图：真标定相机 ⇒ 普通剪影 6-DoF（`fit_wrist`）。
    · up 视图：**没有可用的平移**（旧数据集取景变过，单平面单应也只能把内参定到
      「焦距↔高度」的退化族里）⇒ 造一台**观测锚定的局部针孔**：
      主点 = 观测形心、焦距由「渲出剪影面积 = 观测面积」反解。
      **朝向**分两种：`--up-rot real`（默认）用 2026-10-19 标定的安装旋转
      `R_cam`（像面 x≈(0.161,−0.987,0.017)，不是 base x！），
      `vertical` 是旧的「垂直下视、像面 x 对齐 base x」假设。
      长条物必须用 real：零 roll 假设会把投影方向拧反（见 HANDOVER §6 的 2026-10-20 更正）。
      橡皮还和腕视图共用同一个 `R_fixed = R_gripper @ GRASP_R_OFFSET`，只解 x/y。

三层合成**照抄 `pose6d_swap.up_compose`**，一个字都不另起 —— 两条路的差异必须只剩
「注入物」，否则 `pose6d_up_fit_vs_ball_*.png` 那张对照图就没意义了。up 视图的 L2 语义
和腕视图不同，照抄它的理由：

    * **观测 = `BS.up_erase_mask`**（旧物体在 up 图里露出来的那一片）。它已经剔掉受光的
      手指、也抓不到手指的暗面 ⇒ 「新物体能画在哪」与「夹爪在哪挡着」是同一件事的两种
      说法，**不靠任何颜色判据**（颜色判据两条死路都量过并弃用，见 `up_finger_mask`）。
    * `out[cm == 0] = up[cm == 0]` ⇒ 夹爪永远在新物体前面，且**红线是构造性等式**
      （改动只能落在擦除掩码里），报出的 `outside_px` ≡ 0 只是回归检查。

★ 两条要标在图上的已知偏差，不能藏：
  ① 真 up 光轴是**斜**的（像心雅可比两主方向比 1.10 ⇒ ~20° 量级），本相机假定垂直下视
     ⇒ **着色**会差这一个旋转（剪影不受影响，那是面积锚死的）。`fit_mesh_pose.tilt_study`
     量过：β=10° IoU 0.902 / p95 2.32mm，β=20° 0.819 / 3.65mm。
  ② **f 与相机高退化**：报出的 f 不是标定值，只有「面积对上了」这一条是可检验的。

用法：
    PY=/home/zjq/miniconda3/envs/lerobot/bin/python
    $PY -I workplace/calibration/pose6d_up_fit.py --frames 246,348,360
    $PY -I workplace/calibration/pose6d_up_fit.py --mesh cube21.5 --frames 246,348,360
"""
import argparse
import importlib.util
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


MR = _load("_puf_mr", HERE / "meshrender.py")
F = _load("_puf_fit", HERE / "fit_mesh_pose.py")
P6 = _load("_puf_p6", HERE / "pose6d_swap.py")
PW = _load("_puf_pw", HERE / "pose6d_wrist_fit.py")   # 复用材质 + 限角顶点法线
BS = P6.BS
A = P6.A


def load_up_cam():
    """当前机位的 up 真相机（2026-10-19 标定）。只对当前取景有效！"""
    import json
    ip = json.loads((HERE / "calib/up_intrinsics.json").read_text())
    pp = json.loads((HERE / "calib/up_camera_pose.json").read_text())
    K = np.asarray(ip["K"], float)
    dist = np.asarray(ip["dist"], float)
    T = np.asarray(pp["T_base_cam"], float)
    return MR.Cam(K, dist, T, W=int(ip["image_size"][0]),
                  H=int(ip["image_size"][1]))


def load_up_rot():
    """当前 up 相机的**朝向**（cam→base，3×3），来自 2026-10-19 外参。

    ⚠️ 只取旋转，不取平移：旧数据集的取景变了，平移不可用；但同一台相机的
    安装 roll/倾角没变，像面朝向可以用它。用于合成相机的 `R_cam`。
    """
    import json
    pp = json.loads((HERE / "calib/up_camera_pose.json").read_text())
    return np.asarray(pp["T_base_cam"], float)[:3, :3]


def up_fit_frame(ds, i, g, ut, x_cube, mesh, albedo=PW.SERVO_ALBEDO, fix_s=1.0,
                 cell_fit=None, cell_draw=None, lighting=None, mat=None,
                 smooth=True, verbose=True, up_cam=None, R_fixed=None,
                 R_cam=None):
    """up 视图一帧：旧物体的观测掩码 → `fit_up` 位姿 → 网格渲进图。返回 dict。

    观测、相机锚点、擦除掩码全部走 `pose6d_swap.up_compose` 那条路，一个字不另起。
    """
    up = BS.up_bgr(ds, i)
    uw = ut.get(i) or {}
    if uw.get("mask") is None:
        return None

    fr = ds[i]
    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    P = (g["urdf"].fk(q, BS.TCP_LINK) @ np.append(x_cube, 1.0))[:3]   # 方块几何中心
    h_mm = max(0.0, float(P[2]) * 1000.0)

    # 观测 = 擦除掩码（0/255）⇒ 二值化。**不能直接把 0/255 喂给 `loss_soft`**：
    # 它的 inter/union 对 obs 的**尺度不是不变的**（inter 和 miss 都随 obs 线性缩放，
    # 而 extra 不变），喂 255 会静默改变目标函数。
    cm = BS.up_erase_mask(up, uw["mask"], uw.get("_allc"))
    obs = (cm > 0)
    if int(obs.sum()) < 40:
        if verbose:
            print(f"  f{i}: up 观测只有 {int(obs.sum())}px，跳过")
        return None

    # ---- L1 的位姿：**这一行就是和球那条路的唯一实质差别** ----
    t0 = time.time()
    if up_cam is not None:
        # 当前机位的真 K+dist+T_base_cam；z 由 FK 给定，只解 x/y（+ 自由 yaw）
        T, info = F.fit_up_real(mesh, up_cam, obs.astype(np.float32),
                                z_mm=h_mm, R_fixed=R_fixed, fix_s=fix_s,
                                cell=cell_fit, verbose=False)
        cam_kind = "real"
    else:
        T, info = F.fit_up(mesh, g, obs.astype(np.float32), h_mm=h_mm,
                           fix_s=fix_s, cell=cell_fit, verbose=False, seed=i,
                           R_fixed=R_fixed, R_cam=R_cam)
        cam_kind = "synthetic" if R_cam is None else "synthetic+real-rot"
    t_fit = time.time() - t0

    if lighting is None:
        lighting = P6.load_lighting()
    cell_used = cell_draw or mesh.cells[-1]
    tris_b = MR.transform(mesh.level(cell_used), T)
    cam = info["cam"]
    ras = MR.rasterize(tris_b, cam)
    if smooth:
        ras = PW.smooth_normals(ras, tris_b, cam,
                                PW.smooth_for(mesh, cell_used, tris_b))
    rgb = MR.shade(ras, cam, mat or PW.mesh_material(lighting), albedo)
    mesh_bgr = rgb * 255.0
    bm = ras["mask"]

    # ---- L0 底板：只抹物体本身（不抹整个盘），grow=0/shadow=False 与 up_compose 一致 ----
    plate = BS.erase_cube(up, cm, grow=0, shadow=False, fill="diffuse")[0]

    # ---- L0/L1/L2（照抄 up_compose 的定稿语义）----
    sel = bm & (cm > 0)
    out = plate.copy()
    out[bm] = mesh_bgr[bm]             # L1：整剪影画上（= 不做遮挡的错版本，留作对照）
    l1 = out.copy()
    out[cm == 0] = up[cm == 0]         # L2：擦除掩码外逐位回原帧 ⇒ 夹爪永在新物体前面

    # ---- 客观量 ----
    inter = float((bm & obs).sum())
    union = float((bm | obs).sum())
    erase_px = int((cm > 0).sum())
    # 红线：`out[cm==0] = up[cm==0]` 已把这条变成构造性等式，这里量出来是**回归检查**
    allowed = cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    changed = np.abs(out.astype(np.int16) - up.astype(np.int16)).sum(2) > 0
    res = dict(frame=i, obj_px=int((uw["mask"] > 0).sum()), erase_px=erase_px,
               mesh_px=int(bm.sum()), sel_px=int(sel.sum()),
               occl_px=int(bm.sum()) - int(sel.sum()),
               iou=inter / max(union, 1e-9),
               # `miss` = 观测到了但网格没盖住；`extra` = 网格盖到了但观测里没有。
               # ★ up 视图里 `extra ≡ occl`（观测就是 `cm>0`，一个数两个说法），
               #   打印/图注里只报一次，别让人以为是两个独立指标。
               extra_px=int((bm & ~obs).sum()), miss_px=int((~bm & obs).sum()),
               # 露在外面的**补出来的背景**：旧物体比新物体大时才有，是唯一一处「编内容」
               patch_px=max(0, erase_px - int(bm.sum())),
               area_ratio=info["area_ratio"], theta_deg=info["theta_deg"],
               f_px=info["f_px"], depth_mm=info["depth_mm"],
               shift_mm=info["shift_mm"], t_fit=t_fit, loss=info["loss"],
               outside_px=int((changed & ~allowed).sum()), cm=cm,
               cam_kind=cam_kind)
    if verbose:
        print(f"  f{i}: 观测 {res['erase_px']}px → 网格 {res['mesh_px']}px  "
              f"IoU {res['iou']:.4f}  漏 {res['miss_px']}px  "
              f"露出 {res['sel_px']}px / 被夹爪挡回 {res['occl_px']}px"
              f"（= 盖到观测外，同一个数）  "
              f"面积比 {res['area_ratio']:.4f}  绕竖轴 {res['theta_deg']:+.1f}°  "
              f"f={res['f_px']:.0f}px(退化解) d={res['depth_mm']:.0f}mm  "
              f"红线外 {res['outside_px']}px  拟合 {t_fit:.0f}s")
    return dict(out=out, l1=l1, up=up, plate=plate, ras=ras, rgb=rgb, mesh=mesh,
                T=T, info=info, **res)


def figure_up_fit(rows, out_path, title=""):
    """up 视图 6 联图，列与 `pose6d_swap.figure_up` 一一对应，便于和 `pose6d_up_l2.png` 并排看。

    第 3 栏是 L1（整剪影画上、**没做遮挡**），第 5 栏是合成；3↔5 并排就是「夹爪挡回去多少」
    的肉眼核对。第 4 栏把 L2 的两把尺画出来：擦除掩码(绿) × 网格剪影(黄)。"""
    panels = []
    for r in rows:
        if r is None:
            continue
        bm = r["ras"]["mask"]
        ov = r["l1"].copy()
        cv2.drawContours(ov, cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 1)
        d = r["out"].astype(np.int16) - r["up"].astype(np.int16)
        dv = cv2.applyColorMap(np.clip(np.abs(d).sum(2) * 2, 0, 255).astype(np.uint8),
                               cv2.COLORMAP_INFERNO)
        mk = r["out"].copy()
        for msk, col in ((r["cm"], (0, 255, 0)), (bm, (0, 255, 255))):
            cv2.drawContours(mk, cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)[0], -1, col, 1)
        # 文字全用 ASCII：cv2 的 Hershey 字体不认中文（踩过）
        cam_note = {"real": "real K+dist+T_base_cam (2026-10-19)",
                    "synthetic+real-rot": "obs-anchored pinhole + real mount rotation",
                    }.get(r.get("cam_kind"),
                          "local pinhole anchored to obs area; up axis ~20deg off")
        row = [MR.put_lines(r["up"], [f"original up f{r['frame']}"], org=(8, 22), scale=0.5),
               MR.put_lines(r["plate"], [f"L0  erase old object ({r['erase_px']}px)"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(ov, [f"L1  {r['mesh'].name}  area={r['mesh_px']}px",
                                 f"    {cam_note}"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(mk, ["L2 mask: erase mask (green) x mesh silhouette (yellow)",
                                 f"    mesh visible {r['sel_px']}px, kept back "
                                 f"{r['occl_px']}px (gripper in front)"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(r["out"], [f"composite  IoU={r['iou']:.3f}  "
                                       f"mesh outside obs {r['extra_px']}px / "
                                       f"miss {r['miss_px']}px",
                                       f"    theta={r['theta_deg']:+.1f}deg  area ratio "
                                       f"{r['area_ratio']:.4f}  red-line outside "
                                       f"{r['outside_px']}px"],
                            org=(8, 22), scale=0.5),
               MR.put_lines(dv, ["|diff| vs original"], org=(8, 22), scale=0.5)]
        panels.append(np.hstack(row))
    img = np.vstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[图] {out_path}  — {title}")
    return img


def figure_up_compare(pairs, out_path, title=""):
    """对照图：**上行=球那条路（`pose6d_up_l2` 的产物），下行=网格这条路**。

    同一帧、同一擦除掩码、同一台局部针孔，只有注入物不同 —— 差异全部归给「形状」。
    """
    rows = []
    for a, b in pairs:
        if a is None or b is None:
            continue
        bm = b["ras"]["mask"]
        ov = b["l1"].copy()
        cv2.drawContours(ov, cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 1)
        cmv = a["l1"].copy()
        cv2.drawContours(cmv, cv2.findContours(a["ball_ras"]["mask"].astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                         (0, 255, 255), 1)
        f = b["frame"]
        rows.append(np.hstack([
            MR.put_lines(b["up"], [f"original up f{f}"], org=(8, 22), scale=0.5),
            MR.put_lines(cmv, [f"ball  r={a['r_up']:.0f}px  area={a['ball_px']}px"],
                         org=(8, 22), scale=0.5),
            MR.put_lines(a["out"], [f"composite (ball)  sel={a['sel_px']}px"],
                         org=(8, 22), scale=0.5),
            MR.put_lines(ov, [f"mesh {b['mesh'].name}  area={b['mesh_px']}px  "
                              f"IoU={b['iou']:.3f}"], org=(8, 22), scale=0.5),
            MR.put_lines(b["out"], [f"composite (mesh)  sel={b['sel_px']}px"],
                         org=(8, 22), scale=0.5)]))
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
    ap.add_argument("--frames", default="338,246,350,402",
                    help="默认与 `figures/pose6d_up_l2.png`（球那条路）同一组帧，便于并排对照")
    ap.add_argument("--mesh", default="sts3215_03a_v1_v50_h0.5_m1",
                    help="calib/recon/ 下的重建网格名（或 .stl 路径）；"
                         "`cube21.5` 走解析方盒，用来和球/参考位姿对照")
    ap.add_argument("--radius-mm", type=float, default=12.0, help="球那条路（对照臂）的半径")
    ap.add_argument("--cube-h-mm", type=float, default=21.5)
    ap.add_argument("--grasp", type=int, default=246)
    ap.add_argument("--release", type=int, default=409)
    ap.add_argument("--free-s", action="store_true", help="尺度也当参数解（默认钉死 CAD 尺度）")
    ap.add_argument("--albedo", type=float, nargs=3, default=None,
                    metavar=("B", "G", "R"),
                    help="注入物颜色（BGR，0~1）；默认 cube→黄、eraser→白、其他→舵机")
    ap.add_argument("--fit-cell-mm", type=float, default=3.0,
                    help="**拟合**用的减面尺度。up 里物体只有 ~50px 宽，拟合用细网格纯属浪费")
    ap.add_argument("--draw-cell-mm", type=float, default=1.5, help="出图渲染用的减面尺度")
    ap.add_argument("--up-cam", choices=("synthetic", "real"), default="synthetic",
                    help="synthetic=旧的观测锚定局部针孔（对旧数据集）；"
                         "real=当前机位标定的 K+dist+T_base_cam")
    ap.add_argument("--up-rot", choices=("real", "vertical"), default="real",
                    help="合成相机的**像面朝向**：real=用 2026-10-19 标定的安装旋转"
                         "（长条物必须，否则 yaw/roll 会差 ~90°）；vertical=旧的"
                         "「垂直下视、像面 x 对齐 base x」假设（球/立方体可接受）")
    ap.add_argument("--allow-framing-mismatch", action="store_true",
                    help="明知数据集是旧机位也强行用 real（几乎一定会投错位置，仅用于演示）")
    ap.add_argument("--no-ball", action="store_true", help="不出对照臂那张图")
    ap.add_argument("--flat", action="store_true",
                    help="关掉顶点法线插值，回到逐面着色（棱纹会明显；用来对照）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    frames = [int(x) for x in a.frames.split(",") if x.strip()]
    up_cam = None
    if a.up_cam == "real":
        if a.dataset in ("newdata1", "newdata") and not a.allow_framing_mismatch:
            print("[拒绝] `newdata1/newdata` 的 up 帧是**旧机位**录的（CAMERAS.md："
                  "朝向+距离都变了）。2026-10-19 的 K/T_base_cam 只对当前取景有效；\n"
                  "        直接用会把物体投到错位置。请用当前机位重录一段，"
                  "或加 --allow-framing-mismatch 看错误示范。\n"
                  "        当前机位的独立验证见 `up_live_eraser.py`（桌上橡皮，IoU 0.95）。")
            return 2
        up_cam = load_up_cam()
        print(f"[up 相机] 真标定 K+dist+T_base_cam  "
              f"t={np.round(up_cam.T_base_cam[:3, 3] * 1000, 1)}mm")
    g = BS.load_geom()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(a.dataset, root=str(REPO / "dataset" / a.dataset),
                        video_backend=a.video_backend)

    ut, seed = P6.up_track_ctx(ds, g, a.grasp, a.release)
    if not ut:
        print("[up] 夹取前没在 up 图里找到方块 → 退出")
        return 1
    hit = [v for v in ut.values() if v.get("ball_px") is not None]
    print(f"[up 跟踪] 种子 f{seed[1]} px={np.round(seed[0],1)}；"
          f"夹持段 {len(ut)} 帧，其中 {len(hit)} 帧有球心路径")

    if a.mesh.startswith("cube"):
        e = float(a.mesh.replace("cube", "")) / 1000.0
        mesh = F.MeshObj(MR.box(e, e, e), name=a.mesh, cells=(2e-3, 1e-3, 0.5e-3))
    else:
        # 拟合/出图要用到的减面级别都必须预先建好（MeshObj 只缓存 cells 里的）
        cs = sorted({4e-3, a.fit_cell_mm / 1000.0, a.draw_cell_mm / 1000.0},
                    reverse=True)
        mesh = F.load_mesh(a.mesh, cells=tuple(cs))
    if a.albedo is not None:
        albedo = np.asarray(a.albedo, float)
    elif a.mesh.startswith("cube"):
        albedo = PW.CUBE_ALBEDO
    elif a.mesh.startswith("eraser"):
        albedo = PW.ERASER_ALBEDO
    else:
        albedo = PW.SERVO_ALBEDO
    print(f"注入网格：{mesh.name}  尺寸 "
          f"{np.round(np.ptp(mesh.tris0.reshape(-1, 3), axis=0) * 1000, 1)}mm（长×宽×高），"
          f"面数 {len(mesh.tris0)}")

    # ★ 和 `pose6d_swap.main` 的 up 分支用**同一个**锚点和**同一条** up 掩码路，
    #   这样两条路的擦除掩码与局部相机完全一致，差异只剩「注入物」。
    X_cube, note = BS.estimate_x_tcp_cube(
        ds, a.grasp, g, BS.UP_CAM_H_MM / 1000.0, a.cube_h_mm / 1000.0,
        BS.cube_mask_fns(g, "up", "yellow")[0])
    print(f"[常量] X_tcp_cube = {np.round(np.asarray(X_cube) * 1000, 2)}mm  ({note})")

    R_cam = load_up_rot() if (up_cam is None and a.up_rot == "real") else None
    if R_cam is not None:
        print(f"[合成相机朝向] 用真安装旋转：像面 x≈{np.round(R_cam[:,0],3)} "
              f"（过去的假设是 [1,0,0]）")
    rows = []
    for i in frames:
        R_fixed = None
        if a.mesh.startswith("eraser"):
            # 和腕视图**同一个**真实夹取朝向：12mm 厚度沿夹爪接近方向，22mm 跨指，
            # 40mm 沿指长。以前只在真 up 相机那条路上锁；合成相机仍自由搜 yaw，
            # 于是两边解出的 3D 朝向能差 20°+（40×22 的剪影对 yaw 约束很弱），
            # 看起来就是「front 往左转、up 往右转」。现在两条路都锁同一个 R。
            st = ds[i]["observation.state"].numpy().astype(float)
            q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
            T_gr = g["urdf"].fk(q, "gripper_frame_link")
            R_fixed = T_gr[:3, :3] @ PW.GRASP_R_OFFSET
        r = up_fit_frame(ds, i, g, ut, X_cube, mesh, albedo=albedo,
                         fix_s=None if a.free_s else 1.0,
                         cell_fit=a.fit_cell_mm / 1000.0,
                         cell_draw=a.draw_cell_mm / 1000.0,
                         smooth=not a.flat, up_cam=up_cam, R_fixed=R_fixed,
                         R_cam=R_cam)
        if r is not None:
            rows.append(r)

    if not rows:
        print("没有一帧拟合成功")
        return 1
    io = np.array([r["iou"] for r in rows])
    ar = np.array([r["area_ratio"] for r in rows])
    print(f"\n汇总 {len(rows)} 帧：剪影 IoU 中位 {np.median(io):.4f} 最小 {io.min():.4f}"
          f"｜面积比 {ar.min():.4f}~{ar.max():.4f}（1 = 面积锚死）"
          f"｜平均 {np.mean([r['t_fit'] for r in rows]):.0f}s/帧")

    tag = mesh.name.replace("/", "_")
    out = Path(a.out) if a.out else FIG / f"pose6d_up_fit_{tag}.png"
    figure_up_fit(rows, out, title=f"up 视图：重建网格 {mesh.name} 按拟合位姿渲回真帧")

    if not a.no_ball:
        print("\n对照臂：球那条路（pose6d_swap.up_compose，一个字没改）")
        pairs = []
        for r in rows:
            b = P6.up_compose(ds, r["frame"], g, ut, X_cube, a.radius_mm / 1000.0,
                              a.cube_h_mm / 1000.0, verbose=False)
            pairs.append((b, r))
        figure_up_compare(pairs, FIG / f"pose6d_up_fit_vs_ball_{tag}.png",
                          title="同一帧同一掩码，只有注入物不同（上：球 / 下：网格）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
