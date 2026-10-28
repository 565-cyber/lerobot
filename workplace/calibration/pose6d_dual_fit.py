#!/usr/bin/env python
"""双相机联合：**同一个 6D 位姿**渲进 up（固定标定相机）+ front（腕相机），出橡皮的剪影掩码。

和 `pose6d_wrist_fit.py` / `pose6d_up_fit.py` 的关系
---------------------------------------------------
那两路各自只用一个相机、各自解一个位姿，所以「up 里的橡皮」和「腕里的橡皮」**不是
同一个物体**。这里要的是 Pose6DAug 意义上的 6D 注入：两个视角共用 `T_base_obj`。

做法是把「夹持期物体和夹爪刚性固定」这件事用满（`ball_swap.py` 的注释里已经用
观测独立证实过：搬运段方块相对腕相机**像素级不动**）：

    设  t_tool   = 物心在 TCP 帧（`BS.TCP_LINK`）里的常向量
        t_cam    = 物心在**腕相机原始手眼链**里的常向量
    则  up  视角： P(k) = FK(k, TCP_LINK) @ t_tool           —— up 相机是标定过的
        front 视角： P(k) = FK(k, wrist_camera_link) @ X @ t_cam

两者要对上，差的就是**手眼外参的平移误差**：

    X_corr = X + [0 | Δ],   Δ = t_tool_wc − t_x − R_x @ t_cam     （全在一个 link 帧里，常数）

`Δ` 一算出来就是常数，所以「联合」不是逐帧硬凑：**全局先解一次 t_tool、t_cam、Δ，
之后每帧的位姿完全由 FK 决定**。这也是能写出「同一个位姿」的前提。

实测（newdata_current，yellow 方块 → 橡皮，f240 夹取 / f356 释放）
------------------------------------------------------------------
· 旧常量 `X_tcp_cube = (14.1, 24.2, −13.6)mm` 是**错的**：它把方块放在指尖附近；
  拿 up 观测重新解出来是 `(−9.3, 14.0, −49.7)mm`（夹在两指根部）。
  这一个常量错了 40mm，正是旧腕图里「FK 原差 149px」的主因（不是手眼链坏了）。
· 修正后：腕相机原始手眼把方块投到观测 **17px** 以内；up 观测的常偏移拟合残差
  中位 9.5px（p90 23px）。
· 腕相机深度还差 29mm（`t_cam` 模长 114.5mm，手眼链给 85.6mm），由 `Δ` 一次修掉。

用法
----
    PY=/home/wzy/anaconda3/envs/lerobot/bin/python
    PYTHONPATH=/data1/lerobot/workplace $PY -I pose6d_dual_fit.py \\
        --dataset newdata_current --grasp 240 --release 356 \\
        --frames 260,280,300,320,340 --save-masks /tmp/dual_masks

产物：`figures/pose6d_dual_fit_eraser.png`（每帧 2 行 × 5 栏）+ 两个视角的剪影 PNG。
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


MR = _load("_pdu_mr", HERE / "meshrender.py")
F = _load("_pdu_fit", HERE / "fit_mesh_pose.py")
P6 = _load("_pdu_p6", HERE / "pose6d_swap.py")
PW = _load("_pdu_pw", HERE / "pose6d_wrist_fit.py")
PU = _load("_pdu_pu", HERE / "pose6d_up_fit.py")
BS = P6.BS
A = P6.A


# ================================================================ 全局常量：一次解完，全段复用
def _up_obs(ds, ut, i, min_area=300):
    """up 视角这一帧的「旧物体在哪」：擦除掩码 → (obs_bool, centroid_px)。

    观测走 `up_erase_mask`（与 `up_compose` 同一条路）。面积太小/没有掩码就返回 None。
    """
    u = ut.get(i) or {}
    if u.get("mask") is None:
        return None
    cm = BS.up_erase_mask(BS.up_bgr(ds, i), u["mask"], u.get("_allc"))
    if int((cm > 0).sum()) < min_area:
        return None
    ys, xs = np.nonzero(cm)
    return cm > 0, np.array([xs.mean(), ys.mean()])


def fit_t_tool(ds, upc, ut, frames, g, p0, f_scale=5.0):
    """解「物心在 TCP 帧里的常向量」。up 相机 + FK 都可信，所以这是纯最小二乘。

    `p0` 用 `BS.estimate_x_tcp_cube` 的老估计当种子 —— 实测它会偏 ~40mm（见模块
    docstring），但只是种子，不影响收敛。返回 `(t_tool, info)`。
    """
    from scipy.optimize import least_squares

    items = []
    for i in frames:
        got = _up_obs(ds, ut, i)
        if got is None:
            continue
        st = ds[i]["observation.state"].numpy().astype(float)
        q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
        items.append((g["urdf"].fk(q, BS.TCP_LINK), got[1]))
    if len(items) < 8:
        raise RuntimeError(f"up 观测只有 {len(items)} 帧，解不出常量")

    def resid(p):
        r = []
        for T, c in items:
            uv, _ = upc.project((T @ np.append(p, 1.0))[:3].reshape(1, 3))
            r.extend(uv[0] - c)
        return np.asarray(r, float)

    sol = least_squares(resid, np.asarray(p0, float), loss="soft_l1",
                        f_scale=f_scale, max_nfev=200)
    r = np.linalg.norm(resid(sol.x).reshape(-1, 2), axis=1)
    r0 = np.linalg.norm(resid(p0).reshape(-1, 2), axis=1)
    return sol.x, dict(n=len(items), res_med=float(np.median(r)),
                       res_p90=float(np.percentile(r, 90)), res_max=float(r.max()),
                       res0_med=float(np.median(r0)), res0_p90=float(np.percentile(r0, 90)))


def fit_t_cam_obj(ds, trk, g, frames, cube_m, cell, verbose=True):
    """解「物心在**腕相机原始手眼链**里的常向量」，只用 front 观测（自由 6-DoF 剪影拟合）。

    ★ 这里**不加 per-frame δ**：δ 是拿「FK 预测的物心」去解相机的，会把物体相对夹爪的
      任何偏差吸进相机里，于是「物心在相机系里恒定」变成构造性等式、测不出东西。
      直接用原始手眼 `FK@X` 当相机，拟合出来的平移才可跨帧比较。
    """
    tvals, rows = [], []
    for i in frames:
        rec = trk.get(i)
        if rec is None or rec.get("c") is None:
            continue
        fr = ds[i]
        fg = P6.bgr_of(fr["observation.images.front"])
        obj, _ = P6.object_mask(rec, fg, np.asarray(rec["c"], float))
        obs = obj > 0
        if obs.sum() < 1000:
            continue
        st = fr["observation.state"].numpy().astype(float)
        q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
        T_raw = P6.cam_from_state(g, q)
        cam = MR.Cam(g["K"], g["dist"], T_raw)
        T, info = F.fit_wrist(cube_m, cam, obs.astype(np.float32), fix_s=1.0,
                              cell_fine=cell, verbose=False, seed=i)
        t_co = (np.linalg.inv(T_raw) @ T)[:3, 3]
        tvals.append(t_co)
        rows.append(dict(frame=i, t_mm=(t_co * 1000).round(2).tolist(),
                         depth_mm=float(np.linalg.norm(t_co) * 1000),
                         iou=float(info.get("iou", 0.0))))
        if verbose:
            print(f"  [front 标定 f{i}] 物心(相机系) {np.round(t_co * 1000, 2)}mm "
                  f"|d|={np.linalg.norm(t_co) * 1000:.1f}mm IoU {info.get('iou', 0):.3f}")
    if not tvals:
        raise RuntimeError("front 观测一帧都没拟合成功")
    return np.median(np.asarray(tvals), axis=0), rows


def handeye_delta(t_tool, t_cam, g, q_ref):
    """手眼平移修正 Δ（EE 帧）。推导见模块 docstring；返回 `(X_corr, Delta, t_tool_wc)`。"""
    X = np.asarray(g["X"], float)
    R_x, t_x = X[:3, :3], X[:3, 3]
    T_wc = g["urdf"].fk(q_ref, "wrist_camera_link")
    T_tcp = g["urdf"].fk(q_ref, BS.TCP_LINK)
    t_tool_wc = (np.linalg.inv(T_wc) @ T_tcp @ np.append(t_tool, 1.0))[:3]
    delta = t_tool_wc - t_x - R_x @ np.asarray(t_cam, float)
    Xc = X.copy()
    Xc[:3, 3] = X[:3, 3] + delta
    return Xc, delta, t_tool_wc


# ================================================================ 一帧：同一个位姿 → 两个视角
# 橡皮姿态就用真实夹取方式（40mm 跨指、22mm 沿指长、12mm 沿接近方向）——
# 用户确认过实物就是这样夹的，别按 docstring 的字面意思去「修」。
GRASP_R = PW.GRASP_R_OFFSET


def eraser_pose(q, g, t_tool):
    """橡皮的 `T_base_obj`：**平移**取物心（t_tool，TCP 帧），**朝向**取真实夹取方式。

    朝向 = `FK(gripper_frame_link) @ GRASP_R`，`GRASP_R` 的列就是物体局部 x/y/z 在
    夹爪系里的方向：local x(40mm) → +夹爪 y，local y(22mm) → −夹爪 x，local z(12mm)
    → +夹爪 z。和平移分开写，是因为平移由观测定、朝向是设计选择。
    """
    T_g = g["urdf"].fk(q, "gripper_frame_link")
    T_tcp = g["urdf"].fk(q, BS.TCP_LINK)
    t_g = (np.linalg.inv(T_g) @ T_tcp @ np.append(t_tool, 1.0))[:3]
    T = np.eye(4)
    T[:3, :3] = GRASP_R
    T[:3, 3] = t_g
    return T_g @ T, t_g


def render(mesh, cell, T, cam, light, mat, albedo, smooth=True):
    tris = MR.transform(mesh.level(cell), T)
    ras = MR.rasterize(tris, cam)
    if smooth:
        ras = PW.smooth_normals(ras, tris, cam, PW.smooth_for(mesh, cell, tris))
    rgb = MR.shade(ras, cam, mat, albedo)
    return ras, np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def clip_height(bm, obs):
    """只夹**图像纵向**：剪影上下不越出红框（`obs`）的 y 范围；**横向不夹**。

    用户 2026-10-21 的原话：「我是指竖着的这个不要超过，横着的可以超过」。
    up 视角里橡皮 40mm 那条长轴几乎横躺在图像上（投影 ~87px），旧方块那个方向只有
    27px —— 横向本来就该超出，夹了反而把长轴切掉、up 里变成一根竖条，和 front 的横躺
    平板对不上。竖直方向（22mm 轴 + 12mm 厚度）投影 22~50px，旧方块那个方向 46~68px，
    夹住上下沿，橡皮就不会比旧物体高出去。
    """
    if not obs.any() or not bm.any():
        return bm
    ys = np.nonzero(obs)[0]
    y0, y1 = float(ys.min()), float(ys.max())
    v = np.arange(bm.shape[0], dtype=np.float32)[:, None]
    return bm & (v >= y0) & (v <= y1)


def cover_obj(bm_vis, bm_src, need, bgr):
    """把 `need`（旧物体观测）里剪影还没盖住的像素补进剪影，返回 (mask, 颜色图)。

    橡皮 12mm 比方块 21.5mm 薄，斜视投影时会在纵向短一截（f320 渲染 50px 高、红框
    61px），旧物体就会从橡皮边上露出来；补上之后绿框把红框整个包住。

    补出来的像素没有渲染颜色，用 `distanceTransformWithLabels` 从**橡皮自身**取最近邻
    的色，接缝在剪影外侧一圈，看不出拼接。
    """
    add = need & ~bm_vis
    if not add.any() or not bm_src.any():
        return bm_vis, bgr
    # 只补**贴着剪影**的那部分：f340 旧方块在被黑线缆切断的地方还露出顶上一小条，
    # 补上去就是一块悬空的橡皮色，反而更假 —— 隔着一个线缆宽度的连通块直接丢掉。
    n, lab = cv2.connectedComponents(add.astype(np.uint8), connectivity=8)
    near = cv2.dilate(bm_vis.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    keep = [k for k in np.unique(lab[near]) if k]
    if not keep:
        return bm_vis, bgr
    add = np.isin(lab, keep)
    # 颜色取剪影里最近的一颗（distanceTransform 的零像素 = 有颜色的像素）。
    # 只拿**亮的那一半**当色源：剪影最上面几行是背光的暗侧棱，拿它往上补会在橡皮顶上
    # 糊出一圈灰；用顶面那一片补，接上去才像同一个物体。（按中位数分亮/暗，所以
    # 橡皮 2026-10-21 改成「用过的偏黑」深灰之后这里不用改，照样是顶面最亮那半边。）
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright = bm_src & (v >= np.median(v[bm_src]))
    if not bright.any():
        bright = bm_src
    _, labels = cv2.distanceTransformWithLabels(
        (~bright).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    src = np.flatnonzero(bright.ravel())      # labels 的 1..N 就按这个行主序编号
    dst = np.flatnonzero(add.ravel())
    out = bgr.copy().reshape(-1, 3)
    out[dst] = out[src[labels.ravel()[dst] - 1]]
    out = out.reshape(bgr.shape)
    # 最近邻是「整列取同一个色」，补的区域一高就出竖条纹（f320 顶上那道灰白条纹）。
    # 补的像素做一次高斯就平了；原渲染像素一点不动。
    out[add] = cv2.GaussianBlur(out, (0, 0), 2.0)[add]
    return bm_vis | add, out


def compose_up(up, cm, bm, eraser_bgr, fore):
    """up 视角 L0/L1/L2。`bm` 传进来的已经是**对过下缘、夹过宽度、补满旧物体**的剪影。"""
    # grow=2：旧方块的抗锯齿边 1~2px 落在掩码外，不扩的话会在橡皮边缘留一圈黄绿细线
    # （f300 实测：橡皮左/下缘那道线就是这么来的）。up 里物体只有 ~30px 见方，2px 不影响。
    plate = BS.erase_cube(up, cm, grow=2, shadow=False, fill="diffuse")[0]
    out = plate.copy()
    out[bm] = eraser_bgr[bm]
    on_plate = out.copy()
    out[fore] = up[fore]
    return out, on_plate, plate


def compose_front(fg, obj, bm, eraser_bgr, front):
    """腕视角 L0/L1/L2 —— 照抄 `pose6d_wrist_fit.wrist_fit_frame` 的定稿写法。"""
    cm = cv2.bitwise_or(obj, (bm.astype(np.uint8) * 255))
    plate = BS.erase_cube(fg.copy(), cm, grow=3, shadow=True, fill="diffuse")[0]
    out = plate.copy()
    out[bm] = eraser_bgr[bm]
    on_plate = out.copy()
    out[front] = fg[front]
    return out, on_plate, plate


def dual_frame(ds, i, g, upc, t_tool, Xc, mesh, cube, cell, light, mat, trk, ut,
               smooth=True, verbose=True):
    fr = ds[i]
    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    T_wc = g["urdf"].fk(q, "wrist_camera_link")
    T_tcp = g["urdf"].fk(q, BS.TCP_LINK)

    # ---- 两个相机：up 是标定的固定相机；front = FK ⊗ 修正后的手眼 ----
    cam_f = MR.Cam(g["K"], g["dist"], T_wc @ Xc)
    P_obj = (T_tcp @ np.append(t_tool, 1.0))[:3]

    # ---- 观测（两个视角各一份）----
    fg = P6.bgr_of(fr["observation.images.front"])
    rec = trk.get(i)
    if rec is None or rec.get("c") is None:
        return None
    c_f = np.asarray(rec["c"], float)
    obj, obj_src = P6.object_mask(rec, fg, c_f)
    obs_f = obj > 0

    up = BS.up_bgr(ds, i)
    got_u = _up_obs(ds, ut, i)
    if got_u is None:
        return None
    cm = BS.up_erase_mask(up, ut[i]["mask"], ut[i].get("_allc"))
    # ★ 夹持初期（f264~f282）`up_erase_mask` 会顺带圈进**手指的受光面**：它的放宽条件是
    #   「离追踪中心 70px、H8~48/S≥45/V≥55 的连通块」，而手指被暖光照到正好落在这个色域里。
    #   实测 f278：追踪掩码 1698px，擦除掩码 3877px、bbox 竖跨 122px（方块本体才 ~55px），
    #   形心被拽低 20px —— 于是 `cover_obj` 把橡皮抹到手指上，位姿误差也从 1px 虚报成 39px。
    #   夹在「追踪掩码外扩 51px」以内就够了：方块暗半边离亮半边只有十几像素，手指那段进不来。
    #   只在 dual 这条链上夹，不动 `ball_swap` 里那套已经调好的参数。
    _near = (cv2.dilate(ut[i]["mask"].astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))) > 0)
    cm = cm * _near.astype(cm.dtype)
    obs_u = cm > 0

    # ---- 同一个位姿 ----
    T_e, t_g = eraser_pose(q, g, t_tool)
    ras_f, bgr_f = render(mesh, cell, T_e, cam_f, light, mat, PW.ERASER_ALBEDO, smooth)
    ras_u, bgr_u = render(mesh, cell, T_e, upc, light, mat, PW.ERASER_ALBEDO, smooth)
    bm_f, bm_u = ras_f["mask"], ras_u["mask"]

    # ---- 拟合质量的客观量：**代理物体**（方块）的剪影 vs 观测 ----
    T_p, _ = eraser_pose(q, g, t_tool)                       # 同心，用小方块量位姿
    T_p[:3, :3] = np.eye(3)
    # 代理方块的朝向无所谓（立方体），只量中心对不对
    cc = cell if cell in cube.cells else cube.cells[-1]
    cov_f = F.coverage(F.mesh_tris_at(cube, cc, T_p), cam_f, ss=2)
    cov_u = F.coverage(F.mesh_tris_at(cube, cc, T_p), upc, ss=2)
    lf = F.loss_soft(cov_f, obs_f.astype(np.float32))
    lu = F.loss_soft(cov_u, obs_u.astype(np.float32))
    proj_f, _ = cam_f.project(P_obj.reshape(1, 3))
    proj_u, _ = upc.project(P_obj.reshape(1, 3))

    # ---- 前景掩码（L2）----
    # up：FK 投影的夹爪（up 相机+FK 都准，实测投影严丝合缝）+ 亮白手指，外扩 2px
    # ★ up 视角这里**刻意不用 FK 夹爪栅格**：up 相机是在桌面上标定的，夹爪抬起
    #   150~280mm 后 `gripper_link`/`moving_jaw` 的栅格比实拍高出 ~130px（实测 f300：
    #   栅格 bbox y 233..471，实拍白色夹爪 y 330..480；方块的观测 bbox 被它整个盖住，
    #   `cm & arm = 2116/2336`）。拿它当 L2 的话，原方块会被整块贴回去 —— 试过，
    #   L2 图里橡皮完全消失、方块还在。所以这里只用**观测**：亮白低饱和 = 手指。
    #   亮白只在**橡皮附近**算：up 画面里还有别的白色东西（白盒/反光），全图找会
    #   在橡皮上凿出洞来；手指就在橡皮边上，60px 的带足够。
    # 阈值按实测取样定：白色塑料 V≈195~255 / S≈3~13，桌面 V≈160~180 / S≈19~46，
    # 影子里塑料 V≈170 / S≈25。V≥210 会漏掉受光较弱的腕壳（f300 实测 V=200），
    # 于是 f260 那块橡皮直接压在腕壳上没人管 —— 这就是用户看到的问题。
    #
    # ★ 2026-10-21 用户：「320 帧 up 视角夹爪像素没在上面，要让上夹爪全部露出来」。
    #   原因不是 V 不够，是**饱和度**：夹爪顶上那截塑料被红盒子反光照成暖白，
    #   S≈24~50（正的白色 S≈3~13），被 S<=20 整条挡在门外 —— f320 实测只有 440px
    #   留下，整根手指 3000px 全被橡皮盖住。
    #   补一条 S<=55 & V>=180：正好罩住暖白手指（实测 V 176~204 / S 24~50），
    #   桌面是 V160~180 / S19~46、红盒是 S=255，两条边都够高/够红，进不来。
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))
    near = cv2.dilate(((cm > 0) | bm_u).astype(np.uint8), kk) > 0
    white = (((hsv[:, :, 1] <= 20) & (hsv[:, :, 2] >= 165)) |
             ((hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 180))) & near
    # 只留**和原来那批正白塑料连通**的：散在桌上的亮斑（反光、盒沿）不跟手指连着，
    # 去掉它们，免得在橡皮上凿洞。
    seed = cv2.dilate((white & (hsv[:, :, 1] <= 20)).astype(np.uint8),
                      np.ones((9, 9), np.uint8)) > 0
    _, lab = cv2.connectedComponents(white.astype(np.uint8), 8)
    keep = np.unique(lab[seed]); keep = keep[keep > 0]
    if len(keep):
        white = np.isin(lab, keep)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fore_u = cv2.dilate(white.astype(np.uint8), k) > 0
    # front：观测凸包 + FK 几何两条路（`pose6d_wrist_fit` 的定稿）
    _, front_obs, _ = P6.front_mask_observational(fg, obj, P6._carry_yellow)
    _, front_geo, _ = P6.front_mask_geometric(ds, i, g, cam_f, ras_f, obj)
    front = front_obs | front_geo

    # 纵向夹到红框上下沿 → 补满旧物体。横向不夹（见 `clip_height` 的说明）。
    bm_u_vis = clip_height(bm_u, obs_u)
    bm_u_vis, bgr_u_vis = cover_obj(bm_u_vis, bm_u, obs_u, bgr_u)
    dy_u = 0
    out_u, l1_u, plate_u = compose_up(up, cm, bm_u_vis, bgr_u_vis, fore_u)
    out_f, l1_f, plate_f = compose_front(fg, obj, bm_f, bgr_f, front)

    res = dict(frame=i, P_mm=(P_obj * 1000).round(2).tolist(),
               proj_f=proj_f[0].round(1).tolist(), obs_f=c_f.round(1).tolist(),
               proj_u=proj_u[0].round(1).tolist(),
               obs_u_c=np.round(got_u[1], 1).tolist(),
               err_f=float(np.linalg.norm(proj_f[0] - c_f)),
               err_u=float(np.linalg.norm(proj_u[0] - got_u[1])),
               obs_u_px=int(obs_u.sum()), obs_f_px=int(obs_f.sum()),
               eraser_u_px=int(bm_u_vis.sum()), eraser_f_px=int(bm_f.sum()),
               eraser_u_full_px=int(bm_u.sum()),
               shift_u_px=int(dy_u),
               proxy_iou_f=float(lf[1]["iou"]), proxy_iou_u=float(lu[1]["iou"]),
               front_px=int(front.sum()), fore_u_px=int(fore_u.sum()),
               obj_src=obj_src)
    if verbose:
        print(f"  f{i}: 橡皮剪影 up {res['eraser_u_px']}px / front {res['eraser_f_px']}px  "
              f"位姿投影误差 up {res['err_u']:.1f}px / front {res['err_f']:.1f}px  "
              f"代理方块 IoU up {res['proxy_iou_u']:.3f} / front {res['proxy_iou_f']:.3f}")
    return dict(out_u=out_u, l1_u=l1_u, plate_u=plate_u, out_f=out_f, l1_f=l1_f,
                plate_f=plate_f, up=up, fg=fg, bm_u=bm_u_vis, bm_u_full=bm_u,
                bm_f=bm_f, cm=cm, obj=obj, fore_u=fore_u, front=front, **res)


def figure_rows(rows):
    panels = []
    for r in rows:
        if r is None:
            continue
        up_view = MR.put_lines(r["up"].copy(), [
            f"up f{r['frame']}  {r['obj_src']}  obs {r['obs_u_px']}px  "
            f"eraser {r['eraser_u_px']}px  err {r['err_u']:.0f}px"], org=(8, 22), scale=0.45)
        cv2.drawContours(up_view,
                         cv2.findContours(r["cm"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 1)
        cv2.drawContours(up_view,
                         cv2.findContours(r["bm_u"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 1)
        fg_view = MR.put_lines(r["fg"].copy(), [
            f"front f{r['frame']}  {r['obj_src']}  obs {r['obs_f_px']}px  "
            f"eraser {r['eraser_f_px']}px  err {r['err_f']:.0f}px"], org=(8, 22), scale=0.45)
        cv2.drawContours(fg_view,
                         cv2.findContours(r["obj"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 1)
        cv2.drawContours(fg_view,
                         cv2.findContours(r["bm_f"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 1)

        def mask3(m, name):
            img = np.zeros((*m.shape, 3), np.uint8)
            img[m] = (90, 90, 90)
            return MR.put_lines(img, [f"{name} mask {int(m.sum())}px"], org=(8, 22),
                                scale=0.45)

        for img, ras, tag in ((up_view, r["bm_u"], "up"),
                              (fg_view, r["bm_f"], "front")):
            panels.append([
                img,
                MR.put_lines(r[f"plate_{'u' if tag == 'up' else 'f'}"].copy(),
                             ["L0  erase old + inpaint"], org=(8, 22), scale=0.45),
                mask3(ras, tag),
                MR.put_lines(r[f"l1_{'u' if tag == 'up' else 'f'}"].copy(),
                             ["L1  eraser (no occlusion)"], org=(8, 22), scale=0.45),
                MR.put_lines(r[f"out_{'u' if tag == 'up' else 'f'}"].copy(),
                             ["L2  + robot foreground"], org=(8, 22), scale=0.45),
            ])
    return panels


def vrange(spec, grasp, release):
    """`--video-frames`：空 = grasp..release；`a:b` = 闭区间；`a,b,c` = 列举。"""
    spec = (spec or "").strip()
    if not spec:
        return list(range(grasp, release + 1))
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def video_img(r):
    """视频一帧的版式：上面一行原图（up | front），下面一行换完橡皮（up | front）。"""
    h = r["up"].shape[0]
    img = np.vstack([np.hstack([r["up"], r["fg"]]),
                     np.hstack([r["out_u"], r["out_f"]])])
    # ★ 必须是英文：`cv2.putText` 只有 Hershey 矢量字，中文会全渲成 "???"
    img = MR.put_lines(img, [f"f{r['frame']}   original   up | front"],
                       org=(10, 26), scale=0.7)
    return MR.put_lines(img, [f"f{r['frame']}   eraser swapped   up | front"],
                        org=(10, h + 26), scale=0.7)


def grid(panels, out_path, title=""):
    if not panels:
        raise RuntimeError("没有可出的图")
    h, w = panels[0][0].shape[:2]
    cols = len(panels[0])
    rows = len(panels) // 2
    gaps = 4
    W = cols * w + (cols - 1) * gaps
    H = rows * (2 * h + gaps) + max(0, rows - 1) * (gaps * 3)
    canvas = np.full((H, W, 3), 255, np.uint8)
    top = 0
    for rr in range(rows):
        for cc in range(cols):
            r_idx = 2 * rr
            canvas[top:top + h, cc * (w + gaps):cc * (w + gaps) + w] = panels[r_idx][cc]
            canvas[top + h + gaps:top + 2 * h + gaps,
                   cc * (w + gaps):cc * (w + gaps) + w] = panels[r_idx + 1][cc]
        top += 2 * h + gaps + 3 * gaps
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"[图] {out_path}  — {title}")


# ================================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata_current")
    ap.add_argument("--root", default=None)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--grasp", type=int, default=240, help="夹取帧（夹爪合上）")
    ap.add_argument("--release", type=int, default=356, help="释放帧（夹爪张开）")
    ap.add_argument("--frames", default="260,280,300,320,340")
    ap.add_argument("--fit-frames", default="",
                    help="解 t_cam 用哪几帧（空 = 夹持段均匀取 3 帧）")
    ap.add_argument("--mesh", default="eraser_40x22x12_r3n80")
    ap.add_argument("--proxy-cube-mm", type=float, default=21.5,
                    help="局部观测里旧物体的边长（newdata_current 是黄色方块 21.5mm）；"
                         "设 0 则直接用 --mesh 当代理")
    ap.add_argument("--cell-mm", type=float, default=0.5)
    ap.add_argument("--out", default="figures/pose6d_dual_fit_eraser.png")
    ap.add_argument("--save-masks", default=None, help="把两个视角的橡皮剪影 PNG 存到这里")
    ap.add_argument("--video", default=None,
                    help="整段夹持期的成片 mp4（上排原图 / 下排换完橡皮，up|front）")
    ap.add_argument("--video-frames", default="",
                    help="视频帧范围，如 240:356；空 = grasp..release 全段")
    ap.add_argument("--video-fps", type=float, default=15.0,
                    help="成片帧率（原始 30fps，逐帧要跑跟踪+渲染，默认 15 好播）")
    ap.add_argument("--t-tool-mm", type=float, nargs=3, default=None,
                    help="手动给定 t_tool（TCP 帧，mm）；不给就现场解")
    ap.add_argument("--no-smooth", action="store_true")
    a = ap.parse_args()

    frames = [int(x) for x in a.frames.split(",") if x.strip()]
    fit_frames = ([int(x) for x in a.fit_frames.split(",") if x.strip()]
                  if a.fit_frames else
                  np.linspace(a.grasp + 15, a.release - 10, 3).round().astype(int).tolist())

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    g = BS.load_geom()
    upc = PU.load_up_cam()
    root = a.root or str(REPO / "dataset" / a.dataset)
    ds = LeRobotDataset(a.dataset, root=root, video_backend=a.video_backend)
    print(f"[数据] {a.dataset}  {root}  夹持段 f{a.grasp}~f{a.release}")

    trk = BS.carry_track(ds, a.grasp, a.release)
    ut, seed = P6.up_track_ctx(ds, g, a.grasp, a.release)
    print(f"[跟踪] front {len(trk)} 帧；up {len(ut)} 帧（种子 f{seed[1]}）")

    # ---- 1) t_tool：up 观测 + FK（up 相机是标定的）----
    p0, note = BS.estimate_x_tcp_cube(ds, a.grasp, g, BS.UP_CAM_H_MM / 1000.0,
                                      (a.proxy_cube_mm or 0.0215) / 1000.0,
                                      BS.cube_mask_fns(g, "up", "yellow")[0])
    if p0 is None:                     # 夹取前 up 图没干净看到方块：给一个中性种子
        p0 = np.array([0.0, 0.0, -0.035])
        note += "（种子解不出来，用 0,0,-35mm）"
    print(f"[种子] X_tcp_cube = {np.round(p0 * 1000, 2)}mm  ({note})")
    if a.t_tool_mm is not None:
        t_tool = np.asarray(a.t_tool_mm, float) / 1000.0
        t_info = dict(n=0, res_med=float("nan"), res_p90=float("nan"),
                      res_max=float("nan"), res0_med=float("nan"), res0_p90=float("nan"))
    else:
        t0 = time.time()
        t_tool, t_info = fit_t_tool(ds, upc, ut, range(a.grasp, a.release), g, p0)
        print(f"[常量 t_tool] {np.round(t_tool * 1000, 2)}mm  (TCP 帧)  "
              f"up 投影残差 中位 {t_info['res_med']:.1f}px / p90 {t_info['res_p90']:.1f}px "
              f"（旧种子 中位 {t_info['res0_med']:.1f}px / p90 {t_info['res0_p90']:.1f}px）"
              f"  {time.time() - t0:.1f}s  n={t_info['n']}")

    # ---- 2) t_cam：front 观测 + 原始手眼（不加 δ，见 docstring）----
    cell = a.cell_mm / 1000.0
    if a.proxy_cube_mm and a.proxy_cube_mm > 0:
        e = a.proxy_cube_mm / 1000.0
        cube = F.MeshObj(MR.box(e, e, e), name=f"cube{a.proxy_cube_mm:g}",
                         cells=(2 * cell, cell, cell / 2))
    else:
        cube = F.load_mesh(a.mesh, cells=(2 * cell, cell, cell / 2))
    t_cam, c_rows = fit_t_cam_obj(ds, trk, g, fit_frames, cube, cell)
    print(f"[常量 t_cam] {np.round(t_cam * 1000, 2)}mm  (腕相机原始手眼帧)  "
          f"{len(c_rows)} 帧中位")

    # ---- 3) 手眼平移修正 Δ（常数）----
    q_ref = A.norm_to_q(
        {m: float(v) for m, v in zip(A.MOTORS,
                                     ds[(a.grasp + a.release) // 2]["observation.state"]
                                     .numpy().astype(float))}, g["urdf"])
    Xc, delta, t_tool_wc = handeye_delta(t_tool, t_cam, g, q_ref)
    print(f"[手眼] X 的平移 {np.round(g['X'][:3, 3] * 1000, 1)} → "
          f"{np.round(Xc[:3, 3] * 1000, 1)}mm  Δ_ee={np.round(delta * 1000, 1)}mm "
          f"(|Δ|={np.linalg.norm(delta) * 1000:.1f}mm)")

    # ---- 4) 网格 + 材质 ----
    mesh = F.load_mesh(a.mesh, cells=(2 * cell, cell, cell / 2))
    light = P6.load_lighting()
    mat = MR.Light(direction=light.dir, ambient=PW.ERASER_AMBIENT,
                   diffuse=PW.ERASER_DIFFUSE, specular=PW.ERASER_SPECULAR,
                   shininess=PW.ERASER_SHININESS)
    print(f"[网格] {mesh.name}  尺寸 {np.round(np.ptp(mesh.tris0.reshape(-1, 3), axis=0) * 1000, 1)}mm"
          f"（长×宽×高）  拟合代理 {cube.name}")

    # ---- 5) 逐帧：同一个位姿 → 两个视角 ----
    rows, t0 = [], time.time()
    for i in frames:
        print(f"[帧] f{i}")
        r = dual_frame(ds, i, g, upc, t_tool, Xc, mesh, cube, cell, light, mat,
                       trk, ut, smooth=not a.no_smooth)
        if r is not None:
            rows.append(r)
            if a.save_masks:
                d = Path(a.save_masks)
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"mask_up_{i:05d}.png"),
                            (r["bm_u"].astype(np.uint8) * 255))
                cv2.imwrite(str(d / f"mask_front_{i:05d}.png"),
                            (r["bm_f"].astype(np.uint8) * 255))
    if not rows:
        print("没有任何帧成功")
        return 1

    p_out = Path(a.out)
    if not p_out.is_absolute():
        p_out = HERE / p_out
    grid(figure_rows(rows), p_out,
         f"dual-camera eraser  t_tool={np.round(t_tool * 1000, 1)}mm  "
         f"Δ={np.round(delta * 1000, 1)}mm")

    print(f"\n汇总 {len(rows)} 帧（{time.time() - t0:.1f}s）：")
    for r in rows:
        print(f"  f{r['frame']}: 位姿投影误差 up {r['err_u']:5.1f}px / front {r['err_f']:5.1f}px"
              f"  代理 IoU up {r['proxy_iou_u']:.3f} / front {r['proxy_iou_f']:.3f}"
              f"  橡皮掩码 up {r['eraser_u_px']}px / front {r['eraser_f_px']}px")
    if a.save_masks:
        print(f"[掩码] 已写入 {a.save_masks}/mask_{{up,front}}_XXXXX.png")

    # ---- 6) 整段成片 ----
    if a.video:
        vf = vrange(a.video_frames, a.grasp, a.release)
        p_vid = Path(a.video)
        if not p_vid.is_absolute():
            p_vid = HERE / p_vid
        p_vid.parent.mkdir(parents=True, exist_ok=True)
        cache = {r["frame"]: r for r in rows}
        vw, t1, miss = None, time.time(), []
        for k, i in enumerate(vf):
            r = cache.get(i)
            if r is None:
                r = dual_frame(ds, i, g, upc, t_tool, Xc, mesh, cube, cell, light,
                               mat, trk, ut, smooth=not a.no_smooth, verbose=False)
            if r is None:                       # 跟踪断了（夹爪挡住/离开视野）就跳过
                miss.append(i)
                continue
            img = video_img(r)
            if vw is None:
                vw = cv2.VideoWriter(str(p_vid), cv2.VideoWriter_fourcc(*"mp4v"),
                                     a.video_fps, (img.shape[1], img.shape[0]))
                if not vw.isOpened():
                    raise SystemExit(f"写不了视频 {p_vid}")
            vw.write(img)
            if (k + 1) % 10 == 0 or k + 1 == len(vf):
                print(f"  [视频] {k + 1}/{len(vf)}  {time.time() - t1:.1f}s")
        if vw is not None:
            vw.release()
        print(f"[视频] {p_vid}  {len(vf) - len(miss)} 帧 @ {a.video_fps:g}fps"
              + (f"  跟踪缺失跳过 {len(miss)} 帧：{miss}" if miss else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
