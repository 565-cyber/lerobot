"""夹持可行性闸门：开口 vs 命令，以及「这个物体装不装得进这对手指」。

为什么需要（用户 2026-10-15 指出的事，也是 Pose6DAug 论文自己承认的失败模式）：
Pose6DAug 的旋转/平移/缩放**不是抓取规划**，它只在已有成功轨迹上调整物体几何、尽量保持
原来的接触关系。所以新物体和原物体的宽度/尺寸/中心差太多时，替换模型**会和手指相交**或
**伸出可抓取区域**。这件事**改掩码修不掉**，必须单独量物体的原点、尺寸和相对夹爪的变换。

本机的事实（`so101_new_calib.urdf` 实测，不是查手册）：
  · **静颚 = `gripper_link`，它的 visual 就是 `sts3215_03a_v1.stl` —— 正是本轮要重建的那个
    舵机**。所以「手指 vs 物体」这对几何是天然自洽的：舵机既是夹爪的壳，又是要夹的物体。
  · 动颚 = `moving_jaw_so101_v1.stl`（铰链颚，不是平行滑动颚 ⇒ 开口不是平移量）。
  · 闭合轴 `u = [-0.682, 0, +0.731]`（base 系），命令 0→100 时动颚质心走过 55.6mm。

**开口怎么量**：三个仪器都试过，前两个是假的，别再用：
  · 「两网格最小距离」→ 量到安装座上，实测恒为 ~0mm；
  · 「自由空间最大内切球」→ 夹持槽是**敞口**的，球会跑到夹爪外的空地上，实测恒为
    搜索上限 60mm，中心在夹爪外面；
  · 沿 ±u 从候选点射线**要求撞到不同的颚** → 找到的是切点不是槽壁（g=17.4 时
    "动颚墙"只有 0.25mm 远），曲线非单调。
铰链颚的开口本来就不是一个数：沿颚长是变的（近铰链宽、指尖窄），所以「命令→开口」
这条曲线**取决于物体坐在哪**。真正良定义、也是真正该问的问题只有一个：
**给定物体的位姿，它和两颚相交吗？** 那就是 `object_vs_jaws`，它不需要开口曲线。

**物体摆在夹爪系的哪里**：必须走 **FK ⊗ 手眼** `T_base_cam = fk(q,'wrist_camera_link') @ X`
（`X = calib/board_to_base.json:X_ee_cam`），不能用裸 FK —— 裸 FK 里没有相机连杆，
`so101_new_calib.urdf` 根本没有 camera link，`fk` 会**静默返回单位阵**。实测判决见
`--validate-anchor`：把腕视野里拟合出的手持方块（`calib/held_pose_wrist.json`）放到这条
链上，它离静颚 4.79mm、离动颚 2.03mm（g=17.4，边长 21.19mm 实测），**正好卡在两颚之间**，
且开大夹爪时动颚侧余量单调涨到 21.19mm（g=92）。这条链是自洽的。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
URDF_DIR = REPO / "workplace" / "so101_urdf"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


if str(URDF_DIR) not in sys.path:
    sys.path.insert(0, str(URDF_DIR))

MR_ = _load("_gc_meshrender", HERE / "meshrender.py")
A = _load("_gc_arm", HERE / "arm.py")
MFV = _load("_gc_mfv", HERE / "mesh_from_views.py")
VC = _load("_gc_vc", HERE / "validate_carve.py")           # voxelize_mesh 住在这里

DEFAULT_URDF = URDF_DIR / "so101_new_calib.urdf"
FIXED = "gripper_link"                       # 静颚（舵机壳）
MOVING = "moving_jaw_so101_v1_link"          # 动颚


def jaws_in_base(g: float, urdf_path=None, arm_norm=None):
    """两个颚的三角形，摆在 base 系。`g` 是归一化夹爪命令（0~100）。"""
    from fk_check import Urdf
    urdf_path = Path(urdf_path or DEFAULT_URDF)
    u = Urdf(urdf_path)
    meshes = MR_.urdf_link_meshes(urdf_path)
    arm = dict(arm_norm or {m: 0.0 for m in A.MOTORS})
    q = A.norm_to_q({**arm, "gripper": float(g)}, u)
    out = []
    for link in (FIXED, MOVING):
        T_link = u.fk(q, link)
        parts = []
        for T_vis, p in meshes[link]:
            v = MR_.stl_cached(p).reshape(-1, 3)
            T = T_link @ T_vis
            parts.append(v @ T[:3, :3].T + T[:3, 3])
        out.append(np.concatenate(parts).reshape(-1, 3, 3).astype(np.float64))
    return out


def _occ(tris, origin, h, shape):
    """网格落在一套格点上（只返回占据掩码，不关心内部/外部的一致性）。"""
    return VC.voxelize_mesh(tris, origin, h, shape, verbose=False)


# 夹持点（**实测校正过的**）：「方块几何中心在 gripper_frame_link 里」。
#
# 旧值 (5.9, 31.1, -17.4) 来自 `ball_swap.estimate_x_tcp_cube`（走 up 相机 + 板单应 + FK
# 那条链），它和**腕相机剪影拟合**那条链**差 42.19mm**，而且它的 y=31.1mm 落在两颚的
# y 范围（≤23.8mm）**之外** —— 早先三个「开口仪器」一直报「锚点在夹爪外面」，
# 不是仪器坏，是这个数错。
#
# 新值由 `--validate-anchor` 判出：走 FK⊗手眼把实测方块摆进去，它正好卡在两颚之间。
# 这条链（URDF 颚几何 + FK + 手眼 X + 腕剪影拟合）四者互相自洽，所以才采信它。
GRASP_LINK = "gripper_frame_link"
X_GRASP_MM = (-12.51, -3.10, -33.88)
# 与旧值的差，留着当数字报，不要悄悄改掉
X_GRASP_OLD_MM = (5.9, 31.1, -17.4)
# 手眼：T_base_cam = fk(q, 'wrist_camera_link') @ X_EE_CAM
CAM_URDF = URDF_DIR / "so101_new_calib_camera.urdf"
BOARD_TO_BASE = HERE / "calib" / "board_to_base.json"


def X_ee_cam():
    import json
    return np.array(json.loads(BOARD_TO_BASE.read_text())["X_ee_cam"], float)


def cam_frame(g: float, arm_norm=None, urdf_path=None):
    """腕相机的 `T_base_cam`：**FK ⊗ 手眼**。裸 FK 拿不到相机连杆（会返回单位阵）。"""
    from fk_check import Urdf
    u = Urdf(Path(urdf_path or CAM_URDF))
    arm = dict(arm_norm or {m: 0.0 for m in A.MOTORS})
    q = A.norm_to_q({**arm, "gripper": float(g)}, u)
    return u.fk(q, "wrist_camera_link") @ X_ee_cam(), q


def _dist_at(tf, tm, pts, h):
    """每个点（base 系，米）到两颚的**无符号**距离（米）。点必须落在颚外。"""
    allv = np.concatenate([tf.reshape(-1, 3), tm.reshape(-1, 3), np.atleast_2d(pts)])
    lo, hi = allv.min(0) - 0.01, allv.max(0) + 0.01
    shape = np.maximum(np.ceil((hi - lo) / h).astype(int), 2)
    of, om = _occ(tf, lo, h, shape), _occ(tm, lo, h, shape)
    df = ndimage.distance_transform_edt(~of) * h
    dm = ndimage.distance_transform_edt(~om) * h
    idx = np.clip(np.floor((np.atleast_2d(pts) - lo) / h).astype(int),
                  0, np.array(shape) - 1)
    j = tuple(idx.T)
    return df[j], dm[j], (of[j] | om[j])


def local_opening(g: float, x_link=None, h: float = 2.5e-4, urdf_path=None,
                  arm_norm=None, max_reach_m: float = 0.06):
    """从夹持点沿**闭合轴 u** 往两边射线步进，找各自第一堵墙 ⇒ 局部开口。

    比「两网格最小距离」或「自由空间最大内切球」都干脆：投影图上两颚分不分得开是看不出来
    的（点云投影会互相盖住），而射线只问「这个方向走多远撞到东西」。
    返回 dict(gap_mm=两墙距离, to_fixed_mm, to_moving_mm, hit_fixed, hit_moving)。
    """
    from fk_check import Urdf
    urdf_path = Path(urdf_path or DEFAULT_URDF)
    u = Urdf(urdf_path)
    arm = dict(arm_norm or {m: 0.0 for m in A.MOTORS})
    q = A.norm_to_q({**arm, "gripper": float(g)}, u)
    x = np.asarray(x_link if x_link is not None else X_GRASP_MM, float)
    if x_link is None:
        x = x * 1e-3
    P = (u.fk(q, GRASP_LINK) @ np.append(x, 1.0))[:3]
    tf, tm = jaws_in_base(g, urdf_path, arm_norm)
    # 闭合轴
    c0 = jaws_in_base(0.0, urdf_path, arm_norm)[1].reshape(-1, 3).mean(0)
    c1 = jaws_in_base(100.0, urdf_path, arm_norm)[1].reshape(-1, 3).mean(0)
    ux = (c1 - c0) / np.linalg.norm(c1 - c0)
    allv = np.concatenate([tf.reshape(-1, 3), tm.reshape(-1, 3)])
    lo = allv.min(0) - 0.02
    shape = np.maximum(np.ceil((allv.max(0) + 0.02 - lo) / h).astype(int), 2)
    of, om = _occ(tf, lo, h, shape), _occ(tm, lo, h, shape)

    def march(sign):
        n = int(max_reach_m / (h * 0.5))
        for k in range(1, n + 1):
            p = P + sign * k * h * 0.5 * ux
            j = tuple(np.clip(np.floor((p - lo) / h).astype(int), 0,
                              np.array(shape) - 1))
            if of[j]:
                return k * h * 0.5, "fixed"
            if om[j]:
                return k * h * 0.5, "moving"
        return max_reach_m, None

    dp, hp = march(+1.0)
    dn, hn = march(-1.0)
    both = {hp, hn} - {None}
    return dict(gap_mm=(dp + dn) * 1e3,
                to_plus_mm=dp * 1e3, to_minus_mm=dn * 1e3,
                hit_plus=hp, hit_minus=hn,
                gap_valid=(both == {"fixed", "moving"}),
                centre_base=P, u=ux)


def pocket_opening(g: float, h: float = 5.0e-4, urdf_path=None, arm_norm=None,
                   max_reach_m: float = 0.045):
    """**夹持槽**开口：两个颚之间那个「两边都撞得到、且撞的是不同颚」的最大净空。

    为什么不用夹持点：实测 `X_GRASP`（方块心在 gripper_frame_link 里）换到 base 后落在
    颚外 7mm（y=30.7 而两颚都 ≤23.8），沿闭合轴射线**两边都撞不到任何东西** —— 说明它
    不是「两指之间」那个点。与其猜坐标系，不如让几何自己说：

      候选点 = 两颚包围盒的**交集**（自动排除空地）→ 沿 ±u 步进 → 记住先撞到哪一颚。
      合格点 = 两个方向撞到**不同**的颚（=真的夹在中间）；开口 = 两墙距离的最大值。

    返回 dict(gap_mm, centre_base, hit_plus, hit_minus, to_plus_mm, to_minus_mm, n_ok)。
    向量化：所有候选点同步步进，比逐点快几百倍。
    """
    from fk_check import Urdf
    urdf_path = Path(urdf_path or DEFAULT_URDF)
    u = Urdf(urdf_path)
    arm = dict(arm_norm or {m: 0.0 for m in A.MOTORS})
    tf, tm = jaws_in_base(g, urdf_path, arm_norm)
    c0 = jaws_in_base(0.0, urdf_path, arm_norm)[1].reshape(-1, 3).mean(0)
    c1 = jaws_in_base(100.0, urdf_path, arm_norm)[1].reshape(-1, 3).mean(0)
    ux = (c1 - c0) / np.linalg.norm(c1 - c0)

    vf, vm = tf.reshape(-1, 3), tm.reshape(-1, 3)
    lo = np.minimum(vf.min(0), vm.min(0)) - 0.004
    hi = np.maximum(vf.max(0), vm.max(0)) + 0.004
    shape = np.maximum(np.ceil((hi - lo) / h).astype(int), 2)
    of, om = _occ(tf, lo, h, shape), _occ(tm, lo, h, shape)

    # 候选点 = 两颚**包围盒的交集**（空地自动排除），1mm 粗搜
    ilo = np.maximum(vf.min(0), vm.min(0)) - 0.002
    ihi = np.minimum(vf.max(0), vm.max(0)) + 0.002
    grids = [np.arange(ilo[k], ihi[k], 1.0e-3) for k in range(3)]
    G = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
    if len(G) == 0:
        return dict(gap_mm=0.0, centre_base=None, n_ok=0)

    # 两个方向各扫一遍：所有点同步步进，第一个撞上东西的点就此定住
    step = h * 0.5
    n = int(max_reach_m / step)
    dP = np.full(len(G), np.nan)
    dM = np.full(len(G), np.nan)
    wP = np.zeros(len(G), np.int8)
    wM = np.zeros(len(G), np.int8)
    for sign, dd, ww in ((+1.0, dP, wP), (-1.0, dM, wM)):
        live = np.ones(len(G), bool)
        for k in range(1, n + 1):
            if not live.any():
                break
            P = (G + sign * k * step * ux)[live]
            idx = np.floor((P - lo) / h).astype(int)
            ok = np.all((idx >= 0) & (idx < np.array(shape)), 1)
            j = tuple(np.clip(idx, 0, np.array(shape) - 1).T)
            hf = ok & of[j]
            hm = ok & (om[j] & ~hf)
            any_hit = hf | hm
            idxs = np.flatnonzero(live)[any_hit]
            dd[idxs] = k * step
            ww[idxs] = np.where(hf[any_hit], 1, 2)
            live[idxs] = False
    ok_pt = np.isfinite(dP) & np.isfinite(dM) & (wP != wM) & (wP > 0) & (wM > 0)
    if not ok_pt.any():
        return dict(gap_mm=0.0, centre_base=None, n_ok=0)
    gap = (dP + dM)[ok_pt]
    k = int(np.argmax(gap))
    gsel = np.flatnonzero(ok_pt)
    c = G[gsel[k]]
    return dict(gap_mm=float(gap[k]) * 1e3, centre_base=c, n_ok=int(ok_pt.sum()),
                hit_plus=("fixed" if wP[gsel[k]] == 1 else "moving"),
                hit_minus=("fixed" if wM[gsel[k]] == 1 else "moving"),
                to_plus_mm=float(dP[gsel[k]]) * 1e3,
                to_minus_mm=float(dM[gsel[k]]) * 1e3, u=ux)


def opening_at(g: float, x_link=None, h: float = 5.0e-4, urdf_path=None,
               arm_norm=None):
    """以**夹持点**为中心：该命令下能塞进去的最大球直径 = 开口。

    `x_link` 是夹持点在 `GRASP_LINK` 里的坐标（米），默认用实测的方块中心。
    """
    from fk_check import Urdf
    urdf_path = Path(urdf_path or DEFAULT_URDF)
    u = Urdf(urdf_path)
    arm = dict(arm_norm or {m: 0.0 for m in A.MOTORS})
    q = A.norm_to_q({**arm, "gripper": float(g)}, u)
    x = np.asarray(x_link if x_link is not None else X_GRASP_MM, float)
    if x_link is None:
        x = x * 1e-3
    p = (u.fk(q, GRASP_LINK) @ np.append(x, 1.0))[:3]
    tf, tm = jaws_in_base(g, urdf_path, arm_norm)
    df, dm, inside = _dist_at(tf, tm, p, h)
    return dict(width_m=2.0 * float(min(df[0], dm[0])),
                d_fixed_m=float(df[0]), d_moving_m=float(dm[0]),
                inside=bool(inside[0]), centre_base=p, centre_link=x)


def opening_curve(gs=(0, 8, 17.4, 30, 50, 70, 92, 100), **kw):
    rows = []
    for g in gs:
        d = opening_at(g, **kw)
        rows.append((float(g), d["width_m"], d["centre_base"]))
    return rows


def solve_command(width_m: float, rows) -> float | None:
    """反解：宽 `width_m` 的物体对应哪个命令（线性插值，超出范围返回 None）。"""
    gs = np.array([r[0] for r in rows])
    w = np.array([r[1] for r in rows])
    if width_m < w.min() or width_m > w.max():
        return None
    return float(np.interp(width_m, w, gs))


def object_vs_jaws(obj_tris, g: float, h: float = 5.0e-4, urdf_path=None,
                   arm_norm=None):
    """物体已经摆好（base 系）时：与两颚的**相交体积**、在闭合轴上的余量。

    返回 dict(intersect_mm3, intersect_fixed_mm3, intersect_moving_mm3,
              clearance_mm, radius_mm, centre)。`clearance_mm` 是物体表面到最近颚的
    最小距离（物体没和颚相交时才有意义）。
    """
    tf, tm = jaws_in_base(g, urdf_path, arm_norm)
    allv = np.concatenate([obj_tris.reshape(-1, 3), tf.reshape(-1, 3),
                           tm.reshape(-1, 3)])
    lo, hi = allv.min(0) - 0.01, allv.max(0) + 0.01
    shape = np.maximum(np.ceil((hi - lo) / h).astype(int), 2)
    ov = _occ(obj_tris, lo, h, shape)
    of = _occ(tf, lo, h, shape)
    om = _occ(tm, lo, h, shape)
    vox = h ** 3 * 1e9                                     # mm³/格
    if_f, if_m = int((ov & of).sum()), int((ov & om).sum())
    if if_f or if_m:
        clr = -1.0
    else:
        df = ndimage.distance_transform_edt(~of) * h
        dm = ndimage.distance_transform_edt(~om) * h
        clr = float(np.min(np.minimum(df, dm)[ov])) * 1e3
    return dict(intersect_mm3=(if_f + if_m) * vox,
                intersect_fixed_mm3=if_f * vox, intersect_moving_mm3=if_m * vox,
                clearance_mm=clr)


def box_tris(side_m, T):
    """边长 `side_m` 的立方体，按 `T`（4×4）摆放，返回三角汤。"""
    h = side_m / 2
    cor = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1)
                    for k in (-1, 1)], float) * h
    F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    tris = np.array([[cor[i], cor[j], cor[k]] for i, j, k in F])
    return tris @ T[:3, :3].T + T[:3, 3]


def anchor_pose(g: float = 17.4, arm_norm=None):
    """实测夹持锚点的 4×4 位姿（base 系）：`X_GRASP_MM` @ `GRASP_LINK`。"""
    from fk_check import Urdf
    arm = dict(arm_norm or {m: 0.0 for m in A.MOTORS})
    T = Urdf(DEFAULT_URDF).fk(A.norm_to_q({**arm, "gripper": float(g)},
                                          Urdf(DEFAULT_URDF)), GRASP_LINK)
    T = T.copy()
    T[:3, 3] = (T @ np.append(np.array(X_GRASP_MM) * 1e-3, 1.0))[:3]
    return T


def validate_anchor(h: float = 5.0e-4):
    """判决实验：实测的**手持方块**必须卡在两颚之间。闸门靠这一条才算标定过。

    走 `FK ⊗ 手眼`（不是裸 FK —— 裸 FK 里没有相机连杆，会静默返回单位阵）。
    """
    import json
    d = json.loads((HERE / "calib" / "held_pose_wrist.json").read_text())
    T_cam_cube = np.array(d["T_cam_cube"], float)
    edge = float(d["cube_edge_mm"]) * 1e-3
    print(f"实测手持方块（腕相机系剪影拟合）：边长 {edge*1e3:.2f}mm，"
          f"质心 {np.round(T_cam_cube[:3, 3]*1e3, 2)}mm")
    T_base_cam, _ = cam_frame(17.4)
    cube = box_tris(edge, T_base_cam @ T_cam_cube)
    cb = cube.reshape(-1, 3).mean(0) * 1e3
    print(f"  → base 系质心 {np.round(cb, 2)}mm")
    # 折算到 project 一直用的锚点系里，和旧值对表
    from fk_check import Urdf
    q = A.norm_to_q({**{m: 0.0 for m in A.MOTORS}, "gripper": 17.4}, Urdf(DEFAULT_URDF))
    rel = (np.linalg.inv(Urdf(DEFAULT_URDF).fk(q, GRASP_LINK))
           @ np.append(cube.reshape(-1, 3).mean(0), 1.0))[:3] * 1e3
    err = np.linalg.norm(np.array(X_GRASP_OLD_MM) - rel)
    print(f"  在 {GRASP_LINK} 系里 {np.round(rel, 2)}mm"
          f"｜旧值 {X_GRASP_OLD_MM} ⇒ 差 **{err:.2f}mm**")
    print(f"\n{'命令 g':>7} {'到静颚':>9} {'到动颚':>9}   （mm；必须都 >0，且静颚侧恒定）")
    from scipy.spatial import cKDTree
    for g in (0.0, 8.0, 17.4, 30.0, 50.0, 70.0, 92.0):
        # 全在 **base 系**里比：方块走 FK⊗手眼搬过来，两颚本来就是 base 系
        Tbg, _ = cam_frame(g)
        cub = box_tris(edge, Tbg @ T_cam_cube).reshape(-1, 3)
        tf, tm = jaws_in_base(g)
        da = cKDTree(tf.reshape(-1, 3)[::10]).query(cub)[0].min() * 1e3
        db = cKDTree(tm.reshape(-1, 3)[::10]).query(cub)[0].min() * 1e3
        print(f"{g:7.1f} {da:9.2f} {db:9.2f}")
    print("\n读法：静颚不动 ⇒ 该列应**恒定**；动颚是铰链颚 ⇒ 开大夹爪该列应**单调涨**。"
          "\ng=17.4 就是搬运段的真实命令，此刻方块贴着两颚（2~5mm，在拟合精度内）"
          " ⇒ 链自洽。")


def cube_rotations() -> np.ndarray:
    """24 个标准正交旋转（立方体旋转群）—— 给 R 搜索一个不含糊的离散起点集。"""
    R = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for s in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                  (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
            M = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, s)):
                M[i, p] = sg
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                R.append(M)
    return np.array(R)


def feasible_over_rotations(mesh, gs, h: float = 5.0e-4, scale: float = 1.0,
                            n_rand: int = 0, seed: int = 0, decim_m: float = 0.0):
    """**R 搜索下的闸门**：物体绕自身原点转，报每个 g 下**能做到的最小相交体积**。

    两颚的占据只体素化一次（与 R 无关），所以这一圈很便宜。返回
    {g: (min_vox_mm3, R_best, moving_mm3, fixed_mm3)}。物体原点仍是自身中心。

    `decim_m` > 0 时先对物体减面 —— 相交**体积**对减面不敏感（轮廓才是），但体素化快
    一个数量级；R 搜索只关心「哪个朝向最少重叠」，用减面网格是合适的。
    """
    if isinstance(mesh, (str, Path)):
        tris = np.asarray(MR_.stl_cached(Path(mesh)), float).reshape(-1, 3, 3)
    else:
        tris = np.asarray(mesh, float).reshape(-1, 3, 3)
    if decim_m:
        tris = np.asarray(MR_.decimate(tris, float(decim_m)), float).reshape(-1, 3, 3)
    tris = tris * scale
    Rs = list(cube_rotations())
    if n_rand:
        rng = np.random.default_rng(seed)
        for _ in range(n_rand):
            A_ = rng.normal(size=(3, 3))
            Q, _ = np.linalg.qr(A_)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            Rs.append(Q)
    origin_local = tris.reshape(-1, 3).mean(0)          # 自身质心（原点约定）
    tris0 = tris - origin_local
    out = {}
    for g in gs:
        tf, tm = jaws_in_base(g)
        T = anchor_pose(g)
        # 包围盒要盖住**所有**候选朝向的物体，否则某些朝向会被裁掉
        rad = np.abs(tris0.reshape(-1, 3)).max()
        allv = np.concatenate([tf.reshape(-1, 3), tm.reshape(-1, 3),
                               (np.array([[1, 1, 1], [-1, -1, -1]]) * rad) + T[:3, 3]])
        lo, hi = allv.min(0) - 0.005, allv.max(0) + 0.005
        shape = np.maximum(np.ceil((hi - lo) / h).astype(int), 2)
        of = _occ(tf, lo, h, shape)
        om = _occ(tm, lo, h, shape)
        vox = h ** 3 * 1e9
        best = None
        for R in Rs:
            tris_g = (tris0 @ R.T) @ T[:3, :3].T + T[:3, 3]
            ov = _occ(tris_g, lo, h, shape)
            nf, nm = int((ov & of).sum()), int((ov & om).sum())
            if best is None or (nf + nm) < (best[1] + best[2]):
                best = (R, nf * vox, nm * vox)
        out[float(g)] = dict(min_mm3=best[1] + best[2], R=best[0],
                             moving_mm3=best[2], fixed_mm3=best[1])
    return out


def check_mesh(mesh, g: float, h: float = 5.0e-4, scale: float = 1.0):
    """**闸门本体**：把一个网格按实测夹持位姿摆好，报它和两颚的相交与余量。

    `mesh` 可以是 STL 路径（重建产物），或 `(N,3,3)` 三角汤（如解析球对照臂）。
    网格的**自身原点**就是物体中心 —— 这正是用户要查的那条（原点/尺寸/相对夹爪的变换）。
    """
    if isinstance(mesh, (str, Path)):
        tris = np.asarray(MR_.stl_cached(Path(mesh)), float).reshape(-1, 3, 3)
    else:
        tris = np.asarray(mesh, float).reshape(-1, 3, 3)
    tris = tris * scale                              # 绕自身原点缩放
    T = anchor_pose(g)
    tris = tris @ T[:3, :3].T + T[:3, 3]
    return object_vs_jaws(tris, g, h=h)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="夹持可行性闸门（默认跑判决实验）")
    ap.add_argument("--validate-anchor", action="store_true",
                    help="判决实验：实测手持方块 vs 两颚（默认动作）")
    ap.add_argument("--mesh", default=None, help="要过闸门的网格（STL），按实测夹持位姿摆")
    ap.add_argument("--sphere-mm", type=float, default=None,
                    help="对照组：直径这么多毫米的解析球")
    ap.add_argument("--search-rot", action="store_true",
                    help="在 24 个立方体旋转里搜 R，报每个 g 下**最小**相交体积")
    ap.add_argument("--n-rand", type=int, default=0, help="额外随机 SO(3) 采样数")
    ap.add_argument("--decim-mm", type=float, default=0.0,
                    help="先对网格减面（mm）；R 搜索建议 2.0")
    ap.add_argument("--scale", type=float, default=1.0, help="网格缩放（s）")
    ap.add_argument("--opening", action="store_true",
                    help="**已证伪**的「命令→开口」曲线，只留作对照")
    ap.add_argument("--gs", default="0,8,17.4,30,50,70,92,100")
    ap.add_argument("--h", type=float, default=5.0e-4)
    ap.add_argument("--solve", default="21.5,24.0,24.8",
                    help="反解这些宽度（mm）对应的命令")
    a = ap.parse_args()
    gs = [float(s) for s in a.gs.split(",")]

    if a.mesh or a.sphere_mm:
        if a.sphere_mm:
            obj, nam = MR_.uv_sphere(a.sphere_mm * 5e-4), f"解析球 ⌀{a.sphere_mm:g}mm"
        else:
            obj, nam = a.mesh, f"{Path(a.mesh).name}（scale={a.scale:g}）"
        if a.search_rot:
            print(f"闸门（R 搜索：24 个立方体旋转"
                  f"{f' + {a.n_rand} 随机' if a.n_rand else ''}）：{nam}"
                  f"｜体素 {a.h*1e3:g}mm")
            res = feasible_over_rotations(obj, gs, h=a.h,
                                         scale=1.0 if a.sphere_mm else a.scale,
                                         n_rand=a.n_rand, decim_m=a.decim_mm * 1e-3)
            print(f"\n{'命令 g':>8} {'最小相交(mm³)':>14} {'静颚':>10} {'动颚':>10}"
                  f"   {'最优 R（行=物体轴在夹爪系）':>10}")
            for g in gs:
                r = res[float(g)]
                ok = "可行" if r["min_mm3"] <= 0 else "**仍相交**"
                print(f"{g:8.1f} {r['min_mm3']:14.2f} {r['fixed_mm3']:10.2f}"
                      f" {r['moving_mm3']:10.2f}   {ok}"
                      f"  {np.round(r['R'], 2).tolist()}")
            return
        print(f"闸门：{nam} @ 实测夹持位姿｜体素 {a.h*1e3:g}mm")
        print(f"\n{'命令 g':>8} {'相交体积(mm³)':>14} {'静颚':>10} {'动颚':>10} {'余量(mm)':>10}")
        for g in gs:
            r = check_mesh(obj, g, h=a.h, scale=1.0 if a.sphere_mm else a.scale)
            print(f"{g:8.1f} {r['intersect_mm3']:14.2f} "
                  f"{r['intersect_fixed_mm3']:10.2f} {r['intersect_moving_mm3']:10.2f}"
                  f" {r['clearance_mm']:10.2f}"
                  + ("   ← **与手指相交**" if r["intersect_mm3"] > 0 else ""))
        return

    if a.opening:
        print("!! 这个「命令→开口」曲线**已被证伪**：(a) 与经验锚不符（g=17.4 应≈21.5mm），"
              "\n!! (b) 非单调（g=70 报 79.5mm、g=92 塌到 28.25mm）—— 找到的是切点不是槽壁。"
              "\n!! 铰链颚的开口沿颚长是变的，不是命令的单一函数。只留作对照。\n")
        print(f"URDF {DEFAULT_URDF.name}｜静颚 {FIXED}（= sts3215_03a_v1.stl）"
              f"｜动颚 {MOVING}｜体素 {a.h*1e3:g}mm")
    print(f"\n{'命令 g':>8} {'开口(mm)':>10} {'+u撞谁':>9} {'−u撞谁':>9}"
          f" {'到+墙':>7} {'到−墙':>7} {'合格点数':>9}")
    rows = []
    for g in gs:
        d = pocket_opening(g, h=a.h)
        rows.append((float(g), d["gap_mm"] * 1e-3, d["centre_base"]))
        if d["n_ok"] == 0:
            print(f"{g:8.1f} {'—':>10}   **没有找到夹持槽**（没有点能沿 ±u 撞到不同的颚）")
            continue
        print(f"{g:8.1f} {d['gap_mm']:10.2f} {str(d['hit_plus']):>9} "
              f"{str(d['hit_minus']):>9} {d['to_plus_mm']:7.2f} {d['to_minus_mm']:7.2f}"
              f" {d['n_ok']:9d}")
    ws = [float(s) for s in a.solve.split(",")]
    print()
    for w in ws:
        g = solve_command(w * 1e-3, rows)
        print(f"  宽 {w:5.1f}mm → " + (f"命令 {g:5.1f}" if g is not None
                                       else f"**超出开口 {rows[0][1]*1e3:.1f}~"
                                            f"{rows[-1][1]*1e3:.1f}mm**"))
    lohi = (rows[0][1] * 1e3, rows[-1][1] * 1e3)
    print(f"\n可用开口范围（命令 {gs[0]:g}~{gs[-1]:g}）= {lohi[0]:.1f}~{lohi[1]:.1f}mm；"
          f"命令被夹在 [8,92] ⇒ 实际可用 "
          f"{2*np.interp(8, [r[0] for r in rows], [r[1]/2 for r in rows])*1e3:.1f}~"
          f"{2*np.interp(92, [r[0] for r in rows], [r[1]/2 for r in rows])*1e3:.1f}mm")
    return


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) == 1 or "--validate-anchor" in _s.argv:
        validate_anchor()
    else:
        main()
