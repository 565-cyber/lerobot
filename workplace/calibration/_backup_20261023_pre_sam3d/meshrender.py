#!/usr/bin/env python
"""网格渲染器：把三角网格按**真相机内参**渲进图像，产出 深度/法线/掩码。

为什么要它：`workplace/so101_urdf/render_urdf.py` 已经能做「整臂按 FK 摆位 + z-buffer
光栅化」，但那是为了**肉眼核对运动学**——用 fov 造相机、只出颜色、平面着色、没有畸变。
做 Pose6DAug 那套（网格 + 6D 位姿 + 遮挡合成）需要的是另一组东西：

  * 相机是**标定出来的**（`calib/front_intrinsics.json` 的 K + 5 参数畸变，rms 0.30px）；
  * 产出**深度缓冲**和**法线**，因为遮挡合成靠深度排序，着色靠法线；
  * 网格可以**按连杆挑**（遮挡层只要夹爪那几个），并且能**减面**——
    整臂 ~38 万个面，逐面 Python 循环撑不住位姿拟合的迭代。

本机没网、装不了 trimesh/pyrender/nvdiffrast（实测 PyPI 不通），所以自写。
三角面数降到几千、逐面循环、面内向量化——对这个规模够用。

坐标约定（与 `ball_swap.project` 一致，别混）：
    `Cam.T_base_cam` 是**相机在 base 系里的位姿**（4×4, cam→base）。
    投影 = cv2.projectPoints(inv(T_base_cam) @ X_base, 0, 0, K, dist)。

跑法：
    python workplace/calibration/meshrender.py --selftest      # 自检 + 出图
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
URDF_DIR = REPO / "workplace/so101_urdf"
FIG = HERE / "figures"

if str(URDF_DIR) not in sys.path:                 # fk_check / render_urdf 住在那儿
    sys.path.insert(0, str(URDF_DIR))

from fk_check import rpy_to_R, Urdf                # noqa: E402
from render_urdf import load_stl                   # noqa: E402


# ---------------------------------------------------------------------- 网格

_STL_CACHE: dict[Path, np.ndarray] = {}


def stl_cached(path: Path) -> np.ndarray:
    if path not in _STL_CACHE:
        _STL_CACHE[path] = load_stl(path)
    return _STL_CACHE[path]


def urdf_link_meshes(urdf_path: Path, visual_only: bool = True) -> dict:
    """连杆名 -> [(相对连杆系的 4×4, stl 路径), ...]。

    只取 `visual`：`render_urdf.link_meshes` 把 visual 和 collision 都收进来，
    同一个 STL 被算了**两遍**（面数翻倍），渲染不需要这份冗余。
    """
    root = ET.parse(urdf_path).getroot()
    tags = ("visual",) if visual_only else ("visual", "collision")
    out: dict[str, list] = {}
    for link in root.findall("link"):
        for tag in tags:
            for el in link.findall(tag):
                mesh = el.find("geometry/mesh")
                if mesh is None or not mesh.get("filename"):
                    continue
                p = URDF_DIR / "assets" / Path(mesh.get("filename")).name
                if not p.exists():
                    continue
                o = el.find("origin")
                xyz = np.array([float(v) for v in ((o.get("xyz") if o is not None
                                                    else None) or "0 0 0").split()])
                rpy = np.array([float(v) for v in ((o.get("rpy") if o is not None
                                                    else None) or "0 0 0").split()])
                T = np.eye(4)
                T[:3, :3] = rpy_to_R(rpy)
                T[:3, 3] = xyz
                s = mesh.get("scale")
                if s:
                    T = T @ np.diag(np.append([float(v) for v in s.split()], 1.0))
                out.setdefault(link.get("name"), []).append((T, p))
    return out


def gather(urdf_path: Path, q: dict, links=None, cell: float | None = None):
    """把（挑出来的）连杆网格摆到 **base 系**，返回 (M,3,3) 三角形。

    links=None 表示整臂；cell 给了就按该边长（米）做顶点聚类减面。
    """
    root_tris, used = [], []
    meshes = urdf_link_meshes(urdf_path)
    u = Urdf(urdf_path)
    for link, items in meshes.items():
        if links is not None and link not in links:
            continue
        if link not in u.parent_link:
            continue                                       # 不在 base 的链上
        T_link = u.fk(q, link)
        got = False
        for T_vis, p in items:
            v = stl_cached(p)
            T = T_link @ T_vis
            w = (v.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]).reshape(-1, 3, 3)
            root_tris.append(w.astype(np.float32))
            got = True
        if got:
            used.append(link)
    if not root_tris:
        raise SystemExit(f"{urdf_path.name}: 没有任何 mesh 落在 base 链上（links={links}）")
    tris = np.concatenate(root_tris)
    if cell:
        tris = decimate(tris, cell)
    return tris, used


# 相机自身那两个 link 永远不该出现在渲染里：
#   1) 物理上相机看不见自己；
#   2) 更要紧的是 URDF 里那块相机网格**根本是错的**——它比相机真光心靠前 ~30mm
#      （手眼 X_ee_cam 那 30mm 系统差就来自这里），网格整个落在 z=+4~38mm 上，
#      投影后一个面就能盖掉 87% 画面（实测 _m2.py）。
# 遮挡层是要眼睛/夹爪这类**真会挡住的**几何，不是相机自己。
SELF_LINKS = ("wrist_camera_link", "wrist_camera_mount_link")


def arm_tris(urdf_path: Path, q: dict, cell: float | None = None,
             links=None, drop_self: bool = True):
    """整臂（或指定连杆）摆到 base 系，默认**剔掉相机自身的两个 link**。

    这就是 render_urdf.gather 加上「相机看不见自己」这一条；遮挡层、剪影拟合、
    出图都走这个入口，别再直接调 gather（除非确实要连相机机身一起渲，比如画
    CAD 装配图）。
    """
    if links is None and drop_self:
        links = set(urdf_link_meshes(urdf_path)) - set(SELF_LINKS)
    return gather(urdf_path, q, links=links, cell=cell)


def decimate(tris: np.ndarray, cell: float) -> np.ndarray:
    """顶点聚类减面：把顶点吸到 cell 边长的格子上，丢掉退化面。

    不是最优（不保特征），但对「几千面的遮挡层」够用，且不用装任何库。
    cell=0.003（3mm）时夹爪从十几万面掉到几千面。
    """
    v = tris.reshape(-1, 3)
    key = np.floor(v / cell).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    reps = np.zeros((len(counts), 3), np.float64)
    np.add.at(reps, inv, v)
    reps /= counts[:, None]
    t = inv.reshape(-1, 3)
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return reps[t[keep]].astype(np.float32)


# ------------------------------------------------------------------ 基本图元

def uv_sphere(radius: float, n_lat: int = 24, n_lon: int = 48) -> np.ndarray:
    """UV 球 → (2*n_lat*n_lon, 3, 3)。这就是「换上去的那个物体」的网格。"""
    th = np.linspace(0, np.pi, n_lat + 1)
    ph = np.linspace(0, 2 * np.pi, n_lon + 1)

    def P(i, j):
        return np.array([np.sin(th[i]) * np.cos(ph[j]),
                         np.sin(th[i]) * np.sin(ph[j]),
                         np.cos(th[i])]) * radius
    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b, c, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)
            tris += [np.stack([a, b, c]), np.stack([a, c, d])]
    return np.asarray(tris, np.float32)


def box(sx: float, sy: float, sz: float) -> np.ndarray:
    """轴对齐长方体（中心在原点）→ (12,3,3)，绕序朝外。"""
    s = np.array([sx, sy, sz]) / 2
    v = np.array([[i, j, k] for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)],
                 float) * s
    # 顶点编号: bit0=x, bit1=y, bit2=z（-1/1）
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
         (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in f:
        tris += [np.stack([v[a], v[b], v[c]]), np.stack([v[a], v[c], v[d]])]
    t = np.asarray(tris, np.float32)
    # 让法线朝外（原点指向面心）
    ctr = t.mean(1)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (nrm * ctr).sum(1) < 0
    t[flip] = t[flip][:, ::-1]
    return t


def transform(tris: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (tris.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]).reshape(-1, 3, 3)


# -------------------------------------------------------------------- 相机

class Cam:
    """针孔 + 畸变。`T_base_cam` = 相机在 base 系里的位姿（cam→base）。"""

    def __init__(self, K, dist, T_base_cam, W=640, H=480):
        self.K = np.asarray(K, float)
        self.dist = np.asarray(dist, float).reshape(-1)
        self.T_base_cam = np.asarray(T_base_cam, float)
        self.T_cam_base = np.linalg.inv(self.T_base_cam)
        self.W, self.H = int(W), int(H)

    def project(self, P_base):
        """(...,3) base 点 → (uv (...,2), z_cam (...,))。走 cv2，含畸变。"""
        P = np.asarray(P_base, float).reshape(-1, 3)
        Pc = P @ self.T_cam_base[:3, :3].T + self.T_cam_base[:3, 3]
        uv, _ = cv2.projectPoints(Pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                  self.K, self.dist)
        return uv.reshape(-1, 2), Pc[:, 2]


# ------------------------------------------------------------------ 光栅化

def clip_tri_near(Pc: np.ndarray, near: float) -> np.ndarray:
    """Sutherland–Hodgman：把 (3,3) 相机系三角形裁到 z ≥ near 一侧。

    返回 (n,3,3)，n = 0（整块在近平面里）/1/2（三角形跨越近平面，裁成两个）。
    绕序不变（SH 保持顶点顺序），所以裁完还能接着做屏幕面积背面剔除。
    """
    res = []
    for i in range(3):
        a, b = Pc[i], Pc[(i + 1) % 3]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            res.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            res.append(a + t * (b - a))
    if len(res) < 3:
        return np.zeros((0, 3, 3))
    v = np.asarray(res, float)
    return np.stack([v[[0, i, i + 1]] for i in range(1, len(v) - 1)])


def rasterize(tris, cam: Cam, cull: bool = False, max_depth: float = 5.0,
              near: float = 0.015):
    """z-buffer 光栅化 → dict(depth, mask, normal, tri)。

    depth 单位是**相机系 z（米）**，越大越远；mask 是覆盖到的像素。
    法线存的是 **base 系**法线（着色用世界光方向）。

    ★ 近平面必须**真裁**（Sutherland–Hodgman），不能只做「整块在里面才画」的判据。
      原因：腕相机是长在手腕上的，`wrist_roll` / 相机支架的网格就**包着光心**，
      顶点深度从 −30mm 一直铺到 +70mm。只判「三顶点都 > near」挡不住
      「z=15mm 但横向 60mm 宽」那种面 —— 投影半径 409·60/15 ≈ 1600px，一个三角形
      铺满全图。先前就是这么踩的：整臂覆盖率 96%，渲染出来是一张放射状的「蛛网」。
      没裁之前还查到一个更脏的：`wrist_camera_mount_so101_v1.stl` 整个文件是**毫米**
      单位（外接盒 66.1 m × 42.6 m × 35.0 m，其余 17 个 STL 都是 20~140mm），
      一放进来就是一颗 66 米的巨物包住相机 —— 所以相机自身那两个 link 直接排除，
      见 `SELF_LINKS`。
    """
    H, W = cam.H, cam.W
    if len(tris) == 0:
        return {"depth": np.full((H, W), np.inf, np.float32),
                "mask": np.zeros((H, W), bool), "normal": np.zeros((H, W, 3), np.float32),
                "tri": np.full((H, W), -1, np.int32)}

    Pc = tris.reshape(-1, 3) @ cam.T_cam_base[:3, :3].T + cam.T_cam_base[:3, 3]
    Pc = Pc.reshape(-1, 3, 3)
    zc = Pc[:, :, 2]

    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

    ok = (zc >= near).all(1)                       # 整块在近平面之后，直接画
    cut = ~ok & (zc >= near).any(1)                # 跨近平面，要裁
    parts, srcs = [Pc[ok]], [np.nonzero(ok)[0]]
    if cut.any():
        for i in np.nonzero(cut)[0]:
            f = clip_tri_near(Pc[i], near)
            if len(f):
                parts.append(f)
                srcs.append(np.full(len(f), i, np.int64))
    Pf = np.concatenate(parts)
    src = np.concatenate(srcs)
    if len(Pf) == 0:
        return {"depth": np.full((H, W), np.inf, np.float32),
                "mask": np.zeros((H, W), bool), "normal": np.zeros((H, W, 3), np.float32),
                "tri": np.full((H, W), -1, np.int32)}

    uv, _ = cv2.projectPoints(Pf.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.dist)
    uv = uv.reshape(-1, 2)
    sx, sy = uv[:, 0].reshape(-1, 3), uv[:, 1].reshape(-1, 3)
    zf = Pf[:, :, 2].astype(np.float64)

    depth = np.full((H, W), np.inf, np.float32)
    nbuf = np.zeros((H, W, 3), np.float32)
    triid = np.full((H, W), -1, np.int32)

    if cull:
        # 屏幕系有向面积（和 render_urdf 一致：正 = 面朝相机）
        area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0])
                - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
        idx = np.nonzero(area > 0.5)[0]
    else:
        idx = np.arange(len(Pf))

    yy, xx = np.mgrid[0:H, 0:W]
    for i in idx:
        x0, x1, x2 = sx[i]
        y0, y1, y2 = sy[i]
        lo_x = max(int(math.floor(min(x0, x1, x2))), 0)
        hi_x = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        lo_y = max(int(math.floor(min(y0, y1, y2))), 0)
        hi_y = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        if lo_x > hi_x or lo_y > hi_y:
            continue
        px = xx[lo_y:hi_y + 1, lo_x:hi_x + 1] + 0.5
        py = yy[lo_y:hi_y + 1, lo_x:hi_x + 1] + 0.5
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-12:
            continue
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d
        w2 = 1.0 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any():
            continue
        zi = w0 * zf[i, 0] + w1 * zf[i, 1] + w2 * zf[i, 2]
        sub_z = depth[lo_y:hi_y + 1, lo_x:hi_x + 1]
        upd = m & (zi < sub_z)
        if not upd.any():
            continue
        sub_z[upd] = zi[upd]
        triid[lo_y:hi_y + 1, lo_x:hi_x + 1][upd] = src[i]
        nbuf[lo_y:hi_y + 1, lo_x:hi_x + 1][upd] = nrm[src[i]]
    mask = np.isfinite(depth) & (depth < max_depth)
    return {"depth": depth, "mask": mask, "normal": nbuf, "tri": triid}


# -------------------------------------------------------------------- 着色

class Light:
    """Phong：环境 + 一个平行光 + 一个镜面项。方向 `dir` 是**从物体指向光源**（base 系）。"""

    def __init__(self, direction=(0.3, -0.4, 0.87), ambient=0.35, diffuse=0.75,
                 specular=0.25, shininess=24.0, up_bright_gain=1.0):
        d = np.asarray(direction, float)
        self.dir = d / np.linalg.norm(d)
        self.ambient, self.diffuse = float(ambient), float(diffuse)
        self.specular, self.shininess = float(specular), float(shininess)

    def to_dict(self):
        return {"direction": self.dir.tolist(), "ambient": self.ambient,
                "diffuse": self.diffuse, "specular": self.specular,
                "shininess": self.shininess}

    @staticmethod
    def from_dict(d):
        return Light(d["direction"], d["ambient"], d["diffuse"], d["specular"],
                     d["shininess"])


def shade(ras, cam: Cam, light: Light, albedo, view_fn=None):
    """深度+法线 → 着色图（float32 BGR，0~1）。

    ★ 法线先按「朝向观察者」定一次向，再算 **单面** 漫反射 `max(0, n·l)`；
      **不要图省事取 `|n·l|`**（STL 绕序不保证时看着能糊弄）。球面上 `|n·l|` 是错的：
      可见半球上它有两个极大（法线朝光、背光各一个），于是球被渲成**一只碗** ——
      上下各一道亮弧、中间一条暗带（2026-10-14 实测：f336/f400 的球一眼就是碗，
      连高光都落在「碗底」）。按视线定向后等价于单面着色，球和机械臂都对。
    """
    m = ras["mask"]
    n = (ras["normal"] / np.maximum(
        np.linalg.norm(ras["normal"], axis=2, keepdims=True), 1e-12)).copy()
    img = np.zeros((cam.H, cam.W, 3), np.float32)
    alb = np.asarray(albedo, np.float32).reshape(1, 1, 3)
    img[:] = alb * light.ambient
    ys, xs = np.nonzero(m)
    if not len(xs):
        return img
    if view_fn is None:
        # **朝向观察者**的单位向量（表面 → 相机）。注意 `Pc` 按 (x-cx)/fx 张出来的是
        # **相机指向场景**的向量，要取负号 —— 一开始没取，法线全被翻到背面，
        # 球就变成「背光」的一只碗（亮在下方、暗在上方，正好左右颠倒）。
        Pc = np.stack([(xs - cam.K[0, 2]) / cam.K[0, 0],
                       (ys - cam.K[1, 2]) / cam.K[1, 1],
                       np.ones_like(xs, float)], 1)
        vw = -((Pc / np.linalg.norm(Pc, axis=1, keepdims=True)) @ cam.T_base_cam[:3, :3].T)
    else:
        vw = np.asarray(view_fn(xs, ys), float)      # 约定同样是「表面 → 相机」
    nv = n[ys, xs]
    nv *= np.where((nv * vw).sum(1, keepdims=True) < 0, -1.0, 1.0)      # 朝观察者
    lam = np.clip((nv * light.dir).sum(1), 0, None)
    img[ys, xs] = (alb * (light.ambient + light.diffuse * lam[:, None]))
    if light.specular > 0:
        h = vw + light.dir
        h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
        sp = np.clip((nv * h).sum(1), 0, None) ** light.shininess
        img[ys, xs] += light.specular * sp[:, None]
    return np.clip(img, 0, 1)


# ------------------------------------------------------------------ 合成

def composite(plate_bgr, layers):
    """按深度**由远到近**把各层画上去。layers=[(depth, mask, bgr), ...]。

    这就是论文那步「合成机器人/夹爪图层以保证正确遮挡」：谁在前面由深度决定，
    不是由绘制顺序决定。
    """
    out = plate_bgr.astype(np.float32).copy()
    order = sorted(range(len(layers)), key=lambda k: np.nanmedian(
        np.where(layers[k][1], layers[k][0], np.nan)))
    for k in reversed(order):                    # 远的先画
        dep, m, img = layers[k]
        if m.any():
            out[m] = img[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay(bgr, mask, color=(0, 255, 255), alpha=0.45, contour=True):
    out = bgr.copy()
    if mask.any():
        sel = out[mask].astype(np.float32) * (1 - alpha) + np.asarray(color, np.float32) * alpha
        out[mask] = sel.astype(np.uint8)
    if contour:
        c, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, c, -1, color, 2)
    return out


def put_lines(img, lines, org=None, scale=0.5, pad=6):
    """左上角（org=None）或左下角写多行白底黑字——图是交付物，文字要看得清。"""
    y0 = (img.shape[0] - pad - 20 * (len(lines) - 1)) if org is None else org[1]
    x0 = pad if org is None else org[0]
    for j, t in enumerate(lines):
        p = (x0, y0 + 20 * j)
        cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    return img


# ------------------------------------------------------------------- 加载

def load_cams(cam="front"):
    """按相机名装标定。front=腕相机（有真内参）；up 只有 z=0 单应，另见 up_calib_mesh.py。"""
    kd = json.loads((HERE / "calib/front_intrinsics.json").read_text())
    return np.array(kd["K"], float), np.array(kd["dist"], float), kd.get("image_size", [640, 480])


# ------------------------------------------------------------------ 自检

def selftest():
    FIG.mkdir(parents=True, exist_ok=True)
    K, dist, (W, H) = load_cams()
    print(f"[相机] fx={K[0,0]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f} dist={np.round(dist,4)} "
          f"{W}×{H}")
    T = np.eye(4)                                   # 相机在原点，看向 +z（OpenCV 系）
    cam = Cam(K, dist, T, W, H)
    ok = True

    # 1) 球：剪影面积 vs 解析投影圆面积
    R, Z = 0.012, 0.11
    sph = transform(uv_sphere(R, 40, 80), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, Z], [0, 0, 0, 1.0]]))
    ras = rasterize(sph, cam)
    a_px = float(ras["mask"].sum())
    # 透视球剪影角半径：sinθ = R/Z → tanθ' ...
    th = math.asin(min(R / Z, 0.999))
    r_exact = K[0, 0] * math.tan(th) * 1.0
    a_exact = math.pi * r_exact ** 2
    cc = ras["mask"].astype(np.uint8)
    cnts, _ = cv2.findContours(cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    r_meas = math.sqrt(cv2.contourArea(max(cnts, key=cv2.contourArea)) / math.pi)
    print(f"[球] R={R*1000:.1f}mm Z={Z*1000:.0f}mm  解析 r={r_exact:.3f}px / 实测 r={r_meas:.3f}px"
          f"  面积 {a_exact:.1f} vs {a_px:.1f}px  → 差 {100*abs(r_meas-r_exact)/r_exact:.2f}%")
    if abs(r_meas - r_exact) / r_exact > 0.03:
        ok = False
        print("   ✗ 超出 3%")

    # 2) 遮挡：远处大球 + 近处小球。判据取**严格的**：合成图的深度必须**逐像素等于**
    #    近球单独渲染的结果（只要不是，就是 z-buffer 或排序错了）。
    #    注意别拿「球心深度」当期望：光轴打到的是球的**前表面**（z−R），不是球心。
    far = transform(uv_sphere(0.03, 20, 40), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1.0]]))
    near = transform(uv_sphere(0.008, 20, 40), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.08], [0, 0, 0, 1.0]]))
    r_near = rasterize(near, cam)
    r2 = rasterize(np.concatenate([far, near]), cam)
    both_m = r_near["mask"] & r2["mask"]
    dmax = float(np.abs(r2["depth"][both_m] - r_near["depth"][both_m]).max())
    ctr = r2["depth"][240, 320]
    print(f"[遮挡] 近球前表面 {ctr*1000:.1f}mm（=80mm−R）｜合成 vs 近球单独渲 最大差 "
          f"{dmax*1e9:.0f}nm（应 0）", "✓" if dmax == 0 and abs(ctr - 0.072) < 0.004 else "✗")
    if dmax != 0 or not abs(ctr - 0.072) < 0.004:
        ok = False

    # 3) 投影自洽：cv2.projectPoints vs 光栅化顶点
    p = np.array([[0.01, 0.005, 0.2]])
    uv1, _ = cam.project(p)
    Tt = np.eye(4)
    tri = np.tile(p, (1, 3)).reshape(1, 3, 3) + np.array([[0, 0, 0], [1e-5, 0, 0], [0, 1e-5, 0]])
    V = tri.reshape(-1, 3)
    uv2, _ = cam.project(V)
    print(f"[自洽] 顶点投影两路差 {np.abs(uv1[0]-uv2[0]).max():.2e}px",
          "✓" if np.abs(uv1[0] - uv2[0]).max() < 1e-6 else "✗")

    # 4) 畸变确实在起作用
    _, zc = cam.project(np.array([[0.0, 0.0, 0.2]]))
    uv_c = cam.project(np.array([[0.15, 0.1, 0.2]]))[0][0]
    uv_nod = np.array([K[0, 0] * 0.15 / 0.2 + K[0, 2], K[1, 1] * 0.1 / 0.2 + K[1, 2]])
    print(f"[畸变] 边缘点 带畸变 {np.round(uv_c,2)} vs 无畸变 {np.round(uv_nod,2)}"
          f"  → 差 {np.linalg.norm(uv_c-uv_nod):.1f}px")

    # 5) 出图
    plate = np.full((H, W, 3), 40, np.uint8)
    li = Light(direction=(0.3, -0.5, 0.8), ambient=0.3, diffuse=0.8, specular=0.3)
    ras = rasterize(transform(uv_sphere(0.012, 30, 60),
                              np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.11], [0, 0, 0, 1.0]])), cam)
    ball = shade(ras, cam, li, (0.12, 0.12, 0.85))
    blk = transform(box(0.05, 0.05, 0.05), np.array([[1, 0, 0, 0.02], [0, 1, 0, 0.0], [0, 0, 1, 0.15], [0, 0, 0, 1.0]]))
    ras2 = rasterize(blk, cam)
    cube = shade(ras2, cam, li, (0.15, 0.75, 0.15))
    comp = composite(plate, [(ras["depth"], ras["mask"], (ball * 255)),
                             (ras2["depth"], ras2["mask"], (cube * 255))])
    l, r = comp[:, :W // 2], comp[:, W // 2:]
    put_lines(l, [f"sphere r={r_meas:.2f}px  ({a_px:.0f}px)"], org=(8, 24), scale=0.5)
    put_lines(r, ["box 50mm @ {:.0f}mm".format(150)], org=(8, 24), scale=0.5)
    cv2.imwrite(str(FIG / "_meshrender_selftest.png"),
                np.hstack([overlay(comp, ras["mask"], (0, 255, 255)),
                           overlay(comp, ras2["mask"], (255, 200, 0))]))
    print("→ figures/_meshrender_selftest.png", "自检", "通过 ✓" if ok else "**未通过** ✗")
    return 0 if ok else 1


def look_at(eye, target, up_world=(0.0, 0.0, -1.0)) -> np.ndarray:
    """三点造 `T_base_cam`（OpenCV 系：x 右、y 下、z 前，右手 x×y=z）。

    base 系里 z 是**上**（桌面 z=0），所以 up_world 取 −z 表示「下」。
    """
    eye, tgt = np.asarray(eye, float), np.asarray(target, float)
    f = tgt - eye
    f /= np.linalg.norm(f)
    d = np.asarray(up_world, float)
    y = d - d.dot(f) * f
    y /= np.linalg.norm(y)
    x = np.cross(y, f)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, f], axis=1)
    T[:3, 3] = eye
    return T


# ------------------------------------------------------- Stage 1 验收：整臂进腕相机

def _load_module(name: str, path: Path):
    """按**文件路径**导入。加 sys.path 会把 object_generalization/ 下的 datasets/
    顶掉 HuggingFace 的 datasets（README 记过这个坑），所以一律走这里。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def armcheck(frame: int = 242, dataset: str = "newdata1", cell: float | None = None,
             target: str = "yellow", cube_h_m: float = 0.0215):
    """把整臂渲进腕相机，看覆盖率 + 印出「FK 链把方块投到哪」。

    这一条是 Stage 1 的出口检查，同时也是 Stage 2「以观测为主、FK 只做校验」的
    直接证据：same 一帧上，把方块中心按 FK⊗手眼 投进腕图，和它在腕图里的实测位置
    比 —— ep0 f242 是已知的 (354.5,245.5) vs (269.0,128.0) = 145px（ball_swap.py 里
    有记录）。这里用新渲染器重跑，确认数字复现、而不是渲染器自己引入的偏差。
    """
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(HERE))
    A = _load_module("_mr_arm", HERE / "arm.py")
    BS = _load_module("_mr_bs", HERE / "ball_swap.py")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    def _bgr(t):
        return cv2.cvtColor(np.clip(t.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR)

    FIG.mkdir(parents=True, exist_ok=True)
    K, dist, (W, H) = load_cams()
    urdf_path = A.URDF_PATH
    g = BS.load_geom()
    ds = LeRobotDataset(dataset, root=str(REPO / "dataset" / dataset))

    fr = ds[frame]
    st = fr["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, g["urdf"])
    # 腕相机在 base 系的位姿：FK(腕相机 link) ⊗ 手眼 X（和 ball_swap.render_frame 同式）
    T_base_cam = g["urdf"].fk(q, "wrist_camera_link") @ g["X"]
    cam = Cam(K, dist, T_base_cam, W, H)

    print(f"[帧] {dataset} f{frame}  T_base_cam 平移 {np.round(T_base_cam[:3,3]*1000,1)}mm")

    # ---- 逐连杆诊断 ----------------------------------------------------------
    # 这里要回答的是「搅进画面的到底是哪块几何、为什么」。判据是**顶点到光心的
    # 最近距离**：相机长在手腕上，所以手腕那几块的几何会贴到 30mm 以内，而
    # `z`（沿光轴的分量）在离轴 70° 的地方可以小到几毫米 —— 那种顶点一投影就是
    # 上万像素，跨过视场的三角形会把整幅图糊掉。这不是渲染器的错（`--armview`
    # 里同一套代码把整臂渲得干干净净），是**位姿的错**：手眼链在夹持深度上差
    # ~150px（下面 [链] 那一行就是这个数），40mm 的位姿误差放在 90mm 距离上，
    # 手腕几何就会以掠射角扫过整个视场。
    tris_all, used_all = gather(urdf_path, q)
    tris, used = arm_tris(urdf_path, q, cell=cell)
    print(f"[面数] 剔相机自身前面 {len(tris_all)} 面 / 后 {len(tris)} 面（cell={cell}）")
    print(f"[连杆] 参与渲染 {len(used)} 条；排除 {[l for l in used_all if l not in used]}")
    print(f"{'  连杆':<32s}{'覆盖':>7s}{'|Pc| 最近':>11s}{'|Pc| 最远':>11s}  ")
    near_lim = 0.04                      # 40mm：手眼系统差 ~30mm 的量级
    bad = []
    for L in used:
        tl, _ = gather(urdf_path, q, links={L})
        Pc = tl.reshape(-1, 3) @ cam.T_cam_base[:3, :3].T + cam.T_cam_base[:3, 3]
        d = np.linalg.norm(Pc, axis=1)
        rl = rasterize(tl, cam, cull=True)
        flag = ""
        if d.min() < near_lim:
            flag = "  ← 贴在光心上，位姿误差主导"
            bad.append(L)
        print(f"  {L:<30s}{100*rl['mask'].mean():6.1f}%{d.min()*1000:10.0f}mm"
              f"{d.max()*1000:10.0f}mm{flag}")

    ras = rasterize(tris, cam, cull=True)
    cov = float(ras["mask"].mean())
    z = ras["depth"][ras["mask"]]
    print(f"[覆盖] 整臂 {int(ras['mask'].sum())}px = 画面 {100*cov:.1f}%"
          f"｜深度 {z.min()*1000:.0f}~{z.max()*1000:.0f}mm")
    # 真正会挡住方块的只有夹爪那两块（离相机 50~150mm，投影干净）。出图以它为准。
    occl = [L for L in used if L in ("gripper_link", "moving_jaw_so101_v1_link")]
    if occl:
        to, _ = gather(urdf_path, q, links=set(occl))
        ro = rasterize(to, cam, cull=True)
        print(f"[遮挡层] {occl} → {100*ro['mask'].mean():.1f}%（Stage 2/4 的遮挡层就用它）")

    # FK 链 vs 观测：方块中心投到腕图
    # 掩码必须**按相机分开**：up 用目标色的 up 档阈值，腕图用 carry_cube 那套搬运阈值。
    # ★ 踩过：up 侧误用 blue_mask_bgr（那是干扰物蓝块）⇒ 拿蓝块的 FK 投影去比黄块的
    #   观测，得到 112px 的假误差；换成黄块掩码后才是真数（见下方打印）。
    tgt_up, _, tname, _ = BS.cube_mask_fns(g, "up", target)
    cmask = BS.carry_cube(_bgr(fr["observation.images.front"]))
    X_cube, src = BS.estimate_x_tcp_cube(ds, frame, g, 0.7, cube_h_m, tgt_up)
    line = f"f{frame} 目标={tname}"
    if X_cube is not None:
        P = (g["urdf"].fk(q, "gripper_frame_link") @ np.append(X_cube, 1.0))[:3]
        uv, _ = cam.project(P.reshape(1, 3))
        line += (f"｜X_tcp {np.round(X_cube*1000,1)}mm｜FK 投方块心 {np.round(uv[0],1)}")
        if cmask:
            line += (f" vs 观测 {np.round(cmask['c'],1)}"
                     f"  → 差 {np.linalg.norm(uv[0]-cmask['c']):.1f}px")
        line += f"\n     [{tname}] {src}"
    print("[链] " + line)

    # 出图：原图 | 渲染叠加 | 剪影 vs 分割出的夹爪
    real = _bgr(fr["observation.images.front"])
    li = Light(direction=(0.3, -0.5, 0.8), ambient=0.35, diffuse=0.7, specular=0.2)
    arm = shade(ras, cam, li, (0.86, 0.86, 0.88))
    img = composite(real.copy(), [(ras["depth"], ras["mask"], arm * 255)])
    vis = overlay(img, ras["mask"], (255, 200, 0), alpha=0.0)
    diff = real.copy()
    diff[ras["mask"]] = (0.4 * diff[ras["mask"]] + 0.6 * np.array([0, 200, 255]))
    out = np.hstack([put_lines(real, [f"real f{frame}"], org=(8, 22)),
                     put_lines(vis, [f"mesh {100*cov:.0f}%"], org=(8, 22)),
                     put_lines(diff, ["silhouette"], org=(8, 22))])
    p = FIG / "_mesh_arm_check.png"
    cv2.imwrite(str(p), out)
    print("→", p.relative_to(REPO))
    return 0


def armview(frame: int = 242, dataset: str = "newdata1", cell: float = 0.004,
            eye=(0.62, 0.52, 0.40), target=(0.12, 0.08, 0.11)):
    """第三人称视角看整臂：**这一步只验渲染器，不验位姿**。

    为什么必须分开验：腕相机（`wrist_camera_link` ⊗ 手眼 X）那条位姿链在夹持深度上
    差 ~150px（见 `--armcheck` 的打印），而相机又是**长在手腕上**的 —— 位姿差 40mm
    时，贴着手腕的那些网格会以掠射角扫过整个视场，渲出来是一张「蛛网」。那是**位姿
    的错，不是渲染器的错**，混在一起就查不清了。所以这里把相机放到旁边，用干净的
    针孔看整臂：看着像一只手臂，渲染器就没问题。
    """
    sys.path.insert(0, str(REPO))
    A = _load_module("_mr_arm", HERE / "arm.py")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    FIG.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset(dataset, root=str(REPO / "dataset" / dataset))
    st = ds[frame]["observation.state"].numpy().astype(float)
    q = A.norm_to_q({m: float(v) for m, v in zip(A.MOTORS, st)}, Urdf(A.URDF_PATH))
    W, H = 900, 700
    K = np.array([[0.88 * W, 0, W / 2], [0, 0.88 * W, H / 2], [0, 0, 1.0]])
    cam = Cam(K, np.zeros(5), look_at(eye, target), W, H)

    tris, used = arm_tris(A.URDF_PATH, q, cell=cell, drop_self=False)
    print(f"[整臂] {len(tris)} 面（cell={cell}）｜{len(used)} 条连杆"
          f"｜相机 eye={np.round(np.asarray(eye)*1000,0)}mm target={np.round(np.asarray(target)*1000,0)}mm")
    ras = rasterize(tris, cam, cull=True)
    z = ras["depth"][ras["mask"]]
    print(f"[覆盖] {100*ras['mask'].mean():.1f}%｜深度 {z.min()*1000:.0f}~{z.max()*1000:.0f}mm")

    li = Light(direction=(-0.4, -0.55, 0.75), ambient=0.35, diffuse=0.75, specular=0.25)
    plate = np.full((H, W, 3), 34, np.uint8)
    # 桌面：base z=0 上一块 500mm 见方，给个参照系
    table = box(0.5, 0.5, 0.004)
    table = transform(table, np.array([[1, 0, 0, 0.18], [0, 1, 0, 0.10], [0, 0, 1, -0.002], [0, 0, 0, 1.0]]))
    rt = rasterize(table, cam, cull=True)
    tab = shade(rt, cam, li, (0.62, 0.62, 0.60))
    body = shade(ras, cam, li, (0.80, 0.80, 0.82))
    img = composite(plate, [(rt["depth"], rt["mask"], tab * 255),
                            (ras["depth"], ras["mask"], body * 255)])

    # 第二栏：按连杆着色的「爆炸图」——用户要解释性，这一张就是说清「哪些网格在挡」
    # 颜色是 **BGR**（shade 吃的就是 BGR，别倒过来，倒过来红蓝互换过一次）
    COL = {"base_link": (110, 110, 110), "shoulder_link": (220, 60, 60),
           "upper_arm_link": (60, 200, 60), "lower_arm_link": (40, 120, 220),
           "wrist_link": (200, 60, 200), "gripper_link": (220, 220, 40),
           "moving_jaw_so101_v1_link": (30, 30, 255),
           "wrist_camera_mount_link": (0, 165, 255), "wrist_camera_link": (255, 0, 255)}
    lay = []
    for L, c in COL.items():
        try:
            tl, _ = gather(A.URDF_PATH, q, links={L})
        except SystemExit:
            continue
        r = rasterize(tl, cam, cull=True)
        sh = shade(r, cam, li, tuple(v / 255 for v in c))
        lay.append((r["depth"], r["mask"], sh * 255))
    img2 = composite(plate, [(rt["depth"], rt["mask"], tab * 255)] + lay)

    out = np.hstack([put_lines(img, [f"arm f{frame}  {len(tris)} tris  cell={cell*1000:.0f}mm"],
                               org=(10, 26), scale=0.7),
                     put_lines(img2, ["by link (colour = one URDF link)"],
                               org=(10, 26), scale=0.7)])
    p = FIG / "mesh_wireframe.png"
    cv2.imwrite(str(p), out)
    print("→", p.relative_to(REPO))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--armview", action="store_true", help="第三人称看整臂（只验渲染器）")
    ap.add_argument("--armcheck", action="store_true", help="整臂渲进腕相机的 Stage 1 验收")
    ap.add_argument("--frame", type=int, default=242)
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--cell", type=float, default=None, help="顶点聚类减面边长（米）")
    ap.add_argument("--target", default="yellow", choices=("yellow", "blue"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.armcheck:
        return armcheck(frame=a.frame, dataset=a.dataset, cell=a.cell, target=a.target)
    if a.armview:
        return armview(frame=a.frame, dataset=a.dataset,
                       cell=0.004 if a.cell is None else a.cell)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
