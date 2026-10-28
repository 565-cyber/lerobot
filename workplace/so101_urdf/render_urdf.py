#!/usr/bin/env python
"""把 SO101 的 URDF + STL 渲染成图 —— 用来**肉眼核对**运动学是对的。

只依赖 numpy + PIL：不给这个环境装 trimesh/pybullet（本机一个都没有）。

为什么需要它：`fk_check.py` 打印的是数字，数字对不代表装配对。这里把 18 个
STL 按 FK 摆到各自连杆的坐标系里光栅化，出一张图，和真机摆成同样关节角比一下。
**姿态像不像，一眼就能判**；坐标对不对，谁都看不出来。

用法：
    python workplace/so101_urdf/render_urdf.py                       # new_calib @ q=0
    python workplace/so101_urdf/render_urdf.py --urdf so101_old_calib.urdf
    python workplace/so101_urdf/render_urdf.py --pose 0 0 -30 60 0 50
    python workplace/so101_urdf/render_urdf.py --compare             # 两套零位并排

输出写到 `workplace/so101_urdf/figures/`（图是交付物，不写 /tmp）。
"""

from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fk_check import HERE, JOINT_ORDER, TCP_LINK, Urdf, rpy_to_R

ASSETS = HERE / "assets"
OUTDIR = HERE / "figures"


# --------------------------------------------------------------------------- #
#  STL：二进制是 80B 头 + uint32 面数 + 每面 50B（法线 12B + 三顶点 36B + 属性 2B）
# --------------------------------------------------------------------------- #

def load_stl(path: Path) -> np.ndarray:
    """返回 (N,3,3) float32 的三角形顶点。"""
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            raw = np.frombuffer(data, dtype=np.uint8, offset=84, count=50 * n)
            rec = raw.reshape(n, 50)[:, :48].copy().view("<f4").reshape(n, 4, 3)
            return rec[:, 1:4, :].astype(np.float32)      # 第 0 行是法线，扔掉
    # ASCII 兜底
    verts = [ln.split()[1:4] for ln in data.decode("ascii", "ignore").splitlines()
             if ln.strip().startswith("vertex ")]
    if not verts:
        raise ValueError(f"{path} 既不是合法二进制 STL 也不是 ASCII STL")
    return np.asarray(verts, dtype=np.float32).reshape(-1, 3, 3)


# --------------------------------------------------------------------------- #
#  URDF 里的 mesh：<geometry><mesh filename="..."/>，可能在 visual 或 collision 下
# --------------------------------------------------------------------------- #

def link_meshes(root: ET.Element) -> dict[str, list[tuple[np.ndarray, Path]]]:
    """连杆名 -> [(相对连杆系的 4x4, stl 路径), ...]"""
    out: dict[str, list] = {}
    for link in root.findall("link"):
        name = link.get("name")
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                mesh = el.find("geometry/mesh")
                if mesh is None:
                    continue
                fn = mesh.get("filename")
                if not fn:
                    continue
                p = ASSETS / Path(fn).name
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
                    sc = np.array([float(v) for v in s.split()])
                    T = T @ np.diag(np.append(sc, 1.0))
                out.setdefault(name, []).append((T, p))
    return out


def gather(urdf_path: Path, q: dict[str, float]):
    """把所有 mesh 顶点变到 base_link 系，返回拼接后的 (M,3,3) 三角形数组。"""
    root = ET.parse(urdf_path).getroot()
    meshes = link_meshes(root)
    u = Urdf(urdf_path)
    tris, used = [], []
    cache: dict[Path, np.ndarray] = {}
    for link, items in meshes.items():
        if link not in u.parent_link:
            continue                                   # 不在 base 的链上
        T_link = u.fk(q, link)
        got = False
        for T_vis, p in items:
            if p not in cache:
                cache[p] = load_stl(p)
            v = cache[p]
            T = T_link @ T_vis
            w = v @ T[:3, :3].T + T[:3, 3]
            tris.append(w.astype(np.float32))
            got = True
        if got:
            used.append(link)
    if not tris:
        raise SystemExit(f"{urdf_path.name}: 没有任何 mesh 落在 base_link 的链上"
                         f"（assets/ 里有 {len(list(ASSETS.glob('*.stl')))} 个 STL）")
    return np.concatenate(tris), used


# --------------------------------------------------------------------------- #
#  光栅化：look-at + 透视 + z-buffer + 平面着色。~380k 面，逐面循环但面内向量化。
# --------------------------------------------------------------------------- #

def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.array(up, dtype=float))
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    # 这三行必须是**行**（world→camera），不是列。写成列的话 M[:3,:3] 是 R 而不是
    # Rᵀ，`M[:3,:3] @ Δ` 方向就反了，任何点都投影不到该在的位置（踩过一次：
    # 目标点本该落在画面正中，却投影到 (-75, 819)）。
    M[:3, :3] = np.stack([r, u, -f], axis=0)
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def orbit_eye(center, dist, azim_deg, elev_deg):
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    return center + dist * np.array([math.cos(e) * math.cos(a),
                                     math.cos(e) * math.sin(a),
                                     math.sin(e)])


def rasterize(tris, W, H, eye, target, fov_deg=35.0, bg=(0.10, 0.11, 0.13),
              seed_color=(0.86, 0.77, 0.24), light=(0.4, -0.7, 0.6)):
    V = look_at(np.asarray(eye, float), np.asarray(target, float))
    # 相机看向 -Z，故正前方的点 p[:,2] < 0、深度取 -p[:,2]。
    p = tris.reshape(-1, 3) @ V[:3, :3].T + V[:3, 3]
    z = -p[:, 2]
    foc = (H / 2) / math.tan(math.radians(fov_deg) / 2)
    sx = W / 2 + foc * p[:, 0] / np.maximum(z, 1e-6)
    sy = H / 2 - foc * p[:, 1] / np.maximum(z, 1e-6)
    sx, sy, z = sx.reshape(-1, 3), sy.reshape(-1, 3), z.reshape(-1, 3)

    # 世界系面法线 + 朗伯
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-12)
    lt = np.asarray(light, float)
    lt = lt / np.linalg.norm(lt)
    shade = np.abs(nrm @ lt)                            # 双面：取绝对值
    shade = 0.25 + 0.75 * shade
    rgb = np.clip(np.asarray(seed_color)[None, :] * shade[:, None], 0, 1)

    # 背面剔除（屏幕系有向面积）
    area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) \
        - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
    keep = (area > 0.5) & (z.min(axis=1) > 1e-4)
    idx = np.nonzero(keep)[0]

    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
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
        zi = w0 * z[i, 0] + w1 * z[i, 1] + w2 * z[i, 2]
        sub_z = zbuf[lo_y:hi_y + 1, lo_x:hi_x + 1]
        upd = m & (zi < sub_z)
        if not upd.any():
            continue
        sub_z[upd] = zi[upd]
        sub_i = img[lo_y:hi_y + 1, lo_x:hi_x + 1]
        sub_i[upd] = rgb[i]
    return img


# --------------------------------------------------------------------------- #

def draw(urdf_path, q, W, H, azim, elev, fov, label, marks=()):
    tris, used = gather(urdf_path, q)
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    center = (lo + hi) / 2
    # 距离由**外接球**和 fov 反推，保证装得下（用对角线会裁掉一半）
    radius = float(np.linalg.norm(hi - lo)) / 2
    aspect = W / H
    half = math.radians(fov) / 2
    dist = 1.15 * radius / math.tan(min(half, math.atan(math.tan(half) * aspect)))
    eye = orbit_eye(center, dist, azim, elev)
    img = rasterize(tris, W, H, eye, center, fov)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    V = look_at(eye, center)
    dr = ImageDraw.Draw(pil)
    foc = (H / 2) / math.tan(math.radians(fov) / 2)

    def proj(pt):
        pc = V[:3, :3] @ np.asarray(pt, float) + V[:3, 3]
        zz = -pc[2]                                     # 同 rasterize：看向 -Z
        if zz <= 1e-6:
            return None
        return W / 2 + foc * pc[0] / zz, H / 2 - foc * pc[1] / zz

    # 地面网格（z=0）。两套 URDF 的 X/Y 差了约 90° 方位，但 **Z 都是朝上的**，
    # 所以「相对地面平不平」是唯一跨约定可直接比的东西 —— 网格就是那个参照。
    span = max(float(np.abs(tris.reshape(-1, 3)[:, :2]).max()), 0.45) * 1.15
    step = 0.1
    g = np.arange(-span, span + 1e-9, step)
    for c in g:
        for a, b in ((( c, -span, 0), ( c,  span, 0)),
                     ((-span,  c, 0), ( span,  c, 0))):
            pa, pb = proj(a), proj(b)
            if pa and pb:
                edge = abs(c) < 1e-9
                dr.line([pa, pb], fill=(96, 104, 118) if edge else (46, 51, 61),
                        width=2 if edge else 1)
    o = proj((0, 0, 0))
    if o:
        dr.line([o, proj((0, 0, 0.15))], fill=(220, 80, 80), width=3)
        dr.text((o[0] + 4, o[1] - 20), "+Z", fill=(220, 80, 80))

    # 关键点（TCP / 腕部相机），确认它们落在夹爪上
    for name, pt, col in marks:
        q = proj(pt)
        if q and 0 <= q[0] < W and 0 <= q[1] < H:
            r = 7
            dr.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], outline=col, width=3)
            dr.text((q[0] + r + 3, q[1] - 6), name, fill=col)

    # 文字用 ASCII：PIL 的默认字体没有中文字形，写中文会出豆腐块（踩过）
    dr.text((10, 10), label, fill=(255, 255, 255))
    dr.text((10, 26), f"azim={azim} elev={elev}  {len(used)} mesh links, "
                      f"{len(tris)} tris", fill=(170, 175, 185))
    return pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(HERE / "so101_new_calib_camera.urdf"))
    ap.add_argument("--pose", nargs=6, type=float, default=[0] * 6)
    ap.add_argument("--compare", action="store_true",
                    help="和 so101_old_calib.urdf 在同样关节角下并排")
    ap.add_argument("--azim", type=float, default=125.0)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--fov", type=float, default=35.0)
    ap.add_argument("--size", default="900x700")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    W, H = (int(v) for v in a.size.lower().split("x"))
    path = Path(a.urdf)
    u = Urdf(path)
    q = {n: math.radians(v) for n, v in zip(u.revolute, a.pose)}
    pose_s = " ".join(f"{v:g}" for v in a.pose)

    marks = []
    if TCP_LINK in u.parent_link:
        marks.append(("TCP", u.fk(q, TCP_LINK)[:3, 3], (60, 230, 120)))
    if any(j.find("child").get("link") == "wrist_camera_link"
           for j in ET.parse(path).getroot().findall("joint")):
        marks.append(("wrist_cam", u.fk(q, "wrist_camera_link")[:3, 3],
                      (90, 190, 255)))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if a.compare:
        old = HERE / "so101_old_calib.urdf"
        ims = [
            draw(path, q, W, H, a.azim, a.elev, a.fov, f"{path.stem}  q={pose_s}", marks),
            draw(old, q, W, H, a.azim, a.elev, a.fov, f"{old.stem}  q={pose_s}", marks),
        ] if old.exists() else [draw(path, q, W, H, a.azim, a.elev, a.fov,
                                     f"{path.stem}  q={pose_s}", marks)]
        out = Image.new("RGB", (W * len(ims), H))
        for i, im in enumerate(ims):
            out.paste(im, (i * W, 0))
        name = a.out or "urdf_compare.png"
    else:
        out = draw(path, q, W, H, a.azim, a.elev, a.fov,
                   f"{path.stem}  q={pose_s}", marks)
        name = a.out or f"urdf_{path.stem}.png"

    dst = OUTDIR / name
    out.save(dst)
    print(f"→ {dst}   ({out.width}×{out.height})")
    print("核对方法：把它和真机摆成同样的六个关节角，比姿态。")
    print("  q=0 时 new_calib 的大臂应陡升、小臂水平；old_calib 应整条臂平躺。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
