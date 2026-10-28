#!/usr/bin/env python
"""生成一个居中的长方体 STL（毫米输入），用于换物体注入。

    python make_box_stl.py --dims 45 20 12 --out calib/recon/eraser_45x20x12.stl

为什么需要它：`pose6d_up_fit.py` / `pose6d_wrist_fit.py` 只认
`calib/recon/*.stl`（或 `cube<边长>`）。橡皮是长方体，没必要做 visual hull，
按实测长宽高直接造一个水密盒子即可；剪影拟合要的是形状和尺度，不是扫描细节。

注意：输出 STL 的**单位是米**（`MR.stl_cached` 的约定），命令行输入是毫米。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def box_tris(sx: float, sy: float, sz: float) -> np.ndarray:
    """轴对齐长方体（中心在原点）→ (12,3,3)，绕序朝外。与 `meshrender.box` 同一套。"""
    s = np.array([sx, sy, sz], float) / 2
    v = np.array([[i, j, k] for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)],
                 float) * s
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in faces:
        tris += [np.stack([v[a], v[b], v[c]]), np.stack([v[a], v[c], v[d]])]
    t = np.asarray(tris, np.float32)
    ctr = t.mean(1)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (nrm * ctr).sum(1) < 0
    t[flip] = t[flip][:, ::-1]
    return t


def _rounded_box_sdf(p: np.ndarray, half: np.ndarray, r: float) -> np.ndarray:
    """圆角盒 SDF（中心在原点）。p: (N,3)。"""
    q = np.abs(p) - (half - r)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside - r


def rounded_box_tris(sx: float, sy: float, sz: float, radius: float,
                     n: int = 24) -> np.ndarray:
    """圆角长方体：从中心按球面方向发射线，求 SDF=0 的表面点。

    橡皮不是直角盒子 —— 角是圆的。直角盒会在剪影上多出尖角，绕竖轴的
    位姿拟合会被带偏。这里生成真正的水密圆角盒（平面 + 圆柱棱 + 球角），
    且不依赖 scipy：锥形束在凸形状上不会自遮挡，得到的是星形参数化。

    `n` 是纬度方向环数（默认 24；经度取 2n 份，面数约 4n²）。
    """
    a, b, c = sx / 2, sy / 2, sz / 2
    half = np.array([a, b, c], float)
    r = float(radius)
    if not (0 < r < min(a, b, c)):
        raise ValueError(f"radius={r} 必须在 (0, {min(a, b, c)}) 内")
    if n < 6:
        raise ValueError("n 至少为 6")

    n_theta, n_phi = int(n), int(2 * n)
    # 注意：theta=0 / pi 是两极；theta=pi/2 与 phi=0 / pi/2 必须被采到，
    # 这样包围盒的 6 个面心才精确落在 ±half 上。
    theta = np.linspace(0.0, np.pi, n_theta + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dirs = []
    dirs.append([0.0, 0.0, 1.0])                       # 北极
    for i in range(1, n_theta):
        st, ct = np.sin(theta[i]), np.cos(theta[i])
        for p in phi:
            dirs.append([st * np.cos(p), st * np.sin(p), ct])
    dirs.append([0.0, 0.0, -1.0])                      # 南极
    dirs = np.asarray(dirs, float)

    # 每条射线 t*d 上二分求 sdf=0（sdf(0)=-r<0）
    lo = np.zeros(len(dirs))
    hi = np.full(len(dirs), 2.0 * half.max())
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        s = _rounded_box_sdf(mid[:, None] * dirs, half, r)
        neg = s < 0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    verts = (0.5 * (lo + hi))[:, None] * dirs

    # 顶点索引
    north = 0
    south = len(verts) - 1
    ring = lambda i, j: 1 + (i - 1) * n_phi + (j % n_phi)
    faces = []
    for j in range(n_phi):                              # 北极扇
        faces.append([north, ring(1, j), ring(1, j + 1)])
    for i in range(1, n_theta - 1):                     # 环带
        for j in range(n_phi):
            a0, a1 = ring(i, j), ring(i, j + 1)
            b0, b1 = ring(i + 1, j), ring(i + 1, j + 1)
            faces += [[a0, b0, b1], [a0, b1, a1]]
    for j in range(n_phi):                              # 南极扇
        faces.append([south, ring(n_theta - 1, j + 1),
                      ring(n_theta - 1, j)])
    tris = verts[np.asarray(faces, int)].astype(np.float32)
    # 统一朝外（形状以原点为中心）
    ctr = tris.mean(1)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = (nrm * ctr).sum(1) < 0
    tris[flip] = tris[flip][:, ::-1]
    return tris


def write_stl(path: Path, tris: np.ndarray, name: str) -> None:
    """ASCII STL（`meshrender.stl_cached` 能原样读回）。"""
    tris = np.asarray(tris, np.float32)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    out = [f"solid {name}"]
    for t, n in zip(tris, nrm):
        out.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        out.append("    outer loop")
        for p in t:
            out.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        out.append("    endloop")
        out.append("  endfacet")
    out.append(f"endsolid {name}")
    path.write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=float, nargs=3, required=True,
                    metavar=("L", "W", "H"), help="长 宽 高（毫米）")
    ap.add_argument("--out", required=True, help="输出 .stl 路径")
    ap.add_argument("--name", default=None, help="STL 内部 solid 名（默认取文件名）")
    ap.add_argument("--round", type=float, default=0.0, metavar="R",
                    help="圆角半径（mm）；0 = 直角盒（默认）")
    ap.add_argument("--round-n", type=int, default=24,
                    help="圆角纬度环数（默认 24；越大越圆、面数越多）")
    a = ap.parse_args()
    if min(a.dims) <= 0:
        raise SystemExit("--dims 必须全为正数（mm）")
    if a.round < 0:
        raise SystemExit("--round 不能为负")

    p = Path(a.out)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    p.parent.mkdir(parents=True, exist_ok=True)
    dims = [d / 1000.0 for d in a.dims]
    if a.round > 0:
        tris = rounded_box_tris(*dims, radius=a.round / 1000.0, n=a.round_n)
    else:
        tris = box_tris(*dims)
    write_stl(p, tris, a.name or p.stem)
    print(f"[save] {p}  尺寸 {a.dims[0]:g}×{a.dims[1]:g}×{a.dims[2]:g} mm"
          f"  圆角 {a.round:g} mm  面数 {len(tris)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
