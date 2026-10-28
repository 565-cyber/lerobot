#!/usr/bin/env python
"""当前 up 机位 + 新外参：把桌上的真实橡皮拟合回去，验证 `T_base_cam` 是否对得上。

背景（必须记住）：`calib/up_intrinsics.json` / `up_camera_pose.json` 是 **2026-10-19
当前机位** 的标定。`dataset/newdata1` 的 up 帧是 **旧机位** 录的（`CAMERAS.md` 有记录），
新外参不能直接套到那些帧上；否则物体会投到错位置。

这个脚本用当前机位拍的一张图（`calib/eraser_on_table.png`）做**独立验证**：
  · 橡皮静置在桌面上 ⇒ 中心高度 = 厚度/2 = 6mm，朝向只有绕竖直轴的 yaw 未知；
  · 用真 K + dist + T_base_cam，把 40×22×12 圆角网格按 (x,y,yaw) 拟合到 GrabCut 剪影；
  · 如果外参错，40mm 的物体在 685mm 处会明显错位/错尺度，IoU 会立刻塌。

用法：
    PY=/home/wzy/anaconda3/envs/lerobot1/bin/python
    $PY -I workplace/calibration/up_live_eraser.py
    $PY -I workplace/calibration/up_live_eraser.py --image calib/eraser_on_table.png \
        --mesh eraser_40x22x12_r3n80 --out figures/up_live_eraser_fit.png
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
REPO = HERE.parents[1]
FIG = HERE / "figures"
CAL = HERE / "calib"


def _load(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


MR = _load("_ule_mr", HERE / "meshrender.py")
F = _load("_ule_f", HERE / "fit_mesh_pose.py")
P6 = _load("_ule_p6", HERE / "pose6d_swap.py")
PW = _load("_ule_pw", HERE / "pose6d_wrist_fit.py")


def load_up_cam():
    ip = json.loads((CAL / "up_intrinsics.json").read_text())
    pp = json.loads((CAL / "up_camera_pose.json").read_text())
    K = np.asarray(ip["K"], float)
    dist = np.asarray(ip["dist"], float)
    T = np.asarray(pp["T_base_cam"], float)
    cam = MR.Cam(K, dist, T, W=int(ip["image_size"][0]), H=int(ip["image_size"][1]))
    return cam, ip, pp


def grabcut_mask(img, rect):
    mask = np.zeros(img.shape[:2], np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="calib/eraser_on_table.png")
    ap.add_argument("--box", type=int, nargs=4, default=(210, 120, 120, 85),
                    metavar=("X", "Y", "W", "H"), help="GrabCut 初始框")
    ap.add_argument("--mesh", default="eraser_40x22x12_r3n80")
    ap.add_argument("--cell-mm", type=float, default=0.5)
    ap.add_argument("--thick-mm", type=float, default=12.0,
                    help="橡皮厚度；静置时中心 z = 厚度/2")
    ap.add_argument("--n-rot", type=int, default=72)
    ap.add_argument("--out", default="figures/up_live_eraser_fit.png")
    a = ap.parse_args()

    p = Path(a.image)
    if not p.is_absolute():
        p = HERE / p
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"读不到 {p}")
    cam, ip, pp = load_up_cam()
    if (cam.W, cam.H) != (img.shape[1], img.shape[0]):
        cam = MR.Cam(cam.K, cam.dist, cam.T_base_cam,
                     W=img.shape[1], H=img.shape[0])

    m = grabcut_mask(img, tuple(a.box))
    ys, xs = np.nonzero(m)
    print(f"[mask] GrabCut {int(m.sum())}px  bbox x{xs.min()}..{xs.max()} "
          f"y{ys.min()}..{ys.max()}")
    obs = m.astype(np.float32)

    cell = a.cell_mm / 1000.0
    mesh = F.load_mesh(a.mesh, cells=(max(2e-3, 2 * cell), max(1e-3, cell), cell))
    z_mm = a.thick_mm / 2.0
    T, info = F.fit_up_real(mesh, cam, obs, z_mm=z_mm, fix_s=1.0, cell=cell,
                            n_rot=a.n_rot, verbose=True,
                            xy_range_mm=30.0, xy_step_mm=1.0,
                            yaw_step_deg=5.0)

    # 渲染（和腕视图同一套材质/限角顶点法线）
    tris_b = MR.transform(mesh.level(cell), T)
    ras = MR.rasterize(tris_b, cam)
    ras = PW.smooth_normals(ras, tris_b, cam, PW.smooth_for(mesh, cell, tris_b))
    mat = MR.Light(direction=P6.load_lighting().dir, ambient=PW.ERASER_AMBIENT,
                   diffuse=PW.ERASER_DIFFUSE, specular=PW.ERASER_SPECULAR,
                   shininess=PW.ERASER_SHININESS)
    rgb = MR.shade(ras, cam, mat, PW.ERASER_ALBEDO)
    bm = ras["mask"]

    out = img.copy()
    out[bm] = (rgb * 255.0)[bm]
    ov = img.copy()
    cv2.drawContours(ov, cv2.findContours(m, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                     (0, 255, 0), 1)
    cv2.drawContours(ov, cv2.findContours(bm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                     (0, 255, 255), 1)
    inter = float((bm & m.astype(bool)).sum())
    union = float((bm | m.astype(bool)).sum())
    iou = inter / max(union, 1e-9)
    print(f"[fit] IoU={iou:.4f}  P=({info['P'][0]*1000:.1f},"
          f"{info['P'][1]*1000:.1f},{info['P'][2]*1000:.1f})mm  "
          f"yaw={info['yaw_deg']:+.1f}°  面积比={info['area_ratio']:.4f}")

    # 出图：原图 / 观测掩码 / 重叠 / 合成 / 放大
    row = [MR.put_lines(img, ["original (current up cam)"], org=(8, 22), scale=0.55),
           MR.put_lines(cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR),
                        [f"GrabCut mask {int(m.sum())}px"], org=(8, 22), scale=0.55),
           MR.put_lines(ov, ["observed (green) / mesh (yellow)"], org=(8, 22),
                        scale=0.55),
           MR.put_lines(out, [f"composite  IoU={iou:.3f}  yaw={info['yaw_deg']:+.1f}deg"],
                        org=(8, 22), scale=0.55)]
    canvas = np.hstack(row)
    # 放大橡皮区域拼在下面
    x0, y0, w, h = a.box
    pad = 20
    crop = lambda im: cv2.resize(
        im[max(0, y0 - pad):y0 + h + pad, max(0, x0 - pad):x0 + w + pad],
        (420, 300), interpolation=cv2.INTER_NEAREST)
    zoom = np.hstack([crop(img), crop(ov), crop(out)])
    if zoom.shape[1] != canvas.shape[1]:
        zoom = cv2.resize(zoom, (canvas.shape[1],
                                 int(round(zoom.shape[0] * canvas.shape[1]
                                           / zoom.shape[1]))),
                          interpolation=cv2.INTER_NEAREST)
    canvas = np.vstack([canvas, zoom])
    op = Path(a.out)
    if not op.is_absolute():
        op = HERE / op
    op.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(op), canvas)
    print(f"[图] {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
