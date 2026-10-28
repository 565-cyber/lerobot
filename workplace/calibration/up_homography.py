#!/usr/bin/env python3
"""第二步 · 走 A：`up`（静止俯视）的**桌面单应** —— 不动臂。

板平放在桌面上 → 板平面 ↔ 图像 是一个单应 H。有了 H：
  图像上任一点 ⇄ 桌面平面上的点（单位米，板坐标系）。
本步只求 H 本身；把「板坐标系」对到「机械臂 base 系」是另一件事
（需要臂去碰一个已知点，属动臂那一段）。

用法：
    python workplace/calibration/up_homography.py --board-mm 100.0 --measured-mm 98.5
    # --board-mm 板上的标称长度（100），--measured-mm 你实测的长度
    # 不给 --measured-mm 就用标称值（会差百分之几，不建议）

产物：
    calib/up_homography.json     H（板米→像素）、H⁻¹、内点/残差/重复性
    figures/up_homography_check.png   把板格按 H 投回图上的**肉眼验收图**
    figures/up_homography_raw.png     原始抓帧（留档）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
CALIB_DIR = HERE.parent
sys.path.insert(0, str(CALIB_DIR))
from cam_id import CAMS, resolve                      # noqa: E402  (单一来源：CAMERAS.md)

FIG_DIR = CALIB_DIR / "figures"
OUT_DIR = CALIB_DIR / "calib"
META = CALIB_DIR / "board" / "charuco_board.json"


def board_from(meta: dict, square_m: float):
    d = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, meta["dict"]))
    board = cv2.aruco.CharucoBoard((meta["squares_x"], meta["squares_y"]),
                                   square_m, square_m * meta["marker_mm_nominal"]
                                   / meta["square_mm_nominal"], d)
    return cv2.aruco.CharucoDetector(board), board


def board_points_m(board, ids: np.ndarray) -> np.ndarray:
    """被检出的 charuco 角点在**板平面**上的坐标（米，z=0）。"""
    return board.getChessboardCorners()[ids.flatten()][:, :2]


def capture(dev: Path, n: int) -> list[np.ndarray]:
    import front_handeye as FH                   # 分辨率必须显式设，见那里的说明
    cap = FH.open_camera(dev)
    if cap is None:
        raise SystemExit(f"打不开 {dev}")
    out = []
    try:
        while len(out) < n:
            ok, f = cap.read()
            if ok:
                out.append(f)
    finally:
        cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-mm", type=float, default=100.0,
                    help="板上那把尺的标称长度（mm），默认 100")
    ap.add_argument("--measured-mm", type=float, default=None,
                    help="你实测的长度（mm）。不给则用标称值")
    ap.add_argument("--frames", type=int, default=12, help="抓多少帧做重复性检查")
    ap.add_argument("--cam", default="camera_A",
                    help="camera_A = 无序列号 = up（见 CAMERAS.md）")
    a = ap.parse_args()

    scale = (a.measured_mm / a.board_mm) if a.measured_mm else 1.0
    meta = json.loads(META.read_text())
    square_m = meta["square_mm_nominal"] / 1000.0 * scale
    if a.measured_mm is None:
        print("[!] 没给实测尺长 —— 按标称算，尺度会有百分之几的系统误差")
    print(f"[板] {meta['squares_x']}x{meta['squares_y']}  格 = {square_m*1000:.2f} mm"
          f"（标称 {meta['square_mm_nominal']:.0f} × {scale:.4f}）")

    devs = resolve()
    if a.cam not in devs:
        raise SystemExit(f"没有 {a.cam}；有的：{list(devs)}")
    det, board = board_from(meta, square_m)

    frames = capture(devs[a.cam], a.frames)
    print(f"[抓图] {devs[a.cam]} × {len(frames)} 帧")

    Hs, errs, okf = [], [], 0
    for i, f in enumerate(frames):
        cc, ci, _, _ = det.detectBoard(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        if ci is None or len(ci) < 8:
            print(f"  帧{i:02d}: 角点不足（{0 if ci is None else len(ci)}）—— 跳过")
            continue
        obj = board_points_m(board, ci).astype(np.float32)
        img = cc.reshape(-1, 2).astype(np.float32)
        H, mask = cv2.findHomography(obj, img, cv2.RANSAC, 3.0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.linalg.norm(proj - img, axis=1).mean())
        Hs.append(H)
        errs.append(err)
        okf += 1
        print(f"  帧{i:02d}: 角点 {len(ci):3d}  内点 {int(mask.sum()):3d}  平均重投影 {err:5.2f} px")

    if not Hs:
        raise SystemExit("没有一帧认得出板 —— 板在画面里吗？光线够吗？")

    H = np.median(np.stack(Hs), axis=0)
    H = H / H[2, 2]
    print(f"\n[单应] 用了 {okf} 帧，中位数合成")
    print(f"  平均重投影 {np.mean(errs):.2f} px（最大 {np.max(errs):.2f}）")

    # 重复性：各帧解出的像素映射互相差多少（板不动、相机不动 ⇒ 应几乎为 0）
    W, Hh = frames[0].shape[1], frames[0].shape[0]
    grid = np.stack(np.meshgrid(np.linspace(0, W - 1, 9), np.linspace(0, Hh - 1, 7)),
                    -1).reshape(-1, 1, 2).astype(np.float32)
    pts = np.stack([cv2.perspectiveTransform(grid, cv2.invert(h)[1]) for h in Hs])
    rep = float(np.linalg.norm(pts - pts.mean(0), axis=2).mean())
    print(f"  帧间重复性（图像→桌面，逐帧互差）= {rep*1000:.2f} mm"
          f"  ← 板没动、相机没动的话这个数应该很小")

    # 尺度换算：一个像素在同一平面上是多少毫米（在画面中心附近）
    c = np.array([[[W / 2, Hh / 2]]], np.float32)
    o = cv2.perspectiveTransform(c, cv2.invert(H)[1])[0, 0]
    o2 = cv2.perspectiveTransform(np.array([[[W / 2 + 100, Hh / 2]]], np.float32),
                                  cv2.invert(H)[1])[0, 0]
    print(f"  画面中心的尺度 ≈ {np.linalg.norm(o2-o)*10:.2f} mm/px"
          f" ⇒ 660 px 宽约 {(np.linalg.norm(o2-o)*10)*W:.0f} mm")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "up_homography.json").write_text(json.dumps({
        "camera": a.cam, "device_by_id": str(CAMS[a.cam].name),
        "device_node": str(devs[a.cam]),
        "square_m": square_m, "measured_mm": a.measured_mm, "nominal_mm": a.board_mm,
        "H_board_to_px": H.tolist(),
        "Hinv_px_to_board": cv2.invert(H)[1].tolist(),
        "mean_reproj_px": float(np.mean(errs)), "max_reproj_px": float(np.max(errs)),
        "frame_repeatability_mm": rep * 1000,
        "n_frames": okf,
    }, indent=2))

    # ---- 肉眼验收图：把板格按 H 投回去，网格压在板的棋盘格上才算对 ----
    vis = frames[-1].copy()
    cc, ci, _, _ = det.detectBoard(cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY))
    if ci is not None:
        obj = board_points_m(board, ci).astype(np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(obj, H).reshape(-1, 2)
        for p in proj:
            cv2.circle(vis, tuple(np.round(p).astype(int)), 3, (0, 0, 255), -1)
        bad = np.linalg.norm(proj - cc.reshape(-1, 2), axis=1) > 3
        for p in proj[bad]:
            cv2.circle(vis, tuple(np.round(p).astype(int)), 7, (0, 255, 255), 2)
        cv2.putText(vis, f"reproj {np.mean(errs):.2f}px  (>3px 标黄)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIG_DIR / "up_homography_check.png"), vis)
    cv2.imwrite(str(FIG_DIR / "up_homography_raw.png"), frames[-1])
    print(f"\n[save] {OUT_DIR/'up_homography.json'}")
    print(f"[save] {FIG_DIR/'up_homography_check.png'}  ← 必须肉眼回看：红点要压在棋盘角上")
    return 0


if __name__ == "__main__":
    sys.exit(main())
