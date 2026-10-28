#!/usr/bin/env python3
"""扫数据集，逐帧估出「腕相机位置修正」δ，问：δ 是不是关节角的函数？

为什么问：`ball_swap.py` 现在的每帧修正是**拿观测到的方块当锚**做的。如果 δ 能从
`observation.state` 直接算出来，那被夹爪挡住、或者出腕相机视野的帧也能修正 ——
对「跨视角一致替换」来说这是能否上量的分水岭。

判据看**留一**，不看拟合残差：δ 有 3 个分量、5 个本体关节，参数一多必然过拟合
（这个坑在 joint_offset_probe 上踩过：12 参 15 样本压到 10.3mm，但那是假的）。

只读离线，不碰臂。

用法：
    python workplace/calibration/wrist_delta_probe.py --stride 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workplace" / "so101_urdf"))
import arm as A                                                     # noqa: E402
import ball_swap as BS                                              # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset        # noqa: E402

BODY = A.BODY


def bgr(t):
    return cv2.cvtColor(np.clip(t.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--root", default=None)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--cube-h-mm", type=float, default=22.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = a.root or str(REPO / "dataset" / a.dataset)
    ds = LeRobotDataset(a.dataset, root=root)
    g = BS.load_geom()
    z_c = a.cube_h_mm / 2000.0
    bmask = BS.blue_mask_bgr(g)

    rows = []
    for i in range(0, len(ds), a.stride):
        fr = ds[i]
        up, fg = bgr(fr["observation.images.up"]), bgr(fr["observation.images.front"])
        st = fr["observation.state"].numpy().astype(float)
        norm = {m: float(v) for m, v in zip(A.MOTORS, st)}
        T = g["urdf"].fk(A.norm_to_q(norm, g["urdf"]), "wrist_camera_link") @ g["X"]
        ct = BS.contact_point(up, g, 0.5, a.cube_h_mm / 1000.0, bmask)
        if ct is None:
            continue
        P = np.array([ct["base_xy"][0], ct["base_xy"][1], z_c])
        pred = BS.predict_px(g["K"], g["dist"], T, P)
        got = BS.blob_near(bmask(fg), pred, area_range=(250, 6000), max_dist=200)
        if got is None:
            continue
        d, _ = BS.solve_cam_delta(g["K"], g["dist"], T, [(P, got[1])])
        rows.append(dict(frame=i, ep=int(ds.hf_dataset[i]["episode_index"]),
                         q=st.tolist(), delta_mm=(d * 1000).round(2).tolist(),
                         resid_px=round(float(np.hypot(*(got[1] - pred))), 1)))

    print(f"[扫描] {len(rows)} 个可用帧（stride {a.stride}，跳过腕图/up 图里没检出方块的）")
    out = Path(a.out) if a.out else HERE / "calib" / "wrist_delta_scan.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"[存] {out}")

    Q = np.array([r["q"] for r in rows])[:, :5]
    D = np.array([r["delta_mm"] for r in rows])
    print(f"\n|δ| 中位 {np.median(np.linalg.norm(D, axis=1)):.1f}mm  "
          f"范围 {np.linalg.norm(D, axis=1).min():.1f}–{np.linalg.norm(D, axis=1).max():.1f}mm")

    # δ = b + A q，做两种留出：逐帧 LOO 和**整 episode 留出**。
    # 逐帧 LOO 在重复任务的数据上会虚高 —— 相邻帧几乎是同一个样本，留走一个
    # 它的近邻还在训练集里。整 episode 留出才是「换一条演示还能不能预测」。
    X = np.hstack([Q, np.ones((len(Q), 1))])
    Xq = np.hstack([Q, Q ** 2, np.ones((len(Q), 1))])
    for name, Xm in (("δ ≈ b + Σ a_k q_k        (6 参/分量)", X),
                     ("δ ≈ b + Σ a_k q_k + Σ c_k q_k²  (11 参/分量)", Xq)):
        fit = Xm @ np.linalg.lstsq(Xm, D, rcond=None)[0] - D
        loo = np.zeros_like(D)
        for j in range(len(Q)):
            m = np.ones(len(Q), bool); m[j] = False
            loo[j] = Xm[j] @ np.linalg.lstsq(Xm[m], D[m], rcond=None)[0] - D[j]
        ep = np.array([r["ep"] for r in rows])
        lep = np.zeros_like(D)
        for e in np.unique(ep):
            m = ep != e
            lep[~m] = Xm[~m] @ np.linalg.lstsq(Xm[m], D[m], rcond=None)[0] - D[~m]
        def st(v):
            n = np.linalg.norm(v, axis=1)
            return f"中位 {np.median(n):5.2f}  90% {np.percentile(n, 90):5.2f}  最大 {n.max():5.2f}"
        print(f"\n  {name}")
        print(f"    拟合残差   |Δδ| {st(fit)} mm")
        print(f"    逐帧留出   |Δδ| {st(loo)} mm")
        print(f"    episode 留出 |Δδ| {st(lep)} mm   ({len(np.unique(ep))} 条演示)")

    W = np.linalg.lstsq(X, D, rcond=None)[0]
    print("\n  线性模型系数 (mm / 归一化单位):")
    for k, m in enumerate(BODY):
        print(f"    {m:<14} x={W[k,0]:+7.3f}  y={W[k,1]:+7.3f}  z={W[k,2]:+7.3f}")
    print(f"    {'常数':<14} x={W[5,0]:+7.3f}  y={W[5,1]:+7.3f}  z={W[5,2]:+7.3f}")

    # 落盘，给 ball_swap.py --delta-model 用（那边拿来修「方块不在腕图里」的帧）
    mdl = HERE / "calib" / "wrist_delta_model.json"
    mdl.write_text(json.dumps(dict(
        note="δ = W.T @ [q_body..., 1]，δ 是**腕相机在 base 系里的平移修正** (mm)；"
             "在 newdata1 上 episode 留出中位 4.44mm。换台臂/动过相机即作废。",
        joints=list(BODY), dataset=a.dataset, stride=a.stride, n_frames=len(rows),
        W=(W * 1.0).round(6).tolist()), indent=1))
    print(f"[存] {mdl}")
    print("\n判读：**episode 留出**若掉到 ~3-4mm（= 同构型重复测量的一致度），说明 δ 只由构型"
          "\n      决定、可以从 state 算出来，那就**不需要方块在腕图里可见**；若只是逐帧留出好看"
          "\n      而 episode 留出差，那它只在这批演示的轨迹流形上插值，别外推。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
