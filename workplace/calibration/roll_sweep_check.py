#!/usr/bin/env python3
"""**只用视觉**量 `wrist_roll` 的增益 —— 回答「一个 norm 单位到底转了多少度」。

为什么非量不可：板↔base 解出来还剩一个 2–6.5° 的、**X 吸收不掉**的旋转不一致，
而它几乎全部压在 `wrist_roll` 上 —— 5 个位姿里只有 v5 动过手腕（+22.6 norm，其余
都是 +0.4），于是「Δroll≈0 的 6 对差 1.36°」对「Δroll=22 的 4 对差 4.43°」，
差 3.3 倍。但「偏置」是测不出来的（末关节常量偏置被 X 完全吸收，见
`joint_offset_probe.py`），能测的只有**增益**。

原理（不含 X、不含板、不含任何拟合参数）：
同一条固定不动的板，两次拍摄的相机姿态分别是 R_ee_i X、R_ee_j X，而板在
base 系固定 ⇒ 相对旋转满足手眼关系 `A·X = X·B`，其中
    A = R_ee_iᵀ R_ee_j   （FK 侧），  B = R_cb_iᵀ R_cb_j   （PnP 侧）
共轭不改变转角 ⇒ **angle(A) ≡ angle(B)**，与 X 无关。

于是：如果两次拍摄**只动了 wrist_roll**，则 angle(A) 就是 FK 认定的 Δroll，
而 angle(B) 是 PnP 实测的 Δroll。两者的**斜率**就是这条映射的增益是不是 1。

⚠️ 判读前提：得确认「确实只动了手腕」。所以下面把每个关节的 norm 变化也打出来 ——
若其它关节也在飘，斜率就不是纯 roll 的增益，脚本会直接说不合格。

用法：
    python workplace/calibration/roll_sweep_check.py            # 取 r* 标签
    python workplace/calibration/roll_sweep_check.py --prefix r
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                      # noqa: E402
import front_handeye as FH                                           # noqa: E402
from fk_check import Urdf, CAM_LINK                                  # noqa: E402

CAL = HERE / "calib"


def collect(prefix: str):
    """返回 (tags, norms, R_ee, R_cb)，只收 tag 以 prefix 开头且角点够多的记录。"""
    u = Urdf(A.URDF_PATH)
    Kd = json.loads((CAL / "front_intrinsics.json").read_text())
    K, dist = np.array(Kd["K"]), np.array(Kd["dist"])
    cb, _, _ = FH.board_setup()
    objp = FH.board_object_points(cb)

    recs = [r for r in json.loads((CAL / "manual_capture.json").read_text())
            if r["tag"].startswith(prefix) and r["n_corners"] >= 6]
    recs.sort(key=lambda r: float(r["norm"]["wrist_roll"]))
    tags, norms, Rees, Rcbs = [], [], [], []
    for r in recs:
        n = {m: float(r["norm"][m]) for m in A.MOTORS}
        ids = np.array(r["ids"], int)
        o = objp[ids].reshape(-1, 1, 3).astype(np.float32)
        im = np.array(r["corners"], np.float32).reshape(-1, 1, 2)
        _, rv, _ = cv2.solvePnP(o, im, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        tags.append(r["tag"]); norms.append(n)
        Rees.append(u.fk(A.norm_to_q(n, u), CAM_LINK)[:3, :3])
        Rcbs.append(cv2.Rodrigues(rv)[0])
    return tags, norms, Rees, Rcbs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r", help="只看这个前缀的标签（默认 r）")
    ap.add_argument("--max-drift", type=float, default=2.0,
                    help="非 roll 关节最大允许的 norm 跨度（超过就判不合格）")
    a = ap.parse_args()

    tags, norms, Rees, Rcbs = collect(a.prefix)
    if len(tags) < 3:
        print(f"[不足] 前缀 '{a.prefix}' 只找到 {len(tags)} 个有效位姿（要 ≥3，推荐 6–10）")
        return 1

    print(f"[数据] {len(tags)} 个位姿（按 wrist_roll 排序）")
    print(" 标签        wrist_roll")
    other_span = {}
    for m in A.BODY:
        if m == "wrist_roll":
            continue
        other_span[m] = max(n[m] for n in norms) - min(n[m] for n in norms)
    worst_m = max(other_span, key=other_span.get)
    for t, n in zip(tags, norms):
        print(f"  {t:<8}    {n['wrist_roll']:+8.2f}")
    print(f"\n  手腕跨度 {max(n['wrist_roll'] for n in norms) - min(n['wrist_roll'] for n in norms):.2f} norm")
    print("  非 roll 关节跨度：" + "  ".join(
        f"{m}={v:.2f}" for m, v in other_span.items()))
    clean = other_span[worst_m] <= a.max_drift
    print(f"  ⇒ {'只动了手腕 ✓' if clean else f'⚠ {worst_m} 也动了 {other_span[worst_m]:.2f} norm，斜率不纯'}")

    # ---- 逐对：FK 认定的 Δroll vs PnP 实测的 Δroll ----
    print("\n 位姿对        FK Δ角°   PnP Δ角°    差°    Δroll(norm)")
    claimed, measured, droll = [], [], []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            ca = math.degrees(Rotation.from_matrix(Rees[i].T @ Rees[j]).magnitude())
            mb = math.degrees(Rotation.from_matrix(Rcbs[i].T @ Rcbs[j]).magnitude())
            claimed.append(ca); measured.append(mb)
            droll.append(norms[j]["wrist_roll"] - norms[i]["wrist_roll"])
            print(f"  {tags[i]}-{tags[j]:<6} {ca:8.2f}  {mb:8.2f}  {mb-ca:+7.2f}   {droll[-1]:+7.2f}")
    claimed = np.array(claimed); measured = np.array(measured); droll = np.array(droll)

    # 过原点回归：measured = g * claimed  ⇒ g 就是「真实转了多少 / 我们以为转了多少」
    g = float((claimed @ measured) / (claimed @ claimed))
    resid = measured - g * claimed
    g2 = np.polyfit(claimed, measured, 1)
    print(f"\n 过原点拟合增益 g = {g:.4f}   （= 真实 Δroll / FK 认定的 Δroll）")
    print(f"   残差 rms {np.sqrt((resid**2).mean()):.2f}°   最大 {np.abs(resid).max():.2f}°")
    print(f" 带截距拟合 斜率 {g2[0]:.4f}  截距 {g2[1]:+.2f}°"
          f"   ← 截距应≈0（≈0 说明差异确实是增益而不是常数偏置）")
    print(f"\n 判读：g 明显偏离 1（>2% 且残差随 |Δroll| 增长）⇒ 标定文件里 wrist_roll 的 "
          f"range（当前 {_rng()}）不是真实行程，增益要按 g 改。")
    print(f"       g ≈ 1 且残差小 ⇒ 手腕增益没问题，那 2–6° 得另找来源（PnP 系统误差/机械回差）。")
    return 0


def _rng():
    try:
        cal = json.loads(A.FOLLOWER_CAL.read_text())
        return f"{cal['wrist_roll']['range_min']}..{cal['wrist_roll']['range_max']}"
    except Exception:                                                # noqa: BLE001
        return "?"


if __name__ == "__main__":
    sys.exit(main())
