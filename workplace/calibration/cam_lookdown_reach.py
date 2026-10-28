#!/usr/bin/env python3
"""腕（front）相机**朝下看桌面**时的可达范围 —— 决定 B 路（手眼）能不能做。

背景：B 路要拿腕相机看标定板。实测「front」相机的光轴是**背对夹爪接近方向**的，
所以要让相机朝下，夹爪就得朝上；能摆到哪、半径多大，只能靠 FK 穷举问清楚。

做法：直接**正向枚举**关节角（不用 IK —— 5 关节 5 约束的 IK 容易卡在局部极小，
实测 cam_err 上百 mm），对每个位姿算相机的
- 位置（base 系，mm）
- 光轴方向（link 的 -z）
- 离桌面高度、最低连杆高度、TCP 高度（安全）
筛出「光轴离竖直 ≤ 25°」的位姿，看它们的半径/高度覆盖。

只读，不动臂。

用法：
    python workplace/calibration/cam_lookdown_reach.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                    # noqa: E402
from fk_check import TCP_LINK, CAM_LINK                            # noqa: E402

TILT_MAX = 25.0          # 光轴离竖直的最大夹角


def main() -> int:
    am = A.Arm()
    u = am.u
    seed = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
            "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 20.0}

    grid = itertools.product(
        range(-100, 101, 20),          # shoulder_pan（先固定 0，看平面内的解）
        range(-100, 101, 10),          # shoulder_lift
        range(-90, 91, 15),            # elbow_flex
        range(-90, 91, 15),            # wrist_flex
        range(-150, 151, 30),          # wrist_roll
    )
    rows = []
    for pn, sl, el, wf, wr in grid:
        n = dict(seed, shoulder_pan=float(pn), shoulder_lift=float(sl),
                 elbow_flex=float(el), wrist_flex=float(wf), wrist_roll=float(wr))
        q = A.norm_to_q(n, u)
        Tc = u.fk(q, CAM_LINK)
        c = Tc[:3, 3] * 1000
        ax = -Tc[:3, 2]
        tilt = math.degrees(math.acos(np.clip(-ax[2], -1, 1)))   # 离竖直(朝下)的角
        if tilt > TILT_MAX:
            continue
        z = c[2]
        if not (180 <= z <= 420):
            continue
        clr = min(A.link_clearance(u, q).values())
        if clr < 25:
            continue
        tcp = u.fk(q, TCP_LINK)[:3, 3] * 1000
        rows.append({"n": n, "c": c, "tilt": tilt, "clr": clr, "tcp_z": tcp[2]})

    print(f"[可达] 光轴离竖直 ≤{TILT_MAX:.0f}°、相机高 180–420mm、最低连杆 ≥25mm 的位姿："
          f"{len(rows)} 个")
    if not rows:
        print("  → 腕相机**无法**朝下看桌面。B 路（腕相机看板）不成立。")
        return 0

    r = np.array([np.hypot(x["c"][0], x["c"][1]) for x in rows])
    z = np.array([x["c"][2] for x in rows])
    tz = np.array([x["tcp_z"] for x in rows])
    print(f"  相机半径 r: {r.min():.0f} – {r.max():.0f} mm（中位 {np.median(r):.0f}）")
    print(f"  相机高度 z: {z.min():.0f} – {z.max():.0f} mm")
    print(f"  TCP 高度 : {tz.min():.0f} – {tz.max():.0f} mm")

    # 按 (r, z) 网格看覆盖
    print("\n[覆盖] 行 = 相机半径，列 = 相机高度，数字 = 该格子里的位姿数")
    rs = [0, 50, 100, 150, 200, 250, 300, 350]
    zs = [180, 220, 260, 300, 340, 380, 420]
    print("   r\\z  " + "".join(f"{v:>6}" for v in zs))
    for lo in rs:
        line = f"  {lo:>4} "
        for zl in zs:
            c = int(((r >= lo) & (r < lo + 50) & (z >= zl) & (z < zl + 40)).sum())
            line += f"{c:>6}"
        print(line)

    # 覆盖最广的几个
    from collections import Counter
    rr = Counter((int(x["c"][0] // 50) * 50, int(x["c"][1] // 50) * 50) for x in rows)
    print(f"\n[XY 覆盖] 有解的 50mm 格子数：{len(rr)}")
    xs = sorted({k[0] for k in rr}); ys = sorted({k[1] for k in rr})
    if xs and ys:
        print(f"  x ∈ [{xs[0]},{xs[-1]}] mm   y ∈ [{ys[0]},{ys[-1]}] mm")

    # 视场覆盖估算：能看到的桌面圆斑半径
    print("\n[视场估算] 假设水平 FOV 60°，相机在高度 z 处朝下看到桌面圆斑半径 "
          "≈ z·tan(30°)")
    for zl in (200, 250, 300, 350):
        sel = (z >= zl - 10) & (z < zl + 10)
        if sel.sum():
            print(f"  z≈{zl}mm：{sel.sum()} 个位姿，视场半径 ≈{zl*math.tan(math.radians(30)):.0f}mm"
                  f"  → 覆盖半径 {r[sel].min():.0f}–{r[sel].max():.0f}mm 环带")

    print("\n[样例] 按半径排序取几个：")
    idx = np.argsort(r)
    for k in np.linspace(0, len(idx) - 1, min(8, len(idx))).astype(int):
        x = rows[idx[k]]
        c = x["c"]
        print(f"  r={r[idx[k]]:6.1f} z={c[2]:6.1f}  相机({c[0]:7.1f},{c[1]:7.1f}) "
              f"倾角{x['tilt']:5.1f}° TCP z={x['tcp_z']:6.1f} 最低连杆{x['clr']:5.1f} "
              f" norm=" + ",".join(f"{k2[:4]}={v:.0f}" for k2, v in x["n"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
