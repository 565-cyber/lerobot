#!/usr/bin/env python3
"""利用「**板平贴在桌面上**」这个硬约束，直接联合解出 板↔base 和 相机↔ee。

为什么不直接用 `cv2.calibrateHandEye`（实测踩到的坑）：
标准手眼要同时猜 X = 相机↔ee 的**全部 6 个自由度**，而它唯一的信息来源是
「不同位姿之间的相对转动」。实测 5 个位姿时（倾角 26–56°、方位都在同一侧、
wrist_roll 基本没动），相对转轴几乎共面 ⇒ 解病态：板原点散布 27–37mm，
解出来的板面甚至在桌面下方 34mm。换内参假设、留一法都改不动，说明是**整体性的**。

这条路换了个提法：**不猜 X 的全部自由度，而是把已知的物理约束钉死**。
板是用户平贴在桌上的 ⇒ 板面法线 = base 的 z 轴 ⇒ 板在 base 系里只剩 4 个自由度
（x, y, z, 绕 z 的偏航）。未知数 6(X) + 4(板) = 10，远少于 5 个位姿能提供的 30 个约束。

    T_base_board  =  T_base_ee  ⊗  X  ⊗  T_cam_board
      ↑ 待求(4)      ↑ FK 已知     ↑ 待求(6)   ↑ solvePnP 已知

用 `scipy.optimize.least_squares` 同时最小化所有位姿的残差。

**可验证**：留一位姿不参与拟合，看它能不能被预测出来 —— 这是真检验，
比「散布小」更有说服力。

⚠️ **两个曾经把结果毁掉的坑（都踩过，记在这）**

1. **板法线有上下两个分支，不能只写「纯偏航」。**
   板平贴桌面只说明**板面**是桌面；板坐标系的 z 轴朝上还是朝下是自由的
   （两个法线指同一个平面）。实测数据解出来是**朝下**（n=[−0.075,+0.039,−0.996]）。
   只允许 `Rz(yaw)` 就等于钉死朝上，优化器绕不开那 180°，只好用
   「X 平移撑到 285mm + 把板抬到桌面上方 172mm」去凑 —— 残差只是从 26 涨到 208，
   看着像「有点大」，完全不像少了一个分支。`unpack(..., flip=True)` 就是它。

2. **别用「线性最小二乘」求初值。**
   `Re·A·Rc = G` 这类方程对 (A,G) 是**齐次**的（右边是 0），所以
   **A=G=0 永远是一个精确解**：它一次满足 45 个旋转方程，代价只是 15 个平移方程。
   最小二乘会欣然选它（实测解出 A 的奇异值 0.09、X 转出 179°）。
   正确的初值是**先解自由手眼**（X 6 + 板 12，不含板平约束）—— 它没有这个问题，
   3000 次随机起点**全部**收敛到 rms 26.23，非常稳 —— 再把板的旋转投影到
   最近的那个平分支，最后在板平约束下精修。

3. **板的 z 不能当自由参数拟合，它是已知量。**
   板平贴在桌面上 ⇒ 它的平面高度**就等于桌面高度**，没什么可拟合的。
   留成自由参数的话这一维几乎是**退化**的：钉在 −100mm 和钉在 0，rms 只差
   0.8（28.91 vs 29.70）—— 于是优化器会飘到一个「物理上不可能」的位置
   （实测解出板在桌面**下方** 66.8mm），还会顺手把 X 的平移撑大去配它
   （285mm → 相机跑到腕 link 外面去了）。钉死 z=0 之后 X 平移 = [18,−1,7]mm，
   相机距腕 link 约 19mm，这才对。`--board-z` 可以改这个高度（板的厚度）。

只读离线，不碰臂。

用法：
    python workplace/calibration/solve_flat_board.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                     # noqa: E402
import front_handeye as FH                                          # noqa: E402
from fk_check import Urdf, CAM_LINK                                 # noqa: E402

URDF = Urdf(A.URDF_PATH)

CAL = HERE / "calib"

BOARD_Z = 0.0          # 板图案面在 base 系的高度（米）。main() 里按 --board-z 覆盖。
                       # 板平贴在桌上 ⇒ 这是**已知量**，不是拟合参数（见文件头「坑 3」）。


def load(source: str = "manual_capture.json"):
    """读采集记录 + 内参，返回 [(name, T_base_ee, T_cam_board)]。

    ⚠️ `T_base_ee` 是**从记录里的 `norm` 现算**的，不用 json 里存死的那个 ——
    存的值是采集当时用 `arm.norm_to_q` 算的；那条映射后来改过（半行程从
    URDF 限位改成标定实测行程），旧 JSON 里的值是**错的**。`norm` 才是原始数据。
    """
    Kd = json.loads((CAL / "front_intrinsics.json").read_text())
    K, dist = np.array(Kd["K"]), np.array(Kd["dist"])
    cb, _, _ = FH.board_setup()
    objp = FH.board_object_points(cb)                    # 米

    recs = [r for r in json.loads((CAL / source).read_text()) if r["n_corners"] >= 6]
    out = []
    for r in recs:
        norm = {m: float(r["norm"][m]) for m in A.MOTORS}
        T_ee = URDF.fk(A.norm_to_q(norm, URDF), CAM_LINK)
        ids = np.array(r["ids"], int)
        o = objp[ids].reshape(-1, 1, 3).astype(np.float32)
        im = np.array(r["corners"], np.float32).reshape(-1, 1, 2)
        ok, rv, tv = cv2.solvePnP(o, im, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        T_cb = np.eye(4)
        T_cb[:3, :3] = cv2.Rodrigues(rv)[0]
        T_cb[:3, 3] = tv.ravel()
        out.append((r["tag"], T_ee, T_cb))
    return out


def unpack(p, flip=False):
    """参数 → (X, T_board)。p 共 9 个：X 的 rotvec(3) + 平移(3)，板的 x, y(2) + 偏航(1)。

    板的 **z 用 BOARD_Z（已知）**，不在参数里 —— 见文件头「坑 3」。
    `flip` = 板坐标系的 z 轴朝下（见文件头「坑 1」）。两个分支都要试。
    """
    X = np.eye(4)
    X[:3, :3] = Rotation.from_rotvec(p[0:3]).as_matrix()
    X[:3, 3] = p[3:6]
    Tb = np.eye(4)
    Rz = Rotation.from_euler("z", p[8]).as_matrix()
    Tb[:3, :3] = Rz @ np.diag([1.0, -1.0, -1.0]) if flip else Rz
    Tb[:3, 3] = [p[6], p[7], BOARD_Z]
    return X, Tb


def residuals(p, data, flip=False):
    X, Tb = unpack(p, flip)
    out = []
    for _, T_ee, T_cb in data:
        B = T_ee @ X @ T_cb                       # 这位姿推出的 base←板
        out.extend((Rotation.from_matrix(B[:3, :3].T @ Tb[:3, :3])
                    .as_rotvec() * 1000.0).tolist())      # 姿态残差（放大约当量）
        out.extend(((B[:3, 3] - Tb[:3, 3]) * 1000.0).tolist())   # 位置残差 mm
    return out


def rms(p, data, flip=False):
    r = np.asarray(residuals(p, data, flip))
    return float(np.sqrt((r ** 2).mean()))


def residuals_free(p, data):
    """**不含板平约束**的手眼残差：X(6) + 板(12) 全自由。只用来求初值。"""
    X = np.eye(4)
    X[:3, :3] = Rotation.from_rotvec(p[0:3]).as_matrix()
    X[:3, 3] = p[3:6]
    Tb = np.eye(4)
    Tb[:3, :3] = Rotation.from_rotvec(p[6:9]).as_matrix()
    Tb[:3, 3] = p[9:12]
    out = []
    for _, T_ee, T_cb in data:
        B = T_ee @ X @ T_cb
        out.extend((Rotation.from_matrix(B[:3, :3].T @ Tb[:3, :3])
                    .as_rotvec() * 1000.0).tolist())
        out.extend(((B[:3, 3] - Tb[:3, 3]) * 1000.0).tolist())
    return out


def free_init(data, n_starts: int = 60, seed: int = 0):
    """自由手眼多起点，返回收敛最好的 (X, T_board)。

    实测这一步**很稳**：3000 次随机起点全部收敛到同一个 rms，所以几十次就够。
    (X, T_b) 之间本来就有规范自由度，所以 X 不是唯一确定的 —— 但**板所在的平面**
    是确定的（法线离竖直 4.9°），拿它投影成板平初值正合适。
    """
    rng = np.random.default_rng(seed)
    best = None
    for t in range(n_starts):
        p = (np.zeros(12) if t == 0 else
             np.r_[Rotation.random(random_state=rng).as_rotvec(), rng.normal(0, 0.2, 3),
                   Rotation.random(random_state=rng).as_rotvec(), rng.normal(0, 0.4, 3)])
        try:
            s = least_squares(residuals_free, p, args=(data,), method="lm", max_nfev=20000)
        except Exception:                                  # noqa: BLE001
            continue
        if best is None or s.cost < best.cost:
            best = s
    X = np.eye(4)
    X[:3, :3] = Rotation.from_rotvec(best.x[0:3]).as_matrix()
    X[:3, 3] = best.x[3:6]
    Tb = np.eye(4)
    Tb[:3, :3] = Rotation.from_rotvec(best.x[6:9]).as_matrix()
    Tb[:3, 3] = best.x[9:12]
    return X, Tb


def initial_p(data, n_starts: int = 60, seed: int = 0):
    """自由手眼 → 投影到平分支 → 返回 (p0, flip)。

    板那一项只取偏航（法线方向由 `flip` 定），X 则整份沿用自由解 —— 投影只动了
    板姿态里那 4.9° 的倾斜，X 没动，所以这个初值本身就已经很接近最优。
    """
    X, Tb = free_init(data, n_starts, seed)
    Rb = Tb[:3, :3]
    yaw = math.atan2(Rb[1, 0], Rb[0, 0])
    flip = bool(Rb[2, 2] < 0)                    # 法线朝下 ⇒ 走翻转分支
    p0 = np.r_[Rotation.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3],
               Tb[0, 3], Tb[1, 3], yaw]
    return p0, flip


def fit(data, seed: int = 0, n_starts: int = 60):
    """返回 (sol, flip)：板平约束下的最优解（两个 flip 分支都试）。"""
    p0, flip_guess = initial_p(data, n_starts, seed)
    rng = np.random.default_rng(seed)
    best, best_flip = None, None
    for flip in (flip_guess, not flip_guess):            # 先试初值指的那个分支
        for trial in range(40):
            q = np.array(p0, float)
            if trial:
                q = q + rng.normal(0, [0.25] * 3 + [0.02] * 3 + [0.15, 0.15, 0.4])
            try:
                s = least_squares(residuals, q, args=(data, flip),
                                  method="lm", max_nfev=20000)
            except Exception:                            # noqa: BLE001
                continue
            if best is None or s.cost < best.cost:
                best, best_flip = s, flip
    return best, best_flip


def report(tag, sol, data, flip):
    X, Tb = unpack(sol.x, flip)
    r = np.array(residuals(sol.x, data, flip))
    per = np.linalg.norm(r.reshape(len(data), 6)[:, 3:], axis=1)      # 各位姿位置残差 mm
    print(f"  {tag}: rms={np.sqrt((r**2).mean()):.2f}  最大位置残差 {per.max():.2f}mm  "
          f"板原点 ({Tb[0,3]*1000:.1f},{Tb[1,3]*1000:.1f},{Tb[2,3]*1000:.1f})mm  "
          f"偏航 {math.degrees(math.atan2(Tb[1,0],Tb[0,0])):+.1f}°  "
          f"法线{'朝下' if flip else '朝上'}")
    return X, Tb, per


def main() -> int:
    global BOARD_Z
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="manual_capture.json")
    ap.add_argument("--board-z", type=float, default=0.0,
                    help="板图案面在 base 系的高度（米）。板平贴在桌上 ⇒ 这是已知量，"
                         "不拟合（这一维几乎是退化的，拟合会飘到桌面下方）")
    a = ap.parse_args()
    BOARD_Z = a.board_z

    data = load(a.source)
    print(f"[数据] {len(data)} 个位姿：{', '.join(d[0] for d in data)}")
    print(f"[板高] BOARD_Z = {BOARD_Z*1000:+.1f} mm（钉死，不拟合）\n")

    print("[全部一起拟合]")
    sol, flip = fit(data)
    X, Tb, per = report("联合解", sol, data, flip)
    print(f"    X 平移（相机相对腕 link）= {(X[:3,3]*1000).round(1)} mm"
          f"   ← 相机应在腕 link 附近几十 mm 内")
    print("    逐位姿位置残差 mm：" + "  ".join(f"{d[0]}={p:.2f}" for d, p in zip(data, per)))

    print("\n[留一法：用 N-1 个位姿解，预测被留下的那个]")
    errs = []
    for k in range(len(data)):
        sub = [d for i, d in enumerate(data) if i != k]
        s, fl = fit(sub, seed=k + 1)
        if s is None:
            print(f"  留 {data[k][0]}: 拟合失败"); continue
        Xs, Tbs = unpack(s.x, fl)
        _, T_ee, T_cb = data[k]
        B = T_ee @ Xs @ T_cb                                   # 预测被留位姿
        dt = np.linalg.norm(B[:3, 3] - Tbs[:3, 3]) * 1000
        dr = np.degrees(np.linalg.norm(Rotation.from_matrix(
            B[:3, :3].T @ Tbs[:3, :3]).as_rotvec()))
        errs.append(dt)
        print(f"  留 {data[k][0]:<4} → 预测板原点差 {dt:6.2f}mm   姿态差 {dr:5.2f}°")
    if errs:
        print(f"  **留一预测误差：中位 {np.median(errs):.2f}mm  最大 {max(errs):.2f}mm**"
              f"   ← 这才是有说服力的指标")

    out = {"X_ee_cam": X.tolist(), "T_base_board": Tb.tolist(),
           "board_xyzyaw_mm_deg": [Tb[0,3]*1000, Tb[1,3]*1000, Tb[2,3]*1000,
                                   math.degrees(math.atan2(Tb[1,0],Tb[0,0]))],
           "board_normal_up": (not flip),
           "n_poses": len(data),
           "note": "解时钉死了「板平贴在桌面上」（板面=桌面；板坐标系 z 轴朝上/朝下都允许）"}
    (CAL / "board_to_base.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[save] {CAL/'board_to_base.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
