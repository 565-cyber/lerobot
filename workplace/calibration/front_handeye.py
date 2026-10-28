#!/usr/bin/env python3
"""B 路：**腕（front）相机看 ChArUco 板** → 手眼标定出 **板系 ↔ base 系**。

为什么走这条（而不是 A 路用夹爪尖端）：A 路卡在**分割**上 —— 白色的臂压在浅色桌面上，
背景差分掩码到处是洞，9 帧里只有臂竖直朝上那 1 帧能可靠分出尖端，只有 1 个可信对应
⇒ 视差模型定不住（见 `fit_tip_model.py`）。B 路检测的是**板**（高对比、结构化，
up 相机上已验证 0.19 px），而且还顺带解出腕相机内参和安装变换。

几何前提（**必须知道**，见记忆 front-camera-optical-axis-is-link-plus-z）：
腕相机光轴 = `wrist_camera_link` 的 **+z** ⇒ **夹爪朝下的自然抓取姿势，相机本来就朝下
看桌面**，不需要把夹爪翻过来。相机比 TCP 高、且在夹爪左侧：实测可用位姿里相机
149–172 mm、TCP 只有 28–53 mm。而 up 相机吊在桌面上方 30–45 cm ⇒ 全臂要压住。

⚠️ 2026-10-08：这里原来写的是「光轴背对夹爪 ⇒ 抬相机=抬 TCP ⇒ 250–350mm 俯瞰不可达」，
那是按 `arm.py` 里错的 `CAM_AXIS=("z",-1)` 推的，**已作废**。

⚠️ 手眼标定的**转动多样性**：所有位姿的相机朝向不能只差一个方位角 —— 那样相对转轴
全平行于竖直轴，AX=XB 退化。实测（`--tilt-max 55`）：含倾角 48.7° 那批时转轴各向
最小占比 0.053（够用）；守卫把大倾角那批砍掉后只剩 2.8–11.9°，掉到 0.011（退化）。
所以**大倾角位姿不能少**，砍之前先验多样性。

整条链：
    板 --H--> up 像素                （已有：calib/up_homography.json）
    板 --solvePnP--> 相机            （本脚本，需要腕相机内参）
    相机 --X--> 腕                                   （calibrateHandEye 解 X）
    腕 --FK--> base                                  （已有：URDF）
    ⇒ base ← 板 = FK · X · solvePnP

子命令：
    scan   粗扫：一批位姿 ⇒ 看哪些能拍到板（≥6 角点）
    solve  离线：标内参 → 手眼 → 写 calib/board_to_base.json

安全：全臂 ≤ `--max-link`，最低连杆 ≥ `--min-clear`，夹爪夹在 `[8,92]`；
`--dry` 只算不走。

用法：
    python workplace/calibration/front_handeye.py scan --dry
    python workplace/calibration/front_handeye.py scan
    python workplace/calibration/front_handeye.py solve
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A                                                    # noqa: E402
from cam_id import resolve                                         # noqa: E402
from fk_check import TCP_LINK, CAM_LINK                            # noqa: E402

CAL = HERE / "calib"
FIG = HERE / "figures"
SHOTS = CAL / "front_shots"

FRONT_CAM = "camera_B"          # 腕上那台（带序列号）
UP_CAM = "camera_A"             # 俯视那台

# 安全高度要盯的连杆。**必须含 TCP**（= gripper_frame_link）—— 它是夹爪尖端，
# 比 gripper_link 还往前 98mm，是整条臂最高的地方；漏了它就会让尖端顶到 up 相机。
LINKS = ["shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link",
         "gripper_link", "wrist_camera_mount_link", "wrist_camera_link",
         "gripper_frame_link", "moving_jaw_so101_v1_link"]

PAN_DEG_PER_NORM = 110.0        # shoulder_pan: norm ±100 ↔ ±110°


# ------------------------------------------------------------------ 标定板

def board_setup():
    b = json.loads((HERE / "board" / "charuco_board.json").read_text())
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    sq, mk = b["square_mm_nominal"] / 1000.0, b["marker_mm_nominal"] / 1000.0
    cb = cv2.aruco.CharucoBoard((b["squares_x"], b["squares_y"]), sq, mk, d)
    return cb, cv2.aruco.CharucoDetector(cb), b


def board_object_points(cb) -> np.ndarray:
    """板系里所有**内角点**的 3D 坐标（z=0），顺序与 detectBoard 的 charucoIds 一致。

    板系约定与 `up_homography.json` 一致（原点=棋盘左上角，+x 沿 150mm 边）。
    """
    return cb.getChessboardCorners()


# ------------------------------------------------------------------ 位姿枚举

def cam_axis_of(Tc, sign: float):
    """相机光轴（base 系，单位向量）。`sign=+1` 取 link 的 +z，`-1` 取 -z。

    ⚠️ 2026-10-08 实测更正：**物理光轴是 link 的 +z，不是 -z**。
    `arm.py` 里的 `CAM_AXIS = ("z", -1)` 是错的（或至少这台机子上是反的）：
    按 -z 规划的「朝下」位姿，腕相机实际全在**拍天花板**（见 figures/front_first3.png）。
    用错了符号，整批位姿就是背对着桌面拍的，还会白跑一趟臂。
    """
    return sign * Tc[:3, 2]


def enumerate_flat(u, max_link: float, min_clear: float,
                   h_lo: float, h_hi: float, tilt_max: float, cam_sign: float = 1.0):
    """正向穷举关节角（pan 固定 0，靠对称性后面再转），找出「相机俯视桌面」的位姿。

    返回里 pts 是 pan=0 时的桌面视点 (mm)，az_local 是它的本地方位角（度）。
    """
    out = []
    for sl in range(-100, 101, 10):
        for el in range(-90, 91, 15):
            for wf in range(-90, 91, 15):
                for wr in range(-150, 151, 30):
                    n = {"shoulder_pan": 0.0, "shoulder_lift": float(sl),
                         "elbow_flex": float(el), "wrist_flex": float(wf),
                         "wrist_roll": float(wr), "gripper": 20.0}
                    q = A.norm_to_q(n, u)
                    zs = [(u.fk(q, l)[2, 3] - A.TABLE_Z) * 1000 for l in LINKS]
                    if max(zs) > max_link or min(zs) < min_clear:
                        continue
                    Tc = u.fk(q, CAM_LINK)
                    c, a = Tc[:3, 3], cam_axis_of(Tc, cam_sign)
                    ch = (c[2] - A.TABLE_Z) * 1000
                    if not (h_lo <= ch <= h_hi) or a[2] > -0.2:
                        continue
                    tilt = math.degrees(math.acos(min(1.0, -a[2])))
                    if tilt > tilt_max:
                        continue
                    t = (A.TABLE_Z - c[2]) / a[2]          # 视线到桌面的距离（米）
                    if not (0.10 < t < 0.90):
                        continue
                    hit = (c + t * a)[:2] * 1000
                    r = float(np.hypot(*hit))
                    if not (40 <= r <= 400):
                        continue
                    out.append({"norm": n, "hit_r": r,
                                "hit_az": math.degrees(math.atan2(hit[1], hit[0])),
                                "cam_h": ch, "tilt": tilt, "ray_mm": t * 1000,
                                "max_link": max(zs), "tcp_h": (u.fk(q, TCP_LINK)[2, 3]
                                                               - A.TABLE_Z) * 1000})
    return out


def pick_targets(cand, n_radii, n_azim):
    """铺视点：每个候选位姿靠 pan 能把视点放到**半径为 r_hit 的整圈**上。

    所以按 r_hit 分档（每档取倾角最小的那个），再沿方位均匀取 `n_azim` 个。
    """
    groups: dict[int, list] = {}
    for c in cand:
        groups.setdefault(round(c["hit_r"] / 25.0), []).append(c)
    keys = sorted(groups)
    if len(keys) > n_radii:
        idx = sorted({int(round(i)) for i in np.linspace(0, len(keys) - 1, n_radii)})
        keys = [keys[i] for i in idx]

    poses = []
    for k in keys:
        best = min(groups[k], key=lambda c: c["tilt"])
        for m in range(n_azim):
            az_t = 360.0 * m / n_azim
            dtheta = (az_t - best["hit_az"] + 180) % 360 - 180
            pan_norm = dtheta / PAN_DEG_PER_NORM * 100.0
            if abs(pan_norm) > 100:
                continue
            poses.append({
                "norm": dict(best["norm"], shoulder_pan=float(pan_norm)),
                "want_r": best["hit_r"], "want_az": az_t,
                "cam_h": best["cam_h"], "tilt": best["tilt"],
                "ray_mm": best["ray_mm"], "max_link": best["max_link"],
                "tcp_h": best["tcp_h"]})
    return poses


# ------------------------------------------------------------------ 采集

def find_transit(u, min_clear: float = 60.0, max_link: float = 180.0,
                 tcp_min: float = 120.0):
    """找一个「转运位姿」：夹爪抬起来、各连杆都不高 —— 所有目标位姿都经过它。

    为什么需要：目标位姿之间直接走关节空间插值会扫到桌面（实测踩过一次：
    插值到 45% 时最低连杆 −49 mm，判定后中止）。走「转运点 → 目标 → 转运点」
    的话，只要每段 转运↔目标 安全，整个序列就都安全。
    """
    centres = {m: 0.0 for m in A.MOTORS}
    best, bd = None, 1e9
    for sl in range(-100, 101, 10):
        for el in range(-90, 91, 15):
            for wf in range(-90, 91, 15):
                n = {"shoulder_pan": 0.0, "shoulder_lift": float(sl),
                     "elbow_flex": float(el), "wrist_flex": float(wf),
                     "wrist_roll": 0.0, "gripper": 20.0}
                q = A.norm_to_q(n, u)
                zs = [(u.fk(q, l)[2, 3] - A.TABLE_Z) * 1000 for l in LINKS]
                tcp_h = (u.fk(q, TCP_LINK)[2, 3] - A.TABLE_Z) * 1000
                if min(zs) < min_clear or max(zs) > max_link or tcp_h < tcp_min:
                    continue
                d = abs(sl) + abs(el) + abs(wf)          # 越靠关节行程中位越好
                if d < bd:
                    bd, best = d, n
    if best is None:
        raise SystemExit("找不出转运位姿 —— 放宽 min_clear / max_link / tcp_min")
    return best


def path_min_mm(am, target, steps=11, start=None):
    """插值路径上最低的连杆高度（mm）。`start` 省略则用当前读数。"""
    u = am.u
    start = start or am.read_norm()
    return min(min(A.link_clearance(u, A.norm_to_q(
        {m: start[m] + (target[m] - start[m]) * k / steps for m in A.MOTORS}, u)).values())
        for k in range(steps + 1))


def safe_goto(am, target, seconds=2.5, min_clear=25.0, steps=11):
    """`goto` 之前先检查**整条插值路径**的连杆高度。

    为什么不能只查终点：`goto` 是关节空间线性插值，两点都安全不代表中间安全 ——
    从 shoulder_lift=-100 走到 +35 的直线上，夹爪可能中途扫到桌面。所以按同样的
    插值取 `steps` 个中间点逐个验，任何一个低于 `min_clear` 就中止。
    """
    u = am.u
    start = am.read_norm()
    worst, worst_at = 1e9, None
    for k in range(steps + 1):
        a = k / steps
        n = {m: start[m] + (target[m] - start[m]) * a for m in A.MOTORS}
        lo = min(A.link_clearance(u, A.norm_to_q(n, u)).values())
        if lo < worst:
            worst, worst_at = lo, a
    if worst < min_clear:
        raise SystemExit(f"路径不安全：插值到 {worst_at:.0%} 处最低连杆只剩 {worst:.1f} mm"
                         f"（下限 {min_clear} mm），中止")
    return am.goto(target, seconds=seconds), worst


CAM_W, CAM_H = 640, 480


def open_camera(dev, w=CAM_W, h=CAM_H, warmup=15, verbose=True):
    """开一台相机并把分辨率**显式**设成 w×h —— 返回 cap 或 None。

    ⚠ 为什么非设不可：相机默认给 640×480，但**每次 USB 重新枚举后会回到
    iSpring 的默认 160×120**（实测 2026-10-09 18:16 腕相机就飘成这样）。
    120 像素高的画面里 ChArUco 格子太小，`detectBoard` 返回 **0 个角点** ——
    而采集脚本只看「角点少就跳过」，于是**静默产出废图**，不报错。
    同一天已经因此废掉两张（a5 和 _peek）。
    """
    cap = cv2.VideoCapture(str(dev))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(warmup):            # 丢开头几帧，等自动曝光稳下来
        cap.read()
    if verbose:
        ok, fr = cap.read()
        if ok and fr is not None and (fr.shape[1], fr.shape[0]) != (w, h):
            print(f"[相机 {dev}] ⚠ 实际分辨率 {fr.shape[1]}×{fr.shape[0]} ≠ {w}×{h}"
                  f" —— 这种尺寸检不出 ChArUco")
            sys.stdout.flush()
    return cap


def grab(cap, n=6):
    for _ in range(3):
        cap.read()
    fs = []
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            fs.append(f)
    return fs


def dark_blob(img, k=7, thr=70):
    """在 **up 图**里找腕相机机身那种**很暗**的块 —— 用来判断臂有没有挪到板上方。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (cv2.blur(g, (k, k)) < thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    best, ba = None, 400
    for kk in range(1, nl):
        x, y, w, h, area = st[kk]
        if area > ba and 0.2 < w / max(h, 1) < 5:
            best, ba = (x + w / 2, y + h / 2, area), area
    return best


def cmd_scan(a) -> int:
    cb, det, _ = board_setup()
    devs = resolve()
    for tag in (FRONT_CAM, UP_CAM):
        if tag not in devs:
            raise SystemExit(f"找不到 {tag}")

    am = A.Arm().connect()
    capf = capu = None
    try:
        u = am.u
        seed = am.read_norm()
        seed = dict(seed, gripper=float(np.clip(seed["gripper"], A.GRIP_MIN, A.GRIP_MAX)))
        print(f"[起点] " + "  ".join(f"{m}={seed[m]:+.1f}" for m in A.MOTORS))

        print(f"[枚举] 全臂 ≤{a.max_link}mm、相机高 {a.h_lo}–{a.h_hi}mm、"
              f"倾角 ≤{a.tilt_max}° …")
        cand = enumerate_flat(u, a.max_link, a.min_clear, a.h_lo, a.h_hi,
                              a.tilt_max, a.cam_sign)
        print(f"  {len(cand)} 个候选位姿")
        if not cand:
            raise SystemExit("没有候选 —— 放宽 --max-link / --tilt-max")

        poses = pick_targets(cand, a.n_radii, a.n_azim)
        print(f"\n[计划] {len(poses)} 个位姿（视点 {a.n_radii} 档半径 × {a.n_azim} 个方位）")
        print(f"  {'视点 r':>7}{'az':>6}{'相机高':>8}{'倾角':>7}{'视线长':>8}"
              f"{'最高连杆':>9}{'TCP高':>8}")
        for p in poses:
            print(f"  {p['want_r']:>7.0f}{p['want_az']:>6.0f}{p['cam_h']:>8.1f}"
                  f"{p['tilt']:>7.1f}{p['ray_mm']:>8.0f}{p['max_link']:>9.1f}"
                  f"{p['tcp_h']:>8.1f}")

        capf = open_camera(devs[FRONT_CAM])
        capu = open_camera(devs[UP_CAM])
        if capf is None:
            raise SystemExit("打不开腕相机")
        SHOTS.mkdir(parents=True, exist_ok=True)

        # 转运位姿：夹爪抬起、各连杆不高。每个目标都 转运→目标→转运 地走，
        # 于是只要每一段单独安全，整串就安全（目标之间直连会扫桌面，实测踩过）。
        transit = find_transit(u, min_clear=a.transit_clear,
                               max_link=a.transit_link, tcp_min=a.transit_tcp)
        print(f"\n[转运] " + "  ".join(f"{m}={transit[m]:+.0f}" for m in A.MOTORS)
              + f"   最低连杆/最高/TCP = {min(A.link_clearance(u, A.norm_to_q(transit, u)).values()):.0f}"
                f" / {max(A.link_clearance(u, A.norm_to_q(transit, u)).values()):.0f}"
                f" / {(u.fk(A.norm_to_q(transit, u), TCP_LINK)[2,3]-A.TABLE_Z)*1000:.0f} mm")
        # 先筛掉「转运↔目标」会扫桌面的目标 —— 否则臂会走到一半才中止、停在半路
        keep = []
        for p in poses:
            lo = min(path_min_mm(am, p["norm"], start=transit),
                     path_min_mm(am, transit, start=p["norm"]))
            if lo < a.min_clear:
                print(f"  [跳过] 视点 r={p['want_r']:.0f} az={p['want_az']:.0f}："
                      f"转运路径最低 {lo:.1f} mm < {a.min_clear:.0f} mm")
                continue
            keep.append(p)
        poses = keep
        if not poses:
            raise SystemExit("所有位姿的转运路径都不安全 —— 调 --transit-* 换个转运位姿")
        print(f"  （转运路径逐段验证：{len(poses)} 个位姿保留）")

        # 起步这一段：起点可能就是**贴着桌面**的（夹爪尖 FK 只有 −4mm，实测）。
        # 那不算危险 —— 已经在那儿了；但「起步」和「转运」不是一回事：
        # 起步只要求**别比现在更低**，转运才要求 ≥min_clear。
        seed_clear = min(A.link_clearance(u, A.norm_to_q(seed, u)).values())
        dep_clear = min(a.min_clear, seed_clear)
        if seed_clear < a.min_clear:
            print(f"[起步] 起点自身最低连杆 {seed_clear:.1f} mm（低于下限 {a.min_clear:.0f} mm）"
                  f"—— 起步只要求不比现在更低")
        dep = path_min_mm(am, transit, start=seed)
        if dep < dep_clear:
            raise SystemExit(f"起步路径最低 {dep:.1f} mm < {dep_clear:.1f} mm，中止"
                             f"（先手动把臂抬离桌面再跑）")
        if a.dry:                                     # 只算不动，到这里就够
            print(f"[起步] 起点→转运 最低 {dep:.1f} mm ✓（下限 {dep_clear:.1f} mm）")
            print("\n--dry：只算不动。")
            return 0

        am.goto(seed, seconds=2.0)
        recs = []
        for i, p in enumerate(poses):
            safe_goto(am, transit, seconds=2.0,
                      min_clear=(dep_clear if i == 0 else a.min_clear))
            _, path_min = safe_goto(am, p["norm"], seconds=2.5, min_clear=a.min_clear)
            time.sleep(0.8)
            fs = grab(capf)
            if not fs:
                print(f"  {i+1:>2} 腕相机读不到帧"); continue
            img = fs[len(fs) // 2]
            ch, ci, m, mi = det.detectBoard(img)
            n = 0 if ch is None else len(ch)
            q = A.norm_to_q(am.read_norm(), u)
            Tc = u.fk(q, CAM_LINK)
            up = None
            if capu is not None and capu.isOpened():
                uf = grab(capu, 4)
                if uf:
                    up = uf[len(uf) // 2]
                    cv2.imwrite(str(SHOTS / f"up{i:02d}.png"), up)
            blob = dark_blob(up) if up is not None else None
            f = SHOTS / f"front{i:02d}_n{n:02d}.png"
            cv2.imwrite(str(f), img)
            recs.append({
                "i": i, "n_corners": n, "shot": f.name,
                "corners": None if ch is None else ch.reshape(-1, 2).tolist(),
                "ids": None if ci is None else ci.reshape(-1).tolist(),
                "T_base_ee": Tc.tolist(),
                "cam_mm": (Tc[:3, 3] * 1000).tolist(),
                "axis": (-Tc[:3, 2]).tolist(),
                "norm": {k: float(v) for k, v in p["norm"].items()},
                "tcp_mm": (u.fk(q, TCP_LINK)[:3, 3] * 1000).tolist(),
                "up_blob": blob,
            })
            print(f"  {i+1:>2}/{len(poses)} 角点 {n:>3}"
                  f"{'  ✓ 看见板' if n >= 6 else ''}"
                  f"   相机({Tc[0,3]*1000:>7.1f},{Tc[1,3]*1000:>7.1f},{Tc[2,3]*1000:>6.1f})mm"
                  + (f"  up图暗块=({blob[0]:.0f},{blob[1]:.0f})" if blob else ""))

        (CAL / "front_scan.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False))
        hits = [r["i"] for r in recs if r["n_corners"] >= 6]
        print(f"\n[结果] {len(hits)}/{len(recs)} 个位姿看见板（≥6 角点）：{hits}")
        print(f"[save] {CAL/'front_scan.json'}   帧在 {SHOTS}/")

        tiles = []
        for r in recs:
            v = cv2.imread(str(SHOTS / r["shot"]))
            cv2.putText(v, f"#{r['i']} r={r['norm']['shoulder_pan']:.0f} n={r['n_corners']}",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 0) if r["n_corners"] >= 6 else (0, 0, 255), 2)
            tiles.append(cv2.resize(v, (320, 240)))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        w = max(x.shape[1] for x in rows)
        rows = [np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in rows]
        cv2.imwrite(str(FIG / "front_scan.png"), np.vstack(rows))
        print(f"[save] {FIG/'front_scan.png'}  ← 回看")

        # 收工：先把臂放回原来的姿势（标定板一动不动，臂也别留在半空）
        try:
            am.goto(seed, seconds=2.5)
            print("[收工] 已回到起点姿势")
        except Exception as e:                                     # noqa: BLE001
            print(f"[收工] 回起点失败（臂仍停在最后位姿）：{e}")
        return 0
    finally:
        for c in (capf, capu):
            if c is not None:
                c.release()
        am.disconnect()


# ------------------------------------------------------------------ 求解

def rot_angle_deg(R) -> float:
    """旋转矩阵的转角（度）。**正确公式是 arccos((trace-1)/2)**。

    ⚠️ 这里原来写的是 `arccos(clip(trace/2*3 - 1))`，是错的：它等于
    `0.5 + 3cosθ`，只要 cosθ > 1/6 就被 clip 成 1、角度一律报 0 —— 会把真实的
    姿态离散度**完全掩盖**掉。而离散度正是判断手眼解算质量的主要依据，所以必须修。
    """
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def cmd_solve(a) -> int:
    cb, _, bmeta = board_setup()
    # ⚠️ 单位必须是**米**：`board_object_points` 返回的已经是米（板按 0.025m 构造），
    # 而 `T_base_ee` 也是米。原来这里 `* 1000` 转成毫米，于是 solvePnP 的 tvec 是毫米、
    # FK 的 t 是米，`calibrateHandEye` 两侧平移差 1000 倍 ⇒ 解出来板原点散布 40 米。
    objp = board_object_points(cb)                      # (N,3) 米
    src = Path(a.source)
    if not src.is_absolute():
        src = CAL / src
    recs = json.loads(src.read_text())
    good = [r for r in recs if r["n_corners"] >= 6]
    print(f"[数据] {src.name}：{len(recs)} 个位姿，其中 {len(good)} 个看见 ≥6 角点")
    if len(good) < 4:
        raise SystemExit("看见板的位姿太少 —— 先跑 scan 找到板，再补采")

    imgp, objps, rvecs, tvecs = [], [], [], []
    for r in good:
        ids = np.array(r["ids"], int)
        imgp.append(np.array(r["corners"], np.float32).reshape(-1, 1, 2))
        objps.append(objp[ids].reshape(-1, 1, 3).astype(np.float32))
    size = (640, 480)

    # ---- 1. 内参 ----
    ok, K, dist, rvs, tvs = cv2.calibrateCamera(
        objps, imgp, size, None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)
    errs = []
    for i in range(len(objps)):
        pr, _ = cv2.projectPoints(objps[i], rvs[i], tvs[i], K, dist)
        errs.append(float(np.linalg.norm(pr - imgp[i], axis=2).mean()))
    print(f"\n[内参] rms={ok:.3f} px   fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"  畸变 k1={dist.ravel()[0]:+.4f} k2={dist.ravel()[1]:+.4f} "
          f"p1={dist.ravel()[2]:+.4f} p2={dist.ravel()[3]:+.4f}")
    print(f"  逐位姿重投影误差 px：" + " ".join(f"{e:.2f}" for e in errs))
    (CAL / "front_intrinsics.json").write_text(json.dumps(
        {"K": K.tolist(), "dist": dist.ravel().tolist(), "image_size": list(size),
         "rms_px": float(ok), "per_pose_px": errs,
         "note": "board square 按用户实测 100mm ⇒ 标称 25mm，见 up_homography.json"},
        indent=2, ensure_ascii=False))
    print(f"[save] {CAL/'front_intrinsics.json'}")

    # ---- 2. 手眼 ----
    Rg, tg, Rt, tt = [], [], [], []
    ee = []
    for r in good:
        T = np.array(r["T_base_ee"]).reshape(4, 4)
        ee.append(T)
    for i, r in enumerate(good):
        ids = np.array(r["ids"], int)
        o = objp[ids].reshape(-1, 1, 3).astype(np.float32)
        im = np.array(r["corners"], np.float32).reshape(-1, 1, 2)
        okp, rv, tv = cv2.solvePnP(o, im, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        Rt.append(cv2.Rodrigues(rv)[0]); tt.append(tv)
        T = ee[i]
        Rg.append(T[:3, :3]); tg.append(T[:3, 3].reshape(3, 1))
    for meth, name in ((cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
                       (cv2.CALIB_HAND_EYE_PARK, "PARK"),
                       (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
                       (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
                       (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")):
        try:
            Rc, tc = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=meth)
        except cv2.error as e:
            print(f"  {name}: 失败 {e}"); continue
        T_ee_cam = np.eye(4); T_ee_cam[:3, :3] = Rc; T_ee_cam[:3, 3] = tc.ravel()
        # 逐个位姿算 base←板，看离散度（手眼自洽性）
        Bs = [ee[i] @ T_ee_cam @ np.block([[Rt[i], tt[i]], [0, 0, 0, 1]])
              for i in range(len(good))]
        ps = np.array([B[:3, 3] for B in Bs]) * 1000
        Rs = np.array([B[:3, :3] for B in Bs])
        spread_t = float(np.linalg.norm(ps - ps.mean(0), axis=1).max())
        spread_r = max(rot_angle_deg(Rs[i].T @ Rs[j])
                       for i in range(len(Rs)) for j in range(i + 1, len(Rs)))
        print(f"  {name:11s} 板原点散布 ≤{spread_t:6.2f}mm  姿态散布 ≤{spread_r:5.2f}°")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="粗扫：哪些位姿能拍到板")
    s.add_argument("--dry", action="store_true")
    s.add_argument("--max-link", type=float, default=220.0,
                   help="最高连杆（含 TCP 尖端）上限 mm。up 相机在 300–450mm，留 ≥80mm 余量")
    s.add_argument("--min-clear", type=float, default=25.0)
    s.add_argument("--h-lo", type=float, default=100.0, help="相机高度下限 mm")
    s.add_argument("--h-hi", type=float, default=200.0)
    s.add_argument("--tilt-max", type=float, default=30.0, help="光轴离竖直上限（度）")
    s.add_argument("--cam-sign", type=float, default=1.0, choices=(1.0, -1.0),
                   help="光轴取 link 的 +z(+1, 实测正确) 还是 -z(-1, arm.py 旧约定)")
    s.add_argument("--n-radii", type=int, default=5, help="视点铺几档半径")
    s.add_argument("--n-azim", type=int, default=4, help="每档半径绕几个方位")
    s.add_argument("--transit-clear", type=float, default=60.0,
                   help="转运位姿的最低连杆高度 mm（越高越安全）")
    s.add_argument("--transit-link", type=float, default=180.0,
                   help="转运位姿的最高连杆高度 mm")
    s.add_argument("--transit-tcp", type=float, default=120.0,
                   help="转运位姿的 TCP 高度 mm（抬起来才不扫桌面）")
    s.set_defaults(fn=cmd_scan)

    v = sub.add_parser("solve", help="离线解内参 + 手眼")
    v.add_argument("--source", default="manual_capture.json",
                   help="采集记录（相对 calib/，或绝对路径）")
    v.set_defaults(fn=cmd_solve)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
