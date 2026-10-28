#!/usr/bin/env python
"""标定第 1 步：new_calib / old_calib 零点判决实验（需真机）。

## 为什么是这一步

官方两套 URDF 的**零点约定**不同（`workplace/so101_urdf/FINDINGS.md`），
q=0 时 `new_calib` 的大臂陡升 +70.7°、`old_calib` 整条臂平躺 +10.3°。
本机用哪套**只能实测**，因为：

- **行程区分不了** —— 两套的关节行程逐字相同（lift 200/200、pan 220/220、
  roll 320/320），因为物理行程没变，只是零点挪了。
- 唯一区分的是 `shoulder_lift` / `elbow_flex` 的**限位中点**：
  new 是 0°/0°（对称），old 是 **−90°/+85°**（不对称）。

这一步决定后面一切：零点错 → `q_real` 错 → `T_real` 错 → 投影全错。

## 关键推论：**不需要驱动机械臂**

两套约定下 `q_urdf = mid + (norm/100)·half`，所以对**同一个 norm**：

| 关节 | q_new | q_old | 差 |
|---|---|---|---|
| shoulder_lift | norm | norm − 90° | **−90°** |
| elbow_flex | 0.968·norm | 85° + 0.95·norm | **≈ +85°** |

⇒ 差值是**常数**（不是只在 norm=0 才显出来）。所以读**当前**归一化值，
按两种假设各渲染一次，跟当前姿态的照片比 —— **臂一动不动就能判**。

## 三个 phase，默认只跑 A

    A（默认，**只读**）  认端口 + 比对 EEPROM 与 json + 印当前姿态 + 渲染两种假设
    B（--go）            把五个本体关节缓动到归一化 0（夹爪不动），回读确认后重拍
    C（--photo）         用 cv2 直接抓相机

A 只读寄存器，不写、不改扭矩、不使能。B 会**真的驱动机械臂**，必须显式给 `--go`。

用法：
    python workplace/calibration/step1_verdict.py                 # 只读
    python workplace/calibration/step1_verdict.py --photo          # 只读 + 抓图
    python workplace/calibration/step1_verdict.py --go --photo     # 会动臂
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
CALIB_DIR = HERE.parent
REPO = HERE.parents[2]
WORKPLACE = HERE.parents[1]
for _p in (str(REPO / "src"), str(WORKPLACE / "so101_urdf")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

URDF_DIR = REPO / "workplace" / "so101_urdf"
FIG_DIR = CALIB_DIR / "figures"

FOLLOWER_CAL = (Path.home() / ".cache/huggingface/lerobot/calibration/robots"
                / "so101_follower/my_awesome_follower_arm.json")
LEADER_CAL = (Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators"
              / "so101_leader/my_awesome_leader_arm.json")

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
BODY = MOTORS[:5]                      # 五个本体关节，RANGE_M100_100
CPD = 4096 / 360.0                     # sts3215：counts per degree（= 11.378）

# 只读探测用的寄存器；Min/Max_Position_Limit 正是 `FeetechMotorsBus.is_calibrated`
# 用来判断「json 与电机是否一致」的那两个（feetech.py:233）。
PROBE_REGS = ["Min_Position_Limit", "Max_Position_Limit", "Homing_Offset"]


# ---------------------------------------------------------------- URDF 限位

def urdf_limits(path: Path) -> dict[str, tuple[float, float]]:
    """{关节名: (lower°, upper°)}。**单位是度**，URDF 里存的是弧度。"""
    out = {}
    for j in ET.parse(path).getroot().findall("joint"):
        lim = j.find("limit")
        if lim is None or lim.get("lower") is None:
            continue
        out[j.get("name")] = (math.degrees(float(lim.get("lower"))),
                              math.degrees(float(lim.get("upper"))))
    return out


def q_urdf_from_norm(norm: float, lo: float, hi: float) -> float:
    """LeRobot 归一化 → URDF 关节角（度）。

    `norm ∈ [−100, 100]` 线性映到 `[lo, hi]`：`norm = −100` → `lo`，
    `+100` → `hi`，`0` → 中点。读自 `motors_bus.py:770` 的 RANGE_M100_100 分支。
    """
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return mid + (norm / 100.0) * half


def q_urdf_from_norm_0_100(norm: float, lo: float, hi: float) -> float:
    """夹爪专用：`Range0_100`，`0 = lo = 闭合端`。**不是中点**（FINDINGS 第二节）。"""
    return lo + (norm / 100.0) * (hi - lo)


def pose_for(norms: dict[str, float], limits: dict[str, tuple[float, float]]) -> dict:
    """{关节名: 弧度}，给 render_urdf.gather 用。"""
    q = {}
    for m in MOTORS:
        if m not in limits:
            continue
        lo, hi = limits[m]
        n = norms[m]
        deg = (q_urdf_from_norm_0_100(n, lo, hi) if m == "gripper"
               else q_urdf_from_norm(n, lo, hi))
        q[m] = math.radians(deg)
    return q


# ---------------------------------------------------------------- 硬件：只读

def probe_port(port: str) -> dict | None:
    """只读探测一个串口：ping 1..6 号电机 + 读三个 EEPROM 寄存器。

    **不写任何寄存器、不改扭矩、不使能** —— `disconnect(disable_torque=False)`
    是刻意的：默认值会在断开时关扭矩，那也是一种状态改变。
    """
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {m: Motor(i + 1, "sts3215",
                       MotorNormMode.RANGE_0_100 if m == "gripper"
                       else MotorNormMode.RANGE_M100_100)
              for i, m in enumerate(MOTORS)}
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)
    try:
        bus.connect(handshake=True)            # 只 ping，不写
    except Exception as e:                      # noqa: BLE001
        print(f"  {port}: 连不上（{type(e).__name__}: {str(e).strip()[:80]}）")
        return None
    try:
        reg = {}
        for m in MOTORS:
            reg[m] = {r: bus.read(r, m, normalize=False) for r in PROBE_REGS}
            reg[m]["Present_Position"] = bus.read("Present_Position", m, normalize=False)
        return reg
    finally:
        bus.disconnect(disable_torque=False)   # ← 关键，见 docstring
        print(f"  {port}: 读完，已断开（扭矩状态未改）")


def match(reg: dict, cal: dict) -> tuple[bool, list[str]]:
    """电机 EEPROM 里的 range 是否与某个标定 json 逐关节相同。"""
    bad = [m for m in MOTORS
           if reg[m]["Min_Position_Limit"] != cal[m]["range_min"]
           or reg[m]["Max_Position_Limit"] != cal[m]["range_max"]]
    return (not bad), bad


# ------------------------------------------------- 跨约定可比量：大臂仰角

# 两套 URDF 的**底座坐标系相差 90°**（new 的大臂沿 −X 伸、old 沿 +Y），
# 所以「TCP 在 base 系的位置」**不能跨约定比** —— 差的那 8–11 cm 全是底座
# 旋转造成的假象。唯一能跨约定直接比的是 **Z 方向**：相对地面的仰角。
# （`so101_urdf/FINDINGS.md` 已记过这点，这里把它变成代码。）
LINKS = {
    "new_calib": ("so101_new_calib.urdf", "upper_arm_link", "lower_arm_link", "wrist_link"),
    "old_calib": ("so101_old_calib.urdf", "upper_arm", "lower_arm", "wrist"),
}


def elevations(norms: dict[str, float]) -> dict[str, tuple[float, float]]:
    """{约定: (大臂仰角°, 小臂仰角°)}。仰角 = 关节点连线与水平面的夹角。"""
    from fk_check import Urdf
    out = {}
    for tag, (fn, up, lo_, wr) in LINKS.items():
        u = Urdf(URDF_DIR / fn)
        q = pose_for(norms, urdf_limits(URDF_DIR / fn))
        v1 = u.fk(q, lo_)[:3, 3] - u.fk(q, up)[:3, 3]      # 肩→肘
        v2 = u.fk(q, wr)[:3, 3] - u.fk(q, lo_)[:3, 3]      # 肘→腕
        deg = lambda v: math.degrees(math.asin(                      # noqa: E731
            np.clip(v[2] / np.linalg.norm(v), -1, 1)))
        out[tag] = (deg(v1), deg(v2))
    return out


def degeneracy_proof() -> None:
    """本文件的**主结论**（2026-10-07）：两套约定在姿态上不可分，也不必分。

    三件事一次算完：
      1. 全量程扫描：同一个 norm 下两套约定的大臂仰角逐点相同（差恒为 0.0°）；
      2. 根因：把 old 的「肩→肘」向量绕 Z 转 +90° 后与 new 夹角 0.00°
         —— 底座坐标系差 90°，而 shoulder_lift 零点也差 90°，两个 90° 抵消；
      3. 唯一真判据：行程。六关节里五个两套标称相同，只有 elbow 不同
         （new 193.66° vs old 190°），实测 193.9° ⇒ 弱偏向 new。
    """
    from fk_check import Urdf

    print("\n【去退化证明】两套约定在形态上是不是同一个东西？")
    print(f"  {'norm':>6s} {'new大臂°':>10s} {'old大臂°':>10s} {'差°':>6s}")
    worst = 0.0
    for n in range(-100, 101, 25):
        ns = {m: float(n) for m in MOTORS}
        e = elevations(ns)
        dn = abs(e["new_calib"][0] - e["old_calib"][0])
        worst = max(worst, dn)
        print(f"  {n:6d} {e['new_calib'][0]:10.1f} {e['old_calib'][0]:10.1f} {dn:6.1f}")
    print(f"  ⇒ 全量程最大差 {worst:.1f}° ⇒ **同一个 norm 就是同一个物理姿态**，"
          "\n     所以「把臂摆到某处看姿态」永远判不出约定 —— 别再上真机做这个实验。")

    # 根因：两个 90° 抵消。用向量角直接验。
    import numpy as _np
    ns = {m: 0.0 for m in MOTORS}
    un = Urdf(URDF_DIR / LINKS["new_calib"][0])
    uo = Urdf(URDF_DIR / LINKS["old_calib"][0])
    qn = pose_for(ns, urdf_limits(URDF_DIR / LINKS["new_calib"][0]))
    qo = pose_for(ns, urdf_limits(URDF_DIR / LINKS["old_calib"][0]))
    # 肩→肘 = 肘链节原点 − 肩链节原点（与 elevations() 同一口径）
    vn = (un.fk(qn, LINKS["new_calib"][2])[:3, 3]
          - un.fk(qn, LINKS["new_calib"][1])[:3, 3])
    vo = (uo.fk(qo, LINKS["old_calib"][2])[:3, 3]
          - uo.fk(qo, LINKS["old_calib"][1])[:3, 3])

    def rotz(v, deg):
        t = math.radians(deg)
        return _np.array([math.cos(t) * v[0] - math.sin(t) * v[1],
                          math.sin(t) * v[0] + math.cos(t) * v[1], v[2]])

    def ang(a, b):
        c = float(a @ b) / (_np.linalg.norm(a) * _np.linalg.norm(b))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    print(f"\n  肩→肘  new = {_np.round(vn, 3)}")
    print(f"  肩→肘  old = {_np.round(vo, 3)}")
    print(f"  old 绕 Z 转 +90° 后与 new 的夹角 = {ang(vn, rotz(vo, 90)):.2f}°"
          "   ← 0.00 就是「只差一个方位角」")
    print(f"  （不转 Z 直接比：{ang(vn, vo):.2f}° —— 这 90° 是底座系的，不是姿态的）")

    # 唯一真判据
    if FOLLOWER_CAL.is_file():
        cal = json.loads(FOLLOWER_CAL.read_text())
        lim_n = urdf_limits(URDF_DIR / "so101_new_calib.urdf")
        lim_o = urdf_limits(URDF_DIR / "so101_old_calib.urdf")
        print("\n【唯一有判别力的量：行程】（谁扫到硬限位就与谁的标称值对齐）")
        print(f"  {'关节':14s} {'实测°':>8s} {'new°':>9s} {'Δnew':>7s}"
              f" {'old°':>9s} {'Δold':>7s}")
        for m in MOTORS:
            t = (cal[m]["range_max"] - cal[m]["range_min"]) / CPD
            dn_, do_ = lim_n[m][1] - lim_n[m][0], lim_o[m][1] - lim_o[m][0]
            same = "（两套相同）" if abs(dn_ - do_) < 0.01 else ""
            print(f"  {m:14s} {t:8.1f} {dn_:9.2f} {abs(t - dn_):7.2f}"
                  f" {do_:9.2f} {abs(t - do_):7.2f} {same}")
        print("  ⇒ 五个关节两套逐字相同，只有 elbow_flex 分得开："
              "\n     实测 193.9° 距 new 0.23°、距 old 3.89° ⇒ **弱偏向 new_calib**。")


def homing_diagnostic(cal: dict, lim_new: dict, lim_old: dict) -> None:
    """**已作废（2026-10-07）—— 这条是循环论证，不要再用它出结论。**

    原逻辑：`set_half_turn_homings`（`motors_bus.py:686`）把 homing offset 写成使
    当前位姿的 present position 恰为 2047；标定时用户被告知此时把臂摆在**行程中点**。
    于是算 `obs = (2047 − 中点)/行程 × 200`，与两套约定的预言比。

    错在哪：`obs` 恒等于 ≈0（因为标定流程就要求用户把臂停在行程中点），而
    `pred["new"]` 也恒等于 0（new 的 URDF 零点按定义就在限位中点）。**两边都由构造
    决定，与真机无关** ⇒ 它永远报 new_calib。和「摆到归一化 0 看姿态」是同一类错：
    把「同一个数值 q」当成了「同一个物理姿态」。

    留着这个函数只为记录这个错误，不再参与判决。
    """
    print("\n【半圈位姿诊断】—— 已作废，不再出票")
    print("  它恒报 new_calib：obs≈0 由标定流程定死，pred_new=0 由定义定死。"
          "\n  与真机无关，是循环论证。见 FINDINGS.md 第一节。")


# ---------------------------------------------------------------- 渲染

def render_hypotheses(norms: dict[str, float], out: Path) -> Path:
    """两种零点假设下、**同一个实测 norm** 的姿态并排，图上标的是**仰角**。

    ⚠️ 两幅**不能按像素重叠**比 —— 底座坐标系差 90°。只能比各自标注的仰角，
    以及"大臂相对地面陡不陡"。这是这个实验最容易看走眼的地方。
    """
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from render_urdf import draw

    new_p, old_p = URDF_DIR / "so101_new_calib.urdf", URDF_DIR / "so101_old_calib.urdf"
    lim_new, lim_old = urdf_limits(new_p), urdf_limits(old_p)
    elev = elevations(norms)
    d_big = abs(elev["new_calib"][0] - elev["old_calib"][0])

    ims = []
    for tag, p, lim, key in (("new_calib", new_p, lim_new, "new_calib"),
                             ("old_calib", old_p, lim_old, "old_calib")):
        e1, e2 = elev[key]
        ims.append((tag, draw(p, pose_for(norms, lim), 760, 620, 125.0, 22.0, 35.0,
                              f"{tag} | big-arm {e1:+.1f}deg  forearm {e2:+.1f}deg", ()),
                    e1, e2))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    for ax, (tag, im, e1, e2) in zip(axes, ims):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{tag}：大臂仰角 {e1:+.1f}°   小臂仰角 {e2:+.1f}°", fontsize=11)
    ns = " ".join(f"{norms[m]:+.1f}" for m in MOTORS)
    # 注意：matplotlib 不解析 markdown 的 `**`，字库里也没有 ⚠️（会出豆腐块），
    # 所以图上的文字一律用纯 ASCII 标记 + 中文，别用 emoji / 星号。
    if d_big < 8:
        head = (f"[!] 这个姿态判不了：两套假设的大臂仰角只差 {d_big:.1f} 度\n"
                "（判别力在行程中点最大，端点处两套收敛到同一个机械止点）")
        col = "crimson"
    else:
        head = (f"两套假设的大臂仰角差 {d_big:.1f} 度 —— 和真机照片比哪一幅吻合")
        col = "black"
    fig.suptitle(
        head + f"\n归一化（pan lift elbow wflex wroll grip）= {ns}\n"
        "[!] 两幅的底座坐标系差 90 度，不能按像素重叠比 —— 只比标注的仰角",
        fontsize=12, color=col)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=85)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 相机

def grab(dev: str, out: Path, warm: int = 8) -> bool:
    """直接开 V4L2 抓一帧。不走 LeRobot 的相机配置（那几个是给数据集用的）。"""
    import cv2
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        return False
    try:
        frame = None
        for _ in range(warm):                  # 头几帧常是黑的/自动曝光没稳
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            return False
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        return True
    finally:
        cap.release()


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(
        description="new_calib / old_calib 零点判决实验")
    ap.add_argument("--ports", nargs="*", default=None,
                    help="默认自动枚举 /dev/ttyACM*")
    ap.add_argument("--go", action="store_true",
                    help="真的驱动机械臂到归一化 0（危险，需确认工作空间清空）")
    ap.add_argument("--photo", action="store_true", help="抓相机")
    ap.add_argument("--cams", nargs="*", default=None,
                    help="相机设备，默认试 /dev/video0 /dev/video2")
    ap.add_argument("--out", default=str(FIG_DIR))
    a = ap.parse_args()
    outdir = Path(a.out)

    follower_cal = json.loads(FOLLOWER_CAL.read_text())
    leader_cal = json.loads(LEADER_CAL.read_text()) if LEADER_CAL.is_file() else None

    # ---------------------------------------------------------- 认端口
    ports = a.ports
    if ports is None:
        ports = sorted(str(p) for p in Path("/dev").glob("ttyACM*"))
    if not ports:
        print("没找到 /dev/ttyACM*。机械臂的串口没插？")
        return 2

    print(f"探测 {len(ports)} 个串口（只读寄存器）……")
    found = {}
    for p in ports:
        reg = probe_port(p)
        if reg is None:
            continue
        is_f, bad_f = match(reg, follower_cal)
        is_l, _ = match(reg, leader_cal) if leader_cal else (False, [])
        tag = "**跟随臂**" if is_f else ("leader 臂" if is_l else "未知/未标定")
        print(f"  → {p}: {tag}"
              + (f"（{len(bad_f)} 个关节的 range 与 follower json 不符）"
                 if not is_f else ""))
        found[p] = (reg, is_f, is_l)

    foll_port = next((p for p, (_, f, _) in found.items() if f), None)
    if foll_port is None:
        print("\n没有任何串口的 EEPROM 与 follower 标定 json 一致 —— 停下。")
        print("可能：臂没上电 / 标定文件不是这台臂的 / 需要先跑标定。")
        return 3
    print(f"\n跟随臂在 {foll_port}")

    # ---------------------------------------------------------- 行程对照
    # 目的：再确认一次「行程区分不了」这个前提本身（本文件开头那条结论）。
    lim_new = urdf_limits(URDF_DIR / "so101_new_calib.urdf")
    lim_old = urdf_limits(URDF_DIR / "so101_old_calib.urdf")
    print("\n【行程对照】标定 vs 两套 URDF —— 看它**分不分得开**")
    print(f"  {'关节':14s} {'标定行程°':>10s} {'new限位°':>10s} {'old限位°':>10s}"
          f" {'new中点°':>10s} {'old中点°':>10s}")
    for m in MOTORS:
        if m not in follower_cal or m not in lim_new:
            continue
        t = (follower_cal[m]["range_max"] - follower_cal[m]["range_min"]) / CPD
        n_lo, n_hi = lim_new[m]
        o_lo, o_hi = lim_old[m]
        print(f"  {m:14s} {t:10.1f} {n_hi - n_lo:10.2f} {o_hi - o_lo:10.2f}"
              f" {(n_lo + n_hi) / 2:10.2f} {(o_lo + o_hi) / 2:10.2f}")
    print("  行程列几乎两两相同 ⇒ 行程判不了。中点列 new/old 在 shoulder_lift、"
          "elbow_flex 上差 90°/85° ⇒ **只有中点判得了**。")

    # 这一条不用看图、不用动臂 —— 先跑它。
    homing_diagnostic(follower_cal, lim_new, lim_old)

    # ---------------------------------------------------------- 当前姿态
    reg, _, _ = found[foll_port]
    raw = {m: reg[m]["Present_Position"] for m in MOTORS}
    norms = {}
    for m in MOTORS:
        lo, hi = follower_cal[m]["range_min"], follower_cal[m]["range_max"]
        bounded = min(hi, max(lo, raw[m]))
        norms[m] = ((bounded - lo) / (hi - lo)) * 200 - 100 if m != "gripper" \
            else ((bounded - lo) / (hi - lo)) * 100
        if bounded != raw[m]:
            print(f"  ⚠️ {m}: raw {raw[m]} 超出标定 range [{lo},{hi}]，已夹紧")

    print("\n【当前姿态】")
    print(f"  {'关节':14s} {'raw':>7s} {'归一化':>9s} {'q_new°':>9s} {'q_old°':>9s}")
    for m in MOTORS:
        if m not in lim_new:
            continue
        qn = (q_urdf_from_norm_0_100(norms[m], *lim_new[m]) if m == "gripper"
              else q_urdf_from_norm(norms[m], *lim_new[m]))
        qo = (q_urdf_from_norm_0_100(norms[m], *lim_old[m]) if m == "gripper"
              else q_urdf_from_norm(norms[m], *lim_old[m]))
        print(f"  {m:14s} {raw[m]:7d} {norms[m]:9.1f} {qn:9.1f} {qo:9.1f}")
    print("  ↑ 注意：这两列是**各自约定下的关节角标签**，差是个常数；"
          "但**不能**由此推出姿态差也是常数 —— 见下面仰角。")

    elev = elevations(norms)
    e_n, e_o = elev["new_calib"], elev["old_calib"]
    d_big = abs(e_n[0] - e_o[0])
    print(f"\n【跨约定可比量：相对地面的仰角】（底座坐标系差 90°，位置不可比，仰角可比）")
    print(f"  {'假设':10s} {'大臂仰角°':>10s} {'小臂仰角°':>10s}")
    print(f"  {'new_calib':10s} {e_n[0]:10.1f} {e_n[1]:10.1f}")
    print(f"  {'old_calib':10s} {e_o[0]:10.1f} {e_o[1]:10.1f}")
    print(f"  ⇒ 大臂仰角差 {d_big:.1f}°。**这个差在任意姿态都≈0** —— 见下面全量程扫描。")

    degeneracy_proof()

    p = render_hypotheses(norms, outdir / "verdict_hypotheses.png")
    print(f"\n[save] {p}")

    # ---------------------------------------------------------- 抓图
    if a.photo:
        cams = a.cams or ["/dev/video0", "/dev/video2"]
        print("\n【抓图】")
        for d in cams:
            dst = outdir / f"pose_{Path(d).name}.png"
            ok = grab(d, dst)
            print(f"  {d}: {'→ ' + str(dst) if ok else '打不开/抓不到'}")

    # ---------------------------------------------------------- B: 归零
    if a.go:
        print("\n" + "=" * 66)
        print("--go：即将**驱动机械臂**。五个本体关节缓动到归一化 0，夹爪不动。")
        print("=" * 66)
        ans = input("工作空间已清空、臂已上电、确认继续？输入 GO 确认：").strip()
        if ans != "GO":
            print("未确认，跳过驱动。")
            return 0
        from lerobot.robots.so101_follower import (SO101Follower,
                                                   SO101FollowerConfig)
        robot = SO101Follower(SO101FollowerConfig(
            port=foll_port, id="my_awesome_follower_arm",
            calibration_dir=FOLLOWER_CAL.parent))
        robot.connect(calibrate=False)         # 只连，不重标定（不写 EEPROM）
        try:
            robot.bus.enable_torque()
            start = robot.bus.sync_read("Present_Position")
            goal = {m: 0.0 for m in BODY}
            goal["gripper"] = start["gripper"]     # 夹爪保持不动
            n_steps, hz = 120, 30.0
            print(f"缓动 {n_steps} 步 @ {hz:.0f}Hz ……")
            for k in range(1, n_steps + 1):
                t = k / n_steps
                cur = {m: start[m] + (goal[m] - start[m]) * t for m in MOTORS}
                robot.bus.sync_write("Goal_Position", cur)
                time.sleep(1.0 / hz)
            end = robot.bus.sync_read("Present_Position")
            print("  回读（应≈0）："
                  + " ".join(f"{m}={end[m]:+.1f}" for m in MOTORS))
            if a.photo:
                for d in (a.cams or ["/dev/video0", "/dev/video2"]):
                    dst = outdir / f"zero_{Path(d).name}.png"
                    print(f"  {d}: {'→ ' + str(dst) if grab(d, dst) else '打不开'}")
            input("看完姿态后按回车释放扭矩 ……")
        finally:
            robot.bus.disable_torque()
            robot.disconnect()
        print("已释放扭矩。")

    print("\n" + "=" * 66)
    print("结论（2026-10-07 改成）：**这个实验判不了，也不用判**。")
    print("  两套约定把同一个 norm 映到不同的 q，而底座坐标系正好反着差 90°，"
          "\n  两个 90° 精确抵消 ⇒ 同一个 norm 的物理姿态完全相同。")
    print("  ⇒ 直接用 new_calib（官方现行 + 唯一带相机 URDF），别混搭限位。")
    print("  唯一弱证据是 elbow 行程：实测 193.9° vs new 193.66° / old 190°。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
