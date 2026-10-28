#!/usr/bin/env python3
"""把数据集中**要抓取的那块方块**换成**同一个 3D 小球**，up 和 front 两个相机各渲一遍。

★ 目标块默认是**黄块**（`--target yellow`）。newdata1 的 task 是
  "Pick up yellow block and put it in trash can"：逐帧看 ep1/ep3，被夹起来放进红盒子的
  是黄块，蓝块全程没动、是**干扰物**。另一块自动成为留出检验（`--target blue` 反过来）。
  初版把这两个角色搞反了 —— 换的是干扰物、拿真目标当留出。

「跨视角一致」在这里有确切含义：**两个相机画的是同一个 base 系里的球心 + 半径**，
不是两张各自「看着像」的图。所以链路必须是一张共同的三维图：

    数据集 observation.state(归一化)
      → norm_to_q → FK(CAM_LINK) → T_base_ee
      → ⊗ X_ee_cam（来自 calib/board_to_base.json）→ T_base_cam   [front 用这个投影]

    up 图像 → 目标块掩码 → 顶面形心像素 p
      → Hinv → 板系 (x, y, 0) → T_base_board → base 接触点
      → 球心 = 接触点 + (0, 0, R)                                 [两个相机共用的三维球]

up 相机只有**平面单应**（z=0 精确），所以：
  · 接触点（z=0）用 H 是**精确**的；
  · 球的轮廓有个随高度来的视差，用 `--cam-h-mm` 显式补偿（见 `contact_point` 的推导）。

已知误差（都写进 stderr，别当没有）：
  · 手眼标定本身 ~10mm（留一中位 10.3 / 最大 20.4），在腕相机 ~180mm 处约 20px；
  · up 相机高度是估的，只影响视差补偿那一项（几毫米量级）。

v2：腕链每帧修正（默认开）
  up 链可信，腕链（FK ⊗ X）差 ~30mm 且**随构型变**。既然目标块两个相机里都看得见，
  每帧就量一次「模型预测像素 vs 观测像素」，反解一个相机平移修正 δ（3 DOF，
  最小范数）。**球心不动，只改「腕相机在哪」的估计。** 另一块不参与拟合，当留出
  检验——它跟着对上，才说明「误差确实是个相机平移」这个模型站得住。
  `--no-fix` 退回 v1 行为。

v2 的抹除（`erase_cube`，up 和腕图都做）
  方块**两个相机里都得抹掉**，否则球看着是「叠上去」而不是「替换」。两处踩过的坑：
  · 掩码要盖住**整个剪影**。黄块在 up 图里只有**顶面**是亮的、侧面是暗橄榄色，
    在腕图里亮面/暗面饱和度差到 S110 vs S83。`grow` 太小或阈值卡太紧都会留下一条边；
  · 补洞别用 TELEA —— 洞紧贴着方块自己的暗影，TELEA 把那条暗影拖进洞里、补出一坨
    灰褐斑。桌面是光滑渐变，用 `fill_hole` 的**调和插值**（默认）干净得多。
  `--no-erase` 整个关掉，`--no-erase-shadow` 只关阴影，`--fill telea` 换回旧算法。

⚠ 目标块是**被夹走的那个**，所以它只在「臂还没伸过来」的帧里好好待在桌上。中后段
  它被夹爪挡住 / 已经进了红盒，up 链会直接判不出来。→ 见下面 v4。

v4：夹持 / 搬运期（默认开，`--no-held` 退回旧行为）
  上面那段 ⚠ 原来是对的：方块被夹起来之后 up 链就断了，后半段出不了图。现在补上：
  · **夹持区间**由 `RobotStatePrior` 给（首个 grasping → 首个 placing）。相位只读
    `observation.state` 的表列，不解视频（200 帧的 getitem 要 1.7s，表列是微秒级）；
  · **定位**用的是「前段先锁住方块、之后只在上一帧位置附近找」这一招
    （`carry_cube(prev_px=…)`，窗口 90px）：夹持期方块与腕相机刚性固定，所以这招
    在腕视图里几乎零成本，全夹持段 163/163 帧跟到，形心从 f292 起就钉在 (314,110)；
  · **球心画在腕图里直接观测到的方块形心上**，半径 = **方块自己当尺子**：
    `r = √面积 · R/方块高`（方块是正方形，面积开方就是边长；同一深度上像素比 =
    物理比，深度约掉了，所以不吃 FK 的深度误差）。f340 实测 r=54px，球径 108px
    ≈ 方块 bbox 110px。注：球物理上 24mm、方块实测 21.5~22mm，球**真的**宽 9%；
    想要和方块一模一样大就 `--radius-mm 11`。
    （上一版半径兜到剪影外接圆 r≈66px，球比方块大 20%，用户看出来了 —— 已改掉。）
  · **不用 FK+手眼当锚点** —— 原本的计划是这么写的，实测被否掉：夹持深度 ~110mm 上
    手眼误差被 1/depth 放大，FK 链给 (361.5, 257.8)、直接观测在 (314.1, 110.3)，
    **差 150~155px**。所以 FK 只留在 stderr 里当交叉检验，画图一律用观测。
  · **up 视图这一期也画球**（v5，见下）。方块在 up 图里**是看得见的**（先前判
    「0 像素」是错的，被台面期的标定阈值骗了）：它是夹爪两指之间露出来的橄榄色楔形，
    色相被举起后从 H≈27 漂到 10~20，台面那套阈值 163 帧只抓到 3452px。改用搬运期
    自己的门限 + 邻近追踪（`carry_wedge`，种子取夹取前最后一帧标定掩码的位置），
    163/163 帧跟到，楔形 200~1600px。**擦掉**（补洞会绕开指头亮白像素，不在指头上
    啃缺口）。

v5：up 视图夹持期**也画球**（位置靠观测、大小靠一个名义相机高度）
  v4 里「up 不画球」的理由是「楔形是偏心的一块，量不出球心和半径」——**只对了一半**。
  重做后的结论：
  · **球心不需要高度模型。** 擦除掩码（`up_erase_mask`）取到的其实是方块的**整块
    footprint**（顶面 + 侧面 + 抗锯齿边），不是「楔形的亮半边」：f250 实测 761px /
    bbox 36×36，正好等于该高度下方块应有的像素尺寸。剪影形心 ≈ 方块几何中心的投影
    （底面相对顶面朝天底点偏 (h_cube/D)·|C*−T̄|，剪影形心落在两者中间，与真值差
    ~3px）。腕视图用的也是同一招（画在观测形心上），两个视角口径一致。
  · 逐帧噪声/污染靠**时间平滑**压掉：`up_carry_path` 先对观测形心做 7 帧中值
    （吃掉 f304/f345 这类被臂/盒子带偏的单帧），再做 7 点二次 Savitzky-Golay
    （对二次趋势无相位滞后，快摆段不会拖后腿）。
  · **半径要 D**：`r_up = R·√detJ(xy)·D/(D−h)`，h 直接取 FK 给的**方块几何中心**的
    z（`P = T_tcp @ X_tcp_cube`，这个 X 本来就是拿夹取那一刻的 up 观测 + 夹取帧的
    TCP 位姿解出来的，所以在夹取帧上 h 自动等于方块半高 11mm —— **不要再加偏移**，
    加了 18mm 就是重复计入）。而 D 从这份数据集里**定不出来**（下面「up 高度标定」
    段有三条否掉的路），只能取名义值 `UP_CAM_H_MM`（700mm，旁证区间 450~1400 ⇒
    搬运最高点处半径 ±25%）。D 只进半径、不进位置；位置全部来自观测。
  · 擦除区 = footprint ∪ 球自己的 footprint（和腕视图同一套写法），保证补洞整个躲在
    球底下；球悬空搬运，**不画落影**。
  · 位置的可信度分两档，图上和 stderr 都注明：掩码是**一块**时形心 ≈ 方块中心投影
    （~3px）；**手指横穿方块**时检测掩码裂成两块（实测 f336/348/354/360 两块都是
    方块色 H 16~29 / S 83~149，中间那条白的才是手指），形心落在手指上 ⇒ 位置误差
    上限 ~1 个方块宽（40px）。这两种都是「观测到了才画」，不是编位置。
    （比较过四种形心估计——擦除掩码联合形心 / 最大块 / 检测掩码最大块 / 检测掩码
    联合形心——路径形状与抖动统计都分不开（逐帧最大跳变 30~34px、95% 分位 9.1~10.6px），
    说明瓶颈是遮挡本身，不是估计器。取联合形心：它落在两块中间，最接近真中心。）
  · 验收：桌面帧逐位回归不变（md5 `1cdda71a80e3688274fb72676faf1243`）；夹持段擦完的
    残块不变（残块在画球**之前**统计，口径与 v4 一致）；`info["up_ball"]["on_mask"]`
    报出球心是否落在擦除掩码内。

up 相机的高度 D：从这份数据集里**定不出来**（三条路都试过、都有实测记录）
  模型：俯视相机高 D、天底点 C*，`u_h = C* + k·(u0 − C*)`，`k = D/(D−h)`。
  1) **台面期铰链**（`_up_height_probe.py`，49 帧跨 50 个 episode）：方块在桌面上
     h=21.5mm，Δ(u) = (h/D)·max(0,(C*−T̄)·u) 能定**方向**（C* 在 (430,−685) 附近），
     但 T̄ 只跨 ~95px ⇒ 力臂太短，残差对 D 几乎不敏感：D=150→3.85px、400→1.80、
     1000→1.51、1360→1.52、2000→1.54、∞→1.61。拟合出的 D=1359 不可信。
  2) **FK + 手眼**（`_up_height_fkfit.py`）：拿 FK 的 TCP 位置投到 up 图，再拟合
     (C*, D, base 常量偏移)。RMSE 16.7px、最大 52px，拟合出的偏移 (−218,216,7)mm
     也没有物理意义 —— 这条链在夹持深度上的绝对精度就是不够（见 v4 那段）。
  3) **夹持期铰链**（`_up_height_carryfit.py`）：想借搬运时 T̄ 跨 ~340px 的力臂。
     但顶面/侧面要在搬运光照下用 Otsu-on-V 分开，实测顶面占比 0.14~0.57、|a| 散在
     4~58px ⇒ 数据不可用（拟合给 D=4755mm、夹持残差 18.6px，纯噪声）。
  剩下能用的只是**旁证**，而且互相差得多：单应分解 978mm（fx/fy=2324/3078，主点
  居中的假设不成立 ⇒ 不可信）、视场角反推 ~350~460mm、CAMERAS.md 从像素大小给
  ~200mm、「方块当尺子」（`r=R/cube_h·√A_top`，深度全约掉）在**遮挡最轻的那几帧**
  借 FK 的 h 反推 D≈480~670mm。⇒ 取 `UP_CAM_H_MM = 700`。要做实，得拿实物标准：
  标定板垫到 2~3 个已知高度再拍 up 图（`anchor_base.py::fit_parallax`，从未跑通存档），
  或拿斜放标定板一次标出 up 相机内参。

跑法：
    python workplace/calibration/ball_swap.py --frames 120,2033,3443 --radius-mm 12
    python workplace/calibration/ball_swap.py --frames 246,340,400   # ep0 夹持段
    python workplace/calibration/ball_swap.py --target blue      # 反过来：换干扰物
    python workplace/calibration/ball_swap.py --no-fix           # 对照
    python workplace/calibration/ball_swap.py --no-held          # 关掉夹持期分支（对照）
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workplace" / "so101_urdf"))
import arm as A                                                    # noqa: E402
from fk_check import Urdf, CAM_LINK, TCP_LINK                      # noqa: E402

CAL = HERE / "calib"
FIG = HERE / "figures"
BALL_BGR = np.array([70, 90, 235], dtype=float)      # 偏红的实心球，和蓝方块区分得开


# ---------------------------------------------------------------- 素材装载

def _load_module(name: str, path: Path):
    """按**文件路径**导入 —— 不能往 sys.path 加 object_generalization/，
    那下面的 datasets/ 会把 HuggingFace 的 datasets 顶掉（README 记过这个坑）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                 # dataclass 需要它先注册
    spec.loader.exec_module(mod)
    return mod


def load_geom():
    bom = _load_module("bom", REPO / "workplace/object_generalization/masks/blue_object_mask.py")
    bt = json.loads((CAL / "board_to_base.json").read_text())
    hom = json.loads((CAL / "up_homography.json").read_text())
    kd = json.loads((CAL / "front_intrinsics.json").read_text())
    return {
        "cfg": bom.BlueMaskConfig(),
        "mask_fn": bom.blue_threshold_mask,
        "X": np.array(bt["X_ee_cam"]),
        "T_base_board": np.array(bt["T_base_board"]),
        "H": np.array(hom["H_board_to_px"]),
        "Hi": np.array(hom["Hinv_px_to_board"]),
        "K": np.array(kd["K"], dtype=float),
        "dist": np.array(kd["dist"], dtype=float),
        "urdf": Urdf(A.URDF_PATH),
    }


# ---------------------------------------------------------------- 夹持期
#
# 为什么需要这一段：up 相机只在每段 episode 的前 43~56% 看得见黄块（实测 ep0 46% /
# ep1 56% / ep2 47% / ep3 43%）。臂一伸过去把方块罩住，up 链就断了，现在 render_frame
# 直接 return None —— 后半段一张图都出不来。
#
# 试过、**不行**的路子：workplace/object_generalization/masks/ 下三个追踪器
# （InteractionObjectTracker / FrontInteractionObjectTracker / MultiViewInteractionTracker）
# 在目标一个像素都看不到时一律返回**空掩码**，遮挡只是把搜索框绕最后位置放大、
# 从不预测（occlusion_timeout=150 帧后终态 lost）。夹爪完全挡住方块的那段正是夹持期，
# 所以那套「锁定+邻近追踪」吃不下搬运帧。
#
# 能用的路子：**夹持中方块和相机都刚性固定在夹爪上**。实测 f330→f435 搬运全程，
# gripper_frame_link 投到腕图是恒定像素 —— 所以腕视图里方块位置可以用 FK 直接算：
#
#     P_球心(base) = fk(q, TCP_LINK) @ X_tcp_cube      TCP_LINK = 两指中点
#
# X_tcp_cube 只估一次：取**最后一次 up 图还干净看到方块**的球心位置 P_last，
# 配 grasping 那一帧的 T_base_tcp 反解。方块在那之前一直静止在桌上，所以合法。
#
# up 视图这一段**不画球**：夹持期 up 图里方块掩码是 0 像素（f360~f420 逐帧看过，
# 只有 f350 有 119px 残块）。要画就得知道 90~190mm 高度上的投影，而 up 相机只有
# z=0 的单应 —— 见文件末尾注释，那个高度没有可信模型。所以这一段**不画**：
# 没有可信投影就画不出对的东西，少画一帧不等于错一帧，画错才是错。


def load_prior():
    """按**文件路径**装 RobotStatePrior（同 _load_module 的理由：不能把
    object_generalization/ 加进 sys.path，那下面的 datasets/ 会顶掉 HF 的 datasets）。"""
    return _load_module("rsp", REPO / "workplace/object_generalization/masks/robot_state_prior.py")


def episode_bounds(ds, i):
    """含绝对帧号 i 的那个 episode 的 [起, 止)。"""
    for e in ds.meta.episodes:
        i0, i1 = int(e["dataset_from_index"]), int(e["dataset_to_index"])
        if i0 <= i < i1:
            return i0, i1
    return 0, len(ds)


def episode_phases(ds, i, rsp_mod):
    """从 episode 首帧顺序重放到 i，返回该帧的相位 dict。

    RobotStatePrior 是有状态的（window=15 帧的关节差分），而本脚本按绝对帧号随机
    访问，所以必须重放。阈值本来就是**这套数据集**标出来的（grip_closed_thr=16 /
    grip_open_thr=22，见 robot_state_prior 的 docstring），observation.state 里的
    gripper 值量级一致（搬运期实测 17.4），不需要再换算。
    """
    i0, _ = episode_bounds(ds, i)
    prior = rsp_mod.RobotStatePrior()
    out = prior.update(ds[i0]["observation.state"].numpy().astype(float))
    for k in range(i0 + 1, i + 1):
        out = prior.update(ds[k]["observation.state"].numpy().astype(float))
    return out


HELD_PHASES = ("grasping", "moving")


def estimate_x_tcp_cube(ds, i_grasp, g, cam_h_m, cube_h_m, mask_fn, min_area=80.0):
    """方块几何中心在夹爪(TCP)坐标系里的固定位置。返回 (X(3,), 出处说明) 或 (None, 说明)。"""
    u = g["urdf"]
    z_c = cube_h_m / 2.0
    i0, _ = episode_bounds(ds, i_grasp)
    # 1) 往前找最近一帧 up 图干净看到方块 —— 那就是方块被夹之前的位置（静止不动）
    for k in range(i_grasp, max(i0 - 1, i_grasp - 400), -1):
        fr = ds[k]
        up = cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255,
                                  0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        ct = contact_point(up, g, cam_h_m, cube_h_m, mask_fn, min_area=min_area)
        if ct is None:
            continue
        P_last = np.array([ct["base_xy"][0], ct["base_xy"][1], z_c])
        # 2) grasping 那一帧的 TCP 位姿
        stg = ds[i_grasp]["observation.state"].numpy().astype(float)
        normg = {m: float(v) for m, v in zip(A.MOTORS, stg)}
        T_tcp = u.fk(A.norm_to_q(normg, u), TCP_LINK)
        X = (np.linalg.inv(T_tcp) @ np.append(P_last, 1.0))[:3]
        return X, f"最后观测 f{k} → P_last {np.round(P_last*1000,1)}mm, 夹取帧 f{i_grasp}"
    return None, f"f{i_grasp} 往前 400 帧都没在 up 图里看到方块"


# ------------------------------------------------- 夹持期：腕图里直接找方块
#
# ★ 结论已改：**夹持期的锚点用观测，不用 FK 链。**
#
# 原计划是拿 FK 推球心（P = fk(q, TCP_LINK) @ X_tcp_cube）。实测证否：ep0 f242 是
# **up 链还干净**的一帧（X_tcp_cube 就是拿它算的），把那时的球心投到腕图是
# (354.5, 245.5)，而方块**实际就在** (269.0, 128.0) —— 差 145px。
# （2026-10-14 用现管线在夹持段上独立复核，逐帧都是同一量级：f246 155px、
#   f300 161px、f336 162px、f400 160px —— 数字随帧而变是因为分母在变，
#   结论不变：这条链在搬运深度上误差是 150px 量级。）
# 误差几乎全在腕相机那条手眼链上（X_ee_cam 平移只有 (0.1, 16.8, -10.8)mm，
# 但腕图的误差 ∝ 1/深度：桌面帧 200~240mm 时是 50~60px，所以那时逐帧 δ 修正
# 压得住；搬运期只有 ~110mm，同样的角度/平移误差放大成 100~150px）。
# 换句话说：**手眼链在搬运深度上不可用**，画出来的球会落在一米以外的桌面上。
#
# 而夹持期恰恰是**直接观测最可靠**的时候：方块又大（~110px 宽）又完全没被挡。
# 判据在 ep0 f240~f408 上逐帧量过（不是拍脑袋）：
#   方块亮面 H 21~23 / S 163~176 / V 227~234；（f292 之后这些值逐帧几乎不变
#   —— 这从**观测**这一侧独立证实了「方块与腕相机刚性固定」，先前只有 FK 预测
#   的恒定性这一个间接证据）
#   桌面    H 18~21 / S  57~ 90 / V 187~208       → S 一档就切开
#   红盒子  H177~178/ S 186~219 / V 252~255       → H 一档就切开
# 所以 H∈[10,40] & S≥140 就能同时避开桌面和红盒。S 门槛必须高：S≥78 时整个桌面
# 都过（ep0 f246 出过一个 53795px 的连通域），质心完全没意义。
CARRY_H = (10, 40)
CARRY_S_MIN = 140
CARRY_V_MIN = 140
CARRY_AREA = (1500, 20000)
CARRY_WIN = 90.0            # 追踪窗口：只在上一帧位置附近找（用户说的「邻近追踪」）

# 擦除用的**放一档**阈值。检测判据卡在 S≥140，可方块背光那一圈 / 被红盒反光染红
# 的棱过不了它，只擦检测掩码会在球外留一道黄边（f360/f400 实测，放大 2.5 倍才看清）。
# 在检测掩码 ±15px 的领圈内实测漏掉的那批像素：H 8~45（中位 16）、S 87~99（中位 94，
# 就差一点）、V 中位 157。于是 S 底放到 85、H 放到 [8,45]；红盒 H≈177 仍进不来。
# 进一步放宽会开始吃桌面（H 18~21 / S 57~90）—— 先不放。
CARRY_ERASE_LO = (8, 85, 100)
CARRY_ERASE_HI = (45, 255, 255)

# ---- up 视图里的方块（夹持期）------------------------------------------------
# 先前这里判「up 图那段一个像素都看不到方块」是**错的**——那是被台面期的标定阈值
# 骗了。实测（ep0 f246~f408 共 163 帧，全帧）：方块在 up 图里一直看得见，它是
# 夹爪两指之间露出来的**橄榄色楔形**；之所以标定掩码只抓到 3452px（21px/帧），
# 是因为它被举起来以后离开了暖光/过曝的台面条件，色相从 H≈27 漂到 H≈10~20、
# 饱和度掉到 50~160。换一套搬运期自己的门限，楔形每帧 200~1600px（均值 609px）。
# 台面期的分档（H 18~21 / S 57~90 桌面色）在这里不成立，所以另立一套。
UP_WEDGE_H = (10, 45)
UP_WEDGE_S_MIN = 50
UP_WEDGE_V_MIN = 90
UP_WEDGE_MIN_AREA = 25
UP_WEDGE_WIN = 70.0         # 邻近追踪窗口，和腕图那套同一个思路


def carry_wedge(bgr, prev_px=None, win=UP_WEDGE_WIN):
    """夹持期在 **up 图**里找方块露出来的那部分。返回 dict(c, area, mask, bbox) 或 None。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hc, S, V = (hsv[:, :, j].astype(np.float32) for j in range(3))
    m = ((Hc >= UP_WEDGE_H[0]) & (Hc <= UP_WEDGE_H[1])
         & (S >= UP_WEDGE_S_MIN) & (V >= UP_WEDGE_V_MIN)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    keep = []
    for j in range(1, n):
        a = int(st[j, cv2.CC_STAT_AREA])
        if a < UP_WEDGE_MIN_AREA:
            continue
        if prev_px is not None and float(np.hypot(*(ce[j] - np.array(prev_px)))) > win:
            continue
        keep.append(j)
    if not keep:
        return None
    big = max(st[j, cv2.CC_STAT_AREA] for j in keep)
    # 楔形被手指切成两块，两块的面积能差 3 倍以上（f347：右 1370px / 左 400px），
    # 门槛卡在 0.30 时左边那块会被丢掉 → 手指左边剩一条黄绿擦不掉。放到 0.15，
    # 「别把远处的暖色物体捎进来」这件事交给上面那个 70px 邻近窗口去做。
    keep = [j for j in keep if st[j, cv2.CC_STAT_AREA] >= 0.15 * big]
    mm = np.isin(lab, keep).astype(np.uint8) * 255
    ys, xs = np.nonzero(mm)
    return dict(c=ce[max(keep, key=lambda j: st[j, cv2.CC_STAT_AREA])].copy(),
                area=int(mm.sum() / 255), mask=mm,
                bbox=(int(xs.min()), int(ys.min()),
                      int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                _allc=np.array([xs.mean(), ys.mean()]))


def carry_wedge_track(ds, ig, ir, seed_px):
    """夹爪举起方块的这段里，逐帧在 up 图上跟那个楔形（顺序走，靠上一帧位置当先验）。

    seed_px 用**夹取前最后一帧** up 图里标定掩码给的方块位置 —— 那时方块还在台面上，
    标定阈值是准的；举起来之后才切到这套搬运期门限。跟丢就沿用上次位置（方块在爪上
    相对 up 相机也是刚性运动），标 stale。
    """
    out, prev, stale = {}, np.asarray(seed_px, float), 0
    max_stale = max(30, (ir - ig) // 4)
    for k in range(ig, ir):
        fr = ds[k]
        up = cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255,
                                  0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        got = carry_wedge(up, prev_px=prev)
        if got is not None:
            prev, stale = got["_allc"], 0
            out[k] = dict(got, stale=False)
        elif stale < max_stale:
            stale += 1
            out[k] = dict(c=prev.copy(), area=None, mask=None, bbox=None, stale=True)
    return out


def up_erase_mask(bgr, strict_mask, centre_px, reach=70.0):
    """up 视图擦除用的掩码：楔形**整块**，不只是检测门限抓到的那部分。

    检测门限卡在 S≥50 / **V≥90**，漏的正是楔形的**暗半边**：举起来以后楔形一侧
    离开暖光，V 掉到 55~90（f277 实测：整块 646px 里只有 234px 过得了 V≥90）。
    只按检测掩码擦，擦完手指边就还剩半个楔形 —— 这就是用户看到的「up 没换」。
    另试过把 V 底放到 90 同时把「相接」这一刀放宽到 21px、邻近半径放到 60px，
    结果反而更糟：擦除面积反而更大（均值 1383 vs 1209px、最大 7539 vs 4729），
    擦完残留也更差（真值门限 S≥60 下均值 199 vs 20px、最大 649 vs 227px）——
    放宽「相接」会把手指边上的暖色/阴影一起圈进来，补洞再吐出一片假暖色。

    所以改成：门限 S≥45 / V≥55 取连通块，只留**和检测掩码 7px 内相接**的，
    或**离追踪中心 70px 以内且 ≥25px**的（左右两块骑在中心上，被手指隔开时两块
    相隔 ~40px；再远的暖色物体才不要）。最后外扩 5px 吃抗锯齿边。
    这组数在 ep0 全部 163 帧上量过：擦除面积均值 1255px（旧参数 1383px，反而更小），
    擦完用检测门限回扫的残块均值 0.5px、最大 77px、只有 1/163 帧 >40px
    （旧参数：均值 199px、最大 649px）；邻近半径从 35px 放到 70px 只让均值涨 3.5%。
    剩下那 1 帧的 77px 是擦除边界上的一条黄绿细线，继续放门限就会开始吃手指。
    最后一刀切掉亮白（指头）——补洞在指头上啃缺口比留点黄还难看。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hc, S, V = (hsv[:, :, j].astype(np.float32) for j in range(3))
    loose = ((Hc >= 8) & (Hc <= 48) & (S >= 45) & (V >= 55)).astype(np.uint8) * 255
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    near = cv2.dilate(strict_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, ce = cv2.connectedComponentsWithStats(loose, 8)
    keep = {j for j in range(1, n) if (lab[near > 0] == j).any()}
    if centre_px is not None:
        c = np.asarray(centre_px, float)
        keep |= {j for j in range(1, n)
                 if np.hypot(*(ce[j] - c)) <= reach and st[j, cv2.CC_STAT_AREA] >= 25}
    keep = list(keep)
    cm = strict_mask.copy()
    if keep:
        cm = cv2.bitwise_or(cm, np.isin(lab, keep).astype(np.uint8) * 255)
    cm = cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    finger = ((V >= 218) & (S <= 40)).astype(np.uint8) * 255
    return cv2.bitwise_and(cm, cv2.bitwise_not(
        cv2.dilate(finger, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))))


# ---- up 视图夹持期的球心/半径 --------------------------------------------------
# 球**位置**不需要高度模型：直接用观测 footprint 的形心（见文件头 v5 段）。
# 只有**半径**要 D，而 D 定不出来（见文件头「up 相机的高度 D」段）⇒ 用名义值。
UP_CAM_H_MM = 700.0          # 名义相机高度。要改只改这一处（只影响球的大小）
# 搬运高度 h 直接用 FK 给的**方块几何中心** z（`P = T_tcp @ X_tcp_cube`），不加偏移：
# X_tcp_cube 是拿夹取帧的 P_last（up 图 z=0 链给的接触点 + 方块半高 11mm）反解出来的，
# 所以 P 的 z 在夹取那一刻**就是准的**（实测 f246 给 11.0mm ✓）。先前想加 18mm 是拿
# FK 的 **TCP** z（f246 给 −6.2mm）当基准 —— TCP 在两指中点、比方块中心低 18mm，
# 拿它比就会重复计入一次。


def up_bgr(ds, i):
    fr = ds[i]
    return cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255,
                                0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _med7(x):
    xp = np.pad(np.asarray(x, float), 3, mode="edge")
    return np.array([np.median(xp[j:j + 7]) for j in range(len(x))])


def _sg7(x):
    """7 点二次 Savitzky-Golay（对二次趋势无相位滞后，快摆段不拖后腿）。"""
    c = np.array([-2, 3, 6, 7, 6, 3, -2], float) / 21.0
    return np.convolve(np.pad(np.asarray(x, float), 3, mode="edge"), c[::-1], "valid")


def up_carry_path(ds, ig, ir, up_trk):
    """夹持期 up 视图的**球心路径**：逐帧观测 footprint 形心 → 时间平滑。

    观测：`up_erase_mask` 的形心（= 方块整块 footprint 的形心，不是楔形亮半边的）。
    该帧掩码空就退回追踪器给的中心，仍取不到就留缺口、由后面线性插值补上。
    平滑：7 帧中值（吃掉被臂/红盒带偏的单帧，如 f304 的 y 偏 +20px、f345 偏 +20px）
    → 7 点二次 SG（快摆段 15px/帧 的速度下，中值+SG 的滞后 < 2px，比单帧 3~10px 的
    观测噪声小）。返回 {帧: dict(px=(x,y), obs=bool)}；有效帧太少则返回 {}。
    """
    frames = list(range(ig, ir))
    raw = []
    for k in frames:
        u = up_trk.get(k) or {}
        p = None
        if u.get("mask") is not None:
            cm = up_erase_mask(up_bgr(ds, k), u["mask"], u.get("_allc"))
            if cm.any():
                ys, xs = np.nonzero(cm)
                p = (float(xs.mean()), float(ys.mean()))
        if p is None and u.get("c") is not None and not u.get("stale"):
            p = (float(u["c"][0]), float(u["c"][1]))
        raw.append(p)
    obs = np.array([p is not None for p in raw])
    if obs.sum() < 5:
        return {}
    P = np.array([[p[0], p[1]] if p is not None else [np.nan, np.nan] for p in raw], float)
    idx = np.arange(len(frames))
    for j in (0, 1):
        col = P[:, j]
        good = ~np.isnan(col)
        P[:, j] = _sg7(_med7(np.interp(idx, idx[good], col[good])))
    return {k: dict(px=(float(P[t, 0]), float(P[t, 1])), obs=bool(obs[t]))
            for t, k in enumerate(frames)}


def last_up_px(ds, i_grasp, mask_fn, back=200, min_area=80.0):
    """夹取前最后一帧 up 图里方块的位置（像素）。当搬运期追楔形的种子用。

    台面期用**标定**掩码（那时方块在暖光下的过曝条件里，阈值是照着它量的），
    举起之后才切到搬运期门限。
    """
    i0, _ = episode_bounds(ds, i_grasp)
    for k in range(i_grasp, max(i0 - 1, i_grasp - back), -1):
        fr = ds[k]
        up = cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255,
                                  0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        n, lab, st, ce = cv2.connectedComponentsWithStats(mask_fn(up), 8)
        if n > 1:
            j = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
            if st[j, cv2.CC_STAT_AREA] >= min_area:
                return ce[j], k
    return None, None


def carry_cube(bgr, prev_px=None, win=CARRY_WIN):
    """夹持期在**腕图**里找方块。返回 dict(c=形心, area, w, h, mask, src) 或 None。

    prev_px 给了就只在它 win 像素内找 —— 这就是「前期锁定、之后按邻近位置追踪」，
    夹持期方块不动，所以窗口小、不会跳到红盒子上去。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hc, S, V = (hsv[:, :, j].astype(np.float32) for j in range(3))
    m = ((Hc >= CARRY_H[0]) & (Hc <= CARRY_H[1]) & (S >= CARRY_S_MIN) & (V >= CARRY_V_MIN))
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for j in range(1, n):
        a = int(stats[j, cv2.CC_STAT_AREA])
        w, h = int(stats[j, cv2.CC_STAT_WIDTH]), int(stats[j, cv2.CC_STAT_HEIGHT])
        if not (CARRY_AREA[0] <= a <= CARRY_AREA[1]):
            continue
        if not (0.5 <= w / max(h, 1) <= 2.0):          # 方块该是方的
            continue
        c = cent[j]
        if prev_px is not None and float(np.hypot(*(c - prev_px))) > win:
            continue
        if best is None or a > best[1]:
            best = (j, a, w, h, c)
    if best is None:
        return None
    j, a, w, h, c = best
    return dict(c=c, area=a, w=w, h=h, mask=((lab == j).astype(np.uint8) * 255), src="obs")


def carry_track(ds, ig, ir, min_frames=8):
    """夹取帧 → 释放帧逐帧在腕图里跟方块，返回 {帧号: carry_cube 结果}。

    顺序走一遍是**有意的**：跟踪靠上一帧的位置当先验（邻近追踪）。跟丢的帧沿用
    最后一次看到的位置**和大小**（夹持期方块相对腕相机是刚性的，位置和视角大小都
    不变，所以沿用是合法的，不是瞎猜），标 stale=True；连续跟丢超过
    max(30, 全程/4) 帧就整段作废（那说明判据在这个 episode 上不成立 ——
    整段作废比交一份带错地面真值的数据强）。
    """
    out, prev, area, s_last, stale = {}, None, None, None, 0
    max_stale = max(30, (ir - ig) // 4)
    for k in range(ig, ir):
        fr = ds[k]
        fg = cv2.cvtColor(np.clip(fr["observation.images.front"].permute(1, 2, 0).numpy() * 255,
                                  0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        got = carry_cube(fg, prev_px=prev)
        if got is not None:
            prev, stale = got["c"], 0
            area, s_last = got["area"], float(np.sqrt(got["area"]))
            out[k] = dict(got, stale=False)
        elif prev is not None and stale < max_stale:
            stale += 1
            # 沿用上次的**边长**（不是半径）：夹持期方块相对腕相机刚性，边长不变。
            out[k] = dict(c=prev, area=area, s=s_last, w=None, h=None, mask=None,
                          src="stale", stale=True)
    if prev is None or len(out) < min_frames:
        return {}
    return out


# ---------------------------------------------------------------- 几何

def contact_point(up_bgr, g, cam_h_m, cube_h_m, mask_fn=None, min_area=150.0):
    """掩码 → **接触点**在 base 系里的 (x, y)。mask_fn 不给就默认找蓝块。

    为什么不能直接把掩码形心喂给 Hinv：掩码是**顶面**，它在高度 h 上，而 H 只管
    z=0。同一个像素 p，光线打在 z=0 的 X0 和打在 z=h 的 X_h 不是一回事：

        X_h = C + (1 - h/Hcam) · (X0 - C)        C = 相机天底(桌面上正对光心那点)

    所以要的是反过来的那一步 —— **把 X0 往天底 C 拉回来**，才是顶面中心在桌面上的
    投影，而顶面中心的 (x,y) 就是底面中心的 (x,y)（方块是平放的立方体）。
    不补这一下，球会朝**远离天底**的方向偏出去，up 里偏几 px、front 里放大成十几 px。
    """
    H, W = up_bgr.shape[:2]
    m = (mask_fn or blue_mask_bgr(g))(up_bgr)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area:
        return None
    p_top = c.reshape(-1, 2).mean(0)

    def inv(px):
        q = g["Hi"] @ np.array([px[0], px[1], 1.0])
        return q[:2] / q[2]

    x0 = inv(p_top)

    # ★ 形状门（2026-10-13）：面积绝对下限挡不住**红盒子的锁扣**。方块放进盒子之后
    #   （ep0 f433~f525 约 46 帧），up 图里唯一的暖色物就是盒壁上那个锁扣：轮廓 86px、
    #   bbox 14×11，正好压过调用方给的 min_area=80 —— 于是代码会把锁扣擦掉、在那儿画个球，
    #   也就是**改了红盒子**（红线）。真正的方块面在同位置是 816~914px / bbox 36~39。
    #   用**该处局部 px/m** 算出方块面应有的面积，弱于 15% 的一律判否：
    #   实测面积比是干净双峰（锁扣 0.06~0.08，真方块面 0.68~0.85，遮挡最重也 ≥0.23），
    #   0.15 两侧各有 2 倍以上余量。门槛宁可紧一点 —— 这一刀漏判只会少画一帧，
    #   误判却会擦掉红盒子的锁扣（动到红线）。
    spm, _ = up_scale_px_per_m(g, x0)
    side_px = cube_h_m * spm
    if area < 0.15 * side_px ** 2:
        return None
    # ★ 形状门之二（同一轮）：面积比单独一条挡不住另外两类假阳性 ——（a）**机器人自己的
    #   黄色颚**（URDF 里 `3d_printed` 材质 rgba=(1.0,0.82,0.12)，是黄的！臂一伸过来
    #   探进 up 视野就成了最大暖色块），（b）红盒内壁那块**浅色方板**（形状、大小、
    #   颜色都跟方块面像，面积比 0.224）。裁图逐个看过（figures/_gate_crops.png）。
    #   它们与真方块面的差别在**尺度**：真方块面 bbox 36~39px（两个方向都是），
    #   假阳性 12~23px。所以要求两个方向都够方块尺寸，
    #   取 0.7 倍：真方块面留 43% 余量，上面那批假阳性（22/19/23/12）全判否。
    x_, y_, w_, h_ = cv2.boundingRect(c)
    if min(w_, h_) < 0.70 * side_px:
        return None
    C = inv(np.array([W / 2.0, H / 2.0]))            # 天底：假设主点在图像中心
    xy = C + (1.0 - cube_h_m / cam_h_m) * (x0 - C)
    Tbb = g["T_base_board"]
    pb = Tbb @ np.array([xy[0], xy[1], 0.0, 1.0])
    return {"px": p_top, "xy_board": xy, "base_xy": pb[:2],
            "pull_mm": float(np.linalg.norm((x0 - xy)) * 1000), "mask": m, "contour": c}


def up_scale_px_per_m(g, xy):
    """单应在 xy 处的局部尺度（像素/米）—— 用于把球的半径画到 up 图上。"""
    Hm = g["H"]
    x, y = xy
    h31 = Hm[2, 0] * x + Hm[2, 1] * y + Hm[2, 2]
    def proj(x, y):
        d = Hm[2, 0] * x + Hm[2, 1] * y + Hm[2, 2]
        return np.array([(Hm[0, 0] * x + Hm[0, 1] * y + Hm[0, 2]) / d,
                         (Hm[1, 0] * x + Hm[1, 1] * y + Hm[1, 2]) / d])
    e = 1e-4
    J = np.column_stack([(proj(x + e, y) - proj(x - e, y)) / (2 * e),
                         (proj(x, y + e) - proj(x, y - e)) / (2 * e)])
    return float(np.sqrt(abs(np.linalg.det(J)))), h31


def project(K, dist, T_base_cam, P_base):
    """base 系点 → 像素 + 相机系深度。用 OpenCV 的相机系（X_ee_cam 就是这个系的）。"""
    T = np.linalg.inv(T_base_cam)
    Pc = (T @ np.append(P_base, 1.0))[:3]
    px, _ = cv2.projectPoints(Pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(2), Pc


# ---------------------------------------------------------------- 腕链每帧修正
#
# 背景：up 链（单应 + 常量板锚）可信，腕链（FK(q) ⊗ X）在数据集构型上差 ~30mm，
# 而且**随构型变**——不是常量 X 能吸收的（拟合 6 参数 ΔX 只把 RMSE 26.8→19.3mm，
# 解出 −75mm 荒谬平移）。见 memory/cross-camera-3d-disagreement。
#
# 修法：方块在两个相机里**都看得见**，所以每帧都能直接量出「模型预测的像素 vs
# 实际观测的像素」，反解一个**相机在 base 系里的平移修正** δ。
# 同一帧里这个差近似是纯平移（同帧两个相距 65mm 的方块 Δ 只差 2.2mm），
# 所以 3 个自由度够用 —— 不用去碰关节模型，也就不受柔度/回差困扰。
#
# ★ 这不是在挪球：球心仍在 base 里纹丝不动，改的是**「腕相机在哪」这个估计**。
#   两相机画的仍然是同一个 3D 球。

# 黄块阈值**按相机分开**（2026-10-11 实测；记忆里的 color-thresholds-must-be-measured）：
#   up    黄块 H25-28  S62-98   V213-250   桌面 H12-14 S86 V125  → 靠 **V** 分
#   wrist 黄块 H24-34  S83-110  V228-254  桌面 H18   S39  V233    → 靠 **S** 分
# 一个阈值套两个相机会静默判掉整类像素：f1500 的 up 图黄块 S 只有 62，
# 用腕相机那套 S≥90 就整块消失，up 链直接断掉（踩过）。
#
# 2026-10-13 再踩一次：腕图里方块**两个面亮度差很大** —— 亮面 H24 S110，暗面
# （绿调那一侧）H34 S83，桌面 S39。原来腕图阈值卡 S≥90，只抓到亮面（774px），
# 暗面整块漏掉、擦不干净，球边上支着一角淡绿。放宽到 S≥68 / H≤45 后同一帧
# 抓满 2018px（真值 ~2460px），质心也从 (504,354) 回到 (491,357)。
YEL_HSV = {
    "up":    (np.array([15, 55, 170], np.uint8), np.array([42, 255, 255], np.uint8)),
    "wrist": (np.array([16, 68, 150], np.uint8), np.array([45, 255, 255], np.uint8)),
}


def yellow_mask(bgr, cam="wrist"):
    """黄方块掩码（**BGR** 进，跟 blue_threshold_mask 要 RGB 不一样，别混）。"""
    lo, hi = YEL_HSV[cam]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, k)


def blue_mask_bgr(g):
    """blue_threshold_mask 要 **RGB** 输入，这里包一层让它吃 BGR。"""
    return lambda bgr: g["mask_fn"](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), g["cfg"])


# ⚠ 别再把「蓝=要换的、黄=留出」当默认。newdata1 的 task 是
#   **"Pick up yellow block and put it in trash can"** —— 被夹起来放进红盒子的
#   是**黄块**，蓝块全程没动过（ep1/ep3 逐帧看过）。所以默认 target 是黄的。
#   初始版本把两个颜色搞反了：换的是干扰物、拿目标当留出检验。
def cube_mask_fns(g, cam, target):
    """返回 (目标方块掩码, 另一块掩码, 目标名, 另一块名)，都是吃 BGR 的函数。"""
    blue = blue_mask_bgr(g)
    yellow = lambda bgr: yellow_mask(bgr, cam)                     # noqa: E731
    if target == "yellow":
        return yellow, blue, "yellow", "blue"
    return blue, yellow, "blue", "yellow"


def blob_near(mask, target_px, area_range=(250, 6000), max_dist=None):
    """掩码里离 target_px 最近的连通域 (距离, 形心, 面积)，找不到返回 None。

    **必须先按面积筛** —— 腕相机左上角那本蓝皮书面积上万，不筛的话最近邻会去
    找书，量出来的是书不是方块（这个坑踩过一次，见 color-thresholds-must-be-measured）。
    """
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if not (area_range[0] <= a <= area_range[1]):
            continue
        c = cent[i]
        d = float(np.hypot(c[0] - target_px[0], c[1] - target_px[1]))
        if max_dist is not None and d > max_dist:
            continue
        if best is None or d < best[0]:
            best = (d, c, a)
    return best


def predict_px(K, dist, T_base_cam, P_base):
    return project(K, dist, T_base_cam, P_base)[0]


def solve_cam_delta(K, dist, T_base_cam, targets, iters=8):
    """解相机平移修正 δ：让 π(T(t+δ), P) 落到观测像素 u。targets = [(P_base, u_px), ...]

    单点时欠定 —— **沿视线移动相机，像素纹丝不动**，那个方向不可观测。所以
    在 Gauss-Newton 的每一步取 **最小范数解**（lstsq），修正自动落在垂直于视线的
    分量上，沿视线那一段保持 0。多点时就是普通的过定解。
    """
    def resid(d):
        T = T_base_cam.copy()
        T[:3, 3] = T_base_cam[:3, 3] + d
        return np.concatenate([predict_px(K, dist, T, P) - u for P, u in targets])

    d = np.zeros(3)
    for _ in range(iters):
        r = resid(d)
        J = np.zeros((r.size, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = 2e-4
            J[:, k] = (resid(d + e) - resid(d - e)) / 4e-4
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        d = d + step
        if np.linalg.norm(step) < 1e-7:      # 5mm 尺度上 1e-7 m 已是收敛
            break
    return d, resid(d)


def load_delta_model(path):
    """读 `wrist_delta_probe.py` 落盘的那份：δ = q · W，q = [5 个本体关节, 1]，单位 mm。

    δ 是**腕相机在 base 系里的平移修正**，在 newdata1 上 episode 留出中位 4.44mm。
    好处是不需要方块在腕图里可见 —— 被夹爪挡住或出视野的帧也能修。
    ⚠ 它是**这台臂 + 这个相机安装**的属性：动过相机、重标过臂，就得重跑 probe。
    """
    d = json.loads(Path(path).read_text())
    return np.array(d["W"], float), list(d["joints"]), d


def delta_from_model(model, norm):
    W, joints, _ = model
    q = np.array([norm[m] for m in joints] + [1.0])
    return (q @ W) / 1000.0                      # mm → m


def draw_marker(img, p, color, r=3, thick=1):
    cv2.circle(img, tuple(np.round(p).astype(int)), r, color, thick, cv2.LINE_AA)


def draw_offset(img, p_from, p_to, color):
    """从模型预测位置画到观测位置 —— 这条线就是每帧修正量。"""
    cv2.arrowedLine(img, tuple(np.round(p_from).astype(int)),
                    tuple(np.round(p_to).astype(int)), color, 1, cv2.LINE_AA, tipLength=0.3)


# ---------------------------------------------------------------- 画球

def draw_ball(img, cx, cy, r, rgb=None, light=(-0.42, -0.45), alpha=1.0):
    """画一个带漫反射 + 高光的实心球（半径 r 像素）。"""
    rgb = BALL_BGR if rgb is None else np.asarray(rgb, float)
    H, W = img.shape[:2]
    x0, x1 = max(0, int(cx - r - 2)), min(W, int(cx + r + 2))
    y0, y1 = max(0, int(cy - r - 2)), min(H, int(cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = (xx - cx) / r
    dy = (yy - cy) / r
    d2 = dx * dx + dy * dy
    inside = d2 <= 1.0
    z = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
    lx, ly = light
    lz = 0.72
    n = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / n, ly / n, lz / n
    lam = np.clip(dx * lx + dy * ly + z * lz, 0.0, 1.0)
    shade = 0.22 + 0.78 * lam ** 0.85
    spec = np.exp(-(((dx - lx) ** 2 + (dy - ly) ** 2) / (2 * 0.13 ** 2))) * 210.0
    # 边缘暗化，让球「圆」起来
    rim = np.clip(1.0 - (1.0 - z) * 1.15, 0.0, 1.0)
    patch = img[y0:y1, x0:x1].astype(float)
    for ch in range(3):
        col = rgb[ch] * shade * rim + spec
        patch[..., ch] = np.where(inside, (1 - alpha) * patch[..., ch] + alpha * np.clip(col, 0, 255),
                                  patch[..., ch])
    img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)


def draw_shadow(img, cx, cy, r, scale=(1.25, 0.55), offset=(0.25, 0.45), alpha=0.33):
    """球下方的软阴影（椭圆 + 高斯羽化）。"""
    H, W = img.shape[:2]
    rx, ry = r * scale[0], r * scale[1]
    ox, oy = r * offset[0], r * offset[1]
    # 包围盒要留够：高斯在 (rx, ry) 处还有 exp(-1)=0.37，贴着椭圆切一刀会留下
    # **硬边矩形**（肉眼看就是球旁边一块方方正正的亮/暗斑）。留 2.6σ 让衰减到位。
    px_, py_ = 2.6 * rx, 2.6 * ry
    x0, x1 = max(0, int(cx + ox - px_)), min(W, int(cx + ox + px_))
    y0, y1 = max(0, int(cy + oy - py_)), min(H, int(cy + oy + py_))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-(((xx - cx - ox) / rx) ** 2 + ((yy - cy - oy) / ry) ** 2) / 2.0)
    patch = img[y0:y1, x0:x1].astype(float) * (1 - alpha * g[..., None])
    img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)


def keep_nearest_comp(mask, target_px, area_range=(250, 6000), max_dist=200):
    """只留掩码里离 target 最近的那一个连通域（先按面积筛）。

    腕视图里抹方块要用它 —— 不能直接拿整张蓝掩码去 inpaint，左上角那本蓝皮书
    也在掩码里，一抹就是一大片。
    """
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        if not (area_range[0] <= a <= area_range[1]):
            continue
        d = float(np.hypot(cent[k][0] - target_px[0], cent[k][1] - target_px[1]))
        if max_dist is not None and d > max_dist:
            continue
        if best is None or d < best[0]:
            best = (d, k)
    return None if best is None else ((lab == best[1]).astype(np.uint8) * 255)


def fill_hole(bgr, mask, method="diffuse", iters=400):
    """把 mask 区域补上。

    为什么不用 cv2.inpaint 的 TELEA：它把边界的**结构**往洞里拉，而这里的洞紧贴着
    方块自己投下的暗影，于是把那条暗影拖进洞里，补出来是一坨灰褐斑。桌面基本是
    光滑渐变，**调和插值**（把洞初始化成环上中位色，再反复用邻域均值只在洞内迭代）
    更对路，也不会长出本来没有的暗块。TELEA 仍留作 `--fill telea` 对照。
    """
    if method == "telea":
        return cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
    hole = mask > 0
    if not hole.any():
        return bgr
    out = bgr.astype(np.float32)
    ring = (cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0) & (~hole)
    out[hole] = np.median(bgr[ring].reshape(-1, 3), axis=0) if ring.any() else out[hole]
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    k /= k.sum()
    for _ in range(iters):
        out[hole] = cv2.filter2D(out, -1, k, borderType=cv2.BORDER_REPLICATE)[hole]
    return np.clip(out, 0, 255).astype(np.uint8)


def erase_cube(bgr, mask, grow=13, shadow=True, pad=14, dark_ratio=0.86, max_sat=70,
               fill="diffuse"):
    """把方块（含顶上那圈暗倒角）抹掉，好让球看起来是**替换**而不是叠上去。

    方块**压在桌面上的那圈阴影**也得一起抹：阴影不是蓝色，蓝掩码抓不到，留着就会
    和我们给球画的那道阴影叠成两重。做法是在掩码外扩的一圈里量桌面亮度的中位，
    把明显更暗的像素并进 inpaint 掩码。**只在这个局部里找** —— 不然机械臂自己的
    影子也会被算进去。

    两个防误伤的闸门（踩过坑）：
      - `max_sat`：只吃**低饱和**的暗像素。桌面阴影是灰的，而外扩圈常常正好够到
        旁边那本**蓝皮书**——书的暗边是饱和的，不加这个闸门就会被当成阴影抹掉，
        沿书边糊出一条带；
      - 连通性：暗像素必须与 `big` **贴着**才并入，孤立在圈里的暗斑不算。
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    big = cv2.dilate(mask, k)
    if shadow:
        kp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (grow + pad) + 1,) * 2)
        region = cv2.dilate(mask, kp)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        V, S = hsv[:, :, 2], hsv[:, :, 1]
        ring = (region > 0) & (big == 0)
        if int(ring.sum()) >= 50:
            ref = float(np.median(V[ring]))
            dark = ((V < dark_ratio * ref) & (S < max_sat) & (region > 0)).astype(np.uint8) * 255
            n, lab = cv2.connectedComponents(dark, 8)
            kt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            near = cv2.dilate(big, kt)
            keep = set(lab[near > 0].tolist()) - {0}          # 与 big 相邻的暗块才要
            if keep:
                touch = np.isin(lab, list(keep)).astype(np.uint8) * 255
                big = cv2.bitwise_or(big, touch)
    return fill_hole(bgr, big, method=fill), big


# ---------------------------------------------------------------- 主流程

def render_frame(ds, i, g, radius_m, cam_h_m, cube_h_m, erase=True, fix=True,
                 fit_both=False, diag=True, model=None, use_model=False,
                 erase_shadow=True, target="yellow", fill="diffuse",
                 x_tcp_cube=None, phase=None, held=None, carry_cb=None,
                 up_carry_cb=None):
    fr = ds[i]
    up = cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255, 0, 255)
                      .astype(np.uint8), cv2.COLOR_RGB2BGR)
    fg = cv2.cvtColor(np.clip(fr["observation.images.front"].permute(1, 2, 0).numpy() * 255, 0, 255)
                      .astype(np.uint8), cv2.COLOR_RGB2BGR)
    st = fr["observation.state"].numpy().astype(float)
    norm = {m: float(v) for m, v in zip(A.MOTORS, st)}

    T_ee = g["urdf"].fk(A.norm_to_q(norm, g["urdf"]), CAM_LINK)
    T_base_cam = T_ee @ g["X"]
    cam_xyz = T_base_cam[:3, 3]
    K, dist = g["K"], g["dist"]
    tgt_up, hold_up, tgt_name, hold_name = cube_mask_fns(g, "up", target)
    tgt_wr, hold_wr, _, _ = cube_mask_fns(g, "wrist", target)

    # 目标方块（=要换掉的那个）。up 图里它常常被暖光洗成米白、面积偏小，
    # 所以面积下限比蓝块那套（150）放松。
    z_c = cube_h_m / 2.0                                             # 方块几何中心的高度
    # 夹持区间由调用方按「首个 grasping → 首个 placing」给出（见 episode_ctx）；
    # 不给就退回按相位名判断。
    if held is None:
        held = phase in HELD_PHASES
    held = bool(held) and x_tcp_cube is not None
    ct, ct_h, hfit, hskip = None, None, None, None
    if held:
        # 夹持/搬运：3D 球心仍按 FK 算 —— 但**只用于报告/交叉检验**，画图用腕图里的
        # 直接观测（carry_cube）。理由见 carry_cube 上面那段：手眼链在这个深度差 145px。
        T_tcp = g["urdf"].fk(A.norm_to_q(norm, g["urdf"]), TCP_LINK)
        P = np.asarray((T_tcp @ np.append(x_tcp_cube, 1.0))[:3], float)   # 球心=方块几何中心
        P_fit = P
        u_t_pred = predict_px(K, dist, T_base_cam, P)
    else:
        ct = contact_point(up, g, cam_h_m, cube_h_m, tgt_up, min_area=80.0)
        if ct is None:
            return None
        ct_h = contact_point(up, g, cam_h_m, cube_h_m, hold_up, min_area=80.0)
        P = np.array([ct["base_xy"][0], ct["base_xy"][1], radius_m])      # 球心
        P_fit = np.array([ct["base_xy"][0], ct["base_xy"][1], z_c])
        u_t_pred = predict_px(K, dist, T_base_cam, P_fit)

    # --- 每帧估相机位置修正：拿**观测到的**目标方块当锚 ---
    # 拟合用的是方块的**几何中心**（z=z_c），不是顶面 —— 腕图里量到的是整个
    # 可见剪影的形心，对斜视角来说更接近几何中心的投影，不是顶面中心的投影。
    T_cam_fix = T_base_cam.copy()
    delta = np.zeros(3)
    # 夹持期**不做 δ 修正**：δ 是「相机在 base 系里挪一点」的补丁，只在桌面链
    # （up 给 3D、腕图给像素）里成立。夹持期画图用的是腕图自身的观测，没有东西要补。
    got_t = None if held else blob_near(tgt_wr(fg), u_t_pred, area_range=(250, 6000),
                                        max_dist=200)
    if ct_h is None:
        hskip = (f"夹持期：up 图里看不到{hold_name}块/不参与检验" if held
                 else f"up 图里没检出{hold_name}块（被臂挡住/阈值筛掉了）")
    else:
        P_h0 = np.array([ct_h["base_xy"][0], ct_h["base_xy"][1], z_c])
        got_h0 = blob_near(hold_wr(fg), predict_px(K, dist, T_base_cam, P_h0),
                           area_range=(80, 5000), max_dist=200)
        hfit = (P_h0, got_h0)
        if got_h0 is None:
            hskip = f"腕图里没检出{hold_name}块（阈值/面积/距离筛掉了）"
    d_model = None if model is None else delta_from_model(model, norm)
    applied, src = False, "none"
    if use_model and d_model is not None:
        delta, applied, src = d_model, True, "model"
    elif fix and got_t is not None:
        tg = [(P_fit, got_t[1])]
        if fit_both and hfit and hfit[1] is not None:
            tg.append((hfit[0], hfit[1]))
        delta, _ = solve_cam_delta(K, dist, T_base_cam, tg)
        applied, src = True, "fit"
    if applied:
        T_cam_fix[:3, 3] = T_base_cam[:3, 3] + delta

    # --- up 视图 ---（走单应，不碰腕链，所以**不受这个修正影响**）
    r_up = 0.0
    up_resid = None
    up_ball = None
    if held:
        # 夹持期 up 视图：**擦掉方块 footprint，并在同一个形心上画球**。
        #
        # 擦除：搬运期自己的门限 + 邻近追踪找楔形，再用 `up_erase_mask` 收到整块
        # footprint（先前用台面期标定阈值，163 帧只抓到 3452px，楔形原样留在图上 ——
        # 这就是用户看到的「up 相机好像没换」）。
        # 画球：位置用**观测**的 footprint 形心（`up_carry_path` 平滑过的逐帧路径；
        # 剪影形心 ≈ 方块几何中心的投影，与真值差 ~3px），不用 FK —— 夹持深度上
        # FK+手眼的绝对误差 150px（见 v4 段的实测）。大小用名义高度模型
        # r = R·√detJ·D/(D−h)：h 取 FK 的 TCP z + 18mm 系统差，D 取 UP_CAM_H_MM。
        up_out = up.copy()
        uw = up_carry_cb or {}
        if erase and uw.get("mask") is not None:
            cm = up_erase_mask(up, uw["mask"], uw.get("_allc"))
            # 残留：**画球之前**用同一条门限回扫一遍（口径与 v4 一致，可逐帧对照）
            pre = erase_cube(up, cm, grow=0, shadow=False, fill=fill)[0]
            again = carry_wedge(pre, prev_px=uw["_allc"])
            up_resid = 0 if again is None else int(again["area"])
            c_up = uw.get("ball_px")
            if c_up is None:                       # 没给路径（单帧调用）→ 用本帧形心
                ys, xs = np.nonzero(cm)
                c_up = (float(xs.mean()), float(ys.mean())) if len(xs) else None
            if c_up is not None:
                q = g["Hi"] @ np.array([c_up[0], c_up[1], 1.0])
                spm, _ = up_scale_px_per_m(g, q[:2] / q[2])
                h_mm = max(0.0, float(P[2]) * 1000.0)
                kk = UP_CAM_H_MM / max(UP_CAM_H_MM - h_mm, 0.25 * UP_CAM_H_MM)
                r_up = float(radius_m * spm * kk)
                # 擦除区 = footprint ∪ 球自己的 footprint：补洞整个躲在球底下，球外
                # 不会露一圈补出来的桌面（腕视图同一套写法，那边踩过「领圈」的坑）。
                ci = (int(round(c_up[0])), int(round(c_up[1])))
                on_mask = bool(cm[ci[1], ci[0]])       # 球心是否落在观测 footprint 里
                # 手指横穿方块时**检测掩码会裂成两块**（实测 f336/348/354/360 两块
                # 都是方块色：H 16~29 / S 83~149，中间那条白的是手指），形心于是落在
                # 手指上。这两块哪块都不是整块方块，位置误差上限 ~1 个方块宽（40px）。
                # 记下来、写在图上：宁可标注不确定，也不要假装精确。
                nsub, _, sst, _ = cv2.connectedComponentsWithStats(
                    (uw["mask"] > 0).astype(np.uint8), 8)
                ncomp = int(sum(1 for j in range(1, nsub) if sst[j, 4] >= 40))
                cv2.circle(cm, ci, int(r_up) + 1, 255, -1)
                up_out = erase_cube(up, cm, grow=0, shadow=False, fill=fill)[0]
                draw_ball(up_out, c_up[0], c_up[1], r_up)      # 悬空搬运，不画落影
                up_ball = dict(px=[round(c_up[0], 1), round(c_up[1], 1)],
                               r_px=round(r_up, 1), h_mm=round(h_mm, 1), k=round(kk, 3),
                               src=("path" if uw.get("ball_px") is not None else "frame"),
                               obs=bool(uw.get("ball_obs", False)), on_mask=on_mask,
                               ncomp=ncomp)
            else:
                up_out = pre
    else:
        up_out = erase_cube(up, ct["mask"], shadow=erase_shadow, fill=fill)[0] if erase else up.copy()
        spm, _ = up_scale_px_per_m(g, ct["xy_board"])
        r_up = radius_m * spm
        c_up = (g["H"] @ np.array([ct["xy_board"][0], ct["xy_board"][1], 1.0]))
        c_up = c_up[:2] / c_up[2]
        draw_shadow(up_out, c_up[0], c_up[1], r_up)
        draw_ball(up_out, c_up[0], c_up[1], r_up)

    # --- front（腕）视图 ---
    fr_out = fg.copy()
    (c_fr, Pc) = project(K, dist, T_cam_fix, P)
    d = float(np.linalg.norm(Pc))
    fx = K[0, 0]
    r_fr = fx * radius_m / np.sqrt(max(d * d - radius_m * radius_m, 1e-9))
    carry = None
    if held:
        # 夹持期：球画在**腕图里直接观测到的**方块形心上，大小用**方块自己当尺子**。
        #
        # 为什么不拿 FK 算半径：r = fx·R/√(d²−R²) 要吃深度 d，而夹持期 d 是 FK+手眼
        # 给的（实测偏 ~30mm：按投影反推真值 ~82mm，FK 说 110mm）—— 半径会跟着一起错。
        #
        # 用方块当尺子就与深度无关：方块边长 s（mm）、球直径 2R（mm）在同一深度上，
        # 投到像素就是同一个比例，
        #       r_px = s_px · R / cube_h_m
        # 深度在分子分母里约掉了。
        # s_px 取 **√面积**：方块（可见的）就是个正方形，面积开方与它转多少度无关。
        # 另外两种量法都不如它：bbox 会随转角涨到 s·(|cosθ|+|sinθ|)（f340 实测 bbox
        # 110px、边长只有 99px）；minAreaRect 又被夹爪啃掉的那块缺口和边缘毛刺撑大
        # （f340 给 113×109）。同一个方块在 up 视图里量过：面积开方 21.5mm ≈ 标称 22mm，
        # 说明开方这把尺在这块方块上是准的。误差量级 ±10%，比原来那版好得多。
        #
        # 上一版把半径兜到**剪影外接圆**（r≈66px）——那是我为了让擦除区整个躲在球底下
        # 才放大的，视觉上球比方块还大 20%，用户看出来了。现在按尺子来（r≈54px），
        # 方块的角会露到球外 ~16px，那部分照擦（见下）。
        # 注：球物理上是 24mm、方块实测 21.5~22mm，所以球比方块**真的**宽 9%；
        # 想让它和方块一模一样大就 `--radius-mm 11`。
        cb = carry_cb or {}
        if cb.get("c") is not None:
            c_fr = np.asarray(cb["c"], float)
            s_px = cb.get("s")
            if s_px is None and cb.get("area"):
                s_px = float(np.sqrt(cb["area"]))
            if s_px:
                r_fr = float(np.clip(float(s_px) * radius_m / cube_h_m, 20.0, 110.0))
                r_src = "cube-ruler" + ("/stale" if cb.get("stale") else "")
            else:
                r_fr, r_src = 45.0, "fallback"
            r_sil = 0.0                      # 只留给报告看，不再拿它当半径
            if cb.get("mask") is not None:
                ys, xs = np.nonzero(cb["mask"])
                if len(xs):
                    r_sil = float(np.hypot(xs - c_fr[0], ys - c_fr[1]).max())
            if erase:
                # 擦除区 = **检测掩码 ∪ 球自己的 footprint**。球比方块小，方块四角会
                # 露在球外 ~16px，那部分必须擦掉（留着就是四块黄角，正是要消的东西），
                # 擦完由 inpaint 补成桌面。掩码外扩 3px 吃掉抗锯齿边。
                # 先前那版在这外面还放一档阈值取「完整剪影」，取到的 1/3 是桌面
                # （领圈里 78% 的像素过 S≥100），擦除区比球宽出 25px，补洞环就露在
                # 球外一圈 —— 那圈已经去掉，现在多出球的只有四个角的小透镜。
                cm = np.zeros(fg.shape[:2], np.uint8)
                if cb.get("mask") is not None:
                    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    cm = cv2.bitwise_or(cm, cv2.dilate(cb["mask"], k3))
                cv2.circle(cm, (int(round(c_fr[0])), int(round(c_fr[1]))),
                           int(r_fr) + 1, 255, -1)
                # 阴影照旧一起抹：夹取那几帧方块还压在桌面上，留着阴影就是球旁边
                # 一块孤零零的暗斑（f260/f300/f340 实测）。erase_cube 的暗像素闸门
                # 只吃低饱和的暗块（max_sat=70），红盒子 S186+ 进不来。
                fr_out = erase_cube(fr_out, cm, grow=0, shadow=erase_shadow,
                                    fill=fill)[0]
            draw_ball(fr_out, c_fr[0], c_fr[1], r_fr)      # 悬空，不画落影
            u_fk = np.asarray(u_t_pred, float).reshape(2)
            carry = dict(px=c_fr.round(1).tolist(),
                         area=None if not cb.get("area") else int(cb["area"]),
                         s_px=None if not s_px else round(float(s_px), 1),
                         src=cb.get("src"), stale=bool(cb.get("stale")),
                         r_px=round(r_fr, 1), r_src=r_src,
                         r_sil_px=round(r_sil, 1),
                         fk_px=u_fk.round(1).tolist(),
                         fk_err_px=round(float(np.hypot(*(u_fk - c_fr))), 1))
    elif Pc[2] > 0 and np.isfinite(r_fr):
        # 方块在腕图里**也得抹掉**（先前只抹了 up 视图）：球落在方块上，方块的角
        # 和倒角还支在球外面，看着就是「叠上去」而不是「替换」。只抹离球心最近的
        # 那一块 —— 面积筛挡住左上角那本蓝皮书。
        cm = keep_nearest_comp(tgt_wr(fg), c_fr, area_range=(250, 6000), max_dist=200) if erase else None
        if cm is not None:
            fr_out = erase_cube(fr_out, cm, grow=9, shadow=erase_shadow, fill=fill)[0]
        draw_shadow(fr_out, c_fr[0], c_fr[1], r_fr, scale=(1.2, 0.45), offset=(0.3, 0.5))
        draw_ball(fr_out, c_fr[0], c_fr[1], r_fr)

    # --- 留出检验：**另一块不参与拟合**，看修正有没有把它也带对 ---
    hchk = None
    if hfit and hfit[1] is not None:
        P_h, (_, u_h_obs, area_h) = hfit[0], hfit[1]
        e0 = float(np.hypot(*(u_h_obs - predict_px(K, dist, T_base_cam, P_h))))
        e1 = float(np.hypot(*(u_h_obs - predict_px(K, dist, T_cam_fix, P_h))))
        hchk = dict(obs=u_h_obs.round(1).tolist(), area=int(area_h), name=hold_name,
                    err_before=round(e0, 1), err_after=round(e1, 1))

    if diag:
        w = (255, 255, 255)
        if got_t is not None:
            draw_offset(fr_out, u_t_pred, got_t[1], (0, 215, 255))   # 琥珀：模型→观测
            draw_marker(fr_out, got_t[1], w, 3)
        if hchk is not None:
            p_h_pred = predict_px(K, dist, T_cam_fix, hfit[0])
            draw_offset(fr_out, p_h_pred, np.array(hchk["obs"]), (60, 220, 60))
            draw_marker(fr_out, np.array(hchk["obs"]), w, 3, 1)
        if carry is not None:
            # 青色圈标 FK 链的预测像素（夹持期差得远，这就是不用它的原因），
            # 白十字标直接观测到的方块形心 —— 球就画在那儿。
            draw_marker(fr_out, np.array(carry["fk_px"]), (255, 255, 0), 3, 2)
            draw_marker(fr_out, np.array(carry["px"]), w, 3, 4)
        tag = (f"carry[obs {carry['src']}{' STALE' if carry['stale'] else ''}] "
               f"r={carry['r_px']:.0f}px({carry['r_src']})  FK err {carry['fk_err_px']:.0f}px"
               if carry is not None else
               f"fix[{src}] |d|={np.linalg.norm(delta)*1000:.1f}mm")
        if hchk:
            tag += f"  hold-out {hold_name[:3]} {hchk['err_before']:.0f}->{hchk['err_after']:.0f}px"
        cv2.putText(fr_out, tag, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(fr_out, tag, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if held:
            # up 视图这一段的处理是**约定**，得写在图上：球心是观测的 footprint 形心，
            # 半径吃了一个名义相机高度 D —— 这两件事看图的人猜不到。
            # 注：cv2 的 Hershey 字体没有汉字（渲染成 ???），所以标注一律用 ASCII。
            # 这几行放**底部**：up 图里方块就压在左上角 (up x 30~76, y 64~110)，
            # 放顶上会正好把要看的那块盖住（第一版就是这样，出图后才看出来）。
            if up_ball is None:
                ulines = (f"held: up view ERASE-ONLY (cube resid {up_resid}px)",
                          "cube footprint not observed this frame",
                          "-> no ball drawn (better under- than mis-augment)")
            else:
                ulines = (
                    f"held: up ball at observed centroid "
                    f"({up_ball['px'][0]:.0f},{up_ball['px'][1]:.0f}) r={up_ball['r_px']:.0f}px"
                    f"; resid {up_resid}px",
                    f"r = R * scale * D/(D-h): h={up_ball['h_mm']:.0f}mm "
                    f"k={up_ball['k']:.2f} D={UP_CAM_H_MM:.0f}mm nominal",
                    "D NOT identifiable from this dataset -> radius band +-25%",
                    "cube centroid observed -> position needs no height model")
                if up_ball.get("ncomp", 1) >= 2:
                    ulines += (f"finger splits the cube into {up_ball['ncomp']} "
                               f"-> centre +-1 cube width",)
            # 每行 <=62 字符（640px 宽、0.45 字号下再长就被右边缘切掉，实测过），
            # 整块**底对齐**：行数随遮挡标注变化时不会顶到画面外。
            y0 = up_out.shape[0] - 10 - 20 * (len(ulines) - 1)
            for j, ut in enumerate(ulines):
                cv2.putText(up_out, ut, (8, y0 + 20 * j),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4)
                cv2.putText(up_out, ut, (8, y0 + 20 * j),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    info = dict(frame=i, held=held, phase=phase, up_resid_px=up_resid, carry=carry,
                up_ball=up_ball, up_cam_h_mm=UP_CAM_H_MM,
                up_wedge=None if not (held and up_carry_cb) else dict(
                    px=None if up_carry_cb.get("_allc") is None
                    else up_carry_cb["_allc"].round(1).tolist(),
                    area=up_carry_cb.get("area"), bbox=up_carry_cb.get("bbox"),
                    stale=bool(up_carry_cb.get("stale"))),
                cube_px=None if ct is None else ct["px"].round(1).tolist(),
                pull_mm=None if ct is None else round(ct["pull_mm"], 1),
                contact_board_mm=None if ct is None else (ct["xy_board"] * 1000).round(1).tolist(),
                ball_base_mm=(P * 1000).round(1).tolist(),
                cam_base_mm=(cam_xyz * 1000).round(1).tolist(),
                dist_mm=round(d * 1000, 1), r_up_px=round(r_up, 1),
                r_front_px=round(float(r_fr), 1), in_front=bool(Pc[2] > 0),
                fix_mm=np.round(delta * 1000, 1).tolist(),
                fix_delta_mm=round(float(np.linalg.norm(delta)) * 1000, 1),
                fix_src=src, fix_applied=applied,
                model_mm=None if d_model is None else np.round(d_model * 1000, 1).tolist(),
                model_vs_fit_mm=(None if (d_model is None or src != "fit")
                                 else round(float(np.linalg.norm(d_model - delta)) * 1000, 1)),
                cam_fix_mm=np.round(T_cam_fix[:3, 3] * 1000, 1).tolist(),
                target=target, hold=hold_name,
                tgt_pred_px=u_t_pred.round(1).tolist(),
                tgt_obs_px=None if got_t is None else got_t[1].round(1).tolist(),
                tgt_obs_area=None if got_t is None else int(got_t[2]),
                hold_chk=hchk, hold_skip=hskip)
    return up_out, fr_out, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--root", default=None)
    # 默认挑的是 ep0/ep4/ep7 各 +120~+180 帧：那时臂还没伸到桌上，黄块完整可见。
    # 原来默认 0,150,1000 是蓝块那套 —— f1000 时黄块已经进红盒了，up 链判不出来。
    ap.add_argument("--frames", default="120,2033,3443")
    ap.add_argument("--target", choices=("yellow", "blue"), default="yellow",
                    help="换掉哪一块。**默认黄** —— newdata1 的 task 是 "
                         "\"Pick up yellow block and put it in trash can\"，被夹的是黄块，"
                         "蓝块是干扰物（ep1/ep3 逐帧看过：黄块被夹起放进红盒子，蓝块没动过）。"
                         "另一块自动成为留出检验。")
    ap.add_argument("--radius-mm", type=float, default=12.0)
    ap.add_argument("--cube-h-mm", type=float, default=22.0, help="方块高（视差补偿用）")
    ap.add_argument("--cam-h-mm", type=float, default=500.0, help="up 相机离桌面高度（估的）")
    ap.add_argument("--no-erase", action="store_true", help="不抹掉原方块")
    ap.add_argument("--no-erase-shadow", action="store_true",
                    help="抹方块时**不连阴影一起抹**（阴影是暗色不是蓝色，默认连它一起抹；"
                         "关掉会留下方块压桌面的暗弧）")
    ap.add_argument("--no-fix", action="store_true",
                    help="不做每帧相机位置修正（= v1 的行为，腕图里球会偏出方块）")
    ap.add_argument("--fit-both", action="store_true",
                    help="两块一起拟合（另一块就**不再是**留出检验了）")
    ap.add_argument("--delta-model", default=str(HERE / "calib" / "wrist_delta_model.json"),
                    help="δ 的线性模型 json（wrist_delta_probe.py 生成）")
    ap.add_argument("--use-model", action="store_true",
                    help="δ 由 state 直接算，不靠腕图里看得见方块（覆盖被夹爪挡住的帧）")
    ap.add_argument("--fill", choices=("diffuse", "telea"), default="diffuse",
                    help="补洞算法。diffuse=调和插值（默认，桌面这种光滑渐变更对路）；"
                         "telea=cv2.INPAINT_TELEA（会把边界暗影拖进洞里）")
    ap.add_argument("--no-held", action="store_true",
                    help="关掉夹持期分支：夹持/搬运帧退回旧行为（up 图里没有方块 → 整帧跳过）。"
                         "对照用。")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = a.root or str(REPO / "dataset" / a.dataset)
    ds = LeRobotDataset(a.dataset, root=root)
    g = load_geom()
    mdl_path = Path(a.delta_model)
    model = load_delta_model(mdl_path) if mdl_path.exists() else None
    if model is None:
        print(f"[模型] 没找到 {mdl_path} —— δ 只能用逐帧观测拟合"
              f"（跑 wrist_delta_probe.py 生成）")
    radius_m, cam_h_m, cube_h_m = a.radius_mm / 1000, a.cam_h_mm / 1000, a.cube_h_mm / 1000

    print(f"[几何] X_ee_cam 平移 {np.round(g['X'][:3,3]*1000,1)} mm | 板原点 "
          f"{np.round(g['T_base_board'][:3,3]*1000,1)} mm | 球半径 {a.radius_mm} mm")
    print(f"[视差] 方块高 {a.cube_h_mm} mm, up 相机高 {a.cam_h_mm} mm")

    # --- 相位 / 夹持区间 / X_tcp_cube（每个 episode 只算一次）---
    rsp_mod = load_prior()
    tgt_up_fn = cube_mask_fns(g, "up", a.target)[0]
    # 相位只看 observation.state，走 hf_dataset 的表格列（不碰视频解码，快 100 倍）
    states = np.stack([np.asarray(x, float) for x in ds.hf_dataset["observation.state"]])
    ep_cache: dict[int, dict] = {}

    def episode_ctx(i):
        """含帧 i 的那个 episode 的相位表 + 夹持区间 + X_tcp_cube（缓存）。"""
        i0, i1 = episode_bounds(ds, i)
        if i0 in ep_cache:
            return ep_cache[i0]
        prior = rsp_mod.RobotStatePrior()
        ph = [prior.update(states[k])["phase"] for k in range(i0, i1)]
        ig = next((i0 + j for j, p in enumerate(ph) if p == "grasping"), None)
        # 夹持区间取到**第一次 placing** 为止：搬完之后臂可能又合上夹爪（相位会回到
        # moving），但那时方块已经在红盒子里、不在爪上了，接着按 FK 画球是错的。
        ir = next((i0 + j for j, p in enumerate(ph)
                   if p == "placing" and ig is not None and i0 + j > ig), i1)
        X, note = None, "这一段没有夹取事件（相位里没出现过 grasping）"
        trk, up_trk = {}, {}
        if ig is not None:
            if a.no_held:
                X, note = None, "--no-held：夹持期分支关掉了"
            else:
                X, note = estimate_x_tcp_cube(ds, ig, g, cam_h_m, cube_h_m, tgt_up_fn)
                trk = carry_track(ds, ig, ir)
                if not trk:
                    note += "；腕图里也没跟到方块 → 夹持期不出图"
                else:
                    st = [k for k, v in trk.items() if v["stale"]]
                    seen = [v for v in trk.values() if not v["stale"]]
                    note += (f"；腕图跟到 {len(seen)}/{len(trk)} 帧"
                             f"（跟丢沿用上次位置 {len(st)} 帧）")
                seed, seed_f = last_up_px(ds, ig, tgt_up_fn)
                if seed is not None:
                    up_trk = carry_wedge_track(ds, ig, ir, seed)
                    hit = [v for v in up_trk.values() if not v["stale"] and v["area"]]
                    note += (f"；up 图楔形（种子 f{seed_f}）跟到 {len(hit)}/{len(up_trk)} 帧，"
                             f"平均 {int(np.mean([v['area'] for v in hit])) if hit else 0}px")
                    # 球心路径：逐帧 footprint 形心 → 中值 + SG 平滑（见 up_carry_path）
                    path = up_carry_path(ds, ig, ir, up_trk)
                    for kk_, v in path.items():
                        if kk_ in up_trk:
                            up_trk[kk_]["ball_px"] = v["px"]
                            up_trk[kk_]["ball_obs"] = v["obs"]
                    n_obs = sum(1 for v in path.values() if v["obs"])
                    note += (f"；up 球心路径 观测 {n_obs}/{len(path)} 帧"
                             + ("（其余由相邻帧插值）" if n_obs < len(path) else ""))
                else:
                    note += "；up 图夹取前没找到方块 → 夹持期 up 只做原样"
        ctx = dict(ph=ph, i0=i0, i1=i1, ig=ig, ir=ir, X=X, note=note, track=trk,
                   up_track=up_trk)
        ep_cache[i0] = ctx
        if ig is not None:
            xs = "None" if X is None else np.round(X * 1000, 1).tolist()
            print(f"[ep{i0}-{i1}] 夹取 f{ig} → 释放 f{ir}（夹持 {ir-ig} 帧）| "
                  f"X_tcp_cube(mm)={xs} | {note}")
        else:
            print(f"[ep{i0}-{i1}] {note}")
        return ctx

    rows, infos = [], []
    for s in a.frames.split(","):
        i = int(s)
        ctx = episode_ctx(i)
        ph = ctx["ph"][i - ctx["i0"]]
        in_span = ctx["ig"] is not None and ctx["ig"] <= i < ctx["ir"]
        r = render_frame(ds, i, g, radius_m, cam_h_m, cube_h_m, erase=not a.no_erase,
                         fix=not a.no_fix, fit_both=a.fit_both,
                         model=model, use_model=a.use_model,
                         erase_shadow=not a.no_erase_shadow, target=a.target, fill=a.fill,
                         x_tcp_cube=ctx["X"], phase=ph, held=in_span,
                         carry_cb=ctx["track"].get(i),
                         up_carry_cb=ctx["up_track"].get(i))
        if r is None:
            why = (f"（夹持期：{ctx['note']}）" if in_span else "")
            print(f"  frame {i}: 没检出方块，跳过 {why}")
            continue
        up_out, fr_out, info = r
        infos.append(info)
        if info["held"]:
            uw = info.get("up_wedge")
            print(f"  frame {i:5d}  [夹持期 phase={info['phase']}] up 图："
                  + ("楔形未跟到 → 本帧不擦" if not uw or uw["area"] is None else
                     f"楔形 {uw['area']}px bbox {uw['bbox'][2]}x{uw['bbox'][3]} @ {uw['px']}"
                     + ("(STALE)" if uw["stale"] else ""))
                  + f"  → 擦完残留 {info['up_resid_px']}px")
            ub = info.get("up_ball")
            if ub is None:
                print("        up 图没画球（footprint 没观测到）")
            else:
                print(f"        up 画球：球心 {ub['px']}（观测形心"
                      f"{'平滑路径' if ub['src'] == 'path' else '本帧'}）"
                      f" r={ub['r_px']}px（球径 {2*ub['r_px']:.0f}px）"
                      f"  = R × 局部尺度 × k，h={ub['h_mm']:.0f}mm k={ub['k']:.3f}"
                      f"  ← D={info['up_cam_h_mm']:.0f}mm 是名义值（见文件头）"
                      f"；球心落在 footprint 内={ub['on_mask']}"
                      + (f"；⚠ 手指把方块切成 {ub['ncomp']} 块，形心落在手指上"
                         f"（位置误差上限 ~1 个方块宽）" if ub.get("ncomp", 1) >= 2 else ""))
            cf = info["carry"]
            if cf is None:
                print("        ⚠ 腕图里没跟到方块 → 这一帧**不画球**（没有位姿就没有\n                       对的球，宁可不画）")
            else:
                print(f"        腕图直接观测[{cf['src']}"
                      f"{' STALE 沿用上次位置' if cf['stale'] else ''}] 形心 {cf['px']}"
                      f" 面积 {cf['area']}（边长 {cf['s_px']}px）  →  球心画在这里，"
                      f"r={cf['r_px']}px（球径 {2*cf['r_px']:.0f}px = 边长×"
                      f"{2*cf['r_px']/cf['s_px']:.3f}）"
                      f"（{cf['r_src']}；剪影角伸到 {cf['r_sil_px']}px，露在球外照擦）")
                print(f"        **交叉检验**：FK 链（手眼）给 {cf['fk_px']}，"
                      f"与直接观测差 {cf['fk_err_px']:.0f}px  ← 这就是夹持期不用 FK 的原因")
        else:
            print(f"  frame {i:5d}  方块像素 {info['cube_px']}  视差回拉 {info['pull_mm']:4.1f}mm"
                  f"  球心(base) {info['ball_base_mm']}  腕相机距球 {info['dist_mm']:6.1f}mm"
                  f"  → up {info['r_up_px']:.1f}px / front {info['r_front_px']:.1f}px"
                  + ("" if info["in_front"] else "  ⚠ 球在腕相机背后"))
        if not info["held"]:
            if info["tgt_obs_px"] is not None:
                e0 = float(np.hypot(*(np.array(info["tgt_obs_px"])
                                      - np.array(info["tgt_pred_px"]))))
                print(f"        {info['target']}块(要换的) 拟合前残差 {e0:5.1f}px"
                      f" (观测面积 {info['tgt_obs_area']}px²)")
            else:
                print(f"        {info['target']}块(要换的) 腕图里没找到（面积/距离筛掉了）")
        if info["held"]:
            pass                        # 夹持期不做 δ 修正（见 render_frame 里的注释）
        elif info["fix_applied"]:
            print(f"        ⇒ 相机修正 [{info['fix_src']}] {info['fix_mm']} mm"
                  f"  (|δ|={info['fix_delta_mm']:.1f})"
                  + ("" if info["model_vs_fit_mm"] is None else
                     f"   | 模型(state)给 {info['model_mm']}，与逐帧拟合差 {info['model_vs_fit_mm']:.1f}mm"))
        else:
            print("        ⇒ 本帧不做修正（没开 --use-model，且方块在腕图里没检出）")
        hc = info.get("hold_chk")
        if info["held"]:
            pass                       # 夹持期腕相机对着夹爪，另一块根本不在视野里
        elif hc:
            print(f"        留出检验 {hc['name']}块(不参与拟合): {hc['err_before']:5.1f}px"
                  f" → {hc['err_after']:5.1f}px   (观测面积 {hc['area']}px²)")
        elif info.get("hold_skip"):
            print(f"        留出检验 {info['hold']}块：没做成 —— {info['hold_skip']}")
        lab = f'{a.dataset} f{i}  [{info["target"]}->ball]'
        for im, t in ((up_out, f"up  {lab}"), (fr_out, f"front(wrist)  {lab}")):
            cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.hstack([up_out, fr_out]))
    if not rows:
        return 1
    out = np.vstack(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    name = "ball_swap_v1.png" if a.no_fix else (
        ("ball_swap_v2_fitboth" if a.fit_both else "ball_swap_v2_fixed")
        + ("" if a.target == "yellow" else "_blue")
        + ("_noshadow" if a.no_erase_shadow else "") + ".png")
    path = Path(a.out) if a.out else FIG / name
    cv2.imwrite(str(path), out)
    print(f"[出图] {path}  ({out.shape[1]}x{out.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
