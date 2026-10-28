# SO101 URDF —— 本地核对结论

2026-09-29。对应 `fk_check.py`（算）、`render_urdf.py`（画）。所有数字都是本机算出来的，
不是从文档抄的。

## 目录内容

| 文件 | 来源 |
|---|---|
| `so101_new_calib_camera.urdf` | 官方，带腕部相机 |
| `so101_new_calib.urdf` | 官方，无相机 |
| `so101_old_calib.urdf` | 官方，旧零位约定，无相机（官方**没有** old_calib 的带相机版，404） |
| `assets/*.stl` | 官方 18 个（URDF 实际引用 15 个；`Handle_/Trigger_/Wrist_Roll_SO101` 是备用件） |
| `README.md` `joints_properties.xml` | 官方原文 |

`render_urdf.py` 只用 numpy + PIL —— 本机没有 trimesh / pybullet / yourdfpy，别去装，
二进制的 STL 格式 30 行就解析完了。

## 一、`new_calib` vs `old_calib`：两套只差**一个绕 Z 的方位角**，不是姿态

⚠️ **2026-10-07 更正**：本节原先说「判据是姿态，不是行程」——**错**。真相是
**两套约定对同一个 LeRobot 归一化预言的物理姿态完全相同**，姿态也判不了、也不必判。
证明见下面「判决实验（已作废）」。下面这张表比的是**同一个数值 q** 下两套 URDF 的
姿态（new@0 vs old@0），那是**两个不同的物理姿态**，不是「同一台臂的两种解释」：

| | `new_calib` @ q=0 | `old_calib` @ q=0 |
|---|---|---|
| 大臂与地面夹角 | **+70.7°** | **+10.3°** |
| 小臂与地面夹角 | +2.2° | +2.2° |
| 各连杆高度 z (m) | 0.062 → 0.229 → 0.234 | 0.095 → 0.121 → 0.126 |
| shoulder_lift 限位 | [−100°, +100°] **对称** | [−190°, **+10°**] 零点贴在端点 |
| elbow_flex 限位 | [−96.8°, +96.8°] **对称** | [−10°, **+180°**] 零点贴在端点 |

`old_calib` 在 q=0 时**整条臂平躺**，与官方 README 的原话（old_calib 零位 = 水平完全伸展）
**逐字吻合** —— 这是对 `fk_check.py` 的独立验证：README 的定义是算出来的，不是假设的。

限位的**对称性**是一条更省事的判据：`new_calib` 的中间四个关节限位关于 0 对称，
`old_calib` 不对称。看 URDF 里的 `limit lower/upper` 就能一眼分辨，不用跑 FK。

**看 `figures/urdf_compare.png`**：两幅同一视角、带 z=0 地面网格（Z 轴两套共用，
所以「相对地面平不平」是唯一跨约定可直接比的量）。

### 判决实验（2026-10-07 做成并作废：**这个二分不成立**）

原提议：把真机六个关节都摆到 LeRobot 归一化 **0**，看它是「大臂陡升 + 小臂水平」
（= new_calib）还是「整条臂平躺」（= old_calib）。

**做不到。** 上面那张表比的是**同一个数值 q**（new@0 vs old@0），但实验里扫的是
**归一化 norm**，而每套约定把同一个 norm 映到**不同的 q**：
`q = mid + (norm/100)·half`，`mid` 在 lift 上差 90°、elbow 上差 85°。两个 90° 恰好抵消：

| | norm → q（shoulder_lift） | 归一化 0 处的大臂仰角 |
|---|---|---|
| new | `0 + (n/100)·100` | **76.0°** |
| old | `−90 + (n/100)·100` | **76.0°** |

全量程实测（`workplace/calibration/step1_verdict.py`，2026-10-07，本机新算）：

| norm | −100 | −75 | −50 | −25 | 0 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| new 大臂仰角° | 4.0 | 29.0 | 54.0 | 79.0 | 76.0 | 51.0 | 26.0 | 1.0 | −24.0 |
| old 大臂仰角° | 4.0 | 29.0 | 54.0 | 79.0 | 76.0 | 51.0 | 26.0 | 1.0 | −24.0 |
| **差** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

⇒ 在**任意**归一化下两套约定预言的物理姿态完全相同。这个实验只会永远报「new_calib」，
它不是判据。**别再为它上真机。**

根因（本机验算）：两套 URDF 是同一个机器人，**底座坐标系差 +90° 绕 Z**（new 的大臂沿
−X 伸、old 沿 +Y），而 `shoulder_lift` 的零点也正好差 **90°**；两个 90° 精确抵消。
验算：把 old 的「肩→肘」向量绕 Z 转 +90°，与 new 的夹角 **0.00°**。
`figures/verdict_hypotheses.png` 两幅形状一样、只差一个方位角 —— 那张图本身就说明了这点。

### 唯一有判别力的量：行程（弱，偏向 `new_calib`）

六joint 里**五个**两套约定的标称行程逐字相同（pan/lift/wrist_flex/wrist_roll/gripper），
只有 `elbow_flex` 不同：new 193.66° vs old 190°。若标定时扫到了物理硬限位，实测行程
应等于正确那套的标称值：

| 关节 | 实测° | new° | Δ | old° | Δ |
|---|---|---|---|---|---|
| elbow_flex | 193.9 | 193.66 | **0.23** | 190.00 | 3.89 |
| 其余五个 | — | 与 old 相同 | — | — | — |

（STS3215 = 4096 counts/转 = 11.378 counts/°。）**证据只有这一条，且差 3.7° —— 弱。**

### 结论：不用判、也判不了 —— 直接用 `new_calib`，但**不要混搭**

- 两套约定对「同一个 norm 的物理姿态」给出相同答案 ⇒ FK **不会**因为选错而错。
  先前「零点错 → q_real 错 → T_real 错 → 投影全错」的担心**不成立**，撤回。
- 差的那 90° 是**底座坐标系绕 Z 的方位角**（朝哪个方向伸），不是姿态形状。
  它会被「底座→相机」外参标定整个吸收；只有把 URDF 对到**外部参照系**
  （LeIsaac 场景、别人的数据集）时才需要显式对齐这一个常数。
- `new_calib` 是官方现行约定，也是**唯一有带相机版 URDF** 的
  （`so101_new_calib_camera.urdf`）⇒ 投影工作直接用 new。
- **红线：不要混搭**。new 的限位必须配 new 的 URDF（old 配 old）。混搭才真的差 90°，
  差的是方位角，表现为整条臂在地上转了四分之一圈。

## 二、归一化 ↔ URDF 关节角

读自 `src/lerobot/motors/motors_bus.py:770`。`src/lerobot/robots/so101_follower/so101_follower.py:48-57`：

- 五个本体关节 = `RangeM100_100`：`norm = (tick − range_min)/(range_max − range_min)·200 − 100`
  → **norm = 0 ⟺ 标定 range 的中点**
- **夹爪 = `Range0_100`**：`norm` 落在 **0…100**，`0 = range_min = 闭合端`
  → 夹爪的 0 **不是**中点，别和其它五个关节套同一个公式

**`homing_offset` 在这个差分里约掉了** —— 所以做 FK 只需要 `range_min`/`range_max`，
不需要 `homing_offset`。（我先前说 homing_offset 是 URDF↔真机的桥，那是错的，已更正。）

用户标定文件在
`~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json`。
把它和 URDF 限位对起来：

```
q_urdf = mid(URDF限位) + (norm / 100) * half(URDF限位)      # 本体五个关节
```

## 三、相机：官方只给了**一个**，而它是腕部相机

官方 URDF 只有 `wrist_camera_link`，经 `wrist_camera_mount_link` 挂在 `gripper_link` 上。
**没有 up / 底座相机的外参** —— 那个必须自己标。

本机算出的官方腕部相机（q=0，base 系）：

- 光心 `[+0.2866, +0.0731, +0.2404]`，距 base 原点 0.381 m
- 光轴（取 link 的 **+Z**）`[+0.906, −0.423, −0.021]` → **俯仰 −1.2°（几乎水平）**，方位 −25°

**「+Z 是光轴」是 URDF 相机帧的常见惯例，但这份 URDF 里没有 `<sensor>` 标注，
所以这是假设不是事实。** 用它之前先确认。

### 与 LeIsaac 模板的命名冲突（要留意）

`leisaac/tasks/template/single_arm_env_cfg.py` 里：`wrist` 挂在 `Robot/gripper/wrist_camera`
（手眼），`front` 挂在 `Robot/base/front_camera`（**底座**）。

但用户自己的 `workplace/background_generalization/CLAUDE.md` 记的实测是
**up 静止、front 随臂动** —— 即用户场景里 **front 才是手眼相机**。

**两套命名是反的。** 谈「front 相机」时先说清是哪一个，否则后面全是错的。
官方 URDF 的 `wrist_camera_link` 对应的是用户的 **front**，不是 up。

## 四、夹爪：URDF 和 LeRobot 语义不一致，差异是**可量化的**

- LeRobot：线性，`Range0_100`，0 = 闭合 / 100 = 张开
- URDF：`revolute`，限位 [−10°, +100°] → 总行程 **110°**
- 用户标定的夹爪行程：**141.6°**

110 ≠ 141.6，差 29%。所以 **URDF 的 gripper 关节角不能从 LeRobot 的夹爪值线性换算出来**
（官方 README 已声明该映射「not yet reflected in the URDF」，这里给出了差多少）。
做 TCP 的 FK 影响小；渲染夹爪的 mask 必须自己处理。

## 五、这台臂是 SO101 克隆（用权威源确认过）

淘宝臂 + LeRobot 代码。用标定反推行程与官方 URDF 对比，六个关节五个对上（见上表），
夹爪的偏差由上一条解释。结论：**运动学用官方 URDF，外形（渲染 mask 时）才需要自己的 mesh。**
两个用途保真度要求不同 —— 干扰物放置只要运动学对，mask 渲染还要外形对。

## 复现

```bash
cd workplace/so101_urdf
python fk_check.py                                  # 数字 + 归一化映射
python fk_check.py --also so101_old_calib.urdf      # 两套零位并排
python render_urdf.py --compare                     # → figures/urdf_compare.png
python render_urdf.py --pose 0 0 -30 60 0 50        # 任意姿态
```

`render_urdf.py` 踩过的两个坑（都在代码注释里）：`look_at` 的旋转矩阵必须写成**行**
（写成列的话目标点投不到画面中心，会偏到 (−75, 819)）；PIL 默认字体没有中文字形，
图上的文字只能用 ASCII，否则出豆腐块。
