# 换物体 + SaGA 增强：方法、数字、复现

**这份文件是给「下一次开新对话」用的自包含上下文。** 读完它应当能直接接手，不用翻旧记录。

覆盖两件事：

1. **换物体**（把桌上干扰物 / 被夹的目标换成重建网格）是怎么做出来的；
2. 在**换完的图**上再叠 **RoboSaGA** 背景增强，出效果图。

相机标定的公式与误差棒不在这里，见 [`../calibration/CAMERA_CALIB.md`](../calibration/CAMERA_CALIB.md)。
SaGA 本身的实现约定见 [`CLAUDE.md`](CLAUDE.md)。

---

## 1. 一句话

> 两条路共用同一台**抹掉 → 画上 → 把真前景摆回**的合成器；
> 唯一的差别是**新物体的位姿 `T` 从哪来**：
> 干扰物没动 ⇒ **量**出来；目标在动 ⇒ 从剪影**拟合**出来。

---

## 2. 三层合成器（L0 / L1 / L2）

任何一帧都走这三层，与换的是什么物体无关：

| 层 | 做什么 | 实现 |
|---|---|---|
| **L0 抹掉** | 擦掉旧物体的像素并修补背景 | `BS.erase_cube(orig, mask, grow, shadow, fill="diffuse")` |
| **L1 画上** | 重建网格按 `T` 光栅化 + 着色 | `MR.transform` → `MR.rasterize` → `PW.smooth_normals` → `MR.shade` |
| **L2 摆回** | 真前景（夹爪/手指/机械臂）盖回前面 | 见下，逐脚本略有不同 |

L2 是**构造性等式**，不是"尽量"：

- `pose6d_up_fit.py:120` —— `out[cm == 0] = up[cm == 0]`：擦除掩码外**逐位**回原帧 ⇒
  夹爪永远在新物体前面，红线因此可证。
- `pose6d_wrist_fit.py` —— `out[front] = fg[front]`，`front = front_obs | front_geo`
  （观测路 ∪ 几何路）。
- `pose6d_distractor.py` —— `composite()` 里的 `front` 是机械臂掩码；红线换成
  「**被擦掉的那块 ∪ 网格剪影**之外逐位等于原帧」。注意这边**不能**照抄
  `out[cm==0] = orig[cm==0]`：注入的舵机比蓝块大一圈，剪影会伸到擦除掩码外面，
  照抄会把舵机削掉一块。

**被擦掉的那块要从 `plate` 自己量**（`plate != orig`），不要拿 `obsm` 猜 ——
`erase_cube` 会按 `grow`/`shadow` 外扩，猜边界只会漏掉那圈羽化（踩过：虚报 1105px）。

---

## 3. 位姿 `T` 从哪来 —— 唯一的分岔

### 3.1 换干扰物（蓝块）：**量**出来

蓝块静止躺在桌上，位姿可以事先量准：

```
up 图里蓝块形心 (px)
    ──up 单应 H──▶  板系桌面点 (x, y)  ──T_base_board──▶  base 系
    ep0 上每 10 帧采一次、共 44 次，取中位
```

实测锚点 **`(239.98, 117.42, 0) mm`**，重复性：最大偏离 **1.14mm**、中位 **0.24mm**
（`pose6d_distractor.anchor()`）。

- **位置**：上面的锚点。**不逐帧反算** —— 蓝块没动，逐帧反算只是把掩码噪声当物点。
- **朝向**：**显式设计选择**。蓝块不携带"它原本朝哪"的信息，所以就是
  「静置姿态 + 绕竖直轴转 `yaw`」，一行写在 `pose_T()` 里，不藏在优化器里。
  ⚠️ `rest_pose()` 已经**把静置旋转烘进网格顶点**了，`pose_T` 再乘一次 `R_rest`
  就是转两遍 ⇒ 底面沉到桌面以下 **6.3mm**（踩过，现在 `z_min` = **+0.1mm**）。
- **两个相机渲染同一个 base 系 `T`** —— 物体是死的、位置是量的，没有理由让两个视角各说各话。

相机：up 用造出来的度量相机（主点 = 观测形心、光轴竖直、焦距由局部 px/m 定），
腕用真标定的 `K + dist + FK⊗手眼`。

### 3.2 换目标（黄块）：**拟合**出来

黄块被夹爪夹着、一直在动，没有桌面锚点可量。而且这条臂的手眼链本身差
**~30mm / 150px**，所以两个视角**不可能共用一个 6D 位姿**。

反过来做 —— **从剪影反求位姿**（analysis-by-synthesis）：

| | 腕视图 `F.fit_wrist` | up 视图 `F.fit_up` |
|---|---|---|
| 相机 | 真标定：`K + dist + FK⊗手眼 + 每帧平移修正 δ` | **造出来的**：主点 = 观测形心、光轴竖直向下、焦距由「渲出剪影面积 = 观测面积」反解 |
| 解什么 | 6-DoF 剪影拟合 | 平面内平移 + 绕竖轴转角 + 深度 |
| 为什么不同 | 有真内参 | 没有内参，且相机高度**不可辨识**（见 `CAMERA_CALIB.md` §6） |

两个视角**各自拟合、各自锚定自己看得见的观测**；跨视角那 ~30mm 的残差
**当数字报出来**，不假装是 0。

`X_tcp_cube` 只在**夹取帧**算一次（搬运期一帧一算会解出 300mm 的假偏移）。
实测 `X_tcp_cube = [2.95, 31.76, −16.97] mm`。

---

## 4. 新物体是什么

注入的是**重建网格**，不是照片贴图，也不是 CAD 直读：

```
calib/recon/sts3215_03a_v1_v50_h0.5_m1.stl     ← 实际注入的
```

从多视角剪影**雕刻**出的 visual hull（`mesh_from_views.py`）。自检数字：

| 量 | 值 |
|---|---|
| 三角面 | 232 428 |
| 边界边 | **0**（水密） |
| 有符号体积 | 34 511.6 mm³（体素体积 34 896.4 ⇒ 差 1.1%） |
| winding 通过率 | **0.99991** |
| 视角数 / 余量 / 体素 | 50 / 1.0 px / 0.5 mm |
| CAD 真值 bbox | 45.4 × 24.8 × 39.6 mm（`sts3215_03a_v1.stl`，19 080 面） |

**它会比原来的方块大一圈，图上看得见 —— 这是换物体的必然结果，不是 bug。**

历史：最初注入的其实是 `MR.uv_sphere(12mm)` 一个**解析球**，半径来自卡尺，
一个像素都没参与。换成舵机就是为了让「图像→网格重建」这一步真的发生。

---

## 5. SaGA 叠在换完的图上

### 5.1 关键顺序：**必须把替换图塞回 batch**

```
替换图（哪个来源都行）
    ↓  当成数据集里的那一帧塞回去
preprocessor          ImageNet 归一化，和训练完全同一条路
    ↓
RoboSaGA              FullGrad 显著度 → 缓冲区 → 逐像素混合 OOD 背景
    ↓
图
```

**不能**拿原帧的显著度去叠替换图 —— SaGA 的显著度是**策略对当前这张图**算出来的
（论文 Algorithm 1 在线那一步）。拿原帧的权重配替换图，权重和它下面那张图对不上，
成了两件事拼一起。

### 5.2 batch 的摆法

前 `m` 个位置放「要替换、要展示」的帧，后面用另一集（ep1）的真实帧填空：

- SaGA 硬要求 batch ≥ 1/β = **10**（β=0.1，`int(n×β)` 截断取整）；
- α=0.5 只增强前一半 ⇒ 目标帧必须摆在前半段。

填空帧不替换，只是为了凑 batch。显著度逐帧独立、按全局帧 id 索引，跨帧不串，
所以不影响被展示那几帧。

### 5.3 预热必须模拟

不预热的话缓冲区初值全 1 ⇒ `s ≡ clip(1, 0, λ)² = 0.64` 是个**常数**，
图上就是 fixed-weight random overlay，**显著度引导完全没生效** —— 不报错、不崩。

本脚本的做法是对前 `k = n/2` 帧直接算 FullGrad 写进缓冲区：

```python
for key in camera_keys:
    maps = normalize_smaps(saga._fullgrad(batch[key][:k].clone())).detach()
    saga.buffers[key].set(maps, gid[:k])
```

⚠️ `saga.buffers` 按**完整键**索引（`observation.images.up`），不是短名 `up`。

### 5.4 参数（和训练一致）

| 参数 | 值 |
|---|---|
| `aug_ratio` α | 0.5 |
| `update_ratio` β | 0.1（⇒ batch ≥ 10） |
| `saliency_saga_cap` λ | 0.8（**权重上限是 λ² = 0.64**，不是 0.8） |
| `buffer_shape` | (120, 160) = 480×640 的 1/16 |
| OOD 池 | `../background_generalization/backgrounds/solid8`（8 张纯色，两臂共用） |
| `n` | 16 |
| device | **cpu**（两个 GPU 常被掩码任务占着） |

---

## 6. 实测数字

### 6.1 换物体的几何自检

| 来源 | 视角 | 旧物 → 网格 | 覆盖率 | 面积比 | 脚底 z | 红线外 |
|---|---|---|---|---|---|---|
| distractor | up | 1127–1256 px → 4602–4623 px | 100.0% | ×3.65–4.09 | +0.1 mm | 0 px |
| distractor | wrist | 1030–2389 px → 4113–7308 px | 100.0% | ×3.06–3.99 | +0.1 mm | 0 px |
| up_fit | up | 拟合 IoU 报在各行 | — | — | — | 0 px |
| wrist_fit | front | 拟合 IoU 报在各行 | — | — | — | 0 px |

腕视图锚点偏移 22–62 px（1 个候选）；每帧平移修正 δ =
`[5.6,8.8,17.9] / [6.4,22.4,16.5] / [3.3,27.4,5.3] / [−1.6,13.5,1.0] mm`。

### 6.2 SaGA 结果（注入区 s 均值 vs 全图）

| 来源 | 相机 | 注入区 | 全图 |
|---|---|---|---|
| distractor | up | **0.565**（18 436 px） | 0.219 |
| distractor | front | **0.576**（22 323 px） | 0.320 |
| up_fit | up | **0.621**（6 321 px） | 0.238 |
| wrist_fit | front | **0.612**（28 420 px） | 0.267 |

**读数**：SaGA 保护被注入的舵机明显强于桌面（约 2–3 倍），但仍然替换掉约 **4 成**。
原因：显著度来自策略自己的 backbone，而舵机是它训练时**从没见过的物体**
（训练集只有黄块、蓝块、红盒），对它没有可用的显著度。
**这条对「换物体」这个泛化目标是不利的，值得如实报出来。**

⚠️ 不要报「被改动像素占比」：在 solid8 这种纯色 OOD 池下它**恒等于 100%**
（s=0.5 的混合早就超过 1/255），没有分辨力，摆在图上是假数字。

---

## 7. 复现命令

```bash
cd /home/zjq/lerobot
PY=/home/zjq/miniconda3/envs/lerobot/bin/python

# 三个来源，出三张图 → workplace/saga/figures/swap_saga/
$PY -I workplace/saga/scripts/preview_swap_saga.py --source distractor
$PY -I workplace/saga/scripts/preview_swap_saga.py --source up_fit
$PY -I workplace/saga/scripts/preview_swap_saga.py --source wrist_fit

# 只出物体替换、不叠 SaGA 的原始图
$PY -I workplace/calibration/pose6d_distractor.py
$PY -I workplace/calibration/pose6d_up_fit.py
$PY -I workplace/calibration/pose6d_wrist_fit.py
```

帧号默认值（**绝对**数据集帧号，这一集从 0 开始）：

| 来源 | 帧 | 说明 |
|---|---|---|
| distractor | 0, 20, 100, 340 | 跨集散布，姿态差异才看得出来 |
| up_fit | 338, 246, 350, 402 | 246 = 夹取帧 |
| wrist_fit | 300, 336, 400 | 搬运段 |

可用 `--frames 0,20,100,340` 覆盖任意来源的帧列表。

产物：

```
workplace/saga/figures/swap_saga/distractor_saga.png
workplace/saga/figures/swap_saga/up_fit_saga.png
workplace/saga/figures/swap_saga/wrist_fit_saga.png
```

每张 4 栏 × (帧 × 相机) 行：
**① 原图 ② 物体替换后 ③ SaGA 权重 s（在②上算） ④ SaGA 增强后**。

---

## 8. 踩过的坑（按「静默失败」排序）

1. **`LeRobotDataset(...)` ≠ `make_dataset(...)`。**
   `make_dataset` 在 `use_imagenet_stats=True`（默认）下把**相机**的 mean/std 换成
   ImageNet 常数；直接构造拿到的是数据集自己的统计量（up 是 `mean 0.742 / std 0.0137`）。
   两者归一化空间差约 **17 倍**，OOD 背景与图像不在同一空间，混合出来就是垃圾 ——
   **不报错，只是图花**。脚本里有硬断言挡这条。
2. **不预热 ⇒ `s ≡ 0.64` 常数。** 见 §5.3。
3. **batch < 10 ⇒ 静默退化**（`int(n×β)=0`，缓冲区永不更新），跑的其实是
   fixed-weight random overlay，还挂着 SaGA 的名。
4. **判「改没改」必须在显示空间按 1/255 判。** ImageNet std≈0.22 ⇒ 归一化空间的
   1/255 只相当于显示空间的 ~1.8%，实测会得出「全图 100% 被改动」这种没有分辨力的数。
5. **色标要用专用 axes。** `ax=list(axes[:, 2])` 会让 matplotlib 从那一列中间切一条缝，
   色标正好压在子图标题上。用 `fig.add_axes([0.958, 0.30, 0.011, 0.42])`。
6. **`pose_T` 双重旋转** ⇒ 舵机底面沉到桌面以下 6.3mm。见 §3.1。
7. **`saga.buffers` 的键是完整键**，不是短相机名。
8. **`s` 是 (N,1,H,W)，`inj_t` 是 (N,H,W)** ⇒ 索引用 `s[:, 0][inj_t]`。
9. **干扰物那组必须用散布帧。** 连着几帧姿态几乎一样，四行会看着像同一张。
10. **cv2 Hershey 字体不能渲染中文** —— `pose6d_*` 系列图里全是 ASCII；
    saga 的 matplotlib 图用 `font.sans-serif = ["Noto Sans CJK JP", ...]`。

---

## 9. 红线与回归

- `state` / `action` / `timestamp` **逐位不变**（SaGA 只替换相机图像键，
  其余键原样引用；脚本里有逐位断言）。
- 机械臂、夹爪、蓝块、红盒、桌面：L2 构造性保证，实测红线外 **0 px**。
- 桌面回归基线：
  ```bash
  $PY -I workplace/calibration/ball_swap.py --frames 120,2033,3443
  # md5 必须仍是 1cdda71a80e3688274fb7266faf1243
  ```
- `workplace/so101_urdf/assets/*.stl` 是**只读输入**，不改。
- **在线增强只改训练时的张量，数据集落盘内容一个字节都不动。**

---

## 10. 相关文件

| 文件 | 作用 |
|---|---|
| `workplace/saga/scripts/preview_swap_saga.py` | **主脚本**。三个来源共用一个驱动 → `figures/swap_saga/*.png` |
| `workplace/calibration/pose6d_distractor.py` | 换干扰物。含 `anchor()` / `build_mesh()` / `composite()` |
| `workplace/calibration/pose6d_up_fit.py` | 换目标（up 视图），`fit_up` |
| `workplace/calibration/pose6d_wrist_fit.py` | 换目标（腕视图），`fit_wrist` |
| `workplace/calibration/fit_mesh_pose.py` | 位姿拟合 + `load_mesh` |
| `workplace/calibration/mesh_from_views.py` | visual hull 重建 |
| `workplace/calibration/CAMERA_CALIB.md` | 标定公式与误差预算 |
| `workplace/saga/CLAUDE.md` | SaGA 实现约定、开销、已知偏离论文之处 |
| `workplace/saga/figures/swap_saga/` | **三张交付图的落盘位置** |

---

## 11. 未决 / 没做的

- **注入的舵机渲成暗砖红**，来自既有的 `SERVO_ALBEDO = [0.20, 0.28, 0.66]`（BGR）。
  改它会让 `pose6d_distractor.py` 和 `pose6d_wrist_fit.py` 对材质不一致，
  所以**问了但没动**。
- 跨视角仍然**各自拟合**，不共享一个 6D 位姿（那 ~30mm 继续当数字报）。
- up 相机真实高度仍然不可辨识（名义 700mm，只影响球半径/`f↔h` 的解释）。
