# 相机标定：公式、约定与误差棒

本文只写**这台机器上真的解出来的东西**，以及每个数字的来源和它的**类型**
（残差 / 重复性 / 一致性 / 模型误差 / 假设）。汇报时这几类不能混着说 ——
0.04mm 的重复性和 0.04mm 的准度是完全不同的两句话。

两台相机：

| 相机 | 代号 | 位置 | 分辨率 | 标定产物 |
|---|---|---|---|---|
| 俯视 | `camera_A`（无序列号） | 桌面正上方吊装 | 640×480 | **内参 K+畸变（2026-10-19）+ 单应 H + 外参 T_base_cam（2026-10-19）** |
| 腕装 | `camera_B`（有序列号） | 夹爪上方偏左，光轴 = `wrist_camera_link` 的 **+z** | 640×480 | 内参 K + 畸变 + 手眼 X |

---

## 1. 坐标系的约定（先立这个，否则公式全部读反）

| 符号 | 含义 | 单位 |
|---|---|---|
| `base` | 机械臂底座系，z 轴朝上 | m |
| `board` | ChArUco 板系，原点 = **棋盘左上角**，+x 沿 150mm 边 | m |
| `cam` | OpenCV 相机系（x 右、y 下、z 朝前） | m |
| `T_A_B` | **B 在 A 系里的位姿**，即把 B 系的点变到 A 系：`P_A = T_A_B · P_B` | — |
| `q` | 关节角（已从 lerobot 的 ±100 归一化值换算成弧度） | rad |

正运动学 `urdf.fk(q, link)` 返回 `T_base_link`。板系 z 轴朝**下**
（`board_to_base.json: board_normal_up = false`），即板坐标系的 +z 与 base 的 −z 同向 ——
这不是笔误，是「板平贴桌面」这个约束的两个法线分支里数据选中的那一个，见 §4。

---

## 2. 针孔模型 + 畸变（腕相机）

$$\begin{bmatrix}u\\v\\1\end{bmatrix} \sim K\,[R\mid t]\,P_{base},\qquad P_c = T_{cam\_base}\,P_{base}$$

`MR.Cam.T_base_cam` 存的是 cam→base，`project()` 内部取逆：

```python
Pc = P_base @ T_cam_base[:3,:3].T + T_cam_base[:3,3]          # base → cam
uv, _ = cv2.projectPoints(Pc, 0, 0, K, dist)                  # 含畸变
```

畸变按 OpenCV 的 5 参数 `[k1, k2, p1, p2, k3]`（径向 + 切向）。

**实测**（`calib/front_intrinsics.json`，`cv2.calibrateCamera(..., CALIB_FIX_ASPECT_RATIO)`，
fx = fy 钉成相等）：

| 量 | 值 |
|---|---|
| fx = fy | 409.572 px |
| cx, cy | 321.709, 209.719 px |
| k1, k2 | −0.3449, +0.1647 |
| p1, p2 | −0.00132, −9.77e−5 |
| k3 | −0.04644 |
| **重投影 rms** | **0.300 px** |
| 逐位姿 | 0.301 / 0.274 / 0.305 / 0.247 / 0.169 px |

k1 = −0.345 是**相当大**的桶形畸变。把半径缩放因子写全：

$$s(r) = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6,\qquad r^2 = \frac{(u-c_x)^2}{f_x^2}+\frac{(v-c_y)^2}{f_y^2}$$

在 640×480 的**画面角点** r² = 1.039 ⇒ s = 0.767，也就是角点被**向内拉了 23%**
（若只算 k1 则被 k2/k3 部分抵消掉一半，所以单看 k1 会高估）。
这个量级意味着任何线性针孔近似都不能用 —— 本仓库所有投影像素都走 `cv2.projectPoints`，
光栅化里也带着同一组 dist。

---

## 2b. up 相机内参（2026-10-19 实测）

`up_intrinsics.py`：手举 ChArUco 板在 up 画面里做 **32 个不同位置/倾角**的姿势
（首轮 20 张 + 2026-10-19 补拍的 12 张左右倾），
`cv2.calibrateCamera(..., CALIB_FIX_ASPECT_RATIO)`。板格尺度**不进内参** ——
内参只需要板的透视，不需要知道一格多少毫米（所有格等比例缩放不改变 K）。

| 量 | 值 |
|---|---|
| 视角 | 32（角点 19–35/张） |
| 采用变体 | `fix_k2_k3`（只解 k1 + p1,p2；k2 = k3 = 0） |
| fx = fy | **1228.79 px** |
| cx, cy | 298.84, 242.75 px |
| k1 | **−0.4463** |
| p1, p2 | −0.00501, +0.00114 |
| **重投影 rms** | **0.404 px**（逐视角 0.18–0.52） |
| 板倾角 | 30 张 >15°，10 张 >30°，6 张 >45°（最大 68°） |
| 覆盖 | 68% × 81% 画面，9/9 宫格 |
| 方位覆盖 | −45°~+180°（首轮 20 张只有 +70°~+147°，左右倾是补拍才有的） |

为什么不用 5 参数：自由解给出 `k2 = +0.09, k3 = −7.08`（明显过拟合，rms 与
3 参数同为 0.404 px）。自动选型保留能收敛且不过拟合的 `fix_k2_k3`。

**稳定性（重采样，32 张）**：
- 24/32 bootstrap 95% CI：fx ±1.17%，cx ±5.2%，cy ±5.9%，k1 ±3.70%；
- 留一法（每次去掉 1 张重解）：fx ±6.1px，cx ±6.0px，cy ±7.4px；
- 自由长宽比解 fx=1234.82, fy=1232.12（fx/fy=1.0022，差 0.22%）。

> 首轮 20 张的解（fx=1226.66, cx,cy=304.41,220.83, k1=−0.4640）保留在
> `calib/up_intrinsics.json.bak-20views-20261019` 和
> `figures/up_intrinsics_check.20views.png`。补拍后 cy 移动了 +21.9px
> （旧 20 张的 bootstrap 95% CI 约 ±16px）；新解主点更接近画面中心，
> 且倾角方位从一维变成二维，所以采用 32 张解。

产物：`calib/up_intrinsics.json`、`figures/up_intrinsics_check.png`
（32 张的检出角点绿十字 vs 重投影红点）。

**适用边界（重要）**：这套内参对应的是**当前这次会话的 up 机位/取景**。
`CAMERAS.md` 已记录：数据集的 up 帧和现在实时抓的 up 帧朝向、距离都不同。
把旧数据集的 up 帧套这组 K 之前，必须先做图像对齐/确认取景一致
（或者用那批帧自己的相机重标）。

---

## 2c. up 相机外参（2026-10-19，平放板 PnP）

把 ChArUco 板**平放**在桌面，位置与 `calib/board_to_base.json` 解算时**同一物理位姿**。
实测板位差 **(+1.3, −5.4) mm / −1.83°**，板面相对桌面的平面拟合残差
mean 0.12mm / max 0.29mm —— 板位足够一致，可以直接用现有 `T_base_board`。

用 §2b 的 K+dist 对板角点做 `solvePnP(..., SOLVEPNP_IPPE)`，再左乘 `T_base_board`：

| 量 | 值 |
|---|---|
| `T_base_cam` 平移 | (459.13, 77.52, **684.67**) mm |
| 高度 `D = t_z` | **684.7 mm**（桌面按 z=0） |
| 光轴方向（base） | (−0.4326, −0.0859, −0.8975) |
| 光轴与垂直向下夹角 | **26.2°**（up 机位固有的斜视，不是板没放平） |
| PnP 重投影 | **0.160 px** |
| 检出角点 | 30/35（板顶边被画面上缘裁掉 5 个，不影响 PnP） |

产物：`calib/up_camera_pose.json`；抓图 `calib/up_flat_board.png`。
这个高度落在 §6 旧单应分解给出的 665.7–707.3mm 区间内，倾角也和
23.4–28.3° 一致，三条独立路径互相印证。

**尺度注意**：本次没传 `--measured-mm`，高度按板格**名义 25mm** 解算。
若打印出的 100mm 边长实测差 1%，D 会差 ~6.8mm。量出实际边长后重跑
（`--measured-mm <mm>`）即可把尺度钉到 ~1%。

**独立验证（2026-10-19，当前机位）**：把 40×22×12mm 的真实橡皮静置在桌面上，
中心高度 z = 6mm，用真 K+dist+`T_base_cam` 拟合 x/y/yaw（`up_live_eraser.py`）：

| 量 | 结果 |
|---|---|
| 剪影 IoU（覆盖率 vs GrabCut） | **0.9522** |
| 渲/观测面积比 | **1.0027**（尺度对） |
| 横向修正量 | **0.29 mm** |
| 解出位置 / yaw | (185.7, 41.5, 6.0) mm / +89.5° |

渲染图 `figures/up_live_eraser_fit.png`。外参若有几毫米/几度的错，40mm 的物体在
685mm 处会立刻显出尺度或位置偏差；这个结果说明 **§2c 的 T_base_cam 在当前取景下是可用的**。

**像面朝向（容易被忽略）**：`T_base_cam` 的旋转给出像面 x 轴
**(0.161, −0.987, 0.017)** ≈ **−base y**，像面 y 轴 ≈ (−0.887, −0.137, 0.441)。
也就是说这台相机的像面**不是**「x 对齐 base x」——差 ~90° 的 roll。
球/立方体看不出来；长条物按真实朝向渲染时，用错 roll 会让它在图里转到相反方向
（`pose6d_up_fit.py --up-rot real` 已默认用真实旋转，见 HANDOVER §6）。

**限制（必须记住）**：`dataset/newdata1` 的 up 帧是**旧机位**录的（`CAMERAS.md`：
朝向 + 距离都变了，红盒 200×230 → 130×150px）。上面的 K/外参**只对当前取景有效**，
不能直接套到旧数据集。`pose6d_up_fit.py --up-cam real` 对 `newdata1/newdata`
会**拒绝运行**（除非显式加 `--allow-framing-mismatch`，那只是错误示范）。
要用于旧数据，得先重录或做一次图像/机位配准。

---

## 3. 桌面单应（up 相机）

up 相机的内参已由 §2b 给出（2026-10-19）；但桌面接触面仍可以直接用单应描述，
而且**单应的求解不需要 K**：板平放在桌面上 ⇒ 板平面到图像是一个平面到平面的投影：

$$\begin{bmatrix}u\\v\\1\end{bmatrix} \sim H\begin{bmatrix}x_b\\y_b\\1\end{bmatrix},
\qquad \begin{bmatrix}x_b\\y_b\end{bmatrix} = \Big(H^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix}\Big)_{1:2}\Big/\big(\cdot\big)_3$$

**解法**（`up_homography.py`）：逐帧 `cv2.findHomography(obj_pts_m, img_px, RANSAC, 3.0)`，
取 12 帧的**中位数**合成，再按 `H[2,2]` 归一。

**尺度**：板是按 `square_m = 0.025 × (measured_mm / board_mm)` 构造的，也就是
**只有实测尺长能进公制尺度**。标称 25mm 直接当真的用，会有百分之几的系统误差一路进网格尺度。

**实测**（`calib/up_homography.json`）：

| 量 | 值 | 类型 |
|---|---|---|
| 平均重投影 | 0.185 px（最大 0.195） | 残差 |
| 帧间重复性（图像→桌面，逐帧互差） | **0.039 mm** | 重复性 |
| H[0,0], H[1,1] | 1784.3, 1725.1 px/m | — |

**关键局限**：单应只对 **z = 0**（接触面）精确。物体有高度 h 时，形心在 z = h/2 处，
投到 z = 0 平面上会有视差。对 h = 25mm 的方块，用 D ∈ [450, 700]mm 估出来是
**2~3 mm**。这个数**没有修正**，因为修正需要 up 相机的真实高度 D；
2026-10-19 已由平放板 PnP 解出 **D = 684.7mm**（§2c），需要时可据此补上视差修正。

**局部尺度**（把米换算成像素，比如画球的半径）：

$$\text{spm}(x,y) = \sqrt{\left|\det J\right|},\qquad
J = \frac{\partial\,\pi_H(x,y)}{\partial (x,y)}$$

用 2 点中心差分求 J。实测在蓝块位置是 **1688 px/m**。

---

## 4. 手眼标定与板↔base

要求的是

$$T_{base\_board} \;=\; \underbrace{T_{base\_ee}}_{\text{FK 已知}} \otimes \underbrace{X_{ee\_cam}}_{\text{待求}} \otimes \underbrace{T_{cam\_board}}_{\text{solvePnP}}$$

### 4.1 标准路：AX = XB

`front_handeye.py` 先用 5 个位姿跑 `cv2.calibrateHandEye` 的五种解法
（TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS），每一种都用
「逐位姿各自推一个 `T_base_board`，看它们散不散」来检验自洽性：

$$\text{spread}_t = \max_i \big\|\,\mathbf{p}_i - \bar{\mathbf{p}}\,\big\|,
\qquad \text{spread}_r = \max_{i,j}\ \angle\big(R_i^{\top}R_j\big),
\qquad \angle(R)=\arccos\frac{\mathrm{tr}R-1}{2}$$

**这条路失败了，而且失败原因值得汇报**：AX=XB 的**唯一**信息来源是不同位姿之间的
**相对转动**。实测那 5 个位姿倾角 26–56°、方位都在同一侧、wrist_roll 几乎没动 ⇒
相对转轴几乎共面 ⇒ 解病态：板原点散布 **27–37 mm**，解出来的板面甚至跑到桌面**下方 34mm**。
换内参假设、做留一法都改不动，说明是**整体性的**，不是数据不够。

判据（可复用）：所有位姿的相对转轴各向占比。含 48.7° 倾角那批时最小占比 0.053（够用）；
把大倾角砍掉后只剩 2.8–11.9°，掉到 0.011（退化）。**大倾角位姿不能少。**

### 4.2 用上的路：把物理约束钉死

`solve_flat_board.py` 换了个提法 —— 不去猜 X 的全部 6 个自由度，而是把已知约束写进去。
板平贴桌面 ⇒ 板面法线 = base 的 z 轴 ⇒ 板只剩 4 个自由度：

$$T_{base\_board} = \mathrm{Trans}(x,y,z_0)\cdot R_z(\psi)\cdot \mathrm{diag}(1,\mp1,\mp1)$$

未知数 **6(X) + 4(板) = 10**，而 14 个位姿提供 14×6 = 84 个约束。用
`scipy.optimize.least_squares` 同时最小化所有位姿的残差：

$$r_i = \Big[\ \mathrm{rotvec}\big(R_{B_i}^{\top}R_{board}\big)\cdot1000,\ \ (t_{B_i}-t_{board})\cdot1000\ \Big],
\qquad B_i = T_{base\_ee}^{(i)}\,X\,T_{cam\_board}^{(i)}$$

**实测**（`calib/board_to_base.json`，14 个位姿）：

| 量 | 值 |
|---|---|
| 板在 base 系 | x=295.094, y=105.861, z=0 mm，偏航 −86.903° |
| 板法线 | **朝下**（`board_normal_up = false`） |
| X 平移 | ≈ [18, −1, 7] mm（相机距腕 link 约 19mm，物理上合理） |

**三个把它毁过一轮的坑（都是一行代码的事，值得在汇报里当案例讲）：**

1. **板法线有上下两个分支。** 板平贴只说明板*面*是桌面，板坐标系的 z 朝上还是朝下是自由的。
   只允许 `Rz(ψ)` 等于钉死朝上，优化器绕不开那 180°，只好用
   「X 平移撑到 285mm + 把板抬到桌面上方 172mm」去凑 —— 残差只是从 26 涨到 208，
   *看着像「有点大」，完全不像少了一个分支*。
2. **别用线性最小二乘求初值。** `R_e A R_c = G` 对 (A, G) 是**齐次**的 ⇒
   **A = G = 0 永远是一个精确解**：一次满足 45 个旋转方程，代价只是 15 个平移方程。
   最小二乘会欣然选它。正确做法是先解**自由手眼**（X 6 + 板 12，无板平约束）当初值。
3. **板的高度 z 不是自由参数**，它等于桌面高度。留成自由参数的话这一维几乎退化
   （钉在 −100mm 和钉在 0，rms 只差 0.8），优化器会飘到物理上不可能的位置。

---

## 5. 每帧的腕相机位姿，以及那个平移修正 δ

腕相机的实时位姿就是 FK ⊗ 手眼：

$$T_{base\_cam}(q) = \mathrm{FK}\big(q,\ \texttt{wrist\_camera\_link}\big)\cdot X$$

**实测它不够准**：在数据集的构型上，模型预测的像素和观测像素差 **~30 mm / 150 px**，
而且这个残差**随构型变**，不是常量 —— 拟合一个 6 参数 ΔX 只把 RMSE 从 26.8 压到 19.3mm，
还解出 −75mm 的荒谬平移。所以常量手眼吸收不掉它。

**修法**：方块在 up 和腕两个相机里**都看得见**，于是每帧直接量出「预测像素 vs 观测像素」，
反解一个**相机在 base 系里的平移修正 δ**：

$$\delta^\star = \arg\min_\delta \sum_k \big\|\,\pi\big(K, dist,\ T(q) + \delta,\ P_k\big) - u_k\,\big\|^2$$

用 Gauss-Newton 解，每步取**最小范数解**：

$$\delta \leftarrow \delta + J^{+}(-r),\qquad J^{+}=\text{lstsq}(J)$$

**为什么必须是最小范数**：单点时这个 3 自由度问题是**欠定**的 ——
**沿视线平移相机，像素纹丝不动**，那个方向不可观测。最小范数解自动把修正落在
垂直于视线的分量上，沿视线那一段保持 0，不产生虚假的深度修正。多点时它退化成普通的过定解。

还有一条**不需要物体可见**的版本：把 δ 拟合成关节角的线性函数
`δ(q) = [q_1..q_5, 1] · W`（`calib/wrist_delta_model.json`），episode 留出检验中位
**4.44 mm**。代价是它是**这台臂 + 这个相机安装**的属性 —— 动过相机或重标过臂就得重跑。

---

## 6. up 视图的「局部相机」：为什么它不是标定

物体有高度时单应不够用（§3）。本仓库里对 up 视图所有非接触面几何，用的是一台
**造出来的、自洽的**针孔相机（`fit_mesh_pose.up_cam_from_area` / `pose6d_distractor.up_metric_cam`）：

- 光轴**垂直下视**（base −z）；
- 相机在「观测形心打到 z=0 的那个点」的正上方 `cam_h_mm`；
- **主点 = 观测形心** ⇒ 物心正好落在光轴上；
- 焦距 `f = spm(x,y) · cam_h_mm`（用 §3 实测的局部尺度），或由「渲出的剪影面积 = 观测面积」迭代反解。

$$\pi:\quad u = c_x + f\frac{X_c}{Z_c},\qquad v = c_y + f\frac{Y_c}{Z_c}$$

**必须说清楚的两件事：**

1. **f 现在有标定值了：1228.79 px**（§2b）。旧文里的 `f ≈ spm · cam_h`
   是 **K 未知**时唯一能写的约束；`spm` 是**桌面上**的局部 px/m，而桌面相对光轴
   斜 ~20–23°（下条），所以 `spm` 里含着平面斜视的投影压缩，**它不是 f / D 的直读**。
   蓝块处 `1688 px/m × 名义 700mm = 1182 px` 是**等效焦距**（保物点处的 px/m），
   和标定的 1228.79 px 差 ~3.8% —— 两个数回答的是不同问题，别混用。
   换物体渲染用的仍是「锚在观测形心 + 剪影半径」的局部针孔；它的 f 是等效量，
   **不要直接把 1228.79 塞进去**，除非同时把相机位姿改成下面解出的真实位姿。
2. **高度 D 已钉住：684.7mm**（§2c，平放板 PnP，rms 0.160px）。
   三条路径互相印证：新 K 直接分解旧单应给 **D = 707.3mm、倾角 23.4°**；
   按 K+dist 去畸变后重分解给 **665.7mm、28.3°**；平放板 PnP 给
   **684.7mm、26.2°**，正落在前两者的区间内。差异来自旧单应是在无内参时
   拟合的、以及板格名义尺度未实测，量级 ~20mm。
   `spm` 仍是**桌面 z=0** 的局部尺度；有高度 h 的物体的视差修正现在可以用
   D=684.7mm 估了（§3）。

---

## 7. 误差预算（拿这张表汇报）

| 环节 | 量 | 数字 | 这是**什么类型**的数 |
|---|---|---|---|
| 腕内参 | 重投影 rms | 0.300 px | 残差（自洽，不证明对） |
| up 内参 | 重投影 rms | **0.404 px**（32 视角） | 残差（自洽，不证明对） |
| up 单应 | 重投影 | 0.185 px (max 0.195) | 残差 |
| up 单应 | 帧间重复性 | **0.039 mm** | **重复性** |
| 桌面锚点 | 44 次采样散布 | ≤ 1.14 mm（中位 0.24） | 重复性 |
| 板↔base（标准手眼） | 板原点散布 | 27–37 mm | 一致性（**已弃用**） |
| 板↔base（板平约束） | 留一位姿预测误差 | 见 §4.2 | 留出检验 |
| 腕 FK ⊗ 手眼 | 跨相机 3D 不一致 | **~30 mm** | 一致性 |
| 腕 δ 修正后 | episode 留出中位 | **4.44 mm** | 留出误差 |
| up 单应 | z ≠ 0 的视差（h=25mm） | 2~3 mm | **模型误差** |
| up 相机外参 | PnP 重投影 | **0.160 px**（平放板，30 角点） | 残差 |
| up 相机位姿 | 板位一致性 | 1.3 / 5.4 mm，1.83° | 一致性（重放同一板位） |
| up 相机高度 | D | **684.7 mm**（平放板 PnP） | **标定值，尺度依赖板格实测长** |
| 公制尺度 | 板格长度 | 依赖实测尺，1.5% 尺误差 → 1.5% 尺度误差 | **系统误差** |

---

## 8. 汇报时不要说的话

- ❌「up 相机全标定好了，绝对精度 mm 级」→ ✅ 「up 的**内参** K+畸变（32 视角，
  rms 0.404px）和**外参** `T_base_cam`（平放板 PnP，rms 0.160px，D=684.7mm）都已标定。
  桌面单应仍只对 z=0 精确；公制尺度依赖板格实测长度（本次按名义 25mm，
  1% 尺误差 → ~6.8mm 高度误差）。」
- ❌「重复性 0.04mm，所以精度 0.04mm」→ ✅ 重复性 0.04mm，**准度**受制于板格实测尺（系统）
  和 z≠0 视差（模型），量级是 mm。
- ❌「f = 1182px / 597px 是 up 相机焦距」→ ✅ 标定焦距是 **1228.79px**（§2b）；
  1182/597 是换物体渲染里的**等效焦距**（分别保「物点处 px/m」和「剪影面积」），
  不是同一个量，不要再拿它们当标定值。
- ❌「手眼标定出 X 了，FK 是准的」→ ✅ X 解出来了，但 FK⊗X 在这台臂上差 ~30mm 且
  **随构型变**，所以每帧还要解一个平移修正 δ，δ 的沿视线分量**不可观测**。

## 9. 复现命令

```bash
cd /data1/lerobot
PY=/home/wzy/anaconda3/envs/lerobot1/bin/python

# up 内参（2026-10-19）。--serve 是无窗口 SSH 下的 MJPEG 预览；--auto N 采够就停
$PY -I workplace/calibration/up_intrinsics.py capture --auto 20 --serve 8080 --min-move 30
$PY -I workplace/calibration/up_intrinsics.py solve

$PY -I workplace/calibration/up_intrinsics.py extrinsic \
    --shot calib/<平放板图>.png --board-to-base board_to_base.json \
    --measured-mm <实测的 100mm 尺长>        # 钉死 D / T_base_cam

$PY -I workplace/calibration/up_homography.py --board-mm 100.0 --measured-mm <你量到的>
$PY -I workplace/calibration/front_handeye.py scan          # 找能拍到板的位姿（会动臂）
$PY -I workplace/calibration/front_handeye.py solve         # 内参 + 标准手眼
$PY -I workplace/calibration/solve_flat_board.py            # 板平约束，写 board_to_base.json
```

产物：`calib/up_intrinsics.json`、`figures/up_intrinsics_check.png`、
`calib/up_camera_pose.json`（跑 extrinsic 后）、`calib/up_homography.json`、
`calib/front_intrinsics.json`、`calib/board_to_base.json`、`calib/wrist_delta_model.json`。
