# CLAUDE.md

本目录有**两条独立的增强路径**，别把它们混起来：

| 路径 | 做法 | 文档 |
|---|---|---|
| **Stage 2B：桌面干扰物**（RoboAgent 的第二条轴） | **最小改动合成** —— 只在干扰物的 alpha 内作画，其余像素与原帧逐位相同；**不做整张背景替换** | `README.md`，**改代码前先读它** |
| **RoboSaGA：整张背景替换**（2026-09 加的第二轮） | 前景掩码之外**整块换成纯色/噪声背景** | 入口是 `masks/foreground_mask.py` 的模块 docstring，实测都记在那儿 |

两条路径的**安全红线相同**（文末），但"桌面"的处境正好相反：干扰物路径不动背景，
背景替换路径**整张桌面就是要被换掉的东西**。所以红线第一条点名的那几样
（机械臂、夹爪、蓝方块、红盒）必须由**前景掩码**保证逐位不被替换 ——
那份掩码存在的全部理由就是这个。

## 运行脚本

```bash
cd /home/zjq/lerobot
PY=/home/zjq/miniconda3/envs/lerobot/bin/python
S=workplace/background_generalization/scripts

$S/run_background_augmentation.sh                     # 全量（K=3，约 2.5 h）

# 冒烟（先跑这个，2 集 5 分钟）
$PY workplace/background_generalization/masks/build_plate.py --episodes 0 1 --out /tmp/smoke/plate
$PY $S/generate_distractor_dataset.py --episodes 0 1 --variants 3 \
    --plate-root /tmp/smoke/plate --out /tmp/smoke/ds --report-dir /tmp/smoke/reports
$PY $S/verify_distractor_dataset.py --report-dir /tmp/smoke/reports --stride 10
$PY $S/inspect_distractor_dataset.py --report-dir /tmp/smoke/reports --out /tmp/smoke/vis

# 遮挡审计（全量约 30 分钟）。改了阴影门槛/判据之后**必须重跑这个**：
# verify 只查自洽性，查不出「这次遮挡本来就是错的」——那正是阴影 bug 藏身的地方。
$PY $S/audit_occlusion.py --report-dir workplace/background_generalization/outputs/distract
$PY $S/audit_occlusion.py --report-dir /tmp/smoke_shade/reports --gate-override 2.0  # 自测
```

## 运行脚本（RoboSaGA 整张背景替换）

```bash
cd /home/zjq/lerobot
PY=/home/zjq/miniconda3/envs/lerobot/bin/python
S=workplace/background_generalization/scripts

# 1. 机械臂/夹爪掩码（SAM2，要 GPU；两卡各跑一半约 2 h）
$PY $S/build_robot_mask_cache.py --episodes all --skip-existing --retries 3
#    --retries 3 = 自检不过就换提示重跑（见下文「重试阶梯」）。--skip-existing
#    只做缺文件的那几档，所以补跑很便宜；跑完再看还剩哪几档是真的过不了。
#    --retries 默认 0，逐位等于没有阶梯 —— 已经落盘的掩码不会因为开了它而变化。

# 2. 前景掩码 = SAM2 机械臂 ∪ 黄/红/蓝（含外扩）。纯 CPU，约 1-2 min/集/相机
$PY $S/build_foreground_mask.py --episodes all --skip-existing

# 3. 背景池。--mode solid 就是增强本身做的那件事（纯色）
$PY $S/make_backgrounds.py --mode solid --n 8 --out workplace/background_generalization/backgrounds/solid8

# 4. 生成（--out 与 --report-dir 都必填；--out 必须是**不存在**的目录，
#    脚本拒绝已存在的目标，且 LeRobotDataset.create 不能续跑）
#    --no-mask-policy passthrough：某集某相机没有前景掩码时，这一档不增强、原样复制
#    （见下文「掩码修不好时怎么办」）。不加这个就是默认的 fail=当场停。
$PY $S/generate_background_dataset.py --episodes all --cameras up front \
    --no-mask-policy passthrough \
    --background-root workplace/background_generalization/backgrounds/solid8 \
    --out workplace/background_generalization/outputs/bg_all50 \
    --report-dir workplace/background_generalization/outputs/bg_all50_reports

# 5. 校验（--report-dir 与 --base-root 必填；豁免档数会跟 RESULT 一起印出来）
$PY $S/verify_background_dataset.py \
    --report-dir workplace/background_generalization/outputs/bg_all50_reports \
    --base-root /home/zjq/lerobot/dataset/newdata1 --stride 10
```

冒烟再加 `--limit 40`（每集只用前 40 帧），几分钟就能把四个阶段走一遍。

`--cameras up` / `--cameras front` 是「up 和 front 需不需要不同的增强」那个问题的对照臂。

### 子集生成时 `verify` 会**挂**在 huggingface 连接上，而不是报错

生成集里的 episode 编号是**按写入顺序从 0 重排**的，不等于 base episode
（`_cache_io.gen_positions` 的 docstring 记了这件事）。`verify_background_dataset.py`
的 phase 1 用了那个映射，phase 2 漏了 —— 它拿 base 号去
`LeRobotDataset(gen_repo, root=gen_root, episodes=[ep])`，子集生成时那一集的本地数据不
存在，于是走进 `LeRobotDataset.__init__` 里"从 hub 拉"的分支，**挂在连接超时上**
（`connect timeout=None`，实测几分钟到无限，退出码还可能是 0）。全量生成（`--episodes all`）
时两个编号恰好相等，所以一直没暴露 —— 这条路径只在**子集生成 + verify** 时才踩。

已修：phase 2 改用 `generate.json` 里记录的 `gen_episode`（比 `gen_positions` 还稳，
顺带不依赖"请求列表升序"这个假设）。所以现在 `--episodes 5 --out ... ` 这种子集生成
也能正常 verify —— 上面那个 passthrough 冒烟就是这么验的。

### 掩码修不好时怎么办（`--no-mask-policy passthrough`）

掩码建不出来的那些档（第 2 步自检挡下的），到第 4 步可以选**不增强这一档**：原样复制
基线的帧。这是红线第三条「宁可漏增强，不要错增强」的**安全侧** —— 不增强属于「漏」，
而拿一份 `ok=False` 的掩码去合成是把机械臂换成背景色（「错」）。

但它必须**显式且可查**，因为这条路的危险形态是**静默**：那一档会变成基线的逐位复制品
却占着「已增强」的编号。所以：

- `generate` 把 `passthrough` 逐 (集, 相机) 写进 `generate.json`，并在结尾大声报
  「N/M 档没有增强，处理被稀释 N/M」；
- `verify` 读这份名单，把这些档**排除**在常规检查外（它们没有掩码，`np.load` 会直接抛；
  背景本来就不该是池子里那张；「背景只变了 N%」必然是 0%），换成**反向检查**：
  `gen` 与 `base` 的中位差只该有码流噪声（`TOL_PASS_DRIFT = 8/255`，AV1 实测中位约 2），
  超了就是「标了不增强其实改了」。豁免档数会跟着 `RESULT` 一起印：
  `PASS (0 条失败，7 档豁免未验)`) —— 免得「PASS」看起来像全都验过了。

**这条反查是双向的，两边都实测过**（冒烟数据集 base ep4：真增强的 up 报了 61.5% 改动）：

| generate.json 里的说法 | 实际 | verify 结果 |
|---|---|---|
| 老实标 passthrough | 真没增强 | `PASS (0 条失败，1 档豁免未验)`；漂移 2.0/255（码流噪声底） |
| **伪标 passthrough** | 真被增强了 | `FAIL ep4 up: 标着「不增强」，但 gen 与 base 中位差 80.0/255 > 8.0` |
| 把它当"已增强" | 真没增强 | `FAIL … 背景只变了 0.1% < 50%`、`背景对不上 96.0/255`（原有两条） |

所以**那份名单必须和实际一致**，两个方向都藏不住：伪标 passthrough 会被漂移闸门抓，
把 pass-through 说成已增强会被"背景变了"抓。

**两个必须一起看的东西**：那些档把**原背景**喂进训练集，所以处理被稀释（按上面约
19% 估，不是小数目），方向上偏向「增强没用」（保守），要在报告里写明比例 —— 比例从
`generate.json` 的 `passthrough` 长度直接算，别照抄这里的估计。以及**对照臂之间必须
一致**：要么三臂都用 passthrough（都 50 集），要么三臂都删**同一批**集（取并集）；
**千万不能各臂删自己缺的那几集**，那样三臂的 episode 集合不同，跨臂比较就被污染了。
按臂算缺口：`--cameras up` 只要 up 掩码、`--cameras front` 只要 front，两路都跑才是并集
（当前各约 6 档，见上）。

### 掩码约定与三处会搞反的极性

掩码本身：**1 = 保留（前景），0 = 替换**。

| 文件 | 陷阱 |
|---|---|
| `robot_mask` 缓存 | 存 **0/255**，255 = 机械臂。`load_robot_mask` **原样返回 0/255、不做换算**，消费方一律用 `> 0` 判断 —— 拿它当 0/1 用（`mask == 1`、当 alpha 直接乘）会**静默**出错 |
| `foreground_mask` 缓存 | 存 **0/1**（不是 0/255）。`save_foreground_mask` 会拦下 >1 的输入，**但它的旧理由（"存 0/255 会让合成什么都不替换"）是错的** —— `_to_01` 在 `max > 1.5` 时自己除 255，两种格式都吃。闸门留着是为了磁盘上只有一种格式，见那个函数的 docstring |
| `placement.npy`（干扰物路径） | 1 = **背景**，和前景掩码的"1 = 保留"**相反** |

### 这个路径上的坑

**`_grow` 在空掩码上会凭空造出前景（2026-09-22 全量构建时踩到，已修）**

`ndimage.distance_transform_edt` 在**没有零点的输入**上（= 掩码整帧为空）把数组**外面**
当背景，`edt <= r` 于是返回画面左上角一个半径 r 的圆盘象限。`binary_dilation` 对全
False 返回全 False，是对的 —— 而 `_grow` 换 EDT 的动机只是速度（r=24 时 0.21 s/帧 →
0.013 s/帧）。

链式反应，数字都对得上：黄方块在 up ep0 里**恰好空 164 帧**（常被夹爪挡住）→
`_grow(空, task_dilate_px=3)` 在左上角造出 **7 px** 假种子 → 并进 union →
`_grow(·, 24)` 长成 **556 px** 角落圆盘（逐位可复现）。后果是左上角 27×27 被判成
"要保留"，合成时那块**真背景不替换** —— 增强里留了一小片原背景，正是红线第三条
要挡的方向。实测差异：up ep0 **91184 px**、front ep0 **33916 px**。

修法是在 `_grow` 里对空帧直接返回空，另加 `_check_no_phantom` 不变量：**外扩不能让
一整帧空掩码长出东西来**。这条断言是**会失败的**（喂错误实现 → 报"164 帧、首帧
t=264"，和上面同一组数字），不是"测自己定义"的恒真断言。改完 up/front ep0 与
改动前的磁盘掩码 **0 px 差**。

教训：当初宣称"EDT 与 `binary_dilation` 逐位等价"时**只测了非空掩码**。验等价性
必须包含**退化输入**（全空、单像素、贴边），否则只是把没覆盖到的那一支换了个实现。

**`--skip-existing` 之前先验已落盘的掩码干不干净**

判据要**逐位**：`mask[t][:27,:27]` 是否**恰好等于**幻影模板
（`_grow(_grow(全零, 3), 24)` 那份）。用"与模板相交"会**误报** —— 合法前景进入角落
也会命中，实测在 ep0 up 上误报 21 帧，而那份掩码已被证明与修复版逐位一致。先在
已知好/坏两份上标定（坏文件应报 164/61 帧，好文件 0 帧），再去扫全量。

**自检不过的 5 集：提示选错了目标，不是掩码质量差（`--retries` 重试阶梯）**

全量构建时自检挡下一批。跑到 ep20/ep34（52 档）时是 **12 档失败 ≈ 19%**，**两路各 6 档、
完全对称**（ep5/10/12/14/31/32 front，ep9/10/11/14/18/30 up；ep10、ep14 是两路都失败）。
所以全量 100 档估计 **18–20 档**，每路各约 9–10 档。典型样子是**面积 0.0%、覆盖 0–85%**。

这些不是"掩码差一点"，是**提示本身指错了东西** —— 同一份提示重跑一百次都是同一结果，
必须换提示。**对称**这件事对本阶段的实验是运气好的：`--cameras up` 与 `--cameras front`
两个对照臂被 pass-through 稀释的程度因此大致相同（各约 18–20%），跨臂比较没有被单边
削弱。如果它俩不对称，那才是真麻烦。

- ep5 front：`median-structure` 取"上半部最大结构块"，ep5 里有两个相当的候选
  （5197 px 列 50–176、3932 px 列 289–396），代码取了前者，**夹爪是后者**
  → 实测 rank=1 给出的框是 `[277,0,408,157]`，正是那一块。
- ep9/10/11 up：`frame-diff(p95)` 的框把机械臂裁掉了（ep9/ep11 的 y1=479 贴着下缘），
  或者框本身不含机械臂（ep10 的框 `[224,203,533,351]` 完全在画面内部）→ 后几档放大框
  或退回整幅图。

`--retries N` 打开阶梯：第 0 档自检不过就按 `prompt_candidates` 换下一档重跑 SAM2，
**每档都过同一个 `_self_check`**（阶梯只多试、不降标准，所以最坏结果与不重试时相同，
多花的只是 GPU 时间 —— 一档约 1 min）。报告里 `attempts` 逐档记录、`n_attempts` 是实际
跑了几档。三条已用假传播函数验过的性质：中间档成功就停；全不过时返回**第 0 档**的掩码
（后面的档一档比一档宽松，末尾是整幅图，`--force` 拿到它会得到一份"什么都不替换"的废
掩码，看着像成功）；`override` 给的手动框**不参与**阶梯（人指定的目标自检不过就该报出来）。

- `retries=0`（默认）逐位等于没有阶梯，**已落盘的掩码不受影响**，不用重建。
  rank=0 的提示和加阶梯之前是同一个（已逐位验过）。
- 但**阶梯能否真的救回这些档还没验**（要 GPU 实跑）。CPU 上只验到"给出的是不同的、
  指向夹爪的提示"，没验"过了自检"。跑完看它们落没落盘才算数。
- 若某档最终仍不过，再考虑手工框（`build_robot_mask(..., override=...)`，`tier=manual`）
  或把它从 `--episodes` 里去掉。注意 `generate_background_dataset.py` 遇到缺掩码会
  SystemExit，而它那时**已经建好输出目录**、那个目录又不能续跑（`mkdir(exist_ok=False)`），
  所以必须在上游把掩码补齐，或者事先把这些集从 `--episodes` 里剔掉。

**跨相机比较只有 `--fg-mode union` 一档可比**

`action_saliency.py --fg-mode auto` 会给两个相机选**不同**判据（实测 up 走 motion、
前景 64.4%；front 走 sam2、前景 14.2%），于是比的是**背景池大小**而不是敏感度。
`union` 档读的是增强本身那份掩码（up/front 都约 32%）。即便这样，**逐格对比的位置
仍不可比**（两路之间没有单应，见上文四条证据），只控住了面积。

**动作敏感度回答不了"要不要不同的增强"**

它测的是**在分布**的扰动，而泛化失败是**离分布**的。实测：`bg_const` 的 up/front 比值
中位 **1.49x**（5 集从不下穿 1），但**非背景**对照档更大 —— `fg_const` **2.56x**、
`full_noise` 1.83x、`full_const` 1.78x —— 所以那是**普遍**的不对称，不是背景特有的；
逐格等面积下纯背景格中位只有 **0.97x**（5 集里 3 集 front ≥ up）。要回答训练层面的
问题只能真去训两组策略比。原始数字在 `outputs/saliency/*.json`，汇总脚本
`compare_camera_sensitivity.py` 的 docstring 列了四条读表注意。

## 踩坑记录（Stage 2B 干扰物）

- `datasets` 模块遮蔽：本目录的 `datasets/` 会在 `workplace/` 加入 `sys.path`
  时遮蔽 huggingface 的 `datasets` 包。脚本要把仓库根和 `workplace/` 插到
  `sys.path` **最前面**，并先 import `lerobot.datasets.lerobot_dataset`。
- 写入格式：`LeRobotDataset.add_frame` 的图像必须是 **(H, W, 3) uint8**；
  frame 字典里不要放 `timestamp`（`validate_frame` 会报 "Extra feature"）。
- `LeRobotDataset.create` **不能续跑**（`mkdir(exist_ok=False)`），`--out` 必须是
  不存在的目录。跑到一半失败只能挪走重来；stage 1（plate）是断点续跑的，
  stage 2 不是。
- 生成一集约 55 s（511 帧、K=1），50 集 × K=3 ≈ 2 h。瓶颈是写盘 + AV1 编码，
  不是合成。

### 三个"名字会骗人"的数组

| 文件 | 陷阱 |
|---|---|
| `plate.npy` | **RGB**，不是 BGR。拿它直接喂 `cvtColor(..., BGR2HSV)` 会得到零个红色像素，看着像"红的被漏掉了"，其实是通道顺序。 |
| `freq.npy` | uint8，存的是 `clip(freq,0,1)*255`。直接打印会看到 6.4 这种"大于 1 的分数"。 |
| `placement.npy` | 极性与 object_generalization 的 `up_mask.npy` **相反**：这里 1 = "可以放干扰物"（背景），那里 1 = "要保留"（前景）。同名同 dtype 反语义是会被静默搞混的，所以这里故意换了名字。 |

### 两个必须守住的耦合

- `LayoutConfig.fg_threshold` **不得大于** `PlateConfig.diff_threshold`。大于会让
  合成器把 `placement` 认为是前景的像素判成背景，从而**把干扰物画到机械臂上**。
  `build_plate` 把 config 写进 `done.json`，生成脚本读回来，`FrameCompositor.__init__`
  会直接 raise。
- `ProceduralConfig.feather_px` ≤ `PlateConfig.exclude_dilate_px`，否则羽化边缘
  会渗回红盒。

### 合成器的 alpha 必须是 `paint_union`，不是 `alpha_union`

`render_background_layer` 把**接触阴影**也烤进了背景层，所以 alpha 必须同时覆盖
轮廓**和阴影**，否则阴影被烤进去却从没被合出来——每个干扰物变成没有影子的
"贴纸"，而同一张图里的方块、红盒、机械臂全都有影子。

实测：背景层改了 4482 px（1884 物体 + 2608 阴影），只有 1884 px 进了画面。

这个错误**单看任何一帧都不明显**，所以 `FrameCompositor.__init__` 会校验传入的
alpha 是否等于 `layout.paint_union(...)`，不等直接 raise。`alpha_union` 保持
"只有轮廓"的语义——遮挡度量和 footprint 断言说的是物体本身，不是它的影子。

### 阴影要"压暗"，不是"擦掉"

`foreground_t` 是 `|frame - plate| > 18`，**故意多盖**——它把机械臂投在桌上的
**影子**也算进去了，因为安全方向是宁可藏起干扰物，也不能把干扰物画到机械臂上。
但"多盖就擦掉"是错的，而且非常显眼：机械臂进出工作区时，影子是**整片扫过**桌面
的，于是所有干扰物**同时**消失。实测：6 个冒烟变体**全部**出现 `max_hidden = 1.0`
（至少一个物体被整块擦掉），最长 1.4 s。

影子不是遮挡物。它是同一块桌面在更少的光下，物体还在那儿，应该被**压暗**而不是
删除。两者逐像素可分，不需要额外标定：影子是**乘性**的，`frame_c / plate_c` 在
R、G、B 上取同一个值；真挡在桌面前面的东西不会。

逐像素判据外面还套一层**逐物体**门槛：一个物体只有核心 ≥ `shadow_core_frac`
(0.90) 的像素满足乘性才算"在阴影里"。这层门槛才是安全论证——黑色的机械臂连杆
压在干扰物上时，逐像素看大部分像素仍然"像影子"，纯逐像素规则会留下物体、把它画到
机械臂上。实测 40 个遮挡区间：36 个影子区间得 99.5–100%，3 个真机械臂区间得 0%，
唯一的混合区间（连杆压住约 34% 核心）得 65.7%——门槛落在一个空带里，不是调出来的
边界。

这个 bug **所有校验器都看不见**：帧和代码意图完全一致，没有任何自洽性被破坏。
能抓住它的只有「这一次遮挡是哪个机制造成的」这个问题，所以有了
`scripts/audit_occlusion.py`。它逐区间分类（影子 / 机械臂 / 混合），并用
`--gate-override 2.0` 提供**会失败的自测**：门槛开 → 4 次遮挡 0 影子；门槛关 →
40 次 36 影子。第二个数字要和第五节的旧普查对得上，否则说明工具坏了而不是代码对了。

`shadow_k_lo` 取 0.55，而不是"略低于实测最暗影子"（0.73），因为两种错误的代价不对等：
取高了会把真影子擦掉（就是这次要修的闪烁），取低了会把中灰的机械臂部件当成影子，
即**把干扰物画到机械臂上**。

注意 `k` 是在 **gain 归一化之后**测的（`frame * gains[0]/gains[t]` 再除以 plate），
所以它只反映**局部**光照变化（影子），**不含**本帧的全局曝光——这是好事，判据因此
对曝光变化免疫。

**但这不代表合成器没有曝光问题。** 另外还有一个**尚未修**的老问题：背景层是
frame-0 光度级的，`composite` 却把它直接和 frame-t 的**原始帧**混，而
`frame_table = plate_table * gains[t]/gains[0]`（实测相关 -0.99）。所以拉进来的物体
相对周围桌面偏了 `gains[t]/gains[0]`，ep0 最多 6.7%、ep1 最多 18.5%。修法是在
`composite` 里把背景层乘上这个比值（等价于除以 `front_scale`），但那会改动每一帧的
渲染，需要单独验证，所以没有和阴影修复捆在一起。

### 坐标约定（现在是 identity，但接口没变）

`warp.npy` 里存的仍然是 `(T, 2, 3)`，`to_frame0` / `to_frame_t` 仍然要按
**frame-0 → frame-t** 解释：

- 把 frame t 重采样回 frame-0：`warpAffine(frame_t, M_t, flags=... | WARP_INVERSE_MAP)`
- 把 frame-0 的图层渲染回 frame t：`warpAffine(arr0, M_t)`（**不加** inverse 标志）

搞反不会报错，只会让对齐悄悄镜像。

**但 `_align` / `findTransformECC` 已经删了**，现在 `build_plate._static_warps`
直接返回 identity。原因是 UP 相机的移动量在整集里 ≤ `max_camera_shift_px`，
identity 与真值只差几个像素、且**每帧差的是同样几个像素**，估 warp 反而引入
逐帧抖动。identity 不是"近似了一个大位移"，它就是答案。

保留数组、保留接口而**不是**教调用方跳过 warp，是为了让"场景静止"这个假设
继续是一个**被读出来、被检查的值**，而不是散在四个文件里的特例。

以前那段"不做对齐右侧整条边会被判成前景"的论证是 ECC 时代的，**已经不成立**：
现在两边都是 identity，没有对齐误差可言。真正的位移风险改由
`PlateConfig.max_camera_shift_px` 这个**被检查的量**承担。

**这个 identity 结论只管 UP 相机。** 实测（phase correlation，整集）：
UP 的 `dx,dy ≈ 0.1–0.5 px`（响应 0.2–0.77），确实静止；**front 不是**。

### front 视角：up→front 的单应解不出来（实测，四条独立证据）

想让 front 也带上同一批干扰物，就得有 up 像素 → front 像素的映射。桌面是平面，
所以这个映射**理论上**是一个单应，而且**理论上**不用 diffusion、不用重建。
实测**解不出来**：

| 做法 | 结果 |
|---|---|
| 帧对特征匹配（SIFT/ORB，跨 24 帧汇总） | ep0/1/2 各 466/451/386 个好匹配 → RANSAC 内点 **21/25/21（4.5–5.5%）** |
| └ 独立验证：up 的方块 mask 经该 H 之后 vs front 的方块 mask | **IoU = 0.000**（三集都是） |
| └ 独立验证：各集独立解出的 H 应当一致（相机没动） | 最大绝对差 **63.9**（H 元素量级 365）—— 互相矛盾 |
| 时域中值图（把机械臂中值掉）再匹配 | 442×199 关键点 → **5 个好匹配，0 内点**。桌面几乎无纹理 |
| 用方块 mask（两边都有）当对应点直接拟合 | 接触点法 LSQ 残差中位 **55 px**，而方块在 front 里只有约 50 px 宽 |

`front_x vs up_x` 的散点**不是函数**（同一个 up_x 对应 240–450 的 front_x）。
注意这条**不能**单独下结论：方块在两个视角的**可见性几乎是互补的** ——
up 的 mask 有 **175/568** 帧为空、front 有 **157/568** 帧为空；t=425–550
up 有 324 px（俯视能看进红盒）而 front 是 **0**（低角度看不进去），
t=200–400 反过来。**没有任何一段帧区间里方块同时是两边干净的平面点。**

根因是几何 + 这份数据缺标定：front 是**近距离低角度掠视**（画面里同时有绿垫、
蓝/橙纸盒、墙），up 是**俯视整张桌面**；两者的共享桌面比看上去小，而且在 front
里被掠视压缩。另外 front 视角在整集里**不稳定**：背景纸盒在画面里明显平移、
尺寸也变了，front 的时域中值图整体是重影。是相机在动还是场景被动过，
**我没能测准**（短窗中值图之间只有 7–18 个内点、解出的变换是退化解，
缩放到 0.077/0.31 这种不可能的值），所以只能报"不稳定"，不能报成因。

**要真正做这件事，缺的是标定，不是代码。** 一次棋盘格录制（两个相机装在
同样位置，覆盖桌面）→ 内参 + 外参 → 精确投影，顺带给出**深度序**，
而那正是"把干扰物放进机械臂工作区"也缺的那一样。**一次标定同时解开两件事。**
另一条不用重录的路：用 `observation.state` + SO101 URDF 做正运动学，
拿夹爪 TCP 的 3D 点当已知点标两个相机（`src/lerobot/model/kinematics.py`
需要 URDF + `placo`，仓库里没带 SO101 的 URDF），但虚拟 TCP 点在图像里的
定位精度是瓶颈，是个正经课题不是顺手能做的。

### 写校验时：小心"永远不可能失败"的断言

`verify_distractor_dataset.py` 里原本有一条 "干扰物不能画到前景上"：
`(a > 0) & real`。它**恒为 0**——因为合成器定义 `a = paint * (1 - fg)`，
所以 `a > 0` 本身就蕴含 `fg == 0`，而 `real ⊆ fg`。这种断言看着像证据，
其实是在测自己的定义，报 0 报一辈子。

现在的做法是**跨实现比对**：用 `_independent_fg` 从 cache 独立重建一遍前景
mask（自己的阈值、面积下限、膨胀、方块并集），要求两张 mask 逐位一致。
这样"没画到机械臂上"才落在 `plate.npy` / `gain.npy` / `warp.npy` 上，
而不是落在某一个函数的自洽性上。

写新断言时先问一句：**这条断言在什么输入下会失败？** 答不上来就是恒真。

### 度量本身也可能是错的

拟合阴影时踩过两次，都是"工具错了而不是模型错了"：

- `sqrt(area/pi)` 是**等面积半径**。对方块/团块按"1.25 r"采样，落点相对真实边缘的
  位置和圆盘完全不同，于是同一个正确的模型报出 0.19 的误差。换成形状无关的量
  （"阴影衰减到边缘值 10% 时离轮廓的距离 / 该半径"）后是 0.98，而方块实测 ≈1.0。
- 用**解码后的 AV1 帧**去比对精确期望值、再按 `max` 判失败，会得到 43/255 的
  "失败"——那全是编码噪声。改成按中位比值（对边缘不敏感）判，噪声降到 0.014。

报误差之前先确认**这个误差是不是度量方式造出来的**。

### 度量指标

判断"干扰物被挡了多少"时，分母要用物体**核心**（`alpha >= 0.5`），不能用
`alpha > 0`：羽化边缘按构造就满足 `alpha < 0.5`，用 `alpha > 0` 会让每个物体
都恒定假报 ~22% 被遮挡。库里的 `FrameCompositor.footprint_occlusion` 已经处理好了。

### 第二个原因（同一次会话里查出来的）：`gains` 量的是机械臂面积，不是光

上面那个阴影修复**只解决了一半**。修完重跑全量 50 集 + 遮挡审计，72 次遮挡里
**影子 0 次**（门槛守住了），但出现了 **59 次 `LIGHT`**：

`PlateConfig.brightness_percentile = 60`，而机械臂是暗的、最多占画面 **32%**。
机械臂一旦占够面积，**第 60 百分位就落进机械臂内部**，`gains[t]` 于是跟着
**机械臂的面积**走，而不是跟着光走：

| 检查 | ep0 | ep3 | ep7 |
|---|---|---|---|
| `corr(桌面占比, gains[t])` | **-0.80** | **-0.92** | -0.44 |

ep1 尾段最清楚（t=568..570）：桌面自己的**原始** `frame/plate` 全程 0.96–1.006
**完全没变**，而 `gains` 从 195 掉到 162，`front_scale` 涨到 1.178。于是
`foreground` 拿一张**没变过的桌面**去比一个把它抬亮 17% 的"校正"，残差 21 灰阶
超过 `fg_threshold = 18` → 整片静态桌面被判成前景 → **放在上面的干扰物全被擦掉**。
ep3 上这个假前景平均占合法区的 **13.1%**。

实测修法（在桌面上打分 `norm_ratio = median(frame_norm/plate)`，应当恒为 1.0）：

| 增益估计 | 桌面 norm_ratio 中位 | p95 | ep3 假前景占比 |
|---|---|---|---|
| 现在的 全局 p60 | 1.067 | 1.091 | 0.131 |
| 全局 p85 | 1.068 | 1.091 | 0.123 |
| **静态桌面内 `frame/plate` 中位** | **1.000** | **1.000** | **0.025** |

**提高百分位没用**（机械臂面积可以超过 40%），必须换成**在静态桌面区域内**估计。
注意 0.025 是 `fg_dilate_px` 的既定外扩，不是缺陷。

所以 26 集 `max_hidden = 1.0` 的成因是**这个**，不是机械臂：审计里 26 集的
`max_hidden=1.0` 全部有 `LIGHT` 解释，且**没有任何一次遮挡被归为机械臂**。
（会话早期我把 gen4/gen5 边界那几次当成"真机械臂遮挡"报给你，那是错的。）

## 安全红线（本目录内不可放松）

- 绝不重着色 / 移动 机械臂、夹爪、蓝方块、红盒、桌面。
- `state` / `action` / `timestamp` 必须逐帧 bit-identical。
- 宁可漏增强，不要错增强。
- 不改 `src/lerobot/` 下任何文件。

**第一条在两条路径上怎么读**（2026-09 你确认过）：它约束的是**重着色这个动作本身**
（颜色替换那一轴的规定），不是"背景增强"。所以：

- 机械臂 / 夹爪 / 蓝方块 / 红盒 —— 两条路径都**逐位不动**，由前景掩码保证
  （`foreground_mask.py` 的模块 docstring 记了这份掩码为什么必须存在、为什么宁可多盖）。
- **桌面** —— 干扰物路径不动它；背景替换路径**它就是待替换的背景**，这正是第二阶段
  的目的。不是疏忽，是设计。

其余三条两条路径完全一致：`state`/`action`/`timestamp` 逐位相同、宁可漏增强不要错增强、
不碰 `src/lerobot/`。第三四五节那些"哪些像素被改了"的度量方法对这条路径同样适用，
但注意上面那条**极性表** —— 两条路径的掩码语义是反的。

## 本目录现在是对照臂（2026-09-22 定位变更）

用户 2026-09-22 决定改做**真 RoboSaGA**，主臂移到了 `workplace/saga/`。
本目录的离线硬掩码背景替换**降级为对照/基线臂，保留不删**。

这不是"旧代码没用了"，而是"对照组"。它的全部痛苦（SAM2 分割、自检、~19% 掩码失败率、
passthrough 稀释之争）都是「离线掩码」这个选择的产物 —— 论文的方法根本不需要掩码。
**对照臂的价值恰恰在于此**：它量的是"硬掩码 + 离线落盘"这条工程路线的实际收益。

| | 位置 | 机制 |
|---|---|---|
| **主臂** | `workplace/saga/` | 在线 SaGA。用当前策略自己的显著度做逐像素混合，训练时算，不落盘。 |
| **对照臂** | 本目录 | 离线硬掩码。前景（机械臂/夹爪/物体）逐位保留，背景整片替换。 |

**两臂共用同一个 OOD 背景池**：本目录的 `backgrounds/solid8`（8 张纯色）。
`workplace/saga/CLAUDE.md` 里也记了这一条。这样臂间差异只来自增强机制本身，
不来自背景分布。

### 红线在本目录与主臂的差别

上面那四条安全红线里，前三条在本目录与 `workplace/saga/` 完全一致。
**第四条「不碰 `src/lerobot/`」只在主臂被放松** —— 因为在线方法必须接进训练循环。
本目录的离线管线**不需要**改 `src/`，那条红线在这里**依然有效**。

主臂那处放开是有约束的：只改了两个文件（`configs/train.py`、`scripts/lerobot_train.py`），
各留了 `.bak-20260922-2100` 备份，回退命令写在 `workplace/saga/CLAUDE.md` 里。
方法实现本身在 `workplace/saga/`，`src/` 里只有装配。

## 掩码链在 2026-09-22 23:00 被**主动中断**（可续，没丢东西）

`/tmp/chain_fg2.sh` 那条链跑到**第 2 阶段（重试轮）中途**被停：

| 阶段 | 状态 |
|---|---|
| 1. SAM2 全量机械臂掩码 | 完成 |
| 2. `build_robot_mask_cache.py --episodes all --skip-existing --retries 3` | **中断**（当时 `outputs/robot_mask` 有 **85/100** 个 `.npy`） |
| 3. `build_foreground_mask.py --episodes all --skip-existing` | **从未开始** |

**为什么停**：它在 GPU0 上满载，把机箱烤到 88 °C，导致同时在 GPU1 上跑的 SaGA 训练
被热降频到 495 MHz、步时 460 → 820 ms。停下后 GPU0 降到 30 °C、SaGA 恢复到 ~505 ms/步。
详情见 `workplace/saga/CLAUDE.md` 的「显卡热降频」一节。

**续跑**：所有阶段都是 `--skip-existing`，已完成的不重做。直接重跑整条链即可
（不用 `nohup` 也行，但建议后台）：

```bash
cd /home/zjq/lerobot
PY=/home/zjq/miniconda3/envs/lerobot/bin/python
S=workplace/background_generalization/scripts
$PY -u $S/build_robot_mask_cache.py --episodes all --skip-existing --retries 3
$PY -u $S/build_foreground_mask.py  --episodes all --skip-existing
```

**建议等 SaGA 训练跑完再续**（约 14 h，见 `workplace/saga/CLAUDE.md`），
否则两张卡一起满载会再次热降频、两边都慢。若一定要并行，先确认频率没被压：

```bash
nvidia-smi --query-gpu=index,clocks.current.sm,clocks_event_reasons.sw_thermal_slowdown --format=csv
```
