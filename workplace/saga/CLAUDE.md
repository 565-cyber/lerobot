# CLAUDE.md — workplace/saga

真 RoboSaGA（在线、显著度引导的背景增强）的实现在这里。

论文与官方代码在上层目录：`workplace/background_generalization/zhuang25b.pdf`
与 `RoboSaGA-master.zip`。**方法细节全部读自原文与源码**，这份文件里凡是数字都
能在源码里指到具体行。

## 目标

**把论文的方法当数据增强，同时施加到两个相机上。**

不是拿 up 和 front 做对比 —— 相机之间的差异不是这里要交付的东西。

论文的 Algorithm 1 本来就是**逐视觉模态**跑的：参考实现
`saliency_guided_augmentation.py:95` 是
`for obs_key in obs_meta["visual_modalities"]`，每个 key 一份 extractor、一份缓冲区。
LeRobot 的 ACT 结构一致 —— 同一个 backbone 在循环里逐相机前向
（`modeling_act.py:472`：`for img in batch[OBS_IMAGES]: cam_features = self.backbone(img)`）。
所以「每个相机各自算显著度、各自更新缓冲区、各自被增强」是**参考实现本来的结构**，
不是本移植发明的。两个相机共用一组 backbone 权重，但显著度是按相机分别求的。

## 定位：主臂 vs 对照臂

| | 在哪 | 是什么 |
|---|---|---|
| **主臂** | 本目录 | 真 SaGA。在线用**当前策略自己的显著度**做逐像素混合，训练时算，不落盘。 |
| **对照臂** | `../background_generalization/` | 离线**硬掩码**背景替换。不是 RoboSaGA，是刻意留的对照/基线。 |

两条臂**共用同一个 OOD 背景池**（`../background_generalization/backgrounds/solid8`，
8 张纯色），这是 2026-09-22 定的，为了让臂间差异只来自增强机制本身。

## 目录

| 文件 | 作用 |
|---|---|
| `fullgrad.py` | FullGrad 显著度。逐行对 `robosaga/fullgrad.py` + `tensor_extractors.py`，五处 ACT 适配写在文件头。 |
| `buffer.py` | 全局显著度缓冲区，uint8 单通道，按全局帧 id 索引。 |
| `backgrounds.py` | OOD 背景源，ImageNet 归一化后输出。 |
| `saga.py` | `RoboSaGA` 驱动，照论文 Algorithm 1。 |
| `saliency_probe.py` | 出图看两个相机各自的显著图 —— 确认增强在正常工作，不是相机对比。 |
| `scripts/verify_saga.py` | 31 项验证，跑在 CPU 上。 |
| `scripts/preview_augmentation.py` | 把增强实际作用到真实数据上，两个相机各出一组「原图 / 增强后 / 权重 / 被替换比例」。 |
| `scripts/preview_swap_saga.py` | 把增强作用到**物体已被换掉**的图上（`--source distractor/up_fit/wrist_fit`）→ `figures/swap_saga/*_saga.png`。关键：替换图要**塞回 batch** 再算显著度，不能拿原帧的显著度去叠 —— 见文件头。 |
| `scripts/draw_algorithm.py` | **论文式算法图** → `saga_algorithm.{png,svg,pdf}`：左面板是照原文 Algorithm 1 体例的伪代码，右面板是同一套逻辑的流程示意。只有算法逻辑，无工程细节。 |
| `scripts/draw_framework.py` | 工程实现图 → `saga_framework.{png,svg}`：标注注入点、归一化空间、步时等实现细节。与上面那张互补。 |
| `scripts/compare_saliency.py` | 基线 vs SaGA 训练后的显著图**逐像素对比** → `saliency_diff_{up,front}.png`。回答「两张图怎么看着一样」。 |
| `scripts/edge_vs_saliency.py` | 替换率 `1−s` vs **局部边缘强度**（Sobel，分位数分箱，**不需要手工掩码**）→ `edge_vs_saliency.png`。答「SaGA 到底保护了什么」。 |
| `scripts/two_views_independent_bg.py` | 同一时刻两个相机的原图/增强图并排（按「两边同时看得见方块」选帧）→ `two_views_independent_bg.png`。 |
| `scripts/cube_visible_in_up.py` | **更正**：黄方块在 up 里看得见 → `cube_visible_in_up.png`。含分组对照（front 夹着 vs 没夹）。 |
| `FINDINGS_where_replaced.md` | 结论：**约 75% 替换量由 `normalize_smaps`+`s²` 定死**；全域 56–83%、无一档被保护；**两个相机共享被夹住的黄方块**。含两批**撤回**的数字。 |
| `SWAP_SAGA.md` | **换物体 + SaGA 的自包含交接文档**：L0/L1/L2、位姿从哪来（量 vs 拟合）、三张交付图、全部实测数字、复现命令、踩过的坑。开新对话先读这个。 |
| `figures/` | **出图的落盘位置**。脚本默认就写这里，别写 `/tmp` —— 这些图是交付物，重启就没了。 |

### 出图的三条约定（都是踩过的）

1. **色标用 `inferno`，不用 `jet`。** 逐像素权重是「量级」数据，彩虹色标会凭空造出
   数据里没有的亮暗分界（读者会把某条色带误当阈值），对红绿色盲也不友好。
   `inferno` 感知均匀、亮度单调、色盲安全，且就是「热度」语义。
2. **色标要有图例。** 叠加图同时用颜色和不透明度编码，没图例读者无法定量。
   `labelpad` 只能建完 colorbar 之后 `cb.set_label(..., labelpad=10)` 设 ——
   直接传给 `fig.colorbar` 会 `TypeError`。
3. **`fig.suptitle` 会盖掉第一行的子图标题**（第一版探针图顶上那团糊字就是它），
   用 `tight_layout(rect=[0, 0, w, 0.96])` 给它留出顶部空间。

`saliency_probe.py --tag _baseline` / `--tag _saga` 可对两个 checkpoint 分别出图；
不加 `--tag` 后一次会覆盖前一次。`preview_augmentation.py --start/--stride`
可跨时段取样（SaGA 对帧连续性没有要求，缓冲区按全局帧 id 索引）。

## 先跑验证

```bash
python workplace/saga/scripts/verify_saga.py      # 纯 CPU，约 1-2 min
```

31 项全过再谈训练。里面最要紧的两条：

- **完备性测试**（原作者自带，`fullgrad.py` 的 `checkCompleteness`）：
  `Σ(input_grad·x) + Σ(bias_grads) == model(x).sum()`。这是移植是否忠实的**最强单项检查**。
  小卷积网 + 真 ACT backbone 各跑一遍 —— 后者是 `FrozenBatchNorm2d` 那处适配**唯一**的裁判。
- **可证伪性**：`aug_ratio=0` 时整个 batch 逐位不变。这是防「静默改了不该改的东西」的守门测试。

## 跑训练

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.path=<ckpt> --dataset.repo_id=record-test10 --dataset.root=dataset/newdata1 \
  --saga.warmup_epochs=0 \
  ...
```

不加任何 `--saga.*` 就是完全关闭，训练路径与上游 LeRobot 逐位一致。

**`--saga.warmup_epochs` 默认 10**（论文：前 10 个 epoch 完全不增强）。做 30 步的
短程试跑时务必显式设成 0，否则整个试跑都在 warmup 里、增强根本不发生 ——
现象是「跑了但看不出任何区别」。

另两个跑一次就会撞上的坑：

- **本机连不上 huggingface.co**，训练结束时的 `push_model_to_hub` 会抛
  `ConnectTimeout`。加 `--policy.push_to_hub=false`。（它是最后一步，训练本身已经
  跑完了、checkpoint 也存好了，所以只丢日志不丢结果 —— 但退出码不是 0，别被误导。）
- `--output_dir` 指向**已存在**的目录会直接 `FileExistsError`（除非 `--resume`）。
  重跑前先 `rm -rf`。

## 与 `src/lerobot/` 的接口（唯一的侵入点）

改动只有两个文件、共 +87 行，且方法实现**全在本目录**（`src/` 里只留装配）：

1. `src/lerobot/configs/train.py` —— `SagaConfig`（本文件内定义）+ `TrainPipelineConfig.saga: SagaConfig | None = None`。
   放在这里是为了让设置随 `train_config.json` 存进 checkpoint，满足溯源要求。
2. `src/lerobot/scripts/lerobot_train.py` —— `_build_saga()` 装配函数，以及训练循环里
   `batch = preprocessor(batch)` **之后**的一行 `if saga is not None: batch = saga(batch, step)`。

注入位置有两处讲究，都在循环那行的注释里：**必须在 preprocessor 之后**（显著度与混合
都在 ImageNet 归一化空间里做），**必须在 autocast 之外**（fp32 算显著度）。

**分辨率与缓冲区容量不写进配置**，由数据集元信息推出来（`_build_saga` 里：
`dataset.meta.features[...]["shape"][:2]` 和 `total_frames + 1`），免得配置里留一份
会和实际数据漂移的副本。

### 备份与回退

改动前已备份（2026-09-22 21:00）。回退：

```bash
cp src/lerobot/scripts/lerobot_train.py.bak-20260922-2100 src/lerobot/scripts/lerobot_train.py
cp src/lerobot/configs/train.py.bak-20260922-2100          src/lerobot/configs/train.py
```

## 实测开销（2026-09-22，单卡 2080Ti，batch_size=16，480×640）

| | ms/步 | 来源 |
|---|---|---|
| 基线 | 400–420 | 200 步对照跑（`eval_freq=0`） |
| 开 SaGA | 440–460 | 同上 |
| **增量** | **约 +45（1.10×）** | |
| 基线（**权威值**） | **452** | 主线 run `act_record_test10` 自己的 checkpoint 时间戳 |

**以 452 ms/步为准。** 那个数是从 `outputs/train/act_record_test10/checkpoints/` 里
5 个 checkpoint 的 mtime 间隔算的：每 20000 步稳定 2.50–2.52 h，四段全是 450–454。
它是**同配置、长程、真实**的运行，比 200 步的短跑可信。

对着 452 算训练预算：**100k 步 ≈ 12.6 h**；开 SaGA 约 **12.8 h**。

⚠️ **量步时之前先确认日志的 `log_freq`。** 短跑用的是 `log_freq=50`，长跑是 `200`；
按错的除数换算会差 4 倍（我自己就栽过一次，把 400 ms 算成了 100 ms，进而得出
「基线 105 ms/步」这个不可能的结论）。判据：看相邻两行的 `step:` 增量。

不是从 `updt_s` 读的 —— **`updt_s` 里根本没有 SaGA 的时间**：它在 `update_policy`
内部计时，而 SaGA 在 `preprocessor` 之后、`update_policy` 之前调用，所以那段时间
既不计入 `updt_s` 也不计入 `data_s`。要量它只能用墙钟（或自己加表）。

比论文自陈的 1.5× 便宜，原因：论文那个数是在 84×84 上量的，训练步本身极短，
显著度那一遍就显大；我们 480×640、batch 16 的一步本来就要 400 ms。
另外 β=0.1 意味着每步只对 **1 张**图算 FullGrad（不是整批），这是省钱的主因。

### ⚠️ 显卡热降频能把步时翻倍 —— 量之前先确认频率

2026-09-22 撞到一次：SaGA 训练跑在 GPU1 上，同时对照臂的掩码任务在 GPU0 上满载，
机箱被烤到 88°C，**GPU1 被热降频到 495 MHz**（上限 2175），步时从 ~460 变成 **820 ms**
（ETA 12.8 h → 22.8 h）。停下掩码任务后回到 ~505 ms/步。

判据（别只看 `nvidia-smi` 的利用率 —— 热降频时利用率照样显示 100%）：

```bash
nvidia-smi -i N --query-gpu=clocks.current.sm,temperature.gpu,\
clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.sw_power_cap --format=csv
```

`sw_thermal_slowdown: Active` + SM 频率远低于上限 + 功耗明显低于 TDP（如 130 W / 260 W）
= 在空转等降温，不是算力不够。**先看频率再报步时，否则会把环境问题记成方法的开销。**

## ⚠️ batch_size 有硬下限：≥ 1/β = 10

`n_updates = int(batch_size × β)` 是**截断**取整。β=0.1 时：

| batch | int(n×β) | 每步更新几张 | 结果 |
|---|---|---|---|
| 8 | 0 | 0 | **静默退化** |
| 9 | 0 | 0 | **静默退化** |
| 10 / 16 | 1 | 1 | 正常 |
| 32 | 3 | 3 | 正常 |

退化时：缓冲区永远停在初值 1.0 → `s ≡ clip(1,0,λ)² = 0.64` 是个**常数** →
实际跑的是 `random_overlay` 加固定权重，**显著度引导完全没生效**。
不报错、不崩、loss 照样降 —— 跑的却是另一个方法，还挂着 SaGA 的名。

所以 `saga.py` 在第一次遇到 0 更新时会打一行 `[saga][!!]` 警告；
`verify_saga.py` 的第 9 组测试把这个门槛钉死了（直接数缓冲区被改动的格子数：
batch 8/9 → 0，10/16 → 19200，32 → 57600）。

**本项目实际训练用 batch_size=16**，是真的在跑 SaGA。

## 基线 vs SaGA 的显著度：几乎一样（2026-09-23 实测，别当成 bug）

把两个 checkpoint 的显著图并排出图，肉眼看不出区别。**这不是加载错了** ——
量化下来（`scripts/compare_saliency.py`，8 个 (相机,帧) 组合）：

| | 值 |
|---|---|
| Pearson r（逐像素） | 均值 **0.9817**，最小 0.9700 |
| 平均 \|Δ\| | **0.0463**（s 的值域是 0–1，即约 4.6%） |
| 最大 \|Δ\| | 0.2458 |
| 集中度变化 | 大多 ±0.01–0.04，**方向一致地略微变高** |

两个 checkpoint 的权重确实是不同的（234 个张量里 153 个有差异，backbone 最大权重差
0.023，全模型 0.087），所以是**显著图本身真的几乎没变**，不是加载了同一份权重。

原因：显著度取自**共享的 backbone**，而 ACT 的微调对 backbone 的改动本来就轻微；
且 FullGrad 在低层特征上算，显著度由**图像的边缘/对比度结构**主导 —— Δ 图正是这个形态
（沿轮廓的高频纹理），不是随机噪声，说明变化是系统性的、只是幅度小。

**这条结论只能说明「编码器的显著度没怎么变」，不等于 SaGA 没效果。**
SaGA 是拿这个显著度去做**数据增强**、改变**策略**学到的东西；策略有没有变好，
只能靠真机评测回答。把那两张几乎一样的图当成「SaGA 无效」的证据是**错的读法**。

值得注意的一处**定性**变化（front，ep0 第 300 帧）：集中度 0.30 → 0.34，
且显著度明显收拢到夹爪 + 黄色方块上。方向是对的，但样本太少，不能当结论。

## 缓冲区刷新速度（算训练预算时要看）

每步只更新 1 帧 → 把 21999 帧的缓冲区完整碰一遍要约 **21999 步 ≈ 16 个 epoch**
（batch 16 时 1375 步/epoch）。跑到第 10 个 epoch 才刚开始增强，那时缓冲区里
只有约 13750/21999 ≈ 62% 的帧有真实显著图，其余仍是初值 1.0（即常数权重）。
100k 步下每帧平均被更新约 4.5 次。

这是论文 β=0.1 的固有性质，不是 bug；但如果想要缓冲区更快收敛，
调 `--saga.update_ratio`（代价是每步多算几张 FullGrad）。

## 已知偏离论文的地方（都要记着，别当成等价）

1. **OOD 池是 8 张纯色**，不是论文的 5000 MSCOCO + 1000 合成图。用户 2026-09-22 定的，
   两臂统一。→ 结果**不能**和论文的 0.14 / 0.05 这类数字并列。
2. **不加 Random Crop**。论文里它是所有方法的默认基底，我们不用。
3. **缓冲区分辨率 `buffer_shape=(120,160)`**，即 480×640 的 1/16 像素数。
   论文的缓冲区是**与输入同分辨率**（84×84 输入存 84×84）。这是为了显存做的妥协
   （全分辨率 21999 帧要 6.6 GB/进程，2080Ti 放不下）。论文自己也写了缓冲区是
   "at a reduced resolution"，但我们这一步降得比它狠，**影响未实测**。可配。
4. **旋转不实现**（`backgrounds.py`）：纯色图旋转是恒等变换，写它是纯仪式。
   将来若换成真实照片池，这行必须补上。
5. **亮度抖动的加/乘语义没核对上**：参考实现用 kornia 的 `adjust_brightness`，本机没装
   kornia，取的是**加性**。±0.1 在两种语义下都只是很小的颜色偏移。
6. **多卡下缓冲区是按进程分的**：`accelerator.prepare` 把 dataloader 分片，每个进程一份
   自己的缓冲区。论文是单进程。等价于全局缓冲的分片，但「最久未更新」的选择变成进程内的
   —— 标称 β=10%，实际是每卡 10%。两卡训练时记得这一点。

## 源码里论文正文没写的东西（照源码，不照正文）

- **`s = s²`**：取回并 `clip(·, λ)` **之后**还有一次平方（`saliency_guided_augmentation.py:137`）。
  平方让 s 更小 → 更多 OOD 覆盖。所以权重上限是 **λ² = 0.64**，不是 0.8。
- **缓冲区存的是归一化后、未裁剪的图**，`clip(·, λ)` 在**取回时**才做
  （`extract_and_update` vs `retrieve_smaps_from_buffer`）。
- **FullGrad 的第二项是 `grad ⊙ bias`，不是 `grad ⊙ activation`**（`fullgrad.py:59`）。

## 四个踩过的坑（都是静默失败，值得记牢）

1. **`register_full_backward_hook` 在 torchvision resnet18 上用不了。**
   它把模块输出包成自定义 Function 的输出（输入的 view），而 residual block 里到处是
   原地操作（`relu(inplace=True)`、`out += identity`），autograd 直接报
   "BackwardHookFunctionBackward is a view and is being modified inplace"。
2. **换成 `autograd.grad(f, 各层输出)` 会静默算错。** 原地 relu 会改写被捕获张量的
   `grad_fn`，于是 autograd 把它当 relu 的**输出**求导，返回的是过完 relu 的梯度。
   真 ACT backbone 上表现为完备性差 6%（`raw=527.96` vs `fullgrad=495.15`），
   **不报任何异常**。
   最终办法：forward hook 里拿输出张量、立刻 `register_hook`（张量 hook 不受原地操作影响）。
   完整推理写在 `fullgrad.py` 文件头。
3. **`IntermediateLayerGetter` 在 `layer4` 就返回，`fc` 从不执行**，但参考实现的
   `get_biases()` 遍历 `model.modules()` 会把它收进去 → 两份列表长度不等 → 整条配对
   **静默错位**。本实现改成「谁前向跑了就记谁」，天然不可能错位。
4. **加载数据集必须走 `make_dataset`，不能直接 `LeRobotDataset(...)`。**
   `make_dataset` 在 `use_imagenet_stats=True`（默认）下会把 `meta.stats` 里**相机**的
   mean/std **替换成 ImageNet 常数**（`datasets/factory.py:126`）。直接构造
   `LeRobotDataset` 拿到的是数据集自己的统计量 —— 实测 `up` 是
   `mean 0.742 / std 0.0137`，不是 `0.485 / 0.229`。
   **两者归一化空间差约 17 倍**，于是 `backgrounds.py` 按 ImageNet 归一化出来的 OOD
   背景与图像**不在同一空间**，`x·s + x_O·(1−s)` 的混合结果就是垃圾。
   表现：不报错、不崩，只是图变花（第一次跑 `preview_augmentation.py` 就是这么栽的）。
   `preview_augmentation.py` 里现在有硬断言挡这一条。

## 安全红线（与 `../background_generalization/CLAUDE.md` 一致）

- 不重着色 / 移动机械臂、夹爪、蓝方块、红盒、桌面。
- `state` / `action` / `timestamp` 逐帧 bit-identical。→ 已验证：SaGA 只替换相机图像键，
  其余键原样引用（`verify_saga.py` 有逐位断言）。
- 宁可漏增强，不要错增强。→ `warmup` 与 `aug_ratio=0` 两条守门测试就是这条的看门人。

**在线增强只改训练时的张量，数据集落盘内容一个字节都不动** —— 所以后两条红线在这里
是自动满足的，比离线管线干净。
