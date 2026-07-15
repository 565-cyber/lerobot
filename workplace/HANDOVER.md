# 交接文件：lerobot_record FSM 多任务录制系统

---

## 1. 我正在做什么

基于 lerobot 官方录制脚本 (`lerobot_recordcopy.py`)，开发了一个**多任务 FSM 自动切换录制系统** (`lerobot_record.py`)。机械臂通过 3 个 binary detector (ResNet18) 实时检测场景状态，自动在三组 ACT 策略间切换：

```
State 1 (夹黄块扔垃圾桶) → State 2 (夹蓝块扔垃圾桶) → State 3 (关垃圾桶盖)
```

## 2. 最终目标

1. 多任务 FSM 切换稳定可靠（detector 准确率足够 + blend 平滑过渡）
2. 产出组会汇报材料：理论验证 (C² 连续性) + 真实机械臂实验数据 (p/v/a 曲线)
3. 后续：接入 VLM Agent，实现自然语言 → 任务序列的自动编排

## 3. 已确定的信息和结论

### FSM 切换条件

```python
# lerobot_record.py 第 814-818 行
if current_state == 1 and not d1 and d2:
    target = 2       # 黄块不在 + 蓝块在 → State 2
elif current_state == 2 and not d2 and d3:
    target = 3       # 蓝块不在 + 盖子开 → State 3
```

- 迁移需要连续 5 帧确认 (`FSM_TRANSITION_CONFIRM=5`)
- 切换后有 30 帧冷却期 (`FSM_TRANSITION_COOLDOWN=30`)
- State 3 检测到盖子关闭后先排空 50 帧再保持位置 (`STATE3_DRAIN_FRAMES=50`)

### Quintic Blending 参数（关键）

```python
BLEND_FRAMES = 120         # 过渡总帧数 (~4.0s @30fps)
BLEND_WARMUP_FRAMES = 20   # 预热帧数，跳过 quintic，纯用旧策略
BLEND_METHOD = "quintic"   # 五次多项式
BLEND_MAX_STEP = 0.05      # 每帧单关节最大移动量 (rad)
SG_ENABLED = True           # Savitzky-Golay jerk 平滑
SG_WINDOW = 7
SG_ORDER = 3
```

### 三组 ACT 策略路径

```python
PICK_UP_YELLOW_BLOCK_POLICY_PATH = "...train/act_record_test10/checkpoints/last/pretrained_model"
PICK_UP_BLUE_BLOCK_POLICY_PATH   = "...train/act_record_test8/checkpoints/last/pretrained_model"
CLOSE_TRASH_CAN_POLICY_PATH      = "...train/act_record_test9/checkpoints/last/pretrained_model"
```

## 4. 偏好、限制和要求

- 使用 conda 环境 `lerobot`，激活命令：`source /home/zjq/miniconda3/etc/profile.d/conda.sh && conda activate lerobot`
- 项目根目录：`/home/zjq/lerobot`
- 俯视摄像头 key：`observation.images.up`（detector 使用这个）
- 调用摄像头用 `--camera 1`（俯视，具体编号视机器而定）

## 5. 已尝试过的方法

### A. Detector 精度调试

- 使用 `workplace/extract_binary_detector_dataset.py` 从视频自动提取标注帧
- 使用 `workplace/capture_detector_images.py` 手动补充样本
- 问题：yellow detector 一度偏向全判 positive（recall=1.0, precision=0.63），导致 `not d1` 永远不成立
- 问题：blue detector 同样偏向全判 positive，`not d2` 永远不成立
- 解决：手动补充 negative 样本后重训，yellow 提升至 F1=0.97

### B. Blend 暂停问题

- 问题：切换任务时机器人暂停然后快速移动
- 原因 1：新策略 reset() 后前几帧输出无效 → 加 `BLEND_WARMUP_FRAMES=20`
- 原因 2：quintic 边界条件 `vT=0, aT=0` 强制减速 → warmup 期跳过 quintic，直接沿用旧策略动作
- 原因 3：`BLEND_MAX_STEP=0.015` 太保守 → 改为 0.05

### C. Blend 验证

- 理论验证：`workplace/verify_quintic.py`，用纯数学方程证明 C² 连续
- 真实数据验证：在 `lerobot_record.py` 加了 blend_log 记录，用 `workplace/plot_blend_verification.py` 读取

## 6. 失败过或不要重复的做法

- **不要只靠 `extract_binary_detector_dataset.py` 自动标注**：时间分段 heuristic 不够精确，必须用手动 `capture_detector_images.py` 补充
- **不要只更新一个 detector**：yellow 和 blue 的状态互补，两个都要调好
- **不要用 BLEND_MAX_STEP < 0.05**：太小的钳制值会导致 blend 没走完就结束，产生跳变
- **不要在 warmup 期用 quintic**：`vT=0, aT=0` 边界条件会强制减速

## 7. 关键文件

| 文件 | 作用 |
|------|------|
| `src/lerobot/scripts/lerobot_record.py` | 主录制脚本（含 FSM + quintic blend） |
| `src/lerobot/scripts/lerobot_recordcopy.py` | 原始官方录制脚本（未修改的备份） |
| `src/lerobot/utils/control_utils.py` | 键盘监听（已改右箭头为 2 次切换逻辑） |
| `workplace/binary_detector_model.py` | BinaryDetector 类（ResNet18 + 5 帧投票） |
| `workplace/capture_detector_images.py` | 手动采集 detector 训练数据 |
| `workplace/train_binary_detector.py` | 训练 binary detector |
| `workplace/extract_binary_detector_dataset.py` | 从录好的视频自动提取标注帧 |
| `workplace/verify_quintic.py` | 理论验证：quintic vs linear p/v/a 曲线 |
| `workplace/plot_blend_verification.py` | 从录制数据 + blend_log.json 生成验证图 |
| `workplace/plot_real_trajectory.py` | 从录制数据集画原始轨迹 |
| `workplace/plot_fsm_blend.py` | 从录制数据集 + heuristic 找 blend 区 |
| `workplace/paper_detector_*.pt` | 三个 detector 权重 |
| `outputs/eval/eval_record_planner_3step3/` | 最新 FSM 录制（含 blend_log.json） |

### 关键代码位置（lerobot_record.py）

- `record_loop()` 第 463 行：主录制循环入口
- 第 600-703 行：blend 动作计算（warmup + quintic）
- 第 783-934 行：FSM detector 检测 + 迁移条件判断 + 策略选择
- 第 855-879 行：FSM 迁移触发 + 新策略 reset
- 第 1013-1017 行：blend_log 写入
- 第 1273 行：blend_log 初始化
- 第 1337-1343 行：blend_log 保存为 JSON

## 8. 当前进度

| 模块 | 状态 |
|------|:---:|
| FSM 三阶段切换 | ✅ 可用 |
| Quintic blending | ✅ 可用 |
| 右箭头控制（1次=保存+reset，2次=退出） | ✅ 完成 |
| blend_log.json 日志 | ✅ 完成 |
| Detector yellow | ⚠️ 基本可用，需更多测试 |
| Detector blue | ⚠️ 仍有误判（State 3 时 blue=True(1.00)） |
| Detector lid | ✅ 可用 |
| 理论验证图 (quintic vs linear) | ✅ 已生成 |
| 真实机械臂 blend 验证图 | ✅ eval_record_planner_3step3 已生成 |
| VLM task planning | ❌ 未开始 |

## 9. 还没解决的问题

1. **blue_on_table detector 在 State 3 仍有误判**：蓝块已扔进垃圾桶但仍返回 `blue=True(1.00)`，需要补充更多 negative 样本（蓝块不在桌上的各种场景）
2. **State 3 排空可能不完整**：50 帧 + 前面三个任务时间可能超过 60 秒 episode 限制，可加大 `--dataset.episode_time_s`
3. **detector 用的是 up 摄像头，但 capture_detector_images.py 默认打开 camera 0 (front)**：需要用 `--camera 1` 指定俯视摄像头

## 10. 下一步

1. **继续补充 detector 训练数据**，尤其是 blue negative 样本和 yellow positive/negative 多样化场景
2. **用 blend 验证图组会汇报**——理论图 (`quintic_verification.png`) + 真实数据图 (`eval_record_planner_3step3_blend_*.png`)
3. **VLM task planning**（学长给的任务）：在现有 FSM 上加一层 LLM，自然语言 → 技能序列
4. **VLM 完成判断**：用 VLM 替代 binary detector 做"任务是否完成"判断，不再需要训练 detector

---

## 11. 新 AI 开场提示词

```
我正在进行一个基于 lerobot 的机械臂多任务自动切换录制项目。

**项目路径**：/home/zjq/lerobot
**核心文件**：src/lerobot/scripts/lerobot_record.py（主录制脚本）
**环境**：conda activate lerobot（/home/zjq/miniconda3/envs/lerobot）

**系统架构**：
- 3 个 binary detector (ResNet18) 通过俯视摄像头 (observation.images.up) 检测场景
- 3 组 ACT 策略分别处理：夹黄块扔垃圾桶 → 夹蓝块扔垃圾桶 → 关垃圾桶盖
- FSM 根据 detector 结果自动切换策略
- 策略切换时使用 quintic 五次多项式 blend (C² 连续过渡)
- 数据集录制在 outputs/eval/ 下

**当前主要问题**：
1. blue_on_table detector 误判率偏高（蓝块已拿走但仍返回 True）
2. 需要补充 detector 训练数据（workplace/capture_detector_images.py 手动采集，workplace/train_binary_detector.py 训练）
3. 下一步要接 VLM agent 做 task planning

**关键参数**：
- BLEND_FRAMES=120, BLEND_WARMUP_FRAMES=20, BLEND_MAX_STEP=0.05
- FSM_TRANSITION_CONFIRM=5, FSM_TRANSITION_COOLDOWN=30
- 三组策略路径在 lerobot_record.py 第 351-353 行

**已修改的文件**（diff based on lerobot_recordcopy.py）：
- lerobot_record.py：新增 FSM + quintic blend + blend_log
- control_utils.py：右箭头改为 2 次按压逻辑 (1st=保存+reset, 2nd=退出)

请先阅读 lerobot_record.py 了解整体结构，特别是 record_loop 函数。
```
