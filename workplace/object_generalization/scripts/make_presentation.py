"""Generate the object-color-augmentation project presentation (PPTX).

Covers: research question, task/data, the label-preserving augmentation, hard
constraints, the tracker evolution (HSV -> multiview -> SAM2), the SAM2 drift
safety fix, the SAM2-vs-lightweight comparison, the hybrid mask, dataset
generation, verification results, visualizations, and next steps.

Output: outputs/object_color_augmentation.pptx
"""

from __future__ import annotations

import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(WORKPLACE, "object_generalization", "outputs")
IMG_DIR = os.path.join(OUT_DIR, "generated_check")
OUT_PPTX = os.path.join(OUT_DIR, "object_color_augmentation.pptx")

# palette
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2E, 0x6E, 0xC8)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
GRAY = RGBColor(0x40, 0x40, 0x40)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
RED = RGBColor(0xC0, 0x39, 0x2B)

FONT = "Microsoft YaHei"


def _set_font(run, size=18, bold=False, color=GRAY, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", name)


def _add_bar(slide, text, sub=None):
    """Top navy bar with title + optional subtitle, returns the slide."""
    bar = slide.shapes.add_shape(1, 0, 0, prs_width, Inches(1.15))  # 1 = rectangle
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.5)
    tf.margin_top = Inches(0.12)
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    _set_font(r, size=26, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        _set_font(r2, size=13, bold=False, color=RGBColor(0xCF, 0xDD, 0xEE))
    # accent underline
    line = slide.shapes.add_shape(1, 0, Inches(1.15), prs_width, Pt(3))
    line.fill.solid()
    line.fill.fore_color.rgb = GREEN
    line.line.fill.background()


def _title_slide(prs, title, subtitle, footer):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.shapes.add_shape(1, 0, 0, prs_width, prs_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    # accent stripe
    stripe = slide.shapes.add_shape(1, 0, Inches(4.1), prs_width, Inches(0.12))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = GREEN
    stripe.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(2.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    _set_font(r, size=40, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

    tb2 = slide.shapes.add_textbox(Inches(0.9), Inches(4.35), Inches(11.5), Inches(1.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    r2 = p2.add_run()
    r2.text = subtitle
    _set_font(r2, size=20, bold=False, color=RGBColor(0xCF, 0xDD, 0xEE))

    tb3 = slide.shapes.add_textbox(Inches(0.9), Inches(6.6), Inches(11.5), Inches(0.6))
    tf3 = tb3.text_frame
    p3 = tf3.paragraphs[0]
    r3 = p3.add_run()
    r3.text = footer
    _set_font(r3, size=14, bold=False, color=RGBColor(0x8F, 0xA8, 0xC8))
    return slide


def _bullets_slide(prs, title, bullets, sub=None, width=12.3, top=1.5, height=5.6):
    """bullets: list of (text, level, bold, color) or plain str."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bar(slide, title, sub)
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in bullets:
        if isinstance(item, str):
            item = (item, 0, False, GRAY)
        text, lvl, bold, color = item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(8)
        prefix = "▪ " if lvl == 0 else "– "
        r = p.add_run()
        r.text = prefix + text
        _set_font(r, size=17 if lvl == 0 else 15, bold=bold, color=color)
    return slide


def _table_slide(prs, title, header, rows, note, col_widths=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bar(slide, title)
    n_rows = len(rows) + 1
    n_cols = len(header)
    total_w = 10.5
    col_w = col_widths or [total_w / n_cols] * n_cols
    left = Inches((13.333 - total_w) / 2)
    top = Inches(1.6)
    height = Inches(0.42 * n_rows)
    tbl = slide.shapes.add_table(n_rows, n_cols, left, top, Inches(total_w), height).table
    for j, h in enumerate(header):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                _set_font(r, size=15, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    _set_font(r, size=14, bold=False, color=GRAY)
    if note:
        tb = slide.shapes.add_textbox(Inches(0.6), top + height + Inches(0.25), Inches(12.1), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = note
        _set_font(r, size=14, bold=False, color=NAVY)
    return slide


def _image_grid_slide(prs, title, images, sub=None):
    """images: list of (path, caption). 2x2 grid."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_bar(slide, title, sub)
    positions = [
        (Inches(0.5), Inches(1.5)),
        (Inches(6.9), Inches(1.5)),
        (Inches(0.5), Inches(4.15)),
        (Inches(6.9), Inches(4.15)),
    ]
    W, H = Inches(5.9), Inches(2.21)  # 1280x480 -> 2.67:1
    for (path, caption), (x, y) in zip(images, positions):
        if path and os.path.exists(path):
            slide.shapes.add_picture(path, x, y, width=W, height=H)
        tb = slide.shapes.add_textbox(x, y + H - Inches(0.05), W, Inches(0.3))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = caption
        _set_font(r, size=11, bold=False, color=NAVY)
    return slide


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    global prs_width, prs_height
    prs_width, prs_height = prs.slide_width, prs.slide_height

    # 1 title
    _title_slide(
        prs,
        "目标颜色泛化：蓝方块 → 绿方块增强数据集",
        "Object Color Augmentation — a label-preserving visual augmentation for ACT",
        "SO101 真机 · LeRobot · ACT 策略    |    2026-09",
    )

    # 2 research question
    _bullets_slide(
        prs,
        "研究问题与动机",
        [
            ("问题：把抓取目标方块从蓝色换成绿色，vanilla ACT 的成功率会掉多少？", 0, True, NAVY),
            ("假设：用「蓝 → 绿」增强的演示数据训练，能否减小这个掉点？", 0, True, NAVY),
            ("核心：目标物体颜色是标签无关的视觉属性 —— 换颜色不应改变任务语义。", 0, False, GRAY),
            ("方法：构造标签保持的增强数据集，做 vanilla vs augmented 对照训练。", 0, False, GRAY),
        ],
    )

    # 3 task & data
    _bullets_slide(
        prs,
        "任务与数据",
        [
            ("任务：Pick up blue block and put it in trash can.", 0, True, GRAY),
            ("数据：newdata（SO101 真机采集）", 0, True, NAVY),
            ("全量 50 集 / 25538 帧；本次实验用 0–2 集 / 1684 帧。", 1, False, GRAY),
            ("双相机：up（俯视）+ front（正面），均为 (3, 480, 640)，fps 30。", 1, False, GRAY),
            ("ACT 同时使用两路相机作为视觉特征。", 1, False, GRAY),
            ("状态 / 动作：6 维 —— 5 关节（shoulder_pan / lift, elbow_flex, wrist_flex / roll）+ 夹爪开度。", 0, False, GRAY),
        ],
    )

    # 4 augmentation definition
    _bullets_slide(
        prs,
        "增强定义：标签保持的视觉增强",
        [
            ("形式：(T_color(I), s, a) —— 只对图像做颜色变换，状态 s 与动作 a 标签完全不变。", 0, True, NAVY),
            ("变换：HSV 空间内，仅旋转 mask 内像素的色相 H（蓝 H=120 → 绿 H=60），保留 S / V。", 0, False, GRAY),
            ("保留：位置、形状、大小、阴影、3D 结构、机器人状态、动作标签。", 0, False, GRAY),
            ("保证：mask 外像素逐 bit 一致（recolor 只在 mask 内生效）。", 0, False, GRAY),
        ],
    )

    # 5 constraints
    _bullets_slide(
        prs,
        "硬约束",
        [
            ("不修改 src/lerobot 任何原始文件（新增独立模块 > Wrapper > Subclass > 复制副本 > 改原文件〔禁止〕）。", 0, True, RED),
            ("不修改 ACT / backbone / transformer / loss / optimizer。", 0, True, RED),
            ("宁可漏增强，不要错增强 —— 绝不重染机械臂 / 夹爪 / 红色容器 / 桌面 / 背景。", 0, True, RED),
            ("state / action 逐 bit 一致（torch.equal 断言）。", 0, True, RED),
        ],
    )

    # 6 tracker evolution
    _bullets_slide(
        prs,
        "目标跟踪：从传统 CV 到 SAM2",
        [
            ("① HSV 阈值 → 单视图（up）跟踪器：frame-0 初始化 + 逐帧搜索。", 0, False, GRAY),
            ("② 机器人状态先验（无 FK，6 维状态 → 相位）+ 正面几何门（extent / area / aspect 区分方块 vs 大块背景蓝色）。", 0, False, GRAY),
            ("③ 多视图跟踪器：up 管身份，front 用 TargetTemplate 做局部关联。", 0, False, GRAY),
            ("④ SAM2 视频实例跟踪（Segment Anything 2）—— 本方案最终采用。", 0, True, GREEN),
        ],
    )

    # 7 lightweight tracker explained
    _bullets_slide(
        prs,
        "什么是轻量跟踪器？",
        [
            ("本质：经典 CV 跟踪器（无学习模型）—— 对比 SAM2（学习式视频实例分割）。", 0, True, NAVY),
            ("① 初始化（frame 0）：HSV 蓝色阈值 → 连通域候选 → 几何门 → 打分锁定目标。", 0, False, GRAY),
            ("几何门：area / aspect / extent（小、方、紧凑），天然排除大块背景蓝色。", 1, False, GRAY),
            ("② 逐帧更新：在上一帧 bbox 附近搜索，取最近、面积/颜色相近的候选；遮挡时扩大搜索区。", 0, False, GRAY),
            ("连续 N 帧无候选 → 终止丢失（宁丢不换目标）。", 1, False, GRAY),
            ("③ front 相机：几何门（area / aspect / extent）区分方块 vs 大块背景蓝；RobotStatePrior（无 FK，6 维状态 → 相位）只用于相位标注/调试，不参与匹配。", 0, False, GRAY),
            ("④ 多视图：up 管身份，front 用 TargetTemplate（形状+面积+HSV 嵌入，EMA 更新）做局部关联。", 0, False, GRAY),
            ("up 退化（夹爪遮挡）时模板冻结，front 继续跟踪过抓取/搬运。", 1, False, GRAY),
            ("特点：无模型、可解释、遮挡后能重新找回；但 front 受背景蓝色干扰。", 0, True, GREEN),
        ],
    )

    # 8 SAM2 drift + fix
    _bullets_slide(
        prs,
        "SAM2 的漂移问题与安全修复",
        [
            ("问题：纯 SAM2 mask 会漂移 —— 抓取/放置时漂到夹爪 / 红色容器，却仍持续输出 mask。", 0, True, RED),
            ("证据：蓝色占比抓取 f220/f260 掉到 0.05–0.10，放置 f420 掉到 0.00。", 1, False, GRAY),
            ("风险：recolor 是 mask 内无条件色相旋转 → 漂移 mask 会把红容器/桌面染成蓝色（错增强）。", 0, True, RED),
            ("修复：SAM2 实例 mask ∩ 蓝色阈值 mask —— mask 天然只含蓝色像素。", 0, True, GREEN),
            ("效果：漂移到非蓝区域 → 空 mask → 不增强（宁可漏增强，不要错增强）。", 1, False, GRAY),
        ],
    )

    # 9 comparison table
    _table_slide(
        prs,
        "SAM2 vs 轻量跟踪器（覆盖率，互补）",
        ["相位", "相机", "SAM2", "轻量", "IoU"],
        [
            ["approaching", "front", "100%", "83.7%", "1.00"],
            ["moving", "front", "88.3%", "73.1%", "1.00"],
            ["placing", "front", "12.8%", "4.3%", "1.00"],
            ["home", "front", "65%", "26%", "1.00"],
            ["home", "up", "84.8%", "100%", "1.00"],
            ["retracting", "up", "69.1%", "97.9%", "1.00"],
        ],
        "front（抓取/搬运关键视角，有背景蓝色干扰）：SAM2 学习式传播大幅胜出。up（单一蓝色物体、无歧义）：轻量跟踪器在夹爪遮挡后能重新找回，而 SAM2 丢了就不再找回 —— 例：ep2 up 相机 SAM2 仅 38%，轻量 78%。双方都跟踪处 IoU = 1.00。",
        col_widths=[1.7, 1.3, 1.4, 1.4, 1.2],
    )

    # 10 hybrid
    _bullets_slide(
        prs,
        "最终方案：SAM2 ∪ 轻量（hybrid）",
        [
            ("互补分工：front（有背景蓝干扰）→ SAM2；up（无歧义、遮挡后可找回）→ 轻量。", 0, True, NAVY),
            ("并集 = 两边各自的最优覆盖，取长补短。", 0, False, GRAY),
            ("安全性：两者 mask 都蓝受限 + IoU = 1.00（双方都跟踪处位置完全一致）→ 并集不会引入错误区域。", 0, False, GRAY),
            ("缓存：outputs/cache_hybrid/（每集 up_mask.npy + front_mask.npy）。", 0, False, GRAY),
        ],
    )

    # 10 generation
    _bullets_slide(
        prs,
        "数据集生成（newdata_green）",
        [
            ("官方写入路径：LeRobotDataset.create → add_frame → save_episode。", 0, True, NAVY),
            ("同 AV1 codec、fps 30、同 feature schema。", 0, False, GRAY),
            ("逐帧重染后写回视频；state / action / timestamp 直接拷贝。", 0, False, GRAY),
            ("输出：/home/zjq/lerobot/dataset/newdata_green（3 集 / 1684 帧）。", 0, False, GRAY),
        ],
    )

    # 11 verification
    _bullets_slide(
        prs,
        "验证结果",
        [
            ("state / action：1684 帧逐 bit 一致，0 不一致。", 0, True, GREEN),
            ("frame / episode / timestamp 对齐：0 不一致。", 0, True, GREEN),
            ("方块色相（mask 内）：up 121.6 → 62.5，front 114.7 → 54.9（蓝 → 绿）。", 0, False, GRAY),
            ("mask 外漂移：均值 ~1.5/255（AV1 有损重编码，非 recolor 逻辑）；2524 相机帧被重染。", 0, False, GRAY),
        ],
    )

    # 12 visualizations
    imgs = [
        (os.path.join(IMG_DIR, "ep0_f0_up.png"), "ep0 f0 · up（home：方块在桌面）"),
        (os.path.join(IMG_DIR, "ep0_f180_front.png"), "ep0 f180 · front（接近/抓取）"),
        (os.path.join(IMG_DIR, "ep1_f260_front.png"), "ep1 f260 · front（抓取/搬运）"),
        (os.path.join(IMG_DIR, "ep2_f120_up.png"), "ep2 f120 · up（接近）"),
    ]
    _image_grid_slide(
        prs,
        "可视化：base vs generated",
        imgs,
        sub="左 = base（蓝方块），右 = generated（绿方块），绿色轮廓 = 目标 mask。仅方块变绿，机械臂/容器/桌面/背景不变。",
    )

    # 13 next steps
    _bullets_slide(
        prs,
        "下一步",
        [
            ("训练 ACT 对照：newdata（蓝，基线） vs newdata_green（绿，增强）。", 0, True, NAVY),
            ("需要更大数据量时，扩展到全 50 集（给 3–49 集补建 mask，约 4–5 小时 SAM2）。", 0, False, GRAY),
        ],
    )

    prs.save(OUT_PPTX)
    print(f"saved {OUT_PPTX} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
