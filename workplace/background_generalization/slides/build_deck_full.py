"""Build the FULL (22-slide) Stage 2B background-generalization deck.

Usage (from the repo root):
    python workplace/background_generalization/slides/build_deck.py

Writes slides/background_generalization_full.pptx next to this file. This is the
long version; the short deck that gets presented is slides/build_deck.py. Every number in
it is either read from the plate cache at build time or quoted from the
project's README.md / source docstrings.
"""
from __future__ import annotations

import glob
import json
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
BG = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figs")
PLATE = os.path.join(BG, "outputs", "plate")
OUT = os.path.join(HERE, "background_generalization_full.pptx")

# ---------------------------------------------------------------- palette
INK = RGBColor(0x1F, 0x29, 0x33)
BODY = RGBColor(0x33, 0x3D, 0x47)
MUTED = RGBColor(0x6B, 0x72, 0x80)
BLUE = RGBColor(0x2E, 0x5C, 0x8A)
LBLUE = RGBColor(0x4C, 0x78, 0xA8)
ORANGE = RGBColor(0xC9, 0x6A, 0x0F)
GREEN = RGBColor(0x1A, 0x7F, 0x37)
RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BGSOFT = RGBColor(0xF4, 0xF6, 0xF8)
RULE = RGBColor(0xD8, 0xDE, 0xE4)

EA = "Noto Sans CJK SC"
LAT = "Arial"

SW, SH = 13.333, 7.5
M = 0.62                      # page margin, inches
CW = SW - 2 * M               # content width


def _set_font(run, size, bold=False, color=BODY, latin=LAT, ea=EA, italic=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = latin
    # python-pptx exposes only the latin typeface; CJK needs the <a:ea> element
    # or PowerPoint substitutes a fallback face for the Chinese glyphs.
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", ea)


def textbox(slide, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, first, text, size=15, bold=False, color=BODY, space_before=0,
         space_after=6, align=PP_ALIGN.LEFT, line=1.25, indent=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.line_spacing = line
    if indent:
        p.level = indent
    r = p.add_run()
    r.text = text
    _set_font(r, size, bold, color)
    return p


def rich(tf, first, parts, size=15, space_before=0, space_after=6, line=1.25,
         align=PP_ALIGN.LEFT):
    """One paragraph built from [(text, bold, color), ...] runs."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.line_spacing = line
    for item in parts:
        text, bold, color = (list(item) + [False, BODY])[:3]
        r = p.add_run()
        r.text = text
        _set_font(r, size, bold, color)
    return p


def rect(slide, l, t, w, h, fill, line=None, lw=1.0):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t),
                                Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    sh.text_frame.word_wrap = True
    return sh


def hline(slide, l, t, w, color=RULE, h=1.2):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t),
                                Inches(w), Pt(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def title_slide(prs):
    s = blank(prs)
    rect(s, 0, 0, SW, SH, RGBColor(0x14, 0x2A, 0x40))
    rect(s, 0, 0, SW, 0.14, ORANGE)
    tf = textbox(s, M, 2.05, CW, 1.5)
    para(tf, True, "Background Generalization", 46, True, WHITE, space_after=2)
    para(tf, False, "Stage 2B —— 桌面干扰物增强", 26, False, RGBColor(0x9F, 0xBF, 0xDC),
         space_after=0)
    hline(s, M, 3.72, 2.4, ORANGE, 3)
    tf = textbox(s, M, 4.02, CW, 1.6)
    para(tf, True, "RoboAgent 的第二条增强轴：保留工作区，往桌面上加杂物。",
         17, False, RGBColor(0xD5, 0xDF, 0xE8), space_after=8)
    para(tf, False, "不重着色、不移动机械臂 / 夹爪 / 蓝方块 / 红盒 / 桌面；",
         15, False, RGBColor(0x9F, 0xBF, 0xDC), space_after=2)
    para(tf, False, "state / action / timestamp 逐帧 bit-identical；不改 src/lerobot 下任何文件。",
         15, False, RGBColor(0x9F, 0xBF, 0xDC), space_after=0)
    tf = textbox(s, M, 6.35, CW, 0.5)
    para(tf, True, "组会汇报 · 只讲方法", 12.5, False, RGBColor(0x6E, 0x88, 0xA0))
    return s


def head(s, title, sub=None):
    """Standard slide header. Returns the y at which content may start."""
    rect(s, 0, 0, SW, 0.10, LBLUE)
    tf = textbox(s, M, 0.34, CW, 0.62)
    para(tf, True, title, 27, True, INK, space_after=0)
    y = 1.06
    if sub:
        tf = textbox(s, M, y, CW, 0.42)
        para(tf, True, sub, 14, False, MUTED, space_after=0)
        y += 0.50
    hline(s, M, y, CW)
    return y + 0.22


def footer(s, n):
    tf = textbox(s, M, SH - 0.46, CW, 0.3)
    p = para(tf, True, "", 10, False, MUTED)
    r = p.add_run(); r.text = f"Background Generalization · Stage 2B        {n}"
    _set_font(r, 10, False, MUTED)


def fit(prs, s, path, left, top, maxw, maxh, center=True):
    """Place an image scaled to fit a box, preserving aspect ratio."""
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    w, h = maxw, maxw / ar
    if h > maxh:
        h, w = maxh, maxh * ar
    l = left + (maxw - w) / 2 if center else left
    t = top + (maxh - h) / 2 if center else top
    return s.shapes.add_picture(path, Inches(l), Inches(t), Inches(w), Inches(h))


def bullets(s, y, items, size=15, gap=9, width=CW, left=M, line=1.3):
    tf = textbox(s, left, y, width, SH - y - 0.7)
    for i, it in enumerate(items):
        if isinstance(it, str):
            it = [("▪  ", True, LBLUE), (it, False, BODY)]
        rich(tf, i == 0, it, size=size, space_after=gap, line=line)
    return tf


def keyline(s, y, text, color=ORANGE, size=15.5, width=CW, left=M):
    """A one-line takeaway in a tinted strip."""
    h = 0.62
    rect(s, left, y, width, h, BGSOFT, line=RULE, lw=0.75)
    rect(s, left, y, 0.075, h, color)
    tf = textbox(s, left + 0.26, y, width - 0.5, h, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, True, text, size, True, INK, space_after=0)
    return y + h


def code(s, y, lines, size=12.5, left=M, width=CW, pad=0.3, line=1.32):
    h = pad * 2 + len(lines) * size * line / 72.0
    rect(s, left, y, width, h, RGBColor(0xF7, 0xF8, 0xFA), line=RULE, lw=0.75)
    tf = textbox(s, left + 0.28, y + pad, width - 0.56, h - 2 * pad)
    for i, ln in enumerate(lines):
        if isinstance(ln, str):
            ln = [(ln, False, INK)]
        p = rich(tf, i == 0, ln, size=size, space_after=0, line=line)
        for r in p.runs:
            _set_font(r, size, r.font.bold, r.font.color.rgb, latin="Consolas", ea=EA)
    return y + h


# ---------------------------------------------------------------- numbers
def load_stats():
    frs, blobs = [], []
    for p in sorted(glob.glob(os.path.join(PLATE, "episode_*", "done.json"))):
        r = json.load(open(p))
        frs.append(r["placement_frac"] * 100)
        blobs.append(r["placement_largest_blob"])
    return frs, blobs


import numpy as np  # noqa: E402

FRS, BLOBS = load_stats()
EP0 = json.load(open(os.path.join(PLATE, "episode_0", "done.json")))


# ---------------------------------------------------------------- slides
def s_problem(prs, n):
    s = blank(prs)
    y = head(s, "要解决的问题", "示教数据里，桌上除了目标物什么都没有")
    bullets(s, y, [
        [("基线数据是「干净桌面」分布：50 集示教里，桌面区域只有蓝方块和红盒，",
          False, BODY), ("没有任何其他物体", True, RED), ("。", False, BODY)],
        [("策略很容易把「桌上没别的东西」当成任务的一部分 —— 这是与任务无关的",
          False, BODY), ("虚假相关", True, RED), ("，部署时一换环境就崩。", False, BODY)],
        [("这正是 RoboAgent (arXiv:2309.01918) 指出的问题：数据增强必须同时沿",
          False, BODY), ("目标物", True, BLUE), ("和", False, BODY),
         ("场景", True, BLUE), ("两条轴走。", False, BODY)],
    ], size=15.5, gap=13)
    y = keyline(s, y + 2.35,
                "本目录负责第二条轴：保留工作区，往桌上加杂物（clutter）。")
    footer(s, n)
    return s


def s_axes(prs, n):
    s = blank(prs)
    y = head(s, "在项目里的位置", "RoboAgent 的两条增强轴")
    box_w = (CW - 0.55) / 2
    for i, (tag, name, sub, lines, color, hi) in enumerate([
        ("Stage 2A", "object_generalization", "目标物：蓝方块 → 绿方块",
         ["按目标物的掩码改写颜色", "已有 up_mask / cache 子系统",
          "本阶段复用了它的缓存 IO 与掩码"],
         LBLUE, False),
        ("Stage 2B", "background_generalization", "场景：往桌面加干扰物",
         ["保留工作区，只在空白处粘贴物体", "不重着色任何任务相关物体",
          "本轮汇报的内容"],
         ORANGE, True),
    ]):
        l = M + i * (box_w + 0.55)
        rect(s, l, y, box_w, 3.0, BGSOFT if not hi else RGBColor(0xFD, 0xF3, 0xE6),
             line=color, lw=1.8 if hi else 1.0)
        tf = textbox(s, l + 0.28, y + 0.22, box_w - 0.56, 2.6)
        para(tf, True, tag, 13, True, color, space_after=6)
        para(tf, False, name, 16.5, True, INK, space_after=2, line=1.2)
        para(tf, False, sub, 16.5, True, INK, space_after=10, line=1.2)
        for ln in lines:
            para(tf, False, "▪  " + ln, 13, False, BODY, space_after=5)
    y += 3.3
    y = keyline(s, y, "两条轴共享同一个前提：改动只发生在被明确圈定的区域内，其余像素逐位不动。")
    footer(s, n)
    return s


def s_decision(prs, n):
    s = blank(prs)
    y = head(s, "核心设计决策", "为什么不做「整张背景替换」")
    box_w = (CW - 0.6) / 2
    # left: the rejected approach
    rect(s, M, y, box_w, 2.55, RGBColor(0xFD, 0xF0, 0xEE), line=RED, lw=1.4)
    tf = textbox(s, M + 0.28, y + 0.22, box_w - 0.56, 2.2)
    para(tf, True, "✗  直觉做法：整张背景替换", 16, True, RED, space_after=8)
    for ln in ["aug = m · img + (1−m) · new_background",
               "改动了整幅画面：桌面的真实明暗、纹理、",
               "以及所有物体的影子一起被丢掉。",
               "改动面大，而且无法判定「改坏了没有」。"]:
        para(tf, False, ln, 12.5, False, BODY, space_after=4,
             line=1.2)
    # right: what this pipeline does
    rect(s, M + box_w + 0.6, y, box_w, 2.55, RGBColor(0xEE, 0xF6, 0xEE), line=GREEN, lw=1.4)
    tf = textbox(s, M + box_w + 0.88, y + 0.22, box_w - 0.56, 2.2)
    para(tf, True, "✓  本管线：最小改动合成", 16, True, GREEN, space_after=8)
    for ln in ["aug = (1−a) · img + a · background_layer",
               "只在干扰物的 alpha 内作画 —— 物体轮廓",
               "以及它自己的接触阴影。",
               "其余每一个像素与原帧逐位相同。"]:
        para(tf, False, ln, 12.5, False, BODY, space_after=4, line=1.2)
    y += 2.85
    y = keyline(s, y, "代价是前期要花力气找出「哪里可以画」；换来的是一个可以判定的不变式。",
                GREEN)
    tf = textbox(s, M, y + 0.16, CW, 0.9)
    para(tf, True, "这个不变式是可执行的：alpha == 0 的每个像素，输出必须逐位等于输入；"
                   "不等就直接 raise，而不是靠肉眼看「像不像」。",
         13.5, False, MUTED, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_pipeline(prs, n):
    s = blank(prs)
    y = head(s, "流水线", "四个阶段，各自可单独重跑")
    fit(prs, s, os.path.join(FIGS, "fig04_pipeline.png"), M, y - 0.1, CW, 3.3)
    y += 3.4
    tf = textbox(s, M, y, CW, 1.5)
    rich(tf, True, [("阶段之间只传文件：", True, INK),
                    ("Stage 1 产出的 .npy 缓存 + .jsonl 账本，Stage 2 产出一个完整的 LeRobot 数据集。",
                     False, BODY)], size=14, space_after=8, line=1.3)
    rich(tf, False, [("Stage 1 可断点续跑", True, GREEN),
                     ("（--skip-existing，且配置变了会自动判 stale 重建）；", False, BODY),
                     ("Stage 2 不可续跑", True, RED),
                     ("（LeRobotDataset.create 拒绝已存在的目录，跑到一半失败只能挪走重来）。",
                      False, BODY)], size=14, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_stage1_time(prs, n):
    s = blank(prs)
    y = head(s, "Stage 1 · 底板", "用「时间」定义背景，而不是「参考图」")
    fit(prs, s, os.path.join(FIGS, "fig01_plate_vs_frame.png"), M, y - 0.05, CW, 2.12)
    y += 2.22
    tf = textbox(s, M, y, CW, 2.4)
    rich(tf, True, [("旧方案（v1）是拿一张手工拍的「空工作台」参考图做灰度差，在这里根本跑不起来：",
                     False, BODY)], size=14, space_after=7, line=1.3)
    for t in ["那张参考图不存在；",
              "UP 相机自动曝光是开着的（单集内实测增益 0.933–1.071），固定灰度阈值会整帧同时触发；",
              "静态的机器人部件会留在参考图里、被判成「背景」，于是被当成可以覆盖的区域 —— 这是直接错误的增强；",
              "FRONT 是腕部相机、随机械臂运动，参考图相减在那里没有意义。"]:
        para(tf, False, "▪  " + t, 13, False, BODY, space_after=5, line=1.25)
    code(s, y + 1.42, [
        [("gain[t]  ", True, ORANGE), ("= 该帧 60 分位灰度 → 归一到 frame 0 的亮度（先做光度归一）", False, BODY)],
        [("plate    ", True, ORANGE), ("= 150 帧等间隔采样后的、逐像素时序中位数", False, BODY)],
        [("freq     ", True, ORANGE), ("= 每个像素在多少比例的帧里「远离」底板（用全部帧算）", False, BODY)],
    ], size=12.5)
    footer(s, n)
    return s


def s_stage1_formula(prs, n):
    s = blank(prs)
    y = head(s, "Stage 1 · 合法区", "placement —— 干扰物唯一被允许落脚的地方")
    code(s, y, [
        [("placement  =  (freq ≤ 0.10)", False, INK)],
        [("           ∩  ¬dilate(蓝方块的抓取轨迹)", False, INK),
         ("      # 不能挡住抓取点", False, MUTED)],
        [("           ∩  ¬dilate(红盒)", False, INK),
         ("                # 绝不碰任务相关物体", False, MUTED)],
        [("           ⊖  erode(安全边距, 半径 5 px)", False, INK)],
    ], size=13.5)
    y += 1.62
    tf = textbox(s, M, y, CW, 3.4)
    rich(tf, True, [("placement 是「哪里能放」的唯一权威", True, BLUE),
                    ("。它不是启发式，而是从数据里算出来的事实：一个像素只有在这整段示教里"
                     "机械臂和它的影子都几乎没碰过，才算合法。", False, BODY)],
         size=14.5, space_after=12, line=1.32)
    rich(tf, False, [("因此干扰物永远不可能落在录制轨迹会与之碰撞的位置上 —— "
                      "否则 (state, action) 标签就错了，", False, BODY),
                     ("这比「增强得不够」是严重得多的失败", True, RED), ("。", False, BODY)],
         size=14.5, space_after=12, line=1.32)
    rich(tf, False, [("排除项里真正起作用的是红盒", True, ORANGE),
                     ("：盒子内部是静止的，freq 在那里接近 0，会被当成合法落点；"
                      "排除它去掉了 ep0 原始合法面积的 16%、ep1 的 28%。", False, BODY)],
         size=14.5, space_after=12, line=1.32)
    rich(tf, False, [("方块轨迹这一项几乎是惰性的", True, ORANGE),
                     ("（ep0: 0.1221 → 0.1214），因为方块本身就在机械臂扫过的高频区里；"
                      "仍然保留，是为了那些不成立的集数。", False, BODY)],
         size=14.5, space_after=0, line=1.32)
    footer(s, n)
    return s


def s_stage1_result(prs, n):
    s = blank(prs)
    y = head(s, f"Stage 1 · 实测（episode 0）",
             f"原始合法像素 {33.5:.2f}%  →  最终 {EP0['placement_frac']*100:.2f}%")
    fit(prs, s, os.path.join(FIGS, "fig02_placement.png"), M, y - 0.05, CW, 4.35)
    y += 4.45
    y = keyline(s, y, "合法区的大小是物理决定的，不是实现选择：机械臂和它的影子扫过 UP 画面的大部分，"
                      "合法落点只剩下左右边缘和顶部。")
    footer(s, n)
    return s


def s_stage1_stats(prs, n):
    s = blank(prs)
    y = head(s, "Stage 1 · 全量", "50 集的合法区分布 —— 这是整个实验的预算")
    fit(prs, s, os.path.join(FIGS, "fig03_placement_stats.png"), M, y - 0.05, CW, 2.75)
    y += 2.85
    tf = textbox(s, M, y, CW, 2.3)
    rich(tf, True, [("ep0 / ep1 是全部 50 集里最紧的两集", True, RED),
                    ("，而它们当初是被当作冒烟集选的。用最紧的两集去定全量的密度和闸门阈值，"
                     "是这个项目里返工过的地方。", False, BODY)],
         size=14, space_after=10, line=1.3)
    rich(tf, False, [("全量统计：合法面积 min ", False, BODY),
                     (f"{FRS[0]:.1f}%", True, INK),
                     (" / median ", False, BODY), (f"{np.median(FRS):.1f}%", True, INK),
                     (" / max ", False, BODY), (f"{max(FRS):.1f}%", True, INK),
                     ("；最大连通块 min ", False, BODY),
                     (f"{min(BLOBS):,}", True, INK), (" px / median ", False, BODY),
                     (f"{int(np.median(BLOBS)):,}", True, INK), (" px。", False, BODY)],
         size=14, space_after=8, line=1.3)
    rich(tf, False, [("up_mask.npy 的极性和这里相反这一点是被刻意区分的：", False, BODY),
                     ("本目录把文件命名为 placement.npy 而不是 up_mask.npy", True, ORANGE),
                     ("，因为同名同 dtype 反语义是会被静默搞混的。", False, BODY)],
         size=14, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_sweep(prs, n):
    s = blank(prs)
    y = head(s, "Stage 1 · 一个关键旋钮", "place_freq_max：用面积换稳定性，0.10 是实测拐点")
    fit(prs, s, os.path.join(FIGS, "fig08_freq_sweep.png"), M, y - 0.1, CW, 3.7)
    y += 3.8
    tf = textbox(s, M, y, CW, 1.5)
    rich(tf, True, [("两个方向都被限制，而只有一个是正确性下限：", True, INK)],
         size=13.5, space_after=5, line=1.26)
    rich(tf, False, [("硬上限 0.5", True, RED),
                     ("：plate 是逐像素时序中位数，过半之后机械臂在多数帧里就在那儿，中位数本身就变成"
                      "了一张机械臂的图 —— 从它合成出来的干扰物是彻底错误的增强。", False, BODY)],
         size=12.5, space_after=5, line=1.26)
    rich(tf, False, [("软的、远在 0.5 以下", True, ORANGE),
                     ("：放宽阈值会让物体开始「住在」机械臂影子的梯度上，残差在 diff_threshold "
                      "附近来回跳，fg 翻转，物体在片段中途开始闪烁。", False, BODY)],
         size=12.5, space_after=0, line=1.26)
    footer(s, n)
    return s


def s_layout(prs, n):
    s = blank(prs)
    y = head(s, "Stage 2 · 采样布局", "约束宁可放弃，也不放松")
    tf = textbox(s, M, y, CW, 2.0)
    rich(tf, True, [("sample_layout", True, BLUE),
                    (" 在 placement 的连通块里按面积加权采点，每个候选位置要过四道检查，"
                     "任何一道不过就换下一个位置；尝试 ", False, BODY),
                    ("80 次还放不下就直接少放一个", True, ORANGE),
                    ("，而不是放宽任何一条约束。", False, BODY)],
         size=14.5, space_after=10, line=1.32)
    for t, hi in [
        ("物体的整个轮廓必须完整落在 placement 内 —— 只按锚点贴会让 50 px 的物体"
         "从 75 px 宽的窄条上探出去，压到机械臂路径或红盒上。", BLUE),
        ("轮廓不能被画面边框切掉。这一条是「在 placement 内」查不出来的：右边缘的合法条一直延伸到"
         "最后一列，半个身子在画面外的物体照样通过检查。实测 ep0/ep1×3 变体里 11 个物体有 4 个被切，"
         "最坏情况 2224 px 只剩 1234 px。", ORANGE),
        ("两个干扰物之间至少留 8 px；物体的影子允许出画面，只有轮廓受约束。", BLUE),
        ("每集目标 2–4 个物体。这个区间是自限的：房间不够就少放，跑批不会因为放不下而失败。", BLUE),
    ]:
        rich(tf, False, [("▪  ", True, LBLUE), (t, False, BODY)],
             size=13.5, space_after=6, line=1.3)
    y += 2.78
    tf = textbox(s, M, y, CW, 1.7)
    rich(tf, True, [("最容易漏掉的失败模式：放 0 个物体的变体", True, RED)],
         size=14.5, space_after=7, line=1.3)
    rich(tf, False, [("它照样能通过验证 —— 没有 alpha，所有内存断言都真空成立 —— "
                      "但它其实是基础集的逐位复制品，却占着一个「已增强」的编号。"
                      "所以跑生成之前先做一次布局普查（--dry-run）：",
                      False, BODY)], size=13.5, space_after=6, line=1.3)
    rich(tf, False, [("采样全部 150 个布局，任何一个放 0 个就直接 fail。本次 census: ",
                      False, BODY),
                     ("2 个物体 × 49 / 3 个 × 45 / 4 个 × 56，没有空布局。", True, GREEN)],
         size=13.5, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_composite(prs, n):
    s = blank(prs)
    y = head(s, "Stage 2 · 合成", "最小改动合成 —— 以及它买到的不变式")
    fit(prs, s, os.path.join(FIGS, "fig05_composite.png"), M, y - 0.05, CW, 3.75)
    y += 3.85
    tf = textbox(s, M, y, CW, 1.9)
    rich(tf, True, [("变体之间只有视觉上下文不同，动作完全相同", True, BLUE),
                    ("。每个基础集渲染 K 份（默认 3），K 份携带完全相同的 action —— "
                     "于是「50 集基线 vs 50×K 增强」是一个干净的对照，"
                     "不会把「数据更多」和「多样性更多」混在一起。", False, BODY)],
         size=14, space_after=9, line=1.3)
    rich(tf, False, [("物体在整段片段里不移动", True, ORANGE),
                     ("，所以布局和背景层是每集算一次，不是每帧算一次。", False, BODY)],
         size=14, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_cutouts(prs, n):
    s = blank(prs)
    y = head(s, "Stage 2 · 干扰物资产", "程序化生成，但光照参数是从数据集里量出来的")
    fit(prs, s, os.path.join(FIGS, "fig06_cutouts.png"), M, y - 0.05, CW, 3.6)
    y += 3.72
    tf = textbox(s, M, y, CW, 1.7)
    rich(tf, True, [("提供方是可插拔的", True, BLUE),
                    ("：下游只认 ", False, BODY), ("CutoutProvider", True, INK),
                    (" 协议（sample(rng) → Cutout），真实拍摄的物体以后可以直接换进来，"
                     "不用改合成和验证代码 —— 程序化物体与 rollout 时真实杂物之间存在分布偏移。",
                     False, BODY)],
         size=12.5, space_after=6, line=1.26)
    rich(tf, False, [("颜色避开蓝方块和红盒的 HSV 盒子，而且是在", False, BODY),
                     ("着色之后", True, ORANGE),
                     ("的物体上检查，用的是目标检测器自己的盒子：着色会截断通道、把边界附近的色相"
                      "推回盒内。实测 300 个 cutout，只查基础色相会漏掉约 0.7% 完全落在蓝盒内；"
                      "改查着色结果后，落进蓝盒的核心像素从 0.41% 降到 0.0027%。",
                      False, BODY)],
         size=12.5, space_after=0, line=1.26)
    footer(s, n)
    return s


def s_shadow(prs, n):
    s = blank(prs)
    y = head(s, "Stage 2 · 接触阴影", "阴影是一个梯度，不是平移复制")
    fit(prs, s, os.path.join(FIGS, "fig07_shadow_measured.png"), M, y - 0.05, CW, 3.35)
    y += 3.45
    tf = textbox(s, M, y, CW, 2.2)
    rich(tf, True, [("阴影一度是刚性偏移复制", True, RED),
                    ("（均匀暗度、整体平移一份），渲染出来像是每个干扰物后面坐着一个"
                     "「第二个灰色物体」。", False, BODY)],
         size=14, space_after=9, line=1.3)
    rich(tf, False, [("现在的做法", True, GREEN),
                     ("：把轮廓的多份拷贝沿光照方向、以高斯权重叠放（shadow_spread_ratio 就是半影）。"
                      "三个参数是对四个实测点做网格搜索拟合出来的："
                      "(0.30, 0.40, blur 1.0) 在圆盘上复现到 0.05 以内。", False, BODY)],
         size=14, space_after=9, line=1.3)
    rich(tf, False, [("测量本身也踩过坑：", True, ORANGE),
                     ("围绕红盒测出来的 240° 是错的 —— 红盒被机械臂的漫反射阴影环抱。"
                      "蓝方块是唯一干净的探针，因为它的影子随它一起离开画面（帧 300 前消失，"
                      "并被底板的中位数抹掉）。", False, BODY)],
         size=14, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_coupling(prs, n):
    s = blank(prs)
    y = head(s, "把假设变成异常", "两条耦合关系不是写在文档里，而是写成了 raise")
    code(s, y, [
        [("if", True, RED), (" self.cfg.fg_threshold > float(plate_diff_threshold):", False, INK)],
        [("    raise", True, RED), (" ValueError(", False, INK),
         ('"fg_threshold exceeds the plate\'s diff_threshold, which built', False, INK)],
        [("        ", False, INK), ('`placement` ... could paint a distractor over the arm.")', False, INK)],
    ], size=12)
    y += 1.35
    tf = textbox(s, M, y, CW, 4.0)
    rich(tf, True, [("耦合 1：", True, ORANGE),
                    ("合成器判定前景用的阈值 ", False, BODY),
                    ("必须不高于", True, RED),
                    (" 建 placement 时用的 diff_threshold。", False, BODY)],
         size=14.5, space_after=6, line=1.3)
    rich(tf, False, [("高于的话，合成器会把 placement 认为是前景的像素判成背景，"
                      "于是", False, BODY),
                     ("把干扰物画到机械臂上", True, RED),
                     ("。build_plate 把 config 写进每集的 done.json，生成脚本读回来，"
                      "FrameCompositor.__init__ 直接 raise。", False, BODY)],
         size=13.5, space_after=12, line=1.3)
    rich(tf, False, [("耦合 2：", True, ORANGE),
                     ("ProceduralConfig.feather_px ≤ PlateConfig.exclude_dilate_px。", False, BODY)],
         size=14.5, space_after=6, line=1.3)
    rich(tf, False, [("否则羽化的物体边缘会渗回红盒 —— 一个本管线绝不能碰的物体。",
                      False, BODY)], size=13.5, space_after=12, line=1.3)
    rich(tf, False, [("这两条的共同点：", True, BLUE),
                     ("违反它们不会报错、也不会让任何一帧看起来不对劲，"
                      "只会在数据里静默地产生错误的监督信号。所以它们必须是异常，不是注释。",
                      False, BODY)], size=14, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_pit_paint(prs, n):
    s = blank(prs)
    y = head(s, "踩坑 1", "合成器的 alpha 必须是 paint_union，不是 alpha_union")
    tf = textbox(s, M, y, CW, 1.5)
    rich(tf, True, [("render_background_layer", True, BLUE),
                    (" 把干扰物的", False, BODY),
                    ("接触阴影也烤进了背景层", True, ORANGE),
                    ("。所以合成时的 alpha 必须同时覆盖轮廓", False, BODY),
                    ("和阴影", True, ORANGE),
                    ("；只传轮廓的 alpha_union，阴影被烤进去却从来没被合出来。", False, BODY)],
         size=14.5, space_after=0, line=1.32)
    y += 1.28
    # the measured cost
    rect(s, M, y, CW, 1.05, RGBColor(0xFD, 0xF0, 0xEE), line=RED, lw=1.3)
    tf = textbox(s, M + 0.3, y, CW - 0.6, 1.05, anchor=MSO_ANCHOR.MIDDLE)
    rich(tf, True, [("实测代价：", True, RED),
                    ("背景层一共改了 4482 px（1884 物体 + 2608 阴影），"
                     "而只有 1884 px 真正进了画面。", False, BODY)],
         size=14.5, space_after=0, line=1.3)
    y += 1.28
    tf = textbox(s, M, y, CW, 2.6)
    rich(tf, True, [("为什么它特别危险：", True, RED),
                    ("这个错误单看任何一帧都不明显。", False, BODY)],
         size=14.5, space_after=8, line=1.3)
    rich(tf, False, [("每个干扰物变成一张没有影子的「贴纸」，而同一张图里的蓝方块、红盒、"
                      "机械臂全都有影子 —— 肉眼扫过去完全合理。已存在的测试也抓不到，"
                      "因为报错的不是数值，是语义。", False, BODY)],
         size=13.5, space_after=12, line=1.3)
    rich(tf, False, [("所以现在由 FrameCompositor.__init__ 校验传入的 alpha 是否等于 "
                      "layout.paint_union(...)，不等直接 raise。", True, GREEN)],
         size=14, space_after=8, line=1.3)
    rich(tf, False, [("alpha_union 保持「只有轮廓」的语义不变：遮挡度量和 footprint 断言"
                      "说的是物体本身，不是它的影子。", False, MUTED)],
         size=13.5, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_pit_ecc(prs, n):
    s = blank(prs)
    y = head(s, "踩坑 2", "ECC 对齐：越对齐，结果越差")
    tf = textbox(s, M, y, CW, 1.3)
    rich(tf, True, [("早期版本对每帧估计 ECC 变换，把所有帧重采样到 frame-0 坐标。"
                     "它", False, BODY),
                    ("在试过的每一集上都比「什么都不做」更差", True, RED),
                    ("，于是被删掉了。", False, BODY)],
         size=14.5, space_after=0, line=1.32)
    y += 1.10
    box_w = (CW - 0.5) / 2
    rect(s, M, y, box_w, 1.62, BGSOFT, line=RULE)
    tf = textbox(s, M + 0.26, y + 0.16, box_w - 0.52, 1.3)
    para(tf, True, "UP 相机本来就是静止的", 14.5, True, BLUE, space_after=6)
    para(tf, False, "整集 frame-0 → frame-t 的单应变换保持在 5.6 px 以内。"
                    "没有东西需要对齐 —— 估计出来的每个变换都是在估计一个非常小的量。",
         12.5, False, BODY, space_after=0, line=1.25)
    rect(s, M + box_w + 0.5, y, box_w, 1.62, RGBColor(0xFD, 0xF0, 0xEE), line=RED, lw=1.3)
    tf = textbox(s, M + box_w + 0.76, y + 0.16, box_w - 0.52, 1.3)
    para(tf, True, "但 ECC 会锁到「机械臂」上", 14.5, True, RED, space_after=6)
    para(tf, False, "机械臂覆盖画面大部分（freq > 0.1 的像素约占 70%），"
                    "于是最优相关经常是通过跟着机械臂走找到的 —— 实测最大 58 px，比干扰物本身还宽。",
         12.5, False, BODY, space_after=0, line=1.25)
    y += 1.85
    tf = textbox(s, M, y, CW, 2.6)
    rich(tf, True, [("两个后果，都被「不对齐」修掉了：", True, ORANGE)],
         size=14, space_after=7, line=1.3)
    rich(tf, False, [("1.  物体被画在 to_frame_t(alpha, warp) 上，错误的 warp 会", False, BODY),
                     ("位移物体", True, RED),
                     (" —— 那个「干扰物逐帧跳动」的闪烁就是这么来的。不对齐之后，"
                      "物体在每一帧都坐在完全相同的像素上（实测逐帧落点位移 0.0000 px）。",
                      False, BODY)], size=13.5, space_after=6, line=1.3)
    rich(tf, False, [("2.  「已对齐」的帧反而比原始帧更不对齐，中位数底板发糊"
                      "（Laplacian 方差 46–71，未 warp 是 100–120），freq 被抬高，"
                      "合法区反而", False, BODY),
                     ("缩小", True, RED),
                     ("：ep0 从 19.2% 掉到 6.1%。", False, BODY)],
         size=13.5, space_after=10, line=1.3)
    rich(tf, False, [("现在 _static_warps 返回恒等变换，但仍然是个 (T,2,3) 数组、仍然写 warp.npy —— "
                      "让「场景是否静止」保持成一个被读出、被检查的值，而不是散落在四个文件里的特例。",
                      False, MUTED)], size=13, space_after=0, line=1.3)
    footer(s, n)
    return s


def s_pit_tautology(prs, n):
    s = blank(prs)
    y = head(s, "踩坑 3", "一条永远不可能失败的断言")
    tf = textbox(s, M, y, CW, 1.2)
    rich(tf, True, [("校验里原本有一条「干扰物不能画到前景上」，写成 ", False, BODY),
                    ("(a > 0) & real", True, INK),
                    ("。它恒为 0，报 0 报一辈子。", False, BODY)],
         size=13.5, space_after=7, line=1.28)
    rich(tf, False, [("因为合成器定义 ", False, BODY), ("a = paint · (1 − fg)", True, INK),
                     ("，所以 a > 0 本身就蕴含 fg == 0，而 real ⊆ fg。"
                      "这条断言看着像证据，其实是在测自己的定义。", False, BODY)],
         size=13.5, space_after=0, line=1.28)

    y = code(s, y + 1.18, [
        [("# 恒真：测的是自己的定义", False, MUTED)],
        [("bad = (a > 0) & real", False, RED), ("          # ≡ 0", False, MUTED)],
    ], size=12.5, pad=0.22, line=1.26)

    y = code(s, y + 0.18, [
        [("# 现在：跨实现比对", False, MUTED)],
        [("fg_ind = _independent_fg(frame, cache, cube, t, ...)", False, GREEN),
         ("   # 独立重建", False, MUTED)],
        [("bad = ((a > 0) != (a_ind > 0)).sum()", False, GREEN),
         ("        # 必须 == 0", False, MUTED)],
    ], size=12.5, pad=0.22, line=1.26)

    tf = textbox(s, M, y + 0.22, CW, 1.6)
    rich(tf, True, [("用 cache 独立重建一遍前景 mask（自己的阈值、面积下限、膨胀、方块并集、"
                     "warp 约定），要求两张 mask 逐位一致 ——「没画到机械臂上」才落在 "
                     "plate.npy / gain.npy / warp.npy 上，而不是落在某个函数的自洽性上。",
                     False, BODY)], size=12.5, space_after=8, line=1.28)
    rich(tf, False, [("但要把适用范围读清楚：", True, RED),
                     ("两个实现按构造共用 18.0 这个阈值，所以它抓不到阈值错误；"
                      "差异只可能出现在残差离作画区 fg_dilate_px 以内的地方，"
                      "实测 ep0 v0 的 57 个采样帧里这个机会集是 0 px。这里的 0 意味着"
                      "「在可能不一致的地方一致」，", False, BODY),
                     ("不是「已被证明安全」", True, ORANGE), ("。", False, BODY)],
         size=12.5, space_after=0, line=1.28)
    footer(s, n)
    return s


def s_pit_metric(prs, n):
    s = blank(prs)
    y = head(s, "踩坑 4", "度量本身也可能是错的")
    box_w = (CW - 0.5) / 2
    rect(s, M, y, box_w, 2.55, BGSOFT, line=RULE)
    tf = textbox(s, M + 0.26, y + 0.2, box_w - 0.52, 2.2)
    para(tf, True, "① 等面积半径不是半径", 15, True, ORANGE, space_after=7)
    para(tf, False, "√(area/π) 是等面积半径。对方块或团块按「1.25 r」采样，"
                    "落点相对真实边缘的位置和圆盘完全不同 —— 同一个正确的模型会报出 0.19 的误差。",
         12.5, False, BODY, space_after=7, line=1.25)
    para(tf, False, "换成形状无关的量（阴影衰减到边缘值 10% 时离轮廓的距离 ÷ 该半径）后是 0.98，"
                    "而蓝方块实测 ≈ 1.0。", 12.5, False, BODY, space_after=0, line=1.25)
    rect(s, M + box_w + 0.5, y, box_w, 2.55, BGSOFT, line=RULE)
    tf = textbox(s, M + box_w + 0.76, y + 0.2, box_w - 0.52, 2.2)
    para(tf, True, "② 拿编码噪声当失败", 15, True, ORANGE, space_after=7)
    para(tf, False, "用解码后的 AV1 帧去比对精确期望值、再按 max 判失败，"
                    "会得到 43/255 的「失败」—— 那全是编码噪声。",
         12.5, False, BODY, space_after=7, line=1.25)
    para(tf, False, "改成按中位比值（对边缘不敏感）判，噪声降到 0.014。",
         12.5, False, BODY, space_after=0, line=1.25)
    y += 2.78
    tf = textbox(s, M, y, CW, 1.5)
    rich(tf, True, [("同类问题还有第三处：遮挡度量的分母", True, ORANGE)],
         size=14, space_after=6, line=1.28)
    rich(tf, False, [("判断「干扰物被挡了多少」时，分母必须用物体", False, BODY),
                     ("核心", True, ORANGE), ("（alpha ≥ 0.5），不能用 alpha > 0："
                      "羽化边缘按构造就满足 alpha < 0.5，用 alpha > 0 会让每个物体"
                      "都恒定假报约 22% 被遮挡。", False, BODY)],
         size=13, space_after=0, line=1.28)
    keyline(s, y + 1.32,
            "报误差之前先确认：这个误差是不是度量方式自己造出来的。", RED)
    footer(s, n)
    return s


def s_verify(prs, n):
    s = blank(prs)
    y = head(s, "验证", "两级「相等」，不要混为一谈")
    tf = textbox(s, M, y, CW, 0.9)
    rich(tf, True, [("写出的数据集是 AV1 重编码过的，所以逐位相同只可能在内存里成立。",
                     False, BODY)], size=14, space_after=0, line=1.3)
    y += 0.72
    rows = [
        ("内存", "composite(base) == base  wherever alpha == 0，逐位",
         "硬失败", GREEN),
        ("磁盘", "decode(gen) ≈ composite(decode(base))，差值是 AV1 漂移",
         "只报 mean / p99，不对 max 设闸", ORANGE),
    ]
    for tag, desc, note, c in rows:
        rect(s, M, y, CW, 0.74, BGSOFT, line=RULE)
        rect(s, M, y, 0.07, 0.74, c)
        tf = textbox(s, M + 0.26, y, 1.0, 0.74, anchor=MSO_ANCHOR.MIDDLE)
        para(tf, True, tag, 15, True, c, space_after=0)
        tf = textbox(s, M + 1.3, y, 6.6, 0.74, anchor=MSO_ANCHOR.MIDDLE)
        para(tf, True, desc, 13, False, BODY, space_after=0)
        tf = textbox(s, M + 8.1, y, CW - 8.4, 0.74, anchor=MSO_ANCHOR.MIDDLE)
        para(tf, True, note, 12.5, True, c, space_after=0)
        y += 0.84
    tf = textbox(s, M, y, CW, 4.0)
    rich(tf, True, [("Phase 1 —— 穷尽的 parquet 检查：", True, BLUE),
                    ("对", False, BODY), ("每一个", True, RED),
                    ("生成帧断言 state / action / timestamp / frame_index 逐位相同。",
                     False, BODY)], size=13, space_after=5, line=1.25)
    rich(tf, False, [("Phase 2 —— 按 --stride 的像素检查，逐（基础集，变体）：", True, BLUE)],
         size=13, space_after=4, line=1.25)
    for t in [
        "布局可复现：从记录的 seed 重建布局，要求与 layouts.jsonl 完全一致 —— "
        "生成器只存布局元数据、不存像素，所以可复现性本身是一条被检查的声明。",
        "落点重新证明：每个 footprint 都在 placement 内；placement 与红盒、与方块轨迹在同样的"
        "膨胀下不相交；placement 内没有 freq 超阈值的像素。",
        "背景层安全且完整：在 frame-0 空间里，它改动的每一处都在 placement 内（安全），且都被 "
        "paint_union 覆盖（完整）—— 后半条正是「只传轮廓 alpha」会静默违反的。",
        "alpha 的前景遮挡被跨实现比对。",
        "front 光度：front ≈ base × 该帧增益，比值误差 ≤ 0.035（应用增益实测 0.0139，不应用 0.067）。",
        "AV1 漂移，以及专门针对红盒的一条：|gen − base| 在红盒上的均值 < 3/255（AV1 漂移本身约 1.5），"
        "这条抓的就是「有干扰物被画到盒子上了」。",
    ]:
        rich(tf, False, [("▪  ", True, LBLUE), (t, False, BODY)],
             size=12, space_after=3, line=1.22)
    footer(s, n)
    return s


def s_limits(prs, n):
    s = blank(prs)
    y = head(s, "已知局限", "rollout 之前必须先读这一页")
    items = [
        ("跨视角不一致", "UP 视角获得了干扰物，FRONT 视角没有（只接收一个全局光度增益）。"
                     "正面视角以夹爪和方块为主，所以影响可能很小 —— 但这是一个假设，"
                     "而且恰好是 rollout 应该被设计来检测的那类假设。"),
        ("施加的是 UP 的增益", "正面相机自己的曝光没有在这里测量；把 UP 的增益用在它身上，"
                         "前提是两个相机跟踪同一套光照。它是个全局标量，不可能移动或重着色任何东西。"),
        ("干扰物会在片段首尾短暂消失", "一个落点可能被机械臂或它的影子扫过几帧，这些帧里物体被遮掉。"
                              "它不会闪 —— mask 是稳定的、物体在每一帧都坐在相同的像素上 —— "
                              "但它确实会短暂消失。"),
        ("干扰物是按「空桌面」打的光", "它们从 plate 合成，而 plate 用时序中位数去掉了机械臂的漫反射阴影。"
                              "所以站在机械臂影子里的干扰物会被画成没有阴影的。这正是前景 mask 里包含"
                              "影子的原因：那些帧干脆不画物体，而不是画错光。"),
        ("程序化 ≠ 真实", "见前页。真实物体提供方应当在拍摄完成后成为主实验，程序化的这份作为对照。"),
        ("每集 2–4 个，且必然在画面边缘", "中位集有约 24% 的合法面积，但它被限制在左右边缘和顶部，"
                                 "所以物体在位置上是外围的。要「高密度杂物」—— 把杂物放进工作区本身 —— "
                                 "从这个相机出发是做不到的，除非让录制的 (state, action) 标签失效。"),
    ]
    tf = textbox(s, M, y, CW, SH - y - 0.7)
    for i, (t, d) in enumerate(items):
        rich(tf, i == 0, [("▪  " + t + "  ", True, ORANGE), (d, False, BODY)],
             size=13.5, space_after=9, line=1.3)
    footer(s, n)
    return s


def s_summary(prs, n):
    s = blank(prs)
    y = head(s, "小结")
    tf = textbox(s, M, y, CW, 4.6)
    pts = [
        ("问题", "基线数据里桌面永远是干净的，策略会把「桌上没别的东西」当成任务的一部分。"),
        ("做法", "保留工作区，只在 placement 允许的位置粘贴干扰物；"
               "alpha 之外逐位不动，state / action / timestamp 完全不变。"),
        ("placement 是全部的关键", "它把「哪里可以画」变成从数据里算出来的事实，"
                             "而不是一个启发式 —— 干扰物永远不可能落在录制轨迹会撞到的位置上。"),
        ("每一个会静默出错的地方都被改成了异常", "alpha 必须覆盖阴影、"
                                      "fg_threshold 不得高于 diff_threshold、羽化半径不得大于排除膨胀、"
                                      "放 0 个物体的布局直接 fail。"),
        ("度量本身也被审查过", "等面积半径、AV1 编码噪声、alpha > 0 的分母 —— "
                        "三处都会让一个正确的实现报出错误的误差。"),
    ]
    for i, (t, d) in enumerate(pts):
        rich(tf, i == 0, [("▪  ", True, LBLUE), (t + "：", True, INK), (d, False, BODY)],
             size=14.5, space_after=13, line=1.32)
    footer(s, n)
    return s


def main():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    title_slide(prs)
    s_problem(prs, 2)
    s_axes(prs, 3)
    s_decision(prs, 4)
    s_pipeline(prs, 5)
    s_stage1_time(prs, 6)
    s_stage1_formula(prs, 7)
    s_stage1_result(prs, 8)
    s_stage1_stats(prs, 9)
    s_sweep(prs, 10)
    s_layout(prs, 11)
    s_composite(prs, 12)
    s_cutouts(prs, 13)
    s_shadow(prs, 14)
    s_coupling(prs, 15)
    s_pit_paint(prs, 16)
    s_pit_ecc(prs, 17)
    s_pit_tautology(prs, 18)
    s_pit_metric(prs, 19)
    s_verify(prs, 20)
    s_limits(prs, 21)
    s_summary(prs, 22)

    prs.save(OUT)
    print("wrote", OUT, f"({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
