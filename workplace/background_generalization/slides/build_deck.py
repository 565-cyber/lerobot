"""Build the Stage 2B background-generalization deck (5 slides).

Usage (from the repo root):
    python workplace/background_generalization/slides/build_deck.py

Writes slides/background_generalization.pptx next to this file. Every number in
it is either read from the plate cache at build time or quoted from the
project's README.md / source docstrings.

The 22-slide long version lives in build_deck_full.py ->
background_generalization_full.pptx; the layout helpers are imported from it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pptx import Presentation  # noqa: E402
from pptx.enum.text import MSO_ANCHOR  # noqa: E402
from pptx.util import Inches  # noqa: E402

from build_deck_full import (  # noqa: E402
    BGSOFT, BLUE, BODY, BLOBS, CW, FRS, FIGS, GREEN, INK, LBLUE, M, MUTED,
    ORANGE, RED, RGBColor, RULE, SH, SW, WHITE,
    blank, bullets, code, fit, footer, head, hline, keyline, para, rect, rich,
    textbox, title_slide,
)

OUT = os.path.join(HERE, "background_generalization.pptx")


# ---------------------------------------------------------------- 1. problem
def s_problem(prs, n):
    s = blank(prs)
    y = head(s, "要解决的问题", "示教数据里，桌上除了目标物什么都没有")

    tf = textbox(s, M, y, CW, 1.5)
    rich(tf, True, [("基线数据是「干净桌面」分布：", False, BODY),
                    ("50 集里桌面区域只有蓝方块和红盒，没有任何其他物体。", True, RED)],
         size=15, space_after=7, line=1.3)
    rich(tf, False, [("策略很容易把「桌上没别的东西」当成任务的一部分 —— 这是与任务无关的",
                      False, BODY), ("虚假相关", True, RED),
                     ("，一换环境就崩。RoboAgent 指出：数据增强必须同时沿",
                      False, BODY), ("目标物", True, BLUE), ("和", False, BODY),
                     ("场景", True, ORANGE), ("两条轴走。本目录负责第二条。", False, BODY)],
         size=15, space_after=0, line=1.3)
    y += 1.15

    box_w = (CW - 0.55) / 2
    for i, (tag, title, lines, color, fill) in enumerate([
        ("✗  直觉做法", "整张背景替换",
         ["aug = m · img + (1−m) · new_background",
          "桌面的明暗、纹理、所有物体的影子一起被换掉。",
          "改动面大，而且无法判定「改坏了没有」。"],
         RED, RGBColor(0xFD, 0xF0, 0xEE)),
        ("✓  本管线", "最小改动合成",
         ["aug = (1−a) · img + a · background_layer",
          "只在干扰物的 alpha 内作画（轮廓 + 它自己的接触阴影）。",
          "其余每一个像素与原帧逐位相同。"],
         GREEN, RGBColor(0xEC, 0xF7, 0xEF)),
    ]):
        l = M + i * (box_w + 0.55)
        rect(s, l, y, box_w, 2.35, fill, line=color, lw=1.6)
        tf = textbox(s, l + 0.26, y + 0.2, box_w - 0.52, 2.0)
        para(tf, True, tag, 13.5, True, color, space_after=3)
        para(tf, False, title, 19, True, INK, space_after=8)
        for ln in lines:
            c = INK if ln.startswith("aug") else BODY
            para(tf, False, ln, 12.5, ln.startswith("aug"), c, space_after=5, line=1.25)
    y += 2.62

    keyline(s, y, "最小改动的回报是一条可判定的不变式：alpha == 0 的每个像素，输出必须逐位等于输入。")
    footer(s, n)
    return s


# ------------------------------------------------------- 2. pipeline + stage1
def s_pipeline(prs, n):
    s = blank(prs)
    y = head(s, "方法 · 四阶段流水线", "Stage 1 算「哪里可以画」，Stage 2 才画")
    fit(prs, s, os.path.join(FIGS, "fig04_pipeline.png"), M, y - 0.08, CW, 2.05)
    y += 2.05

    y = code(s, y, [
        [("placement = (freq ≤ 0.10)", False, INK),
         ("            # 这片像素在整段示教里几乎没被机械臂碰过", False, MUTED)],
        [("          ∩ ¬dilate(蓝方块的抓取轨迹)", False, INK),
         ("  # 不能挡住抓取点", False, MUTED)],
        [("          ∩ ¬dilate(红盒)", False, INK),
         ("              # 绝不碰任务相关物体", False, MUTED)],
        [("          ⊖ erode(安全边距, 半径 5 px)", False, INK)],
    ], size=12.5)
    y += 0.22

    tf = textbox(s, M, y, CW, 1.6)
    rich(tf, True, [("plate", True, BLUE), (" 是 150 帧的逐像素时序中位数 —— 用",
                     False, BODY), ("时间", True, INK),
                    ("而不是「参考图」定义背景，因为在自动曝光下拿一张空桌面参考图做灰度差",
                     False, BODY), ("根本跑不起来", True, RED), ("。", False, BODY)],
         size=13, space_after=6, line=1.26)
    stats = (f"这四步不是启发式：ep0 从 33.52% 收到 19.27%；50 集 min "
             f"{float(np.min(FRS)):.1f}% / median {float(np.median(FRS)):.1f}% / max "
             f"{float(np.max(FRS)):.1f}%，最大的一块中位还有 "
             f"{float(np.median(BLOBS)) / 1000.0:.0f}k px。")
    rich(tf, False, [(stats, False, BODY)], size=13, space_after=0, line=1.26)
    footer(s, n)
    return s


# ------------------------------------------------------- 3. stage 2 composite
def s_composite(prs, n):
    s = blank(prs)
    y = head(s, "方法 · 合成", "最小改动合成 —— 以及它买到的不变式")
    fit(prs, s, os.path.join(FIGS, "fig05_composite.png"), M, y - 0.1, CW, 3.5)
    y += 3.5

    tf = textbox(s, M, y, CW, 2.1)
    rich(tf, True, [("干扰物的每个参数都是从数据里量出来的，不是编的：", True, INK)],
         size=13.5, space_after=6, line=1.26)
    for t in [
        "颜色避开蓝方块与红盒的 HSV 盒子，且是在着色之后的物体上检查（落进蓝盒的核心像素"
        " 0.41% → 0.0027%）。",
        "接触阴影是一个梯度而不是平移复制：0.30 长度 / 0.40 展开 / 0.43 强度，由实测点网格搜索拟合。",
        "alpha 必须同时覆盖轮廓和阴影（paint_union）：只传轮廓的话，阴影被烤进背景层却从没被合出来，"
        "每个干扰物变成一张没有影子的贴纸。",
    ]:
        rich(tf, False, [("▪  ", True, LBLUE), (t, False, BODY)],
             size=12.5, space_after=5, line=1.24)
    footer(s, n)
    return s


# ------------------------------------------------------- 4. pitfalls + verify
def s_pitfalls(prs, n):
    s = blank(prs)
    y = head(s, "踩坑与验证", "每一个会静默出错的地方，都被改成异常或跨实现比对")

    col_w = (CW - 0.5) / 2
    tf = textbox(s, M, y, col_w, 4.6)
    rich(tf, True, [("四个坑", True, RED)], size=15, space_after=8, line=1.25)
    for t, d in [
        ("恒真的断言", "「干扰物不能画到前景上」写成 (a > 0) & real —— 它恒为 0，"
                  "因为 a > 0 本身就蕴含 fg == 0。现在改成跨实现比对："
                  "从 cache 独立重建一遍前景 mask，要求逐位一致。"),
        ("ECC 对齐反而更差", "机械臂覆盖了画面大部分，最优相关会跟着机械臂走，"
                       "实测最大 58 px 偏移。删掉对齐后，物体逐帧落点位移 0.0000 px。"),
        ("只传轮廓 alpha", "阴影被烤进背景层却从没被合出来 —— 单看任何一帧都不明显，"
                     "因为网格里的前提是「这一帧看起来对不对」。"
                     "实测背景层改了 4482 px，只有 1884 px 进了画面。"),
        ("度量本身是错的", "等面积半径、AV1 编码噪声、alpha > 0 的分母 —— "
                     "三处都会让一个正确的实现报出错误的误差。"),
    ]:
        rich(tf, False, [("▪  " + t + "　", True, ORANGE), (d, False, BODY)],
             size=12, space_after=8, line=1.24)

    tf = textbox(s, M + col_w + 0.5, y, col_w, 4.6)
    rich(tf, True, [("验证：两级「相等」，不要混为一谈", True, GREEN)],
         size=15, space_after=8, line=1.25)
    for t in [
        "内存级：composite(base) == base wherever alpha == 0，逐位 —— 硬失败。",
        "磁盘级：decode(gen) ≈ composite(decode(base))，差值是 AV1 漂移 —— 只报 mean / p99，"
        "不对 max 设闸。",
        "Phase 1：对每一个生成帧穷尽断言 state / action / timestamp / frame_index 逐位相同。",
        "Phase 2：从 seed 重建布局要求与 layouts.jsonl 完全一致；每个 footprint 重新证明"
        "落在 placement 内；背景层在 frame-0 空间既安全（改动都在 placement 内）又完整"
        "（都被 paint_union 覆盖）。",
    ]:
        rich(tf, False, [("▪  ", True, LBLUE), (t, False, BODY)],
             size=12, space_after=8, line=1.24)
    rich(tf, False, [("已知局限：", True, ORANGE),
                     ("跨视角不一致（干扰物只在 UP 视角）；物体被限制在画面边缘，"
                      "做不到「高密度杂物」；程序化物体与真实杂物存在分布偏移，"
                      "真实拍摄的提供方应当在 rollout 前换进来。", False, BODY)],
         size=12, space_after=0, line=1.24)
    footer(s, n)
    return s


# ------------------------------------------------------- 5. summary
def s_summary(prs, n):
    s = blank(prs)
    y = head(s, "小结")
    tf = textbox(s, M, y, CW, 4.8)
    pts = [
        ("问题", "基线数据里桌面永远是干净的，策略会把「桌上没别的东西」当成任务的一部分。"),
        ("做法", "保留工作区，只在 placement 允许的位置粘贴干扰物；alpha 之外逐位不动，"
               "state / action / timestamp 完全不变。"),
        ("placement 是关键", "它把「哪里可以画」变成从数据里算出来的事实，而不是启发式 —— "
                        "干扰物永远不可能落在录制轨迹会撞到的位置上。"),
        ("会静默出错的地方都改成了异常", "alpha 必须覆盖阴影、fg_threshold 不得高于 "
                              "diff_threshold、羽化半径不得大于排除膨胀、放 0 个物体的布局直接 fail。"),
        ("度量本身也被审查过", "等面积半径、AV1 编码噪声、alpha > 0 的分母 —— "
                        "三处都会让一个正确的实现报出错误的误差。"),
    ]
    for i, (t, d) in enumerate(pts):
        rich(tf, i == 0, [("▪  ", True, LBLUE), (t + "：", True, INK), (d, False, BODY)],
             size=15, space_after=15, line=1.32)
    footer(s, n)
    return s


def main():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    title_slide(prs)
    s_problem(prs, 2)
    s_pipeline(prs, 3)
    s_composite(prs, 4)
    s_pitfalls(prs, 5)
    s_summary(prs, 6)

    prs.save(OUT)
    print("wrote", OUT, f"({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
