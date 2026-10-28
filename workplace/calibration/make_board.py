#!/usr/bin/env python3
"""生成可打印的标定板（ChArUco）—— A4、300 dpi、1:1。

为什么用 ChArUco 而不是纯棋盘格：手眼标定时**机械臂会挡住标定板**，纯棋盘格一挡就
整块认不出来；ChArUco 的每个 ArUco marker 独立可认，挡掉一半还剩一半。

板上带一把 **100 mm 自检尺**（10 mm 一格）。打印后请用真尺子量它，把实测值告诉我 ——
**公制尺度按实测值算，不按标称值**（打印机「适应页面」会把 25 mm 印成 24.2 mm，
差 3% 会一路带进投影里）。

输出：
    board/charuco_A4_6x8_25mm.pdf          ← **打印这个**，页面框就是 A4，不会被缩放
    board/charuco_A4_6x8_25mm_300dpi.png   预览 / 留档（同等内容，但没有物理尺寸）
    board/charuco_board.json               几何参数
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "board"

DPI = 300
MM = DPI / 25.4                       # 每毫米像素数 = 11.811

PAGE_MM = (210.0, 297.0)              # A4，物理尺寸的**唯一来源**

SQUARES_X, SQUARES_Y = 6, 8
SQUARE_MM, MARKER_MM = 25.0, 18.0
DICT = cv2.aruco.DICT_5X5_100


def mm(v: float) -> int:
    return int(round(v * MM))


def write_pdf(page_bgr, path: Path) -> None:
    """把页面位图按**精确 A4 物理尺寸**写成 PDF。

    为什么不用 PNG 打印：PNG 里没有可信的物理尺寸，打印机会自己假定 72 dpi 或
    直接「适应页面」（差 3%）。PDF 的 /MediaBox 是权威的 —— 页面框 210x297 mm
    写死，打印对话框默认「实际大小」。位图仍以 300 dpi 嵌进去（1 px = 1/300 in）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w_mm, h_mm = PAGE_MM                             # = (210.0, 297.0)
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))   # 页面 = figsize，无白边
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w_mm)                              # 坐标直接用 mm，y 朝下
    ax.set_ylim(h_mm, 0)
    ax.axis("off")
    ax.imshow(cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB),
              extent=(0, w_mm, h_mm, 0), interpolation="none")
    fig.savefig(path, format="pdf")                   # 不加 bbox_inches，否则页面会被裁
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    page_w, page_h = mm(PAGE_MM[0]), mm(PAGE_MM[1])    # A4
    bw, bh = mm(SQUARES_X * SQUARE_MM), mm(SQUARES_Y * SQUARE_MM)

    d = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                   SQUARE_MM / 1000.0, MARKER_MM / 1000.0, d)
    bimg = board.generateImage((bw, bh), marginSize=0, borderBits=1)

    page = np.full((page_h, page_w, 3), 255, np.uint8)
    x0, y0 = (page_w - bw) // 2, mm(28)
    page[y0:y0 + bh, x0:x0 + bw] = cv2.cvtColor(bimg, cv2.COLOR_GRAY2BGR)

    # 四角对准十字（裁切/摆正用）
    for cx, cy in ((x0, y0), (x0 + bw, y0), (x0, y0 + bh), (x0 + bw, y0 + bh)):
        cv2.drawMarker(page, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, mm(6), 2)

    # ---- 100 mm 自检尺 -------------------------------------------------
    yb = y0 + bh + mm(14)
    xs, xe = x0, x0 + mm(100)
    cv2.line(page, (xs, yb), (xe, yb), (0, 0, 0), 3)
    for k in range(11):                         # 每 10 mm 一格
        x = xs + mm(10) * k
        cv2.line(page, (x, yb - mm(4 if k % 5 else 7)), (x, yb), (0, 0, 0), 2)
    cv2.putText(page, "0", (xs, yb + mm(7)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(page, "100 mm (MEASURE THIS)", (xe - mm(52), yb + mm(7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # ---- 说明（只用 ASCII：PIL/cv2 默认字体没有中文字形，会出豆腐块）----
    cap = [
        f"ChArUco  {SQUARES_X}x{SQUARES_Y} squares   SQUARE={SQUARE_MM:.0f}mm  MARKER={MARKER_MM:.0f}mm",
        f"dict=DICT_5X5_100   A4  {DPI} dpi   PRINT AT 100% / ACTUAL SIZE (do NOT 'fit to page')",
        "GLUE/ TAPE FLAT on a rigid board before use; keep it fixed during hand-eye capture.",
    ]
    for i, t in enumerate(cap):
        cv2.putText(page, t, (x0, yb + mm(20) + mm(9) * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

    png = OUT / f"charuco_A4_{SQUARES_X}x{SQUARES_Y}_{SQUARE_MM:.0f}mm_{DPI}dpi.png"
    cv2.imwrite(str(png), page)

    # ---- PDF：**这个才是拿去打印的**，PNG 只当预览/留档 -------------------
    pdf = OUT / f"charuco_A4_{SQUARES_X}x{SQUARES_Y}_{SQUARE_MM:.0f}mm.pdf"
    write_pdf(page, pdf)

    meta = {
        "pattern": "charuco",
        "dict": "DICT_5X5_100",
        "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
        "square_mm_nominal": SQUARE_MM, "marker_mm_nominal": MARKER_MM,
        "dpi": DPI, "page": "A4",
        "board_px": [bw, bh], "board_origin_px": [x0, y0],
        "checkerbar_mm_nominal": 100.0,
        "checkerbar_px": [xs, yb, xe],
        "NOTE": "真正用的时候 square_mm 要按用户实测的 100mm 尺长等比例缩放："
                "square_mm = 25.0 * (measured_100mm / 100.0)",
    }
    (OUT / "charuco_board.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[save] {pdf}  ← 打印这个（A4 / 实际大小）")
    print(f"[save] {png}  ({page.shape[1]}x{page.shape[0]} px)  仅预览/留档")
    print(f"[save] {OUT / 'charuco_board.json'}")
    print(f"  棋盘 {bw}x{bh}px @ ({x0},{y0})；尺 {xs}..{xe} @ y={yb}")
    print(f"  标称：{SQUARES_X}x{SQUARES_Y} 格，格 {SQUARE_MM:.0f} mm，marker {MARKER_MM:.0f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
