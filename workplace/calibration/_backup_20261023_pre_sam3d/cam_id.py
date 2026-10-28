#!/usr/bin/env python3
"""认相机机位 —— 按 `/dev/v4l/by-id` 的**序列号**认，**不按 `/dev/videoN` 编号**。

为什么不能用编号：UVC 相机的 `/dev/videoN` 由枚举顺序决定，插拔一次就可能换号。
上一轮记录说「video2 = up」，今天抓的图却相反 —— 就是换号了（或当时就记错了）。

判据用**画面空间标准差**，不用亮度：这两个相机带自动曝光/增益，挡住镜头后
画面会被自动增益推**亮**（不是变黑），但**均匀** —— 均匀就造不出细节，std 会掉下去。

用法：
    python workplace/calibration/cam_id.py                 # 40 秒，见下面时间表

时间表（脚本会在 stdout 打出来，也在 `cam_id_log.txt` 留一份）：
    0 .. lead                 什么也别挡（读提示的时间）
    lead .. lead+dwell        挡住【装在机械臂手腕上】那个镜头
    +dwell .. +2*dwell        松开，改挡【架在桌上、俯视工作台】那个镜头
    最后 tail                 全部松开
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
CALIB_DIR = HERE.parent
BYID = Path("/dev/v4l/by-id")
OUT_DIR = CALIB_DIR / "cam_id"

# 同一型号两台，唯一分得开的就是序列号：一台带序列号、一台不带。
CAMS = {
    "camera_A": BYID / "usb-icSpring_icspring_camera-video-index0",
    "camera_B": BYID / "usb-icSpring_icspring_camera_202404160005-video-index0",
}


def resolve() -> dict[str, Path]:
    out = {}
    for tag, p in CAMS.items():
        if not p.exists():
            print(f"[!] 找不到 {p} —— 相机没插？")
            continue
        out[tag] = Path(p.resolve())          # 解到 /dev/videoN，只为了打印
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=float, default=14.0, help="开场留白（秒）")
    ap.add_argument("--dwell", type=float, default=15.0, help="每段遮挡时长（秒）")
    ap.add_argument("--tail", type=float, default=6.0, help="收尾（秒）")
    ap.add_argument("--hz", type=float, default=4.0, help="采样率")
    a = ap.parse_args()

    devs = resolve()
    if len(devs) < 2:
        return 2

    print("[设备] by-id → 实际节点")
    for tag, dev in devs.items():
        print(f"  {tag:9s} → {dev}")

    t0 = time.time()
    # (段名, 起, 止)，起止都是相对 t0 的秒
    bounds = [
        ("lead",     0.0,                a.lead),
        ("wrist",    a.lead,             a.lead + a.dwell),
        ("overhead", a.lead + a.dwell,   a.lead + 2 * a.dwell),
        ("tail",     a.lead + 2 * a.dwell, a.lead + 2 * a.dwell + a.tail),
    ]
    total = bounds[-1][2]

    print("\n[时间表] 从现在起：")
    print(f"  0 – {a.lead:.0f}s             什么也别挡")
    print(f"  – {a.lead + a.dwell:.0f}s         挡住【手腕上那个】相机的镜头")
    print(f"  – {a.lead + 2 * a.dwell:.0f}s         松开，改挡【架在桌上俯视工作台那个】相机的镜头")
    print(f"  – {total:.0f}s         全部松开\n")
    sys.stdout.flush()

    def phase_at(el: float) -> str:
        for name, lo, hi in bounds:
            if lo <= el < hi:
                return name
        return bounds[-1][0]

    caps = {}
    for tag, dev in devs.items():
        c = cv2.VideoCapture(str(dev))
        if not c.isOpened():
            print(f"[!] 打不开 {dev}")
            continue
        caps[tag] = c

    samples: list[tuple[float, str, float, float]] = []   # (el, tag, mean, std)
    saved: set[tuple[str, str]] = set()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        el = time.time() - t0
        if el >= total:
            break
        ph = phase_at(el)
        for tag, cap in caps.items():
            ok, f = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            samples.append((el, tag, float(g.mean()), float(g.std())))
            # 每段过半后存一张代表帧，供肉眼回看
            key = (ph, tag)
            lo = next(l for n, l, _ in bounds if n == ph)
            if key not in saved and el > lo + 5:
                cv2.imwrite(str(OUT_DIR / f"{ph}_{tag}.png"), f)
                saved.add(key)
        time.sleep(1.0 / a.hz)

    for c in caps.values():
        c.release()

    # ---------------------------------------------------------------- 汇总
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s)
        lines.append(s)

    emit("\n[结果] 每段的画面空间标准差（std 掉下去 = 镜头被挡）")
    emit(f"  {'段':10s}" + "".join(f" {t:>22s}" for t in sorted(devs)))
    emit(f"  {'':10s}" + "".join(f" {'mean':>10s} {'std':>10s}" for _ in sorted(devs)))

    by_phase: dict[str, dict[str, tuple[float, float]]] = {}
    for name, _, _ in bounds:
        row = f"  {name:10s}"
        d: dict[str, tuple[float, float]] = {}
        for tag in sorted(devs):
            v = [(m, s) for t, tg, m, s in samples if tg == tag and phase_at(t) == name]
            if not v:
                row += f" {'—':>10s} {'—':>10s}"
                continue
            d[tag] = (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
            row += f" {d[tag][0]:10.1f} {d[tag][1]:10.1f}"
        by_phase[name] = d
        emit(row)

    emit("")
    tags = sorted(devs)
    for name, want in (("wrist", "手腕上那个"), ("overhead", "俯视工作台那个")):
        d, base = by_phase.get(name, {}), by_phase.get("lead", {})
        if len(d) < 2 or len(base) < 2:
            continue
        drop = {t: base[t][1] - d[t][1] for t in d}
        hit = min(drop, key=drop.get)          # std 掉得最多的 = 被挡的那个
        other = [t for t in d if t != hit][0]
        emit(f"  挡住【{want}】那一轮：{hit} 的 std {base[hit][1]:.1f} → {d[hit][1]:.1f}"
             f"（−{drop[hit]:.1f}）；{other} 变动 {drop[other]:+.1f}")
        emit(f"  ⇒ {hit}（{devs[hit]}）= 【{want}】的相机")
        emit("")

    (CALIB_DIR / "cam_id_log.txt").write_text("\n".join(lines) + "\n")
    print(f"[save] {CALIB_DIR / 'cam_id_log.txt'}；代表帧在 {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
