#!/usr/bin/env python
"""给 LeRobot 数据集的目标物体烘焙「绿色视觉提示」，用于提升多物体场景下的抓取成功率。

思路（对应论文 "Annotation-Guided Pick-and-Place" 的 mask/bbox 条件，自动化版）：
  1. 感知层 detect_target()：用颜色分割定位「目标物体」，输出 mask + bbox。
     后续接入 VLM agent 时，只把这里换成 VLM / Grounding-SAM 的 grounding 输出即可。
  2. 接入层 render()：把 mask/bbox 画成绿色覆盖层，直接烘焙进 RGB 视频（方法①，零模型改动）。
  3. 复制数据集 → 重编码选中相机视频 → 原 parquet / stats / info.json 不变（只改 codec 字段）。

为什么这样做能提升抓取成功率：
  多物体场景的本质是「目标消歧」——policy 需要知道抓哪一个。绿色框/掩膜就是给
  policy 的这个「目标说明符」。训练时每集把目标框画上，ACT 就能学会「框住的那个才是目标」。

用法示例（先预览，确认分割对不对）：
  python workplace/annotate_target.py \
      --src dataset/newdata --dst dataset/newdata_prompted \
      --camera up --color blue --render bbox --preview 60

确认无误后，去掉 --preview 跑完整标注：
  python workplace/annotate_target.py \
      --src dataset/newdata --dst dataset/newdata_prompted \
      --camera up --color blue --render bbox

依赖：conda 环境 lerobot（含 av / cv2 / pyarrow），已验证可解码 AV1、编码 h264。
"""

from __future__ import annotations

import argparse
import json
import shutil
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 颜色范围（HSV）。数值来自对 dataset/newdata 实际帧的统计，可按需微调。
# --------------------------------------------------------------------------- #
COLOR_HSV = {
    "blue": ((95, 70, 50), (140, 255, 255)),
    "yellow": ((15, 70, 90), (40, 255, 255)),
}


# --------------------------------------------------------------------------- #
# 感知层：目标定位
# --------------------------------------------------------------------------- #
def detect_target(frame_bgr: np.ndarray, hsv_range, min_area: int):
    """在单帧里定位目标物体，返回 (mask, bbox)。

    bbox 为 [x1, y1, x2, y2]（像素，闭区间），未检测到返回 (None, None)。

    ★ VLM 替换点：接入 VLM agent 时，用 VLM 对当前帧做 grounding，直接返回同样的
      (mask, bbox) 即可，本脚本其余部分（render + 重编码）无需改动。
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v) = hsv_range
    mask = cv2.inRange(hsv, np.array([lo_h, lo_s, lo_v]), np.array([hi_h, hi_s, hi_v]))

    # 小形态学去噪（目标块在俯视图里只有 ~40px，内核别开太大）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # 取面积最大的连通域作为目标
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, None
    x, y, w, h = cv2.boundingRect(c)
    return mask, (x, y, x + w, y + h)


# --------------------------------------------------------------------------- #
# 接入层：烘焙进图
# --------------------------------------------------------------------------- #
def render(frame_bgr: np.ndarray, mask, bbox, mode: str, pad: int = 0) -> np.ndarray:
    """把目标 mask/bbox 画进 RGB 图，返回新图。

    pad：bbox 外扩像素数（留白），避免绿线压在物体轮廓上造成边界损失。
         留白太大又会框进旁边物体（多物体场景反而降低消歧精度），需权衡。
    """
    out = frame_bgr.copy()
    if mode in ("mask", "both") and mask is not None:
        overlay = out.copy()
        overlay[mask > 0] = (0, 255, 0)  # 绿色填充目标
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    if mode in ("bbox", "both") and bbox is not None:
        h, w = out.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return out


# --------------------------------------------------------------------------- #
# 数据集元信息：读每集帧范围
# --------------------------------------------------------------------------- #
def read_episode_bounds(src: Path) -> list[tuple[int, int]]:
    """返回 [(from_index, to_index), ...]，按全局帧索引排序。

    数据存于 meta/episodes/chunk-*/file-*.parquet，字段 dataset_from_index /
    dataset_to_index 即该集在整份数据集里的帧范围（半开区间）。
    """
    import pyarrow.parquet as pq

    bounds = []
    for f in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f).to_pandas()
        for _, row in t.iterrows():
            bounds.append((int(row["dataset_from_index"]), int(row["dataset_to_index"])))
    bounds.sort()
    return bounds


def episode_index_for(bounds: list[tuple[int, int]], frame_idx: int) -> int | None:
    """全局帧索引 -> 集编号。"""
    starts = [b[0] for b in bounds]
    i = bisect_right(starts, frame_idx) - 1
    if i < 0:
        return None
    return i if frame_idx < bounds[i][1] else None


# --------------------------------------------------------------------------- #
# 视频处理
# --------------------------------------------------------------------------- #
def _open_video(path: Path):
    import av

    return av.open(str(path))


def _iter_frames(container):
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        yield frame.to_ndarray(format="bgr24")


def get_frame_count(path: Path) -> int:
    """返回视频文件帧数（用于多文件时累计全局帧偏移）。"""
    container = _open_video(path)
    n = container.streams.video[0].frames
    container.close()
    return int(n)


def detect_static_boxes(video_path: Path, bounds, hsv_range, min_area, detect_frames=40, offset=0):
    """每集取前 detect_frames 帧检测目标，返回每集的中位 bbox（静态框）。

    静态框和原论文一致（一集一个固定目标框），也能避免逐帧检测的抖动。
    offset：该视频文件在整个数据集里的全局起始帧索引（多文件时用）。
    """
    container = _open_video(video_path)
    acc = {}  # episode -> list of bbox
    for i, img in enumerate(_iter_frames(container)):
        g = offset + i  # 全局帧索引
        ep = episode_index_for(bounds, g)
        if ep is None:
            continue
        # 只检测每集开头一段（此时目标还在桌上）
        if g - bounds[ep][0] >= detect_frames:
            continue
        _, bbox = detect_target(img, hsv_range, min_area)
        if bbox is not None:
            acc.setdefault(ep, []).append(bbox)
    container.close()

    boxes = {}
    for ep, lst in acc.items():
        arr = np.array(lst)
        boxes[ep] = tuple(np.median(arr, axis=0).astype(int).tolist())
    return boxes


def process_video(
    src_mp4: Path,
    dst_mp4: Path,
    bounds,
    boxes: dict,
    hsv_range,
    min_area,
    mode: str,
    render_mode: str,
    preview_dir: Path | None,
    crf: int,
    offset: int = 0,
    pad: int = 0,
):
    """解码 -> 渲染 -> 编码。offset：该视频文件的全局起始帧索引。"""
    import av

    container = _open_video(src_mp4)
    stream = container.streams.video[0]
    fps = stream.average_rate or 30  # Fraction，直接传给 av（不能用 float）
    w, h = stream.width, stream.height

    if preview_dir is not None:
        # 预览模式：只保存带框的样本 PNG，不重编码
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(_iter_frames(container)):
            g = offset + i
            ep = episode_index_for(bounds, g)
            bbox = boxes.get(ep) if ep is not None else None
            mask = None
            if render_mode == "static":
                pass  # 用静态框，无需逐帧检测
            else:
                mask, bbox = detect_target(img, hsv_range, min_area)
            if bbox is not None:
                img = render(img, mask, bbox, mode, pad=pad)
            cv2.imwrite(str(preview_dir / f"frame_{i:06d}.png"), img)
        container.close()
        return

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_mp4.with_suffix(".tmp.mp4")
    out = av.open(str(tmp), "w")

    # 用与源视频相同的编码器输出，保证与原数据集 codec 一致（否则 merge 时
    # concatenate_video_files 走 stream copy，混用不同 codec 会失败）。
    codec_name = stream.codec_context.codec.name
    encoder, encoder_opts = {
        "av1": ("libsvtav1", {"crf": str(crf), "preset": "8"}),
        "h264": ("libx264", {"crf": str(crf), "preset": "fast"}),
        "hevc": ("libx265", {"crf": str(crf), "preset": "fast"}),
    }.get(codec_name, ("libsvtav1", {"crf": str(crf), "preset": "8"}))

    ostr = out.add_stream(encoder, rate=fps)
    ostr.width, ostr.height, ostr.pix_fmt = w, h, "yuv420p"
    ostr.options = encoder_opts

    n = 0
    for i, img in enumerate(_iter_frames(container)):
        g = offset + i
        ep = episode_index_for(bounds, g)
        bbox = boxes.get(ep) if ep is not None else None
        mask = None
        if render_mode == "dynamic":
            det_mask, det_bbox = detect_target(img, hsv_range, min_area)
            if det_bbox is not None:
                mask, bbox = det_mask, det_bbox
        if bbox is not None:
            img = render(img, mask, bbox, mode, pad=pad)
        vf = av.VideoFrame.from_ndarray(img, format="bgr24")
        for pkt in ostr.encode(vf):
            out.mux(pkt)
        n += 1
        if n % 2000 == 0:
            print(f"    {n} 帧已处理...")
    for pkt in ostr.encode():
        out.mux(pkt)
    out.close()
    container.close()
    tmp.replace(dst_mp4)
    print(f"    完成 {dst_mp4.name}：共 {n} 帧")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="给目标物体烘焙绿色视觉提示（mask/bbox）")
    ap.add_argument("--src", required=True, help="源数据集目录")
    ap.add_argument("--dst", required=True, help="输出数据集目录")
    ap.add_argument("--camera", default="up", choices=["up", "front", "both"])
    ap.add_argument("--color", default="blue", choices=list(COLOR_HSV.keys()))
    ap.add_argument("--render", default="bbox", choices=["bbox", "mask", "both"],
                    help="烘焙方式：bbox=绿色框 / mask=半透明填充 / both")
    ap.add_argument("--render-mode", default="static", choices=["static", "dynamic"],
                    help="static=每集一个固定框（推荐，和论文一致）；dynamic=逐帧检测")
    ap.add_argument("--min-area", type=int, default=300, help="最小目标面积（像素），过滤噪声")
    ap.add_argument("--detect-frames", type=int, default=40, help="static 模式每集检测前多少帧")
    ap.add_argument("--crf", type=int, default=30, help="编码质量，越小越清晰（LeRobot 默认 av1 用 30）")
    ap.add_argument("--pad", type=int, default=6, help="bbox 外扩像素（留白），避免绿线压住目标轮廓")
    ap.add_argument("--preview", type=int, default=0, help=">0 时只保存前 N 帧预览 PNG，不做完整标注")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    hsv_range = COLOR_HSV[args.color]

    bounds = read_episode_bounds(src)
    print(f"数据集 {src.name}：{len(bounds)} 集，目标色 {args.color}，相机 {args.camera}")

    cameras = ["up", "front"] if args.camera == "both" else [args.camera]

    if args.preview > 0:
        # 预览模式：不复制数据集，直接在前 N 帧上画框存图
        pdir = Path("/tmp/annotate_preview")
        import shutil as _sh

        _sh.rmtree(pdir, ignore_errors=True)
        pdir.mkdir(parents=True, exist_ok=True)
        for cam in cameras:
            vids = sorted((src / "videos" / f"observation.images.{cam}").rglob("*.mp4"))
            if not vids:
                print(f"  [warn] 相机 {cam} 没有视频文件")
                continue
            # 先算静态框
            boxes = detect_static_boxes(vids[0], bounds, hsv_range, args.min_area, args.detect_frames)
            print(f"  相机 {cam} 检测到的静态框（episode -> [x1,y1,x2,y2]）：")
            for ep in sorted(boxes)[:5]:
                print(f"    ep{ep}: {boxes[ep]}")
            # 只处理第一个视频的前 N 帧
            first = vids[0]
            import av as _av

            cont = _open_video(first)
            n = 0
            for i, img in enumerate(_iter_frames(cont)):
                if n >= args.preview:
                    break
                ep = episode_index_for(bounds, i)
                bbox = boxes.get(ep) if ep is not None else None
                if bbox is not None:
                    img = render(img, None, bbox, args.render, pad=args.pad)
                cv2.imwrite(str(pdir / f"{cam}_frame_{i:06d}.png"), img)
                n += 1
            cont.close()
        print(f"预览帧已保存到 {pdir}，请人工检查绿色框是否框住目标。")
        return

    # 完整标注：复制数据集（非破坏），原地覆盖选中相机视频
    if dst.exists():
        raise FileExistsError(f"{dst} 已存在，请换一个 --dst 或先删除。")
    shutil.copytree(src, dst)
    print(f"已复制数据集到 {dst}")

    for cam in cameras:
        vids = sorted((dst / "videos" / f"observation.images.{cam}").rglob("*.mp4"))
        if not vids:
            print(f"  [warn] 相机 {cam} 没有视频文件")
            continue
        print(f"相机 {cam}：{len(vids)} 个视频文件")
        offset = 0
        for v in vids:
            # 注意：检测/读取都用「源」视频（dst 是复制出来的副本，读 src 避免误读已覆盖文件）
            src_v = src / v.relative_to(dst)
            if args.render_mode == "static":
                boxes = detect_static_boxes(
                    src_v, bounds, hsv_range, args.min_area, args.detect_frames, offset=offset
                )
                print(f"  {v.name}：{len(boxes)} 集检测到目标框")
            else:
                boxes = {}
            process_video(src_v, v, bounds, boxes, hsv_range,
                          args.min_area, args.render, args.render_mode, None, args.crf, offset=offset, pad=args.pad)
            offset += get_frame_count(src_v)

    # 注意：不修改 info.json 的 codec 字段。合并数据集时 LeRobot 会对 features 做严格
    # 相等比较，改 codec 会导致与原数据集无法 merge。解码走的是对视频文件本身的探测
    # （get_video_info），info.json 里的 codec 只是元数据，无需改。
    print("完成。训练时把 --dataset.repo_id 指向", dst)


if __name__ == "__main__":
    main()
