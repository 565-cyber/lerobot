#!/usr/bin/env python3
"""up（静止俯视）相机**内参**标定 —— 补上 `calib/up_homography.json` 缺的 K 和畸变。

## 为什么需要另做一步

`up_homography.py` 只解了一张**桌面单应**：它把 z=0 平面上的点映到像素，精确、
可复现（0.039mm），但它**不含任何内参**。所以：

  * 物体有高度时（换物体要画的球/网格），视差只能靠名义 cam_h=700mm 去猜；
  * 以前从单应反解焦距被证伪 —— 单平面下 **f 与相机高度 D 退化**，
    `HANDOVER.md §5` 的结论是「K 与 D 不可辨识」。

**破退化的唯一办法：让标定板出现多个不同倾角。** 标定板平放在桌上滑来滑去
（哪怕换 20 个位置）仍然全是同一个平面，`calibrateCamera` 一样退化。必须**手拿着板
倾斜**（±20°~±45°），并且让角点铺满画面。这是本脚本存在的理由。

## 用法

    PY=/home/wzy/anaconda3/envs/lerobot1/bin/python

    # 1) 采集（有显示器：空格存一张；q 结束）
    $PY -I workplace/calibration/up_intrinsics.py capture

    #    没有显示器 / 想一个人举板：自动模式，每 1.5s 判一次，位姿够新才存
    $PY -I workplace/calibration/up_intrinsics.py capture --auto 20

    #    SSH 进来、看不到桌面：再开网页实时预览（浏览器打开 http://<本机IP>:8080/）
    $PY -I workplace/calibration/up_intrinsics.py capture --auto 20 --serve 8080

    # 2) 求解 + 质量报告（写 calib/up_intrinsics.json + figures/up_intrinsics_check.png）
    $PY -I workplace/calibration/up_intrinsics.py solve

    # 3)（可选）板平放桌面时，求 up 相机在 base 系的位姿
    $PY -I workplace/calibration/up_intrinsics.py extrinsic \
        --shot calib/up_intrinsics_shots/flat.png

**内参不受板格公制尺度影响**（3D 物点整体缩放只会缩放 tvec，不改变 f）；
但 `extrinsic` 的深度/高度**吃实测尺长**，所以打印/量尺那一步别省。

采集建议（`solve` 会逐条检查）：
  * 12~25 张，越多越好，但**倾角多样性比张数重要**；
  * 板要占画面 1/3~2/3，角点覆盖四个角（别都在中间）；
  * 至少 5 张倾角 > 15°，其中 2 张 > 30°；倾斜方向要转一圈（前后左右）；
  * 每换一个姿势停 1s 再拍，避开自动曝光/卷帘的瞬时抖动；
  * 画面里**不要有机械臂挡板**（up 相机是固定的，臂收起来就行）。

产物：
    calib/up_intrinsics_shots/        原图 + 每张的角点 JSON
    calib/up_intrinsics.json          K / dist / rms / 逐视角误差 / 倾角
    figures/up_intrinsics_check.png   检测点(绿) vs 重投影(红) 的肉眼验收图
    calib/up_camera_pose.json         （仅 extrinsic 模式）T_base_cam_up
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cam_id import CAMS, resolve                       # noqa: E402  (单一来源：CAMERAS.md)

CAL = HERE / "calib"
SHOTS = CAL / "up_intrinsics_shots"
FIG = HERE / "figures"
META = HERE / "board" / "charuco_board.json"

CAM_W, CAM_H = 640, 480          # 与数据集/其它标定一致；分辨率变了 K 就不能用


# ------------------------------------------------------------------ 板

def board_setup(measured_mm: float | None = None):
    """按 board/charuco_board.json 构造板 + 检测器。

    `measured_mm` 是板上 100mm 自检尺的**实测**长度；只影响公制尺度，
    不影响内参。不给就按标称（extrinsic 会被提醒）。
    """
    meta = json.loads(META.read_text())
    scale = (measured_mm / meta["checkerbar_mm_nominal"]) if measured_mm else 1.0
    sq = meta["square_mm_nominal"] / 1000.0 * scale
    mk = sq * meta["marker_mm_nominal"] / meta["square_mm_nominal"]
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, meta["dict"]))
    board = cv2.aruco.CharucoBoard(
        (meta["squares_x"], meta["squares_y"]), sq, mk, d)
    return board, cv2.aruco.CharucoDetector(board), meta, scale


def detect(det, frame):
    """返回 (charucoCorners Nx1x2, ids Nx1)。没检到就是 (None, None)。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None or len(ci) == 0:
        return None, None
    return cc.reshape(-1, 2).astype(np.float32), ci.flatten().astype(int)


def object_points(board, ids):
    """板系 3D 点（米，z=0），顺序与 charucoIds 一致。"""
    return board.getChessboardCorners()[ids].astype(np.float64)


def board_extent_m(meta, scale):
    sq = meta["square_mm_nominal"] / 1000.0 * scale
    return meta["squares_x"] * sq, meta["squares_y"] * sq


def quad_and_tilt(board, ids, corners, meta, scale):
    """板外框四角在图像里的位置 + 一个无量纲的倾角代理。

    倾角代理来自 H 的第三行：fronto-parallel 时 h20=h21=0。
    没有 K 时它不能给出真实角度，只能判断「这次和上次是不是不同的倾斜」。
    """
    obj = object_points(board, ids)[:, :2].astype(np.float32)
    img = corners.reshape(-1, 2).astype(np.float32)
    H, _ = cv2.findHomography(obj, img, 0)
    if H is None:
        return None, None
    w, h = board_extent_m(meta, scale)
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(rect, H).reshape(-1, 2)
    Hn = H / H[2, 2]
    tilt = float(math.hypot(Hn[2, 0], Hn[2, 1]))
    return quad, tilt


def novelty_px(quad, prev_quads) -> float:
    """与已存所有板外框的最小逐角平均距离（像素）。越大 = 新姿势越不一样。"""
    if not prev_quads:
        return float("inf")
    return min(float(np.linalg.norm(quad - q, axis=1).mean()) for q in prev_quads)


# ------------------------------------------------------------------ 相机

def open_camera(dev: Path, warmup: int = 15):
    """显式设 640×480 并丢掉开头几帧。

    ⚠ 沿用 `front_handeye.open_camera` 的教训：UVC 重新枚举后会回到 160×120，
    那个尺寸里 ChArUco 一个角点都检不出来，而脚本若只看「角点少就跳过」就会
    **静默产出废图**。所以这里必须核对实际分辨率并硬报错。
    """
    cap = cv2.VideoCapture(str(dev))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    for _ in range(warmup):
        cap.read()
    ok, fr = cap.read()
    if not ok or fr is None:
        cap.release()
        return None
    if (fr.shape[1], fr.shape[0]) != (CAM_W, CAM_H):
        print(f"[!] 实际分辨率 {fr.shape[1]}×{fr.shape[0]} ≠ {CAM_W}×{CAM_H}")
        cap.release()
        return None
    return cap


def camera_for(args) -> Path | None:
    if args.device:
        return Path(args.device)
    devs = resolve()
    if args.cam not in devs:
        print(f"[!] 找不到 {args.cam}（{CAMS[args.cam]}）—— 相机插了吗？")
        return None
    return devs[args.cam]


# ------------------------------------------------------------------ 采集

def load_quads() -> list[np.ndarray]:
    out = []
    for j in sorted(SHOTS.glob("shot_*.json")):
        r = json.loads(j.read_text())
        if r.get("quad") is not None:
            out.append(np.array(r["quad"], np.float32))
    return out


def next_index() -> int:
    idx = [int(p.stem.split("_")[1]) for p in SHOTS.glob("shot_*.json")
           if p.stem.split("_")[-1].isdigit()]
    return max(idx) + 1 if idx else 0


def save_shot(frame, corners, ids, quad, tilt) -> Path:
    SHOTS.mkdir(parents=True, exist_ok=True)
    i = next_index()
    png = SHOTS / f"shot_{i:03d}.png"
    js = SHOTS / f"shot_{i:03d}.json"
    cv2.imwrite(str(png), frame)
    js.write_text(json.dumps({
        "image": png.name,
        "corner_ids": ids.tolist(),
        "corners": corners.tolist(),
        "quad": None if quad is None else quad.tolist(),
        "tilt_proxy": tilt,
        "n_corners": int(len(ids)),
    }, indent=2))
    return png


def draw_overlay(frame, corners, ids, quad, n_saved, tilt, msg="",
                 hint="SPACE=save   d=undo   q=quit"):
    vis = frame.copy()
    if corners is not None:
        for (u, v) in corners:
            cv2.circle(vis, (int(u), int(v)), 3, (0, 255, 0), -1)
        if quad is not None:
            q = quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [q], True, (0, 200, 255), 2)
        cv2.putText(vis, f"corners={len(ids)}  tilt~{tilt:.4f}  saved={n_saved}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "NO BOARD", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, f"saved={n_saved}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, hint, (10, CAM_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if msg:
        cv2.putText(vis, msg, (10, CAM_H - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


def cmd_capture(args) -> int:
    dev = camera_for(args)
    if dev is None:
        return 2
    board, det, meta, scale = board_setup(args.measured_mm)
    if args.reset and SHOTS.exists():
        bak = SHOTS.with_name(f"{SHOTS.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.move(str(SHOTS), str(bak))
        print(f"[reset] 旧采集已移到 {bak}")
    SHOTS.mkdir(parents=True, exist_ok=True)

    prev = load_quads()
    print(f"[设备] {args.cam} → {dev}")
    print(f"[已有] {len(prev)} 张；新图从 shot_{next_index():03d} 开始")

    cap = open_camera(dev)
    if cap is None:
        print("[!] 打不开相机 / 分辨率不是 640×480")
        return 2

    state = None
    httpd = None
    if args.serve:
        if not args.auto:
            print("[!] --serve 只能和 --auto 一起用（否则没有主循环）")
            cap.release()
            return 2
        state = PreviewState()
        try:
            httpd = start_server(state, args.host, args.serve)
        except OSError as e:
            print(f"[!] 端口 {args.serve} 打不开：{e}")
            cap.release()
            return 2
        print(f"[web] 预览已启动：http://<本机IP>:{args.serve}/   "
              f"（Ctrl-C 或网页上的「结束」退出）", flush=True)

    if args.auto:
        try:
            return _capture_auto(args, cap, board, det, meta, scale, prev, state)
        finally:
            if httpd is not None:
                httpd.shutdown()

    print("\n[采集] 空格=存   d=删上一张   q=结束。"
          "\n  提示：倾斜板（±20°~45°），换倾斜方向，让角点铺满画面。\n")
    saved, msg = len(prev), ""
    last_quad = None
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                continue
            corners, ids = detect(det, fr)
            quad = tilt = None
            if corners is not None and len(ids) >= args.min_corners:
                quad, tilt = quad_and_tilt(board, ids, corners, meta, scale)
            cv2.imshow("up intrinsics", draw_overlay(
                fr, corners, ids, quad, saved, tilt or 0.0, msg))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d") and saved:
                shot = next_index() - 1
                for ext in (".png", ".json"):
                    p = SHOTS / f"shot_{shot:03d}{ext}"
                    if p.exists():
                        p.unlink()
                saved -= 1
                prev = load_quads()
                msg = f"undo shot_{shot:03d}"
                continue
            if key == ord(" "):
                if corners is None or len(ids) < args.min_corners:
                    msg = f"角点不足（{0 if ids is None else len(ids)}）—— 没存"
                    continue
                nov = novelty_px(quad, prev) if quad is not None else float("inf")
                if quad is not None:
                    if nov < args.min_move:
                        msg = f"姿势和已有的太像（{nov:.0f}px）—— 换个倾角再拍"
                        continue
                    prev.append(quad)
                p = save_shot(fr, corners, ids, quad, tilt)
                saved += 1
                msg = f"saved {p.name}  tilt~{tilt:.4f}"
                print(f"  [{saved:02d}] {p.name}  corners={len(ids)}  tilt~{tilt:.4f}"
                      f"  novelty={nov:.0f}px")
                last_quad = quad
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[完成] 共 {saved} 张 → {SHOTS}")
    _capture_hint(saved)
    return 0


class PreviewState:
    """采集循环和 HTTP 线程之间唯一的共享状态（图像 + 三个按钮标志）。

    只有最新一帧被保存：预览是「看现在」，不是录像。条件变量唤醒所有
    正在等新帧的 /stream 连接，不用轮询。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpeg = None
        self._seq = 0
        self._save = False
        self._undo = False
        self._quit = False

    def set_jpeg(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_jpeg(self, last_seq: int, timeout: float = 5.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._jpeg, self._seq

    def request(self, what: str) -> None:
        with self._lock:
            setattr(self, f"_{what}", True)

    def take(self, what: str) -> bool:
        with self._lock:
            v = getattr(self, f"_{what}")
            setattr(self, f"_{what}", False)
            return v


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>up camera calibration</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.5 sans-serif;margin:0;padding:12px}
 img{max-width:100%;border:1px solid #333;display:block}
 button{font-size:16px;padding:8px 18px;margin:8px 8px 0 0;border-radius:6px;
        border:1px solid #555;background:#222;color:#eee;cursor:pointer}
 button:hover{background:#333}
 code{color:#8f8}
</style></head><body>
<h3>up camera calibration</h3>
<img src="/stream.jpg" alt="stream">
<div>
 <button onclick="fetch('/save')">存这一张</button>
 <button onclick="fetch('/undo')">删上一张</button>
 <button onclick="fetch('/quit')">结束</button>
</div>
<p>画面上的 <code>saved=N</code> 是已存张数；<code>tilt~</code> 越大 = 板越倾斜。<br>
自动模式：摆出一个新姿势停一下，脚本会自己存；按钮可以强制存。<br>
目标：15~25 张，倾角 0°→±45°，前后左右都倾斜，角点铺满画面。</p>
</body></html>"""


def make_handler(state: PreviewState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # 别刷屏
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/snapshot.jpg":
                jpeg, _ = state.wait_jpeg(-1, timeout=0.2)
                if jpeg is None:
                    self._send(503, b"no frame yet", "text/plain")
                else:
                    self._send(200, jpeg, "image/jpeg")
            elif path == "/stream.jpg":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = -1
                while True:
                    jpeg, last = state.wait_jpeg(last)
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n")
                        self.wfile.write(jpeg + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
            elif path in ("/save", "/undo", "/quit"):
                state.request(path[1:])
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def start_server(state: PreviewState, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _publish(state: PreviewState | None, frame, corners, ids, quad,
             saved: int, tilt: float, msg: str) -> None:
    if state is None:
        return
    vis = draw_overlay(frame, corners, ids, quad, saved, tilt or 0.0, msg,
                       hint="web: new pose = auto save | /save force | /undo | /quit")
    ok, buf = cv2.imencode(".jpg", vis,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        state.set_jpeg(buf.tobytes())


def _capture_auto(args, cap, board, det, meta, scale, prev,
                  state: PreviewState | None = None) -> int:
    """无窗口自动采集：位姿够新才存。适合一个人举板 + 远程终端 / 网页预览。

    和 GUI 版的区别：读帧按相机帧率跑（网页才是流畅的），但**判定/存图**
    每 `--interval` 秒最多一次 —— 否则同一姿势会被连存十几张。
    """
    saved = len(prev)
    target = args.auto
    print(f"[auto] 目标总张数 {target}（已有 {saved}，还需 {max(0, target-saved)}），"
          f"最长 {args.timeout:.0f}s；每 {args.interval:.1f}s 判一次。Ctrl-C 结束。\n",
          flush=True)
    t0 = time.time()
    last_eval = 0.0
    msg = ""
    corners = ids = quad = None
    tilt = 0.0
    try:
        while saved < target and time.time() - t0 < args.timeout:
            if state is not None and state.take("quit"):
                print("\n[auto] 网页点了结束", flush=True)
                break
            if state is not None and state.take("undo") and saved:
                shot = next_index() - 1
                for ext in (".png", ".json"):
                    p = SHOTS / f"shot_{shot:03d}{ext}"
                    if p.exists():
                        p.unlink()
                saved -= 1
                prev = load_quads()
                msg = f"undo shot_{shot:03d}"
                print(f"  [undo] shot_{shot:03d}（剩 {saved} 张）", flush=True)

            ok, fr = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            corners, ids = detect(det, fr)
            quad = tilt = None
            if corners is not None and len(ids) >= args.min_corners:
                quad, tilt = quad_and_tilt(board, ids, corners, meta, scale)

            force = state.take("save") if state is not None else False
            now = time.time()
            do_eval = force or (now - last_eval) >= args.interval

            if do_eval:
                last_eval = now
                if corners is None or len(ids) < args.min_corners:
                    msg = f"no board (corners={0 if ids is None else len(ids)})"
                elif quad is None:
                    msg = "board found, quad failed - change angle"
                else:
                    nov = novelty_px(quad, prev)
                    if force or nov >= args.min_move:
                        p = save_shot(fr, corners, ids, quad, tilt)
                        prev.append(quad)
                        saved += 1
                        msg = f"saved {p.name}  ({saved}/{target})"
                        print(f"  [{saved:02d}/{target}] {p.name}"
                              f"  corners={len(ids)}  tilt~{tilt:.4f}"
                              f"  novelty={nov:.0f}px", flush=True)
                    else:
                        msg = f"too similar ({nov:.0f}px) - change tilt"

            _publish(state, fr, corners, ids, quad, saved, tilt or 0.0, msg)
            # 相机 read() 自己会阻塞到下一帧，这里不用再 sleep
    except KeyboardInterrupt:
        print("\n[auto] 用户中断", flush=True)
    finally:
        cap.release()

    print(f"\n[完成] 共 {saved} 张 → {SHOTS}", flush=True)
    _capture_hint(saved)
    return 0 if saved else 1


def _capture_hint(n: int) -> None:
    if n < 10:
        print(f"[!] 只有 {n} 张，内参不稳。建议 15~25 张，且**倾角多样**。")
    print("下一步：up_intrinsics.py solve")


# ------------------------------------------------------------------ 求解

def load_shots() -> list[dict]:
    recs = []
    for js in sorted(SHOTS.glob("shot_*.json")):
        r = json.loads(js.read_text())
        r["png"] = SHOTS / r["image"]
        recs.append(r)
    return recs


def tilt_deg_from_R(R) -> float:
    """板法线（板系 +z）与相机光轴（+z）的夹角，取锐角。

    板平放、相机正下视 ⇒ 0°；真实倾角就是这个量的补角（倾 30° ⇒ 这里 30°）。
    """
    n = R[:, 2]
    c = float(np.clip(abs(n[2]), 0.0, 1.0))
    return math.degrees(math.acos(c))


def reprojection_errors(objps, imgps, K, dist, rvs, tvs):
    errs = []
    for o, im, rv, tv in zip(objps, imgps, rvs, tvs):
        pr, _ = cv2.projectPoints(o, rv, tv, K, dist)
        errs.append(float(np.linalg.norm(pr.reshape(-1, 2) - im.reshape(-1, 2),
                                         axis=1).mean()))
    return errs


def coverage_report(imgps, w=CAM_W, h=CAM_H) -> dict:
    pts = np.vstack([im.reshape(-1, 2) for im in imgps])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    # 3×3 宫格里有几格出现过角点
    cells = np.zeros((3, 3), int)
    for x, y in pts:
        cx = min(2, max(0, int(x / (w / 3))))
        cy = min(2, max(0, int(y / (h / 3))))
        cells[cy, cx] += 1
    return {
        "bbox_px": [float(x0), float(y0), float(x1), float(y1)],
        "bbox_frac": [float((x1 - x0) / w), float((y1 - y0) / h)],
        "grid_cells_nonempty": int((cells > 0).sum()),
        "grid_cells_total": 9,
    }


def cmd_solve(args) -> int:
    recs = load_shots()
    if len(recs) < 4:
        raise SystemExit(f"{SHOTS} 里只有 {len(recs)} 张 —— 先 capture")
    board, _, meta, scale = board_setup(args.measured_mm)

    objps, imgps, tilts, names = [], [], [], []
    for r in recs:
        ids = np.array(r["corner_ids"], int)
        for drop in args.drop:
            ids = ids[ids != drop]
        if len(ids) < args.min_corners:
            print(f"  [跳过] {r['image']}: 有效角点 {len(ids)}")
            continue
        corners = np.array(r["corners"], np.float32)
        if len(corners) != len(r["corner_ids"]):
            print(f"  [跳过] {r['image']}: corners/ids 长度不一致")
            continue
        keep = np.isin(np.array(r["corner_ids"], int), ids)
        img = corners[keep]
        objps.append(object_points(board, ids).reshape(-1, 1, 3).astype(np.float32))
        imgps.append(img.reshape(-1, 1, 2).astype(np.float32))
        names.append(r["image"])

    if len(objps) < 6:
        raise SystemExit("有效视角 < 6，无法标定。")

    size = (CAM_W, CAM_H)

    def _calib(flags):
        return cv2.calibrateCamera(objps, imgps, size, None, None, flags=flags)

    # 主解：方形像素（fx=fy）。这与 front_intrinsics.json 的做法一致。
    rms5, K5, d5m, rvs5, tvs5 = _calib(cv2.CALIB_FIX_ASPECT_RATIO)

    # 受限变体。5 参数在「角点只覆盖中间、畸变本身很弱」的数据上会过拟合
    # （k2/k3 互相抵消，rms 只降一点点，但画面外推会炸）。
    rms4, K4, d4m, rvs4, tvs4 = _calib(
        cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3)
    rms3, K3, d3m, rvs3, tvs3 = _calib(
        cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)

    # 诊断解：不固定长宽比。两者差 >2% 说明数据有问题（镜头/板不平/角点误检）。
    rms_free, Kf, distf, _, _ = _calib(0)
    ratio = float(Kf[0, 0] / Kf[1, 1]) if Kf[1, 1] else float("nan")

    d5 = d5m.ravel()
    overfit = abs(d5[1]) > 0.5 or abs(d5[4]) > 5
    variant = args.variant
    if variant == "auto":
        variant = "3" if overfit else "5"
    if variant == "3":
        K, dist, rms, rvs, tvs = K3, d3m, rms3, rvs3, tvs3
        selected = "fix_k2_k3 (k1 + p1,p2)"
    elif variant == "4":
        K, dist, rms, rvs, tvs = K4, d4m, rms4, rvs4, tvs4
        selected = "fix_k3 (k1,k2 + p1,p2)"
    else:
        K, dist, rms, rvs, tvs = K5, d5m, rms5, rvs5, tvs5
        selected = "full_5"
    errs = reprojection_errors(objps, imgps, K, dist, rvs, tvs)

    for rv, tv in zip(rvs, tvs):
        tilts.append(tilt_deg_from_R(cv2.Rodrigues(rv)[0]))

    cov = coverage_report(imgps)
    n_t15 = sum(t > 15 for t in tilts)
    n_t30 = sum(t > 30 for t in tilts)
    n_t45 = sum(t > 45 for t in tilts)

    d = dist.ravel()
    warnings = []
    if len(objps) < 10:
        warnings.append(f"视角只有 {len(objps)} 个（建议 ≥12，最好 15~25）")
    if max(tilts) < 15:
        warnings.append("所有板几乎都平放：f 与高度退化，K 不可信")
    if n_t15 < 5:
        warnings.append(f"倾角 >15° 的视角只有 {n_t15} 个（建议 ≥5）")
    if n_t30 < 2:
        warnings.append(f"倾角 >30° 的视角只有 {n_t30} 个（建议 ≥2）")
    if cov["grid_cells_nonempty"] < 6:
        warnings.append(f"角点只覆盖 {cov['grid_cells_nonempty']}/9 个宫格，"
                        "四个角都要拍到")
    if rms > 1.0:
        warnings.append(f"rms {rms:.2f}px > 1.0px")
    if not math.isnan(ratio) and abs(ratio - 1) > 0.02:
        warnings.append(f"自由解 fx/fy = {ratio:.3f}（偏离 1 超过 2%）")
    if overfit:
        warnings.append(
            f"5 参数畸变疑似过拟合（k2={d5[1]:+.2f}, k3={d5[4]:+.2f}）—— "
            f"已选用受限变体 {selected}（rms {rms:.3f}px vs 5 参数 {rms5:.3f}px）")

    print("\n================ up 相机内参 ================")
    print(f"  视角        {len(objps)}")
    print(f"  采用变体    {selected}")
    print(f"  rms         {rms:.3f} px   （自由长宽比解 {rms_free:.3f} px）")
    print(f"  fx = fy     {K[0,0]:.2f} px   （自由解 fx={Kf[0,0]:.2f}, fy={Kf[1,1]:.2f}）")
    print(f"  cx, cy      {K[0,2]:.2f}, {K[1,2]:.2f} px")
    print(f"  k1,k2       {d[0]:+.4f}, {d[1]:+.4f}")
    print(f"  p1,p2       {d[2]:+.5f}, {d[3]:+.5f}")
    print(f"  k3          {d[4]:+.4f}")
    d4v, d3v = d4m.ravel(), d3m.ravel()
    print(f"  三个解      5 参数 rms {rms5:.3f}  k1={d5[0]:+.4f} k2={d5[1]:+.4f} k3={d5[4]:+.4f}")
    print(f"              4 参数 rms {rms4:.3f}  k1={d4v[0]:+.4f} k2={d4v[1]:+.4f}")
    print(f"              3 参数 rms {rms3:.3f}  k1={d3v[0]:+.4f}")
    print(f"  倾角°       " + " ".join(f"{t:.0f}" for t in tilts))
    print(f"  倾角统计    >15°: {n_t15}   >30°: {n_t30}   >45°: {n_t45}")
    print(f"  覆盖        bbox {cov['bbox_frac'][0]*100:.0f}% × "
          f"{cov['bbox_frac'][1]*100:.0f}% of frame, "
          f"{cov['grid_cells_nonempty']}/9 宫格")
    print(f"  逐视角 rms  " + " ".join(f"{e:.2f}" for e in errs))
    if warnings:
        print("\n  [!] 质量警告：")
        for w in warnings:
            print(f"      - {w}")
    else:
        print("\n  [OK] 质量检查全过")

    out = {
        "camera": "camera_A (up, no serial)",
        "K": K.tolist(),
        "dist": d.tolist(),
        "image_size": list(size),
        "rms_px": float(rms),
        "selected_variant": variant,
        "selected_variant_name": selected,
        "rms_free_aspect_px": float(rms_free),
        "K_free_aspect": Kf.tolist(),
        "dist_free_aspect": distf.ravel().tolist(),
        "variants": {
            "full_5": {"K": K5.tolist(), "dist": d5.tolist(),
                       "rms_px": float(rms5)},
            "fix_k3": {"K": K4.tolist(), "dist": d4m.ravel().tolist(),
                       "rms_px": float(rms4)},
            "fix_k2_k3": {"K": K3.tolist(), "dist": d3m.ravel().tolist(),
                          "rms_px": float(rms3)},
        },
        "per_view_rms_px": errs,
        "views": names,
        "tilt_deg": tilts,
        "coverage": cov,
        "n_views": len(objps),
        "warnings": warnings,
        "board": {
            "square_mm_nominal": meta["square_mm_nominal"],
            "measured_mm": args.measured_mm,
            "scale": scale,
            "note": "内参不受板格公制尺度影响；extrinsic 的深度才受影响",
        },
        "note": "up 相机内参。桌面单应仍用 calib/up_homography.json；"
                "两者独立。标定板必须倾斜采集，否则 f 与高度退化。",
    }
    (CAL / "up_intrinsics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[save] {CAL/'up_intrinsics.json'}")

    _write_check_montage(objps, imgps, K, dist, rvs, tvs, names)
    print(f"[save] {FIG/'up_intrinsics_check.png'}")
    return 0 if not warnings else 1


def _write_check_montage(objps, imgps, K, dist, rvs, tvs, names) -> None:
    """绿=检测角点，红=重投影。四列拼图，肉眼能看出系统性偏差。"""
    tiles = []
    for o, im, rv, tv, name in zip(objps, imgps, rvs, tvs, names):
        p = SHOTS / name
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        pr, _ = cv2.projectPoints(o, rv, tv, K, dist)
        pr = pr.reshape(-1, 2)
        for (u, v) in im.reshape(-1, 2):
            cv2.drawMarker(fr, (int(u), int(v)), (0, 255, 0),
                           cv2.MARKER_CROSS, 7, 1)
        for (u, v) in pr:
            cv2.circle(fr, (int(u), int(v)), 2, (0, 0, 255), -1)
        cv2.putText(fr, name, (6, CAM_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        tiles.append(cv2.resize(fr, (CAM_W // 2, CAM_H // 2)))
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    blank = np.zeros((CAM_H // 2, CAM_W // 2, 3), np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    FIG.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIG / "up_intrinsics_check.png"), grid)


# ------------------------------------------------------------------ 外参（可选）

def cmd_extrinsic(args) -> int:
    """板**平放在桌面**且**位置与 `board_to_base.json` 解算时一致**时，
    求 up 相机在 base 系的位姿 T_base_cam。

    ⚠ 这一步的前提比内参苛刻得多，务必读警告：
      * `board_to_base.json` 里的 T_base_board 是**某一次摆板**解出来的。
        板挪过位置/转过角度，这个数就作废 —— 此时请先重跑
        `solve_flat_board.py`（会动臂、需要腕相机），本脚本不替你重解板位。
      * 平放板在正下视下做 PnP，深度完全由**已知板格尺度**撑起来，
        所以 `--measured-mm` 的实测值直接进高度 D。量错了 D 就错。

    有了这一步，`HANDOVER.md §5` 里「up 相机高度不可辨识」就被真正解开：
    f 已由内参钉住，PnP 的 tvec 不再和 f 退化。
    """
    ip = CAL / "up_intrinsics.json"
    bp = Path(args.board_to_base)
    if not bp.is_absolute():
        bp = CAL / bp
    if not ip.exists():
        raise SystemExit(f"先跑 solve：缺 {ip}")
    if not bp.exists():
        raise SystemExit(f"缺 {bp}")

    intr = json.loads(ip.read_text())
    b2b = json.loads(bp.read_text())
    K = np.array(intr["K"], float)
    dist = np.array(intr["dist"], float)
    T_base_board = np.array(b2b["T_base_board"], float)

    board, det, meta, scale = board_setup(args.measured_mm)
    shot = Path(args.shot)
    if not shot.is_absolute():
        shot = CAL / shot
    fr = cv2.imread(str(shot))
    if fr is None:
        raise SystemExit(f"读不到 {shot}")
    corners, ids = detect(det, fr)
    if corners is None or len(ids) < 8:
        raise SystemExit("这张图里角点不足 —— 板平放、完整、别被臂挡住")

    obj = object_points(board, ids).reshape(-1, 1, 3).astype(np.float32)
    img = corners.reshape(-1, 1, 2).astype(np.float32)
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise SystemExit("solvePnP 失败")
    R, _ = cv2.Rodrigues(rv)
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = tv.ravel()
    T_base_cam = T_base_board @ np.linalg.inv(T_cam_board)

    # 重投影残差（PnP 只有 6 个自由度，这个数应当远小于内参 rms? 不一定）
    pr, _ = cv2.projectPoints(obj, rv, tv, K, dist)
    pnp_rms = float(np.linalg.norm(
        pr.reshape(-1, 2) - img.reshape(-1, 2), axis=1).mean())

    axis_base = T_base_cam[:3, :3][:, 2]          # 相机 +z 在 base 系的方向
    cand_mm = T_base_cam[:3, 3] * 1000.0
    D_mm = float(cand_mm[2])                      # base z 朝上 ⇒ 相机在桌面上方，t_z 即高度

    print("\n============ up 相机在 base 系的位姿 ============")
    print("  T_base_cam (m):")
    for row in T_base_cam:
        print("   ", " ".join(f"{v: .6f}" for v in row))
    print(f"  平移 (mm)      {cand_mm.round(2).tolist()}")
    tilt_deg = math.degrees(math.acos(float(np.clip(-axis_base[2], -1.0, 1.0))))
    print(f"  高度 D         {D_mm:.1f} mm   （= t_z，桌面按 z=0）")
    print(f"  光轴方向(base) {axis_base.round(4).tolist()}  "
          f"（与垂直向下夹角 {tilt_deg:.1f}°）")
    print(f"  PnP 重投影     {pnp_rms:.3f} px")
    if tilt_deg > 45:
        print("  [!] 光轴与垂直向下夹角 >45° —— 板没放平，或板位/尺度有问题")
    if abs(D_mm - 700) > 50:
        print(f"  [!] D={D_mm:.0f}mm 与名义 700mm 差 >50mm —— "
              "核对板格实测尺长与 board_to_base 的板位")

    out = {
        "T_base_cam": T_base_cam.tolist(),
        "translation_mm": cand_mm.tolist(),
        "D_mm": D_mm,
        "optical_axis_in_base": axis_base.tolist(),
        "pnp_rms_px": pnp_rms,
        "shot": shot.name,
        "intrinsics": "calib/up_intrinsics.json",
        "board_to_base": str(bp.relative_to(HERE)),
        "measured_mm": args.measured_mm,
        "caveat": "仅当板与 board_to_base.json 解算时处于同一物理位姿才有效；"
                  "板挪过就必须重跑 solve_flat_board.py",
    }
    (CAL / "up_camera_pose.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[save] {CAL/'up_camera_pose.json'}")
    return 0


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cam", default="camera_A",
                       help="camera_A = 无序列号 = up（见 CAMERAS.md）")
        p.add_argument("--device", default=None,
                       help="直接给 /dev/videoN（by-id 不存在时的退路）")
        p.add_argument("--measured-mm", type=float, default=None,
                       help="板上 100mm 尺的实测长度；只影响 extrinsic 深度")
        p.add_argument("--min-corners", type=int, default=8,
                       help="一张图至少检出多少 charuco 角点才算数")

    p = sub.add_parser("capture", help="交互采集（空格存图，q 结束）")
    common(p)
    p.add_argument("--auto", type=int, default=0,
                   help="无窗口自动模式：**目标总张数**（含已有），达到 N 张就停")
    p.add_argument("--interval", type=float, default=1.5,
                   help="自动模式每次判定的间隔（秒）")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="自动模式最长运行时间（秒）")
    p.add_argument("--min-move", type=float, default=30.0,
                   help="与已有姿势的最小差异（像素），太小就拒存")
    p.add_argument("--reset", action="store_true",
                   help="把旧采集移到带时间戳的备份目录后重来")
    p.add_argument("--serve", type=int, default=0, metavar="PORT",
                   help="无窗口时开网页实时预览（需配合 --auto），例如 8080")
    p.add_argument("--host", default="0.0.0.0",
                   help="网页服务绑定地址，默认 0.0.0.0（只给别人看不放心就 127.0.0.1）")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("solve", help="用已采集的图求解内参")
    common(p)
    p.add_argument("--drop", type=int, nargs="*", default=[],
                   help="丢弃这些 charuco id（个别角点误检时用）")
    p.add_argument("--variant", choices=["auto", "5", "4", "3"], default="auto",
                   help="最终采用哪个畸变模型：auto=5 参数过拟合时自动退到 3 参数")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("extrinsic", help="板平放时求 T_base_cam（可选）")
    common(p)
    p.add_argument("--shot", required=True, help="板平放的 up 原图")
    p.add_argument("--board-to-base", default="board_to_base.json")
    p.set_defaults(func=cmd_extrinsic)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
