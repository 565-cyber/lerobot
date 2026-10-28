"""扫一遍整条 episode：黄块（=被夹的目标）在 up / 腕两个视图里各看得见多少，
按夹爪开合分阶段。

目的是回答「后期被夹走的能替换吗」：替换需要**两个视图里都定位到方块**。
up 视图给的是 base 系的绝对位置（单应 z=0 精确），腕视图给的是像素锚点。
只要有一个断了，那一段就换不了 —— 除非换成别的定位来源。

输出：每 `--step` 帧一行，并给每个阶段的可见率统计。

    python workplace/calibration/phase_visibility_scan.py --episode 0 --step 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workplace" / "so101_urdf"))

import arm as A                                                     # noqa: E402
import ball_swap as bs                                              # noqa: E402


def best_blob(mask, area_range=(80, 12000)):
    """掩码里最像方块的那一块。返回 (面积, 形心) 或 (0, None)。"""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = (0, None)
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        if not (area_range[0] <= a <= area_range[1]):
            continue
        if not (0.35 <= w / max(h, 1) <= 2.9):        # 方块该是方的
            continue
        if a > best[0]:
            best = (a, cent[k])
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newdata1")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--target", choices=("yellow", "blue"), default="yellow")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(a.dataset, root=str(REPO / "dataset" / a.dataset))
    g = bs.load_geom()
    tgt_up, _, name, _ = bs.cube_mask_fns(g, "up", a.target)
    tgt_wr, _, _, _ = bs.cube_mask_fns(g, "wrist", a.target)

    ep = ds.meta.episodes[a.episode]
    i0 = int(ep["dataset_from_index"]) if isinstance(ep, dict) else int(ep.dataset_from_index)
    i1 = int(ep["dataset_to_index"]) if isinstance(ep, dict) else int(ep.dataset_to_index)
    print(f"ep{a.episode}: 帧 [{i0}, {i1})  共 {i1 - i0} 帧, 每 {a.step} 帧取一次")

    rows = []
    print(f"{'frame':>6} {'grip':>6} {'up面积':>7} {'腕面积':>7}  up质心      腕质心")
    for i in range(i0, i1, a.step):
        fr = ds[i]
        up = cv2.cvtColor(np.clip(fr["observation.images.up"].permute(1, 2, 0).numpy() * 255,
                                  0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        fg = cv2.cvtColor(np.clip(fr["observation.images.front"].permute(1, 2, 0).numpy() * 255,
                                  0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        st = fr["observation.state"].numpy().astype(float)
        grip = float(st[list(A.MOTORS).index("gripper")])
        au, cu = best_blob(tgt_up(up))
        aw, cw = best_blob(tgt_wr(fg))
        cu_s = "" if cu is None else f"({cu[0]:.0f},{cu[1]:.0f})"
        cw_s = "" if cw is None else f"({cw[0]:.0f},{cw[1]:.0f})"
        print(f"{i:>6} {grip:>6.1f} {au:>7} {aw:>7}  {cu_s:<12} {cw_s}")
        rows.append((i, grip, au, aw))

    print("\n--- 按夹爪开合分档 ---")
    # 归一化夹爪：小=合拢（夹住），大=张开。分档只为看趋势。
    for lo, hi, tag in ((0, 12, "合拢/夹紧"), (12, 22, "半开"), (22, 101, "张开")):
        seg = [r for r in rows if lo <= r[1] < hi]
        if not seg:
            continue
        n = len(seg)
        vu = sum(1 for r in seg if r[2] > 0)
        vw = sum(1 for r in seg if r[3] > 0)
        print(f"  夹爪 {lo:>3}-{hi:<3} ({tag:<8}) n={n:<4} "
              f"up 可见 {100*vu/n:5.1f}%   腕可见 {100*vw/n:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
