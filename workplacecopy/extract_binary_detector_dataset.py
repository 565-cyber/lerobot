"""
Extract frames from lerobot videos and label them for 3 binary detectors.

Each frame gets 3 binary labels:
  - yellow_on_table: 1 if yellow block is visible on the table
  - blue_on_table:   1 if blue block is visible on the table
  - lid_open:        1 if the trash can lid is open

Output: 3 ImageFolder datasets (positive/negative folders) under binary_detector_dataset/

Labeling strategy (temporal heuristics since no per-frame annotations):

  newdata1 (Pick up yellow block):
    Segment 0-33%:  yellow=1, blue=1, lid=0   (State 1: both blocks present)
    Segment 33-66%: yellow=0, blue=1, lid=0   (State 2: yellow removed)
    Segment 66-100%: yellow=0, blue=1, lid=1  (State 2→3: lid open)

  newdata (Pick up blue block + blue block on table scenes):
    For "pick up blue block" task videos:
      Segment 0-50%:  yellow=0, blue=1, lid=0
      Segment 50-100%: yellow=0, blue=0, lid=1
    For "blue block on table" scene (same videos, filtered for blue visibility):
      Entire video: yellow=0, blue=1, lid=0
    Note: newdata appears TWICE — as task video AND as blue-block scene.
          We extract once but label frames based on temporal position,
          AND also add static blue-block frames from the visibility-filtered set.

  newdata2 (Close trash can):
    Segment 0-50%:  yellow=0, blue=0, lid=1   (lid initially open)
    Segment 50-100%: yellow=0, blue=0, lid=0  (lid closing / closed)

We cap positive/negative counts per detector to keep datasets roughly balanced.
"""
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
from PIL import Image


# ── Paths ──────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/zjq/lerobot/dataset")
OUTPUT_DIR = Path("/home/zjq/lerobot/workplace/binary_detector_dataset")
TMP_DIR = Path("/home/zjq/lerobot/workplace/binary_detector_tmp")

CAMERA_NAME = "observation.images.up"
FFMPEG = "/home/zjq/miniconda3/envs/lerobot/bin/ffmpeg"
FFPROBE = "/home/zjq/miniconda3/envs/lerobot/bin/ffprobe"

# ── Parameters ─────────────────────────────────────────────────────────
FRAME_STRIDE = 5
MAX_PER_CLASS = 300        # max frames per (detector, split, positive/negative)
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ── Video configs: (dataset_name, segment_start, segment_end, yellow, blue, lid) ──
# Each defines a temporal segment of a video and the binary labels for frames in it.
VIDEO_SEGMENTS = [
    # newdata1: Pick up yellow block
    ("newdata1", 0.00, 0.33, 1, 1, 0),
    ("newdata1", 0.33, 0.66, 0, 1, 0),
    ("newdata1", 0.66, 1.00, 0, 1, 1),

    # newdata: Pick up blue block (task videos)
    ("newdata",  0.00, 0.50, 0, 1, 0),
    ("newdata",  0.50, 1.00, 0, 0, 1),

    # newdata2: Close trash can
    ("newdata2", 0.00, 0.50, 0, 0, 1),
    ("newdata2", 0.50, 1.00, 0, 0, 0),
]


def run_cmd(cmd):
    """Run a command, print it, raise on failure."""
    print("  " + " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Command failed")
    return result


def find_videos(dataset_dir: Path) -> list[Path]:
    video_dir = dataset_dir / "videos" / CAMERA_NAME
    videos = sorted(video_dir.glob("chunk-*/file-*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No videos: {video_dir}/chunk-*/file-*.mp4")
    return videos


def get_video_frame_count(video_path: Path) -> int:
    """Get total frame count of a video using ffprobe."""
    cmd = [
        FFPROBE,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        # Fallback: use ffprobe to get duration * fps
        cmd2 = [
            FFPROBE,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ]
        r2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(r2.stdout.strip())
        return int(duration * 30)  # assume 30fps


def extract_frames(video_path: Path, out_dir: Path, prefix: str, stride: int = FRAME_STRIDE):
    """Extract frames from a single video using ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / f"{prefix}_%06d.jpg"

    cmd = [
        FFMPEG,
        "-y", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "none",
        "-i", str(video_path),
        "-vf", f"select='not(mod(n\\,{stride}))'",
        "-vsync", "vfr",
        "-q:v", "2",
        str(out_pattern),
    ]
    run_cmd(cmd)


def extract_all_videos():
    """Extract frames from all videos into TMP_DIR, organized by dataset."""
    datasets_seen = set()
    for dataset_name, *_ in VIDEO_SEGMENTS:
        datasets_seen.add(dataset_name)

    for dataset_name in sorted(datasets_seen):
        dataset_dir = DATASET_ROOT / dataset_name
        if not dataset_dir.exists():
            print(f"SKIP: {dataset_dir} not found")
            continue

        videos = find_videos(dataset_dir)
        print(f"\n{dataset_name}: {len(videos)} videos")

        out_dir = TMP_DIR / dataset_name
        for vi, vp in enumerate(videos):
            prefix = f"{dataset_name}_{vi:03d}"
            extract_frames(vp, out_dir, prefix)


def label_and_distribute():
    """Label each extracted frame and copy to the 3 detector ImageFolder datasets."""
    # Counters for balancing
    counters: dict[str, dict[str, dict[str, int]]] = {}  # detector -> split -> pos/neg -> count

    detectors = ["detector_yellow_on_table", "detector_blue_on_table", "detector_lid_open"]
    label_keys = ["yellow", "blue", "lid"]

    for det in detectors:
        counters[det] = {"train": {"positive": 0, "negative": 0},
                         "val":   {"positive": 0, "negative": 0}}

    # Collect all frame paths with their labels
    all_labeled: list[dict] = []

    for dataset_name, seg_start, seg_end, yellow, blue, lid in VIDEO_SEGMENTS:
        frames_dir = TMP_DIR / dataset_name
        if not frames_dir.exists():
            continue

        all_frames = sorted(frames_dir.glob("*.jpg"))
        if not all_frames:
            continue

        n = len(all_frames)
        start_idx = int(n * seg_start)
        end_idx = int(n * seg_end)
        seg_frames = all_frames[start_idx:end_idx]

        for fp in seg_frames:
            all_labeled.append({
                "path": fp,
                "yellow": yellow,
                "blue": blue,
                "lid": lid,
            })

        print(f"  {dataset_name} [{seg_start:.2f}-{seg_end:.2f}]: "
              f"{len(seg_frames)} frames  (yellow={yellow}, blue={blue}, lid={lid})")

    print(f"\nTotal labeled frames: {len(all_labeled)}")

    # Shuffle and split train/val
    random.shuffle(all_labeled)
    train_n = int(len(all_labeled) * TRAIN_RATIO)

    for split_name, split_frames in [("train", all_labeled[:train_n]),
                                      ("val", all_labeled[train_n:])]:
        for detector, key in zip(detectors, label_keys):
            pos_dir = OUTPUT_DIR / detector / split_name / "positive"
            neg_dir = OUTPUT_DIR / detector / split_name / "negative"
            pos_dir.mkdir(parents=True, exist_ok=True)
            neg_dir.mkdir(parents=True, exist_ok=True)

            for frame_info in split_frames:
                label = frame_info[key]
                target_dir = pos_dir if label == 1 else neg_dir
                label_str = "positive" if label == 1 else "negative"

                c = counters[detector][split_name][label_str]
                if c >= MAX_PER_CLASS:
                    continue

                idx = c + 1
                dst = target_dir / f"{idx:06d}.jpg"
                shutil.copy2(frame_info["path"], dst)
                counters[detector][split_name][label_str] = c + 1

    # Print summary
    print("\n" + "=" * 60)
    print("Final dataset counts:")
    print("=" * 60)
    for detector in detectors:
        print(f"\n{detector}:")
        for split in ["train", "val"]:
            p = counters[detector][split]["positive"]
            n = counters[detector][split]["negative"]
            print(f"  {split}: positive={p}, negative={n}")


def filter_blue_block_visible(images: list) -> list:
    """Keep only images where the blue block is clearly visible (not occluded)."""
    kept = []
    for img_path in images:
        arr = np.array(Image.open(img_path))
        r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)
        blue_mask = (b > r * 1.15) & (b > g * 1.1) & (b > 40)
        if blue_mask.sum() > 1000:
            kept.append(img_path)
    return kept


def extract_blue_block_static():
    """
    Extract additional "blue block on table" static frames from newdata.
    These are frames where blue block is clearly visible — useful as extra
    positives for detector_blue_on_table.
    """
    dataset_dir = DATASET_ROOT / "newdata"
    if not dataset_dir.exists():
        return

    videos = find_videos(dataset_dir)
    out_dir = TMP_DIR / "newdata_blue_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract with tighter stride (2) to get more frames
    for vi, vp in enumerate(videos):
        prefix = f"bluestatic_{vi:03d}"
        extract_frames(vp, out_dir, prefix, stride=2)

    all_frames = sorted(out_dir.glob("*.jpg"))
    print(f"\nBlue static: extracted {len(all_frames)} raw frames")

    visible = filter_blue_block_visible(all_frames)
    print(f"Blue static: {len(visible)} after blue-block filter")

    # Cap
    if len(visible) > 150:
        random.shuffle(visible)
        keep = set(visible[:150])
        for img in visible:
            if img not in keep:
                img.unlink()
    visible = sorted(out_dir.glob("*.jpg"))

    # Add to detector_blue_on_table only (yellow=0, blue=1, lid=0)
    # Cap to MAX_PER_CLASS to maintain balance with negatives
    random.shuffle(visible)
    visible = visible[:MAX_PER_CLASS]  # cap total
    train_n = int(len(visible) * TRAIN_RATIO)

    detector = "detector_blue_on_table"
    for split_name, split_frames in [("train", visible[:train_n]),
                                      ("val", visible[train_n:])]:
        pos_dir = OUTPUT_DIR / detector / split_name / "positive"
        pos_dir.mkdir(parents=True, exist_ok=True)

        # Only add up to MAX_PER_CLASS
        existing = len(list(pos_dir.glob("*.jpg")))
        room = max(0, MAX_PER_CLASS - existing)
        to_add = split_frames[:room]
        for i, src in enumerate(to_add, start=existing + 1):
            dst = pos_dir / f"{i:06d}.jpg"
            shutil.copy2(src, dst)

        print(f"Blue static → {detector}/{split_name}/positive: {len(to_add)} added (capped)")


def main():
    random.seed(RANDOM_SEED)

    # Clean up
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    print("=" * 60)
    print("Phase 1: Extract frames from all videos")
    print("=" * 60)
    extract_all_videos()

    print("\n" + "=" * 60)
    print("Phase 2: Label frames by temporal segment")
    print("=" * 60)
    label_and_distribute()

    print("\n" + "=" * 60)
    print("Phase 3: Extract blue-block static frames (extra positives)")
    print("=" * 60)
    extract_blue_block_static()

    # Cleanup tmp
    shutil.rmtree(TMP_DIR)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
