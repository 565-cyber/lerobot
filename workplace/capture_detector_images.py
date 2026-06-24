"""
Manual capture script for binary detector datasets — NO GUI required.

Saves the current camera frame on key press in the TERMINAL.
Works over SSH, no GTK/QT/matplotlib needed.

Usage:
  python workplace/capture_detector_images.py --detector detector_yellow_on_table
  python workplace/capture_detector_images.py --detector detector_blue_on_table
  python workplace/capture_detector_images.py --detector detector_lid_open

Terminal keys:
  p / Enter  — capture as POSITIVE (condition is TRUE)
  n          — capture as NEGATIVE (condition is FALSE)
  s          — toggle train / val split
  d          — delete last saved image (undo)
  q          — quit

Captured images:
  workplace/binary_detector_dataset/<detector>/<split>/positive/000001.jpg
  workplace/binary_detector_dataset/<detector>/<split>/negative/000001.jpg
"""
import argparse
import sys
import select
import termios
import tty
from pathlib import Path

import cv2
import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/zjq/lerobot/workplace/binary_detector_dataset")

DETECTOR_INFO = {
    "detector_yellow_on_table": "Yellow block on table?",
    "detector_blue_on_table": "Blue block on table?",
    "detector_lid_open": "Trash can lid open?",
}


def get_next_index(folder: Path) -> int:
    if not folder.exists():
        return 1
    existing = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    if not existing:
        return 1
    indices = [int(f.stem) for f in existing if f.stem.isdigit()]
    return max(indices) + 1 if indices else 1


def count_existing(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))


def get_detector_dir(detector: str) -> Path:
    if not detector.startswith("detector_"):
        detector = f"detector_{detector}"
    return OUTPUT_DIR / detector


def getchar():
    """Read a single character from stdin without waiting for Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if r:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Simple JPEG viewer using child process ──
def show_frame_jpeg(frame: np.ndarray, jpeg_path: str = "/tmp/_capture_preview.jpg"):
    """Save frame as JPEG and return path for optional external viewer."""
    cv2.imwrite(jpeg_path, frame)
    return jpeg_path


def main():
    parser = argparse.ArgumentParser(
        description="Manually capture and label images for binary detectors (terminal mode)"
    )
    parser.add_argument("--detector", type=str, default="detector_yellow_on_table")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--default-class", type=str, default="positive",
                        choices=["positive", "negative"],
                        help="Which class [Enter] key saves to (default: positive)")
    parser.add_argument("--preview", action="store_true",
                        help="Save a JPEG preview file to /tmp after each capture for external viewing")
    args = parser.parse_args()

    detector_dir = get_detector_dir(args.detector)
    detector_name = detector_dir.name
    question = DETECTOR_INFO.get(detector_name, "Condition TRUE?")

    # ── Open camera ──
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera} opened: {actual_w}x{actual_h}")

    # ── Prepare folders ──
    split = args.split

    def ensure_dirs(s: str):
        pos = detector_dir / s / "positive"
        neg = detector_dir / s / "negative"
        pos.mkdir(parents=True, exist_ok=True)
        neg.mkdir(parents=True, exist_ok=True)
        return pos, neg

    pos_dir, neg_dir = ensure_dirs(split)
    idx_pos = get_next_index(pos_dir)
    idx_neg = get_next_index(neg_dir)

    # Existing totals across both splits
    total_pos = count_existing(detector_dir / "train" / "positive")
    total_pos += count_existing(detector_dir / "val" / "positive")
    total_neg = count_existing(detector_dir / "train" / "negative")
    total_neg += count_existing(detector_dir / "val" / "negative")

    print(f"Detector : {detector_name}  ({question})")
    print(f"Split    : {split}")
    print(f"Existing : {total_pos} positive, {total_neg} negative")
    print()
    default_is_pos = (args.default_class == "positive")
    print(f"Default class for [Enter]: {args.default_class.upper()}")
    print()
    print("Keys (press in this terminal):")
    print(f"  [p]             — POSITIVE  (condition is TRUE)")
    print(f"  [n]             — NEGATIVE  (condition is FALSE)")
    print(f"  [Enter]         — {args.default_class.upper()} (current default, --default-class to change)")
    print("  [s]             — toggle train / val")
    print("  [d]             — delete last saved image (undo)")
    print("  [q]             — quit")
    print()

    last_saved = None  # path of last saved file, for undo
    frame_count = 0

    print("Waiting for key... (make sure the camera is pointing at the scene)")

    while True:
        # Grab latest frame
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1

        # Show heartbeat every ~30 frames (~1 sec at 30 fps)
        if frame_count % 30 == 0:
            sys.stdout.write(f"\r[{split}] POS={idx_pos - 1} NEG={idx_neg - 1}  waiting...  ")
            sys.stdout.flush()

        key = getchar()
        if key is None:
            continue

        key = key.lower()

        if key == 'q':
            break

        elif key == 'p':  # p = always positive
            ret_save, save_frame = cap.read()
            if not ret_save:
                save_frame = frame
            path = pos_dir / f"{idx_pos:06d}.jpg"
            cv2.imwrite(str(path), save_frame)
            print(f"\n  [+] POSITIVE  -> {path.name}  (total pos: {total_pos + idx_pos})")
            last_saved = path
            idx_pos += 1
            if args.preview:
                show_frame_jpeg(save_frame)

        elif key == 'n':  # n = always negative
            ret_save, save_frame = cap.read()
            if not ret_save:
                save_frame = frame
            path = neg_dir / f"{idx_neg:06d}.jpg"
            cv2.imwrite(str(path), save_frame)
            print(f"\n  [-] NEGATIVE  -> {path.name}  (total neg: {total_neg + idx_neg})")
            last_saved = path
            idx_neg += 1
            if args.preview:
                show_frame_jpeg(save_frame)

        elif key == '\r' or key == '\n':  # Enter = default class
            ret_save, save_frame = cap.read()
            if not ret_save:
                save_frame = frame
            if default_is_pos:
                path = pos_dir / f"{idx_pos:06d}.jpg"
                label = "POSITIVE"
                idx_pos += 1
            else:
                path = neg_dir / f"{idx_neg:06d}.jpg"
                label = "NEGATIVE"
                idx_neg += 1
            cv2.imwrite(str(path), save_frame)
            print(f"\n  [{'+' if default_is_pos else '-'}] {label}  -> {path.name}  (total: {total_pos + idx_pos - 1} pos, {total_neg + idx_neg - 1} neg)")
            last_saved = path
            if args.preview:
                show_frame_jpeg(save_frame)

        elif key == 's':
            split = "val" if split == "train" else "train"
            pos_dir, neg_dir = ensure_dirs(split)
            idx_pos = get_next_index(pos_dir)
            idx_neg = get_next_index(neg_dir)
            print(f"\n  [S] Switched to split: {split}")

        elif key == 'd':
            if last_saved is not None and last_saved.exists():
                if "positive" in str(last_saved):
                    idx_pos -= 1
                else:
                    idx_neg -= 1
                print(f"\n  [D] Deleted: {last_saved.name}")
                last_saved.unlink()
                last_saved = None
            else:
                print(f"\n  [D] Nothing to undo")

    cap.release()

    # Final summary
    final_pos = total_pos + idx_pos - 1
    final_neg = total_neg + idx_neg - 1
    print()
    print(f"Done. {detector_name}: {final_pos} positive, {final_neg} negative")
    if final_pos == 0 or final_neg == 0:
        print("WARNING: one class has 0 samples — training will fail!")
    elif abs(final_pos - final_neg) > max(final_pos, final_neg) * 0.5:
        print("NOTE: class imbalance > 2:1 — consider collecting more of the minority class")


if __name__ == "__main__":
    main()
