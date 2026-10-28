"""Shared cache housekeeping for the mask pipeline (no algorithmic code).

Three things every stage of the pipeline needs once it runs over all 50 episodes
instead of episodes 0-2:

  * a single way to turn ``--episodes all`` into an explicit index list,
  * crash-safe mask writes (a killed process must never leave a half-written
    ``.npy`` that a later run mistakes for a finished episode),
  * a per-episode ``done.json`` marker that is written **last**, so a re-run can
    skip finished work -- and that also recognises the pre-existing episodes 0-2
    caches, which have valid masks but no marker (status ``legacy``).

The marker is checked against the actual array shapes, never trusted on its own:
a stale marker over a truncated ``.npy`` would silently produce a wrong dataset.
"""

from __future__ import annotations

import json
import os

import numpy as np

DONE_NAME = "done.json"


def total_episodes(root: str) -> int:
    """Episode count from a dataset root's ``meta/info.json``."""
    with open(os.path.join(root, "meta", "info.json"), "r") as f:
        return int(json.load(f)["total_episodes"])


def episode_lengths(root: str) -> dict:
    """``{episode_index: length}`` from the dataset metadata (no video decode)."""
    import glob

    import pandas as pd

    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "*.parquet")))
    if not files:
        raise SystemExit(f"no episode metadata under {root}/meta/episodes")
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "length"]) for f in files])
    return {int(r.episode_index): int(r.length) for r in df.itertuples()}


def parse_episodes(value, total: int) -> list[int]:
    """``--episodes`` value -> sorted explicit list.

    Accepts ``None`` or ``"all"`` (every episode), a list of ints, or the string
    ``"3-49"``. Out-of-range indices raise rather than being silently dropped.
    """
    if value is None or value == "all":
        eps = list(range(total))
    elif isinstance(value, str):
        lo, _, hi = value.partition("-")
        eps = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    else:
        eps = [int(v) for v in value]
    bad = [e for e in eps if not 0 <= e < total]
    if bad:
        raise SystemExit(f"episodes out of range 0..{total - 1}: {bad}")
    return sorted(set(eps))


def gen_positions(episodes) -> dict:
    """``{base episode_index: episode_index inside the generated dataset}``.

    ``LeRobotDataset.create`` + ``add_frame``/``save_episode`` numbers episodes
    from 0 in the order the frames are added, which is the ascending order of the
    requested episodes. A subset run (``--episodes 3 22``) therefore writes
    episodes 0 and 1, *not* 3 and 22 -- so a checker that addresses the generated
    file by base episode index silently reads the wrong episode (or nothing).
    For a full run (0..49 in order) this is the identity map.
    """
    return {ep: i for i, ep in enumerate(sorted(episodes))}


def atomic_save_npy(path: str, arr: np.ndarray) -> None:
    """``np.save`` to a sibling temp name, then ``os.replace`` (atomic on POSIX)."""
    tmp = path + ".part.npy"  # np.save only appends '.npy' when absent
    np.save(tmp, arr)
    os.replace(tmp, path)


def write_done(ep_dir: str, rec: dict) -> None:
    """Commit record for one episode. MUST be the last write of that episode."""
    tmp = os.path.join(ep_dir, DONE_NAME + ".part")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(ep_dir, DONE_NAME))


def read_done(ep_dir: str) -> dict | None:
    p = os.path.join(ep_dir, DONE_NAME)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def entry_status(ep_dir, n_frames, cams=("up", "front"), shape_hw=(480, 640)):
    """Classify one cached episode: complete | legacy | missing | error.

    ``legacy`` = every mask present with the right shape but no ``done.json``
    (this is exactly the state of ``outputs/cache/episode_{0,1,2}`` from the
    3-episode runs) -- treated as done so those masks are reused, never recomputed.
    """
    for cam in cams:
        p = os.path.join(ep_dir, f"{cam}_mask.npy")
        if not os.path.isfile(p):
            return "missing", None
        try:
            shape = tuple(np.load(p, mmap_mode="r").shape)
        except (ValueError, OSError):
            return "missing", None
        if shape != (n_frames, *shape_hw):
            return "missing", None
    rec = read_done(ep_dir)
    if rec is None:
        return "legacy", None
    if rec.get("status") == "error":
        return "error", rec
    return "complete", rec


def is_done(status: str) -> bool:
    return status in ("complete", "legacy")


def append_jsonl(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
