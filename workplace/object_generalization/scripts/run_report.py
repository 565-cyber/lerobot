"""Roll the per-stage ledgers up into one report and enforce the quality gates.

The driver calls this between stages: a run that produced masks for only 30 of 50
episodes should stop before spending 30 minutes generating a dataset, not after.

Reads the JSONL ledgers written by build_sam2_cache / build_hybrid_cache plus the
generation and verification reports, prints a per-episode table, writes
``outputs/reports/full_run_summary.json``, and exits non-zero if a gate fails.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKPLACE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKPLACE)
for _p in (REPO, WORKPLACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from object_generalization.scripts import _cache_io  # noqa: E402

OUT = os.path.join(WORKPLACE, "object_generalization", "outputs")
SAM2_LEDGER = os.path.join(OUT, "cache", "reports", "sam2_status.jsonl")
HYBRID_LEDGER = os.path.join(OUT, "cache_hybrid", "reports", "hybrid_status.jsonl")
GEN_REPORT = os.path.join(OUT, "reports", "generate_status.json")
VERIFY_REPORT = os.path.join(OUT, "reports", "verify_report.json")

# statuses that mean "this episode produced usable masks"
GOOD = ("ok", "legacy")


def _load_ledger(path):
    """Last record per episode (a resumed episode appears more than once)."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out[int(rec["episode"])] = rec
    return out


def _print_table(title, per_ep, episodes, cols):
    print(f"--- {title}")
    for ep in episodes:
        rec = per_ep.get(ep)
        if rec is None:
            print(f"  ep{ep:<3d} MISSING")
            continue
        bits = " ".join(f"{k}={rec.get(k)}" for k in cols)
        print(f"  ep{ep:<3d} status={rec.get('status'):<20s} {bits}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sam2", "hybrid", "generate", "verify", "all"], default="all")
    ap.add_argument("--episodes", nargs="+", default=["all"])
    ap.add_argument("--min-ok-episodes", type=int, default=45,
                    help="gate: episodes that must have usable masks before generating")
    ap.add_argument("--root", default="/home/zjq/lerobot/dataset/newdata")
    ap.add_argument("--sam2-cache", default=os.path.join(OUT, "cache"),
                    help="used for episodes that predate the ledger (no JSONL line)")
    ap.add_argument("--sam2-ledger", default=SAM2_LEDGER)
    ap.add_argument("--hybrid-ledger", default=HYBRID_LEDGER)
    ap.add_argument("--gen-report", default=GEN_REPORT)
    ap.add_argument("--verify-report", default=VERIFY_REPORT)
    ap.add_argument("--report", default=os.path.join(OUT, "reports", "full_run_summary.json"))
    args = ap.parse_args()

    gen_report, verify_report = args.gen_report, args.verify_report

    total = _cache_io.total_episodes(args.root)
    episodes = _cache_io.parse_episodes(args.episodes[0] if len(args.episodes) == 1 else args.episodes, total)

    summary = {"episodes_requested": len(episodes)}
    failures = []

    lengths = _cache_io.episode_lengths(args.root)
    sam2 = _load_ledger(args.sam2_ledger)
    hybrid = _load_ledger(args.hybrid_ledger)

    # Episodes cached before the ledger existed (the 3-episode runs: 0-2) have
    # valid masks but no JSONL line. Counting ledger rows only would report
    # 47/50 and make the gate look tighter than the data is; ask the disk.
    for ep in episodes:
        if ep in sam2:
            continue
        status, _ = _cache_io.entry_status(
            os.path.join(args.sam2_cache, f"episode_{ep}"), lengths.get(ep, -1)
        )
        if _cache_io.is_done(status):
            sam2[ep] = {"episode": ep, "n_frames": lengths.get(ep), "status": status,
                        "up_safe_cov": None, "front_safe_cov": None, "seconds": None,
                        "note": "no ledger line (cached before the ledger existed)"}

    if args.stage in ("sam2", "all"):
        _print_table("SAM2", sam2, episodes, ["n_frames", "up_safe_cov", "front_safe_cov", "seconds"])
        good = [ep for ep in episodes if sam2.get(ep, {}).get("status") in GOOD]
        summary["sam2_ok"] = len(good)
        summary["sam2_degraded"] = {
            ep: {"status": sam2[ep].get("status"), "error": sam2[ep].get("error")}
            for ep in episodes if ep in sam2 and sam2[ep].get("status") not in GOOD
        }
        for cam in ("up", "front"):
            # disk-only (pre-ledger) episodes have no recorded coverage: skip them
            # rather than letting a None poison the median
            covs = [sam2[ep][f"{cam}_safe_cov"] for ep in good
                    if isinstance(sam2[ep].get(f"{cam}_safe_cov"), (int, float))]
            if covs:
                summary[f"sam2_{cam}_cov_median"] = round(statistics.median(covs), 3)
                print(f"  {cam} coverage: median={statistics.median(covs):.3f} "
                      f"min={min(covs):.3f} over {len(covs)} episodes")
        if len(good) < args.min_ok_episodes:
            failures.append(f"SAM2: only {len(good)}/{len(episodes)} usable (need >= {args.min_ok_episodes})")

    if args.stage in ("hybrid", "all"):
        _print_table("hybrid", hybrid, episodes, ["up_cov", "front_cov", "up_cov_sam2", "up_cov_lw"])
        bad = [ep for ep in episodes if hybrid.get(ep, {}).get("status") not in GOOD]
        summary["hybrid_ok"] = len(episodes) - len(bad)
        if bad:
            failures.append(f"hybrid: {len(bad)} episode(s) without usable masks: {bad[:10]}")

    if args.stage in ("generate", "all") and os.path.isfile(gen_report):
        with open(gen_report) as f:
            gen = json.load(f)
        summary["generated_frames"] = gen["total_frames"]
        summary["recolored_camera_frames"] = gen["recolored_camera_frames"]
        summary["generated_out"] = gen["out"]
        print(f"--- generate: {gen['total_frames']} frames, "
              f"{gen['recolored_camera_frames']} camera-frames recolored -> {gen['out']}")

    if args.stage in ("verify", "all") and os.path.isfile(verify_report):
        with open(verify_report) as f:
            ver = json.load(f)
        summary["verify_pass"] = ver["pass"]
        summary["verify_failures"] = ver["failures"]
        print(f"--- verify: {'PASS' if ver['pass'] else 'FAIL'}")
        for f_ in ver["failures"][:10]:
            print(f"  FAIL {f_}")
        if not ver["pass"]:
            failures.append(f"verify reported {len(ver['failures'])} failure(s)")

    summary["failures"] = failures
    summary["pass"] = not failures
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    print(f"summary -> {args.report}")
    if failures:
        print("GATE FAILED: " + "; ".join(failures), file=sys.stderr)
        sys.exit(1)
    print("GATE OK")


if __name__ == "__main__":
    main()
