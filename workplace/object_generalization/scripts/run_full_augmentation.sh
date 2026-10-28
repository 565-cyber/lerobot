#!/usr/bin/env bash
# Driver for the full blue->green augmentation run (50 episodes / 25538 frames).
#
# Deliberately a thin bash layer: it schedules, samples resources and enforces
# stage gates. Every stage is resumable on its own (`--skip-existing`), so
# re-running this script after a Ctrl-C continues where it stopped instead of
# redoing finished episodes.
#
# Stages: SAM2 (2 workers, one per GPU) -> hybrid -> generate -> verify -> spot-check.
# Gates: run_report.py between stages, so a half-finished mask cache stops the
# run before it spends 30 minutes generating a dataset.
#
# Usage:
#   scripts/run_full_augmentation.sh                     # full run
#   GPUS="0" scripts/run_full_augmentation.sh            # serial fallback (RAM)
#   EPISODES="3 22" CACHE=.../smoke/cache HYBRID=.../smoke/cache_hybrid \
#     DATA_ROOT=... GEN_OUT=... SKIP_BACKUP=1 scripts/run_full_augmentation.sh   # smoke
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKPLACE="$(dirname "$(dirname "$HERE")")"
REPO="$(dirname "$WORKPLACE")"
PY="${PY:-/home/zjq/miniconda3/envs/lerobot/bin/python}"

DATA_ROOT="${DATA_ROOT:-$REPO/dataset/newdata}"
DATASET="${DATASET:-newdata}"
CACHE="${CACHE:-$WORKPLACE/object_generalization/outputs/cache}"
HYBRID="${HYBRID:-$WORKPLACE/object_generalization/outputs/cache_hybrid}"
GEN_OUT="${GEN_OUT:-$REPO/dataset/newdata_green}"
GEN_REPO_ID="${GEN_REPO_ID:-newdata_green}"
LOGDIR="${LOGDIR:-$WORKPLACE/object_generalization/outputs/logs}"
REPORTS="${REPORTS:-$WORKPLACE/object_generalization/outputs/reports}"
SPOTCHECK="${SPOTCHECK:-$WORKPLACE/object_generalization/outputs/generated_check/spotcheck}"
EPISODES="${EPISODES:-all}"
GPUS="${GPUS:-0 1}"
MIN_OK="${MIN_OK:-45}"
SKIP_BACKUP="${SKIP_BACKUP:-0}"
STRIDE="${STRIDE:-20}"

mkdir -p "$LOGDIR"
# subprocess progress bars interleave into the logs and hide the per-episode
# lines that matter; tqdm honours this, the scripts do not need to know.
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { printf '\n=== %s (%s)\n' "$*" "$(date -Is)"; }

# --- resource sampler -------------------------------------------------------
# RAM is the real risk here, not VRAM: two workers peak near 19.6 GB against
# ~25 GB available. There is 59 GB of swap, so the failure mode is thrashing
# rather than an OOM kill -- which is exactly what a sampler catches and a
# crash log does not.
WORKER_PIDS=""
collector() {
  trap - EXIT INT TERM  # the child must not run the parent's cleanup on its own exit
  local pids="$1"
  while :; do
    local rss="" p
    for p in $pids; do
      if [ -r "/proc/$p/status" ]; then
        rss="$rss$(awk '/VmRSS/{print $2}' "/proc/$p/status" 2>/dev/null) "
      else
        rss="$rss- "
      fi
    done
    printf '{"t":"%s","mem_available_kb":%s,"worker_rss_kb":"%s","gpu_used_mib":"%s"}\n' \
      "$(date -Is)" \
      "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "$rss" \
      "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd';')" \
      >> "$LOGDIR/resources.jsonl"
    sleep 30
  done
}
cleanup() {
  [ -n "$WORKER_PIDS" ] && kill $WORKER_PIDS 2>/dev/null
  kill "$COLLECTOR_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup INT TERM EXIT

# --- stage 0: preflight -----------------------------------------------------
TOTAL="$("$PY" -c "import json;print(json.load(open('$DATA_ROOT/meta/info.json'))['total_episodes'])")"
log "preflight: $TOTAL episodes in $DATA_ROOT, GPUs=[$GPUS], episodes=$EPISODES"
df -h "$(dirname "$GEN_OUT")" | tail -1
"$PY" "$HERE/build_sam2_cache.py" --root "$DATA_ROOT" --dataset "$DATASET" \
  --out "$CACHE" --episodes $EPISODES --dry-run | tail -3
# --dry-run is not optional here: without it this preflight would BUILD the
# hybrid masks it is only meant to count, spending ~10 min and ~16 GB on a union
# that stage 2 immediately rebuilds (the SAM2 signature changes underneath it).
"$PY" "$HERE/build_hybrid_cache.py" --root "$DATA_ROOT" --dataset "$DATASET" \
  --sam2-cache "$CACHE" --out "$HYBRID" --episodes $EPISODES --skip-existing --dry-run | tail -3

# split episodes by parity so the two GPUs hold one worker each; the split does
# not need to be exact, only disjoint -- every worker skips what is already done
EVEN=""; ODD=""
for ((e = 0; e < TOTAL; e++)); do
  if ((e % 2 == 0)); then EVEN="$EVEN $e"; else ODD="$ODD $e"; fi
done

# --- stage 1: SAM2 (2 GPUs in parallel, ~2.4 h) -----------------------------
if [ "$EPISODES" != "all" ]; then EVEN="$EPISODES"; ODD="$EPISODES"; fi
log "stage 1: SAM2 on GPUs [$GPUS]"
i=0
for g in $GPUS; do
  gpu_eps="$EVEN"; [ $((i % 2)) -eq 1 ] && gpu_eps="$ODD"
  logpath="$LOGDIR/sam2_gpu$g.log"
  CUDA_VISIBLE_DEVICES="$g" "$PY" "$HERE/build_sam2_cache.py" \
    --root "$DATA_ROOT" --dataset "$DATASET" --out "$CACHE" --device cuda:0 \
    --front --skip-existing --episodes $gpu_eps > "$logpath" 2>&1 &
  pid=$!
  WORKER_PIDS="$WORKER_PIDS $pid"
  echo "  gpu$g pid=$pid episodes:$(echo $gpu_eps | wc -w) log=$logpath"
  i=$((i + 1))
done
COLLECTOR_PID=""
collector "$WORKER_PIDS" &
COLLECTOR_PID=$!

status=0
for p in $WORKER_PIDS; do
  wait "$p" || { echo "  worker pid=$p exited $?" >&2; status=1; }
done
WORKER_PIDS=""

# gate: a mask cache that covers only part of the dataset must stop the run now
log "gate: SAM2 coverage"
"$PY" "$HERE/run_report.py" --stage sam2 --episodes $EPISODES --root "$DATA_ROOT" \
  --sam2-ledger "$CACHE/reports/sam2_status.jsonl" --min-ok-episodes "$MIN_OK" \
  --report "$REPORTS/full_run_summary.json" \
  || { echo "SAM2 gate failed; not generating. worker status=$status" >&2; exit 1; }

# --- stage 2: hybrid union (~12 min) ----------------------------------------
# FORCE_HYBRID=1 rebuilds every episode even though the done markers match. The
# skip check only compares the SAM2 signature, so a change to the union LOGIC
# (not the SAM2 input) would otherwise be silently ignored.
log "stage 2: hybrid union${FORCE_HYBRID:+ (forced rebuild)}"
SKIP_FLAG="--skip-existing"
[ "${FORCE_HYBRID:-0}" = "1" ] && SKIP_FLAG="--no-skip-existing"
"$PY" "$HERE/build_hybrid_cache.py" --root "$DATA_ROOT" --dataset "$DATASET" \
  --sam2-cache "$CACHE" --out "$HYBRID" --episodes $EPISODES $SKIP_FLAG \
  2>&1 | tail -5 || exit 1

# --- stage 3: generate (~30 min) -------------------------------------------
# The backup happens HERE, not at the start: moving the old 3-episode dataset
# aside during stage 0 would leave `newdata_green` missing for the whole ~2.4 h
# SAM2 run, and anything that loads it meanwhile would fail for no reason.
if [ "$SKIP_BACKUP" != "1" ] && [ -e "$GEN_OUT" ]; then
  BACKUP="${GEN_OUT}_$(date +%Y%m%d_%H%M)"
  log "backing up existing $GEN_OUT -> $BACKUP"
  mv "$GEN_OUT" "$BACKUP"
fi

log "stage 3: generate -> $GEN_OUT"
"$PY" "$HERE/generate_augmented_dataset.py" --root "$DATA_ROOT" --dataset "$DATASET" \
  --mask-cache "$HYBRID" --out "$GEN_OUT" --repo-id "$GEN_REPO_ID" --episodes $EPISODES --report "$REPORTS/generate_status.json" \
  2>&1 | grep -a -v -e '^Svt' -e '^\s*$' | tail -8 || exit 1

# --- stage 4: verify (~5 min) ----------------------------------------------
log "stage 4: verify"
"$PY" "$HERE/verify_generated_dataset.py" --base-root "$DATA_ROOT" --dataset "$DATASET" \
  --gen-root "$GEN_OUT" --repo-id "$GEN_REPO_ID" --cache "$HYBRID" \
  --episodes $EPISODES --stride "$STRIDE" \
  --report "$REPORTS/verify_report.json" \
  2>&1 | grep -a -v -e '^Svt' -e '^\s*$' | tail -25 || exit 1

# --- stage 5: spot-check (~1 min) ------------------------------------------
log "stage 5: spot-check"
"$PY" "$HERE/inspect_generated_dataset.py" --base-root "$DATA_ROOT" --dataset "$DATASET" \
  --gen-root "$GEN_OUT" --repo-id "$GEN_REPO_ID" --cache "$HYBRID" \
  --episodes $EPISODES --out "$SPOTCHECK" \
  2>&1 | tail -4 || exit 1

log "final gate"
"$PY" "$HERE/run_report.py" --stage all --episodes $EPISODES --root "$DATA_ROOT" \
  --sam2-ledger "$CACHE/reports/sam2_status.jsonl" \
  --hybrid-ledger "$HYBRID/reports/hybrid_status.jsonl" \
  --gen-report "$REPORTS/generate_status.json" \
  --verify-report "$REPORTS/verify_report.json" \
  --min-ok-episodes "$MIN_OK" \
  --report "$REPORTS/full_run_summary.json"

log "done: $GEN_OUT"
# S8: the two numbers that decide whether 2 workers may run in parallel at all
# (plan gate: peak worker VmRSS < 9.5 GB, MemAvailable never < 2 GB)
"$PY" - "$LOGDIR/resources.jsonl" <<'PYEOF'
import json
import sys

mins, peak, n = None, 0, 0
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    n += 1
    m = r.get("mem_available_kb")
    if isinstance(m, int) and (mins is None or m < mins):
        mins = m
    for tok in str(r.get("worker_rss_kb", "")).split():
        if tok.isdigit():
            peak = max(peak, int(tok))
if not n:
    print("  no resource samples (run too short to sample)")
else:
    gb = 1 / 1048576
    print(f"  {n} samples: min MemAvailable {mins * gb:.2f} GB, peak worker VmRSS {peak * gb:.2f} GB")
    if mins * gb < 2.0 or peak * gb > 9.5:
        print("  WARN S8 exceeded -- re-run with GPUS=\"0\" (serial) before trusting the timing")
PYEOF
