#!/usr/bin/env bash
# Driver for the full Stage 2B distractor-augmentation run (50 episodes x K variants).
#
# Deliberately a thin bash layer: it schedules, samples resources and enforces
# stage gates. Unlike Stage 2A there is no GPU work -- plates and compositing are
# OpenCV on the CPU -- so there is no worker fan-out, and RAM is not the risk it
# was for SAM2. The sampler stays because S8 asks for the numbers, and because
# "it ran fine" is not the same claim as "it stayed under 9.5 GB".
#
# Stages: plate (~15 min) -> generate (~2 h for K=3) -> verify -> inspect.
# Every stage is resumable on its own (`--skip-existing` for the plate), so
# re-running after a Ctrl-C continues instead of redoing finished episodes.
# `generate` is NOT resumable: LeRobotDataset.create refuses an existing root.
#
# Usage:
#   scripts/run_background_augmentation.sh                      # full run, K=3
#   VARIANTS=1 scripts/run_background_augmentation.sh           # smaller set
#   EPISODES="0 1" GEN_OUT=/tmp/smoke/ds PLATE=/tmp/smoke/plate \
#     REPORTS=/tmp/smoke/reports scripts/run_background_augmentation.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BG="$(dirname "$HERE")"
WORKPLACE="$(dirname "$BG")"
REPO="$(dirname "$WORKPLACE")"
PY="${PY:-/home/zjq/miniconda3/envs/lerobot/bin/python}"

DATA_ROOT="${DATA_ROOT:-$REPO/dataset/newdata}"
DATASET="${DATASET:-newdata}"
PLATE="${PLATE:-$BG/outputs/plate}"
HYBRID="${HYBRID:-$WORKPLACE/object_generalization/outputs/cache_hybrid}"
GEN_OUT="${GEN_OUT:-$REPO/dataset/newdata_distract}"
GEN_REPO_ID="${GEN_REPO_ID:-newdata_distract}"
REPORTS="${REPORTS:-$BG/outputs/distract}"
CHECK="${CHECK:-$BG/outputs/distract_check}"
LOGDIR="${LOGDIR:-$BG/outputs/logs}"
EPISODES="${EPISODES:-all}"
VARIANTS="${VARIANTS:-3}"
SEED="${SEED:-0}"
STRIDE="${STRIDE:-20}"
SKIP_BACKUP="${SKIP_BACKUP:-0}"
MIN_PLACEMENT_FRAC="${MIN_PLACEMENT_FRAC:-0.12}"
MIN_LARGEST_BLOB="${MIN_LARGEST_BLOB:-27000}"

mkdir -p "$LOGDIR" "$REPORTS"
# subprocess progress bars interleave into the logs and hide the per-episode
# lines that matter; tqdm honours this, the scripts do not need to know.
export TQDM_DISABLE=1

log() { printf '\n=== %s (%s)\n' "$*" "$(date -Is)"; }
strip_svt() { grep -a -v -e '^Svt' -e '^\[mp4' -e '^\s*$'; }

# --- resource sampler -------------------------------------------------------
COLLECTOR_PID=""
collector() {
  trap - EXIT INT TERM
  local pid="$1"
  while :; do
    printf '{"t":"%s","mem_available_kb":%s,"worker_rss_kb":"%s"}\n' \
      "$(date -Is)" \
      "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null || echo -)" \
      >> "$LOGDIR/resources_bg.jsonl"
    sleep 30
  done
}
cleanup() {
  [ -n "$COLLECTOR_PID" ] && kill "$COLLECTOR_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup INT TERM EXIT

# --- stage 0: preflight -----------------------------------------------------
TOTAL="$("$PY" -c "import json;print(json.load(open('$DATA_ROOT/meta/info.json'))['total_episodes'])")"
log "preflight: $TOTAL episodes in $DATA_ROOT, episodes=$EPISODES, K=$VARIANTS seed=$SEED"
df -h "$(dirname "$GEN_OUT")" | tail -1
"$PY" "$BG/masks/build_plate.py" --root "$DATA_ROOT" --dataset "$DATASET" --out "$PLATE" \
  --episodes $EPISODES --dry-run | tail -3

# --- stage 1: plate + placement map (~15 min, resumable) --------------------
log "stage 1: plate + placement -> $PLATE"
"$PY" "$BG/masks/build_plate.py" --root "$DATA_ROOT" --dataset "$DATASET" --out "$PLATE" \
  --hybrid-cache "$HYBRID" --episodes $EPISODES --skip-existing \
  > "$LOGDIR/plate.log" 2>&1 &
PLATE_PID=$!
collector "$PLATE_PID" &
COLLECTOR_PID=$!
wait "$PLATE_PID"; PLATE_STATUS=$?
kill "$COLLECTOR_PID" 2>/dev/null; COLLECTOR_PID=""
tail -3 "$LOGDIR/plate.log"
[ "$PLATE_STATUS" -eq 0 ] || { echo "build_plate exited $PLATE_STATUS" >&2; exit 1; }

# gate: an episode whose placement map is too small cannot host anything, and a
# run that quietly degenerates to "duplicate the base dataset" is the failure
# this catches. Thresholds sit below the worst episode of all 50 (18.2% /
# 45156 px, being episodes 28 and 0) rather than at the median, so the gate fires
# on a broken run and not on a merely tight episode -- the per-episode density
# self-adjusts (see LayoutConfig.n_per_episode) and does not need protecting.
# See PlateConfig.
#
# These are sized for `place_freq_max = 0.10`. The 0.06 / 12000 pair they
# replaced were sized for 0.05, where the worst episode was 10.0% / 21803 px;
# leaving them there would have left the gate asserting nothing. Re-measure
# against `outputs/plate/episode_*/done.json` if that threshold moves again.
log "gate: placement maps"
"$PY" - "$PLATE" "$MIN_PLACEMENT_FRAC" "$MIN_LARGEST_BLOB" "$EPISODES" <<'PYEOF' || exit 1
import json, os, sys
plate, min_frac, min_blob, eps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
eps = sorted(int(d.split('_')[1]) for d in os.listdir(plate) if d.startswith('episode_'))
if eps == []:
    sys.exit("no episodes in the plate cache")
weak, err, ok = [], [], 0
for ep in eps:
    rec = json.load(open(os.path.join(plate, f'episode_{ep}', 'done.json')))
    if rec.get('status') != 'ok':
        err.append((ep, rec.get('error')))
        continue
    ok += 1
    if rec['placement_frac'] < min_frac or rec['placement_largest_blob'] < min_blob:
        weak.append((ep, rec['placement_frac'], rec['placement_largest_blob']))
print(f"  {ok}/{len(eps)} episode(s) ok; {len(weak)} with a placement map below "
      f"{min_frac:.0%} / {min_blob}px")
if err:
    print(f"  FAIL {len(err)} episode(s) errored: {err[:5]}")
if weak:
    print(f"  WARN weak placement: {weak[:10]}")
    print(f"  {len(weak)}/{len(eps)} below the gate -- those episodes will get few or no "
          f"distractors, so the augmented set is partly duplicates. Inspect "
          f"`outputs/plate/episode_*/placement.npy` before trusting the run.")
if err or len(weak) > 0.2 * len(eps):
    sys.exit(1)
PYEOF

# gate: what the layouts would actually place, per episode, before spending two
# hours rendering them. The placement-map gate above cannot see a map that is
# large enough yet still hosts nothing; this samples all 50*K layouts and fails
# on any that would place zero objects, because such a variant is a bit-exact
# duplicate of its base episode that still passes verification. It renders
# nothing, so it is cheap -- and it is why --dry-run needs no free --out.
log "gate: layout census (nothing rendered)"
"$PY" "$HERE/generate_distractor_dataset.py" --root "$DATA_ROOT" --dataset "$DATASET" \
  --plate-root "$PLATE" --hybrid-cache "$HYBRID" --out "$GEN_OUT" --repo-id "$GEN_REPO_ID" \
  --episodes $EPISODES --variants "$VARIANTS" --seed "$SEED" --report-dir "$REPORTS" \
  --dry-run 2>&1 | tee "$LOGDIR/census.log" | tail -8
CENSUS_STATUS="${PIPESTATUS[0]}"
[ "$CENSUS_STATUS" -eq 0 ] || { echo "layout census placed nothing for $CENSUS_STATUS layout(s); not generating" >&2; exit 1; }

# --- stage 2: generate (~2 h at K=3, NOT resumable) -------------------------
if [ "$SKIP_BACKUP" != "1" ] && [ -e "$GEN_OUT" ]; then
  BACKUP="${GEN_OUT}_$(date +%Y%m%d_%H%M)"
  log "backing up existing $GEN_OUT -> $BACKUP"
  mv "$GEN_OUT" "$BACKUP"
fi
log "stage 2: generate $EPISODES x $VARIANTS -> $GEN_OUT"
"$PY" "$HERE/generate_distractor_dataset.py" --root "$DATA_ROOT" --dataset "$DATASET" \
  --plate-root "$PLATE" --hybrid-cache "$HYBRID" --out "$GEN_OUT" --repo-id "$GEN_REPO_ID" \
  --episodes $EPISODES --variants "$VARIANTS" --seed "$SEED" --report-dir "$REPORTS" \
  > "$LOGDIR/generate.log" 2>&1 &
GEN_PID=$!
collector "$GEN_PID" &
COLLECTOR_PID=$!
wait "$GEN_PID"; GEN_STATUS=$?
kill "$COLLECTOR_PID" 2>/dev/null; COLLECTOR_PID=""
strip_svt < "$LOGDIR/generate.log" | tail -8
[ "$GEN_STATUS" -eq 0 ] || { echo "generation exited $GEN_STATUS" >&2; exit 1; }

# --- stage 3: verify --------------------------------------------------------
# Written to a log and tailed rather than piped straight through: with
# `pipefail`, `... | tail` reports *tail*'s status, so a FAILing verifier would
# sail past the `|| exit 1` it looks like it is guarding.
log "stage 3: verify"
"$PY" "$HERE/verify_distractor_dataset.py" --dataset "$DATASET" --report-dir "$REPORTS" \
  --hybrid-cache "$HYBRID" --episodes $EPISODES --stride "$STRIDE" \
  > "$LOGDIR/verify.log" 2>&1
VERIFY_STATUS=$?
strip_svt < "$LOGDIR/verify.log" | tail -40
[ "$VERIFY_STATUS" -eq 0 ] || { echo "verify FAILED (exit $VERIFY_STATUS); not inspecting" >&2; exit 1; }

# --- stage 4: inspect -------------------------------------------------------
log "stage 4: inspect"
"$PY" "$HERE/inspect_distractor_dataset.py" --dataset "$DATASET" --report-dir "$REPORTS" \
  --episodes $EPISODES --out "$CHECK" > "$LOGDIR/inspect.log" 2>&1
INSPECT_STATUS=$?
tail -6 "$LOGDIR/inspect.log"
[ "$INSPECT_STATUS" -eq 0 ] || { echo "inspect exited $INSPECT_STATUS" >&2; exit 1; }

log "done: $GEN_OUT"
# S8: the two numbers the plan asks for before trusting the timing
"$PY" - "$LOGDIR/resources_bg.jsonl" <<'PYEOF'
import json, sys
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
    tok = str(r.get("worker_rss_kb", "")).strip()
    if tok.isdigit():
        peak = max(peak, int(tok))
if not n:
    print("  no resource samples (run too short to sample)")
else:
    gb = 1 / 1048576
    print(f"  {n} samples: min MemAvailable {mins * gb:.2f} GB, peak worker VmRSS {peak * gb:.2f} GB")
    if mins * gb < 2.0 or peak * gb > 9.5:
        print("  WARN S8 exceeded")
PYEOF
