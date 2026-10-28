# Object Color Generalization (Stage 2A)

Research question: does changing only the *interaction object's* color (blue cube
-> green cube) hurt vanilla ACT success rate, and does training on blue->green
augmented demonstrations reduce that drop?

This is a **label-preserving visual augmentation** `(T_color(I), s, a)`:
positions, shape, size, shadow structure, robot state and action labels are all
left unchanged; only the robot's grasp-target cube becomes green.

## Approach: interaction-object tracking (not "recolor all blue")

Naive HSV ("every blue pixel -> green") cannot tell *which* blue object the robot
is manipulating. Instead we first *initialize* on the interaction object (frame 0
of an episode) and then *track its identity* frame-by-frame:

```
masks/blue_object_mask.py            # HSV blue -> threshold mask / components / single mask
masks/interaction_object_tracker.py  # UP-camera episode tracker: initialize + update
masks/front_interaction_tracker.py   # FRONT-camera tracker (geometry + robot-state prior)
masks/robot_state_prior.py           # RobotStatePrior: 6-d state -> phase / grasp / release
masks/multiview_object_tracker.py    # MultiViewInteractionTracker (up=identity, front=local)
transforms/object_recolor.py         # hue-rotation recolor (blue->green, keep S/V)
datasets/object_augmented_dataset.py # Dataset wrapper (camera_mode + offline mask cache)
scripts/analyze_robot_state.py       # state analysis: joint names / ranges / gripper events
scripts/inspect_object_recolor.py    # up validation: stats + invariance + figures
scripts/inspect_front_tracking.py    # front validation: stats + switch/lost + figures
scripts/inspect_robot_state_tracking.py # state-aware front tracker + baseline comparison
scripts/inspect_multiview_tracking.py # multiview vs state-aware front + cross-view viz
scripts/build_multiview_cache.py     # offline per-episode up/front mask cache (lightweight)
scripts/verify_augmented_dataset.py  # end-to-end wrapper test (state/action + invariance)
scripts/sam2_smoke.py                # SAM2 smoke test on ep0 up stream
scripts/build_sam2_cache.py          # SAM2 video-propagation cache (blue-intersected)
scripts/build_hybrid_cache.py        # SAM2 ∪ lightweight union cache (recommended)
scripts/compare_sam2_lightweight.py  # per-phase coverage + IoU comparison
scripts/verify_hybrid_cache.py       # end-to-end hybrid-cache test (all frames)
scripts/generate_augmented_dataset.py # build newdata_green (recolor -> new LeRobot dataset)
scripts/verify_generated_dataset.py  # base-vs-generated state/action + hue check
scripts/inspect_generated_dataset.py # base-vs-generated side-by-side panels
outputs/                             # figures / debug panels / cache/ + cache_hybrid/ + generated_check/
```

- `InteractionObjectTracker.initialize`: HSV candidates -> gate by area
  `[min_area, max_area]`, aspect `[0.4, 2.5]`, extent `>= 0.35` (small, square,
  compact) -> score by `|log(area/prior)| + 2|aspect-1|` -> lock target.
- `InteractionObjectTracker.update`: search near the previous bbox (region grows
  while occluded), pick the candidate nearest to the previous center with similar
  area/color. No candidate for `occlusion_timeout` frames -> terminally `lost`
  (never re-acquire a *different* object).
- Robot state: the FRONT tracker consumes `RobotStatePrior` (phase from the 6-d
  state, no FK). The UP tracker still ignores state (a single blue object is
  present there, so no disambiguation is needed).
- `MultiViewInteractionTracker` (master/slave): the UP camera owns the object
  *identity* (one cube, seen from above); the FRONT camera *associates* its own
  local blue candidates to that identity via a `TargetTemplate` (shape + area +
  HSV crop embedding, EMA-updated only while the up view is healthy). Front
  matching priority is temporal (its own previous target, scale-invariant and
  staleness-scaled) > shape > appearance > color. The template freezes when the
  up view degrades (gripper occluding the cube from above), so the front keeps
  tracking through grasp/carry. A background blue object (large, irregular) is
  excluded by the front geometry gate (area 800-12000, extent>=0.75), and the
  up identity prevents switching to a *different* blue object.

## Constraints

- No file under `src/lerobot/` is modified.
- SAM2 is used **only** as a video instance tracker (user-approved), and its mask is
  always intersected with the blue threshold so it can never recolor a non-blue
  region. No GroundingDINO / DINOv2 / SD / VLM.
- ACT / backbone / transformer / loss / optimizer untouched.
- `state` / `action` byte-identical (asserted).
- 宁可漏增强，不要错增强.

## Validation results (newdata, episodes 0-2, 1684 frames)

- Up camera has a **single blue object** (the cube) through the whole episode
  (max area ~1300 px; the 2-3 "components" seen during grasp are the same cube
  split by the gripper). No background blue box / blue robot part exists in the
  up view.
- init success rate 3/3; tracking 73.0%, occluded 27.0% (grasp + drop), lost 0.
- Recolor: hue 119->62 (blue->green), V preserved exactly, S preserved (~1 unit),
  mask-outside max diff = 0.0.
- Occluded frames (e.g. f200) are returned unchanged (frame_unchanged=True).
- `state` / `action` byte-identical through the wrapper and a DataLoader batch.
- **Front camera** uses a dedicated `FrontInteractionObjectTracker` (v2,
  robot-state-aware): geometry gates (compact `extent>=0.75`, `area<=12000`,
  near-square) separate the target cube (~7.5k px, extent 0.8-0.9) from the large
  irregular blue background (20-46k px, extent 0.5-0.7), and a `RobotStatePrior`
  (no FK) gates tracking by phase and adjusts the search radius / target-consistency
  weight. Phases: home/approaching/grasping/moving/placing/retracting, derived
  from `elbow_flex + shoulder_lift + gripper(+delta)`. State layout (parquet):
  `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`.
  Overall tracking 50.4% (vs 43.7% single-joint baseline); **approach tracking
  83.7% (vs 60.0%)**, grasp 100%, moving 73.1%, placing 4.3% (conservative: the
  cube is released); home/retract 0% (correct). Locked earlier (f~86-113 vs
  f~133-146). **0 wrong background selections** (switches identical {0:1,2:1},
  still the cube). Front cube recolor: H 113->53, V preserved exactly.
- **Multi-view tracker** matches the state-aware front on every interacting phase
  (approach 83.7%, grasp 100%, moving 73.1%, placing 4.3%) and additionally
  tracks the cube at **home (26%)** — the up identity disambiguates the cube on
  the table (front bbox area 2587, extent 0.80) from the background blue (39558,
  extent 0.66), which the single-view front deliberately suppressed. Target
  switches and lost frames are **identical** to the single-view (switches
  {0:1,2:1}, lost 12). Frame sync: up/front co-located in the same row (0/1684
  misaligned). `camera_mode` (`up_only`/`front_only`/`multiview`) selects which
  cached mask is applied; offline cache at `outputs/cache/episode_*/{up,front}_mask.npy`.
  End-to-end wrapper: state/action bit-identical, mask-outside diff = 0.0.

## SAM2 (video instance tracking) + hybrid result

SAM2 (`sam2.1_hiera_base_plus`) was evaluated as the mask source. A critical
safety flaw appears if its mask is used **raw**: during grasp/placing SAM2 drifts
off the cube onto the gripper/red container and keeps emitting a mask (up-camera
blue fraction drops to 0.05-0.10 at grasp, 0.00 at placing). Since the recolor
hue-rotates unconditionally under the mask, a raw SAM2 mask would turn the red
container/table blue — a wrong augmentation. The fix is to intersect SAM2's
instance mask with the blue threshold mask (`SAM2_mask ∩ blue`), which makes the
mask blue-restricted by construction: drift onto a non-blue region yields an empty
mask (no recolor). This preserves 宁可漏增强，不要错增强.

Per-phase coverage (SAM2-safe vs lightweight multiview), IoU where both track:

| phase       | cam    | SAM2   | LW     | IoU   |
|-------------|--------|--------|--------|-------|
| home        | front  | 65.0%  | 26.0%  | 1.00  |
| approaching | front  | 100.0% | 83.7%  | 1.00  |
| grasping    | front  | 100.0% | 100.0% | 1.00  |
| moving      | front  | 88.3%  | 73.1%  | 1.00  |
| placing     | front  | 12.8%  | 4.3%   | 1.00  |
| home        | up     | 84.8%  | 100.0% | 1.00  |
| retracting  | up     | 69.1%  | 97.9%  | 1.00  |

- **Front camera: SAM2 strictly dominates** (approach +16.3, moving +15.2,
  placing +8.5, home +39 pts) — the learned propagation resolves the background
  blue (20-46k px) far better than the geometry gate.
- **Up camera: lightweight strictly dominates** (home +15.2, retract +28.8) — a
  single blue object lives there (no ambiguity), and the lightweight tracker
  re-acquires after the grasp occlusion while SAM2's propagation does not.
- **IoU = 1.00 everywhere both track**: the two methods agree exactly on the
  cube's location, so the union cannot introduce a wrong region.

Recommended final cache: `outputs/cache_hybrid/` = `SAM2_mask ∪ lightweight_mask`.
End-to-end on the hybrid cache (episodes 0-2): 1684 frames, 0 state/action
mismatches, max outside-mask diff = 0.0000, 2524 camera-frames recolored.

**Correction (found on the full 50-episode run).** The claim that *both* inputs are
blue-restricted held for SAM2 only. SAM2's mask is intersected with the blue
threshold inside `build_sam2_cache._safe_mask`; the lightweight tracker's mask is a
**tracked** mask, not a per-pixel threshold map, so it can carry a rim of pixels
just outside the threshold. Measured: episode 39 frame 100 (front) had 210 such
pixels, all 8-adjacent to blue, hue 114.7, saturation 71.2 straddling the
`sat >= 70` floor. `build_hybrid_cache._blue_only()` now intersects the lightweight
mask with the same threshold too, making the union blue-restricted **by
construction** rather than by assumption. Cost: zero — frame coverage is unchanged
on all 50 episodes (the removed pixels never emptied a frame).

## Full dataset run (50 episodes, 25538 frames)

`scripts/run_full_augmentation.sh` drives the whole pipeline (SAM2 on 2 GPUs →
hybrid union → generate → verify → spot-check) with a gate between stages, and is
resumable: every stage skips finished episodes, so a re-run continues instead of
redoing ~2.5 h of SAM2.

| stage | result |
|---|---|
| SAM2 (episodes 3-49; 0-2 reused from the earlier runs) | 47/47 `status=ok`, median 319 s/episode (228-681 s) |
| up coverage (SAM2-safe) | median 0.818 (min 0.424, ep35) |
| front coverage (SAM2-safe) | median 0.785 (min 0.721) |
| hybrid union | 50/50 episodes, **0 with `sam2_missing`** |
| generate | 25538 frames, 41146 camera-frames recolored, 0 episodes left un-augmented |

Union vs its two inputs over the full dataset confirms the per-phase table above:

- **front**: the union never exceeds SAM2 (0/50 episodes) and exceeds the
  lightweight mask in **50/50** (up to +0.805, ep36). SAM2 strictly dominates
  (median 0.785 vs 0.540).
- **up**: the union exceeds SAM2 in **13/50** episodes (up to +0.394 ep2, +0.335
  ep35, +0.317 ep4) and the lightweight mask in 36/50 (up to +0.052); medians are
  close (0.815 vs 0.808), i.e. the lightweight tracker's contribution is
  concentrated in specific episodes, exactly as the per-phase numbers predicted.

### Generated dataset (`newdata_green`, 50 episodes)

`scripts/generate_augmented_dataset.py` writes the augmented episodes to
`/home/zjq/lerobot/dataset/newdata_green` via the official `LeRobotDataset.create`
+ `add_frame` + `save_episode` path (same AV1 codec, fps 30, same feature schema).
Verification vs the base dataset (`scripts/verify_generated_dataset.py`):

- **50 episodes / 25538 frames / splits `0:50` / fps 30**, task string unchanged.
- **25538 rows: `state` / `action` / `timestamp` bit-identical** (0 mismatches),
  episode/frame indices aligned, `meta/stats.json` state mean unchanged.
- mask hue blue -> green: up **59.1** (min 58.7), front **59.9** (min 59.5) — the
  median hue shift per episode, all well past the 40 bar.
- safety invariant re-proved dataset-wide: **0 recolored pixels were non-blue** in
  the base frame, **0 masks** exceed the 15000 px area gate.
- outside-mask drift from the AV1 re-encode: mean 1.47/255 (up) and 1.56/255
  (front), max 1.60/255 — lossy re-compression, NOT the recolor (recolor is
  strictly inside the mask; the online `ObjectColorAugmentedDataset` wrapper stays
  bit-exact outside the mask if zero re-encode drift is required).

Visual check: `outputs/generated_check/spotcheck/overview_front.png` (all 50
episodes, frame 0, front base | gen) and one `ep<NN>_sheet.png` per episode.

Load: `LeRobotDataset("newdata_green", root="/home/zjq/lerobot/dataset/newdata_green")`.

The previous 3-episode version is kept at
`dataset/newdata_green_20260916_2259` (3 episodes / 1684 frames).

## Run

```bash
cd /home/zjq/lerobot
# whole pipeline, resumable, gate between stages
bash workplace/object_generalization/scripts/run_full_augmentation.sh

# single stages
/home/zjq/miniconda3/envs/lerobot/bin/python \
  workplace/object_generalization/scripts/inspect_object_recolor.py \
  --root /home/zjq/lerobot/dataset/newdata --dataset newdata
```

Note: launch the driver detached (`setsid nohup … &`) — it runs for hours, and a
`LeRobotDataset.create` output directory cannot be resumed, so a teardown during
the generate stage means regenerating from scratch (the mask caches survive).

