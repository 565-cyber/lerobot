# Background Generalization — Stage 2B (desktop distractors)

RoboAgent (arXiv:2309.01918) augments along two axes: the **target object** and
the **scene**. Stage 2A (`object_generalization/`, blue cube → green cube)
covered a narrow version of the first. This directory is the second: keep the
workspace, add **clutter**, so the policy stops keying on the absence of
anything else on the table.

Evaluated on the real robot, in a cluttered scene, against the same policy
trained on the base data.

**No file under `src/lerobot` is modified.** Nothing is recoloured or moved:
the augmenter pastes objects and leaves every other pixel bit-identical.

## Results at a glance

All measured on `newdata` episodes 0/1 (568 / 583 frames); see
`outputs/distract/verify_report.json` after a run.

**Full set (K=1)**: all 50 episodes, 25538 frames, generated in 56 min and
verified `PASS` (phase 1 bit-identity, `mask_diff=0`, `paint_dropped=0`,
`paint_outside=0`); inspect sheets and the 50-tile overview are in
`outputs/distract_check/`. Peak worker VmRSS 5.29 GB against the 9.5 GB budget.
Every one of the 50 layouts places 2–4 objects (mean 3.06), none empty.
`mean_hidden` is 0.0 at the median and 0.032 at worst; the occlusion audit
behind limitation 3 is what makes those numbers mean anything.

| | ep0 | ep1 |
|---|---|---|
| legal area, raw (`freq <= 0.10`) | 33.5% | 35.3% |
| **legal area, after exclusions + margin** | **19.3%** | **20.8%** |
| largest connected legal blob | 45156 px | 46450 px |
| distractors placed per episode | 4 | 3 |

The gap between raw and final is almost entirely the **red container** and the
**erosion margin**: the container's interior is static, so `freq` there is ~0 and
it reads as a legal site, but it is a task-relevant object the augmenter must
never touch. Excluding it and the cube's trajectory, then eroding the margin,
removes 42.5% (ep0) / 41.1% (ep1) of the raw free area.

> **Episodes 0 and 1 were the two tightest under `freq <= 0.05`**, which is why
> they are the smoke set — and using their numbers to size the whole run is what
> made the density setting wrong for a while. Under `freq <= 0.10` they are no
> longer the extreme: the tightest are ep28 (18.2%) and ep6 (18.7%), and ep0 is
> the smallest blob (45156 px). Over the full set the legal area is min 18.2%,
> p25 21.7%, **median 23.8%**, p75 27.1%, max 34.4%, and the largest blob is
> min 45156 px, median 50790, max 63612.
>
> Read from `outputs/plate/episode_*/done.json` after a plate run — the gate in
> `run_background_augmentation.sh` compares against min 18.2% / 45156 px, so if
> these move the gate has stopped asserting anything.

**The area limit is physical, not an implementation choice.** The arm and its
shadow sweep most of the UP frame, so legal sites survive only in the left/right
margins and the top strip. Putting a distractor where the recorded trajectory
passed would make the `(state, action)` labels wrong — which is a different and
much worse failure than not augmenting at all.

Within that limit, density is a choice, and it is set to **2–4 objects per
episode** (`LayoutConfig.n_per_episode`). Over all 150 layouts of a K=3 run the
census places a mean of 3.05, min 2, max 4, with no layout empty
(`--dry-run` samples them all and fails if any does — see *Verification*).
`sample_layout` gives up after `max_site_attempts` and places fewer rather than
loosening a constraint, so no episode fails for want of room; the census is what
proves that it did not instead give up entirely.

## Pipeline

Four stages, each runnable and re-runnable on its own.

```bash
cd /home/zjq/lerobot
PY=/home/zjq/miniconda3/envs/lerobot/bin/python
S=workplace/background_generalization/scripts

# 0. everything at once, with stage gates and a resource sampler
$S/run_background_augmentation.sh                    # ~2.5 h, K=3

# ...or stage by stage
$PY $S/../masks/build_plate.py --episodes all ...        # stage 1, ~15 min, resumable
$PY $S/generate_distractor_dataset.py --variants 3 ...   # stage 2, ~2 h, NOT resumable
$PY $S/verify_distractor_dataset.py ...                  # stage 3
$PY $S/inspect_distractor_dataset.py --out /tmp/vis ...  # stage 4 -> pictures
```

Smoke first — two episodes, five minutes, scratch directory:

```bash
$PY $S/../masks/build_plate.py --episodes 0 1 --out /tmp/smoke/plate
$PY $S/generate_distractor_dataset.py --episodes 0 1 --variants 3 \
    --plate-root /tmp/smoke/plate --out /tmp/smoke/ds --report-dir /tmp/smoke/reports
$PY $S/verify_distractor_dataset.py --report-dir /tmp/smoke/reports --stride 10
```

### Stage 1 — `masks/build_plate.py`

Per episode, for the **UP camera only**: per-frame brightness gain (60th
percentile), per-pixel temporal median (`plate`), the fraction of frames each
pixel is far from the plate (`freq`), and the map of where a distractor may
legally sit (`placement`). No frame is warped — see *No alignment, deliberately*
below.

```
placement = (freq <= place_freq_max)
          ∩ ¬dilate(cube trajectory)      # the grasp site must stay reachable
          ∩ ¬dilate(red container)        # never touch a task-relevant object
          ⊖ erode(margin)                 # safety margin, as a RADIUS in px
```

The FRONT camera is a wrist camera and rides the arm (homography scale 0.12–1.31
within one episode); it has no usable plate, so it is not touched here.

### Stage 2 — `scripts/generate_distractor_dataset.py`

Renders each base episode `--variants K` times (default 3), each with its own
static layout. The K copies carry **identical actions** — only the visual context
differs — so "50 base episodes vs 50×K augmented" is a clean contrast that does
not confound "more data" with "more diversity".

```
aug = img                                          where alpha == 0  (bit-identical)
aug = (1-a)*img + a*(plate ⊕ objects) * L_t        otherwise
a   = paint_union * (1 - erase_t)                  arm/cube pixels hide the object
erase = foreground_t AND NOT (shadow_t AND shaded) the arm's shadow only shades
```

`paint_union` is the objects' silhouettes **and their contact shadows**. It has
to be both: the background layer bakes the shadows in, so a compositor handed
only the silhouettes blends that layer under the objects and drops the shadow
entirely — every object reads as a flat sticker, in a scene where the cube, the
container and the arm all cast one. `FrameCompositor` raises if its alpha does
not cover `Layout.paint_union`, because the wrong argument is invisible in any
single frame. `Layout.alpha_union` keeps its silhouette-only meaning: the
occlusion metric and the footprint assertions are about the object, not its
shadow.

`L_t` is the frame's illumination — `frame/plate`, per pixel — and is 1.0
outside a cast shadow, so it does not weaken the identity above.

#### Shade, don't erase

`foreground_t` is `|frame - plate| > 18`, which over-covers **on purpose**: it
includes the arm's *cast shadow* on the table, because the safe direction is to
hide a distractor rather than risk painting one onto the arm. Erasing on that
over-cover is wrong, and visibly so: the arm's shadow sweeps the whole resting
area at once as the arm enters and leaves, so every distractor blinks out
together. Measured on the smoke set, that was **all 6 variants reaching
`max_hidden = 1.0`** — at least one object fully erased — in runs of up to
1.4 s.

A shadow is not an occluder. It is the same plate under less light, so the
object under it is still there and belongs *darkened*, not deleted. The two are
separable per pixel with no extra calibration, because a shadow is
multiplicative: `frame_c / plate_c` takes the same value on R, G and B, which
something else in front of the plate does not.

The per-pixel test is then gated **per object**: an object is shaded only when
≥ `shadow_core_frac` (0.90) of its core is shadow-explained. That gate is the
safety argument — a black arm link lying across a distractor is still
shadow-*like* over most of its pixels one at a time, so a per-pixel rule would
keep the object and paint it onto the arm. Measured over all 40 occlusion runs
of the smoke set, the gate sits in an empty band: 36 shadow runs score 99.5–100%
and the remaining 4 score 65.7% or less.

> The "3 arm runs score 0%" in earlier revisions of this section came from a
> classifier that could not tell a deep shadow from an occluder, and the 3 runs
> it called the arm are now known to be the gain artefact, not the arm —
> see limitation 3 and the audit below. The *gate* argument is unaffected: what
> it needs is the shadows at 99.5–100% versus everything else, and that holds.

`shadow_k_lo` is set to 0.55 rather than just below the darkest measured shadow
(0.73), because the two mistakes do not cost the same: too high and a real
shadow is erased — the blink this exists to fix; too low and a mid-gray arm part
is shaded instead of erased, i.e. a distractor drawn **on** the arm.

Effect on the smoke set, per variant, with the layouts held fixed (same seed, so
the only difference is the compositing):

| gen | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `mean_hidden` before | 0.0352 | 0.0249 | 0.0315 | 0.0811 | 0.0581 | 0.0353 |
| `mean_hidden` after | 0.0 | 0.0 | 0.0 | 0.0034 | 0.0013 | 0.0034 |
| `max_hidden` before | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `max_hidden` after | 0.0624 | 0.0 | 0.0 | 0.6556 | 1.0 | 1.0 |

**Every variant used to fully erase at least one object**, and the fix removes
the shadow cause of that. What survives is *not* the arm, though this table
alone cannot say so — the audit in limitation 3 does: gen3's 0.66 and
gen4/gen5's 1.0 are all the gain artefact (`frame/plate` 0.96–1.13, channel-
consistent, so nothing is in front). This is why the number going to 0.0 would
have been the wrong thing to want: the metric was still reporting a real erasure,
and only the mechanism audit could say the erasure was still spurious.

The shading costs a real darkening on 157 frames (median 13.7% of the object,
max 32.5%, vs the erase it replaces) and ≤6.7% at the soft penumbra of the
shadow on 3144 more. Nothing else in the set changed.

The objects do not move during an episode, so the layout and the background
layer are computed once per episode, not per frame.

### Stage 3 — `scripts/verify_distractor_dataset.py`

Phase 1 is an exhaustive parquet check: `state` / `action` / `timestamp` /
`frame_index` bit-identity over **every** generated frame. Phase 2 works on
pixels at `--stride` (see *Verification* below).

### Stage 4 — `scripts/inspect_distractor_dataset.py`

Per-episode sheets (base | expected | written | `|written-expected|`), one
placement overlay per episode, and a grid overview of the whole set. This is the
S7 gate: **the arm, gripper, cube, container and table edges must be untouched**,
and the only new thing must be the objects.

## Design

### Minimal-change compositing, not background replacement

The obvious formulation — `aug = m*img + (1-m)*new_background` — replaces the
whole background, discarding the table's real shading, shadows and texture. The
change surface is large and the risk is high. This instead composites **only
under the objects**, which gives a hard, checkable invariant: *outside the
objects' alpha, the output equals the input bit for bit.*

### Foreground without SAM2

`foreground_t` is the frame-vs-plate residual (dilated, small components
dropped), unioned with the blue cube's mask. It **over-covers** by construction
— it includes the arm's shadow — so it can only hide a distractor, never paint
one onto the arm. No second 2.4 h SAM2 run.

### No alignment, deliberately

An earlier version estimated a per-frame ECC warp and resampled every frame into
frame-0 coordinates. It was removed, because it measured **worse than doing
nothing on every episode tried** — and it was the cause of the distractors
stepping around in the smoke video.

The UP camera is static, so there is nothing to align to: the frame-0 → frame-t
homography stays within 5.6 px across a whole episode on episode 0. Every warp
the estimator returns is therefore an estimate of something very small, and when
ECC locks onto the *arm* — which moves a lot — it returns something large and
wrong instead: up to 58 px, wider than the distractors themselves. Since the
object is composited at `to_frame_t(alpha, warp)`, a bad warp **displaces** it.
That is the flicker.

It also made everything else worse. Because the "aligned" frames were less
aligned than the raw ones, the median plate came out blurred (Laplacian variance
46–71 against 100–120 unwarped) and `freq` was inflated, which *shrank* the legal
area: on episode 0, 6.1% warped against 19.2% unwarped. A masked second pass that
restricted ECC to the static background was tried and is worse still, not better:
it leaves only smooth low-texture table, where the objective is ill-conditioned
and the drift grows (t95 25 px → 50 px).

So `build_plate._static_warps` returns identity warps. It still returns a
`(T, 2, 3)` array and still writes `warp.npy`, rather than teaching four
consumers to skip warping: the "is the scene static" assumption stays a value
that is read and checked (`PlateConfig.max_camera_shift_px`, and the
`camera_shift_px` field it writes to every ledger entry, which is now always
0.0) rather than a special case spread across the compositor and the verifier.

Object and foreground masks are transformed by the *same* `M_t`, so frame-0's
disjointness — "never paint on the arm" — survives any warp error, including a
large one. That is why this was a quality bug and not a correctness bug: a bad
warp could only move the object, inflate `freq`, and blur the plate.

### One safety coupling, enforced rather than documented

`LayoutConfig.fg_threshold` **must not exceed** the `PlateConfig.diff_threshold`
that built `placement`. If it did, the compositor would call pixels background
that the placement map called foreground, and a distractor could be painted onto
the arm. `build_plate` records its config in each `done.json`, the generator
reads it back, and `FrameCompositor.__init__` raises on a violation.

### Distractor assets are pluggable

`ProceduralCutoutProvider` draws a random box / cylinder / sphere / blob with
Lambertian shading and a contact shadow. Everything downstream only knows the
`CutoutProvider` protocol (`sample(rng) -> Cutout`), so **a real photographed
object can be dropped in later without touching the compositing or verification
code** — which matters, because procedurally generated objects are a distribution
shift from the real clutter used at rollout time.

The lighting is measured, not invented, from the blue cube on frames 0/10/20/30/40
of episode 0 (where it sits alone on the table and the arm is at home):

| | |
|---|---|
| shadow azimuth | 275° (image coords, +y down) |
| shadow core | table brightness × 0.57, at the contact edge |
| shadow reach | ≈ 1.0 × the silhouette's radius beyond its edge |
| shadow profile | 0.55 → 0.77 → 0.93 → 1.00 at 1.25 / 1.5 / 1.75 / 2.0 radii |
| table baseline | gray 181, RGB ≈ (176, 161, 171) |

> An earlier measurement around the red container reported 240° and is **wrong**:
> the container is ringed by the arm's diffuse shadow in the plate. The cube is
> the only clean probe, because its shadow leaves with it (gone by frame 300, and
> the plate removes it).

**The shadow is a gradient, not a translated copy.** It was one for a while — a
rigid offset duplicate at uniform darkness — and it rendered as a second grey
object sitting behind each distractor. The fix is a stack of copies spread
around the mean offset with a Gaussian weight (`shadow_spread_ratio`), which is
what a 45° source and an object resting on the table actually produce. The three
parameters are a grid-search fit to the four measured points above: `(0.30,
0.40, blur 1.0)` reproduces them to within 0.05 on a disc.

Validating it needs a shape-independent metric, though. `sqrt(area/π)` is an
equivalent-area radius, so sampling a box or a blob at "1.25 r" lands somewhere
different relative to its real edge, and the same fit reads as 0.19 error even
though the model is fine. The shape-free form of the measurement is the distance
beyond the silhouette at which the shadow drops below 10% of its edge value, in
units of that radius: the cube gives ≈ 1.0, this gives 0.98 (0.93–1.08 over 14
draws).

Colours avoid the blue-target and red-container HSV boxes. The check is applied
to the **shaded** object, in the target detector's own box — checking the drawn
base hue is not enough, because shading clips channels and slides a
near-boundary hue back inside. Measured on 300 cutouts, the base-hue-only check
left 2 (~0.7%) entirely inside the blue box; checking the shaded result takes
core pixels in that box from 0.41% to 0.0027%, with no cutout over the 2% budget
and the neutral-grey fallback never firing.

### The per-frame gain is the largest remaining defect

`build_plate._frame_gain` estimates each frame's exposure as
`percentile(grey, brightness_percentile=60)` and normalizes by it. That statistic
is not a light meter: the arm is dark and reaches **32%** of the frame, so once
the arm covers enough of the image the 60th percentile lands *inside the arm*.
`gains[t]` then tracks the arm's **area** — `corr(table_frac, gains)` is −0.80
(ep0) and −0.92 (ep3) — and dividing by it inflates the untouched table.

`illumination` and `foreground` both consume it, and `foreground` is an
absolute-difference test against a plate at frame-0 level. On ep1 at
t=568–570 the table's raw `frame/plate` stays at 0.96–1.006 while `gains` falls
195→162, so the correction lifts an unchanged table by 17%, the residual is 21
grey levels against `fg_threshold` 18, and the static table reads as an occluder.
That erases every distractor on it, and it accounts for **59 of the 72** hidden
runs over all 50 episodes — more than the shadow defect did. Up to 13.1% of the
legal area on ep3 is false foreground.

Raising the percentile does not fix it (p85 leaves the table at 1.068 with p95
1.091, because the arm can exceed 40% of the image). Estimating the gain as the
**median `frame/plate` inside the static table region** gives **exactly 1.000**
at p95 across ep0/1/3/7/36 and cuts the false foreground to 2.5% — the 2.5%
being the intended `fg_dilate_px` over-cover. It is not applied: it changes every
plate, so it needs a plate rebuild and a dataset regeneration. Note the
chicken-and-egg — the static mask comes from `freq`, which comes from the plate —
so it is a two-pass fix.

## Verification

### Two levels of "equal" — do not conflate them

Written datasets are **AV1 re-encoded**, so bit-identity can only hold in memory.

| level | assertion | how |
|---|---|---|
| memory | `composite(base) == base` wherever `alpha == 0`, bit for bit | hard failure |
| disk | `decode(gen) ≈ composite(decode(base))`, differing by AV1 drift | reported as mean/p99; **not** gated on max |

Phase 2 checks, per (base episode, variant):

1. **reproducibility** — rebuild the layout from its recorded seed and require it
   to match `layouts.jsonl` exactly. The generator stores layout *metadata*, not
   cutout pixels, so reproducibility is itself a checked claim.
2. **placement re-proof** — every footprint lies inside `placement`; `placement`
   is clear of the container and of the cube's trajectory under the same
   dilation stage 1 applied; and no pixel in `placement` has `freq` above the
   threshold. These are facts about the cache and do not depend on the
   compositor.
3. **in-memory identity** outside the alpha (hard).
4. **the background layer is safe and complete** — in frame-0 space, without any
   warp: everything the layer changed lies inside `placement` (safe), and
   everything it changed is covered by `paint_union` (complete). The second half
   is what a silhouette-only alpha silently violates: the shadows are baked into
   the layer and then never blended, and every frame still looks plausible on
   its own. Also asserts the baked shadows actually darken the layer.
5. **the alpha's foreground masking, cross-checked** — the compositor's alpha
   mask must equal one rebuilt from the cache by a second implementation
   (its own gain normalization, area floor, dilation, cube union and warp
   convention).

   > **This is not where the interesting failures are.** Everything above asks
   > whether the pipeline is *self-consistent*, and the shadow defect — which
   > erased a distractor whenever the arm's shadow crossed it — passed every one
   > of these checks, because the frames were exactly what the code intended.
   > `scripts/audit_occlusion.py` is the separate instrument for the question
   > these checks cannot ask: given that an object was hidden, *which mechanism*
   > hid it, and was that the right call. It changes no pass/fail, and it must be
   > re-run after any change to the shadow gate or the gain. Its
   > `--gate-override 2.0` mode is a self-test that disables the gate and must
   > light up with 36 `SHADOW` runs — a clean result from a detector that can
   > never fire is worth nothing.

   > The obvious phrasing — "no alpha pixel may sit on frame foreground" —
   > cannot fail. The compositor *defines* `a = paint * (1 - fg)`, so testing
   > `(a > 0) & fg` tests a property its own definition guarantees and reports 0
   > forever, which is exactly what it did before this was rewritten.

   **Read the scope before citing this.** The two implementations share the 18.0
   threshold by construction, so it cannot catch a threshold error; and a
   difference is only possible where the residual comes within `fg_dilate_px` of
   the paint region — measured over 57 sampled frames of ep0 v0 that opportunity
   set is **0 px**, because the arm never comes close to the legal strip. A 0
   here means "consistent where it was possible to differ", not "proven safe".
   What actually proves no distractor covers the arm is check 2's
   `freq[placement] ≤ place_freq_max`: every placement pixel was background in
   ≥ 95 % of frames. The `shadow_*` thresholds are shared config, so this check
   cannot catch a *wrong threshold* there either — only the gate's implementation
   drifting from its independent rebuild.

   Note the `erase`/`fg` split when reading the counters: `speckle_px` is
   measured against `foreground`, which since the shadow fix is a superset of
   what gets erased. It rose from 69–135 to 82–435 with no change in behaviour —
   the alpha it is measured against simply covers the shaded regions now.
6. **front photometry** — `front` is the base frame times the gain measured for
   that frame, within 2/255.
7. **AV1 drift**, plus the container specifically: `mean |gen - base|` on the
   container must stay under 3/255 (AV1 drift alone is ~1.5), which is what
   catches "a distractor got painted onto it".

## Known limitations — read before rollout

1. **Cross-view inconsistency.** The UP view gains distractors; the FRONT view
   does not. The front camera only receives a global photometric gain (disable
   with `--no-front-gain`). The front view is dominated by the gripper and the
   cube, so the effect is probably small — but it is an assumption, and it is
   exactly the kind that a rollout should be designed to detect.
2. **The applied gain is UP's.** The front camera's own exposure is not measured
   here; UP's gain is applied to it on the assumption that both track the same
   lighting. It is a global scalar, so it cannot move or recolour anything.
3. **Distractors are briefly hidden — and NOT by the arm.** A site gets swept
   for a few frames and the object is masked out for those; the mask is stable
   and the object sits on identical pixels whenever it *is* drawn, so there is
   no positional flicker, but it does briefly disappear. `audit_occlusion.py`
   classifies every one of those runs by mechanism, over all 50 episodes
   (K=1, 72 runs):

   | class | runs | mid-sequence | what it is |
   |---|---|---|---|
   | `SHADOW` | **0** | 0 | a cast shadow that erased instead of shading — the defect that was fixed |
   | `LIGHT` | **59** | 33 | the gain artefact in limitation 4 |
   | `DARK` | 13 | 13 | ambiguous: deep shadow or dark achromatic occluder |
   | `ARM` | **0** | 0 | the arm's body in front of the object |

   Two things to take from this. The shadow gate holds everywhere — 0 runs. And
   **nothing was occluded by the arm**: the 26 episodes whose `max_hidden`
   reaches 1.0 are all explained by `LIGHT`, not by the arm. An earlier revision
   of this list claimed the opposite (that the smoke set's boundary runs were
   genuine arm occlusions); `frame/plate` over their cores is 1.11–1.13, i.e.
   *brighter* than the plate and channel-consistent, so nothing is in front of
   them. The 13 `DARK` runs are genuinely undecided — a shadow deeper than
   `shadow_k_lo` and a dark neutral occluder are the same pixels, so they need a
   look at the frames rather than a threshold.
4. **The distractors are lit by the frame they are drawn into.** They are
   composited from `plate`, which has the arm's diffuse shadow removed by the
   temporal median, so a distractor standing where the arm's shadow falls would
   otherwise be drawn *unshadowed*. It is now multiplied by that frame's
   illumination `frame/plate`, so it darkens with the shadow like everything
   else.

   **The gain this uses is wrong, and it is the largest single cause of hidden
   distractors.** `_frame_gain` is `brightness_percentile = 60` of the frame's
   grey image, and the arm is dark and reaches 32% of the frame — so once the arm
   covers enough of the image the 60th percentile lands *inside the arm* and
   `gains[t]` tracks the arm's **area**, not the light. `corr(table_frac, gains)`
   is −0.80 (ep0) and −0.92 (ep3). On ep1 at t=568–570 the table's raw
   `frame/plate` never leaves 0.96–1.006 while `gains` falls 195→162, so the
   "correction" inflates an unchanged table by 17%. `foreground` is an
   absolute-difference test against that, the residual is 21 grey levels against
   `fg_threshold` 18, and the static table — where distractors are placed by
   construction — reads as an occluder and erases them. Up to **13.1%** of the
   legal area on ep3.

   Measured fix, scored on the table where `median(frame_norm/plate)` should be
   exactly 1.0: the global p60 gives median 1.067 / p95 1.091 and 13.1% false
   foreground; raising it to p85 barely helps (1.068 / 1.091, 12.3%); estimating
   the gain as the **median `frame/plate` inside the static table region** gives
   **1.000 / 1.000** and drops the false foreground to 2.5% (0.025 being the
   intended `fg_dilate_px` over-cover). Not yet applied: it changes every plate,
   so it needs a plate rebuild and a dataset regeneration.
5. **Procedural ≠ real.** See above. The real-object provider should be the main
   experiment once photographed, with this as the control.
6. **2–4 distractors per episode, at the frame edges.** The median episode has
   23.8% legal area, and it is no longer a thin strip: the widest inscribed
   circle in the legal region has diameter min 200 / median 236 / max 292 px,
   and on ep0 25921 of its 46961 legal px fall within the widest 40 px band.
   The old "confined to strips 27–75 px wide" figure was an ECC-era number and
   is wrong. The objects are still peripheral by necessity — the region hugs
   the left/right margins and the top and reaches the frame edges — but it is
   contiguous area, not a strip.
   "High-density clutter" — clutter in the workspace itself — is not reachable
   from this camera without invalidating the recorded `(state, action)` labels.

## Layout

```
background_generalization/
├── masks/build_plate.py            # stage 1: plate, freq, placement
├── distractors/procedural.py       # CutoutProvider protocol + procedural source
├── transforms/distractor_layout.py # site choice, background layer, FrameCompositor
├── transforms/background_replacement.py  # pure blend helper (Stage 2A, reused)
├── datasets/background_dataset.py  # Stage 2A's wrapper, unchanged (see below)
├── scripts/generate_distractor_dataset.py
├── scripts/verify_distractor_dataset.py
├── scripts/inspect_distractor_dataset.py
├── scripts/run_background_augmentation.sh
└── scripts/inspect_bg_dataset.py   # Stage 2A's wrapper smoke test, still valid
```

`datasets/background_dataset.py` is Stage 2A's in-process wrapper and is **not**
part of the Stage 2B path. The plan had it consume the new per-episode layouts;
it was left alone deliberately. It is not on the critical path (the generated
dataset is the deliverable), and a second, unverified implementation of the
compositing would undermine the one property this pipeline has — that there is
exactly one code path, and it is the one the verifier exercises.

## Rollback

Nothing here modifies the base dataset or `src/lerobot`. To undo a run:

```bash
rm -rf dataset/newdata_distract \
       workplace/background_generalization/outputs/plate \
       workplace/background_generalization/outputs/distract*
```

Stage 1 is resumable and its ledger records exactly which episodes need redoing;
stage 2 is not (`LeRobotDataset.create` refuses an existing directory), so move
`dataset/newdata_distract` aside before re-running it.
