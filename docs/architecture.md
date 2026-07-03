# Architecture & environment notes

---

## Overview — Phase 5 MVP summary (read this first)

### Pipeline

```
bundled nuScenes / Waymo scenarios
        │
        ▼
  MetaDrive ScenarioEnv (reset to logged start state)
        │
        ▼
  Pure-pursuit + PID tracking-controller expert
  (acts through EnvInputPolicy/real physics; reactive; queryable off-trajectory)
        │
        ▼
  BC dataset (obs, expert_action) pairs          DAgger dataset (policy-visited states, expert labels)
        │                                                │
        └──────────────── union ─────────────────────────┘
                              │
                              ▼
                   Behavior cloning: MLPPolicy
                   [128, 64] hidden, tanh output, MSE loss, AMP
                              │
                              ▼
                Closed-loop eval: policy drives ego through real physics
                on held-out nuScenes + Waymo scenarios (never seen in training)
                              │
                              ▼
                Metrics: route completion, success, collision, off-road, deviation
                + per-step lateral drift signal + action-gap diagnostic
```

### Observation & action specification

- **Observation:** `Box(-0.0, 1.0, (161,), float32)` — MetaDrive's default `LidarStateObservation`:
  ego kinematics (heading diff, speed, steering, last actions, yaw rate), side/lane-line distances
  or lidar cloud points (12 + 0 = 12 side + 1 lane = 13 non-lidar dims), 120-point lidar cloud,
  and 4 nearby-vehicle relative poses — all in [0,1]. Navigation navi info (10 dims) included.
- **Action:** `Box(-1.0, 1.0, (2,), float32)` = `[steering (norm), throttle_brake (norm)]`
- **Expert:** pure-pursuit (lookahead = 5m + 0.5·speed) for steering; PID (kp=0.6, ki=0.05, kd=0.05)
  tracking logged speed profile for throttle/brake. Closed-loop, queryable at any state.

### VRAM budget

- GPU: NVIDIA RTX A1000 Laptop GPU, 4096 MiB total, ~4085 MiB free at idle (driver 570.211.01, CUDA 12.8).
- Training peak VRAM: **17.9 MiB** (`torch.cuda.max_memory_allocated()`). AMP + pin_memory in place.
- MetaDrive top-down renderer runs entirely on CPU (numpy/pygame). GPU never touched during sim.

### Headline result: the debugging arc (see `outputs/headline_figure.png`)

| stage | open-loop val MSE | success | off-road | route completion (overall) | nuScenes | Waymo |
|---|---|---|---|---|---|---|
| (i) heuristic labels (Phase 2 original) | 0.1405 | 0% | 33.3% | 29.7% | 7.8% | 51.5% |
| (ii) reactive-expert labels, BC only (Phase 4.5) | 0.0678 | 0% | 100% | 19.7% | 9.2% | 30.2% |
| (iii) BC + 1 DAgger iteration (Phase 4.6) | 0.0642 | **16.7%** | 83.3% | **37.5%** | 12.1% | **62.8%** |

**The headline:** open-loop val loss monotonically improved (0.1405 → 0.0678 → 0.0642) while
closed-loop route completion *did not* — it dropped at stage ii (better labels, worse closed-loop)
before DAgger lifted it above stage i. This single picture demonstrates that **one-step imitation
loss does not predict closed-loop driving quality**.

### Diagnosed cause (Phase 4.6 verdict)

Two-stage diagnosis, both stages confirmed by direct measurement:
1. **Label–dynamics mismatch (fixed):** the original heuristic throttle pseudo-labels were not
   physically consistent; when replayed through real physics they achieved only 49.9% mean route
   completion with half ending in collision. Fixed by the reactive tracking-controller expert (95%+
   standalone completion). This is why stage ii has lower open-loop loss but *worse* closed-loop
   than stage i.
2. **Covariate shift (partially addressed by DAgger):** after fixing the labels, the lateral-drift
   diagnostic (`outputs/lateral_drift.png`) showed every eval rollout climbing smoothly to the
   off-road boundary — textbook compounding steering error. The action-gap diagnostic
   (`outputs/action_gap.png`) showed the expert/policy steering disagreement growing steadily over
   policy-visited training states. One DAgger iteration (543 new transitions, 1402 total) lifted
   overall route completion from 19.7% → 37.5% and produced the first genuine success
   (Waymo:0, 95.8% completion). The improvement was stronger on Waymo than nuScenes; with n=3
   per dataset this asymmetry is noted but not over-interpreted.

### Failure bucketing

| stage | success | off-road | collision | timeout/stall |
|---|---|---|---|---|
| (i) heuristic labels | 0 | 2 | 0 | **4** — car stalls/wanders |
| (ii) reactive BC | 0 | **6** | 0 | 0 — all leave road quickly |
| (iii) BC+DAgger | **1** | 4 | 1 | 0 — making more progress → more collisions |

Signature shift: stage i = stall-dominated (heuristic throttle → car barely moves);
stage ii-iii = off-road-dominated (covariate shift → steering diverges from route).

### Written analysis

The gap between open-loop loss and closed-loop route completion in this project has a two-part
explanation established by direct measurement, not asserted by analogy.

**Stage i → ii (open-loop improves, closed-loop gets worse):** the jump from heuristic to
reactive-expert labels cut the open-loop val MSE by half (0.14 → 0.068) because the new labels are
physically consistent — the expert itself achieves 95%+ route completion when driven in simulation.
Yet closed-loop route completion *dropped* (29.7% → 19.7% overall, 51.5% → 30.2% on Waymo). This
happened because the reactive-expert labels produce more demanding actions (sharper corrections,
active throttle modulation) that are harder to imitate faithfully from a small dataset. The trained
policy begins to attempt corrections at road boundaries but over- or under-corrects, triggering
off-road departures that never happened with the stall-prone heuristic-label policy. This is a
classic covariate-shift setup: at the boundary states where correction matters most, the policy has
never seen training examples at those exact states.

**Stage ii → iii (DAgger partially closes the gap):** DAgger directly addresses the training-set
hole: it rolls the policy out under real physics (so it reaches boundary states it actually visits),
then labels *those* states with the reactive expert. With only one iteration (543 new transitions),
the Waymo route completion jumped from 30.2% → 62.8% and the project's first genuine closed-loop
success appeared. The lateral-drift and action-gap diagnostics provided two independent measurements
confirming the covariate-shift mechanism before DAgger ran, so the improvement confirmed the
diagnosis rather than just being a black-box improvement in numbers.

**What remains:** nuScenes improved much less (9.2% → 12.1%); all three nuScenes eval scenarios
still leave the road. This may reflect harder geometry (dense urban intersections vs Waymo highway-
adjacent roads), but also suggests that one DAgger iteration does not fully close the gap when the
base dataset is tiny (859 BC + 543 DAgger transitions total). Additional iterations (Phase 6) or a
larger training corpus would be the natural next steps. Be honest about this in any writeup: the
project demonstrates the covariate-shift mechanism and shows DAgger working in principle on real
scenarios from two datasets — it does not solve the problem at this data scale.

### Key output files

| file | description |
|---|---|
| `outputs/headline_figure.png` | Two-panel headline figure (open-loop loss vs closed-loop completion) |
| `outputs/lateral_drift.png` | Lateral offset vs. step — covariate-shift signature |
| `outputs/action_gap.png` | Expert/policy steering gap growing over rollout |
| `outputs/bc_vs_dagger_waymo0.gif` | Side-by-side BC (fail, step 18) vs DAgger (success, step 82) |
| `outputs/bc_best.pt` | BC-only checkpoint (reactive-expert labels, stage ii) |
| `outputs/bc_dagger.pt` | BC+DAgger checkpoint (one iteration, stage iii) |
| `outputs/metrics.json` | BC-only closed-loop metrics (per dataset) |
| `outputs/metrics_dagger.json` | BC+DAgger closed-loop metrics (per dataset) |
| `outputs/metrics_before_fix.json` | Heuristic-label BC metrics (pre-Phase 4.5, stage i) |
| `outputs/failure_buckets.json` | Success/off-road/collision/timeout counts per stage |
| `outputs/dagger_progress.png` | Success rate + route completion vs. DAgger iteration (Phase 6a) |
| `outputs/bc_vs_dagger4_nuscenes7.gif` | Side-by-side: BC off-road at step 116 vs DAgger iter4 success at step 225 |
| `outputs/bc_dagger4.pt` | Final DAgger checkpoint (4 iterations) |

---

## Phase 0 — environment

- Conda env: `e2ecl`, Python 3.11.15
- GPU: NVIDIA RTX A1000 Laptop GPU, driver 570.211.01, CUDA 12.8
- Free VRAM at idle (before any training/sim): ~4085 MiB / 4096 MiB total (`nvidia-smi`, 2026-06-28)
- `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` — `torch.cuda.is_available()` returns `True`
- MetaDrive installed from source (editable) at `metadrive/`, version 0.4.3
- ScenarioNet installed from source (editable) at `scenarionet/`
- Bundled assets (incl. mini nuScenes/Waymo scenario splits) pulled via `python -m metadrive.pull_asset`
  into `metadrive/metadrive/assets/{nuscenes,waymo}/`
- Verified: `ScenarioEnv` loads the bundled nuScenes split, steps under the default policy, and
  `env.render(mode="top_down")` produces a real top-down frame of the reconstructed road network
  with agents — confirmed visually (frame saved during verification, not committed).

## Phase 1 — env wrapper & observation/action spaces

- `src/env/make_env.py` exposes `make_train_env()`, `make_eval_env()`, and
  `make_expert_collect_env()` (training-range env with `agent_policy=ReplayEgoCarPolicy`, used only
  by Phase 2). All three build on MetaDrive's `ScenarioEnv` and read scenario split + dataset choice
  from `configs/default.yaml`.
- **Scenario index == seed.** `ScenarioDataManager` only supports a *contiguous* range
  (`start_scenario_index`, `num_scenarios`); there's no arbitrary-index-list option in the installed
  API. Train/eval splits are therefore two disjoint contiguous ranges within the bundled nuScenes
  mini split (10 scenarios total): train = indices `[0, 8)`, eval = indices `[8, 10)`. `env.reset(seed=i)`
  must use `i` inside that env's own configured range or it asserts.
- **Dataset:** bundled `nuscenes` mini split (10 scenarios; `waymo` split has only 4 and is available
  as an alternate `dataset:` value in the config).
- **Observation space (default, `state_vector` mode):** `Box(-0.0, 1.0, (161,), float32)` — MetaDrive's
  default `LidarStateObservation`: ego kinematics/navigation state + side/lane-line distances + 240
  lidar points + nearby-vehicle info. No custom obs code needed; this *is* the compact state-vector
  representation the roadmap recommends as the MVP default.
- **Action space:** `Box(-1.0, 1.0, (2,), float32)` = `[steering, throttle/brake]`, consumed by the
  default `EnvInputPolicy` (i.e. whatever the policy outputs each step actually drives the car through
  physics — this is the same interface the BC policy and closed-loop eval will use).
- **Termination / metrics signals available in `info`** (for Phase 4): `info["route_completion"]`
  (float), and `TerminationState` keys `arrive_dest` (success), `out_of_road`, `crash`,
  `crash_vehicle`, `crash_object`, `crash_human`, `max_step` (see `metadrive/constants.py`).
- `outputs/demo_replay.gif` (168 frames, training scenario index 0, `ReplayEgoCarPolicy`) confirms a
  real logged ego trajectory replaying with visibly moving traffic — generated by
  `python -m src.eval.record_video`.
- **Known cosmetic issue, not blocking:** the top-down camera frames the whole local road network
  rather than tightly tracking the ego at a small radius; revisit camera args
  (`scaling`, `target_agent_heading_up`) when producing the Phase 7 showcase GIF.

## Phase 2 — expert dataset collection

- **The bundled scenario data has no logged control channel.** `ReplayEgoCarPolicy` (the "expert")
  drives the ego by teleporting it to the logged position/heading/velocity each 0.1s step
  (`metadrive/policy/replay_policy.py`) — it bypasses vehicle physics entirely, so there is no
  literal "steering/throttle the human used" to read off. Since this project's action space is
  `[steering, throttle_brake]` (the same space used by closed-loop eval), `src/data/collect_expert.py`
  derives a **pseudo-action** per transition instead of using a real recorded action:
  - `steering = atan(wheelbase * curvature) / max_steering_rad`, `curvature = yaw_rate / speed` —
    an exact kinematic-bicycle-model inversion (geometric, independent of engine dynamics).
  - `throttle_brake = clip(accel / 3.0, -1, 1)`, `accel = (speed[t+1] - speed[t]) / 0.1` — a
    **heuristic** normalization (3 m/s² is a plausible urban accel/decel scale), *not* an inversion
    of MetaDrive's nonlinear engine-force/brake model. This is the weakest approximation in the
    pipeline and a plausible contributor to the open-loop/closed-loop gap story in Phase 5 — flagged
    here for the write-up's honesty section.
  - obs[t] is paired with the pseudo-action computed from the track's state[t]→state[t+1] transition.
- Confirmed via direct inspection: `wheelbase = FRONT_WHEELBASE + REAR_WHEELBASE ≈ 2.469 m`,
  `max_steering = 40°` for the default ego vehicle class (`DefaultVehicle`); track timestep `dt = 0.1s`
  (confirmed from scenario `metadata["ts"]` diffs) for the bundled nuScenes split.
- Collected from the 8 training scenarios (indices 0–7): **1181 transitions**, obs shape `(1181, 161)`,
  action shape `(1181, 2)`. Steering mean ≈ 0, std ≈ 0.09, range [-0.45, 0.40] (centered, as expected).
  Throttle/brake mean ≈ 0, frac_positive ≈ 0.50 (balanced accel/decel, plausible for urban driving
  with stops). Saved to `data/bc_dataset.npz` (arrays `obs`, `act`).
  Per-scenario transition counts varied a lot (1 to 192) — verified this is **not a bug**: MetaDrive's
  `_is_arrive_destination` ends an episode once `route_completion > 0.95`, and several bundled clips
  are short, low-speed logged segments that reach 95% route completion in well under their nominal
  length (one clip, scenario 2, is essentially stationary and "arrives" after 1 step).
  - **Caveat vs. the roadmap's "a few thousand+ transitions" target:** the bundled nuScenes mini split
    only has 10 scenarios total (8 train / 2 eval here), capping the achievable transition count well
    below "thousands" regardless of collection method. 1181 transitions is a reasonable training set
    for a small MLP on a 161-dim input, but is honestly a demonstrator-scale dataset, not a benchmark
    one — consistent with this project's stated scope (Section 2: "a demonstrator of the methodology").
- Train/eval split is reproducible via `configs/default.yaml`'s `scenario_split` (deterministic
  contiguous ranges, not a stored ad-hoc list) — eval indices `[8, 10)` were never used for collection.

## Phase 3 — behavior-cloning model & training

- `src/models/mlp_policy.py`: `MLPPolicy`, plain MLP with `tanh` output head (actions live in
  `[-1, 1]`, matching the env's action space exactly).
- `src/train/train_bc.py`: loads `data/bc_dataset.npz`, splits 90/10 by transition (random, fixed
  seed), trains with MSE loss, AMP autocast + `GradScaler`, `pin_memory`, checkpoints the best-val-loss
  epoch to `outputs/bc_best.pt` (model state dict + full config + epoch + val loss).
- **First run, config default (`hidden_sizes=[256,256]`, 50 epochs, no weight decay): overfit hard.**
  Train loss fell from 0.143→0.060, but val loss bottomed out at epoch 8 (0.143) then rose
  monotonically to 0.21+ by epoch 50 — expected, since a ~108k-parameter net is heavily
  overparameterized for 1063 training transitions (Phase 2's dataset-size caveat compounding here).
  Caught from the loss plot, not assumed.
- **Fix:** shrank the net to `hidden_sizes=[128, 64]` (~29k params), added `weight_decay=1e-4`, and
  added early stopping (`patience=10` epochs on val loss) — all now in `configs/default.yaml`. Re-run:
  stopped at epoch 24, best checkpoint at epoch 14 (val loss 0.1453). Val curve is noisy (only 118
  validation transitions) but trends down into the selected checkpoint rather than diverging — a much
  more honest "decreasing val curve" than the first run. Saved to `outputs/bc_loss.png`.
- **VRAM:** peak allocated 17.9 MiB during training (`torch.cuda.max_memory_allocated()`), i.e.
  negligible against the 4096 MiB budget — expected, since a state-vector MLP this small does not
  stress the GPU at all. AMP/batch-size hygiene from the roadmap is in place but is not load-bearing
  at this model scale; it will matter more if/when Phase 6c's BEV-CNN is built.
- Open-loop validation loss being low is *not* evidence of good driving — that's exactly the
  covariate-shift question Phase 4 (closed-loop eval) and Phase 5 (the headline figure) exist to
  answer.

## Phase 6a — more DAgger iterations (stretch)

Ran 3 additional DAgger iterations (2, 3, 4) beyond the Phase 4.6 result using
`src/train/run_dagger_iterations.py`. Each iteration: roll out latest policy on training scenarios,
query the reactive expert on visited states, aggregate all prior data, retrain the same MLP,
evaluate on held-out eval set. Final aggregate at iter 4: 859 (BC) + 543+722+864+893 (DAgger iters
1-4) = 3881 total transitions.

| iter | new transitions | overall success | nuScenes success | Waymo success | overall RC |
|---|---|---|---|---|---|
| 0 (BC only) | — | 0% | 0% | 0% | 19.7% |
| 1 | 543 | 16.7% | 0% | 33.3% | 37.5% |
| **2** | 722 | **66.7%** | 66.7% | 66.7% | 76.7% |
| 3 | 864 | 66.7% | 66.7% | 66.7% | 75.3% |
| 4 | 893 | 66.7% | 66.7% | 66.7% | 76.1% |

**Result:** success rate rises from 0% (BC) to 16.7% (iter 1) to **66.7% (iter 2)** then plateaus.
Route completion peaks at ~77% overall and ~85% on Waymo. Iterations 2-4 plateau at 4 of 6
held-out scenarios consistently succeeding; the one persistent failure is `nuscenes:6` (off-road in
~20 steps at every iteration — a geometrically challenging intersection that the training coverage
never adequately captures). The rising-then-plateau shape is the expected DAgger convergence pattern
for a bounded dataset: early iterations fill the most critical distribution gaps, later iterations
add coverage at states already well-represented. See `outputs/dagger_progress.png`.

Side-by-side `nuscenes:7` GIF (`outputs/bc_vs_dagger4_nuscenes7.gif`): BC-only goes off-road at
step 116; DAgger iter4 succeeds at step 225 on the same scenario/starting state.

## Phase 4 — closed-loop evaluation

- **Widened the eval split before this phase** (see "scope change" note): the original 8/2 train/eval
  split left only 2 held-out scenarios, too few for a meaningful success-rate statistic or for 3+
  distinct rollout GIFs. Changed `scenario_split` to train `[0,6)` / eval `[6,10)` and **redid Phase 2
  collection (803 transitions from 6 scenarios) and Phase 3 retraining** (best checkpoint epoch 14,
  val loss 0.1405) under the new split before running closed-loop eval.
- `src/utils/metrics.py`: per-episode `success`/`collision`/`off_road`/`max_step` from
  `TerminationState` keys in `info`, `route_completion` from `info["route_completion"]`, and
  `mean_route_deviation` from `abs(vehicle.navigation.current_lateral)` averaged over the episode —
  this is exactly the lateral offset from the logged route, since `TrajectoryNavigation` builds its
  reference path from the logged ego track itself.
- `src/eval/closed_loop_eval.py`: loads `outputs/bc_best.pt`, drives the ego with the policy's own
  action each step (deterministic, no exploration noise) on `make_eval_env()` (default
  `EnvInputPolicy`, i.e. real physics — not replay), for every held-out scenario index, to
  termination or the configured horizon (1000 steps).

### Headline result (the covariate-shift gap, made visible)

| metric | value |
|---|---|
| episodes | 4 |
| success rate | 25.0% |
| collision rate | 0.0% |
| off-road rate | 25.0% |
| timeout (max_step) rate | 50.0% |
| mean route completion | 5.9% |
| mean route deviation | 1.78 m |

Despite a low-ish open-loop validation loss (0.14 MSE, Phase 3), the closed-loop policy **fails to
make meaningful route progress in 3 of 4 held-out scenarios** (route completion 0–11%) — it either
drifts off-road quickly (scenario 6, 25 steps) or stalls/wanders near the start until timeout
(scenarios 7 and 8, full 1000-step horizon, route completion 3.3% and 11.4%). This is the textbook
open-loop-vs-closed-loop gap the project is built to demonstrate: a policy that fits the logged
one-step transitions reasonably well still drifts off the training distribution once its own
(imperfect) actions compound over a rollout. This is Phase 5's headline figure.
- **Known artifact, not a real success:** scenario 9 reports `success=True` at `route_completion=0%`
  after a single step. This is the same degenerate case seen in Phase 2 (scenario 2 of the old split):
  `_is_arrive_destination()` returns `True` whenever
  `vehicle.navigation.reference_trajectory.length < 2`, i.e. the logged route is essentially
  stationary, regardless of policy quality. Treat the 25% success rate as inflated by this artifact —
  the honest headline is 0/3 *meaningful* successes among scenarios with a non-trivial route.
- Saved `outputs/metrics.json` (full per-episode data) and 4 rollout GIFs
  (`outputs/rollout_{6,7,8,9}.gif`, capped at 300 frames each for file size; full metrics used the
  uncapped 1000-step horizon). Visually confirmed (frames inspected): scenario 6 shows the ego
  leaving the paved road near a curve; scenario 8 shows the ego on-road but not progressing —
  exactly the "drift off-road" vs. "stall" failure modes Phase 5 asks to bucket.
- Reproducible end-to-end via `python -m src.eval.closed_loop_eval`.

## Phase 4.5 — diagnose before naming a cause (ROADMAP.md update)

The roadmap was updated to insert this phase after the Phase 4 result looked suspiciously like a
"car that won't move" problem rather than a textbook drift-then-fail covariate-shift signature.
Ran all required steps; full evidence chain below. `outputs/metrics_before_fix.json` is the
cleaned/expanded-eval-set baseline with the *original* heuristic-label model; `outputs/metrics.json`
is the final Phase 4 result with the fixed pipeline (these intentionally tell different parts of the
story — see verdict).

**Step 1 — per-dimension loss vs. naive baselines** (`outputs/loss_by_dim.{json,png}`), original
heuristic-label model: steering MSE 0.00073 vs. zero/mean baselines ~0.0033 (model ~4x better — genuinely
learned); throttle_brake MSE 0.280 vs. baselines 0.306/0.312 (model only ~8% better than guessing the
training mean — **essentially never learned throttle**). This pointed straight at the throttle
pseudo-label as suspect, exactly as flagged when it was created in Phase 2.

**Step 2 — decisive test** (`outputs/expert_replay_check.json`): fed the original heuristic
pseudo-action labels for all 6 training scenarios through *real physics* (`make_train_env()`,
`EnvInputPolicy`) from each logged start state. Mean route completion: **49.9%**, with 3 of 6
scenarios ending in collision (scenarios 3, 4, 5) and scenario 0/1 stalling at 12.3%/25.1%. **Verdict:
label–dynamics mismatch, not covariate shift.** BC was never going to drive these routes — its own
ground-truth labels can't.

**Step 3 — obs-drift: deferred at first** (only meaningful once labels are vindicated; they weren't
yet), revisited after the fix (see below).

**Step 4 — cleaned + expanded eval set** (regardless of verdict): surveyed all 10 nuScenes + 3 bundled
Waymo scenarios for `reference_trajectory.length < 2` (near-stationary logged routes that
`_is_arrive_destination` marks "arrived" after one step regardless of policy quality). Found 2
degenerate nuScenes scenarios (indices 2 and 9); index 9 was in the eval range and is now excluded
via `scenario_split.eval_exclude_indices`. Also discovered the bundled Waymo split has **3** scenarios,
not 4 as assumed in Phase 2 (`configs/default.yaml`'s `eval_extra_num: 3`). New eval set: nuScenes
{6,7,8} + Waymo {0,1,2} = 6 genuinely held-out, non-degenerate scenarios, evaluated via
`make_eval_env()` + a new `make_eval_env_extra()` builder, combined in `closed_loop_eval.evaluate()`.
Re-ran Phase 4 with the **original** model on this cleaned/expanded set as a baseline
(`outputs/metrics_before_fix.json`): 0% genuine success (the old 25% was entirely the degenerate-
scenario artifact), 33.3% off-road, 66.7% timeout, mean route completion 29.7%.

**Conditional fix — pure-pursuit + PID tracking-controller expert**
(`src/data/tracking_expert.py`): since step 2 implicated the labels, replaced the open-loop
kinematic-inversion pseudo-action expert with a closed-loop controller that reacts to the vehicle's
*actual* current state each step: pure pursuit (lookahead = 5m + 0.5·speed) for steering, PID
(kp=0.6, ki=0.05, kd=0.05) tracking a logged speed profile for throttle/brake. It drives through
`make_train_env()`'s real `EnvInputPolicy` physics, so the recorded action *is* the action applied —
no after-the-fact inversion, and it's queryable at any state (not just logged-path points), which is
what would make DAgger (Phase 6a) meaningful later. **Standalone validation on the 6 training
scenarios: 95%+ route completion on 5 of 6** (vs. 49.9% mean for the old labels); the 6th is the known
degenerate scenario 2.
- Re-collected Phase 2 with the new expert: 859 transitions (`src/data/collect_expert.py`'s
  `collect_scenario`, old heuristic kept as `collect_scenario_legacy_heuristic` for reference).
- Retrained Phase 3: val loss 0.140 → **0.072**, and the loss curve is now a clean decreasing curve
  with no overfitting/early-stop needed (`outputs/bc_loss.png`).
- Re-ran step 1 on the new model: throttle MSE **0.143 vs. baselines 0.257/0.228** — now meaningfully
  better than both naive baselines (vs. ~8% before), i.e. the model genuinely learned throttle this
  time. Confirms the original underfitting-looking throttle signal was a label-quality artifact, not
  a model-capacity problem.

**Re-ran Phase 4 with the fixed pipeline** (`outputs/metrics.json`, the now-canonical Phase 4 result):

| metric | before fix (cleaned/expanded set) | after fix |
|---|---|---|
| success rate | 0.0% | 0.0% |
| collision rate | 0.0% | 0.0% |
| off-road rate | 33.3% | **100.0%** |
| timeout rate | 66.7% | 0.0% |
| mean route completion | 29.7% | **17.9%** |
| mean route deviation | 2.06 m | 1.59 m |

**This is the surprising part: closed-loop performance got *worse* after the fix**, despite much
better labels (95% standalone completion vs. 49.9%) and much better open-loop learning (val loss
0.072 vs. 0.140, throttle now genuinely learned). Every eval episode now ends in an off-road
departure rather than stalling/timing out.

**Step 3, now applicable** (`outputs/obs_drift.png`): with labels vindicated, measured nearest-
neighbor L2 distance from each closed-loop observation to the training-obs set, over rollouts of 21
steps (scenario 6) and 97 steps (scenario 7). **Distance does not show a growing trend** — it
oscillates roughly in the 2.6–4.1 range throughout, including a *decrease* through the second half of
the 97-step rollout. Per the roadmap's own criterion ("if it doesn't grow, the failure is upstream —
labels/underfitting, not distribution drift"), this is evidence **against** a classic compounding-
drift covariate-shift signature.

### Diagnosed conclusion

Two distinct causes were established, in sequence, with direct evidence for each. **(1) The original
near-total Phase 4 failure was predominantly label–dynamics mismatch**: the heuristic pseudo-action
expert's own labels could not drive the logged routes through real physics (49.9% mean completion,
half ending in collision) — BC was fitting noise on the throttle dimension, confirmed by both the
per-dimension loss test and the decisive replay test. This is now fixed via a closed-loop pure-
pursuit + PID tracking-controller expert that achieves 95%+ standalone route completion. **(2) After
fixing the labels, the residual closed-loop failure (now manifesting as off-road departures rather
than stalls, and if anything a *lower* mean route completion) is best attributed to underfitting /
training-data scarcity, not covariate shift**: open-loop loss improved substantially and the obs-drift
measurement shows no growing-distance signature during rollouts, which the roadmap's own criterion
treats as evidence against compounding drift. With only 5 usable training scenarios (859 transitions,
one dataset) for a generic small MLP, the policy most likely never saw enough road-geometry diversity
to generalize to held-out scenarios from either nuScenes or an entirely different dataset (Waymo) —
a data-scarcity story consistent with this project's own "demonstrator, not benchmark" framing
(Section 2). Phase 5 should lead with **this two-stage diagnosis** (label-dynamics mismatch, fixed;
residual failure = underfitting/data scarcity, not asserted covariate shift) rather than a generic
covariate-shift headline.

**Superseded by Phase 4.6 below** — the obs-drift test here used a 161-dim nearest-neighbor distance
that turned out to be insensitive to the one dimension that actually mattered (lateral offset), and
was measured on rollouts too short (21–97 steps) for a trend to clearly form. Phase 4.6 redid this
diagnosis with a sharper signal and found real evidence of covariate shift after all.

## Phase 4.6 — disambiguate the residual failure & run one DAgger iteration (ROADMAP.md update)

Phase 4.5's fix improved labels and open-loop fit but made closed-loop **uniformly worse** (off-road
33%→100%), which is itself a covariate-shift-shaped signature, not obviously "underfitting" — so the
roadmap was updated to settle this properly rather than rest on the underpowered NN-distance test.

**Step 1 — the right drift signal** (`outputs/lateral_drift.png`): plotted
`abs(vehicle.navigation.current_lateral)` vs. step for the BC policy on all 6 eval scenarios. Every
single rollout shows a **clean, smooth, monotonically accelerating climb** to the 4m off-road
threshold (`max_lateral_dist`) — textbook compounding steering error. This is the signal the Phase
4.5 161-dim observation-space nearest-neighbor test was structurally blind to.

**Step 2 — split metrics by eval dataset** (now baked into `outputs/metrics.json`'s `{overall,
nuscenes, waymo}` structure, via `aggregate_metrics_by_dataset` in `src/utils/metrics.py` and
`closed_loop_eval.py`'s `label="dataset:idx"` episode tagging). Confirms training (nuScenes-only) vs.
eval (nuScenes + Waymo) was a domain-shift confound worth isolating before drawing conclusions.

**Step 3 — diagnose and collect DAgger labels in one pass** (`src/data/collect_dagger.py`): rolled
out the BC policy (`outputs/bc_best.pt`) under real physics on the 6 **training** scenarios; at every
visited state, queried the reactive tracking-controller expert (`tracking_expert.py` — well-defined
off-trajectory by construction) for the action *it* would take there. `outputs/action_gap.{json,png}`
shows the `|expert_steering − policy_steering|` gap **growing steadily over the rollout** in every
non-degenerate scenario (e.g. scenario 1: ~0 → 0.7 over 300 steps) — a large, *systematic*, growing
gap, exactly the demonstrated (not asserted) covariate-shift signature the roadmap asked for. This
pass simultaneously collected the DAgger dataset: 543 (visited-state, expert-action) pairs.
- Note: a checkpoint-naming mistake during this phase briefly overwrote the Phase 4.5 `bc_best.pt`/
  `bc_loss.png` with the DAgger run's output before `train_bc.py`'s CLI exposed `--ckpt_name`; the
  DAgger artifacts were rescued (renamed to `bc_dagger.pt`/`bc_dagger_loss.png`), the BC-only
  checkpoint was retrained fresh from the same dataset/config (val loss 0.068, statistically
  consistent with the original 0.072 but not byte-identical weights), and all dependent
  artifacts (`metrics.json`, `lateral_drift.png`, `action_gap.*`, rollout GIFs) were regenerated
  from it for internal consistency. `train_bc.py` now takes `--ckpt_name`/`--loss_plot_name` to
  prevent recurrence.

**Step 4 — aggregate + retrain:** `D_bc` (859, tracking-expert labels) ∪ `D_dagger` (543) = 1402
transitions → retrained the same MLP/config → `outputs/bc_dagger.pt` (val loss 0.064).

**Step 5 — closed-loop re-eval, per dataset** (`outputs/metrics_dagger.json`), full BC → BC+DAgger
progression with the earlier Phase 4.5 stages for context:

| stage | success | collision | off-road | mean route completion (overall) | nuScenes RC | Waymo RC | mean deviation |
|---|---|---|---|---|---|---|---|
| (i) heuristic labels (`metrics_before_fix.json`) | 0% | 0% | 33.3% | 29.7% | — | — | 2.06 m |
| (ii) tracking-expert labels, BC only (`metrics.json`) | 0% | 0% | 100% | 19.7% | 9.2% | 30.2% | 1.53 m |
| (iii) BC + 1 DAgger iteration (`metrics_dagger.json`) | **16.7%** | 16.7% | 83.3% | **37.5%** | 12.1% | **62.8%** | **1.13 m** |

One DAgger iteration **more than doubled mean route completion** (19.7%→37.5%) and produced the
project's **first genuine closed-loop success** (`waymo:0`, 95.8% route completion, visually confirmed
in `outputs/rollout_dagger_waymo_0.gif` — a real multi-agent intersection scene, ego reaches the
destination). Route deviation also dropped monotonically across all three stages (2.06→1.53→1.13 m).
Collision rate rose with DAgger (0%→16.7%) — expected: the policy now makes enough progress to reach
traffic conflict points it never survived to before. Rollout GIFs for both checkpoints across both
datasets saved as `outputs/rollout_{bc,dagger}_{nuscenes,waymo}_{idx}.gif`.

The improvement is **uneven across datasets**: substantial on Waymo (30.2%→62.8% RC, 0%→33% success)
but modest on nuScenes (9.2%→12.1% RC, still 0% success, still 100% off-road) — despite the DAgger
data being collected exclusively from nuScenes training scenarios. With only 3 eval scenarios per
dataset this asymmetry shouldn't be over-read as a clean second cause; it may simply reflect that
these particular nuScenes eval intersections are harder, as much as any residual capacity limit.

### Step 6 — verdict

**DAgger produced a real, non-trivial closed-loop improvement from one iteration — covariate shift
was a genuine, binding constraint, not just data/capacity, and the canonical closed-loop-imitation
fix (DAgger) partially closed it.** This is supported by two independent pieces of *measured* (not
assumed) evidence: the lateral-drift plot shows compounding steering error converging on the road
boundary in every rollout, and the action-gap test shows the expert/policy steering disagreement
growing over the same rollouts on training scenarios. The DAgger delta (>2x route completion, first
successes) confirms the diagnosis by partially fixing it. The residual gap — especially nuScenes
still failing 100% of the time — is consistent with this being a *genuine but only partially
addressed* covariate-shift problem at this data scale (859+543 transitions, ~5-6 usable training
scenarios from one source dataset): one DAgger iteration helps but does not fully close the gap,
which is exactly what the literature predicts for a single iteration on a small base dataset. Phase 5
should lead with this **measured covariate-shift story and the BC→BC+DAgger progression**, not the
Phase 4.5 underfitting framing, which the sharper Phase 4.6 signals supersede.
