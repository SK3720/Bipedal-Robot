# Autonomous walking work log

Goal: **sustained genuine alternating-foot bipedal walking** — real foot lift-off,
real support transitions. NOT reward-farming (no hop / bound / shuffle / drag /
bilateral flight / standing).

Preserved & untouched: `robot/robot.xml`, `runs/recovery_s1/`, `runs/walk_w11/`,
`runs/walk_s5/` (the 230 N recovery deliverable), `runs/walk_s3/`.

## Prior attempts (this session, before autonomous block)

| # | approach | outcome | why it failed |
|---|---|---|---|
| loco pre-1 | pure RL residual + attitude-LQR, big feet, loose reward | learns forward motion, 0 falls | it's a **bound** — both feet push off a big sole, brief flight, land |
| loco pre-2 | + strict no-flight/step-len reward, normal→1.3× feet, cold start | **stands, won't step forward** (safe local optimum) | first step risks a fall; standing is safe |
| loco pre-3 | + CPG leg-cycling scaffold (any foot 1.0–1.5×) | **topples in single support in 1–2 s** | open-loop leg motion destabilises; attitude-LQR-only can't catch it |
| loco pre-4 | + frontal CoP feedback stabiliser | still topples | stabiliser not fast/strong enough for dynamic leg motion |
| loco_x2 | strict reward + reference-state-init, 3× hands unlocked, 1.3× feet, 22M planned | **degraded** (falls 24→50 %), det. eval 7 % of target speed | RL from standing start won't commit to stepping; RSI just adds fall episodes |

**Key lesson:** RL from a *standing* cold start converges to "don't step". The
recovery policy `runs/walk_s5` already learned to balance THROUGH a single-support
step (unload → swing → plant → catch → stance). Walking = that, chained + driven
forward.

## Autonomous plan

- **A (primary):** warm-start from `walk_s5`. Add a `variant="walk"` to
  biped_walk_env: continuous alternating stepping on a cadence (no push, no
  settle), forward-drive reward + strict gait terms, fixed-length episodes,
  terminate on fall. Diagnose s5+continuous-stepping first.
- **B:** 1.5× feet + strict locomotion env (only size where RL found forward
  motion); the strict flight-termination now blocks the pure bound.
- **C:** hand-craft a *proper* walking CPG (foot placement ahead of CoM +
  push-off, not marching) + the frontal stabiliser, then RL residual.

---

## Run journal

### 2026-09-07 — HIL prep for `runs/sim2real_v1` (no policy change)

Documented the exact obs/action/controller pipeline and built a hardware-
interface / HIL package (`hil/`), all validated bit-for-bit against
`biped_sim2real_env`.  **No repo hardware code exists** — the whole hardware
surface is 4 stub methods in `hil/robot_interface.py::HardwareRobot`
(IMU / encoders / contacts / servos).

- `hil/spec.py` — single source of truth: 217-D obs layout (4×50 sensor frames
  oldest-first + 17 extra), joint orders (MuJoCo + POLICY_JOINT_ORDER =
  `ACT_CTRL`), rates (200 Hz control, gait phase `2π·0.80·dt`/step), frames
  (chest body: **+Y = up**, walk dir world −Y; gyro obs = `R_chestᵀ·ω_body`;
  accel obs = raw body-frame specific force ~[0,9.81,0] at rest), norm from
  `vecnormalize_best.pkl`, `ACT_SCALE`, joint limits, `spec.SAFE`, train noise σ.
- `hil/kinematics.py` FK + leg-odometry `v_est` (encoders only) — exact vs sim.
- `hil/observation.py` sensor→217-D obs (+history+phase).
- `hil/base_controller.py` IMU-only `_compose_ctrl` → 15-D pos cmd — exact vs sim.
- `hil/policy.py` pure-numpy MLP + norm (no torch at runtime), matches SB3 7e-7,
  inference ~0.08 ms.
- `hil/safety.py` NaN/range/limit/tilt/rate checks.
- `hil/robot_interface.py` SimRobot + HardwareRobot(stubs);
  `hil/run_hil.py` **log-only** loop (motors off by default) → `hil/logs/*.npz+txt`
  (obs, actions, cmd, per-stage latency, rate, safety events);
  `hil/validate_stack.py` → **PASS** (FK/base 1e-16, obs 1e-3, policy 7e-7).
- Consts: `runs/sim2real_v1/base_controller_consts.npz` (regen `python -m hil.export_consts`).
- Dev timing: full loop ~0.6 ms, sensor→action ~0.3 ms (~8× headroom @200 Hz).
- Only env edit: `V_EST_NOISE = 0.03` made a named constant (was inline) so
  validation can zero it — zero behaviour change, policy untouched.

### 2026-09-06 — `runs/sim2real_v1`: IMU + encoder observation (real sensing)

User: loco_w3 is THE sim-to-real candidate; **don't modify it** — work on a copy
(`runs/loco_w3_sim2real/`). Real sensing = chest IMU + joint encoders (+ maybe
foot contact switches). Plan: bench HIL sensor/inference validation soon, so the
policy obs must map 1:1 to real driver outputs.

`biped_sim2real_env.py` (subclass of BipedLocomotionEnv):
- **obs rebuilt**: chest gravity vec + heading (IMU), body-frame gyro, body-frame
  specific force (accelerometer, finite-diff + gravity), 14 joint pos + vel
  (encoders), 2 foot-contact booleans — each noised to real-sensor 1σ, **stacked
  over the last 4 frames** (→ velocity inferable). + current clock/speed/prev-act.
  obs = 181-d. DROPPED (privileged): CoM velocity, absolute height, world-frame
  foot positions, contact-force magnitudes.
- **base controller reduced to IMU+encoder**: CPG kept; attitude LQR with the
  HEIGHT terms masked out (orientation + rates only); the CoM/CoP frontal
  stabiliser replaced by a chest-roll + roll-rate PD (ankle-roll + loaded-leg
  hip-roll, contact-switch gated).
- moderate DR fixed at `robust=0.15` (friction/mass/gain/latency/pushes via the
  parent machinery) + IMU bias / encoder noise here. Phase-2 resume ramps DR.
- Zero-action base survives ~0.9 s (similar foothold to w3's pre-training base).

`train_sim2real_ppo.py`: FRESH PPO (obs + base both changed → no warm-start),
zero-init head, speed curriculum 0.12→0.30.

**v1a (killed @3 M):** partial observability BLOCKED learning — survive stuck
~16 %, 100 % falls, ~0 genuine steps, best score negative. Missing the balance-
critical signals (CoM velocity, CoM-vs-stance-foot) that w3's obs had.
**v1b fix:** added deployable versions of those to the obs —
  * **leg odometry** base-velocity estimate (`v_est`: −d/dt of the stance foot's
    base-frame position, blended over contacting feet — standard proprioceptive
    odometry), 3-d;
  * **FK foot positions in the base frame** (from encoders), 6-d.
obs now 217-d.
**v1b ALSO failed** (15 % survive @3.25 M). Isolated it with `privileged_obs` /
`imu_ctrl` diagnostic flags: **it's the BASE CONTROLLER, not the obs.** w3's own
policy in the s2r env:
  * privileged obs + w3 controller → 100 % survive (sanity ✓)
  * privileged obs + my crude roll-PD IMU controller → **22 %** (the blocker)
**v1c fix:** rebuilt w3's exact CoP law from deployable inputs — `err = −(mean
base-frame lateral offset of the contacting feet)` [FK], `com_vx → v_est`
[leg odometry], same `FR_*` gains. w3's policy + this + privileged obs → **100 %
survive, 2.28 m, 20 steps** (identical to native w3). Masked LQR (no height) is
fine.
**v1c (fresh train, killed @3.5 M): STILL 19 % survive.** So the realistic obs is
*also* a real obstacle for cold-start RL — the fresh policy can't find the gait
through the noisy/partial/history signal even with a good base controller.

**→ TEACHER-STUDENT DISTILLATION (`distill_s2r.py`).** Run w3 as teacher
(privileged obs + rebuilt IMU controller), record (realistic_obs → w3_action),
behaviour-clone a student on the realistic obs.  Env got `_realistic_obs()` (the
217-d IMU/encoder obs, always) + `privileged_obs`/`imu_ctrl` diagnostic flags.
40-rollout / 20-epoch pilot: teacher 98 % survive, BC MSE 0.12→0.009,
**student on realistic obs → 83 % survive, 1.84 m, 20 genuine steps.**  Massive
jump from cold-RL's 19 %.

**Full distillation** (450 rollouts, 80 epochs, BC MSE → 0.0040):
`runs/sim2real_v1/policy_distilled.pth`. Broader eval (realistic obs, 16 seeds):
robust 0.0 → 80 %, robust 0.2 → 93 % survive (small-sample noisy; BC has some
action-compounding error). dist ~1.8-2.0 m, ~20 genuine steps, chatter ~0.10.

**RL fine-tune FAILED** — collapsed the distilled walk to ~14 % survival despite
gentle settings (lr 1.2e-4, ent 0.002, log_std −2.4). The known repo failure:
fresh value fn + advantage noise knocks a working policy out of its basin. ep_rew
7 k → 600. Restored `policy_distilled.pth`.

**→ PURE SUPERVISED with DAgger** (`distill_s2r.py` rewritten). No RL at all:
init BC on teacher rollouts, then 3 DAgger rounds — the STUDENT drives (β 0.5→1.0),
teacher labels every visited realistic_obs, aggregate + retrain. ~1/3 of every
rollout under `robust=0.15` so the student is dynamics-robust without RL.

**SUCCESS.** init 350 + 3×160 rollouts, 1.12 M samples, 60 epochs/round:
| stage | survive @rob 0 | @rob 0.25 | dist |
|-------|-----|-----|-----|
| init BC | 88 % | 84 % | 2.02 |
| DAgger 1 | 99 % | 91 % | 2.27 |
| DAgger 2 | 94 % | 86 % | 2.10 |
| DAgger 3 (β=1) | 96 % | 91 % | 2.19 |
Independent 30-seed eval of the final policy (`runs/sim2real_v1/policy_best.pth`
== `policy_dagger3.pth`): **robust 0.0 → 97 % survive, 2/30 fell, 2.21 m/ep, 25
genuine steps, 4 % flight, 0 dives, chatter 0.10**; robust 0.15 → 84 %.
Clean upright gait, indistinguishable from w3's — but on **IMU + joint-encoder
data only** (obs 217-d, no privileged state), and **no reinforcement learning**
anywhere in the pipeline (RL fine-tune collapsed it; RL cold-start never
converged). Sent `walk_sim2real.mp4`.
Preserved: `policy_distilled.pth` (v1), `policy_dagger3.pth`,
`policy_at_dagger3_97pct.pth`, matching vecnormalize. `runs/loco_w3/` untouched.

### 2026-09-04 — `runs/robust_r1`: SIM-TO-REAL hardening of loco_w3

loco_w3 designated the sim-to-real candidate (`runs/loco_w3/SIM2REAL_CANDIDATE.md`).
`biped_locomotion_env.py` gains `robust` (0 = w3 setup unchanged; curriculum 0→1):
- per-episode DR: friction ±35 %, dof-damping ±40–60 %, servo-gain (kp) ±18 %,
  link mass ±12 % + random torso payload (0–25 % of chest mass).
- control channel: 0–3 control-step action latency, gaussian obs noise (σ≤0.02).
- random torso pushes: 8–22 N for ~25 ms, every 1.5–4 s.
- reward: contact-chatter penalty (>2 make/break events/step), jerk + effort
  penalties — targets the w3 gait's foot buzz (~280 contact events/ep).
- BUG fixed: `mj_setConst` after a mass change corrupts the servo/spring
  reference → instant fall; dropped it (mj_forward propagates mass fine).
`train_robust_ppo.py`: warm from w3, robust curriculum [0.3, 0.5, 0.7, 1.0],
advance at survive ≥ 88 % & fell ≤ 15 % & dived ≤ 5 %. Speed fixed 0.30. 16 M.
Zero-shot w3 under robust: 0.3 → 100 % survive, 0.6 → 87 %, 1.0 → 44 %.

**p1 (0→2.3 M):** ripped through robust 0.3→0.5→0.7 (w3 already handled the easy
levels). Held eval survival ~90 % at robust 0.7 but **plateaued** — advance gate
`fell ≤ 0.15` too strict under heavy DR, and the **contact-chatter penalty never
fired** (its `−2 events/step` threshold vs a gait that averages 0.2/step).
**p2 fixes:** chatter counter now uses hysteresis (enter contact nf>10, exit
nf<2) → w3's true chatter ≈ 0.10 transitions/step (~138/ep vs ~44 clean);
penalty is now `−(2.5+4·robust)·max(0, rate − 0.06)` (persistent) + a burst term;
advance gate relaxed to `fell ≤ 0.30`. Resumed from 2.3 M. ep_rew dropped
3.3k→2.6k (chatter penalty biting), chatter 0.13 and falling. Running.

## Run journal

### 2026-09-03 (cont.) — `runs/bigstep_b1`: BIGGER STEPS

Research (chat has sources — Duan 2022 footstep-target, Li 2021 Cassie "change
stride not cadence", PLOS One 2025 step-length/frequency model): w3's gait
converges to short strides (~0.14 m) / high cadence because nothing rewards
stride length and the step bonus caps at 0.11 m.

`biped_locomotion_env.py` gains `stride_mode` (default OFF → w3 setup unchanged):
- `GAIT_HZ 0.80→0.60`; `_leg_ref` swing reach/lift scaled by `stride_tgt/0.14`.
- step reward PEAKS on `stride_tgt` (`4.5·exp(−((L−tgt)/0.055)²)`) not a fixed cap.
- **cadence cap**: genuine step < 0.5·(stride_tgt/speed_tgt) after the last → penalty.
- **soft-landing** term: reward low swing-foot horizontal speed at touchdown.
- `stride_tgt` appended to obs (68→69). Speed fixed 0.24 m/s while stride grows.

`train_bigstep_ppo.py`: warm-start from `runs/loco_w3` (first MLP layer padded
68→69, VecNormalize seeded + padded). Stride curriculum [0.17, 0.21, 0.25, 0.29],
advance when `mean_stride ≥ 0.85·tgt & survive ≥ 70 % & dived ≤ 10 %`.
Speed fixed 0.28 m/s. 16 M steps, 5 envs alongside w3's 8.

**v1 (killed @1 M):** the stride metric was corrupted by contact chatter — a naive
lift/land test counted the foot's rapid make/break contact (w3: ~130 contact
events/ep, most within one stance) as steps, so `_lift_fwd` reset constantly and
`step_len` read ~0.05 m. Fix: a debounced per-foot swing state machine (6 ms
genuinely-airborne → "swing", 4 ms solid contact → "stance"; stride = fwd
displacement between consecutive genuine landings). Verified: w3's true gait is
**~0.16 m/step, ~2.7 steps/s** — a moderate walk, not the shuffle the broken
metric implied. STRIDE_REF set to 0.16, curriculum starts just above.
**v2 (killed @1 M):** stride SHRANK — a *per-step* bonus rewards *more, smaller*
steps. **v3 (killed @1 M):** two-gaussian (length × interval) bonus + cadence
penalty — STILL shrank; the bonus terms are small perturbations on the dominant
`+6.5` progress term, and small steps are more stable.
**v4 fix — make stride a first-class objective:** the forward-progress term is
now GATED on `stride_ema` being near target: `×max(0.35, exp(−((ema−tgt)/0.07)²))`.
A fast tiny-step shuffle at target speed earns only ~35 % of the progress reward.
`stride_ema` + `stride_tgt` both in obs (→70 dims). Verified: warm-policy return
drops 9.5k→5.1k as stride_tgt goes 0.17→0.27 while it keeps short steps — a
~46 % hit, a strong gradient. ent 0.006→0.010. w3 stopped (converged) → bigstep
now has all 8 envs. 18 M steps.

**v4 IS WORKING.** | step | stride_tgt | mean_stride | cadence | survive |
|---|---|---|---|---|
| 0.25M | 0.17 | 0.118 | 2.7 | 94% |
| 1.75M | 0.17 | **0.164** | **2.3** | 94% | → curric L1 (0.21) |
| 2.25M (train) | 0.21 | ~0.19 | 2.2 | ~88% |
Stride grew 0.12→0.19, cadence dropped 2.7→2.2 (w3 was 0.16 / 2.7). Fewer,
bigger steps.
- @2.25–3.0 M: **best window** — stride 0.17–0.18 m, cadence 1.9–2.3/s,
  survive 93–99 % at the 0.21 m curriculum level. Genuinely bigger + slower
  than w3.
- @3.25 M onward: **diverged** — deterministic eval collapsed (stride →0.10,
  survive →48 %, fell →94 %, flight →0.13, some dives) while training kept
  chasing stride ~0.20. The 0.17→0.21 curriculum jump (+24 %) + lr 2e-4 +
  ent 0.010 walked it off the good solution into a "big step or fall" policy.
Killed @4.75 M; preserved `policy_1p75M_s2p965.pth` (stride 0.16, 94 % survive).

**b2 (killed @2.4 M):** gentler curriculum held survival at 100 % but the
DETERMINISTIC eval stayed conservative — stride 0.13–0.15 m while training did
0.18–0.19. The exp(-²) progress gate with floor 0.35 / width 0.07 wasn't sharp
enough to force the deterministic policy to commit to the long step it can
already take under exploration.
**b3:** progress gate is now an **asymmetric ramp** — `clip((ema − 0.65·tgt) /
(0.35·tgt), 0.15, 1)`: ~0.15× credit at 0.65× target, full at target, flat above
(no penalty for a longer step). Verified: warm-policy return 6.6k→4.9k as
stride_tgt 0.17→0.21 while it keeps 0.13 m — a >55 % loss, unambiguous gradient.
Curriculum [0.17, 0.19, 0.21], advance frac 0.90. Fresh warm from w3.
lr 1.2e-4, ent 0.006.
- @1.0 M: **best window** — stride 0.187 m, cadence 2.1/s, 100 % survive, 0 falls.
- @1.5 M onward (curriculum → 0.19): deterministic eval oscillates 0.12–0.16 m,
  survive 62–97 %, fell 6–81 % — never consolidates a stable ≥0.19 m gait.
Killed @5 M.

**BIGGER-STEPS CONCLUSION.** On the canonical foot (~0.088 m fore-aft) + ±2.3 N·m
there is a hard **stable-stride ceiling around 0.16–0.18 m** — every variant
(b1/b2/b3, ~10 reward iterations) converges there because a longer step lands
the swing foot further ahead of the CoM → a bigger "catch" torque → the same
sagittal torque wall the whole project has fought (and the PLOS "control-hard
regime": energy-sensitivity of long/slow steps).
Best stable checkpoint: **`runs/bigstep_b1/policy_1p75M_s2p965.pth`** — 30-ep
eval @ stride_tgt 0.17: **stride 0.174 m · cadence 2.4 steps/s · 100 % survive ·
0 falls** vs w3's 0.16 m / 2.7 steps/s.  A modest, real gain: ~9 % longer stride,
~11 % slower / more deliberate cadence, still rock stable — but not dramatic.
For dramatically bigger steps the robot needs a longer foot (the long-foot w2
morphology had the fore-aft margin).

### 2026-09-03 (cont.) — `runs/loco_w3`: NORMAL feet, heavier forward emphasis

User ask: retry with regular-size feet, 20 M+ steps, heavier forward emphasis,
watch for the "lunge forward and die" loophole.

Changes vs w2:
- MODEL `robot/_exp_hands_3x.xml` — **canonical foot** (sole ~0.088 × 0.066 m).
  The small support polygon the whole project has fought.
- Reward: alive `2.6 → 2.0`; forward-progress term `4.4 → 6.5` and now **gated on
  being upright** (`×clip(1.15 − tilt/20, 0, 1)` → ~0 by 23° tilt) so a tilted
  lunge earns almost nothing; small bonus to 1.15× target; overspeed wall moved
  `1.35× → 1.18×` and strengthened `-1.6 → -2.2`.
- **DIVE guard:** `vfwd > 1.15×tgt AND fwd_lean > 24°` ends the episode with the
  −50 fall penalty (info `dived`, in eval print + score −1.5·dived).
- `WALK_LEAN 0.09 → 0.04`.  24 M steps, curriculum 0.12→0.30, cold start, ent 0.008.
- Open question: normal feet may not be walkable at ±2.3 N·m (torque/pitch wall);
  the long-foot w2 result stands regardless.

**w3 v1 (killed @2.3 M) — the lunge farm the user warned about appeared.**
Eval vfwd 132-149 % of target, `fell` 100 %, survive ~20 %, dist 0.2-0.3 m,
training vfwd ~0.37. The soft overspeed wall (1.18×, −2.2) + a 1.15× progress
bonus let it barrel forward and eat the fall. Curriculum even advanced on the
overspeed.  **v2 fixes:**
- progress term HARD-clipped at 1.0× target (zero credit for overspeed),
- steep overspeed penalty `−7.0·max(0, v − 1.06×tgt)`,
- fall penalty scales with time left: `−50 − 45·(1 − t/EP)` (−95 at t=0),
- dive guard tightened (`v>1.10×tgt & lean>20°`),
- curriculum advance now gated on `survive ≥ 65 % & dived ≤ 10 %` and measured
  against speed **capped at target**.
Ordering now: lunge-die (~950) ≪ stand (~2800) ≪ walk (~12000). Relaunched 24 M.

**w3 v2 progress** (user: run to ≥10 M):
| step | tgt | survive | fell | vfwd | dist | genuine | flight | dived |
|------|-----|---------|------|------|------|---------|--------|-------|
| 2.5M | 0.12 | 24% | 100% | 0.12 | 0.18 | 0.8 | 0.07 | 6% |
| 4.25M| 0.12 | 66% | 88%  | 0.13 | 0.59 | 1.0 | 0.06 | 0% |
| 4.75M| 0.12 | 69% | 69%  | 0.15 | 0.77 | 1.6 | 0.05 | 0% | → curric L1 (0.18) |
| 5.0M | 0.18 | 74% | 56%  | 0.16 | 0.85 | 1.8 | 0.04 | 0% |
| 5.75M| 0.18 | 40% | 94% | 0.21 | 0.59 | 0.9 | 0.07 | 0% | rough patch |
| 6.0-6.25M | 0.18 | 64-81% | — | — | — | — | 0.29-0.31 | 0% | brief bound explore, strict reward killed it |
| 6.75M| 0.18 | 99% | 6%  | 0.21 | 1.43 | 1.1 | 0.00 | 0% | recovered |
| 7.0M | 0.24 | 100%| 0%  | 0.26 | 1.80 | 3.1 | 0.01 | 0% | |
| 7.5M | 0.24 | 100%| 0%  | 0.26 | 1.84 | 4.1 | 0.00 | 0% | → curric L3 (0.30) |
| 7.75M| 0.30 | 100%| 0%  | 0.31 | 2.17 | 9.1 | 0.00 | 0% | best 3.823 |
| 8.0M | 0.30 | 100%| 0%  | 0.32 | 2.25 | 11.8| 0.00 | 0% | best 4.143 (> w2's 3.601) |

**NORMAL FEET WALK — SUCCESS.** Independent 30-ep eval of the 7.75 M checkpoint
@0.30 m/s: 100 % survival, 0 % falls, 0.31 m/s (104 %), 8.9 genuine steps/ep,
2.2 m/ep, 0.3 % flight, 0 % dives. Clean upright gait, canonical foot geometry,
±2.3 N·m. Sent `walk_7p75M.mp4`. Anti-lunge fences (upright-gated progress, hard
speed cap, steep overspeed penalty, time-scaled fall penalty, dive guard) all
reading zero — no farming. Preserved `policy_at_7p75M_s3p823.pth`. Continuing to 24 M.

**w3 @ 10 M:** eval @0.30 m/s (40 ep) — 98 % survival, 2 % falls, 0.33 m/s (110 %),
**17.6 genuine steps/ep** (~2.5 Hz cadence), 2.28 m/ep, 0.7 % flight, 0 % dives.
Best score 4.413. One noisy dip @9.75 M (survive 79 %) recovered next eval.
Preserved `policy_at_10M_s4p413.pth`. Sent `walk_10M.mp4`. Filmstrip: clean
upright stride, consistent, no lean. Still improving; running to 24 M.

### 2026-09-02/03 — autonomous block

**Approach A (recovery-stack continuous stepping) — ABANDONED.**
Built `biped_walk_cont_env.py`: subclass of the recovery env (cp variant), phase
machine `shift → swing → descend → hold → next step`, warm-startable from
`walk_s5`. Added: a lateral weight-shift phase, LIPM lateral foot placement,
true sole-vertex clearance measurement (anti-scrape), scripted knee-lift on top
of the whip, sagittal pitch feedback.
- Validated in isolation: the `shift` phase keeps side-lean small; flipped
  `_HR_SIGN` = {L:+1,R:-1} is correct; lateral placement + stance hip-roll hold
  help.
- BUT: the whip is a *retract-to-recentre* primitive. Its descend knots pull the
  planted foot backward → net foot travel negative → 0 genuine steps + backward
  CoM drift. Every fix traded pitch↔drift↔lateral. Zero-action base: ~1 s
  survival, `genuine=0`, dist ≈ −0.1 m. This re-confirms the memory note
  "whip-retract step doesn't chain under the scaffold". Not a good RL prior.
  Files kept (`biped_walk_cont_env.py`, `train_wcont_ppo.py`) but not trained.

**Approach B (CPG locomotion env) — TRAINING.**
Reworked `biped_locomotion_env.py`:
- `CPG_GAIN 0 → 0.85` (rhythm is a strong prior; zero action ≈ CPG scaffold walk,
  not "stand"), `GAIT_HZ 0.95 → 0.80`, `DUTY 0.62 → 0.64`, `WALK_LEAN → 0.07`,
  `STEP_FWD → 0.13`, `ACT_SCALE` reduced (residual refines, not overrides).
- MODEL `robot/_exp_hands_3x_feet_1p5.xml` (NEW: 3× hands unlocked, **1.5× feet**
  per the user's 2026-09-02 note that 2× feet edge-contact during swing).
- TRUE sole-to-floor clearance (`_sole_clear`, foot collision-mesh vertices):
  the per-frame clearance reward and the completed-step bonus now require
  peak sole clearance > 14 mm + 16 ms true air + 22 mm forward travel, else the
  "step" is scored as a scrape (−2.0). A pivot / scuff of the big foot cannot be
  farmed as a step.
- RSI: 50 % of episodes start mid-stride near a double-support hand-off (not a
  random deep single-support pose — that caused launch/bound blow-ups), CoM
  already moving forward at ~target speed.
- Zero-action base (speed_tgt 0.16): mean survival ~350 control steps (1.75 s),
  best seed 6.4 s, ~2/18 blow-ups, marches ~in place (dist ≈ 0), genuine ≈ 0.5.
  A legitimate RL starting point with a clear reward gradient toward walking.
- Trainer `train_loco_ppo.py`: curriculum [0.12, 0.18, 0.24, 0.30], score now
  rewards genuine_steps and hard-penalises flight fraction.

**`runs/loco_w1` (archived `runs/_loco_w1_marched_dead/`) — killed at 1.0 M.**
Deterministic-eval forward speed *decreased* over training: 0.06 → 0.05 → 0.03 →
0.03 m/s. ep_rew rose (500→710) purely from marginally longer survival. survive
~30 %, still 100 % falls, genuine ≈ 0.5. Classic loco_x2 failure: with `+3.0`
alive dominant and only a `+2.4` *peaked* speed term, a surviving stand (+3/step)
beats a slow walk, so the policy learned to minimise motion. Filmstrip: wide
stance, topples, no stepping.

**`runs/loco_w2` — TRAINING (progress-dominant reward + long foot).**
Diagnosis of w1: the reward ordering was stand ≈ walk. Fixed:
- reward now: alive `+2.6`, **forward-progress `+4.4·clip(vfwd/tgt,0,1.05)`
  (largest term, ~linear not peaked)**, `-1.6` overspeed, `-0.8` backward,
  stability penalties cut ~40 % so they can't out-vote progress. Ordering is now
  fall(~800) ≪ surviving-stand(~3400) ≪ surviving-walk(~9800).
- MODEL `robot/_exp_hands_3x_longfoot.xml` (NEW): foot **fore-aft 0.21 m, width
  0.09 m** — long, not wide. Fore-aft length is exactly what the recovery work
  showed beats the sagittal pitch/torque wall; near-normal width avoids the
  swing-edge-strike the user flagged for 2× duck feet.
- `CPG_GAIN 0.85 → 0.75`, `ACT_SCALE` raised (policy must ADD forward drive the
  open-loop CPG lacks), `WALK_LEAN 0.07 → 0.09`, RSI forward vel → ~target,
  `ent 0.004 → 0.008`.
- Zero-action base: meanT ~223, marches (~0). 14 M steps, curriculum 0.12→0.30.

**`runs/loco_w2` @ 2.2 M — IMPROVING (first run that is).**
| step | survive | fell | eval vfwd | dist | genuine | flight |
|------|---------|------|-----------|------|---------|--------|
| 0.2M | 26% | 100% | 0.03 | 0.04 | 0.7 | 0.08 |
| 1.0M | 36% | 94% | 0.05 | 0.11 | 1.1 | 0.07 |
| 1.8M | 69% | 62% | 0.07 | 0.30 | 1.3 | 0.06 |
| 2.0M | 84% | 31% | 0.05 | 0.33 | 1.6 | 0.02 |
| 2.2M | 72% | 50% | 0.07 | 0.32 | 1.5 | 0.07 |
Filmstrip @2.0M: genuine leg lift + forward foot placement, upright for the
first ~half of the episode, then pitches forward and lunges/catches on the long
foot. Not a clean walk yet — "step, step, forward lunge" — but genuine lift-off
+ forward progress + seconds upright, flight low (no hop/bound). Concern: eval
vfwd stuck ~0.06 (≈50 % of the 0.12 target). Watching whether tilt/fall
penalties pull it upright by ~5 M, else targeted anti-pitch fix + resume.
Training-wheels torso-assist prototyped then shelved (xfrc sign/tuning
destabilised the scaffold; not worth the time while w2 climbs).

**w2 — GENUINE WALKING, improves through the whole curriculum.**
| step | tgt | survive | fell | vfwd | dist | genuine | flight | note |
|------|-----|---------|------|------|------|---------|--------|------|
| 3.4M | 0.18 | 80% | 31% | 0.13 | 0.70 | 3.2 | 0.06 | first clear walk; stiff, forward lean |
| 4.2M | 0.24 | 99% | 6%  | 0.15 | 1.02 | 5.3 | 0.01 | best-score (2.446) for a while |
| 5.4M | 0.24 | 93% | 19% | 0.18 | 1.16 | 6.3 | 0.01 | |
| 6.2M | 0.24 | 93% | 12% | 0.19 | 1.19 | 5.8 | 0.04 | → curriculum L3 (0.30) |
| 6.8M | 0.30 | 97% | 6%  | 0.20 | 1.38 | 7.7 | 0.01 | best-score 2.771; torso notably more vertical |
| 7.4M | 0.30 | 98% | 12% | 0.24 | 1.69 | 11.1 | 0.02 | clean upright walk |
| 9.4M | 0.30 | 99% | 6%  | 0.29 | 2.04 | 11.6 | 0.01 | full target speed |
| 9.6M | 0.30 | 100%| 6%  | 0.30 | 2.12 | 13.1 | 0.01 | **best score 3.601 — the deliverable** |
| 10.8M| 0.30 | 96% | 6%  | 0.33 | 2.23 | 15.6 | 0.01 | slight overspeed |
| 12.2M| 0.30 | 81% | 38% | 0.39 | 2.22 | 14.6 | 0.14 | **over-optimised speed → degrading**; killed here |

**Outcome: SUCCESS — genuine sustained bipedal walking.** Best checkpoint
`runs/loco_w2/policy_best.pth` == `policy_at_9p6M_s3p601.pth` (@9.6 M steps).
Fresh independent 40-episode eval @ target 0.30 m/s (`eval_loco_ckpt.py`):
**97 % survival · 8 % falls · 0.31 m/s (103 % of target) · 12.8 genuine
alternating steps/episode · 2.11 m/episode · 1.7 % bilateral flight.** Upright
torso, real foot lift-off (true sole-mesh-vertex clearance gated), real support
transitions — no hop / bound / shuffle / drag / stand. Clean full-episode clip:
`runs/loco_w2/walk_final_9p6M.mp4` (7 s, no fall, 2.45 m).

Past ~10.8 M the policy started trading survival for speed (drifting to 0.39 m/s,
falls 6 %→38 %, flight →0.14) because the progress term saturates at 1.05× target
but the overspeed penalty only bites at 1.35×; killed the run at 12.3 M since
`policy_best` was frozen at the clean 9.6 M version. If resumed, tighten the
overspeed penalty (bite at ~1.1×) or lower the progress cap.

Preserved checkpoints (all + matching vecnormalize): `policy_at_4p2M_s2p446`,
`policy_at_6p8M_s2p771`, `policy_at_7p4M_s3p189`, `policy_at_9p4M_s3p544`,
`policy_at_9p6M_s3p601`, `snap_10p9M`. Milestone videos sent: `eval_3400000.mp4`,
`eval_6800000.mp4`; final clip from `eval_loco_ckpt.py`.

## Run journal


### Sep 20 2026 — Raspberry Pi + MPU-6050 HIL (additive; nothing existing changed)

User bench facts: Pi 4B (host `shiv`, sk@10.0.0.71), MPU-6050 on I2C bus 1 @ 0x68 works at ±2 g/±250 °/s; PCA9685 +
TD-7120MG servos not used yet (hobby servos have no position feedback; no foot sensors listed → joint/contact obs
still have no hardware source).  Added in `hil/`: `mpu6050.py` (driver + `FakeBus`), `imu_fusion.py` (calibration +
Mahony AHRS, heading matched to sim standing pose), `calibrate_mpu.py` (stationary / six-point / mount / live),
`hybrid_robot.py` (real IMU + sim joints, torso pinned, log-only), `run_hil.py --mpu` (additive flag),
`selftest_mpu.py`, `make_pi_bundle.py` (+ `requirements-pi.txt`), README §9.  Pi path is torch-free
(`runs/sim2real_v1/hil_policy.npz`, made by `hil/export_pi_bundle.py`; numpy-vs-.pth action diff 0.0).
Non-impact proof: `--sim` log md5 identical before/after; `validate_stack` numbers unchanged; only `run_hil.py`
edited (new flags), no policy/checkpoint/robot.xml touched.
Findings: (1) walking direction is **+Z_chest** (sim robot travels +1.22 m along it); the env's `_fwd_local = -Z`
is the backward axis, `fwd_xy` is only a yaw reference — chest frame = +X left, +Y up, +Z forward.  I briefly
"corrected" this the wrong way mid-session and reverted after measuring.  (2) sim IMU spikes to ~8 g / ~750 °/s at
foot strikes (finite-diff accel + rigid impacts): ±2 g/±250 °/s clips in walking and clipped gyro spikes integrate
into yaw error → use ≥ ±8 g / ±1000 °/s for dynamic tests; the driver counts saturated samples.
(3) MPU-6050 has no magnetometer → yaw drifts; bench Z ≈ 1.07–1.11 g means accel offset/scale calibration is needed.
Self-test (fake bus fed by the sim): six-point + mount recovered to 1e-4, gyro 0.002 rad/s, tilt p95 2.6°, yaw
integration exact, signs verified; `run_hil --mpu` runs end-to-end, ~0.6 ms sensor→action.

### Sep 20 2026 (later) — Pi IMU stage verified by the user; servo bring-up scaffolding added
User ran selftest / calibration / live / `run_hil --mpu --seconds 30 --accel-g 8 --gyro-dps 1000` on the Pi (via
SSH from the PC PowerShell): "all looks good" (sensor→observation→inference stage on real MPU-6050 complete).
Added `hil/pca9685.py`, `hil/servo_test.py`, `hil/selftest_servo.py` (16 checks PASS on a fake bus): dry-run by
default, no assumed servo parameters (null map template), slew + pulse clamps, release on exit.  Not run on hardware.

### Sep 20 2026 (evening) — measured: sim2real_v1 vs real-actuator behaviour (`hil/SERVO_ROBUSTNESS.md`)
New `hil/eval_servo_robust.py` (eval only; nothing existing modified, `--sim` log md5 unchanged).  Findings on the
DEPLOYMENT stack with IMU noise MEASURED from the user's Pi log (gyro 0.0007 rad/s, accel 0.016 m/s², bias 0.0005 —
30×/22×/60× below training noise): ideal actuators **62 % survive @200 Hz base controller vs 92 % @1 kHz env-like base**
(env's 97 % used a 1 kHz controller on ground-truth state); 200 Hz stack chatters (hip-pitch cmd rate p95 ≈ 180 rad/s vs 30).
Actuator cliffs (200 Hz stack): lag 5 ms → 38 %, 10 ms → 4 %; delay 5 ms → 4 %; slew 50 rad/s → 8 %.  1 kHz base:
lag 20 ms → 67 %, slew 25 rad/s → 71 %, 12 rad/s → 0 %; lag40+6 rad/s+delay10 → 0 %.  Encoder-free (cmd or smoothed cmd
as joint feedback) → 0–4 % even with ideal actuators.  Policy saturates the 2.3 N·m limit (p95) on every joint.
=> not deployable on rate-limited / laggy / feedback-less hobby servos; next decision = hardware with feedback vs a NEW
policy trained against the deployment pipeline (base ctrl at deploy rate, measured noise, servo model).
Pi finding: user's 30 s `--mpu` run achieved 162.6 Hz (sensor read 2.78 ms, likely I2C @100 kHz) — policy clock skews.

### Sep 20 2026 (late) — live link: real MPU-6050 on the Pi driving the sim on the PC
New `hil/imu_stream.py` (Pi, UDP sender), `hil/live_sim.py` (PC, receiver + torso puppet + MuJoCo viewer),
`hil/selftest_live.py` (loopback: 0 drops, 200 Hz, ~2 ms age, 17.3° vs 17.2° sway, torso quaternion == real quaternion,
policy command changes 0.6 rad with tilt).  Additive only; `--sim` log md5 unchanged.  Viewer not exercisable in the tool
sandbox (no OpenGL) — existing repo scripts use the same launch_passive call.  Pi bench found earlier: I2C still @100 kHz
(config.txt has no baudrate) -> 400 kHz edit proposed to the user.
Follow-up: the user's PC GPU (Radeon RX 480) reports Status: Error -> no OpenGL -> MuJoCo viewer cannot open (confirmed on
the user's own run; the Pi->PC UDP link itself worked: "receiving IMU packets").  Added `hil/stick_viz.py` (tkinter, no
OpenGL) and made it the `live_sim` default (`--viz stick`).  Verified with the fake stream: 200 Hz, 0 drops, snapshot checked.
