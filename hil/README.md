# HIL / hardware-transfer package for `runs/sim2real_v1`

Everything needed to run the `sim2real_v1` walking policy against real sensors,
**log-only, motors disabled**, and to measure the sensor -> observation ->
inference pipeline.  The trained policy and every checkpoint are untouched.

```
hil/
  spec.py              single source of truth: layouts, rates, frames, limits, noise
  export_consts.py     regenerate the base-controller constants npz from the env
  kinematics.py        FK (foot-in-base-frame) + leg-odometry velocity, encoders-only
  observation.py       build the 217-D observation from sensor readings
  base_controller.py   the IMU-only base controller -> 15-D position command
  policy.py            load + normalise + run the policy (pure numpy, no torch/SB3)
  safety.py            sensor / attitude / command / timing checks
  robot_interface.py   SimRobot (dry-run) + HardwareRobot (driver STUBS)
  run_hil.py           the log-only loop
  validate_stack.py    proves hil/* == biped_sim2real_env  (run this first)
```

## TL;DR for tomorrow

```bash
# 1. prove the software side (no hardware) -- must print "PASS"
python -m hil.validate_stack

# 2. dry-run the whole loop against the sim
python -m hil.run_hil --sim --seconds 8
python -m hil.run_hil --sim --noise --seconds 8      # + training sensor noise

# 3. wire up hil/robot_interface.py::HardwareRobot  (4 methods, all marked >>> WIRE UP <<<)

# 4. real robot, LOG-ONLY (motors never driven):
python -m hil.run_hil --hardware --seconds 20
#    -> hil/logs/hil_<ts>.npz  + .txt summary  (obs, actions, commands, latencies, safety)
```

`--enable-motors` exists but is **not** part of the sensor/inference validation
you asked for; do not use it until the bench checks below pass.

---

## What already exists vs. what you must wire up

| piece | status |
|---|---|
| MuJoCo model, kinematics, StandingLQR | ✅ in repo (`robot/_exp_hands_3x.xml`, `standing_balance_lqr.py`) |
| policy + obs-normalisation stats | ✅ `runs/sim2real_v1/policy_best.pth`, `vecnormalize_best.pkl` |
| base-controller constants (K, qpos0, …) | ✅ `runs/sim2real_v1/base_controller_consts.npz` (regen: `python -m hil.export_consts`) |
| observation builder, FK, leg-odometry, policy runner, base controller, safety | ✅ this package (validated bit-for-bit vs the sim) |
| **IMU driver** (quaternion + gyro + accel) | ❌ `HardwareRobot._read_imu` — `>>> WIRE UP <<<` |
| **joint-encoder driver** (pos + vel, 14 joints) | ❌ `HardwareRobot._read_encoders` |
| **foot-contact driver** (2 booleans) | ❌ `HardwareRobot._read_contacts` |
| **servo driver** (15 position commands) | ❌ `HardwareRobot._write_positions` (only used with `--enable-motors`) |

There is **no** hardware/serial/CAN/I2C/ROS code anywhere in the repo today —
`grep` for it comes back empty.  The four stubs above are the entire hardware
surface.

---

## 1. Rates

| quantity | value |
|---|---|
| sim timestep | 1 ms |
| FRAME_SKIP | 5 |
| **control / policy rate** | **200 Hz** (`CONTROL_DT = 5 ms`) |
| gait clock | `GAIT_HZ = 0.80` cycle/s → one L+R cycle = 1.25 s |
| gait-phase advance | `2π · 0.80 · dt` rad per control step (`= 0.02513` at 200 Hz) |

The observation builder advances the gait phase itself; if your real loop does
not hit exactly 200 Hz, pass the true `dt` to `ObservationBuilder(control_dt=…)`
and `run_hil.py --rate` so the phase advance stays wall-clock-correct.

Measured on the dev machine (`run_hil.py --sim --no-throttle`): full loop
~0.6 ms, sensor→action ~0.3 ms, so 200 Hz has ~8× headroom.  Inference alone is
~0.08 ms (pure-numpy 2×256 MLP).

**Sim/real gap to know:** in the sim the base controller (LQR + CPG + frontal)
runs 5× per control step (at 1 kHz); on hardware `run_hil.py` runs it once per
control step (200 Hz).  Irrelevant for log-only.  For the eventual closed loop,
running the base controller faster than 200 Hz (if the hardware allows) better
matches the sim.

## 2. Coordinate frames & units

**Chest body frame** (the IMU frame the policy expects).  From
`robot/_exp_hands_3x.xml`, the Chest body's local axes are:

```
  +X_chest  ≈ world +X   =  robot LEFT
  +Y_chest  ≈ world +Z   =  UP  (toward the head)          <-- note: Y is up, not Z
  +Z_chest  ≈ world -Y   =  FORWARD (the walk direction; measured: the sim robot travels +1.22 m along +Z_chest)
```

The robot walks toward world **−Y** (= +Z_chest).  The env's `_fwd_local` = −Z_chest is thus the *backward*
axis (misnamed); `fwd_xy` is only a yaw reference and is reproduced exactly as-is.  "World" for the IMU = the frame the IMU
fusion is zeroed to at startup, with the robot held in the nominal standing pose
(`qpos0`, roughly straight legs, arms ~0.13 rad out).

| observation element | frame | units | notes |
|---|---|---|---|
| `up` (3) | **world** | unit vec | `R_chest · [0,1,0]`; ≈ `[0,0,1]` upright. From IMU orientation. |
| `fwd_xy` (2) | **world** | unit vec | x,y of `R_chest · [0,0,−1]`; ≈ `[0,1]` at start. Encodes heading (yaw). |
| `gyro` (3) | see note | rad/s | `R_chestᵀ · ω_body`. **NOT the raw gyro** — the sim rotates the body-frame rate by `R_chestᵀ`. `observation.py` does this for you given `(quat, raw_gyro)`. |
| `accel` (3) | **chest body** | m/s² | specific force incl. gravity reaction — a normal accelerometer reading. ≈ `[0, +9.81, 0]` at rest. Feed the raw accel straight in. |
| `joint_pos` / `joint_vel` (14 each) | joint | rad, rad/s | POLICY_JOINT_ORDER (below), MuJoCo sign & zero |
| `contact` (2) | — | 0/1 | `[L, R]`, 1 = foot loaded |
| `v_est` (3) | **chest body** | m/s | leg-odometry base-velocity estimate — **computed by `hil.kinematics`**, not a sensor |
| `foot_L_rel`, `foot_R_rel` (3 each) | **chest body** | m | foot position from FK — **computed by `hil.kinematics`** |
| `clock_sin`, `clock_cos` | — | — | internal gait clock |
| `speed_tgt` | — | m/s | constant **0.30** |
| `prev_action` (14) | — | [−1,1] | last policy output, POLICY_JOINT_ORDER |

**IMU quaternion** is `[w, x, y, z]`, unit, body→world.  If your IMU gives
`[x,y,z,w]` reorder it.  If it gives Euler, convert.  If it gives only raw
gyro+accel (no fusion) you must run an AHRS filter (Madgwick/Mahony) to get the
orientation — `up` is well-observed from accel, heading (`fwd` yaw) needs
gyro-integration or a magnetometer.

**IMU mounting.**  The IMU's measurement axes must equal the chest body axes
above.  If they don't, set `HardwareRobot(imu_mount_R=R)` where
`x_chest = R · x_imu` — it is applied to gyro and accel, and you must also rotate
the quaternion.

**Gyro bias.**  Calibrate at rest and subtract before passing in
(`HardwareRobot._gyro_bias`).  Training assumed bias ≤ 0.03 rad/s.

## 3. Observation vector (217-D)

```
obs = [ frame(t-3) | frame(t-2) | frame(t-1) | frame(t) | extra ]
        \_____________ 4 × 50-D sensor frames, oldest first ____/   \_ 17-D _/
```

One **50-D sensor frame**, in order:
`up(3) fwd_xy(2) gyro(3) accel(3) joint_pos(14) joint_vel(14) contact(2) v_est(3) foot_L_rel(3) foot_R_rel(3)`

The **17-D extra**: `clock_sin clock_cos speed_tgt prev_action(14)`

At startup the 4-frame history is filled with copies of the first frame
(`ObservationBuilder.reset`).

**Normalisation** (do this before the policy — `Policy` does it internally):
```
x = clip( (obs − MEAN) / sqrt(VAR + 1e-8) , −10, +10 )
```
`MEAN`, `VAR` are the 217-D arrays in `vecnormalize_best.pkl`
(`VecNormalize.obs_rms`).  **Fixed** — never updated at run time.

## 4. Joint ordering

MuJoCo `ctrl` / `qpos[7:22]` / actuator order (15):
```
0 neck   1 L_shoulder  2 L_elbow  3 R_shoulder  4 R_elbow
5 L_hip_roll  6 L_hip_pitch  7 L_knee  8 L_ankle_pitch  9 L_ankle_roll
10 R_hip_roll 11 R_hip_pitch 12 R_knee 13 R_ankle_pitch 14 R_ankle_roll
```

**POLICY_JOINT_ORDER** (the 14-D used for the joint block of the observation AND
for the action — `spec.ACT_CTRL = [5,6,7,8,9,10,11,12,13,14,1,2,3,4]`):
```
0 L_hip_roll  1 L_hip_pitch  2 L_knee  3 L_ankle_pitch  4 L_ankle_roll
5 R_hip_roll  6 R_hip_pitch  7 R_knee  8 R_ankle_pitch  9 R_ankle_roll
10 L_shoulder 11 L_elbow    12 R_shoulder 13 R_elbow
```
The **neck is not in the policy** — it is always commanded to 0.

Joint limits / axes / signs: `spec.CTRL_RANGE_MJ` and the XML.  **Every encoder
must report the joint angle with the same zero and sign as MuJoCo `qpos`.**
Verify each joint in sim first (`python interactive_test_joints.py`) — drive one
joint, note direction, match it on hardware.

## 5. Policy action semantics

```
action           = policy(normalised_obs)            # 14-D, deterministic mean, clipped to [-1,1]
residual (rad)    = action * ACT_SCALE               # per-joint, POLICY_JOINT_ORDER
u (15-D, rad)     = base_controller(phase, IMU, FK, v_est, contact)   # CPG + LQR + frontal
u[POLICY joints] += residual
u                = clip(u, joint_limits);   u[neck] = 0
```

`ACT_SCALE` = `[0.20, 0.52, 0.55, 0.34, 0.20,  0.20, 0.52, 0.55, 0.34, 0.20,  0.55, 0.42, 0.55, 0.42]`.

`u` is a **joint position target** for the position servos
(`kp = 30`, torque limit **±2.3 N·m** — do not exceed).

**The arm/shoulder command rides against its limit.**  `Chest_L_shoulder`
range is `[-3.578, +0.175]`; the CPG arm-swing saturates the `+0.175` end ~50% of
the time (same in sim — the policy was trained with this clip).  The real
shoulder joints need at least this range; the code clips so nothing past the
limit is ever commanded.

The `base_controller` frontal stabiliser gates on foot **load > 12 N** in sim;
with only a contact switch it falls back to the contact boolean, which makes
single-support transitions slightly different from sim (fine for log-only —
matters for closed-loop).  If the robot has foot force/pressure sensing, pass
`foot_load_L/R` (Newtons) to `BaseController.compose`.

## 6. Safety checks (`hil/safety.py`, thresholds in `spec.SAFE`)

Every step, `run_hil.py` checks and logs:

* **NaN / shape** on every sensor field and the observation → `abort`
* **IMU sanity**: `|quat| ≈ 1`; `|accel| ∈ [2, 60] m/s²`; `|gyro| < 25 rad/s`
* **joints**: `|joint_vel| < 30 rad/s`; `joint_pos` within limits (+0.05 margin)
* **attitude**: chest tilt from vertical — `warn` > 25°, `abort` > 40°
* **command**: shape/NaN; per-joint "within `cmd_margin` of a limit" fraction;
  neck must be 0
* **timing**: rolling loop rate (`abort` < 120 Hz) and jitter (`warn` > 6 ms)

In log-only mode an `abort` stops the loop.  With `--enable-motors` you must
additionally latch a safe hold + cut torque on `abort` (not implemented — that
is a closed-loop concern).

## 7. The log

`hil/logs/hil_<timestamp>.npz` — per step: `t`, `read_ms`, `obs_ms`,
`infer_ms`, `compose_ms`, `loop_ms`, `obs` (217), `action` (14), `command` (15),
`tilt`, raw `imu_quat/imu_gyro/imu_accel`, `joint_pos/joint_vel`, `contact`,
`v_est`, `phase`, and an `events` list.  Plus a human-readable `.txt` summary
(latency percentiles, achieved rate, sensor ranges, saturation, safety events).

## 8. Bench-test order (before any closed loop)

1. `python -m hil.validate_stack` → PASS.
2. IMU: hold the chest level → `accel ≈ [0, 9.81, 0]`, `up ≈ [0,0,1]`, `gyro ≈ 0`.
   Tilt it forward (toward the walk direction, world −Y) → `up.y` goes negative, `up.z` < 1.
   Yaw it → `fwd_xy` rotates.  Spin about each axis → one `gyro` axis responds.
3. Encoders: move each joint by hand, confirm sign & zero match sim
   (`interactive_test_joints.py`).
4. Contacts: press each foot → the right boolean flips.
5. `python -m hil.run_hil --hardware --seconds 30` with the robot **held /
   hanging**, motors off.  Check the `.txt`: loop rate steady near target,
   `non-finite: 0`, `|accel|`/`|gyro|` sane, no `abort` events, sensor→action
   latency well under 5 ms.
6. Only then discuss closed-loop on a gantry (separate work — needs the
   fall-safe latch, torque ramp-in, and probably running the base controller
   faster than 200 Hz).

## 9. Raspberry Pi + MPU-6050 (real IMU, simulated body)  — added Sep 20 2026

Hardware confirmed by the user on the bench: Pi 4B (64-bit OS, host `shiv`, `sk@10.0.0.71`), MPU-6050 on I2C bus 1
at `0x68` (VCC→pin 1, GND→pin 6, SDA→pin 3, SCL→pin 5), read OK at ±2 g / ±250 °/s.  PCA9685 + TD-7120MG servos are
**not** used yet.  Note: hobby servos have **no position feedback** and no foot sensors were listed, so
`joint_pos/joint_vel` (encoders) and `contact` still have no hardware source — this mode takes them from the sim.

New files (all additive; existing scripts/results untouched — verified bit-identical `--sim` logs):

| file | role |
|---|---|
| `hil/mpu6050.py` | register-level I2C driver (smbus2, lazy import) + `FakeBus` register emulator; flags saturated samples |
| `hil/imu_fusion.py` | calibration (gyro bias, per-axis accel offset/scale, sensor→chest rotation), 6-axis Mahony AHRS → quaternion, start-up heading matched to the sim standing pose |
| `hil/calibrate_mpu.py` | `stationary` · `six-point` · `mount` · `live` (writes `hil/mpu_calib.json`, git-ignored) |
| `hil/hybrid_robot.py` | real IMU + MuJoCo joints/contacts, torso pinned (gantry), log-only |
| `hil/run_hil.py --mpu` | additive flag: the same loop/logging/safety with the hybrid robot |
| `hil/selftest_mpu.py` | hardware-free proof of the whole chain (fake bus fed by the sim) |
| `hil/make_pi_bundle.py` | builds `dist/hil_pi_bundle.zip` (hil/ + model + `hil_policy.npz` + consts; no torch needed) |

Run order on the Pi: `selftest_mpu` → `calibrate_mpu stationary / six-point / mount / live` → `run_hil --sim` →
`run_hil --mpu`.  Chest frame for the mount step: **+X left, +Y up, +Z forward**.

Known limits: MPU-6050 has no magnetometer → yaw (`fwd_xy`) drifts (~1° in 3 s in the fake-sensor test, real drift is
temperature/bias dependent); the pinned sim torso ignores the real IMU, so this validates **sensor conventions,
noise, latency and the policy's response to a real IMU signal — not balance**.  ±2 g / ±250 °/s clip in walking
(sim IMU peaks ≈ 8 g / 750 °/s at foot strikes, partly a finite-difference artifact) → use `--accel-g 8 --gyro-dps 1000`
or higher for anything dynamic; saturation is counted in the log summary.

## 10. Servo bring-up (PCA9685) — added Sep 20 2026, NOT yet run on hardware

`hil/pca9685.py` (driver + `ServoBank`: 15-D MuJoCo-order command → pulses, per-joint slew limit, hard pulse clamps,
skips any joint with an incomplete map) and `hil/servo_test.py` (single-servo bench sweep, **dry-run unless
`--enable`**, PWM released on exit).  `hil/selftest_servo.py` proves the logic on a fake bus.  Nothing about the
servo (pulse range, µs/rad, direction, channel) is assumed: `python -m hil.servo_test --write-template` writes an
all-`null` `hil/servo_map.json`.  `--amp` is capped at 0.5 rad.  The first pulse snaps the servo to angle 0.
Not connected to `HardwareRobot`/`run_hil` yet — the servo bank is only reachable through `servo_test`.

## 11. Actuator-robustness eval — added Sep 20 2026 (eval only)

`python -m hil.eval_servo_robust --n 24` runs the deployment stack against MuJoCo with a servo model (hold-rate → delay →
deadband → slew limit → lag) and encoder-free feedback variants, using IMU noise **measured on the user's MPU-6050**.
Full results and conclusions: `hil/SERVO_ROBUSTNESS.md`.  Headline: with ideal actuators the 200 Hz deployment stack
survives 62 % (92 % with an env-like 1 kHz base controller); +10 ms lag / +5 ms delay / 50 rad/s slew / no joint
feedback each drop it to ≈ 0–8 % — `sim2real_v1` needs fast, feedback-equipped actuators or a new policy trained
against the deployment pipeline.  Also found: the Pi `--mpu` loop ran at 162.6 Hz, not 200 Hz.

## 12. Live link: real MPU-6050 (Pi) -> simulator with viewer (PC) — added Sep 20 2026

`hil/imu_stream.py` (Pi) streams the calibrated IMU (quaternion, gyro, accel; same chain as `run_hil --mpu`) over UDP;
`hil/live_sim.py` (PC) receives it, PUPPETS the sim torso to the real quaternion (position pinned), runs the
deployment stack (obs → sim2real_v1 → base controller) on the real IMU data with sim joints/contacts, and shows it in the
MuJoCo viewer.  No actuator is driven.  Tests: `python -m hil.selftest_live` (fake MPU → loopback UDP → live_sim:
0 dropped packets, 200 Hz, ~2 ms packet age, sway 17.3° vs 17.2° sent, torso == real quaternion, policy responds).
The viewer window itself could not be exercised in the tool sandbox (no OpenGL); the repo's existing scripts use the
same `launch_passive` call.  Windows Firewall may prompt for Python on the private network the first time.

**Viewer note (Sep 20 2026):** on the user's PC the MuJoCo viewer cannot open a window — the Radeon RX 480 reports
`Status: Error` (Win32_VideoController) → "WGL: The driver does not appear to support OpenGL" (MuJoCo hard-exits the
process and appends an ERROR line to the tracked `MUJOCO_LOG.TXT` — restore it with `git checkout -- MUJOCO_LOG.TXT`).
`live_sim` therefore defaults to `--viz stick` (`hil/stick_viz.py`, tkinter side + front stick figures with the chest
up/forward axes, foot-contact markers and IMU readouts; no OpenGL/matplotlib/OpenCV needed).  `--viz mujoco` = the
OpenGL viewer once the GPU driver is healthy; `--viz none` = headless.
