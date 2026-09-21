# loco_w3 — SIM-TO-REAL CANDIDATE

Designated 2026-09-03 as the walking policy to carry toward hardware transfer.

## Checkpoint
- `runs/loco_w3/policy_best.pth` (== `policy_at_14M_s4p466.pth`, 14 M PPO steps)
- `runs/loco_w3/vecnormalize_best.pth` — obs normalisation (REQUIRED at inference)
- also preserved: `policy_at_{7p75M,10M}.pth`, `snap_final.pth`

## Morphology
- `robot/_exp_hands_3x.xml` — canonical `robot/robot.xml` kinematics/geometry
  (untouched, md5 `b4c4585dda6a6f41365484908f6377ea`), **normal feet**
  (sole ~0.088 m fore-aft × 0.066 m), hand mass 3× (arms RL-controlled).
- Actuators: ±2.3 N·m force range, position servos kp≈30, 1 kHz sim,
  FRAME_SKIP 5 → 200 Hz control.

## Measured performance (40-ep eval @ 0.30 m/s)
| metric | value |
|---|---|
| survival | 96 % |
| falls | 5 % |
| forward speed | 0.31 m/s (105 % of target) |
| stride | ~0.16 m/step |
| cadence | ~2.7 steps/s |
| distance | 2.14 m per 7 s episode |
| bilateral flight | 1.9 % |
| dives / lunges | 0 % |

## Policy interface
- obs: 68-d (see `_build_obs`) — chest up/fwd axes, base ang-vel, CoM vel,
  height err, foot fwd/z, inter-foot vector, contacts + normal forces,
  14 leg+arm joint pos, 14 joint vel, gait clock (sin/cos), speed target+err,
  prev action. **Normalise with the saved VecNormalize.**
- action: 14-d in [-1,1] → position residual (ACT_SCALE) on the CPG reference +
  attitude-only StandingLQR torso hold + frontal-plane CoP feedback.
- The CPG clock advances at GAIT_HZ=0.80; the LQR/CPG base is deterministic
  given state, so the full controller = f(state, clock, policy).

## IMU + ENCODER POLICY — `runs/sim2real_v1/` (2026-09-06)

`loco_w3` uses privileged observations; the deployable version is
`runs/sim2real_v1/policy_best.pth`.

- **Observation (217-d)** — chest gravity vec + heading (IMU), body-frame gyro,
  body-frame specific force (accelerometer), 14 joint pos + vel (encoders), 2
  foot-contact booleans, a leg-odometry base-velocity estimate, FK foot
  positions in the base frame — each with real-sensor noise, **stacked over 4
  frames** — plus gait clock, speed command, prev action.  Every element maps to
  a real driver output; see `biped_sim2real_env.py::_sensor_frame`.
- **Base controller** — CPG reference + attitude LQR with the height terms masked
  out + a frontal CoP law rebuilt from FK + leg-odometry (`_compose_ctrl`).  All
  IMU + encoder derivable.
- **How it was trained** — pure supervised: `loco_w3` as teacher, behaviour
  clone + 3 DAgger rounds (`distill_s2r.py`).  NO reinforcement learning (RL
  cold-start didn't converge; RL fine-tuning collapsed the distilled policy).
- **Performance** (30-seed eval, deterministic): robust 0.0 → 97 % survive,
  2.21 m/ep, 25 genuine steps, 4 % flight; robust 0.15 → 84 %.
- **Run it (sim):** `python biped_sim2real_env.py --watch --policy runs/sim2real_v1`
- Inference = `policy_best.pth` on obs normalised by `vecnormalize_best.pkl`.

### HARDWARE / HIL  ->  `hil/`  (2026-09-07)

Full hardware-interface + log-only HIL package, validated bit-for-bit vs the
sim.  `hil/README.md` has the complete spec (217-D obs layout, frames, units,
joint orders, normalisation, action semantics, rates, safety).  Start with:

    python -m hil.validate_stack          # must print PASS
    python -m hil.run_hil --sim           # dry-run the whole stack
    # wire up hil/robot_interface.py::HardwareRobot (4 stubs), then:
    python -m hil.run_hil --hardware      # real sensors, motors DISABLED, logs everything

Base-controller constants exported to `base_controller_consts.npz` in this dir.

## KNOWN sim-to-real gaps (to close before hardware)
1. **Foot-contact chatter** — the sim gait makes/breaks contact ~130×/episode
   (feet vibrating). Needs a chatter penalty + contact filtering.
2. **No domain randomisation yet** — trained on nominal friction/mass/gains,
   no sensor noise, no actuation latency, no external disturbances.
3. **Privileged obs** — `sample_balance` uses true CoM / full state. Real robot
   has IMU + joint encoders (+ maybe foot contacts) only. Needs an obs redesign
   or an asymmetric actor/critic.
4. Torso attitude via `StandingLQR` linearised about standing — fine in sim,
   assumes a good state estimate on hardware.

See `../../WALKING_LOG.md` and the `normal-feet-walk-and-stride-ceiling` memory.

## Update Sep 20 2026 — real MPU-6050 on a Raspberry Pi

`hil/` now supports a real MPU-6050 (I2C 0x68) driving the IMU channels of the sim2real_v1 observation with the
joints/contacts simulated and the torso pinned (`python -m hil.run_hil --mpu`, log-only, no actuators).  See
`hil/README.md` §9.  Pi bundle: `python -m hil.make_pi_bundle` → `dist/hil_pi_bundle.zip` (torch-free,
`runs/sim2real_v1/hil_policy.npz`).  The policy, checkpoints and `robot.xml` are untouched.
