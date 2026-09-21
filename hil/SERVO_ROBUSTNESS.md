# sim2real_v1 vs real-actuator behaviour — measurement report (Sep 20 2026)

**Eval only. No policy, checkpoint or existing script was modified or retrained.**
Reproduce: `python -m hil.eval_servo_robust --n 24` (≈1 min on 11 cores).  Raw output: `hil/logs/servo_robust_*.txt/.json`.

## Conclusion

`sim2real_v1` is **not deployable on rate-limited, laggy, feedback-less servos as it stands**, and it has almost no
margin even with ideal actuators when run the way the Pi would run it:

1. **Deployment rate matters.** The env runs the attitude/CoP base controller at 1 kHz on ground-truth state.  The
   deployment stack runs it once per 200 Hz step on sensor data.  Ideal actuators + *measured* MPU-6050 noise:
   **62 % survive at 200 Hz vs 92 % with a 1 kHz base controller** (24 episodes, 7 s each).  The 200 Hz stack also
   makes the leg commands chatter (hip-pitch command rate p95 ≈ 180 rad/s vs ≈ 30 rad/s at 1 kHz).
2. **Actuator tolerance, 200 Hz stack (as coded for the Pi):** +5 ms lag → 38 %, +10 ms lag → 4 %, +5 ms delay → 4 %,
   50 rad/s slew limit → 8 %.  Essentially zero margin.
3. **Actuator tolerance, 1 kHz base (best case):** 20 ms lag → 67 %, 25 rad/s slew → 71 %, 12 rad/s slew → 0 %,
   lag 40 ms / 6 rad/s / delay 10 ms combined → 0 %.
4. **Joint feedback is required.** Replacing encoders by the commanded angles (or a smoothed/dead-reckoned version,
   τ = 15–60 ms) gives 0–4 % even with ideal actuators.
5. **The policy saturates the sim actuators** (p95 per-step peak torque = the 2.3 N·m limit on every actuated joint) and
   demands command rates of 20–70 rad/s p95 on the leg joints (1 kHz base) — check any candidate servo's datasheet
   speed (typical hobby servos are rated on the order of 60° per 0.1–0.2 s ≈ 5–10 rad/s — verify yours) and stall torque.

Implication: either (A) hardware with position feedback, ≲ 20 ms effective lag, ≳ 25 rad/s slew and ≥ the sim torque,
plus a fast inner control loop; and/or (B) a **new, separately-named** policy trained/distilled in a sim that includes
the deployment-rate base controller, measured sensor noise, a servo model and the chosen feedback.  (C) Measure the
real servo first (lag / slew / deadband / stall torque — MPU-6050 taped to the horn gives lag/slew without encoders)
and read the answer off this table.

## Method and caveats

* Stack = `hil.ObservationBuilder + hil.Policy + hil.BaseController` (the code that runs on the Pi) against MuJoCo
  (`robot/_exp_hands_3x.xml`), 200 Hz policy, 7 s episodes, `fell()` = tilt > 48° or chest drop > 0.30 m.
* Noise profile `hw`: MEASURED on the user's calibrated MPU-6050 from the Pi log `hil_20260920_214508`
  (still windows): gyro white 0.0007 rad/s, accel 0.016 m/s², residual gyro bias ≈ 0.0005 rad/s — 30×/22×/60× below the
  0.02 / 0.35 / 0.03 the policy was distilled against.  No encoder noise, no contact dropout.  (`--noise train` = the
  full training noise, 12-episode diagnostic: ideal survival ≈ 8 %.)
* Servo model (a **sweep, not the TD-7120MG's real numbers**): command → hold at PWM-frame rate → transport delay →
  deadband → slew limit → first-order lag → MuJoCo position actuator (kp = 30, ±2.3 N·m).
* `1 kHz base` rows call the base controller every 1 ms on **ground-truth** state (as the env does) — an optimistic upper
  bound, not deployable on the Pi.
* Each config is a deterministic policy under small random sensor noise, 24 seeds: ±10 % is within noise; the
  qualitative cliffs (0 % vs 60–90 %) are not.
* Not modelled: servo torque < 2.3 N·m, backlash, compliance, supply sag, gear-train friction, real foot contact.

## Pi timing found in the same session (user's 30 s `--mpu` run on the Pi 4B)

Achieved loop 162.6 Hz (target 200; dt 4.8–9.4 ms): sensor read 2.78 ms, obs 0.66, inference 0.53, compose 0.41,
full loop 6.2 ms (includes sim stepping + logging).  Likely cause: I²C at the default 100 kHz.  The policy clock assumes
5 ms steps, so a slower real loop skews gait timing and the finite-difference terms.  Also `ServoBank.write` does 15
separate I²C transactions per step (too slow at 100 kHz) — needs one auto-increment block write.

## Results

```
sim2real_v1 vs servo behaviour  (24 episodes/config, 7 s each, deployment stack, noise profile 'hw'; NOT the env's own eval)

config                                        survive  dist m    m/s rate-limited
ideal (reference)                                 62%    1.92   0.32         0.0%
no encoders (cmd feedback)                         0%    0.03   0.02         0.0%
lag 40 ms                                          0%    0.31   0.32         0.0%
lag 80 ms                                          0%    0.28   0.29         0.0%
rate limit 6 rad/s                                 0%    0.22   0.17        82.2%
rate limit 3 rad/s                                 0%    0.33   0.36        87.2%
rate limit 1.5 rad/s                               0%    0.33   0.39        88.6%
delay 20 ms                                        0%    0.07   0.10         0.0%
delay 40 ms                                        0%    0.18   0.26         0.0%
deadband 0.02 rad                                 54%    1.76   0.34         0.0%
cmd refresh 100 Hz (PWM frame)                     0%    0.73   0.33         0.0%
cmd refresh 50 Hz (PWM frame)                      0%    0.32   0.35         0.0%
combo A (lag40+6r/s+d10+db.01)                     0%    0.03   0.07        81.9%
combo A + no encoders                              0%    0.08   0.12        85.9%
combo C (lag80+3r/s+d20+db.02)+no enc              0%    0.21   0.19        86.6%
1 kHz base ctrl (env-like), ideal                 92%    2.24   0.32         0.0%
1 kHz base + rate 6 r/s                            4%    0.37   0.27        79.6%
1 kHz base + combo A + no encoders                 0%    0.13   0.18        86.3%
lag 5 ms                                          38%    1.55   0.32         0.0%
lag 10 ms                                          4%    0.64   0.23         0.0%
lag 20 ms                                          4%    0.41   0.19         0.0%
rate limit 50 rad/s                                8%    0.98   0.33        29.7%
rate limit 25 rad/s                               12%    0.57   0.24        45.4%
rate limit 12 rad/s                                8%    0.25   0.19        67.7%
delay 5 ms                                         4%    0.72   0.29         0.0%
delay 10 ms                                        0%    0.19   0.23         0.0%
1 kHz base + lag 10 ms                            75%    1.79   0.30         0.0%
1 kHz base + lag 20 ms                            67%    1.72   0.30         0.0%
1 kHz base + rate 25 r/s                          71%    2.01   0.32        39.7%
1 kHz base + rate 12 r/s                           0%    0.18   0.22        46.9%
no enc: smoothed cmd, tau 15 ms                    0%   -0.00   0.00         0.0%
no enc: smoothed cmd, tau 30 ms                    0%    0.12   0.11         0.0%
no enc: smoothed cmd, tau 60 ms                    4%    0.27   0.19         0.0%

What the policy demands of each actuator (ideal actuators; rate = |d command/dt| at the 200 Hz step, torque = per-step peak; sim torque limit 2.3 N*m):
joint                  | 200 Hz base: rate p95    max | 1 kHz base: rate p95    max | torque p95   max
Chest_neck             |                   0.0      0 |                  0.0      0 |       0.00  0.00
Chest_L_shoulder       |                  33.5    141 |                 25.2    155 |       2.30  2.30
L_arm_L_elbow          |                  35.9     97 |                 30.1     90 |       2.30  2.30
Chest_R_shoulder       |                  45.3    151 |                 48.7    177 |       2.30  2.30
R_arm_R_elbow          |                  33.7     97 |                 22.3    103 |       2.30  2.30
Chest_L_hip_roll       |                  74.5    136 |                 42.6    113 |       2.30  2.30
L_hip_L_hip_pitch      |                 180.0    316 |                 29.6    216 |       2.30  2.30
L_leg_L_knee           |                 134.0    262 |                 71.1    200 |       2.30  2.30
L_shin_L_ankle_pitch   |                  44.1    154 |                 34.1    117 |       2.30  2.30
L_ankle_L_ankle_roll   |                  24.0     73 |                 23.2     69 |       2.30  2.30
Chest_R_hip_roll       |                  72.3    114 |                 23.9     73 |       2.30  2.30
R_hip_R_hip_pitch      |                 162.7    451 |                 35.1    183 |       2.30  2.30
R_leg_R_knee           |                 131.5    256 |                 46.8    171 |       2.30  2.30
R_shin_R_ankle_pitch   |                  51.1    114 |                 34.0    101 |       2.30  2.30
R_ankle_R_ankle_roll   |                  31.8     92 |                 20.5     93 |       2.30  2.30
```
