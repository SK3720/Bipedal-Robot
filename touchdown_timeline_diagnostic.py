"""Baseline touchdown timeline from staged_forward_catch_test.

Records airborne start, peak swing, descent, first contact, sole angles.
"""

from __future__ import annotations

import sys

import numpy as np

from biped_env import BipedalWalkEnv
from foot_geometry import legacy_foot_pitch_rad, sole_metrics
from staged_forward_catch_test import (
    Phase,
    QPOS_L_KNEE,
    RunState,
    StepDiagnostics,
    _foot_contact,
    _foot_normal_force,
    _foot_pos,
    _forward_mm,
    _forward_vel,
    _is_unloaded,
    _lerp_ctrl,
    _reset,
    _sim_step,
    _smooth,
    left_leg_pose,
    rapid_shift_pose,
    forward_lean_pose,
    run_staged_forward_catch,
)
from staged_forward_catch_continue_test import _r_load_fraction
import staged_forward_catch_test as base

from biped_env import DEFAULT_POSE

IDX_L_ANKLE_P = 8
QPOS_ANKLE_P = 15
FOOT_CONTACT_Z = 1.042


class TimelineRecorder:
    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.airborne_start: int | None = None
        self.peak_fwd_mm = 0.0
        self.peak_fwd_step: int | None = None
        self.first_contact_step: int | None = None
        self.was_airborne = False

    def maybe_record(
        self,
        env,
        model,
        data,
        st: RunState,
        phase: str,
    ) -> None:
        if phase not in (Phase.HIP_SWING.value, Phase.CATCH.value):
            return
        l_contact = _foot_contact(model, data, "L")
        l_pos = _foot_pos(model, data, "L")
        clearance_mm = (l_pos[2] - FOOT_CONTACT_Z) * 1000.0
        airborne = not l_contact

        if airborne and self.airborne_start is None and phase == Phase.HIP_SWING.value:
            self.airborne_start = st.diag.global_step

        if airborne and st.airborne_ref_y is not None:
            fwd = _forward_mm(l_pos[1], st.airborne_ref_y)
            if fwd > self.peak_fwd_mm:
                self.peak_fwd_mm = fwd
                self.peak_fwd_step = st.diag.global_step

        if self.was_airborne and l_contact and self.first_contact_step is None:
            self.first_contact_step = st.diag.global_step

        if airborne:
            self.was_airborne = True

        if phase == Phase.CATCH.value or (
            phase == Phase.HIP_SWING.value and airborne
        ):
            if st.diag.global_step % 5 == 0 or l_contact:
                m = sole_metrics(model, data)
                self.samples.append(
                    {
                        "step": st.diag.global_step,
                        "phase": phase,
                        "contact": l_contact,
                        "clearance_mm": clearance_mm,
                        "fwd_air_mm": _forward_mm(l_pos[1], st.airborne_ref_y)
                        if st.airborne_ref_y
                        else 0.0,
                        "ankle_cmd": float(data.ctrl[IDX_L_ANKLE_P]),
                        "ankle_qpos": float(data.qpos[QPOS_ANKLE_P]),
                        "knee_qpos": float(data.qpos[QPOS_L_KNEE]),
                        "sole_horiz_deg": np.degrees(m["sole_angle_from_horizontal_rad"]),
                        "heel_toe_pitch_deg": np.degrees(m["heel_toe_pitch_rad"]),
                        "legacy_pitch_deg": np.degrees(legacy_foot_pitch_rad(model, data)),
                        "heel_clr_mm": m["heel_clearance_mm"],
                        "toe_clr_mm": m["toe_clearance_mm"],
                        "l_nf": _foot_normal_force(model, data, "L"),
                        "r_nf": _foot_normal_force(model, data, "R"),
                        "r_frac": _r_load_fraction(model, data),
                        "torso_fwd_vel": _forward_vel(data),
                        "torso_tilt": env._quat_tilt_rad(),
                    }
                )


def run_baseline_timeline() -> TimelineRecorder:
    env = BipedalWalkEnv()
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)
    rec = TimelineRecorder()

    shifted = rapid_shift_pose()
    lean = forward_lean_pose()
    st = RunState(
        ctrl=DEFAULT_POSE.copy(),
        phase=Phase.STAND,
        stand_r_xy=_foot_pos(model, data, "R")[:2].copy(),
        diag=StepDiagnostics(),
        knee_cmd=float(shifted[IDX_L_KNEE]) if False else float(shifted[7]),
        hip_cmd=float(shifted[6]),
    )

    # Reuse logic inline to inject recorder — call patched sim
    from staged_forward_catch_test import IDX_L_KNEE, IDX_L_HIP_PITCH

    for s in range(base.STAND_STEPS):
        st.ctrl = _lerp_ctrl(st.ctrl, DEFAULT_POSE, _smooth((s + 1) / base.STAND_STEPS), cr)
        _sim_step(env, model, data, st, None, False)
    for s in range(base.FALL_RAMP_STEPS):
        st.ctrl = _lerp_ctrl(st.ctrl, lean, (s + 1) / base.FALL_RAMP_STEPS, cr)
        _sim_step(env, model, data, st, None, False)
    for _ in range(base.FALL_MOMENTUM_STEPS):
        st.ctrl = lean.copy()
        _sim_step(env, model, data, st, None, False)
    shift_start = st.ctrl.copy()
    for s in range(base.RAPID_SHIFT_STEPS):
        st.ctrl = _lerp_ctrl(shift_start, shifted, (s + 1) / base.RAPID_SHIFT_STEPS, cr)
        _sim_step(env, model, data, st, None, False)
    st.phase = Phase.KNEE_LIFT
    while st.knee_lift_steps < base.KNEE_LIFT_MAX_STEPS and st.phase == Phase.KNEE_LIFT:
        if _is_unloaded(model, data):
            st.clearance_knee = st.knee_cmd
            st.phase = Phase.HIP_SWING
            st.diag.hip_swing_start_step = st.diag.global_step + 1
            break
        st.knee_cmd = max(st.knee_cmd - base.KNEE_LIFT_STEP, base.KNEE_LIFT_TARGET)
        st.ctrl = left_leg_pose(st.knee_cmd, st.hip_cmd)
        _sim_step(env, model, data, st, None, False)
        st.knee_lift_steps += 1
    if st.phase == Phase.KNEE_LIFT:
        st.phase = Phase.HIP_SWING
    st.diag.hip_swing_start_step = st.diag.global_step + 1

    while st.hip_swing_steps < base.HIP_SWING_MAX_STEPS and st.phase == Phase.HIP_SWING:
        if st.airborne_ref_y is None and not _foot_contact(model, data, "L"):
            st.airborne_ref_y = float(_foot_pos(model, data, "L")[1])
        knee_hold = st.clearance_knee if st.clearance_knee is not None else st.knee_cmd
        st.hip_cmd = max(st.hip_cmd - base.HIP_RAMP_PER_STEP, base.HIP_SWING_TARGET)
        st.ctrl = left_leg_pose(knee_hold, st.hip_cmd)
        _sim_step(env, model, data, st, None, False)
        rec.maybe_record(env, model, data, st, Phase.HIP_SWING.value)
        st.hip_swing_steps += 1
        fwd_air = _forward_mm(_foot_pos(model, data, "L")[1], st.airborne_ref_y) if st.airborne_ref_y else 0
        if fwd_air >= base.MIN_FWD_AIRBORNE_MM and not _foot_contact(model, data, "L"):
            st.phase = Phase.CATCH
            break

    if st.phase == Phase.HIP_SWING:
        st.phase = Phase.CATCH

    catch_start = st.ctrl.copy()
    catch_target = left_leg_pose(base.L_KNEE_CATCH, base.L_HIP_CATCH, base.L_ANKLE_CATCH)
    while st.catch_steps < base.CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(base.CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start, catch_target, alpha, cr)
        _sim_step(env, model, data, st, None, False)
        rec.maybe_record(env, model, data, st, Phase.CATCH.value)
        st.catch_steps += 1
        if rec.first_contact_step is not None and st.catch_steps > 30:
            break

    return rec


def main() -> None:
    rec = run_baseline_timeline()
    print("BASELINE TOUCHDOWN TIMELINE (staged_forward_catch_test)")
    print(f"AIRBORNE_START_STEP = {rec.airborne_start}")
    print(f"PEAK_FWD_STEP = {rec.peak_fwd_step}  PEAK_FWD_MM = {rec.peak_fwd_mm:.1f}")
    print(f"FIRST_CONTACT_STEP = {rec.first_contact_step}")

    if rec.first_contact_step:
        pre = [s for s in rec.samples if s["step"] < rec.first_contact_step and not s["contact"]]
        at = [s for s in rec.samples if s["step"] == rec.first_contact_step]
        if pre:
            last = pre[-1]
            print("\nLAST AIRBORNE BEFORE CONTACT:")
            for k, v in last.items():
                print(f"  {k} = {v}")
        if at:
            print("\nAT FIRST CONTACT:")
            for k, v in at[0].items():
                print(f"  {k} = {v}")

    print("\nSAMPLES (last 15 before contact):")
    if rec.first_contact_step:
        pre = [s for s in rec.samples if s["step"] <= rec.first_contact_step][-15:]
        print(" step  clr   sole°  ht°   legacy° ankle_q  knee_q  l_nf  fwd_v")
        for s in pre:
            print(
                f"{s['step']:5d} {s['clearance_mm']:5.1f} "
                f"{s['sole_horiz_deg']:6.1f} {s['heel_toe_pitch_deg']:5.1f} "
                f"{s['legacy_pitch_deg']:7.1f} {s['ankle_qpos']:7.3f} "
                f"{s['knee_qpos']:7.3f} {s['l_nf']:5.1f} {s['torso_fwd_vel']:.3f}"
            )


if __name__ == "__main__":
    main()
