"""Ankle pitch vs sole angle at catch-like leg pose (not standing)."""

from __future__ import annotations

import numpy as np
import mujoco

from biped_env import BipedalWalkEnv, CHEST_Z_CONTACT, DEFAULT_POSE, STANDING_QUAT
from foot_geometry import sole_metrics

IDX_L_HIP_P = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
QPOS_HIP = 13
QPOS_KNEE = 14
QPOS_ANKLE_P = 15

CATCH_HIP = -0.32
CATCH_KNEE = -0.14


def main() -> None:
    env = BipedalWalkEnv()
    model, data = env.model, env.data
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = 0.0
    data.qvel[:] = 0.0

    ctrl = DEFAULT_POSE.copy()
    ctrl[IDX_L_HIP_P] = CATCH_HIP
    ctrl[IDX_L_KNEE] = CATCH_KNEE

    print("SOLE ANGLE vs L ANKLE PITCH at catch-like pose (hip=-0.32, knee=-0.14)")
    print("pitch_cmd  pitch_qpos  sole_from_horiz°  heel_toe_pitch°  heel_clr_mm")
    for ap in np.linspace(-0.4, 0.6, 11):
        ctrl[IDX_L_ANKLE_P] = float(ap)
        for _ in range(600):
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)
        m = sole_metrics(model, data)
        print(
            f"{ap:8.2f}  {data.qpos[QPOS_ANKLE_P]:10.3f}  "
            f"{np.degrees(m['sole_angle_from_horizontal_rad']):14.1f}  "
            f"{np.degrees(m['heel_toe_pitch_rad']):14.1f}  "
            f"{m['heel_clearance_mm']:10.1f}"
        )

    best = min(
        [
            (
                ap,
                sole_metrics(model, data)["sole_angle_from_horizontal_rad"],
            )
            for ap in np.linspace(-0.5, 0.5, 21)
            for _ in [
                data.__setitem__(
                    "ctrl",
                    np.array(
                        [
                            *ctrl[:IDX_L_ANKLE_P],
                            ap,
                            *ctrl[IDX_L_ANKLE_P + 1 :],
                        ]
                    ),
                )
            ]
            for _ in range(400)
            for _ in [mujoco.mj_step(model, data)]
        ],
        key=lambda x: x[1],
    )
    print(f"\nFlattest near ankle pitch cmd ~ {best[0]:.2f} (sole horiz {np.degrees(best[1]):.1f}°)")


if __name__ == "__main__":
    main()
