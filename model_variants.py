"""Isolated MuJoCo model variants for the actuator / stance-width experiment.

The BASELINE is robot/robot.xml (unchanged; also frozen as robot/robot_baseline.xml).
Variants are generated from it by string substitution and written to robot/ so the
mesh paths resolve, then loaded.  Nothing here modifies robot.xml.

Variants (change one thing at a time):
  baseline   : robot/robot.xml as-is
               leg actuators +/- 2.3 N.m ; hips at x = +0.055 / +0.015 (40 mm, both +x)
  strong_act : leg actuator forcerange -> +/- 6.0 N.m (2.6x)  [everything else baseline]
  wide       : hips moved to x = +0.085 / -0.015 (100 mm, symmetric about x=+0.035)
  strong     : strong_act + wide  (both changes)

`kp` (position-servo stiffness) is left at 30 for all variants - the experiment is
about torque authority and stance geometry, not the servo gain.
"""

from __future__ import annotations

import os
import re
import tempfile

import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(_HERE, "robot", "robot.xml")

STRONG_FORCE = 6.0          # N.m  (baseline 2.3)
# stance centred on x = +0.035 (the whole-body CoM x); baseline width 40 mm
L_HIP_X_WIDE = 0.0775       # baseline 0.0549754   -> 85 mm stance, symmetric
R_HIP_X_WIDE = -0.0075      # baseline 0.0148881

# The baseline arms hang right next to the hips; widening the stance makes the
# hand collision geoms penetrate the hip collision geoms at rest (a modelling
# artifact - verified in _wide_diag2).  Exclude those self-collision pairs in
# ALL variants so the comparison is clean; baseline numbers are re-verified to be
# unchanged by this.
_CONTACT_EXCLUDES = """
  <contact>
    <exclude body1="L_hand" body2="L_hip"/>
    <exclude body1="R_hand" body2="R_hip"/>
  </contact>
"""
# baseline geometry, for reference / metrics
BASE_LEG_FORCE = 2.3
BASE_L_HIP_X = 0.0549754
BASE_R_HIP_X = 0.0148881
STANCE_MID_X = 0.035

VARIANTS = ("baseline", "strong_act", "wide", "strong")


def _variant_xml(strong_actuators: bool, wide_stance: bool, excludes: bool = True) -> str:
    with open(_BASE_PATH) as f:
        s = f.read()

    if excludes and "<contact>" not in s:
        s = s.replace("</worldbody>", "</worldbody>\n" + _CONTACT_EXCLUDES)

    if strong_actuators:
        # only the 10 leg actuators (names contain hip / knee / ankle)
        def bump(m):
            line = m.group(0)
            if any(k in line for k in ("hip", "knee", "ankle")):
                return line.replace('forcerange="-2.3 2.3"', f'forcerange="-{STRONG_FORCE} {STRONG_FORCE}"')
            return line
        s = re.sub(r'<position name=[^/]*/>', bump, s)

    if wide_stance:
        s = s.replace('<body name="L_hip" pos="0.0549754 -0.02575 0.0452117"',
                      f'<body name="L_hip" pos="{L_HIP_X_WIDE} -0.02575 0.0452117"')
        s = s.replace('<body name="R_hip" pos="0.0148881 -0.02575 0.0452634"',
                      f'<body name="R_hip" pos="{R_HIP_X_WIDE} -0.02575 0.0452634"')
    return s


def variant_config(variant: str) -> dict:
    strong = variant in ("strong_act", "strong")
    wide = variant in ("wide", "strong")
    return dict(
        variant=variant,
        strong_actuators=strong,
        wide_stance=wide,
        leg_force=STRONG_FORCE if strong else BASE_LEG_FORCE,
        l_hip_x=L_HIP_X_WIDE if wide else BASE_L_HIP_X,
        r_hip_x=R_HIP_X_WIDE if wide else BASE_R_HIP_X,
    )


def load_model(variant: str = "baseline") -> mujoco.MjModel:
    """All variants (incl. 'baseline') get the arm/hip contact excludes so the
    A/B/C/D comparison is apples-to-apples.  The truly raw model is robot/robot.xml
    (unchanged) and robot/robot_baseline.xml; use load_raw_baseline() for that."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; have {VARIANTS}")
    cfg = variant_config(variant)
    xml = _variant_xml(cfg["strong_actuators"], cfg["wide_stance"])
    # write next to robot.xml so meshes/<...>.stl resolve
    path = os.path.join(_HERE, "robot", f"_variant_{variant}.xml")
    with open(path, "w") as f:
        f.write(xml)
    try:
        return mujoco.MjModel.from_xml_path(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def load_raw_baseline() -> mujoco.MjModel:
    """The unchanged robot/robot.xml exactly as the existing experiments use it."""
    return mujoco.MjModel.from_xml_path(_BASE_PATH)


def describe(variant: str) -> str:
    m = load_model(variant)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    import numpy as np
    lf = d.xpos[m.body("L_foot").id][0]
    rf = d.xpos[m.body("R_foot").id][0]
    fr = m.actuator_forcerange
    # leg actuator ids
    leg = [i for i in range(m.nu)
           if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
                  for k in ("hip", "knee", "ankle"))]
    leg_force = fr[leg[0], 1] if leg else float("nan")
    return (f"{variant:11s}  leg_force=+/-{leg_force:.1f} N.m   "
            f"foot_x L={lf:+.3f} R={rf:+.3f}  stance_width={abs(lf-rf)*1000:.0f}mm  "
            f"total_mass={m.body_mass.sum():.2f}kg")


if __name__ == "__main__":
    for v in VARIANTS:
        print(describe(v))
