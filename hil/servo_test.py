"""Single-servo bench test through the PCA9685.  DRY-RUN unless --enable is given.

  python -m hil.servo_test --write-template                 # create hil/servo_map.json (all null)
  python -m hil.servo_test --joint L_leg_L_knee --dry       # prints the pulses it WOULD send, no I2C
  python -m hil.servo_test --joint L_leg_L_knee --enable    # moves that one servo (horn FREE, no load)

Motion: neutral (angle 0) -> +amp -> -amp -> neutral, each leg slew-limited, pulses hard-clamped to
[min_us, max_us], PWM released on exit / Ctrl-C / any exception.  Uses ONLY the map entry of the named
joint.  --channel/--zero-us/--us-per-rad/--min-us/--max-us/--direction override the map for a first
bench run before you have written the map.  Do the first run with the servo on its own supply, horn
off / no linkage, and start with a SMALL --amp.
NOTE: the first pulse commands angle 0 immediately -- a servo that is elsewhere will snap to it at full
speed (slew limiting only applies to the sweep).  Keep the horn free until you know where 0 is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil import spec                                                     # noqa: E402
from hil import pca9685 as P                                             # noqa: E402


def plan(amp, slew, dt):
    """angle sequence: 0 -> +amp -> -amp -> 0 at `slew` rad/s."""
    pts = [0.0, amp, -amp, 0.0]
    out = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(abs(b - a) / (slew * dt))))
        out += list(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-template", action="store_true")
    ap.add_argument("--joint", choices=spec.JOINTS_MJ)
    ap.add_argument("--amp", type=float, default=0.15, help="rad, +/- around 0 (default 0.15)")
    ap.add_argument("--slew", type=float, default=0.5, help="rad/s (default 0.5)")
    ap.add_argument("--enable", action="store_true", help="actually drive the servo (default: dry-run)")
    ap.add_argument("--dry", action="store_true")
    for k in ("channel",):
        ap.add_argument(f"--{k}", type=int)
    for k in ("zero-us", "us-per-rad", "min-us", "max-us"):
        ap.add_argument(f"--{k}", type=float)
    ap.add_argument("--direction", type=int, choices=[-1, 1])
    ap.add_argument("--map", default=P.SERVO_MAP_FILE)
    ap.add_argument("--dt", type=float, default=spec.CONTROL_DT)
    a = ap.parse_args(argv)

    if a.write_template:
        if os.path.exists(a.map):
            print(f"{a.map} exists -- not overwriting"); return 1
        with open(a.map, "w") as f:
            json.dump(P.template_map(), f, indent=2)
        print(f"wrote {a.map}  (fill in the numbers, or use the --channel/--zero-us/... overrides)"); return 0
    if not a.joint:
        ap.error("--joint required")

    cfg = json.load(open(a.map)) if os.path.exists(a.map) else P.template_map()
    e = cfg["joints"][a.joint]
    for k, v in dict(channel=a.channel, zero_us=a.zero_us, us_per_rad=a.us_per_rad,
                     min_us=a.min_us, max_us=a.max_us, direction=a.direction).items():
        if v is not None:
            e[k] = v
    missing = [k for k in ("channel", "direction", "zero_us", "us_per_rad", "min_us", "max_us") if e.get(k) is None]
    if missing:
        print(f"[servo_test] {a.joint}: missing {missing}.  Nothing will be sent.\n"
              "  Fill hil/servo_map.json or pass the overrides (see --help). No values are assumed.")
        return 2
    lo, hi = spec.CTRL_RANGE_MJ[spec.JOINTS_MJ.index(a.joint)]
    if a.amp > 0.5:
        print(f"[servo_test] --amp {a.amp} rad exceeds the 0.5 rad bench cap"); return 2

    seq = np.clip(plan(a.amp, a.slew, a.dt), lo, hi)      # sweep never leaves the joint's sim range
    cfg["joints"] = {a.joint: e}
    pulses = [float(np.clip(e["zero_us"] + e["direction"] * e["us_per_rad"] * x, e["min_us"], e["max_us"])) for x in seq]
    print(f"[servo_test] {a.joint} ch{e['channel']}  {len(seq)} steps @ {1/a.dt:.0f} Hz "
          f"({len(seq)*a.dt:.1f} s)  pulse range {min(pulses):.0f}-{max(pulses):.0f} us "
          f"(clamp {e['min_us']:.0f}-{e['max_us']:.0f})")
    if a.dry or not a.enable:
        print("  DRY-RUN (no I2C access). Add --enable to move the servo.")
        return 0
    input("  --enable: servo on its own supply, horn free / no load, hand near the power switch?  Enter to go, Ctrl-C to abort. ")
    pca = P.PCA9685(**{k: cfg["pca9685"][k] for k in ("bus", "addr", "freq_hz")})
    try:
        t = time.perf_counter()
        for ang, us in zip(seq, pulses):
            pca.set_pulse_us(e["channel"], us)
            t += a.dt
            while time.perf_counter() < t:
                time.sleep(0)
        time.sleep(0.3)
    finally:
        pca.release(e["channel"])
        pca.close()
        print("  PWM released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
