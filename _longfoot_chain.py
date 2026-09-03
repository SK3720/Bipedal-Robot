"""Step 6 attempt on the long-foot variant: full
push -> pre-shift -> swing -> plant -> double support -> LQR settle chain,
reusing push_recovery_step.run() verbatim, only swapping in a fore-aft-
lengthened foot model.  Sweeps push magnitude; reports where the chain breaks.

Run:  python _longfoot_chain.py --sweep 150,200,240,280
      python _longfoot_chain.py --push 240 --slow --sz 1.8
"""
import argparse, sys, time
import mujoco, mujoco.viewer, numpy as np
from pathlib import Path
from push_recovery_step import Cfg, run, print_result

BASE = Path("robot/robot.xml").read_text()


def build(sz):
    s = BASE
    for L in ("L", "R"):
        s = s.replace(
            f'mesh name="{L}_foot" content_type="model/stl" file="meshes/{L}_foot.stl" scale="0.001 0.001 0.001"',
            f'mesh name="{L}_foot" content_type="model/stl" file="meshes/{L}_foot.stl" scale="0.001 0.001 {0.001*sz}"')
    p = Path("robot/_lfc.xml"); p.write_text(s)
    try:
        return mujoco.MjModel.from_xml_path(str(p))
    finally:
        p.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--push", type=float, default=220.0)
    p.add_argument("--sweep", default=None)
    p.add_argument("--sz", type=float, default=1.8)
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    a = p.parse_args(argv)

    model = build(a.sz); data = mujoco.MjData(model)
    cfg = Cfg(swing=a.swing)
    print(f"[long-foot chain]  sz={a.sz}  (fore-aft foot ~{86*a.sz:.0f} mm)")

    if a.sweep:
        rs = []
        for pn in (float(x) for x in a.sweep.split(",")):
            r = run(model, data, cfg, pn, verbose=True)
            print_result(r)
            rs.append(r)
        print("\n" + "=" * 70)
        for r in rs:
            print(f"  {r.push_n:5.0f} N  trig={r.triggered} plant={r.planted} "
                  f"sep@plant {r.foot_sep_fwd_at_plant_mm:5.0f}mm  endTilt {r.end_up_tilt:4.1f}  {r.outcome}")
        return

    if a.headless:
        print_result(run(model, data, cfg, a.push, verbose=True), timeline=True)
        return

    with mujoco.viewer.launch_passive(model, data) as vw:
        vw.cam.lookat[:] = [0.0, -0.1, 1.05]; vw.cam.distance = 2.0
        vw.cam.azimuth = 90; vw.cam.elevation = -8
        r = run(model, data, cfg, a.push, viewer=vw, slow=a.slow)
        print_result(r, timeline=True)
        while vw.is_running():
            vw.sync(); time.sleep(0.02 if a.slow else 0.003)


if __name__ == "__main__":
    main(sys.argv[1:])
