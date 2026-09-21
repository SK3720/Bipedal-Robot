"""Hardware-free test of hil/pca9685.py + hil/servo_test.py (no I2C, no servo).  python -m hil.selftest_servo"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hil import spec, pca9685 as P, servo_test as S                     # noqa: E402


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return bool(ok)


def main():
    ok = True
    fake = P.FakePCA()
    pca = P.PCA9685(smbus=fake)
    ok &= check("prescale for 50 Hz = 121 (datasheet)", fake.regs[P.PRESCALE] == 121, f"{fake.regs[P.PRESCALE]}")
    pca.set_pulse_us(3, 1500.0)
    ok &= check("1500 us -> 307 counts -> ~1499.5 us", abs(fake.pulse_us(3) - 1500) < 6, f"{fake.pulse_us(3):.1f} us")
    pca.release(3)
    ok &= check("release sets full-off bit", fake.pulse_us(3) is None)

    cfg = P.template_map()
    bank = P.ServoBank(P.PCA9685(smbus=P.FakePCA()), cfg)
    ok &= check("template map drives nothing", len(bank.j) == 0)

    j = "L_leg_L_knee"; i = spec.JOINTS_MJ.index(j)
    cfg["max_slew_rad_s"] = 1.0
    cfg["joints"][j] = dict(channel=5, direction=-1, zero_us=1500.0, us_per_rad=400.0, min_us=1000.0, max_us=2000.0)
    fk = P.FakePCA()
    bank = P.ServoBank(P.PCA9685(smbus=fk), cfg)
    ok &= check("only the filled joint is driven", list(bank.j) == [i])
    u = np.zeros(15); u[i] = 0.0
    bank.write(u)
    ok &= check("zero angle -> zero_us", abs(fk.pulse_us(5) - 1500) < 6, f"{fk.pulse_us(5):.1f}")
    u[i] = 1.0
    bank.write(u)                                                       # 1 rad asked, slew 1 rad/s * 5 ms
    ok &= check("slew-limited step (5 mrad)", abs(bank.state[i] - 0.005) < 1e-9, f"{bank.state[i]:.4f} rad")
    ok &= check("direction -1 respected", fk.pulse_us(5) < 1500)
    for _ in range(400):
        bank.write(u)
    ok &= check("pulse clamp at min_us (1 rad*400=400 us -> 1100 within, so within range)", fk.pulse_us(5) > 1000 - 6, f"{fk.pulse_us(5):.0f}")
    cfg["joints"][j]["us_per_rad"] = 2000.0
    bank2 = P.ServoBank(P.PCA9685(smbus=(f2 := P.FakePCA())), cfg)
    bank2.write(u); [bank2.write(u) for _ in range(400)]
    ok &= check("hard clamp holds", f2.pulse_us(5) >= 1000 - 6, f"{f2.pulse_us(5):.0f} us")
    bank2.release()
    ok &= check("release() frees channel", f2.pulse_us(5) is None)

    d = tempfile.mkdtemp(); m = os.path.join(d, "m.json")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc0 = S.main(["--write-template", "--map", m])
        rc1 = S.main(["--joint", j, "--map", m, "--enable"])                # missing values -> nothing sent
        rc2 = S.main(["--joint", j, "--map", m, "--channel", "5", "--direction", "1", "--zero-us", "1500",
                      "--us-per-rad", "400", "--min-us", "1000", "--max-us", "2000", "--dry"])
        rc3 = S.main(["--joint", j, "--map", m, "--channel", "5", "--direction", "1", "--zero-us", "1500",
                      "--us-per-rad", "400", "--min-us", "1000", "--max-us", "2000", "--amp", "0.9", "--dry"])
    ok &= check("template written", rc0 == 0 and os.path.exists(m))
    ok &= check("missing map values -> refuses, sends nothing", rc1 == 2 and "missing" in buf.getvalue())
    ok &= check("dry-run OK without --enable", rc2 == 0 and "DRY-RUN" in buf.getvalue())
    ok &= check("amp > 0.5 rad refused", rc3 == 2)
    seq = S.plan(0.15, 0.5, 0.005)
    ok &= check("plan: 0 -> +a -> -a -> 0, slew respected",
                seq[0] == 0 and abs(seq[-1]) < 1e-12 and np.abs(np.diff(seq)).max() <= 0.5 * 0.005 + 1e-9,
                f"{len(seq)} steps, max step {np.abs(np.diff(seq)).max():.4f} rad")
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
