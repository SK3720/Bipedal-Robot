"""Build dist/hil_pi_bundle.zip -- everything the Raspberry Pi needs (runs/ and hil/ are
untracked in git, so `git clone` alone does not work).  Read-only w.r.t. the repo.

  python -m hil.make_pi_bundle
  scp dist/hil_pi_bundle.zip sk@10.0.0.71:~/
"""
from __future__ import annotations

import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hil import spec                                                     # noqa: E402

REPO = spec.REPO
OUT = os.path.join(REPO, "dist", "hil_pi_bundle.zip")
SKIP_DIRS = {"__pycache__", "logs"}
SKIP_FILES = {"mpu_calib.json"}          # calibration is per machine -- never ship the PC's


def add(z, path, arc):
    z.write(path, arc.replace("\\", "/"))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(os.path.join(REPO, "hil")):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f in SKIP_FILES or f.endswith(".pyc"):
                    continue
                p = os.path.join(root, f)
                add(z, p, os.path.relpath(p, REPO)); n += 1
        add(z, spec.MODEL_XML, os.path.relpath(spec.MODEL_XML, REPO)); n += 1
        md = os.path.join(REPO, "robot", "meshes")
        for f in sorted(os.listdir(md)):
            add(z, os.path.join(md, f), os.path.join("robot", "meshes", f)); n += 1
        for f in (spec.HIL_POLICY_FILE, spec.CONSTS_FILE):
            if not os.path.exists(f):
                sys.exit(f"missing {f} -- run  python -m hil.export_pi_bundle / hil.export_consts")
            add(z, f, os.path.relpath(f, REPO)); n += 1
    print(f"wrote {OUT}  ({n} files, {os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
