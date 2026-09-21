"""OpenGL-free live view of the sim robot (tkinter stick figure) -- for machines where the MuJoCo viewer
cannot open a window (e.g. a GPU driver in an error state: "WGL: The driver does not appear to support OpenGL").

Draws two orthographic views of the MuJoCo state: SIDE (robot walks to the RIGHT) and FRONT (robot's LEFT = screen
right), the skeleton (parent -> child body links), the chest's UP axis (green) and FORWARD axis (magenta), foot markers
that turn red on contact, and a text panel.  Pure standard library (tkinter) -- no OpenGL, matplotlib or OpenCV.
"""
from __future__ import annotations

import numpy as np


class StickViz:
    def __init__(self, model, title="hil.live_sim  (real IMU -> sim)", width=980, height=560):
        import tkinter as tk
        self.tk = tk
        self.m = model
        self.w, self.h = width, height
        self.root = tk.Tk()
        self.root.title(title)
        self.cv = tk.Canvas(self.root, width=width, height=height, bg="#0f1318", highlightthickness=0)
        self.cv.pack()
        self._alive = True
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        names = [model.body(i).name for i in range(model.nbody)]
        self.idx = {n: i for i, n in enumerate(names)}
        self.edges = [(int(model.body_parentid[i]), i) for i in range(1, model.nbody) if model.body_parentid[i] > 0]
        self.chest = self.idx["Chest"]
        self.color = {}
        for i, n in enumerate(names):
            self.color[i] = ("#4aa3ff" if n.startswith("L_") else "#ffa04a" if n.startswith("R_") else "#c9d1d9")
        self.foot = {"L": self.idx.get("L_foot"), "R": self.idx.get("R_foot")}
        import mujoco
        planes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        self.floor_z = float(model.geom_pos[planes[0]][2]) if planes else 0.0      # this model's floor is at z = 1.0
        self.scale = (height * 0.62) / 0.5                                # px per metre: the robot is ~0.4 m tall
        self.ground_y = int(height * 0.84)

    def _close(self):
        self._alive = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def alive(self):
        return self._alive

    def pump(self):
        if self._alive:
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception:
                self._alive = False

    def _proj(self, p, view, ref):
        """world point -> pixel.  side: right = world -Y (walk direction);  front: right = world +X."""
        s = self.scale
        if view == "side":
            return (self.w * 0.25 + s * (-(p[1] - ref[1])), self.ground_y - s * (p[2] - self.floor_z))
        return (self.w * 0.75 + s * (p[0] - ref[0]), self.ground_y - s * (p[2] - self.floor_z))

    def draw(self, data, lines, contact_L=False, contact_R=False):
        if not self._alive:
            return
        cv = self.cv
        cv.delete("all")
        ref = data.xpos[self.chest].copy()
        for view, cx, label in (("side", self.w * 0.25, "SIDE  (walks ->)"), ("front", self.w * 0.75, "FRONT  (robot left = right)")):
            cv.create_line(cx - self.w * 0.22, self.ground_y, cx + self.w * 0.22, self.ground_y, fill="#39424e", width=2)
            cv.create_text(cx, self.h - 18, text=label, fill="#7d8896", font=("Segoe UI", 10))
            for a, b in self.edges:
                pa, pb = self._proj(data.xpos[a], view, ref), self._proj(data.xpos[b], view, ref)
                cv.create_line(*pa, *pb, fill=self.color[b], width=6, capstyle="round")
            for i in range(1, self.m.nbody):
                x, y = self._proj(data.xpos[i], view, ref)
                cv.create_oval(x - 4, y - 4, x + 4, y + 4, fill=self.color[i], outline="")
            head = self.idx.get("Head-v6")
            if head is not None:
                x, y = self._proj(data.xpos[head], view, ref)
                cv.create_oval(x - 16, y - 16, x + 16, y + 16, outline="#c9d1d9", width=2)
            R = data.xmat[self.chest].reshape(3, 3)
            o = data.xpos[self.chest]
            for axis, col, ln in ((R[:, 1], "#3ddc84", 0.14), (R[:, 2], "#ff4fd8", 0.09)):     # up, forward
                pa, pb = self._proj(o, view, ref), self._proj(o + ln * axis, view, ref)
                cv.create_line(*pa, *pb, fill=col, width=3, arrow="last")
            for side, con in (("L", contact_L), ("R", contact_R)):
                bi = self.foot[side]
                if bi is not None:
                    x, y = self._proj(data.xpos[bi], view, ref)
                    cv.create_oval(x - 11, y - 11, x + 11, y + 11, outline="#ff5555" if con else "#59636e",
                                   fill="#ff5555" if con else "", width=2)
        for k, t in enumerate(lines):
            cv.create_text(14, 14 + 18 * k, anchor="nw", text=t, fill="#d7dde5", font=("Consolas", 11))
        cv.create_text(self.w - 12, 12, anchor="ne", fill="#3ddc84", text="up axis", font=("Segoe UI", 9))
        cv.create_text(self.w - 12, 28, anchor="ne", fill="#ff4fd8", text="forward axis", font=("Segoe UI", 9))
        self.pump()

    def snapshot(self, path):
        """Save what is on screen to a PNG (PIL.ImageGrab)."""
        from PIL import ImageGrab
        self.pump()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        ImageGrab.grab(bbox=(x, y, x + self.w, y + self.h)).save(path)
