"""CMA-ES optimisation of frontal_lqr_step.py's parameters over a multi-push grid.

The analytical assessment's recommended step BEFORE any RL: does a genuinely
robust region exist in the FIXED parameterisation of frontal_lqr_step.py, or does
retuning only move a point solution?  And if a robust region exists, does it
require collapsing the step to a tiny shuffle?

Method
------
CMA-ES over 14 controller parameters.  Two independent optimisation runs:
  A "pure recovery"  - objective = mean recovery score over the push grid,
                       NO step-size term (let it collapse to a shuffle if that
                       is what is robust).
  B "recovery + step" - same, but a successful trial is only fully credited if
                        the forward foot separation at plant is >= ~50 mm.

Then, on each optimum:
  * fine-grained success sweep (push 120-150 N, 2 N steps)
  * step-size distribution on successful trials
  * failure-mode breakdown
  * one-at-a-time parameter sensitivity (+/-10 %, +/-20 %) -> knife-edge check
  * push-duration (impulse) robustness on the final params
  * comparison vs. the current hand-tuned Cfg() baseline

Does not modify robot.xml, frontal_lqr_step.py, or any golden script.  Controller
builders are memoised per worker process (they depend only on the fixed model).

    python cma_optimize_frontal_step.py                 # full experiment
    python cma_optimize_frontal_step.py --quick         # small budget smoke test
    python cma_optimize_frontal_step.py --analyse-only results/cma_frontal.json
"""
from __future__ import annotations

import argparse
import functools
print = functools.partial(print, flush=True)
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

# ----------------------------------------------------------------------------
# parameter space:  name, (lo, hi), x0 (current hand-tuned value)
# ----------------------------------------------------------------------------
PARAMS = [
    ("x_ss_mm",          38.0,  72.0,  50.0),
    ("front_pole_re",   -13.0,  -3.0,  -5.0),
    ("front_pole_im",     1.5,   7.0,   3.0),
    ("shift_ms",         120.0, 280.0, 200.0),
    ("shift_min_ms",     100.0, 190.0, 140.0),
    ("swing_start_ms",    90.0, 240.0, 150.0),
    ("swing_pr_gate",     0.10,  0.90,  0.40),
    ("swing_ms",          95.0, 220.0, 165.0),
    ("swing_hip_rad",     0.28,  0.78,  0.60),
    ("swing_knee_rad",    0.14,  0.46,  0.36),
    ("hip_retract_frac",  0.04,  0.46,  0.24),
    ("descend_ms",        40.0, 125.0,  80.0),
    ("x_transfer_mm",      8.0,  46.0,  24.0),
    ("transfer_ms",      110.0, 320.0, 220.0),
    ("imp_amp",           0.10,  0.30,  0.22),
]
PNAMES = [p[0] for p in PARAMS]
PLO = np.array([p[1] for p in PARAMS])
PHI = np.array([p[2] for p in PARAMS])
PX0 = np.array([p[3] for p in PARAMS])

# CMA loop grid (kept small for speed); final analysis uses a finer sweep
CMA_PUSH_GRID = (126.0, 130.0, 134.0, 138.0, 142.0)
FINE_PUSH_GRID = tuple(float(x) for x in range(120, 151, 2))
STEP_TARGET_MM = 50.0     # run B: "real step" threshold

_worker_ready = [False]


# ---------------------------------------------------------------- worker setup
def _worker_init():
    """memoise the three controller builders inside each pool process - they
    depend only on robot.xml, which never changes, so building them once per
    process (instead of once per run(), incl. a 340-step SS ramp) is a ~2x
    speed-up with byte-identical behaviour."""
    import frontal_lqr_step as F
    cache = {}
    o_sl, o_ab, o_ss = F.StandingLQR, F._LQRAbout, F._SingleSupportLQR

    def SL(m, d, verbose=False):
        if "sl" not in cache:
            cache["sl"] = o_sl(m, d, verbose=False)
        return cache["sl"]

    def AB(m, d, pose, tag="ab", verbose=False):
        if "ab" not in cache:
            cache["ab"] = o_ab(m, d, pose, tag="ab", verbose=False)
        return cache["ab"]

    def SS(m, d, ab, scfg, verbose=False):
        if "ss" not in cache:
            cache["ss"] = o_ss(m, d, ab, scfg, verbose=False)
        return cache["ss"]

    F.StandingLQR = SL
    F._LQRAbout = AB
    F._SingleSupportLQR = SS
    _worker_ready[0] = True


def _vec_to_cfg(x):
    from frontal_lqr_step import Cfg
    x = np.clip(np.asarray(x, float), PLO, PHI)
    kw = {n: float(v) for n, v in zip(PNAMES, x)}
    # keep the ordering sane: swing can't start before the min-shift gate
    kw["swing_start_ms"] = float(max(kw["swing_start_ms"], kw["shift_min_ms"] + 5))
    return Cfg(**kw)


def _classify(r):
    if r is None:
        return "error"
    if not r["triggered"]:
        return "no_trigger"
    if r["success"]:
        return "success"
    if r["fell"]:
        # use end pose to guess direction
        es, eu = r["end_side"], r["end_up"]
        # peak_fwd tiny + big end_up along fwd -> forward/back; big |side| -> lateral
        if abs(es) > 25:
            return "fell_lateral"
        # fwd vs back: end pitch sign not in dict; use n_planted + final_sep
        return "fell_sagittal"
    if r.get("n_planted", 0) == 0:
        return "no_plant"
    return "planted_but_unstable"


def _score(r, step_gate=False):
    if r is None:
        return 0.0
    if r["fell"]:
        return 0.0
    if r["success"]:
        if not step_gate:
            return 1.0
        sep = _step_sep(r)
        # full credit at >= STEP_TARGET_MM, 0.35 floor for a clean shuffle
        return float(np.clip(0.35 + 0.65 * (sep - 15.0) / (STEP_TARGET_MM - 15.0),
                             0.35, 1.0))
    # partial credit: reward getting deep into a clean maneuver even if it later
    # falls (peak_up is ~55 whenever it falls, so use plant quality instead)
    s = 0.05
    if r["triggered"]:
        s += 0.05
    if r.get("n_planted", 0) >= 1:
        s += 0.20
        st_side = abs(r.get("plant_side", r.get("end_side", 90.0)))
        st_sole = abs(r.get("plant_sole", 90.0))
        st_sep = r.get("step_sep", 0.0)
        s += 0.15 * max(0.0, 1.0 - st_side / 20.0)      # level at plant
        s += 0.10 * max(0.0, 1.0 - st_sole / 25.0)      # flat sole at plant
        s += 0.10 * max(0.0, min(1.0, st_sep / 40.0))   # foot forward, not behind
    if r["end_ds"]:
        s += 0.10
    s += 0.10 * max(0.0, 1.0 - r["end_up"] / 30.0)
    s += 0.05 * max(0.0, 1.0 - abs(r["end_side"]) / 30.0)
    return float(min(s, 0.90))


def _step_sep(r):
    if r and r.get("steps"):
        return float(max(s["sep_mm"] for s in r["steps"]))
    return float(r.get("peak_fwd", 0.0)) if r else 0.0


# -------------------------------------------------------------- single eval
def _eval_one(task):
    """task = (x_vector, push_n, push_dur_steps). returns a compact result dict."""
    x, push_n, dur = task
    import frontal_lqr_step as F
    if not _worker_ready[0]:
        _worker_init()
    old_dur = F.PUSH_DURATION_STEPS
    try:
        if dur is not None:
            F.PUSH_DURATION_STEPS = int(dur)
        cfg = _vec_to_cfg(x)
        r = F.run(push_n, cfg, verbose=False)
        out = {k: (float(r[k]) if isinstance(r[k], (int, float, np.floating)) else r[k])
               for k in ("success", "fell", "triggered", "n_planted", "peak_up",
                         "peak_fwd", "final_sep_mm", "end_up", "end_side",
                         "end_speed", "end_ds")}
        out["step_sep"] = _step_sep(r)
        if r.get("steps"):
            s0 = max(r["steps"], key=lambda z: z["sep_mm"])
            out["plant_side"] = float(s0.get("side", 90.0))
            out["plant_sole"] = float(s0.get("sole", 90.0))
        out["cls"] = _classify(r)
        return out
    except Exception as e:  # pathological params
        return {"success": False, "fell": True, "triggered": False, "err": repr(e),
                "cls": "error", "step_sep": 0.0, "peak_up": 99.0, "end_up": 99.0,
                "end_side": 99.0, "end_speed": 9.0, "end_ds": False, "n_planted": 0,
                "peak_fwd": 0.0, "final_sep_mm": 0.0}
    finally:
        F.PUSH_DURATION_STEPS = old_dur


# -------------------------------------------------------------- CMA-ES driver
def run_cma(tag, step_gate, pool, gens, popsize, seed=1, quick=False):
    import cma
    grid = CMA_PUSH_GRID if not quick else (128.0, 138.0)
    x0n = (PX0 - PLO) / (PHI - PLO)          # normalise to [0,1]
    es = cma.CMAEvolutionStrategy(
        list(x0n), 0.28,
        {"bounds": [0, 1], "popsize": popsize, "seed": seed,
         "maxiter": gens, "verbose": -9})
    hist = []
    best = {"fit": 1e9, "x": None, "scores": None}
    t0 = time.time()
    gen = 0
    while not es.stop():
        gen += 1
        sols = es.ask()
        tasks = []
        for si, sn in enumerate(sols):
            xr = PLO + np.asarray(sn) * (PHI - PLO)
            for pn in grid:
                tasks.append((xr, pn, None))
        results = pool.map(_eval_one, tasks)
        # regroup
        per = len(grid)
        fits = []
        for si in range(len(sols)):
            rs = results[si * per:(si + 1) * per]
            sc = np.array([_score(r, step_gate) for r in rs])
            fit = -(sc.mean()) + 0.15 * sc.std()      # prefer high + uniform
            fits.append(fit)
            if fit < best["fit"]:
                xr = PLO + np.asarray(sols[si]) * (PHI - PLO)
                best = {"fit": float(fit), "x": xr.tolist(),
                        "scores": sc.tolist(), "succ": float(np.mean([r["success"] for r in rs])),
                        "med_sep": float(np.median([r["step_sep"] for r in rs if r["success"]] or [0.0]))}
        es.tell(sols, fits)
        m_succ = np.mean([r["success"] for r in results])
        hist.append({"gen": gen, "best_fit": best["fit"], "mean_succ_gen": float(m_succ),
                     "best_succ": best.get("succ", 0), "best_med_sep": best.get("med_sep", 0)})
        if gen % 5 == 0 or gen == 1:
            print(f"  [{tag}] gen {gen:3d}  best_fit {best['fit']:+.3f}  "
                  f"best gridSucc {best.get('succ',0):.2f}  medSep {best.get('med_sep',0):.0f}mm  "
                  f"({time.time()-t0:.0f}s)")
    print(f"  [{tag}] done: {gen} gens, {time.time()-t0:.0f}s")
    return {"tag": tag, "step_gate": step_gate, "best_x": best["x"],
            "best_fit": best["fit"], "cma_grid_succ": best.get("succ", 0),
            "cma_grid_med_sep": best.get("med_sep", 0),
            "params": {n: v for n, v in zip(PNAMES, best["x"])}, "history": hist}


# -------------------------------------------------------------- analysis
def analyse(name, x, pool):
    """fine sweep + step distribution + failure modes for a parameter vector."""
    tasks = [(x, pn, None) for pn in FINE_PUSH_GRID]
    res = pool.map(_eval_one, tasks)
    succ = [r["success"] for r in res]
    seps = [r["step_sep"] for r, s in zip(res, succ) if s]
    modes = {}
    per_push = []
    for pn, r in zip(FINE_PUSH_GRID, res):
        modes[r["cls"]] = modes.get(r["cls"], 0) + 1
        per_push.append({"push": pn, "success": bool(r["success"]), "cls": r["cls"],
                         "step_sep": round(r["step_sep"], 1), "peak_up": round(r["peak_up"], 1),
                         "end_up": round(r["end_up"], 1), "end_side": round(r["end_side"], 1)})
    # push-duration robustness (impulse): grid push x {5,6,7} duration steps
    dtasks = [(x, pn, dd) for pn in (126.0, 132.0, 138.0) for dd in (4, 5, 6, 7)]
    dres = pool.map(_eval_one, dtasks)
    dur_grid = []
    i = 0
    for pn in (126.0, 132.0, 138.0):
        for dd in (4, 5, 6, 7):
            dur_grid.append({"push": pn, "dur": dd, "success": bool(dres[i]["success"]),
                             "step_sep": round(dres[i]["step_sep"], 1)})
            i += 1
    out = {
        "name": name, "x": list(map(float, x)),
        "params": {n: round(float(v), 4) for n, v in zip(PNAMES, x)},
        "n_push": len(FINE_PUSH_GRID),
        "overall_success_rate": round(float(np.mean(succ)), 3),
        "success_pushes": [round(p, 0) for p, s in zip(FINE_PUSH_GRID, succ) if s],
        "fail_pushes": [round(p, 0) for p, s in zip(FINE_PUSH_GRID, succ) if not s],
        "contiguous_success_band": _contig(FINE_PUSH_GRID, succ),
        "step_sep_on_success": {
            "n": len(seps),
            "median": round(float(np.median(seps)), 1) if seps else None,
            "mean": round(float(np.mean(seps)), 1) if seps else None,
            "min": round(float(np.min(seps)), 1) if seps else None,
            "max": round(float(np.max(seps)), 1) if seps else None,
            "p25": round(float(np.percentile(seps, 25)), 1) if seps else None,
            "p75": round(float(np.percentile(seps, 75)), 1) if seps else None,
            "all": [round(s, 1) for s in seps],
        },
        "failure_modes": modes,
        "per_push": per_push,
        "duration_robustness": dur_grid,
        "dur_success_rate": round(float(np.mean([d["success"] for d in dur_grid])), 3),
    }
    return out


def _contig(grid, succ):
    best_lo = best_hi = None
    best_len = 0
    i = 0
    n = len(grid)
    while i < n:
        if succ[i]:
            j = i
            while j < n and succ[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_lo, best_hi = grid[i], grid[j - 1]
            i = j
        else:
            i += 1
    return {"lo": best_lo, "hi": best_hi, "width_N": (best_hi - best_lo) if best_lo is not None else 0}


def sensitivity(x, pool, base_rate):
    """one-at-a-time +/-10 %, +/-20 % on each param -> success-rate delta."""
    rows = []
    for pi, name in enumerate(PNAMES):
        deltas = {}
        for frac in (-0.20, -0.10, 0.10, 0.20):
            xx = np.array(x, float)
            span = PHI[pi] - PLO[pi]
            xx[pi] = np.clip(xx[pi] + frac * abs(xx[pi] if xx[pi] != 0 else span), PLO[pi], PHI[pi])
            tasks = [(xx, pn, None) for pn in FINE_PUSH_GRID]
            r = pool.map(_eval_one, tasks)
            deltas[f"{int(frac*100):+d}%"] = round(float(np.mean([q["success"] for q in r])) - base_rate, 3)
        worst = min(deltas.values())
        rows.append({"param": name, "value": round(float(x[pi]), 4), "deltas": deltas,
                     "worst_drop": worst})
    rows.sort(key=lambda z: z["worst_drop"])
    return rows


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gens", type=int, default=45)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--analyse-only", default=None)
    ap.add_argument("--out", default="results/cma_frontal.json")
    a = ap.parse_args(argv)
    os.makedirs("results", exist_ok=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=_worker_init)
    try:
        if a.analyse_only:
            data = json.load(open(a.analyse_only))
            xb = data["runB"]["best_x"] if "runB" in data else data["runA"]["best_x"]
            print(analyse("reanalyse", xb, pool))
            return

        gens = 6 if a.quick else a.gens
        pop = 8 if a.quick else a.popsize

        print("=== baseline (current hand-tuned Cfg) ===")
        base = analyse("baseline_handtuned", list(PX0), pool)
        print(f"  success {base['overall_success_rate']:.2f}  "
              f"band {base['contiguous_success_band']}  "
              f"medSep {base['step_sep_on_success']['median']}mm")

        print("\n=== CMA-ES run A: pure recovery (no step-size term) ===")
        runA = run_cma("A", step_gate=False, pool=pool, gens=gens, popsize=pop, seed=1, quick=a.quick)
        anA = analyse("cma_A_pure_recovery", runA["best_x"], pool)
        print(f"  -> success {anA['overall_success_rate']:.2f}  band {anA['contiguous_success_band']}  "
              f"medSep {anA['step_sep_on_success']['median']}mm")

        print("\n=== CMA-ES run B: recovery + real step (>=50 mm) ===")
        runB = run_cma("B", step_gate=True, pool=pool, gens=gens, popsize=pop, seed=2, quick=a.quick)
        anB = analyse("cma_B_recovery_and_step", runB["best_x"], pool)
        print(f"  -> success {anB['overall_success_rate']:.2f}  band {anB['contiguous_success_band']}  "
              f"medSep {anB['step_sep_on_success']['median']}mm")

        print("\n=== sensitivity (run A optimum) ===")
        sensA = sensitivity(runA["best_x"], pool, anA["overall_success_rate"])
        for row in sensA[:6]:
            print(f"  {row['param']:<18} worst drop {row['worst_drop']:+.2f}  {row['deltas']}")

        out = {"baseline": base, "runA": runA, "runA_analysis": anA,
               "runB": runB, "runB_analysis": anB, "sensitivity_A": sensA,
               "param_space": [{"name": n, "lo": lo, "hi": hi, "x0": x0}
                               for n, lo, hi, x0 in PARAMS]}
        json.dump(out, open(a.out, "w"), indent=2, default=float)
        print(f"\nsaved -> {a.out}")
        _print_report(out)
    finally:
        pool.close()
        pool.join()


def _print_report(o):
    b, A, B = o["baseline"], o["runA_analysis"], o["runB_analysis"]
    print("\n" + "=" * 74)
    print("REPORT")
    print("=" * 74)
    for nm, d in (("baseline (hand-tuned)", b), ("CMA-A pure recovery", A),
                  ("CMA-B recovery+step", B)):
        ss = d["step_sep_on_success"]
        print(f"\n{nm}")
        print(f"  success rate (push 120-150 N, 2 N grid): {d['overall_success_rate']:.2f}  "
              f"({len(d['success_pushes'])}/{d['n_push']})")
        print(f"  contiguous success band: {d['contiguous_success_band']['lo']}-"
              f"{d['contiguous_success_band']['hi']} N "
              f"(width {d['contiguous_success_band']['width_N']:.0f} N)")
        print(f"  step separation on successes: median {ss['median']} mm  "
              f"mean {ss['mean']}  range [{ss['min']}, {ss['max']}]  (n={ss['n']})")
        print(f"  step-size distribution: {ss['all']}")
        print(f"  failure modes: {d['failure_modes']}")
        print(f"  push-duration robustness: {d['dur_success_rate']:.2f}")
    print("\nsensitivity (run A) - params most likely on a knife-edge:")
    for row in o["sensitivity_A"][:8]:
        print(f"  {row['param']:<18} v={row['value']:<8.3f} worst-drop {row['worst_drop']:+.2f}  "
              f"{row['deltas']}")


if __name__ == "__main__":
    main(sys.argv[1:])
