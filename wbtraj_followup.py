"""Follow-up to wbtraj_opt.py:
  (1) re-run heavier hands 2x cleanly (the --all run hit CMA flat-fitness early-stop),
  (2) push each hand mass toward a ROBUST >=60 mm step with a step-size-heavy
      objective + a wider push grid, to settle answer A/B/C/D.
Reuses wbtraj_opt unchanged.  robot.xml untouched.
"""
from __future__ import annotations
import functools, json, os, sys, time
print = functools.partial(print, flush=True)
import numpy as np
import wbtraj_opt as W
from wbtraj_opt import JOINTS_B, JOINTS_A, TrajCfg, _eval, _winit, _vec_to_tc, analyse, make_hand_model, _BUILD, FINE


def score_bigstep(r):
    if r is None:
        return 0.0
    rec = 0.0
    if not r["fell"] and r["triggered"]:
        rec = 0.20
        if r["n_planted"]:
            rec += 0.15
            rec *= (0.7 + 0.3 * max(0.0, 1.0 - abs(r["touchdown"].get("sole", 90)) / 22.0))
        if r["end_ds"]:
            rec += 0.10
        rec += 0.25 * max(0.0, 1.0 - r["end_up"] / 15.0)
        rec += 0.15 * max(0.0, 1.0 - abs(r["end_side"]) / 15.0)
        rec += 0.15 * max(0.0, 1.0 - r["end_speed"] / 0.28)
    if r["success"]:
        rec = 1.0
    sep = r.get("step_sep_mm", 0.0)
    sep_f = float(np.clip((sep - 25.0) / (65.0 - 25.0), 0.0, 1.35))
    return float(rec * (0.15 + 0.85 * sep_f))


def optimise_bigstep(joint_list, model_path, pool, grid, gens, popsize, seed, x0=None):
    import cma
    from wbtraj_opt import _pack_layout
    layout, _ = _pack_layout(joint_list)
    lo = np.array([p[1] for p in layout]); hi = np.array([p[2] for p in layout])
    x00 = np.array([p[3] for p in layout]) if x0 is None else np.asarray(x0, float)
    x0n = (np.clip(x00, lo, hi) - lo) / (hi - lo)
    es = cma.CMAEvolutionStrategy(list(x0n), 0.32, {
        "bounds": [0, 1], "popsize": popsize, "seed": seed, "maxiter": gens,
        "verbose": -9, "tolflatfitness": gens, "tolstagnation": gens,
        "tolfun": 1e-4, "tolx": 1e-5})
    best = {"fit": 1e9, "x": None}
    t0 = time.time(); g = 0
    while not es.stop():
        g += 1
        sols = es.ask()
        tasks = []
        for sn in sols:
            xr = lo + np.asarray(sn) * (hi - lo)
            for pn in grid:
                tasks.append((xr, joint_list, TrajCfg(), pn, model_path))
        res = pool.map(_eval, tasks)
        ng = len(grid); fits = []
        for i, sn in enumerate(sols):
            rs = res[i * ng:(i + 1) * ng]
            sc = np.array([score_bigstep(r) for r in rs])
            fit = -(sc.mean()) + 0.15 * sc.std()
            fits.append(fit)
            if fit < best["fit"]:
                best = {"fit": float(fit), "x": (lo + np.asarray(sn) * (hi - lo)).tolist(),
                        "grid_succ": float(np.mean([r["success"] for r in rs])),
                        "grid_seps_ok": [round(r["step_sep_mm"], 1) for r in rs if r["success"]],
                        "grid_seps": [round(r["step_sep_mm"], 1) for r in rs]}
        es.tell(sols, fits)
        if g % 4 == 0 or g == 1:
            print(f"    gen {g:3d} fit {best['fit']:+.3f} gSucc {best.get('grid_succ',0):.2f} "
                  f"sepsOK {best.get('grid_seps_ok',[])} ({time.time()-t0:.0f}s)")
    return best


def main():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    GRID = (124.0, 128.0, 132.0, 136.0, 140.0)
    OUT = {}
    outp = "results/wb_followup.json"
    os.makedirs("results", exist_ok=True)

    plan = [("1.0x", 1.0, 30), ("2.0x", 2.0, 30), ("3.0x", 3.0, 28), ("5.0x", 5.0, 28)]
    prev = json.load(open("results/wb_stages.json"))
    x0 = prev["stageA"]["best"]["x"][:15] + [0.0] * (4 * (len(JOINTS_B) - len(JOINTS_A)))

    for tag, mult, gens in plan:
        mp_path = f"robot/_exp_hands_{mult:g}x.xml" if mult != 1.0 else "robot/robot.xml"
        if mult != 1.0:
            make_hand_model(mult, mp_path)
        _BUILD.pop(mp_path, None)
        print(f"\n=== hands {tag}  (big-step objective, grid {GRID}) ===")
        pool = ctx.Pool(10, initializer=_winit, initargs=(mp_path,))
        try:
            b = optimise_bigstep(JOINTS_B, mp_path, pool, GRID, gens, 20, seed=int(mult * 7) + 3, x0=x0)
            an = analyse(f"hands{tag}_big", b["x"], JOINTS_B, TrajCfg(), mp_path, pool)
        finally:
            pool.close(); pool.join()
        OUT[tag] = dict(best=b, analysis=an, model_path=mp_path)
        json.dump(OUT, open(outp, "w"), indent=2, default=float)
        ss = an["sep_ok_list"]
        print(f"  -> fine success {an['success_rate']:.2f}  band {an['band']['width_N']}N  "
              f"bestSepRec {an['best_sep_recovered']}  medSepOK {an['median_sep_ok']}  "
              f"torsoPitch {an['peak_torso_pitch_mean']}  sat {an['sat_frac_mean']}")
        n60 = sum(1 for pp in an["per_push"] if pp["success"] and pp["sep"] >= 58)
        print(f"     pushes recovered with sep>=58mm: {n60}/{an['n']}   "
              f"(all recovered seps: {sorted(ss)})")

    print("\n" + "=" * 70)
    print(f"{'hands':<8}{'fineSucc':>10}{'band_N':>8}{'bestSepRec':>12}{'medSepOK':>10}"
          f"{'>=58mm&rec':>12}{'torsoP':>8}{'sat':>7}")
    for tag in OUT:
        an = OUT[tag]["analysis"]
        n60 = sum(1 for pp in an["per_push"] if pp["success"] and pp["sep"] >= 58)
        print(f"{tag:<8}{an['success_rate']:>10.2f}{an['band']['width_N']:>8.0f}"
              f"{str(an['best_sep_recovered']):>12}{str(an['median_sep_ok']):>10}"
              f"{n60:>8}/{an['n']:<3}{an['peak_torso_pitch_mean']:>8}{an['sat_frac_mean']:>7.2f}")
    json.dump(OUT, open(outp, "w"), indent=2, default=float)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
