"""Round 2's report (2026-10-19): the MAP planners and the adaptive FTG (`tune2.py`) on FRESH seeds, and
ours again on the same code (worktree /data/wt/v2), with `final_eval.py`'s protocol and selection.

(The round-1 text follows.) The report after the fair tuning (2026-10-18): every system on FRESH seeds, ours included.

Each baseline runs with the best setting its own studies found (`tune.py`), chosen among each study's
top two by the mean of the tuning measurement and one re-measurement on an unseen seed (82 race, 124
solo; `best`): the race setting in the
race protocol (4 planner opponents x seeds 91-92, race of 2, props on the line, 32 envs x 2400 steps),
the solo setting in the time trial (128 trials, seed 321). Our pick runs the same two protocols on the
same seeds. Nothing here was seen by any selection, ours or theirs.

usage: final_eval.py GPU MAP [METHOD ...]     (no METHOD: ours + every method with both studies done)
Writes tune/final/<map>/<system>_{race_s<seed>_<kind>,solo}.json and appends a line per system to
tune/final/<map>/summary.log.
"""
from __future__ import annotations

import json
import os
import sys

import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune2 as TU  # noqa: E402

SEEDS = (91, 92)
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
SOLO_SEED = 321
OURS = {"real:korea_2026_competition": "/home/shchon11/F1tenth_E2E/checkpoints/korea26_fixphys_s1201_u384.pt",
        "real:korea_2025_iccas": "/home/shchon11/F1tenth_E2E/checkpoints/iccas_fixphys_s1102_u384.pt",
        "real:icra2022": "/home/shchon11/f1sim_runs/map3R2_s1321/ppo_u384.pt"}
METHODS = ("raceline", "forzaeth", "forzaeth_pred", "lane_switch", "ftg")
# round 6 (10-20): a map's own checkpoint can be replaced for one run (OURS_CKPT_<short map name>=path)
for _t in list(OURS):
    _o = os.environ.get("OURS_CKPT_" + _t.split(":", 1)[1])
    if _o:
        OURS[_t] = _o


#: The selection re-check: one more measurement of each study's top `RECHECK` settings on a seed the
#: tuning never saw (and the report never uses), so the setting carried forward is the best on two
#: measurements, not the luckiest single one -- a race trial is ~2.5 km, and its hard-hit count moves
#: by +-1.5 /km between seeds.
RECHECK = 2
RECHECK_SEED = {"race": 82, "solo": 124}


def best(track: str, method: str, objective: str, sel_dir: str = ""):
    name = f"{TU.SHORT[track]}__{method}__{objective}"
    db = os.path.join(TU.TUNE, name, "study.db")
    if not os.path.exists(db):
        return None
    st = optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return None
    top = sorted(done, key=lambda t: t.value, reverse=True)[:RECHECK]
    scored = []
    for t in top:
        p = _complete(method, track, dict(t.params))
        if sel_dir:
            f = os.path.join(sel_dir, f"select_{method}_{objective}_t{t.number:03d}.json")
            if objective == "race":
                r = TU.race(method, track, p, f, seed=RECHECK_SEED["race"]) if not os.path.exists(f) \
                    else _race_score_inputs(f)
                v2 = r["speed"] - 2.0 * max(0.0, r["hard"] - TU.H_REF[track])
            else:
                r = TU.solo(method, track, p, f, seed=RECHECK_SEED["solo"])
                v2 = -r["lap_time_s"] - 1.0 * max(0, 62 - r["completions"])
            scored.append((0.5 * (t.value + v2), t, v2))
        else:
            scored.append((t.value, t, None))
    m, b, v2 = max(scored, key=lambda x: x[0])
    return {"params": _complete(method, track, dict(b.params)), "number": b.number, "value": b.value, "recheck": v2, "mean": m,
            "trials": len(done), "candidates": [[x[1].number, x[1].value, x[2]] for x in scored]}


def _complete(method: str, track: str, p: dict) -> dict:
    """A planner trial's knobs plus the map's controller-stage MAP knobs it was run with -- the speed-plan knobs a
    round-6b planner trial tuned itself (`tune2._planner_speed`) staying the trial's own."""
    if method not in TU.PLANNERS:
        return p
    out = dict(p, **TU.ctrl_best(track))
    for k in TU.SPEED_PASS:
        if k in p:
            out[k] = p[k]
    return out


def _race_score_inputs(f: str) -> dict:
    n = _race_numbers(f)
    return {"speed": n["speed"], "hard": (n["wall_prop"] + n["car"]) / n["km"]}


#: Our pick with its plan followed by MAP (the user, 10-19: "우리도 그냥 map 쓰자"): the paper's variants with
#: each car's calibrated delay -- zero-shot on ICCAS's solo protocol 7.92 s 128/128 against the iLQR's 8.40.
OURS_MAP = None


def _ours_map_params(track: str = None) -> dict:
    """The MAP our plans are followed by. Round 3 (TUNE_DIR=tune3, 10-20): the map's controller-stage best on the
    tracking RMSE -- the same controller every planner on that map drives with. Round 2: the paper's formulation
    with each car's calibrated delay (chosen on lap time, which the user ruled out for a controller)."""
    if track is not None and os.path.basename(TU.TUNE) != "tune2":
        return TU.map_params(TU.ctrl_best(track))
    return dict(TU.map_params(dict(TU.MAP_PAPER, scale_span=1.0)), delay_comp=-1.0)


def ours_race(track: str, out: str, seed: int, kind: str, follower: bool = False) -> dict:
    """Our pick in exactly the race protocol `tune.race` runs (one opponent kind)."""
    import subprocess
    out = os.path.abspath(out)
    slot = {"kind": kind, "speed_scale": [0.7, 1.0]}
    cmd = [TU.PY, "-u", "-m", "f1sim.learn.evaluate", OURS[track], "--action-mode", "plan", "--tracks", track,
           "--envs", "32", "--steps", "2400", "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5",
           "--teacher-a-brake", "4.0", "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0",
           "--device", "cuda:0", "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles",
           "--spawn-runway", "3.0", "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
           # RACE_CORRIDOR (10-24): "on" lays the props off the racing line (the layout's default); the finals used "off"
           "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off"),
           "--opp-slots", json.dumps([slot]), "--output", out]
    if follower:
        cmd += ["--plan-follower", "map", "--plan-follower-map", json.dumps(_ours_map_params(track))]
    if os.environ.get("OURS_DIAL"):
        cmd += ["--dial-value", os.environ["OURS_DIAL"]]        # round 5: nobody is told the floor
    if os.environ.get("TUNE_FIXED_MU"):
        cmd += TU.floor_flags()                                # round 6: every car on the representative floor
    if os.environ.get("OURS_SPEED_SCALE"):
        cmd += ["--plan-speed-scale", os.environ["OURS_SPEED_SCALE"]]   # the venue's speed knob
    if os.environ.get("OURS_EXTRA"):
        cmd += os.environ["OURS_EXTRA"].split()                 # 10-26: e.g. the drive-time speed limits
    subprocess.run(cmd, env=TU._env(), cwd=TU.WT, check=True, stdout=open(out + ".log", "w"),
                   stderr=subprocess.STDOUT, timeout=7200)
    return _race_numbers(out)


def _trk(d: dict) -> dict:
    """The tracking meter's numbers of one run, as (count, sum of squares) so cells can be pooled."""
    t = (d.get("tracking") or {}).get("horizons") or {}
    out = {}
    for key, h, f in (("pos05", "0.5", "pos_rmse_m"), ("lat02", "0.2", "lat_rmse_m"), ("lat05", "0.5", "lat_rmse_m")):
        v = t.get(h) or {}
        n = int(v.get("n") or 0)
        out[key] = (n, n * float(v.get(f) or 0.0) ** 2)
    return out


def _pool(parts: list) -> dict:
    res = {}
    for key in ("pos05", "lat02", "lat05"):
        n = sum(p[key][0] for p in parts if key in p)
        e2 = sum(p[key][1] for p in parts if key in p)
        res[key] = (e2 / n) ** 0.5 if n else None
    return res


def _race_numbers(out: str) -> dict:
    d = json.load(open(out))["nominal"]
    cs = d.get("contact_severity") or {}
    km = max(1e-6, float(cs.get("learner_distance_m", 0.0)) / 1000.0)
    h = {k: int(sum(((cs.get(k) or {}).get("fine_hist") or [])[15:])) for k in ("wall_prop", "car")}
    return {"speed": float(d["mean_speed"]), "wall_prop": h["wall_prop"], "car": h["car"], "km": km,
            "passes": int(d["traffic"].get("passes") or 0), "trk": _trk(d)}


def ours_solo(track: str, out: str, follower: bool = False) -> dict:
    import subprocess
    out = os.path.abspath(out)
    pf = f", plan_follower='map', plan_follower_map={_ours_map_params(track)!r}" if follower else ""
    if os.environ.get("OURS_DIAL"):
        pf += f", dial_value={float(os.environ['OURS_DIAL'])!r}"
    _f = float(os.environ.get("FLOOR_MU", "1.0"))
    floor = (f"cfg.rand.ranges['vehicle.mu'] = ({_f}, {_f}); cfg.rand.ranges['vehicle.mu_f_scale'] = (0.92, 0.92)\n"
             if os.environ.get("TUNE_FIXED_MU") else "")
    if os.environ.get("OURS_SPEED_SCALE"):
        pf += f", plan_speed_scale={float(os.environ['OURS_SPEED_SCALE'])!r}"
    if os.environ.get("OURS_SOLO_KW"):
        pf += os.environ["OURS_SOLO_KW"]                        # 10-26: e.g. ", opponent_speed_limit=True, ..."
    code = (f"import json\nfrom f1sim.learn.evaluate import evaluate\nfrom f1sim.params import Config\n"
            f"cfg = Config()\n{floor}"
            f"r = evaluate({OURS[track]!r}, [{track!r}], 128, 2400, 8.0, 'cuda', seed={SOLO_SEED}, cfg=cfg, "
            f"protocol='trials', budget_laps=3.0{pf})\njson.dump(r, open({out!r}, 'w'), default=str)\n")
    subprocess.run([TU.PY, "-c", code], env=TU._env(), cwd=TU.WT, check=True, stdout=subprocess.DEVNULL,
                   stderr=open(out + ".err", "w"), timeout=3600)
    r = json.load(open(out))
    return {"completions": int(r.get("completions") or 0), "lap_time_s": float(r.get("lap_time_s") or 99.0),
            "trk": _trk(r)}


def main():
    gpu, track = sys.argv[1], sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    want = sys.argv[3:] or ["ours", *METHODS]
    d = os.path.join(TU.TUNE, os.environ.get("FINAL_DIRNAME", "final"), TU.SHORT[track])   # 10-24: another table beside it
    os.makedirs(d, exist_ok=True)
    log = open(os.path.join(d, "summary.log"), "a")
    for sysname in want:
        try:
            line = _one(sysname, track, d)
        except Exception as exc:                     # one system failing must not stop the others
            print(f"{sysname}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if line is None:
            continue
        print(json.dumps(line), flush=True)
        log.write(json.dumps(line) + "\n"); log.flush()


def _one(sysname: str, track: str, d: str):
    if True:
        if sysname in ("ours", "ours_map"):
            fol = sysname == "ours_map"
            race_cells = []
            for s in SEEDS:
                for k in KINDS:
                    f = os.path.join(d, f"{sysname}_race_s{s}_{k}.json")
                    race_cells.append(_race_numbers(f) if os.path.exists(f) else ours_race(track, f, s, k, fol))
            f = os.path.join(d, f"{sysname}_solo.json")
            if os.path.exists(f):
                r = json.load(open(f))
                so = {"completions": int(r.get("completions") or 0), "lap_time_s": float(r.get("lap_time_s") or 99.0),
                      "trk": _trk(r)}
            else:
                so = ours_solo(track, f, fol)
            setting = os.path.basename(OURS[track]) + (" + MAP " + json.dumps(_ours_map_params(track)) if fol else "")
        else:
            br, bs = best(track, sysname, "race", d), best(track, sysname, "solo", d)
            if br is None or bs is None:
                print(f"{sysname}: studies not done, skipped", flush=True)
                return None
            race_cells = []
            for s in SEEDS:
                for k in KINDS:
                    f = os.path.join(d, f"{sysname}_race_s{s}_{k}.json")
                    if os.path.exists(f):
                        race_cells.append(_race_numbers(f))
                    else:
                        TU.race(sysname, track, br["params"], f, seed=s, kinds=[k])
                        race_cells.append(_race_numbers(f))
            f = os.path.join(d, f"{sysname}_solo.json")
            if not os.path.exists(f):
                TU.solo(sysname, track, bs["params"], f, seed=SOLO_SEED, envs=128)
            r = json.load(open(f))
            so = {"completions": int(r.get("completions") or 0), "lap_time_s": float(r.get("lap_time_s") or 99.0),
                  "trk": _trk(r)}
            setting = json.dumps({"race": br["params"], "solo": bs["params"], "race_trial": br["number"],
                                  "solo_trial": bs["number"], "trials": [br["trials"], bs["trials"]],
                                  "race_selection": br["candidates"], "solo_selection": bs["candidates"]})
        km = sum(c["km"] for c in race_cells)
        line = {"system": sysname, "map": track, "solo_lap_s": so["lap_time_s"], "solo_done": so["completions"],
                "race_speed": sum(c["speed"] for c in race_cells) / len(race_cells),
                "hard_wall_prop_km": sum(c["wall_prop"] for c in race_cells) / km,
                "hard_car_km": sum(c["car"] for c in race_cells) / km, "passes_km": sum(c["passes"] for c in race_cells) / km,
                "race_km": km, "setting": setting,
                # the tracking meter: position RMSE 0.5 s after issue, lateral RMSE 0.2 / 0.5 s (None: no reference)
                "trk_solo": _pool([so["trk"]]), "trk_race": _pool([c["trk"] for c in race_cells])}
        return line


if __name__ == "__main__":
    main()
