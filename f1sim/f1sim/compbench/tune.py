"""(11-28 PM decision: NOT run -- the old opponents are not to be tuned further; their rebuild is a separate task. Kept
because a rebuilt opponent will need exactly this. Smoke-tested: the FTG split run on 6 sessions.)

Fair tuning of a baseline on this suite's map, on the competition objective, with a stated budget.

  python -m f1sim.compbench tune <kind> <stage> [n_candidates]      kind: ftg | disparity | forzaeth

Sessions: the TUNE bank (64 layouts drawn by exam_line_bank.py with seed 20261130 -- not dev's exam3, not confirm's),
120 s sessions, boxes removed at 60 s, condition real (MAP + default physics), every session boxed. Neither dev nor
confirm sessions are ever used to pick a setting.
  stage 1: K candidates share one 128-session run (`engine_exam.SplitDriver` / `SplitPlannerDriver`: session r driven
           by candidate (r + r // 64) % K), physics seed 20261130. Candidates: 0 = the setting tuned on the neighbouring
           map korea_2026_competition (solo), 1 = its race setting, 2 = published defaults (sensor kinds), the rest
           random around candidate 0 within the bounds below (log-uniform per knob).
  stage 2: the best 4 of stage 1 again on another 128 sessions (32 each, physics seed 20261131); the best is 'tuned'.
Objective (the qualifier's own score): mean over sessions of the longest run of consecutive laps without a hard
  contact; ties (within 0.25 laps) broken by the median best lap. Budget recorded in DATA/tuned/k26scan.json."""
import os
import sys
import json
import copy
import time

import numpy as np

from . import DATA, VERSION
from .suite import load_suite, resolve_path
from . import drivers as D

TUNE_BANK = "banks/tune_k26scan"
SEEDS = {1: 20261130, 2: 20261131}

BOUNDS = {
    "ftg": {"max_speed": (4.0, 9.5), "safety_radius": (8, 45), "max_lidar_dist": (2.5, 9.0), "track_width": (1.2, 3.6),
            "r_min": (0.5, 1.0), "r_max": (2.5, 5.5), "t_react": (0.15, 0.6), "a_brake": (3.5, 8.5),
            "scan_quantile": (0.45, 0.75), "scan_gain": (0.3, 0.9), "gap_margin": (0.12, 0.55)},
    "disparity": {"top_speed": (4.0, 9.5), "tolerance": (0.1, 0.4), "disparity_threshold": (0.15, 0.6),
                  "side_margin": (0.12, 0.3), "d_near": (0.4, 1.0), "d_mid": (1.5, 3.5), "d_far": (4.0, 9.0),
                  "v_min": (0.6, 1.5)},
    "forzaeth": {"teacher.a_lat": (4.5, 9.0), "teacher.a_acc": (3.5, 7.5), "teacher.a_brake": (3.5, 7.5),
                 "teacher.v_max_profile": (4.0, 9.0), "planner.evasion_dist": (0.5, 1.4),
                 "planner.side_switch_d": (0.2, 0.7), "planner.obs_traj_thresh": (0.1, 0.4),
                 "planner.lookahead": (6.0, 14.0), "det.map_margin": (0.18, 0.4), "speed_scale": (0.8, 1.1)},
}
LIMIT_SETS = ([9.0, 7.5, 6.0], [8.0, 7.0, 5.0], [7.0, 6.5, 4.0])     # the min-time lines cached for this map (another set = 1-2 min of CPU inside a GPU job)


def _get(d, key):
    for p in key.split("."):
        d = (d or {}).get(p)
    return d


def _set(d, key, v):
    ps = key.split(".")
    for p in ps[:-1]:
        d = d.setdefault(p, {})
    d[ps[-1]] = v


def candidates(kind, n, seed=20261130):
    rng = np.random.default_rng(seed)
    base_solo = D._baseline_setting(kind, "k26comp", solo=True)
    base_race = D._baseline_setting(kind, "k26comp", solo=False)
    out = [dict(base_solo, source=f"cand0 = {base_solo.get('source')}"), dict(base_race, source=f"cand1 = {base_race.get('source')}")]
    if kind != "forzaeth":
        out.append(dict(options={}, source="cand2 = published defaults"))
    B = BOUNDS[kind]
    while len(out) < n:
        c = copy.deepcopy(base_solo)
        holder = c.setdefault("options", {}) if kind != "forzaeth" else c.setdefault("stack_params", {})
        for k, (lo, hi) in B.items():
            if kind == "forzaeth" and k == "speed_scale":
                c["speed_scale"] = float(rng.uniform(lo, hi))
                continue
            cur = _get(holder, k)
            if cur is None:
                cur = 0.5 * (lo + hi)
            # around the base: log-uniform within x0.67..x1.5 of it, clipped to the bounds
            v = float(np.clip(float(cur) * np.exp(rng.uniform(np.log(0.67), np.log(1.5))), lo, hi))
            if isinstance(lo, int) and isinstance(hi, int):
                v = int(round(v))
            _set(holder, k, v)
        if kind == "ftg":
            holder["target"] = str(rng.choice(["middle", "weighted"]))
        if kind == "forzaeth":
            c["limits"] = list(LIMIT_SETS[int(rng.integers(0, len(LIMIT_SETS)))])
        c["source"] = f"cand{len(out)} random (seed {seed})"
        out.append(c)
    return out[:n]


def spec_of(kind, c):
    if kind == "forzaeth":
        sp = dict(c.get("stack_params") or {})
        if "speed_scale" in c:
            sp["speed_scale"] = float(c["speed_scale"])
        return {"kind": kind, "stack_params": sp, "limits": list(c.get("limits") or [9.0, 7.5, 6.0])}
    return {"kind": kind, "options": dict(c.get("options") or {})}


def run_stage(kind, stage, cands, idx):
    from .runner import base_job, run_job
    from .score import score_record
    from .engine_exam import split_assign
    s = load_suite("dev")
    m = dict(s["maps"][0], id="tune_k26scan", bank=TUNE_BANK, bank_sha256=None)
    os.environ["COMPBENCH_UNFROZEN"] = "1"
    drv = dict(id=f"tune_{kind}_stage{stage}", engine="exam", label=f"{kind} tuning stage {stage}",
               ego={"kind": "split", "n_lay": 64, "members": [spec_of(kind, cands[i]) for i in idx]})
    job = base_job(s, "real", m, drv, SEEDS[stage], 120.0, 60.0, 256)
    od = os.path.join(DATA, "raw", "tune", kind)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"stage{stage}.npz")
    job.update(out=out, ego=drv["ego"], candidates=[cands[i] for i in idx], candidate_ids=list(idx),
               raceline_cache_only=False)           # a candidate's own line limits may need one built (1-2 min of CPU)
    if not os.path.exists(out):
        rc = run_job(job, out[:-4] + ".log", "exam")
        if rc != 0:
            raise RuntimeError(f"tuning run failed: {out[:-4]}.log")
    R = score_record(out)
    who = split_assign(len(R["sessions"]), len(idx), 64)
    rank = []
    for k, ci in enumerate(idx):
        S = [r for r, w in zip(R["sessions"], who) if w == k]
        g = lambda key: np.array([r[key] for r in S], float)
        rank.append(dict(cand=int(ci), sessions=len(S), streak_hard=float(g("streak_hard").mean()),
                         best=float(np.nanmedian(g("best"))), dnf=float(g("dnf").mean()), clean=float(g("clean").mean()),
                         dist=float(g("dist").mean()), wall_hard=float(g("wall_hard").mean()), box_hard=float(g("box_hard").mean())))
    rank.sort(key=lambda r: (-round(r["streak_hard"] / 0.25), r["best"] if np.isfinite(r["best"]) else 1e9))
    return rank, out


def main(argv):
    kind, stage = argv[0], int(argv[1])
    n = int(argv[2]) if len(argv) > 2 else (12 if kind == "forzaeth" else 16)
    tf = D.tuned_file()
    os.makedirs(os.path.dirname(tf), exist_ok=True)
    allt = json.load(open(tf)) if os.path.exists(tf) else {}
    ent = allt.get(kind) or {}
    if stage == 1:
        cands = candidates(kind, n)
        ent = dict(candidates=cands, bounds=BOUNDS[kind], objective="mean longest run of laps without a hard contact; "
                   "tie (0.25 laps) -> median best lap", tune_bank=TUNE_BANK, compbench=VERSION)
        t0 = time.time()
        rank, out = run_stage(kind, 1, cands, list(range(len(cands))))
        ent.update(stage1=rank, stage1_record=out, stage1_s=time.time() - t0)
    else:
        cands = ent["candidates"]
        top = [r["cand"] for r in ent["stage1"][:4]]
        t0 = time.time()
        rank, out = run_stage(kind, 2, cands, top)
        best = cands[rank[0]["cand"]]
        pick = dict(best, source=f"compbench tune {kind}: cand{rank[0]['cand']} ({best.get('source')}), stage 2 best of {top}")
        ent.update(stage2=rank, stage2_record=out, stage2_s=time.time() - t0, solo=pick, race=pick,
                   budget=dict(sessions=256 * 1, session_s=120, runs=2, candidates=len(cands),
                               note="stage 1: all candidates on 128 sessions (~128/K each); stage 2: top 4 on 128 more (32 each); "
                                    "the race setting is the solo pick (not tuned separately)"))
    allt[kind] = ent
    json.dump(allt, open(tf, "w"), indent=1, default=float)
    for r in rank:
        print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1:])
