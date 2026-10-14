"""Label replay: a teacher's own plans driven closed-loop through an env's tracker, in either plan
format -- the expressiveness check of the output-format design (docs/research/output-format-
2026-10-14.md section 4) and the closed-loop label-quality gates of its section 7 (tests 3, 4, 6), on
this tree's `mpc` and teachers.

The teachers speak "wp4" natively (`RacelineTeacher._plan_wp4`), so the label is whatever the
production path hands a student: `env.teacher_label(teacher)`, i.e. the raceline teacher behind the
env's guard (an `InteractiveTeacher` with the line's own speed family, previewed on the env's
tracker). `--ego raw` skips the guard, `--ego interactive` labels with the traffic teacher DAgger's
`--teacher interactive` builds, `--ego layout` with the per-layout line teacher.

    python f1sim/scripts/wp4_label_replay.py --track real:korea_2025_iccas --format wp4 --envs 256 --steps 1200
    python f1sim/scripts/wp4_label_replay.py --format wp4 --opponent forzaeth_pred --envs 128     # traffic
    python f1sim/scripts/wp4_label_replay.py --format wp4 --opp-format linear --opponent lane_switch  # linear cars
    python f1sim/scripts/wp4_label_replay.py --format linear --noise                                # gate 6

Counted on the ego rows (`env.learner`): wall/prop contact onsets (`info["wall_onset"]`), car contact
onsets except in the first 2 s of an episode (a spawn-contact bug is being fixed separately), km,
laps, the share of steps stopped off the line (< 0.3 m/s and > 0.15 m from the raceline, after 2 s)
and the share of label steps whose first curvature number jumps by more than 0.1 (design 7 test 3);
walls after 2 s apart, and how often the guard's (or the interactive ego's) label is the stop, an
evasion, a braking evasion.

History. UNIT 1 (2026-10-14, 8 CPU envs, 60 s, seed 0, the min-time teacher 7 / 6.5 / 4) replayed
labels built outside the teacher: the design prototype's (a GN fit straight to the line) reproduced
the design exactly on this tree's decoder (ICCAS 7.925 s, 0 contacts, |v_profile - v| 0.906;
icra2022 15.350 s, 9 contacts), and the design-section-5 refit (the 1.5 s plan's first window) lapped
1.6 % / 1.3 % quicker but touched the icra2022 walls 14 times against 9 -- one seed, ~2 km. That is
the question `RacelineTeacher.wp4_path` was settled on, at scale and through the guard (see there):
"refit", 978 contacts in 977 km at 4.63 m/s, three tracks x two seeds solo and two x two in traffic.

The closed-loop gates on that teacher (2026-10-14, docs/research/wp4-label-gates-2026-10-14/), each
against the linear teacher's production labels, which drive slower: props 0 contacts in 17.5 km
(pass); traffic car contacts per km 1.38 / 1.84 / 0.39 against 0.33 / 1.72 / 0.06 with lane_switch /
forzaeth_pred / interactive opponents, first-curvature jumps 0.48 / 1.48 / 0.35 % (fail); icra2022
walls 3.31 per km at 3.92 m/s against 2.44 at 3.13 for linear with a 0.6 s end and no certification
(fail); noise replay 5.06 against 3.39 per km at 3.93 against 3.68 m/s (fail). At equal pace -- raw
labels against that linear -- the wp4 labels touch the icra2022 walls 20 % less (5.59 against 7.02
per km, both 15.03 s laps) and under noise 41 % less (6.25 against 10.56).

After the registration (RESULTS 7): the stop candidate and the braking evasions are -1 in wp4 too,
and `--opp-format linear` keeps the opponents in the plan they were built for while the ego plans wp4
(`EnvConfig.opponent_speed_mode`). Against lane_switch / forzaeth_pred / interactive cars the guard's
wp4 ego then touched them 0.40 / 1.29 / 0.24 times per km, production linear 0.33 / 1.72 / 0.06 and
linear with a 0.6 s end and no certification 0.37 / 1.37 / 0.12 -- pooled 0.64 against 0.69 / 0.61 at
4.05 against 3.88 / 3.95 m/s; only the interactive excess is past 1 sigma. Alone on icra2022 the
guard's walls fell from 3.31 to 2.48 per km at 3.92 m/s with the -1 stop.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from f1sim import mpc
from f1sim.gym_env import EnvConfig
from f1sim.interactive_teacher import InteractiveTeacher
from f1sim.learn import common
from f1sim.params import Config
from f1sim.teacher import RacelineTeacher

LIMITS = dict(a_lat=7.0, a_acc=6.5, a_brake=4.0)          # the design's min-time teacher
#: PPO's initial exploration per dimension (design 5): wp4's kappa_1 / kappa_2 at -2.07 so the executed
#: steering noise matches linear's, everything else at linear's -1.8.
LOG_STD = {"wp4": (-2.07, -2.07) + (-1.8,) * 6, "linear": (-1.8,) * 8}
SPAWN_S = 2.0                                              # car contacts before this are not counted
STOP_V, STOP_LAT = 0.3, 0.15                               # "stopped off the line"
JUMP = 0.1                                                 # a first-curvature label jump


def build(a):
    trs, rls = common.load_tracks([a.track], racelines=True, objective="min_time", **LIMITS)
    cfg = Config(); cfg.sim.compile = False
    kw = dict(speed_cap=a.speed_cap, action_mode="plan", speed_mode=a.format, max_steps=a.steps + 10,
              laps=10 ** 6, collision_mode="soft", compile_tracker=False,
              opponent_speed_mode=None if a.opp_format == "same" else a.opp_format)
    if a.opponent == "teacher":
        kw.update(race_size=2, opponent="teacher", opp_speed_range=tuple(a.opp_speed))
    elif a.opponent != "none":
        kw.update(race_size=2, opponent="slots",
                  opponent_slots=({"kind": a.opponent, "speed_scale": list(a.opp_speed)},))
    if a.props:
        # the design's prop replay (free proposal, layout teacher): crates on the racing line, the
        # spawn kept 3 m clear of them
        kw.update(movable_obstacles=True, procedural_obstacles=1.0, procedural_density=1.5, procedural_max_props=10,
                  procedural_raceline_corridor="off", spawn_runway=3.0)
    env = common.make_env(trs, a.envs, a.device, EnvConfig(**kw), cfg=cfg, seed=a.seed, rls=rls, teacher_limits=LIMITS)
    teacher = common.make_teacher(rls, env, **LIMITS)
    if a.evasion_brake:
        InteractiveTeacher.wp4_evasion_brake = a.evasion_brake       # the guard's, the ego's and the opponents'
    if a.recipe:
        teacher.wp4_path = a.recipe
    if a.linear_horizon is not None:
        teacher.speed_horizon_s = a.linear_horizon
    if a.no_certify:
        teacher.certify_speeds = False
    if env.teacher is not None:                  # the opponents' reference teacher reads the same recipe
        env.teacher.wp4_path = teacher.wp4_path
    label = teacher
    if a.ego == "interactive":
        label = InteractiveTeacher(teacher, env, cand_iters=2)            # learn/dagger.py's defaults
    elif a.ego == "layout":
        from f1sim.layout_line import LayoutLineTeacher
        label = LayoutLineTeacher(teacher, env)
    return env, teacher, label, rls


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--track", default="real:korea_2025_iccas")
    ap.add_argument("--format", choices=("wp4", "linear"), default="wp4")
    ap.add_argument("--recipe", choices=RacelineTeacher.WP4_PATHS, default=None,
                    help="RacelineTeacher.wp4_path (default: the teacher's own)")
    ap.add_argument("--ego", choices=("guard", "raw", "interactive", "layout"), default="guard")
    ap.add_argument("--opponent", default="none",
                    help="none | teacher (raceline opponents) | an opponent-slot kind (lane_switch, forzaeth_pred, "
                         "interactive, ...)")
    ap.add_argument("--opp-speed", type=float, nargs=2, default=(0.7, 1.0))
    ap.add_argument("--opp-format", choices=("same", "linear", "wp4"), default="same",
                    help="the opponents' plan format (EnvConfig.opponent_speed_mode): 'same' is --format's")
    ap.add_argument("--evasion-brake", choices=("stop", "curve"), default=None,
                    help="InteractiveTeacher.wp4_evasion_brake (default: the class's own)")
    ap.add_argument("--props", action="store_true", help="procedural props on the floor (use --ego layout)")
    ap.add_argument("--noise", action="store_true", help="add PPO's initial per-dimension noise (LOG_STD) to the label")
    ap.add_argument("--linear-horizon", type=float, default=None, help="RacelineTeacher.speed_horizon_s (linear)")
    ap.add_argument("--no-certify", action="store_true", help="RacelineTeacher.certify_speeds = False (linear)")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1200, help="control steps (40 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed-cap", type=float, default=9.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(2)
    env, teacher, label, rls = build(a)
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=env.device).manual_seed(1000 + a.seed)
    sigma = torch.tensor(LOG_STD[a.format], device=env.device).exp()
    ego = env.learner.clone()
    dt = env.sim.control_dt
    spawn_steps = int(round(SPAWN_S / dt))
    env.reset()
    B = env.B
    laps = []
    wall = car = car_spawn = prop = wall_spawn = 0
    km = 0.0
    n_after = stop_off = 0
    jumps = n_pairs = 0
    prev_lab, prev_car = None, torch.zeros(B, dtype=torch.bool, device=env.device)
    vsum, nv = 0.0, 0
    blocked = valid_n = 0
    # which candidate the label is: the on-line stop, an evasion, a braking evasion (ego rows, after 2 s)
    chooser = label if a.ego == "interactive" else None             # the guard exists after its first label
    n_choice = ch_stop = ch_evade = ch_evade_brake = 0
    track_props = bool(a.props)
    t0 = time.time()
    for k in range(a.steps):
        st = env.sim.state
        lab = env.teacher_label(label) if a.ego != "raw" else teacher.plan_action(
            st, env.sim.P, env.sim.tid, env.ecfg.v_max_policy, env.tracker.spec, plan_speed=env._tracker_plan_speed())
        valid = getattr(label, "last_label_valid", None)
        if torch.is_tensor(valid) and valid.shape == ego.shape:
            blocked += int((~valid & ego).sum()); valid_n += int(ego.sum())
        if chooser is None and a.ego == "guard":
            chooser = getattr(env, "_raceline_teacher_guards", {}).get(teacher)
        if chooser is not None and chooser.last_choice is not None:
            j = chooser.last_choice
            reg = chooser.offsets.numel() * chooser.speeds.numel()       # then: stop, (coast, brake) x 2 sides
            on = ego & (env.ep_step >= spawn_steps)
            n_choice += int(on.sum())
            ch_stop += int((on & (j == reg)).sum())
            ch_evade += int((on & (j > reg)).sum())
            ch_evade_brake += int((on & (j > reg) & ((j - reg) % 2 == 0)).sum())
        act = lab
        if a.noise:
            act = (lab + sigma * torch.randn(lab.shape, generator=gen, device=env.device)).clamp(-1.0, 1.0)
        settled = ego & (env.ep_step >= spawn_steps)
        if prev_lab is not None:
            same = settled & (env.ep_step > 0)
            jumps += int((((lab[:, 0] - prev_lab[:, 0]).abs() > JUMP) & same).sum()); n_pairs += int(same.sum())
        prev_lab = lab
        idx, lat = teacher.project(st[:, :2], env.sim.tid)
        n_after += int(settled.sum())
        stop_off += int((settled & (st[:, 3].abs() < STOP_V) & (lat > STOP_LAT)).sum())
        _, _, _, _, info = env.step(act)
        on = info["wall_onset"] & ego
        wall += int(on.sum())
        wall_spawn += int((on & (env.ep_step <= spawn_steps)).sum())       # counted, and also shown apart
        if track_props:
            # a wall/prop onset whose footprint is clear of the walls' distance field was a prop
            prop += int((on & (info["wall_dist"] > 0.0)).sum())
        cc = info.get("car_collision")
        if cc is not None:
            cc = cc.bool()
            onset = cc & ~prev_car & ego
            spawn = onset & (env.ep_step <= spawn_steps)
            car += int((onset & ~spawn).sum()); car_spawn += int(spawn.sum())
            prev_car = cc.clone()
        v = env.sim.state[:, 3]
        km += float((v.clamp_min(0) * dt)[ego].sum()) / 1000.0
        vsum += float(v[ego].sum()); nv += int(ego.sum())
        if info["lap_ids"].numel():
            keep = ego[info["lap_ids"]]
            laps += info["lap_times"][keep].tolist()
    wall_s = time.time() - t0
    laps = np.asarray(laps)
    res = {"track": a.track, "format": a.format, "recipe": teacher.wp4_path if a.format == "wp4" else None,
           "ego": a.ego, "opponent": a.opponent, "opp_speed": list(a.opp_speed), "props": a.props, "noise": a.noise,
           "linear_horizon": a.linear_horizon, "certify": teacher.certify_speeds, "envs": a.envs,
           "opp_format": env.opp_plan_spec.speed_mode if a.opponent != "none" else None,
           "evasion_brake": InteractiveTeacher.wp4_evasion_brake,
           "ego_rows": int(ego.sum()), "steps": a.steps, "seed": a.seed,
           "km": km, "laps": int(laps.size),
           "lap_median": float(np.median(laps)) if laps.size else None,
           "lap_mean": float(laps.mean()) if laps.size else None,
           "mean_speed": vsum / max(nv, 1),
           "wall_contacts": wall, "wall_per_km": wall / max(km, 1e-9), "wall_contacts_first_2s": wall_spawn,
           "wall_per_km_after_2s": (wall - wall_spawn) / max(km, 1e-9),
           "prop_contacts": prop if track_props else None,
           "car_contacts": car, "car_per_km": car / max(km, 1e-9), "car_contacts_first_2s": car_spawn,
           "offline_stop_frac": stop_off / max(n_after, 1),
           "kappa1_jump_frac": jumps / max(n_pairs, 1),
           "label_blocked_frac": blocked / max(valid_n, 1) if valid_n else None,
           "choice_stop_frac": ch_stop / n_choice if n_choice else None,
           "choice_evade_frac": ch_evade / n_choice if n_choice else None,
           "choice_evade_brake_frac": ch_evade_brake / n_choice if n_choice else None,
           "wall_s": wall_s, "ms_per_step": 1000 * wall_s / a.steps}
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
