"""Label replay for the "wp4" plan: the raceline teacher's labels encoded as waypoints, decoded and
driven closed-loop through the env's own tracker -- the expressiveness check of the output-format
design (docs/research/output-format-2026-10-14.md section 4) on the repo's `mpc` rather than the
design's prototype.

The label is the one design section 5 describes: the teacher's linear 1.5 s plan (its 6-knot
Gauss-Newton fit to the line), whose first window is refitted to the four waypoint curvatures
(`mpc.wp4_refit`), and the teacher's own grip profile sampled at the waypoints, times its off-line
slowdown -- no speed lead, no certification; the decoder's backward pass is the feasibility step.
Until the teachers encode wp4 themselves (migration step 3) this script is where that label lives.

    python f1sim/scripts/wp4_label_replay.py --track real:korea_2025_iccas --envs 8 --steps 2400

Design reference (min-time teacher 7 / 6.5 / 4, 8 envs, 60 s, seed 0, the prototype's labels fitted
straight to the raceline points): ICCAS 7.925 s median lap, 0 contacts, |v_profile - v| 0.906 m/s;
icra2022 15.350 s, 8 contacts (the design's own run without the lateral cap it then dropped; the 4 in
its table had the cap). The same prototype labels on this tree's decoder and tracker give the
same numbers (ICCAS exactly; icra2022 15.350 s, 9 contacts). This script's refitted labels, measured
2026-10-14 on the same setup: ICCAS 7.800 s (mean 7.890 against 7.986), 0 contacts, 0.874 m/s;
icra2022 15.150 s with 14 contacts against the line fit's 9 -- faster, and closer to the duct walls
of a narrow track (likely, not measured: the 1.5 s plan is fitted to the line only from 40 % of its
length on, so nothing holds its first 0.6 s to the line). One seed, ~2 km each: weak, but the wall
count is the thing to watch when the teacher's encoder is ported: the design's "20 wall contacts ->
0" is its icra2022 traffic replay, where only the OFFSET candidates were refitted and the on-line
rows kept the direct fit.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from f1sim import mpc
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.params import Config

LIMITS = dict(a_lat=7.0, a_acc=6.5, a_brake=4.0)          # the design's min-time teacher


@torch.no_grad()
def wp4_label(teacher, env, spec_wp4: mpc.PlanSpec) -> torch.Tensor:
    """(B, 8) the teacher's plan as a wp4 action (see the module docstring)."""
    state, P, tid = env.sim.state, env.sim.P, env.sim.tid
    v_max = float(env.ecfg.v_max_policy)
    v_geom = env._tracker_plan_speed()                    # the speed the env's decoder lays the plan out with
    lin = mpc.PlanSpec()
    a_lin = teacher.plan_action(state, P, tid, v_max, lin, plan_speed=v_geom)
    k6, Lp, _, _ = mpc.decode(a_lin, v_geom, v_max, torch.full_like(v_geom, v_max), lin)
    # The bound the teacher's own knots live under (`RacelineTeacher.plan_action`): 85 % of the tightest
    # arc this car's steering holds. Under the format's bare 1.6 1/m the refit put 5 % of its knots past
    # it on icra2022 (up to 19 % in a step). Bounded, icra2022 laps 15.150 s against 15.175 with the
    # same 14 contacts and ICCAS is unchanged: not the cause of the walls, but the teacher's own limit.
    wheelbase = P["lf"] + P["lr"]
    steering = torch.as_tensor(P["s_max"], device=state.device).clamp(max=teacher.steer_max)
    k_lim = 0.85 * torch.minimum(torch.full_like(v_geom, spec_wp4.kappa_max), torch.tan(steering) / wheelbase)
    kappa = mpc.wp4_refit(k6, Lp, v_geom, spec_wp4, k_lim=k_lim)
    idx, lat_err = teacher.project(state[:, :2], tid)
    L = mpc.plan_length(v_geom, spec_wp4)
    s_k = L[:, None] * torch.arange(1, mpc.WP_K + 1, device=L.device, dtype=L.dtype) / mpc.WP_K
    kidx = (idx[:, None] + (s_k / teacher.ds[tid][:, None]).round().long()) % teacher.N
    gb = teacher.grip_bin(P, state.shape[0], state.device)
    slow = (1.0 - teacher.lat_slow * lat_err).clamp(0.3, 1.0)
    v = teacher.speed_at(tid[:, None].expand_as(kidx), kidx, gb[:, None].expand_as(kidx)) * slow[:, None]
    return mpc.encode_wp4(kappa, v, v_max, spec_wp4, v_meas=v_geom)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--track", default="real:korea_2025_iccas")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2400, help="control steps (40 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed-cap", type=float, default=9.0)
    ap.add_argument("--out", default="", help="write the result as JSON here too")
    a = ap.parse_args()
    torch.set_num_threads(2)
    trs, rls = common.load_tracks([a.track], racelines=True, objective="min_time", **LIMITS)
    cfg = Config(); cfg.sim.compile = False
    env = common.make_env(trs, a.envs, "cpu", EnvConfig(speed_cap=a.speed_cap, action_mode="plan", speed_mode="wp4",
                                                        max_steps=a.steps + 10, collision_mode="soft",
                                                        compile_tracker=False), cfg=cfg, seed=a.seed)
    teacher = common.make_teacher(rls, env, **LIMITS)
    teacher.certify_speeds = False          # only the linear plan's PATH is used; its speeds are not
    spec = env.tracker.spec
    torch.manual_seed(a.seed)
    env.reset()
    laps, contacts, dist, verr, n = [], 0, 0.0, 0.0, 0
    t0 = time.time()
    for _ in range(a.steps):
        st = env.sim.state
        idx, _ = teacher.project(st[:, :2], env.sim.tid)
        v_prof = teacher.speed_at(env.sim.tid, idx, teacher.grip_bin(env.sim.P, a.envs, st.device))
        verr += float((v_prof - st[:, 3]).abs().sum()); n += a.envs
        _, _, _, _, info = env.step(wp4_label(teacher, env, spec))
        laps += info["lap_times"].tolist()
        contacts += int(info["contact_onset"].sum())
        dist += float(env.sim.state[:, 3].clamp_min(0).sum()) * env.sim.control_dt
    laps = np.asarray(laps)
    res = {"track": a.track, "envs": a.envs, "steps": a.steps, "seed": a.seed, "laps": int(laps.size),
           "lap_median": float(np.median(laps)) if laps.size else None,
           "lap_mean": float(laps.mean()) if laps.size else None,
           "lap_p10": float(np.percentile(laps, 10)) if laps.size else None,
           "lap_p90": float(np.percentile(laps, 90)) if laps.size else None,
           "contacts": contacts, "km": dist / 1000, "mean_abs_vprofile_minus_v": verr / max(n, 1),
           "wall_s": time.time() - t0}
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
