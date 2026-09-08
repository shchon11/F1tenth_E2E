"""One command, one table: what a checkpoint can actually do.

    python3 scripts/capability_report.py ~/f1sim_runs/ppo_v17/ppo_final.pt [--quick] [--teacher]

Runs the same batteries this project's decisions were made on, all deterministic with fixed seeds and
20 s per car: solo driving on the held-out maps, static obstacles of kinds the policy never trained on
(including boxes sitting on the racing line), and races against raceline-teacher cars at two speed
ratios. Add --teacher for the same rows driven by the teacher (map + raceline) as a reference.

Race rows report contact *per overtake* as well: a policy that never attempts a pass has no contacts,
so the rate alone says little.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from f1sim import maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.learn.model import load_checkpoint
from f1sim.learn.obs import flatten_obs
from f1sim.raceline import Raceline

CLEAN = ["real:korea_2025_iccas", "real:korea_2025_iccas~rev", "rt:Monza", "real:blackbox2022_3~lane",
         "real:icra2022", "real:blackbox2022_1~mir"]


def n_obs(t):
    L = float(np.linalg.norm(np.roll(t.centerline, -1, 0) - t.centerline, axis=1).sum())
    return int(np.clip(round(L / 35.0), 3, 8))


OBSTACLES = [
    ("no obstacles", lambda t, s: t),
    ("boxes at the lane side", lambda t, s: t.with_lane_obstacles(seed=s, n=n_obs(t))),
    ("big boxes 0.5-0.9 m", lambda t, s: t.with_lane_obstacles(seed=s, n=n_obs(t), size=(0.5, 0.9), min_passage=1.0)),
    ("cylinders", lambda t, s: t.with_lane_obstacles(seed=s, n=n_obs(t), size=(0.3, 0.6), kind="cyl")),
    ("anywhere across the lane", lambda t, s: t.with_lane_obstacles(seed=s, n=n_obs(t), lateral="random")),
    ("on the racing line", lambda t, s: t.with_lane_obstacles(seed=s, n=n_obs(t), on_path=Raceline.build_cached(t).xy)),
]


def drive(tracks, rls, model, spec, cap, cars_per_track, steps, dev, race=1, opp_speed=(1.0, 1.0), seed=7):
    """Returns (crashes, car contacts, overtakes, progress) per learner car over `steps` control steps."""
    M = race
    B = len(tracks) * cars_per_track * M
    env = common.make_env(tracks, B, dev, EnvConfig(action_mode="plan", speed_cap=cap, resample_track_on_reset=False,
                                                    hist_len=spec.get("hist_len", 0), scan_stack=spec.get("scan_stack", 3),
                                                    scan_stride=spec.get("scan_stride", 1), proximity_speed_ref=4.0,
                                                    reward_collision=-150.0, reward_collision_speed=20.0, race_size=M,
                                                    opponent="teacher" if M > 1 else "policy", opp_speed_range=opp_speed),
                          seed=seed, rls=rls)
    obs, _ = env.reset(seed=seed)
    teacher = common.make_teacher(rls, env) if model is None else None
    lm = env.learner; L = env.sim.track.length[env.sim.tid]
    lead = (torch.arange(B, device=dev) // M) * M
    coll = torch.zeros(B, device=dev); carc = torch.zeros(B, device=dev); prog = torch.zeros(B, device=dev)
    passes = 0; prev = None
    for _ in range(steps):
        if model is None:
            a = env.teacher_label(teacher)
        else:
            with torch.no_grad():
                a, _ = model.act(*flatten_obs(obs), deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        coll += term.float(); prog += info["progress"]
        if M > 1:
            carc += (term & info["car_collision"]).float()
            d = ((env.last_result.s[lead] - env.last_result.s + L / 2) % L) - L / 2
            re = (torch.zeros(B // M, device=dev).index_add_(0, torch.arange(B, device=dev) // M, (term | trunc).float()) > 0)[torch.arange(B, device=dev) // M]
            if prev is not None:
                ok = ~re & ~lm & ((d - prev).abs() < 3.0)
                passes += int(((prev < 0) & (d > 0) & ok).sum())
            prev = d.clone()
    n = int(lm.sum())
    return float(coll[lm].mean()), float(carc[lm].mean()), passes / n, float(prog[lm].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--teacher", action="store_true", help="also drive every row with the raceline teacher")
    ap.add_argument("--quick", action="store_true", help="fewer cars per track")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, extra = load_checkpoint(a.ckpt, dev); model.eval(); spec = extra.get("spec", {})
    k = 8 if a.quick else 16
    steps = 800                                                    # 20 s at 40 Hz
    who = [("policy", model)] + ([("teacher", None)] if a.teacher else [])
    print(f"{os.path.basename(os.path.dirname(a.ckpt))}/{os.path.basename(a.ckpt)}   {k} cars per track, {steps / 40:.0f} s each, deterministic\n")
    print("STATIC  (held-out maps; crashes per car per 20 s)")
    print(f"  {'obstacles':26s} {'cap 4':>12s} {'cap 6':>12s}")
    for name, fn in OBSTACLES:
        tracks = [fn(maps.load(b), 100 + i) for i, b in enumerate(CLEAN)]
        rls = [Raceline.build_cached(t) for t in tracks]
        cells = []
        for cap in (4.0, 6.0):
            for label, m in who:
                c, _, _, p = drive(tracks, rls, m, spec, cap, k, steps, dev)
                cells.append(f"{c:.2f}" + (f"/{label[0]}" if len(who) > 1 else ""))
        print(f"  {name:26s} " + "".join(f"{c:>12s}" for c in cells))
    print("\nRACING  (against raceline-teacher cars, which never yield)")
    print(f"  {'situation':26s} {'contact':>9s} {'overtakes':>10s} {'per pass':>9s} {'progress':>9s}")
    tracks, rls = common.load_tracks(CLEAN, racelines=True)
    for cap in (5.0, 6.0):
        for opp, tag in (((0.5, 0.7), "slow traffic"), ((0.8, 1.0), "similar speed")):
            _, cc, ov, p = drive(tracks, rls, model, spec, cap, max(4, k // 2), steps, dev, race=3, opp_speed=opp)
            print(f"  cap {cap:.0f}, {tag:18s} {cc:9.2f} {ov:10.2f} {cc / max(ov, 1e-6):9.2f} {p:8.1f} m")


if __name__ == "__main__":
    main()
