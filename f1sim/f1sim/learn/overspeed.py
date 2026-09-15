"""How often does the student ask for more speed than the teacher's limit, and at which frictions?

This is the measurement `--speed-loss asym` exists to move. With `--teacher-grip true` the teacher
plans at each env's own friction limit, so its speed label IS the limit at that state, and the
quantity that matters is the signed error

    e = v_student - v_teacher      [m/s], on the plan's two speed targets

A symmetric regression fits the conditional mean of labels that are up to 2.2x apart, which puts it
*above* the limit on the low-grip draws -- precisely where above the limit ends the episode. So the
number to look at is not the mean error but the **over-speed rate as a function of mu**: a student
that has resolved grip from its history should be near zero everywhere, and one that has not should
show a rate that climbs as mu falls.

Reported per mu bin: how often e > 0 at all, how often it exceeds `--over-margin`, the mean and p90
of the over-speed, and the mean under-speed for contrast (an asymmetric loss is supposed to trade
some of the second for the first, and the trade has to be visible).

    python -m f1sim.learn.overspeed CKPT --tracks eval --envs 96 --steps 600
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..gym_env import EnvConfig
from ..interactive_teacher import InteractiveTeacher
from ..mpc import ACT_DIM, N_KNOTS
from ..params import Config
from . import common
from .memory import policy_fn as memory_policy_fn
from .model import load_checkpoint


@torch.no_grad()
def collect(env, policy, teacher, steps: int, seed: int = 0):
    """(e (T, L, 2) m/s, mu (T, L)) -- the student's speed targets minus the teacher's, per step."""
    lid = env.learner_ids
    v_max = float(env.ecfg.v_max_policy)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    errs, mus = [], []
    for _ in range(steps):
        lab = env.teacher_label(teacher)
        act = policy(obs)
        # normalized plan speed a -> m/s, the same map `mpc.decode` uses before the cap
        e = (act[:, N_KNOTS:ACT_DIM] - lab[:, N_KNOTS:ACT_DIM]) * (0.5 * v_max)
        errs.append(e[lid].float().cpu())
        mus.append((env.sim.P["mu"] * env.sim.P["mu_f_scale"])[lid].float().cpu())
        obs, _r, term, trunc, _i = env.step(act)
        if hasattr(policy, "reset"):
            policy.reset(term | trunc)
    return torch.stack(errs), torch.stack(mus)


def bin_by_mu(err: torch.Tensor, mu: torch.Tensor, edges, over_margin: float = 0.0) -> list:
    """Per-bin over-speed statistics. `err` (T, L, 2), `mu` (T, L)."""
    e = err.reshape(-1, err.shape[-1]).numpy().astype(np.float64)
    m = np.repeat(mu.reshape(-1).numpy().astype(np.float64), 1)
    m = np.repeat(m[:, None], e.shape[1], 1).reshape(-1)
    e = e.reshape(-1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (m >= lo) & (m < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        x = e[sel]
        over = x[x > 0]
        under = x[x < 0]
        rows.append({
            "mu_lo": float(lo), "mu_hi": float(hi), "n": n,
            "over_rate": float((x > 0).mean()),
            "over_rate_beyond_margin": float((x > over_margin).mean()),
            "over_mean_mps": float(over.mean()) if over.size else 0.0,
            "over_p90_mps": float(np.percentile(over, 90)) if over.size else 0.0,
            "under_mean_mps": float(under.mean()) if under.size else 0.0,
            "mean_signed_mps": float(x.mean()),
        })
    return rows


def render(rows, meta) -> str:
    out = [f"student {meta['checkpoint']} | speed loss recorded as: {meta['speed_loss']}",
           "",
           "| mu | n | over-speed rate | over > margin | mean over [m/s] | p90 over | mean under | mean signed |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['mu_lo']:.2f}-{r['mu_hi']:.2f} | {r['n']} | {r['over_rate']:.1%} | "
                   f"{r['over_rate_beyond_margin']:.1%} | {r['over_mean_mps']:.3f} | "
                   f"{r['over_p90_mps']:.3f} | {r['under_mean_mps']:.3f} | {r['mean_signed_mps']:+.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tracks", default="eval")
    ap.add_argument("--envs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--race-size", type=int, default=3)
    ap.add_argument("--teacher-kind", default="interactive", choices=["raceline", "interactive"])
    ap.add_argument("--teacher-grip", default="true", choices=["true", "nominal", "conservative"])
    ap.add_argument("--teacher-speed", type=float, default=1.0)
    ap.add_argument("--over-margin", type=float, default=0.25, metavar="MPS",
                    help="[m/s] over-speed beyond which it is reported separately: a centimetre "
                         "over the limit is not the failure this measures")
    ap.add_argument("--bins", default="0.45,0.65,0.80,0.95,1.10,1.30")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    names = common.track_names(a.tracks)
    tracks, rls = common.load_tracks(names, racelines=True)
    model, extra = load_checkpoint(a.ckpt, a.device)
    model.eval()
    spec = extra.get("spec") or {}
    cfg = Config()
    if a.eager:
        cfg.sim.compile = False
    ecfg = EnvConfig(action_mode="plan", speed_cap=9.0, race_size=a.race_size, opponent="teacher",
                     opp_speed_range=(0.6, 1.15),
                     opp_events=("brake", "stop", "shift", "defend", "yield", "line", "oblivious"),
                     opp_event_rate=1.0, opp_defend_prob=0.3, opp_yield_prob=0.2,
                     opp_line_prob=0.3, opp_oblivious_prob=0.1,
                     scan_stack=spec.get("scan_stack", 6), scan_stride=spec.get("scan_stride", 1),
                     hist_len=spec.get("hist_len", 20), hist_stride=spec.get("hist_stride", 2),
                     opp_token=spec.get("opp_token", ""), compile_tracker=not a.eager)
    env = common.make_env(tracks, a.envs, a.device, ecfg, cfg=cfg, seed=a.seed, rls=rls,
                          teacher_grip=a.teacher_grip)
    env.sim.warmup()
    base = common.make_teacher(rls, env, grip=a.teacher_grip)
    base.speed_scale = a.teacher_speed
    teacher = InteractiveTeacher(base, env) if a.teacher_kind == "interactive" else base
    policy = memory_policy_fn(model, env.B, device=a.device, deterministic=True)
    err, mu = collect(env, policy, teacher, a.steps, seed=a.seed)
    edges = [float(x) for x in a.bins.split(",")]
    rows = bin_by_mu(err, mu, edges, a.over_margin)
    meta = {"checkpoint": a.ckpt, "speed_loss": extra.get("speed_loss", "unrecorded"),
            "speed_loss_constants": extra.get("speed_loss_constants"),
            "teacher_kind": a.teacher_kind, "teacher_grip": a.teacher_grip,
            "teacher_speed": a.teacher_speed, "over_margin": a.over_margin,
            "tracks": names, "envs": a.envs, "steps": a.steps, "seed": a.seed}
    print(render(rows, meta))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"meta": meta, "bins": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
