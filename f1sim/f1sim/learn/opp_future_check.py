"""Does the privileged opponent future say where the car actually went?

`gym_env.F1VecEnv.car_future` is a *label*, not a peek: it carries each car's own controller
forward (a teacher-driven car's raceline plan with its scheduled event applied, a policy-driven
car's plan-tracker trajectory) rather than stepping the simulator ahead. A label like that is only
worth putting into an observation or into a teacher's cost if it is closer to the realised future
than the free alternative, which is constant velocity. That is the whole content of this script.

Protocol. Roll the env forward; at every step record, for every car, the predicted displacement
over each horizon, and the pose it was predicted from. `h / control_dt` steps later, compare it
against where the car actually is. A sample is dropped when ANY car of that race ended its episode
in between (`race_boundary`): an opponent that crashes is respawned in place, and its position half
a second later is not the continuation of the motion anybody extrapolated.

Reported per horizon, per model, and separately for teacher-driven and policy-driven cars:

    MAE [m]   mean |predicted position - realised position|
    R^2       on the DISPLACEMENT in the car's own body frame at prediction time, the quantity the
              token actually encodes. Against raw world position every model scores ~1.0 because
              the track is large, which measures the track and not the model.

    python -m f1sim.learn.opp_future_check --tracks gen:control:1400 --steps 600
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Dict

import torch

from ..gym_env import OPP_FUTURE_MODELS, OPP_FUTURE_TIMES, EnvConfig
from . import common


def _r2(pred: torch.Tensor, real: torch.Tensor) -> float:
    """1 - SSE/SST over a flat pair of vectors."""
    if pred.numel() == 0:
        return float("nan")
    sse = ((pred - real) ** 2).sum()
    sst = ((real - real.mean()) ** 2).sum()
    return float(1.0 - sse / sst) if float(sst) > 0 else float("nan")


@torch.no_grad()
def measure(env, policy, steps: int, models=OPP_FUTURE_MODELS, times=OPP_FUTURE_TIMES,
            warmup: int = 20) -> Dict:
    dt = float(env.sim.control_dt)
    lag = [int(round(h / dt)) for h in times]
    if any(l < 1 for l in lag):
        raise ValueError(f"horizons {list(times)} s are shorter than one control step ({dt} s)")
    depth = max(lag) + 1
    hist = {m: [] for m in models}          # rings of (pred (B,K,2), pose (B,3), alive (B,))
    acc = {(m, i): {"pred": [], "real": [], "pose": [], "teacher": []} for m in models for i in range(len(times))}
    obs, _ = env.reset(seed=0)
    #: Control steps each row has run since the last race boundary. A prediction made l steps ago
    #: is usable exactly when this has reached l -- an opponent that crashed was respawned in place
    #: and nobody extrapolated the car that is there now.
    since = torch.zeros(env.B, dtype=torch.long, device=env.device)
    for t in range(steps):
        pose = env.sim.state[:, :3].clone()
        drv = env.teacher_driven.clone()
        for m in models:
            hist[m].append((env.car_future(times, model=m).clone(), pose, drv))
            if len(hist[m]) > depth:
                hist[m].pop(0)
        obs, _r, term, trunc, info = env.step(policy(obs))
        done = env.race_boundary(term | trunc)
        since = torch.where(done, torch.zeros_like(since), since + 1)
        if t < warmup:
            continue
        now = env.sim.state[:, :2]                      # the state l steps after hist[m][-l]
        for m in models:
            for i, l in enumerate(lag):
                if len(hist[m]) < l:
                    continue
                pred, p0, dr = hist[m][-l]
                ok = since >= l
                if not bool(ok.any()):
                    continue
                a = acc[(m, i)]
                a["pred"].append((pred[ok, i] - p0[ok, :2]).clone())
                a["real"].append((now[ok] - p0[ok, :2]).clone())
                a["pose"].append(p0[ok, 2].clone())
                a["teacher"].append(dr[ok].clone())
    out = {"horizons_s": list(times), "control_dt": dt, "steps": steps, "models": {}}
    for m in models:
        rows = []
        for i, h in enumerate(times):
            a = acc[(m, i)]
            if not a["pred"]:
                continue
            pred = torch.cat(a["pred"]); real = torch.cat(a["real"])
            yaw = torch.cat(a["pose"]); tch = torch.cat(a["teacher"])
            c, s = torch.cos(yaw), torch.sin(yaw)
            rot = lambda d: torch.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], 1)
            bp, br = rot(pred), rot(real)
            row = {"horizon_s": h, "n": int(pred.shape[0]),
                   "mae_m": float((pred - real).norm(dim=1).mean()),
                   "r2_lon": _r2(bp[:, 0], br[:, 0]), "r2_lat": _r2(bp[:, 1], br[:, 1]),
                   "moved_m": float(real.norm(dim=1).mean())}
            for name, mask in (("teacher", tch), ("policy", ~tch)):
                if int(mask.sum()) == 0:
                    continue
                row[name] = {"n": int(mask.sum()),
                             "mae_m": float((pred[mask] - real[mask]).norm(dim=1).mean()),
                             "r2_lon": _r2(bp[mask, 0], br[mask, 0]),
                             "r2_lat": _r2(bp[mask, 1], br[mask, 1])}
            rows.append(row)
        out["models"][m] = rows
    return out


def render(res: Dict) -> str:
    lines = ["| model | horizon [s] | n | mean travel [m] | MAE [m] | R2 lon | R2 lat | MAE teacher | MAE policy |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, rows in res["models"].items():
        for r in rows:
            g = lambda k: (f"{r[k]['mae_m']:.3f}" if k in r else "-")
            lines.append(f"| {m} | {r['horizon_s']:.2f} | {r['n']} | {r['moved_m']:.2f} | {r['mae_m']:.3f} | "
                         f"{r['r2_lon']:+.3f} | {r['r2_lat']:+.3f} | {g('teacher')} | {g('policy')} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="gen:control:1400")
    ap.add_argument("--envs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--race-size", type=int, default=3)
    ap.add_argument("--opponent", default="teacher", choices=["teacher", "policy"])
    ap.add_argument("--opp-events", default="brake,stop,shift,defend,yield,line,oblivious")
    ap.add_argument("--opp-event-rate", type=float, default=1.0)
    ap.add_argument("--opp-speed", type=float, nargs=2, default=(0.6, 1.15))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--times", default="",
                    help="comma-separated horizons [s]; default is gym_env.OPP_FUTURE_TIMES. Each "
                         "has to be a whole number of control steps")
    ap.add_argument("--models", default=",".join(OPP_FUTURE_MODELS))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from ..params import Config
    names = common.track_names(a.tracks)
    tracks, rls = common.load_tracks(names, racelines=True)
    ev = tuple(x for x in a.opp_events.split(",") if x)
    ec = EnvConfig(action_mode="plan", speed_cap=9.0, race_size=a.race_size, opponent=a.opponent,
                   opp_events=ev, opp_event_rate=a.opp_event_rate,
                   opp_speed_range=tuple(a.opp_speed), scan_stack=6, hist_len=20,
                   opp_defend_prob=0.3, opp_yield_prob=0.2, opp_line_prob=0.3, opp_oblivious_prob=0.1,
                   compile_tracker=not a.eager)
    cfg = Config()
    if a.eager:
        cfg.sim.compile = False
    env = common.make_env(tracks, a.envs, a.device, ec, cfg=cfg, seed=0, rls=rls)
    env.sim.warmup()
    teacher = common.make_teacher(rls, env)
    policy = lambda obs: env.teacher_label(teacher)
    times = tuple(float(x) for x in a.times.split(",") if x) or OPP_FUTURE_TIMES
    models = tuple(m for m in a.models.split(",") if m)
    res = measure(env, policy, a.steps, models=models, times=times)
    res["config"] = {"tracks": names, "envs": a.envs, "race_size": a.race_size,
                     "opponent": a.opponent, "opp_events": list(ev), "opp_event_rate": a.opp_event_rate}
    print(render(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
