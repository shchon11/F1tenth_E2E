"""The open-loop label-quality gates of the output-format design (docs/research/output-format-
2026-10-14.md section 7): test 2, the noise bias, and test 5, the low-speed authority. The closed-loop
gates (3, 4, 6) are `wp4_label_replay.py`. Pre-registration and results:
docs/research/wp4-label-gates-2026-10-14/.

Both start from states the teacher itself visits: a wp4 label replay (production labels, i.e. the
raceline teacher behind the env's guard) on ICCAS, with the tracker's inputs captured exactly as
`F1VecEnv.step` hands them to `PlanTracker` -- the measured speed, the cap, the gyro, the calibrated
delay, the last command and the shifted warm start -- and then solved offline with `mpc.solve`.

    python f1sim/scripts/wp4_label_gates.py noise-bias --out bias.json      # test 2
    python f1sim/scripts/wp4_label_gates.py authority --out auth.json       # test 5

noise-bias: 464 states (8 envs, every 20 steps from step 50 of 1200) x 16 draws of PPO's initial
noise per dimension (wp4: kappa_1 / kappa_2 at log_std -2.07, the rest -1.8). The bias is the mean
executed v_cmd with the decoder's backward pass minus without it; the steering noise is the rms of
the executed first steer around the label's, against linear's at its own -1.8 around the linear
teacher's label at the same states and tracker inputs.

authority: up to 512 states whose measured speed is 1-2 m/s (they come from the spawns, which start
at 0-3 m/s); each dimension moved +-0.05, and how far that moves the tracker's predicted state at
the end of its window -- lateral position [m/unit] and speed [m/s/unit] -- plus the executed steer
[deg/unit] and v_cmd [m/s/unit], the columns of the design's authority table.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from f1sim import mpc
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.params import Config

LIMITS = dict(a_lat=7.0, a_acc=6.5, a_brake=4.0)
LOG_STD = {"wp4": (-2.07, -2.07) + (-1.8,) * 6, "linear": (-1.8,) * 8}


def make(track, envs, seed, device, steps):
    trs, rls = common.load_tracks([track], racelines=True, objective="min_time", **LIMITS)
    cfg = Config(); cfg.sim.compile = False
    env = common.make_env(trs, envs, device, EnvConfig(speed_cap=9.0, action_mode="plan", speed_mode="wp4",
                                                       max_steps=steps + 10, laps=10 ** 6, collision_mode="soft",
                                                       compile_tracker=False), cfg=cfg, seed=seed)
    return env, common.make_teacher(rls, env, **LIMITS)


def tracker_inputs(env):
    """What `F1VecEnv.step` will hand the tracker for this step's plan (`PlanTracker.__call__`)."""
    tr, sp, lr = env.tracker, env.tracker.spec, env.last_result
    v = env._tracker_plan_speed()
    yaw = (lr.imu[:, :, 2].mean(1) if (lr is not None and lr.imu is not None and lr.imu.shape[1] > 0)
           else v.abs() * torch.tan(tr.u_prev[:, 0]) / (tr.wb + sp.k_us * v * v))
    d = env.tracker_delay
    delay = torch.full_like(v, sp.delay) if d is None else (d if torch.is_tensor(d) else torch.full_like(v, float(d)))
    warm = torch.cat([tr.u_seq[:, 1:], tr.u_seq[:, -1:]], 1)
    return dict(v=v.clone(), cap=env.speed_cap.clone(), yaw=yaw.clone(), delay=delay.clone(), u_prev=tr.u_prev.clone(),
                warm=warm.clone())


def command(env, a, inp, spec):
    """(steer [rad], v_cmd [m/s], predicted states (B, N+1, 6)) of plan `a` from captured inputs."""
    tr = env.tracker
    u, z, _ = mpc.solve(a, inp["v"], inp["cap"], inp["yaw"], inp["delay"], inp["u_prev"], inp["warm"], spec, tr.wb,
                        tr.s_max, tr.v_max)
    k_ = max(1, int(round(spec.v_cmd_lead / spec.dt)))
    return u[:, 0, 0], torch.minimum(z[:, k_, 3], inp["cap"]), z


def _cat(dicts):
    return {k: torch.cat([d[k] for d in dicts], 0) for k in dicts[0]}


def _no_backward_pass(action, v_meas, v_max, speed_cap, spec):
    """`mpc.decode_wp4` without its backward braking pass: the reference the bias is measured against."""
    lay = mpc.plan_layout("wp4")
    a = action.clamp(-1.0, 1.0)
    return (a[:, lay.path] * mpc.kappa_scale(v_meas, spec), mpc.plan_length(v_meas, spec),
            torch.minimum((a[:, lay.speed] + 1.0) * 0.5 * v_max, speed_cap[:, None]))


@torch.no_grad()
def noise_bias(a):
    env, t = make(a.track, 8, a.seed, a.device, a.steps)
    spec_w, spec_l = env.tracker.spec, mpc.PlanSpec()
    vmax = env.ecfg.v_max_policy
    env.reset()
    inputs, lab_w, lab_l = [], [], []
    for k in range(a.steps):
        lab = env.teacher_label(t)
        if k % 20 == 10 and k > 40:
            inputs.append(tracker_inputs(env))
            lab_w.append(lab.clone())
            lab_l.append(t.plan_action(env.sim.state, env.sim.P, env.sim.tid, vmax, spec_l,
                                       plan_speed=env._tracker_plan_speed()))
        env.step(lab)
    inp = _cat(inputs); aw = torch.cat(lab_w); al = torch.cat(lab_l)
    S, R = aw.shape[0], a.draws
    g = torch.Generator(device=env.device).manual_seed(a.seed)
    rep = lambda x: x.repeat(R, *([1] * (x.dim() - 1)))
    inpR = {k: rep(v) for k, v in inp.items()}
    out = {"states": S, "draws": R, "track": a.track, "seed": a.seed}
    s0_w, v0_w, _ = command(env, aw, inp, spec_w)
    s0_l, v0_l, _ = command(env, al, inp, spec_l)
    for name, sig in (("wp4_per_dim", LOG_STD["wp4"]), ("wp4_flat_-1.8", LOG_STD["linear"])):
        sigma = torch.tensor(sig, device=env.device).exp()
        an = rep(aw) + sigma * torch.randn(R * S, 8, generator=g, device=env.device)
        s_n, v_n, _ = command(env, an, inpR, spec_w)
        held = mpc.decode_wp4
        mpc.decode_wp4 = _no_backward_pass
        try:
            _, v_nb, _ = command(env, an, inpR, spec_w)
        finally:
            mpc.decode_wp4 = held
        dv = v_n - v_nb
        ds = torch.rad2deg(s_n - rep(s0_w))
        out[name] = {"vcmd_bias_mean": float(dv.mean()), "vcmd_bias_p10": float(dv.quantile(0.10)),
                     "share_binding": float((dv.abs() > 1e-4).float().mean()),
                     "steer_noise_rms_deg": float(ds.square().mean().sqrt())}
    sigma = torch.tensor(LOG_STD["linear"], device=env.device).exp()
    an = rep(al) + sigma * torch.randn(R * S, 8, generator=g, device=env.device)
    s_n, _, _ = command(env, an, inpR, spec_l)
    out["linear"] = {"steer_noise_rms_deg": float(torch.rad2deg(s_n - rep(s0_l)).square().mean().sqrt())}
    out["steer_noise_ratio"] = out["wp4_per_dim"]["steer_noise_rms_deg"] / out["linear"]["steer_noise_rms_deg"]
    out["pass_bias"] = out["wp4_per_dim"]["vcmd_bias_mean"] >= -0.10
    out["pass_steer"] = out["steer_noise_ratio"] <= 1.1
    return out


@torch.no_grad()
def authority(a):
    env, t = make(a.track, 1024, a.seed, a.device, 400)
    spec = env.tracker.spec
    env.reset()
    states, labels = [], []
    n = 0
    for k in range(400):
        lab = env.teacher_label(t)
        inp = tracker_inputs(env)
        m = (inp["v"].abs() >= 1.0) & (inp["v"].abs() <= 2.0)
        if bool(m.any()):
            states.append({kk: vv[m] for kk, vv in inp.items()}); labels.append(lab[m].clone())
            n += int(m.sum())
        env.step(lab)
        if n >= 4 * a.max_states:
            break
    inp = _cat(states); lab = torch.cat(labels)
    g = torch.Generator(device=env.device).manual_seed(a.seed)
    pick = torch.randperm(lab.shape[0], generator=g, device=env.device)[:a.max_states]
    inp = {k: v[pick] for k, v in inp.items()}; lab = lab[pick]
    S, D, dl = lab.shape[0], lab.shape[1], 0.05
    rows = []
    for d in range(D):
        up, dn = lab.clone(), lab.clone()
        up[:, d] = (up[:, d] + dl).clamp(-1, 1); dn[:, d] = (dn[:, d] - dl).clamp(-1, 1)
        da = up[:, d] - dn[:, d]
        ok = da > 1e-6
        s_u, v_u, z_u = command(env, up, inp, spec)
        s_d, v_d, z_d = command(env, dn, inp, spec)
        per = lambda x: float((x.abs()[ok] / da[ok]).mean()) if bool(ok.any()) else 0.0
        rows.append({"dim": d, "states": int(ok.sum()),
                     "end_lateral_m_per_unit": per(z_u[:, -1, 1] - z_d[:, -1, 1]),
                     "end_speed_mps_per_unit": per(z_u[:, -1, 3] - z_d[:, -1, 3]),
                     "steer_deg_per_unit": per(torch.rad2deg(s_u - s_d)),
                     "vcmd_mps_per_unit": per(v_u - v_d)})
    for r in rows:
        r["pass"] = r["end_lateral_m_per_unit"] >= 0.1 or r["end_speed_mps_per_unit"] >= 0.1
    return {"states": S, "speed_mean": float(inp["v"].abs().mean()), "track": a.track, "seed": a.seed,
            "dims": rows, "pass": all(r["pass"] for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gate", choices=("noise-bias", "authority"))
    ap.add_argument("--track", default="real:korea_2025_iccas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1200, help="noise-bias: replay length (464 states at 1200)")
    ap.add_argument("--max-states", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(2)
    t0 = time.time()
    res = noise_bias(a) if a.gate == "noise-bias" else authority(a)
    res["wall_s"] = time.time() - t0
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
