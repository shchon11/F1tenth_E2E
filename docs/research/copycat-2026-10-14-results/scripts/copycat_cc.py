"""Copycat probe for the copycat study (adapted from v2smoke/copycat.py to read each checkpoint's own
observation contract: the column table, `obs.ego_columns`, says where the issued actions and the
measured speeds are, so the probe runs on the full contract and on arm A's 42-column one alike).

On a held-out buffer saved by heldout_bc.py (HELD_SAVE), every checkpoint is run on the same windows
(seed 1234; 8 batches x 48 windows of 120 steps, burn-in 40) with its ego vector
  * recorded                    -- as stored;
  * zero_actions                -- every issued-action column zeroed (full contract only);
  * teacher_history             -- every issued-action column replaced by the teacher's label at the
                                   same lag within the episode (full contract only);
  * act_speed_plus_0.5mps       -- every issued SPEED column (wp4 dims 4..7, a_{t-2} and the history)
                                   +0.1 normalised = +0.5 m/s (full contract only);
  * meas_speed_hist_plus_0.5mps -- every MEASURED speed column of the history (lag >= 1) +0.05
                                   normalised = +0.5 m/s (both contracts; speed@t, which the decode
                                   lays the plan out from, is left alone).
Reported per checkpoint and variant: speed bias (student - label) and MAE of the decoded WP4 speeds,
lateral MAE, the mean decoded output speed; and the gains d(output speed) / 0.5 m/s.
usage: python copycat_cc.py BUF.pt OUT.json CKPT [CKPT ...]"""
import json
import sys

import torch

from f1sim.gym_env import EnvConfig
from f1sim.learn.model import load_checkpoint
from f1sim.learn.obs import ego_columns
from f1sim.learn.train_v2 import wp4_decoded
from f1sim.mpc import PlanSpec, plan_layout

buf_path, out_path, ckpts = sys.argv[1], sys.argv[2], sys.argv[3:]
dev = torch.device("cuda")
D = torch.load(buf_path, map_location="cpu")
T, B = D["scan"].shape[:2]
_e = EnvConfig()
PSPEC = PlanSpec(speed_mode="wp4", a_brake_profile=float(_e.plan_a_brake), kappa_mode=_e.plan_kappa_mode,
                 kappa_a_lat=float(_e.plan_kappa_a_lat))
SPEED_DIMS = set(range(plan_layout("wp4").speed.start, plan_layout("wp4").speed.stop))
newep = D["newep"].bool()
ep0 = torch.zeros(T, B, dtype=torch.long)
cur = torch.zeros(B, dtype=torch.long)
for t in range(T):
    cur = torch.where(newep[t], torch.full_like(cur, t), cur)
    ep0[t] = cur
torch.manual_seed(1234)
L, burn, m, nb = 120, 40, 48, 8
windows = [(torch.randint(T - L + 1, (m,)), torch.randint(B, (m,))) for _ in range(nb)]
res = {}
for ck in ckpts:
    model, extra = load_checkpoint(ck, dev)
    model.eval()
    spec = model.obs_spec
    if D["ego"].shape[-1] != spec.ego_dim:
        raise SystemExit(f"{ck}: ego {spec.ego_dim} wide, the buffer's is {D['ego'].shape[-1]}")
    cols = ego_columns(spec)
    act = [c for c in cols if c.quantity.startswith("action[")]
    act_speed = [c.index for c in act if int(c.quantity[7:-1]) in SPEED_DIMS]
    meas_hist = [c.index for c in cols if c.quantity == "speed" and c.lag > 0]
    by_lag = {}
    for c in act:
        by_lag.setdefault(c.lag, []).append(c.index)
    variants = ["recorded", "meas_speed_hist_plus_0.5mps"]
    if act:
        variants += ["zero_actions", "teacher_history", "act_speed_plus_0.5mps"]
    rows = {k: [] for k in variants}
    for t0, b in windows:
        ts = t0[None, :] + torch.arange(L)[:, None]
        bs = b[None, :].expand(L, m)
        scan = D["scan"][ts, bs].float()[:, :, None, :]
        ego = D["ego"][ts, bs].float()
        lab = D["label"][ts, bs].float()
        c = D["c"][ts, bs].float()
        keep = (~newep[ts, bs]).float()
        valid = D["valid"][ts, bs].bool()
        egos = {"recorded": ego}
        e = ego.clone()
        e[..., meas_hist] = (ego[..., meas_hist] + 0.05)
        egos["meas_speed_hist_plus_0.5mps"] = e
        if act:
            e = ego.clone(); e[..., [c.index for c in act]] = 0.0
            egos["zero_actions"] = e
            e = ego.clone(); e[..., act_speed] = (ego[..., act_speed] + 0.1).clamp(-1, 1)
            egos["act_speed_plus_0.5mps"] = e
            e = ego.clone()
            for lag, idx in by_lag.items():
                tl = ts - lag
                ok = (tl >= ep0[ts, bs]) & (tl >= 0)
                sub = D["label"][tl.clamp(min=0), bs].float()                 # (L, m, 8), in column order
                e[..., idx] = torch.where(ok[..., None], sub, ego[..., idx])
            egos["teacher_history"] = e
        sl = slice(burn, None)
        labf = lab[sl].reshape(-1, lab.shape[-1]).to(dev)
        vm = (ego[sl][..., 0] * spec.v_max).reshape(-1).to(dev)
        vf = valid[sl].reshape(-1).to(dev)
        yt, vt = wp4_decoded(labf, vm, PSPEC, spec.v_max)
        for name, eg in egos.items():
            with torch.no_grad():
                out = model.actor.sequence(scan.to(dev), eg.to(dev), c.to(dev), keep=keep.to(dev), burn_in=burn)
            mu = out["mu"].reshape(-1, lab.shape[-1])
            ys, vs = wp4_decoded(mu, vm, PSPEC, spec.v_max)
            rows[name].append({"bias_v": float((vs - vt).mean(-1)[vf].mean()),
                               "mae_v": float((vs - vt).abs().mean(-1)[vf].mean()),
                               "mae_lat": float((ys - yt).abs().mean(-1)[vf].mean()),
                               "v_student": float(vs.mean(-1)[vf].mean()), "v_label": float(vt.mean(-1)[vf].mean())})
    r = {k: {q: sum(x[q] for x in v) / len(v) for q in v[0]} for k, v in rows.items()}
    base = r["recorded"]["v_student"]
    r["gains"] = {"meas_speed_hist": (r["meas_speed_hist_plus_0.5mps"]["v_student"] - base) / 0.5}
    if act:
        r["gains"]["act_speed_hist"] = (r["act_speed_plus_0.5mps"]["v_student"] - base) / 0.5
    r["contract"] = {"ego_dim": spec.ego_dim, "ego_actions": bool(spec.ego_actions), "n_action_cols": len(act),
                     "n_act_speed_cols": len(act_speed), "n_meas_hist_cols": len(meas_hist), "iter": extra.get("iter")}
    res[ck] = r
    print(ck.split("/")[-1], json.dumps({k: ({q: round(x, 3) for q, x in v.items()} if isinstance(v, dict) else v)
                                         for k, v in r.items()}), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
