"""Where and why the policy fails on the held-out proxy tracks: for every terminated trial record
speed, mu, what was hit (added obstacle vs original wall vs car), and whether the policy's own last
plan already intersected the obstacle (policy error) or was clear (tracking/dynamics error)."""
import sys, json, numpy as np, torch
from f1sim.learn.evaluate import load_checkpoint, common, EnvConfig, Config
from f1sim.learn.memory import policy_fn as memory_policy_fn
from f1sim import maps
ck, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
device = "cuda"
TRACKS = ["scene:scene_0912_2355", "scene:scene_0912_2355+hard1", "real:map16x07+hard2", "real:map12x16+hard3",
          "gen:control:9100+hard4", "real:korea_2025_iccas+hard5", "real:map16x07", "real:map12x16"]
model, meta = load_checkpoint(ck, device); model.eval(); spec = meta.get("spec", {})
summary = {}
for name in TRACKS:
    trs, rls = common.load_tracks([name])
    tr = trs[0]; base = maps.load(name.split("+hard")[0]) if "+hard" in name else tr
    added = torch.tensor(tr.occupancy & ~base.occupancy, device=device)
    occ = torch.tensor(tr.occupancy, device=device); edt_t = torch.tensor(tr.edt, device=device)
    e = EnvConfig(speed_cap=9.0, resample_track_on_reset=True, action_mode="plan", race_size=1,
                  scan_stack=spec.get("scan_stack", 6), scan_stride=1, hist_len=spec.get("hist_len", 20), max_steps=1600)
    env = common.make_env(trs, 32, device, e, cfg=Config(), seed=4401, rls=rls); env.sim.warmup()
    obs, _ = env.reset(seed=4401)
    res, ox, oy = tr.resolution, tr.origin[0], tr.origin[1]
    def cell(xy):
        j = ((xy[:, 0] - ox) / res).long().clamp(0, occ.shape[1] - 1); i = ((xy[:, 1] - oy) / res).long().clamp(0, occ.shape[0] - 1); return i, j
    events = []; done_once = torch.zeros(32, dtype=torch.bool, device=device); steps = 0
    policy = memory_policy_fn(model, env.B, device=device, deterministic=True)
    policy.reset()
    for t in range(1200):
        st = env.sim.state.clone(); mu = env.sim.P["mu"].clone()
        with torch.no_grad():
            a = policy(obs)
        ref = getattr(env.tracker, "last_ref", None)
        plan_clear = None
        if ref is not None and ref.dim() == 3:
            c, s = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
            px = st[:, 0:1] + ref[:, :, 0] * c - ref[:, :, 1] * s; py = st[:, 1:2] + ref[:, :, 0] * s + ref[:, :, 1] * c
            i, j = cell(torch.stack([px.reshape(-1), py.reshape(-1)], 1)); hit = occ[i, j].view(px.shape)
            plan_clear = ~hit.any(1)
            plan_min_clear = (edt_t[i, j].view(px.shape) - 0.14).min(1).values          # body-edge clearance along the plan
            plan_reach = torch.linalg.vector_norm(ref[:, -1, :2], dim=1)                # how far the plan reaches [m]
            car_clear = edt_t[cell(st[:, :2])] - 0.14
        obs, _, term, trunc, info = env.step(a)
        policy.reset(term | trunc)
        new = term & ~done_once
        for k in torch.nonzero(new).flatten().tolist():
            # what is within 0.35 m of the car centre at impact
            xy = st[k, :2]; r = int(0.35 / res)
            i, j = cell(xy[None]); i, j = int(i), int(j)
            win_a = added[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1].any().item(); win_w = occ[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1].any().item()
            events.append({"speed": float(st[k, 3]), "mu": float(mu[k]), "hit": "obstacle" if win_a else ("wall" if win_w else "unknown"),
                           "plan_clear": (None if plan_clear is None else bool(plan_clear[k])), "t": t,
                           "plan_min_clear": (None if plan_clear is None else float(plan_min_clear[k])),
                           "plan_reach": (None if plan_clear is None else float(plan_reach[k])),
                           "car_clear_before": (None if plan_clear is None else float(car_clear[k]))})
        done_once |= term | trunc
        if done_once.all(): break
    n = len(events); s = {"trials": 32, "collisions": n, "completed_or_timeout": 32 - n}
    if n:
        s["hit_obstacle"] = sum(e["hit"] == "obstacle" for e in events); s["hit_wall"] = sum(e["hit"] == "wall" for e in events)
        s["mean_speed_at_impact"] = float(np.mean([e["speed"] for e in events])); s["mean_mu_at_impact"] = float(np.mean([e["mu"] for e in events]))
        s["low_mu_share"] = float(np.mean([e["mu"] < 0.85 for e in events]))
        pc = [e["plan_clear"] for e in events if e["plan_clear"] is not None]
        s["plan_was_clear_share"] = (float(np.mean(pc)) if pc else None)
        for key in ("plan_min_clear", "plan_reach", "car_clear_before"):
            vals = [e[key] for e in events if e.get(key) is not None]
            if vals: s[key + "_median"] = float(np.median(vals)); s[key + "_p25"] = float(np.percentile(vals, 25))
        s["events"] = events
    summary[name] = s; print(name, json.dumps(s), flush=True)
    del env; torch.cuda.empty_cache()
json.dump({"ckpt": ck, "tag": tag, "tracks": summary}, open(out, "w"), indent=1)
