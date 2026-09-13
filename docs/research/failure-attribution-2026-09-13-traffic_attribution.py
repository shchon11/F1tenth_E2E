"""Why traffic trials end: for every learner termination in a 2-car race with scripted opponent
events, record contact-vs-wall, where the opponent was (ahead/alongside/behind, lateral offset),
closing speed, whether the opponent was inside an event in the last second, and whether the
policy's own plan ran through the opponent's footprint."""
import sys, json, math, numpy as np, torch
from f1sim.learn.evaluate import load_checkpoint, common, EnvConfig, Config
from f1sim.learn.memory import policy_fn as memory_policy_fn
ck, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
device = "cuda"; L_CAR, W_CAR = 0.58, 0.31
TRACKS = ["real:map16x07", "gen:control:9100", "real:korea_2025_iccas"]
model, meta = load_checkpoint(ck, device); model.eval(); spec = meta.get("spec", {})
summary = {}
for name in TRACKS:
    trs, rls = common.load_tracks([name], racelines=True)
    e = EnvConfig(speed_cap=9.0, resample_track_on_reset=True, action_mode="plan", race_size=2, opponent="teacher",
                  opp_speed_range=(0.6, 0.8), opp_events=("brake", "stop", "shift"), opp_event_rate=3.0,
                  scan_stack=spec.get("scan_stack", 6), scan_stride=1, hist_len=spec.get("hist_len", 20), max_steps=1600)
    env = common.make_env(trs, 32, device, e, cfg=Config(), seed=4401, rls=rls); env.sim.warmup()
    obs, _ = env.reset(seed=4401); learner = env.learner.clone(); lidx = torch.nonzero(learner).flatten()
    policy = memory_policy_fn(model, env.B, device=device, deterministic=True); policy.reset()
    other = env.sim.other_idx[:, 0]
    recent_event = torch.zeros(env.B, device=device)      # steps since the opponent's last active event step
    events = []; done_once = torch.zeros(env.B, dtype=torch.bool, device=device)
    def in_event(info):
        oe = info.get("opp_event") or {}
        for k, v in oe.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.bool and v.shape[0] == env.B: return v
            if isinstance(v, torch.Tensor) and v.shape[0] == env.B and k in ("active", "kind"): return v != 0
        return torch.zeros(env.B, dtype=torch.bool, device=device)
    for t in range(1400):
        st = env.sim.state.clone()
        with torch.no_grad(): a = policy(obs)
        ref = getattr(env.tracker, "last_ref", None)
        obs, _, term, trunc, info = env.step(a); policy.reset(term | trunc)
        ev_now = in_event(info)
        recent_event = torch.where(ev_now[other], torch.zeros_like(recent_event), recent_event + 1)
        car_hit = env.sim.car_collision.clone() if env.sim.car_collision is not None else torch.zeros(env.B, dtype=torch.bool, device=device)
        new = term & learner & ~done_once
        for k in torch.nonzero(new).flatten().tolist():
            o = int(other[k]); me, op = st[k], st[o]
            d = op[:2] - me[:2]; c, s = math.cos(float(me[2])), math.sin(float(me[2]))
            ahead = float(d[0] * c + d[1] * s); lat = float(-d[0] * s + d[1] * c)
            rel = "ahead" if ahead > L_CAR else ("behind" if ahead < -L_CAR else "alongside")
            plan_hit = None
            if ref is not None and ref.dim() == 3:
                px = me[0] + ref[k, :, 0] * c - ref[k, :, 1] * s; py = me[1] + ref[k, :, 0] * s + ref[k, :, 1] * c
                dx, dy = px - op[0], py - op[1]; co, so = math.cos(float(op[2])), math.sin(float(op[2]))
                u = dx * co + dy * so; v = -dx * so + dy * co
                plan_hit = bool(((u.abs() <= L_CAR / 2 + 0.05) & (v.abs() <= W_CAR / 2 + 0.05)).any())
            events.append({"t": t, "contact": bool(car_hit[k]) or (math.hypot(float(d[0]), float(d[1])) < 0.7), "rel": rel, "ahead_m": ahead, "lat_m": lat,
                           "v_me": float(me[3]), "v_opp": float(op[3]), "closing": float(me[3] - op[3]), "opp_event_within_1s": bool(recent_event[k] < 40), "plan_through_opponent": plan_hit})
        done_once |= term | trunc
        if bool(done_once[lidx].all()): break
    n = len(events); s = {"learner_trials": int(learner.sum()), "terminations": n}
    if n:
        s["contacts"] = sum(e["contact"] for e in events); s["walls"] = n - s["contacts"]
        for r in ("ahead", "alongside", "behind"): s[f"contact_{r}"] = sum(e["contact"] and e["rel"] == r for e in events)
        s["contact_after_event_1s"] = sum(e["contact"] and e["opp_event_within_1s"] for e in events)
        s["contact_plan_through_opp"] = sum(e["contact"] and e["plan_through_opponent"] for e in events)
        s["mean_closing_at_contact"] = float(np.mean([e["closing"] for e in events if e["contact"]] or [0]))
        s["wall_alongside_or_ahead"] = sum((not e["contact"]) and e["rel"] in ("alongside", "behind") for e in events)  # learner passing/being near
        s["events"] = events
    summary[name] = s; print(name, json.dumps({k: v for k, v in s.items() if k != "events"}), flush=True)
    del env; torch.cuda.empty_cache()
json.dump({"ckpt": ck, "tag": tag, "tracks": summary}, open(out, "w"), indent=1)
