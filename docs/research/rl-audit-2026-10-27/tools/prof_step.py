"""Profile one PPO v2 rollout step of the current recipe, component by component.

Runs `ppo.main` with the given argv, but replaces `ppo_v2.train` by a loop that drives the env like the trainer's
rollout (actor sample, critic, reference actor, privileged/aux label reads, env.step) and times each piece with
cuda synchronisation around it. Inner wrappers are installed only after warm-up steps (CUDA graphs are captured
before / during the first steps).

usage: python prof_step.py WARMUP MEASURE -- <ppo.py argv...>
"""
import collections
import sys
import time

import torch

WARM, MEAS = int(sys.argv[1]), int(sys.argv[2])
argv = sys.argv[sys.argv.index("--") + 1:]
sys.argv = ["ppo"] + argv

from f1sim.learn import ppo, ppo_v2  # noqa: E402
from f1sim.learn import conditioning as cond_mod  # noqa: E402
from f1sim.learn.model import load_checkpoint  # noqa: E402
from f1sim.learn.memory import reset_hidden  # noqa: E402

T_ = collections.defaultdict(float)
N_ = collections.defaultdict(int)
DEPTH = [0]


def timed(label, fn):
    def w(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        DEPTH[0] += 1
        try:
            return fn(*a, **k)
        finally:
            DEPTH[0] -= 1
            torch.cuda.synchronize()
            T_[label] += time.perf_counter() - t0
            N_[label] += 1
    return w


def wrap_attr(obj, name, label):
    if obj is None or not hasattr(obj, name):
        print(f"  (no {label})")
        return
    setattr(obj, name, timed(label, getattr(obj, name)))


def train(a, *, env, obs, device, names, init_ck, controller, graph_rt, applied, league_seats=()):
    model, extra = load_checkpoint(a.init, device)
    actor = model.actor
    import copy
    ref = copy.deepcopy(actor).eval()
    lid = env.learner_ids
    held = getattr(a, "dial_constant", None)
    dial = cond_mod.ConstantDial(env, held) if held is not None else cond_mod.DialDraw(env, a.dial_margin, a.dial_exact)
    cspec = cond_mod.CondSpec.from_meta(model.meta["cond"])
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp)
    h = actor.initial_hidden(env.B, device=device)
    h_ref = ref.initial_hidden(env.B, device=device)
    env.sim.warmup()
    print(f"B={env.B} learners={lid.numel()} M={env.M}", flush=True)
    print("opp kinds:", getattr(env, "alt_teacher_kinds", None), "pool entries:",
          len(env.pool.entries) if getattr(env, "pool", None) is not None else 0, flush=True)

    def one_step(t_label=True):
        nonlocal obs, h, h_ref
        c = cond_mod.mu_to_c(dial.value(), cspec).to(obs["ego"].dtype)
        priv = env.privileged_v2()
        with torch.no_grad(), ac:
            d, hn = actor.step_dist(obs["scan"], obs["ego"], c, h)
            act = d.sample()
            v, lg = model.critic(obs["scan"][lid], obs["ego"][lid], c[lid], priv[lid])
            mu_ref, h_ref = ref.step(obs["scan"], obs["ego"], c, h_ref)
        if actor.b8 is not None:
            env.opponent_tracks_v2()
        if actor.near_car is not None:
            env.nearest_visible_car_v2()
        obs, rew, term, trunc, info = env.step(act.clamp(-1, 1))
        done = term | trunc
        h = reset_hidden(hn.float(), done)
        h_ref = reset_hidden(h_ref, done)
        if "final" in info:
            fo = info["final_obs"]; ids = info["final"]["ids"]
            with torch.no_grad(), ac:
                model.critic(fo["scan"], fo["ego"], c[ids], info["final_priv_v2"])

    for i in range(WARM):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(MEAS):
        one_step()
    torch.cuda.synchronize()
    plain = (time.perf_counter() - t0) / MEAS
    print(f"\nPLAIN (no inner sync): {plain*1e3:.1f} ms/step -> {lid.numel()/plain:.0f} learner samples/s, "
          f"{env.B/plain:.0f} car-steps/s", flush=True)

    # ---- install the wrappers
    actor.step_dist = timed("actor.step_dist", actor.step_dist)
    ref.step = timed("ref.step (KL monitor)", ref.step)
    model.critic.forward = timed("critic", model.critic.forward)
    wrap_attr(env, "privileged_v2", "env.privileged_v2")
    wrap_attr(env, "opponent_tracks_v2", "env.opponent_tracks_v2 (B8 labels)")
    wrap_attr(env, "nearest_visible_car_v2", "env.nearest_visible_car_v2 (near labels)")
    wrap_attr(env, "step", "env.step [total]")
    wrap_attr(env, "_opponent_actions", "  env._opponent_actions [total]")
    if getattr(env, "pool", None) is not None:
        wrap_attr(env.pool, "act", "    pool.act (league nets)")
    wrap_attr(env, "_opponent_obs", "    env._opponent_obs")
    if getattr(env, "events", None) is not None:
        wrap_attr(env.events, "step", "    events.step")
        wrap_attr(env.events, "step_reactive", "    events.step_reactive")
    wrap_attr(env, "follow_cap", "    env.follow_cap")
    orig_tn = env._teacher_normalized

    def tn(teacher, *a_, **k_):
        return timed(f"    teacher {type(teacher).__name__}", orig_tn)(teacher, *a_, **k_)
    env._teacher_normalized = tn
    tr = env.tracker
    if tr is not None:
        cls = type(tr)
        cls.__call__ = timed(f"  tracker ({cls.__name__})", cls.__call__)
    wrap_attr(env, "_tracker_plan_speed", "  env._tracker_plan_speed")
    wrap_attr(env, "_capture_plan", "  env._capture_plan")
    wrap_attr(env.sim, "step", "  sim.step (physics+lidar)")
    wrap_attr(env.sim, "_roll", "    sim._roll (substeps)")
    wrap_attr(env.sim, "_car_contacts", "    sim._car_contacts")
    wrap_attr(env.sim, "_car_boxes", "    sim._car_boxes")
    wrap_attr(env.sim, "_post", "    sim._post")
    wrap_attr(env.sim.lidar, "scan", "    lidar.scan")
    wrap_attr(env.sim.lidar, "trace", "      lidar.trace (walls+props)")
    wrap_attr(env.sim.lidar, "_merge_props", "        lidar._merge_props")
    wrap_attr(env.sim.lidar, "_post", "      lidar._post (noise, eager)")
    wrap_attr(env.sim.lidar, "_rays", "      lidar._rays")
    import f1sim.lidar as L_
    L_.ray_slices_hits = timed("      ray_slices_hits (car mesh)", L_.ray_slices_hits)
    L_.ray_box_hits = timed("      ray_box_hits (rear boxes)", L_.ray_box_hits)
    L_.trace3d_triton = timed("        trace3d_triton (walls)", L_.trace3d_triton)
    wrap_attr(env.sim, "_prop_contact", "    sim._prop_contact")
    wrap_attr(env.sim, "_update_contact_frame", "    sim._update_contact_frame")
    wrap_attr(env, "_severity_gate", "  env._severity_gate")
    wrap_attr(env, "_car_charge", "  env._car_charge")
    wrap_attr(env, "_math", "  env._math (reward)")
    wrap_attr(env, "_obs", "  env._obs")
    wrap_attr(env, "_plan_clearance", "  env._plan_clearance")
    wrap_attr(env, "_widen_clearance_to_props", "  env._widen_clearance_to_props")
    if getattr(env, "procedural", None) is not None:
        wrap_attr(env.procedural, "advance", "  procedural.advance")
    wrap_attr(env, "_sector_time_reward", "  env._sector_time_reward")
    wrap_attr(env, "_reset_envs", "  env._reset_envs")
    wrap_attr(env, "_seat_builders", "  env._seat_builders(call)")
    for k in list(T_):
        T_.pop(k); N_.pop(k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(MEAS):
        one_step()
    torch.cuda.synchronize()
    tot = (time.perf_counter() - t0) / MEAS
    print(f"\nWITH SYNC WRAPPERS: {tot*1e3:.1f} ms/step", flush=True)
    for k, v in sorted(T_.items(), key=lambda kv: -kv[1]):
        print(f"{k:55s} {1e3*v/MEAS:8.2f} ms/step  ({N_[k]/MEAS:.2f} calls/step)")
    print(f"mem allocated {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    raise SystemExit(0)


ppo_v2.train = train
ppo.main()
