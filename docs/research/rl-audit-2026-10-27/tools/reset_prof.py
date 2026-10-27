"""Time env resets (level replay on) and their parts, on a real race env. usage: python reset_prof.py WARM STEPS -- <ppo argv>"""
import collections, sys, time
import torch
WARM, STEPS = int(sys.argv[1]), int(sys.argv[2])
argv = sys.argv[sys.argv.index("--") + 1:]
sys.argv = ["ppo"] + argv
from f1sim.learn import ppo, ppo_v2  # noqa: E402
T_ = collections.defaultdict(float); N_ = collections.defaultdict(int)
def timed(label, fn):
    def w(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize(); T_[label] += time.perf_counter() - t0; N_[label] += 1
    return w
def wrap(obj, name, label):
    if obj is not None and hasattr(obj, name):
        setattr(obj, name, timed(label, getattr(obj, name)))
def train(a, *, env, obs, device, names, init_ck, controller, graph_rt, applied, league_seats=()):
    act = torch.zeros(env.B, env.act_dim, device=device)
    for _ in range(WARM):
        env.step(act)
    wrap(env, "step", "env.step")
    wrap(env, "_reset_envs", "_reset_envs [total]")
    wrap(env, "_reset_envs_core", "  _reset_envs_core")
    for n in ("_redraw_layouts", "_spawn_order", "_slot_speeds", "_slot_grid", "_seat_builders", "_row_format_codes",
              "_opp_format_rows", "_note_spawn_formation", "_fill_scans", "_calibrate_tracker", "_calibrate_external"):
        wrap(env, n, "    " + n)
    for n in ("reset", "sample_spawn", "_spawn_in_prop", "park", "unpark"):
        wrap(env.sim, n, "    sim." + n)
    pass
    wrap(getattr(env, "pool", None), "reset", "    pool.reset")
    wrap(getattr(env, "events", None), "reset", "    events.reset")
    wrap(getattr(env, "level_replay", None), "choose", "    level_replay.choose")
    if getattr(env, "procedural", None) is not None:
        for n in ("redraw", "draw", "reset", "place"):
            wrap(env.procedural, n, "    procedural." + n)
    for _ in range(STEPS):
        env.step(act)
    print(f"\nRESET PROFILE over {STEPS} steps (B={env.B})")
    for k, v in sorted(T_.items(), key=lambda kv: -kv[1]):
        print(f"{k:42s} {1e3 * v / STEPS:8.2f} ms/step  {N_[k] / STEPS:6.2f} calls/step  {1e3 * v / max(N_[k], 1):8.2f} ms/call")
    raise SystemExit(0)
ppo_v2.train = train
ppo.main()
