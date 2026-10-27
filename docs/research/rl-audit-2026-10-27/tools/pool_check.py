"""Check OpponentPool.act subset mode == the old whole-width mode on a real race env, step by step, from the same
hidden states; then time both.

usage: python pool_check.py WARM STEPS -- <ppo argv>
"""
import sys
import time

import torch

WARM, STEPS = int(sys.argv[1]), int(sys.argv[2])
argv = sys.argv[sys.argv.index("--") + 1:]
sys.argv = ["ppo"] + argv
from f1sim.learn import ppo, ppo_v2  # noqa: E402
from f1sim.learn.opponent_pool import OpponentPool  # noqa: E402
from f1sim.gym_env import OPP_DRIVER_POOL  # noqa: E402


def clone_h(h):
    return None if h is None else h.clone()


def train(a, *, env, obs, device, names, init_ck, controller, graph_rt, applied, league_seats=()):
    pool = env.pool
    orig_act = OpponentPool.act
    stats = {"steps": 0, "max_act": 0.0, "max_h": 0.0, "rows": 0, "t_old": 0.0, "t_new": 0.0}

    def checking_act(self, obs_, driver):
        saved = [clone_h(e.hidden) for e in self.entries]
        self.subset = False
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out_old = orig_act(self, obs_, driver)
        torch.cuda.synchronize(); stats["t_old"] += time.perf_counter() - t0
        h_old = [clone_h(e.hidden) for e in self.entries]
        for e, h in zip(self.entries, saved):
            e.hidden = clone_h(h)
        self.subset = True
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out_new = orig_act(self, obs_, driver)
        torch.cuda.synchronize(); stats["t_new"] += time.perf_counter() - t0
        drv = driver >= OPP_DRIVER_POOL
        if bool(drv.any()):
            stats["max_act"] = max(stats["max_act"], float((out_old - out_new)[drv].abs().max()))
            stats["rows"] += int(drv.sum())
            for j, e in enumerate(self.entries):
                rows_j = driver == OPP_DRIVER_POOL + j
                if not bool(rows_j.any()) or e.hidden is None or h_old[j] is None:
                    continue
                for t_o, t_n in zip(h_old[j], e.hidden):
                    if t_o is None or t_n is None:
                        continue
                    stats["max_h"] = max(stats["max_h"], float((t_o[:, rows_j] - t_n[:, rows_j]).abs().max()))
        stats["steps"] += 1
        return out_new

    act = torch.zeros(env.B, env.act_dim, device=device)
    for _ in range(WARM):
        obs, *_ = env.step(act)
    OpponentPool.act = checking_act
    for _ in range(STEPS):
        obs, *_ = env.step(act)
    OpponentPool.act = orig_act
    n = max(stats["steps"], 1)
    print(f"\nPOOL CHECK: {stats['steps']} steps, {stats['rows'] / n:.1f} pool-driven rows a step | max |action diff| "
          f"{stats['max_act']:.3g} | max |hidden diff| on driven rows {stats['max_h']:.3g} | "
          f"old {1e3 * stats['t_old'] / n:.1f} ms, subset {1e3 * stats['t_new'] / n:.1f} ms a step", flush=True)
    raise SystemExit(0)


ppo_v2.train = train
ppo.main()
