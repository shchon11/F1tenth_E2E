"""_fill_scans subset path vs the old whole-batch scan: noise-free ranges and hit types of the reset rows must agree."""
import sys, time
import torch
WARM = int(sys.argv[1]); N = int(sys.argv[2])
argv = sys.argv[sys.argv.index("--") + 1:]
sys.argv = ["ppo"] + argv
from f1sim.learn import ppo, ppo_v2  # noqa: E402
def train(a, *, env, obs, device, names, init_ck, controller, graph_rt, applied, league_seats=()):
    act = torch.zeros(env.B, env.act_dim, device=device)
    for _ in range(WARM):
        env.step(act)
    worst_r = 0.0; type_mis = 0; n_beams = 0; t_old = t_new = 0.0
    for trial in range(N):
        ids = torch.randperm(env.B, device=device)[:4 * (1 + trial % 4)].sort().values
        speed = env.sim.odom.state[ids, 3]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cars = env.sim._car_boxes(env.sim.state)
        _, true_o, type_o = env.sim.lidar.scan(env.sim.state[:, :3], None, env.sim.P, motion_distortion=False, tid=env.sim.tid,
                                              cars=cars, compiled=True, eid=env.sim.eid)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        _, true_n, type_n = env._fill_scans(ids, speed)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        t_old += t1 - t0; t_new += t2 - t1
        worst_r = max(worst_r, float((true_o[ids] - true_n[ids]).abs().max()))
        type_mis += int((type_o[ids] != type_n[ids]).sum()); n_beams += type_o[ids].numel()
    print(f"\nFILL CHECK B={env.B}: {N} trials | max |range_true diff| {worst_r:.3g} m | type mismatches {type_mis}/{n_beams} | "
          f"old whole-batch scan {1e3 * t_old / N:.1f} ms, subset {1e3 * t_new / N:.1f} ms", flush=True)
    raise SystemExit(0)
ppo_v2.train = train
ppo.main()
