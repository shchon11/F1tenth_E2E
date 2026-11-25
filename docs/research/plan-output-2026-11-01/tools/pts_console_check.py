"""(11-25) plan_rl --arch pts: does the console's plan driver load and drive a pts checkpoint? The console's path exactly
(`plan_policy.PlanRunner`, called once per control step, a decision every 4): a few races of the race world, the runner in
mode "ideal" (training's execution; MAP / the follower downstream of the plan are the same for every plan policy). Prints:
the decisions made, metres driven per car, the share of decisions whose point tokens had all 5 frames and whose anchors came
from a previous plan, whether every plan stayed finite. No training, no exam numbers -- a load-and-drive check.
usage: python pts_console_check.py CK [SECONDS 15] [RACES 4]"""
import os
import sys

import torch

torch.set_grad_enabled(False)
ck_path = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
G = int(sys.argv[3]) if len(sys.argv) > 3 else 4
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of, PlanRunner
os.environ["F1SIM_LIDAR_MOUNT_X"] = str(lidar_mount_of(ck))
os.environ["F1SIM_PLAN_LAT_MARGIN"] = str((ck.get("args") or {}).get("plan_lat_margin", 0.95))
from f1sim.learn import plan_rl as PR
from f1sim.learn import pts_policy as PTSP
from f1sim import plan_steps as PS

assert PTSP.is_pts(ck), "not a --arch pts checkpoint"
W = os.path.join(os.path.dirname(PR.WORLD), "plan_world_korea26scan_race.json")
for mode in ("ideal",):
    env, lim, a = PR.build(W, 2 * G, 21, dev, overrides={"opp_present": 0.0, "warps": 0, "align_episodes": True})
    env.ep_limit[:] = 10 ** 6
    runner = PlanRunner(env, ck, dev, mode=mode)
    assert runner.pts and isinstance(runner.obs, PTSP.PtsObs)
    lid = env.learner_ids
    act0 = torch.zeros(env.B, env.act_dim, device=dev)
    xy0 = env.sim.state[lid, :2].clone(); prev = xy0.clone(); dist = torch.zeros(lid.numel(), device=dev)
    n_dec, full, anc, finite = 0, 0.0, 0.0, True
    for k in range(int(round(secs / PS.DT)) * PS.SUB):
        due = (env._dec_cnt[lid] % PS.SUB) == 0
        act = runner()
        _, _, te, trc, _ = env.step(act0 if act is None else act)
        if bool(due.any()):
            n_dec += 1
            P = runner.obs.pts_input()
            full += float(P.fok.all(1).float().mean()); anc += float(P.anc_ok.float().mean())
            finite &= bool(torch.isfinite(runner.last_mean).all())
        xy = env.sim.state[lid, :2]
        step = (xy - prev).norm(dim=1)
        dist += torch.where(step < 1.0, step, torch.zeros_like(step)); prev = xy.clone()
    print(f"[콘솔 점검] {mode}: 결정 {n_dec}번, 차마다 간 거리 {[round(float(d), 1) for d in dist]} m, 5장 다 있는 결정 "
          f"{full / max(1, n_dec) * 100:.0f} %, 지난 plan에서 온 anchor {anc / max(1, n_dec) * 100:.0f} %, plan 유한 {finite}", flush=True)
    del env, runner
    torch.cuda.empty_cache()
