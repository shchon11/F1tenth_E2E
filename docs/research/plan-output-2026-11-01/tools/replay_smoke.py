"""닿았던 경주 다시 내기(plan_rl.HitLevels)가 제대로 도나 보는 짧은 시험 (11-12). 짧은 경주(기본 6 s)를 마구잡이 행동으로 돌리며
 - 같은 자리에서 같은 이름(시드)으로 다시 나온 경주가 정말 같은 경주인지: 트랙, 상자 종류·자리, 상대 유무, 출발 자리
 - 다시 낸 비율, 자리당 쌓인 경주 수, 점수가 있는 경주 비율, 점수 높은 경주가 더 자주 다시 나오는지
인자: [환경 수 64] [결정 수 1500] [경주 길이 s 6]"""
import sys, os, collections
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2
from f1sim.learn.plan_reward import PlanReward, SEEN_S
from f1sim import plan_steps as PS, dynamics as dyn

n_env = int(sys.argv[1]) if len(sys.argv) > 1 else 64
n_dec = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
ep_s = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
dev = torch.device("cuda")
OV = {"episode_s": ep_s}
if os.environ.get("SMOKE_SCALES"):
    OV["track_scales"] = list(PR.SR.TRACK_SCALES_25)
env, lim, world = PR.build(PR.WORLD, n_env, 1, dev, overrides=OV, fmt="v2")
lid = env.learner_ids; Bl = int(lid.numel())
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
levels = PR.HitLevels(env, lid, p_replay=0.5, size=24, seed=1)
env.level_replay = levels
proc = env.procedural
opp_rows = lid + 1
g = torch.Generator(device="cpu"); g.manual_seed(0)
seen = collections.defaultdict(list)          # (seat, seed) -> [(tid, sid tuple, box xy, opponent on, start xy)]
scores = {}
n_start = 0
for t in range(n_dec):
    u = (torch.randn(Bl, PS.ACT_DIM, generator=g) * 0.6).to(dev)
    u[:, PS.N_STEP:] += 0.3                    # mostly forwards
    hard0 = env.sim.contact_counts[lid, 3:6].clone()
    out, p, _ = drv.decision(torch.tanh(u))
    hd = (env.sim.contact_counts[lid, 3:6] - hard0).clamp_min(0).float()
    hd[:, 2] = hd[:, 2] * (pr.seen_t <= SEEN_S + PS.DT).float()
    levels.add(hd.sum(1) + PR.HitLevels.STAND * (env.sim.state[lid, dyn.IVX].abs() < 0.3).float(), ~out["restarted"])
    for b in torch.nonzero(out["restarted"]).flatten().tolist():
        row = int(lid[b]); seat = int(env.race[row]); seed = int(env.level_seed[row])
        if seed < 0:
            continue
        n_start += 1
        eid = int(env.sim.eid[row]) if hasattr(env.sim, "eid") else row
        poses = proc.slots()[0][eid].cpu().numpy()
        sid = proc.p_sid[eid].cpu().numpy()
        on = bool(env.opp_on[opp_rows[b]]) if hasattr(env, "opp_on") else None
        seen[(seat, seed)].append((int(env.sim.tid[row]), tuple(sid.tolist()), np.round(poses[sid >= 0, :2], 3),
                                   on, env.sim.state[row, :2].cpu().numpy().copy()))
bad = 0; rep = 0
for (seat, seed), v in seen.items():
    for w in v[1:]:
        rep += 1
        same = (w[0] == v[0][0] and w[1] == v[0][1] and w[2].shape == v[0][2].shape and np.allclose(w[2], v[0][2], atol=2e-3)
                and np.linalg.norm(w[4] - v[0][4]) < 0.05)
        if not same:
            bad += 1
            if bad <= 5:
                print(f"  다름: 자리 {seat} 시드 {seed}: 트랙 {v[0][0]} / {w[0]}, 상자 수 {(np.array(v[0][1]) >= 0).sum()} / "
                      f"{(np.array(w[1]) >= 0).sum()}, 출발 자리 차이 {np.linalg.norm(w[4] - v[0][4]):.3f} m")
st = levels.stats()
print(f"트랙 {len(env.sim.track.tracks)}개, 학습 차 {Bl}대, {n_dec}결정 x 경주 {ep_s:g} s | 이름 있는 경주 시작 {n_start}, 그중 같은 이름이 다시 나온 것 "
      f"{rep} ({100 * rep / max(1, n_start):.0f} %), 다시 나왔는데 다른 경주였던 것 {bad}")
print(f"새 경주 {st['new']}, 다시 낸 경주 {st['replayed']} ({100 * st['replayed'] / max(1, st['new'] + st['replayed']):.0f} %), 자리당 쌓인 경주 "
      f"{st['buffer_mean']:.1f}, 넘겨받은 경주 {levels.ended} 중 점수 > 0: {levels.ended_hit}, 쌓인 점수 평균 {st['score_mean']:.4f}")
# do the replayed ones score higher than the stored average? (rank picks the highest)
sc_all = np.array([v[0] for b in levels.buf for v in b.values()])
print(f"쌓인 경주 점수: 0인 것 {100 * (sc_all == 0).mean():.0f} %, 중앙 {np.median(sc_all):.4f}, 위 10 % {np.quantile(sc_all, 0.9):.4f}")
