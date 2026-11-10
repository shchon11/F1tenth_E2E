"""늦게 보이는 상자가 얼마나 있고 거기서 얼마나 닿나 (11-11, 사용자: "이건 좀 심도깊게 생각좀 해야겠는걸" -- 기억을 운전에 쓰게 만들기 전에
기억으로 얻을 수 있는 몫부터 잼). 학습 세계(plan_rl.build), 학습 조건, 혼자, 상자 계속, 통째 에피소드.
차가 상자 하나를 지나갈 때마다 한 건: 그 상자가 처음 보인 때(차 -> 상자 중심 직선이 벽을 안 지남, blind_probe와 같은 판정)부터 옆을 지날
때까지의 시간 = 미리 본 시간. 그 사이 그 상자에 닿았는지(상자 접촉이 새로 생긴 결정에 차에서 가장 가까운 상자, 1 m 안), 세게(1 m/s 넘게)인지.
미리 본 시간 구간 x 그 상자를 처음 지남 / 다시 지남(지난 바퀴에 이미 봄)으로 나눔. 늦게 보이는 상자에서 다시 지날 때 접촉이 줄면 기억이
일하는 것, 안 줄면 그 구간의 접촉이 기억이 가져갈 수 있는 몫.
인자: checkpoint [환경 수, 기본 256] [초, 기본 60]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path = sys.argv[1]
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 256
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
env, lim, world = PR.build(PR.WORLD, n_env, 13, dev, fmt="v2", overrides={
    "prop_fraction": 1.0, "opp_present": 0.0, "removal_p": 0.0, "episode_s": secs + 5.0})
lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
po = env.procedural
tid = env.sim.tid[lid]
C = int(po.p_poses.shape[1])
NS = 24
tt = torch.linspace(0.05, 0.92, NS, device=dev)
R = 12.0
i_on, i_hd = CC.index("onsets_prop"), CC.index("hard_prop")
BIG = 10 ** 9
seen_kd = torch.full((Bl, C), BIG, device=dev, dtype=torch.long)       # the decision the box was first seen on this approach
pass_kd = torch.full((Bl, C), BIG, device=dev, dtype=torch.long)       # the decision the car went by it
hits = torch.zeros(Bl, C, 2, device=dev)                               # contacts with it on this approach (all, hard)
n_pass = torch.zeros(Bl, C, device=dev, dtype=torch.long)              # times the car has gone by it before
ds_prev = torch.full((Bl, C), float("nan"), device=dev)
cnt_prev = env.sim.contact_frame()[lid].float()[:, [i_on, i_hd]].clone()
EDGES = (0.3, 0.6, 1.0, 1.5)                                           # [s] lead-time bins
NB = len(EDGES) + 1
acc = torch.zeros(NB, 2, 4, device=dev)                                # bin x (first, again) x (n, hit, hard, hit count)
L = pr.goal.lap[tid].float()[:, None]
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].float()
    pos = po.p_poses[lid, :, :2].float()
    live = po.p_sid[lid] >= 0
    r_car = pr.goal.remaining(st[:, :2], tid)
    r_box = pr.goal.remaining(pos.reshape(-1, 2), tid.repeat_interleave(C)).view(Bl, C)
    ds = torch.remainder(r_car[:, None] - r_box + 0.5 * L, L) - 0.5 * L
    ok = live & torch.isfinite(ds)
    px = st[:, None, 0:1] + (pos[..., 0:1] - st[:, None, 0:1]) * tt
    py = st[:, None, 1:2] + (pos[..., 1:2] - st[:, None, 1:2]) * tt
    rem = pr.goal.remaining(torch.stack([px, py], -1).reshape(-1, 2), tid.repeat_interleave(C * NS)).view(Bl, C, NS)
    vis = ok & ~torch.isnan(rem).any(2)
    appr = ok & (ds > 0.0) & (ds <= R)
    seen_kd = torch.where(appr & vis & (seen_kd == BIG), torch.full_like(seen_kd, kd), seen_kd)
    # going by = the distance left along the track crosses the box's AND the box is beside the car (straight-line 2.5 m: the
    # distance field alone also crosses for a box in the neighbouring stretch behind a wall)
    beside = ((pos - st[:, None, :2]) ** 2).sum(2) < 2.5 ** 2
    went = ok & beside & (ds <= 0.0) & (ds_prev > 0.0) & (ds_prev < 3.0) & (pass_kd == BIG)
    pass_kd = torch.where(went, torch.full_like(pass_kd, kd), pass_kd)
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    out, _, _ = drv.decision(torch.tanh(mean), p)
    cnt = env.sim.contact_frame()[lid].float()[:, [i_on, i_hd]]
    dc = (cnt - cnt_prev).clamp(min=0.0); cnt_prev = cnt.clone()
    d2 = ((pos - st[:, None, :2]) ** 2).sum(2)
    d2 = torch.where(live, d2, torch.full_like(d2, 1e9))
    near = d2.argmin(1)
    close = d2.gather(1, near[:, None])[:, 0] < 1.0
    rows = torch.nonzero(close & (dc[:, 0] > 0))[:, 0]
    hits[rows, near[rows]] += dc[rows]
    # an approach ends 0.5 s after the car went by: one record
    fin = (pass_kd != BIG) & (kd - pass_kd >= 5)
    if bool(fin.any()):
        lead = (pass_kd - torch.minimum(seen_kd, pass_kd)).float() * PS.DT
        b = torch.bucketize(lead, torch.tensor(EDGES, device=dev))
        again = (n_pass > 0).long()
        idx = (b * 2 + again)[fin]
        h = hits[fin]
        acc.view(-1, 4).index_add_(0, idx, torch.stack([torch.ones_like(h[:, 0]), (h[:, 0] > 0).float(),
                                                        (h[:, 1] > 0).float(), h[:, 0]], 1))
        n_pass = n_pass + fin.long()
        seen_kd[fin] = BIG; pass_kd[fin] = BIG; hits[fin] = 0.0
    ds_prev = ds
    rs = out["restarted"]
    if bool(rs.any()):                                                 # a new episode: a new layout, nothing seen
        seen_kd[rs] = BIG; pass_kd[rs] = BIG; hits[rs] = 0.0; n_pass[rs] = 0; ds_prev[rs] = float("nan")
        cnt_prev[rs] = env.sim.contact_frame()[lid].float()[:, [i_on, i_hd]][rs]
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
tot = acc.sum((0, 1))
print(f"{name}: 학습 조건, 혼자, 상자 계속, 학습 차 {Bl}대 x {secs:.0f} s -- 상자를 지나간 {int(tot[0])}건, 그중 닿은 {int(tot[1])}건"
      f"(세게 {int(tot[2])}건)", flush=True)
lab = ["0.3 s 안"] + [f"{EDGES[i]:g}~{EDGES[i + 1]:g} s" for i in range(len(EDGES) - 1)] + [f"{EDGES[-1]:g} s 넘게"]
for b in range(NB):
    parts = []
    for a, an in ((0, "처음 지남"), (1, "다시 지남")):
        v = acc[b, a]; n = max(float(v[0]), 1.0)
        parts.append(f"{an} {int(v[0])}건: 닿음 {float(v[1]) / n * 100:.1f} % (세게 {float(v[2]) / n * 100:.1f} %)")
    share = float(acc[b, :, 1].sum()) / max(float(tot[1]), 1.0) * 100
    print(f"  미리 본 시간 {lab[b]} | {parts[0]} | {parts[1]} | 닿은 건의 {share:.0f} %", flush=True)
