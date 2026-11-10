"""기억이 사각지대 상자를 아나 (11-10, 사용자: "사각지대에 장애물이 있어서 사고나는거 ... Tta는 필수"; plan25n5 --mem-aux의 판정 1).
학습 세계(plan_rl.build, 학습 조건, 도중에 치우기 없음)를 policy로 몰며 결정마다: 트랙을 따라 앞 R m 안 가장 가까운 상자(track_boxes의
첫 칸, R = run의 --box-range)가 지금 보이는지(LiDAR 자리 = 차 -> 상자 중심 직선이 벽을 지나면 '안 보임'; 다른 상자·차에 가린 건 셈에
안 넣음)와, 기억의 맞히기 출력(Actor2.aux)이 그 칸을 '있음'으로 보는지. 나눠서:
  안 보이는 상자 x {첫 바퀴 = 에피소드 6 s 전(아직 본 적 없음), 다음 바퀴들 = 9 s 뒤(지난 바퀴에 봤음)}, 보이는 상자도 같이.
기억이 사각지대를 기억하면 '안 보임·다음 바퀴'의 맞힘이 '안 보임·첫 바퀴'보다 확실히 높아야 함. 기억 상태는 에피소드 처음(0)에서
커 가므로 앞에 상자가 없을 때 '있음'이라 한 비율(헛짚음)도 첫 바퀴/다음 바퀴로 같이 -- 맞힘이 오른 만큼 헛짚음도 오르면 아는 게 아님.
인자: checkpoint [차수, 기본 256] [초, 기본 30]"""
import os, sys
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path = sys.argv[1]
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 256
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, n_env, 11, dev, fmt="v2")
lid = env.learner_ids; Bl = int(lid.numel())
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
if not hasattr(actor, "aux"):
    raise SystemExit("이 checkpoint에는 기억 맞히기 출력(--mem-aux)이 없음")
R = float(ck["args"].get("box_range") or 12.0)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
tid = env.sim.tid[lid]
NS = 24
tt = torch.linspace(0.05, 0.92, NS, device=dev)
cnt = {k: torch.zeros(2, device=dev) for k in ("hid_first", "hid_later", "vis_first", "vis_later", "none_first", "none_later")}
dt_ctrl = float(env.sim.control_dt)
for kd in range(int(secs / PS.DT)):
    tb = PR.track_boxes(env, pr.goal, lid, 3, R).view(Bl, 3, 4)          # the boxes ahead at this decision
    st = env.sim.state[lid].float()
    present = tb[:, 0, 0] > 0.5
    dx, dy = tb[:, 0, 1] * 5.0, tb[:, 0, 2] * 5.0
    c, s_ = torch.cos(st[:, 2]), torch.sin(st[:, 2])
    bx, by = st[:, 0] + dx * c - dy * s_, st[:, 1] + dx * s_ + dy * c
    px = st[:, 0:1] + (bx - st[:, 0])[:, None] * tt; py = st[:, 1:2] + (by - st[:, 1])[:, None] * tt
    rem = pr.goal.remaining(torch.stack([px, py], -1).reshape(-1, 2), tid.repeat_interleave(NS)).view(Bl, NS)
    hidden = present & torch.isnan(rem).any(1)                            # the straight line to it crosses a wall
    visible = present & ~hidden
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)                       # steps the memory with this decision's view
    said = actor.aux(obs.mem_h)[:, 0] > 0                                  # the memory's read-out: 'a box in the first slot'
    t_ep = env.ep_step[lid].float() * dt_ctrl
    first, later = t_ep < 6.0, t_ep >= 9.0
    for k, m in (("hid_first", hidden & first), ("hid_later", hidden & later), ("vis_first", visible & first),
                 ("vis_later", visible & later), ("none_first", ~present & first), ("none_later", ~present & later)):
        cnt[k][0] += (said & m).float().sum(); cnt[k][1] += m.float().sum()
    drv.decision(torch.tanh(mean), p)
f = lambda k: (f"{float(cnt[k][0] / cnt[k][1].clamp_min(1)) * 100:.0f} % ({int(cnt[k][1])}번)")
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
print(f"{name}: 트랙을 따라 앞 {R:g} m 가장 가까운 상자를 기억이 '있음'으로 맞힌 비율 | 벽에 가려 안 보임: 첫 바퀴 {f('hid_first')} -> "
      f"다음 바퀴들 {f('hid_later')} | 보임: 첫 바퀴 {f('vis_first')} -> 다음 바퀴들 {f('vis_later')} | 없는데 '있음'(헛짚음): 첫 바퀴 "
      f"{f('none_first')} -> 다음 바퀴들 {f('none_later')} (학습 차 {Bl}대 x {secs:.0f} s)", flush=True)
