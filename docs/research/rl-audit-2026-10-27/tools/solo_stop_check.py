"""혼자 판에서 왜 서나 (2026-10-29): 선생을 상대 없는 세계(상자 있음)에서 몰며 차 4대의 시간별 속도 명령·실제 속도·조향·접촉과,
관측의 '상대 있음' 칸·남은 시간 칸을 그림. 사용: python solo_stop_check.py CKPT --like RACE_JSON --out PNG"""
import argparse, sys
import numpy as np, torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn.state_rl import ActorCritic, RunningNorm, StateObs, Unwedge
from f1sim.learn.state_student import make_env_like
from f1sim.gym_env import EnvConfig
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--like", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--race", type=int, default=0, help="1: race_size 2 with the opponent absent (as in training); 0: race_size 1")
ap.add_argument("--unwedge", type=int, default=1)
a = ap.parse_args(); dev = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
ov = dict(max_steps=1610)
if not a.race:
    ov.update(race_size=1, opponent=EnvConfig().opponent, opponent_slots=None)
env, cfg, ecfg, md = make_env_like(a.like, dev, int(ck["args"].get("beams", 36)), 1, 1, True, envs=8, override=ov)
obs, _ = env.reset(seed=5); lid = env.learner_ids; B = int(lid.numel())
sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)), corridor=bool(ck["args"].get("corridor_obs", 0)))
last = torch.zeros(B, 2, device=dev); o = sobs(last)
net = ActorCritic(o.shape[1]).to(dev); net.load_state_dict(ck["model"], strict=False); net.eval()
nrm = RunningNorm(o.shape[1], dev); nrm.load_state_dict(ck["norm"]); act_full = torch.zeros(env.B, env.act_dim, device=dev)
T = 600; rec = np.zeros((T, B, 7), np.float32)
uw = Unwedge(env, lid) if a.unwedge else None
with torch.no_grad():
    for t in range(T):
        o = sobs(last); act = net.pi(nrm(o)).clamp(-1, 1); last = act; act_full[lid] = act
        for r in range(2):
            if uw is not None: uw.before(act_full)
            env.step(act_full)
            if uw is not None: uw.after()
        st = env.sim.state[lid]; res = env.last_result
        cont = (res.contact_onsets[lid].any(1) if res.contact_onsets is not None else torch.zeros(B, dtype=torch.bool, device=dev))
        rec[t, :, 0] = ((act[:, 1] + 1) / 2 * 10).cpu().numpy(); rec[t, :, 1] = st[:, 3].cpu().numpy()
        rec[t, :, 2] = (act[:, 0] * float(env.s_max) * 57.3).cpu().numpy(); rec[t, :, 3] = cont.float().cpu().numpy()
        rec[t, :, 4] = o[:, 8].cpu().numpy(); rec[t, :, 5] = o[:, 45].cpu().numpy(); rec[t, :, 6] = res.wall_dist[lid].cpu().numpy() if res.wall_dist is not None else 0
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(4, 1, figsize=(12, 12), dpi=90, sharex=True); tt = np.arange(T) * 0.05
for j in range(4):
    ax = axs[j]
    ax.plot(tt, rec[:, j, 0], label="속도 명령 m/s"); ax.plot(tt, rec[:, j, 1], label="실제 속도 m/s")
    ax.plot(tt, rec[:, j, 2] / 10, label="조향/10 (도)"); ax.plot(tt, rec[:, j, 6], label="벽까지 m", alpha=0.6)
    c = rec[:, j, 3] > 0.5; ax.plot(tt[c], np.full(c.sum(), -1.0), "rx", label="접촉 시작")
    ax.plot(tt, rec[:, j, 4] * 3, ":", label="관측: 상대 있음 x3"); ax.plot(tt, rec[:, j, 5] * 3, ":", label="관측: 남은 시간 x3")
    ax.set_ylabel(f"차 {j}"); ax.grid(alpha=0.3)
    if j == 0: ax.legend(fontsize=7, ncol=4)
axs[-1].set_xlabel("시간 s")
fig.suptitle(f"{a.ckpt.split('/')[-2]}/{a.ckpt.split('/')[-1]}: 상대 없는 세계({'경주 2대 중 상대 결석' if a.race else '1대'})에서 시간별 명령·속도·접촉", fontsize=10)
fig.tight_layout(); fig.savefig(a.out); print("saved", a.out, "| 끼임 복구 횟수", uw.count if uw is not None else "-")
