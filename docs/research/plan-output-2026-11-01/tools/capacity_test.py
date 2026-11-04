"""지금 크기의 네트워크 하나로 네 경우를 다 담을 수 있나 (11-04 19:50, 사용자: "지금의 단순한 네트워크가 저 상황들을 충분히 학습할 수 있을까?").
base를 섞은 checkpoint(빈 구간 = 고정 solo policy, obstacle·차 근처 = race policy)의 실행 평균 action을, 같은 크기 MLP 하나(입력 409 -> 256 -> 256
-> 8, tanh)에 supervised(MSE)로 맞춤 -- DAgger: 1회차는 teacher가 운전, 이후 student가 운전(환경 절반씩)하며 teacher가 정답을 붙임, 모은 것 전부로
다시 맞춤. world = one-world 그대로(obstacle 50 % x 상대 차 50 %, removal). 측정용일 뿐 학습 방법이 아님(사용자: imitation은 초기값만).
판정(미리 정함): student가 빈 트랙 solo lap <= teacher + 0.10 s 이고 obstacle 앞 8 layout hard <= teacher x 1.5 + 2 면 크기는 충분(문제는 RL 쪽),
아니면 크기 부족 의심. 인자: teacher체크포인트 출력디렉터리 환경수 회차 회차당결정수 [회차당학습스텝] [student 은닉 폭]"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR

src, out = sys.argv[1], sys.argv[2]
n_env, n_iter, n_dec = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
n_fit = int(sys.argv[6]) if len(sys.argv) > 6 else 3000
dev = torch.device("cuda")
torch.manual_seed(0)
ck = torch.load(src, map_location=dev, weights_only=False)
ar = dict(ck["args"]); ar["envs"] = n_env; ar["level_replay"] = 0.0
a = argparse.Namespace(**ar)
env = SR.build_env(a, dev)
from f1sim.learn.graph_runtime import prepare_graph_runtime
prepare_graph_runtime(env, log=lambda _t: None)   # as state_rl.train: the compiled sim
env.reset(seed=1234)
remover = SR.PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=1234) \
    if float(getattr(a, "removal_p", 0) or 0) > 0 else None
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids
B, A, R = int(lid.numel()), tp.A, int(a.repeat)
D = tp.o.shape[1]
h0 = int(ar.get("hidden", 256) or 256)
H = int(sys.argv[7]) if len(sys.argv) > 7 else h0              # [student hidden width]: wider = the race policy widened (Net2Net)
stu = SR.ActorCritic(D, act_dim=A, hidden=H).to(dev)
sd0 = tp.net.state_dict()
stu.load_state_dict(SR.widen_mlp_state(sd0, h0, H) if H > h0 else sd0, strict=False)   # the race policy's weights as the start
with torch.no_grad():
    stu.mean_lo.fill_(-1e6); stu.mean_hi.fill_(1e6)                # identity: fit the executed (clamped) action directly
    stu.actor_keep.copy_(tp.net.actor_keep)
opt = torch.optim.Adam(stu.pi.parameters(), lr=3e-4)
act_full = torch.zeros(env.B, env.act_dim, device=dev)
X, Y, G = [], [], []
print(f"[capacity] teacher {src}: 환경 {env.B} (학습 행 {B}), 관측 {D}, action {A}, 회차 {n_iter} x 결정 {n_dec}", flush=True)
for it in range(n_iter):
    t0 = time.time()
    last = torch.zeros(B, A, device=dev)
    drive_s = torch.zeros(B, dtype=torch.bool, device=dev)
    if it > 0:
        drive_s[torch.randperm(B, device=dev)[: B // 2]] = True     # half the cars driven by the student
    n_gate = 0.0
    with torch.no_grad():
        for t in range(n_dec):
            if t % 200 == 0:
                print(f"    회차 {it} 결정 {t} ({time.time() - t0:.0f} s)", flush=True)
            a_t = tp(last)                                          # the gated mean action (also steps the gate's hold)
            o_n = tp.nrm(tp.o)
            X.append(o_n.half().cpu()); Y.append(a_t.half().cpu())
            g = tp.mix.gate if tp.mix is not None else torch.ones(B, dtype=torch.bool, device=dev)
            G.append(g.cpu()); n_gate += float(g.float().mean())
            exe = torch.where(drive_s[:, None], SR.clamp_any(stu.pi(o_n), tp.ar), a_t) if it > 0 else a_t
            act_full[lid] = exe
            for k in range(R):
                env.step(act_full)
                if remover is not None:
                    remover.step()
            last = exe
    Xc = torch.cat(X); Yc = torch.cat(Y)              # half precision on the CPU
    N = Xc.shape[0]
    stu.train()
    for s in range(n_fit):
        j = torch.randint(0, N, (4096,))
        xb, yb = Xc[j].to(dev).float(), Yc[j].to(dev).float()
        loss = F.mse_loss(stu.pi(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    stu.eval()
    with torch.no_grad():
        j = torch.randint(0, N, (min(N, 65536),))
        Gc = torch.cat(G)[j]
        e = (stu.pi(Xc[j].to(dev).float()) - Yc[j].to(dev).float()).pow(2).mean(1).cpu()
    print(f"  회차 {it}: 표본 {N}, race policy 운전 비율 {n_gate / n_dec * 100:.0f} %, 맞춘 오차 MSE 전체 {float(e.mean()):.4f} | "
          f"base 구간 {float(e[~Gc].mean()) if (~Gc).any() else float('nan'):.4f} | race 구간 {float(e[Gc].mean()) if Gc.any() else float('nan'):.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)
    del Xc, Yc
os.makedirs(out, exist_ok=True)
ar_s = dict(ck["args"]); ar_s["base"] = ""; ar_s["name"] = os.path.basename(out); ar_s["hidden"] = H
torch.save({"model": stu.state_dict(), "norm": tp.nrm.state_dict(), "ret_rms": ck.get("ret_rms"), "obs_dim": D, "args": ar_s,
            "update": 0, "ahead": SR.AHEAD, "bev": None}, os.path.join(out, "state_u0.pt"))
json.dump(ar_s, open(os.path.join(out, "args.json"), "w"), indent=1)
print(f"[capacity] 저장 {out}/state_u0.pt", flush=True)
