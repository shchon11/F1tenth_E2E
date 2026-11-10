"""기억이 운전에 이득인가 (11-10, 사용자: "기억을 넣었을 때 이득이 되는지 간접적으로 알 수 있는 부분이 장애물이 좀 거시기하게 있는 맵들에서
갈수록 랩타임이 줄어드는지를 보는게 직빵"). 학습 세계(plan_rl.build: 원래 트랙 + 변형 16개, 학습 상자 배치), 학습 조건(plan대로 정확히),
혼자, 상자는 치우지 않고 계속, 차마다 배치 하나, `secs` 초. 같은 policy·같은 시드로 세 번:
  (가) 기억 그대로   (나) 바퀴를 넘을 때마다 그 차의 기억을 지움(매 바퀴가 처음 보는 배치)   (빈) 상자 없음(바퀴 순서 자체의 효과를 보는 기준)
바퀴 = 출발 뒤 처음 결승선을 넘은 때부터 다음 넘을 때까지(1번 바퀴 = 첫 완주). 바퀴 번호마다 중앙 시간과 접촉(벽+상자, 세게 = 1 m/s 넘게).
'거시기한' 배치 = (가)의 1번 바퀴가 같은 트랙 (빈) 바퀴 중앙보다 0.5 s 넘게 느린 차.
인자: checkpoint [환경 수, 기본 512] [초, 기본 60]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path = sys.argv[1]
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 512
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:      # the tyre limit the checkpoint was trained with
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
SEED = 77
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop")]
i_hd = [CC.index(k) for k in ("hard_wall", "hard_prop")]


def run(props: bool, forget: bool):
    env, lim, _ = PR.build(PR.WORLD, n_env, SEED, dev, fmt="v2", overrides={
        "prop_fraction": 1.0 if props else 0.0, "opp_present": 0.0, "removal_p": 0.0, "episode_s": secs + 5.0})
    lid = env.learner_ids; B = int(lid.numel())
    actor = load_actor2(ck, dev)
    pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
    obs = PlanObs2(env, lim, pr.goal)
    drv = PR.Driver(env, lim, pr, obs)
    t_cross = [[] for _ in range(B)]                          # decision index of each finish-line crossing
    c_cross = [[] for _ in range(B)]                          # contacts (all, hard) so far at each crossing
    first_rs = np.full(B, 10 ** 9)                            # the decision of a car's first restart: its laps count up to it
    for kd in range(int(secs / PS.DT)):
        _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
        out, _, _ = drv.decision(torch.tanh(mean), p)
        rs = out["restarted"].cpu().numpy()
        first_rs = np.where(rs & (first_rs > kd), kd, first_rs)
        crossed = (out["lap"] > 0)
        if bool(crossed.any()):
            cf = env.sim.contact_frame()[lid].float()
            on, hd = cf[:, i_on].sum(1).cpu().numpy(), cf[:, i_hd].sum(1).cpu().numpy()
            for i in torch.nonzero(crossed)[:, 0].tolist():
                t_cross[i].append(kd); c_cross[i].append((on[i], hd[i]))
            if forget and getattr(obs, "mem_h", None) is not None:
                obs.mem_h[crossed] = 0.0                     # every lap starts remembering nothing
    tid = env.sim.tid[lid].cpu().numpy()
    laps = {}                                                 # lap number -> list of (car, time, contacts, hard)
    n_rs = int((first_rs < 10 ** 9).sum())
    print(f"  ({'상자' if props else '빈 트랙'}{', 바퀴마다 기억 지움' if forget else ''}) 차 {B}대, 도중에 다시 출발한 차 {n_rs}대"
          + (f" (첫 재출발 중앙 {np.median(first_rs[first_rs < 10 ** 9]) * PS.DT:.0f} s)" if n_rs else "") + " -- 그 전 바퀴까지 셈", flush=True)
    for i in range(B):
        tc = [x for x in t_cross[i] if x <= first_rs[i]]
        cc = c_cross[i][:len(tc)]
        for k in range(1, len(tc)):
            laps.setdefault(k, []).append((i, (tc[k] - tc[k - 1]) * PS.DT, cc[k][0] - cc[k - 1][0], cc[k][1] - cc[k - 1][1]))
    del env
    torch.cuda.empty_cache()
    return laps, tid


name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
E, tid_e = run(False, False)
A, tid = run(True, False)
N, _ = run(True, True)
# the control (11-11): wiping on the EMPTY track -- if that slows the car too, the wipe disturbs the driving itself (the memory
# carries more than the layout) and the boxes' difference is not layout memory
EN, _ = run(False, True)
# the empty track's lap per track (laps 1..)
emp = {}
for k, L in E.items():
    for (i, t, _, _) in L:
        emp.setdefault(int(tid_e[i]), []).append(t)
emp_med = {t: float(np.median(v)) for t, v in emp.items()}
first = {i: t for (i, t, _, _) in A.get(1, [])}
tricky = {i for i, t in first.items() if t - emp_med.get(int(tid[i]), t) > 0.5}
print(f"{name}: 학습 조건, 혼자, 상자 계속, {secs:.0f} s, 차 {len(first)}대(상자) | '거시기한' 배치(1번 바퀴가 빈 트랙보다 0.5 s 넘게 느림) "
      f"{len(tricky)}대", flush=True)


def row(D, k, sel=None):
    L = [x for x in D.get(k, []) if sel is None or x[0] in sel]
    if len(L) < 5:
        return None
    return (float(np.median([x[1] for x in L])), float(np.mean([x[2] for x in L])), float(np.mean([x[3] for x in L])), len(L))


for title, sel in (("모든 상자 배치", None), ("'거시기한' 배치만", tricky)):
    print(f"[{title}] 바퀴 | (빈 트랙) | (가) 기억 그대로: 중앙 시간, 접촉/바퀴(세게) | (나) 바퀴마다 기억 지움 | (가) - (나)", flush=True)
    for k in range(1, 9):
        e = row(E, k)
        a, n = row(A, k, sel), row(N, k, sel)
        if a is None or n is None:
            continue
        es = f"{e[0]:.2f} s" if e else "–"
        print(f"  {k}번 바퀴 | {es} | {a[0]:.2f} s, {a[1]:.2f}({a[2]:.2f}) [{a[3]}대] | {n[0]:.2f} s, {n[1]:.2f}({n[2]:.2f}) [{n[3]}대] | "
              f"{a[0] - n[0]:+.2f} s, 접촉 {a[1] - n[1]:+.2f}", flush=True)

# one line for the table (11-11, the user agreed: judge the memory by whether wiping it makes driving worse, not by what it
# knows): the laps after the first (the memory has seen the layout once), all layouts pooled
def pooled(D):
    L = [x for k, v in D.items() if k >= 2 for x in v]
    return (float(np.mean([x[1] for x in L])), float(np.mean([x[2] for x in L])), float(np.mean([x[3] for x in L])), len(L)) if len(L) >= 20 else None


a, n = pooled(A), pooled(N)
e, en = pooled(E), pooled(EN)
if a and n:
    print(f"요약 2번 바퀴부터(혼자, 상자 계속): 기억 그대로 평균 {a[0]:.2f} s 접촉 {a[1]:.2f}/바퀴(세게 {a[2]:.2f}) [{a[3]}바퀴] | 바퀴마다 지움 "
          f"{n[0]:.2f} s 접촉 {n[1]:.2f}/바퀴(세게 {n[2]:.2f}) [{n[3]}바퀴] | 기억 그대로 - 지움: 바퀴 시간 {a[0] - n[0]:+.2f} s, "
          f"접촉 {a[1] - n[1]:+.2f}/바퀴"
          + (f" | 빈 트랙에서 같은 비교: 바퀴 시간 {e[0] - en[0]:+.2f} s, 접촉 {e[1] - en[1]:+.2f}/바퀴 [{e[3]}/{en[3]}바퀴]" if e and en else ""),
          flush=True)
else:
    print("요약 2번 바퀴부터: 바퀴가 모자람(20바퀴 미만)", flush=True)
