"""멈추는 상자에 정책이 반응은 하나 (11-03): corridor_cover.py -- 배치 0..6의 라인 위 상자 25개 모두 5 m 앞 통로 칸에 2~6칸으로
잡힘(배치 5의 멈추는 상자 #1 포함) -> 정보는 입력에 있음. 그런데 일곱 선생 모두 같은 상자에 박음. 반실험: 콘솔과 같은 배치를
그대로 달리며(console_layouts.py와 같은 난수 순서) 결정마다 같은 순간의 관측을 세 번 -- (실제) 그대로, (지움) 앞의 가장 가까운
보이는 상자를 정책의 상자 입력(보이는 상자 칸 3개 + 보이는 통로 칸)에서 지움, (홀로) 그 상자만 보이고 다른 상자는 지움.
라이다 칸은 그대로(상자는 거기 남음). 상자 정책의 평균 행동 중 추종기가 쓰는 둘: kappa_1 [1/m], v_1 정규화값.
실제 = 지움이면 그 순간 정책은 상자 입력에 반응하지 않음; 홀로가 다르면 다른 상자들이 반응을 가림.
첫 상자 접촉 앞 2 s를 0.1 s마다. 인자: 체크포인트 맵id 배치들(쉼표) 시드"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.learn import state_rl as SR
from f1sim import mpc


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, lays, seed = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4])
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
env = s["env"]; proc = env.procedural; runner = s["act_fn"]
ck = torch.load(run, map_location="cuda", weights_only=False)
tp2 = SR.TeacherPolicy(env, ck, torch.device("cuda"))      # its own StateObs: the runner's stays untouched
KMAX = mpc.PlanSpec().kappa_max
_vis = SR.visible_props
MASK = [None]


def _patched(env_, lid_, tol=0.20):
    v = _vis(env_, lid_, tol)
    return v if (v is None or MASK[0] is None) else v & MASK[0]


SR.visible_props = _patched
i_on, i_hd = CC.index("onsets_prop"), CC.index("hard_prop")
lid = env.learner_ids
for L in range(max(lays) + 1):
    if L > 0:
        s["obs"], _ = env.reset()
    if L not in lays:
        for k in range(int(40.0 / 0.025)):
            w._step_once(s)
        continue
    rows = []
    cf = env.sim.contact_frame()[0].float().cpu().numpy(); p_on, p_hd = cf[i_on], cf[i_hd]
    for k in range(int(40.0 / 0.025)):
        w._step_once(s)
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        on, hd = cf[i_on] > p_on, cf[i_hd] > p_hd
        p_on, p_hd = cf[i_on], cf[i_hd]
        if k % 2:
            rows.append(None if not on else ("hit", k * 0.025, int(hd)))
            continue
        st = env.sim.state[0]
        vis = _vis(env, lid)
        live = proc.p_sid[0] >= 0
        d = proc.p_poses[0, :, :2] - st[None, :2]
        c_, s_ = torch.cos(st[2]), torch.sin(st[2])
        dx, dy = d[:, 0] * c_ + d[:, 1] * s_, -d[:, 0] * s_ + d[:, 1] * c_
        dist = d.norm(dim=-1)
        cand = live & vis[0] & (dx > 0) & (dy.abs() < 1.2)
        last = runner.last
        with torch.no_grad():
            MASK[0] = None
            mu = tp2.net.pi(tp2.nrm(tp2.sobs(last)))[0].clamp(-1, 1)
            if bool(cand.any()):
                j = int(torch.where(cand, dist, torch.full_like(dist, 1e9)).argmin())
                m = torch.ones(1, proc.C, dtype=torch.bool, device=dist.device); m[0, j] = False
                MASK[0] = m
                mh = tp2.net.pi(tp2.nrm(tp2.sobs(last)))[0].clamp(-1, 1)
                MASK[0] = ~m
                mo = tp2.net.pi(tp2.nrm(tp2.sobs(last)))[0].clamp(-1, 1)
                MASK[0] = None
            else:
                j, mh, mo = -1, mu, mu
        g = runner.gate
        rows.append(("d", k * 0.025, float(st[3]), j, float(dist[j]) if j >= 0 else -1, float(dx[j]) if j >= 0 else 0,
                     float(dy[j]) if j >= 0 else 0, int(bool(g[0])) if g is not None else -1,
                     float(mu[0]) * KMAX, float(mh[0]) * KMAX, float(mo[0]) * KMAX, float(mu[4]), float(mh[4]), float(mo[4]),
                     int(on), int(hd)))
    hits = [i for i, r in enumerate(rows) if r is not None and (r[0] == "hit" or r[14])]
    print(f"== 배치 {L} ({os.path.basename(os.path.dirname(run))})", flush=True)
    if not hits:
        print("  상자 접촉 없음", flush=True)
        continue
    k0 = min(hits)
    print("     t   속도  상자#  거리    dx     dy  게이트 | kappa1 실제/지움/홀로 [1/m] | v1 실제/지움/홀로", flush=True)
    for r in rows[max(0, k0 - 80):k0 + 6]:
        if r is None or r[0] != "d":
            continue
        if int(round(r[1] / 0.025)) % 4:
            continue
        print(f"  {r[1]:5.2f} {r[2]:5.2f}  {r[3]:3d} {r[4]:5.2f} {r[5]:+5.2f} {r[6]:+5.2f}   {r[7]:2d}  | {r[8]:+.3f} {r[9]:+.3f} {r[10]:+.3f} | "
              f"{r[11]:+.3f} {r[12]:+.3f} {r[13]:+.3f}{'  <- 접촉' if r[14] else ''}", flush=True)
    t0 = rows[k0][1]
    print(f"  첫 상자 접촉 {t0:.2f} s", flush=True)
