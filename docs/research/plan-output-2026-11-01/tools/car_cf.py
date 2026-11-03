"""경주에서 정책이 상대 차에 반응은 하나 (11-03): 상대 단계 srp25h3gwt_race u50은 시작점보다 차와 세게 104 -> 36으로 줄였지만 추월
한 번에 세게 0.4~0.9번. 상자 때(crate_cf.py)처럼 반실험: 콘솔 경주(console_races.py와 같은 세션·시드·판 순서)를 그대로 달리며
결정마다 같은 순간의 관측을 두 번 -- (실제) 그대로, (지움) 상대 차를 정책의 차 입력(통로에 겹친 차 시공간 칸, `visible_cars`)에서
지움(라이다 칸에는 남음). 추종기가 쓰는 kappa_1 [1/m], v_1. 차와 세게 부딪히기 전 1.5 s를 0.1 s마다: 출발 뒤 시각, 간격(+ = 내가 앞),
상대의 내 차 기준 dx/dy, 두 속도, 게이트, 실제/지움. 끝에 모든 세게 접촉의 앞 1 s 평균 |차이|와 종류(출발 3 s 안 / 뒤에서 / 나란히).
인자: 체크포인트 맵id 판당초 판수 상대종류 상대속도배율 시드"""
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


run, mp, secs, n_race, kind, sc, seed = (sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
                                         float(sys.argv[6]), int(sys.argv[7]))
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=2, device="cuda", compile=False,
                                    collision_mode="soft", procedural=False, episode_s=3600.0, randomize=False,
                                    opponent_slots=[{"kind": kind, "speed_scale": [sc, sc]}]), 0)
env = s["env"]; runner = s["act_fn"]
ck = torch.load(run, map_location="cuda", weights_only=False)
tp2 = SR.TeacherPolicy(env, ck, torch.device("cuda"))
KMAX = mpc.PlanSpec().kappa_max
_vc = SR.visible_cars
HIDE = [False]


def _patched(env_, lid_, tol=0.20):
    v = _vc(env_, lid_, tol)
    return v if (v is None or not HIDE[0]) else torch.zeros_like(v)


SR.visible_cars = _patched
i_c, i_h = CC.index("onsets_car"), CC.index("hard_car")
lid = env.learner_ids
allpre, kinds = [], {"출발 3 s 안": 0, "뒤에서(간격 < -0.5 m)": 0, "나란히(|간격| <= 0.5 m)": 0, "앞에서(간격 > 0.5 m)": 0}
for R in range(n_race):
    if R > 0:
        s["obs"], _ = env.reset()
    rows = []
    cf = env.sim.contact_frame()[0].float().cpu().numpy(); p_c, p_h = cf[i_c], cf[i_h]
    gap = None
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        r = env.last_result
        if gap is None:
            Lt = float(env.sim.track.length[env.sim.tid[0]])
            gap = float(r.s[0] - r.s[1]); gap = (gap + 0.5 * Lt) % Lt - 0.5 * Lt
        else:
            gap += float(r.progress[0] - r.progress[1])
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        hard = cf[i_h] > p_h
        p_c, p_h = cf[i_c], cf[i_h]
        if k % 2:
            if hard:
                rows.append(("hit", k * 0.025))
            continue
        st = env.sim.state
        c_, s_ = torch.cos(st[0, 2]), torch.sin(st[0, 2])
        d = st[1, :2] - st[0, :2]
        dx, dy = float(d[0] * c_ + d[1] * s_), float(-d[0] * s_ + d[1] * c_)
        last = runner.last
        with torch.no_grad():
            HIDE[0] = False
            mu = tp2.net.pi(tp2.nrm(tp2.sobs(last)))[0].clamp(-1, 1)
            HIDE[0] = True
            mh = tp2.net.pi(tp2.nrm(tp2.sobs(last)))[0].clamp(-1, 1)
            HIDE[0] = False
        g = runner.gate
        rows.append(("d", k * 0.025, gap, dx, dy, float(st[0, 3]), float(st[1, 3]), int(bool(g[0])) if g is not None else -1,
                     float(mu[0]) * KMAX, float(mh[0]) * KMAX, float(mu[4]), float(mh[4]), int(hard)))
    hits = [i for i, x in enumerate(rows) if x[0] == "hit" or (x[0] == "d" and x[-1])]
    if not hits:
        continue
    print(f"== 판 {R}: 세게 차 접촉 {len(hits)}번", flush=True)
    last_k = -1e9
    for k0 in hits:
        t0 = rows[k0][1]
        if t0 - last_k < 1.5:
            continue
        last_k = t0
        dec = [x for x in rows[max(0, k0 - 70):k0 + 1] if x[0] == "d"]
        pre = [x for x in dec if t0 - x[1] <= 1.0]
        if pre:
            allpre.append((np.mean([abs(x[8] - x[9]) for x in pre]), np.mean([abs(x[10] - x[11]) for x in pre])))
        g0 = dec[-1][2] if dec else 0.0
        kind_ = "출발 3 s 안" if t0 < 3.0 else ("뒤에서(간격 < -0.5 m)" if g0 < -0.5 else ("나란히(|간격| <= 0.5 m)" if g0 <= 0.5 else "앞에서(간격 > 0.5 m)"))
        kinds[kind_] += 1
        print(f"  세게 {t0:.2f} s ({kind_})  t  간격  상대dx  dy  내속도 상대속도 게이트 | kappa1 실제/지움 | v1 실제/지움", flush=True)
        for x in dec[::2]:
            if t0 - x[1] <= 1.5:
                print(f"    {x[1]:5.2f} {x[2]:+5.2f} {x[3]:+5.2f} {x[4]:+5.2f} {x[5]:4.1f} {x[6]:4.1f}  {x[7]:2d} | {x[8]:+.3f} {x[9]:+.3f} | {x[10]:+.3f} {x[11]:+.3f}", flush=True)
if allpre:
    a = np.array(allpre)
    print(f"== 합계: 세게 접촉 장면 {len(allpre)}개, 앞 1 s 평균 |kappa1 실제-지움| {a[:, 0].mean():.3f} 1/m, |v1 실제-지움| {a[:, 1].mean():.3f}", flush=True)
print("== 종류: " + ", ".join(f"{k} {v}" for k, v in kinds.items()), flush=True)
