"""차-차 접촉 세기 분포 (11-02 21:30): 콘솔은 1.0 m/s 이상(contact_severity.HARD_MPS)을 세게로 세는데 학습 보상은 1.5 m/s
이상(contact_severity_min)만 값을 매기고, 또 우리가 0.6 m 넘게 앞서 있으면 상대 책임으로 보고 안 매김(gym_env.car_contact_charge)
-> 콘솔 경주(console_races.py와 같은 워커·판)의 차 접촉 시작마다 닫힘 속도(콘솔 계기와 같은 수)와 그 순간 앞뒤 간격을 모아,
학습이 값을 매기는 것 / 세기 때문에 공짜 / 책임 때문에 공짜로 나눔.
인자: 체크포인트 맵id 판당초 판수 상대종류 상대속도배율 시드"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC


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
env = s["env"]
i_on, i_hd = CC.index("onsets_car"), CC.index("hard_car")
imp = []
for L in range(n_race):
    if L > 0:
        s["obs"], _ = env.reset()
    prev = env.sim.contact_frame()[0].float().cpu().numpy()
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        if cf[i_on] > prev[i_on]:
            r = env.last_result
            dn = env.signed_gaps(r.s, env.sim.tid)[0]
            dn = float(dn[dn.abs().argmin()])                                 # the nearest car, as car_contact_charge
            # the console meter's own number (sim._update_contact_frame: car_impact_roll under car_response, which is
            # also what training reads as car_impact) -- StepResult.car_impact is None without measure_impacts
            imp.append((float(env.sim.car_impact_roll[0]), bool(cf[i_hd] > prev[i_hd]), dn))
        prev = cf
v = np.array([x[0] for x in imp]); hd = np.array([x[1] for x in imp]); dn = np.array([x[2] for x in imp])
resp = dn > -0.6                                                              # gym_env.car_contact_charge
edges = [0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 99]
cnt = [int(((v >= a) & (v < b)).sum()) for a, b in zip(edges[:-1], edges[1:])]
print(f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)} vs {kind} x{sc}: 차 접촉 {len(v)} (콘솔 세게 {int(hd.sum())}) | "
      + " ".join(f"[{a},{b}) {c}" for a, b, c in zip(edges[:-1], edges[1:], cnt))
      + f" | 콘솔 세게(1.0 이상) 중: 학습이 값 매김 {int(((v >= 1.5) & resp).sum())}, 세기 때문에 공짜(1.0~1.5) {int(((v >= 1.0) & (v < 1.5) & resp).sum())},"
      f" 우리가 앞서서 공짜 {int(((v >= 1.0) & ~resp).sum())} | 앞뒤 간격 중앙 {np.median(dn) if len(dn) else float('nan'):+.2f} m")
