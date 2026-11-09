"""콘솔에서 plan policy가 왜 느린가 (11-09, 사용자: "simtime 왜이렇게 느려?"). 콘솔 워커(SimWorker.build_session, 그래프 빠른 경로
포함)로 혼자 달리기 세션을 만들고 400걸음(10 s 시뮬)을 돌려 실제 걸린 시간 -> 배속, 그리고 cProfile로 어디서 시간이 드는지.
인자: checkpoint [걸음 수]   (F1SIM_PLAN_MODE = track / map / ideal)"""
import os, sys, time, cProfile, pstats, io
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
w = sim_worker.SimWorker(_N(), _N()); logs = []
w.say = lambda k, **f: logs.append(f.get("text") or ""); w.stage = lambda *a, **k: None; w._hold_parked_ok = True
w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
#: F1SIM_PROF_OPP=<kind> (11-09, the user: "상대차 있는곳도 simtime 신경좀"): the console's 1:1 race, one opponent of that kind
#: at the console's default speeds (ForzaETH with prediction, 0.7-1.0)
OPP = os.environ.get("F1SIM_PROF_OPP", "")
s = w.build_session(P.SessionConfig(run=run, map_name="real/iccas25", seed=0, races=1, cars_per_race=2 if OPP else 1, device="cuda",
                                    compile=False, collision_mode="soft", procedural=False, episode_s=120.0, randomize=False,
                                    opponent="slots" if OPP else "teacher",
                                    opponent_slots=[{"kind": OPP, "speed_scale": [0.7, 1.0]}] if OPP else None), 0)
print("모드", os.environ.get("F1SIM_PLAN_MODE", "track"), "| 상대", OPP or "없음", "| 그래프:", [x for x in logs if "그래프" in x][:2])
for _ in range(120):
    w._step_once(s)
torch.cuda.synchronize()
t0 = time.time()
for k in range(n):
    w._step_once(s)
    if k % 8 == 7:
        w._snapshot(s)                    # the console sends a frame about every 8 steps (~5 per 0.2 s at 0.5x...)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"{n}걸음 = 시뮬 {n * 0.025:.1f} s, 실제 {dt:.2f} s -> 배속 {n * 0.025 / dt:.2f}x, 걸음당 {dt / n * 1000:.1f} ms", flush=True)
pr = cProfile.Profile(); pr.enable()
for k in range(200):
    w._step_once(s)
    if k % 8 == 7:
        w._snapshot(s)
torch.cuda.synchronize()
pr.disable()
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(40)
print("\n".join(l[:170] for l in st.getvalue().splitlines()[:70]))
