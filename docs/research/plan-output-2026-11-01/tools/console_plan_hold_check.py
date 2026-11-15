"""(11-15) MAP 추종기 수정 확인: (1) plan 캐시를 매 걸음 지워도(=예전 계산) 차 상태가 같은가 (2) 붙잡은 plan을 세계 좌표로 그리면 다음 결정까지 그대로인가"""
import os, sys, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P

class _N:
    def send(self, *a): pass
    def poll(self, *a): return False

def session():
    w = sim_worker.SimWorker(_N(), _N()); w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    s = w.build_session(P.SessionConfig(run=sys.argv[1], map_name="real/iccas25", seed=0, races=1, cars_per_race=1, device="cuda",
                                        compile=False, collision_mode="soft", procedural=False, episode_s=120.0, randomize=False,
                                        opponent="teacher"), 0)
    return w, s

def world(ref, pose):
    c, s_ = torch.cos(pose[2]), torch.sin(pose[2])
    return torch.stack([pose[0] + c * ref[:, 0] - s_ * ref[:, 1], pose[1] + s_ * ref[:, 0] + c * ref[:, 1]], 1)

for clear in (False, True):
    torch.manual_seed(0)
    w, s = session(); env = s["env"]; trk = env.tracker
    drift = []
    prev = None
    for k in range(240):
        if clear:
            trk._tw_cache = None
        w._step_once(s)
        ref = trk.last_ref[0]; pose = env.sim.state[0, :3]
        wp = world(ref[:, :2].float(), pose.float())
        held = bool(trk.hold_rows[0]) if trk.hold_rows is not None else False
        if held and prev is not None:
            drift.append(float((wp[-1] - prev[-1]).norm()))
        prev = wp
    st = env.sim.state[0, :4].tolist()
    print(f"캐시 지움={clear}: 240걸음 뒤 차 상태 {[round(x, 4) for x in st]} | 붙잡은 걸음에서 그린 plan 끝점 이동 중앙 {sorted(drift)[len(drift)//2]*100:.2f} cm, 최대 {max(drift)*100:.2f} cm")
