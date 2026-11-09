"""경주 세션에서 MAP을 우리 차 몫에서 뺀 수정(11-09)이 주행을 바꾸지 않았는지: 같은 시드의 콘솔 1:1 경주 세션을 두 번 -- 지금 코드 그대로 /
MAP을 예전처럼 우리 차까지 풀게 강제(tracker.map_rows = 정책 차 전부) -- 돌려 10 s 동안 두 차의 위치를 비교. 같아야 함(MAP의 우리 차 답은
외부 명령이 덮어쓰므로).
인자: checkpoint [상대 종류] [상대 속도 배수]"""
import os, sys
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


run = sys.argv[1]; kind = sys.argv[2] if len(sys.argv) > 2 else "forzaeth_pred"; sc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85


def drive(force_old: bool):
    torch.manual_seed(0)
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
    s = w.build_session(P.SessionConfig(run=run, map_name="real/iccas25", seed=7, races=1, cars_per_race=2, device="cuda",
                                        compile=False, collision_mode="soft", procedural=False, episode_s=120.0, randomize=False,
                                        opponent="slots", opponent_slots=[{"kind": kind, "speed_scale": [sc, sc]}]), 0)
    env = s["env"]
    if force_old:
        trk = env.tracker
        orig = type(trk).__call__

        def call_old(self_, *a, **k):
            self_.map_rows = env.learner                      # MAP for the policy's car too, as before the change
            return orig(self_, *a, **k)
        type(trk).__call__ = call_old
    pos = []
    for k in range(400):
        w._step_once(s)
        if k % 8 == 7:
            w._snapshot(s)
        pos.append(env.sim.state[:, :4].clone())
    if force_old:
        type(env.tracker).__call__ = orig
    return torch.stack(pos, 1).cpu()


new = drive(False)
old = drive(True)
d = (new[..., :2] - old[..., :2]).norm(dim=-1)                 # (cars, steps)
print(f"상대 {kind} x{sc}: 10 s 동안 위치 차이 최대 -- 우리 차 {float(d[0].max()) * 100:.3f} cm, 상대 차 {float(d[1].max()) * 100:.3f} cm | "
      f"10 s 뒤 우리 차 속도 새 {float(new[0, -1, 3]):.2f} / 예전 {float(old[0, -1, 3]):.2f} m/s", flush=True)
