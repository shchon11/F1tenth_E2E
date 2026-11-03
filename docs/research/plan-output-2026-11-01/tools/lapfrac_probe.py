"""선생이 랩 위치로 선을 외웠나 (11-03): 상태 선생의 행위자 입력(privileged_v2 자아 8칸)에 lap_frac(트랙 몇 % 지점)가 그대로 있음 --
학생(라이다 프레임 + 속도·회전율·지난 행동)에는 없는 특권 입력이고, 첫 층 열 크기 ~1.0(속도 입력과 같음). 반실험에서 상자 입력을 지워도
행동이 거의 같고(곡률 차 0.01~0.05 1/m), 문제 상자는 시작선 근처(s 37~43, 2~4 m)에 몰림 -> 위치에 묶인 외운 선이 상자 반응을 누르나?
콘솔과 같은 SimWorker(깨끗한 맵, 상자 없음, 혼자)에서 선생(혼합: 혼자 정책 + 상자 정책 모두 같은 StateObs)의 lap_frac 칸을
(가) 그대로 (나) +0.05(약 2 m 앞) (다) +0.25 (라) 0.5 고정으로 바꿔 60 s씩: 랩, 벽 접촉. 위치를 외웠다면 (나)~(라)에서 무너짐.
인자: 체크포인트 맵id"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.learn import state_rl as SR
from f1sim.gym_env import PRIV_V2_EGO_KEYS


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp = sys.argv[1], sys.argv[2]
J = PRIV_V2_EGO_KEYS.index("lap_frac")
MODE = [None]
_call = SR.StateObs.__call__


def _patched(self, *a, **k):
    out = _call(self, *a, **k)
    m = MODE[0]
    if m is None:
        return out
    out = out.clone()
    out[:, J] = (out[:, J] + m[1]) % 1.0 if m[0] == "shift" else torch.full_like(out[:, J], m[1])
    return out


SR.StateObs.__call__ = _patched
i_w, i_wh = CC.index("onsets_wall"), CC.index("hard_wall")
for mode, label in ((None, "(가) 그대로"), (("shift", 0.05), "(나) +0.05"), (("shift", 0.25), "(다) +0.25"), (("fix", 0.5), "(라) 0.5 고정")):
    MODE[0] = mode
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=0, races=1, cars_per_race=1, device="cuda", compile=False,
                                        collision_mode="soft", procedural=False, episode_s=3600.0, randomize=False), 0)
    env = s["env"]
    tr = env.sim.track
    lid = env.learner_ids
    s_prev, laps, t_last, vs = None, [], 0.0, []
    cf0 = env.sim.contact_frame()[0].float().cpu().numpy()
    for k in range(int(60.0 / 0.025)):
        w._step_once(s)
        st = env.sim.state[0]
        vs.append(float(st[3]))
        sc, _, _ = tr.project(st[None, :2].float(), env.sim.tid[lid][:1])
        sc = float(sc)
        if s_prev is not None and s_prev > 0.8 * float(tr.length[env.sim.tid[0]]) and sc < 0.2 * float(tr.length[env.sim.tid[0]]):
            t = k * 0.025
            if t_last > 0:
                laps.append(round(t - t_last, 2))
            t_last = t
        s_prev = sc
    cf = env.sim.contact_frame()[0].float().cpu().numpy() - cf0
    print(f"{label}: 랩 {laps} | 평균 속도 {sum(vs) / len(vs):.2f} m/s | 벽 접촉 {int(cf[i_w])} (세게 {int(cf[i_wh])})", flush=True)
    del s, env, w
    torch.cuda.empty_cache()
