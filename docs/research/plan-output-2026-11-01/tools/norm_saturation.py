"""선생 입력 정규화 포화 진단 (11-03, 사용자: "장애물 없이 1대1만 해도 출력이 망가져. occupancy 해상도가 낮아서 가깝다고 판단하나?").
선생의 입력 정규화 RunningNorm은 (x - 평균) / (표준편차 + 1e-6)을 ±8로 자름 -> 학습 중 한 번도 안 변한 칸(표준편차 0)은 콘솔에서
조금만 달라져도 ±8. 콘솔 워커 세션(혼자 / 학습 장애물 / 1대1 장애물 없음)에서 결정마다 두 정책(상자·경주 정책, 혼자 정책)이
실제로 받는 입력을 정규화해, ±7.9 이상으로 포화된 칸이 몇 개인지, 어느 묶음(내 상태·상대 칸·상자 칸·앞길·통로·라이다)인지 센다.
인자: 체크포인트 맵id 조건(solo/props/race) 초 시드"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.learn import state_rl as SR
from f1sim.gym_env import PRIV_V2_OPP_KEYS, PRIV_V2_OPP_SLOTS, PRIV_V2_PROP_DIM


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, cond, secs, seed = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
REC = []
_orig = SR.SoloMix.__call__


def _rec(self, a_pol, last):
    out = _orig(self, a_pol, last)
    tp = TP[0]
    ob = SR.blank_opponents(self.sobs(last.clamp(-1, 1))); ob[:, self.prop_cols] = 0.0
    REC.append((tp.o[0].detach().clone(), ob[0].detach().clone(), bool(self.gate[0])))
    return out


SR.SoloMix.__call__ = _rec
TP = []
_tp_init = SR.TeacherPolicy.__init__


def _tp_init_rec(self, *a, **k):
    _tp_init(self, *a, **k); TP.append(self)


SR.TeacherPolicy.__init__ = _tp_init_rec
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
kw = dict(run=run, map_name=mp, seed=seed, races=1, device="cuda", compile=False, collision_mode="soft", episode_s=3600.0,
          randomize=False)
if cond == "race":
    kw.update(cars_per_race=2, procedural=False, opponent_slots=[{"kind": "forzaeth", "speed_scale": [0.8, 0.8]}])
else:
    kw.update(cars_per_race=1, procedural=(cond == "props"))
s = w.build_session(P.SessionConfig(**kw), 0)
env = s["env"]
for k in range(int(secs / 0.025)):
    w._step_once(s)
tp = TP[0]; mix = tp.mix
pvd = int(env.privileged_v2().shape[1])
ko = PRIV_V2_OPP_SLOTS * len(PRIV_V2_OPP_KEYS)


def groups(dim, pol):
    """column index -> group name for an observation of this layout (privileged vector first, then the extras)"""
    g = ["?"] * dim
    for i in range(min(pvd, dim)):
        g[i] = "내 상태" if i < 8 else ("상대 차 칸" if i < 12 + ko else ("상자 칸" if i < 12 + ko + PRIV_V2_PROP_DIM else "내 상태(시간)"))
    j = pvd
    for name, n in (("앞길", 52), ("조향", 1), ("지난 행동", 8)):
        for i in range(j, min(j + n, dim)):
            g[i] = name
        j += n
    a = tp.ar if pol else {}
    # StateObs: [privileged, ahead, steer, last action, scan, (plan free), corridor, (car channel)]
    if pol and a.get("scan_obs"):
        for i in range(j, min(j + 108, dim)):
            g[i] = "라이다"
        j += 108
    if pol and a.get("corridor_obs"):
        if dim - j == 92:
            g[j] = "계획 여유"; j += 1
        for i in range(j, min(j + 91, dim)):
            g[i] = "통로"
        j += 91
    return g


def report(label, X, nrm, pol):
    Y = nrm(X)
    sat = Y.abs() >= 7.9
    g = groups(X.shape[1], pol)
    zero_var = (nrm.var.sqrt() < 1e-3).cpu().numpy()
    per = sat.float().sum(1).cpu().numpy()
    print(f"  {label}: 결정당 포화(±8) 칸 평균 {per.mean():.1f}, 최대 {per.max():.0f} / {X.shape[1]}칸 "
          f"(학습 중 안 변한 칸 {int(zero_var.sum())}개)")
    cnt = {}
    S = sat.cpu().numpy()
    for i in np.nonzero(S.any(0))[0]:
        cnt.setdefault(g[i], []).append((i, S[:, i].mean(), bool(zero_var[i])))
    for k_, v_ in cnt.items():
        z = sum(1 for _, _, zv in v_ if zv)
        print(f"     {k_}: {len(v_)}칸이 포화된 적 있음(그중 학습 중 안 변한 칸 {z}), 포화 비율 평균 {np.mean([m for _, m, _ in v_]) * 100:.0f} %")


Xp = torch.stack([r[0] for r in REC]); Xb = torch.stack([r[1] for r in REC]); G = np.array([r[2] for r in REC])
print(f"== {os.path.basename(os.path.dirname(run))} @ {mp} {cond} {secs:.0f} s: 결정 {len(REC)}, 상자·경주 정책 운전 {G.mean() * 100:.0f} %")
report("상자·경주 정책 입력 (운전한 결정만)" if G.any() else "상자·경주 정책 입력", Xp[torch.from_numpy(G)] if G.any() else Xp, tp.nrm, True)
report("혼자 정책 입력 (운전한 결정만)" if (~G).any() else "혼자 정책 입력", Xb[torch.from_numpy(~G)] if (~G).any() else Xb, mix.nrm, False)
