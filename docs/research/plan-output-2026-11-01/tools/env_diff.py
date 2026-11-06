"""콘솔 세션 env와 학습 env의 설정 차이 (11-06 14:20). 같은 checkpoint로 학습 env를 평가처럼(상자 전부, 원래 트랙, 상대 없음) 맞춰도 2 s 넘게
서 있는 시간 5.8 %, 콘솔 평가는 50-57 % -> 콘솔 세션이 학습과 다른 차/세계를 만드는지. EnvConfig와 Config(차량, LiDAR, sim, 무작위화)의
모든 칸을 비교해 다른 것만 출력. 인자: 체크포인트 맵id"""
import os, sys, argparse, dataclasses
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.learn import state_rl as SR


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


path, mp = sys.argv[1], sys.argv[2]
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=path, map_name=mp, seed=777, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
env_c = s["env"]
ck = torch.load(path, map_location="cuda", weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = 4, 1, "cuda"
a.prop_count, a.prop_fraction, a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop = "fixed", 1.0, 0.0, 0, 0.0, 0.0
env_t = SR.build_env(a, torch.device("cuda"))


def flat(x, pre=""):
    out = {}
    if dataclasses.is_dataclass(x):
        for f in dataclasses.fields(x):
            out.update(flat(getattr(x, f.name), f"{pre}{f.name}."))
    elif isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{pre}{k}."))
    else:
        out[pre[:-1]] = x
    return out


for name, xc, xt in (("EnvConfig", env_c.ecfg, env_t.ecfg), ("Config", env_c.cfg, env_t.cfg)):
    fc, ft = flat(xc), flat(xt)
    keys = sorted(set(fc) | set(ft))
    diff = [(k, fc.get(k, "-"), ft.get(k, "-")) for k in keys if repr(fc.get(k, "-")) != repr(ft.get(k, "-"))]
    print(f"== {name}: 다른 칸 {len(diff)}개 (콘솔 | 학습)")
    for k, vc, vt in diff:
        print(f"  {k}: {repr(vc)[:90]} | {repr(vt)[:90]}")
print("== sim.P (차량 물리 값, 첫 차) 다른 것:")
for k in sorted(set(env_c.sim.P) & set(env_t.sim.P)):
    vc, vt = env_c.sim.P[k], env_t.sim.P[k]
    try:
        a_, b_ = float(vc.flatten()[0]), float(vt.flatten()[0])
        if abs(a_ - b_) > 1e-6 * max(1.0, abs(a_)):
            print(f"  {k}: {a_:.4g} | {b_:.4g}")
    except Exception:
        pass
