"""학습 세계에 혼자 주행선 위 상자가 얼마나 있나 (11-03): 상자 정책(--vis-actor)은 보이는 상자 하나를 주행선 위에 두면 12곳 중 5곳에서
선을 그대로 탄 채 박음(가로 위치가 상자 없을 때와 같음). 첫 층 가중치: 혼자 정책 입력 열 1.0, 새 상자 입력 열(0에서 시작) 200회 뒤 0.14~0.16
-- 상자에 반응하라는 학습 신호가 약함. 가설: 상자 생성기는 패턴을 한쪽 벽에서 재는 띠 안에 놓고 1.2 m 넘는 틈을 남기므로, 학습 배치에서
상자가 혼자 주행선을 막는 일이 드묾 -> 상자를 피해야 하는 결정이 드묾. 측정: 콘솔 워커로 혼자 정책(상자 정책 혼합의 빈 길)이 실제 달린 한
바퀴의 길, 학습 세계 상자 생성기(같은 체크포인트 인자)로 배치 수천 개를 뽑아 상자 바닥 다각형과 그 길의 거리 < 차 반폭 0.155 m(그대로
달리면 닿음)인 상자를 셈. 개수 방식 uniform(지금 학습) / fixed(콘솔 학습 장애물).
인자: 맵id 혼합 체크포인트 상자 체크포인트(학습 인자) 배치 수"""
import os, sys, math, argparse
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.learn import state_rl as SR
from f1sim import props as _props


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


mp, mix, pck, n_lay = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
dev = torch.device("cuda")
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=mix, map_name=mp, seed=0, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=False, episode_s=3600.0, randomize=False), 0)
env = s["env"]
xy = []
for k in range(int(12.0 / 0.025)):
    w._step_once(s)
    if k >= 120:
        xy.append((float(env.sim.state[0, 0]), float(env.sim.state[0, 1])))
path = torch.tensor(np.array(xy), device=dev, dtype=torch.float32)            # a lap and more of the solo line
del w, s, env
torch.cuda.empty_cache()
HALF = 0.155
ck = torch.load(pck, map_location=dev, weights_only=False)
for mode in ("uniform", "fixed"):
    a = argparse.Namespace(**ck["args"])
    a.envs, a.seed, a.device, a.prop_count = 256, 3, "cuda", mode
    env = SR.build_env(a, dev)
    proc = env.procedural
    fps = []
    for sh in proc.shapes:
        fps.append(np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float))
    # each footprint as points along its edges, prop frame (S, K, 2)
    K = 24
    pts = []
    for fp in fps:
        e = np.concatenate([fp, fp[:1]], 0)
        seg = [e[i] + (e[i + 1] - e[i]) * t for i in range(len(fp)) for t in np.linspace(0, 1, K // len(fp) + 1, endpoint=False)]
        seg = np.array(seg)[:K] if len(seg) >= K else np.concatenate([np.array(seg), np.repeat(np.array(seg)[-1:], K - len(seg), 0)])
        pts.append(seg)
    pts = torch.tensor(np.stack(pts), device=dev, dtype=torch.float32)       # (S, K, 2)
    per_layout, n_all, n_on, empty = [], 0, 0, 0
    done = 0
    while done < n_lay:
        env.reset()
        sid = proc.p_sid.clone(); pose = proc.p_poses.clone()                 # (B, C), (B, C, 3)
        live = sid >= 0
        P_ = pts[sid.clamp_min(0)]                                            # (B, C, K, 2)
        c, sn = torch.cos(pose[..., 2])[..., None], torch.sin(pose[..., 2])[..., None]
        wx = pose[..., 0:1] + P_[..., 0] * c - P_[..., 1] * sn
        wy = pose[..., 1:2] + P_[..., 0] * sn + P_[..., 1] * c
        W = torch.stack([wx, wy], -1).reshape(-1, 2)                          # (B*C*K, 2)
        dmin = torch.cdist(W, path).amin(-1).reshape(sid.shape[0], sid.shape[1], K).amin(-1)   # (B, C)
        on = live & (dmin < HALF)
        nl = live.sum(1); no = on.sum(1)
        per_layout += no.tolist()
        n_all += int(nl.sum()); n_on += int(no.sum()); empty += int((nl == 0).sum())
        done += sid.shape[0]
    per_layout = np.array(per_layout)
    print(f"== 개수 {mode}: 배치 {done}개 (상자 없는 배치 {empty / done * 100:.0f} %), 배치당 상자 {n_all / done:.1f}개, "
          f"그중 혼자 주행선 위(그대로 달리면 닿음) {n_on / max(n_all, 1) * 100:.1f} % = 배치당 {n_on / done:.2f}개 | "
          f"주행선 위 상자가 하나라도 있는 배치 {np.mean(per_layout > 0) * 100:.0f} %", flush=True)
    del env
    torch.cuda.empty_cache()
