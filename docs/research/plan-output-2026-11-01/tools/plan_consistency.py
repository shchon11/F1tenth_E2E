"""정책 궤적이 차가 실제로 달린 길과 얼마나 맞는지 (11-01): 콘솔 워커로 혼자 달리며 결정마다 콘솔이 그리는 궤적(tracker.last_ref,
0.6 s 걸음)을 기록하고, 궤적 위 각 점에서 실제 주행 경로(그 뒤 2 s)까지의 거리를 잰다.
구간별: 정책이 낸 계획 [0, L] (L = 0.3 s x 속도), MAP 조준점까지 [0, L1], 콘솔이 그리는 전체. 인자: 체크포인트 맵id 초 [그림]"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
png = sys.argv[4] if len(sys.argv) > 4 else ""
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=False, episode_s=max(60.0, secs + 5), randomize=False), 0)
env = s["env"]
hz = float(env.tracker.spec.wp_horizon_s)
mp_ = env.tracker.map.p if hasattr(env.tracker, "map") else None
pos, plans = [], []
for k in range(int(secs / 0.025)):
    w._step_once(s)
    st = env.sim.state[0]
    x, y, yaw, v = (float(st[i]) for i in range(4))
    pos.append((x, y))
    if k % 2 == 0 and env.tracker.last_ref is not None:
        ref = env.tracker.last_ref[0].detach().cpu().numpy().astype(float)
        cs, sn = np.cos(yaw), np.sin(yaw)
        pw = np.stack([x + ref[:, 0] * cs - ref[:, 1] * sn, y + ref[:, 0] * sn + ref[:, 1] * cs], 1)
        L = min(max(hz * abs(v), 1.0), 6.0)
        L1 = -0.4957 + 0.4732 * abs(v)
        L1 = min(max(L1, 0.957), 4.78)
        plans.append((k, pw, L, L1, v))
pos = np.asarray(pos)


def dist_to_path(p, path):
    a, b = path[:-1], path[1:]
    ab = b - a; t = np.clip(((p[None] - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    q = a + ab * t[:, None]
    return float(np.min(np.linalg.norm(q - p[None], axis=1)))


rows = []
for k, pw, L, L1, v in plans:
    fut = pos[k:k + 80]                                   # the next 2 s actually driven
    if len(fut) < 80 or v < 1.0:
        continue
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pw, axis=0), axis=1))]
    d = np.array([dist_to_path(p, fut) for p in pw])
    rows.append((d[seg <= L].max(), d[seg <= L1].max(), d.max(), seg[-1] / L))
r = np.asarray(rows)
print(f"{os.path.basename(os.path.dirname(run))} @ {mp}: 결정 {len(r)}개 | 그리는 궤적 길이 = 계획 L의 {np.median(r[:, 3]):.1f}배")
for i, name in enumerate(("계획 [0, L]", "MAP 조준점까지 [0, L1]", "콘솔이 그리는 전체")):
    print(f"  {name}: 실제 경로에서 가장 먼 점 중앙 {np.median(r[:, i]):.2f} m, p90 {np.percentile(r[:, i], 90):.2f} m, "
          f"0.3 m 넘는 결정 {100 * (r[:, i] > 0.3).mean():.0f} %")
