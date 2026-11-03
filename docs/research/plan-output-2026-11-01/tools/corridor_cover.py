"""통로 칸이 주행선 위 상자를 담나 (11-03): 일곱 선생(h3, b, d, e, g, g2, w) 모두 같은 30배치 중 5, 11, 20에서 멈춤 -- 학습법이
아니라 표현 문제일 수 있음. 상자 정책이 상자를 아는 길은 둘: 가까운 보이는 상자 3칸(유클리드 거리 순, 옆·벽 너머 상자도 칸을 차지)과
통로 칸(`corridor_block`: 중심선 점마다 그 점의 EDT(가장 가까운 벽까지 거리) w로 [-0.93 w, +0.93 w]를 7칸). 중심선이 넓은 곳의
한쪽 벽(얇은 벽 조각, 섬)에 붙어 있으면 창이 좁아 반대쪽을 지나는 주행선이 창 밖일 수 있음 -> 그 상자는 통로 칸에 안 잡힘.
콘솔과 같은 배치(console_layouts.py: 배치 0..L-1을 40 s씩 달린 뒤 리셋)에서 상자마다: 중심선 s, 가로 위치, 그 s의 창 반폭 0.93 w,
레이싱 라인 가로 위치, 상자만 보인다고 했을 때 그 상자 5 m 앞에서 통로 칸이 몇 칸 막히나. 인자: 체크포인트 맵id 배치수 시드"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
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


run, mp, n_lay, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
env = s["env"]; proc = env.procedural; tr = env.sim.track
lid = env.learner_ids
tid = env.sim.tid[lid]
pl = SR._line_planner(env)
dev = env.device
tot = dict(on=0, on_out=0, all=0, all_out=0)
for L in range(n_lay):
    if L > 0:
        s["obs"], _ = env.reset()
    live = (proc.p_sid[0] >= 0).nonzero().flatten().tolist()
    pose = proc.p_poses[0]
    dd = proc.p_d[0]
    ext = torch.where(torch.isfinite(dd), dd, torch.zeros_like(dd)).amax(-1)
    rows = []
    for c in live:
        xy = pose[c:c + 1, :2].float()
        sc, lat, _ = tr.project(xy, tid[:1])
        xy_cl, _ = tr.pose_at_s(sc, tid[:1])
        wc = float(tr.sample_edt(xy_cl, tid[:1]))
        rl = float(pl.rl_offset(sc[:, None], tid[:1, None]).flatten()[0])
        vis = torch.zeros(1, proc.C, dtype=torch.bool, device=dev); vis[0, c] = True
        nb = []
        for back in (5.0, 3.0, 1.5):
            cb = SR.corridor_block(env, lid[:1], (sc - back) % tr.length[tid[:1]], tid[:1], visible=vis)
            nb.append(int(cb.sum()))
        on = abs(float(lat) - rl) < 0.35 + float(ext[c])
        rows.append((c, float(sc), float(lat), 0.93 * wc, rl, float(ext[c]), on, nb))
        tot["all"] += 1; tot["all_out"] += int(nb[0] == 0)
        if on:
            tot["on"] += 1; tot["on_out"] += int(nb[0] == 0)
    if L < n_lay - 1:
        for k in range(int(40.0 / 0.025)):
            w._step_once(s)
    bad = [r for r in rows if r[6] and r[7][0] == 0]
    print(f"배치 {L}: 상자 {len(rows)}개, 레이싱 라인 위 {sum(r[6] for r in rows)}개, 그중 5 m 앞 통로 칸에 안 잡힘 {len(bad)}개", flush=True)
    for r in rows:
        if r[6]:
            c, sc_, lat, half, rl, ex, on, nb = r
            print(f"    상자 #{c:2d}: s {sc_:6.2f} m, 가로 {lat:+.2f} (라인 {rl:+.2f}), 창 반폭 {half:.2f}, 상자 반크기 {ex:.2f} -> "
                  f"막힌 칸 5 m/3 m/1.5 m 앞: {nb[0]}/{nb[1]}/{nb[2]}", flush=True)
print(f"합계: 상자 {tot['all']}개 중 5 m 앞 통로 칸에 안 잡힘 {tot['all_out']}개 | 라인 위 {tot['on']}개 중 {tot['on_out']}개", flush=True)
