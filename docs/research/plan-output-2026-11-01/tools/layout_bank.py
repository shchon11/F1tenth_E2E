"""경로 생성기로 판정한 상자 배치 은행 (11-03 밤). 사용자: "고르게 뽑히라고 uniform하게. 다만 지나갈 차 폭 고려해서 아예 못지나가게만
배치하지 말고" + "그 주행가능한지는 path generator 로직 사용해서 판단하는거지?" -- 생성기 안의 0.70 m 격자 검사는 회전 반경을 모름.
그래서: 균일 배치(ProceduralObstacles place="uniform", 판당 상자 C개 전부)를 트랙(원본 + 뒤튼 16개)마다 뽑아 배치별 최소시간 라인 생성기
(layout_line.LayoutLines: 상자·벽 거리장 안에서 스플라인 라인을 랩 시간으로 최적화, 조향 한계 곡률 벌점)에 넣고, 나온 라인이
  - 차 몸체(0.58 x 0.31 m)가 라인을 따라 상자와 0.10 m, 벽과 0.10 m 이상 떨어지고
  - 곡률(0.2 m 평활)이 조향 한계 tan(s_max)/휠베이스 이하
이면 "지나갈 수 있음". 못 지나가면 그 라인이 0.10 m를 어긴 상자와 곡률 넘친 곳 1 m 안 상자만(없으면 가장 가까운 하나) 빼고 다시
판정, 4번까지. (처음엔 라인 0.30 m 안 상자를 다 뺐는데, 한국 2026에서 판당 3~4개 = 길을 안 막은 라인 근처 상자까지 지워 다시 라인
근처가 비는 편향이라 바꿈)
통과한 배치만 트랙 이름별로 저장 -> 학습/콘솔은 리셋마다 같은 트랙 은행에서 하나를 꺼내 그 일부(1..전부)를 씀(길이 있는 배치에서
상자를 빼도 그 길은 남음).
인자: 체크포인트(학습 인자) 출력.pt 트랙당배치수 [한번에판정할배치수 최적화반복]"""
import os, sys, time, argparse
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim import layout_line as LL
from f1sim.prop_math import _polygon_vertices, _slot_is_live, _to_world
from f1sim.raceline import curvature

pck, out, n_per = sys.argv[1], sys.argv[2], int(sys.argv[3])
nb = int(sys.argv[4]) if len(sys.argv) > 4 else 32
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 300
CLR_PROP, CLR_WALL, ROUNDS = 0.10, 0.10, 4
dev = torch.device("cuda")
ck = torch.load(pck, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = 64, int(time.time()) % 100000, "cuda"
a.prop_place, a.prop_count, a.prop_fraction = "uniform", "fixed", 1.0
env = SR.build_env(a, dev)
proc = env.procedural
names = getattr(env, "_track_names", None) or [SR.KOREA]
_, rls = SR.common.load_tracks(names, racelines=True, **dict(SR.common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
tracks = env.sim.track.tracks
T = len(tracks)
assert len(rls) == T, (len(rls), T)
builders = [LL.LayoutLines(tracks[i], rls[i], dev, veh_width=float(env.cfg.vehicle.width), vehicle=env.cfg.vehicle, iters=iters)
            for i in range(T)]
kmax = builders[0].kmax
C = proc.C
print(f"== 은행 {out}: 트랙 {T}개 ({tracks[0].name} ...), 트랙당 {n_per}, 판정 묶음 {nb}, 최적화 {iters}번, 슬롯 {C}, 곡률 한계 {kmax:.3f}", flush=True)


def row_pieces(b):
    """row b's live slots: (sid, pose (3,), world polygon)"""
    poses, pn, pd, zlo, zhi = proc.slots()
    live = _slot_is_live(pn[b], zlo[b], zhi[b]) & (proc.p_sid[b] >= 0)
    outp = []
    for c in torch.nonzero(live).flatten().tolist():
        nw, dw = _to_world(pn[b, c][None], pd[b, c][None], poses[b, c][None])
        valid = pn[b, c].pow(2).sum(-1) > 0.5
        poly = _polygon_vertices(nw, dw, valid[None])[0][valid].cpu().numpy().astype(float)
        outp.append((int(proc.p_sid[b, c]), poses[b, c].cpu().numpy().astype(np.float32), poly))
    return outp


bank = {i: [] for i in range(T)}
pend = {i: [] for i in range(T)}          # (round, pieces)
st = dict(judged=0, ok_first=0, fixed=0, dropped=0, removed=0, start_pieces=0, kept_pieces=0)
t0 = time.time()


def judge(ti):
    batch, pend[ti] = pend[ti][:nb], pend[ti][nb:]
    lines, laps, clr, _ = builders[ti].build([[p[2] for p in ps] for _, ps in batch])
    for (rnd, ps), xy, lp, (cp, cw) in zip(batch, lines, laps, clr):
        st["judged"] += 1
        kap = np.abs(uniform_filter1d(curvature(xy), 5, mode="wrap"))
        good = cp >= CLR_PROP and cw >= CLR_WALL and kap.max() <= kmax and np.isfinite(lp)
        if good:
            st["ok_first" if rnd == 0 else "fixed"] += 1
            st["kept_pieces"] += len(ps)
            bank[ti].append((ps, float(lp)))
            continue
        if rnd >= ROUNDS - 1:
            st["dropped"] += 1
            continue
        cpi = np.array([LL.check_line(xy, [p[2]], tracks[ti])[0] for p in ps])
        rm = cpi < CLR_PROP
        hot = xy[kap > kmax]
        if len(hot):
            cen = np.array([p[2].mean(0) for p in ps])
            rm |= np.linalg.norm(cen[:, None] - hot[None], axis=-1).min(1) < 1.0
        if not rm.any():
            rm[int(np.argmin(cpi))] = True
        st["removed"] += int(rm.sum())
        pend[ti].append((rnd + 1, [p for p, r in zip(ps, rm) if not r]))


last = 0.0
while min(len(bank[i]) for i in range(T)) < n_per:
    env.reset()
    tid = env.sim.tid.cpu().numpy()
    for b in range(env.B):
        ti = int(tid[b])
        if len(bank[ti]) + len(pend[ti]) >= n_per + nb:
            continue
        ps = row_pieces(b)
        if ps:
            st["start_pieces"] += len(ps)
            pend[ti].append((0, ps))
    for ti in range(T):
        while len(pend[ti]) >= nb and len(bank[ti]) < n_per:
            judge(ti)
    if time.time() - last > 120:
        last = time.time()
        n = max(st["judged"], 1)
        print(f"  [{(time.time() - t0) / 60:5.1f} 분] 은행 트랙별 최소 {min(len(bank[i]) for i in range(T))} / {n_per}, 판정 {st['judged']}: "
              f"한 번에 통과 {st['ok_first']}, 상자 빼고 통과 {st['fixed']}, 버림 {st['dropped']}, 뺀 상자 {st['removed']}", flush=True)

per = {}
for ti in range(T):
    items = bank[ti][:n_per]
    sid = torch.full((len(items), C), -1, dtype=torch.long)
    pose = torch.zeros(len(items), C, 3)
    for k, (ps, _) in enumerate(items):
        for c, p in enumerate(ps[:C]):
            sid[k, c] = p[0]
            pose[k, c] = torch.as_tensor(p[1])
    per[tracks[ti].name] = dict(sid=sid, pose=pose, lap=torch.tensor([lp for _, lp in items]),
                                length=float(env.sim.track.length[ti]))
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(dict(per_track=per, C=C, judge=dict(clr_prop=CLR_PROP, clr_wall=CLR_WALL, kmax=kmax, rounds=ROUNDS, iters=iters,
                                               source=os.path.basename(os.path.dirname(pck)))), out)
n = max(st["judged"], 1)
print(f"== 끝 {(time.time() - t0) / 60:.1f} 분: 판정 {st['judged']}, 한 번에 통과 {st['ok_first'] / n:.0%}, 상자 빼고 통과 {st['fixed'] / n:.0%}, "
      f"버림 {st['dropped'] / n:.0%}, 처음 상자 {st['start_pieces']} 중 뺀 것 {st['removed']}, 은행 판당 상자 "
      f"{np.mean([len(ps) for i in range(T) for ps, _ in bank[i][:n_per]]):.1f}, 저장 {out}", flush=True)
