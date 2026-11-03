"""통로 띠 밖 상자 (11-04 02:05): state_rl의 통로 칸(corridor_block)과 혼자->상자 정책 넘겨받기(SoloMix.near)는 중심선 양옆
CORRIDOR_U(+-0.93) x 그 점의 벽까지 거리(EDT = 가까운 벽)만 훑음 -> 중심선이 치우친 넓은 곳에선 넓은 쪽 띠가 빠짐. 은행 배치(원본 트랙)에서
상자 중 그 띠에 전혀 안 걸리는 비율(상자 + 차 반폭 0.15가 띠 끝 너머), 혼자 정책이 다니는 최소시간 라인이 띠 밖인 랩 비율.
인자: 은행.pt 지도이름(트랙 이름)"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import maps
from f1sim.learn import common
bank, name = sys.argv[1], sys.argv[2]
tr = maps.load(name)
_, rls = common.load_tracks([name], racelines=True, **dict(common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
cl = np.asarray(tr.centerline, float)
t = np.roll(cl, -1, 0) - np.roll(cl, 1, 0); t /= np.linalg.norm(t, axis=1, keepdims=True)
nrm = np.stack([-t[:, 1], t[:, 0]], 1)
res, org, edt = float(tr.resolution), np.asarray(tr.origin, float), tr.edt
def edt_at(p):
    ij = np.floor((p - org) / res).astype(int)
    return edt[np.clip(ij[:, 1], 0, edt.shape[0] - 1), np.clip(ij[:, 0], 0, edt.shape[1] - 1)]
w = edt_at(cl)
def proj(p):
    d = ((p[:, None] - cl[None]) ** 2).sum(-1); j = d.argmin(1)
    return j, ((p - cl[j]) * nrm[j]).sum(1)
b = torch.load(bank, map_location="cpu", weights_only=False)["per_track"][tr.name]
live = b["sid"] >= 0
xy = b["pose"][..., :2][live].numpy().astype(float)
from f1sim.procedural_obstacles import build_catalogue, catalogue_k_pad
shapes, _, _ = build_catalogue(catalogue_k_pad())
half = np.array([0.5 * max(shapes[i].across, shapes[i].along) for i in b["sid"][live].numpy()])
j, v = proj(xy)
out = np.abs(v) - half - 0.15 > 0.93 * w[j]
rl = np.asarray(rls[0].xy, float)
jr, vr = proj(rl)
rout = np.abs(vr) > 0.93 * w[jr]
print(f"== {name}: 은행 상자 {len(xy)}개 중 통로 띠에 안 잡힘 {out.mean():.0%}; 최소시간 라인 점 중 띠 밖 {rout.mean():.0%} "
      f"(띠 반폭 중앙 {np.median(0.93 * w):.2f} m, 중심선 점에서 가까운 벽 중앙 {np.median(w):.2f} m)", flush=True)
s = np.cumsum(np.r_[0, np.linalg.norm(np.diff(rl, axis=0), axis=1)])
runs = np.flatnonzero(np.diff(np.r_[0, rout.astype(int), 0]))
for a_, b_ in zip(runs[::2], runs[1::2]):
    if s[min(b_, len(s) - 1)] - s[a_] > 0.3:
        print(f"   라인이 띠 밖: 라인 {s[a_]:.1f}~{s[min(b_, len(s) - 1)]:.1f} m, 띠 끝에서 최대 {np.max(np.abs(vr[a_:b_]) - 0.93 * w[jr[a_:b_]]):.2f} m 밖", flush=True)
