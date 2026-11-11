"""크기를 줄인 트랙의 헤어핀을 차가 돌 수 있나 (11-12: 크기 변형 없이 학습한 plan25r u300이 x0.60 / 0.63 / 0.68 트랙의 같은 헤어핀에서
벽에 세게 닿음 -- 혼자 벽에 닿은 126건 중 87건. 환경이 못 도는 코너를 만든 것인지부터 확인).
트랙마다 캐시된 주행선(상대 차가 달리는 선)의 가장 큰 곡률을 차의 최소 회전 반경과 견주고, 선 위에서 벽까지 여유, 헤어핀 자리의 통로 폭을 냄.
GPU를 쓰지 않음(CUDA_VISIBLE_DEVICES= 로 돌릴 것: 4070에 다른 작업이 뜨면 guard가 학습을 세움). 인자 없음."""
import os, sys, glob
import numpy as np
from f1sim import maps
from f1sim.raceline import Raceline
from f1sim.track import curvature_np
from f1sim.params import VehicleParams

vp = VehicleParams()
kmax = float(np.tan(vp.s_max) / (vp.lf + vp.lr))
hw = 0.5 * float(vp.width)
print(f"차: 조향 끝 {np.degrees(vp.s_max):.1f} deg, 축거 {vp.lf + vp.lr:.3f} m -> 최소 회전 반경 {1 / kmax:.2f} m (곡률 {kmax:.2f}/m), 차 폭 {2 * hw:.2f} m")
P = "real:korea_2025_iccas"
spots = {"+warp0+scale060": (4.6, -3.3), "+warp7+scale063": (5.0, -3.7), "+warp14+scale065": None, "+warp5+scale068": (5.5, -3.7),
         "+warp12+scale071": (6.0, -3.8), "+warp3+scale074": None, "": None}
for suf, spot in spots.items():
    name = P + suf
    tr = maps.load(name)
    fs = sorted(glob.glob(os.path.expanduser(f"~/.cache/f1sim/racelines/{name}_*.csv")))
    if not fs:
        print(name, "주행선 캐시 없음"); continue
    for f in fs[:1] if suf else fs[:1]:
        rl = Raceline.load(f)
        xy = np.asarray(rl.xy, float); v = np.asarray(rl.v, float) if hasattr(rl, "v") else np.asarray(rl.speed, float)
        k = np.abs(curvature_np(xy))
        r, c = tr.world_to_grid(xy)
        ri = np.clip(np.round(r).astype(int), 0, tr.occupancy.shape[0] - 1); ci = np.clip(np.round(c).astype(int), 0, tr.occupancy.shape[1] - 1)
        cl = tr.edt[ri, ci]
        L = float(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1).sum())
        i = int(k.argmax())
        print(f"{suf or '(원래)':<18} 한 바퀴 {L:5.1f} m, 점 {len(xy)} | 곡률 최대 {k.max():.2f}/m (반경 {1 / k.max():.2f} m, 한계의 {100 * k.max() / kmax:.0f} %) 자리 ({xy[i, 0]:.1f}, {xy[i, 1]:.1f}) "
              f"거기 속도 {v[i]:.1f} m/s | 벽까지 여유(차 중심) 최소 {cl.min():.2f} m = 차 옆면에서 {cl.min() - hw:.2f} m | 통로 폭(= 2 x 벽 거리 최대) 중앙 {2 * np.median(cl):.2f} m")
        if spot is not None:
            d = np.linalg.norm(xy - np.array(spot), axis=1); m = d < 2.0
            if m.any():
                j = np.nonzero(m)[0][k[m].argmax()]
                nrm = np.array([-(xy[(j + 1) % len(xy)] - xy[j - 1])[1], (xy[(j + 1) % len(xy)] - xy[j - 1])[0]]); nrm /= np.linalg.norm(nrm)
                wl = tr.free_width_along(xy[j], nrm); wr = tr.free_width_along(xy[j], -nrm)
                print(f"      벽에 몰려 닿은 헤어핀 {spot} 2 m 안: 곡률 최대 {k[m].max():.2f}/m (반경 {1 / k[m].max():.2f} m, 한계의 {100 * k[m].max() / kmax:.0f} %), 주행선 속도 최저 {v[m].min():.1f} m/s, "
                      f"벽 여유 최소 {cl[m].min():.2f} m | 가장 꺾는 점에서 벽까지 왼 {wl:.2f} · 오 {wr:.2f} m")
