"""크기를 바꾼 트랙 변형(+scaleNNN)에서 상대 차가 따라갈 주행선을 미리 만들어 캐시에 넣음 (11-12).
학습 세계와 같은 설정(가속 한계 7 / 6.5 / 4, 최소 시간)으로 Raceline.build_cached -- 변형의 바탕 트랙 주행선을 같은 배율로 늘려
시작점으로 씀. 트랙마다 걸린 시간, 한 바퀴 시간, 벽과의 최소 거리, 풀이 상태를 출력.
인자: 몇번째부터 몇개씩 건너뛰며 [크기 목록(퍼센트, 쉼표)]   예: 0 3  -> 0, 3, 6, ...번째 변형"""
import os, sys, time
os.environ.pop("F1SIM_RACELINE_CACHE_ONLY", None)
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import maps
from f1sim.raceline import Raceline
from f1sim.learn import common, state_rl as SR

first, step = int(sys.argv[1]), int(sys.argv[2])
SC = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else SR.TRACK_SCALES_25
names = SR.scale_variants(["real:korea_2025_iccas"], 16, SC)[first::step]
kw = dict(common.teacher_limits(7.0, 6.5, 4.0), objective="min_time")
for n in names:
    t0 = time.time()
    t = maps.load(n)
    rl = Raceline.build_cached(t, **kw)
    c = common.raceline_clearance(t, rl)
    st = (rl.optimization or {}).get("status", "cache")
    print(f"{n}: {(time.time() - t0) / 60:.1f} 분, 한 바퀴 {rl.lap_time:.2f} s, 길이 {rl.length:.1f} m, 벽과 최소 {c:.3f} m, 상태 {st}", flush=True)
print("== 끝", flush=True)
