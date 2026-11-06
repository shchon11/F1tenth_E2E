"""시간별 waypoint plan을 따라가는 제어기 앞보기 거리 맞추기 (11-06 22:20, 사용자: "가로질러가면 튜닝좀해 ld가 먼저잖슴"; 제어기는 정해진 경로와의
거리 오차로 맞춤). tw_tracking.py의 네 기준 궤적(중앙선 3·4 m/s, 후진 -> 급회전 전진 둘)을 같은 차·같은 출발로 달리며 MAP의 앞보기
L1 = max(t_clip_min, q_l1 + m_l1 v) 와 후진 pure pursuit 앞보기 max(back_la_min, back_la_t |v|)를 격자로 바꿔 RMSE를 잼.
목표: 네 궤적 RMSE 평균(벽 접촉 한 대당 +5 cm). 인자: 체크포인트(학습 인자만) 차수"""
import sys, os, dataclasses, itertools
sys.path.insert(0, os.path.dirname(__file__))
import tw_tracking as T

trk = T.env.tracker
p0 = trk.map.p
rows = []
grid_f = [dict(t_clip_min=c, m_l1=m, q_l1=q) for c, m, q in itertools.product((0.35, 0.5, 0.65, 0.8, 0.96), (0.3, 0.47), (-0.5, 0.0))]
grid_b = [(0.35, 0.3), (0.45, 0.4), (0.6, 0.5), (0.8, 0.5)]
print(f"지금 MAP: t_clip_min {p0.t_clip_min:.3f}, m_l1 {p0.m_l1:.3f}, q_l1 {p0.q_l1:.3f}, 후진 {trk.back_la_min}/{trk.back_la_t}", flush=True)


def score(fw, bk, mode="pp"):
    trk.map.p = dataclasses.replace(p0, **fw)
    trk.back_la_min, trk.back_la_t = bk
    trk.back_mode = mode
    res = [T.run(nm, mk, du, op, quiet=True) for nm, mk, du, op in T.SCENARIOS]
    obj = sum((r["rmse"] if r["rmse"] == r["rmse"] else 1.0) for r in res) / len(res) + 0.05 * sum(r["n_wall"] for r in res) / T.B
    return obj, res


best = None
for fw in grid_f:                                     # forward lookahead first, at the current backward one
    obj, res = score(fw, (0.6, 0.5))
    line = " | ".join(f"{r['rmse'] * 100:.1f}" for r in res)
    print(f"전진 {fw} -> 평균 {obj * 100:.1f} cm (A/B/C/D {line}, 벽 {sum(r['n_wall'] for r in res)})", flush=True)
    if best is None or obj < best[0]:
        best = (obj, fw)
fw = best[1]
bbest = None
for bk in grid_b:
    obj, res = score(fw, bk)
    print(f"후진 {bk} (전진 {fw}) -> 평균 {obj * 100:.1f} cm (C 후진 {res[2]['rmse_back'] * 100:.1f}, D 후진 {res[3]['rmse_back'] * 100:.1f})", flush=True)
    if bbest is None or obj < bbest[0]:
        bbest = (obj, bk)
obj0, res0 = score(dict(t_clip_min=p0.t_clip_min, m_l1=p0.m_l1, q_l1=p0.q_l1), (0.6, 0.5))
objb, resb = score(fw, bbest[1])
objm, resm = score(fw, bbest[1], "map")
for lab, rr, ob in (("지금 설정", res0, obj0), ("맞춘 설정 (후진 pure pursuit)", resb, objb), ("맞춘 설정 (후진도 MAP)", resm, objm)):
    print(f"== {lab}: 평균 {ob * 100:.1f} cm | " + " | ".join(
        f"{nm[:1]} RMSE {r['rmse'] * 100:.1f} cm (후진 {r['rmse_back'] * 100:.1f} / 전진 {r['rmse_fwd'] * 100:.1f}, 벽 {r['n_wall']})"
        for (nm, *_), r in zip(T.SCENARIOS, rr)), flush=True)
print(f"BEST 전진 {fw} 후진 {bbest[1]}", flush=True)
