"""(11-24) MAP 다시 맞추기의 표: 같은 exam3 세션(같은 시드)을 plan대로 정확히 / MAP 설정별로 달린 기록들을 한 표로.
열: 따라가기 오차(0.2 / 0.5 s 뒤 점 대 점 · 옆), 요 각속도 흔들림(혼자·빈 트랙), 미끄럼각 90 %, 혼자·빈 트랙 한 바퀴(세션 중앙값들의 중앙 [95 %]),
혼자·빈 트랙 세션당 세게 닿은 벽, 상자 세션당 세게 닿은 벽 / 상자 (평균 [95 %]); 아래에 기준 행(둘째 인자)과의 세션 짝 차이.
인자: 이름=기록 ... (첫째 = plan대로 정확히, 둘째 = 차이의 기준)"""
import sys
import numpy as np
from maptune_metrics import score

RNG = np.random.default_rng(0)


def ci(v, stat=np.mean, n=4000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    idx = RNG.integers(0, len(v), (n, len(v)))
    s = v[idx].mean(1) if stat is np.mean else np.array([stat(v[i]) for i in idx])
    return np.percentile(s, [2.5, 97.5])


def f_mean(v):
    lo, hi = ci(v)
    return f"{np.mean(v):.2f} [{lo:.2f}-{hi:.2f}]"


def f_med(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    lo, hi = ci(v, np.median, 2000)
    return f"{np.median(v):.2f} [{lo:.2f}-{hi:.2f}]"


args = [a.split("=", 1) for a in sys.argv[1:]]
W = score(args[1][1])["cell_n"] if len(args) > 1 else None     # the objective's cell weights: the baseline's (map_tune_scan.py)
rows = [(tag, score(fn, W)) for tag, fn in args]
print(f"| {'':24s} | 0.2 s 점/옆 [cm] | 0.5 s 점/옆 [cm] | 흔들림 [rad/s] | 미끄럼 90 % | 빈 트랙 한 바퀴 [s] | 빈 트랙 벽/세션 | 상자 세션 벽/세션 | 상자 세션 상자/세션 |")
print("|" + "---|" * 10)
for tag, d in rows:
    print(f"| {tag:24s} | {d['pos2'] * 100:.1f} / {d['lat2'] * 100:.1f} | {d['pos5'] * 100:.1f} / {d['lat5'] * 100:.1f} | {d['wobble_solo']:.2f} | "
          f"{d['slip_p90']:.1f}° | {f_med(d['lap_solo_per'])} | {f_mean(d['hw_solo_per'])} | {f_mean(d['hw_box_per'])} | {f_mean(d['hp_box_per'])} |")
if len(rows) > 2:
    tb, b = rows[1]
    print(f"\n기준 '{tb}'과의 세션 짝 차이 (평균 [95 %]):")
    for tag, d in rows[2:]:
        out = []
        for k, nm in (("lap_solo_per", "빈 트랙 한 바퀴 s"), ("hw_solo_per", "빈 트랙 벽"), ("hw_box_per", "상자 세션 벽"), ("hp_box_per", "상자 세션 상자")):
            x = np.asarray(d[k], float) - np.asarray(b[k], float); x = x[np.isfinite(x)]
            lo, hi = ci(x)
            out.append(f"{nm} {x.mean():+.2f} [{lo:+.2f} {hi:+.2f}]")
        print(f"  {tag}: " + " · ".join(out))
    te, e = rows[0]
    print(f"\n'{te}'과 '{tb}' 사이 간격 중 메운 몫:")
    for tag, d in rows[2:]:
        fr = []
        for k, nm in (("obj", "따라가기 오차(목표)"), ("pos5", "0.5 s 점"), ("wobble_solo", "흔들림"), ("slip_p90", "미끄럼 90 %"), ("lap_solo", "빈 트랙 한 바퀴"),
                      ("hard_wall", "벽"), ("hard_prop_box", "상자")):
            gap = b[k] - e[k]
            fr.append(f"{nm} {100 * (b[k] - d[k]) / gap:.0f} %" if abs(gap) > 1e-9 else f"{nm} -")
        print(f"  {tag}: " + " · ".join(fr))
