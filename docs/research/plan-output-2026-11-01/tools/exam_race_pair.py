"""exam4 짝 비교 (11-25): 같은 시드의 exam4 기록 둘 이상을 세션끼리 짝지어 비교 -- 첫 기록이 기준.
exam_race_table.py를 기록마다 그대로 돌려 세션별 값(앞지름, 우리 책임 차 접촉, 세게 닿은 벽·상자·차, 앞차 plan 겹침)을 꺼내고,
같은 세션 번호(같은 조건·같은 상자 배치·같은 출발)끼리 뺀 차이를 세션 다시 뽑기(짝을 같이 뽑음)로 95 % 구간.
인자: 기준.npz 비교1.npz [비교2.npz ...]"""
import contextlib
import io
import os
import runpy
import sys

import numpy as np

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exam_race_table.py")
RNG = np.random.default_rng(1)


def load(fn):
    argv = sys.argv
    sys.argv = [TOOL, fn]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g = runpy.run_path(TOOL, run_name="__main__")
    finally:
        sys.argv = argv
    return {k: np.asarray(g[k]) for k in ("made", "fault", "hard", "pc_n", "pc_hit", "c_pace", "c_start", "c_box", "laps")}


def name(fn):
    b = os.path.basename(fn)
    for p in ("exam4_",):
        b = b[len(p):] if b.startswith(p) else b
    return b.split("_ideal")[0]


def mean_ci(a, b, m):
    d = (b - a)[m].astype(float)
    idx = RNG.integers(0, len(d), (4000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return b[m].mean(), d.mean(), lo, hi


def share_ci(na, ha, nb, hb, m):
    na, ha, nb, hb = (x[m].astype(float) for x in (na, ha, nb, hb))
    sa, sb = ha.sum() / max(na.sum(), 1.0), hb.sum() / max(nb.sum(), 1.0)
    idx = RNG.integers(0, len(na), (4000, len(na)))
    d = hb[idx].sum(1) / np.maximum(nb[idx].sum(1), 1.0) - ha[idx].sum(1) / np.maximum(na[idx].sum(1), 1.0)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return 100 * sb, 100 * (sb - sa), 100 * lo, 100 * hi


fns = sys.argv[1:]
R = [load(f) for f in fns]
A = R[0]
for r in R[1:]:
    assert np.array_equal(r["c_pace"], A["c_pace"]) and np.array_equal(r["c_start"], A["c_start"]) and np.array_equal(r["c_box"], A["c_box"]), \
        "세션 조건이 다름: 같은 시드의 exam4 기록끼리만"
allm = np.ones(len(A["made"]), bool)
p06, p08 = np.isclose(A["c_pace"], 0.6), np.isclose(A["c_pace"], 0.8)
rows = [("앞지름 (전체)", lambda r: r["made"], allm), ("우리 책임 차 접촉 (전체)", lambda r: r["fault"], allm),
        ("세게 닿음 차 (전체)", lambda r: r["hard"][:, 2], allm), ("세게 닿음 벽 (전체)", lambda r: r["hard"][:, 0], allm),
        ("세게 닿음 상자 (전체)", lambda r: r["hard"][:, 1], allm), ("우리 책임 차 접촉 (0.6배)", lambda r: r["fault"], p06),
        ("앞지름 (0.6배)", lambda r: r["made"], p06), ("앞지름 (0.8배)", lambda r: r["made"], p08),
        ("바퀴 (전체)", lambda r: r["laps"], allm)]
print(f"== exam4 짝 비교: 기준 {name(fns[0])} (세션 {len(A['made'])}개, 같은 시드라 세션끼리 짝), 차이 = 비교 - 기준 [세션 다시 뽑기 95 %] ==")
for fn, r in zip(fns[1:], R[1:]):
    print(f"-- {name(fn)}")
    for lab, f, m in rows:
        v, d, lo, hi = mean_ci(f(A).astype(float), f(r).astype(float), m)
        print(f"  {lab:<22} {v:6.2f}  (기준 {f(A)[m].mean():5.2f}, 차이 {d:+.2f} [{lo:+.2f} {hi:+.2f}])")
    for lab, m in (("앞차 plan 겹침 0.6배", p06), ("앞차 plan 겹침 0.8배", p08), ("앞차 plan 겹침 전체", allm)):
        v, d, lo, hi = share_ci(A["pc_n"], A["pc_hit"], r["pc_n"], r["pc_hit"], m)
        va = 100 * A["pc_hit"][m].sum() / max(A["pc_n"][m].sum(), 1)
        print(f"  {lab:<22} {v:5.1f} %  (기준 {va:4.1f} %, 차이 {d:+.1f} [{lo:+.1f} {hi:+.1f}] %p)")
