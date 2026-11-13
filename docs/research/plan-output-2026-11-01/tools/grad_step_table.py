"""Paired exam table for stepped checkpoints (grad_step.py): each row against the unmoved checkpoint, session by session (11-13).

Same seed -> the same worlds, cases and box layouts, so the difference per session is taken and its mean +- standard error
over sessions is the row. Columns: distance driven on the empty track alone [m / 90 s], distance over all sessions, hard
contacts per session (wall / box / car), sessions that stood over 2 s.

    python3 grad_step_table.py BASE.npz LABEL=EXAM.npz [...]
"""
import sys
import numpy as np
import exam_table as ET


def load(fn):
    z = np.load(fn, allow_pickle=False)
    dt = float(z["dt"]); cnt = z["cnt"].astype(float); T, B = cnt.shape[:2]
    st = z["st"].astype(float); on = z["on"][:T]; sid = z["sid"]
    L = float(z["lapL"][int(z["tid"][0, 0])])
    box = (sid[0] >= 0).any(-1); opp = on.any(0)
    d = np.clip(np.diff(np.concatenate([np.zeros((1, B, 6)), cnt], 0), axis=0), 0, None)
    hard = d[:, :, 3:6].sum(0)
    v = np.abs(st[:T, :, 3]); slow = v < 0.3; slow[:int(round(1.0 / dt))] = False
    stalls, _ = ET.runs(slow, int(round(ET.STALL_S / dt)))
    P, _ = ET.progress(z["rem"], L)
    return dict(dist=P[-1], hard=hard, stall=(stalls > 0).astype(float), box=box, opp=opp)


def ms(x):
    return f"{x.mean():+.2f} ±{x.std(ddof=1) / np.sqrt(len(x)):.2f}"


b = load(sys.argv[1]); es = ~b["box"] & ~b["opp"]
print(f"기준 {sys.argv[1].split('/')[-1]}: 세션 {len(es)}개 (혼자·빈 트랙 {int(es.sum())}개) | 빈 트랙 혼자 달린 거리 {b['dist'][es].mean():.1f} m · 모든 세션 {b['dist'].mean():.1f} m | "
      f"세션당 세게 닿음 벽 {b['hard'][:, 0].mean():.2f} · 상자 {b['hard'][:, 1].mean():.2f} · 차 {b['hard'][:, 2].mean():.2f} · 합 {b['hard'].sum(1).mean():.2f} | 2 s 넘게 선 세션 {b['stall'].mean() * 100:.0f} %")
print("기준과의 차 (세션끼리 짝지어, 평균 ± 표준오차):")
print(f"  {'':<34} {'빈 트랙 혼자 거리 [m]':>18} {'모든 세션 거리 [m]':>18} {'세게 닿음 합':>14} {'벽':>12} {'상자':>12} {'차':>12} {'선 세션':>12}")
for a in sys.argv[2:]:
    lab, fn = a.split("=")
    try:
        s = load(fn)
    except Exception as e:
        print(f"  {lab:<34} (없음: {type(e).__name__})"); continue
    assert (s["box"] == b["box"]).all() and (s["opp"] == b["opp"]).all()
    dd = s["dist"] - b["dist"]; dh = s["hard"] - b["hard"]
    print(f"  {lab:<34} {ms(dd[es]):>18} {ms(dd):>18} {ms(dh.sum(1)):>14} {ms(dh[:, 0]):>12} {ms(dh[:, 1]):>12} {ms(dh[:, 2]):>12} {ms(s['stall'] - b['stall']):>12}")
