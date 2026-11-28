"""입력 조작 시험(probe_box.py) 분석: 상자가 정책 입력에 들어온 뒤 계획이 언제, 얼마나, 어느 쪽으로 바뀌나.
만남마다(events.py) 가장 가까운 순간(또는 세게 닿은 결정)까지 결정마다:
  nb_real = 시뮬레이터 스캔이 실제로 상자에 맞은 빔 수, dlat10 = 상자 있음 - 상자 지움의 1.0 s 점 옆 차이 [m] (차 틀, + 왼쪽),
  dsteer = 첫 0.1 s 끝 조향각 차이 [rad]. 상자의 차 틀 옆 위치 y_box (+ 왼쪽) -> '비켜 가는 쪽' = sign(dlat10) == -sign(y_box).
  보인 시각 = nb_real >= 2가 닿을 때까지 이어진 구간의 시작(1결정 끊김 허용). 반응 시각 = 보인 뒤 처음 |dlat10| > 5 cm.
인자: events_probe32.json probe32_probe.npz out_prefix"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec, ray_free, Caster

E = json.load(open(sys.argv[1]))["probe32"]
P = np.load(sys.argv[2])
R = Rec(E["file"]); C = Caster(R); st = R.st.astype(float)
nb, dl, ds, dv = P["nb_real"], P["dlat10"], P["dsteer"], P["dv1"]
hard_at = {}
for h in E["hard"]:
    hard_at.setdefault((h["b"], h["j"]), []).append(h["t"])
by = {}
for e in E["enc"]:
    by.setdefault((e["b"], e["j"]), []).append(e)
rows = []
for (b, j), L in by.items():
    L = sorted(L, key=lambda e: e["t0"])
    Pb = R.boxes(0, b)[j]; c = Pb.mean(0)
    # wall gap either side of the box, across the car's heading at the first closest approach
    yaw = st[int(round(L[0]["tmin"])), b, 2]
    n_ = np.array([-np.sin(yaw), np.cos(yaw)])
    loc = ((Pb - c) * n_).sum(1)
    gap = min(ray_free(C, Pb[np.argmax(loc)] + 0.01 * n_, n_), ray_free(C, Pb[np.argmin(loc)] - 0.01 * n_, -n_))
    gap_max = max(ray_free(C, Pb[np.argmax(loc)] + 0.01 * n_, n_), ray_free(C, Pb[np.argmin(loc)] - 0.01 * n_, -n_))
    prev_end = 0
    for k, e in enumerate(L):
        hits = [t for t in hard_at.get((b, j), []) if e["t0"] - 1 <= t <= e["t1"] + 1]
        te = hits[0] if hits else int(round(e["tmin"]))
        ta = max(prev_end, te - 40); prev_end = e["t1"]
        tt = np.arange(ta, te + 1)
        if len(tt) < 3 or te >= nb.shape[0]:
            continue
        vis = nb[tt, b] >= 2
        i = len(vis) - 1; first = None; gp = 0
        while i >= 0:
            if vis[i]:
                first = i; gp = 0
            else:
                gp += 1
                if gp > 1:
                    break
            i -= 1
        if first is None:
            continue
        # box lateral place in the car frame per decision
        yb = -(c[0] - st[tt, b, 0]) * np.sin(st[tt, b, 2]) + (c[1] - st[tt, b, 1]) * np.cos(st[tt, b, 2])
        dist = np.linalg.norm(c[None] - st[tt, b, :2], axis=1)
        x = dl[tt, b]
        resp = np.nonzero(np.abs(x[first:]) > 0.05)[0]
        on = first + resp[0] if len(resp) else None
        away = np.sign(x) == -np.sign(yb)
        w = slice(first, len(tt))
        rows.append(dict(b=int(b), j=int(j), enc=k, hit=bool(hits), any=e["any"] > 0, clear=e["clear"], gap=float(gap), gap_max=float(gap_max),
                         vis_s=float((len(tt) - 1 - first) * 0.1), d_vis=float(dist[first]), v_vis=float(abs(st[tt[first], b, 3])),
                         lat_s=float((on - first) * 0.1) if on is not None else np.nan, d_on=float(dist[on]) if on is not None else np.nan,
                         max_dl=float(np.abs(x[w]).max()), max_ds=float(np.abs(ds[tt[w], b]).max()),
                         away_frac=float(np.mean(away[w][np.abs(x[w]) > 0.02])) if (np.abs(x[w]) > 0.02).any() else np.nan,
                         yb_vis=float(yb[first]), dl_series=[round(float(v), 3) for v in x[first:]], nb_series=[int(v) for v in nb[tt[first:], b]],
                         yb_series=[round(float(v), 3) for v in yb[first:]], dist_series=[round(float(v), 2) for v in dist[first:]],
                         dv_series=[round(float(v), 3) for v in dv[tt[first:], b]]))
json.dump(rows, open(sys.argv[3] + "_rows.json", "w"))
A = np.array


def summ(nm, S):
    if not S:
        print(f"  {nm}: 0"); return
    lat = A([s["lat_s"] for s in S]); md = A([s["max_dl"] for s in S]); aw = A([s["away_frac"] for s in S])
    print(f"  {nm:<34} n {len(S):4d} | 보인 시각 {np.median([s['vis_s'] for s in S]):.1f} s 전, {np.median([s['d_vis'] for s in S]):.1f} m | "
          f"반응(>5 cm) 없음 {100 * np.mean(~np.isfinite(lat)):3.0f} % | 반응까지 중앙 {np.nanmedian(lat):.1f} s (p75 {np.nanpercentile(lat, 75) if np.isfinite(lat).any() else np.nan:.1f}) "
          f"| 최대 계획 변화 중앙 {100 * np.median(md):.0f} cm | 비켜 가는 쪽 비율 중앙 {100 * np.nanmedian(aw):.0f} %")


print("== 입력 조작 시험 (scan31b u475, 이상 실행, 세션 32개 x 60 s, 상자 계속 있음)")
summ("안 닿은 통과 전부", [s for s in rows if not s["any"] and not s["hit"]])
summ("세게 닿은 만남", [s for s in rows if s["hit"]])
summ("안 닿은 통과 · 첫 만남", [s for s in rows if not s["any"] and s["enc"] == 0])
summ("안 닿은 통과 · 상자 양옆 중 좁은 쪽 < 0.5 m", [s for s in rows if not s["any"] and s["gap"] < 0.5])
summ("안 닿은 통과 · 좁은 쪽 >= 0.5 m", [s for s in rows if not s["any"] and s["gap"] >= 0.5])
summ("안 닿은 통과 · 3 s 넘게 일찍 보임(>6 m)", [s for s in rows if not s["any"] and s["d_vis"] > 6])
summ("안 닿은 통과 · 늦게 보임(<4 m)", [s for s in rows if not s["any"] and s["d_vis"] < 4])
# response vs distance: decisions pooled
print("  거리별 계획 반응 (모든 통과의 결정, 상자 빔 >= 2): |dlat10| 중앙 / 5 cm 넘는 비율")
D = []
for s in rows:
    for x, n, d in zip(s["dl_series"], s["nb_series"], s["dist_series"]):
        if n >= 2:
            D.append((d, abs(x), n, s["hit"]))
D = A(D)
for lo, hi in ((8, 10), (6, 8), (4, 6), (3, 4), (2, 3), (1, 2), (0, 1)):
    m = (D[:, 0] >= lo) & (D[:, 0] < hi)
    if m.any():
        print(f"     {lo}-{hi} m: 결정 {m.sum():5d} · 빔 중앙 {np.median(D[m, 2]):4.0f} · |dlat10| 중앙 {100 * np.median(D[m, 1]):5.1f} cm · >5 cm {100 * np.mean(D[m, 1] > 0.05):4.0f} %"
              f" | 세게 닿은 접근만: |dlat10| 중앙 {100 * np.median(D[m & (D[:, 3] > 0), 1]) if (m & (D[:, 3] > 0)).any() else float('nan'):5.1f} cm")
print("  세게 닿은 만남 하나하나:")
for s in rows:
    if s["hit"]:
        print(f"    세션 {s['b']:2d} 상자 {s['j']} {s['enc'] + 1}번째 · 보임 {s['vis_s']:.1f} s 전 {s['d_vis']:.1f} m ({s['v_vis']:.1f} m/s) · 좁은 쪽 빈 폭 {s['gap']:.2f} m"
              f" · 반응까지 {s['lat_s']:.1f} s · 최대 변화 {100 * s['max_dl']:.0f} cm · 비켜 가는 쪽 {s['away_frac']:.2f}")
        print(f"      dlat10 cm: {[round(100 * v) for v in s['dl_series']]}")
        print(f"      상자 옆 위치 cm: {[round(100 * v) for v in s['yb_series']]} · 빔: {s['nb_series']}")
