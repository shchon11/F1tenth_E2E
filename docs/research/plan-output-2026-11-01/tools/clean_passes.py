"""(11-27, the user: "위험을 감수할 필요는 없지. 분명히 추월할 수 있을 것 같은데 안하는게 문제고 조심하는건 죄가 아니지") Are clean
passes possible here at all? Every pass in an exam4 record (exam_race_table.py's count: the along-track gap changes sign by
HYST m), and whether it was CLEAN: no car contact of any kind (cnt onsets_car / hard_car) and no at-fault charge from 3 s before
the pass completed to 1 s after it, and no wall / box hard contact in that window. Per opponent pace. Also where passes happen:
the track position (fraction of the lap) of each clean pass, so a policy that stops passing can be checked against the places
another policy passed cleanly.  usage: clean_passes.py EXAM4.npz [...]"""
import sys

import numpy as np

HYST = 1.0
W0, W1 = 3.0, 1.0


def progress(rem, L, skip=None):
    r = rem.astype(float).copy()
    for t in range(1, r.shape[0]):
        bad = ~np.isfinite(r[t]); r[t, bad] = r[t - 1, bad]
    r[~np.isfinite(r)] = 0.0
    d = r[:-1] - r[1:]
    d = np.mod(d + 0.5 * L, L) - 0.5 * L
    if skip is not None:
        d[skip] = 0.0
    return np.concatenate([np.zeros((1, r.shape[1])), np.cumsum(d, 0)], 0), r


for fn in sys.argv[1:]:
    z = np.load(fn)
    dt = float(z["dt"]); cnt = z["cnt"].astype(float); T, B = cnt.shape[:2]
    L = float(z["lapL"][int(z["tid"][0, 0])])
    c_start, c_pace = z["c_start"][:B].astype(int), z["c_pace"][:B].astype(float)
    d = np.clip(np.diff(np.concatenate([np.zeros((1, B, 6)), cnt], 0), axis=0), 0, None)   # onsets per step
    bad = d[:, :, 2] + d[:, :, 5] + d[:, :, 3] + d[:, :, 4] + (z["cfault"][:T] > 0)           # car onset/hard, wall/box hard, fault
    P, _ = progress(z["rem"], L); ors = z["ors"][:T]; Po, _ = progress(z["orem"], L, skip=ors)
    g0 = np.mod(z["orem"][0].astype(float) - z["rem"][0].astype(float) + 0.5 * L, L) - 0.5 * L
    g0 = np.where(c_start == 0, -0.5 * L, np.nan_to_num(g0))
    Gp = g0[None, :] + P - Po
    cell = np.floor(Gp[0] / L)
    ev = []
    for t in range(1, T + 1):
        re_ = ors[t - 1]
        cell = np.where(re_, np.floor(Gp[t] / L), cell)
        up = Gp[t] >= (cell + 1) * L + HYST; dn = Gp[t] <= cell * L - HYST
        for b in np.nonzero(up & ~re_)[0]:
            a0, a1 = max(0, t - int(W0 / dt)), min(T, t + int(W1 / dt))
            clean = not bad[a0:a1, b].any()
            ev.append((b, t, clean, float(np.mod(z["rem"][t, b], L)) / L))
        cell = cell + up.astype(int) - dn.astype(int)
    print(f"== {fn.split('/')[-1]}")
    for lo, hi, nm in ((0, 0.7, "0.6배"), (0.7, 0.9, "0.8배"), (0.9, 2, "0.95배·제 페이스")):
        ss = (c_pace >= lo) & (c_pace < hi); n = ss.sum()
        e = [x for x in ev if lo <= c_pace[x[0]] < hi]
        c = sum(x[2] for x in e)
        print(f"  상대 {nm:<10} 세션 {n:3d} | 앞지름 {len(e) / n:.2f}/세션 | 그중 깔끔(앞 3 s~뒤 1 s 접촉 없음) {c / n:.2f}/세션 "
              f"({100 * c / max(1, len(e)):.0f} %) | 깔끔한 앞지름이 한 번이라도 있는 세션 {100 * np.mean([any(x[2] for x in e if x[0] == b) for b in np.nonzero(ss)[0]]):.0f} %")
    cl = np.array([x[3] for x in ev if x[2] and 0.7 <= c_pace[x[0]] < 0.9])
    if cl.size:
        h, _ = np.histogram(cl, bins=10, range=(0, 1))
        print("  0.8배 깔끔한 앞지름이 끝난 자리 (남은 거리/한 바퀴, 10칸):", " ".join(str(int(v)) for v in h))
