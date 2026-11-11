"""기억이 벌 수 있는 것이 학습 세계에 있나 (11-11: 기억을 지워도 주행이 그대로 -- 기억이 쓸모 있을 자리가 있는지부터 잼).
기록된 주행(drive_record.py의 npz)에서 상자 옆을 지나간 건마다: 지나가기 전에 그 상자가 몇 초 동안 벽에 가리지 않고 보였나
(차에서 상자 중심까지 직선이 벽을 안 지남 -- 다른 상자·상대 차에 가린 것은 안 봄), 처음 보였을 때 거리, 그 차가 그 상자를 몇 번째
지나가나(두 번째부터가 기억이 미리 알 수 있는 건), 결과(상자에 닿음 / 세게 닿음, 앞 5 m ~ 뒤 2 m를 달린 시간, 가장 느렸던 속도).
늦게 보인 건이 더 많이 닿거나 더 느리면 기억이 벌 것이 있고, 차이가 없으면 벌 것이 없음.
인자: npz [지나감으로 치는 거리 m, 기본 1.2]"""
import sys
import numpy as np

D = np.load(sys.argv[1])
NEAR = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
st, sid, pose, cnt, rs, tid, dt = D["st"], D["sid"], D["pose"], D["cnt"], D["rs"], D["tid"], float(D["dt"])
T1, B, P = sid.shape
T = T1 - 1
_edt, _off, _H, _W, _res, _org = D["edt"], D["t_off"], D["t_H"].astype(int), D["t_W"].astype(int), D["t_res"], D["t_origin"]
_occ = {}


def wall_at(k, pts):
    if k not in _occ:
        _occ[k] = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k]) <= 0.0
    c = np.floor((pts[..., 0] - _org[k][0]) / _res[k]).astype(int); r = np.floor((pts[..., 1] - _org[k][1]) / _res[k]).astype(int)
    ok = (c >= 0) & (c < _W[k]) & (r >= 0) & (r < _H[k])
    out = np.ones(pts.shape[:-1], bool)
    out[ok] = _occ[k][r[ok], c[ok]]
    return out


F = np.linspace(0.0, 1.0, 240)[None, :, None]
dprop = np.diff(np.concatenate([np.zeros((1, B, 6), np.float32), cnt], 0), axis=0).clip(min=0.0)       # (T, B, 6) onsets in decision t
rows = []          # per pass: lead s, first-sight distance, pass number, touched, hard, time over 5 m before..2 m after, min speed, speed 1 s before, censored
for b in range(B):
    if not (sid[:, b] >= 0).any():
        continue
    k = int(tid[0, b])
    xy = st[:, b, :2].astype(float); v = np.abs(st[:, b, 3].astype(float))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    live = sid[:, b] >= 0                                                         # (T1, P)
    bx = pose[:, b, :, :2].astype(float)
    dist = np.where(live, np.linalg.norm(bx - xy[:, None], axis=2), 1e9)          # (T1, P)
    near_j = dist.argmin(1)
    start = np.concatenate([[0], np.nonzero(rs[:, b])[0] + 1])                    # decisions where an episode starts
    for j in range(P):
        if not live[:, j].any():
            continue
        d = dist[:, j]
        # line of sight to the box centre, stopping 0.45 m short of it
        seg = xy[:, None, :] + F[0] * (bx[:, j] - xy)[:, None, :]
        fr = np.clip(1.0 - 0.45 / np.maximum(d, 0.5), 0.0, 1.0)
        blk = (wall_at(k, seg) & (F[0, :, 0][None] <= fr[:, None])).any(1)
        vis = live[:, j] & ~blk
        n_pass = 0
        t = 1
        while t < T:
            lo, hi = max(t - 10, 0), min(t + 10, T)
            if d[t] < NEAR and d[t] <= d[lo: hi + 1].min():
                # sight run ending at the pass
                u = t
                while u > 0 and vis[u - 1]:
                    u -= 1
                ep0 = int(start[start <= t].max())
                censored = u <= ep0
                lead = (t - u) * dt
                w0, w1 = max(t - 8, 0), min(t + 5, T - 1)
                mine = near_j[w0: w1 + 1] == j
                touch = bool((dprop[w0: w1 + 1, b, 1] * mine).sum() > 0.5); hard = bool((dprop[w0: w1 + 1, b, 4] * mine).sum() > 0.5)
                i0 = int(np.searchsorted(s, s[t] - 5.0)); i1 = int(np.searchsorted(s, s[t] + 2.0))
                span = (i1 - i0) * dt if (i0 > ep0 and i1 < T1 and s[t] - s[ep0] > 5.0) else np.nan
                rows.append((lead, d[u], n_pass, touch, hard, span, v[max(t - 10, 0): t + 6].min(), v[max(t - 10, 0)], censored, d[t], b, j, t, u))
                n_pass += 1
                t = hi + 1
            else:
                t += 1
R = np.array(rows, float)
print(f"{D['name']}: 학습 차 {B}대 x {T * dt:.0f} s, 상자 {NEAR:g} m 안으로 지나간 건 {len(R)} (닿음 {int(R[:, 3].sum())}, 세게 {int(R[:, 4].sum())})")
print(f"기록 전체의 상자 접촉 {int(dprop[:, :, 1].sum())}, 세게 {int(dprop[:, :, 4].sum())} (지나간 건에 못 붙인 것은 빠짐)")


def table(M, title):
    print(f"\n[{title}] 건수 {len(M)}")
    print("  지나가기 전 보인 시간 | 건수(비율) | 처음 보인 거리 중앙 | 닿음 % | 세게 닿음 % (건) | 앞 5 m~뒤 2 m 시간 중앙 | 가장 느린 속도 중앙 | 1 s 전 속도 중앙")
    for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 99.0)):
        m = M[(M[:, 0] >= lo) & (M[:, 0] < hi)]
        if len(m) == 0:
            continue
        lab = f"{lo:g}~{hi:g} s" if hi < 99 else f"{lo:g} s 넘게"
        print(f"  {lab:>10} | {len(m):5d} ({len(m) / len(M) * 100:4.1f} %) | {np.median(m[:, 1]):4.1f} m | {m[:, 3].mean() * 100:4.1f} | "
              f"{m[:, 4].mean() * 100:4.2f} ({int(m[:, 4].sum())}) | {np.nanmedian(m[:, 5]):.2f} s | {np.median(m[:, 6]):.2f} m/s | {np.median(m[:, 7]):.2f} m/s")


U = R[R[:, 8] < 0.5]            # the sight run does not reach back to the episode start
table(U, "모든 지나감 (출발 때부터 보이던 건 빼고)")
table(U[U[:, 2] < 0.5], "그 상자를 처음 지나감 = 기억이 알 수 없음")
table(U[U[:, 2] >= 0.5], "두 번째부터 = 기억이 미리 알 수 있음")
late = U[(U[:, 2] >= 0.5) & (U[:, 0] < 1.0)]; early = U[(U[:, 2] >= 0.5) & (U[:, 0] >= 1.5)]
if len(late) and len(early):
    print(f"\n두 번째부터 지나간 건에서, 1 s 안 되게 보인 건 {len(late)} vs 1.5 s 넘게 보인 건 {len(early)}: "
          f"세게 닿음 {late[:, 4].mean() * 100:.2f} % vs {early[:, 4].mean() * 100:.2f} %, 닿음 {late[:, 3].mean() * 100:.1f} % vs {early[:, 3].mean() * 100:.1f} %, "
          f"시간 {np.nanmedian(late[:, 5]):.2f} s vs {np.nanmedian(early[:, 5]):.2f} s")
np.save(sys.argv[1].replace(".npz", "_sight.npy"), R)
