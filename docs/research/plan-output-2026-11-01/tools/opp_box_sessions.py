"""(11-16) 상대 차 + 상자 세션에서 상대 차를 배울 거리가 있나 (사용자: "상대차랑 장애물 세션에서 장애물이 지나치게 많아서 뭔가 상대를
충분히 배우기 힘들 것 같아. 상대도 주행 제대로 못할 것 같구").

frozen exam 기록 하나로, 상자를 치우기 전(rm_t 전)만, 세션 종류별(상대 차 + 상자 / 상대 차만 / 상자만 / 혼자 빈 트랙):
- 내 차: 평균 속도, 앞 0.6-3 m에 상대 차가 있는 결정 /분, 추월 /분(트랙을 따라 상대 차 뒤 3 m 안 -> 앞으로), 세게 닿음 /분(벽·상자·차)
- 상대 차: 달릴 때 평균 속도, 다시 출발(2 s 동안 0.5 m 못 감) /분, 세게 닿음 /분(벽·상자)
- 상자: 세션당 수, 내 길(같은 policy가 혼자·빈 트랙에서 달린 길) 위에 놓인 비율(가장자리까지 0.25 m 안 = 차 반폭 + 여유),
  상대 차 길(상자 없는 세션에서 상대 차가 지나간 점들) 위에 놓인 비율
인자: 기록npz [...]
"""
import sys

import numpy as np

AHEAD, SIDE, BIN, ON_PATH = (0.6, 3.0), 0.5, 0.2, 0.25

for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True)
    st, op, on, rs, sid, pose = z["st"].astype(float), z["op"].astype(float), z["on"], z["rs"], z["sid"], z["pose"].astype(float)
    cnt, ocnt, ors = z["cnt"].astype(float), z["ocnt"].astype(float), z["ors"]
    T, B = rs.shape
    dt = float(z["dt"]); rm = int(round(float(z["rm_t"]) / dt)) if "rm_t" in z.files else T
    W = slice(0, min(T, rm))                                   # before the boxes go
    tw = (W.stop - W.start) * dt / 60.0
    L = float(z["lapL"][0])
    d = np.clip(np.diff(cnt, axis=0, prepend=cnt[:1]), 0, None) * (~rs)[..., None]
    od = np.clip(np.diff(ocnt, axis=0, prepend=ocnt[:1]), 0, None)
    has = (sid[0] >= 0).any(-1)
    opp = on.any(0)
    # own line (solo, empty track)
    pos = np.mod(L - z["rem"][:T].astype(float), L)
    nb = int(np.ceil(L / BIN)); k = np.floor(np.nan_to_num(pos) / BIN).astype(int) % nb
    sel = (~opp & ~has)[None, :] & ~rs & np.isfinite(pos)
    sx, sy, sn = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    np.add.at(sx, k[sel], st[:T, :, 0][sel]); np.add.at(sy, k[sel], st[:T, :, 1][sel]); np.add.at(sn, k[sel], 1)
    line = np.stack([sx[sn > 0] / sn[sn > 0], sy[sn > 0] / sn[sn > 0]], 1)
    # opponents' paths: every point an opponent passed in races without boxes
    m_o = (opp & ~has)[None, :] & on[:T] & ~ors
    opts = op[:T, :, :2][m_o][::3]
    # boxes at the start: edge distance ~ centre distance - half the larger footprint side
    half = {}
    for kk in z.files:
        if kk.startswith("fp_"):
            fp = z[kk]; half[int(kk[3:])] = 0.5 * float(max(np.ptp(fp[:, 0]), np.ptp(fp[:, 1])))

    def on_frac(pts, b_list):
        n_on = 0
        for (x, y, s) in b_list:
            dd = np.sqrt(((pts - [x, y]) ** 2).sum(1)).min() - half.get(int(s), 0.15)
            n_on += dd < ON_PATH
        return n_on / max(1, len(b_list))
    boxes = [(pose[0, b, j, 0], pose[0, b, j, 1], sid[0, b, j]) for b in range(B) for j in range(sid.shape[2]) if sid[0, b, j] >= 0]
    print(f"== {fn.split('/')[-1]} | 상자 치우기 전 {W.stop * dt:.0f} s만 | 상자 {len(boxes)}개 중 내 길 위 {100 * on_frac(line, boxes):.0f} %, "
          f"상대 차 길 위 {100 * on_frac(opts, boxes):.0f} % (가장자리까지 {ON_PATH} m 안)")
    # passes: along the track, the opponent within 3 m ahead -> within 3 m behind
    opos = np.mod(L - z["orem"][:T].astype(float), L)
    rel = np.mod(pos - opos + 0.5 * L, L) - 0.5 * L                    # > 0: I am ahead
    yaw = st[:T, :, 2]
    rx, ry = op[:T, :, 0] - st[:T, :, 0], op[:T, :, 1] - st[:T, :, 1]
    fx = np.cos(yaw) * rx + np.sin(yaw) * ry
    fy = -np.sin(yaw) * rx + np.cos(yaw) * ry
    ahead = on[:T] & ~rs & (fx > AHEAD[0]) & (fx < AHEAD[1]) & (np.abs(fy) < SIDE)
    print(f"  {'세션':<12}{'수':>4}{'상자/세션':>9}{'내 속도':>8}{'앞에 차/분':>10}{'추월/분':>8}{'세게/분 벽·상자·차':>20}"
          f"{'상대 속도':>9}{'상대 다시 출발/분':>14}{'상대 세게/분 벽·상자':>18}")
    for nm, m in (("상대 차+상자", opp & has), ("상대 차만", opp & ~has), ("상자만", ~opp & has), ("혼자 빈 트랙", ~opp & ~has)):
        if not m.any():
            continue
        n = m.sum()
        v = st[W, m, 3].mean()
        h = d[W][:, m, 3:6].sum((0, 1)) / (n * tw)
        line_ = f"  {nm:<12}{n:>4}{(sid[0, m] >= 0).sum(-1).mean():>9.1f}{v:>8.2f}"
        if (m & opp).any():
            a = ahead[W][:, m].sum() / (n * tw)
            r = rel[W][:, m]
            ok = np.isfinite(r)
            r = np.where(ok, r, 0)
            passes = ((r[:-1] < 0) & (r[:-1] > -3) & (r[1:] > 0) & (r[1:] < 3) & ok[:-1] & ok[1:] & ~ors[W][:-1, m] & ~ors[W][1:, m]).sum() / (n * tw)
            run = on[W][:, m] & ~ors[W][:, m]
            ov = op[W][:, m, 3][run].mean()
            orst = ors[W][:, m].sum() / (n * tw)
            oh = od[W][:, m, 3:5].sum((0, 1)) / (n * tw)
            line_ += f"{a:>10.1f}{passes:>8.2f}{h[0]:>9.2f}·{h[1]:.2f}·{h[2]:.2f}{ov:>9.2f}{orst:>14.2f}{oh[0]:>12.2f}·{oh[1]:.2f}"
        else:
            line_ += f"{'':>10}{'':>8}{h[0]:>9.2f}·{h[1]:.2f}·{h[2]:.2f}"
        print(line_)
