"""(11-16) 앞에 상대 차가 있을 때 plan이 얼마나 비켜 가나 -- 상대 차가 나보다 빠르거나 같을 때도 장애물처럼 돌아가나
(사용자: "상대차도 스캔에 보이면 어차피 이놈 앞으로 갈거라 내갈길 가도 별 상관없는데 꼭 장애물 있는것처럼 planner가 빙빙 돌아가고
스스로 위험에 자초하는 듯한 움직임이 보여").

frozen exam 기록(exam_record.py, 학습 세계와 같은 상대 차) 하나로:
- 기준 길 = 같은 policy가 빈 트랙 혼자일 때 달린 길(진행 0.2 m 칸마다 평균 위치). 레이싱 라인이 아니라 이 policy 자신의 길.
- '앞에 차' = 상대 차가 내 차 기준 앞 0.6-3.0 m, 옆 0.5 m 안, 같은 방향(45도 안).
- 비켜 감 = plan 10점이 기준 길에서 가장 멀리 벗어난 거리. 대조: 같은 진행 칸에서 혼자·빈 트랙일 때의 같은 값.
- 상대 차 빠르기 칸 = 그 차의 지금 속도 / 같은 자리에서 이 policy가 혼자 달리는 속도 (내 차가 이미 뒤에서 속도를 맞춘 경우를
  '같은 속도'로 세지 않으려고 -- 상대 - 나 속도는 따라가며 속도를 줄이면 0이 됨): >= 0.95(나만큼 빠름), 0.8-0.95, < 0.8(느림).
- 그 뒤 2 s 안에 세게 닿음(벽 · 상자 · 차).
인자: 기록npz [...]
"""
import sys

import numpy as np

AHEAD, SIDE, BIN = (0.6, 3.0), 0.5, 0.2
NEXT = 20                                        # decisions (2 s) after, for the contact outcome

for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True)
    st, op, on, pl = z["st"].astype(float), z["op"].astype(float), z["on"], z["pl"].astype(float)
    rs, cnt, rem, sid = z["rs"], z["cnt"].astype(float), z["rem"].astype(float), z["sid"]
    T, B = rs.shape
    d = np.clip(np.diff(cnt, axis=0, prepend=cnt[:1]), 0, None) * (~rs)[..., None]   # per-decision onsets/hard
    hard = d[..., 3:6]
    L = float(z["lapL"][0])
    pos = np.mod(L - rem[:T], L)
    solo = ~on.any(0) & ~(sid[0] >= 0).any(-1)
    # the policy's own line: mean position per progress bin over its empty-track solo sessions
    nb = int(np.ceil(L / BIN))
    k = np.floor(pos / BIN).astype(int) % nb
    sel = solo[None, :] & ~rs & np.isfinite(pos)
    sx, sy, sn, sv = np.zeros(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb)
    np.add.at(sx, k[sel], st[:T, :, 0][sel]); np.add.at(sy, k[sel], st[:T, :, 1][sel]); np.add.at(sn, k[sel], 1)
    np.add.at(sv, k[sel], st[:T, :, 3][sel])
    v_solo = np.where(sn > 0, sv / np.maximum(sn, 1), np.nan)
    opos = np.mod(L - z["orem"][:T].astype(float), L)
    ok_ = np.isfinite(opos)
    ratio = np.full(opos.shape, np.nan)
    ratio[ok_] = op[:T, :, 3][ok_] / v_solo[np.floor(opos[ok_] / BIN).astype(int) % nb]
    ok = sn > 0
    line = np.stack([sx[ok] / sn[ok], sy[ok] / sn[ok]], 1)
    if len(line) == 0:
        # (11-17) a record without solo empty-track sessions (oppexam_run.sh: an opponent in every session): the own line and
        # its speeds from the same checkpoint's frozen exam (OWN_LINE=that npz, or the exam1 file named after this one)
        import os, re as _re
        ref = os.environ.get("OWN_LINE") or os.path.join(os.path.dirname(os.path.abspath(fn)), "..", "exam",
                                                          _re.sub(r"^oppexam_(.*)_u(\d+)\.npz$", r"exam1_\1_u\2_ideal_uniform_n256.npz",
                                                                  os.path.basename(fn)))
        zr = np.load(ref, allow_pickle=True)
        st_r, rs_r, on_r, sid_r = zr["st"].astype(float), zr["rs"], zr["on"], zr["sid"]
        Tr = rs_r.shape[0]
        pos_r = np.mod(L - zr["rem"][:Tr].astype(float), L)
        k_r = np.floor(np.nan_to_num(pos_r) / BIN).astype(int) % nb
        sel_r = (~on_r.any(0) & ~(sid_r[0] >= 0).any(-1))[None, :] & ~rs_r & np.isfinite(pos_r)
        sx, sy, sn = np.zeros(nb), np.zeros(nb), np.zeros(nb)
        np.add.at(sx, k_r[sel_r], st_r[:Tr, :, 0][sel_r]); np.add.at(sy, k_r[sel_r], st_r[:Tr, :, 1][sel_r]); np.add.at(sn, k_r[sel_r], 1)
        if "sv" in dir():
            sv = np.zeros(nb); np.add.at(sv, k_r[sel_r], st_r[:Tr, :, 3][sel_r])
        line = np.stack([sx[sn > 0] / sn[sn > 0], sy[sn > 0] / sn[sn > 0]], 1)
        v_solo = np.where(sn > 0, sv / np.maximum(sn, 1), np.nan)          # the speeds behind `ratio` from the same sessions
        ratio[ok_] = op[:T, :, 3][ok_] / v_solo[np.floor(opos[ok_] / BIN).astype(int) % nb]
        print(f"  (혼자 세션 없음: 내 길은 {os.path.basename(ref)}에서)")
    tang = np.roll(line, -1, 0) - np.roll(line, 1, 0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9

    def lat(p):
        """signed distance of points p (..., 2) from the own line (nearest line point)"""
        q = p.reshape(-1, 2)
        out = np.empty(len(q))
        for i in range(0, len(q), 4096):
            dd = ((q[i:i + 4096, None, :] - line[None]) ** 2).sum(-1)
            j = dd.argmin(1)
            r = q[i:i + 4096] - line[j]
            out[i:i + 4096] = tang[j, 0] * r[:, 1] - tang[j, 1] * r[:, 0]
        return out.reshape(p.shape[:-1])

    # opponent in the ego frame
    yaw = st[:T, :, 2]
    rx, ry = op[:T, :, 0] - st[:T, :, 0], op[:T, :, 1] - st[:T, :, 1]
    fx = np.cos(yaw) * rx + np.sin(yaw) * ry
    fy = -np.sin(yaw) * rx + np.cos(yaw) * ry
    dyaw = np.abs(np.angle(np.exp(1j * (op[:T, :, 2] - yaw))))
    dv = op[:T, :, 3] - st[:T, :, 3]
    ahead = on[:T] & ~rs & (fx > AHEAD[0]) & (fx < AHEAD[1]) & (np.abs(fy) < SIDE) & (dyaw < np.pi / 4) & (st[:T, :, 3] > 1.0)
    ahead[T - NEXT:] = False
    tt, bb = np.nonzero(ahead)
    off_plan = np.abs(lat(pl[tt, bb])).max(-1)
    # control: the same progress bins, solo empty track (where the car is when no car is ahead)
    cs = sel.copy(); cs[T - NEXT:] = False
    if not cs.any():                                   # no solo sessions in this record: no control column
        cs = ahead.copy()
    ct, cb = np.nonzero(cs)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ct), min(len(ct), 40000), replace=False)
    ct, cb = ct[pick], cb[pick]
    off_ctl = np.abs(lat(pl[ct, cb])).max(-1)
    ck = k[ct, cb]
    ctl_by_bin = np.full(nb, np.nan)
    for j in np.unique(ck):
        ctl_by_bin[j] = np.median(off_ctl[ck == j])
    base = ctl_by_bin[k[tt, bb]]
    nxt = np.stack([hard[t + 1:t + 1 + NEXT, b].sum(0) for t, b in zip(tt, bb)]) if len(tt) else np.zeros((0, 3))
    nxt_ctl = np.stack([hard[t + 1:t + 1 + NEXT, b].sum(0) for t, b in zip(ct, cb)])
    print(f"== {fn.split('/')[-1]} | 상대 차 평균 속도 {op[:T, :, 3][on[:T]].mean():.2f} m/s, 내 차 {st[:T, :, 3].mean():.2f} m/s | "
          f"앞에 차 결정 {len(tt)}개 (상대 차 있는 세션의 결정 중 {100 * len(tt) / max(1, (on[:T] & ~rs).sum()):.1f} %)")
    print(f"  {'상대 / 내 혼자 속도':<16}{'결정 수':>8}{'비율':>7}{'plan이 내 길에서 벗어남':>22}{'같은 자리 혼자':>14}"
          f"{'2 s 안 세게 닿음 (벽/상자/차)':>32}")
    rr = ratio[tt, bb]
    for name, m in (("나만큼 빠름 >=0.95", rr >= 0.95), ("조금 느림 0.8-0.95", (rr < 0.95) & (rr >= 0.8)),
                    ("느림 < 0.8", rr < 0.8)):
        if not m.any():
            print(f"  {name:<16}{0:>8}")
            continue
        h = nxt[m]
        big = off_plan[m] > 0.5
        hb, hs = (h[big].sum(1) > 0).mean() if big.any() else np.nan, (h[~big].sum(1) > 0).mean() if (~big).any() else np.nan
        print(f"    (plan이 0.5 m 넘게 벗어난 {100 * big.mean():.0f} %: 2 s 안 세게 닿음 {100 * hb:.1f} % · 나머지 {100 * hs:.1f} %)")
        print(f"  {name:<16}{m.sum():>8}{100 * m.mean():>6.0f}%{np.median(off_plan[m]):>17.2f} m (상위 10 % {np.percentile(off_plan[m], 90):.2f})"
              f"{np.nanmedian(base[m]):>10.2f} m{100 * (h.sum(1) > 0).mean():>12.1f} % ({100 * (h[:, 0] > 0).mean():.1f}/"
              f"{100 * (h[:, 1] > 0).mean():.1f}/{100 * (h[:, 2] > 0).mean():.1f})")
    print(f"  {'혼자·빈 트랙(대조)':<16}{len(ct):>8}{'':>7}{np.median(off_ctl):>17.2f} m (상위 10 % {np.percentile(off_ctl, 90):.2f})"
          f"{'':>12}{100 * (nxt_ctl.sum(1) > 0).mean():>12.1f} %")
