"""느린 상대 차를 따라잡은 만남 하나하나가 어떻게 끝났나 (11-14 밤, 사용자: edge case를 보고 학습 환경을 다시 짜라).
시험 기록(exam_record.py npz)에서 상대가 트랙을 따라 앞 5 m 안으로 들어온 때부터 우리가 1.5 m 넘게 앞설 때(앞지름) / 7 m 넘게
벌어질 때 / 20 s / 다시 출발까지를 만남 하나로 봄. 만남마다: 차와 세게·가볍게 닿은 수, 벽·상자에 세게 닿은 수, 걸린 시간에서
'빈 트랙을 혼자 달릴 때 같은 구간에 걸리는 시간'(혼자·빈 트랙 세션의 자리별 속도 중앙값)을 뺀 잃은 시간, 학습 보상의 접촉 값
(0.5 s + 닿는 속도 1 m/s당 1 s), 닿기 전에 줄였나(그 자리 혼자 속도의 80 % 아래로 0.3 s 넘게).
인자: 기록npz [...]   환경변수 ENC_ROWS=파일.npy 로 만남별 행 저장"""
import os, sys
import numpy as np
from rec_lib import Rec

T_PRICE0, T_PRICE1 = 0.5, 1.0


def wrap(g, L):
    return (g + 0.5 * L) % L - 0.5 * L


for fn in sys.argv[1:]:
    R = Rec(fn); z = R.z; T, B, dt = R.T, R.B, R.dt
    L = float(z["lapL"][0]); rem = z["rem"].astype(float); orem = z["orem"].astype(float)
    on = R.on[:T]; st = R.st.astype(float); op = R.op.astype(float)
    v = np.abs(st[:T, :, 3]); vo = np.abs(op[:T, :, 3])
    box = (R.sid[0] >= 0).any(-1); opp = on.any(0)
    d = R.d; sev = R.held[:, :, 3].astype(float)
    # solo speed by place, from the empty solo sessions
    nb = int(np.ceil(L / 0.25)); solo = np.where(~box & ~opp)[0]
    k = np.clip((rem[:T, solo] / 0.25).astype(int), 0, nb - 1)
    vs = np.full(nb, np.nan)
    t0 = int(round(10.0 / dt))                                   # after the start from rest
    for i in range(nb):
        m = k[t0:] == i
        if m.sum() >= 3:
            vs[i] = np.median(v[t0:, solo][m])
    idx = np.arange(nb); ok = ~np.isnan(vs); vs = np.interp(idx, idx[ok], vs[ok], period=nb)
    vsolo = vs[np.clip((rem[:T] / 0.25).astype(int), 0, nb - 1)]            # (T, B)
    g = wrap(rem[:T] - orem[:T], L)                                # + = the other car is ahead along the track
    prog = wrap(rem[:-1] - rem[1:], L)[:T]                         # [m] our progress in each decision
    rs = R.rs | z["ors"][:T]
    rows = []
    for b in np.where(opp)[0]:
        t = 1
        while t < T:
            if not (on[t, b] and 0.0 < g[t, b] <= 5.0 and (g[t - 1, b] > 5.0) and v[t, b] > vo[t, b] + 0.5):
                t += 1; continue
            a = t; end = "시간"
            while t < T - 1 and t - a < int(20 / dt):
                t += 1
                if rs[t, b] or not on[t, b]:
                    end = "다시 출발"; break
                if g[t, b] < -1.5:
                    end = "앞지름"; break
                if g[t, b] > 7.0:
                    end = "벌어짐"; break
            seg = slice(a, t + 1)
            dur = (t + 1 - a) * dt
            exp = float(np.sum(np.clip(prog[seg, b], 0, None) / np.maximum(vsolo[seg, b], 0.5)))
            dc = d[seg, b]                                          # light wall, box, car; hard wall, box, car
            price = float(np.sum((T_PRICE0 + T_PRICE1 * sev[seg, b]) * (dc.sum(1) > 0)))
            tc = np.nonzero(dc[:, 5] > 0)[0]
            first = int(tc[0]) if len(tc) else len(dc)
            pre = slice(a, a + first + 1)
            slow = (v[pre, b] < 0.8 * vsolo[pre, b]).sum() * dt >= 0.3
            wd = R.wall_dist(int(R.tid[a + min(first, len(dc) - 1), b]), op[a + min(first, len(dc) - 1), b, None, :2])[0]
            rows.append((b, a, t, ["앞지름", "벌어짐", "시간", "다시 출발"].index(end), dc[:, 5].sum(), dc[:, 2].sum(), dc[:, 3].sum() + dc[:, 4].sum(),
                         dur, dur - exp, price, vo[a, b], v[a, b], float(slow), float(box[b]), wd, int(R.kind[a, b]), float(R.scale[a, b]),
                         (0.0 < g[seg, b]).sum() * dt))
            t += 1
    r = np.array(rows, float)
    if os.environ.get("ENC_ROWS"):
        np.save(os.environ["ENC_ROWS"], r)
    print(f"== {fn.split('/')[-1]}: 느린 상대를 따라잡은 만남 {len(r)}건 (상대 있는 세션 {int(opp.sum())}개, 세션당 {len(r) / max(opp.sum(), 1):.1f})")

    def line(name, m):
        if m.sum() == 0:
            print(f"  {name}: 0건"); return
        q = r[m]
        print(f"  {name}: {int(m.sum())}건 ({100 * m.mean():.0f} %) | 앞지름으로 끝남 {100 * (q[:, 3] == 0).mean():.0f} % | 잃은 시간 평균 {q[:, 8].mean():+.2f} s (중앙 {np.median(q[:, 8]):+.2f}) | "
              f"접촉 값 평균 {q[:, 9].mean():.2f} s | 합 {q[:, 8].mean() + q[:, 9].mean():+.2f} s | 상대 뒤에 있던 시간(중앙) {np.median(q[:, 17]):.1f} s | 닿기 전/지나기 전 줄인 만남 {100 * q[:, 12].mean():.0f} % | "
              f"상대 {q[:, 10].mean():.1f} · 우리 {q[:, 11].mean():.1f} m/s")
    hardc, lightc, other = r[:, 4] > 0, (r[:, 4] == 0) & (r[:, 5] > 0), r[:, 6] > 0
    clean = ~hardc & ~lightc & ~other
    line("차와 세게 닿음", hardc); line("차와 가볍게만 닿음", lightc); line("차는 안 닿고 벽·상자에 세게", ~hardc & ~lightc & other); line("아무것도 안 닿음", clean)
    print("  -- 안 닿은 만남을 줄였나로 나눔");  line("    줄이고 지나감", clean & (r[:, 12] > 0)); line("    안 줄이고 지나감", clean & (r[:, 12] == 0))
    print("  -- 상대 속도(만남 시작 때)로 나눔: 세게 닿은 비율")
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 9)):
        m = (r[:, 10] >= lo) & (r[:, 10] < hi)
        if m.sum():
            print(f"    상대 {lo}-{hi} m/s: {int(m.sum())}건, 차와 세게 {100 * hardc[m].mean():.0f} %, 잃은 시간+접촉 값 평균 {(r[m, 8] + r[m, 9]).mean():+.2f} s (안 닿은 만남만 {r[m & clean, 8].mean() if (m & clean).sum() else float('nan'):+.2f} s)")
    print("  -- 상자 있는 세션 / 없는 세션: 세게 닿은 비율 " + " / ".join(f"{100 * hardc[r[:, 13] == x].mean():.0f} % ({int((r[:, 13] == x).sum())}건)" for x in (1.0, 0.0)))
    kn = R.kinds
    print("  -- 상대 종류: " + " · ".join(f"{kn[int(kk)]} {int((r[:, 15] == kk).sum())}건 세게 {100 * hardc[r[:, 15] == kk].mean():.0f} %" for kk in np.unique(r[:, 15])))
