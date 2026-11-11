"""상대 차 종류별로 얼마나 제대로 달리나 (11-12 사용자: "잘 포팅된게 단 한개도 없어. 다 좆같이 달려" -> 종류마다 따로 봄).
drive_record.py 기록(npz)에서 상대 차가 있는 경주만: 종류별 대수, 달린 시간, 평균 속도(속도 배율로 나눈 값도), 서 있는 시간 비율
(0.3 m/s 아래), 2 s 동안 0.5 m도 못 간 시간 비율, 새로 닿은 횟수/분(벽·상자·우리 차, 전체 / 세게). 상자가 있는 동안 / 없는 동안으로
나누고, 속도 배율 구간(0.4-0.7 / 0.7-1.0 / 1.0-1.2)으로도 나눔. 맨 아래에 같은 경주의 우리 차(학습 차)를 같은 방식으로.
인자: 기록 npz [기록 npz ...]"""
import sys
import numpy as np

for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=False)
    dt = float(z["dt"]); T = z["cnt"].shape[0]
    kinds = [str(k) for k in z["kinds"]]
    on = z["on"][:T]; kind = z["kind"][:T]; scale = z["scale"][:T]
    box = (z["sid"][:T] >= 0).any(-1)
    n2 = int(round(2.0 / dt))
    print(f"== {z['name']}  ({fn.split('/')[-1]}, {z['cnt'].shape[1]}대 x {T * dt:.0f} s) ==")

    def rows(st, cnt, mask, label):
        """st (T+1, B, 4), cnt (T, B, 6) cumulative, mask (T, B): the decisions that count"""
        v = np.abs(st[:T, :, 3].astype(float)); xy = st[:, :, :2].astype(float)
        d = np.diff(np.concatenate([np.zeros((1,) + cnt.shape[1:], np.float32), cnt], 0), axis=0)
        d = np.clip(d, 0, None)                                        # a restart sets the counters back to 0
        step = np.linalg.norm(xy[1:T + 1] - xy[:T], axis=2)
        d[step > 2.0] = 0.0                                            # a re-placed car: not a contact
        net = np.full((T, st.shape[1]), np.inf)
        net[:T - n2] = np.linalg.norm(xy[n2:T] - xy[:T - n2], axis=2)
        jump = np.zeros((T, st.shape[1]), bool)                        # a re-placement inside the 2 s window: not a stall
        cs = np.concatenate([np.zeros((1, st.shape[1])), np.cumsum(step > 2.0, 0)], 0)
        jump[:T - n2] = (cs[n2:T] - cs[:T - n2]) > 0
        m = mask
        if m.sum() == 0:
            return
        mins = m.sum() * dt / 60.0
        ok2 = m & np.isfinite(net) & ~jump
        c = d[m].sum(0) / mins
        print(f"  {label:<26} {mins:6.1f}분 | 속도 {v[m].mean():4.2f} m/s | 섬 {100 * (v[m] < 0.3).mean():4.1f} % | "
              f"2 s에 0.5 m 못 감 {100 * (net[ok2] < 0.5).mean():4.1f} % | 닿음/분 벽 {c[0]:4.2f} 상자 {c[1]:4.2f} 차 {c[2]:4.2f} | "
              f"세게 벽 {c[3]:4.2f} 상자 {c[4]:4.2f} 차 {c[5]:4.2f}")

    for name, sel in (("상자 있는 동안", box), ("상자 없는 동안", ~box)):
        print(f" [{name}]")
        for k, kn in enumerate(kinds):
            m = on & (kind == k) & sel
            if m.sum() == 0:
                continue
            n_car = int((on & (kind == k)).any(0).sum())
            rows(z["op"], z["ocnt"], m, f"상대 {kn} ({n_car}대)")
            for lo, hi in ((0.0, 0.7), (0.7, 1.0), (1.0, 9.9)):
                mm = m & (scale >= lo) & (scale < hi)
                if mm.sum() * dt >= 60.0:
                    rows(z["op"], z["ocnt"], mm, f"    배율 {max(lo, 0.4):.1f}-{min(hi, 1.2):.1f}")
        rows(z["op"], z["ocnt"], on & sel, "상대 전체")
        rows(z["st"], z["cnt"], on & sel, "우리 차 (상대 있는 경주)")
        rows(z["st"], z["cnt"], ~on & sel, "우리 차 (혼자)")
