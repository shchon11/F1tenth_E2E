"""혼자·빈 트랙 세션에서 실행 방식별로 시간이 어디서 가는지: 닿지 않은 바퀴의 lap, plan이 낸 속도 대비 실제 속도, 옆가속도.
plan의 속도는 기록에 따로 없어서 plan 점 사이 거리 / 0.1 s로 구함.
인자: 기록npz ...   (EXEC_GAP_FIG=png 를 주면 그림도: 속도 변화 요청 대비 실제, plan 속도 대비 실제의 분포, 한 세션의 속도 기록)"""
import os
import sys
import numpy as np
from rec_lib import Rec, plt
FIG = os.environ.get("EXEC_GAP_FIG", "")
COL = ("#555555", "#1f77b4", "#e6820e", "#2ca02c", "#9467bd")
keep = []
for fn in sys.argv[1:]:
    R = Rec(fn); z = R.z; T, B, dt = R.T, R.B, R.dt
    st = R.st.astype(float); vx = st[:, :, 3]; yaw = st[:, :, 2]
    hard = (R.d[:, :, 3:6] > 0).any(-1); touch = (R.d[:, :, 0:3] > 0).any(-1) if R.d.shape[-1] >= 6 else hard
    solo = ~R.on[:T].any(0); nobox = (z["sid"][0] < 0).all(-1) if z["sid"].ndim == 3 else np.ones(B, bool)
    m = solo & nobox
    lapsig = z["lap"]; laps = []; clean = []
    for b in np.nonzero(m)[0]:
        net = np.cumsum(lapsig[:, b]); hi = 0; t_prev = None; prev_ok = False
        for t in np.nonzero(lapsig[:, b] > 0)[0]:
            if net[t] > hi:
                hi = net[t]
                if t_prev is not None:
                    ok = not hard[t_prev:t + 1, b].any()
                    laps.append((t + 1 - t_prev) * dt)
                    if ok and prev_ok:
                        clean.append((t + 1 - t_prev) * dt)
                    prev_ok = ok
                else:
                    prev_ok = not hard[:t + 1, b].any()
                t_prev = t + 1
    laps = np.array(laps); clean = np.array(clean)
    r = np.angle(np.exp(1j * (yaw[1:] - yaw[:-1]))) / dt
    ay = np.abs(vx[:T] * r)
    rec = np.zeros((T, B), bool)
    for k in range(31):
        rec[k:] |= hard[: T - k]
    ok = ~rec & (vx[:T] > 2.0); ok[:30] = False; ok &= m[None]
    pl = R.pl.astype(float)                                   # (T, B, 10, 2)
    p0 = st[:T, :, None, :2]
    seg = np.linalg.norm(np.diff(np.concatenate([p0, pl], 2), axis=2), axis=-1) / 0.1     # planned speed per plan step
    out = [f"{z['exam']} · {z['mode']} · {R.name} — 혼자·빈 트랙 세션 {int(m.sum())}개"]
    out.append(f"  바퀴 전체 {len(laps)}개: 중앙 {np.median(laps):.2f} s | 세게 닿지 않은 바퀴(앞 바퀴도 안 닿음) {len(clean)}개 = {100 * len(clean) / max(len(laps), 1):.0f} %: 중앙 {np.median(clean):.2f} s, 10~90 % {np.percentile(clean, 10):.2f}~{np.percentile(clean, 90):.2f} s")
    out.append(f"  속도(닿은 뒤 3 s 빼고): 중앙 {np.median(vx[:T][ok]):.2f} m/s, 평균 {vx[:T][ok].mean():.2f} | 옆가속도: 중앙 {np.median(ay[ok]):.1f}, 90 % {np.percentile(ay[ok], 90):.1f}, 99 % {np.percentile(ay[ok], 99):.1f} m/s²")
    for i in (1, 3, 5):
        if T - i <= 0:
            continue
        dv = vx[i:T + 0][:, :] - seg[: T - i, :, i - 1] if False else (vx[i:T] - seg[: T - i, :, i - 1])
        okk = ok[: T - i] & ok[i:T]
        out.append(f"  plan이 {0.1 * i:.1f} s 뒤에 낸 속도 대비 실제 속도: 중앙 {np.median(dv[okk]):+.2f} m/s, 10~90 % {np.percentile(dv[okk], 10):+.2f}~{np.percentile(dv[okk], 90):+.2f}")
    # where the plan asks to speed up / slow down: achieved change over 0.3 s vs asked
    ask = seg[:, :, 2] - vx[:T]; got = np.zeros_like(ask); got[: T - 3] = vx[3:T] - vx[: T - 3]
    okk = ok.copy(); okk[T - 3:] = False
    for name, sel in (("가속 요청(0.3 s에 +0.5 m/s 넘게)", ask > 0.5), ("감속 요청(-0.5 m/s 넘게)", ask < -0.5)):
        s2 = okk & sel
        if s2.sum() > 50:
            out.append(f"  {name} {100 * s2.sum() / okk.sum():.0f} %: 요청 중앙 {np.median(ask[s2]):+.2f}, 실제 {np.median(got[s2]):+.2f} m/s")
    print("\n".join(out))
    if FIG:
        dv3 = (vx[3:T] - seg[: T - 3, :, 2])[ok[: T - 3] & ok[3:T]]
        b0 = next((b for b in np.nonzero(m)[0] if not hard[200:360, b].any() and vx[200:360, b].min() > 2.0), int(np.nonzero(m)[0][0]))
        lab = dict(ideal="plan대로 정확히(학습 조건)", map="MAP", track="시간 추종기")[str(z["mode"])]
        ex = str(z["exam"])
        lab += (" + 미끄럼 각(참값)" if "slip_oracle" in ex else "") + (" + 미끄럼 각(표)" if "slip_table" in ex else "") + (f" + 속도 앞당김 {ex.split('lead')[1]} s" if "lead" in ex else "")
        keep.append(dict(lab=lab, ask=ask[okk], got=got[okk], dv3=dv3, t=np.arange(200, 360) * dt, v=vx[200:360, b0], v1=seg[200:360, b0, 0], v3=seg[197:357, b0, 2], b=b0))
if FIG and keep:
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    ax = axs[0]; edges = np.arange(-2.0, 2.01, 0.25); mid = 0.5 * (edges[1:] + edges[:-1])
    ax.plot([-2, 2], [-2, 2], ":", color="k", lw=1.0, label="요청한 만큼 그대로")
    for k, c in zip(keep, COL):
        i = np.digitize(k["ask"], edges) - 1
        y = np.array([np.median(k["got"][i == j]) if (i == j).sum() >= 50 else np.nan for j in range(len(mid))])
        ax.plot(mid, y, "-o", color=c, lw=2, ms=4, label=k["lab"])
    ax.axhline(0, color="#bbbbbb", lw=0.8); ax.axvline(0, color="#bbbbbb", lw=0.8); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("plan이 0.3 s 뒤에 낸 속도 - 지금 속도 (m/s)"); ax.set_ylabel("0.3 s 뒤 실제로 바뀐 속도, 중앙값 (m/s)")
    ax.set_title("속도를 올리라는 요청은 MAP에서 절반도 안 나옴, 줄이라는 요청은 나옴", fontsize=11)
    ax = axs[1]; e2 = np.arange(-2.0, 1.01, 0.1)
    for k, c in zip(keep, COL):
        h, _ = np.histogram(k["dv3"], e2); ax.step(0.5 * (e2[1:] + e2[:-1]), 100 * h / max(h.sum(), 1), where="mid", color=c, lw=2,
                                                   label=f"{k['lab']}: 중앙 {np.median(k['dv3']):+.2f} m/s")
    ax.axvline(0, color="k", ls=":", lw=1.0); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("0.3 s 뒤 실제 속도 - plan이 그 시점에 낸 속도 (m/s)"); ax.set_ylabel("결정의 비율 (%)")
    ax.set_title("실제 속도가 plan보다 낮은 쪽으로 치우침", fontsize=11)
    ax = axs[2]; k = keep[min(1, len(keep) - 1)]
    ax.plot(k["t"], k["v3"], "-", color="#e377c2", lw=1.5, label="0.3 s 전의 plan이 이 시점에 낸 속도")
    ax.plot(k["t"], k["v1"], "-", color="#999999", lw=1.2, label="plan 첫 0.1 s 구간의 속도 (MAP이 속도 명령으로 쓰는 값)")
    ax.plot(k["t"], k["v"], "-", color="#1f77b4", lw=2.2, label="실제 속도")
    ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="lower right"); ax.set_xlabel("세션 시간 (s)"); ax.set_ylabel("속도 (m/s)")
    ax.set_title(f"{k['lab']} — 혼자·빈 트랙 세션 {k['b']}의 16 s", fontsize=11)
    fig.suptitle("혼자·빈 트랙 세션(같은 policy plan25r u300, 닿은 뒤 3 s는 뺌): plan이 낸 속도와 실제 속도", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG, dpi=70); print("그림", FIG)
