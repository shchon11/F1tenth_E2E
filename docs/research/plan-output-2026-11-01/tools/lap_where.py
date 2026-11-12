"""빈 트랙 한 바퀴의 시간 차가 트랙 어디서 나는가 (lap_where.py, 11-12).
시험 기록 두 개(같은 시드)의 혼자·빈 트랙 세션에서, 트랙을 따라 잰 자리(보상이 쓰는 벽만의 남은 거리; 분석에서만 씀)를 0.5 m 칸으로 나눠
칸마다 한 바퀴에 머문 시간·달린 길이·속도·벽까지 거리를 냄. 두 policy가 달린 평균 자리를 지도에 겹치고, 속도와 달린 길이·시간 차의 누적을
자리 순서로 그림. 남은 거리 값은 출발선 바로 뒤 약 1.5 m가 비어 있어(그 구간이 좁게 눌림) 첫 2.5 m는 한 칸으로 묶음.
인자: 출력머리말 기준기록npz 비교기록npz"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt, along_track

DS, T_SKIP, SEAM = 0.5, 100, 2.5          # [m] bin; decisions skipped at the session start; [m] merged first stretch
C_A, C_B, MUTED, INK = "#2a78d6", "#eb6834", "#898781", "#0b0b0b"


def profile(fn):
    r = Rec(fn); z = r.z
    T, B = r.T, r.B
    s, lap, L = along_track(z, T)
    box = (r.sid[0] >= 0).any(1); opp = r.on[:T].any(0)
    m = ~box & ~opp
    nb = int(np.ceil(L / DS)); n0 = int(round(SEAM / DS))
    st = r.st[:T].astype(float)
    k = int(r.tid[0, 0])
    step = np.hypot(np.diff(r.st[:, :, 0].astype(float), axis=0), np.diff(r.st[:, :, 1].astype(float), axis=0))   # [m] driven per decision
    yawr = (np.mod(np.diff(r.st[:, :, 2].astype(float), axis=0) + np.pi, 2 * np.pi) - np.pi) / r.dt
    alat = np.abs(st[..., 3] * yawr)                          # [m/s^2] speed x yaw rate
    wall = np.stack([r.wall_dist(k, st[:, b, :2]) for b in range(B)], 1)
    ok = np.zeros((T, B), bool); ok[T_SKIP:, m] = True
    ok &= ~np.concatenate([np.zeros((1, B), bool), r.rs[:-1].astype(bool)], 0).cumsum(0).astype(bool)   # up to the first restart only
    hit = (r.d[..., 3:].sum(2) > 0)
    for b in np.nonzero(m)[0]:                                # drop the 3 s after any hard contact (not the question here)
        for t in np.nonzero(hit[:, b])[0]:
            ok[t:t + 30, b] = False
    ib = np.maximum(np.minimum((s / DS).astype(int), nb - 1), n0 - 1) - (n0 - 1)       # the first SEAM metres = bin 0
    nb = nb - (n0 - 1)
    prog = np.mod(np.diff(np.concatenate([s, s[-1:]], 0), axis=0) + 0.5 * L, L) - 0.5 * L        # [m] progress per decision
    laps = prog[ok].sum() / L
    out = dict(n=np.bincount(ib[ok], minlength=nb).astype(float))
    for key, v in (("x", st[..., 0]), ("y", st[..., 1]), ("wall", wall), ("alat", alat)):
        out[key] = np.bincount(ib[ok], weights=v[ok], minlength=nb) / np.maximum(out["n"], 1)
    out["t"] = out["n"] * r.dt / laps                         # [s] spent in the bin per lap
    out["p"] = np.bincount(ib[ok], weights=step[ok], minlength=nb) / laps      # [m] driven in the bin per lap
    out["v"] = out["p"] / out["t"]
    out["x"][0] = out["y"][0] = np.nan                        # the merged stretch has no single place
    lo = np.r_[0.0, np.arange(n0, n0 + nb - 1) * DS]; hi = np.r_[lo[1:], L]
    return r, k, L, out, lo, hi, dict(sessions=int(m.sum()), laps=laps, path=out["p"].sum(), lap=out["t"].sum(), vmax=out["v"].max(),
                                      alat90=np.percentile(alat[ok], 90), alat99=np.percentile(alat[ok], 99),
                                      wall10=np.percentile(wall[ok], 10))


out_prefix, fa, fb = sys.argv[1:4]
ra, k, L, A, lo, hi, sa = profile(fa)
rb, _, _, Bp, _, _, sb = profile(fb)
na, nbn = ra.name.split("/")[0], rb.name.split("/")[0]
x = 0.5 * (lo + hi)
dT, dP = Bp["t"] - A["t"], Bp["p"] - A["p"]
print(f"[빈 트랙 한 바퀴, 어디서 차이가 나는가] 기준 {ra.name} · 비교 {rb.name} (혼자·빈 트랙 세션, 첫 {T_SKIP * ra.dt:.0f} s와 세게 닿은 뒤 3 s 제외)")
print(f"{'':<10}{'세션':>5}{'바퀴 수':>8}{'한 바퀴 [s]':>12}{'달린 길이 [m]':>14}{'평균 속도':>10}{'가장 빠른 칸 [m/s]':>18}{'횡가속도 90 % / 99 %':>22}{'벽까지 거리 하위 10 % [m]':>24}")
for nm, s_ in ((na, sa), (nbn, sb)):
    print(f"{nm:<10}{s_['sessions']:>5}{s_['laps']:>9.0f}{s_['lap']:>12.2f}{s_['path']:>14.2f}{s_['path'] / s_['lap']:>11.2f}{s_['vmax']:>16.2f}"
          f"{s_['alat90']:>16.1f} / {s_['alat99']:.1f}{s_['wall10']:>18.2f}")
va = sa["path"] / sa["lap"]
print(f"한 바퀴 차이 {sb['lap'] - sa['lap']:+.2f} s · 달린 길이 차이 {sb['path'] - sa['path']:+.2f} m (기준 평균 속도 {va:.2f} m/s로 {(sb['path'] - sa['path']) / va:+.2f} s) · "
      f"평균 속도 차이 {sb['path'] / sb['lap'] - va:+.2f} m/s")
# sections: the track cut where the reference speed crosses its median (fast = straights, slow = corners)
fast = A["v"] >= np.median(A["v"])
edges = np.nonzero(np.diff(fast.astype(int)) != 0)[0] + 1
print("구간별 (기준 속도가 빠름/느림으로 바뀌는 자리에서 자름):")
print(f"{'자리 [m]':<14}{'종류':<5}{'시간 기준 → 비교 [s]':>22}{'차이':>7}{'달린 길이 기준 → 비교 [m]':>26}{'차이':>7}{'속도 기준 → 비교 [m/s]':>24}{'벽까지 기준 → 비교 [m]':>24}")
for a, b in zip(np.r_[0, edges], np.r_[edges, len(fast)]):
    w = slice(a, b)
    print(f"{lo[a]:5.1f}-{hi[b - 1]:5.1f}   {'직선' if fast[a] else '코너':<5}{A['t'][w].sum():>14.2f} → {Bp['t'][w].sum():.2f}{dT[w].sum():>+8.2f}"
          f"{A['p'][w].sum():>18.2f} → {Bp['p'][w].sum():.2f}{dP[w].sum():>+8.2f}"
          f"{A['p'][w].sum() / A['t'][w].sum():>15.2f} → {Bp['p'][w].sum() / Bp['t'][w].sum():.2f}{np.average(A['wall'][w], weights=A['n'][w]):>16.2f} → {np.average(Bp['wall'][w], weights=Bp['n'][w]):.2f}")

fig = plt.figure(figsize=(16, 9.2))
gs = fig.add_gridspec(3, 2, width_ratios=(1.15, 1.0), hspace=0.42, wspace=0.12)
ax = fig.add_subplot(gs[:, 0])
ra.draw_map(ax, k)
ax.plot(A["x"], A["y"], "-", color=C_A, lw=2.2, zorder=4, label=f"{na} ({sa['lap']:.2f} s, {sa['path']:.1f} m)")
ax.plot(Bp["x"], Bp["y"], "-", color=C_B, lw=2.2, zorder=5, label=f"{nbn} ({sb['lap']:.2f} s, {sb['path']:.1f} m)")
for i in range(1, len(x)):
    if abs(lo[i] / 5.0 - round(lo[i] / 5.0)) < 1e-6:
        ax.plot([A["x"][i], Bp["x"][i]], [A["y"][i], Bp["y"][i]], "-", color=INK, lw=1.0, zorder=6)
        ax.annotate(f"{lo[i]:.0f} m", (Bp["x"][i], Bp["y"][i]), xytext=(7, 7), textcoords="offset points", fontsize=9, color=INK, zorder=7)
xs_, ys_ = np.r_[A["x"][1:], Bp["x"][1:]], np.r_[A["y"][1:], Bp["y"][1:]]
ax.set_xlim(xs_.min() - 1.6, xs_.max() + 1.6); ax.set_ylim(ys_.min() - 1.6, ys_.max() + 1.6)
ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="upper right")
ax.set_title("자리 칸마다 달린 평균 자리 (검은 눈금 = 같은 자리 값, 5 m마다)", fontsize=10.5, loc="left")
ax.set_xticks([]); ax.set_yticks([])
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(x, A["v"], "-", color=C_A, lw=2, label=na)
ax1.plot(x, Bp["v"], "-", color=C_B, lw=2, label=nbn)
ax1.set_ylabel("속도 [m/s]"); ax1.legend(frameon=False, fontsize=9.5, loc="lower right", ncol=2)
ax1.set_title("자리별 속도", fontsize=10.5, loc="left")
ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax2.axhline(0, color=MUTED, lw=1)
ax2.plot(np.r_[0, hi], np.r_[0, np.cumsum(dP)], "-", color=INK, lw=2)
ax2.set_ylabel("누적 [m]"); ax2.set_title(f"{nbn}가 더 달린 길이의 누적 (끝 = {dP.sum():+.2f} m)", fontsize=10.5, loc="left")
ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
ax3.axhline(0, color=MUTED, lw=1)
ax3.plot(np.r_[0, hi], np.r_[0, np.cumsum(dT)], "-", color=INK, lw=2)
ax3.set_ylabel("누적 [s]"); ax3.set_xlabel("트랙을 따라 잰 자리 [m] (출발선 0)")
ax3.set_title(f"{nbn}가 더 쓴 시간의 누적 (끝 = {dT.sum():+.2f} s)", fontsize=10.5, loc="left")
for a_ in (ax1, ax2, ax3):
    a_.grid(color="#e6e4dd", lw=0.8); a_.set_axisbelow(True); a_.set_xlim(0, L)
    for sp_ in ("top", "right"):
        a_.spines[sp_].set_visible(False)
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(fast)]):
        if not fast[a]:
            a_.axvspan(lo[a], hi[b - 1], color="#f1efe8", zorder=0)
fig.suptitle(f"빈 트랙 한 바퀴: {nbn} {sb['lap']:.2f} s vs {na} {sa['lap']:.2f} s - 어디서 차이가 나는가 (옅은 띠 = 코너 쪽 칸)", fontsize=12.5, x=0.01, ha="left")
fig.savefig(out_prefix + ".png", dpi=115, bbox_inches="tight"); plt.close(fig)
print(f"그림: {out_prefix}.png")
