"""(11-18, 사용자: "시험에서는 막 쭈르륵 장애물 깔지 말고, 빈 트랙에서 가는 길에 장애물을 그 근처에서 뽑히게", "맵당 한 대략 3개 씩?",
"적용하기 전에 한번 어떻게 뽑힐지 분포 및 예시 트랙 몇개를 보내봐") 시험 상자 배치 견본 -- 시험에는 아직 안 씀.
기준 길: 기준 checkpoint들의 시험 기록에서 빈 트랙 세션의 차 위치를 트랙 따라 0.25 m 칸마다 모아 중앙값(checkpoint마다 하나), 그 길들의
칸별 중앙값 = 기준 길(시험이 얼어 있어야 하므로 checkpoint마다 바뀌지 않는 하나). 시험용 상자 은행(학습에 안 쓴 400 배치)에서 중심이 기준 길
0.40 m 안인 상자만 '길 위'로 표시하고(학습의 --online-boxes와 같은 기준; qual26e u100 시험의 세게 닿은 상자 23/23이 그 안), 상자 세션마다
'길 위' 상자가 3개 이상인 배치 하나를 골라 그 가운데 3개(은행의 배치는 한 배치째로 차가 지나갈 길이 있는지 검사된 것 -> 그 일부도 지나감).
그림 1 = 분포(지금 시험과 나란히), 그림 2 = 예시 세션.
인자: 출력머리 기준기록npz... (첫 기록의 상자 세션 수·지도·지금 배치를 씀)"""
import sys
import numpy as np
import torch
from rec_lib import Rec, along_track, plt

OUT, RECS = sys.argv[1], sys.argv[2:]
BANK = "/home/shchon11/f1sim_runs/prop_banks_exam1_korea_2026_competition/korea_2026_competition_exam1.pt"
NEAR, N_BOX, BIN, SEED = 0.40, 3, 0.25, 20261112


def driven_line(fn):
    """(n_bins, 2) median position per 0.25 m bin along the track over the record's empty-track sessions (NaN where unseen)"""
    R = Rec(fn); z = R.z
    s, lap, L = along_track(z, R.T)
    opp = z["on"][:R.T].any(0); box = (R.sid[0] >= 0).any(-1)
    empty = ~opp & ~box
    xy = R.st[:R.T, empty, :2].reshape(-1, 2).astype(float); ss = s[:R.T, empty].reshape(-1)
    nb = int(np.ceil(L / BIN)); b = np.minimum((ss / BIN).astype(int), nb - 1)
    line = np.full((nb, 2), np.nan)
    for i in range(nb):
        m = b == i
        if m.sum() >= 5:
            line[i] = np.median(xy[m], 0)
    return line, L, R


lines, L, R0 = [], None, None
for fn in RECS:
    ln, L_, R_ = driven_line(fn)
    lines.append(ln); L = L_ if L is None else L
    R0 = R_ if R0 is None else R0
ref = np.nanmedian(np.stack(lines), 0)
ok = np.isfinite(ref).all(1)
refp = ref[ok]; refs = (np.arange(len(ref)) * BIN + 0.5 * BIN)[ok]
np.savez(OUT + "_refline.npz", xy=refp, s=refs, L=L)                     # exam_line_bank.py reads it
for fn, ln in zip(RECS, lines):
    m = np.isfinite(ln).all(1) & ok
    d = np.linalg.norm(ln[m] - ref[m], axis=1)
    print(f"{fn.split('exam1_')[-1].split('_ideal')[0]}: 기준 길과의 거리 중앙 {100 * np.median(d):.0f} cm, 90 % {100 * np.percentile(d, 90):.0f} cm, 최대 {100 * d.max():.0f} cm")
import os
if os.environ.get("EXAM_LINE_ONLY"):
    # (11-23) the reference line only (another map's exam): the preview below reads the Korea 2026 exam1 bank
    raise SystemExit(0)


def to_line(xy):
    """distance to the reference line, the along-track place of the nearest line point, and the signed side (+ = left)"""
    dd = np.linalg.norm(xy[:, None, :] - refp[None], axis=2)
    j = dd.argmin(1); dist = dd[np.arange(len(xy)), j]
    tan = refp[(j + 1) % len(refp)] - refp[j - 1]
    side = np.sign(tan[:, 0] * (xy[:, 1] - refp[j, 1]) - tan[:, 1] * (xy[:, 0] - refp[j, 0]))
    return dist, refs[j], side * dist, j


def gap(xy, j, fp_r):
    """[m] free width beside a box on its more open side, across the line's direction: wall along the normal minus the box's
    half size (its footprint's radius)"""
    tan = refp[(j + 1) % len(refp)] - refp[j - 1]
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1) / np.linalg.norm(tan, axis=1, keepdims=True)
    d = np.arange(0.0, 4.0, 0.02)
    out = []
    for p, n_, r in zip(xy, nrm, fp_r):
        best = 0.0
        for sg in (1.0, -1.0):
            pts = p[None] + sg * d[:, None] * n_[None]
            hit = R0.wall_dist(0, pts) <= 0.0
            w = d[hit.argmax()] if hit.any() else 4.0
            best = max(best, w - r)
        out.append(best)
    return np.array(out)


bank = torch.load(BANK, map_location="cpu", weights_only=False)["per_track"]["korea_2026_competition"]
sid, pose = bank["sid"].numpy(), bank["pose"].numpy()
live = sid >= 0
pxy = pose[..., :2].reshape(-1, 2)
pdist, ps_, plat, pj = to_line(pxy)
pdist, ps_, plat, pj = (a.reshape(sid.shape) for a in (pdist, ps_, plat, pj))
near = live & (pdist <= NEAR)
n_near = near.sum(1)
fp_r = {int(k): float(np.linalg.norm(R0.z[f"fp_{int(k)}"], axis=1).max()) for k in np.unique(sid[live]) if f"fp_{int(k)}" in R0.z.files}
print(f"은행: 배치 {sid.shape[0]}개, 상자 {int(live.sum())}개 중 기준 길 {NEAR:.2f} m 안 {int(near.sum())}개 ({100 * near.sum() / live.sum():.0f} %), "
      f"'길 위' 상자 3개 이상인 배치 {int((n_near >= N_BOX).sum())}개")

# the exam's box sessions (the first record's), each a layout with >= 3 pieces on the line, 3 of them
box_sess = np.nonzero((R0.sid[0] >= 0).any(-1) & ~R0.z["on"][:R0.T].any(0))[0]
rng = np.random.default_rng(SEED)
cand = np.nonzero(n_near >= N_BOX)[0]
new = []                                                            # (session, layout, piece) x 3
for b in box_sess:
    k = int(rng.choice(cand)); js = rng.choice(np.nonzero(near[k])[0], N_BOX, replace=False)
    new.append((b, k, np.sort(js)))
nx = np.array([pose[k, j, :2] for _, k, js in new for j in js]); nsid = np.array([sid[k, j] for _, k, js in new for j in js])
n_d, n_s, n_lat, n_j = to_line(nx)
n_gap = gap(nx, n_j, [fp_r.get(int(k), 0.3) for k in nsid])
sp = []
for i in range(len(new)):
    s3 = np.sort(n_s[3 * i: 3 * i + 3]); dif = np.diff(np.concatenate([s3, [s3[0] + L]]))
    sp.extend(dif.tolist())
# today's exam: the boxes each box session starts with
cx, csid, ccount = [], [], []
for b in box_sess:
    m = R0.sid[0, b] >= 0
    cx.extend(R0.pose[0, b, m, :2].tolist()); csid.extend(R0.sid[0, b, m].tolist()); ccount.append(int(m.sum()))
cx = np.array(cx, float); c_d, c_s, c_lat, c_j = to_line(cx)
c_gap = gap(cx, c_j, [fp_r.get(int(k), 0.3) for k in csid])
print(f"상자 세션 {len(box_sess)}개 | 지금 시험: 세션당 {np.mean(ccount):.1f}개 ({min(ccount)}-{max(ccount)}), 기준 길 {NEAR:.2f} m 안 {100 * np.mean(c_d <= NEAR):.0f} % | "
      f"견본: 세션당 {N_BOX}개 전부 {NEAR:.2f} m 안, 옆 빈 폭 중앙 {np.median(n_gap):.2f} m (최소 {n_gap.min():.2f}), 한 세션 안 상자 사이 트랙 거리 중앙 {np.median(sp):.1f} m")

# figure 1: the distribution, today's exam beside the proposal
C_NEW, C_OLD, C_LINE = "#d95f02", "#8c8c8c", "#1f77b4"
fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 4, width_ratios=[1.05, 1.05, 1, 1])
for col, (xy_, c_, ttl) in enumerate(((cx, C_OLD, f"지금 시험: 상자 세션 {len(box_sess)}개, 세션당 {np.mean(ccount):.1f}개 ({len(cx)}개)"),
                                      (nx, C_NEW, f"견본: 세션당 {N_BOX}개, 기준 길 {NEAR:.1f} m 안 ({len(nx)}개)"))):
    ax = fig.add_subplot(gs[:, col]); R0.draw_map(ax, 0)
    ax.plot(refp[:, 0], refp[:, 1], "-", color=C_LINE, lw=1.2, label="기준 길 (빈 트랙에서 달린 길)")
    ax.scatter(xy_[:, 0], xy_[:, 1], s=9, color=c_, alpha=0.75, label="상자 중심", zorder=3)
    ax.set_title(ttl, fontsize=10); ax.legend(loc="lower right", fontsize=8)
ax = fig.add_subplot(gs[0, 2])
bins = np.arange(0, L + 2, 2.0)
ax.hist(c_s, bins, color=C_OLD, alpha=0.6, label="지금 시험", weights=np.full(len(c_s), 1.0 / len(box_sess)))
ax.hist(n_s, bins, color=C_NEW, alpha=0.75, label="견본", weights=np.full(len(n_s), 1.0 / len(box_sess)))
ax.set_xlabel("트랙 따라 위치 [m] (출발선부터)"); ax.set_ylabel("세션당 상자 수 / 2 m"); ax.legend(fontsize=8); ax.set_title("트랙 어디에 놓이나", fontsize=10)
ax = fig.add_subplot(gs[0, 3])
lb = np.arange(-2.0, 2.05, 0.1)
ax.hist(np.clip(c_lat, -2, 2), lb, color=C_OLD, alpha=0.6, label="지금 시험", density=True)
ax.hist(n_lat, lb, color=C_NEW, alpha=0.75, label="견본", density=True)
ax.axvspan(-NEAR, NEAR, color=C_LINE, alpha=0.08)
ax.set_xlabel("기준 길에서 상자 중심까지 [m] (+ = 왼쪽)"); ax.set_ylabel("비율"); ax.legend(fontsize=8); ax.set_title("길에서 얼마나 떨어져 놓이나", fontsize=10)
ax = fig.add_subplot(gs[1, 2])
gb = np.arange(0, 2.6, 0.1)
ax.hist(np.clip(c_gap, 0, 2.5), gb, color=C_OLD, alpha=0.6, label="지금 시험", density=True)
ax.hist(np.clip(n_gap, 0, 2.5), gb, color=C_NEW, alpha=0.75, label="견본", density=True)
ax.axvline(2 * R0.hw, color="k", ls=":", lw=1); ax.text(2 * R0.hw, ax.get_ylim()[1] * 0.92, " 차 폭", fontsize=8)
ax.set_xlabel("상자 옆 빈 폭, 넓은 쪽 [m]"); ax.set_ylabel("비율"); ax.legend(fontsize=8); ax.set_title("옆으로 지나갈 폭", fontsize=10)
ax = fig.add_subplot(gs[1, 3])
ax.hist(sp, np.arange(0, L + 2, 2.0), color=C_NEW, alpha=0.75)
ax.set_xlabel("한 세션 안 이웃 상자 사이 트랙 거리 [m]"); ax.set_ylabel("수"); ax.set_title(f"견본: 상자끼리 간격 (한 바퀴 {L:.1f} m)", fontsize=10)
fig.suptitle("시험 상자 배치: 지금 vs 견본 (빈 트랙에서 달린 길 근처 3개) -- 아직 시험에 안 씀", fontsize=12)
fig.tight_layout(); fig.savefig(OUT + "_dist.png", dpi=80); print("그림", OUT + "_dist.png")

# figure 2: example sessions
pick = np.linspace(0, len(new) - 1, 8).round().astype(int)
fig, axs = plt.subplots(2, 4, figsize=(17, 13))
for ax, i in zip(axs.flat, pick):
    b, k, js = new[i]
    R0.draw_map(ax, 0)
    ax.plot(refp[:, 0], refp[:, 1], "-", color=C_LINE, lw=1.0)
    for n_, j in enumerate(js):
        q = R0.z[f"fp_{int(sid[k, j])}"].astype(float); p = pose[k, j]; c, s_ = np.cos(p[2]), np.sin(p[2])
        poly = np.stack([p[0] + q[:, 0] * c - q[:, 1] * s_, p[1] + q[:, 0] * s_ + q[:, 1] * c], 1)
        ax.fill(poly[:, 0], poly[:, 1], color="#a0522d", zorder=4)
        ax.add_patch(plt.Circle(p[:2], 0.9, fill=False, color=C_NEW, lw=1.5, zorder=5))
        ax.text(p[0] + 1.0, p[1], f"{n_ + 1}: {100 * abs(plat[k, j]):.0f} cm {'왼' if plat[k, j] > 0 else '오른'}쪽", fontsize=8, color="#7a2e00", zorder=6)
    ax.set_title(f"세션 {b} (은행 배치 {k}): 상자 {N_BOX}개", fontsize=10)
fig.suptitle("견본 예시 세션 8개: 파란 선 = 기준 길, 갈색 = 상자(주황 원은 찾기 쉽게), 숫자 = 기준 길에서 상자 중심까지", fontsize=12)
fig.tight_layout(); fig.savefig(OUT + "_examples.png", dpi=70); print("그림", OUT + "_examples.png")
