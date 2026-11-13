"""What the PPO update is given about "more room at a box": the training batch itself.  (11-13, structure audit)

Input: the rollouts `plan_rl --dump-rollout` saved from a checkpoint (`batch_dump.sh`): the checkpoint's own policy, noise,
world and critic, no update made. Per decision: the first 0.1 s steering the sample commanded and the one the mean would have,
the advantage the update would use, the reward parts, hard contacts, the boxes ahead along the track, the pose.

For the decisions with a box ahead that the car then passes within 0.8 m (or touches), by how far ahead the box is:
  - the slope of the advantage on the steering the noise added AWAY from the box [s per degree] and its standard error
    (the noise is drawn independently of everything else, so the slope is the batch's own estimate of d(return)/d(steering));
  - the same slope of what was actually collected in the 3 s after (no critic): contact price and progress apart;
  - how many such decisions an update holds, and the slope / standard error one update alone gives;
  - how much of the collected return's spread the critic explains there.
"Away": from the side the box is on when the car is level with it (also given: by the side the box is on at the decision).

    python3 batch_signal.py OUT_PREFIX DUMP_DIR [first update used, default 4]
"""
import glob
import os
import sys
import numpy as np
import rec_lib  # noqa: F401  (sets the Korean font)
import matplotlib.pyplot as plt

OUT, DIR = sys.argv[1], sys.argv[2]
U0 = int(sys.argv[3]) if len(sys.argv) > 3 else 4
T_REF = 7.5
BINS = ((0.3, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 6.0), (6.0, 10.0))
KF = 30                                   # 3 s ahead
LOG = open(OUT + ".txt", "w")


def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def one(path):
    z = np.load(path)
    adv = z["adv"].astype(np.float64) * T_REF; v = z["v"].astype(np.float64) * T_REF; r = z["r"].astype(np.float64) * T_REF
    done = z["done"].astype(bool); T, B = adv.shape
    gamma = float(z["gamma"])
    pose = z["pose"].astype(np.float64)
    x, y, yaw = pose[..., 0], pose[..., 1], pose[..., 2]
    noise_deg = np.degrees(z["d_samp"].astype(np.float64) - z["d_mean"].astype(np.float64))
    zs = (z["u"][..., 0] - z["mean"][..., 0]) / z["std"][..., 0]
    zv = (z["u"][..., 1] - z["mean"][..., 1]) / z["std"][..., 1]
    bx = z["boxes"].astype(np.float32); K = bx.shape[2] // 4; bx = bx.reshape(T, B, K, 4)
    ds = bx[..., 3] * 5.0
    ahead = (bx[..., 0] > 0.5) & (ds > 0.3)
    # the nearest box ahead along the track
    first = np.where(ahead, ds, 1e6).argmin(2)
    has = np.take_along_axis(ahead, first[..., None], 2)[..., 0]
    g = lambda c: np.take_along_axis(bx[..., c], first[..., None], 2)[..., 0].astype(np.float64) * 5.0
    dx, dy, dist = g(1), g(2), g(3)
    wx = x + dx * np.cos(yaw) - dy * np.sin(yaw); wy = y + dx * np.sin(yaw) + dy * np.cos(yaw)
    # the pass: the closest the car comes to that box in the next 3 s of the same episode, and the side the box is on then
    dmin = np.full((T, B), np.inf); side = np.zeros((T, B)); alive = np.ones((T, B), bool); seen_end = np.zeros((T, B), bool)
    hard = z["hard"].astype(np.float64); contact = np.abs(z["contact"].astype(np.float64)) * T_REF
    prog = (z["lap"].astype(np.float64) + z["shaping"].astype(np.float64)) * T_REF
    g_contact = np.zeros((T, B)); g_prog = np.zeros((T, B)); hit3 = np.zeros((T, B), bool); disc = np.ones((T, B))
    for k in range(KF):
        tt = np.minimum(np.arange(T) + k, T - 1)
        inb = (np.arange(T) + k < T)[:, None] & alive
        ex, ey = wx - x[tt], wy - y[tt]
        d = np.hypot(ex, ey)
        lat = -ex * np.sin(yaw[tt]) + ey * np.cos(yaw[tt])
        better = inb & (d < dmin)
        dmin = np.where(better, d, dmin); side = np.where(better, np.sign(lat), side)
        g_contact += np.where(inb, disc * contact[tt], 0.0); g_prog += np.where(inb, disc * prog[tt], 0.0)
        hit3 |= inb & (hard[tt][..., 1] > 0)
        seen_end |= (np.arange(T) + k >= T)[:, None] & alive       # the rollout ended before the 3 s did
        alive = inb & ~done[tt]
        disc = disc * gamma
    full = ~seen_end                                                # rows whose 3 s (or episode end) are inside the rollout
    ret = adv + v
    return dict(adv=adv, ret=ret, v=v, noise=noise_deg, z=zs.astype(np.float64), zv=zv.astype(np.float64), has=has, dist=dist, dy=dy, dmin=dmin, side=side, full=full,
                g_contact=g_contact, g_prog=g_prog, hit3=hit3, yaw=yaw, d_mean=np.degrees(z["d_mean"].astype(np.float64)),
                std=float(z["std"][0, 0, 0]), speed=pose[..., 3], x=x, y=y, done=done)


files = sorted(glob.glob(os.path.join(DIR, "u*.npz")), key=lambda p: int(os.path.basename(p)[1:-4]))
files = [f for f in files if int(os.path.basename(f)[1:-4]) >= U0]
D = [one(f) for f in files]
say(f"{DIR}: update {U0}부터 {len(D)}개 rollout, 한 rollout {D[0]['adv'].size}결정 · 첫 0.1 s 조향 잡음 표준편차 {D[0]['std']:.2f} (출력 단위) = "
    f"{np.mean([d['noise'].std() for d in D]):.2f}도")

# sign convention check: positive steering turns the car left (yaw grows)
d0 = D[0]; dyaw = np.diff(np.unwrap(d0["yaw"], axis=0), axis=0)
say(f"  (부호 확인: 평균 조향과 그 뒤 0.1 s 방향 변화의 상관 {np.corrcoef(d0['d_mean'][:-1].ravel(), dyaw.ravel())[0, 1]:+.2f} -> +면 왼쪽이 +)")


def slope(yv, xv):
    """OLS slope of yv on xv and its standard error"""
    xv = xv - xv.mean(); yv = yv - yv.mean()
    sxx = (xv ** 2).sum()
    if len(xv) < 30 or sxx <= 0:
        return np.nan, np.nan
    b = (xv * yv).sum() / sxx
    return b, np.sqrt(((yv - b * xv) ** 2).sum() / (len(xv) - 2) / sxx)


def cat(key, masks):
    return np.concatenate([d[key][m] for d, m in zip(D, masks)])


RES = {}
for name, away_of in (("나란할 때 상자가 있는 쪽의 반대", lambda d: -d["side"]), ("결정 순간 상자가 있는 쪽의 반대", lambda d: -np.sign(d["dy"]))):
    say(f"\n=== '상자 반대쪽' = {name} ===")
    say("  상자까지 [m]   결정 수/update (batch의 %)  3 s 안 세게 닿음 %  | advantage 기울기 [s/도] (±표준오차, 비)  한 update만: 비 중앙 · 같은 부호 update"
        "  | 실제로 받은 3 s: 접촉 값 기울기 · 진행 기울기 [s/도]")
    rows = []
    for lo, hi in BINS:
        masks = [d["has"] & d["full"] & (d["dist"] >= lo) & (d["dist"] < hi) & (d["dmin"] < 0.8) & (away_of(d) != 0) for d in D]
        a = np.concatenate([(d["noise"] * away_of(d))[m] for d, m in zip(D, masks)])
        adv = cat("adv", masks); gc = cat("g_contact", masks); gp = cat("g_prog", masks); h3 = cat("hit3", masks)
        b, se = slope(adv, a); bc, sec = slope(-gc, a); bp, sep = slope(gp, a)
        per = [slope(d["adv"][m], (d["noise"] * away_of(d))[m]) for d, m in zip(D, masks)]
        tper = np.array([p[0] / p[1] for p in per if np.isfinite(p[1])])
        n_upd = np.mean([m.sum() for m in masks])
        rows.append((lo, hi, n_upd, b, se, bc, sec, bp, sep, h3.mean(), np.median(tper) if len(tper) else np.nan,
                     int((np.sign(tper) == np.sign(b)).sum()), len(tper), adv.std(), a.std()))
        say(f"  {lo:>4.1f}~{hi:<4.1f}   {n_upd:>8.0f} ({n_upd / D[0]['adv'].size * 100:4.1f} %)   {h3.mean() * 100:>6.1f}        | {b:+.4f} ±{se:.4f} ({b / se:+.1f})"
            f"       {np.median(tper):+.1f} · {int((np.sign(tper) == np.sign(b)).sum())}/{len(tper)}"
            f"            | {bc:+.4f} ±{sec:.4f} · {bp:+.4f} ±{sep:.4f}")
    RES[name] = rows

# the critic there: how much of the collected return's spread is left as advantage
say("\n=== critic이 그 결정들의 return을 얼마나 설명하나 (가까이 지나가는 상자가 앞에 있는 결정) ===")
say("  상자까지 [m]   advantage 표준편차 [s]  (3 s 안에 세게 닿는 것 / 안 닿는 것)   return 표준편차 [s]   critic이 설명하는 몫   advantage 평균: 닿는 것 / 안 닿는 것 [s]")
for lo, hi in BINS:
    masks = [d["has"] & d["full"] & (d["dist"] >= lo) & (d["dist"] < hi) & (d["dmin"] < 0.8) for d in D]
    adv = cat("adv", masks); ret = cat("ret", masks); h3 = cat("hit3", masks)
    say(f"  {lo:>4.1f}~{hi:<4.1f}        {adv.std():.2f}  ({adv[h3].std():.2f} / {adv[~h3].std():.2f})                 {ret.std():.2f}"
        f"              {1 - adv.var() / ret.var():.2f}            {adv[h3].mean():+.2f} / {adv[~h3].mean():+.2f}")
masks = [~d["has"] & d["full"] for d in D]
adv = cat("adv", masks); ret = cat("ret", masks)
say(f"  앞 12 m에 상자 없음: advantage 표준편차 {adv.std():.2f} s, return 표준편차 {ret.std():.2f} s, critic이 설명하는 몫 {1 - adv.var() / ret.var():.2f} "
    f"({np.mean([m.mean() for m in masks]) * 100:.0f} % of batch)")
# sanity: with no box ahead the advantage must not depend on left/right noise
b, se = slope(adv, cat("noise", masks))
say(f"  (확인: 상자 없는 결정에서 advantage의 왼쪽 잡음 기울기 {b:+.4f} ±{se:.4f} s/도)")

# what one decision's noise does to the car: sideways distance and heading later, per degree of noise (the following decisions
# keep their own noise and the policy's own corrections), all speeds and at race speed
say("\n=== 잡음이 차를 실제로 얼마나 옮기나: 그 결정의 잡음 1도당, 몇 초 뒤 옆으로 간 거리 [cm] / 방향 [도] (같은 평균 조향끼리 견줌) ===")
for tag, v0, v1 in (("모든 속도", 0.0, 99.0), ("5~6.5 m/s", 5.0, 6.5)):
    sdn = np.mean([d["noise"][(d["speed"] > v0) & (d["speed"] < v1)].std() for d in D])
    big = np.mean([[(np.abs(d["noise"][(d["speed"] > v0) & (d["speed"] < v1)]) > t).mean() * 100 for t in (2, 3, 4)] for d in D], 0)
    ac = np.mean([np.corrcoef(d["z"][:-k][~d["done"][:-k]], d["z"][k:][~d["done"][:-k]])[0, 1] for d in D for k in (1,)])
    say(f"  [{tag}] 잡음이 더한 조향 목표의 표준편차 {sdn:.2f}도 · |잡음| > 2/3/4도인 결정 {big[0]:.1f}/{big[1]:.2f}/{big[2]:.3f} % · 다음 결정의 잡음과의 상관 {ac:+.2f}")
    for k in (1, 3, 5, 10, 15):
        rows = []
        for d in D:
            alive = np.ones_like(d["done"][:-k])
            for j in range(k):
                alive &= ~d["done"][j:len(d["done"]) - k + j]
            ex, ey = d["x"][k:] - d["x"][:-k], d["y"][k:] - d["y"][:-k]
            lat = -ex * np.sin(d["yaw"][:-k]) + ey * np.cos(d["yaw"][:-k])
            dyw = np.degrees((d["yaw"][k:] - d["yaw"][:-k] + np.pi) % (2 * np.pi) - np.pi)
            m = alive & (d["speed"][:-k] > v0) & (d["speed"][:-k] < v1) & (np.abs(lat) < 3)
            rows.append(np.stack([d["noise"][:-k][m], d["d_mean"][:-k][m], lat[m] * 100, dyw[m]], 1))
        A = np.concatenate(rows); X = np.stack([np.ones(len(A)), A[:, 1], A[:, 0]], 1)
        say(f"     {k * 0.1:.1f} s 뒤: 옆으로 {np.linalg.lstsq(X, A[:, 2], rcond=None)[0][2]:+.2f} cm · 방향 {np.linalg.lstsq(X, A[:, 3], rcond=None)[0][2]:+.2f}도")

# the same slopes for decisions at race speed only (where the replay tests were made and the hard hits happen)
say("\n=== 빠른 결정(5~6.5 m/s)만: 0.8 m 안으로 지나갈 상자, 상자 반대쪽으로 민 잡음 1도당 ===")
for name, away_of in (("나란할 때 기준", lambda d: -d["side"]), ("결정 순간 기준", lambda d: -np.sign(d["dy"]))):
    say(f"  [{name}] 상자까지 / 결정 수(update당) / advantage [s/도] / 3 s 안 세게 닿음 [%p/도] / 가장 가까울 때 거리 [cm/도] / 3 s 접촉 값 [s/도]")
    for lo, hi in BINS[1:]:
        mk = [d["has"] & d["full"] & (d["dist"] >= lo) & (d["dist"] < hi) & (d["dmin"] < 0.8) & (away_of(d) != 0) & (d["speed"] > 5.0) & (d["speed"] < 6.5) for d in D]
        a = np.concatenate([(d["noise"] * away_of(d))[m] for d, m in zip(D, mk)])
        rr = [slope(cat(k, mk).astype(float) * sc, a) for k, sc in (("adv", 1.0), ("hit3", 100.0), ("dmin", 100.0), ("g_contact", 1.0))]
        say(f"     {lo:g}~{hi:g} m  {len(a) / len(D):>5.0f}  " + "  ".join(f"{b:+.3f} ±{e:.3f}" for b, e in rr) + f"   (세게 닿음 {cat('hit3', mk).mean() * 100:.1f} %)")

# for scale: signals the same batch holds about things the training does follow (speed) and about the noise size itself
say("\n=== 견줄 것: 같은 batch 안의 다른 신호 (잡음 1 표준편차당 advantage [s], 비 = 기울기/표준오차, 한 update만의 비 중앙) ===")
allm = [d["full"] for d in D]
boxm = [d["has"] & d["full"] & (d["dist"] >= 1.5) & (d["dist"] < 6.0) & (d["dmin"] < 0.8) for d in D]
nobox = [~d["has"] & d["full"] for d in D]
for tag, masks in (("모든 결정", allm), ("앞 12 m에 상자 없는 결정", nobox), ("가까이 지나갈 상자가 1.5~6 m 앞인 결정", boxm)):
    for nm, f in (("첫 0.1 s 속도를 더 높인 잡음", lambda d: d["zv"]), ("조향 잡음의 크기 (z^2 - 1; -면 '잡음을 줄여라')", lambda d: d["z"] ** 2 - 1),
                  ("속도 잡음의 크기 (z^2 - 1)", lambda d: d["zv"] ** 2 - 1)):
        xs_ = np.concatenate([f(d)[m] for d, m in zip(D, masks)]); b, se = slope(cat("adv", masks), xs_)
        tp = np.array([slope(d["adv"][m], f(d)[m]) for d, m in zip(D, masks)]); tp = tp[:, 0] / tp[:, 1]
        say(f"  {tag:<28} {nm:<44} {b:+.4f} ±{se:.4f} ({b / se:+.1f})   한 update {np.median(tp):+.1f}")
for name, away_of in (("나란할 때 기준", lambda d: -d["side"]), ("결정 순간 기준", lambda d: -np.sign(d["dy"]))):
    mk = [m & (away_of(d) != 0) for d, m in zip(D, boxm)]
    xs_ = np.concatenate([(d["z"] * away_of(d))[m] for d, m in zip(D, mk)]); b, se = slope(cat("adv", mk), xs_)
    tp = np.array([slope(d["adv"][m], (d["z"] * away_of(d))[m]) for d, m in zip(D, mk)]); tp = tp[:, 0] / tp[:, 1]
    say(f"  {'가까이 지나갈 상자가 1.5~6 m 앞인 결정':<28} {'조향을 상자 반대쪽으로 민 잡음 (' + name + ')':<44} {b:+.4f} ±{se:.4f} ({b / se:+.1f})   한 update {np.median(tp):+.1f}"
        f"   [{np.mean([m.sum() for m in mk]):.0f}결정/update]")

# figure: the slope by distance, both definitions
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
xs = np.arange(len(BINS)); lab = [f"{lo:g}~{hi:g}" for lo, hi in BINS]
rows = RES["나란할 때 상자가 있는 쪽의 반대"]
b = np.array([r[3] for r in rows]); se = np.array([r[4] for r in rows])
bc = np.array([r[5] for r in rows]); sec = np.array([r[6] for r in rows]); bp = np.array([r[7] for r in rows]); sep = np.array([r[8] for r in rows])
ax[0].axhline(0, color="#868e96", lw=0.8)
ax[0].errorbar(xs - 0.12, 2 * b, 2 * 1.96 * se, fmt="o", color="#1c7ed6", capsize=3, label="update가 쓰는 advantage")
ax[0].errorbar(xs + 0.12, 2 * (bc + bp), 2 * 1.96 * np.hypot(sec, sep), fmt="s", color="#e8590c", capsize=3, label="그 뒤 3 s에 실제로 받은 것 (critic 없이)")
ax[0].set_xticks(xs); ax[0].set_xticklabels(lab); ax[0].set_xlabel("결정 순간 상자까지 거리 [m]")
ax[0].set_ylabel("상자 반대쪽으로 2도 더 튼 결정의 이득 [s]\n(막대 = 95 % 구간)")
ax[0].set_title("학습 batch 안의 신호: 탐색 잡음이 상자 반대쪽으로\n조향을 더 튼 결정이 얼마나 더 받았나", fontsize=11); ax[0].legend(fontsize=9)
tm = np.array([r[10] for r in rows])
ax[1].bar(xs, tm, color="#1c7ed6", width=0.55)
ax[1].axhline(0, color="#868e96", lw=0.8); ax[1].axhline(2, color="#868e96", lw=0.8, ls="--"); ax[1].axhline(-2, color="#868e96", lw=0.8, ls="--")
for i, r in enumerate(rows):
    ax[1].text(i, tm[i] + (0.15 if tm[i] >= 0 else -0.45), f"{r[2]:.0f}결정", ha="center", fontsize=9, color="#495057")
ax[1].set_xticks(xs); ax[1].set_xticklabels(lab); ax[1].set_xlabel("결정 순간 상자까지 거리 [m]")
ax[1].set_ylabel("update 한 번의 batch에서 기울기 / 표준오차 (중앙)")
ax[1].set_title("update 한 번이 그 신호를 얼마나 또렷이 보나\n(점선 ±2 밖이면 한 번에 방향이 보임)", fontsize=11)
for a in ax:
    a.grid(alpha=0.25)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle(sys.argv[4] if len(sys.argv) > 4 else "", fontsize=12)
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=110)
say("그림", OUT + ".png")
