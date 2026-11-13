"""Does a training batch hold a direction for the first 0.1 s that repeats from update to update?  (11-13, structure audit)

`batch_signal.py` asks about one direction I name ("away from the box"); naming the side is the weak part (the side a box 5 m
ahead is on in the car's frame says little on a curved track, and the side at the pass depends on the noise itself). PPO needs
no such name: its gradient moves the mean wherever `advantage x noise` is consistently signed in similar states. So here the
decisions are put in cells of similar state (box distance x the box's bearing side x how the car is turning x speed) and, per
cell, the slope of the advantage on the noise [s per 1 SD of noise] is taken on two halves of the updates:
  - "true size": sqrt(mean over cells of slope(half A) x slope(half B)) -- the halves' errors are independent, so this is the
    size of the direction that is really there (0 if the cells' slopes are noise);
  - "seen in one update": that size / the slope's standard error in ONE update's worth of the cell.
The same for cells without a box (turn x speed) and for the speed noise, as a scale.

With noise that stays over decisions (`plan_rl --noise-corr RHO`) the noise of a decision is tied to where the earlier noise has
already put the car, so its slope mixes "what this push does" with "the critic misjudging the displaced state". The part of the
noise that is NEW at the decision (z_t - RHO z_t-1, scaled to SD 1; it then fades by RHO per decision) is independent of the
state, so its slope is the clean effect of a push that starts now. Both are printed.

Also, over the box cells: in the direction the advantage favours, how the share of hard box hits within 3 s and the 3 s of
progress change (is the direction about not hitting, or about pace).

    python3 batch_cells.py OUT.txt NAME=DUMP_DIR[=RHO] [...]        (first 3 updates of every dump are dropped)
"""
import sys, io, contextlib
import numpy as np

LOG = None


def say(*a):
    print(*a, flush=True)
    if LOG is not None:
        print(*a, file=LOG, flush=True)


def load(d, rho=0.0):
    sys.modules.pop("batch_signal", None)
    sys.argv = ["x", "/tmp/claude-1000/-home-shchon11-F1tenth-E2E/b681fa17-d53e-4caa-8c76-9ce9b0410ed3/scratchpad/bc_tmp", d, "4"]
    with contextlib.redirect_stdout(io.StringIO()):
        import batch_signal as B
    for q in B.D:
        for k in ("z", "zv"):
            zz = q[k]; zi = zz.copy()
            if rho > 0:
                zi[1:] = np.where(q["done"][:-1], zz[1:], (zz[1:] - rho * zz[:-1]) / np.sqrt(1.0 - rho * rho))
            q[k + "i"] = zi
        if rho > 0:
            q["full"] = q["full"].copy(); q["full"][0] = False               # no earlier noise to take off
    return B.D


def slope(y, x):
    x = x - x.mean(); sxx = (x * x).sum()
    if len(y) < 30 or sxx <= 0:
        return np.nan, np.nan
    b = (x * (y - y.mean())).sum() / sxx; r = y - y.mean() - b * x
    return b, np.sqrt((r * r).sum() / (len(y) - 2) / sxx)


def cells_box(d):
    db = np.digitize(d["dist"], (1.5, 2.5, 4.0, 6.0)) - 1                    # 0..2 inside 1.5-6 m
    m = d["has"] & d["full"] & (db >= 0) & (db <= 2)
    c = db * 30 + np.digitize(d["dy"], (-0.75, -0.25, 0.25, 0.75)) * 6 + np.digitize(d["d_mean"], (-2.0, 2.0)) * 2 + (d["speed"] > 5.0)
    return m, c


def cells_free(d):
    m = ~d["has"] & d["full"]
    return m, np.digitize(d["d_mean"], (-6.0, -2.0, 2.0, 6.0)) * 3 + np.digitize(d["speed"], (3.5, 5.0))


def measure(D, cells, zkey):
    A, Bh = D[0::2], D[1::2]
    def per(Ds):
        y = np.concatenate([d["adv"][cells(d)[0]] for d in Ds]); x = np.concatenate([d[zkey][cells(d)[0]] for d in Ds])
        c = np.concatenate([cells(d)[1][cells(d)[0]] for d in Ds])
        return {k: slope(y[c == k], x[c == k]) + ((c == k).sum(),) for k in np.unique(c)}
    pa, pb, pall = per(A), per(Bh), per(D)
    ks = [k for k in pall if k in pa and k in pb and np.isfinite(pa[k][0]) and np.isfinite(pb[k][0])]
    w = np.array([pall[k][2] for k in ks], float); w /= w.sum()
    power = float((w * np.array([pa[k][0] * pb[k][0] for k in ks])).sum())
    se1 = np.array([pall[k][1] for k in ks]) * np.sqrt(len(D))                # one update's standard error of a cell's slope
    chi = float(sum((pall[k][0] / pall[k][1]) ** 2 for k in ks))
    n_upd = sum(pall[k][2] for k in ks) / len(D)
    # in the favoured direction of every cell (sign from half A, effect measured on half B): hard hits and progress
    eff = {}
    for key, sc in (("hit3", 100.0), ("g_prog", 1.0), ("g_contact", 1.0)):
        yb = np.concatenate([d[key][cells(d)[0]] for d in Bh]).astype(float) * sc; xb = np.concatenate([d[zkey][cells(d)[0]] for d in Bh])
        cb = np.concatenate([cells(d)[1][cells(d)[0]] for d in Bh])
        eff[key] = float(sum(wk * np.sign(pa[k][0]) * slope(yb[cb == k], xb[cb == k])[0] for wk, k in zip(w, ks)))
    eff["adv"] = float(sum(wk * np.sign(pa[k][0]) * pb[k][0] for wk, k in zip(w, ks)))
    return np.sign(power) * np.sqrt(abs(power)), float((w * se1).sum()), chi, len(ks), n_upd, eff


if __name__ == "__main__":
    LOG = open(sys.argv[1], "w")
    runs = [a.split("=") for a in sys.argv[2:]]
    for r in runs:
        name, d, rho = r[0], r[1], float(r[2]) if len(r) > 2 else 0.0
        D = load(d, rho)
        ac = np.mean([np.corrcoef(q["zi"][1:][q["full"][1:]], q["z"][:-1][q["full"][1:]])[0, 1] for q in D])
        say(f"\n=== {name}: rollout {len(D)}개 (확인: 그 결정에서 새로 생긴 잡음과 직전 결정의 잡음의 상관 {ac:+.3f}) ===")
        say("  상황 묶음 / 잡음 / 결정 수(update당) / 묶음 수 / 진짜로 있는 방향 신호의 크기 [잡음 1 표준편차당 s] / 한 update에서 한 묶음의 표준오차 [s] / "
            "한 update에 보이는 정도(크기/표준오차) / 묶음들이 0과 다른 정도 (chi2/묶음 수, 1이면 잡음뿐) / "
            "그 방향으로 1 표준편차 밀면: advantage [s] · 3 s 안 세게 닿음 [%p] · 3 s 진행 [s] · 3 s 접촉 값 [s]")
        for tag, cells in (("가까운 상자 1.5~6 m 앞", cells_box), ("앞 12 m에 상자 없음", cells_free)):
            for nm, zk in (("조향", "z"), ("속도", "zv")) + ((("조향(새로 생긴 몫)", "zi"), ("속도(새로 생긴 몫)", "zvi")) if rho > 0 else ()):
                s_, se1, chi, nc, n, eff = measure(D, cells, zk)
                say(f"  {tag:<16} {nm:<12}  {n:>7.0f}  {nc:>3d}  {s_:+.4f}  {se1:.4f}  {s_ / se1:+.2f}  {chi / nc:.2f}   "
                    f"{eff['adv']:+.3f} · {eff['hit3']:+.2f} · {eff['g_prog']:+.3f} · {eff['g_contact']:+.3f}")
