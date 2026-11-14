"""후보 (가) "닿을 뻔한 통과에도 그 통과의 예상 접촉 값"을 학습 batch에 넣으면 "상자에서 비키기" 신호가 생기나 (11-14 밤, 학습 전 반증).
입력: plan_rl --dump-rollout batch (batch_dump_26l.sh). 상자 통과 = 앞의 상자에 3 s 안에 가장 가까워지는 결정(같은 차·같은 상자는 한 번).
통과마다 중심 사이 최소 거리 dmin과 그 통과에서 세게 닿았나(가장 가까운 결정 앞뒤 0.3 s). 같은 batch에서 dmin 구간별 세게 닿을 확률 p와
한 번 세게 닿을 때 받은 접촉 값 c(초)를 잼. 보상 셋으로 advantage를 다시 계산(같은 critic 값, 같은 gamma·lambda):
  지금      : 저장된 보상 그대로 (다시 계산한 것이 저장된 advantage와 같은지 확인)
  가-더함   : 안 닿은 통과에도 p(dmin) x c 를 가장 가까운 결정에서 뺌
  가-바꿈   : 닿았든 안 닿았든 통과의 실제 상자 접촉 값을 빼고 그 대신 p(dmin) x c (기댓값으로 바꿈)
잡음은 다른 모든 것과 독립으로 뽑으므로 "advantage를 상자 반대쪽 조향 잡음에 회귀한 기울기"는 critic이 틀려도 치우치지 않음.
판정: 1.5-6 m 앞 결정에서 가-더함/가-바꿈의 기울기가 지금보다 뚜렷이 +(표준오차의 2배 넘게)가 아니면 (가) 버림.
인자: DUMP_DIR [첫 update, 기본 3]"""
import glob, os, sys
import numpy as np

DIR = sys.argv[1]; U0 = int(sys.argv[2]) if len(sys.argv) > 2 else 3
T_REF, KF = 7.5, 30
BINS = ((0.3, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 6.0))
DBINS = (0.0, 0.30, 0.40, 0.50, 0.60, 0.80)


def gae(r, v, v_last, done, gamma, lam):
    T = r.shape[0]; adv = np.zeros_like(r); g = np.zeros(r.shape[1])
    for t in range(T - 1, -1, -1):
        vn = v_last if t == T - 1 else v[t + 1]
        vn = np.where(done[t] > 0, v[t], vn)                      # plan_rl: a restart bootstraps from the value before it
        g = r[t] + gamma * vn - v[t] + gamma * lam * (1 - done[t]) * g
        adv[t] = g
    return adv


def load(path):
    z = np.load(path)
    r = z["r"].astype(np.float64); v = z["v"].astype(np.float64); done = z["done"].astype(np.float64)
    T, B = r.shape
    pose = z["pose"].astype(np.float64); x, y, yaw = pose[..., 0], pose[..., 1], pose[..., 2]
    noise = np.degrees(z["d_samp"].astype(np.float64) - z["d_mean"].astype(np.float64))
    bx = z["boxes"].astype(np.float32); K = bx.shape[2] // 4; bx = bx.reshape(T, B, K, 4)
    ds = bx[..., 3] * 5.0; ahead = (bx[..., 0] > 0.5) & (ds > 0.3)
    first = np.where(ahead, ds, 1e6).argmin(2)
    has = np.take_along_axis(ahead, first[..., None], 2)[..., 0]
    g_ = lambda c: np.take_along_axis(bx[..., c], first[..., None], 2)[..., 0].astype(np.float64) * 5.0
    dx, dy, dist = g_(1), g_(2), g_(3)
    wx = x + dx * np.cos(yaw) - dy * np.sin(yaw); wy = y + dx * np.sin(yaw) + dy * np.cos(yaw)
    hard_box = z["hard"][..., 1].astype(np.float64) > 0
    contact = z["contact"].astype(np.float64)                       # (T, B) reward part (negative), lap units
    dmin = np.full((T, B), np.inf); tmin = np.full((T, B), -1); side = np.zeros((T, B)); alive = np.ones((T, B), bool); full = np.ones((T, B), bool)
    for k in range(KF):
        tt = np.minimum(np.arange(T) + k, T - 1)
        inb = (np.arange(T) + k < T)[:, None] & alive
        ex, ey = wx - x[tt], wy - y[tt]; d = np.hypot(ex, ey)
        lat = -ex * np.sin(yaw[tt]) + ey * np.cos(yaw[tt])
        better = inb & (d < dmin)
        dmin = np.where(better, d, dmin); tmin = np.where(better, np.broadcast_to(tt[:, None], (T, B)), tmin); side = np.where(better, np.sign(lat), side)
        full &= ~((np.arange(T) + k >= T)[:, None] & alive)
        alive = inb & (done[tt] == 0)
    # passes: (car, closest decision, box position) once each
    passes = {}
    for t, b in zip(*np.nonzero(has & (dmin < 0.8) & (tmin >= 0))):
        key = (b, int(tmin[t, b]), round(wx[t, b], 1), round(wy[t, b], 1))
        if key not in passes:
            tc = int(tmin[t, b]); a, e = max(0, tc - 3), min(T, tc + 4)
            hit = bool(hard_box[a:e, b].any())
            cost = float(-contact[a:e, b][hard_box[a:e, b]].sum()) if hit else 0.0     # this pass's box contact price (lap units)
            passes[key] = (b, tc, float(dmin[t, b]), hit, cost, a, e)
    # merge passes of the same car closer than 3 decisions (the same box seen from successive decisions)
    P = sorted(passes.values(), key=lambda p: (p[0], p[1]))
    out = []
    for p in P:
        if out and out[-1][0] == p[0] and abs(out[-1][1] - p[1]) <= 3:
            if p[2] < out[-1][2]:
                out[-1] = p
            continue
        out.append(p)
    return dict(z=z, r=r, v=v, done=done, T=T, B=B, noise=noise, has=has, dist=dist, dmin=dmin, side=side, full=full, passes=out,
                v_last=z["v_last"].astype(np.float64), gamma=float(z["gamma"]), lam=float(z["lam"]), hard_box=hard_box, contact=contact,
                adv0=z["adv"].astype(np.float64))


files = sorted(glob.glob(os.path.join(DIR, "u*.npz")), key=lambda p: int(os.path.basename(p)[1:-4]))
files = [f for f in files if int(os.path.basename(f)[1:-4]) >= U0]
D = [load(f) for f in files]
allp = np.array([(p[2], p[3], p[4]) for d in D for p in d["passes"]])
print(f"{DIR}: update {U0}부터 {len(D)}개, 상자 통과 {len(allp)}번 (update당 {len(allp) / len(D):.0f}), 세게 닿음 {int(allp[:, 1].sum())}번")
c_hit = allp[allp[:, 1] > 0, 2].mean() if (allp[:, 1] > 0).any() else 0.0
print(f"  세게 닿은 통과의 상자 접촉 값 평균 {c_hit * T_REF:.2f} s")
p_bin = []
for lo, hi in zip(DBINS[:-1], DBINS[1:]):
    m = (allp[:, 0] >= lo) & (allp[:, 0] < hi)
    p = allp[m, 1].mean() if m.any() else 0.0
    p_bin.append(p)
    print(f"  중심 거리 {lo:.2f}-{hi:.2f} m: 통과 {int(m.sum()):5d}, 세게 닿음 {100 * p:5.1f} %  -> 예상 접촉 값 {p * c_hit * T_REF:.2f} s")


def p_of(dm):
    for (lo, hi), p in zip(zip(DBINS[:-1], DBINS[1:]), p_bin):
        if lo <= dm < hi:
            return p
    return 0.0


def rewards(d, kind):
    r = d["r"].copy()
    if kind == "now":
        return r
    for b, tc, dm, hit, cost, a, e in d["passes"]:
        ec = p_of(dm) * c_hit
        if kind == "add" and not hit:
            r[tc, b] -= ec
        if kind == "swap":
            if hit:
                r[a:e, b] += np.where(d["hard_box"][a:e, b], -d["contact"][a:e, b], 0.0)   # take this pass's box price back out
            r[tc, b] -= ec
    return r


def slope(yv, xv):
    xv = xv - xv.mean(); yv = yv - yv.mean(); sxx = (xv ** 2).sum()
    if len(xv) < 30 or sxx <= 0:
        return np.nan, np.nan
    b = (xv * yv).sum() / sxx
    return b, np.sqrt(((yv - b * xv) ** 2).sum() / (len(xv) - 2) / sxx)


chk = np.abs(gae(D[0]["r"], D[0]["v"], D[0]["v_last"], D[0]["done"], D[0]["gamma"], D[0]["lam"]) - D[0]["adv0"]).max() * T_REF
print(f"  (확인: 저장된 보상으로 다시 계산한 advantage와 저장된 advantage의 최대 차 {chk:.4f} s)")
print("\n상자 반대쪽 조향 잡음 1도당 advantage [s] (± 표준오차, 비) -- 3 s 안에 0.8 m 안으로 지나는 상자가 앞에 있는 결정")
print(f"  {'상자까지':>10} {'결정 수':>8}   {'지금':>24} {'가-더함':>24} {'가-바꿈':>24}")
ADV = {k: [gae(rewards(d, k), d["v"], d["v_last"], d["done"], d["gamma"], d["lam"]) * T_REF for d in D] for k in ("now", "add", "swap")}
for lo, hi in BINS:
    masks = [d["has"] & d["full"] & (d["dist"] >= lo) & (d["dist"] < hi) & (d["dmin"] < 0.8) & (d["side"] != 0) for d in D]
    a = np.concatenate([(d["noise"] * -d["side"])[m] for d, m in zip(D, masks)])
    cells = []
    for k in ("now", "add", "swap"):
        y = np.concatenate([A[m] for A, m in zip(ADV[k], masks)])
        b, se = slope(y, a)
        cells.append(f"{b:+.4f} ±{se:.4f} ({b / se:+.1f})")
    print(f"  {lo:>4.1f}-{hi:<4.1f} m {len(a):>8d}   " + " ".join(f"{c:>24}" for c in cells))

# 11-14 22:00: why no signal -- does one decision's steering noise change the pass at all? slope of the centre distance at the
# closest approach (dmin) on the away-from-box noise, and of the time the car was beside the box a little later
print("\n결정 하나의 조향 잡음이 그 통과를 얼마나 바꾸나: 상자 반대쪽 1도당 가장 가까운 중심 거리 변화 [cm] (± 표준오차, 비)")
for lo, hi in BINS:
    masks = [d["has"] & d["full"] & (d["dist"] >= lo) & (d["dist"] < hi) & (d["dmin"] < 0.8) & (d["side"] != 0) for d in D]
    a = np.concatenate([(d["noise"] * -d["side"])[m] for d, m in zip(D, masks)])
    y = np.concatenate([d["dmin"][m] for d, m in zip(D, masks)]) * 100
    b, se = slope(y, a)
    print(f"  {lo:>4.1f}-{hi:<4.1f} m  {b:+.3f} ±{se:.3f} ({b / se:+.1f})   잡음 표준편차 {a.std():.2f}도 -> 한 표준편차의 잡음이 바꾸는 틈 {abs(b) * a.std():.2f} cm")
