"""상자가 '보인다'가 policy 안에서도 그런가 (11-12, 사용자: "희안하네. 보고도 미리 안 피하다니").
box_profile.py의 '보이기 시작'은 기하(스캐너에서 상자 중심까지 벽이 안 가림)일 뿐, 신경망이 그것을 읽었다는 뜻은 아님. 여기서는 길 위 상자를
닿지 않고 지난 통과(혼자 달리는 상자 세션)마다, 상자가 보이기 시작한 때를 0으로 놓고 그 전후의 결정에서
 - LiDAR 인코딩(256개), 기억 값(128개)
 - 출력: 지금 실행되는 첫 0.1 s의 조향·속도, plan 뒤쪽(0.5~1.0 s)의 조향
이 '같은 세션이 상자를 치운 뒤 같은 자리를 지날 때'와 다른지를 잼. 기준은 치운 뒤의 바퀴들끼리 서로 다른 정도(짝 비교라 자리 효과가 지워짐):
AUC 0.5 = 상자 없을 때와 구분 안 됨, 1.0 = 늘 구분됨. 다른 상자가 함께 보이는 결정은 뺌. 보이기 전(−)의 값이 0.5 근처여야 비교가 깨끗함.
자리 맞춤은 기록된 참 위치(분석에만 씀). 인자: checkpoint 기록npz 통과행npy [그림png]"""
import sys
import numpy as np
import torch
from f1sim.learn.plan_policy import load_actor2
from f1sim import plan_steps as PS
from rec_lib import Rec, plt, along_track, auc, auc_se

torch.set_grad_enabled(False)
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
TAUS = [-5, -3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
START, NEAR, YAW_TOL = 70, 0.30, 0.25
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
actor = load_actor2(ck, torch.device("cpu"), check_mount=False)
assert actor.mem_front, "기억이 몸통 앞에 있는 policy만"
R = Rec(sys.argv[2]); z = R.z; T, B, dt = R.T, R.B, R.dt
rows = np.load(sys.argv[3]); fig_out = sys.argv[4] if len(sys.argv) > 4 else ""
mm = np.load(sys.argv[2][:-4] + "_mem.npz")
F = mm["feat"].astype(np.float32); H = mm["mem"].astype(np.float32)
NS = F.shape[2] - 36 if F.shape[2] > 256 else F.shape[2]; NS = 256
O = np.zeros((T, B, 2 * PS.N_STEP), np.float32)
for t in range(T):
    O[t] = torch.tanh(actor.heads(actor.trunk(torch.cat([torch.tensor(F[t]), torch.tensor(H[t])], 1)))[0]).numpy()
st = R.st.astype(float); xy = st[:, :, :2]; yaw = st[:, :, 2]; V = np.abs(st[:, :, 3]); pose = R.pose.astype(float)
k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
s, lap, L = along_track(z, T)
mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
FOV, RMAX = np.radians(135.0), 10.0


def sees(tk, t, b, c):
    """exam_boxes.sees: the box centre c in the scan's direct view at decision t"""
    sx, sy = xy[t, b, 0] + mount * np.cos(yaw[t, b]), xy[t, b, 1] + mount * np.sin(yaw[t, b])
    dx, dy = c[0] - sx, c[1] - sy; dist = np.hypot(dx, dy)
    a = (np.arctan2(dy, dx) - yaw[t, b] + np.pi) % (2 * np.pi) - np.pi
    if dist > RMAX or abs(a) > FOV:
        return False
    n = max(int((dist - 0.25) / 0.05), 1)
    f = np.linspace(0.0, max(dist - 0.25, 0.0) / max(dist, 1e-6), n)
    return bool((R.wall_dist(tk, np.stack([sx + f * dx, sy + f * dy], 1)) > 0.0).all())


def refs(b, k):
    """decisions of the same session after the boxes were taken away at the same place and heading, one per lap"""
    kk = np.arange(k_rm + 20, T - 1)
    kk = kk[~R.rs[kk, b]]
    d = np.linalg.norm(xy[kk, b] - xy[k, b], axis=1)
    dy_ = np.abs((yaw[kk, b] - yaw[k, b] + np.pi) % (2 * np.pi) - np.pi)
    out = []
    for l_ in np.unique(lap[kk, b]):
        m = (lap[kk, b] == l_) & (dy_ < YAW_TOL)
        if m.any():
            i = np.nonzero(m)[0][d[m].argmin()]
            if d[i] < NEAR:
                out.append(int(kk[i]))
    return out


sel = (rows[:, IGAP] < 0.40) & (rows[:, IOPP] == 0) & (rows[:, ITOUCH] == 0) & (rows[:, IT] >= START) & (rows[:, IT] < k_rm - 3) & (rows[:, ISEE] > 0)
solo = np.nonzero(~R.on[:T].any(0))[0]
post = np.concatenate([np.stack([np.full(T - k_rm - 21, b), np.arange(k_rm + 20, T - 1)], 1) for b in solo])
sd = {name: X[post[:, 1], post[:, 0]].std(0).clip(min=1e-3) for name, X in (("scan", F[:, :, :NS]), ("mem", H), ("out", O))}
PARTS = (("LiDAR 인코딩", F[:, :, :NS], sd["scan"], slice(None)), ("기억 값", H, sd["mem"], slice(None)),
         ("출력: 지금 실행되는 첫 조향", O, sd["out"], slice(0, 1)), ("출력: 지금 실행되는 첫 속도", O, sd["out"], slice(PS.N_STEP, PS.N_STEP + 1)),
         ("출력: plan 뒤쪽(0.5~1.0 s) 조향", O, sd["out"], slice(5, PS.N_STEP)))
S = {tau: [] for tau in TAUS}
n_other = 0
for r in rows[sel]:
    b, j, t = int(r[IB]), int(r[IJ]), int(r[IT]); tk = int(R.tid[0, b]); k_see = t - int(round(r[ISEE] / dt))
    others = [jj for jj in np.nonzero(R.sid[0, b] >= 0)[0] if jj != j]
    for tau in TAUS:
        k = k_see + tau
        if k < START or k > t or R.rs[k - 20: k + 1, b].any():
            continue
        if any(sees(tk, k, b, pose[k, b, jj, :2]) for jj in others):
            n_other += 1
            continue
        c = refs(b, k)
        if len(c) < 3:
            continue
        S[tau].append((k, b, c, t - k, float(np.linalg.norm(pose[k, b, j, :2] - xy[k, b])), float(r[ISHIFT]), int(r[IN])))


def apart(X, sdv, cols, samp):
    """AUC of 'the box-side decision is farther from the mean of its place-matched decisions (all but the first) than the first is'"""
    dp, dc = [], []
    for k, b, c, *_ in samp:
        rest = np.mean([X[kc, b][cols] for kc in c[1:]], 0)
        dp.append(np.sqrt((((X[k, b][cols] - rest) / sdv[cols]) ** 2).mean())); dc.append(np.sqrt((((X[c[0], b][cols] - rest) / sdv[cols]) ** 2).mean()))
    dp, dc = np.array(dp), np.array(dc)
    a = auc(np.concatenate([dp, dc]), np.concatenate([np.ones(len(dp), bool), np.zeros(len(dc), bool)]))
    return a, auc_se(a, len(dp), len(dc)), float(np.median(dp)), float(np.median(dc))


print(f"{z['exam']} · {R.name} · 실행 {z['mode']} · 상자 {z['count']} — 길 위 상자를 닿지 않고 지난 통과 {int(sel.sum())}번 (혼자 달리는 상자 세션, 첫 7 s 뺌). "
      f"다른 상자가 함께 보여 뺀 결정 {n_other}개. 자리 맞춤: 치운 뒤 같은 세션, {NEAR:.2f} m·{np.degrees(YAW_TOL):.0f}도 안, 바퀴 3개 이상")
print("보이기 시작한 뒤 | 표본 | 상자까지 거리 | 가장 가까워지기까지 | " + " | ".join(p[0] for p in PARTS) + "   (AUC ± 표준오차; 괄호 = 상자 쪽 거리 / 짝끼리 거리)")
RES = {}
for tau in TAUS:
    samp = S[tau]
    if len(samp) < 12:
        continue
    res = [apart(X, sdv, cols, samp) for _, X, sdv, cols in PARTS]
    RES[tau] = (len(samp), res)
    print(f"  {tau * dt:+.1f} s | {len(samp)} | {np.median([q[4] for q in samp]):.1f} m | {np.median([q[3] for q in samp]) * dt:.1f} s | "
          + " | ".join(f"{a:.2f} ± {se:.2f} ({p_ / max(c_, 1e-9):.1f}배)" for a, se, p_, c_ in res))
big = {tau: [q for q in S[tau] if q[5] >= 0.30] for tau in TAUS}
print("같은 표, 0.30 m 넘게 비킨 통과만")
for tau in TAUS:
    samp = big[tau]
    if len(samp) < 12:
        continue
    res = [apart(X, sdv, cols, samp) for _, X, sdv, cols in PARTS]
    print(f"  {tau * dt:+.1f} s | {len(samp)} | {np.median([q[4] for q in samp]):.1f} m | {np.median([q[3] for q in samp]) * dt:.1f} s | "
          + " | ".join(f"{a:.2f} ± {se:.2f} ({p_ / max(c_, 1e-9):.1f}배)" for a, se, p_, c_ in res))

# (3) a read-out fitted on half of the sessions and scored on the other half: is the box in there at all, however small
print("(3) 세션 절반으로 맞춘 선형 판독기를 나머지 절반에서 채점 (상자 쪽 결정 대 같은 세션·같은 자리의 상자 없는 결정 하나; 보인 뒤 0~1.0 s의 표본을 모아서 맞춤)")
READ = (("LiDAR 인코딩", F[:, :, :NS]), ("기억 값", H), ("LiDAR 인코딩 + 기억 값", np.concatenate([F[:, :, :NS], H], 2)))
pool = [(tau, q) for tau in TAUS for q in S[tau]]
neg_of = lambda q: q[2][q[6] % len(q[2])]                       # another lap after removal for each pass of the same box
PR = {}
for name, X in READ:
    xp = np.stack([X[q[0], q[1]] for _, q in pool]); xn = np.stack([X[neg_of(q), q[1]] for _, q in pool])
    g = np.array([q[1] for _, q in pool]); tau_ = np.array([tau for tau, _ in pool])
    score_p, score_n = np.full(len(pool), np.nan), np.full(len(pool), np.nan)
    for half in (0, 1):
        tr = (g % 2 == half) & (tau_ >= 0); te = g % 2 != half
        Xtr = np.concatenate([xp[tr], xn[tr]]); ytr = np.concatenate([np.ones(tr.sum()), -np.ones(tr.sum())])
        mu, sg = Xtr.mean(0), Xtr.std(0).clip(min=1e-3); Z = (Xtr - mu) / sg
        w = np.linalg.solve(Z.T @ Z + 0.1 * len(Z) * np.eye(Z.shape[1]), Z.T @ ytr)
        score_p[te] = ((xp[te] - mu) / sg) @ w; score_n[te] = ((xn[te] - mu) / sg) @ w
    PR[name] = {}
    for tau in TAUS:
        m = tau_ == tau
        if m.sum() < 12:
            continue
        a = auc(np.concatenate([score_p[m], score_n[m]]), np.concatenate([np.ones(m.sum(), bool), np.zeros(m.sum(), bool)]))
        PR[name][tau] = (a, auc_se(a, int(m.sum()), int(m.sum())), float((score_p[m] > score_n[m]).mean()))
print("  보이기 시작한 뒤 | " + " | ".join(f"{n_} (AUC · 짝보다 높게 읽은 비율)" for n_, _ in READ))
for tau in TAUS:
    if tau in PR[READ[0][0]]:
        print(f"  {tau * dt:+.1f} s | " + " | ".join(f"{PR[n_][tau][0]:.2f} ± {PR[n_][tau][1]:.2f} · {PR[n_][tau][2] * 100:.0f} %" for n_, _ in READ))

if fig_out and RES:
    COL = ["#2a78d6", "#7a5cc7", "#eb6834", "#1f9d8a", "#b0457b"]
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.2))
    x = np.array(sorted(RES)) * dt
    for i, (name, *_r) in enumerate(PARTS):
        a = np.array([RES[tau][1][i][0] for tau in sorted(RES)]); se = np.array([RES[tau][1][i][1] for tau in sorted(RES)])
        ax.fill_between(x, a - se, a + se, color=COL[i], alpha=0.12, lw=0)
        ax.plot(x, a, "-o", color=COL[i], lw=2.2, ms=5, label=name + (" (전체가 달라진 정도)" if i < 2 else ""))
    xr = np.array(sorted(PR["LiDAR 인코딩"])) * dt
    ax.plot(xr, [PR["LiDAR 인코딩"][tau][0] for tau in sorted(PR["LiDAR 인코딩"])], "--s", color=COL[0], lw=1.8, ms=4, label="LiDAR 인코딩 (맞춘 판독기)")
    ax.axhline(0.5, color="#333333", lw=0.9, ls=":"); ax.axvline(0.0, color="#333333", lw=1.1, ls="--")
    ax.text(0.01, 1.03, "상자가 보이기 시작", fontsize=10, color="#333333", va="bottom")
    ax.text(x[0], 0.505, "0.5 = 상자 없을 때와 구분 안 됨", fontsize=9, color="#333333", va="bottom")
    ax.set_ylim(0.3, 1.08); ax.set_xlabel("상자가 보이기 시작한 뒤 [s]  (− = 아직 안 보임)"); ax.set_ylabel("상자 없는 같은 자리와 구분되는 정도 (AUC)")
    ax.set_title(f"상자가 보인 뒤, policy 안의 무엇이 언제 달라지나 — {R.name} (띠 = 표준오차)", fontsize=12)
    ax.legend(fontsize=10, frameon=False, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_out, dpi=80); print("그림", fig_out)
