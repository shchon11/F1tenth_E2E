"""GRU 기억이 지난 일을 얼마나 오래 들고 있나 (11-12, 사용자: "기억이 반복이 되던데 그 반복되는거에서 특정 구간을 뭔가 이용할 수는
없나", "gru 자체에서 바로 써먹을 수 있는게 있는지"). 시험 기록(exam_record.py, F1SIM_EXAM_SAVE_MEM=1)에 남긴 결정별 입력 특징(feat)과
기억 상태(mem)만으로, 주행을 다시 하지 않고 CPU에서:
 (1) 확인: 기록된 특징으로 GRU를 한 결정 다시 돌리면 기록된 기억이 그대로 나오는가
 (2) 지우고 다시 채우기: k 결정 전의 기억을 0으로 지우거나(지움) 다른 세션의 같은 시각 기억으로 바꾼 뒤(바꿈) 이 세션의 입력을 그대로 넣어
     지금까지 돌린 기억이 실제 기억과 얼마나 다른가. k가 커져도 남는 차이만큼만 기억이 'k 결정보다 오래된 일'을 들고 있는 것
     (바퀴마다 되풀이되는 모양이 '같은 길이라 같은 입력이 들어와서'인지 '지난 바퀴를 들고 있어서'인지를 가름).
     값 128개 각각(차이의 제곱 평균 / 그 값의 분산)과 plan 출력(첫 단계 조향·속도, 출력 단위)으로.
 (3) 값마다 유지 문(update gate)의 평균 -> 그 값이 한 결정에 얼마나 새로 쓰이나
 (4) 오래 드는 값이 있으면 그것이 무엇에 따라 달라지나: 상자 있는 세션 / 치운 뒤 / 상대 차 / 최근 8 s 안의 센 접촉 (세션을 반으로 나눠 한쪽에서
     맞춘 직선 판별기를 다른 쪽에서 채점, AUC)
 (5) 그림: 한 세션(혼자·상자)의 기억 값 열지도(오래 드는 값이 위) + 바퀴끼리 같은 자리의 기억 차이(자리 = 보상이 쓰는 벽 기준 거리장, 분석에만 씀)
인자: checkpoint 기록npz [그림png]"""
import os
import sys
import warnings
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn.plan_policy import load_actor2
from f1sim import plan_steps as PS
from rec_lib import Rec, plt, along_track, probe_auc

torch.set_grad_enabled(False)
torch.manual_seed(0)
ck_path, npz = sys.argv[1], sys.argv[2]
fig_out = sys.argv[3] if len(sys.argv) > 3 else ""
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
actor = load_actor2(ck, torch.device("cpu"), check_mount=False)
R = Rec(npz); z = R.z; T, B, dt = R.T, R.B, R.dt
mm = np.load(npz[:-4] + "_mem.npz")
F = torch.tensor(mm["feat"].astype(np.float32)); H = torch.tensor(mm["mem"].astype(np.float32))
assert F.shape[:2] == (T, B) and H.shape == (T, B, actor.memory), (F.shape, H.shape, T, B)
NU = actor.memory
front = bool(actor.mem_front)
G_IN = F if front else actor.trunk(F.view(T * B, -1)).view(T, B, -1)          # what the GRU reads


def out_of(f, g, h):
    """the policy's 20 outputs (tanh of the mean) from the features and a memory state"""
    fm = actor.trunk(torch.cat([f, h], 1)) if front else g + actor.mem_out(h)
    return torch.tanh(actor.heads(fm)[0])


rm_t = float(z["rm_t"]); k_rm = int(round(rm_t / dt))
box_w = (R.sid[0] >= 0).any(-1); opp_w = R.on[:T].any(0)
hard = R.d[:, :, 3:6].sum(-1) > 0
print(f"{z['exam']} · {R.name} · 실행 {z['mode']} · 상자 {z['count']} — 세션 {B}개 x {T * dt:.0f} s (혼자 빈 {int((~box_w & ~opp_w).sum())}, 혼자 상자 "
      f"{int((box_w & ~opp_w).sum())}, 상대 빈 {int((~box_w & opp_w).sum())}, 상대 상자 {int((box_w & opp_w).sum())}), 기억 값 {NU}개, 기억이 몸통 "
      f"{'앞' if front else '뒤'}")

# (1) one step replayed
t_chk = np.arange(50, T, 97)
e = max(float((actor.gru(G_IN[t], H[t - 1]) - H[t]).abs().max(1)[0][torch.tensor(R.k0[t] < t - 1)].max()) for t in t_chk)   # rows not just restarted
print(f"(1) 한 결정 다시 돌린 기억과 기록된 기억의 가장 큰 차 {e:.4f} (값 범위 -1..1; 반정밀도 저장 오차 수준이어야 함)")
assert e < 0.02, e

# (2) wipe / swap k decisions ago, replay this session's inputs
KS = [1, 3, 5, 10, 20, 40, 80, 160]
KMAX = max(KS)
ts = np.arange(KMAX + 5, T, 13)
tb = np.array([(t, b) for t in ts for b in range(B) if R.k0[t, b] <= t - KMAX - 1])          # no restart inside the replay window
tt, bb = torch.tensor(tb[:, 0]), torch.tensor(tb[:, 1])
perm = torch.randperm(B)
bb_sw = perm[bb]
same = bb_sw == bb
h_true = H[tt, bb]; f_now = F[tt, bb]; g_now = G_IN[tt, bb]
o_true = out_of(f_now, g_now, h_true)
var_u = h_true.var(0).clamp_min(1e-6)
o_sd = o_true.std(0)
i_d, i_v = 0, PS.N_STEP
print(f"(2) 표본 {len(tb)}개 (세션 {B}개 x 시각 {len(ts)}개). 실제 기억 값의 표준편차 중앙 {float(var_u.sqrt().median()):.2f}; plan 출력 표준편차: 첫 조향 "
      f"{float(o_sd[i_d]):.2f}, 첫 속도 {float(o_sd[i_v]):.2f} (출력 단위 -1..1)")
print("    k 결정 전 |  지움: 기억 차(분산 대비) 평균 · 아직 5 % 넘게 다른 값 수 · 첫 조향 차 · 첫 속도 차 |  바꿈: 같은 네 칸")
NERR = {}
for mode in ("wipe", "swap"):
    for k in KS:
        if mode == "wipe":
            h = torch.zeros(len(tb), NU)
        else:
            h = H[tt - k, bb_sw].clone()
        for j in range(k - 1, -1, -1):
            h = actor.gru(G_IN[tt - j, bb], h)
        ne = ((h - h_true) ** 2).mean(0) / var_u
        o = out_of(f_now, g_now, h)
        dd = (o - o_true).abs()
        NERR[(mode, k)] = (ne.numpy(), float(dd[:, i_d].mean()), float(dd[:, i_v].mean()),
                           float(dd[:, :PS.N_STEP].mean()), float(dd[:, PS.N_STEP:].mean()))
for k in KS:
    row = f"    {k:4d} ({k * dt:4.1f} s) |"
    for mode in ("wipe", "swap"):
        ne, dd0, dv0, dda, dva = NERR[(mode, k)]
        row += f"  {ne.mean():7.4f} · {int((ne > 0.05).sum()):3d}개 · {dd0:.4f} · {dv0:.4f} |"
    print(row)
print(f"    (참고: 첫 조향·속도 출력이 표본끼리 다른 정도 = 표준편차 {float(o_sd[i_d]):.2f} / {float(o_sd[i_v]):.2f}. 바꿈에서 같은 세션끼리 바뀐 표본 "
      f"{int(same.sum())}개)")

# how long each value holds: the first k from which the swapped history leaves under 5 % of its variance
hold = np.full(NU, KS[-1] * 2)
for u in range(NU):
    for k in KS:
        if all(NERR[("swap", k2)][0][u] < 0.05 for k2 in KS if k2 >= k):
            hold[u] = k; break
edges = [(1, "0.1 s"), (3, "0.3 s"), (5, "0.5 s"), (10, "1 s"), (20, "2 s"), (40, "4 s"), (80, "8 s"), (160, "16 s"), (320, "16 s 넘게")]
print("    값 128개가 '다른 세션의 기억으로 바꿔도 그 뒤 같아지는 데 걸리는 시간'(차이가 분산의 5 % 아래로): "
      + " · ".join(f"{name} {int((hold == k).sum())}개" for k, name in edges if (hold == k).any()))

# (3) the update gate
W_ih, W_hh, b_ih, b_hh = actor.gru.weight_ih, actor.gru.weight_hh, actor.gru.bias_ih, actor.gru.bias_hh
zs = []
for t in range(5, T, 7):
    gi = G_IN[t] @ W_ih.T + b_ih; gh = H[t - 1] @ W_hh.T + b_hh
    zs.append(torch.sigmoid(gi[:, NU:2 * NU] + gh[:, NU:2 * NU]))
zbar = torch.cat(zs, 0).mean(0).numpy()
print(f"(3) 유지 문 평균(1 = 그대로 둠, 0 = 매 결정 새로 씀): 중앙 {np.median(zbar):.2f}, 값 128개 중 0.9 넘음(1 s 넘게 둠) {int((zbar > 0.9).sum())}개, "
      f"0.975 넘음(4 s) {int((zbar > 0.975).sum())}개, 0.9875 넘음(8 s) {int((zbar > 0.9875).sum())}개")


# (4) what the long-held values tell
Hn = H.numpy(); Fn = F.numpy()
n8 = int(round(8.0 / dt))
hit8 = np.zeros((T, B), bool)                       # a hard contact within the last 8 s
last = np.full(B, -10 ** 6)
for t in range(T):
    last = np.where(hard[t], t, last)
    hit8[t] = (t - last) <= n8
tq = np.arange(100, T, 9)
before = tq[tq < k_rm - 10]; after = tq[tq >= k_rm + 100]
sets = {"오래 드는 값(8 s 넘게)": hold > 80, "4 s 넘게": hold > 40, "2 s 넘게": hold > 20, "값 전부": np.ones(NU, bool)}
tasks = []
gb = np.arange(B)
if len(before):
    tasks.append(("상자 있는 세션인가(치우기 전; 세션끼리)", before, np.ones(B, bool), lambda t: np.broadcast_to(box_w, (len(t), B))))
    tasks.append(("상대 차가 있는 세션인가(치우기 전)", before, np.ones(B, bool), lambda t: np.broadcast_to(opp_w, (len(t), B))))
    tasks.append(("최근 8 s 안에 세게 닿았나(상자 세션, 치우기 전)", before, box_w, lambda t: hit8[t]))
if len(after):
    tasks.append(("상자 세션이었나(치운 지 10 s 넘은 뒤; 세션끼리)", after, np.ones(B, bool), lambda t: np.broadcast_to(box_w, (len(t), B))))
print("(4) 기억 값들에서 직선 판별기로 맞히기 (AUC, 0.5 = 모름, 1 = 다 맞힘; 세션을 반으로 나눠 한쪽에서 맞추고 다른 쪽에서 채점)")
for name, tsel, bsel, yfun in tasks:
    bi = np.nonzero(bsel)[0]
    y = np.asarray(yfun(tsel))[:, bi].reshape(-1)
    g = np.broadcast_to(bi[None], (len(tsel), len(bi))).reshape(-1)
    row = f"    {name}: 맞는 쪽 {100 * y.mean():.0f} % ·"
    for sname, um in sets.items():
        X = Hn[tsel][:, bi][:, :, um].reshape(len(y), -1)
        row += f" {sname}({int(um.sum())}개) {probe_auc(X, y, g)[0]:.2f} ·"
    Xf = Fn[tsel][:, bi].reshape(len(y), -1)
    row += f" 지금 입력 특징만 {probe_auc(Xf, y, g)[0]:.2f}"
    print(row)

# (5) picture
if fig_out:
    s_pos, lap_i, L = along_track(z, T)                          # along the track (walls-only distance field), analysis only
    cand = np.nonzero(box_w & ~opp_w & (R.rs.sum(0) == 0))[0]
    assert len(cand), "혼자·상자 세션이 없습니다"
    nh = np.array([int(hard[:k_rm, b].sum()) for b in cand]); nl = np.array([lap_i[-1, b] for b in cand])
    pick = cand[np.argsort(-(nh > 0).astype(int) * 100 - nl)[:2]]          # sessions with a hard hit first, then most laps
    b, b2 = int(pick[0]), int(pick[-1] if len(pick) > 1 and pick[-1] != pick[0] else (pick[0] + 1) % B)
    order = np.argsort(-hold - zbar)                             # long-held values on top
    SB = 1.0; nb = int(np.ceil(L / SB))                          # one decision moves 0.7-0.8 m at race pace
    solo = np.nonzero(~opp_w & (R.rs.sum(0) == 0))[0]
    lap_s = np.concatenate([np.diff(np.nonzero(np.diff(lap_i[k_rm:, s]) > 0)[0]) * dt for s in solo])
    lap_med = float(np.median(lap_s)) if len(lap_s) else float("nan")

    def lap_table(s):
        """rows = laps, columns = along-track bins: the mean memory vector there"""
        n_l = lap_i[-1, s] + 1
        tab = np.full((n_l, nb, NU), np.nan)
        jb = np.minimum((s_pos[:, s] / SB).astype(int), nb - 1)
        for l in range(n_l):
            for j in range(nb):
                m = (lap_i[:, s] == l) & (jb == j)
                if m.any():
                    tab[l, j] = Hn[m, s].mean(0)
        return tab

    # the floor: lap-to-lap difference at the same place in empty solo sessions (laps 2..)
    fl = []
    for s in np.nonzero(~box_w & ~opp_w & (R.rs.sum(0) == 0))[0][:12]:
        tab = lap_table(s)
        if tab.shape[0] > 3:
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fl.append(np.sqrt(np.nanmean((tab[2:] - tab[1:-1]) ** 2, -1)).reshape(-1))
    floor = float(np.nanmedian(np.concatenate(fl))) if fl else 1.0

    # every 10 s the memory is replaced by another session's, then this session's recorded inputs are replayed
    n10 = int(round(10.0 / dt))
    Dsw = np.full((T, NU), np.nan)
    for t0 in range(n10, T, n10):
        h = H[t0 - 1, b2][None].clone()
        for t in range(t0, min(t0 + n10, T)):
            h = actor.gru(G_IN[t, b][None], h)
            Dsw[t] = (h[0] - H[t, b]).abs().numpy()

    INK, BLUE, ORANGE = "#52514e", "#2a78d6", "#eb6834"
    fig, axs = plt.subplots(2, 2, figsize=(22, 13.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax, a2, a3, a4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    cross = np.nonzero(np.diff(lap_i[:, b]) > 0)[0] + 1

    def marks(a):
        for c in cross:
            a.axvline(c * dt, color="k", lw=0.8, alpha=0.6)
        a.axvline(rm_t, color="#0ca30c", lw=2.5)
        for t in np.nonzero(hard[:, b])[0]:
            a.plot([t * dt], [-1.5], "v", color="#e34948", ms=10, clip_on=False)

    im = ax.imshow(Hn[:, b][:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, extent=(0, T * dt, NU, 0), interpolation="nearest")
    marks(ax)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="기억 값 (-1..1)")
    ax.set_xlabel("세션 시간 (s)"); ax.set_ylabel(f"기억 값 {NU}개 (위 = 오래 드는 값)")
    ax.set_title(f"① 기록된 기억 값 전부 — 세션 {b} (혼자, 상자 2개)\n검은 세로선 = 결승선 지남, 초록 = 상자 치움({rm_t:g} s), 빨간 세모 = 세게 닿음", fontsize=12)

    tab = lap_table(b)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D = np.sqrt(np.nanmean((tab[1:] - tab[:-1]) ** 2, -1)) / floor
    cm = plt.get_cmap("magma").copy(); cm.set_bad("#d9d8d2")
    im2 = a2.imshow(np.ma.masked_invalid(D), aspect="auto", cmap=cm, vmin=0, vmax=6, extent=(0, L, D.shape[0] + 0.5, 0.5), interpolation="nearest")
    fig.colorbar(im2, ax=a2, fraction=0.03, pad=0.01, label="빈 트랙에서의 바퀴 간 차이의 몇 배")
    xy = R.st[:T, b, :2].astype(float)
    for j in np.nonzero(R.sid[0, b] >= 0)[0]:                    # the car's place when nearest to each live box before the removal
        dd = np.linalg.norm(xy[:max(k_rm, 1)] - R.pose[0, b, j, :2].astype(float)[None], axis=1)
        if dd.min() < 3.0:
            a2.axvline(s_pos[int(dd.argmin()), b], color="#00d0ff", lw=1.8, ls="--")
    kr = min(k_rm, T - 1)
    a2.plot([s_pos[kr, b]], [lap_i[kr, b]], "*", color="#30ff60", ms=18, mec="k")
    for t in np.nonzero(hard[:, b])[0]:
        a2.plot([s_pos[t, b]], [lap_i[t, b]], "v", color="#ff5050", ms=10, mec="k")
    a2.set_xlabel("트랙을 따라 자리 (m; 분석에서 줄 맞추는 데만 씀)"); a2.set_ylabel("바퀴 (바로 앞 바퀴와 견줌)")
    a2.set_title("② 같은 자리에서 바로 앞 바퀴의 기억과 얼마나 다른가 (같은 세션)\n하늘색 점선 = 상자 자리, 초록 별 = 상자를 치운 순간, 빨간 세모 = 세게 닿은 자리", fontsize=12)

    cm3 = plt.get_cmap("Blues").copy(); cm3.set_bad("#eceae4")
    im3 = a3.imshow(np.ma.masked_invalid(Dsw[:, order].T), aspect="auto", cmap=cm3, vmin=0, vmax=0.6, extent=(0, T * dt, NU, 0), interpolation="nearest")
    for t0 in range(n10, T, n10):
        a3.axvline(t0 * dt, color="#eb6834", lw=1.6)
    fig.colorbar(im3, ax=a3, fraction=0.02, pad=0.01, label="실제 기억과의 차 (절대값)")
    a3.set_xlabel("세션 시간 (s)"); a3.set_ylabel(f"기억 값 {NU}개 (①과 같은 순서)")
    a3.set_title(f"③ 같은 세션에서 10 s마다(주황 선) 기억을 다른 세션(세션 {b2}) 것으로 통째로 바꿔 넣고, 그 뒤 실제와 얼마나 다른지\n"
                 "흰 곳 = 실제 기억과 같아짐. 바꾼 흔적이 몇 초 안에 사라짐 = 그보다 오래된 일은 기억에 없음", fontsize=12)

    xs = np.array(KS) * dt
    for mode, col, lab in (("swap", BLUE, "다른 세션의 기억으로 바꿈"), ("wipe", ORANGE, "0으로 지움")):
        ys = [int((NERR[(mode, k)][0] > 0.05).sum()) for k in KS]
        a4.plot(xs, ys, "-o", color=col, lw=2, ms=8, label=lab)
        for x, y in zip(xs, ys):
            if mode == "swap":
                a4.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=10, color=INK)
    if np.isfinite(lap_med):
        a4.axvline(lap_med, color=INK, lw=1.5, ls="--")
        a4.text(lap_med * 1.04, NU * 0.86, f"한 바퀴\n(중앙 {lap_med:.1f} s)", fontsize=11, color=INK, va="top")
    a4.set_xscale("log"); a4.set_xticks(xs); a4.set_xticklabels([f"{x:g}" for x in xs]); a4.minorticks_off()
    a4.set_ylim(-6, NU + 4); a4.grid(alpha=0.25); a4.legend(fontsize=11, loc="upper right", frameon=False)
    for sp in ("top", "right"):
        a4.spines[sp].set_visible(False)
    a4.set_xlabel("기억을 건드린 뒤 지난 시간 (s)"); a4.set_ylabel(f"아직 실제와 다른 기억 값 수 ({NU}개 중)")
    k4 = 40 if 40 in KS else KS[-2]
    a4.set_title(f"④ 기억을 건드린 뒤 몇 초면 다시 같아지나 (세션 {B}개 x 시각 {len(ts)}개)\n{k4 * dt:g} s 뒤 plan 출력 차: 첫 조향 "
                 f"{NERR[('swap', k4)][1]:.3f}, 첫 속도 {NERR[('swap', k4)][2]:.3f} (출력이 표본끼리 다른 정도 {float(o_sd[i_d]):.2f} / {float(o_sd[i_v]):.2f})", fontsize=12)
    fig.suptitle(f"{z['exam']} · {R.name} · plan대로 정확히 · 상자 {z['count']} — GRU 기억은 얼마나 오래된 일을 들고 있나", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(fig_out, dpi=62); print("그림", fig_out)
