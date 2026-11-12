"""GRU 기억이 '지금 안 보이는 상자'를 들고 있나 (11-12, 사용자: "gru 자체에서 바로 써먹을 수 있는게 있는지").
시험 기록(exam_record.py, F1SIM_EXAM_SAVE_MEM=1; 상자 2개 배치)의 상자 세션, 상자를 치우기 전에서 세 가지 순간을 모음:
 (보임)   상자 중심이 스캔에 보이기 시작하고 0.5 s 뒤                       -> 재는 방법이 되는지 확인용(당연히 구분돼야 함)
 (접근)   상자가 보이기 시작하기 0.3 s / 1.0 s 전, 최근 2 s 동안 어떤 상자도 안 보임 -> 첫 통과(알 방법이 없음 = 0.5여야 함) / 둘째 통과부터
 (지난 뒤) 상자가 시야에서 사라지고 0.5 / 1 / 2 / 4 s 뒤, 그동안 다른 상자 안 보임
순간마다 짝: 다른 상자 세션의 같은 자리(±0.3 m), 같은 바퀴 구분(첫 바퀴 / 그 뒤), 상대 차 유무 같음, 최근 2 s 동안 상자 안 보임,
그 앞(또는 뒤) 구간에 상자 없음. GRU_AHEAD_STRICT=1 이면 더 엄하게: 짝을 같은 반(세션 번호 짝/홀)에서만 골라 판별기의 두 반이 세션을
나눠 갖게 하고, '(접근)'은 마지막으로 상자를 본 뒤 지난 시간도 0.5 s 안으로 맞춤(막 지나온 다른 상자의 흔적이 섞이지 않게) -- 표본이 크게 줆.
판별기 AUC는 엄한 방식에서만 냄(아니면 짝과 상자 쪽이 두 반에 갈려 자리->답 지름길이 생겨 값이 치우침). 자리는 보상이 쓰는 벽 기준 거리장으로 맞춤 -- 분석에서 줄을 맞추는 데만 씀.
재는 것: (1) 짝들의 평균에서 떨어진 거리(값마다 표준편차로 나눔): 상자 쪽 vs 짝 하나를 빼서 같은 식으로 -> 그 둘이 구분되는 AUC, 기억 값으로 /
지금 입력 특징만으로 (2) 직선 판별기(세션 반으로 맞추고 다른 반에서 채점; 짝은 제 상자 쪽과 같은 반)의 AUC (3) 기억만 짝의 것으로 바꿔 넣을 때 plan 출력이 변하는 정도: 상자 쪽 vs 짝끼리.
'접근·둘째 통과부터'의 AUC가 '접근·첫 통과'와 '지금 입력'보다 뚜렷이 높으면 = 지난 바퀴의 상자를 기억이 들고 온 것.
인자: checkpoint 기록npz [그림.png]   (옆에 _mem.npz 와 exam_boxes.py의 _boxrows.npy)"""
import os
import sys
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn.plan_policy import load_actor2
from f1sim import plan_steps as PS
from rec_lib import Rec, along_track, auc, auc_se, probe_auc

torch.set_grad_enabled(False)
rng = np.random.default_rng(0)
STRICT = os.environ.get("GRU_AHEAD_STRICT", "") == "1"       # pairs from the same half of the sessions + matched time since a box was in view
ck_path, npz = sys.argv[1], sys.argv[2]
actor = load_actor2(torch.load(ck_path, map_location="cpu", weights_only=False), torch.device("cpu"), check_mount=False)
assert actor.memory and actor.mem_front, "기억이 몸통 앞에 있는 policy용"
R = Rec(npz); z = R.z; T, B, dt = R.T, R.B, R.dt
mm = np.load(npz[:-4] + "_mem.npz"); Hn = mm["mem"].astype(np.float32); Fn = mm["feat"].astype(np.float32)
rows = np.load(npz[:-4] + "_boxrows.npy")
IB, IJ, IT, IN, IGAP = 0, 1, 2, 3, 4
GAP = 0.6                                              # boxes whose centre is this close to the path driven without boxes
k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
st = R.st.astype(float); xy = st[:T, :, :2]; yaw = st[:T, :, 2]; V = np.abs(st[:T, :, 3])
sid, pose = R.sid, R.pose.astype(float)
box_w = (sid[0] >= 0).any(-1); opp_w = R.on[:T].any(0)
mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
FOV, RMAX = np.radians(135.0), 10.0
s_pos, lap_i, L = along_track(z, T)
P = sid.shape[2]
n2 = int(round(2.0 / dt)); n3 = int(round(3.0 / dt))


def sees(tk, t, b, c):
    """is the box centre c in the scan's direct view at decision t (as exam_boxes.py: field of view, range, no wall on the way)"""
    sx, sy = xy[t, b, 0] + mount * np.cos(yaw[t, b]), xy[t, b, 1] + mount * np.sin(yaw[t, b])
    dx, dy = c[0] - sx, c[1] - sy; dist = np.hypot(dx, dy)
    a = (np.arctan2(dy, dx) - yaw[t, b] + np.pi) % (2 * np.pi) - np.pi
    if dist > RMAX or abs(a) > FOV:
        return False
    n = max(int((dist - 0.25) / 0.05), 1)
    f = np.linspace(0.0, max(dist - 0.25, 0.0) / max(dist, 1e-6), n)
    return bool((R.wall_dist(tk, np.stack([sx + f * dx, sy + f * dy], 1)) > 0.0).all())


vis = np.zeros((k_rm, B, P), bool)
box_s = np.full((B, P), np.nan)                        # where along the track each live box sits (the car's place when nearest)
for b in np.nonzero(box_w)[0]:
    tk = int(R.tid[0, b])
    for j in np.nonzero(sid[0, b] >= 0)[0]:
        d = np.linalg.norm(pose[:k_rm, b, j, :2] - xy[:k_rm, b], axis=1)
        box_s[b, j] = s_pos[int(d.argmin()), b]
        for t in np.nonzero(d < RMAX + 0.5)[0]:
            vis[t, b, j] = sees(tk, t, b, pose[t, b, j, :2])


def since(mask):
    """(T', B) decisions since the mask was last true (large before the first)"""
    out = np.zeros(mask.shape, int); cur = np.full(mask.shape[1], 10 ** 6)
    for t in range(mask.shape[0]):
        cur = np.where(mask[t], 0, cur + 1); out[t] = cur
    return out


since_vis = since(vis.any(2))
since_hard = since(R.d[:, :, 3:6].sum(-1) > 0)[:k_rm]
with np.errstate(invalid="ignore"):
    ahead = np.nanmin(np.where(np.isfinite(box_s)[None], np.mod(box_s[None] - s_pos[:k_rm, :, None], L), np.inf), axis=2)
    behind = np.nanmin(np.where(np.isfinite(box_s)[None], np.mod(s_pos[:k_rm, :, None] - box_s[None], L), np.inf), axis=2)
tgrid = np.arange(k_rm)[:, None]
drive = box_w[None] & (V[:k_rm] > 2.0) & (since_hard > n3) & (tgrid >= n3)        # normal driving in a box session before the removal
lapc = lap_i[:k_rm] >= 1
SBIN = 0.5; nb = int(np.ceil(L / SBIN))
qt, qb = np.nonzero(drive & (since_vis >= n2))                                    # the pool the pairs come from
qbin = np.minimum((s_pos[qt, qb] / SBIN).astype(int), nb - 1)
by_bin = [np.nonzero(qbin == i)[0] for i in range(nb)]


def pairs(t, b, need_ahead=None, need_behind=None, min_since=n2, match_since=False, n=4):
    """up to n (t', b') from other sessions at the same place: one per session (the nearest in place), sessions at random.
    Only sessions of the same half (even / odd number) as b, so the read-out's two halves share no session."""
    i = min(int(s_pos[t, b] / SBIN), nb - 1)
    cand = np.concatenate([by_bin[(i + k) % nb] for k in (-1, 0, 1)])
    ct, cb = qt[cand], qb[cand]
    ds = np.abs(np.mod(s_pos[ct, cb] - s_pos[t, b] + 0.5 * L, L) - 0.5 * L)
    m = (cb != b) & ((cb % 2 == b % 2) | (not STRICT)) & (ds <= 0.3) & (lapc[ct, cb] == lapc[t, b]) & (opp_w[cb] == opp_w[b]) & (since_vis[ct, cb] >= min_since)
    if match_since and STRICT:                                    # the same time since any box was last in view (within 0.5 s, capped at 6 s)
        m &= np.abs(np.minimum(since_vis[ct, cb], 60) - min(int(since_vis[t, b]), 60)) <= 5
    if need_ahead is not None:
        m &= ahead[ct, cb] > need_ahead
    if need_behind is not None:
        m &= behind[ct, cb] > need_behind
    k = np.nonzero(m)[0]
    if len(k) == 0:
        return []
    k = k[np.argsort(ds[k])]
    _, first = np.unique(cb[k], return_index=True)
    k = k[first]
    k = k[rng.permutation(len(k))[:n]]
    return [(int(ct[i_]), int(cb[i_])) for i_ in k]


G = {}                                                                             # group name -> samples (t, b, pairs)


def add(name, t, b, **kw):
    c = pairs(t, b, **kw)
    if len(c) >= 3:
        G.setdefault(name, []).append((t, b, c))


n_row = 0
for r in rows:
    b, j, tp, npass = int(r[IB]), int(r[IJ]), int(r[IT]), int(r[IN])
    if r[IGAP] >= GAP or tp >= k_rm:
        continue
    v = vis[:, b, j]
    near = np.nonzero(v[max(tp - 10, 0): min(tp + 5, k_rm)])[0]
    if len(near) == 0:
        continue                                                                   # never in direct view around this pass
    n_row += 1
    ts = te = max(tp - 10, 0) + int(near[-1])
    miss = 0
    while ts > 0 and miss <= 2:                                                    # back to where this visible stretch began
        if v[ts - 1]:
            ts -= 1; miss = 0
        else:
            k = ts - 1; miss = 0
            while k >= 0 and not v[k] and miss <= 2:
                k -= 1; miss += 1
            if miss <= 2 and k >= 0:
                ts = k; miss = 0
            else:
                break
    while te + 1 < k_rm and v[te + 1]:
        te += 1
    cls = "첫 통과" if (npass == 1 and not v[:max(ts - 1, 0)].any()) else ("둘째 통과부터" if npass >= 2 else None)
    t = ts + 5                                                                     # in view for 0.5 s
    if t < k_rm and v[t] and drive[t, b]:
        add("(보임) 보이기 시작하고 0.5 s 뒤", t, b, need_ahead=float(np.mod(box_s[b, j] - s_pos[t, b], L)) + 6.0)
    for m_, lab in ((3, "0.3 s"), (10, "1.0 s")):
        t = ts - m_
        if cls and t >= n3 and drive[t, b] and since_vis[t, b] >= n2:
            add(f"(접근) 보이기 {lab} 전 · {cls}", t, b, need_ahead=float(np.mod(box_s[b, j] - s_pos[t, b], L)) + 6.0, match_since=True)
    for m_ in (5, 10, 20, 40):
        t = te + m_
        if t < k_rm and drive[t, b] and since_vis[t, b] == m_:
            add(f"(지난 뒤) 시야에서 사라지고 {m_ * dt:.1f} s 뒤", t, b,
                need_behind=float(np.mod(s_pos[t, b] - box_s[b, j], L)) + 6.0, min_since=max(n2, m_ + 10))


def out_of(f, h):
    return torch.tanh(actor.heads(actor.trunk(torch.cat([torch.tensor(f), torch.tensor(h)], 1)))[0]).numpy()


print(f"{z['exam']} · {R.name} · 실행 {z['mode']} · 상자 {z['count']}{' · 엄한 짝' if STRICT else ''} — 상자 세션 {int(box_w.sum())}개(혼자 {int((box_w & ~opp_w).sum())}), 치우기 전 "
      f"{k_rm * dt:.0f} s. 길에서 {GAP} m 안의 상자를 직접 보고 지난 통과 {n_row}번. 보통 주행 결정 가운데 '최근 2 s 동안 상자가 안 보인' 비율 "
      f"{100 * (drive & (since_vis >= n2)).sum() / max(drive.sum(), 1):.0f} %")
sd_h = Hn[::7].reshape(-1, Hn.shape[2]).std(0) + 1e-6; sd_f = Fn[::7].reshape(-1, Fn.shape[2]).std(0) + 1e-6


def apart(X, S, sd):
    """AUC of 'the box side is farther from the mean of its pairs (all but the first) than the first pair is'; paired, so the place cancels"""
    tb_ = np.array([(t, b) for t, b, _ in S]); c0_ = np.array([c[0] for _, _, c in S])
    rest = np.stack([np.mean([X[t, b] for t, b in c[1:]], 0) for _, _, c in S])
    dp = np.sqrt((((X[tb_[:, 0], tb_[:, 1]] - rest) / sd) ** 2).mean(1)); dc = np.sqrt((((X[c0_[:, 0], c0_[:, 1]] - rest) / sd) ** 2).mean(1))
    a = auc(np.concatenate([dp, dc]), np.concatenate([np.ones(len(S), bool), np.zeros(len(S), bool)]))
    return a, auc_se(a, len(S), len(S)), float(np.median(dp)), float(np.median(dc))


print("순간 | 표본(세션) | 짝들의 평균에서 떨어진 정도로 구분(AUC): 기억 값 (상자 쪽 · 짝 거리) · 지금 입력 | 판별기 AUC: 기억 값 · 지금 입력 | "
      "기억만 짝 것으로 바꾸면 plan 변화(첫 조향·첫 속도 출력): 상자 쪽 · 짝끼리")
ORDER = [f"(접근) 보이기 {lab} 전 · {c}" for c in ("첫 통과", "둘째 통과부터") for lab in ("1.0 s", "0.3 s")] + ["(보임) 보이기 시작하고 0.5 s 뒤"] \
    + [f"(지난 뒤) 시야에서 사라지고 {m_ * dt:.1f} s 뒤" for m_ in (5, 10, 20, 40)]
RES = []
for name in ORDER:
    S = G.get(name, [])
    if len(S) < 16:
        print(f"{name} | 표본 {len(S)}개: 부족"); continue
    tb = np.array([(t, b) for t, b, _ in S])
    hp, fp = Hn[tb[:, 0], tb[:, 1]], Fn[tb[:, 0], tb[:, 1]]
    c0 = np.array([c[0] for _, _, c in S]); c1 = np.array([c[1] for _, _, c in S])
    hc0, hc1, fc1 = Hn[c0[:, 0], c0[:, 1]], Hn[c1[:, 0], c1[:, 1]], Fn[c1[:, 0], c1[:, 1]]
    a_dh, e_dh, m_p, m_c = apart(Hn, S, sd_h); a_df, e_df, _, _ = apart(Fn, S, sd_f)
    ca = np.array([c_ for _, _, c in S for c_ in c])
    y = np.concatenate([np.ones(len(S), bool), np.zeros(len(ca), bool)])
    g = np.concatenate([tb[:, 1], ca[:, 1]])                                       # pairs come from the box side's own half of the sessions
    a_h, e_h = probe_auc(np.concatenate([hp, Hn[ca[:, 0], ca[:, 1]]]), y, g) if STRICT else (np.nan, np.nan)
    a_f, e_f = probe_auc(np.concatenate([fp, Fn[ca[:, 0], ca[:, 1]]]), y, g) if STRICT else (np.nan, np.nan)
    o_p, o_ps = out_of(fp, hp), out_of(fp, hc0)                                    # the box side with its own / a pair's memory
    o_c, o_cs = out_of(fc1, hc1), out_of(fc1, hc0)                                 # a pair with its own / another pair's memory
    i_v = PS.N_STEP
    RES.append((name, len(S), a_dh, e_dh, a_df, e_df, a_h, e_h, a_f, e_f))
    pa = f"{a_h:.2f}±{e_h:.2f} · {a_f:.2f}±{e_f:.2f}" if STRICT else "- (엄한 짝에서만)"
    print(f"{name} | {len(S)}개({len(np.unique(tb[:, 1]))}) | {a_dh:.2f}±{e_dh:.2f} ({m_p:.2f} · {m_c:.2f}) · {a_df:.2f}±{e_df:.2f} | "
          f"{pa} | 조향 {np.abs(o_p - o_ps)[:, 0].mean():.3f} · {np.abs(o_c - o_cs)[:, 0].mean():.3f}, "
          f"속도 {np.abs(o_p - o_ps)[:, i_v].mean():.3f} · {np.abs(o_c - o_cs)[:, i_v].mean():.3f}")
print("읽는 법: AUC 0.5 = 구분 못 함, 1 = 다 구분. ± = 표준오차. '(보임)'은 방법 확인용. '(접근)·첫 통과'는 알 방법이 없으므로 0.5 근처여야 하고, "
      "그보다 높으면 그만큼은 새는 것(상자 모서리가 중심보다 먼저 보임 등). '(접근)·둘째 통과부터'가 그 둘('첫 통과', '지금 입력')보다 뚜렷이 높아야 "
      "지난 바퀴의 상자를 기억이 들고 온 것. '(지난 뒤)'는 흔적이 남는 시간.")

if len(sys.argv) > 3 and RES:
    from rec_lib import plt
    INK, BLUE, ORANGE = "#52514e", "#2a78d6", "#eb6834"
    SHORT = {"(접근) 보이기 1.0 s 전 · 첫 통과": "보이기 1.0 s 전\n(처음 만나는 상자)", "(접근) 보이기 0.3 s 전 · 첫 통과": "보이기 0.3 s 전\n(처음 만나는 상자)",
             "(접근) 보이기 1.0 s 전 · 둘째 통과부터": "보이기 1.0 s 전\n(지난 바퀴에 지나간 상자)", "(접근) 보이기 0.3 s 전 · 둘째 통과부터": "보이기 0.3 s 전\n(지난 바퀴에 지나간 상자)",
             "(보임) 보이기 시작하고 0.5 s 뒤": "보이기 시작하고\n0.5 s 뒤"}
    fig, ax = plt.subplots(figsize=(2.3 * len(RES) + 3, 6.2))
    x = np.arange(len(RES)); w = 0.36
    for k, (col, lab, ia) in enumerate(((BLUE, "GRU 기억 값 128개", 2), (ORANGE, "지금 입력 특징만 (기억 없이)", 4))):
        v = np.array([r[ia] for r in RES]); e = np.array([r[ia + 1] for r in RES])
        ax.bar(x + (k - 0.5) * (w + 0.03), v - 0.5, w, bottom=0.5, color=col, label=lab, yerr=e, ecolor=INK, capsize=3, error_kw=dict(lw=1.2))
        for xi, vi, ei in zip(x + (k - 0.5) * (w + 0.03), v, e):
            ax.text(xi, max(vi, 0.5) + ei + 0.012, f"{vi:.2f}", ha="center", fontsize=10, color=INK)
    ax.axhline(0.5, color=INK, lw=1.2)
    ax.text(1.005, 0.5, "0.5 =\n구분 못 함", fontsize=10, color=INK, ha="left", va="center", transform=ax.get_yaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels([SHORT.get(r[0], r[0].replace("(지난 뒤) 시야에서 ", "").replace("사라지고", "시야에서 사라지고\n")) + f"\n표본 {r[1]}개" for r in RES], fontsize=10)
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("상자 없는 같은 자리와 구분되는 정도 (AUC; 1 = 다 구분)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title(f"{z['exam']} · {R.name} · 상자 {z['count']} — 상자가 안 보이는 순간, GRU 기억이 그 상자를 들고 있나\n"
                 "왼쪽 = 다가가는 중(아직 안 보임) -> 가운데 = 보임 -> 오른쪽 = 지나간 뒤(다시 안 보임). 막대 끝 선 = 표준오차", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.96, 1)); fig.savefig(sys.argv[3], dpi=75); print("그림", sys.argv[3])
