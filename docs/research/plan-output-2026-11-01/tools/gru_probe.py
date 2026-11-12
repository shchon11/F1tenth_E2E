"""같은 크기의 GRU를 같은 입력으로 '따로' 가르치면 지난 바퀴의 상자를 들고 올 수 있나 (11-12; 사용자: "기억 네트워크 설계자체를 다시 생각해봐야 할 부분인가").
policy 학습에 넣는 것이 아니라 기록으로 하는 측정임. 시험 기록(exam_record.py, F1SIM_EXAM_SAVE_MEM=1)에 남은 'policy의 GRU가 읽은 입력'(LiDAR 인코딩 256 +
상태 36 = 292)을 그대로 새 GRU(값 128개, policy와 같은 크기·같은 기본 초기값)에 넣고 세 가지를 따로 가르침:
 (가) 직접: "전에 지나간 상자가 1 s 안에 시야에 들어온다"(지금은 안 보임)를 맞히기.
 (나) 앞으로 3 s 동안 나아가는 거리 맞히기 (value가 배우는 것의 짧은 꼴).
 (다) 학습의 value와 같은 꼴: 결정마다 0.995로 줄여 더한 앞으로의 진행(바퀴) - 접촉 벌(0.5 s + 잃은 속도 x 1 s/(m/s), 한 바퀴 7.5 s로 나눔), 30 s까지.
비교하는 네트워크: 기억 없는 것(지금 입력만) / 최근 3 s만 보는 GRU(3 s 조각으로 학습·채점) / 세션 처음부터 보는 GRU(기본 초기값) / 같은 것에 update gate
bias만 1~30 s에 퍼지게 초기화. 세션을 반으로 나눠 한 반으로 가르치고(그 안의 20 %는 멈출 때를 고르는 데 씀) 다른 반에서 채점, 반을 바꿔 한 번 더.
채점하는 순간: 길 위(지나간 길에서 0.6 m 안) 상자의 둘째 통과부터, 그 상자가 시야에 들어오기 0.5 s / 1.0 s 전. 그 상자는 최근 3 s 동안 안 보였고 어떤 상자도
최근 0.5 s 동안 안 보임. 짝 두 가지와 견줌(같은 반 안에서만):
 '어디' = 다른 상자 세션의 같은 자리(±0.3 m; 그 앞 구간에 상자 없음, 둘째 바퀴부터, 상대 차 유무 같음, 최근 0.5 s 상자 안 보임, 마지막으로 상자를 본 뒤
         지난 시간이 ±0.5 s 안으로 같음 -- 안 맞추면 짝 쪽이 '방금 상자를 지난' 경우가 많아 기억 없는 네트워크도 0.5 아래로 치우침) -> 상자가 '어디' 있는지 알아야 구분됨
 '있다' = 상자가 한 번도 없던 세션의 같은 자리·같은 시각(±4 s; 같은 반, 상대 차 유무 같음) -> '이 세션에 상자가 있다'만 들고 있어도 구분됨
         (처음에는 같은 세션에서 상자를 치운 뒤의 같은 자리와 견줬는데, 시간만 세는 네트워크도 치우기 전과 뒤를 가를 수 있어 버림)
AUC = 상자 쪽 점수가 짝보다 높은 비율(0.5 = 구분 못 함). (가)는 네트워크의 출력이 점수. (나)(다)는 예측한 값이 낮을수록 높은 점수(상자를 알면 덜 나아간다고
예측해야 함)와, 기억 값 128개에서 직선으로 읽은 점수(가르친 반의 순간들로 맞추고 다른 반에서 채점) 둘 다. 기준 줄: policy가 실제로 들고 있던 기억과 지금 입력을 같은 식으로 읽음.
자리는 보상이 쓰는 벽 기준 거리장으로 맞춤 -- 분석에서 줄을 맞추는 데만 씀(policy 입력 아님).
인자: 결과.txt 기록npz [기록npz ...]   (기록 옆에 _mem.npz, _boxrows.npy; 결과.txt 옆에 줄들을 .npy로도 남김)
      그림.png 줄.npy [줄.npy ...]   (앞서 남긴 줄들을 모아 그림만 그림)
환경변수: GRU_PROBE_ARMS(기본 ABC = 가나다) GRU_PROBE_SEEDS(1) GRU_PROBE_STEPS(240) GRU_PROBE_FIG(그림 png) GRU_PROBE_MATCH(짝의 '상자를 본 뒤 지난 시간'을 맞추는 폭 [결정], 기본 5)
GRU_PROBE_SEED0(첫 시드 번호, 기본 0) GRU_PROBE_NETS(네트워크를 고름, 기본 0123; 4 = 퍼지게 + 학습 창(12.8 s) 조각으로만 가르침, 5 = 옥타브마다 고르게 퍼지게, 6 = 5를 조각으로만, 7 = 퍼지게 + 조각 + 조각 앞의 상태는 이어받음 = policy 학습과 같은 조건, 8 = 7을 기본 초기값으로)"""
import os
import sys
import time
import numpy as np
import torch
from torch import nn
from rec_lib import Rec, along_track, auc, plt

IB, IJ, IT, IN, IGAP = 0, 1, 2, 3, 4
GAP, NU, MB, CTX, TMAX, GAMMA, HZ, VAL, EMP_DT = 0.6, 128, 32, 30, 300, 0.995, 300, 0.2, 40
PPO_L = 128                                                 # the policy training's window (plan_rl --mem-chunk) [decisions]
ARMS = os.environ.get("GRU_PROBE_ARMS", "ABC")
SEEDS = int(os.environ.get("GRU_PROBE_SEEDS", "1"))
STEPS = int(os.environ.get("GRU_PROBE_STEPS", "240"))
FIG = os.environ.get("GRU_PROBE_FIG", "")
MATCH = int(os.environ.get("GRU_PROBE_MATCH", "5"))       # [decisions] the pairs' time since a box was last in view: within this of the moment's (capped at 6 s); 0 = not matched
PREP_ONLY = os.environ.get("GRU_PROBE_PREP_ONLY", "") == "1"
SEED0 = int(os.environ.get("GRU_PROBE_SEED0", "0"))       # the first seed (one seed per run = a row per seed)
PICK = os.environ.get("GRU_PROBE_NETS", "0123")          # which of the four networks
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
out_txt, recs = sys.argv[1], sys.argv[2:]
rng = np.random.default_rng(0); rng2 = np.random.default_rng(1)
def draw(RES, path):
    BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#777777", "#333333"
    fig, axs = plt.subplots(1, 2, figsize=(15, 0.42 * len(RES) + 1.8), sharey=True)
    for ax, (ia, ie, ttl) in zip(axs, ((1, 2, "다른 상자 세션의 같은 자리와 구분 ('어디' 있는지)"), (3, 4, "상자가 없던 세션의 같은 자리·같은 시각과 구분 (이 세션에 상자가 '있다'는 것)"))):
        for k, r_ in enumerate(RES):
            yk = len(RES) - 1 - k; a, se = float(r_[ia]), float(r_[ie])
            col = GRAY if "(" != r_[0][0] else (ORANGE if "처음부터" in r_[0] else BLUE)
            if np.isfinite(a):
                ax.plot([a - se, a + se], [yk, yk], color=col, lw=2); ax.plot([a], [yk], "o", color=col, ms=8)
                ax.text(min(a + se, 0.98) + 0.012, yk, f"{a:.2f}", va="center", fontsize=9, color=INK)
        ax.axvline(0.5, color=INK, lw=1, ls="--"); ax.set_xlim(0.3, 1.05); ax.set_title(ttl, fontsize=11); ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("상자 쪽 점수가 짝보다 높은 비율 (0.5 = 구분 못 함; 막대 = ± 표준오차)")
    axs[0].set_yticks(range(len(RES))); axs[0].set_yticklabels([r_[0] for r_ in RES][::-1], fontsize=9)
    fig.suptitle("지난 바퀴에 지나간 상자가 시야에 들어오기 0.5~1.0 s 전: 따로 가르친 GRU가 그것을 들고 있나 (가르치지 않은 세션에서 채점; 주황 = 세션 처음부터 보는 GRU)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=80); print("그림 " + path)


if recs[0].endswith(".npy"):                                                # only draw: the rows of earlier runs (the reference rows once)
    rr, seen = [], set()
    for f in recs:
        for r_ in np.load(f):
            if r_[0] not in seen:
                seen.add(r_[0]); rr.append(tuple(r_))
    draw(rr, out_txt); sys.exit(0)
log = open(out_txt, "w")


def say(*a):
    print(*a, flush=True); print(*a, file=log, flush=True)


def since(mask):
    """(T', B) decisions since the mask was last true (large before the first)"""
    out = np.zeros(mask.shape, int); cur = np.full(mask.shape[1], 10 ** 6)
    for t in range(mask.shape[0]):
        cur = np.where(mask[t], 0, cur + 1); out[t] = cur
    return out


def prep(npz, s0):
    """one record -> its sessions' inputs, targets, masks and the scored moments with their pairs (session numbers offset by s0)"""
    R = Rec(npz); z = R.z; T, B, dt = R.T, R.B, R.dt
    mm = np.load(npz[:-4] + "_mem.npz"); rows = np.load(npz[:-4] + "_boxrows.npy")
    k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
    st = R.st.astype(float); xy = st[:T, :, :2]; yaw = st[:T, :, 2]; V = np.abs(st[:, :, 3])
    sid, pose = R.sid, R.pose.astype(float); P = sid.shape[2]
    box_w = (sid[0] >= 0).any(-1); opp_w = R.on[:T].any(0)
    mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
    FOV, RMAX = np.radians(135.0), 10.0
    s_pos, lap_i, L = along_track(z, T); u = lap_i * L + s_pos
    n3 = int(round(3.0 / dt))

    def sees(tk, t, b, c):
        """is the box centre c in the scan's direct view at decision t (as exam_boxes.py / gru_ahead.py)"""
        sx, sy = xy[t, b, 0] + mount * np.cos(yaw[t, b]), xy[t, b, 1] + mount * np.sin(yaw[t, b])
        dx, dy = c[0] - sx, c[1] - sy; dist = np.hypot(dx, dy)
        a = (np.arctan2(dy, dx) - yaw[t, b] + np.pi) % (2 * np.pi) - np.pi
        if dist > RMAX or abs(a) > FOV:
            return False
        n = max(int((dist - 0.25) / 0.05), 1)
        f = np.linspace(0.0, max(dist - 0.25, 0.0) / max(dist, 1e-6), n)
        return bool((R.wall_dist(tk, np.stack([sx + f * dx, sy + f * dy], 1)) > 0.0).all())

    vis = np.zeros((k_rm, B, P), bool); box_s = np.full((B, P), np.nan)
    for b in np.nonzero(box_w)[0]:
        tk = int(R.tid[0, b])
        for j in np.nonzero(sid[0, b] >= 0)[0]:
            d = np.linalg.norm(pose[:k_rm, b, j, :2] - xy[:k_rm, b], axis=1)
            box_s[b, j] = s_pos[int(d.argmin()), b]
            for t in np.nonzero(d < RMAX + 0.5)[0]:
                vis[t, b, j] = sees(tk, t, b, pose[t, b, j, :2])
    vis_any = np.zeros((T, B), bool); vis_any[:k_rm] = vis.any(2)
    since_vis = since(vis_any); since_hard = since(R.d[:, :, 3:6].sum(-1) > 0)
    with np.errstate(invalid="ignore"):
        ahead = np.nanmin(np.where(np.isfinite(box_s)[None], np.mod(box_s[None] - s_pos[:k_rm, :, None], L), np.inf), axis=2)
    tgrid = np.arange(T)[:, None]
    moving = (V[:T] > 2.0) & (since_hard > n3) & (tgrid >= n3)
    drive = box_w[None] & moving & (tgrid < k_rm)                         # normal driving in a box session before the removal
    lapc = lap_i >= 1

    # (가) the label and where it is asked
    Y = np.zeros((T, B), np.float32); dont = np.zeros((T, B), bool); P_ = []
    for r in rows:
        b, j, tp, npass = int(r[IB]), int(r[IJ]), int(r[IT]), int(r[IN])
        if tp >= k_rm:
            continue
        v = vis[:, b, j]
        near = np.nonzero(v[max(tp - 10, 0): min(tp + 5, k_rm)])[0]
        if len(near) == 0:
            continue                                                       # never in direct view around this pass
        ts = max(tp - 10, 0) + int(near[-1]); miss = 0
        while ts > 0 and miss <= 2:                                        # back to where this visible stretch began (gru_ahead.py)
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
        w0, w1 = max(ts - 10, 0), ts
        if r[IGAP] >= GAP:
            dont[w0: w1, b] = True                                         # a box off the path: not asked
            continue
        if npass < 2:
            continue                                                       # first pass: nothing to know, the label stays 0
        Y[w0: w1, b] = 1.0; dont[max(ts - 20, 0): w0, b] = True            # 1-2 s before: not asked
        if v[max(ts - 30, 0): ts].any():
            continue                                                       # this box was in view in the last 3 s: not scored
        for dd in (5, 10):
            t = ts - dd
            if t >= n3 and drive[t, b] and since_vis[t, b] >= 5:
                P_.append((b, t, dd, float(np.hypot(*(pose[ts, b, j, :2] - xy[ts, b]))), float(np.mod(box_s[b, j] - s_pos[t, b], L))))
    M = (~vis_any | (Y > 0)) & ~(dont & (Y == 0))
    M[k_rm: k_rm + 100, box_w] = False                                     # the first lap after the removal: it cannot know yet

    # the pairs
    half = np.arange(B) % 2
    SBIN = 0.5; nb = int(np.ceil(L / SBIN))
    qt, qb = np.nonzero(drive & (since_vis >= 5) & (Y == 0))
    qbin = np.minimum((s_pos[qt, qb] / SBIN).astype(int), nb - 1)
    by_bin = [np.nonzero(qbin == i)[0] for i in range(nb)]
    et, eb = np.nonzero(moving & ~box_w[None])                             # empty sessions: the same clock, no box ever
    ebin = np.minimum((s_pos[et, eb] / SBIN).astype(int), nb - 1)
    e_bin = [np.nonzero(ebin == i)[0] for i in range(nb)]
    pos, oth, rmv = [], [], []
    for b, t, dd, sight, togo in P_:
        i = min(int(s_pos[t, b] / SBIN), nb - 1)
        cand = np.concatenate([by_bin[(i + k) % nb] for k in (-1, 0, 1)])
        ct, cb = qt[cand], qb[cand]
        ds = np.abs(np.mod(s_pos[ct, cb] - s_pos[t, b] + 0.5 * L, L) - 0.5 * L)
        ok = (cb != b) & (half[cb] == half[b]) & (ds <= 0.3) & lapc[ct, cb] & (opp_w[cb] == opp_w[b]) & (ahead[ct, cb] > togo + 6.0)
        if MATCH:                                                          # else the pairs have boxes just behind them more often than the moment has
            ok &= np.abs(np.minimum(since_vis[ct, cb], 60) - min(int(since_vis[t, b]), 60)) <= MATCH
        k = np.nonzero(ok)[0]
        k = k[np.argsort(ds[k])]
        _, first = np.unique(cb[k], return_index=True)
        k = k[first]; k = k[rng.permutation(len(k))[:4]]
        o_ = np.stack([s0 + cb[k], ct[k]], 1) if len(k) else np.zeros((0, 2), int)
        # empty sessions at the same place within EMP_DT of the same session time (11-12: the same session after the removal
        # was told apart by a network that only counts time -- 0.95 for one that had learned nothing about boxes)
        cand = np.concatenate([e_bin[(i + k_) % nb] for k_ in (-1, 0, 1)])
        ct2, cb2 = et[cand], eb[cand]
        ds2 = np.abs(np.mod(s_pos[ct2, cb2] - s_pos[t, b] + 0.5 * L, L) - 0.5 * L)
        k2 = np.nonzero((half[cb2] == half[b]) & (ds2 <= 0.3) & (opp_w[cb2] == opp_w[b]) & (np.abs(ct2 - t) <= EMP_DT))[0]
        k2 = k2[np.argsort(np.abs(ct2[k2] - t))]
        _, first = np.unique(cb2[k2], return_index=True)
        k2 = k2[first]; k2 = k2[rng2.permutation(len(k2))[:4]]
        r_ = np.stack([s0 + cb2[k2], ct2[k2]], 1) if len(k2) else np.zeros((0, 2), int)
        pos.append((s0 + b, t, dd, sight, min(int(since_vis[t, b]), 60) * dt, np.minimum(since_vis[ct[k], cb[k]], 60).mean() * dt if len(k) else np.nan))
        oth.append(o_); rmv.append(r_)

    # (나) the next 3 s of progress, (다) the value-like sum
    P3 = np.zeros((T, B), np.float32); P3m = np.zeros((T, B), bool)
    P3[: T - CTX] = u[CTX:] - u[: T - CTX]; P3m[: T - CTX] = True
    du = np.diff(u, axis=0)
    hit = R.d[:, :, 0:3].sum(-1)[: T - 1]; lost = np.clip(V[: T - 1] - V[1: T], 0, None)
    rew = du / L - (0.5 * hit + lost * (hit > 0)) / 7.5
    G = np.zeros((T, B), np.float32); Gm = np.zeros((T, B), bool); acc = np.zeros((T - HZ, B))
    for k in range(HZ - 1, -1, -1):
        acc = rew[k: k + T - HZ] + GAMMA * acc
    G[: T - HZ] = acc; Gm[: T - HZ] = True
    info = (f"{z['exam']} · {R.name} · 실행 {z['mode']} · 상자 {z['count']}: 세션 {B}개(상자 있는 것 {int(box_w.sum())}), '전에 지나간 상자가 1 s 안에 보인다' 결정 "
            f"{int((Y > 0).sum())}개 = 묻는 결정의 {100 * (Y > 0).sum() / M.sum():.1f} %, 채점 순간 {len(pos)}개")
    return dict(F=mm["feat"].astype(np.float32).transpose(1, 0, 2), H=mm["mem"].astype(np.float32).transpose(1, 0, 2), Y=Y.T, M=M.T, P3=P3.T, P3m=P3m.T, G=G.T, Gm=Gm.T,
                pos=pos, oth=oth, rmv=rmv, half=half, box=box_w, info=info, T=T)


D = []; s0 = 0
for f in recs:
    D.append(prep(f, s0)); s0 += len(D[-1]["half"]); say(D[-1]["info"])
T = D[0]["T"]
X = torch.tensor(np.concatenate([d["F"] for d in D])); Hpol = np.concatenate([d["H"] for d in D]); NS, _, NI = X.shape
half = np.concatenate([d["half"] for d in D]); box = np.concatenate([d["box"] for d in D])
pos = np.array([p for d in D for p in d["pos"]]); oth = [o for d in D for o in d["oth"]]; rmv = [r for d in D for r in d["rmv"]]
ps, pt, pd = pos[:, 0].astype(int), pos[:, 1].astype(int), pos[:, 2].astype(int)
TG = {"A": (np.concatenate([d["Y"] for d in D]), np.concatenate([d["M"] for d in D])),
      "B": (np.concatenate([d["P3"] for d in D]), np.concatenate([d["P3m"] for d in D])),
      "C": (np.concatenate([d["G"] for d in D]), np.concatenate([d["Gm"] for d in D]))}
n_o = np.array([len(o) for o in oth]); n_r = np.array([len(r) for r in rmv])
say(f"세션 {NS}개(반 {int((half == 0).sum())}/{int((half == 1).sum())}; 상자 세션 {int(box[half == 0].sum())}/{int(box[half == 1].sum())}), 채점 순간 {len(pos)}개"
    f"(세션 {len(np.unique(ps))}개): 0.5 s 전 {int((pd == 5).sum())} · 1.0 s 전 {int((pd == 10).sum())} | '어디' 짝이 있는 것 {int((n_o > 0).sum())}(짝 평균 "
    f"{n_o[n_o > 0].mean() if (n_o > 0).any() else 0:.1f}) · '있다' 짝이 있는 것 {int((n_r > 0).sum())} | 보이기 시작할 때 상자까지 거리 25/50/75 %: "
    + " / ".join(f"{x:.1f}" for x in np.percentile(pos[:, 3], [25, 50, 75])) + " m | 마지막으로 상자를 본 뒤 지난 시간(6 s에서 자름) 중앙: 상자 쪽 "
    f"{np.median(pos[n_o > 0, 4]):.1f} s · 그 짝 {np.nanmedian(pos[n_o > 0, 5]):.1f} s" + (f" (짝은 ±{MATCH / 10:.1f} s 안으로 맞춤)" if MATCH else " (맞추지 않음)"))
if PREP_ONLY:
    sys.exit(0)
mu_x = X[::3].reshape(-1, NI).mean(0); sd_x = X[::3].reshape(-1, NI).std(0) + 1e-3
X = ((X - mu_x) / sd_x).clamp(-6, 6)                                       # a fixed rescale of the inputs (the same for every network)


class Net(nn.Module):
    def __init__(self, kind, chrono, seed):
        super().__init__()
        torch.manual_seed(seed); self.kind = kind
        if kind == "mlp":
            self.body = nn.Sequential(nn.Linear(NI, NU), nn.Tanh(), nn.Linear(NU, NU), nn.Tanh())
        else:
            self.gru = nn.GRU(NI, NU, batch_first=True)                    # PyTorch default init, as the policy's GRUCell
            if chrono:                                                     # Tallec & Ollivier 2018: each value's keep time spread over 1..TMAX decisions
                with torch.no_grad():                                      # "log": as many values per octave of keep time (half of them under ~1.7 s)
                    keep_t = (torch.exp(torch.empty(NU).uniform_(0.0, float(np.log(TMAX)))) if chrono == "log" else torch.empty(NU).uniform_(1.0, TMAX - 1.0))
                    self.gru.bias_ih_l0[NU: 2 * NU] = torch.log(keep_t.clamp_min(1.0 + 1e-3) - 1.0 if chrono == "log" else keep_t); self.gru.bias_hh_l0[NU: 2 * NU] = 0.0
        self.head = nn.Linear(NU, 1)

    def forward(self, x):
        h = self.body(x) if self.kind == "mlp" else self.gru(x)[0]
        return h, self.head(h)[..., 0]


def chunks(x, o, L=None):
    L = CTX if L is None else L
    n = (x.shape[1] - o) // L
    return x[:, o: o + n * L].reshape(-1, L, *x.shape[2:])


def fit(kind, chrono, seed, y, m, idx, bce):
    """fit on the sessions idx (VAL of them choose when to stop); returns the network at its best validation loss"""
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(len(idx), generator=g).numpy()]; nv = max(int(VAL * len(idx)), 1)
    iv, it = torch.tensor(perm[:nv]), torch.tensor(perm[nv:])
    net = Net(kind, chrono, seed); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-4)
    pw = torch.tensor(float((m[it] & (y[it] == 0)).sum() / max((m[it] & (y[it] > 0)).sum(), 1))) if bce else None

    def loss_of(ii, o):
        x, yy, mm_ = X[ii], y[ii], m[ii]
        if kind == "gru3":
            x, yy, mm_ = chunks(x, o), chunks(yy, o), chunks(mm_, o).clone(); mm_[:, :10] = False
        elif kind == "gruT":                                               # as the policy's training: learning reaches back PPO_L decisions at most
            x, yy, mm_ = chunks(x, 4 * o, PPO_L), chunks(yy, 4 * o, PPO_L), chunks(mm_, 4 * o, PPO_L)   # (each piece from a blank state: harsher than PPO, which carries it)
        if kind == "gruC":                                                 # as the policy's training: each piece starts from the state the
            o4 = 4 * o; n_p = (x.shape[1] - o4) // PPO_L                   # network had there, and learning does not reach back past it
            st = o4 + PPO_L * torch.arange(n_p)
            with torch.no_grad():
                h0 = net.gru(x)[0][:, (st - 1).clamp_min(0)] * (st > 0)[None, :, None]
            x, yy, mm_ = chunks(x, o4, PPO_L), chunks(yy, o4, PPO_L), chunks(mm_, o4, PPO_L)
            out = net.head(net.gru(x, h0.reshape(1, -1, NU).contiguous())[0])[..., 0]
        else:
            out = net(x)[1]
        return nn.functional.binary_cross_entropy_with_logits(out[mm_], yy[mm_], pos_weight=pw) if bce else ((out[mm_] - yy[mm_]) ** 2).mean()

    best, keep = float("inf"), None
    for step in range(STEPS):
        ii = it[torch.randperm(len(it), generator=g)[:MB]]
        loss = loss_of(ii, int(torch.randint(0, CTX, (1,), generator=g)))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 20 == 0 or step == STEPS - 1:
            with torch.no_grad():
                lv = float(loss_of(iv, 0))
            if lv < best:
                best, keep = lv, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(keep)
    return net


@torch.no_grad()
def run(net):
    """every session through the network: hidden (NS, T, NU), output (NS, T). The 3 s network sees only the last CTX decisions at each one."""
    if net.kind != "gru3":
        h, o = net(X)
        return h.numpy(), o.numpy()
    Hh = np.zeros((NS, T, NU), np.float32); O = np.zeros((NS, T), np.float32)
    for s in range(NS):
        w = X[s].unfold(0, CTX, 1).permute(0, 2, 1)                        # (T - CTX + 1, CTX, NI)
        h, o = net(w)
        Hh[s, CTX - 1:] = h[:, -1].numpy(); O[s, CTX - 1:] = o[:, -1].numpy()
    return Hh, O


def beat(S, which):
    """per scored moment: the share of its pairs its score beats (nan without pairs)"""
    w = np.full(len(ps), np.nan)
    for i in range(len(ps)):
        pr = oth[i] if which == "oth" else rmv[i]; ng = S[pr[:, 0], pr[:, 1]]
        if len(ng):
            w[i] = np.mean((S[ps[i], pt[i]] > ng) + 0.5 * (S[ps[i], pt[i]] == ng))
    return w


def summ(w, sel):
    """mean and a session-bootstrap standard error of the per-moment shares"""
    ok = sel & np.isfinite(w)
    if ok.sum() < 8:
        return float("nan"), float("nan"), int(ok.sum())
    g = ps[ok]; ww = w[ok]; ug = np.unique(g); r_ = np.random.default_rng(1)
    by = [ww[g == q] for q in ug]
    bs = [np.concatenate([by[q] for q in r_.integers(0, len(ug), len(ug))]).mean() for _ in range(400)]
    return float(ww.mean()), float(np.std(bs)), int(ok.sum())


def readout(Z, which):
    """a linear read-out of 'the box side' from states (NS, T, D): fit on the other half's moments and their pairs, scored on this half
    -> (NS, T) scores. Z = one array, or {h: array} when the states of fold h come from the network fit on the half that is not h."""
    S = np.zeros((NS, T), np.float32)
    for h in (0, 1):
        Zf = Z[h] if isinstance(Z, dict) else Z
        A, yy = [], []
        for i in np.nonzero(half[ps] != h)[0]:
            pr = oth[i] if which == "oth" else rmv[i]; ng = Zf[pr[:, 0], pr[:, 1]]
            if len(ng):
                A += [Zf[ps[i], pt[i]][None], ng]; yy += [1.0] + [-1.0] * len(ng)
        if len(A) < 10:
            continue
        A = np.concatenate(A).astype(np.float64); yy = np.array(yy)
        mu, sd = A.mean(0), A.std(0) + 1e-4; A = (A - mu) / sd
        wt = np.where(yy > 0, 0.5 / (yy > 0).mean(), 0.5 / (yy < 0).mean())
        coef = np.linalg.solve((A * wt[:, None]).T @ A / wt.sum() + 0.1 * np.eye(A.shape[1]), (A * wt[:, None]).T @ (yy - (wt * yy).sum() / wt.sum()) / wt.sum())
        te = np.nonzero(half == h)[0]
        S[te] = (((Zf[te] - mu) / sd) @ coef).astype(np.float32)
    return S


def row(name, w_o, w_r, extra=""):
    cells = []
    for w in (w_o, w_r):
        for dd in (5, 10):
            a, se, n = summ(w, pd == dd)
            cells.append(f"{a:.2f}±{se:.2f} ({n})" if np.isfinite(a) else f"부족 ({n})")
    a, se, n = summ(w_o, np.ones(len(ps), bool)); a2, se2, n2 = summ(w_r, np.ones(len(ps), bool))
    RES.append((name, a, se, a2, se2))
    say(f"  {name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]}" + (f" | {extra}" if extra else ""))


RES = []; KEEP = {}
HEAD = "  무엇 | '어디' 짝: 0.5 s 전 | 1.0 s 전 | '있다' 짝: 0.5 s 전 | 1.0 s 전 | 가르친 것 자체를 맞힌 정도: 채점 반 (가르친 반)"
say("\n[기준] 기록에 남은 것을 직선으로 읽음 (가르친 반의 순간으로 맞추고 다른 반에서 채점)"); say(HEAD)
row("policy가 들고 있던 기억(128)", beat(readout(Hpol, "oth"), "oth"), beat(readout(Hpol, "rmv"), "rmv"))
Xn = X.numpy()
row("지금 입력(292)", beat(readout(Xn, "oth"), "oth"), beat(readout(Xn, "rmv"), "rmv"))

NETS = (("기억 없음(지금 입력만)", "mlp", False), ("최근 3 s만 보는 GRU", "gru3", False), ("처음부터 보는 GRU, 기본 초기값", "gru", False),
        (f"처음부터 보는 GRU, update gate bias를 1~{TMAX // 10} s에 퍼지게", "gru", True),
        (f"처음부터 보는 GRU, 1~{TMAX // 10} s에 퍼지게, {PPO_L / 10:.1f} s 조각으로만 가르침", "gruT", True),
        (f"처음부터 보는 GRU, 1~{TMAX // 10} s에 옥타브마다 고르게 퍼지게", "gru", "log"),
        (f"처음부터 보는 GRU, 1~{TMAX // 10} s에 옥타브마다 고르게 퍼지게, {PPO_L / 10:.1f} s 조각으로만 가르침", "gruT", "log"),
        (f"처음부터 보는 GRU, 1~{TMAX // 10} s에 퍼지게, {PPO_L / 10:.1f} s 조각 + 조각 앞의 상태는 이어받음(policy 학습과 같은 조건)", "gruC", True),
        (f"처음부터 보는 GRU, 기본 초기값, {PPO_L / 10:.1f} s 조각 + 조각 앞의 상태는 이어받음(policy 학습과 같은 조건)", "gruC", False))
TAG = {"A": "(가)", "B": "(나)", "C": "(다)"}
TITLE = {"A": "(가) 직접 가르침: '전에 지나간 상자가 1 s 안에 시야에 들어온다'", "B": "(나) 앞으로 3 s 동안 나아가는 거리를 가르침",
         "C": "(다) 학습의 value와 같은 꼴(0.995로 줄여 더한 앞으로의 진행 - 접촉 벌)을 가르침"}
for arm in ARMS:
    y_np, m_np = TG[arm]; bce = arm == "A"
    if not bce:
        y_np = (y_np - y_np[m_np].mean()) / (y_np[m_np].std() + 1e-6)
    y, m = torch.tensor(y_np.astype(np.float32)), torch.tensor(m_np)
    say(f"\n[{TITLE[arm]}]"); say(HEAD)
    for name, kind, chrono in [NETS[int(c_)] for c_ in PICK]:
        t0 = time.time(); W = {k: [] for k in ("o", "r", "ro", "rr", "io")}; q_out, q_in = [], []
        for seed in range(SEED0, SEED0 + SEEDS):
            O = np.zeros((NS, T), np.float32); Oin = np.zeros((NS, T), np.float32); Hin = {}
            for h in (0, 1):                                               # fit on the other half, score half h
                net = fit(kind, chrono, 100 * seed + h, y, m, np.nonzero(half != h)[0], bce)
                hh, oo = run(net)
                O[half == h] = oo[half == h]; Oin[half != h] = oo[half != h]; Hin[h] = hh
            KEEP[f"{arm}_{kind}{('_chrono' + (chrono if isinstance(chrono, str) else '')) if chrono else ''}_s{seed}"] = O.astype(np.float16)   # held-out outputs: re-scoring needs no refit
            for name_q, OO in (("out", O), ("in", Oin)):                   # how well the taught thing itself is told
                if bce:
                    q = auc(OO[m_np][::5], y_np[m_np][::5] > 0)
                else:
                    q = 1.0 - ((OO[m_np] - y_np[m_np]) ** 2).mean() / y_np[m_np].var()
                (q_out if name_q == "out" else q_in).append(q)
            if bce:
                W["o"].append(beat(O, "oth")); W["r"].append(beat(O, "rmv")); W["io"].append(beat(Oin, "oth"))
            else:
                W["o"].append(beat(-O, "oth")); W["r"].append(beat(-O, "rmv"))
                W["ro"].append(beat(readout(Hin, "oth"), "oth")); W["rr"].append(beat(readout(Hin, "rmv"), "rmv"))
        mean = lambda k: np.nanmean(np.stack(W[k]), 0)
        qual = (f"AUC {np.mean(q_out):.2f} ({np.mean(q_in):.2f})" if bce else f"설명한 분산 {100 * np.mean(q_out):.0f} % ({100 * np.mean(q_in):.0f} %)") + f" · {time.time() - t0:.0f} s"
        if bce:
            row(f"{TAG[arm]} {name}", mean("o"), mean("r"), qual)
            a, se, n = summ(mean("io"), np.ones(len(ps), bool))
            say(f"      (가르친 반의 순간에서 '어디' 짝: {a:.2f}±{se:.2f} -- 이것만 높고 위가 0.5면 외운 것)")
        else:
            row(f"{TAG[arm]} {name} · 예측값이 낮은 쪽", mean("o"), mean("r"), qual)
            row(f"{TAG[arm]} {name} · 기억 값을 직선으로 읽음", mean("ro"), mean("rr"))
say("\n읽는 법: 숫자 = 상자 쪽 점수가 짝보다 높은 비율 ± 표준오차(세션을 다시 뽑아 잼), 괄호 = 순간 수. 0.5 = 구분 못 함. '기억 없음'이 0.5보다 높으면 그만큼은 기억이 아니라 "
    "지금 입력에서 새는 것(상자 모서리가 중심보다 먼저 보임 등). '처음부터 보는 GRU'가 '최근 3 s만 보는 GRU'와 '기억 없음'보다 뚜렷이 높아야 3 s보다 오래된 것을 들고 온 것.")
np.save(out_txt[:-4] + ".npy", np.array([(n,) + tuple(f"{x:.4f}" for x in r_) for n, *r_ in RES]))
np.savez_compressed(out_txt[:-4] + "_scores.npz", **KEEP)

if FIG:
    draw(RES, FIG)
