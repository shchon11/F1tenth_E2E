"""기억(GRU)이 plan과 이어져 있는가 -- policy 안을 직접 봄 (11-13, 사용자: "feature도 좀 들여다 보고 gradient도 좀 볼 수 있으면 보고
기억이 planner 랑 연관이 생기고 있는지도 좀 보고"). 숫자 표(바퀴 시간·접촉)가 아니라 결정마다 policy의 계산을 다시 해서 봄.
입력: 시험 기록(exam_record.py, F1SIM_EXAM_SAVE_MEM=1 -> 옆의 _mem.npz: 결정마다 policy가 읽은 특징 feat = 스캔 인코더 출력 256 +
정규화한 상태 36, 그 결정 뒤의 기억 mem 128)과 그 checkpoint. policy(mem_front): h1 = GRU(feat, h0), plan = pi(trunk([feat, h1])).
기록된 기억을 checkpoint 무게로 다시 계산해 맞는지 먼저 확인함.
  1) 지울 때: 같은 입력에서 직전 기억 h0만 0으로 -> plan 1.0 s 점의 옆 위치 차 [m] · 1.0 s 속도 차 [m/s] · 20개 출력의 차를 policy 탐색 잡음
     폭(std)으로 나눈 크기. 견줄 것: 스캔만 0.1 s 전 것으로 바꿨을 때의 차(한 결정 사이에 스캔이 바뀌는 만큼).
  2) 기울기 몫: plan 1.0 s 점의 옆 위치와 속도의 기울기 x 그 입력의 흔들림 폭(std) -> 묶음별 크기(지금 스캔 256 · 지금 상태 36 · 직전 기억 128)
     와 직전 기억의 몫.
  3) 시간을 거슬러: 지금 plan(1.0 s 옆 위치)의 k 결정 전 입력(feat)에 대한 기울기 크기 / k = 0 의 크기, k = 1..50 (0.1~5 s).
  4) 기억에 무엇이 들어 있나(선형 탐침, 세션 단위로 나눠 검증한 AUC): 직전 기억 h0 vs 지금 입력 feat 로
     (가) 이 세션에 상자가 있다 (상자 안 보이는 순간만) (나) 길 따라 3 m 안 앞에 지금 안 보이는 상자가 있다.
상황: 빈 세션 / 상자 세션(치우기 전)·앞 3 m 안 상자 안 보임 / 같은 곳 상자 보임 / 상자에 세게 닿기 1 s 안 / 상대 차가 3 m 안.
상자 위치·보임·길 따라 거리는 분석에서만 씀(policy 입력 아님). plan 점은 출발 조향 0으로 펼침(두 plan의 차를 보는 데만 씀).
인자: checkpoint=기록npz [checkpoint=기록npz ...] 출력앞머리"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, along_track, probe_auc, plt
from policy_lib import limits_for, plan_of

PAIRS = [a.split("=", 1) for a in sys.argv[1:-1]]
OUT = sys.argv[-1]
dev = torch.device("cpu")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
KS = [1, 2, 3, 5, 10, 20, 30, 50]                   # decisions back for the gradient through time
N_GRAD = 3000                                       # rows per situation for the gradients
N_TBT = 400                                         # rows per situation for the gradient through time
RNG = np.random.default_rng(0)


def situations(R, z, T):
    """(T, B) masks of the situations, from the record (analysis only)"""
    B = R.B; dt = R.dt
    st = R.st.astype(float); xy = st[:T, :, :2]; yaw = st[:T, :, 2]; V = np.abs(st[:T, :, 3])
    sid, pose = R.sid, R.pose.astype(float); P = sid.shape[2]
    k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
    box_w = (sid[0] >= 0).any(-1)
    mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
    FOV, RMAX = np.radians(135.0), 10.0
    s_pos, lap_i, L = along_track(z, T)

    def sees(tk, t, b, c):
        sx, sy = xy[t, b, 0] + mount * np.cos(yaw[t, b]), xy[t, b, 1] + mount * np.sin(yaw[t, b])
        dx, dy = c[0] - sx, c[1] - sy; dist = np.hypot(dx, dy)
        a = (np.arctan2(dy, dx) - yaw[t, b] + np.pi) % (2 * np.pi) - np.pi
        if dist > RMAX or abs(a) > FOV:
            return False
        n = max(int((dist - 0.25) / 0.05), 1)
        f = np.linspace(0.0, max(dist - 0.25, 0.0) / max(dist, 1e-6), n)
        return bool((R.wall_dist(tk, np.stack([sx + f * dx, sy + f * dy], 1)) > 0.0).all())

    vis = np.zeros((T, B, P), bool); box_s = np.full((B, P), np.nan)
    for b in np.nonzero(box_w)[0]:
        tk = int(R.tid[0, b])
        for j in np.nonzero(sid[0, b] >= 0)[0]:
            d = np.linalg.norm(pose[:k_rm, b, j, :2] - xy[:k_rm, b], axis=1)
            box_s[b, j] = s_pos[int(d.argmin()), b]
            for t in np.nonzero(d < RMAX + 0.5)[0]:
                vis[t, b, j] = sees(tk, t, b, pose[t, b, j, :2])
    with np.errstate(invalid="ignore"):
        gap = np.where(np.isfinite(box_s)[None], np.mod(box_s[None] - s_pos[:, :, None], L), np.inf)    # (T, B, P) along the track
    near = (gap < 3.0)
    tgrid = np.arange(T)[:, None]
    before = box_w[None] & (tgrid < k_rm)
    moving = V > 1.0
    blind = before & moving & (near & ~vis).any(2) & ~(near & vis).any(2)
    seen = before & moving & (near & vis).any(2)
    hard_box = R.d[:T, :, 4] > 0
    pre_hit = np.zeros((T, B), bool)
    for t, b in zip(*np.nonzero(hard_box)):
        pre_hit[max(t - 10, 0): t, b] = True
    op = R.op.astype(float)[:T] if R.op is not None else None
    on = R.on[:T].astype(bool) if R.on is not None else np.zeros((T, B), bool)
    near_car = on & moving & (np.hypot(*(op[:, :, :2] - xy).transpose(2, 0, 1)) < 3.0) if op is not None else np.zeros((T, B), bool)
    empty = ~box_w[None] & ~on.any(0)[None] & moving & (tgrid >= 30)
    no_box_seen = before & moving & ~vis.any(2)
    S = {"빈 세션": empty, "상자 세션 · 앞 3 m 상자 안 보임": blind, "상자 세션 · 앞 3 m 상자 보임": seen,
         "상자에 세게 닿기 1 s 안": pre_hit & moving, "상대 차 3 m 안": near_car}
    labels = {"box_session": (before | (~box_w[None] & (tgrid < k_rm))) & moving & ~vis.any(2) & ~on, "box_w": box_w,
              "blind_ahead": blind, "no_box_seen_box": no_box_seen}
    return S, labels, box_w


def analyse(ck_path, npz, lim_cache):
    from f1sim.learn.plan_policy import load_actor2
    from f1sim import plan_steps as PS
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    key = json.dumps(ck["args"].get("plan_lat_margin")) + str(ck["args"].get("lidar_mount_x"))
    if key not in lim_cache:
        lim_cache[key] = limits_for(ck)
    lim = lim_cache[key]
    actor = load_actor2(ck, dev, check_mount=False).eval()
    assert actor.memory and actor.mem_front, "mem_front memory policies only"
    for p_ in actor.parameters():
        p_.requires_grad_(False)
    R = Rec(npz); z = R.z
    mm = np.load(npz[:-4] + "_mem.npz")
    F = torch.from_numpy(mm["feat"].astype(np.float32)); H = torch.from_numpy(mm["mem"].astype(np.float32))
    T, B = F.shape[0], F.shape[1]
    rs = R.rs[:T].astype(bool)
    H0 = torch.zeros_like(H); H0[1:] = H[:-1]
    H0[1:][torch.from_numpy(rs[:-1])] = 0.0                            # a restart begins with an empty memory
    name = f"{os.path.basename(os.path.dirname(ck_path))} u{int(ck.get('update', 0))}"
    with torch.no_grad():
        h1 = actor.gru(F.reshape(-1, F.shape[2]), H0.reshape(-1, H.shape[2])).reshape(T, B, -1)
    err = float((h1 - H).abs().max())
    print(f"== {name} ({os.path.basename(npz)}): 결정 {T} x 차 {B} | 기록된 기억을 다시 계산한 차 최대 {err:.4f} (float16 기록)", flush=True)
    S, lab, box_w = situations(R, z, T)
    std = torch.exp(torch.maximum(actor.log_std, actor.log_std_floor)).detach()
    sd_f = F.reshape(-1, F.shape[2]).std(0); sd_h = H0.reshape(-1, H.shape[2]).std(0)
    res = {"name": name}
    for sname, m in S.items():
        tt, bb = np.nonzero(m)
        n = len(tt)
        if n < 30:
            print(f"  [{sname}] 결정 {n}개: 너무 적음", flush=True); continue
        pick = RNG.choice(n, min(n, N_GRAD), replace=False)
        ti, bi = torch.from_numpy(tt[pick]), torch.from_numpy(bb[pick])
        f, h0 = F[ti, bi], H0[ti, bi]
        fprev = F[(ti - 1).clamp_min(0), bi].clone(); fprev[:, 256:] = f[:, 256:]      # only the scan part 0.1 s old
        with torch.no_grad():
            y, v, mu, _ = plan_of(actor, lim, f, h0)
            yA, vA, muA, _ = plan_of(actor, lim, f, torch.zeros_like(h0))
            yS, vS, muS, _ = plan_of(actor, lim, fprev, h0)
        dA = ((mu - muA) / std).pow(2).mean(1).sqrt(); dS = ((mu - muS) / std).pow(2).mean(1).sqrt()
        # 2) gradient shares
        fg = f.clone().requires_grad_(True); hg = h0.clone().requires_grad_(True)
        yg, vg, _, _ = plan_of(actor, lim, fg, hg)
        gy = torch.autograd.grad(yg.sum(), [fg, hg], retain_graph=True)
        gv = torch.autograd.grad(vg.sum(), [fg, hg])
        def share(gf, gh):
            sc = (gf[:, :256] * sd_f[:256]).norm(dim=1); ss = (gf[:, 256:] * sd_f[256:]).norm(dim=1); sh = (gh * sd_h).norm(dim=1)
            return sc, ss, sh, sh / (sc + ss + sh + 1e-9)
        scy, ssy, shy, fry = share(*gy)
        scv, ssv, shv, frv = share(*gv)
        # 3) gradient through time (lat at 1.0 s), rows with 50 decisions of history in the same episode
        ok = (tt >= max(KS)) & (R.k0[tt, bb] <= tt - max(KS))
        cand = np.nonzero(ok)[0]
        tbt = {}
        if len(cand) >= 20:
            pk = RNG.choice(cand, min(len(cand), N_TBT), replace=False)
            t2, b2 = torch.from_numpy(tt[pk]), torch.from_numpy(bb[pk])
            K = max(KS)
            seq = torch.stack([F[t2 - K + i, b2] for i in range(K + 1)], 0).clone().requires_grad_(True)    # (K+1, n, 292)
            h = H0[t2 - K, b2]
            for i in range(K):
                h = actor.gru(seq[i], h)
            yT, _, _, _ = plan_of(actor, lim, seq[K], h)
            g = torch.autograd.grad(yT.sum(), seq)[0]                  # (K+1, n, 292)
            sens = (g * sd_f).norm(dim=2)                              # (K+1, n)
            base = sens[K]
            for k in KS:
                tbt[k] = float(torch.median(sens[K - k] / (base + 1e-9)))
        r = dict(n=n, dA_y=float(torch.median((y - yA).abs())), dA_y90=float(torch.quantile((y - yA).abs(), 0.9)),
                 dA_v=float(torch.median((v - vA).abs())), dS_y=float(torch.median((y - yS).abs())), dS_v=float(torch.median((v - vS).abs())),
                 dA_std=float(torch.median(dA)), dS_std=float(torch.median(dS)),
                 fr_y=float(torch.median(fry)), fr_v=float(torch.median(frv)),
                 g_scan_y=float(torch.median(scy)), g_state_y=float(torch.median(ssy)), g_mem_y=float(torch.median(shy)), tbt=tbt)
        res[sname] = r
        print(f"  [{sname}] 결정 {n}개", flush=True)
        print(f"    1) 직전 기억을 0으로: 1.0 s 점 옆 위치 차 중앙 {r['dA_y']:.3f} m (90 % {r['dA_y90']:.3f} m) · 1.0 s 속도 차 {r['dA_v']:.3f} m/s · "
              f"출력 차 = 탐색 잡음 폭의 {r['dA_std']:.2f}배 | 견줌: 스캔만 0.1 s 전 것 -> {r['dS_y']:.3f} m · {r['dS_v']:.3f} m/s · {r['dS_std']:.2f}배", flush=True)
        print(f"    2) 기울기 x 흔들림 폭: 1.0 s 옆 위치 <- 스캔 {r['g_scan_y']:.3f} · 상태 {r['g_state_y']:.3f} · 직전 기억 {r['g_mem_y']:.3f} m "
              f"(기억 몫 {100 * r['fr_y']:.0f} %) | 1.0 s 속도의 기억 몫 {100 * r['fr_v']:.0f} %", flush=True)
        if tbt:
            print("    3) 지금 plan의 옆 위치가 k 결정 전 입력에 기대는 정도(k = 0 대비, 중앙): " +
                  " · ".join(f"{k * R.dt:.1f} s {tbt[k]:.3f}" for k in KS), flush=True)
    # 4) probes: what the memory holds that the current input does not
    g_id = np.broadcast_to(np.arange(B)[None], (T, B))
    pr = {}
    for lname, title, mask, yv in (
            ("box_session", "(가) 이 세션에 상자가 있다 (상자 안 보이는 순간)", lab["box_session"], np.broadcast_to(lab["box_w"][None], (T, B))),
            ("blind_ahead", "(나) 길 따라 3 m 안 앞에 지금 안 보이는 상자 (상자 세션, 상자 안 보이는 순간)", lab["no_box_seen_box"], lab["blind_ahead"])):
        tt, bb = np.nonzero(mask)
        if len(tt) > 20000:
            k = RNG.choice(len(tt), 20000, replace=False); tt, bb = tt[k], bb[k]
        y = np.asarray(yv)[tt, bb]
        a_h = probe_auc(H0[tt, bb].numpy(), y, g_id[tt, bb])
        a_f = probe_auc(F[tt, bb].numpy(), y, g_id[tt, bb])
        a_b = probe_auc(np.concatenate([H0[tt, bb].numpy(), F[tt, bb].numpy()], 1), y, g_id[tt, bb])
        pr[lname] = (a_h, a_f, a_b, int(y.sum()), int((~y).sum()))
        print(f"  4) {title}: 직전 기억으로 AUC {a_h[0]:.3f} ±{a_h[1]:.3f} · 지금 입력으로 {a_f[0]:.3f} ±{a_f[1]:.3f} · 둘 다 {a_b[0]:.3f} "
              f"[양 {int(y.sum())} / 음 {int((~y).sum())}]", flush=True)
    res["probe"] = pr
    return res


lim_cache = {}
ALL = [analyse(c, n, lim_cache) for c, n in PAIRS]

# figure: memory's share (gradient) and the plan's reach back in time
sit = ["빈 세션", "상자 세션 · 앞 3 m 상자 안 보임", "상자 세션 · 앞 3 m 상자 보임", "상자에 세게 닿기 1 s 안", "상대 차 3 m 안"]
cols = ["#2a6fdb", "#d9480f", "#2f9e44", "#7048e8", "#868e96", "#e8590c"]
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
w = 0.8 / max(len(ALL), 1)
for i, r in enumerate(ALL):
    xs = np.arange(len(sit)) + (i - (len(ALL) - 1) / 2) * w
    ax[0].bar(xs, [100 * r[s]["fr_y"] if s in r else np.nan for s in sit], w, color=cols[i % len(cols)], label=r["name"])
    ax[1].bar(xs, [r[s]["dA_y"] if s in r else np.nan for s in sit], w, color=cols[i % len(cols)])
    s0 = "상자 세션 · 앞 3 m 상자 안 보임"
    if s0 in r and r[s0]["tbt"]:
        ax[2].plot([k * 0.1 for k in KS], [r[s0]["tbt"][k] for k in KS], "o-", color=cols[i % len(cols)], label=r["name"])
for a_ in ax[:2]:
    a_.set_xticks(np.arange(len(sit))); a_.set_xticklabels([s.replace(" · ", "\n") for s in sit], fontsize=8)
ax[0].set_ylabel("기억 몫 [%]"); ax[0].set_title("plan(1.0 s 점 옆 위치)의 민감도 가운데 직전 기억의 몫\n(기울기 x 입력 흔들림 폭)", fontsize=10)
ax[0].legend(fontsize=8)
ax[1].set_ylabel("[m]"); ax[1].set_title("직전 기억을 0으로 바꾸면 1.0 s 점 옆 위치가 움직이는 거리(중앙)", fontsize=10)
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("몇 초 전 입력 [s]"); ax[2].set_ylabel("지금 입력 대비")
ax[2].set_title("앞 3 m에 안 보이는 상자가 있을 때: 지금 plan이\n몇 초 전 입력에 기울기로 기대는 정도(중앙)", fontsize=10)
ax[2].grid(alpha=0.3, which="both"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=100); plt.close(fig)
print("그림", OUT + ".png", flush=True)
