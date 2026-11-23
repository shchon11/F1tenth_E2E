"""(11-18, 사용자: "좀 더 균일하고 라인쪽에 밀집되게", 앞서 "빈 트랙에서 가는 길에 장애물을 그 근처에서 뽑히게", "맵당 한 대략 3개 씩?")
시험 상자 배치 은행 견본: 배치마다 상자 3개, 빈 트랙에서 달린 기준 길(exam_line_layout.py의 _refline.npz) 위.
  - 트랙 따라: 한 바퀴를 (배치 수 x 3)개의 같은 토막으로 나눠 토막마다 상자 하나 -> 모든 배치의 상자를 합치면 트랙 전체에 고르게.
    배치마다 세 상자는 한 바퀴의 1/3쯤씩 떨어짐
  - 옆으로: 기준 길에서 상자 중심까지 정규분포(표준편차 SIG)를 ±0.40 m에서 자름 -> 길 위에 밀집
  - 모양: 시험 은행(exam1)의 상자 모양을 그 비율대로, 방향은 무작위
  - 판정: layout_bank.py와 같음(최소시간 라인 생성기: 차 몸체가 상자·벽과 0.10 m 이상, 곡률이 조향 한계 이하). 못 지나가는 배치는
    통째로 다시 뽑음(상자를 빼지 않음: 늘 3개). 상자가 벽에 0.10 m 안이면 그 상자만 옆 거리를 다시 뽑음.
은행은 exam1 은행과 같은 형식(per_track: sid, pose, lap, length) -> 시험은 상자 개수 "fixed"로 3개를 다 씀.
인자: 세계json 기준길npz 그림용시험기록npz 출력.pt 배치수 [그림머리]   (CPU에서 돎: GPU 두 개는 학습 중)"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "f1sim")))
from f1sim.learn import state_rl as SR
from f1sim import layout_line as LL
from f1sim.raceline import curvature
from rec_lib import Rec, along_track, plt

WJ, REFL, RREC, OUT, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
FIG = sys.argv[6] if len(sys.argv) > 6 else OUT[:-3]
EXAM1 = "/home/shchon11/f1sim_runs/prop_banks_exam1_korea_2026_competition/korea_2026_competition_exam1.pt"
NEAR, SIG, N_BOX, SEED, NB, ITERS = 0.40, float(os.environ.get("LINE_SIG", 0.15)), 3, 20261118, 32, 300
CLR_PROP, CLR_WALL = 0.10, 0.10
rng = np.random.default_rng(SEED)
dev = torch.device("cpu")

a = argparse.Namespace(**json.load(open(WJ)))
a.envs, a.seed, a.device = 2, SEED, "cpu"
a.prop_place, a.prop_count, a.prop_fraction, a.prop_bank = "uniform", "fixed", 1.0, ""
a.warps, a.track_scales = 0, None
if os.environ.get("BANK_TRACKS"):
    # (11-23) the map on its own grid instead of the world's +smooth one (same name = same bank key, 16x fewer cells to judge on)
    a.tracks = os.environ["BANK_TRACKS"]
env = SR.build_env(a, dev)
names = getattr(env, "_track_names", None) or [SR.KOREA]
track = env.sim.track.tracks[0]
if os.environ.get("BANK_LINE", "") in ("centerline", "smooth"):
    # (11-23) as layout_bank.py: the judge's line search starts from the lane's centerline ("smooth": smoothed), no racing line
    from types import SimpleNamespace
    from f1sim.layout_line import smooth_line
    rls = [SimpleNamespace(xy=smooth_line(np.asarray(track.centerline, float)) if os.environ["BANK_LINE"] == "smooth"
                           else np.asarray(track.centerline, float))]
else:
    _, rls = SR.common.load_tracks(names[:1], racelines=True, **dict(SR.common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
builder = LL.LayoutLines(track, rls[0], dev, veh_width=float(env.cfg.vehicle.width), vehicle=env.cfg.vehicle, iters=ITERS)
kmax = builder.kmax
C = env.procedural.C

R0 = Rec(RREC)
ref = np.load(REFL); refp, refs, L = ref["xy"], ref["s"], float(ref["L"])
tan = np.roll(refp, -1, 0) - np.roll(refp, 1, 0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)                                   # + = left of the direction of travel
if str(getattr(a, "prop_shapes", "all") or "all") == "real_boxes":
    # (11-23) the bag-scan map's exam: the venue's measured box sizes (`REAL_BOX_SHAPES`), each equally likely, footprints from
    # the catalogue itself (a record holds only the shapes its own sessions drew)
    from f1sim import props as _props
    po = env.procedural
    pool = po.line_pool.cpu().numpy()
    FP = {int(k): np.asarray(_props.build(po.shapes[k].style, seed=0, **dict(po.shapes[k].dims)).envelope.footprint, float)
          for k in pool}
else:
    b1 = torch.load(EXAM1, map_location="cpu", weights_only=False)["per_track"]["korea_2026_competition"]
    pool = b1["sid"].numpy(); pool = pool[pool >= 0]                         # the exam bank's shapes, at its own rates
    FP = {int(k): R0.z[f"fp_{int(k)}"].astype(float) for k in np.unique(pool)}


def poly(k, x, y, yaw):
    q = FP[k]; c, s_ = np.cos(yaw), np.sin(yaw)
    return np.stack([x + q[:, 0] * c - q[:, 1] * s_, y + q[:, 0] * s_ + q[:, 1] * c], 1)


def lateral():
    while True:
        d = rng.normal(0.0, SIG)
        if abs(d) <= NEAR:
            return d


def place(s, k=None):
    """a box centred near the reference line at s: (sid, x, y, yaw, s, d), or None when 50 draws all come within 0.10 m of a wall"""
    j = int(np.argmin(np.abs(((refs - s + 0.5 * L) % L) - 0.5 * L)))
    k = int(rng.choice(pool)) if k is None else k
    for _ in range(50):
        d = lateral(); yaw = rng.uniform(0.0, np.pi)
        x, y = refp[j] + d * nrm[j]
        if R0.wall_dist(0, np.vstack([poly(k, x, y, yaw), [[x, y]]])).min() >= CLR_WALL:
            return (k, x, y, yaw, refs[j], d)
    return None


def stretch(k):
    """[m] a place in the middle 60 % of stretch k of the lap"""
    return (phase0 + (k + rng.uniform(0.2, 0.8)) * L / K) % L


def fill(lay, jb):
    """box jb of the layout at its place along the lap; a place where no draw clears the walls moves inside its own stretch"""
    while True:
        b = place(lay["s"][jb])
        if b is not None:
            lay["boxes"][jb] = b
            return
        st["moved"] += 1
        lay["s"][jb] = stretch(lay["slot"][jb])


# (11-18, "좀 더 균일하게") the lap cut into K = N x 3 equal stretches with one box in each: all the layouts' boxes together cover
# the lap evenly by construction. Layout i takes stretch i of the first third and, through two random permutations that move an
# index by at most N/8, one stretch of each later third -> its boxes are a third of a lap apart give or take a quarter of a third.
# A layout no car line gets through keeps its places: only the boxes in the way (as layout_bank.py finds them) are drawn again
# across the line, so where the boxes sit along the lap does not depend on which places were hard to pass; after 4 such rounds a
# box in the way moves inside its own stretch
st = dict(judged=0, failed=0, moved=0, redrawn=0)
K = N * N_BOX
phase0 = rng.uniform(0.0, L / K)


def near_perm(w):
    return np.argsort(np.arange(N) + rng.uniform(-w, w, N))


p1, p2 = near_perm(N / 8), near_perm(N / 8)
lays = []
for i in range(N):
    slot = np.array([i, N + p1[i], 2 * N + p2[p1[i]]])
    lays.append(dict(slot=slot, s=np.array([stretch(k) for k in slot]), boxes=[None] * N_BOX, rnd=0))
for lay in lays:
    for jb in range(N_BOX):
        fill(lay, jb)
pend, laps_ok, t0 = list(range(N)), {}, time.time()
while pend:
    batch, pend = pend[:NB], pend[NB:]
    lines, laps, clr, _ = builder.build([[poly(*b[:4]) for b in lays[i]["boxes"]] for i in batch])
    for i, xy, lp, (cp, cw) in zip(batch, lines, laps, clr):
        st["judged"] += 1
        kap = np.abs(uniform_filter1d(curvature(xy), 5, mode="wrap"))
        if cp >= CLR_PROP and cw >= CLR_WALL and kap.max() <= kmax and np.isfinite(lp):
            laps_ok[i] = float(lp)
            continue
        st["failed"] += 1
        lay = lays[i]; lay["rnd"] += 1
        P = [poly(*b[:4]) for b in lay["boxes"]]
        cpi = np.array([LL.check_line(xy, [p_], track)[0] for p_ in P])
        rm = cpi < CLR_PROP
        hot = xy[kap > kmax]
        if len(hot):
            cen = np.array([[b[1], b[2]] for b in lay["boxes"]])
            rm |= np.linalg.norm(cen[:, None] - hot[None], axis=-1).min(1) < 1.0
        if not rm.any():
            rm[int(np.argmin(cpi))] = True
        for jb in np.nonzero(rm)[0]:
            st["redrawn"] += 1
            if lay["rnd"] > 4:
                st["moved"] += 1
                lay["s"][jb] = stretch(lay["slot"][jb])
            fill(lay, jb)
        pend.append(i)
    print(f"  [{(time.time() - t0) / 60:4.1f} 분] 통과 {len(laps_ok)}/{N}, 판정 {st['judged']}, 못 지나감 {st['failed']}, "
          f"다시 놓은 상자 {st['redrawn']} (그중 트랙 따라 옮김 {st['moved']})", flush=True)
bank = [([b for b in lays[i]["boxes"]], laps_ok[i]) for i in range(N)]
bank = bank[:N]
sid = torch.full((N, C), -1, dtype=torch.long); pose = torch.zeros(N, C, 3)
for i, (lay, _) in enumerate(bank):
    for c, (k, x, y, yaw, _, _) in enumerate(lay):
        sid[i, c] = k; pose[i, c] = torch.tensor([x, y, yaw])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(dict(per_track={track.name: dict(sid=sid, pose=pose, lap=torch.tensor([lp for _, lp in bank]), length=float(env.sim.track.length[0]))},
                C=C, judge=dict(clr_prop=CLR_PROP, clr_wall=CLR_WALL, kmax=kmax, iters=ITERS, near=NEAR, sig=SIG, n_box=N_BOX,
                                refline=os.path.abspath(REFL), seed=SEED)), OUT)
print(f"== 은행 {OUT}: 배치 {N}개 x 상자 {N_BOX}개 | 판정 {st['judged']} 중 못 지나감 {st['failed']}, 다시 놓은 상자 {st['redrawn']}, 트랙 따라 옮긴 상자 {st['moved']}")

# figures: the distribution beside today's exam (the record's box sessions), and example layouts
s_all = np.array([b[4] for lay, _ in bank for b in lay]); d_all = np.array([b[5] for lay, _ in bank for b in lay])
xy_all = np.array([[b[1], b[2]] for lay, _ in bank for b in lay])
sp = []
for lay, _ in bank:
    s3 = np.sort([b[4] for b in lay]); sp.extend(np.diff(np.concatenate([s3, [s3[0] + L]])).tolist())
box_sess = np.nonzero((R0.sid[0] >= 0).any(-1) & ~R0.z["on"][:R0.T].any(0))[0]
cx = np.array([p for b in box_sess for p, k in zip(R0.pose[0, b, :, :2], R0.sid[0, b]) if k >= 0], float)
dd = np.linalg.norm(cx[:, None, :] - refp[None], axis=2); cj = dd.argmin(1)
c_s = refs[cj]; c_d = np.sign(((cx - refp[cj]) * nrm[cj]).sum(1)) * dd[np.arange(len(cx)), cj]
C_NEW, C_OLD, C_LINE = "#d95f02", "#8c8c8c", "#1f77b4"
fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 4, width_ratios=[1.05, 1.05, 1, 1])
for col, (xy_, c_, ttl) in enumerate(((cx, C_OLD, f"지금 시험: 상자 세션 {len(box_sess)}개에 {len(cx)}개"),
                                      (xy_all, C_NEW, f"견본: 배치 {N}개 x {N_BOX}개 ({len(xy_all)}개)"))):
    ax = fig.add_subplot(gs[:, col]); R0.draw_map(ax, 0)
    ax.plot(refp[:, 0], refp[:, 1], "-", color=C_LINE, lw=1.2, label="기준 길 (빈 트랙에서 달린 길)")
    ax.scatter(xy_[:, 0], xy_[:, 1], s=7, color=c_, alpha=0.6, label="상자 중심", zorder=3)
    ax.set_title(ttl, fontsize=10); ax.legend(loc="lower right", fontsize=8)
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(0, L, int(round(L / 2.0)) + 1)
ax.hist(c_s, bins, color=C_OLD, alpha=0.6, label="지금 시험", weights=np.full(len(c_s), 1.0 / len(box_sess)))
ax.hist(s_all, bins, color=C_NEW, alpha=0.75, label="견본", weights=np.full(len(s_all), 1.0 / N))
ax.axhline(N_BOX * 2.0 / L, color="k", ls=":", lw=1, label="완전히 고를 때")
ax.set_xlabel("트랙 따라 위치 [m] (출발선부터)"); ax.set_ylabel("배치(세션)당 상자 수 / 2 m"); ax.legend(fontsize=8); ax.set_title("트랙 어디에 놓이나", fontsize=10)
ax = fig.add_subplot(gs[0, 3])
lb = np.arange(-2.0, 2.05, 0.05)
ax.hist(np.clip(c_d, -2, 2), lb, color=C_OLD, alpha=0.6, label="지금 시험", density=True)
ax.hist(d_all, lb, color=C_NEW, alpha=0.75, label=f"견본 (표준편차 {SIG:.2f} m, ±{NEAR:.1f} m에서 자름)", density=True)
ax.set_xlabel("기준 길에서 상자 중심까지 [m] (+ = 왼쪽)"); ax.set_ylabel("비율"); ax.legend(fontsize=8); ax.set_title("길에서 얼마나 떨어져 놓이나", fontsize=10)
ax = fig.add_subplot(gs[1, 2])
ax.hist(np.abs(d_all), np.arange(0, NEAR + 0.025, 0.025), color=C_NEW, alpha=0.75, cumulative=True, density=True)
ax.set_xlabel("기준 길에서 상자 중심까지 |거리| [m]"); ax.set_ylabel("누적 비율"); ax.set_title(f"견본: 0.10 m 안 {100 * np.mean(np.abs(d_all) <= 0.1):.0f} %, 0.20 m 안 {100 * np.mean(np.abs(d_all) <= 0.2):.0f} %", fontsize=10)
ax = fig.add_subplot(gs[1, 3])
ax.hist(sp, np.arange(0, L + 1, 1.0), color=C_NEW, alpha=0.75)
ax.set_xlabel("한 배치 안 이웃 상자 사이 트랙 거리 [m]"); ax.set_ylabel("수"); ax.set_title(f"견본: 상자끼리 간격 (한 바퀴 {L:.1f} m)", fontsize=10)
fig.suptitle("시험 상자 배치: 지금 vs 견본 2 (한 바퀴 세 칸마다 하나, 길 위에 밀집, 지나갈 길 판정 통과) -- 아직 시험에 안 씀", fontsize=12)
fig.tight_layout(); fig.savefig(FIG + "_dist.png", dpi=80); print("그림", FIG + "_dist.png")
pick = np.linspace(0, N - 1, 8).round().astype(int)
fig, axs = plt.subplots(2, 4, figsize=(17, 13))
for ax, i in zip(axs.flat, pick):
    lay, lp = bank[i]
    R0.draw_map(ax, 0)
    ax.plot(refp[:, 0], refp[:, 1], "-", color=C_LINE, lw=1.0)
    for n_, (k, x, y, yaw, s, d) in enumerate(sorted(lay, key=lambda b: b[4])):
        P = poly(k, x, y, yaw)
        ax.fill(P[:, 0], P[:, 1], color="#a0522d", zorder=4)
        ax.add_patch(plt.Circle((x, y), 0.9, fill=False, color=C_NEW, lw=1.5, zorder=5))
        ax.text(x + 1.0, y, f"{n_ + 1}: {100 * abs(d):.0f} cm {'왼' if d > 0 else '오른'}쪽", fontsize=8, color="#7a2e00", zorder=6)
    ax.set_title(f"배치 {i}: 상자 {N_BOX}개 (판정 라인 한 바퀴 {lp:.1f} s)", fontsize=10)
fig.suptitle("견본 2 예시 배치 8개: 파란 선 = 기준 길, 갈색 = 상자(주황 원은 찾기 쉽게), 숫자 = 기준 길에서 상자 중심까지", fontsize=12)
fig.tight_layout(); fig.savefig(FIG + "_examples.png", dpi=70); print("그림", FIG + "_examples.png")
