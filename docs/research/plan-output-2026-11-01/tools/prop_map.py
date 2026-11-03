"""학습 때 뽑히는 상자 전체 위치 분포 (11-03, 사용자: "학습때 뽑힐 상자 전체 위치분포를 시각화해서 보여줘"): 학습은 상자 위치를 남기지
않으므로 체크포인트의 학습 인자 그대로 환경을 만들어 리셋을 거듭하며 뽑힌 상자를 모두 모음. 두 설정 -- (현재) 체크포인트의 --prop-line
(라인 위 상자 비율), (라인 상자 없음) 0 = 경주 단계와 11-03 이전 학습. 상자마다 자기 트랙에서 s, 가로 위치(왼쪽 +), 그 자리 왼/오른 벽까지
거리, 최소시간 레이싱 라인 가로 위치. 그림: (위) 원본 트랙(트랙 0)에 뽑힌 상자 중심 -- 좌표 그대로, (가운데) 펼친 트랙: 가로 = 랩 위치,
세로 = 차선 폭 안의 위치(0 오른쪽 벽, 1 왼쪽 벽), 색 = 상자 수, 흰 선 = 레이싱 라인 -- 뒤튼 트랙 16개 포함 전부, (아래) 라인에서 상자 중심까지
가로 거리, 상자 가장자리에서 가까운 벽까지 틈, 랩 2 % 칸마다 라인 막는 상자(|상자-라인| < 상자 반크기 + 0.155).
인자: 체크포인트 환경수 리셋수 출력png 제목"""
import os, sys, argparse
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR

for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break

pck, n_env, n_rep, out, title = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
dev = torch.device("cuda")
ck = torch.load(pck, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
line_now = float(getattr(a, "prop_line", 0.0) or 0.0)
a.envs, a.seed, a.device = n_env, 11, "cuda"
env = SR.build_env(a, dev)
proc, tr = env.procedural, env.sim.track
T, N = tr.cl_s.shape
rl_all = SR._line_planner(env).rl_offset(tr.cl_s, torch.arange(T, device=dev)[:, None].expand_as(tr.cl_s))  # (T, N)
print(f"== {title}: 트랙 {T}개, 길이 " + " ".join(f"{float(x):.1f}" for x in tr.length) + f", 상자 있는 환경 비율 {a.prop_fraction}", flush=True)


def collect(share):
    proc.line_share = share
    rows = []
    for rep in range(n_rep):
        env.reset()
        sid, pose, dd = proc.p_sid, proc.p_poses, proc.p_d
        ext = torch.where(torch.isfinite(dd), dd, torch.zeros_like(dd)).amax(-1)
        live = (sid >= 0) & (proc.p_zhi > proc.p_zlo)
        B, C = sid.shape
        tid = env.sim.tid[:, None].expand(B, C)
        xy, tid, ext = pose[..., :2][live].float(), tid[live], ext[live]
        s, lat, idx = tr.project(xy, tid)
        idx = idx.long() % N
        wl, wr, rl = proc.lane_l[tid, idx], proc.lane_r[tid, idx], rl_all[tid, idx]
        rows.append(torch.stack([tid.float(), s / tr.length[tid], lat, wl, wr, rl, ext, xy[:, 0], xy[:, 1]], 1).cpu())
    r = torch.cat(rows).numpy()
    print(f"  라인 상자 비율 {share}: 상자 {len(r)}개 ({n_env} 환경 x {n_rep} 리셋)", flush=True)
    return r


sets = [(f"현재 학습 (라인 위 상자 {line_now:.0%})", collect(line_now)), ("라인 상자 없음 (경주 단계·이전 학습)", collect(0.0))]

t0 = tr.tracks[0]
occ = np.asarray(t0.occupancy)
res, ox, oy = float(t0.resolution), float(t0.origin[0]), float(t0.origin[1])
cl0 = tr.cl[0].cpu().numpy(); tg0 = tr.cl_tangent[0].cpu().numpy()
nrm0 = np.stack([-tg0[:, 1], tg0[:, 0]], 1)
rl0 = rl_all[0].cpu().numpy()
line0 = cl0 + rl0[:, None] * nrm0
L0 = float(tr.length[0])
ys, xs = np.where(~occ)
pad = 1.0
box = (ox + xs.min() * res - pad, ox + xs.max() * res + pad, oy + ys.min() * res - pad, oy + ys.max() * res + pad)

fig = plt.figure(figsize=(18, 23))
gs = fig.add_gridspec(5, 6, height_ratios=[1.35, 0.8, 0.8, 0.75, 0.75], hspace=0.38, wspace=0.5)
fig.suptitle(f"{title}: 학습 때 뽑히는 상자 위치 분포 ({os.path.basename(os.path.dirname(pck))} 학습 인자 그대로, 리셋마다 새로 뽑음)", fontsize=16)
axs_map = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])]
for ax, (name, r) in zip(axs_map, sets):
    ax.imshow(occ, origin="lower", cmap="Greys", vmin=0, vmax=1.6, extent=(ox, ox + occ.shape[1] * res, oy, oy + occ.shape[0] * res))
    m = r[:, 0] == 0
    ax.scatter(r[m, 7], r[m, 8], s=7, c="tab:red", alpha=0.3, lw=0)
    ax.plot(line0[:, 0], line0[:, 1], "-", c="tab:blue", lw=1.2, label="레이싱 라인")
    ax.plot(*cl0[0], "k^", ms=9)
    for fr in np.arange(0, 1, 0.1):
        i = int(fr * N)
        ax.annotate(f"{fr * L0:.0f}", cl0[i], fontsize=8, color="green")
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
    ax.set_title(f"{name}\n원본 트랙에 뽑힌 상자 중심 {int(m.sum())}개 (점 하나 = 상자 하나, 진할수록 자주), 초록 = 랩 위치 m, ▲ 시작", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
# 펼친 트랙: 랩 위치 x 차선 안 위치
nb_s, nb_u = 100, 24
s_grid = np.arange(N) / N
u_rl0 = (rl0 + proc.lane_r[0].cpu().numpy()) / (proc.lane_l[0] + proc.lane_r[0]).cpu().numpy()
for k, (name, r) in enumerate(sets):
    ax = fig.add_subplot(gs[1 + k, :])
    u = np.clip((r[:, 2] + r[:, 4]) / (r[:, 3] + r[:, 4]), 0, 1)
    h, _, _ = np.histogram2d(r[:, 1], u, bins=(nb_s, nb_u), range=((0, 1), (0, 1)))
    im = ax.imshow(h.T, origin="lower", aspect="auto", extent=(0, L0, 0, 1), cmap="magma")
    ax.plot(s_grid * L0, u_rl0, "-", c="white", lw=1.6, label="레이싱 라인 (원본 트랙)")
    ax.set_ylabel("차선 안 위치\n0 = 오른쪽 벽, 1 = 왼쪽 벽")
    ax.set_xlabel("랩 위치 [m] (원본 트랙 길이로 환산, 뒤튼 트랙 16개 포함)")
    ax.set_title(f"{name}: 펼친 트랙 위 상자 수 (칸 = 랩 1 % x 차선 폭 1/24)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01, label="상자 수")
# 아래: 라인에서 거리, 벽 틈, 라인 막는 상자
cols = ["tab:red", "tab:gray"]
ax1 = fig.add_subplot(gs[3, 0:2]); ax2 = fig.add_subplot(gs[3, 2:4]); ax3 = fig.add_subplot(gs[3, 4:6])
ax4 = fig.add_subplot(gs[4, :])
stats = []
for (name, r), c in zip(sets, cols):
    d = r[:, 2] - r[:, 5]
    blk = np.abs(d) < r[:, 6] + 0.155
    gap = np.minimum(r[:, 3] - (r[:, 2] + r[:, 6]), (r[:, 2] - r[:, 6]) + r[:, 4])
    u = np.clip((r[:, 2] + r[:, 4]) / (r[:, 3] + r[:, 4]), 0, 1)
    ax1.hist(d, bins=np.linspace(-2.5, 2.5, 101), histtype="step", lw=1.6, color=c, density=True, label=name)
    ax2.hist(gap, bins=np.linspace(-0.2, 2.5, 55), histtype="step", lw=1.6, color=c, density=True, label=name)
    ax3.hist(u, bins=np.linspace(0, 1, 25), histtype="step", lw=1.6, color=c, density=True, label=name)
    hb, _ = np.histogram(r[blk, 1], bins=50, range=(0, 1))
    ha, _ = np.histogram(r[:, 1], bins=50, range=(0, 1))
    n_layout = n_env * n_rep * float(a.prop_fraction)
    ax4.step(np.arange(50) * 2 * L0 / 100, hb / n_layout * 100, where="post", color=c, lw=1.8, label=f"{name}: 라인 막는 상자")
    ax4.step(np.arange(50) * 2 * L0 / 100, ha / n_layout * 100, where="post", color=c, lw=0.8, ls="--", label=f"{name}: 상자 전체")
    stats.append((name, len(r), blk.mean(), (gap < 0.3).mean(), np.median(np.abs(d)), (np.abs(d) < 0.5).mean()))
ax1.axvspan(-0.45, 0.45, color="tab:blue", alpha=0.08)
ax1.set_title("레이싱 라인에서 상자 중심까지 가로 거리\n(+ = 라인 왼쪽, 파란 띠 ~ 라인 막음)", fontsize=11); ax1.set_xlabel("[m]")
ax2.set_title("상자 가장자리에서 가까운 벽까지 틈", fontsize=11); ax2.set_xlabel("[m]")
ax3.set_title("차선 안 위치 (0 오른쪽 벽, 1 왼쪽 벽)", fontsize=11)
ax4.set_title("랩 2 % 칸마다 배치 100개당 상자 수 (굵은 선 = 라인 막는 상자, 점선 = 전체)", fontsize=11)
ax4.set_xlabel("랩 위치 [m] (원본 트랙 길이로 환산)"); ax4.set_ylabel("배치 100개당")
for ax in (ax1, ax2, ax3, ax4):
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.savefig(out, dpi=90, bbox_inches="tight")
for name, n, b, w, md, near in stats:
    print(f"  {name}: 상자 {n}개, 라인 막음 {b:.0%}, 벽 틈 0.3 m 미만 {w:.0%}, 라인에서 중앙 {md:.2f} m, 0.5 m 안 {near:.0%}", flush=True)
print(f"  그림 {out}", flush=True)
