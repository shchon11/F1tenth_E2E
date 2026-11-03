"""학습 상자 빈도 지도 (11-03, 사용자: "맵에 그냥 빈도수 map을 입히면 되잖어"): 체크포인트의 학습 인자 그대로(라인 위 상자 비율, 밀도,
개수 포함) 원본 트랙에서만 리셋을 거듭해 상자를 뽑고, 지도 칸(0.05 m)마다 상자에 덮인 판의 비율을 색으로 칠함. 상자 모양은 생성기의
반평면 그대로(회전 포함). 학습은 원본 + 뒤튼 트랙 16개지만 생성기는 같음.
인자: 체크포인트 환경수 리셋수 출력png 제목 [배치방식 pattern|uniform: 체크포인트 인자 대신]
(11-03 밤) 통과 시험: 첫 리셋의 판마다 상자를 지도 칸에 찍고 빈 곳을 0.20 m(차 반폭 0.15 + 0.05) 깎은 뒤, 시작선 단면을 막고 그 바로 뒤와
바로 앞이 같은 연결 덩어리인지 = 한 바퀴를 돌 길이 남았는지. 고른지: 상자가 놓인 영역에서 벽 0.3 m 안 면적 비율 대 상자가 덮은 양 비율
(같으면 고름). (11-04 사용자: 최소시간 라인 기준 '라인 막음'은 강화학습 정책이 그 라인을 안 따르니 의미 없음 -> 뺌.)"""
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
from scipy import ndimage

for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break

pck, n_env, n_rep, out, title = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
dev = torch.device("cuda")
ck = torch.load(pck, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device, a.warps = n_env, 13, "cuda", 0
if len(sys.argv) > 6:
    if os.path.exists(sys.argv[6]):
        a.prop_bank = sys.argv[6]
    else:
        a.prop_place = sys.argv[6]
place = str(getattr(a, "prop_place", "pattern") or "pattern")
env = SR.build_env(a, dev)
proc, tr = env.procedural, env.sim.track
place = proc.place
t0 = tr.tracks[0]
occ = np.asarray(t0.occupancy).astype(bool)
H, W = occ.shape
res, ox, oy = float(t0.resolution), float(t0.origin[0]), float(t0.origin[1])
R = int(np.ceil(0.5 / res))                                            # 패치 반경 칸 수 (상자 외접원 < 0.5 m)
g = torch.arange(-R, R + 1, device=dev)
pr, pc = torch.meshgrid(g, g, indexing="ij")
pr, pc = pr.reshape(-1), pc.reshape(-1)
cover = torch.zeros(H * W, device=dev, dtype=torch.float64)
T, N = tr.cl_s.shape
n_layout = 0
n_crate = 0
for rep in range(n_rep):
    env.reset()
    sid = proc.p_sid
    live = (sid >= 0) & (proc.p_zhi > proc.p_zlo)
    n_layout += int(live.any(1).sum())
    pose, nn, dd = proc.p_poses[live].float(), proc.p_n[live].float(), proc.p_d[live].float()
    n_crate += pose.shape[0]
    r0 = torch.floor((pose[:, 1] - oy) / res).long(); c0 = torch.floor((pose[:, 0] - ox) / res).long()
    rr, cc = r0[:, None] + pr[None], c0[:, None] + pc[None]              # (M, S)
    px, py = ox + (cc.float() + 0.5) * res - pose[:, :1], oy + (rr.float() + 0.5) * res - pose[:, 1:2]
    cy, sy = torch.cos(pose[:, 2:3]), torch.sin(pose[:, 2:3])
    lx, ly = cy * px + sy * py, -sy * px + cy * py                       # 상자 몸 좌표
    inside = ((nn[:, None, :, 0] * lx[..., None] + nn[:, None, :, 1] * ly[..., None]) <= dd[:, None, :]).all(-1)
    ok = inside & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    cover.index_add_(0, (rr * W + cc)[ok], torch.ones(int(ok.sum()), device=dev, dtype=torch.float64))
    if rep == 0:
        eid = torch.nonzero(live)[:, 0]
        occ_e = torch.zeros(n_env, H * W, dtype=torch.bool, device=dev)
        occ_e[eid[:, None].expand_as(rr)[ok], (rr * W + cc)[ok]] = True
        occ_e = occ_e.reshape(n_env, H, W).cpu().numpy()
        has = live.any(1).cpu().numpy()
freq = (cover / max(n_layout, 1)).reshape(H, W).cpu().numpy() * 100.0
cl0 = tr.cl[0].cpu().numpy(); tg0 = tr.cl_tangent[0].cpu().numpy()
L0 = float(tr.length[0])

# 통과 시험
rc = int(np.ceil(0.20 / res))
yy, xx = np.mgrid[-rc:rc + 1, -rc:rc + 1]
disk = (yy ** 2 + xx ** 2) <= rc ** 2
n0 = np.array([-tg0[0, 1], tg0[0, 0]])
wl0, wr0 = float(proc.lane_l[0, 0]), float(proc.lane_r[0, 0])
cut = [cl0[0] + n0 * q for q in np.arange(-wr0 - 0.3, wl0 + 0.3, res * 0.5)]
k_ = max(2, int(0.6 / (L0 / N)))


def _cells(i):
    nn_ = np.array([-tg0[i, 1], tg0[i, 0]])
    pts = [cl0[i] + nn_ * q for q in np.arange(-float(proc.lane_r[0, i]), float(proc.lane_l[0, i]), res)]
    return [(int((p[1] - oy) / res), int((p[0] - ox) / res)) for p in pts]


after, before = _cells(k_), _cells(N - k_)
n_try = n_ok = 0
bad = []
for e in range(n_env):
    if not has[e]:
        continue
    free = ~(occ | occ_e[e])
    for p in cut:
        r_, c_ = int((p[1] - oy) / res), int((p[0] - ox) / res)
        free[max(r_ - 1, 0):r_ + 2, max(c_ - 1, 0):c_ + 2] = False
    fe = ndimage.binary_erosion(free, structure=disk)
    lab, _ = ndimage.label(fe)
    la = {lab[r_, c_] for r_, c_ in after if 0 <= r_ < H and 0 <= c_ < W} - {0}
    lb = {lab[r_, c_] for r_, c_ in before if 0 <= r_ < H and 0 <= c_ < W} - {0}
    n_try += 1
    if la & lb:
        n_ok += 1
    else:
        bad.append(e)
print(f"== {title} ({place}): 통과 시험 판 {n_try}개 중 한 바퀴 길 남음 {n_ok}개 (막힌 판 {bad[:10]}), 시작선 차선 왼 {wl0:.2f} 오른 {wr0:.2f} m", flush=True)
edt = ndimage.distance_transform_edt(~occ) * res
reg = freq > 0
near = reg & (edt < 0.3)
print(f"   벽 0.3 m 안: 면적 {near.sum() / max(reg.sum(), 1):.0%}, 상자가 덮은 양 {freq[near].sum() / max(freq[reg].sum(), 1e-9):.0%}", flush=True)
np.save(out.replace(".png", "_freq.npy"), freq)
ys, xs = np.where(~occ)
pad = 0.6
fig, ax = plt.subplots(figsize=(13, 11))
ext = (ox, ox + W * res, oy, oy + H * res)
ax.imshow(np.where(occ, 0.45, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext)
vmax = float(np.percentile(freq[freq > 0], 99.5))
im = ax.imshow(np.ma.masked_less_equal(freq, 0.0), origin="lower", cmap="turbo", vmin=0, vmax=vmax, extent=ext, alpha=0.9)
ax.plot(*cl0[0], "^", c="lime", mec="k", ms=12, label="시작선")
for fr in np.arange(0.1, 1, 0.1):
    i = int(fr * N)
    ax.annotate(f"{fr * L0:.0f} m", cl0[i], fontsize=10, color="k", ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax.set_xlim(ox + xs.min() * res - pad, ox + xs.max() * res + pad)
ax.set_ylim(oy + ys.min() * res - pad, oy + ys.max() * res + pad)
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label="이 칸이 상자에 덮이는 판의 비율 [%]")
ax.set_title(f"{title}: 학습 상자 빈도 지도 ({os.path.basename(os.path.dirname(pck))} 학습 인자, "
             f"{'경로 생성기로 판정한 균일 배치 은행' if place == 'bank' else '균일 배치' if place == 'uniform' else '무늬 배치, 라인 위 상자 ' + format(float(getattr(a, 'prop_line', 0) or 0), '.0%')})\n상자 있는 판 {n_layout}개, 상자 {n_crate}개 (판당 {n_crate / max(n_layout, 1):.1f}개)",
             fontsize=13)
ax.legend(loc="upper right")
fig.savefig(out, dpi=100, bbox_inches="tight")
print(f"== {title}: 판 {n_layout}, 상자 {n_crate}, 칸 최대 {freq.max():.2f} %, 그림 {out}", flush=True)
