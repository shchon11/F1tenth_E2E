"""차는 막히는데 LiDAR에는 안 보이는 벽 (11-12: 벽에 서 버린 차 6대가 모두 코앞 벽을 5 m로 읽음 -> 그 자리 벽 칸이 덕트도 높은 벽도 아님).
학습 세계의 트랙마다: 차가 닿는 벽의 경계 칸(막힌 칸 가운데 달리는 구역과 붙은 칸)을 LiDAR가 보는 종류(덕트 / 높은 벽)와 안 보는 종류로 나눠
길이를 세고, 안 보이는 구간(바로 뒤 0.1 m 안에도 보이는 칸이 없는 것)을 이어진 덩어리로 묶어 폭을 적음.
기록(drive_record.py npz)을 주면: 벽에 세게 닿은 건·벽에 닿은 건·서 있는 시간(0.3 m/s 아래)을 안 보이는 벽 0.5 m 안 / 그 밖으로 나누고,
그 자리에 머문 시간 비율과 견줌. 그림: 트랙 지도에 보이는 벽(진회색)·안 보이는 벽(빨강), 세게 닿은 자리(x), 2 s 넘게 서 있던 자리(원).
환경 변수 WV_SCALES=1 이면 크기 변형(state_rl.TRACK_SCALES_25)까지 넣은 33개 트랙을 봄.
인자: 그림 png [기록 npz]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"
import numpy as np, torch
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_lib  # noqa: F401  (한글 글꼴)
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR

out_png = sys.argv[1]
z = np.load(sys.argv[2], allow_pickle=False) if len(sys.argv) > 2 else None
dev = torch.device("cuda")
OV = {"episode_s": 45.0}
if os.environ.get("WV_SCALES"):
    OV["track_scales"] = SR.TRACK_SCALES_25
env, lim, world = PR.build(PR.WORLD, 64, 4242, dev, fmt="v2", overrides=OV)
names = list(getattr(env, "_track_names", []))
tr = env.sim.track
nT = len(tr.t_H)
NEAR = 0.5                                                             # [m] nose to an unseen wall cell: "at an unseen wall"
M = {}
rows = []
for t in range(nT):
    H, W, off = int(tr.t_H[t]), int(tr.t_W[t]), int(tr.t_off[t])
    res = float(tr.t_res[t])
    g = lambda f: f[off: off + H * W].view(H, W).float().cpu().numpy()
    occ = g(tr.edt) <= 0.0; vis = (g(tr.edt_duct) <= 0.0) | (g(tr.edt_tall) <= 0.0)
    lab, n = ndi.label(~occ)
    big = lab == (1 + int(np.argmax(ndi.sum(~occ, lab, range(1, n + 1)))))   # the region the cars drive in
    edge = occ & ndi.binary_dilation(big, structure=np.ones((3, 3), bool))
    # a boundary cell with a seen cell within 0.1 m behind it still stops the beam about where the car stops
    unseen = edge & ~ndi.binary_dilation(vis, structure=np.ones((3, 3), bool), iterations=max(2, int(round(0.1 / res))))
    lab2, n2 = ndi.label(unseen, structure=np.ones((3, 3), bool))
    sizes = ndi.sum(unseen, lab2, range(1, n2 + 1)) * res if n2 else np.array([])
    d_un = ndi.distance_transform_edt(~unseen) * res if unseen.any() else np.full((H, W), 1e3)
    M[t] = dict(occ=occ, edge=edge, unseen=unseen, big=big, d_un=d_un, H=H, W=W,
                ox=float(tr.t_origin[t, 0]), oy=float(tr.t_origin[t, 1]), res=float(tr.t_res[t]))
    rows.append((t, edge.sum() * res, unseen.sum() * res, n2, sizes))
print("트랙 | 격자 한 칸[cm] | 벽 길이[m] | 안 보이는 벽[m] | 덩어리 수 | 덩어리 폭[m] (큰 것부터)")
for t, e, u, n2, sizes in rows:
    print(f"{t:3d} {(names[t].split(':')[-1] if t < len(names) else ''):<34} | {100 * M[t]['res']:4.1f} | {e:6.1f} | {u:5.1f} ({100 * u / e:4.1f} %) | {n2:3d} | " + " ".join(f"{s:.2f}" for s in sorted(sizes, reverse=True)[:10]))
print(f"합계: 벽 {sum(r[1] for r in rows):.0f} m 중 안 보이는 벽 {sum(r[2] for r in rows):.1f} m, 안 보이는 벽이 있는 트랙 {sum(r[2] > 0.3 for r in rows)}개 / {nT}")

hits, stands = {}, {}
if z is not None:
    st = z["st"].astype(float); T = z["cnt"].shape[0]; tid = z["tid"][:T]; dt = float(z["dt"])
    cnt = z["cnt"].astype(float)
    d = np.clip(np.diff(np.concatenate([np.zeros((1,) + cnt.shape[1:]), cnt], 0), axis=0), 0, None)
    step = np.linalg.norm(st[1:T + 1, :, :2] - st[:T, :, :2], axis=2)
    d[step > 2.0] = 0.0
    nx = st[:T, :, 0] + 0.29 * np.cos(st[:T, :, 2]); ny = st[:T, :, 1] + 0.29 * np.sin(st[:T, :, 2])
    near = np.zeros((T, st.shape[1]), bool)
    for t in range(nT):
        m = tid == t
        if not m.any():
            continue
        c = np.clip(np.round((nx[m] - M[t]["ox"]) / M[t]["res"]).astype(int), 0, M[t]["W"] - 1)
        r = np.clip(np.round((ny[m] - M[t]["oy"]) / M[t]["res"]).astype(int), 0, M[t]["H"] - 1)
        near[m] = M[t]["d_un"][r, c] < NEAR
    stand = np.abs(st[:T, :, 3]) < 0.3
    tot_min = T * st.shape[1] * dt / 60.0
    print(f"\n기록 {sys.argv[2].split('/')[-1]}: {st.shape[1]}대 x {T * dt:.0f} s")
    print(f"안 보이는 벽 {NEAR} m 안에 코가 있던 시간: {100 * near.mean():.2f} %")
    for name, x in (("벽에 닿음(전체)", d[:, :, 0]), ("벽에 세게 닿음", d[:, :, 3])):
        print(f"{name}: {int(x.sum())}건 중 안 보이는 벽에서 {int(x[near].sum())}건 ({100 * x[near].sum() / max(x.sum(), 1):.0f} %) | "
              f"분당 안 보이는 벽 곁 {x[near].sum() / max(near.sum() * dt / 60, 1e-9):.2f} / 그 밖 {x[~near].sum() / ((~near).sum() * dt / 60):.3f}")
    print(f"서 있는 시간(0.3 m/s 아래): 전체 {stand.sum() * dt:.0f} s ({100 * stand.mean():.2f} %) 중 안 보이는 벽 곁 {(stand & near).sum() * dt:.0f} s "
          f"({100 * (stand & near).sum() / max(stand.sum(), 1):.0f} %)")
    n20 = int(round(2.0 / dt))
    long_stand = np.zeros_like(stand)
    run = np.zeros(st.shape[1], int)
    for k in range(T):
        run = np.where(stand[k], run + 1, 0); long_stand[k] = run == n20
    print(f"2 s 넘게 선 건: {int(long_stand.sum())}건 중 안 보이는 벽 곁 {int((long_stand & near).sum())}건")
    for t in range(nT):
        m = (tid == t)
        hk, hb = np.nonzero((d[:, :, 3] > 0) & m); hits[t] = (nx[hk, hb], ny[hk, hb], near[hk, hb])
        sk, sb = np.nonzero(long_stand & m); stands[t] = (nx[sk, sb], ny[sk, sb], near[sk, sb])

ncol = 6; nrow = int(np.ceil((nT + 1) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow)); axs = axs.ravel()
for ax in axs[nT:]:
    ax.axis("off")
for t in range(nT):
    m = M[t]; ax = axs[t]; H, W = m["H"], m["W"]; res = m["res"]
    img = np.ones((H, W, 3)); img[m["occ"]] = (0.93, 0.93, 0.93); img[m["edge"]] = (0.35, 0.35, 0.35)
    img[ndi.binary_dilation(m["unseen"], iterations=2)] = (0.9, 0.0, 0.0)
    ax.imshow(img, origin="lower", extent=(m["ox"] - res / 2, m["ox"] + (W - 0.5) * res, m["oy"] - res / 2, m["oy"] + (H - 0.5) * res))
    if t in hits:
        x, y, nr = hits[t]; ax.plot(x[~nr], y[~nr], "x", color="#555555", ms=6, mew=1.5); ax.plot(x[nr], y[nr], "x", color="#1f5fbf", ms=9, mew=2.5)
        x, y, nr = stands[t]; ax.plot(x[~nr], y[~nr], "o", mfc="none", mec="#2a9d3a", ms=9, mew=1.8); ax.plot(x[nr], y[nr], "o", mfc="none", mec="#1f5fbf", ms=12, mew=2.5)
    ys, xs = np.nonzero(m["big"])
    ax.set_xlim(m["ox"] + (xs.min() - 15) * res, m["ox"] + (xs.max() + 15) * res); ax.set_ylim(m["oy"] + (ys.min() - 15) * res, m["oy"] + (ys.max() + 15) * res)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"트랙 {t}{' (원본)' if t == 0 else ''}: 벽 {rows[t][1]:.0f} m 중 안 보임 {rows[t][2]:.1f} m ({rows[t][3]}곳)", fontsize=14)
axs[nT].axis("off")
axs[nT].text(0.0, 0.9, "진회색 = 벽\n빨강 = 차는 막히는데 LiDAR에는 안 보이는 벽\nx = 벽에 세게 닿은 자리 (파랑: 안 보이는 벽)\n"
             "원 = 2 s 넘게 선 자리 (파랑: 안 보이는 벽 곁, 초록: 그 밖)", fontsize=15, va="top", transform=axs[nT].transAxes)
fig.tight_layout(); fig.savefig(out_png, dpi=70)
print("그림", out_png)
