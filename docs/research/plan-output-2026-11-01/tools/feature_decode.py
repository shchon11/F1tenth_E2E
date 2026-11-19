"""(11-19, 사용자: "그 feature space에서 decoder로 맵과 pose를 복원할 수 있는지를 학습해볼래") policy 안쪽 값에서 차의 pose와
둘레 지도를 되살리는 decoder를 학습해, 신경망이 트랙(지도)을 외우고 있는지 본다. policy는 그대로(얼림), decoder만 배움.
입력: F1SIM_EXAM_SAVE_MEM=1 시험 기록(npz + _mem.npz): 결정마다 feat(스캔 인코더 출력 + 정규화한 상태 = 기억과 몸통이 읽는 것),
mem(그 결정 뒤 기억 상태). 정답(분석에만 씀, policy 입력 아님): 그 결정의 pose(x, y, yaw), 차 기준 둘레 지도(앞 -2..8 m, 옆 ±5 m,
0.25 m 칸, 벽이면 1).
- decoder 셋: feat만 / mem만 / feat+mem. 2층 MLP(512), pose(x, y, cos, sin) + 지도 칸 1600개(BCE).
- 세션으로 나눔: 4번째 세션마다 시험용(32), 나머지 학습용(96). 같은 세션의 이웃 결정은 거의 같아서 결정 단위로 나누면 부풀려짐.
- 지도 칸은 지금 스캔에 보이는 칸 / 안 보이는 칸(벽 뒤, 270° 밖)으로 나눠 맞힘을 따로 냄. 안 보이는 벽까지 되살리면 차가 어디인지
  알고 그 자리의 지도를 아는 것. 단, decoder도 한 지도에서 배우므로 지도 자체는 decoder가 외울 수 있음 -> 핵심은 pose가 되살아나는가.
인자: 기록npz 출력png앞머리 [결정 간격 3]"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec, plt

rec, OUT = sys.argv[1], sys.argv[2]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 3
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
R = Rec(rec); z = R.z
M = np.load(rec[:-4] + "_mem.npz")
feat, mem = M["feat"].astype(np.float32), M["mem"].astype(np.float32)          # (T, B, F), (T, B, H)
T, B = feat.shape[:2]
st = R.st[:T].astype(np.float64)                                                 # pose each decision was made from
ok = ~R.rs[:T].astype(bool)
ok[:10] = False                                                                  # the first second: memory still empty
tt, bb = np.nonzero(ok)
sel = (tt % STRIDE) == 0
tt, bb = tt[sel], bb[sel]
test = (bb % 4) == 0
print(f"결정 {len(tt)}개 (학습 {int((~test).sum())}, 시험 {int(test.sum())}) | feat {feat.shape[2]} · mem {mem.shape[2]}", flush=True)

# --- targets: pose and the local wall map in the car's frame
X, Y, YAW = st[tt, bb, 0], st[tt, bb, 1], st[tt, bb, 2]
g, ext = R.grid(0)
x0, x1, y0, y1 = ext
xc = np.arange(-2.0, 8.0, 0.25) + 0.125; yc = np.arange(-5.0, 5.0, 0.25) + 0.125
GX, GY = np.meshgrid(xc, yc, indexing="ij")                                    # (40, 40) car frame: x ahead, y left
cell = np.stack([GX.ravel(), GY.ravel()], 1)                                    # (1600, 2)
NC = cell.shape[0]


def world(xs, ys, yaws, pts):
    c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
    return np.stack([xs[:, None] + pts[None, :, 0] * c - pts[None, :, 1] * s,
                     ys[:, None] + pts[None, :, 0] * s + pts[None, :, 1] * c], -1)


occ = np.zeros((len(tt), NC), np.float32)
for i in range(0, len(tt), 2000):
    w = world(X[i:i + 2000], Y[i:i + 2000], YAW[i:i + 2000], cell).reshape(-1, 2)
    occ[i:i + 2000] = (R.wall_dist(0, w) <= 0.0).reshape(-1, NC)
# visibility from the LiDAR (nose, 270 deg, 10 m): a cell is seen if no wall lies between the sensor and it
MOUNT = 0.12555 + R.hl * 0.0
ang = np.deg2rad(np.arange(-135.0, 135.01, 1.0)); dr = np.arange(0.05, 10.0, 0.05)
sens = np.array([MOUNT, 0.0])
cell_rel = cell - sens; cell_r = np.hypot(cell_rel[:, 0], cell_rel[:, 1]); cell_a = np.arctan2(cell_rel[:, 1], cell_rel[:, 0])
ia = np.clip(np.round(np.rad2deg(cell_a) + 135.0).astype(int), 0, len(ang) - 1)
infov = np.abs(cell_a) <= np.deg2rad(135.0)
vis = np.zeros((len(tt), NC), bool)
ti = np.nonzero(test)[0]
for i in range(0, len(ti), 500):
    k = ti[i:i + 500]
    ray_pts = np.stack([sens[0] + dr[None, :] * np.cos(ang)[:, None], sens[1] + dr[None, :] * np.sin(ang)[:, None]], -1).reshape(-1, 2)
    w = world(X[k], Y[k], YAW[k], ray_pts).reshape(-1, 2)
    hit = (R.wall_dist(0, w) <= 0.0).reshape(len(k), len(ang), len(dr))
    rng = np.where(hit.any(-1), dr[hit.argmax(-1)], 10.0)                       # (n, angles) range to the first wall
    vis[k] = infov[None, :] & (cell_r[None, :] <= rng[:, ia] + 0.15)
print(f"목표 만듦: 지도 칸 중 벽 {occ.mean() * 100:.0f} % | 시험 결정의 칸 중 지금 보이는 칸 {vis[test].mean() * 100:.0f} %", flush=True)

# --- decoders
pose_t = np.stack([(X - x0) / (x1 - x0), (Y - y0) / (y1 - y0), np.cos(YAW), np.sin(YAW)], 1).astype(np.float32)
inputs = {"feat": feat[tt, bb], "mem": mem[tt, bb], "feat+mem": np.concatenate([feat[tt, bb], mem[tt, bb]], 1)}


class Dec(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.pose = nn.Linear(512, 4); self.map = nn.Linear(512, NC)

    def forward(self, x):
        h = self.body(x)
        return self.pose(h), self.map(h)


res = {}
for name, Xin in inputs.items():
    mu, sd = Xin[~test].mean(0), Xin[~test].std(0) + 1e-4
    xin = torch.tensor((Xin - mu) / sd, device=dev)
    pt, mt = torch.tensor(pose_t, device=dev), torch.tensor(occ, device=dev)
    tr = torch.tensor(np.nonzero(~test)[0], device=dev); te = torch.tensor(np.nonzero(test)[0], device=dev)
    net = Dec(xin.shape[1]).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    t0 = time.time()
    for ep in range(60):
        for idx in tr[torch.randperm(len(tr), device=dev)].split(1024):
            p, m = net(xin[idx])
            loss = 20.0 * ((p - pt[idx]) ** 2).mean() + nn.functional.binary_cross_entropy_with_logits(m, mt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p, m = net(xin[te])
    p, m = p.cpu().numpy(), torch.sigmoid(m).cpu().numpy()
    k = test.nonzero()[0]
    px, py = x0 + p[:, 0] * (x1 - x0), y0 + p[:, 1] * (y1 - y0)
    e = np.hypot(px - X[k], py - Y[k])
    eyaw = np.rad2deg(np.abs(np.angle(np.exp(1j * (np.arctan2(p[:, 3], p[:, 2]) - YAW[k])))))
    yh = m > 0.5; yt = occ[k] > 0.5; vv = vis[k]
    def iou(msk):
        a, b = yh & msk, yt & msk
        return (a & b).sum() / max((a | b).sum(), 1)
    res[name] = dict(px=px, py=py, e=e, eyaw=eyaw, m=m)
    print(f"[{name}] 위치 오차 중앙 {np.median(e):.2f} m · 90 % {np.percentile(e, 90):.2f} m · 1 m 넘는 결정 {np.mean(e > 1.0) * 100:.1f} % | "
          f"방향 오차 중앙 {np.median(eyaw):.1f}° | 벽 칸 IoU: 보이는 칸 {iou(vv):.2f} · 안 보이는 칸 {iou(~vv):.2f} | {time.time() - t0:.0f} s", flush=True)
# a reference for the map: the average patch of the training decisions (what a decoder gives knowing nothing of the place)
k = test.nonzero()[0]; avg = occ[~test].mean(0) > 0.5
for msk_name, vv in (("보이는 칸", vis[k]), ("안 보이는 칸", ~vis[k])):
    a = np.broadcast_to(avg, occ[k].shape) & vv; b = (occ[k] > 0.5) & vv
    print(f"[기준: 평균 지도] 벽 칸 IoU {msk_name} {(a & b).sum() / max((a | b).sum(), 1):.2f}")

# --- figures
fig, axs = plt.subplots(1, 3, figsize=(17, 9))
for ax, name in zip(axs, inputs):
    r = res[name]
    R.draw_map(ax, 0)
    sc = ax.scatter(r["px"], r["py"], c=np.clip(r["e"], 0, 2.0), cmap="RdYlGn_r", s=3, vmin=0, vmax=2.0)
    ax.set_title(f"{name}에서 되살린 위치 (시험 세션)\n중앙 오차 {np.median(r['e']):.2f} m · 90 % {np.percentile(r['e'], 90):.2f} m", fontsize=11)
fig.colorbar(sc, ax=axs, shrink=0.5, label="위치 오차 [m]")
fig.savefig(OUT + "_pose.png", dpi=100, bbox_inches="tight"); print(OUT + "_pose.png")
# local maps: a few test decisions spread over the lap, the ones with the most hidden wall first
k = test.nonzero()[0]
hidden_wall = ((occ[k] > 0.5) & ~vis[k]).sum(1)
pick = np.argsort(-hidden_wall)[:: max(1, len(k) // 400)][:6]
best = "feat+mem"
fig, axs = plt.subplots(len(pick), 3, figsize=(10, 3.3 * len(pick)))
for row, j in enumerate(pick):
    tru = occ[k[j]].reshape(40, 40); dec = res[best]["m"][j].reshape(40, 40); vsb = vis[k[j]].reshape(40, 40)
    ext_c = [-5.0, 5.0, -2.0, 8.0]
    axs[row, 0].imshow(tru[:, ::-1], origin="lower", cmap="Greys", extent=ext_c, vmin=0, vmax=1)
    axs[row, 0].contour(np.linspace(5, -5, 40), np.linspace(-2, 8, 40), vsb.astype(float), levels=[0.5], colors="tab:blue", linewidths=1)
    axs[row, 1].imshow(dec[:, ::-1], origin="lower", cmap="Greys", extent=ext_c, vmin=0, vmax=1)
    axs[row, 2].imshow(np.where(vsb, np.nan, (dec > 0.5) != (tru > 0.5))[:, ::-1], origin="lower", cmap="Reds", extent=ext_c, vmin=0, vmax=1)
    for c in range(3):
        axs[row, c].plot([0], [0], "^", c="tab:orange", ms=8); axs[row, c].set_xticks([]); axs[row, c].set_yticks([])
    axs[row, 0].set_ylabel(f"시험 결정 {j}\n위치 오차 {res[best]['e'][j]:.2f} m", fontsize=9)
axs[0, 0].set_title("정답 둘레 지도 (검정 = 벽)\n파란 선 안 = 지금 스캔에 보이는 곳", fontsize=10)
axs[0, 1].set_title(f"{best}에서 되살린 지도", fontsize=10)
axs[0, 2].set_title("안 보이는 칸 중 틀린 칸 (빨강)", fontsize=10)
fig.suptitle("차 기준 둘레 지도 (차 = 주황 세모, 위 = 차 앞쪽 8 m, 좌우 ±5 m)", fontsize=11)
fig.savefig(OUT + "_map.png", dpi=100, bbox_inches="tight"); print(OUT + "_map.png")
