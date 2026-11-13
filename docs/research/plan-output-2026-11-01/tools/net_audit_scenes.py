"""구조 점검의 장면 그림: 상자가 가는 길 위에 보이는 결정에서, 그 순간 policy가 낸 plan(1.0 s)과 차가 실제로 그 뒤 간 길을 한 장면에 겹쳐 그림.
net_audit.py가 고른 결정(_pairs.npz) 가운데 거리 묶음 하나에서 아무렇게나 8개. 차 기준 좌표(앞 = 위쪽).
인자: checkpoint 기록npz _pairs.npz 출력png [묶음 번호(0~4), 기본 2 = 2.5~4 m]"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt
from policy_lib import limits_for, dev
from f1sim.learn.plan_policy import load_actor2
from f1sim import plan_steps as PS

ck_path, npz, pairs, out = sys.argv[1:5]
bin_i = int(sys.argv[5]) if len(sys.argv) > 5 else 2
BL = ["0.8~1.5 m", "1.5~2.5 m", "2.5~4 m", "4~5.5 m", "5.5~7 m"]
ck = torch.load(ck_path, map_location=dev, weights_only=False)
lim = limits_for(ck); actor = load_actor2(ck, dev, check_mount=False).eval()
R = Rec(npz); z = R.z
mm = np.load(npz[:-4] + "_mem.npz"); sc = np.load(npz[:-4] + "_scan.npz")
F = mm["feat"].astype(np.float32); H = mm["mem"].astype(np.float32); T = F.shape[0]
H0 = np.zeros_like(H); H0[1:] = H[:-1]; H0[1:][R.rs[:T - 1].astype(bool)] = 0.0
SC = sc["scan"]; RMAX = float(sc["range_max"]); ANG = sc["angles"].astype(float)
mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
st = R.st.astype(float)
NM, NS = actor.norm.mean, torch.sqrt(actor.norm.var + 1e-8)
P = np.load(pairs)
tb = P["tb"]; LOOK = 20
ok = np.array([t + LOOK < T and not R.rs[t: t + LOOK, b].any() for t, b in tb])
idx = np.nonzero(P["ONP"] & (P["BIN"] == bin_i) & ok)[0]
rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
idx = rng.choice(idx, min(8, len(idx)), replace=False)
fig, axs = plt.subplots(2, 4, figsize=(18, 11))
for ax, i in zip(axs.ravel(), idx):
    t, b = tb[i]
    f = torch.from_numpy(F[t, b][None]); h0 = torch.from_numpy(H0[t, b][None])
    with torch.no_grad():
        h1 = actor.gru(f, h0)
        mean, _, v_hat = actor.heads(actor.trunk(torch.cat([f, h1], 1)))
        raw = f[:, 256:] * NS + NM
        p = PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), raw[:, 6], lim)
    x, y, yaw = st[t, b, :3]; c, s_ = np.cos(yaw), np.sin(yaw)
    loc = lambda q: np.stack([(q[:, 0] - x) * c + (q[:, 1] - y) * s_, -(q[:, 0] - x) * s_ + (q[:, 1] - y) * c], 1)
    r = SC[t, b].astype(float) * RMAX; m = r < RMAX * 0.999
    ax.plot(-(r[m] * np.sin(ANG[m])), mount + r[m] * np.cos(ANG[m]), ".", color="#868e96", ms=2, label="지금 스캔의 점")
    for q in R.boxes(t, b):
        ql = loc(q); ax.fill(-ql[:, 1], ql[:, 0], color="#d9480f", alpha=0.55)
    px, py = p["x"][0].numpy(), p["y"][0].numpy()
    ax.plot(-py, px, "-", color="#1971c2", lw=2.5, label="이 순간 낸 plan (1.0 s)")
    pts = p["pts"][0].numpy(); ax.plot(-pts[:, 1], pts[:, 0], "o", color="#1971c2", ms=4)
    d = loc(st[t: t + LOOK + 1, b, :2])
    ax.plot(-d[:11, 1], d[:11, 0], "-", color="#2b8a3e", lw=2.5, label="실제로 간 길 (1.0 s)")
    ax.plot(-d[10:, 1], d[10:, 0], ":", color="#2b8a3e", lw=2, label="그 뒤 1.0 s")
    ax.plot([0], [0], "k^", ms=9)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.5, 8.0); ax.set_aspect("equal")
    ax.set_title(f"상자까지 {P['AX'][i]:.1f} m · 속도 {abs(st[t, b, 3]):.1f} m/s · 첫 0.1 s 조향 변화 {P['d_first'][i]:+.1f}도", fontsize=10)
axs[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"{os.path.basename(os.path.dirname(ck_path))} u{int(ck.get('update', 0))}: 상자(주황)가 가는 길 위에 보이는 순간({BL[bin_i]} 앞)의 plan과 실제로 간 길 -- 차 기준, 위쪽이 앞 [m]", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=85); print("그림", out)
