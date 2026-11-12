"""상자를 보고도 왜 그대로 가는가 -- policy 안에서 (11-13, 사용자: "feature도 좀 들여다 보고 gradient도 좀 볼 수 있으면 보고";
"지가 말한것도 다 쳐 까먹고 그냥 충돌 그대론데요?"). 낮의 표: 세게 닿는 상자는 1.2~2.6 s 전부터 보였는데 0.5 s 전 plan이 상자를 지나감.
입력: 시험 기록(exam_record.py, F1SIM_EXAM_SAVE_MEM=1 + F1SIM_EXAM_SAVE_SCAN=1: _mem.npz 특징·기억, _scan.npz 가장 새 스캔·빔이 맞은 것)과
그 checkpoint. policy 입력의 스캔 = 지금과 앞 두 결정의 스캔(기록에서 다시 쌓고, 인코더 출력이 기록된 특징과 맞는지 먼저 확인).
고르는 결정: 상자 세션, 치우기 전, 1 m/s 넘게 달림, 상자가 차 앞 0.8~4 m · 옆 1.5 m 안에 있고 지금 스캔에 그 상자에 맞은 빔이 있음.
  닿음 = 1.5 s 안에 그 상자(아무 상자)에 세게 닿음 / 깨끗 = 3 s 안에 상자에 전혀 안 닿음.
결정마다:
  1) 상자 지운 스캔: 상자에 맞은 빔(끝점이 상자 모양 + 5 cm 안)을 지도 벽까지의 거리로 바꾼 세 장의 스캔 -> 같은 상태·같은 기억으로 plan을 다시
     계산. 상자 때문에 plan이 상자 반대쪽으로 비킨 거리(1.0 s · 0.5 s 점) [m], 상자 때문에 줄인 1.0 s 속도 [m/s].
  2) 실제 plan(기록의 10점)이 상자에서 떨어진 가장 가까운 거리 - 차 반폭(0.155 m): 음수면 plan의 선이 상자를 지나감.
  3) 기울기: 1.0 s 점 옆 위치의 스캔 기울기 가운데 상자 빔에 쏠린 몫 / 상자 빔이 전체 빔에서 차지하는 몫 (1보다 크면 상자를 더 봄).
  4) 같은 계산에서 직전 기억을 0으로: 상자 반응(1)이 기억에 따라 달라지나.
상자 위치·모양·지도는 분석에서만 씀(policy 입력 아님).
인자: checkpoint 기록npz 출력앞머리"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt
from policy_lib import limits_for, plan_of, dev

ck_path, npz, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
RNG = np.random.default_rng(0)
N_CLEAN = 2500
HALF_W = 0.155

from f1sim.learn.plan_policy import load_actor2, N_SCANS
ck = torch.load(ck_path, map_location=dev, weights_only=False)
lim = limits_for(ck)
actor = load_actor2(ck, dev, check_mount=False).eval()
for p_ in actor.parameters():
    p_.requires_grad_(False)
R = Rec(npz); z = R.z
mm = np.load(npz[:-4] + "_mem.npz"); sc = np.load(npz[:-4] + "_scan.npz")
F = mm["feat"].astype(np.float32); H = mm["mem"].astype(np.float32)
SC = sc["scan"]; RMAX = float(sc["range_max"]); ANG = sc["angles"].astype(float)
T, B = F.shape[0], F.shape[1]
H0 = np.zeros_like(H); H0[1:] = H[:-1]; H0[1:][R.rs[:T - 1].astype(bool)] = 0.0
dt = R.dt
k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
st = R.st.astype(float)
name = f"{os.path.basename(os.path.dirname(ck_path))} u{int(ck.get('update', 0))}"


def stack3(t, b, scan_t):
    """the policy's scan input at decision t: (N_SCANS, beams), newest first, an episode's first decisions repeat its first scan"""
    k0 = R.k0[t, b]
    return np.stack([scan_t[max(t - i, k0)][b] if scan_t is not None else SC[max(t - i, k0), b] for i in range(N_SCANS)], 0)


def in_poly(pts, poly, pad=0.05):
    """points inside a convex polygon grown by `pad` (signed distance to each edge)"""
    c = poly.mean(0); inside = np.ones(len(pts), bool)
    n = len(poly)
    for i in range(n):
        a, b_ = poly[i], poly[(i + 1) % n]
        e = b_ - a; nrm = np.array([e[1], -e[0]]) / (np.hypot(*e) + 1e-9)
        if np.dot(c - a, nrm) > 0:
            nrm = -nrm                                                  # outward normal
        inside &= ((pts - a) @ nrm) <= pad
    return inside


def poly_dist(pts, poly):
    """[m] distance from each point to a convex polygon (0 inside)"""
    d = np.full(len(pts), np.inf); n = len(poly)
    for i in range(n):
        a, b_ = poly[i], poly[(i + 1) % n]; e = b_ - a
        u = np.clip(((pts - a) @ e) / (e @ e), 0.0, 1.0)
        d = np.minimum(d, np.hypot(*(pts - (a + u[:, None] * e)).T))
    return np.where(in_poly(pts, poly, 0.0), 0.0, d)


def nobox(t, b):
    """frames t, t-1, t-2 with the box beams set to the map's wall distance: (scans3 with, scans3 without, box-beam mask of frame t)"""
    k0 = R.k0[t, b]; tk = int(R.tid[0, b])
    w, wo, m0 = [], [], None
    for i in range(N_SCANS):
        tf = max(t - i, k0)
        r = SC[tf, b].astype(float) * RMAX
        x, y, yaw = st[tf, b, 0], st[tf, b, 1], st[tf, b, 2]
        sx, sy = x + mount * np.cos(yaw), y + mount * np.sin(yaw)
        a = yaw + ANG
        end = np.stack([sx + r * np.cos(a), sy + r * np.sin(a)], 1)
        m = np.zeros(len(r), bool)
        for poly in R.boxes(min(tf, k_rm - 1), b):
            m |= in_poly(end, poly) & (r < RMAX * 0.999)
        r2 = r.copy()
        if m.any():
            d = np.arange(0.05, RMAX, 0.03)
            idx = np.nonzero(m)[0]
            px = sx + d[None, :] * np.cos(a[idx])[:, None]; py = sy + d[None, :] * np.sin(a[idx])[:, None]
            wd = R.wall_dist(tk, np.stack([px.ravel(), py.ravel()], 1)).reshape(len(idx), len(d))
            hit = wd <= 0.0
            r2[idx] = np.where(hit.any(1), d[hit.argmax(1)], RMAX)
        w.append(r / RMAX); wo.append(np.clip(r2 / RMAX, 0.0, 1.0))
        if i == 0:
            m0 = m
    return np.stack(w, 0), np.stack(wo, 0), m0


# check the scan input rebuilt from the record against the recorded encoder output
with torch.no_grad():
    tt = RNG.integers(5, T, 64); bb = RNG.integers(0, B, 64)
    S3 = torch.from_numpy(np.stack([stack3(t, b, None) for t, b in zip(tt, bb)]).astype(np.float32))
    err = float((actor.enc(S3) - torch.from_numpy(F[tt, bb, :256])).abs().max())
print(f"== {name} ({os.path.basename(npz)}): 다시 쌓은 스캔의 인코더 출력과 기록된 특징의 차 최대 {err:.4f}", flush=True)

# the decisions: a box in front (0.8-4 m ahead, within 1.5 m sideways) with beams on it now
box_w = (R.sid[0] >= 0).any(-1)
V = np.abs(st[:T, :, 3])
hard_prop = R.d[:T, :, 4] > 0; on_prop = R.d[:T, :, 1] > 0
cand = []
for b in np.nonzero(box_w)[0]:
    for t in range(N_SCANS, k_rm):
        if V[t, b] < 1.0:
            continue
        x, y, yaw = st[t, b, :3]
        c, s_ = np.cos(yaw), np.sin(yaw)
        best = None
        for poly in R.boxes(t, b):
            cx, cy = poly.mean(0) - [x, y]
            ax_, ay_ = cx * c + cy * s_, -cx * s_ + cy * c
            if 0.8 < ax_ < 4.0 and abs(ay_) < 1.5 and (best is None or ax_ < best[0]):
                best = (ax_, ay_, poly)
        if best is None:
            continue
        hit = hard_prop[t + 1: t + 16, b].any()
        clean = not on_prop[t + 1: t + 31, b].any()
        if hit or clean:
            cand.append((t, b, best[0], best[1], bool(hit)))
hits = [c_ for c_ in cand if c_[4]]; cleans = [c_ for c_ in cand if not c_[4]]
if len(cleans) > N_CLEAN:
    cleans = [cleans[i] for i in RNG.choice(len(cleans), N_CLEAN, replace=False)]
print(f"  고른 결정: 상자 앞 0.8~4 m · 1.5 s 안에 세게 닿음 {len(hits)}개 / 3 s 안에 안 닿음 {len(cleans)}개(뽑음)", flush=True)

rows = []
for (t, b, ax_, ay_, hit) in hits + cleans:
    w3, wo3, m0 = nobox(t, b)
    if not m0.any():
        continue                                                       # no beam on the box now: not "seen"
    rows.append((t, b, ax_, ay_, hit, w3, wo3, m0))
print(f"  그중 지금 스캔에 상자 빔이 있는 결정: 닿음 {sum(r[4] for r in rows)} · 깨끗 {sum(not r[4] for r in rows)}", flush=True)
tb = np.array([(r[0], r[1]) for r in rows]); AX = np.array([r[2] for r in rows]); AY = np.array([r[3] for r in rows])
HIT = np.array([r[4] for r in rows]); M0 = np.stack([r[7] for r in rows])
W3 = torch.from_numpy(np.stack([r[5] for r in rows]).astype(np.float32)); WO3 = torch.from_numpy(np.stack([r[6] for r in rows]).astype(np.float32))
Fs = torch.from_numpy(F[tb[:, 0], tb[:, 1], 256:]); h0 = torch.from_numpy(H0[tb[:, 0], tb[:, 1]])
with torch.no_grad():
    fw = torch.cat([actor.enc(W3), Fs], 1); fo = torch.cat([actor.enc(WO3), Fs], 1)
    yw, vw, _, y5w = plan_of(actor, lim, fw, h0); yo, vo, _, y5o = plan_of(actor, lim, fo, h0)
    zw, zvw, _, _ = plan_of(actor, lim, fw, torch.zeros_like(h0)); zo, zvo, _, _ = plan_of(actor, lim, fo, torch.zeros_like(h0))
side = -np.sign(AY)                                                   # away from the box = this sign of lateral
away = side * (yw - yo).numpy(); away5 = side * (y5w - y5o).numpy(); slow = (vo - vw).numpy()
away_z = side * (zw - zo).numpy()
# 2) the recorded plan's clearance from the box
clr = np.full(len(rows), np.nan)
pl = R.pl
for i, (t, b, *_r) in enumerate(rows):
    polys = R.boxes(t, b)
    if pl is not None and len(polys):
        pts = pl[t, b].astype(float)
        clr[i] = min(float(poly_dist(pts, q).min()) for q in polys) - HALF_W
# 3) gradient share on the box beams (frame t)
S3g = W3.clone().requires_grad_(True)
yg, _, _, _ = plan_of(actor, lim, torch.cat([actor.enc(S3g), Fs], 1), h0)
g = torch.autograd.grad(yg.sum(), S3g)[0][:, 0].abs().numpy()        # (n, beams) newest frame
att = (g * M0).sum(1) / (g.sum(1) + 1e-12) / (M0.sum(1) / M0.shape[1])

bins = [(0.8, 1.5), (1.5, 2.5), (2.5, 4.0)]
for title, sel in (("세게 닿음(1.5 s 안)", HIT), ("안 닿음(3 s 안)", ~HIT)):
    print(f"  [{title}] 결정 {int(sel.sum())}개", flush=True)
    for lo, hi in bins:
        m = sel & (AX >= lo) & (AX < hi)
        if m.sum() < 5:
            print(f"    상자까지 {lo}~{hi} m: {int(m.sum())}개 -- 너무 적음", flush=True); continue
        print(f"    상자까지 {lo}~{hi} m ({int(m.sum())}개): 상자 때문에 비킨 거리 1.0 s 점 중앙 {np.median(away[m]):+.3f} m (0.5 s 점 {np.median(away5[m]):+.3f}) · "
              f"상자 때문에 줄인 1.0 s 속도 {np.median(slow[m]):+.2f} m/s | 실제 plan이 상자를 지나가는(여유 < 0) 비율 {100 * np.mean(clr[m] < 0):.0f} % · "
              f"여유 중앙 {np.nanmedian(clr[m]):+.2f} m | 상자 빔에 쏠린 기울기 {np.median(att[m]):.1f}배 | 기억 0일 때 비킨 거리 {np.median(away_z[m]):+.3f} m", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for sel, col, lab_ in ((~HIT, "#2a6fdb", "안 닿음"), (HIT, "#d9480f", "세게 닿음")):
    ax[0].scatter(AX[sel], away[sel], s=6 if lab_ == "안 닿음" else 14, color=col, alpha=0.4 if lab_ == "안 닿음" else 0.9, label=lab_)
    ax[1].scatter(AX[sel], slow[sel], s=6 if lab_ == "안 닿음" else 14, color=col, alpha=0.4 if lab_ == "안 닿음" else 0.9, label=lab_)
    ax[2].scatter(AX[sel], clr[sel], s=6 if lab_ == "안 닿음" else 14, color=col, alpha=0.4 if lab_ == "안 닿음" else 0.9, label=lab_)
for a_, yl, tt_ in ((ax[0], "[m]", "상자를 스캔에서 지울 때와 견줘 plan이 상자 반대쪽으로 비킨 거리\n(1.0 s 점, +면 비킴)"),
                    (ax[1], "[m/s]", "상자 때문에 plan이 줄인 1.0 s 속도 (+면 줄임)"),
                    (ax[2], "[m]", "실제 plan의 선이 상자에서 떨어진 여유(차 반폭 뺌, 음수 = 지나감)")):
    a_.axhline(0, color="#868e96", lw=0.8); a_.set_xlabel("상자까지 앞 거리 [m]"); a_.set_ylabel(yl); a_.set_title(tt_, fontsize=10); a_.legend(fontsize=8)
fig.suptitle(f"{name}: 상자가 보이는 결정에서 policy의 반응 (시험 기록, plan대로 정확히)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=100); plt.close(fig)
print("그림", OUT + ".png", flush=True)
