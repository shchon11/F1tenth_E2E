"""세게 닿은 장면을 네트워크 안에서 따라가기 (11-13 구조 점검의 셋째 측정).
질문: 상자에 세게 닿은 사건마다 (1) 그 상자가 스캔에 처음 보인 때 (2) 그때 차가 낼 수 있는 가장 큰 움직임으로 피할 수 있었나
(3) 네트워크가 그 상자 때문에 첫 0.1 s 조향·속도를 바꾸기 시작한 때 -- 셋의 순서와 간격.
  "보였을 때 이미 못 피함" -> 어떤 층도 못 고침(그 자리를 느리게 들어가야 함).
  "피할 수 있었는데 늦게 움직임" -> policy가 미리 정하지 않는 것이 원인.
피할 수 있었나: 그 결정의 실제 속도·조향각에서 plan과 같은 한계(조향 속도, 타이어 한계, 가속·제동: plan_steps.roll) 안의 움직임 묶음
(조향 3구간 x 속도 4가지 = 1,800가지, 2.0 s)을 펴서, 차 몸통(둘레 점 7 cm 간격, 0.025 s마다)이 벽에 안 닿고 상자와 5 cm 넘게 떨어진 채 그 상자를 지나가는 것이 하나라도 있으면 "돌아갈 수 있음",
지나가지 못해도 2.0 s 동안 안 닿는 것(줄여서 멈춤)이 있으면 "멈출 수는 있음". 상자는 그 결정 때 자리에 서 있다고 봄.
네트워크의 반응: 같은 장면에서 그 상자만 지운 스캔과 견준 첫 0.1 s 조향(1도 넘게)·속도(0.2 m/s 넘게) 차 (net_audit.py와 같은 방법).
견줌: 가는 길 위 상자(2.5~3.2 m 앞, 옆 0.3 m 안)를 3 s 동안 안 닿고 지나간 경우.
상자 위치·지도는 분석에서만 씀. 인자: checkpoint 출력앞머리 기록npz [기록npz ...]"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec
from policy_lib import limits_for, dev
from f1sim.learn.plan_policy import load_actor2, N_SCANS
from f1sim import plan_steps as PS

ck_path, OUT, recs = sys.argv[1], sys.argv[2], sys.argv[3:]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
ck = torch.load(ck_path, map_location=dev, weights_only=False)
lim = limits_for(ck)
actor = load_actor2(ck, dev, check_mount=False).eval()
NM, NS = actor.norm.mean.clone(), torch.sqrt(actor.norm.var + 1e-8)
LOOK = 30                                    # decisions before the hit (3 s)
GD = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # the collision checks of the move family
SCN = os.environ.get("NET_HITS_SCENES", "")  # png: draw scenes from this tool's output OUT.npz instead of measuring
CKP = os.environ.get("NET_HITS_CKPTS", "")  # net_audit.py's _pairs.npz: the reaction by distance for each checkpoint (arguments after the record)
GAP = os.environ.get("NET_HITS_PLANGAP", "0") == "1"   # from OUT.npz: would the plan drawn before each hit have touched the box
log = open(OUT + (f".scenes_{os.environ.get('NET_HITS_WHICH', 'early')}.txt" if SCN else ".plangap.txt" if GAP else ".ckpts.txt" if CKP else ".txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); log.write(s + "\n"); log.flush()


# the family of moves: steering a1 for k1 steps, a2 until k2, a3 after; one speed setting throughout (-1 = hardest braking)
SEQ = []
for a1 in (-1.0, -0.5, 0.0, 0.5, 1.0):
    for a2 in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for a3 in (-1.0, 0.0, 1.0):
            for k1, k2 in ((1, 4), (2, 6), (3, 9), (5, 11), (7, 14), (10, 20)):
                for sp in (-1.0, -0.5, 0.0, 0.6):
                    u = np.zeros((2, PS.ACT_DIM), np.float32)
                    st_ = np.where(np.arange(20) < k1, a1, np.where(np.arange(20) < k2, a2, a3))
                    u[0, :10], u[1, :10] = st_[:10], st_[10:]
                    u[:, 10:] = sp
                    SEQ.append(u)
SEQ = torch.from_numpy(np.stack(SEQ))        # (n, 2, 20)
NSEQ = SEQ.shape[0]
EFFORT = SEQ[:, :, :10].abs().sum((1, 2)) / 20 + 0.5 * (SEQ[:, 0, 10] < 0).float() + 0.3 * (SEQ[:, 0, :10] - SEQ[:, 1, :10]).abs().mean(1)


def in_poly(pts, poly, pad=0.0):
    c = poly.mean(0); inside = np.ones(len(pts), bool); n = len(poly)
    for i in range(n):
        a, b_ = poly[i], poly[(i + 1) % n]
        e = b_ - a; nrm = np.array([e[1], -e[0]]) / (np.hypot(*e) + 1e-9)
        if np.dot(c - a, nrm) > 0:
            nrm = -nrm
        inside &= ((pts - a) @ nrm) <= pad
    return inside


def moves(v0, d0):
    """the family from speed v0 and steering angle d0, in the car's frame: x, y, psi, v at every 0.025 s over 2.0 s (n, 81)"""
    n = NSEQ
    with torch.no_grad():
        p1 = PS.roll(SEQ[:, 0], torch.full((n,), float(v0)), torch.full((n,), float(d0)), lim)
        p2 = PS.roll(SEQ[:, 1], p1["v"][:, -1], p1["delta"][:, -1], lim)
        c, s = torch.cos(p1["psi"][:, -1:]), torch.sin(p1["psi"][:, -1:])
        x2 = p1["x"][:, -1:] + p2["x"] * c - p2["y"] * s
        y2 = p1["y"][:, -1:] + p2["x"] * s + p2["y"] * c
        g = lambda a, b_: torch.cat([a, b_[:, 1:]], 1).to(GD)
        X, Y, P, V = g(p1["x"], x2), g(p1["y"], y2), g(p1["psi"], p1["psi"][:, -1:] + p2["psi"]), g(p1["v"], p2["v"])
        al = (V[:, 1:] * (P[:, 1:] - P[:, :-1]) / (PS.DT / PS.SUB)).abs()                # sideways acceleration [m/s^2]
        return X, Y, P, V, torch.cummax(torch.cat([al[:, :1], al], 1), 1).values


def perimeter(hl, hw, step=0.0725):
    xs = np.arange(-hl, hl + 1e-6, step); ys = np.arange(-hw, hw + 1e-6, step)
    return np.concatenate([np.stack([xs, np.full_like(xs, hw)], 1), np.stack([xs, np.full_like(xs, -hw)], 1),
                           np.stack([np.full_like(ys, hl), ys], 1), np.stack([np.full_like(ys, -hl), ys], 1)])


class One:
    def __init__(self, fn):
        self.R = R = Rec(fn); z = R.z
        mm = np.load(fn[:-4] + "_mem.npz"); sc = np.load(fn[:-4] + "_scan.npz")
        self.F = mm["feat"]; self.H = mm["mem"]
        self.SC = sc["scan"]; self.RMAX = float(sc["range_max"]); self.ANG = sc["angles"].astype(float)
        self.T = T = self.F.shape[0]
        self.k_rm = min(int(round(float(z["rm_t"]) / R.dt)), T)
        self.mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
        self.st = R.st.astype(float)
        self.PER = torch.from_numpy(perimeter(R.hl, R.hw)).float().to(GD)
        self._g = {}
        self.move = None

    def polys(self, t, b):
        """{box index: outline} at decision t"""
        R = self.R; out = {}
        for j, k in enumerate(R.sid[t, b]):
            if k < 0:
                continue
            if int(k) not in R._fp:
                R._fp[int(k)] = R.z[f"fp_{int(k)}"].astype(float)
            q = R._fp[int(k)]; p = R.pose[t, b, j]; c, s_ = np.cos(p[2]), np.sin(p[2])
            out[j] = np.stack([p[0] + q[:, 0] * c - q[:, 1] * s_, p[1] + q[:, 0] * s_ + q[:, 1] * c], 1)
        return out

    def scans(self, t, b, j):
        """the three scans at decision t as read, and with box j's (j None: every box's) beams set to the map's wall distance;
        beams on box j now"""
        R = self.R; k0 = R.k0[t, b]; tk = int(R.tid[0, b]); st = self.st
        w, wo, n0 = [], [], 0
        for i in range(N_SCANS):
            tf = max(t - i, k0)
            r = self.SC[tf, b].astype(float) * self.RMAX
            x, y, yaw = st[tf, b, :3]
            sx, sy = x + self.mount * np.cos(yaw), y + self.mount * np.sin(yaw)
            a = yaw + self.ANG
            end = np.stack([sx + r * np.cos(a), sy + r * np.sin(a)], 1)
            pp = self.polys(min(tf, self.k_rm - 1), b)
            m = np.zeros(len(r), bool)
            for o_j, o_p in pp.items():
                if j is None or o_j == j:
                    m |= in_poly(end, o_p, 0.05) & (r < self.RMAX * 0.999)
            r2 = r.copy()
            if m.any():
                d = np.arange(0.05, self.RMAX, 0.03); idx = np.nonzero(m)[0]
                px = sx + d[None, :] * np.cos(a[idx])[:, None]; py = sy + d[None, :] * np.sin(a[idx])[:, None]
                hit = R.wall_dist(tk, np.stack([px.ravel(), py.ravel()], 1)).reshape(len(idx), len(d)) <= 0.0
                for o_j, o_p in pp.items():                                   # another box behind it stays in view
                    if j is not None and o_j != j:
                        hit |= in_poly(np.stack([px.ravel(), py.ravel()], 1), o_p, 0.0).reshape(len(idx), len(d))
                r2[idx] = np.where(hit.any(1), d[hit.argmax(1)], self.RMAX)
            w.append(r / self.RMAX); wo.append(np.clip(r2 / self.RMAX, 0.0, 1.0))
            if i == 0:
                n0 = int(m.sum())
        return np.stack(w, 0), np.stack(wo, 0), n0

    def plan(self, t, b, S3):
        """the plan the network draws at decision t from scans S3 (plan_steps.roll's output)"""
        f_s = torch.from_numpy(self.F[t, b, 256:].astype(np.float32))[None]
        h0 = torch.zeros(1, self.H.shape[2]) if (t == 0 or self.R.rs[t - 1, b]) else torch.from_numpy(self.H[t - 1, b].astype(np.float32))[None]
        with torch.no_grad():
            f = torch.cat([actor.enc(torch.from_numpy(S3.astype(np.float32))[None]), f_s], 1)
            h1 = actor.gru(f, h0)
            mean, _, v_hat = actor.heads(actor.trunk(torch.cat([f, h1], 1)))
            raw = f_s * NS + NM
            return PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), raw[:, 6], lim)

    def first_step(self, t, b, S3):
        """the first 0.1 s of that plan: (steering angle [deg], speed [m/s])"""
        p = self.plan(t, b, S3)
        return float(np.degrees(p["delta"][0, PS.SUB])), float(p["v"][0, PS.SUB])

    def feasible(self, t, b, j, keep=False):
        """(a move of the family passes box j without touching anything, a move touches nothing for 2.0 s, the least peak sideways
        acceleration [m/s^2] of a passing move that does not brake, of any passing move);
        keep: store the gentlest passing move's path (world frame) in self.move"""
        R = self.R; st = self.st; tk = int(R.tid[0, b])
        x, y, yaw, v = st[t, b, :4]
        d0 = float(self.F[t, b, 256 + 6]) * float(NS[6]) + float(NM[6])
        X, Y, P, V, AL = moves(v, d0)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        wx, wy, wp = x + X * c - Y * s, y + X * s + Y * c, yaw + P                      # (n, 41)
        cp, sp_ = torch.cos(wp)[..., None], torch.sin(wp)[..., None]
        bx = wx[..., None] + self.PER[:, 0] * cp - self.PER[:, 1] * sp_                 # (n, 41, m)
        by = wy[..., None] + self.PER[:, 0] * sp_ + self.PER[:, 1] * cp
        if tk not in self._g:
            self._g[tk] = torch.from_numpy(np.ascontiguousarray(R.grid(tk)[0])).float().to(GD)
        g = self._g[tk]; res = float(R._res[tk])
        ci = torch.floor((bx - float(R._org[tk][0])) / res).long(); ri = torch.floor((by - float(R._org[tk][1])) / res).long()
        ok = (ci >= 0) & (ci < g.shape[1]) & (ri >= 0) & (ri < g.shape[0])
        bad = ~ok | (g[ri.clamp(0, g.shape[0] - 1), ci.clamp(0, g.shape[1] - 1)] <= 0.0)
        pp = self.polys(t, b)
        for poly in pp.values():
            if np.hypot(*(poly.mean(0) - [x, y])) > 18.0:
                continue
            cen_ = poly.mean(0); inside = torch.ones_like(bad); n = len(poly)
            for i in range(n):
                a, b_ = poly[i], poly[(i + 1) % n]
                e = b_ - a; nrm = np.array([e[1], -e[0]]) / (np.hypot(*e) + 1e-9)
                if np.dot(cen_ - a, nrm) > 0:
                    nrm = -nrm
                inside &= ((bx - float(a[0])) * float(nrm[0]) + (by - float(a[1])) * float(nrm[1])) <= 0.05
            bad |= inside
        bad = bad.any(-1)                                                               # (n, 41)
        cen = pp[j].mean(0)
        past = (((float(cen[0]) - wx) * torch.cos(wp) + (float(cen[1]) - wy) * torch.sin(wp)) < -(R.hl + 0.15)) & (P.abs() < 1.3)
        nb = torch.cumsum(bad.long(), 1); L = bad.shape[1]
        has = past.any(1); ip = torch.where(has, past.long().argmax(1), torch.zeros_like(nb[:, 0]))
        end = (ip + 12).clamp(max=L - 1)
        around = has & (nb.gather(1, end[:, None])[:, 0] == 0)
        if keep:
            self.move = None
            if bool(around.any()):
                k = int(torch.where(around, EFFORT.to(GD), torch.full_like(EFFORT, 1e9).to(GD)).argmin())
                e = int(end[k]) + 1
                self.move = np.stack([wx[k, :e].cpu().numpy(), wy[k, :e].cpu().numpy()], 1)
        # the least sideways acceleration a passing move needs (its peak until it is past): without braking / with any speed setting
        pk = AL.gather(1, end[:, None])[:, 0]; nobr = around & (SEQ[:, 0, 10] >= 0).to(GD)
        need = [float(pk[m_].min()) if bool(m_.any()) else np.nan for m_ in (nobr, around)]
        return bool(around.any()), bool(around.any() or (nb[:, -1] == 0).any()), need[0], need[1]

    def trace(self, b, t_end, j):
        """per decision of the 3 s before t_end: (seconds before, box ahead [m], speed, beams on the box, steering change [deg],
        speed taken off [m/s], can go round, can at least stop, the car's steering angle [deg], the least peak sideways acceleration [m/s^2] of a passing move without braking / of any,
        the box's place sideways [m, left +], the car's own sideways acceleration over the next 0.1 s); the "can" and "least" columns
        are nan before the box is in view"""
        R = self.R; rows = []; vis = False
        for t in range(max(t_end - LOOK, R.k0[t_end, b] + N_SCANS), t_end):
            pp = self.polys(t, b)
            if j not in pp:
                continue
            x, y, yaw, v = self.st[t, b, :4]
            dx, dy = pp[j].mean(0) - [x, y]
            ax = dx * np.cos(yaw) + dy * np.sin(yaw)
            w, wo, n0 = self.scans(t, b, j)
            dd, dv = 0.0, 0.0
            if n0 >= 2:
                d1, v1 = self.first_step(t, b, w); d2, v2 = self.first_step(t, b, wo)
                dd, dv = d1 - d2, v2 - v1
            vis = vis or n0 >= 2
            ar, an, nd0, nd1 = self.feasible(t, b, j) if vis else (np.nan,) * 4
            ay = -dx * np.sin(yaw) + dy * np.cos(yaw)
            y1 = self.st[min(t + 1, self.T - 1), b, 2]
            a_car = abs(v * ((y1 - yaw + np.pi) % (2 * np.pi) - np.pi) / R.dt)
            d_now = np.degrees(float(self.F[t, b, 256 + 6]) * float(NS[6]) + float(NM[6]))
            rows.append(((t_end - t) * R.dt, ax, abs(v), n0, dd, dv, ar, an, d_now, nd0, nd1, ay, a_car))
        return np.array(rows, float)


def summarise(tag, TR):
    """TR: list of traces"""
    if not TR:
        say(f"  [{tag}] 없음"); return {}
    seen, around0, any0, lastA, lastS, react, v_seen, d_seen = [], [], [], [], [], [], [], []
    for tr in TR:
        vis = tr[:, 3] >= 2
        if not vis.any():
            seen.append(0.0); around0.append(False); any0.append(False); lastA.append(np.nan); lastS.append(np.nan); react.append(np.nan)
            v_seen.append(np.nan); d_seen.append(np.nan); continue
        i0 = int(np.argmax(vis))
        seen.append(tr[i0, 0]); v_seen.append(tr[i0, 2]); d_seen.append(tr[i0, 1])
        around0.append(bool(tr[i0:, 6].any())); any0.append(bool(tr[i0:, 7].any()))
        ia = np.nonzero(tr[:, 6] > 0)[0]; is_ = np.nonzero(tr[:, 7] > 0)[0]
        lastA.append(tr[ia[-1], 0] if len(ia) else np.nan); lastS.append(tr[is_[-1], 0] if len(is_) else np.nan)
        rc = (np.abs(tr[:, 4]) > 1.0) | (tr[:, 5] > 0.2)
        react.append(tr[int(np.argmax(rc)), 0] if rc.any() else 0.0)
    seen, lastA, lastS, react = map(np.array, (seen, lastA, lastS, react))
    around0, any0 = np.array(around0), np.array(any0)
    n = len(TR); q = lambda a: f"{np.nanmedian(a):.1f} ({np.nanpercentile(a, 25):.1f}~{np.nanpercentile(a, 75):.1f})"
    say(f"  [{tag}] {n}건")
    say(f"    상자가 스캔에 처음 보인 때(빔 2개 이상): {q(seen)} s 전 · 그때 상자까지 {q(np.array(d_seen))} m · 속도 {q(np.array(v_seen))} m/s"
        f" | 3 s 내내 보인 것 {np.mean(seen >= LOOK * 0.1 - 0.35) * 100:.0f} %")
    say(f"    보인 뒤 돌아갈 수 있는 순간이 있었던 것 {around0.mean() * 100:.0f} % · 돌아가지는 못해도 멈출 수는 있었던 것 "
        f"{(any0 & ~around0).mean() * 100:.0f} % · 보였을 때 이미 못 피함 {(~any0).mean() * 100:.0f} %")
    d_seen, v_seen = np.array(d_seen), np.array(v_seen)
    for nm_, m_ in (("돌아갈 수 있었음", around0), ("멈출 수만 있었음", any0 & ~around0), ("이미 못 피함", ~any0)):
        if m_.any():
            say(f"      {nm_} {int(m_.sum())}건: 처음 보인 때 {q(seen[m_])} s 전 · 상자까지 {q(d_seen[m_])} m · 속도 {q(v_seen[m_])} m/s")
    ok = around0 & np.isfinite(lastA)
    if ok.any():
        mg = (seen - lastA)[ok]
        say(f"    (돌아갈 수 있었던 {int(ok.sum())}건) 보인 뒤 돌아갈 수 있던 시간이 0.3 s 이상 {np.mean(mg >= 0.3) * 100:.0f} % · 0.5 s 이상 {np.mean(mg >= 0.5) * 100:.0f} % · "
            f"1.0 s 이상 {np.mean(mg >= 1.0) * 100:.0f} %")
        say(f"    (돌아갈 수 있었던 {int(ok.sum())}건) 돌아갈 수 있던 마지막 순간: {q(lastA[ok])} s 전 · 그때까지 상자가 보인 시간(여유): {q((seen - lastA)[ok])} s")
        say(f"      네트워크가 그 상자 때문에 첫 0.1 s를 바꾸기 시작한 때: {q(react[ok])} s 전 | 마지막 순간보다 먼저 바꾸기 시작 "
            f"{np.mean(react[ok] > lastA[ok]) * 100:.0f} % · 마지막 순간이 지나도록 안 바꿈 {np.mean(react[ok] <= lastA[ok]) * 100:.0f} %")
    return dict(seen=seen, lastA=lastA, lastS=lastS, react=react, around0=around0, any0=any0, v_seen=np.array(v_seen), d_seen=np.array(d_seen))


def scenes():
    """8 hard hits of one kind, each at one decision, car frame (up = ahead): the newest scan, the boxes, the plan drawn at that decision,
    the path driven until the hit, and (if any) the gentlest move of the family that passes the box without touching"""
    from rec_lib import plt
    zz = np.load(OUT + ".npz", allow_pickle=True)
    TRS = [np.asarray(a, float) for a in zz["hit"]]; MT = [m.split("|") for m in zz["meta"]]
    which = os.environ.get("NET_HITS_WHICH", "early")
    cand = []
    for i, tr in enumerate(TRS):
        vis = tr[:, 3] >= 2
        if not vis.any():
            continue
        i0 = int(np.argmax(vis)); ia = np.nonzero(tr[:, 6] > 0)[0]
        if which == "early" and len(ia) and tr[i0, 0] - tr[ia[-1], 0] >= 0.3:   # 1.0 s before the hit, or when it came into view if later
            cand.append((i, int(np.argmin(np.abs(tr[:, 0] - min(tr[i0, 0], 1.0)))), tr[i0, 0]))
        if which == "late" and len(ia) and tr[i0, 0] - tr[ia[-1], 0] >= 0.3:
            cand.append((i, int(ia[-1]), tr[i0, 0]))
        if which == "never" and not (tr[i0:, 6] > 0).any():
            cand.append((i, i0, tr[i0, 0]))
    rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
    pick = [cand[k] for k in sorted(rng.choice(len(cand), min(8, len(cand)), replace=False))]
    by = {os.path.basename(fn): fn for fn in recs}
    fig, axs = plt.subplots(2, 4, figsize=(18, 12)); axs = axs.ravel(); o = None; cur = None
    for ax, (i, ir, t_seen) in zip(axs, sorted(pick, key=lambda c: MT[c[0]][0])):
        fn_, b, t_end, j = MT[i][0], int(MT[i][1]), int(MT[i][2]), int(MT[i][3])
        if cur != fn_:
            del o; o = One(by[fn_]); cur = fn_
        R = o.R; tr = TRS[i]; t = t_end - int(round(tr[ir, 0] / R.dt))
        w, wo, n0 = o.scans(t, b, j); p = o.plan(t, b, w); o.feasible(t, b, j, keep=True)
        x, y, yaw = o.st[t, b, :3]; c, s_ = np.cos(yaw), np.sin(yaw)
        loc = lambda q_: np.stack([(q_[:, 0] - x) * c + (q_[:, 1] - y) * s_, -(q_[:, 0] - x) * s_ + (q_[:, 1] - y) * c], 1)
        r = w[0] * o.RMAX; m = r < o.RMAX * 0.999
        ax.plot(-(r[m] * np.sin(o.ANG[m])), o.mount + r[m] * np.cos(o.ANG[m]), ".", color="#868e96", ms=2, label="지금 스캔의 점")
        for jj, q_ in o.polys(t, b).items():
            ql = loc(q_); ax.fill(-ql[:, 1], ql[:, 0], color="#c92a2a" if jj == j else "#e8590c", alpha=0.75 if jj == j else 0.4,
                                  label="닿은 상자" if jj == j else None)
        ax.plot(-p["y"][0].numpy(), p["x"][0].numpy(), "-", color="#1971c2", lw=2.5, label="이 순간 낸 plan (1.0 s)")
        d = loc(o.st[t: t_end + 1, b, :2])
        ax.plot(-d[:, 1], d[:, 0], "-", color="#2b8a3e", lw=2.5, label="실제로 간 길 (닿을 때까지)")
        ax.plot([-d[-1, 1]], [d[-1, 0]], "x", color="k", ms=12, mew=3, label="세게 닿은 곳")
        if o.move is not None:
            mv = loc(o.move); ax.plot(-mv[:, 1], mv[:, 0], "--", color="#862e9c", lw=2.5, label="안 닿고 지나가는 움직임 (같은 한계 안)")
        ax.plot([0], [0], "k^", ms=9)
        ax.set_xlim(-3.0, 3.0); ax.set_ylim(-0.8, 8.5); ax.set_aspect("equal")
        ax.set_title(f"닿기 {tr[ir, 0]:.1f} s 전 · 상자 {tr[ir, 1]:.1f} m 앞 · {tr[ir, 2]:.1f} m/s\n처음 보인 때 {t_seen:.1f} s 전 · 상자 때문에 바꾼 조향 {tr[ir, 4]:+.1f}도 · 필요한 옆 가속 {tr[ir, 9]:.1f} m/s2", fontsize=10)
        say(f"장면 {fn_} world {b} 닿은 결정 {t_end} 상자 {j}: " + " | ".join(f"{a[0]:.1f}s {a[1]:.1f}m {a[2]:.1f}m/s 빔{int(a[3])} 조향{a[8]:+.0f} 바꿈{a[4]:+.1f}/{a[5]:+.2f} "
                                                              f"{'돌' if a[6] > 0 else ('멈' if a[7] > 0 else ('X' if np.isfinite(a[6]) else '-'))}" for a in tr[::2]))
    axs[0].legend(fontsize=8, loc="upper left")
    ttl = {"early": "보인 뒤 0.3 s 넘게 돌아갈 수 있었는데 세게 닿은 장면 -- 닿기 1.0 s 전(그보다 늦게 보였으면 처음 보인 순간)",
           "late": "보인 뒤 0.3 s 넘게 돌아갈 수 있었는데 세게 닿은 장면 -- 돌아갈 수 있던 마지막 순간",
           "never": "보인 뒤 돌아갈 방법이 한 번도 없었던 세게 닿은 장면 -- 상자가 처음 보인 순간"}[which]
    fig.suptitle(f"{os.path.basename(os.path.dirname(ck_path))} u{int(ck.get('update', 0))}: {ttl} ({len(cand)}건 가운데 아무렇게나 {len(pick)}건) -- 차 기준, 위쪽이 앞 [m]", fontsize=13)
    fig.tight_layout(); fig.savefig(SCN, dpi=85); say("그림", SCN)


def by_checkpoint():
    """the same scenes (a box on the path, by distance; net_audit.py's choice on this record) put to each checkpoint of the run:
    how often the box changes the first 0.1 s. Each checkpoint reads the scans with its own encoder; its memory is run over the 2 s before"""
    P = np.load(CKP); o = One(recs[0]); R = o.R; rng = np.random.default_rng(0); W = 20
    BL = ["0.8~1.5 m", "1.5~2.5 m", "2.5~4 m", "4~5.5 m", "5.5~7 m"]
    sel = np.concatenate([rng.choice(ix, min(120, len(ix)), replace=False) for ix in (np.nonzero(P["ONP"] & (P["BIN"] == i))[0] for i in range(5))])
    D = []
    for n_, i in enumerate(sel):
        t, b = int(P["tb"][i, 0]), int(P["tb"][i, 1]); t0 = max(t - W, int(R.k0[t, b]) + 1)
        w, wo, _ = o.scans(t, b, None)
        S = np.stack([o.scans(tt, b, -1)[0] for tt in range(t0, t)] + [w, wo]).astype(np.float32)        # -1: no box erased
        raw = torch.from_numpy(o.F[t0: t + 1, b, 256:].astype(np.float32)) * NS + NM
        D.append((int(P["BIN"][i]), torch.from_numpy(S), raw))
        if n_ % 100 == 0:
            print("scene", n_, len(sel), file=sys.stderr, flush=True)
    bins = np.array([d[0] for d in D])
    say(f"== 같은 장면({os.path.basename(recs[0])}에서 상자가 가는 길 위에 보이는 결정 {len(D)}개)을 checkpoint마다 넣어 본 것: 상자 때문에 첫 0.1 s 조향이 1도 넘게 바뀐 비율 [%] / 줄인 속도 중앙 [m/s]")
    say("   " + " | ".join(f"{x:>12}" for x in ["checkpoint"] + BL))
    DY = []
    for path in sys.argv[4:]:
        c_ = torch.load(path, map_location=dev, weights_only=False); a = load_actor2(c_, dev, check_mount=False).eval()
        dd, dv, dy = [], [], []
        with torch.no_grad():
            for _, S, raw in D:
                n = raw.shape[0]
                f = a.features(S[:n], raw); h = torch.zeros(1, a.memory)
                for k in range(n - 1):
                    h = a.gru(f[k: k + 1], h)
                f2 = a.features(S[n:], raw[-1:])
                out = []
                for fk in (f[n - 1: n], f2):
                    mean, _, v_hat = a.heads(a.trunk(torch.cat([fk, a.gru(fk, h)], 1)))
                    p_ = PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), raw[-1:, 6], lim)
                    out.append((float(np.degrees(p_["delta"][0, PS.SUB])), float(p_["v"][0, PS.SUB]), float(p_["y"][0, 5 * PS.SUB])))
                dd.append(out[0][0] - out[1][0]); dv.append(out[1][1] - out[0][1]); dy.append(out[0][2] - out[1][2])
        dd, dv = np.array(dd), np.array(dv); DY.append((int(c_.get('update', 0)), np.abs(np.array(dy))))
        say("   " + " | ".join(f"{x:>12}" for x in [f"u{int(c_.get('update', 0))}"] + [
            f"{np.mean(np.abs(dd[bins == i]) > 1.0) * 100:.0f} / {np.median(dv[bins == i]):+.2f}" for i in range(5)]))
    say("== 같은 장면에서 상자 때문에 plan의 0.5 s 뒤 지점이 옆으로 옮겨진 거리: 10 cm 넘게 옮긴 비율 [%] / 중앙 [cm]")
    for u, a_ in DY:
        say("   " + " | ".join(f"{x:>12}" for x in [f"u{u}"] + [f"{np.mean(a_[bins == i] > 0.10) * 100:.0f} / {np.median(a_[bins == i]) * 100:.0f}" for i in range(5)]))


def plan_gap():
    """every hard hit, at 1.0 / 0.5 / 0.2 s before: would the plan drawn then, driven exactly, touch the hit box; and where the car
    really was at the hit, measured sideways from that plan (+ = further into the turn the car was making)"""
    zz = np.load(OUT + ".npz", allow_pickle=True)
    MT = [m.split("|") for m in zz["meta"]]; by = {os.path.basename(fn): fn for fn in recs}; o = None; cur = None
    rows = {10: [], 5: [], 2: []}
    for fn_, b, t_end, j in sorted(MT):
        b, t_end, j = int(b), int(t_end), int(j)
        if cur != fn_:
            del o; o = One(by[fn_]); cur = fn_
        R = o.R; per = perimeter(float(R.hl), float(R.hw))
        for back in rows:
            t = t_end - back
            if t <= R.k0[t_end, b] or j not in o.polys(t, b):
                continue
            w, _, n0 = o.scans(t, b, j); p = o.plan(t, b, w)
            x, y, yaw = o.st[t, b, :3]; c, s_ = np.cos(yaw), np.sin(yaw)
            q_ = o.polys(t, b)[j]; ql = np.stack([(q_[:, 0] - x) * c + (q_[:, 1] - y) * s_, -(q_[:, 0] - x) * s_ + (q_[:, 1] - y) * c], 1)
            px, py, pp_ = p["x"][0].numpy(), p["y"][0].numpy(), p["psi"][0].numpy()
            k_hit = back * PS.SUB                                        # the plan's sample at the time of the hit
            touch = False
            for k in range(0, min(k_hit + 2 * PS.SUB, len(px) - 1) + 1):
                cc, ss = np.cos(pp_[k]), np.sin(pp_[k])
                pts = np.stack([px[k] + per[:, 0] * cc - per[:, 1] * ss, py[k] + per[:, 0] * ss + per[:, 1] * cc], 1)
                if in_poly(pts, ql, 0.0).any():
                    touch = True; break
            xe, ye, yawe = o.st[t_end, b, :3]
            dl = np.array([(xe - x) * c + (ye - y) * s_, -(xe - x) * s_ + (ye - y) * c])
            kk = min(k_hit, len(px) - 1); nrm = np.array([-np.sin(pp_[kk]), np.cos(pp_[kk])])
            side = float((dl - np.array([px[kk], py[kk]])) @ nrm)        # + = left of the plan
            turn = float(np.arctan2(np.sin(yawe - yaw), np.cos(yawe - yaw)))
            box_side = float((ql.mean(0) - np.array([px[kk], py[kk]])) @ nrm)
            rows[back].append((touch, side * np.sign(turn) if abs(turn) > 0.05 else np.nan, abs(turn), box_side * np.sign(turn) if abs(turn) > 0.05 else np.nan, n0))
    say(f"== 세게 닿은 {len(MT)}건: 닿기 몇 초 전에 낸 plan을 그대로 달렸다면 그 상자에 닿았나, 차가 실제로 닿은 곳은 그 plan에서 옆으로 얼마나 벗어났나")
    for back, L in rows.items():
        A = np.array(L, float); tn = np.isfinite(A[:, 1])
        say(f"  닿기 {back * 0.1:.1f} s 전 ({len(A)}건, 그때 상자가 스캔에 보임 {np.mean(A[:, 4] >= 2) * 100:.0f} %): plan대로 가면 닿음 {A[:, 0].mean() * 100:.0f} % | "
            f"돌고 있던 것 {tn.mean() * 100:.0f} %, 그 가운데 실제 닿은 곳이 plan보다 도는 쪽 안으로 {np.mean(A[tn, 1] > 0.05) * 100:.0f} % · 바깥으로 {np.mean(A[tn, 1] < -0.05) * 100:.0f} % · "
            f"벗어난 거리 중앙 {np.median(A[tn, 1]) * 100:+.0f} cm ({np.percentile(A[tn, 1], 25) * 100:+.0f}~{np.percentile(A[tn, 1], 75) * 100:+.0f}) | "
            f"상자가 plan의 도는 쪽 안에 있음 {np.mean(A[tn, 3] > 0) * 100:.0f} %")


if GAP:
    plan_gap(); sys.exit(0)
if CKP:
    by_checkpoint(); sys.exit(0)
if SCN:
    scenes(); sys.exit(0)
HIT, REHIT, CLEAN = [], 0, []
META = []
MERGE = os.environ.get("NET_HITS_MERGE", "0") == "1"          # the arguments are this tool's outputs (one per record): join them
if MERGE:
    for fn in recs:
        zz = np.load(fn, allow_pickle=True)
        HIT += [np.asarray(a, float) for a in zz["hit"]]; CLEAN += [np.asarray(a, float) for a in zz["clean"]]
        REHIT += int(zz["rehit"]); META += [tuple(m.split("|")) for m in zz["meta"]]
    recs = []
for fn in recs:
    o = One(fn); R = o.R; T = o.T
    hard = R.d[:T, :, 4] > 0; anyhard = R.d[:T, :, 3:6].sum(-1) > 0; touch = R.d[:T, :, 1] > 0
    box_w = (R.sid[0] >= 0).any(-1)
    n_h = 0
    for b in np.nonzero(box_w)[0]:
        for t in np.nonzero(hard[:o.k_rm, b])[0]:
            if t - R.k0[t, b] < 12:
                continue
            if anyhard[max(t - LOOK, 0): t, b].any() or abs(o.st[t - 1, b, 3]) < 1.0:
                REHIT += 1; continue
            pp = o.polys(t - 1, b)
            if not pp:
                continue
            body = R.body(*o.st[t, b, :3]).mean(0)
            j = min(pp, key=lambda k: np.hypot(*(pp[k].mean(0) - body)))
            HIT.append(o.trace(b, t, j)); META.append((os.path.basename(fn), int(b), int(t), int(j))); n_h += 1
            print(f"world {b} t {t}: hits {len(HIT)} clean {len(CLEAN)}", file=sys.stderr, flush=True)
        # clean passes of a box on the path
        t = 3; n_c = 0
        while t < o.k_rm - 30 and n_c < 2:
            x, y, yaw = o.st[t, b, :3]; c, s = np.cos(yaw), np.sin(yaw); best = None
            for j, poly in o.polys(t, b).items():
                dx, dy = poly.mean(0) - [x, y]; ax, ay = dx * c + dy * s, -dx * s + dy * c
                if 2.5 < ax < 3.2 and abs(ay) < 0.3:
                    best = (j, ax)
            if best is not None and abs(o.st[t, b, 3]) > 1.0 and not touch[t: t + 30, b].any() and not anyhard[max(t - 20, 0): t, b].any() and t - R.k0[t, b] > 12:
                te = t                                                      # the decision the car is level with the box
                while te < o.k_rm - 1:
                    pp = o.polys(te, b); x, y, yaw = o.st[te, b, :3]
                    dx, dy = pp[best[0]].mean(0) - [x, y]
                    if dx * np.cos(yaw) + dy * np.sin(yaw) < 0.3:
                        break
                    te += 1
                CLEAN.append(o.trace(b, te, best[0])); n_c += 1
                t = te + 40
            else:
                t += 1
    say(f"== {os.path.basename(fn)}: 세게 닿음 {n_h}건 (지금까지 안 닿고 지나감 {len(CLEAN)}건)")
    del o

name = f"{os.path.basename(os.path.dirname(ck_path))} u{int(ck.get('update', 0))}"
say(f"\n== {name}: 상자에 세게 닿은 사건 {len(HIT) + REHIT}건 가운데 {REHIT}건은 앞선 세게 닿음 뒤 3 s 안이거나 1 m/s 밑(박힌 뒤 다시 닿음) -> 따로. 나머지 {len(HIT)}건:")
sh = summarise("세게 닿음", HIT)
sc_ = summarise("안 닿고 지나감(가는 길 위 상자)", CLEAN)
# how the family does against the car itself: the share of decisions it finds a way, by time before
for tag, TR in (("세게 닿음", HIT), ("안 닿고 지나감", CLEAN)):
    if TR:
        A = np.concatenate(TR)
        say(f"  [{tag}] 몇 초 전 결정에서 (상자가 보인 뒤) 돌아갈 수 있음 / 2 s 동안 안 닿을 수 있음 [%]: " + " · ".join(
            f"{lo:.1f}~{hi:.1f} s {A[m_, 6].mean() * 100:.0f}/{A[m_, 7].mean() * 100:.0f} ({int(m_.sum())}개)"
            for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))
            for m_ in [(A[:, 0] > lo + 1e-6) & (A[:, 0] <= hi + 1e-6) & np.isfinite(A[:, 6])] if m_.any()))
        say(f"  [{tag}] 몇 초 전 결정에서 상자 때문에 바뀐 첫 0.1 s (보이는 결정만): 조향 중앙 |도| / 줄인 속도 중앙: " + " · ".join(
            f"{lo:.1f}~{hi:.1f} s {np.median(np.abs(A[m_, 4])):.1f}/{np.median(A[m_, 5]):+.2f}"
            for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))
            for m_ in [(A[:, 0] > lo) & (A[:, 0] <= hi) & (A[:, 3] >= 2)] if m_.any()))
        say(f"  [{tag}] 몇 초 전 결정에서 속도를 안 줄이고 돌아가는 데 필요한 가장 작은 옆 가속 [m/s2] 중앙 (25~75 %) / 차가 그때 실제로 낸 옆 가속 중앙 / 속도 중앙 [m/s]: " + " · ".join(
            f"{lo:.1f}~{hi:.1f} s {np.nanmedian(A[m_, 9]):.1f} ({np.nanpercentile(A[m_, 9], 25):.1f}~{np.nanpercentile(A[m_, 9], 75):.1f}) / {np.median(A[m_, 12]):.1f} / {np.median(A[m_, 2]):.1f}"
            for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 2.0))
            for m_ in [(A[:, 0] > lo + 1e-6) & (A[:, 0] <= hi + 1e-6) & np.isfinite(A[:, 6])] if m_.any() and np.isfinite(A[m_, 9]).any()))
        say(f"  [{tag}] 속도를 안 줄이고는 못 돌아가는 결정의 비율 [%]: " + " · ".join(
            f"{lo:.1f}~{hi:.1f} s {np.mean(~np.isfinite(A[m_, 9])) * 100:.0f}"
            for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 2.0))
            for m_ in [(A[:, 0] > lo + 1e-6) & (A[:, 0] <= hi + 1e-6) & np.isfinite(A[:, 6])] if m_.any()))
def obj(L):
    a = np.empty(len(L), dtype=object)
    for i, x in enumerate(L):
        a[i] = x
    return a


np.savez_compressed(OUT + ".npz", hit=obj(HIT), clean=obj(CLEAN),
                    meta=np.array([f"{a}|{b}|{c}|{d}" for a, b, c, d in META]), rehit=REHIT)
say("저장", OUT + ".npz")
