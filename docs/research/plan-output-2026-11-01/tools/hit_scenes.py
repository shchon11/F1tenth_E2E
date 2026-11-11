"""보고도 닿는 장면을 갈래로 나눔 (11-11: 상자 접촉의 86 %가 1.5 s 넘게 미리 보이던 상자 -- 몰라서가 아니라 보고도 닿음. 사용자와 정한 다음
일: 닿는 장면을 전부 뽑아 왜 닿는지부터). 학습 세계(plan_rl.build, 상자 치움 포함), 학습 조건(plan대로 정확히), policy 평균 행동, 고정 시드,
통째 에피소드. 학습 차가 벽·상자·상대 차에 새로 닿을 때마다 한 건. 갈래(순서대로 먼저 맞는 것):
  1 앞선 접촉에 이어짐   그 차가 2 s 안에 이미 닿았음(튕기거나 끼어서 연달아)
  2 거의 서 있다가       앞 1 s 평균 속도 1.5 m/s 아래(끼어서 빠져나오는 중)
  3 달리다 처음 닿음     -- 상자: 닿은 상자 양옆의 빈 폭(차 진행 방향에 수직, 상자 끝에서 벽·다른 상자·상대 차까지)으로
                            '넉넉한 쪽인데 닿음'(택한 쪽 >= 차 폭 + 0.2 m) / '좁은 쪽으로 감'(택한 쪽은 모자라고 반대쪽은 넉넉) / '양쪽 다 좁음'
                         -- 벽: 1.5 m 안에 상대 차가 있음 / 상자가 있음 / 아무것도 없음
                         -- 상대 차: 닿는 순간 상대가 우리 차 기준 앞(0.3 m 넘게) = 앞차를 박음(상대가 0.3 m/s 아래면 '서 있는 상대를 박음'),
                            뒤 = 뒤에서 박힘, 그 사이 = 옆으로 닿음
출력: 갈래별 건수(세게 = 1 m/s 넘게 / 전체), 갈래마다 장면 12장(세게 닿은 것 먼저, ±2.5 m, 벽은 ±4 m), 건별 기록 JSON.
더 재는 것:
  * 에피소드 시작 5 s 안 / 그 뒤의 세게 닿음 /분 (출발 직후에 몰리나)
  * 상대 차: 닿는 순간 두 차 중심을 잇는 방향으로 각자 다가간 속도(우리 / 상대) -- 누가 누구에게 갔나
  * 상자 옆을 지나간 모든 건(2 m 안에 들어와 앞에서 뒤로 넘어감, 평균 1.5 m/s 넘게): 차 중심 경로와 상자 사이 거리 - 차 반폭 = 지나간 여유,
    그리고 가장 가까워지기 전 1 s 동안 결정마다 plan이 상자의 어느 쪽으로 지나가나(왼/오) -- 쪽이 바뀐 건의 비율, plan이 상자를 스치는
    (plan과 상자 거리 < 차 반폭) 결정의 비율을 세게 닿은 건 / 안 닿은 건으로 나눠 봄. 그림 hits_NAME_pass.png
장면: 파랑 = 우리 차(점 = 2 s 전부터의 자취, 진할수록 최근), 빨강 = 상대 차, 갈색 = 상자, 검은 선 = 닿기 0.5 s 전 plan, 분홍 선 = 닿는 결정의
plan, 노란 별 = 닿은 점.
인자: checkpoint 출력폴더 [환경 수, 기본 512] [초, 기본 40]"""
import os, sys, json
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, PT_IDX
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import props as _props
from f1sim import plan_steps as PS
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

torch.set_grad_enabled(False)
ck_path, out_dir = sys.argv[1], sys.argv[2]
n_env = int(sys.argv[3]) if len(sys.argv) > 3 else 512
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
SEED = 4242
os.makedirs(out_dir, exist_ok=True)
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
env, lim, world = PR.build(PR.WORLD, n_env, SEED, dev, fmt="v2", overrides={"episode_s": secs + 5.0})
lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
po = env.procedural
rm_p = float(getattr(world, "removal_p", 0.0))
remover = SR.PropRemoval(env, rm_p, *[float(v) for v in getattr(world, "removal_s", (10.0, 30.0))], seed=SEED) if rm_p > 0 else None
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
I_CNT = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")]
i_hk, i_hx, i_hy, i_hs = (CC.index(k) for k in ("held_kind", "held_x", "held_y", "held_severity"))
KIND = ("벽", "상자", "상대 차")
ii = PT_IDX.to(dev)
T = int(secs / PS.DT)
ST, OP, ON, PL, BS, BP = [], [], [], [], [], []
events = []
prev = env.sim.contact_frame()[lid].float()[:, I_CNT].clone()
for kd in range(T):
    ST.append(env.sim.state[lid, :4].float().cpu()); OP.append(env.sim.state[others, :4].float().cpu())
    ON.append((~absent[others]).cpu())
    BS.append(po.p_sid[eid].to(torch.int16).cpu()); BP.append(po.p_poses[eid].float().cpu())
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    out, _, (x, y, _) = drv.decision(torch.tanh(mean), p)
    if remover is not None:
        remover.step()
    PL.append(torch.stack([x[:, ii], y[:, ii]], -1).cpu())
    cf = env.sim.contact_frame()[lid].float()
    now = cf[:, I_CNT]
    d = (now - prev).clamp(min=0.0); prev = now.clone()
    rs = out["restarted"]
    for b, k in torch.nonzero((d[:, :3] > 0.5) & ~rs[:, None]).tolist():
        own = int(cf[b, i_hk]) == k + 1                              # the held record is this kind's (the hold's hardest)
        events.append(dict(kd=kd, b=b, kind=k, hard=bool(d[b, 3 + k] > 0.5),
                           px=float(cf[b, i_hx]) if own else float("nan"), py=float(cf[b, i_hy]) if own else float("nan"),
                           sev=float(cf[b, i_hs]) if own else float("nan")))
ST.append(env.sim.state[lid, :4].float().cpu()); OP.append(env.sim.state[others, :4].float().cpu())
ST, OP, ON, PL = torch.stack(ST).numpy(), torch.stack(OP).numpy(), torch.stack(ON).numpy(), torch.stack(PL).numpy()
BS, BP = torch.stack(BS).numpy(), torch.stack(BP).numpy()
TID = env.sim.tid[lid].cpu().numpy()
trk = env.sim.track
_edt = trk.edt.float().cpu().numpy(); _off = trk.t_off.cpu().numpy(); _H = trk.t_H.cpu().numpy().astype(int); _W = trk.t_W.cpu().numpy().astype(int)
_res = trk.t_res.cpu().numpy(); _org = trk.t_origin.cpu().numpy()
hl, hw = 0.5 * float(env.cfg.vehicle.length), 0.5 * float(env.cfg.vehicle.width)
NEED = 2 * hw + 0.20                                                # a gap the car fits with 10 cm a side
_fp, _grid = {}, {}


def grid(k):
    if k not in _grid:
        g = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k])
        _grid[k] = (g <= 0.0, [_org[k][0], _org[k][0] + _W[k] * _res[k], _org[k][1], _org[k][1] + _H[k] * _res[k]])
    return _grid[k]


def wall_at(k, pts):
    occ, _ = grid(k)
    c = np.floor((pts[:, 0] - _org[k][0]) / _res[k]).astype(int); r = np.floor((pts[:, 1] - _org[k][1]) / _res[k]).astype(int)
    ok = (c >= 0) & (c < _W[k]) & (r >= 0) & (r < _H[k])
    out_ = np.ones(len(pts), bool)
    out_[ok] = occ[r[ok], c[ok]]
    return out_


def body(x, y, yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    q = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.stack([x + q[:, 0] * c - q[:, 1] * s_, y + q[:, 0] * s_ + q[:, 1] * c], 1)


def boxes(sid, pose):
    polys = []
    for j, k in enumerate(sid):
        if k < 0:
            continue
        if int(k) not in _fp:
            sh = po.shapes[int(k)]
            _fp[int(k)] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float)
        q = _fp[int(k)]; c, s_ = np.cos(pose[j, 2]), np.sin(pose[j, 2])
        polys.append(np.stack([pose[j, 0] + q[:, 0] * c - q[:, 1] * s_, pose[j, 1] + q[:, 0] * s_ + q[:, 1] * c], 1))
    return polys


def free(k, origin, direc, polys, maxd=4.0, step=0.02):
    """[m] from `origin` along `direc` to the first wall cell or polygon."""
    dd = np.arange(step, maxd, step)
    pts = origin[None] + dd[:, None] * direc[None]
    blk = wall_at(k, pts)
    for q in polys:
        blk |= Path(q).contains_points(pts)
    return float(dd[int(np.argmax(blk))]) if blk.any() else maxd


def pose_at(kd, b, pt):
    """The car's pose within decision `kd` nearest the contact point (the point is the sub-step's)."""
    s0, s1 = ST[kd, b], ST[kd + 1, b]
    if not np.isfinite(pt).all() or np.hypot(*(s1[:2] - s0[:2])) > 1.5:      # no point of its own, or a restart in between
        return s0, 0.0
    al = np.linspace(0, 1, 11)
    xy = s0[None, :2] + al[:, None] * (s1[None, :2] - s0[None, :2])
    a = float(al[int(np.argmin(((xy - pt[None]) ** 2).sum(1)))])
    dyaw = np.arctan2(np.sin(s1[2] - s0[2]), np.cos(s1[2] - s0[2]))
    return np.array([*(s0[:2] + a * (s1[:2] - s0[:2])), s0[2] + a * dyaw, s0[3] + a * (s1[3] - s0[3])]), a


# ---- classify
last_hit = {}
rows = []
pt_d = []
for e in sorted(events, key=lambda z: (z["kd"], z["b"], z["kind"])):
    kd, b, k = e["kd"], e["b"], e["kind"]
    pt = np.array([e["px"], e["py"]])
    s, a = pose_at(kd, b, pt)
    o = OP[kd, b] + a * (OP[kd + 1, b] - OP[kd, b]) if np.hypot(*(OP[kd + 1, b, :2] - OP[kd, b, :2])) < 1.5 else OP[kd, b]
    on = bool(ON[kd, b])
    if np.isfinite(pt).all():
        pt_d.append(float(np.hypot(*(pt - s[:2]))))
    k0 = max(0, kd - 10)
    v1, vm = float(ST[k0, b, 3]), float(np.abs(ST[k0:kd + 1, b, 3]).mean())
    polys = boxes(BS[kd, b], BP[kd, b])
    live = [j for j, q in enumerate(BS[kd, b]) if q >= 0]
    has_box = bool((BS[0, b] >= 0).any())
    r = dict(e, tid=int(TID[b]), v1=v1, v0=float(ST[kd, b, 3]), case=("상자+상대" if has_box and on else "상자" if has_box else "상대 차" if on else "빈 트랙"),
             x=float(s[0]), y=float(s[1]), yaw=float(s[2]), a=a, on=on, vo=float(o[3]), wt=float("nan"), wo=float("nan"))
    prev_kd = last_hit.get(b, -10 ** 6)
    last_hit[b] = kd
    ref = pt if np.isfinite(pt).all() else s[:2]
    d_box = min([float(np.hypot(*(BP[kd, b, j, :2] - ref))) for j in live], default=9.9)
    d_opp = float(np.hypot(*(o[:2] - s[:2]))) if on else 9.9
    r["d_box"], r["d_opp"] = d_box, d_opp
    r["t"] = kd * PS.DT
    r["jbox"] = int(min(live, key=lambda n: float(np.hypot(*(BP[kd, b, n, :2] - ref))))) if live else -1
    if on:                                                          # who went to whom: each car's speed along the line between them
        n_ = (o[:2] - s[:2]) / max(d_opp, 1e-6)
        r["to_ours"] = float(ST[kd, b, 3] * (np.cos(s[2]) * n_[0] + np.sin(s[2]) * n_[1]))
        r["to_theirs"] = float(-o[3] * (np.cos(o[2]) * n_[0] + np.sin(o[2]) * n_[1]))
    if kd - prev_kd <= 20:
        cls = "앞선 접촉에 이어짐"
    elif vm < 1.5:
        cls = "거의 서 있다가"
    elif k == 1 and live:
        j = min(range(len(live)), key=lambda n: float(np.hypot(*(BP[kd, b, live[n], :2] - ref))))
        q = polys[j]; c = BP[kd, b, live[j], :2]
        perp = np.array([-np.sin(s[2]), np.cos(s[2])])
        lat = (q - c[None]) @ perp
        rest = [polys[n] for n in range(len(polys)) if n != j] + ([body(*o[:3])] if on else [])
        w_l = free(r["tid"], c + perp * (lat.max() + 0.01), perp, rest)
        w_r = free(r["tid"], c + perp * (lat.min() - 0.01), -perp, rest)
        side = float((s[:2] - c) @ perp)
        r["wt"], r["wo"] = (w_l, w_r) if side >= 0 else (w_r, w_l)
        r["off"] = side
        cls = "넉넉한 쪽인데 닿음" if r["wt"] >= NEED else ("좁은 쪽으로 감" if r["wo"] >= NEED else "양쪽 다 좁음")
    elif k == 0:
        cls = "상대 차 옆에서 벽에" if d_opp < 1.5 else ("상자 옆에서 벽에" if d_box < 1.5 else "혼자 벽에")
    elif k == 2:
        c_, s_ = np.cos(s[2]), np.sin(s[2])
        dx = float((o[0] - s[0]) * c_ + (o[1] - s[1]) * s_)
        r["dx"] = dx
        cls = ("서 있는 상대를 박음" if abs(o[3]) < 0.3 else "달리는 앞차를 박음") if dx > 0.3 else ("뒤에서 박힘" if dx < -0.3 else "옆으로 닿음")
    else:
        cls = "기타"
    r["cls"] = cls
    rows.append(r)

name = f"{os.path.basename(os.path.dirname(ck_path))}_{os.path.basename(ck_path).replace('plan_', '').replace('.pt', '')}"
mins = Bl * secs / 60.0
print(f"{name}: 학습 세계, 학습 조건, 학습 차 {Bl}대 x {secs:.0f} s = {mins:.0f}대·분 | 차 폭 {2 * hw:.2f} m, 넉넉한 틈 = {NEED:.2f} m 이상 | "
      f"닿은 점과 차 중심 거리 중앙 {np.median(pt_d) if pt_d else float('nan'):.2f} m (차체 안팎이어야 맞음)", flush=True)
for k in range(3):
    rk = [r for r in rows if r["kind"] == k]
    nh = sum(r["hard"] for r in rk)
    print(f"[{KIND[k]}] 닿음 {len(rk)}건({len(rk) / mins:.2f}/분), 세게 {nh}건({nh / mins:.2f}/분)", flush=True)
    for cls in sorted({r["cls"] for r in rk}, key=lambda c: -sum(r["hard"] for r in rk if r["cls"] == c)):
        rc = [r for r in rk if r["cls"] == cls]; rh = [r for r in rc if r["hard"]]
        base = rh if rh else rc
        extra = ""
        if k == 1 and np.isfinite([r["wt"] for r in base]).any():
            extra = f" | 틈 중앙: 택한 쪽 {np.nanmedian([r['wt'] for r in base]):.2f} m, 반대쪽 {np.nanmedian([r['wo'] for r in base]):.2f} m"
        sv = [r["sev"] for r in base if np.isfinite(r["sev"])]
        print(f"    {cls}: 세게 {len(rh)}건({len(rh) / max(nh, 1) * 100:.0f} %), 전체 {len(rc)}건 | 속도 중앙 1 s 전 {np.median([r['v1'] for r in base]):.1f} -> "
              f"닿을 때 {np.median([r['v0'] for r in base]):.1f} m/s" + (f", 충격 {np.median(sv):.1f} m/s" if sv else "") + extra
              + " | 경우: " + " · ".join(f"{c} {sum(r['case'] == c for r in base)}" for c in ("빈 트랙", "상대 차", "상자", "상자+상대")
                                       if any(r['case'] == c for r in base)), flush=True)
# the start of the episode against the rest
m0, m1 = Bl * min(5.0, secs) / 60.0, Bl * max(secs - 5.0, 0.0) / 60.0
print("[에피소드 시작 5 s 안 / 그 뒤] 세게 닿음 /분: " + " | ".join(
    f"{KIND[k]} {sum(r['hard'] and r['kind'] == k and r['t'] < 5.0 for r in rows) / max(m0, 1e-6):.2f} / "
    f"{sum(r['hard'] and r['kind'] == k and r['t'] >= 5.0 for r in rows) / max(m1, 1e-6):.2f}" for k in range(3)), flush=True)
# car contacts: who went to whom
for cls in ("옆으로 닿음", "달리는 앞차를 박음"):
    rc = [r for r in rows if r["kind"] == 2 and r["cls"] == cls and r["hard"] and "to_ours" in r]
    if rc:
        ours = sum(r["to_ours"] > r["to_theirs"] for r in rc)
        print(f"[상대 차 / {cls}, 세게 {len(rc)}건] 두 차를 잇는 방향으로 다가간 속도 중앙: 우리 {np.median([r['to_ours'] for r in rc]):+.1f} m/s, "
              f"상대 {np.median([r['to_theirs'] for r in rc]):+.1f} m/s | 우리가 더 다가간 건 {ours}건, 상대가 더 다가온 건 {len(rc) - ours}건", flush=True)


# ---- every pass by a box: the clearance, and which side of the box the plans of the last second went
def dense(P, step):
    seg = np.hypot(*(P[1:] - P[:-1]).T)
    keep = np.concatenate([[True], seg > 1e-6])
    P = P[keep]
    if len(P) < 2:
        return P
    seg = np.hypot(*(P[1:] - P[:-1]).T); cum = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.arange(0.0, cum[-1] + 1e-9, step)
    return np.stack([np.interp(u, cum, P[:, 0]), np.interp(u, cum, P[:, 1])], 1)


def poly_dist(pts, q):
    """(n,) [m] from each point to the polygon `q` (0 inside)."""
    A, B_ = q, np.roll(q, -1, 0)
    ab = (B_ - A)[None]; ap = pts[:, None] - A[None]
    t = np.clip((ap * ab).sum(2) / np.maximum((ab * ab).sum(2), 1e-12), 0.0, 1.0)
    d = np.hypot(*np.moveaxis(ap - t[..., None] * ab, 2, 0)).min(1)
    d[Path(q).contains_points(pts)] = 0.0
    return d


hit_at = {}
for r in rows:
    if r["kind"] == 1 and r["jbox"] >= 0:
        hit_at.setdefault((r["b"], r["jbox"]), []).append((r["kd"], r["hard"]))
passes = []
C_ = BS.shape[2]
for b in range(Bl):
    for j in range(C_):
        live_t = BS[:, b, j] >= 0
        if not live_t.any():
            continue
        dist = np.hypot(*(ST[:T, b, :2] - BP[:, b, j, :2]).T)
        near = live_t & (dist < 2.0)
        if not near.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate([[0], near.astype(int), [0]])))
        for k_in, k_out in zip(edges[::2], edges[1::2] - 1):
            if k_out - k_in < 1 or np.hypot(*(ST[k_in + 1:k_out + 1, b, :2] - ST[k_in:k_out, b, :2]).T).max() > 1.5:
                continue                                              # a restart inside
            if np.abs(ST[k_in:k_out + 1, b, 3]).mean() < 1.5:
                continue
            sid, pose = BS[k_in, b].copy(), BP[k_in, b]
            only = np.full_like(sid, -1); only[j] = sid[j]
            q = boxes(only, pose)[0]
            c = pose[j, :2]

            def ahead(k):
                s = ST[k, b]
                return (c[0] - s[0]) * np.cos(s[2]) + (c[1] - s[1]) * np.sin(s[2])
            hits = [h for (kd_, h) in hit_at.get((b, j), []) if k_in <= kd_ <= k_out + 1]
            if not (ahead(k_in) > 0 and (ahead(k_out) < 0 or hits)):
                continue                                              # did not go by it
            path = dense(ST[max(0, k_in - 1):k_out + 2, b, :2], 0.02)
            clr = float(poly_dist(path, q).min() - hw) if len(path) > 1 else float("nan")
            kc = min([kd_ for (kd_, h) in hit_at.get((b, j), []) if k_in <= kd_ <= k_out + 1], default=k_in + int(np.argmin(dist[k_in:k_out + 1])))
            sides, touch = [], []
            for k in range(kc - 10, kc + 1):
                if k < 0 or np.hypot(*(ST[k, b, :2] - ST[kc, b, :2])) > 12.0:
                    sides.append(np.nan); touch.append(np.nan); continue
                pl = dense(np.concatenate([ST[k, b, None, :2], PL[k, b]], 0), 0.05)
                if len(pl) < 3:
                    sides.append(np.nan); touch.append(np.nan); continue
                dd = np.hypot(*(pl - c[None]).T); i = int(np.argmin(dd))
                if i == len(pl) - 1 and dd[i] > 0.5:                   # the box is beyond the plan's end
                    sides.append(np.nan); touch.append(np.nan); continue
                i0, i1 = max(0, i - 1), min(len(pl) - 1, i + 1)
                tg = pl[i1] - pl[i0]
                sides.append(float(np.sign(tg[0] * (c[1] - pl[i, 1]) - tg[1] * (c[0] - pl[i, 0]))))
                touch.append(float(poly_dist(pl, q).min() < hw))
            sd = np.array(sides); sd = sd[np.isfinite(sd) & (sd != 0)]
            # the room either side of the box and the speed, 1 s before (the walls and the other boxes; across the car's heading)
            k1 = kc - 10 if kc >= 10 and np.hypot(*(ST[kc - 10, b, :2] - ST[kc, b, :2])) < 12.0 else kc
            perp = np.array([-np.sin(ST[k1, b, 2]), np.cos(ST[k1, b, 2])])
            lat = (q - c[None]) @ perp
            rest = sid.copy(); rest[j] = -1
            oth = boxes(rest, pose)
            wl = free(int(TID[b]), c + perp * lat.max(), perp, oth); wr = free(int(TID[b]), c + perp * lat.min(), -perp, oth)
            passes.append(dict(b=b, j=j, kc=int(kc), clr=clr, hit=bool(hits), hard=any(hits), flips=int((np.diff(sd) != 0).sum()) if len(sd) > 1 else 0,
                               touch=touch, v=float(np.abs(ST[k_in:k_out + 1, b, 3]).mean()), wl=wl, wr=wr, v1=float(abs(ST[k1, b, 3]))))
if passes:
    cl = np.array([p_["clr"] for p_ in passes]); hd = np.array([p_["hard"] for p_ in passes]); ht = np.array([p_["hit"] for p_ in passes])
    ok = np.isfinite(cl)
    print(f"[상자 옆을 지나간 {len(passes)}건] 닿음 {int(ht.sum())}건(세게 {int(hd.sum())}건 = {hd.mean() * 100:.2f} %) | 안 닿은 건의 여유(차 옆면과 상자 사이): "
          + " · ".join(f"{q_}번째 백분위 {np.percentile(cl[ok & ~ht], q_) * 100:.0f} cm" for q_ in (5, 10, 25, 50, 75)), flush=True)
    for lab, m in (("세게 닿은 건", hd), ("안 닿은 건", ~ht)):
        sel = [p_ for p_, mm in zip(passes, m) if mm]
        if not sel:
            continue
        fl = np.mean([p_["flips"] > 0 for p_ in sel]) * 100
        tc = np.array([p_["touch"] for p_ in sel], float)             # (n, 11): 1.0 s before .. the closest decision
        txt = " · ".join(f"{(10 - i) / 10:.1f} s 전 {np.nanmean(tc[:, i]) * 100:.0f} %" for i in (0, 3, 5, 7, 9) if np.isfinite(tc[:, i]).any())
        print(f"    {lab} {len(sel)}건: 앞 1 s 동안 plan이 지나가는 쪽(왼/오)이 바뀐 건 {fl:.0f} % | plan이 그 상자를 스치는 결정의 비율: {txt}", flush=True)
    # is a hard contact likelier when both sides are open and alike (an output that cannot pick a side) or at speed?
    wl_, wr_ = np.array([p_["wl"] for p_ in passes]), np.array([p_["wr"] for p_ in passes])
    v1_ = np.array([p_["v1"] for p_ in passes])
    both = np.minimum(wl_, wr_) >= NEED
    asym = np.abs(wl_ - wr_) / np.maximum(wl_ + wr_, 1e-6)

    def rate(m):
        return f"{int(m.sum())}건 중 세게 {int(hd[m].sum())}건 = {hd[m].mean() * 100 if m.any() else 0:.2f} %"
    print(f"    [양옆 빈 폭, 1 s 전] 양쪽 다 차가 지나갈 폭({NEED:.2f} m 이상): {rate(both)} | 한쪽만: {rate(~both)}", flush=True)
    if both.sum() >= 30:
        q1, q2 = np.percentile(asym[both], [33.3, 66.7])
        print(f"    [양쪽 다 열린 건, 두 폭의 차이 |왼-오|/(왼+오)] 비슷함(<{q1:.2f}): {rate(both & (asym < q1))} | 중간: "
              f"{rate(both & (asym >= q1) & (asym < q2))} | 한쪽이 훨씬 넓음(>{q2:.2f}): {rate(both & (asym >= q2))}", flush=True)
    s1, s2 = np.percentile(v1_, [33.3, 66.7])
    print(f"    [1 s 전 속도] {s1:.1f} m/s 아래: {rate(v1_ < s1)} | {s1:.1f}~{s2:.1f}: {rate((v1_ >= s1) & (v1_ < s2))} | {s2:.1f} m/s 넘게: {rate(v1_ >= s2)}", flush=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.arange(-0.2, 1.21, 0.05)
    a1.hist(np.clip(cl[ok & ~ht], -0.2, 1.2) , bins=bins, color="#4c78a8", label=f"안 닿음 {int((ok & ~ht).sum())}건")
    a1.hist(np.clip(cl[ok & ht], -0.2, 1.2), bins=bins, color="#e45756", label=f"닿음 {int((ok & ht).sum())}건")
    a1.axvline(0.0, color="k", lw=0.8)
    a1.set_xlabel("지나간 여유: 차 옆면과 상자 사이 [m] (0 아래 = 겹침)"); a1.set_ylabel("건수"); a1.legend(frameon=False)
    a1.set_title("상자 옆을 얼마나 띄우고 지나가나", fontsize=11)
    xs = -(10 - np.arange(11)) / 10.0
    for lab, m, col in (("세게 닿은 건", hd, "#e45756"), ("안 닿은 건", ~ht, "#4c78a8")):
        sel = np.array([p_["touch"] for p_, mm in zip(passes, m) if mm], float)
        if len(sel):
            a2.plot(xs, np.nanmean(sel, 0) * 100, "-o", color=col, lw=2, ms=4, label=f"{lab} {len(sel)}건")
    a2.set_xlabel("가장 가까워지기(닿기) 전 시간 [s]"); a2.set_ylabel("plan이 그 상자를 스치는 결정 [%]"); a2.set_ylim(0, 100); a2.legend(frameon=False)
    a2.set_title("닿기 전 1 s 동안 plan이 그 상자 위를 지나가고 있었나", fontsize=11)
    for a_ in (a1, a2):
        a_.spines[["top", "right"]].set_visible(False); a_.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{name} · 학습 세계, 학습 조건 · 상자 옆을 지나간 {len(passes)}건", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); fn = os.path.join(out_dir, f"hits_{name}_pass.png"); plt.savefig(fn, dpi=90); plt.close(fig)
    print(f"그림 {fn} = 지나간 여유와 plan", flush=True)
json.dump(rows, open(os.path.join(out_dir, f"hits_{name}.json"), "w"), ensure_ascii=False)

# ---- scenes: one sheet per (kind, class), the hard ones first, different cars first
MAPC = ListedColormap(["#666666", "white"])
SLUG = {"벽": "wall", "상자": "box", "상대 차": "car"}
sheet = 0
for k in range(3):
    for cls in sorted({r["cls"] for r in rows if r["kind"] == k}):
        rc = sorted([r for r in rows if r["kind"] == k and r["cls"] == cls], key=lambda r: (not r["hard"], -(r["sev"] if np.isfinite(r["sev"]) else 0.0)))
        if sum(r["hard"] for r in rc) < 3:
            continue
        pick, seen = [], set()
        for r in rc:                                                  # one scene a car first
            if r["b"] not in seen:
                pick.append(r); seen.add(r["b"])
        pick = (pick + [r for r in rc if r not in pick])[:12]
        fig, axs = plt.subplots(3, 4, figsize=(20, 15.5))
        for ax in axs.flat:
            ax.set_axis_off()
        for ax, r in zip(axs.flat, pick):
            ax.set_axis_on(); ax.set_facecolor("#666666")
            kd, b = r["kd"], r["b"]
            occ, ext = grid(r["tid"])
            ax.imshow(~occ, cmap=MAPC, vmin=0, vmax=1, origin="lower", extent=ext, zorder=0)
            for q in boxes(BS[kd, b], BP[kd, b]):
                ax.fill(q[:, 0], q[:, 1], color="saddlebrown", alpha=0.9, lw=0, zorder=2)
            k0, k1 = max(0, kd - 20), min(T, kd + 1)
            tr = ST[k0:k1 + 1, b]
            tt = np.linspace(0, 1, len(tr))
            ax.scatter(tr[:, 0], tr[:, 1], c=tt, cmap="Blues", s=16, vmin=-0.3, vmax=1, zorder=3)
            if r["on"]:
                otr = OP[k0:k1 + 1, b]
                ax.scatter(otr[:, 0], otr[:, 1], c=tt, cmap="Reds", s=16, vmin=-0.3, vmax=1, zorder=3)
                oa = OP[kd, b] + r["a"] * (OP[kd + 1, b] - OP[kd, b]) if np.hypot(*(OP[kd + 1, b, :2] - OP[kd, b, :2])) < 1.5 else OP[kd, b]
                ax.add_patch(Polygon(body(*oa[:3]), closed=True, fc="tab:red", ec="k", alpha=0.85, zorder=5))
            ax.add_patch(Polygon(body(r["x"], r["y"], r["yaw"]), closed=True, fc="tab:blue", ec="k", alpha=0.85, zorder=5))
            pl = PL[max(0, kd - 5), b]; ax.plot(pl[:, 0], pl[:, 1], "k-", lw=1.6, zorder=4)
            pl = PL[kd, b]; ax.plot(pl[:, 0], pl[:, 1], "-", color="deeppink", lw=1.4, zorder=4)
            if np.isfinite(r["px"]):
                ax.plot(r["px"], r["py"], "*", color="gold", ms=16, mec="k", zorder=6)
            Z = 4.0 if k == 0 else 2.5
            ax.set_xlim(r["x"] - Z, r["x"] + Z); ax.set_ylim(r["y"] - Z, r["y"] + Z)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            t2 = f"틈: 택한 쪽 {r['wt']:.2f} / 반대쪽 {r['wo']:.2f} m" if np.isfinite(r["wt"]) else (f"상대 {r['vo']:.1f} m/s" if r["on"] else "")
            ax.set_title(f"{'세게' if r['hard'] else '살짝'} · {r['case']} · 1 s 전 {r['v1']:.1f} -> {r['v0']:.1f} m/s"
                         + (f" · 충격 {r['sev']:.1f}" if np.isfinite(r["sev"]) else "") + f"\n{t2} · {kd * PS.DT:.1f} s"
                         + (" · 원래 트랙" if r["tid"] == 0 else f" · 변형 {r['tid']}"), fontsize=10)
        nh = sum(r["hard"] for r in rc)
        fig.suptitle(f"{name} · {KIND[k]}에 닿음 · 갈래 '{cls}' (세게 {nh}건, 전체 {len(rc)}건) -- 파랑 = 우리 차와 2 s 자취, 빨강 = 상대 차, 갈색 = 상자, "
                     f"검은 선 = 0.5 s 전 plan, 분홍 선 = 닿는 결정의 plan, 별 = 닿은 점", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fn = os.path.join(out_dir, f"hits_{name}_{SLUG[KIND[k]]}_{sheet:02d}.png")
        plt.savefig(fn, dpi=70); plt.close(fig)
        print(f"그림 {fn} = {KIND[k]} / {cls}", flush=True)
        sheet += 1
