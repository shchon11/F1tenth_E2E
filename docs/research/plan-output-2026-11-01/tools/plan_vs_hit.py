"""상자에 세게 부딪힐 때, 그 직전 policy의 plan은 상자를 비켜 갔나 (11-05 23:40, 사용자: "장애물 계속 못피하는건 왜그러지").
두 경우를 가름: (A) plan 자체가 상자를 지나감 = policy가 피하려고 하지 않음(학습 쪽), (B) plan은 차 폭까지 넣고 비켜 갔는데 차가 닿음 = plan과
실제 주행이 어긋남(11-01: PPO wp4 plan이 MAP 주행 경로와 0.4~0.6 m 어긋났음). console_layouts.py와 같은 세션·layout 순서(시드, env.reset)로
달리며 결정마다 plan(world 좌표, 콘솔 그림의 그 plan)을 기록, 세게 상자에 부딪힐 때마다 0.25 / 0.5 / 1.0 s 전 plan과 부딪힌 상자 footprint
사이 최소 간격 - 차 반폭. 0보다 크면 plan은 비켜 감 -- 단 가장 가까운 점이 plan 끝이면 plan이 상자까지 닿지 않은 것(plan 길이 = 0.6 s x 속도). 또 부딪힌 상자 크기 vs layout 전체 상자 크기(입력에 상자 크기가 없음:
slot은 중심 위치만). 인자: 체크포인트 맵id 초 layout수 시드"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
from matplotlib.path import Path
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import props as _props
from f1sim.learn import state_rl as SR


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, n_lay, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
env = s["env"]; proc = env.procedural
hw = 0.5 * float(env.cfg.vehicle.width)
i_hd = CC.index("hard_prop")
_fp = {}


def footprints(slots=False):
    out, idx = [], []
    sid = proc.p_sid[0].cpu().numpy(); pose = proc.p_poses[0].cpu().numpy()
    for j, k in enumerate(sid):
        if k < 0:
            continue
        idx.append(j)
        sh = proc.shapes[int(k)]
        if int(k) not in _fp:
            _fp[int(k)] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float)
        p = _fp[int(k)]; c, sn = np.cos(pose[j, 2]), np.sin(pose[j, 2])
        out.append(np.stack([pose[j, 0] + p[:, 0] * c - p[:, 1] * sn, pose[j, 1] + p[:, 0] * sn + p[:, 1] * c], 1))
    return (out, idx) if slots else out


def poly_dist(pts, poly, where=False):
    """min distance from points (n, 2) to a closed polygon (m, 2); 0 inside. where: also the index of the nearest point"""
    ins = Path(poly).contains_points(pts)
    a = poly; b = np.roll(poly, -1, 0); ab = b - a
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)[None], 0, 1)
    q = a[None] + t[..., None] * ab[None]
    d = np.sqrt(((pts[:, None, :] - q) ** 2).sum(-1)).min(1)
    d = np.where(ins, 0.0, d)
    i = int(np.argmin(d))
    return (float(d[i]), i) if where else float(d[i])


def size(poly):
    return float(np.sqrt(((poly[:, None] - poly[None]) ** 2).sum(-1)).max())


LAGS = (0.25, 0.5, 1.0)
res = {L_: [] for L_ in LAGS}
hit_sizes, all_sizes = [], []
vis_rows = []                                                          # per hit: (size, seen at 1.0 s, seen at 0.5 s, seen for [s] before)
lid0 = torch.zeros(1, dtype=torch.long, device=env.sim.state.device)
for L in range(n_lay):
    if L > 0:
        s["obs"], _ = env.reset()
    fps = footprints()
    all_sizes += [size(p) for p in fps]
    plans = []                                                         # (t, world plan (n, 2))
    vhist = []                                                         # (t, visible (C,) bool) every decision
    prev = float(env.sim.contact_frame()[0, i_hd])
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        t = (k + 1) * 0.025
        if k % 2 == 0:
            vp = SR.visible_props(env, lid0)
            if vp is not None:
                vhist.append((t, vp[0].cpu().numpy().copy()))
                vhist = vhist[-160:]
            fr = w._snapshot(s)
            if fr.get("plan") is not None:
                pl = np.asarray(fr["plan"], dtype=float)[:, :2]
                plans.append((t, pl))
                plans = plans[-40:]
        hd = float(env.sim.contact_frame()[0, i_hd])
        if hd > prev:
            car = env.sim.state[0, :2].cpu().numpy()
            fps, slot = footprints(slots=True)
            j = int(np.argmin([poly_dist(car[None], p) for p in fps]))
            hit_sizes.append(size(fps[j]))
            c_ = slot[j]
            seen = lambda lag: any(bool(v[c_]) for (tt, v) in vhist if t - lag - 0.05 <= tt <= t - lag + 0.05)
            run_ = 0.0
            for (tt, v) in reversed(vhist):
                if tt > t - 0.3:
                    continue                                           # the last 0.3 s: the crate is under the nose
                if not bool(v[c_]):
                    break
                run_ = t - tt
            vis_rows.append((size(fps[j]), seen(1.0), seen(0.5), run_))
            for lag in LAGS:
                old = [p for (tt, p) in plans if tt <= t - lag]
                if old and len(old[-1]):
                    g, i = poly_dist(old[-1], fps[j], where=True)
                    res[lag].append((g - hw, i >= len(old[-1]) - 2))   # nearest at the plan's end: it stops short of the crate
        prev = hd
    print(f"layout {L}: 세게 상자 누적 {len(hit_sizes)}", flush=True)
name = f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)}"
print(f"== {name} {mp} {n_lay} layout: 상자에 세게 {len(hit_sizes)}번", flush=True)
for lag in LAGS:
    if res[lag]:
        g = np.array([x[0] for x in res[lag]]); short = np.array([x[1] for x in res[lag]])
        print(f"  {lag:.2f} s 전 plan: 상자를 지나감 {int((g <= 0).sum())} / 아슬아슬(0~5 cm) {int(((g > 0) & (g <= 0.05)).sum())} / "
              f"차 폭 넣고 비켜 감 {int(((g > 0.05) & ~short).sum())} / plan이 상자 앞에서 끝남 {int(((g > 0.05) & short).sum())} "
              f"(간격 중간값 {np.median(g):+.2f} m)", flush=True)
if vis_rows:
    vr = np.array(vis_rows, dtype=float)
    print(f"  부딪힌 상자가 actor에게 보였나(LiDAR 시야·가림 기준, actor 입력과 같은 판정): 1.0 s 전 {int(vr[:, 1].sum())}/{len(vr)}, "
          f"0.5 s 전 {int(vr[:, 2].sum())}/{len(vr)} | 부딪히기 0.3 s 전까지 계속 보인 시간 중간값 {np.median(vr[:, 3]):.2f} s", flush=True)
    sm = vr[:, 0] < np.median(np.array(all_sizes))
    if sm.any() and (~sm).any():
        print(f"  작은 상자(전체 중간값보다 작음) {int(sm.sum())}번: 0.5 s 전 보임 {int(vr[sm, 2].sum())} | 큰 상자 {int((~sm).sum())}번: 0.5 s 전 보임 "
              f"{int(vr[~sm, 2].sum())}", flush=True)
hs, al = np.array(hit_sizes), np.array(all_sizes)
if len(hs):
    print(f"  상자 크기(가장 긴 대각선): 부딪힌 상자 중간값 {np.median(hs):.2f} m vs layout 전체 {np.median(al):.2f} m | "
          f"0.5 m 넘는 큰 상자 비율: 부딪힌 것 {np.mean(hs > 0.5) * 100:.0f} % vs 전체 {np.mean(al > 0.5) * 100:.0f} %", flush=True)
