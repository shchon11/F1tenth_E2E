"""서 버린 차가 왜 후진을 안 하나, 후진만 하면 빠져나오나 (11-12: 코가 막힌 차는 후진 없이 끝까지 섬 -- 사용자: 닿은 뒤 스스로
빠져나오면 손으로 옮기기·위치 다시 잡기를 건너뛰므로 이득이 큼). 학습 세계(plan_rl.build), 학습 조건, policy 평균 행동, 고정 시드.
출발 3 s 뒤부터, 2 s 동안 0.3 m도 못 간 차를 '선 차'로 잡음(차마다 첫 번만). 그 순간부터:
  - 절반(무작위)은 그대로 둠, 절반은 0.8 s 동안 정해 둔 후진(곧게, 0.4 s 가속 + 0.4 s 감속, 약 0.8 m)을 시킨 뒤 policy에 돌려줌
    (후진 plan은 실제 속도·조향에서 그림: policy의 속도 추정이 틀려도 후진이 되게).
  - 결정마다 기록: policy의 속도 추정 `vhat`, 첫 0.1 s 속도 명령의 평균(-1..1) `asp`, 그 범위 `lo`/`hi` [m/s], 탐색 표준편차 `sd`,
    앞 +-10도 스캔의 가장 가까운 거리 `front` [m].
표: 선 순간의 실제 속도·속도 추정·첫 속도 명령 범위(후진이 가능한 범위인가)·후진이 뽑힐 확률, 그 뒤 8 s 안에 3 m 넘게 벗어난 비율
(그대로 / 후진시킴), 무엇에 막혔나(닿아 있는 것)별.
기록 npz는 drive_record.py와 같은 형식 + 위 값들 + 사건 목록(ev_*) -- rec_lib.Rec로 읽어 장면을 그림.
인자: checkpoint 출력npz [환경 수, 기본 512] [초, 기본 40]"""
import os, sys, json
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, PT_IDX
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import props as _props
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path, out_npz = sys.argv[1], sys.argv[2]
n_env = int(sys.argv[3]) if len(sys.argv) > 3 else 512
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
ep_s = float(sys.argv[5]) if len(sys.argv) > 5 and float(sys.argv[5]) > 0 else secs + 5.0
OV = {"episode_s": ep_s}
if len(sys.argv) > 6:
    OV.update(json.loads(sys.argv[6]))
restart_s = float(OV.pop("opp_restart_s", 0.0))
SEED = 4242
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
env, lim, world = PR.build(PR.WORLD, n_env, SEED, dev, fmt="v2", overrides=OV)
lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
po = env.procedural
rm_p = float(getattr(world, "removal_p", 0.0))
remover = SR.PropRemoval(env, rm_p, *[float(v) for v in getattr(world, "removal_s", (10.0, 30.0))], seed=SEED) if rm_p > 0 else None
restarter = SR.OppRestart(env, restart_s, PS.DT) if restart_s > 0 and env.M > 1 else None
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
COLS = ("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")
I_CNT = [CC.index(k) for k in COLS]
i_hk, i_hx, i_hy, i_hs = (CC.index(k) for k in ("held_kind", "held_x", "held_y", "held_severity"))
KN = list(getattr(env, "alt_teacher_kinds", ())) + ["raceline"]
ii = PT_IDX.to(dev)
T = int(secs / PS.DT)
R = {k: [] for k in ("st", "op", "on", "pl", "rs", "cnt", "ocnt", "held", "sid", "pose", "tid", "kind", "scale", "lap", "ors", "newp",
                     "vhat", "asp", "lo", "hi", "sd", "front", "script")}
from f1sim import dynamics as dyn
N_REV = 8                                                          # decisions of the scripted reverse: 4 speeding up backwards, 4 slowing
gen = torch.Generator().manual_seed(SEED)
treat_row = (torch.rand(Bl, generator=gen) < 0.5).to(dev)          # which cars get the scripted reverse when they stall
hist = torch.zeros(int(2.0 / PS.DT) + 1, Bl, 2, device=dev)        # positions of the last 2 s
age = torch.zeros(Bl, dtype=torch.long, device=dev)                # decisions since the episode began
ev_done = torch.zeros(Bl, dtype=torch.bool, device=dev)
rev_k = torch.full((Bl,), -1, dtype=torch.long, device=dev)        # the scripted reverse: decision index within it, -1 = none
EV = []
nb = env.last_result.scan.shape[1]
fov = float(getattr(env.sim.cfg.lidar, "fov", 4.71238898)); fov = fov if fov < 7 else np.deg2rad(fov)
kf = max(1, int(round(nb * np.deg2rad(10.0) / fov)))
HOLD = int(os.environ.get("F1SIM_HOLD", "1"))
p_held, hold_k = None, torch.zeros(Bl, dtype=torch.long, device=dev)
fresh = torch.ones(Bl, dtype=torch.bool, device=dev)              # rows that must plan anew (the start, a restart)


def snap():
    """the world as the decision about to be made sees it"""
    R["st"].append(env.sim.state[lid, :4].float().cpu()); R["op"].append(env.sim.state[others, :4].float().cpu())
    R["on"].append((~absent[others]).cpu())
    R["sid"].append(po.p_sid[eid].to(torch.int16).cpu()); R["pose"].append(po.p_poses[eid].float().cpu())
    R["tid"].append(env.sim.tid[lid].to(torch.int16).cpu())
    code = env.slot_kind_code[others] if hasattr(env, "slot_kind_code") else torch.full((Bl,), -1, device=dev, dtype=torch.long)
    R["kind"].append(torch.where(code < 0, torch.full_like(code, len(KN) - 1), code).to(torch.int8).cpu())
    sc = env.teacher.speed_scale[others] if torch.is_tensor(getattr(env.teacher, "speed_scale", None)) else torch.ones(Bl, device=dev)
    R["scale"].append(sc.float().cpu())


for kd in range(T):
    snap()
    _, mean, log_std, v_hat, p = plan_v2(actor, obs, lim)
    newp = fresh | (hold_k >= HOLD - 1)
    if HOLD > 1:
        # as plan_rl's --hold-p: a new plan is kept; rows inside a held one drive its next 0.1 s, re-expressed from where they are
        sel = lambda A, B_: torch.where(newp.view(-1, *([1] * (A.dim() - 1))), A, B_)
        p_held = p if p_held is None else {k_: sel(p[k_], p_held[k_]) if torch.is_tensor(p[k_]) and p[k_].dim() and p[k_].shape[0] == Bl
                                           else p[k_] for k_ in p}
        hold_k = torch.where(newp, torch.zeros_like(hold_k), hold_k + 1)
        p = PS.shift_plan(p_held, hold_k)
    st_now = env.sim.state[lid]
    v_true, d_true = st_now[:, dyn.IVX].float(), st_now[:, dyn.ISTEER].float()
    a = torch.tanh(mean)
    v0 = v_hat.detach().clamp(-PS.V_REV, PS.V_FWD)
    # the first step's speed range from where the policy believes it is, and the chance its exploration draws a reverse
    lo = PS.step_targets(torch.cat([a[:, :PS.N_STEP], -torch.ones_like(a[:, PS.N_STEP:])], 1), v0, obs.cmd_d, lim)[1][:, 0]
    hi = PS.step_targets(torch.cat([a[:, :PS.N_STEP], torch.ones_like(a[:, PS.N_STEP:])], 1), v0, obs.cmd_d, lim)[1][:, 0]
    sc_m = env.last_result.scan[lid].float()
    R["vhat"].append(v0.cpu()); R["asp"].append(a[:, PS.N_STEP].cpu()); R["lo"].append(lo.cpu()); R["hi"].append(hi.cpu())
    R["sd"].append(log_std.exp().expand_as(mean)[:, PS.N_STEP].cpu())
    R["front"].append(sc_m[:, nb // 2 - kf: nb // 2 + kf + 1].min(1).values.cpu())
    # a car that has not moved 0.3 m in 2 s (3 s or more into its episode), the first time
    hist = torch.cat([hist[1:], st_now[None, :, :2].float()], 0)
    net = (hist[-1] - hist[0]).norm(dim=1)
    stalled = (age >= int(3.0 / PS.DT)) & (net < 0.3) & ~ev_done & (rev_k < 0)
    if bool(stalled.any()):
        for b in torch.nonzero(stalled)[:, 0].tolist():
            EV.append((b, kd, int(treat_row[b])))
        ev_done |= stalled
        rev_k = torch.where(stalled & treat_row, torch.zeros_like(rev_k), rev_k)
    scripted = rev_k >= 0
    R["script"].append(scripted.cpu())
    if bool(scripted.any()):
        ds = lim.sv_max * PS.DT
        a_s = torch.zeros_like(a)
        a_s[:, 0] = (-d_true / ds).clamp(-1.0, 1.0)                    # steering back to straight
        a_s[:, PS.N_STEP:] = torch.where(rev_k < N_REV // 2, -1.0, 1.0)[:, None]
        a = torch.where(scripted[:, None], a_s, a)
        p2 = PS.roll(a, torch.where(scripted, v_true, v0), torch.where(scripted, d_true, obs.cmd_d), lim)
        p = {k_: (torch.where(scripted.view(-1, *([1] * (p[k_].dim() - 1))), p2[k_], p[k_])
                  if torch.is_tensor(p[k_]) and p[k_].dim() and p[k_].shape[0] == Bl else p[k_]) for k_ in p}
        rev_k = torch.where(scripted, rev_k + 1, rev_k)
        rev_k = torch.where(rev_k >= N_REV, torch.full_like(rev_k, -1), rev_k)
    out, _, (x, y, _) = drv.decision(a, p)
    fresh = out["restarted"].clone()
    age = torch.where(out["restarted"], torch.zeros_like(age), age + 1)
    if bool(out["restarted"].any()):
        hist[:, out["restarted"]] = env.sim.state[lid][out["restarted"], :2].float()[None]
        rev_k[out["restarted"]] = -1
    R["newp"].append(newp.cpu())
    if remover is not None:
        remover.step()
    ors = torch.zeros(env.B, dtype=torch.bool, device=dev)
    if restarter is not None:
        pre = env.sim.state[:, :2].clone()
        if restarter.step():
            ors = (env.sim.state[:, :2] - pre).norm(dim=1) > 1e-6          # the restarted cars are the ones that moved
    R["ors"].append(ors[others].cpu())
    R["pl"].append(torch.stack([x[:, ii], y[:, ii]], -1).cpu())
    R["rs"].append(out["restarted"].cpu()); R["lap"].append(out["lap"].float().cpu())
    cf = env.sim.contact_frame()
    R["cnt"].append(cf[lid].float()[:, I_CNT].cpu()); R["ocnt"].append(cf[others].float()[:, I_CNT].cpu())
    R["held"].append(cf[lid].float()[:, [i_hk, i_hx, i_hy, i_hs]].cpu())
snap()
D = {k: torch.stack(v).numpy() for k, v in R.items()}
D["ev"] = np.array(EV, np.int32).reshape(-1, 3)                       # car, decision, 1 = scripted reverse
if restarter is not None:
    print(f"{restart_s:g} s 동안 0.5 m도 못 간 상대 차 다시 출발 {restarter.count}번", flush=True)
trk = env.sim.track
D.update(edt=trk.edt.float().cpu().numpy(), t_off=trk.t_off.cpu().numpy(), t_H=trk.t_H.cpu().numpy(), t_W=trk.t_W.cpu().numpy(),
         t_res=trk.t_res.cpu().numpy(), t_origin=trk.t_origin.cpu().numpy(),
         car=np.array([float(env.cfg.vehicle.length), float(env.cfg.vehicle.width)]), dt=np.array(PS.DT),
         kinds=np.array(KN), cols=np.array(COLS), name=np.array(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"),
         ep_s=np.array(ep_s))
for k in sorted({int(s) for s in np.unique(D["sid"]) if s >= 0}):
    sh = po.shapes[k]
    D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez_compressed(out_npz, **D)
c = D["cnt"][-1].sum(0)
print(f"{D['name']}: 학습 차 {Bl}대 x {secs:.0f} s (에피소드 {ep_s:g} s), 다시 출발 {int(D['rs'].sum())}번 | 마지막 누적(다시 출발하면 0부터): "
      + " · ".join(f"{k} {int(v)}" for k, v in zip(COLS, c)) + f" | 저장 {out_npz}", flush=True)

# ------------------------------------------------------------------------------------------------ the table
ev = D["ev"]; dt = PS.DT; n8 = int(8.0 / dt)
xy = D["st"][:, :, :2].astype(float); vv = D["st"][:, :, 3].astype(float)
HK = {0: "아무것도 안 닿음", 1: "벽", 2: "상자", 3: "상대 차"}
print(f"선 차(2 s에 0.3 m 못 감, 차마다 첫 번): {len(ev)}건 / {Bl}대, 그대로 둠 {int((ev[:, 2] == 0).sum())} · 후진시킴 {int((ev[:, 2] == 1).sum())}")
rows = []
for b, k, tr in ev:
    if k + n8 >= T:
        continue
    seg = slice(k, k + n8 + 1)
    stop = np.nonzero(D["rs"][k:k + n8, b])[0]                         # an episode restart ends the look
    e = k + (int(stop[0]) if len(stop) else n8)
    far = np.linalg.norm(xy[k:e + 1, b] - xy[k, b], axis=1).max()
    own_rev = bool((vv[k:e + 1, b] < -0.3).any()) if not tr else True
    hk = int(round(float(D["held"][max(k - 1, 0), b, 0])))
    rows.append((b, k, tr, far, own_rev, hk, vv[k, b], D["vhat"][k, b], D["lo"][k, b], D["hi"][k, b], D["asp"][k, b], D["sd"][k, b],
                 D["front"][k, b], bool((D["sid"][k, b] >= 0).any()), bool(D["on"][k, b])))
rows = np.array(rows, float)
if len(rows):
    from math import erf, sqrt, atanh
    def p_rev(lo, hi, asp, sd):
        """chance one draw of the exploration noise makes the first step's speed negative"""
        if lo >= -0.02:
            return 0.0
        a0 = max(min(-1.0 + 2.0 * (0.0 - lo) / max(hi - lo, 1e-6), 0.999), -0.999)   # the action that commands 0 m/s
        m = atanh(max(min(asp, 0.999999), -0.999999))
        return 0.5 * (1.0 + erf((atanh(a0) - m) / (sd * sqrt(2.0))))
    pr_ = np.array([p_rev(r[8], r[9], r[10], r[11]) for r in rows])
    print(f"8 s 뒤까지 볼 수 있는 {len(rows)}건. 선 순간: 실제 속도 중앙 {np.median(rows[:, 6]):+.2f} m/s, policy의 속도 추정 중앙 "
          f"{np.median(rows[:, 7]):+.2f} m/s (0.3 m/s 넘게 앞으로 간다고 봄 {100 * (rows[:, 7] > 0.3).mean():.0f} %), "
          f"첫 0.1 s 속도 범위 중앙 {np.median(rows[:, 8]):+.2f}..{np.median(rows[:, 9]):+.2f} m/s (후진이 범위에 없음 {100 * (rows[:, 8] >= -0.02).mean():.0f} %), "
          f"속도 명령 평균(-1..1) 중앙 {np.median(rows[:, 10]):+.2f}, 탐색 표준편차 {np.median(rows[:, 11]):.2f}, "
          f"탐색 한 번에 후진이 뽑힐 확률 중앙 {100 * np.median(pr_):.1f} % (평균 {100 * pr_.mean():.1f} %), 앞 스캔 거리 중앙 {np.median(rows[:, 12]):.2f} m")
    def line(m, label):
        if m.sum() == 0:
            return
        r = rows[m]
        c, t_ = r[r[:, 2] == 0], r[r[:, 2] == 1]
        f = lambda q: f"{int((q[:, 3] >= 3.0).sum())}/{len(q)} ({100 * (q[:, 3] >= 3.0).mean():.0f} %)" if len(q) else "0건"
        print(f"  {label:<18} 그대로 둠: 3 m 넘게 벗어남 {f(c)}" + (f", 스스로 후진 {int(c[:, 4].sum())}/{len(c)}" if len(c) else "")
              + f" | 후진시킴: {f(t_)}")
    line(np.ones(len(rows), bool), "전체")
    for hk, nm in HK.items():
        line(rows[:, 5] == hk, f"닿아 있는 것: {nm}")
    line(rows[:, 12] < 0.35, "앞 스캔 0.35 m 안")
    line(rows[:, 12] >= 0.35, "앞 스캔 0.35 m 밖")
    line(rows[:, 13] > 0, "상자 있는 경주")
    line(rows[:, 13] == 0, "상자 없는 경주")
