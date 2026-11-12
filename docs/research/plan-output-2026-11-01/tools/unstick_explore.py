"""서 버린 차가 학습의 탐색만으로 빠져나오는 길을 만나는가 (11-12; 닿은 뒤 스스로 빠져나오기는 policy가 RL로 배워야 함 -- 정해 둔 후진은 쓰지 않음).
RL이 '뒤로 물러났다 돌아가기'를 배우려면 학습 중 탐색이 그 길을 실제로 밟아야 함. 학습(plan_rl)의 탐색은 결정(0.1 s)마다 새로 뽑는 noise라서
1 s쯤 이어지는 후진은 거의 뽑히지 않는다는 것이 가설. 이전 학습 코드(state_rl --noise-corr / --corr-slow, 11-06~07)에는 noise를 시간적으로 이어 주는
옵션이 있었고 지금 학습 코드에는 없음.
학습 세계(plan_rl.build), 학습 조건, policy 평균 행동으로 달리다가, 출발 3 s 뒤부터 2 s 동안 0.3 m도 못 간 차를 '선 차'로 잡음(차마다 첫 번만).
그 순간부터 그 차만 셋 중 하나(차마다 무작위)로 몲 -- 에피소드가 끝날 때까지:
  0 = 평균 행동 그대로(평가와 같음)
  1 = 학습과 같은 탐색: 첫 0.1 s의 조향·속도 출력에 학습의 표준편차(속도는 --speed-std-floor 포함)로 결정마다 새로 뽑은 noise
  2 = 같은 크기의 noise를 시간적으로 이어서: eps <- c * eps + sqrt(1 - c^2) * 새 noise (c = UNSTICK_CORR, 기본 0.9 = 1 s쯤)
표: 묶음별로 선 뒤 8 s / 15 s 안에 3 m 넘게 벗어난 비율, 후진(-0.3 m/s 아래)이 나온 비율, 벗어날 때까지 걸린 시간, 그 사이 새로 세게 닿은 횟수,
무엇에 막혔나별. 반증: 2가 1보다 많이 벗어나지 않으면 noise의 시간 구조는 원인이 아님.
기록 npz는 unstick_probe.py와 같은 형식(+ `armon`, ev의 셋째 칸 = 묶음).
인자: checkpoint 출력npz [환경 수, 기본 512] [초, 기본 45]        (환경변수 UNSTICK_SEED 기본 4242, UNSTICK_CORR 기본 0.9)
  또는 표만: --table 기록npz [기록npz ...]"""
import os, sys, json
import numpy as np

ARMS = ("평균 행동 그대로", "학습과 같은 탐색(결정마다 새 noise)", "noise를 시간적으로 이음")
HK = {0: "아무것도 안 닿음", 1: "벽", 2: "상자", 3: "상대 차"}


def rows_of(D):
    """per stalled car: arm, far within 8 s / 15 s, time to 3 m [s] (nan = never), reversed, held kind, new hard contacts, stood [s]"""
    dt = float(D["dt"]); T = D["st"].shape[0] - 1; n8, n15 = int(8.0 / dt), int(15.0 / dt)
    xy = D["st"][:, :, :2].astype(float); vv = D["st"][:, :, 3].astype(float)
    cols = [str(c) for c in D["cols"]]; ih = [cols.index(k) for k in ("hard_wall", "hard_prop", "hard_car")]
    out = []
    for b, k, arm in D["ev"]:
        if k + n15 >= T:
            continue
        stop = np.nonzero(D["rs"][k:k + n15, b])[0]                       # an episode restart ends the look
        if len(stop):
            continue
        d = np.linalg.norm(xy[k:k + n15 + 1, b] - xy[k, b], axis=1)
        j = np.nonzero(d >= 3.0)[0]
        t_free = j[0] * dt if len(j) else np.nan
        e = k + (int(j[0]) if len(j) else n15)
        hard = D["cnt"][e, b][ih].sum() - D["cnt"][k, b][ih].sum()
        out.append((arm, d[:n8 + 1].max() >= 3.0, d.max() >= 3.0, t_free, (vv[k:e + 1, b] < -0.3).any(),
                    int(round(float(D["held"][max(k - 1, 0), b, 0]))), max(hard, 0.0), (np.abs(vv[k:e + 1, b]) < 0.3).sum() * dt,
                    float(D["sd"][k, b]), float(D["asp"][k, b]), float(D["vhat"][k, b]), float(D["lo"][k, b])))
    return np.array(out, float).reshape(-1, 12)


def table(rows, corr):
    print(f"선 뒤 15 s를 볼 수 있는 선 차 {len(rows)}대. 선 순간: 속도 명령 평균(-1..1) 중앙 {np.median(rows[:, 9]):+.2f}, 탐색 표준편차(속도) 중앙 "
          f"{np.median(rows[:, 8]):.2f}, policy의 속도 추정 중앙 {np.median(rows[:, 10]):+.2f} m/s, 첫 0.1 s에 후진이 범위에 없음 "
          f"{100 * (rows[:, 11] >= -0.02).mean():.0f} %")

    def line(m, label):
        if m.sum() == 0:
            return
        print(f"  [{label}]")
        for a, nm in enumerate(ARMS):
            r = rows[m & (rows[:, 0] == a)]
            if len(r) == 0:
                print(f"    {nm:<22} 0대"); continue
            tf = r[:, 3][np.isfinite(r[:, 3])]
            print(f"    {nm + (f' (c = {corr:g})' if a == 2 else ''):<22} {len(r):3d}대 | 3 m 넘게 벗어남: 8 s 안 {int(r[:, 1].sum())} ({100 * r[:, 1].mean():.0f} %), "
                  f"15 s 안 {int(r[:, 2].sum())} ({100 * r[:, 2].mean():.0f} %) | 후진이 나옴 {int(r[:, 4].sum())} ({100 * r[:, 4].mean():.0f} %) | "
                  f"벗어날 때까지 중앙 {np.median(tf) if len(tf) else float('nan'):.1f} s | 그 사이 새로 세게 닿음 평균 {r[:, 6].mean():.2f}번 | 서 있던 시간 중앙 {np.median(r[:, 7]):.1f} s")

    line(np.ones(len(rows), bool), "전체")
    for hk, nm in HK.items():
        line(rows[:, 5] == hk, f"닿아 있는 것: {nm}")


if sys.argv[1] == "--table":
    rr, corr = [], 0.9
    for fn in sys.argv[2:]:
        z = np.load(fn); rr.append(rows_of(z)); corr = float(z["corr"])
        print(f"{fn.rsplit('/', 1)[-1]}: {z['name']} · 시드 {int(z['seed'])} · 선 차 {len(z['ev'])}대 / 학습 차 {z['st'].shape[1]}대")
    table(np.concatenate(rr, 0), corr)
    sys.exit(0)

os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import math
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, PT_IDX
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import props as _props
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn

torch.set_grad_enabled(False)
ck_path, out_npz = sys.argv[1], sys.argv[2]
n_env = int(sys.argv[3]) if len(sys.argv) > 3 else 512
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 45.0
ep_s = secs + 5.0
OV = {"episode_s": ep_s}
SEED = int(os.environ.get("UNSTICK_SEED", "4242")); CORR = float(os.environ.get("UNSTICK_CORR", "0.9"))
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
restart_s = float(ck["args"].get("opp_restart_s", 0.0) or 0.0)       # the other cars as the training has them
env, lim, world = PR.build(PR.WORLD, n_env, SEED, dev, fmt="v2", overrides=OV)
lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
floor = float(ck["args"].get("speed_std_floor", 0.0) or 0.0)
if floor > 0:
    actor.log_std_floor[PS.N_STEP] = math.log(floor)                   # the training's exploration: the first step's speed
tailc = float(ck["args"].get("tail_consistency", 0.0) or 0.0) > 0
rl_mask = torch.ones(PS.ACT_DIM, device=dev)
if tailc:
    rl_mask.zero_(); rl_mask[0] = 1.0; rl_mask[PS.N_STEP] = 1.0        # as the training: noise on the first step only
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
                     "vhat", "asp", "lo", "hi", "sd", "front", "script", "armon")}
gen = torch.Generator().manual_seed(SEED)
arm_row = torch.randint(0, len(ARMS), (Bl,), generator=gen).to(dev)  # how a car is driven once it stalls
gn = torch.Generator(device=dev); gn.manual_seed(SEED + 1)
eps = torch.zeros(Bl, PS.ACT_DIM, device=dev)
arm_on = torch.zeros(Bl, dtype=torch.bool, device=dev)
hist = torch.zeros(int(2.0 / PS.DT) + 1, Bl, 2, device=dev)        # positions of the last 2 s
age = torch.zeros(Bl, dtype=torch.long, device=dev)                # decisions since the episode began
ev_done = torch.zeros(Bl, dtype=torch.bool, device=dev)
EV = []
nb = env.last_result.scan.shape[1]
fov = float(getattr(env.sim.cfg.lidar, "fov", 4.71238898)); fov = fov if fov < 7 else np.deg2rad(fov)
kf = max(1, int(round(nb * np.deg2rad(10.0) / fov)))


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
    st_now = env.sim.state[lid]
    a = torch.tanh(mean)
    v0 = v_hat.detach().clamp(-PS.V_REV, PS.V_FWD)
    lo = PS.step_targets(torch.cat([a[:, :PS.N_STEP], -torch.ones_like(a[:, PS.N_STEP:])], 1), v0, obs.cmd_d, lim)[1][:, 0]
    hi = PS.step_targets(torch.cat([a[:, :PS.N_STEP], torch.ones_like(a[:, PS.N_STEP:])], 1), v0, obs.cmd_d, lim)[1][:, 0]
    sc_m = env.last_result.scan[lid].float()
    sd = log_std.exp().expand_as(mean)
    R["vhat"].append(v0.cpu()); R["asp"].append(a[:, PS.N_STEP].cpu()); R["lo"].append(lo.cpu()); R["hi"].append(hi.cpu())
    R["sd"].append(sd[:, PS.N_STEP].cpu())
    R["front"].append(sc_m[:, nb // 2 - kf: nb // 2 + kf + 1].min(1).values.cpu())
    # a car that has not moved 0.3 m in 2 s (3 s or more into its episode), the first time
    hist = torch.cat([hist[1:], st_now[None, :, :2].float()], 0)
    net = (hist[-1] - hist[0]).norm(dim=1)
    stalled = (age >= int(3.0 / PS.DT)) & (net < 0.3) & ~ev_done
    new = torch.randn(Bl, PS.ACT_DIM, device=dev, generator=gn)
    if bool(stalled.any()):
        for b in torch.nonzero(stalled)[:, 0].tolist():
            EV.append((b, kd, int(arm_row[b])))
        ev_done |= stalled
        arm_on |= stalled
        eps = torch.where(stalled[:, None], torch.randn(Bl, PS.ACT_DIM, device=dev, generator=gn), eps)
    held = (arm_row == 2)[:, None]
    eps = torch.where(held, CORR * eps + math.sqrt(1.0 - CORR * CORR) * new, new)
    noisy = arm_on & (arm_row > 0)
    R["script"].append(torch.zeros(Bl, dtype=torch.bool)); R["armon"].append(arm_on.cpu())
    if bool(noisy.any()):
        a2 = torch.tanh(mean + sd * eps * rl_mask)
        a = torch.where(noisy[:, None], a2, a)
        p2 = PS.roll(a, v0, obs.cmd_d, lim)                            # as the training: from the speed estimate and the last command
        p = {k_: (torch.where(noisy.view(-1, *([1] * (p[k_].dim() - 1))), p2[k_], p[k_])
                  if torch.is_tensor(p[k_]) and p[k_].dim() and p[k_].shape[0] == Bl else p[k_]) for k_ in p}
    out, _, (x, y, _) = drv.decision(a, p)
    age = torch.where(out["restarted"], torch.zeros_like(age), age + 1)
    if bool(out["restarted"].any()):
        hist[:, out["restarted"]] = env.sim.state[lid][out["restarted"], :2].float()[None]
        arm_on &= ~out["restarted"]
    R["newp"].append(torch.ones(Bl, dtype=torch.bool))
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
D["ev"] = np.array(EV, np.int32).reshape(-1, 3)                       # car, decision, arm
trk = env.sim.track
D.update(edt=trk.edt.float().cpu().numpy(), t_off=trk.t_off.cpu().numpy(), t_H=trk.t_H.cpu().numpy(), t_W=trk.t_W.cpu().numpy(),
         t_res=trk.t_res.cpu().numpy(), t_origin=trk.t_origin.cpu().numpy(),
         car=np.array([float(env.cfg.vehicle.length), float(env.cfg.vehicle.width)]), dt=np.array(PS.DT),
         kinds=np.array(KN), cols=np.array(COLS), name=np.array(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"),
         ep_s=np.array(ep_s), seed=np.array(SEED), corr=np.array(CORR), arms=np.array(ARMS))
for k in sorted({int(s) for s in np.unique(D["sid"]) if s >= 0}):
    sh = po.shapes[k]
    D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez_compressed(out_npz, **D)
print(f"{D['name']} · 시드 {SEED}: 학습 차 {Bl}대 x {secs:.0f} s, 선 차 {len(EV)}대 (묶음별 "
      + " / ".join(str(int((D['ev'][:, 2] == a_).sum())) for a_ in range(len(ARMS))) + f") | 속도 표준편차 바닥 {floor:g}, "
      f"noise는 {'첫 0.1 s의 조향·속도에만' if tailc else '출력 전체에'} | 저장 {out_npz}", flush=True)
table(rows_of(D), CORR)
