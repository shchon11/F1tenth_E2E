"""학습 세계 주행을 통째로 기록 (11-11: 벽으로 직진하는 건과 출발 직후 접촉을 장면으로 파 보려고 -- 그림을 고칠 때마다 다시 달리지 않게
기록과 그림을 나눔). 학습 세계(plan_rl.build, 상자 치움 포함), 학습 조건(plan대로 정확히), policy 평균 행동, 고정 시드.
결정(0.1 s)마다 학습 차·상대 차의 자세와 속도, 상대 차가 있나, plan 10점, 다시 출발했나, 접촉 누적(벽·상자·상대 차, 전체 / 세게),
상자 종류·자세, 상대 차 종류·속도 배율을 npz 하나로 저장. 지도(벽 거리 격자)와 상자 외곽선도 같이.
인자: checkpoint 출력npz [환경 수, 기본 512] [초, 기본 40] [에피소드 길이 s, 기본 초 + 5 = 통째] [세계 설정 덮어쓰기 JSON]
덮어쓰기 JSON의 "opp_restart_s"는 세계 설정이 아니라 선 상대 차를 다시 출발시키는 시간(state_rl.OppRestart); 그 결정은 `ors`에 기록."""
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
R = {k: [] for k in ("st", "op", "on", "pl", "rs", "cnt", "ocnt", "held", "sid", "pose", "tid", "kind", "scale", "lap", "ors")}


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
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    out, _, (x, y, _) = drv.decision(torch.tanh(mean), p)
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
