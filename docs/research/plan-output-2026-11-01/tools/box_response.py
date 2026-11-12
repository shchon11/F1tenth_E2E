"""plan이 상자에 언제부터 반응하나 (11-11: 세게 닿는 상자는 1~2.5 s 전부터 보이던 것 -- 보이는 상자에 늦게 반응하는지 직접 잼).
학습 세계, 학습 조건, policy 평균 행동, 고정 시드. 결정마다 policy를 두 번 돌림: 실제 스캔으로 한 번(이 plan대로 달림), 같은 자리에서
상자만 지운 스캔으로 한 번(처음부터 상자를 한 번도 못 본 것으로 -- 스캔 3장과 기억 상태를 따로 이어 감; 상대 차·벽은 그대로).
두 plan의 차이 = 상자 때문에 바뀐 것. 상자 옆을 지나간 건마다 지나가기 전 시각별로: 그 상자에 닿은 스캔 빔 수, 거리, plan의 1.0 s 점과
0.5 s 점이 옮겨진 거리, 첫 0.1 s 점이 옮겨진 거리(차가 실제로 가는 점), 0.5 s 점 속도 차이, 상자 없는 plan / 실제 plan이 상자를
지나는가(상자 중심에서 0.40 m 안).
인자: checkpoint 출력npz [환경 수, 기본 1024] [초, 기본 40]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, plan_world, PT_IDX
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import props as _props
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path, out_npz = sys.argv[1], sys.argv[2]
n_env = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
SEED = int(os.environ.get("F1SIM_EVAL_SEED", 4242))
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
A = ck.get("args") or {}
if A.get("plan_lat_margin") is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(A["plan_lat_margin"])
env, lim, world = PR.build(PR.WORLD, n_env, SEED, dev, fmt="v2", overrides={"episode_s": secs + 5.0})
lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
pr = PlanReward(env, float(A.get("gamma", 0.995)), dev)
obs = PlanObs2(env, lim, pr.goal)
if getattr(actor, "branches", 1) == 2:
    obs.prev_branch = torch.zeros(Bl, device=dev)   # a two-plan policy's input exists before the first decision (state_in below)
drv = PR.Driver(env, lim, pr, obs)
po = env.procedural
rm_p = float(getattr(world, "removal_p", 0.0))
remover = SR.PropRemoval(env, rm_p, *[float(v) for v in getattr(world, "removal_s", (10.0, 30.0))], seed=SEED) if rm_p > 0 else None
rs_s = float(os.environ.get("F1SIM_OPP_RESTART_S", A.get("opp_restart_s", 0.0) or 0.0))
restarter = SR.OppRestart(env, rs_s, PS.DT) if rs_s > 0 and env.M > 1 else None
eid = env.sim.eid[lid]
lidar = env.sim.lidar
N = int(lidar.angles.numel()); fov = float(lidar.fov)
MOUNT_X = float(env.sim.cfg.lidar.mount_x)
ii = PT_IDX.to(dev)
I_CNT = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")]
B_all = env.B


def static_scan(props: bool):
    """noise-free scan of the learner rows from where they stand, walls (+ boxes if `props`), no cars"""
    tr = lidar.track
    keep = tr.has_props
    cut = lambda t: t[lid] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == B_all else t
    tr.has_props = bool(props and keep)
    try:
        _, true, _ = lidar.scan(env.sim.state[lid, :3], None, {k: cut(v) for k, v in env.sim.P.items()}, motion_distortion=False,
                                noisy=False, tid=env.sim.tid[lid], cars=None, compiled=False, eid=eid)
    finally:
        tr.has_props = keep
    return true.float()


def no_box_scan():
    """the scan the actor gets now with the box returns replaced by the wall behind them; also which box slots the scan shows"""
    real = env.last_result.scan[lid].float()
    r_w, r_n = static_scan(True), static_scan(False)
    box = (r_w < r_n - 0.03).float()[:, None]
    boxd = Fn.max_pool1d(box, 7, 1, 3)[:, 0] > 0.5                        # +-3 beams: the real scan is taken while moving
    rw_min = -Fn.max_pool1d(-r_w[:, None], 7, 1, 3)[:, 0]
    rep = boxd & (real < r_n - 0.05) & (real > rw_min - 0.2)              # nearer than the box = a car in front of it: kept
    cf = torch.where(rep, r_n, real)
    # which boxes the scan shows: beams that end on a box (not the wall behind) within 0.8 m of that box's centre
    st = env.sim.state[lid].float()
    live = po.p_sid[eid] >= 0
    ca, sa = torch.cos(st[:, 2:3] + lidar.angles[None].float()), torch.sin(st[:, 2:3] + lidar.angles[None].float())
    hx = st[:, 0:1] + MOUNT_X * torch.cos(st[:, 2:3]) + r_w * ca; hy = st[:, 1:2] + MOUNT_X * torch.sin(st[:, 2:3]) + r_w * sa
    on_box = r_w < r_n - 0.03
    bx = po.p_poses[eid, :, :2].float()
    near = ((hx[:, None, :] - bx[:, :, 0:1]) ** 2 + (hy[:, None, :] - bx[:, :, 1:2]) ** 2) < 0.8 ** 2      # (B_l, P, N)
    seen = ((near & on_box[:, None, :]).sum(2) * live).to(torch.int16)                                     # beams on each box
    return (cf / obs.range).clamp(0.0, 1.0), seen, rep.sum(1), (real - r_w).abs().median()


def near_box(xy, bxy, live):
    """(B_l, P): nearest distance from each box centre to the plan's dense points"""
    d = (xy[:, None, :, :] - bxy[:, :, None, :]).norm(dim=3).min(2).values
    return torch.where(live, d, torch.full_like(d, 1e3))


cf_now, seen, nb, err0 = no_box_scan()
print(f"정지 스캔과 실제 스캔의 차이 중앙 {float(err0) * 100:.1f} cm (스캔 시점 확인)", flush=True)
scans_cf = cf_now[:, None, :].expand(-1, obs.scans.shape[1], -1).clone()
h_cf = torch.zeros(Bl, int(getattr(actor, "memory", 0)), device=dev) if getattr(actor, "memory", 0) else None
T = int(secs / PS.DT)
R = {k: [] for k in ("st", "sid", "pose", "seen", "nb", "d1", "d5", "d10", "dv5", "cfhit", "rlhit", "rs", "cnt", "tid", "pl", "plc")}
for kd in range(T):
    st = env.sim.state[lid].clone()
    live = po.p_sid[eid] >= 0; bxy = po.p_poses[eid, :, :2].float()
    R["st"].append(st[:, :4].float().cpu()); R["sid"].append(po.p_sid[eid].to(torch.int16).cpu()); R["pose"].append(po.p_poses[eid].float().cpu())
    R["seen"].append(seen.cpu()); R["nb"].append(nb.to(torch.int16).cpu())
    state_in = obs.state()
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    m_cf, _, v_cf = actor(scans_cf, state_in, h_cf)
    if h_cf is not None:
        h_cf = actor.h_out.detach()
    p_cf = PS.roll(torch.tanh(m_cf), v_cf.detach().clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
    x, y, _ = plan_world(p, st); xc, yc, _ = plan_world(p_cf, st)
    xy, xyc = torch.stack([x, y], -1), torch.stack([xc, yc], -1)
    dd = (xy[:, ii] - xyc[:, ii]).norm(dim=2)
    R["d1"].append(dd[:, 0].cpu()); R["d5"].append(dd[:, 4].cpu()); R["d10"].append(dd[:, 9].cpu())
    R["pl"].append(xy[:, ii].half().cpu()); R["plc"].append(xyc[:, ii].half().cpu())
    R["dv5"].append((p["pv"][:, 4] - p_cf["pv"][:, 4]).float().cpu())
    R["cfhit"].append((near_box(xyc, bxy, live) < 0.40).cpu()); R["rlhit"].append((near_box(xy, bxy, live) < 0.40).cpu())
    out, _, _ = drv.decision(torch.tanh(mean), p)
    cf_now, seen, nb, _ = no_box_scan()
    scans_cf = torch.cat([cf_now[:, None, :], scans_cf[:, :-1]], 1)
    rs = out["restarted"]
    if bool(rs.any()):
        scans_cf[rs] = cf_now[rs][:, None, :].expand(-1, scans_cf.shape[1], -1)
        if h_cf is not None:
            h_cf[rs] = 0.0
    if remover is not None:
        remover.step()
    if restarter is not None:
        restarter.step()
    R["rs"].append(rs.cpu()); R["cnt"].append(env.sim.contact_frame()[lid].float()[:, I_CNT].cpu())
    R["tid"].append(env.sim.tid[lid].to(torch.int16).cpu())
    if kd % 100 == 99:
        print(f"  {kd + 1}/{T} 결정", flush=True)
D = {k: torch.stack(v).numpy() for k, v in R.items()}
trk = env.sim.track
D.update(edt=trk.edt.float().cpu().numpy(), t_off=trk.t_off.cpu().numpy(), t_H=trk.t_H.cpu().numpy(), t_W=trk.t_W.cpu().numpy(),
         t_res=trk.t_res.cpu().numpy(), t_origin=trk.t_origin.cpu().numpy(), range_max=np.array(float(obs.range)), fov=np.array(fov))
for k in sorted({int(v) for v in np.unique(D["sid"]) if v >= 0}):
    sh = po.shapes[k]
    D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez_compressed(out_npz, dt=np.array(PS.DT), name=np.array(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"), **D)
print(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}: 학습 차 {Bl}대 x {secs:.0f} s 저장 {out_npz}", flush=True)
