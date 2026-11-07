"""planner만 따로 재기 (11-07 16:40, 사용자: "planner만 독립적으로 평가하는 메트릭을 만들어놓을까? closed loop으로 평가하는거지,
계속 openloop으로 제어기까지 얹고 하려니까 문제생기는 것 같은데"). 학습 인자로 env를 만들어 차 N대를 평균 행동으로 달리며(closed
loop: plan이 차를 움직이고 그 결과가 다음 plan의 입력), 결정(0.05 s)마다 policy의 plan(세계 좌표, 시각 포함)과 차 자세를 기록.

두 방식:
  map   -- plan을 MAP(후진은 추종기)이 따라감: 실차와 같은 조건
  ideal -- 제어기 없이 차를 plan 위로 그대로 옮김(매 제어 step 뒤 자세·속도·각속도를 그 step에 쓴 plan의 dt 시점 값으로
           덮어씀): plan을 글자 그대로 몰면 무슨 일이 나는가. plan을 핸들 손잡이로 쓰는 policy는 여기서 드러남

잣대(planner만):
  일관성  -- 결정 k의 plan이 시각 t_k + 0.05 + h에 있겠다던 곳과 결정 k+1의 plan이 같은 시각(자기 h)에 있겠다는 곳의 거리;
             0.3 s 점이 바로 전 plan과 반대쪽(차 기준 옆, 둘 다 10 cm 넘게)으로 넘어간 횟수 / s
  정직함  -- plan이 h 뒤 있겠다던 곳과 그 시각 실제 차 위치의 거리(옆 방향 따로)
  실현성  -- plan의 횡가속도(|곡률| x 속도^2)가 MAP 조향 표의 정상 상태 최대값을 넘는 plan 비율
  안전    -- plan의 중심선이 벽이나 상자에 0.2 m 안으로 들어가는 plan 비율(0.3 s 안 / 전체)
  결과    -- 세게 부딪힘(벽, 상자) / 분, 평균 속도, 멈춘 시간
인자: 체크포인트 차수 초 [map|ideal] [상자 0|1]"""
import os, sys, math, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim import plan_tw
from f1sim import dynamics as dyn
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

torch.set_grad_enabled(False)
path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "map"
props = len(sys.argv) > 5 and sys.argv[5] == "1"
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, 81, "cuda"
a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop, a.stage = 0.0, 0, 0.0, 0.0, "props"
a.prop_fraction = 1.0 if props else 0.0
a.prop_count = "uniform"
a.episode_s = 3600.0
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=81)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel()); R = int(a.repeat); dt = env.sim.control_dt
step_s = R * dt
tr = env.sim.track; tid = env.sim.tid[lid]
P = env.sim.P
lut = env.tracker.map.lut
amax_t = torch.nan_to_num(lut.table, nan=-1.0).amax(1).clamp_min(0.5).float()      # (G, V)
from f1sim.map_controller import grip_index
g_row = grip_index(env.sim.P, env.tracker.nominal)[lid]
i_hd = [CC.index(k) for k in ("hard_wall", "hard_prop")]
RAD = 0.2                                                    # [m] a plan point closer than this to a wall / prop "hits"


def world(dn, pose):
    """(B, n, 5) body-frame plan -> (B, n, 2) world points"""
    c, s_ = torch.cos(pose[:, 2])[:, None], torch.sin(pose[:, 2])[:, None]
    return torch.stack([pose[:, 0:1] + dn[..., 0] * c - dn[..., 1] * s_, pose[:, 1:2] + dn[..., 0] * s_ + dn[..., 1] * c], -1)


def at_time(dn, t):
    """(B, 5) the dense plan at time t (linear between its samples, held past the end)"""
    tt = dn[0, :, 4]
    j = torch.searchsorted(tt.contiguous(), torch.tensor([float(t)], device=tt.device)).clamp(1, tt.numel() - 1)[0]
    w = ((t - tt[j - 1]) / (tt[j] - tt[j - 1]).clamp_min(1e-9)).clamp(0.0, 1.0)
    return dn[:, j - 1] * (1 - w) + dn[:, j] * w


plans, poses, hits = [], [], []
last = torch.zeros(B, tp.A, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
cf0 = env.sim.contact_frame()[lid].float()[:, i_hd].clone()
speed_sum, stop_t = torch.zeros(B, device=dev), torch.zeros(B, device=dev)
for k in range(int(secs / dt)):
    if k % R == 0:
        last = tp(last); act_full[lid] = last
    pose0 = env.sim.state[lid, :3].clone()                  # the tracker anchors this step's plan here
    _, _, te, trc, _ = env.step(act_full)
    dn = env.tracker.tw_plan[lid].float()                  # (B, n, 5) x, y, psi, v, t in the frame of pose0
    if k % R == 0:
        plans.append(world(dn, pose0)); poses.append(pose0)
    if mode == "ideal":
        # the car put exactly where this step's plan said it would be dt later, at its speed and turn rate
        p1 = at_time(dn, dt)
        p2 = at_time(dn, 2 * dt)
        xy = world(p1[:, None, :], pose0)[:, 0]
        yaw = pose0[:, 2] + p1[:, 2]
        v = p1[:, 3]
        r = torch.atan2(torch.sin(p2[:, 2] - p1[:, 2]), torch.cos(p2[:, 2] - p1[:, 2])) / dt
        st = env.sim.state.clone()
        keep = ~(te | trc)[lid]                              # a row that respawned this step keeps its new start
        rows = lid[keep]
        st[rows, dyn.IX], st[rows, dyn.IY], st[rows, dyn.IYAW] = xy[keep, 0], xy[keep, 1], yaw[keep]
        st[rows, dyn.IVX], st[rows, dyn.IVY], st[rows, dyn.IR] = v[keep], 0.0, r[keep]
        kap = r[keep] / v[keep].abs().clamp_min(0.3) * torch.sign(v[keep])
        st[rows, dyn.ISTEER] = torch.atan(kap * float(P["lf"][0] + P["lr"][0])).clamp(-float(env.s_max), float(env.s_max)) \
            if torch.is_tensor(P["lf"]) else st[rows, dyn.ISTEER]
        st[rows, dyn.IOMEGA] = v[keep] / P["r_w"][rows] if torch.is_tensor(P["r_w"]) else st[rows, dyn.IOMEGA]
        env.sim.state.copy_(st)
    sp = env.sim.state[lid, 3].abs()
    speed_sum += sp * dt
    stop_t += (sp < 0.3).float() * dt
cf1 = env.sim.contact_frame()[lid].float()[:, i_hd]
hard = (cf1 - cf0).sum(0)
W = torch.stack(plans, 0); X = torch.stack(poses, 0)        # (T, B, n, 2), (T, B, 3)
T_plan = float(env.tracker.tw_plan[0, -1, 4])
tt = env.tracker.tw_plan[0, :, 4].float()


def w_at(Wk, t):
    """(..., 2) world plan points at time t"""
    j = int(torch.searchsorted(tt.contiguous(), torch.tensor([float(t)], device=tt.device)).clamp(1, tt.numel() - 1)[0])
    w = float(((t - tt[j - 1]) / (tt[j] - tt[j - 1]).clamp_min(1e-9)).clamp(0.0, 1.0))
    return Wk[..., j - 1, :] * (1 - w) + Wk[..., j, :] * w


name = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
print(f"{name} [{a.plan_format}] {'MAP으로 따라감' if mode == 'map' else 'plan 그대로(제어기 없음)'}, "
      f"{'상자 배치' if props else '빈 트랙'}, 차 {B}대 x {secs:.0f} s, plan 길이 {T_plan:.1f} s", flush=True)
H = [h for h in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9) if h + step_s <= T_plan + 1e-6]
cons = []
for h in H:
    d = (w_at(W[:-1], h + step_s) - w_at(W[1:], h)).norm(dim=-1)
    cons.append(f"{h:.1f} s {float(d.median()) * 100:.0f}/{float(d.quantile(0.9)) * 100:.0f}")
# flips of the 0.3 s point, sideways in the frame of the later decision
a_ = w_at(W[:-1], 0.3 + step_s) - X[1:, :, :2]; b_ = w_at(W[1:], 0.3) - X[1:, :, :2]
yy = X[1:, :, 2]
sa = -a_[..., 0] * torch.sin(yy) + a_[..., 1] * torch.cos(yy)
sb = -b_[..., 0] * torch.sin(yy) + b_[..., 1] * torch.cos(yy)
flip = ((sa * sb < 0) & (sa.abs() > 0.1) & (sb.abs() > 0.1)).float().mean() / step_s
print(f"  일관성: 같은 미래 시각을 다음 plan이 다르게 가리킨 거리 중앙/p90 cm -- {', '.join(cons)} | "
      f"0.3 s 점 반대쪽으로 넘어감 1 s에 {float(flip):.2f}번", flush=True)
hon = []
for h in (0.1, 0.2, 0.3, 0.5):
    if h > T_plan + 1e-6:
        continue
    kk = int(round(h / step_s))
    d = w_at(W[:-kk], h) - X[kk:, :, :2]
    yy = X[kk:, :, 2]
    cr = (-d[..., 0] * torch.sin(yy) + d[..., 1] * torch.cos(yy)).abs()
    hon.append(f"{h:.1f} s {float(d.norm(dim=-1).median()) * 100:.0f} (옆 {float(cr.median()) * 100:.0f})")
print(f"  정직함: plan이 h 뒤 있겠다던 곳과 실제 위치 거리 중앙 cm -- {', '.join(hon)}", flush=True)
# feasibility over the whole plan: |kappa| v^2 against the table's steady-state limit at that speed
seg = W[:, :, 1:, :] - W[:, :, :-1, :]
hd = torch.atan2(seg[..., 1], seg[..., 0])
dh = torch.atan2(torch.sin(hd[..., 1:] - hd[..., :-1]), torch.cos(hd[..., 1:] - hd[..., :-1]))
ds = seg.norm(dim=-1)[..., 1:].clamp_min(1e-4)
kap = (dh / ds).abs()
v_ = ds / float(tt[1] - tt[0])
vj = torch.searchsorted(lut.vels.float().contiguous(), v_.contiguous()).clamp(0, lut.vels.numel() - 1)
amax = amax_t[g_row[None, :, None].expand_as(vj), vj]
ratio = torch.where((ds > 0.005) & (v_ > 1.0), kap * v_ ** 2 / amax, torch.zeros_like(kap))
over = (ratio.amax(-1) > 1.0).float().mean()
print(f"  실현성: plan 어딘가의 횡가속도가 타이어 한계(조향 표 정상 상태 최대)를 넘는 plan {float(over) * 100:.0f} %, "
      f"plan마다 최대 비율의 중앙 {float(ratio.amax(-1).median()):.2f}", flush=True)
# safety: plan points within RAD of a wall (the track's distance field) or of a prop
Ws = W[:, :, ::3]                                            # every 0.025 s of the plan
tts = tt[::3]
pts = Ws.reshape(-1, 2)
tid_p = tid[None, :, None].expand(Ws.shape[0], B, Ws.shape[2]).reshape(-1)
edt = tr.sample_edt(pts, tid_p).reshape(Ws.shape[:3])
clr = edt
if props and getattr(env, "procedural", None) is not None:
    eid = env.sim.eid[lid][None, :, None].expand(Ws.shape[0], B, Ws.shape[2]).reshape(-1)
    pc = torch.cat([env.procedural.clearance(pts[i:i + 200000], eid[i:i + 200000]) for i in range(0, pts.shape[0], 200000)]).reshape(Ws.shape[:3])
    clr = torch.minimum(edt, pc)
n03 = int((tts <= 0.3 + 1e-6).sum())
bad_all = (clr < RAD).any(-1).float().mean()
bad_03 = (clr[..., :n03] < RAD).any(-1).float().mean()
print(f"  안전: plan이 벽·상자 {RAD:.1f} m 안으로 들어가는 plan -- 0.3 s 안 {float(bad_03) * 100:.0f} %, 전체 {float(bad_all) * 100:.0f} %", flush=True)
mins = secs * B / 60.0
print(f"  결과: 세게 부딪힘 벽 {float(hard[0]) / mins:.1f} 상자 {float(hard[1]) / mins:.1f} /분(차 1대) | 평균 속도 "
      f"{float(speed_sum.mean()) / secs:.2f} m/s | 멈춘 시간 {float(stop_t.mean()) / secs * 100:.0f} %", flush=True)
