"""빈 트랙에서 학습된 planner가 달린 라인이 보상 때문에 어느 쪽으로 끌렸나 (11-08 05:00, 사용자: "빈 트랙에서 달리는 저 라인이 보상설계에
의해 편향될가능성이 있나?"). 원래 트랙(변형 없음), 빈 트랙, 상대 없음으로 차 N대를 몰아(조건: ideal = 학습 조건 / map = 실차 조건) 지나간
자리·속도를 기록하고, 두 선과 겹쳐 그림: (1) 결승선까지 벽 사이 최단 경로(중간 점수의 기준) -- 여기 붙으면 중간 점수 쪽으로 끌린 것,
(2) 랩타임 최소 레이싱라인(상대 차가 쓰는 min_time 선, 비교용) -- 여기 붙으면 랩타임 쪽. 각 선까지의 거리 중앙도 셈.
인자: checkpoint 조건(ideal|map) 차수 초 출력png"""
import os, sys, math, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import GoalField
from f1sim.learn.plan_policy import PlanObs, load_actor, plan_world, plan_segments, PT_IDX, PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

torch.set_grad_enabled(False)
ck_path, mode, n_env, secs, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD))); a.envs, a.seed, a.device = n_env, 7, "cuda"
a.warps, a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0, 0.0, 0.0, 0.0, secs + 5.0
if mode == "map":
    a.plan_format, a.repeat = "tw", 4
env = SR.build_env(a, dev)
if mode == "ideal":
    env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=7)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
V2 = ck.get("format") == "plan_steps2"
actor = (load_actor2 if V2 else load_actor)(ck, dev); obs = (PlanObs2 if V2 else PlanObs)(env, lim, None)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
XY, SP = [], []
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    if V2:
        _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    else:
        mean, _ = actor(obs.scans, obs.state())
        p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
    x, y, psi = plan_world(p, st)
    if mode == "map":
        k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(act0)
        XY.append(env.sim.state[lid, :2].cpu().numpy().copy()); SP.append(env.sim.state[lid, dyn.IVX].cpu().numpy().copy())
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
    if V2:
        obs.set_plan2(p)
XY = np.stack(XY, 1); SP = np.stack(SP, 1)                    # (B, T, 2), (B, T)
skip = int(3.0 / env.sim.control_dt)                          # the start (standing) aside
XYd, SPd = XY[:, skip:], SP[:, skip:]
# the lines to compare with: the shortest path to the finish (the shaping's measure) and the min-time raceline
tr = env.sim.track
goal = GoalField(tr, dev)
rl = env._rls[0] if getattr(env, "_rls", None) else None
rl_xy = np.asarray(rl.xy, float) if rl is not None else None
rl_v = np.asarray(rl.v, float) if rl is not None else None
# shortest path: from just past the finish line, follow the distance field down (8-neighbour descent on the grid)
res = float(tr.t_res[0]); ox, oy = [float(v) for v in tr.t_origin[0]]; H, W = int(tr.t_H[0]), int(tr.t_W[0]); off = int(tr.t_off[0])
D = goal.field[off: off + H * W].view(H, W).cpu().numpy()
xy0, yaw0 = tr.pose_at_s(torch.zeros(1, device=tr.t_res.device), torch.zeros(1, dtype=torch.long, device=tr.t_res.device))
p0 = xy0[0].cpu().numpy() + 0.3 * np.array([math.cos(float(yaw0[0])), math.sin(float(yaw0[0]))])
i, j = int(round((p0[1] - oy) / res)), int(round((p0[0] - ox) / res))
sp_path = []
for _ in range(20000):
    sp_path.append((ox + j * res, oy + i * res))
    best, bi, bj = D[i, j], i, j
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i + di, j + dj
            if 0 <= ii < H and 0 <= jj < W and np.isfinite(D[ii, jj]) and D[ii, jj] < best:
                best, bi, bj = D[ii, jj], ii, jj
    if (bi, bj) == (i, j) or best <= 0.05:
        break
    i, j = bi, bj
sp_path = np.array(sp_path)
def dist_to(line, pts):
    d = np.sqrt(((pts[:, None, :] - line[None, ::2, :]) ** 2).sum(-1)).min(1)
    return d
pts = XYd.reshape(-1, 2)[::5]
d_sp = dist_to(sp_path, pts); d_rl = dist_to(rl_xy, pts) if rl_xy is not None else np.full(len(pts), np.nan)
occ = tr.edt[off: off + H * W].view(H, W).cpu().numpy() <= 0
ext = [ox - res / 2, ox + (W - 0.5) * res, oy - res / 2, oy + (H - 0.5) * res]
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.5)
for b in range(B):
    sc = ax.scatter(XYd[b, ::4, 0], XYd[b, ::4, 1], c=SPd[b, ::4], s=2, cmap="turbo", vmin=0, vmax=8)
ax.plot(sp_path[:, 0], sp_path[:, 1], "k--", lw=1.6, label="결승선까지 최단 경로 (중간 점수 기준)")
if rl_xy is not None:
    ax.plot(rl_xy[:, 0], rl_xy[:, 1], "m-", lw=1.4, label=f"랩타임 최소 레이싱라인 (비교용, 최고 {rl_v.max():.1f} m/s)")
plt.colorbar(sc, ax=ax, fraction=0.03, label="policy 속도 [m/s]")
ax.legend(loc="upper left", fontsize=10); ax.set_aspect("equal")
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
ax.set_title(f"{name} [{'학습 조건' if mode == 'ideal' else '실차 조건'}] 빈 트랙 차 {B}대 x {secs:.0f} s: 달린 자리(색 = 속도) | 최단 경로까지 거리 중앙 "
             f"{np.median(d_sp) * 100:.0f} cm, 레이싱라인까지 {np.nanmedian(d_rl) * 100:.0f} cm", fontsize=11)
plt.tight_layout(); plt.savefig(out, dpi=70)
print(f"{out} | 최단 경로까지 중앙 {np.median(d_sp) * 100:.0f} cm, 레이싱라인까지 중앙 {np.nanmedian(d_rl) * 100:.0f} cm | 평균 속도 {SPd.mean():.2f} m/s")
