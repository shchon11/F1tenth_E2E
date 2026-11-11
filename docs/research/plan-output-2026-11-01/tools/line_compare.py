"""빈 트랙에서 학습된 planner가 달린 라인이 보상 때문에 어느 쪽으로 끌렸나 (11-08 05:00, 사용자: "빈 트랙에서 달리는 저 라인이 보상설계에
의해 편향될가능성이 있나?"). 원래 트랙(변형 없음), 빈 트랙, 상대 없음으로 차 N대를 몰아(조건: ideal = 학습 조건 / map = 실차 조건) 지나간
자리·속도를 기록해 지도 위에 그림(색 = 속도). 겹쳐 그리는 선은 없음(11-12: 보상의 거리장을 따라 내려가 그리던 최단 경로가 몇 m에서 끊겨 있었음 -> 뺌).
(11-09 사용자: 레이싱라인은 쓰지 않으니 그림·계산에서 뺌)
인자: checkpoint 조건(ideal|map) 차수 초 출력png"""
import os, sys, argparse, json
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
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
follower = None
if mode == "track":                                          # 11-09: physics + plan_follow.PlanFollower
    from f1sim.plan_follow import PlanFollower, to_car
    follower = PlanFollower(B, dev, s_max=float(env.s_max), lut=env.tracker.map.lut,
                            grip=M.GRIP_LEVELS.index(1.0), wheelbase=float(env.tracker.nominal["lf"]) + float(env.tracker.nominal["lr"]))
    env.ext_mask[lid] = True; env._ext_ids.update(int(i) for i in lid.tolist())
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
    if follower is not None:
        follower.set_plan(slice(None), x, y, psi, p["v"], p["delta"])
    if mode == "map":
        k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        if follower is not None:
            env.ext_cmd[lid] = to_car(env, lid, follower.command(env.sim.state[lid]))
        env.step(act0)
        XY.append(env.sim.state[lid, :2].cpu().numpy().copy()); SP.append(env.sim.state[lid, dyn.IVX].cpu().numpy().copy())
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
    if V2:
        obs.set_plan2(p)
XY = np.stack(XY, 1); SP = np.stack(SP, 1)                    # (B, T, 2), (B, T)
skip = int(3.0 / env.sim.control_dt)                          # the start (standing) aside
XYd, SPd = XY[:, skip:], SP[:, skip:]
# no line is overlaid: the descent along the reward's distance field stopped a few metres after the start (cause not looked into),
# so the dashed "shortest path" was a stub and the "distance to it" read 6.3-6.9 m in every plan25r / plan25x picture
tr = env.sim.track
res = float(tr.t_res[0]); ox, oy = [float(v) for v in tr.t_origin[0]]; H, W = int(tr.t_H[0]), int(tr.t_W[0]); off = int(tr.t_off[0])
occ = tr.edt[off: off + H * W].view(H, W).cpu().numpy() <= 0
ext = [ox - res / 2, ox + (W - 0.5) * res, oy - res / 2, oy + (H - 0.5) * res]
fig, ax = plt.subplots(1, 1, figsize=(11.5, 9))
ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.5)
for b in range(B):
    sc = ax.scatter(XYd[b, ::4, 0], XYd[b, ::4, 1], c=SPd[b, ::4], s=2, cmap="turbo", vmin=0, vmax=8)
plt.colorbar(sc, ax=ax, fraction=0.03, label="policy 속도 [m/s]")
ax.set_aspect("equal")
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
ax.set_title(f"{name} [{dict(ideal='학습 조건', map='실차 조건 MAP', track='물리 + 새 제어기')[mode]}] 빈 트랙 차 {B}대 x {secs:.0f} s: 달린 자리(색 = 속도) | 평균 속도 "
             f"{SPd.mean():.2f} m/s", fontsize=11)
plt.tight_layout(); plt.savefig(out, dpi=70, bbox_inches="tight")
print(f"{out} | 평균 속도 {SPd.mean():.2f} m/s")
