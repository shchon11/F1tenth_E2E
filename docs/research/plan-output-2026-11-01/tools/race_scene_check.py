"""(11-24, 경주 단계) 복사본이 모는 상대 차까지 장면 재생이 그대로 되살리나 -- plan_rl --scene-copies 의 확인.

scene_selftest.py와 같은 순서(학습의 재시작 경로: _reset_envs_core -> SceneReplay.load -> _fill_scans -> after_fill, 그리고 plan_rl이
하는 대로 우리 policy 입력 pol_* 과 복사본의 것 so_* 를 되돌림)를 경주 세계(상대 차 = scan31b 복사본 두 개 -- update 번호가 다른 둘,
배율 0.6-0.95 / 제 페이스)에서:
  1. W 결정 달린 뒤 모든 경주를 장면으로 저장
  2. 그대로 N 결정 (갈래 A)
  3. 장면으로 되돌려 N 결정 (B1), 4. 한 번 더 되돌려 N 결정 (B2)
센서 잡음 0, 두 차 모두 평균 행동, 난수 생성기는 갈래마다 장면 때 상태로 맞춤. 우리 차와 상대 차의 자리 차이(최대)를 냄:
B1 vs B2 (같은 장면 두 번 재생), A vs B1 (원래 이어 달린 것과 재생).
인자: checkpoint [경주 수 16] [W 30] [N 40] [장치 cuda]
"""
import os, sys, json
os.environ["F1SIM_NO_STAGGER"] = "1"
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
torch.set_grad_enabled(False)

ck_path = sys.argv[1]
G = int(sys.argv[2]) if len(sys.argv) > 2 else 16
W = int(sys.argv[3]) if len(sys.argv) > 3 else 30
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
dev = torch.device(sys.argv[5] if len(sys.argv) > 5 else "cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of, load_actor2, PlanObs2, plan_v2, plan_world, PT_IDX
os.environ["F1SIM_LIDAR_MOUNT_X"] = str(lidar_mount_of(ck))
os.environ["F1SIM_PLAN_LAT_MARGIN"] = str((ck.get("args") or {}).get("plan_lat_margin", 0.95))
from f1sim.learn import plan_rl as PR
from f1sim.learn.scene_replay import SceneReplay
from f1sim import plan_steps as PS

WORLD = os.path.join(os.path.dirname(PR.WORLD), "plan_world_korea26scan_race.json")
env, lim, a = PR.build(WORLD, 2 * G, 7, dev, overrides={"opp_present": 1.0, "align_episodes": True})
M = int(env.M); lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
obs = PlanObs2(env, lim, None)
so = PR.SelfOpponents(env, lim, 1.0, seed=9, max_pool=2, p_newest=0.5, pace=(0.6, 0.95, 0.2))
so.add(actor, 500); so.add(load_actor2(ck, dev), 525)            # two copies (same weights, two tags): identity must come back
so.draw(torch.arange(so.on.numel(), device=dev))
others = env.sim.other_idx[lid][:, 0]
opp_index = torch.full((env.B,), -1, dtype=torch.long, device=dev); opp_index[so.rows] = torch.arange(so.rows.numel(), device=dev)
oi = opp_index[others]
scene = SceneReplay(env, 1.0, G, 10 ** 6)
env.scene_replay = scene
act0 = torch.zeros(env.B, env.act_dim, device=dev)
ii = PT_IDX.to(dev)


def quiet():
    P = env.sim.params.P
    for n in P:
        if n in ("dropout_value", "road_tilt_v"):
            continue
        if (any(s in n for s in ("noise", "jitter", "dropout", "spike", "bias_walk", "shock_rate"))
                or n.startswith(("vib_accel", "vib_gyro")) or n == "road_tilt"):
            env.sim.params.set_all(n, 0.0)
    for n in ("car_porosity", "car_rear_porosity"):
        t = getattr(env.sim, n, None)
        if torch.is_tensor(t):
            t.zero_()


def decide():
    st = env.sim.state[lid].clone()
    _, _, _, _, p = plan_v2(actor, obs, lim)
    x, y, _ = plan_world(p, st)
    pts = torch.stack([x[:, ii], y[:, ii]], -1)
    so.plan()
    env.sim.ideal_vcap[lid] = 1e3
    rs = torch.zeros(Bl, dtype=torch.bool, device=dev)
    for j in range(PS.SUB):
        scene.to_boundary = PS.SUB - 1 - j
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        so.command(j)
        _, _, te, trc, _ = env.step(act0)
        so.stepped(te | trc); rs |= (te | trc)[lid]
    so.after()
    obs.push_scan(); obs.set_plan(pts, p["pv"]); obs.set_plan2(p)
    if bool(rs.any()):
        obs.reset_rows(torch.nonzero(rs)[:, 0])
    return rs


def gens_of(root):
    out, seen = [], set()

    def rec(o, d=0):
        if id(o) in seen or d > 6:
            return
        seen.add(id(o))
        if isinstance(o, torch.Generator):
            out.append(o); return
        if isinstance(o, (list, tuple)):
            for v in o: rec(v, d + 1)
        elif isinstance(o, dict):
            for v in list(o.values()): rec(v, d + 1)
        elif type(o).__module__.startswith("f1sim") and hasattr(o, "__dict__"):
            for v in list(vars(o).values()): rec(v, d + 1)
    rec(root)
    return out


def drive_branch():
    me, op, rs_ = [], [], torch.zeros(Bl, dtype=torch.bool, device=dev)
    for _ in range(N):
        rs_ |= decide(); quiet()
        me.append(env.sim.state[lid, :2].clone()); op.append(env.sim.state[others, :2].clone())
    return torch.stack(me), torch.stack(op), rs_


def restore():
    rows = torch.arange(env.B, device=dev); g = torch.arange(G, device=dev)
    env._reset_envs_core(rows, fill=False)
    scene.load(g, g)
    env._fill_scans(rows, env.sim.odom.state[rows, 3])
    scene.after_fill(env)
    g_l, i_l = scene.take_loaded()
    Dd = scene.data
    obs.scans[g_l] = Dd["pol_scans"][i_l]
    obs.rel[g_l] = Dd["pol_rel"][i_l]; obs.rel_v[g_l] = Dd["pol_rel_v"][i_l]
    obs.cmd_d[g_l] = Dd["pol_cmd_d"][i_l]; obs.cmd_v[g_l] = Dd["pol_cmd_v"][i_l]
    obs.plan_xy[g_l] = Dd["pol_plan_xy"][i_l]; obs.plan_v[g_l] = Dd["pol_plan_v"][i_l]
    env.last_result.imu[lid[g_l]] = Dd["pol_imu"][i_l][:, None, :].to(env.last_result.imu.dtype)
    obs.mem_h[g_l] = Dd["pol_mem"][i_l]
    # the copies as plan_rl puts them back (--scene-copies); first scramble them, so nothing of the run before survives
    so.on[:] = False; so.pace[:] = 0.77; so.row_tag[:] = 525; so.obs.reset_rows(torch.arange(so.on.numel(), device=dev))
    so.restore(oi[g_l], {k: Dd["so_" + k][i_l] for k in PR.SelfOpponents.PACE_KEYS})
    env.ep_limit[:] = 10 ** 6
    quiet()
    for x, s_ in zip(gens, g_states):
        x.set_state(s_)
    torch.set_rng_state(cpu_state)
    if dev.type == "cuda":
        torch.cuda.set_rng_state(cuda_state)


env.ep_limit[:] = 10 ** 6
quiet()
for _ in range(W):
    decide(); quiet()
snap = scene.capture()
snap.update(pol_scans=obs.scans.clone(), pol_rel=obs.rel.clone(), pol_rel_v=obs.rel_v.clone(), pol_cmd_d=obs.cmd_d.clone(),
            pol_cmd_v=obs.cmd_v.clone(), pol_plan_xy=obs.plan_xy.clone(), pol_plan_v=obs.plan_v.clone(),
            pol_imu=env.last_result.imu[lid].float().mean(1), pol_mem=obs.mem_h.clone())
for k, v in so.snapshot(oi).items():
    snap["so_" + k] = v
snap["hit_kind"] = torch.zeros(G, dtype=torch.long, device=dev)
scene.add(snap)
tags0, pace0, on0 = so.row_tag[oi].clone(), so.pace[oi].clone(), so.on[oi].clone()
gens = gens_of(env)
g_states = [x.get_state() for x in gens]
cpu_state = torch.get_rng_state(); cuda_state = torch.cuda.get_rng_state() if dev.type == "cuda" else None
meA, opA, rsA = drive_branch()
restore()
ok_id = bool(torch.equal(so.row_tag[oi], tags0) and torch.equal(so.pace[oi], pace0) and torch.equal(so.on[oi], on0))
meB1, opB1, rsB1 = drive_branch()
restore()
meB2, opB2, rsB2 = drive_branch()
ok = ~(rsA | rsB1 | rsB2)
d = lambda a_, b_: float((a_ - b_).norm(dim=-1)[:, ok].max()) if bool(ok.any()) else float("nan")
res = {"races": int(ok.sum()), "copy_identity_restored": ok_id,
       "B1_vs_B2_ego_m": d(meB1, meB2), "B1_vs_B2_opp_m": d(opB1, opB2),
       "A_vs_B1_ego_m": d(meA, meB1), "A_vs_B1_opp_m": d(opA, opB1),
       "tags": sorted(set(tags0.tolist())), "paces": [round(float(x), 3) for x in pace0.tolist()]}
print(f"[장면 재생 확인 · 복사본 상대 차] 경주 {int(ok.sum())}개 (재시작 없는 것), {W} 결정 뒤 장면, {N} 결정씩 ({N * PS.DT:.0f} s)")
print(f"  복사본 정체(update 번호)·배율·켜짐 되살림: {'같음' if ok_id else '다름'} (update {res['tags']}, 배율 {res['paces']})")
print(f"  같은 장면 두 번 재생(B1 vs B2) 최대 자리 차이: 우리 차 {res['B1_vs_B2_ego_m']:.4f} m · 상대 차 {res['B1_vs_B2_opp_m']:.4f} m")
print(f"  원래 이어 달린 것과 재생(A vs B1): 우리 차 {res['A_vs_B1_ego_m']:.4f} m · 상대 차 {res['A_vs_B1_opp_m']:.4f} m")
print(json.dumps(res))
