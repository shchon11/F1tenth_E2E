"""(11-16) 장면 재생(learn.scene_replay)이 실제 학습 설정에서 장면을 그대로 되살리나 -- 자기 점검.

사용자(11-16): "똑바로 안해놓고 이 방향성은 안돼" -- 장면 재생의 시험(tests/test_scene_replay.py)은 CPU · 직접 조향 · 상대 차 없음 · 생성
트랙이었음. 여기서는 학습과 같은 세계(checkpoint의 세계 json, 변형 트랙 포함, 상대 차, GPU 그래프, plan대로 정확히 실행)에서
  1. W 결정 달린 뒤 모든 경주를 장면으로 저장(plan_rl.py와 같은 항목: 환경 쪽 SceneReplay.capture + policy 입력 pol_*)
  2. 그대로 N 결정 달리며 차 상태 기록 (갈래 A)
  3. 모든 경주를 다시 출발시키고(학습의 재시작 경로 그대로: _reset_envs_core -> SceneReplay.load -> _fill_scans), 자기 장면을 덮어씀,
     policy 입력도 plan_rl.py처럼 되돌림
  4. 같은 N 결정 달리며 기록 (갈래 B)
센서 잡음은 모두 0(box_replay.py의 quiet), policy는 평균 행동 -> 장면이 온전하면 B는 A와 같은 길을 감. 경주별 최대 자리 차이를
상대 차 없는 경주 / 있는 경주로 나눠 냄(상대 차의 planner 속 상태는 장면에 없다고 적혀 있음 -> 그쪽은 '비슷'만 기대).
인자: checkpoint [경주 수 64] [W 30] [N 40]
"""
import os, sys, json, time
os.environ["F1SIM_NO_STAGGER"] = "1"
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ck_path = sys.argv[1]
G = int(sys.argv[2]) if len(sys.argv) > 2 else 64
W = int(sys.argv[3]) if len(sys.argv) > 3 else 30
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of
_mx = lidar_mount_of(ck)
if _mx > 0:
    os.environ["F1SIM_LIDAR_MOUNT_X"] = str(_mx)
_mg = (ck.get("args") or {}).get("plan_lat_margin")
if _mg is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(_mg)
from world_of import world_path
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, plan_world, PT_IDX
from f1sim.learn.scene_replay import SceneReplay
from f1sim import plan_steps as PS

WORLD = world_path(ck)
M_guess = 2
env, lim, a = PR.build(WORLD, G * M_guess, 7, dev, overrides={"align_episodes": True})
M = int(env.M); assert env.B == G * M, (env.B, G, M)
actor = load_actor2(ck, dev)
lid = env.learner_ids; Bl = int(lid.numel()); assert Bl == G, (Bl, G)
obs = PlanObs2(env, lim, None)
scene = SceneReplay(env, 1.0, G, 10 ** 6)
env.scene_replay = scene
act0 = torch.zeros(env.B, env.act_dim, device=dev)
DIAG = os.environ.get("SCENE_DIAG", "") == "1"


def walk(root):
    """{path: tensor} of every per-car tensor (axis 0 = env.B) reachable through f1sim objects (box_replay.py's walk)"""
    out, seen = {}, set()

    def rec(o, path, depth):
        if id(o) in seen:
            return
        seen.add(id(o))
        if torch.is_tensor(o):
            if o.ndim >= 1 and o.shape[0] == env.B and o.numel() > 0:
                out[path] = o
            return
        if depth > 6 or isinstance(o, (torch.nn.Module, str, bytes, int, float, bool, type(None), np.ndarray)):
            return
        if isinstance(o, dict):
            for k, v in list(o.items()):
                rec(v, path + f"[{k!r}]", depth + 1)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                rec(v, path + f"[{i}]", depth + 1)
        elif type(o).__module__.startswith("f1sim"):
            for k in (list(vars(o).keys()) if hasattr(o, "__dict__") else []):
                if "_captures" in k:
                    continue
                rec(getattr(o, k), path + "." + k, depth + 1)
    rec(root, "env", 0)
    return out
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
    x, y, psi = plan_world(p, st)
    pts = torch.stack([x[:, ii], y[:, ii]], -1)
    env.sim.ideal_vcap[lid] = 1e3
    restarted = torch.zeros(Bl, dtype=torch.bool, device=dev)
    for j in range(PS.SUB):
        scene.to_boundary = PS.SUB - 1 - j
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        _, _, te, trc, _ = env.step(act0)
        restarted |= (te | trc)[lid]
    obs.push_scan(); obs.set_plan(pts, p["pv"]); obs.set_plan2(p)
    if bool(restarted.any()):
        obs.reset_rows(torch.nonzero(restarted)[:, 0])
    return restarted


env.ep_limit[:] = 10 ** 6                                  # no episode ends inside the test
quiet()
with torch.no_grad():
    for _ in range(W):
        decide(); quiet()
    # 1. every race as a scene (plan_rl.py's entries)
    snap = scene.capture()
    snap.update(pol_scans=obs.scans.clone(), pol_rel=obs.rel.clone(), pol_rel_v=obs.rel_v.clone(), pol_cmd_d=obs.cmd_d.clone(),
                pol_cmd_v=obs.cmd_v.clone(), pol_plan_xy=obs.plan_xy.clone(), pol_plan_v=obs.plan_v.clone(),
                pol_imu=env.last_result.imu[lid].float().mean(1))
    if getattr(obs, "mem_h", None) is not None:
        snap["pol_mem"] = obs.mem_h.clone()
    snap["hit_kind"] = torch.zeros(G, dtype=torch.long, device=dev)
    scene.add(snap)
    full0 = {k: v.clone() for k, v in walk(env).items()} if DIAG else None
    # the random streams as they stand: with SCENE_SAME_RNG=1 run B draws what run A drew (the opponents' event draws are
    # random by design -- this separates them from what the restore misses)
    gens = []
    def _gens(o, depth=0, seen=set()):
        if id(o) in seen or depth > 6:
            return
        seen.add(id(o))
        if isinstance(o, torch.Generator):
            gens.append(o); return
        if isinstance(o, (list, tuple)):
            for v in o: _gens(v, depth + 1)
        elif isinstance(o, dict):
            for v in list(o.values()): _gens(v, depth + 1)
        elif type(o).__module__.startswith("f1sim") and hasattr(o, "__dict__"):
            for v in list(vars(o).values()): _gens(v, depth + 1)
    _gens(env)
    g_states = [x.get_state() for x in gens]
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    # 2. on from here (A)
    A, rsA = [], torch.zeros(G, dtype=torch.bool, device=dev)
    for _ in range(N):
        rsA |= decide(); quiet(); A.append(env.sim.state[lid, :4].clone())
    # 3. every race back to its scene through the training's restart path
    rows = torch.arange(env.B, device=dev)
    g = torch.arange(G, device=dev)
    env._reset_envs_core(rows, fill=False)
    scene.load(g, g)
    env._fill_scans(rows, env.sim.odom.state[rows, 3])
    scene.after_fill(env)                                  # gym_env._reset_envs's order
    g_l, i_l = scene.take_loaded()
    Dd = scene.data
    obs.scans[g_l] = Dd["pol_scans"][i_l]
    obs.rel[g_l] = Dd["pol_rel"][i_l]; obs.rel_v[g_l] = Dd["pol_rel_v"][i_l]
    obs.cmd_d[g_l] = Dd["pol_cmd_d"][i_l]; obs.cmd_v[g_l] = Dd["pol_cmd_v"][i_l]
    obs.plan_xy[g_l] = Dd["pol_plan_xy"][i_l]; obs.plan_v[g_l] = Dd["pol_plan_v"][i_l]
    env.last_result.imu[lid[g_l]] = Dd["pol_imu"][i_l][:, None, :].to(env.last_result.imu.dtype)
    if "pol_mem" in Dd:
        obs.mem_h[g_l] = Dd["pol_mem"][i_l]
    env.ep_limit[:] = 10 ** 6
    quiet()
    s0 = env.sim.state[lid, :4].clone()
    if DIAG:                                               # what the restart + scene did not put back, per kind of race
        opp_race = ~(snap["absent"][:, 1:].all(1))
        rows_o = (torch.nonzero(opp_race)[:, 0][:, None] * M + torch.arange(M, device=dev)[None]).reshape(-1)
        rows_s = (torch.nonzero(~opp_race)[:, 0][:, None] * M + torch.arange(M, device=dev)[None]).reshape(-1)
        for k, v in walk(env).items():
            if k not in full0 or full0[k].shape != v.shape:
                continue
            x, y = full0[k], v
            for nm, rr in (("상대 차 경주", rows_o), ("혼자 경주", rows_s)):
                a_, b_ = x[rr], y[rr]
                if a_.is_floating_point():
                    bad = (a_ != b_) & ~(a_.isnan() & b_.isnan())
                    mag = float((a_ - b_).abs().nan_to_num(0).max()) if bool(bad.any()) else 0.0
                else:
                    bad = a_ != b_; mag = float(bad.sum())
                if bool(bad.any()):
                    print(f"    다름 [{nm}] {k} {tuple(v.shape)} 최대 차 {mag:.3g}")
    if os.environ.get("SCENE_SAME_RNG", "") == "1":
        for x, st_ in zip(gens, g_states):
            x.set_state(st_)
        torch.set_rng_state(cpu_state); torch.cuda.set_rng_state(cuda_state)
        print(f"  난수 생성기 {len(gens)}개 + 전역 상태를 A와 같게 맞춤")
    # 4. the same decisions again (B)
    B_, rsB = [], torch.zeros(G, dtype=torch.bool, device=dev)
    for _ in range(N):
        rsB |= decide(); quiet(); B_.append(env.sim.state[lid, :4].clone())
A, B_ = torch.stack(A), torch.stack(B_)
ok = ~(rsA | rsB)
gap = (A[..., :2] - B_[..., :2]).norm(dim=-1)                       # (N, G)
absent = snap["absent"][:, 1:].all(1) if snap["absent"].shape[1] > 1 else torch.ones(G, dtype=torch.bool, device=dev)
first_mm = torch.where(gap > 1e-3, torch.arange(N, device=dev)[:, None].expand(N, G), torch.full((N, G), N, device=dev)).amin(0)
st_snap = snap["sim_state"][:, 0]
print(f"[장면 재생 자기 점검] {ck_path} · 세계 {WORLD} · 경주 {G} (자리 {M}) · 앞 {W} 결정 · 비교 {N} 결정 · 센서 잡음 0, 평균 행동")
print(f"  되살린 직후 차 자리 차이(장면 저장 순간과): 최대 {float((s0[:, :2] - st_snap[:, :2].float()).norm(dim=1).max()):.2e} m")
for name, m in (("상대 차 없는 경주", absent & ok), ("상대 차 있는 경주", ~absent & ok)):
    if not bool(m.any()):
        print(f"  {name}: 없음"); continue
    gm = gap[:, m]
    print(f"  {name} {int(m.sum())}개: 자리 차이 최대 {float(gm.max()):.2e} m · 끝에서 중앙 {float(gm[-1].median()):.2e} m · "
          f"1 mm 넘게 벌어진 경주 {int((gm.amax(0) > 1e-3).sum())}개 (처음 벌어진 결정 중앙 {float(first_mm[m].float().median()):.0f}) · "
          f"10 cm 넘게 {int((gm.amax(0) > 0.1).sum())}개")
print(f"  다시 출발이 끼어 뺀 경주 {int((~ok).sum())}개")
