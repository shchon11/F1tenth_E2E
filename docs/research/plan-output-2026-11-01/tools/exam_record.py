"""시험 기록 (exam1, 11-12). 학습 표(plan_validate ideal -> plan_table)는 학습 세계 그대로라 "학습이 움직이나"만 말해 줌(11-11: 같은
트랙 변형·같은 상자 은행·plan대로 정확히·40 s·세션 단위 점수 없음·기준 행 없음). 이 도구는 조건을 한 번 정해 얼려 둔 시험:
  - 트랙: 실제 Korea 2025 지도 하나 (뒤튼 변형·크기 변형 없음)
  - 실행: 물리 전부 + 제어기 (map = MAP이 plan을 따라감, 실차 방식 / track = 콘솔 기본인 시간 추종기 / ideal = plan대로 정확히, 비교용)
  - 상자: 학습에 쓰인 적 없는 은행(prop_banks_exam1, 11-12 새로 뽑음), 개수는 인자 (max2 = 대회처럼 2개 / uniform = 1개~전부, 학습 분포)
  - 세션: 90 s 통째(약 10바퀴), 상자가 있는 세션은 50 s에 모두 치움 (대회 형식: 고정 배치 -> 치우면 가장 빠른 길로)
  - 네 경우(혼자·상대 차 x 상자 유무)는 학습 세계와 같은 반반 추첨, 상대 차는 학습 세계의 종류·속도 배율 0.4~1.2 그대로,
    2 s 동안 0.5 m도 못 간 상대 차는 다시 출발 (모든 checkpoint에 똑같이)
  - 고정 시드, policy 평균 행동, LiDAR 자리·타이어 한계 여유는 checkpoint가 학습한 값
기록 형식은 drive_record.py와 같아(장면 도구가 그대로 읽음) 거기에 두 차의 '트랙 따라 남은 거리'(보상의 벽 기준 거리장, 중심선 없음)와
한 바퀴 길이를 더함 -> 누가 더 멀리 갔나·앞지름을 exam_table.py가 셈.
인자: checkpoint 출력npz [제어기 map|track|ideal, 기본 map] [환경 수, 기본 1024] [상자 개수 max2|uniform, 기본 max2] [초 90] [치우는 시각 50]
F1SIM_EXAM_SEED: 두 번째 시드(같은 조건에서 숫자가 얼마나 흔들리나)."""
import os, sys, json, argparse, hashlib, time
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
os.environ.pop("F1SIM_WORLD_TRACKS", None)
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")

EXAM = "exam1"
BANK = "/home/shchon11/f1sim_runs/prop_banks_exam1"
SEED = int(os.environ.get("F1SIM_EXAM_SEED", 20261112))
OPP_RESTART_S = 2.0

torch.set_grad_enabled(False)
ck_path, out_npz = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "map"
n_env = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
count = sys.argv[5] if len(sys.argv) > 5 else "max2"
secs = float(sys.argv[6]) if len(sys.argv) > 6 else 90.0
rm_t = float(sys.argv[7]) if len(sys.argv) > 7 else 50.0
assert mode in ("map", "track", "ideal"), mode
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
assert ck.get("format") == "plan_steps2", ck.get("format")
# the checkpoint's own scanner place and tyre-limit margin, set before anything reads them
from f1sim.learn.plan_policy import lidar_mount_of
_mx = lidar_mount_of(ck)
if _mx > 0:
    os.environ["F1SIM_LIDAR_MOUNT_X"] = str(_mx)
else:
    os.environ.pop("F1SIM_LIDAR_MOUNT_X", None)
_mg = (ck.get("args") or {}).get("plan_lat_margin")
if _mg is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(_mg)
else:
    os.environ.pop("F1SIM_PLAN_LAT_MARGIN", None)
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import PlanReward
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, plan_world, plan_segments, PT_IDX
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import map_controller as M
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.lidar import HIT_CAR
from f1sim import props as _props

a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = n_env, SEED, "cuda"
a.episode_s = secs + 5.0                                   # one episode per car: no respawn inside the session
a.warps, a.track_scales = 0, None                          # the real map only
a.prop_bank, a.prop_count = BANK, count
if mode == "map":
    a.plan_format, a.repeat = "tw", 4
torch.manual_seed(SEED)
env = SR.build_env(a, dev)
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
if mode == "ideal":
    env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=SEED)
lid = env.learner_ids; Bl = int(lid.numel())
follower = None
if mode == "track":
    # the console's follower with its default gains (`plan_policy.PlanRunner`, mode "track")
    from f1sim.plan_follow import PlanFollower, to_car
    follower = PlanFollower(Bl, dev, k_v=2.0, k_lat=1.0, k_head=1.0, s_max=float(env.s_max), lut=env.tracker.map.lut,
                            grip=M.GRIP_LEVELS.index(1.0), wheelbase=float(env.tracker.nominal["lf"]) + float(env.tracker.nominal["lr"]))
    env.ext_mask[lid] = True; env._ext_ids.update(int(i) for i in lid.tolist())
actor = load_actor2(ck, dev)
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
obs = PlanObs2(env, lim, pr.goal)
po = env.procedural
remover = SR.PropRemoval(env, 1.0, rm_t, rm_t, seed=SEED) if po is not None else None
restarter = SR.OppRestart(env, OPP_RESTART_S, PS.DT) if env.M > 1 else None
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
COLS = ("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")
I_CNT = [CC.index(k) for k in COLS]
i_hk, i_hx, i_hy, i_hs = (CC.index(k) for k in ("held_kind", "held_x", "held_y", "held_severity"))
KN = list(getattr(env, "alt_teacher_kinds", ())) + ["raceline"]
ii = PT_IDX.to(dev)
T = int(round(secs / PS.DT))
R = {k: [] for k in ("st", "op", "on", "pl", "rs", "cnt", "ocnt", "held", "sid", "pose", "tid", "kind", "scale", "lap", "ors",
                     "rem", "orem")}
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)


def snap():
    """the world as the decision about to be made sees it"""
    R["st"].append(env.sim.state[lid, :4].float().cpu()); R["op"].append(env.sim.state[others, :4].float().cpu())
    R["on"].append((~absent[others]).cpu())
    if po is None:
        R["sid"].append(torch.full((Bl, 1), -1, dtype=torch.int16)); R["pose"].append(torch.zeros(Bl, 1, 3))
    else:
        R["sid"].append(po.p_sid[eid].to(torch.int16).cpu()); R["pose"].append(po.p_poses[eid].float().cpu())
    R["tid"].append(env.sim.tid[lid].to(torch.int16).cpu())
    code = env.slot_kind_code[others] if hasattr(env, "slot_kind_code") else torch.full((Bl,), -1, device=dev, dtype=torch.long)
    R["kind"].append(torch.where(code < 0, torch.full_like(code, len(KN) - 1), code).to(torch.int8).cpu())
    sc = env.teacher.speed_scale[others] if torch.is_tensor(getattr(env.teacher, "speed_scale", None)) else torch.ones(Bl, device=dev)
    R["scale"].append(sc.float().cpu())
    # the walls-only distance to go of both cars (NaN in a wall / off the grid)
    R["rem"].append(pr.goal.remaining(env.sim.state[lid, :2].float(), env.sim.tid[lid]).float().cpu())
    R["orem"].append(pr.goal.remaining(env.sim.state[others, :2].float(), env.sim.tid[others]).float().cpu())


t0 = time.time()
for kd in range(T):
    snap()
    st = env.sim.state[lid].clone()
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    x, y, psi = plan_world(p, st)
    pc = pr.plan_cost(x, y, psi, p["v"])
    pts = torch.stack([x[:, ii], y[:, ii]], -1)
    if mode == "map":
        k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    elif mode == "track":
        follower.set_plan(slice(None), x, y, psi, p["v"], p["delta"])
    else:
        env.sim.ideal_vcap[lid] = 1e3
    restarted = torch.zeros(Bl, dtype=torch.bool, device=dev)
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        if follower is not None:
            env.ext_cmd[lid] = to_car(env, lid, follower.command(env.sim.state[lid]))
        _, _, te, trc, _ = env.step(act0)
        restarted |= (te | trc)[lid]
        pr.control_step((env.last_result.scan_type[lid] == HIT_CAR).any(1))
    out = pr.decision(pts, pc["price"], restarted)
    obs.push_scan(); obs.set_plan(pts, p["pv"]); obs.set_plan2(p)
    if bool(restarted.any()):
        rows = torch.nonzero(restarted)[:, 0]
        obs.reset_rows(rows)
        if hasattr(follower, "reset_rows"):
            follower.reset_rows(rows)
    if remover is not None:
        remover.step()
    ors = torch.zeros(env.B, dtype=torch.bool, device=dev)
    if restarter is not None:
        pre = env.sim.state[:, :2].clone()
        if restarter.step():
            ors = (env.sim.state[:, :2] - pre).norm(dim=1) > 1e-6          # the restarted cars are the ones that moved
    R["ors"].append(ors[others].cpu())
    R["pl"].append(pts.cpu()); R["rs"].append(restarted.cpu()); R["lap"].append(out["lap"].float().cpu())
    cf = env.sim.contact_frame()
    R["cnt"].append(cf[lid].float()[:, I_CNT].cpu()); R["ocnt"].append(cf[others].float()[:, I_CNT].cpu())
    R["held"].append(cf[lid].float()[:, [i_hk, i_hx, i_hy, i_hs]].cpu())
    if kd in (9, 99) or (kd + 1) % 300 == 0:
        print(f"  {(kd + 1) * PS.DT:5.1f} s / {secs:g} s ({time.time() - t0:.0f} s)", flush=True)
snap()
D = {k: torch.stack(v).numpy() for k, v in R.items()}
trk = env.sim.track
_bank = sorted(f for f in os.listdir(BANK) if f.endswith(".pt"))
_sha = hashlib.sha256(b"".join(open(os.path.join(BANK, f), "rb").read() for f in _bank)).hexdigest()[:12]
D.update(edt=trk.edt.float().cpu().numpy(), t_off=trk.t_off.cpu().numpy(), t_H=trk.t_H.cpu().numpy(), t_W=trk.t_W.cpu().numpy(),
         t_res=trk.t_res.cpu().numpy(), t_origin=trk.t_origin.cpu().numpy(),
         car=np.array([float(env.cfg.vehicle.length), float(env.cfg.vehicle.width)]), dt=np.array(PS.DT),
         kinds=np.array(KN), cols=np.array(COLS), name=np.array(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"),
         ep_s=np.array(secs + 5.0), level=env.level_seed[lid].cpu().numpy(), lapL=pr.goal.lap.float().cpu().numpy(),
         exam=np.array(EXAM), mode=np.array(mode), count=np.array(count), rm_t=np.array(rm_t), seed=np.array(SEED),
         bank=np.array(f"{BANK} ({', '.join(_bank)}; sha256 {_sha})"), opp_restart_s=np.array(OPP_RESTART_S),
         lidar_mount_x=np.array(_mx), lat_margin=np.array(float(_mg) if _mg is not None else 1.0))
for k in sorted({int(s) for s in np.unique(D["sid"]) if s >= 0}):
    sh = po.shapes[k]
    D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez_compressed(out_npz, **D)
c = D["cnt"][-1].sum(0)
print(f"[{EXAM} · {mode} · 상자 {count} · 시드 {SEED}] {D['name']}: 세션 {Bl}개 x {secs:.0f} s (상자는 {rm_t:g} s에 치움), 도중 다시 출발 "
      f"{int(D['rs'].sum())}번, 선 상대 차 다시 출발 {restarter.count if restarter is not None else 0}번 | 누적: "
      + " · ".join(f"{k} {int(v)}" for k, v in zip(COLS, c)) + f" | {time.time() - t0:.0f} s | 저장 {out_npz}", flush=True)
