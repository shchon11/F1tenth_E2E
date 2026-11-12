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
F1SIM_EXAM_SEED: 두 번째 시드(같은 조건에서 숫자가 얼마나 흔들리나).
F1SIM_EXAM_SLIP_FRAME=oracle (11-12, 원인 검증용이지 시험 점수가 아님): plan의 모형은 미끄러지지 않는 차(차체 방향에서 조금 안쪽으로 나아감)인데
실제 차는 코너에서 차체 방향보다 바깥쪽으로 나아감 -> policy가 보는 스캔·가속도계와 plan의 출발 방향을 '미끄러지지 않는 차라면 지금 진행
방향으로 나아가고 있을 차체 방향'으로 돌려 줌 (돌리는 각 = 실제 진행 방향(시뮬레이터의 참 옆속도: 실차에는 없는 값) − 모형의 진행 방향).
MAP이 받는 구간별 곡률·속도는 그대로. 기록에 그 각(dlt, 결정마다)을 남김.
F1SIM_EXAM_SLIP_FRAME=table + F1SIM_EXAM_SLIP_TABLE=표npz: 같은 일을 참 옆속도 없이, 차가 스스로 아는 값(속도 x 회전 속도 = 옆가속도, 속도)으로
표(ay, v, beta[rad])에서 읽은 미끄럼 각으로 함 (속도와 회전 속도는 시뮬레이터의 참값: 센서 잡음은 안 넣음). 참 옆속도(vy)는 기록에만 남김.
F1SIM_EXAM_SPEED_LEAD=초 (제어기 쪽 진단, 시험 점수가 아님): MAP이 속도 명령으로 plan의 첫 0.1 s 구간 속도 대신 그 초만큼 뒤의 속도를 냄
(속도 루프가 명령을 시간 상수 0.2 s로 따라가서, 첫 구간 속도를 그대로 주면 plan이 낸 가속의 절반도 안 나옴).
F1SIM_EXAM_SAVE_MEM=1 (11-12, 사용자: "기억이 반복이 되던데 그 반복되는거에서 특정 구간을 뭔가 이용할 수는 없나"): 결정마다 policy가 읽은
특징(스캔 인코더 출력 + 정규화한 상태 입력; 기억이 읽는 것)과 그 결정 뒤의 기억 상태를 `출력_mem.npz`(feat, mem; float16)에 따로 남김.
주행·기록은 그대로.
F1SIM_EXAM_OPP_SLOT=JSON (11-12, 시험 점수가 아니라 상대 차 하나를 따로 보는 기록): 상대 차 자리 설정에 덮어씀(state_rl.build_env의
`--opp-slot`과 같음). 예: AuRoRus 포팅본만 상대로 `{"kind": "aurorus", "kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0,
"speed_scale": [1.0, 1.0]}` (그 차는 자기 선 속도의 0.8 x speed_scale로 달리고, 자기 LiDAR로 본 점만으로 비켜 감). 기록의 opp_slot에 남김.
F1SIM_EXAM_START_REST=1 (11-12, 대회 규칙의 출발을 따로 보는 기록): 모든 차가 선 채로(0 m/s) 출발. 규칙(RoboRacer 2025)은 1대1을 같은 출발선에서
30 cm 떨어져 나란히 세움 -> `F1SIM_EXAM_OPP_SLOT='{"spawn": "alongside"}'`(상대 차 종류·속도는 그대로, 배치만 나란히)와 같이 씀.
기록의 start_rest에 남김.
F1SIM_EXAM_HALF_LAP=1 (11-12, 사용자가 나간 대회의 1대1 출발): 상대 차를 반 바퀴 떨어진 반대쪽에 세움(spawn_gap = 트랙 길이의 절반).
`F1SIM_EXAM_OPP_SLOT='{"spawn": "ahead"}'` + F1SIM_EXAM_START_REST=1과 같이 씀. 기록의 half_lap에 남김."""
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
SLIP = os.environ.get("F1SIM_EXAM_SLIP_FRAME", "")
assert SLIP in ("", "oracle", "table"), SLIP
LEAD = float(os.environ.get("F1SIM_EXAM_SPEED_LEAD") or 0.0)
SAVE_MEM = os.environ.get("F1SIM_EXAM_SAVE_MEM", "") == "1"
OPP_SLOT = os.environ.get("F1SIM_EXAM_OPP_SLOT", "")                # "" = the training world's opponents
REST = os.environ.get("F1SIM_EXAM_START_REST", "") == "1"        # every car starts standing (the rule's grid)
HALF = os.environ.get("F1SIM_EXAM_HALF_LAP", "") == "1"          # the opponent starts half a lap away (the user's events)

torch.set_grad_enabled(False)
ck_path, out_npz = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "map"
n_env = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
count = sys.argv[5] if len(sys.argv) > 5 else "max2"
secs = float(sys.argv[6]) if len(sys.argv) > 6 else 90.0
rm_t = float(sys.argv[7]) if len(sys.argv) > 7 else 50.0
assert mode in ("map", "track", "ideal"), mode
dev = torch.device(os.environ.get("F1SIM_EXAM_DEVICE", "cuda"))      # "cpu": a handful of sessions for a quick look
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
from f1sim import dynamics as dyn
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.lidar import HIT_CAR
from f1sim import props as _props

a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = n_env, SEED, dev.type
a.episode_s = secs + 5.0                                   # one episode per car: no respawn inside the session
a.warps, a.track_scales = 0, None                          # the real map only
a.prop_bank, a.prop_count = BANK, count
if OPP_SLOT:
    a.opp_slot = OPP_SLOT                                   # merged over the training world's slot (`state_rl.build_env`)
if mode == "map":
    a.plan_format, a.repeat = "tw", 4
torch.manual_seed(SEED)
env = SR.build_env(a, dev, extra={"spawn_speed_max": 0.0} if REST else None)
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
if mode == "ideal":
    env.enable_ideal()
if LEAD > 0:
    assert mode == "map", mode
    env.tracker.tw_speed_lead = LEAD
if HALF:
    h = 0.5 * float(env.sim.track.length.min())                 # the real map only: one lap length
    env.ecfg.spawn_gap = (h, h)                                 # read when the reset lays the race out
    print(f"반 바퀴 출발: 트랙 길이 {[round(float(x), 1) for x in env.sim.track.length]} m -> 간격 {h:.1f} m", flush=True)
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


class SlipObs(PlanObs2):
    """`PlanObs2` from the frame turned by `turn` (B_l,) [rad] about the CoG: the accelerometer's x / y in that frame (the scan
    is turned at the scanner, `sim.P["mount_yaw"]`; the gyro's z and the last plan in its own frame do not change)"""
    turn = None

    def state(self) -> torch.Tensor:
        s = super().state()
        if self.turn is None:
            return s
        c, s_ = torch.cos(self.turn), torch.sin(self.turn)
        o = s.clone()
        o[:, 3] = s[:, 3] * c + s[:, 4] * s_
        o[:, 4] = -s[:, 3] * s_ + s[:, 4] * c
        return o


if SLIP == "table":
    _t = np.load(os.environ["F1SIM_EXAM_SLIP_TABLE"])
    TAB_A, TAB_V, TAB_B = (torch.tensor(_t[k], dtype=torch.float32, device=dev) for k in ("ay", "v", "beta"))
    assert TAB_B.shape == (len(TAB_A), len(TAB_V)) and float(TAB_B.abs().max()) < 0.5


def table_slip(ay: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[rad] slip angle read from the table (rows = lateral acceleration, columns = speed), bilinear, held at the table's edges"""
    def cell(x, g):
        x = x.clamp(g[0], g[-1]); i = torch.bucketize(x, g).clamp(1, len(g) - 1)
        return i, (x - g[i - 1]) / (g[i] - g[i - 1])
    ia, wa = cell(ay, TAB_A); iv, wv = cell(v, TAB_V)
    return ((1 - wa) * (1 - wv) * TAB_B[ia - 1, iv - 1] + wa * (1 - wv) * TAB_B[ia, iv - 1]
            + (1 - wa) * wv * TAB_B[ia - 1, iv] + wa * wv * TAB_B[ia, iv])


def slip_turn(d_next: torch.Tensor) -> torch.Tensor:
    """[rad] (B_l,) the car's true direction of travel relative to its body (atan2(vy, vx)) minus the plan model's (a
    kinematic bicycle about the CoG at the next plan's first steering angle `d_next`); fades to 0 below 1.5 m/s, 0 backward"""
    s = env.sim.state[lid]
    vx, vy = s[:, dyn.IVX], s[:, dyn.IVY]
    if SLIP == "table":                                         # from what the car knows: speed and yaw rate
        b_real = table_slip(vx * s[:, dyn.IR], vx)
    else:
        b_real = torch.atan2(vy, vx.clamp_min(1e-3))
    b_kin = torch.atan(lim.lr / lim.wheelbase * torch.tan(d_next))
    w = (vx - 0.5).clamp(0.0, 1.0)
    return (w * (b_real - b_kin)).clamp(-0.35, 0.35)


obs = (SlipObs if SLIP else PlanObs2)(env, lim, pr.goal)
po = env.procedural
remover = SR.PropRemoval(env, 1.0, rm_t, rm_t, seed=SEED) if po is not None else None
restarter = SR.OppRestart(env, OPP_RESTART_S, PS.DT) if env.M > 1 else None
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
COLS = ("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")
I_CNT = [CC.index(k) for k in COLS]
i_hk, i_hx, i_hy, i_hs = (CC.index(k) for k in ("held_kind", "held_x", "held_y", "held_severity"))
KN = list(getattr(env, "alt_teacher_kinds", ())) + [str(json.loads(OPP_SLOT).get("kind", "raceline")) if OPP_SLOT else "raceline"]
ii = PT_IDX.to(dev)
T = int(round(secs / PS.DT))
R = {k: [] for k in ("st", "op", "on", "pl", "rs", "cnt", "ocnt", "held", "sid", "pose", "tid", "kind", "scale", "lap", "ors",
                     "rem", "orem")}
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
turn = torch.zeros(Bl, device=dev)                          # the frame the newest scan was taken in (SLIP)
if SLIP:
    R["dlt"] = []; R["vy"] = []
    yaw0 = env.sim.P["mount_yaw"][lid].clone()
    # the scanner does turn with P["mount_yaw"] (a compiled scan path could have kept its angles): a scan taken 0.2 rad round
    # must be the scan before, 0.2 rad of beams along (one control step of a standing car, before the session's first decision)
    if mode == "map":
        env.ext_plan_kv = (kk, vv)                          # a standing plan
    _s0 = obs._scan().clone(); env.sim.P["mount_yaw"][lid] = yaw0 + 0.2; env.step(act0)
    _s1 = obs._scan().clone(); env.sim.P["mount_yaw"][lid] = yaw0
    _n = int(round(0.2 / float(env.sim.lidar.angles[1] - env.sim.lidar.angles[0])))
    _e_turn = float((_s1[:, :-_n] - _s0[:, _n:]).abs().median()); _e_same = float((_s1 - _s0).abs().median())
    print(f"스캐너 돌림 확인: 0.2 rad({_n} 빔) 돌린 스캔과 원래 스캔을 {_n} 빔 민 것의 차 중앙 {_e_turn:.4f}, 안 민 것과의 차 {_e_same:.4f} (거리/최대거리)", flush=True)
    assert _e_turn < 0.25 * max(_e_same, 1e-4), "the scan did not turn with P['mount_yaw']"
    env.step(act0); obs.reset_rows(torch.arange(Bl, device=dev))


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


MEM = {"feat": [], "mem": []}
if SAVE_MEM:
    assert getattr(actor, "memory", 0), "이 checkpoint에는 기억이 없습니다"
t0 = time.time()
for kd in range(T):
    snap()
    st = env.sim.state[lid].clone()
    if SLIP:
        R["dlt"].append(turn.float().cpu()); R["vy"].append(st[:, dyn.IVY].float().cpu())
        st[:, dyn.IYAW] = st[:, dyn.IYAW] + turn            # the plan starts in the frame its scan was taken in
        obs.turn = turn
    if SAVE_MEM:
        MEM["feat"].append(torch.cat([actor.enc(obs.scans), actor.norm(obs.state())], 1).half().cpu())
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    if SAVE_MEM:
        MEM["mem"].append(obs.mem_h.half().cpu())
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
        if SLIP:
            turn = slip_turn(p["delta"][:, PS.SUB]); env.sim.P["mount_yaw"][lid] = yaw0 + turn
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
         exam=np.array(EXAM + ("+slip_" + SLIP if SLIP else "") + (f"+lead{LEAD:.2f}" if LEAD > 0 else "")), speed_lead=np.array(LEAD), mode=np.array(mode), count=np.array(count), rm_t=np.array(rm_t), seed=np.array(SEED),
         bank=np.array(f"{BANK} ({', '.join(_bank)}; sha256 {_sha})"), opp_restart_s=np.array(OPP_RESTART_S),
         lidar_mount_x=np.array(_mx), lat_margin=np.array(float(_mg) if _mg is not None else 1.0), opp_slot=np.array(OPP_SLOT), start_rest=np.array(REST), half_lap=np.array(HALF))
for k in sorted({int(s) for s in np.unique(D["sid"]) if s >= 0}):
    sh = po.shapes[k]
    D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez_compressed(out_npz, **D)
if SAVE_MEM:
    np.savez(out_npz[:-4] + "_mem.npz", feat=torch.stack(MEM["feat"]).numpy(), mem=torch.stack(MEM["mem"]).numpy())
c = D["cnt"][-1].sum(0)
print(f"[{EXAM}{' · 진행 방향 틀(' + SLIP + ')' if SLIP else ''} · {mode} · 상자 {count} · 시드 {SEED}] {D['name']}: 세션 {Bl}개 x {secs:.0f} s (상자는 {rm_t:g} s에 치움), 도중 다시 출발 "
      f"{int(D['rs'].sum())}번, 선 상대 차 다시 출발 {restarter.count if restarter is not None else 0}번 | 누적: "
      + " · ".join(f"{k} {int(v)}" for k, v in zip(COLS, c)) + f" | {time.time() - t0:.0f} s | 저장 {out_npz}", flush=True)
