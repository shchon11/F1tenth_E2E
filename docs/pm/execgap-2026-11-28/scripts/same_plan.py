"""같은 plan을 추종기마다 따라가게 함 (열린 고리 1.0 s, 다시 계획 없음, 벽·상자 없음, CPU).
plan = R0(이상 실행) 기록에서 정책이 실제로 낸 결정(행동 20개, 출발 속도·조향; exam_record F1SIM_EXAM_SAVE_PLAN)을 뽑아,
이전 개발자의 제어기 방(tools/map_room.py: Room = Sim._roll_physics와 같은 물리에서 벽·LiDAR만 뺀 것, 명령 지연·서보·속도 루프·타이어)을
그대로 씀. 차는 그 출발 속도·조향으로 1.5 s 달려 정상 상태로 만든 뒤 plan을 받음.
추종기: MAP(학습·시험 기본 = state_rl TW_MAP), MAP 전방주시 x1.4, MAP 조향 이득 1.0, MAP 조향표 마찰 0.9배, 시간 추종기(PlanFollower,
콘솔 track), 모형 예측 추종기(PlanMPC), 그리고 '이상 실행'(plan 그 자체).
XG_RAND=1이면 차를 학습 범위(params.py RandomizationConfig 기본) 안에서 무작위로 뽑음(지연·서보·모터·질량·타이어 강성 등).
인자: R0_plan.npz 사례수 출력npz"""
import os, sys, math, json
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("XG_THREADS", "4")))
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
import map_room as MR
from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn

rec_npz, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dev = torch.device("cpu")
torch.manual_seed(0)
room = MR.Room(n, dev)
if os.environ.get("XG_RAND") == "1":
    from f1sim.params import Config
    from f1sim.randomization import ParamSet
    cfg = Config()
    ps = ParamSet(cfg, n, dev, torch.Generator(device="cpu").manual_seed(7))
    ps.resample(torch.arange(n))
    for k in room.P:
        if k in ps.P and torch.is_tensor(ps.P[k]):
            room.P[k] = ps.P[k].clone()
    room.P["mu"] = torch.full_like(room.P["mu"], float(room.nominal["mu"]))      # training pins the friction (state_rl)
    room.H = int(math.ceil(float(room.P["cmd_delay"].max()) / room.cdt)) + 2
print({k: (round(float(room.P[k].min()), 4), round(float(room.P[k].max()), 4)) for k in ("cmd_delay", "servo_tau", "motor_tau", "a_max", "mu", "B_f")}, flush=True)

z = np.load(rec_npz)
v0r, d0r, ar = z["v0"].reshape(-1), z["d0"].reshape(-1), z["act"].reshape(-1, 2 * PS.N_STEP)
pick = np.random.default_rng(1128).choice(np.nonzero(v0r >= 1.5)[0], n, replace=False)
t_ = lambda a: torch.tensor(a[pick], dtype=torch.float32, device=dev)
vr, dr = t_(v0r), t_(d0r)
zr = torch.zeros(n, device=dev)
c = dict(v_set=vr, d_cmd=dr, d_prev=dr, t_sw=zr + 99.0, trans=zr.bool(), act=t_(ar), off=zr.bool(), e_y=zr, e_p=zr,
         rec=torch.ones(n, dtype=torch.bool, device=dev))
st0, ax0, hist0 = room.settle(c)
res = {}
TRJ = {}
TW = dict(MR.NOW)
VAR = {"MAP(기본)": TW, "MAP 전방주시 x1.4": dict(TW, t_clip_min=TW["t_clip_min"] * 1.4, m_l1=TW["m_l1"] * 1.4, q_l1=TW["q_l1"] * 1.4),
       "MAP 조향 이득 1.0": dict(TW, v_steer_gain_max=1.0), "MAP 조향표 0.9배": TW}
for name, mp in VAR.items():
    _gi = M.grip_index
    if name == "MAP 조향표 0.9배":
        M.grip_index = lambda P, nom: torch.full_like(_gi(P, nom), M.GRIP_LEVELS.index(0.9))
    p, S, C = room.follow(c, st0, ax0, hist0, mp)
    M.grip_index = _gi
    res[name] = MR.metrics(p, S, C, room.cdt, room.lim.wheelbase); TRJ[name] = S[..., :3].numpy()
    print(name, MR.summary(res[name], np.ones(n, bool)), flush=True)
for kind, nm in (("time_table", "시간 추종기(PlanFollower)"), ("mpc", "모형 예측(PlanMPC)")):
    p, S, C = MR.follow_ref(room, c, st0, ax0, hist0, kind)
    res[nm] = MR.metrics(p, S, C, room.cdt, room.lim.wheelbase); TRJ[nm] = S[..., :3].numpy()
    print(nm, MR.summary(res[nm], np.ones(n, bool)), flush=True)
TRJ["plan"] = np.stack([p["x"].numpy(), p["y"].numpy(), p["psi"].numpy()], -1)
np.savez_compressed(out, **{f"m__{k}__{q}": v for k, m in res.items() for q, v in m.items() if np.ndim(v) >= 1},
                    **{f"trj__{k}": v for k, v in TRJ.items()}, pick=pick, v0=vr.numpy(), d0=dr.numpy())
print("saved", out)
