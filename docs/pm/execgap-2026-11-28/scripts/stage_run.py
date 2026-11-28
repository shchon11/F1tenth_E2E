"""실행 격차 분해 실험 (2026-11-28). 이전 개발자의 exam_record.py를 고치지 않고(runpy) 감싸서, 같은 정책·같은 세션 묶음을
단계별 실행 조건으로 돌리고, 결정마다 plan 전체(조밀한 41점: 위치·방향·속도·조향)와 제어 주기(0.025 s)마다 차 상태·명령을 따로 남긴다.

XG_STAGE (필수):
  ideal        R0  이상 실행 (exam_record mode ideal 그대로: 명령한 조향·속도를 운동학 자전거로 그대로)
  delay        R1  이상 실행 + 명령 지연 (차마다 뽑힌 cmd_delay; 동역학 경로와 같은 명령 이력에서 읽음)
  servo        R2  R1 + 서보 응답 (차마다 servo_tau 1차 지연 + sv_max 속도 한계; 보정 오차 steer_gain/bias 없음)
  servo_nodelay R2s 이상 실행 + 서보 응답만 (지연 없음)
  servo_motor  R2b R2 + 속도 루프 (VESC 1차 지연 motor_tau, a_max/v_switch 출력 한계, a_brake, 구름·공기 저항)
  map          R4  실제 동역학 + MAP (exam_record mode map 그대로, 학습·시험 기본 설정)
  map_la       R5a MAP 전방주시 거리 x1.4 (t_clip_min, m_l1, q_l1 x1.4)
  map_gain     R5b MAP 조향 이득 1.47 -> 1.0 (v_steer_gain_max)
  map_lut      R5c MAP 조향표를 마찰 0.9배 표로 읽음 (같은 횡가속에 더 큰 조향)
  mpc          R3  실제 동역학 + 모형 예측 추종기 plan_mpc.PlanMPC (시뮬레이터와 같은 단일 트랙·Pacejka 모형, 명목 상수)
  pf           R6  실제 동역학 + 시간 추종기 plan_follow.PlanFollower (콘솔 'track', 명목 지연·서보·모터 역보상 + 조향표)
  pf_oracle    R3b R6과 같되 역보상 상수를 차마다 실제로 뽑힌 cmd_delay/servo_tau/motor_tau로 (차의 지연을 정확히 앎)
XG_NOMINAL=1: 물리 무작위화를 끔 (모든 차가 명목 차; 추종기 모형이 정확해짐).
인자: exam_record.py와 같음 (ckpt out.npz n_env count secs rm_t); mode는 단계가 정함.
추가 기록: out의 _xg.npz (dense plan, 제어 주기 상태·명령, 차마다 물리 값)."""
import os, sys, runpy, json, inspect, textwrap
TOOLS = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools"
sys.path.insert(0, TOOLS)
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
import numpy as np
import torch

STAGE = os.environ["XG_STAGE"]
MODE_OF = {"ideal": "ideal", "delay": "ideal", "servo": "ideal", "servo_motor": "ideal", "servo_nodelay": "ideal", "map": "map", "map_la": "map",
           "map_gain": "map", "map_lut": "map", "mpc": "track", "pf": "track", "pf_oracle": "track"}
assert STAGE in MODE_OF, STAGE
MODE = MODE_OF[STAGE]
NOMINAL = os.environ.get("XG_NOMINAL", "") == "1"

# ---------------------------------------------------------------- MAP variants (state_rl reads F1SIM_MAP_OVERRIDE)
TW_MAP = dict(t_clip_min=0.78, m_l1=0.37, q_l1=-0.49)        # state_rl.py TW_MAP (the tuned default)
if STAGE == "map_la":
    os.environ["F1SIM_MAP_OVERRIDE"] = json.dumps({k: v * 1.4 for k, v in TW_MAP.items()})
elif STAGE == "map_gain":
    os.environ["F1SIM_MAP_OVERRIDE"] = json.dumps({"v_steer_gain_max": 1.0})
else:
    os.environ.pop("F1SIM_MAP_OVERRIDE", None)
if STAGE == "mpc":
    os.environ["F1SIM_EXAM_TRACK_MPC"] = "1"
else:
    os.environ.pop("F1SIM_EXAM_TRACK_MPC", None)

import f1sim.sim as SIM
import f1sim.dynamics as dyn
import f1sim.map_controller as MC
import f1sim.learn.common as C
from f1sim.learn import state_rl as SR
import f1sim.learn.plan_policy as PP
from f1sim.learn.plan_reward import PlanReward
from f1sim.plan_follow import PlanFollower

# ---------------------------------------------------------------- R1/R2/R2b: the ideal branch of the roll, patched
if STAGE in ("delay", "servo", "servo_motor", "servo_nodelay"):
    src = textwrap.dedent(inspect.getsource(SIM.Simulator._roll_physics))
    old = ("        d_i = cmd_hist[:, 0, 0]\n"
           "        v_i = torch.maximum(torch.minimum(cmd_hist[:, 0, 1], self.ideal_vcap), -self.ideal_vcap)\n")
    new = ("        d_c = eff[:, 0]; v_c = eff[:, 1]                      # XG: the delayed command (this car's cmd_delay)\n"
           "        if _XG_MODE == 'servo_nodelay':\n"
           "            d_c = cmd_hist[:, 0, 0]; v_c = cmd_hist[:, 0, 1]\n"
           "        if _XG_MODE in ('servo', 'servo_motor', 'servo_nodelay'):\n"
           "            _s0 = st_prev[:, dyn.ISTEER]\n"
           "            _rt = ((d_c - _s0) / P['servo_tau']).clamp(-P['sv_max'], P['sv_max'])\n"
           "            d_i = (_s0 + _rt * self.dt).clamp(-P['s_max'], P['s_max'])\n"
           "        else:\n"
           "            d_i = d_c\n"
           "        v_i = torch.maximum(torch.minimum(v_c, self.ideal_vcap), -self.ideal_vcap)\n"
           "        if _XG_MODE == 'servo_motor':\n"
           "            _v0 = st_prev[:, dyn.IVX]\n"
           "            _a = dyn.motor_accel((v_i - _v0) / P['motor_tau'], _v0, P)\n"
           "            _res = P['c_roll'] * torch.tanh(_v0 / 0.05) + P['c_drag'] * _v0 * _v0.abs()\n"
           "            _res = torch.sign(_v0) * torch.minimum(_res.abs(), _v0.abs() / self.dt)\n"
           "            v_i = _v0 + (_a - _res) * self.dt\n")
    # the source is dedented by 4: the two lines sit at 12 spaces there
    old = textwrap.indent(old, "    "); new = textwrap.indent(new, "    ")
    assert src.count(old) == 1, "ideal branch (indented) not found"
    src = src.replace(old, new)
    g = SIM.__dict__
    g["_XG_MODE"] = STAGE
    loc = {}
    exec(compile(src, "<xg_roll_physics>", "exec"), g, loc)
    SIM.Simulator._roll_physics = loc["_roll_physics"]
    print(f"[XG] roll patched: ideal execution + {STAGE}", flush=True)

# ---------------------------------------------------------------- R5c: MAP reads another grip level's steering table
if STAGE == "map_lut":
    _gi = MC.grip_index
    LV = MC.GRIP_LEVELS.index(0.9)

    def grip_index(P, nominal):
        return torch.full_like(_gi(P, nominal), LV)
    MC.grip_index = grip_index
    print(f"[XG] MAP reads the steering table of grip {MC.GRIP_LEVELS[LV]}", flush=True)

# ---------------------------------------------------------------- capture env; nominal physics
ENV = {}
_mk = C.make_env


def make_env(*a, **k):
    cfg = k.get("cfg")
    if NOMINAL and cfg is not None:
        cfg.rand.enabled = False
        print("[XG] physics randomisation OFF (nominal car)", flush=True)
    return _mk(*a, **k)


C.make_env = make_env
_be = SR.build_env


def build_env(a, device, extra=None):
    env = _be(a, device, extra=extra)
    ENV["env"] = env
    return env


SR.build_env = build_env

# ---------------------------------------------------------------- recording
XG = {"d_st": [], "d_x": [], "d_y": [], "d_psi": [], "d_v": [], "d_delta": [], "d_vhat": [], "d_cmdd": [],
      "c_st": [], "c_cmd": []}
_pv2 = PP.plan_v2


def plan_v2(actor, obs, lim, *a, **k):
    out = _pv2(actor, obs, lim, *a, **k)
    env = ENV["env"]; lid = env.learner_ids
    p = out[4]
    XG["d_st"].append(env.sim.state[lid].float().cpu())
    for kk in ("x", "y", "psi"):
        XG["d_" + kk].append(p[kk].float().cpu())
    XG["d_v"].append(p["v"].half().cpu()); XG["d_delta"].append(p["delta"].half().cpu())
    XG["d_vhat"].append(out[3].detach().float().reshape(-1).cpu()); XG["d_cmdd"].append(obs.cmd_d.float().cpu())
    return out


PP.plan_v2 = plan_v2
_cs = PlanReward.control_step


def control_step(self, *a, **k):
    env = self.env; lid = env.learner_ids
    XG["c_st"].append(env.sim.state[lid].float().cpu()); XG["c_cmd"].append(env.sim.cmd[lid].float().cpu())
    return _cs(self, *a, **k)


PlanReward.control_step = control_step

# ---------------------------------------------------------------- R3b: PlanFollower with each car's own lags
if STAGE == "pf_oracle":
    _cmd = PlanFollower.command

    def command(self, st):
        if not getattr(self, "_xg_set", False):
            env = ENV["env"]; lid = env.learner_ids; P = env.sim.P
            self.delay = P["cmd_delay"][lid].float().to(self.device)
            self.servo_tau = P["servo_tau"][lid].float().to(self.device)
            self.motor_tau = P["motor_tau"][lid].float().to(self.device)
            self._xg_set = True
            print(f"[XG] PlanFollower lags per car: delay {float(self.delay.mean()):.4f} servo {float(self.servo_tau.mean()):.4f} "
                  f"motor {float(self.motor_tau.mean()):.4f}", flush=True)
        return _cmd(self, st)
    PlanFollower.command = command

ck, out = sys.argv[1], sys.argv[2]
sys.argv = [os.path.join(TOOLS, "exam_record.py"), ck, out, MODE] + sys.argv[3:]
print(f"[XG] stage {STAGE} -> exam_record mode {MODE}; MAP override {os.environ.get('F1SIM_MAP_OVERRIDE', '')}; "
      f"track_mpc {os.environ.get('F1SIM_EXAM_TRACK_MPC', '')}; nominal {NOMINAL}", flush=True)
runpy.run_path(os.path.join(TOOLS, "exam_record.py"), run_name="__main__")

env = ENV["env"]; lid = env.learner_ids; P = env.sim.P
D = {k: torch.stack(v).numpy() for k, v in XG.items() if v}
for key in ("cmd_delay", "servo_tau", "motor_tau", "mu", "mu_f_scale", "a_max", "a_brake", "steer_gain", "steer_bias",
            "B_f", "B_r", "m", "Iz", "sv_max", "s_max", "v_switch", "speed_gain"):
    if key in P and torch.is_tensor(P[key]):
        D["P_" + key] = P[key][lid].float().cpu().numpy()
D["stage"] = np.array(STAGE); D["nominal"] = np.array(NOMINAL)
D["control_dt"] = np.array(float(env.sim.control_dt))
np.savez_compressed(out[:-4] + "_xg.npz", **D)
print(f"[XG] saved {out[:-4]}_xg.npz: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in D.items() if v.ndim > 1), flush=True)
for key in ("cmd_delay", "servo_tau", "motor_tau", "a_max", "mu_f_scale"):
    x = D["P_" + key]
    print(f"[XG] physics {key}: min {x.min():.4f} mean {x.mean():.4f} max {x.max():.4f} first3 {np.round(x[:3], 4)}", flush=True)
