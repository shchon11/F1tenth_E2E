"""(11-24, 사용자: "제어기잖아, 튜닝 룸 같은 걸 만들어서 다양한 waypoint 출력이나 ego status에 따라 어떻게 될지 기하학적·결정론적으로
튜닝할 수 있는 부분 아냐?", "체크포인트마다 제어기 튜닝을 해야 하는 게 안 이상했어?") 제어기 튜닝 방: MAP은 차와 plan 형식의 것이지
checkpoint의 것이 아님 -> policy 없이, plan 공간 전체에서 한 번 맞춤.

사례 은행 (결정론적, 한 번 정해 둠):
  - plan = plan_steps.roll(행동, 출발 속도, 출발 조향, 한계): policy가 낼 수 있는 plan 전부가 이 꼴(행동 20개 in [-1, 1], 한계 = 명목 차 +
    MAP 조향표, 옆 여유 0.95). 행동은 매끈한 기저(조향 쪽 코사인 4개, 속도 쪽 2개)의 계수를 Sobol 점으로 뽑음 -- 곧은 길, 원호, 코너 들어가기,
    차선 바꾸기·S자, 가속·감속이 다 그 안에 있음.
  - 차 상태 = 출발 속도 1-8 m/s x 조향(그 속도 타이어 한계의 -90..90 %) x 직전 조향 명령이 바뀐 지 0.025-0.2 s(과도 상태: 서보·요 각속도가
    아직 안 가라앉음) 또는 오래됨(정상 상태). 먼저 그 명령으로 1.5 s 달려 미끄럼·요 각속도·바퀴·명령 지연 줄을 진짜로 만든 뒤 plan을 줌.
    plan의 출발 = 바퀴 속도(차의 속도계), 마지막 조향 명령 (plan_v2와 같음). 일부 사례(1/4)는 plan 원점에서 옆 0-0.2 m · 방향 0-0.1 rad
    어긋난 데서 출발(MAP의 되돌아오기; 실제 주행에선 plan이 늘 차 자리에서 나오므로 따로 셈).
굴림: 그 plan 하나를 1.0 s, 명목 차의 물리 그대로(명령 지연, 서보, 속도 루프·모터 한계, 타이어, 바퀴) -- 벽·LiDAR·policy·다시 계획 없음.
MAP은 MAPPlanTracker 그대로(tw plan, 0.1 s 구간 곡률·속도; 콘솔·실차와 같은 길).
속도 명령은 0.1 s마다 다시 읽음(실차처럼 plan이 0.1 s마다 새로 오면 그때 '앞당김' 초 뒤 속도; 1.0 s 동안 첫 구간 속도로 묶어 두면
속도 오차가 꾸며짐). 차례로 바뀌는 plan(policy가 매번 다르게 낸 것)에서 오는 흔들림은 따로(sequence).
점수(사례마다, 0.1 ... 1.0 s): plan의 시간 점과 위치·옆·앞뒤·속도 차, 조향 흔들림(실제 조향 속도 - plan 조향 속도의 RMS, 명령 조향 속도 부호
바뀜), 요 각속도 넘침(실제 - plan의 최대), 미끄럼각.
인자: measure 이름 설정json [사례 수 16384] -> out/maptune_room/이름.npz + 표 (lab에서)"""
import json, math, os, sys, time
import numpy as np
import torch

from f1sim.params import Config
from f1sim.randomization import ParamSet
from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim.actuators import servo_target, vesc_accel

#: the follower the console drives today (state_rl.plan_env_fields: PLAN_MAP under TW_MAP, speed_lookahead 0)
PLAN_MAP = {"t_clip_min": 0.9573609450243445, "t_clip_max": 4.779545745206895, "m_l1": 0.4731895571264694,
            "q_l1": -0.4957209879277062, "speed_lookahead": 0.0, "lat_err_coeff": 0.0, "acc_scaler_for_steer": 0.8857957332213734,
            "dec_scaler_for_steer": 0.8092827304787531, "start_scale_speed": 8.700352304982314,
            "downscale_factor": 0.24481196266259284, "speed_lookahead_for_steer": 0.060053547789199635,
            "v_steer_gain_max": 1.2810367013484487, "l1_speed": "measured", "l1_floor": True, "l1_point": "arc",
            "steer_speed": "planned", "lut_interp": "code", "delay_comp": 0.021416617709494996, "lat_slow": 0.0,
            "end_scale_speed": 9.580093136639107}
TW_MAP = dict(t_clip_min=0.78, m_l1=0.37, q_l1=-0.49, lut_interp="linear", v_steer_gain_max=1.47, acc_scaler_for_steer=1.0,
              dec_scaler_for_steer=1.0, delay_comp=0.009, speed_lookahead_for_steer=0.133, tw_speed_lead=0.141)
NOW = dict(PLAN_MAP, speed_lookahead=0.0, **TW_MAP)
SETTLE_S = 1.5
T_S = 1.0


class _Inner:
    """the env's iLQR in MAPPlanTracker's place: every row is MAP's, so it is asked for nothing but its shape"""
    def __init__(self, B, dev, s_max):
        self.B, self.device, self.s_max, self.spec, self.last_ref = B, dev, s_max, None, None

    def __call__(self, action, v_meas, cap, yaw_rate=None, delay=None):
        return torch.zeros(self.B, 2, device=self.device)


def sobol(n, d, seed=1124):
    from scipy.stats import qmc
    return torch.tensor(qmc.Sobol(d, scramble=True, seed=seed).random(n), dtype=torch.float32)


class Room:
    def __init__(self, B, dev, dr=None):
        self.B, self.dev = B, dev
        self.cfg = cfg = Config()
        self.ps = ParamSet(cfg, B, dev, torch.Generator(device="cpu"))
        self.P = self.ps.P
        for k, v in (dr or {}).items():                              # a car at a randomisation extreme: {name: value}
            self.P[k].fill_(float(v))
        self.nominal = dict(self.ps.nominal)
        self.dt = float(cfg.sim.physics_dt); self.cdt = 1.0 / float(cfg.sim.control_rate)
        self.sub = int(round(self.cdt / self.dt)); self.wheel = bool(cfg.vehicle.wheel_model)
        self.lut, _ = M.config_lut(cfg, device=dev)
        self.lim = PS.Limits.from_nominal(self.nominal, self.lut, M.GRIP_LEVELS.index(1.0), dev, margin=0.95)
        self.H = int(math.ceil(float(self.P["cmd_delay"].max()) / self.cdt)) + 2
        self.ar = torch.arange(B, device=dev)
        self.L = float(self.nominal["lf"]) + float(self.nominal["lr"])

    # ------------------------------------------------------------------ physics (Sim._roll_physics without walls / IMU)
    def control_step(self, state, ax, hist, cmd):
        hist = torch.cat([cmd[:, None, :], hist[:, :-1]], 1)
        P = self.P
        for k in range(self.sub):
            age = P["cmd_delay"] - (k + 1) * self.dt
            idx = torch.ceil(age / self.cdt).long().clamp(0, self.H - 1)
            idx_m = torch.ceil((age + P["motor_delay"]) / self.cdt).long().clamp(0, self.H - 1)
            st_c, sp_c = hist[self.ar, idx, 0], hist[self.ar, idx_m, 1]
            steer_tgt = servo_target(st_c, P)
            v_fb = state[:, dyn.IOMEGA] * P["r_w"] if self.wheel else state[:, dyn.IVX]
            a_cmd = vesc_accel(sp_c, v_fb, P)
            state, ax, _ = dyn.step_dynamics(state, steer_tgt, a_cmd, ax, P, P["servo_tau"], self.dt, wheel=self.wheel)
        return state, ax, hist

    def v_meas(self, state):
        return state[:, dyn.IOMEGA] * self.P["r_w"] if self.wheel else state[:, dyn.IVX]

    # ------------------------------------------------------------------ the bank
    def bank(self, n, seed=1124, n_rec=0, rec_npz=None):
        """cases: settle commands (speed, steering now, steering before + when it changed), plan action, start offset"""
        u = sobol(n, 14, seed).to(self.dev)
        B = n
        v_set = 1.0 + 8.5 * u[:, 0]                                  # recorded plans start at 2.4-9.1 m/s (p1-p99)
        d_grip = PS.kappa_to_steer(self.lim.kappa_at(v_set), self.lim).clamp(max=self.lim.s_max)
        d_cmd = (2 * u[:, 1] - 1) * 0.9 * d_grip
        trans = u[:, 2] < 0.5                                         # half: the steering command changed just before
        d_prev = torch.where(trans, (d_cmd + (2 * u[:, 3] - 1) * 0.6 * d_grip).clamp(-0.9 * d_grip, 0.9 * d_grip), d_cmd)
        t_sw = torch.where(trans, 0.025 + 0.175 * u[:, 4], torch.full_like(v_set, 99.0))
        kk = torch.arange(PS.N_STEP, device=self.dev, dtype=torch.float32)[None, :] / (PS.N_STEP - 1)
        c = (2 * u[:, 5:9] - 1) * 0.7
        turn = c[:, :1] + sum(c[:, j:j + 1] * torch.cos(math.pi * j * kk) for j in (1, 2, 3))
        s = (2 * u[:, 9:11] - 1) * 0.8
        speed = s[:, :1] + s[:, 1:2] * torch.cos(math.pi * kk)
        act = torch.cat([turn, speed], 1).clamp(-1, 1)
        off = u[:, 11] < 0.25                                         # a quarter: start off the plan's origin
        e_y = torch.where(off, (2 * u[:, 12] - 1) * 0.2, torch.zeros_like(v_set))
        e_p = torch.where(off, (2 * u[:, 13] - 1) * 0.1, torch.zeros_like(v_set))
        c = dict(v_set=v_set, d_cmd=d_cmd, d_prev=d_prev, t_sw=t_sw, trans=trans, act=act, off=off, e_y=e_y, e_p=e_p,
                 rec=torch.zeros(B, dtype=torch.bool, device=self.dev))
        if n_rec:
            # decisions a policy really made (exam_record.py F1SIM_EXAM_SAVE_PLAN under exact execution): its action from its start
            # speed and steering, the car settled there in steady state first -- the bank's check and its heavier weight
            z = np.load(rec_npz)
            v0r, d0r, ar = z["v0"].reshape(-1), z["d0"].reshape(-1), z["act"].reshape(-1, 2 * PS.N_STEP)
            pick = np.random.default_rng(seed).choice(np.nonzero(v0r >= 1.5)[0], n_rec, replace=False)
            t_ = lambda a: torch.tensor(a[pick], dtype=torch.float32, device=self.dev)
            vr, dr = t_(v0r), t_(d0r)
            zr = torch.zeros(n_rec, device=self.dev)
            rec = dict(v_set=vr, d_cmd=dr, d_prev=dr, t_sw=zr + 99.0, trans=zr.bool(), act=t_(ar), off=zr.bool(), e_y=zr, e_p=zr,
                       rec=torch.ones(n_rec, dtype=torch.bool, device=self.dev))
            c = {k: torch.cat([c[k], rec[k]], 0) for k in c}
        return c

    def settle(self, c):
        """drive each case's settle commands SETTLE_S: (state, ax, command history), the car as it is when the plan comes"""
        B = self.B
        st = torch.zeros(B, dyn.STATE_DIM, device=self.dev)
        v0 = c["v_set"] * 0.97
        st[:, dyn.IVX] = v0; st[:, dyn.ISTEER] = c["d_prev"]
        st[:, dyn.IR] = v0 * torch.tan(c["d_prev"]) / self.L
        st[:, dyn.IOMEGA] = v0 / self.P["r_w"]
        ax = torch.zeros(B, device=self.dev)
        hist = torch.stack([c["d_prev"], c["v_set"]], 1)[:, None, :].repeat(1, self.H, 1)
        n = int(round(SETTLE_S / self.cdt))
        for i in range(n):
            t_left = (n - i) * self.cdt
            d = torch.where(t_left <= c["t_sw"], c["d_cmd"], c["d_prev"])
            st, ax, hist = self.control_step(st, ax, hist, torch.stack([d, c["v_set"]], 1))
        return st, ax, hist

    # ------------------------------------------------------------------ one plan, followed
    def follow(self, c, st0, ax0, hist0, mp: dict):
        mp = dict(mp)
        lead = float(mp.pop("tw_speed_lead", 0.0))
        trk = M.MAPPlanTracker(_Inner(self.B, self.dev, self.lim.s_max), self.lut, M.MAPParams.from_dict(dict(M.POLICY_MAP, **mp)),
                               self.nominal)
        trk.map_rows = torch.ones(self.B, dtype=torch.bool, device=self.dev)
        trk.P = self.P; trk.tw = True; trk.tw_fmt = "tw"; trk.tw_speed_lead = lead; trk.auto_backout = False
        v0 = self.v_meas(st0)
        p = PS.roll(c["act"], v0, c["d_cmd"], self.lim)
        idx = torch.arange(0, PS.N_STEP + 1, device=self.dev) * PS.SUB
        dk, vk = p["delta"][:, idx], p["v"][:, idx]
        kap = torch.tan(0.5 * (dk[:, 1:] + dk[:, :-1])) / self.lim.wheelbase
        trk.tw_kv = (kap.contiguous(), (0.5 * (vk[:, 1:] + vk[:, :-1])).contiguous())
        st = st0.clone(); st[:, dyn.IX] = 0.0; st[:, dyn.IY] = c["e_y"]; st[:, dyn.IYAW] = c["e_p"]
        ax, hist = ax0.clone(), hist0.clone()
        delay = self.P["cmd_delay"] + 0.5 * self.P["servo_tau"]          # the env's tracker_delay (its 0-0.02 s draw left out)
        cap = torch.full((self.B,), 10.0, device=self.dev)
        a0 = torch.zeros(self.B, 2, device=self.dev)
        # the plan is issued at the origin (its own frame); a dry call stores it there, then every call reads the car's pose
        # in that frame -- the hold the env runs between decisions (`EnvConfig.decision_every`)
        trk.hold_rows = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        trk.pose_now = torch.zeros(self.B, 3, device=self.dev)
        trk(a0, v0, cap, st[:, dyn.IR], delay=delay)
        trk.map.curr_steer = c["d_cmd"].clone()                       # the rate limit's reference: the last command
        trk.hold_rows = torch.ones(self.B, dtype=torch.bool, device=self.dev)
        n = int(round(T_S / self.cdt))
        S, C = [st[:, :8].clone()], []
        for j in range(n):
            trk.pose_now = st[:, :3]
            # the speed is re-read every 0.1 s as if the plan were re-issued then (on the car a new plan comes every 0.1 s and MAP
            # commands its speed `lead` s on; held 1.0 s, the first segment's speed would stand for the whole second)
            w = j // PS.SUB
            trk.tw_speed_lead = lead + PS.DT * w + (1e-3 if (w > 0 and lead == 0) else 0.0)
            cmd = trk(a0, self.v_meas(st), cap, st[:, dyn.IR], delay=delay)
            st, ax, hist = self.control_step(st, ax, hist, cmd)
            S.append(st[:, :8].clone()); C.append(cmd.clone())
        return p, torch.stack(S, 1), torch.stack(C, 1)


def follow_ref(room, c, st0, ax0, hist0, kind, p=None):
    """the same cases with another follower already in the repo, its defaults (no tuning): "time_table" = plan_follow.PlanFollower
    with MAP's steering table (the exam's "track" / console's time follower), "time_kin" = the same without the table,
    "mpc" = plan_mpc.PlanMPC (iLQR on the single-track model, batched)"""
    from f1sim.plan_follow import PlanFollower
    B, dev, lim = room.B, room.dev, room.lim
    v0 = room.v_meas(st0)
    if p is None:
        p = PS.roll(c["act"], v0, c["d_cmd"], lim)
    g = M.GRIP_LEVELS.index(1.0)
    nom = room.nominal
    if kind == "mpc":
        from f1sim.plan_mpc import PlanMPC
        fol = PlanMPC(B, dev, nom, s_max=lim.s_max, lut=room.lut, grip=g)
    else:
        fol = PlanFollower(B, dev, motor_tau=float(nom["motor_tau"]), servo_tau=float(nom["servo_tau"]), delay=float(nom["cmd_delay"]),
                           k_v=2.0, k_lat=1.0, k_head=1.0, s_max=lim.s_max, lut=room.lut if kind == "time_table" else None, grip=g,
                           wheelbase=lim.wheelbase)
    fol.set_plan(torch.arange(B, device=dev), p["x"], p["y"], p["psi"], p["v"], p["delta"])
    st = st0.clone(); st[:, dyn.IX] = 0.0; st[:, dyn.IY] = c["e_y"]; st[:, dyn.IYAW] = c["e_p"]
    ax, hist = ax0.clone(), hist0.clone()
    S, C = [st[:, :8].clone()], []
    for j in range(int(round(T_S / room.cdt))):
        cmd = fol.command(st)
        st, ax, hist = room.control_step(st, ax, hist, cmd)
        S.append(st[:, :8].clone()); C.append(cmd.clone())
    return p, torch.stack(S, 1), torch.stack(C, 1)


def metrics(p, S, C, cdt, wheelbase):
    """per case: errors to the time-indexed plan at 0.1 .. 1.0 s, steering / yaw-rate wobble, slip (all numpy)"""
    n = C.shape[1]
    px, py, ppsi, pv, pd = (p[k][:, :n + 1] for k in ("x", "y", "psi", "v", "delta"))
    x, y = S[..., dyn.IX], S[..., dyn.IY]
    dx, dy = x - px, y - py
    c_, s_ = torch.cos(ppsi), torch.sin(ppsi)
    lon = dx * c_ + dy * s_; lat = -dx * s_ + dy * c_                 # + = left of the plan
    dv = S[..., dyn.IVX] - pv
    j = torch.arange(1, 11, device=S.device) * 4
    # + = toward the inside of the plan's turn over its first 0.5 s (left-right averages would cancel)
    turn = torch.sign(ppsi[:, 20] - ppsi[:, 0]) * ((ppsi[:, 20] - ppsi[:, 0]).abs() > 0.05)
    out = {"pos": (dx ** 2 + dy ** 2).sqrt()[:, j], "lat": lat[:, j] * turn[:, None], "lon": lon[:, j], "dv": dv[:, j]}
    b_kin = torch.atan(0.5 * torch.tan(S[:, 0, dyn.ISTEER]))           # lr / L ~ 0.52: the bicycle's own slip
    out["beta0"] = torch.rad2deg((torch.atan2(S[:, 0, dyn.IVY], S[:, 0, dyn.IVX].abs().clamp_min(0.3)) - b_kin).abs())
    # inside (corner cutting) / outside (running wide) of the plan at 0.2-0.5 s, by the plan's own turn at that instant (yaw
    # rate > 0.2 rad/s; on a straight bit either side counts as both); the worst of the four instants per case
    r_pl = torch.diff(ppsi, dim=1) / cdt
    sg = torch.sign(torch.cat([r_pl[:, :1], r_pl], 1)) * (torch.cat([r_pl[:, :1], r_pl], 1).abs() > 0.2)
    lat_in = torch.where(sg != 0, lat * sg, lat.abs()); lat_out = torch.where(sg != 0, -lat * sg, lat.abs())
    jj = j[1:5]
    out["in_max"] = lat_in[:, jj].clamp_min(0).amax(1); out["out_max"] = lat_out[:, jj].clamp_min(0).amax(1)
    out["alat_abs"] = lat[:, jj].abs().mean(1)
    bk_t = torch.atan(0.52 * torch.tan(S[..., dyn.ISTEER]))
    out["slip_ex"] = torch.rad2deg((torch.atan2(S[..., dyn.IVY], S[..., dyn.IVX].abs().clamp_min(0.3)) - bk_t).abs())[:, 1:].amax(1)
    r_car = S[..., dyn.IR][:, 1:]
    r_plan = torch.diff(ppsi, dim=1) / cdt
    out["yaw_over"] = (r_car - r_plan).abs().amax(1)
    out["yaw_rms"] = (r_car - r_plan).square().mean(1).sqrt()
    sr_car = torch.diff(S[..., dyn.ISTEER], dim=1) / cdt
    sr_plan = torch.diff(pd, dim=1) / cdt
    out["steer_rate_rms"] = (sr_car - sr_plan).square().mean(1).sqrt()
    cr = torch.diff(C[..., 0], dim=1)
    out["steer_flips"] = ((torch.sign(cr[:, 1:]) * torch.sign(cr[:, :-1]) < 0) & (cr[:, 1:].abs() > 1e-3)).float().sum(1) / (n * cdt)
    out["slip"] = torch.rad2deg(torch.atan2(S[..., dyn.IVY], S[..., dyn.IVX].abs().clamp_min(0.3)).abs())[:, 1:].amax(1)
    out["slip_all"] = torch.rad2deg(torch.atan2(S[..., dyn.IVY], S[..., dyn.IVX].abs().clamp_min(0.3)).abs())[:, 1:]
    # plan features (situation bins)
    k_ = torch.tan(pd) / wheelbase
    alat = (pv ** 2 * k_.abs())
    out["f_v0"] = pv[:, 0]
    out["f_alat"] = alat.amax(1)
    out["f_acc"] = (pv[:, -1] - pv[:, 0]) / (n * cdt)
    out["f_swerve"] = (pd[:, ::4].amax(1) > 0.03) & (pd[:, ::4].amin(1) < -0.03)          # turns both ways within the second
    out["f_dsteer"] = (torch.diff(pd[:, ::4], dim=1).abs().amax(1)) / 0.1
    return {k: v.cpu().numpy() for k, v in out.items()}


def wpct(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][np.searchsorted(cw, q / 100 * cw[-1])])


def rec_like(m):
    """plans like the ones policies issue (scan31b2 u600 exact exam3: steering change <= 0.10 rad per 0.1 s at p99, |accel| <= 3.8)"""
    return (m["f_dsteer"] <= 1.2) & (np.abs(m["f_acc"]) <= 4.0)


W_REC = 3.0


def tail(m, sel, rec):
    """(11-24, coordinator: mean RMSE did not predict wall hits) the tuning objective [cm]: p95 of the worst inside offset +
    p95 of the worst outside offset at 0.2-0.5 s, + 2 cm per rad/s of p95 yaw-rate overshoot, + 1 cm per % of cases that spin
    (slip > 15 deg beyond the bicycle's); the recorded policy decisions (`rec`) weigh W_REC, the plan-space bank 1"""
    w = np.where(rec, W_REC, 1.0)[sel]
    return (100 * (wpct(m["in_max"][sel], w, 95) + wpct(m["out_max"][sel], w, 95)) + 2.0 * wpct(m["yaw_over"][sel], w, 95)
            + 100 * float(np.sum(w * (m["slip_ex"][sel] > 15)) / np.sum(w)))


def summary(m, sel):
    """mean and p95 side by side (cm, rad/s, deg)"""
    rms = float(np.sqrt(np.mean(m["pos"][sel, :5] ** 2)))
    return (f"위치 0.1-0.5 s RMS {100 * rms:5.1f} | 옆 평균 {100 * np.mean(m['alat_abs'][sel]):4.1f} | 안쪽 95 % {100 * np.percentile(m['in_max'][sel], 95):5.1f} "
            f"바깥 95 % {100 * np.percentile(m['out_max'][sel], 95):5.1f} | 요넘침 중앙 {np.median(m['yaw_over'][sel]):4.2f} 95 % {np.percentile(m['yaw_over'][sel], 95):4.2f} | "
            f"미끄럼(자전거 초과) 90 % {np.percentile(m['slip_ex'][sel], 90):4.1f}도, 15도 넘음 {100 * np.mean(m['slip_ex'][sel] > 15):4.1f} % | n {int(sel.sum())}")


def objective(m, sel):
    """RMS position error over 0.1-0.5 s, cases `sel`"""
    return float(np.sqrt(np.mean(m["pos"][sel, :5] ** 2)))


def table(m, c, title):
    """per-situation error table (Korean, for the user): rows = speed x lateral demand x swerve / brake / accel"""
    # a start the car cannot hold (after settling it slides > 6 deg beyond the bicycle's own slip): its own row, not in the rest
    bad = m["beta0"] > 6.0
    sel0 = ~c["off"] & ~bad
    v0 = m["f_v0"]; al = m["f_alat"]; acc = m["f_acc"]; sw = m["f_swerve"]; tr = c["trans"]
    # what recorded policy plans look like (scan31b2 u600 exact exam3: steering changes <= 0.10 rad per 0.1 s at p99, |accel| <= 3.8)
    rec = c["rec"] if "rec" in c else rec_like(m)
    out = [f"== {title}: 사례 {int(sel0.sum())}개 (plan 원점 출발), 0.1-0.5 s 위치 오차 RMS {100 * objective(m, sel0):.1f} cm =="]
    out.append(f"{'상황':<34}{'n':>6}{'0.2s 위치':>9}{'0.5s 위치':>9}{'0.5s 옆':>8}{'0.5s 앞뒤':>9}{'1.0s 위치':>9}{'0.5s 속도':>9}"
               f"{'요넘침':>7}{'조향흔들':>8}{'미끄럼max':>9}")
    out.append("(0.5s 옆: + = plan이 도는 쪽 안으로, - = 바깥으로 / 앞뒤: + = plan보다 앞 / 요넘침: 실제 - plan 요 각속도 최대 [rad/s] 중앙값 / "
               "조향흔들: 실제 - plan 조향 속도 RMS [rad/s] 중앙값 / 미끄럼max: 사례마다 최대 미끄럼각의 90 %)")
    rms = lambda a: float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")

    def row(name, s):
        s = s & sel0
        if s.sum() < 20:
            return
        out.append(f"{name:<34}{int(s.sum()):>6}{100 * rms(m['pos'][s, 1]):>8.1f} {100 * rms(m['pos'][s, 4]):>8.1f} {100 * np.mean(m['lat'][s, 4]):>+7.1f} "
                   f"{100 * np.mean(m['lon'][s, 4]):>+8.1f} {100 * rms(m['pos'][s, 9]):>8.1f} {np.mean(m['dv'][s, 4]):>+8.2f} "
                   f"{np.median(m['yaw_over'][s]):>6.2f} {np.median(m['steer_rate_rms'][s]):>7.2f} {np.percentile(m['slip'][s], 90):>8.1f}")
    row("전체", np.ones_like(sel0))
    row("기록된 policy 결정", rec)
    row("plan 공간 (Sobol)", ~rec)
    row("조향 명령이 막 바뀐 뒤(과도 상태)", tr)
    row("정상 상태에서", ~tr)
    for lo, hi in ((1, 3), (3, 5), (5, 7), (7, 10)):
        vs = (v0 >= lo) & (v0 < hi)
        row(f"속도 {lo}-{hi} m/s", vs)
        for nm, s in ((" 곧게(옆가속 < 2)", al < 2), (" 돌기(2-5)", (al >= 2) & (al < 5)), (" 세게 돌기(>= 5)", al >= 5),
                      (" 좌우 바꿈(S·차선)", sw), (" 감속(< -1.5 m/s2)", acc < -1.5), (" 가속(> 1.5 m/s2)", acc > 1.5)):
            row(nm, vs & s)
    out.append(f"(정착 뒤 이미 자전거 모형보다 6도 넘게 미끄러지는 출발 {int((bad & ~c['off']).sum())}개는 뺌: 그 속도·조향을 차가 버티지 못함 -- "
               f"속도 중앙 {np.median(c['v_set'][bad]) if bad.any() else float('nan'):.1f} m/s)")
    so = c["off"] & ~bad
    if so.sum() >= 20:
        out.append(f"(plan 원점에서 어긋나 출발한 사례 {int(so.sum())}개: 0.5 s 위치 RMS {100 * rms(m['pos'][so, 4]):.1f} cm, 1.0 s {100 * rms(m['pos'][so, 9]):.1f} cm)")
    return "\n".join(out)


#: the knobs searched on the bank, with their bounds (the rest of the follower as `NOW`)
KNOBS = {"t_clip_min": (0.3, 2.0), "m_l1": (0.1, 0.9), "q_l1": (-1.0, 0.5), "v_steer_gain_max": (0.8, 1.8),
         "speed_lookahead_for_steer": (0.0, 0.3), "tw_speed_lead": (0.0, 0.45), "delay_comp": (0.0, 0.12)}


def run_opt(name, n, dev, O, n_rec=0, rec_npz=None):
    """analytic first (delay compensation = command delay + half the servo lag; speed lead = speed loop lag + command delay),
    then Nelder-Mead on the tail objective (`tail`) over the even cases, checked on the odd ones; the other followers as
    reference rows; per-knob sensitivity by situation"""
    from scipy.optimize import minimize
    room = Room(n + n_rec, dev)
    c = room.bank(n, n_rec=n_rec, rec_npz=rec_npz)
    st0, ax0, h0 = room.settle(c)
    P = room.P
    n = n + n_rec
    ana = dict(delay_comp=float(P["cmd_delay"][0] + 0.5 * P["servo_tau"][0]),
               tw_speed_lead=float(P["motor_tau"][0] + P["cmd_delay"][0]))
    cn = {k: v.cpu().numpy() for k, v in c.items()}
    ev = {}
    log = []

    def say(s_):
        print(s_, flush=True); log.append(s_)

    def evaluate(over):
        key = json.dumps(over, sort_keys=True)
        if key not in ev:
            p, S, C = room.follow(c, st0, ax0, h0, dict(NOW, **over))
            ev[key] = metrics(p, S, C, room.cdt, room.lim.wheelbase)
        return ev[key]
    m0 = evaluate({})
    ok = ~cn["off"] & (m0["beta0"] <= 6.0)
    idx = np.arange(n)
    tr, va = ok & (idx % 2 == 0), ok & (idx % 2 == 1)
    rl = cn["rec"]
    fast = ok & (m0["f_v0"] >= 7)
    say(f"[opt] 사례 {n} (학습 반 {int(tr.sum())}, 확인 반 {int(va.sum())}, 기록된 plan 같은 것 {int((ok & rl).sum())}); "
        f"(그중 기록된 policy 결정 {int(cn['rec'].sum())}개, 무게 {W_REC:g}); 계산으로 정한 것: {json.dumps({k: round(v, 3) for k, v in ana.items()})}")
    t0 = time.time()

    def show(tag, mm):
        say(f"  {tag:<26} 목표(꼬리) 학습 {tail(mm, tr, cn['rec']):6.1f} 확인 {tail(mm, va, cn['rec']):6.1f} cm")
        say(f"      전체          {summary(mm, ok)}")
        say(f"      plan 공간    {summary(mm, ok & ~rl)}")
        say(f"      기록된 결정  {summary(mm, ok & rl)}")
        say(f"      7-10 m/s      {summary(mm, fast)}")
        say(f"      7-10 m/s 기록된 결정 {summary(mm, fast & rl)}")
        say(f"      3-7 m/s 기록된 결정 {summary(mm, ok & rl & (mm['f_v0'] >= 3) & (mm['f_v0'] < 7))}")
    show("지금 설정", m0)
    say(f"  (따라가기 한 번 {time.time() - t0:.1f} s)")
    show("지금 + 계산값", evaluate(ana))
    base = dict(ana)
    for cat in ({"l1_point": "euclid"}, {"steer_speed": "measured"}, {"v_steer_gain_max": 1.0}, {"lut_interp": "code"}):
        mc = evaluate(dict(base, **cat))
        say(f"  계산값 + {json.dumps(cat)}: 목표 학습 {tail(mc, tr, cn['rec']):.1f} cm")
    names = list(KNOBS)
    lo = np.array([KNOBS[k][0] for k in names]); hi = np.array([KNOBS[k][1] for k in names])
    x0 = np.array([float(dict(NOW, **base)[k]) for k in names])
    calls = [0]

    def f(x):
        x = np.clip(x, lo, hi); calls[0] += 1
        return tail(evaluate(dict(base, **{k: float(v) for k, v in zip(names, x)})), tr, cn['rec'])
    init = np.stack([x0] + [x0 + np.eye(len(names))[i] * 0.15 * (hi - lo)[i] for i in range(len(names))])
    res = minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=int(os.environ.get("ROOM_MAXFEV", 160)),
                                                             initial_simplex=np.clip(init, lo, hi), xatol=1e-3, fatol=0.05))
    best = dict(base, **{k: float(v) for k, v in zip(names, np.clip(res.x, lo, hi))})
    mb = evaluate(best)
    say(f"  Nelder-Mead {calls[0]}번 ({time.time() - t0:.0f} s): " + json.dumps({k: round(v, 3) for k, v in best.items()}))
    show("맞춘 설정", mb)
    for kind in ("time_table", "time_kin", "mpc"):
        try:
            t1 = time.time()
            p, S, C = follow_ref(room, c, st0, ax0, h0, kind)
            show(f"참고: {kind}", metrics(p, S, C, room.cdt, room.lim.wheelbase))
            say(f"      ({time.time() - t1:.0f} s)")
        except Exception as e:                                   # a reference row is a reference: report and go on
            say(f"  참고: {kind} 실패: {type(e).__name__}: {e}")
    for tag, mm in (("지금", m0), ("맞춘", mb)):
        say(table(mm, cn, f"{name} {tag}"))
    # what each knob does where: +10 % of its range from the tuned value -- inside / outside p95 [cm], yaw overshoot p95
    v0 = mb["f_v0"]; al = mb["f_alat"]; acc = mb["f_acc"]; sw = mb["f_swerve"]
    bins = {"1-4 m/s 돌기": ok & (v0 < 4) & (al >= 2), "4-7 m/s 세게": ok & (v0 >= 4) & (v0 < 7) & (al >= 5),
            "7-10 m/s 세게": ok & (v0 >= 7) & (al >= 5), "S·차선 5 m/s+": ok & sw & (v0 >= 5), "감속": ok & (acc < -1.5),
            "가속": ok & (acc > 1.5)}
    p95 = lambda mm, k, s_: 100 * float(np.percentile(mm[k][s_], 95))
    say("손잡이마다 (맞춘 값에서 범위의 +10 %): 안쪽 95 % / 바깥 95 % 변화 [cm] / 요넘침 95 % 변화 [rad/s]")
    say(f"{'':30s}" + "".join(f"{b:>24s}" for b in bins))
    for k in names:
        d = 0.1 * (KNOBS[k][1] - KNOBS[k][0])
        mp_ = evaluate(dict(best, **{k: float(np.clip(best[k] + d, *KNOBS[k]))}))
        cells = [f"{p95(mp_, 'in_max', s_) - p95(mb, 'in_max', s_):+5.1f}/{p95(mp_, 'out_max', s_) - p95(mb, 'out_max', s_):+5.1f}/"
                 f"{np.percentile(mp_['yaw_over'][s_], 95) - np.percentile(mb['yaw_over'][s_], 95):+5.2f}" for s_ in bins.values()]
        say(f"{k + f' +{d:.3g}':30s}" + "".join(f"{x:>24s}" for x in cells))
    json.dump(dict(best=best, ana=ana, log=log), open(os.path.join(O, f"{name}_opt.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    dev = torch.device(os.environ.get("ROOM_DEVICE", "cuda"))
    cmd = sys.argv[1]
    O = os.path.expanduser(os.environ.get("ROOM_OUT", "~/F1tenth/exam_lab/out/maptune_room"))
    os.makedirs(O, exist_ok=True)
    if cmd == "measure":
        name, over = sys.argv[2], json.loads(sys.argv[3])
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
        t0 = time.time()
        room = Room(n, dev)
        c = room.bank(n)
        st0, ax0, h0 = room.settle(c)
        t1 = time.time()
        p, S, C = room.follow(c, st0, ax0, h0, dict(NOW, **over))
        t2 = time.time()
        m = metrics(p, S, C, room.cdt, room.lim.wheelbase)
        cn = {k: v.cpu().numpy() for k, v in c.items()}
        np.savez_compressed(os.path.join(O, f"{name}.npz"), **{f"m_{k}": v for k, v in m.items() if k != "slip_all"},
                            **{f"c_{k}": v for k, v in cn.items()}, setting=json.dumps(dict(NOW, **over)))
        print(f"[room] {n} 사례, 자리 잡기 {t1 - t0:.1f} s, 따라가기 {t2 - t1:.1f} s, 미끄럼 90 % (모든 시각) {np.percentile(m['slip_all'], 90):.1f}도")
        print(table(m, cn, name))
    elif cmd == "opt":
        run_opt(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16384, dev, O,
                n_rec=int(sys.argv[4]) if len(sys.argv) > 4 else 0, rec_npz=sys.argv[5] if len(sys.argv) > 5 else None)
