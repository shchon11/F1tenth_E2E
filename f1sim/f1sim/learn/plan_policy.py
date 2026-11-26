"""The redesigned planner policy (11-07, agreed with the user): what it sees, the networks, and a runner that drives it.

What the actor sees -- only what the car can measure (no map, no localisation, no line):
  * the LiDAR ranges of the last 3 scans, 0.1 s apart (1081 beams each, / range_max, clipped to 1). The sim's per-beam
    wall / box / car labels are NOT given: the real scan has none.
  * its own speed (the VESC's wheel speed), steering angle (what it last commanded), yaw rate (the gyro)
  * its previous plan: the 10 points (x, y) and speeds, in the car's frame now
The critic (training only) sees the same and the simulator's state: `GymEnv.privileged_v2` (opponents, props, the
track around), the shortest distance left to the finish line / a lap, the time left in the episode.

The action: 20 numbers, `plan_steps` (each 0.1 s step's turn and speed, spread over what the car can reach). The policy
outputs the mean of a Gaussian in an unbounded space and the action is its tanh (so the whole output range maps to
distinct plans and the gradient never meets a clamp)."""
import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .. import plan_steps as PS
from .. import dynamics as dyn

N_SCANS = 3
SCAN_DT = 0.1             # [s] between the scans the actor sees
STATE_DIM = 3 + 3 * PS.N_STEP


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = b_mean - self.mean
        tot = self.count + n
        self.mean += d * n / tot
        self.var = (self.var * self.count + b_var * n + d * d * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


class ScanEncoder(nn.Module):
    """(B, N_SCANS, beams) -> (B, out): 1-D convolutions along the beams, the scans as channels"""

    def __init__(self, beams: int, out: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_SCANS, 32, 7, stride=3, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=3, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=3, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.ReLU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, N_SCANS, beams)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n, out), nn.ReLU())

    def forward(self, scans: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(scans))


class Actor(nn.Module):
    def __init__(self, beams: int, hidden: int = 256, std0: float = 0.6):
        super().__init__()
        self.enc = ScanEncoder(beams)
        self.norm = RunningNorm(STATE_DIM)
        self.mlp = nn.Sequential(nn.Linear(256 + STATE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, PS.ACT_DIM))
        nn.init.orthogonal_(self.mlp[-1].weight, 0.01); nn.init.zeros_(self.mlp[-1].bias)
        self.log_std = nn.Parameter(torch.full((PS.ACT_DIM,), math.log(std0)))

    def forward(self, scans: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([self.enc(scans), self.norm(state)], 1)
        return self.mlp(h), self.log_std.expand(scans.shape[0], -1)


class Critic(nn.Module):
    def __init__(self, beams: int, priv_dim: int, hidden: int = 256, state_dim: int = STATE_DIM):
        super().__init__()
        self.enc = ScanEncoder(beams)
        self.norm = RunningNorm(state_dim + priv_dim)
        self.mlp = nn.Sequential(nn.Linear(256 + state_dim + priv_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, scans: torch.Tensor, state: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.enc(scans), self.norm(torch.cat([state, priv], 1))], 1))[:, 0]


def tanh_logp(u: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """log-probability of a = tanh(u), u ~ N(mean, std)"""
    lp = torch.distributions.Normal(mean, log_std.exp()).log_prob(u)
    return (lp - 2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))).sum(-1)


class PlanObs:
    """the actor's and the critic's inputs for the learner rows of `env`, and each row's last plan"""

    def __init__(self, env, lim: PS.Limits, goal=None, rows: Optional[torch.Tensor] = None):
        self.env, self.lim, self.goal = env, lim, goal
        self.lid = env.learner_ids if rows is None else rows       # `rows`: other cars' rows (a policy-driven opponent)
        Bl = int(self.lid.numel())
        self.beams = int(env.sim.lidar.angles.numel())
        self.range = float(env.sim.cfg.lidar.range_max)
        dev = env.device
        self.scans = torch.zeros(Bl, N_SCANS, self.beams, device=dev)
        # the last plan, in the world: its 10 points and speeds (B_l, 10, 2) / (B_l, 10)
        self.plan_xy = torch.zeros(Bl, PS.N_STEP, 2, device=dev)
        self.plan_v = torch.zeros(Bl, PS.N_STEP, device=dev)
        self.reset_rows(torch.arange(Bl, device=dev))

    def _scan(self) -> torch.Tensor:
        return (self.env.last_result.scan[self.lid].float() / self.range).clamp(0.0, 1.0)

    def reset_rows(self, rows: torch.Tensor) -> None:
        """rows (into the learner rows) just started: every scan is the one now, the plan is the car standing where it is"""
        if rows.numel() == 0:
            return
        self.scans[rows] = self._scan()[rows][:, None, :].expand(-1, N_SCANS, -1)
        st = self.env.sim.state[self.lid][rows]
        self.plan_xy[rows] = st[:, None, :2].expand(-1, PS.N_STEP, -1)
        self.plan_v[rows] = 0.0

    def push_scan(self, rows: Optional[torch.Tensor] = None) -> None:
        """once a decision (every SCAN_DT): the newest scan in front (`rows` (B_l,) bool: only those)"""
        new = torch.cat([self._scan()[:, None, :], self.scans[:, :-1]], 1)
        self.scans = new if rows is None else torch.where(rows[:, None, None], new, self.scans)

    def set_plan(self, xy: torch.Tensor, v: torch.Tensor, rows: Optional[torch.Tensor] = None) -> None:
        if rows is None:
            # its own copies: `reset_rows` writes rows in place, and the caller's plan must not change under it (until 11-13
            # a restart zeroed the caller's plan speeds: the tail's speed targets of the decision before an episode's end)
            self.plan_xy, self.plan_v = xy.clone(), v.clone()
        else:
            self.plan_xy = torch.where(rows[:, None, None], xy, self.plan_xy)
            self.plan_v = torch.where(rows[:, None], v, self.plan_v)

    def state(self) -> torch.Tensor:
        """(B_l, STATE_DIM): speed, steering, yaw rate, the last plan in the car's frame"""
        r = self.env.last_result
        lid = self.lid
        v = r.odom[lid, 3].float()
        yr = r.odom[lid, 4].float()
        st = self.env.sim.state[lid]
        steer = st[:, dyn.ISTEER].float()
        c, s_ = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
        dx = self.plan_xy[..., 0] - st[:, None, 0]; dy = self.plan_xy[..., 1] - st[:, None, 1]
        bx, by = dx * c + dy * s_, -dx * s_ + dy * c
        return torch.cat([v[:, None], steer[:, None], yr[:, None], bx, by, self.plan_v], 1)

    def priv(self) -> torch.Tensor:
        """(B_l, P): the simulator's state for the critic"""
        env = self.env
        pv = env.privileged_v2()[self.lid].float()
        # no centerline (11-09, the user: "레이싱라인 ... 쓰지도않는거", "Centerline 도 마찬가지야"): the offset / heading to it,
        # the lap fraction and the progress rate along it are left out; the distance left to the line is the walls-only one below
        keep = torch.ones(pv.shape[1], dtype=torch.bool, device=pv.device); keep[_PRIV_DROP] = False
        pv = pv[:, keep]
        extra = []
        if self.goal is not None:
            tid = env.sim.tid[self.lid]
            d = self.goal.remaining(env.sim.state[self.lid, :2], tid)
            L = self.goal.lap[tid]
            extra.append(torch.where(torch.isfinite(d), d / L, torch.ones_like(d))[:, None])
        t_left = 1.0 - env.ep_step[self.lid].float() / float(env.ecfg.max_steps)
        extra.append(t_left[:, None])
        return torch.cat([pv] + extra, 1)


def plan_world(p: Dict[str, torch.Tensor], st: torch.Tensor):
    """a `plan_steps.roll` plan in the world, from the pose `st` (B, >= 3) it was made at: dense x, y, psi (B, n)"""
    c, s_ = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
    x = st[:, 0:1] + p["x"] * c - p["y"] * s_
    y = st[:, 1:2] + p["x"] * s_ + p["y"] * c
    return x, y, st[:, dyn.IYAW:dyn.IYAW + 1] + p["psi"]


PT_IDX = torch.arange(1, PS.N_STEP + 1) * PS.SUB             # the 10 points among the dense poses
# (11-22, the user: "0.1초가 지났어. 나머지 0.9초짜리는 그대로 유지하면서 새로운 0.1초짜리를 ... 점점 이어나가게") a rolling plan:
# every decision keeps the committed plan's next 0.9 s and the policy adds the 0.1 s after it -- its outputs' last step (turn,
# speed) are the decision; a whole new plan only at a start or when something unseen blocks the committed part
ROLL_IDX = (PS.N_STEP - 1, 2 * PS.N_STEP - 1)


def roll_shift(u_prev: torch.Tensor, u_new: torch.Tensor) -> torch.Tensor:
    """the committed plan's action (B, 20, pre-tanh) one step on, `u_new`'s last step appended: rolled from the committed
    plan's state at its first step's end (`plan_steps.roll` is a recursion over the steps), it is the same 0.9 s and then
    the new 0.1 s"""
    N = PS.N_STEP
    return torch.cat([u_prev[:, 1:N], u_new[:, N - 1:N], u_prev[:, N + 1:], u_new[:, 2 * N - 1:]], 1)


# (11-22, plan_rl --roll-grow) the committed part grown a step at a time: roll29a-e learnt to stand still (a decision moved
# the car 0.9 s later; resumed from a driving policy it drove 10 m and stopped in every exam session), so the commitment
# starts at 0 steps (every decision a whole new plan, the policy it resumes) and grows to 9 (the rolling plan above)
def roll_shift_c(u_prev: torch.Tensor, u_new: torch.Tensor, c: int) -> torch.Tensor:
    """the committed plan's next `c` steps kept (its steps 2..c+1), `u_new`'s steps c+1..10 after them (pre-tanh): c = 9 is
    `roll_shift`, c = 0 `u_new`"""
    N = PS.N_STEP
    return torch.cat([u_prev[:, 1:1 + c], u_new[:, c:N], u_prev[:, N + 1:N + 1 + c], u_new[:, N + c:]], 1)


def commit_masks(c: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """--roll-grow with `c` steps carried over: the outputs a whole new plan commits (steps 1..c+1, turn and speed) and the
    ones a rolling decision commits (step c+1); the steps after them are a forecast, drawn again next decision"""
    N = PS.N_STEP
    m_all = torch.zeros(PS.ACT_DIM, dtype=torch.bool, device=device); m_all[:c + 1] = True; m_all[N:N + c + 1] = True
    m_new = torch.zeros(PS.ACT_DIM, dtype=torch.bool, device=device); m_new[c] = True; m_new[N + c] = True
    return m_all, m_new


def tanh_logp_m(u: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """`tanh_logp` over the outputs in the mask `m` ((B, ACT_DIM) or (ACT_DIM,)) only"""
    lp = torch.distributions.Normal(mean, log_std.exp()).log_prob(u)
    return ((lp - 2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))) * m).sum(-1)

def _priv_drop() -> list:
    """the centerline columns of `GymEnv.privileged_v2` (the columns after its keys -- props -- are kept)"""
    from ..gym_env import PRIV_V2_KEYS
    return [i for i, k in enumerate(PRIV_V2_KEYS) if k in ("lateral", "heading_err", "lap_frac", "progress_rate")]


_PRIV_DROP = _priv_drop()


def plan_segments(p: Dict[str, torch.Tensor], lim: PS.Limits) -> Tuple[torch.Tensor, torch.Tensor]:
    """a plan as MAP's time-waypoint follower reads one (`plan_tw`, 10 segments of 0.1 s): each step's curvature at its
    mean steering and its mean speed"""
    d = p["delta"]; v = p["v"]
    idx = torch.arange(0, PS.N_STEP + 1, device=d.device) * PS.SUB
    dk, vk = d[:, idx], v[:, idx]
    d_mid = 0.5 * (dk[:, 1:] + dk[:, :-1]); v_mid = 0.5 * (vk[:, 1:] + vk[:, :-1])
    return torch.tan(d_mid) / lim.wheelbase, v_mid


def load_actor(ck: dict, device) -> Actor:
    actor = Actor(int(ck["beams"]), std0=float(ck["args"].get("std0", 0.6))).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


class PlanRunner:
    """Drives a `plan_rl` checkpoint in an env: called once per control step (the console's actor runner's place). Each
    learner row plans on its own decision steps (`GymEnv._dec_cnt` % SUB == 0, the env built with decision_every SUB, so
    a row that respawned mid-way keeps its own phase), from the actor's mean. mode "map": MAP follows the plan
    (`GymEnv.ext_plan_kv`, the env built with learner_plan "tw" and the tuned follower, which holds it in the world
    between decisions) -- the real car's way; mode "ideal": the car drives the plan exactly (`GymEnv.enable_ideal`) --
    the training's way. Returns the env's (B, act_dim) action (zeros: the plan goes in through the env)."""

    def __init__(self, env, ck: dict, device, mode: str = "map"):
        from .. import map_controller as M
        self.env, self.device, self.mode = env, device, mode
        self.kind = "plan"
        self.lid = env.learner_ids
        self.v3 = ck.get("format") == "plan_steps3"
        self.v2 = ck.get("format") == "plan_steps2" or self.v3
        _mg = (ck.get("args") or {}).get("plan_lat_margin")      # the limit the checkpoint was trained with (11-10)
        self.lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), device,
                                          margin=None if _mg is None else float(_mg))
        if self.v3:
            self.actor = load_actor3(ck, device)
            self.obs = PlanObs3(env, self.lim, None)
            self.act3, self.turn3 = plan3_models(env, self.lim, device)
        elif self.v2:
            # the console's env is not built per checkpoint: move its scanner to where this one's training had it
            mx = lidar_mount_of(ck) or float(type(env.cfg.lidar)().mount_x)
            if abs(float(env.cfg.lidar.mount_x) - mx) > 1e-6:
                env.cfg.lidar.mount_x = mx
                env.sim.params.set_all("mount_x", mx)
                print(f"[plan] LiDAR {mx:.3f} m ahead of the CoG (the checkpoint's)", flush=True)
            self.actor = load_actor2(ck, device, check_mount=False)
            self.obs = PlanObs2(env, self.lim, None)
            if getattr(self.actor, "pace_levels", 0):            # its state inputs exist before anything reads the state
                self.obs.pace_on(self.actor.pace_levels, self.actor.pace_hold)
            if self.actor.branches == 2:
                self.obs.prev_branch = torch.zeros(int(self.lid.numel()), device=device)
        else:
            self.actor = load_actor(ck, device)
            self.obs = PlanObs(env, self.lim, None)
        self.v_hat = None
        Bl = int(self.lid.numel())
        # (11-22) a rolling-plan checkpoint (plan_rl --plan-roll) is driven the way it was trained: each decision the committed
        # plan keeps its next 0.9 s and the policy's last step is appended; a whole new plan at a start, after a contact, or
        # when something out of view when its part was added now blocks the committed path (`plan_rl.roll_blocked`)
        _ca = ck.get("args") or {}
        self.roll = bool(self.v2 and not self.v3 and (_ca.get("plan_roll", False) or _ca.get("roll_grow", False)))
        # --roll-grow: the steps kept when this checkpoint was saved (--plan-roll: 9)
        self.r_c = int(ck.get("roll_c", PS.N_STEP - 1)) if _ca.get("roll_grow", False) else PS.N_STEP - 1
        self.r_any = bool(_ca.get("roll_replan_seen", False))      # also re-plan for what is in view / known now
        if self.roll:
            from .plan_reward import PlanReward
            self.r_pr = PlanReward(env, float((ck.get("args") or {}).get("gamma", 0.995)), device)
            self.r_u = torch.zeros(Bl, PS.ACT_DIM, device=device); self.r_p = None
            self.r_full = torch.ones(Bl, dtype=torch.bool, device=device)
            self.r_w = torch.zeros(Bl, PS.N_STEP, 3, device=device)
            self.r_t = torch.zeros(Bl, PS.N_STEP, dtype=torch.long, device=device)
            self.r_dec = torch.zeros(Bl, dtype=torch.long, device=device)            # each row's own decision count
            self.r_seenb = None; self.r_scanb = torch.zeros(11, Bl, 3, device=device)
            self.r_seen = self.r_ds = self.r_pos = None
            self.r_cnt = env.sim.contact_counts[self.lid, 0:3].clone()               # contact onsets at the last decision
            self.r_mx = float(env.cfg.lidar.mount_x); self.r_rmax = float(env.cfg.lidar.range_max)
        self.cmd_d = torch.zeros(Bl, PS.N_STEP * PS.SUB, device=device)
        self.cmd_v = torch.zeros(Bl, PS.N_STEP * PS.SUB, device=device)
        self.kk = torch.zeros(env.B, PS.N_STEP, device=device)
        self.vv = torch.zeros(env.B, PS.N_STEP, device=device)
        self.first = torch.ones(Bl, dtype=torch.bool, device=device)
        self.last_ep = env.ep_step[self.lid].clone()
        self.session = None
        self.last_mean = None
        # (11-24) a `pred_opp` policy's prediction of the other car, per learner row: the 10 points in the world (x, y
        # interleaved) and P(visible) last -- the console draws it (`pred_row`); None for a policy without the head
        self.pred_w = (torch.zeros(Bl, PRED_OUT, device=device)
                       if getattr(getattr(self, "actor", None), "pred_opp", False) else None)
        self._row_of = {int(r): k for k, r in enumerate(self.lid.tolist())}
        # (11-26) a state decoder (`state_decoder.StateDecoder`, `attach_decoder`): read-only, what the policy's features and
        # memory hold, decoded at each decision for every learner row -- the console draws it (`dec_out`, `dec_pose`)
        self.dec = None
        self.dec_set = "feat+mem"
        self.dec_out = None                     # {name: (B_l, ...) tensor} raw decoder outputs of each row's last decision
        self.dec_pose = None                    # (B_l, 3) the true pose each row's last decision was made at
        # mode "track" (11-09, the user: "얼추 여기에 맞는 제어기 좀 만들어놔"): full physics, `plan_follow.PlanFollower` follows
        # the plan in time (actuator lags inverted), its command goes in through the env's external-command path
        self.follower = None
        if mode == "track":
            from ..plan_follow import PlanFollower
            # diagnosis knobs (11-10, the user: the car wobbles more and more in the console): F1SIM_TRACK_GAINS="k_v,k_lat,k_head",
            # F1SIM_TRACK_SERVO_TAU (the servo-lag lead; 0 = none), F1SIM_TRACK_NOLUT=1 (the plan's own steering, no table)
            _g = [float(v) for v in os.environ.get("F1SIM_TRACK_GAINS", "2.0,1.0,1.0").split(",")]
            _kw = {"servo_tau": float(os.environ["F1SIM_TRACK_SERVO_TAU"])} if os.environ.get("F1SIM_TRACK_SERVO_TAU") else {}
            self.follower = PlanFollower(Bl, device, s_max=float(env.s_max),
                                         lut=None if os.environ.get("F1SIM_TRACK_NOLUT") == "1" else env.tracker.map.lut,
                                         grip=M.GRIP_LEVELS.index(1.0), wheelbase=self.lim.wheelbase,
                                         k_v=_g[0], k_lat=_g[1], k_head=_g[2],
                                         ff_window=float(os.environ.get("F1SIM_TRACK_FFWIN", "0")),
                                         slow_tau=float(os.environ.get("F1SIM_TRACK_SLOWTAU", "0")),
                                         k_yaw=float(os.environ.get("F1SIM_TRACK_KYAW", "0")), **_kw)
            if os.environ.get("F1SIM_TRACK_MPC") == "1":
                # 11-10 (the user: "제어기좀 똑바로 만들어라"): the model-predictive follower (`plan_mpc.PlanMPC`); weights
                # F1SIM_MPC_W="w_lat,w_head,w_rate,w_yaw", horizon F1SIM_MPC_H, device F1SIM_MPC_DEV (cpu for one car)
                from ..plan_mpc import PlanMPC
                self.follower = PlanMPC(Bl, os.environ.get("F1SIM_MPC_DEV", str(device)), env.tracker.nominal,
                                        s_max=float(env.s_max), k_v=_g[0], horizon=int(os.environ.get("F1SIM_MPC_H", "20")),
                                        sub=int(os.environ.get("F1SIM_MPC_SUB", "4")), every=int(os.environ.get("F1SIM_MPC_EVERY", "1")),
                                        lut=env.tracker.map.lut, grip=M.GRIP_LEVELS.index(1.0),
                                        **dict(zip(("w_lat", "w_head", "w_rate", "w_yaw"),
                                                   [float(v) for v in os.environ.get("F1SIM_MPC_W", "400,100,2500,0").split(",")])))
            env.claim_external(self.lid)

    def attach_decoder(self, path: str, which: str = "feat+mem") -> str:
        """load a checkpoint's state decoder (`<checkpoint>_dec.npz`); raises ValueError when it does not fit this policy"""
        from .state_decoder import StateDecoder
        if not self.v2 or self.v3:
            raise ValueError("state decoders read v2 plan policies only")
        dec = StateDecoder(path, self.device)
        if which not in dec.nets:
            raise ValueError(f"no '{which}' decoder in {path}")
        sd = int(self.actor.norm.mean.shape[0])
        if dec.feat_dim != 256 + sd or dec.mem_dim != int(getattr(self.actor, "memory", 0) or 0):
            raise ValueError(f"decoder for feat {dec.feat_dim} / mem {dec.mem_dim}, this policy has {256 + sd} / "
                             f"{int(getattr(self.actor, 'memory', 0) or 0)}")
        self.dec, self.dec_set = dec, which
        return dec.checkpoint

    @torch.no_grad()
    def _decode(self, due: torch.Tensor, feat: torch.Tensor, st: torch.Tensor) -> None:
        """the decoder on this decision's features and the memory after it, kept for the rows deciding now"""
        mem = getattr(self.obs, "mem_h", None)
        if mem is None:
            return
        o = self.dec.raw(feat.float(), mem.float(), self.dec_set)
        if self.dec_out is None:
            self.dec_out = {k: v.clone() for k, v in o.items()}
            self.dec_pose = st[:, :3].float().clone()
        else:
            for k, v in o.items():
                self.dec_out[k] = torch.where(due.view(-1, *([1] * (v.dim() - 1))), v, self.dec_out[k])
            self.dec_pose = torch.where(due[:, None], st[:, :3].float(), self.dec_pose)

    def pred_row(self, env_row: int) -> int:
        """the learner index of env row `env_row` (-1: not a learner car) -- the console's frame reads `pred_w[k]`"""
        return self._row_of.get(int(env_row), -1)

    def _roll_check(self, due: torch.Tensor) -> None:
        """--plan-roll, deciding rows: the known boxes and the scanner now; a contact since the row's last decision or the
        committed path blocked by something unseen when its part was added -> a whole new plan"""
        from .plan_rl import box_along, boxes_in_view, roll_blocked
        env, lid = self.env, self.lid
        Bl = int(lid.numel()); dev = self.device
        cnt = env.sim.contact_counts[lid, 0:3]
        hit = (cnt - self.r_cnt).clamp_min(0).sum(1) > 0
        # not where the last plan put it (held by a wall, pushed; with a follower other than "ideal": its tracking error)
        off = (env.sim.state[lid, :2].float() - self.r_w[:, 0, :2]).norm(dim=-1) > 0.01 if self.r_p is not None else hit
        self.r_full |= due & (hit | off)
        self.r_cnt = torch.where(due[:, None], cnt, self.r_cnt)
        if getattr(env, "procedural", None) is None:
            return
        ds, live, pos = box_along(env, self.r_pr.goal, lid)
        if self.r_seen is None:
            self.r_seen = torch.zeros_like(live); self.r_ds = torch.full_like(ds, float("nan")); self.r_pos = pos.clone()
        moved = ((pos - self.r_pos).abs().sum(-1) > 0.05) | ~live
        passed = live & ~moved & (self.r_ds > 0.0) & (self.r_ds <= 3.0) & (ds <= 0.0) & (ds > -3.0)
        view = boxes_in_view(env, lid, pos, live, self.r_mx, self.r_rmax)
        seen = (self.r_seen & ~moved) | passed | view
        d2 = due[:, None]
        self.r_seen = torch.where(d2, seen, self.r_seen)
        self.r_ds = torch.where(d2, torch.where(live, ds, torch.full_like(ds, float("nan"))), self.r_ds)
        self.r_pos = torch.where(d2[..., None], pos, self.r_pos)
        sn = env.sim.state[lid, :3].float()
        sc = torch.stack([sn[:, 0] + self.r_mx * torch.cos(sn[:, 2]), sn[:, 1] + self.r_mx * torch.sin(sn[:, 2]), sn[:, 2]], 1)
        ar = torch.arange(Bl, device=dev)
        k = self.r_dec % 11
        if self.r_seenb is None:
            self.r_seenb = torch.zeros(11, *self.r_seen.shape, dtype=torch.bool, device=dev)
        self.r_scanb[k, ar] = torch.where(due[:, None], sc, self.r_scanb[k, ar])
        self.r_seenb[k, ar] = torch.where(d2, self.r_seen, self.r_seenb[k, ar])
        chk = due & ~self.r_full
        if self.r_p is not None and bool(chk.any()):
            K = self.r_c + 1                                      # the committed points only
            ix = self.r_t[:, :K] % 11
            kn = ((self.r_seen & ~moved)[:, None, :].expand(-1, K, -1) if self.r_any
                  else self.r_seenb[ix, ar[:, None]] & ~moved[:, None, :])
            bw, bb = roll_blocked(env, lid, self.r_pr, self.r_w[:, :K], kn, self.r_scanb[ix, ar[:, None]], view, self.r_mx,
                                  self.r_rmax, any_seen=self.r_any)
            self.r_full |= chk & (bw | bb)

    def _roll_plan(self, due: torch.Tensor, u: torch.Tensor, v_hat: torch.Tensor):
        """--plan-roll, deciding rows: the committed 0.9 s rolled on from its own state, the policy's last step appended (a
        whole new plan: from the speed estimate and the last command)"""
        env, lid = self.env, self.lid
        u_x = u if self.r_p is None else torch.where(self.r_full[:, None], u, roll_shift_c(self.r_u, u, self.r_c))
        v_s = v_hat.detach().clamp(-PS.V_REV, PS.V_FWD); d_s = self.obs.cmd_d
        if self.r_p is not None:
            v_s = torch.where(self.r_full, v_s, self.r_p["v"][:, PS.SUB]); d_s = torch.where(self.r_full, d_s, self.r_p["delta"][:, PS.SUB])
        p = PS.roll(torch.tanh(u_x), v_s, d_s, self.lim)
        t_now = self.r_dec[:, None]
        t_new = torch.where(self.r_full[:, None], t_now.expand_as(self.r_t),
                            torch.cat([self.r_t[:, 1:self.r_c + 1], t_now.expand(-1, PS.N_STEP - self.r_c)], 1))
        d2 = due[:, None]
        self.r_t = torch.where(d2, t_new, self.r_t)
        self.r_u = torch.where(d2, u_x, self.r_u)
        self.r_p = p if self.r_p is None else {k_: (torch.where(due.view(-1, *([1] * (p[k_].dim() - 1))), p[k_], self.r_p[k_])
                                                    if torch.is_tensor(p[k_]) and p[k_].dim() and p[k_].shape[0] == due.shape[0]
                                                    else p[k_]) for k_ in p}
        xr, yr, pr_ = plan_world(p, env.sim.state[lid]); ir = PT_IDX.to(self.device)
        self.r_w = torch.where(d2[..., None], torch.stack([xr[:, ir], yr[:, ir], pr_[:, ir]], -1), self.r_w)
        self.r_full = self.r_full & ~due if self.r_c > 0 else torch.ones_like(self.r_full)   # 0 kept: always a whole plan
        self.r_dec = self.r_dec + due.long()
        return p

    @torch.no_grad()
    def __call__(self, scan=None, ego=None):
        env, lid = self.env, self.lid
        ep = env.ep_step[lid]
        back = ep < self.last_ep                                  # an episode restarted (a reset, the console's 리셋)
        if bool(back.any()):
            self.obs.reset_rows(torch.nonzero(back)[:, 0])
            self.first |= back
            if self.roll:
                self.r_full |= back
            if hasattr(self.follower, "reset_rows"):
                self.follower.reset_rows(torch.nonzero(back)[:, 0])
        self.last_ep = ep.clone()
        j = env._dec_cnt[lid] % PS.SUB
        due = j == 0
        if bool(due.any()):
            self.obs.push_scan(due & ~self.first)                 # a fresh start already holds its scan three times
            self.first &= ~due
            st = env.sim.state[lid].clone()
            if self.v3:
                _, mean, _, v_hat, p = plan_v3(self.actor, self.obs, self.lim, self.act3, self.turn3)
                self.v_hat = v_hat
            elif self.v2:
                h_old = getattr(self.obs, "mem_h", None)
                h_old = h_old.clone() if h_old is not None else None
                b_old = self.obs.prev_branch.clone() if self.obs.prev_branch is not None else None
                pc_old = ((self.obs.pace_idx.clone(), self.obs.pace_left.clone())
                          if getattr(self.obs, "pace_idx", None) is not None else None)
                if self.roll:
                    self._roll_check(due)
                # what the memory reads this decision (Actor2.features), for the decoder only
                f_dec = (torch.cat([self.actor.enc(self.obs.scans), self.actor.norm(self.obs.state())], 1)
                         if self.dec is not None else None)
                u_, mean, _, v_hat, p = plan_v2(self.actor, self.obs, self.lim)
                if self.pred_w is not None and self.actor.pred_out is not None:
                    # the other car's predicted next second, from the pose this decision is made at, in the world
                    pts, pv = pred_points(self.actor.pred_out.float())
                    c_, s_ = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
                    wx = st[:, 0:1] + pts[..., 0] * c_ - pts[..., 1] * s_
                    wy = st[:, 1:2] + pts[..., 0] * s_ + pts[..., 1] * c_
                    new = torch.cat([torch.stack([wx, wy], -1).reshape(-1, 2 * PRED_N), pv[:, None]], 1)
                    self.pred_w = torch.where(due[:, None], new.to(self.pred_w.dtype), self.pred_w)
                if self.roll:
                    p = self._roll_plan(due, u_, v_hat)
                if h_old is not None and getattr(self.actor, "memory", 0):
                    # a memory steps once per decision: rows not deciding now keep theirs
                    self.obs.mem_h = torch.where(due[:, None], self.obs.mem_h, h_old)
                if f_dec is not None:
                    self._decode(due, f_dec, st)
                if b_old is not None:
                    self.obs.prev_branch = torch.where(due, self.obs.prev_branch, b_old)
                if pc_old is not None:                            # the pace level: only deciding rows move on
                    self.obs.pace_idx = torch.where(due, self.obs.pace_idx, pc_old[0])
                    self.obs.pace_left = torch.where(due, self.obs.pace_left, pc_old[1])
                self.v_hat = v_hat
            else:
                mean, _ = self.actor(self.obs.scans, self.obs.state())
                p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], self.lim)
            self.last_mean = mean
            x, y, psi = plan_world(p, st)
            if self.follower is not None:
                # v3 too follows the plan's poses in time (11-09: its own commands sent as they are, plus a small feedback,
                # hit the walls 5 / min -- the car is off the plan's model 37 cm in 1 s); its servo angle as the kinematic
                # steering of the same turn, which the follower turns back into the car's through MAP's table
                d_kin = self.turn3.eff(p["delta"], p["v"]) if self.v3 else p["delta"]
                self.follower.set_plan(due, x[due], y[due], psi[due], p["v"][due], d_kin[due])
            idx = PT_IDX.to(self.device)
            self.obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"], due)
            if self.v2:
                self.obs.set_plan2(p, due)
            self.cmd_d = torch.where(due[:, None], p["cmd_d"], self.cmd_d)
            self.cmd_v = torch.where(due[:, None], p["cmd_v"], self.cmd_v)
            # MAP follows these segments; in "ideal" its command is not used, but the viewers draw the plan from them
            # (11-08: without, the console / videos showed a stale straight line as the policy's plan)
            k_, v_ = plan_segments3(p, self.lim, self.turn3) if self.v3 else plan_segments(p, self.lim)
            rows = lid[due]
            self.kk[rows] = k_[due]; self.vv[rows] = v_[due]
            env.ext_plan_kv = (self.kk, self.vv)
            if self.mode == "ideal":
                env.sim.ideal_vcap[lid[due]] = 1e3
        if self.follower is not None:
            from ..plan_follow import to_car
            st_now = env.sim.state[lid]
            env.ext_cmd[lid] = to_car(env, lid, self.follower.command(st_now))
            # the plan for the viewers (x, y, heading, speed in the car's frame now, `GymEnv.ext_ref`): the env does not run
            # MAP for a car on its own follower, and MAP's reference is what the console drew the plan from (11-09)
            f = self.follower
            c, s_ = torch.cos(st_now[:, 2:3]), torch.sin(st_now[:, 2:3])
            _d = st_now.device                                      # the follower may keep its plan elsewhere (an MPC on the cpu)
            dx, dy = f.X.to(_d) - st_now[:, 0:1], f.Y.to(_d) - st_now[:, 1:2]
            ref = torch.stack([dx * c + dy * s_, -dx * s_ + dy * c, f.PSI.to(_d) - st_now[:, 2:3], f.V.to(_d)], -1)
            if env.ext_ref is None or env.ext_ref.shape[1:] != ref.shape[1:]:
                env.ext_ref = torch.zeros(env.B, *ref.shape[1:], device=self.device)
            env.ext_ref[lid] = ref
        if self.mode == "ideal":
            jj = j[:, None]
            env.ideal_cmd[lid, 0] = self.cmd_d.gather(1, jj)[:, 0]
            env.ideal_cmd[lid, 1] = self.cmd_v.gather(1, jj)[:, 0]
        return torch.zeros(env.B, env.act_dim, device=self.device)


# ------------------------------------------------------------------------------------------------ v2 (11-08)
# The user: "Imu랑 모터에 주는 전류값정도밖에 모르잖슴", "추정하라 하는게 낫지않냐 원시센서값만 주고" (raw sensors only, the policy
# estimates its own speed), "2" (the plan starts from that estimate). What v2's actor sees:
#   * the LiDAR ranges of the last 3 scans (as v1)
#   * the IMU, raw: gyro xyz [rad/s], accelerometer xyz [m/s^2] (the last control step's mean)
#   * what it last commanded: the steering and the speed of its last plan 0.1 s in
#   * its last plan, in the frame of the pose that plan said the car would be in now: points 2..10 and the 10 speeds
#     -- no measurement needed (v1 moved it with the TRUE pose)
# No wheel speed, no true steering angle, no pose. The plan starts from the actor's own speed estimate (a head trained on
# the simulator's true speed, training only) and the last steering command.
STATE_DIM2 = 6 + 2 + 2 * (PS.N_STEP - 1) + PS.N_STEP
FORMAT2 = "plan_steps2"

# (11-24, the race phase agreed with the user) where the nearest other car will be: 10 points 0.1 .. 1.0 s ahead, in the
# car's frame at the decision, [m] / PRED_SCALE, and a 'visible' logit -- from the policy's own inputs only
PRED_N = PS.N_STEP
PRED_SCALE = 5.0
PRED_OUT = 2 * PRED_N + 1                 # x0, y0, ..., x9, y9, visible
PRED_IN = 1 + 2 * PRED_N                  # what the trunk reads: P(visible), then the points x P(visible)

# (11-25) `pred_smooth` (plan_rl --pred-smooth; the user watched the live view: "기이하게 예측되면 안되잖어"). Each of the 10
# points was regressed on its own and the path zig-zagged -- race32c u850 on exam4, decisions with the car there: the mean
# |turn| between successive predicted segments 41.5 deg a step, the realised paths' 6.4 deg. The smooth head outputs the
# other car's pose now (x0, y0, heading0, in our frame at the decision) and its speed and curvature over the next second,
# each a cubic in time (4 Legendre coefficients), and the 10 points are that path integrated -- the same output format as
# before, so everything that reads the prediction is unchanged. Why a cubic and not 10 free steps: fitted to the same
# record's realised 1 s paths, 4 + 4 coefficients miss them by 0.8 cm median / 2.7 cm p90 at 1 s (10 + 10: 0.8 / 2.2 cm),
# while free steps could still alternate the curvature's sign every 0.1 s (a zig-zag of up to 2 v k_max dt a step)
PRED_NB = 4                               # Legendre coefficients of the speed and of the curvature over the second
PRED_KIN_OUT = 3 + 2 * PRED_NB + 1        # x0 / PRED_SCALE, y0 / PRED_SCALE, heading0 / pi, speed's, curvature's, visible
PRED_V_UNIT = 2.0                         # [m/s] the speed is this x softplus(the speed's cubic)
PRED_V0 = 4.0                             # [m/s] a fresh head's speed (the realised paths' median 3.7 m/s)
PRED_W_POSE = 0.2                         # the loss on the pose now: Huber on (x0, y0) / PRED_SCALE, this x the points' mean
PRED_W_HEAD = 0.05                        # and (1 - cos) of heading0 against the other car's yaw

# (11-25) `pred_clear` (plan_rl --pred-clear, agreed with the user): the trunk also reads, for t_k = 0.1 k s (k = 1..10), the
# clearance between where its PREVIOUS plan puts the car at t_k and where the head predicts the other car(s) at t_k -- the
# plan-car cost's 3-disc geometry (`plan_car_cost.clearance`, its interface takes any number of predicted things), the
# minimum over them, `PC_FAR` when nothing predicted is there (P(there) < PC_PMIN) or there is no previous plan (a start),
# clipped to [PC_LO, PC_HI] m and / PC_SCALE. Why: the policy detoured around where the other car IS, decision after
# decision (an endless chase), and the predicted path it was given as 20 raw coordinates moved its plan 0.5 cm for a 0.5 m
# shift (race33b u875): "does my plan meet it" was never read out of them. Detached (computed from the head's output and the
# previous plan, nothing flows back), its 10 trunk columns last and zero at the start -- a fork drives as its parent
PC_N = PS.N_STEP
PC_FAR, PC_LO, PC_HI, PC_SCALE = 3.0, -1.0, 3.0, 3.0     # [m]
PC_PMIN = 0.5                                             # a predicted car counts from this P(there) on


def pred_basis(n_b: int = PRED_NB, n: int = PRED_N) -> torch.Tensor:
    """(n_b, n): Legendre polynomials P0, P1, .. at the middle of each 0.1 s step, the second mapped to [-1, 1]"""
    t = (torch.arange(n, dtype=torch.float32) + 0.5) / n * 2.0 - 1.0
    P = [torch.ones_like(t), t]
    for j in range(2, n_b):
        P.append(((2 * j - 1) * t * P[-1] - (j - 1) * P[-2]) / j)
    return torch.stack(P[:n_b], 0)


def unicycle(x0: torch.Tensor, y0: torch.Tensor, th0: torch.Tensor, v: torch.Tensor, k: torch.Tensor,
             dt: float = PS.DT) -> torch.Tensor:
    """(n,) x3 the start pose, (n, N) speeds [m/s] >= 0 and curvatures [1/m] per step -> (n, N, 2) the positions at the end
    of each step. A step is an exact arc: its heading turns v k dt, its chord (length v dt sinc) points along the step's
    middle heading"""
    phi = v * k * dt
    th_mid = th0[:, None] + torch.cumsum(phi, 1) - 0.5 * phi
    ds = v * dt * torch.sinc(phi / (2.0 * math.pi))               # torch.sinc(x) = sin(pi x) / (pi x)
    return torch.stack([x0[:, None] + torch.cumsum(ds * torch.cos(th_mid), 1),
                        y0[:, None] + torch.cumsum(ds * torch.sin(th_mid), 1)], -1)


class OppPredictor(nn.Module):
    """`Actor2.pred` (plan_rl --pred-opp): the nearest other car's next 1 s from what the policy itself reads -- the LiDAR
    stack (through an encoder of its own, started as a copy of the policy's), the normalised state and the memory's state
    (both detached). Taught by a supervised loss with its own optimizer; nothing of the policy's loss reaches it, and the
    policy reads its answer detached (`Actor2.pred_feed`).
    `smooth` (11-25, see PRED_NB): the last layer gives the pose now and the cubics, `forward` returns the integrated points
    in the old format. Speed = PRED_V_UNIT softplus(.) >= 0; curvature = k_max(speed) tanh(.), k_max the car's own limit
    `kin` = (speeds (G,) [m/s] from 0, evenly spaced; the curvature held there (G,) [1/m]) -- `plan_steps.Limits` of the
    nominal car at full grip: the smaller of the steering lock's tan(s_max) / wheelbase and the tyres' a_lat(v) / v^2.
    Kept in the checkpoint (`kin_v`, `kin_k`)."""

    def __init__(self, beams: int, state_dim: int, memory: int, hidden: int = 256, smooth: bool = False, kin=None):
        super().__init__()
        self.smooth = bool(smooth)
        self.enc = ScanEncoder(beams)
        self.mlp = nn.Sequential(nn.Linear(256 + state_dim + memory, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, PRED_KIN_OUT if self.smooth else PRED_OUT))
        # (11-25) the smooth head's heading at each predicted point (n, PRED_N) [rad, our frame at the decision], from its last
        # `forward` (detached; `pred_clear` places the predicted car's footprint with it). None for the old head
        self.psi_out = None
        if self.smooth:
            if kin is None:
                raise ValueError("pred_smooth needs the car's curvature limit: kin = (speeds (G,) [m/s], curvature (G,) [1/m])")
            v_g, k_g = (torch.as_tensor(x).detach().float().cpu().clone() for x in kin)
            if v_g.shape != k_g.shape or v_g.numel() < 2 or float(v_g[0]) != 0.0 or \
                    float((v_g.diff() - (v_g[1] - v_g[0])).abs().max()) > 1e-4:
                raise ValueError("kin: speeds from 0, evenly spaced, one curvature each")
            self.register_buffer("kin_v", v_g)
            self.register_buffer("kin_k", k_g)
            self.register_buffer("basis", pred_basis(), persistent=False)
            self.kin_reset()

    def kin_reset(self) -> None:
        """the pose / speed / curvature outputs afresh (the 'visible' row untouched): weights x 0.01, so a fresh head says a
        car here, heading as we do, PRED_V0 straight on, whatever it reads, and learns from there"""
        last = self.mlp[-1]
        with torch.no_grad():
            w = torch.empty(PRED_KIN_OUT - 1, last.in_features)
            nn.init.orthogonal_(w, 0.01)
            last.weight[:-1] = w.to(last.weight)
            last.bias[:-1] = 0.0
            last.bias[3] = math.log(math.expm1(PRED_V0 / PRED_V_UNIT))   # the speed's constant term (P0 = 1)

    def kappa_max(self, v: torch.Tensor) -> torch.Tensor:
        """the curvature the car holds at speed v >= 0 (linear in the table; above its top speed the top's lateral
        acceleration: k_top (v_top / v)^2)"""
        n = self.kin_v.numel()
        g = (v / (self.kin_v[1] - self.kin_v[0])).clamp(0.0, n - 1 - 1e-3)
        i0 = g.floor().long()
        w = g - i0.to(g.dtype)
        k = self.kin_k[i0] * (1.0 - w) + self.kin_k[i0 + 1] * w
        v_top = self.kin_v[-1]
        return torch.where(v > v_top, self.kin_k[-1] * (v_top / v.clamp_min(1e-6)) ** 2, k)

    def kin_vk(self, o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """the last layer's output (n, PRED_KIN_OUT) -> each step's speed [m/s] and curvature [1/m], (n, PRED_N) each"""
        nb = PRED_NB
        v = PRED_V_UNIT * nn.functional.softplus(o[:, 3:3 + nb].float() @ self.basis)
        return v, self.kappa_max(v) * torch.tanh(o[:, 3 + nb:3 + 2 * nb].float() @ self.basis)

    def kin_path(self, o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """the last layer's output (n, PRED_KIN_OUT) -> (points (n, PRED_N, 2) [m], pose now (n, 3): x0, y0 [m], heading0)"""
        o = o.float()
        x0, y0, th0 = o[:, 0] * PRED_SCALE, o[:, 1] * PRED_SCALE, o[:, 2] * math.pi
        v, k = self.kin_vk(o)
        return unicycle(x0, y0, th0, v, k), torch.stack([x0, y0, th0], 1)

    def kin_heading(self, o: torch.Tensor) -> torch.Tensor:
        """the last layer's output (n, PRED_KIN_OUT) -> (n, PRED_N) the heading at the end of each step: heading0 + the turns so
        far (`unicycle`'s)"""
        v, k = self.kin_vk(o)
        return o[:, 2].float()[:, None] * math.pi + torch.cumsum(v * k * PS.DT, 1)

    def forward(self, scans: torch.Tensor, state_n: torch.Tensor, h: torch.Tensor, pose: bool = False):
        """(n, PRED_OUT): the points / PRED_SCALE and the 'visible' logit; with `pose`, (that, the pose now (n, 3) or None)"""
        o = self.mlp(torch.cat([self.enc(scans), state_n.detach(), h.detach()], 1))
        if not self.smooth:
            return (o, None) if pose else o
        self.psi_out = self.kin_heading(o.detach())
        pts, z = self.kin_path(o)
        q = torch.cat([pts.reshape(-1, 2 * PRED_N) / PRED_SCALE, o[:, -1:].float()], 1).to(o.dtype)
        return (q, z) if pose else q


def pred_points(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """a prediction (n, PRED_OUT) -> (points (n, PRED_N, 2) [m] in the car's frame at the decision, P(visible) (n,))"""
    return q[:, :2 * PRED_N].reshape(-1, PRED_N, 2) * PRED_SCALE, torch.sigmoid(q[:, -1])


def pred_headings(pts: torch.Tensor, psi: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(n, N, 2) predicted points -> (n, N) the predicted car's heading at each: `psi` when the head gives it (the smooth
    head's integrated heading), else the direction from the point before (the first point: towards the second)"""
    if psi is not None:
        return psi.float()
    seg = pts[:, 1:] - pts[:, :-1]
    a = torch.atan2(seg[..., 1], seg[..., 0])
    return torch.cat([a[:, :1], a], 1)


def prev_plan_poses(rel: torch.Tensor, rel_psi: torch.Tensor, rel_v: torch.Tensor) -> torch.Tensor:
    """--pred-clear: where the previous plan puts the car at t_k = 0.1 k s after this decision, k = 1..10, (n, 10, 3) x, y,
    heading -- in the frame of the pose that plan said the car would be in now (its 0.1 s point; the frame `PlanObs2.rel` is
    in, which the trunk reads as 'the car now'). The plan was drawn 0.1 s ago: its point j falls at t - 0.1 + 0.1 j, so its
    points 2..10 are t_1..t_9 (`rel`, `rel_psi`). It ends at t_9: t_10 is its last point carried on 0.1 s straight along its
    last heading at its last speed (`rel_v`'s last)"""
    x9, h9, v9 = rel[:, -1], rel_psi[:, -1], rel_v[:, -1]
    x10 = x9 + (v9 * PS.DT)[:, None] * torch.stack([torch.cos(h9), torch.sin(h9)], -1)
    return torch.cat([torch.cat([rel, x10[:, None]], 1), torch.cat([rel_psi, h9[:, None]], 1)[..., None]], -1)


def pred_zigzag(pts: torch.Tensor, real: torch.Tensor, min_seg: float = 0.05) -> torch.Tensor:
    """(11-25) how much a predicted path zig-zags: (n, N, 2) predicted points and (n, N, 2) where the car really was ->
    (m,) [deg] per decision, the mean |turn| between successive segments of the prediction minus the same of the realised
    path, over the turns where the realised car moved >= `min_seg` [m] in both segments (a standing car has no direction);
    decisions with no such turn are left out"""
    def turns(P):
        s = P[:, 1:] - P[:, :-1]
        a = torch.atan2(s[..., 1], s[..., 0])
        return (torch.remainder(a[:, 1:] - a[:, :-1] + math.pi, 2.0 * math.pi) - math.pi).abs(), s.norm(dim=-1)
    tp, _ = turns(pts.float())
    tr, lr = turns(real.float())
    m = ((lr[:, 1:] >= min_seg) & (lr[:, :-1] >= min_seg)).float()
    n = m.sum(1)
    z = ((tp - tr) * m).sum(1) / n.clamp_min(1.0)
    return torch.rad2deg(z[n > 0])


class Actor2(nn.Module):
    """v2: a trunk, the plan's Gaussian mean (20) and a speed estimate [m/s].

    `memory` > 0 (11-09, the user chose "A1": adapt by remembering): a GRU of that size reads the trunk's features every
    decision and its state is added back to them (`mem_out`, zero at the start, so a memory actor made from a v2 one
    drives exactly as it did). In a session the obstacles stay where they are until they are taken away, and an empty
    track stays empty -- what the car saw a lap ago tells it what is behind the chicane's walls now. The state is passed
    in and out (`forward(..., h)` sets `h_out`); `PlanObs2.mem_h` carries it between decisions and clears it at a restart.
    `log_std_floor` (a buffer, not saved): the exploration noise's floor per output (11-09: the first step's speed output had
    shrunk to 0.17 and the policy never tried full throttle at the end of the straight).
    `aux_dim` > 0 (11-10, the user: "기억은 필요해 ... 사각지대에 장애물이 있어서 사고나는거"): a linear read-out of the memory's
    state, trained (plan_rl --mem-aux) to say where the boxes ahead are -- hidden ones too -- so the memory holds what it saw a
    lap ago. It is a training aid only: driving never reads it.
    `mem_front` (11-11, the user: "앞쪽으로 가자"): the memory sits BEFORE the trunk. The GRU reads the trunk's own inputs (the
    LiDAR encoding and the normalised state -- the speed and steering it needs to carry what it saw forward) and the trunk
    reads those inputs plus the memory's state, so two nonlinear layers mix what is remembered with what is seen now and
    the reaction to what is in view does not have to pass through the memory. Why: added after the trunk, only a linear
    layer followed the memory -- wiping it moved the plans 12-17 cm at 1 s, toward the hidden boxes, and laps were no
    faster for it (plan25n8). No `mem_out`; such an actor is trained from scratch with its memory.
    `mem_value` (11-12, the user: "ㅇㅋ"): a linear read-out of the memory's state that is taught the return (plan_rl
    --mem-value). Measured: of the memory's two teachers only the speed estimate pushed it one way (agreement between
    minibatches 0.8-1.0, the policy loss -0.1..0.1), and a memory the same size taught a value-like target on recorded inputs
    came to hold "this session has boxes" (0.92 against the policy's own 0.61) -- with `mem_keep_s`. A training aid only:
    driving never reads it; the privileged critic still makes the advantages.
    `mem_keep_s` > 0 (11-12, at the start only): the GRU's update gate begins holding, each value's keep time spread over
    0.2 s .. this long (Tallec & Ollivier 2018; `mem_keep_log`: as many values per octave, so half of them stay short).
    PyTorch's default starts every value at ~0.2 s: the learning signal a lap back was 1e-12, and the value-like target
    above left the memory at 0.56 (= no memory) without this.
    `branches` = 2 (11-13, the user: "2는 괜찮네."): two plans and a choice between them. Measured on plan25y u125
    (`tools/side_flip.py`, `box_react.py`): before a box it hits hard the plan aims within 0.3 m of the box's centre in 44 %
    of the decisions beyond 1.5 m (clean passes 14 %) and settles its side at 1.9 m (clean 3.1 m) -- one Gaussian between
    "left of it" and "right of it" has its mean on the box. `pi` gives two means (`all_heads`), `choice` two logits. The
    two are told apart by one thing only: the first step's turn of plan 0 is never right of plan 1's (their middle and a
    gap >= 0, softplus -- so the two cannot swap roles from state to state, and where one way is right the gap goes to 0);
    every other output is each plan's own. The speed estimate and the noise are shared. Driving takes the plan with the
    larger logit; training samples the choice (plan_rl --branches). The state gets one more input, the plan chosen last
    (`PlanObs2.prev_branch`: +1 plan 0, -1 plan 1, 0 at a start), so a side once taken can be kept. `heads` returns the
    chosen plan in the one-plan shape and leaves `branch_out`, `logits_out`, `means_out` for those who want both.
    `box_memory` > 0 (11-19, with `mem_front`; the user: "맵에 대한 정보가 신경망 속에 은연중에 외우고 있을건데, 그걸 활용할 수
    없나?" -> "함 해봐"): a second GRU of this size beside the first, reading the same inputs; the trunk reads both states
    (the state passed around is the two side by side, `memory` = their total, `memory_main` = the first's). The side task
    (`aux`, plan_rl --mem-aux-seen --mem-aux-lr) reads and teaches only the second. Why apart: the first GRU carries where the
    car is (decoder 0.32 m); taught the side task in the policy's step it learnt nothing in 60 updates, taught by an
    optimizer of its own it learnt (AUC 0.37 -> 0.68 in 5 updates) and the car lost its way -- walls 0.05 -> 8.4 a minute,
    and 9.8-11.8 at the same update on both GPUs even at a tenth of the step, while the side task left in the policy's
    loss stayed at 0.07-0.29. From a checkpoint without it the trunk's columns for it start at zero: same driving.
    `aux_in` (11-20, with `box_memory` and `aux_dim`; the user: "명시적으로 장애물이 어디인지 기억하는지까지 학습시켰는데 달라지는게
    없냐"): the trunk also reads the box memory's own answer -- the read-out's first AUX_IN_SLOTS boxes as (P(present),
    position x P) -- through weights it cannot move (detached), so the answer keeps the side task's meaning. Why: the box
    memory told passed boxes apart at 98-100 % in training (0.87-0.94 on held-out layouts) while the trunk's weights on its
    128 values were a tenth of the others' and re-hits swung 17 % <-> 44-63 % between checkpoints 25 updates apart.
    `pred_clear` (11-25, with `pred_opp`; see PC_N): the trunk also reads PC_N clearances between the previous plan and the
    predicted other car(s), after every other input, zero-weighted at the start. `pc_geom` = (car length, width, margin) [m]
    (the plan-car cost's), kept in the checkpoint (`pc_geom` buffer). `forward` / `remember` then need `prev` = (the previous
    plan's poses at t_1..t_10 (n, 10, 3), whether there is one (n,) bool) -- `PlanObs2.prev_poses()`; the values used are left
    in `pc_out`."""

    GAP0 = 0.5                # the gap between the two plans' first-step turn at the start (the action before its tanh)
    AUX_IN_SLOTS, AUX_IN_KEYS = 3, 4     # `aux_in`: the read-out's boxes (present, dx/5, dy/5, distance/5 along the track)

    def __init__(self, beams: int, hidden: int = 256, std0: float = 0.6, state_dim: int = None, memory: int = 0,
                 aux_dim: int = 0, mem_front: bool = False, mem_value: bool = False, mem_keep_s: float = 0.0,
                 mem_keep_log: bool = False, branches: int = 1, box_memory: int = 0, aux_in: bool = False,
                 pace_caps: str = "", pace_hold: int = 10, pace_floor: float = 0.05, line_shifts: str = "",
                 pred_opp: bool = False, pred_smooth: bool = False, pred_kin=None, pred_clear: bool = False, pc_geom=None):
        super().__init__()
        self.branches = int(branches)
        # (11-21 redesign, the user "ㄱㄱ": speed only, a level every 1 s, absolute caps, a floor per level) the pace: every
        # `pace_hold` decisions the policy picks one of 1 + len(caps) levels -- no cap, then each cap [m/s] -- and every plan
        # until the next pick is drawn under it (`plan_steps.roll` cap). The plan's own noise is taken back by the next
        # decision (re-planned every 0.1 s: approach speed to a blind box swung 0.2 m/s between passes); a held level is not --
        # an oracle cap of 5.5 m/s 10 m before boxes cut hard box hits 91 -> 30 for +0.3 s a lap
        caps = [float(x) for x in str(pace_caps or "").split(",") if x.strip()]
        # (11-22, the user: "미리 알고 미리 라인 다르게"; output noise on the plan is re-planned away within 0.1 s -- the driven line
        # spread 0.09 m in training, and held output noise (qual26w/x) pushed the car into the bottom U-turn's walls) the
        # line: the same held, picked level, its levels a lateral offset [m] 1 s ahead of the policy's own line (+ left); the
        # policy sees the level and `plan_rl --line-follow` makes the plan honour it. One or the other of pace and line
        shifts = [float(x) for x in str(line_shifts or "").split(",") if x.strip()]
        if caps and shifts:
            raise ValueError("pace caps or line shifts, not both")
        self.pace_kind = "line" if shifts else "cap"
        self.pace_levels = (1 + len(caps or shifts)) if (caps or shifts) else 0
        self.pace_hold, self.pace_floor = int(pace_hold), float(pace_floor)
        if self.pace_levels and self.branches != 1:
            raise ValueError("pace: one plan only")
        if self.pace_levels and not 0.0 <= self.pace_floor * self.pace_levels < 1.0:
            raise ValueError("pace_floor x levels must stay below 1")
        if self.branches not in (1, 2):
            raise ValueError("branches: 1 or 2")
        sd = (STATE_DIM2 + (1 if self.branches == 2 else 0) + ((self.pace_levels + 1) if self.pace_levels else 0)
              if state_dim is None else int(state_dim))
        self.mem_front = bool(mem_front) and int(memory) > 0
        self.enc = ScanEncoder(beams)
        self.norm = RunningNorm(sd)
        self.box_memory = int(box_memory) if self.mem_front else 0
        if int(box_memory) and not self.mem_front:
            raise ValueError("box_memory needs mem_front")
        self.aux_in = bool(aux_in)
        if self.aux_in and not (self.box_memory and int(aux_dim) >= self.AUX_IN_SLOTS * self.AUX_IN_KEYS):
            raise ValueError("aux_in needs box_memory and the boxes' read-out (aux_dim)")
        n_ai = self.AUX_IN_SLOTS * self.AUX_IN_KEYS if self.aux_in else 0
        # (11-24) `pred_opp`: the other car's predicted next second (`OppPredictor`), read by the trunk detached, its columns
        # last and zero at the start -- a policy given the head drives exactly as it did without it
        self.pred_opp = bool(pred_opp)
        if self.pred_opp and not self.mem_front:
            raise ValueError("pred_opp needs memory with mem_front")
        # (11-25) `pred_smooth`: the head's points are a path integrated from its pose now, speed and curvature
        # (`OppPredictor.smooth`; `pred_kin` the car's curvature limit) -- same output, same trunk inputs
        self.pred_smooth = bool(pred_smooth)
        if self.pred_smooth and not self.pred_opp:
            raise ValueError("pred_smooth needs pred_opp")
        n_pr = PRED_IN if self.pred_opp else 0
        self.pred_clear = bool(pred_clear)
        if self.pred_clear and not self.pred_opp:
            raise ValueError("pred_clear needs pred_opp")
        if self.pred_clear:
            if pc_geom is None or len(pc_geom) != 3:
                raise ValueError("pred_clear needs pc_geom = (car length, car width, margin) [m]")
            self.pc_dims = tuple(float(x) for x in pc_geom)
            self.register_buffer("pc_geom", torch.tensor(self.pc_dims, dtype=torch.float32))
        n_pc = PC_N if self.pred_clear else 0
        self.pc_out = None
        self.trunk = nn.Sequential(nn.Linear(256 + sd + ((int(memory) + self.box_memory + n_ai + n_pr + n_pc) if self.mem_front else 0), hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.pred = None
        self.pred_out = None
        if self.pred_opp:
            self.pred = OppPredictor(beams, sd, int(memory) + self.box_memory, hidden, smooth=self.pred_smooth, kin=pred_kin)
            with torch.no_grad():
                self.trunk[0].weight[:, -(n_pr + n_pc):] = 0.0      # the prediction's columns (and the clearances' after them)
                self.pred.enc.load_state_dict(self.enc.state_dict())
        self.pi = nn.Linear(hidden, self.branches * PS.ACT_DIM)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
        if self.branches == 2:
            self.choice = nn.Linear(hidden, 2)
            nn.init.zeros_(self.choice.weight); nn.init.zeros_(self.choice.bias)       # both plans as likely at the start
            with torch.no_grad():
                self.pi.bias[PS.ACT_DIM] = math.log(math.expm1(self.GAP0))
        self.branch_out = self.logits_out = self.means_out = None
        self.pace_logits_out = None
        if self.pace_levels:
            self.pace_head = nn.Linear(hidden, self.pace_levels)
            nn.init.zeros_(self.pace_head.weight); nn.init.zeros_(self.pace_head.bias)
            with torch.no_grad():
                self.pace_head.bias[0] = 1.0                    # no cap twice as likely as each cap at the start
            self.register_buffer("pace_v", torch.tensor([math.inf] + (caps or [math.inf] * len(shifts))))
            if shifts:
                self.register_buffer("line_dy", torch.tensor([0.0] + shifts))
        self.v_est = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((PS.ACT_DIM,), math.log(std0)))
        self.memory_main = int(memory)
        self.memory = self.memory_main + self.box_memory          # the state passed between decisions: both GRUs' side by side
        if self.memory:
            self.gru = nn.GRUCell(256 + sd if self.mem_front else hidden, self.memory_main)
            if not self.mem_front:
                self.mem_out = nn.Linear(self.memory, hidden)
                nn.init.zeros_(self.mem_out.weight); nn.init.zeros_(self.mem_out.bias)
            if self.box_memory:
                self.gru_box = nn.GRUCell(256 + sd, self.box_memory)
                self.keep_init(30.0, True, self.gru_box)          # keep times 0.1 .. 30 s, as many per octave
            if aux_dim:
                self.aux = nn.Linear(self.box_memory or self.memory, int(aux_dim))
            if mem_value:
                self.v_mem = nn.Linear(self.memory_main, 1)
            if mem_keep_s > 0:
                self.keep_init(float(mem_keep_s), bool(mem_keep_log))
        self.register_buffer("log_std_floor", torch.full((PS.ACT_DIM,), -math.inf), persistent=False)
        self.h_out = None

    def features(self, scans: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """what the memory reads: the trunk's output, or with `mem_front` the trunk's inputs (`remember` runs the trunk)"""
        x = torch.cat([self.enc(scans), self.norm(state)], 1)
        return x if self.mem_front else self.trunk(x)

    def heads(self, f: torch.Tensor):
        if self.branches == 2:
            means, logits, log_std, v_hat = self.all_heads(f)
            br = logits.argmax(1)
            self.branch_out, self.logits_out, self.means_out = br, logits, means
            return means[torch.arange(f.shape[0], device=f.device), br], log_std, v_hat
        log_std = torch.maximum(self.log_std, self.log_std_floor)
        if self.pace_levels:
            self.pace_logits_out = self.pace_head(f)
        return self.pi(f), log_std.expand(f.shape[0], -1), self.v_est(f)[:, 0] * 5.0

    def pace_p(self, logits: torch.Tensor) -> torch.Tensor:
        """the levels' probabilities, never below `pace_floor` each"""
        return self.pace_floor + (1.0 - self.pace_levels * self.pace_floor) * torch.softmax(logits, 1)

    def all_heads(self, f: torch.Tensor):
        """two plans: (means (B, 2, 20), the choice's logits (B, 2), log_std (B, 20), the speed estimate (B,)). Plan 0's
        first-step turn is the middle + gap / 2, plan 1's the middle - gap / 2 (a larger turn output steers left)"""
        raw = self.pi(f).view(-1, 2, PS.ACT_DIM)
        mid, gap = raw[:, 0, 0], nn.functional.softplus(raw[:, 1, 0])
        first = torch.stack([mid + 0.5 * gap, mid - 0.5 * gap], 1)
        means = torch.cat([first[:, :, None], raw[:, :, 1:]], 2)
        log_std = torch.maximum(self.log_std, self.log_std_floor)
        return means, self.choice(f), log_std.expand(f.shape[0], -1), self.v_est(f)[:, 0] * 5.0

    def keep_init(self, keep_s: float, log: bool = False, cell: Optional[nn.GRUCell] = None) -> None:
        """the update gate's bias so that value k of the memory, left alone, keeps 1 - 1/T_k of itself per decision"""
        cell = self.gru if cell is None else cell
        n, t_max = cell.hidden_size, keep_s / PS.DT
        with torch.no_grad():
            t = (torch.exp(torch.empty(n).uniform_(0.0, math.log(t_max))).clamp_min(1.0 + 1e-3) if log
                 else torch.empty(n).uniform_(2.0, t_max))
            cell.bias_ih[n: 2 * n] = torch.log(t - 1.0).to(cell.bias_ih)
            cell.bias_hh[n: 2 * n] = 0.0

    def mem_cell(self, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """one decision of the whole memory: (n, F), (n, memory) -> (n, memory)"""
        if not self.box_memory:
            return self.gru(f, h)
        m = self.memory_main
        return torch.cat([self.gru(f, h[:, :m]), self.gru_box(f, h[:, m:])], 1)

    def aux_read(self, hs: torch.Tensor) -> torch.Tensor:
        """the side task's read-out: of the box memory's part when there is one"""
        return self.aux(hs[:, self.memory_main:] if self.box_memory else hs)

    def aux_feed(self, hs: torch.Tensor) -> torch.Tensor:
        """`aux_in`: the box memory's answer as the trunk reads it, (n, 12): per box P(present) and its position x P (an absent
        box reads zeros, as the side task's target does). Detached: driving cannot bend what the side task taught"""
        k, n = self.AUX_IN_SLOTS, self.AUX_IN_KEYS
        a = nn.functional.linear(hs[:, self.memory_main:].detach(), self.aux.weight[:k * n].detach(), self.aux.bias[:k * n].detach())
        a = a.view(-1, k, n)
        p = torch.sigmoid(a[..., :1])
        return torch.cat([p, a[..., 1:] * p], -1).reshape(-1, k * n)

    def mem_trunk(self, f: torch.Tensor, hs: torch.Tensor, pin: Optional[torch.Tensor] = None,
                  pc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`mem_front`: the trunk on the inputs and the memory's state (and its answer with `aux_in`; `pin`: the other car's
        prediction as `pred_feed` gives it, with `pred_opp`; `pc`: the PC_N clearances as `clear_in` gives them, with
        `pred_clear`)"""
        x = [f, hs, self.aux_feed(hs)] if self.aux_in else [f, hs]
        if self.pred_opp:
            if pin is None:
                raise ValueError("a pred_opp actor's trunk reads the prediction: pass pin (pred_feed(pred_q(...)))")
            x = x + [pin]
        if self.pred_clear:
            if pc is None:
                raise ValueError("a pred_clear actor's trunk reads the plan clearances: pass pc (clear_in(...))")
            x = x + [pc.to(f.dtype)]
        return self.trunk(torch.cat(x, 1))

    @torch.no_grad()
    def clear_in(self, q: torch.Tensor, prev, psi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`pred_clear`: the trunk's PC_N inputs (n, PC_N) from a prediction `q` (n, PRED_OUT) and `prev` = (the previous plan's
        poses (n, PC_N, 3) at t_1..t_PC_N in the frame of the car now, (n,) bool there is one): at each t_k the smallest
        clearance [m] between our car where the previous plan puts it and every predicted car that counts (P(there) >=
        PC_PMIN; today the head's one car), PC_FAR without either, clipped, / PC_SCALE. Built on `plan_car_cost.clearance`'s
        element interface: a head that predicts more cars (slots with their own P(there)) or obstacle points adds elements
        there, the input stays PC_N numbers. `psi` (n, PRED_N): the predicted car's headings (default: the smooth head's own,
        else from the points). Detached"""
        from .plan_car_cost import car_cover, clearance
        if prev is None:
            raise ValueError("a pred_clear actor needs prev = (the previous plan's poses, there is one): PlanObs2.prev_poses()")
        poses, ok = prev
        pts, pv = pred_points(q.detach().float())
        if psi is None:
            psi = self.pred.psi_out if (self.pred is not None and self.pred.psi_out is not None
                                        and self.pred.psi_out.shape[0] == q.shape[0]) else None
        psi = pred_headings(pts, psi)
        L, W, mg = self.pc_dims
        elem = car_cover(torch, pts[:, None], psi[:, None], L, W, conf=pv[:, None])      # one car: 3 discs, P(there)
        c = clearance(torch, poses[..., :2].float(), poses[..., 2].float(), *elem, L, W, mg, far=PC_FAR, lo=PC_LO, hi=PC_HI,
                      w_min=PC_PMIN)
        c = torch.where(ok[:, None], c, torch.full_like(c, PC_FAR))
        return c / PC_SCALE

    def pred_q(self, scans: torch.Tensor, f: torch.Tensor, hs: torch.Tensor, pose: bool = False):
        """`pred_opp`: the raw prediction (n, PRED_OUT) from the scans, the features `f` (their normalised-state part) and
        the memory's state `hs` -- the head's own weights only carry a gradient. `pose`: (that, the smooth head's pose now
        (n, 3) -- None for the old head)"""
        return self.pred(scans, f[:, 256:], hs, pose)

    @staticmethod
    def pred_feed(q: torch.Tensor) -> torch.Tensor:
        """what the trunk reads of a prediction, detached: P(visible), then the points x P(visible) (a car not there reads
        zeros)"""
        q = q.detach()
        p = torch.sigmoid(q[:, -1:])
        return torch.cat([p, q[:, :-1] * p], 1)

    def vmem_read(self, hs: torch.Tensor) -> torch.Tensor:
        """the value read-out: of the first GRU's part"""
        return self.v_mem(hs[:, :self.memory_main])

    def memory_alone(self, f_seq: torch.Tensor, h: torch.Tensor, done: torch.Tensor,
                     h_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """the memory's states along a replayed sequence on features it cannot move: (L, n, F) from the state `h` each began
        with -> (L * n, memory). What the training's read-outs (`aux`, `v_mem`) learn from: they teach the memory and leave
        what feeds it -- the LiDAR part, the policy's -- alone. `done` (L, n): a restart clears the state for the decisions
        after it -- or, with `h_start` (L, n, memory), sets it to the state that decision was driven from (zero after an
        ordinary restart, a stored scene's memory after a scene restart)"""
        hs = []
        for i in range(f_seq.shape[0]):
            if i > 0:
                h = (h * (1.0 - done[i - 1])[:, None] if h_start is None
                     else torch.where(done[i - 1][:, None] > 0, h_start[i], h))
            h = self.mem_cell(f_seq[i].detach(), h)
            hs.append(h)
        return torch.cat(hs, 0)

    def remember(self, f: torch.Tensor, h: Optional[torch.Tensor], scans: Optional[torch.Tensor] = None, prev=None,
                 pc_over=None):
        """one decision of the memory: (features with the memory added, the new state). A `pred_opp` actor also needs the
        `scans` (its prediction, left in `pred_out`), a `pred_clear` one `prev` (the clearances, left in `pc_out`).
        `pc_over` (11-25, plan_rl --pred-clear-oracle): (clearances (n, PC_N) at the trunk's scale, rows (n,) bool) -- on those
        rows the trunk reads these instead of the head's (`pred_oracle.copy_oracle`)"""
        h0 = h if h is not None else torch.zeros(f.shape[0], self.memory, device=f.device, dtype=f.dtype)
        h1 = self.mem_cell(f, h0)
        if self.mem_front:
            pin = pc = None
            if self.pred_opp:
                if scans is None:
                    raise ValueError("a pred_opp actor's memory step needs the scans (its prediction reads them)")
                self.pred_out = self.pred_q(scans, f, h1)
                pin = self.pred_feed(self.pred_out)
            if self.pred_clear:
                pc = self.clear_in(self.pred_out, prev)
                if pc_over is not None:
                    pc = torch.where(pc_over[1][:, None], pc_over[0].to(pc.dtype), pc)
                self.pc_out = pc
            return self.mem_trunk(f, h1, pin, pc), h1
        return f + self.mem_out(h1), h1

    def forward(self, scans: torch.Tensor, state: torch.Tensor, h: Optional[torch.Tensor] = None, prev=None, pc_over=None):
        f = self.features(scans, state)
        if self.memory:
            f, self.h_out = self.remember(f, h, scans, prev, pc_over)
        return self.heads(f)

    def policy_parameters(self):
        """every weight but the prediction head's (`pred_opp`: it has its own optimizer and no policy gradient)"""
        return [p for n, p in self.named_parameters() if not n.startswith("pred.")]


class PlanObs2(PlanObs):
    """v2's inputs: LiDAR (as v1), the raw IMU, the last commands, the last plan in its own frame"""

    def __init__(self, env, lim: PS.Limits, goal=None, rows: Optional[torch.Tensor] = None):
        Bl = int((env.learner_ids if rows is None else rows).numel())
        dev = env.device
        self.rel = torch.zeros(Bl, PS.N_STEP - 1, 2, device=dev)     # last plan's points 2..10 in its 0.1 s pose's frame
        self.rel_v = torch.zeros(Bl, PS.N_STEP, device=dev)          # its speeds
        # (11-25, --pred-clear; not trunk inputs) the last plan's heading at points 2..10 in the same frame, and whether there
        # is a last plan (False from a start until the first plan is drawn)
        self.rel_psi = torch.zeros(Bl, PS.N_STEP - 1, device=dev)
        self.prev_ok = torch.zeros(Bl, dtype=torch.bool, device=dev)
        self.cmd_d = torch.zeros(Bl, device=dev)                     # its steering / speed 0.1 s in: what was commanded
        self.cmd_v = torch.zeros(Bl, device=dev)
        # a two-plan actor (`Actor2.branches`): which plan was chosen last, +1 / -1, 0 at a start (B_l,). None = no such
        # input (every one-plan policy)
        self.prev_branch = None
        # a pace actor (`Actor2.pace_levels`): the level in force (B_l,) long and the decisions it still holds; None = no such
        # input. A new episode picks at once (left 0, level 0 = no cap until then)
        self.pace_idx = self.pace_left = None
        self.pace_k = 0
        super().__init__(env, lim, goal, rows)

    def pace_on(self, levels: int, hold: int) -> None:
        Bl = self.rel.shape[0]
        self.pace_k, self.pace_hold = int(levels), int(hold)
        self.pace_idx = torch.zeros(Bl, dtype=torch.long, device=self.rel.device)
        self.pace_left = torch.zeros(Bl, dtype=torch.long, device=self.rel.device)

    def reset_rows(self, rows: torch.Tensor) -> None:
        super().reset_rows(rows)
        if rows.numel() == 0:
            return
        self.rel[rows] = 0.0; self.rel_v[rows] = 0.0; self.cmd_d[rows] = 0.0; self.cmd_v[rows] = 0.0
        self.rel_psi[rows] = 0.0; self.prev_ok[rows] = False
        if getattr(self, "mem_h", None) is not None:
            self.mem_h[rows] = 0.0                                   # a new episode remembers nothing
        if self.prev_branch is not None:
            self.prev_branch[rows] = 0.0
        if self.pace_idx is not None:
            self.pace_idx[rows] = 0; self.pace_left[rows] = 0

    def set_plan2(self, p: Dict[str, torch.Tensor], rows: Optional[torch.Tensor] = None) -> None:
        """the plan just drawn (`plan_steps.roll`, in the frame it was drawn from): kept in its own 0.1 s pose's frame"""
        i1 = PS.SUB
        x1, y1, h1 = p["x"][:, i1], p["y"][:, i1], p["psi"][:, i1]
        idx = (torch.arange(2, PS.N_STEP + 1, device=x1.device) * PS.SUB)
        dx = p["x"][:, idx] - x1[:, None]; dy = p["y"][:, idx] - y1[:, None]
        c, s_ = torch.cos(h1)[:, None], torch.sin(h1)[:, None]
        rel = torch.stack([dx * c + dy * s_, -dx * s_ + dy * c], -1)
        rpsi = p["psi"][:, idx] - h1[:, None]
        cd, cv = p["delta"][:, i1], p["v"][:, i1]
        if rows is None:
            self.rel, self.rel_v, self.cmd_d, self.cmd_v = rel, p["pv"].clone(), cd.clone(), cv.clone()   # its own (see `set_plan`)
            self.rel_psi = rpsi.clone(); self.prev_ok = torch.ones_like(self.prev_ok)
        else:
            self.rel_psi = torch.where(rows[:, None], rpsi, self.rel_psi)
            self.prev_ok = self.prev_ok | rows
            self.rel = torch.where(rows[:, None, None], rel, self.rel)
            self.rel_v = torch.where(rows[:, None], p["pv"], self.rel_v)
            self.cmd_d = torch.where(rows, cd, self.cmd_d)
            self.cmd_v = torch.where(rows, cv, self.cmd_v)

    def prev_poses(self):
        """--pred-clear: (the last plan's poses at t_1..t_10 after this decision (B_l, 10, 3), there is a last plan (B_l,))"""
        return prev_plan_poses(self.rel, self.rel_psi, self.rel_v), self.prev_ok

    def state(self) -> torch.Tensor:
        r = self.env.last_result
        Bl = self.rel.shape[0]
        if r is not None and r.imu is not None and r.imu.shape[1] > 0:
            imu = r.imu[self.lid].float().mean(1)                  # gyro xyz, accel xyz
        else:
            imu = torch.zeros(Bl, 6, device=self.rel.device)
        s = torch.cat([imu, self.cmd_d[:, None], self.cmd_v[:, None], self.rel.reshape(Bl, -1), self.rel_v], 1)
        if self.prev_branch is not None:
            s = torch.cat([s, self.prev_branch[:, None]], 1)
        if self.pace_idx is not None:                               # the level in force and the share of its hold left
            s = torch.cat([s, nn.functional.one_hot(self.pace_idx, self.pace_k).to(s.dtype),
                           (self.pace_left.to(s.dtype) / float(self.pace_hold))[:, None]], 1)
        return s

    def priv(self) -> torch.Tensor:
        """v1's critic extras and the true speed / steering (training only)"""
        st = self.env.sim.state[self.lid]
        return torch.cat([super().priv(), st[:, dyn.IVX:dyn.IVX + 1].float(), st[:, dyn.ISTEER:dyn.ISTEER + 1].float()], 1)


def lidar_mount_of(ck: dict) -> float:
    """the scanner's place ahead of the CoG a checkpoint was trained with (0 = the config's default, the nose tip)"""
    return float((ck.get("args") or {}).get("lidar_mount_x") or 0.0)


def load_actor2(ck: dict, device, check_mount: bool = True) -> Actor2:
    # 11-12: a policy reads the scan from where its training's scanner sat; `state_rl.build_env` places it by F1SIM_LIDAR_MOUNT_X
    if check_mount:
        want, have = lidar_mount_of(ck), float(os.environ.get("F1SIM_LIDAR_MOUNT_X") or 0.0)
        if abs(want - have) > 1e-6:
            raise SystemExit(f"this checkpoint was trained with the LiDAR at {want or 'the default 0.297'} m ahead of the CoG, the "
                             f"environment is built with {have or 'the default 0.297'}: set F1SIM_LIDAR_MOUNT_X="
                             f"{want if want else ''} before the environment is built")
    aux = ck["actor"].get("aux.weight")
    actor = Actor2(int(ck["beams"]), std0=float(ck["args"].get("std0", 0.6)),
                   memory=int(ck["args"].get("memory", 0) or 0), aux_dim=int(aux.shape[0]) if aux is not None else 0,
                   mem_front=bool(ck["args"].get("mem_front", False)), mem_value="v_mem.weight" in ck["actor"],
                   branches=int(ck["args"].get("branches", 1) or 1),
                   box_memory=int(ck["args"].get("box_memory", 0) or 0),
                   aux_in=bool(ck["args"].get("mem_aux_in", False)),
                   pace_caps=str(ck["args"].get("pace_caps", "") or ""), pace_hold=int(ck["args"].get("pace_hold", 10) or 10),
                   pace_floor=float(ck["args"].get("pace_floor", 0.05) or 0.05),
                   line_shifts=str(ck["args"].get("line_shifts", "") or ""),
                   pred_opp=any(k.startswith("pred.") for k in ck["actor"]),
                   pred_smooth="pred.kin_k" in ck["actor"],
                   pred_kin=((ck["actor"]["pred.kin_v"], ck["actor"]["pred.kin_k"]) if "pred.kin_k" in ck["actor"] else None),
                   pred_clear="pc_geom" in ck["actor"],
                   pc_geom=(ck["actor"]["pc_geom"].tolist() if "pc_geom" in ck["actor"] else None)
                   ).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


def plan_v2(actor: Actor2, obs: PlanObs2, lim: PS.Limits, sample: bool = False, eps: Optional[torch.Tensor] = None,
            cap: Optional[torch.Tensor] = None):
    """one decision of a v2 policy: (u, mean, log_std, v_hat, plan). The plan starts from the speed estimate and the last
    steering command. A memory actor's state is `obs.mem_h` (made here, cleared by `obs.reset_rows`), stepped once per call:
    call once per decision. A two-plan actor (`Actor2.branches`): the plan with the larger logit; `obs.prev_branch` (made
    here) is the input that says which was chosen last, `actor.means_out` / `logits_out` hold both plans."""
    two = getattr(actor, "branches", 1) == 2
    if two and getattr(obs, "prev_branch", None) is None:
        obs.prev_branch = torch.zeros(obs.scans.shape[0], device=obs.scans.device)
    h = None
    if getattr(actor, "memory", 0):
        h = getattr(obs, "mem_h", None)
        if h is None or h.shape != (obs.scans.shape[0], actor.memory):
            h = obs.mem_h = torch.zeros(obs.scans.shape[0], actor.memory, device=obs.scans.device)
    pace = getattr(actor, "pace_levels", 0)
    if pace and getattr(obs, "pace_idx", None) is None:
        obs.pace_on(pace, actor.pace_hold)
    if getattr(actor, "pred_clear", False):                 # (11-25) the clearances read the last plan
        mean, log_std, v_hat = actor(obs.scans, obs.state(), h, prev=obs.prev_poses())
    else:
        mean, log_std, v_hat = actor(obs.scans, obs.state(), h)
    if getattr(actor, "memory", 0):
        obs.mem_h = actor.h_out.detach()
    if pace:
        # the pace: rows whose level ran out pick again (sampled with `sample`, else the likeliest), held `pace_hold` decisions
        due = obs.pace_left <= 0
        pick = (torch.multinomial(actor.pace_p(actor.pace_logits_out), 1)[:, 0] if sample
                else actor.pace_logits_out.argmax(1))
        obs.pace_idx = torch.where(due, pick, obs.pace_idx)
        obs.pace_left = torch.where(due, torch.full_like(obs.pace_left, actor.pace_hold), obs.pace_left) - 1
        pc = actor.pace_v[obs.pace_idx]
        cap = pc if cap is None else torch.minimum(cap, pc)
    if two:
        obs.prev_branch = 1.0 - 2.0 * actor.branch_out.float()
    u = mean + log_std.exp() * (torch.randn_like(mean) if eps is None else eps) if sample else mean      # `eps`: the caller's noise (e.g. held)
    p = PS.roll(torch.tanh(u), v_hat.detach().clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim, cap)     # `cap`: a speed cap per row
    return u, mean, log_std, v_hat, p


# ------------------------------------------------------------------------------------------------ v3 (11-09)
# The user: "저만큼 제어딜레이 가정해서 출력하게하면 되잖아". The plan knows the car's control delay (`plan_steps3`): the 20
# numbers are commands, the plan is the car's response through the command delay, the servo, the speed loop and MAP's
# steering table. v3's actor sees v2's inputs and the servo angle the last plan says the car has now (nothing measures
# it); "the last commands" are the commands last sent, which are still in effect for the first 15 ms of the next plan.
STATE_DIM3 = STATE_DIM2 + 1
FORMAT3 = "plan_steps3"


class PlanObs3(PlanObs2):
    """v3's inputs: v2's and the servo angle (from the last plan)"""

    def __init__(self, env, lim: PS.Limits, goal=None):
        Bl = int(env.learner_ids.numel())
        self.servo = torch.zeros(Bl, device=env.device)
        super().__init__(env, lim, goal)

    def reset_rows(self, rows: torch.Tensor) -> None:
        super().reset_rows(rows)
        if rows.numel():
            self.servo[rows] = 0.0

    def set_plan2(self, p: Dict[str, torch.Tensor], rows: Optional[torch.Tensor] = None) -> None:
        super().set_plan2(p, rows)
        i1 = PS.SUB
        sv, cd, cv = p["delta"][:, i1], p["cmd_d"][:, i1 - 1], p["cmd_v"][:, i1 - 1]
        if rows is None:
            self.servo, self.cmd_d, self.cmd_v = sv, cd, cv
        else:
            self.servo = torch.where(rows, sv, self.servo)
            self.cmd_d = torch.where(rows, cd, self.cmd_d)
            self.cmd_v = torch.where(rows, cv, self.cmd_v)

    def state(self) -> torch.Tensor:
        return torch.cat([super().state(), self.servo[:, None]], 1)


def plan3_models(env, lim: PS.Limits, device):
    """(actuators, turning) of the nominal car for `plan_steps3`"""
    from .. import plan_steps3 as P3
    from .. import map_controller as M
    return (P3.Actuators.nominal(env.tracker.nominal),
            P3.Turning(env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), lim.wheelbase, device))


def plan_v3(actor: Actor2, obs: PlanObs3, lim: PS.Limits, act, turn, sample: bool = False):
    """one decision of a v3 policy: (u, mean, log_std, v_hat, plan). The plan starts from the speed estimate, the servo
    angle the last plan left and the command still in effect."""
    from .. import plan_steps3 as P3
    mean, log_std, v_hat = actor(obs.scans, obs.state())
    u = mean + log_std.exp() * torch.randn_like(mean) if sample else mean
    p = P3.roll(torch.tanh(u), v_hat.detach().clamp(-PS.V_REV, PS.V_FWD), obs.servo,
                torch.stack([obs.cmd_d, obs.cmd_v], 1), lim, act, turn)
    return u, mean, log_std, v_hat, p


def load_actor3(ck: dict, device) -> Actor2:
    actor = Actor2(int(ck["beams"]), std0=float(ck["args"].get("std0", 0.6)), state_dim=STATE_DIM3).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


def plan_segments3(p: Dict[str, torch.Tensor], lim: PS.Limits, turn) -> Tuple[torch.Tensor, torch.Tensor]:
    """a v3 plan as MAP reads one: each 0.1 s step's curvature (the table's, for the servo angle and the speed) and mean
    speed"""
    d = p["delta"]; v = p["v"]
    idx = torch.arange(0, PS.N_STEP + 1, device=d.device) * PS.SUB
    dk, vk = d[:, idx], v[:, idx]
    d_mid = 0.5 * (dk[:, 1:] + dk[:, :-1]); v_mid = 0.5 * (vk[:, 1:] + vk[:, :-1])
    return torch.tan(turn.eff(d_mid, v_mid)) / lim.wheelbase, v_mid
