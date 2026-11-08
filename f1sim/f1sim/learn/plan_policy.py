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

    def __init__(self, env, lim: PS.Limits, goal=None):
        self.env, self.lim, self.goal = env, lim, goal
        self.lid = env.learner_ids
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
            self.plan_xy, self.plan_v = xy, v
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
        self.lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), device)
        if self.v3:
            self.actor = load_actor3(ck, device)
            self.obs = PlanObs3(env, self.lim, None)
            self.act3, self.turn3 = plan3_models(env, self.lim, device)
        elif self.v2:
            self.actor = load_actor2(ck, device)
            self.obs = PlanObs2(env, self.lim, None)
        else:
            self.actor = load_actor(ck, device)
            self.obs = PlanObs(env, self.lim, None)
        self.v_hat = None
        Bl = int(self.lid.numel())
        self.cmd_d = torch.zeros(Bl, PS.N_STEP * PS.SUB, device=device)
        self.cmd_v = torch.zeros(Bl, PS.N_STEP * PS.SUB, device=device)
        self.kk = torch.zeros(env.B, PS.N_STEP, device=device)
        self.vv = torch.zeros(env.B, PS.N_STEP, device=device)
        self.first = torch.ones(Bl, dtype=torch.bool, device=device)
        self.last_ep = env.ep_step[self.lid].clone()
        self.session = None
        self.last_mean = None
        # mode "track" (11-09, the user: "얼추 여기에 맞는 제어기 좀 만들어놔"): full physics, `plan_follow.PlanFollower` follows
        # the plan in time (actuator lags inverted), its command goes in through the env's external-command path
        self.follower = None
        if mode == "track":
            from ..plan_follow import PlanFollower
            self.follower = PlanFollower(Bl, device, s_max=float(env.s_max), lut=env.tracker.map.lut,
                                         grip=M.GRIP_LEVELS.index(1.0), wheelbase=self.lim.wheelbase)
            env.ext_mask[self.lid] = True
            env._ext_ids.update(int(i) for i in self.lid.tolist())

    @torch.no_grad()
    def __call__(self, scan=None, ego=None):
        env, lid = self.env, self.lid
        ep = env.ep_step[lid]
        back = ep < self.last_ep                                  # an episode restarted (a reset, the console's 리셋)
        if bool(back.any()):
            self.obs.reset_rows(torch.nonzero(back)[:, 0])
            self.first |= back
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
                _, mean, _, v_hat, p = plan_v2(self.actor, self.obs, self.lim)
                self.v_hat = v_hat
            else:
                mean, _ = self.actor(self.obs.scans, self.obs.state())
                p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], self.lim)
            self.last_mean = mean
            x, y, psi = plan_world(p, st)
            if self.follower is not None:
                self.follower.set_plan(due, x[due], y[due], psi[due], p["v"][due], p["delta"][due],
                                       *((p["cmd_d"][due], p["cmd_v"][due]) if self.v3 else ()))
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
            env.ext_cmd[lid] = to_car(env, lid, self.follower.command(env.sim.state[lid]))
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


class Actor2(nn.Module):
    """v2: a trunk, the plan's Gaussian mean (20) and a speed estimate [m/s]"""

    def __init__(self, beams: int, hidden: int = 256, std0: float = 0.6, state_dim: int = None):
        super().__init__()
        sd = STATE_DIM2 if state_dim is None else int(state_dim)
        self.enc = ScanEncoder(beams)
        self.norm = RunningNorm(sd)
        self.trunk = nn.Sequential(nn.Linear(256 + sd, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, PS.ACT_DIM)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
        self.v_est = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((PS.ACT_DIM,), math.log(std0)))

    def forward(self, scans: torch.Tensor, state: torch.Tensor):
        h = self.trunk(torch.cat([self.enc(scans), self.norm(state)], 1))
        return self.pi(h), self.log_std.expand(scans.shape[0], -1), self.v_est(h)[:, 0] * 5.0


class PlanObs2(PlanObs):
    """v2's inputs: LiDAR (as v1), the raw IMU, the last commands, the last plan in its own frame"""

    def __init__(self, env, lim: PS.Limits, goal=None):
        Bl = int(env.learner_ids.numel())
        dev = env.device
        self.rel = torch.zeros(Bl, PS.N_STEP - 1, 2, device=dev)     # last plan's points 2..10 in its 0.1 s pose's frame
        self.rel_v = torch.zeros(Bl, PS.N_STEP, device=dev)          # its speeds
        self.cmd_d = torch.zeros(Bl, device=dev)                     # its steering / speed 0.1 s in: what was commanded
        self.cmd_v = torch.zeros(Bl, device=dev)
        super().__init__(env, lim, goal)

    def reset_rows(self, rows: torch.Tensor) -> None:
        super().reset_rows(rows)
        if rows.numel() == 0:
            return
        self.rel[rows] = 0.0; self.rel_v[rows] = 0.0; self.cmd_d[rows] = 0.0; self.cmd_v[rows] = 0.0

    def set_plan2(self, p: Dict[str, torch.Tensor], rows: Optional[torch.Tensor] = None) -> None:
        """the plan just drawn (`plan_steps.roll`, in the frame it was drawn from): kept in its own 0.1 s pose's frame"""
        i1 = PS.SUB
        x1, y1, h1 = p["x"][:, i1], p["y"][:, i1], p["psi"][:, i1]
        idx = (torch.arange(2, PS.N_STEP + 1, device=x1.device) * PS.SUB)
        dx = p["x"][:, idx] - x1[:, None]; dy = p["y"][:, idx] - y1[:, None]
        c, s_ = torch.cos(h1)[:, None], torch.sin(h1)[:, None]
        rel = torch.stack([dx * c + dy * s_, -dx * s_ + dy * c], -1)
        cd, cv = p["delta"][:, i1], p["v"][:, i1]
        if rows is None:
            self.rel, self.rel_v, self.cmd_d, self.cmd_v = rel, p["pv"], cd, cv
        else:
            self.rel = torch.where(rows[:, None, None], rel, self.rel)
            self.rel_v = torch.where(rows[:, None], p["pv"], self.rel_v)
            self.cmd_d = torch.where(rows, cd, self.cmd_d)
            self.cmd_v = torch.where(rows, cv, self.cmd_v)

    def state(self) -> torch.Tensor:
        r = self.env.last_result
        Bl = self.rel.shape[0]
        if r is not None and r.imu is not None and r.imu.shape[1] > 0:
            imu = r.imu[self.lid].float().mean(1)                  # gyro xyz, accel xyz
        else:
            imu = torch.zeros(Bl, 6, device=self.rel.device)
        return torch.cat([imu, self.cmd_d[:, None], self.cmd_v[:, None], self.rel.reshape(Bl, -1), self.rel_v], 1)

    def priv(self) -> torch.Tensor:
        """v1's critic extras and the true speed / steering (training only)"""
        st = self.env.sim.state[self.lid]
        return torch.cat([super().priv(), st[:, dyn.IVX:dyn.IVX + 1].float(), st[:, dyn.ISTEER:dyn.ISTEER + 1].float()], 1)


def load_actor2(ck: dict, device) -> Actor2:
    actor = Actor2(int(ck["beams"]), std0=float(ck["args"].get("std0", 0.6))).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


def plan_v2(actor: Actor2, obs: PlanObs2, lim: PS.Limits, sample: bool = False):
    """one decision of a v2 policy: (u, mean, log_std, v_hat, plan). The plan starts from the speed estimate and the last
    steering command."""
    mean, log_std, v_hat = actor(obs.scans, obs.state())
    u = mean + log_std.exp() * torch.randn_like(mean) if sample else mean
    p = PS.roll(torch.tanh(u), v_hat.detach().clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
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
