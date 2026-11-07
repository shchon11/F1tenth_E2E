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
    def __init__(self, beams: int, priv_dim: int, hidden: int = 256):
        super().__init__()
        self.enc = ScanEncoder(beams)
        self.norm = RunningNorm(STATE_DIM + priv_dim)
        self.mlp = nn.Sequential(nn.Linear(256 + STATE_DIM + priv_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
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
        self.actor = load_actor(ck, device)
        self.lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), device)
        self.obs = PlanObs(env, self.lim, None)
        Bl = int(self.lid.numel())
        self.cmd_d = torch.zeros(Bl, PS.N_STEP * PS.SUB, device=device)
        self.cmd_v = torch.zeros(Bl, PS.N_STEP * PS.SUB, device=device)
        self.kk = torch.zeros(env.B, PS.N_STEP, device=device)
        self.vv = torch.zeros(env.B, PS.N_STEP, device=device)
        self.first = torch.ones(Bl, dtype=torch.bool, device=device)
        self.last_ep = env.ep_step[self.lid].clone()
        self.session = None
        self.last_mean = None

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
            s = self.obs.state()
            mean, _ = self.actor(self.obs.scans, s)
            self.last_mean = mean
            st = env.sim.state[lid].clone()
            p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], self.lim)
            x, y, _ = plan_world(p, st)
            idx = PT_IDX.to(self.device)
            self.obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"], due)
            self.cmd_d = torch.where(due[:, None], p["cmd_d"], self.cmd_d)
            self.cmd_v = torch.where(due[:, None], p["cmd_v"], self.cmd_v)
            if self.mode == "map":
                k_, v_ = plan_segments(p, self.lim)
                rows = lid[due]
                self.kk[rows] = k_[due]; self.vv[rows] = v_[due]
                env.ext_plan_kv = (self.kk, self.vv)
            else:
                env.sim.ideal_vcap[lid[due]] = 1e3
        if self.mode == "ideal":
            jj = j[:, None]
            env.ideal_cmd[lid, 0] = self.cmd_d.gather(1, jj)[:, 0]
            env.ideal_cmd[lid, 1] = self.cmd_v.gather(1, jj)[:, 0]
        return torch.zeros(env.B, env.act_dim, device=self.device)
