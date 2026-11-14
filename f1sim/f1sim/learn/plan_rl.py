"""PPO for the redesigned planner (11-07, every choice agreed with the user step by step; see plan_steps, plan_reward,
plan_policy). A new, plain trainer -- none of state_rl's switches (spawn behind boxes, stall price,
exploration floors, correlated noise, KL leashes): the world, the plan format, the reward, and PPO.

* World: `configs/plan_world_korea25.json` -- Korea 2025 and 16 warped variants, the four cases (empty, opponent,
  boxes, boxes + opponent) about 25 % each, box layouts drawn per episode (1..all), half of them removed at 10-30 s,
  ForzaETH opponents at 0.4-1.2x, 40 s episodes.
* Every 0.1 s each learner car draws a plan and drives its first 0.1 s exactly (`GymEnv.enable_ideal`); contacts are the
  simulator's own.
* PPO: gamma 0.995 per decision (a ~20 s horizon, about three laps), lambda 0.95, 4 epochs x 8 minibatches, clip 0.2,
  lr 3e-4 -> 1e-4, exploration std 0.6 at the start and learned.
* `--hold-p` (11-09, agreed with the user): that share of new plans is driven 0.2-0.4 s (`--hold-len`) without a new
  one, from where the car is (`plan_steps.shift_plan`). Re-planning every 0.1 s, only a plan's first 0.1 s was ever
  driven, and its later points were a rough shape (0.3 / 0.5 s on, 12 / 44 cm off where the car went) that MAP and
  the real car's controller follow. The policy cannot tell which plan will be held; the critic is told how many
  decisions the plan in hand still drives; held decisions are left out of the policy loss and the advantage runs
  through them undiscounted by lambda. Drive time (evaluation, console, car) re-plans every 0.1 s as before.
* `--level-replay` (11-12, opt-in): that share of race resets replays a race this seat has had before, picked by what
  went wrong in it (`HitLevels`). Measured on plan25r u300 before it was built (`tools/level_repeat.py`, 512 races
  driven twice, only the exploration noise differing): a race with a hard contact has one again 59 % of the time,
  one without 13 %; the 20 % of races that scored highest hold 55-59 % of the second drive's hard contacts.
* `--branches 2` (11-13, the user: "2는 괜찮네."): two plans and a choice (`plan_policy.Actor2`). Training samples the
  choice from its probabilities, never below `--branch-floor` each, and keeps a sampled choice from decision to decision
  the way a held random number does (`--branch-keep`: the number is redrawn with 1 - keep; with it fixed the choice
  changes only when the probabilities cross it) -- a choice redrawn every 0.1 s would never drive one side long enough
  to learn what that side is worth. The policy's log-probability is the chosen plan's + the choice's.
* `--scene-replay` (11-13, the user: "1도 같이 해야하는거 아님?"): that share of the races restart from the state 1-3 s
  before a past hard contact and drive 5 s from it (`scene_replay`); the other races are as they were and are where the
  scenes come from. The regular statistics count the other races only.

usage: python -m f1sim.learn.plan_rl --name NAME [--updates 2000] [--envs 1024] [--seed 1]"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time

import torch
import torch.nn as nn

from .. import plan_steps as PS
from .. import dynamics as dyn
from .. import map_controller as M
from . import state_rl as SR
from .plan_policy import Actor, Critic, PlanObs, plan_world, tanh_logp, PT_IDX, Actor2, PlanObs2, FORMAT2
from .plan_policy import PlanObs3, FORMAT3, STATE_DIM3, plan3_models
from .. import plan_steps3 as P3
from ..gym_env import PRIV_V2_PROP_KEYS, PRIV_V2_PROP_SLOTS
from .plan_reward import PlanReward, SEEN_S
from .level_replay import LevelReplay
from .scene_replay import SceneReplay

WORLD = os.path.join(os.path.dirname(__file__), "configs", "plan_world_korea25.json")


def build(world: str, envs: int, seed: int, device, overrides=None, fmt: str = "v2"):
    a = argparse.Namespace(**json.load(open(world)))
    a.envs, a.seed, a.device = envs, seed, str(device)
    for k, v in (overrides or {}).items():
        setattr(a, k, v)
    # the scanner's place (`--lidar-mount-x` -> F1SIM_LIDAR_MOUNT_X) is read by `state_rl.build_env` itself
    env = SR.build_env(a, device)
    if fmt == "v3":
        # the plan knows the control delay (11-09): the training's execution goes through the same delay, lags and table;
        # set before the graphs are captured (the roll reads it then)
        _lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), device)
        env.enable_ideal(lag=True, turn=plan3_models(env, _lim, device)[1])
    else:
        env.enable_ideal()
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: print(f"[graph] {_t}", flush=True) if ("eager" in _t or "goes back" in _t or "captured in" in _t) else None)
    env.reset(seed=seed)
    lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), device)
    return env, lim, a


def track_boxes(env, goal, lid: torch.Tensor, K: int, R: float) -> torch.Tensor:
    """(B_l, K*4) the boxes ahead ALONG THE TRACK (11-10, the user, of the straight-line block that also lists the boxes behind
    a wall on the opposite straight: \"이건 아니지\"): live boxes whose walls-only distance to go (`plan_reward.GoalField`, the
    reward's own field -- no centerline) is 0..R m less than the car's (from -1 m: beside it), nearest along the track first:
    present, dx/5, dy/5 in the car's frame, the along-track distance/5 -- hidden ones too. The lap's wrap is taken care of."""
    po = env.procedural
    Bl = int(lid.numel())
    out = torch.zeros(Bl, K, 4, device=lid.device)
    if po is None or po.C == 0:
        return out.reshape(Bl, -1)
    st = env.sim.state[lid].float()
    tid = env.sim.tid[lid]
    pos = po.p_poses[lid, :, :2].float()                                   # (B_l, C, 2)
    C = pos.shape[1]
    r_car = goal.remaining(st[:, :2], tid)
    r_box = goal.remaining(pos.reshape(-1, 2), tid.repeat_interleave(C)).view(Bl, C)
    L = goal.lap[tid].float()[:, None]
    ds = torch.remainder(r_car[:, None] - r_box + 0.5 * L, L) - 0.5 * L     # ahead along the track, in (-L/2, L/2]
    live = (po.p_sid[lid] >= 0) & torch.isfinite(ds) & (ds > -1.0) & (ds <= R)
    d = pos - st[:, None, :2]
    c, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
    dx, dy = d[..., 0] * c + d[..., 1] * s_, -d[..., 0] * s_ + d[..., 1] * c
    blk = torch.stack([live.float(), dx / 5.0, dy / 5.0, torch.nan_to_num(ds) / 5.0], 2) * live[..., None].float()
    order = torch.where(live, ds, torch.full_like(ds, 1e6)).argsort(1)
    blk = blk.gather(1, order[..., None].expand(-1, -1, 4))
    n = min(K, C)
    out[:, :n] = blk[:, :n]
    return out.reshape(Bl, -1)


def track_opp(env, goal, lid: torch.Tensor, ahead: float, behind: float = 3.0) -> torch.Tensor:
    """(B_l, 5) the nearest other car ALONG THE TRACK within -behind..ahead m (11-10: the plans drove through slower cars ahead --
    the driving features explained only 0.21-0.47 of where they are and how fast they close, `opp_probe.py`): measured like
    `track_boxes` (no car behind a wall on the opposite straight), its seat picked by the smallest |along-track distance|:
    present, dx/5, dy/5 in the car's frame, the relative speed dvx/5, dvy/5 (the critic's `_opp_rel` arithmetic)."""
    Bl = int(lid.numel())
    out = torch.zeros(Bl, 5, device=lid.device)
    if env.sim.other_idx is None:
        return out
    st = env.sim.state.float()
    dx, dy, _, dvx, dvy, _, present = env._opp_rel(st)
    dx, dy, dvx, dvy, present = dx[lid], dy[lid], dvx[lid], dvy[lid], present[lid]
    o = env.sim.other_idx[lid]
    tid = env.sim.tid[lid]
    C = o.shape[1]
    r_car = goal.remaining(st[lid, :2], tid)
    r_o = goal.remaining(st[o.reshape(-1), :2], tid.repeat_interleave(C)).view(Bl, C)
    L = goal.lap[tid].float()[:, None]
    ds = torch.remainder(r_car[:, None] - r_o + 0.5 * L, L) - 0.5 * L
    ok = present & torch.isfinite(ds) & (ds > -behind) & (ds <= ahead)
    j = torch.where(ok, ds.abs(), torch.full_like(ds, 1e6)).argmin(1, keepdim=True)
    okj = ok.gather(1, j)[:, 0].float()
    sel = lambda v: v.gather(1, j)[:, 0] / 5.0 * okj
    return torch.stack([okj, sel(dx), sel(dy), sel(dvx), sel(dvy)], 1)


class SelfOpponents:
    """A share of the other cars is driven by frozen copies of the policy (11-12). Measured that day (`opp_encounters.py`,
    `opp_audit_scenes.py`): the four scripted kinds are one line follower (same line, same speed profile, 7-8 % off our lap
    at scale 1.0, no perception), the training mix leaves the car near ours at 2.8 m/s against our 4.7, and our car is
    passed 0.04 times a minute -- it never races a car of its own pace. A copy drives at our pace by construction and sees
    with its own LiDAR.

    Each race, with probability `share`, the other car is handed to one of `pool` (the learner's weights at earlier
    updates, `add`): its inputs are its own row's LiDAR / IMU (`PlanObs2` on that row), it draws the plan's mean, and the
    plan's first 0.1 s is driven exactly, like the learner's (`env.ideal_extra`). The scripted driver's command for such a
    row is computed and not used (its speed scale and injected events with it). `Driver` calls `plan` / `command` /
    `stepped` / `after` around a decision's control steps."""

    def __init__(self, env, lim, share: float, seed: int = 0, max_pool: int = 4, p_newest: float = 0.5, vcap=None):
        self.env, self.lim, self.share = env, lim, float(share)
        self.rows = torch.nonzero(~env.learner).flatten()
        dev = env.device
        Bo = int(self.rows.numel())
        self.obs = PlanObs2(env, lim, None, rows=self.rows)
        self.on = torch.zeros(Bo, dtype=torch.bool, device=dev)       # driven by a copy this race
        self.which = torch.zeros(Bo, dtype=torch.long, device=dev)    # by which of the pool
        # `vcap` (lo, hi) [m/s]: each copy-driven race draws a top speed for its copy (11-12: copies of our own pace gave
        # 0.00-0.02 passes a minute either way, `out/self_meas_summary.txt`). The cap is on the speed driven
        # (`sim.ideal_vcap`), not on the command: a command scaled down would shrink the speed decision after decision,
        # since a plan's speeds start from the car's own
        self.vcap_range = None if vcap is None else (float(vcap[0]), float(vcap[1]))
        self.cap = torch.full((Bo,), 1e3, device=dev)
        self.pool, self.tags, self.newest = [], [], -1
        self.max_pool, self.p_newest = int(max_pool), float(p_newest)
        self.gen = torch.Generator(device="cpu"); self.gen.manual_seed(int(seed) + 7717)
        self.p = None
        self.restarted = torch.zeros(Bo, dtype=torch.bool, device=dev)
        self.xy = env.sim.state[self.rows, :2].clone()
        env.ideal_extra = torch.zeros(env.B, dtype=torch.bool, device=dev)

    def add(self, actor, tag) -> None:
        """a frozen copy of `actor` joins the pool; a full pool gives up a random one of its older copies"""
        import copy
        a = copy.deepcopy(actor).eval()
        for q in a.parameters():
            q.requires_grad_(False)
        if len(self.pool) < self.max_pool:
            self.pool.append(a); self.tags.append(tag); self.newest = len(self.pool) - 1
        else:
            old = [k for k in range(len(self.pool)) if k != self.newest]
            k = old[int(torch.randint(len(old), (1,), generator=self.gen))] if old else self.newest
            self.pool[k], self.tags[k], self.newest = a, tag, k
        if len(self.pool) == 1:
            every = torch.arange(self.on.numel(), device=self.on.device)
            self.obs.reset_rows(every)
            self.draw(every)

    def draw(self, rows: torch.Tensor) -> None:
        """rows (into the opponent rows) whose race just started: a copy or the scripted driver, and which copy"""
        n = int(rows.numel())
        if n == 0:
            return
        dev = self.on.device
        if not self.pool:
            self.on[rows] = False
            return
        u = torch.rand(n, 3, generator=self.gen)
        up = torch.rand(n, generator=self.gen) if self.vcap_range is not None else None
        # a race with a name (level replay, `env.level_seed`) draws the same every time it is played: the replay is of the
        # race as it was, a copy at the wheel included (which of the pool's copies moves on with the pool)
        seed = getattr(self.env, "level_seed", None)
        if seed is not None:
            seed = seed[self.rows[rows]].tolist()
            for i, sd in enumerate(seed):
                if sd >= 0:
                    g = torch.Generator(device="cpu"); g.manual_seed(int(sd) + 7717)
                    u[i] = torch.rand(3, generator=g)
                    if up is not None:
                        up[i] = torch.rand(1, generator=g)
        u = u.to(dev)
        self.on[rows] = u[:, 0] < self.share
        k = (u[:, 2] * len(self.pool)).long().clamp_max(len(self.pool) - 1)
        self.which[rows] = torch.where(u[:, 1] < self.p_newest, torch.full_like(k, self.newest), k)
        if up is not None:
            lo, hi = self.vcap_range
            self.cap[rows] = lo + (hi - lo) * up.to(dev)

    @torch.no_grad()
    def plan(self) -> None:
        """before a decision's control steps: each copy-driven car's plan from its own inputs"""
        env, obs = self.env, self.obs
        Bo = int(self.on.numel())
        dev = self.on.device
        # a car put somewhere else since the last decision (`state_rl.OppRestart` runs after it): its scans, last plan and
        # memory are of the old place, and it would drive one decision on them
        xy = env.sim.state[self.rows, :2]
        moved = (xy - self.xy).norm(dim=1) > 2.0
        if bool(moved.any()):
            obs.reset_rows(torch.nonzero(moved).flatten())
            self.xy = xy.clone()
        live = self.on & ~env.sim.absent[self.rows] if getattr(env.sim, "absent_any", False) else self.on
        mean = torch.zeros(Bo, PS.ACT_DIM, device=dev)
        v_hat = torch.zeros(Bo, device=dev)
        if bool(live.any()):
            s = obs.state()
            for k, actor in enumerate(self.pool):
                idx = torch.nonzero(live & (self.which == k)).flatten()
                if idx.numel() == 0:
                    continue
                h = None
                if getattr(actor, "memory", 0):
                    if getattr(obs, "mem_h", None) is None or obs.mem_h.shape[1] != actor.memory:
                        obs.mem_h = torch.zeros(Bo, actor.memory, device=dev)
                    h = obs.mem_h[idx]
                m, _, vh = actor(obs.scans[idx], s[idx], h)
                if h is not None:
                    obs.mem_h[idx] = actor.h_out
                mean[idx] = m.float(); v_hat[idx] = vh.float()
        self.live = live
        self.p = PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, self.lim)
        self.restarted.zero_()
        env.sim.ideal_vcap[self.rows[live]] = self.cap[live]
        env.ideal_extra.zero_()
        env.ideal_extra[self.rows] = live

    def command(self, j: int) -> None:
        go = self.live & ~self.restarted
        z = torch.zeros_like(self.p["cmd_d"][:, j])
        self.env.ideal_cmd[self.rows, 0] = torch.where(go, self.p["cmd_d"][:, j], z)
        self.env.ideal_cmd[self.rows, 1] = torch.where(go, self.p["cmd_v"][:, j], z)

    def stepped(self, done: torch.Tensor) -> None:
        self.restarted |= done[self.rows]

    def after(self) -> None:
        obs = self.obs
        obs.push_scan()
        obs.set_plan2(self.p)
        # a car put somewhere else without a new race (`state_rl.OppRestart`): its scans and last plan are of the old place
        xy = self.env.sim.state[self.rows, :2]
        moved = (xy - self.xy).norm(dim=1) > 2.0
        self.xy = xy.clone()
        new = self.restarted | moved
        if bool(new.any()):
            obs.reset_rows(torch.nonzero(new).flatten())
        if bool(self.restarted.any()):
            self.draw(torch.nonzero(self.restarted).flatten())

    def share_on(self) -> float:
        """the share of the other cars in a race that a copy drives now"""
        pres = ~self.env.sim.absent[self.rows] if getattr(self.env.sim, "absent_any", False) else torch.ones_like(self.on)
        return float((self.on & pres).float().sum() / pres.float().sum().clamp_min(1.0))


class HitLevels(LevelReplay):
    """Level replay (`learn.level_replay`, one buffer per race seat) scored by what went wrong in the race instead of by
    |advantage|: per decision played, the learner's hard contacts (walls, boxes, and cars the reward charges: seen
    within `plan_reward.SEEN_S`) + 0.05 for a decision that ends standing (2 s standing = one hard contact). This is
    the score `tools/level_repeat.py` ranked the races by. A race replayed keeps half of its old score (one clean
    drive of a race the car hit in is no proof), so a race is replayed until it is driven clean a few times.

    `add` after every decision (sums per seat on the GPU), `choose` (called by the env when a race resets whole) hands
    the seat's sums over first. A race that began before the replay was switched on has no name and is not stored."""

    STAND = 0.05

    def __init__(self, env, lid, **kw):
        super().__init__(env.B // env.M, **kw)
        self.seat = env.race[lid]
        self.sum = torch.zeros(self.n, device=env.device)
        self.cnt = torch.zeros(self.n, device=env.device)
        self.ended, self.ended_hit = 0, 0                      # races handed over / of them with a score above 0

    def add(self, score: torch.Tensor, live: torch.Tensor) -> None:
        """score (B_l,) of the decision just driven; live (B_l,) bool = rows whose race did not reset in it"""
        self.sum.index_add_(0, self.seat[live], score[live].float())
        self.cnt.index_add_(0, self.seat[live], torch.ones_like(score[live], dtype=torch.float))

    def choose(self, seat: int) -> int:
        n = int(self.cnt[seat])
        if self.cur[seat] >= 0 and n > 0:
            sm = float(self.sum[seat])
            self.accumulate(seat, self.cur[seat], sm, n)
            self.ended += 1; self.ended_hit += int(sm > 0)
        self.sum[seat] = 0.0; self.cnt[seat] = 0.0
        return super().choose(seat)

    def _commit(self, seat: int) -> None:
        s = self.cur[seat]
        if s < 0 or self.play_cnt[seat] == 0:
            return
        b = self.buf[seat]
        new = self.play_sum[seat] / self.play_cnt[seat]
        b[s] = [0.5 * b[s][0] + 0.5 * new if s in b else new, self.tick]
        if len(b) > self.size:
            del b[min(b, key=lambda k: b[k][0])]


class Driver:
    """drives the learner rows one decision: plan from the action, the 4 control steps, the reward"""

    def __init__(self, env, lim, pr: PlanReward, obs: PlanObs):
        self.env, self.lim, self.pr, self.obs = env, lim, pr, obs
        self.lid = env.learner_ids
        self.opp = None                                        # `SelfOpponents`: other cars driven by copies of the policy
        self.scene = None                                      # `SceneReplay`: told where in the decision each step is
        self.act0 = torch.zeros(env.B, env.act_dim, device=env.device)
        from ..lidar import HIT_CAR
        self._hit_car = HIT_CAR

    def decision(self, a: torch.Tensor, p=None, charge=None):
        """drive one decision: `p` a plan already made (v2: from the policy's speed estimate and last command), else
        made from `a` at the car's true state (v1). `charge` (B_l,) bool: rows whose plan is new this decision -- the
        plan-hit cost is charged once, when a plan is drawn (a held plan's later decisions pay none)"""
        env, lid = self.env, self.lid
        st = env.sim.state[lid].clone()
        if p is None:
            p = PS.roll(a, st[:, dyn.IVX], st[:, dyn.ISTEER], self.lim)
        x, y, psi = plan_world(p, st)
        pc = self.pr.plan_cost(x, y, psi, p["v"])
        idx = PT_IDX.to(a.device)
        pts = torch.stack([x[:, idx], y[:, idx]], -1)
        env.sim.ideal_vcap[lid] = 1e3
        restarted = torch.zeros(lid.numel(), dtype=torch.bool, device=a.device)
        if self.opp is not None:
            self.opp.plan()
        for j in range(PS.SUB):
            cmd_d = torch.where(restarted, torch.zeros_like(p["cmd_d"][:, j]), p["cmd_d"][:, j])
            cmd_v = torch.where(restarted, torch.zeros_like(p["cmd_v"][:, j]), p["cmd_v"][:, j])
            env.ideal_cmd[lid, 0] = cmd_d; env.ideal_cmd[lid, 1] = cmd_v
            if self.opp is not None:
                self.opp.command(j)
            if self.scene is not None:
                self.scene.to_boundary = PS.SUB - 1 - j        # a scene starts on a decision, never inside one
            _, _, te, trc, _ = env.step(self.act0)
            restarted |= (te | trc)[lid]
            if self.opp is not None:
                self.opp.stepped(te | trc)
            r = env.last_result
            self.pr.control_step((r.scan_type[lid] == self._hit_car).any(1), (te | trc)[lid])     # a beam ended on a car
        price, hit = pc["price"], pc["hit"].float()
        if charge is not None:
            price, hit = torch.where(charge, price, torch.zeros_like(price)), hit * charge.float()
        out = self.pr.decision(pts, price, restarted)
        out["plan_hit_price"] = out.pop("plan_hit")
        out["plan_hit"] = hit
        self.obs.push_scan()
        self.obs.set_plan(pts, p["pv"])
        if hasattr(self.obs, "set_plan2"):
            self.obs.set_plan2(p)
        if bool(restarted.any()):
            rows = torch.nonzero(restarted)[:, 0]
            self.obs.reset_rows(rows)
        if self.opp is not None:
            self.opp.after()
        out["restarted"] = restarted
        return out, p, (x, y, psi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--world", default=WORLD)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=64, help="decisions per rollout (0.1 s each)")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="minibatches whose gradients are summed into one optimizer step "
                    "(1 = a step per minibatch, as before; = --minibatches: one step per epoch over the whole rollout)")
    ap.add_argument("--accum-actor", type=int, default=1, help="as --accum for the actor only: the critic keeps its own optimizer "
                    "and steps every minibatch (11-14: per-minibatch actor steps flip sign along the direction the whole rollout "
                    "points to and the actor churns ~1.0 per update; one step per epoch halved the churn's damage)")
    ap.add_argument("--accum-actor-auto", action="store_true", help="the actor's minibatches per step chosen every update from the "
                    "policy gradient's noise scale (McCandlish et al. 2018: B_noise = tr(Sigma)/|G|^2, the decisions a step must "
                    "average for its signal to match its noise), measured on the rollout before the epochs and smoothed over updates: "
                    "the smallest power of two of minibatches covering it, up to --minibatches (11-14: plan25y's noise scale grew from "
                    "~5k decisions at u25 to 25k-1.4M at u150; a minibatch is 4k)")
    ap.add_argument("--noise-probe-every", type=int, default=1,
                    help="with --accum-actor-auto: measure the noise scale every N updates (it is smoothed over updates and moves "
                         "slowly; the measurement is one more forward/backward over the whole rollout, ~1/5 of an update)")
    ap.add_argument("--opp-every", type=int, default=0,
                    help="(11-14) the scripted opponents' planners every N control steps, their command held in between "
                         "(GymEnv.opp_every; 0 = leave the env's, F1SIM_OPP_EVERY or 1). Exam on plan26f u150: opponents' speed, "
                         "stops, restarts and our results unchanged at 2 and 4; rollout -37 % at 4")
    ap.add_argument("--accum-auto-frac", type=float, default=0.0, help="with --accum-actor-auto: instead of the smallest power of two "
                    "covering the noise scale, the largest one whose decisions stay within FRAC x the noise scale (1 below that); "
                    "0.5 = steps a little smaller than the noise scale, the sample-efficient side (11-14: plan26e's covering rule "
                    "took 2-4 minibatches at 6-13k and its u25 drove half as far as plan25y's)")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--std0", type=float, default=0.6)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--format", default="v2", choices=("v1", "v2", "v3"),
                    help="v2 (11-08): raw sensors only, the plan starts from the policy's own speed estimate")
    ap.add_argument("--vest-coef", type=float, default=1.0, help="v2: weight of the speed estimate's loss [per (m/s)^2]")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--init", default="", help="v3: start the actor from this v2 checkpoint (its 36 inputs + the servo angle at 0), "
                    "the critic's scan encoder too")
    ap.add_argument("--noise-corr", type=float, default=0.0,
                    help="exploration noise that stays over decisions: AR(1) coefficient per decision (0 = a fresh draw every "
                         "decision; 0.8 = half gone after 0.3 s). The size of the noise at one decision is unchanged")
    ap.add_argument("--dump-rollout", default="", help="measurement only (with --init --init-critic: a checkpoint's own training "
                    "batches): every update's rollout is saved here as u<N>.npz -- per decision the first step's noise, the "
                    "steering it gave, advantage, value, reward parts, hard contacts, the boxes ahead, the pose -- and NO update "
                    "is made")
    ap.add_argument("--track-steps", default="", help="measurement (11-14): an npz of directions in the actor's weight space (names, "
                    "sizes, dir_<key> unit vectors over its trainable weights, + = the way that drove better). Every optimizer step's "
                    "move of the actor is projected on each, and in the first epoch each loss term's gradient too; saved per update "
                    "as <run>/track/t<N>.npz. Training itself is unchanged")
    ap.add_argument("--actor-freeze", type=int, default=0, help="updates in which only the critic learns (a started actor)")
    ap.add_argument("--plan-hit-share", type=float, default=0.1, help="the plan-hit cost's share of a contact (0 = off)")
    ap.add_argument("--stall-s", type=float, default=0.0,
                    help="the stall price of 11-06, back (11-13): [s] per second the progress over the last --stall-window "
                         "falls short of --stall-vmin, x the shortfall's share; 0 = off")
    ap.add_argument("--stall-vmin", type=float, default=1.0, help="[m/s] --stall-s charges a progress rate below this")
    ap.add_argument("--stall-window", type=float, default=2.0, help="[s] --stall-s: the progress window")
    ap.add_argument("--contact-scale", type=float, default=1.0,
                    help="a factor on every contact's time penalty, (0.5 s + 1 s per m/s of impact) x this (11-13; 1 = the 11-07 prices)")
    ap.add_argument("--plan-hit-s", type=float, default=1.0, help="the plan-hit cost looks this far into the plan [s]")
    ap.add_argument("--hold-p", type=float, default=0.0,
                    help="v2/v3: the share of new plans driven longer than 0.1 s without a new one (11-09, the user: 0.2)")
    ap.add_argument("--hold-len", default="2,4", help="a held plan drives this many decisions, drawn uniformly (2,4 = 0.2-0.4 s)")
    ap.add_argument("--hold-clear", type=float, default=0.0,
                    help="hold only when no other car and no box is within this many metres; one that comes inside it ends a "
                         "hold at once (11-09, the user: racing needs the reaction -- holds near them teach timid plans). 0 = anywhere")
    ap.add_argument("--init-critic", action="store_true", help="with --init: the critic too, when its inputs match (a run continued)")
    ap.add_argument("--tail-consistency", type=float, default=0.0,
                    help="v2: only the plan's first 0.1 s is the RL action (its noise and log-probability); the other 18 outputs are "
                         "taught, with this weight, to say what the NEXT plan says for the same moments (11-09, the user chose (가): "
                         "the previous plan's 0.1-1.0 s pulled toward the new plan's 0-0.9 s, never the other way). The output layer "
                         "alone learns it: nothing that drives is touched. 0 = off (all 20 outputs are actions)")
    ap.add_argument("--tail-target", default="next", choices=("next", "driven"),
                    help="what the plan's 0.2-1.0 s points learn: 'next' = the next plan's same moments (11-09 (가)); 'driven' "
                         "(11-14, the user chose 2) = where the car actually was 0.2-1.0 s after the decision (positions and "
                         "speeds from the rollout; points past a restart or the rollout's end are left out). 11-13: with 'next' the "
                         "1.0 s point was 63 cm beside the driven path while a fresh net on the same inputs got 5 cm")
    ap.add_argument("--tail-trunk", action="store_true",
                    help="the tail's loss also trains the trunk, scan encoder and memory (default: the output layer alone on "
                         "detached features). Logged per update: its trunk gradient / the policy's")
    ap.add_argument("--tail-trunk-ratio", type=float, default=0.0,
                    help="with --tail-trunk: keep the tail's gradient on the shared weights (encoder, trunk, memory) at about this "
                         "fraction of the policy's, by scaling only what flows into them (the output layer always learns the tail "
                         "fully); measured each update, adjusted between updates. 0 = no scaling (11-14 small test at weight 1: "
                         "0.7-3.4x the policy's)")
    ap.add_argument("--memory", type=int, default=0,
                    help="v2: a GRU of this size in the actor (11-09, the user: \"A1\" -- adapt by remembering); 0 = none")
    ap.add_argument("--mem-chunk", type=int, default=32, help="the memory's update replays sequences of this many decisions")
    ap.add_argument("--mem-front", action="store_true",
                    help="with --memory: the memory before the trunk (`Actor2.mem_front`; 11-11, the user: \"앞쪽으로 가자\") -- the"
                         " GRU reads the LiDAR encoding and the state, the trunk reads them plus the memory's state. From scratch,"
                         " or continued from a run that had it")
    ap.add_argument("--plan-lat-margin", type=float, default=1.0,
                    help="the plans' tyre limit as this share of the steering table's highest steady-state lateral acceleration"
                         " (11-10, the user: \"ㅇㅋ\" to 0.95): at 1 the policy drove >= 95 %% of it 29 %% of the time, where the table's"
                         " flat top needs 18 -> 24 deg (lock) for the last 5 %% -- a real car's follower had no steering left to correct"
                         " with and the car wobbled. Every Limits of this run (and its evaluations, read from the checkpoint) use it")
    ap.add_argument("--box-range", type=float, default=0.0,
                    help="> 0: the boxes ahead -- the critic's box block and --mem-aux's target -- counted ALONG THE TRACK within this many"
                         " metres (`track_boxes`) instead of the straight-line 12 m that also lists boxes behind a wall (11-10). 0 = as before")
    ap.add_argument("--smooth-reg", type=float, default=0.0,
                    help="with --memory: the weight of a cost on the plans' zigzag (11-10, the user approved a small cost on the"
                         " executed first step's steering change between decisions: the car wobbled more and more in the console, the"
                         " plans' first-step steering reversed every ~8th decision, 14 %% at u300, and the tyre table near the limit made"
                         " that 3-4x). On the policy's mean, along each replayed sequence: the change of the first step's steering"
                         " action between consecutive decisions (a reversal of the steering rate, not the rate itself, so a steady"
                         " turn-in costs nothing), in lateral-jerk units v^2 ds/L (ds = the servo's 0.1 s range), over 4 m/s^2, squared."
                         " Not in the reward: there the exploration noise's own reversals were twice the policy's and would pull the"
                         " noise down. 0 = off")
    ap.add_argument("--aux-opp", type=float, default=0.0,
                    help="> 0, with --mem-aux: the memory's side task also says where the nearest other car is along the track"
                         " (-3 .. this many m) and how fast it closes (`track_opp`; 11-10). 0 = boxes only")
    ap.add_argument("--grad-split", action="store_true",
                    help="a measurement, nothing learns from it: in each update's first pass, what the policy loss and the speed "
                         "estimate's loss each send to the memory's weights (progress.jsonl: g_mem_*). 11-12: the GRU has these "
                         "two teachers and nobody had measured which one moves it")
    ap.add_argument("--mem-aux", type=float, default=0.0,
                    help="with --memory: the weight of a side task for the memory alone -- from its state, say where the boxes ahead are"
                         " (the critic's privileged block: the nearest 3 live boxes not behind the car within 12 m, hidden ones too) every"
                         " decision (11-10, the user: \"기억은 필요해 ... 사각지대에 장애물이 있어서 사고나는거\"; --mem-chunk 128 alone left"
                         " the memory unused for 100 updates). The LiDAR features feed it detached; driving never reads it. 0 = off")
    ap.add_argument("--mem-value", type=float, default=0.0,
                    help="with --memory: the weight of a value read-out of the memory's state (`Actor2.mem_value`; 11-12, the user:"
                         " \"ㅇㅋ\") -- taught this update's returns, each update's scaled to mean 0 and spread 1, so the logged loss is"
                         " the share of the returns' spread it does not explain. The features feed it detached; driving never reads"
                         " it; the privileged critic still makes the advantages. 0 = off")
    ap.add_argument("--mem-keep-s", type=float, default=0.0,
                    help="with --memory, from scratch: the GRU's update gate starts holding, keep times spread over 0.2 s .. this many"
                         " seconds (`Actor2.keep_init`). 0 = PyTorch's default (every value ~0.2 s)")
    ap.add_argument("--mem-keep-log", action="store_true", help="with --mem-keep-s: as many values per octave of keep time")
    ap.add_argument("--speed-std-floor", type=float, default=0.0,
                    help="v2: the exploration noise's floor on the first step's speed output (11-09: it had shrunk to 0.17)")
    ap.add_argument("--removal-p", type=float, default=-1.0,
                    help="the share of races whose obstacles are taken away mid-episode (-1 = the world's `removal_p`; 0 = none). "
                         "11-09: plan_rl never ran it, so no plan policy had seen the competition's removal")
    ap.add_argument("--opp-restart-s", type=float, default=-1.0,
                    help="an opponent that has stood still this long restarts half a lap away (-1 = the world's `opp_restart_s`; "
                         "0 = never). 11-11: the opponents stood against boxes 18 % of a box race (`state_rl.OppRestart`)")
    ap.add_argument("--level-replay", type=float, default=0.0,
                    help="share of race resets that replay a race this seat had before, picked by its hard contacts and "
                         "standing time (`HitLevels`); 0 = every race new")
    ap.add_argument("--level-replay-size", type=int, default=24, help="races kept per seat")
    ap.add_argument("--level-replay-from", type=int, default=0,
                    help="switch the replay on at this update: before the car laps, the standing-time score ranks the "
                         "races by how badly the policy drives everywhere, not by the race")
    ap.add_argument("--self-opp", type=float, default=0.0,
                    help="share of races whose other car is driven by a frozen copy of this policy (`SelfOpponents`, v2); "
                         "0 = the world's scripted cars only. 11-12: the scripted kinds are one line follower 7-8 %% off our "
                         "lap, and the car near ours runs 2.8 m/s against our 4.7 (`tools/opp_encounters.py`)")
    ap.add_argument("--self-opp-every", type=int, default=25, help="a copy of the learner joins the pool every this many updates")
    ap.add_argument("--self-opp-from", type=int, default=0,
                    help="the first copy joins at this update (0 = at the first --self-opp-every); a copy of a policy that "
                         "does not lap yet is a car standing on the track")
    ap.add_argument("--self-opp-pool", type=int, default=4, help="copies kept; a full pool gives up a random one of its older copies")
    ap.add_argument("--self-opp-newest", type=float, default=0.5, help="share of the copy-driven cars that get the newest copy")
    ap.add_argument("--self-opp-vcap", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="each copy-driven race draws its copy's top speed from [LO, HI] m/s (a cap on the speed driven); "
                         "off = copies at their own pace, which gives almost no passing either way")
    ap.add_argument("--lidar-mount-x", type=float, default=0.0,
                    help="the scanner's place ahead of the simulator's pose (the CoG) in metres; 0 = the config's 0.297, the nose "
                         "tip every policy up to plan25r saw from. The car's own is 0.12555 (`params.LIDAR_MOUNT_X_COG`: rear "
                         "axle + 0.297). Stored in the checkpoint; tools take it from F1SIM_LIDAR_MOUNT_X")
    ap.add_argument("--track-scales", default="",
                    help="size variants added to the world's tracks: per cents, comma-separated, or 'korea25' = "
                         "`state_rl.TRACK_SCALES_25` (0.60-0.92 and 1.08-1.20 of a warp each; 11-12, the user: \"트랙 변형도 스케일 "
                         "자체가 변하는것도\"). '' = the world's `track_scales` (none in plan_world_korea25)")
    ap.add_argument("--branches", type=int, default=1, choices=(1, 2),
                    help="2 = two plans and a choice between them (`Actor2.branches`); --format v2 with --tail-consistency")
    ap.add_argument("--branch-floor", type=float, default=0.1,
                    help="training never samples either plan with less than this probability (both keep being tried)")
    ap.add_argument("--branch-keep", type=float, default=0.9,
                    help="per decision, the probability that the random number the choice is sampled with stays the same")
    ap.add_argument("--scene-replay", type=float, default=0.0,
                    help="share of the races that restart from the state before a past hard contact (`scene_replay`); 0 = off")
    ap.add_argument("--scene-lag", type=int, default=20,
                    help="a scene is the state this many decisions before the contact, drawn per scene from 0.5x to 1.5x of it")
    ap.add_argument("--scene-size", type=int, default=1024, help="scenes kept (the oldest make room)")
    ap.add_argument("--scene-episode-s", type=float, default=5.0, help="[s] an episode that starts from a scene lasts this long")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dist-world", type=int, default=1, help="(11-14) processes training ONE policy together, one per GPU, each "
                    "driving its own cars (--envs per process: give the slower GPU fewer); every optimizer step averages their "
                    "gradients weighted by decisions, the observation statistics and return scale are merged after each rollout. "
                    "Start one process per GPU with the same arguments except --dist-rank / --envs / CUDA_VISIBLE_DEVICES")
    ap.add_argument("--dist-rank", type=int, default=0, help="0 logs and saves; others write only to <run>/rank<N>/")
    ap.add_argument("--dist-init", default="tcp://127.0.0.1:29533")
    ap.add_argument("--dist-backend", default="gloo", choices=("gloo", "nccl"))
    ap.add_argument("--dist-same-seed", action="store_true", help="test only: every process drives the same seed")
    ap.add_argument("--runs", default=os.path.expanduser("~/f1sim_runs"))
    args = ap.parse_args(argv)
    dist = None
    if args.dist_world > 1:
        import datetime
        import torch.distributed as dist
        if (args.dump_rollout or args.track_steps or args.level_replay > 0 or args.self_opp > 0 or args.branches == 2
                or args.hold_p > 0 or args.scene_replay > 0 or args.accum > 1):
            raise SystemExit("--dist-world: not with --dump-rollout / --track-steps / --level-replay / --self-opp / --branches 2 / "
                             "--hold-p / --scene-replay / --accum (use --accum-actor)")
        dist.init_process_group(args.dist_backend, init_method=args.dist_init, rank=args.dist_rank, world_size=args.dist_world,
                                timeout=datetime.timedelta(hours=24))    # gpu_guard3 may pause both while the console runs
        if args.dist_rank > 0 and not args.dist_same_seed:
            args.seed += 7919 * args.dist_rank          # its own cars, boxes, opponents and noise; the weights come from rank 0
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    run = os.path.join(args.runs, args.name)
    if args.dist_rank > 0:
        run = os.path.join(run, f"rank{args.dist_rank}")
    if os.path.exists(os.path.join(run, "progress.jsonl")):
        raise SystemExit(f"{run} already has a run: pick an unused name")
    os.makedirs(run, exist_ok=True)
    if args.plan_lat_margin != 1.0:
        os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(args.plan_lat_margin)   # every Limits built in this process
    if args.lidar_mount_x > 0:
        os.environ["F1SIM_LIDAR_MOUNT_X"] = str(args.lidar_mount_x)
    else:
        os.environ.pop("F1SIM_LIDAR_MOUNT_X", None)                       # the checkpoint says 0: build it that way
    os.environ.pop("F1SIM_WORLD_TRACKS", None)                            # a measurement tool's track choice never trains
    ov = {}
    if args.track_scales:
        ov["track_scales"] = (list(SR.TRACK_SCALES_25) if args.track_scales == "korea25"
                              else [int(x_) for x_ in args.track_scales.split(",")])
    env, lim, world = build(args.world, args.envs, args.seed, dev, overrides=ov, fmt=args.format)
    if args.opp_every > 0:
        env.opp_every = int(args.opp_every)
    lid = env.learner_ids; Bl = int(lid.numel())
    pr = PlanReward(env, args.gamma, dev)
    pr.plan_hit_share = float(args.plan_hit_share)
    pr.contact_scale = float(args.contact_scale)
    pr.stall_s, pr.stall_vmin, pr.stall_window_s = float(args.stall_s), float(args.stall_vmin), float(args.stall_window)
    pr.plan_hit_n = None if args.plan_hit_s >= PS.N_STEP * PS.DT else int(round(args.plan_hit_s / PS.DT * PS.SUB))
    v3 = args.format == "v3"
    v2 = args.format == "v2" or v3                     # v3 is v2 with the delay-aware plan and the servo angle
    obs = (PlanObs3 if v3 else PlanObs2 if v2 else PlanObs)(env, lim, pr.goal)
    two = args.branches == 2
    if two:
        if not v2 or v3 or args.tail_consistency <= 0 or args.hold_p > 0 or args.self_opp > 0 or args.init:
            raise SystemExit("--branches 2 needs --format v2 with --tail-consistency, from scratch, without --hold-p / --self-opp")
        if not 0.0 <= args.branch_floor < 0.5:
            raise SystemExit("--branch-floor: in [0, 0.5)")
        obs.prev_branch = torch.zeros(Bl, device=dev)          # the state's last input: the plan chosen last (+1 / -1 / 0)
    act3, turn3 = plan3_models(env, lim, dev) if v3 else (None, None)
    drv = Driver(env, lim, pr, obs)
    if args.self_opp > 0:
        if not v2 or v3 or env.M < 2:
            raise SystemExit("--self-opp needs --format v2 and a world with another car")
        drv.opp = SelfOpponents(env, lim, args.self_opp, seed=args.seed, max_pool=args.self_opp_pool, p_newest=args.self_opp_newest,
                                vcap=args.self_opp_vcap)
    hold = args.hold_p > 0
    if hold and not v2:
        raise SystemExit("--hold-p needs --format v2 or v3")
    h_lo, h_hi = (int(x_) for x_ in args.hold_len.split(","))
    hold_left = torch.zeros(Bl, dtype=torch.long, device=dev)      # decisions a held plan still drives after this one
    hold_k = torch.zeros(Bl, dtype=torch.long, device=dev)         # whole steps of the held plan already driven
    p_held = None
    others = env.sim.other_idx[lid] if env.M > 1 else None

    def near_threat() -> torch.Tensor:
        """(Bl,) another car of the race or a live box within `--hold-clear` metres of the car"""
        st_ = env.sim.state[lid, :2]
        near = torch.zeros(Bl, dtype=torch.bool, device=dev)
        if others is not None:
            d = (env.sim.state[others, :2] - st_[:, None, :]).norm(dim=-1)
            if getattr(env.sim, "absent_any", False):
                d = torch.where(env.sim.absent[others], torch.full_like(d, float("inf")), d)
            near |= d.min(1).values < args.hold_clear
        if env.procedural is not None:
            near |= env.procedural.clearance(st_, env.sim.eid[lid]) < args.hold_clear
        return near
    # the critic is told how many decisions the plan in hand still drives (the policy is not: it cannot know which plan
    # will be held, so every plan's later steps have to be ones the car can drive)
    priv_h = (lambda: torch.cat([obs.priv(), hold_left[:, None].float() / max(1, h_hi - 1)], 1)) if hold else obs.priv
    if args.box_range > 0:
        # the critic's box block, along the track: find its columns (they must equal the straight-line block now), then refill them
        _pv0, _pb0 = obs.priv(), env._prop_block(env.sim.state)[lid].float()
        _w = _pb0.shape[1]
        _hit = [j for j in range(_pv0.shape[1] - _w + 1) if torch.allclose(_pv0[:, j:j + _w].float(), _pb0, atol=1e-5)]
        if len(_hit) != 1 or float(_pb0.abs().sum()) == 0.0:
            raise SystemExit(f"--box-range: the critic's box block not found unambiguously ({_hit})")
        _P0, _priv_base = _hit[0], priv_h
        def priv_h():
            pv = _priv_base()
            pv[:, _P0:_P0 + _w] = track_boxes(env, pr.goal, lid, PRIV_V2_PROP_SLOTS, args.box_range)
            return pv
        print(f"[plan_rl] the boxes ahead along the track within {args.box_range:g} m (critic columns {_P0}-{_P0 + _w - 1}, "
              f"--mem-aux target)", flush=True)
    priv_dim = int(priv_h().shape[1])
    # the competition's sessions: the obstacles stay where they are for a while, then go (`state_rl.PropRemoval`, per race,
    # at a time drawn in removal_s after the episode's start)
    rm_p = float(getattr(world, "removal_p", 0.0)) if args.removal_p < 0 else float(args.removal_p)
    rm_s = tuple(getattr(world, "removal_s", (10.0, 30.0)))
    remover = SR.PropRemoval(env, rm_p, float(rm_s[0]), float(rm_s[1]), seed=args.seed) if rm_p > 0 else None
    if args.opp_restart_s >= 0:
        world.opp_restart_s = float(args.opp_restart_s)
    restarter = SR.opp_restart(env, world, PS.DT)
    levels = (HitLevels(env, lid, p_replay=args.level_replay, size=args.level_replay_size, seed=args.seed)
              if args.level_replay > 0 else None)
    scene = None
    if args.scene_replay > 0:
        if not v2 or v3 or levels is not None or args.self_opp > 0 or args.hold_p > 0:
            raise SystemExit("--scene-replay needs --format v2, without --level-replay / --self-opp / --hold-p")
        if not torch.equal(lid, torch.arange(Bl, device=dev) * env.M):
            raise SystemExit("--scene-replay: the learner is expected in the first seat of every race")
        scene = SceneReplay(env, args.scene_replay, args.scene_size, int(round(args.scene_episode_s / PS.DT)) * PS.SUB,
                            seed=args.seed)
        env.scene_replay = scene
        drv.scene = scene
    lag_lo, lag_hi = max(args.scene_lag // 2, 1), args.scene_lag + args.scene_lag // 2
    hist = collections.deque(maxlen=lag_hi + 1)                    # the last decisions' states, the newest last
    since = torch.zeros(Bl, dtype=torch.long, device=dev)          # decisions driven since the row's last restart
    eps_n = torch.randn(Bl, PS.ACT_DIM, device=dev)                # the exploration noise that stays (--noise-corr)
    cool = torch.full((Bl,), 10 ** 6, dtype=torch.long, device=dev)   # ... since a scene was last taken from the row
    rep_hit = torch.zeros(Bl, dtype=torch.bool, device=dev)        # the scene episode in progress has had a hard contact
    # the rows the regular statistics are taken from: every race that never restarts from a scene
    ordm = ~scene.races if scene is not None else torch.ones(Bl, dtype=torch.bool, device=dev)
    n_ord = int(ordm.sum())
    ar_l = torch.arange(Bl, device=dev)
    br_u = torch.rand(Bl, device=dev)                              # the number each row's choice of plan is sampled with
    br_prev = torch.zeros(Bl, dtype=torch.long, device=dev); arg_prev = torch.zeros(Bl, dtype=torch.long, device=dev)
    branch_p = lambda lg: args.branch_floor + (1.0 - 2.0 * args.branch_floor) * torch.softmax(lg, 1)
    if args.memory and (not v2 or v3):
        raise SystemExit("--memory needs --format v2")
    if args.mem_front and not args.memory:
        raise SystemExit("--mem-front needs --memory")
    aux_on = args.mem_aux > 0
    if aux_on and not args.memory:
        raise SystemExit("--mem-aux needs --memory")
    memv_on = args.mem_value > 0
    if (memv_on or args.mem_keep_s > 0) and not args.memory:
        raise SystemExit("--mem-value / --mem-keep-s need --memory")
    if args.mem_keep_s > 0 and args.init:
        raise SystemExit("--mem-keep-s is a starting value: it has no meaning with --init (the checkpoint's GRU is loaded)")
    AUXK = PRIV_V2_PROP_SLOTS
    AUXB = AUXK * len(PRIV_V2_PROP_KEYS)                            # the boxes' part of the read-out; the car's 5 follow
    aux_opp = aux_on and args.aux_opp > 0
    actor = (Actor2(obs.beams, std0=args.std0, state_dim=STATE_DIM3) if v3 else
             Actor2(obs.beams, std0=args.std0, memory=args.memory, aux_dim=(AUXB + (5 if aux_opp else 0)) if aux_on else 0,
                    mem_front=args.mem_front, mem_value=memv_on, mem_keep_s=args.mem_keep_s,
                    mem_keep_log=args.mem_keep_log, branches=args.branches) if v2
             else Actor(obs.beams, std0=args.std0)).to(dev)
    tailc = args.tail_consistency > 0
    if tailc and (not v2 or v3 or args.hold_p > 0):
        raise SystemExit("--tail-consistency needs --format v2 and no holding")
    rl_mask = torch.zeros(PS.ACT_DIM, dtype=torch.bool, device=dev)
    rl_mask[0] = True; rl_mask[PS.N_STEP] = True                   # the first step's steering and speed
    rl_idx = torch.nonzero(rl_mask)[:, 0]
    if args.speed_std_floor > 0:
        actor.log_std_floor[PS.N_STEP] = math.log(args.speed_std_floor)        # the first step's speed: the one driven
    if args.memory:
        obs.mem_h = torch.zeros(Bl, args.memory, device=dev)
    critic = Critic(obs.beams, priv_dim, state_dim=int(obs.state().shape[1])).to(dev)
    if args.init:
        # 11-09: every v3 run from scratch learned to stand still (the lagging car crashed while learning to drive); v3's
        # commands take v2's step sizes, so a v2 policy's outputs mean the same commands -- start from one, RL on from there
        ck0 = torch.load(args.init, map_location=dev, weights_only=False)
        if bool((ck0.get("args") or {}).get("mem_front", False)) != bool(args.mem_front):
            raise SystemExit("--init: the checkpoint's memory sits elsewhere (--mem-front differs); train from scratch")
        sd = dict(ck0["actor"])
        w = sd["trunk.0.weight"]
        sd["trunk.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], actor.trunk[0].in_features - w.shape[1], device=w.device)], 1)
        for k_, fill in (("norm.mean", 0.0), ("norm.var", 1.0)):
            v_ = sd[k_]
            sd[k_] = torch.cat([v_, torch.full((actor.norm.mean.numel() - v_.numel(),), fill, device=v_.device)])
        for k_, v_ in actor.state_dict().items():
            if k_ not in sd:                                    # a memory actor from one without: its GRU fresh, output zero
                sd[k_] = v_
            elif k_ in ("aux.weight", "aux.bias") and sd[k_].shape != v_.shape:
                # a wider read-out (--aux-opp) from a boxes-only one: the boxes' rows kept, the new ones fresh
                nv = v_.clone(); nv[: sd[k_].shape[0]] = sd[k_]; sd[k_] = nv
        actor.load_state_dict(sd)
        enc = {k_[4:]: v_ for k_, v_ in ck0["critic"].items() if k_.startswith("enc.")}
        critic.enc.load_state_dict(enc)
        if args.init_critic:
            c_sd = ck0["critic"]
            if all(k_ in c_sd and c_sd[k_].shape == v_.shape for k_, v_ in critic.state_dict().items()):
                critic.load_state_dict(c_sd)
                print("[plan_rl] critic too (its inputs match)", flush=True)
            else:
                print("[plan_rl] critic NOT taken: its inputs differ", flush=True)
        print(f"[plan_rl] actor from {args.init} (update {ck0.get('update')}), critic's scan encoder too", flush=True)
    if args.accum_actor > 1 and args.accum > 1:
        raise SystemExit("--accum and --accum-actor: one or the other")
    if args.accum_actor > 1 or args.accum_actor_auto:
        opt = torch.optim.Adam(list(actor.parameters()), lr=args.lr, eps=1e-5)
        opt_c = torch.optim.Adam(list(critic.parameters()), lr=args.lr, eps=1e-5)
    else:
        opt = torch.optim.Adam([{"params": list(actor.parameters())}, {"params": list(critic.parameters())}], lr=args.lr, eps=1e-5)
        opt_c = None
    dist_w = 1.0                                    # this process's share of the update's decisions (--dist-world)

    def dist_grads(params):
        """--dist-world: every process's gradient on `params`, weighted by its share of the decisions, summed into all"""
        if dist is None:
            return
        ps_ = list(params)
        flat = torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_)).reshape(-1) for p_ in ps_]) * dist_w
        dist.all_reduce(flat)
        o_ = 0
        for p_ in ps_:
            n_ = p_.numel()
            p_.grad = flat[o_:o_ + n_].view_as(p_).clone(); o_ += n_

    def dist_sum(*xs):
        """--dist-world: float sums over the processes (a tensor of them)"""
        t_ = torch.tensor([float(x_) for x_ in xs], dtype=torch.float64, device=dev)
        if dist is not None:
            dist.all_reduce(t_)
        return t_

    def norm_snap():
        return [(m_.mean.double().clone(), m_.var.double().clone(), m_.count.double().clone()) for m_ in (actor.norm, critic.norm)]

    def norm_merge(snap):
        """--dist-world: the observation statistics as if one process had seen every process's rollout (each started from the
        same `snap`; parallel mean/variance from the counts, sums and sums of squares each added)"""
        if dist is None:
            return
        for m_, (m0, v0, c0) in zip((actor.norm, critic.norm), snap):
            m1, v1, c1 = m_.mean.double(), m_.var.double(), m_.count.double()
            add = torch.cat([(c1 - c0).reshape(1), m1 * c1 - m0 * c0, (v1 + m1 * m1) * c1 - (v0 + m0 * m0) * c0])
            dist.all_reduce(add)
            n_ = m0.numel(); c_ = c0 + add[0]
            mean_ = (m0 * c0 + add[1:1 + n_]) / c_
            var_ = ((v0 + m0 * m0) * c0 + add[1 + n_:]) / c_ - mean_ * mean_
            m_.mean.copy_(mean_.float()); m_.var.copy_(var_.clamp_min(0.0).float()); m_.count.copy_(c_.float())

    if dist is not None:
        # one policy: every process starts from rank 0's weights and statistics
        for t_ in list(actor.parameters()) + list(actor.buffers()) + list(critic.parameters()) + list(critic.buffers()):
            dist.broadcast(t_.data, 0)
    json.dump({"args": vars(args), "world": vars(world), "priv_dim": priv_dim, "beams": obs.beams},
              open(os.path.join(run, "args.json"), "w"), indent=1, default=str)
    print(f"[plan_rl] {args.name}: 환경 {env.B} (학습 차 {Bl}), 결정 0.1 s, 롤아웃 {args.horizon}결정, LiDAR {obs.beams}빔 x {3}장, "
          f"critic 추가 입력 {priv_dim}", flush=True)
    T = args.horizon
    log = open(os.path.join(run, "progress.jsonl"), "a")
    t0 = time.time()
    trk = None
    ns_ema, b_noise = None, float("nan")       # --accum-actor-auto: smoothed |G|^2 and tr(Sigma) of the policy gradient
    tail_alpha = 1.0                            # --tail-trunk-ratio: the share of the tail's gradient let into the shared weights
    if args.track_steps:
        # --track-steps: how far each optimizer step moves the actor along directions measured elsewhere
        import numpy as np
        zt = np.load(args.track_steps)
        trk_p = [p_ for p_ in actor.parameters() if p_.requires_grad]
        assert [str(n_) for n_ in zt["names"]] == [n_ for n_, p_ in actor.named_parameters() if p_.requires_grad]
        trk = {k_[4:]: torch.as_tensor(zt[k_], dtype=torch.float32, device=dev) for k_ in zt.files if k_.startswith("dir_")}
        trk_flat = lambda: torch.cat([p_.detach().reshape(-1) for p_ in trk_p])
        os.makedirs(os.path.join(run, "track"), exist_ok=True)
    for upd in range(1, args.updates + 1):
        frac = (upd - 1) / max(1, args.updates - 1)
        for gp in opt.param_groups + (opt_c.param_groups if opt_c is not None else []):
            gp["lr"] = args.lr + (args.lr_end - args.lr) * frac
        if upd <= args.actor_freeze:
            opt.param_groups[0]["lr"] = 0.0                    # the critic catches up with a started actor first
        S, X, PV, U, LP, V, R, D, VT, FM, HM = [], [], [], [], [], [], [], [], [], [], []
        V0, D0, ST0, PTS, PVS = [], [], [], [], []
        BR = []                                                # --branches 2: the plan each decision drove
        br_n = {k: 0.0 for k in ("b0", "gap", "switch", "arg_switch", "pmax", "pairs")}
        sc_new = torch.zeros(3, device=dev)                    # --scene-replay: scenes taken this update, by what was hit
        sc_dec = sc_end = sc_end_hit = 0
        AX = []                                                # --mem-aux: the boxes ahead at each decision
        DMP = {k: [] for k in ("mean", "d_samp", "d_mean", "hard", "contact", "lap", "shaping", "pose", "std")}   # --dump-rollout
        n_removed = 0
        n_restarted = 0
        near_sum = torch.zeros(Bl, device=dev)
        vest_err = []
        stats = {k: torch.zeros(Bl, device=dev) for k in ("lap", "shaping", "contact", "plan_hit_cost", "plan_hit")}
        speed_sum, stop_sum = torch.zeros(Bl, device=dev), torch.zeros(Bl, device=dev)
        c0 = pr.n_contacts.clone()
        st0 = pr.stall_time.clone()
        if levels is not None and upd == max(1, args.level_replay_from):
            env.level_replay = levels                          # races reset from here on are named and scored
            print(f"[plan_rl] level replay on: {args.level_replay:g} of the resets, {args.level_replay_size} races per seat",
                  flush=True)
        lv0 = (levels.n_new, levels.n_replay, levels.ended, levels.ended_hit) if levels is not None else None
        tr0 = time.time()
        nsnap = norm_snap() if dist is not None else None
        actor.eval(); critic.eval()
        for t in range(T):
            with torch.no_grad():
                sc, s = obs.scans.clone(), obs.state()
                pv_ = priv_h()
                if scene is not None:
                    # the state this decision is taken from, env and policy inputs alike: what a scene is made of
                    snap = scene.capture()
                    snap.update(pol_scans=sc, pol_rel=obs.rel.clone(), pol_rel_v=obs.rel_v.clone(), pol_cmd_d=obs.cmd_d.clone(),
                                pol_cmd_v=obs.cmd_v.clone(), pol_plan_xy=obs.plan_xy.clone(), pol_plan_v=obs.plan_v.clone(),
                                pol_imu=env.last_result.imu[lid].float().mean(1))
                    if args.memory:
                        snap["pol_mem"] = obs.mem_h.clone()
                    if two:
                        snap["pol_branch"] = obs.prev_branch.clone()
                    hist.append(snap)
                    hard0 = env.sim.contact_counts[lid, 3:6].clone()
                    in_rep0 = scene.in_replay.clone()
                if hold and args.hold_clear > 0:
                    # another car or a box within reach: no hold, and one in progress ends now (the car re-plans as it will
                    # on the road)
                    near = near_threat()
                    hold_left = torch.where(near, torch.zeros_like(hold_left), hold_left)
                    near_sum += near.float()
                newp = hold_left == 0                              # rows that draw a plan now (the others drive a held one)
                actor.norm.update(s); critic.norm.update(torch.cat([s, pv_], 1))
                if args.memory:
                    HM.append(obs.mem_h.clone())                  # the state this decision starts from
                    if aux_on:
                        ax_ = (track_boxes(env, pr.goal, lid, AUXK, args.box_range) if args.box_range > 0
                               else env._prop_block(env.sim.state)[lid].float().clone())
                        if aux_opp:
                            ax_ = torch.cat([ax_, track_opp(env, pr.goal, lid, args.aux_opp)], 1)
                        AX.append(ax_)
                    mean, log_std, v_hat = actor(sc, s, obs.mem_h)
                    obs.mem_h = actor.h_out.detach()               # a restart below clears its rows (`obs.reset_rows`)
                elif v2:
                    mean, log_std, v_hat = actor(sc, s)
                else:
                    mean, log_std = actor(sc, s)
                if two:
                    # the choice: sampled with a number that mostly stays (see the module's note), both plans kept alive
                    pb = branch_p(actor.logits_out)
                    br_u = torch.where(torch.rand(Bl, device=dev) < args.branch_keep, br_u, torch.rand(Bl, device=dev))
                    br = (br_u >= pb[:, 0]).long()
                    mean = actor.means_out[ar_l, br]
                    BR.append(br)
                    arg_ = actor.logits_out.argmax(1); same_ep = (since > 0).float()
                    br_n["b0"] += float((br == 0).float().mean()); br_n["pmax"] += float(pb.max(1).values.mean())
                    br_n["gap"] += float((actor.means_out[:, 0, 0] - actor.means_out[:, 1, 0]).mean())
                    br_n["switch"] += float(((br != br_prev).float() * same_ep).sum())
                    br_n["arg_switch"] += float(((arg_ != arg_prev).float() * same_ep).sum())
                    br_n["pairs"] += float(same_ep.sum())
                    br_prev, arg_prev = br, arg_
                if args.noise_corr > 0:
                    # exploration noise that stays for several decisions (AR(1), each decision still N(0, 1) by itself; a
                    # restarted row draws afresh below). PPO's ratio uses that per-decision Gaussian, as gSDE (Raffin et al.
                    # 2021) and coloured noise in PPO (Hollenstein et al. 2024) do
                    eps_n = args.noise_corr * eps_n + (1.0 - args.noise_corr ** 2) ** 0.5 * torch.randn_like(mean)
                    eps = eps_n
                else:
                    eps = torch.randn_like(mean)
                if tailc:
                    u = mean + log_std.exp() * eps * rl_mask                         # the tail: its mean, no noise
                    lp = tanh_logp(u[:, rl_idx], mean[:, rl_idx], log_std[:, rl_idx])
                else:
                    u = mean + log_std.exp() * eps
                    lp = tanh_logp(u, mean, log_std)
                if two:
                    lp = lp + pb[ar_l, br].log()
                    obs.prev_branch = (1.0 - 2.0 * br).to(s.dtype)                   # a restart below clears its rows
                v = critic(sc, s, pv_)
                v_true = env.sim.state[lid, dyn.IVX].float().clone()
                if args.dump_rollout:
                    # the first 0.1 s steering the sample commands and the one the mean would have [rad]
                    vh_ = v_hat.clamp(-PS.V_REV, PS.V_FWD)
                    DMP["mean"].append(mean[:, rl_idx].clone()); DMP["std"].append(log_std.exp().expand_as(mean)[:, rl_idx].clone())
                    DMP["d_samp"].append(PS.step_targets(torch.tanh(u), vh_, obs.cmd_d, lim)[0][:, 0].clone())
                    DMP["d_mean"].append(PS.step_targets(torch.tanh(mean), vh_, obs.cmd_d, lim)[0][:, 0].clone())
                    st_ = env.sim.state[lid].float()
                    DMP["pose"].append(torch.stack([st_[:, 0], st_[:, 1], st_[:, 2], st_[:, dyn.IVX], env.sim.tid[lid].float()], 1))
                    hard_d0 = env.sim.contact_counts[lid, 3:6].clone()
                p = None
                if v3:
                    # the plan knows the control delay: from the speed estimate, the servo angle the last plan left and the
                    # command still in effect
                    p = P3.roll(torch.tanh(u), v_hat.clamp(-PS.V_REV, PS.V_FWD), obs.servo,
                                torch.stack([obs.cmd_d, obs.cmd_v], 1), lim, act3, turn3)
                    vest_err.append((v_hat - v_true).abs())
                elif v2:
                    # the plan starts from what the policy thinks its speed is and what it last commanded
                    p = PS.roll(torch.tanh(u), v_hat.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
                    vest_err.append((v_hat - v_true).abs())
                    if tailc:
                        V0.append(v_hat.clamp(-PS.V_REV, PS.V_FWD).clone()); D0.append(obs.cmd_d.clone())
                        ST0.append(env.sim.state[lid].clone())
                if hold:
                    # a new plan is kept; rows inside a held plan drive its next 0.1 s, re-expressed from where they are
                    sel = lambda A, B_: torch.where(newp.view(-1, *([1] * (A.dim() - 1))), A, B_)
                    p_held = p if p_held is None else {k_: sel(p[k_], p_held[k_]) if p[k_].dim() and p[k_].shape[0] == Bl
                                                       else p[k_] for k_ in p}
                    hold_k = torch.where(newp, torch.zeros_like(hold_k), hold_k + 1)
                    p = PS.shift_plan(p_held, hold_k)
                    more = torch.randint(h_lo - 1, h_hi, (Bl,), device=dev)
                    start = newp & (torch.rand(Bl, device=dev) < args.hold_p)
                    if args.hold_clear > 0:
                        start &= ~near
                    hold_left = torch.where(start, more, torch.where(newp, torch.zeros_like(hold_left), hold_left - 1))
                if levels is not None:
                    hard0 = env.sim.contact_counts[lid, 3:6].clone()
                out, p, (xw_, yw_, _) = drv.decision(torch.tanh(u), p, charge=newp if hold else None)
                if args.dump_rollout:
                    DMP["hard"].append((env.sim.contact_counts[lid, 3:6] - hard_d0).clamp_min(0).float())
                    for k in ("contact", "lap", "shaping"):
                        DMP[k].append(out[k].clone())
                if levels is not None and env.level_replay is not None:
                    hd = (env.sim.contact_counts[lid, 3:6] - hard0).clamp_min(0).float()
                    hd[:, 2] = hd[:, 2] * (pr.seen_t <= SEEN_S + PS.DT).float()      # a car the reward charges for
                    levels.add(hd.sum(1) + HitLevels.STAND * (env.sim.state[lid, dyn.IVX].abs() < 0.3).float(),
                               ~out["restarted"])
                if two:
                    br_u = torch.where(out["restarted"], torch.rand(Bl, device=dev), br_u)
                if scene is not None:
                    # 1. races that began from a scene in this decision: the policy's inputs as they were at the scene (the
                    #    env's part is in place); before any scene is added below -- the store's places are reused
                    g_l, i_l = scene.take_loaded()
                    if g_l.numel():
                        Dd = scene.data
                        obs.scans[g_l] = Dd["pol_scans"][i_l]
                        obs.rel[g_l] = Dd["pol_rel"][i_l]; obs.rel_v[g_l] = Dd["pol_rel_v"][i_l]
                        obs.cmd_d[g_l] = Dd["pol_cmd_d"][i_l]; obs.cmd_v[g_l] = Dd["pol_cmd_v"][i_l]
                        obs.plan_xy[g_l] = Dd["pol_plan_xy"][i_l]; obs.plan_v[g_l] = Dd["pol_plan_v"][i_l]
                        r_ = env.last_result
                        r_.imu[lid[g_l]] = Dd["pol_imu"][i_l][:, None, :].to(r_.imu.dtype)
                        if args.memory:
                            obs.mem_h[g_l] = Dd["pol_mem"][i_l]      # what the car knew then is part of the moment
                        if two:
                            obs.prev_branch[g_l] = Dd["pol_branch"][i_l]
                    # 2. hard contacts of this decision (a car's only if the reward charges for it)
                    hd = (env.sim.contact_counts[lid, 3:6] - hard0).clamp_min(0).float()
                    hd[:, 2] = hd[:, 2] * (pr.seen_t <= SEEN_S + PS.DT).float()
                    hit = (hd.sum(1) > 0) & ~out["restarted"]
                    # 3. a scene from every one outside a scene episode: the state `lag` decisions earlier in the same episode
                    take = hit & ~in_rep0 & (since >= lag_lo) & (cool >= lag_hi)
                    if bool(take.any()):
                        rows_ = torch.nonzero(take)[:, 0]
                        k_ = torch.randint(lag_lo, lag_hi + 1, (rows_.numel(),), device=dev, generator=scene.gen)
                        k_ = torch.minimum(k_, since[rows_]).clamp(max=len(hist) - 1)
                        kind_ = hd[rows_].argmax(1)
                        for kk in k_.unique().tolist():
                            mk_ = k_ == kk
                            new_ = {key: val[rows_[mk_]] for key, val in hist[-1 - kk].items()}
                            new_["hit_kind"] = kind_[mk_]
                            scene.add(new_)
                        cool[rows_] = 0
                        sc_new += torch.bincount(kind_, minlength=3).float()
                    # 4. how the scene episodes go
                    rep_hit |= hit & in_rep0
                    ended_ = out["restarted"] & in_rep0
                    sc_dec += int(in_rep0.sum()); sc_end += int(ended_.sum()); sc_end_hit += int((ended_ & rep_hit).sum())
                    rep_hit &= ~out["restarted"]
                since = torch.where(out["restarted"], torch.zeros_like(since), since + 1)
                if args.noise_corr > 0:
                    eps_n = torch.where(out["restarted"][:, None], torch.randn_like(eps_n), eps_n)
                cool += 1
                if tailc:
                    ii_ = PT_IDX.to(dev)
                    PTS.append(torch.stack([xw_[:, ii_], yw_[:, ii_]], -1)); PVS.append(p["pv"].clone())
                if remover is not None:
                    was = remover.removed.clone()
                    remover.step()
                    n_removed += int((remover.removed & ~was).sum())
                if restarter is not None:
                    n_restarted += restarter.step()
                if hold:
                    hold_left = torch.where(out["restarted"], torch.zeros_like(hold_left), hold_left)
            S.append(s); X.append(sc); PV.append(pv_); U.append(u); LP.append(lp); V.append(v); VT.append(v_true)
            FM.append(newp.float())
            R.append(out["reward"]); D.append(out["restarted"].float())
            for k in ("lap", "shaping", "contact"):
                stats[k] += out[k]
            stats["plan_hit_cost"] += out["plan_hit_price"]
            stats["plan_hit"] += out["plan_hit"]
            sp = env.sim.state[lid, dyn.IVX].abs()
            speed_sum += sp; stop_sum += (sp < 0.3).float()
        with torch.no_grad():
            v_last = critic(obs.scans, obs.state(), priv_h())
        t_roll = time.time() - tr0
        if dist is not None:
            norm_merge(nsnap)
        # GAE; a restart (episode end) cuts the trace and bootstraps from the value before it
        V_ = torch.stack(V); R_ = torch.stack(R); D_ = torch.stack(D); FM_ = torch.stack(FM)
        adv = torch.zeros_like(R_); gae = torch.zeros(Bl, device=dev)
        for t in reversed(range(T)):
            v_next = v_last if t == T - 1 else V_[t + 1]
            v_next = torch.where(D_[t] > 0, V_[t], v_next)
            delta = R_[t] + args.gamma * v_next - V_[t]
            # inside a held plan the trace runs on undiscounted by lambda: a plan's advantage is all it drove, then
            # lambda from the next plan on
            lam_t = args.lam if t == T - 1 else torch.where(FM_[t + 1] > 0, torch.full_like(gae, args.lam), torch.ones_like(gae))
            gae = delta + args.gamma * lam_t * (1.0 - D_[t]) * gae
            adv[t] = gae
        ret = adv + V_
        if args.dump_rollout:
            import numpy as np
            os.makedirs(args.dump_rollout, exist_ok=True)
            f16 = lambda a: a.cpu().numpy().astype(np.float16)
            np.savez_compressed(os.path.join(args.dump_rollout, f"u{upd}.npz"),
                                u=torch.stack(U)[:, :, rl_idx].cpu().numpy(), adv=adv.cpu().numpy(), v=V_.cpu().numpy(),
                                v_last=v_last.cpu().numpy(), r=R_.cpu().numpy(), done=D_.cpu().numpy().astype(np.uint8),
                                boxes=f16(torch.stack(PV)[:, :, _P0:_P0 + _w]), gamma=args.gamma, lam=args.lam,
                                **{k: torch.stack(v_).cpu().numpy() for k, v_ in DMP.items()})
        bS, bX, bPV, bU, bVT = torch.cat(S), torch.cat(X), torch.cat(PV), torch.cat(U), torch.cat(VT)
        bLP, bA, bR, bV = torch.stack(LP).reshape(-1), adv.reshape(-1), ret.reshape(-1), V_.reshape(-1)
        bFM = FM_.reshape(-1)                                  # the policy learns from the decisions it made
        N = bS.shape[0]
        if args.memory:
            # whole sequences of `mem_chunk` decisions per car, replayed from the state each began with; a restart inside
            # one clears the state for the decisions after it
            Lc = args.mem_chunk; nC = T // Lc
            bHM, bD = torch.cat(HM), D_.reshape(-1)
            bAX = torch.cat(AX) if aux_on else None
            tt = torch.arange(nC * Lc, device=dev).view(nC, Lc)
            seq_idx = (tt[:, :, None] * Bl + torch.arange(Bl, device=dev)[None, None, :]).permute(1, 0, 2).reshape(Lc, nC * Bl)

        def batches():
            if args.memory:
                for sq in torch.randperm(seq_idx.shape[1], device=dev).chunk(args.minibatches):
                    yield seq_idx[:, sq]
            else:
                for mb_ in torch.randperm(N, device=dev).chunk(args.minibatches):
                    yield mb_

        def heads_on(fc, mb_):
            if not two:
                return (*actor.heads(fc), mb_, fc)
            # the plan each decision drove, and the choice's logits for its probability
            means_, aux_box["logits"], ls_, vh_ = actor.all_heads(fc)
            return means_[torch.arange(mb_.numel(), device=dev), bBR[mb_]], ls_, vh_, mb_, fc

        def actor_on(sel):
            if not args.memory:
                return heads_on(actor.features(bX[sel], bS[sel]), sel)
            mb_ = sel.reshape(-1)
            f = actor.features(bX[mb_], bS[mb_]).view(sel.shape[0], sel.shape[1], -1)
            h = bHM[sel[0]]
            outs = []
            for i in range(sel.shape[0]):
                if i > 0:
                    # after a restart: the state the next decision was driven from -- zero, or a scene's memory
                    h = torch.where(bD[sel[i - 1]][:, None] > 0, bHM[sel[i]], h)
                fi, h = actor.remember(f[i], h)
                outs.append(fi)
            fc = torch.cat(outs, 0)
            if aux_on or memv_on:
                # the read-outs' own replay: the same memory on features it cannot move (the LiDAR part stays the policy's)
                hs = actor.memory_alone(f, bHM[sel[0]], bD[sel], bHM[sel])
                if aux_on:
                    aux_box["pred"] = actor.aux(hs)
                if memv_on:
                    aux_box["v"] = actor.v_mem(hs)[:, 0]
            return heads_on(fc, mb_)

        actor.train(); critic.train()
        kls, cfs, vls, els, cls_, ce5 = [], [], [], [], [], []
        ce10, tr_ratio = [], []
        als, aac, aac_far, alo, sms, mvs = [], [], [], [], [], []
        aux_box = {}
        g_split = {"policy": [], "speed": [], "value": []}
        if dist is not None:
            ss_ = dist_sum(bFM.sum(), bR.numel(), bR.sum(), bR.pow(2).sum())
            dist_w = float(bFM.sum()) / float(ss_[0]); n_all = float(ss_[0])
            r_mu = ss_[2] / ss_[1]; r_sd = (ss_[3] / ss_[1] - r_mu * r_mu).clamp_min(1e-12).sqrt()
        if memv_on:
            bRn = ((bR - bR.mean()) / bR.std().clamp_min(1e-6) if dist is None else
                   (bR - float(r_mu)) / max(float(r_sd), 1e-6))
        if two:
            bBR = torch.cat(BR)
        if tailc:
            bV0, bD0, bST0 = torch.cat(V0), torch.cat(D0), torch.cat(ST0)
            bPTS, bPVS = torch.cat(PTS), torch.cat(PVS)
            NB = bV0.shape[0]
            # a decision has a next plan to learn from unless it is the rollout's last or its car restarted in it
            c_ok = torch.ones(NB, dtype=torch.bool, device=dev); c_ok[-Bl:] = False
            c_ok &= D_.reshape(-1) <= 0
            if args.tail_target == "driven":
                # where the car was k x 0.1 s after each decision (k = 1..10): the state k decisions later, unless the car
                # restarted in between or the rollout ended first
                S_ = bST0.view(T, Bl, -1); Dn_ = (D_ > 0).long(); cs_ = torch.cumsum(Dn_, 0)
                DRV = torch.zeros(T, Bl, PS.N_STEP, 2, device=dev); DRVV = torch.zeros(T, Bl, PS.N_STEP, device=dev)
                DRVM = torch.zeros(T, Bl, PS.N_STEP, dtype=torch.bool, device=dev)
                for k_ in range(1, PS.N_STEP + 1):
                    if k_ >= T:
                        break
                    DRV[:T - k_, :, k_ - 1] = S_[k_:, :, 0:2]; DRVV[:T - k_, :, k_ - 1] = S_[k_:, :, dyn.IVX]
                    before_ = torch.cat([torch.zeros(1, Bl, dtype=cs_.dtype, device=dev), cs_[:T - k_ - 1]], 0)
                    DRVM[:T - k_, :, k_ - 1] = (cs_[k_ - 1:T - 1] - before_) == 0     # no restart in decisions t .. t+k-1
                bDRV, bDRVV, bDRVM = DRV.reshape(NB, PS.N_STEP, 2), DRVV.reshape(NB, PS.N_STEP), DRVM.reshape(NB, PS.N_STEP)
                c_ok = bDRVM[:, 1]                                  # at least the 0.2 s point is known
                del DRV, DRVV, DRVM, S_
        if args.dump_rollout:
            # the direction this rollout alone would push the actor (no step is made): the policy loss of the first pass
            # (ratio 1, advantage normalised over the whole rollout), its gradient through the steering mean, the speed
            # mean and the noise sizes kept apart. Rollouts are independent, so the cosine between two of them is the
            # share of one rollout's push that is signal.
            pr_ = [p_ for p_ in actor.parameters() if p_.requires_grad]
            gsum = {k_: [torch.zeros_like(p_) for p_ in pr_] for k_ in ("steer", "speed", "std")}
            nF = bFM.sum().clamp_min(1.0); muA = (bA * bFM).sum() / nF
            bAn = (bA - muA) / (((bA - muA).pow(2) * bFM).sum() / nF).sqrt().add(1e-8)
            for sel in batches():
                mean, log_std, _, mb, _ = actor_on(sel) if v2 else (*actor(bX[sel], bS[sel]), None, sel, None)
                w_ = bAn[mb] * bFM[mb] / nF
                ii_ = rl_idx if tailc else torch.arange(PS.ACT_DIM, device=dev)
                st_ = ii_ < PS.N_STEP
                um_, mm_, ls_ = bU[mb][:, ii_], mean[:, ii_], log_std[:, ii_]
                ls_parts = (("steer", tanh_logp(um_[:, st_], mm_[:, st_], ls_[:, st_].detach())),
                            ("speed", tanh_logp(um_[:, ~st_], mm_[:, ~st_], ls_[:, ~st_].detach())),
                            ("std", tanh_logp(um_, mm_.detach(), ls_)))
                for j_, (k_, lp_) in enumerate(ls_parts):
                    for a_, g_ in zip(gsum[k_], torch.autograd.grad(-(lp_ * w_).sum(), pr_, retain_graph=j_ < 2, allow_unused=True)):
                        if g_ is not None:
                            a_ += g_
                del mean, log_std, ls_parts, lp_, um_, mm_, ls_      # the graph goes before the next minibatch's is built
                aux_box.clear()
            names_ = [n_ for n_, p_ in actor.named_parameters() if p_.requires_grad]
            np.savez(os.path.join(args.dump_rollout, f"g{upd}.npz"), names=np.array(names_), sizes=np.array([p_.numel() for p_ in pr_]),
                     **{k_: torch.cat([a_.reshape(-1) for a_ in v_]).cpu().numpy() for k_, v_ in gsum.items()})
            print(f"[plan_rl] update {upd}: rollout + its policy gradient saved ({t_roll:.0f} s), no update made", flush=True)
            del gsum, bAn, w_
            bS = bX = bPV = bU = bVT = bLP = bA = bR = bV = bFM = None   # the next rollout needs the room (7.5 GB card)
            torch.cuda.empty_cache()
            continue
        tu0 = time.time()
        if args.accum_actor_auto and (upd - 1) % max(1, args.noise_probe_every) == 0:
            # the policy gradient's noise scale on this rollout, at the policy that drove it (no step): per minibatch the
            # gradient of the policy loss at ratio 1 (advantage normalised per minibatch, as the epochs do), then
            # |G|^2 and tr(Sigma) from the minibatch gradients and their mean (two batch sizes), smoothed over updates
            pr_ = [p_ for p_ in actor.parameters() if p_.requires_grad]
            gsum_, gs2_, nb_, bs_ = None, 0.0, 0, 0.0
            for sel in batches():
                mean, log_std, _, mb, _ = actor_on(sel) if v2 else (*actor(bX[sel], bS[sel]), None, sel, None)
                ii_ = rl_idx if tailc else torch.arange(PS.ACT_DIM, device=dev)
                lp_ = tanh_logp(bU[mb][:, ii_], mean[:, ii_], log_std[:, ii_])
                m_ = bFM[mb]; n_ = m_.sum().clamp_min(1.0); a_ = bA[mb]; mu_ = (a_ * m_).sum() / n_
                a_ = (a_ - mu_) / (((a_ - mu_).pow(2) * m_).sum() / n_).sqrt().add(1e-8)
                g_ = torch.autograd.grad(-(lp_ * a_ * m_).sum() / n_, pr_, allow_unused=True)
                g_ = torch.cat([(x_ if x_ is not None else torch.zeros_like(p_)).reshape(-1) for x_, p_ in zip(g_, pr_)])
                gsum_ = g_ if gsum_ is None else gsum_ + g_
                gs2_ += float(g_.pow(2).sum()); nb_ += 1; bs_ += float(n_)
                del mean, log_std, lp_, g_
                aux_box.clear()
            if dist is None:
                bs_ /= nb_; bb_ = bs_ * nb_
                gb2_ = float((gsum_ / nb_).pow(2).sum()); gs2_ /= nb_
            else:
                # every process's minibatch gradients: the big batch = all decisions (weighted mean of the processes' means)
                gm_ = gsum_ / nb_ * dist_w
                dist.all_reduce(gm_)
                tt_ = dist_sum(gs2_, nb_, bs_)
                gb2_ = float(gm_.pow(2).sum()); gs2_ = float(tt_[0] / tt_[1]); bs_ = float(tt_[2] / tt_[1]); bb_ = n_all
                del gm_
            G2_ = (bb_ * gb2_ - bs_ * gs2_) / (bb_ - bs_); tr_ = (gs2_ - gb2_) / (1.0 / bs_ - 1.0 / bb_)
            ns_ema = (G2_, tr_) if ns_ema is None else (0.9 * ns_ema[0] + 0.1 * G2_, 0.9 * ns_ema[1] + 0.1 * tr_)
            b_noise = ns_ema[1] / ns_ema[0] if ns_ema[0] > 0 else float("inf")
            k_ = 1
            bstep_ = bb_ / args.minibatches                     # decisions in one minibatch step (all processes)
            if args.accum_auto_frac > 0:
                while k_ * 2 <= args.minibatches and k_ * 2 * bstep_ <= args.accum_auto_frac * b_noise:
                    k_ *= 2
            else:
                while k_ * 2 <= args.minibatches and k_ * bstep_ < b_noise:
                    k_ *= 2
            args.accum_actor = k_
            del gsum_
        acc_i = 0
        if trk is not None:
            th_prev = trk_flat(); th_start = th_prev.clone()
            T_ = {k_: [] for k_ in ("step_norm", "grad_norm", "clip_coef")}
            for k_ in trk:
                T_[f"step_{k_}"] = []; T_[f"grad_{k_}"] = []
            T_terms = {}
        for ep in range(args.epochs):
            for sel in batches():
                if v2:
                    mean, log_std, v_hat, mb, f_heads = actor_on(sel)
                    el = (v_hat - bVT[mb]).pow(2).mean()
                else:
                    mb = sel
                    mean, log_std = actor(bX[mb], bS[mb])
                    el = torch.zeros((), device=dev)
                if tailc:
                    lp = tanh_logp(bU[mb][:, rl_idx], mean[:, rl_idx], log_std[:, rl_idx])
                else:
                    lp = tanh_logp(bU[mb], mean, log_std)
                if two:
                    lp = lp + branch_p(aux_box["logits"])[torch.arange(mb.numel(), device=dev), bBR[mb]].log()
                ratio = (lp - bLP[mb]).exp()
                m_ = bFM[mb]; n_ = m_.sum().clamp_min(1.0)
                a_ = bA[mb]; mu_ = (a_ * m_).sum() / n_
                a_ = (a_ - mu_) / (((a_ - mu_).pow(2) * m_).sum() / n_).sqrt().add(1e-8)
                pl = -(torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_) * m_).sum() / n_
                vl = 0.5 * (critic(bX[mb], bS[mb], bPV[mb]) - bR[mb]).pow(2).mean()
                loss = pl + args.vf_coef * vl + args.vest_coef * el
                terms_ = {"policy": pl, "vest": args.vest_coef * el}
                if memv_on:
                    lmv = (aux_box["v"] - bRn[mb]).pow(2).mean()
                    loss = loss + args.mem_value * lmv
                    terms_["memv"] = args.mem_value * lmv
                    mvs.append(float(lmv.detach()))
                if args.grad_split and args.memory and v2 and ep == 0:
                    for k_, l_ in (("policy", pl), ("speed", args.vest_coef * el)) + ((("value", args.mem_value * lmv),) if memv_on else ()):
                        g_split[k_].append(torch.cat([g_.reshape(-1) for g_ in torch.autograd.grad(l_, list(actor.gru.parameters()), retain_graph=True)]))
                if tailc:
                    # (가): this decision's plan, its first step as driven, its tail from the output layer on features it
                    # does not move, laid where the car was -- against the next decision's plan for the same moments
                    ok_ = c_ok[mb]
                    if bool(ok_.any()):
                        mb_c = mb[ok_]
                        fh_ = (f_heads[ok_] * tail_alpha + f_heads[ok_].detach() * (1.0 - tail_alpha) if args.tail_trunk
                               else f_heads[ok_].detach())     # same values; only tail_alpha of the gradient reaches the trunk
                        tail_mean = (actor.all_heads(fh_)[0][torch.arange(mb_c.numel(), device=dev), bBR[mb_c]]
                                     if two else actor.pi(fh_))      # the plan that was driven
                        a_c = torch.where(rl_mask[None, :], torch.tanh(bU[mb_c]), torch.tanh(tail_mean))
                        pc_ = PS.roll(a_c, bV0[mb_c], bD0[mb_c], lim)
                        xw_c, yw_c, _ = plan_world(pc_, bST0[mb_c])
                        ii_ = PT_IDX.to(dev)
                        pts_c = torch.stack([xw_c[:, ii_], yw_c[:, ii_]], -1)
                        if args.tail_target == "driven":
                            tgt, tgv, vm_ = bDRV[mb_c], bDRVV[mb_c], bDRVM[mb_c][:, 1:].float()
                            lpos = nn.functional.smooth_l1_loss(pts_c[:, 1:], tgt[:, 1:], reduction="none").sum(-1)
                            lv = nn.functional.smooth_l1_loss(pc_["pv"][:, 1:], tgv[:, 1:], reduction="none")
                            lc = ((lpos + lv) * vm_).sum() / vm_.sum().clamp_min(1.0)
                            with torch.no_grad():
                                for k_, lst_ in ((4, ce5), (9, ce10)):     # the 0.5 s / 1.0 s point beside the driven path
                                    ok5_ = bDRVM[mb_c][:, k_]
                                    if bool(ok5_.any()):
                                        lst_.append(float((pts_c[ok5_, k_] - tgt[ok5_, k_]).norm(dim=-1).median()))
                        else:
                            tgt, tgv = bPTS[mb_c + Bl], bPVS[mb_c + Bl]
                            lpos = nn.functional.smooth_l1_loss(pts_c[:, 1:], tgt[:, :-1], reduction="none").sum(-1)
                            lv = nn.functional.smooth_l1_loss(pc_["pv"][:, 1:], tgv[:, :-1], reduction="none")
                            lc = (lpos + lv).mean()
                            ce5.append(float((pts_c[:, 5] - tgt[:, 4]).norm(dim=-1).median()))
                        loss = loss + args.tail_consistency * lc
                        terms_["tail"] = args.tail_consistency * lc
                        cls_.append(float(lc.detach()))
                        if args.tail_trunk and ep == 0 and len(tr_ratio) < 4:
                            # how hard the tail pulls on the shared weights next to the policy (11-14: the speed estimate's
                            # trunk gradient was 18x the policy's and set Adam's step size there)
                            sh_ = [p_ for n_, p_ in actor.named_parameters() if p_.requires_grad and n_.startswith(("enc.", "trunk.", "gru."))]
                            gp_ = torch.autograd.grad(pl, sh_, retain_graph=True, allow_unused=True)
                            gt_ = torch.autograd.grad(args.tail_consistency * lc, sh_, retain_graph=True, allow_unused=True)
                            nrm_ = lambda gs: float(torch.sqrt(sum((g_.pow(2).sum() for g_ in gs if g_ is not None), torch.zeros((), device=dev))))
                            tr_ratio.append(nrm_(gt_) / max(nrm_(gp_), 1e-12))
                if aux_on:
                    pa_all, ta_all = aux_box["pred"], bAX[mb]
                    pa = pa_all[:, :AUXB].reshape(-1, AUXK, len(PRIV_V2_PROP_KEYS)); ta = ta_all[:, :AUXB].reshape(-1, AUXK, len(PRIV_V2_PROP_KEYS))
                    m_p = ta[..., 0]
                    l_pres = nn.functional.binary_cross_entropy_with_logits(pa[..., 0], m_p)
                    l_pos = (nn.functional.smooth_l1_loss(pa[..., 1:], ta[..., 1:], reduction="none").sum(-1) * m_p).sum() / m_p.sum().clamp_min(1.0)
                    la = l_pres + l_pos
                    if aux_opp:
                        po_, to_ = pa_all[:, AUXB:], ta_all[:, AUXB:]
                        m_o = to_[:, 0]
                        lo = nn.functional.binary_cross_entropy_with_logits(po_[:, 0], m_o) + (
                            nn.functional.smooth_l1_loss(po_[:, 1:], to_[:, 1:], reduction="none").sum(-1) * m_o).sum() / m_o.sum().clamp_min(1.0)
                        la = la + lo
                        with torch.no_grad():
                            alo.append(float(lo))
                    loss = loss + args.mem_aux * la
                    with torch.no_grad():
                        als.append(float(la)); aac.append(float(((pa[..., 0] > 0).float() == m_p).float().mean()))
                        far = m_p[:, 0] > 0.5
                        far = far & (ta[:, 0, 3] * 5.0 > 4.0)          # the nearest box 4-12 m away: mostly not yet seen up close
                        if bool(far.any()):
                            aac_far.append(float((pa[far, 0, 0] > 0).float().mean()))
                if args.smooth_reg > 0 and args.memory:
                    Ls_, ns_ = sel.shape
                    m0 = torch.tanh(mean[:, 0]).view(Ls_, ns_)
                    vv = bVT[mb].view(Ls_, ns_)
                    ok_s = 1.0 - bD[sel[:-1]]                       # no pair across a restart
                    jerk = vv[1:].pow(2) * (lim.sv_max * PS.DT) / lim.wheelbase * (m0[1:] - m0[:-1]) / 4.0
                    # Huber (11-10 test: squared, the mean was 5.7-7.6 -- reversals of ~0.2 in the action at 5-7 m/s -- and the
                    # first update's KL 0.20): linear past 4 m/s^2 so a few large reversals do not outweigh the policy's own loss
                    ls_ = (nn.functional.smooth_l1_loss(jerk, torch.zeros_like(jerk), reduction="none") * ok_s).sum() / ok_s.sum().clamp_min(1.0)
                    loss = loss + args.smooth_reg * ls_
                    sms.append(float(ls_.detach()))
                if trk is not None and ep == 0:
                    for k_, l_ in terms_.items():
                        if not l_.requires_grad:
                            continue
                        gs_ = torch.autograd.grad(l_, trk_p, retain_graph=True, allow_unused=True)
                        g_ = -torch.cat([(g1 if g1 is not None else torch.zeros_like(p_)).reshape(-1) for g1, p_ in zip(gs_, trk_p)])
                        e_ = T_terms.setdefault(k_, {"norm": [], "sum": torch.zeros_like(g_), **{d_: [] for d_ in trk}})
                        e_["norm"].append(float(g_.norm())); e_["sum"] += g_
                        for d_, v_ in trk.items():
                            e_[d_].append(float(g_ @ v_))
                        del gs_, g_
                if opt_c is not None:
                    # --accum-actor: the critic steps on this minibatch; the actor's gradient keeps adding up
                    if acc_i % args.accum_actor == 0:
                        opt.zero_grad(set_to_none=True)
                    opt_c.zero_grad(set_to_none=True)
                    loss.backward(); acc_i += 1
                    dist_grads(critic.parameters())
                    nn.utils.clip_grad_norm_(list(critic.parameters()), args.max_grad)
                    opt_c.step()
                    if acc_i % args.accum_actor:
                        with torch.no_grad():
                            kls.append(float(((bLP[mb] - lp) * m_).sum() / n_))
                            cfs.append(float((((ratio - 1).abs() > args.clip).float() * m_).sum() / n_))
                            vls.append(float(vl)); els.append(float(el))
                        continue
                else:
                    if acc_i % args.accum == 0:
                        opt.zero_grad(set_to_none=True)
                    (loss / args.accum if args.accum > 1 else loss).backward(); acc_i += 1
                    if acc_i % args.accum:
                        with torch.no_grad():
                            kls.append(float(((bLP[mb] - lp) * m_).sum() / n_))
                            cfs.append(float((((ratio - 1).abs() > args.clip).float() * m_).sum() / n_))
                            vls.append(float(vl)); els.append(float(el))
                        continue
                dist_grads(list(actor.parameters()) + ([] if opt_c is not None else list(critic.parameters())))
                gn_all = nn.utils.clip_grad_norm_(list(actor.parameters()) + ([] if opt_c is not None else list(critic.parameters())), args.max_grad)
                if trk is not None:
                    g_ = -torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_)).reshape(-1) for p_ in trk_p])
                    T_["grad_norm"].append(float(g_.norm())); T_["clip_coef"].append(min(1.0, args.max_grad / (float(gn_all) + 1e-6)))
                    for d_, v_ in trk.items():
                        T_[f"grad_{d_}"].append(float(g_ @ v_))
                opt.step()
                if trk is not None:
                    th_ = trk_flat(); dth = th_ - th_prev; th_prev = th_
                    T_["step_norm"].append(float(dth.norm()))
                    for d_, v_ in trk.items():
                        T_[f"step_{d_}"].append(float(dth @ v_))
                with torch.no_grad():
                    kls.append(float(((bLP[mb] - lp) * m_).sum() / n_))
                    cfs.append(float((((ratio - 1).abs() > args.clip).float() * m_).sum() / n_))
                    vls.append(float(vl)); els.append(float(el))
        t_upd = time.time() - tu0
        if args.tail_trunk and args.tail_trunk_ratio > 0 and tr_ratio:
            # measured with the present share: move the share toward the target ratio (damped; the next update uses it)
            r_ = sum(tr_ratio) / len(tr_ratio)
            if dist is not None:
                r_ = float(dist_sum(r_)[0]) / args.dist_world          # one share for every process
            tail_alpha = float(min(1.0, max(0.01, tail_alpha * (args.tail_trunk_ratio / max(r_, 1e-6)) ** 0.5)))
        if trk is not None:
            # per optimizer step: the actor's move and the clipped gradient (descent sign) along each direction; per loss term
            # (first epoch, per minibatch): its descent direction along each, and the term summed over the first epoch
            dth = th_prev - th_start
            sv_ = {k_: np.array(v_) for k_, v_ in T_.items()}
            sv_["lr"] = np.array(opt.param_groups[0]["lr"]); sv_["update_norm"] = np.array(float(dth.norm()))
            for d_, v_ in trk.items():
                sv_[f"update_{d_}"] = np.array(float(dth @ v_))
            for k_, e_ in T_terms.items():
                sv_[f"term_{k_}_norm"] = np.array(e_["norm"]); sv_[f"term_{k_}_sum"] = e_["sum"].cpu().numpy().astype(np.float16)
                for d_ in trk:
                    sv_[f"term_{k_}_{d_}"] = np.array(e_[d_])
            sv_["update_move"] = dth.cpu().numpy().astype(np.float16)
            np.savez(os.path.join(run, "track", f"t{upd}.npz"), **sv_)
            del dth, th_start
        secs = T * PS.DT
        mins = secs * n_ord / 60.0
        dc = (pr.n_contacts - c0)[ordm].sum(0)
        tg_ = env.__dict__.get("_teacher_normalized")
        if tg_ is not None and hasattr(tg_, "fell_back") and upd <= 3:
            # (11-14) how the opponents' planners ran: graph replays, eager steps, planners that went back to eager for good
            print(f"[plan_rl] update {upd}: 상대 차 planner graph 재생 {getattr(tg_, 'replays', 0)}번, 한 번씩 eager "
                  f"{getattr(tg_, 'eager_steps', 0)}번, 아예 eager로 돌아간 planner {dict(tg_.fell_back)}", flush=True)
            for k_, n_ in sorted(getattr(tg_, "eager_why", {}).items(), key=lambda kv: -kv[1])[:6]:
                print(f"[plan_rl]     eager {n_}번: {k_}", flush=True)
        if args.accum_actor_auto:
            print(f"[plan_rl] update {upd}: policy gradient 잡음 크기 {b_noise:.0f} 결정 -> actor는 minibatch {args.accum_actor}개를 합쳐 한 걸음", flush=True)
        row = {"update": upd, "min": (time.time() - t0) / 60, "t_roll": t_roll, "t_upd": t_upd,
               "accum_actor": args.accum_actor, "noise_scale": b_noise if math.isfinite(b_noise) else None,
               "dist_world": args.dist_world, "decisions_all": n_all if dist is not None else None,
               "actor_checksum": float(torch.cat([p_.detach().reshape(-1) for p_ in actor.parameters()]).double().abs().sum()),
               "laps_per_min": float(stats["lap"][ordm].sum()) / mins, "speed": float(speed_sum[ordm].mean()) / T,
               "stopped": float(stop_sum[ordm].mean()) / T, "contacts_wall_per_min": float(dc[0]) / mins,
               "stall_s_per_min": float((pr.stall_time - st0)[ordm].sum()) / mins,
               "contacts_prop_per_min": float(dc[1]) / mins, "contacts_car_per_min": float(dc[2]) / mins,
               "plan_hit_share": float(stats["plan_hit"][ordm].mean()) / T, "reward": float(R_.mean()),
               "shaping": float(stats["shaping"][ordm].mean()) / T, "contact_cost": float(stats["contact"][ordm].mean()) / T,
               "kl": sum(kls) / len(kls), "clipfrac": sum(cfs) / len(cfs), "vloss": sum(vls) / len(vls),
               "vest_mae": float(torch.stack(vest_err).mean()) if vest_err else 0.0, "held_share": 1.0 - float(FM_.mean()), "removed": n_removed, "opp_restarted": n_restarted,
               "tail_loss": (sum(cls_) / len(cls_)) if cls_ else None, "tail_next_05": (sum(ce5) / len(ce5)) if ce5 else None,
               "tail_driven_10": (sum(ce10) / len(ce10)) if ce10 else None, "tail_trunk_ratio": (sum(tr_ratio) / len(tr_ratio)) if tr_ratio else None, "tail_alpha": tail_alpha if args.tail_trunk else None, "near_share": float(near_sum.mean()) / T,
               "std": [round(float(x), 3) for x in actor.log_std.detach().exp()],
               "aux_loss": (sum(als) / len(als)) if als else None, "aux_acc": (sum(aac) / len(aac)) if aac else None,
               "aux_far_recall": (sum(aac_far) / len(aac_far)) if aac_far else None,
               "aux_opp_loss": (sum(alo) / len(alo)) if alo else None,
               "smooth_loss": (sum(sms) / len(sms)) if sms else None,
               "mem_value_loss": (sum(mvs) / len(mvs)) if mvs else None}
        if g_split["policy"]:
            # size = a minibatch's gradient on the GRU's weights; agree = the mean cosine between two minibatches' gradients:
            # 1 when every minibatch pushes the same way, 0 when they are unrelated (then the steps cancel)
            gp_, ge_ = torch.stack(g_split["policy"]), torch.stack(g_split["speed"])
            for k_, g_ in (("policy", gp_), ("speed", ge_)) + ((("value", torch.stack(g_split["value"])),) if g_split["value"] else ()):
                gn_ = nn.functional.normalize(g_, dim=1); c_ = gn_ @ gn_.T; n_g = g_.shape[0]
                row[f"g_mem_{k_}"] = float(g_.norm(dim=1).mean())
                row[f"g_mem_{k_}_agree"] = float((c_.sum() - c_.diagonal().sum()) / max(1, n_g * (n_g - 1)))
            row["g_mem_cos"] = float(nn.functional.cosine_similarity(gp_.mean(0), ge_.mean(0), dim=0))
        if two:
            row.update(branch0_share=br_n["b0"] / T, branch_gap=br_n["gap"] / T, branch_pmax=br_n["pmax"] / T,
                       branch_switch=br_n["switch"] / max(br_n["pairs"], 1.0),
                       branch_arg_switch=br_n["arg_switch"] / max(br_n["pairs"], 1.0))
        if scene is not None:
            row.update(scene_store=scene.n, scene_new=[int(x_) for x_ in sc_new], scene_share=sc_dec / (T * Bl),
                       scene_ended=sc_end, scene_ended_hit=sc_end_hit)
        if drv.opp is not None:
            row.update(self_opp_share=drv.opp.share_on(), self_opp_pool=[int(t_) for t_ in drv.opp.tags])
        if levels is not None:
            n_new, n_rep = levels.n_new - lv0[0], levels.n_replay - lv0[1]
            n_end, n_hit = levels.ended - lv0[2], levels.ended_hit - lv0[3]
            row.update(replay_share=n_rep / max(1, n_new + n_rep), replay_ended=n_end, replay_ended_scored=n_hit,
                       replay_buffer=levels.stats()["buffer_mean"], replay_score=levels.stats()["score_mean"])
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"[{time.strftime('%H:%M')}] 업데이트 {upd} | 바퀴 {row['laps_per_min']:.2f}/분 | 속도 {row['speed']:.2f} m/s | 멈춤 "
              f"{row['stopped'] * 100:.0f} % " + (f"(서 있는 벌점 {row['stall_s_per_min']:.1f} s/분) " if args.stall_s > 0 else "")
              + f"| 접촉 벽 {row['contacts_wall_per_min']:.2f} 상자 {row['contacts_prop_per_min']:.2f} 차 "
              f"{row['contacts_car_per_min']:.2f} /분 | 박는 plan {row['plan_hit_share'] * 100:.0f} % | 보상 {row['reward']:.4f} | "
              f"KL {row['kl']:.4f} 잘림 {row['clipfrac'] * 100:.0f} % | 속도 짐작 오차 {row['vest_mae']:.2f} m/s | 표준편차 {sum(row['std']) / len(row['std']):.3f} | "
              + (f"붙잡은 결정 {row['held_share'] * 100:.0f} % | " if hold else "")
              + (f"가까이 차·상자 {row['near_share'] * 100:.0f} % | " if hold and args.hold_clear > 0 else "")
              + (f"상자 치움 {n_removed}판 | " if remover is not None else "")
              + (f"두 plan: 0번 고름 {row['branch0_share'] * 100:.0f} %, 첫 조향 차이 {row['branch_gap']:.2f}, 고른 plan 바뀜 "
                 f"{row['branch_switch'] * 100:.1f} % (확률 큰 쪽 바뀜 {row['branch_arg_switch'] * 100:.1f} %), 큰 쪽 확률 "
                 f"{row['branch_pmax'] * 100:.0f} % | " if two else "")
              + (f"접촉 직전 장면: 보관 {row['scene_store']}개 (새로 벽 {row['scene_new'][0]} 상자 {row['scene_new'][1]} 차 "
                 f"{row['scene_new'][2]}), 장면에서 시작한 결정 {row['scene_share'] * 100:.1f} %, 끝난 {sc_end}판 중 다시 세게 닿음 "
                 f"{sc_end_hit / max(sc_end, 1) * 100:.0f} % | " if scene is not None else "")
              + (f"선 상대 차 다시 출발 {n_restarted}번 | " if restarter is not None else "")
              + (f"복사본이 모는 상대 차 {row['self_opp_share'] * 100:.0f} % (update {row['self_opp_pool']}의 복사본) | " if drv.opp is not None else "")
              + (f"다시 낸 경주 {row['replay_share'] * 100:.0f} % (끝난 경주 {row['replay_ended']} 중 점수 있는 것 "
                 f"{row['replay_ended_scored']}, 자리당 {row['replay_buffer']:.1f}개) | " if levels is not None else "")
              + (f"기억 보조: 손실 {row['aux_loss']:.3f}, 상자 있음 맞힘 {row['aux_acc'] * 100:.0f} %, 4-12 m 앞 상자 맞힘 "
                 f"{(row['aux_far_recall'] or 0) * 100:.0f} % | " if row.get("aux_loss") is not None else "")
              + (f"상대 차 보조 손실 {row['aux_opp_loss']:.3f} | " if row.get("aux_opp_loss") is not None else "")
              + (f"지그재그 비용 {row['smooth_loss']:.3f} | " if row.get("smooth_loss") is not None else "")
              + (f"기억이 읽은 value: return 퍼짐의 {(1 - row['mem_value_loss']) * 100:.0f} % 설명 | " if row.get("mem_value_loss") is not None else "")
              + (f"뒷부분 손실 {row['tail_loss']:.3f}, " + (f"실제로 간 길과 0.5 s / 1.0 s 차이 {row['tail_next_05'] * 100:.0f} / {(row['tail_driven_10'] or 0) * 100:.0f} cm"
                 if args.tail_target == "driven" else f"다음 plan과 0.5 s 차이 {row['tail_next_05'] * 100:.0f} cm")
                 + (f", 몸통 gradient policy 대비 {row['tail_trunk_ratio']:.2f}배" if row.get("tail_trunk_ratio") is not None else "") + " | "
                 if row.get("tail_loss") is not None else "")
              + f"{t_roll:.1f}+{t_upd:.1f} s", flush=True)
        if (upd % args.save_every == 0 or upd == args.updates) and args.dist_rank == 0:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "args": vars(args), "world": vars(world),
                        "priv_dim": priv_dim, "beams": obs.beams, "update": upd, "format": FORMAT3 if v3 else FORMAT2 if v2 else "plan_steps"},
                       os.path.join(run, f"plan_u{upd}.pt"))
        if drv.opp is not None and upd >= max(args.self_opp_from, 1) and (upd - args.self_opp_from) % args.self_opp_every == 0:
            drv.opp.add(actor, upd)


if __name__ == "__main__":
    main()
