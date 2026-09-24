"""A privileged teacher that chooses its move by simulating it, with the other cars reacting.

Every teacher in the tree so far predicts the other car and then plans against the prediction. The
prediction is the other car's last plan carried forward, and the other car is not a recording: a
`forzaeth` opponent sees the learner inside ten metres and bends its spline round it, a lane switcher
changes lane when blocked. Measured against `forzaeth` on ICCAS, the interactive teacher's forecast
of the opponent was 0.11 m off at 0.5 s in close company and 0.37 m off at the moments it made
contact, and 11 of its 12 contacts came after every candidate it had was already predicted to
collide. Inflating the other car by the forecast error did not fix it; it traded passes for
contacts, down to 0 of each at 0.1 m + 0.5 m/s.

This teacher knows how the other car will react because it runs it. A shadow `F1VecEnv` holds K
copies of every race; at each decision the real env's per-row state is copied into all K, copy k
drives candidate policy k -- "follow the raceline at offset o_k, speed scale s_k", re-planned every
step -- for `horizon_s`, and the opponents in every copy are driven by their own planners, exactly
as in the real env. A candidate that touches a car, a wall or a prop anywhere in the horizon is out;
of the rest, the one that made the most progress wins. Between decisions the chosen policy keeps
driving, which is what the shadow simulated.

The copy is exact up to sensor noise: the same policy run in the real env and in a shadow copy
agree to 1.2 mm (median) after 10 steps and ~3 cm after 40. The noise is not copied: the shadow
draws its own, which is a feature -- a decision that only survives one noise draw is fragile.

Cost is set by kernel launches, not by K or B: a shadow step of 896 rows is ~70 ms with the
physics compiled, so a decision over 1.5 s is ~3 s, whatever the batch.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .teacher import RacelineTeacher

#: Row tensors whose VALUES are row indices (or derived from the row's position in the batch). The
#: shadow's own are right for the shadow; a tiled copy of the real env's would point every copy at
#: the first one.
INDEX_STATE = frozenset({"sim.eid", "sim.other_idx", "race", "slot", "spawn_order_race"})


def _resolve(root, path):
    obj = root
    for kind, key in path:
        obj = getattr(obj, key) if kind == "a" else obj[key]
    return obj


def _assign(root, path, value) -> None:
    parent = _resolve(root, path[:-1])
    kind, key = path[-1]
    if kind == "a":
        setattr(parent, key, value)
    else:
        parent[key] = value


def _row_paths(env, max_depth: int = 7) -> Dict[tuple, torch.Tensor]:
    """Every attribute / key path from `env` to a tensor whose first dimension is the batch.

    The walk `viewer.graph_fastpath._tensor_paths` does, on any device (that one collects CUDA
    tensors only, which is what a graph capture needs and would copy nothing on the CPU)."""
    out: Dict[tuple, torch.Tensor] = {}
    seen: set = set()
    atoms = (str, bytes, int, float, bool, type(None))

    def rec(obj, path, depth):
        if torch.is_tensor(obj):
            if obj.dim() >= 1 and obj.shape[0] == env.B:
                out[path] = obj
            return
        if depth > max_depth or isinstance(obj, atoms) or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                rec(v, path + (("k", k),), depth + 1)
        elif isinstance(obj, (list, tuple)) and len(obj) <= 64:
            for i, v in enumerate(obj):
                rec(v, path + (("k", i),), depth + 1)
        elif isinstance(obj, torch.nn.Module):
            for n, t in list(obj.named_parameters()) + list(obj.named_buffers()):
                rec(t, path + tuple(("a", part) for part in n.split(".")), depth + 1)
        elif type(obj).__module__.startswith("f1sim") and hasattr(obj, "__dict__"):
            for k, v in list(vars(obj).items()):
                rec(v, path + (("a", k),), depth + 1)

    rec(env, (), 0)
    return out


def _name(path) -> str:
    return ".".join(str(k) for _, k in path)


def _tile_rows(obj, B: int, K: int) -> None:
    """Give a shallow copy its own K-times tensors for every per-row attribute."""
    for k, v in list(vars(obj).items()):
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B:
            setattr(obj, k, v.repeat(K, *([1] * (v.dim() - 1))).clone())


class ShadowEnv:
    """K copies of an env's races, refreshed from it on demand."""

    def __init__(self, env, K: int, *, lidar_beams: Optional[int] = 36):
        from .gym_env import F1VecEnv
        self.env, self.K = env, int(K)
        cfg = copy.deepcopy(env.cfg)
        if lidar_beams:
            # Nothing the teacher or the opponents read comes from the scan.
            cfg.lidar.n_beams = int(lidar_beams)
        self.S = F1VecEnv(env.sim.tracks, cfg, copy.deepcopy(env.ecfg), num_envs=self.K * env.B,
                          device=str(env.device))
        if getattr(env, "teacher", None) is not None:
            t = copy.copy(env.teacher)
            _tile_rows(t, env.B, self.K)
            self.S.set_teacher(t)
        self.S.reset(seed=0)
        self.S.step(torch.zeros(self.S.B, self.S.act_dim, device=self.S.device))   # lazy state
        self._paths = None

    def refresh(self) -> None:
        """Copy every per-row state tensor of the real env into all K copies."""
        env, S, K = self.env, self.S, self.K
        # Paths are cached, tensors are not: the simulator rebinds some attributes every step
        # (`self.s = s`, `self.lap = lap`), and a cached tensor is a dead one after the first step.
        if self._paths is None:
            rE, rS = _row_paths(env), _row_paths(S)
            # Same trailing shape now: the scan, whose beam count the shadow cuts, drops out here.
            self._paths = [p for p, t in rE.items() if _name(p) not in INDEX_STATE and p in rS
                           and rS[p].dtype == t.dtype and rS[p].shape[1:] == t.shape[1:]]
        for p in self._paths:
            t, s = _resolve(env, p), _resolve(S, p)
            if t.shape[0] != env.B or s.shape[0] != S.B:
                continue
            tiled = t.repeat(K, *([1] * (t.dim() - 1)))
            if s.shape[1:] != t.shape[1:]:           # IMU samples per step alternate 1 / 2
                _assign(S, p, tiled)
            else:
                s.copy_(tiled)
        # Python-side clock and sensor phase: odometry is stamped against `t`, and `_odom_t_prev`
        # (a row tensor, copied above) is a time on the real env's clock.
        S.sim.t = env.sim.t
        S.sim._imu_phase = env.sim._imu_phase


class RolloutTeacher:
    """`plan_action` like any teacher; decides by shadow simulation every `every` calls."""

    def __init__(self, base: RacelineTeacher, env, *,
                 offsets: Sequence[float] = (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6),
                 speeds: Sequence[float] = (1.0, 0.8, 0.6, 0.35),
                 horizon_s: float = 1.5, every: int = 10, lidar_beams: Optional[int] = 36):
        if not isinstance(base, RacelineTeacher):
            raise TypeError(f"RolloutTeacher drives a RacelineTeacher's line, not {type(base).__name__}")
        if 0.0 not in offsets or 1.0 not in speeds:
            raise ValueError("the candidate set must contain the plain raceline (offset 0, speed 1)")
        self.env, self.device = env, env.device
        cand: List[Tuple[float, float]] = [(float(o), float(s)) for o in offsets for s in speeds]
        self.K = len(cand)
        self.c_off = torch.tensor([c[0] for c in cand], device=self.device)
        self.c_spd = torch.tensor([c[1] for c in cand], device=self.device)
        self.i_line = cand.index((0.0, 1.0))
        self.H = max(1, int(round(horizon_s / env.sim.control_dt)))
        self.every = max(1, int(every))
        self.base = base
        self.shadow = ShadowEnv(env, self.K, lidar_beams=lidar_beams)
        self.base_S = copy.copy(base)
        _tile_rows(self.base_S, env.B, self.K)
        B = env.B
        self.choice = torch.full((B,), self.i_line, device=self.device, dtype=torch.long)
        #: (B,) False where every candidate touched something inside the horizon.
        self.last_label_valid = torch.ones(B, dtype=torch.bool, device=self.device)
        self.last_first_hit = torch.full((B,), float(self.H), device=self.device)
        self._calls = 0
        self.decisions = 0

    # The env and the DAgger loop set these on the teacher they label with.
    @property
    def speed_scale(self):
        return self.base.speed_scale

    @speed_scale.setter
    def speed_scale(self, v):
        self.base.speed_scale = v

    @property
    def label_grip(self):
        return self.base.label_grip

    @label_grip.setter
    def label_grip(self, v):
        self.base.label_grip = v

    def _candidate(self, teacher, env, off, spd, plan_speed=None):
        a = teacher.plan_action(env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy, env.tracker.spec,
                                offset=off, plan_speed=env._tracker_plan_speed() if plan_speed is None else plan_speed)
        a = a.clone()
        a[:, -2:] = ((a[:, -2:] + 1.0) * spd[:, None] - 1.0).clamp(-1.0, 1.0)
        return a

    @torch.no_grad()
    def decide(self) -> None:
        env, sh, S, K = self.env, self.shadow, self.shadow.S, self.K
        B = env.B
        sh.refresh()
        # Mirror the per-row settings the env or a DAgger loop may have put on the base teacher.
        for k in ("speed_scale", "label_grip_codes"):
            v = getattr(self.base, k, None)
            setattr(self.base_S, k, v.repeat(K, *([1] * (v.dim() - 1))) if torch.is_tensor(v) and v.shape[:1] == (B,) else v)
        self.base_S.label_grip = self.base.label_grip
        off = self.c_off.repeat_interleave(B)            # shadow row k*B + i drives candidate k
        spd = self.c_spd.repeat_interleave(B)
        hit = torch.zeros(K * B, dtype=torch.bool, device=self.device)
        first = torch.full((K * B,), float(self.H), device=self.device)
        s0 = S.sim.s.clone()
        L = S.sim.track.length[S.sim.tid]
        for t in range(self.H):
            S.step(self._candidate(self.base_S, S, off, spd))
            h = S.sim.car_collision | S.last_result.collision
            first = torch.where(h & ~hit, torch.full_like(first, float(t)), first)
            hit |= h
        prog = (torch.remainder(S.sim.s - s0 + L / 2, L) - L / 2).view(K, B)
        hit, first = hit.view(K, B), first.view(K, B)
        # Clear candidates by progress, a little toward the line on a tie; with none clear, the one
        # whose first contact is latest.
        score = torch.where(hit, -1e3 + first, prog - 0.05 * self.c_off.abs()[:, None])
        self.choice = score.argmax(0)
        self.last_label_valid = ~hit.all(0)
        self.last_first_hit = first.gather(0, self.choice[None])[0]
        self.decisions += 1

    @torch.no_grad()
    def plan_action(self, state, P=None, tid=None, v_max: float = 8.0, spec=None, iters: int = 6,
                    offset=None, idx=None, plan_speed=None):
        if state.shape[0] != self.env.B:
            raise ValueError("RolloutTeacher labels its own env's whole batch")
        if self._calls % self.every == 0:
            self.decide()
        self._calls += 1
        off = self.c_off[self.choice] if offset is None else self.c_off[self.choice] + offset
        return self._candidate(self.base, self.env, off, self.c_spd[self.choice], plan_speed)

    def __call__(self, state, P=None, tid=None, offset=None):
        raise TypeError("RolloutTeacher is a plan-mode teacher; use plan_action")
