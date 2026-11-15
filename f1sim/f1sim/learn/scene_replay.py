"""Races restarted from the seconds before a past hard contact (11-13, the user: "1도 같이 해야하는거 아님?").

Why. Measured on plan25y: the 1-2 s before a hard contact are ~0.3 % of a rollout's decisions, so the policy gets
almost no practice at exactly the moments it fails in (it sees the box, settles the side late: 1.9 m in hits against
3.1 m in clean passes). The user's box design stays as it is -- boxes uniform over the track, every situation that
gives -- and only how OFTEN the critical moments are met changes: a fixed share of the races (`share`) never start
from a fresh grid again once the store holds something; they restart from a stored scene, drive a short episode
(`episode_steps`), and restart from another. The other races are untouched and are the only source of scenes, so the
store is "moments before a hard contact under the user's design", not a loop feeding on its own replays.

A scene is the state `lag` decisions before the contact, and holds what the next decision depends on:
  - the track, the race's box layout (slot for slot), both cars' simulator state (pose, speeds, steering, commands and
    their history, attitude, IMU filter), which cars are in the race, the opponent's driver kind and pace, the cars'
    drawn physics and body sizes;
  - the policy's own inputs at that decision (`pol_*`, written and read by the trainer): the scan stack, the state
    vector (IMU, last commands, last plan, previous branch) and the memory. The memory is KEPT, not cleared: a box that
    was seen a lap earlier and is hidden now is exactly where a memory pays, and a scene that wiped it would train the
    moment without what the car knew (stored-state replay, Kapturowski et al. 2019, R2D2).
(11-16) ... and EVERY other per-car tensor of the env (`ROW_STATE`: found by walking the env like tools/box_replay.py, whose
copies are exact), minus a car's identity and role in the batch and the episode's own bookkeeping (a scene starts a fresh
episode). Before, the opponents came back as fresh drivers: the follower's MPC warm start and last reference, its per-car
delay and calibration, the drivers' committed side / lane, the timed events (defend, yield, corner mode and their clocks),
the decision phase, the held commands -- all redrawn by the restart. Measured on Korea 2026 (tools/scene_selftest.py, sensor
noise off, mean actions, 4 s): solo scenes came back within 8 mm, but 6 of 29 car scenes drifted more than 10 cm (up to
1.6 m) -- the car scenes replayed in plan25z were other situations than the ones that ended in a contact.

How it sits in the env: `F1VecEnv._reset_envs` does its ordinary reset, then calls `apply`, which writes a scene over
every race of its own that reset whole, then the env takes one scan. A car rejoining its race (`restart_cars`) is not a
whole-race reset and passes through untouched.

The trainer drives 4 control steps a decision and a scene has to start ON a decision: started in the middle, the
decision's remaining steps would drive the restored car with what is left of the old plan. `to_boundary` (set by the
trainer before every step: control steps left in this decision after the one about to run) says where the reset falls;
off the boundary the fresh grid is kept for the 1-3 steps that remain and the scene is loaded when they are over.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .. import dynamics as dyn


class SceneReplay:
    #: simulator tensors copied row by row, after `Simulator.reset` has projected the cars on the scene's track
    SIM = ("state", "cmd", "cmd_hist", "ax", "ay", "att", "att_prev", "imu_state", "pose_prev")
    #: the body a car was drawn with (what the other car's LiDAR sees of it)
    CAR = ("car_dims", "car_porosity", "car_rear", "car_rear_porosity")
    #: per-row pace handles of an opponent (`F1VecEnv._slot_speeds`), restored as they stood
    PACE = ("slot_scale", "opp_scale", "cap_scale")

    #: never copied from one race to another: who a row is and what part it plays in the batch (tools/box_replay.py's lists),
    #: the episode's bookkeeping (a scene is a fresh episode), and caches the next step rebuilds
    NEVER_LEAF = set("""eid other_idx ideal_rows tid absent learner learner_ids race slot spawn_order_race teacher_race level_seed ext_mask
external_driven on_policy pool_driven teacher_driven hold_rows map_rows opp_driver slot_cap slot_cap_set slot_scale_hi slot_scale_lo
spawn_abreast _slot_mixed_rows slot_spawn_code ep_step ep_limit lap_start_step prev_lap ep_return ep_progress ep_contacts lap
in_replay""".split())
    NEVER_PART = ("_alt_rows.", "alt_teacher_mask.", ".params.", "_captures", "._g.", "_last_obs", "scene_replay.", "level_replay.")

    @classmethod
    def _rows_walk(cls, env) -> Dict[str, tuple]:
        """{name: path} of every tensor reachable from the env through f1sim objects, lists and dicts whose axis 0 is the cars"""
        B, out, seen = env.B, {}, set()

        def rec(o, path, depth):
            if id(o) in seen:
                return
            seen.add(id(o))
            if torch.is_tensor(o):
                if o.ndim >= 1 and o.shape[0] == B and o.numel() > 0:
                    out[".".join(str(k) for _, k in path)] = path
                return
            if depth > 6 or isinstance(o, (torch.nn.Module, str, bytes, int, float, bool, type(None))):
                return
            if isinstance(o, dict):
                for k, v in list(o.items()):
                    rec(v, path + (("k", k),), depth + 1)
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    rec(v, path + (("i", i),), depth + 1)
            elif type(o).__module__.startswith("f1sim"):
                for k in (list(vars(o).keys()) if hasattr(o, "__dict__") else []):
                    rec(getattr(o, k), path + (("a", k),), depth + 1)
        rec(env, (), 0)
        keep = {}
        for name, path in out.items():
            leaf = str(path[-1][1]) if path else ""
            if leaf in cls.NEVER_LEAF or any(w in "." + name + "." for w in cls.NEVER_PART):
                continue
            keep[name] = path
        return keep

    @staticmethod
    def _resolve(env, path):
        o = env
        try:
            for kind, k in path:
                o = getattr(o, k) if kind == "a" else o[k]
        except (AttributeError, KeyError, IndexError, TypeError):
            return None
        return o if torch.is_tensor(o) else None

    @staticmethod
    def _opp_state(env) -> Dict[str, torch.Tensor]:
        """{key: (B, ...) live tensor} of what the opponents' drivers carry from step to step (11-16)"""
        B, out = env.B, {}
        for i, alt in enumerate(getattr(env, "alt_teachers", []) or []):
            for k, v in vars(alt).items():
                if (torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == B and k.startswith("_")
                        and not any(w in k for w in ("rows", "mask", "code", "idx"))):
                    out[f"opp_alt{i}{k}"] = v
        solver = getattr(getattr(getattr(env, "tracker", None), "_inner", None), "_solver", None)
        for j, v in enumerate(getattr(solver, "_out", None) or []):
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == B:
                out[f"opp_mpc{j}"] = v
        return out

    def __init__(self, env, share: float, size: int, episode_steps: int, seed: int = 0):
        if not 0.0 < share <= 1.0:
            raise ValueError(f"share {share}: the part of the races that restart from scenes, in (0, 1]")
        if getattr(env, "_league", None) or env.ecfg.opponent in ("mixed", "pool"):
            raise NotImplementedError("scene replay with league seats or a per-race draw of who drives the other car: "
                                      "that draw is not part of a scene yet")
        self.env = env
        dev = env.device
        self.G = env.B // env.M
        every = max(int(round(1.0 / share)), 1)
        #: (G,) the races that restart from scenes (every `every`-th, so they are spread over the batch)
        self.races = (torch.arange(self.G, device=dev) % every) == 0
        self.size = int(size)
        self.episode_steps = int(episode_steps)
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(int(seed) + 60013)
        self.data: Optional[Dict[str, torch.Tensor]] = None       # (size, ...) per entry, made by the first `add`
        self.n = 0                                                # scenes held
        self.head = 0                                             # where the next one is written (oldest first out)
        self.to_boundary = 0
        #: (G,) the race is in an episode that began from a scene
        self.in_replay = torch.zeros(self.G, dtype=torch.bool, device=dev)
        self._loaded = []                                         # (races, scene indices) since `take_loaded`
        self.n_loaded = 0
        self.n_added = 0
        #: (11-16) every per-car tensor a scene carries besides the named ones: {name: path}, fixed at the first capture
        self.row_paths: Optional[Dict[str, tuple]] = None

    # ------------------------------------------------------------------ taking
    def capture(self) -> Dict[str, torch.Tensor]:
        """every race as it stands now, (G, ...) per entry (the env's part of a scene)"""
        env, sim, M, G = self.env, self.env.sim, self.env.M, self.G
        lead = torch.arange(G, device=env.device) * M
        per_race = lambda t: t.reshape(G, M, *t.shape[1:]).clone()
        absent = getattr(sim, "absent", None)
        d = {"tid": sim.tid[lead].clone(),
             "absent": per_race(absent) if absent is not None else torch.zeros(G, M, dtype=torch.bool, device=env.device),
             "params": per_race(sim.params.buf.t())}
        for k in self.SIM + self.CAR:
            d["sim_" + k] = per_race(getattr(sim, k))
        for k in self.PACE:
            d["env_" + k] = per_race(getattr(env, k))
        if getattr(env, "slots", None) is not None:
            d["kind"] = per_race(env.slot_kind_code)
        ts = getattr(getattr(env, "teacher", None), "speed_scale", None)
        if torch.is_tensor(ts):
            d["t_scale"] = per_race(ts)
        pr = env.procedural
        if pr is not None and pr.C > 0:
            for k, v in pr.snapshot(lead).items():
                d["box_" + k] = v
        for k, v in self._opp_state(env).items():
            d[k] = per_race(v)
        if self.row_paths is None:
            self.row_paths = self._rows_walk(env)
        for name, path in self.row_paths.items():
            t = self._resolve(env, path)
            if t is not None and t.shape[0] == env.B:
                d["row:" + name] = per_race(t)
        return d

    def add(self, scene: Dict[str, torch.Tensor]) -> None:
        """scenes, (n, ...) per entry, into the store; the oldest make room"""
        n = int(next(iter(scene.values())).shape[0])
        if n == 0:
            return
        if self.data is None:
            self.data = {k: torch.zeros((self.size,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
                         for k, v in scene.items()}
        if set(scene) != set(self.data):
            raise KeyError(f"a scene's entries changed: {sorted(set(scene) ^ set(self.data))}")
        if n > self.size:
            scene, n = {k: v[-self.size:] for k, v in scene.items()}, self.size
        at = (self.head + torch.arange(n, device=self.env.device)) % self.size
        for k, v in scene.items():
            self.data[k][at] = v
        self.head = (self.head + n) % self.size
        self.n = min(self.n + n, self.size)
        self.n_added += n

    # ------------------------------------------------------------------ putting back
    def apply(self, env, ids: torch.Tensor) -> None:
        """called by `F1VecEnv._reset_envs` after the ordinary reset of `ids`"""
        M, dev = env.M, env.device
        in_ids = torch.zeros(env.B, dtype=torch.bool, device=dev); in_ids[ids] = True
        full = in_ids.view(-1, M).all(1)
        self.in_replay &= ~full                              # a fresh grid, until a scene is loaded below
        g = torch.nonzero(full & self.races).flatten()
        if g.numel() == 0 or self.n == 0:
            return
        rows = (g[:, None] * M + torch.arange(M, device=dev)[None, :]).reshape(-1)
        if self.to_boundary > 0:
            env.ep_limit[rows] = int(self.to_boundary)       # the fresh grid lasts until the decision is over
            return
        idx = torch.randint(self.n, (g.numel(),), device=dev, generator=self.gen)
        self.load(g, idx)

    def load(self, g: torch.Tensor, idx: torch.Tensor) -> None:
        """write scenes `idx` over races `g` (whole races that have just been reset)"""
        env, sim, M, D = self.env, self.env.sim, self.env.M, self.data
        dev = env.device
        rows = (g[:, None] * M + torch.arange(M, device=dev)[None, :]).reshape(-1)
        n = rows.numel()
        flat = lambda k: D[k][idx].reshape((n,) + tuple(D[k].shape[2:]))          # (n_races, M, ...) -> (rows, ...)
        sim.tid[rows] = D["tid"][idx].repeat_interleave(M)
        pr = env.procedural
        if pr is not None and pr.C > 0:
            pr.restore(rows, {k[4:]: v[idx].repeat_interleave(M, 0) for k, v in D.items() if k.startswith("box_")})
        st = flat("sim_state")
        absent = flat("absent")
        # the simulator's own reset on the scene's track and poses: arc position, odometry, contact state, counters;
        # it also draws new physics and bodies, which the stored ones then replace
        sim.reset(rows, st[:, :3].to(sim.state.dtype).clone(), st[:, dyn.IVX].to(sim.state.dtype).clone())
        sim.params.buf[:, rows] = flat("params").t().to(sim.params.buf.dtype)
        for k in self.SIM + self.CAR:
            getattr(sim, k)[rows] = flat("sim_" + k)
        sim.unpark(rows[~absent])
        sim.park(rows[absent])                               # an empty seat: off the map at this row's own spot
        speed = sim.state[rows, dyn.IVX].to(sim.odom.state.dtype).clone()
        env.spawn_abreast[rows] = False
        if "kind" in D:
            mixed = env._slot_mixed_rows[rows]
            env.slot_kind_code[rows] = torch.where(mixed, flat("kind"), env.slot_kind_code[rows])
            env._apply_slot_kinds()
        for k in self.PACE:
            getattr(env, k)[rows] = flat("env_" + k)
        env.speed_cap[rows] = env.cap_base * env.cap_scale[rows]
        if "t_scale" in D:
            env.teacher.speed_scale[rows] = flat("t_scale")
        env._reset_bookkeeping(rows, speed)                  # follower, drivers, latches, counters: from the scene's cars
        live = self._opp_state(env)                          # ... and then the drivers' own state as it stood (11-16)
        for k, v in live.items():
            if k in D:
                v[rows] = flat(k).to(v.dtype)
        for name, path in (self.row_paths or {}).items():    # ... and every other per-car tensor (11-16, `ROW_STATE`)
            t = self._resolve(env, path)
            if t is not None and ("row:" + name) in D and t.shape[1:] == D["row:" + name].shape[2:]:
                t[rows] = flat("row:" + name).to(t.dtype)
        env.ep_limit[rows] = self.episode_steps
        self._refill = (rows, idx.repeat_interleave(M), M)
        self.in_replay[g] = True
        self._loaded.append((g, idx))
        self.n_loaded += int(g.numel())

    #: what `F1VecEnv._fill_scans` rewrites after a reset (a fresh scan stack, the action history) -- put back after it
    FILLED = ("scan_hist", "hist", "hist2", "_fresh2")

    def after_fill(self, env) -> None:
        """called by `F1VecEnv._reset_envs` after `_fill_scans`: the scene rows' scan and action histories as they stood
        (the fill rebuilt them from one fresh scan; the opponents' drivers read them -- 11-16)"""
        r = getattr(self, "_refill", None)
        self._refill = None
        if r is None or self.data is None:
            return
        rows, idx_rows, M = r
        D = self.data
        g_of = idx_rows                                       # the scene of each row
        seat = torch.arange(rows.numel(), device=rows.device) % M
        for name in self.FILLED:
            k = "row:" + name
            t = getattr(env, name, None)
            if k in D and torch.is_tensor(t) and t.shape[1:] == D[k].shape[2:]:
                t[rows] = D[k][g_of, seat].to(t.dtype)

    def take_loaded(self):
        """(races, scene indices) loaded since the last call, for the trainer to put the policy's inputs back"""
        if not self._loaded:
            e = torch.zeros(0, dtype=torch.long, device=self.env.device)
            return e, e
        g = torch.cat([a for a, _ in self._loaded]); idx = torch.cat([b for _, b in self._loaded])
        self._loaded = []
        return g, idx
