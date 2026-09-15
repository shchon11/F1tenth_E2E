"""The fair comparison: their architectures, our teacher, our data, our sensor.

CONTRACT.md, Deliverable 3. End2Race is an imitation framework and so is ours, so the honest
comparison is not "their released weights against our trained ones" -- that measures four
differences at once (expert, data, track set, sensor). It is: **the same expert, the same
demonstrations, the same DAgger schedule, the same evaluation, and only the network in between.**

What is held fixed
------------------
* **The expert.** `f1sim.interactive_teacher.InteractiveTeacher` (worker 17), or `RacelineTeacher`
  while that branch is unmerged -- `--teacher` picks, and the choice is recorded on the checkpoint.
* **The demonstrations.** One collection loop, here. The environment is the one worker 17's D3 runs
  (`work/interactive-teacher/work/d3_dagger.sh`): the training track set, race size 3, teacher
  opponents with the scripted and reactive event set, procedural obstacles from the track list.
  Iteration 0 drives the teacher, so with the same seed it is **bit-identical** across
  architectures; later iterations are each student's own on-policy states, which is what DAgger is
  and what ours does too.
* **The label.** The teacher's *tracked command*: its plan pushed through the same iLQR plan tracker
  the car runs, read back as `env.last_cmd_raw` -- (steer [rad], speed [m/s]). The environment stays
  in `plan` action mode and the student drives through `ext_cmd`, the same external-command path the
  ROS bridge uses, so the label is what the teacher WOULD have commanded at the state the student
  reached, and the opponents behave exactly as they do in our own run.
* **The sensor.** This car's 1081-beam 270 deg 10 m scan, for every architecture. For End2Race that
  is root's declared deviation: 270 evenly spaced beams (one per degree) instead of 360 over a full
  circle, so each learned per-beam `k` still indexes a bearing.
* **The evaluation.** Suite v2, v2.1 family T, the traffic proxy -- the same cells, the same seeds.

What is deliberately NOT held fixed: the loss and the optimiser. Each architecture is trained with
**its own repo's** loss and hyperparameters at their defaults (`end2race/train.py:183-185`,
`TinyLidarNet/train.py:58-62`), because "their architecture under our training recipe" would be a
third thing that is neither theirs nor ours.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .common import BaselineError

#: Where the fair-comparison arm's environment comes from, as worker 17's D3 runs it.
#: Restated here rather than imported because their script is a shell file on another branch; every
#: value is quoted with the flag it corresponds to, and `collection_protocol()` puts them on the
#: checkpoint so a later reader can check them against that script.
D3_ENV = {
    "race_size": 3, "opponent": "teacher", "opp_speed_range": (0.6, 1.15),
    "spawn_order": "random", "speed_cap": 9.0,
    "opp_events": ("brake", "stop", "shift", "defend", "yield", "line", "oblivious"),
    "opp_event_rate": 1.0,
    "opp_defend_prob": 0.3, "opp_yield_prob": 0.2, "opp_line_prob": 0.3,
    "opp_oblivious_prob": 0.1,
    "iters": 8, "steps": 250, "beta0": 0.6, "seed": 701,
}


# --------------------------------------------------------------------------- the buffer
@dataclass
class DemoBuffer:
    """(T, n_learners) demonstrations: the scan the student saw, its speed, the teacher's command.

    Scans are fp16 metres -- 1081 beams at 40 Hz for 250 steps x 86 learners is 46 M values per
    iteration and fp16 is the sensor's own precision to well within its noise floor (`LidarParams`
    measures 7.4 mm on a 10 m beam; fp16's spacing at 10 m is 4.9 mm). Labels stay fp32: they are
    the regression target.

    `new_episode[t, i]` marks a row whose episode began at `t`, so a sequence sampler can refuse to
    run a GRU across a respawn.
    """
    range_max: float
    scan: list = field(default_factory=list)
    speed: list = field(default_factory=list)
    label: list = field(default_factory=list)
    newep: list = field(default_factory=list)

    def add(self, scan_m, speed_mps, label, new_episode):
        self.scan.append(np.asarray(scan_m, dtype=np.float16))
        self.speed.append(np.asarray(speed_mps, dtype=np.float32))
        self.label.append(np.asarray(label, dtype=np.float32))
        self.newep.append(np.asarray(new_episode, dtype=bool))

    def finalize(self):
        self.S = np.stack(self.scan)                    # (T, B, n_beams) fp16 metres
        self.V = np.stack(self.speed)                   # (T, B)
        self.L = np.stack(self.label)                   # (T, B, 2) steer rad, speed m/s
        self.N = np.stack(self.newep)                   # (T, B)
        self.T, self.B = self.S.shape[:2]
        self.scan = self.speed = self.label = self.newep = []
        return self

    def __len__(self):
        return int(self.T * self.B)

    def speed_range(self) -> tuple:
        """`(min_speed, max_speed)` of the label, the two constants `TinyLidarNet/train.py:135`
        scales its speed target by. `min_speed` is 0 there, hard-coded (`train.py:66`), so it is 0
        here; the maximum is the data's."""
        return (0.0, float(self.L[:, :, 1].max()))

    def save(self, path):
        np.savez_compressed(path, scan=self.S, speed=self.V, label=self.L, newep=self.N,
                            range_max=np.float32(self.range_max))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        b = cls(range_max=float(d["range_max"]))
        b.S, b.V, b.L, b.N = d["scan"], d["speed"], d["label"], d["newep"]
        b.T, b.B = b.S.shape[:2]
        return b


# --------------------------------------------------------------------------- collection
def make_collection_env(tracks, rls, n_learners, device, *, spec_beams, range_max, v_max,
                        seed, overrides=None):
    """The environment D3 collects in: worker 17's race, in `plan` action mode.

    `plan` and not `direct`, although every student here emits (steer, speed). Three reasons, in
    order of how much they matter:

    1. the **label** is the teacher's tracked command, and the tracker that produces it is the one
       inside `env.step` -- calling a second tracker for the label would either need its own warm
       start (`PlanTracker.u_prev/u_seq` are per-env state) or corrupt the real one;
    2. the **opponents** are driven through `plan_action` and that same tracker in our own run
       (`gym_env._opponent_actions:986-995`); in `direct` mode they take a different branch
       (`:977-985`) and would not behave identically, so the demonstrations would not be of the same
       traffic;
    3. the student still drives, through `ext_cmd` -- the external-command path the ROS bridge uses,
       applied after the action mapping and before the physics, so the observation, the history and
       `last_cmd` all see the command that was actually driven.
    """
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common

    cfg = dict(D3_ENV)
    cfg.update(overrides or {})
    ecfg = EnvConfig(
        speed_cap=float(cfg["speed_cap"]), action_mode="plan", resample_track_on_reset=True,
        race_size=int(cfg["race_size"]), opponent=str(cfg["opponent"]),
        opp_speed_range=tuple(cfg["opp_speed_range"]), spawn_order=str(cfg["spawn_order"]),
        opp_events=tuple(cfg["opp_events"]), opp_event_rate=float(cfg["opp_event_rate"]),
        opp_defend_prob=float(cfg["opp_defend_prob"]), opp_yield_prob=float(cfg["opp_yield_prob"]),
        opp_line_prob=float(cfg["opp_line_prob"]),
        opp_oblivious_prob=float(cfg["opp_oblivious_prob"]),
        # The observation the ENV builds is not what these students read -- they take the raw scan
        # out of it -- so it is kept at its cheapest. Nothing physical depends on it.
        scan_stack=1, scan_stride=1, action_history=1, hist_len=0,
        v_max_policy=float(v_max))
    from f1sim.params import Config
    c = Config()
    c.lidar.n_beams = int(spec_beams)
    c.lidar.range_max = float(range_max)
    env = common.make_env(tracks, int(n_learners) * int(cfg["race_size"]), device, ecfg, cfg=c,
                          seed=int(seed), rls=rls)
    return env, cfg


def drive_external(env, rows, cmd) -> None:
    """Batched `env.set_external_command` for the learner rows.

    Written out rather than looped over the public one-car method: that method builds a tensor and
    ships it to the device per call, and this runs 86 times per control step. The semantics are
    copied exactly -- steering clamped to the vehicle's lock, speed passed through as an external
    controller commanded it (`gym_env.py:1364-1374`).
    """
    s_max = float(env.s_max)
    c = torch.as_tensor(np.asarray(cmd, dtype=np.float32), device=env.device)
    c = torch.stack([c[:, 0].clamp(-s_max, s_max), c[:, 1]], 1)
    env.ext_cmd[rows] = c
    env.ext_mask[rows] = True
    env._ext_ids.update(int(i) for i in rows.tolist())


def collect(env, teacher, driver, steps: int, beta: float, buf: DemoBuffer, *, v_max: float,
            range_max: float, log=None) -> DemoBuffer:
    """One DAgger iteration. `beta = 1` drives the teacher; below that the student drives.

    `driver` is a `BaselineDriver` already bound to this car's scanner, or None at `beta = 1`.
    """
    rows = torch.nonzero(env.on_policy).flatten()
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    new_ep = np.ones(rows.numel(), dtype=bool)
    if driver is not None:
        driver.reset()
    teacher_rows = None
    t0 = time.time()
    for t in range(steps):
        scan_m = (obs["scan"][rows, 0] * range_max).cpu().numpy().astype(np.float32)
        speed = (obs["speed"][rows, 0] * v_max).cpu().numpy().astype(np.float32)
        # The teacher's plan for every car. The opponents' own actions are overwritten inside
        # `step` by `_opponent_actions`; this only has to be right for the learner rows.
        plan = teacher.plan_action(env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy,
                                   env.tracker.spec)
        if driver is not None and beta < 1.0:
            cmd = driver.command(driver.adapt(scan_m), speed if driver.needs_speed else None)
            # Per row, per iteration: which car the student drives is redrawn every step in DAgger's
            # original formulation, and `beta` is the probability the EXPERT drives.
            use_t = torch.rand(rows.numel(), generator=env.sim.gen, device=env.device) < beta
            teacher_rows = use_t
            driven = rows[~use_t]
            if driven.numel():
                drive_external(env, driven, cmd[(~use_t).cpu().numpy()])
            held = rows[use_t]
            if held.numel():
                env.ext_mask[held] = False
                env._ext_ids.difference_update(int(i) for i in held.tolist())
        obs, _rew, _term, _trunc, info = env.step(plan)
        # Read AFTER the step and BEFORE anything else: `last_cmd_raw` is what the tracker asked for
        # from the pre-step state, i.e. the label for the observation captured above. It is the raw
        # tracker output, not `last_cmd`, which has this particular car's randomised steering and
        # speed calibration divided out of it -- a property of the vehicle, not of the decision.
        label = env.last_cmd_raw[rows].cpu().numpy().astype(np.float32)
        buf.add(scan_m, speed, label, new_ep)
        new_ep = np.zeros(rows.numel(), dtype=bool)
        done_rows = None
        if "final" in info:
            ids = info["final"]["ids"]
            mark = torch.zeros(env.B, dtype=torch.bool, device=env.device)
            mark[ids] = True
            done_rows = mark[rows].cpu().numpy()
            new_ep |= done_rows
        if driver is not None:
            # Per row: a student with memory must not carry a hidden state across a respawn, and
            # `runner.run_cell` clears ours at exactly the same boundary.
            driver.reset(done_rows if done_rows is not None else np.zeros(rows.numel(), bool))
        if log and (t + 1) % max(1, steps // 5) == 0:
            log(f"    step {t + 1}/{steps}  {(time.time() - t0):.0f} s  "
                f"|steer| {np.abs(label[:, 0]).mean():.3f} rad  v {label[:, 1].mean():.2f} m/s")
    env.clear_external_command()
    return buf


def collection_protocol(env, cfg, tracks, teacher_kind: str, teacher_note: str) -> dict:
    """Everything a reader needs to check these demonstrations against worker 17's D3 run."""
    return {"teacher": teacher_kind, "teacher_note": teacher_note,
            "n_tracks": len(tracks), "n_cars": int(env.B),
            "n_learners": int(env.on_policy.sum().item()),
            "race_size": int(env.M), "opponent": str(env.ecfg.opponent),
            "opp_speed_range": list(env.ecfg.opp_speed_range),
            "spawn_order": str(env.ecfg.spawn_order),
            "opp_events": list(env.ecfg.opp_events),
            "opp_event_rate": float(env.ecfg.opp_event_rate),
            "opp_reactive_probs": {k: float(getattr(env.ecfg, f"opp_{k}_prob")) for k in
                                   ("defend", "yield", "line", "oblivious")},
            "speed_cap": float(env.ecfg.speed_cap), "v_max_policy": float(env.ecfg.v_max_policy),
            "n_beams": int(env.n_beams), "range_max": float(env.range_max),
            "fov": float(env.sim.lidar.fov), "control_rate_hz": 1.0 / float(env.sim.control_dt),
            "label": "the teacher's tracked command, env.last_cmd_raw (steer rad, speed m/s)",
            "action_mode": "plan (the student drives through ext_cmd)",
            "iters": int(cfg["iters"]), "steps": int(cfg["steps"]), "beta0": float(cfg["beta0"]),
            "seed": int(cfg["seed"]),
            "matches": "work/interactive-teacher/work/d3_dagger.sh"}


# --------------------------------------------------------------------------- training
#: Which side of the fair comparison each hyperparameter comes from.
#:
#: The DAgger *schedule* is ours, because that is the thing being held equal -- how many iterations,
#: how long each is, how fast beta decays, how much of the aggregate is kept. Everything inside one
#: training round is **theirs, at its repo default**: the loss, the optimiser, the learning rate, the
#: batch size, the gradient clip, the scheduler. A run with their architecture under our optimiser
#: would be a third system that is neither theirs nor ours.
HYPERPARAMETERS = {
    "schedule (ours, work/interactive-teacher/work/d3_dagger.sh)":
        ["iters", "steps", "beta0", "epochs_per_iter", "keep_iters", "envs", "seed"],
    "tinylidarnet (theirs, TinyLidarNet/train.py:58-61,187)":
        ["Adam(lr=5e-5, eps=1e-7)", "loss='huber' (delta=1.0)", "batch=64",
         "glorot_uniform kernels, zero biases", "labels: steer in radians, speed min-max scaled "
         "to [0,1] by train.py:135"],
    "end2race (theirs, End2Race/train.py:28-30,132-135,183-185)":
        ["Adam(lr=1e-3)", "MSE, steer + 0.05 * speed", "batch=16 sequences",
         "ReduceLROnPlateau(min, factor=0.5, patience=10)", "clip_grad_norm=1.0",
         "mask_prob=0.1", "hidden_scale=4", "sequence length 80 steps"],
}


def _sequences(bufs, seq_len, rng, n):
    """`n` random (buffer, start, row) windows of `seq_len` that do not cross an episode boundary.

    Their `SequenceDataset` takes whole successful episodes and nothing else (`train.py:51-70`
    reads only the `success/` directory). A window that spans a respawn is the same mistake wearing
    a different name: the GRU would carry a state across a teleport.
    """
    out = []
    weights = np.array([b.T * b.B for b in bufs], dtype=np.float64)
    weights /= weights.sum()
    guard = 0
    while len(out) < n and guard < 50 * n:
        guard += 1
        k = int(rng.choice(len(bufs), p=weights))
        b = bufs[k]
        if b.T < seq_len:
            continue
        t0 = int(rng.integers(0, b.T - seq_len + 1))
        i = int(rng.integers(0, b.B))
        if b.N[t0 + 1:t0 + seq_len, i].any():          # a respawn inside the window (not at its head)
            continue
        out.append((k, t0, i))
    if not out:
        raise BaselineError(f"no {seq_len}-step window avoids an episode boundary; collect longer "
                            f"iterations or shorten the sequence")
    return out


def train_tinylidarnet(model, bufs, *, epochs: float, device, log, driver_beams_idx,
                       speed_range, batch: int = 64, lr: float = 5e-5, opt=None) -> float:
    """Their loss and optimiser (`TinyLidarNet/train.py:58-61,187-196`), on our demonstrations.

    Labels: steering in radians straight through (`train.py:117` takes `msg.drive.steering_angle`
    unscaled), speed min-max scaled into [0, 1] by `train.py:135` with `min_speed = 0`.
    """
    import torch.nn as nn

    crit = nn.HuberLoss(delta=1.0)                     # Keras `loss='huber'` is delta = 1.0
    opt = opt or torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)   # Keras Adam epsilon
    n_total = sum(len(b) for b in bufs)
    steps = max(1, int(epochs * n_total / batch))
    rng = np.random.default_rng(0)
    lo, hi = speed_range
    weights = np.array([b.T * b.B for b in bufs], dtype=np.float64)
    weights /= weights.sum()
    model.train()
    losses = []
    for i in range(steps):
        k = int(rng.choice(len(bufs), p=weights))
        b = bufs[k]
        t = rng.integers(0, b.T, batch)
        j = rng.integers(0, b.B, batch)
        x = np.asarray(b.S[t, j], dtype=np.float32)[:, driver_beams_idx]
        y = np.asarray(b.L[t, j], dtype=np.float32).copy()
        y[:, 1] = (y[:, 1] - lo) / max(1e-9, hi - lo)                    # train.py:135
        xb = torch.as_tensor(x, device=device).unsqueeze(-1)
        yb = torch.as_tensor(y, device=device)
        loss = crit(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if log and (i + 1) % max(1, steps // 4) == 0:
            log(f"    tln step {i + 1}/{steps}  huber {np.mean(losses[-200:]):.5f}")
    model.eval()
    return float(np.mean(losses[-200:]))


def train_end2race(model, bufs, *, epochs: float, device, log, driver_beams_idx,
                   seq_len: int = 80, batch: int = 16, lr: float = 1e-3, opt=None,
                   sched=None) -> float:
    """Their loss and optimiser (`End2Race/train.py:111-149,183-185`), on our demonstrations.

    Labels are the teacher's tracked command in physical units -- steering in radians and speed in
    m/s -- exactly as their `steer` / `desired_speed` columns are (`demonstration.py:216`), so no
    scaling is applied to either. Their loss weights them `steer + 0.05 * speed`, which only makes
    sense on unscaled units and is the reason it is 0.05.

    Deviation, declared: the speed INPUT is the previous step's **measured** speed. Their evaluation
    feeds that (`eval_singleagent.py:126`) and their training feeds the previous step's *commanded*
    speed instead (`train.py:87`, the `desired_speed` label). The two disagree in their own repo;
    the evaluation convention is the one their driver runs here, so training on the other would fit
    a network to an input it never sees.
    """
    import torch.nn as nn

    crit = nn.MSELoss()
    opt = opt or torch.optim.Adam(model.parameters(), lr=lr)
    n_total = sum(len(b) for b in bufs)
    steps = max(1, int(epochs * n_total / (batch * seq_len)))
    rng = np.random.default_rng(0)
    model.train()
    losses = []
    for i in range(steps):
        wins = _sequences(bufs, seq_len, rng, batch)
        lidar = np.stack([np.asarray(bufs[k].S[t0:t0 + seq_len, j], np.float32)[:, driver_beams_idx]
                          for k, t0, j in wins])                         # (B, T, F)
        lab = np.stack([np.asarray(bufs[k].L[t0:t0 + seq_len, j], np.float32) for k, t0, j in wins])
        # the PREVIOUS step's measured speed, and for the first step of a window the step's own --
        # the same one-step allowance `end2race.End2Race._forward` makes at the start of an episode
        spd = np.stack([np.asarray(bufs[k].V[max(0, t0 - 1):t0 + seq_len - 1, j], np.float32)
                        if t0 > 0 else
                        np.concatenate([bufs[k].V[t0:t0 + 1, j], bufs[k].V[t0:t0 + seq_len - 1, j]])
                        for k, t0, j in wins]).astype(np.float32)
        x = torch.as_tensor(lidar, device=device)
        v = torch.as_tensor(spd, device=device).unsqueeze(-1)
        y = torch.as_tensor(lab, device=device)
        pred, _h = model(x, v, None)
        p = pred.reshape(-1, pred.shape[-1]); tgt = y.reshape(-1, y.shape[-1])
        loss = crit(p[:, 0], tgt[:, 0]) + 0.05 * crit(p[:, 1], tgt[:, 1])   # train.py:130-132
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # train.py:135
        opt.step()
        losses.append(float(loss.detach()))
        if log and (i + 1) % max(1, steps // 4) == 0:
            log(f"    e2r step {i + 1}/{steps}  mse {np.mean(losses[-50:]):.5f}")
    out = float(np.mean(losses[-50:]))
    if sched is not None:
        sched.step(out)
    model.eval()
    return out
