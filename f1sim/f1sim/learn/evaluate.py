"""Evaluate deterministic policies with rolling or first-attempt trial protocols.

Races here can carry the scripted opponent behaviours of `f1sim.opponent_events` -- a car that
brakes, a car that has stopped, a car that moves across the lane -- and every run with an opponent
reports the traffic metrics the benchmark's T family reports, from the same implementation
(`learn.benchmark.overtake.TrafficMeter`). This is the quick check *outside* the frozen suite: it
loads no roster, freezes nothing, and its numbers are not benchmark scores. What it is for is
answering "did that change do anything in traffic" without booking a GPU and a lease.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..gym_env import OPP_FUTURE_MODELS, EnvConfig
from .. import opponent_slots as opp_sl
from .. import opponent_events as opp_ev
from ..opponent_events import describe as describe_events, parse_events, split_events
from ..params import Config
from . import common
from . import grip_runtime
from .evaluation_metrics import TrialAccumulator
from .model import load_checkpoint


def budget_steps(tracks, speed_cap: float, step_dt: float, budget_laps: float, max_steps: int) -> int:
    """Steps needed for `budget_laps` laps of the longest track at the speed cap.

    A fixed step budget silently fails every track longer than `cap * steps * dt`: a 444 m circuit
    at a 4 m/s cap needs 111 s, so a 60 s budget scores it 0 % completion no matter how well the
    policy drives it. Scaling with the track removes that artifact.
    """
    lengths = []
    for t in tracks:
        if getattr(t, "centerline", None) is None:
            raise ValueError(f"track {getattr(t, 'name', t)!r} has no centerline; cannot derive a budget")
        lengths.append(float(np.linalg.norm(np.roll(t.centerline, -1, 0) - t.centerline, axis=1).sum()))
    longest = max(lengths)
    return int(min(max_steps, math.ceil(budget_laps * longest / max(speed_cap, 1e-6) / step_dt)))


def resolve_steps(protocol: str, steps: int, budget_laps: float | None, tracks, speed_cap: float,
                  step_dt: float, max_steps: int) -> int:
    """Step budget for a run. Only the first-attempt protocol scales with the track: `rolling`
    reports a hazard rate, which needs a fixed exposure to stay comparable between tracks."""
    if budget_laps is None or protocol != "trials":
        return steps
    return budget_steps(tracks, speed_cap, step_dt, budget_laps, max_steps)


def _oracle_token(spec: dict, model) -> str:
    """The opponent-token mode a checkpoint was actually built with, or "off"."""
    def _m(d):
        v = str((d or {}).get("opp_token") or "off").strip()
        return "" if v in ("", "off") else v
    return _m(getattr(model, "meta", None) if model is not None else None) or _m(spec) or "off"


@torch.no_grad()
def evaluate(ckpt: str, tracks, envs: int, steps: int, speed_cap: float, device, seed=123, cfg: Config = None,
             teacher=False, action_mode="direct", *, protocol="rolling", race_size=1, opponent="policy",
             budget_laps: float | None = None, max_steps: int = 24000, raceline_margin: float | None = None,
             teacher_grip: str = "true", teacher_recover_time: float = 0.0,
             raceline_objective: str | None = None, teacher_limits: dict | None = None,
             speed_mode: str | None = None, dial_offset: float = 0.0,
             opp_slots="", opp_speed_range: tuple | None = None, opp_events=(), opp_event_rate: float = 0.0,
             opp_reactive_probs: dict | None = None,
             contention_range_m: float = 12.0, attack_range_m: float = 3.0,
             controller: str = "legacy", estimator: str = "",
             teacher_kind: str = "raceline", teacher_speed: float = 1.0,
             mu: float | None = None, teacher_horizon: float = 1.0,
             teacher_cand_iters: int = 2, teacher_cost: str = "",
             opp_future_model: str = EnvConfig.opp_future_model, opp_extra: dict | None = None,
             opp_token_ablate: bool = False,
             external: dict | None = None, research_estimator: bool = False,
             research_profile: str | None = None, graph_runtime: bool = False) -> dict:
    """Keep rolling metrics compatible; trials count only initial learner attempts.

    budget_laps: derive the step budget from the track length instead of using `steps`.
    opp_events / opp_event_rate: scripted opponent behaviour, as in training. Empty is off, and off
    is the run this function has always done -- nothing is stepped and nothing is drawn from the
    simulator's generator.
    opp_reactive_probs: the per-race probability of each REACTIVE disposition
    (`defend`, `yield`, `line`, `oblivious`). These four are not timed events: `opp_event_rate` does
    nothing for them, and their `EnvConfig.opp_<name>_prob` defaults are 0.0 -- so naming one in
    `opp_events` without a probability is a run that lists a behaviour and never produces it, which
    is refused below for exactly the reason a zero rate is.
    contention_range_m / attack_range_m: the two arc windows the traffic metrics measure over. The
    defaults are the env's own `overtake_range` and the benchmark's tight window.
    teacher_speed: scale on the teacher's own speed profile. The teacher plans at the grip limit,
    so 1.0 is the limit and collection has historically used less (`RacelineTeacher.grip_bin`:
    "which is why collection uses speed_scale < 1"). What the right value is has never been
    measured; `--teacher-speed` with `--mu` is how it gets measured.
    mu: pin the plant's friction instead of randomising it, so a number is *about* that friction.
    Randomisation is switched off with it, because with it on `vehicle.mu` is a scale applied to a
    draw and the episode's friction is not the number asked for -- the same argument
    `benchmark.model_adapter.eval_config` makes.
    teacher_kind: which privileged teacher `--teacher` drives. "raceline" is the one this function
    has always driven. "interactive" is `f1sim.interactive_teacher`, which scores a family of plans
    against the opponents' predicted motion -- the only one of the two that can demonstrate a pass,
    and the thing `docs/research/interactive-teacher-2026-09-15.md` measures against the other.
    opp_extra: extra `EnvConfig` fields this function has no parameter of its own for. Two things
    are deliberately NOT among them: the privileged opponent block, which is read off the
    checkpoint's own spec so that a policy trained with it cannot be scored without it by accident
    (`f1sim.opp_token`), and the reactive behaviour probabilities, which travel as
    `opp_reactive_probs` because that route checks them against the events actually named. Either
    passed twice would reach `EnvConfig` twice under one keyword.
    controller: the arm to install between the policy and the wheels (`grip_runtime.ARMS`).
    `legacy` installs nothing, which is what every caller before 2026-09-13 got. The arm is built
    and installed after `sim.warmup()` for the same reason `ppo.py` builds it after the graph
    runtime: whatever captures `mpc.solve` last owns the solver.

    external: a published baseline (`{"kind", "weights", "options"}`) in place of `ckpt`. It runs in
    `direct` action mode with **no controller arm** -- it publishes a command, so there is no plan
    for a tracker to follow -- and it is the same driver object `f1sim_ros.baseline_node` and the
    benchmark's `external` roster kind use, so "the same evaluation" means the same code and not a
    second implementation of it. `controller` must be `legacy` (i.e. nothing installed) with it.
    """
    if protocol not in ("rolling", "trials") or steps < 1 or envs < 1 or race_size < 1 or envs % race_size:
        raise ValueError("Require a valid protocol, positive steps/envs, and envs divisible by race_size")
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Triton compiles for the CURRENT device, not for the device its arguments live on, so on a
    # machine with two different cards an evaluation on the second one gets a kernel built for the
    # first and dies at launch with "no kernel image is available for execution on the device".
    # The arguments are all on `device` already; this is the one thing that says so to Triton.
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index if dev.index is not None else 0)
    cfg = cfg or Config()
    if graph_runtime:
        cfg.sim.compile = False
    if mu is not None:
        cfg.rand.enabled = False
        cfg.vehicle.mu = float(mu)
    rl_kw = {} if raceline_margin is None else {"margin": raceline_margin}
    limits = dict(teacher_limits or {})        # a_lat / a_acc / a_brake: the line is optimised for the
    rl_kw.update(limits)                       # same profile the teacher then drives on it
    if raceline_objective is not None:
        rl_kw["objective"] = raceline_objective
    # A named opponent, if one was asked for. Parsed with the trainer's own parser, so that a table
    # which trains is a table which evaluates -- one spelling, which is what `opponent_slots` is
    # for. Resolved here and not later because whether the racelines are needed depends on it: a
    # slot driven by any teacher kind follows one, and `make_env` would otherwise rebuild them
    # itself, ignoring the margin and the acceleration limits this run was given.
    slots = None
    if opp_slots:
        slots = opp_slots if isinstance(opp_slots, (list, tuple)) else opp_sl.parse_slots(opp_slots)
        if race_size < 2:
            raise ValueError(f"--opp-slots needs a race: race_size is {race_size}. A slot table "
                             f"describes the OTHER cars, and with one car there are none.")
        opp_sl.validate_slots(slots, race_size)
        opponent = "slots"
    want_rls = teacher or (race_size > 1 and opponent == "teacher") or \
        bool(slots and any(opp_sl.kind_of(sl.kind).teacher for sl in slots))
    trs, rls = common.load_tracks(tracks, racelines=want_rls, **rl_kw)
    model = None
    metadata = {}
    policy_checkpoint = {}
    controller_record = None
    driver = None
    if external:
        if teacher or ckpt:
            raise ValueError("external is exclusive with a checkpoint and with --teacher")
        if controller != "legacy":
            raise ValueError(f"a published baseline has no plan tracker, so it cannot wear the "
                             f"{controller!r} arm; leave the controller at 'legacy' (nothing "
                             f"installed) and say so in the report")
        from .benchmark import model_adapter as _ma
        entry = {"kind": external["kind"], "weights": external["weights"], "arm": "none",
                 "options": dict(external.get("options") or {})}
        driver, metadata = _ma.load_external(entry, device)
    elif not teacher:
        # allow_oracle: this function builds the simulator, which is the one thing that can produce
        # the privileged opponent block, and the env below is configured from the checkpoint's own
        # spec so it produces exactly the one the policy was trained on. Every consumer that cannot
        # -- the exporter, the ROS node -- still refuses it.
        policy_checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
        controller_record = grip_runtime.validate_runtime_checkpoint(
            policy_checkpoint, controller, research_estimator=research_estimator)
        model, metadata = load_checkpoint(ckpt, device, allow_oracle=True, allow_conditional=True,
                                           allow_controller=(controller_record.get("arm", "legacy") != "legacy"))
        if controller_record.get("arm") == "auto":
            from .policy_adaptation import require_exact_actor
            require_exact_actor(model, policy_checkpoint)
        model.eval()
    mode = ("direct" if external else
            ("plan" if (model is not None and model.meta.get("act_dim", 2) >= 5)
             or (teacher and action_mode == "plan") else "direct"))
    spec = metadata.get("spec", {})
    if model is not None and not spec:
        raise ValueError("policy checkpoint has no observation specification")
    if spec:
        cfg.lidar.range_max = float(spec.get("range_max", cfg.lidar.range_max))
        if "n_beams" in spec:
            cfg.lidar.n_beams = int(spec["n_beams"])
        if not math.isclose(float(spec.get("att_scale", .35)), .35, rel_tol=0., abs_tol=1e-9):
            raise ValueError("policy attitude normalization is incompatible with the simulator")
    events = parse_events(opp_events)
    if events and not (race_size > 1 and opponent in ("teacher", "mixed")):
        raise ValueError(f"opponent events {list(events)} need race_size > 1 and a teacher-driven "
                         f"opponent; got race_size {race_size}, opponent {opponent!r}. Without them "
                         f"there is no car for the events to script and the run is silently the "
                         f"unflagged one.")
    from ..opponent_events import REACTIVE_NAMES, split_events
    timed, reactive = split_events(events)
    if timed and not opp_event_rate > 0:
        raise ValueError(f"opponent events {list(timed)} at rate {opp_event_rate}: the rate is "
                         f"events per opponent per 10 s, so at 0 nothing ever fires.")
    probs = {k: float((opp_reactive_probs or {}).get(k, 0.0)) for k in REACTIVE_NAMES}
    missing = [k for k in reactive if probs[k] <= 0.0]
    if missing:
        raise ValueError(
            f"reactive opponent behaviour(s) {missing} were named with probability 0. They are not "
            f"timed events -- opp_event_rate does nothing for them -- so the run would list them "
            f"and never produce one. Pass opp_reactive_probs.")
    extra_probs = [k for k, v in probs.items() if v > 0 and k not in reactive]
    if extra_probs:
        raise ValueError(f"probabilities given for {extra_probs}, which are not in opp_events; the "
                         f"run would produce a behaviour the report does not name")
    # A checkpoint says which speed dimensions it was trained to emit; the flag is for the teacher.
    speed_mode = speed_mode or metadata.get("speed_mode", "linear")
    # ... and what its curvature knots mean. Scoring a "feasible" policy under "absolute" multiplies
    # every plan it emits by five at racing speed, so this is read, never assumed.
    exp_meta = metadata.get("experiment") or {}
    kappa_mode = str(exp_meta.get("plan_kappa_mode", "absolute"))
    ecfg = EnvConfig(speed_cap=speed_cap, resample_track_on_reset=True, action_mode=mode,
                     plan_kappa_mode=kappa_mode,
                     plan_kappa_a_lat=float(exp_meta.get("plan_kappa_a_lat", 8.0)),
                     speed_mode=speed_mode if mode == "plan" else "linear",
                     plan_a_brake=float(metadata.get("plan_a_brake", limits.get("a_brake", 3.0))),
                     compile_tracker=not graph_runtime and bool(cfg.sim.compile),
                     race_size=race_size, opponent=opponent,
                     opp_events=events, opp_event_rate=float(opp_event_rate),
                     **{f"opp_{k}_prob": v for k, v in probs.items()},
                     scan_stack=spec.get("scan_stack", 3), scan_stride=spec.get("scan_stride", 1),
                     hist_len=spec.get("hist_len", 0), hist_stride=spec.get("hist_stride", 2),
                     opp_future_model=opp_future_model,
                     action_history=int(spec.get("action_history", EnvConfig.action_history)),
                     v_max_policy=float(spec.get("v_max", EnvConfig.v_max_policy)),
                     imu_gyro_scale=float(spec.get("gyro_scale", EnvConfig.imu_gyro_scale)),
                     imu_accel_scale=float(spec.get("accel_scale", EnvConfig.imu_accel_scale)),
                     # meta FIRST, and 'off'/'' read as absent. `extra["spec"]` records 'off' as a
                     # literal string, and 'off' is truthy, so `spec.get(...) or meta.get(...)`
                     # short-circuits on it and never consults meta -- which is the only place
                     # ActorCritic records the token the network was actually built with (both
                     # writers, model.py:975 and :1484, are guarded by != "off", so meta is either
                     # absent or a real mode). If those two ever disagree, the old order silently
                     # built a NON-oracle env for an oracle checkpoint, and the guard below could
                     # not fire either, because it tests the value this line just produced.
                     opp_token=_oracle_token(spec, model),
                     opp_token_ablate=bool(opp_token_ablate),
                     opponent_slots=slots,
                     **(opp_extra or {}))
    if ecfg.opp_token != "off" and not (race_size > 1 and mode == "plan"):
        raise ValueError(f"this checkpoint was trained with privileged opponent tokens "
                         f"(opp_token={ecfg.opp_token!r}); evaluating it needs race_size > 1 and "
                         f"the plan action space, so that the block exists at all. Got race_size "
                         f"{race_size}, action mode {mode!r}. Feeding it zeros instead would "
                         f"measure a policy driving on an input it was trained to believe.")
    if teacher and teacher_kind == "interactive" and not (race_size > 1 and opponent == "teacher"):
        raise ValueError(f"--teacher-kind interactive with race_size {race_size} and opponent "
                         f"{opponent!r}: its opponent term is identically zero without another car, "
                         f"so the run would be a raceline run under a different name.")
    if teacher and teacher_kind == "interactive" and mode != "plan":
        raise ValueError("--teacher-kind interactive needs --action-mode plan: its candidates are "
                         "plans.")
    if opp_speed_range is not None:
        ecfg.opp_speed_range = tuple(float(x) for x in opp_speed_range)
    step_dt = 1.0 / cfg.sim.control_rate
    steps = resolve_steps(protocol, steps, budget_laps, trs, speed_cap, step_dt, max_steps)
    if protocol == "trials":
        ecfg.max_steps = steps
    env = common.make_env(trs, envs, device, ecfg, cfg=cfg, seed=seed, rls=rls, teacher_grip=teacher_grip,
                         teacher_recover_time=teacher_recover_time)
    if model is not None:
        common.validate_policy_observation(spec, env)
    env.sim.warmup()
    graph_holder = None
    if graph_runtime and torch.device(device).type == "cuda":
        from .graph_runtime import prepare_graph_runtime
        graph_holder = prepare_graph_runtime(env, log=lambda _text: None)
    ctrl = grip_runtime.ControllerRuntime(
        env, controller, estimator or None, device=device,
        research_estimator=research_estimator, research_profile=research_profile,
        checkpoint_meta=(controller_record if controller_record and controller_record.get("arm") == "auto" else None))
    try:
        ctrl.install(graph_rt=graph_holder)
        if policy_checkpoint:
            grip_runtime.validate_runtime_checkpoint(policy_checkpoint, ctrl, research_estimator=research_estimator)
    except BaseException:
        ctrl.release()
        if graph_holder is not None:
            graph_holder.release()
        raise
    if external:
        from .benchmark import model_adapter as _ma
        from ..params import VehicleParams
        policy = _ma.external_policy(driver, spec, float(VehicleParams().s_max))
    elif teacher:
        teacher_policy = common.make_teacher(rls, env, grip=teacher_grip, recover_time=teacher_recover_time, **limits)
        teacher_policy.speed_scale = float(teacher_speed)
        if teacher_kind == "interactive":
            from ..interactive_teacher import InteractiveTeacher, TeacherCost
            w = [float(x) for x in teacher_cost.split(",")] if teacher_cost else None
            teacher_policy = InteractiveTeacher(teacher_policy, env, horizon_s=teacher_horizon,
                                                cost=TeacherCost(*w) if w else TeacherCost(),
                                                cand_iters=teacher_cand_iters,
                                                future_model=opp_future_model)
        def policy(obs):
            return env.teacher_label(teacher_policy)
    else:
        # Carries the hidden state and any extra scan channel between steps, and exposes `.reset`
        # so both protocols below can clear them at an episode boundary. For a feedforward
        # checkpoint it is `model.act` with nothing else happening.
        # A dial student is scored with its dial set `dial_offset` away from the floor's true friction:
        # 0 is "set exactly right" (the reference), negative is the safe side an operator would err on.
        dial = None if not dial_offset else (lambda: env.sim.P["mu"].reshape(-1) + float(dial_offset))
        # A dial student is scored with its dial set `dial_offset` away from the floor's true
        # friction: 0 is "set exactly right" (the reference), negative is the safe side an operator
        # would err on.
        dial = None if not dial_offset else (lambda: env.sim.P["mu"].reshape(-1) + float(dial_offset))
        policy = common.student_policy(model, env, device, dial=dial)
    from .benchmark.overtake import TrafficMeter
    from .encounter import EncounterMeter
    meter = TrafficMeter(env, contention_range_m=contention_range_m,
                         attack_range_m=attack_range_m, vehicle_length=(cfg or Config()).vehicle.length)
    # Counted per opportunity, and with the walls told apart from the crates -- neither of which
    # `collisions_per_km` can do. Both meters wrap `sim.step`; entered in this order the encounter
    # meter wraps the traffic meter's wrapper and unwinds first, so each sees the same transition.
    enc = EncounterMeter(env)
    if protocol == "rolling":
        # The meter wraps `sim.step` so every quantity is read BEFORE the auto-reset, the same
        # discipline the benchmark's own loop keeps. `rollout_metrics` is left untouched: it is a
        # validated implementation of a different measurement, and a second caller poking at its
        # internals is how two measurements start disagreeing.
        with meter, enc:
            result = common.rollout_metrics(env, policy, steps, speed_cap, controller=ctrl)
        for key, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                result[key] = None
    else:
        obs, _ = env.reset(seed=seed)
        ctrl.begin(obs)
        if hasattr(policy, "reset"):
            policy.reset()                      # the trial starts with no memory of anything
        learner = env.learner.clone()
        initial_ids = env.sim.tid[learner].cpu().numpy().copy()
        lengths = env.sim.track.length[env.sim.tid[learner]].cpu().numpy().astype(np.float64)
        if np.any(lengths <= 0):
            raise ValueError("Trial evaluation requires positive centerline lengths")
        trials = TrialAccumulator(lengths, env.sim.control_dt,
                                  time_budget_s=steps * env.sim.control_dt, speed_cap=speed_cap)
        with meter, enc:
            for _ in range(steps):
                ctrl.pre_action(obs)
                obs, _, term, trunc, info = env.step(policy(obs))
                ctrl.post_step(term, trunc)
                if hasattr(policy, "reset"):
                    policy.reset(term | trunc)
                meter.observe(info)
                # Privileged transition velocity is captured before any simulator autoreset.
                speed = torch.linalg.vector_norm(info['priv'][learner, :2], dim=1)
                dynamics = info['priv'][learner, :5].cpu().numpy()
                trials.update(info['progress'][learner].cpu().numpy(), speed.cpu().numpy(),
                              term[learner].cpu().numpy(), trunc[learner].cpu().numpy(), yaw_rate=dynamics[:, 2],
                              heading_error=dynamics[:, 4], longitudinal_speed=dynamics[:, 0], lateral_speed=dynamics[:, 1])
                if not trials.active.any():
                    break
        trials.finish()
        result = trials.report()
        result['initial_track_ids'] = initial_ids.tolist()
        result['initial_track_lengths_m'] = lengths.tolist()
    result.update(meter.report())
    result.update(enc.report())
    ctrl_metrics, _ = ctrl.collect_metrics()
    ctrl.release()
    if graph_holder is not None:
        graph_holder.release()
    config_report = asdict(env.cfg)
    dropout = config_report['lidar']['dropout_value']
    if not math.isfinite(dropout):
        config_report['lidar']['dropout_value'] = str(dropout)
    result['metadata'] = {
        'protocol': protocol, 'tracks': list(tracks), 'seed': seed,
        'seeds': {'numpy': seed, 'torch': seed, 'simulator': seed, 'reset': seed if protocol == 'trials' else None},
        'checkpoint': str(ckpt), 'teacher': teacher, 'teacher_kind': teacher_kind if teacher else None,
        'raceline_objective': raceline_objective or 'min_curvature', 'teacher_limits': limits,
        'speed_mode': env.ecfg.speed_mode, 'dial_offset': float(dial_offset),
        # What the run actually drove under, not what was asked for. The cap is part of the
        # observation (`obs["speed_cap"]`), so a policy trained at 9 and scored at this flag's
        # default of 8 is a policy scored on an input it never saw -- which is what happened to
        # every table in `_eval/mintime-teacher-2026-09-19` before 2026-09-23. It was invisible
        # because the value was never written down.
        'speed_cap': float(env.ecfg.speed_cap), 'max_steps': int(env.ecfg.max_steps),
        'collision_mode': str(env.ecfg.collision_mode),
        'procedural': {'fraction': float(env.ecfg.procedural_obstacles),
                       'density': float(env.ecfg.procedural_density),
                       'max_props': int(env.ecfg.procedural_max_props),
                       'raceline_corridor': str(env.ecfg.procedural_raceline_corridor),
                       'movable': bool(env.ecfg.movable_obstacles),
                       'spawn_runway': float(env.ecfg.spawn_runway)} if env.procedural is not None else None,
        'plan_kappa_mode': str(getattr(env.ecfg, 'plan_kappa_mode', 'absolute')),
        'teacher_speed': float(teacher_speed) if teacher else None,
        'pinned_mu': float(mu) if mu is not None else None,
        'opp_token': (env.opp_token if env.opp_token != "off" else None),
        'opp_token_ablate': (bool(opp_token_ablate) if env.opp_token != "off" else None),
        'opp_future_model': opp_future_model if env.opp_token != "off" else None,
        'external': (metadata.get("external") if external else None),
        'deterministic_policy': True,
        'action_mode': mode, 'device': str(device), 'steps': steps, 'step_dt': env.sim.control_dt,
        'time_budget_s': steps * env.sim.control_dt, 'envs': envs,
        'race_size': race_size, 'opponent': opponent, 'learners': int(env.learner.sum()),
        'opp_speed_range': list(env.ecfg.opp_speed_range),
        'opp_events': list(events), 'opp_event_rate': float(opp_event_rate),
        'opp_reactive_probs': probs,
        'opp_event_note': describe_events(events, float(opp_event_rate)),
        'traffic_convention': (
            'contention/following/attacking are fractions of measured learner-seconds, not of wall '
            'time; pace_vs_opponent pools arc over the learners rather than averaging per-learner '
            'ratios; passes are held passes counted over the rollout, and an auto-reset re-seeds a '
            'pair rather than voiding what it already counted'),
        'race_convention': 'respawning traffic; not strict no-respawn competition',
        'spawn_convention': 'slot 0 starts ahead of following opponents; completion does not establish overtaking',
        'trial_success': 'signed cumulative progress >= initial track length; collision takes precedence' if protocol == 'trials' else None,
        'distance_convention': ('ground-truth planar speed integrated over active control steps' if protocol == 'trials'
                                else 'distance not reported; mean_speed averages post-step longitudinal speed including resets'),
        'instability_convention': ('spin entry: heading error crosses beyond 90 deg; high yaw rate: >5 rad/s; '
                                   'large slip: >20 deg while speed >1 m/s' if protocol == 'trials' else None),
        'config': config_report, 'env_config': asdict(env.ecfg),
        'config_nonfinite_convention': 'nonfinite lidar dropout sentinel encoded as a string',
        'controller_arm': controller, 'wheel_model': bool(env.sim.wheel_model),
        'research_estimator': bool(research_estimator), 'research_profile': research_profile,
        'controller_runtime_version': ctrl.runtime_version if controller == "auto" else None,
        'controller_profile': ctrl.gspec.profile_version if controller == "auto" else None,
        'graph_runtime': graph_holder is not None,
        'graph_runtime_requested': bool(graph_runtime),
    }
    if ctrl_metrics:
        result['controller'] = ctrl_metrics
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--action-mode", default="direct", choices=["direct", "plan"])
    ap.add_argument("--tracks", default="eval", help="'train', 'eval', 'eval_obstacles' or comma separated catalog names")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2400, help="60 s at 40 Hz (ignored when --budget-laps is set)")
    ap.add_argument("--budget-laps", type=float, default=2.0,
                    help="time budget as laps of the track at the speed cap; scales with track length so a long "
                         "circuit is not scored 0 %% for a reason that has nothing to do with the policy. "
                         "0 = use the fixed --steps budget instead")
    ap.add_argument("--max-steps", type=int, default=24000, help="ceiling for the derived budget (10 min at 40 Hz)")
    ap.add_argument("--speed-cap", type=float, default=8.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", action="store_true", help="robustness: mu, latency, lidar height")
    ap.add_argument("--per-track", action="store_true")
    ap.add_argument("--opp-token-ablate", action="store_true",
                    help="DIAGNOSTIC: run a checkpoint trained with privileged opponent tokens with "
                         "its block zeroed, keeping the width. It measures how much the policy's "
                         "driving depends on the oracle -- a large drop means the planner used the "
                         "information, none means it never read it. Not a deployment path: the "
                         "checkpoint is still refused by the exporter and the ROS node, and the "
                         "result is a diagnostic, never a score.")
    ap.add_argument("--protocol", choices=["rolling", "trials"], default="rolling")
    ap.add_argument("--race-size", type=int, default=1)
    ap.add_argument("--opponent", choices=["policy", "teacher"], default="policy")
    ap.add_argument("--opp-slots", default="", metavar="JSON|@FILE",
                    help="per-opponent configuration, the same JSON `--opp-slots` takes in "
                         "training: a JSON array with one object per car of a race "
                         "(race_size - 1 of them), or @path/to/slots.json. This is how a policy is "
                         f"raced against a *named* opponent -- kind is one of "
                         f"{'|'.join(opp_sl.KIND_NAMES)}, which includes the reproduced planners "
                         f"(forzaeth, forzaeth_pred) a published number can be read against. "
                         "Sets --opponent slots; empty is off.")
    ap.add_argument("--opp-speed-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="fraction of its raceline profile each teacher opponent drives at "
                         "(default: the EnvConfig 0.6 0.8). The benchmark's traffic family uses "
                         "0.5 0.7 for a clearly slower car and 0.8 0.95 for one at pace")
    ap.add_argument("--opp-events", default="", metavar="A,B",
                    help=f"comma-separated scripted behaviours for the teacher-driven opponents "
                         f"({','.join(opp_ev.EVENT_NAMES)}); empty = off, and off is bit-identical "
                         f"to a run without them. Needs --race-size > 1 and --opponent teacher")
    ap.add_argument("--opp-event-rate", type=float, default=0.0, metavar="PER10S",
                    help="expected events per teacher opponent per 10 s (events do not overlap, so "
                         "the realized rate is a little below this)")
    ap.add_argument("--contention-range", type=float, default=12.0, metavar="M",
                    help="[m] arc within which a car counts as being in traffic (default: the "
                         "env's own overtake_range)")
    ap.add_argument("--attack-range", type=float, default=3.0, metavar="M",
                    help="[m] the tight window: close enough behind that a pass is actually on. "
                         "The wide window saturates on a 33 m lap, which is why there are two")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--controller", default="legacy",
                    help="controller arm to install between the policy and the wheels; one of "
                         "grip_runtime.ARMS. 'legacy' installs nothing and is the default, so an "
                         "unflagged run is unchanged. '+clearance' arms keep the plan a stated "
                         "margin off the local occupancy built from the scan alone. '+tcs' arms "
                         "run the car's traction guard "
                         "inside the loop and need --wheel-model on.")
    ap.add_argument("--estimator", default="",
                    help="frozen grip-estimator checkpoint; required by the estimated arms")
    ap.add_argument("--research-estimator", action="store_true",
                    help="explicit research-only evaluation of an unapproved observer")
    ap.add_argument("--graphs", action="store_true", help="explicit CUDA physics/controller graphs")
    ap.add_argument("--research-profile", choices=["historical-global-v1"], default=None,
                    help="paired old-auto ablation; requires --research-estimator")
    ap.add_argument("--wheel-model", choices=["on", "off", "default"], default="default",
                    help="vehicle.wheel_model: the rear axle's rotation state, the ERPM odometry "
                         "and the IMU shock term. 'default' leaves params.py's value alone.")
    ap.add_argument("--raceline-margin", type=float, default=None,
                    help="[m] free space the teacher's raceline keeps from the boundary (default 0.40)")
    ap.add_argument("--teacher-grip", choices=["true", "nominal", "conservative"], default="true",
                    help="which grip the teacher's speed profile assumes: 'true' is privileged (unobservable to a "
                         "student), 'nominal'/'conservative' are constant and therefore imitable")
    ap.add_argument("--teacher-recover-time", type=float, default=0.0,
                    help="[s] >0: cap the teacher's commanded speed at what can still be steered back onto the lane "
                         "(v <= a_lat * t / heading_error). 0 keeps the old behaviour, where a car facing "
                         "backwards on the line is told to carry full racing speed")
    ap.add_argument("--raceline-objective", choices=["min_curvature", "min_time"], default=None,
                    help="line the teacher follows (default: min_curvature)")
    ap.add_argument("--teacher-a-lat", type=float, default=None, help="[m/s^2] lateral limit of the teacher's speed profile")
    ap.add_argument("--teacher-a-acc", type=float, default=None, help="[m/s^2] drive limit of the profile")
    ap.add_argument("--teacher-a-brake", type=float, default=None, help="[m/s^2] braking limit of the profile")
    ap.add_argument("--speed-mode", choices=["linear", "envelope", "knots"], default=None,
                    help="what the plan's speed dimensions mean (f1sim.mpc.SPEED_MODES). Default: the checkpoint's own")
    ap.add_argument("--dial-offset", type=float, default=0.0,
                    help="dial checkpoints: the dial is set to the floor's true friction plus this (default 0: exactly "
                         "right; -0.15 is an operator erring on the safe side)")
    ap.add_argument("--teacher-kind", default="raceline", choices=["raceline", "interactive"],
                    help="which privileged teacher --teacher drives. raceline: pure pursuit on the "
                         "precomputed line, blind to the other cars. interactive: "
                         "f1sim.interactive_teacher, which scores a family of plans against the "
                         "opponents' predicted motion -- the only one of the two that can pass")
    ap.add_argument("--teacher-speed", type=float, default=1.0, metavar="SCALE",
                    help="scale on the teacher's own speed profile. The profile already plans at the "
                         "grip limit, so 1.0 IS the limit; collection has historically used less")
    ap.add_argument("--mu", type=float, default=None,
                    help="pin the plant's friction (and switch randomisation off) so the result is "
                         "about that friction. The suite's levels are 0.73423 / 0.94401 / 1.15379")
    ap.add_argument("--teacher-horizon", type=float, default=1.0, metavar="S",
                    help="[s] how far the interactive teacher rolls each candidate out")
    ap.add_argument("--teacher-cand-iters", type=int, default=2, metavar="N",
                    help="Gauss-Newton iterations per candidate plan")
    ap.add_argument("--teacher-cost", default="", metavar="PROG,WALL,OPP,CLEAR,SMOOTH",
                    help="the five interactive-teacher cost weights; empty = its defaults")
    ap.add_argument("--opp-future-model", default=EnvConfig.opp_future_model, choices=list(OPP_FUTURE_MODELS),
                    help="which prediction the interactive teacher reads the opponents with")
    ap.add_argument("--opp-defend-prob", type=float, default=0.0,
                    help="P(a teacher opponent defends the inside line when caught)")
    ap.add_argument("--opp-yield-prob", type=float, default=0.0)
    ap.add_argument("--opp-line-prob", type=float, default=0.0)
    ap.add_argument("--opp-oblivious-prob", type=float, default=0.0,
                    help="the four reactive behaviours' per-race probabilities, as in training. "
                         "They are dispositions, not timed events: --opp-event-rate does nothing "
                         "for them, so a named reactive behaviour at probability 0 is never given "
                         "to anybody and the run would silently be the unflagged one -- refused")
    # The training environment's obstacles and contact model. Without these an evaluation could
    # only score a policy on the empty track, so spec_korea_contact_s911 -- trained on crates on
    # the racing line with soft, recoverable contact -- had no way to be measured on what it was
    # trained for. Same names, types and defaults as `learn.ppo`; unset leaves `EnvConfig`'s.
    ap.add_argument("--procedural-obstacles", type=float, default=None, metavar="FRAC",
                    help="share of resets that draw a fresh obstacle layout (as in training)")
    ap.add_argument("--procedural-density", type=float, default=None, metavar="PER10M")
    ap.add_argument("--procedural-max-props", type=int, default=None, metavar="N")
    ap.add_argument("--procedural-raceline-corridor", choices=["on", "off"], default=None)
    ap.add_argument("--spawn-runway", type=float, default=None, metavar="M")
    ap.add_argument("--collision-mode", choices=["terminate", "soft"], default=None,
                    help="'soft' counts every contact onset per km and lets the car recover; "
                         "'terminate' counts the crashes that end episodes")
    ap.add_argument("--movable-obstacles", action="store_true", default=None)
    ap.add_argument("--output", type=Path, help="write the complete strict JSON report")
    ap.add_argument("--eager", action="store_true", help="disable simulator compilation (CPU smoke tests)")
    ap.add_argument("--external-kind", default="", choices=["", "tinylidarnet", "end2race"],
                    help="evaluate a published baseline (f1sim.learn.baselines) instead of a "
                         "checkpoint: it drives (steer, speed) directly and wears no controller arm")
    ap.add_argument("--external-weights", default="")
    ap.add_argument("--external-options", default="",
                    help="JSON object of driver options, e.g. '{\"scan_fill\": 0.0}'")
    a = ap.parse_args()
    if a.steps < 1 or a.envs < 1 or a.race_size < 1 or a.envs % a.race_size:
        ap.error("steps/envs/race-size must be positive and envs divisible by race-size")
    if a.external_kind and not a.external_weights:
        ap.error("--external-kind needs --external-weights")
    if a.external_weights and not a.external_kind:
        ap.error("--external-weights needs --external-kind")
    external = None
    if a.external_kind:
        if a.teacher or a.ckpt:
            ap.error("--external-kind is exclusive with a checkpoint and with --teacher")
        try:
            opts = json.loads(a.external_options) if a.external_options else {}
        except ValueError as exc:
            ap.error(f"--external-options is not JSON: {exc}")
        if not isinstance(opts, dict):
            ap.error("--external-options must be a JSON object")
        external = {"kind": a.external_kind, "weights": a.external_weights, "options": opts}
    if not a.teacher and not a.ckpt and not external:
        ap.error("provide a checkpoint, --teacher, or --external-kind")
    # Refused here rather than silently ignored, the same way training refuses it: a run that names
    # events and then runs without them looks like evidence that the events change nothing.
    try:
        a.opp_events = parse_events(a.opp_events)
    except ValueError as exc:
        ap.error(f"--opp-events: {exc}")
    if a.opp_events:
        if a.race_size < 2 or a.opponent != "teacher":
            ap.error(f"--opp-events {','.join(a.opp_events)} needs --race-size > 1 and --opponent "
                     f"teacher: the events script the teacher-driven cars of a race, and there are "
                     f"none here (--race-size {a.race_size}, --opponent {a.opponent}).")
        timed, react = split_events(a.opp_events)
        if timed and not a.opp_event_rate > 0:
            ap.error(f"--opp-events {','.join(timed)} with --opp-event-rate "
                     f"{a.opp_event_rate}: the rate is events per opponent per 10 s, so at 0 the "
                     f"named events never fire and the run is silently the unflagged one.")
        for name in react:
            flag = f"--opp-{name}-prob"
            if not 0.0 < float(getattr(a, f"opp_{name}_prob")) <= 1.0:
                ap.error(f"--opp-events {name} with {flag} "
                         f"{getattr(a, f'opp_{name}_prob')}: the probability is how often a "
                         f"teacher-driven car is given that behaviour for a race, so at 0 it is "
                         f"never given and the run is silently the unflagged one.")
    tracks = common.track_names(a.tracks)
    env_extra = {k: v for k, v in (("procedural_obstacles", a.procedural_obstacles),
                                   ("procedural_density", a.procedural_density),
                                   ("procedural_max_props", a.procedural_max_props),
                                   ("procedural_raceline_corridor", a.procedural_raceline_corridor),
                                   ("spawn_runway", a.spawn_runway),
                                   ("collision_mode", a.collision_mode),
                                   ("movable_obstacles", a.movable_obstacles)) if v is not None}

    def run(names, config):
        if a.wheel_model != "default":
            config.vehicle.wheel_model = a.wheel_model == "on"
        return evaluate(a.ckpt, names, a.envs, a.steps, a.speed_cap, a.device, seed=a.seed, cfg=config,
                        teacher=a.teacher, action_mode=a.action_mode, protocol=a.protocol,
                        opp_token_ablate=a.opp_token_ablate,
                        race_size=a.race_size, opponent=a.opponent, opp_slots=a.opp_slots,
                        budget_laps=a.budget_laps if a.budget_laps > 0 else None, max_steps=a.max_steps,
                        raceline_margin=a.raceline_margin, teacher_grip=a.teacher_grip,
                        raceline_objective=a.raceline_objective, speed_mode=a.speed_mode, dial_offset=a.dial_offset,
                        teacher_limits=common.teacher_limits(a.teacher_a_lat, a.teacher_a_acc, a.teacher_a_brake),
                        teacher_recover_time=a.teacher_recover_time,
                        opp_speed_range=a.opp_speed_range, opp_events=a.opp_events,
                        opp_event_rate=a.opp_event_rate,
                        teacher_kind=a.teacher_kind, teacher_speed=a.teacher_speed, mu=a.mu,
                        teacher_horizon=a.teacher_horizon,
                        teacher_cand_iters=a.teacher_cand_iters, teacher_cost=a.teacher_cost,
                        opp_future_model=a.opp_future_model,
                        opp_reactive_probs={"defend": a.opp_defend_prob, "yield": a.opp_yield_prob,
                                            "line": a.opp_line_prob,
                                            "oblivious": a.opp_oblivious_prob},
                        contention_range_m=a.contention_range, attack_range_m=a.attack_range,
                        controller=a.controller, estimator=a.estimator, external=external,
                        research_estimator=a.research_estimator, research_profile=a.research_profile,
                        graph_runtime=a.graphs, opp_extra=env_extra or None)

    nominal = Config()
    if a.eager:
        nominal.sim.compile = False
    res = {t: run([t], nominal) for t in tracks} if a.per_track else {"nominal": run(tracks, nominal)}
    if a.sweep:
        for name, grp, key, vals in (("mu", "vehicle", "mu", [0.7, 0.85, 1.05]),
                                     ("cmd_delay", "actuator", "cmd_delay", [0.02, 0.06, 0.1]),
                                     ("lidar_z", "lidar", "mount_z", [0.12, 0.18])):
            for value in vals:
                cfg = Config()
                cfg.rand.enabled = False
                cfg.sim.compile = not a.eager
                setattr(getattr(cfg, grp), key, value)
                if a.per_track:
                    for track in tracks:
                        res[f"{name}={value}/{track}"] = run([track], cfg)
                else:
                    res[f"{name}={value}"] = run(tracks, cfg)
    report = json.dumps(res, indent=2, allow_nan=False)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
