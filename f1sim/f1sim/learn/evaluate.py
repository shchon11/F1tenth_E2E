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

from ..gym_env import EnvConfig
from .. import opponent_events as opp_ev
from ..opponent_events import describe as describe_events, parse_events
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


@torch.no_grad()
def evaluate(ckpt: str, tracks, envs: int, steps: int, speed_cap: float, device, seed=123, cfg: Config = None,
             teacher=False, action_mode="direct", *, protocol="rolling", race_size=1, opponent="policy",
             budget_laps: float | None = None, max_steps: int = 24000, raceline_margin: float | None = None,
             teacher_grip: str = "true", teacher_recover_time: float = 0.0,
             raceline_objective: str | None = None, teacher_limits: dict | None = None,
             speed_mode: str | None = None, dial_offset: float = 0.0,
             opp_speed_range: tuple | None = None, opp_events=(), opp_event_rate: float = 0.0,
             contention_range_m: float = 12.0, attack_range_m: float = 3.0,
             controller: str = "legacy", estimator: str = "") -> dict:
    """Keep rolling metrics compatible; trials count only initial learner attempts.

    budget_laps: derive the step budget from the track length instead of using `steps`.
    opp_events / opp_event_rate: scripted opponent behaviour, as in training. Empty is off, and off
    is the run this function has always done -- nothing is stepped and nothing is drawn from the
    simulator's generator.
    contention_range_m / attack_range_m: the two arc windows the traffic metrics measure over. The
    defaults are the env's own `overtake_range` and the benchmark's tight window.
    controller: the arm to install between the policy and the wheels (`grip_runtime.ARMS`).
    `legacy` installs nothing, which is what every caller before 2026-09-13 got. The arm is built
    and installed after `sim.warmup()` for the same reason `ppo.py` builds it after the graph
    runtime: whatever captures `mpc.solve` last owns the solver.
    """
    if protocol not in ("rolling", "trials") or steps < 1 or envs < 1 or race_size < 1 or envs % race_size:
        raise ValueError("Require a valid protocol, positive steps/envs, and envs divisible by race_size")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rl_kw = {} if raceline_margin is None else {"margin": raceline_margin}
    limits = dict(teacher_limits or {})                # a_lat / a_acc / a_brake: the line is optimised for the
    rl_kw.update(limits)                               # same profile the teacher then drives on it
    if raceline_objective is not None:
        rl_kw["objective"] = raceline_objective
    trs, rls = common.load_tracks(tracks, racelines=teacher or (race_size > 1 and opponent == "teacher"), **rl_kw)
    model = None
    metadata = {}
    if not teacher:
        model, metadata = load_checkpoint(ckpt, device, allow_conditional=True)   # the env can produce a lab oracle's input
        model.eval()
    mode = "plan" if (model is not None and model.meta.get("act_dim", 2) >= 5) or (teacher and action_mode == "plan") else "direct"
    spec = metadata.get("spec", {})
    events = parse_events(opp_events)
    if events and not (race_size > 1 and opponent in ("teacher", "mixed")):
        raise ValueError(f"opponent events {list(events)} need race_size > 1 and a teacher-driven "
                         f"opponent; got race_size {race_size}, opponent {opponent!r}. Without them "
                         f"there is no car for the events to script and the run is silently the "
                         f"unflagged one.")
    if events and not opp_event_rate > 0:
        raise ValueError(f"opponent events {list(events)} at rate {opp_event_rate}: the rate is "
                         f"events per opponent per 10 s, so at 0 nothing ever fires.")
    # A checkpoint says which speed dimensions it was trained to emit; the flag is for the teacher.
    speed_mode = speed_mode or metadata.get("speed_mode", "linear")
    ecfg = EnvConfig(speed_cap=speed_cap, resample_track_on_reset=True, action_mode=mode,
                     speed_mode=speed_mode if mode == "plan" else "linear",
                     plan_a_brake=float(metadata.get("plan_a_brake", limits.get("a_brake", 3.0))),
                     race_size=race_size, opponent=opponent,
                     opp_events=events, opp_event_rate=float(opp_event_rate),
                     scan_stack=spec.get("scan_stack", 3), scan_stride=spec.get("scan_stride", 1),
                     hist_len=spec.get("hist_len", 0), hist_stride=spec.get("hist_stride", 2))
    if opp_speed_range is not None:
        ecfg.opp_speed_range = tuple(float(x) for x in opp_speed_range)
    step_dt = 1.0 / (cfg or Config()).sim.control_rate
    steps = resolve_steps(protocol, steps, budget_laps, trs, speed_cap, step_dt, max_steps)
    if protocol == "trials":
        ecfg.max_steps = steps
    env = common.make_env(trs, envs, device, ecfg, cfg=cfg, seed=seed, rls=rls, teacher_grip=teacher_grip,
                         teacher_recover_time=teacher_recover_time)
    env.sim.warmup()
    ctrl = grip_runtime.ControllerRuntime(env, controller, estimator or None, device=device)
    ctrl.install()
    if teacher:
        teacher_policy = common.make_teacher(rls, env, grip=teacher_grip, recover_time=teacher_recover_time, **limits)
        def policy(obs):
            return env.teacher_label(teacher_policy)
    else:
        # Carries the hidden state and any extra scan channel between steps, and exposes `.reset`
        # so both protocols below can clear them at an episode boundary. For a feedforward
        # checkpoint it is `model.act` with nothing else happening.
        # A dial student is scored with its dial set `dial_offset` away from the floor's true friction:
        # 0 is "set exactly right" (the reference), negative is the safe side an operator would err on.
        dial = None if not dial_offset else (lambda: env.sim.P["mu"].reshape(-1) + float(dial_offset))
        policy = common.student_policy(model, env, device, dial=dial)
    from .benchmark.overtake import TrafficMeter
    meter = TrafficMeter(env, contention_range_m=contention_range_m,
                         attack_range_m=attack_range_m, vehicle_length=(cfg or Config()).vehicle.length)
    if protocol == "rolling":
        # The meter wraps `sim.step` so every quantity is read BEFORE the auto-reset, the same
        # discipline the benchmark's own loop keeps. `rollout_metrics` is left untouched: it is a
        # validated implementation of a different measurement, and a second caller poking at its
        # internals is how two measurements start disagreeing.
        with meter:
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
        with meter:
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
    ctrl_metrics, _ = ctrl.collect_metrics()
    ctrl.release()
    config_report = asdict(env.cfg)
    dropout = config_report['lidar']['dropout_value']
    if not math.isfinite(dropout):
        config_report['lidar']['dropout_value'] = str(dropout)
    result['metadata'] = {
        'protocol': protocol, 'tracks': list(tracks), 'seed': seed,
        'seeds': {'numpy': seed, 'torch': seed, 'simulator': seed, 'reset': seed if protocol == 'trials' else None},
        'checkpoint': str(ckpt), 'teacher': teacher, 'deterministic_policy': True,
        'raceline_objective': raceline_objective or 'min_curvature', 'teacher_limits': limits,
        'speed_mode': env.ecfg.speed_mode, 'dial_offset': float(dial_offset),
        'action_mode': mode, 'device': str(device), 'steps': steps, 'step_dt': env.sim.control_dt,
        'time_budget_s': steps * env.sim.control_dt, 'envs': envs,
        'race_size': race_size, 'opponent': opponent, 'learners': int(env.learner.sum()),
        'opp_speed_range': list(env.ecfg.opp_speed_range),
        'opp_events': list(events), 'opp_event_rate': float(opp_event_rate),
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
    ap.add_argument("--protocol", choices=["rolling", "trials"], default="rolling")
    ap.add_argument("--race-size", type=int, default=1)
    ap.add_argument("--opponent", choices=["policy", "teacher"], default="policy")
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
    # No `--controller` / `--estimator`: the tracker is the tracker (see `ppo.py`). The keyword
    # arguments above remain, because the frozen benchmark scores historical systems under the arm
    # they were measured with and calls this function directly.
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
                    help="line the teacher follows: 'min_time' refines the minimum-curvature line by descending "
                         "the lap time of the speed profile at the --teacher-a-* limits (default: min_curvature)")
    ap.add_argument("--speed-mode", choices=["linear", "envelope", "knots"], default=None,
                    help="what the plan's speed dimensions mean (f1sim.mpc.SPEED_MODES). Default: the checkpoint's "
                         "own, or 'linear' for the teacher")
    ap.add_argument("--dial-offset", type=float, default=0.0,
                    help="dial checkpoints: the dial is set to the floor's true friction plus this (default 0: exactly "
                         "right; -0.15 is an operator erring on the safe side)")
    ap.add_argument("--teacher-a-lat", type=float, default=None,
                    help="[m/s^2] lateral limit of the teacher's speed profile at nominal grip (default 6.0; the car "
                         "measured 9.2-11.5, the simulator's front axle gives out at ~9.5)")
    ap.add_argument("--teacher-a-acc", type=float, default=None, help="[m/s^2] drive limit of the profile (default 6.0)")
    ap.add_argument("--teacher-a-brake", type=float, default=None,
                    help="[m/s^2] braking limit of the profile (default 3.0; the regen limit gives 5.0 nominal here)")
    ap.add_argument("--output", type=Path, help="write the complete strict JSON report")
    ap.add_argument("--eager", action="store_true", help="disable simulator compilation (CPU smoke tests)")
    a = ap.parse_args()
    if a.steps < 1 or a.envs < 1 or a.race_size < 1 or a.envs % a.race_size:
        ap.error("steps/envs/race-size must be positive and envs divisible by race-size")
    if not a.teacher and not a.ckpt:
        ap.error("provide a checkpoint or --teacher")
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
        if not a.opp_event_rate > 0:
            ap.error(f"--opp-events {','.join(a.opp_events)} with --opp-event-rate "
                     f"{a.opp_event_rate}: the rate is events per opponent per 10 s, so at 0 the "
                     f"named events never fire and the run is silently the unflagged one.")
    tracks = common.track_names(a.tracks)


    def run(names, config):
        if a.wheel_model != "default":
            config.vehicle.wheel_model = a.wheel_model == "on"
        return evaluate(a.ckpt, names, a.envs, a.steps, a.speed_cap, a.device, seed=a.seed, cfg=config,
                        teacher=a.teacher, action_mode=a.action_mode, protocol=a.protocol,
                        race_size=a.race_size, opponent=a.opponent,
                        budget_laps=a.budget_laps if a.budget_laps > 0 else None, max_steps=a.max_steps,
                        raceline_margin=a.raceline_margin, teacher_grip=a.teacher_grip,
                        teacher_recover_time=a.teacher_recover_time,
                        raceline_objective=a.raceline_objective, speed_mode=a.speed_mode, dial_offset=a.dial_offset,
                        teacher_limits=common.teacher_limits(a.teacher_a_lat, a.teacher_a_acc, a.teacher_a_brake),
                        opp_speed_range=a.opp_speed_range, opp_events=a.opp_events,
                        opp_event_rate=a.opp_event_rate,
                        contention_range_m=a.contention_range, attack_range_m=a.attack_range,
                        )

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
