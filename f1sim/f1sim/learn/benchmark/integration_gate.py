"""Opponent-independence gate, on a real env with the real controller runtime.

Stub trackers prove wiring. They cannot prove independence, because the property under test belongs
to `gym_env`: `step` pushes the merged learner+opponent plan tensor through one tracker (`:537`) and
the teacher builds its plan from `tracker.spec` (`:483`, `:772`).

What this runs:

  * the **actual** `ControllerRuntime`, installed on the original `env.tracker` **before** it is
    wrapped -- that ordering is the contract (`grip_runtime.py:272-276`); installing after the wrap
    would hook the wrapper rather than the candidate;
  * its **full per-step lifecycle** -- `begin`, `pre_action`, `post_step`, `release` -- because an
    arm constructed but never driven proves nothing about the path a measurement takes;
  * **2 actors x 2 arms** (`legacy`, and `estimated` with the real frozen student) from an identical
    seeded plant and sensor state, spanning cold and warm estimator history;
  * no fallback. A missing estimator or a failed install raises. A gate that quietly downgrades to a
    speed-ceiling tweak would report success for something it never tested.

The control -- the same cases unrouted -- must diverge. A gate that passes both ways measures
nothing, which is exactly how the first version behaved.
"""
from __future__ import annotations

import torch

from .routed_tracker import RoutedTracker

ARMS = ("legacy", "estimated")


def _build(map_id: str, *, envs: int, race_size: int, seed: int, device: str = "cpu",
           tracks=None, racelines=None, force_teacher: bool = False):
    """`tracks` overrides the loaded map -- used to build on a Track carrying an obstacle.

    The env holds `TrackTensors`, not `Track`, so an obstacle must be stamped into the `Track`
    before the env is constructed; there is no way to add one afterwards.
    """
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    if tracks is not None:
        trs, rls = tracks, racelines
    else:
        trs, rls = common.load_tracks([map_id], racelines=True, drop_infeasible=False)
    ecfg = EnvConfig(action_mode="plan", race_size=race_size, opponent="teacher",
                     opp_speed_range=(0.6, 0.8), max_steps=4000)
    env = common.make_env(trs, envs, torch.device(device), ecfg, seed=seed, rls=rls)
    if force_teacher and env.teacher is None:
        if rls is None:
            from f1sim.track import Raceline
            rls = [Raceline.build_cached(t) for t in trs]
        env.set_teacher(common.make_teacher(rls, env))
    env.sim.warmup()
    return env


def _reset(env, seed):
    """`gym_env.reset` returns `(obs, info)`; the controller runtime needs the obs dict itself."""
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r


def _actor(kind: str):
    """Two distinct, deterministic candidate action streams. No checkpoint is loaded.

    A candidate reaches the tracker boundary as an action stream plus an arm, so two streams and two
    arms is the whole 2x2 that boundary can distinguish.
    """
    def stream(obs, k, env):
        a = torch.zeros(env.B, env.act_dim)
        phase = 0.1 if kind == "A" else 0.37
        amp = 0.5 if kind == "A" else -0.8
        a[env.on_policy] = amp * torch.sin(
            torch.arange(env.act_dim, dtype=torch.float32) * (k + 1) * phase)
        return a.clamp(-1.0, 1.0)
    return stream


def _snapshot(env) -> dict:
    """Everything a step reads. Enough to put the world back exactly where it was."""
    sim = env.sim
    snap = {"state": sim.state.clone(), "s": sim.s.clone(), "steps": sim.steps.clone(),
            "t": float(sim.t), "phase": int(sim._imu_phase),
            "P": {k: v.clone() for k, v in sim.P.items()},
            "last_cmd": env.last_cmd.clone(),
            "prev_steer": env.prev_steer_norm.clone(),
            "ep_step": env.ep_step.clone(),
            "odom": sim.odom.state.clone() if hasattr(sim, "odom") else None}
    # The opponent's own configuration is part of the world, not of the candidate. It is drawn at
    # reset from the global RNG, and loading an estimator advances that RNG -- so without these the
    # `estimated` case races a differently-scaled opponent and the gate blames the arm for it.
    for name in ("opp_scale", "cap_scale", "speed_cap", "teacher_race", "on_policy",
                 "gap_prev", "gap_valid"):
        v = getattr(env, name, None)
        snap[name] = v.clone() if torch.is_tensor(v) else v
    return snap


def _restore(env, snap: dict) -> None:
    sim = env.sim
    sim.state.copy_(snap["state"]); sim.s.copy_(snap["s"]); sim.steps.copy_(snap["steps"])
    sim.t = snap["t"]; sim._imu_phase = snap["phase"]
    for k, v in snap["P"].items():
        sim.P[k].copy_(v)
    env.last_cmd.copy_(snap["last_cmd"])
    env.prev_steer_norm.copy_(snap["prev_steer"])
    env.ep_step.copy_(snap["ep_step"])
    if snap["odom"] is not None and hasattr(sim, "odom"):
        sim.odom.state.copy_(snap["odom"])
    for name in ("opp_scale", "cap_scale", "speed_cap", "teacher_race", "on_policy",
                 "gap_prev", "gap_valid"):
        cur, want = getattr(env, name, None), snap.get(name)
        if torch.is_tensor(cur) and torch.is_tensor(want):
            cur.copy_(want)


def probe_at_fixed_state(map_id: str, *, snapshots, actor_kind: str, arm: str,
                         estimator_path: str | None, envs: int, race_size: int, seed: int,
                         routed: bool, device: str = "cpu", warm_probe: dict | None = None):
    """Opponent commands measured from an IDENTICAL world state, varying only actor and arm.

    Free-running comparison cannot test this. `follow_cap` (gym_env.py:489-498) reads every car's
    arc position and the speed of the car ahead, so a teacher opponent legitimately reacts to where
    the candidate is; two actors drive to different places and their opponents *should* differ.

    Restoring the plant before each probe step removes that legitimate coupling and leaves only the
    illegitimate one: whether the candidate's controller reaches the opponent's issued command.
    """
    from f1sim.learn import grip_runtime as gr
    from f1sim.mpc import PlanTracker

    torch.manual_seed(seed)
    env = _build(map_id, envs=envs, race_size=race_size, seed=seed, device=device)
    rt = gr.ControllerRuntime(env=env, arm=arm,
                              estimator_path=estimator_path if arm == "estimated" else None,
                              device=env.device)
    rt.install()
    if routed:
        original = env.tracker
        reference = PlanTracker(env.B, env.device, original.wb, original.s_max, original.v_max)
        env.tracker = RoutedTracker(candidate=original, reference=reference, env=env)

    actor = _actor(actor_kind)
    opp = ~env.on_policy
    learner = env.on_policy
    rows = []
    cold_seen = warm_seen = None
    try:
        # Seed here, not before the build: constructing the estimated arm loads a checkpoint, which
        # advances the global RNG, so a seed set earlier leaves the two arms drawing different IMU
        # noise. That showed up as a 0.278 rad/s yaw-rate difference reaching the opponent through
        # `gym_env.py:536` -- sensor noise, not controller leakage, and it would have been read as
        # a failed independence gate.
        torch.manual_seed(seed)
        obs = _reset(env, seed)
        rt.begin(obs)
        for k, snap in enumerate(snapshots):
            _restore(env, snap)                     # identical plant and sensor state
            rt.pre_action(obs)
            if arm == "estimated" and rt.history is not None:
                # Learner rows only. Counting every row lets an opponent's full history stand in
                # for the candidate's, which is the opposite of what the gate is asking.
                counts = rt.history.inputs()[1].sum(1)[learner]
                lo, hi = int(counts.min()), int(counts.max())
                cold_seen = lo if cold_seen is None else min(cold_seen, lo)
                warm_seen = hi if warm_seen is None else max(warm_seen, hi)
            a = actor(obs, k, env)
            obs, _r, term, trunc, _info = env.step(a)
            rt.post_step(term, trunc)
            rows.append(env.last_cmd[opp].clone())
        if warm_probe is not None and arm == "estimated" and rt.history is not None:
            # Cold and warm are properties of the estimator's history, not of a step count.
            wf = int(rt.history.spec.warm_frames)
            final = rt.history.inputs()[1].sum(1)[learner]
            warm_probe[f"{actor_kind}/{arm}"] = {
                "warm_frames": wf,
                "learner_rows": int(final.numel()),
                "min_valid_seen": cold_seen,
                "max_valid_seen": warm_seen,
                "started_cold": cold_seen is not None and cold_seen < wf,
                # every learner must reach a full history, not merely one of them
                "all_learners_warm": bool(int(final.min()) >= wf)}
    finally:
        rt.release()
        if routed and isinstance(env.tracker, RoutedTracker):
            env.tracker.uninstall()
    return torch.stack(rows)


def reference_snapshots(map_id: str, *, steps: int, envs: int, race_size: int, seed: int,
                        device: str = "cpu"):
    """Probe points from a rollout that keeps every learner alive.

    The reference must be long enough for an estimator history to fill WITHOUT a learner reset. A
    synthetic actor crashes early, and each crash empties that learner's history -- measured, the
    learner histories peaked at 38 of 40 and ended at 22 while only the opponent ever reached a full
    window, so the gate could not honestly claim it had probed a warm candidate.

    So the reference is driven by the raceline teacher, which survives, and it stops at the first
    learner reset rather than continuing past one. Refuses rather than returning a short trajectory:
    a gate that silently probes 20 cold steps proves nothing about a warm estimator.
    """
    torch.manual_seed(seed)
    env = _build(map_id, envs=envs, race_size=race_size, seed=seed, device=device,
                 force_teacher=True)
    obs = _reset(env, seed)
    learner = env.on_policy
    snaps = []
    for _k in range(steps):
        snaps.append(_snapshot(env))
        a = env.teacher.plan_action(env.sim.state, env.sim.P, env.sim.tid,
                                    env.ecfg.v_max_policy, env.tracker.spec)
        obs, _r, term, trunc, _i = _step_any(env, a)
        if bool((term | trunc)[learner].any()):
            raise RuntimeError(
                f"a learner ended after {len(snaps)} of {steps} reference steps on {map_id}; the "
                f"probe needs an uninterrupted trajectory, because a reset empties the estimator "
                f"history the gate is trying to warm")
    return snaps


def _step_any(env, a):
    r = env.step(a)
    return r if isinstance(r, tuple) else (r,)


def run_gate(map_id: str = "gen:control:1400", *, steps: int = 60, envs: int = 2,
             race_size: int = 2, seed: int = 4401, routed: bool = True,
             estimator_path: str | None = None, arms=ARMS, actors=("A", "B")) -> dict:
    """2 actors x len(arms), each probed from an IDENTICAL plant and sensor state.

    No fallback and no skipping: a missing estimator or a failed arm install raises.
    """
    est = estimator_path
    snaps = reference_snapshots(map_id, steps=steps, envs=envs, race_size=race_size, seed=seed)
    cases, keys = [], []
    warm: dict = {}
    for actor_kind in actors:
        for arm in arms:
            cases.append(probe_at_fixed_state(map_id, snapshots=snaps, actor_kind=actor_kind,
                                              arm=arm, estimator_path=est, envs=envs,
                                              race_size=race_size, seed=seed, routed=routed,
                                              warm_probe=warm))
            keys.append(f"{actor_kind}/{arm}")

    base = cases[0]
    deltas = {k: float((c - base).abs().max()) for k, c in zip(keys[1:], cases[1:])}
    same = all(bool(torch.equal(c, base)) for c in cases[1:])

    # Coverage is per estimated case, and every one must both start cold and reach a full history on
    # EVERY learner. A single warm row somewhere, or one case's record overwriting another's, is not
    # coverage -- and a step count is not evidence of either regime.
    est_cases = [k for k in keys if k.endswith("/estimated")]
    covered = bool(est_cases) and all(
        warm.get(k, {}).get("started_cold") and warm.get(k, {}).get("all_learners_warm")
        for k in est_cases)
    return {"routed": routed, "cases": keys, "n_cases": len(cases),
            "opponent_commands_identical": same,
            "max_abs_delta": max(deltas.values()) if deltas else 0.0,
            "per_case_delta": deltas, "steps": steps,
            "estimator_history": warm or None,
            "covers_cold_and_warm": covered,
            "uncovered_cases": [k for k in est_cases
                                if not (warm.get(k, {}).get("started_cold")
                                        and warm.get(k, {}).get("all_learners_warm"))],
            "shape": list(base.shape)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m f1sim.learn.benchmark gate")
    ap.add_argument("--estimator", required=True, help="pinned frozen student for the estimated arm")
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args(argv)
    routed = run_gate(routed=True, steps=a.steps, estimator_path=a.estimator)
    control = run_gate(routed=False, steps=a.steps, estimator_path=a.estimator)
    print(f"cases: {routed['cases']}  cold+warm covered={routed['covers_cold_and_warm']}")
    for k, v in (routed["estimator_history"] or {}).items():
        print(f"  {k}: valid {v['min_valid_seen']}..{v['max_valid_seen']} of {v['warm_frames']} "
              f"on {v['learner_rows']} learners  cold={v['started_cold']} warm={v['all_learners_warm']}")
    print(f"routed   identical={routed['opponent_commands_identical']} "
          f"max|delta|={routed['max_abs_delta']:.3e}")
    print(f"unrouted identical={control['opponent_commands_identical']} "
          f"max|delta|={control['max_abs_delta']:.3e}   (must differ, else vacuous)")
    print(f"unrouted per case: { {k: round(v, 4) for k, v in control['per_case_delta'].items()} }")
    if not routed["opponent_commands_identical"]:
        print("GATE FAILED: the opponent moved with the candidate")
        return 1
    if control["opponent_commands_identical"]:
        print("GATE VACUOUS: the unrouted control did not diverge, so this proves nothing")
        return 2
    if not routed["covers_cold_and_warm"]:
        print(f"GATE INCOMPLETE: cold+warm not covered for {routed['uncovered_cases']}")
        return 3
    print("GATE PASSED: dependency demonstrated unrouted, removed when routed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
