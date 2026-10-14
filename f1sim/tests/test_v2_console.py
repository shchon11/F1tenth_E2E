"""The console drives a v2 checkpoint: the worker's own session path, in process and as a process.

The user works through the console (`f1sim.viewer.sim_worker.SimWorker`), so "a v2 policy runs" is
only true once that path runs it: the env built on observation contract v2 from the checkpoint's own
contract, the actor through `learn.watch.MemoryActorRunner` (its GRU state kept between steps and
cleared at every episode boundary), the grip dial as its input, its WP4 plans tracked by the legacy
controller it was trained with, and the saliency / internals panels working on it.

* in process: the console's actions are, bit for bit, `model.act` with the hidden state carried
  and the session's dial -- before and after a reset, which clears the state;
* as a process (slow): the real spawned worker, over its pipes, starts the session, streams frames
  of a car that moves, follows the dial and the reset commands, and draws the saliency map.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.viewer.console import protocol as P  # noqa: E402

MAP = "gen:competition:0"


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory):
    """A run directory with one v2 checkpoint (deterministic, at the simulator's 1081 beams), or the
    trained one `F1SIM_V2_CONSOLE_CKPT` names."""
    given = os.environ.get("F1SIM_V2_CONSOLE_CKPT")
    if given:
        return given
    from v2_fixtures import generated_v2_checkpoint
    d = tmp_path_factory.mktemp("runs") / "tiny_v2"
    d.mkdir()
    generated_v2_checkpoint(str(d / "student_latest.pt"))
    return str(d)


class _Null:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


def _worker(logs):
    from f1sim.viewer import sim_worker
    w = sim_worker.SimWorker(_Null(), _Null())
    w.say = lambda kind, **f: logs.append((kind, f.get("text") or f.get("message") or ""))
    w.stage = lambda *a, **k: None
    w._hold_parked_ok = True
    return w


def _cfg(run, **kw):
    args = dict(run=run, map_name=MAP, races=1, cars_per_race=1, device="cpu", compile=False, collision_mode="soft")
    args.update(kw)
    return P.SessionConfig(**args)


@pytest.mark.slow
def test_the_console_session_drives_the_v2_actor_exactly_as_model_act_does(v2_run):
    from f1sim.learn.memory import Hidden
    from f1sim.learn.obs import ObsSpecV2, flatten_obs
    logs = []
    w = _worker(logs)
    s = w.build_session(_cfg(v2_run), 0)
    try:
        env, model = s["env"], s["model"]
        assert env.obs_version == 2 and isinstance(model.obs_spec, ObsSpecV2)
        assert type(s["act_fn"]).__name__ == "MemoryActorRunner" and s["dial"] is not None
        facts = w.session_facts(s, 0)
        assert facts["obs_version"] == 2 and facts["controller"] == "legacy" and facts["action_mode"] == "plan"
        assert "v2:" in facts["info_line"]
        acted = []
        run = s["act_fn"]
        s["act_fn"] = lambda scan, ego: acted.append(run(scan, ego).clone()) or acted[-1]
        dial = s["dial"]

        def compare(n, h):
            for _ in range(n):
                scan, ego = flatten_obs(s["obs"])
                with torch.no_grad():
                    want, _lp, h = model.act(scan, ego, deterministic=True, c=dial.c(scan.shape[0]), h=h)
                w._step_once(s)
                assert torch.equal(acted[-1], want), f"console action differs at step {len(acted)}"
            return h
        h = compare(60, Hidden())
        assert float(run.h.abs().sum()) > 0, "the runner must carry the GRU state between steps"
        assert float(env.sim.state[0, 3]) > 0.5, "the car should be moving"
        # the dial is live: the next action reads the new number (the reference reads the same source)
        w._apply_on_sim_thread(s, {"kind": P.CMD_SET_DIAL, "seq": 1, "mu": 1.1})
        assert dial.mu == pytest.approx(1.1)
        h = compare(5, h)
        # a reset is a new episode: the runner's state is cleared, and the actions are a fresh state's
        w._apply_on_sim_thread(s, {"kind": P.CMD_RESET, "seq": 2})
        assert float(run.h.abs().sum()) == 0.0
        compare(20, Hidden())
        # the panels: saliency and the internals of the v2 network (its fusion layer, its stem)
        w._overlay.update(saliency=True, internals=True)
        fr = w._snapshot(s)
        assert fr["plan"] is not None and fr["saliency"].shape == (env.n_beams,) and fr["hidden"].shape == (256,)
        assert np.isfinite(fr["saliency"]).all() and fr["saliency"].max() > 0
    finally:
        w._release_session(s)


@pytest.mark.slow
def test_the_console_drives_a_v2_actor_without_issued_actions(tmp_path):
    """The copycat study's arm A in the console: the session's env emits the 42-wide ego the checkpoint
    reads, and the console's actions are `model.act`'s with the state carried."""
    from f1sim.learn.memory import Hidden
    from f1sim.learn.obs import flatten_obs
    from v2_fixtures import generated_v2_checkpoint
    d = tmp_path / "tiny_v2_a"
    d.mkdir()
    generated_v2_checkpoint(str(d / "student_latest.pt"), ego_actions=False)
    logs = []
    w = _worker(logs)
    s = w.build_session(_cfg(str(d)), 0)
    try:
        env, model = s["env"], s["model"]
        assert env.obs_version == 2 and env.ecfg.ego_actions is False and model.obs_spec.ego_dim == 42
        assert s["obs"]["ego"].shape[1] == 42
        acted = []
        run = s["act_fn"]
        s["act_fn"] = lambda scan, ego: acted.append(run(scan, ego).clone()) or acted[-1]
        h = Hidden()
        for _ in range(40):
            scan, ego = flatten_obs(s["obs"])
            with torch.no_grad():
                want, _lp, h = model.act(scan, ego, deterministic=True, c=s["dial"].c(scan.shape[0]), h=h)
            w._step_once(s)
            assert torch.equal(acted[-1], want), f"console action differs at step {len(acted)}"
        assert float(env.sim.state[0, 3]) > 0.5, "the car should be moving"
    finally:
        w._release_session(s)


@pytest.mark.slow
def test_a_v2_session_takes_the_legacy_controller_only(v2_run):
    from f1sim.viewer.sim_worker import StartConfigError
    logs = []
    w = _worker(logs)
    with pytest.raises(StartConfigError, match="legacy"):
        w.build_session(_cfg(v2_run, controller="fixed_low"), 0)
    s = w.build_session(_cfg(v2_run, controller="auto"), 1)
    try:
        assert w.session_facts(s, 1)["controller"] == "legacy"
        assert any("legacy 로 주행" in text for _k, text in logs)
    finally:
        w._release_session(s)


@pytest.mark.slow
def test_a_v2_session_races_a_v1_league_seat_in_its_own_contract_and_format(v2_run):
    """The console's race against the boot camp's league (UNIT 5b): a v2 session with a v1 checkpoint in
    its opponent table builds (it used to be refused by name), the scripted / league cars plan linear
    while the learner plans wp4, the league seat is shown its own v1 observation, and every car drives."""
    v1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "checkpoints", "iccas_bootcamp_safest.pt")
    if not os.path.exists(v1):
        pytest.skip("archived v1 checkpoint not present")
    logs = []
    w = _worker(logs)
    slots = [{"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [v1], "ckpt_p": 1.0, "speed_scale": [0.8, 0.9]}]
    s = w.build_session(_cfg(v2_run, cars_per_race=2, opponent_slots=slots), 0)
    try:
        env = s["env"]
        assert env.obs_version == 2 and env.ecfg.opponent_speed_mode == "linear"
        ent = env.pool.entries[0]
        assert ent.seat_key is not None and ent.speed_mode == "linear" and ent.seat_spec.scan_stack == 6
        x0 = env.sim.state[:, :2].clone()
        for _ in range(40):
            w._step_once(s)
        assert env.tracker.row_format.tolist() == [1, 0]                 # the learner wp4, the league seat linear
        assert float((env.sim.state[1, :2] - x0[1]).norm()) > 0.5, "the league car should drive"
        assert w._snapshot(s)["plan"] is not None
    finally:
        w._release_session(s)


def test_the_contract_check_names_a_switch_between_v1_and_v2():
    """`obs_spec_problems`, which both the start and the hot reload run: a v2 checkpoint in a v2
    session is accepted field by field; across contracts it is one sentence."""
    from f1sim.learn.model_v2 import ActorCriticV2
    from f1sim.learn.obs import ObsSpec, ObsSpecV2
    from f1sim.viewer.sim_worker import obs_spec_problems
    torch.manual_seed(0)
    m = ActorCriticV2(n_beams=61)
    assert obs_spec_problems(m.meta, ObsSpecV2(n_beams=61), [], 8) == []
    assert obs_spec_problems(m.meta, ObsSpec(n_beams=61), [], 8) == ["관측 계약: 체크포인트 v2 vs 이 세션 v1"]
    v1_meta = {"n_beams": 61, "n_stack": 3, "act_dim": 8}
    assert obs_spec_problems(v1_meta, ObsSpecV2(n_beams=61), [], 8) == ["관측 계약: 체크포인트 v1 vs 이 세션 v2"]
    assert obs_spec_problems(v1_meta, ObsSpec(n_beams=61, scan_stack=3, act_dim=8), [], 8) == []
    bad = obs_spec_problems(m.meta, ObsSpecV2(n_beams=61, n_frames=2), ["actor.mu.weight"], 8)
    assert len(bad) == 2 and "스캔 프레임 수" in bad[0] and "초기값" in bad[1]
    # the copycat study's contract without the issued actions is a different observation, named as such
    no_act = obs_spec_problems(m.meta, ObsSpecV2(n_beams=61, ego_actions=False), [], 8)
    assert len(no_act) == 1 and "지난 행동 열" in no_act[0]
    # and a meta written before the field existed is the contract with them
    old = dict(m.meta, obs={k: v for k, v in m.meta["obs"].items() if k != "ego_actions"})
    assert obs_spec_problems(old, ObsSpecV2(n_beams=61), [], 8) == []


# ==================================================================== the real worker process
@pytest.mark.slow
def test_the_spawned_console_worker_drives_a_v2_checkpoint(v2_run, tmp_path):
    from test_sim_worker_smoke import READY_TIMEOUT, WorkerProc
    wp = WorkerProc(tmp_path)
    try:
        wp.send(P.CMD_START, gen=7, config=_cfg(v2_run).to_dict())
        facts = wp.wait_for(P.MSG_READY, timeout=READY_TIMEOUT, gen=7)["facts"]
        assert facts["obs_version"] == 2 and facts["action_mode"] == "plan" and facts["controller"] == "legacy"
        assert facts["dial"] is not None
        frames = wp.collect_frames(6.0)
        assert len(frames) >= 3, f"only {len(frames)} frames in 6 s"
        assert all(f["gen"] == 7 and np.isfinite(f["x"]).all() for f in frames)
        assert frames[-1]["t"] > frames[0]["t"] and frames[-1]["plan"] is not None
        moved = float(np.hypot(frames[-1]["x"][0] - frames[0]["x"][0], frames[-1]["y"][0] - frames[0]["y"][0]))
        assert moved > 0.05, "the v2 policy should drive the car"
        ack = wp.wait_for(P.MSG_ACK, seq=wp.send(P.CMD_SET_DIAL, mu=1.0), timeout=60)
        assert ack["state"]["dial"] == pytest.approx(1.0)
        wp.wait_for(P.MSG_ACK, seq=wp.send(P.CMD_OVERLAY, overlay={"saliency": True}), timeout=60)
        sal = [f for f in wp.collect_frames(4.0) if "saliency" in f]
        assert sal and np.isfinite(sal[-1]["saliency"]).all()
        wp.wait_for(P.MSG_ACK, seq=wp.send(P.CMD_RESET), timeout=60)
        after = wp.collect_frames(3.0)
        assert after and all(np.isfinite(f["x"]).all() for f in after)
        errs = [m for m in wp.messages if m.get("kind") == P.MSG_ERROR]
        assert not errs, errs[0].get("message")
        rate = [f["sim_rate"] for f in after if f["sim_rate"] > 0]
        print(f"v2 console smoke: {len(frames)} frames in 6 s, moved {moved:.2f} m, sim rate "
              f"{np.median(rate) if rate else float('nan'):.2f}x realtime (CPU)")
    finally:
        wp.close()
