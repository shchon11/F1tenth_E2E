"""`f1sim_ros eval`: scoring a cell with the graph in the middle of the measurement loop.

The value of the command is that it reuses the benchmark's own `prepare_cell` and `run_cell`, so
the numbers are comparable to a batched row by construction. The risk is everything around that: a
graph running a different speed cap than the suite declares, a command paired with the wrong scan,
a loop so slow the nodes correctly decide the sensors have stopped. Each of those produces a
plausible wrong number rather than an error, so each has a check and each check is tested here.

The end-to-end test wires the three nodes together in one process -- the eval node's publishers
call the policy's and the controller's callbacks directly, and the controller's `/drive` publisher
calls the eval node back. That is the real `score_cell`, the real `run_cell` and the real
`ObsBuilder`; only the transport is replaced, which is what makes it fast enough to be a test.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "f1sim_ros"))

pytest.importorskip("rclpy")
pytest.importorskip("f1sim_interfaces.msg",
                    reason="f1sim_interfaces is not built; see f1sim_interfaces/README.md")
torch = pytest.importorskip("torch")

import rclpy                                                   # noqa: E402
import f1sim_ros.eval_node as ev                               # noqa: E402
from f1sim.learn.benchmark import suite as suite_mod           # noqa: E402


class Logger:
    def __init__(self):
        self.lines = []

    def _add(self, kind):
        return lambda s: self.lines.append((kind, s))

    def __getattr__(self, name):
        return self._add(name)

    def text(self, kind=None):
        return " | ".join(s for k, s in self.lines if kind is None or k == kind)


def bare_node(**kw):
    """An `EvalNode` with only the fields the check under test reads."""
    n = ev.EvalNode.__new__(ev.EvalNode)
    n._log = Logger()
    n.get_logger = lambda: n._log
    n.suite = suite_mod.Suite(speed_cap=9.0)
    n.arm = "fixed_low"
    n.diag = {}
    n.pin_calibration = False
    n.calibration = {"cmd_delay_s": 0.035, "steer_bias_rad": 0.0, "steer_gain": 1.0,
                     "speed_gain": 1.0, "wheelbase_m": 0.3302, "steer_max_rad": 0.4189}
    for k, v in kw.items():
        setattr(n, k, v)
    return n


# ==================================================================== the suite and the cell
def test_a_suite_version_resolves_to_its_frozen_file():
    s, freeze = ev._load_suite("v2.1")
    assert s.version == "v2.1" and len(freeze) == 64
    assert s.freeze_hash() == freeze, "load() verifies the freeze; so should this"


def test_an_unknown_suite_or_cell_says_what_is_available():
    with pytest.raises(SystemExit, match="no suite"):
        ev._load_suite("v9")
    s, _ = ev._load_suite("v2.1")
    with pytest.raises(SystemExit, match="is not a cell of suite v2.1"):
        ev._find_cell(s, "S:nope:0.5:1")
    assert ev._find_cell(s, s.cells()[0].cell_id()) == s.cells()[0]


# ==================================================================== the graph checks
def test_a_graph_running_a_different_speed_cap_is_refused():
    """The policy reads the cap as an OBSERVATION channel, so this is not a scaling difference --
    it is a different policy answering a different question."""
    n = bare_node(diag={"speed_cap_mps": "4.000", "arm": "fixed_low"})
    with pytest.raises(SystemExit, match="speed_cap"):
        n.check_the_graph_is_the_one_the_suite_declares()


def test_a_graph_running_a_different_arm_is_refused():
    n = bare_node(diag={"speed_cap_mps": "9.000", "arm": "legacy"})
    with pytest.raises(SystemExit, match="arm"):
        n.check_the_graph_is_the_one_the_suite_declares()


def test_a_matching_graph_passes_and_is_logged():
    n = bare_node(diag={"speed_cap_mps": "9.000", "arm": "fixed_low", "traction": "off",
                        "cmd_delay_s": "0.03500", "sensor_timeout_s": "10.000"})
    n.check_the_graph_is_the_one_the_suite_declares()
    assert "graph checked" in n._log.text("info")


def test_no_diagnostic_yet_warns_rather_than_claiming_a_check_happened():
    n = bare_node()
    n.check_the_graph_is_the_one_the_suite_declares()
    assert "cannot confirm" in n._log.text("warning")


def test_a_calibration_mismatch_warns_and_is_fatal_only_when_the_env_was_pinned_to_it():
    diag = {"speed_cap_mps": "9.000", "arm": "fixed_low", "cmd_delay_s": "0.01000"}
    n = bare_node(diag=diag)
    n.check_the_graph_is_the_one_the_suite_declares()
    assert "tracker calibration is not the one recorded" in n._log.text("warning")
    pinned = bare_node(diag=diag, pin_calibration=True)
    with pytest.raises(SystemExit, match="pin_calibration"):
        pinned.check_the_graph_is_the_one_the_suite_declares()


def test_a_loop_slower_than_the_graphs_patience_is_refused():
    """Every scan arriving after the last one expired makes the graph correct and the run
    meaningless: the policy would clear its observation history on every frame."""
    n = bare_node(diag={"sensor_timeout_s": "0.250"})
    n.check_the_loop_is_faster_than_the_graphs_patience(0.030)          # 30 ms/step: fine
    with pytest.raises(SystemExit, match="ms per step"):
        n.check_the_loop_is_faster_than_the_graphs_patience(0.200)      # 200 ms/step: not
    bare_node().check_the_loop_is_faster_than_the_graphs_patience(9.0)  # no diag: nothing to check


# ==================================================================== pairing a command to a scan
def test_only_the_command_that_answers_this_scan_counts_as_the_answer():
    """The controller stamps a tracked command with its scan's stamp and a watchdog brake with the
    current time. Taking any `/drive` let the brake published before the first scan existed count
    as the answer to step one, and the cell then started from a command nobody issued."""
    from builtin_interfaces.msg import Time
    from ackermann_msgs.msg import AckermannDriveStamped
    n = bare_node(answers={}, published_keys={}, unsolicited=0, cmd=None, cmd_count=0)

    def drive(sec, steer, speed):
        m = AckermannDriveStamped()
        m.header.stamp = Time(sec=sec, nanosec=0)
        m.drive.steering_angle = steer; m.drive.speed = speed
        return m

    n.on_drive(drive(0, 0.0, 0.0))                     # a brake stamped "now", no scan published
    assert n.answers == {} and n.unsolicited == 1
    n.published_keys[(7, 0)] = 0
    n.on_drive(drive(7, 0.2, 3.0))
    assert n.answers == {(7, 0): (0.2, 3.0)} and n.unsolicited == 1
    n.on_drive(drive(9, 0.9, 9.0))                     # a command for a scan we never published
    assert n.unsolicited == 2 and (9, 0) not in n.answers


def test_the_oldest_scan_stamps_are_forgotten_first():
    """A set would drop them in hash order, which half the time is the scan being waited on -- the
    run then stalls at the first trim and looks exactly like a graph that stopped answering."""
    n = bare_node(answers={}, published_keys={}, unsolicited=0, publishes=0)
    for i in range(ev.KEYS_KEPT + 10):
        key = (i, 0)
        n.published_keys[key] = i
        n.answers[key] = (0.0, float(i))
        while len(n.published_keys) > ev.KEYS_KEPT:
            old, _ = next(iter(n.published_keys.items()))
            del n.published_keys[old]
            n.answers.pop(old, None)
        assert key in n.published_keys, f"the newest key was trimmed at {i}"
    assert min(k[0] for k in n.published_keys) == 10


# ==================================================================== end to end
@pytest.mark.parametrize("sync", [True])
def test_a_cell_is_scored_through_the_graph(tmp_path, sync):
    """The real `score_cell`, with the three nodes wired to each other in one process.

    The suite here is a hand-built one with a tiny lap budget -- the point is the plumbing and the
    row, not a score. A real cell of suite v2.1 takes about a hundred seconds of CPU; the numbers
    for three of them are in docs/research/ros-graph-2026-09-15.md.
    """
    import dataclasses
    import json

    from f1sim.learn.model import ActorCritic, save_checkpoint
    from f1sim.learn.obs import ObsSpec

    spec = ObsSpec(n_beams=1081, scan_stack=2, scan_stride=1, action_history=2, act_dim=8,
                   hist_len=0, range_max=10.0, v_max=10.0)
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=9, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(99)
        model = ActorCritic(**meta)
    ckpt = str(tmp_path / "plan.pt")
    save_checkpoint(ckpt, model, {"spec": dataclasses.asdict(spec)})

    params = {"f1sim_policy": {"checkpoint": ckpt, "device": "cpu", "speed_cap": 4.0,
                               "sensor_timeout": 1e6},
              "f1sim_controller": {"checkpoint": ckpt, "device": "cpu", "speed_cap": 4.0,
                                   "controller": "fixed_low", "sensor_timeout": 1e6,
                                   "plan_timeout": 1e6},
              "f1sim_eval": {"checkpoint": ckpt, "device": "cpu", "arm": "fixed_low",
                             "suite": "v2.1", "cell": "S:gen:competition:0:0.94401:4401",
                             "startup_wait": 5.0, "sync": bool(sync)}}
    import graph_replay as gr
    pf = gr.write_params(str(tmp_path / "eval.yaml"), params)

    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init(args=["--ros-args", "--params-file", pf])
    try:
        from f1sim_ros.controller_node import ControllerNode
        from f1sim_ros.policy_node import PolicyNode
        node = ev.EvalNode()
        policy = PolicyNode()
        controller = ControllerNode()
        # The wire, in one process: every publisher calls the subscriber that would have got it.
        node.pub_scan.publish = lambda m: (controller.on_scan(m), policy.on_scan(m))
        node.pub_odom.publish = lambda m: (policy.on_odom(m), controller.on_odom(m))
        node.pub_imu_raw.publish = lambda m: (policy.on_imu(m), controller.on_imu(m))
        node.pub_gt.publish = lambda m: None
        node.pub_reset.publish = lambda m: (policy.on_reset(m), controller.on_reset(m))
        policy.pub_plan.publish = controller.on_plan
        policy.pub_state.publish = lambda m: None
        controller.pub.publish = node.on_drive
        controller.pub_diag.publish = node.on_diag
        # A lap budget of a few steps: this is about the plumbing and the row.
        node.suite = dataclasses.replace(node.suite, budget_laps=0.01)
        row = ev.score_cell(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    assert row["cell_id"] == "S:gen:competition:0:0.94401:4401"
    assert row["n_envs"] == 1 and row["runtime"] == "fixed_low"
    g = row["graph"]
    assert g["batched"] is False and g["declared_envs"] == 8 and g["learners"] == 1
    assert g["arm_installed_in"] == "controller_node"
    assert g["steps"] > 0 and g["command_timeouts"] == 0
    assert g["answered"] == g["steps"] - g["warmup_steps"], (
        "every step after the warm-up must be answered by the command for that scan")
    assert row["result"]["tally"]["denominator"] == 1
    assert row["suite_freeze_sha256"] and row["identity_sha256"]
    json.dumps(row)                       # the row is what gets written; it has to serialise


def test_the_row_is_not_mistakable_for_a_batched_suite_row(tmp_path):
    """`benchmark report` must refuse it: the suite declares eight trials and one was run. The row
    says so in three places rather than relying on the reader to notice."""
    from f1sim.learn.benchmark import report as report_mod
    s, _ = ev._load_suite("v2.1")
    cell = s.cells()[0]
    row = {"cell_id": cell.cell_id(), "suite": cell.suite, "map_id": cell.map_id, "mu": cell.mu,
           "seed": cell.seed, "n_envs": 1, "runtime": "fixed_low",
           "graph": {"batched": False, "declared_envs": 8, "learners": 1},
           "result": {"n": 1, "tally": {"successes": 0, "denominator": 1, "expected_n": 1}}}
    with pytest.raises(report_mod.ReportError):
        report_mod.validate_cell(row, cell, suite=s)
