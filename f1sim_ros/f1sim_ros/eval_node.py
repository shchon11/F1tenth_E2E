"""`f1sim_ros eval`: score one suite cell by driving it through the ROS graph.

    ros2 run f1sim_ros eval --ros-args \
        -p suite:=v2.1 -p cell:='S:gen:control:1400:0.73423:4401' \
        -p checkpoint:=$HOME/f1sim_runs/ppo_race_0910/ppo_latest.pt -p out:=cell.jsonl

The batched runner scores a cell by stepping 8 learners inside one process and calling the actor on
a tensor. This scores the same cell with the car's own boundary in the middle of the loop: the
simulator publishes `/scan`, `/odom` and `/sensors/imu/raw`, `policy_node` turns them into a plan,
`controller_node` turns the plan into `/drive`, and the simulator applies it. Nothing about the
measurement changes -- the env is built by `benchmark/model_adapter.prepare_cell` and scored by
`benchmark/runner.run_cell`, the same two functions `python -m f1sim.learn.benchmark run` calls, so
the numbers are comparable by construction rather than by a second implementation agreeing.

What is different, and stated in every row this writes:

* **one learner, not eight.** A graph has one `/drive`. `n_envs` is 1 and `graph.batched` is false,
  which is what stops a row from being mistaken for a suite row -- `benchmark report` refuses it,
  correctly, because eight trials were declared and one was run.
* **suites are still scored batched.** This is a boundary check and a demo, not a second
  leaderboard: 80 cells x 8 trials over topics is days of wall clock (see the throughput below).
* **the graph's staleness timeouts have to fit this loop, not the car's.** `policy_node` and
  `controller_node` measure freshness against the wall clock, because on a car that is the only
  clock that means anything -- a scan from 300 ms ago is stale whatever the reason. A CPU
  simulator steps at about 0.2x real time, so a step takes longer than the 0.25 s default and both
  nodes would (correctly) conclude the sensors had stopped, clear their histories and drive on a
  one-frame observation. `graph_eval.launch.py` therefore raises `sensor_timeout` and
  `plan_timeout`, and this node refuses the run if the measured step period gets close to what the
  controller reports on its diagnostics. It is a property of the harness, not of the graph.
* **`sync` (default true) steps the simulator only after the command for the step it published has
  come back.** That makes the run deterministic and the comparison against the batched path exact
  up to the arm's own arithmetic. `sync:=false` free-runs at the control rate and drops whatever is
  late, which is what the car does; the difference between the two is the number worth reporting,
  and `docs/research/ros-graph-2026-09-15.md` reports it.
"""
import json
import os
import time

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Empty as EmptyMsg

from f1sim_ros.controller_node import stamp_key
from f1sim_ros.sim_messages import gt_odom_message, imu_messages, odom_message, scan_message, \
    scan_sweep_seconds


#: Steps to wait for the controller's first diagnostic before giving up on checking the graph
#: against the suite. A couple of control periods; the controller publishes one per command.
CHECK_WITHIN_STEPS = 20

#: Scan stamps kept, so a command can still be matched to the scan it answers. Generous: the
#: memory is a tuple per step, and a command that arrives later than this is one the loop already
#: gave up on.
KEYS_KEPT = 512


class CommandTimeout(RuntimeError):
    """The graph did not answer a published scan. In `sync` mode that voids the run: a cell scored
    with commands that never arrived measures the machine, not the policy."""


class EvalNode(Node):
    def __init__(self):
        super().__init__("f1sim_eval")
        d = self.declare_parameter
        d("suite", "v2.1"); d("cell", ""); d("checkpoint", ""); d("device", "cpu")
        d("out", ""); d("arm", "fixed_low")
        d("envs", 1); d("sync", True); d("command_timeout", 5.0); d("max_seconds", 0.0)
        d("publish_gt", True); d("system_id", "")
        # The graph takes a moment to exist (a checkpoint load) and then a couple of scans to start
        # answering (the policy inhibits until it has a fresh attitude, an IMU mean and an odom
        # reading -- exactly as it does on the car). Neither is a failure, so neither trips the
        # command timeout: `startup_wait` is how long to wait for the two nodes to appear, and
        # while nothing has EVER answered, a step that goes unanswered within `warmup_timeout`
        # advances the simulator with a zero command instead of voiding the run.
        d("startup_wait", 120.0); d("warmup_timeout", 0.5)
        # The tracker's model of the car. The simulator draws a per-car command latency and
        # actuator calibration for every episode (`gym_env._calibrate_tracker`: +-20 ms of
        # residual latency, +-0.01 rad of servo offset, +-4 % / +-3 % of gain) and the BATCHED
        # runner's tracker is given them. The controller has only these nominal numbers -- which is
        # the real car's situation exactly, since on the car they are parameters somebody
        # measured. `pin_calibration` overrides the simulator's draw with these, which is what
        # isolates "the graph" from "the graph does not know this car's calibration"; both numbers
        # are worth having and the row says which one it is.
        d("pin_calibration", False)
        # The first commands of the cell, in the row. Cheap, and it is the first thing anybody
        # comparing a graph run against a batched one needs: a divergence that starts at step 0 is
        # a wiring difference, one that starts at step 200 is the trajectory.
        d("record_commands", 40)
        d("cmd_delay", 0.035); d("steer_bias", 0.0); d("steer_gain", 1.0); d("speed_gain", 1.0)
        d("wheelbase", 0.3302); d("steer_max", 0.4189)
        p = lambda n: self.get_parameter(n).value

        self.device = str(p("device"))
        self.sync = bool(p("sync"))
        self.command_timeout = float(p("command_timeout"))
        self.startup_wait = float(p("startup_wait"))
        self.warmup_timeout = float(p("warmup_timeout"))
        self.max_seconds = float(p("max_seconds"))
        self.publish_gt = bool(p("publish_gt"))

        self.suite, self.freeze = _load_suite(str(p("suite")))
        self.cell = _find_cell(self.suite, str(p("cell")))
        self.declared_envs = int(self.cell.envs)
        self.arm = str(p("arm"))
        self.checkpoint = str(p("checkpoint"))
        self.system_id = str(p("system_id")) or f"graph/{os.path.basename(self.checkpoint)}"
        self.out = str(p("out"))
        self.envs = int(p("envs"))
        self.pin_calibration = bool(p("pin_calibration"))
        self.record_commands = int(p("record_commands"))
        self.commands_log = []
        self.calibration = {"cmd_delay_s": float(p("cmd_delay")),
                            "steer_bias_rad": float(p("steer_bias")),
                            "steer_gain": float(p("steer_gain")),
                            "speed_gain": float(p("speed_gain")),
                            "wheelbase_m": float(p("wheelbase")),
                            "steer_max_rad": float(p("steer_max"))}

        self.pub_scan = self.create_publisher(LaserScan, "/scan", 1)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 1)
        self.pub_gt = self.create_publisher(Odometry, "/ego_racecar/odom", 1)
        # `/sensors/imu/raw` and NOT `/sensors/imu`. On the car the summary topic is
        # `vesc_msgs/VescImuStamped`, which is the type both graph nodes subscribe to; the
        # simulator bridge publishes a plain `sensor_msgs/Imu` there, so that edge never connects
        # and the attitude reaches the graph on the raw topic in both cases. Publishing an `Imu`
        # here would add a second publisher of a third type to a topic nothing in the graph reads,
        # and in a single process it is a hard type conflict.
        self.pub_imu_raw = self.create_publisher(Imu, "/sensors/imu/raw", 10)
        self.pub_reset = self.create_publisher(EmptyMsg, "/f1sim/reset", 1)
        self.create_subscription(AckermannDriveStamped, "/drive", self.on_drive, 1)
        # The controller says what it is doing on its own diagnostics topic, which is how this node
        # checks the graph it is scoring against the suite it is scoring for -- without reaching
        # into another process's parameters.
        self.create_subscription(DiagnosticArray, "/f1sim/controller/diag", self.on_diag, 1)
        self.diag = {}

        self.cmd = None                  # the newest /drive, or None until one arrives
        self.cmd_count = 0
        self.timeouts = 0
        self.warmup_steps = 0
        self.publishes = 0
        #: `/drive` by the stamp of the scan it answers. The controller stamps a tracked command
        #: with that scan's stamp and a zero-speed inhibit or brake with the current time, so this
        #: is how the loop tells "the command for the step I just published" from "the controller
        #: is braking because it has nothing". Waiting for any `/drive` instead let the very first
        #: watchdog brake -- published before a single scan existed -- count as the answer to step
        #: one, and the cell then started from a command the policy never issued.
        self.answers = {}
        #: Scan stamps this node has published, oldest first. A dict and not a set because it is
        #: trimmed, and a set's iteration order is its hash order -- trimming one dropped arbitrary
        #: keys including, half the time, the scan being waited on, which stalled the run at the
        #: first trim and looked exactly like a graph that had stopped answering.
        self.published_keys = {}
        self.unsolicited = 0
        self.get_logger().info(
            f"eval: suite {self.suite.version} ({self.freeze[:12]}) cell {self.cell.cell_id()} "
            f"on {self.device}, {self.envs} learner(s) of the {self.declared_envs} the suite "
            f"declares, arm {self.arm} (installed in controller_node, not here), "
            f"{'sync' if self.sync else 'free-running'}")

    # ---------------------------------------------------------------- the graph
    def on_drive(self, m: AckermannDriveStamped):
        self.cmd = (float(m.drive.steering_angle), float(m.drive.speed))
        self.cmd_count += 1
        key = stamp_key(m.header.stamp)
        if key is None or key not in self.published_keys:
            # A zero-speed inhibit or a watchdog brake: the controller stamps those with the
            # current time, not with a scan's. Counted, not applied -- the loop is waiting for the
            # command that answers a specific scan.
            self.unsolicited += 1
        else:
            self.answers[key] = self.cmd

    def wait_for_graph(self):
        """Wait for something to be listening to `/scan` and publishing `/drive`.

        Without this, launching the three nodes together means the eval publishes its first scans
        into an empty graph while `policy_node` is still loading 27 MB of weights, and the run is
        scored from a standing start it never recovers from.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.startup_wait:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.count_subscribers("/scan") and self.count_publishers("/drive"):
                self.get_logger().info(
                    f"graph up after {time.monotonic() - t0:.1f} s: "
                    f"{self.count_subscribers('/scan')} scan subscriber(s), "
                    f"{self.count_publishers('/drive')} /drive publisher(s)")
                return True
        self.get_logger().warning(
            f"no complete graph after {self.startup_wait:.0f} s "
            f"({self.count_subscribers('/scan')} scan subscriber(s), "
            f"{self.count_publishers('/drive')} /drive publisher(s)): scoring anyway, which will "
            f"be a cell of zero-speed commands.")
        return False

    def on_diag(self, m: DiagnosticArray):
        for st in m.status:
            for kv in st.values:
                self.diag[kv.key] = kv.value

    def check_the_graph_is_the_one_the_suite_declares(self):
        """The suite's speed cap and the controller's arm, against what the graph says it is running.

        `speed_cap` is not a ceiling the tracker applies and nothing else: the POLICY reads it as an
        observation channel, so a graph capped at 4 m/s scoring a suite that declares 9 is a
        different policy answering a different question, and every number would be wrong in a way
        nothing else would show. The controller publishes both on its diagnostics, so this is a
        check and not a convention.
        """
        want = float(self.suite.speed_cap)
        got = self.diag.get("speed_cap_mps")
        if got is None:
            self.get_logger().warning(
                "no /f1sim/controller/diag yet: cannot confirm the graph's speed cap or arm "
                f"against the suite's ({want} m/s, arm {self.arm}).")
            return
        if abs(float(got) - want) > 1e-6:
            raise SystemExit(
                f"the controller is running speed_cap {got} m/s and suite {self.suite.version} "
                f"declares {want}. The policy reads the cap as an observation channel, so this is "
                f"not a scaling difference -- relaunch the graph with speed_cap:={want}.")
        arm = self.diag.get("arm")
        if arm is not None and arm != self.arm:
            raise SystemExit(f"the controller is running arm {arm!r} and this row would be "
                             f"recorded as {self.arm!r}. Pass -p arm:={arm} or relaunch the "
                             f"controller with controller:={self.arm}.")
        wrong = {k: (v, self.diag[k]) for k, v in self.calibration.items()
                 if k in self.diag and abs(float(self.diag[k]) - v) > 1e-5}
        if wrong and self.pin_calibration:
            raise SystemExit(
                f"pin_calibration is on and the simulator was pinned to this node's numbers, but "
                f"the controller is running different ones: "
                + ", ".join(f"{k} {a} here vs {b} there" for k, (a, b) in sorted(wrong.items())))
        if wrong:
            self.get_logger().warning(
                "the controller's tracker calibration is not the one recorded in this row: "
                + ", ".join(f"{k} {a} vs {b}" for k, (a, b) in sorted(wrong.items())))
        self.get_logger().info(f"graph checked: arm {arm}, speed cap {got} m/s, "
                               f"traction {self.diag.get('traction')}, "
                               f"cmd_delay {self.diag.get('cmd_delay_s')} s, "
                               f"sensor timeout {self.diag.get('sensor_timeout_s')} s")

    def check_the_loop_is_faster_than_the_graphs_patience(self, seconds_per_step: float):
        """A step slower than the graph's staleness threshold makes the graph correct and the run
        meaningless.

        Both nodes measure freshness against the wall clock. If this loop takes longer per step
        than `sensor_timeout`, every scan arrives after the previous one has expired: the policy
        clears its observation history and its episode memory on every frame, the controller clears
        its tracker warm start, and the cell is scored on a policy that never sees motion. It fails
        loudly instead, because the failure is otherwise invisible -- the numbers look like a bad
        policy rather than a bad harness.
        """
        want = self.diag.get("sensor_timeout_s")
        if want is None:
            return
        want = float(want)
        if seconds_per_step > 0.5 * want:
            raise SystemExit(
                f"this loop is taking {seconds_per_step * 1e3:.0f} ms per step and the graph calls "
                f"a sensor stale after {want * 1e3:.0f} ms. Every scan would arrive after the last "
                f"one expired, and the policy would run on a one-frame history. Relaunch the graph "
                f"with sensor_timeout and plan_timeout of at least "
                f"{max(1.0, 4 * seconds_per_step):.0f} s (graph_eval.launch.py does), or run this "
                f"on a device that holds the control rate.")

    def publish_sensors(self, env, r):
        """Car 0's sensors, exactly as `bridge_node` publishes them -- same builders.

        Odometry and the IMU before the scan, because that is the order the car produces them and
        the order the policy consumes them: `on_scan` is what runs the actor, and it reads the
        speed and the IMU mean that arrived before it.
        """
        now = self.get_clock().now()
        from rclpy.duration import Duration
        od = r.odom[0].cpu().numpy(); st = r.state[0].cpu().numpy()
        self.pub_odom.publish(odom_message(od, now.to_msg()))
        if self.publish_gt:
            self.pub_gt.publish(gt_odom_message(st, now.to_msg()))
        if r.imu is not None and r.imu.shape[1] > 0:
            raw, _summary = imu_messages(r.imu[0].cpu().numpy(), r.imu_att[0].cpu().numpy(),
                                         r.imu_offsets.cpu().numpy(),
                                         lambda dt: (now - Duration(seconds=dt)).to_msg())
            for m in raw:
                self.pub_imu_raw.publish(m)
        meta = env.sim.scan_meta()
        ranges = r.scan[0].cpu().numpy()
        sweep = scan_sweep_seconds(meta, len(ranges))
        stamp = (now - Duration(seconds=sweep)).to_msg()
        self.pub_scan.publish(scan_message(meta, ranges, stamp))
        self.publishes += 1
        key = stamp_key(stamp)
        self.published_keys[key] = self.publishes
        while len(self.published_keys) > KEYS_KEPT:      # oldest first; the newest always survives
            old, _ = next(iter(self.published_keys.items()))
            del self.published_keys[old]
            self.answers.pop(old, None)
        return key

    def wait_for_command(self, key):
        """Spin until the `/drive` that answers the scan stamped `key` arrives.

        In `sync` mode the simulator does not advance without it, so the loop cannot silently score
        a cell the graph was too slow for, and it cannot mistake a watchdog brake for a command.
        Before the graph has answered anything at all the policy is legitimately inhibiting -- it
        does that on the car too, until it has a fresh attitude, IMU mean and odometry -- so those
        steps advance with a zero command rather than voiding the run.
        """
        warming = not self.answers
        deadline = self.warmup_timeout if warming else self.command_timeout
        t0 = time.monotonic()
        while key not in self.answers:
            rclpy.spin_once(self, timeout_sec=0.002)
            if time.monotonic() - t0 > deadline:
                if warming:
                    self.warmup_steps += 1
                    return None
                self.timeouts += 1
                if self.sync:
                    raise CommandTimeout(
                        f"no /drive for the scan published {self.command_timeout:.1f} s ago "
                        f"({self.publishes} published, {len(self.answers)} answered). The policy "
                        f"or the controller stopped, or is inhibiting mid-run.")
                return self.cmd
        return self.answers[key]

    def free_run_command(self, key, period: float):
        """Take the answer to this scan if it arrives inside one control period, else the newest
        command there is -- which is what a car does with a late command."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < period:
            rclpy.spin_once(self, timeout_sec=min(0.002, max(0.0, period - (time.monotonic() - t0))))
            if key in self.answers:
                return self.answers[key]
        self.timeouts += 1
        return self.cmd

    def announce_reset(self):
        self.pub_reset.publish(EmptyMsg())


def _load_suite(name: str):
    """A frozen suite: a path, or a version whose shipped example freeze to use."""
    from f1sim.learn.benchmark import suite as suite_mod
    if os.path.exists(name):
        return suite_mod.load(name)
    here = os.path.dirname(os.path.abspath(suite_mod.__file__))
    path = os.path.join(here, f"suite-{name}.example.json")
    if not os.path.exists(path):
        raise SystemExit(f"no suite {name!r}: pass a frozen suite JSON, or one of "
                         f"{sorted(f[6:-13] for f in os.listdir(here) if f.startswith('suite-'))}")
    return suite_mod.load(path)


def _find_cell(suite, cell_id: str):
    cells = {c.cell_id(): c for c in suite.cells()}
    if cell_id in cells:
        return cells[cell_id]
    raise SystemExit(f"{cell_id!r} is not a cell of suite {suite.version}. Examples:\n  " +
                     "\n  ".join(sorted(cells)[:6]) + f"\n  ... ({len(cells)} in all)")


def _checkpoint_arm(path: str) -> str:
    """The controller arm a checkpoint was TRAINED under. `legacy` for everything written before
    controller-arm training existed."""
    from f1sim.learn.model import controller_arm_of
    return controller_arm_of(torch.load(path, map_location="cpu", weights_only=False))


def _entry(checkpoint: str, arm: str, system_id: str):
    """A roster entry in the shape the adapter and `protocol_identity` read.

    The arm recorded here is `legacy`: the env must NOT install one, because the arm under test is
    `controller_node`'s and installing a second copy on the env's tracker would apply it twice. The
    declared arm travels in the row as `runtime`.
    """
    import hashlib
    h = hashlib.sha256()
    with open(checkpoint, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()

    class _E:
        system_id = None
        checkpoint_sha256 = sha
        controller_arm = "legacy"
        estimator_sha256 = None
        estimator_path = None
        cross_runtime = False
        path = checkpoint

        def resolved(self):
            return checkpoint

    e = _E(); e.system_id = system_id
    return e, {"path": checkpoint, "arm": "legacy", "cross_runtime": False,
               "system_id": system_id, "checkpoint_sha256": sha,
               "estimator_path": None, "estimator_sha256": None}


def score_cell(node: EvalNode):
    """Build the cell the benchmark's way, drive it over the graph, score it the benchmark's way."""
    import dataclasses

    from f1sim.learn.benchmark import __main__ as bm
    from f1sim.learn.benchmark import model_adapter as ma

    entry, entry_d = _entry(node.checkpoint, node.arm, node.system_id)
    _model, extra = ma.load_actor(entry_d, node.device)
    cell = dataclasses.replace(node.cell, envs=node.envs)
    prepared, s_obs, router = bm._prepare(entry_d, extra, cell, node.suite, node.device)
    env = prepared.env
    period = float(env.sim.control_dt)
    started = time.monotonic()

    # A reset of the LEARNER's row is an episode boundary for the graph too: the policy's memory
    # and the controller's tracker history describe a run that is over. Announced on the topic both
    # nodes listen to, from inside `_reset_envs`, which is where the env decides it.
    original_reset = env._reset_envs

    def reset_envs(ids):
        out = original_reset(ids)
        if node.pin_calibration and getattr(env, "tracker_delay", None) is not None:
            # After the reset, because `_reset_envs` calls `_calibrate_tracker`, which draws a
            # fresh residual for every restarting car. Pinning once before the run would be undone
            # by the first auto-reset and the cell would be half one thing and half the other.
            c = node.calibration
            env.tracker_delay[:] = c["cmd_delay_s"]
            env.tracker_cal[:, 0] = c["steer_bias_rad"]
            env.tracker_cal[:, 1] = c["steer_gain"]
            env.tracker_cal[:, 2] = c["speed_gain"]
        try:
            if ids is not None and int(ids.numel()) and bool((ids == 0).any()):
                node.announce_reset()
        except Exception:
            node.announce_reset()
        return out

    env._reset_envs = reset_envs

    checked = []

    def graph_policy(_obs):
        """The 'policy' the benchmark loop calls: publish, wait for `/drive`, apply it to car 0.

        The action returned is ignored for car 0 -- `set_external_command` overrides it -- and the
        opponents keep being driven by the env's own teacher, which is what keeps an O or T cell's
        opponent independent of the candidate.
        """
        if node.max_seconds and time.monotonic() - started > node.max_seconds:
            raise TimeoutError(f"eval exceeded max_seconds={node.max_seconds}")
        if node.publishes == CHECK_WITHIN_STEPS and node.answers:
            node.check_the_loop_is_faster_than_the_graphs_patience(
                (time.monotonic() - started) / node.publishes)
        r = env.last_result
        key = node.publish_sensors(env, r)
        cmd = (node.wait_for_command(key) if node.sync
               else node.free_run_command(key, period))
        if cmd is None:
            cmd = (0.0, 0.0)               # nothing has ever answered: the safe command, as a mux
        elif not checked:
            # The arm and the speed cap the controller is actually running are on its diagnostics
            # topic. Retried until one arrives rather than checked once: the first `/drive` can
            # beat the first diagnostic through the middleware, and a check that quietly did not
            # happen is worse than no check. Checked here rather than before the loop because
            # there is no `env.last_result` to publish from until `run_cell`'s seeded reset.
            if node.diag:
                checked.append(True)
                node.check_the_graph_is_the_one_the_suite_declares()
            elif node.publishes > CHECK_WITHIN_STEPS:
                checked.append(True)
                node.get_logger().warning(
                    f"no /f1sim/controller/diag after {node.publishes} steps: scoring without "
                    f"confirming the graph's arm and speed cap against the suite's "
                    f"({node.suite.speed_cap} m/s, arm {node.arm}).")
        env.set_external_command(0, cmd[0], cmd[1])
        if len(node.commands_log) < node.record_commands:
            node.commands_log.append([round(cmd[0], 6), round(cmd[1], 6)])
        return torch.zeros(env.B, env.act_dim, device=env.device)

    node.wait_for_graph()
    t0 = time.perf_counter()
    try:
        res = bm._run_one(prepared, graph_policy, cell, s_obs, node.suite)
    finally:
        if router is not None:
            router.uninstall()
        env._reset_envs = original_reset
        prepared.close()
    wall = time.perf_counter() - t0

    from f1sim.learn.benchmark import suite as suite_mod
    row = dict(suite_mod.protocol_identity(node.suite, entry, effective=prepared.protocol))
    row.update({
        "suite": cell.suite, "variant": cell.variant, "map_id": cell.map_id, "mu": cell.mu,
        "seed": cell.seed, "n_envs": cell.envs, "runtime": node.arm,
        "cell_id": cell.cell_id(), "wall_seconds": round(wall, 2), "result": res,
        # Everything that makes this row NOT a suite row, in the row.
        "graph": {"batched": False, "source": "f1sim_ros eval",
                  "declared_envs": node.declared_envs, "learners": node.envs,
                  "sync": node.sync, "control_dt_s": period,
                  "steps": node.publishes, "commands": node.cmd_count,
                  "answered": len(node.answers), "unsolicited_commands": node.unsolicited,
                  "command_timeouts": node.timeouts, "warmup_steps": node.warmup_steps,
                  "steps_per_second": (node.publishes / wall) if wall > 0 else 0.0,
                  "realtime_factor": (node.publishes * period / wall) if wall > 0 else 0.0,
                  "arm_installed_in": "controller_node",
                  # The env installs NO arm (`_entry` declares legacy), so the adapter's
                  # cross-runtime gate never fires here -- and the fact it exists to record is
                  # still true whenever the controller's arm is not the one the checkpoint was
                  # trained under. Recorded in the row rather than left to be inferred.
                  "checkpoint_arm": _checkpoint_arm(node.checkpoint),
                  "cross_runtime": _checkpoint_arm(node.checkpoint) != node.arm,
                  "pin_calibration": node.pin_calibration,
                  "calibration": dict(node.calibration),
                  "first_commands": node.commands_log},
    })
    return row


def main():
    rclpy.init()
    node = EvalNode()
    code = 0
    try:
        row = score_cell(node)
        t = row["result"]["tally"]
        g = row["graph"]
        node.get_logger().info(
            f"{row['cell_id']}: {t['successes']}/{t['denominator']} in {row['wall_seconds']:.1f} s "
            f"({g['steps']} steps, {g['steps_per_second']:.1f} step/s, "
            f"{g['realtime_factor']:.2f}x real time, {g['warmup_steps']} warm-up steps, "
            f"{g['command_timeouts']} command timeouts)")
        text = json.dumps(row)
        if node.out:
            os.makedirs(os.path.dirname(os.path.abspath(node.out)) or ".", exist_ok=True)
            with open(node.out, "a") as fh:
                fh.write(text + "\n")
            node.get_logger().info(f"appended to {node.out}")
        else:
            print(text)
    except (CommandTimeout, TimeoutError, SystemExit) as exc:
        node.get_logger().error(str(exc))
        code = 1
    except KeyboardInterrupt:
        code = 130
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
