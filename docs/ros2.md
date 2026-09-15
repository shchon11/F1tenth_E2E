# ROS 2

**The ROS 2 topic graph is the system boundary.** Every consumer of sensor data — inference,
evaluation, the training-side observation builder, calibration, the probes — reads the same
messages, and the plan tracker with its runtime arms is its own node. A node written against this
graph runs against the simulator, against a console session and against the real car without a line
changed, because in all three cases it is reading `/scan`, `/odom` and `/sensors/imu/raw` and
writing `/drive`.

```
sensor source                     policy                    controller
─────────────                     ──────                    ──────────
 bridge_node      /scan          ┌──────────────┐          ┌────────────────────┐
 vesc_sim_node    /odom          │ policy_node  │          │ controller_node    │
 console link  ──►/sensors/imu ──►  ObsBuilder  ├─/f1sim/──►  PlanTracker       ├──►/drive
 real drivers     /sensors/imu/raw│  + actor    │   plan   │  + arm             │
 eval_node        /f1sim/reset   └──────┬───────┘          │  + traction guard  │
                                        │                  └─────────┬──────────┘
                                        └─►/f1sim/policy_state       └─►/f1sim/controller/diag
                                                                        /f1sim/viz/plan

 baseline_node (worker 18) ────────────────────────────────────────────►/drive   (bypasses both)
```

| node | in | out |
| --- | --- | --- |
| [`policy_node`](../f1sim_ros/f1sim_ros/policy_node.py) | `/scan`, `/odom`, `/sensors/imu`, `/sensors/imu/raw`, `/f1sim/reset` | `/f1sim/plan`, `/f1sim/policy_state` |
| [`controller_node`](../f1sim_ros/f1sim_ros/controller_node.py) | `/f1sim/plan`, `/scan`, `/odom`, `/sensors/imu/raw`, `/sensors/core`, `/f1sim/reset` | `/drive`, `/f1sim/controller/diag`, `/f1sim/viz/plan`, `/f1sim/viz/clearance` |
| [`system_check`](../f1sim_ros/f1sim_ros/system_check_node.py) | everything in `config/record.yaml` | `/f1sim/system_check` |
| [`eval`](../f1sim_ros/f1sim_ros/eval_node.py) | `/drive`, `/f1sim/controller/diag` | the sensor topics, and a benchmark row |

## Launching it

Three targets, differing **only** in where the sensor topics come from. Every parameter that can
change what the car does is in one file, [`config/graph.yaml`](../f1sim_ros/config/graph.yaml),
which all three read — so a sim run and a car run are the same run with a different sensor source,
and "which arm, at what friction, with what calibration" is one file to read and one file to diff.

```bash
# the standalone simulator bridge + the graph + rviz
ros2 launch f1sim_ros graph_sim.launch.py \
    checkpoint:=$HOME/f1sim_runs/ppo_race_0910/ppo_latest.pt controller:=fixed_low+clearance

# a running console session (고급 설정 → ROS2 연동 = "센서 발행 + /drive 로 외부 제어")
ros2 launch f1sim_ros graph_console.launch.py checkpoint:=...

# the real car: f1tenth_stack's own drivers, then the graph
ros2 launch f1sim_ros graph_car.launch.py checkpoint:=...
```

| argument | default | meaning |
| --- | --- | --- |
| `config_yaml` | `config/graph.yaml` | the whole parameter set |
| `checkpoint` | `""` | the exported actor; **also** where the controller reads the observation geometry its clearance grid is built on |
| `device` | `cpu` | the policy's device (`sim_device` is the simulator's, in `graph_sim`) |
| `speed_cap` | `4.0` | given to **both** nodes: the policy reads it as an observation channel, the tracker as a ceiling |
| `controller` | `fixed_low` | `legacy`, `fixed_low`, `clearance`, `fixed_low+clearance` |
| `traction` | `off` | the guard; see below |
| `viz` | `false` | `/f1sim/viz/plan` and `/f1sim/viz/clearance` markers |
| `policy` | `true` | `false` runs the controller alone, for an external planner publishing `/f1sim/plan` |

Build first — the graph needs a message package, so `colcon` is not optional:

```bash
cd ~/F1tenth && colcon build --packages-select f1sim_interfaces f1sim_ros && source install/setup.bash
```

## The messages

[`f1sim_interfaces/Plan`](../f1sim_interfaces/msg/Plan.msg) is the boundary:

| field | meaning |
| --- | --- |
| `header.stamp` | the stamp of the **`/scan` this plan was computed from** — not the publish time |
| `float32[8] plan` | the raw normalized action, six curvature knots and two speed targets, exactly as `f1sim.mpc.decode` expects it |
| `string checkpoint` | `<run>/<file>@<sha256[:12]>`, on every message |
| `uint32 seq` | one per plan; a gap is a dropped plan and a repeat is a held one |

The plan is **arm-agnostic**: nothing on the topic knows whether the controller will run `legacy`,
`fixed_low`, `+clearance` or the traction guard, so one recorded plan stream can be replayed
through any arm and the arms stay comparable.

The scan stamp is load-bearing. `controller_node` keeps the last few scans' sensor snapshots — the
measured speed, the IMU mean, the normalized scan frame — and tracks each plan against **the
snapshot its stamp names**. That is what makes the split bit-exact: a plan that arrives late cannot
be paired with the next scan's measured speed. A plan that arrives *before* its own scan (a few per
cent of the time in a live graph, because the two nodes receive `/scan` independently) is held for
one scan rather than mispaired; one whose scan never arrives is tracked against the newest and
counted in `plan_unmatched` on the diagnostics.

[`f1sim_interfaces/PolicyState`](../f1sim_interfaces/msg/PolicyState.msg) carries what the policy
knows about itself: whether the checkpoint has episode memory, how many times it has been cleared
and why, whether the node is inhibiting, and the age of each input. Diagnostic only — nothing in
the control path reads it. `/f1sim/controller/diag` is a `diagnostic_msgs/DiagnosticArray` with the
arm, the friction in force, the clearance margin, the traction state, the plan sequence and
checkpoint, and the tracker's own model of the car (command latency, wheelbase, calibration).

## Parity: what the split guarantees

`policy_node` used to be the whole deployment — observation, actor, tracker, arms, guard, `/drive`.
Splitting it is only a refactor if the car cannot tell, so that is measured rather than argued.

[`f1sim/tests/reference/monolithic_policy_node.py`](../f1sim/tests/reference/monolithic_policy_node.py)
is the old node, byte for byte as it was at `0b78111`, kept for no other purpose and pinned by
hash. [`test_graph_parity.py`](../f1sim/tests/test_graph_parity.py) replays a recorded scan / IMU /
odometry sequence through it and through the split graph, over a **simulator bag** and a **real car
bag**:

| | |
| --- | --- |
| arms × traction × bags × frames | 4 × 2 × 2 × 120 = **1920 commands** |
| worst difference, steering | **0.000e+00 rad** |
| worst difference, speed | **0.000e+00 m/s** |

The same plans through a bare `mpc.PlanTracker` with the **training-side** arms installed
(`learn/grip_runtime`, `learn/clearance` — what `benchmark/model_adapter.prepare_cell` installs on
`env.tracker`) match exactly too, which is the other half of the claim: the controller adds nothing
to the path the benchmark scores. The sequences come from real bags via
[`scripts/make_graph_fixture.py`](../f1sim/scripts/make_graph_fixture.py) and are committed, so the
claim is reproducible without the 22 GB of recordings.

What makes this structural rather than lucky is that the two nodes share
[`f1sim_ros/deploy.py`](../f1sim_ros/f1sim_ros/deploy.py): the arm installers, the geometry
constants, and `SensorIntake` — the `/odom` and `/sensors/imu` reader both nodes run. The policy
needs the IMU mean for the observation and the controller needs its gyro-z for the tracker and its
accelerometer for the guard; those are one implementation called twice.

The one place the live graph can differ from the replay is ordering: two processes receive `/scan`
independently, so the controller's snapshot for a scan occasionally arrives after the plan for it.
That is held for one scan (above), and what remains is counted, not hidden.

## When the car stops

Three ways, all of which publish zero speed rather than letting the last command stand:

* **stale sensors** — the monolithic node's rule, unchanged, at the same `sensor_timeout` (0.25 s),
  evaluated on `controller_node`'s own subscriptions. A dead LiDAR stops the car at the instant it
  used to. The policy inhibits on the same rule at the same threshold and stops publishing plans.
* **`plan_timeout`** (0.25 s) — the policy process died, or is inhibiting. The sensors are fine and
  the controller says so in the log; this is the only thing between a dead policy and a car still
  driving on the last plan it was sent.
* **no scan at all yet** — nothing to track a plan against.

While a plan is younger than `plan_timeout` the last command stands, which is what a mux does
anyway; past it the controller publishes zeros at `watchdog_period` (10 ms) until a plan returns.

## Direct-action checkpoints and baseline nodes bypass the controller

Two publishers put `/drive` on the graph without a plan, and both are deliberate:

* a **direct-action checkpoint** (`act_dim == 2`, the legacy `--action-mode direct`) has no plan to
  publish. `policy_node` detects it from the checkpoint's own observation spec, publishes `/drive`
  itself and logs that it is doing so. Do not run `controller_node` in that graph; there is nothing
  for it to track, and it would brake the car at `plan_timeout`.
* a **baseline node** — `f1sim_ros/baseline_node.py` (worker 18's published-baseline comparison) —
  is a network that emits steering and speed directly. It reads `/scan` (and `/odom` when its model
  does) and publishes `/drive`, bypassing the tracker and every arm. That is the point of it: the
  comparison is between whole systems, and giving a baseline our runtime would answer a different
  question.

`/f1sim/plan` is therefore `required: false` in the record profile, and `system_check` reports a
baseline run as healthy without it.

## Policy memory on the car

A checkpoint trained with `--memory gru` (see [training.md](training.md#policy-memory---memory-gru))
carries a recurrent hidden state between control steps, and one trained with
`--scan-channels memory` carries a decayed per-bearing occupancy map. `policy_node` keeps both —
one car, one row — in `self.policy_state`, threads the hidden state through every `/scan` callback,
and **clears both in exactly two places**:

* `_resume()`, when the scan stream comes back after a gap. The observation history is cleared
  there for the same reason: it describes a segment that is over, and stitching the new one onto it
  feeds the policy a history that never happened.
* `on_reset()`, a `std_msgs/Empty` message on the `/f1sim/reset` **topic** (parameter
  `reset_topic`). This is deliberately *not* a second server for the `std_srvs/Empty` service of
  the same name: topic and service names are separate in the ROS graph, and offering a second
  server would make which node answers a reset ambiguous. The simulator nodes (`bridge_node`,
  `vesc_sim_node`, `eval_node`) publish on that topic straight after they reset a car, so a policy
  driving the simulator hears its own reset. On the real car nothing publishes it and the node
  behaves exactly as it did before.

Every clear is counted and timestamped on `/f1sim/policy_state`, with the reason, so a bag says how
many times the memory was cleared and what cleared it.

Known gap: the console's own ROS link (`f1sim/viewer/ros_link.py`) offers the `/f1sim/reset`
service but does not announce on the topic, so a policy node driving a *console* session is cleared
by a scan gap and not by the console's reset button.

A legacy (feedforward) checkpoint reaches none of this: `policy_state` is inert, holds nothing and
the published plan is what it always was. The startup log names the memory when there is one. Unit
tests: `f1sim/tests/test_policy_node_memory.py`.

## The controller's arms

`controller_node` installs a grip-aware limit on the plan tracker by default
(`controller:=fixed_low`): corner speed and the acceleration/brake budgets are bounded for a
constant conservative friction (0.73423, the low end of the training range; override with
`grip_mu`). It needs no estimator and reads no sensor, and on suite v1 (2026-09-12) it was the
safest arm on every stability column for the same policy at a ~2 % lap-time cost.
`controller:=legacy` is the untouched tracker. The `estimated` and `reactive` arms are simulator
research arms and are refused here. The startup log line names the arm and the friction in force,
and so does every `/f1sim/controller/diag` message. Unit tests:
`f1sim/tests/test_controller_arms.py`.

### `+clearance` — the plan kept off what `/scan` can see

`controller:=fixed_low+clearance` (or `clearance` on its own) adds the geometry layer described in
[training.md](training.md#clearance--the-plan-kept-off-what-the-lidar-can-see). Every scan it turns
**that scan and nothing else** into a coarse occupancy grid in the car's own frame, builds a
distance field on it, and bends or slows the plan until every point of it keeps `clearance_margin`
(default 0.20 m body edge, 0.34 m from a plan point to the nearest return) from anything the
scanner saw. There is no map on this path, no pose, and no state carried between scans — which is
the reason the simulator and the car can run the same module unmodified.

```bash
ros2 launch f1sim_ros graph_sim.launch.py checkpoint:=... controller:=fixed_low+clearance
ros2 run f1sim_ros controller --ros-args -p checkpoint:=... \
    -p controller:=fixed_low+clearance -p clearance_margin:=0.25
```

Three things worth knowing before it drives:

* It binds `PlanTracker._plan_hook` while `fixed_low` binds `._solver`, so the two are installed
  independently and the order cannot change what is published.
* The grid is built from the beam *bearings*, so the node checks the first `LaserScan`'s own
  `angle_min` / `angle_max` against the nominal 270° window and re-declares them if the driver
  publishes something else, logging that it did. A window taken on trust would put every return at
  a bearing it does not have, and nothing downstream would look wrong.
* The grid is fed the scan **in the observation's own units, at the observation's beam count** —
  which is why the controller reads the observation spec from the `checkpoint` parameter. Passing
  `n_beams` / `range_max` / `v_max` instead is for a controller with no checkpoint on its disk;
  giving both and disagreeing is refused rather than resolved by precedence.
* `base_link → laser` is taken as (0.297, 0, 0.110), the value `/tf_static` carries in all 22
  recordings. The plan is in `base_link` and the returns are in the sensor's frame, and 0.297 m is
  one and a half of the margin being defended, so a node on a car with a different mount needs
  `deploy.LIDAR_MOUNT_X` changed before the arm means anything.

Cost on this desk's CPU, single thread, batch 1: **1.25 ms** of the 25 ms a 40 Hz scan allows
(`python3 -m f1sim.learn.budget --clearance`). It never raises a commanded speed. Unit tests:
`f1sim/tests/test_controller_clearance.py`.

## Traction guard

`controller_node` can watch the wheel for lock-up and spin and shape the speed command it
publishes. **Off by default** (`traction:=off`): it has been validated by replaying the real
recordings and never on a moving car, and it is the one thing in this graph that can raise a
commanded speed the policy lowered. Turn it on deliberately, at walking pace, with a hand on the
kill switch:

```bash
ros2 launch f1sim_ros graph_car.launch.py checkpoint:=... traction:=on
ros2 run f1sim_ros controller --ros-args -p checkpoint:=... -p traction:=on \
    -p traction_params:="lock_rate=22, release_max=1.0"
```

### What it detects

[`f1sim_ros/f1sim_ros/traction.py`](../f1sim_ros/f1sim_ros/traction.py) is a ROS-free class
(`math` only; no numpy, no torch, no rclpy) that compares two measurements the car already
publishes:

| signal | topic | note |
| --- | --- | --- |
| wheel speed | `/odom` `twist.twist.linear.x` | `vesc_to_odom`'s **ERPM** wheel speed, not ground speed — under lock or spin it is wrong about the vehicle, which is what makes it a slip sensor |
| body acceleration | `/sensors/imu/raw` `linear_acceleration.x` | published in **g** on this car; detected and scaled (`deploy.SensorIntake`, `calib.bagread.accel_scale_for`) before the guard sees it |
| motor current | `/sensors/core` `state.current_motor` | optional corroboration: a launch spin needs drive torque. Absent topic or no `vesc_msgs` ⇒ the check is skipped, not failed |

A rolling wheel cannot change speed faster than the body, and the body cannot exceed µ·g ≈ 10.3 m/s²
on this floor. So both states need an **absolute** gate and a **residual** against the (clamped)
IMU:

* **lock** — wheel deceleration past `lock_accel` (18 m/s², 1.75 µ·g) *and* a residual
  (body − wheel acceleration) past `lock_rate` (18 m/s²), while the recent peak body speed is above
  `v_lock_min` (1 m/s). Held until the wheel speed rejoins the body estimate, or `max_hold`.
* **spin** — wheel acceleration past `spin_accel` (14 m/s²) and a residual past `spin_rate`
  (12 m/s²) for `spin_persist` (2) consecutive samples, with motor current above
  `spin_current_min` when it is known.

The guard also carries a plausible body speed: the clamped IMU acceleration integrated, pulled back
onto the wheel speed (slew-limited to µ·g) whenever nothing is slipping, and decayed by at least
`lock_decay_frac`·µ·g while a lock is latched — a sliding tyre is at the friction limit, and the
accelerometer is the one signal that cannot be trusted during the event.

### The two actions

* **lock → release the brake.** The command is raised towards `release_frac`·(body speed) at
  `release_rate`, and while the lock is latched it may not be *cut* faster than `brake_rate`. The
  guard never commands less than it was asked for, and `release_max` (2 m/s) is a hard ceiling on
  how far above the policy's command it can go — because these two sensors cannot tell "the wheel is
  sliding at 7 m/s of body speed" from "the car has stopped and the estimate has not caught up".
* **spin → cap the command.** At `body speed + spin_margin`, never across zero, then the cap is held
  for `spin_ramp` seconds after the state clears while growing at µ·g, and then dropped.

The guard is fed at the **`/odom` rate** (50 Hz on this car, the rate the replay validated it at),
not at the scan rate, and shapes once per published command. State changes are logged at INFO with
the numbers behind them, and the current state is on `/f1sim/controller/diag`:

```
[f1sim_controller]: traction lock: wheel +0.44 m/s at -167.9 m/s^2, body +6.62 m/s at +3.0 m/s^2
                    (residual +170.9, slip -6.18), 1 locks / 0 spins so far
```

### Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `controller` | `fixed_low` | `legacy`, `fixed_low`, `clearance`, `fixed_low+clearance`. Anything else — the simulator's `oracle` / `estimated` / `+tcs` arms — is refused rather than silently downgraded |
| `clearance_margin` | `0.0` | body-edge margin for a `+clearance` arm, in metres; `0.0` means the module default (0.20 m) |
| `traction` | `off` | `off` installs nothing at all; `on` installs the replay-validated guard. Anything else is refused |
| `traction_params` | `""` | `NAME=VALUE` pairs (comma or space separated) overriding any field of `TractionParams` — every threshold is reachable from the launch line |

`TractionParams` carries the rest, each field documented where it is declared with the bag and
timestamp that set it: `lock_accel`, `lock_rate`, `spin_accel`, `spin_rate`, `clear_frac`,
`slip_hold`, `lock_persist`, `spin_persist`, `v_lock_min`, `v_ref_hold`, `spin_current_min`,
`min_hold`, `max_hold`, `wheel_fc`, `imu_fc`, `v_body_tau`, `a_body_max`, `min_diff_dt`,
`max_diff_dt`, `max_step_dt`, `lock_decay_frac`, `release_frac`, `release_rate`, `brake_rate`,
`release_max`, `spin_margin`, `spin_ramp`. Nonsense values are refused at construction rather than
at 8 m/s.

### Why it is validated offline, and how to replay

When this rule was written the simulator **could not produce either failure**:
`f1sim/f1sim/dynamics.py` had no wheel rotation state and `f1sim/f1sim/odom.py` reported ground
speed, so the simulated wheel speed *was* the body speed and this residual was identically zero.
The real car shows both failures clearly in all 22 recordings under `real_data/` — wheel
decelerations of −40 … −143 m/s² against a body decel of −3 … −16 — so the rule lives on the car and
is scored by replay:

```bash
source /home/shchon11/F1tenth/activate.sh            # rosbag2_py, for the reader
python3 scripts/replay_traction.py --markdown        # the table in REPORT.md
python3 scripts/replay_traction.py --bag 20260826-173704 --events
python3 scripts/replay_traction.py --set lock_rate=22 --quiet        # sweep a threshold
```

Current result: 22 bags, 1088 s of motion, 20 of the 22 must-catch runs caught, no firing while
stationary or cruising, largest command change 2.60 m/s.

Unit tests: [`f1sim/tests/test_traction_guard.py`](../f1sim/tests/test_traction_guard.py) (detector,
shaper, reset, determinism, parameter sanity) and
[`f1sim/tests/test_controller_traction.py`](../f1sim/tests/test_controller_traction.py) (the
parameter, the wiring, and that `off` is inert).

## On the car

Nothing in this repository has driven a physical vehicle. Before `graph_car.launch.py` is run with
a motor connected:

* **`steer_gain` is 1.0 and must stay there until the servo configuration is confirmed.** The two
  recorded configurations have OPPOSITE polarity — competition (0826–0827)
  `servo = 0.40 + 0.52·steering_angle`, pre-competition (0714–0727)
  `servo = 0.61 − 0.35·steering_angle`. Confirm the steering direction at walking pace first.
* `cmd_delay` (0.035 s) is this car's measured command latency and is what the tracker predicts
  over. It is on `/f1sim/controller/diag` so a bag says which one was in force.
* `traction:=off` unless you have read the section above.
* Run `ros2 run f1sim_ros system_check` first. It is the "is everything there?" tool.

## Evaluation over the graph

```bash
ros2 launch f1sim_ros graph_eval.launch.py \
    checkpoint:=$HOME/f1sim_runs/ppo_race_0910/ppo_latest.pt speed_cap:=9.0 \
    suite:=v2.1 cell:='S:gen:competition:0:0.73423:4401' out:=/tmp/cell.jsonl
```

`eval_node` owns the environment the benchmark builds for that cell — the same track, the same
frozen obstacle placement, the same opponents, the same pinned friction — publishes car 0's sensors
on the graph's own topics, and applies whatever comes back on `/drive`. The cell is built by
`benchmark/model_adapter.prepare_cell` and scored by `benchmark/runner.run_cell`: the same two
functions `python -m f1sim.learn.benchmark run` calls, so the numbers are comparable by
construction rather than because a second implementation agreed. The row it writes has the batched
runner's schema, plus a `graph` object saying what it actually was.

**It is one learner, not eight.** A graph has one `/drive`. `n_envs` is 1 and `graph.batched` is
false, which is what stops the row being mistaken for a suite row — `benchmark report` refuses it,
correctly, because the suite declares eight trials and one was run. **Suites are still scored
batched.** This is a boundary check and a demo, not a second leaderboard.

Three things the node checks, because each of them would otherwise produce a plausible wrong number:

* **the graph's speed cap against the suite's.** The policy reads the cap as an *observation
  channel*, so a graph capped at 4 m/s scoring a suite that declares 9 is a different policy
  answering a different question. Read off `/f1sim/controller/diag` and refused on mismatch, along
  with the arm.
* **the loop against the graph's patience.** Both nodes measure freshness against the wall clock,
  because on a car that is the only clock that means anything. A CPU simulator steps at a fraction
  of real time, so a scan legitimately arrives hundreds of milliseconds after the last one and the
  graph would (correctly) call it stale, clear its histories and score a policy that never sees
  motion. `graph_eval.launch.py` therefore raises `sensor_timeout` and `plan_timeout` to 10 s, and
  the eval node refuses the run if the measured step period gets close to what the controller
  reports. It is a property of the harness, not of the graph.
* **which `/drive` answers which scan.** The controller stamps a tracked command with its scan's
  stamp and a watchdog brake with the current time, so the loop waits for the command for *the step
  it published* and never mistakes a brake for an answer.

`sync` (the default) steps the simulator only once that command has come back. `sync:=false`
free-runs at the control rate and takes whatever has arrived, which is what a car does with a late
command.

### Throughput

Measured on this desk, CPU only (`device:=cpu`), one learner, suite v2.1 speed cap 9 m/s, arm
`fixed_low`:

| | |
| --- | --- |
| simulator step, CPU eager, 1 env | ~30 ms |
| policy inference (1081 beams, 6 frames), CPU | 3–8 ms typical, up to 240 ms under contention |
| tracker + arm, CPU | 25–55 ms |
| **graph eval, sync** | **~5 steps/s, 0.13× real time** |
| a 3-lap S cell (551 steps) | ~100 s wall |
| the same cell batched, 1 learner | ~35 s wall |
| the same cell batched, 8 learners, GPU | seconds |

Suite v2.1 is 144 cells × 8 trials. Over the graph at one learner that is of the order of a day per
system, against minutes batched on a GPU, which is why the suite is scored batched and this
command exists to show that the boundary is real.

### Graph against batched, on cells

Deltas on three cells, and what they mean, are in
[`docs/research/ros-graph-2026-09-15.md`](research/ros-graph-2026-09-15.md). The short version: the
**outcome** agrees, and the continuous metrics do not, for a reason that is neither a bug nor
avoidable. A node joining a sensor stream has no history — it fills its six-frame scan stack and
its 20-row proprio history from the first scans it sees — while the batched env is initialised with
the spawn. `test_obs_identity.py` shows the two observations become bit-identical once both have
seen the same `(scan_stack−1)·scan_stride` frames, and the actions with them; before that they
differ, and a closed loop at the friction limit turns that into a different lap. Graph-vs-batched
parity on a cell is therefore an **outcome-level** claim, and the note reports the run-to-run
variation of the graph against itself alongside it, so the two can be told apart.

### rviz

`graph_sim.launch.py viz:=true rviz:=true` opens `config/graph.rviz`: the plan the tracker is
following (`/f1sim/viz/plan`, in `base_link`, *after* the clearance bend and the grip speed limit —
the thing the car is actually driving), the clearance grid built from the current scan
(`/f1sim/viz/clearance`), both odometries, the scan and the map. The markers cost the controller
one `Marker` and one `CUBE_LIST` per command and are off by default (`viz:=false`).

## Recording, datasets and `system_check`

[`config/record.yaml`](../f1sim_ros/config/record.yaml) is the record profile: exactly the topics
the graph needs, each with the rate it should arrive at, whether it is required, whether it is
simulator-only, and why it is there. One list, three consumers:

```bash
# on the car
ros2 bag record $(python3 -m f1sim_ros.recorder --args --car)
# in the simulator: the bridge writes the same profile itself, so the bags are interchangeable
ros2 launch f1sim_ros graph_sim.launch.py checkpoint:=... record:=~/bags
# either one, afterwards
python3 -m f1sim_ros.system_check ~/bags/f1sim_sim_20260915-123456
```

`system_check` reports, for every topic in the profile, whether it is present, at what rate, how
stale it is — and what was in force: the checkpoint (from `/f1sim/policy_state`, or from any
`/f1sim/plan`, which carries it on every message), the memory kind and how many times it was
cleared, the arm, the friction, the clearance margin and the traction state. A **required** topic
missing is a failure and a non-zero exit; an optional one missing is information, which is why a
baseline run with no `/f1sim/plan` reports healthy. Live:

```bash
ros2 run f1sim_ros system_check                      # every second, and on /f1sim/system_check
ros2 run f1sim_ros system_check --ros-args -p once:=true -p car:=true   # exit code says ready
```

[`learn/bagdata.py`](../f1sim/f1sim/learn/bagdata.py) turns a bag into a dataset:

```bash
python3 -m f1sim.learn.bagdata ~/bags/run --spec-from ~/f1sim_runs/.../ppo_latest.pt \
    --out ~/datasets/run.npz
```

For every `/scan` it rebuilds the observation `ObsBuilder` would have built — `resample_ranges`,
the mean of the IMU samples since the previous scan, the last `/odom` wheel speed, roll and pitch
from the `Imu` quaternion — and pairs it with the `/drive` that answered it, the pose, and the
episode the frame belongs to. That is what DAgger relabelling (replay the poses in the simulator,
label with `env.teacher_label`), the hidden-state probes, the calibration tools and a baseline's
scan-for-scan parity check all need.

Two things it cannot invent, and says so rather than pretending:

* **the previous-action channels.** They are in the bag only when `/f1sim/plan` was recorded, or
  when a direct-action run's `/drive` inverts back into an action. Otherwise they are zeros and
  `meta["action_source"]` is `"zeros"` — the rest of the observation is still exact, but it is not
  the observation the policy saw.
* **episode boundaries.** A `/f1sim/reset` clears the builder and increments `episode`, because a
  history stitched across a reset describes a run that never happened.

`calib/bagread.py` underneath it now detects whether `linear_acceleration` is in g or m/s² from the
gravity vector instead of assuming g — the car publishes g and the simulator publishes SI on the
same topic, and reading a simulator bag as if it were g is a silent factor of 9.81. The factor
chosen is on `BagData.accel_scale`, and it is the same rule `deploy.SensorIntake` applies live.

## Training on the topic contract

**Batched PPO cannot run through ROS topics.** Training steps 258 environments at 40 Hz; the graph
carries one car and, on this desk's CPU, at about a fifth of real time. There is no version of that
which is a training loop. So the contract is held **structurally** instead:

* **One observation encoding.** Every arithmetic step between a sensor value and an observation
  element is a function in [`learn/obs.py`](../f1sim/f1sim/learn/obs.py) — `resample_ranges`,
  `norm_scan`, `norm_imu`, `norm_att`, `norm_speed`, `ATT_SCALE` — and both `ObsBuilder` (the node)
  and `gym_env` (the batched env) call them rather than writing the arithmetic out. The attitude
  scale used to be `ObsSpec`'s default on one side and a literal `0.35` on the other, with nothing
  making them equal.
* **One executable claim.** `F1VecEnv.message_inputs(i)` states a simulator step as
  `obs.MessageInputs` — `LaserScan.ranges`, the `Imu` samples, `Odometry`'s wheel speed, the
  orientation's roll and pitch — which is the only vocabulary the env and the node have in common,
  and contains nothing privileged. `ObsBuilder.build_message` turns one into an observation exactly
  as `policy_node` does. `test_obs_identity.py` drives both and requires the scan, the proprio, the
  **action**, the recurrent hidden state and the decayed occupancy channel to be bit-identical, for
  a legacy, a recurrent and a scan-channel checkpoint.
* **The one deliberate difference is pinned as a test.** `resample_ranges` saturates a non-positive
  range and the 65.533 m miss sentinel the real `urg_node` emits; `gym_env._norm_scan` does neither,
  because the simulator emits neither. On anything a simulator produces the two are identical.

What the rolling history buffers still do not share is their storage: the env's are fused into a
compiled step (`_step_math`, one CUDA graph) and the node's are its own. The identity test covers
the composition, and `test_hist_parity.py` covers the proprio history rows step by step.

## The console on ROS 2

The console (`python -m f1sim.viewer.console`) renders one environment in its own window. With
**고급 설정 → ROS2 연동** set, that same environment is on the ROS graph:

| setting | what happens |
| --- | --- |
| `끄기` | nothing (default) |
| `센서 토픽 발행 (정책이 주행)` | the policy drives; car 0's sensors and the whole scene are published for anyone listening -- a localiser, a planner under test, rviz |
| `센서 발행 + /drive 로 외부 제어` | the same, and car 0 is driven by whatever publishes `/drive`; the policy keeps driving every other car |

Open the console from a shell that sourced the ROS workspace (`source activate.sh`), otherwise the
worker refuses the mode with a message saying so. The header shows `ROS2 발행` / `ROS2 /drive 제어`
while the link is up. The link lives in the simulation worker
([`viewer/ros_link.py`](../f1sim/f1sim/viewer/ros_link.py)); it publishes once per control step
from the simulation thread, and its own executor thread takes the callbacks.

```bash
# terminal 1: console, ROS2 연동 = "센서 발행 + /drive 로 외부 제어", start a session
python -m f1sim.viewer.console
# terminal 2: the graph, or just an external controller + rviz
ros2 launch f1sim_ros graph_console.launch.py checkpoint:=...
ros2 launch f1sim_ros pure_pursuit.launch.py
ros2 launch f1sim_ros rviz.launch.py
```

Car 0 is the ROS car (the environment's index 0, whichever car the console focuses). Its sensors go
out exactly as the [standalone bridge](#standalone-bridge) publishes them -- same names, frames,
stamps and QoS.

| topic | type | note |
| --- | --- | --- |
| `/scan`, `/odom`, `/ego_racecar/odom`, `/sensors/imu`, `/sensors/imu/raw`, `/map`, `/f1sim/collision`, TF | as the bridge | `/odom` drifts; `/ego_racecar/odom` is ground truth |
| `/drive` | `AckermannDriveStamped` (in) | steering [rad], speed [m/s]; silent for 0.5 s → speed 0 |
| `/f1sim/reset` | `std_srvs/Empty` (in) | resets every car, like the console's button |
| `/f1sim/reset` | `std_msgs/Empty` (out, topic) | announced after a reset by the simulator nodes, so a policy node clears its observation and memory. A topic of the same name as the service above; the two are separate in the ROS graph. **The console link does not announce it** — see "Policy memory on the car" |
| `/f1sim/raceline`, `/f1sim/raceline_speed` | `Path`, `Float32MultiArray` (latched) | the map's racing line and its target speed per pose |
| `/f1sim/centerline` | `Path` (latched) | |
| `/f1sim/viz/cars` | `MarkerArray` (20 Hz) | every car as a box: green = ROS car, orange = its rivals, grey = the rest |
| `/f1sim/viz/props` | `MarkerArray` (latched) | placed obstacles as wireframe prisms |
| `/f1sim/viz/raceline` | `Marker` (latched) | the line coloured by target speed, blue → red |
| `/f1sim/viz/plan` | `Marker` | the plan tracker's reference — from the console while the policy drives car 0, or from `controller_node` when the graph does |

`/initialpose` is not taken here (the console owns spawn placement); use the bridge for that.

### The pure-pursuit example

[`pure_pursuit_node.py`](../f1sim_ros/f1sim_ros/pure_pursuit_node.py) is the smallest external
controller: it reads `/f1sim/raceline` (+ `_speed`) and a pose, and publishes `/drive`. It knows
nothing about the simulator, which is the point. Its pose source defaults to the ground truth
`/ego_racecar/odom`; `odom_topic:=/odom` runs it on the drifting dead-reckoning instead, which walks
it off the line within a lap -- the difference a localiser has to close.

```bash
ros2 launch f1sim_ros pure_pursuit.launch.py max_speed:=4.5 lookahead_gain:=0.35
ros2 run f1sim_ros pure_pursuit --ros-args -p odom_topic:=/odom -p max_speed:=3.0
```

Measured 2026-09-13 (console worker, `gen:competition:2`, `max_speed` 4.5): three laps in 50 s, no
collisions, `/scan` at 40 Hz, simulation at 0.98x real time with the link on. Tests:
`f1sim/tests/test_ros_link_builders.py` (message assembly, no ROS), `test_ros_link_live.py` (a
real rclpy round trip, skipped without rclpy), `test_gym_env_external_cmd.py` (the `/drive`
override in both action modes), `test_pure_pursuit.py` (the geometry), `test_sim_worker_ros2.py`
(the worker process with the link on).

## The f1tenth_system stack on the simulator

The simulator replaces **only the hardware drivers**. `f1sim_ros/vesc_sim_node.py` speaks the
interfaces of `vesc_driver` and `urg_node`, so the rest of the stack — `ackermann_mux`,
`ackermann_to_vesc`, `vesc_to_odom`, `f1tenth_stack` — runs unchanged, with its own configuration
files.

```bash
ros2 launch f1sim_ros f1tenth_stack_sim.launch.py map:=gen:competition:3
ros2 launch f1sim_ros f1tenth_stack_sim.launch.py map:=rt:Spielberg viewer:=false publish_gt_tf:=false
ros2 topic pub -r 40 /drive ackermann_msgs/msg/AckermannDriveStamped \
  "{drive: {steering_angle: 0.1, speed: 2.0}}"
```

| argument | default | meaning |
| --- | --- | --- |
| `map` | `gen:competition:0` | catalogue name or a map YAML |
| `device` | `cuda` | simulator device |
| `randomize` | `true` | per-episode parameter randomisation |
| `viewer` | `true` | open the 3D viewer window |
| `publish_gt_tf` | `true` | ground-truth `map`→`odom`; set `false` when running your own localiser |
| `teleop` | `true` | keyboard driving in the viewer window, published to `/teleop` |
| `joy` / `joy_preset` | `false` / `xbox` | gamepad teleoperation |
| `policy` / `policy_speed_cap` | `""` / `4.0` | drive `/drive` from an exported policy checkpoint (the monolithic path; `graph_sim.launch.py` is the graph) |

| topic | type | published by |
| --- | --- | --- |
| `/drive`, `/teleop` → `/ackermann_cmd` | `AckermannDriveStamped` | `ackermann_mux` (stack) |
| `/commands/motor/speed`, `/commands/servo/position` | `Float64` | `ackermann_to_vesc` (stack) → **vesc_sim** |
| `/sensors/core`, `/sensors/servo_position_command` | `VescStateStamped`, `Float64` | **vesc_sim** → `vesc_to_odom` (stack) |
| `/odom` + tf `odom`→`base_link` | `Odometry` | `vesc_to_odom` (stack), from simulated telemetry |
| `/sensors/imu`, `/sensors/imu/raw` | `VescImuStamped`, `Imu` | **vesc_sim** |
| `/scan` | `LaserScan` | **vesc_sim** |
| `/ego_racecar/odom`, `/map`, `/f1sim/collision`, tf `map`→`odom` | — | simulator ground truth only |

The distinction that matters: `/odom` is the drifting dead-reckoned estimate the car actually has,
while `/ego_racecar/odom` is ground truth that exists only in simulation. A policy or planner that
consumes the latter will not transfer.

[`f1sim_ros/config/vesc.yaml`](../f1sim_ros/config/vesc.yaml) is the stack's own file with the
wheelbase set for this car (0.3302 m). `vesc_sim` reads the same file the stack does, so a mismatch
between the stack's calibration constants and the simulated car's "true" behaviour shows up in the
randomised actuator and odometry parameters — which is exactly where it lives on the real vehicle.

## Standalone bridge

One environment in real time with plain topics, closer to `f1tenth_gym_ros`. `graph_sim.launch.py`
is this plus the graph; `sim.launch.py` is it on its own:

```bash
ros2 launch f1sim_ros sim.launch.py                                   # random track, rviz
ros2 launch f1sim_ros sim.launch.py map_yaml:=/abs/path/map.yaml device:=cuda
```

| argument | default |
| --- | --- |
| `map_yaml` | `""` — empty means a random procedural track |
| `random_track_seed` | `0` |
| `device` | `cuda` |
| `randomize` | `true` |
| `rviz` | `true` |
| `record` | `""` — a directory turns on recording of `config/record.yaml`'s topics |

Topics: `/drive` in; `/scan`, `/odom`, `/ego_racecar/odom`, `/sensors/imu`, `/sensors/imu/raw`,
`/map`, `/f1sim/collision` out. `/initialpose` teleports the car and `/f1sim/reset` is a reset
service (and an announcement on the topic of the same name). TF is `map`→`odom`→`base_link`→`laser`.
The sensor messages are built by [`sim_messages.py`](../f1sim_ros/f1sim_ros/sim_messages.py), which
`eval_node` shares — two simulator-side publishers that disagreed about a field would make "a node
cannot tell which simulator it is attached to" false.

Sample maps with centerlines are in [`f1sim_ros/maps/`](../f1sim_ros/maps/); regenerate them with
`f1sim/scripts/make_maps.py`. To add a real track, place the SLAM map YAML and PGM there, optionally
with a `<name>_centerline.csv` in `f1tenth_racetracks` format.

## Manual driving

[`teleop.py`](../f1sim/f1sim/teleop.py) converts pedals and sticks into the car's
(steering, target speed) command with racing-game dynamics: throttle is acceleration with a soft
pedal curve, releasing coasts down, braking at a standstill engages reverse, the target speed is not
allowed to run far ahead of the measured speed, and steering lock shrinks with speed under a
slew-rate limit. Keyboard input is ramped into analogue values so it feels like a stick.

- In the simulator window: launch the stack and drive with `W`/`S`, `A`/`D` or the arrow keys, `Shift`
  to boost, `Space` for handbrake, `R` to reset. The window publishes `/teleop`, which reaches the
  simulated VESC through the same mux and conversion nodes as a real joystick.
- Gamepad: `joy:=true joy_preset:=xbox`, or `joy_preset:=f1tenth` for the stack's own mapping.
  Standalone: `ros2 run f1sim_ros teleop --ros-args -p preset:=xbox`.
- Without ROS: `python3 f1sim/scripts/demo_viewer.py --manual`.
