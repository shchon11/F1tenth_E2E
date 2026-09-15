# ROS 2

Three ways in, for three different purposes:

| entry | purpose |
| --- | --- |
| [console → ROS2 연동](#the-console-on-ros-2) | the driving console's own session, published as topics; `/drive` can take the car over; rviz sees the whole scene |
| [`f1tenth_stack_sim.launch.py`](#the-f1tenth_system-stack-on-the-simulator) | run the real car's unmodified `f1tenth_system` stack against the simulator |
| [`sim.launch.py`](#standalone-bridge) | a standalone gym-style bridge: one environment, real time, plain topics |

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
# terminal 2: an external controller + rviz with the console layout
ros2 launch f1sim_ros pure_pursuit.launch.py
# or just watch:
ros2 launch f1sim_ros rviz.launch.py
```

### Topics

Car 0 is the ROS car (the environment's index 0, whichever car the console focuses). Its sensors
go out exactly as the [standalone bridge](#standalone-bridge) publishes them -- same names, frames,
stamps and QoS -- so a node written against one runs against the other and against the real car.

| topic | type | note |
| --- | --- | --- |
| `/scan`, `/odom`, `/ego_racecar/odom`, `/sensors/imu`, `/sensors/imu/raw`, `/map`, `/f1sim/collision`, TF | as the bridge | `/odom` drifts; `/ego_racecar/odom` is ground truth |
| `/drive` | `AckermannDriveStamped` (in) | steering [rad], speed [m/s]; silent for 0.5 s → speed 0 |
| `/f1sim/reset` | `std_srvs/Empty` (in) | resets every car, like the console's button |
| `/f1sim/reset` | `std_msgs/Empty` (out, topic) | announced after a reset, so a policy node clears its observation and memory. A topic of the same name as the service above; the two are separate in the ROS graph |
| `/f1sim/raceline`, `/f1sim/raceline_speed` | `Path`, `Float32MultiArray` (latched) | the map's racing line and its target speed per pose |
| `/f1sim/centerline` | `Path` (latched) | |
| `/f1sim/viz/cars` | `MarkerArray` (20 Hz) | every car as a box: green = ROS car, orange = its rivals, grey = the rest |
| `/f1sim/viz/props` | `MarkerArray` (latched) | placed obstacles as wireframe prisms (empty on a map without props) |
| `/f1sim/viz/raceline` | `Marker` (latched) | the line coloured by target speed, blue → red |
| `/f1sim/viz/plan` | `Marker` | the plan tracker's reference while the policy drives car 0 in plan mode |

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

Build instructions are in [Getting started](getting_started.md#ros-2-workspace). Source the workspace
first:

```bash
source install/setup.bash        # or setup.zsh
```

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
| `policy` / `policy_speed_cap` | `""` / `4.0` | drive `/drive` from an exported policy checkpoint |

### Topics

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

### Calibration

[`f1sim_ros/config/vesc.yaml`](../f1sim_ros/config/vesc.yaml) is the stack's own file with the
wheelbase set for this car (0.3302 m). `vesc_sim` reads the same file the stack does, so a mismatch
between the stack's calibration constants and the simulated car's "true" behaviour shows up in the
randomised actuator and odometry parameters — which is exactly where it lives on the real vehicle.

## Standalone bridge

One environment in real time with plain topics, closer to `f1tenth_gym_ros`:

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

Topics: `/drive` in; `/scan`, `/odom`, `/ego_racecar/odom`, `/sensors/imu`, `/sensors/imu/raw`,
`/map`, `/f1sim/collision` out. `/initialpose` teleports the car and `/f1sim/reset` is a reset
service. TF is `map`→`odom`→`base_link`→`laser`.

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

## Running a policy

```bash
ros2 launch f1sim_ros f1tenth_stack_sim.launch.py map:=gen:competition:2 \
  policy:=$HOME/f1sim_runs/ppo_v1/ppo_latest.pt policy_speed_cap:=5.0
```

`policy_node.py` subscribes to `/scan`, `/odom` and `/sensors/imu` and publishes `/drive`, building
its observation with the same [`learn/obs.py`](../f1sim/f1sim/learn/obs.py) used in training. Because
it consumes only topics the real car also publishes, the node runs unchanged against real hardware —
though nothing in this repository has been tested on a physical vehicle.

### Policy memory on the car

A checkpoint trained with `--memory gru` (see
[training.md](training.md#policy-memory---memory-gru)) carries a recurrent hidden state between
control steps, and one trained with `--scan-channels memory` carries a decayed per-bearing
occupancy map. The node keeps both — one car, one row — in `self.policy_state`, threads the hidden
state through every `/scan` callback, and **clears both in exactly two places**:

* `_resume()`, when the scan stream comes back after a gap. The observation history is cleared
  there for the same reason: it describes a segment that is over, and stitching the new one onto it
  feeds the policy a history that never happened.
* `on_reset()`, a `std_msgs/Empty` message on the `/f1sim/reset` **topic** (parameter
  `reset_topic`). This is deliberately *not* a second server for the `std_srvs/Empty` service of
  the same name: topic and service names are separate in the ROS graph, and offering a second
  server would make which node answers a reset ambiguous. The simulator nodes (`bridge_node`,
  `vesc_sim_node`) publish on that topic straight after they reset a car, so a policy driving the
  simulator hears its own reset. On the real car nothing publishes it and the node behaves exactly
  as it did before.

Known gap: the console's own ROS link (`f1sim/viewer/ros_link.py`) offers the `/f1sim/reset`
service but does not announce on the topic, so a policy node driving a *console* session is cleared
by a scan gap and not by the console's reset button.

A legacy (feedforward) checkpoint reaches none of this: `policy_state` is inert, holds nothing and
the published command is what it always was. The startup log names the memory when there is one.
Unit tests: `f1sim/tests/test_policy_node_memory.py`.

## Published baselines on the same link

The two published F1TENTH end-to-end baselines run here as ROS 2 nodes, on the topics this car
already publishes, so they can be driven by the console's ROS 2 mode, the standalone bridge,
`f1tenth_stack_sim.launch.py` or the real car without changing anything about them.

```bash
# TinyLidarNet (IROS 2024) -- LiDAR only
ros2 launch f1sim_ros baseline.launch.py model:=tinylidarnet \
  weights:=$HOME/.../models/tinylidarnet_L_1081.onnx

# End2Race (arXiv 2509.16894) -- LiDAR + measured speed, GRU
ros2 launch f1sim_ros baseline.launch.py model:=end2race \
  weights:=$HOME/.../End2Race/pretrained/end2race.pth
```

`baseline_node.py` subscribes to `/scan` (and `/odom` only when the model reads speed) and
publishes `/drive`. **It publishes `/drive` directly**: these networks emit a steering angle and a
speed, so there is no plan to track and this path has no `PlanTracker`, no controller arm, no
clearance layer and no traction guard. In the split graph (`policy_node` → `/f1sim/plan` →
`controller_node` → `/drive`) the baseline node therefore stands in for **both** halves at once, and
must not be run alongside a `controller_node` that is also publishing `/drive` unless a mux is
arbitrating. Everything else is the same contract as `policy_node`: the same topic names, the same
`/f1sim/reset` **topic** (`std_msgs/Empty`, parameter `reset_topic`), the same watchdog.

The preprocessing is **not re-derived from the papers**. Each model's beam selection, clipping or
pressure-token normalisation, speed scaling and output range is transcribed from its own repository
with the file and line quoted in `f1sim/f1sim/learn/baselines/`, and the node and the batched
benchmark adapter call the *same object*, so the parity test between them is about the message
plumbing rather than about two transcriptions.

| | TinyLidarNet | End2Race |
| --- | --- | --- |
| weights | `Models/f1_tenth_model.h5` → ONNX, 220 686 params | `pretrained/end2race.pth`, 11 301 482 params |
| scan it wants | 1081 beams over 270°, clipped at 10 m | 360 of a **1440-beam 360°** scan, metres |
| reads speed | no | yes (the previous step's `/odom`) |
| output | steer [rad] straight through; speed `linear_map(out, 0,1, 1,8)` | steer clipped to ±0.52 rad; speed unclipped |
| its own rate | 40 Hz | 100 Hz at their eval, 10 Hz in their training data |
| step cost here | **0.083 ms** (CPU, 1 thread, batch 1) | **1.853 ms** |

TinyLidarNet's scan is exactly this car's, so nothing is resampled. End2Race's is not: a Hokuyo
UST-10LX spans 270°, so **90 of its 360 features have no measurement behind them**. The node maps
the scan by *bearing* (never by index — an index map on a different window silently rotates the
world) and fills the unseen quarter with the model's own no-return value, logging that once, loudly,
at startup. The `scan_fill` parameter chooses the fill; both conventions their code uses are scored
separately (`docs/research/baselines-2026-09-15.md`).

**Staleness is per model.** `policy_node` inhibits when the IMU, the attitude or the odometry go
stale because its policy reads all three. These do not. `/scan` always counts — through a watchdog
timer, because the scan callback is exactly the thing that stops running when the LiDAR goes away —
and `/odom` counts when and only when `driver.needs_speed`. Stopping a LiDAR-only network because an
IMU it never reads went quiet is not safety.

The published command is clipped to what the plant can execute: `|steer| ≤ steer_max` and
`0 ≤ speed ≤ speed_cap`. The floor at zero is not a safety choice but a parity one — the batched
action space these nodes are checked against cannot express a reverse command either
(`gym_env.py:1054`), and a parity claim has to be about the same command.

Tests: `f1sim/tests/test_baseline_node.py` (reset, staleness, the window mapping, and node ↔ adapter
parity on 100 recorded scans), `f1sim/tests/test_baselines_preprocessing.py` (our preprocessing
against the vendored code, executed).

## Plan controller on the car

`policy_node` installs a grip-aware limit on the plan tracker by default (`controller:=fixed_low`):
corner speed and the acceleration/brake budgets are bounded for a constant conservative friction
(0.73423, the low end of the training range; override with `grip_mu`). It needs no estimator and reads
no sensor, and on suite v1 (2026-09-12) it was the safest arm on every stability column for the same
policy at a ~2 % lap-time cost. `controller:=legacy` is the untouched tracker. The `estimated` and
`reactive` arms are simulator research arms and are refused here. The startup log line names the arm
and the friction in force. Unit tests: `f1sim/tests/test_policy_node_grip.py`.

### `+clearance` — the plan kept off what `/scan` can see

`controller:=fixed_low+clearance` (or `clearance` on its own) adds the geometry layer described in
[training.md](training.md#clearance--the-plan-kept-off-what-the-lidar-can-see). Every scan callback
it turns **that scan and nothing else** into a coarse occupancy grid in the car's own frame, builds
a distance field on it, and bends or slows the plan until every point of it keeps
`clearance_margin` (default 0.20 m body edge, 0.34 m from a plan point to the nearest return) from
anything the scanner saw. There is no map on this path, no pose, and no state carried between
scans — which is the reason the simulator and the car can run the same module unmodified.

```bash
ros2 run f1sim_ros policy --ros-args -p checkpoint:=... -p controller:=fixed_low+clearance
ros2 run f1sim_ros policy --ros-args -p checkpoint:=... -p controller:=fixed_low+clearance \
  -p clearance_margin:=0.25
```

Three things worth knowing before it drives:

* It binds `PlanTracker._plan_hook` while `fixed_low` binds `._solver`, so the two are installed
  independently and the order cannot change what is published.
* The grid is built from the beam *bearings*, so the node checks the first `LaserScan`'s own
  `angle_min` / `angle_max` against the nominal 270° window and re-declares them if the driver
  publishes something else, logging that it did. A window taken on trust would put every return at
  a bearing it does not have, and nothing downstream would look wrong.
* `base_link → laser` is taken as (0.297, 0, 0.110), the value `/tf_static` carries in all 22
  recordings. The plan is in `base_link` and the returns are in the sensor's frame; 0.297 m is one
  and a half of the margin being defended, so a node on a car with a different mount needs this
  changed in `policy_node.LIDAR_MOUNT_X` before the arm means anything.

Cost on this desk's CPU, single thread, batch 1: **1.25 ms** of the 25 ms a 40 Hz scan allows
(`python3 -m f1sim.learn.budget --clearance`). It never raises a commanded speed. Unit tests:
`f1sim/tests/test_policy_node_clearance.py`.

## Traction guard

`policy_node` can watch the wheel for lock-up and spin and shape the speed command it publishes.
**Off by default** (`traction:=off`): it has been validated by replaying the real recordings and
never on a moving car, and it is the one thing in this node that can raise a commanded speed the
policy lowered. Turn it on deliberately, at walking pace, with a hand on the kill switch:

```bash
ros2 launch f1sim_ros f1tenth_stack_sim.launch.py policy:=$HOME/f1sim_runs/ppo_v1/ppo_latest.pt
ros2 run f1sim_ros policy --ros-args -p checkpoint:=... -p traction:=on
ros2 run f1sim_ros policy --ros-args -p checkpoint:=... -p traction:=on \
  -p traction_params:="lock_rate=22, release_max=1.0"
```

### What it detects

[`f1sim_ros/f1sim_ros/traction.py`](../f1sim_ros/f1sim_ros/traction.py) is a ROS-free class
(`math` only; no numpy, no torch, no rclpy) that compares two measurements the car already
publishes:

| signal | topic | note |
| --- | --- | --- |
| wheel speed | `/odom` `twist.twist.linear.x` | `vesc_to_odom`'s **ERPM** wheel speed, not ground speed — under lock or spin it is wrong about the vehicle, which is what makes it a slip sensor |
| body acceleration | `/sensors/imu/raw` `linear_acceleration.x` | published in **g** on this car; the node detects and scales it (`imu_accel_scale`) before the guard sees it |
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

State changes are logged at INFO with the numbers behind them:

```
[f1sim_policy]: traction lock: wheel +0.44 m/s at -167.9 m/s^2, body +6.62 m/s at +3.0 m/s^2
                (residual +170.9, slip -6.18), 1 locks / 0 spins so far
```

### Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `controller` | `fixed_low` | `legacy`, `fixed_low`, `clearance`, `fixed_low+clearance`. Anything else — the simulator's `oracle` / `estimated` / `+tcs` arms — is refused rather than silently downgraded |
| `clearance_margin` | `0.0` | body-edge margin for a `+clearance` arm, in metres; `0.0` means the module default (0.20 m) |
| `traction` | `off` | `off` installs nothing at all; `on` installs the replay-validated guard. Anything else is refused |
| `traction_params` | `""` | `NAME=VALUE` pairs (comma or space separated) overriding any field of `TractionParams` — every threshold above is reachable from the launch line |

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
python3 scripts/replay_traction.py --root /path/to/other/bags       # default: ~/F1tenth/real_data
```

The script re-derives the labels of the wheel-slip survey that established these events
(`work/real-car-tcs/evidence/wheelslip_bags.py`, outside this repository) off the same arrays it
feeds the guard, and scores hit / miss / false alarm per bag with the
timestamps and the commanded-speed change at every event. Exit status is 0 only when nothing in the
must-catch set (labelled runs with |a_wheel| ≥ 30 m/s²) is missed and nothing fires while the car is
stationary or cruising. Current result: 22 bags, 1088 s of motion, 20 of the 22 must-catch runs
caught, no firing while stationary or cruising, largest command change 2.60 m/s. The two misses and
the trade-off on the weaker events are in REPORT.md.

Unit tests: [`f1sim/tests/test_traction_guard.py`](../f1sim/tests/test_traction_guard.py) (detector,
shaper, reset, determinism, parameter sanity) and
[`f1sim/tests/test_policy_node_traction.py`](../f1sim/tests/test_policy_node_traction.py) (the
parameter, the wiring, and that `off` is inert).
