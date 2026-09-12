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

## Plan controller on the car

`policy_node` installs a grip-aware limit on the plan tracker by default (`controller:=fixed_low`):
corner speed and the acceleration/brake budgets are bounded for a constant conservative friction
(0.73423, the low end of the training range; override with `grip_mu`). It needs no estimator and reads
no sensor, and on suite v1 (2026-09-12) it was the safest arm on every stability column for the same
policy at a ~2 % lap-time cost. `controller:=legacy` is the untouched tracker. The `estimated` and
`reactive` arms are simulator research arms and are refused here. The startup log line names the arm
and the friction in force. Unit tests: `f1sim/tests/test_policy_node_grip.py`.
