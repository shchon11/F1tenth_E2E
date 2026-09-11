# ROS 2

Two launch files, for two different purposes:

| launch | purpose |
| --- | --- |
| [`f1tenth_stack_sim.launch.py`](#the-f1tenth_system-stack-on-the-simulator) | run the real car's unmodified `f1tenth_system` stack against the simulator |
| [`sim.launch.py`](#standalone-bridge) | a standalone gym-style bridge: one environment, real time, plain topics |

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
