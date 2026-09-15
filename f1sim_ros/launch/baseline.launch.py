"""A published baseline driving `/drive` on the existing link.

    ros2 launch f1sim_ros baseline.launch.py model:=tinylidarnet weights:=/abs/path.onnx
    ros2 launch f1sim_ros baseline.launch.py model:=end2race weights:=/abs/path.pth scan_fill:=30.0

Sensors come from elsewhere -- the console's ROS 2 mode, `sim.launch.py`'s standalone bridge,
`f1tenth_stack_sim.launch.py`, or the real car's drivers -- exactly as for `policy_node`. This file
brings up the model and (optionally) rviz with the console layout, and nothing else, so it composes
with whichever sensor source is already running.

`baseline_node` publishes `/drive` **directly**: no plan tracker, no controller arm, no clearance
layer, no traction guard. With the split graph (`policy_node` -> `/f1sim/plan` -> `controller_node`
-> `/drive`) this node stands in for both halves at once. Do not run it alongside a
`controller_node` that is also publishing `/drive` unless a mux is arbitrating between them.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("f1sim_ros")
    args = [
        DeclareLaunchArgument("model", description="tinylidarnet | end2race"),
        DeclareLaunchArgument("weights", description="absolute path to the .onnx / .pth"),
        DeclareLaunchArgument("drive_topic", default_value="drive"),
        DeclareLaunchArgument("speed_cap", default_value="9.0"),
        DeclareLaunchArgument("steer_max", default_value="0.4189"),
        DeclareLaunchArgument("sensor_timeout", default_value="0.25"),
        DeclareLaunchArgument("enabled", default_value="true"),
        # TinyLidarNet: which of the two upstream output mappings (sim = 1..8 m/s, car = -0.5..7.0)
        DeclareLaunchArgument("speed_map", default_value="sim"),
        DeclareLaunchArgument("skip_n", default_value="0", description="0 = infer from the model"),
        # End2Race
        DeclareLaunchArgument("hidden_scale", default_value="4"),
        DeclareLaunchArgument("n_features", default_value="0", description="0 = their 360"),
        DeclareLaunchArgument("scan_fill", default_value="nan",
                              description="metres written into the bearings this 270 deg scanner "
                                          "cannot see; nan = the model's own no-return value"),
        DeclareLaunchArgument("tick_hz", default_value="0.0",
                              description="0 = one model tick per scan; 100 = their eval rate"),
        DeclareLaunchArgument("caller_rate", default_value="40.0"),
        DeclareLaunchArgument("repo", default_value=""),
        DeclareLaunchArgument("rviz", default_value="false"),
    ]
    node = Node(package="f1sim_ros", executable="baseline", name="f1sim_baseline", output="screen",
                parameters=[{k: LaunchConfiguration(k) for k in
                             ("model", "weights", "drive_topic", "speed_cap", "steer_max",
                              "sensor_timeout", "enabled", "speed_map", "skip_n", "hidden_scale",
                              "n_features", "scan_fill", "tick_hz", "caller_rate", "repo")}])
    rviz = Node(package="rviz2", executable="rviz2",
                arguments=["-d", os.path.join(share, "config", "console.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")))
    return LaunchDescription(args + [node, rviz])
