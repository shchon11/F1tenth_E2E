"""The graph on the real car: the `f1tenth_system` drivers, then the policy and the controller.

    ros2 launch f1sim_ros graph_car.launch.py checkpoint:=/home/.../ppo_latest.pt

Nothing in this repository has driven a physical vehicle. Read `docs/ros2.md` -- "On the car" --
before this is launched with a motor connected; in particular `steer_gain` is left at 1.0 because
the two recorded servo configurations have OPPOSITE polarity, and the traction guard has been
validated only by replaying the recordings.

The drivers come from `f1tenth_stack` (`bringup_launch.py`: `urg_node`, `vesc_driver`,
`ackermann_to_vesc`, `vesc_to_odom`, `ackermann_mux`, the joystick). `mux:=false` publishes
straight to `/drive` for a stack that is already running elsewhere.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from f1sim_ros.launch_graph import common_arguments, graph_nodes


def _stack_launch():
    """`f1tenth_stack`'s own bringup, if it is installed. Absent on a desk machine, where this file
    still has to be importable -- the launch-file tests load every launch file in the package."""
    try:
        return os.path.join(get_package_share_directory("f1tenth_stack"), "launch",
                            "bringup_launch.py")
    except Exception:
        return ""


def generate_launch_description():
    args = common_arguments() + [
        DeclareLaunchArgument("drivers", default_value="true",
                              description="bring up f1tenth_stack's own drivers"),
    ]
    stack = _stack_launch()
    out = list(args)
    if stack:
        out.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(stack),
                                            condition=IfCondition(LaunchConfiguration("drivers"))))
    return LaunchDescription(out + graph_nodes())
