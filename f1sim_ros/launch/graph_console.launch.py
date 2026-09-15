"""The graph against a running console session (고급 설정 -> ROS2 연동).

    python -m f1sim.viewer.console          # set ROS2 연동 = "센서 발행 + /drive 로 외부 제어"
    ros2 launch f1sim_ros graph_console.launch.py checkpoint:=...

The console owns the simulation and publishes car 0's sensors itself, so this launch brings up
nothing but the two nodes. It is the same policy and the same controller as `graph_sim` and
`graph_car`, reading the same `config/graph.yaml`.

One known gap, unchanged by the split: the console's ROS link offers the `/f1sim/reset` SERVICE but
does not announce on the `/f1sim/reset` TOPIC, so the policy's memory and the controller's tracker
history are cleared by a scan gap and not by the console's reset button
(`f1sim/viewer/ros_link.py`).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from f1sim_ros.launch_graph import common_arguments, graph_nodes, share


def generate_launch_description():
    args = common_arguments() + [
        DeclareLaunchArgument("rviz", default_value="false"),
    ]
    rviz = Node(package="rviz2", executable="rviz2",
                arguments=["-d", os.path.join(share(), "config", "console.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")))
    return LaunchDescription(args + graph_nodes() + [rviz])
