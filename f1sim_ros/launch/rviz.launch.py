"""rviz with the console's visualisation layout. Pair it with a console session whose ROS 2 link is
on ("센서 발행" or "/drive 제어"), or with the standalone bridge.

    ros2 launch f1sim_ros rviz.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("f1sim_ros")
    rviz = Node(package="rviz2", executable="rviz2", name="rviz2",
                arguments=["-d", os.path.join(share, "config", "console.rviz")])
    return LaunchDescription([rviz])
