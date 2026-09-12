"""The pure-pursuit example against a simulator that is already publishing (console with the ROS 2
link in "/drive 제어" mode, or the standalone bridge), plus rviz.

    ros2 launch f1sim_ros pure_pursuit.launch.py
    ros2 launch f1sim_ros pure_pursuit.launch.py max_speed:=5.0 odom_topic:=/odom rviz:=false
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
        DeclareLaunchArgument("odom_topic", default_value="/ego_racecar/odom",
                              description="pose source; /odom is the drifting dead-reckoning"),
        DeclareLaunchArgument("max_speed", default_value="4.0"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument("lookahead_gain", default_value="0.35"),
        DeclareLaunchArgument("rviz", default_value="true"),
    ]
    pp = Node(package="f1sim_ros", executable="pure_pursuit", name="pure_pursuit", output="screen",
              parameters=[{"odom_topic": LaunchConfiguration("odom_topic"),
                           "max_speed": LaunchConfiguration("max_speed"),
                           "speed_scale": LaunchConfiguration("speed_scale"),
                           "lookahead_gain": LaunchConfiguration("lookahead_gain")}])
    rviz = Node(package="rviz2", executable="rviz2", name="rviz2",
                arguments=["-d", os.path.join(share, "config", "console.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")))
    return LaunchDescription(args + [pp, rviz])
