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
        DeclareLaunchArgument("map_yaml", default_value="", description="ROS map yaml; empty -> random track"),
        DeclareLaunchArgument("random_track_seed", default_value="0"),
        DeclareLaunchArgument("config_yaml", default_value=os.path.join(share, "config", "default.yaml")),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("randomize", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
    ]
    bridge = Node(package="f1sim_ros", executable="bridge", name="f1sim_bridge", output="screen",
                  parameters=[{"map_yaml": LaunchConfiguration("map_yaml"),
                               "random_track_seed": LaunchConfiguration("random_track_seed"),
                               "config_yaml": LaunchConfiguration("config_yaml"),
                               "device": LaunchConfiguration("device"),
                               "randomize": LaunchConfiguration("randomize")}])
    rviz = Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "config", "sim.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")))
    return LaunchDescription(args + [bridge, rviz])
