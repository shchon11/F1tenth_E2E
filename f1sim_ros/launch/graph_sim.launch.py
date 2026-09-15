"""The whole graph against the standalone simulator bridge: one env, real time, plain topics.

    ros2 launch f1sim_ros graph_sim.launch.py \
        checkpoint:=$HOME/f1sim_runs/ppo_race_0910/ppo_latest.pt controller:=fixed_low+clearance

Sensor source: `bridge_node`. Everything else -- which arm, which friction, which calibration --
is `config/graph.yaml`, the same file the car launch reads.
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
        DeclareLaunchArgument("map_yaml", default_value="", description="ROS map yaml; empty -> random track"),
        DeclareLaunchArgument("random_track_seed", default_value="0"),
        DeclareLaunchArgument("sim_config_yaml", default_value=os.path.join(share(), "config", "default.yaml")),
        DeclareLaunchArgument("sim_device", default_value="cpu",
                              description="the SIMULATOR's device; `device` is the policy's"),
        DeclareLaunchArgument("randomize", default_value="true"),
        DeclareLaunchArgument("record", default_value="",
                              description="directory for a rosbag2 of config/record.yaml's topics"),
        DeclareLaunchArgument("rviz", default_value="true"),
    ]
    bridge = Node(package="f1sim_ros", executable="bridge", name="f1sim_bridge", output="screen",
                  parameters=[{"map_yaml": LaunchConfiguration("map_yaml"),
                               "random_track_seed": LaunchConfiguration("random_track_seed"),
                               "config_yaml": LaunchConfiguration("sim_config_yaml"),
                               "device": LaunchConfiguration("sim_device"),
                               "record": LaunchConfiguration("record"),
                               "record_profile": os.path.join(share(), "config", "record.yaml"),
                               "randomize": LaunchConfiguration("randomize")}])
    rviz = Node(package="rviz2", executable="rviz2",
                arguments=["-d", os.path.join(share(), "config", "graph.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")))
    return LaunchDescription(args + [bridge] + graph_nodes() + [rviz])
