"""f1tenth_system bringup with the simulator standing in for the hardware (vesc_driver + urg_node).
Nodes and configs are the ones from f1tenth_stack (humble-devel), unchanged:
  ackermann_mux (teleop/drive -> ackermann_drive), ackermann_to_vesc (ackermann_cmd -> commands/*),
  vesc_to_odom (sensors/core -> odom + odom->base_link tf), static tf base_link->laser (0.27 0 0.11).
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_share = get_package_share_directory("f1sim_ros")
    try:
        stack_share = get_package_share_directory("f1tenth_stack")
        mux_default = os.path.join(stack_share, "config", "mux.yaml")
    except Exception:
        mux_default = os.path.join(sim_share, "config", "mux.yaml")
    args = [
        DeclareLaunchArgument("map", default_value="gen:competition:0", description="catalog name (gen:competition:N, rt:Spielberg, gym:levine) or map yaml"),
        DeclareLaunchArgument("vesc_config", default_value=os.path.join(sim_share, "config", "vesc.yaml"),
                              description="f1tenth_stack vesc.yaml (ours = the stack's with wheelbase 0.3302)"),
        DeclareLaunchArgument("mux_config", default_value=mux_default),
        DeclareLaunchArgument("config_yaml", default_value=os.path.join(sim_share, "config", "default.yaml")),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("randomize", default_value="true"),
        DeclareLaunchArgument("viewer", default_value="true"),
        DeclareLaunchArgument("publish_gt_tf", default_value="true", description="false when running your own localization"),
        DeclareLaunchArgument("teleop", default_value="true", description="keyboard driving in the viewer window -> /teleop"),
        DeclareLaunchArgument("joy", default_value="false", description="start joy_node + gamepad teleop (preset xbox|f1tenth)"),
        DeclareLaunchArgument("joy_preset", default_value="xbox"),
        DeclareLaunchArgument("teleop_a_throttle", default_value="2.0", description="manual throttle accel [m/s^2]"),
        DeclareLaunchArgument("teleop_v_max", default_value="5.0"),
        DeclareLaunchArgument("policy", default_value="", description="checkpoint (.pt) of the e2e policy to drive /drive"),
        DeclareLaunchArgument("policy_speed_cap", default_value="4.0"),
    ]
    vesc_sim = Node(package="f1sim_ros", executable="vesc_sim", name="vesc_sim", output="screen",
                    parameters=[LaunchConfiguration("vesc_config"),
                                {"map": LaunchConfiguration("map"), "config_yaml": LaunchConfiguration("config_yaml"),
                                 "device": LaunchConfiguration("device"), "randomize": LaunchConfiguration("randomize"),
                                 "viewer": LaunchConfiguration("viewer"), "publish_gt_tf": LaunchConfiguration("publish_gt_tf"),
                                 "teleop": LaunchConfiguration("teleop"), "teleop_a_throttle": LaunchConfiguration("teleop_a_throttle"),
                                 "teleop_v_max": LaunchConfiguration("teleop_v_max")}])
    joy_node = Node(package="joy", executable="joy_node", name="joy", parameters=[{"deadzone": 0.02, "autorepeat_rate": 50.0}],
                    condition=IfCondition(LaunchConfiguration("joy")))
    joy_teleop = Node(package="f1sim_ros", executable="teleop", name="f1sim_teleop", output="screen",
                      parameters=[{"preset": LaunchConfiguration("joy_preset")}], condition=IfCondition(LaunchConfiguration("joy")))
    ackermann_to_vesc = Node(package="vesc_ackermann", executable="ackermann_to_vesc_node", name="ackermann_to_vesc_node",
                             parameters=[LaunchConfiguration("vesc_config")])
    vesc_to_odom = Node(package="vesc_ackermann", executable="vesc_to_odom_node", name="vesc_to_odom_node",
                        parameters=[LaunchConfiguration("vesc_config")])
    mux = Node(package="ackermann_mux", executable="ackermann_mux", name="ackermann_mux",
               parameters=[LaunchConfiguration("mux_config")], remappings=[("ackermann_cmd_out", "ackermann_drive")])
    static_tf = Node(package="tf2_ros", executable="static_transform_publisher", name="static_baselink_to_laser",
                     arguments=["0.27", "0.0", "0.11", "0.0", "0.0", "0.0", "base_link", "laser"])
    from launch.conditions import LaunchConfigurationNotEquals
    policy = Node(package="f1sim_ros", executable="policy", name="f1sim_policy", output="screen",
                  parameters=[{"checkpoint": LaunchConfiguration("policy"), "speed_cap": LaunchConfiguration("policy_speed_cap")}],
                  condition=LaunchConfigurationNotEquals("policy", ""))
    return LaunchDescription(args + [vesc_sim, ackermann_to_vesc, vesc_to_odom, mux, static_tf, joy_node, joy_teleop, policy])
