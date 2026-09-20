"""The two nodes of the graph, and the arguments every target shares.

`graph_sim.launch.py`, `graph_console.launch.py` and `graph_car.launch.py` differ only in where
`/scan`, `/odom` and `/sensors/imu*` come from. Keeping the policy and controller definitions here
is what makes that true: a parameter added to one target is added to all three, and a car launch
cannot quietly be running a different controller than the simulator launch it was validated
against.

In the python package rather than next to the launch files because a launch file is loaded from
its path with `importlib`, which does not put its directory on `sys.path` -- so a launch file's
sibling module is not importable, while `f1sim_ros.launch_graph` is.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def arg(name, value_type=str):
    """A launch argument as a parameter of the RIGHT TYPE.

    `LaunchConfiguration` is a string, and `launch_ros` infers a type from it with YAML rules. That
    is not a cosmetic problem: YAML 1.1 reads `off` as the boolean false, so
    `traction:=off` -- the deployment default, the value that means "install nothing" -- arrived at
    the controller as `False` against a STRING parameter and killed the node on startup. Every
    argument that reaches a node therefore says what it is.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def share():
    return get_package_share_directory("f1sim_ros")


def common_arguments():
    """Arguments every target takes. `config_yaml` is the whole parameter set; the rest are the
    handful worth having on the command line, and they override the file."""
    return [
        DeclareLaunchArgument("config_yaml", default_value=os.path.join(share(), "config", "graph.yaml"),
                              description="every parameter of the graph; see config/graph.yaml"),
        DeclareLaunchArgument("checkpoint", default_value="",
                              description="exported actor; also the controller's observation spec"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("speed_cap", default_value="4.0",
                              description="told to BOTH nodes: it is an observation channel"),
        DeclareLaunchArgument("controller", default_value="fixed_low",
                              description="legacy | fixed_low | clearance | fixed_low+clearance"),
        DeclareLaunchArgument("traction", default_value="off"),
        DeclareLaunchArgument("viz", default_value="false",
                              description="controller markers on /f1sim/viz/plan and /clearance"),
        DeclareLaunchArgument("policy", default_value="true",
                              description="false runs the controller alone, for an external planner"),
    ]


def _overrides():
    """The command-line arguments, as a parameter dict laid over `config_yaml`.

    `speed_cap` reaches both nodes from one argument on purpose: the policy reads it as an
    observation channel and the tracker reads it as a ceiling, and two different values is a policy
    driving to a cap the car will not honour.
    """
    return {"checkpoint": arg("checkpoint"), "device": arg("device"),
            "speed_cap": arg("speed_cap", float)}


def policy_node(**kw):
    params = dict(_overrides()); params.update(kw)
    return Node(package="f1sim_ros", executable="policy", name="f1sim_policy", output="screen",
                condition=IfCondition(LaunchConfiguration("policy")),
                parameters=[LaunchConfiguration("config_yaml"), params])


def controller_node(**kw):
    params = dict(_overrides())
    params.update({"controller": arg("controller"), "traction": arg("traction"),
                   "viz": arg("viz", bool)})
    params.update(kw)
    return Node(package="f1sim_ros", executable="controller", name="f1sim_controller",
                output="screen", parameters=[LaunchConfiguration("config_yaml"), params])


def graph_nodes(policy_kw=None, controller_kw=None):
    """The two nodes, with per-target parameter overrides. `graph_eval` uses them to raise the
    staleness timeouts, because its "sensor stream" is a CPU simulator and not a LiDAR."""
    return [policy_node(**(policy_kw or {})), controller_node(**(controller_kw or {}))]
