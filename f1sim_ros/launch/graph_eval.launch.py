"""Score one suite cell over the graph: the eval node's simulator, the policy and the controller.

    ros2 launch f1sim_ros graph_eval.launch.py \
        checkpoint:=$HOME/f1sim_runs/ppo_race_0910/ppo_latest.pt \
        suite:=v2.1 cell:='S:gen:control:1400:0.73423:4401' out:=/tmp/cell.jsonl

The eval node IS the sensor source here -- it owns the env the benchmark builds for that cell -- so
there is no bridge. One learner, real time, and the row it writes says so; suites are still scored
batched (`python -m f1sim.learn.benchmark run`). See docs/ros2.md, "Evaluation over the graph".
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from f1sim_ros.launch_graph import arg, common_arguments, graph_nodes


def generate_launch_description():
    args = common_arguments() + [
        DeclareLaunchArgument("suite", default_value="v2.1"),
        DeclareLaunchArgument("cell", default_value=""),
        DeclareLaunchArgument("out", default_value=""),
        DeclareLaunchArgument("envs", default_value="1"),
        DeclareLaunchArgument("sync", default_value="true",
                              description="false free-runs at the control rate, as the car does"),
        # The car's 0.25 s is a statement about a real LiDAR. Here the "LiDAR" is a CPU simulator
        # stepping at a fraction of real time, so a scan legitimately arrives hundreds of
        # milliseconds after the last one and the graph would (correctly) call it stale, clear its
        # histories and score a policy that never sees motion. The eval node refuses the run if
        # these are too small for the loop it measures.
        DeclareLaunchArgument("sensor_timeout", default_value="10.0"),
        DeclareLaunchArgument("plan_timeout", default_value="10.0"),
    ]
    ev = Node(package="f1sim_ros", executable="eval", name="f1sim_eval", output="screen",
              parameters=[LaunchConfiguration("config_yaml"),
                          {"suite": arg("suite"), "cell": arg("cell"), "out": arg("out"),
                           "envs": arg("envs", int), "sync": arg("sync", bool),
                           "arm": arg("controller"), "checkpoint": arg("checkpoint"),
                           "device": arg("device")}])
    return LaunchDescription(args + [ev] + graph_nodes(
        policy_kw={"sensor_timeout": arg("sensor_timeout", float)},
        controller_kw={"sensor_timeout": arg("sensor_timeout", float),
                       "plan_timeout": arg("plan_timeout", float)}))
