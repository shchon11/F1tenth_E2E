"""The launch files and `config/graph.yaml`: one parameter set, three sensor sources.

The claim the launch layout makes is that a car run and a simulator run are the same run with a
different sensor source -- so the thing to check is not that the files parse (they would parse
while launching the wrong arm) but that:

* all three targets bring up the same two nodes, from the same parameter file;
* `config/graph.yaml` actually carries **every** parameter both nodes declare, so nothing that can
  change what the car does is reachable only from a command line somebody has to remember;
* the defaults in that file are the deployment defaults -- `fixed_low`, traction `off`,
  `steer_gain` 1.0 -- because the file is what a reader will believe.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "f1sim_ros"))

pytest.importorskip("rclpy")
pytest.importorskip("launch")
pytest.importorskip("f1sim_interfaces.msg",
                    reason="f1sim_interfaces is not built; see f1sim_interfaces/README.md")
torch = pytest.importorskip("torch")

ROS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "f1sim_ros")
LAUNCH = os.path.join(ROS, "launch")
GRAPH_YAML = os.path.join(ROS, "config", "graph.yaml")
TARGETS = ("graph_sim", "graph_console", "graph_car")

#: rclpy declares this one itself on every node; it is not ours to carry in the file.
BUILTIN = {"use_sim_time"}


def load_launch(name):
    from launch.launch_description_sources import get_launch_description_from_python_launch_file
    return get_launch_description_from_python_launch_file(os.path.join(LAUNCH, f"{name}.launch.py"))


def nodes_of(ld):
    from launch_ros.actions import Node
    out = []
    for e in ld.entities:
        if isinstance(e, Node):
            out.append(e)
    return out


def executables(ld):
    return sorted(str(n.node_executable) if isinstance(n.node_executable, str)
                  else str(n.node_executable.describe()) for n in nodes_of(ld))


@pytest.fixture(scope="module")
def graph_yaml():
    with open(GRAPH_YAML) as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("target", TARGETS)
def test_every_target_brings_up_the_policy_and_the_controller(target):
    ld = load_launch(target)
    ex = executables(ld)
    assert "policy" in ex and "controller" in ex, (target, ex)


def test_the_targets_differ_only_in_the_sensor_source():
    ex = {t: set(executables(load_launch(t))) for t in TARGETS}
    graph = {"policy", "controller"}
    for t, e in ex.items():
        assert graph <= e, (t, e)
    assert ex["graph_console"] - graph <= {"rviz2"}, "the console owns its own simulation"
    assert "bridge" in ex["graph_sim"], "the simulator target needs the standalone bridge"
    assert "bridge" not in ex["graph_car"], "a car launch must not start a simulator"


@pytest.mark.parametrize("target", TARGETS)
def test_every_target_takes_the_arguments_that_change_what_the_car_does(target):
    from launch.actions import DeclareLaunchArgument
    ld = load_launch(target)
    names = {e.name for e in ld.entities if isinstance(e, DeclareLaunchArgument)}
    for a in ("config_yaml", "checkpoint", "device", "speed_cap", "controller", "traction"):
        assert a in names, (target, a, sorted(names))


def test_the_yaml_defaults_are_the_deployment_defaults(graph_yaml):
    c = graph_yaml["/**/f1sim_controller"]["ros__parameters"]
    p = graph_yaml["/**/f1sim_policy"]["ros__parameters"]
    assert c["controller"] == "fixed_low"
    assert c["traction"] == "off", "the guard has never run on a moving car; it is opt-in"
    assert c["steer_gain"] == 1.0, "the two recorded servo configurations have opposite polarity"
    assert c["speed_cap"] == p["speed_cap"], "the policy reads the cap as an observation channel"
    assert c["sensor_timeout"] == p["sensor_timeout"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from f1sim.learn.model import ActorCritic, save_checkpoint
    from f1sim.learn.obs import ObsSpec
    spec = ObsSpec(n_beams=64, scan_stack=2, act_dim=8, hist_len=0)
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=9, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(3)
        model = ActorCritic(**meta)
    path = str(tmp_path_factory.mktemp("launch") / "c.pt")
    save_checkpoint(path, model, {"spec": spec.__dict__.copy()})
    return path


def test_graph_yaml_carries_every_parameter_both_nodes_declare(checkpoint, graph_yaml):
    """The point of one file is that it is the whole configuration. A parameter the nodes declare
    and the file does not mention is one whose value nobody reviewing a deployment will see."""
    import rclpy
    from f1sim_ros.controller_node import ControllerNode
    from f1sim_ros.policy_node import PolicyNode

    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init(args=["--ros-args", "--params-file", GRAPH_YAML,
                     "-p", f"checkpoint:={checkpoint}", "-p", "device:=cpu"])
    try:
        for cls, section in ((PolicyNode, "/**/f1sim_policy"),
                             (ControllerNode, "/**/f1sim_controller")):
            node = cls()
            declared = set(node._parameters) - BUILTIN
            node.destroy_node()
            in_file = set(graph_yaml[section]["ros__parameters"])
            assert declared <= in_file, (
                f"{section}: declared but not in config/graph.yaml: {sorted(declared - in_file)}")
            assert in_file <= declared, (
                f"{section}: in config/graph.yaml but declared by nobody: "
                f"{sorted(in_file - declared)}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_the_system_check_section_is_the_checkers_own_parameters(graph_yaml, checkpoint):
    """And not a second copy of the topic list: `config/record.yaml` owns which topics the graph
    needs, so the checker and the recorder cannot disagree about it."""
    import rclpy
    from f1sim_ros.system_check_node import SystemCheckNode
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init(args=["--ros-args", "--params-file", GRAPH_YAML])
    try:
        node = SystemCheckNode()
        declared = set(node._parameters) - BUILTIN
        node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    in_file = set(graph_yaml["/**/f1sim_system_check"]["ros__parameters"])
    assert declared == in_file, (sorted(declared - in_file), sorted(in_file - declared))
    assert "topics" not in in_file, "the topic list lives in config/record.yaml"
