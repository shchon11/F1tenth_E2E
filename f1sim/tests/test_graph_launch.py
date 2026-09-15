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


def _declared_types(checkpoint):
    """`{parameter: python type}` for every parameter the two nodes declare, read off real nodes."""
    import rclpy
    from f1sim_ros.controller_node import ControllerNode
    from f1sim_ros.policy_node import PolicyNode
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init(args=["--ros-args", "--params-file", GRAPH_YAML,
                     "-p", f"checkpoint:={checkpoint}", "-p", "device:=cpu"])
    out = {}
    try:
        for cls in (PolicyNode, ControllerNode):
            node = cls()
            for name, p in node._parameters.items():
                out[name] = type(p.value)
            node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return out


@pytest.mark.parametrize("target", TARGETS + ("graph_eval",))
def test_every_launch_argument_reaches_its_node_as_the_type_it_declares(target, checkpoint):
    """A `LaunchConfiguration` is a string, and `launch_ros` guesses a type from it with YAML rules.

    YAML 1.1 reads `off` as the boolean false, so `traction:=off` -- the deployment default, the
    value that means "install nothing" -- arrived at the controller as `False` against a STRING
    parameter and killed the node on startup. Every argument that reaches a node is therefore
    wrapped in a `ParameterValue` with an explicit type, and this evaluates the launch description
    to check it, rather than trusting that somebody remembered.
    """
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    want = _declared_types(checkpoint)
    ld = load_launch(target)
    ctx = LaunchContext()
    for e in ld.entities:
        if isinstance(e, DeclareLaunchArgument):
            e.execute(ctx)
    checked = 0
    for node in nodes_of(ld):
        for d in evaluate_parameters(ctx, node._Node__parameters):
            if not isinstance(d, dict):
                continue                       # a params FILE; its types are the file's
            for name, value in d.items():
                if name not in want:
                    continue
                checked += 1
                assert isinstance(value, want[name]), (
                    f"{target}: {name} reaches the node as {type(value).__name__} "
                    f"{value!r}, declared {want[name].__name__}")
    assert checked, f"{target}: no parameters were checked, so nothing was"


def test_the_traction_default_survives_the_launch_files_yaml_guess():
    """The specific value that broke: `off`. Named, because it is the deployment default and
    because the failure was a node that would not start rather than one that drove wrong."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.utilities import evaluate_parameters
    ld = load_launch("graph_car")
    ctx = LaunchContext()
    for e in ld.entities:
        if isinstance(e, DeclareLaunchArgument):
            e.execute(ctx)
    got = [d["traction"] for n in nodes_of(ld)
           for d in evaluate_parameters(ctx, n._Node__parameters)
           if isinstance(d, dict) and "traction" in d]
    assert got == ["off"], got


def test_the_rviz_layout_shows_what_the_graph_publishes():
    """`config/graph.rviz` has to name the topics the controller actually publishes.

    A layout that referenced `/f1sim/viz/plan_ref` would open, show nothing, and look like a
    controller that was not publishing. rviz itself is not opened here -- that is a GUI process and
    this suite does not start one -- so what is checked is that the file parses and that every
    topic it displays is one something in this package publishes.
    """
    path = os.path.join(ROS, "config", "graph.rviz")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    topics = set()
    for d in cfg["Visualization Manager"]["Displays"]:
        t = d.get("Topic")
        if isinstance(t, dict) and "Value" in t:
            topics.add(t["Value"])
    assert {"/f1sim/viz/plan", "/f1sim/viz/clearance", "/f1sim/viz/diag"} <= topics, sorted(topics)
    assert {"/scan", "/odom", "/ego_racecar/odom", "/map"} <= topics, sorted(topics)
    published = open(os.path.join(ROS, "f1sim_ros", "controller_node.py")).read()
    for t in ("/f1sim/viz/plan", "/f1sim/viz/clearance", "/f1sim/viz/diag"):
        assert f'"{t}"' in published, f"{t} is in the rviz layout and nothing publishes it"


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
