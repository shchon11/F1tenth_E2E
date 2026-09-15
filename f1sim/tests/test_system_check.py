""""Is everything there?" -- the checks, over a bag and over a synthetic timeline.

`system_check` is the tool somebody runs on the car with the graph up and a bag recording, so the
question it has to answer correctly is the boring one: which topic is missing, which is slow, and
what was actually driving. These tests are the rules; the node that applies them live is a thin
wrapper over `check_timeline`, which is what is exercised here.

The distinction that carries the most weight is **required vs optional**. `/f1sim/plan` is absent
from every baseline run by design (worker 18's `baseline_node` publishes `/drive` directly), and a
checker that called that unhealthy would be wrong about the one deployment it was most needed for.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "f1sim_ros"))

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")
pytest.importorskip("f1sim_interfaces.msg",
                    reason="f1sim_interfaces is not built; see f1sim_interfaces/README.md")

import bag_fixtures as bf                                      # noqa: E402
from f1sim_ros import record_profile, system_check as sc       # noqa: E402

DT = 0.025


@pytest.fixture(scope="module")
def profile():
    return record_profile.load()


def by_name(rep):
    return {t.name: t for t in rep.topics}


# ==================================================================== the profile
def test_the_profile_lists_the_graphs_topics_and_says_which_are_required(profile):
    names = profile.names
    for t in ("/scan", "/sensors/imu/raw", "/odom", "/drive"):
        assert t in profile.required, f"{t} is what the graph cannot run without"
    assert "/f1sim/plan" in names and "/f1sim/plan" not in profile.required, (
        "a baseline node publishes /drive with no plan; requiring one would fail every such run")
    assert "/ego_racecar/odom" not in profile.on_car().names, "ground truth is simulator-only"


def test_a_profile_with_an_unknown_key_or_a_duplicate_is_refused(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("topics:\n  - {name: /a, type: std_msgs/msg/Empty, rat_hz: 1}\n")
    with pytest.raises(ValueError, match="unknown key"):
        record_profile.load(str(p))
    p.write_text("topics:\n"
                 "  - {name: /a, type: std_msgs/msg/Empty}\n"
                 "  - {name: /a, type: std_msgs/msg/Empty}\n")
    with pytest.raises(ValueError, match="more than once"):
        record_profile.load(str(p))


# ==================================================================== the timeline rules
def _healthy(profile, seconds=5.0):
    out = {}
    for t in profile.topics:
        hz = t.rate_hz or 1.0
        out[t.name] = np.arange(0.0, seconds, 1.0 / hz)
    return out


def test_a_full_healthy_stream_is_ok(profile):
    rep = sc.check_timeline(_healthy(profile), profile, duration_s=5.0)
    assert rep.ok, rep.text()
    assert all(t.status == sc.OK for t in rep.topics), rep.text()


def test_a_missing_required_topic_is_not_ok_and_is_named(profile):
    st = _healthy(profile)
    del st["/odom"]
    rep = sc.check_timeline(st, profile, duration_s=5.0)
    assert not rep.ok
    assert rep.missing_required == ["/odom"]
    assert "MISSING REQUIRED: /odom" in rep.text()


def test_a_missing_optional_topic_is_reported_but_healthy(profile):
    st = _healthy(profile)
    del st["/f1sim/plan"]; del st["/sensors/core"]
    rep = sc.check_timeline(st, profile, duration_s=5.0)
    assert rep.ok, rep.text()
    assert by_name(rep)["/f1sim/plan"].status == sc.MISSING
    assert "optional" in by_name(rep)["/f1sim/plan"].note


def test_a_topic_that_stopped_is_stale_not_merely_slow(profile):
    st = _healthy(profile)
    st["/scan"] = st["/scan"][st["/scan"] < 2.0]
    rep = sc.check_timeline(st, profile, duration_s=5.0, now=5.0, stale_after=0.5)
    assert not rep.ok
    c = by_name(rep)["/scan"]
    assert c.status == sc.WARN and "stale" in c.note and c.age_s > 2.9


def test_a_topic_at_a_third_of_its_rate_is_slow(profile):
    st = _healthy(profile)
    st["/scan"] = np.arange(0.0, 5.0, 1.0 / 12.0)          # 12 Hz against a declared 40
    rep = sc.check_timeline(st, profile, duration_s=5.0)
    c = by_name(rep)["/scan"]
    assert c.status == sc.WARN and "slow" in c.note
    assert c.rate_hz == pytest.approx(12.0, rel=0.05)


def test_a_topic_well_above_its_nominal_rate_is_flagged_too(profile):
    """`/drive` at 55 Hz against a 40 Hz nominal is the controller's watchdog braking between
    plans. It looks like a healthy command stream, and it is the opposite."""
    st = _healthy(profile)
    st["/drive"] = np.arange(0.0, 5.0, 1.0 / 90.0)
    rep = sc.check_timeline(st, profile, duration_s=5.0)
    c = by_name(rep)["/drive"]
    assert c.status == sc.WARN and "fast" in c.note
    assert not rep.ok, "/drive is required, so an unhealthy /drive is an unhealthy graph"


def test_a_latched_topic_is_not_judged_on_its_rate_or_its_age(profile):
    """`/tf_static` and `/f1sim/reset` fire once and then never again. Silence is their normal
    state, and a "rate" over two latched messages a microsecond apart is a number with no meaning."""
    st = _healthy(profile)
    st["/tf_static"] = np.array([0.0, 1e-6])
    st["/f1sim/reset"] = np.array([0.4])
    rep = sc.check_timeline(st, profile, duration_s=5.0, now=5.0)
    for name in ("/tf_static", "/f1sim/reset"):
        c = by_name(rep)[name]
        assert c.status == sc.OK and c.rate_hz == 0.0, (name, c.note)
    assert rep.ok


# ==================================================================== over a bag
def test_a_bag_of_the_graph_reports_what_was_driving(tmp_path, profile):
    path = bf.graph_run(str(tmp_path / "run"), frames=80, dt=DT, with_state=True)
    rep = sc.check_bag(path, profile, stale_after=0.5)
    c = by_name(rep)
    assert c["/scan"].count == 80 and c["/scan"].rate_hz == pytest.approx(40.0, rel=0.05)
    assert c["/f1sim/plan"].status == sc.OK
    assert rep.versions["checkpoint"].endswith("0123456789ab")
    assert rep.versions["memory"] == "gru(16)"
    assert rep.versions["arm"] == "fixed_low" and rep.versions["mu"] == "0.73423"
    assert rep.versions["traction"] == "off"
    assert rep.missing_required == []
    assert "in force:" in rep.text()


def test_the_checkpoint_is_recoverable_from_the_plan_alone(tmp_path, profile):
    """A bag recorded with a narrower profile still says which weights drove: the `Plan` carries
    the checkpoint on every message, which is what that field is for."""
    path = bf.graph_run(str(tmp_path / "planonly"), frames=20, dt=DT, with_state=False)
    rep = sc.check_bag(path, profile)
    assert rep.versions["checkpoint"].endswith("0123456789ab")
    assert "arm" not in rep.versions, "no diag in the bag, so no arm is claimed"


def test_a_bag_with_a_missing_topic_says_which_and_fails(tmp_path, profile):
    """The contract's case: a recording that looks fine until something reads it."""
    path = bf.graph_run(str(tmp_path / "nodrive"), frames=40, dt=DT, with_drive=False)
    rep = sc.check_bag(path, profile)
    assert not rep.ok
    assert rep.missing_required == ["/drive"]
    assert by_name(rep)["/drive"].status == sc.MISSING
    assert "MISS" in rep.text() and "/drive" in rep.text()
    assert rep.as_dict()["ok"] is False


def test_a_bag_whose_odom_stops_halfway_is_caught_even_though_the_topic_is_there(tmp_path, profile):
    """The failure a topic list cannot see: `/odom` is in the bag, and dead for the second half."""
    path = bf.graph_run(str(tmp_path / "halfodom"), frames=80, dt=DT, drop_odom_from=40)
    rep = sc.check_bag(path, profile, stale_after=0.2)
    c = by_name(rep)["/odom"]
    assert c.status == sc.WARN and c.count == 40
    assert not rep.ok and "stale" in c.note


def test_a_baseline_style_bag_with_no_plan_is_healthy(tmp_path, profile):
    """A `/drive` publisher that is not this graph: every required topic, no plan, no diag."""
    path = bf.graph_run(str(tmp_path / "baseline"), frames=60, dt=DT, with_plan=False)
    rep = sc.check_bag(path, profile)
    assert rep.ok, rep.text()
    assert by_name(rep)["/f1sim/plan"].status == sc.MISSING
    assert rep.versions == {}, "nothing in the bag said what was driving, so nothing is claimed"
