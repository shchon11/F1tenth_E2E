"""`learn/bagdata.py`: a rosbag2 becomes the observations a node would have built.

The dataset is a *reconstruction* -- no bag holds a stacked observation, only the scans it is built
from -- so what has to be checked is that the reconstruction is the node's. These tests drive
`ObsBuilder` by hand over the same synthetic recording and require the arrays to match exactly,
then check the three things a reconstruction can silently get wrong: the action history it cannot
always know, the episode boundaries it must not stitch across, and which `/drive` answers which
scan.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")
pytest.importorskip("f1sim_interfaces.msg",
                    reason="f1sim_interfaces is not built; see f1sim_interfaces/README.md")
torch = pytest.importorskip("torch")

import bag_fixtures as bf                                     # noqa: E402
from f1sim.learn import bagdata                               # noqa: E402
from f1sim.learn.obs import ObsBuilder, ObsSpec, resample_ranges  # noqa: E402

SPEC = ObsSpec(n_beams=64, scan_stack=3, scan_stride=1, action_history=2, act_dim=8, hist_len=0,
               range_max=10.0, v_max=8.0)
DT = 0.025


def test_the_dataset_is_the_observation_the_node_would_have_built(tmp_path):
    path = bf.graph_run(str(tmp_path / "run"), frames=24, dt=DT)
    d = bagdata.from_bag(path, SPEC, speed_cap=4.0)
    assert len(d) == 24 and d.meta["action_source"] == "plan"

    # The same frames through `ObsBuilder` directly, which is what `policy_node.on_scan` does.
    b = ObsBuilder(SPEC, "cpu")
    for i in range(24):
        r = resample_ranges(np.full(SPEC.n_beams, 3.0 + 0.01 * i, np.float32), SPEC.range_max,
                            SPEC.n_beams)
        s, p = b.build(r, 0.1 * i, np.array([0.0, 0.0, 0.02 * i, 0.5, 0.0, 9.80665]),
                       (0.01, -0.02), 4.0)
        b.push_action(np.full(8, 0.01 * i, dtype=np.float32))
        assert np.array_equal(d.scan[i], s[0].numpy()), f"frame {i}: scan"
        assert np.allclose(d.proprio[i], p[0].numpy(), atol=0, rtol=0), f"frame {i}: proprio"


def test_the_command_that_answers_a_scan_is_the_one_labelled(tmp_path):
    path = bf.graph_run(str(tmp_path / "run"), frames=10, dt=DT)
    d = bagdata.from_bag(path, SPEC)
    assert d.command_valid.all()
    assert d.command[:, 1] == pytest.approx([0.5 + 0.05 * i for i in range(10)])
    # A window narrower than the publish lag labels nothing rather than labelling the wrong row.
    tight = bagdata.from_bag(path, SPEC, command_window=1e-6)
    assert not tight.command_valid.any()


def test_a_reset_starts_a_new_episode_and_clears_the_history(tmp_path):
    """A history stitched across a reset describes a run that never happened. The node clears it
    there, so the dataset has to as well -- and a consumer has to be able to see where."""
    path = bf.graph_run(str(tmp_path / "run"), frames=20, dt=DT, reset_at=10)
    d = bagdata.from_bag(path, SPEC)
    assert d.episode[:10].tolist() == [0] * 10
    assert d.episode[10:].tolist() == [1] * 10
    # frame 10 is the first of its segment: every frame of its scan stack is that one scan
    assert np.array_equal(d.scan[10][0], d.scan[10][1]) and np.array_equal(d.scan[10][1], d.scan[10][2])
    # and the previous-action channels are zero again
    assert np.allclose(d.proprio[10][1:1 + SPEC.act_dim * SPEC.action_history], 0.0)
    assert not np.allclose(d.proprio[9][1:1 + SPEC.act_dim * SPEC.action_history], 0.0)


def test_without_a_plan_topic_the_action_history_is_zeros_and_says_so(tmp_path):
    """A raw car recording has no `/f1sim/plan`. The rest of the observation is still exact; the
    previous-action channels are not, and a dataset that did not say so would be a trap."""
    path = bf.graph_run(str(tmp_path / "run"), frames=8, dt=DT, with_plan=False)
    d = bagdata.from_bag(path, SPEC)
    assert d.meta["action_source"] == "zeros"
    assert np.allclose(d.proprio[:, 1:1 + SPEC.act_dim * SPEC.action_history], 0.0)
    assert d.plan is None


def test_a_direct_action_run_recovers_its_action_history_from_drive(tmp_path):
    path = bf.graph_run(str(tmp_path / "run"), frames=8, dt=DT, with_plan=False)
    spec2 = ObsSpec(n_beams=64, scan_stack=3, action_history=2, act_dim=2, hist_len=0,
                    range_max=10.0, v_max=8.0)
    d = bagdata.from_bag(path, spec2)
    assert d.meta["action_source"] == "drive"
    # frame 1's "previous action" is frame 0's command, inverted back into the action space
    assert d.proprio[1][1] == pytest.approx(0.0 / 0.4189, abs=1e-6)
    assert d.proprio[1][2] == pytest.approx(2.0 * 0.5 / 8.0 - 1.0, abs=1e-6)


def test_pose_is_the_drifting_odom_and_ground_truth_is_kept_apart(tmp_path):
    path = bf.graph_run(str(tmp_path / "run"), frames=6, dt=DT)
    d = bagdata.from_bag(path, SPEC)
    assert d.pose is not None and d.pose.shape == (6, 3)
    assert d.pose[3, 0] == pytest.approx(0.15)           # 0.05 * 3, the /odom x
    assert d.pose_gt is None, "a bag with no /ego_racecar/odom must not invent ground truth"


def test_a_bag_missing_a_required_topic_is_refused_with_the_reason(tmp_path):
    rec = [(100.0 + i * DT, "/scan", bf.scan(100.0 + i * DT)) for i in range(5)]
    path = bf.write(str(tmp_path / "scanonly"), rec)
    with pytest.raises(ValueError, match="/sensors/imu/raw"):
        bagdata.from_bag(path, SPEC)


def test_a_bag_that_was_never_finalised_is_diagnosed_rather_than_reported_as_not_a_bag(tmp_path):
    """The failure people hit: a recorder killed before `metadata.yaml` was written. The `.db3` is
    full of data and nothing can open it, and rosbag2's "No storage could be initialized" reads as
    "this is not a bag"."""
    from f1sim.calib.bagread import check_readable
    path = bf.graph_run(str(tmp_path / "run"), frames=4, dt=DT)
    check_readable(path)                                   # a finished bag: nothing to say
    os.remove(os.path.join(path, "metadata.yaml"))
    with pytest.raises(RuntimeError, match="ros2 bag reindex"):
        check_readable(path)
    with pytest.raises(RuntimeError, match="no metadata.yaml"):
        bagdata.from_bag(path, SPEC)
    # Anything else is left to the reader: guessing would replace one wrong diagnosis with another.
    check_readable(str(tmp_path / "does-not-exist"))
    (tmp_path / "empty").mkdir()
    check_readable(str(tmp_path / "empty"))


def test_round_trip_through_npz(tmp_path):
    path = bf.graph_run(str(tmp_path / "run"), frames=12, dt=DT)
    d = bagdata.from_bag(path, SPEC, speed_cap=4.0)
    out = d.save(str(tmp_path / "ds.npz"))
    back = bagdata.load(out)
    assert back.name == d.name and back.spec == d.spec
    assert back.meta["action_source"] == d.meta["action_source"]
    for f in ("t", "scan", "proprio", "command", "command_valid", "speed", "att", "imu",
              "episode", "pose", "plan"):
        a, b = getattr(d, f), getattr(back, f)
        assert (a is None) == (b is None), f
        if a is not None:
            assert np.array_equal(a, b), f
    assert back.summary() == d.summary()


def test_the_dropped_odom_is_held_not_interpolated(tmp_path):
    """The node holds the last `/odom` it was given. An interpolated wheel speed is a speed the car
    never reported, and it is the channel the traction guard reasons about."""
    path = bf.graph_run(str(tmp_path / "run"), frames=10, dt=DT, drop_odom_from=5)
    d = bagdata.from_bag(path, SPEC)
    assert d.speed[:5] == pytest.approx([0.1 * i for i in range(5)])
    assert d.speed[5:] == pytest.approx([0.4] * 5)       # frame 4's value, held
