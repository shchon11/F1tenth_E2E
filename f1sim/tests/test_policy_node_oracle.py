"""The ROS node will not open a checkpoint trained on privileged opponent tokens.

Two independent refusals stand between `f1sim.opp_token` and a real car, and this pins both:
`load_checkpoint` refuses the checkpoint (its `meta` says what it was trained on) and `ObsBuilder`
refuses the observation spec (it is what would have to build the block and cannot). Either alone
would be enough; both exist because a checkpoint written by an older path may carry only one of the
two records, and a policy that was trained to believe an oracle and is then fed zeros is a policy
driving on a lie, at speed, next to a wall.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "f1sim_ros"))

pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")

import f1sim_ros.policy_node as pn                                        # noqa: E402
from f1sim.learn.model import ActorCritic, load_checkpoint, save_checkpoint   # noqa: E402
from f1sim.learn.obs import ObsBuilder, ObsSpec                           # noqa: E402


def _oracle(tmp_path, meta_too=True):
    spec = ObsSpec(n_beams=32, scan_stack=2, act_dim=2, opp_token="future")
    model = ActorCritic(2, 32, spec.proprio_dim, 12, act_dim=2,
                        opp_token="future" if meta_too else None)
    p = str(tmp_path / f"oracle_{int(meta_too)}.pt")
    save_checkpoint(p, model, {"spec": spec.__dict__})
    return p, spec


def test_the_node_refuses_the_checkpoint_the_way_it_refuses_a_conditional_one(tmp_path) -> None:
    p, _spec = _oracle(tmp_path)
    with pytest.raises(ValueError, match="ORACLE"):
        load_checkpoint(p, "cpu")
    assert pn.ObsBuilder is ObsBuilder                # the node builds the car's observation with it


def test_a_checkpoint_carrying_only_the_spec_is_refused_as_well(tmp_path) -> None:
    """`meta` is the record `ActorCritic` writes; `extra['spec']` is the one the trainer writes.
    A file with only the second is refused too -- `oracle_inputs_of` reads both -- and the
    observation builder refuses it a second time from the spec alone."""
    p, _spec = _oracle(tmp_path, meta_too=False)
    with pytest.raises(ValueError, match="ORACLE"):
        load_checkpoint(p, "cpu")
    _model, extra = load_checkpoint(p, "cpu", allow_oracle=True)
    assert extra["spec"]["opp_token"] == "future"
    with pytest.raises(ValueError, match="not deployable"):
        ObsBuilder(ObsSpec(**extra["spec"]))
