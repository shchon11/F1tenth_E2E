"""A held dial (2026-10-20): one friction for every car and episode, so the dial carries no information."""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn.conditioning import ConstantDial, DialDraw   # noqa: E402


def test_the_constant_dial_ignores_the_floor_and_the_episode():
    env = SimpleNamespace(B=4, device="cpu", sim=SimpleNamespace(P={"mu": torch.tensor([0.7, 0.8, 1.0, 1.1])}))
    d = ConstantDial(env, 0.94)
    d.redraw(torch.ones(4, dtype=torch.bool))
    assert torch.equal(d.value(), torch.full((4,), 0.94))
    # the drawn dial, by contrast, follows each floor
    dd = DialDraw(env, 0.0, 1.0)
    dd.redraw(torch.ones(4, dtype=torch.bool))
    assert torch.allclose(dd.value(), env.sim.P["mu"].clamp_min(dd.floor))


def test_evaluate_finds_the_held_dial_where_ppo_records_it():
    """`ppo --dial-constant` writes it into extra["experiment"]; evaluate's fallback read the top level only."""
    from f1sim.learn.evaluate import held_dial_of
    assert held_dial_of({"experiment": {"dial_constant": 1.0489}}) == 1.0489
    assert held_dial_of({"dial_constant": 0.9}) == 0.9
    assert held_dial_of({"experiment": {}}) is None and held_dial_of(None) is None
