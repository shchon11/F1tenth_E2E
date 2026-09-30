"""`ppo.lagrange_step`: contact rates as constraints, their prices set by dual ascent."""
import torch

from f1sim.learn.ppo import lagrange_step


def _spec(target, lam):
    return {"wall": {"index": 1, "weight": 8.0, "target": target, "lam": lam, "rate": float("nan")}}


def test_a_violated_constraint_raises_its_price_and_the_reward_carries_it_instead_of_the_fixed_weight():
    T, B = 100, 4
    comp = torch.zeros(T, B, 3)
    comp[10, 0, 1] = -8.0                                        # one onset at the env's weight 8
    prog = torch.full((T, B), 0.5)                               # 200 m in all -> 5 contacts/km
    rew = comp.sum(2) + 0.1
    spec = _spec(target=1.0, lam=10.0)
    out = lagrange_step(rew.clone(), comp, prog, torch.ones(T, B), spec, lr=2.0, lam_max=100.0)
    assert abs(spec["wall"]["rate"] - 5.0) < 1e-6
    assert abs(spec["wall"]["lam"] - 18.0) < 1e-6                # 10 + 2 * (5 - 1)
    assert abs(float(out[10, 0]) - (0.1 - 18.0)) < 1e-5          # the fixed -8 is gone, -lambda is in
    assert abs(float(out[11, 0]) - 0.1) < 1e-6


def test_a_satisfied_constraint_lowers_its_price_but_never_below_zero():
    T, B = 50, 2
    comp = torch.zeros(T, B, 3)
    spec = _spec(target=1.0, lam=1.0)
    lagrange_step(torch.zeros(T, B), comp, torch.full((T, B), 1.0), torch.ones(T, B), spec, lr=2.0, lam_max=100.0)
    assert spec["wall"]["rate"] == 0.0 and spec["wall"]["lam"] == 0.0
