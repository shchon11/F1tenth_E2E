"""--rand-widen / --rand-range: the boot-camp widening of the physics randomisation."""
import pytest

from f1sim.params import Config


def test_widen_one_is_identity_and_widening_keeps_the_midpoint_and_signs():
    c = Config()
    assert c.rand.widened(1.0) == c.rand.ranges
    w = c.rand.widened(2.0)
    for k, (lo, hi) in c.rand.ranges.items():
        nlo, nhi = w[k]
        assert nhi - nlo >= (hi - lo) - 1e-12
        if lo >= 0:
            assert nlo >= 0
        if k in c.rand.scale_fields:
            assert nlo >= 0.1
    lo, hi = c.rand.ranges["vehicle.mu"]
    assert w["vehicle.mu"] == pytest.approx((lo - (hi - lo) / 2, hi + (hi - lo) / 2))


def test_widen_refuses_a_non_positive_factor():
    with pytest.raises(ValueError):
        Config().rand.widened(0.0)
