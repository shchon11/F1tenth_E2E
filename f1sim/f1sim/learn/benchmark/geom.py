"""Longitudinal-arc constants, derived from the footprint the contact test actually uses.

`_car_contacts` (sim.py:211-214) does not test the four `vehicle.length x vehicle.width` corners
alone: it appends two rear-box corners at depth `car_rear[:, 0]`, drawn per env from u(0.06, 0.11)
(sim.py:172). Nose-to-tail contact is therefore 0.64-0.69 m, not 0.58. We take the declared worst
case so one suite-wide constant holds against any draw, and assert the draw stays inside it.

These are thresholds on arc separation. On a curve, arc separation and Euclidean body gap differ;
nothing here is a 2-D physical body-gap guarantee.
"""
from __future__ import annotations

#: Upper bound of `car_rear[:, 0]` = u(0.06, 0.11), sim.py:172. Asserted against the live sim.
REAR_BOX_MAX_M = 0.11

#: Declared arc clearance added beyond contact to call a pass "clear".
ARC_CLEARANCE_M = 0.20


def contact_extent_m(vehicle_length_m: float, rear_box_m: float = REAR_BOX_MAX_M) -> float:
    """Worst-case longitudinal contact extent, nose to tail."""
    return vehicle_length_m + rear_box_m


def thresholds(vehicle_length_m: float, rear_box_m: float = REAR_BOX_MAX_M,
               clearance_m: float = ARC_CLEARANCE_M) -> tuple[float, float]:
    """`(overlap, clear)` in metres of arc. overlap = bodies may touch; clear = clean air."""
    overlap = contact_extent_m(vehicle_length_m, rear_box_m)
    return overlap, overlap + clearance_m


def assert_rear_box_within_bound(car_rear_col0, bound: float = REAR_BOX_MAX_M) -> float:
    """Fail the cell closed if a future randomisation widens the rear box past the declared bound.

    Takes the (B,) rear-depth column; returns the observed maximum so the runner can record it.
    """
    observed = float(car_rear_col0.max())
    if observed > bound + 1e-9:
        raise ValueError(
            f"rear box {observed:.4f} m exceeds the declared bound {bound} m: the pass thresholds "
            f"were derived from that bound and no longer clear the contact envelope")
    return observed
