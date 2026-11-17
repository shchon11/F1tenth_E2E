"""`Track.with_smooth_walls` (2026-11-17): a wall off the grid axes is a straight wall, not 5 cm stairs; real corners
and the hose / wall classes stay where they were; the line and the venue's name carry over."""
from __future__ import annotations

import numpy as np

from f1sim import Track

RES = 0.05


def _slanted(slope=0.1, offset=1.0, size=(80, 120)):
    """solid above y = slope * x + offset (about 6 degrees: one 5 cm jog every half metre)"""
    H, W = size
    yy, xx = np.mgrid[0:H, 0:W]
    x, y = (xx + 0.5) * RES, (yy + 0.5) * RES
    occ = y > slope * x + offset
    return Track.from_occupancy(occ, RES, (0.0, 0.0), duct=occ.copy(), tall=np.zeros_like(occ), duct_height=0.33)


def _boundary_y(t):
    """per column of the grid, the y of the first solid cell's lower edge"""
    occ = np.asarray(t.occupancy, bool)
    first = occ.argmax(0)
    x = (np.arange(occ.shape[1]) + 0.5) * t.resolution + t.origin[0]
    return x, first * t.resolution + t.origin[1]


def test_a_slanted_wall_is_straight_not_stairs():
    t = _slanted()
    s = t.with_smooth_walls()
    for tr, worst in ((t, 0.03), (s, 0.012)):
        x, yb = _boundary_y(tr)
        keep = (x > 0.5) & (x < 5.5)
        err = np.abs(yb[keep] - (0.1 * x[keep] + 1.0))
        assert err.max() < worst + 1e-9, (tr.resolution, err.max())
    x, yb = _boundary_y(_slanted())
    assert np.abs(yb - (0.1 * x + 1.0))[(x > 0.5) & (x < 5.5)].max() > 0.02      # the stairs were there to remove


def test_a_real_corner_stays_and_classes_carry_over():
    H, W = 100, 100
    occ = np.zeros((H, W), bool); occ[50:, 50:] = True                             # a right angle at (2.5, 2.5)
    duct = occ.copy(); duct[60:, 60:] = False                                       # hose shell, unseen solid inside
    t = Track.from_occupancy(occ, RES, (0.0, 0.0), duct=duct, tall=np.zeros_like(occ), duct_height=0.33)
    s = t.with_smooth_walls()
    f = int(round(RES / s.resolution))
    assert s.occupancy.shape == (H * f, W * f)
    # the corner cell's neighbourhood: solid within ~4 cm of the corner, floor beyond it
    def solid_at(x, y):
        return bool(s.occupancy[int(y / s.resolution), int(x / s.resolution)])
    assert solid_at(2.54, 2.54) and solid_at(3.0, 2.52) and solid_at(2.52, 3.0)
    assert not solid_at(2.45, 3.0) and not solid_at(3.0, 2.45)
    # classes: what the original cell was
    assert s.duct[int(2.6 / s.resolution), int(2.6 / s.resolution)]
    assert not s.duct[int(4.0 / s.resolution), int(4.0 / s.resolution)] and s.occupancy[int(4.0 / s.resolution), int(4.0 / s.resolution)]


def test_the_original_rides_along_for_its_line_and_name():
    t = _slanted()
    t.centerline = np.array([[0.5, 0.5], [5.5, 0.5], [5.5, 0.9], [0.5, 0.9]])
    s = t.with_smooth_walls()
    assert s.name == t.name and s.smooth_parent is t
    r = s.reversed()
    assert r.smooth_parent is not None and np.allclose(r.smooth_parent.centerline, t.centerline[::-1])
