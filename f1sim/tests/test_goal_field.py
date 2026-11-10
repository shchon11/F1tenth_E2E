"""The reward's distance left to the finish line (`plan_reward._field_one`) follows the course, not a gap in a wall."""
import numpy as np
from scipy import ndimage

from f1sim.learn.plan_reward import _field_one

RES = 0.05


def _ring_with_slit(slit: float):
    """A 12 x 8 m box, a 8 x 4 m island in it (corridors 2 m wide), the island cut by a `slit` m gap at x = 7"""
    H, W = int(8 / RES), int(12 / RES)
    wall = np.zeros((H, W), bool)
    wall[[0, -1], :] = True; wall[:, [0, -1]] = True
    wall[int(2 / RES):int(6 / RES), int(2 / RES):int(10 / RES)] = True
    wall[:, int(7 / RES):int((7 + slit) / RES)] &= ~((np.arange(H)[:, None] >= int(2 / RES)) & (np.arange(H)[:, None] < int(6 / RES)))
    return ndimage.distance_transform_edt(~wall).astype(np.float32) * RES


def _loop():
    """the middle of the corridors, forwards (+x along the bottom), from just past the line at x = 4"""
    pts = []
    for x in np.arange(4.2, 11.0, 0.1): pts.append((x, 1.0))
    for y in np.arange(1.0, 7.0, 0.1): pts.append((11.0, y))
    for x in np.arange(11.0, 1.0, -0.1): pts.append((x, 7.0))
    for y in np.arange(7.0, 1.0, -0.1): pts.append((1.0, y))
    for x in np.arange(1.0, 3.8, 0.1): pts.append((x, 1.0))
    return np.array(pts)


def _along(D):
    p = _loop()
    return D[np.round(p[:, 1] / RES).astype(int), np.round(p[:, 0] / RES).astype(int)]


def test_distance_left_never_grows_along_the_course_past_a_gap():
    edt = _ring_with_slit(0.3)
    D = _field_one(edt, RES, (0.0, 0.0), np.array([4.0, 1.0]), 0.0)
    r = _along(D)
    assert np.isfinite(r).all()
    d = np.diff(r)
    assert d[d > 0].sum() < 0.5                              # forwards the distance left falls (running wide into the four corners: 0.33 m)
    assert 26.0 < r[0] < 34.0                                # a whole lap just past the line
    # one cell of clearance (the old field) cut through the gap: the top corridor read the way back as shorter
    r0 = _along(_field_one(edt, RES, (0.0, 0.0), np.array([4.0, 1.0]), 0.0, clear=0.0))
    d0 = np.diff(r0)
    assert d0[(d0 > 0) & (d0 < 5.0)].sum() > 2.0


def test_cells_near_the_walls_read_their_own_side():
    edt = _ring_with_slit(0.3)
    D = _field_one(edt, RES, (0.0, 0.0), np.array([4.0, 1.0]), 0.0)
    free = edt > RES
    assert np.isfinite(D[free]).mean() > 0.999               # every free cell has a value
    i_top, i_bot, j = int(6.15 / RES), int(1.85 / RES), int(6.0 / RES)   # hugging the island, above and below it
    assert abs(D[i_top, j] - D[int(7.0 / RES), j]) < 1.0     # the top wall strip reads the top corridor
    assert abs(D[i_bot, j] - D[int(1.0 / RES), j]) < 1.0
