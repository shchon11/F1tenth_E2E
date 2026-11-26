"""How a state decoder's walls and boxes are drawn (11-26, f1sim/learn/dec_shapes.py): their edges where the probability
crosses one half, as strong as the decoder is sure, fading out before the grid's border (no frame, no fill), dashed."""
import math

import numpy as np

from f1sim.learn import dec_shapes as DS

GRID = [-3.0, 9.0, -6.0, 6.0, 0.25]


def cells():
    x0, x1, y0, y1, r = GRID
    xc = x0 + (np.arange(int((x1 - x0) / r)) + 0.5) * r
    yc = y0 + (np.arange(int((y1 - y0) / r)) + 0.5) * r
    return np.meshgrid(xc, yc, indexing="ij")


def sig(v):
    return 1.0 / (1.0 + np.exp(-v))


def test_edge_sits_where_the_probability_crosses_half():
    X, Y = cells()
    p = sig((2.0 - np.hypot(X - 3.0, Y)) / 0.1)          # a disc of 2 m about (3, 0), sharp
    segs, st, keys = DS.edges(p, GRID)
    r = np.hypot(segs[..., 0] - 3.0, segs[..., 1])
    assert len(segs) > 30 and np.abs(r - 2.0).max() < 0.02
    lines = DS.chains(segs, st, keys)
    assert len(lines) == 1                               # one closed loop
    pts, w = lines[0]
    assert np.allclose(pts[0], pts[-1]) and len(pts) == len(segs) + 1 and w.shape == (len(pts),)
    assert w.min() > 0.6                                 # far from the border and crisp: strong


def test_unsure_edges_and_the_border_fade_out():
    X, Y = cells()
    mush = sig((Y - 3.0) / 2.0) * 0.2 + 0.4              # 0.4 .. 0.6: crosses one half but means little
    _, st, _ = DS.edges(mush, GRID)
    assert len(st) and st.max() < 0.05
    wall = sig((Y - 3.0) / 0.05)                          # a crisp wall along the whole grid
    segs, st, _ = DS.edges(wall, GRID)
    mid = segs.mean(1)
    assert st[np.abs(mid[:, 0] - 3.0) < 1.0].min() > 0.9                         # the middle: full strength
    assert st[(mid[:, 0] < -2.7) | (mid[:, 0] > 8.7)].max() < 1e-6              # at the grid's ends: gone
    o = np.argsort(mid[:, 0]); x_ = mid[o, 0]; s_ = st[o]
    assert np.all(np.diff(s_[x_ < 0.0]) >= -1e-6) and np.all(np.diff(s_[x_ > 6.0]) <= 1e-6)   # and fading toward them
    allwall = np.ones_like(wall)                         # a grid full of wall has no edge at all: nothing outlines the grid
    assert len(DS.edges(allwall, GRID)[0]) == 0


def test_dashes_in_the_world():
    X, Y = cells()
    wall = sig((Y - 3.0) / 0.05)
    d, w = DS.dashed_edges(wall, GRID, (10.0, 5.0, math.pi / 2), on=0.14, off=0.09)
    assert d.shape[1:] == (2, 2) and len(d) == len(w) and w.min() >= 0.06
    # the wall at y = 3 in the car's frame, the car at (10, 5) facing +y: world x = 10 - 3 = 7
    assert np.abs(d[..., 0] - 7.0).max() < 0.02
    drawn = np.hypot(*(d[:, 1] - d[:, 0]).T).sum()
    span = d[..., 1].max() - d[..., 1].min()
    assert 0.55 < drawn / span < 0.7                     # on / (on + off) = 0.61 of the line is drawn


def test_outline_and_ring():
    o = DS.outline(1.0, 2.0, 0.0, 0.6, 0.3)
    assert o.shape == (5, 2) and np.allclose(o[0], o[-1]) and np.allclose(o[0], [1.3, 2.15])
    r = DS.ring(0.0, 0.0, 0.5)
    assert np.allclose(np.hypot(r[:, 0], r[:, 1]), 0.5, atol=1e-6)
