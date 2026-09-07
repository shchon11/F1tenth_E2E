import numpy as np
import torch
from f1sim import maps
from f1sim.track import TrackTensors


def test_rev_and_mir_modifiers_share_grids():
    a = maps.load("gen:competition:7")
    r = maps.load("gen:competition:7~rev")
    m = maps.load("gen:competition:7~mir")
    mr = maps.load("gen:competition:7~mir~rev")
    assert a.edt is r.edt and np.allclose(a.centerline, r.centerline[::-1])
    assert m.occupancy.shape == a.occupancy.shape and not np.array_equal(m.occupancy, a.occupancy)
    assert np.allclose(m.centerline, mr.centerline[::-1])
    tt = TrackTensors([a, r, m, mr], "cpu")
    assert tt.T == 4 and tt.edt.numel() == 2 * a.occupancy.size           # two unique grids
    assert int(tt.t_off[0]) == int(tt.t_off[1]) and int(tt.t_off[2]) == int(tt.t_off[3])
    assert torch.allclose(tt.length[0], tt.length[1], rtol=1e-3)
    # lap direction flips: signed area of the centerline
    def area(c): return 0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
    assert area(a.centerline) * area(r.centerline) < 0 and area(a.centerline) * area(m.centerline) < 0


def test_real_map_crop_keeps_lane():
    t = maps.load("real:icra2022")
    H, W = t.occupancy.shape
    assert W * t.resolution < 30 and H * t.resolution < 25          # cropped from the SLAM canvas
    from scipy import ndimage
    rc = np.stack([(t.centerline[:, 1] - t.origin[1]) / t.resolution, (t.centerline[:, 0] - t.origin[0]) / t.resolution])
    d = ndimage.map_coordinates(t.edt, rc, order=1)
    assert d.min() > 0.4


def test_pockets_are_walled_dead_ends_off_the_lane():
    """+pk<seed>: free cells are only added (never removed from the lane), every added free cell is a
    dead end (the lane stays one free component), the raceline is the base track's, and the base follows
    through ~mir / ~rev."""
    import numpy as np
    from scipy import ndimage
    from f1sim import maps
    from f1sim.raceline import Raceline
    b = maps.load("real:blackbox2021_1"); t = maps.load("real:blackbox2021_1+pk0")
    assert not (b.occupancy == False).sum() > (~t.occupancy & ~b.occupancy).sum()      # no lane cell removed
    carved = ~t.occupancy & b.occupancy
    assert carved.sum() * t.resolution ** 2 > 2.0                                     # pockets exist
    lab, n = ndimage.label(~t.occupancy); c0 = int(round((t.centerline[0, 0] - t.origin[0]) / t.resolution)); r0 = int(round((t.centerline[0, 1] - t.origin[1]) / t.resolution))
    lab_b, _ = ndimage.label(~b.occupancy)
    lane_b = lab_b == lab_b[r0, c0]; lane_t = lab == lab[r0, c0]
    assert (lane_t & ~lane_b & ~carved).sum() == 0                                    # the lane grew only by the pockets
    assert np.allclose(Raceline.build_cached(t).xy, Raceline.build_cached(b).xy)       # raceline ignores pockets
    m = maps.load("real:blackbox2021_1+pk0~mir~rev")
    assert m.base is not None and np.allclose(Raceline.build_cached(m).xy, Raceline.build_cached(maps.load("real:blackbox2021_1~mir~rev")).xy)
