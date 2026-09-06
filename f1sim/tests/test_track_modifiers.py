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
