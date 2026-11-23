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


def test_hose_variants_move_hoses_without_second_walls():
    """(11-23) `+hose<k>`: the lane's boundary hoses moved in smooth sections, drawn again as one band -- the lane stays one
    ring at least 0.55 m a side at the centerline, its size within a few per cent, and no duct cell lies more than the
    band's thickness away from the lane (a hose left behind = the second wall the old warps had); `+duct` keeps the lane"""
    import numpy as np
    from scipy import ndimage
    from f1sim import maps
    base = maps.load("real:korea26_scan")
    relaid = maps.load("real:korea26_scan+duct")
    lane0 = ~base.occupancy
    assert abs(int((~relaid.occupancy).sum()) - int(lane0.sum())) / lane0.sum() < 0.01
    for k in (0, 5, 11):
        t = maps.load(f"real:korea26_scan+hose{k}")
        lane = ~t.occupancy
        lab, n = ndimage.label(lane)
        assert n == 1, (k, n)
        assert abs(int(lane.sum()) - int(lane0.sum())) / lane0.sum() < 0.05
        far = ndimage.distance_transform_edt(~lane) * t.resolution
        assert float(far[t.duct].max()) <= 0.10 + 2 * t.resolution, (k, float(far[t.duct].max()))
        cl = t.centerline; o = t.origin; r = t.resolution
        ci = ((cl[:, 1] - o[1]) / r).astype(int); cj = ((cl[:, 0] - o[0]) / r).astype(int)
        assert float(t.edt[ci, cj].min()) >= 0.5, (k, float(t.edt[ci, cj].min()))


def test_real_size_boxes_appended_after_the_old_ids():
    """(11-23) the measured box sizes are a new group at the end: every older shape id keeps its meaning"""
    from f1sim.procedural_obstacles import build_catalogue, catalogue_k_pad, BOX_SHAPES, REAL_BOX_SHAPES
    b, r = [], []
    shapes, _, _ = build_catalogue(catalogue_k_pad(), b, r)
    assert r == list(range(len(shapes) - len(REAL_BOX_SHAPES), len(shapes)))
    assert b[-1] < r[0] and len(b) == len(BOX_SHAPES)
    assert all(0.18 <= dict(shapes[i].dims)["depth"] <= dict(shapes[i].dims)["width"] <= 0.36 for i in r)
