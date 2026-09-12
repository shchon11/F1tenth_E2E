"""`+hard<seed>`: hand-built-style obstacle patterns (gate / diagonal / chicane / apex / cluster)."""
import numpy as np
import pytest
from scipy import ndimage

from f1sim import maps
from f1sim.hard_obstacles import ERODE_M, with_hard_obstacles


@pytest.mark.parametrize("base", ["gen:control:1400", "real:blackbox2022_1"])
def test_patterns_are_placed_and_the_lap_stays_drivable(base):
    t0 = maps.load(base)
    t = with_hard_obstacles(t0, seed=7)
    kinds = [k for _, k in t.hard_patterns]
    assert len(kinds) >= 3 and len(set(kinds)) >= 3, kinds
    added = t.occupancy & ~t0.occupancy
    assert added.sum() > 0 and (t.tall & ~t0.tall).sum() >= added.sum()   # boxes are tall: LiDAR sees them
    # proof: erode by the car's half width + margin and the lap is still one loop
    free = ndimage.binary_erosion(~t.occupancy, iterations=int(round(ERODE_M / t.resolution)), border_value=0)
    lab, _ = ndimage.label(free)
    cl = t.centerline; N = len(cl)
    comps = []
    for i in (0, N // 4, N // 2, 3 * N // 4):
        p = cl[i]; best = 0
        for d in np.linspace(0, 1.2, 13):
            for s in (0.0, 1.0, -1.0):
                tang = cl[(i + 1) % N] - cl[i - 1]; tang /= np.linalg.norm(tang) + 1e-9
                q = p + s * d * np.array([-tang[1], tang[0]])
                jj = int((q[0] - t.origin[0]) / t.resolution); ii = int((q[1] - t.origin[1]) / t.resolution)
                if 0 <= ii < lab.shape[0] and 0 <= jj < lab.shape[1] and lab[ii, jj] > 0:
                    best = lab[ii, jj]; break
            if best: break
        comps.append(int(best))
    assert all(c > 0 for c in comps) and len(set(comps)) == 1, comps


def test_seed_is_the_layout_and_the_catalog_name_works():
    a = maps.load("gen:control:1401+hard3"); b = maps.load("gen:control:1401+hard3"); c = maps.load("gen:control:1401+hard4")
    assert a.name.endswith("_hard3") and np.array_equal(a.occupancy, b.occupancy)
    assert not np.array_equal(a.occupancy, c.occupancy)
    r = maps.load("gen:control:1401+hard3~rev")
    assert r.occupancy.sum() == a.occupancy.sum()


def test_small_objects_are_part_of_the_mix():
    """The user's point: avoiding only big boxes is not avoiding. Over a few seeds, some added
    obstacle components must be small (under 0.25 m across) and scatter patterns must occur."""
    t0 = maps.load("real:blackbox2022_1")
    small = 0; kinds = set()
    for seed in range(1, 5):
        t = with_hard_obstacles(t0, seed=seed)
        kinds |= {k for _, k in t.hard_patterns}
        lab, n = ndimage.label(t.occupancy & ~t0.occupancy)
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        small += int(np.sum(np.asarray(sizes) <= (0.25 / t.resolution) ** 2))
    assert "scatter" in kinds and small >= 4, (kinds, small)
