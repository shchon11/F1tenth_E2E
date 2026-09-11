"""Blocking-obstacle geometry: the three proofs, on synthetic tracks with known answers."""
from __future__ import annotations
import importlib
import numpy as np
import pytest

pytest.importorskip("scipy")


@pytest.fixture
def obs(bench):
    return importlib.import_module("f1sim.learn.benchmark.obstacle")


@pytest.fixture
def Track():
    return pytest.importorskip("f1sim.track").Track


def straight_corridor(width_m=3.0, length_m=30.0, res=0.05):
    """A straight free lane with walls, plus its centreline. Widths are exact by construction."""
    H = int(round((width_m + 2.0) / res))
    W = int(round(length_m / res))
    occ = np.zeros((H, W), bool)
    half = int(round((width_m / 2) / res))
    mid = H // 2
    occ[: mid - half, :] = True
    occ[mid + half:, :] = True
    ys = (mid) * res
    cl = np.stack([np.arange(W) * res, np.full(W, ys)], 1)
    return occ, cl, res


def make(Track, width_m=3.0):
    occ, cl, res = straight_corridor(width_m)
    return Track.from_occupancy(occ, res, (0.0, 0.0), cl, "synthetic")


def test_placement_blocks_the_line_and_leaves_a_corridor(obs, Track):
    t = make(Track)
    new, place = obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)
    # the line is obstructed
    proofs = obs.prove(new, t, place, vehicle_length=0.58, vehicle_width=0.31)
    # the corridor is measured inside the lane, so it is the real gap and not the space outside
    assert proofs["corridor"]["widths_m"]["left"] <= 3.0
    assert proofs["blocking"]["blocked_samples"] > 0
    # and a real corridor survives on exactly one side
    assert proofs["corridor"]["passable_sides"]
    assert proofs["corridor"]["widths_m"][proofs["corridor"]["passable_sides"][0]] >= 0.51
    assert proofs["connectivity"]["component"] > 0


def test_obstacle_actually_appears_in_occupancy(obs, Track):
    t = make(Track)
    new, place = obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)
    added = int((new.occupancy & ~t.occupancy).sum())
    assert added == place.n_cells > 0


def test_wall_hugging_placement_is_rejected(obs, Track):
    """The stock `with_lane_obstacles` behaviour: a box against the wall leaves the line clear."""
    t = make(Track)
    res, (ox, oy) = t.resolution, t.origin
    occ = t.occupancy.copy()
    H, W = occ.shape
    mid = H // 2
    # a box hugging the upper wall, never touching the centreline row
    r_lo = mid - int(round(1.4 / res))
    r_hi = mid - int(round(0.6 / res))
    c_lo, c_hi = int(round(14.8 / res)), int(round(15.2 / res))
    occ[r_lo:r_hi, c_lo:c_hi] = True
    hugged = type(t).from_occupancy(occ, res, (ox, oy), t.centerline, "hugged")
    place = obs.Placement(s_obs_m=15.0, centre_xy=(15.0, 1.0), size_m=(0.4, 0.8), side=1,
                          corridor_width_m=0.51, required_corridor_m=0.51,
                          blocked_span_m=0.4, n_cells=1)
    with pytest.raises(obs.GeometryError, match="does not block the line"):
        obs.prove_blocking(hugged, t, place, vehicle_length=0.58)


def test_impassable_obstacle_is_rejected(obs, Track):
    """A box spanning the whole lane has no corridor; it must not become a scenario."""
    t = make(Track)
    res = t.resolution
    occ = t.occupancy.copy()
    c_lo, c_hi = int(round(14.8 / res)), int(round(15.2 / res))
    occ[:, c_lo:c_hi] = True                     # wall to wall
    blocked = type(t).from_occupancy(occ, res, t.origin, t.centerline, "blocked")
    place = obs.Placement(s_obs_m=15.0, centre_xy=(15.0, 1.5), size_m=(0.4, 3.0), side=1,
                          corridor_width_m=0.51, required_corridor_m=0.51,
                          blocked_span_m=0.4, n_cells=1)
    with pytest.raises(obs.GeometryError, match="not passable"):
        obs.prove_corridor(blocked, place, vehicle_width=0.31, base_track=t)


def test_narrow_lane_cannot_host_an_obstacle(obs, Track):
    """0.8 m of lane cannot hold a box and still leave 0.51 m: refuse rather than place."""
    t = make(Track, width_m=0.8)
    with pytest.raises(obs.GeometryError, match="cannot hold an obstacle"):
        obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)


def test_pocket_is_rejected_by_connectivity(obs, Track):
    """A gap beside the obstacle that dead-ends is not an alternative route."""
    t = make(Track)
    new, place = obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)
    res = new.resolution
    occ = new.occupancy.copy()
    # seal the corridor just downstream, turning it into a pocket
    c_lo, c_hi = int(round(15.6 / res)), int(round(16.0 / res))
    occ[:, c_lo:c_hi] = True
    pocket = type(new).from_occupancy(occ, res, new.origin, new.centerline, "pocket")
    with pytest.raises(obs.GeometryError, match="pocket|narrower than the car"):
        obs.prove_connectivity(pocket, place, vehicle_length=0.58, vehicle_width=0.31)


def test_connectivity_erodes_by_the_footprint_not_a_point(obs, Track):
    """A 0.25 m slit passes a point but not a 0.31 m car; erosion must catch it."""
    t = make(Track)
    new, place = obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)
    res = new.resolution
    occ = new.occupancy.copy()
    H = occ.shape[0]
    mid = H // 2
    c_lo, c_hi = int(round(15.6 / res)), int(round(16.0 / res))
    occ[:, c_lo:c_hi] = True
    slit = int(round(0.125 / res))               # a 0.25 m opening, narrower than the body
    occ[mid - slit: mid + slit, c_lo:c_hi] = False
    narrow = type(new).from_occupancy(occ, res, new.origin, new.centerline, "slit")
    with pytest.raises(obs.GeometryError):
        obs.prove_connectivity(narrow, place, vehicle_length=0.58, vehicle_width=0.31)


def test_blocking_sweep_includes_the_rear_box(obs):
    """The sweep window is vehicle_length + rear box, matching what `_car_contacts` tests."""
    gm = importlib.import_module("f1sim.learn.benchmark.geom")
    assert gm.REAR_BOX_MAX_M == 0.11


def test_placement_is_deterministic(obs, Track):
    t = make(Track)
    a, pa = obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)
    b, pb = obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)
    assert pa.as_dict() == pb.as_dict()
    assert np.array_equal(a.occupancy, b.occupancy)


def test_placement_leaves_rasterisation_slack(obs, Track):
    """Sizing to leave exactly `required` lands short once cells are snapped.

    Measured on gen:control:1400: 0.45 m against a 0.51 m requirement. The constructor now designs
    for required + 2 cells, so the proof it runs on itself passes.
    """
    t = make(Track, width_m=1.6)                 # narrow, where the slack matters
    new, place = obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)
    got = obs.prove_corridor(new, place, vehicle_width=0.31, base_track=t)
    assert got["passable_sides"]
    assert max(got["widths_m"].values()) >= 0.51 - got["tolerance_m"]


def test_corridor_is_measured_inside_the_lane_not_outside(obs, Track):
    """Real track walls are ~1 cell thick with open ground beyond; the corridor must not report it.

    This reproduces what happened on gen:control:1400: an unbounded scan walked through the thin
    wall and returned metres of off-track space, so every map looked passable.
    """
    res = 0.05
    lane_m, outside_m = 3.0, 4.0           # generous ground beyond the wall, as a real map has
    H = int(round((lane_m + 2 * outside_m) / res))
    W = int(round(30.0 / res))
    occ = np.zeros((H, W), bool)
    mid, half = H // 2, int(round((lane_m / 2) / res))
    occ[mid - half, :] = True              # one-cell walls, free on both sides of them
    occ[mid + half, :] = True
    cl = np.stack([np.arange(W) * res, np.full(W, mid * res)], 1)
    t = Track.from_occupancy(occ, res, (0.0, 0.0), cl, "thinwall")

    new, place = obs.place_blocking_obstacle(t, 15.0, vehicle_length=0.58, vehicle_width=0.31)
    bounded = obs.prove_corridor(new, place, vehicle_width=0.31, base_track=t)
    unbounded = obs.prove_corridor(new, place, vehicle_width=0.31, base_track=None)

    assert max(bounded["widths_m"].values()) <= lane_m, "a bounded corridor cannot exceed the lane"
    assert max(unbounded["widths_m"].values()) > max(bounded["widths_m"].values()), \
        "the unbounded scan should escape through the thin wall and report the ground outside"
    assert max(unbounded["widths_m"].values()) > lane_m


def test_feasible_search_is_deterministic_and_reports_rejections(obs, Track):
    t = make(Track)
    a = obs.find_feasible_s(t, vehicle_length=0.58, vehicle_width=0.31, start_m=10.0)
    b = obs.find_feasible_s(t, vehicle_length=0.58, vehicle_width=0.31, start_m=10.0)
    assert a[2]["s_obs_m"] == b[2]["s_obs_m"]
    assert isinstance(a[2]["rejected"], list)


def test_feasible_search_gives_up_rather_than_weakening_margins(obs, Track):
    t = make(Track, width_m=0.8)                 # no position can host the scenario
    with pytest.raises(obs.GeometryError, match="drop it rather than weaken the margins"):
        obs.find_feasible_s(t, vehicle_length=0.58, vehicle_width=0.31, max_tries=5)


def annular_track(Track, res=0.05, r_in=4.0, r_out=5.6):
    """A closed ring: every point reaches every other the long way round.

    This is the shape that broke a global connected-component test. The corridor can be sealed
    downstream and the flood fill still joins approach to exit by going the other way around.
    """
    R = r_out + 1.0
    n = int(round(2 * R / res))
    ys, xs = np.mgrid[0:n, 0:n]
    cx = cy = n / 2.0
    d = np.hypot((xs - cx) * res, (ys - cy) * res)
    occ = ~((d >= r_in) & (d <= r_out))          # free only inside the annulus
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    rm = (r_in + r_out) / 2
    cl = np.stack([cx * res + rm * np.cos(th), cy * res + rm * np.sin(th)], 1)
    return Track.from_occupancy(occ, res, (0.0, 0.0), cl, "annulus")


def test_annular_backway_is_not_accepted_as_connectivity(obs, Track):
    """HIGH1 regression: seal the corridor downstream on a ring.

    A global component test passes this -- approach and exit are connected the long way round -- and
    reports success on a scenario no car can drive through. The local window must refuse it.
    """
    t = annular_track(Track)
    new, place = obs.place_blocking_obstacle(t, 6.0, vehicle_length=0.58, vehicle_width=0.31)
    res = new.resolution
    occ = new.occupancy.copy()

    # wall off the lane a short way downstream, across the full annulus width
    cl = new.centerline
    i = obs._index_at_arc(cl, place.s_obs_m + 2.0)
    c, _, nrm = obs._frame_at(cl, i)
    for tt in np.arange(-2.0, 2.0, res / 2):
        for along in np.arange(-0.3, 0.3, res / 2):
            tang = np.array([-nrm[1], nrm[0]])
            p = c + nrm * tt + tang * along
            r = int(round((p[1] - new.origin[1]) / res))
            cc = int(round((p[0] - new.origin[0]) / res))
            if 0 <= r < occ.shape[0] and 0 <= cc < occ.shape[1]:
                occ[r, cc] = True
    sealed = type(new).from_occupancy(occ, res, new.origin, cl, "sealed")

    # the global test would still see one component; the local one must not
    with pytest.raises(obs.GeometryError, match="pocket|narrower than the car"):
        obs.prove_connectivity(sealed, place, vehicle_length=0.58, vehicle_width=0.31)


def test_local_connectivity_still_accepts_a_real_ring_scenario(obs, Track):
    """The fix must not reject a genuinely drivable ring."""
    t = annular_track(Track)
    new, place = obs.place_blocking_obstacle(t, 6.0, vehicle_length=0.58, vehicle_width=0.31)
    got = obs.prove_connectivity(new, place, vehicle_length=0.58, vehicle_width=0.31)
    assert got["component"] > 0 and got["window_m"] == 8.0
