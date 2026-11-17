"""Two properties of the duct tube mesh that were silently wrong.

Both are checked on a plain analytic circle rather than on a map: a circle has no staircase and no
folds, so a failure here belongs to the mesh generator and not to the contour it was given. The
visual A/B lives in `work/claude-duct-fix`; these are the invariants worth keeping in the repo.

`duct_mesh_arrays` smooths what it is handed, so the ring count is not the input point count -- it
is read back from the array instead.
"""
from __future__ import annotations

import numpy as np

from f1sim.viewer.gl_scene import duct_mesh_arrays

RADIAL = 14          # duct_mesh_arrays' default ring resolution


def _circle(n=64, r=4.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def _tube(pts):
    pos, nrm, _, idx = duct_mesh_arrays([pts], 0.33, occupied=None, resolution=0.05, radial=RADIAL)
    return pos, nrm, idx, pos.shape[0] // RADIAL


def _axis_radius(pts):
    pos, _, _, rings = _tube(pts)
    return np.linalg.norm(pos.reshape(rings, RADIAL, 3).mean(1)[:, :2], axis=1)


def test_tube_triangles_are_wound_outward():
    """Faces must agree with the normals they ship, or backface culling hides the wrong side.

    `Mesh.cull` is on and the main pass culls back faces, so an inward winding discards exactly the
    surface pointing at the camera and leaves the inside of the far wall showing; the shadow pass
    culls front faces, so it shadow-maps the outward surface at the same time and the two passes
    disagree about which side the tube has. Before the fix every triangle disagreed with its normal
    -- mean dot -0.997 on this circle.

    A handful at the seam, where the closed loop wraps back onto its first ring, still come out
    negative. That is bounded and is asserted as such rather than rounded to "all".
    """
    pos, nrm, idx, _ = _tube(_circle())
    tri = idx.reshape(-1, 3)
    v0, v1, v2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    face = np.cross(v1 - v0, v2 - v0)
    face /= np.linalg.norm(face, axis=1, keepdims=True) + 1e-12
    supplied = nrm[tri].mean(1)
    supplied /= np.linalg.norm(supplied, axis=1, keepdims=True) + 1e-12
    dot = np.einsum("ij,ij->i", face, supplied)
    assert dot.mean() > 0.9, f"mean face/normal agreement {dot.mean():.3f}; the tube is inside out"
    assert float(np.mean(dot > 0)) > 0.99, \
        f"{int((dot <= 0).sum())} of {len(dot)} triangles wound against their normal"


def test_tangent_is_read_from_arc_length_not_from_vertex_spacing():
    """Vertex spacing is the simplifier's business; the tangent has to describe the wall.

    The axis is the contour offset by R, and offsetting scales the local step by (1 - kR): wherever
    the estimated curvature k exceeds 1/R the offset curve reverses and the tube folds through
    itself. A tangent taken from immediate neighbours makes k a function of however the vertices
    happened to be spaced, which on Korea's duct is about 16 mm -- a tenth of R -- so it read
    sampling noise as curvature.

    The helper is exercised directly rather than through `duct_mesh_arrays`, because the smoothing
    in front of it re-simplifies a clean circle down to a handful of points and both estimators then
    agree. That agreement would look like a passing test while proving nothing.
    """
    from f1sim.viewer.gl_scene import _tangent_arclength

    # a circle sampled at wildly uneven angles: the analytic tangent is exactly known everywhere
    rng = np.random.default_rng(0)
    a = np.sort(rng.uniform(0, 2 * np.pi, 400))
    pts = np.stack([4.0 * np.cos(a), 4.0 * np.sin(a)], 1)
    truth = np.stack([-np.sin(a), np.cos(a)], 1)          # CCW unit tangent

    arclen = _tangent_arclength(pts, 0.165)                # R for a 0.33 m hose
    neighbour = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    neighbour /= np.linalg.norm(neighbour, axis=1, keepdims=True) + 1e-9

    err_arclen = float(np.abs(np.einsum("ij,ij->i", arclen, truth) - 1.0).max())
    err_neighbour = float(np.abs(np.einsum("ij,ij->i", neighbour, truth) - 1.0).max())
    assert err_arclen < 0.02, f"arc-length tangent is off the true tangent by {err_arclen:.4f}"
    assert err_arclen < err_neighbour / 5, (
        f"arc-length tangent ({err_arclen:.4f}) is not clearly better than reading neighbours "
        f"({err_neighbour:.4f}); uneven spacing is still leaking into the estimate")


# ---------------------------------------------------------------- the lane side (2026-11-17)
# A map built from scans marks a hose's face a few cells deep; the space behind it is solid but unseen.
# Both sides of that thin band were contoured and each got a tube, the back one standing in the lane
# (bag-scan Korea 2026: 43 % of the hose length more than 20 cm into the lane). Built on a grid here,
# through `build_track_geometry`, because the lane is a property of the map, not of a contour.

RES = 0.05


class _Grid:
    """The fields `build_track_geometry` reads off a Track."""

    def __init__(self, occupancy, duct):
        self.occupancy, self.duct = occupancy, duct
        self.tall = np.zeros_like(duct)
        self.shape, self.resolution, self.origin = duct.shape, RES, (0.0, 0.0)
        self.duct_height, self.centerline, self.name, self.props = 0.33, None, "grid", []


def _ducts(occupancy, duct):
    from f1sim.viewer.geometry import build_track_geometry
    g = build_track_geometry(_Grid(occupancy, duct))
    return np.asarray(g["ducts"][0], np.float64)


def _rings_r(pos):
    """radius from the grid centre of every vertex, and of every ring's centre (the axis)"""
    r_v = np.hypot(pos[:, 0] - 4.0, pos[:, 1] - 4.0)
    c = pos.reshape(-1, RADIAL, 3).mean(1)
    return r_v, np.hypot(c[:, 0] - 4.0, c[:, 1] - 4.0)


def _disk(r_in, r_out):
    """cells (8 x 8 m grid) whose centre lies between r_in and r_out of (4, 4)"""
    yy, xx = np.mgrid[0:160, 0:160]
    r = np.hypot((xx + 0.5) * RES - 4.0, (yy + 0.5) * RES - 4.0)
    return (r >= r_in) & (r < r_out)


def test_a_thin_scanned_band_draws_one_hose_and_none_in_the_lane():
    """Lane inside r < 2 m, the scanned face a 2-cell band, unseen solid beyond: no tube vertex stands
    in the lane, and the axis is a full radius behind the face -- one hose, not one per band side."""
    occ = _disk(2.0, 99.0)
    duct = _disk(2.0, 2.1)
    pos = _ducts(occ, duct)
    assert pos is not None
    r_v, r_axis = _rings_r(pos)
    assert r_v.min() > 2.0 - 0.06, r_v.min()                 # the cell edge (2.5 cm) + smoothing; the old back tube stood 23 cm in
    # every ring a hose radius behind the face, within the circle's pixel staircase -- the old second tube's
    # axis sat at 1.9-2.0 m, on the lane side of the face
    assert np.abs(r_axis - (2.0 + 0.165)).max() < 0.07, (r_axis.min(), r_axis.max())


def test_a_divider_thinner_than_the_hose_is_drawn_as_wide_as_it_is():
    """Lane on both sides of a 3-cell ring: the hose stands in neither lane and keeps its height."""
    duct = _disk(1.5, 1.65)
    pos = _ducts(duct.copy(), duct)
    assert pos is not None
    r_v, _ = _rings_r(pos)
    assert r_v.min() > 1.5 - 0.06 and r_v.max() < 1.65 + 0.06, (r_v.min(), r_v.max())
    assert pos[:, 2].max() > 0.30
