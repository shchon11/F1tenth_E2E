"""plan_rl --plan-car-cost's geometry (`learn.plan_car_cost`): time-aligned 3-disc footprints, the knowable / reset / rollout-end
cut-offs and the env's fault rule read just before the first overlap. Env-free (numpy + torch on the CPU)."""
import math

import numpy as np
import pytest
import torch

from f1sim.learn import plan_car_cost as C

L, W = 0.58, 0.31            # the car (exam / env cfg vehicle)
K = 10
DT = 0.1
LAP = 1000.0                 # a long straight: distance to go = 500 - x


def scene(T, plan, opp, brk=None, know=None, ok=None):
    """numpy arrays for `plan_car_cost`: `plan(t, k)` -> (x, y, psi, speed) our car (k = 0) / plan point k at decision t;
    `opp(tau)` -> (x, y, psi, speed) the other car at time tau [s]"""
    me = np.array([[plan(t, k) for k in range(K + 1)] for t in range(T)], np.float64)[:, None]     # (T, 1, K+1, 4)
    op = np.array([opp(t * DT) for t in range(T + 1)], np.float64)[:, None]                        # (T+1, 1, 4)
    vel = lambda a: np.stack([a[..., 3] * np.cos(a[..., 2]), a[..., 3] * np.sin(a[..., 2])], -1)
    return dict(me_xy=me[..., :2], me_psi=me[..., 2], me_v=vel(me), me_rem=500.0 - me[..., 0],
                op_xy=op[..., :2], op_psi=op[..., 2], op_v=vel(op), op_rem=500.0 - op[..., 0],
                op_ok=np.ones((T + 1, 1), bool) if ok is None else ok, brk=np.zeros((T, 1), bool) if brk is None else brk,
                know=np.ones((T, 1), bool) if know is None else know, lap=np.full((T, 1), LAP))


def run(sc, xp=np, margin=0.05):
    if xp is torch:
        sc = {k: torch.as_tensor(v) for k, v in sc.items()}
    out = C.plan_car_cost(xp, **sc, kofs=list(range(1, K + 1)), w=C.weights(K, 0.3), length=L, width=W, margin=margin)
    return {k: np.asarray(v) for k, v in out.items()}


def straight(x0, v, y=0.0):
    """a car at x0 + v t on y, heading +x"""
    return lambda tau: (x0 + v * tau, y, 0.0, v)


def planned(x0, v, y=0.0):
    """our car at x0 + v t; each decision's plan continues at v"""
    return lambda t, k: (x0 + v * (t + k) * DT, y, 0.0, v)


def test_geometry_numbers():
    r = C.disc_radius(L, W)
    assert abs(r - math.sqrt(0.155 ** 2 + (0.58 / 6) ** 2)) < 1e-12 and 0.18 < r < 0.19
    w = C.weights(10, 0.3)
    assert abs(w[0] - 1.0) < 1e-12 and abs(w[-1] - 0.3) < 1e-12 and all(a > b for a, b in zip(w, w[1:]))


@pytest.mark.parametrize("xp", [np, torch])
def test_aligned_rear_end_half_metre_overlaps(xp):
    # the other car 0.5 m ahead (centres), same line, same speed: the plan sits in it from the first point -- charged
    out = run(scene(30, planned(0.0, 4.0), straight(0.5, 4.0)), xp)
    assert out["over"][0, 0] and out["charged"][0, 0] and out["first"][0, 0] == 1
    assert abs(out["price"][0, 0] - 1.0 * (C.CONTACT_S + 0.0)) < 1e-9          # w_1 = 1, no relative speed
    A = (lambda v: np.asarray(v, np.float64)) if xp is np else (lambda v: torch.as_tensor(v, dtype=torch.float64))
    assert bool(C.overlap(xp, A([[0.0, 0.0]]), A([0.0]), A([[0.5, 0.0]]), A([0.0]), L, W, 0.05)[0])


@pytest.mark.parametrize("xp", [np, torch])
def test_side_by_side_tight_pass_not_charged(xp):
    # side by side, centres 0.35 m apart (4 cm of daylight between the bodies), same speed: no overlap with margin 0.05
    out = run(scene(30, planned(0.0, 4.0), straight(0.0, 4.0, y=0.35)), xp)
    assert not out["over"].any() and not out["charged"].any() and (out["price"] == 0).all()
    # 0.30 m apart the bodies are in each other: an overlap (the margin is not a free pass for a real touch)
    out = run(scene(30, planned(0.0, 4.0), straight(0.0, 4.0, y=0.30)), xp)
    assert out["over"][0, 0]


@pytest.mark.parametrize("xp", [np, torch])
def test_time_misalignment_no_overlap(xp):
    # the same path, but the other car 0.5 s ahead on it (2 m at 4 m/s): every plan point lies on its path, never at the
    # same time -- no overlap. Compared without time it would be one (shown below)
    sc = scene(30, planned(0.0, 4.0), straight(4.0 * 0.5, 4.0))
    out = run(sc, xp)
    assert not out["over"].any() and (out["price"] == 0).all()
    path_hit = any(C.overlap(np, sc["me_xy"][0, 0, k], sc["me_psi"][0, 0, k], np.array([4.0 * tau, 0.0]), np.float64(0.0), L, W, 0.05)
                   for k in range(1, K + 1) for tau in np.arange(0.5, 2.0, 0.025))
    assert path_hit


@pytest.mark.parametrize("xp", [np, torch])
def test_fault_from_behind_not_charged(xp):
    # we stand (plan stays put); the other car comes from behind at 3 m/s and its true path runs into ours: an overlap,
    # but the car was behind us (more than a car length) just before it -> no cost
    out = run(scene(30, planned(0.0, 0.0), straight(-1.5, 3.0)), xp)
    assert out["over"][0, 0] and not out["charged"][0, 0] and out["price"][0, 0] == 0.0
    # (read AT the first overlapping point, -0.6 m, and the next one, -0.3 m, it would have counted as alongside)
    assert out["first"][0, 0] == 3
    # mirrored: we close at 3 m/s on a car standing ahead -> charged, priced at the relative speed
    out = run(scene(30, planned(-1.5, 3.0), straight(0.0, 0.0)), xp)
    k = int(out["first"][0, 0])
    assert out["charged"][0, 0] and k == 3
    assert abs(out["price"][0, 0] - C.weights(K, 0.3)[k - 1] * (C.CONTACT_S + C.CONTACT_S_PER_MPS * 3.0)) < 1e-9


@pytest.mark.parametrize("xp", [np, torch])
def test_cutoffs_reset_knowable_rollout_end(xp):
    # we close at 4 m/s on a car standing 2 m ahead
    pl, op = planned(-2.0, 4.0), straight(0.0, 0.0)             # first overlap at point 4 (centres 0.4 m) from decision 0
    base = run(scene(30, pl, op), xp)
    assert base["charged"][0, 0] and base["first"][0, 0] == 4 and base["n_ok"][0, 0] == K
    # a restart during decision 1: decision 0 compares point 1 only (point k needs decisions t .. t+k-1 without one)
    brk = np.zeros((30, 1), bool); brk[1] = True
    out = run(scene(30, pl, op, brk=brk), xp)
    assert out["n_ok"][0, 0] == 1 and not out["over"][0, 0]
    # not knowable at the decision: nothing compared
    know = np.ones((30, 1), bool); know[0] = False
    out = run(scene(30, pl, op, know=know), xp)
    assert out["n_ok"][0, 0] == 0 and not out["charged"][0, 0]
    # the other car absent at t+3: the points from there on are not compared
    ok = np.ones((31, 1), bool); ok[3] = False
    out = run(scene(30, pl, op, ok=ok), xp)
    assert out["n_ok"][0, 0] == 2
    # the rollout's end: the last decision compares one point, the one before two (t + k <= T)
    T = 30
    assert list(base["n_ok"][T - 3:, 0]) == [3, 2, 1]


def test_numpy_and_torch_agree_on_random_scenes():
    rng = np.random.default_rng(3)
    T, B = 40, 64
    me_xy = rng.normal(0, 1.0, (T, B, K + 1, 2)); op_xy = rng.normal(0, 1.0, (T + 1, B, 2))
    sc = dict(me_xy=me_xy, me_psi=rng.uniform(-3, 3, (T, B, K + 1)), me_v=rng.normal(0, 3, (T, B, K + 1, 2)),
              me_rem=rng.uniform(0, 40, (T, B, K + 1)), op_xy=op_xy, op_psi=rng.uniform(-3, 3, (T + 1, B)),
              op_v=rng.normal(0, 3, (T + 1, B, 2)), op_rem=rng.uniform(0, 40, (T + 1, B)), op_ok=rng.random((T + 1, B)) < 0.95,
              brk=rng.random((T, B)) < 0.03, know=rng.random((T, B)) < 0.8, lap=np.full((T, B), 41.8))
    sc["me_rem"][rng.random((T, B, K + 1)) < 0.05] = np.nan          # off the distance field: the heading fallback
    a, b = run(sc, np), run(sc, torch)
    for k in a:
        assert np.allclose(a[k], b[k], atol=1e-9, equal_nan=True), k
    assert a["over"].any() and a["charged"].any() and (a["charged"] <= a["over"]).all()
