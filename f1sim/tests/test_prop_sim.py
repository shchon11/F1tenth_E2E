"""Static props through the real `Simulator`, not through the SAT primitive.

`prop_math`'s own tests establish that the SAT and the ray test are correct, and
`work/claude-map-redesign/check_props.py` establishes that the tensors reach them. Neither says
whether `StepResult.collision` ends up True, whether `terminate_on_collision` freezes the env, or
whether a partial reset clears the right rows -- those are properties of `Simulator.step`, and the
existing suite predates props entirely, so nothing covered them.

The prop is deliberately a `marker_post`: 5.5 cm circumradius against a 58 x 31 cm footprint, so it
sits wholly between the four corners `_footprint_clearance` samples. If the collision flag comes up
for this prop, it did not come from corner sampling.

Small on purpose: two envs, 24 beams.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from f1sim.params import Config
from f1sim.props import build as build_prop
from f1sim.sim import Simulator
from f1sim.track import StaticProp, Track

POST_XY = (6.0, 3.0)          # on the centerline, far from any wall


def _track(with_prop: bool) -> Track:
    """12 x 12 m of free space walled at the border, with a circular centerline through the post."""
    res, N = 0.05, 240
    occ = np.zeros((N, N), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    cl = np.stack([6.0 + 3.0 * np.cos(a - np.pi / 2), 6.0 + 3.0 * np.sin(a - np.pi / 2)], 1)
    t = Track.from_occupancy(occ, res, (0.0, 0.0), cl, "propfix", duct=np.zeros_like(occ),
                             tall=occ.copy(), duct_height=0.33)
    if with_prop:
        t.props = (StaticProp("marker_post", POST_XY[0], POST_XY[1], 0.0, seed=0),)
    return t


def _cfg(terminate: bool) -> Config:
    cfg = Config()
    cfg.sim.device = "cpu"
    cfg.sim.terminate_on_collision = terminate
    cfg.lidar.n_beams = 24
    cfg.imu.enabled = False
    return cfg


def _sim(with_prop=True, terminate=True, B=2):
    sim = Simulator(_track(with_prop), _cfg(terminate), num_envs=B, device="cpu")
    # Straddle the post exactly. Spawn sampling would reject this pose, which is the point of
    # passing it in: the physics has to cope with a car that is already on top of a prop.
    pose = torch.tensor([[POST_XY[0], POST_XY[1], 0.0]] * B)
    sim.reset(poses=pose)
    return sim


def _drive(sim, n=1, speed=1.0):
    a = torch.zeros(sim.B, 2)
    a[:, 1] = speed
    r = None
    for _ in range(n):
        r = sim.step(a)
    return r


def test_post_fits_between_the_sampled_corners():
    """Guard on the premise: if the post ever grew past the footprint, these tests would pass for
    the wrong reason -- corner sampling would find it and prove nothing about the SAT path."""
    r = float(build_prop("marker_post", seed=0).envelope.radius)
    cfg = Config()
    assert r < min(cfg.vehicle.length, cfg.vehicle.width) / 2


def test_a_post_just_off_the_flank_is_not_a_contact():
    """The case that separates a perimeter corner order from the stored bow-tie one.

    `Sim.corners` is [front-left, front-right, rear-left, rear-right]. Walking it in order gives
    edges FL->FR, FR->RL, RL->RR, RR->FL: the front, a diagonal, the rear, the other diagonal. The
    car's *lateral* axis never appears, and it is the only axis that separates a post sitting just
    off the flank. A post at the car's centre overlaps on every axis either way, so it cannot tell
    the two orders apart -- this can, and it is why the SAT call reorders to [0, 1, 3, 2].
    """
    cfg = Config()
    gap = 0.03
    lateral = cfg.vehicle.width / 2 + float(build_prop("marker_post", seed=0).envelope.radius) + gap
    sim = _sim(terminate=True)
    pose = torch.tensor([[POST_XY[0], POST_XY[1] - lateral, 0.0]] * sim.B)
    sim.reset(poses=pose)
    assert float(sim._prop_contact(sim.state)[0].max()) < 1e-4, \
        "a post clear of the flank by 3 cm is not a contact"
    assert not bool(_drive(sim).collision.any())


def test_collision_reported_and_env_frozen_when_terminating():
    sim = _sim(terminate=True)
    r = _drive(sim)
    assert bool(r.collision.all()), "a car standing on a prop must report a collision"
    before = sim.state.clone()
    _drive(sim, n=3, speed=3.0)
    assert torch.equal(sim.state, before), "terminate_on_collision must freeze the collided env"


def test_no_prop_no_collision_at_the_same_pose():
    """The control: the same pose on the same map without the prop is not a collision, so the flag
    above came from the prop and not from the walls or the spawn."""
    sim = _sim(with_prop=False, terminate=True)
    r = _drive(sim)
    assert not bool(r.collision.any())


def test_soft_mode_resolves_the_overlap_and_still_reports_it():
    sim = _sim(terminate=False)
    assert float(sim._prop_contact(sim.state)[0].max()) > 0, "the car should start overlapping"
    r = _drive(sim)
    assert bool(r.collision.all()), "a resolved contact is still a contact and must be reported"
    # 0.1 mm, not exact zero: the push-out lands on float32 coordinates of a few metres, so the
    # residual is epsilon-sized and device-dependent -- 0.0 on CPU, 2.4e-7 m on CUDA. Asserting
    # equality would report that rounding as a physics difference between the two backends.
    assert float(sim._prop_contact(sim.state)[0].max()) < 1e-4, "soft mode must push the car clear"


def test_substep_contact_survives_to_the_step_result():
    """`prop_touched` exists so a contact entered and left inside one control step is not lost."""
    sim = _sim(terminate=False)
    sim.prop_touched.zero_()
    sim.state[:, :2] = torch.tensor([[0.0, 0.0]])       # nowhere near the post at step time
    sim.prop_touched[:] = True                          # as a substep would have latched it
    r = _drive(sim)
    assert bool(r.collision.all())
    assert not bool(sim.prop_touched.any()), "the latch must be cleared once the step has read it"


def test_partial_reset_clears_only_the_selected_rows():
    sim = _sim(terminate=False)
    sim.prop_touched[:] = True
    sim.collided[:] = True
    sim.reset(env_ids=torch.tensor([0]))
    assert not bool(sim.prop_touched[0]), "the reset env's latch must be cleared"
    assert bool(sim.prop_touched[1]), "a partial reset must not touch the other env"
    assert not bool(sim.collided[0]) and bool(sim.collided[1])


def test_warmup_restores_the_latch():
    sim = _sim(terminate=False)
    sim.prop_touched[:] = torch.tensor([True, False])
    sim.warmup()
    assert list(sim.prop_touched) == [True, False], "warmup's throw-away steps must not leak state"


def test_spawn_avoids_the_props():
    """Sampled spawns must not land inside a prop; the wall EDT cannot see one."""
    sim = _sim(terminate=False, B=8)
    pose = sim.sample_spawn(8, tid=torch.zeros(8, dtype=torch.long))
    depth = sim._spawn_in_prop(pose, torch.zeros(8, dtype=torch.long))
    assert not bool(depth.any())


def test_a_prop_blocking_every_spawn_fails_loudly():
    """If nowhere on the lap is clear, that is a broken map and it has to say so rather than
    spawn the car inside a crate and call it a spawn."""
    t = _track(True)
    cl = t.centerline
    t.props = tuple(StaticProp("crate_stack_low", float(x), float(y), 0.0, seed=i)
                    for i, (x, y) in enumerate(cl[::8]))
    # The constructor resets, so the failure surfaces there rather than at an explicit call. That is
    # the behaviour worth pinning: a map like this cannot be built and quietly used.
    with pytest.raises(RuntimeError, match="no spawn pose clear"):
        Simulator(t, _cfg(False), num_envs=1, device="cpu")
