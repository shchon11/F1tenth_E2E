"""The time reward: what it pays for a lap that got faster, and when it pays nothing.

Progress reward already points at lap time, but it is blunt in two ways this term has to fix. It
cannot see a tenth (under 1 % of the return, and less the longer the track), and it treats every
second as worth the same, when a second found while still far off the pace is easy and a hundredth
found at the limit is the harder result by far.
"""
import numpy as np
import torch

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv, REWARD_COMPONENT_KEYS

LT = REWARD_COMPONENT_KEYS.index("lap_time")


def _env(n=4, **kw):
    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    e = EnvConfig(reward_lap_time=1.0, reward_progress=0.0, **kw)
    env = F1VecEnv(Track.generate_random(4), cfg, e, num_envs=n, device="cpu")
    env.reset(seed=5)
    env.sim.tid[:] = 0
    return env


def _cross(env, sector_time, ref, lim, frac=0.0, valid=True, sector=3):
    """Score one crossing of the boundary out of `sector`, having taken `sector_time` to drive it.

    frac is where inside the last control step the boundary was crossed, which is the whole point of
    interpolating: a tenth of a lap spread over twelve sectors is under a hundredth each, far below
    the 25 ms step that a step count would round it to.
    """
    S = env.sector_lim.shape[1]
    L = float(env.sim.track.length[0])
    dt = env.sim.control_dt
    env.sector_lim[0] = lim
    env.sector_ref[0] = ref
    env.sector_idx[:] = sector
    env.sector_valid[:] = valid
    n = 400
    env.ep_step[:] = n
    # entry time pinned to the step grid, so a later crossing inside the final step *is* a slower
    # sector -- otherwise the helper would cancel out the very interpolation under test
    env.sector_t0[:] = n * dt - sector_time
    b = (sector + 1) * L / S                                  # boundary just crossed
    ds = 0.2                                                  # arc covered in the last step
    s = torch.full((env.B,), b + frac * ds)
    env.s_prev[:] = s - ds
    reward = torch.zeros(env.B)
    comp = torch.zeros(env.B, len(REWARD_COMPONENT_KEYS))
    env._sector_time_reward(s, reward, comp)
    return reward, comp


def test_a_hundredth_at_the_limit_outweighs_most_of_a_second_far_from_it() -> None:
    """The reward is in fractions of the headroom left, not in seconds.

    A track whose raceline says 10.0 s: coming down from 15 s to 14 s is a second, and going from
    10.10 s to 10.09 s is a hundredth -- a hundred times less time and much the harder result. Flat
    seconds would pay the easy one a hundred times more. Headroom pays them within a factor of three.
    """
    S = 12
    lim = 10.0 / S
    far = _cross(_env(), 14.0 / S, ref=15.0 / S, lim=lim)[0][0].item() * S     # 15 s -> 14 s
    near = _cross(_env(), 10.09 / S, ref=10.10 / S, lim=lim)[0][0].item() * S  # 10.10 s -> 10.09 s
    assert far > 0 and near > 0
    # a hundredth at the limit is worth a large fraction of a whole second far from it
    assert near / far > 0.35, (near, far)
    # and a second far from the pace is still worth more than a hundredth near it, not less
    assert far > near


def test_equal_fractions_of_the_headroom_pay_equally() -> None:
    """Halving the gap to the raceline is one result whether the gap was 4 s or 0.04 s."""
    S = 12
    lim = 10.0 / S
    wide = _cross(_env(), (10.0 + 3.0) / S, ref=(10.0 + 4.0) / S, lim=lim)[0][0].item()
    tight = _cross(_env(), (10.0 + 0.03) / S, ref=(10.0 + 0.04) / S, lim=lim)[0][0].item()
    assert abs(wide - tight) < 1e-4, (wide, tight)


def test_a_sector_slower_than_the_reference_is_never_charged() -> None:
    """One-sided by design. A toll at a boundary is hackable: stopping short of it at the end of an
    episode costs almost no progress reward, so a policy would learn to wait there."""
    S = 12
    r, comp = _cross(_env(), 2.0 / S, ref=1.0 / S, lim=0.5 / S)
    assert bool((r == 0).all()) and bool((comp[:, LT] == 0).all())


def test_a_sector_entered_mid_way_is_not_scored() -> None:
    """A car spawns in the middle of a sector, so the time to its first boundary is not a sector it
    drove -- and it is short, which would make it look like the fastest sector ever run."""
    r, _ = _cross(_env(), 0.05, ref=1.0, lim=0.5, valid=False)
    assert bool((r == 0).all())


def test_the_crossing_instant_is_interpolated_not_rounded_to_the_step() -> None:
    """Two cars whose boundary crossing lands on the same control step did not cross at the same
    time: one is barely over the line, the other most of the way into the next sector.

    Rounding both to the step would quantise sector times to 25 ms, and a tenth of a lap spread over
    twelve sectors is under a hundredth each -- the whole signal, rounded away.
    """
    barely_over = _cross(_env(), 1.0, ref=1.02, lim=0.5, frac=0.0)[0][0].item()
    well_past = _cross(_env(), 1.0, ref=1.02, lim=0.5, frac=0.9)[0][0].item()
    assert barely_over != well_past
    # being well past the line means it was crossed early in the step: a shorter sector, paid more
    assert well_past > barely_over


def test_the_reference_follows_the_pace_but_never_sinks_below_the_raceline() -> None:
    env = _env(n=64)
    S = env.sector_lim.shape[1]
    env.sector_ref[0] = 0.0                                   # unseen: the first clean time seeds it
    for _ in range(400):
        _cross(env, 0.80, ref=env.sector_ref[0].clone(), lim=0.90)
    ref = float(env.sector_ref[0, 3])
    assert ref >= 0.90, ref                                   # floored at the raceline, not chased past it
    env.sector_ref[0] = 0.0
    for _ in range(400):
        _cross(env, 0.80, ref=env.sector_ref[0].clone(), lim=0.50)
    assert abs(float(env.sector_ref[0, 3]) - 0.80) < 0.02      # converges on what is actually driven


def test_sector_references_sum_to_the_ideal_lap() -> None:
    """Sectors are cut on centerline arc and the raceline is a lateral offset of it, so every
    raceline point lands in exactly one sector and the parts add up to the whole."""
    from f1sim.raceline import Raceline
    tracks = [Track.generate_random(i) for i in range(3)]
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    env = F1VecEnv(tracks, cfg, EnvConfig(reward_lap_time=1.0), num_envs=3, device="cpu")
    rls = [Raceline.build_cached(t) for t in tracks]
    env.set_ideal_lap(rls)
    torch.testing.assert_close(env.sector_lim.sum(1), env.ideal_lap, rtol=1e-4, atol=1e-4)
    assert bool((env.sector_lim > 0).all())


def test_a_wobble_across_a_boundary_is_not_a_sector() -> None:
    """The lap counter is symmetric -- it counts down when the car slips back over a line and up
    again when it returns -- so a car sitting on a boundary crosses it forward again and again.
    Timed naively each of those is the fastest sector ever driven, and repeatable at will: the cap on
    a single sector's gain would become a payout per oscillation.

    A sector cannot be driven faster than its arc length at the speed limit, and that is the bound.
    """
    env = _env()
    S = env.sector_lim.shape[1]
    bound = float(env.sim.track.length[0]) / S / env.ecfg.v_max_policy
    wobble, _ = _cross(env, 0.7 * bound - 1e-3, ref=1.0, lim=0.5)
    assert bool((wobble == 0).all())
    driven, _ = _cross(_env(), 1.5 * bound, ref=1.0, lim=0.5)
    assert bool((driven > 0).all())


def test_a_wobble_does_not_move_the_reference_either() -> None:
    """An unscored crossing must not seed or drag the EMA: a reference pulled down to a wobble's
    time would stop paying for real sectors on that part of the track entirely."""
    env = _env()
    S = env.sector_lim.shape[1]
    bound = float(env.sim.track.length[0]) / S / env.ecfg.v_max_policy
    env.sector_ref[0] = 0.0
    for _ in range(50):
        _cross(env, 0.5 * bound, ref=env.sector_ref[0].clone(), lim=0.5)
    assert float(env.sector_ref[0, 3]) == 0.0            # still unseen
