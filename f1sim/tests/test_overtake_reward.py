"""What the overtake reward pays over a complete pass.

The point of the term is to make passing worth doing: a collision penalty on its own teaches
avoidance and only avoidance, and sitting behind is free. So the one step that must not be punished
is the step where the pass completes.
"""
import torch

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv, REWARD_COMPONENT_KEYS

OT = REWARD_COMPONENT_KEYS.index("overtake")


def _race(m=2, n=4):
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
    env = F1VecEnv(Track.generate_random(2), cfg,
                   EnvConfig(race_size=m, opponent="policy", reward_overtake=1.0, reward_progress=0.0),
                   num_envs=n, device="cpu")
    env.reset(seed=1)
    return env


def _drive(env, positions):
    """Score the real reward path over a sequence of arc positions, one row per car.

    The arc positions are handed straight to the scoring function rather than driven to: what is
    under test is the geometry of a pass, and making the cars actually arrive at it would test the
    controller instead.
    """
    tid = env.sim.tid
    gap = torch.zeros(env.B, env.M - 1); valid = torch.zeros(env.B, dtype=torch.bool)
    out = []
    for row in positions:
        s = torch.tensor(row, dtype=torch.float32).repeat(env.B // env.M)
        gain, gap = env.overtake_gain(s, tid, gap, valid)
        valid = torch.ones_like(valid)
        out.append(float(gain[0]))
    return out


def test_the_step_that_completes_a_pass_is_not_punished() -> None:
    """It used to be the most punished step of the manoeuvre. The gap was the *forward* arc to the
    other car, which jumps from ~0 to ~L the instant you get in front, and the reward read that jump
    as a full lap lost: -2.00 (the clamp) where it should have been +0.25."""
    env = _race()
    gains = _drive(env, [[6.03 + 0.25 * i, 10.0] for i in range(33)])
    assert min(gains) > -0.01, f"a step of the pass was punished: {min(gains):.2f}"
    assert max(gains) > 0.2, gains


def test_a_whole_pass_pays_the_arc_it_actually_took_out() -> None:
    env = _race()
    gains = _drive(env, [[6.03 + 0.25 * i, 10.0] for i in range(33)])
    # 8 m of arc taken out, less the first step, which has no previous gap to compare against
    assert abs(sum(gains) - 7.75) < 0.3, sum(gains)


def test_being_passed_costs_what_passing_pays() -> None:
    """Symmetric by construction -- otherwise the cheapest way to score is to be overtaken and
    re-overtake the same car all race."""
    env = _race()
    passing = sum(_drive(env, [[6.03 + 0.25 * i, 10.0] for i in range(33)]))
    passed = sum(_drive(_race(), [[10.0, 6.03 + 0.25 * i] for i in range(33)]))
    assert abs(passing + passed) < 0.05, (passing, passed)


def test_three_cars_score_the_whole_field_not_one_slot() -> None:
    """other_idx is a fixed roster, so with three cars reading slot 0 tracks whichever car sits
    there. Moving up on both opponents has to pay whichever one that is."""
    env = _race(m=3, n=6)
    assert env.gap_prev.shape == (env.B, 2)
    gains = _drive(env, [[6.0 + 0.5 * i, 12.0, 20.0] for i in range(8)])
    assert min(gains[1:]) > 0.0, gains


def test_a_lap_apart_is_not_scored_as_a_pass() -> None:
    """Half a lap is the furthest two cars can be from each other; beyond that the shorter way round
    is behind. A gap that wraps must not read as a sudden lap gained."""
    env = _race()
    L = float(env.sim.track.length[0])
    gains = _drive(env, [[0.0, L / 2 - 0.2 + 0.1 * i] for i in range(6)])
    assert max(abs(g) for g in gains) < 1e-6, gains        # out of contention: not scored at all


def test_a_car_out_of_contention_is_not_scored() -> None:
    """Closing on someone 20 m up the lane is not a pass in progress, and paying for it would make
    the term a second, noisier progress reward."""
    env = _race()
    far = _drive(env, [[0.0 + 0.25 * i, 25.0] for i in range(8)])
    assert max(abs(g) for g in far) < 1e-6, far
    close = _drive(_race(), [[0.0 + 0.25 * i, 5.0] for i in range(8)])
    assert sum(close) > 1.0, close


def _state(rows):
    """(x, y, yaw, vx) per car, repeated over the races of a 2-car env."""
    st = torch.zeros(len(rows), 7)
    for i, (x, y, yaw, vx) in enumerate(rows):
        st[i, 0], st[i, 1], st[i, 2], st[i, 3] = x, y, yaw, vx
    return st


def test_tailgating_is_charged_and_a_clean_pass_is_not() -> None:
    """The car has a proximity gradient now, as the wall always had. Closing on a bumper at 0.1 m
    body gap costs; sitting 0.8 m to the side of the same car costs nothing."""
    env = _race(n=2)
    e = env.ecfg
    tail = _state([(0.0, 0.0, 0.0, 4.0), (e.car_len + 0.10, 0.0, 0.0, 3.0)])         # me behind, closing 1 m/s
    beside = _state([(0.0, 0.0, 0.0, 4.0), (0.0, e.car_wid + 0.80, 0.0, 4.0)])        # beside, outside safe gap
    assert float(env.car_proximity(tail)[0]) > 0.8, float(env.car_proximity(tail)[0])
    assert float(env.car_proximity(beside)[0]) == 0.0


def test_closing_fast_costs_more_than_closing_slowly() -> None:
    env = _race(n=2); e = env.ecfg
    slow = _state([(0.0, 0.0, 0.0, 3.5), (e.car_len + 0.2, 0.0, 0.0, 3.0)])
    fast = _state([(0.0, 0.0, 0.0, 7.0), (e.car_len + 0.2, 0.0, 0.0, 3.0)])
    assert float(env.car_proximity(fast)[0]) > float(env.car_proximity(slow)[0]) * 1.5


def test_a_pass_still_pays_more_than_the_room_it_costs() -> None:
    """The user's actual worry: overtaking must be worth it after the proximity charge, and it must
    not be worth it to tailgate instead. Two full passes over 8 m of relative arc: one clean, going
    round at 0.5 m of body gap, one squeezing through at 0.05 m."""
    env = _race(n=2)
    e = env.ecfg
    ot, prox = 1.0, 0.5                                   # the coefficients the race leg uses
    tid = env.sim.tid
    def pass_total(lateral):
        gap = torch.zeros(env.B, 1); valid = torch.zeros(env.B, dtype=torch.bool); total = 0.0
        for i in range(33):
            x = 6.03 + 0.25 * i
            st = _state([(x, lateral, 0.0, 5.0), (10.0, 0.0, 0.0, 4.0)])
            s = torch.tensor([x, 10.0]); gain, gap = env.overtake_gain(s, tid, gap, valid); valid[:] = True
            travelled = 5.0 * env.sim.control_dt
            total += ot * float(gain[0]) - prox * float(env.car_proximity(st)[0]) * travelled
        return total
    clean = pass_total(e.car_wid + 0.5)
    squeeze = pass_total(e.car_wid + 0.05)
    assert clean > 6.0, clean                              # 8 m of arc taken out, nearly all of it kept
    assert squeeze < clean, (squeeze, clean)               # the room costs something
    assert squeeze > 0.0, squeeze                          # but a pass is never worse than not passing


def test_with_teacher_opponents_the_learner_starts_behind_and_in_contention() -> None:
    """An overtake is a car behind passing a car ahead. The learner used to spawn as the leader
    with the teacher 2.5-6 m *behind* it, so the situation the overtake reward exists for never
    occurred at spawn, and the only passes it saw were the ones done to it."""
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
    env = F1VecEnv(Track.generate_random(2), cfg,
                   EnvConfig(race_size=2, opponent="teacher", reward_overtake=1.0), num_envs=8, device="cpu")
    from f1sim.raceline import Raceline
    from f1sim.learn.common import make_teacher
    env.set_teacher(make_teacher([Raceline.build_cached(env.sim.tracks[0])], env))
    env.reset(seed=3)
    d = env.signed_gaps(env.sim.s, env.sim.tid)[env.learner][:, 0]     # + = opponent ahead of the learner
    assert bool((d > 0).all()), d
    assert bool((d < env.ecfg.overtake_range).all()), d                # close enough to be racing it


def test_a_teacher_opponent_holds_speed_behind_a_car() -> None:
    """The raceline teacher is blind; without this it drives into the car it was just passed by."""
    env = _race(n=2)
    env.sim.s[:] = torch.tensor([5.0, 4.0])                     # car 1 is 1 m behind car 0
    st = torch.zeros(2, 7); st[:, 3] = torch.tensor([3.0, 6.0]) # car 0 slow, car 1 fast
    follow, v_cap = env.follow_cap(st)
    assert follow.tolist() == [False, True]                      # only the one with a car ahead
    assert abs(float(v_cap[1]) - 3.0 * env.ecfg.opp_follow_ratio) < 1e-6


def test_the_contact_penalty_falls_on_the_car_behind() -> None:
    env = _race(n=2)
    tid = env.sim.tid; hit = torch.ones(2)
    s = torch.tensor([4.0, 5.0])                                 # car 0 a full length behind car 1
    charged = env.car_contact_charge(s, tid, hit)
    assert charged.tolist() == [1.0, 0.0]
    # alongside (arc gap under a car length): both are charged -- the contact is the pass itself, and
    # a near-zero arc gap's sign says nothing about who moved into whom
    s = torch.tensor([5.0, 5.2])
    assert env.car_contact_charge(s, tid, hit).tolist() == [1.0, 1.0]


def test_alongside_the_penalty_still_has_a_gradient() -> None:
    """A rub at 0.05 m of daylight and a pass at 0.30 m must not score the same: the old
    centre-distance form saturated before either, so the policy had nothing to climb."""
    env = _race(n=2); e = env.ecfg
    near = _state([(0.0, 0.0, 0.0, 4.0), (0.0, e.car_wid + 0.05, 0.0, 4.0)])
    far = _state([(0.0, 0.0, 0.0, 4.0), (0.0, e.car_wid + 0.30, 0.0, 4.0)])
    pn, pf = float(env.car_proximity(near)[0]), float(env.car_proximity(far)[0])
    assert pn > pf + 0.3, (pn, pf)
    assert pf > 0.0


def test_self_play_slows_the_front_car_so_there_is_something_to_pass() -> None:
    """Identical cars never pass each other: the overtake term read 0.0006/step in self-play against
    0.016 with teacher opponents. The front car of each self-play race drives under a scaled cap."""
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
    env = F1VecEnv(Track.generate_random(2), cfg,
                   EnvConfig(race_size=2, opponent="policy", speed_cap=6.0, opp_speed_range=(0.5, 0.8)),
                   num_envs=8, device="cpu")
    env.reset(seed=2)
    cap = env.speed_cap.view(4, 2)
    assert bool((cap[:, 0] < cap[:, 1] - 1e-6).all()), cap        # front car (slot 0) capped lower
    assert bool((cap[:, 1] == 6.0).all())
    # scaled against the pace the policy actually drives (5 m/s), not the 6.0 cap: 0.5-0.8 -> 2.5-4.0
    assert bool(((cap[:, 0] >= 2.5 - 1e-6) & (cap[:, 0] <= 4.0 + 1e-6)).all()), cap
    env.set_speed_cap(8.0)                                        # a curriculum step keeps the scaling
    assert bool((env.speed_cap.view(4, 2)[:, 1] == 8.0).all()) and bool((env.speed_cap.view(4, 2)[:, 0] < 8.0).all())
    # teacher opponents: the learner's cap is untouched (the teacher is slowed through its own action)
    env2 = F1VecEnv(Track.generate_random(2), cfg, EnvConfig(race_size=2, opponent="teacher", speed_cap=6.0),
                    num_envs=4, device="cpu")
    env2.reset(seed=2)
    assert bool((env2.speed_cap == 6.0).all())


def test_mixed_opponents_split_races_between_teacher_and_self_play() -> None:
    """Each race is either a teacher race (learner behind a teacher-driven car, that car not in the
    loss) or a self-play race (both cars the policy's, front car capped). Both kinds in one batch."""
    from f1sim.raceline import Raceline
    from f1sim.learn.common import make_teacher
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
    env = F1VecEnv(Track.generate_random(2), cfg,
                   EnvConfig(race_size=2, opponent="mixed", mixed_teacher_frac=0.5, speed_cap=6.0,
                             opp_speed_range=(0.5, 1.0), selfplay_pace_ref=5.0),
                   num_envs=64, device="cpu")
    env.set_teacher(make_teacher([Raceline.build_cached(env.sim.tracks[0])], env)); env.reset(seed=4)
    assert bool(env.learner.all())                                   # every car sits in the buffer
    tr = env.teacher_race.view(32, 2); onp = env.on_policy.view(32, 2); cap = env.speed_cap.view(32, 2)
    assert bool((tr[:, 0] == tr[:, 1]).all())                        # a race is one kind or the other
    n_teacher = int(tr[:, 0].sum()); assert 6 <= n_teacher <= 26, n_teacher   # both kinds present
    # teacher races: slot 0 (learner) is on-policy, slot 1 is teacher-driven and masked out
    assert bool(onp[tr[:, 0], 0].all()) and not bool(onp[tr[:, 0], 1].any())
    # self-play races: both on-policy; the front car (slot 0) capped below 6.0
    assert bool(onp[~tr[:, 0]].all())
    assert bool((cap[~tr[:, 0], 0] < 6.0 - 1e-6).all()) and bool((cap[~tr[:, 0], 1] == 6.0).all())
    assert bool((cap[tr[:, 0]] == 6.0).all())                        # teacher races: caps untouched
    # spawn order: teacher races have the learner behind; self-play has the front car in slot 0
    d = env.signed_gaps(env.sim.s, env.sim.tid).view(32, 2)[:, 0]    # slot 0's gap to slot 1 (+ = ahead of it)
    assert bool((d[tr[:, 0]] > 0).all()) and bool((d[~tr[:, 0]] < 0).all())
    # and the teacher actually drives the masked cars: their action differs from what was passed in
    a = torch.zeros(64, env.act_dim)
    a2 = env._opponent_actions(a)
    changed = (a2 != a).any(1).view(32, 2)
    assert bool(changed[tr[:, 0], 1].all()) and not bool(changed[:, 0].any()) and not bool(changed[~tr[:, 0], 1].any())
