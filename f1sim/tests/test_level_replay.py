"""Prioritized level replay: the sampler, and a level seed reproducing its race exactly."""
import numpy as np
import torch

from f1sim.learn.level_replay import LevelReplay


def test_fresh_levels_until_the_buffer_fills_then_the_high_scores_come_back():
    lr = LevelReplay(1, p_replay=0.9, size=8, min_fill=4, seed=0)
    seeds = [lr.choose(0) for _ in range(1)]
    hard = None
    score = lambda s_: 10.0 if s_ == hard else 0.1                  # one level with much more to learn
    for i in range(4):
        s = seeds[-1]
        if i == 2:
            hard = s
        lr.accumulate(0, s, score(s), 1)
        seeds.append(lr.choose(0))
    assert len(set(seeds[:4])) == 4                                # all fresh until min_fill are stored
    picks = []
    for _ in range(300):
        s = lr.cur[0]
        lr.accumulate(0, s, score(s), 1)
        picks.append(lr.choose(0))
    replayed = [p_ for p_ in picks if p_ in seeds]
    assert replayed.count(hard) > 0.5 * len(replayed)
    assert len(lr.buf[0]) <= 8


def test_an_unplayed_level_is_not_stored_and_p_zero_never_replays():
    lr = LevelReplay(2, p_replay=0.0, seed=1)
    a = lr.choose(1); b = lr.choose(1)
    assert a != b and len(lr.buf[1]) == 0                          # `a` scored nothing: not kept
    assert lr.stats()["replayed"] == 0


def test_a_level_seed_reproduces_its_race_exactly():
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    from f1sim.params import Config
    trs, rls = common.load_tracks(["gen:competition:0"], racelines=False)
    cfg = Config(); cfg.sim.compile = False
    env = common.make_env(trs, 4, "cpu", EnvConfig(action_mode="direct", race_size=1, procedural_obstacles=1.0,
                                                    procedural_density=1.5, procedural_max_props=10,
                                                    procedural_raceline_corridor="off", spawn_runway=3.0), cfg=cfg, seed=0)

    class Fixed:                                                    # always the same level in seat 2
        def choose(self, seat):
            return 1234 if seat == 2 else 99 + seat
    env.level_replay = Fixed()
    env.reset(seed=5)
    snap = lambda: (env.sim.state[2].clone(), torch.stack([v[2] for v in env.sim.P.values()]).clone(),
                    env.procedural.p_poses[2].clone())
    env._reset_envs(torch.tensor([2]))
    a = snap()
    for _ in range(30):
        env.step(torch.zeros(4, 2))
    env._reset_envs(torch.tensor([1, 2, 3]))                       # with other seats resetting beside it
    b = snap()
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    assert int(env.level_seed[2]) == 1234
