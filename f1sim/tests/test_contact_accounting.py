"""Who pays for a car contact, and at which price (2026-10-13 audit).

The simulator ORs the car-contact flag into its collision flag, and the `collision` reward component --
the one `--lagrange wall=` counts -- was read from that flag. So every car contact was charged at the
wall's multiplier as well as the car's, and the car-behind exemption only ever exempted the car half.
The exemption itself was re-read every step of a contact, and in soft mode cars pass through each other,
so the order of the two cars flips within a few steps and the exempt car was charged anyway.
"""
import torch

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv, REWARD_COMPONENT_KEYS

COLL = REWARD_COMPONENT_KEYS.index("collision")
CAR = REWARD_COMPONENT_KEYS.index("car_contact")


def _race(**kw):
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
    env = F1VecEnv(Track.generate_random(2), cfg,
                   EnvConfig(race_size=2, opponent="policy", collision_mode="soft",
                             reward_collision=-1.0, reward_car_contact=1.0, **kw),
                   num_envs=2, device="cpu")
    env.reset(seed=3)
    return env


def _overlap(env, dx=0.2):
    """Car 1 put on top of car 0, `dx` metres ahead along car 0's heading: a contact, no wall near."""
    st = env.sim.state
    yaw = st[0, 2]
    st[1, :] = st[0, :]
    st[1, 0] = st[0, 0] + dx * torch.cos(yaw)
    st[1, 1] = st[0, 1] + dx * torch.sin(yaw)


def _step(env):
    _o, _r, _t, _tr, info = env.step(torch.zeros(env.B, env.act_dim))
    comp = info["reward_components"]
    return comp["collision"].clone(), comp["car_contact"].clone(), info


def test_a_car_contact_is_not_also_charged_as_a_wall() -> None:
    env = _race()
    _overlap(env)
    coll, car, info = _step(env)
    r = env.last_result
    assert bool(r.car_collision.all()), "the forced overlap did not register as a car contact"
    assert not bool(r.wall_collision.any()), "the test scene touches a wall; move it"
    assert bool(info["contact_onset"].all())                 # the contact itself is still an onset
    assert coll.tolist() == [0.0, 0.0], coll                 # ... but not a wall/prop collision
    assert bool((car < 0).all()), car                        # alongside: both cars pay the car price


def test_the_legacy_flag_reproduces_the_double_charge() -> None:
    env = _race(collision_counts_cars=True, car_charge_latch=False)
    _overlap(env)
    coll, car, _info = _step(env)
    assert coll.tolist() == [-1.0, -1.0], coll
    assert bool((car < 0).all()), car


def test_the_exemption_is_decided_when_the_contact_begins() -> None:
    """Car 1 a full length ahead at the first step of the contact is exempt; when the cars have slid
    alongside two steps later, the instantaneous rule would charge it, and the latch must not."""
    env = _race()
    tid = env.sim.tid; hit = torch.ones(2)
    ahead = torch.tensor([4.0, 5.0])                          # car 1 a full length ahead of car 0
    alongside = torch.tensor([5.0, 5.1])
    assert env._car_charge(ahead, tid, hit).tolist() == [1.0, 0.0]
    assert env._car_charge(alongside, tid, hit).tolist() == [1.0, 0.0]    # held
    assert env._car_charge(alongside, tid, torch.zeros(2)).tolist() == [0.0, 0.0]   # separated
    assert env._car_charge(alongside, tid, hit).tolist() == [1.0, 1.0]    # a new contact, judged anew
    legacy = _race(car_charge_latch=False)
    assert legacy._car_charge(ahead, tid, hit).tolist() == [1.0, 0.0]
    assert legacy._car_charge(alongside, tid, hit).tolist() == [1.0, 1.0]  # re-read: charged


def test_a_wall_contact_is_still_charged_once() -> None:
    """The wall half is untouched: a car driven into the boundary pays the collision component on the
    first step of the contact and not on the steps it stays there."""
    env = _race()
    st = env.sim.state
    st[1, 0] = st[0, 0] + 5.0                                 # the other car well out of the way
    paid = []
    for _ in range(80):                                       # steer hard left at speed into the boundary
        a = torch.zeros(env.B, env.act_dim); a[0, 0] = 1.0; a[0, 1] = 1.0
        _o, _r, _t, _tr, info = env.step(a)
        paid.append(float(info["reward_components"]["collision"][0]))
    assert sum(1 for p in paid if p < 0) >= 1, "never reached the wall"
    # every charge is an onset: no two consecutive charged steps
    assert all(not (a < 0 and b < 0) for a, b in zip(paid, paid[1:])), paid
