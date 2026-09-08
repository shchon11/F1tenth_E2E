import math
import numpy as np
import torch
from f1sim import Config, Simulator
from f1sim.track import Track
from f1sim.lidar import HIT_CAR, ray_box_hits
from f1sim.gym_env import EnvConfig, F1VecEnv
from f1sim.learn import common


def open_floor():
    occ = np.zeros((800, 800), bool); occ[:2] = occ[-2:] = True; occ[:, :2] = occ[:, -2:] = True
    return Track.from_occupancy(occ, 0.05, (-20.0, -20.0), name="open")


def test_ray_box_hits_geometry():
    N = 5
    origin = torch.zeros(1, N, 3); origin[..., 2] = 0.15
    ang = torch.tensor([0.0, 0.03, -0.03, 0.5, math.pi / 2])
    dh = torch.stack([torch.cos(ang), torch.sin(ang)], -1)[None]
    k = torch.zeros(1, N)
    boxes = torch.tensor([[[3.0, 0.0, 0.0]]]); dims = torch.tensor([[[0.5, 0.3, 0.2]]])
    r, hit = ray_box_hits(origin, dh, k, boxes, dims)
    assert hit[0, 0] and abs(float(r[0, 0]) - 2.75) < 1e-4            # front face of the box
    assert hit[0, 1] and hit[0, 2] and not hit[0, 3] and not hit[0, 4]
    # a beam climbing above the roof passes over the car
    k_up = torch.full((1, N), 0.05)                                    # +5 cm per metre: 0.15 + 0.1375 > 0.2? no -> 0.29 > 0.2 yes
    r2, hit2 = ray_box_hits(origin, dh, k_up, boxes, dims)
    assert not hit2[0, 0]


def test_race_lidar_and_contact():
    cfg = Config(); cfg.rand.enabled = False; cfg.lidar.motion_distortion = False
    sim = Simulator(open_floor(), cfg, num_envs=2, device="cuda" if torch.cuda.is_available() else "cpu", race_size=2)
    dev = sim.device
    poses = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=dev)     # car 1 two metres ahead of car 0
    sim.reset(torch.arange(2, device=dev), poses=poses, speed=torch.zeros(2, device=dev))
    r = sim.step(torch.zeros(2, 2, device=dev))
    n = sim.lidar.n; mid = n // 2
    assert int(r.scan_type[0, mid]) == HIT_CAR
    # LiDAR sits 0.27 m ahead of the rear axle; the first surface is the rear detection box behind the body
    expect = 2.0 - 0.27 - 0.5 * float(sim.car_dims[1, 0]) - float(sim.car_rear[1, 0])
    assert abs(float(r.scan_true[0, mid]) - expect) < 0.03
    beams = r.scan_type[0] == HIT_CAR
    assert bool(torch.isfinite(r.scan[0][beams]).float().mean() > 0.9)   # the rear box gives solid returns
    assert int(r.scan_type[1, mid]) != HIT_CAR                          # the leader looks at empty floor
    assert not bool(r.collision.any())
    # drive them into each other: overlap -> both collide
    poses = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.05, 0.3]], device=dev)
    sim.reset(torch.arange(2, device=dev), poses=poses, speed=torch.zeros(2, device=dev))
    r = sim.step(torch.zeros(2, 2, device=dev))
    assert bool(r.car_collision.all()) and bool(r.collision.all())


def test_race_env_with_teacher_opponents():
    from f1sim import maps
    tracks = [maps.load("gen:competition:0"), maps.load("gen:competition:1")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = common.make_env(tracks, 8, dev, EnvConfig(race_size=2, opponent="teacher", scan_stack=3, scan_stride=3, speed_cap=4.0), seed=3)
    obs, info = env.reset(seed=3)
    assert obs["scan"].shape == (8, 3, env.n_beams) and env.hist_len == 7
    assert env.learner.tolist() == [True, False] * 4
    tid = env.sim.tid.view(4, 2)
    assert bool((tid[:, 0] == tid[:, 1]).all())                         # both cars of a race share the track
    s = env.sim.s.view(4, 2); L = env.sim.track.length[tid[:, 0]]
    behind = torch.remainder(s[:, 0] - s[:, 1], L)
    assert bool(((behind > 2.0) & (behind < 6.5)).all())                # opponent spawned 2.5-6 m behind the leader
    priv = env.privileged(env.last_result)
    assert priv.shape[1] == 8 + 4 + len(env.PRIV_PARAMS) + 1
    seen = 0
    for _ in range(60):
        a = torch.zeros(8, 2, device=dev); a[:, 1] = -0.2                  # learners crawl, opponents come from behind
        obs, rew, term, trunc, info = env.step(a)
        seen += int((env.last_result.scan_type[env.learner] == HIT_CAR).any())
    assert seen > 0                                                     # a learner saw an opponent in its scan


def test_car_proximity_penalty_falls_on_the_following_car_only():
    """Two cars in line: the one behind pays for the gap, the one in front pays nothing."""
    import torch
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks, _ = common.load_tracks(["gen:competition:0"])
    cfg = Config(); cfg.rand.enabled = False
    env = common.make_env(tracks, 2, dev, EnvConfig(race_size=2, opponent="policy", reward_car_proximity=1.0, car_safe_dist=0.5,
                                                    reward_progress=0.0, reward_alive=0.0, reward_steer_rate=0.0, reward_proximity=0.0,
                                                    reward_wrong_way=0.0), cfg=cfg, seed=1)
    env.reset(seed=1)
    st = env.sim.state.clone()
    st[0, :3] = torch.tensor([0.0, 0.0, 0.0], device=st.device)          # car 0 behind, facing +x
    st[1, :3] = torch.tensor([env.car_len + 0.2, 0.0, 0.0], device=st.device)   # car 1 a 0.2 m gap ahead
    st[:, 3] = 0.0
    env.sim.state = st
    _, rew, _, _, _ = env.step(torch.zeros(2, env.act_dim, device=st.device))
    assert rew[0] < -0.3, rew                                             # following car: (0.5-0.2)/0.5 = 0.6
    assert rew[1] > rew[0] and rew[1] > -0.05, rew                        # leading car: nothing


def test_mixed_opponents_split_races_between_teacher_and_self_play():
    """opponent='mix': a share of the races is driven by the raceline teacher (so the policy meets cars
    much slower than itself, which self-play never produces), the rest is self-play with every car learning."""
    import torch
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks, rls = common.load_tracks(["gen:competition:0"], racelines=True)
    cfg = Config(); cfg.rand.enabled = False
    env = common.make_env(tracks, 12, dev, EnvConfig(race_size=3, opponent="mix", teacher_race_frac=0.5,
                                                     opp_speed_range=(0.4, 0.6), action_mode="plan"), cfg=cfg, seed=2, rls=rls)
    env.reset(seed=2)
    lm = env.learner.view(-1, 3)
    assert lm[:, 0].all()                                        # the lead car of every race learns
    assert (~lm[:2, 1:]).all() and lm[2:, 1:].all()              # first half teacher-driven, rest self-play
    a = torch.zeros(12, env.act_dim, device=env.device); a[:, -2:] = -1.0   # policy asks for the lowest speed
    for _ in range(20):
        env.step(a)
    v = env.sim.state[:, 3].view(-1, 3)
    assert v[:2, 1:].abs().max() > 0.5                           # teacher cars drive anyway
    assert v[2:, 1:].abs().max() < 0.5 * float(v[:2, 1:].abs().max())   # self-play cars obey that, teacher cars do not


def test_plan_car_penalty_fires_when_the_plan_aims_at_the_car_ahead():
    """The planned path is scored against where the other cars will be, not where they are: a plan that
    runs into a slow car ahead costs, the same plan with that car off to the side costs nothing."""
    import torch
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks, _ = common.load_tracks(["gen:competition:0"])
    cfg = Config(); cfg.rand.enabled = False
    def rew(other_xy):
        env = common.make_env(tracks, 2, dev, EnvConfig(race_size=2, opponent="policy", action_mode="plan", reward_plan_car=1.0,
                                                        plan_car_margin=0.55, reward_progress=0.0, reward_alive=0.0, reward_steer_rate=0.0,
                                                        reward_proximity=0.0, reward_wrong_way=0.0, reward_plan_clearance=0.0,
                                                        reward_collision=0.0, reward_collision_speed=0.0), cfg=cfg, seed=1)
        env.reset(seed=1)
        a = torch.zeros(2, env.act_dim, device=env.device)                     # plan: straight ahead
        env.step(a)                                                            # the tracker builds that plan
        st = env.sim.state.clone()
        st[0, :4] = torch.tensor([0.0, 0.0, 0.0, 3.0], device=st.device)       # ego at 3 m/s facing +x
        st[1, :4] = torch.tensor([other_xy[0], other_xy[1], 0.0, 0.2], device=st.device)   # a crawling car
        env.sim.state = st
        _, r, _, _, _ = env.step(a)
        return float(r[0])
    in_the_way = rew((0.7, 0.0))
    off_to_the_side = rew((0.7, 3.0))
    assert in_the_way < -0.1, in_the_way
    assert off_to_the_side > -0.01, off_to_the_side
