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
