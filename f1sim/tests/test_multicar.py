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
    # Where the first surface sits depends on which car model the LiDAR traces: "parts" returns the
    # rear detection box, "mesh" returns the sliced body, which lies *inside* that box. Pinning the
    # box face to 3 cm was pinning the old model. What holds for both -- and what a wrong return
    # would violate -- is that the hit lies between the enclosing box's near face and the opponent's
    # centre. The mesh outline itself is covered by test_car_mesh_lidar.py.
    centre = 2.0 - 0.27                                   # beam distance to car 1's centre
    box_face = centre - 0.5 * float(sim.car_dims[1, 0]) - float(sim.car_rear[1, 0])
    got = float(r.scan_true[0, mid])
    assert box_face - 0.03 <= got <= centre, (
        f"first car surface {got:.3f} m outside [{box_face:.3f}, {centre:.3f}]: a return may not "
        f"stick out past the enclosing box nor land beyond the opponent's centre")
    beams = r.scan_type[0] == HIT_CAR
    # A no-return reads the driver's 65.533 m sentinel, which IS finite, so isfinite() passes for
    # every beam and proved nothing. What "solid" means is that the beam came back from inside the
    # sensor's range; porosity and dropout are off here (rand disabled), so a car beam that does not
    # is a miss.
    returned = r.scan[0][beams] < float(sim.cfg.lidar.range_max)
    assert bool(returned.float().mean() > 0.9), (
        f"only {100 * float(returned.float().mean()):.0f}% of car beams returned in range")
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
    ahead = torch.remainder(s[:, 1] - s[:, 0], L)
    # the learner (slot 0) spawns 2.5-6 m *behind* the teacher: an overtake is a car behind passing
    # a car ahead, and a learner spawned as the leader only ever got overtaken
    assert bool(((ahead > 2.0) & (ahead < 6.5)).all()), ahead
    priv = env.privileged(env.last_result)
    assert priv.shape[1] == 8 + 4 + len(env.PRIV_PARAMS) + 1
    seen = 0
    for _ in range(60):
        a = torch.zeros(8, 2, device=dev); a[:, 1] = -0.2                  # learners crawl; the opponent is ahead in the scan
        obs, rew, term, trunc, info = env.step(a)
        seen += int((env.last_result.scan_type[env.learner] == HIT_CAR).any())
    assert seen > 0                                                     # a learner saw an opponent in its scan
