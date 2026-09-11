"""Several tracks in one simulator batch: per-env grids, centerlines, racelines."""
import numpy as np
import torch

from f1sim import Track, Config, Simulator, maps
from f1sim.learn.common import TRAIN_TRACKS
from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher


def test_training_set_covers_the_diversity_axes() -> None:
    """The set has to cover the axes the real venues occupy, not hit a particular size.

    Counting tracks says nothing: the previous set was 112 names of which 32 came from one
    generator whose own docstring says more seeds add little new geometry, while the layout every
    real venue actually has -- a lap that folds back beside itself behind one hose -- appeared in
    none of them. Measured, procedural tracks never came within 5.8 m of themselves against 2.4-3.3 m
    for the real ones, held their width to a 1.05 pinch ratio against 1.35-1.98, and carried no
    obstacle on the racing line at all.

    So this asserts coverage of each axis, and asserts nothing about the total.
    """
    n = len(TRAIN_TRACKS)
    def share(pred):
        return sum(bool(pred(t)) for t in TRAIN_TRACKS) / n

    # Fold-back -- a lap running beside itself behind one hose -- is the axis the real venues have
    # and no generator did. `serpentine` was written for it and is held back until the centerline
    # projection can carry procedural folds (see common.GEN_TRAIN), so the exposure comes from the
    # real maps that fold: icra2022 at 2.48 m, korea_2026 at 2.36 m.
    assert any("icra2022" in t for t in TRAIN_TRACKS)
    assert share(lambda t: "+pinch" in t) > 0.05                     # the lane closes down
    assert share(lambda t: t.startswith("gen:control:")) > 0.05      # hairpins, chicanes, width
    assert share(lambda t: "+rlobs" in t) > 0.10                     # obstacles ON the racing line
    assert share(lambda t: "obs" in t) > 0.20                        # obstacles of either kind
    assert share(lambda t: t.startswith("real:")) > 0.25             # real venue geometry
    # no single generator may dominate the way gen:competition used to
    for fam in ("gen:competition:", "gen:hallway:", "gen:circuit:", "gen:control:"):
        assert share(lambda t, f=fam: t.startswith(f)) < 0.30, fam
    # hallway scans are near-indistinguishable between far-apart places (aliasing 0.087 against
    # 0.28-0.58 elsewhere): keep it present but small
    assert share(lambda t: t.startswith("gen:hallway:")) < 0.10
    assert len(set(TRAIN_TRACKS)) == n                               # no duplicates


def test_duplicated_track_matches_single_track():
    tr = Track.generate_random(3, style="competition")
    cfg = Config(); cfg.rand.enabled = False; cfg.lidar.noise_std = 0.0; cfg.lidar.dropout_prob = 0.0
    cfg.imu.enabled = False
    single = Simulator(tr, cfg, num_envs=4, device="cpu")
    multi = Simulator([tr, tr], cfg, num_envs=4, device="cpu", track_ids=[0, 1, 0, 1])
    s0 = torch.tensor([1.0, 5.0, 9.0, 13.0])
    p = single.sample_spawn(4, 0.0, 0.0, s0); single.reset(poses=p); multi.reset(poses=p)
    a = torch.tensor([[0.05, 2.0]] * 4)
    for _ in range(30):
        r1 = single.step(a); r2 = multi.step(a)
    assert torch.allclose(r1.state, r2.state, atol=1e-5)
    assert torch.allclose(r1.scan_true, r2.scan_true, atol=1e-4)
    assert torch.allclose(r1.s, r2.s, atol=1e-4) and torch.equal(r1.collision, r2.collision)


def test_two_different_tracks_in_one_batch_and_teacher_laps():
    tracks = [Track.generate_random(0, style="competition"), Track.generate_random(0, style="hallway")]
    rls = [Raceline.build(t) for t in tracks]
    cfg = Config(); cfg.rand.enabled = False
    B = 6
    sim = Simulator(tracks, cfg, num_envs=B, device="cpu", track_ids=[0, 1] * 3)
    sim.reset(poses=sim.sample_spawn(B, 0.0, 0.0, torch.zeros(B), tid=sim.tid))
    L = sim.track.length[sim.tid]
    assert abs(L[0] - L[1]) > 1.0 and torch.allclose(L[0], L[2])          # per-env lap lengths
    teacher = RacelineTeacher(rls, wheelbase=cfg.vehicle.lf + cfg.vehicle.lr)
    r = None; total = torch.zeros(B)
    for i in range(int(2.0 * L.max() / 3.0 / sim.control_dt)):
        r = sim.step(teacher(r.state if r is not None else sim.state, sim.P, sim.tid))
        total += r.progress
        assert not r.collision.any(), (i, r.collision, sim.tid)
        if (r.lap >= 1).all():
            break
    assert (r.lap >= 1).all(), (r.lap, total)


def test_catalog_loading_racetrack_and_gym_map():
    sp = maps.load("rt:Spielberg")
    assert sp.centerline is not None and 300 < float(np.linalg.norm(np.roll(sp.centerline, -1, 0) - sp.centerline, axis=1).sum()) < 450
    lv = maps.load("gym:levine")
    assert lv.tall.sum() > 0.5 * lv.occupancy.size and (~lv.occupancy).sum() > 40000   # cropped to free bbox + 2 m
    sim = Simulator([sp, lv], Config(), num_envs=2, device="cpu")
    assert sim.track.dtype == (torch.float16 if sim.track.edt.numel() > 6e6 else torch.float32)   # big sets -> half
    pose = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # spawn on Spielberg's centerline; levine has no centerline -> most open cell
    sim.reset(poses=torch.stack([sim.sample_spawn(1, 0.0, 0.0, torch.zeros(1), tid=torch.zeros(1, dtype=torch.long))[0],
                                 sim.sample_spawn(1, 0.0, 0.0, tid=torch.ones(1, dtype=torch.long))[0]]))
    for _ in range(20):
        r = sim.step(torch.tensor([[0.0, 2.0], [0.0, 1.0]]))
    assert torch.isfinite(r.state).all() and not r.collision.any()
