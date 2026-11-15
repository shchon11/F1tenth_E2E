"""A map made from a recording is a catalogue entry without a code edit (11-15).

`slam_map --install` used to print `user:<id>`, a name nothing could load. The map registry
(`map_registry`: one folder per map) now feeds `maps.REAL` and `tracks.REGISTRY`, so the map loads
as `real:<id>` and `user:<id>` alike, shows in the catalogue as `real/<slug>`, and keeps the
direction the car drove it. Also here: the scan-matching mapper rebuilds a real venue from scans
cast in it, and the console's box bank follows the chosen map's training world.
"""
import json
import math
import os
import shutil

import numpy as np
import pytest

from f1sim import maps, map_registry, tracks


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """An empty map registry in tmp, and the catalogues re-read from it; restored afterwards."""
    monkeypatch.setenv("F1SIM_MAPS", str(tmp_path / "maps"))
    maps.refresh_registered(); tracks.refresh_registered(force=True)
    yield tmp_path
    monkeypatch.undo()
    maps.refresh_registered(); tracks.refresh_registered(force=True)


def _venue(tmp_path, name="venue"):
    """A real venue as a map_server pair (the map12x16 floor), under another name."""
    src = os.path.join(maps.ASSET_MAPS, "map12x16")
    d = tmp_path / "src"
    d.mkdir(exist_ok=True)
    shutil.copy(src + ".pgm", d / f"{name}.pgm")
    (d / f"{name}.yaml").write_text(open(src + ".yaml").read().replace("map12x16.pgm", f"{name}.pgm"))
    return str(d / f"{name}.yaml")


def _area(xy):
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def test_an_installed_map_loads_as_real_and_user_and_by_its_short_id(registry):
    from f1sim import slam_map
    slam_map.install(_venue(registry, "venue_t1"), "venue_t1", note="test venue", display="시험 장소")
    assert "venue_t1" in maps.REAL and "venue_t1" not in maps.REAL_BUILTIN
    a, b = maps.load("real:venue_t1"), maps.load("user:venue_t1")
    tid = tracks.id_of_legacy("real:venue_t1")
    assert tid and tid.startswith("real/") and tracks.id_of_legacy("user:venue_t1") == tid
    e = tracks.get(tid)
    assert e.display == "시험 장소" and e.note == maps.REAL["venue_t1"][3] and len(e.slug) <= 12
    c = maps.load(tid)
    assert (a.occupancy == b.occupancy).all() and (a.occupancy == c.occupancy).all()
    assert a.name == b.name == "venue_t1" and a.centerline is not None
    assert np.allclose(a.centerline, c.centerline)
    assert tid in {x.id for x in tracks.catalog(scenes=False)}
    # modifiers and families work as on any real map
    assert maps.load("real:venue_t1+warp2").name == "real:venue_t1+warp2"
    assert tracks.parse(f"{tid}@rev").legacy() == "real:venue_t1~rev"


def test_a_map_the_car_drove_clockwise_traces_clockwise(registry):
    from f1sim import slam_map
    y = _venue(registry, "venue_t2")
    slam_map.install(y, "venue_t2")
    slam_map.install(y, "venue_t2cw", direction="cw")
    ccw, cw = maps.load("real:venue_t2"), maps.load("real:venue_t2cw")
    assert _area(ccw.centerline) > 0 > _area(cw.centerline)


def test_a_registered_id_never_shadows_a_hand_written_one(registry):
    from f1sim import slam_map
    before = maps.REAL["map12x16"]
    slam_map.install(_venue(registry, "map12x16"), "map12x16")
    assert maps.REAL["map12x16"] == before
    assert [e.id for e in tracks.catalog(scenes=False)].count("real/lab12x16") == 1


def test_scan_mapper_rebuilds_a_real_venue_from_scans_cast_in_it():
    """Scans and drifting odometry simulated in the map12x16 floor; the mapper must give the floor back."""
    from scipy import ndimage
    from f1sim import scan_mapper as SM
    ref = SM.GridMap.load(os.path.join(maps.ASSET_MAPS, "map12x16.yaml"))
    solid = ~ref.free
    edt = ndimage.distance_transform_edt(~solid) * ref.res
    t = maps.load("real:map12x16")
    cl = t.centerline
    L = float(np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1).sum())
    dt, v = 0.025, 3.0
    s = np.arange(0.0, 2.2 * L, v * dt)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([cl, cl[:1]]), axis=0), axis=1))]
    xy = np.stack([np.interp(s % L, seg, np.r_[cl[:, 0], cl[0, 0]]), np.interp(s % L, seg, np.r_[cl[:, 1], cl[0, 1]])], 1)
    yaw = np.unwrap(np.arctan2(*np.gradient(xy, axis=0)[:, ::-1].T))
    ang = np.deg2rad(np.arange(-135, 135.5, 1.0))
    K = len(s)
    ranges = np.full((K, len(ang)), np.inf, np.float32)
    for k in range(K):                                       # sphere tracing on the distance field
        th = yaw[k] + ang
        r = np.zeros(len(ang))
        for _ in range(80):
            px, py = xy[k, 0] + r * np.cos(th), xy[k, 1] + r * np.sin(th)
            j = np.clip(((px - ref.x0) / ref.res).astype(int), 0, edt.shape[1] - 1)
            i = np.clip(((py - ref.y0) / ref.res).astype(int), 0, edt.shape[0] - 1)
            d = edt[i, j]
            r = np.minimum(r + np.where(d > 0, np.maximum(d - 0.5 * ref.res, 0.01), 0.0), 20.0)
        ranges[k] = np.where(r < 15.0, r, np.inf)
    rng = np.random.default_rng(0)
    ranges += rng.normal(0, 0.01, ranges.shape).astype(np.float32)
    # wheel odometry 3 % long with a yaw-rate bias: it must not end up in the map
    w_true = np.gradient(yaw) / dt
    w_odo = w_true + 0.03
    yo = yaw[0] + np.r_[0, np.cumsum(w_odo[:-1] * dt)]
    step = 1.03 * v * dt
    xo = xy[0, 0] + np.r_[0, np.cumsum(step * np.cos(yo[:-1]))]
    yo_ = xy[0, 1] + np.r_[0, np.cumsum(step * np.sin(yo[:-1]))]
    tt = np.arange(K) * dt
    scans = SM.Scans(t=tt, ranges=ranges, angle_min=float(ang[0]), angle_inc=float(ang[1] - ang[0]))
    odom = SM.Odom(t=tt, x=xo, y=yo_, yaw=yo, v=np.full(K, 1.03 * v), w=w_odo)
    res = SM.build_map(scans, odom, match_step=2, log=lambda *a: None)
    out = SM.GridMap(free=np.flipud(res.image()) == 254, occ=np.flipud(res.image()) == 0, x0=res.x0, y0=res.y0, res=res.res)
    T, sc, inl = SM.align(out, ref, guess=np.zeros(3))
    cmp_ = SM.compare(out, ref, SM.align(out, ref, guess=np.zeros(3), fit_scale=False)[0], 1.0)
    assert abs(sc - 1.0) < 0.02, sc
    assert cmp_["lane_iou"] > 0.85, cmp_
    assert cmp_["wall_mean_cm"] < 10.0, cmp_
    true_xy = xy[np.searchsorted(tt, res.t)]
    err = np.hypot(*(res.poses[:, :2] - true_xy).T)
    assert np.sqrt(np.mean(err ** 2)) < 0.10, float(np.sqrt(np.mean(err ** 2)))


def test_the_console_box_bank_follows_the_chosen_map(tmp_path, monkeypatch):
    from f1sim.viewer import sim_worker as SW
    cfg = tmp_path / "configs"; cfg.mkdir()
    bank = tmp_path / "bank_k26"; bank.mkdir()
    (bank / "k26.pt").write_bytes(b"")
    (cfg / "plan_world_korea_2026_competition.json").write_text(json.dumps(
        {"tracks": "real:korea_2026_competition", "prop_bank": str(bank)}))
    (cfg / "plan_world_nobank.json").write_text(json.dumps(
        {"tracks": "real:icra2022", "prop_bank": str(tmp_path / "missing")}))
    monkeypatch.setattr(SW, "WORLD_CONFIGS", str(cfg))
    assert SW.world_prop_bank("real:korea_2026_competition") == str(bank)
    assert SW.world_prop_bank("real:icra2022") is None           # its world's bank is not on disk
    assert SW.world_prop_bank("real:map16x07") is None           # no world for it
    assert SW.world_prop_bank("") is None
