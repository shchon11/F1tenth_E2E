"""bag_profile: synthetic recordings with KNOWN values -> the tool must give them back.

The synthetic car is integrated here, independently of `bag_profile.models` (its own explicit single-track loop at
1 ms), so a shared bug in the model cannot make the round trip pass by construction.
"""
import math
import os

import numpy as np
import pytest

from f1sim.bag_profile import bagio as B, fit as F, measure as MS, models as M, signals as SG, windows as WN

G = 9.80665
TRUE = dict(servo_g=0.62, servo_o=0.41, win=(-0.26, 0.40), erpm=4100.0, delay=0.06, tau_s=0.03, k=1.1, b=-0.02,
            sf=3.0, sr=6.0, mu=1.4, C=1.3, a_brake=4.0)


def _drive(T=70.0, seed=0, gyro_deg=False, accel_g=True):
    """A car on an open floor driven by a smooth random steering / speed program. Returns the message list."""
    rng = np.random.default_rng(seed)
    dt = 0.001
    n = int(T / dt)
    t = np.arange(n) * dt
    # command program at 40 Hz: smoothed random steering, speed ramps with hard brakes, a 3 s standstill at start
    tc = np.arange(0, T, 0.025)
    st = np.convolve(rng.normal(0, 0.25, len(tc)), np.ones(30) / 30, "same") * 1.8
    st += 0.35 * np.sin(2 * np.pi * tc / 7.0)
    vcmd = 3.0 + 2.5 * np.sin(2 * np.pi * tc / 11.0) + 1.0 * np.sin(2 * np.pi * tc / 3.1)
    vcmd = np.clip(vcmd, 1.0, 6.5)
    vcmd[(tc % 10.0) > 9.0] = 1.0                         # a hard stop every 10 s: ~1 s at the regen limit
    st[(tc % 8.0) > 6.0] = 0.0                            # a 2 s straight every 8 s (what the trim is read from)
    vcmd[tc < 3.0] = 0.0
    st[tc < 3.0] = 0.0
    # servo map + VESC clamp: the APPLIED command is the clamped one
    servo_cmd = TRUE["servo_g"] * st + TRUE["servo_o"]
    lo_s = TRUE["servo_g"] * TRUE["win"][0] + TRUE["servo_o"]
    hi_s = TRUE["servo_g"] * TRUE["win"][1] + TRUE["servo_o"]
    servo_app = np.clip(servo_cmd, lo_s, hi_s)
    st_app = (servo_app - TRUE["servo_o"]) / TRUE["servo_g"]
    # integrate
    lf, lr, iz = M.LF, M.LR, M.IZ_M
    L = lf + lr
    B_ = TRUE["sf"] / (TRUE["mu"] * TRUE["C"]); Br = TRUE["sr"] / (TRUE["mu"] * TRUE["C"])
    x = y = yaw = vy = r = delta = 0.0
    v = 0.0
    out = dict(t=[], r=[], v=[], ay=[], ax=[], x=[], y=[], yaw=[], cur=[])
    for i in range(n):
        ti = t[i]
        j = min(int(max(ti - TRUE["delay"], 0) / 0.025), len(tc) - 1)
        tgt = TRUE["k"] * st_app[j] + TRUE["b"]
        delta += (tgt - delta) * (1 - math.exp(-dt / TRUE["tau_s"]))
        jv = min(int(max(ti - 0.1, 0) / 0.025), len(tc) - 1)
        a = np.clip((vcmd[jv] - v) / 0.15, -TRUE["a_brake"], 4.5)
        cur = 60.0 * a / 4.5 if a > 0 else 30.0 * a / TRUE["a_brake"]     # current ~ torque, saturating at the limits
        v = max(v + a * dt, 0.0)
        vs = max(v, 0.3)
        af = delta - math.atan2(vy + lf * r, vs)
        ar = -math.atan2(vy - lr * r, vs)
        fyf = TRUE["mu"] * 9.81 * lr / L * math.sin(TRUE["C"] * math.atan(B_ * af))
        fyr = TRUE["mu"] * 9.81 * lf / L * math.sin(TRUE["C"] * math.atan(Br * ar))
        vyd_d = fyf * math.cos(delta) + fyr - v * r
        rd_d = (lf * fyf * math.cos(delta) - lr * fyr) / iz
        beta = math.atan(lr / L * math.tan(delta))
        w = min(max((v - 0.5) / 1.0, 0.0), 1.0)              # dynamic above 1.5 m/s, kinematic below 0.5
        vyd = w * vyd_d + (1 - w) * 20.0 * (v * math.tan(beta) - vy)
        rd = w * rd_d + (1 - w) * 20.0 * (v * math.cos(beta) * math.tan(delta) / L - r)
        vy += vyd * dt
        r += rd * dt
        yaw += r * dt
        x += (v * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (v * math.sin(yaw) + vy * math.cos(yaw)) * dt
        if i % 10 == 0:                                   # 100 Hz truth
            for k_, val in (("t", ti), ("r", r), ("v", v), ("ay", vyd + v * r), ("ax", a), ("x", x - lr * math.cos(yaw)),
                            ("y", y - lr * math.sin(yaw)), ("yaw", yaw), ("cur", cur)):
                out[k_].append(val)
    truth = {k_: np.array(v_) for k_, v_ in out.items()}
    # messages
    msgs = []
    for i, ti in enumerate(tc):
        msgs.append(("/ackermann_cmd", ti, B.enc_ackermann(ti, float(st[i]), float(vcmd[i]))))
        msgs.append(("/commands/servo/position", ti + 0.002, B.enc_float64(float(servo_cmd[i]))))
        msgs.append(("/commands/motor/speed", ti + 0.002, B.enc_float64(float(vcmd[i] * TRUE["erpm"]))))
        msgs.append(("/sensors/servo_position_command", ti + 0.004, B.enc_float64(float(servo_app[i]))))
    tt = truth["t"]
    for i in range(0, len(tt), 2):                        # 50 Hz IMU / VESC / odom / pose
        ti = tt[i]
        wz = truth["r"][i] * (180 / math.pi if gyro_deg else 1.0) + rng.normal(0, 0.003)
        sc = 1 / G if accel_g else 1.0
        acc = (truth["ax"][i] * sc, truth["ay"][i] * sc, G * sc)
        msgs.append(("/sensors/imu/raw", ti, B.enc_imu(ti, (0.0, 0.0, wz), acc)))
        erpm = round(truth["v"][i] * TRUE["erpm"])
        msgs.append(("/sensors/core", ti + 0.001, B.enc_vesc_state(ti, float(erpm), float(truth["cur"][i]))))
        msgs.append(("/odom", ti + 0.002, B.enc_odom(ti, 0.0, 0.0, 0.0, erpm / TRUE["erpm"])))
        msgs.append(("/pf/pose/odom", ti + 0.003, B.enc_odom(ti, truth["x"][i], truth["y"][i], truth["yaw"][i], truth["v"][i])))
    for i in range(0, len(tt), 10):                       # 10 Hz scans: a wall ring, a few short dropouts
        ranges = np.full(1081, 5.0, np.float32) + rng.normal(0, 0.006, 1081).astype(np.float32)
        ranges[rng.integers(0, 1081, 3)] = 65.533
        msgs.append(("/scan", tt[i] + 0.004, B.enc_scan(tt[i], ranges)))
    msgs.append(("/tf_static", 0.0, B.enc_tf([("base_link", "laser", 0.27, 0.0, 0.11)])))
    topics = {"/ackermann_cmd": "ackermann_msgs/msg/AckermannDriveStamped", "/commands/servo/position": "std_msgs/msg/Float64",
              "/commands/motor/speed": "std_msgs/msg/Float64", "/sensors/servo_position_command": "std_msgs/msg/Float64",
              "/sensors/imu/raw": "sensor_msgs/msg/Imu", "/sensors/core": "vesc_msgs/msg/VescStateStamped",
              "/odom": "nav_msgs/msg/Odometry", "/pf/pose/odom": "nav_msgs/msg/Odometry", "/scan": "sensor_msgs/msg/LaserScan",
              "/tf_static": "tf2_msgs/msg/TFMessage"}
    return topics, msgs, truth


@pytest.fixture(scope="module")
def synth_bag(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("bags") / "synth_rad")
    topics, msgs, truth = _drive()
    B.write_bag(d, topics, msgs)
    return d, truth


@pytest.fixture(scope="module")
def loaded(synth_bag):
    return SG.load(synth_bag[0])


def test_reader_round_trip(tmp_path):
    d = str(tmp_path / "rt")
    msgs = [("/a", 0.5, B.enc_float64(1.25)), ("/a", 0.25, B.enc_float64(-3.0)),
            ("/d", 0.3, B.enc_ackermann(1.0, 0.1, 2.5)),
            ("/i", 0.4, B.enc_imu(2.0, (0.1, 0.2, 0.3), (1.0, 2.0, 3.0))),
            ("/o", 0.6, B.enc_odom(3.0, 1.0, 2.0, 0.5, 4.0, 0.1, 0.2))]
    B.write_bag(d, {"/a": "std_msgs/msg/Float64", "/d": "ackermann_msgs/msg/AckermannDriveStamped",
                    "/i": "sensor_msgs/msg/Imu", "/o": "nav_msgs/msg/Odometry"}, msgs)
    bag = B.open_bag(d)
    assert bag.topics["/a"].count == 2
    Z = bag.read(["/a", "/d", "/i", "/o"])
    assert np.allclose(Z["/a"][:, 0], [0.25, 0.5]) and np.allclose(Z["/a"][:, 1], [-3.0, 1.25])   # sorted by t
    assert np.allclose(Z["/d"][0, 2:4], [0.1, 2.5], atol=1e-6)
    assert np.allclose(Z["/i"][0, 2:8], [0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
    assert np.allclose(Z["/o"][0, 2:8], [1.0, 2.0, 0.5, 4.0, 0.1, 0.2])


def test_static_calibration_recovered(loaded):
    bag, roles, missing, c, S = loaded
    assert not missing.get("imu") and "vesc" in roles
    assert c.servo_g == pytest.approx(TRUE["servo_g"], abs=1e-3)
    assert c.servo_o == pytest.approx(TRUE["servo_o"], abs=1e-3)
    lo, hi = c.steer_window_rad()
    assert lo == pytest.approx(TRUE["win"][0], abs=2e-3) and hi == pytest.approx(TRUE["win"][1], abs=2e-3)
    assert c.clamp_seen_lo and c.clamp_seen_hi
    assert c.erpm_gain == pytest.approx(TRUE["erpm"], rel=1e-3)
    assert c.gyro_unit == "rad/s" and c.accel_unit == "g"


def test_gyro_in_degrees_is_detected(tmp_path):
    d = str(tmp_path / "synth_deg")
    topics, msgs, truth = _drive(T=40.0, seed=1, gyro_deg=True, accel_g=False)
    B.write_bag(d, topics, msgs)
    bag, roles, missing, c, S = SG.load(d)
    assert c.gyro_unit == "deg/s"
    assert c.accel_unit == "m/s^2"
    mv = S["v_wheel"] > 2
    tr = np.interp(S["t"][mv], truth["t"], truth["r"])
    assert np.median(np.abs(S["r"][mv] - tr)) < 0.05      # back in rad/s


def test_delay_trim_and_brake(loaded):
    bag, roles, missing, c, S = loaded
    d = MS.delay_xcorr(S)
    # apparent lag = pure delay + servo lag + tyre/yaw build-up: at least the pure delay, under 0.2 s here
    assert TRUE["delay"] <= d["value"] <= 0.2
    tr = MS.trim(S, TRUE["k"])
    assert tr["trim_b"] == pytest.approx(TRUE["b"], abs=0.006)
    acc = MS.accel_brake(S)
    assert acc["a_brake"]["value"] == pytest.approx(TRUE["a_brake"], rel=0.1)


def test_replay_fit_recovers_known_values(loaded):
    """(b) single track fitted on the synthetic windows returns the generating delay, gain, trim and stiffness."""
    bag, roles, missing, c, S = loaded
    W = WN.windows_from_grid(S, "synth", c.pose_ok, stride=50)
    assert len(W["r"]) > 60
    base = dict(M.ST_DEFAULT, Cf=TRUE["C"], Cr=TRUE["C"], mf=TRUE["mu"], mr=TRUE["mu"],
                k=0.9, b=0.0, tau_d=0.03, tau_s=0.05, sf=4.0, sr=4.5, tau_m=0.15, tau_v=0.1, a_a=4.5, a_b=TRUE["a_brake"])
    p, J, _ = F.fit(M.st_rollout, W, base, ["k", "b", "tau_d", "tau_s", "sf", "sr"], max_nfev=80)
    assert p["k"] == pytest.approx(TRUE["k"], rel=0.05)
    assert p["b"] == pytest.approx(TRUE["b"], abs=0.004)
    assert p["tau_d"] + p["tau_s"] == pytest.approx(TRUE["delay"] + TRUE["tau_s"], abs=0.015)
    assert p["sf"] == pytest.approx(TRUE["sf"], rel=0.15)
    assert p["sr"] == pytest.approx(TRUE["sr"], rel=0.25)
    assert np.sqrt(J) < 0.05


def test_lidar_stats(synth_bag, loaded):
    bag, roles, missing, c, S = loaded
    t, st, meta, R = bag.read_scans("/scan")
    li = MS.lidar_stats(t, R, meta, S)
    assert li["beams"] == 1081
    assert li["valid_frac"] == pytest.approx(1 - 3 / 1081, abs=1e-3)
    near = [x for x in li["dropout_by_range"] if x["r"] == (4, 6)][0]
    assert near["miss_frac"] == pytest.approx(3 / 1081, rel=0.3)
    sig = [x for x in li["still_noise"] if x["r"] == (4, 6)]
    assert sig and sig[0]["sigma_mm"] == pytest.approx(6.0, rel=0.3)


def test_end_to_end_outputs(synth_bag, tmp_path):
    from f1sim.bag_profile import report
    from f1sim.profiles.loader import load_profile, config_from_profile
    out = str(tmp_path / "out")
    R = report.run(synth_bag[0], out=out, do_map=False, do_tyre=False, log=lambda *a: None)
    for f in ("car_calibration.yaml", "place_profile.yaml", "summary.md", "bag_profile.json"):
        assert os.path.exists(os.path.join(out, f))
    car = load_profile(os.path.join(out, "car_calibration.yaml"))
    assert car["car"]["servo_map_gain"]["value"] == pytest.approx(TRUE["servo_g"], abs=1e-3)
    assert car["lidar"]["mount_x"]["value"] == pytest.approx(0.27 - M.LR, abs=1e-6)
    cfg = config_from_profile(car)
    assert cfg.lidar.mount_x == pytest.approx(0.27 - M.LR, abs=1e-6)
    assert cfg.vehicle.a_brake == pytest.approx(TRUE["a_brake"], rel=0.1)
    assert R["lin_fit"]["p"]["b"] == pytest.approx(TRUE["b"], abs=0.008)
    # (a)'s single delay is the whole command -> yaw lag: at least the pure delay + servo lag, well under 0.2 s
    assert TRUE["delay"] + TRUE["tau_s"] - 0.02 <= R["lin_fit"]["p"]["tau"] <= 0.2


def test_shipped_profiles_load_and_apply():
    from f1sim.profiles.loader import available, load_profile, config_from_profile, unapplied
    from f1sim.params import Config
    names = available()
    assert names, "no profile yaml shipped"
    default = Config()
    for n in names:
        prof = load_profile(n)
        cfg = config_from_profile(prof)
        for grp in ("vehicle", "actuator"):
            for k, e in (prof.get(grp) or {}).items():
                if e.get("apply") is False or e["status"] == "not_applied" or not hasattr(getattr(cfg, grp), k):
                    continue
                assert getattr(getattr(cfg, grp), k) == pytest.approx(e["value"]), (n, grp, k)
        assert isinstance(unapplied(prof), list)
    # the defaults are untouched by loading profiles
    assert Config().vehicle.B_f == default.vehicle.B_f == 2.0


def test_profile_validation_rejects_bad_entries(tmp_path):
    from f1sim.profiles.loader import load_profile
    p = tmp_path / "bad.yaml"
    p.write_text("vehicle:\n  mu: {value: 1.2, source: guessed, status: measured}\n")
    with pytest.raises(ValueError):
        load_profile(str(p))


def test_profile_builds_a_running_simulator():
    """the default profile goes through Config.from_dict into a Simulator that steps (CPU, no compile)."""
    torch = pytest.importorskip("torch")
    from f1sim.profiles.loader import make_simulator, available
    from f1sim.track import Track
    if "default_comp_car_pre_place" not in available():
        pytest.skip("default profile not shipped")
    track = Track.generate_random(4)
    sim = make_simulator("default_comp_car_pre_place", track, num_envs=2, device="cpu",
                         base_dict={"sim": {"compile": False}, "lidar": {"n_beams": 31}, "rand": {"enabled": False}})
    assert sim.cfg.vehicle.h == 0.0 and sim.cfg.vehicle.mu == pytest.approx(1.0)
    sim.reset(poses=sim.sample_spawn(2, lateral_std=0.0, yaw_std=0.0))
    cmd = torch.tensor([[0.1, 2.0], [-0.1, 2.0]])
    for _ in range(40):
        sim.step(cmd)
    assert torch.isfinite(sim.state).all() and float(sim.state[:, 3].min()) > 0.5
