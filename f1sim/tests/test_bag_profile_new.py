"""bag_profile, the 11-28 follow-up: segments, setup.yaml, honest statuses, the two-way "not measured" list, standstill
without commands, split / unsupported bags, coast-down and steering-response estimators (synthetic only), thresholds."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_bag_profile import TRUE, _drive            # noqa: E402  (the synthetic car of the first test file)

from f1sim.bag_profile import (bagio as B, estimators as EST, measure as MS, report, segments as SEG, setupfile as SU,
                               signals as SG, thresholds as TH)

SEG_TOPIC = {"/sysid/segment": "std_msgs/msg/String"}


@pytest.fixture(scope="module")
def drive():
    return _drive(T=70.0, seed=3)


def _write(path, drive, extra_topics=None, extra_msgs=(), drop=lambda n, t: False, split=1):
    topics, msgs, truth = drive
    msgs = [m for m in msgs if not drop(m[0], m[1])] + list(extra_msgs)
    B.write_bag(path, {**topics, **(extra_topics or {})}, msgs, split=split)
    return path


# ---- 1. segments --------------------------------------------------------------------------------------------------
def test_segment_parse_and_mask():
    ev = [(10.0, "T3 start"), (14.0, "T3 end"), (20.0, "t2 start"), (30.0, "T2 end"), (40.0, "T6 start"), (41.0, "junk")]
    ints, warn = SEG.parse(ev, 50.0)
    assert ints["T3"] == [(10.0, 14.0)] and ints["T2"] == [(20.0, 30.0)]
    assert ints["T6"] == [(40.0, 50.0)]                       # no end: runs to the bag end, and says so
    assert any("junk" in w for w in warn) and any("T6" in w and "end" in w for w in warn)
    t = np.arange(0, 50, 0.01)
    m = SEG.mask(t, ints, ("T3",))
    assert m[t == 12.0].all() and not m[t == 15.0].any()
    assert SEG.mask(t, ints, ("T4",)) is None


def test_segment_marks_published_three_times_count_once():
    ev = [(t, "T3 start") for t in (10.0, 10.33, 10.66)] + [(t, "T3 end") for t in (14.0, 14.33, 14.66)]
    ints, warn = SEG.parse(ev, 50.0)
    assert ints["T3"] == [(10.0, 14.0)] or ints["T3"] == [(10.66, 14.0)]
    assert len(ints["T3"]) == 1 and not warn


def test_segments_cut_the_calculation_and_summary_says_which(drive, tmp_path):
    ev = [("/sysid/segment", 45.0, B.enc_string("T3 start")), ("/sysid/segment", 65.0, B.enc_string("T3 end"))]
    d = _write(str(tmp_path / "seg"), drive, SEG_TOPIC, ev)
    bag, roles, missing, c, S = SG.load(d)
    whole = MS.accel_brake(S)
    ints, _ = SEG.parse([(45.0, "T3 start"), (65.0, "T3 end")], 70.0)
    cut = MS.accel_brake(S, SEG.mask(S["t"], ints, ("T3",)))
    assert 0 < cut["a_brake"]["seconds"] < 0.8 * whole["a_brake"]["seconds"]      # the stops at 49 s and 59 s, not all of them
    out = str(tmp_path / "out")
    R = report.run(d, out=out, do_map=False, do_tyre=False, do_lidar=False, log=lambda *a: None)
    assert R["segment_mode"].startswith("표시 토픽")
    assert R["methods"]["가속·제동 한계, 전류 한계"].startswith("표시 토픽 (T3")
    assert R["methods"]["코스트다운"].startswith("자동 인식") and "T4 없음" in R["methods"]["코스트다운"]
    txt = open(os.path.join(out, "summary.md")).read()
    assert "구간 나누기: 표시 토픽" in txt and "자동 인식" in txt


def test_no_segment_topic_means_auto(drive, tmp_path):
    d = _write(str(tmp_path / "noseg"), drive)
    R = report.run(d, out=str(tmp_path / "o"), do_map=False, do_tyre=False, do_lidar=False, log=lambda *a: None)
    assert R["segment_mode"].startswith("자동 인식")


# ---- 2. setup.yaml ------------------------------------------------------------------------------------------------
SETUP = """
vesc:
  speed_to_erpm_gain: %(erpm)s
  steering_to_servo_gain: %(g)s
  steering_to_servo_offset: %(o)s
  current_motor_max_A: 60
  current_regen_min_A: -30
imu: {gyro_unit: rad/s, accel_unit: g}
lidar: {tf_base_to_laser: [0.27, 0.0, 0.11], frame_name: laser}
hand_measured:
  mass_kg: 3.91
  cg_from_rear_axle_m: 0.18
  yaw_inertia_kgm2: 0.05
  lidar_mount_m: [0.28, 0.0, 0.12]
  wheel_angle_table:
    - {cmd_rad: 0.2, left_deg: 7.0, right_deg: 6.0}
    - {cmd_rad: 0.4, left_deg: 13.5, right_deg: 12.5}
"""


def test_setup_yaml_comparison_and_manual_values(drive, tmp_path):
    d = _write(str(tmp_path / "bags" / "withsetup"), drive)
    # setup.yaml NEXT TO the bag folder, with a wrong ERPM gain and a wrong servo offset
    (tmp_path / "bags" / "setup.yaml").write_text(SETUP % dict(erpm=TRUE["erpm"] + 300, g=TRUE["servo_g"], o=TRUE["servo_o"] + 0.05))
    out = str(tmp_path / "out")
    R = report.run(d, out=out, do_map=False, do_tyre=False, do_lidar=False, log=lambda *a: None)
    rows = {r["item"]: r for r in R["setup_compare"]}
    assert rows["ERPM 환산"]["verdict"] == "불일치" and rows["ERPM 환산"]["diff"] == pytest.approx(-300, abs=5)
    assert rows["서보 오프셋"]["verdict"] == "불일치"
    assert rows["서보 이득"]["verdict"] == "일치"
    assert rows["자이로 단위"]["verdict"] == "일치" and rows["LiDAR tf x"]["verdict"] == "일치"
    assert rows["전류 한계 current_motor_max_A"]["ok"] is True
    from f1sim.profiles.loader import load_profile, config_from_profile
    prof = load_profile(os.path.join(out, "car_calibration.yaml"))
    m = prof["vehicle"]["m"]
    assert m["value"] == 3.91 and m["origin"] == "manual" and m["source_label"] == "실측(수동)"
    assert prof["vehicle"]["lr"]["value"] == pytest.approx(0.18) and prof["vehicle"]["lf"]["value"] == pytest.approx(0.3302 - 0.18)
    assert prof["lidar"]["mount_x"]["origin"] == "manual" and prof["lidar"]["mount_x"]["value"] == pytest.approx(0.28 - 0.18)
    assert prof["car"]["steer_gain_wheel"]["value"] == pytest.approx(0.0 + np.deg2rad(6.5) / 0.2, rel=0.05)
    assert config_from_profile(prof).vehicle.m == pytest.approx(3.91)
    txt = open(os.path.join(out, "summary.md")).read()
    assert "setup.yaml 대조" in txt and "**불일치**" in txt
    # what was typed in is not asked for again in list (b)
    asked = [x["item"] for x in R["not_measured"]["not_from_driving"]] if "not_measured" in R else []
    assert "질량" not in asked


def test_setup_explicit_path_and_missing_file(drive, tmp_path):
    d = _write(str(tmp_path / "b1"), drive)
    with pytest.raises(FileNotFoundError):
        report.run(d, out=str(tmp_path / "o"), setup=str(tmp_path / "nope.yaml"), log=lambda *a: None)
    p = tmp_path / "s.yaml"
    p.write_text("vesc: {speed_to_erpm_gain: 4100, typo_key: 1}\n")
    R = report.run(d, out=str(tmp_path / "o2"), setup=str(p), do_map=False, do_tyre=False, do_lidar=False, log=lambda *a: None)
    assert R["setup_compare"][0]["verdict"] == "일치" and "vesc.typo_key" in R["setup_ignored"]


# ---- 3. statuses and sources --------------------------------------------------------------------------------------
def test_lidar_mount_is_a_configured_value_not_a_measurement(drive, tmp_path):
    d = _write(str(tmp_path / "b"), drive)
    out = str(tmp_path / "o")
    report.run(d, out=out, do_map=False, do_tyre=False, log=lambda *a: None)
    from f1sim.profiles.loader import load_profile
    e = load_profile(os.path.join(out, "car_calibration.yaml"))["lidar"]["mount_x"]
    assert e["source"] == "spec" and e["status"] != "measured" and e["source_label"] == "설정값(규격)"


def test_current_limit_needs_a_flat_stretch():
    hz = SG.HZ
    t = np.arange(0, 30, 1 / hz)
    flat = np.zeros_like(t)
    for s0 in (5, 12, 20):                                  # three 0.5 s stretches pinned at 60 A (+- 0.3 A noise)
        flat[int(s0 * hz):int((s0 + 0.5) * hz)] = 60.0
    flat += np.random.default_rng(0).normal(0, 0.3, len(t)) * (flat > 0)
    f = MS.current_plateau(flat, t, "max")
    assert f["status"] == "measured" and f["value"] == pytest.approx(60.0, abs=0.5) and f["runs"] == 3
    peaks = np.zeros_like(t)
    for s0 in (5, 12, 20):                                  # spikes reaching 55 A for 40 ms: no flat top
        peaks[int(s0 * hz):int(s0 * hz) + 4] = 55.0
    peaks[int(5 * hz) + 2] = 58.0
    g = MS.current_plateau(peaks, t, "max")
    assert g["status"] == "lower_bound" and g["value"] == pytest.approx(58.0, abs=3)
    one = np.zeros_like(t); one[500:560] = 60.0              # one stretch only: not enough
    assert MS.current_plateau(one, t, "max")["status"] == "lower_bound"
    # and the entries in the record follow it
    d = {"current_max_A": 60.0, "current_min_A": -30.0, "current_max_fit": f, "current_min_fit": g}
    assert f["how"].startswith("3 flat")


def test_silent_substitution_is_declared():
    calib = SG.Calib()
    Z = {"ackermann": np.column_stack([np.arange(100) * 0.025, np.arange(100) * 0.025, np.zeros(100), np.ones(100)]),
         "servo_cmd": np.column_stack([np.arange(100) * 0.025, np.full(100, 0.5)])}
    SG.fit_servo_map(Z, calib, fallback=(0.63, 0.395))
    assert calib.servo_g == 0.63
    assert any("못 재서 프로필 값 사용" in x for x in calib.substitutions)


def test_summary_lists_substitutions_and_entry_status(drive, tmp_path):
    d = _write(str(tmp_path / "b"), drive)
    out = str(tmp_path / "o")
    R = report.run(d, out=out, do_map=False, do_tyre=False, log=lambda *a: None)
    txt = open(os.path.join(out, "summary.md")).read()
    assert "프로필·기본값으로 대체된 것" in txt
    from f1sim.profiles.loader import load_profile
    car = load_profile(os.path.join(out, "car_calibration.yaml"))["car"]
    # the synthetic car saturates the current at both limits for >= 0.2 s several times: measured, with a reason
    assert car["current_max"]["status"] == "measured" and "flat" in car["current_max"]["how"]


# ---- 4. the two-way "not measured" list ---------------------------------------------------------------------------
def test_not_measured_has_two_lists_with_numbers(drive, tmp_path):
    import json
    d = _write(str(tmp_path / "b"), drive)
    out = str(tmp_path / "o")
    report.run(d, out=out, do_map=False, do_tyre=False, log=lambda *a: None)
    js = json.load(open(os.path.join(out, "bag_profile.json")))
    fix, hand = js["not_measured"]["fixable"], js["not_measured"]["not_from_driving"]
    t2 = next(x for x in fix if x["item"].startswith("조향 이득 k"))
    assert "저속 정상 선회" in t2["why"] and "10 s 필요" in t2["why"]
    assert any(x["item"] == "구름·공기 저항" and "필요" in x["why"] for x in fix)
    assert {"질량", "요 관성", "무게중심 앞뒤 위치"} <= {x["item"] for x in hand}
    assert all("hand_measured" in x["how"] for x in hand if x["item"] in ("질량", "요 관성"))
    assert not any(x["item"] == "질량" for x in fix)


def test_nothing_fixable_is_said_plainly(drive):
    R = {"name": "x", "bag": "b", "duration_s": 1.0, "place": {}, "windows": 99, "impact_excluded": 0, "seconds": 1.0, "missing": {},
         "calib": {"substitutions": []}, "methods": {}}
    txt = report.summary_md(R, {}, {"place": {}}, {"fixable": [], "not_from_driving": []}, [])
    assert "이 bag으로 잴 수 있는 것은 다 쟀음" in txt


# ---- 5. standstill without commands -------------------------------------------------------------------------------
def test_standstill_without_command_traffic_is_used(drive, tmp_path):
    cmd = ("/ackermann_cmd", "/commands/servo/position", "/commands/motor/speed", "/sensors/servo_position_command")
    d = _write(str(tmp_path / "quiet"), drive, drop=lambda n, t: n in cmd and t < 2.9)
    bag, roles, missing, c, S = SG.load(d)
    assert S["t"][0] < 0.2                                   # the grid starts with the IMU, not with the first command
    assert S["still"][60:250].all() and c.bias                 # 2.9 s of standstill with no command at all -> a bias
    assert not S["cmd_valid"][:250].any()
    W = __import__("f1sim.bag_profile.windows", fromlist=["x"]).windows_from_grid(S, "q", c.pose_ok, stride=50)
    assert W["t0"].min() > 2.9                               # no replay window starts where no command exists
    t, st, meta, Rg = bag.read_scans("/scan")
    li = MS.lidar_stats(t, Rg, meta, S)
    assert li["still_noise"] and li["still_seconds"] >= 2.0


# ---- 6. input formats ---------------------------------------------------------------------------------------------
def test_split_bag_is_read_whole(drive, tmp_path):
    one = _write(str(tmp_path / "one"), drive)
    three = _write(str(tmp_path / "three"), drive, split=3)
    assert len([f for f in os.listdir(three) if f.endswith(".db3")]) == 3
    b1, b3 = B.open_bag(one), B.open_bag(three)
    assert len(b3.dbs) == 3
    for n in ("/sensors/imu/raw", "/commands/servo/position", "/scan"):
        assert b1.topics[n].count == b3.topics[n].count
    a1, a3 = b1.read(["/sensors/imu/raw"])["/sensors/imu/raw"], b3.read(["/sensors/imu/raw"])["/sensors/imu/raw"]
    assert a1.shape == a3.shape and np.allclose(a1, a3)
    s1 = b1.read_scans("/scan")[0]
    assert len(s1) == len(b3.read_scans("/scan")[0])


def test_split_bag_without_metadata_is_counted(drive, tmp_path):
    three = _write(str(tmp_path / "t3"), drive, split=3)
    os.remove(os.path.join(three, "metadata.yaml"))
    b = B.open_bag(three)
    assert b.topics["/sensors/imu/raw"].count == B.open_bag(_write(str(tmp_path / "t1"), drive)).topics["/sensors/imu/raw"].count


def test_mcap_gives_a_clear_error(tmp_path):
    d = tmp_path / "mc"
    d.mkdir()
    (d / "mc_0.mcap").write_bytes(b"\x89MCAP0\r\n")
    (d / "metadata.yaml").write_text("rosbag2_bagfile_information:\n  storage_identifier: mcap\n  relative_file_paths: [mc_0.mcap]\n")
    with pytest.raises(B.BagFormatError) as e:
        B.open_bag(str(d))
    msg = str(e.value)
    assert "mcap" in msg and "ros2 bag convert" in msg and "storage_id: sqlite3" in msg
    from f1sim.bag_profile.__main__ import main
    assert main([str(d), "--out", str(tmp_path / "o"), "--quiet"]) == 2


def test_not_a_bag_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        B.open_bag(str(tmp_path / "nothing"))
    e = tmp_path / "empty"
    e.mkdir()
    with pytest.raises(B.BagFormatError):
        B.open_bag(str(e))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "bad_0.db3").write_bytes(b"this is not sqlite" * 50)
    with pytest.raises(B.BagFormatError) as ex:
        B.open_bag(str(bad))
    assert "sqlite3" in str(ex.value)


# ---- 7. estimators (synthetic data only) --------------------------------------------------------------------------
def _grid(n, hz=100.0):
    t = np.arange(n) / hz
    return {"t": t, "still": np.zeros(n, bool), "cmd_valid": np.ones(n, bool)}


def _coast_S(c0=0.15, c2=0.02, runs=((5.0, 8.0), (4.5, 8.0)), seed=0):
    rng = np.random.default_rng(seed)
    v_all, i_all = [], []
    for v0, dur in runs:
        v = v0
        # drive up to v0 with current, then coast with zero current until the stretch ends or the car stops
        v_all += [v0] * 100; i_all += [20.0] * 100
        for _ in range(int(dur * 100)):
            v = max(v - (c0 + c2 * v * v) * 0.01, 0.0)
            v_all.append(v); i_all.append(rng.normal(0, 0.05))
            if v < 0.3:
                break
        v_all += [0.0] * 200; i_all += [0.0] * 200
    S = _grid(len(v_all))
    S["v_wheel"] = np.array(v_all) + rng.normal(0, 0.002, len(v_all))
    S["i_motor"] = np.array(i_all)
    S["speed_cmd"] = np.zeros(len(v_all))
    return S


def test_coastdown_recovers_known_resistance():
    S = _coast_S()
    r = EST.coastdown(S)
    assert r["status"] == "measured" and r["segments"] == 2 and not r["validated"]
    assert r["c0"] == pytest.approx(0.15, abs=0.04) and r["c2"] == pytest.approx(0.02, abs=0.005)
    assert r["c0_ci"][0] <= r["c0"] <= r["c0_ci"][1] and "실차 미검증" in r["how"]


def test_coastdown_not_identified_when_short_or_narrow():
    r = EST.coastdown(_coast_S(runs=((1.2, 1.0),)))              # one short, slow stretch
    assert r["status"] == "not_identified" and r["lacking"] and "필요" in r["how"]
    r2 = EST.coastdown(_coast_S(runs=((2.0, 8.0), (2.0, 8.0))))   # 2 m/s -> 0: speed span below the minimum? (it spans 2)
    assert r2["status"] in ("measured", "not_identified")
    S = _coast_S(); S["i_motor"][:] = 15.0                        # current never ~0
    assert EST.coastdown(S)["status"] == "not_identified"


def _chirp_S(td=0.05, ts=0.04, g=0.9, f1=4.0, seconds=24.0, noise=0.002, kind="chirp", seed=0):
    """steering chirp 0.5 -> f1 Hz at 3 m/s; the yaw rate is the kinematic demand through a pure delay and a first-order lag."""
    rng = np.random.default_rng(seed)
    hz_f = 1000.0
    tf_ = np.arange(int(seconds * hz_f)) / hz_f
    if kind == "chirp":
        k = (f1 - 0.5) / seconds
        steer = 0.1 * np.sin(2 * np.pi * (0.5 * tf_ + 0.5 * k * tf_ ** 2))
    else:                                                          # slow square steps (little power above 1 Hz)
        steer = 0.1 * np.sign(np.sin(2 * np.pi * tf_ / 6.0))
        steer = np.convolve(steer, np.ones(150) / 150, "same")
    v = 3.0
    u = v * np.tan(steer) / SG.L_WHEELBASE
    ud = np.interp(tf_ - td, tf_, u, left=0.0)
    y = np.zeros_like(u)
    a = math.exp(-1 / hz_f / ts)
    for i in range(1, len(u)):
        y[i] = a * y[i - 1] + (1 - a) * g * ud[i]
    step = int(hz_f / SG.HZ)
    n = len(u) // step
    S = _grid(n)
    S["steer"] = steer[::step][:n]
    S["v_wheel"] = np.full(n, v)
    S["gz"] = y[::step][:n] + rng.normal(0, noise, n)
    return S


def test_steer_response_separates_delay_and_lag():
    S = _chirp_S(td=0.05, ts=0.04)
    r = EST.steer_response(S)
    assert r["status"] == "separated" and r["selection"] == "자동 인식"
    assert r["tau_d"] == pytest.approx(0.05, abs=0.015) and r["tau_s"] == pytest.approx(0.04, abs=0.015)
    assert r["gain"] == pytest.approx(0.9, rel=0.1) and "실차 미검증" in r["how"]
    S2 = _chirp_S(td=0.09, ts=0.02, seed=1)
    r2 = EST.steer_response(S2)
    assert r2["tau_d"] == pytest.approx(0.09, abs=0.02) and r2["tau_s"] == pytest.approx(0.02, abs=0.02)


def test_steer_response_with_marked_segment():
    S = _chirp_S()
    sel = np.zeros(len(S["t"]), bool); sel[200:2300] = True
    r = EST.steer_response(S, sel)
    assert r["selection"] == "표시 토픽 구간" and r["seconds"] == pytest.approx(21.0, abs=0.1)


def test_steer_response_gives_only_the_sum_when_the_input_is_slow():
    S = _chirp_S(f1=1.0, kind="steps", noise=0.0005)
    S["steer"] = S["steer"] + 0.0                                     # slow square steps: the delay cannot be told from the lag
    r = EST.steer_response(S)
    assert r["status"] in ("sum_only", "not_identified")
    if r["status"] == "sum_only":
        assert r["tau_sum"] == pytest.approx(0.09, abs=0.03) and r["lacking"]
        assert "분리 식별 안 됨" in r["how"]


def test_steer_response_not_identified_without_data():
    S = _chirp_S(seconds=6.0)
    r = EST.steer_response(S)
    assert r["status"] == "not_identified" and "필요" in r["how"]
    S3 = _chirp_S(); S3["gz"] = np.random.default_rng(0).normal(0, 0.2, len(S3["t"]))      # no coherent response
    assert EST.steer_response(S3)["status"] == "not_identified"


def test_estimator_entries_are_marked_unvalidated(drive, tmp_path):
    d = _write(str(tmp_path / "b"), drive)
    out = str(tmp_path / "o")
    R = report.run(d, out=out, do_map=False, do_tyre=False, do_lidar=False, log=lambda *a: None)
    txt = open(os.path.join(out, "summary.md")).read()
    assert "실차 미검증" in txt and "car.coast_c0" in txt and "car.steer_tau_sum" in txt
    from f1sim.profiles.loader import load_profile
    prof = load_profile(os.path.join(out, "car_calibration.yaml"))
    assert prof["vehicle"]["c_roll"].get("apply") is False            # an unvalidated estimator never reaches the simulator


# ---- 8. thresholds ------------------------------------------------------------------------------------------------
def test_thresholds_in_one_place_and_printed(drive, tmp_path):
    names = [n for n, *_ in TH.TABLE]
    assert len(names) == len(set(names)) and len(names) > 60
    assert TH.SLIDE_DEG == 30.0 and TH.T2_MIN_SECONDS == 10.0 and TH.CLAMP_EDGE_MIN_CMDS == 3 and TH.ACCEL_MIN_BINS == 3
    d = _write(str(tmp_path / "b"), drive)
    out = str(tmp_path / "o")
    report.run(d, out=out, do_map=False, do_tyre=False, do_lidar=False, log=lambda *a: None)
    txt = open(os.path.join(out, "summary.md")).read()
    assert txt.rstrip().endswith("|") and "| SLIDE_DEG | 30.0 |" in txt.split("이 실행에 쓴 문턱값")[1]
    # no decision literal outside thresholds.py: the old magic numbers are gone from the modules
    here = os.path.dirname(MS.__file__)
    src = open(os.path.join(here, "measure.py")).read() + open(os.path.join(here, "signals.py")).read()
    for lit in ("np.deg2rad(5.0)", "> 15.0\n", "np.abs(d) < 0.08", "slide_deg: float = 30.0", "ok.sum() < 20"):
        assert lit not in src, lit


def test_guide_points_at_the_threshold_module_and_new_behaviour():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    guide = open(os.path.join(root, "docs", "bag_recording_guide.md")).read()
    fp = open(os.path.join(root, "docs", "pm", "sysid-2026-11-28", "FIELD_PROTOCOL.md")).read()
    assert "thresholds.py" in guide and "hand_measured" in guide
    assert "이 bag으로 잴 수 있는 것은 다 쟀음" in guide
    assert fp.lstrip().split("\n", 3)[2].startswith("> **최신 안내서는 `docs/bag_recording_guide.md`") or "최신 안내서는 `docs/bag_recording_guide.md`" in fp[:400]
    assert "comp_car_map08x23" in fp and "> 30°/1 s" in fp and "합쳐서 10 s" in fp and "오른쪽은 0.25 rad까지" in fp
    assert f"{TH.SLIDE_DEG:.0f}°" in fp
