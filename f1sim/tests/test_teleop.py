import math
from f1sim.teleop import DriveModel, Inputs, KeyRamp, TeleopConfig, JoyDecoder, PRESETS


def drive(model, inp, v_follow=True, t=1.0, dt=0.02, v0=0.0):
    v = v0
    for _ in range(int(t / dt)):
        steer, v_cmd = model.update(inp, v, dt)
        if v_follow:
            v += max(-6 * dt, min(6 * dt, v_cmd - v))       # the car follows the target with a lag
    return steer, v_cmd, v


def test_throttle_is_acceleration_not_speed():
    m = DriveModel(TeleopConfig(a_throttle=2.0, v_max=5.0, throttle_gamma=1.8))
    _, v1, v = drive(m, Inputs(throttle=1.0), t=0.5)
    _, v2, v = drive(m, Inputs(throttle=1.0), t=0.5, v0=v)
    assert 0.9 < v1 < 1.1 and abs(v2 - 2 * v1) < 0.05        # 2 m/s^2, keeps building while held
    _, vh, _ = drive(DriveModel(TeleopConfig(a_throttle=2.0, throttle_gamma=1.8)), Inputs(throttle=0.5), t=1.0)
    assert vh < 0.7                                          # half pedal is soft (curve)
    _, v3, v = drive(m, Inputs(throttle=1.0), t=4.0, v0=v)
    assert abs(v3 - 5.0) < 1e-6                              # capped at v_max
    _, v4, _ = drive(m, Inputs(throttle=1.0, boost=True), t=1.0, v0=v)
    assert v4 > 5.5                                          # boost lifts the cap


def test_coast_and_brake():
    m = DriveModel(TeleopConfig())
    _, v_before, v = drive(m, Inputs(throttle=1.0), t=1.0)
    _, v_coast, v = drive(m, Inputs(), t=1.0, v0=v)
    assert v_before - 1.0 < v_coast < v_before - 0.6          # engine braking ~0.8 m/s^2
    _, v_brake, v = drive(m, Inputs(brake=1.0), t=0.3, v0=v)
    assert v_brake < v_coast - 1.2 and v_brake >= 0.0          # brake is much stronger and never below 0


def test_reverse_engages_after_holding_brake_at_standstill():
    m = DriveModel(TeleopConfig())
    steer, v, _ = drive(m, Inputs(brake=1.0), t=0.2, v_follow=False)
    assert not m.reverse and v == 0.0
    _, v, _ = drive(m, Inputs(brake=1.0), t=0.6, v_follow=False)
    assert m.reverse and v < 0.0
    _, v, _ = drive(m, Inputs(brake=1.0), t=2.0, v_follow=False, v0=-1.0)
    assert v >= -1.2 - 1e-6                                   # reverse speed cap
    _, v, _ = drive(m, Inputs(throttle=1.0), t=1.0, v_follow=False, v0=0.0)
    assert not m.reverse and v >= 0.0                         # throttle brings it back to forward gear


def test_speed_sensitive_steering_lock_and_slew():
    m = DriveModel(TeleopConfig(steer_rate=4.0, lock_v_ref=2.5, lock_min=0.35))
    s_slow, _, _ = drive(m, Inputs(steer=1.0), t=1.0, v_follow=False, v0=1.0)
    assert abs(s_slow - m.cfg.steer_max) < 1e-6              # full lock at low speed
    m.reset()
    s_fast, _, _ = drive(m, Inputs(steer=1.0), t=1.0, v_follow=False, v0=8.0)
    assert abs(s_fast - m.cfg.steer_max * 0.35) < 1e-6        # lock shrinks at speed (floor 35 %)
    m.reset()
    s_step, _, _ = drive(m, Inputs(steer=1.0), t=0.05, v_follow=False, v0=0.0)
    assert abs(s_step - 4.0 * 0.04) < 1e-6                    # slew-rate limited: 2 steps of 20 ms at 4 rad/s


def test_target_never_runs_away_when_blocked():
    m = DriveModel(TeleopConfig(max_lead=2.5))
    _, v_cmd, _ = drive(m, Inputs(throttle=1.0), t=3.0, v_follow=False, v0=0.0)   # car stuck at 0
    assert v_cmd <= 2.5 + 1e-6


def test_key_ramp_feels_analog():
    k = KeyRamp(TeleopConfig(key_press_time=0.2, key_release_time=0.1))
    vals = [k.update(False, True, True, False, 0.02).steer for _ in range(5)]
    assert vals[0] == -0.1 and abs(vals[-1] - (-0.5)) < 1e-9   # ramps toward -1 (right) at 1/0.2 per s
    rel = [k.update(False, False, False, False, 0.02).steer for _ in range(3)]
    assert rel[-1] >= vals[-1] + 0.5                            # returns to center faster (reaches 0 within 60 ms)


def test_joy_presets():
    cfg = TeleopConfig()
    d = JoyDecoder(cfg, PRESETS["xbox"])
    inp = d.decode([0.5, 0, 1.0, 0, 0, -1.0], [0, 0, 0, 0, 1, 0])   # half stick left, RT fully pulled, LB held
    assert 0 < inp.steer < 0.5 and inp.throttle == 1.0 and inp.brake == 0.0 and inp.boost and inp.deadman
    inp0 = JoyDecoder(cfg, PRESETS["xbox"]).decode([0, 0, 0, 0, 0, 0], [0] * 6)   # untouched triggers report 0
    assert inp0.throttle == 0.0
    f = JoyDecoder(cfg, PRESETS["f1tenth"])
    inp = f.decode([0, 0.8, -0.3, 0, 0, 0], [0, 0, 0, 0, 0, 0])
    assert not inp.deadman                                     # LB not held -> stop
    inp = f.decode([0, 0.8, -0.3, 0, 0, 0], [0, 0, 0, 0, 1, 0])
    assert inp.deadman and inp.throttle > 0.5 and inp.steer < 0  # stick up = throttle, right stick X right = steer right
