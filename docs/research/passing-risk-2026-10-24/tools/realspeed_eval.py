"""Research (2026-10-26): evaluate with the REAL CAR'S SPEED LOOP. 38 sustained speed-command rises in the 08-26/27 competition
recordings fit an acceleration-capped first-order loop (ActuatorParams: cap 2.5 m/s^2 on the drive side, tau 0.7 s, 0.1 s extra
delay of the speed channel); every policy was trained and evaluated with the fast default (tau 0.1-0.3, no cap, no extra delay),
and the speed limits assume 0.2 s of lag. This runs `evaluate` with that loop switched on (motor_ramp, motor_delay on the nominal
config; tau via --rand-range actuator.motor_tau). usage: realspeed_eval.py <evaluate args...>   env RAMP, DELAY, TAU override."""
import os
import sys

from f1sim.learn import evaluate as ev

if os.environ.get("LIMIT_LAG"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lag_patch  # noqa: F401  -- the speed limits' lag (10-26)

RAMP, DELAY, TAU = float(os.environ.get("RAMP", "2.5")), float(os.environ.get("DELAY", "0.1")), os.environ.get("TAU", "0.7")
_C = ev.Config


def C(*a, **k):
    c = _C(*a, **k)
    c.actuator.motor_ramp = RAMP
    c.actuator.motor_delay = DELAY
    return c


ev.Config = C
# TAU="" keeps the default motor_tau range: the 0.7 s fit is from speed RISES only, and the competition car brakes at the regen
# limit (-30 A: ~5.5 m/s^2, params.py a_brake), which a 0.7 s loop in both directions would not
sys.argv = ["evaluate", *sys.argv[1:], *(["--rand-range", f"actuator.motor_tau={TAU},{TAU}"] if TAU else [])]
ev.main()
