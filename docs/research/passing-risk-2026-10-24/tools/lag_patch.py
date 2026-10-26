"""Research (2026-10-26): the speed limits with another LAG (env LIMIT_LAG, seconds; both limits). They assume 0.2 s from command to
braking; the competition car's speed channel has ~0.1 s more (params.py motor_delay). Imported by realspeed_eval.py when set."""
import os
from dataclasses import replace

LAG = float(os.environ["LIMIT_LAG"])
from f1sim import opponent_speed_limit as osl_car
from f1sim import obstacle_speed_limit as osl_obs

_car = osl_car.speed_limit
_obs = osl_obs.speed_limit
P_CAR = replace(osl_car.DEFAULT, lag=LAG)
P_OBS = replace(osl_obs.DEFAULT, lag=LAG)


def car_limit(reading, v_me, mount_x=0.297, mount_y=0.0, p=None, held=None):
    return _car(reading, v_me, mount_x, mount_y, p=P_CAR, held=held)


def obs_limit(*a, **kw):
    kw["p"] = P_OBS
    return _obs(*a, **kw)


osl_car.speed_limit = car_limit       # the env imports both at call time
osl_obs.speed_limit = obs_limit
print(f"[lag_patch] both speed limits assume {LAG:.2f} s of lag", flush=True)
