"""Research (2026-10-25): a latched lane check for the speed limit for a car ahead (imported by the research runners when LATCH
is set). Diagnosed on a near-stopped car (slowcar_limit_diag.py): with the slow-car head the reading is right in speed and
distance, but in the last 0.2 s its lateral offset drifts outward (|dy| 0.37 vs a true 0.27 m) and the 0.35 m lane check
lets go of the car 57 % of the time just before the rear-end. The fix: a car the limit already holds stays held while the
head still sees it wholly ahead within LATCH m sideways (default 0.5); it is released when it leaves that, is no longer
wholly ahead, or is not seen. Patches `f1sim.opponent_speed_limit.speed_limit`, which `F1VecEnv.step` imports every step."""
import dataclasses
import os

import torch

from f1sim import opponent_speed_limit as osl

EXIT = float(os.environ.get("LATCH", "0.5"))
_orig = osl.speed_limit
STATE = {"held": None}


def speed_limit(reading, v_me, mount_x=0.297, mount_y=0.0, p=osl.DEFAULT):
    base = _orig(reading, v_me, mount_x, mount_y, p)
    wide = _orig(reading, v_me, mount_x, mount_y, dataclasses.replace(p, lane_half_width=EXIT))
    held = STATE["held"]
    if held is None or held.shape != base.shape or held.device != base.device:
        held = torch.zeros(base.shape, dtype=torch.bool, device=base.device)
    lim = torch.where(held, wide, base)
    STATE["held"] = torch.isfinite(lim)
    return lim


osl.speed_limit = speed_limit
