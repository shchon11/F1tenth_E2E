"""Open-loop replay windows: 0.5 s warm-up history + 2.0 s free prediction at 100 Hz.

The same rules as the 11-28 audit (`docs/pm/audit-2026-11-28/D_sim_fidelity.md` section 3.1), so its tables and these
are comparable: start every 0.25 s; speed >= 1 m/s at the start and >= 0.3 m/s throughout; no command gap > 0.1 s; no
impact (0.05 s mean |a| > 15 m/s^2) anywhere in the window (no model here has walls).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import thresholds as TH
from .signals import HZ, smooth

WARM, HOR = 50, 200
KEYS = ("steer", "steer_cmd", "speed_cmd", "r", "v", "ay", "ax", "i_motor")


def windows_from_grid(S: Dict[str, np.ndarray], bag_name: str, pose_ok: bool, stride: int = int(TH.WIN_STRIDE_S * HZ),
                      v_min: float = TH.WIN_MIN_V_START, warm: int = WARM, hor: int = HOR) -> Dict[str, np.ndarray]:
    n = len(S["t"])
    a_imp = smooth(np.hypot(S["ax"], S["ay"]), 5)
    imp = a_imp > TH.IMPACT_ACC
    v = S["v_wheel"]
    src = dict(steer=S["steer"], steer_cmd=S["steer_cmd"], speed_cmd=S.get("speed_cmd", np.full(n, np.nan)),
               r=S["r"], v=v, ay=S["ay_s"], ax=S["ax_s"], i_motor=S.get("i_motor", np.full(n, np.nan)))
    has_pose = pose_ok and "px" in S
    rows: List[int] = []
    n_imp = 0
    for i0 in range(warm + 1, n - hor - 6, stride):
        sl = slice(i0 - warm, i0 + hor + 1)
        if v[i0] < v_min or v[i0:i0 + hor + 1].min() < TH.WIN_MIN_V_ALL:
            continue
        if S["cmd_age"][sl].max() > TH.WIN_MAX_CMD_GAP_S:
            continue
        if imp[sl].any():
            n_imp += 1
            continue
        rows.append(i0)
    W: Dict[str, np.ndarray] = {}
    idx = np.array(rows, dtype=int)
    span = np.arange(-warm, hor + 1)
    if len(idx) == 0:
        for k in KEYS + ("px", "py", "pyaw"):
            W[k] = np.zeros((0, warm + hor + 1))
        W["has_pf"] = np.zeros(0, bool); W["bag"] = np.zeros(0, dtype="<U40"); W["t0"] = np.zeros(0)
        W["n_impact_excluded"] = n_imp
        return W
    G = idx[:, None] + span[None, :]
    for k in KEYS:
        W[k] = src[k][G]
    for k in ("px", "py", "pyaw"):
        W[k] = S[k][G] if has_pose else np.zeros_like(W["r"])
    W["has_pf"] = np.full(len(idx), has_pose)
    W["bag"] = np.array([bag_name] * len(idx), dtype="<U40")
    W["t0"] = S["t"][idx]
    W["n_impact_excluded"] = n_imp
    return W


def concat(Ws: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    Ws = [W for W in Ws if len(W["r"])]
    keys = [k for k in Ws[0] if isinstance(Ws[0][k], np.ndarray)]
    return {k: np.concatenate([W[k] for W in Ws]) for k in keys}


def subset(W: Dict[str, np.ndarray], idx) -> Dict[str, np.ndarray]:
    n = len(W["r"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in W.items()}


def real_sideslip_change(W: Dict[str, np.ndarray], warm: int = WARM) -> np.ndarray:
    """Body sideslip change the real car went through, from beta_dot = a_y / v - r (IMU lateral accel, gyro, wheel
    speed), integrated from t0 over the horizon: (N, HOR+1). A car that holds a corner keeps it near 0; a spin runs
    it to tens of degrees. Gyro / accelerometer bias drift over 2 s is a few degrees at most."""
    v = np.maximum(W["v"][:, warm:], 0.3)
    bd = W["ay"][:, warm:] / v - W["r"][:, warm:]
    out = np.zeros_like(bd)
    out[:, 1:] = np.cumsum(0.5 * (bd[:, 1:] + bd[:, :-1]), 1) / HZ
    return out
