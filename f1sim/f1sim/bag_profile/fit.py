"""Least-squares fitting, profile likelihood and block bootstrap for the replay models.

The objective everywhere is the audit's: mean squared yaw-rate error over the first second of every window (or the
speed error, for the longitudinal model). Profile likelihood: fix one parameter on a grid, refit the rest, report
the objective; a parameter whose profile stays within the noise band over its whole physical range is 'not
identified', one whose profile is flat on one side only is a bound ('lower bound only').
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .windows import WARM

#: physical bounds for every free parameter any model fits
BOUNDS = {
    "k": (0.3, 2.5), "b": (-0.12, 0.12), "tau": (0.0, 0.25), "kus": (0.0, 0.2),
    "tau_d": (0.0, 0.25), "tau_s": (0.005, 0.25),
    "sf": (0.5, 60.0), "mf": (0.3, 3.0), "Cf": (1.0, 2.2),
    "sr": (0.5, 60.0), "mr": (0.3, 3.0), "Cr": (1.0, 2.2),
    "tau_m": (0.03, 1.5), "tau_v": (0.0, 0.3), "a_a": (0.5, 15.0), "a_b": (0.5, 15.0), "v_s": (2.0, 30.0),
    "iz_m": (0.004, 0.03),
}


def residual_fn(model: Callable, W, base: dict, names: Sequence[str], target: str = "r", h: int = 100,
                idx=None, steer_key="steer"):
    sel = np.arange(len(W["r"])) if idx is None else np.asarray(idx)
    meas = W[target][sel, WARM + 1:WARM + h + 1]

    def f(x):
        p = dict(base)
        p.update(zip(names, x))
        pr = model(W, p, idx=sel, steer_key=steer_key, hor=h) if target == "r" else model(W, p, idx=sel, hor=h)
        res = pr["r" if target == "r" else "v"][:, 1:h + 1] - meas
        res = np.where(np.isfinite(res), res, 10.0)        # a model that blew up is just wrong, not NaN
        return res.ravel()
    return f


def fit(model: Callable, W, base: dict, names: Sequence[str], target: str = "r", h: int = 100, idx=None,
        steer_key="steer", max_nfev: int = 200, verbose: int = 0, x0: Optional[Sequence[float]] = None):
    f = residual_fn(model, W, base, names, target, h, idx, steer_key)
    lo = np.array([BOUNDS[n][0] for n in names]); hi = np.array([BOUNDS[n][1] for n in names])
    x0 = np.clip(np.array([base[n] for n in names] if x0 is None else x0, dtype=float), lo + 1e-9, hi - 1e-9)
    r = least_squares(f, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3, max_nfev=max_nfev, verbose=verbose)
    p = dict(base)
    p.update(zip(names, r.x.tolist()))
    J = float(np.mean(r.fun ** 2))
    return p, J, r


def objective(model, W, p, target="r", h=100, idx=None, steer_key="steer"):
    f = residual_fn(model, W, p, [], target, h, idx, steer_key)
    return float(np.mean(f(np.zeros(0)) ** 2))


def profile(model, W, p_best: dict, names: Sequence[str], which: str, grid: Sequence[float], h=100, idx=None,
            steer_key="steer", max_nfev=60) -> List[Tuple[float, float, dict]]:
    """Profile of the objective in parameter `which`: for each grid value refit the other `names`. Walks outward
    from the best value in both directions, each point warm-started from its inner neighbour, so a far grid point
    cannot drag the path into another basin before the points near the optimum are visited."""
    rest = [n for n in names if n != which]
    grid = sorted(float(g) for g in grid)
    x0 = float(p_best[which])
    up = [g for g in grid if g >= x0]
    down = [g for g in grid if g < x0][::-1]
    res = {}
    for path in (up, down):
        p_prev = dict(p_best)
        for g in path:
            base = dict(p_prev)
            base[which] = g
            if rest:
                p, J, _ = fit(model, W, base, rest, h=h, idx=idx, steer_key=steer_key, max_nfev=max_nfev)
            else:
                p, J = base, objective(model, W, base, h=h, idx=idx, steer_key=steer_key)
            res[g] = (g, J, p)
            p_prev = p
    return [res[g] for g in grid]


def blocks(W, seconds: float = 20.0) -> np.ndarray:
    """block id per window: (bag, t0 // seconds) -- overlapping windows share a block, so a bootstrap over blocks
    does not pretend that neighbouring windows are independent."""
    keys = [f"{b}:{int(t // seconds)}" for b, t in zip(W["bag"], W["t0"])]
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def bootstrap(model, W, p_best: dict, names: Sequence[str], n_rep: int = 20, seconds: float = 20.0, h=100,
              seed: int = 0, max_nfev=40, steer_key="steer") -> np.ndarray:
    """Block bootstrap of the fit: (n_rep, len(names)). Warm-started at the best fit."""
    rng = np.random.default_rng(seed)
    bl = blocks(W, seconds)
    ub = np.unique(bl)
    by = {b: np.flatnonzero(bl == b) for b in ub}
    reps = []
    for _ in range(n_rep):
        pick = rng.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([by[b] for b in pick])
        p, J, _ = fit(model, W, p_best, names, h=h, idx=idx, max_nfev=max_nfev, steer_key=steer_key)
        reps.append([p[n] for n in names])
    return np.array(reps)


def summarize(W, pr, idx=None, horizons=(50, 100, 200)):
    """median / p90 per error kind and horizon -> nested dict (the audit's table cells)."""
    from .models import errors
    e = errors(W, pr, idx, horizons)
    return {h: {k: (float(np.nanmedian(v)), float(np.nanpercentile(v, 90))) for k, v in d.items()}
            for h, d in e.items()}


def fit_grid(model: Callable, W, base: dict, names: Sequence[str], grid_name: str, grid: Sequence[float],
             target: str = "r", h: int = 100, idx=None, steer_key="steer", max_nfev: int = 60):
    """A delay (or any parameter the model only sees in whole samples) on a grid, the rest by least squares at each
    grid value, warm-started along the grid. Returns (p, J, [(g, J)])."""
    rest = [n for n in names if n != grid_name]
    best, curve, prev = None, [], dict(base)
    for g in grid:
        b = dict(prev)
        b[grid_name] = float(g)
        if rest:
            p, J, _ = fit(model, W, b, rest, target=target, h=h, idx=idx, steer_key=steer_key, max_nfev=max_nfev)
        else:
            p, J = b, objective(model, W, b, target=target, h=h, idx=idx, steer_key=steer_key)
        curve.append((float(g), J))
        prev = p
        if best is None or J < best[1]:
            best = (p, J)
    return best[0], best[1], curve
