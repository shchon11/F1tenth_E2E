"""Identifiability verdicts: profile likelihood with a block-paired significance test.

Windows overlap (2.5 s windows every 0.25 s), so they are not independent samples. A profile point is
'indistinguishable from the best' when the per-block mean increase of the yaw-rate error is within two standard
errors over blocks (20 s chunks of a bag) -- the blocks, not the windows, are the units. The set of indistinguishable
grid values is the interval; if it runs into the end of the physical range the parameter is only bounded there.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from . import fit as F
from .windows import WARM


def per_window_mse(model: Callable, W, p: dict, h: int = 100, steer_key: str = "steer", idx=None) -> np.ndarray:
    sel = np.arange(len(W["r"])) if idx is None else np.asarray(idx)
    pr = model(W, p, idx=sel, steer_key=steer_key, hor=h)
    e = pr["r"][:, 1:h + 1] - W["r"][sel, WARM + 1:WARM + h + 1]
    e = np.where(np.isfinite(e), e, 10.0)
    return np.mean(e ** 2, 1)


def paired_test(e_a: np.ndarray, e_b: np.ndarray, blocks: np.ndarray):
    """(mean increase of e_a over e_b, its standard error) over blocks."""
    ub = np.unique(blocks)
    d = np.array([np.mean(e_a[blocks == u] - e_b[blocks == u]) for u in ub])
    w = np.array([np.sum(blocks == u) for u in ub], dtype=float)
    m = float(np.sum(w * d) / np.sum(w))
    se = float(np.sqrt(np.sum(w ** 2 * (d - m) ** 2)) / np.sum(w)) if len(ub) > 1 else float("inf")
    return m, se


def profile_verdict(model: Callable, W, p_best: dict, names: Sequence[str], which: str, grid: Sequence[float],
                    h: int = 100, max_nfev: int = 40, steer_key: str = "steer", z: float = 2.0) -> Dict:
    """Profile over `grid` (refitting the other `names`), then the interval of grid values not significantly worse
    than the best, and a verdict: measured (interval inside the range) / lower_bound / upper_bound / not_identified."""
    blocks = F.blocks(W)
    rows = F.profile(model, W, p_best, names, which, grid, h=h, max_nfev=max_nfev, steer_key=steer_key)
    e_best = per_window_mse(model, W, p_best, h, steer_key)
    J_best = float(np.mean(e_best))
    pts = []
    for g, J, p in rows:
        e = per_window_mse(model, W, p, h, steer_key)
        m, se = paired_test(e, e_best, blocks)
        pts.append(dict(x=g, J=float(np.mean(e)), dJ=m, se=se, worse=bool(m > z * se and m > 0)))
    xs = np.array([q["x"] for q in pts])
    ok = np.array([not q["worse"] for q in pts])
    if not ok.any():
        lo = hi = float(p_best[which])
    else:
        lo, hi = float(xs[ok].min()), float(xs[ok].max())
    at_lo, at_hi = bool(ok[0]), bool(ok[-1])
    if at_lo and at_hi:
        verdict = "not_identified"
    elif at_hi:
        verdict = "lower_bound"
    elif at_lo:
        verdict = "upper_bound"
    else:
        verdict = "measured"
    return dict(param=which, best=float(p_best[which]), J_best=J_best, interval=(lo, hi), verdict=verdict,
                points=pts, n_blocks=int(len(np.unique(blocks))))
