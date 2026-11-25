"""(11-25) `plan_rl --commit-plan W`: the 1 s plan as a COMMITMENT, not a forecast.

Why (the learning-signal audit of 11-22..25): with `--tail-consistency` only the plan's first 0.1 s (2 of its 20 outputs) was
the RL action; the other 18 were taught to copy the next plan / the driven path. Every cost put on the tail priced a forecast
the car never had to keep -- qual28d's wall price (100 % of the priced plans never near a wall in reality), dg35a's direct
car gradient (race33b's 12.7 % plan overlap with a slow car's future was 12.3 points forecast; the driven path 0.5 %) -- and
exploration was 0.1 s deep (+-0.1 m a decision), so a multi-second manoeuvre had no gradient path.

The mode:
* The action is the WHOLE plan. The exploration noise is drawn on an orthonormal basis of each channel's 10-step profile
  (turn, speed): discrete orthonormal polynomials over the second, degree 0 .. 9 (`coef_basis`). The first `smooth`
  degrees carry the exploration (an arc / a lane change / an S, a speed-up / brake-then-go), the higher ones a small
  remainder so the distribution keeps full rank (every output has a log-probability, PPO can move every one of them).
  u = mean + Q^T (sigma * z), z ~ N(0, 1) per coefficient; the policy's `log_std` (20 numbers, the same tensor and shape as
  before) is the log of sigma per coefficient: [turn deg 0..9, speed deg 0..9]. Q is orthonormal, so the log-probability is
  the sum over the 20 coefficients of N(c_j; 0, sigma_j), c = Q (u - mean), plus tanh's Jacobian (`commit_logp`).
* Execution is unchanged: the first 0.1 s exactly, then a new plan.
* The commitment cost (in the reward, `promise_cost`): at every decision the new plan's 0.1 .. 0.9 s (points 1..9) against
  the previous plan's 0.2 .. 1.0 s (its points 2..10: the same moments), positions and speeds. Each point's gap as a re-plan
  RATE -- metres per second of the new plan's own horizon, e_k / (0.1 k) -- because a plan's honest drift grows with the
  horizon (race33b u975, exam4: the median gap is 0.12 m per second of horizon at every point, p90 0.3), then a Huber of it over
  `ref` (quadratic below, linear above), averaged over the 9 points, x W laps. Not charged after a restart (no promise) or
  when the car is not where the promise put it (a contact, or > `off_m` off its 0.1 s point): the promise was broken for it.
* The previous plan is in the policy's and the critic's state already (`PlanObs2.rel`, `rel_v`: its points 2..10 and its 10
  speeds in the frame of the pose it said the car would be in now -- exactly the promise), so the value can price it."""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .. import plan_steps as PS

N = PS.N_STEP


def coef_basis(n: int = N, dtype=torch.float64) -> torch.Tensor:
    """(n, n): row j is the discrete orthonormal polynomial of degree j over the n step midpoints of the second (the
    Gram-Schmidt of 1, t, t^2, .. with t in [-1, 1]); each row's last value made positive. Q Q^T = I"""
    t = (torch.arange(n, dtype=torch.float64) + 0.5) / n * 2.0 - 1.0
    V = torch.stack([t ** j for j in range(n)], 1)                  # (n, n) columns 1, t, t^2, ..
    q, r = torch.linalg.qr(V)                                       # columns orthonormal, in degree order
    q = q * torch.sign(torch.diagonal(r))[None, :]                  # the leading coefficient of each positive
    Q = q.T.contiguous()
    return Q.to(dtype)


def to_steps(c: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """(B, 2N) coefficients [turn deg 0..N-1, speed deg 0..N-1] -> (B, 2N) per-step values [turn steps, speed steps]"""
    B = c.shape[0]
    return (c.view(B, 2, N) @ Q.to(c.dtype)).reshape(B, 2 * N)


def to_coef(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """the inverse of `to_steps` (Q orthonormal: Q^-1 = Q^T)"""
    B = x.shape[0]
    return (x.reshape(B, 2, N) @ Q.T.to(x.dtype)).reshape(B, 2 * N)


def sample(mean: torch.Tensor, log_std: torch.Tensor, Q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """the plan's pre-tanh action: mean + Q^T (sigma * z), `z` (B, 2N) standard normal per coefficient"""
    return mean + to_steps(log_std.exp().expand_as(z) * z, Q)


def commit_logp(u: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """log-probability of a = tanh(u), u = mean + Q^T (sigma * z): the sum over the 2N coefficients c = Q (u - mean) of
    N(c_j; 0, sigma_j) (Q orthonormal: no Jacobian of its own), plus tanh's Jacobian per output (as `tanh_logp`)"""
    c = to_coef(u - mean, Q)
    ls = log_std.expand_as(c)
    lp = -0.5 * (c * torch.exp(-ls)) ** 2 - ls - 0.5 * math.log(2.0 * math.pi)
    jac = 2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))
    return lp.sum(-1) - jac.sum(-1)


def init_log_std(smooth: int, s_turn: float, s_speed: float, s_rest: float, device=None) -> torch.Tensor:
    """(2N,) the starting log sigma per coefficient: the `smooth` lowest degrees of each channel s_turn / s_speed, the rest
    s_rest"""
    s = torch.full((2 * N,), float(s_rest), dtype=torch.float32, device=device)
    s[:smooth] = float(s_turn)
    s[N:N + smooth] = float(s_speed)
    return s.log()


def floor_vec(smooth: int, f_turn: float, f_speed: float, f_rest: float = 0.0, device=None) -> torch.Tensor:
    """(2N,) log-std floors: the smooth turn / speed coefficients and the rest (0 = none: -inf)"""
    lg = lambda x: math.log(x) if x > 0 else -math.inf
    f = torch.full((2 * N,), lg(f_rest), dtype=torch.float32, device=device)
    f[:smooth] = lg(f_turn)
    f[N:N + smooth] = lg(f_speed)
    return f


def step_std(log_std: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """(2N,) the marginal std of each per-step output under the coefficient noise: sqrt(sum_j Q_jk^2 sigma_j^2)"""
    s2 = log_std.detach().float().exp().pow(2).view(2, N)
    return (s2 @ Q.float().pow(2)).sqrt().reshape(-1)


#: the compared points: the new plan's 1..9 (index 0..8 of its 10) against the previous plan's 2..10 (index 1..9)
TAU = torch.arange(1, N, dtype=torch.float32) * PS.DT                # [s] the new plan's own time of each compared point


def promise_gap(new_pts: torch.Tensor, new_v: torch.Tensor, old_pts: torch.Tensor, old_v: torch.Tensor,
                v_s: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """time-aligned gaps between a new plan and the previous one (the promise): (B, N-1) each -- positions [m], speeds
    [m/s], and the combined gap e = sqrt(dp^2 + (v_s dv)^2) [m]. `*_pts` (B, N, 2) world points, `*_v` (B, N) speeds"""
    dp = (new_pts[:, :N - 1] - old_pts[:, 1:]).norm(dim=-1)
    dv = (new_v[:, :N - 1] - old_v[:, 1:]).abs()
    return dp, dv, torch.sqrt(dp * dp + (v_s * dv) ** 2)


def huber1(x: torch.Tensor) -> torch.Tensor:
    """x^2 for |x| <= 1, 2|x| - 1 beyond (continuous with its slope)"""
    a = x.abs()
    return torch.where(a <= 1.0, a * a, 2.0 * a - 1.0)


class PromiseGate:
    """Per learner row: may this decision's plan be charged against the last one? Not at a start (no promise; a restart,
    `reset_rows` lays a standing plan where the car spawned), not after a decision with a contact (the promise was broken for
    the car, not by it), not with the car more than `off_m` off the promise's 0.1 s point (pushed, or held against a wall /
    a box), not with the car slower than `v_min` (a car standing or stuck re-plans freely: backing out of a box, starting off
    -- roll29a-c from scratch stood still when a stopped car could not leave its plan). A scene restored with its last plan
    is a promise again (`scene`)."""

    def __init__(self, n: int, device):
        self.ok = torch.zeros(n, dtype=torch.bool, device=device)     # there is a promise (a plan drawn since the start)
        self.hit = torch.zeros(n, dtype=torch.bool, device=device)    # the decision just driven had a contact

    def charge(self, off: torch.Tensor, off_m: float, v: Optional[torch.Tensor] = None, v_min: float = 0.0) -> torch.Tensor:
        """(n,) bool: the rows whose new plan pays for leaving the promise; `off` (n,) [m] the car from the promise's 0.1 s
        point, `v` (n,) [m/s] its speed now"""
        ok = self.ok & ~self.hit & (off <= off_m)
        return ok & (v.abs() >= v_min) if v is not None else ok

    def after(self, restarted: torch.Tensor, contacted: torch.Tensor) -> None:
        """a decision was driven: its plan is the next promise unless the row restarted in it"""
        self.ok = ~restarted
        self.hit = contacted.clone()

    def scene(self, rows: torch.Tensor) -> None:
        """rows restored from a stored scene, its last plan with them"""
        self.ok[rows] = True
        self.hit[rows] = False


def promise_cost(new_pts: torch.Tensor, new_v: torch.Tensor, old_pts: torch.Tensor, old_v: torch.Tensor,
                 ok: torch.Tensor, w: float, ref: float, v_s: float) -> Dict[str, torch.Tensor]:
    """The commitment cost of a new plan [laps], (B,): W x the mean over the 9 compared points of huber(e_k / (ref t_k)),
    t_k = 0.1 k s the new plan's time of point k; 0 where `ok` is False (no promise / broken by a contact). Also the gaps"""
    dp, dv, e = promise_gap(new_pts, new_v, old_pts, old_v, v_s)
    tau = TAU.to(e.device, e.dtype)
    x = e / (ref * tau)
    c = w * huber1(x).mean(1)
    c = torch.where(ok, c, torch.zeros_like(c))
    return {"cost": c, "dp": dp, "dv": dv, "e": e}
