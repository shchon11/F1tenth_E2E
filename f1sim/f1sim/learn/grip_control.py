"""Friction-informed plan/MPC controller. Opt-in; the legacy path is untouched.

Three arms share this code and differ only in the friction handed to it:

    legacy      nothing installed -- `mpc.solve` as it was
    fixed       mu = 0.73423 everywhere (1.0489 * 0.70, the bottom of the randomization)
    oracle      mu = the episode's true `Simulator.P["mu"]`, per env

The actor is the original frozen checkpoint in all three. Nothing here sees the map, the raceline,
or any future surface: the inputs are the 8D plan the actor emitted and the car's measured speed.

Formulas and their justification are in
`work/learning-next/grip-control/core-status.md`; the short version:

    a_lat_max = mu * mu_f_scale * g                       front saturates first, load share cancels
    rho       = clamp(v^2 |kappa| / a_lat_max, 0, RHO_MAX)  planned lateral utilization, per sample
    q         = mu * mu_r_scale * sqrt(1 - rho^2)           residual coefficient for drive/brake
    a_acc     = min(a_max,   q g lf / (L - q h))            accel loads the rear
    a_brk     = min(a_brake, q g lf / (L + q h))            braking unloads it

then a curvature cap, a backward braking pass and a forward acceleration pass over the plan's own
dense samples. No second MPC: `mpc.path_points`, `mpc.reference` and `mpc.ilqr` do the work.

**This is a planning approximation, not a safety guarantee**, and not a claim of achieved
feasibility. The profile is an *upper envelope over the path*: it is not seeded from the measured
speed and says nothing about whether the current state can reach it -- see §2b and §3 of the status
note.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

from .. import mpc as _mpc

G = 9.81

#: The fixed arm's friction: `vehicle.mu` nominal 1.0489 times the bottom of its randomization, 0.70.
MU_FIXED_LOW = 0.73423

#: Planned lateral utilization. Reaching the tyre's peak needs the optimal slip angle, which a
#: tracking controller never targets. One number, used for both the curvature cap and the residual,
#: so the allocation is consistent. Frozen before evaluation.
RHO_MAX = 0.85

#: Lead term only, and only for a step whose `u_prev` is zero: `v_cmd_lead / motor_tau = 0.15/0.20`.
#: It is **not** the fraction of a planned acceleration the whole loop delivers, and must not be read
#: as one.
LEAD_FRACTION_FIRST_STEP = 0.75

#: The same ratio once `u_prev ~= u`, which is the steady case: `z0`'s speed already carries
#: `u_prev * delay`, so the lead is `(v_cmd_lead + delay) / motor_tau = (0.15 + 0.035)/0.20`.
#: Backend's `tracker_chain_probe` measured it directly -- a steady `u = -5` produces a motor request
#: of -4.625 -- and the achieved longitudinal acceleration at low/mid/high grip was
#: -3.300 / -3.977 / -4.429 m/s^2, the mild case slightly *exceeding* |u| because drag also brakes.
#: The remaining shortfall at low mu is tyre saturation, which is the thing the experiment measures.
LEAD_FRACTION_STEADY = 0.925

#: **Neither number is applied anywhere.** They are reported so the command-path mismatch is a
#: measured quantity in the evaluation instead of an assumption inside the controller. Scaling the
#: budgets by one of them would not make the prediction exact -- the response still lags -- it would
#: only make the controller slower while looking principled.
TRANSFER_APPLIED = False

#: `estimated` is the same controller as `oracle`, fed the frozen estimator's filtered lower
#: quantile instead of the truth. It differs from `oracle` only in where `mu` comes from, which is
#: the entire point of the comparison -- so it shares the code path exactly.
MODES = ("legacy", "fixed", "oracle", "estimated")

#: Modes whose friction changes during a run, so `update()` is meaningful.
DYNAMIC_MODES = ("oracle", "estimated")


@dataclass(frozen=True)
class GripSpec:
    """Everything the controller needs besides `mu`, all nominal. Recorded with any result."""
    mode: str = "legacy"
    mu_fixed: float = MU_FIXED_LOW
    rho_max: float = RHO_MAX
    lf: float = 0.15875
    lr: float = 0.17145
    h: float = 0.074
    mu_f_scale: float = 0.92
    mu_r_scale: float = 1.0
    a_max: float = 7.0
    a_brake: float = 5.0
    n_path: int = 25

    @property
    def L(self) -> float:
        return self.lf + self.lr

    def to_meta(self) -> dict:
        d = asdict(d_self := self)
        d["L"] = d_self.L
        d["lead_fraction_first_step"] = LEAD_FRACTION_FIRST_STEP
        d["lead_fraction_steady"] = LEAD_FRACTION_STEADY
        d["transfer_applied"] = TRANSFER_APPLIED
        return d

    def validate(self) -> "GripSpec":
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 < self.rho_max <= 1.0:
            raise ValueError(f"rho_max must be in (0, 1], got {self.rho_max}")
        # `a_acc = q g lf / (L - q h)` diverges at q = L/h; q never gets near it here, but a spec
        # that made it possible would produce a silently enormous budget.
        if self.h > 0 and self.L / self.h <= 2.0:
            raise ValueError(f"L/h = {self.L / self.h:.2f} is too small for the accel formula")
        return self


def budgets(mu: torch.Tensor, rho: torch.Tensor, spec: GripSpec):
    """(a_lat_max, a_acc, a_brk) for a given friction and planned lateral utilization.

    `mu` broadcasts against `rho`, so the same code serves a per-env scalar and a per-sample field.
    """
    a_lat_max = mu * spec.mu_f_scale * G
    q = mu * spec.mu_r_scale * torch.sqrt((1.0 - rho.clamp(0.0, 1.0) ** 2).clamp_min(0.0))
    a_acc = (q * G * spec.lf / (spec.L - q * spec.h).clamp_min(1e-3)).clamp(max=spec.a_max)
    a_brk = (q * G * spec.lf / (spec.L + q * spec.h)).clamp(max=spec.a_brake)
    return a_lat_max, a_acc, a_brk


def speed_envelope(k: torch.Tensor, Lp: torch.Tensor, speed_cap: torch.Tensor,
                   v0: torch.Tensor, v1: torch.Tensor, mu: torch.Tensor, spec: GripSpec):
    """The friction-limited speed at each dense sample of the plan: (B, n), plus diagnostics.

    Curvature cap, then a backward pass (braking anticipation: the speed allowed here is what can
    still be shed before the corner ahead) and a forward pass (no more speed than can be gained
    along the path). Both loops run over a Python constant, so they unroll and stay graph-safe.

    **An upper envelope over the path, not a trajectory from the current state.** The measured speed
    is deliberately not an input: it is not used to seed sample 0, and nothing here claims the car
    can reach this profile from where it is. A car already faster than the envelope simply gets a
    target below its current speed, which `mpc.reference`'s rate-limited walk turns into a braking
    demand at the budget.
    """
    n = spec.n_path
    _, _, _, s, kap = _mpc.path_points(k, Lp, n=n, return_kappa=True)
    ds = (Lp / (n - 1)).clamp_min(1e-3)[:, None]                            # (B,1)
    mu_c = mu.reshape(-1, 1)                                                # (B,1)

    frac = torch.linspace(0.0, 1.0, n, device=k.device, dtype=k.dtype)[None]
    v_plan = v0[:, None] + (v1 - v0)[:, None] * frac                        # the plan's own profile
    a_lat_max = mu_c * spec.mu_f_scale * G
    v_curve = torch.sqrt((spec.rho_max * a_lat_max) / kap.abs().clamp_min(1e-3))
    cap = torch.minimum(torch.minimum(v_plan, v_curve), speed_cap[:, None]).clamp_min(0.0)

    # Residual longitudinal budget at the lateral use this cap actually implies. One shot, from the
    # capped speed, before the passes -- explicit rather than iterated.
    rho = (cap ** 2 * kap.abs() / a_lat_max.clamp_min(1e-6)).clamp(0.0, spec.rho_max)
    _, a_acc, a_brk = budgets(mu_c, rho, spec)                              # (B,n) each

    # One bound per env, the tightest along this plan, and the passes use exactly what the reference
    # walk and the iLQR clamp will use. Per-sample budgets in the passes would let the backward pass
    # plan a deceleration at a point where the solver is not allowed to command it -- inconsistent
    # even as an approximation. The minimum is the conservative reconciliation.
    a_acc_e = a_acc.min(1).values                                           # (B,)
    a_brk_e = a_brk.min(1).values

    # Columns, not slice-and-concat: the passes are sequential over a Python constant, so they
    # unroll cleanly and stay CUDA-graph-safe.
    cols = list(cap.unbind(1))
    for i in range(n - 2, -1, -1):          # backward: brake in time for what is ahead
        cols[i] = torch.minimum(cols[i], torch.sqrt(cols[i + 1] ** 2 + 2.0 * a_brk_e * ds[:, 0]))
    for i in range(1, n):                   # forward: no more speed than can be gained by here
        cols[i] = torch.minimum(cols[i], torch.sqrt(cols[i - 1] ** 2 + 2.0 * a_acc_e * ds[:, 0]))
    v = torch.stack(cols, 1)

    diag = {"kappa": kap, "v_plan": v_plan, "v_curve": v_curve, "v_limit": v,
            "a_acc": a_acc, "a_brk": a_brk, "a_acc_bound": a_acc_e, "a_brk_bound": a_brk_e,
            "rho": rho, "a_lat_max": a_lat_max, "arc": s}
    return v, diag


def solve_grip(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm, mu,
               spec: _mpc.PlanSpec, wb: float, s_max: float, v_max: float,
               gspec: GripSpec, consts=None, out_bounds=None):
    """`mpc.solve` with a friction-limited reference and per-env control bounds.

    `mu` is a **tensor argument**, so under `GraphedCallable` it becomes a static input buffer that
    is copied per call -- a live graph input, not a scalar frozen into the capture. `PlanSpec` is
    never mutated.

    `consts` is `mpc.build_ilqr_consts`' tuple. Pass it whenever this will be captured: with
    `consts=None` the iLQR builds its cost tensors inline from Python tuples, and the first such
    `torch.tensor(spec.q)` is a pageable host-to-device copy that fails capture -- and a failed
    capture poisons the CUDA context rather than falling back.
    """
    k, Lp, v0, v1 = _mpc.decode(action, v_meas, v_max, speed_cap, spec)
    v_limit, diag = speed_envelope(k, Lp, speed_cap, v0, v1, mu, gspec)

    # The same per-env budgets the envelope's passes used -- the tightest point of this plan, at the
    # lateral utilization actually planned rather than at rho = 0. The solver may not command more
    # than the worst point of the path allows.
    a_acc_e, a_brk_e = diag["a_acc_bound"], diag["a_brk_bound"]
    lo = torch.stack([torch.full_like(a_acc_e, -s_max), -a_brk_e], 1)
    hi = torch.stack([torch.full_like(a_acc_e, s_max), a_acc_e], 1)
    bounds = torch.stack([lo, hi], 1)                                       # (B, 2, 2)
    a_walk = torch.stack([a_brk_e, a_acc_e], 1)                             # (B, 2)
    if out_bounds is not None:
        # The limits this solve actually imposed, written into a caller-owned buffer. A closed-over
        # tensor is written on every graph replay too, so the training loop can log the realised
        # bound without recomputing the envelope or synchronising -- and without confusing it with
        # the straight-line budget, which is a different (larger) number on any curved plan.
        out_bounds.copy_(a_walk)

    return _mpc.solve(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm, spec, wb, s_max,
                      v_max, consts=consts, v_limit=v_limit, bounds=bounds, a_walk=a_walk)


class GripMPC:
    """Installs `solve_grip` as the tracker's solver, keeps `mu` live, and puts things back.

    Ordering matters: `prepare_graph_runtime` captures `mpc.solve` and installs it as
    `tracker._solver`, so it would overwrite a hook installed earlier. Install this one **after** it.
    """

    def __init__(self, tracker, gspec: GripSpec, batch: int, device, wb: float, s_max: float,
                 v_max: float):
        self.gspec = gspec.validate()
        self.tracker = tracker
        self.B = int(batch)
        self.device = torch.device(device)
        self.wb, self.s_max, self.v_max = float(wb), float(s_max), float(v_max)
        #: The live friction. `update()` writes this buffer in place; the graph copies from it, so a
        #: new value reaches a captured graph without recapture.
        self.mu = torch.full((self.B,), float(gspec.mu_fixed), device=self.device)
        #: Built once, here, and bound into the solver -- exactly what `graph_fastpath.capture_mpc`
        #: does and for the same reason: `ilqr`'s inline `torch.tensor(spec.q)` is a pageable H2D copy
        #: that fails CUDA-graph capture, and a failed capture is fatal to the process rather than
        #: recoverable. `dtype` follows the default dtype so the bound path is numerically the inline
        #: path. Nothing is cached globally; these live and die with this object.
        self._consts = _mpc.build_ilqr_consts(tracker.spec, self.s_max, self.device,
                                              dtype=torch.get_default_dtype())
        #: The limits the last solve actually imposed, `[a_brk, a_acc]` per env. Written by the
        #: solver (inside the graph too, since it is closed over rather than passed), so a caller can
        #: log the realised bound instead of the straight-line budget, which on a curved plan is a
        #: different and larger number.
        self.last_bounds = torch.zeros(self.B, 2, device=self.device)
        self._prev_solver = None
        self._installed = False
        self._graphed = None

    # -- friction ----------------------------------------------------------------
    def update(self, mu: torch.Tensor) -> None:
        """Set the per-env friction. Dynamic arms only; the fixed arm never calls this."""
        if self.gspec.mode not in DYNAMIC_MODES:
            raise RuntimeError(f"update() is for the oracle arm and the estimated arm; "
                               f"this one is {self.gspec.mode!r}")
        m = torch.as_tensor(mu, device=self.device, dtype=self.mu.dtype).reshape(-1)
        if m.numel() != self.B:
            raise ValueError(f"expected {self.B} friction values, got {m.numel()}")
        self.mu.copy_(m)

    # -- install / release -------------------------------------------------------
    def _bound(self, *args):
        """The 12-argument solver the graph captures: the tracker's 11, plus `mu`.

        `gspec` and `consts` are closed over rather than passed, so the captured argument list is
        tensors and atoms only -- the same shape `capture_mpc` uses. Eager runs through this too, so
        an eager/graph comparison is a comparison of the same function.
        """
        return solve_grip(*args, self.gspec, consts=self._consts, out_bounds=self.last_bounds)

    def install(self, graph: bool = False, example_args=None) -> "GripMPC":
        if self.gspec.mode == "legacy":
            raise RuntimeError("the legacy arm installs nothing; do not call install()")
        if self._installed:
            raise RuntimeError("already installed")
        if graph:
            if example_args is None:
                raise ValueError("graph=True needs example_args recorded from a real step")
            if self.device.type != "cuda":
                raise RuntimeError(f"graph capture needs CUDA; this one is on {self.device}")
            from ..viewer.graph_fastpath import GraphedCallable
            a = list(example_args)
            full = a[:7] + [self.mu] + a[7:]                   # (…, warm, mu, spec, wb, s_max, v_max)
            self._graphed = GraphedCallable(self._bound, full, name="solve_grip")
            self._graphed.adopt()
            fn = lambda *args: self._graphed(*args[:7], self.mu, *args[7:])
        else:
            fn = lambda *args: self._bound(*args[:7], self.mu, *args[7:])
        self._prev_solver = getattr(self.tracker, "_solver", None)
        self.tracker._solver = fn
        self._installed = True
        return self

    def release(self) -> None:
        """Restore whatever solver was installed before. Safe to call twice."""
        if self._installed:
            self.tracker._solver = self._prev_solver
            self._installed = False
        self._prev_solver = None
        self._graphed = None

    # -- diagnostics -------------------------------------------------------------
    def diagnostics(self, action, v_meas, speed_cap) -> dict:
        """Recompute the envelope for inspection. Read-only; does not touch the installed solver."""
        sp = self.tracker.spec
        with torch.no_grad():
            k, Lp, v0, v1 = _mpc.decode(action, v_meas, self.v_max, speed_cap, sp)
            v_limit, d = speed_envelope(k, Lp, speed_cap, v0, v1, self.mu, self.gspec)
        return {
            "mode": self.gspec.mode, "mu": self.mu.detach().clone(),
            "v0_raw": v0.detach().clone(), "v1_raw": v1.detach().clone(),   # raw decoded plan speeds
            "v_plan": d["v_plan"].detach().clone(),
            "v_curve": d["v_curve"].detach().clone(),
            "v_limit": v_limit.detach().clone(),                            # limited reference speeds
            "a_acc_bound": d["a_acc_bound"].detach().clone(),               # what the iLQR is clamped to
            "a_brk_bound": d["a_brk_bound"].detach().clone(),               # and what the passes used
            "a_acc_path": d["a_acc"].detach().clone(),                      # per-sample, before the min
            "a_brk_path": d["a_brk"].detach().clone(),
            "a_lat_max": d["a_lat_max"].detach().clone(),
            "rho": d["rho"].detach().clone(),
            "kappa": d["kappa"].detach().clone(),
            # Reported, never applied, and neither is a whole-loop delivery fraction: the first is the
            # lead term alone at u_prev = 0, the second the measured steady ratio. Compare the
            # achieved acceleration against them; do not read either as a correction factor.
            "lead_fraction_first_step": LEAD_FRACTION_FIRST_STEP,
            "lead_fraction_steady": LEAD_FRACTION_STEADY,
            "transfer_applied": TRANSFER_APPLIED,
        }
