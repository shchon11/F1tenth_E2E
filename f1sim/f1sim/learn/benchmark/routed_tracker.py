"""RoutedTracker: the candidate's arm drives the candidate only.

`gym_env` runs every car's plan through one tracker (`step:537`), and the teacher builds its plan
from `tracker.spec` (`:483` and `:772`). So without this, changing the candidate's controller changes
the opponent's issued commands too, and "overtaking against a fixed opponent" is not what gets
measured.

Both trackers are full-B and both are called on the whole batch; outputs are selected by mask.
Row-subsetting is not an option: `ControllerRuntime.install` reads `env.tracker` with no candidate
argument (grip_runtime.py:289) and sizes `GripMPC` at `env.B`.

Install order is fixed -- see `install_order()`.
"""
from __future__ import annotations
import torch


class RoutedTracker:
    """Wraps the candidate tracker; routes opponent rows to a pinned reference tracker."""

    def __init__(self, candidate, reference, env):
        self.candidate, self.reference, self.env = candidate, reference, env
        self._original = candidate

    # -- the mask is read live: `on_policy` is rewritten on reset in mixed mode ------------------
    @property
    def _mask(self) -> torch.Tensor:
        return self.env.on_policy

    def __call__(self, action, v_meas, speed_cap, yaw_rate, delay=None):
        raw_c = self.candidate(action, v_meas, speed_cap, yaw_rate, delay)
        raw_r = self.reference(action, v_meas, speed_cap, yaw_rate, delay)
        return torch.where(self._mask[:, None], raw_c, raw_r)

    # -- teacher plan construction must not see the candidate's spec (gym_env 483, 772) ----------
    @property
    def spec(self):
        return self.reference.spec

    # -- geometry the callers read; identical on both, taken from the reference -----------------
    @property
    def wb(self):
        return self.reference.wb

    @property
    def s_max(self):
        return self.reference.s_max

    @property
    def v_max(self):
        return self.reference.v_max

    # -- solver lifecycle: proxies the CANDIDATE, so a later assignment cannot retarget the ------
    #    reference and quietly give the opponent the candidate's arm.
    @property
    def _solver(self):
        return self.candidate._solver

    @_solver.setter
    def _solver(self, fn):
        self.candidate._solver = fn

    @property
    def u_prev(self):
        return self.candidate.u_prev

    @property
    def compile_solver(self):
        return self.candidate.compile_solver

    # -- per-row state lives in both; a reset must reach both -----------------------------------
    def reset(self, ids) -> None:
        self.candidate.reset(ids)
        self.reference.reset(ids)

    # -- plans read back by reward_plan_clearance (gym_env:546) and viewers ----------------------
    @property
    def last_ref(self):
        return self._mask_plan("last_ref")

    @property
    def last_pred(self):
        return self._mask_plan("last_pred")

    def _mask_plan(self, name):
        c, r = getattr(self.candidate, name), getattr(self.reference, name)
        if c is None or r is None:
            return c if r is None else r
        return torch.where(self._mask[:, None, None], c, r)

    def uninstall(self) -> None:
        """Restore `env.tracker` to the original object. Identity, not equality."""
        self.env.tracker = self._original


def install_order() -> tuple[str, ...]:
    """The order is load-bearing; two steps cannot move.

    2 -> 3: `prepare_graph_runtime` captures `mpc.solve` and assigns `tracker._solver`, so an arm
    hook installed before it is silently overwritten and the run is a legacy run wearing another
    arm's name (grip_runtime.py:272-276).

    3 -> 5: `ControllerRuntime.install()` has no way to target anything but `env.tracker`, so the arm
    must go onto the original tracker while it is still `env.tracker`, and only then may it be
    wrapped.
    """
    return ("build_env(race_size=M)",
            "prepare_graph_runtime(env)",
            "ControllerRuntime(arm).install(graph_rt=holder)",
            "reference = PlanTracker(env.B, ...)",
            "env.tracker = RoutedTracker(candidate=original, reference=reference, env=env)",
            "env.set_teacher(make_teacher(rls, env, ...))",
            "seeded reset",
            "install MuInvariantTrace",
            "run")
