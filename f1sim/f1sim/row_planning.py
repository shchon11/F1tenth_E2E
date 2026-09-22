"""Planning for some rows of a batch: the opponent teachers' shared half of `plan_rows`.

An opponent teacher in a race drives the rows of its own slot, and in a partitioned mix
(`EnvConfig.kind_mix_assign = "partition"`) a fixed quarter of those. It used to be asked for the
whole batch every step -- learners included -- and its answer selected afterwards, and its work
scales with the rows it is asked for: in s911's recipe at 256 envs the four planners were 61 % of an
eager env step. A teacher that mixes this in can be asked for a static index of rows instead
(`F1VecEnv._alt_rows`), and gives each of them the command the whole-batch call would.

Two kinds of tensor are involved, and keeping them apart is the whole job:

* **the ego's**, which the call plans for: the state, the parameters, the track ids, the offsets --
  cut to the rows by the caller of `_row_scope` -- and the env-wide per-car tensors the teacher reads
  on its own (the speed cap, the tracker's warm start, the gyro, the car geometry), cut with `_r`;
* **the opponents'**, which are rows of the *whole* batch (`sim.other_idx` points into it): read
  through `_opp_state`, never through the cut state.

A teacher's memory across steps (`GRAPH_STATE`) stays whole-batch sized, so the env's per-row
resets keep addressing it by env row; a subset call reads its rows with `_r` and writes them back
in place with `_put`.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch


class RowPlanning:
    """Mixin for a teacher with a `base` reference teacher and an `env`."""

    #: Set only inside `_row_scope`: the env rows this call plans for, and the whole batch's state.
    _rows: Optional[torch.Tensor] = None
    _full_state: Optional[torch.Tensor] = None

    def _r(self, t):
        """`t` for the rows being planned. Env-wide per-car tensors are (B, ...), the call (R, ...)."""
        return t if self._rows is None or not torch.is_tensor(t) or t.dim() == 0 else t[self._rows]

    def _opp_state(self, state):
        """The state an opponent index (`sim.other_idx`) points into: the whole batch's."""
        return state if self._full_state is None else self._full_state

    def _env_rows(self, n: int, device) -> torch.Tensor:
        """The env id of each of the call's `n` rows -- what a per-env lookup (a prop layout) is
        keyed by. Batch position and env id coincide only for a whole-batch call."""
        return torch.arange(n, device=device) if self._rows is None else self._rows

    def _put(self, full: torch.Tensor, value: torch.Tensor) -> None:
        """Write the call's rows of a whole-batch state tensor, in place (see `GRAPH_STATE`)."""
        if self._rows is None:
            full.copy_(value)
        else:
            full.index_copy_(0, self._rows, value.to(full.dtype))

    @contextmanager
    def _row_scope(self, rows: torch.Tensor, state: torch.Tensor):
        """Plan for `rows` of a batch whose state is `state`; yields the cut for (B, ...) arguments.

        The base teacher carries two per-car settings as (B,) tensors -- its speed scale and its
        grip codes -- and `project_plan_action` tiles the codes to whatever batch it is handed by
        repeating them, so a subset call against the full-length codes would either raise or,
        where the lengths happen to divide, read another car's code without a word. They are cut to
        the same rows for the duration and put back after.
        """
        base, B = self.base, state.shape[0]
        cut = lambda t: t[rows] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == B else t
        held = (base.speed_scale, base.label_grip_codes)
        self._rows, self._full_state = rows, state
        base.speed_scale, base.label_grip_codes = cut(held[0]), cut(held[1])
        try:
            yield cut
        finally:
            base.speed_scale, base.label_grip_codes = held
            self._rows = self._full_state = None

    def project_rows(self, action: torch.Tensor, rows: torch.Tensor, state: torch.Tensor, P, v_max: float,
                     spec, plan_speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The base teacher's `project_plan_action` for an (R, ACT_DIM) `action` of `rows`."""
        with self._row_scope(rows, state) as cut:
            return self.base.project_plan_action(action, state[rows],
                                                 None if P is None else {k: cut(v) for k, v in P.items()},
                                                 v_max, spec, plan_speed=cut(plan_speed))
