"""The hybrid wp4 label (2026-10-27): a trained policy's own plan, its waypoint speeds raised to the raceline teacher's
where that teacher is the speed authority -- its guard chose full pace (candidate 0: nothing in its 1 s preview, cars and
props included, asked it to slow) and the two paths stay within `dev` of each other over the plan.

Why: the Korea entry corners faster than any teacher we have (a wider hairpin line, learned over weeks of PPO) but never
commands more than ~6.5 m/s on the straights, where the fixed teacher (min-time line for a_lat 9, margin 0.35) drives
8-9: its labels came from the v1 policy's (`docs/research/teacher-speed-2026-10-26`). A general DAgger from the fixed
teacher brought the straights back but lost the corners and the racing, and a short Korea PPO then slowed it again.
Driven as a rule before any training on it (10-27, Korea, seeds 81-82, the entry's own MAP): solo 8.59 -> 8.52 s
(dev 0.15) / 8.42 s (dev 0.30) with 0 contacts; races of 2 vs 4 kinds (dev 0.15, 36 km): 4.64 -> 4.71 m/s, hard
1.07 -> 0.99 /km, passes 0.63 -> 0.67 /min.

The max of two brake-feasible speed profiles is brake-feasible, and the decoder's backward pass keeps every plan so; the
curvature knots are the policy's own throughout."""
from __future__ import annotations

import os
from typing import Optional

import torch

from .. import mpc
from . import conditioning as cond_mod
from .model import load_checkpoint


def combine(a_pol: torch.Tensor, a_teacher: torch.Tensor, clear: torch.Tensor, v_meas: torch.Tensor,
            cap: torch.Tensor, spec, v_max: float, dev: float):
    """-> (label (R, D), raised (R,) bool): `a_pol` with its speed dimensions replaced by max(policy, teacher)
    waypoint speeds on the rows where `clear` (R,) and the decoded paths stay within `dev` [m] of each other;
    elsewhere `a_pol` untouched. a_pol / a_teacher (R, D): normalised wp4 plans; v_meas, cap (R,)."""
    kP, L, vP = mpc.decode_wp4(a_pol.float(), v_meas.float(), v_max, cap.float(), spec)
    kT, _, vT = mpc.decode_wp4(a_teacher.float(), v_meas.float(), v_max, cap.float(), spec)
    xP, yP = mpc.wp4_path(kP, L, spec, extend=1.0)[:2]
    xT, yT = mpc.wp4_path(kT, L, spec, extend=1.0)[:2]
    near = torch.hypot(xP - xT, yP - yT).amax(1) < float(dev)
    raised = clear.to(near.device).bool() & near & (vT > vP + 1e-3).any(1)
    v = torch.maximum(vP, vT)
    sp = mpc.plan_layout(spec).speed
    out = a_pol.clone()
    enc = (v / float(v_max) * 2 - 1).clamp(-1, 1).to(a_pol.dtype)
    out[:, sp] = torch.where(raised[:, None], enc, a_pol[:, sp])
    return out, raised


class PolicyV2Teacher:
    """A frozen v2 checkpoint acting on the learners' own observation -- a v2 observation carries no issued actions
    (`ego_actions` off), so the student's observation is exactly the one it would have driving -- with its own recurrent
    state, zeroed where an episode starts, and its dial held (default: the `dial_constant` it records)."""

    def __init__(self, path: str, device, dial: Optional[float] = None):
        model, extra = load_checkpoint(path, device, allow_conditional=True)
        if (model.meta or {}).get("arch") != "v2":
            raise ValueError(f"{os.path.basename(path)} is not a v2 checkpoint; the hybrid label's policy is a v2 policy")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model, self.device = model, device
        held = ((extra or {}).get("experiment") or {}).get("dial_constant")
        self.mu = float(dial if dial is not None else (held if held is not None else 1.0489))
        self.cspec = cond_mod.CondSpec.from_meta(model.meta["cond"])
        self.h = None
        self.last_action: Optional[torch.Tensor] = None
        self.name = os.path.basename(path)

    @torch.no_grad()
    def observe(self, scan: torch.Tensor, ego: torch.Tensor, new_ep: torch.Tensor) -> torch.Tensor:
        """Advance on this step's observation (B rows); returns and keeps its plan (B, D)."""
        B = scan.shape[0]
        if self.h is None or self.h.shape[1] != B:
            self.h = self.model.actor.initial_hidden(B, device=self.device)
        self.h = self.h * (~new_ep.to(self.h.device)).to(self.h.dtype)[None, :, None]
        c = cond_mod.mu_to_c(torch.full((B,), self.mu, device=scan.device, dtype=scan.dtype), self.cspec).to(scan.dtype)
        a, self.h = self.model.actor.step(scan.to(self.device), ego.to(self.device), c.to(self.device), self.h)
        self.last_action = a
        return a

    def describe(self) -> str:
        return f"policy {self.name} (v2, dial {self.mu:g}, its own observation and recurrent state)"
