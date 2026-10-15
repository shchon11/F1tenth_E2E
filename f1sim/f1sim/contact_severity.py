"""How hard a contact was: the impact severity a contact onset carries.

Every contact used to be one onset whatever it was: a 0.2 m/s graze along a hose and a 5 m/s head-on
hit cost the same, in the reward and in `collisions_per_km`. The simulator measures the difference
(`StepResult.wall_impact`, `car_impact`): the speed of the two surfaces into each other along the
contact normal, at the contact point. This module turns those per-step numbers into the severity of
an *onset* -- the rising edge the env counts and charges -- so that the env (`EnvConfig.
contact_severity_min`) and the evaluation (`learn.severity_meter`) read one definition.

Walls and props. The severity of an onset is the largest `wall_impact` since the physical contact
began -- not the onset step's alone. The contact flag is a step-end test on a nearest-cell distance
field, and the soft contact pushes a car back by a share of its penetration every few milliseconds,
so a car that hits a hose is often resolved back out of the wall cell before the step ends: the flag
rises one or more steps after the impact, on a step whose own resolutions see only the car
re-pressing at scrape speed. The physical contact is the run of steps that resolved a contact
(`wall_impact > 0`) or reported one, and a run is charged its impact at most once: after an onset the
carry restarts, so the flag flickering during a scrape gives onsets at scrape speed.

Since `SimParams.wall_latch` (2026-10-15, on by default) the flag is no longer that step-end test alone:
every wall contact the roll resolved is latched into its step, and a contact is held until the footprint is
a grid cell clear, so the flag rises on the step of the hit and the carry is the onset step's own impact.
The carry stays: it is what a `--legacy-contacts` run reads.

Cars. Soft mode has no car-car response and the car test is exact geometry at the end of the step,
so the flag rises on the step the bodies first overlap, at their approach speed: the onset step's
`car_impact` is the severity.
"""
from __future__ import annotations

import torch

#: [m/s] the reporting buckets: [0, 0.5), [0.5, 1), [1, 2), [2, inf).
BUCKET_EDGES = (0.5, 1.0, 2.0)
#: [m/s] what the reports call a hard contact.
HARD_MPS = 1.0


def carry_wall_impact(pending: torch.Tensor, impact: torch.Tensor, level: torch.Tensor):
    """One step of the wall/prop carry. `pending` (B,) is the largest impact since the physical
    contact began (or since its last onset), `impact` this step's `wall_impact`, `level` this step's
    `wall_collision`. Returns the carry including this step: an onset this step has exactly this
    severity, and the caller zeroes the carry on the rows it charged (`after_onset`)."""
    touching = (impact > 0) | level
    return torch.where(touching, torch.maximum(pending, impact), torch.zeros_like(pending))


def after_onset(carry: torch.Tensor, onset: torch.Tensor) -> torch.Tensor:
    """The carry the next step starts from: an onset has taken its impact."""
    return torch.where(onset, torch.zeros_like(carry), carry)


def bucket_of(sev: torch.Tensor) -> torch.Tensor:
    """(...) long bucket index of each severity: 0 .. len(BUCKET_EDGES)."""
    edges = torch.tensor(BUCKET_EDGES, dtype=sev.dtype, device=sev.device)
    return torch.bucketize(sev, edges, right=True)
