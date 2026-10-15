"""Contact onsets by how hard they were (`f1sim.contact_severity`), for an evaluation.

`collisions_per_km` counts a 0.2 m/s graze along a hose and a 5 m/s head-on hit as one contact each.
This meter counts the same onsets the env charges -- a wall or prop contact's onset as the `collision`
reward component sees it (the rising edge of `wall_collision`), a car contact's as the car constraint
sees it (the rising edge of `car_collision`) -- and files each by its impact severity: the speed of
the two surfaces into each other along the contact normal, at the contact point, in the buckets
[0, 0.5), [0.5, 1), [1, 2), [2, inf) m/s, per km of the learners' own driving.

It also counts the wall and prop contacts **physically**: a run of steps on which the soft contact
resolved something (`wall_impact > 0`) or the flag was up, with the run's largest impact. The flag
is a step-end test on a nearest-cell distance field and the contact pushes the car back out every few
milliseconds, so a contact can begin steps before its flag rises, or be resolved and left inside one
step and never be flagged at all. The runs say how many contacts the onsets miss, and how hard those
were.

Like `EncounterMeter`, it wraps `sim.step` -- every quantity is read before the gym layer's
auto-reset -- keeps the learner rows only (roles are read every step), and holds its counters on the
device: one sync at `report()`. Entering it switches `sim.measure_impacts` on (car severities, and a
terminate crash's); that is eager work after the roll and changes nothing the physics computes.
"""
from __future__ import annotations

import math

import torch

from ..contact_severity import BUCKET_EDGES, HARD_MPS, after_onset, carry_wall_impact

#: [m/s] the fine histogram's bin width and top (one overflow bin past it), for the distribution's shape
FINE_BIN = 0.1
FINE_TOP = 10.0
#: `prop` / `prop_physical` are the subsets of `wall_prop` / `wall_prop_physical` that touched a prop
KINDS = ("wall_prop", "wall_prop_onset_step", "wall_prop_physical", "wall_prop_physical_unflagged", "car",
         "prop", "prop_physical")


class ContactSeverityMeter:
    def __init__(self, env):
        self.env = env
        sim = env.sim
        dev = sim.device
        B = int(sim.state.shape[0])
        self.dt = float(sim.control_dt)
        self.races = sim.M > 1
        self.edges = torch.tensor(BUCKET_EDGES, dtype=torch.float32, device=dev)
        self.n_fine = int(round(FINE_TOP / FINE_BIN)) + 1
        zb = lambda: torch.zeros(B, dtype=torch.bool, device=dev)
        zf = lambda: torch.zeros(B, dtype=torch.float32, device=dev)
        self._wall_prev, self._car_prev = zb(), zb()
        self._pend = zf()
        self._run_on, self._run_flagged, self._run_learner, self._run_prop = zb(), zb(), zb(), zb()
        self._prop_step = zb()                 # a prop was touched this step (the substeps included)
        self._run_peak = zf()
        self._cnt = {k: torch.zeros(len(BUCKET_EDGES) + 1, dtype=torch.long, device=dev) for k in KINDS}
        self._fine = {k: torch.zeros(self.n_fine, dtype=torch.long, device=dev) for k in KINDS}
        self._sum = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in KINDS}
        self._max = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in KINDS}
        # onsets the carry put in a higher bucket than the onset step's own resolutions did
        self._lifted = torch.zeros((), dtype=torch.long, device=dev)
        self._dist = torch.zeros((), dtype=torch.float64, device=dev)
        self._prog = torch.zeros((), dtype=torch.float64, device=dev)
        self.steps = 0
        self._orig_step = None
        self._orig_measure = None
        self._orig_contact = None
        self._contact_was_attr = False
        self.mode = "soft" if not sim.cfg.sim.terminate_on_collision else "terminate"

    # -- attach / detach ------------------------------------------------------------------------
    def __enter__(self):
        sim = self.env.sim
        self._orig_step = sim.step
        self._orig_measure = sim.measure_impacts
        sim.measure_impacts = True
        self._contact_was_attr = "_prop_contact" in sim.__dict__
        self._orig_contact = sim._prop_contact

        def contact_spy(state, *a, **kw):
            # `EncounterMeter`'s split: `Simulator.step` ORs `prop_touched` (the substeps) with this
            # end-of-step test and zeroes it right after, so here is the one place both are readable
            out = self._orig_contact(state, *a, **kw)
            self._prop_step |= (out[0] > 0) | sim.prop_touched
            return out

        def step_spy(*a, **kw):
            entry_steps = sim.steps.clone()
            self._prop_step.zero_()
            r = self._orig_step(*a, **kw)
            self._fold(r, entry_steps)
            return r

        sim._prop_contact = contact_spy
        sim.step = step_spy
        return self

    def __exit__(self, *exc):
        sim = self.env.sim
        if self._orig_step is not None:
            sim.step = self._orig_step                     # whatever was there: the class's, or an outer wrapper
            self._orig_step = None
        if self._orig_measure is not None:
            sim.measure_impacts = self._orig_measure
            self._orig_measure = None
        if self._orig_contact is not None:
            if self._contact_was_attr:
                sim._prop_contact = self._orig_contact
            else:
                sim.__dict__.pop("_prop_contact", None)
            self._orig_contact = None
        return False

    def observe(self, info=None) -> None:
        """No-op: folding happens in the wrapper."""
        return

    # -- one transition -------------------------------------------------------------------------
    def _add(self, kind: str, sev: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.long()
        sev = sev.float()
        self._cnt[kind].index_add_(0, torch.bucketize(sev, self.edges, right=True), m)
        fine = (sev / FINE_BIN).floor().long().clamp(0, self.n_fine - 1)
        self._fine[kind].index_add_(0, fine, m)
        ms = sev * mask.float()
        self._sum[kind] += ms.sum().double()
        self._max[kind] = torch.maximum(self._max[kind], ms.max().double())

    def _fold(self, r, entry_steps) -> None:
        fresh = entry_steps == 0                 # the first step of an episode: nothing carries over
        lm = self.env.learner
        level = r.wall_collision
        imp = r.wall_impact
        if level is None or imp is None:         # nothing to file (no wall flag, or nothing measured)
            self.steps += 1
            return
        kinds = getattr(r, "contact_onsets", None)
        prop = self._prop_step
        if kinds is not None:
            # The simulator's onsets per kind (`SimParams.wall_latch`): the flag rises on the step of the hit,
            # so an onset's severity is its own step's impact for its own kind, and a crate hit during a wall
            # contact is filed as well. `wall_prop` is one per step, as the collision component charges it.
            ki = r.contact_impacts.float()
            onset = kinds[:, 0] | kinds[:, 1]
            self._wall_prev = level.clone()
            carry = torch.maximum(torch.where(kinds[:, 0], ki[:, 0], torch.zeros_like(ki[:, 0])),
                                  torch.where(kinds[:, 1], ki[:, 1], torch.zeros_like(ki[:, 1])))
            on_l = onset & lm
            self._add("wall_prop", carry, on_l)
            self._add("prop", ki[:, 1], kinds[:, 1] & lm)
            self._add("wall_prop_onset_step", carry, on_l)
        else:
            onset = level & ~(self._wall_prev & ~fresh)
            self._wall_prev = level.clone()
            carry = carry_wall_impact(torch.where(fresh, torch.zeros_like(self._pend), self._pend), imp.float(), level)
            self._pend = after_onset(carry, onset)
            on_l = onset & lm
            self._add("wall_prop", carry, on_l)
            self._add("prop", carry, on_l & prop)
            self._add("wall_prop_onset_step", imp, on_l)
            bc = torch.bucketize(carry, self.edges, right=True)
            bs = torch.bucketize(imp.float(), self.edges, right=True)
            self._lifted += ((bc > bs) & on_l).long().sum()
        # physical runs: open while the roll resolved something or the flag is up
        touch = (imp > 0) | level
        closing = self._run_on & (fresh | ~touch)
        self._add("wall_prop_physical", self._run_peak, closing & self._run_learner)
        self._add("wall_prop_physical_unflagged", self._run_peak, closing & self._run_learner & ~self._run_flagged)
        self._add("prop_physical", self._run_peak, closing & self._run_learner & self._run_prop)
        start = touch & (fresh | ~self._run_on)
        cont = touch & ~start
        self._run_peak = torch.where(start, imp.float(), torch.where(cont, torch.maximum(self._run_peak, imp.float()),
                                                                     torch.zeros_like(self._run_peak)))
        self._run_flagged = torch.where(start, onset, (self._run_flagged | onset) & cont)
        self._run_learner = torch.where(start, lm, self._run_learner & cont)
        self._run_prop = torch.where(start, prop, (self._run_prop | prop) & cont)
        self._run_on = touch.clone()
        if self.races and r.car_collision is not None and r.car_impact is not None:
            c_on = r.car_collision & ~(self._car_prev & ~fresh) if kinds is None else kinds[:, 2]
            self._car_prev = r.car_collision.clone()
            self._add("car", r.car_impact, c_on & lm)
        self._dist += (r.state[:, 3].abs() * lm.float()).sum().double() * self.dt
        self._prog += (r.progress * lm.float()).sum().double()
        self.steps += 1

    # -- result ---------------------------------------------------------------------------------
    def report(self) -> dict:
        # runs still open at the end were contacts too
        self._add("wall_prop_physical", self._run_peak, self._run_on & self._run_learner)
        self._add("wall_prop_physical_unflagged", self._run_peak, self._run_on & self._run_learner & ~self._run_flagged)
        self._add("prop_physical", self._run_peak, self._run_on & self._run_learner & self._run_prop)
        self._run_on = torch.zeros_like(self._run_on)
        km = float(self._dist.item()) / 1000.0
        per_km = lambda c: (c / km if km > 0.001 else None)
        hard_from = sum(1 for e in BUCKET_EDGES if e <= HARD_MPS)         # the bucket that starts at HARD_MPS
        out = {}
        for k in KINDS:
            if k == "car" and not self.races:
                continue
            cnt = [int(x) for x in self._cnt[k].tolist()]
            n = sum(cnt)
            fine = [int(x) for x in self._fine[k].tolist()]
            while fine and fine[-1] == 0:
                fine.pop()
            hard = sum(cnt[hard_from:])
            out[k] = {"onsets": n, "counts": cnt, "per_km": [per_km(c) for c in cnt],
                      "hard": hard, "hard_per_km": per_km(hard), "all_per_km": per_km(n),
                      "mean_mps": (float(self._sum[k].item()) / n) if n else None,
                      "max_mps": float(self._max[k].item()) if n else None,
                      "fine_hist": fine}
        edges = [0.0] + list(BUCKET_EDGES) + [None]
        return {"contact_severity": {
            **out,
            "onsets_lifted_by_carry": int(self._lifted.item()),
            "buckets_mps": [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)],
            "hard_mps": HARD_MPS, "fine_bin_mps": FINE_BIN,
            "learner_distance_m": float(self._dist.item()), "learner_progress_m": float(self._prog.item()),
            "steps": self.steps, "collision_mode": self.mode,
            "note": ("severity = speed of the two surfaces into each other along the contact normal at the "
                     "contact point [m/s]. wall_prop: onsets of wall_collision (what the collision component "
                     "charges), severity carried from the physical contact's start (f1sim.contact_severity); "
                     "wall_prop_onset_step: the same onsets, the onset step's own resolutions only; "
                     "wall_prop_physical: resolved contacts as runs of steps, flagged or not, each its peak; "
                     "_unflagged: those runs no onset counted. prop / prop_physical: the subsets that touched a "
                     "prop (the rest are walls). car: onsets of car_collision, closing speed at the onset step. "
                     "per_km over the learners' distance driven (|vx| dt)."),
        }}
