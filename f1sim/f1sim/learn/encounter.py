"""Obstacle encounters, counted once each.

`collisions_per_km` counts `info["contact_onset"]`, which is an edge: the step a contact begins.
A car that scrapes along a crate, separates by a centimetre and touches again logs a second edge,
so the metric is "how often it hits" multiplied by "how long it scrapes". Measured over eight
matched runs on 2026-09-23, onsets per contacted encounter ranged 1.8 to 12.8 -- and the spread
follows the draw, not the checkpoint, so the multiplier is noise sitting on top of the quantity
anyone actually wants. Two measurements of the same driving, the diagnostics harness and the
evaluation, disagreed by 3x until they were counted this way.

An **encounter** here is one approach to one piece: it opens when a piece the car is on course to
hit comes inside `window_m` of the car's body, and closes when that piece stops being the nearest
one ahead, the episode resets, or the approach drags past `max_age` steps. The rate that comes out
-- contacted encounters over encounters -- is a probability per opportunity, and opportunities are
what differ between two obstacle draws.

Three details that the older number got wrong and this one does not:

* **Walls and props are separated.** `contact_onset` merges them, and `traffic["wall_collisions"]`
  cannot split them either -- its definition is `hit & ~car`, so every crate is a wall to it. The
  split here is read where the simulator itself reads it: `Simulator.step` ORs `prop_touched` (set
  in the substeps) with an end-of-step `_prop_contact` and then zeroes the flag, so the one moment
  the answer exists is inside that call. Hence the spy.
* **Distances are to the car's body, not its centre.** A centre-based clearance is optimistic by up
  to half the diagonal: contacts on 2026-09-23 were logged at a mean surface distance of 0.32 m
  from the centre against a half-width of 0.155 m, because the car arrives at an angle and a corner
  touches first. Each face normal's separation subtracts the footprint's support along it, which is
  the same quantity SAT uses to declare the contact.
* **The learner rows only.** Opponents drive into props too, and counting them reads as the policy
  getting worse when the traffic gets denser.

The window is 6 m and the car has to be above 2 m/s (a crawling car is not approaching anything),
which is where the diagnostics harness
(`_eval/mintime-teacher-2026-09-19/diag/forced_offset.py`) drew them. Its other two thresholds are
deliberately not kept, and the rates are not comparable to its tables because of it:

* It required the piece to be **at least 2 m away** when the record opened, because it was arming
  an intervention and an intervention needs room. A metric does not: a piece that only comes into
  the lane 1.5 m ahead, which is what a corner does, is the approach most worth counting. Measured
  on s915u768 over 7.4 km, the 2 m floor left 7 of 29 prop contacts belonging to no approach at
  all; without it, 0 of 29.
* It picked candidates in the **car's frame** (10 m ahead, 2 m to either side). A piece just past
  an apex is 8 m ahead along the track and 5 m off-axis, so that window admitted it only after the
  car had turned in. Candidates here are picked in lane coordinates, which is why this meter sees
  about 106 approaches per km where the harness saw 28.
"""
from __future__ import annotations

import math

import torch

from ..prop_math import _to_world


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% interval for a proportion. Normal-approximation intervals put the bound below zero at
    these rates (2-4 contacts in 700 encounters), which is how a rate gets reported as significant
    when it is not."""
    if n <= 0:
        return None
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


class EncounterMeter:
    """Counts obstacle encounters over a rollout. Use as a context manager, like `TrafficMeter`:

        with EncounterMeter(env) as meter:
            ...                       # any loop, including one inside another function
        meter.report()

    It wraps `sim.step`, so every quantity is read before the gym layer's auto-reset -- a value read
    after `env.step` returns belongs to the next episode. It holds no host-side loop: the whole
    bookkeeping is a handful of (B,) tensors updated in place, so it costs one device sync at
    `report()` and none per step.
    """

    def __init__(self, env, *, window_m: float = 6.0, min_surface_m: float = 0.0,
                 min_speed_mps: float = 2.0, ahead_m: float = 10.0, in_lane_margin_m: float = 0.10,
                 max_age_steps: int = 200):
        self.env = env
        self.proc = getattr(env, "procedural", None)
        self.enabled = self.proc is not None and bool(getattr(env.sim.track, "has_props", False))
        self.window_m = float(window_m)
        self.min_surface_m = float(min_surface_m)
        self.min_speed_mps = float(min_speed_mps)
        self.ahead_m = float(ahead_m)
        self.in_lane_margin_m = float(in_lane_margin_m)
        self.max_age_steps = int(max_age_steps)
        self._orig_step = None
        self._orig_contact = None
        self._contact_was_attr = False
        if not self.enabled:
            return
        sim = env.sim
        dev = sim.device
        B = int(sim.state.shape[0])
        self.dt = float(sim.control_dt)
        veh = env.cfg.vehicle
        self.half_l = 0.5 * float(veh.length)
        self.half_w = 0.5 * float(veh.width)
        # The piece's own radius, so "in the way" is about the pair and not about a point. Same
        # definition the generator uses to space pieces: half the footprint diagonal.
        self.shape_rad = torch.tensor(
            [0.5 * math.hypot(s.across, s.along) for s in self.proc.shapes],
            dtype=torch.float32, device=dev)
        z_long = lambda: torch.zeros(B, dtype=torch.long, device=dev)
        z_bool = lambda: torch.zeros(B, dtype=torch.bool, device=dev)
        self._open_slot = torch.full((B,), -1, dtype=torch.long, device=dev)
        self._open_age = z_long()
        self._open_hit = z_bool()
        self._open_onsets = z_long()
        self._last_fired = torch.full((B,), -2, dtype=torch.long, device=dev)
        self._was_touching = z_bool()
        self._prop_step = z_bool()
        # Counters live on the device so no step needs a sync.
        self._n_enc = z_long()
        self._n_hit = z_long()
        self._onsets_in_hit = z_long()
        self._prop_onsets = z_long()
        self._prop_orphans = z_long()
        self._wall_onsets = z_long()
        self._car_onsets = z_long()
        self._dist_m = torch.zeros((), dtype=torch.float64, device=dev)
        self._rows = torch.arange(B, device=dev)
        self.steps = 0

    # -- attach / detach ------------------------------------------------------------------------
    def __enter__(self):
        if not self.enabled:
            return self
        sim = self.env.sim
        self._orig_step = sim.step
        self._contact_was_attr = "_prop_contact" in sim.__dict__
        self._orig_contact = sim._prop_contact

        def contact_spy(state, *a, **kw):
            out = self._orig_contact(state, *a, **kw)
            # Read `prop_touched` here and nowhere else: `Simulator.step` zeroes it on the line
            # after this call. It carries the substeps, where a box can be entered and pushed back
            # out inside one control step -- a touch that leaves no penetration at the end of it.
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
            sim.step = self._orig_step
            self._orig_step = None
        if self._orig_contact is not None:
            if self._contact_was_attr:
                sim._prop_contact = self._orig_contact
            else:
                sim.__dict__.pop("_prop_contact", None)
            self._orig_contact = None
        return False

    def observe(self, info=None) -> None:
        """No-op: folding happens in the wrapper. Here so a loop written against the meter reads."""
        return

    # -- one transition -------------------------------------------------------------------------
    def _fold(self, r, entry_steps) -> None:
        sim = self.env.sim
        proc = self.proc
        state = sim.state
        xy = state[:, :2]
        yaw = state[:, 2]
        v = state[:, 3]
        eid = sim.eid
        # `p_*` are indexed by the env-props row, which is not the car row once a layout is shared
        # between the cars in a race (`procedural_shared`). `props_near` indexes them exactly this
        # way; doing it differently here would measure another env's obstacles.
        poses = proc.p_poses[eid]                                   # (B, C, 3)
        sid = proc.p_sid[eid]                                       # (B, C)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        B, C = sid.shape
        rad_all = self.shape_rad[sid.clamp(min=0)]                  # (B, C)
        # Ahead is measured along the LANE, not along the car's nose. A piece sitting just past a
        # corner's apex is 8 m ahead on the track and 5 m off to the side in the car's frame, so a
        # body-frame window drops it until the car has already turned in -- which is precisely the
        # approach that matters. Measured: with the body-frame window, 22 of 29 prop contacts
        # belonged to no approach at all.
        chord = torch.linalg.vector_norm(poses[..., :2] - xy[:, None, :], dim=-1)
        tid_rep = sim.tid.repeat_interleave(C)
        # Seeded with the car's own centerline index: a global argmin jumps between the two lanes
        # where the lap folds back on itself, and a piece projected onto the far side of a fold
        # reads as being in a lane the car is not in. The window has to cover the search, so it is
        # the window this meter looks over plus a margin; pieces further out are cut by `chord`
        # below, which cannot be fooled by a fold because a chord is never longer than its arc.
        s_p, n_p, _ = sim.track.project(poses[..., :2].reshape(-1, 2), tid_rep,
                                        prev_idx=sim.cl_idx.repeat_interleave(C),
                                        window=self.ahead_m + 2.0)
        s_p, n_p = s_p.view(B, C), n_p.view(B, C)
        L = sim.track.length[sim.tid].clamp_min(1e-6)[:, None]
        ds = (s_p - r.s[:, None] + L / 2.0) % L - L / 2.0            # signed arc ahead of the car
        dn_all = r.lateral[:, None] - n_p
        in_way_all = dn_all.abs() < (self.half_w + rad_all + self.in_lane_margin_m)
        cand = ((sid >= 0) & (ds > 0.0) & (ds < self.ahead_m) & (chord < self.ahead_m)
                & in_way_all)
        dsel = torch.where(cand, ds, torch.full_like(ds, 1e9))
        near, tgt = dsel.min(1)
        has = near < 1e8
        g = tgt.clamp(min=0)
        pose_t = poses[self._rows, g]                               # (B, 3)
        pn_t = proc.p_n[eid, g]                                     # (B, K, 2)
        pd_t = proc.p_d[eid, g]                                     # (B, K)
        valid = pn_t.pow(2).sum(-1) > 0.5
        nw, dw = _to_world(pn_t, pd_t, pose_t)
        # Separation from the car's BODY to the piece, along each of the piece's face normals: the
        # centre-to-plane distance less the footprint's support in that direction. Max over the
        # faces is the SAT separation -- negative means overlap, which is the same test the
        # simulator resolves a contact with.
        support = ((nw[..., 0] * cy[:, None] + nw[..., 1] * sy[:, None]).abs() * self.half_l
                   + (-nw[..., 0] * sy[:, None] + nw[..., 1] * cy[:, None]).abs() * self.half_w)
        sd = (nw * xy[:, None, :]).sum(-1) - dw - support
        d_body = torch.where(valid, sd, torch.full_like(sd, -float("inf"))).max(-1).values
        tgt = torch.where(has, tgt, torch.full_like(tgt, -1))

        # The same edge the gym layer publishes as `contact_onset`, recomputed here because the gym
        # layer rewrites `r.collision` into that edge only after this wrapper has returned.
        level = r.collision
        onset = level & ~self._was_touching
        self._was_touching.copy_(level)
        car = (r.car_collision if getattr(r, "car_collision", None) is not None
               else torch.zeros_like(onset))
        # Read the roles every step, not once: a reset redraws which rows are learners
        # (`gym_env` rebinds `self.learner`), and a mask cached at construction would keep scoring
        # the row an opponent now drives.
        lm = self.env.learner
        prop_onset = onset & self._prop_step & ~car & lm
        car_onset = onset & car & lm
        wall_onset = onset & ~self._prop_step & ~car & lm
        self._prop_onsets += prop_onset.long()
        self._car_onsets += car_onset.long()
        self._wall_onsets += wall_onset.long()
        self._dist_m += (v.abs() * lm.float()).sum().double() * self.dt

        fresh = entry_steps == 0
        self._last_fired = torch.where(fresh, torch.full_like(self._last_fired, -2), self._last_fired)
        is_open = self._open_slot >= 0
        # Charge the contact to the open record BEFORE closing it: a touch on the step the piece
        # goes behind is still that approach's touch.
        self._open_hit |= is_open & prop_onset
        self._open_onsets += (is_open & prop_onset).long()
        # A prop contact with no open record. Reported, because a rate whose denominator misses
        # most of the contacts is a rate about something else and nothing in the rate would show
        # it. Traced on 2026-09-23: with a raceline teacher, which avoids nothing, all five of
        # these were a car ALREADY wedged against a crate touching it again -- 0.1 to 1.9 m/s for
        # fourteen straight steps with the piece 0.5 m away and not receding. That is not a new
        # opportunity, and excluding it is the point: it is exactly the repetition that made
        # `collisions_per_km` a measure of how long the car scrapes. With a policy driving at
        # racing speed, 0 of 29 prop contacts fell outside an approach.
        self._prop_orphans += (prop_onset & ~is_open).long()
        self._open_age += is_open.long()
        # The approach ends when the PIECE is behind the car, not when it stops being the nearest
        # thing ahead. Those are different moments and the difference is where the contacts are:
        # the candidate test needs the piece's centre ahead of the CoG, and a piece the car is
        # about to scrape is 0.3 m ahead of the CoG at most, so closing on "no longer the target"
        # shut the record one or two steps BEFORE the touch it was opened to measure. Measured on
        # s915u768 over 7.4 km with the old rule: 24 of 29 prop contacts fell outside any open
        # record. The car's half-length plus the piece's radius is the distance at which the two
        # footprints can no longer overlap.
        og = self._open_slot.clamp(min=0)[:, None]
        ds_open = ds.gather(1, og).squeeze(1)
        rad_open = rad_all.gather(1, og).squeeze(1)
        past = ds_open < -(self.half_l + rad_open)
        # A different piece meeting the trigger does end the record, though: that is a second
        # approach starting, and merging the two would hide one opportunity inside the other.
        trig = (has & (d_body < self.window_m) & (d_body > self.min_surface_m)
                & (v > self.min_speed_mps))
        close = is_open & (fresh | past | (trig & (tgt != self._open_slot))
                           | (self._open_age >= self.max_age_steps))
        self._n_enc += close.long()
        self._n_hit += (close & self._open_hit).long()
        self._onsets_in_hit += torch.where(close & self._open_hit, self._open_onsets,
                                           torch.zeros_like(self._open_onsets))
        keep = ~close
        self._open_slot = torch.where(keep, self._open_slot, torch.full_like(self._open_slot, -1))
        self._open_age *= keep.long()
        self._open_hit &= keep
        self._open_onsets *= keep.long()

        # Fire after closing, so the next piece can open on the step the last one was passed.
        fire = trig & lm & (self._open_slot < 0) & (tgt != self._last_fired)
        self._open_slot = torch.where(fire, tgt, self._open_slot)
        self._last_fired = torch.where(fire, tgt, self._last_fired)
        self._open_age = torch.where(fire, torch.zeros_like(self._open_age), self._open_age)
        self._open_hit &= ~fire
        self._open_onsets = torch.where(fire, torch.zeros_like(self._open_onsets), self._open_onsets)
        self.steps += 1

    # -- result ---------------------------------------------------------------------------------
    def report(self) -> dict:
        if not self.enabled:
            return {"obstacle_encounters": None,
                    "obstacle_encounters_note": "no procedural obstacles in this env"}
        # Records still open at the last step never passed through a close. Counting them keeps the
        # denominator honest -- they were approaches -- and they carry whatever they touched.
        still = self._open_slot >= 0
        n_enc = int((self._n_enc.sum() + still.long().sum()).item())
        n_hit = int((self._n_hit.sum() + (still & self._open_hit).long().sum()).item())
        onsets_hit = int((self._onsets_in_hit.sum()
                          + torch.where(still & self._open_hit, self._open_onsets,
                                        torch.zeros_like(self._open_onsets)).sum()).item())
        dist_m = float(self._dist_m.item())
        km = dist_m / 1000.0
        ci = _wilson(n_hit, n_enc)
        # undefined is None, not nan: the report is written as strict JSON, and a run with no
        # contact at all (no contacted encounter to divide by) used to fail to save.
        per_km = lambda c: (c / km if km > 0.001 else None)
        prop = int(self._prop_onsets.sum().item())
        wall = int(self._wall_onsets.sum().item())
        cars = int(self._car_onsets.sum().item())
        return {"obstacle_encounters": {
            "encounters": n_enc,
            "contacted": n_hit,
            "contact_rate": (n_hit / n_enc) if n_enc else None,
            "contact_rate_ci95": list(ci) if ci else None,
            "onsets_per_contacted_encounter": (onsets_hit / n_hit) if n_hit else None,
            "open_at_end": int(still.long().sum().item()),
            "prop_onsets": prop, "wall_onsets": wall, "car_onsets": cars,
            "prop_onsets_outside_encounter": int(self._prop_orphans.sum().item()),
            "prop_onsets_per_km": per_km(prop),
            "wall_onsets_per_km": per_km(wall),
            "car_onsets_per_km": per_km(cars),
            "learner_distance_m": dist_m,
            "steps": self.steps,
            "window_m": self.window_m, "min_surface_m": self.min_surface_m,
            "min_speed_mps": self.min_speed_mps, "max_age_steps": self.max_age_steps,
            "in_lane_margin_m": self.in_lane_margin_m,
            "note": ("an encounter is one approach to one piece, opened when it comes inside "
                     "window_m of the car's BODY while in the lane ahead and closed when it is "
                     "passed; distances subtract the footprint's support, not the centre. "
                     "Approaches cluster by track section, so the Wilson interval is optimistic: "
                     "it assumes independent trials."),
        }}
