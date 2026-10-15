"""Contact physics of 2026-10-15 (feat/v2-physics): what a contact does, and that each one is reported.

The user's goal: run fast without crashing -- a graze is fine, a hard bang is not -- and contacts that look
and act like the real thing in the console: cars that shove each other, crates that are pushed, a hose that
gives. Every switch here is on by default and `params.legacy_contacts` turns all of them off, which is
feat/v2-severity bit for bit (the A/B hashes are in the commit messages).

Scenes are scripted on the open floor of `test_contact_severity` (duct-hose walls on the 5 cm grid of the
real map), so each contact's true speeds are known.
"""
from __future__ import annotations

import math

import pytest
import torch

from f1sim.params import CONTACT_FIXES, Config, legacy_contacts
from f1sim.sim import Simulator
from test_contact_severity import _cfg, _floor


def _onsets(flags):
    return [i for i in range(len(flags)) if flags[i] and (i == 0 or not flags[i - 1])]


def _wall_run(latch: bool, plan, pose=(10.0, 0.75, -math.pi / 2), v0=None):
    """One car on the open floor driven by the speed commands in `plan`; per step (flag, impact)."""
    cfg = _cfg()
    cfg.sim.wall_latch = latch
    sim = Simulator(_floor(), cfg, num_envs=1, device="cpu")
    sim.reset(torch.arange(1), torch.tensor([list(pose)]), torch.tensor([plan[0] if v0 is None else v0]))
    flags, imps = [], []
    for v in plan:
        r = sim.step(torch.tensor([[0.0, float(v)]]))
        flags.append(bool(r.wall_collision[0]))
        imps.append(float(r.wall_impact[0]))
    return flags, imps


def test_the_switches_default_on_and_legacy_turns_them_off():
    cfg = Config()
    assert all(getattr(cfg.sim, k) is True for k in CONTACT_FIXES)
    legacy_contacts(cfg)
    assert all(getattr(cfg.sim, k) is False for k in CONTACT_FIXES)


# ------------------------------------------------------------------ 1. every wall contact is counted
@pytest.mark.parametrize("ang, want", [(0.0, 1.5), (0.6, 1.5 * math.cos(0.6))])
def test_a_duct_hit_at_1_5_m_s_is_flagged_once_with_its_impact(ang, want):
    """Square and at 0.6 rad, then the car brakes and settles against the hose. One contact, one onset, its
    severity the approach speed; before the latch the step-end grid test never saw this hit at all."""
    plan = [1.5] * 9 + [0.0] * 51                     # the first contact comes at step 7 or 8
    flags, imps = _wall_run(True, plan, pose=(10.0, 0.75, -math.pi / 2 + ang))
    on = _onsets(flags)
    assert len(on) == 1, (on, imps)
    first = next(i for i, x in enumerate(imps) if x > 0)
    assert on[0] == first                              # flagged on the step the hit was resolved
    assert abs(imps[first] - want) < 0.03 * want, (imps[first], want)
    # the settling contacts after it -- the hose pushing the car 6 mm out, the car creeping back -- are
    # the same contact: the grid cannot tell a gap under one cell from contact
    assert sum(1 for x in imps if x > 0) > 4
    legacy, _ = _wall_run(False, plan, pose=(10.0, 0.75, -math.pi / 2 + ang))
    assert not any(legacy), "the old step-end test used to miss this hit; the scene no longer shows the bug"


def test_backing_a_cell_away_ends_the_contact():
    """Hit, reverse half a metre, hit again: two contacts, each with its own impact."""
    plan = [1.5] * 12 + [-1.0] * 14 + [1.5] * 30
    flags, imps = _wall_run(True, plan, v0=1.5)
    on = _onsets(flags)
    assert len(on) == 2, (on, imps)
    assert imps[on[0]] > 1.4 and imps[on[1]] > 0.9


def test_the_latch_is_soft_contact_only():
    """Under terminate the step-end test is the crash and nothing is resolved; the switch changes nothing."""
    out = []
    for latch in (False, True):
        cfg = _cfg(terminate=True)
        cfg.sim.wall_latch = latch
        sim = Simulator(_floor(), cfg, num_envs=1, device="cpu")
        sim.reset(torch.arange(1), torch.tensor([[10.0, 0.75, -math.pi / 2]]), torch.tensor([3.0]))
        out.append(torch.stack([sim.step(torch.tensor([[0.0, 3.0]])).state for _ in range(12)]))
    assert torch.equal(out[0], out[1])
