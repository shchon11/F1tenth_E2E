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


# ------------------------------------------------------------------ 2. props: the impulse, one set per race
BOX = (12.0, 15.0)


def _prop_scene(style: str = "cardboard_box", relative: bool = True, share: bool = True, B: int = 2):
    """A race of `B` cars on the open floor and one prop of `style` at `BOX`, movable, every other slot dead."""
    from f1sim.procedural_obstacles import ProceduralObstacles
    cfg = _cfg()
    cfg.sim.prop_relative = relative
    cfg.sim.props_per_race = share
    sim = Simulator(_floor(), cfg, num_envs=B, device="cpu", race_size=B)
    po = ProceduralObstacles(sim.track, B, sim.gen, density=1.0, max_props=4)
    sim.track.attach_env_props(po)
    sim.movable_obstacles = True
    sim.prop_reach = float(sim.corners.norm(dim=1).max()) + po.max_radius
    po.share_m = B if share else 1
    sid = next(i for i, s in enumerate(po.shapes) if s.style == style)
    po.p_poses.zero_(); po.p_zlo.zero_(); po.p_zhi.fill_(-1.0); po.p_mass.zero_(); po.p_vel.zero_()
    po.p_sid.fill_(-1)
    po.p_poses[:, 0] = torch.tensor([BOX[0], BOX[1], 0.0])
    po.p_n[:, 0] = po.sh_n[sid]; po.p_d[:, 0] = po.sh_d[sid]
    po.p_zlo[:, 0] = po.sh_z0[sid]; po.p_zhi[:, 0] = po.sh_z1[sid]; po.p_mass[:, 0] = po.sh_mass[sid]
    po.p_sid[:, 0] = sid
    return sim, po


def _touching_pose(sim, y: float, yaw: float = 0.0, depth: float = 0.004):
    """x at which a car at (x, y, yaw) overlaps the prop by about `depth`, walking in from the west."""
    for i in range(2000):
        x = BOX[0] - 1.2 + 0.001 * i
        st = torch.zeros(sim.B, sim.state.shape[1])
        st[:, 0], st[:, 1], st[:, 2] = x, y, yaw
        st[1:, 0], st[1:, 1] = 25.0, 25.0                        # the other cars far away
        if float(sim._prop_contact(st)[0][0]) > depth:
            return st
    raise AssertionError("never reached the prop")


def _energy_momentum(sim, po, st):
    m, Iz = float(sim.P["m"][0]), float(sim.P["Iz"][0])
    yaw = float(st[0, 2]); c, s = math.cos(yaw), math.sin(yaw)
    vx, vy, w = float(st[0, 3]), float(st[0, 4]), float(st[0, 5])
    v = torch.tensor([vx * c - vy * s, vx * s + vy * c])
    mp = float(po.p_mass[0, 0]); vp = po.p_vel[0, 0].clone()
    ke = 0.5 * m * float(v @ v) + 0.5 * Iz * w * w + 0.5 * mp * float(vp @ vp)
    return ke, m * v + mp * vp


@pytest.mark.parametrize("style, y, yaw, v, w", [
    ("cardboard_box", 15.0, 0.0, 3.0, 0.0),             # square, a 1.2 kg box
    ("cardboard_box", 15.13, 0.0, 3.0, 0.0),            # a front corner
    ("cardboard_box", 15.13, 0.3, 3.0, 4.0),            # a corner, the car spinning and at an angle
    ("wooden_crate", 15.0, 0.0, 4.0, 0.0),              # 10 kg
    ("steel_drum", 15.1, -0.2, 2.0, -3.0),              # 18 kg, off centre, spinning the other way
])
def test_a_prop_impulse_conserves_momentum_and_creates_no_energy(style, y, yaw, v, w):
    sim, po = _prop_scene(style)
    st = _touching_pose(sim, y, yaw)
    st[0, 3], st[0, 5] = v, w
    ke0, p0 = _energy_momentum(sim, po, st)
    out, _a = sim._resolve_wall_contact(st, dt=0.005)
    ke1, p1 = _energy_momentum(sim, po, out)
    assert float(po.p_vel[0, 0].norm()) > 0.1, "the prop was not pushed"
    assert torch.allclose(p1, p0, atol=1e-4 * float(p0.norm())), (p0, p1)
    assert ke1 <= ke0 * (1 + 1e-6), (ke0, ke1)
    assert torch.equal(po.p_vel[1, 0], po.p_vel[0, 0])          # the race's other car holds the same crate


def test_one_impact_is_one_impulse():
    """3 m/s into a 1.2 kg box: it leaves at (1+e) m_c/(m_c+m_p) of the closing speed (less the floor's pull
    over the rest of the step) and the car keeps the rest. The old impulse took the car's own speed at each of
    the step's resolutions while the box stood still: 7.6 m/s for the box, 0.53 left to the car."""
    out = {}
    for rel in (False, True):
        sim, po = _prop_scene(relative=rel)
        sim.reset(torch.arange(2), torch.tensor([[10.6, 15.0, 0.0], [20.0, 25.0, 0.0]]), torch.tensor([3.0, 0.0]))
        v_hit = None
        for t in range(25):
            r = sim.step(torch.tensor([[0.0, 3.0], [0.0, 0.0]]))
            if float(r.wall_impact[0]) > 0:
                v_hit, box, car = float(r.wall_impact[0]), float(po.p_vel[0, 0, 0]), float(r.state[0, 3])
                break
            v_before = float(r.state[0, 3])
            if not rel:
                po.advance(sim.control_dt)                      # what the env did after every step
        assert v_hit is not None
        out[rel] = (v_hit, box, car, v_before)
    v_hit, box, car, v0 = out[True]
    m_c, m_p, e = 3.74, 1.2, 0.2
    want = (1 + e) * m_c / (m_c + m_p) * v_hit
    assert want - 0.13 < box <= want + 1e-3, (box, want)        # one impulse, then up to 20 ms of floor
    assert 2.0 < car < 2.4
    old_box = out[False][1]
    assert old_box > 2.4 * out[False][0], "the scene no longer shows the old impulse's energy gain"


def test_two_cars_of_a_race_push_one_crate_and_add_up():
    sim, po = _prop_scene("wooden_crate")
    slot = torch.zeros(2, dtype=torch.long)
    po.push(torch.arange(2), slot, torch.tensor([[0.3, 0.0], [0.2, 0.1]]), torch.tensor([[0.01, 0.0], [0.0, 0.02]]))
    for row in range(2):
        assert torch.allclose(po.p_vel[row, 0], torch.tensor([0.5, 0.1]))
        assert torch.allclose(po.p_poses[row, 0, :2], torch.tensor([BOX[0] + 0.01, BOX[1] + 0.02]))
    sim1, po1 = _prop_scene("wooden_crate", share=False)
    po1.push(torch.arange(2), slot, torch.tensor([[0.3, 0.0], [0.2, 0.1]]), torch.zeros(2, 2))
    assert torch.allclose(po1.p_vel[0, 0], torch.tensor([0.3, 0.0]))     # a set per car: each its own


@pytest.mark.parametrize("share", [True, False])
def test_the_other_car_meets_the_crate_where_it_was_pushed(share):
    """Car 0 shoves a box 0.6 m along; car 1 then drives across the box's old place and its new one. With a
    set per race it passes the empty old place and hits the box where it is; with a set per car (the old way)
    it hit a box that was no longer there and drove through the one that was."""
    sim, po = _prop_scene(share=share)
    sim.reset(torch.arange(2), torch.tensor([[10.6, 15.0, 0.0], [20.0, 25.0, 0.0]]), torch.tensor([3.0, 0.0]))
    for t in range(60):
        r = sim.step(torch.tensor([[0.0, 3.0 if t < 14 else -1.0], [0.0, 0.0]]))
    moved = float(po.p_poses[0, 0, 0])
    assert moved > BOX[0] + 0.4 and float(po.p_vel[0, 0].norm()) == 0.0
    contacts = {}
    for name, x in (("old place", BOX[0]), ("new place", moved)):
        hits = 0
        sim.reset(torch.tensor([1]), torch.tensor([[x, BOX[1] - 1.2, math.pi / 2]]), torch.tensor([1.5]))
        for t in range(30):
            r = sim.step(torch.tensor([[0.0, 0.0], [0.0, 1.5]]))
            hits += int(bool(sim.prop_touched[1]) or float(r.wall_impact[1]) > 0)
        sim.reset(torch.tensor([1]), torch.tensor([[20.0, 25.0, 0.0]]), torch.tensor([0.0]))
        contacts[name] = hits
    if share:
        assert contacts["old place"] == 0 and contacts["new place"] > 0, contacts
    else:
        assert contacts["old place"] > 0 and contacts["new place"] == 0, contacts


def test_a_crate_hit_while_scraping_a_hose_is_its_own_onset():
    """A car scrapes along the hose into a crate standing against it. The wall contact holds the whole time
    (the merged flag never drops), so the hit was no rising edge of it and nothing counted or charged it;
    per kind it is a prop onset with its own 1.8 m/s impact."""
    sim, po = _prop_scene("wooden_crate")
    po.p_poses[:, 0] = torch.tensor([12.0, 0.45, 0.0])
    sim.reset(torch.arange(2), torch.tensor([[10.0, 0.38, -0.08], [20.0, 25.0, 0.0]]), torch.tensor([2.0, 0.0]))
    flags, on, imp = [], [], []
    for _ in range(40):
        r = sim.step(torch.tensor([[0.0, 2.0], [0.0, 0.0]]))
        flags.append(bool(r.wall_collision[0]))
        on.append(r.contact_onsets[0].clone())
        imp.append(r.contact_impacts[0].clone())
    on, imp = torch.stack(on), torch.stack(imp)
    assert len(_onsets(flags)) == 1, "the merged flag rose once: the scrape"
    assert int(on[:, 0].sum()) == 1 and int(on[:, 2].sum()) == 0
    first_prop = int(torch.nonzero(on[:, 1])[0])
    assert first_prop > int(torch.nonzero(on[:, 0])[0]) and flags[first_prop - 1], "hit during the scrape"
    assert float(imp[first_prop, 1]) > 1.5 and float(imp[first_prop, 0]) == 0.0


# ------------------------------------------------------------------ 3. the contact's acceleration, whole
def _contact_accel(monkeypatch, hold: bool, v: float = 4.0, steps: int = 40):
    """A square duct hit at `v`. Per substep, the acceleration the springs and the accelerometer were given
    minus the tyres' own -- what the contact added -- and per resolution the velocity change it made."""
    import f1sim.dynamics as dyn_mod
    import f1sim.imu as imu_mod
    cfg = _cfg()
    cfg.sim.contact_accel_hold = hold
    sim = Simulator(_floor(), cfg, num_envs=1, device="cpu")
    tyre, added, dv = [], [], []
    real_dyn, real_imu, real_res = dyn_mod.step_dynamics, imu_mod.substep, Simulator._resolve_wall_contact

    def spy_dyn(*a, **kw):
        out = real_dyn(*a, **kw)
        tyre.append(float(out[1][0]))
        return out

    def spy_imu(imu_state, ax, ay, *a, **kw):
        added.append(float(ax[0]) - tyre[-1])
        return real_imu(imu_state, ax, ay, *a, **kw)

    def spy_res(self, state, dt=None):
        out = real_res(self, state, dt=dt)
        dv.append(float(out[0][0, 3] - state[0, 3]))
        return out

    monkeypatch.setattr(dyn_mod, "step_dynamics", spy_dyn)
    monkeypatch.setattr(imu_mod, "substep", spy_imu)
    monkeypatch.setattr(Simulator, "_resolve_wall_contact", spy_res)
    sim.reset(torch.arange(1), torch.tensor([[10.0, 0.9, -math.pi / 2]]), torch.tensor([v]))
    imu = []
    for t in range(steps):
        r = sim.step(torch.tensor([[0.0, v if t < 6 else 0.0]]))
        imu.append(r.imu[0])
    return added, dv, torch.cat(imu), sim


@pytest.mark.parametrize("hold", [True, False])
def test_the_springs_and_the_imu_feel_the_whole_impact(monkeypatch, hold):
    """Integrated over the contact, the acceleration the suspension and the accelerometer are given equals
    the velocity change the contact made. Before, each resolution's change was applied for one 1 ms substep
    of its 5 ms window, and clipped at 16 g: a fifth of every impact, less where it clipped."""
    added, dv, imu, sim = _contact_accel(monkeypatch, hold)
    dt = sim.dt
    total_dv = sum(dv)
    assert total_dv < -3.0, total_dv                          # a 4 m/s hit, stopped by the hose
    felt = sum(added) * dt
    if hold:
        assert abs(felt - total_dv) < 0.01 * abs(total_dv), (felt, total_dv)
        # unclipped where the body feels it -- the first window takes 1 m/s off in 5 ms, 21 g ...
        assert max(abs(a) for a in added) > 16 * 9.80665
    else:
        assert abs(felt) < 0.25 * abs(total_dv), (felt, total_dv)
    # ... and the accelerometer still clips at its range when it is sampled
    rng = float(sim.cfg.imu.accel_range)
    assert float(imu[:, 3:6].abs().max()) <= rng + 1e-3
