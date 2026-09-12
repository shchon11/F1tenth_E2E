"""The T (traffic) family: the metric, the state machine in repeat mode, and the frozen scenarios.

Five things are worth testing here and the rest is decoration:

1. **O is untouched.** `repeat=False` is the machine the overtaking family has always had, branch
   for branch, and adding a second mode to a shared class is exactly how that stops being true.
2. **A pass is counted once, and counted again only when it was made again.** The failure modes are
   symmetric and both silent: a lead that wobbles across the clear margin counted four times, and a
   genuine second pass counted zero.
3. **The continuous side means what it says.** Contention, attacking and pace are the columns that
   carry a cell where nobody passes anybody, so a wrong denominator there is a wrong benchmark.
4. **The metric is not gameable in the obvious direction.** A car that hangs back and never tries
   scores 1.000 on "clean"; the panel has to separate it from one that races.
5. **The frozen suite says what it claims.** v2.1's S/A/O cells are v2's, its traffic maps are held
   out, and a row measured against the wrong opponent is refused rather than rendered.
"""
from __future__ import annotations

import importlib

import pytest

LENGTH = 50.0
HOLD = 40          # 1.0 s at 40 Hz


@pytest.fixture
def ot(bench):
    return importlib.import_module("f1sim.learn.benchmark.overtake")


@pytest.fixture
def mk(ot):
    gm = importlib.import_module("f1sim.learn.benchmark.geom")

    def _mk(repeat=True, vehicle_length=0.58, length=LENGTH):
        overlap, clear = gm.thresholds(vehicle_length)
        return ot.PassDetector(length=length, overlap=overlap, clear=clear, hold_steps=HOLD,
                               repeat=repeat, keep_history=False)
    return _mk


def ramp(start, end, step=0.02):
    g, out = start, []
    for _ in range(int(abs(end - start) / step)):
        g += step if end > start else -step
        out.append(g)
    return out


def drive(det, gaps, **kw):
    for g in gaps:
        det.update(g, **kw)
    return det


# --------------------------------------------------------------------- 1. the O family is untouched

def test_repeat_off_is_the_overtaking_machine_unchanged(ot, mk):
    """Same states, same terminal success, same `repassed` invalidation, same interruption count."""
    det = mk(repeat=False)
    det.start(3.0)
    drive(det, ramp(3.0, -3.0))
    drive(det, [-3.0] * HOLD)
    assert det.succeeded and det.state is ot.State.PASS_HELD
    assert det.passes == 0 and det.repasses == 0            # the new counters stay out of O's way
    assert ot.outcome(det)["success"] is True


def test_repeat_off_still_voids_a_race_on_a_re_pass(ot, mk):
    det = mk(repeat=False)
    det.start(3.0)
    drive(det, ramp(3.0, -0.95))
    drive(det, ramp(-0.95, 2.0))
    assert det.state is ot.State.INVALID and det.reason == "repassed"


def test_armed_no_longer_needs_the_trace_to_exist(ot, mk):
    """`outcome` used to scan `history`; a traffic stint cannot afford to keep one."""
    kept, dropped = mk(repeat=False), mk(repeat=False)
    kept.keep_history = True
    for det in (kept, dropped):
        det.start(3.0)
        drive(det, ramp(3.0, -3.0))
        drive(det, [-3.0] * HOLD)
    assert ot.outcome(kept)["armed"] is ot.outcome(dropped)["armed"] is True
    assert dropped.history == [] and kept.history


def test_a_race_that_never_armed_reports_armed_false(ot, mk):
    det = mk(repeat=False)
    det.start(0.0)                       # starts alongside; never clearly behind
    drive(det, [0.0] * 50)
    assert ot.outcome(det)["armed"] is False


# --------------------------------------------------------------------- 2. counting passes

def test_a_completed_pass_is_counted_and_the_stint_keeps_running(ot, mk):
    det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0))
    drive(det, [-3.0] * HOLD)
    assert det.passes == 1
    assert det.state is ot.State.LED
    assert not det.done, "a counted pass must not end the trial: T runs to the budget"


def test_two_passes_need_the_learner_to_fall_behind_in_between(ot, mk):
    det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0)); drive(det, [-3.0] * HOLD)
    assert det.passes == 1
    drive(det, ramp(-3.0, 3.0))                 # re-passed: a lead taken and given back
    assert det.passes == 1 and det.repasses == 1 and det.state is ot.State.ARMED_BEHIND
    drive(det, ramp(3.0, -3.0)); drive(det, [-3.0] * HOLD)
    assert det.passes == 2


def test_a_lead_that_wobbles_across_the_margin_is_not_four_passes(ot, mk):
    """The failure this exists to stop: dropping to alongside and re-clearing is one pass finishing,
    not a second one being made. Only falling clearly behind re-arms a count."""
    det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0)); drive(det, [-3.0] * HOLD)
    assert det.passes == 1
    for _ in range(3):
        drive(det, ramp(-3.0, -0.2))            # back alongside, never behind
        drive(det, ramp(-0.2, -3.0))
        drive(det, [-3.0] * HOLD)
    assert det.passes == 1, "re-clearing after a dip counted as a new pass"
    assert det.repasses == 0


def test_a_hold_that_is_never_completed_is_not_a_pass(mk):
    """One step short. `ramp` lands exactly on -0.90, just past the 0.89 clear margin, so the
    crossing step is hold step 1 and HOLD-2 more leaves the hold one short of the second."""
    det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.90))
    drive(det, [-0.90] * (HOLD - 2))
    assert det.passes == 0
    det.update(-0.90)
    assert det.passes == 1


def test_contact_stops_the_pair_and_keeps_what_it_already_counted(ot, mk):
    det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0)); drive(det, [-3.0] * HOLD)
    det.update(-3.0, contact=True)
    assert det.state is ot.State.INVALID and det.reason == "contact"
    assert det.passes == 1, "a contact later in the stint does not revoke a completed pass"


def test_an_opponent_respawn_reseeds_the_pair_rather_than_voiding_the_trial(ot, mk):
    """In O a respawn voids the race: the one pass being measured can no longer be attributed. In T
    the learner is still in traffic, and the passes it already made still happened."""
    det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0)); drive(det, [-3.0] * HOLD)
    det.update(4.0, opponent_reset=True)         # the car ahead crashed and came back elsewhere
    assert det.state is not ot.State.INVALID
    assert det.passes == 1 and det.reseeds == 1
    assert det.G == pytest.approx(4.0), "G has to restart at the new gap, not integrate the jump"
    drive(det, ramp(4.0, -3.0)); drive(det, [-3.0] * HOLD)
    assert det.passes == 2


def test_a_respawn_does_not_leave_a_stale_lead_latched(ot, mk):
    """After a re-seed the ego is not 'ahead' of anything; the next car ahead must arm normally."""
    det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0)); drive(det, [-3.0] * HOLD)
    det.update(-2.0, opponent_reset=True)        # respawned BEHIND, which is where the sim puts it
    assert det.repasses == 0
    drive(det, [-2.0] * 20)
    assert det.repasses == 0, "a fresh car behind must not read as having re-passed"


def test_a_wrapping_lap_is_not_a_crossing(mk):
    """The property the whole detector rests on, re-asserted in repeat mode: `G` integrates wrapped
    per-step increments, so a lap boundary cannot look like a pass."""
    det = mk(length=20.0)
    det.start(3.0)
    g = 3.0
    for _ in range(2000):                        # many laps of relative drift at a constant gap
        g = (g + 0.0) % 20.0
        det.update(g)
    assert det.passes == 0


# --------------------------------------------------------------------- 3. the continuous side

def make_trace(ot, contention=12.0, attack=3.0):
    return ot.TrafficTrace(contention_range_m=contention, attack_range_m=attack)


def test_the_windows_are_counted_by_their_own_definitions(ot):
    tr = make_trace(ot)
    dt = 0.025
    for gap in (2.0,) * 40:            # behind, close
        tr.update(dt=dt, ego_progress=0.1, opponent_progress=[0.1], gaps=[gap])
    for gap in (8.0,) * 40:            # behind, in traffic but not attacking
        tr.update(dt=dt, ego_progress=0.1, opponent_progress=[0.1], gaps=[gap])
    for gap in (-2.0,) * 40:           # ahead of it
        tr.update(dt=dt, ego_progress=0.1, opponent_progress=[0.1], gaps=[gap])
    for gap in (20.0,) * 40:           # out of range entirely
        tr.update(dt=dt, ego_progress=0.1, opponent_progress=[0.1], gaps=[gap])
    assert tr.seconds == pytest.approx(4.0)
    assert tr.contention_s == pytest.approx(3.0)      # everything but the last block
    assert tr.following_s == pytest.approx(2.0)       # the two "behind" blocks
    assert tr.attack_s == pytest.approx(1.0)          # only the close one
    assert tr.defending_s == pytest.approx(1.0)


def test_pace_pools_the_opponents_rather_than_picking_one(ot):
    """Matching `gym_env.overtake_gain`: `other_idx` is a roster, not 'the car ahead'. Singling one
    out would make the two-opponent cells measure something the one-opponent cells do not."""
    tr = make_trace(ot)
    for _ in range(100):
        tr.update(dt=0.025, ego_progress=0.10, opponent_progress=[0.05, 0.15], gaps=[1.0, -1.0])
    assert tr.opponent_progress_m == pytest.approx(10.0)
    assert tr.pace_ratio()["value"] == pytest.approx(1.0)


def test_pace_is_withheld_rather_than_invented_when_the_opponent_barely_moved(ot):
    tr = make_trace(ot)
    for _ in range(10):
        tr.update(dt=0.025, ego_progress=0.5, opponent_progress=[0.001], gaps=[1.0])
    r = tr.pace_ratio()
    assert r["value"] is None and "under the" in r["reason"]
    # and the raw progress is still there, so nothing was lost -- only the ratio was withheld
    assert tr.ego_progress_m == pytest.approx(5.0)


def test_closest_approach_is_a_running_minimum_over_every_opponent(ot):
    tr = make_trace(ot)
    tr.update(dt=0.025, ego_progress=0.1, opponent_progress=[0.1, 0.1], gaps=[5.0, -0.4])
    tr.update(dt=0.025, ego_progress=0.1, opponent_progress=[0.1, 0.1], gaps=[3.0, 2.0])
    assert tr.closest_arc_gap_m == pytest.approx(0.4)


def test_a_trial_with_no_opponent_sampled_reports_no_closest_approach(ot):
    tr = make_trace(ot)
    tr.update(dt=0.025, ego_progress=0.1, opponent_progress=[], gaps=[])
    assert tr.closest_arc_gap_m is None


# --------------------------------------------------------------------- 4. the recorder, on arrays

@pytest.fixture
def rn(bench):
    return importlib.import_module("f1sim.learn.benchmark.runner")


def traffic_recorder(rn, n=2, opponents=1, length=LENGTH):
    return rn.CellRecorder(n=n, suite="T", track_length_m=length, vehicle_length=0.58,
                           vehicle_width=0.31, hold_steps=HOLD, frozen=False,
                           n_opponents=opponents, contention_range_m=12.0, attack_range_m=3.0)


def feed(rec, *, steps, gaps, ego=0.10, opp=0.10, collision=None, contact=None, events=None,
         dt=0.025, last_is_budget_end=True):
    """Drive `steps` identical transitions. `gaps[i][j]` may be a scalar or a per-step callable."""
    n = rec.n
    for k in range(steps):
        g = [[gap(k) if callable(gap) else gap for gap in row] for row in gaps]
        rec.update(progress=[ego] * n, speed=[4.0] * n, dt=dt,
                   collision=[bool((collision or {}).get(i) == k) for i in range(n)],
                   truncated=[False] * n,
                   car_contact=[bool((contact or {}).get(i) == k) for i in range(n)],
                   pair_gaps=g, pair_progress=[[opp] * len(row) for row in g],
                   pair_reset=[[False] * len(row) for row in g],
                   opp_event_active=(events[k] if events else None),
                   budget_exhausted=(last_is_budget_end and k == steps - 1))
    return rec


def test_a_traffic_trial_that_survives_the_stint_is_the_success(rn):
    """The metric change this family exists for. Nobody has to pass anybody: coming through the
    stint with no wall and no contact is the outcome, and it is one a 0.70 m half-lane allows."""
    rec = traffic_recorder(rn)
    feed(rec, steps=50, gaps=[[6.0], [6.0]])
    out = rec.finalize()
    assert out["tally"]["successes"] == 2
    assert all(o["success"] and o["reason"] is None for o in out["outcomes"])
    assert sum(out["passes"]) == 0, "and a clean run with no pass is still a clean run"


def test_contact_and_a_wall_are_different_failures(rn):
    rec = traffic_recorder(rn)
    feed(rec, steps=50, gaps=[[2.0], [2.0]], collision={0: 10, 1: 20}, contact={1: 20})
    out = rec.finalize()
    assert [o["reason"] for o in out["outcomes"]] == ["collision", "contact"]
    assert out["wall_collisions"] == 1 and out["car_contacts"] == 1
    assert out["tally"]["successes"] == 0


def test_the_trial_does_not_end_on_a_completed_pass(rn):
    """A pass is a counter, not a finish line. Ending there would measure the first ten seconds of
    the good trials and the whole budget of the bad ones."""
    rec = traffic_recorder(rn)
    cross = ramp(3.0, -3.0) + [-3.0] * (HOLD + 200)
    for k, g in enumerate(cross):
        rec.update(progress=[0.1] * 2, speed=[4.0] * 2, dt=0.025, collision=[False] * 2,
                   truncated=[False] * 2, car_contact=[False] * 2,
                   pair_gaps=[[g], [g]], pair_progress=[[0.05], [0.05]],
                   pair_reset=[[False], [False]], budget_exhausted=(k == len(cross) - 1))
    out = rec.finalize()
    assert out["passes"] == [1, 1]
    assert out["tally"]["successes"] == 2
    # exposure ran the whole budget, not up to the pass
    assert out["elapsed_s"][0] == pytest.approx(len(cross) * 0.025)


def test_the_panel_separates_a_car_that_hangs_back_from_one_that_races(rn):
    """"Clean" alone is gameable: a car that never tries never crashes. The pace column is what
    refuses that, and both are in the same table with their own directions."""
    hangs = traffic_recorder(rn, n=1)
    feed(hangs, steps=400, gaps=[[8.0]], ego=0.05, opp=0.10)
    races = traffic_recorder(rn, n=1)
    feed(races, steps=400, gaps=[[2.0]], ego=0.12, opp=0.10)
    h, r = hangs.finalize(), races.finalize()
    assert h["tally"]["successes"] == r["tally"]["successes"] == 1     # both "clean"
    assert h["pace_ratio"][0]["value"] == pytest.approx(0.5)
    assert r["pace_ratio"][0]["value"] == pytest.approx(1.2)
    assert h["attack_s"][0] == 0.0 and r["attack_s"][0] > 0.0


def test_an_event_outside_the_contention_window_does_not_count_as_landing_in_it(rn):
    """The contract's 'measure it, don't assume': an opponent that brakes half a lap away is a
    schedule entry, not something the learner had to react to."""
    far = traffic_recorder(rn, n=1)
    feed(far, steps=100, gaps=[[20.0]], events=[[True]] * 100)
    near = traffic_recorder(rn, n=1)
    feed(near, steps=100, gaps=[[4.0]], events=[[True]] * 100)
    f, n_ = far.finalize(), near.finalize()
    assert f["event_in_window_trials"] == 0 and f["event_seen_s"][0] == pytest.approx(2.5)
    assert n_["event_in_window_trials"] == 1
    assert n_["event_in_window_s"][0] == pytest.approx(2.5)


def test_two_opponents_are_both_measured(rn):
    rec = traffic_recorder(rn, n=1, opponents=2)
    seq = ramp(3.0, -3.0) + [-3.0] * (HOLD + 5)
    for k, g in enumerate(seq):
        rec.update(progress=[0.1], speed=[4.0], dt=0.025, collision=[False], truncated=[False],
                   car_contact=[False], pair_gaps=[[g, g + 1.5]],
                   pair_progress=[[0.05, 0.05]], pair_reset=[[False, False]],
                   budget_exhausted=(k == len(seq) - 1))
    out = rec.finalize()
    assert out["n_opponents"] == 2
    assert out["passes"] == [2], "both cars were passed and held; both count"


def test_a_trial_that_never_met_traffic_is_reported_as_such(rn):
    """Not a failure, and not silently folded into the conditional rate either."""
    rec = traffic_recorder(rn, n=2)
    feed(rec, steps=40, gaps=[[30.0], [2.0]])
    out = rec.finalize()
    assert out["contended"] == 1
    assert out["clean_conditional"]["value"] == pytest.approx(1.0)
    lonely = traffic_recorder(rn, n=1)
    feed(lonely, steps=40, gaps=[[30.0]])
    assert lonely.finalize()["clean_conditional"]["value"] is None


def test_every_traffic_array_is_per_trial_and_the_report_accepts_it(rn, bench):
    """The producer emits a complete row: the validator length-checks each array against `n`, so an
    array the recorder forgot would be a row the report refuses hours after the GPU time was spent.
    """
    from f1sim.learn.benchmark import report
    rec = traffic_recorder(rn, n=3)
    feed(rec, steps=30, gaps=[[2.0], [2.0], [2.0]])
    out = rec.finalize()
    for key in report.TRAFFIC_TRIAL_ARRAYS:
        assert key in out, f"the recorder does not emit {key}, which the report expects per trial"
        assert len(out[key]) == 3, key


# --------------------------------------------------------------------- 5. the frozen suite

@pytest.fixture
def su(bench):
    return importlib.import_module("f1sim.learn.benchmark.suite")


def packaged(name):
    from importlib import resources
    return str(resources.files("f1sim.learn.benchmark").joinpath(name))


def test_v2_1_adds_the_traffic_family_and_changes_nothing_else(su):
    """The claim the version number makes. Every v2 cell has to survive into v2.1 unchanged, or a
    v2 avoidance row and a v2.1 avoidance row are not the same measurement and the two suites
    cannot be read together."""
    v2, _ = su.load(packaged("suite-v2.example.json"))
    v21, _ = su.load(packaged("suite-v2.1.example.json"))
    assert {c.cell_id(): c for c in v2.cells()} == {
        c.cell_id(): c for c in v21.cells() if c.suite != "T"}
    assert v2.placements == v21.placements, "the obstacle boxes must be v2's, not re-derived ones"
    a, b = v2.as_dict(), v21.as_dict()
    moved = {k for k in a if a[k] != b[k]}
    assert moved == {"version", "traffic", "reused_maps_note", "calibration"}, sorted(moved)


def test_the_v2_1_freeze_hash_verifies_and_differs_from_v2(su):
    _, h2 = su.load(packaged("suite-v2.example.json"))
    _, h21 = su.load(packaged("suite-v2.1.example.json"))
    assert h2 != h21, "a different scenario set must be a different freeze"
    assert h2 == "89805514350d932a36cbfe28eed7ca379ec92d71806ec5202736aa61805394f9", (
        "v2's published freeze moved; every measurement taken against it is now unreproducible")


def test_an_empty_traffic_field_leaves_an_existing_freeze_alone(su):
    """Why `traffic` is omitted from the hash when empty. It declares no scenario and adds no cell,
    so hashing it would have re-hashed v1 and v2 -- published freezes describing measurements that
    have already been taken -- for a change that is not one to them."""
    s = su.Suite()
    before = s.freeze_hash()
    s.traffic = {}
    assert s.freeze_hash() == before
    s.traffic = {"scenarios": [{"id": "x", "race_size": 2, "opp_speed_range": [0.6, 0.8]}]}
    assert s.freeze_hash() != before, "a declared scenario set MUST be hashed"


def test_v1_and_v2_freeze_to_what_they_always_did(su):
    """The two suites this change must not touch, checked against their files rather than against
    a constant written here."""
    for name in ("suite-v1.example.json", "suite-v2.example.json"):
        s, h = su.load(packaged(name))        # `load` recomputes and refuses a mismatch
        assert s.traffic == {} and s.expected_trials().get("T") is None


def test_a_traffic_cell_id_carries_its_scenario(su):
    """Four T cells share a map, a friction and a seed and differ only in what the opponent is
    doing -- which is the point of them. Without the variant in the identity, every duplicate check
    in the report reads them as one cell measured four times."""
    s = su.of("v2.1")
    ids = [c.cell_id() for c in s.cells()]
    assert len(ids) == len(set(ids))
    slow = next(c for c in s.cells() if c.suite == "T" and c.variant == "slow")
    assert slow.cell_id().startswith("T:slow:")
    # and the families that have one scenario keep the id they always had
    assert next(c for c in s.cells() if c.suite == "O").cell_id().startswith("O:gen:control:9100:")


def test_each_scenario_declares_the_axis_it_varies(su):
    s = su.of("v2.1")
    by = {sc["id"]: sc for sc in s.traffic_scenarios()}
    assert set(by) == {"slow", "pace", "event", "pair"}
    assert by["slow"]["opp_speed_range"] == [0.5, 0.7]
    assert by["pace"]["opp_speed_range"] == [0.8, 0.95]
    assert sorted(by["event"]["opp_events"]) == ["brake", "shift", "stop"]
    assert by["event"]["opp_event_rate"] > 0
    assert by["pair"]["race_size"] == 3
    # the two behaviour cells hold the speed axis at v2's own range, so each differs from a
    # comparable baseline in exactly one thing
    assert by["event"]["opp_speed_range"] == by["pair"]["opp_speed_range"] == list(su.OPP_SPEED_RANGE)
    assert by["slow"]["race_size"] == by["pace"]["race_size"] == by["event"]["race_size"] == 2
    assert not by["slow"]["opp_events"] and not by["pace"]["opp_events"]
    assert not by["pair"]["opp_events"]


def test_the_adapter_gets_the_scenario_and_not_the_suite_default(su):
    s = su.of("v2.1")
    cells = {c.variant: c for c in s.cells() if c.suite == "T"}
    assert s.adapter_suite(cells["slow"])["opp_speed_range"] == [0.5, 0.7]
    assert s.adapter_suite(cells["pair"])["race_size"] == 3
    assert sorted(s.adapter_suite(cells["event"])["opp_events"]) == ["brake", "shift", "stop"]
    # an O cell, and no cell at all, keep the suite-wide values
    o = next(c for c in s.cells() if c.suite == "O")
    assert s.adapter_suite(o)["opp_speed_range"] == list(s.opp_speed_range)
    assert s.adapter_suite(o)["opp_events"] == []
    assert s.adapter_suite()["race_size"] == s.race_size


def test_an_unknown_scenario_is_refused_rather_than_defaulted(su):
    with pytest.raises(KeyError, match="declares no traffic scenario"):
        su.of("v2.1").traffic_scenario("weave")


def test_the_grid_is_the_contract_s_maps_frictions_and_seeds(su):
    s = su.of("v2.1")
    t = [c for c in s.cells() if c.suite == "T"]
    assert {c.map_id for c in t} == {"real:map16x07", "real:map12x16", "gen:control:9100",
                                     "real:korea_2025_iccas", "gen:competition:0"}
    assert {c.mu for c in t} == {su.MU_LOW, su.MU_MID}
    assert {c.seed for c in t} == set(su.SEEDS)
    assert len(t) == 4 * 5 * 2 * 2
    assert s.expected_trials()["T"] == len(t) * s.envs


def test_declared_race_size_prefers_the_cell_and_falls_back_to_the_suite(su):
    s = su.of("v2.1")
    pair = next(c for c in s.cells() if c.variant == "pair")
    assert su.declared_race_size(pair, s) == 3
    solo = next(c for c in s.cells() if c.suite == "S")
    assert su.declared_race_size(solo, s) == 1
    # a Cell built by hand, as tests and tools do, keeps the old derivation
    assert su.declared_race_size(su.Cell("O", "m", 0.9, 1, 8), s) == s.race_size
    assert su.declared_race_size(su.Cell("S", "m", 0.9, 1, 8), s) == 1


# --------------------------------------------------------------------- 6. the report

@pytest.fixture
def rp(bench):
    return importlib.import_module("f1sim.learn.benchmark.report")


def t_row(su, variant="slow", n=8, passes=None, clean=None, ego=60.0, opp=50.0, events=0,
          contact=0, wall=0):
    """A synthetic T row shaped exactly as the runner writes one. No simulator, no checkpoint."""
    from f1sim.learn.benchmark import fingerprint as _fp
    s = su.of("v2.1")
    cell = next(c for c in s.cells() if c.suite == "T" and c.variant == variant)
    sc = s.traffic_scenario(variant)
    clean = [True] * n if clean is None else clean
    passes = [0] * n if passes is None else passes
    reasons = ["contact"] * contact + ["collision"] * wall
    outcomes = [{"success": bool(c), "reason": None} if c else
                {"success": False, "reason": reasons.pop(0)} for c in clean]
    fails = {}
    for o in outcomes:
        if not o["success"]:
            fails[o["reason"]] = fails.get(o["reason"], 0) + 1
    res = {
        "n": n, "outcomes": outcomes, "lap_time_s": [None] * n,
        "progress_m": [ego / n] * n, "distance_m": [ego / n] * n,
        "route_progress_fraction": [1.0] * n,
        "passes": list(passes), "leads_lost": [0] * n, "opponent_respawns": [0] * n,
        "contention_s": [10.0] * n, "following_s": [6.0] * n, "attack_s": [3.0] * n,
        "defending_s": [1.0] * n, "opponent_progress_m": [opp / n] * n,
        "pace_ratio": [{"value": ego / opp, "reason": None}] * n,
        "closest_arc_gap_m": [0.9] * n,
        "event_in_window_s": [1.0 if events else 0.0] * n, "event_seen_s": [1.5] * n,
        "contended": n, "car_contacts": contact, "wall_collisions": wall,
        "event_in_window_trials": events,
        "n_opponents": int(sc["race_size"]) - 1,
        "contention_range_m": 12.0, "attack_range_m": 3.0,
        "effective": {"true_mu": cell.mu, "plant_mu": cell.mu, "envs": n * int(sc["race_size"]),
                      "race_size": int(sc["race_size"]),
                      "opp_speed_range": list(sc["opp_speed_range"]),
                      "opp_events": list(sc["opp_events"]),
                      "opp_event_rate": float(sc["opp_event_rate"])},
        "start_fingerprint": {
            "schema_version": _fp.SCHEMA_VERSION,
            "physical_sha256": "a" * 64, "physical_tensors": 20,
            "actor_input_sha256": "b" * 64, "actor_input_tensors": 6,
            "calibration_sha256": "c" * 64, "calibration_tensors": 8,
            "obs_spec_sha256": "d" * 64, "sim_t": 0.3, "sim_imu_phase": 1,
            "n_envs_total": n * int(sc["race_size"]), "race_size": int(sc["race_size"])},
        "tally": {"successes": sum(1 for o in outcomes if o["success"]),
                  "denominator": n, "failures": fails},
    }
    row = {"system_id": "sys", "runtime": "legacy", "controller_arm": "legacy",
           "suite": "T", "variant": variant, "map_id": cell.map_id, "mu": cell.mu,
           "seed": cell.seed, "n_envs": n, "cell_id": cell.cell_id(),
           "speed_cap": s.speed_cap, "budget_laps": s.budget_laps, "sensor_noise": s.sensor_noise,
           "backend": s.backend, "result": res}
    return s, cell, row


def test_a_valid_traffic_row_is_accepted(rp, su):
    s, cell, row = t_row(su)
    rp.validate_cell(row, cell, suite=s)


def test_a_row_measured_against_the_wrong_opponent_is_refused(rp, su):
    """The substitution this family makes possible and no other check would catch: same map, same
    friction, same seed, right cell name -- and a different car to race."""
    s, cell, row = t_row(su, "slow")
    row["result"]["effective"]["opp_speed_range"] = [0.8, 0.95]      # that is the `pace` scenario
    with pytest.raises(rp.ReportError, match="opp_speed_range"):
        rp.validate_cell(row, cell, suite=s)


def test_a_row_whose_events_were_off_cannot_stand_in_for_the_event_cell(rp, su):
    s, cell, row = t_row(su, "event", events=4)
    row["result"]["effective"]["opp_events"] = []
    with pytest.raises(rp.ReportError, match="opp_events"):
        rp.validate_cell(row, cell, suite=s)


def test_a_row_with_the_wrong_number_of_cars_is_refused(rp, su):
    s, cell, row = t_row(su, "pair")
    row["result"]["effective"]["race_size"] = 2
    with pytest.raises(rp.ReportError, match="race_size"):
        rp.validate_cell(row, cell, suite=s)


def test_a_traffic_row_without_its_variant_is_refused(rp, su):
    s, cell, row = t_row(su)
    row["variant"] = ""
    with pytest.raises(rp.ReportError, match="variant"):
        rp.validate_cell(row, cell, suite=s)


def test_the_cell_id_a_row_claims_must_match_the_scenario_it_names(rp, su):
    _, _, row = t_row(su, "slow")
    row["cell_id"] = row["cell_id"].replace("T:slow:", "T:pace:")
    with pytest.raises(rp.ReportError, match="does not match"):
        rp.validate_row_identity(row, {})


def test_two_scenarios_on_one_map_are_two_cells_and_not_a_duplicate(rp, su):
    a = t_row(su, "slow")[2]
    b = t_row(su, "pace")[2]
    assert rp.cell_key(a) != rp.cell_key(b)
    assert rp.row_cell_id(a) != rp.row_cell_id(b)


def test_the_traffic_table_reports_each_scenario_and_a_pooled_row(rp, su):
    cells = []
    for v, pss in (("slow", [1, 1, 0, 1, 1, 1, 0, 1]), ("pace", [0] * 8)):
        s, cell, row = t_row(su, v, passes=pss)
        rp.validate_cell(row, cell, suite=s)
        cells.append(row)
    out = rp.aggregate(cells)
    rows = {r["runtime"]: r for r in out["traffic"]}
    assert set(rows) == {"legacy · slow", "legacy · pace", "legacy · all scenarios"}
    assert rows["legacy · slow"]["values"]["passes/race ↑"]["value"] == pytest.approx(0.75)
    assert rows["legacy · pace"]["values"]["passes/race ↑"]["value"] == 0.0
    assert rows["legacy · all scenarios"]["values"]["passes/race ↑"]["value"] == pytest.approx(0.375)
    # pooled by summing arcs, not by averaging ratios
    assert rows["legacy · all scenarios"]["values"]["pace vs opponent ↑"]["value"] == \
        pytest.approx(60.0 / 50.0)


def test_the_event_column_is_not_diluted_by_cells_that_schedule_no_events(rp, su):
    """A `slow` cell has no events; pooling its zero into the event column would report the feature
    firing less often than it does in the cells that have it."""
    cells = []
    for v, ev in (("slow", 0), ("event", 4)):
        s, cell, row = t_row(su, v, events=ev)
        rp.validate_cell(row, cell, suite=s)
        cells.append(row)
    rows = {r["runtime"]: r for r in rp.aggregate(cells)["traffic"]}
    assert rows["legacy · slow"]["values"]["event-in-window trials"]["value"] is None
    assert rows["legacy · event"]["values"]["event-in-window trials"]["value"] == 4
    assert rows["legacy · all scenarios"]["values"]["event-in-window trials"]["value"] == 4


def test_a_traffic_row_carrying_a_lap_time_is_refused(rp, su):
    """T has no laps. A lap time here would be a fabricated number for a lap nobody ran."""
    s, cell, row = t_row(su)
    row["result"]["lap_time_s"] = [12.0] + [None] * 7
    with pytest.raises(rp.ReportError, match="no laps"):
        rp.validate_cell(row, cell, suite=s)


def test_a_negative_traffic_measurement_is_refused(rp, su):
    s, cell, row = t_row(su)
    row["result"]["attack_s"] = [-1.0] + [0.0] * 7
    with pytest.raises(rp.ReportError, match="negative"):
        rp.validate_cell(row, cell, suite=s)


def test_a_short_traffic_array_is_refused(rp, su):
    s, cell, row = t_row(su)
    row["result"]["passes"] = [0, 0]
    with pytest.raises(rp.ReportError, match="passes"):
        rp.validate_cell(row, cell, suite=s)


def test_the_rendered_table_shows_the_traffic_category(rp, su):
    s, cell, row = t_row(su, "slow", passes=[1] + [0] * 7)
    rp.validate_cell(row, cell, suite=s)
    summary = {"suite_freeze_sha256": "f" * 64, "n_systems": 1, "n_rows": 1}
    summary.update(rp.aggregate([row]))
    md = rp.render_markdown(summary, suite=s)
    assert "## Traffic" in md
    assert "pace vs opponent" in md and "clean ↑" in md
