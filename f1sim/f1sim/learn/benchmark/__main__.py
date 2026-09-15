"""Benchmark CLI.

    plan      matrix, expected trials, cost projection, roster verification   (CPU, no policy)
    geometry  obstacle proofs + non-candidate feasibility smoke, then freeze  (CPU, no candidate)
              `--version v2` freezes the held-out scenario set instead of v1
    run       one system against the frozen suite                             (GPU lease required)
    report    validate raw results and render the leaderboard                 (CPU)

`plan` and `geometry` never load an evaluated checkpoint, so nothing they measure can leak a
candidate outcome into the scenario choice. `run` refuses without a frozen suite.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

from . import roster as roster_mod
from . import suite as suite_mod
from . import report as report_mod

#: Measured on the SGR preflight: 94 cells in 52:05 on a map mix whose mean raceline is 156.98 m.
#: Length scaling is linear in budget_steps; early terminations make it optimistic, the race
#: multiplier is inferred. Rough in both directions -- the geometry smoke replaces it.
REF_SECONDS_PER_CELL = 33.2
REF_MEAN_LENGTH_M = 156.98
RACE_MULTIPLIER = 2.0


def _map_lengths(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path) as fh:
            return {k: v["raceline_length_m"] for k, v in json.load(fh)["maps"].items()}
    return {}


def cmd_plan(a) -> int:
    s, frozen = (suite_mod.load(a.suite) if os.path.exists(a.suite) else (suite_mod.Suite(), None))
    cells = s.cells()
    exp = s.expected_trials()
    lengths = _map_lengths(a.map_lengths)

    print(f"suite {s.version}  freeze {frozen[:16] if frozen else '(not frozen)'}")
    print(f"cells {len(cells)}   trials/system {sum(exp.values())}   " +
          "  ".join(f"{k}={v}" for k, v in exp.items()))
    if s.traffic_scenarios():
        for sc in s.traffic_scenarios():
            n = sum(1 for c in cells if c.suite == "T" and c.variant == sc["id"])
            ev = (f"events {','.join(sc['opp_events'])} @ {sc['opp_event_rate']:g}/10 s"
                  if sc.get("opp_events") else "no events")
            print(f"  T:{str(sc['id']):6s} {n:3d} cells  opponents {int(sc['race_size']) - 1}  "
                  f"speed {tuple(sc['opp_speed_range'])}  {ev}")

    total = 0.0
    for c in cells:
        L = lengths.get(c.map_id)
        scale = (L / REF_MEAN_LENGTH_M) if L else 1.0
        # A race costs roughly its car count: the simulator steps every car, so a 3-car T cell is
        # half again a 2-car one. The old constant assumed every race was two cars, which the
        # traffic family's `pair` scenario is not.
        cars = suite_mod.declared_race_size(c, s)
        total += REF_SECONDS_PER_CELL * scale * (RACE_MULTIPLIER * cars / 2.0 if cars > 1 else 1.0)
    if lengths:
        print(f"projected {total/60:.1f} min/system   (rough: measured rate x length scaling, "
              f"race multiplier inferred)")
    else:
        print("projected cost: unavailable (pass --map-lengths for the pinned geometry file)")

    if a.roster:
        entries = roster_mod.load(a.roster)
        info = roster_mod.verify_all(entries)
        print(f"roster {info['n_systems']} systems, {info['n_unique_weights']} unique weight sets")
        for e in entries:
            est = f" est={e.estimator_sha256[:12]}" if e.estimator_sha256 else ""
            print(f"  {e.system_id:34s} {e.checkpoint_sha256[:12]} arm={e.controller_arm}{est}")
        print(f"total trials {sum(exp.values()) * info['n_systems']}   "
              f"{total*info['n_systems']/3600:.1f} GPU-h" if lengths else "")
    return 0


def cmd_geometry(a) -> int:
    """Prove every obstacle scenario, measure real cell time with no candidate, then freeze."""
    from . import obstacle
    try:
        from f1sim.params import VehicleParams
        from f1sim.learn import common
    except Exception as exc:                      # pragma: no cover - environment dependent
        print(f"f1sim unavailable: {exc}", file=sys.stderr)
        return 2
    vp = VehicleParams()
    # The scenario set is chosen by version, not by which file it is about to be written to: a v1
    # definition frozen into a file named v2 would carry a valid hash and the wrong maps.
    s = suite_mod.of(a.version)
    print(f"suite {s.version}  solo {len(s.solo_maps)}  obstacle {len(s.obstacle_maps)}  "
          f"race {len(s.race_maps)}  traffic {len(s.traffic.get('maps', ()))}x"
          f"{len(s.traffic_scenarios())}  cells {len(s.cells())}")

    inherited = None
    if a.inherit_placements:
        inherited = _inherit_placements(s, a.inherit_placements)

    ok = True
    for map_id in s.obstacle_maps:
        if inherited is not None:
            rec = inherited[map_id]
            s.placements[map_id] = rec
            pl = rec["placement"]
            print(f"  {map_id:32s} INHERITED at s={pl['s_obs_m']:.1f} m  from "
                  f"{os.path.basename(a.inherit_placements)}")
            continue
        try:
            trs, _ = common.load_tracks([map_id], racelines=False, drop_infeasible=False)
            track = trs[0]
            new, place, search = obstacle.find_feasible_s(
                track, vehicle_length=vp.length, vehicle_width=vp.width, start_m=float(a.s_obs))
            proofs = obstacle.prove(new, track, place,
                                    vehicle_length=vp.length, vehicle_width=vp.width)
            s.placements[map_id] = {"placement": place.as_dict(), "proofs": proofs,
                                    "search": search,
                                    "s_start_m": place.s_obs_m + s.s_start_offset_m}
            print(f"  {map_id:32s} PROVEN at s={place.s_obs_m:.1f} m  corridor "
                  f"{proofs['corridor']['widths_m']} >= {proofs['corridor']['required_m']:.2f} m "
                  f"({len(search['rejected'])} positions rejected first)")
        except obstacle.GeometryError as exc:
            ok = False
            print(f"  {map_id:32s} REFUSED {exc}", file=sys.stderr)
    if not ok:
        print("geometry failed: fix or drop the scenario before freezing, never select on scores",
              file=sys.stderr)
        return 1

    # Measured feasibility: drive the raceline teacher through one cell per obstacle map and time
    # it. No evaluated checkpoint is loaded, so nothing here can leak an outcome into the freeze,
    # and the cost projection stops being an extrapolation from another experiment's map mix.
    timings = {}
    if a.measure:
        import time
        from .integration_gate import _build
        from .runner import run_cell
        for map_id in s.obstacle_maps:
            env = _build(map_id, envs=s.envs, race_size=1, seed=s.seeds[0], force_teacher=True)

            def drive(_obs, _env=env):
                return _env.teacher.plan_action(_env.sim.state, _env.sim.P, _env.sim.tid,
                                                _env.ecfg.v_max_policy, _env.tracker.spec)
            t0 = time.perf_counter()
            res = run_cell(env, drive, suite="S", n_steps=a.measure_steps, frozen=False)
            dt = time.perf_counter() - t0
            timings[map_id] = {"seconds_per_cell": round(dt, 2), "envs": s.envs,
                               "steps": a.measure_steps,
                               "completed": int(sum(res.get("completed", []))),
                               "device": "cpu"}
            print(f"  {map_id:32s} feasibility {dt:.1f}s/cell  "
                  f"completed {timings[map_id]['completed']}/{res['n']}")
    s.calibration = {"note": "non-candidate feasibility smoke; no evaluated checkpoint was loaded",
                     "measured": bool(timings), "timings": timings}
    if a.freeze:
        h = s.save(a.suite)
        print(f"frozen -> {a.suite}  {h[:16]}")
    else:
        print("dry run; pass --freeze to write the suite file")
    return 0


def _inherit_placements(s, path: str) -> dict:
    """Take an existing frozen suite's obstacle placements, verbatim, instead of re-deriving them.

    A successor suite that only ADDS a family has to carry the predecessor's avoidance cells
    unchanged, and re-running the search would not do that: the placement depends on `--s-obs`, and
    v2 was frozen at 10 while the flag defaults to 20. Re-deriving would produce a different box on
    every map and a v2.1 A row would silently not be a v2 A row. Loading them is also a *proof*
    rather than a hope -- `suite.load` refuses a file edited after its freeze, so what is copied is
    what was scored.

    Refuses a source that does not cover exactly the same obstacle maps: a partial inheritance would
    leave some cells inherited and some re-derived, which is the worst of both.
    """
    base, base_hash = suite_mod.load(path)
    if not base.placements:
        raise SystemExit(f"{path} carries no frozen placements to inherit")
    missing = [m for m in s.obstacle_maps if m not in base.placements]
    extra = [m for m in base.placements if m not in s.obstacle_maps]
    if missing or extra:
        raise SystemExit(
            f"{path} (suite {base.version}, freeze {base_hash[:12]}) places obstacles on "
            f"{sorted(base.placements)}, this suite declares {sorted(s.obstacle_maps)}. Inheriting "
            f"part of a set would leave the rest re-derived under different defaults; refusing.")
    print(f"  placements inherited from suite {base.version} freeze {base_hash[:16]}")
    return {m: base.placements[m] for m in s.obstacle_maps}


def cmd_gate(a) -> int:
    """Real-env opponent independence. Must pass before any benchmark GPU work."""
    from .integration_gate import run_gate
    arms = tuple(a.arms.split(","))
    routed = run_gate(a.map, steps=a.steps, routed=True, arms=arms, estimator_path=a.estimator)
    control = run_gate(a.map, steps=a.steps, routed=False, arms=arms, estimator_path=a.estimator)
    print(f"cases: {routed['cases']}  cold+warm covered={routed['covers_cold_and_warm']}")
    print(f"routed   identical={routed['opponent_commands_identical']} "
          f"max|delta|={routed['max_abs_delta']:.3e}")
    print(f"unrouted identical={control['opponent_commands_identical']} "
          f"max|delta|={control['max_abs_delta']:.3e}   (must differ, else vacuous)")
    if not routed["opponent_commands_identical"]:
        print("GATE FAILED: opponent moved with the candidate", file=sys.stderr)
        return 1
    if control["opponent_commands_identical"]:
        print("GATE VACUOUS: control did not diverge", file=sys.stderr)
        return 2
    if not routed["covers_cold_and_warm"]:
        print(f"GATE INCOMPLETE: cold+warm not covered for {routed['uncovered_cases']}",
              file=sys.stderr)
        return 3
    print("GATE PASSED")
    return 0


def _synthetic_entry(tmpdir: str, arm: str = "legacy"):
    """A valid checkpoint carrying only an ObsSpec. No candidate weights are involved.

    Feasibility must run through the same adapter, config, reset and routing as scoring, and the
    adapter needs a checkpoint to read a spec from. This supplies one whose parameters are random
    and unused -- the driver is the scripted expert, never this model.
    """
    import dataclasses
    import torch
    from f1sim.learn.model import ActorCritic, save_checkpoint
    from f1sim.learn.obs import ObsSpec
    from f1sim.params import Config

    spec = ObsSpec(n_beams=Config().lidar.n_beams, scan_stack=6, scan_stride=1, hist_len=20,
                   hist_stride=2, action_history=2, act_dim=8)
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=17, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(4242)
        model = ActorCritic(**meta)
    extra = {"spec": dataclasses.asdict(spec), "phase": "ppo", "run": "feasibility-synthetic",
             "update": 1, "updates": 1, "steps": 1, "total_steps": 1, "cap": 9.0}
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, "feasibility_synthetic.pt")
    save_checkpoint(path, model, extra)
    return {"path": path, "arm": arm}, {"spec": dataclasses.asdict(spec), "cap": 9.0}


def cmd_feasibility(a) -> int:
    """Prove the declared scenarios are drivable, on the scoring path.

    Same adapter, same declared friction, same frozen placement and spawn, same routing, same
    seeded reset. The driver is a scripted expert built from the scenario's geometry -- no candidate
    weights are loaded and nothing here is a benchmark score.

    A teacher hitting the obstacle proves it blocks the line. Only a driver that gets through proves
    the scenario is feasible, and both are reported: failures are evidence, not noise.
    """
    import json as _json
    import tempfile
    from .experts import AvoidanceExpert, PassExpert, TrafficExpert

    s, frozen = suite_mod.load(a.suite)
    if not s.placements:
        print("no frozen placements: run `geometry --freeze` first", file=sys.stderr)
        return 1
    entry, extra = _synthetic_entry(a.tmpdir or tempfile.mkdtemp(prefix="bench-feas-"))

    rows, ok = [], True
    mus = [("low", s.solo_mus[0]), ("mid", s.solo_mus[1])]
    plan = [("A", m, "", mu_name, mu) for m in s.obstacle_maps for mu_name, mu in mus]
    plan += [("O", m, "", mu_name, mu) for m in s.race_maps for mu_name, mu in mus]
    plan += [("T", m, str(sc["id"]), mu_name, mu)
             for sc in s.traffic_scenarios()
             for m in s.traffic.get("maps", ())
             for mu_name, mu in mus]
    if a.only:
        want = set(a.only.split(","))
        plan = [row for row in plan if row[0] in want or f"{row[0]}:{row[2]}" in want]

    for kind, map_id, variant, mu_name, mu in plan:
        race_size = (int(s.traffic_scenario(variant)["race_size"]) if kind == "T"
                     else (s.race_size if kind == "O" else 1))
        cell = suite_mod.Cell(kind, map_id, mu, s.seeds[0], a.envs, variant=variant,
                              race_size=race_size)
        prepared, s_obs, router = _prepare(entry, extra, cell, s, a.device)
        try:
            env = prepared.env
            if kind == "A":
                rec = s.placements[map_id]["placement"]
                side = -1 if rec.get("corridor_centre_offset_m", 0.0) < 0 else 1
                driver = AvoidanceExpert(env, s_obs_m=s_obs, free_side=side,
                                         corridor_offset_m=abs(rec.get(
                                             "corridor_centre_offset_m", 0.45)))
            elif kind == "T":
                driver = TrafficExpert(env)
            else:
                driver = PassExpert(env)
            from .runner import run_cell
            from f1sim.params import VehicleParams
            vp = VehicleParams()
            res = run_cell(env, driver, suite=kind, n_steps=int(env.ecfg.max_steps),
                           s_obs_m=s_obs,
                           hold_steps=int(round(s.hold_seconds / float(env.sim.control_dt))),
                           vehicle_length=vp.length, vehicle_width=vp.width, frozen=False,
                           controller=prepared, seed=cell.seed,
                    # the adapter reports the effective layout under "spec"
                    obs_spec=(prepared.protocol or {}).get("spec"),
                    **_traffic_windows(s))
        finally:
            if router is not None:
                router.uninstall()
            prepared.close()

        t = res["tally"]
        if kind == "T":
            feasible, detail = _traffic_feasible(res, prepared.protocol)
        else:
            feasible, detail = t["successes"] > 0, ""
        ok &= feasible
        rows.append({"kind": kind, "map": map_id, "variant": variant,
                     "mu_label": mu_name, "mu": mu,
                     "feasible": feasible, "successes": t["successes"],
                     "denominator": t["denominator"], "reasons": t["failures"],
                     "effective": prepared.protocol, "result": res})
        mark = "FEASIBLE" if feasible else "NOT SHOWN"
        label = f"{kind}:{variant}" if variant else kind
        print(f"  {label:8s} {map_id:24s} mu={mu_name:3s} {t['successes']}/{t['denominator']}  "
              f"{mark}  {t['failures']} {detail}")

    if a.out:
        with open(a.out, "w") as fh:
            for r in rows:
                fh.write(_json.dumps(r) + "\n")
        print(f"wrote {a.out}")
    print("Scripted expert on synthetic metadata; no candidate weights, not a benchmark score.")
    return 0 if ok else 2


def cmd_run(a) -> int:
    """Measure one pinned system against the frozen suite.

    Fully implemented. Execution is still gated: scoring a roster checkpoint requires a frozen
    suite, a verified pin whose recorded arm matches, proven placements, a passing independence
    gate, and an explicit `--lease`, which records that exclusive GPU access was granted.
    """
    import json as _json
    import time
    s, frozen = suite_mod.load(a.suite)           # refuses an edited suite
    entries = {e.system_id: e for e in roster_mod.load(a.roster)}
    if a.system not in entries:
        print(f"{a.system} is not in the roster", file=sys.stderr)
        return 2
    entry = entries[a.system]
    # Suite-level checks first: they are cheap, they are about the protocol rather than the system,
    # and a missing freeze should be reported as such rather than behind a checkpoint load error.
    if not s.placements:
        print("suite has no proven obstacle placements: run `geometry --freeze` first",
              file=sys.stderr)
        return 1
    entry.verify()
    roster_mod.check_arm_matches_record(entry)    # declared arm vs the recorded one
    if not a.lease:
        print(f"system {a.system} verified against suite {frozen[:16]}; "
              f"{len(s.cells())} cells ready.", file=sys.stderr)
        print("refusing to score without --lease: scoring needs exclusive GPU access and a "
              "frozen roster.", file=sys.stderr)
        return 3

    from .integration_gate import run_gate
    # The estimated arm has no default estimator, and 20 steps cannot fill a 40-frame history, so
    # the previous call could neither run nor claim warm coverage. Both are now explicit.
    if not a.estimator:
        print("run needs --estimator: the gate crosses the estimated arm and it has no default",
              file=sys.stderr)
        return 2
    g = run_gate(steps=a.gate_steps, routed=True, estimator_path=a.estimator)
    c = run_gate(steps=a.gate_steps, routed=False, estimator_path=a.estimator)
    if not g["opponent_commands_identical"] or c["opponent_commands_identical"]:
        print(f"independence gate failed or vacuous: routed={g['max_abs_delta']} "
              f"control={c['max_abs_delta']}", file=sys.stderr)
        return 4
    if not g["covers_cold_and_warm"]:
        print(f"independence gate did not cover cold+warm for {g['uncovered_cases']}; a step count "
              f"is not evidence of either regime", file=sys.stderr)
        return 4

    from . import model_adapter as ma
    from f1sim.params import VehicleParams
    ed = _entry_dict(entry)
    model, extra = ma.load_actor(ed, a.device)
    policy = (ma.external_policy(model, extra["spec"], VehicleParams().s_max)
              if ma.is_external(ed) else ma.policy_for(model))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.system.replace('/', '_')}.cells.jsonl")
    ident0 = suite_mod.protocol_identity(s, entry)
    by_id = {c.cell_id(): c for c in s.cells()}
    done = _resume(path, ident0, by_id, entry, suite_obj=s)
    # `--only` selects which of the frozen suite's cells are executed and nothing else: the rows
    # written are the rows an unfiltered run would write, cell for cell, and `protocol_identity` is
    # untouched, so a filtered file resumes into a full one and vice versa. What it does NOT do is
    # make a partial file a leaderboard row -- `report` still requires the suite's cells to be
    # present, and a filtered run is a family table, not a score.
    #
    # It exists because a system may be *unscoreable* on part of the suite rather than merely
    # expensive there: a policy trained with privileged opponent tokens (`f1sim.opp_token`) has no
    # observation at all in a solo cell, and `model_adapter.build_cell` refuses to invent one.
    want = set((a.only or "").split(",")) if a.only else None
    cells = [c for c in s.cells()
             if want is None or c.suite in want or f"{c.suite}:{c.variant}" in want]
    if want is not None:
        print(f"--only {a.only}: {len(cells)} of {len(s.cells())} cells; this file is a family "
              f"table and not a leaderboard row", file=sys.stderr)
        if not cells:
            print(f"--only {a.only} selected no cell of this suite", file=sys.stderr)
            return 2
    written = 0
    with open(path, "a") as fh:
        for cell in cells:
            key = cell.cell_id()
            if key in done:
                continue
            t0 = time.perf_counter()
            prepared, s_obs, router = _prepare(entry, extra, cell, s, a.device)
            try:
                res = _run_one(prepared, policy, cell, s_obs, s)
            finally:
                if router is not None:
                    router.uninstall()          # restore before the adapter tears its arm down
                prepared.close()
            row = dict(suite_mod.protocol_identity(s, entry, effective=prepared.protocol))
            # `n_envs` from the DECLARED cell, never from `res["n"]`: the reporter checks the
            # recorded count against the declared one, and sourcing it from the result would make
            # that check compare the result against itself.
            row.update({"suite": cell.suite, "variant": cell.variant, "map_id": cell.map_id,
                        "mu": cell.mu, "seed": cell.seed, "n_envs": cell.envs,
                        "runtime": entry.controller_arm, "cell_id": key,
                        "wall_seconds": round(time.perf_counter() - t0, 2), "result": res})
            fh.write(_json.dumps(row) + "\n")
            fh.flush()
            written += 1
            print(f"  {key:44s} {res['tally']['successes']}/{res['tally']['denominator']}")
    print(f"wrote {written} cells to {path}")
    return 0


def _resume(path: str, ident: dict, cells_by_id: dict | None = None,
            roster_entry=None, suite_obj=None) -> set:
    """Cells already recorded UNDER THIS EXACT PROTOCOL.

    Resuming on `cell_id` alone would skip a cell recorded under a different suite freeze, roster
    entry, arm or source digest -- the run would look complete while mixing two protocols in one
    file, which is the failure the report exists to catch and would here be created by the runner
    itself. A row that does not match is a hard stop, not a silent re-run.
    """
    import json as _json
    if not os.path.exists(path):
        return set()
    keep, stale, order = set(), [], []
    for ln in open(path):
        if not ln.strip():
            continue
        try:
            r = _json.loads(ln)
        except ValueError:
            raise SystemExit(f"{path}: corrupt line; move the file aside rather than resuming")
        same = all(r.get(k) == ident.get(k) for k in
                   ("suite_freeze_sha256", "checkpoint_sha256", "controller_arm",
                    "estimator_sha256", "identity_sha256"))
        if not same:
            stale.append(r.get("cell_id"))
            continue
        # The same gate the report applies. Without it a row the report would later refuse could be
        # marked done here, and the run would look complete while carrying a row that cannot render.
        cell = (cells_by_id or {}).get(r.get("cell_id"))
        if cell is None:
            raise SystemExit(f"{path}: row {r.get('cell_id')} is not a declared cell of this suite")
        try:
            report_mod.validate_cell(r, cell, suite=suite_obj, roster_entry=roster_entry)
        except report_mod.ReportError as exc:
            raise SystemExit(f"{path}: row {r.get('cell_id')} would be refused by the report, so it "
                             f"is not resumable: {exc}")
        cid = r["cell_id"]
        # A set silently absorbs a repeat, so two rows for one cell would collapse into one "done"
        # and the duplicate -- which the report refuses -- would never be seen here. Detect it
        # before it reaches the set.
        if cid in keep:
            raise SystemExit(f"{path}: cell {cid} appears more than once. A duplicate row is a "
                             f"double measurement, not a resumable one; move the file aside.")
        keep.add(cid)
        order.append(cid)
    if stale:
        raise SystemExit(
            f"{path} holds {len(stale)} rows from a different protocol (e.g. {stale[:2]}). "
            f"Resuming would mix protocols in one file; move it aside and re-run.")
    return keep


def _entry_dict(entry) -> dict:
    """Roster entry in the shape `model_adapter` expects. Idempotent: a dict passes through.

    Feasibility supplies a synthetic entry that is already in adapter shape, and it shares the same
    preparation path as scoring.
    """
    if isinstance(entry, dict):
        return entry
    d = {"path": entry.resolved(), "arm": entry.controller_arm,
         "cross_runtime": bool(entry.cross_runtime),
         "system_id": entry.system_id, "checkpoint_sha256": entry.checkpoint_sha256,
         "estimator_path": entry.estimator_path, "estimator_sha256": entry.estimator_sha256}
    if getattr(entry, "kind", None):
        # An external published baseline. `weights` is the same resolved file as `path`; both are
        # supplied so the adapter can be handed either spelling.
        d.update({"kind": entry.kind, "weights": entry.resolved(),
                  "options": dict(entry.options or {})})
    return d


def _obstacle_track_for(cell, suite_obj):
    """Rebuild the A-cell track from the EXACT frozen placement.

    Re-running the feasibility search during scoring would let the obstacle move between systems --
    the search depends on geometry only, but the frozen suite is what every row claims to share, and
    a scenario that is re-derived is not a frozen one.
    """
    from f1sim.learn import common
    from f1sim.params import VehicleParams
    from . import obstacle
    rec = (suite_obj.placements or {}).get(cell.map_id)
    if not rec:
        raise SystemExit(f"no frozen placement for {cell.map_id}: run `geometry --freeze` first")
    frozen = rec["placement"]
    s_obs = float(frozen["s_obs_m"])
    side = int(frozen.get("side", 1))
    vp = VehicleParams()
    base, rls = common.load_tracks([cell.map_id], racelines=True, drop_infeasible=False)
    trk, place = obstacle.place_blocking_obstacle(base[0], s_obs, vehicle_length=vp.length,
                                                 vehicle_width=vp.width, side=side)

    # The regenerated placement must match the frozen record in FULL, not just in the two inputs it
    # was rebuilt from. Checking only `s_obs` and `side` meant the recorded size, centre and cell
    # count were never enforced: a frozen `size_m` of [100, 100] still accepted an ordinary small
    # box, because nothing compared them. A scenario that is re-derived is not a frozen one.
    got = place.as_dict()
    for key, tol in (("s_obs_m", 1e-9), ("side", 0), ("half_lane_m", 1e-6),
                     ("corridor_centre_offset_m", 1e-6), ("blocked_span_m", 1e-6),
                     ("n_cells", 0)):
        if key not in frozen:
            continue
        a_, b_ = got[key], frozen[key]
        drift = abs(float(a_) - float(b_))
        if drift > tol:
            raise SystemExit(f"{cell.map_id}: regenerated placement differs from the freeze on "
                             f"{key}: {a_!r} != {b_!r}. The suite is frozen; refusing to score a "
                             f"scenario that does not reproduce.")
    for key in ("size_m", "centre_xy"):
        if key not in frozen:
            continue
        a_, b_ = list(got[key]), list(frozen[key])
        if len(a_) != len(b_) or any(abs(float(x) - float(y)) > 1e-6 for x, y in zip(a_, b_)):
            raise SystemExit(f"{cell.map_id}: regenerated placement differs from the freeze on "
                             f"{key}: {a_!r} != {b_!r}. Refusing to score.")

    # and the occupancy itself: the record's cell count is the number of cells the box ADDS
    added = int((trk.occupancy & ~base[0].occupancy).sum())
    if "n_cells" in frozen and added != int(frozen["n_cells"]):
        raise SystemExit(f"{cell.map_id}: the stamped obstacle covers {added} cells, the freeze "
                         f"records {frozen['n_cells']}. Refusing to score.")
    return [trk], rls, s_obs


def _prepare(entry, extra, cell, suite_obj, device, allow_oracle: bool = False):
    """One prepared cell via the adapter, with the frozen placement and declared spawn applied.

    `allow_oracle` is forwarded and nothing in this module ever sets it: `run` scores the suite, and
    a checkpoint that reads the other cars' true state out of the simulator does not produce a suite
    row. It exists for a caller that is deliberately measuring an oracle arm on the same cells and
    will label what it gets as one.
    """
    from . import model_adapter as ma
    tracks = rls = None
    spawn = None
    adapter = suite_obj.adapter_suite(cell)
    race_size = int(adapter["race_size"]) if cell.suite in ("O", "T") else 1
    # INTERFACE (adapter-contract.md): `envs` is the number of measured LEARNERS. The adapter
    # multiplies by race_size to size the simulator. Core must not multiply as well -- doing both
    # gives 32 cars for an 8-learner 2-car race.
    cell_d = {"map": cell.map_id, "true_mu": cell.mu, "seed": cell.seed,
              "envs": cell.envs, "race_size": race_size}
    s_obs = None
    if cell.suite == "A":
        tracks, rls, s_obs = _obstacle_track_for(cell, suite_obj)
        spawn = s_obs + suite_obj.s_start_offset_m
    prepared = ma.prepare_cell(_entry_dict(entry), extra, cell_d, adapter,
                               device, tracks_override=tracks, racelines=rls, spawn_s_m=spawn,
                               allow_oracle=allow_oracle)
    router = None
    if race_size > 1 and prepared.env.tracker is not None:
        # AFTER the adapter: it installs the arm on the original tracker, and wrapping earlier would
        # hook the wrapper instead. Restored before `close()` so the adapter uninstalls the arm from
        # the object it installed it on.
        from f1sim.mpc import PlanTracker
        from .routed_tracker import RoutedTracker
        env = prepared.env
        original = env.tracker
        reference = PlanTracker(env.B, env.device, original.wb, original.s_max, original.v_max)
        router = RoutedTracker(candidate=original, reference=reference, env=env)
        env.tracker = router
    elif race_size > 1:
        # `direct` action mode: there is no tracker to route. That is not a gap in the protocol, it
        # is the protocol being satisfied structurally -- `gym_env._opponent_actions` builds the
        # teacher's command from the raceline and `teacher_action_to_normalized` (`:977-985`),
        # touching nothing the candidate owns, so an external baseline cannot move its opponent even
        # in principle. `RoutedTracker` exists only because the plan path pushes every car's plan
        # through one tracker object; here there is none.
        pass
    return prepared, s_obs, router


def _run_one(prepared, policy, cell, s_obs, suite):
    """Drive one prepared cell. The budget is the one the adapter derived, not a constant."""
    from .runner import run_cell
    from f1sim.params import VehicleParams
    vp = VehicleParams()
    env = prepared.env
    # The adapter derives the budget from the declared laps and the track length and writes it into
    # EnvConfig; take it from there rather than assuming a constant, and cross-check the protocol it
    # reports so a silent disagreement fails here instead of shortening every trial.
    steps = int(env.ecfg.max_steps)
    declared = prepared.protocol.get("budget_steps") if prepared.protocol else None
    if declared is not None and int(declared) != steps:
        raise SystemExit(f"budget disagreement: protocol says {declared}, env says {steps}")
    return run_cell(env, policy, suite=cell.suite, n_steps=steps, s_obs_m=s_obs,
                    hold_steps=int(round(suite.hold_seconds / float(env.sim.control_dt))),
                    vehicle_length=vp.length, vehicle_width=vp.width, frozen=True,
                    controller=prepared, seed=cell.seed,
                    # the adapter reports the effective layout under "spec"
                    obs_spec=(prepared.protocol or {}).get("spec"),
                    **_traffic_windows(suite))


def _traffic_feasible(res: dict, protocol: dict | None) -> tuple:
    """Is a T cell a scenario somebody has driven, and does it contain the thing it claims to?

    Three conditions, and deliberately NOT "a pass was completed". A completed pass is the O
    family's gate, it could not be shown on either real floor by any reference driver anybody has
    written, and making it the gate here would drop the same four scenarios again and leave the
    traffic number a generated-map number. What T needs demonstrated is that the stint is
    *survivable* and that it actually contains traffic:

      1. the reference driver came through at least one trial clean -- no wall, no contact;
      2. at least one trial met an opponent inside the contention window, so the cell measures
         traffic rather than a lonely lap;
      3. on an event scenario, an event landed inside that window -- the contract's "measure it,
         don't assume". A brake that fires while the learner is half a lap away is a schedule entry
         and not something the learner had to react to, and a cell whose defining feature only
         shows up sometimes is two cells sharing a name.

    Passes are reported alongside as evidence about headroom, and never gate anything.
    """
    t = res["tally"]
    clean = int(t["successes"])
    contended = int(res.get("contended", 0))
    passes = sum(res.get("passes", []) or [])
    ev_trials = int(res.get("event_in_window_trials", 0))
    n = int(res.get("n", 0)) or 1
    wants_events = bool((protocol or {}).get("opp_events"))
    ok = clean > 0 and contended > 0 and (ev_trials > 0 or not wants_events)
    why = []
    if clean == 0:
        why.append("no clean run")
    if contended == 0:
        why.append("never met traffic")
    if wants_events and ev_trials == 0:
        why.append("no event inside the contention window")
    detail = (f"clean {clean}/{n} contended {contended}/{n} passes {passes}"
              + (f" event-in-window {ev_trials}/{n}" if wants_events else "")
              + (("  << " + "; ".join(why)) if why else ""))
    return ok, detail


def _traffic_windows(suite) -> dict:
    """The two arc windows the T trace measures over, from the FROZEN suite.

    Read from the suite rather than defaulted at the call site: they decide what "in contention" and
    "close enough to attack" mean, so a run that picked them up from a function default would be
    measuring something the frozen definition does not describe.
    """
    t = getattr(suite, "traffic", None) or {}
    if not t:
        return {}
    return {"contention_range_m": float(t["contention_range_m"]),
            "attack_range_m": float(t["attack_range_m"])}


def _read_results(path: str) -> dict:
    """Read raw cells. Deliberately does NOT aggregate.

    Aggregation now refuses an unvalidated row, and validation happens in `cmd_report`; aggregating
    here would run it first and defeat that ordering.
    """
    if path.endswith(".jsonl"):
        cells = [json.loads(ln) for ln in open(path) if ln.strip()]
    else:
        with open(path) as fh:
            payload = json.load(fh)
        cells = payload.get("cells")
        if cells is None:
            # A caller-supplied `summary` is not evidence. Everything rendered must be derived
            # from raw per-cell records, or there is nothing tying the table to a measurement.
            raise report_mod.ReportError(
                f"{path}: no raw cells. Provide cells.jsonl or a payload with a `cells` list; a "
                f"precomputed `summary` is not accepted, because the renderer must derive it.")
    # The raw result stays on the row: the validator checks the per-trial arrays against the counts
    # derived from them. `n_envs` is NOT overwritten with `result.n` -- that overwrite discarded
    # exactly the disagreement the validator now looks for.
    return {"rows": [dict(c) for c in cells], "cells": cells}


def cmd_report(a) -> int:
    s, frozen = suite_mod.load(a.suite)
    payload = _read_results(a.results)
    entries = roster_mod.load(a.roster)
    try:
        info = report_mod.validate_results(
            payload["rows"], suite_freeze=frozen,
            expected_systems={e.system_id for e in entries},
            expected_trials=s.expected_trials(),
            expected_cells=s.cells(),          # off-grid rows must not satisfy a total
            # The declared speed cap, budget, noise policy and backend are only checked when the
            # Suite reaches `validate_cell`. Omitting it here silently skipped all four on the
            # authoritative path.
            suite=s,
            roster={e.system_id: e for e in entries})
    except report_mod.ReportError as exc:
        print(f"refusing to render: {exc}", file=sys.stderr)
        return 1
    # Only now: aggregation consumes validated rows, never raw input.
    info.update(report_mod.aggregate(payload["cells"]))
    md = report_mod.render_markdown(info, suite=s)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(md)
    print(f"wrote {a.out}  ({info['n_systems']} systems, {info['n_rows']} cells)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m f1sim.learn.benchmark", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("plan"); q.set_defaults(fn=cmd_plan)
    q.add_argument("--suite", default="suite-v1.json")
    q.add_argument("--roster")
    q.add_argument("--map-lengths")

    q = sub.add_parser("geometry"); q.set_defaults(fn=cmd_geometry)
    q.add_argument("--suite", default="suite-v1.json")
    q.add_argument("--version", default=suite_mod.SUITE_VERSION,
                   choices=sorted(suite_mod.VERSIONS),
                   help="which scenario set to prove and freeze (v1 in-distribution, v2 held out)")
    q.add_argument("--s-obs", type=float, default=20.0)
    q.add_argument("--inherit-placements", metavar="SUITE.json",
                   help="take the obstacle placements from an already frozen suite instead of "
                        "re-deriving them. Required to freeze a successor suite whose avoidance "
                        "cells must be the predecessor's: the search depends on --s-obs, and a "
                        "re-derived placement is a different scenario wearing the same name")
    q.add_argument("--measure", action="store_true", help="time a teacher-driven cell per map")
    q.add_argument("--measure-steps", type=int, default=600)
    q.add_argument("--freeze", action="store_true")

    q = sub.add_parser("gate"); q.set_defaults(fn=cmd_gate)
    q.add_argument("--map", default="gen:control:1400")
    q.add_argument("--steps", type=int, default=40)
    q.add_argument("--arms", default="legacy,estimated",
                   help="arms to cross against the actors")
    q.add_argument("--estimator", required=True,
                   help="pinned frozen student; required because the estimated arm has no default")

    q = sub.add_parser("feasibility"); q.set_defaults(fn=cmd_feasibility)
    q.add_argument("--suite", default="suite-v1.json")
    q.add_argument("--envs", type=int, default=4)
    q.add_argument("--device", default="cpu")
    q.add_argument("--tmpdir")
    q.add_argument("--out")
    q.add_argument("--only", help="comma-separated families or family:variant to run "
                                  "(e.g. 'T', 'T:event', 'A,O'); default is every declared cell")

    q = sub.add_parser("run"); q.set_defaults(fn=cmd_run)
    q.add_argument("--suite", default="suite-v1.json")
    q.add_argument("--roster", required=True)
    q.add_argument("--system", required=True)
    q.add_argument("--out", default="results")
    q.add_argument("--device", default="cpu")
    q.add_argument("--lease", action="store_true",
                   help="exclusive GPU access granted; without it `run` verifies and refuses")
    q.add_argument("--estimator", help="pinned frozen student for the independence gate")
    q.add_argument("--only", help="comma-separated families or family:variant to run "
                                  "(e.g. 'T' or 'T:pace'). Selects cells of the frozen suite and "
                                  "changes nothing about how any of them is measured; the result "
                                  "is a family table, not a leaderboard row.")
    q.add_argument("--gate-steps", type=int, default=60,
                   help="must exceed the estimator's warm_frames or the gate cannot cover warm")

    q = sub.add_parser("report"); q.set_defaults(fn=cmd_report)
    q.add_argument("--suite", default="suite-v1.json")
    q.add_argument("--roster", required=True)
    q.add_argument("--results", required=True)
    q.add_argument("--out", default="docs/benchmarks/leaderboard.md")

    a = p.parse_args(argv)
    return a.fn(a)



if __name__ == "__main__":
    raise SystemExit(main())
