"""Initial-state fingerprints, persisted per cell.

The paired-start tests compare tensors at runtime, which proves the pairing held *in that process*.
Nothing was persisted, so after a run there was no way to show that system A and system B had
actually started the same cell from the same place -- and discovering an unpaired 5000-trial matrix
afterwards is the expensive way to learn it.

Three separate digests, following the contract `eval48.state_fingerprint` established, because
matching one of them proves nothing about the rest:

* **physical** -- the plant: every car's state and every randomised parameter, opponents included.
  An opponent that started somewhere else makes a race a different experiment.
* **actor input** -- the OBSERVATION the policy is about to act on, plus the history buffers behind
  it. Two arms can share a physical state and still hand the network different scans, and the
  trajectories then diverge from the first action for a reason no physical hash would reveal.
* **calibration** -- the command path between controller and plant. The arms differ in the
  controller by design; the calibration must not differ too.

Taken immediately after the final seeded reset and BEFORE `controller.begin` or any action, because
`begin` pushes a history row and so is already part of the run rather than of its starting point.
Imports only torch and this package, so it moves with the rest of it.
"""
from __future__ import annotations

import hashlib

#: Schema version. Bump when the field lists change, so an old fingerprint is never silently
#: compared against a new one -- a comparison between two different definitions is not a comparison.
SCHEMA_VERSION = 1

#: Plant. Every car, not only the measured ones -- `sim.state` is already (B*M, ...) and is
#: deliberately NOT sliced to the learner, because an opponent that started elsewhere makes a race
#: a different experiment.
PHYSICAL_ATTRS = ("state", "ax", "ay", "att", "att_prev", "pose_prev", "imu_state", "cmd",
                  "cmd_hist", "s", "cl_idx", "lap", "collided", "steps", "tid",
                  "car_rear", "car_dims", "car_porosity")
#: What the policy sees, and the buffers that produced it.
ACTOR_ENV_ATTRS = ("scan_hist", "act_hist", "hist", "prev_action", "prev_steer_norm", "speed_cap",
                   "on_policy")
#: Plain numbers, recorded alongside the digests. A controller install that advanced the simulator
#: by one step is invisible in a tensor hash but obvious here.
SCALAR_FIELDS = ("t", "_imu_phase")
#: The command path. The arms differ in the controller; this must not differ with them.
CALIBRATION_ENV_ATTRS = ("tracker_cal", "tracker_delay", "last_cmd", "last_cmd_raw")
CALIBRATION_TRACKER_ATTRS = ("u_prev", "u_seq", "last_ref", "last_pred")


def _digest(pairs):
    """sha256 over named tensors -- original bytes, with dtype and shape mixed in.

    Three things this must not do, each of which makes two different starts hash the same:

    * **cast.** Casting to float32 first made two float64 values 1e-10 apart identical, so a plant
      state that genuinely differed read as paired. The original dtype's bytes are hashed.
    * **ignore shape.** (2, 2) and (4,) holding the same numbers are different states and used to
      produce one digest. Shape is mixed in explicitly.
    * **ignore dtype.** An int64 and a float64 tensor of the same numbers are not the same tensor.

    The count is returned because a digest over zero tensors is a perfectly stable hash of nothing,
    which reads as agreement for every system at once -- the failure mode when a field is renamed.
    """
    import torch
    h = hashlib.sha256()
    n = 0
    for name, v in pairs:
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(t.dtype).encode())
            h.update(repr(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())          # original bytes, no lossy cast
            n += 1
    return h.hexdigest(), n


def start_fingerprint(env, obs, *, obs_spec: dict | None = None) -> dict:
    """Hash the start of one cell. Call after the seeded reset, before `begin` and before acting."""
    sim = env.sim
    phys = [(k, getattr(sim, k, None)) for k in PHYSICAL_ATTRS]
    phys += [(f"P.{k}", sim.P[k]) for k in sorted(sim.P)]

    act = [(f"obs.{k}", v) for k, v in sorted((obs or {}).items())]
    act += [(k, getattr(env, k, None)) for k in ACTOR_ENV_ATTRS]

    tracker = getattr(env, "tracker", None)
    cal = [(k, getattr(env, k, None)) for k in CALIBRATION_ENV_ATTRS]
    cal += [(f"tracker.{k}", getattr(tracker, k, None)) for k in CALIBRATION_TRACKER_ATTRS]

    ph, nph = _digest(phys)
    ac, nac = _digest(act)
    cl, ncl = _digest(cal)
    out = {"schema_version": SCHEMA_VERSION,
           "physical_sha256": ph, "physical_tensors": nph,
           "actor_input_sha256": ac, "actor_input_tensors": nac,
           "calibration_sha256": cl, "calibration_tensors": ncl,
           "n_envs_total": int(getattr(env, "B", 0)),
           "race_size": int(getattr(env, "M", 1))}
    for k in SCALAR_FIELDS:
        v = getattr(sim, k, None)
        if v is not None:
            out[f"sim_{k.lstrip('_')}"] = float(v) if k == "t" else int(v)
    if obs_spec is not None:
        # The effective ObsSpec the actor was built against. Two systems can hash identical
        # observations while expecting different layouts, and that is not a paired start either.
        out["obs_spec_sha256"] = hashlib.sha256(
            repr(sorted(obs_spec.items())).encode()).hexdigest()
    if not (nph and nac and ncl):
        raise ValueError(f"start fingerprint covered {nph} physical, {nac} actor and {ncl} "
                         f"calibration tensors; a digest over none of them is a stable hash of "
                         f"nothing and would read as agreement for every system at once")
    return out


_HEX64 = 64
REQUIRED_DIGESTS = ("physical", "actor_input", "calibration")
REQUIRED_SCALARS = ("n_envs_total", "race_size")


def _strict_int(v) -> bool:
    """An int that is not a bool. `isinstance(True, int)` is True in Python, so a plain int check
    accepts `physical_tensors=True` and reads it as one tensor."""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_sha256(v) -> bool:
    """A 64-character lowercase hex digest, and nothing else."""
    return (isinstance(v, str) and len(v) == _HEX64
            and all(c in "0123456789abcdef" for c in v))


def _finite_number(v) -> bool:
    import math
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def validate_fingerprint(fp, *, where: str = "fingerprint", expected_cell=None,
                         suite=None) -> None:
    """A fingerprint is well formed, or it is not evidence.

    Checked BEFORE any comparison, because comparison is equality and `None == None` is True: two
    empty dicts compared clean and reported `paired`, which is the strongest possible false positive
    -- it says two systems started identically when neither recorded where it started.
    """
    if not isinstance(fp, dict) or not fp:
        raise ValueError(f"{where}: empty or missing; it is not evidence of anything")
    # A strict int equal to the version. `True == 1` in Python, so a bare equality accepts
    # schema_version=True; and a caller that int()s its input first turns 1.9 into 1, so a
    # fractional version must be refused here rather than rounded into agreement somewhere else.
    sv = fp.get("schema_version")
    if not _strict_int(sv) or sv != SCHEMA_VERSION:
        raise ValueError(f"{where}: schema_version {sv!r}, expected the integer {SCHEMA_VERSION}")
    # Every digest must be a real digest, the ObsSpec included -- presence alone let a row record
    # "obs_spec_sha256": "short" and still claim a matching observation layout.
    for k in REQUIRED_DIGESTS + ("obs_spec",):
        h = fp.get(f"{k}_sha256")
        if not _is_sha256(h):
            raise ValueError(f"{where}: {k}_sha256 is not a 64-character hex digest ({h!r})")
    # Tensor counts apply only to the three digests taken over tensors; the ObsSpec hash has none.
    for k in REQUIRED_DIGESTS:
        n = fp.get(f"{k}_tensors")
        # calibration too: an empty command path hashes as stably as an empty plant does
        if not _strict_int(n) or n < 1:
            raise ValueError(f"{where}: {k} covered {n!r} tensors; a digest over none is a stable "
                             f"hash of nothing and would read as agreement")
    for k in REQUIRED_SCALARS:
        if not _strict_int(fp.get(k)) or fp[k] < 1:
            raise ValueError(f"{where}: {k} is {fp.get(k)!r}; expected a positive integer "
                             f"(a bool is not a count)")
    for k in ("sim_t", "sim_imu_phase"):
        if k not in fp:
            raise ValueError(f"{where}: {k} absent; an unrecorded value cannot be shown to match")
    # A clock reading of inf or nan is not a time. It compares equal to itself for inf and unequal
    # for nan, so neither is usable as evidence that two runs started together.
    if not _finite_number(fp["sim_t"]):
        raise ValueError(f"{where}: sim_t is {fp['sim_t']!r}, which is not a finite clock reading")
    if not _strict_int(fp["sim_imu_phase"]) or fp["sim_imu_phase"] < 0:
        raise ValueError(f"{where}: sim_imu_phase is {fp['sim_imu_phase']!r}")

    # Bound to what the cell and suite declared, when they are supplied. A fingerprint recording 16
    # cars and a race of 4 is well formed on its own terms and still describes a different
    # experiment from the 8-learner solo cell it claims to be.
    if expected_cell is not None:
        want_race = int(getattr(suite, "race_size", 1)) if (
            suite is not None and getattr(expected_cell, "suite", None) == "O") else 1
        if fp["race_size"] != want_race:
            raise ValueError(f"{where}: race_size {fp['race_size']} but the declared cell is "
                             f"race_size {want_race}")
        want_envs = int(getattr(expected_cell, "envs", 0)) * want_race
        if want_envs and fp["n_envs_total"] != want_envs:
            raise ValueError(f"{where}: n_envs_total {fp['n_envs_total']} but the declared cell is "
                             f"{expected_cell.envs} learners x race_size {want_race} = {want_envs}")


def compare(a: dict, b: dict) -> dict:
    """Which parts of two starts agree. `paired` requires all three, plus the same schema.

    Both sides are validated first: equality between two absent values is not agreement.
    """
    validate_fingerprint(a, where="fingerprint a")
    validate_fingerprint(b, where="fingerprint b")
    if a.get("schema_version") != b.get("schema_version"):
        raise ValueError(f"fingerprint schema {a.get('schema_version')} vs "
                         f"{b.get('schema_version')}: different definitions are not comparable")
    parts = {k: a.get(f"{k}_sha256") == b.get(f"{k}_sha256")
             for k in ("physical", "actor_input", "calibration")}
    for k in ("sim_t", "sim_imu_phase"):
        if k in a or k in b:
            parts[k] = a.get(k) == b.get(k)
    # The shape of the experiment, not only its contents: 8 cars in a solo cell and 16 in a race of
    # 4 are different measurements even if every digest happened to match.
    for k in ("n_envs_total", "race_size"):
        parts[k] = a.get(k) == b.get(k)
    if "obs_spec_sha256" in a or "obs_spec_sha256" in b:
        parts["obs_spec"] = a.get("obs_spec_sha256") == b.get("obs_spec_sha256")
    return {"paired": all(parts.values()), "parts": parts,
            "differs": sorted(k for k, ok in parts.items() if not ok)}


def group_by_obs_spec(rows) -> dict:
    """Split rows by the observation layout their actor expects.

    Systems built on different ObsSpecs are not comparable and must not be pooled, but they are also
    not a fault: a benchmark can legitimately carry both, reported as separate groups. What is a
    fault is treating them as one population, which reads as a difference between policies when it
    is a difference between input layouts.

    All 17 systems currently share one 13-field spec, so today this returns a single group. It
    exists so that stops being an assumption the moment a second spec appears.
    """
    groups: dict = {}
    for r in rows:
        fp = (r.get("result") or {}).get("start_fingerprint") or r.get("start_fingerprint") or {}
        key = fp.get("obs_spec_sha256")
        groups.setdefault(key, []).append(r.get("system_id"))
    return groups


def assert_single_obs_spec(rows, *, cell_id: str) -> str:
    """One observation layout across the rows being compared, or refuse and name the split.

    An absent spec is refused rather than treated as matching: a row that never recorded what it
    expected cannot be shown to expect the same thing as its neighbour.
    """
    groups = group_by_obs_spec(rows)
    if None in groups:
        raise ValueError(f"{cell_id}: {sorted(groups[None])} recorded no ObsSpec, so agreement "
                         f"with the others cannot be shown")
    if len(groups) > 1:
        detail = "; ".join(f"{k[:12]}: {sorted(v)}" for k, v in sorted(groups.items()))
        raise ValueError(f"{cell_id}: rows span {len(groups)} incompatible observation layouts and "
                         f"must be grouped rather than pooled — {detail}")
    return next(iter(groups))


def assert_paired(rows, *, cell_id: str) -> dict:
    """Every row for one cell must have started identically. Raises naming what differed.

    This is the check that makes a paired comparison a paired comparison. Run it before any
    cross-system number is computed, and before reusing a prior result.
    """
    fps = [(r.get("system_id"), (r.get("result") or {}).get("start_fingerprint") or
            r.get("start_fingerprint")) for r in rows]
    missing = [sid for sid, fp in fps if not fp]
    if missing:
        raise ValueError(f"{cell_id}: no start fingerprint for {sorted(missing)}; pairing cannot "
                         f"be shown, so these rows are not comparable")
    # Every row, including the first and including a lone row. `compare` validates the pair it is
    # given, but the base row is only ever compared against later rows: with one row the loop below
    # never runs, and a row carrying nothing but a well-formed obs_spec_sha256 would be returned as
    # paired. A single row is the case where "paired" is least earned, so it is checked hardest.
    for sid, fp in fps:
        validate_fingerprint(fp, where=f"{cell_id}/{sid}")
    # An ObsSpec split is reported as its own failure, because "these systems expect different
    # inputs" and "these systems started in different places" call for different responses.
    assert_single_obs_spec(rows, cell_id=cell_id)
    base_id, base = fps[0]
    for sid, fp in fps[1:]:
        got = compare(base, fp)
        if not got["paired"]:
            raise ValueError(f"{cell_id}: {sid} did not start where {base_id} did — "
                             f"{', '.join(got['differs'])} differ. A comparison between different "
                             f"initial conditions is not a comparison between policies.")
    return {"cell_id": cell_id, "n_systems": len(fps), "paired": True}
