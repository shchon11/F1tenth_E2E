"""Session engine for plan_steps2 (November) checkpoints and baseline ego drivers.

Runs the prior developer's `exam_record.py` UNMODIFIED in this process (`runpy`), as audit C did, with the job's
definition injected around it -- nothing in exam_record is edited:
  - the world (`F1SIM_EXAM_WORLD`): the suite's, never the checkpoint's own (a policy trained on another map is
    examined on the suite's map)
  - the map: `common.load_tracks` swaps the world's track for the job's (held-out variants, another venue)
  - the box layouts: `world_of.exam_bank` returns the job's bank directory (count "line3": layout i -> session i % n)
  - the physics profile: `common.make_env` updates `cfg.rand.ranges` (and dotted cfg fields) before the env is built
  - the ego driver: `state_rl.build_env` installs a baseline through `env.set_ego_driver` (the console's 내 차 path);
    the host checkpoint's plan is still computed every decision but never reaches the car
  - the seed (`F1SIM_EXAM_SEED`), all sessions boxed (`F1SIM_EXAM_ALL_BOXES`), races (`F1SIM_EXAM_OPP_SLOT`,
    `_HALF_LAP`, `_START_REST`, count "race4" with `F1SIM_EXAM_REF` / `F1SIM_EXAM4_PACES`)
Usage: python -m f1sim.compbench.engine_exam job.json"""
import os
import sys
import json
import runpy


def _set_dotted(obj, key, value):
    parts = key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(f"physics profile: no config field {key}")
    setattr(obj, parts[-1], type(getattr(obj, parts[-1]))(value) if not isinstance(value, (list, dict)) else value)


def split_assign(n_rows, K, n_lay=64):
    """member of ego row r (= session r): (r + r // n_lay) % K -- interleaved so that each member meets many layouts
    (session r drives layout r % n_lay, so a plain r % K would give a member only n_lay / K of them)"""
    import numpy as np
    r = np.arange(n_rows)
    return (r + r // max(int(n_lay), 1)) % K


class SplitDriver:
    """Several settings of one stateless sensor baseline on one env (tuning: K candidates share one run's sessions):
    ego row r is driven by member `split_assign(r)`. Not `stateless`: the env must hand it every ego row, in order (with
    `external_skip_unused` it would pass a subset and the positions would no longer name the sessions)."""
    stateless = False

    def __init__(self, members, n_lay=64):
        self.members = list(members)
        self.needs_speed = any(m.needs_speed for m in self.members)
        self.n_lay = n_lay

    def adapt(self, rng):
        return rng

    def reset(self, done=None):
        """an env reset of some ego rows (bool mask over the rows, in order): each member hears about its own"""
        import numpy as np
        if done is None:
            for m in self.members:
                m.reset(None)
            return
        done = np.asarray(done, bool)
        who = split_assign(done.shape[0], len(self.members), self.n_lay)
        for k, m in enumerate(self.members):
            m.reset(done[who == k])

    def command(self, rng, v):
        import numpy as np
        n = rng.shape[0]
        out = np.zeros((n, 2), dtype=np.float64)
        who = split_assign(n, len(self.members), self.n_lay)
        for k, m in enumerate(self.members):
            sel = np.nonzero(who == k)[0]
            if sel.size == 0:
                continue
            c = m.command(m.adapt(rng[sel]), None if (v is None or not m.needs_speed) else v[sel])
            out[sel] = np.asarray(c, dtype=np.float64)
        return out


class SplitPlannerDriver:
    """The same for planner baselines (`PlannerEgoDriver`, privileged-free modular stacks): member k owns its rows'
    particle filter / detection (`ModularStack` over those rows only) and is asked for those rows."""
    needs_speed = False

    def __init__(self, members, member_rows):
        self.members, self.member_rows = members, member_rows
        self.rows = None

    def reset(self, done=None):
        for m in self.members:
            if hasattr(m, "reset"):
                m.reset(None)

    def command_env(self, env, rows):
        import torch
        out = torch.zeros(rows.numel(), 2, device=rows.device)
        pos = {int(r): i for i, r in enumerate(rows.tolist())}
        for m, mr in zip(self.members, self.member_rows):
            c = m.command_env(env, mr)
            idx = torch.as_tensor([pos[int(r)] for r in mr.tolist()], device=rows.device)
            out[idx] = c.to(out.dtype)
        return out


def build_planner(env, spec, track_name, rows=None):
    """`common.build_ego_driver` for a planner kind, its stack over `rows` (default: every ego row) -- the same steps"""
    import torch
    from f1sim.learn import common as C
    from f1sim.raceline import Raceline
    kind = spec["kind"]
    lim = C.teacher_limits(*[float(x) for x in spec["limits"]])
    trs, _ = C.load_tracks([track_name], racelines=False)
    rls = [Raceline.build_cached(trs[0], cache_only=False, objective="min_time", **lim)]
    sp = dict(spec.get("stack_params") or {})
    sp.setdefault("grip", "nominal")
    if rows is None:
        # estimated False: the planner on the simulator's true state, the other cars' and the boxes' true places and this
        # episode's actuator parameters -- what the same kind gets as an opponent slot (the qualification of opponents)
        return C.build_ego_driver(env, kind, "", None, rls=rls, limits=lim, stack_params=sp,
                                  estimated=bool(spec.get("estimated", True)))
    from f1sim.modular_stack import ModularStack
    from f1sim.localization import PFParams
    from f1sim.perception import DetectParams, TrackParams
    ref = C.make_teacher(rls, env, grip=str(sp.get("grip", "true")), **dict(lim, **sp.get("teacher", {})))
    planner = ref
    if kind != "raceline":
        from f1sim.opponent_slots import build_teacher, kind_of
        pk = dict(sp.get("planner", {}))
        if kind in ("forzaeth", "forzaeth_pred"):
            from f1sim.spliner_teacher import UPSTREAM_PRED_TIME
            pk.setdefault("upstream", True)
            if kind == "forzaeth":
                pk.setdefault("fixed_pred_time", UPSTREAM_PRED_TIME)
        planner = build_teacher(kind_of(kind), ref, env, **pk)
    stack = ModularStack(env, rows, ref, pf=PFParams(**sp.get("pf", {})), det=DetectParams(**sp.get("det", {})),
                         trk=TrackParams(**sp.get("trk", {})))
    if planner is not ref:
        planner.observer = stack
    tracker = None
    if sp.get("tracker", "map") == "map":
        # `common.build_map_tracker`, sized to this member's rows (its controller keeps per-row state)
        from f1sim import map_controller as _map
        sim = env.sim
        nominal = sim.params.nominal
        P_row = {k: torch.tensor([float(v)]) for k, v in nominal.items()}
        mus = [float(ref.mu_nom * g) for g in ref.grip_levels]
        lut = _map.lut_for(P_row, mus, wheel=bool(sim.wheel_model), dt=float(sim.dt), s_max=float(nominal["s_max"]),
                           device=sim.state.device)
        tracker = _map.MAPController(lut, _map.MAPParams.from_dict(sp.get("map", {})), int(rows.numel()), sim.state.device,
                                     float(nominal["s_max"]))
    drv = C.PlannerEgoDriver(planner, kind, stack, ref=ref, tracker=tracker)
    drv.speed_scale = float(sp.get("speed_scale", 1.0))
    drv.rows = rows
    return drv


def build_ego(env, spec, track_name):
    import torch
    from f1sim.learn import common as C
    kind = spec["kind"]
    if kind == "split":
        rows = torch.nonzero(env.slot == 0).flatten()
        kinds = {m["kind"] for m in spec["members"]}
        if kinds <= set(C.EGO_PLANNER_KINDS):
            who = torch.as_tensor(split_assign(rows.numel(), len(spec["members"]), spec.get("n_lay", 64)), device=rows.device)
            mrows = [rows[who == k] for k in range(len(spec["members"]))]
            ms = [build_planner(env, m, track_name, rows=r) for m, r in zip(spec["members"], mrows)]
            return SplitPlannerDriver(ms, mrows)
        ms = [build_ego(env, m, track_name) for m in spec["members"]]
        if not all(getattr(m, "stateless", False) and not hasattr(m, "command_env") for m in ms):
            raise ValueError("split: stateless sensor drivers, or planner kinds only")
        return SplitDriver(ms, spec.get("n_lay", 64))
    if kind in C.EGO_PLANNER_KINDS:
        return build_planner(env, spec, track_name)
    if kind == "aurorus":
        # its own line follower + LiDAR avoidance on a min-time line (default: the env's opponent line, limits 7 / 6.5 / 4)
        from f1sim.raceline import Raceline
        lim = C.teacher_limits(*[float(x) for x in (spec.get("limits") or [7.0, 6.5, 4.0])])
        trs, _ = C.load_tracks([track_name], racelines=False)
        rls = [Raceline.build_cached(trs[0], cache_only=False, objective="min_time", **lim)]
        return C.build_ego_driver(env, "aurorus", "", dict(spec.get("options") or {}), rls=rls, limits=lim)
    return C.build_ego_driver(env, kind, str(spec.get("weights") or ""), dict(spec.get("options") or {}))


def install_profile(job, C, torch):
    """the physics profile: `cfg.rand.ranges` and dotted cfg fields set right before `common.make_env` builds the env (after
    state_rl.build_env's own settings, so a profile can also replace its friction pin); the draws actually made are printed"""
    prof = job.get("profile") or {}
    _mk = C.make_env

    def make_env(*a, **k):
        cfg = k.get("cfg")
        if cfg is not None:
            for key, rng in (prof.get("ranges") or {}).items():
                cfg.rand.ranges[key] = tuple(float(x) for x in rng)
            for key, val in (prof.get("cfg") or {}).items():
                _set_dotted(cfg, key, val)
        env = _mk(*a, **k)
        P = env.sim.P
        for key in ("mu", "mu_f_scale", "cmd_delay", "servo_tau", "noise_std", "steer_bias", "a_max"):
            if key in P and torch.is_tensor(P[key]):
                x = P[key].float()
                print(f"[compbench] physics {key}: min {float(x.min()):.4f} mean {float(x.mean()):.4f} max {float(x.max()):.4f}",
                      flush=True)
        return env

    C.make_env = make_env


def install_map(job, C):
    """the job's track in place of the world's first track, wherever `common.load_tracks` is asked for it"""
    world_track = str(json.load(open(job["world"]))["tracks"]).split(",")[0]
    track = job["track"]
    if track != world_track:
        _lt = C.load_tracks

        def load_tracks(names, *a, **k):
            names = [track if n == world_track else n for n in names]
            return _lt(names, *a, **k)

        C.load_tracks = load_tracks
        print(f"[compbench] map {world_track} -> {track}", flush=True)
    return track

def main(job_path):
    job = json.load(open(job_path))
    sys.path.insert(0, job["f1sim_src"])
    sys.path.insert(0, job["tools"])
    env_vars = {
        "F1SIM_EXAM_SEED": str(int(job["seed"])),
        "F1SIM_EXAM_WORLD": job["world"],
        "WANDB_MODE": "disabled",
        "F1SIM_RACELINE_CACHE_ONLY": "1" if job.get("raceline_cache_only", True) else "0",
    }
    if job.get("all_boxes", True):
        env_vars["F1SIM_EXAM_ALL_BOXES"] = "1"
    if job.get("opp_slot"):
        env_vars["F1SIM_EXAM_OPP_SLOT"] = json.dumps(job["opp_slot"])
    if job.get("half_lap"):
        env_vars["F1SIM_EXAM_HALF_LAP"] = "1"
    if job.get("start_rest"):
        env_vars["F1SIM_EXAM_START_REST"] = "1"
    if job.get("race4"):
        env_vars["F1SIM_EXAM_REF"] = os.path.expanduser(job["race4"]["ref"])
        env_vars["F1SIM_EXAM4_PACES"] = ",".join(str(float(p)) for p in job["race4"]["paces"])
    for k in [k for k in os.environ if k.startswith("F1SIM_EXAM_")]:
        if k not in env_vars:
            del os.environ[k]                                    # nothing from the caller's shell leaks into a run
    os.environ.update(env_vars)
    import torch
    print(f"[compbench] device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} on {__import__('socket').gethostname()}", flush=True)
    import f1sim.learn.common as C
    from f1sim.learn import state_rl as SR
    import world_of

    install_profile(job, C, torch)
    track = install_map(job, C)
    world_of.exam_bank = lambda world: job["bank"]
    ego = job.get("ego")
    if ego:
        _be = SR.build_env

        def build_env(a, device, extra=None):
            env = _be(a, device, extra=extra)
            drv = build_ego(env, ego, track)
            env.set_ego_driver(drv)
            print(f"[compbench] ego driver {ego['kind']} ({type(drv).__name__}) on {int(env.external_slot_rows[0].numel())} rows",
                  flush=True)
            return env

        SR.build_env = build_env
    count = "race4" if job.get("race4") else "line3"
    sys.argv = [os.path.join(job["tools"], "exam_record.py"), job["host_ckpt"], job["out"], job["mode"],
                str(int(job["n_env"])), count, str(float(job["secs"])), str(float(job["remove_at"]))]
    runpy.run_path(os.path.join(job["tools"], "exam_record.py"), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1])
