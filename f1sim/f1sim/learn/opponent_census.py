"""Does the learner actually experience each traffic situation? Learner-seconds, per situation.

The opponent-behaviour work built before this one was accepted on "it fires": 37 scripted events in
32 races. That is the wrong bar. An event that fires while the learner is 40 m away on the other
side of the lap is not a lesson, and a training distribution is what the learner *is in*, not what
was scheduled around it. So the acceptance for this branch is a census: run N races with a given
configuration and report, for each situation, how many seconds a car the policy drives spent in it.

The situations are the ones the failure attribution says the training distribution has never
contained (`docs/research/failure-attribution-2026-09-13.md` section 5): the learner always spawned
behind one car doing 0.6-1.0x on the racing line, so it has never been overtaken, never been
blocked, never had a car move away from it, and never met a car that did not brake for it.

    behind a slower car        an opponent ahead within `overtake_range`, slower than me
    alongside                  bodies overlapping longitudinally, within `opp_alongside_lat`
    being overtaken            an opponent behind within the attack window, closing
    defended against           an opponent ahead is running `defend` -- at me, this step
    yielded to                 an opponent alongside is running `yield`
    oblivious car behind       an opponent behind, closing, whose follow-gap slowdown is off
    two opponents in range     two cars within `overtake_range` at once (needs --race-size 3)

Each is a mask over (learner rows) x (steps), summed into seconds, and also counted as "how many of
the learner's cars ever saw it" -- a situation that happens a lot in one race and never in the other
thirty-one is not a distribution either.

Out of the same rollouts comes the second half, which is analysis and not a reward change: the
distribution of the `car_proximity` reward per step and of the body-to-body lateral gap while
alongside. Recipe A charges `--car-proximity-penalty 0.8` below `--car-safe-gap 0.9`, and the
suspicion this measures is that in the situations that matter the term is almost always zero.

Usage:

    python -m f1sim.learn.opponent_census CKPT --races 32 --steps 1200 \\
        --tracks 'real:map16x07,gen:control:9100' \\
        --race-size 2 --opponent pool --opp-pool teacher,self,A.pt \\
        --opp-events defend,yield,line,oblivious --opp-defend-prob 0.5 ... \\
        --spawn-order random --opp-speed 0.6 1.3 --out census.json

One timing convention, stated because it is visible in the numbers: the situation masks are built
from the state *before* each step and the reward components come out of that same step, so the
reward distribution is one control step (25 ms) later than the mask it is grouped by. Nothing in a
seconds census turns on 25 ms, and the alternative -- reading the state back after the auto-reset --
would group a reward with the situation of a different episode.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from ..gym_env import EnvConfig, SPAWN_ORDERS
from ..opponent_events import REACTIVE_BIT
from ..params import Config
from . import common
from . import opponent_config as oc
from .memory import policy_fn as memory_policy_fn
from .model import load_checkpoint

#: The situations, in report order: (key, one-line definition).
SITUATIONS = (
    ("behind_slower", "behind a slower car within overtake_range"),
    ("alongside", "alongside (bodies overlapping along the lane)"),
    ("being_overtaken", "being overtaken: a car behind inside the attack window, closing"),
    ("defended", "defended against: a car ahead is blocking my side, this step"),
    ("yielded_to", "yielded to: a car alongside is moving away"),
    ("oblivious_behind", "an oblivious car closing from behind"),
    ("two_in_range", "two opponents within overtake_range at once"),
    # --- context rows: not on the contract's list, but a behaviour nobody meets is not a behaviour
    ("own_line_opponent", "context: an opponent in range driving its own corner line"),
    ("contention", "context: any opponent within overtake_range"),
    ("attack", "context: a car ahead inside the attack window"),
    ("vs_teacher", "context: nearest opponent in range is teacher-driven"),
    ("vs_pool", "context: nearest opponent in range is a pool checkpoint"),
    ("vs_policy", "context: nearest opponent in range is the policy itself (self-play)"),
)

SITUATION_KEYS = tuple(k for k, _ in SITUATIONS)


class Census:
    """Per-situation learner-seconds, plus the lateral-signal samples, accumulated on device.

    One object per run. `observe` is called once per step with the view built *before* the step and
    the `info` the step returned; everything it touches is a (B,) or (B, C) tensor, so the census
    costs a handful of kernels per step and no host sync until `report`.
    """

    def __init__(self, env, attack_range_m: float, slower_margin: float = 0.10,
                 closing_min: float = 0.20):
        self.env, self.dt = env, env.sim.control_dt
        self.attack = float(attack_range_m)
        self.slower_margin, self.closing_min = float(slower_margin), float(closing_min)
        dev = env.device
        self.K = len(SITUATION_KEYS)
        # Rows whose experience counts: a car the policy drives *and* whose transitions the update
        # would use. In teacher and pool-without-self modes that is slot 0; in mixed and
        # pool-with-self it is every car of a self-play race as well, which is the same policy in
        # the same situations and no reason to throw away.
        self.rows = env.learner.clone()
        self.n_rows = int(self.rows.sum())
        self.seconds = torch.zeros(self.K, device=dev)
        self.ever = torch.zeros(self.K, env.B, dtype=torch.bool, device=dev)
        self.learner_seconds = torch.zeros((), device=dev)
        self.steps = 0
        # terminations, to see whether a configuration is survivable at all
        self.learner_contacts = torch.zeros((), device=dev)
        self.learner_walls = torch.zeros((), device=dev)
        self.opp_contacts = torch.zeros((), device=dev)
        self.opp_walls = torch.zeros((), device=dev)
        self.learner_distance = torch.zeros((), device=dev)
        self.opp_distance = torch.zeros((), device=dev)
        self.ended = torch.zeros((), dtype=torch.long, device=dev)
        self._have_reward = False
        # spawn grids drawn, so `--spawn-order random` can be shown to have drawn all three
        self.grid_counts = torch.zeros(3, device=dev)
        self.grid_seen = torch.zeros(3, env.B, dtype=torch.bool, device=dev)
        # The dense lateral signal. Full-width tensors plus the row masks, selected once in
        # `report`: a boolean index (or a `nonzero`) has a dynamic output shape, so doing it here
        # would be a device-to-host sync on every one of the 1200 steps, and the census would spend
        # most of its time waiting rather than driving.
        self._samples: Dict[str, List[torch.Tensor]] = {k: [] for k in (
            "rows", "along", "prox_reward", "prox_closeness", "lat_gap", "body_gap")}

    # ------------------------------------------------------------------ per step
    def observe(self, view, info, state: torch.Tensor, term: torch.Tensor, trunc: torch.Tensor) -> None:
        e = self.env.ecfg
        o = self.env.sim.other_idx
        rng = float(e.overtake_range)
        gap, lon, lat, closing = view.gap, view.lon, view.lat, view.closing
        v_me, v_op = state[:, 3][:, None], state[o][:, :, 3]
        near = gap.abs() <= rng
        ahead = near & (gap > e.car_len)
        behind = near & (gap < -e.car_len)
        along = (lon.abs() <= e.opp_alongside_lon) & (lat.abs() <= e.opp_alongside_lat)
        slower = v_op < v_me - self.slower_margin
        closing_in = closing > self.closing_min
        ev = info.get("opp_event") or {}
        react = ev.get("react")
        disp = ev.get("disposition")
        z = torch.zeros(self.env.B, dtype=torch.long, device=gap.device)
        react = z if react is None else react
        disp = z if disp is None else disp
        bit = lambda t, name: (t[o] & REACTIVE_BIT[name]) != 0
        # nearest opponent in range, for the "who was it" rows. Read off the env's own per-car
        # masks rather than `opp_driver`, which only carries a meaning in pool mode -- taking the
        # driver code outside it reported every teacher opponent as the policy itself.
        cand = torch.where(near, -gap.abs(), torch.full_like(gap, -math.inf))
        j = cand.argmax(1, keepdim=True)
        near_any = near.any(1)
        pick = lambda m: m[o].gather(1, j)[:, 0]
        near_teacher = pick(self.env.teacher_driven)
        near_pool = pick(self.env.pool_driven)
        near_policy = pick(self.env.on_policy)

        masks = {
            "behind_slower": (ahead & slower).any(1),
            "alongside": along.any(1),
            "being_overtaken": (behind & (gap.abs() <= self.attack) & closing_in).any(1),
            "defended": (ahead & bit(react, "defend")).any(1),
            "yielded_to": (along & bit(react, "yield")).any(1),
            "oblivious_behind": (behind & closing_in & bit(disp, "oblivious")).any(1),
            "two_in_range": near.sum(1) >= 2,
            "own_line_opponent": (near & bit(react, "line")).any(1),
            "contention": near_any,
            "attack": (ahead & (gap <= self.attack)).any(1),
            "vs_teacher": near_any & near_teacher,
            "vs_pool": near_any & near_pool,
            "vs_policy": near_any & near_policy,
        }
        rows = self.rows & self.env.on_policy
        stack = torch.stack([masks[k] & rows for k in SITUATION_KEYS])            # (K, B)
        self.seconds += stack.sum(1).float() * self.dt
        self.ever |= stack
        self.learner_seconds += rows.sum().float() * self.dt
        self.steps += 1
        self.ended += (term | trunc).sum()
        grid = self.env.spawn_order_race
        onehot = torch.nn.functional.one_hot(grid.clamp(0, 2), 3).bool() & rows[:, None]
        self.grid_seen |= onehot.T
        # ---- terminations. `term` is a collision of either kind; a car-car contact is reported
        # separately, so a wall is "terminated and not in contact".
        car = info.get("car_collision")
        car = torch.zeros_like(term) if car is None else car.bool()
        self.learner_contacts += (term & car & rows).sum()
        self.learner_walls += (term & ~car & rows).sum()
        opp_rows = ~self.env.on_policy
        self.opp_contacts += (term & car & opp_rows).sum()
        self.opp_walls += (term & ~car & opp_rows).sum()
        travelled = state[:, 3].abs() * self.dt
        self.learner_distance += torch.where(rows, travelled, torch.zeros_like(travelled)).sum()
        self.opp_distance += torch.where(opp_rows, travelled, torch.zeros_like(travelled)).sum()
        # ---- the dense lateral signal. The unit-less closeness is recomputed from the same
        # geometry as `gym_env.car_proximity`, so the signal is visible even at coefficient 0; the
        # reward itself comes out of the step, which is one control step later than this state (see
        # the module docstring).
        comp = info.get("reward_components") or {}
        prox = comp.get("car_proximity")
        body = torch.sqrt((lon.abs() - e.car_len).clamp_min(0.0) ** 2
                          + (lat.abs() - e.car_wid).clamp_min(0.0) ** 2)
        closeness = ((1.0 - body / e.car_safe_gap).clamp(0.0, 1.0)
                     * (1.0 + closing.clamp_min(0.0) / e.car_prox_speed_ref)).max(1).values
        # the lateral body-to-body gap of the most overlapping car, where there is one
        pick = torch.where(along, -lon.abs(), torch.full_like(lon, -math.inf)).argmax(1, keepdim=True)
        add = self._samples
        add["rows"].append(rows)
        add["along"].append(along.any(1) & rows)
        add["prox_reward"].append(torch.zeros_like(closeness) if prox is None else prox.abs())
        add["prox_closeness"].append(closeness)
        add["body_gap"].append(body.min(1).values)
        add["lat_gap"].append((lat.abs().gather(1, pick)[:, 0] - e.car_wid).clamp_min(0.0))
        self._have_reward = prox is not None

    # ------------------------------------------------------------------ the report
    @staticmethod
    def _quantiles(x: Optional[torch.Tensor]) -> dict:
        if x is None or x.numel() == 0:
            return {"n": 0}
        v = x.detach().float().cpu().numpy()
        qs = np.percentile(v, [50, 75, 90, 99]).tolist()
        return {"n": int(v.size), "mean": float(v.mean()), "zero_fraction": float((v <= 0).mean()),
                "p50": qs[0], "p75": qs[1], "p90": qs[2], "p99": qs[3], "max": float(v.max())}

    def report(self) -> dict:
        e = self.env.ecfg
        secs = self.seconds.detach().cpu().numpy()
        total = float(self.learner_seconds)
        ever = self.ever.sum(1).detach().cpu().numpy()
        km = lambda d: float(d) / 1000.0
        flat = {k: (torch.cat(v).flatten() if v else None) for k, v in self._samples.items()}
        rows, along = flat["rows"], flat["along"]

        def pick(key, mask):
            """Samples of `key` on the rows `mask` selects. One boolean index, at the end."""
            if flat[key] is None or mask is None:
                return None
            return flat[key][mask]
        return {
            "learner_rows": self.n_rows, "steps": self.steps, "learner_seconds": total,
            "races": int(self.env.B // self.env.M), "episodes_ended": int(self.ended),
            "situations": {k: {"seconds": float(s), "fraction": float(s) / total if total else 0.0,
                               "rows_ever": int(n), "rows": self.n_rows, "definition": d}
                           for (k, d), s, n in zip(SITUATIONS, secs, ever)},
            "grids": {name: int(self.grid_seen[i].sum()) for i, name in enumerate(SPAWN_ORDERS[:3])},
            "terminations": {
                "learner_car_contacts": int(self.learner_contacts),
                "learner_wall_collisions": int(self.learner_walls),
                "opponent_car_contacts": int(self.opp_contacts),
                "opponent_wall_collisions": int(self.opp_walls),
                "learner_km": km(self.learner_distance), "opponent_km": km(self.opp_distance),
                "learner_collisions_per_km": (int(self.learner_contacts) + int(self.learner_walls))
                                             / max(km(self.learner_distance), 1e-9),
                "opponent_walls_per_km": int(self.opp_walls) / max(km(self.opp_distance), 1e-9),
            },
            "lateral_signal": {
                "car_proximity_reward_per_step": self._quantiles(pick("prox_reward", rows) if self._have_reward else None),
                "car_proximity_reward_per_step_alongside": self._quantiles(pick("prox_reward", along) if self._have_reward else None),
                "car_proximity_closeness": self._quantiles(pick("prox_closeness", rows)),
                "body_gap_m_nearest": self._quantiles(pick("body_gap", rows)),
                "lateral_body_gap_m_alongside": self._quantiles(pick("lat_gap", along)),
                "coefficients": {"reward_car_proximity": e.reward_car_proximity,
                                 "car_safe_gap": e.car_safe_gap, "car_len": e.car_len,
                                 "car_wid": e.car_wid, "car_prox_speed_ref": e.car_prox_speed_ref},
            },
            "definitions": {"overtake_range_m": e.overtake_range, "attack_range_m": self.attack,
                            "alongside_lon_m": e.opp_alongside_lon,
                            "alongside_lat_m": e.opp_alongside_lat,
                            "slower_margin_mps": self.slower_margin,
                            "closing_min_mps": self.closing_min,
                            "learner_rows": "cars the policy drives whose transitions the update uses"},
        }


def format_report(rep: dict) -> str:
    """The table. Seconds first, because that is the acceptance criterion."""
    total = rep["learner_seconds"]
    w = max(len(d["definition"]) for d in rep["situations"].values())
    lines = [f"{rep['races']} races, {rep['learner_rows']} learner car(s), {rep['steps']} steps, "
             f"{total:.0f} learner-seconds, {rep['episodes_ended']} episodes ended",
             "",
             f"{'situation':{w}}  {'seconds':>9}  {'share':>7}  {'cars ever':>10}"]
    lines.append("-" * (w + 32))
    for key, d in rep["situations"].items():
        lines.append(f"{d['definition']:{w}}  {d['seconds']:9.1f}  {100 * d['fraction']:6.1f}%  "
                     f"{d['rows_ever']:5d}/{d['rows']:<4d}")
    t = rep["terminations"]
    lines += ["", f"learner: {t['learner_car_contacts']} car contacts, {t['learner_wall_collisions']} walls "
                  f"over {t['learner_km']:.2f} km ({t['learner_collisions_per_km']:.2f} /km)",
              f"opponents: {t['opponent_car_contacts']} car contacts, {t['opponent_wall_collisions']} walls "
              f"over {t['opponent_km']:.2f} km ({t['opponent_walls_per_km']:.2f} walls/km)",
              f"grids drawn (cars that saw each): " +
              ", ".join(f"{k} {v}" for k, v in rep["grids"].items()), ""]
    ls = rep["lateral_signal"]
    c = ls["coefficients"]
    lines.append(f"dense lateral signal (car-proximity penalty {c['reward_car_proximity']:g} below "
                 f"{c['car_safe_gap']:g} m):")
    lines.append(f"{'quantity':44} {'n':>8} {'zero':>7} {'p50':>9} {'p90':>9} {'p99':>9} {'max':>9}")
    for key in ("car_proximity_reward_per_step", "car_proximity_reward_per_step_alongside",
                "car_proximity_closeness", "body_gap_m_nearest", "lateral_body_gap_m_alongside"):
        q = ls[key]
        if not q.get("n"):
            lines.append(f"{key:44} {0:>8}  (no samples)")
            continue
        lines.append(f"{key:44} {q['n']:>8} {100 * q['zero_fraction']:6.1f}% "
                     f"{q['p50']:9.4f} {q['p90']:9.4f} {q['p99']:9.4f} {q['max']:9.4f}")
    return "\n".join(lines)


@torch.no_grad()
def run(args) -> dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    names = common.track_names(args.tracks, draws=args.obstacle_draws, seed=args.seed)
    trs, rls = common.load_tracks(names, racelines=oc.needs_racelines(args) or args.teacher)
    model, meta = (None, {})
    if not args.teacher:
        model, meta = load_checkpoint(args.ckpt, device)
        model.eval()
    spec = meta.get("spec", {})
    mode = ("plan" if (model is not None and model.meta.get("act_dim", 2) >= 5)
            else ("plan" if args.teacher and args.action_mode == "plan" else "direct"))
    envs = args.races * args.race_size
    ecfg = EnvConfig(speed_cap=args.speed_cap, resample_track_on_reset=True, action_mode=mode,
                     scan_stack=spec.get("scan_stack", 6), scan_stride=spec.get("scan_stride", 1),
                     hist_len=spec.get("hist_len", 20), hist_stride=spec.get("hist_stride", 2),
                     max_steps=args.steps,
                     reward_car_proximity=args.car_proximity_penalty, car_safe_gap=args.car_safe_gap,
                     reward_overtake=args.overtake_bonus, reward_car_contact=args.car_contact_penalty,
                     **oc.env_kwargs(args))
    cfg = Config()
    env = common.make_env(trs, envs, device, ecfg, cfg=cfg, seed=args.seed, rls=rls,
                          teacher_grip=args.teacher_grip)
    env.sim.warmup()
    print(oc.describe(args), flush=True)
    if args.teacher:
        tp = common.make_teacher(rls, env)
        policy = lambda obs: env.teacher_label(tp)
        policy_reset = None
    else:
        policy = memory_policy_fn(model, env.B, device=device, deterministic=True)
        policy_reset = policy.reset
    obs, _ = env.reset(seed=args.seed)
    if policy_reset:
        policy_reset()
    census = Census(env, attack_range_m=args.attack_range)
    t0 = time.perf_counter()
    for t in range(args.steps):
        # The situation is read from the state this step's command acts on, and so are the reactive
        # bits (`_opponent_actions` computes them from the same state at the top of `step`).
        view = env.learner_view()
        st = env.sim.state.clone()
        obs, _rew, term, trunc, info = env.step(policy(obs))
        if policy_reset:
            policy_reset(term | trunc)
        census.observe(view, info, st, term, trunc)
    rep = census.report()
    rep["metadata"] = {
        "checkpoint": None if args.teacher else str(args.ckpt), "teacher": bool(args.teacher),
        "tracks": list(names), "seed": args.seed, "device": str(device), "envs": envs,
        "races": args.races, "steps": args.steps, "speed_cap": args.speed_cap,
        "action_mode": mode, "wall_time_s": time.perf_counter() - t0,
        "opponent": oc.describe(args), "env_config": asdict(env.ecfg),
        "pool": list(env.pool_names), "pool_loaded": None if env.pool is None else env.pool.describe(),
    }
    return rep


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description="Learner-seconds per traffic situation, for a given opponent configuration.")
    ap.add_argument("ckpt", nargs="?", default="", help="checkpoint the learner drives")
    ap.add_argument("--teacher", action="store_true", help="drive the learner's car with the raceline teacher instead")
    ap.add_argument("--tracks", default="real:map16x07,gen:control:9100")
    ap.add_argument("--obstacle-draws", type=int, default=8)
    ap.add_argument("--races", type=int, default=32, help="parallel races (envs = races x race-size)")
    ap.add_argument("--steps", type=int, default=1200, help="control steps per race (40 = 1 s)")
    ap.add_argument("--speed-cap", type=float, default=9.0)
    ap.add_argument("--attack-range", type=float, default=3.0, metavar="M",
                    help="[m] the tight arc window: close enough that a pass is on, and the window "
                         "'being overtaken' is measured in. The benchmark's T family uses the same 3 m")
    ap.add_argument("--car-proximity-penalty", type=float, default=0.8,
                    help="the recipe's value, so the reported reward distribution is the reward the "
                         "recipe actually pays. This tool never changes a reward; it reports one")
    ap.add_argument("--car-safe-gap", type=float, default=0.9)
    ap.add_argument("--overtake-bonus", type=float, default=1.0)
    ap.add_argument("--car-contact-penalty", type=float, default=5.0)
    ap.add_argument("--action-mode", default="plan", choices=["direct", "plan"])
    ap.add_argument("--teacher-grip", default="true", choices=["true", "nominal", "conservative"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=4401)
    ap.add_argument("--out", default="", help="write the report as JSON here")
    ap.add_argument("--tag", default="", help="label carried into the JSON")
    oc.add_arguments(ap)
    a = ap.parse_args(argv)
    if not a.ckpt and not a.teacher:
        raise SystemExit("give a checkpoint to drive, or --teacher: which car the learner *is* "
                         "decides which situations it gets into, so there is no default.")
    if a.race_size < 2:
        raise SystemExit(f"--race-size {a.race_size}: a census of traffic situations needs traffic.")
    oc.validate(a)
    rep = run(a)
    if a.tag:
        rep["metadata"]["tag"] = a.tag
    print(format_report(rep), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"wrote {a.out}", flush=True)
    return rep


if __name__ == "__main__":
    main()
