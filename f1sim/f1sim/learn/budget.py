"""The Jetson real-time budget, measured.

The car runs the policy at the LiDAR's 40 Hz (`params.py` `control_rate`), so one control step has
25 ms for scan preprocessing + policy forward + the iLQR plan tracker + publishing. The Jetson is
not on this desk, so the number that gates this work is a RATIO on this machine, measured the same
way for both networks, and it is a proxy -- say so wherever it is quoted:

    CPU, single thread, batch 1, fp32, 200 iterations after warm-up (`torch.utils.benchmark`)

Rule (CONTRACT.md): the new actor's forward must stay within **1.5x** the frozen original's, and
its parameter count within **2x**. `python -m f1sim.learn.budget` prints the table.

What is timed is a deployment control step's network part: the extra scan channels (they are built
on the car, once per scan) and then the actor's forward with its hidden state carried in and out --
`Actor.step`, which is what `ObsBuilder` -> `act()` reaches on the real node. The critic is never
timed: it does not run on the car.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

#: The frozen original every ratio is measured against.
BASELINE = os.path.join(os.path.expanduser("~"), "f1sim_runs", "_baselines",
                        "frozen_original_48cc698f.pt")

#: Measurement protocol, quoted with every number. Built from the numbers actually used, so a run
#: with different settings cannot report the default's sentence.
def protocol(iters: int = 200, repeats: int = 5, threads: int = 1) -> str:
    return (f"CPU, {threads} thread, batch 1, fp32, the fastest of {repeats} block(s) of {iters} "
            f"iterations after warm-up (torch.utils.benchmark)")


PROTOCOL = protocol()


def parameter_counts(model) -> dict:
    """Actor / critic / total parameter counts. The budget's 2x rule is on the actor -- it is the
    only half that is exported -- but the total is reported too, because the contract's "2.25 M
    actor" is in fact the whole actor-critic of `frozen_original_48cc698f.pt` (the actor alone is
    1.17 M) and a ratio has to say which number it divided."""
    actor = sum(p.numel() for p in model.actor.parameters())
    critic = sum(p.numel() for p in model.critic.parameters())
    return {"actor": actor, "critic": critic, "total": actor + critic}


def _inputs(model, device="cpu", batch: int = 1):
    meta = model.meta
    extra = len((meta.get("scan_channels") or {}).get("channels") or ())
    g = torch.Generator().manual_seed(0)
    # A corridor, not noise. The aligned channel's warp is a scatter over the beams that returned,
    # so a scan of uniform noise -- every beam a different range, none of them at the clamp -- times
    # a different amount of work from the one the car sees. Half a corridor plus a far field is the
    # cheap approximation of a real scan's structure that keeps the no-return branch alive.
    n = int(meta["n_beams"])
    ramp = torch.linspace(0.2, 1.0, n)
    scan = (ramp[None, None, :].expand(batch, int(meta["n_stack"]), n).contiguous()
            + 0.01 * torch.rand(batch, int(meta["n_stack"]), n, generator=g)).clamp(0, 1).to(device)
    proprio = torch.rand(batch, int(meta["proprio_dim"]), generator=g).to(device)
    return scan, proprio, extra


def actor_step_ms(model, device="cpu", batch: int = 1, iters: int = 200, threads: int = 1,
                  include_channels: bool = True, repeats: int = 5) -> dict:
    """Milliseconds per control step for the actor's network part, under `PROTOCOL`.

    Returns {"forward_ms", "channels_ms", "step_ms"}: the actor's forward, the extra scan channels
    (0.0 when none are enabled), and their sum -- which is what the 25 ms budget actually contains.

    `repeats` blocks of `iters` iterations, and the **minimum** block is reported. A single block is
    one sample: on a machine that is also running a training job or a test suite it reads whatever
    contention it happened to meet, and it read a 1.76x ratio for a network that is 1.11x when the
    machine is quiet. The minimum is the least contaminated estimate of the work itself, which is
    what a ratio between two networks is supposed to compare.
    """
    from torch.utils import benchmark
    from .obs import ScanAugment

    model = model.eval()
    scan, proprio, extra = _inputs(model, device, batch)
    meta = model.meta
    aug = None
    pro_for_channels = None
    if extra:
        chan = meta["scan_channels"]
        aug = ScanAugment((chan["channels"]), int(meta["n_beams"]), batch,
                          device=device, tau_s=float(chan["memory_tau_s"]),
                          aligned=chan.get("aligned"))
        if chan.get("aligned"):
            # The aligned channel warps with the car's own measured motion, so it is timed on a car
            # that is moving: at 9 m/s and 1 rad/s the warp is a real transform rather than the
            # identity, and the scatter fans the beams out the way it does on track.
            pro_for_channels = torch.zeros(batch, int(chan["aligned"]["proprio"]["proprio_dim"]),
                                           device=device)
            idx = chan["aligned"]["proprio"]
            pro_for_channels[:, int(idx["speed"])] = 9.0 / float(idx["v_max"])
            pro_for_channels[:, int(idx["yaw_rate"])] = 1.0 / float(idx["gyro_scale"])
            for _ in range(int(chan["aligned"]["k"]) + 2):        # fill the ring before timing it
                aug(scan, pro_for_channels)
    scan_in = scan if aug is None else aug(scan, pro_for_channels)
    h = model.actor.initial_hidden(batch, device=scan.device, dtype=scan.dtype)

    def forward():
        with torch.no_grad():
            _mu, _h = model.actor.step(scan_in, proprio, None, h)

    def channels():
        with torch.no_grad():
            aug(scan, pro_for_channels)

    def best(fn):
        t = benchmark.Timer(stmt="f()", globals={"f": fn}, num_threads=threads)
        return min(t.timeit(iters).median for _ in range(max(1, repeats))) * 1e3

    fwd = best(forward)
    ch = best(channels) if (aug is not None and include_channels) else 0.0
    return {"forward_ms": fwd, "channels_ms": ch, "step_ms": fwd + ch}


def clearance_step_ms(device="cpu", batch: int = 1, iters: int = 200, threads: int = 1,
                      repeats: int = 5, cspec=None, n_beams: int = 1081,
                      fov: float = 4.71238898, range_max: float = 10.0) -> dict:
    """Milliseconds per control step for the `clearance` arm, under the same `PROTOCOL`.

    Returns {"grid_ms", "adjust_ms", "step_ms"}: building the local occupancy and its distance
    field, choosing and applying the adjustment, and their sum -- which is what the arm adds to the
    25 ms the car has. It is a different kind of cost from the actor's forward (a few thousand
    elementwise passes over a small grid rather than a convolution stack), so it is reported beside
    the network rather than folded into its ratio.

    The scan is a synthetic corridor with an obstacle, not zeros: an empty scan leaves the
    occupancy empty, the distance field saturated and the candidate search unanimous, which times
    the cheapest path the arm has rather than the one it runs.
    """
    import math

    from torch.utils import benchmark

    from . import clearance as cl
    from .. import mpc as _mpc

    spec = (cspec or cl.ClearanceSpec()).validate()
    ang = cl.beam_angles(n_beams, fov, device=device)
    sa, ca = torch.sin(ang), torch.cos(ang)
    r = torch.full((batch, n_beams), range_max, device=device)
    for sgn in (1.0, -1.0):                       # a 1.6 m corridor
        t = torch.where(sa * sgn > 1e-6, 0.8 / (sa * sgn).clamp_min(1e-6), torch.full_like(sa, 1e9))
        r = torch.minimum(r, t[None])
    b = ca * -2.0                                 # and a 0.2 m post 2 m ahead, slightly to one side
    c = 2.0 ** 2 + 0.15 ** 2 - 0.20 ** 2
    disc = b * b - c + 2.0 * 0.15 * sa * 0.0
    t = torch.where(disc > 0, -b - torch.sqrt(disc.clamp_min(0)), torch.full_like(b, 1e9))
    r = torch.minimum(r, torch.where(t > 0, t, torch.full_like(t, 1e9))[None])
    scan = (r / range_max).clamp(0.0, 1.0)
    action = torch.zeros(batch, _mpc.ACT_DIM, device=device)
    v = torch.full((batch,), 3.0, device=device)
    cap = torch.full((batch,), 4.5, device=device)
    pspec = _mpc.PlanSpec()

    def grid():
        with torch.no_grad():
            cl.distance_field(cl.occupancy(scan, ang, spec, range_max), spec)

    dist = cl.distance_field(cl.occupancy(scan, ang, spec, range_max), spec)

    def adjust():
        with torch.no_grad():
            cl.adjust(action, v, cap, dist, pspec, spec, 10.0)

    def best(fn):
        t_ = benchmark.Timer(stmt="f()", globals={"f": fn}, num_threads=threads)
        return min(t_.timeit(iters).median for _ in range(max(1, repeats))) * 1e3

    g, a = best(grid), best(adjust)
    return {"grid_ms": g, "adjust_ms": a, "step_ms": g + a,
            "cells": spec.ny * spec.nx, "candidates": 2 * spec.n_shift + 1,
            "passes": 2 * spec.radius_cells, "batch": batch, "n_beams": n_beams,
            "protocol": protocol(iters, repeats, threads)}


def measure(model, device="cpu", iters: int = 200, threads: int = 1, repeats: int = 5) -> dict:
    torch.set_num_threads(threads)
    out = dict(actor_step_ms(model, device=device, iters=iters, threads=threads, repeats=repeats))
    out.update(parameter_counts(model))
    return out


def against_baseline(model, baseline=None, device="cpu", iters: int = 200, threads: int = 1,
                     repeats: int = 5) -> dict:
    """`measure(model)` plus the ratios the contract's rule is stated in."""
    from .model import load_checkpoint
    base_path = baseline or BASELINE
    base, _extra = load_checkpoint(base_path, device)
    b = measure(base, device=device, iters=iters, threads=threads, repeats=repeats)
    m = measure(model, device=device, iters=iters, threads=threads, repeats=repeats)
    return {"baseline": b, "model": m, "baseline_path": base_path,
            "protocol": protocol(iters, repeats, threads),
            "forward_ratio": m["forward_ms"] / b["forward_ms"],
            "step_ratio": m["step_ms"] / b["step_ms"],
            "param_ratio_actor": m["actor"] / b["actor"],
            "param_ratio_total": m["total"] / b["total"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Jetson real-time budget table (a CPU-ratio proxy).")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5,
                    help="measurement blocks; the fastest is reported (see actor_step_ms)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--variants",
                    default="baseline,gru,gru+memory+edges,gru+memory+edges+aligned,"
                            "gru+memory+edges+aligned+aligned_prev+aligned_valid,"
                            "gru+memory+edges+aligned+aligned_prev+aligned_valid+motion",
                    help="comma separated: 'baseline', then 'gru' with any scan channel appended "
                         "as '+name' (memory, edges, aligned, aligned_prev, aligned_valid) and "
                         "'+motion' for the motion branch (learn.motion), which needs at least one "
                         "aligned row to read. The branch's two train-time heads are NOT built: "
                         "they are never called by the actor's forward and never exported, so "
                         "timing them would be timing something the car does not run")
    ap.add_argument("--motion-hidden", type=int, default=64)
    ap.add_argument("--json", default="", help="also write the table here")
    ap.add_argument("--clearance", action="store_true",
                    help="also time the `clearance` controller arm's per-step cost (batch 1)")
    a = ap.parse_args(argv)

    from .model import load_checkpoint, load_for_memory
    from .memory import memory_spec
    torch.set_num_threads(1)
    base, _ = load_checkpoint(a.baseline, "cpu")
    b = measure(base, iters=a.iters, repeats=a.repeats)
    rows = []
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        if name == "baseline":
            m = b
        else:
            parts = name.split("+")
            if parts[0] != "gru":
                raise SystemExit(f"variant {name!r}: expected 'baseline' or 'gru[+channel...]'")
            want_motion = "motion" in parts[1:]
            chans = [p for p in parts[1:] if p != "motion"]
            cfg = None
            if chans:
                cfg = {"channels": chans}
                if any(c.startswith("aligned") for c in chans):
                    from .obs import ObsSpec, motion_index_spec
                    sp = ObsSpec(n_beams=int(base.meta["n_beams"]), scan_stack=int(base.meta["n_stack"]),
                                 act_dim=int(base.meta["act_dim"]), hist_len=20, action_history=2)
                    if sp.proprio_dim != int(base.meta["proprio_dim"]):
                        raise SystemExit(
                            f"the baseline's proprio is {base.meta['proprio_dim']} wide and the "
                            f"spec this reconstructs is {sp.proprio_dim}: the aligned channel reads "
                            f"columns by index, so timing it against a guessed layout would time "
                            f"the wrong thing.")
                    cfg["aligned"] = {"proprio": motion_index_spec(sp)}
            mot = None
            if want_motion:
                from .motion import motion_spec
                rows = [c for c in chans if c.startswith("aligned")]
                if not rows:
                    raise SystemExit(f"variant {name!r}: the motion branch reads the aligned rows "
                                     f"and none are enabled")
                mot = motion_spec(hidden_size=a.motion_hidden, rows=rows)
            model, _e, _f = load_for_memory(a.baseline, "cpu", memory_spec(hidden_size=a.hidden),
                                            scan_channels=cfg, motion=mot)
            m = measure(model, iters=a.iters, repeats=a.repeats)
        rows.append({"variant": name, **m,
                     "forward_ratio": m["forward_ms"] / b["forward_ms"],
                     "step_ratio": m["step_ms"] / b["step_ms"],
                     "param_ratio_actor": m["actor"] / b["actor"],
                     "param_ratio_total": m["total"] / b["total"]})
    proto = protocol(a.iters, a.repeats)
    print(f"budget proxy: {proto}")
    print(f"baseline: {a.baseline}")
    print(f"{'variant':22s} {'fwd ms':>8s} {'chan ms':>8s} {'step ms':>8s} {'fwd x':>7s} "
          f"{'step x':>7s} {'actor par':>10s} {'act x':>6s} {'total par':>10s} {'tot x':>6s}")
    for r in rows:
        print(f"{r['variant']:22s} {r['forward_ms']:8.3f} {r['channels_ms']:8.3f} {r['step_ms']:8.3f} "
              f"{r['forward_ratio']:7.3f} {r['step_ratio']:7.3f} {r['actor']:10d} "
              f"{r['param_ratio_actor']:6.3f} {r['total']:10d} {r['param_ratio_total']:6.3f}")
    clear = None
    if a.clearance:
        clear = clearance_step_ms(iters=a.iters, repeats=a.repeats)
        print(f"\nclearance arm (batch 1): occupancy + distance field {clear['grid_ms']:.3f} ms, "
              f"bend + cap {clear['adjust_ms']:.3f} ms, total {clear['step_ms']:.3f} ms of the "
              f"25 ms step\n  {clear['cells']} cells, {clear['passes']} transform passes, "
              f"{clear['candidates']} candidate plans, {clear['n_beams']} beams")
    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump({"protocol": proto, "baseline": a.baseline, "rows": rows,
                       "clearance": clear}, f, indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
