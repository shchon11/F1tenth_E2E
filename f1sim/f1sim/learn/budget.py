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
    scan = torch.rand(batch, int(meta["n_stack"]), int(meta["n_beams"]), generator=g).to(device)
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
    if extra:
        aug = ScanAugment((meta["scan_channels"]["channels"]), int(meta["n_beams"]), batch,
                          device=device, tau_s=float(meta["scan_channels"]["memory_tau_s"]))
    scan_in = scan if aug is None else aug(scan)
    h = model.actor.initial_hidden(batch, device=scan.device, dtype=scan.dtype)

    def forward():
        with torch.no_grad():
            _mu, _h = model.actor.step(scan_in, proprio, None, h)

    def channels():
        with torch.no_grad():
            aug(scan)

    def best(fn):
        t = benchmark.Timer(stmt="f()", globals={"f": fn}, num_threads=threads)
        return min(t.timeit(iters).median for _ in range(max(1, repeats))) * 1e3

    fwd = best(forward)
    ch = best(channels) if (aug is not None and include_channels) else 0.0
    return {"forward_ms": fwd, "channels_ms": ch, "step_ms": fwd + ch}


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
    ap.add_argument("--variants", default="baseline,gru,gru+memory,gru+edges,gru+memory+edges",
                    help="comma separated: 'baseline', then 'gru' with any of '+memory' / '+edges'")
    ap.add_argument("--json", default="", help="also write the table here")
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
                raise SystemExit(f"variant {name!r}: expected 'baseline' or 'gru[+memory][+edges]'")
            chans = [p for p in parts[1:]]
            model, _e, _f = load_for_memory(a.baseline, "cpu", memory_spec(hidden_size=a.hidden),
                                            scan_channels={"channels": chans} if chans else None)
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
    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump({"protocol": proto, "baseline": a.baseline, "rows": rows}, f, indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
