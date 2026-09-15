"""DAgger for a published architecture on our teacher's demonstrations (CONTRACT.md, D3).

    python -m f1sim.learn.baselines distill --arch tinylidarnet --name tln_it_s701 \
        --teacher interactive --tracks "$(cat .../future-20260914/tracks.txt)" --device cuda

    python -m f1sim.learn.baselines budget --weights ...           # ms per step, CPU, 1 thread

The schedule is ours (worker 17's D3), everything inside a training round is theirs; see
`distill.HYPERPARAMETERS`, which is written into every checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from . import distill
from .common import BaselineError


def _log(path):
    def log(msg):
        print(msg, flush=True)
        if path:
            with open(path, "a") as fh:
                fh.write(msg + "\n")
    return log


def build_teacher(kind: str, rls, env, log):
    """The expert whose tracked command is the label.

    `interactive` is worker 17's `f1sim.interactive_teacher.InteractiveTeacher`. It lives on
    `feat/interactive-teacher` and is imported only if that branch has been merged; until then this
    refuses rather than silently distilling from a different expert, which is the one substitution
    that would make the whole comparison mean something else. `raceline` is the declared dry run.
    """
    from f1sim.learn import common

    base = common.make_teacher(rls, env, grip="true")
    if kind == "raceline":
        return base, ("RacelineTeacher (grip 'true'): the DRY RUN. It is blind to the other cars, "
                      "so a student distilled from it can at best learn 'pass the car you see'.")
    try:
        from f1sim.interactive_teacher import InteractiveTeacher
    except ImportError as exc:
        raise BaselineError(
            "f1sim.interactive_teacher is not on this branch: worker 17's `feat/interactive-teacher` "
            "has not merged yet. Run with --teacher raceline for the dry run, or re-run this "
            "unchanged once it lands -- that is the only difference between the two.") from exc
    it = InteractiveTeacher(base, env)
    return it, (f"InteractiveTeacher (worker 17): {getattr(it, 'n_candidates', '?')} candidates, "
                f"{getattr(it, 'horizon_s', '?')} s horizon")


def cmd_distill(a) -> int:
    from f1sim.learn import common
    from f1sim.learn.baselines import end2race as e2r_mod
    from f1sim.learn.baselines.tinylidarnet_torch import TinyLidarNetTorch

    os.makedirs(a.out, exist_ok=True)
    log = _log(os.path.join(a.out, f"{a.name}.log"))
    device = torch.device(a.device)
    names = common.track_names(a.tracks) if a.tracks in ("train", "eval") else \
        [t for t in a.tracks.replace("\n", ",").split(",") if t.strip()]
    log(f"{a.name}: {len(names)} tracks, arch {a.arch}, teacher {a.teacher}, device {device}")
    tracks, rls = common.load_tracks(names, racelines=True, drop_infeasible=False)
    log(f"  {len(tracks)} tracks built")

    overrides = {"iters": a.iters, "steps": a.steps, "beta0": a.beta0, "seed": a.seed}
    if a.speed_cap:
        overrides["speed_cap"] = a.speed_cap
    env, cfg = distill.make_collection_env(
        tracks, rls, a.learners, device, spec_beams=a.n_beams, range_max=a.range_max,
        v_max=a.v_max, seed=a.seed, overrides=overrides)
    env.sim.warmup()
    teacher, teacher_note = build_teacher(a.teacher, rls, env, log)
    log(f"  teacher: {teacher_note}")
    proto = distill.collection_protocol(env, cfg, names, a.teacher, teacher_note)
    log(f"  {proto['n_cars']} cars = {proto['n_learners']} learners x {proto['race_size']}, "
        f"{proto['n_beams']} beams over {np.degrees(proto['fov']):.0f} deg, "
        f"{proto['control_rate_hz']:.0f} Hz")

    # -- the student's architecture, at ITS repo's initialisation -------------------------------
    if a.arch == "tinylidarnet":
        idx = np.arange(0, a.n_beams, a.skip_n)
        model = TinyLidarNetTorch(int(idx.size)).to(device)
        deviation = None
        log(f"  TinyLidarNet: {model.n_params} params, {idx.size} of {a.n_beams} beams "
            f"(skip {a.skip_n}), glorot/zeros as Keras initialises it")
    elif a.arch == "end2race":
        model, deviation = e2r_mod.build_untrained(repo=a.repo, hidden_scale=a.hidden_scale,
                                                   n_features=a.n_features, device=str(device))
        idx = np.linspace(0, a.n_beams - 1, a.n_features, dtype=int)
        log(f"  End2Race: {sum(p.numel() for p in model.parameters())} params, "
            f"{a.n_features} beams; model.py deviation {deviation}")
    else:
        raise SystemExit(f"unknown --arch {a.arch}")

    opt = (torch.optim.Adam(model.parameters(), lr=5e-5, eps=1e-7) if a.arch == "tinylidarnet"
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    sched = (None if a.arch == "tinylidarnet" else
             torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10))

    bufs, speed_range, history = [], (0.0, 1.0), []
    t_start = time.time()
    for it in range(a.iters):
        beta = 1.0 if it == 0 else a.beta0 * (0.5 ** (it - 1))
        driver = None
        if beta < 1.0:
            driver = _driver_for(a, model, device, idx, speed_range)
        log(f"  iter {it}: beta {beta:.3f}, collecting {a.steps} steps")
        buf = distill.collect(env, teacher, driver, a.steps, beta,
                              distill.DemoBuffer(range_max=a.range_max),
                              v_max=a.v_max, range_max=a.range_max, log=log,
                              rng=np.random.default_rng(a.seed * 1000 + it)).finalize()
        bufs.append(buf); bufs = bufs[-a.keep_iters:]
        # `TinyLidarNet/train.py:135` scales the speed label by the DATA's own min and max
        # (`min_speed` is hard-coded 0 at `:66`). Their dataset is fixed; a DAgger aggregate grows
        # and old iterations are dropped, so the max is carried as a running maximum rather than
        # recomputed -- otherwise dropping the iteration that contained the fastest label would
        # silently rescale every target, and a label above the current max would be asked of a tanh
        # that cannot reach it. The alternative, pinning the scale to the suite's 9.0 m/s cap, is
        # not taken: their rule is data-derived and this is their rule.
        prev_hi = speed_range[1] if it else 0.0
        hi = max(prev_hi, buf.speed_range()[1])
        speed_range = (0.0, hi)
        if hi > prev_hi + 1e-9:
            log(f"    label speed range {speed_range[0]:.2f}..{speed_range[1]:.2f} m/s "
                f"(TinyLidarNet's min/max scaling, train.py:135, running max over the aggregate)")
        if it == 0:
            if a.dump_iter0:
                p = os.path.join(a.out, f"{a.name}_iter0.npz")
                buf.save(p); log(f"    iteration 0 saved to {p} "
                                 f"(teacher-driven, so identical across architectures at this seed)")
        n = sum(len(b) for b in bufs)
        if a.arch == "tinylidarnet":
            loss = distill.train_tinylidarnet(model, bufs, epochs=a.epochs, device=device, log=log,
                                              driver_beams_idx=idx, speed_range=speed_range,
                                              opt=opt)
        else:
            loss = distill.train_end2race(model, bufs, epochs=a.epochs, device=device, log=log,
                                          driver_beams_idx=idx, seq_len=a.seq_len, opt=opt,
                                          sched=sched)
        history.append({"iter": it, "beta": beta, "samples": n, "loss": loss,
                        "minutes": (time.time() - t_start) / 60})
        log(f"  iter {it}: {n} samples, loss {loss:.5f}, "
            f"{(time.time() - t_start) / 60:.1f} min elapsed")
        _save(a, model, idx, speed_range, proto, deviation, history, it)
    log(f"{a.name}: done in {(time.time() - t_start) / 60:.1f} min")
    return 0


def _driver_for(a, model, device, idx, speed_range):
    """The student, wrapped in its own repo's preprocessing, ready to drive."""
    from f1sim.learn.baselines import end2race as e2r_mod
    from f1sim.learn.baselines import tinylidarnet as tln_mod
    from f1sim.learn.baselines.backends import TorchBackend
    from f1sim.learn.baselines.tinylidarnet_torch import TorchBackendForDriver

    if a.arch == "tinylidarnet":
        d = tln_mod.TinyLidarNet(TorchBackendForDriver(model, device=str(device)),
                                 skip_n=a.skip_n, speed_map="fitted",
                                 speed_range=speed_range,
                                 raw_beams=a.n_beams, raw_fov=a.fov)
    else:
        backend = _WrapModule(model, device)
        d = e2r_mod.End2Race(backend, raw_beams=a.n_beams, raw_fov=a.fov,
                             range_max=a.range_max, n_features=a.n_features,
                             control_rate=40.0)
    d.bind_scanner(n_beams=a.n_beams, fov=a.fov, range_max=a.range_max)
    return d


class _WrapModule:
    """A live `nn.Module` in the shape `backends.TorchBackend` presents, without reloading it."""

    def __init__(self, module, device):
        self.module = module
        self.device = torch.device(device)
        self.path = ""
        self.sha256 = None
        self.n_params = int(sum(p.numel() for p in module.parameters()))

    def describe(self) -> dict:
        return {"backend": "torch (in training)", "version": torch.__version__,
                "device": str(self.device), "threads": None, "path": "", "sha256": None,
                "n_params": self.n_params}


def _save(a, model, idx, speed_range, proto, deviation, history, it):
    meta = {"arch": a.arch, "name": a.name, "iter": it, "iters": a.iters,
            "n_beams": int(a.n_beams), "model_beams": int(idx.size), "skip_n": int(a.skip_n),
            "n_features": int(a.n_features), "fov": float(a.fov),
            "range_max": float(a.range_max), "v_max": float(a.v_max),
            "speed_range": list(speed_range), "collection": proto,
            "upstream_model_deviation": deviation,
            "hyperparameters": distill.HYPERPARAMETERS, "history": history}
    blob = {"state_dict": model.state_dict(), "meta": meta}
    for name in (f"{a.name}_it{it}.pt", f"{a.name}_final.pt"):
        torch.save(blob, os.path.join(a.out, name))
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


def cmd_budget(a) -> int:
    from f1sim.learn import baselines as bl
    from f1sim.learn.budget import baseline_step_ms
    d = bl.load(a.kind, a.weights)
    d.bind_scanner(n_beams=a.n_beams, fov=a.fov, range_max=a.range_max)
    r = baseline_step_ms(d, iters=a.iters, repeats=a.repeats)
    print(json.dumps(r, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="f1sim.learn.baselines", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("distill", help="DAgger a published architecture on our teacher")
    d.add_argument("--arch", required=True, choices=["tinylidarnet", "end2race"])
    d.add_argument("--name", required=True)
    d.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "f1sim_runs",
                                                 "baselines"))
    d.add_argument("--teacher", default="interactive", choices=["interactive", "raceline"])
    d.add_argument("--tracks", default="train")
    d.add_argument("--learners", type=int, default=86, help="races; cars = learners x race_size")
    d.add_argument("--iters", type=int, default=distill.D3_ENV["iters"])
    d.add_argument("--steps", type=int, default=distill.D3_ENV["steps"])
    d.add_argument("--beta0", type=float, default=distill.D3_ENV["beta0"])
    d.add_argument("--epochs", type=float, default=3.0, help="passes over the aggregate per iter")
    d.add_argument("--keep-iters", type=int, default=4)
    d.add_argument("--seed", type=int, default=distill.D3_ENV["seed"])
    d.add_argument("--speed-cap", type=float, default=distill.D3_ENV["speed_cap"])
    d.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    d.add_argument("--n-beams", type=int, default=1081)
    d.add_argument("--fov", type=float, default=4.71238898)
    d.add_argument("--range-max", type=float, default=10.0)
    d.add_argument("--v-max", type=float, default=10.0)
    d.add_argument("--skip-n", type=int, default=1, help="tinylidarnet: scans[::skip_n]")
    d.add_argument("--n-features", type=int, default=270,
                   help="end2race: beams it reads. 270 = one per degree over this car's 270 deg "
                        "window, root's declared deviation from their 360")
    d.add_argument("--hidden-scale", type=int, default=4)
    d.add_argument("--seq-len", type=int, default=80, help="end2race: their sequence length")
    d.add_argument("--repo", default="")
    d.add_argument("--dump-iter0", action="store_true",
                   help="save iteration 0 (teacher-driven, identical across architectures)")
    d.set_defaults(func=cmd_distill)

    b = sub.add_parser("budget", help="ms per control step, CPU, single thread")
    b.add_argument("--kind", required=True, choices=["tinylidarnet", "end2race"])
    b.add_argument("--weights", required=True)
    b.add_argument("--n-beams", type=int, default=1081)
    b.add_argument("--fov", type=float, default=4.71238898)
    b.add_argument("--range-max", type=float, default=10.0)
    b.add_argument("--iters", type=int, default=200)
    b.add_argument("--repeats", type=int, default=5)
    b.set_defaults(func=cmd_budget)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
