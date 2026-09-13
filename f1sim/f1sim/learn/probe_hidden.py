"""Does the actor's hidden state know where the other car will be? Ridge regression says.

`--aux-future` (`f1sim.learn.future`) trains the recurrent state to carry the nearest opponent's
relative state and the ego's own motion K control steps ahead. That the loss falls is evidence the
HEAD learned something; it is not evidence the STATE carries it in a form anything else could use,
and "a latent race state you plan through" is a claim about the state.

So: roll a checkpoint out with traffic and opponent events on, freeze the hidden states it produced,
and fit a *linear* read-out from `h_t` to the same privileged targets at `t + k` that the head is
scored on, with a held-out split. R^2 on the held-out rows is the number. Linear on purpose -- a
nonlinear probe measures the probe, and what is being asked is whether the information sits in the
state legibly enough to be read off.

The comparison the research note reports is three rows per target:

* the frozen original **warm-started** into the same recurrent architecture -- its GRU is at fresh
  init and its output projection is zero, so this is a random recurrent feature map driven by the
  original policy's embedding. `--k 0` on the same checkpoint is the second baseline: what a state
  trivially knows about the PRESENT, which six stacked LiDAR frames already nearly determine;
* a checkpoint trained without the head;
* one trained with it.

    python -m f1sim.learn.probe_hidden A=frozen.pt B=arm_a.pt C=arm_b.pt \\
        --tracks real:blackbox2022_1 --steps 400 --envs 48 --race-size 3 \\
        --opponent teacher --opp-events brake,stop,shift --opp-event-rate 1.0 \\
        --k 0,20 --out probe.json --md probe.md

Reading it: R^2 near 1 means the state determines the target linearly; R^2 near 0 means it does not,
and a NEGATIVE R^2 means the fit is worse on held-out rows than predicting their mean, which is what
an overfit read-out on a state that carries nothing looks like.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from ..gym_env import (EnvConfig, FUTURE_LABEL_DIM, FUTURE_LABEL_KEYS, FUTURE_PRESENT_INDEX)
from ..params import Config
from . import common
from . import grip_runtime
from . import opponent_config as opp_cfg
from .future import FUTURE_K, FUTURE_OPPONENT_KEYS, align_future_targets
from .memory import Hidden, memory_spec, reset_hidden
from .model import load_checkpoint, load_for_memory
from .obs import ScanAugment, flatten_obs


# ------------------------------------------------------------------ the states and the labels
@torch.no_grad()
def collect(env, model, steps: int, device, controller=None, seed: int = 0):
    """Roll the policy out and return (states, labels, boundary), one column per learner car.

    * `states` (steps, L, H): the actor's state at t -- the GRU's hidden state AFTER the step that
      produced the action for observation t. It is taken from `Actor.probe_state`, which returns
      `Actor.future_input`, which is the tensor the head is trained on: one function, so the probe
      cannot end up measuring a differently-timed state. For a checkpoint with no memory it is the
      trunk features instead, and the same argument applies one layer down.
    * `labels` (steps + 1, L, FUTURE_LABEL_DIM): `F1VecEnv.future_labels` for the state at t, for
      t = 0 .. steps. The extra row is what makes step `steps - k` the last one with a label.
    * `boundary` (steps, L): 1 where any car of that race ended its episode on that step.

    The hidden state is threaded through and cleared at episode boundaries, and the extra scan
    channels are advanced with it -- a probe run from a zero state every step would be measuring a
    different policy than the one the checkpoint is.
    """
    lid = env.learner_ids
    chan = list((model.meta.get("scan_channels") or {}).get("channels") or ())
    aug = (ScanAugment(chan, env.n_beams, env.B, device=device,
                       tau_s=float(model.meta["scan_channels"]["memory_tau_s"])) if chan else None)
    obs, _ = env.reset(seed=seed)
    if controller is not None:
        controller.begin(obs)
    if aug is not None:
        aug.reset()
    h = model.actor.initial_hidden(env.B, device=device)
    states, labels, boundary = [], [], []
    for _ in range(steps):
        scan, pro = flatten_obs(obs)
        if aug is not None:
            scan = aug(scan)
        if controller is not None:
            controller.pre_action(obs)
        act, state, h_next = model.actor.probe_state(scan, pro, None, h)
        states.append(state[lid].float().cpu())
        labels.append(env.future_labels()[lid].float().cpu())
        obs, _rew, term, trunc, _info = env.step(act.clamp(-1, 1))
        if controller is not None:
            controller.post_step(term, trunc)
        done = term | trunc
        boundary.append(env.race_boundary(done)[lid].float().cpu())
        h = reset_hidden(h_next, done)
        if aug is not None:
            aug.reset(done)
    labels.append(env.future_labels()[lid].float().cpu())
    return torch.stack(states), torch.stack(labels), torch.stack(boundary)


# ------------------------------------------------------------------ the read-out
def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge weights for standardised `x` with an intercept column appended, penalty off the bias."""
    d = x.shape[1]
    a = x.T @ x + lam * np.eye(d, dtype=x.dtype)
    a[-1, -1] -= lam                                     # the intercept is not shrunk
    return np.linalg.solve(a, x.T @ y)


def fit_probe(state: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray,
              lambdas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0), val_frac: float = 0.2) -> dict:
    """Held-out R^2 of a ridge read-out from `state` to one scalar `target`.

    `train` / `test` are boolean row masks and must come from a split of whole ENV COLUMNS (see
    `split_columns`): consecutive steps of one car are the same situation a few milliseconds apart,
    so a random row split reports how well the probe interpolates inside a trajectory it has already
    seen -- which is near 1 for almost any feature map.

    The penalty is chosen on a slice of the training rows and never on the test rows; the reported
    R^2 is therefore a number no part of the fitting procedure was allowed to look at.
    """
    n_tr, n_te = int(train.sum()), int(test.sum())
    if n_tr < state.shape[1] + 2 or n_te < 2:
        return {"r2": float("nan"), "n_train": n_tr, "n_test": n_te, "rmse": float("nan"),
                "target_std": float("nan"), "lambda": float("nan")}
    mu, sd = state[train].mean(0), state[train].std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    z = lambda rows: np.concatenate([(state[rows] - mu) / sd,
                                     np.ones((int(rows.sum()), 1), dtype=state.dtype)], 1)
    x_tr, y_tr = z(train), target[train]
    cut = max(state.shape[1] + 2, int((1.0 - val_frac) * x_tr.shape[0]))
    best, best_lam = -np.inf, lambdas[0]
    if cut < x_tr.shape[0]:
        for lam in lambdas:
            w = _ridge(x_tr[:cut], y_tr[:cut], lam)
            r = _r2(x_tr[cut:] @ w, y_tr[cut:])
            if r > best:
                best, best_lam = r, lam
    w = _ridge(x_tr, y_tr, best_lam)
    x_te, y_te = z(test), target[test]
    pred = x_te @ w
    return {"r2": float(_r2(pred, y_te)), "n_train": n_tr, "n_test": n_te,
            "rmse": float(np.sqrt(np.mean((pred - y_te) ** 2))), "target_std": float(y_te.std()),
            "lambda": float(best_lam)}


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    var = float(((true - true.mean()) ** 2).mean())
    if var <= 1e-12:
        return float("nan")                              # a constant target has no variance to explain
    return float(1.0 - ((pred - true) ** 2).mean() / var)


def split_columns(n_cols: int, test_frac: float = 0.25, seed: int = 0):
    """(train columns, test columns) as boolean masks over env columns."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_cols)
    n_test = max(1, int(round(test_frac * n_cols)))
    test = np.zeros(n_cols, dtype=bool)
    test[order[:n_test]] = True
    return ~test, test


def probe(states: torch.Tensor, labels: torch.Tensor, boundary: torch.Tensor, k: int,
          test_frac: float = 0.25, seed: int = 0) -> dict:
    """R^2 per target for one lookahead, from one rollout's frozen states.

    The alignment and the masks are `future.align_future_targets` -- the same function the trainer
    uses -- so the probe scores the head's target and not a near relative of it. On top of the
    alignment mask, the four opponent columns keep only the rows whose presence label at t + k is 1,
    exactly as `future.future_loss` weights them.
    """
    target, valid = align_future_targets(labels, boundary, k)
    T, L, _ = target.shape
    x = states.reshape(T * L, -1).numpy().astype(np.float64)
    y = target.reshape(T * L, -1).numpy().astype(np.float64)
    ok = valid.reshape(T * L).numpy() > 0
    col = np.tile(np.arange(L), T)                        # row-major (step, env), like the trainer
    tr_cols, te_cols = split_columns(L, test_frac, seed)
    present = y[:, FUTURE_PRESENT_INDEX] > 0.5
    out = {}
    for i, key in enumerate(FUTURE_LABEL_KEYS):
        rows = ok & present if key in FUTURE_OPPONENT_KEYS else ok
        out[key] = fit_probe(x, y[:, i], rows & tr_cols[col], rows & te_cols[col])
    return out


# ------------------------------------------------------------------ the checkpoints
def peek(path: str) -> tuple:
    """(meta, spec) without building anything: what the env has to be shaped like for this file.

    The probe has to construct the env BEFORE it can load the checkpoint into it (the loaders need
    the env's widths), and guessing `--scan-stack` for a checkpoint that recorded its own is how a
    probe silently measures a policy fed the wrong observation.
    """
    ck = torch.load(path, map_location="cpu")
    return dict(ck.get("meta") or {}), dict((ck.get("extra") or {}).get("spec") or {})


def load_probed(path: str, device, memory_hidden: int, scan_channels,
                spec_override: Optional[dict] = None):
    """(model, metadata, note). A checkpoint with no memory is warm-started into one.

    The frozen original is feedforward and has no hidden state to probe. Warm-starting it is not a
    distortion of the baseline: the GRU's output projection is zero, so the policy it drives with is
    bit-identical to the original's (`tests/test_memory_model.py`), and what the probe then reads is
    a random recurrent feature map fed by the original's own embedding -- which is precisely the
    "before any of this trained" row the comparison needs.
    """
    ck = torch.load(path, map_location="cpu")
    meta = dict(ck.get("meta") or {})
    if meta.get("memory"):
        model, extra = load_checkpoint(path, device, override=spec_override, allow_controller=True)
        note = "as trained"
    else:
        chan = {"channels": list(scan_channels)} if scan_channels else None
        model, extra, fresh = load_for_memory(path, device, memory_spec(hidden_size=memory_hidden),
                                              scan_channels=chan, override=spec_override,
                                              allow_controller=True)
        note = (f"warm-started into gru h={memory_hidden}"
                + (f" + channels {','.join(scan_channels)}" if scan_channels else "")
                + f" ({len(fresh)} fresh tensors, projection zero)")
    model.eval()
    return model, extra, note


def parse_checkpoints(items):
    """`label=path` or `path`, in order. The label is what the tables are indexed by."""
    out = []
    for item in items:
        label, _, path = item.partition("=")
        if not path:
            label, path = os.path.splitext(os.path.basename(item))[0], item
        if not os.path.isfile(path):
            raise SystemExit(f"checkpoint not found: {path}")
        out.append((label, path))
    return out


# ------------------------------------------------------------------ reporting
def markdown(rows, meta) -> str:
    """One table per lookahead: checkpoints down the side, targets across."""
    labels, ks = [], []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
        if r["k"] not in ks:
            ks.append(r["k"])
    by = {(r["label"], r["k"], r["target"]): r for r in rows}
    out = ["# Hidden-state probe: R^2 of a linear read-out from h_t to the target at t + k", ""]
    out.append(f"Rollout: {meta['steps']} steps x {meta['learners']} learner cars on "
               f"{meta['tracks']}, race size {meta['race_size']}, opponents {meta['opponent']}, "
               f"events {','.join(meta['events']) or 'off'}. Held out: {meta['test_frac']:.0%} of the env "
               f"columns (whole cars, never rows inside one trajectory).")
    out.append("")
    for k in ks:
        out.append(f"## k = {k} steps ({k * 0.025:.2f} s)")
        out.append("")
        out.append("| checkpoint | " + " | ".join(FUTURE_LABEL_KEYS) + " | rows (test) |")
        out.append("| --- |" + " --- |" * (len(FUTURE_LABEL_KEYS) + 1))
        for label in labels:
            cells, n_te = [], 0
            for t in FUTURE_LABEL_KEYS:
                r = by.get((label, k, t))
                cells.append("-" if r is None else f"{r['r2']:+.3f}")
                n_te = max(n_te, 0 if r is None else r["n_test"])
            out.append(f"| {label} | " + " | ".join(cells) + f" | {n_te} |")
        out.append("")
    out.append("Notes: `opp_*` are scored only where a car is inside `overtake_range` at t + k, the "
               "same rows the auxiliary loss weights; `opp_present` is a 0/1 target, so its R^2 is "
               "1 - MSE / var. A negative R^2 is a read-out that does worse than the held-out mean.")
    out.append("")
    for label, note in meta["checkpoints"]:
        out.append(f"- `{label}`: {note}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", nargs="+", metavar="[LABEL=]CKPT",
                    help="one or more checkpoints; each gets its own rollout, since a different "
                         "policy visits different situations")
    ap.add_argument("--k", default=f"0,{FUTURE_K}",
                    help="comma-separated lookaheads in control steps (40 Hz). 0 is the present")
    ap.add_argument("--tracks", default="real:blackbox2022_1")
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--steps", type=int, default=400, help="10 s per env at 40 Hz")
    ap.add_argument("--speed-cap", type=float, default=9.0)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--test-frac", type=float, default=0.25,
                    help="share of ENV COLUMNS held out. Columns, not rows: two consecutive steps "
                         "of one car are the same situation and a row split measures nothing")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--memory-hidden", type=int, default=128,
                    help="GRU width a checkpoint WITHOUT memory is warm-started into, so it has a "
                         "state to probe. Match the arms being compared against it")
    ap.add_argument("--scan-channels", default="", metavar="A,B",
                    help="extra scan channels for that warm start; match the arms")
    ap.add_argument("--action-mode", default="auto", choices=("auto", "direct", "plan"),
                    help="'auto' takes it from the checkpoint's act_dim, which is what it was "
                         "trained with; the env's observation shape comes from its recorded spec")
    ap.add_argument("--controller", default="legacy", choices=grip_runtime.ARMS)
    ap.add_argument("--sim-backend", default="compile", choices=("compile", "eager"))
    ap.add_argument("--out", default="", help="write the full result as JSON here")
    ap.add_argument("--md", default="", help="write the markdown table here")
    opp_cfg.add_arguments(ap)
    # Traffic is not optional here: the targets are an opponent's relative motion, and a solo
    # rollout would probe a state for information the situation does not contain. The contract's
    # comparison is run with events on, so those are the defaults too.
    ap.set_defaults(race_size=3, opponent="teacher", opp_speed=(0.6, 1.15), spawn_order="random",
                    opp_events="brake,stop,shift,defend,yield,line,oblivious", opp_event_rate=1.0,
                    opp_defend_prob=0.3, opp_yield_prob=0.2, opp_line_prob=0.3,
                    opp_oblivious_prob=0.1)
    a = ap.parse_args()
    device = torch.device(a.device)
    ks = [int(x) for x in str(a.k).split(",") if x.strip() != ""]
    checkpoints = parse_checkpoints(a.ckpt)
    channels = [c for c in a.scan_channels.split(",") if c]
    opp_cfg.validate(a)
    names = common.track_names(a.tracks)
    need_rl = opp_cfg.needs_racelines(a)
    tracks, rls = common.load_tracks(names, racelines=need_rl)
    cfg = Config(); cfg.sim.seed = a.seed
    cfg.sim.compile = (a.sim_backend == "compile")
    rows, notes, learners = [], [], 0
    env = ctrl = None
    env_key = None
    for label, path in checkpoints:
        # The rollouts are NOT shared -- a different policy visits different situations -- but the
        # env is, when the checkpoints agree on what it has to look like. Its shape comes from what
        # each checkpoint recorded, not from flags; a checkpoint that needs a different one gets a
        # new env, and every rollout resets from the same seed, so the draw of tracks, layouts and
        # opponents each of them starts from is identical.
        meta, spec = peek(path)
        mode = (a.action_mode if a.action_mode != "auto"
                else ("plan" if int(meta.get("act_dim", 2)) >= 5 else "direct"))
        key = (mode, int(spec.get("scan_stack", 6)), int(spec.get("scan_stride", 1)),
               int(spec.get("hist_len", 20)), int(spec.get("hist_stride", 2)))
        if key != env_key:
            env = common.make_env(tracks, a.envs, device,
                                  EnvConfig(speed_cap=a.speed_cap, resample_track_on_reset=True,
                                            action_mode=mode, max_steps=int(a.episode_s * 40),
                                            scan_stack=key[1], scan_stride=key[2],
                                            hist_len=key[3], hist_stride=key[4],
                                            **opp_cfg.env_kwargs(a)),
                                  cfg=cfg, seed=a.seed, rls=rls)
            env.sim.warmup()
            ctrl = grip_runtime.ControllerRuntime(env, a.controller, None, device=device)
            ctrl.install()
            env_key = key
        model, extra, note = load_probed(path, device, a.memory_hidden, channels)
        learners = int(env.learner_ids.numel())
        notes.append((label, note + f" | {os.path.basename(path)} | action mode {mode}"
                      + (" | future head trained" if model.meta.get("future_head") else
                         " | no future head")))
        print(f"[{label}] {note}; rolling out {a.steps} steps x {int(env.learner_ids.numel())} "
              f"learner cars ...", flush=True)
        states, labels, boundary = collect(env, model, a.steps, device, controller=ctrl, seed=a.seed)
        for k in ks:
            for target, res in probe(states, labels, boundary, k, a.test_frac, a.seed).items():
                rows.append({"label": label, "ckpt": path, "k": k, "target": target,
                             "source": "memory" if model.actor.has_memory else "trunk",
                             "has_future_head": bool(model.meta.get("future_head")), **res})
            got = {r["target"]: r["r2"] for r in rows if r["label"] == label and r["k"] == k}
            print(f"  k={k:3d}: " + "  ".join(f"{t} {v:+.3f}" for t, v in got.items()), flush=True)
        del model, states, labels, boundary
        if device.type == "cuda":
            torch.cuda.empty_cache()
    meta = {"steps": a.steps, "envs": a.envs, "learners": learners,
            "tracks": a.tracks, "race_size": a.race_size, "opponent": a.opponent,
            "events": list(a.opp_events), "test_frac": a.test_frac, "seed": a.seed, "k": ks,
            "checkpoints": notes, "targets": list(FUTURE_LABEL_KEYS)}
    text = markdown(rows, meta)
    print()
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)
        print("wrote", a.out)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)
        print("wrote", a.md)


if __name__ == "__main__":
    main()
