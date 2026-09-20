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

## `--current`: the present, not the future

`--current` asks the narrower question first (`docs/research/motion-memory-2026-09-14.md`, E1): can
the state produce the nearest opponent's **present** relative state -- (dx, dy, dv_x, dv_y) in the
ego frame -- at all? It is `--k 0` with the opponent columns named after what they are, and it is
the question worth asking first because the two halves of it have different answers by
construction:

* **(dx, dy) is in one scan.** A car a few metres away is a handful of short beams; where it is
  does not need any memory, and a read-out that cannot recover it says the probe or the
  representation is broken rather than anything about motion.
* **(dv_x, dv_y) is not.** A single range image contains no velocity. Recovering it requires
  relating two instants -- the frame stack, the recurrent state, or the aligned residual channel
  -- so it is the column where "does this architecture build frame-to-frame correspondence?"
  actually gets tested.

Three flags make the ladder of representations that answer it comparable:

* `--stack-mode repeat` feeds the newest frame in every slot of the stack. The network, the
  weights and the proprio are unchanged; the only thing removed is the 150 ms of temporal
  information the stack carries. That is the **memory-off floor** for dv, and it is a floor
  measured on this policy rather than argued from first principles.
* `--probe-state trunk` probes a feedforward checkpoint as it is, reading the trunk features the
  action comes from, instead of warm-starting it into a GRU. That is the **frame-stack** row.
* the default (`auto`) warm-starts a feedforward checkpoint into a GRU, and probes a recurrent
  checkpoint as trained. Those are the **GRU** rows.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from ..gym_env import (EnvConfig, FUTURE_LABEL_KEYS, FUTURE_PRESENT_INDEX,
                       PRIV_OPP_DIST_SCALE)
from ..params import Config
from . import common
from . import grip_runtime
from . import opponent_config as opp_cfg
from .future import FUTURE_K, FUTURE_OPPONENT_KEYS, align_future_targets
from .memory import memory_spec, reset_hidden
from .model import load_checkpoint, load_for_memory
from .obs import ScanAugment, flatten_obs


# ------------------------------------------------------------------ the states and the labels
def flatten_stack(scan: torch.Tensor) -> torch.Tensor:
    """The newest frame in every slot of the stack: the same observation with its motion removed.

    `flatten_obs` hands the policy (B, k, N) with frame 0 the newest. Repeating frame 0 keeps the
    shape, the weights and every other input identical and deletes exactly one thing -- the 150 ms
    of apparent motion the k frames carry between them. It is how the `memory off` row of the E1
    table is produced without training a second network: what that row measures is what this
    policy's representation can still say about the opponent when nothing in its input relates two
    instants.

    It is NOT a policy anyone would deploy: the actor drives on an input off its training
    distribution, so the rollout it produces is its own. That is already true of every row of the
    table (a different policy visits different situations) and is why the probe never shares a
    rollout between checkpoints.
    """
    if scan.dim() != 3:
        raise ValueError(f"scan must be (B, k, N), got {tuple(scan.shape)}")
    return scan[:, :1].expand_as(scan).contiguous()


@torch.no_grad()
def collect(env, model, steps: int, device, controller=None, seed: int = 0,
            stack_mode: str = "full"):
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

    `stack_mode="repeat"` replaces the stack by its newest frame repeated (`flatten_stack`), which
    is the E1 table's memory-off row. It is applied BEFORE the channel augmenter, so a channel that
    is itself a memory would still carry one -- which is why `main` refuses the combination.
    """
    if stack_mode not in ("full", "repeat"):
        raise ValueError(f"stack_mode must be 'full' or 'repeat', got {stack_mode!r}")
    lid = env.learner_ids
    cfg = dict(model.meta.get("scan_channels") or {})
    chan = list(cfg.get("channels") or ())
    aug = (ScanAugment(chan, env.n_beams, env.B, device=device,
                       tau_s=float(cfg["memory_tau_s"]), aligned=cfg.get("aligned"),
                       floor=cfg.get("floor"))
           if chan else None)
    obs, _ = env.reset(seed=seed)
    if controller is not None:
        controller.begin(obs)
    if aug is not None:
        aug.reset()
    h = model.actor.initial_hidden(env.B, device=device)
    states, labels, boundary = [], [], []
    for _ in range(steps):
        scan, pro = flatten_obs(obs)
        if stack_mode == "repeat":
            scan = flatten_stack(scan)
        if aug is not None:
            scan = aug(scan, pro)
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


LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def fit_readout(state: np.ndarray, target: np.ndarray, train: np.ndarray,
                lambdas=LAMBDAS, val_frac: float = 0.2):
    """(weights, mean, std, penalty) of a ridge read-out fitted on `train` only.

    `train` must come from a split of whole ENV COLUMNS (see `split_columns`): consecutive steps of
    one car are the same situation a few milliseconds apart, so a random row split reports how well
    the probe interpolates inside a trajectory it has already seen -- which is near 1 for almost any
    feature map.

    The penalty is chosen on a slice of the training rows and never on the test rows, so every
    number scored with this read-out is one no part of the fitting procedure was allowed to see.
    Returns `None` when there are not enough rows to fit at all.
    """
    if int(train.sum()) < state.shape[1] + 2:
        return None
    mu, sd = state[train].mean(0), state[train].std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    x_tr = np.concatenate([(state[train] - mu) / sd,
                           np.ones((int(train.sum()), 1), dtype=state.dtype)], 1)
    y_tr = target[train]
    cut = max(state.shape[1] + 2, int((1.0 - val_frac) * x_tr.shape[0]))
    best, best_lam = -np.inf, lambdas[0]
    if cut < x_tr.shape[0]:
        for lam in lambdas:
            w = _ridge(x_tr[:cut], y_tr[:cut], lam)
            r = _r2(x_tr[cut:] @ w, y_tr[cut:])
            if r > best:
                best, best_lam = r, lam
    return _ridge(x_tr, y_tr, best_lam), mu, sd, float(best_lam)


def score_readout(state: np.ndarray, target: np.ndarray, rows: np.ndarray, fit, scale: float = 1.0,
                  min_rows: int = 20) -> dict:
    """R^2 AND MAE of a fitted read-out on `rows`.

    Both, because they answer different questions and the contract asks for both. R^2 is scale-free
    and says how much of the variation the read-out explains -- but it divides by the variance of
    exactly these rows, so on a bin where every opponent is doing the same thing it is small however
    good the prediction is. MAE is in the target's own unit (`scale` takes it out of the label's
    normalisation and into metres or m/s) and says how wrong the answer is, which is the number a
    reader can weigh against a car's own size.
    """
    n = int(rows.sum())
    out = {"n": n, "r2": float("nan"), "mae": float("nan"), "target_std": float("nan")}
    if fit is None or n < min_rows:
        return out
    w, mu, sd, _lam = fit
    x = np.concatenate([(state[rows] - mu) / sd, np.ones((n, 1), dtype=state.dtype)], 1)
    y = target[rows]
    pred = x @ w
    out.update(r2=float(_r2(pred, y)), mae=float(np.abs(pred - y).mean() * scale),
               target_std=float(y.std() * scale))
    return out


def fit_probe(state: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray,
              lambdas=LAMBDAS, val_frac: float = 0.2) -> dict:
    """Held-out R^2 and MAE of a ridge read-out from `state` to one scalar `target`."""
    fit = fit_readout(state, target, train, lambdas, val_frac)
    res = score_readout(state, target, test, fit, min_rows=2)
    return {"r2": res["r2"], "mae": res["mae"], "n_train": int(train.sum()),
            "n_test": res["n"], "target_std": res["target_std"],
            "lambda": float("nan") if fit is None else fit[3]}


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


#: Distance bins for the E1 table, in metres, and what they are for: a handful of very close
#: opponents span many beams and dominate any pooled number, so a read-out that is good only when
#: the other car is about to be hit would look like a read-out that works. The edges are the
#: addendum's.
DIST_BINS = (("near", 0.0, 2.0), ("mid", 2.0, 5.0), ("far", 5.0, float("inf")))

#: What one unit of each label column is, so MAE can be reported in metres and m/s rather than in
#: the normalisation the critic happens to want. `gym_env` owns the normalisers; these are them.
LABEL_SCALE = {"opp_lon": PRIV_OPP_DIST_SCALE, "opp_lat": PRIV_OPP_DIST_SCALE,
               "opp_vlon": PRIV_OPP_DIST_SCALE, "opp_vlat": PRIV_OPP_DIST_SCALE,
               "ego_speed": 10.0, "ego_yaw_rate": 5.0, "opp_present": 1.0}
LABEL_UNIT = {"opp_lon": "m", "opp_lat": "m", "opp_vlon": "m/s", "opp_vlat": "m/s",
              "ego_speed": "m/s", "ego_yaw_rate": "rad/s", "opp_present": ""}


def probe(states: torch.Tensor, labels: torch.Tensor, boundary: torch.Tensor, k: int,
          test_frac: float = 0.25, seed: int = 0, splits: int = 1,
          dist_bins=DIST_BINS) -> dict:
    """R^2 and MAE per target for one lookahead, from one rollout's frozen states.

    The alignment and the masks are `future.align_future_targets` -- the same function the trainer
    uses -- so the probe scores the head's target and not a near relative of it.

    **Presence-conditioned.** The four opponent columns keep only the rows whose presence label at
    t + k is 1 -- there is no relative position to a car that is not in range, and scoring the
    read-out on the zeros those rows carry would measure how well it predicts "no car". The share
    of rows that drops out is reported as `excluded_frac`, because a number computed on half the
    rollout should say so.

    **Binned by distance.** Each read-out is fitted once, on the same training rows as before, and
    then scored on the test rows overall AND inside each of `dist_bins` (the distance to the nearest
    opponent at t + k). One read-out, three regimes: a read-out that only works at two metres is a
    different claim from one that works at eight, and a pooled number cannot tell them apart.

    `splits` repeats the whole fit over that many different draws of which cars are held out, and
    reports the mean and the spread. One split is one number with an unknown error bar, and measured
    on this task the error bar is the story: on a 400-step, 32-car rollout the opponent columns move
    by more between two draws of the held-out cars than between two checkpoints. `r2` stays the
    headline (the mean), with `r2_std`, `r2_min`, `r2_max` beside it.
    """
    target, valid = align_future_targets(labels, boundary, k)
    T, L, _ = target.shape
    x = states.reshape(T * L, -1).numpy().astype(np.float64)
    y = target.reshape(T * L, -1).numpy().astype(np.float64)
    ok = valid.reshape(T * L).numpy() > 0
    col = np.tile(np.arange(L), T)                        # row-major (step, env), like the trainer
    present = y[:, FUTURE_PRESENT_INDEX] > 0.5
    # The nearest opponent's distance at t + k, in metres, from the same two label columns the
    # read-out is scored on -- so the bin a row lands in and the target it is scored against cannot
    # come from different instants.
    dist = PRIV_OPP_DIST_SCALE * np.hypot(y[:, 0], y[:, 1])
    excluded = float(1.0 - (ok & present).sum() / max(1, ok.sum()))
    out = {}
    for i, key in enumerate(FUTURE_LABEL_KEYS):
        opp = key in FUTURE_OPPONENT_KEYS
        rows = ok & present if opp else ok
        scale = LABEL_SCALE[key]
        runs, bins = [], {name: [] for name, _lo, _hi in dist_bins}
        for j in range(max(1, int(splits))):
            tr_cols, te_cols = split_columns(L, test_frac, seed + j)
            tr, te = rows & tr_cols[col], rows & te_cols[col]
            fit = fit_readout(x, y[:, i], tr)
            res = score_readout(x, y[:, i], te, fit, scale, min_rows=2)
            runs.append({"r2": res["r2"], "mae": res["mae"], "n_train": int(tr.sum()),
                         "n_test": res["n"], "target_std": res["target_std"],
                         "lambda": float("nan") if fit is None else fit[3]})
            for name, lo, hi in dist_bins:
                sel = te & (dist >= lo) & (dist < hi) if opp else te
                bins[name].append(score_readout(x, y[:, i], sel, fit, scale))
        agg = lambda vals: (float(np.mean(vals)) if len(vals) else float("nan"),
                            float(np.std(vals)) if len(vals) else float("nan"))
        fin = lambda field, src: [float(r[field]) for r in src if np.isfinite(r[field])]
        r2 = np.array([r["r2"] for r in runs], dtype=float)
        finite = r2[np.isfinite(r2)]
        mae = np.array([r["mae"] for r in runs], dtype=float)
        # Every draw's own R^2, not only their summary: the table pools EIGHT draws with TWO
        # independent rollout seeds into one +- spread, and a mean of two means with two standard
        # deviations cannot be pooled into the spread over sixteen numbers. Keeping the draws costs
        # eight floats per cell and makes the pooling arithmetic rather than a guess. Their
        # denominators go with them: R^2 is 1 - MSE / var over the TEST rows, so a draw whose
        # held-out cars barely moved divides by a variance near zero and returns a large negative
        # number that is about the draw and not about the representation -- one draw in sixteen read
        # -3.5 on `ego_yaw_rate` here while the other fifteen read +0.7.
        out[key] = {**runs[0], "splits": len(runs), "unit": LABEL_UNIT[key],
                    "presence_conditioned": bool(opp), "excluded_frac": excluded if opp else 0.0,
                    "r2_draws": [float(v) for v in r2], "mae_draws": [float(v) for v in mae],
                    "target_std_draws": [float(r["target_std"]) for r in runs],
                    "r2": float(finite.mean()) if finite.size else float("nan"),
                    "r2_std": float(finite.std()) if finite.size else float("nan"),
                    "r2_min": float(finite.min()) if finite.size else float("nan"),
                    "r2_max": float(finite.max()) if finite.size else float("nan"),
                    "mae": agg(fin("mae", runs))[0], "mae_std": agg(fin("mae", runs))[1],
                    "bins": {name: {"r2": agg(fin("r2", v))[0], "r2_std": agg(fin("r2", v))[1],
                                    "mae": agg(fin("mae", v))[0], "mae_std": agg(fin("mae", v))[1],
                                    "n": float(np.mean([r["n"] for r in v])),
                                    "r2_draws": [float(r["r2"]) for r in v],
                                    "mae_draws": [float(r["mae"]) for r in v]}
                             for name, v in bins.items()}}
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
                spec_override: Optional[dict] = None, probe_state: str = "auto"):
    """(model, metadata, note). A checkpoint with no memory is warm-started into one.

    The frozen original is feedforward and has no hidden state to probe. Warm-starting it is not a
    distortion of the baseline: the GRU's output projection is zero, so the policy it drives with is
    bit-identical to the original's (`tests/test_memory_model.py`), and what the probe then reads is
    a random recurrent feature map fed by the original's own embedding -- which is precisely the
    "before any of this trained" row the comparison needs.

    `probe_state="trunk"` skips that warm start and probes the feedforward checkpoint as it is:
    `Actor.future_input` then returns the trunk features the action is taken from. That is the E1
    table's frame-stack row -- what the six stacked frames alone put in the representation, with no
    recurrence of any kind, random or trained, in front of it. It is refused on a checkpoint that
    carries memory, whose trunk features are not what `probe_state` returns and would have to be
    read from a second, differently-timed tensor.
    """
    if probe_state not in ("auto", "trunk"):
        raise ValueError(f"probe_state must be 'auto' or 'trunk', got {probe_state!r}")
    ck = torch.load(path, map_location="cpu")
    meta = dict(ck.get("meta") or {})
    if probe_state == "trunk" and meta.get("memory"):
        raise SystemExit(
            f"--probe-state trunk on {os.path.basename(path)}, which carries memory "
            f"({meta['memory']}): its recurrent state is what `Actor.probe_state` returns, and the "
            f"trunk features of a recurrent actor are a different tensor at a different time. Probe "
            f"a feedforward checkpoint with `trunk`, or this one with `auto`.")
    if meta.get("memory"):
        model, extra = load_checkpoint(path, device, override=spec_override, allow_controller=True)
        note = "as trained"
    elif probe_state == "trunk":
        model, extra = load_checkpoint(path, device, override=spec_override, allow_controller=True)
        note = "feedforward, trunk features (no warm start)"
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
#: What the four opponent columns of `FUTURE_LABEL_KEYS` are, when they are read at k = 0. The
#: stored names say which label row they came from; these say what the number means, and the E1
#: table is indexed by them. Nothing but the heading changes -- the column is the same column.
CURRENT_NAMES = {"opp_lon": "dx", "opp_lat": "dy", "opp_vlon": "dv_x", "opp_vlat": "dv_y"}


def markdown(rows, meta) -> str:
    """One table per lookahead: checkpoints down the side, targets across."""
    current = bool(meta.get("current"))
    head = (lambda t: CURRENT_NAMES.get(t, t)) if current else (lambda t: t)
    labels, ks = [], []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
        if r["k"] not in ks:
            ks.append(r["k"])
    by = {(r["label"], r["k"], r["target"]): r for r in rows}
    title = ("# Current-state probe: R^2 of a linear read-out from h_t to the nearest opponent's "
             "present relative state" if current else
             "# Hidden-state probe: R^2 of a linear read-out from h_t to the target at t + k")
    out = [title, ""]
    out.append(f"Rollout: {meta['steps']} steps x {meta['learners']} learner cars on "
               f"{meta['tracks']}, race size {meta['race_size']}, opponents {meta['opponent']}, "
               f"events {','.join(meta['events']) or 'off'}. Held out: {meta['test_frac']:.0%} of the env "
               f"columns (whole cars, never rows inside one trajectory), averaged over "
               f"{meta['splits']} draws of which cars those are (± is the spread over the draws).")
    out.append("")
    for k in ks:
        out.append("## the present (k = 0)" if current and k == 0
                   else f"## k = {k} steps ({k * 0.025:.2f} s)")
        out.append("")
        for metric, fmt in (("r2", "{:+.3f}"), ("mae", "{:.3f}")):
            out.append(f"### {'R^2' if metric == 'r2' else 'MAE (the target unit)'}")
            out.append("")
            out.append("| checkpoint | " + " | ".join(head(t) for t in FUTURE_LABEL_KEYS)
                       + " | rows (test) |")
            out.append("| --- |" + " --- |" * (len(FUTURE_LABEL_KEYS) + 1))
            for label in labels:
                cells, n_te = [], 0
                for t in FUTURE_LABEL_KEYS:
                    r = by.get((label, k, t))
                    if r is None:
                        cells.append("-")
                    else:
                        sd = r.get(metric + "_std")
                        cells.append(fmt.format(r[metric])
                                     + (f" ±{sd:.3f}" if sd is not None and r.get("splits", 1) > 1
                                        else ""))
                    n_te = max(n_te, 0 if r is None else r["n_test"])
                out.append(f"| {label} | " + " | ".join(cells) + f" | {n_te} |")
            out.append("")
        out.append("### By distance to the nearest opponent (R^2 / MAE, one read-out scored in "
                   "three regimes)")
        out.append("")
        bins = list(DIST_BINS)
        out.append("| checkpoint | target | " + " | ".join(f"{n} ({lo:g}-{hi:g} m)"
                                                           for n, lo, hi in bins) + " |")
        out.append("| --- | --- |" + " --- |" * len(bins))
        for label in labels:
            for t in FUTURE_OPPONENT_KEYS:
                r = by.get((label, k, t))
                if r is None or not r.get("bins"):
                    continue
                cells = [f"{r['bins'][n]['r2']:+.3f} / {r['bins'][n]['mae']:.2f} "
                         f"(n {r['bins'][n]['n']:.0f})" for n, _lo, _hi in bins]
                out.append(f"| {label} | {head(t)} | " + " | ".join(cells) + " |")
        out.append("")
    out.append("Notes: `opp_*` are scored only where a car is inside `overtake_range` at t + k, the "
               "same rows the auxiliary loss weights; `opp_present` is a 0/1 target, so its R^2 is "
               "1 - MSE / var. A negative R^2 is a read-out that does worse than the held-out mean.")
    if current:
        out.append("")
        out.append("`dx`/`dy` are the nearest opponent's position in the ego body frame now and "
                   "`dv_x`/`dv_y` its velocity relative to the ego in that frame, both divided by "
                   "`PRIV_OPP_DIST_SCALE` = 5 m. Position is readable from ONE scan and velocity is "
                   "not, so `dv_*` is the column that measures temporal correspondence and `dx`/`dy` "
                   "is the control that says the read-out works at all.")
    ex = [r["excluded_frac"] for r in rows if r["target"] in FUTURE_OPPONENT_KEYS]
    if ex:
        out.append(f"Presence-conditioned: the four opponent columns are scored only on the rows "
                   f"where a car is inside `overtake_range`, which excludes "
                   f"**{float(np.mean(ex)):.1%}** of the otherwise-valid rows. MAE is in metres for "
                   f"`{head('opp_lon')}`/`{head('opp_lat')}` and m/s for `{head('opp_vlon')}`/"
                   f"`{head('opp_vlat')}`.")
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
    ap.add_argument("--current", action="store_true",
                    help="the E1 measurement: the nearest opponent's PRESENT relative state "
                         "(dx, dy, dv_x, dv_y in the ego frame). Forces --k 0 and names the four "
                         "opponent columns after what they are")
    ap.add_argument("--probe-state", default="auto", choices=("auto", "trunk"),
                    help="'auto' warm-starts a feedforward checkpoint into a GRU so it has a state "
                         "to probe; 'trunk' probes it as it is, reading the trunk features the "
                         "action comes from (the frame-stack row of the E1 table)")
    ap.add_argument("--stack-mode", default="full", choices=("full", "repeat"),
                    help="'repeat' feeds the newest frame in every slot of the scan stack, which "
                         "removes the 150 ms of temporal information the stack carries and nothing "
                         "else (the memory-off row of the E1 table)")
    ap.add_argument("--tracks", default="real:blackbox2022_1")
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--steps", type=int, default=400, help="10 s per env at 40 Hz")
    ap.add_argument("--speed-cap", type=float, default=9.0)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--splits", type=int, default=8,
                    help="how many different draws of the held-out cars to average the R^2 over. "
                         "One draw is one number with an unknown error bar, and on this task the "
                         "error bar is larger than the difference between checkpoints")
    ap.add_argument("--save-states", default="",
                    help="write each rollout's frozen states, labels and boundary flags here as a "
                         ".pt, so the fit can be redone without paying for the rollout again")
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
    if a.current:
        if ks != [0] and ap.get_default("k") != a.k:
            raise SystemExit(f"--current is the present and --k {a.k} is not: --current sets k = 0 "
                             f"and there is nothing it could mean together with a lookahead. Drop "
                             f"one of the two.")
        ks = [0]
    channels_for_warm_start = [c for c in a.scan_channels.split(",") if c]
    if a.stack_mode == "repeat":
        # `repeat` is a claim -- "this representation relates no two instants" -- and a decayed
        # occupancy channel or a recurrent state would falsify it silently while the flag went on
        # saying it. Both are refused rather than subtracted.
        if a.probe_state != "trunk":
            raise SystemExit("--stack-mode repeat needs --probe-state trunk: a GRU in front of the "
                             "flattened stack relates two instants again, which is the one thing "
                             "this arm is built to remove.")
        if channels_for_warm_start:
            raise SystemExit(f"--stack-mode repeat with --scan-channels {channels_for_warm_start}: "
                             f"the occupancy channel is itself a memory over steps, so the arm "
                             f"would not be memory-off. Drop the channels.")
    checkpoints = parse_checkpoints(a.ckpt)
    channels = channels_for_warm_start
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
        model, extra, note = load_probed(path, device, a.memory_hidden, channels,
                                         probe_state=a.probe_state)
        if a.stack_mode == "repeat":
            if model.actor.has_memory or (model.meta.get("scan_channels") or {}).get("channels"):
                raise SystemExit(f"--stack-mode repeat on {os.path.basename(path)}, which carries "
                                 f"memory or a scan channel: see --stack-mode's help.")
            note += " | stack flattened (newest frame repeated)"
        learners = int(env.learner_ids.numel())
        notes.append((label, note + f" | {os.path.basename(path)} | action mode {mode}"
                      + (" | future head trained" if model.meta.get("future_head") else
                         " | no future head")))
        print(f"[{label}] {note}; rolling out {a.steps} steps x {int(env.learner_ids.numel())} "
              f"learner cars ...", flush=True)
        states, labels, boundary = collect(env, model, a.steps, device, controller=ctrl,
                                           seed=a.seed, stack_mode=a.stack_mode)
        if a.save_states:
            os.makedirs(a.save_states, exist_ok=True)
            torch.save({"states": states, "labels": labels, "boundary": boundary, "ckpt": path,
                        "label": label, "note": note},
                       os.path.join(a.save_states, f"{label}.pt"))
        for k in ks:
            for target, res in probe(states, labels, boundary, k, a.test_frac, a.seed,
                                     splits=a.splits).items():
                rows.append({"label": label, "ckpt": path, "k": k, "target": target,
                             "source": "memory" if model.actor.has_memory else "trunk",
                             "stack_mode": a.stack_mode,
                             "has_future_head": bool(model.meta.get("future_head")), **res})
            got = {r["target"]: (r["r2"], r["r2_std"]) for r in rows
                   if r["label"] == label and r["k"] == k}
            print(f"  k={k:3d}: " + "  ".join(f"{t} {v:+.3f}+-{sd:.3f}" for t, (v, sd) in got.items()),
                  flush=True)
        del model, states, labels, boundary
        if device.type == "cuda":
            torch.cuda.empty_cache()
    meta = {"steps": a.steps, "envs": a.envs, "learners": learners,
            "tracks": a.tracks, "race_size": a.race_size, "opponent": a.opponent,
            "events": list(a.opp_events), "test_frac": a.test_frac, "splits": a.splits,
            "seed": a.seed, "k": ks, "current": bool(a.current),
            "probe_state": a.probe_state, "stack_mode": a.stack_mode,
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
