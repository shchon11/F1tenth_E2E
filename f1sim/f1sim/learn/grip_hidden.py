"""Can a linear read-out of the recurrent state recover the friction the label was built from?

`RacelineTeacher.grip_bin` picks the teacher's speed profile from each env's true friction, so with
`--teacher-grip true` two identical observations can carry speed labels up to 1/0.45 ~ 2.2x apart.
The docstring's objection to that is old and was written for a feedforward student: the label
depends on something the student cannot see, so a regression can only fit its conditional mean.

A recurrent student with a second of proprio and IMU history is a different proposition. Friction is
not visible in one frame, but it is *inferable from how the car answered its own commands* -- which
is exactly what a hidden state is for. So the question this module asks is the narrow one that
decides whether the mu-dependent label is noise or a teaching signal:

    from the GRU's hidden state alone, how much of the true friction's variance does a LINEAR
    read-out explain?

Deliberately linear, and deliberately the same ridge, the same column split and the same held-out
discipline as `probe_hidden`: a probe that reports a number nobody else's probe can be compared with
is worth very little.

**Episodes are the sample unit, and this is the one thing that would otherwise make the number a
lie.** Friction is drawn once per episode and held, so consecutive steps of one car carry the *same*
target; a step-level split would let the probe see a value in training and be tested on the same
value a few milliseconds later, and would report near 1 for almost any feature map. So: the split is
over whole env columns (as `probe_hidden`), and the reported R^2 is over EPISODE-level predictions --
one row per episode, the mean of the read-out's per-step predictions within it, against that
episode's own friction. The per-step number is reported beside it, clearly labelled, because the gap
between the two is itself the measurement of how badly a step-level probe would have flattered.

    python -m f1sim.learn.grip_hidden CKPT --tracks train --envs 96 --steps 600
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..gym_env import EnvConfig
from ..params import Config
from . import common
from .probe_hidden import _r2, fit_probe, split_columns


@torch.no_grad()
def collect(env, model, steps: int, device, seed: int = 0):
    """(states (T, L, H), mu (T, L), episode id (T, L)) over the learner columns.

    The hidden state is `Actor.probe_state`'s -- the same tensor `probe_hidden` reads, so the two
    probes are answering questions about one object. It is threaded across steps and cleared at
    episode boundaries, and the extra scan channels are advanced with it: a probe run from a zero
    state every step would be measuring a policy the checkpoint is not.
    """
    from .memory import reset_hidden
    from .obs import ScanAugment, flatten_obs

    lid = env.learner_ids
    # The actor's hidden state is threaded as a raw tensor, exactly as `probe_hidden.collect` does:
    # `probe_state` takes the actor's own state, not the `Hidden` pair a PolicyRuntime carries.
    chan = list((model.meta.get("scan_channels") or {}).get("channels") or ())
    aug = (ScanAugment(chan, env.n_beams, env.B, device=device,
                       tau_s=float(model.meta["scan_channels"]["memory_tau_s"])) if chan else None)
    obs, _ = env.reset(seed=seed)
    if aug is not None:
        aug.reset()
    h = model.actor.initial_hidden(env.B, device=device)
    states, mus, eps = [], [], []
    # Which episode each car is on. Friction is redrawn when the car is, so this IS the target's id,
    # and it is what makes the episode the sample unit downstream.
    episode = torch.zeros(env.B, dtype=torch.long, device=device)
    for _ in range(steps):
        scan, pro = flatten_obs(obs)
        if aug is not None:
            scan = aug(scan)
        act, state, h_next = model.actor.probe_state(scan, pro, None, h)
        states.append(state[lid].float().cpu())
        # the quantity `RacelineTeacher.grip_bin` reads, per car, right now
        mus.append((env.sim.P["mu"] * env.sim.P["mu_f_scale"])[lid].float().cpu())
        eps.append(episode[lid].clone().cpu())
        obs, _r, term, trunc, _info = env.step(act.clamp(-1, 1))
        done = term | trunc
        episode = episode + done.long()
        h = reset_hidden(h_next, done)
        if aug is not None:
            aug.reset(done)
    return torch.stack(states), torch.stack(mus), torch.stack(eps)


def probe(states: torch.Tensor, mus: torch.Tensor, eps: torch.Tensor, seed: int = 0,
          test_frac: float = 0.25) -> dict:
    """Held-out R^2 of a linear read-out of friction, scored per episode and per step."""
    T, L, H = states.shape
    train_c, test_c = split_columns(L, test_frac, seed)
    col = np.repeat(np.arange(L)[None, :], T, 0).reshape(-1)
    x = states.reshape(T * L, H).numpy().astype(np.float64)
    y = mus.reshape(-1).numpy().astype(np.float64)
    ep = eps.reshape(-1).numpy()
    train, test = train_c[col], test_c[col]
    fit = fit_probe(x, y, train, test)

    # ... and the same read-out scored with the episode as the unit, which is what it is.
    mu_, sd = x[train].mean(0), x[train].std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    z = lambda rows: np.concatenate([(x[rows] - mu_) / sd, np.ones((int(rows.sum()), 1))], 1)
    from .probe_hidden import _ridge
    w = _ridge(z(train), y[train], fit["lambda"] if np.isfinite(fit["lambda"]) else 1.0)
    pred = z(test) @ w
    key = col[test] * (ep.max() + 1) + ep[test]
    uniq, inv = np.unique(key, return_inverse=True)
    n_ep = len(uniq)
    ep_pred = np.bincount(inv, weights=pred, minlength=n_ep) / np.bincount(inv, minlength=n_ep)
    ep_true = np.bincount(inv, weights=y[test], minlength=n_ep) / np.bincount(inv, minlength=n_ep)
    return {"r2_episode": _r2(ep_pred, ep_true), "n_episodes_test": int(n_ep),
            "r2_step": fit["r2"], "n_rows_test": fit["n_test"], "n_rows_train": fit["n_train"],
            "state_dim": int(H), "mu_std": float(y.std()), "lambda": fit["lambda"],
            "note": ("r2_episode is the number to quote: friction is drawn once per episode, so the "
                     "step-level figure counts one draw as many samples and flatters any feature map")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tracks", default="train")
    ap.add_argument("--envs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--race-size", type=int, default=1)
    ap.add_argument("--opponent", default="policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from .model import load_checkpoint

    names = common.track_names(a.tracks)
    need_rl = a.race_size > 1 and a.opponent == "teacher"
    tracks, rls = common.load_tracks(names, racelines=need_rl)
    model, extra = load_checkpoint(a.ckpt, a.device)
    model.eval()
    if not model.actor.has_memory:
        raise SystemExit(f"{a.ckpt} has no recurrent state; this probe reads one. "
                         f"`learn.grip_probe` is the observation-side question.")
    spec = (extra.get("spec") or {})
    cfg = Config()
    if a.eager:
        cfg.sim.compile = False
    ecfg = EnvConfig(action_mode="plan", speed_cap=spec.get("cap", 9.0) or 9.0,
                     race_size=a.race_size, opponent=a.opponent,
                     scan_stack=spec.get("scan_stack", 6), scan_stride=spec.get("scan_stride", 1),
                     hist_len=spec.get("hist_len", 20), hist_stride=spec.get("hist_stride", 2),
                     opp_token=spec.get("opp_token", ""), compile_tracker=not a.eager)
    env = common.make_env(tracks, a.envs, a.device, ecfg, cfg=cfg, seed=a.seed, rls=rls)
    env.sim.warmup()
    states, mus, eps = collect(env, model, a.steps, a.device, seed=a.seed)
    res = probe(states, mus, eps, seed=a.seed)
    res["checkpoint"] = a.ckpt
    res["config"] = {"tracks": names, "envs": a.envs, "steps": a.steps, "race_size": a.race_size}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
