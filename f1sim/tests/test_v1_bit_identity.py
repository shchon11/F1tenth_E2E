"""The v1 actor, critic and observation stay bit-identical while v2 is built beside them.

The redesign (docs/research/architecture-redesign-2026-10-13.md, 6장 '잃는 것') keeps the old
loader alive on purpose: the league's seats, the baseline evaluations and the car path all still
run v1 checkpoints. So every v2 addition is selected by a version flag, and this file is the proof
that the flag's default changes nothing:

* the archived best policy (`iccas_bootcamp_safest.pt`) gives, for fixed inputs, exactly the
  outputs it gave on the commit the v2 work branched from -- actor step, every auxiliary head, the
  critic, `act`/`evaluate` and the export wrapper;
* a v1 environment (the default `EnvConfig`) emits exactly the observation, privileged vector and
  reward it did, step for step.

Both references were written by `python tests/test_v1_bit_identity.py --write` on that base commit
(feat/v2 8b6c8b9, before any v2 code existed) and are compared with `torch.equal` / sha256 -- not a
tolerance: "bit-identical" is the claim. Everything runs on one CPU thread, because a reduction's
chunking follows the thread count and the claim is about the code, not about the scheduler.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAFEST = os.path.join(ROOT, "checkpoints", "iccas_bootcamp_safest.pt")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIX_MODEL = os.path.join(DATA, "v1_safest_reference.npz")
FIX_ENV = os.path.join(DATA, "v1_env_obs_digest.json")
SEED = 20261014
ENV_STEPS = 12


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:24]


def _model_inputs(meta: dict, batch: int = 3):
    """Deterministic inputs shaped for `meta`: a corridor-like scan (not uniform noise: the min-pool
    and the sector profile see structure), proprio, dial, both hidden states, the critic's vector."""
    g = torch.Generator().manual_seed(SEED)
    n, k = int(meta["n_beams"]), int(meta["n_stack"])
    ramp = torch.linspace(0.15, 1.0, n)
    scan = (ramp[None, None, :] * (0.6 + 0.4 * torch.rand(batch, k, 1, generator=g))
            + 0.02 * torch.rand(batch, k, n, generator=g)).clamp(0, 1)
    pro = 0.3 * torch.randn(batch, int(meta["proprio_dim"]), generator=g)
    c = torch.tensor([[-0.8], [0.0], [0.4]])[:batch]
    ha = 0.5 * torch.randn(1, batch, 128, generator=g)
    hc = 0.5 * torch.randn(1, batch, 128, generator=g)
    priv = 0.5 * torch.randn(batch, int(meta["priv_dim"]), generator=g)
    act = (0.7 * torch.randn(batch, int(meta["act_dim"]), generator=g)).clamp(-1, 1)
    dial = 1.0 + 0.25 * c
    return dict(scan=scan, pro=pro, c=c, ha=ha, hc=hc, priv=priv, act=act, dial=dial)


@torch.no_grad()
def _model_outputs(model, x) -> dict:
    from f1sim.learn.conditioning import CondSpec
    from f1sim.learn.export import RecurrentActorOnly
    from f1sim.learn.memory import Hidden
    a = model.actor
    out = {}
    mu, hn = a.step(x["scan"], x["pro"], x["c"], x["ha"])
    out["actor.step.mu"], out["actor.step.h"] = mu, hn
    mu2, grip, opp, fut, mot, fl, hn2 = a.step_all(x["scan"], x["pro"], x["c"], x["ha"])
    out.update({"step_all.mu": mu2, "step_all.grip": grip, "step_all.opp": opp,
                "step_all.future": fut, "step_all.h": hn2})
    v, hcn = model.critic.step(x["scan"], x["pro"], x["priv"], x["hc"])
    out["critic.v"], out["critic.h"] = v, hcn
    act, logp, hid = model.act(x["scan"], x["pro"], deterministic=True, c=x["c"],
                               h=Hidden(x["ha"], x["hc"]))
    out["act.action"], out["act.logp"] = act, logp
    lp, ent, v2, _d, _h = model.evaluate(x["scan"], x["pro"], x["priv"], x["act"], c=x["c"],
                                         h=Hidden(x["ha"], x["hc"]))
    out["evaluate.logp"], out["evaluate.entropy"], out["evaluate.v"] = lp, ent, v2
    wrap = RecurrentActorOnly(a, CondSpec.from_meta(model.meta.get("cond"))).eval()
    ea, eh = wrap(x["scan"], x["pro"], x["ha"], x["dial"])
    out["export.action"], out["export.h"] = ea, eh
    out["log_std"] = a.log_std.detach().clone()
    return {k: v.detach().clone() for k, v in out.items()}


def _safest():
    from f1sim.learn.model import load_checkpoint
    m, _ = load_checkpoint(SAFEST, "cpu")
    return m.eval()


def _env_digest() -> dict:
    """A v1 env (the default `obs_version`), a two-car race, fixed actions: per step, the sha256 of
    every observation key, of `privileged()`, of `info["priv"]` and of the reward."""
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    tracks, _rls = common.load_tracks(["gen:competition:0"], racelines=True)
    cfg = Config(); cfg.sim.compile = False
    ec = EnvConfig(action_mode="plan", race_size=2, opponent="policy", hist_len=20, hist_stride=2,
                   scan_stack=6, compile_tracker=False, collision_mode="soft")
    env = common.make_env(tracks, 4, "cpu", ec, cfg=cfg, seed=7)
    obs, info = env.reset(seed=7)
    rows = [{"obs": {k: _digest(v) for k, v in sorted(obs.items())}, "priv": _digest(info["priv"]),
             "privileged": _digest(env.privileged(env.last_result))}]
    t = torch.arange(env.B, dtype=torch.float32)
    for s in range(ENV_STEPS):
        a = torch.zeros(env.B, env.act_dim)
        a[:, 0] = 0.3 * torch.sin(0.4 * s + t)                      # curvature knots
        a[:, -2:] = -0.2 + 0.1 * torch.cos(0.3 * s + t)[:, None]     # the two linear speeds
        obs, rew, _te, _tr, info = env.step(a)
        rows.append({"obs": {k: _digest(v) for k, v in sorted(obs.items())},
                     "priv": _digest(info["priv"]), "reward": _digest(rew),
                     "privileged": _digest(env.privileged(env.last_result)),
                     "keys": sorted(obs.keys()),
                     "shapes": {k: list(v.shape) for k, v in sorted(obs.items())}})
    return {"steps": rows}


@pytest.fixture
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


@pytest.mark.skipif(not os.path.isfile(SAFEST), reason="the archived checkpoint is not present")
def test_safest_outputs_are_bit_identical(one_thread):
    import numpy as np
    with np.load(FIX_MODEL) as z:
        digests = json.loads(str(z["input_digests"]))
        want = {k[4:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("out/")}
    m = _safest()
    x = _model_inputs(m.meta)
    assert {k: _digest(v) for k, v in x.items()} == digests, \
        "the generated inputs changed (torch's generator?), so the comparison below would be void"
    got = _model_outputs(m, x)
    assert sorted(got) == sorted(want)
    bad = {k: float((got[k].float() - want[k].float()).abs().max())
           for k in got if not torch.equal(got[k], want[k])}
    assert not bad, f"v1 outputs moved (max |diff| per output): {bad}"


@pytest.mark.slow
def test_v1_env_observation_is_bit_identical(one_thread):
    with open(FIX_ENV) as f:
        ref = json.load(f)
    got = _env_digest()
    for i, (g, r) in enumerate(zip(got["steps"], ref["steps"])):
        assert g == r, f"step {i}: the v1 observation / privileged vector / reward moved"
    assert len(got["steps"]) == len(ref["steps"])


def _write(force: bool):
    torch.set_num_threads(1)
    for p in (FIX_MODEL, FIX_ENV):
        if os.path.exists(p) and not force:
            raise SystemExit(f"{p} exists; it is the reference from the base commit. --force to overwrite")
    m = _safest()
    x = _model_inputs(m.meta)
    import numpy as np
    np.savez(FIX_MODEL, input_digests=np.array(json.dumps({k: _digest(v) for k, v in x.items()})),
             **{f"out/{k}": v.numpy() for k, v in _model_outputs(m, x).items()})
    with open(FIX_ENV, "w") as f:
        json.dump(_env_digest(), f, indent=0)
    print("wrote", FIX_MODEL, FIX_ENV)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "f1sim"))
    if "--write" in sys.argv:
        _write("--force" in sys.argv)
