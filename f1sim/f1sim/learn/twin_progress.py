"""The progress the mean action is worth, measured in a pair of twin worlds (`ppo --twin-progress B`, rl-audit 2026-10-28).

PPO cannot see it: one 25 ms action moves the car ~2 cm, buried in the return's noise. In the training world, 2 M samples
of policy gradient give t ~0 for the first curvature and the first speed on straights, where two worlds that share every
random number resolve the same effect from ~17 k (+0.22 +- 0.06 per 0.5 s; docs/research/rl-audit-2026-10-27, 9.16-9.17).

So a pair of envs is kept beside the training env, built by the same `make_env` call (`common.make_twin`): a master the
current policy drives with its mean action, and a twin that `twin_sync.sync` makes an exact copy of the master at every
branch (both step on the same global random numbers). Each learner row draws z ~ N(0, I) over the live action columns;
the master drives mean + delta*z and the twin mean - delta*z for `kp` steps, then each its own mean action until `h`
steps. dP = what the two collected in the DENSE terms of the reward -- everything but the hit charges (collision,
collision speed, car contact), which PPO prices from its own rollouts -- and

    g = dP z / (2 delta kp)      ~  d(dense reward per step) / d(mean action) at the branch's first state

(Gaussian smoothing, exact up to the shared future). `loss()` is the deterministic policy gradient -mean_i g_i . mu(s_i)
on those states, in reward units per step: divided by PPO's advantage spread it adds to PPO's normalised surrogate in
the same units, so the sum estimates the gradient of the whole objective. The master drives on from each branch's end,
so the states are the deployed policy's own."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

from .memory import reset_hidden
from .twin_sync import sync

#: reward components that are hit charges: PPO keeps those, the twins measure everything else
HIT_COMPONENTS = ("collision", "collision_speed", "car_contact")


def _rng_state():
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda


def _set_rng_state(s) -> None:
    torch.set_rng_state(s[0])
    if s[1] is not None:
        torch.cuda.set_rng_state_all(s[1])


class TwinProgress:
    """A master env and its twin beside the training env; `branch()` measures, `loss()` turns it into a gradient."""

    def __init__(self, master, twin, actor, *, cond: Callable[[], torch.Tensor], live: Sequence[int],
                 deltas: Sequence[float], h: int, kp: int, amp: bool = False, warmup: int = 200, seed: int = 0):
        if len(live) != len(deltas):
            raise ValueError(f"one delta per live action column {list(live)}, got {list(deltas)}")
        if not torch.equal(master.learner_ids, twin.learner_ids):
            raise ValueError("the twin env's learner rows differ from the master's")
        if not 0 < kp <= h:
            raise ValueError(f"need 0 < kp <= h, got kp {kp}, h {h}")
        self.M, self.T, self.actor = master, twin, actor
        self.cond, self.h, self.kp = cond, int(h), int(kp)
        self.dev = master.device
        self.lid = master.learner_ids
        self.B = int(self.lid.numel())
        self.live = torch.tensor(list(live), device=self.dev)
        self.delta = torch.tensor([float(d) for d in deltas], device=self.dev)
        self.ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(amp) and self.dev.type == "cuda")
        from ..gym_env import REWARD_COMPONENT_KEYS
        self._hit_idx = [REWARD_COMPONENT_KEYS.index(k) for k in HIT_COMPONENTS]
        self.fullM = torch.zeros(master.B, master.act_dim, device=self.dev)
        self.fullT = torch.zeros(twin.B, twin.act_dim, device=self.dev)
        s = _rng_state()
        self.oM, _ = master.reset(seed=int(seed))
        _set_rng_state(s)
        twin.reset(seed=int(seed))
        self.hM = actor.initial_hidden(self.B, device=self.dev)
        for _ in range(int(warmup)):
            # lockstep on the master's command: every lazily created piece of state comes to exist in both
            mu, hn = self._mean(self.oM, self.hM)
            self.fullM[self.lid] = mu
            (self.oM, _, te, tr, _), _ = self._step_both(self.fullM, self.fullM)
            self.hM = reset_hidden(hn, (te | tr)[self.lid])
        self.batch: Optional[Dict[str, torch.Tensor]] = None

    def _mean(self, o, h):
        c = self.cond().to(o["ego"].dtype)[self.lid]
        with torch.no_grad(), self.ac:
            d, hn = self.actor.step_dist(o["scan"][self.lid], o["ego"][self.lid], c, h)
        return d.mean.float(), hn.float()

    def _step_both(self, a_m, a_t):
        s = _rng_state()
        out_m = self.M.step(a_m.clamp(-1, 1))
        _set_rng_state(s)                        # the twin draws what the master drew
        out_t = self.T.step(a_t.clamp(-1, 1))
        return out_m, out_t

    def _dense(self, rew, info):
        comp = info["reward_components"]
        r = rew[self.lid].float()
        for k in HIT_COMPONENTS:
            r = r - comp[k][self.lid].float()
        return r, info["progress"][self.lid].float()

    def branch(self) -> Dict[str, float]:
        """One measurement: sync the twin, perturb, drive `h` steps. Stores the batch `loss()` reads."""
        was = self.actor.training
        self.actor.eval()
        sync(self.M, self.T)
        oT = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.oM.items()}
        c0 = self.cond().to(self.oM["ego"].dtype)[self.lid].clone()
        scan0, ego0 = self.oM["scan"][self.lid].clone(), self.oM["ego"][self.lid].clone()
        h0 = self.hM.clone()
        z = torch.randn(self.B, int(self.live.numel()), device=self.dev)
        pert = torch.zeros(self.B, self.M.act_dim, device=self.dev)
        pert[:, self.live] = z * self.delta
        hM, hT = self.hM.clone(), self.hM.clone()
        D = torch.zeros(2, self.B, device=self.dev); P = torch.zeros(2, self.B, device=self.dev)
        ended = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        for k in range(self.h):
            mu_m, hn_m = self._mean(self.oM, hM)
            mu_t, hn_t = self._mean(oT, hT)
            if k < self.kp:
                mu_m = mu_m + pert; mu_t = mu_t - pert
            self.fullM[self.lid] = mu_m; self.fullT[self.lid] = mu_t
            (self.oM, rM, teM, trM, iM), (oT, rT, teT, trT, iT) = self._step_both(self.fullM, self.fullT)
            for j, (r_, i_) in enumerate(((rM, iM), (rT, iT))):
                d_, p_ = self._dense(r_, i_)
                D[j] += d_; P[j] += p_
            dM, dT = (teM | trM)[self.lid], (teT | trT)[self.lid]
            ended |= dM | dT
            hM, hT = reset_hidden(hn_m, dM), reset_hidden(hn_t, dT)
        self.hM = hM                             # the master drives on from the branch's end
        ok = ~ended
        dP = (D[0] - D[1])
        on = self.delta > 0                      # a column held at delta 0 measures nothing
        g = torch.where(on[None, :], dP[:, None] * z / (2.0 * self.delta.clamp_min(1e-12)[None, :] * self.kp),
                        torch.zeros_like(z))
        self.batch = {"scan": scan0[ok], "ego": ego0[ok], "c": c0[ok], "h": h0[:, ok], "g": g[ok]}
        n = int(ok.sum())
        self.actor.train(was)
        return {"n": n, "dP_mean": float(dP[ok].mean()) if n else 0.0, "dP_sd": float(dP[ok].std()) if n > 1 else 0.0,
                "g_rms": float(g[ok].pow(2).mean().sqrt()) if n else 0.0,
                "master_progress_ms": float(P[0].sum() / (self.h * self.M.sim.control_dt) / self.B),
                "master_dense_per_step": float(D[0].mean() / self.h)}

    def loss(self) -> torch.Tensor:
        """-mean_i g_i . mu(s_i) on the last branch's states (reward units per step); 0 without a batch."""
        b = self.batch
        if b is None or b["g"].shape[0] == 0:
            return torch.zeros((), device=self.dev)
        with self.ac:
            d, _ = self.actor.step_dist(b["scan"], b["ego"], b["c"], b["h"])
        mu = d.mean.float()[:, self.live]
        return -(b["g"] * mu).sum(1).mean()
