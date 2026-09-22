"""Soft actor-critic on PPO's own environment and model (`ppo.py --algo sac`).

Why an off-policy learner here at all (docs/research/architecture-rationale-2026-09-22.md, section
3): a PPO update of s911's recipe saw 4096 learner steps -- 0.39 km -- and in them 3.4 wall and prop
contacts, and then threw the data away. Collisions are the thing the policy has to learn and the
thing it sees least. A replay buffer keeps every contact for as long as the buffer holds it, and
`contact_frac` of every batch is drawn from the half second before one.

What is *not* new, on purpose:

* **The environment, the model and the conditioning are PPO's.** `ppo.main` builds all of it --
  tracks, opponents, obstacles, the dial, the CUDA graphs -- and hands it here instead of entering
  its own loop, so a comparison between the two learners is a comparison between two learners.
* **The policy is `model.actor`, unchanged**: the same mean head, the same state-independent
  `log_std`. A SAC checkpoint is an `ActorCritic` checkpoint (`sac_u*.pt`), and the console, the
  evaluation and the ROS node load it exactly as they load a PPO one. The action the env executes is
  `clamp(u, -1, 1)` of the Gaussian sample `u`, which is what PPO does too.

What is new:

* **Twin Q functions** built from the PPO value critic (`QNet`): the same stem and heads, with the
  action appended to the privileged input and its weights **zero** -- so at the start Q(s, a) is
  V(s), the actor's gradient is exactly zero, and the policy cannot be pulled anywhere until the
  critic has learned what an action does. `critic_warmup` updates of the critic alone come first.
* **A replay buffer on the device**, time-major per learner row (`Replay`). The scan stack is not
  stored -- six frames of 1081 beams is 13 kB a transition -- but rebuilt from the newest frame of
  each step, repeating an episode's first frame the way the env fills its history at a reset.
  Frames are kept as uint16 (0.15 mm at 10 m range). The last observation of an episode, which the
  next row of the buffer does not hold, is kept aside.
* **An anchor to the starting actor**, `anchor * ||mu(s) - mu_0(s)||^2`, decaying to zero over
  `anchor_decay` steps: the actor starts from a policy that already races, and the first thousand
  updates against a critic that is still learning are the ones that can wreck it.
"""
from __future__ import annotations

import copy
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .obs import flatten_obs


# ------------------------------------------------------------------------------------------ critics
class QNet(nn.Module):
    """Q(s, a) grown out of a trained V(s) critic (`model.Critic`).

    The action joins the privileged vector at the proprio layer's input (`Critic.pro`), whose new
    columns start at zero: Q = V at initialisation, bit for bit, and dQ/da = 0. The stem, the
    proprio layer's existing columns and the MLP are the value critic's, so the features the critic
    learned over a PPO run are where this one starts."""

    def __init__(self, critic: nn.Module, act_dim: int):
        super().__init__()
        if getattr(critic, "memory", None) is not None:
            raise ValueError("--algo sac: a recurrent critic is not supported (train without --memory)")
        if getattr(critic.stem, "extra_channels", 0):
            raise ValueError("--algo sac: extra scan channels are not supported")
        self.c = copy.deepcopy(critic)
        lin = self.c.pro[0]
        wide = nn.Linear(lin.in_features + act_dim, lin.out_features).to(lin.weight.device, lin.weight.dtype)
        with torch.no_grad():
            wide.weight.zero_()
            wide.weight[:, :lin.in_features] = lin.weight
            wide.bias.copy_(lin.bias)
        self.c.pro[0] = wide

    def forward(self, scan, pro, priv, act):
        c = self.c
        if c._adapt is not None:
            priv = c._adapt(priv)
        x = torch.cat([c.stem(scan), c.pro(torch.cat([pro, priv, act], 1))], 1)
        return c.mlp(x).squeeze(1)


# ------------------------------------------------------------------------------------------- replay
class Replay:
    """Learner transitions, time-major per row: slot (t, j) is row j's step t, t modulo `T`.

    Stored per slot: the newest scan frame (uint16), the proprio vector (fp16), the critic's
    privileged vector (fp32, conditioning included), the actor's conditioning, the executed action,
    the reward, `term` (a real ending -- no bootstrap), `last` (the episode ended here, by `term` or
    by truncation), and `age`, the steps since the episode's first observation, which is what the
    stack is rebuilt from. An episode's final observation -- the one after `last` -- is kept in a
    small side ring (`fin_*`), indexed by `fin_slot`.
    """

    def __init__(self, T: int, n: int, n_stack: int, n_beams: int, pro_dim: int, priv_dim: int,
                 cond_dim: int, act_dim: int, device, n_final: Optional[int] = None):
        self.T, self.n, self.k, self.N = int(T), int(n), int(n_stack), int(n_beams)
        dev = device
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        self.scan = z(T, n, n_beams, dt=torch.int16)            # uint16 bits in an int16 carrier
        self.pro = z(T, n, pro_dim, dt=torch.float16)
        self.priv = z(T, n, priv_dim)
        self.cond = z(T, n, max(cond_dim, 1))
        self.act = z(T, n, act_dim)
        self.rew = z(T, n)
        self.term = z(T, n, dt=torch.bool)
        self.last = z(T, n, dt=torch.bool)
        self.age = z(T, n, dt=torch.int32)
        self.near = z(T, n, dt=torch.bool)                      # within `window` steps before a contact onset
        self.fin_slot = torch.full((T, n), -1, dtype=torch.int32, device=dev)
        F_ = int(n_final or max(4 * n, T * n // 200))
        self.F = F_
        self.fin_scan = z(F_, n_stack, n_beams, dt=torch.int16)
        self.fin_pro = z(F_, pro_dim, dt=torch.float16)
        self.fin_priv = z(F_, priv_dim)
        self.fin_cond = z(F_, max(cond_dim, 1))
        self.fin_ptr = 0
        self.t = 0                                              # steps written so far
        self.cond_dim = cond_dim
        self._age_now = torch.zeros(n, dtype=torch.int32, device=dev)

    # -- quantisation: scan in [0, 1], 16 bits
    @staticmethod
    def _q(x):
        return (x.clamp(0.0, 1.0) * 65535.0).round().to(torch.int32).to(torch.int16)

    @staticmethod
    def _dq(q):
        return (q.to(torch.int32) & 0xFFFF).to(torch.float32) / 65535.0

    @property
    def size(self) -> int:
        return min(self.t, self.T) * self.n

    def add(self, scan_newest, pro, priv, cond, act, rew, term, last, final=None) -> None:
        """One step for every learner row. `final`: (rows j, scan stack, pro, priv, cond) of the
        observations that ended an episode at this step, the ones the next slot does not hold."""
        i = self.t % self.T
        self.scan[i] = self._q(scan_newest)
        self.pro[i] = pro.to(torch.float16)
        self.priv[i] = priv
        if self.cond_dim:
            self.cond[i] = cond
        self.act[i] = act
        self.rew[i] = rew
        self.term[i] = term
        self.last[i] = last
        self.age[i] = self._age_now
        self.near[i] = False
        self.fin_slot[i] = -1
        if final is not None and final[0].numel():
            j, f_scan, f_pro, f_priv, f_cond = final
            m = j.numel()
            at = (self.fin_ptr + torch.arange(m, device=j.device)) % self.F
            self.fin_ptr = (self.fin_ptr + m) % self.F
            self.fin_scan[at] = self._q(f_scan)
            self.fin_pro[at] = f_pro.to(torch.float16)
            self.fin_priv[at] = f_priv
            if self.cond_dim:
                self.fin_cond[at] = f_cond
            self.fin_slot[i, j] = at.to(torch.int32)
        self._age_now = torch.where(last, torch.zeros_like(self._age_now), self._age_now + 1)
        self.t += 1

    def mark_contacts(self, onset: torch.Tensor, window: int) -> None:
        """Flag this row's last `window` steps as leading up to a contact that began just now."""
        if not bool(onset.any()) or self.t == 0:
            return
        i = (self.t - 1) % self.T
        span = min(window, self.t)
        rows = (i - torch.arange(span, device=onset.device)) % self.T
        self.near[rows] |= onset[None, :]

    def _stack(self, ti, j):
        """(B, k, N) scan stacks of slots (ti, j), rebuilt from the newest frames."""
        age = self.age[ti, j].to(torch.long)                                  # (B,)
        k = torch.arange(self.k, device=ti.device)                            # (k,)
        back = torch.minimum(k[None, :], age[:, None])                        # an episode's first frame repeats
        tt = (ti[:, None] - back) % self.T
        return self._dq(self.scan[tt, j[:, None]])                            # (B, k, N)

    def sample(self, B: int, contact_frac: float = 0.0, gen: Optional[torch.Generator] = None,
               n_step: int = 1, gamma: float = 0.99):
        """A batch of n-step transitions: s, a, the discounted return `ret` of the next `n_step`
        rewards (fewer where the episode ends first), `disc` -- the discount on the bootstrap, 0 after
        a real ending -- and the bootstrap state s' (the slot `n_step` on, or the episode's kept
        final observation where it ended by truncation inside the window).

        Why n-step at all: at gamma 0.997 a one-step target has to carry a lap's reward back through
        hundreds of bootstraps, and on s911's recipe the critic's loss climbed from 32 to 100 over
        the first 400 k transitions with the policy frozen. The newest `n_step` slots have no full
        window yet and are not drawn; neither, in a full ring, are the oldest `k`, whose stacks
        reach into overwritten frames. `contact_frac` of the batch comes from the flagged slots."""
        dev = self.scan.device
        n = max(1, int(n_step))
        filled = min(self.t, self.T)
        newest = (self.t - 1) % self.T
        full = filled == self.T
        lo_gap = self.k if full else 0                 # oldest slots whose stacks cannot be rebuilt
        span = (self.T - lo_gap - n - 1) if full else (filled - n - 1)
        if span < 1:
            raise RuntimeError(f"replay holds {filled} steps, too few for {n}-step windows")
        first = (newest + 1 + lo_gap) % self.T if full else 0

        def dist_from_first(t):
            return (t - first) % self.T

        n_c = int(round(B * contact_frac))
        idx_t, idx_j = [], []
        if n_c:
            cand = torch.nonzero(self.near[:filled]).T
            if cand.shape[1]:
                cand = cand[:, dist_from_first(cand[0]) < span]
            if cand.shape[1]:
                pick = torch.randint(cand.shape[1], (n_c,), device=dev, generator=gen)
                idx_t.append(cand[0, pick]); idx_j.append(cand[1, pick])
            else:
                n_c = 0
        u = B - n_c
        idx_t.append((first + torch.randint(span, (u,), device=dev, generator=gen)) % self.T)
        idx_j.append(torch.randint(self.n, (u,), device=dev, generator=gen))
        ti, j = torch.cat(idx_t), torch.cat(idx_j)

        ks = torch.arange(n, device=dev)
        seq = (ti[:, None] + ks[None]) % self.T                                 # (B, n)
        r = self.rew[seq, j[:, None]]
        lst = self.last[seq, j[:, None]]
        trm = self.term[seq, j[:, None]]
        ended = lst.any(1)
        K = torch.where(ended, lst.to(torch.uint8).argmax(1), torch.full_like(ti, n - 1))
        inc = (ks[None] <= K[:, None]).to(r.dtype)
        pw = gamma ** ks.to(r.dtype)
        ret = (r * pw[None] * inc).sum(1)
        ar = torch.arange(ti.numel(), device=dev)
        end_slot = (ti + K) % self.T
        fslot = self.fin_slot[end_slot, j].to(torch.long)
        term_end = ended & trm[ar, K]
        # a truncation whose final observation is no longer kept (its side ring wrapped) is not
        # bootstrapped from a wrong state: it is treated as an ending, which it nearly is
        lost = ended & ~term_end & (fslot < 0)
        disc = (gamma ** (K + 1).to(r.dtype)) * (~(term_end | lost)).to(r.dtype)

        ns = (ti + n) % self.T
        n_scan = self._stack(ns, j)
        n_pro = self.pro[ns, j].float()
        n_priv = self.priv[ns, j]
        n_cond = self.cond[ns, j]
        use_fin = ended & (fslot >= 0)
        if bool(use_fin.any()):
            sl = fslot.clamp(min=0)
            n_scan = torch.where(use_fin[:, None, None], self._dq(self.fin_scan[sl]), n_scan)
            n_pro = torch.where(use_fin[:, None], self.fin_pro[sl].float(), n_pro)
            n_priv = torch.where(use_fin[:, None], self.fin_priv[sl], n_priv)
            n_cond = torch.where(use_fin[:, None], self.fin_cond[sl], n_cond)
        return {"scan": self._stack(ti, j), "pro": self.pro[ti, j].float(), "priv": self.priv[ti, j],
                "cond": self.cond[ti, j], "act": self.act[ti, j], "ret": ret, "disc": disc,
                "n_scan": n_scan, "n_pro": n_pro, "n_priv": n_priv, "n_cond": n_cond,
                "ti": ti, "j": j}


# ------------------------------------------------------------------------------------------ hyper
@dataclass
class SACHyper:
    buffer: int = 500_000          # transitions (learner rows x steps)
    batch: int = 256
    updates_per_step: int = 2      # gradient steps per env step (one env step = one row of the buffer)
    # Transitions in the buffer before any update. The whole batch starts on the same step, so the
    # first thousands are nothing but starts -- contact penalties everywhere, a 16-step return of -10
    # on average against a value of ~55 -- and a critic fitted to them first learns the wrong thing.
    start: int = 100_000
    critic_warmup: int = 10_000    # updates of the critics alone before the actor moves
    actor_every: int = 2           # one actor (and alpha) step per this many critic steps
    tau: float = 0.005
    lr_actor: float = 1e-5
    lr_critic: float = 5e-5
    # The critic's learning rate ramps up from zero over this many updates. Without it, measured on
    # s911's recipe, Adam's first steps moved Q(s, a) from 61 to 26 in five updates against a target
    # of 41 -- every weight at once -- and the loss spent the next thousand updates in the hundreds.
    critic_lr_warmup: int = 2_000
    huber: float = 10.0             # the critic loss is Huber with this delta, not squared error
    lr_alpha: float = 1e-4
    alpha0: float = 0.01
    target_entropy: Optional[float] = None   # None: the starting policy's own entropy
    anchor: float = 5.0            # weight of ||mu - mu_0||^2 at the start
    anchor_decay: float = 2e6      # env steps (learner) over which it decays linearly to zero
    contact_frac: float = 0.25
    contact_window: int = 20       # steps (0.5 s) before a contact onset that are flagged
    n_step: int = 16               # rewards summed before bootstrapping (0.4 s at 40 Hz)
    # "awac": advantage-weighted regression onto the buffer's own actions -- the policy moves only
    # toward actions it actually took that the critic ranks above its average, so an error in Q
    # cannot be climbed the way a reparameterised gradient climbs it. "sac": the soft actor-critic
    # update, dQ/da through the Gaussian sample, with a learned temperature.
    actor_loss: str = "awac"
    awac_beta: float = 1.0         # temperature, in units of the batch's advantage std
    awac_samples: int = 2          # policy samples for the baseline V(s) = E_pi min Q(s, a)
    awac_wmax: float = 20.0


def _entropy(dist) -> torch.Tensor:
    return dist.entropy().sum(1)


def _polyak(dst: nn.Module, src: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_d, p_s in zip(dst.parameters(), src.parameters()):
            p_d.lerp_(p_s, tau)


# ------------------------------------------------------------------------------------------- train
def train(a, *, env, model, obs, lid, device, cond_dim: int, cond_mode: str, cond_spec, dial, dial_new,
          out: str, progress_log, run, spec, steps_base: int, t_start: float, save: Callable,
          hyper: SACHyper, gamma: float, log_every: int, save_every: int, amp: bool) -> None:
    """The SAC loop. Everything named in the signature is what `ppo.main` already built."""
    from . import conditioning as cond_mod

    B_env = env.B
    n = int(lid.numel())
    actor = model.actor
    act_dim = env.act_dim
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda")

    def condition(priv):
        """(c for the actor, privileged vector for the critic) -- `ppo.main.condition_now`, with
        `priv` passed rather than closed over."""
        if dial is None:
            c = cond_mod.make_condition(cond_mode, cond_spec, priv, env.priv_mu_index) if cond_dim else None
            return c, priv
        dial.redraw(dial_new); dial_new.zero_()
        d = dial.value()
        if a.grip_budget_penalty > 0:
            env.set_grip_budget(d)
        c = cond_mod.mu_to_c(d, cond_spec).to(priv.dtype)
        return c, torch.cat([priv, c], 1)

    priv = env.privileged(env.last_result)
    c_t, priv_c = condition(priv)
    scan, pro = flatten_obs(obs)
    k_stack, N_beams = scan.shape[1], scan.shape[2]
    if int(getattr(env.ecfg, "scan_stride", 1)) != 1:
        raise SystemExit("--algo sac rebuilds the scan stack from consecutive frames: --scan-stride 1 only")

    q1, q2 = QNet(model.critic, act_dim).to(device), QNet(model.critic, act_dim).to(device)
    q1_t, q2_t = copy.deepcopy(q1).requires_grad_(False), copy.deepcopy(q2).requires_grad_(False)
    actor0 = copy.deepcopy(actor).requires_grad_(False).eval()
    log_alpha = torch.tensor(math.log(hyper.alpha0), device=device, requires_grad=True)
    with torch.no_grad():
        d0 = actor.step_dist(scan, pro, c_t)[0]
        ent0 = float(_entropy(d0)[lid].mean())
    target_entropy = ent0 if hyper.target_entropy is None else float(hyper.target_entropy)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=hyper.lr_critic)
    opt_pi = torch.optim.Adam(actor.parameters(), lr=hyper.lr_actor)
    opt_a = torch.optim.Adam([log_alpha], lr=hyper.lr_alpha)

    T_cap = max(64, hyper.buffer // n)
    rb = Replay(T_cap, n, k_stack, N_beams, pro.shape[1], priv_c.shape[1], cond_dim, act_dim, device)
    gen = torch.Generator(device=device); gen.manual_seed(int(getattr(a, "seed", 0)) + 911)
    print(f"sac: {n} learner rows, buffer {T_cap} steps x {n} = {T_cap * n} transitions, batch {hyper.batch}, "
          f"{hyper.updates_per_step} update(s)/step, start {hyper.start}, critic warm-up {hyper.critic_warmup}, "
          f"target entropy {target_entropy:.2f} (start policy {ent0:.2f}), contact share {hyper.contact_frac}")

    ep_ret = torch.zeros(B_env, device=device)
    ep_stats = {"progress": [], "contacts": [], "lap_time": [], "steps": []}
    steps_done = 0; updates = 0; step_i = 0; log_i = 0
    total = int(a.total)
    rew_acc = torch.zeros((), device=device); rew_n = 0
    losses = {"q": [], "pi": [], "alpha": [], "ent": [], "q_mean": [], "anchor": [], "kl0": []}
    t_log = time.time(); steps_log0 = 0
    n_logs = max(1, total // (n * log_every))
    cap = a.cap0

    while steps_done < total:
        cap = a.cap0 + (a.cap1 - a.cap0) * min(1.0, steps_done / max(a.cap_steps, 1))
        env.set_speed_cap(cap)
        obs["speed_cap"] = (env.speed_cap / env.ecfg.v_max_policy)[:, None]
        scan, pro = flatten_obs(obs)
        with torch.no_grad(), ac:
            dist = actor.step_dist(scan, pro, c_t)[0]
            u = dist.sample()
        u = u.float()
        act = u.clamp(-1, 1)
        obs_n, rew, term, trunc, info = env.step(act)
        done = term | trunc
        # the transition, learner rows only
        final = None
        if "final" in info:
            f = info["final"]; ids = f["ids"]
            pos = torch.full((B_env,), -1, dtype=torch.long, device=device)
            pos[lid] = torch.arange(n, device=device)
            jj = pos[ids]; keep = jj >= 0
            if bool(keep.any()):
                f_scan, f_pro = flatten_obs(info["final_obs"])
                f_priv = info["final_priv"]
                f_c = None if c_t is None else c_t[ids]
                if dial is not None:
                    f_priv = torch.cat([f_priv, f_c], 1)
                final = (jj[keep], f_scan[keep], f_pro[keep], f_priv[keep],
                         None if f_c is None else f_c[keep])
                # a staggered first episode was cut short on purpose: not a sample of an episode
                m = env.learner[ids] & ~f["staggered"]
                ep_stats["progress"] += f["progress"][m].tolist()
                ep_stats["contacts"] += f["contacts"][m].float().tolist()
                ep_stats["steps"] += f["steps"][m].tolist()
        rb.add(scan[lid, 0], pro[lid], priv_c[lid], None if c_t is None else c_t[lid], act[lid],
               rew[lid], term[lid], done[lid], final)
        onset = info.get("contact_onset")
        if onset is not None:
            rb.mark_contacts(onset[lid], hyper.contact_window)
        ep_stats["lap_time"] += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
        rew_acc += rew[lid].sum(); rew_n += n
        # the next observation
        obs = obs_n
        priv = env.privileged(env.last_result)
        dial_new |= done
        c_t, priv_c = condition(priv)
        steps_done += n; step_i += 1

        # ---- updates
        if rb.size >= hyper.start:
            for _ in range(hyper.updates_per_step):
                bt = rb.sample(hyper.batch, hyper.contact_frac, gen, n_step=hyper.n_step, gamma=gamma)
                # the soft value's entropy bonus belongs to the soft actor only
                alpha = log_alpha.exp().detach() if hyper.actor_loss == "sac" else torch.zeros((), device=device)
                cond_b = bt["cond"] if cond_dim else None
                n_cond_b = bt["n_cond"] if cond_dim else None
                with torch.no_grad(), ac:
                    nd = actor.step_dist(bt["n_scan"], bt["n_pro"], n_cond_b)[0]
                    nu = nd.sample()
                    nlogp = nd.log_prob(nu).sum(1).float()
                    na = nu.clamp(-1, 1)
                    qn = torch.minimum(q1_t(bt["n_scan"], bt["n_pro"], bt["n_priv"], na),
                                       q2_t(bt["n_scan"], bt["n_pro"], bt["n_priv"], na)).float()
                    y = bt["ret"] + bt["disc"] * (qn - alpha * nlogp)
                with ac:
                    qa = q1(bt["scan"], bt["pro"], bt["priv"], bt["act"]).float()
                    qb = q2(bt["scan"], bt["pro"], bt["priv"], bt["act"]).float()
                loss_q = (F.huber_loss(qa, y, delta=hyper.huber) + F.huber_loss(qb, y, delta=hyper.huber))
                for g_ in opt_q.param_groups:
                    g_["lr"] = hyper.lr_critic * min(1.0, (updates + 1) / max(hyper.critic_lr_warmup, 1))
                opt_q.zero_grad(set_to_none=True); loss_q.backward()
                nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.0)
                opt_q.step()
                losses["q"].append(((qa - y) ** 2).mean().detach()); losses["q_mean"].append(qa.detach().mean())
                # Delayed policy updates (TD3's), and none at all while the critics are warming up:
                # the 2026-09-22 smoke gave the actor 200 critic steps' head start and it drove worse
                # for it (0 -> 2.0 contacts/km on the empty track) -- it was following a Q that
                # did not yet know what an action does.
                if updates >= hyper.critic_warmup and updates % max(hyper.actor_every, 1) == 0:
                    frac_anchor = max(0.0, 1.0 - steps_done / max(hyper.anchor_decay, 1.0))
                    if hyper.actor_loss == "awac":
                        with torch.no_grad(), ac:
                            minq = lambda act_: torch.minimum(q1(bt["scan"], bt["pro"], bt["priv"], act_),
                                                              q2(bt["scan"], bt["pro"], bt["priv"], act_)).float()
                            d_old = actor.step_dist(bt["scan"], bt["pro"], cond_b)[0]
                            v = torch.stack([minq(d_old.sample().clamp(-1, 1))
                                             for _ in range(max(1, hyper.awac_samples))]).mean(0)
                            adv = minq(bt["act"]) - v
                            w = torch.exp(adv / (hyper.awac_beta * adv.std().clamp_min(1e-3))).clamp(max=hyper.awac_wmax)
                            mu0 = actor0.step_dist(bt["scan"], bt["pro"], cond_b)[0].mean.float()
                        with ac:
                            d = actor.step_dist(bt["scan"], bt["pro"], cond_b)[0]
                            logp_buf = d.log_prob(bt["act"]).sum(1).float()
                            anchor = ((d.mean.float() - mu0) ** 2).sum(1).mean()
                        loss_pi = -(w * logp_buf).mean() / w.mean().clamp_min(1e-6) \
                            + hyper.anchor * frac_anchor * anchor
                        opt_pi.zero_grad(set_to_none=True); loss_pi.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        opt_pi.step()
                        losses["pi"].append(loss_pi.detach()); losses["anchor"].append(anchor.detach())
                        losses["ent"].append(d.entropy().sum(1).mean().detach().float())
                        losses["alpha"].append(w.mean().detach())          # logged as the mean weight
                    else:
                        with ac:
                            d = actor.step_dist(bt["scan"], bt["pro"], cond_b)[0]
                            uu = d.rsample()
                            logp = d.log_prob(uu).sum(1).float()
                            aa = uu.clamp(-1, 1)
                            qpi = torch.minimum(q1(bt["scan"], bt["pro"], bt["priv"], aa),
                                                q2(bt["scan"], bt["pro"], bt["priv"], aa)).float()
                            with torch.no_grad():
                                mu0 = actor0.step_dist(bt["scan"], bt["pro"], cond_b)[0].mean.float()
                            anchor = ((d.mean.float() - mu0) ** 2).sum(1).mean()
                        loss_pi = (alpha * logp - qpi).mean() + hyper.anchor * frac_anchor * anchor
                        opt_pi.zero_grad(set_to_none=True); loss_pi.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        opt_pi.step()
                        loss_a = -(log_alpha * (logp.detach() - target_entropy)).mean()
                        opt_a.zero_grad(set_to_none=True); loss_a.backward(); opt_a.step()
                        losses["pi"].append(loss_pi.detach()); losses["alpha"].append(log_alpha.detach().exp())
                        losses["ent"].append((-logp).detach().mean()); losses["anchor"].append(anchor.detach())
                _polyak(q1_t, q1, hyper.tau); _polyak(q2_t, q2, hyper.tau)
                updates += 1

        # ---- log (every `log_every` env steps, in PPO's progress schema)
        if step_i % log_every == 0:
            log_i += 1
            dt = time.time() - t_log
            sps = (steps_done - steps_log0) / max(dt, 1e-6)
            t_log = time.time(); steps_log0 = steps_done
            dist_m = float(np.sum(ep_stats["progress"])) if ep_stats["progress"] else 0.0
            with torch.no_grad():
                kl0 = torch.distributions.kl_divergence(
                    actor.step_dist(scan[lid], pro[lid], None if c_t is None else c_t[lid])[0],
                    actor0.step_dist(scan[lid], pro[lid], None if c_t is None else c_t[lid])[0]).sum(1).mean()
            mean = lambda xs: float(torch.stack(xs).float().mean()) if xs else float("nan")
            log = {"rollout/reward_per_step": float(rew_acc) / max(rew_n, 1),
                   "episode/collisions_per_km": (1000.0 * float(np.sum(ep_stats["contacts"])) / dist_m
                                                 if dist_m > 0 else float("nan")),
                   "episode/progress_m": float(np.mean(ep_stats["progress"])) if ep_stats["progress"] else float("nan"),
                   "episode/lap_time_s": float(np.mean(ep_stats["lap_time"])) if ep_stats["lap_time"] else float("nan"),
                   "loss/kl_ref": float(kl0), "time/env_steps_per_s": sps,
                   "sac/q_loss": mean(losses["q"]), "sac/q_mean": mean(losses["q_mean"]),
                   "sac/pi_loss": mean(losses["pi"]),
                   ("sac/awac_weight" if hyper.actor_loss == "awac" else "sac/alpha"): mean(losses["alpha"]),
                   "sac/entropy": mean(losses["ent"]), "sac/anchor": mean(losses["anchor"]),
                   "sac/updates": updates, "sac/buffer": rb.size,
                   "sac/contact_slots": int(rb.near.sum())}
            record = {"kind": "sac", "update": log_i, "total": n_logs, "steps": steps_base + steps_done,
                      "cap": cap, "rew_per_step": log["rollout/reward_per_step"],
                      "coll_per_km": log["episode/collisions_per_km"], "prog_m": log["episode/progress_m"],
                      "lap_s": log["episode/lap_time_s"], "gate": float("nan"), "tk": float("nan"),
                      "kl_ref": log["loss/kl_ref"], "sps": sps, "wall_s": time.time() - t_start}
            record.update({k_: v_ for k_, v_ in log.items() if k_ not in record})
            progress_log.write(record)
            run.log(log, step=steps_base + steps_done)
            print(f"sac {log_i}/{n_logs} steps {(steps_base + steps_done)/1e6:.2f}M | rew/step {log['rollout/reward_per_step']:.3f} "
                  f"coll {log['episode/collisions_per_km']:.1f}/km prog {log['episode/progress_m']:.0f} m "
                  f"lap {log['episode/lap_time_s']:.1f} s | q {log['sac/q_loss']:.3f} Q {log['sac/q_mean']:.1f} "
                  f"pi {log['sac/pi_loss']:.3f} {'w' if hyper.actor_loss == 'awac' else 'a'} "
                  f"{log.get('sac/awac_weight', log.get('sac/alpha', float('nan'))):.4f} H {log['sac/entropy']:.2f} "
                  f"kl0 {log['loss/kl_ref']:.3f} | buf {rb.size} upd {updates} | {sps:.0f} steps/s", flush=True)
            ep_stats = {k_: [] for k_ in ep_stats}
            rew_acc.zero_(); rew_n = 0
            losses = {k_: [] for k_ in losses}
            if log_i % save_every == 0:
                save(os.path.join(out, f"sac_u{log_i}.pt"), log_i, steps_done, cap)
                save(os.path.join(out, "sac_latest.pt"), log_i, steps_done, cap)
                torch.save({"q1": q1.state_dict(), "q2": q2.state_dict(), "q1_t": q1_t.state_dict(),
                            "q2_t": q2_t.state_dict(), "log_alpha": log_alpha.detach(),
                            "target_entropy": target_entropy, "updates": updates},
                           os.path.join(out, "sac_state_latest.pt"))
    save(os.path.join(out, "sac_final.pt"), log_i, steps_done, cap)
