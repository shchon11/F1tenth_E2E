"""(11-26, RQ6 (c) and RQ3 of `analysis/rq_plan_2026-11-26.md`) a held intent in a plan policy's action -- plan_rl --intent
K,H,EPS (default off; off, nothing changes).

Why: measured, the policy is a closed-loop controller that takes back ~73 % of any output deviation every 0.1 s, so output
noise -- white or held -- never makes the 1-2 s manoeuvre a blind box needs (another line +-0.4 m), and the gradient toward
"use the memory to choose the line early" is zero. An intent z in {0..K-1} is a slow, discrete part of the action:

  * the policy's high level pi_high(z | h) is a categorical head on the trunk's features (`Actor2.intent_head`);
  * z is drawn at an episode's start, after a restart and every H decisions, and held in between;
  * training draws it from the mixture (1 - EPS) softmax + EPS / K (an exploration floor: every intent keeps being tried).
    The mixture IS the policy PPO scores -- the behaviour policy and the scored one are the same distribution, so the PPO
    ratio needs no importance correction (the simpler of the two correct options; the floor stays in the ratio's
    denominator and numerator alike, and drive time takes the argmax, which is the mixture's argmax too);
  * z moves the plan through a per-intent vector added to the plan head's 20 pre-tanh outputs (`Actor2.intent_bias`,
    K x 20). This is the concatenation of a one-hot z to the plan head's input: the plan head is one linear layer, so any
    embedding that enters it (concatenated, or added to the trunk's output) is exactly such a vector. A hidden-size
    embedding added to the trunk output would make the intent's effect W_pi e_z -- tied to the plan head's weights, which
    start at gain 0.01 and move every step; the table is the intent's effect itself, so its start is set in metres and it
    moves only when PPO (first step) or the tail loss (the forecast) moves it. The critic reads z (`PlanObs2.intent_priv`:
    z one-hot while held, the share of the hold left) -- z is part of the state the value depends on, and a value blind to
    it would mix the intents' returns into every held decision's baseline;
  * (11-26, the coordinator's decision, plan_rl --intent-trunk, default on) the trunk reads z too: measured on scan31b2 u750
    with the vectors alone (exam3, each intent fixed), the trained closed loop undid 85-93 % of the push -- the outer
    intents at +-0.3 m drove -2 / +3 cm off its line (+-2 m: -15 / +37 cm, slower, wall touches). A vector the policy does
    not know about is output noise held still, and RQ3's mechanism takes it back; design (c) needs z to move the setpoint,
    so the control law itself is z-conditioned: z's one-hot enters the trunk's first layer (`Actor2.intent_in`, zero at the
    start -- the plan is the vectors' alone until PPO moves it). pi_high's logits and the speed estimate read the trunk run
    without z, the memory reads no z. `realized_effect` (plan_rl --intent-effect-every) logs what the outer intents make the
    car drive;
  * the start (`init_bias`): intent k's 1 s plan point lies linspace(-LAT, +LAT, K)[k] m to the side (+ left) of the plan
    the same outputs draw without it, for a fresh net (outputs ~0) at V_REF from straight steering. The shape is the same
    number on all 10 turn outputs: at speed a turn output is the step's steering angle as a share of what the tyres hold
    (`plan_steps.step_targets`: the servo's 0.1 s range is wider than the grip from ~2 m/s on), so this is a steady
    steering offset -- an arc that ends LAT to the side, about the same at 3-7 m/s (the share of the tyres' lateral
    acceleration, 1/2 c a_max t^2). Why an arc and not a lane change that ends parallel: the plan is drawn again every
    0.1 s and only its first step is driven, so whatever the shape, a held intent drives as a steady steering offset of its
    first step; the arc is the plan that shows what that offset does over 1 s if nothing counters it (the policy's own
    closed loop does). The speed outputs start at 0 (the user's goal is a different line at speed, not slowing down); PPO
    may move them. The tail's numbers (steps 2..10) are a forecast and the tail loss re-teaches them like the plan's own.

PPO: the action's log-probability is log pi_high(z | h) + log pi_low(a | h, z) at a redraw, log pi_low(a | h, z) at a held
decision (z is then state). The advantage at a redraw is the ordinary GAE advantage there -- it already holds the hold's
discounted TD errors, and with the critic reading z its first TD error already carries the value of the z drawn. Summing the
hold's advantages instead counts the later TD errors up to ~(1 - (gamma lambda)^H) / (1 - gamma lambda) times (H 15: ~10x the
first one's weight) -- a biased, higher-variance estimate; not used. An entropy bonus on softmax(logits) at the redraws
(plan_rl --intent-ent) keeps the learnt part from collapsing early (the floor keeps the sampling alive regardless)."""
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .. import plan_steps as PS

V_REF = 5.0               # [m/s] the start speed `init_bias` lays the intents out at (the approach speed before boxes)
ENV_FIX = "F1SIM_INTENT_FIX"
FREE = -1                 # F1SIM_INTENT_FIX=free: no intent at all -- the z-free plan (no vector, the trunk without z)


def parse(spec: str) -> Optional[Tuple[int, int, float]]:
    """'K,H,EPS' -> (K, H, EPS); '' -> None. K >= 2 intents, held H >= 1 decisions, drawn uniformly with probability EPS"""
    if not spec:
        return None
    try:
        k, h, e = spec.split(",")
        K, H, EPS = int(k), int(h), float(e)
    except ValueError:
        raise ValueError(f"--intent K,H,EPS (e.g. 4,15,0.1), not {spec!r}")
    if K < 2 or H < 1 or not 0.0 <= EPS < 1.0:
        raise ValueError(f"--intent K,H,EPS: K >= 2, H >= 1, 0 <= EPS < 1 (got {spec!r})")
    return K, H, EPS


def mix_probs(logits: torch.Tensor, eps: float) -> torch.Tensor:
    """(B, K) the high level's distribution: (1 - eps) softmax(logits) + eps / K -- what training draws from and PPO scores"""
    K = logits.shape[1]
    return (1.0 - eps) * torch.softmax(logits.float(), 1) + eps / K


def logp(logits: torch.Tensor, z: torch.Tensor, due: torch.Tensor, eps: float) -> torch.Tensor:
    """(B,) log pi_high(z | h) where `due` (a redraw), 0 where z is held"""
    lp = mix_probs(logits, eps).gather(1, z[:, None].long())[:, 0].clamp_min(1e-12).log()
    return torch.where(due, lp, torch.zeros_like(lp))


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """(B,) the entropy of softmax(logits) (the learnt part, without the floor)"""
    lg = torch.log_softmax(logits.float(), 1)
    return -(lg.exp() * lg).sum(1)


def fixed() -> Optional[int]:
    """F1SIM_INTENT_FIX=<k> (exam, console): this intent at every decision, to see what each one does; 'free': FREE, the
    z-free plan; None = not set"""
    v = os.environ.get(ENV_FIX, "").strip()
    if v == "":
        return None
    return FREE if v == "free" else int(v)


def step(logits: torch.Tensor, z: torch.Tensor, left: torch.Tensor, hold: int, eps: float, mode: str = "sample",
         fix: Optional[int] = None):
    """one decision of the intent: (z (B,) long, left (B,) long, due (B,) bool). Rows whose hold ran out (left <= 0: a start,
    a restart, H decisions done) draw again -- `mode` "sample" from `mix_probs`, "argmax" the likeliest -- and hold it H
    decisions (left = H - 1 after this one); the others keep theirs. `fix`: that intent, every row, every decision"""
    due = left <= 0
    if fix is not None:
        if not 0 <= int(fix) < logits.shape[1]:
            raise ValueError(f"{ENV_FIX}={fix}: this policy has intents 0..{logits.shape[1] - 1}")
        z_new = torch.full_like(z, int(fix))
        due = torch.ones_like(due)
    elif mode == "sample":
        z_new = torch.multinomial(mix_probs(logits, eps), 1)[:, 0].to(z.dtype)
    elif mode == "argmax":
        z_new = logits.argmax(1).to(z.dtype)
    else:
        raise ValueError(mode)
    z = torch.where(due, z_new, z)
    left = torch.where(due, torch.full_like(left, int(hold)), left) - 1
    return z, left, due


def arc_shape(device=None) -> torch.Tensor:
    """(ACT_DIM,) the start's shape on the 20 outputs: 1 on the 10 turn outputs (a steady steering offset: an arc), 0 on the
    speed outputs"""
    g = torch.zeros(PS.ACT_DIM, device=device)
    g[:PS.N_STEP] = 1.0
    return g


def lat_1s(mean: torch.Tensor, bias: torch.Tensor, v0: torch.Tensor, d0: torch.Tensor, lim: PS.Limits) -> torch.Tensor:
    """(K, n) where each intent puts the plan's 1 s point sideways [m, + left, the car's frame at the decision]: the plan
    drawn from `mean` (n, 20, pre-tanh, without the intent) + `bias` (K, 20), from v0 / d0 (n,) -- one roll of K n rows"""
    return lat_of(mean[None, :, :] + bias[:, None, :].to(mean.dtype), v0, d0, lim)


def lat_of(means: torch.Tensor, v0: torch.Tensor, d0: torch.Tensor, lim: PS.Limits) -> torch.Tensor:
    """(K, n, 20) plans' pre-tanh outputs at the same n states -> (K, n) their 1 s points sideways [m]"""
    K, n = means.shape[:2]
    p = PS.roll(torch.tanh(means.reshape(K * n, -1)), v0.repeat(K), d0.repeat(K), lim)
    return p["y"][:, -1].view(K, n)


def pair_spread(lat: torch.Tensor) -> torch.Tensor:
    """(K, n) -> (n,) the mean |difference| over the intents' pairs at each state"""
    K = lat.shape[0]
    d = (lat[:, None, :] - lat[None, :, :]).abs()
    return d.sum((0, 1)) / max(1, K * (K - 1))


@torch.no_grad()
def init_bias(lim: PS.Limits, K: int, lat: float, v_ref: float = V_REF, device=None) -> torch.Tensor:
    """(K, ACT_DIM) the intents' start: intent k = c_k x `arc_shape`, c_k found by bisection so that, from v_ref and
    straight steering with every other output 0 (a fresh net), the plan's 1 s point is linspace(-lat, lat, K)[k] m to the
    side of the plan the zero outputs draw"""
    dev = device if device is not None else lim.v_grid.device
    g = arc_shape(dev)
    target = torch.linspace(-float(lat), float(lat), K, device=dev)
    v0, d0 = torch.full((K,), float(v_ref), device=dev), torch.zeros(K, device=dev)
    y_of = lambda c: PS.roll(torch.tanh(c[:, None] * g[None, :]), v0, d0, lim)["y"][:, -1]
    y0 = y_of(torch.zeros(K, device=dev))
    # the 1 s point moves out with c up to where the arc starts to curl back (~2.4 m at 5 m/s): bisect on that branch only
    grid = torch.linspace(0.0, 3.0, 61, device=dev)
    yg = PS.roll(torch.tanh(grid[:, None] * g[None, :]), torch.full((61,), float(v_ref), device=dev), torch.zeros(61, device=dev),
                 lim)["y"][:, -1]
    c_top = float(grid[int(yg.argmax())])
    lo, hi = torch.full((K,), -c_top, device=dev), torch.full((K,), c_top, device=dev)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = (y_of(mid) - y0) < target                     # the offset grows with c (a left turn first: + left)
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    c = 0.5 * (lo + hi)
    if float((y_of(c) - y0 - target).abs().max()) > 0.005:
        raise ValueError(f"--intent-lat {lat}: the arc cannot reach it from {v_ref} m/s")
    return c[:, None] * g[None, :]


class IntentStats:
    """per update, the learner's intents in the ordinary races: decisions under each, redraws, redraws not the likeliest,
    how far apart the intents put the 1 s plan point at the same state (sampled every `every`-th decision, `rows` rows)"""

    def __init__(self, K: int, device, every: int = 4, rows: int = 64):
        self.K, self.every, self.rows = K, int(every), int(rows)
        self.n = torch.zeros(K, device=device)
        self.red = torch.zeros((), device=device); self.off = torch.zeros((), device=device)
        self.ent = torch.zeros((), device=device)
        self.lat = torch.zeros(K, device=device); self.spread = torch.zeros((), device=device); self.n_lat = 0
        self.lim = None

    def decision(self, t: int, z, due, logits, ordm, mean_of, v0, d0, lim) -> None:
        """`mean_of(idx, k)`: the plan's outputs (n, 20) at rows idx under intent k (k None: the z-free plan)"""
        zo, do = z[ordm], due[ordm]
        self.n += torch.bincount(zo, minlength=self.K).float()
        self.red += do.float().sum()
        self.off += (do & (zo != logits[ordm].argmax(1))).float().sum()
        self.ent += (entropy(logits[ordm]) * do.float()).sum()
        self.lim = lim
        if t % self.every == 0:
            idx = torch.nonzero(ordm)[:, 0]
            idx = idx[:: max(1, idx.numel() // self.rows)][: self.rows]
            if idx.numel():
                L = lat_of(torch.stack([mean_of(idx, k).detach() for k in range(self.K)]), v0[idx], d0[idx], lim)
                L0 = lat_of(mean_of(idx, None).detach()[None], v0[idx], d0[idx], lim)
                self.lat += (L - L0).mean(1); self.spread += pair_spread(L).mean(); self.n_lat += 1

    def row(self, bias: torch.Tensor) -> dict:
        nl = max(1, self.n_lat); red = float(self.red)
        # the table alone, at the start's reference (a zero plan from V_REF, straight): do the intents' vectors still differ --
        # the same-state numbers above shrink with the cars' speed (a standing car's 1 s point barely moves); with the trunk
        # reading z its share is not in this one (it is in the same-state numbers)
        ref = lat_1s(torch.zeros(1, PS.ACT_DIM, device=bias.device), bias.detach(), torch.full((1,), V_REF, device=bias.device),
                     torch.zeros(1, device=bias.device), self.lim)[:, 0] if self.lim is not None else None
        return {"intent_lat_ref": None if ref is None else [round(float(x), 4) for x in ref],
                "intent_share": [round(float(x), 4) for x in self.n / self.n.sum().clamp_min(1.0)],
                "intent_redraws": int(red), "intent_off_argmax": float(self.off) / red if red > 0 else None,
                "intent_entropy": float(self.ent) / red if red > 0 else None,
                "intent_lat_spread": float(self.spread) / nl if self.n_lat else None,
                "intent_lat": [round(float(x), 4) for x in self.lat / nl] if self.n_lat else None,
                "intent_bias_turn0": [round(float(x), 4) for x in bias[:, 0].detach()],
                "intent_bias_speed0": [round(float(x), 4) for x in bias[:, PS.N_STEP].detach()]}


def log_line(row: dict) -> str:
    """the update's log line part (Korean, as the rest of it)"""
    sh = " · ".join(f"{k} {v * 100:.0f} %" for k, v in enumerate(row["intent_share"]))
    off, ent, spr, lat = row["intent_off_argmax"], row["intent_entropy"], row["intent_lat_spread"], row.get("intent_lat")
    off_s = "-" if off is None else f"{off * 100:.0f}"
    ent_s = "-" if ent is None else f"{ent:.2f}"
    spr_s = "-" if spr is None else f"{spr * 100:.1f}"
    ls = "" if lat is None else " (의도별 " + " / ".join(f"{x * 100:+.0f}" for x in lat) + " cm)"
    eff = ""
    if row.get("intent_eff_outer") is not None:
        eff = (f", 실제 주행(의도 고정, z 없는 길 대비) 0번 {row['intent_eff_k0'] * 100:+.1f} cm / {len(row['intent_share']) - 1}번 "
               f"{row['intent_eff_kK'] * 100:+.1f} cm, 차이 {row['intent_eff_outer'] * 100:.1f} cm, 속도 "
               f"{row['intent_eff_speed_k0']:+.2f}/{row['intent_eff_speed_kK']:+.2f} m/s")
    ref = row.get("intent_lat_ref")
    rs = "" if ref is None else f", 5 m/s 기준 의도별 " + " / ".join(f"{x * 100:+.0f}" for x in ref) + " cm"
    return (f"의도: 고른 몫 {sh}, 다시 뽑음 {row['intent_redraws']}번 (가장 큰 확률이 아닌 것 {off_s} %, 엔트로피 {ent_s}), "
            f"같은 상태에서 의도끼리 1 s 점 옆 차이 {spr_s} cm{ls}{rs}{eff} | ")


def widen_critic(c_sd: dict, critic: nn.Module) -> Optional[dict]:
    """--init --init-critic into an --intent run from a checkpoint without it: the critic's input is the same but for the
    intent's columns at the end of its privileged part (the last inputs) -- those start at zero weight, their statistics
    mean 0 / variance 1, everything else as saved. None when the critic differs in anything else"""
    own = critic.state_dict()
    out = {}
    for k, v in own.items():
        if k not in c_sd:
            return None
        s = c_sd[k]
        if s.shape == v.shape:
            out[k] = s
            continue
        if k in ("norm.mean", "norm.var") and s.dim() == 1 and s.numel() < v.numel():
            fill = 0.0 if k == "norm.mean" else 1.0
            out[k] = torch.cat([s, torch.full((v.numel() - s.numel(),), fill, dtype=s.dtype, device=s.device)])
        elif k == "mlp.0.weight" and s.dim() == 2 and s.shape[0] == v.shape[0] and s.shape[1] < v.shape[1]:
            out[k] = torch.cat([s, torch.zeros(s.shape[0], v.shape[1] - s.shape[1], dtype=s.dtype, device=s.device)], 1)
        else:
            return None
    return out


def env_generators(env, depth: int = 4) -> list:
    """every torch.Generator reachable from `env` through its (and its f1sim parts') attributes, lists and dicts"""
    seen, out = set(), []

    def walk(o, d):
        if id(o) in seen or d > depth:
            return
        seen.add(id(o))
        if isinstance(o, torch.Generator):
            out.append(o)
            return
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x, d + 1)
            return
        if isinstance(o, dict):
            for x in o.values():
                walk(x, d + 1)
            return
        if torch.is_tensor(o) or isinstance(o, nn.Module):
            return
        mod = getattr(type(o), "__module__", "") or ""
        if (o is env or mod.startswith("f1sim")) and hasattr(o, "__dict__"):
            for x in vars(o).values():
                walk(x, d + 1)
    walk(env, 0)
    return out


EFFECT_ENVS = 32          # plan_rl --intent-effect-every: the eval's cars (16 races in a two-seat world) and seconds
EFFECT_S = 10.0


@torch.no_grad()
def realized_effect(env, lim: PS.Limits, actor, seconds: float = EFFECT_S, seed: int = 0, skip_s: float = 2.0) -> dict:
    """(11-26, the coordinator: log the intent's REALIZED effect) on a small env of its own (`env`, ideal execution as in
    training): the same start driven three times for `seconds` -- the z-free plan (FREE), the outer intents 0 and K-1 fixed
    -- and each outer run's line against the z-free run's path, per car and decision after `skip_s`: the signed distance
    [m, + left of the z-free path's heading] to its nearest point (within 1.5 m). Returns means: 'intent_eff_k0', ..._kK,
    'intent_eff_outer' (K-1 minus 0), their mean speed against the z-free run [m/s], hard contacts per run, how far apart the
    three starts were [m], and how many points matched"""
    from .plan_policy import PlanObs2, plan_v2, plan_world, PT_IDX
    from .. import dynamics as dyn
    K = int(actor.intent_k)
    lid = env.learner_ids
    dev = env.device
    T = int(round(seconds / PS.DT)); sk = int(round(skip_s / PS.DT))
    act0 = torch.zeros(env.B, env.act_dim, device=dev)
    runs = {}
    gens = env_generators(env)
    g0 = [g.get_state() for g in gens]
    for k in (FREE, 0, K - 1):
        # the same start each time: every generator the env holds goes back to where it was (whatever a run drew from it),
        # then the seeded reset
        for g, s_ in zip(gens, g0):
            g.set_state(s_)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        obs = PlanObs2(env, lim)
        st0 = env.sim.state[lid, :3].clone()
        c0 = env.sim.contact_counts[lid, 3:6].clone()
        P, V = [], []
        for _ in range(T):
            st = env.sim.state[lid].clone()
            P.append(st[:, :3].float()); V.append(st[:, dyn.IVX].float())
            _, _, _, _, p = plan_v2(actor, obs, lim, intent_fix=k)
            env.sim.ideal_vcap[lid] = 1e3
            rs = torch.zeros(lid.numel(), dtype=torch.bool, device=dev)
            for j in range(PS.SUB):
                env.ideal_cmd[lid, 0] = torch.where(rs, torch.zeros_like(p["cmd_d"][:, j]), p["cmd_d"][:, j])
                env.ideal_cmd[lid, 1] = torch.where(rs, torch.zeros_like(p["cmd_v"][:, j]), p["cmd_v"][:, j])
                _, _, te, trc, _ = env.step(act0)
                rs |= (te | trc)[lid]
            x, y, _ = plan_world(p, st)
            ii = PT_IDX.to(dev)
            obs.push_scan(); obs.set_plan(torch.stack([x[:, ii], y[:, ii]], -1), p["pv"]); obs.set_plan2(p)
            if bool(rs.any()):
                obs.reset_rows(torch.nonzero(rs)[:, 0])
        runs[k] = (torch.stack(P), torch.stack(V), st0, float((env.sim.contact_counts[lid, 3:6] - c0).clamp_min(0).sum()))
    Pf, Vf, sf, hf = runs[FREE]
    # how far apart the three starts are [m] (the reset puts the cars back exactly; the wheels' and actuators' state carried
    # from the run before moves them ~2 mm in the reset's one settling step)
    out = {"intent_eff_start_dx": max(float((runs[k][2][:, :2] - sf[:, :2]).norm(dim=-1).max()) for k in runs),
           "intent_eff_hard": [hf, runs[0][3], runs[K - 1][3]]}
    for name, k in (("k0", 0), ("kK", K - 1)):
        Pk, Vk = runs[k][0], runs[k][1]
        q, pf, hd = Pk[sk:, :, :2].transpose(0, 1), Pf[sk:, :, :2].transpose(0, 1), Pf[sk:, :, 2].transpose(0, 1)   # (B, T', .)
        d = torch.cdist(q, pf)                                                                      # (B, T', T')
        dmin, j = d.min(-1)
        pj = pf.gather(1, j[..., None].expand(-1, -1, 2)); hj = hd.gather(1, j)
        lat = -torch.sin(hj) * (q[..., 0] - pj[..., 0]) + torch.cos(hj) * (q[..., 1] - pj[..., 1])
        ok = dmin < 1.5
        out[f"intent_eff_{name}"] = float(lat[ok].mean()) if bool(ok.any()) else None
        out[f"intent_eff_speed_{name}"] = float(Vk[sk:].mean() - Vf[sk:].mean())
        out[f"intent_eff_n_{name}"] = int(ok.sum())
    a, b = out["intent_eff_k0"], out["intent_eff_kK"]
    out["intent_eff_outer"] = (b - a) if a is not None and b is not None else None
    return out
