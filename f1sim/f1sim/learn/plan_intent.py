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
    moves only when PPO (first step) or the tail loss (the forecast) moves it. The trunk, the memory and the speed estimate
    do not read z; the critic does (`PlanObs2.intent_priv`: z one-hot while held, the share of the hold left) -- z is part
    of the state the value depends on, and a value blind to it would mix the intents' returns into every held decision's
    baseline;
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
    """F1SIM_INTENT_FIX=<k> (exam, console): this intent at every decision, to see what each one does; None = not set"""
    v = os.environ.get(ENV_FIX, "")
    return int(v) if v.strip() != "" else None


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
    K, n = bias.shape[0], mean.shape[0]
    u = (mean[None, :, :] + bias[:, None, :].to(mean.dtype)).reshape(K * n, -1)
    p = PS.roll(torch.tanh(u), v0.repeat(K), d0.repeat(K), lim)
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

    def decision(self, t: int, z, due, logits, ordm, mean_nom, bias, v0, d0, lim) -> None:
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
                L = lat_1s(mean_nom[idx].detach(), bias.detach(), v0[idx], d0[idx], lim)
                L0 = lat_1s(mean_nom[idx].detach(), torch.zeros_like(bias[:1]), v0[idx], d0[idx], lim)
                self.lat += (L - L0).mean(1); self.spread += pair_spread(L).mean(); self.n_lat += 1

    def row(self, bias: torch.Tensor) -> dict:
        nl = max(1, self.n_lat); red = float(self.red)
        # the table alone, at the start's reference (a zero plan from V_REF, straight): do the intents' vectors still differ --
        # the same-state numbers above shrink with the cars' speed (a standing car's 1 s point barely moves)
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
    ref = row.get("intent_lat_ref")
    rs = "" if ref is None else f", 5 m/s 기준 의도별 " + " / ".join(f"{x * 100:+.0f}" for x in ref) + " cm"
    return (f"의도: 고른 몫 {sh}, 다시 뽑음 {row['intent_redraws']}번 (가장 큰 확률이 아닌 것 {off_s} %, 엔트로피 {ent_s}), "
            f"같은 상태에서 의도끼리 1 s 점 옆 차이 {spr_s} cm{ls}{rs} | ")


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
