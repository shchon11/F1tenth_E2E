"""The maneuver head: K candidate plans, which one the teacher would take, and what each would lead to.

Why (2026-10-01). The rollout teacher (`f1sim.rollout_teacher`) decides in traffic by simulating K = 28
candidate moves -- "follow the line at lateral offset o, speed scale s" -- for 1.5 s with the other
cars reacting, and takes the clear one that gets furthest. It races at 0-4 car contacts with 54-58
passes on ICCAS over two seeds, the behaviour a human driver shows: see the risk, back off or take
another line. Its DAgger students hit 15-20 contacts per km against the teacher's 1.1. Two reasons,
both structural:

* the teacher's choice is DISCRETE -- pass left, pass right, hold back -- and a student that regresses
  one plan onto it averages two good options into the bad middle one (straight into the car);
* the teacher's choice depends on what the other car will do, which the student was never asked to
  represent (its features carry the opponent's position at R^2 ~0).

So the student is given the teacher's whole decision, not only its outcome:

* `plans`  (B, K, A): every candidate's plan, regressed on the teacher's own plan for that candidate;
* `score`  (B, K):    logits of which candidate the teacher takes (cross-entropy) -- the policy acts on
  the argmax, so a choice between two good options is a choice, never an average;
* `risk`   (B, K):    logit of "this candidate touches a car, a wall or a prop within the horizon",
  from the teacher's shadow simulation (BCE, on the steps where the teacher decided);
* `prog`   (B, K):    the progress it would make over the horizon, metres / PROG_SCALE (Huber).

`risk` and `prog` are the per-option outcome the search computed and used to throw away: 28 labelled
"if I go there" answers per decision, delivered at the moment of the decision, instead of one contact
1-2 s later. They shape the trunk (the same features feed `score` and `plans`) and they are the
guidance an onboard selector or a constrained RL stage can read.
"""
from __future__ import annotations

from typing import Tuple, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

#: metres of progress over the teacher's horizon map to O(1)
PROG_SCALE = 10.0


def maneuver_spec(K: int = 28, hidden: int = 256, offsets: Optional[Sequence[float]] = None,
                  speeds: Optional[Sequence[float]] = None, mode: str = "plans",
                  cand_off: Optional[Sequence[float]] = None, cand_spd: Optional[Sequence[float]] = None,
                  i_line: Optional[int] = None, v_max: Optional[float] = None, k_lim: Optional[float] = None,
                  temp: float = 1.0) -> dict:
    """The recorded build of a head. `offsets` / `speeds` name the candidate set it was trained on (the
    teacher's, candidate k = (offsets[k // len(speeds)], speeds[k % len(speeds)])), for a reader."""
    out = {"K": int(K), "hidden": int(hidden)}
    if mode not in ("plans", "transform"):
        raise ValueError(f"maneuver mode must be 'plans' or 'transform', got {mode!r}")
    if mode == "transform":
        # the candidates are transforms of the plain head's ONE plan (`offset_plan`), so the head needs
        # each candidate's (offset, speed scale), which one is the plain line, and how to decode a plan
        if cand_off is None or cand_spd is None or i_line is None or v_max is None or k_lim is None:
            raise ValueError("maneuver mode 'transform' needs cand_off, cand_spd, i_line, v_max and k_lim")
        out.update(mode="transform", cand_off=[float(x) for x in cand_off], cand_spd=[float(x) for x in cand_spd],
                   i_line=int(i_line), v_max=float(v_max), k_lim=float(k_lim))
    if float(temp) != 1.0:
        # the PPO mixture's choice temperature (`ManeuverDist`); the deterministic action is the argmax either way
        if not float(temp) > 0:
            raise ValueError(f"maneuver temp must be positive, got {temp}")
        out["temp"] = float(temp)
    if offsets is not None:
        out["offsets"] = [float(o) for o in offsets]
    if speeds is not None:
        out["speeds"] = [float(s) for s in speeds]
    return out


class ManeuverHead(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, K: int, hidden: int = 256, transform: bool = False):
        super().__init__()
        self.K, self.A, self.transform = int(K), int(act_dim), bool(transform)
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        # 'transform': no plans of its own -- the candidates are the plain head's plan, shifted
        self.plans = None if self.transform else nn.Linear(hidden, self.K * self.A)
        self.score = nn.Linear(hidden, self.K)
        self.risk = nn.Linear(hidden, self.K)
        self.prog = nn.Linear(hidden, self.K)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(feat)
        out = {"score": self.score(h), "risk": self.risk(h), "prog": self.prog(h)}
        if self.plans is not None:
            out["plans"] = torch.tanh(self.plans(h)).view(-1, self.K, self.A)
        return out

    @torch.no_grad()
    def prefer(self, i: int, margin: float = 4.0) -> None:
        """A fresh head that ranks candidate `i` first everywhere: attached to a trained plain student
        ('transform', i = the plain line) it drives exactly as that student did until it learns otherwise."""
        self.score.weight.mul_(0.1)
        self.score.bias.zero_()
        self.score.bias[int(i)] = float(margin)

    @staticmethod
    def rank(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, K) what the action is chosen by: the teacher-choice logit plus log P(no contact) from the risk
        head -- imitation, vetoed by the predicted outcome. The teacher's own choice is near-tied among
        candidates that are all clear, so its label alone ranked the clear ones well and the ones that
        hit badly (first DAgger iteration: choice accuracy 0.39, the student at 604 contacts/km)."""
        return out["score"] + F.logsigmoid(-out["risk"])

    @staticmethod
    def select(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, A): the plan of the candidate ranked first."""
        k = ManeuverHead.rank(out).argmax(1)
        return out["plans"].gather(1, k[:, None, None].expand(-1, 1, out["plans"].shape[2]))[:, 0]


def maneuver_loss(out: Dict[str, torch.Tensor], plans_t: torch.Tensor, choice_t: torch.Tensor,
                  hit_t: torch.Tensor, prog_t: torch.Tensor, fresh: torch.Tensor,
                  base: Optional[torch.Tensor] = None, i_line: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """The four terms. `plans_t` (B, K, A) the teacher's plan for every candidate, `choice_t` (B,) the
    candidate it drove, `hit_t` / `prog_t` (B, K) the shadow simulation's outcome of every candidate
    (metres), valid on the rows where `fresh` (B,) -- the steps the teacher actually decided on."""
    if base is not None:
        # 'transform': the plain head's one plan learns the teacher's plain-line candidate; every other
        # candidate is a transform of it, so there is nothing else to regress
        plan = F.smooth_l1_loss(base, plans_t[:, int(i_line)], beta=0.1)
    else:
        plan = F.smooth_l1_loss(out["plans"], plans_t, beta=0.1)
        # and the candidate the teacher drove, on its own: the plan that is actually executed most of the time
        A = plans_t.shape[2]
        pick = lambda x: x.gather(1, choice_t[:, None, None].expand(-1, 1, A))[:, 0]
        plan = plan + F.smooth_l1_loss(pick(out["plans"]), pick(plans_t), beta=0.1)
    choice = F.cross_entropy(out["score"], choice_t)
    acc = (out["score"].argmax(1) == choice_t).float().mean()
    f = fresh.bool()
    if f.any():
        risk = F.binary_cross_entropy_with_logits(out["risk"][f], hit_t[f].float())
        prog = F.smooth_l1_loss(out["prog"][f], prog_t[f] / PROG_SCALE, beta=0.1)
    else:
        risk = prog = out["risk"].sum() * 0.0
    return {"plan": plan, "choice": choice, "risk": risk, "prog": prog, "choice_acc": acc}


class ManeuverDist:
    """The policy of an actor with a maneuver head, for PPO: a mixture over the K candidates.

        p(a) = sum_k softmax(score)_k * N(a | plan_k, std)

    Sampling draws the candidate first and then the plan around it, so exploration happens at the
    level of "take another line / back off", not only as per-step noise on one plan. The log
    probability is the exact mixture (logsumexp), so the PPO ratio needs no stored candidate index.
    It quacks like the `Normal` the plain head returns where the callers use it: `log_prob(a)` and
    `entropy()` are (B, A) with the per-sample total spread over the columns (callers `.sum(1)`),
    `mean` is the plan of the top-ranked candidate (the deterministic action), `stddev` is per dim.
    The entropy is the categorical's plus one component's -- an upper bound on the mixture's, used
    only as an exploration bonus.
    """

    def __init__(self, out: dict, std: torch.Tensor, temp: float = 1.0):
        # temp < 1 sharpens the choice. At 1 a DAgger student on ICCAS solo puts 0.77 on its top candidate,
        # so a PPO rollout drew another candidate on 23 % of steps -- a line / speed jump every 0.11 s, and
        # the solo stage's wall rate climbed 16 -> 34/km. 0.33 makes it one every ~2 s (scratchpad
        # choice_temp.py). The argmax, and so the deterministic action, is unchanged.
        self.logits = ManeuverHead.rank(out).float() / float(temp)
        self.plans = out["plans"].float()                              # (B, K, A)
        B, K, A = self.plans.shape
        self.A = A
        self.std = std.float().expand(A)
        self._top = self.plans.gather(1, self.logits.argmax(1)[:, None, None].expand(-1, 1, A))[:, 0]

    @property
    def mean(self) -> torch.Tensor:
        return self._top

    @property
    def stddev(self) -> torch.Tensor:
        return self.std.expand_as(self._top)

    def sample(self) -> torch.Tensor:
        k = torch.distributions.Categorical(logits=self.logits).sample()
        mu = self.plans.gather(1, k[:, None, None].expand(-1, 1, self.A))[:, 0]
        return mu + self.std * torch.randn_like(mu)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        comp = torch.distributions.Normal(self.plans, self.std).log_prob(a.float()[:, None, :]).sum(-1)   # (B, K)
        lp = torch.logsumexp(F.log_softmax(self.logits, 1) + comp, 1)
        return (lp / self.A)[:, None].expand(-1, self.A)

    def entropy(self) -> torch.Tensor:
        h = torch.distributions.Categorical(logits=self.logits).entropy() \
            + torch.distributions.Normal(torch.zeros_like(self.std), self.std).entropy().sum()
        return (h / self.A)[:, None].expand(-1, self.A)


def offset_candidates(base: torch.Tensor, v_meas: torch.Tensor, offsets: torch.Tensor, speeds: torch.Tensor,
                      v_max: float, spec, k_lim: float, iters: int = 6,
                      uniq: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    """(B, K, A) the K candidates as transforms of ONE plan (linear output), the way the rollout teacher
    builds its own: candidate k follows the base plan's path shifted `offsets[k]` metres to its left
    (fitted through the shifted points at 40-100 % of the plan, as `RacelineTeacher.plan_action` fits an
    offset line) with both speeds scaled by `speeds[k]`. The teacher shifts the RACELINE, which only it
    can see; a student shifts its own plan -- the one thing it already predicts well on unseen tracks.
    `offsets` / `speeds` are (K,)."""
    from ..mpc import N_KNOTS, decode, encode, path_points
    from ..teacher import fit_knots
    B, K = base.shape[0], offsets.shape[0]
    cap = torch.full_like(v_meas, v_max)
    k0, Lp, v0, v1 = decode(base, v_meas, v_max, cap, spec)
    x, y, psi, _ = path_points(k0, Lp, 25)
    fr = torch.linspace(0.4, 1.0, 6, device=base.device, dtype=base.dtype)
    si = (fr * 24).round().long()
    px, py, pp = x[:, si], y[:, si], psi[:, si]                       # (B, 6)
    # The path depends on the offset alone -- a speed scale only multiplies the two speeds -- so the
    # Gauss-Newton fit runs once per DISTINCT offset (7 of the 28 candidates) and is shared. It was
    # the whole cost of a maneuver step (104 ms of fits per 256-car step, 441 ms per 1024 minibatch).
    if uniq is None:
        uo, inv = torch.unique(offsets, return_inverse=True)
    else:
        uo, inv = uniq
    U = uo.shape[0]
    o = uo.to(base)[None, :, None]                                    # (1, U, 1)
    tx = (px[:, None] - o * torch.sin(pp)[:, None]).reshape(B * U, -1)
    ty = (py[:, None] + o * torch.cos(pp)[:, None]).reshape(B * U, -1)
    kK = k0.repeat_interleave(U, 0); LpK = Lp.repeat_interleave(U, 0)
    lim = torch.full((B * U,), float(k_lim), device=base.device, dtype=base.dtype)
    fractions = torch.tensor([1., .5, .25, .125, 0.], device=base.device, dtype=base.dtype)
    kf = fit_knots(kK, tx, ty, LpK, fr, lim, iters, fractions)
    au = encode(kf, v0.repeat_interleave(U, 0), v1.repeat_interleave(U, 0), v_max, spec,
                v_meas=v_meas.repeat_interleave(U, 0)).view(B, U, -1)
    a = au[:, inv.to(base.device)]                                    # (B, K, A)
    # a zero offset is the base plan itself, not its refit (as in `offset_plan`)
    z = (offsets == 0).to(base.device)
    a = torch.where(z[None, :, None], base[:, None, :].expand(B, K, -1), a).reshape(B * K, -1)
    s = speeds.to(base).repeat(B)
    a = a.clone(); a[:, -2:] = ((a[:, -2:] + 1.0) * s[:, None] - 1.0).clamp(-1.0, 1.0)
    return a.view(B, K, -1)


def offset_plan(base: torch.Tensor, v_meas: torch.Tensor, off: torch.Tensor, spd: torch.Tensor, v_max: float,
                spec, k_lim: float, iters: int = 6) -> torch.Tensor:
    """(B, A) ONE candidate per row -- `offset_candidates` for per-row (offset, speed scale) `off`, `spd` (B,).
    What a 'transform' head executes: only the chosen candidate is built; a zero offset is the base plan itself."""
    from ..mpc import decode, encode, path_points
    from ..teacher import fit_knots
    B = base.shape[0]
    a = base
    if bool((off != 0).any()):
        cap = torch.full_like(v_meas, v_max)
        k0, Lp, v0, v1 = decode(base, v_meas, v_max, cap, spec)
        x, y, psi, _ = path_points(k0, Lp, 25)
        fr = torch.linspace(0.4, 1.0, 6, device=base.device, dtype=base.dtype)
        si = (fr * 24).round().long()
        o = off.to(base)[:, None]
        tx = x[:, si] - o * torch.sin(psi[:, si]); ty = y[:, si] + o * torch.cos(psi[:, si])
        lim = torch.full((B,), float(k_lim), device=base.device, dtype=base.dtype)
        fractions = torch.tensor([1., .5, .25, .125, 0.], device=base.device, dtype=base.dtype)
        kf = fit_knots(k0, tx, ty, Lp, fr, lim, iters, fractions)
        shifted = encode(kf, v0, v1, v_max, spec, v_meas=v_meas)
        a = torch.where((off != 0)[:, None], shifted, base)
    a = a.clone(); a[:, -2:] = ((a[:, -2:] + 1.0) * spd.to(base)[:, None] - 1.0).clamp(-1.0, 1.0)
    return a
