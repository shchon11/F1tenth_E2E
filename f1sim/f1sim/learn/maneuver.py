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

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

#: metres of progress over the teacher's horizon map to O(1)
PROG_SCALE = 10.0


def maneuver_spec(K: int = 28, hidden: int = 256, offsets: Optional[Sequence[float]] = None,
                  speeds: Optional[Sequence[float]] = None) -> dict:
    """The recorded build of a head. `offsets` / `speeds` name the candidate set it was trained on (the
    teacher's, candidate k = (offsets[k // len(speeds)], speeds[k % len(speeds)])), for a reader."""
    out = {"K": int(K), "hidden": int(hidden)}
    if offsets is not None:
        out["offsets"] = [float(o) for o in offsets]
    if speeds is not None:
        out["speeds"] = [float(s) for s in speeds]
    return out


class ManeuverHead(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, K: int, hidden: int = 256):
        super().__init__()
        self.K, self.A = int(K), int(act_dim)
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.plans = nn.Linear(hidden, self.K * self.A)
        self.score = nn.Linear(hidden, self.K)
        self.risk = nn.Linear(hidden, self.K)
        self.prog = nn.Linear(hidden, self.K)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(feat)
        plans = torch.tanh(self.plans(h)).view(-1, self.K, self.A)
        return {"plans": plans, "score": self.score(h), "risk": self.risk(h), "prog": self.prog(h)}

    @staticmethod
    def select(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, A): the plan of the candidate the score ranks first."""
        k = out["score"].argmax(1)
        return out["plans"].gather(1, k[:, None, None].expand(-1, 1, out["plans"].shape[2]))[:, 0]


def maneuver_loss(out: Dict[str, torch.Tensor], plans_t: torch.Tensor, choice_t: torch.Tensor,
                  hit_t: torch.Tensor, prog_t: torch.Tensor, fresh: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The four terms. `plans_t` (B, K, A) the teacher's plan for every candidate, `choice_t` (B,) the
    candidate it drove, `hit_t` / `prog_t` (B, K) the shadow simulation's outcome of every candidate
    (metres), valid on the rows where `fresh` (B,) -- the steps the teacher actually decided on."""
    plan = F.smooth_l1_loss(out["plans"], plans_t, beta=0.1)
    choice = F.cross_entropy(out["score"], choice_t)
    acc = (out["score"].argmax(1) == choice_t).float().mean()
    f = fresh.bool()
    if f.any():
        risk = F.binary_cross_entropy_with_logits(out["risk"][f], hit_t[f].float())
        prog = F.smooth_l1_loss(out["prog"][f], prog_t[f] / PROG_SCALE, beta=0.1)
    else:
        risk = prog = out["risk"].sum() * 0.0
    return {"plan": plan, "choice": choice, "risk": risk, "prog": prog, "choice_acc": acc}
