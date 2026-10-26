"""Return targets for auto-reset vector environments."""
from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, terminated: torch.Tensor,
    truncated: torch.Tensor, final_values: torch.Tensor, gamma: float, lam: float,
) -> torch.Tensor:
    """Bootstrap truncations from final observations; stop recursion at either boundary. `lam` may be a (T, B)
    tensor, one lambda per step: 1 inside a held plan's block makes the block's middle values cancel, so the
    advantage at a decision is the decision-level GAE (`EnvConfig.decision_every`)."""
    advantage = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        nonterminal = (~terminated[t].bool()).to(rewards.dtype)
        continuing = nonterminal * (~truncated[t].bool()).to(rewards.dtype)
        next_value = torch.where(truncated[t].bool(), final_values[t], values[t + 1])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lam_t = lam[t] if torch.is_tensor(lam) else lam
        last = delta + gamma * lam_t * continuing * last
        advantage[t] = last
    return advantage
