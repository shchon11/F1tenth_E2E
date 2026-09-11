"""PPO must reevaluate the same Gaussian sample that generated its rollout."""

import pytest
import torch

from f1sim.learn.model import ActorCritic
from f1sim.learn.ppo import sample_rollout_action


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_unchanged_policy_ratio_is_one_when_actions_saturate(amp: bool, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Given a policy whose sampled actions regularly exceed the actuator bounds.
    torch.manual_seed(17)
    model = ActorCritic(1, 32, 4, 3).to(device)
    with torch.no_grad():
        model.actor.log_std.fill_(1.0)
    scan = torch.rand(8, 1, 32, device=device)
    proprio = torch.rand(8, 4, device=device)
    privileged = torch.rand(8, 3, device=device)

    # When PPO samples and reevaluates a rollout without an optimizer update.
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
        actions, old_logp = sample_rollout_action(model, scan, proprio)
        new_logp, _, _, _ = model.evaluate(scan, proprio, privileged, actions)
    ratio = (new_logp - old_logp).exp()

    # Then the probability ratio is one even for samples outside the bounds.
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-6, rtol=1e-6)
    assert actions.abs().max() > 1
    assert old_logp.dtype == torch.float32
