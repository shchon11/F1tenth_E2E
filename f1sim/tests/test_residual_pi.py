"""`--residual-solo` (state_rl.ResidualPi): with nothing in view the actor is the frozen solo policy, bit for bit."""
import torch

from f1sim.learn import state_rl as SR


def _model(bounds=True):
    torch.manual_seed(0)
    m = SR.ActorCritic(409, act_dim=8, residual=(107, 256))
    with torch.no_grad():
        m.pi.vis_mask[306:] = True
        m.pi.vis_empty[306:] = torch.randn(103)
        for p in m.pi.res.parameters():           # a trained change, not the near-zero init
            p.normal_(0.0, 0.3)
    if bounds:
        m.set_mean_bounds([-1.0] * 8, [1.0] * 8, [0.2] * 8)
    return m


def test_empty_view_is_the_solo_policy():
    m = _model()
    x = torch.randn(64, 409) * 3
    xe = torch.where(m.pi.vis_mask, m.pi.vis_empty, x)
    with torch.no_grad():
        solo = m.pi.base((xe * m.actor_keep)[:, :107]).clamp(-1.0, 1.0)
        assert torch.equal(m.pi(xe), solo)


def test_something_in_view_changes_the_action_and_trains_only_the_change():
    m = _model()
    x = torch.randn(64, 409) * 3
    out = m.pi(x)
    with torch.no_grad():
        solo = m.pi.base((x * m.actor_keep)[:, :107]).clamp(-1.0, 1.0)
    assert (out - solo).abs().max() > 1e-3
    out.sum().backward()
    assert all(p.grad is None for p in m.pi.base.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.pi.res.parameters())


def test_no_soft_bound_hook_on_the_residual_actor():
    m = _model()
    x = torch.randn(8, 409) * 3
    with torch.no_grad():
        assert torch.equal(m.pi(x), m.raw_mean)


def test_frozen_norm_does_not_move():
    n = SR.RunningNorm(4, "cpu")
    n.frozen = True
    n.update(torch.randn(100, 4) * 5 + 3)
    assert torch.equal(n.mean, torch.zeros(4)) and torch.equal(n.var, torch.ones(4))
