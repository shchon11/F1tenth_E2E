"""Analytic return targets distinguish truncation from termination."""
import torch

from f1sim.learn import ppo


def test_gae_bootstraps_terminal_observation_and_stops_at_episode_boundary():
    # Given normal, truncated, and terminated transitions in two parallel rows.
    rewards = torch.tensor([[1., 1.], [2., 2.], [100., 100.]])
    values = torch.tensor([[3., 3.], [4., 4.], [50., 50.], [60., 60.]])
    terminated = torch.tensor([[False, False], [False, True], [False, False]])
    truncated = torch.tensor([[False, False], [True, False], [False, False]])
    final_values = torch.tensor([[0., 0.], [10., 999.], [0., 0.]])
    # When GAE processes an episode boundary followed by a high-value reset state.
    advantage = ppo.compute_gae(rewards, values, terminated, truncated, final_values, 0.9, 0.8)
    # Then truncation uses V(final)=10 and neither boundary leaks the next episode.
    torch.testing.assert_close(advantage, torch.tensor([[6.64, 0.16], [7., -2.], [104., 104.]]))
