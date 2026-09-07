import torch

from f1sim.mpc import PlanTracker
from f1sim.learn.model import ActorCritic
from f1sim.learn.watch import actor_runner, viewer_config, viewer_threaded


def test_viewer_starts_without_simulator_compilation_by_default() -> None:
    config = viewer_config(False)
    assert config.sim.compile is False


def test_compiled_viewer_avoids_rng_cuda_graph_mode() -> None:
    config = viewer_config(True)
    assert config.sim.compile is True
    assert config.sim.compile_mode == "default"
    assert viewer_threaded(True) is False
    assert viewer_threaded(False) is True


def test_actor_runner_skips_compilation_when_disabled(monkeypatch) -> None:
    model = ActorCritic(1, 32, 4, 3)

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("torch.compile must be opt-in for the viewer")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    run = actor_runner(model, torch.device("cpu"), False)
    action = run(torch.rand(2, 1, 32), torch.rand(2, 4))
    assert action.shape == (2, 2)


def test_plan_tracker_can_skip_compiled_solver(monkeypatch) -> None:
    import f1sim.mpc as mpc
    monkeypatch.setattr(mpc, "solve_fast", lambda *args: (_ for _ in ()).throw(AssertionError("compiled solver called")))
    tracker = PlanTracker(1, "cpu", 0.33, 0.4, 4.0, compile_solver=False)
    command = tracker(torch.zeros(1, 6), torch.ones(1), torch.full((1,), 4.0))
    assert command.shape == (1, 2)
