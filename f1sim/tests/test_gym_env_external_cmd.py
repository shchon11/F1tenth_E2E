"""`F1VecEnv.set_external_command`: an outside (steer, speed) replaces the policy's action for one
car and only that car, in both action modes, and `clear_external_command` hands it back."""
import pytest

torch = pytest.importorskip("torch")

from f1sim.gym_env import EnvConfig, F1VecEnv              # noqa: E402
from f1sim.params import Config                            # noqa: E402
from f1sim.track import Track                              # noqa: E402


@pytest.fixture(scope="module")
def track():
    return Track.generate_random(3)


def _env(track, mode: str, n: int = 2):
    cfg = Config()
    cfg.sim.compile = False
    cfg.rand.enabled = False
    e = EnvConfig(action_mode=mode, speed_cap=4.0, max_steps=200, compile_tracker=False)
    env = F1VecEnv([track], cfg, e, num_envs=n, device="cpu")
    env.reset()
    return env


@pytest.mark.parametrize("mode", ["direct", "plan"])
def test_external_command_drives_only_the_masked_car(track, mode):
    env = _env(track, mode)
    act = torch.zeros(env.B, env.act_dim)
    if mode == "direct":
        act[:, 1] = -1.0                          # policy asks for speed 0 everywhere
    env.set_external_command(0, 0.2, 2.5)
    env.step(act)
    cmd = env.last_cmd
    assert float(cmd[0, 0]) == pytest.approx(0.2) and float(cmd[0, 1]) == pytest.approx(2.5)
    # car 1 is still the policy's: its command is whatever the mapping made of `act`, not ours
    assert not (float(cmd[1, 0]) == pytest.approx(0.2) and float(cmd[1, 1]) == pytest.approx(2.5))
    if mode == "direct":
        assert float(cmd[1, 1]) == pytest.approx(0.0)


def test_steer_is_clamped_to_lock_and_speed_passes_through(track):
    env = _env(track, "direct", n=1)
    env.set_external_command(0, 9.0, 7.0)
    env.step(torch.zeros(1, 2))
    assert float(env.last_cmd[0, 0]) == pytest.approx(float(env.s_max))
    assert float(env.last_cmd[0, 1]) == pytest.approx(7.0)     # above speed_cap on purpose: not capped


def test_clear_returns_the_car_to_the_policy(track):
    env = _env(track, "direct", n=1)
    env.set_external_command(0, 0.1, 3.0)
    env.step(torch.zeros(1, 2))
    env.clear_external_command(0)
    act = torch.zeros(1, 2); act[:, 1] = -1.0
    env.step(act)
    assert float(env.last_cmd[0, 1]) == pytest.approx(0.0)
    assert not env._ext_ids and not bool(env.ext_mask.any())
