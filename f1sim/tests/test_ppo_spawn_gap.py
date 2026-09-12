"""`--spawn-gap LOW HIGH`: the grid stagger becomes a declared range instead of a buried constant.

Three things are worth a test and nothing else is: an unflagged run still spawns at the range every
race so far was trained at, a range that cannot be sampled from is refused before any work is done,
and the flag actually reaches `EnvConfig` and then the reset that draws from it. The last one is the
only one that proves the flag is more than documentation.
"""
import dataclasses

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig, F1VecEnv
from f1sim.learn import ppo

TRAINED_DEFAULT = (2.5, 6.0)          # what every race before this flag existed spawned at
STAGE1_RANGE = (1.5, 9.0)             # the diversity arm's declared range


class _Captured(Exception):
    """Raised out of the stubbed `make_env` once the EnvConfig is in hand."""

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg


def _env_cfg_for(monkeypatch, extra_argv):
    """Run `ppo.main` far enough to build its EnvConfig, then stop.

    Stubbing the track load and `make_env` keeps this a CLI test: it reads what the parser hands the
    environment, with no maps, no model and no device beyond CPU.
    """
    argv = ["ppo", "--device", "cpu", "--envs", "2", "--race-size", "2",
            "--opponent", "mixed", "--tracks", "dummy", "--wandb", "disabled"] + list(extra_argv)
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(ppo.common, "track_names", lambda spec: ["dummy"])
    monkeypatch.setattr(ppo.common, "load_tracks", lambda names, **kw: ([object()], None))

    def capture(tracks, num_envs, device, env_cfg=None, **kw):
        raise _Captured(env_cfg)

    monkeypatch.setattr(ppo.common, "make_env", capture)
    with pytest.raises(_Captured) as caught:
        ppo.main()
    return caught.value.env_cfg


def test_unflagged_run_keeps_the_trained_spawn_range(monkeypatch):
    """No flag = the old behaviour, and the old behaviour is EnvConfig's own default."""
    assert tuple(EnvConfig().spawn_gap) == TRAINED_DEFAULT, \
        "the environment default moved; the CLI default has to move with it or they disagree"
    cfg = _env_cfg_for(monkeypatch, [])
    assert tuple(cfg.spawn_gap) == TRAINED_DEFAULT


@pytest.mark.parametrize("low,high", [
    ("0", "6.0"),        # a zero gap spawns two cars in the same pose
    ("-1.5", "6.0"),
    ("6.0", "2.5"),      # inverted: torch.rand would walk the gap *down* from the high end
    ("nan", "6.0"),
    ("2.5", "inf"),
])
def test_unsamplable_spawn_range_is_refused(monkeypatch, low, high):
    with pytest.raises(SystemExit) as exc:
        _env_cfg_for(monkeypatch, ["--spawn-gap", low, high])
    assert "--spawn-gap" in str(exc.value), f"refused without naming the flag: {exc.value}"


def test_spawn_gap_reaches_env_config_and_changes_nothing_else(monkeypatch):
    """The treatment arm's argv must differ from the control's in exactly this one field."""
    control = _env_cfg_for(monkeypatch, [])
    treatment = _env_cfg_for(monkeypatch, ["--spawn-gap", "1.5", "9.0"])
    assert tuple(treatment.spawn_gap) == STAGE1_RANGE
    differ = {f.name for f in dataclasses.fields(EnvConfig)
              if getattr(control, f.name) != getattr(treatment, f.name)}
    assert differ == {"spawn_gap"}, f"the flag moved more than the spawn range: {sorted(differ)}"


# A real lane, not an open floor: the stagger is an arc along a centerline, and a track without one
# spawns at its most open cell and reports no arc at all.
SPAWN_TRACK = "gen:competition:0"


@pytest.mark.parametrize("gap_range", [TRAINED_DEFAULT, STAGE1_RANGE])
def test_reset_draws_the_gap_from_the_configured_range_without_overlapping_cars(gap_range):
    """The forwarded range is what the reset samples, and its low end still leaves two bodies apart.

    Both arms are checked, because the claim that matters for the pilot is about the *wider* range:
    a 1.5 m low end has to stage a closer approach without staging a crash at t=0.
    """
    cfg = Config(); cfg.rand.enabled = False; cfg.lidar.motion_distortion = False
    ecfg = EnvConfig(race_size=2, opponent="policy", spawn_gap=gap_range, hist_len=0)
    env = F1VecEnv(maps.load(SPAWN_TRACK), cfg, ecfg, num_envs=64, device="cpu")
    env.sim.gen.manual_seed(801); torch.manual_seed(801)
    env._reset_envs(torch.arange(env.B))

    L = float(env.sim.track.length[0])
    s = env.sim.s.view(-1, env.M)                                      # (races, cars) arc positions
    d = (s[:, 0] - s[:, 1] + 0.5 * L) % L - 0.5 * L                    # signed, wrapped: leader ahead
    lo, hi = gap_range
    assert float(d.min()) >= lo - 1e-4 and float(d.max()) <= hi + 1e-4, \
        f"spawn arc gaps [{float(d.min()):.3f}, {float(d.max()):.3f}] m outside the configured {gap_range}"
    assert float(d.max()) - float(d.min()) > 0.25 * (hi - lo), \
        "the gap is not being drawn per reset; one constant would pass the bounds check above"

    xy = env.sim.state[:, :2].view(-1, env.M, 2)
    centres = (xy[:, 0] - xy[:, 1]).norm(dim=1)
    body = float(env.sim.car_dims[:, 0].max())                         # longest silhouette in the grid
    assert float(centres.min()) > body, \
        f"closest spawn pair {float(centres.min()):.3f} m apart, inside one car length ({body:.3f} m)"
    assert hi < 0.5 * L, f"a {hi} m gap is not below half the {L:.1f} m lane: the stagger would wrap"
