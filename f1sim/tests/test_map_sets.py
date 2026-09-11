"""M / N stays inside the set that was picked.

A held-out map and a training map answer different questions -- can it drive something it has never
seen, versus did it learn the thing at all -- and the picker exists so the answer is never read off
the wrong one. The launcher used to hand the viewer the first ten maps of the flat union of every
set, so whichever set was chosen, one press of M walked straight out of it.
"""
import os

import pytest
import torch

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv
from f1sim.viewer.native import NativeViewer


@pytest.fixture
def make_viewer():
    """Builds headless viewers and releases their GL contexts afterwards.

    `NativeViewer.close()` only tears down a glfw window (`native.py:578-582`), so a headless viewer
    leaks its standalone context. Five of them used to accumulate across this module and outlive it.
    """
    pytest.importorskip("moderngl")        # fixture-level: skips these tests, not the session
    made = []

    def _viewer(n_tracks=6):
        cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
        tracks = [Track.generate_random(i) for i in range(n_tracks)]
        env = F1VecEnv(tracks, cfg, EnvConfig(), num_envs=2, device="cpu")
        env.reset(seed=0)
        v = NativeViewer(env.sim, headless=True, max_cars=2)
        made.append(v)
        return v

    yield _viewer
    for v in made:
        v.close()
        ctx = getattr(v, "ctx", None)
        if ctx is not None:
            ctx.release()


def test_m_and_n_stay_inside_the_active_set(make_viewer) -> None:
    v = make_viewer()
    v.track_groups = {"held-out": [0, 1], "training": [2, 3, 4, 5]}
    assert v.set_group("training")
    assert v.track_index == 2
    seen = set()
    for _ in range(8):                                        # twice round the set
        v.step_track(1); seen.add(v.track_index)
    assert seen == {2, 3, 4, 5}, seen
    seen = set()
    for _ in range(8):
        v.step_track(-1); seen.add(v.track_index)
    assert seen == {2, 3, 4, 5}, seen


def test_switching_set_moves_to_that_set(make_viewer) -> None:
    v = make_viewer()
    v.track_groups = {"held-out": [0, 1], "training": [2, 3, 4, 5]}
    v.set_group("training"); v.step_track(1)
    assert v.set_group("held-out")
    assert v.track_index in (0, 1)
    v.step_track(1)
    assert v.track_index in (0, 1)


def test_an_unknown_set_is_refused_rather_than_emptying_the_walk(make_viewer) -> None:
    v = make_viewer()
    v.track_groups = {"held-out": [0, 1]}
    v.set_group("held-out")
    assert not v.set_group("nope")
    assert v.track_group_name == "held-out"                   # unchanged, not blanked
    v.step_track(1)
    assert v.track_index in (0, 1)


def test_with_no_sets_it_walks_everything_loaded(make_viewer) -> None:
    v = make_viewer()
    assert v.group_indices() == list(range(v.sim.track.T))
    start = v.track_index                                     # the focus car's track, whichever it is
    v.step_track(1)
    assert v.track_index == (start + 1) % v.sim.track.T


@pytest.mark.skipif(os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1" or not os.environ.get("DISPLAY"),
                    reason="only meaningful under the software-GL CPU run contract")
def test_a_cpu_run_does_not_take_a_gl_context_on_the_gpu(make_viewer) -> None:
    """`headless=True` asks for `backend="egl"`, and EGL here resolves to the NVIDIA driver no matter
    what `CUDA_VISIBLE_DEVICES`, `LIBGL_ALWAYS_SOFTWARE` or `GALLIUM_DRIVER` say. That silently puts
    a CPU test run onto the GPU a training job is holding, and it poisons GL for every Qt widget
    created afterwards in the same process. `conftest.pytest_configure` redirects it to GLX; this is
    the check that notices if that stops working."""
    info = make_viewer().ctx.info
    renderer, vendor = info["GL_RENDERER"], info["GL_VENDOR"]
    assert "nvidia" not in renderer.lower() and "nvidia" not in vendor.lower(), \
        f"CPU run took a GPU GL context: {vendor} / {renderer}"
    assert "llvmpipe" in renderer.lower() or "softpipe" in renderer.lower(), \
        f"expected a software renderer, got {renderer}"


def test_the_launcher_builds_one_set_per_group_not_a_flat_union() -> None:
    from f1sim.learn.watch_gui import _map_sets
    sets_ = _map_sets()
    assert len(sets_) >= 3
    # every set is a set of its own: the held-out list must not be a prefix of the training list
    held, train = sets_["held-out  (never trained on)"], sets_["training set"]
    assert not (set(held) & set(train)), "held-out maps must not appear in the training set"


def test_a_command_reaches_the_viewer_but_only_once() -> None:
    """The panel writes, the viewer polls. It must not re-apply what it has already seen, and it must
    not apply what a previous session left behind -- that would move the viewer off the set that was
    just picked, on the first frame, for no reason the user can see."""
    import time
    from f1sim.learn import common
    common.viewer_command(set="training")
    stale = __import__("os").path.getmtime(common.VIEWER_CONTROL)
    cmd, t = common.viewer_poll_command(stale)                # a viewer starting now ignores it
    assert cmd is None and t == stale
    time.sleep(0.01)
    common.viewer_command(set="held-out")
    cmd, t = common.viewer_poll_command(stale)
    assert cmd and cmd["set"] == "held-out" and t > stale
    assert common.viewer_poll_command(t)[0] is None           # already seen
