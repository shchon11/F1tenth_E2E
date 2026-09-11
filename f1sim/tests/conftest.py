import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os                                                              # noqa: E402
import dataclasses                                                     # noqa: E402

import pytest                                                          # noqa: E402


def pytest_configure(config):
    """Under the software-GL contract, keep headless viewer contexts off EGL.

    `NativeViewer(headless=True)` asks for `create_standalone_context(backend="egl")`
    (`native.py:120`); EGL ignores `CUDA_VISIBLE_DEVICES` / `LIBGL_ALWAYS_SOFTWARE` /
    `GALLIUM_DRIVER` and can hand back a GPU context, which also breaks any Qt GLX context made
    later in the process. Dropping the backend takes the GLX path (llvmpipe under Xvfb).

    moderngl is an optional viewer dependency: without it there is nothing to patch and the
    non-viewer tests must still collect and run.
    """
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1" or not os.environ.get("DISPLAY"):
        return
    try:
        import moderngl
    except Exception:                          # optional extra absent, or present but unimportable
        return
    if getattr(moderngl, "_f1sim_tests_software_gl", False):
        return
    original = moderngl.create_standalone_context

    def software_standalone(*args, **kw):
        kw.pop("backend", None)
        return original(*args, **kw)

    moderngl.create_standalone_context = software_standalone
    moderngl._f1sim_tests_software_gl = True


@pytest.fixture(scope="session")
def tmp_legacy_run(tmp_path_factory):
    """A run directory holding one checkpoint the DEFAULT consumer can open, built here.

    The console and worker tests used to drive `run="latest"`, which resolves through
    `latest_run()` to whatever trained most recently. Once controller-arm training started, that is
    a checkpoint whose recorded arm is not `legacy`; `load_checkpoint` refuses it without an opt-in
    flag the default consumer does not pass, and every one of those tests failed on a machine whose
    newest run happened to be ours. Scanning for the newest *compatible* run fixes the symptom but
    keeps the test dependent on what is lying around in `~/f1sim_runs` and on file mtimes.

    So generate one instead: fixed seed inside `torch.random.fork_rng()`, so the weights are
    deterministic and the global RNG the rest of the suite shares is left exactly as it was found.
    Legacy by construction -- no `experiment.controller`, no conditioning -- and therefore loadable
    with no flag. `resolve_checkpoint` accepts an absolute path, so this drops straight into a
    `SessionConfig(run=...)`. Session-scoped: about 4 MB, written once.
    """
    import torch

    from f1sim.learn.model import ActorCritic, save_checkpoint
    from f1sim.learn.obs import ObsSpec

    # `n_beams` comes from the simulator's own config, not from `ObsSpec()`'s default: the default
    # is 1080 and the simulator produces 1081, and a checkpoint built on the default loads fine and
    # then fails when the first real observation arrives.
    from f1sim.params import Config
    spec = ObsSpec(n_beams=Config().lidar.n_beams)
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=17, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(20260912)
        model = ActorCritic(**meta)
    run_dir = tmp_path_factory.mktemp("runs") / "tiny_legacy"
    run_dir.mkdir()
    save_checkpoint(str(run_dir / "ppo_latest.pt"), model,
                    {"spec": dataclasses.asdict(spec), "phase": "ppo", "run": "tiny_legacy",
                     "update": 1, "updates": 1, "steps": 1, "total_steps": 1})
    return str(run_dir)
