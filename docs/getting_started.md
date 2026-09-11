# Getting started

Three installation paths, in increasing order of what they require. The first needs only Python.

- [Simulator only](#simulator-only) — CPU or GPU, no ROS
- [Training](#training) — CUDA
- [ROS 2 workspace](#ros-2-workspace) — Humble, `colcon`

## Simulator only

```bash
git clone --recurse-submodules https://github.com/shchon11/F1tenth_E2E.git
cd F1tenth_E2E
python3 -m venv .venv && source .venv/bin/activate
pip install -e f1sim
```

`torch` is a declared dependency but is usually installed to match the local CUDA version first;
install it from the [PyTorch instructions](https://pytorch.org/get-started/locally/) before the line
above if a specific build is needed.

Optional extras, declared in [`../f1sim/pyproject.toml`](../f1sim/pyproject.toml):

```bash
pip install -e "f1sim[gym]"       # gymnasium, for the vectorised training environment
pip install -e "f1sim[viewer]"    # moderngl, glfw, trimesh, for the 3D viewer
pip install -e "f1sim[learn]"     # wandb, for experiment logging
pip install -e "f1sim[dev]"       # pytest
```

**PyQt5 is not in any extra.** The driving console needs it; the project deliberately does not
declare it, so provide it in your environment yourself if you want the console. Everything else —
the headless recording paths, the legacy GL viewer, training — works without Qt.

Verify the installation:

```bash
cd f1sim && python3 -m pytest tests -q
```

### A directory-shadowing trap

The repository root contains a folder called `f1sim/`, and it has no `__init__.py`. Python therefore
treats it as a namespace package and it takes precedence over the installed one:

```console
$ cd F1tenth_E2E && python3 -c "import f1sim; print(f1sim.__file__)"
None                      # the folder, not the package -- `from f1sim import Track` fails here
$ cd /tmp && python3 -c "import f1sim; print(f1sim.__file__)"
.../F1tenth_E2E/f1sim/f1sim/__init__.py
```

`python3 -m f1sim.learn.<module>` and the scripts under `f1sim/scripts/` work from the repository
root regardless. Only plain `import f1sim` in your own script is affected — run it from anywhere
else.

### CPU versus GPU

The simulator runs on either. Two differences matter:

- `Config().sim.compile` defaults to `True` and is a CUDA path. Set `cfg.sim.compile = False` for
  CPU, as the quickstart does.
- CPU throughput is far lower — the 3D LiDAR and suspension models are heavy. Training does run on
  CPU (`--device cpu`), but at a size suited to smoke tests rather than real runs; real-time
  multi-car simulation wants a GPU.

No throughput figures are quoted here; measure on the target machine with
[`../f1sim/scripts/benchmark.py`](../f1sim/scripts/benchmark.py).

## Training

Training needs the `gym` extra; `learn` adds Weights & Biases logging. Runs and checkpoints are
written under `~/f1sim_runs/<name>/`. See [Training](training.md) for the workflows.

CUDA is **recommended, not required**. `--device cpu` works for the training loops and the
estimator, and is what the CPU test suite exercises, but it is slow enough that only small
smoke-sized runs are practical. Large `--envs` counts are a GPU-memory decision.

Checkpoints are not tracked in the repository, and none are distributed with it.

## ROS 2 workspace

The bridge in [`../f1sim_ros/`](../f1sim_ros/) targets ROS 2 Humble. The simulator package carries a
`COLCON_IGNORE` file so `colcon` does not try to build it as an ament package — install it with
`pip` as above.

Several third-party submodules must also be hidden from `colcon`, since only a subset of
`f1tenth_system` is wanted and the map collections are not packages at all:

```bash
bash external/setup_colcon_ignore.sh
```

Then build from the workspace root that contains this repository as `src/`:

```bash
colcon build --symlink-install
source install/setup.bash        # or setup.zsh
```

What is built: `vesc_msgs`, `vesc_ackermann`, `ackermann_mux`, `f1tenth_stack` from
`f1tenth_system`, `diagnostic_updater` from `ros/diagnostics`, and `f1sim_ros` itself. The hardware
drivers (`vesc_driver`, `urg_node`, teleop tools) are ignored, because the simulator replaces exactly
those — see [ROS 2](ros2.md).

## Troubleshooting

**A viewer window opens black, with the HUD missing.** The GL frames are not being presented. The
usual cause is PRIME render-offload environment variables (`__NV_PRIME_RENDER_OFFLOAD`,
`__GLX_VENDOR_LIBRARY_NAME`) exported by a shell profile while the NVIDIA GPU is already primary;
every GLX application is affected, not only this one. The viewer drops those variables automatically
in that configuration — set `F1SIM_KEEP_PRIME_ENV=1` to keep them.

**`torch.compile` errors or long stalls on first run.** Set `cfg.sim.compile = False`, or call
`sim.warmup()` before opening a viewer window so kernel compilation does not stall the first frames.

**No GPU available.** Everything in the quickstart works on CPU. Training does not.

Workstation-specific commands for the machine this was developed on are in
[로컬 실행 안내](local_pc.md).
