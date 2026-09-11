# Visualizer screenshots — provenance

Captured 2026-09-11 from the **real** `f1sim` driving console: the actual `ConsoleWindow` wired to
the actual `f1sim.viewer.sim_worker`, photographed the way a compositor would. Nothing is a mock-up,
and no value in any panel was drawn for effect — if a gauge shows a number, the running session
produced it.

Reproduce with `work/visualizer-images/run.sh` (outside this repository); the script and the full
machine-readable manifest live beside it.

## What is shown

| file | view | crop |
| --- | --- | --- |
| `f1tenth-visualizer-overview.png` | whole application, overview camera | none, 1920 × 1200 |
| `f1tenth-visualizer-chase.png` | chase camera, **setup panel collapsed** via the app's own `설정 패널` toggle so the viewport gets the width; telemetry column and policy strip retained | none, 1920 × 1200 |
| `f1tenth-visualizer-policy-panel.png` | **detail crop** of the policy input/output and ground-truth strip | cropped to that widget's own geometry, 1566 × 250 |
| `f1tenth-visualizer-props.png` | a modelled obstacle on `gen:competition:3+props7`, orbit camera | none, 1920 × 1200 |

## Session

| | |
| --- | --- |
| application | `f1sim` driving console (PyQt5), `f1sim.viewer.console` |
| map | `real:korea_2026_competition` — a SLAM map of a real venue |
| driver | the **frozen original policy** `ppo_race_0910/ppo_latest.pt`, sha256 `48cc698f8c51feb5…` |
| controller arm | **`legacy`** — the untouched tracker path, which is what the default console consumer accepts |
| session | 1 race × 1 car, `device="cpu"`, `compile=False` |
| randomisation | on; the true friction of this episode is shown in the panel as `시뮬 참값 µ` |
| setup panel | the run and map pickers are populated through the app's own `set_runs` / `select` API and set to the session that is actually driving, so the panel does not contradict the picture |

The checkpoint is pinned explicitly rather than taken from `latest_run()`, which currently points at
a research-controller run that the default consumer refuses. Bypassing that refusal to take a
screenshot would have meant running a policy through a tracker it was never shaped for.

## Rendering

| | |
| --- | --- |
| display | Xvfb, 2048 × 1280 × 24 |
| GL renderer | `llvmpipe (LLVM 15.0.7, 256 bits)`, Mesa 23.2.1 — **software, measured, not assumed** |
| GPU | none: `CUDA_VISIBLE_DEVICES=""`, `LIBGL_ALWAYS_SOFTWARE=1` |

Software rendering was deliberate: a training job owned the GPU and these images must not compete
with it.

**Consequence to read honestly:** the `sim 배속` and `렌더 fps` figures visible in the screenshots
(around 0.3–0.4 × real time, 13 fps) are what llvmpipe on a CPU worker produces. They are *not*
representative of the application on a GPU, and should not be read as performance numbers.

## Session health at capture time

552 frames delivered, **0 malformed**, 0 dropped at the worker, the receiver queue or the GUI. The
console validates every frame at its door, so a malformed count of zero means the pictures above are
of a healthy session rather than a degraded one.

## The obstacle image

`f1tenth-visualizer-props.png` is a separate session on **`gen:competition:3+props7`**, same driver
(`ppo_race_0910`, arm `legacy`), same CPU + llvmpipe setup.

The object in frame is **prop index 2, style `marker_post`**, standing at world
`(-7.750, -1.581)`, with the car 0.81 m from it. That identity comes from `track.props` — the
placement record the renderer, the LiDAR and the contact test all read — not from recognising a
shape on screen. The map places three props in total: two `steel_drum` at `(-6.314, 1.441)` and
`(-0.012, 5.139)`, and this marker post.

The viewport reported `prop_counts = (3 props, 1088 triangles, 2 batches)`, so the meshes reached
the renderer; the earlier difficulty was framing, not missing geometry.

The camera is the app's own orbit state (`_orbit = [azimuth, elevation, distance]`, the three
numbers drag and wheel set), placed opposite the post so the car and the object are both in frame,
and the shot is taken when the car is actually beside it. Composing a technical illustration this
way is deliberate; the map, the seed and the checkpoint are unchanged, and no seed was searched for
a more flattering scene.

**Caution that earned its keep:** an earlier attempt showed a pale translucent box near the car and
it was tempting to call it a prop. The viewer also draws the car's own rear detection box
translucently — visible on the car in this image — so that identification was never made. Only the
object matched to a `StaticProp` record is named here.

## One UI detail worth recording

The `차량 번호` display checkbox is left at its default (**on**), and the number above the car is
therefore drawn. Turning it off would not have removed it: `gl_scene.set_car_instances` falls back to
`np.arange(n)` when `labels is None`, so the positional index is drawn in place of the id — identical
to the id when a single car is running. An unchecked box beside a visible label would have been the
misleading screenshot, so the default was kept. This is an observation about the current code, not a
change to it; the runtime is frozen while training runs.

## Source

| | |
| --- | --- |
| repository | `/home/shchon11/F1tenth/F1tenth_E2E` |
| `viewer/console/window.py` | sha256 `de15c430b04df7b1…` |
| `viewer/sim_worker.py` | sha256 `4300ad6efde5d6dc…` |

## Not claimed

These are screenshots of one session on one map with one policy. They show that the application
runs and what it displays. They are **not** evidence of driving quality, lap time, collision rate or
any comparison between controller arms.

## Image hashes

Complete hashes are in `manifest.json` beside the capture script; the prefixes below are for reading.

| file | sha256 (first 16) | size |
| --- | --- | --- |
| `f1tenth-visualizer-overview.png` | `a6ba5ca427aff253…` | `1920 × 1200` |
| `f1tenth-visualizer-chase.png` | `9977c1c20349d229…` | `1920 × 1200` |
| `f1tenth-visualizer-policy-panel.png` | `8aec83fc07774cb1…` | `1566 × 250` |
| `f1tenth-visualizer-props.png` | `8b55fac5146fc7e3…` | `1920 × 1200` |

Nothing was retouched. The frame-rate and speed figures are exactly as the software-rendered session
produced them.
