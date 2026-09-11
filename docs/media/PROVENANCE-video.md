# Demonstration clips — provenance

Two 25-second clips rendered 2026-09-11 from the real `f1sim` stack: the actual `Simulator` and
`F1VecEnv`, the actual `NativeViewer` and `Scene`, stepped at the real 40 Hz control rate. Every
value in either overlay is read from the snapshot that frame drew.

These are illustrative clips, not a performance comparison. Which comparisons exist and what each
answers is in the README's status section.

Render scripts: `work/simulation-videos/`, outside this repository. Beside the media here:
`video-manifest.json` (machine-readable record of every field below) and
`video-continuity-check.json` (frame-difference results).

## Files

| file | clip |
| --- | --- |
| `f1tenth-simulator-demo.mp4` / `.gif` | the simulator, driven by the scripted raceline teacher |
| `f1tenth-learning-demo.mp4` / `.gif` | a trained policy, with the run's own training history |

Each clip also has `-poster.png` (one full-resolution frame) and `-contact-sheet.png` (six frames
across the clip). The GIFs are 5-second excerpts at 480 px and 10 fps; the MP4s are the full article.

## Clip 1 — the simulator

| | |
| --- | --- |
| map | `real:korea_2026_competition` — SLAM map of a real venue |
| driver | `RacelineTeacher(mode="pp", speed_scale=0.92)` — the scripted reference driver, not a learned policy |
| seed | `cfg.sim.seed = 7` (also the post-construction global torch seed) |
| randomisation | on — `Config().rand.enabled` defaults to True |
| episode µ | 0.920972, constant for the clip |
| episodes | 1; no collision, no reset |
| device | `cpu`, `compile=False`, 1 car |
| shots | overview 5 s, chase 9 s, close-up 5 s, orbit 6 s |

Verified by replay: `replay_check.py clip1 1000` re-runs the same `world()` with no viewer and
records every collision flag and every change in `sim.P["mu"]` — 0 collisions, 0 µ changes, µ
constant across 1000 control steps.

## Clip 2 — a trained policy

| | |
| --- | --- |
| checkpoint | `cl_main_fixed_low_s501/ppo_final.pt` — update 128/128, 1,048,576 steps, seed 501 |
| action space | `plan` — the policy emits a local plan, a tracker drives it |
| controller arm | `fixed_low`, from the grip spec recorded in this checkpoint: `mu_fixed` 0.73423, `rho_max` 0.85, `a_max` 7.0, `a_brake` 5.0, `n_path` 25; `estimator_path` null; `flags_fired` empty |
| observation layout | from `extra["spec"]` |
| privileged adapter | `absent_opponent_17_to_21` |
| speed cap | 9.0 m/s, from the checkpoint |
| map | `real:korea_2026_competition` |
| seed | `cfg.sim.seed = 3` |
| randomisation | on |
| episodes | 2 — see below |
| device | `cpu`, `compile=False`, 1 car |
| shots | chase 10 s, overview 5 s, close-up 5 s, orbit 5 s |

`load_checkpoint(..., allow_controller=True)` is required here: the default loader refuses a
checkpoint whose recorded controller arm is not `legacy` (`f1sim/learn/model.py:324-341`), because
replaying its plans through the untouched tracker would evaluate them under a controller they never
saw. The flag is for a caller that installs the matching runtime, and the `ControllerRuntime` here is
built from the grip spec recorded in this checkpoint.

### Collision and reset

The clip contains a collision. `F1VecEnv.step` auto-resets a terminated env inside the same call, so
the car respawns and the randomised parameters are re-drawn.

| | episode 1 | episode 2 |
| --- | --- | --- |
| first control step | 0 | 204 |
| sim time | 0.000 s | 5.125 s |
| drawn µ | 0.75466 | 0.79472 |

Event: `collision_reset`, control step 204, sim t 5.125 s, encoded frame 153. The controller assumed
µ 0.73423 throughout both episodes, which is what the `fixed_low` arm does; both numbers are on
screen and the drawn one is rebuilt each frame from the live record.

Nothing is interpolated across the boundary: pose, camera and telemetry all come from the single
snapshot after the reset, and the chase camera's smoothing is restarted. A `COLLISION — EPISODE
RESET` banner is drawn for 1.5 s from frame 153.

### Chart panel

The run's own per-update training log, parsed from
`wandb/run-20260911_170554-569zhco9/files/output.log`. Collisions per km as `log10(1+x)` (defined at
zero, no floor); progress linear; both with measured end values. The 16 updates that logged `nan` are
drawn as gaps. It is the finished run's history with update 128 marked as the one being played, and
it does not animate during the clip.

## Timing

Both clips run at 1.0 × real time, timeline uncompressed. 40 Hz physics into 30 fps video is 1.333
control steps per frame; the displayed pose and the camera are taken from one snapshot blended
between the two bracketing 40 Hz states, by the rule the live viewer uses (`native.py:374-381`) —
continuous quantities linear, yaw unwrapped first, the LiDAR scan taken whole from the later frame.
The physics is never interpolated. Episode boundaries snap instead of blending.

## Rendering

| | |
| --- | --- |
| display | Xvfb, 1920 × 1080 × 24 |
| GL renderer | `llvmpipe (LLVM 15.0.7, 256 bits)`, Mesa 23.2.1 — software, read off the context |
| GPU | none: `CUDA_VISIBLE_DEVICES=""`, `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe` |
| CPU | `nice -n 10`, `LP_NUM_THREADS=2`, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`, ffmpeg `-threads 2` |
| resolution | 1920 × 1080 native, not upscaled |
| encoder | libx264, preset slow, crf 24, High profile, yuv420p, `+faststart` |

`NativeViewer(headless=True)` requests an EGL context and EGL on this host resolves to the NVIDIA
driver, so `moderngl.create_standalone_context` is redirected inside the render process to take the
GLX path under Xvfb. The runtime is not modified.

Three things are disabled on the viewer instance, not in the source: the development HUD
(`scene.set_hud` made a no-op), the car-number badge (`scene.n_labels = 0` after
`set_car_instances`, which falls back to `np.arange(n)` when `labels is None`), and the tracker's
predicted-motion line in clip 2. The area beyond the ground plate is composited with a dark gradient
in place of the renderer's flat clear colour — a background replacement on pixels that were never
drawn; no geometry is altered or hidden.

## Checks against the delivered files

Every MP4 was fully decoded (`ffmpeg -f null -`), probed with `ffprobe`, and sampled at three points
whose mean and standard deviation were measured; all sampled frames non-black and distinct. Results
per clip in `video-manifest.json` under `decode_check`.

Frame-difference continuity, in `video-continuity-check.json`: expected discontinuities are the shot
cuts and the episode boundary, both read from the render record. Zero unexplained jumps in either
clip — simulator demo cuts at frames 150/420/570, max difference elsewhere 8.45 against threshold
18.67; learning demo cuts at 300/450/600 plus the episode boundary at 153, max elsewhere 9.20 against
threshold 21.19.

## Scope

One map, two rollouts, 1 × real time. These clips are not a benchmark and not a comparison between
controller arms. `cl_main_fixed_low_s501` is a (controller, policy) pair, so nothing here separates
the controller from the policy trained against it.

## Files and hashes

| file | size | duration | sha256 |
| --- | --- | --- | --- |
| `f1tenth-simulator-demo-contact-sheet.png` | 0.88 MB | — | `ade3bb9dd1f7ce416d8e7fc2233eade2e7b840944641aec2a8ee4ad6e56de6e7` |
| `f1tenth-simulator-demo-poster.png` | 0.55 MB | — | `01396965572dda6e454ad197c68c6eefa4e7d447598d7156f2381316cf0ee13a` |
| `f1tenth-simulator-demo.gif` | 2.44 MB | 5 s excerpt from 5.5 s, 480 px, 10 fps | `6994d0098d0819bf540344a4a434a246e5d6e66008b8e29fb7b380b974927018` |
| `f1tenth-simulator-demo.mp4` | 10.32 MB | 25.00 s, 750 frames @ 30 fps | `bd577a076702eb38d47f06f83ba97fbe7f1f9aa413d4015aeb92a1111a5cb9bf` |
| `f1tenth-learning-demo-contact-sheet.png` | 1.05 MB | — | `3dd7470db67be4269c408f0cb6aadf2484f0e1ad1bf0c00aa9edc062076d1bae` |
| `f1tenth-learning-demo-poster.png` | 0.85 MB | — | `57be547b1ec63c7580fc07e4a3a53ee8e97ea5be3d529f2d82e17a0e095fa7b9` |
| `f1tenth-learning-demo.gif` | 2.47 MB | 5 s excerpt from 1.5 s, 480 px, 10 fps | `3858ae4b55b7b2d8be50ffdbbfebee5328550afeb3c16d74142ed60c6c43c2d9` |
| `f1tenth-learning-demo.mp4` | 11.01 MB | 25.00 s, 750 frames @ 30 fps | `d878acf1dbd37cd51837f9130545b6a783687a82ffebe8d819ae124d9adc00e3` |
| `PROVENANCE-video.md` | — | — | (this set; hash after write) |
| `video-manifest.json` | — | — | (this set; hash after write) |
| `video-continuity-check.json` | — | — | (this set; hash after write) |

## Source hashes

| file | sha256 |
| --- | --- |
| `f1sim/viewer/native.py` | `711cb723a0118dace232d788f7cd75fae804c7fe8a0dc0bdacd103f6978e5f77` |
| `f1sim/viewer/gl_scene.py` | `98b879b5c6203a9eb085c298eab1145f680d7ec198cb1e6a01065f134e31cb6b` |
| `f1sim/viewer/console/theme.py` | `416d69d53e9617aae2cb35e33e6d84ead2f615aaebde9e797a82213761fc7975` |

The recorder that produced both clips:

| file | sha256 |
| --- | --- |
| `work/simulation-videos/render_lib.py` | `fab9c3238644b2bd5ef53a201be5d4459e5744c8ab8b5ede8ada060f4a006197` |
| `work/simulation-videos/shoot.py` | `72288b0d070bceec2fc03fd00fefe3ed43ab4c82ea3b93c36b98bb30a0b80978` |
| `work/simulation-videos/clip1.py` | `f4a6b6a98352b0ea3851bf035cd4c6e98d7f6ada62718fbac99c9fdcd5e6c64d` |
| `work/simulation-videos/clip2.py` | `a13531d3ba7f16b60d24f79f623706238ea64ffdeaa581893f8980feba84d0d8` |
| `work/simulation-videos/train_chart.py` | `c091d9c488a329f54d3082bdd0e47e0e197696e3514e87c150fe0b8abd8c388b` |
| `work/simulation-videos/package.py` | `48b776bbd5a3dbb486fce0d3edfee1bc30c1db518cf48092e8ef10f5aae8fdff` |
| `work/simulation-videos/deliver.py` | `7cf6c6736662b7993d7593bb22b582a15a6c5baafaba9d405b95223cff48df40` |

The overlays are composited over the rendered frame; the rendered frame is the simulator's output.
