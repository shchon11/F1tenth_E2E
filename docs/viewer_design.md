# Driving console: visual design

**2026-09-12.** The console's look was rebuilt around how the major simulators present a scene, so
that someone who has used Isaac Sim, CARLA or an AV stack's replay tool finds nothing to relearn. The
references below were read for their *conventions*, not copied: our scene is a 1:10 track with duct
barriers, and the design has to serve that.

## References, and what was taken from each

| reference | convention adopted here |
| --- | --- |
| [Isaac Sim / Omniverse UI](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/gui/reference_user_interface.html) | graphite chrome with **no hue in the panels**, a large central viewport with docked panels either side, small monospace viewport chips (map / run in the corner, state badge), a ground plane with a fine and a coarse lattice that fades with distance, one calm selection accent, NVIDIA-style green reserved for "running" and the start action |
| [CARLA (Unreal Engine)](https://carla.readthedocs.io/en/latest/core_actors/) spectator and [sensor views](https://carla.readthedocs.io/en/latest/ref_sensors/) | a lit scene with a real sky instead of a void: directional sky gradient, horizon haze that the fog fades into, warm key light with cool sky ambient and a faint fresnel rim; LiDAR returns drawn as small soft discs coloured by what they hit |
| [NVIDIA Alpamayo](https://github.com/NVlabs/alpamayo) / DRIVE-style trajectory replay ([blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/)) | the planned trajectory as a **translucent flat ribbon with a bright core** rather than a wire, a ring on the ground under the ego vehicle, trails that fade out behind, neutral vehicle paint so the per-vehicle tint (ego / rival / collided) *is* the vehicle colour |

## Tokens

Defined once in `f1sim/viewer/console/theme.py` and reused by the GL scene (`gl_scene.py` carries
the matching `CLEAR_RGB` / `HORIZON_RGB` / `ZENITH_RGB`).

| token | value | used for |
| --- | --- | --- |
| `bg.window` / `bg.panel` / `bg.card` / `bg.raised` | `#161616` / `#1c1c1c` / `#222222` / `#2c2c2c` | four steps of graphite, never blue |
| `bg.viewport` | `#121316` | the viewport's own ground haze (`CLEAR_RGB`) |
| `line` / `line.strong` | `#303030` / `#454545` | hairlines, gauge tracks |
| `text.0` / `.1` / `.2` | `#e6e6e6` / `#a9a9a9` / `#747474` | primary, secondary, captions |
| `accent` / `accent.deep` | `#5aa9ff` / `#22415f` | selection, focus, measured value arcs |
| `primary` / `primary.line` | `#3d6f0b` / `#76b900` | the start button and the running badge only |
| `ego` | `#59e6ff` | the watched car: ring in the scene, marker in the policy panel |
| `rival` / `warn` / `danger` | `#f2a044` / `#f0a030` / `#ff5c5c` | a rival car and commanded values; preparing; collision and over-cap |

Type: body 12 px, labels 11 px, section headings 11 px with letter-spacing, metrics in
`DejaVu Sans Mono`. Radii 3 px on controls, 4 px on cards. Header 44 px.

## Scene

| element | before | now |
| --- | --- | --- |
| background | flat clear colour, black above the horizon | sky gradient by view direction (zenith `ZENITH_RGB` to horizon `HORIZON_RGB`), fog fades to the horizon tone, ground haze below |
| ground | light grey plate, 1 m lattice only | asphalt-toned apron (`0.135`) that dissolves into a backdrop disc of the same tone; 1 m and 5 m lattices, each fading with pixel footprint |
| duct barriers | mirror-like, ±26 % corrugation bands | satin (`metal` = 48 / 0.55), ±11 % bands |
| lighting | grey hemisphere, single key | cool sky ambient + warm ground bounce, warm key, fresnel rim |
| cars | tint multiplied a blue-grey paint so every car was teal | neutral paint; ego drawn true-colour with a cyan ring on the ground, rival amber, collided red, others dimmed |
| plan | 6 px wire | 0.11 m translucent ribbon + 2.5 px core, speed-coloured; tracker prediction stays a thin line |
| LiDAR | hard 22 px discs, saturated | 12 px soft discs, by hit type (duct amber, tall object magenta, floor teal, car yellow) |
| trails | constant alpha | fade out behind the car |
| chips | rounded cards | translucent graphite slabs with a hairline border, monospace |

## Instruments

The dash strip is an instrument cluster, not a dashboard prop: a flat ring gauge for speed (track,
measured sweep in the accent, over-cap zone red, commanded marker inside the ring, reading inside the
dial), a centred steering bar (fill from centre to the measured angle, marker for the commanded
angle, left steer to the left), and the g-g trace against the friction circle. Columns are separated
by hairlines. Everything a value stands for is unchanged from the previous drawing.

## Verification

Rendered headlessly with the same capture path as the README screenshots (Xvfb + llvmpipe, real
console, real worker, checkpoint `cl_origrecipe_legacy_s701` under the legacy arm). Console and
viewport test files pass (190). The GLSL for the sky and scene programs compiles in a standalone
context; `Scene` constructs headless.
