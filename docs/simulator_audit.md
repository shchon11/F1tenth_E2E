# Simulator audit, September 2026

A pass over the simulator's transition boundaries, its sampling clocks, the learning-side contracts
and the ROS interfaces, prompted by "is there anything wrong with the simulation itself". Everything
below describes the code as it now stands, what was measured to justify it, and what is still
unverified. Findings that were reproduced are fixed with a regression test each; findings that are
modelling choices rather than defects are recorded and left alone.

This file exists so the audit does not have to be redone from the diff.

---

## 1. What changed, and why

### The IMU ran at 40 Hz while claiming 50, and told its noise model 20 ms

`sample_indices` took `floor(imu_rate * control_dt)` samples per control step. At the configured
50 Hz against a 40 Hz control loop that is `floor(1.25) = 1`, so the sensor delivered 40 Hz. Worse,
`Simulator.imu_ts` was `1 / imu_rate = 20 ms`, and that is the step `imu.sample()` integrates with —
the gyro bias random walk (`sqrt(ts)`), the attitude estimate's gyro integration (`gyro * ts`) and
the complementary-filter gain (`ts / tau`). Samples 25 ms apart advancing a 20 ms internal clock left
accumulated bias drift about 11 % low and the filter's effective time constant 25 % off.

A sensor does not restart because a control step ended, so it now keeps its own clock.
`imu.sample_schedule` works out, from `Fraction(rate * control_dt)`, the exact cycle in which the two
clocks realign — four control steps at these rates — and lays the samples out inside it. The count
per control step is therefore not constant: it cycles **1, 1, 1, 2** and averages 50 Hz exactly.
`imu_ts` is `1 / rate` again and is now true.

Because the last sample of a step is no longer on the step boundary, `StepResult` carries
`imu_offsets`: how long before the end of that control step each sample was taken. It is a new
optional field defaulting to `None`, so existing constructions keep working.

**Rates that cannot be scheduled are refused, not truncated.** `sample_schedule` raises on a
non-finite or non-positive rate, on a rate above the physics tick (two samples would share a
substep), on one needing an impractically long cycle (each phase is a separate compiled graph), and
on one slower than the control loop — that last leaves control steps with no sample at all, and
`gym_env._obs` drops the IMU keys when that happens, changing the width of the proprioceptive vector
mid-episode.

`sample_indices` is kept and is still exact when the ratio is an integer; it is documented as that
case only.

`Simulator.warmup` steps once per schedule phase, so every compiled variant is built before the
real-time loop starts, and restores the sensor phase afterwards along with `cl_idx`,
`car_collision`, `sim.gen` and the global CPU/CUDA RNG. (`imu.sample` and the lidar/odom noise draw
from the *global* generator, not `sim.gen`, so restoring only the latter left the noise stream
shifted.) CPU seeded equivalence is asserted by test. The frontend's hardware run confirms it on CUDA as well
— identical IMU and scan sequences with and without warmup on a cached compiled graph, and a
mutation control (the same comparison with the global-RNG restore disabled) that does differ, so the
equality is not vacuous: `work/claude-viewer-rebuild/evidence/h_w1_warmup_rng.json`.

> **This changes observations.** `r.imu.mean(1)` is now the mean of one or two samples depending on
> the step. Anything compared against curves recorded before this needs re-baselining.

### An ended episode was scored against the track it never drove on

`env.step` auto-resets finished rows before it returns, and the reset re-draws the track. The trainer
read `env.sim.tid` afterwards to decide which track's collision history to credit, so every finished
episode was attributed to whatever track its replacement had just been given. The env already
published the pre-reset snapshot as `info["track_id"]`; the trainer uses it now. Totals were never
affected — the per-track speed-cap gate was.

### The role mask belonged to the next episode

`info["on_policy"]` handed out the env's live tensor. `_reset_envs` re-draws which cars the policy
drives, in place, further down the same `step`, and the trainer reads the mask after `step` returns.
The last transition of every episode was therefore weighted by the roles of the episode that
replaced it. It is a snapshot now, like `track_id` beside it.

### The auxiliary opponent loss masked at 30 m, not 6

`privileged()` stores the nearest-opponent block divided by `PRIV_OPP_DIST_SCALE`, so
`priv[:, 11] < 6.0` selected 30 m — three times the LiDAR's own range. Measured on a rollout, that
mask passed 100 % of frames: it removed nothing, and trained the head to predict a car's position on
frames where no beam touched a car (only ~47 % of frames have any). The scale now lives in one place
(`gym_env.PRIV_OPP_DIST_SCALE`, imported by `learn/ppo.py`) and the comparison is made in metres
against `AUX_OPP_RANGE_M`.

> **This changes training.** The auxiliary population drops from 100 % of frames to roughly 77 %. The
> target's *value* is unchanged, so checkpoints trained against it keep their meaning.

### `scan_subsample > 1` crashed

`gym_env` sized the scan with `n_beams // subsample` while `_norm_scan` produced
`scan[:, ::subsample]`. Those agreed while the LiDAR had 1080 beams; the measured count is 1081
(270° / 0.25° with both endpoints), and `1081 // 4 = 270` against a slice of length 271. Every
`scan_subsample > 1` died on a shape mismatch when the scan history was written. The width is a
ceiling now. The default is 1, so training was never affected.

### The GPU grid cache could hand one track another's tall geometry

`Track.grid_key` hashed occupancy and duct but not `tall`. `TrackTensors` dedups on that key and, on
a hit, skips appending `edt`/`edt_duct`/`edt_tall` entirely — so two tracks with the same occupancy
and duct but different `tall` shared the first one's tall field, and the LiDAR traced the second
against geometry it does not have.

This is reachable from the public API, not just from the obstacle builders: `Track.from_occupancy`
takes `duct` and `tall` as independent arrays, and `from_ros_map`'s unknown-floor handling moves
`tall` without touching occupancy or duct. It went unnoticed because the two obstacle builders happen
to write `tall` and `occupancy` together.

Reproduced before fixing: two tracks built through `from_occupancy` differing only in `tall` both
reported a tall-clearance of 2.157 m at a point where one of them has a block and should read 0.
`tall` is now in the hash. Tracks that differ only in centerline — a reversed lap — still share one
grid, which is what the dedup is for.

### The run config could not name its own track set

`learn/ppo.py` reused the local `names` — the track manifest — for the optimizer's parameter names
while restoring Adam state, so every resumed run recorded `actor.log_std`, `actor.stem…` as its
track list. Renamed to `param_names`. Metadata only; nothing trained differently.

### ROS interfaces published times and orientations they had not measured

* `LaserScan.header.stamp` is the acquisition time of the **first ray** per the ROS 2 message
  definition. Both bridges stamped it at the end of the control step, which is when the *last* ray
  was traced. It now goes back by one sweep (18.75 ms at the default rates).
* IMU samples are stamped per sample from `imu_offsets` instead of an assumed even spacing ending at
  `now`; the summary message is stamped at the latest sample, not at the step end.
* The simulator carries one attitude estimate per control step, measured at its end. `vesc_sim_node`
  copied it onto every raw sample, attributing a future attitude to earlier ones. Orientation is now
  published only on the last sample of a step; earlier samples declare it unavailable with
  `orientation_covariance[0] = -1`. A consumer subscribing only to the raw topic still gets a
  quaternion once per step.
* `vesc_sim_node`'s `VescImuStamped.ypr` was `(yaw, pitch, roll)`. Measured against the quaternion in
  the same message across four recordings, the hardware's convention is `ypr.x = -roll`,
  `ypr.y = +pitch`, `ypr.z = -yaw`, in degrees. It matched the dataset in neither order nor sign, so
  a consumer calibrated on a bag disagreed with the same consumer on the simulator. The quaternion
  remains canonical.
* `scan_meta` derives its timing from `control_dt`, which is what the simulator does — one scan per
  control step, beams spread over `pose - pose_prev`. Identical to the old expression at the default
  40 Hz/40 Hz; it only differed if a LiDAR rate other than the control rate was configured, and then
  the header published a rate the simulator was not running at.

### `calib.bagread` put topics on different clocks

The origin was `min(first stamp of each SELECTED topic)`. Two consequences, both of which corrupted a
lag estimate rather than failing: reading a subset moved the origin, so `read(bag, ["/odom"])` and
`read(bag, ["/odom", "/drive"])` disagreed about when `/odom` happened — by seconds, since `/drive`
starts 1.96 s into these recordings — and taking each topic's *first* stamp assumed records were
written in timestamp order.

The origin is now the smallest record timestamp in the **whole bag**, collected **before** the topic
filter, and exposed as `BagData.origin_record_ns`. Stamps are normalised in integer nanoseconds
before conversion: float64 spacing at a 2025 epoch timestamp in ns is 256 ns, so converting first
quantised exactly the sub-microsecond differences these calibrations are quoted at. Each topic is
stable-sorted, because `BagData.at` interpolates with `np.interp`, which requires increasing x and
returns nonsense rather than an error otherwise. An empty bag no longer raises `min()` on an empty
sequence.

> The relative `t` this returns is intentionally different from before for any subset read.

---

## 2. Recorded, deliberately not changed

- **The auxiliary "closing speed" is not a closing speed.** `priv[10]` is `other.vx - ego.vx`, a
  difference of two body-frame longitudinal speeds, not the rate the gap changes at.
  `car_proximity_penalty` computes the real thing separately. Measured, the two correlate at +0.04
  within 6 m — where it matters. The comments now say what the channel is. Changing what it *means*
  would silently reinterpret every checkpoint trained against it.
- **`collisions_per_km` means two different things and both names are kept.** `learn/ppo.py` divides
  collisions by **centerline progress** (`ep_progress`); `learn/evaluation_metrics.py` divides by
  **integrated path length** (`|v| dt`). Backtracking, lateral motion and partial episodes make them
  diverge. Do not paste a number from one next to a number from the other; a training curve and an
  evaluation report are not the same statistic even though they share a label.
- **`info["final"]["lap_time"]` is an episode duration, not a lap time**, and nothing reads it. Real
  lap times come from `info["lap_times"]`, which are filtered by a minimum plausible lap. The field
  is kept for compatibility.
- **The tyre model's `mu` scaling is a modelling choice.** The normalized Pacejka form makes `mu`
  scale the small-slip cornering stiffness as well as the peak, which is why `mu` and the stiffness
  factors are nearly degenerate at low excitation. Changing it without calibration evidence would be
  guessing.
- **The teacher's grip bins clamp at 1.0.** The randomized `mu * mu_f_scale` reaches 1.196 of
  nominal, so roughly the top sixth of the range folds into one bin. Deliberate and documented in
  `teacher.py`.
- **`bagread`'s `/odom` gyro fallback endorsement is withdrawn, with the reasoning qualified.** A
  large raw |gz| is not by itself proof of hardware corruption — a unit mismatch looks the same, and
  two of seven cross-checked bags are consistent with one, so the gyro's units must be validated per
  bag. Independently of that, `/odom` twist.angular.z is not an independent fallback: command
  reconstruction was confirmed on seven pre-competition bags, and the competition bags' producer is
  unverified, which is a reason for caution rather than permission.

---

## 3. Checked and found correct

- `_reset_result` clones every tensor before touching the reset rows, so `info["progress"]`,
  `info["lap"]` and `info["wall_dist"]` keep their pre-reset values. `info["track_id"]` was already
  cloned, and the `info["final"]` block is captured before the reset.
- The PPO buffer's privileged vector is re-read after `step` returns, so it lines up with the
  observation stored beside it; the auxiliary labels are aligned.
- A per-env reset does not disturb the IMU's sample clock.
- **The LiDAR sweep duration was suspected and cleared.** `Lidar.time_frac` already carries the
  `fov / 2π = 0.75` factor, so the beams span `0.75 × control_dt = 18.750 ms`, and the metadata says
  18.750 ms. No change was needed.
- `lidar.rays` at zero attitude gives slope `k ≡ 0`, beam angles spanning exactly ±135°, and an
  origin at the mount lever arm. `mpc.encode`/`decode` are exact inverses at `ACT_DIM = 8`.
  `raceline.speed_profile` respects `sqrt(a_lat · R)` on a circle and never exceeds `v_max`.
  Car-car contact uses a separating-axis test whose four axes are sufficient because the
  footprint-plus-rear-box hull is still a rectangle in body frame.

---

## 4. Validation

All CPU test runs pin `OMP/MKL/OPENBLAS=1`. torch defaults to 10 intra-op threads and the test
config does not limit them, which made small vectorised steps thread-bound: one IMU test went from
14.41 s to 1.93 s (7.5×) with the pin. No global configuration and no physics were changed.

| Run | Targets | Result |
|---|---|---|
| **Final, changed paths** | `test_sim_audit` `test_bagread_origin` `test_ros_publisher_timing` `test_episode_boundaries` `test_multicar` `test_gym_env` `test_imu` `test_multitrack` `test_track_modifiers` `test_ppo_actions` `test_ppo_returns` | **75 passed, 2 skipped** (3:55) |
| Read-only audited areas | `test_car_mesh_lidar` `test_lidar` `test_lidar3d` `test_track_lap` `test_track_modifiers` `test_map_sets` `test_obstacle_seeds` `test_raceline` `test_raceline_feasibility` `test_raceline_obstacles` `test_teacher_grip` `test_teacher_recovery` `test_mpc` `test_dynamics` | **62 passed, 1 skipped** — log: `work/claude-simulator-audit/audit-area-tests.log` |
| Grid-cache fix regression | `test_sim_audit` + `test_track_lap` `test_track_modifiers` `test_map_sets` `test_obstacle_seeds` `test_multitrack` `test_lidar` `test_lidar3d` `test_car_mesh_lidar` | **56 passed, 1 skipped** |
| Before the changes, same targeted list | — | 7 failed, 24 passed — all seven now pass, none newly broken. Log: `work/claude-simulator-audit/targeted-BASELINE.log` |

An earlier, smaller changed-paths run reported 65 passed / 2 skipped; the 75/2 row above is the final
one and supersedes it. Exact commands and per-run detail are in
`work/claude-simulator-audit/evidence.json`.

**GPU parity.** The declared 50 Hz is delivered identically on CPU, CUDA eager and CUDA compiled:
`[1, 1, 1, 2]` per cycle, 5 events in the first 0.1 s, sample-grid error 4.5e-10. With identical
spawn poses copied from the CPU and every stochastic source disabled, the per-env parameter tables
match exactly, **IMU samples and offsets agree to 0.0 over every step and channel**, sample counts
match, and the worst state difference is 1.43e-06 — never above 1e-4 — over 40 steps in which every
car moved more than 0.5 m. The schedule compiles to 5 graphs with no eager fallback, under the
dynamo cache limit of 8. Steady state is 8.2–8.9 ms per step (RTF 2.8–3.0×). Cold compilation with an
empty inductor cache is about 390 s; with a warm on-disk cache about 34 s. The four-phase schedule is
what makes cold compilation longer than it was, and that is the price of the correct sample rate.

**Where the real-time budget goes.** With car count fixed at 4 and one thing changed at a time:
going from 1 to 2 cars per race with the same policy driver costs +19.8 ms, essentially all of it
inside `sim.step` (opponents in each other's LiDAR plus car-car contact); switching that same race
to a teacher driver costs a further +20.0 ms, essentially all of it in the opponent phase. The
0.27× real-time figure is therefore two roughly equal costs, not the teacher alone.

---

## 5. Limits

- The full test suite has **not** been run to completion; verification was by explicit target lists.
- The IMU rate change, the auxiliary mask change and the ROS stamp changes all alter observations or
  training. Comparing against curves recorded before them is invalid without re-baselining.
- CUDA warmup seeded-equivalence is evidenced by the frontend's `h_w1_warmup_rng.json`, not by a
  test in this repository. `sim.py`'s inline comment is left deliberately conservative.
- The absolute phase of the scan sweep against a particular driver's header convention is not
  validated — only that the published duration matches the poses the beams were traced from.
- The read-only audit of `lidar`/`track`/`maps`/`teacher`/`raceline`/`mpc` inspected the functions
  listed in §3 and relied on existing tests elsewhere; the procedural generators in `track.py`, the
  raceline QP interior and `mpc.ilqr`'s convergence were not read line by line.
- `mpc.decode`'s docstring still says "(B,6)" and "(B,4)"; with `N_KNOTS = 6` the action is (B,8) and
  the knots (B,6). Documentation only, left for a later pass.
- No physical coefficient was refitted and no optimisation was implemented.

---

## 6. Files changed by this audit

Seventeen files: seven runtime, two ROS bridges, four updated tests, three new tests, this document.

Runtime: `f1sim/sim.py`, `f1sim/imu.py`, `f1sim/params.py`, `f1sim/gym_env.py`, `f1sim/learn/ppo.py`,
`f1sim/calib/bagread.py`, `f1sim/track.py`.

ROS: `f1sim_ros/bridge_node.py`, `f1sim_ros/vesc_sim_node.py`.

Tests updated (stale assertions that pinned the old behaviour): `tests/test_imu.py`,
`tests/test_gym_env.py`, `tests/test_episode_boundaries.py`, `tests/test_multicar.py`.

Tests added: `tests/test_sim_audit.py`, `tests/test_bagread_origin.py`,
`tests/test_ros_publisher_timing.py`.

Documentation added: this file.
