# A wheel that can slip: the rear-axle rotation state, and what the recordings say about it

2026-09-13, branch `feat/wheel-dynamics`. Companion to
[the traction guard's report](../../../real-car-tcs/REPORT.md) (§9 of which scoped this) and to
[`docs/real_data_calibration.md` §2.12](../real_data_calibration.md), which carries the measurements.

Until now the simulator could not spin or lock a wheel. `dynamics.py` carried no wheel rotation
state, `actuators.vesc_accel` closed its loop on the true body speed, and `odom.py` reported that
same body speed — so the simulated wheel speed **was** the body speed, the residual
`f1sim_ros/traction.py` keys on was identically zero, and there was nothing to train against. This
note is the model that changes that, every number in it, and what the 22 real recordings say about
whether it is right.

## 1. The model

`vehicle.wheel_model` selects it. Off, the simulator behaves as it did before; on:

```
kappa    = (omega_r * r_w - vx) / max(|vx|, v_slip_eps)          longitudinal slip ratio
Fx_rear  = mu * mu_r_scale * Fzr * pacejka(kappa, B_x, C_x, E_x) rear axle, its own load
Fx       = Fx_rear / drive_split_r                               4WD: the front makes the rest
I_w * omega_dot = m * a_cmd * r_w  -  r_w * Fx                   the axle
a_cmd    = (v_tgt - omega_r * r_w) / motor_tau                   the VESC loop, on the WHEEL
```

`Fx_rear` shares the existing friction circle with `Fy_r`, so hard throttle or hard braking in a
corner costs longitudinal force exactly where it should. `a_max` / `a_brake` stay the *commanded*
limits — the actuator model's interface is unchanged — and the motor torque is derived from them.

Two things about it are not obvious and both are load-bearing:

**The loop closes on the wheel.** That one line is what makes either failure possible: a spinning
wheel reads fast so the controller backs off, a locked one reads slow so it keeps driving. The
second half is visible in the recordings — the motor current is *positive* through the hardest brake
locks (`20260826-173704` t=46.5: −143 m/s² at +53 A). Closed on the body speed instead, the loop
fights every slip back to zero and a wheel state has nothing to do.

**The wheel equation is stiff.** Its time constant is `I_w * max(|vx|, v_slip_eps) / (r_w^2 *
dFx/dkappa)` — about 0.25 ms at a standstill against a 1 ms substep, so explicit Euler diverges.
`omega_r` is advanced semi-implicitly on the *local* slope of the tyre curve (`pacejka_slope`, in
closed form). Local, not peak: a locked or spinning tyre is past the peak where the true slope is
~0, and damping the wheel with the peak stiffness there would quietly suppress the transients the
whole model exists to produce.

## 2. Parameters, and what each one rests on

| parameter | value | what it rests on |
| --- | --- | --- |
| `r_w` | 0.055 m | **Measured, twice.** `imu.tire_d` = 0.11 m is the tyre diameter this platform's vibration model was already built on; independently, the recordings' own `speed_to_erpm_gain` (4202.7, below) implies an electrical-to-wheel ratio of 24.2 through `60/(2*pi*r_w)`, i.e. the stock ~8:1 drivetrain on a 6-pole motor. |
| `I_w` | 5.0e-4 kg m² | **The recordings, against a parts count that disagrees.** Parts: four wheels (~7.6e-4) plus a rotor reflected through ~11:1 (~4.4e-4) ≈ 1.2e-3. Recordings: a lock takes the wheel 5–7 m/s → 0 in 60–100 ms, i.e. 50–86 m/s² sustained, which against `m*a_brake*r_w` = 1.03 N m needs I ≤ 6e-4. The parts figure cannot produce locks the car demonstrably has. Matches the independent "order 5e-4" in the real-car report §9. DR range (0.6, 1.4) — deliberately not stretched to the parts value. |
| `drive_split_r` | 0.50 | **Derived from the platform.** Traxxas Slash 4x4: one motor, centre driveshaft through a slipper clutch, so the torque split is ~50/50 and fixed while the load split moves under braking. See §3. |
| `B_x, C_x, E_x` | 12.0, 1.50, 0.55 | **Assumptions**, textbook longitudinal values (peak at kappa ≈ 0.15; sliding plateau 0.71 of peak). A slip-ratio sweep needs a dynamometer or a wheel-speed sensor per corner and this car has neither. Wide DR ranges; what the recordings constrain is the outcome (§4), not the curve. |
| `v_slip_eps` | 0.50 m/s | **Assumption with a reason.** Half the guard's own `v_lock_min`, and a tenth of the slowest labelled slip event, so nothing this model is judged on happens inside the regularised region. |
| `odom.erpm_quantum` | 2.3794e-4 m/s | **Measured.** The smallest non-zero \|dv\| in each of the 12 competition bags, to within one float32 ulp. 1/q = 4202.7 is the VESC `speed_to_erpm_gain`. The 9 pre-competition bags ran 2.2149e-4 (gain 4514.9); the DR range spans both. |
| `odom.stamp_jitter_std` | 2.4 ms | **Measured.** 88 975 `/odom` steps: median 19.998 ms, p5 14.375, p95 25.570. A publish offset of sd 2.4 ms makes the step (a difference of two offsets) sd 3.39 ms, which reproduces both. |
| `odom.stamp_jitter_burst` | 0.0012 | **Measured, and it is a mechanism rather than a fitted tail.** 0.12 % of steps are shorter than 5 ms, down to 0.057 ms. A bounded Gaussian offset cannot produce those at all; a *catch-up publish* — a held message released microseconds after its predecessor while carrying a whole period of new wheel speed — is exactly the `20260827-111616` t=5.541240 / 5.541531 pair (0.291 ms, dv −0.032 m/s) that reads as −110 m/s². |
| `imu.shock_rate / _accel / _alpha` | 2.40 /s, 10.3 m/s², 1.60 | **Measured at the chain's output**, as the `vib_*` coefficients are. The car's raw accelerometer exceeds mu·g 0.291 times per second of motion, 20 m/s² 0.079, 50 m/s² 0.021, peaking at 120.4 — a power-law tail, which is why the magnitude is Pareto rather than Gaussian. Injecting the measured 0.29/s delivered 0.037/s after the 40 Hz low-pass and the 50 Hz sampling, a factor of 8 short; see §5. |
| `actuator.amp_per_nm` | 29.2 A/(N m) | **Measured at one point**: through the hardest 200 ms of braking the regen current sits at 95–100 % of the configured −30 A and the car reaches −4.2 to −5.7 m/s², so 30 A buys `m*a_brake*r_w` = 1.03 N m. Cross-check: the +63 A peak comes out as 10.5 m/s², above the sustained `a_max` of 7 as a launch transient should be. |

## 3. Why the single-track model had to learn about the front axle

The first version put every newton of drive and brake on a rear axle carrying 48 % of the weight.
That caps braking at `mu*Fzr/m` = 4.5 m/s², just under the 5.0 `a_brake` allows — so with a wheel
state, **every full brake command locked the wheel**, and the simulated slip rate came out 4.6× the
recordings'. Treating the drivetrain as rigidly sharing the whole car's grip (`Fx <= mu*m*g`)
removes the lock entirely, which is just as wrong.

What is actually true is a fixed torque division against a load division that moves:

```
                          static      braking at -5 m/s^2
  Fzr                     17.6 N      13.5 N
  rear demand at a_brake   9.35 N      9.35 N     (half of m*a_brake = 18.7 N)
  rear capacity, mu 1.05  18.5 N      14.2 N     -> 60 % margin, no lock
  rear capacity, mu 0.73  12.9 N       9.85 N    -> no margin: any corner, any bump, and it locks
```

That is the brake-lock mechanism on this car, and it is the reason the simulated lock rate now
lands where it does rather than being a parameter someone chose.

Two approximations are left in, both stated because both are in the optimistic direction: the front
axle's share of the longitudinal force is not subtracted from `Fyf` (a real 4WD car loses some front
grip under braking), and the front's own longitudinal capacity is not checked (under a hard launch
the load leaves the front, so `mu*mu_f_scale*Fzf` can fall below the share it is assumed to deliver
— which makes the simulated launch slightly stronger and its wheel spin slightly rarer than the
car's). The lateral balance this file was calibrated to is the rear axle's, and coupling the front
would move the understeer gradient the existing tests pin.

## 4. Acceptance against the recordings

Both columns are produced by `scripts/replay_traction.py` — the same labelling rule
(`evidence/wheelslip_bags.py`) and the same detector (`f1sim_ros/traction.py`, default thresholds),
pointed at two bag directories. The only difference between them is which car made the recordings.

```
python3 scripts/replay_traction.py --quiet --json real.json                       # the 22 bags
python3 scripts/gen_sim_bags.py --out /tmp/simnom --profile racepace --no-dr --seeds 3 --secs 60
python3 scripts/replay_traction.py --root /tmp/simnom --quiet --json simnom.json
python3 scripts/wheelslip_compare.py real.json simnom.json --markdown --driver /tmp/simnom
```

**Real reference, unchanged from the real-car report**: 22 bags, 1088 s of motion, 87 labelled runs
(22 with \|a_wheel\| ≥ 30), 46 firings, 43 hits (20 of 22 must-catch), 0 stationary and 0 cruising
firings, guard active 9.9 s = 0.9 % of motion.

### 4.1 The driver first

The event *rate* is only about the plant if the command process is the recordings'. `racepace` is
calibrated to it, and the comparison script checks that it is:

| statistic | recordings | simulated driver | ratio |
| --- | --- | --- | --- |
| cmd speed mean [m/s] | 3.560 | 3.565 | 1.00 |
| cmd speed sd [m/s] | 1.740 | 1.673 | 1.04 |
| cmd speed p50 [m/s] | 3.390 | 3.612 | 1.07 |
| cmd speed p95 [m/s] | 6.900 | 6.298 | 1.10 |
| cmd change over 250 ms, sd [m/s] | 1.170 | 1.171 | 1.00 |
| hard brake requests (−2 m/s in 250 ms) [1/s] | 0.291 | 0.230 | 1.27 |
| hard launch requests (+2 m/s in 250 ms) [1/s] | 0.280 | 0.233 | 1.20 |
| \|steer\| mean [rad] | 0.161 | 0.072 | 2.24 |
| steer sd [rad] | 0.189 | 0.125 | 1.52 |

The steering rows are half the recordings' by construction: the contract asks for straight *and*
cornering bags, so half of the eighteen steer at zero. The cornering half alone runs at the measured
sd (0.189 → \|steer\| mean 0.151 against 0.161).

A third profile was tried and rejected, and it is worth recording why. `--profile raceline` drives
the project's own `RacelineTeacher` around the two venues the recordings were made on
(`real:korea_2026_competition`, `real:map16x07`) with `label_grip = "nominal"`. It is the most
realistic *path*, and it produced almost no slip at all: **zero** hard brake or launch requests, a
250 ms command-change sd of 0.53 against the measured 1.17. The teacher's speed profile is
continuous; the stack that made the recordings steps its command. Matching the path is not the same
as matching the driver, and it is the driver that decides how often a wheel is asked for more than
it has.

### 4.2 The bands the contract names

18 bags, 1028 s of motion, mu pinned at 0.73 / 0.94 / 1.15 × straight / corner × 3 seeds, every
other parameter nominal.

| band | real runs | real /100 s | sim runs | sim /100 s | ratio | within 2x |
| --- | --- | --- | --- | --- | --- | --- |
| locks −40 … −143 (the contract's band) | 14 | 1.29 | 11 | 1.07 | **1.20** | yes |
| locks ≤ −30 (the must-catch line) | 21 | 1.93 | 16 | 1.56 | **1.24** | yes |
| all locks | 65 | 5.97 | 63 | 6.13 | **1.03** | yes |
| spins > +20 (the contract's band) | 11 | 1.01 | 19 | 1.85 | **1.83** | yes |
| all spins | 22 | 2.02 | 27 | 2.63 | **1.30** | yes |

Guard active: 0.9 % of motion on the car, **1.2 %** in the simulator.

The finer split, which is detail rather than verdict — at six buckets a real count of 6 against a
simulated 2 is Poisson noise (the 95 % interval on 2 covers 0.24–7.2), and reporting only this would
turn sampling noise into a result:

| peak \|a_wheel\| (lock) | real n | real /100 s | sim n | sim /100 s | ratio |
| --- | --- | --- | --- | --- | --- |
| 12–20 | 27 | 2.48 | 21 | 2.04 | 1.21 |
| 20–30 | 17 | 1.56 | 26 | 2.53 | 1.62 |
| 30–40 | 6 | 0.55 | 2 | 0.19 | 2.83 |
| 40–70 | 9 | 0.83 | 8 | 0.78 | 1.06 |
| 70–143 | 5 | 0.46 | 3 | 0.29 | 1.57 |
| 143+ | 1 | 0.09 | 3 | 0.29 | 3.18 |

| peak \|a_wheel\| (spin) | real n | real /100 s | sim n | sim /100 s | ratio |
| --- | --- | --- | --- | --- | --- |
| 12–20 | 11 | 1.01 | 8 | 0.78 | 1.30 |
| 20–30 | 10 | 0.92 | 3 | 0.29 | 3.15 |
| 30+ | 1 | 0.09 | 16 | 1.56 | 16.9 |

### 4.3 The guard on the simulated events

Same class, same default thresholds, no retuning: **38 firings, 37 hits, 0 stationary and 1
cruising firing** over 1028 s. The contract's two forbidden failure modes are (almost) absent — one
cruising firing in 1028 s against zero in the car's 1088 s.

It catches **5 of the 32** runs the label rule calls must-catch, where on the car it catches 20 of
22. That gap is real and it is the honest limitation of this model, so here is what it is:

* In the recordings the ≥ 40 m/s² population is **physical and sustained**. Measured over every
  labelled lock run: in the 40–70 bucket **100 %** were under a motor current above +20 A and 33 %
  had a raw accelerometer peak above 20 m/s²; in the 70+ bucket 83 % and **83 %**, with a median raw
  \|a_x\| peak of **64 m/s²**. Those are wheels being *obstructed* while the VESC drives through
  them — impacts — and they hold the wheel down for 60–240 ms, which the guard latches onto.
* In the simulator the same band is populated mostly by the **catch-up timestamp artefact**: one
  sample, physically nothing, and `TractionParams.min_diff_dt` exists precisely to reject it. The
  guard is doing the right thing; the label rule (a non-causal `np.gradient` on jittered stamps) is
  not, and it does not on the car either — the real-car report's two must-catch misses are the same
  artefact.
* What the simulator does **not** contain is the impact population, because an empty-floor recording
  has nothing to hit. Reproducing it needs a contact force at the wheel, which is a wheel-level
  contact model this contract does not ask for and which nothing here measures. Inventing a
  coupling coefficient to hit an acceptance number would be worse than reporting the gap.

So the distribution bar is met and the detection bar is met on the events the mechanism actually
produces. A policy trained here will see locks and spins at the right rate, with the right sensor
artefacts, and a guard that behaves the way it behaves on the car — but it will not see a wheel
stopped by a kerb.

### 4.4 With domain randomisation on

The same recordings with every other parameter drawn the way a training run draws it (the fleet,
not one car). Lock bands hold; spins overshoot.

| band | real /100 s | sim /100 s | ratio | within 2x |
| --- | --- | --- | --- | --- |
| locks −40 … −143 | 1.29 | 1.26 | 1.03 | yes |
| locks ≤ −30 | 1.93 | 2.12 | 1.10 | yes |
| all locks | 5.97 | 8.01 | 1.34 | yes |
| spins > +20 | 1.01 | 3.57 | **3.53** | NO |
| all spins | 2.02 | 5.12 | **2.53** | NO |

The whole overshoot is the 30+ spin bucket (28 runs against 1): the catch-up artefact lands on a
*rising* wheel speed and the label rule calls it a spin. The physical spin buckets are fine (12–20:
1.53, 20–30: 1.06). This is the same artefact as §4.3 seen from the other side, and it is the one
number in this note that a training run will be exposed to and that the recordings do not support.

### 4.5 The limit profile

`--profile limit` is the contract's explicit torture case — `stand → hard launch to 9 m/s → cruise →
hard brake to 0`, every 5 s, at each friction, straight and cornering. Its rates are a property of
the script (every cycle is a full-authority manoeuvre): **32.0 locks and 9.7 spins per 100 s**
against the recordings' 6.0 and 2.0, with the guard active 9.1 % of the time against 0.9 %. It is
reported for reach, not for rate: it reaches 70–143 at 1.25× the measured rate, and the guard
catches 68 of its 116 must-catch runs with 0 stationary and 4 cruising firings.

## 5. The IMU shock, fitted at the output

`traction.A_BODY_MAX` — the clamp the guard's hardest detection depends on — exists because the raw
accelerometer does not respect the friction bound. Training against a clean IMU would produce a
detector with no reason to clamp. The vibration model already here is stationary and never produces
an isolated spike, so this is a separate term: a Pareto-tailed, exponentially decaying impulse
train, gated on wheel motion by the same onset the vibration floor uses.

| threshold [m/s²] | 10.3 | 20 | 30 | 50 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| measured on the car [1/s of motion] | 0.291 | 0.079 | 0.045 | 0.021 | 0.005 |
| emulated [1/s of motion] | 0.281 | 0.096 | 0.041 | 0.013 | 0.004 |

Fitted at the *output* of the chain, as the `vib_*` coefficients are: injecting at the measured
0.29/s delivered 0.037/s, because a 5 ms pulse loses most of its height to the 40 Hz corner and most
of its firings to the 50 Hz sampling. The deep end stays ~1.6× light; fattening it further
overshoots the 20 m/s² bin, which is the one the clamp lives next to.

Only the x channel's rate was measured, so the split across the three accelerometer axes
(1.0 / 0.7 / 0.7) is a guess, and the gyro is left alone entirely — a real impact shocks it too, but
nothing here measures by how much.

## 6. The `tcs` arm's cost

The guard is host Python over scalars: one `update` + `shape` pair per car per control step, plus
one device→host transfer to fetch its four inputs and one back to return the shaped speed. Nothing
about it is inside `sim._roll`, which is the only thing `viewer/graph_fastpath.py` captures, so it
is outside the CUDA-graph fastpath **by construction**. The two transfers are a synchronise per
step, and on the graphs backend that — not the Python — is the cost.

Measured on CPU, 32 cars, `shape` alone (the test that pins it is
`test_the_arm_costs_what_it_is_documented_to_cost`): well under 20 ms per call, i.e. under 0.6 ms
per car per step including both transfers. A GPU measurement, where the synchronise is what matters,
has not been taken: the GPU was held by a training run for the whole of this session.

## 7. Known gaps

1. **The odometry runs at the control rate.** The car publishes `/odom` at 50 Hz; the simulator
   publishes one sample per 40 Hz control step. The publish-offset *shape* is the measured one, but
   what that puts under the guard's absolute 15 ms window depends on the period: 0.2 % of simulated
   steps against 7.2 % of the car's. Closing it means sampling the wheel speed inside the substep
   loop on the sensor's own clock, the way `imu.sample_schedule` already does — a change to the
   odometry contract that `gym_env` and `obs.py` read, which is why it is not in this branch.
2. **No wheel-level contact.** §4.3. The ≥ 40 m/s² population on the car is impact-dominated and
   the simulator cannot contain it on an empty floor.
3. **The longitudinal slip curve is assumed.** `B_x`, `C_x`, `E_x` have textbook values and wide DR
   ranges. Nothing in the recordings measures them directly.
4. **`I_w` disagrees with its own parts count** by a factor of 2.4 (§2). The recordings decide, and
   the disagreement is recorded rather than averaged.
5. **The 30+ spin bucket overshoots under DR** (§4.4), and it is artefact rather than physics.
6. **The front axle is not load-checked longitudinally** (§3), which makes launches slightly
   stronger and launch spin slightly rarer than the car's.
