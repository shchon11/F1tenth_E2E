"""Parameter dataclasses for vehicle, actuators, sensors and domain randomization.

Defaults target the standard F1TENTH platform (Traxxas-based chassis, VESC 6,
Hokuyo UST-10LX 2D LiDAR, Jetson AGX). Every field of VehicleParams,
ActuatorParams, LidarParams and OdomParams can be randomized per-environment
via RandomizationConfig (see randomization.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Tuple


@dataclass
class VehicleParams:
    # Mass / geometry (f1tenth_gym defaults, measured on the reference car)
    m: float = 3.74            # [kg]
    Iz: float = 0.04712        # [kg m^2] yaw inertia
    lf: float = 0.15875        # [m] CoG -> front axle
    lr: float = 0.17145        # [m] CoG -> rear axle
    h: float = 0.074           # [m] CoG height (for longitudinal load transfer)
    width: float = 0.31        # [m] overall width (collision footprint)
    length: float = 0.58       # [m] overall length (collision footprint)
    # Tire model: Pacejka magic formula, Fy = mu*Fz*D*sin(C*atan(B a - E(B a - atan(B a))))
    mu: float = 1.0489         # peak friction coefficient
    mu_f_scale: float = 0.92   # front/rear grip asymmetry -> understeer at the limit (front saturates first)
    mu_r_scale: float = 1.0
    # B*C ~ cornering stiffness per unit load [1/rad]; f1tenth_gym measured C_Sf=4.72, C_Sr=5.46,
    # peak slip angle ~ 2.57/B rad  (B=8 -> 18 deg, B=9 -> 16 deg). Front softer -> mild understeer.
    B_f: float = 8.0           # stiffness factor, front
    C_f: float = 1.3           # shape factor, front
    E_f: float = 0.97          # curvature factor, front
    B_r: float = 9.0
    C_r: float = 1.3
    E_r: float = 0.97
    # Kinematic <-> dynamic blend: below v_blend_min pure kinematic, above v_blend_max pure dynamic
    v_blend_min: float = 0.8   # [m/s]
    v_blend_max: float = 2.0   # [m/s]
    # Limits
    s_max: float = 0.4189      # [rad] max steering angle (24 deg)
    sv_max: float = 3.2        # [rad/s] max steering rate (servo)
    v_max: float = 12.0        # [m/s]
    v_min: float = -3.0        # [m/s] reverse limit
    a_max: float = 7.0         # [m/s^2] motor accel limit (VESC current limit); traction limit applies on top
    a_brake: float = 5.0       # [m/s^2] max braking decel. Braking on this car is limited by the
                               # VESC regen current, not by the tyres: at the hardest 200 ms of
                               # braking in each recording the regen current sits at 95-100 % of
                               # that recording's configured minimum, and the deceleration reached
                               # is -4.2 to -5.7 with the competition setting (-30 A) and only
                               # -1.7 to -2.3 with the pre-competition one (-5 A). Since 5 < mu*g
                               # the actuator binds first, so simulated braking stops depending on
                               # the surface -- as it does on the car. The 12.0 this replaces let
                               # the tyre limit bind instead, and taught the policy it could scrub
                               # speed twice as fast as the real car can.
                               # (-9.2 and -10.6 also appear in the recordings, but the motor
                               # current is *positive* at those instants: they are impacts.)
    v_switch: float = 7.319    # [m/s] above this, accel scales with v_switch/v (power limit)
    # ---- rear axle as a rotating body (2026-09-13) ----------------------------------------
    # `wheel_model` off reproduces the model that was here before: rear longitudinal force set
    # straight from the commanded acceleration, wheel speed == body speed, so the wheel can neither
    # spin nor lock. On, the parameters below are live. See dynamics.py and
    # docs/research/wheel-model-2026-09-13.md.
    wheel_model: bool = True
    drive_split_r: float = 0.50
                               # fraction of the drivetrain's torque delivered to the REAR axle.
                               # The F1TENTH platform is a Traxxas Slash 4x4: one motor, a centre
                               # driveshaft through a slipper clutch, so the split is essentially
                               # 50/50 and fixed by the drivetrain -- while the *load* split is not.
                               # That is the whole mechanism behind a brake lock on this car. Under
                               # braking the load moves forward (Fzr falls from 17.6 to ~13.5 N at
                               # -5 m/s^2), so the rear axle is asked for half the force on 37 % of
                               # the weight and lets go first, taking the single ERPM speed with it.
                               # The arithmetic it produces is what the recordings show:
                               # m*a_brake*r_w = 1.03 N m needs 18.7 N total, 9.35 N at the rear,
                               # against a rear capacity of mu*Fzr = 14.2 N at mu 1.05 (60 % margin,
                               # no lock) and 9.85 N at mu 0.73 (no margin -- so any cornering, any
                               # bump, and it locks). Set it to 1.0 for a rear-drive car.
    r_w: float = 0.055         # [m] rear wheel rolling radius. MEASURED twice over: `imu.tire_d`
                               # 0.11 m is the tyre diameter this platform's vibration model was
                               # already built on, and the recordings' own VESC calibration agrees
                               # -- `speed_to_erpm_gain` comes out of the bags as 4202.7 ERPM per
                               # m/s (competition) and 4514.9 (pre-competition), and
                               # gain = 60/(2*pi*r_w) * (gear ratio * pole pairs) puts that ratio at
                               # 24.2 and 26.0, i.e. the stock ~8:1 drivetrain on a 6-pole motor.
    I_w: float = 5.0e-4        # [kg m^2] rotational inertia of the whole driven assembly, referred
                               # to the wheel. It sets how fast a wheel can let go:
                               # a_wheel = r_w*(T - r_w*Fx)/I_w.
                               #
                               # Two estimates, and they disagree, so both are stated. From PARTS:
                               # four wheels at ~90 g on 0.055 m are ~7.6e-4 together, and a
                               # 3650-class rotor (~4e-6 kg m^2) reflected through the ~11:1 total
                               # reduction adds ~4.4e-4, so a 4x4 drivetrain is of order 1.2e-3.
                               # From the RECORDINGS: a lock takes the wheel from 5-7 m/s to zero in
                               # 60-100 ms (20260826-173704 t=46.5, 7.65 -> 0.00 in 100 ms;
                               # 20260826-194702 t=9.5, 5.05 -> 0.00 in 59 ms), i.e. 50-86 m/s^2
                               # sustained, and against the m*a_brake*r_w = 1.03 N m the regen limit
                               # allows -- less the sliding tyre's own torque -- that needs I <= 6e-4.
                               # The parts figure cannot produce the locks the car demonstrably has.
                               # 5.0e-4 is the recordings' number and matches the independent
                               # estimate in ../real-car-tcs/REPORT.md section 9 ("order 5e-4"); the
                               # gap is a real limitation, recorded in
                               # docs/research/wheel-model-2026-09-13.md rather than averaged away.
                               # The DR range (0.6, 1.8) spans 3e-4 to 9e-4.
    # Longitudinal magic formula, the same normalized shape `pacejka()` uses for the lateral axis.
    # ASSUMPTIONS with textbook ranges -- a slip-ratio sweep needs a dynamometer or a wheel-speed
    # sensor per corner, and this car has neither. What the recordings *do* pin is the outcome:
    # see the acceptance table in docs/research/wheel-model-2026-09-13.md.
    B_x: float = 12.0          # stiffness factor: peak longitudinal force at kappa ~ 0.15, the
                               # usual value for rubber on a hard smooth floor
    C_x: float = 1.50          # shape factor (Pacejka '89 longitudinal 1.4-1.8); with E_x below it
                               # puts the full-slide plateau at 0.71 of peak
    E_x: float = 0.55          # curvature factor (longitudinal 0.4-0.8)
    v_slip_eps: float = 0.50   # [m/s] the slip ratio's denominator is held at this from below. It
                               # plays the part `v_blend_min` plays laterally: kappa is a ratio to
                               # the ground speed and stops meaning anything as that goes to zero.
                               # 0.5 m/s is half the guard's own `v_lock_min` and a tenth of the
                               # slowest labelled slip event in the recordings, so nothing this
                               # model is judged on happens inside the regularised region.
    # Rolling resistance + aero drag (decel = c_roll + c_drag * v^2)
    c_roll: float = 0.1        # [m/s^2]
    c_drag: float = 0.01       # [1/m]
    # Sprung-mass attitude (what tilts the LiDAR plane): 2nd-order response to body accelerations,
    # plus a random tilt the floor and tyres put in. MEASURED 2026-09-13 from the competition bags
    # by the gyro route (docs/real_data_calibration.md §6.1): integrate the roll / pitch rate,
    # band-pass 0.3-1.5 Hz, bin against the accelerometer's own lateral / longitudinal force; the
    # estimator recovers ~0.9x (roll) / ~0.65x (pitch) of a known gain on the simulator, and the
    # numbers below are corrected by that. Five clean recordings agree within +-20 %.
    roll_per_g: float = 0.03   # [rad/g] 1.7 deg/g; measured 1.5-2.1. Was 0.10 (a guess): 3x too much roll
    pitch_per_g: float = 0.03  # [rad/g] SQUAT under throttle (ax > 0), ~1.7 deg/g; measured 1.5-2.5
    dive_per_g: float = 0.008  # [rad/g] brake DIVE (ax < 0), ~0.5 deg/g: the recordings show almost none
                               # (0-0.7 deg/g) at the -0.45 g the regen limit allows. The earlier LiDAR
                               # floor-strike bracket (5.5-11.8 deg/g) was confounded, as it said.
    # [rad] rms of a random roll / pitch added to the suspension's set point -- OFF. It was 0.017
    # (2026-09-13, from the gyro: 0.8-1.2 deg roll / 0.6-1.8 deg pitch rms driving straight at steady
    # speed, docs/real_data_calibration.md SS6.1a), an Ornstein-Uhlenbeck process in *time*. Nothing
    # in the car produced it, so a car at a constant 3 m/s on a flat floor pitched 1.10 deg rms and
    # 4.09 deg at peak -- the 0.110 m scan plane on the floor 1.6 m ahead -- against 0.018 / 0.11
    # deg from the dynamics alone. The user, 2026-09-21: "동역학에 기반에서 센서 시뮬레이션이 되어야
    # 한다니까? ... 그냥 등속도로 앞으로 가고 있는데에도 스캔이랑 차량이 왜자꾸 출렁거려?" The body
    # now tilts only because the dynamics (a_x, a_y, a contact) tilt it through the springs, and the
    # IMU reads that. A floor that does tilt the car belongs under the wheels as a fixed height map --
    # the same bump in the same place every lap -- not as noise in time; that is not modelled.
    # Left as a knob (the process below is skipped at 0) only so the old behaviour can be reproduced.
    road_tilt: float = 0.0
    # [m/s] the speed at which that wobble reaches `road_tilt`; below it the excitation ramps down
    # linearly to nothing at a standstill. `road_tilt` was measured *driving*, and a floor does not
    # move under a parked car -- but the OU process ran at full amplitude regardless, so a car
    # sitting at 0.000 m/s pitched 1.1 deg rms and 5.5 deg at peak, which puts a 0.110 m scan
    # plane on the floor 1.1 m ahead. The user saw exactly that in the viewer: "움직이지도 않는데
    # 라이다가 이리저리 땅 봤다가". OURS, not measured: three attempts to recover the speed
    # dependence from the recordings (accelerometer tilt, forward-range stability at standstill)
    # were each swamped by something else -- vibration at speed, people and cars crossing in front
    # of the grid -- so the ramp is a first-principles choice and says so.
    road_tilt_v: float = 2.0
    road_tau: float = 0.4      # [s] its correlation time (the roll-rate spectrum is flat above ~0.4 Hz)
    susp_wn: float = 17.6      # [rad/s] suspension natural frequency (~2.8 Hz)
    susp_zeta: float = 0.35    # [-] damping ratio (underdamped: visible overshoot after braking)


@dataclass
class ActuatorParams:
    # Steering servo: first-order lag + rate limit + angle bias
    servo_tau: float = 0.04        # [s] time constant
    steer_bias: float = 0.0        # [rad] mechanical trim error (randomized)
    steer_gain: float = 1.0        # command scaling error
    # VESC speed loop: first-order tracking of commanded speed via PID-like accel
    motor_tau: float = 0.20        # [s] time constant of speed response
    speed_gain: float = 1.0        # ERPM<->m/s calibration error (randomized)
    amp_per_nm: float = 29.2       # [A per N m at the wheel] motor current per unit drive torque,
                                   # used only to emulate `/sensors/core` `current_motor` for the
                                   # traction guard's optional spin gate. MEASURED at the one point
                                   # the recordings pin it: through the hardest 200 ms of braking
                                   # the regen current sits at 95-100 % of the configured -30 A and
                                   # the car reaches -4.2 to -5.7 m/s^2, so 30 A buys
                                   # m * a_brake * r_w = 1.03 N m. Cross-check on the drive side:
                                   # the +63 A peak in the recordings comes out as 10.5 m/s^2, above
                                   # the sustained a_max of 7 as a launch transient should be.
    # Command latency (sensor -> policy -> actuator), applied as delay buffer on commands
    cmd_delay: float = 0.015       # [s] LiDAR -> policy -> VESC on the Jetson. Calibrated against the
                                   # *total* command-to-yaw lag, which is what a recording can show:
                                   # cross-correlating servo command with yaw rate peaks at 100 ms on
                                   # the car. That total also contains servo slew, tyre force build-up
                                   # and yaw inertia, all modelled separately here, so it must not be
                                   # assigned to this term wholesale. See calib/steering.py.
    cmd_delay_jitter: float = 0.0  # [s] uniform jitter per step


@dataclass
class LidarParams:
    # Hokuyo UST-10LX: 1080 beams over 270 deg, 40 Hz, 10 m usable (30 m for UST-30LX)
    n_beams: int = 1081            # the recordings carry 1081: 270 deg / 0.25 deg + 1, endpoints included
    fov: float = 4.71238898        # [rad] 270 deg
    range_max: float = 10.0        # [m]
    range_min: float = 0.02        # [m]
    rate: float = 40.0             # [Hz]
    # From /tf_static in the recordings: base_link -> laser is (0.2970, 0, 0.1100) with no rotation,
    # and the tf tree is map -> odom -> base_link with z = 0 throughout, so base_link is on the
    # ground and 0.110 m is the scan plane height. The 0.15 this replaces was a guess, 36 % high --
    # which matters twice over: a lower scanner sees less over an obstacle, and where the boundary
    # is duct hose it cuts the pipe at a different height, so the gap the policy reads is not the
    # gap the car has.
    mount_x: float = 0.297         # [m] forward of base_link (rear axle center)
    mount_y: float = 0.0
    mount_z: float = 0.110         # [m] scan plane height above the floor at rest
    mount_yaw: float = 0.0         # [rad] mounting misalignment (randomized)
    mount_roll: float = 0.0        # [rad] sensor / platform not level (randomized)
    mount_pitch: float = 0.0       # [rad]
    # Measured per beam over 5700 scans held genuinely still across 7 recordings: the MAD of each
    # beam's own range, binned by distance, Theil-Sen over the bins whose beams agree with each
    # other (p90/median < 4). Distance dependence is weak and only marginally significant:
    #   MAD(r) = 7.4 mm + 0.99 mm/m * r     (slope 95% CI 0.0-2.2 mm/m)
    # An earlier fit here read 8.9 mm at 0-2 m rising to 50.4 at 5-8 and set 0.55 %/m from it, but
    # that pooled every beam in a bin, so the few straddling an edge or a passing person carried
    # it -- the same mistake, in the same direction, as the 1838 mm "noise" before that. Per beam
    # the 5-8 m bins read 10-12 mm. The old model put 59 mm on a 10 m beam against a measured 17,
    # and the far beams are exactly the ones a corner is read from.
    # Short range is quantisation-limited: raw MADs land on integer millimetres (the driver reports
    # mm), so the 7.4 mm floor is an upper bound on the true noise there.
    noise_std: float = 0.0074      # [m] gaussian range noise floor
    noise_std_rel: float = 0.0010  # relative noise (fraction of range)
    dropout_prob: float = 0.0001   # per-beam probability of no return. Measured 0.012 % of beams are
                                   # isolated no-returns; the 1.3 % of beams reading the 65.533 m
                                   # sentinel are mostly genuinely out of range, not dropouts.
    spike_prob: float = 0.0        # per-beam probability of random spurious return
    motion_distortion: bool = True # emit beams over the 1/rate sweep while the car moves
    car_model: str = "mesh"        # what other cars are to the LiDAR: "mesh" = outlines sliced from the
                                   # viewer's own f1tenth_car.glb at each height; "parts" = the older
                                   # hand-placed boxes (chassis, deck, wheels)
    dropout_value: float = 65.533  # what a no-return reads as. The urg_node driver on this car emits
                                   # the 0xFFFF mm sentinel, not inf, on every recording checked;
                                   # obs.py clamps range/range_max to 1.0 either way, but the raw
                                   # value has to match for anything reading scans directly.
    # incidence-dependent returns (randomized): a beam grazing a glossy hall floor gives no return
    # far more often than one hitting it steeply; a duct hose seen edge-on reflects too little.
    duct_scale: float = 1.0        # per-env multiplier on the track's duct hose diameter
    floor_dropout: float = 0.6     # no-return probability of a floor hit at grazing incidence
    floor_graze: float = 0.20      # [rad] incidence angle below which floor returns fade out
    # Measured on the competition track boundary: beams were cast against the recorded /map from the
    # particle-filter pose (only recordings whose pose tracks the gyro to r > 0.9) and the no-return
    # rate binned by |cos| to the boundary normal, over 107k rays in three recordings:
    #   |cos|  0.075  0.225  0.375  0.525  0.70  0.90
    #   miss   8.5 %  5.4 %  1.0 %  0.7 %  0.3 %  0.3 %
    # so grazing incidence raises the miss rate about 28x over head-on, fading out by |cos| ~ 0.4.
    # The 0.3 that this replaces put 24 % on the most grazing bin -- three times what the boundary
    # actually does, i.e. the simulated policy lost sight of the track edge far more often than the
    # car does. An upper bound: the ~0.3 % head-on floor includes map and pose error and the people
    # and other cars that the map does not contain, and some of that leaks into the grazing bins.
    duct_graze_dropout: float = 0.10  # no-return probability of a duct hit seen edge-on
    duct_graze_cos: float = 0.40   # |cos(angle to the hose normal)| below which duct returns fade


@dataclass
class OdomParams:
    """VESC-derived odometry (vesc_to_odom): speed from ERPM, yaw rate from
    commanded steering angle. No IMU by default (optional imu_* fields for later)."""
    speed_noise_std: float = 0.05      # [m/s]
    speed_scale_err: float = 0.0       # multiplicative error, randomized (erpm gain)
    steer_offset: float = 0.0          # [rad] residual servo-offset calibration error (randomized)
    steer_gain_err: float = 0.0        # residual servo-gain calibration error (randomized)
    yaw_rate_noise_std: float = 0.02   # [rad/s]
    # ---- ERPM channel artefacts (live only with vehicle.wheel_model) -----------------------
    # MEASURED over the 22 recordings' 88 975 /odom steps; see
    # docs/real_data_calibration.md "The ERPM channel".
    erpm_quantum: float = 2.3794e-4    # [m/s] one ERPM step. The smallest non-zero |dv| in each
                                       # competition bag lands here to within one float32 ulp, and
                                       # 1/q = 4202.7 is the VESC `speed_to_erpm_gain`. The nine
                                       # pre-competition bags ran a different gain, 4514.9
                                       # (q = 2.2149e-4); the DR range spans both. It is small next
                                       # to `speed_noise_std`, and it is here because it is real and
                                       # because a detector trained on a continuous wheel speed has
                                       # no reason to expect a lattice.
    stamp_jitter_std: float = 0.0024   # [s] sd of the publish offset of an /odom sample against the
                                       # nominal grid. The measured step distribution is median
                                       # 19.998 ms with p5 14.375 and p95 25.570; a Gaussian offset
                                       # of 2.4 ms reproduces both (the step is a difference of two
                                       # offsets, so its sd is 2.4*sqrt(2) = 3.39 ms) and puts
                                       # 7.1 % of steps under 15 ms against a measured 7.22 %.
    stamp_jitter_burst: float = 0.0012 # fraction of samples published immediately after the
                                       # previous one ("catch-up") instead of on the grid. The
                                       # measured rate: 0.12 % of the recordings' 88 975 steps are
                                       # shorter than 5 ms, down to 0.057 ms. A Gaussian offset
                                       # bounded by half a period cannot produce those at all, and
                                       # they are the steps that make one ERPM quantum read as tens
                                       # of m/s^2 -- the artefact `TractionParams.min_diff_dt`
                                       # exists for (real-car REPORT.md section 5).


@dataclass
class ImuParams:
    """VESC 6 built-in 6-axis IMU (BMI160 class) on the lower chassis, published by vesc_driver.
    Modelled per physics substep: specific force at the sensor (gravity leaking in through body
    roll/pitch, lever arm from the CoG), speed-proportional vibration at wheel/motor frequencies,
    the sensor's internal low-pass, sampling at `rate`, bias + random walk, white noise,
    quantization, and the VESC's own attitude estimate (complementary/Mahony-style filter)."""
    enabled: bool = True
    # [m/s^2] full scale of the accelerometer. A BMI160-class part is configured to +-16 g and
    # clips there; the 22 recordings peak at 12.6 g, on the hardest impact in them (8.14 m/s to a
    # standstill in 20 ms), so 16 g is above everything the car has actually produced and below
    # what an unclamped contact would emit -- a 3 m/s change inside one 2.5 ms substep is 122 g.
    accel_range: float = 16.0 * 9.80665
    # Measured on the car: median inter-sample dt over all 22 recordings is 50.00 Hz (per-bag spread
    # 49.91-50.10), not the 100 Hz vesc_tool default this used to assume.
    #
    # This rate is now delivered, not just requested. It used to be neither: `sample_indices` took
    # floor(imu_rate * control_dt) = floor(1.25) = 1 sample per control step, so a 50 Hz sensor came
    # out at 40 Hz, and the emulator was additionally told 1/50 s had passed between samples that
    # were 1/40 s apart -- so the bias random walk and the attitude filter ran on a clock 20 % slow.
    #
    # `imu.sample_schedule` now follows the sensor's own clock: it works out the exact cycle in
    # which the two clocks realign (four control steps here) and emits 1, 1, 1, 2 samples across it,
    # which averages 50 Hz exactly. Each control step therefore carries a varying number of samples,
    # and `StepResult.imu_offsets` says how long before the step's end each was taken.
    #
    # The cost is one compiled graph per phase of that cycle (four at these rates), which is why
    # `sample_schedule` refuses a rate needing a long cycle rather than thrashing the compile cache.
    imu_rate: float = 50.0         # [Hz] sensor clock; delivered exactly, see imu.sample_schedule
    bandwidth: float = 40.0        # [Hz] sensor low-pass (BMI160 ~ODR/2.5)
    # mounting: VESC on the chassis, mid-car, left of the spine; base_link frame
    imu_x: float = 0.16
    imu_y: float = 0.05
    imu_z: float = 0.045
    imu_roll: float = 0.0          # [rad] misalignment (randomized)
    imu_pitch: float = 0.0
    imu_yaw: float = 0.0
    # noise (per sample, after the low-pass), bias, drift
    gyro_noise: float = 0.0031     # [rad/s] rms. Same estimator: measured 0.0025 / 0.0043 / 0.0021
                                   # per axis over 12 recordings with an intact gyro (the five
                                   # GYRO-BAD ones read 0.07-0.40 and are excluded).
    accel_noise: float = 0.018     # [m/s^2] rms. From std(diff(x))/sqrt(2) over the truly-still
                                   # windows of 17 recordings, which measures the white component
                                   # alone; the 0.048 this replaces was a plain standard deviation
                                   # over the window, so it also counted the slow tilt/thermal drift
                                   # (total sd 0.031) and a few windows that still held handling.
    gyro_bias_x: float = 0.0       # [rad/s] zero-rate offset (randomized +-3 deg/s)
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0
    gyro_bias_walk: float = 0.0005 # [rad/s/sqrt(s)] random walk of the gyro bias
    accel_bias_x: float = 0.0      # [m/s^2] (randomized +-0.3)
    accel_bias_y: float = 0.0
    accel_bias_z: float = 0.0
    quant_gyro: float = 0.00107    # [rad/s] LSB at +-2000 deg/s, 16 bit
    quant_accel: float = 0.0048    # [m/s^2] LSB at +-16 g, 16 bit
    # vibration from tires/drivetrain, grows with speed; wheel freq = v / (pi*tire_d), motor = wheel*gear
    # Fitted to the *output* of the IMU chain (see imu.py): rms of the emulated IMU residual above
    # 5 Hz, against the same statistic measured on the car over 22 recordings.
    #   real accel rms  0.018 m/s^2 still ->  1.42 at 0.5 m/s ->  3.3 at 7.5 m/s
    #   real gyro  rms  0.0025 rad/s      ->  0.150           ->  0.41
    # Least squares on the squared rms (floor and slope drive independent noise, so they add in
    # quadrature). Sim vs car after the fit: -11 % at a standstill, +8 % at 0.5 m/s, within 4 % from
    # 1.5 to 7.5 m/s on both channels.
    vib_onset_v: float = 0.30      # [m/s] wheel speed over which the floor ramps in from zero
    vib_accel_floor: float = 5.84  # [m/s^2] broadband, amplitude once the wheels are turning
    vib_gyro_floor: float = 0.608  # [rad/s]
    vib_accel: float = 1.223       # [m/s^2 per m/s] tonal slope on top of the floor
    vib_gyro: float = 0.165        # [rad/s per m/s]
    vib_broadband: float = 0.4     # fraction of vib amplitude that is broadband (white) instead of tonal
    tire_d: float = 0.11           # [m]
    gear_ratio: float = 8.0        # motor revs per wheel rev (Slash 4x4 stock ~ 8-10)
    # ---- impact / shock (live only with vehicle.wheel_model) -------------------------------
    # The vibration model above is stationary: it never produces the isolated spikes the
    # recordings are full of, and `traction.A_BODY_MAX` -- the clamp the whole guard depends on --
    # exists *because* of those. MEASURED over 1088 s of motion in the 22 bags: |a_x| exceeds
    # mu*g (10.3) 0.291 times a second, 20 m/s^2 0.079, 30 m/s^2 0.045, 50 m/s^2 0.021, 100 m/s^2
    # 0.005, peaking at 120.4. That tail is close to a power law, rate ~ A^-1.75, so the magnitude
    # is drawn as a Pareto variate rather than a Gaussian one -- a Gaussian fitted to the 20 m/s^2
    # rate would put nothing at all above 50.
    #
    # Both coefficients are fitted to the OUTPUT of this chain, as the `vib_*` ones are, because a
    # 5 ms pulse loses most of its height to the 40 Hz corner and most of its firings to the 50 Hz
    # sampling: injecting at the measured 0.29/s delivered 0.037/s, a factor of 8 short. Fitted
    # against all five measured thresholds (scratch fit, 96 envs x 12.5 s at 4 m/s):
    #   threshold          10.3     20     30     50    100 m/s^2
    #   measured, /s      0.291  0.079  0.045  0.021  0.005
    #   emulated, /s      0.281  0.096  0.041  0.013  0.004
    # The deep end is ~1.6x light, and that is where it stays: fattening it further (a lower
    # `shock_alpha`) overshoots the 20 m/s^2 bin, which is the one the guard's clamp actually lives
    # next to.
    shock_rate: float = 2.40       # [1/s of motion] rate of injected impacts
    shock_accel: float = 10.3      # [m/s^2] scale of the drawn magnitude (its minimum). mu*g, so
                                   # the smallest modelled shock is exactly the one that starts to
                                   # matter -- anything under it is already covered by `vib_*`.
    shock_alpha: float = 1.60      # Pareto tail exponent. The injected tail is a little fatter than
                                   # the measured one (1.75, from the rate falling 0.291 -> 0.005 /s
                                   # between 10.3 and 100 m/s^2) because the filter eats the short
                                   # spikes hardest.
    shock_tau: float = 0.005       # [s] decay of one impact. A chassis impact is not a single
                                   # sample: it rings, and a delta shorter than the sensor's 40 Hz
                                   # low-pass would be filtered away before it was ever sampled.
                                   # 5 ms keeps a spike visible at 50 Hz, which is how the real
                                   # ones survive to reach the guard.
    # VESC attitude filter (Mahony-like): gyro integration corrected toward the accelerometer
    # gravity direction with time constant ahrs_tau (~1/kp, VESC default kp 0.3); the correction is
    # down-weighted when |accel| deviates from g (VESC accel_confidence_decay), so sustained
    # accelerations still bend the estimate but less than a naive complementary filter would.
    ahrs_tau: float = 3.0          # [s]
    ahrs_accel_decay: float = 1.0  # weight = clamp(1 - | |a|/g - 1 | * decay, 0, 1)


@dataclass
class SimParams:
    physics_dt: float = 0.001     # [s] substep
    control_rate: float = 40.0    # [Hz] policy / lidar rate (one env step)
    terminate_on_collision: bool = True
    # How a contact behaves, per boundary material. The track already classifies every occupied
    # cell as duct hose or tall wall (`Track.duct` / `Track.tall`, with distance fields for both),
    # and they do not behave alike: a hose deforms and absorbs, a wall does not.
    #
    # Measured on the 56 impacts above 1 m/s in the competition recordings, whose boundary is duct
    # hose (`f1sim/calib/impacts.py`):
    #
    #     speed kept, v_after/v_before : p10 0.00   median 0.52   p90 0.77
    #     peak |a|                     : p10 2.74 g median 4.30 g max 12.62 g
    #     contact duration             : p10 0.020 median 0.040 s p90 0.350 s
    #     came to a full stop          : 14 of 56
    #
    # Two things follow, and both were wrong before. A duct does not bounce you -- the lower tail
    # is 0.00 and a quarter of the impacts ended at a standstill -- so its restitution is near
    # zero, not the 0.2 that was applied to everything. And a real contact lasts about 40 ms,
    # sixteen substeps, while this simulator removed the whole normal velocity inside one: the
    # hose stretches and the car stops *gradually*, which is the thing to reproduce.
    #
    # `contact_tau_*` is that: the time constant the into-surface velocity is bled off over, so a
    # contact takes about 2 tau. The duct's is set from the measured median; a tall wall keeps the
    # old near-instant behaviour because nothing in the recordings measures one.
    collision_restitution: float = 0.2   # velocity kept along a TALL WALL on contact
    collision_restitution_duct: float = 0.05    # measured: a hose absorbs, it does not rebound
    contact_tau_wall: float = 0.004      # [s] a tall wall: a few ms, near the old instant reset
    contact_tau_duct: float = 0.020      # [s] half the measured 40 ms median duration
    wall_friction: float = 0.5
    device: str = "cuda"
    compile: bool = True          # torch.compile the physics substep loop on CUDA
    compile_mode: str = "default" # "reduce-overhead" = CUDA graphs: one launch per control step (viewers next to a training job)
    seed: int = 0


# name -> (low, high) uniform range, or (mean, std, 'n') normal, applied per env on reset.
RandRange = Tuple[float, float]


@dataclass
class RandomizationConfig:
    """Uniform ranges per parameter. Keys use '<group>.<field>' naming.
    Values are (low, high) absolute for additive fields, and multiplicative
    (low, high) scale for fields listed in `scale_fields`."""
    enabled: bool = True
    ranges: Dict[str, RandRange] = field(default_factory=lambda: {
        # 2026-09-07: ranges for a small car with a fast servo and a 10-30 ms LiDAR->policy->VESC pipeline
        # on venue floors (mu 0.7-1.1); the earlier extremes (mu 0.55, 100 ms delay) only taught caution
        "vehicle.mu": (0.70, 1.10),
        "vehicle.m": (0.90, 1.10),
        "vehicle.Iz": (0.8, 1.2),
        "vehicle.B_f": (0.8, 1.2),
        "vehicle.B_r": (0.8, 1.2),
        "vehicle.a_max": (0.6, 1.2),
        "vehicle.a_brake": (0.70, 1.15),        # -3.5 to -5.75. The -5 A regen configuration in the
                                                # pre-competition recordings only reaches -2.3 and is
                                                # deliberately outside this: it is a VESC setting, not
                                                # a property of the car, and nobody would race it. Set
                                                # the regen limit to the competition value before
                                                # deploying, or the policy will brake later than the
                                                # car can.
        "vehicle.mu_f_scale": (0.85, 1.0),
        # The wheel model's own parameters. `I_w` and the slip curve are assumptions (see
        # VehicleParams), so their ranges are wide enough to contain the values a measurement would
        # plausibly return rather than tight around a number nobody measured.
        "vehicle.I_w": (0.6, 1.4),           # 3e-4 to 7e-4. Deliberately NOT stretched to the
                                             # 1.2e-3 the parts count gives: the recordings bound it
                                             # at 6e-4 (see VehicleParams.I_w), and a range whose
                                             # upper half the evidence excludes is not a range, it
                                             # is a way of averaging a disagreement away
        "vehicle.drive_split_r": (0.45, 0.60),   # absolute: the slipper clutch and the diffs move
                                                 # the split a little, and 1.0 (rear drive) is a
                                                 # different car rather than a draw from this one
        "vehicle.B_x": (0.7, 1.4),
        "vehicle.C_x": (0.93, 1.13),        # 1.40-1.70: full-slide plateau 0.59-0.81 of peak
        "vehicle.E_x": (0.40, 0.80),        # absolute, the textbook longitudinal span
        "vehicle.r_w": (0.97, 1.03),        # tyre wear and pressure; the ERPM gain is calibrated
                                            # against it on the car, so it cannot drift far
        "odom.erpm_quantum": (2.20e-4, 2.40e-4),   # the two VESC gains the recordings were made on
        "odom.stamp_jitter_std": (0.6, 1.8),
        "odom.stamp_jitter_burst": (0.0, 0.004),
        "imu.shock_rate": (0.0, 2.5),       # venue floors differ by far more than the 22 bags show;
                                            # 0 is a clean floor, 2.5x the fitted rate is a bad one
        "imu.shock_accel": (0.7, 1.5),
        "vehicle.c_roll": (0.5, 2.0),
        "actuator.servo_tau": (0.02, 0.06),
        "actuator.steer_bias": (-0.03, 0.03),
        "actuator.steer_gain": (0.92, 1.08),
        "actuator.motor_tau": (0.10, 0.30),
        "actuator.speed_gain": (0.92, 1.08),
        "actuator.cmd_delay": (0.005, 0.03),
        "vehicle.roll_per_g": (0.6, 1.6),       # 1.0-2.75 deg/g around the measured 1.7
        "vehicle.pitch_per_g": (0.6, 1.4),      # 1.0-2.4 deg/g squat; 2x draws sat 2-4x above the recordings
        "vehicle.dive_per_g": (0.5, 2.0),       # 0.2-0.9 deg/g dive: measured ~0, kept open upward
        "vehicle.road_tilt": (0.5, 1.5),        # scales the nominal, which is 0: no floor wobble
        "vehicle.susp_wn": (0.75, 1.3),
        "vehicle.susp_zeta": (0.25, 0.55),
        "lidar.mount_yaw": (-0.015, 0.015),
        "lidar.mount_roll": (-0.02, 0.02),
        "lidar.mount_pitch": (-0.02, 0.02),
        "lidar.mount_z": (0.095, 0.125),   # +-15 % on the measured 0.110
        "lidar.noise_std": (0.005, 0.011),      # measured 0.0074 (Theil-Sen over still windows)
        "lidar.dropout_prob": (0.0, 0.0008),    # isolated single-beam misses measure 0.00001-0.00053
                                                # per beam over 22 recordings; 0.01 was 19x the worst
        "lidar.duct_scale": (0.7, 1.3),
        "lidar.floor_dropout": (0.2, 0.95),
        "lidar.floor_graze": (0.10, 0.35),
        "lidar.duct_graze_dropout": (0.02, 0.25),  # measured ~0.10; these are absolute, not scaled,
                                                # so the old (0, 0.7) trained at a mean of 0.35 and
                                                # the nominal value never mattered
        "odom.speed_scale_err": (-0.08, 0.08),
        "odom.steer_offset": (-0.01, 0.01),
        "odom.steer_gain_err": (-0.04, 0.04),
        "imu.imu_roll": (-0.07, 0.07),         # +-4 deg: the real VESC IMU leaks 3-5 deg worth of yaw
        "imu.imu_pitch": (-0.07, 0.07),        # rate into its roll axis (measured 2026-09-13)
        "imu.imu_yaw": (-0.02, 0.02),
        "imu.gyro_noise": (0.7, 2.0),
        "imu.accel_noise": (0.7, 2.0),
        # Zero-rate offset over the truly-still windows of 12 recordings with an intact gyro:
        # |bias| never exceeds 0.0018 rad/s on any axis. +-0.05 was 28x that -- a BMI160 spec bound,
        # not this unit -- and taught the policy to distrust a gyro that is in fact steady.
        "imu.gyro_bias_x": (-0.004, 0.004),
        "imu.gyro_bias_y": (-0.004, 0.004),
        "imu.gyro_bias_z": (-0.004, 0.004),
        "imu.accel_bias_x": (-0.45, 0.45),      # measured |bias| up to 0.44 m/s^2, p95 0.35
        "imu.accel_bias_y": (-0.45, 0.45),
        "imu.accel_bias_z": (-0.45, 0.45),
        "imu.vib_accel": (0.5, 2.0),
        "imu.vib_accel_floor": (0.6, 1.8),      # venue-to-venue spread of the road-texture floor
        "imu.vib_gyro_floor": (0.5, 1.6),
        "imu.vib_gyro": (0.5, 2.0),
        "imu.bandwidth": (0.8, 1.2),
        "imu.ahrs_tau": (0.5, 2.0),
    })
    scale_fields: Tuple[str, ...] = (
        "vehicle.I_w", "vehicle.B_x", "vehicle.C_x", "vehicle.r_w",
        "odom.stamp_jitter_std", "imu.shock_rate", "imu.shock_accel",
        "vehicle.mu", "vehicle.m", "vehicle.Iz", "vehicle.B_f", "vehicle.B_r",
        "vehicle.a_max", "vehicle.a_brake", "vehicle.c_roll", "vehicle.roll_per_g", "vehicle.pitch_per_g", "vehicle.dive_per_g", "vehicle.road_tilt", "vehicle.susp_wn",
        "actuator.steer_gain", "actuator.speed_gain", "lidar.duct_scale",
        "imu.gyro_noise", "imu.accel_noise", "imu.vib_accel", "imu.vib_gyro",
        "imu.vib_accel_floor", "imu.vib_gyro_floor", "imu.bandwidth", "imu.ahrs_tau",
    )


@dataclass
class Config:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    actuator: ActuatorParams = field(default_factory=ActuatorParams)
    lidar: LidarParams = field(default_factory=LidarParams)
    odom: OdomParams = field(default_factory=OdomParams)
    imu: ImuParams = field(default_factory=ImuParams)
    sim: SimParams = field(default_factory=SimParams)
    rand: RandomizationConfig = field(default_factory=RandomizationConfig)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        cfg = Config()
        for grp_name in ("vehicle", "actuator", "lidar", "odom", "imu", "sim"):
            grp = getattr(cfg, grp_name)
            for k, v in (d.get(grp_name) or {}).items():
                if not hasattr(grp, k):
                    raise KeyError(f"unknown param {grp_name}.{k}")
                setattr(grp, k, v)
        if "rand" in d and d["rand"] is not None:
            r = d["rand"]
            if "enabled" in r:
                cfg.rand.enabled = bool(r["enabled"])
            if "ranges" in r:
                cfg.rand.ranges.update({k: tuple(v) for k, v in r["ranges"].items()})
        return cfg

    @staticmethod
    def from_yaml(path: str) -> "Config":
        import yaml
        with open(path) as f:
            return Config.from_dict(yaml.safe_load(f) or {})


def float_fields(dc) -> list[str]:
    return [f.name for f in fields(dc) if f.type in ("float", float)]
