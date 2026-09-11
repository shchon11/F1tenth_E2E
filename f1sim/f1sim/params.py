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
    # Rolling resistance + aero drag (decel = c_roll + c_drag * v^2)
    c_roll: float = 0.1        # [m/s^2]
    c_drag: float = 0.01       # [1/m]
    # Sprung-mass attitude (what tilts the LiDAR plane): 2nd-order response to body accelerations
    roll_per_g: float = 0.10   # [rad/g] steady-state roll per lateral g (RC truck on oil shocks ~5-6 deg/g)
                               # NOT measured: the accelerometer route needs an independent a_y, and
                               # both v*omega and a differentiated particle-filter pose are too noisy
                               # -- regressing (measured - true) on true then collapses toward the
                               # errors-in-variables limit and returns -9 deg/g. Left at the guess.
    pitch_per_g: float = 0.09  # [rad/g] brake dive / squat. Bracketed from the LiDAR, which needs no
                               # independent acceleration: tilt the scan plane and beams reach the
                               # floor at z / sin(theta) instead of the wall. Casting the beams
                               # against the recorded map, the fraction stopping short of the
                               # predicted wall goes brake > cruise > push in all six recordings
                               # checked (20.5 / 15.9 / 9.8 % at the top), and so does the no-return
                               # rate -- a grazing floor hit either reads short or does not come back.
                               # The range they stop at implies 2.5-5.3 deg at a median -4.4 m/s^2,
                               # i.e. 5.5-11.8 deg/g against the 5.2 here. A bracket, not a fit: the
                               # cruise baseline shows map and pose error and cars the map does not
                               # contain, so the short-beam population is not purely floor strikes.
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


@dataclass
class ImuParams:
    """VESC 6 built-in 6-axis IMU (BMI160 class) on the lower chassis, published by vesc_driver.
    Modelled per physics substep: specific force at the sensor (gravity leaking in through body
    roll/pitch, lever arm from the CoG), speed-proportional vibration at wheel/motor frequencies,
    the sensor's internal low-pass, sampling at `rate`, bias + random walk, white noise,
    quantization, and the VESC's own attitude estimate (complementary/Mahony-style filter)."""
    enabled: bool = True
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
    collision_restitution: float = 0.2   # velocity kept along wall on contact when not terminating
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
        "vehicle.c_roll": (0.5, 2.0),
        "actuator.servo_tau": (0.02, 0.06),
        "actuator.steer_bias": (-0.03, 0.03),
        "actuator.steer_gain": (0.92, 1.08),
        "actuator.motor_tau": (0.10, 0.30),
        "actuator.speed_gain": (0.92, 1.08),
        "actuator.cmd_delay": (0.005, 0.03),
        "vehicle.roll_per_g": (0.7, 1.5),
        "vehicle.pitch_per_g": (0.7, 1.8),      # 3.6-9.3 deg/g: the LiDAR bracket sits above
                                                # the nominal, so the range reaches up to it
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
        "imu.imu_roll": (-0.02, 0.02),
        "imu.imu_pitch": (-0.02, 0.02),
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
        "vehicle.mu", "vehicle.m", "vehicle.Iz", "vehicle.B_f", "vehicle.B_r",
        "vehicle.a_max", "vehicle.a_brake", "vehicle.c_roll", "vehicle.roll_per_g", "vehicle.pitch_per_g", "vehicle.susp_wn",
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
