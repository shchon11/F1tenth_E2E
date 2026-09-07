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
    a_brake: float = 12.0      # [m/s^2] max braking decel
    v_switch: float = 7.319    # [m/s] above this, accel scales with v_switch/v (power limit)
    # Rolling resistance + aero drag (decel = c_roll + c_drag * v^2)
    c_roll: float = 0.1        # [m/s^2]
    c_drag: float = 0.01       # [1/m]
    # Sprung-mass attitude (what tilts the LiDAR plane): 2nd-order response to body accelerations
    roll_per_g: float = 0.10   # [rad/g] steady-state roll per lateral g (RC truck on oil shocks ~5-6 deg/g)
    pitch_per_g: float = 0.09  # [rad/g] steady-state pitch per longitudinal g (brake dive / squat)
    susp_wn: float = 17.6      # [rad/s] suspension natural frequency (~2.8 Hz)
    susp_zeta: float = 0.35    # [-] damping ratio (underdamped: visible overshoot after braking)


@dataclass
class ActuatorParams:
    # Steering servo: first-order lag + rate limit + angle bias
    servo_tau: float = 0.05        # [s] time constant
    steer_bias: float = 0.0        # [rad] mechanical trim error (randomized)
    steer_gain: float = 1.0        # command scaling error
    # VESC speed loop: first-order tracking of commanded speed via PID-like accel
    motor_tau: float = 0.20        # [s] time constant of speed response
    speed_gain: float = 1.0        # ERPM<->m/s calibration error (randomized)
    # Command latency (sensor -> policy -> actuator), applied as delay buffer on commands
    cmd_delay: float = 0.03        # [s]
    cmd_delay_jitter: float = 0.0  # [s] uniform jitter per step


@dataclass
class LidarParams:
    # Hokuyo UST-10LX: 1080 beams over 270 deg, 40 Hz, 10 m usable (30 m for UST-30LX)
    n_beams: int = 1080
    fov: float = 4.71238898        # [rad] 270 deg
    range_max: float = 10.0        # [m]
    range_min: float = 0.02        # [m]
    rate: float = 40.0             # [Hz]
    mount_x: float = 0.27          # [m] forward of base_link (rear axle center)
    mount_y: float = 0.0
    mount_z: float = 0.15          # [m] scan plane height above the floor at rest
    mount_yaw: float = 0.0         # [rad] mounting misalignment (randomized)
    mount_roll: float = 0.0        # [rad] sensor / platform not level (randomized)
    mount_pitch: float = 0.0       # [rad]
    # Real Hokuyo data is nearly noise-free at these ranges; the dominant artifacts are geometric
    # (tilted scan plane hitting the floor or passing over the duct hoses), modelled in lidar.py.
    noise_std: float = 0.008       # [m] gaussian range noise
    noise_std_rel: float = 0.0     # relative noise (fraction of range)
    dropout_prob: float = 0.001    # per-beam probability of no return (reported as range_max+)
    spike_prob: float = 0.0        # per-beam probability of random spurious return
    motion_distortion: bool = True # emit beams over the 1/rate sweep while the car moves
    dropout_value: float = float("inf")  # what a no-return reads as (Hokuyo driver: inf)
    # incidence-dependent returns (randomized): a beam grazing a glossy hall floor gives no return
    # far more often than one hitting it steeply; a duct hose seen edge-on reflects too little.
    duct_scale: float = 1.0        # per-env multiplier on the track's duct hose diameter
    floor_dropout: float = 0.6     # no-return probability of a floor hit at grazing incidence
    floor_graze: float = 0.20      # [rad] incidence angle below which floor returns fade out
    duct_graze_dropout: float = 0.3  # no-return probability of a duct hit seen edge-on
    duct_graze_cos: float = 0.35   # |cos(angle to the hose normal)| below which duct returns fade


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
    imu_rate: float = 100.0        # [Hz] IMU sample rate (vesc_tool setting; 100 Hz default)
    bandwidth: float = 40.0        # [Hz] sensor low-pass (BMI160 ~ODR/2.5)
    # mounting: VESC on the chassis, mid-car, left of the spine; base_link frame
    imu_x: float = 0.16
    imu_y: float = 0.05
    imu_z: float = 0.045
    imu_roll: float = 0.0          # [rad] misalignment (randomized)
    imu_pitch: float = 0.0
    imu_yaw: float = 0.0
    # noise (per sample, after the low-pass), bias, drift
    gyro_noise: float = 0.0015     # [rad/s] rms
    accel_noise: float = 0.02      # [m/s^2] rms
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
    vib_accel: float = 0.25        # [m/s^2 per m/s] amplitude of the accel vibration
    vib_gyro: float = 0.02         # [rad/s per m/s]
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
        # widened 2026-09-07 (sim2real): the v2 policy lost 63 % of episodes at mu 0.7 and 26 % at
        # 0.1 s delay, both inside what a real car on carpet / a loaded Jetson can show
        "vehicle.mu": (0.55, 1.15),
        "vehicle.m": (0.85, 1.15),
        "vehicle.Iz": (0.8, 1.2),
        "vehicle.B_f": (0.7, 1.3),
        "vehicle.B_r": (0.7, 1.3),
        "vehicle.a_max": (0.6, 1.2),
        "vehicle.mu_f_scale": (0.85, 1.0),
        "vehicle.c_roll": (0.5, 2.0),
        "actuator.servo_tau": (0.03, 0.12),
        "actuator.steer_bias": (-0.04, 0.04),
        "actuator.steer_gain": (0.85, 1.15),
        "actuator.motor_tau": (0.10, 0.40),
        "actuator.speed_gain": (0.85, 1.15),
        "actuator.cmd_delay": (0.0, 0.10),
        "vehicle.roll_per_g": (0.7, 1.5),
        "vehicle.pitch_per_g": (0.7, 1.5),
        "vehicle.susp_wn": (0.75, 1.3),
        "vehicle.susp_zeta": (0.25, 0.55),
        "lidar.mount_yaw": (-0.015, 0.015),
        "lidar.mount_roll": (-0.02, 0.02),
        "lidar.mount_pitch": (-0.02, 0.02),
        "lidar.mount_z": (0.12, 0.18),
        "lidar.noise_std": (0.004, 0.015),
        "lidar.dropout_prob": (0.0, 0.01),
        "lidar.duct_scale": (0.7, 1.3),
        "lidar.floor_dropout": (0.2, 0.95),
        "lidar.floor_graze": (0.10, 0.35),
        "lidar.duct_graze_dropout": (0.0, 0.7),
        "odom.speed_scale_err": (-0.08, 0.08),
        "odom.steer_offset": (-0.01, 0.01),
        "odom.steer_gain_err": (-0.04, 0.04),
        "imu.imu_roll": (-0.02, 0.02),
        "imu.imu_pitch": (-0.02, 0.02),
        "imu.imu_yaw": (-0.02, 0.02),
        "imu.gyro_noise": (0.7, 2.0),
        "imu.accel_noise": (0.7, 2.0),
        "imu.gyro_bias_x": (-0.05, 0.05),
        "imu.gyro_bias_y": (-0.05, 0.05),
        "imu.gyro_bias_z": (-0.05, 0.05),
        "imu.accel_bias_x": (-0.3, 0.3),
        "imu.accel_bias_y": (-0.3, 0.3),
        "imu.accel_bias_z": (-0.3, 0.3),
        "imu.vib_accel": (0.5, 2.0),
        "imu.vib_gyro": (0.5, 2.0),
        "imu.bandwidth": (0.8, 1.2),
        "imu.ahrs_tau": (0.5, 2.0),
    })
    scale_fields: Tuple[str, ...] = (
        "vehicle.mu", "vehicle.m", "vehicle.Iz", "vehicle.B_f", "vehicle.B_r",
        "vehicle.a_max", "vehicle.c_roll", "vehicle.roll_per_g", "vehicle.pitch_per_g", "vehicle.susp_wn",
        "actuator.steer_gain", "actuator.speed_gain", "lidar.duct_scale",
        "imu.gyro_noise", "imu.accel_noise", "imu.vib_accel", "imu.vib_gyro", "imu.bandwidth", "imu.ahrs_tau",
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
