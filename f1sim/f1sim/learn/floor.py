"""Which LiDAR returns are the floor, from ranges and the IMU attitude alone.

Why this exists
---------------
The scan plane sits **0.110 m above the floor** (`docs/real_data_calibration.md` §2.7, measured from
the bags' own `/tf_static`), and it follows the sprung body. A downward tilt of theta puts the floor
across the beam at `mount_z / sin(theta)`:

    0.5 deg -> 12.6 m (out of range)   2 deg -> 3.15 m   3 deg -> 2.10 m   5 deg -> 1.26 m

so a couple of degrees of nose-down is a **phantom wall exactly where a braking car is looking**.
The tilt is not hypothetical: `vehicle.road_tilt` alone is 1 deg rms with a 0.4 s time constant
(§6.1a), the sensor mount is randomised over +-0.02 rad, and squat/dive add 1.7 / 0.46 deg per g on
top. `learn/clearance.py` turns **every** return into an occupied cell, so those returns bend and
slow the executed plan; the policy sees them as the nearest thing in the corridor.

What this module is allowed to use
----------------------------------
Only what the car has: the **ranges** and the **VESC roll/pitch estimate**. No map, no pose, no
privileged state, no `track.edt`. In particular the obvious discriminator -- "this return is shorter
than the free range the map says this bearing has" -- is a map lookup and is deliberately **not**
used, so the identical code runs off `/scan` on the car. `scan_type == HIT_GROUND` appears here only
in the *tests* and in the training-time auxiliary label, never in the runtime path.

The geometry, closed form
-------------------------
`Lidar.rays` builds each beam by rotating the nominal bearing `a` by roll `phi` then pitch `th`
(ROS: +pitch nose down, +roll right side down). Its body-frame direction is a **unit** vector

    bx =  cos a cos th + sin a sin phi sin th
    by =  sin a cos phi
    bz = -cos a sin th + sin a sin phi cos th

and the sensor origin sits at

    oz = mount_z cos(phi) cos(th) - mount_x sin(th)          (mount_y = 0; + mount_y sin(phi) sin(th))

above the floor, because `base_link` *is* the ground plane in this project's frames (§2.7). The
simulator reports the **3-D beam length**, and since `|b| = 1` the height of a return at range `r` is

    z(r) = oz + r * bz                                                                        (1)

and the floor, `z = 0`, is met at

    r_floor = -oz / bz          for bz < 0, and never for bz >= 0.                            (2)

(2) is exact -- no small-angle step, no iteration -- and it is the same quantity `Lidar._trace_torch`
computes as `s_ground * sqrt(1 + k^2)`. That is asserted against the simulator in
`tests/test_floor_mask.py` rather than argued here.

The decision, and its tolerance band
------------------------------------
Everything is decided on (1), the **height of the return**, not on (2), the range it would be at. The
two are equivalent statements and the first is the one that is numerically usable: near the boresight
`dr/dtheta = -r^2 / oz` is 82 m/rad at 3 m, so a *range* band would have to be metres wide at 3 m and
centimetres wide at 1 m, while the same uncertainty is a few centimetres of **height** everywhere.

    sigma_z(r, a)^2 = (dz/dpitch)^2 sigma_pitch^2 + (dz/droll)^2 sigma_roll^2 + sigma_static^2   (3)

with the two derivatives taken exactly from (1). **Per axis, and per bearing**, because the two are
not interchangeable and the difference is worth real precision: `dz/dpitch = -(mount_x + r cos a)`
and `dz/droll = r sin a` to first order, so a beam straight ahead is blind to roll error and a beam
at 90 deg is blind to pitch error. The forward sector -- the one a braking car's plan lives in -- is
therefore governed by the **pitch** error alone, which is the smaller of the two. `sigma_static` is
the range-domain floor the motion-memory work measured on the real recordings for a static-geometry
prediction (0.025 m, `work/motion-memory/REPORT.md` §3/E2).

and the pointwise term is the two-sided band that residual buys:

    p_geom = sharp(r) * exp(-z^2 / (2 sigma_z^2))                                             (4)

**Two-sided, not one-sided.** A return *below* the estimated plane (`z < 0`) is not evidence of the
floor: it is evidence that the attitude estimate is too large, and an over-estimated tilt puts a
solid return beyond the predicted floor range just as readily as it does a floor one. An upward
beam gets `z >= oz = 0.11 m`, which is 2-4 sigma, so it falls out without a special case.

`sharp` is the part that is easy to leave out and wrong to:

    sharp = oz / sqrt(oz^2 + sigma_z_att^2)          (sigma_z_att = (3) without sigma_static) (4a)

The predicted floor range is `oz / |bz|`, so its own uncertainty is `r sigma_z_att / oz` **relative**
to itself -- 30 % at 2 m and 140 % at 10 m. Past `oz/sigma_att` = 7 m the geometry no longer
localises the floor at all: a beam that is level to within a quarter of
a degree meets the floor *somewhere* beyond 10 m, and "this 9 m return might be the floor" is true
and useless. Without (4a) a level sensor reads 0.78 on its most distant returns -- measured, and the
reason this term exists. With it the same beams read 0.45, below the gate, while a real floor arc at
2-3 m still reads 0.9. The cost is stated rather than hidden: **the channel cannot identify a floor
return past ~6 m**, and does not pretend to. That is exactly the regime where the floor is too far
to bend a plan.

The one extra cue: neighbours
-----------------------------
A floor return is never alone. The scan plane meets the floor plane in a **straight line**, so a
tilted scan produces a wide contiguous arc of returns that all sit on `z = 0`; a solid object that
happens to cross the plane does so over a few beams. So the final likelihood is the geometric mean of
(4) at the beam and (4) averaged over its neighbours:

    log p = (1 - w) log p_geom(i) + w * mean_{|j - i| <= W, j valid} log p_geom(j)             (5)

which is one `avg_pool1d` over the beam axis. Beams with no return are excluded from the average --
a dropout in the middle of a floor arc must not argue against it -- and get `p = 0` themselves,
because a beam with no return carries nothing to classify.

Unknown attitude
----------------
`f1sim_ros/policy_node.attitude_from_orientation` returns `None` for an orientation this car cannot
be trusted with, and `_stale_inputs` reports `attitude` when the last usable one is too old. Five of
the thirteen competition recordings carry an estimate that swings the extracted roll past 40 deg
(`work/motion-memory/REPORT.md` §6). When that is the state of the world the channel reports
**`UNKNOWN` = 0.5 on every beam**, not 0: "I cannot tell" must not be spelled the same way as
"solid", because the clearance gate reads this number and a confident 0 would let a bad attitude
estimate silently restore the behaviour this module exists to remove.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn.functional as F

#: What a beam reads when the attitude estimate is not usable. Halfway between "solid" (0) and
#: "floor" (1), and the gate's threshold is above it, so an unknown attitude gates nothing.
UNKNOWN = 0.5

#: [rad] 1-sigma error of the scan plane's tilt as `AttitudeTracker` estimates it, per axis.
#: MEASURED -- `work/measure/out/attitude_tune.json`, `docs/research/floor-mask-2026-09-15.md` --
#: over three tracks, 600 control steps, 16 cars, with the recalibrated attitude model, the +-4 deg
#: IMU misalignment and the +-0.02 rad LiDAR mount randomisation on. For scale: assuming the plane
#: is LEVEL reads 0.024 / 0.023 on the same rows, and the VESC quaternion reads 0.136 / 0.086. What
#: the geometry needs to localise the floor out to 3 m is ~0.015, and nothing here reaches it; see
#: the research note, and `learn/frontend.py` for the estimate that can.
SIGMA_ROLL = 0.027
SIGMA_PITCH = 0.024

#: [m] the range-independent part of the height band (3). It is NOT the LiDAR's range noise: at
#: 3 m a grazing beam's vertical direction cosine is 0.03, so the calibrated 7.4 mm + 1 mm/m of
#: range error is 0.3 mm of height and does not matter. What does matter is the **scan plane's own
#: height**, which the car knows as the 0.110 m `/tf_static` says (§2.7) and the simulator draws
#: over +-15 % of; that is 0.0087 m rms of a bias the car cannot see, and it is what this is.
#: (The motion-memory work's `sigma_static` = 0.025 m is a different quantity -- the residual of a
#: four-frame point-cloud warp -- and importing it here made the band three times too wide, which
#: cost precision against a 1 % base rate. Measured, and written down.)
SIGMA_STATIC = 0.0087

#: Nominal sensor mounting (`params.LidarConfig`), repeated here so a caller that has no `Config`
#: -- the ROS node -- still gets the measured geometry rather than a guess.
MOUNT_X = 0.297
MOUNT_Z = 0.110


@dataclass(frozen=True)
class FloorSpec:
    """Everything the likelihood needs besides the scan and the attitude. Recorded with any result."""

    sigma_roll: float = SIGMA_ROLL    # [rad] see above
    sigma_pitch: float = SIGMA_PITCH  # [rad]
    sigma_static: float = SIGMA_STATIC  # [m]
    mount_x: float = MOUNT_X          # [m] forward of base_link
    mount_z: float = MOUNT_Z          # [m] scan plane above the floor at rest
    mount_y: float = 0.0              # [m] left of base_link
    smooth_beams: int = 12            # +- beams in the neighbourhood term (12 beams = 3 deg)
    smooth_weight: float = 0.5        # w in (5); 0 disables the neighbourhood term entirely
    p_min: float = 1e-6               # clamp on p_geom before the log, so (5) has no -inf
    #: Reported by `gate_weight`: a return is removed from the occupancy grid above this. 0.5 is
    #: `UNKNOWN`, so the threshold is strictly above it and an unknown attitude gates nothing.
    gate_threshold: float = 0.60

    def validate(self) -> "FloorSpec":
        if not (self.sigma_roll > 0.0 and self.sigma_pitch > 0.0):
            raise ValueError(f"sigma_roll/sigma_pitch must be positive, got {self.sigma_roll} / "
                             f"{self.sigma_pitch}: they are the width of the tolerance band and "
                             f"the likelihood divides by them")
        if not self.sigma_static >= 0.0:
            raise ValueError(f"sigma_static must be non-negative, got {self.sigma_static}")
        if not self.mount_z > 0.0:
            raise ValueError(f"mount_z must be positive, got {self.mount_z}: a scan plane at or "
                             f"below the floor has no floor-intersection range")
        if self.smooth_beams < 0:
            raise ValueError(f"smooth_beams must be >= 0, got {self.smooth_beams}")
        if not 0.0 <= self.smooth_weight <= 1.0:
            raise ValueError(f"smooth_weight must be in [0, 1], got {self.smooth_weight}")
        if not 0.0 < self.p_min < 1.0:
            raise ValueError(f"p_min must be in (0, 1), got {self.p_min}")
        if not UNKNOWN < self.gate_threshold <= 1.0:
            raise ValueError(f"gate_threshold {self.gate_threshold} must be above UNKNOWN "
                             f"({UNKNOWN}) and at most 1: at or below it an unknown attitude would "
                             f"gate every beam of every scan")
        return self

    def to_meta(self) -> dict:
        return asdict(self)


def beam_angles(n_beams: int, fov: float, device=None, dtype=None) -> torch.Tensor:
    """The bearings of `n_beams` returns spread over `fov`, endpoints included -- `Lidar.angles`."""
    return torch.linspace(-fov / 2.0, fov / 2.0, int(n_beams), device=device, dtype=dtype)


def _att(roll, pitch, device, dtype, n: int):
    """(B, 1) roll and pitch tensors from whatever the caller passed."""
    r = torch.as_tensor(roll, device=device, dtype=dtype).reshape(-1, 1)
    p = torch.as_tensor(pitch, device=device, dtype=dtype).reshape(-1, 1)
    if r.shape[0] != p.shape[0]:
        raise ValueError(f"roll has {r.shape[0]} rows and pitch has {p.shape[0]}")
    if r.shape[0] not in (1, n):
        raise ValueError(f"attitude has {r.shape[0]} rows for a {n}-row scan")
    return r.expand(n, 1), p.expand(n, 1)


def beam_geometry(angles: torch.Tensor, roll, pitch, spec: FloorSpec, rows: int = 1):
    """`(bz (B, N), oz (B, 1))`: each beam's vertical direction cosine and the sensor height.

    Exactly `Lidar.rays`' construction, with `mount_yaw` left at its nominal zero -- a yaw offset
    rotates the bearings and does not tilt the plane, and the car does not know its own draw.
    """
    ang = angles.reshape(1, -1)
    phi, th = _att(roll, pitch, ang.device, ang.dtype, rows)
    ca, sa = torch.cos(ang), torch.sin(ang)
    cph, sph, cth, sth = torch.cos(phi), torch.sin(phi), torch.cos(th), torch.sin(th)
    bz = -ca * sth + sa * sph * cth
    oz = spec.mount_y * sph * sth + spec.mount_z * cph * cth - spec.mount_x * sth
    return bz, oz


def floor_range(angles: torch.Tensor, roll, pitch, spec: FloorSpec = None,
                rows: int = 1) -> torch.Tensor:
    """(B, N) the 3-D range at which each beam meets the floor -- equation (2). `inf` where it never
    does (a level or rising beam).

    This is the "phantom wall": with a level sensor every entry is `inf`, and at 3 deg of nose-down
    the forward entries are ~2.1 m.
    """
    spec = (spec or FloorSpec()).validate()
    bz, oz = beam_geometry(angles, roll, pitch, spec, rows)
    desc = bz < 0
    return torch.where(desc, oz / (-bz).clamp_min(1e-12), torch.full_like(bz, float("inf")))


def return_height(ranges: torch.Tensor, angles: torch.Tensor, roll, pitch,
                  spec: FloorSpec = None) -> torch.Tensor:
    """(B, N) height above the floor of each return -- equation (1), in metres.

    `ranges` are **3-D beam lengths in metres**, which is what the sensor reports and what
    `Lidar.scan` returns; `gym_env._norm_scan` divides them by `range_max` and the callers that hold
    a normalised scan multiply it back before coming here.
    """
    spec = (spec or FloorSpec()).validate()
    if ranges.dim() != 2:
        raise ValueError(f"ranges must be (B, N), got {tuple(ranges.shape)}")
    if angles.numel() != ranges.shape[1]:
        raise ValueError(f"{angles.numel()} bearings for {ranges.shape[1]} returns: the channel "
                         f"must be built with the beam geometry the scan it is fed actually has")
    bz, oz = beam_geometry(angles.to(ranges.device, ranges.dtype), roll, pitch, spec,
                           ranges.shape[0])
    return oz + ranges * bz


def _height_and_band(ranges: torch.Tensor, angles: torch.Tensor, roll, pitch, spec: FloorSpec):
    """`(z, oz, sigma_z_att)` -- equation (1), the sensor height, and the attitude part of (3).

    The two derivatives are taken exactly from (1) rather than from its small-angle form, because
    the mount lever `-mount_x sin(pitch)` contributes `-mount_x cos(pitch)` to `dz/dpitch` and that
    term does not vanish with the tilt.
    """
    rows = ranges.shape[0]
    ang = angles.reshape(1, -1)
    phi, th = _att(roll, pitch, ang.device, ang.dtype, rows)
    ca, sa = torch.cos(ang), torch.sin(ang)
    cph, sph, cth, sth = torch.cos(phi), torch.sin(phi), torch.cos(th), torch.sin(th)
    bz = -ca * sth + sa * sph * cth
    oz = spec.mount_y * sph * sth + spec.mount_z * cph * cth - spec.mount_x * sth
    z = oz + ranges * bz
    doz_dth = spec.mount_y * sph * cth - spec.mount_z * cph * sth - spec.mount_x * cth
    doz_dph = spec.mount_y * cph * sth - spec.mount_z * sph * cth
    dz_dth = doz_dth + ranges * (-ca * cth - sa * sph * sth)
    dz_dph = doz_dph + ranges * (sa * cph * cth)
    sig_att = torch.sqrt((dz_dth * spec.sigma_pitch) ** 2 + (dz_dph * spec.sigma_roll) ** 2)
    return z, oz.expand_as(z) if oz.shape != z.shape else oz, sig_att


def _neighbour_mean(x: torch.Tensor, valid: torch.Tensor, half: int) -> torch.Tensor:
    """Mean of `x` over +-`half` beams, counting only `valid` ones. (B, N) -> (B, N).

    Replicate padding at the two ends of the 270 deg window: the window does not wrap, and zero
    padding would pull the first and last beams' neighbourhood toward "solid" for no reason.
    """
    k = 2 * int(half) + 1
    w = valid.to(x.dtype)
    pad = (int(half), int(half))
    num = F.avg_pool1d(F.pad((x * w)[:, None], pad, mode="replicate"), k, stride=1)[:, 0]
    den = F.avg_pool1d(F.pad(w[:, None], pad, mode="replicate"), k, stride=1)[:, 0]
    return torch.where(den > 0, num / den.clamp_min(1e-6), x)


def floor_likelihood(ranges: torch.Tensor, roll, pitch, spec: FloorSpec = None, *,
                     angles: torch.Tensor = None, fov: float = 1.5 * math.pi,
                     valid: torch.Tensor = None, att_ok=None) -> torch.Tensor:
    """(B, N) in [0, 1]: how much each return looks like the floor rather than a solid object.

    `ranges`   (B, N) 3-D beam lengths [m]; see `return_height`.
    `roll`,
    `pitch`    (B,) or a scalar [rad], the car's own estimate (`imu_att`, `/sensors/imu/raw`).
    `angles`   (N,) bearings [rad]; built from `fov` when omitted.
    `valid`    (B, N) bool, True where the beam actually returned something. Derived from
               `ranges` (finite and inside the sensor's reach) when omitted.
    `att_ok`   (B,) bool or a scalar, False where the attitude estimate is not usable. Those rows
               read `UNKNOWN` everywhere.

    0 = a solid return (and a beam with no return at all), 1 = the floor, 0.5 = unknown attitude.
    """
    spec = (spec or FloorSpec()).validate()
    if ranges.dim() != 2:
        raise ValueError(f"ranges must be (B, N), got {tuple(ranges.shape)}")
    B, N = ranges.shape
    if angles is None:
        angles = beam_angles(N, fov, device=ranges.device, dtype=ranges.dtype)
    if valid is None:
        valid = torch.isfinite(ranges) & (ranges > 0)
    else:
        valid = valid.to(ranges.device)
        if valid.shape != ranges.shape:
            raise ValueError(f"valid {tuple(valid.shape)} does not match ranges {tuple(ranges.shape)}")
    r = torch.where(valid, ranges, torch.zeros_like(ranges))
    ang = angles.to(ranges.device, ranges.dtype)
    z, oz, sig_att = _height_and_band(r, ang, roll, pitch, spec)
    sig = torch.sqrt(sig_att ** 2 + spec.sigma_static ** 2).clamp_min(1e-6)
    # (4a): how sharply the geometry localises the floor at this range at all. 1 close in, and
    # falling through 0.7 where the predicted floor range is as uncertain as it is large.
    sharp = oz.abs() / torch.sqrt(oz ** 2 + sig_att ** 2).clamp_min(1e-9)
    # (4): the two-sided band the attitude error buys, damped by how much the geometry can say.
    p_geom = (sharp * torch.exp(-0.5 * (z / sig) ** 2)).clamp(spec.p_min, 1.0)
    if spec.smooth_beams > 0 and spec.smooth_weight > 0.0:
        lp = torch.log(p_geom)
        lp = (1.0 - spec.smooth_weight) * lp + spec.smooth_weight * _neighbour_mean(
            lp, valid, spec.smooth_beams)
        p = torch.exp(lp)
    else:
        p = p_geom
    p = torch.where(valid, p, torch.zeros_like(p))
    if att_ok is not None:
        ok = torch.as_tensor(att_ok, device=p.device)
        ok = ok.reshape(-1, 1).expand(B, N) if ok.numel() > 1 else ok.reshape(1, 1).expand(B, N)
        p = torch.where(ok.bool(), p, torch.full_like(p, UNKNOWN))
    return p.clamp(0.0, 1.0)


def floor_likelihood_norm(scan_norm: torch.Tensor, roll, pitch, range_max: float,
                          spec: FloorSpec = None, *, angles: torch.Tensor = None,
                          fov: float = 1.5 * math.pi, range_eps: float = 0.02,
                          att_ok=None) -> torch.Tensor:
    """`floor_likelihood` for a scan in the observation's own units (range / range_max, no return =
    1.0 -- `gym_env._norm_scan`, `obs.ObsBuilder.build`).

    `range_eps` is `clearance.ClearanceSpec.range_eps`: a normalised range within this of 1.0 is a
    no-return, the same convention the occupancy grid drops a beam on. Kept identical so the channel
    and the gate disagree about no returns.
    """
    if scan_norm.dim() != 2:
        raise ValueError(f"scan must be (B, N), got {tuple(scan_norm.shape)}")
    valid = scan_norm < (1.0 - float(range_eps))
    return floor_likelihood(scan_norm * float(range_max), roll, pitch, spec, angles=angles,
                            fov=fov, valid=valid, att_ok=att_ok)


def gate_weight(p_floor: torch.Tensor, spec: FloorSpec = None) -> torch.Tensor:
    """(B, N) in {0, 1}: 1 for a return the occupancy grid should keep, 0 for a likely floor return.

    A hard threshold rather than a soft weight, because the grid it feeds is binary: `occupancy`
    scatters ones, and a 0.7-occupied cell is not a thing the distance transform can represent. The
    softness is in the likelihood, and the threshold is stated, logged and swept.
    """
    spec = (spec or FloorSpec()).validate()
    return (p_floor < spec.gate_threshold).to(p_floor.dtype)


def describe(spec: FloorSpec = None) -> str:
    """One line for a log or a checkpoint header."""
    s = (spec or FloorSpec()).validate()
    return (f"floor channel: sigma roll/pitch {math.degrees(s.sigma_roll):.2f}/"
            f"{math.degrees(s.sigma_pitch):.2f} deg, sigma_static "
            f"{s.sigma_static * 100:.1f} cm, neighbourhood +-{s.smooth_beams} beams at weight "
            f"{s.smooth_weight:g}, mount ({s.mount_x:.3f}, {s.mount_z:.3f}) m, gate above "
            f"{s.gate_threshold:g}")


# ====================================================================== attitude
#: What the VESC attitude estimate is worth for THIS purpose, measured (see
#: `docs/research/floor-mask-2026-09-15.md`): driving the held-out proxy tracks it is wrong by
#: **0.141 rad rms in roll and 0.083 rad in pitch** -- 8.1 and 4.8 degrees. Two causes, both by
#: construction and both in the recordings as well as in the simulator:
#:
#:   * the IMU's own mounting misalignment is randomised over +-0.07 rad (+-4 deg,
#:     `params.py`, measured in `real_data_calibration.md` §6.1a as 2.7-4.6 deg of yaw-rate leak),
#:     and it enters the accelerometer's gravity direction as a **constant offset**;
#:   * the filter pulls toward that gravity direction, so a sustained acceleration tilts it: at the
#:     0.45 g the recordings brake at, `atan2(4.4, 9.81)` is 24 deg of apparent nose-down.
#:
#: At 0.1 rad of attitude error the floor could be anywhere from 1.1 m to beyond the sensor's reach,
#: and a channel keyed on it flags a quarter of all returns. So the channel does not read `imu_att`.
#: `AttitudeTracker` is what it reads instead.
class AttitudeTracker:
    """Roll and pitch of the scan plane, from the gyro, for the beams this module classifies.

    Not a general AHRS -- it answers one question: *how is the scan plane tilted relative to the
    floor it was standing on at rest?* Three differences from the VESC filter, each one measured
    against it in the research note:

    * **the gyro integrates; the accelerometer only corrects while the car is quiet.** The
      correction is gated on `|a| - g` and on the gyro magnitude, so a 0.45 g brake contributes
      nothing instead of 24 degrees. `ahrs_accel_decay = 1.0` in the simulator's VESC model leaves
      a 0.45 g brake at 90 % weight; this leaves it at zero.
    * **a zero reference taken at rest.** A stationary car's body is level, so whatever the filter
      reads then is the sensor's own misalignment -- the 4 degrees no estimator can see from the
      inside -- and it is subtracted from every later output. This is the single largest term and
      it costs one branch.
    * **a slow leak toward that reference**, so gyro bias and its random walk cannot integrate away
      over a long run. The true tilt is a zero-mean OU process (`vehicle.road_tilt`, tau 0.4 s) plus
      a suspension response to accelerations that are themselves zero-mean over a lap, so the leak
      costs a little of the sustained squat and buys a bounded error.
    * **the yaw rate is regressed out of the roll and pitch rates.** The IMU is mounted a few
      degrees off (`imu.imu_roll/imu_pitch`, +-0.07 rad; `real_data_calibration.md` §6.1a measures
      2.7-4.6 deg on the real car), and `imu.misalign` turns that into a LEAK of the yaw rate into
      the other two gyro axes: at 3 rad/s of yaw and 0.07 rad of misalignment, 0.21 rad/s of
      phantom roll rate, which is 12 degrees over a one-second corner. It is the single largest
      error in a gyro-integrated attitude on this car, and it is exactly what §6.1a removes by
      regression before integrating. Here the same regression runs online: a running
      `E[gx gz] / E[gz^2]` with exponential forgetting. It is identifiable because a *steady*
      corner's true roll rate is 90 degrees out of phase with its yaw rate -- the body rolls while
      the yaw rate is changing -- while the leak is exactly in phase, so the covariance sees the
      leak and averages the signal away.

    What it cannot do: the **LiDAR's** mounting misalignment (`lidar.mount_roll/mount_pitch`,
    +-0.02 rad) is between the sensor and the body and no IMU sees it. It is the floor of this
    estimate, and it is what `SIGMA_ATT` is mostly made of.

    Batched: one row per env, state carried across steps, cleared per row at an episode boundary
    exactly as `obs.ScanAugment`'s occupancy memory is.
    """

    #: [m/s^2] |‖a‖ - g| below which the accelerometer is believed to be reading gravity.
    ACC_GATE = 0.35
    #: [rad/s] and the gyro magnitude below which the body is not rotating under it.
    GYRO_GATE = 0.25
    #: [s] how fast the gated correction pulls, and how fast the output leaks back to the reference.
    TAU_ACC = 0.5
    #: [s] The leak is the estimator's whole bandwidth decision, and it was swept rather than
    #: reasoned about. Integrating the gyro accumulates its noise as a random walk -- and on this
    #: car the gyro's noise is *vibration*, 0.278 rad/s rms at 4 m/s, measured on the real recordings
    #: (`real_data_calibration.md` §2.6) -- so the error grows as `sigma_gyro * sqrt(tau * dt / 2)`,
    #: 0.022 rad at 1 s. Shorter is therefore better right down to the point where the leak stops
    #: passing the tilt itself, and the sweep is monotone over 0.1 ... 3.0 s with no interior
    #: optimum: roll/pitch rms 0.026/0.024 at 0.1 s, 0.031/0.026 at 0.3, 0.049/0.035 at 1.0.
    #:
    #: **And every one of them loses to assuming the plane is level, which reads 0.024/0.023.** The
    #: tilt is a zero-mean process with an rms of 0.028 rad; integrating a gyro whose vibration is
    #: ten times the signal's own rate does not beat it. 0.15 s is the default because it is within
    #: 10 % of that bound and, unlike a constant zero, it still carries the transients -- but the
    #: honest reading is that an IMU-only attitude on this car is not worth much, and the research
    #: note says so at length.
    TAU_LEAK = 0.15
    #: [s] forgetting time of the yaw-leak regression. Long, because the coefficient is a property
    #: of how the sensor is bolted down and does not change within a run.
    TAU_LEAK_FIT = 20.0
    #: Ceiling on that coefficient: the misalignment is +-0.07 rad and this is twice it, so a
    #: transient correlation cannot turn the compensator into a second error source.
    LEAK_MAX = 0.15
    #: [m/s] below this the car is "at rest" and the zero reference is (re)learned.
    REST_SPEED = 0.05
    #: [s] time constant of that reference while at rest.
    TAU_REST = 0.3

    def __init__(self, batch: int, device="cpu", dt: float = 0.025, dtype=torch.float32,
                 g: float = 9.81):
        self.batch, self.dt, self.g = int(batch), float(dt), float(g)
        self.device, self.dtype = torch.device(device), dtype
        self.att = torch.zeros(self.batch, 2, device=self.device, dtype=dtype)   # roll, pitch [rad]
        self.ref = torch.zeros(self.batch, 2, device=self.device, dtype=dtype)   # the at-rest reading
        self.seen_rest = torch.zeros(self.batch, dtype=torch.bool, device=self.device)
        #: Running E[gz^2] and E[g_xy gz] for the yaw-leak regression. Kept across episodes with the
        #: rest reference and for the same reason: both describe the mounting, not the episode.
        self.szz = torch.zeros(self.batch, 1, device=self.device, dtype=dtype)
        self.sxz = torch.zeros(self.batch, 2, device=self.device, dtype=dtype)

    def reset(self, done=None) -> None:
        """Clear the integrator for every row, or for the rows whose episode ended.

        The **reference is kept**: it is a property of the sensor's mounting, not of the episode,
        and a car that has already stood still once knows its own misalignment for the rest of the
        run. The integrator is not: the new episode starts from a new pose.
        """
        if done is None:
            self.att.zero_()
            return
        d = done if torch.is_tensor(done) else torch.as_tensor(done, device=self.att.device)
        keep = (~d.bool()).to(self.att.dtype)[:, None]
        self.att = self.att * keep

    def _step(self, state, gyro: torch.Tensor, accel: torch.Tensor, speed: torch.Tensor):
        """`(att, ref, seen_rest)` after one control step, as values. No side effects, so `update`
        and `peek` cannot drift apart: there is one copy of the filter."""
        att0, ref0, seen0 = state
        dt = self.dt
        g = gyro.to(self.dtype)
        att = att0 + (g[:, :2] - self.leak() * g[:, 2:3]) * dt
        amag = torch.linalg.norm(accel, dim=1)
        gmag = torch.linalg.norm(gyro, dim=1)
        quiet = ((amag - self.g).abs() < self.ACC_GATE) & (gmag < self.GYRO_GATE)
        roll_a = torch.atan2(accel[:, 1], accel[:, 2])
        pitch_a = torch.atan2(-accel[:, 0], torch.sqrt(accel[:, 1] ** 2 + accel[:, 2] ** 2))
        meas = torch.stack([roll_a, pitch_a], 1).to(self.dtype)
        k = (dt / self.TAU_ACC) * quiet.to(self.dtype)[:, None]
        att = att + k * (meas - att)
        att = att - (dt / self.TAU_LEAK) * (att - ref0)
        rest = (speed.abs() < self.REST_SPEED) & quiet
        kr = (dt / self.TAU_REST) * rest.to(self.dtype)[:, None]
        # First time a row stands still, adopt the reading outright rather than filtering toward it
        # from a zero that means "perfectly mounted" -- there is nothing yet to filter.
        first = (rest & ~seen0)[:, None]
        ref = torch.where(first, att, ref0 + kr * (att - ref0))
        return att, ref, seen0 | rest

    def _check(self, gyro, accel, rows):
        for t, n in ((gyro, "gyro"), (accel, "accel")):
            if t.dim() != 2 or t.shape != (rows, 3):
                raise ValueError(f"{n} must be ({rows}, 3), got {tuple(t.shape)}")

    def leak(self) -> torch.Tensor:
        """(B, 2) how much of the yaw rate currently leaks into the roll and pitch gyro axes."""
        return (self.sxz / self.szz.clamp_min(1e-9)).clamp(-self.LEAK_MAX, self.LEAK_MAX)

    def _fit_leak(self, gyro: torch.Tensor) -> None:
        g = gyro.to(self.dtype)
        b = min(1.0, self.dt / self.TAU_LEAK_FIT)
        gz = g[:, 2:3]
        self.szz = self.szz + b * (gz * gz - self.szz)
        self.sxz = self.sxz + b * (g[:, :2] * gz - self.sxz)

    @torch.no_grad()
    def update(self, gyro: torch.Tensor, accel: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """One control step. `gyro` (B, 3) [rad/s], `accel` (B, 3) [m/s^2], `speed` (B,) [m/s].

        Returns (B, 2) roll and pitch of the scan plane relative to its at-rest orientation [rad].
        """
        self._check(gyro, accel, self.batch)
        self.att, self.ref, self.seen_rest = self._step(
            (self.att, self.ref, self.seen_rest), gyro, accel, speed)
        # AFTER the step, so the coefficient the step used is the one estimated from everything
        # before it and this sample cannot regress itself out of its own integration.
        self._fit_leak(gyro)
        return self.att - self.ref

    @torch.no_grad()
    def peek(self, gyro: torch.Tensor, accel: torch.Tensor, speed: torch.Tensor, index=None):
        """`(attitude, usable)` for one step **without advancing the filter**, optionally for a row
        subset.

        For a terminal observation: it is scored (the truncation bootstrap reads its value) and
        never acted on, so advancing the tracker for it would leave the next real step carrying a
        step the next episode did not take -- the same rule `ScanAugment.preview` keeps for the
        occupancy memory.
        """
        rows = self.batch if index is None else int(len(index))
        self._check(gyro, accel, rows)
        st = (self.att, self.ref, self.seen_rest)
        if index is not None:
            st = tuple(t[index] for t in st)
        att, ref, seen = self._step(st, gyro, accel, speed)
        return att - ref, seen


#: Calibrated suspension gains, `real_data_calibration.md` §6.1a, and the simulator's own defaults
#: (`params.VehicleParams`). Asymmetric in pitch on purpose: this car squats under throttle far more
#: than it dives under its regen-limited braking.
ROLL_PER_G = 0.03      # [rad/g] 1.7 deg/g; measured 1.5-2.1
SQUAT_PER_G = 0.03     # [rad/g] 1.7 deg/g under throttle; measured 1.5-2.5
DIVE_PER_G = 0.008     # [rad/g] 0.46 deg/g under braking; the recordings show almost none
G_ACC = 9.81

#: [rad] rms of `vehicle.road_tilt`, the floor-and-tyre wobble the suspension setpoint carries on
#: top of the cornering and braking terms. It is **not a function of the ego state**, so it is the
#: floor of any estimator built from the ego state -- including a perfect one.
ROAD_TILT_RMS = 0.017


class EgoStateAttitude:
    """Roll and pitch from what the car is DOING, not from what its IMU tilt says.

    The user's proposal (2026-09-15): *"wouldn't it be easier for the model to get this from the
    rate of change of the ego state than from the IMU?"* The body attitude is quasi-static -- the
    suspension is a second-order system driven by the accelerations the car itself produces -- so it
    can be *computed* from those accelerations instead of *integrated* from a gyro. That removes the
    one term `AttitudeTracker` cannot beat: a gyro whose noise on this car is vibration
    (0.278 rad/s rms at 4 m/s, `real_data_calibration.md` §2.6) accumulates
    `sigma * sqrt(tau * dt / 2)` of angle error, and a static map accumulates nothing.

        a_y = v * omega_z                                    (v from the VESC wheel speed)
        a_x = d/dt of the low-passed wheel speed             (held where the wheel is lying)
        roll_ss  =  roll_per_g  * a_y / g
        pitch_ss = -(squat_per_g if a_x > 0 else dive_per_g) * a_x / g

    then, optionally, the suspension's own second-order response to that setpoint, which is what the
    simulator integrates (`sim.py`: `roll_acc = wn^2 (roll_ss - roll) - 2 zeta wn roll_rate`). The
    static setpoint is right in steady state and early by the suspension's rise time in a transient;
    the filtered version costs two states and gets the transient too.

    **Where the wheel lies.** `/odom` speed is the *wheel* speed and this car locks its wheels:
    §2.12 measures -40 to -143 m/s^2 of wheel deceleration against -3 to -16 of body. Differentiating
    that raw reads a brake lock as 10 g of deceleration and produces 25 degrees of phantom dive. So
    the derivative is low-passed first and then **held** wherever it exceeds what the body can
    actually do -- the same quantity `f1sim_ros.traction.TractionGuard` detects and for the same
    reason.

    **Its floor, and it is not small.** `vehicle.road_tilt` is an OU process of 1 degree rms
    (`ROAD_TILT_RMS`) added to the suspension setpoint, and it is not a function of the ego state.
    No estimator of this family can beat it. Measured against the truth this one lands at roughly
    that floor, which is better than every IMU path and still wider than the geometric channel wants
    -- see `docs/research/floor-mask-2026-09-15.md`.
    """

    #: [m/s^2] the largest body longitudinal acceleration this car produces
    #: (`params.VehicleParams.a_max` 7.0 on the drive side). A wheel-speed derivative past this is
    #: the wheel slipping, not the car accelerating.
    A_BODY_MAX = 12.0
    #: [s] low-pass on the wheel speed before differentiating, and on the yaw rate. 0.08 s keeps the
    #: suspension's own band (a few Hz) and removes the ERPM quantisation and the timestamp jitter
    #: `odom.VescOdom` models.
    TAU_V = 0.08
    TAU_W = 0.04
    #: The suspension, from `params.VehicleParams`. `lag=False` uses the setpoint directly.
    WN = 20.0
    ZETA = 0.7

    def __init__(self, batch: int, device="cpu", dt: float = 0.025, dtype=torch.float32,
                 roll_per_g: float = ROLL_PER_G, squat_per_g: float = SQUAT_PER_G,
                 dive_per_g: float = DIVE_PER_G, lag: bool = True):
        self.batch, self.dt = int(batch), float(dt)
        self.device, self.dtype = torch.device(device), dtype
        self.roll_per_g, self.squat_per_g, self.dive_per_g = (float(roll_per_g), float(squat_per_g),
                                                              float(dive_per_g))
        self.lag = bool(lag)
        z = lambda n=1: torch.zeros(self.batch, n, device=self.device, dtype=dtype)
        self.v_lp = z()[:, 0]
        self.w_lp = z()[:, 0]
        self.ax = z()[:, 0]
        self.att = z(2)
        self.rate = z(2)
        self.started = torch.zeros(self.batch, dtype=torch.bool, device=self.device)

    def reset(self, done=None) -> None:
        """An episode boundary: a new car at a new speed, so the filters start again."""
        if done is None:
            for t in ("v_lp", "w_lp", "ax"):
                getattr(self, t).zero_()
            self.att.zero_(); self.rate.zero_(); self.started.zero_()
            return
        d = done if torch.is_tensor(done) else torch.as_tensor(done, device=self.device)
        keep = (~d.bool()).to(self.dtype)
        self.v_lp = self.v_lp * keep; self.w_lp = self.w_lp * keep; self.ax = self.ax * keep
        self.att = self.att * keep[:, None]; self.rate = self.rate * keep[:, None]
        self.started = self.started & ~d.bool()

    @torch.no_grad()
    def update(self, speed: torch.Tensor, yaw_rate: torch.Tensor) -> torch.Tensor:
        """One control step. `speed` (B,) the VESC wheel speed [m/s], `yaw_rate` (B,) [rad/s].

        Returns (B, 2) roll and pitch [rad].
        """
        for t, n in ((speed, "speed"), (yaw_rate, "yaw_rate")):
            if t.dim() != 1 or t.shape[0] != self.batch:
                raise ValueError(f"{n} must be ({self.batch},), got {tuple(t.shape)}")
        dt = self.dt
        v, w = speed.to(self.dtype), yaw_rate.to(self.dtype)
        kv, kw = min(1.0, dt / self.TAU_V), min(1.0, dt / self.TAU_W)
        # First sample: adopt, do not filter toward it from a zero that means "parked".
        first = ~self.started
        v_new = torch.where(first, v, self.v_lp + kv * (v - self.v_lp))
        self.w_lp = torch.where(first, w, self.w_lp + kw * (w - self.w_lp))
        ax_raw = (v_new - self.v_lp) / dt
        self.v_lp = v_new
        # Held where the wheel is lying, and zero before there is a derivative to take.
        usable = (~first) & (ax_raw.abs() <= self.A_BODY_MAX)
        self.ax = torch.where(usable, ax_raw, self.ax)
        self.started = torch.ones_like(self.started)
        ay = self.v_lp * self.w_lp
        roll_ss = self.roll_per_g * ay / G_ACC
        gain = torch.where(self.ax > 0, torch.full_like(self.ax, self.squat_per_g),
                           torch.full_like(self.ax, self.dive_per_g))
        pitch_ss = -gain * self.ax / G_ACC
        ss = torch.stack([roll_ss, pitch_ss], 1)
        if not self.lag:
            self.att = ss
            return self.att
        acc = self.WN * self.WN * (ss - self.att) - 2.0 * self.ZETA * self.WN * self.rate
        self.rate = self.rate + acc * dt
        self.att = self.att + self.rate * dt
        return self.att

    def state_vector(self) -> torch.Tensor:
        """(B, 4) the physically meaningful quantities this estimator forms: filtered speed, filtered
        yaw rate, longitudinal acceleration, lateral acceleration.

        Deliverable 5 feeds these to the front-end alongside the raw IMU rows, because they are what
        the attitude is a function of and the network should not have to rediscover the map.
        """
        return torch.stack([self.v_lp, self.w_lp, self.ax, self.v_lp * self.w_lp], 1)
