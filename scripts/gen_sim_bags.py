#!/usr/bin/env python3
"""Record the simulator into rosbag2 bags that `scripts/replay_traction.py --root` can score.

This is the acceptance harness for the wheel model (CONTRACT.md deliverable 6): the simulated
wheel-acceleration distribution has to reach the tails the 22 real recordings show, at the measured
rates of occurrence, and the guard has to fire on the simulated events with the thresholds it uses
on the car. Both are decided by running the *same* scorer over these bags and over `real_data/`,
which is why they are written in the same format rather than compared through a bespoke metric.

    python3 scripts/gen_sim_bags.py --out /tmp/simbags            # 18 bags, ~60 s each
    python3 scripts/replay_traction.py --root /tmp/simbags --json /tmp/sim.json
    python3 scripts/wheelslip_compare.py /tmp/sim.json /tmp/real.json

What is driven
--------------
Open loop, because the thing under test is the plant and not a policy. Two profiles, both on an
empty floor, both factorised over friction (mu 0.73 / 0.94 / 1.15), geometry (straight / corner) and
seed. Randomisation is off, so the friction each bag is labelled with is the friction it ran at.

**`racepace`** (the default, and the one the *rate* of slip events is scored on) drives a command
process calibrated to the recordings' own `/drive`, so that how often a wheel lets go is the plant's
answer and not the script's. Measured over the 1088 s of commanded motion in the 22 bags:

    commanded speed        mean 3.56, sd 1.74 m/s; p5 0.26, p50 3.39, p95 6.90, max 9.56; 4.6 % zero
    change over 250 ms     sd 1.17 m/s, p50 exactly 0 (the command is held, then stepped)
    hard brake requests    a drop of >= 2 m/s inside 250 ms: 0.291 per second of motion
    hard launch requests   a rise of >= 2 m/s inside 250 ms: 0.280 per second
    steering command       sd 0.189 rad, |mean| 0.161, max 0.45; autocorrelation 0.71 at 200 ms

Reproduced here as a held level redrawn at a Poisson rate (`CMD_RATE`, set so the >= 2 m/s step rate
lands on the measured one) plus an Ornstein-Uhlenbeck steering angle at the measured sd and
correlation time. `scripts/wheelslip_compare.py --driver` checks the generated bags against every
row above; if the driver does not match, the rate comparison downstream means nothing.

**`raceline`** drives the project's own `RacelineTeacher` around the two real venues the recordings
were made on (`real:korea_2026_competition`, the competition floor, and `real:map16x07`), with
`label_grip = "nominal"` -- the teacher assumes the nominal friction rather than the episode's own,
which is what the stack that made the recordings did, and is what makes it over-drive a slippery
floor. This is the profile that brakes into corners at the limit, which is where the deepest locks
live and which an uncorrelated speed/steering process never reaches.

**`limit`** is the contract's explicit torture case: `stand -> hard launch to 9 m/s -> cruise ->
hard brake to 0`, every 5 s, at each friction, straight and at a fixed steering angle. Its event
*rate* is a property of the script (every cycle is a full-authority manoeuvre) and is reported as
such; what it is for is the *reach* of the tail and the guard's behaviour on the strongest events.

What is recorded
----------------
Exactly the four topics the guard and the replay read, with the units the car publishes them in:

    /odom              nav_msgs/Odometry             twist.linear.x = the ERPM wheel speed, stamped
                                                     with `StepResult.odom_t` (jittered, as measured)
    /sensors/imu/raw   sensor_msgs/Imu               linear_acceleration in **g**, one message per
                                                     emulated sample (~50 Hz), stamped from
                                                     `StepResult.imu_offsets`
    /sensors/core      vesc_msgs/VescStateStamped    state.current_motor [A]
    /drive             ackermann_msgs/...            the commanded speed, so the replay can score
                                                     what the guard would have done to it

The record timestamp is what `calib/bagread.py` reads, so that is where the jitter goes; the header
stamp is written to match it.

One liberty, stated because it is invisible in the output: the car's *position* is wrapped back to
the origin whenever it leaves a box, so a 60 s straight-line run does not need a 400 m floor. Only
`state[:, :2]` is touched -- no velocity, no wheel speed, no sensor state -- and none of the four
recorded topics carries position, so nothing in the scoring can see it.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "f1sim"))

from f1sim import Config, Simulator, Track          # noqa: E402
from f1sim.calib.bagread import G                   # noqa: E402

EPOCH_NS = 1_780_000_000_000_000_000               # arbitrary, fixed: bagread works in differences
TOPICS = {
    "/odom": "nav_msgs/msg/Odometry",
    "/sensors/imu/raw": "sensor_msgs/msg/Imu",
    "/sensors/core": "vesc_msgs/msg/VescStateStamped",
    "/drive": "ackermann_msgs/msg/AckermannDriveStamped",
}

#: (mu, steer [rad], name). For `limit` the cornering angle is fixed and chosen so the cruise speed
#: below puts about 0.7 of mu*g on the tyres at mu = 1.05 -- a corner being driven, not a car being
#: spun -- which leaves little of the friction circle for Fx and is where the deepest locks live.
#: For `racepace` it is the sd of the steering process instead, at the measured 0.189 rad.
CELLS = [(mu, steer, kind)
         for mu in (0.73, 0.94, 1.15)
         for steer, kind in ((0.0, "straight"), (0.12, "corner"))]

# ---- the measured driver (see the module docstring) -------------------------------------------
CMD_MEAN, CMD_SD = 3.56, 1.74      # [m/s] commanded speed while moving
CMD_MAX = 9.56                     # [m/s] the largest command in the recordings
CMD_ZERO_FRAC = 0.046              # fraction of commanded time at exactly zero
CMD_RATE = 1.00                    # [1/s] Poisson rate at which the held level is redrawn. Two
                                   # measured statistics pin it and they pull slightly apart, because
                                   # the real command is not a pure jump process: at 1.0/s the sd of
                                   # the 250 ms change comes out 1.23 against a measured 1.17, and the
                                   # hard-request rates come out 0.26 against 0.291 / 0.280. Both are
                                   # inside 1.15x, which is as close as a two-parameter driver gets.
STEER_SD, STEER_TAU = 0.189, 0.60  # [rad], [s]: sd and correlation time of the steering command
                                   # (autocorrelation 0.71 at 200 ms -> tau = 0.59 s)
STEER_MAX = 0.45                   # [rad] the largest steering command in the recordings


def open_field(size=60.0, res=0.15):
    n = int(size / res)
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return Track.from_occupancy(occ, res, (-size / 2, -size / 2))


def profile(t: float, cycle=5.0, launch_v=9.0, cruise_v=4.5, settle=3.0):
    """Commanded speed at time `t`: stand, hard launch, cruise, hard brake, repeat."""
    if t < settle:
        return 0.0
    p = (t - settle) % cycle
    if p < 2.0:
        return launch_v            # hard launch: the VESC is asked for far more than it can hold
    if p < 3.5:
        return cruise_v
    return 0.0                     # hard brake: the regen limit against whatever grip is left


class RacepaceDriver:
    """The recordings' own command process: a held speed level, and an OU steering angle.

    Deliberately open loop and blind: it does not know the car's speed, so nothing about it can
    react to a wheel letting go. That is the point -- every slip in a `racepace` bag is the plant
    responding to a command a human's planner would plausibly have issued.
    """

    def __init__(self, B: int, dt: float, steer_sd, gen: torch.Generator, device):
        self.B, self.dt, self.gen, self.device = B, dt, gen, device
        self.steer_sd = torch.as_tensor(steer_sd, dtype=torch.float32, device=device)
        self.v = torch.full((B,), CMD_MEAN, device=device)
        self.st = torch.zeros(B, device=device)
        self.k = math.exp(-dt / STEER_TAU)
        self.s_ou = math.sqrt(max(0.0, 1.0 - self.k * self.k))

    def __call__(self):
        redraw = torch.rand(self.B, device=self.device, generator=self.gen) < CMD_RATE * self.dt
        lvl = (CMD_MEAN + CMD_SD * torch.randn(self.B, device=self.device, generator=self.gen)
               ).clamp(0.0, CMD_MAX)
        zero = torch.rand(self.B, device=self.device, generator=self.gen) < CMD_ZERO_FRAC
        self.v = torch.where(redraw, torch.where(zero, torch.zeros_like(lvl), lvl), self.v)
        self.st = (self.st * self.k
                   + self.steer_sd * self.s_ou
                   * torch.randn(self.B, device=self.device, generator=self.gen)
                   ).clamp(-STEER_MAX, STEER_MAX)
        return self.st, self.v


#: `raceline` cells: the two venues the recordings were made on, at each friction.
RACELINE_MAPS = ("real:korea_2026_competition", "real:map16x07")
RACELINE_CELLS = [(mu, m) for mu in (0.73, 0.94, 1.15) for m in RACELINE_MAPS]
#: Speed cap for the teacher, m/s. One parameter, set so the commanded-speed distribution it
#: produces lands on the recordings' (mean 3.56, p95 6.90); the teacher's own speed profile then
#: decides where in a lap the car is slow, which is the part that matters.
RACELINE_V_CAP = 7.0


def run_raceline(seed: int, secs: float, device: str, dr: bool, v_cap: float):
    """The teacher driving the real venues. Returns per-env record lists, one bag's worth each."""
    from f1sim.learn import common
    from f1sim.teacher import RacelineTeacher

    names = [c[1] for c in RACELINE_CELLS]
    trs, rls = common.load_tracks(list(dict.fromkeys(names)), racelines=True)
    order = {n: i for i, n in enumerate(dict.fromkeys(names))}
    tid = torch.tensor([order[c[1]] for c in RACELINE_CELLS])
    cfg = Config()
    cfg.rand.enabled = dr
    cfg.sim.device = device
    cfg.sim.compile = False
    cfg.sim.seed = seed
    cfg.vehicle.wheel_model = True
    # Soft walls: a recording is not an episode, and freezing a car that brushed a hose would stop
    # the bag rather than carry on the way the real one does.
    cfg.sim.terminate_on_collision = False
    B = len(RACELINE_CELLS)
    sim = Simulator(trs, cfg, num_envs=B, device=device, track_ids=tid)
    sim.reset()
    sim.P["mu"].copy_(torch.tensor([c[0] for c in RACELINE_CELLS], device=sim.device))
    teacher = RacelineTeacher(rls, wheelbase=cfg.vehicle.lf + cfg.vehicle.lr, device=sim.device)
    # The stack that made the recordings had no friction estimate, so neither does this one: it
    # plans on the nominal grip and over-drives a slippery floor, which is the whole point.
    teacher.label_grip = "nominal"

    n = int(round(secs / sim.control_dt))
    rec = [dict(odom=[], imu=[], core=[], drive=[]) for _ in range(B)]
    for k in range(n):
        t = k * sim.control_dt
        cmd = teacher(sim.state, sim.P, sim.tid)
        cmd = torch.stack([cmd[:, 0], cmd[:, 1].clamp(max=v_cap)], 1)
        r = sim.step(cmd)
        _collect(rec, sim, r, t, cmd[:, 0], cmd[:, 1])
    return rec


def _collect(rec, sim, r, t, st_cmd, v_cmd):
    """Append one control step's worth of records for every env."""
    odom_t = r.odom_t.cpu().numpy()
    odom_v = r.odom[:, 3].cpu().numpy()
    cur = r.motor_current.cpu().numpy()
    imu = r.imu.cpu().numpy()                                   # (B, K, 6)
    off = r.imu_offsets.cpu().numpy()                           # (K,)
    st_h, v_h = st_cmd.cpu().numpy(), v_cmd.cpu().numpy()
    for b in range(len(rec)):
        rec[b]["odom"].append((float(odom_t[b]), float(odom_v[b])))
        rec[b]["core"].append((float(odom_t[b]), float(cur[b])))
        rec[b]["drive"].append((t, float(st_h[b]), float(v_h[b])))
        for j in range(imu.shape[1]):
            rec[b]["imu"].append((r.t - float(off[j]), imu[b, j].tolist()))


def run_one(seed: int, secs: float, wrap: float, device: str, profile_name: str, dr: bool = True):
    """One simulator run. Returns per-env record lists, one bag's worth each."""
    cfg = Config()
    # Domain randomisation on, but with `vehicle.mu` overwritten per env below, so the friction each
    # bag is labelled with is its own while everything else -- I_w, the drivetrain split, the slip
    # curve, a_brake, the ERPM gain, the shock rate -- is drawn the way a training run draws it.
    # The 22 recordings span three floors, several months, two VESC configurations and whatever
    # state the tyres were in; a simulated set with every parameter pinned at nominal is not the
    # comparable population. `--no-dr` pins them for a single-parameter study.
    cfg.rand.enabled = dr
    cfg.sim.device = device
    cfg.sim.compile = False
    cfg.sim.seed = seed
    cfg.vehicle.wheel_model = True
    B = len(CELLS)
    sim = Simulator(open_field(), cfg, num_envs=B, device=device)
    sim.reset(poses=torch.zeros(B, 3))
    sim.P["mu"].copy_(torch.tensor([c[0] for c in CELLS], device=sim.device))
    steer = torch.tensor([c[1] for c in CELLS], device=sim.device)

    n = int(round(secs / sim.control_dt))
    driver = None
    if profile_name == "racepace":
        # "straight" keeps the steering at zero; "corner" runs it at the measured sd.
        driver = RacepaceDriver(B, sim.control_dt,
                                [0.0 if c[1] == 0.0 else STEER_SD for c in CELLS],
                                sim.gen, sim.device)
    rec = [dict(odom=[], imu=[], core=[], drive=[]) for _ in range(B)]
    for k in range(n):
        t = k * sim.control_dt
        if driver is None:
            st_cmd = steer
            v_cmd = torch.full((B,), profile(t), device=sim.device)
        else:
            st_cmd, v_cmd = driver()
        cmd = torch.stack([st_cmd, v_cmd], 1)
        r = sim.step(cmd)
        # Wrap the pose, not the motion. See the module docstring.
        xy = sim.state[:, :2]
        sim.state[:, :2] = torch.remainder(xy + wrap, 2 * wrap) - wrap
        sim.pose_prev[:, :2] = sim.state[:, :2]

        _collect(rec, sim, r, t, st_cmd, v_cmd)
    return rec


def write_bag(path: str, records: dict) -> None:
    """One rosbag2 sqlite3 bag with the four topics, records written in timestamp order."""
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from ackermann_msgs.msg import AckermannDriveStamped
    from vesc_msgs.msg import VescStateStamped

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    for topic, typ in TOPICS.items():
        writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=typ,
                                                     serialization_format="cdr"))

    def stamp(msg, t):
        ns = EPOCH_NS + int(round(t * 1e9))
        msg.header.stamp.sec = ns // 1_000_000_000
        msg.header.stamp.nanosec = ns % 1_000_000_000
        return ns

    out = []
    for t, v in records["odom"]:
        m = Odometry(); ns = stamp(m, t)
        m.header.frame_id = "odom"; m.child_frame_id = "base_link"
        m.pose.pose.orientation.w = 1.0
        m.twist.twist.linear.x = float(v)
        out.append((ns, "/odom", serialize_message(m)))
    for t, six in records["imu"]:
        m = Imu(); ns = stamp(m, t)
        m.header.frame_id = "imu"
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(x) for x in six[:3])
        # The car publishes linear_acceleration in g; `calib/bagread.read` multiplies by G on the
        # way back in, so anything written here in m/s^2 would be scored 9.8x too large.
        m.linear_acceleration.x = float(six[3]) / G
        m.linear_acceleration.y = float(six[4]) / G
        m.linear_acceleration.z = float(six[5]) / G
        out.append((ns, "/sensors/imu/raw", serialize_message(m)))
    for t, cur in records["core"]:
        m = VescStateStamped(); ns = stamp(m, t)
        m.state.current_motor = float(cur)
        m.state.voltage_input = 11.1
        out.append((ns, "/sensors/core", serialize_message(m)))
    for t, st, v in records["drive"]:
        m = AckermannDriveStamped(); ns = stamp(m, t)
        m.drive.steering_angle = float(st); m.drive.speed = float(v)
        out.append((ns, "/drive", serialize_message(m)))
    out.sort(key=lambda e: e[0])
    for ns, topic, raw in out:
        writer.write(topic, raw, ns)
    del writer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory to write the bags into")
    ap.add_argument("--profile", choices=["raceline", "racepace", "limit"], default="raceline",
                    help="raceline: the teacher driving the two real venues, which is the closest "
                         "thing to the driver that made the recordings. racepace: an open-loop "
                         "command process with the recordings' own statistics. limit: back-to-back "
                         "hard launches and hard brakes, which is what the tail's REACH is shown on.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-dr", action="store_true",
                    help="pin every parameter but mu at nominal instead of drawing the "
                         "randomisation ranges a training run draws")
    ap.add_argument("--secs", type=float, default=60.0, help="length of each recording")
    ap.add_argument("--wrap", type=float, default=20.0, help="[m] half-width of the pose wrap box")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--v-cap", type=float, default=RACELINE_V_CAP,
                    help="[m/s] speed cap for the raceline profile's teacher")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for seed in range(a.seeds):
        if a.profile == "raceline":
            rec = run_raceline(seed, a.secs, a.device, not a.no_dr, a.v_cap)
            cells = [(mu, m.split(":")[-1]) for mu, m in RACELINE_CELLS]
        else:
            rec = run_one(seed, a.secs, a.wrap, a.device, a.profile, dr=not a.no_dr)
            cells = [(mu, kind) for mu, _st, kind in CELLS]
        for b, (mu, kind) in enumerate(cells):
            name = f"sim-{a.profile}-{seed:02d}_mu{mu:.2f}_{kind}_{int(a.secs)}s"
            path = os.path.join(a.out, name)
            if os.path.exists(path):
                raise SystemExit(f"{path} exists; refusing to write over a recording")
            write_bag(path, rec[b])
            n = len(rec[b]["odom"])
            print(f"  {name:44s} {n:6d} /odom  {len(rec[b]['imu']):6d} /sensors/imu/raw")
            total += 1
    print(f"wrote {total} bags under {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
