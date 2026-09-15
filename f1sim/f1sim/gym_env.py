"""Gymnasium-style vectorized environment for the LiDAR-only e2e policy.

Observation (what the real car can provide, nothing else):
    scan        (k, n_beams)   last k scans, ranges / range_max, no-return -> 1.0
    speed       (1,)           VESC measured speed / v_max
    prev_action (2,)           previous normalized action
Privileged info for an asymmetric critic is in `info["priv"]` (B, 8): vx, vy, yaw_rate,
lateral offset, heading error to centerline, s / L, curvature ahead proxy (progress rate), wall clearance.

Action (B, 2) in [-1, 1]:  steer = a0 * s_max,  speed = (a1 + 1) / 2 * v_max_policy.

Reward: progress along the centerline [m] (=> minimum lap time), collision penalty,
small steering-rate penalty. Episodes end on collision (terminated) or after max_steps /
`laps` completed laps (truncated). Auto-reset: envs that end are reset in the same step and
their first observation of the new episode is returned (standard vector-env semantics).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Optional

import math

import numpy as np
import torch

from .mpc import ACT_DIM as PLAN_DIM, PlanSpec, PlanTracker, encode as plan_encode, decode as plan_decode
from .opponent_events import (NO_EVENT, LearnerView, OpponentEvents, raceline_corners,
                               raceline_offset_limit, split_events)
from .learn.obs import ATT_SCALE, MessageInputs, norm_att, norm_imu, norm_scan, norm_speed
from .params import Config
from .sim import Simulator, StepResult
from .track import Track

REWARD_COMPONENT_KEYS = ("progress", "collision", "collision_speed", "steer_rate", "proximity",
                         "plan_clearance", "wrong_way", "lap", "alive", "car_contact", "overtake",
                         "lap_time", "car_proximity", "sideslip")

#: The nearest-opponent columns of privileged() are stored as metres (or m/s) divided by this, to
#: keep them O(1) for the critic. Anything comparing them against a real distance must multiply.
PRIV_OPP_DIST_SCALE = 5.0

#: The columns of `F1VecEnv.future_labels`, in order. The first six are regression targets for the
#: auxiliary future head (`f1sim.learn.future`); the last is the target of a presence *logit*, 1
#: where a car is inside `overtake_range`. The names are what the trainer and the probe log, so
#: changing one renames a metric in every table that has ever been produced.
FUTURE_LABEL_KEYS = ("opp_lon", "opp_lat", "opp_vlon", "opp_vlat",
                     "ego_speed", "ego_yaw_rate", "opp_present")
FUTURE_LABEL_DIM = len(FUTURE_LABEL_KEYS)
FUTURE_PRESENT_INDEX = FUTURE_LABEL_KEYS.index("opp_present")

#: Horizons [s] the privileged opponent block reports future positions at, and the grid the walk
#: that produces them runs on. Fixed here rather than at each call site so that the token layout,
#: `InteractiveTeacher`'s cost and `learn.opp_future_check`'s validation all read one list.
OPP_FUTURE_TIMES = (0.1, 0.25, 0.5, 0.75)
OPP_FUTURE_WALK_DT = 0.05

#: [m] of TRAVEL over which a raceline walk is allowed to still carry the car's present tracking
#: error. Every prediction is anchored at where the car actually is at t = 0 -- a walk otherwise
#: starts at the nearest point of the LINE, up to half a metre from the car, and that offset shows
#: up in the label as a lateral velocity the opponent does not have.
#:
#: Metres and not seconds, which is what this was first written as. A pure-pursuit tracker closes a
#: lateral error over its lookahead, which is a DISTANCE; on a clock, a car that has just been told
#: to stop is predicted to slide half a metre sideways onto the line while standing still, and the
#: braking event this label exists to report is then swamped by it. 2 m is the tracker's own
#: lookahead range (`RacelineTeacher.ld_min` .. `ld_max` is 0.6 .. 2.5 m).
#:
#: The plan tracker's own prediction keeps its offset instead of closing it: there the mismatch is
#: an alignment artefact (the trajectory was solved one control step ago, from the
#: latency-compensated pose) rather than a tracking error.
OPP_FUTURE_REJOIN_M = 2.0

#: How `car_future` predicts where a car will be.
#:   "plan"   -- each car's OWN controller carried forward: a teacher-driven opponent is walked
#:               along its raceline at its commanded speed with its scheduled event applied (the
#:               brake/stop speed scale for as long as the event has left to run, the lateral
#:               offset it is holding), and a policy-driven one is read off its plan tracker's
#:               predicted trajectory. Neither is a peek at the future: both are the controller's
#:               own intention, which the simulator already knows this step.
#:   "pred"   -- THE DEFAULT. The plan tracker's own predicted trajectory for every car, whoever
#:               drives it, continued straight at its final heading and speed past its 0.6 s
#:               horizon. It is the iLQR's forward rollout of the plan the car was actually given,
#:               so the plan's braking, its lane change, its heading and its acceleration bounds
#:               are already in it rather than reconstructed -- and measured, that is worth 3x the
#:               walk's accuracy on the very cars the walk models explicitly. It is also what
#:               worker 16's `--opp-token future` uses.
#:   "hybrid" -- the tracker's rollout inside its own horizon and the raceline walk's increments
#:               beyond it, anchored so the two meet. A hypothesis that the measurement REJECTED and
#:               that is kept because the rejection is the useful part: past 0.6 s the rollout is a
#:               straight line on a curving track, so continuing along the road ought to beat it --
#:               and it does not. At 0.75 / 1.0 s, MAE 0.232 / 0.449 m against "pred"'s 0.214 /
#:               0.410. The walk's increments carry the walk's own error, and inheriting that is
#:               worse over 0.4 s than simply going straight.
#:   "constv" -- world-frame constant velocity from the current state. The floor every other model
#:               has to beat, and what a caller with no teacher gets.
OPP_FUTURE_MODELS = ("pred", "hybrid", "plan", "constv")

#: The privileged opponent block (`EnvConfig.opp_token`), an *oracle input*: it is refused by the
#: exporter and by `f1sim_ros.policy_node`, because no car can measure it.
#:   ""        off, and off is the observation the env has always produced ("off" is accepted as a
#:             spelling of it, because that is what worker 16's `--opp-token` flag takes)
#:   "pos"     the nearest cars' relative position and a presence flag
#:   "posvel"  + their relative velocity
#:   "future"  + where each of them will be at `OPP_FUTURE_TIMES`
OPP_TOKEN_MODES = ("", "off", "pos", "posvel", "future")

#: How many opponents the block describes, nearest first. Two rather than one because the situation
#: the whole line of work is about -- picking the gap a car is leaving -- stops being well posed the
#: moment a second car owns the gap.
OPP_TOKEN_CARS = 2

#: Columns per car, in order: the `pos` triple, then the `posvel` pair, then the `future` pairs.
#: The order is the layout, so a checkpoint written under one mode and read under another would see
#: columns that mean the wrong thing; `ObsSpec.opp_token` records which mode produced it.
OPP_TOKEN_COLS = {"": 0, "off": 0, "pos": 3, "posvel": 5, "future": 5 + 2 * len(OPP_FUTURE_TIMES)}


def opp_token_mode(mode: str) -> str:
    """The canonical spelling of an `opp_token` value; "off" and "" are the same thing."""
    if mode not in OPP_TOKEN_MODES:
        raise ValueError(f"opp_token {mode!r} is not one of {list(OPP_TOKEN_MODES)}")
    return "" if mode == "off" else mode


def opp_token_dim(mode: str) -> int:
    """Width of the privileged opponent block for a mode name."""
    return OPP_TOKEN_CARS * OPP_TOKEN_COLS[opp_token_mode(mode)]


#: Who drives each car, per row, in `opponent == "pool"`. The learner's slot is always
#: `OPP_DRIVER_POLICY`; an opponent's is drawn per race from `opp_pool`.
OPP_DRIVER_POLICY = 0      # the caller's own action: the learner, or a `self` entry (self-play)
OPP_DRIVER_TEACHER = 1     # the raceline teacher, carrying whatever `opp_events` are configured
OPP_DRIVER_POOL = 2        # 2 + j: checkpoint j of the pool (f1sim.learn.opponent_pool)

#: The two `--opp-pool` entries that are not files. `self` is the learner's own current weights,
#: which makes that race self-play; `teacher` is the raceline teacher.
POOL_SELF, POOL_TEACHER = "self", "teacher"

#: `--spawn-order`: where the learner starts relative to the rest of its race.
SPAWN_ORDERS = ("behind", "ahead", "alongside", "random")
SPAWN_ORDER_ID = {name: i for i, name in enumerate(SPAWN_ORDERS[:3])}

#: Who may drive the other cars of a race. `slots` is the per-car form: instead of one rule for
#: every opponent, `EnvConfig.opponent_slots` carries one `f1sim.opponent_slots.OpponentSlot` per
#: grid slot and slot i of every race in the batch is built from spec i.
OPPONENT_MODES = ("policy", "teacher", "mixed", "pool", "slots")

#: `OpponentSlot.spawn` as an integer, in `f1sim.opponent_slots.SLOT_SPAWNS` order. Slot 0 -- the
#: learner -- is not placed by a spec and carries this sentinel.
SLOT_SPAWN_LEARNER = -1
SLOT_SPAWN_ID_AHEAD, SLOT_SPAWN_ID_BEHIND, SLOT_SPAWN_ID_ALONGSIDE = 0, 1, 2


def pool_entries(pool) -> tuple:
    """Normalize an `opp_pool` value (tuple/list, or a comma-separated string) of entry names."""
    if pool is None:
        return ()
    if isinstance(pool, str):
        pool = [p for p in pool.replace(" ", "").split(",") if p]
    return tuple(str(p) for p in pool)


@dataclass
class EnvConfig:
    scan_stack: int = 3
    scan_subsample: int = 1          # 1 = all 1080 beams; 4 -> 270 beams
    action_history: int = 2          # previous normalized actions in the observation
    speed_cap: float = 10.0          # [m/s] current commanded-speed cap (curriculum); observed as speed_cap / v_max_policy
    obs_imu: bool = True             # add step-mean IMU (gyro xyz, accel xyz) and VESC roll/pitch estimate
    imu_gyro_scale: float = 5.0      # [rad/s] normalization
    imu_accel_scale: float = 10.0    # [m/s^2]
    v_max_policy: float = 10.0       # top speed the policy may command (raceline profile peaks here)
    max_steps: int = 1600            # 40 s at 40 Hz
    laps: int = 100                  # truncate after this many laps (racing: keep going)
    reward_progress: float = 1.0     # per meter
    reward_collision: float = -10.0
    reward_steer_rate: float = 0.05  # per unit of normalized steer change
    reward_proximity: float = 0.5    # per metre driven, at zero wall gap
    proximity_speed_ref: float = 0.0 # [m/s] >0: the proximity penalty is scaled by (1 + v / ref): fast past a wall costs more than creeping
    reward_wrong_way: float = 0.2    # per step while facing backwards along the lane (progress is signed anyway; this makes it explicit)
    reward_collision_speed: float = 0.0   # extra collision penalty per m/s of speed at impact (a fast crash costs more than a nudge)
    reward_plan_clearance: float = 0.0
    plan_margin: float = 0.15
    safe_dist: float = 0.30          # [m] body-to-wall gap below which the proximity penalty starts
    reward_lap: float = 0.0          # paid once per completed lap, scaled by the lap's average speed
                                     # (= track length / lap time). Progress reward is already a dense
                                     # stand-in for lap time, but it saturates wherever the speed cap
                                     # rather than the corner decides the speed; this term keeps
                                     # rewarding a genuinely faster lap and requires actually finishing one.
    reward_car_contact: float = 0.0   # extra penalty for touching another car, on top of ending the
                                      # episode. Contact and a wall both just end the episode today,
                                      # so nothing tells the policy that a car is a thing to be
                                      # negotiated with rather than another piece of scenery.
    reward_overtake: float = 0.0      # per metre of arc taken out of the opponents in contention.
                                      # Without it a collision penalty teaches avoidance and only
                                      # avoidance: sitting behind is free, so there is no reason to
                                      # ever pass.
    reward_car_proximity: float = 0.0 # per metre driven, at zero body gap to another car, scaled by
                                      # closing speed. The wall has a proximity term; a car had only the
                                      # contact cliff. So the only gradient near another car was the
                                      # overtake term paying for *closing*, all the way down to touching
                                      # -- "get as close as you can" with nothing saying "leave room".
                                      # A forced, dangerous pass is not something the collision penalty
                                      # can be relied on to price: it is sparse and high-variance, and
                                      # by the time it fires the lesson is one crash out of thousands
                                      # of steps.
    car_safe_gap: float = 0.60        # [m] body-to-body gap below which the car proximity penalty starts
    car_len: float = 0.50             # [m] centre-to-centre distance at which bodies touch nose-to-tail
    car_wid: float = 0.30             # [m] ... and side-by-side
    car_body_dist: float = 0.45       # [m] kept for tests and tools that reason about centre distance
    car_prox_speed_ref: float = 2.0   # [m/s] scale by (1 + closing speed / ref): closing fast on a car
                                      # at 0.3 m is worse than sitting beside it
    overtake_range: float = 12.0      # [m] a car further away than this along the lane is not being
                                      # raced, and scoring it is worse than useless: the signed gap
                                      # has to flip somewhere, and the only place it can flip
                                      # harmlessly is out here rather than at the pass. Two cars
                                      # exactly opposite each other on the lap would otherwise trade
                                      # a spurious clamp-sized jump every time they drifted across
                                      # the far side.
    reward_lap_time: float = 0.0      # time reward: paid per sector for beating this track's own
                                      # recent time through that sector. Progress already points at
                                      # lap time -- over a fixed episode it *is* distance/time -- but
                                      # its sensitivity is hopeless: taking 0.1 s off a 13.5 s lap
                                      # moves the return by 0.95 % on a short track and 0.44 % on a
                                      # long one, against a collision worth about -80, and it sees a
                                      # tenth *less* the longer the track, because progress grows
                                      # with length while the value of a tenth does not.
                                      #
                                      # Three things this term has to get right, none of them
                                      # obvious:
                                      #
                                      # 1. It pays a bonus and never a toll. A negative term at the
                                      #    line is hackable for free: crossing costs a fixed toll,
                                      #    while stopping a centimetre short at the end of an episode
                                      #    costs almost no progress reward, so the policy would learn
                                      #    to coast up to the line and wait. One-sided makes
                                      #    finishing always weakly better than not.
                                      #
                                      # 2. It is scored per *sector*, not per lap. Near a track's
                                      #    limit the policy-driven spread in lap time shrinks toward
                                      #    the spread that randomisation and spawn pose produce on
                                      #    their own, so a once-a-lap signal drowns exactly when the
                                      #    remaining tenths are hardest to find. Sectors give ~12x
                                      #    the events with less accumulated noise in each, and the
                                      #    credit lands at the corner that earned it rather than
                                      #    seconds later at the line. It also still pays on tracks
                                      #    where a full lap is rare, which is most obstacle tracks.
                                      #
                                      # 3. It is measured in *headroom halved*, not seconds saved.
                                      #    Flat seconds get the ordering backwards near the limit: a
                                      #    second found while still far off the pace is easy and a
                                      #    hundredth found at the limit is a real result, and flat
                                      #    seconds pay the easy one a hundred times more. The gain is
                                      #    log(h_ref / h_now) where h is the gap to the raceline's
                                      #    time for that sector, so equal *fractions* of the
                                      #    remaining headroom pay equally: 15 s -> 14 s against a
                                      #    10 s limit pays 0.22, and 10.10 s -> 10.09 s pays 0.10 --
                                      #    still nearly half as much for a hundredth of the time.
    reward_time_sectors: int = 12     # sectors per lap. 12 puts a boundary every 3-6 m on these
                                      # tracks: long enough that one bad corner does not smear across
                                      # several sectors, short enough to time a corner on its own.
    reward_time_momentum: float = 0.98   # EMA setting the time to beat, per track *and* sector. The
                                      # raceline ideal is the physical floor but a poor live target:
                                      # it is out of reach for a long time, so a one-sided bonus
                                      # against it would pay nothing and teach nothing. Racing the
                                      # policy's own recent pace keeps about half of sectors in the
                                      # money at every skill level, and the target falls as it gets
                                      # faster.
    reward_time_floor: float = 0.002  # headroom floor as a fraction of the sector's ideal time, so
                                      # the log stays finite where the policy reaches or beats the
                                      # raceline (which is a point-mass optimum, not a hard bound).
                                      # 0.2 % is 0.02 s over a 10 s lap: the floor has to sit below
                                      # the hundredths this term exists to pay for, or it quietly
                                      # switches the reward off exactly at the limit.
    reward_time_max: float = 0.30     # cap on one sector's gain: halving the headroom in a single
                                      # sector is already a large result, and without a cap a sector
                                      # sitting at the floor turns timing noise into a spike.
    reward_sideslip: float = 0.0      # per metre driven, per radian of body sideslip beyond
                                      # sideslip_free. Sliding is only punished today when it ends
                                      # in a wall; the grip limit itself has no price, so on a
                                      # low-friction draw the policy carries the same corner speed
                                      # it uses on a grippy one and finds out at the wall.
    sideslip_free: float = 0.06       # [rad] ~3.5 deg: the drift angle a clean fast corner has anyway
    reward_alive: float = 0.0
    spawn_lateral_std: float = 0.3
    spawn_yaw_std: float = 0.2
    spawn_min_clearance: float = 0.5  # [m] spawn poses closer to a wall are pulled back to the centerline
    spawn_speed_max: float = 3.0     # random initial speed in [0, this]
    resample_track_on_reset: bool = True   # multi-track sets: pick a random track for each new episode
    scan_stride: int = 1             # frames between stacked scans (3 x stride 3 = 225 ms of history: velocity cues)
    hist_len: int = 0                # >0: proprioceptive history rows (speed, imu, roll/pitch, action) in the observation
    hist_stride: int = 2             # steps between rows (20 x 2 = the last second): implicit identification of grip / lag
    action_mode: str = "direct"      # "direct": (steer, speed); "plan": short local trajectory tracked by an MPC (f1sim.mpc)
    compile_tracker: bool = True
    # races: M cars per track instance, visible to each other's LiDAR, car-car contact = collision
    race_size: int = 1
    opponent: str = "policy"         # "policy": every car is driven by the caller (self-play);
                                     # "teacher": cars 1..M-1 follow the raceline teacher at a random speed scale;
                                     # "mixed": per race, teacher or self-play (mixed_teacher_frac);
                                     # "pool": per race, one entry drawn from `opp_pool` -- a
                                     # population rather than a single kind of opponent.
    opp_speed_range: tuple = (0.6, 1.0)   # teacher opponents: speed-profile scale per race per reset.
                                     # Above 1.0 is allowed and means a *faster* car: the learner is
                                     # the one being overtaken, which is the situation the training
                                     # distribution has never contained (the learner always spawned
                                     # behind a car doing 0.6-1.0x). The teacher's profile already
                                     # plans at the grip limit, so a scale much above ~1.2 is a car
                                     # that leaves the road; `learn.opponent_census` reports opponent
                                     # wall contacts so the ceiling is measured rather than assumed.
    mixed_teacher_frac: float = 0.5   # opponent == "mixed": the share of races whose other cars are
                                      # teacher-driven (the rest are self-play). Measured, each alone
                                      # forgets the other: 6.6M steps of self-play took contact
                                      # avoidance against teachers from 8 to 10.5 per 600 steps while
                                      # passes rose, and the teacher stage had plateaued. Opponent
                                      # *behaviour* is a diversity axis like track shape is; a batch
                                      # with one kind of opponent is a training set with one map.
    spawn_gap: tuple = (2.5, 6.0)    # [m] along the lane between cars of a race at spawn
    spawn_order: str = "behind"      # where the learner (slot 0) starts on the grid of a race whose
                                     # other cars are not the learner itself: "behind" (every race so
                                     # far: the learner at the back, with a pass to make), "ahead"
                                     # (the learner leading, so a faster opponent has to be dealt
                                     # with rather than chased), "alongside" (side by side, which is
                                     # where the contacts actually happen) or "random" (drawn per
                                     # race from the three). In a self-play race every car is the
                                     # learner, so only "alongside" is a different grid there.
    spawn_alongside_sep: float = 0.15  # [m] body-to-body lateral gap asked for on an alongside grid.
                                     # What the lane has room for wins: where two bodies plus this
                                     # gap do not fit, the race spawns staggered instead, because
                                     # two cars spawned in contact terminate on step 1 for ever.
    spawn_alongside_gap: tuple = (0.0, 0.2)   # [m] arc stagger inside an alongside grid, drawn
                                     # independently per car. Small on purpose: the lateral offsets
                                     # of a grid are measured along the centerline normal at each
                                     # car's own arc position, and on a tight corner those normals
                                     # stop agreeing a few tens of centimetres apart. The bodies are
                                     # 0.58 m long, so they overlap across the whole range anyway.
    spawn_alongside_yaw: float = 0.06  # [rad] heading jitter of an abreast row, if that is tighter
                                     # than `spawn_yaw_std`. Two cars side by side need their
                                     # *rotated* footprints to fit between the walls, and at the
                                     # usual 0.2 rad a 0.58 m car sweeps 0.33 m sideways at 3 sigma
                                     # -- more than a 1.4 m lane has to spare. Cars on a grid are
                                     # lined up with the track, so this is what a grid looks like
                                     # anyway; the lateral spread, which is the point, is kept.
    selfplay_front_cap: bool = True   # cap the front car of a self-play race (training). The viewer
                                      # turns it off: what is being watched there is the policy
                                      # against an equal, and a silently slowed opponent reads as a
                                      # cap bug ("why is the other car's cap different?")
    selfplay_pace_ref: float = 5.0    # [m/s] the front car of a self-play race is capped at
                                      # opp_speed_range x this, so the car behind has a pass to make
    opp_follow_gap: float = 1.5       # [m] a teacher opponent this close behind another car -- plus its
                                      # braking distance at the closing speed, see follow_cap -- holds
                                      # its speed to that car's. The raceline teacher is blind to other
                                      # cars: measured, 10 of 37 contacts in a race were the teacher
                                      # driving into the learner it had just been passed by. A pass
                                      # that ends with the passed car ramming you is not a lesson
                                      # about passing, and no opponent that can see drives like that.
    opp_follow_ratio: float = 0.9     # fraction of the car-ahead's speed the follower settles to
    opp_follow_decel: float = 3.0     # [m/s^2] what the tracker actually delivers when asked to brake
                                      # from a plan: measured 3.1 against its nominal 5.0 bound, because
                                      # the reference is walked from the current speed and the speed
                                      # weight in the tracking cost is soft. A fixed 3 m trigger let a
                                      # teacher at 4 m/s run straight into a parked car.
    # Scripted opponent *behaviour* (f1sim.opponent_events). Empty = off, and off is bit-identical to
    # the env before events existed: nothing is stepped and nothing is drawn from the generator.
    # A teacher opponent otherwise presents one problem -- a slightly slower car on the racing line --
    # and the two benchmark dimensions the policy is worst at (obstacle avoidance, overtaking) are the
    # two that need the other problems: a car that brakes, a car that has stopped, a car that moves
    # across the lane. Widening opp_speed_range / spawn_gap was measured to move neither.
    opp_events: tuple = ()            # any of ("brake", "stop", "shift", "weave"), or a comma-separated string
    opp_event_rate: float = 0.0       # expected events per teacher opponent per 10 s of driving
    opp_brake_scale_range: tuple = (0.0, 0.5)     # brake: fraction of its profile speed it drops to
    opp_brake_time_range: tuple = (0.5, 2.5)      # [s] how long it holds that
    opp_stop_time_range: tuple = (1.0, 4.0)       # [s] how long a stopped car stays stopped
    opp_shift_offset_range: tuple = (0.0, 0.35)   # [m] |lateral offset| of a lane change; the sign is
                                                  # drawn separately, so the offset is U(-hi, +hi) in effect
    opp_shift_hold_range: tuple = (0.5, 2.0)      # [s] time at the offset, between the two ramps
    opp_shift_ramp: float = 1.0                   # [s] ramp in and ramp out of the offset
    opp_weave_amp_range: tuple = (0.1, 0.25)      # [m] sinusoidal offset amplitude
    opp_weave_period_range: tuple = (2.0, 4.0)    # [s]
    opp_weave_time_range: tuple = (2.0, 6.0)      # [s] how long a weave lasts
    opp_event_margin: float = 0.10    # [m] free space kept beyond the car's half-width when an event
                                      # offsets it off the raceline. The offset is clamped per raceline
                                      # point against the track's own distance field, so a 0.35 m lane
                                      # change through a 1.6 m section becomes as much of one as fits.
    # Reactive behaviour (the `defend` / `yield` / `line` / `oblivious` entries of `opp_events`).
    # These are not timed: they read the learner's position relative to the opponent every step, so
    # each is a *disposition* drawn per teacher-driven car per race at its own probability rather
    # than an event at a rate. All four probabilities are 0, i.e. off, by default.
    opp_defend_prob: float = 0.0      # P(this opponent defends the inside line when caught)
    opp_yield_prob: float = 0.0       # P(this opponent moves away from a car alongside)
    opp_line_prob: float = 0.0        # P(this opponent drives its own out-in / in-out corner line)
    opp_oblivious_prob: float = 0.0   # P(this opponent's follow-gap slowdown is switched off)
    opp_defend_offset_range: tuple = (0.15, 0.35)   # [m] how far a defending car moves across
    opp_yield_offset_range: tuple = (0.15, 0.35)    # [m] ... a yielding one
    opp_line_offset_range: tuple = (0.15, 0.35)     # [m] ... a corner line, at each end of the sweep
    opp_defend_range: float = 0.0     # [m] how far behind a learner starts being defended against.
                                      # 0 = `overtake_range`, the same window the overtake reward
                                      # calls "being raced".
    opp_defend_full: float = 3.0      # [m] inside this the block is at full amplitude; it ramps to
                                      # zero at `opp_defend_range`. A car that swings off the racing
                                      # line because something is 12 m behind it is not defending.
    opp_alongside_lon: float = 0.8    # [m] |body-frame longitudinal offset| within which two cars
                                      # count as alongside (the bodies are 0.58 m long). Read by
                                      # `yield` and by `learn.opponent_census`, which must measure
                                      # the same situation the behaviour reacts to.
    opp_alongside_lat: float = 1.2    # [m] ... and the lateral bound, so a car on the far side of a
                                      # hairpin is not "alongside" anyone.
    opp_react_max: float = 0.45       # [m] cap on the reactive offset before the lane's own clamp.
                                      # Half a car width of line change is a racing move; a metre is
                                      # a car leaving the road in a way no clamp should have to save.
    opp_react_slew: float = 0.6       # [m/s] rate limit on it. The scripted `shift` ramps 0.35 m
                                      # over 1 s for the same reason: pure pursuit on a target that
                                      # jumps sideways asks for a step steer input.
    opp_corner_kappa: float = 0.15    # [1/m] smoothed raceline curvature above which a point is
                                      # "in a corner" (a 6.7 m radius), for the `line` behaviour
    opp_corner_min_arc: float = 1.0   # [m] shortest run of such points that counts as one corner
    opp_corner_smooth: float = 0.6    # [m] window the curvature is averaged over first. Without it
                                      # a single corner comes apart into a dozen one-point corners
                                      # whose phase sweep is a few centimetres long.
    # Opponent population. `opponent == "pool"`: each entry is a checkpoint path, `self` (the
    # learner's own current weights, which makes that race self-play) or `teacher` (the raceline
    # teacher, carrying whatever `opp_events` are configured). One entry is drawn per race from
    # `sim.gen`. Empty is off and refused by `opponent == "pool"` rather than silently ignored.
    #
    # Why a population at all: measured, a single kind of opponent is a training set with one map.
    # `opponent == "mixed"` already interleaves two kinds; this makes the number of kinds a flag,
    # and lets one of them be a *policy* -- a car that takes its own line, defends its own position
    # and makes its own mistakes, which no scripted behaviour reproduces.
    opp_pool: tuple = ()
    # Privileged opponent block in the observation (`OPP_TOKEN_MODES`). "" is off and off is the
    # observation this env has always produced. Anything else is an ORACLE: it is ground truth from
    # the simulator, appended after every proprio key the policy already had, and it is refused by
    # `learn.export` and by `f1sim_ros.policy_node` so that a checkpoint trained on it can never be
    # mistaken for one that could drive a car.
    opp_token: str = ""
    # Which prediction `car_future` (and so the "future" columns of the block) uses -- see
    # `OPP_FUTURE_MODELS`. Recorded in the spec next to the mode, because "the opponent's future"
    # under two different models is two different labels.
    opp_future_model: str = "pred"
    # Per-opponent configuration (`f1sim.opponent_slots`), the axis every field above is missing:
    # they describe "the opponents" and a race has *opponents*. One `OpponentSlot` per grid slot,
    # `race_size - 1` of them, and slot i of every race in the batch is built from spec i -- driver
    # kind, checkpoint, speed profile, grip label, speed cap, events, reactive probabilities and
    # where that car starts. `opponent` must be "slots" when this is set, and None (the default) is
    # the feature switched off: the env then runs the instructions it ran before slots existed, so
    # every checkpoint and benchmark number measured on that path is reproduced byte for byte.
    #
    # A list of dicts is accepted as well as a tuple of `OpponentSlot`, so a session config or a
    # `--opp-slots` JSON needs no conversion step of its own.
    opponent_slots: Optional[tuple] = None
    # Obstacle layouts redrawn per env at every reset (f1sim.procedural_obstacles). 0 = off, and off
    # is byte-identical to the env before they existed: nothing is allocated, nothing is drawn from
    # the generator and `props_for` returns exactly what it returned. The training set's obstacle
    # layouts are otherwise fixed -- `+rlobs`, `+obs`, `+hard<seed>` are rasterised once at load --
    # so speed through a layout can be learned by knowing the layout, which is what the held-out
    # proxy says is happening (`docs/research/procedural-obstacles-2026-09-13.md`).
    procedural_obstacles: float = 0.0   # share of resets that get a freshly drawn layout
    procedural_density: float = 1.0     # patterns per 10 m of lap
    procedural_max_props: int = 0       # prop slots per env; 0 = from the density and the longest lap
    procedural_raceline_margin: float = 0.25   # [m] kept clear either side of the raceline, beyond the
                                      # car's half-width, wherever a pattern reaches -- so the line the
                                      # teacher opponents drive is never the thing that is blocked


def _interp_path(xy: torch.Tensor, grid: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(B, n, 2) samples taken at times `grid` (n,), linearly resampled at times `t` (K,) -> (B, K, 2).

    Times outside the grid are clamped to its ends; `car_future`'s callers extend past the end
    themselves where a straight-line continuation is the right thing.
    """
    n = grid.shape[0]
    pos = ((t - grid[0]) / (grid[1] - grid[0]).clamp_min(1e-9)).clamp(0.0, float(n - 1))
    i0 = pos.floor().long().clamp(max=max(n - 2, 0))
    w = (pos - i0.to(xy.dtype))[None, :, None]
    return xy[:, i0] * (1 - w) + xy[:, (i0 + 1).clamp(max=n - 1)] * w


class F1VecEnv:
    def __init__(self, track, cfg: Optional[Config] = None, env_cfg: Optional[EnvConfig] = None,
                 num_envs: int = 1024, device: Optional[str] = None):
        """track: a Track or a list of Tracks (envs are spread over them, re-drawn on reset)."""
        self.cfg = cfg or Config()
        self.cfg.sim.terminate_on_collision = True
        self.ecfg = env_cfg or EnvConfig()
        self.sim = Simulator(track, self.cfg, num_envs, device, race_size=self.ecfg.race_size)
        self.B, self.device = num_envs, self.sim.device
        # what `scan[:, ::subsample]` actually yields, which is a ceiling, not a floor. These agreed
        # while the LiDAR had 1080 beams; the measured count is 1081 (270 deg / 0.25 deg, endpoints
        # included), and 1081 // 4 = 270 against a slice of length 271 -- so every scan_subsample > 1
        # died on a shape mismatch when the scan history was written.
        self.n_beams = len(range(0, self.cfg.lidar.n_beams, self.ecfg.scan_subsample))
        self.range_max = self.cfg.lidar.range_max
        self.s_max = self.cfg.vehicle.s_max
        e = self.ecfg
        self.M = e.race_size
        ar = torch.arange(self.B, device=self.device)
        self.race, self.slot = ar // self.M, ar % self.M
        if e.opponent not in OPPONENT_MODES:
            raise ValueError(f"opponent {e.opponent!r} is not one of {list(OPPONENT_MODES)}")
        if e.spawn_order not in SPAWN_ORDERS:
            raise ValueError(f"spawn_order {e.spawn_order!r} is not one of {list(SPAWN_ORDERS)}")
        #: The canonical spelling of `EnvConfig.opp_token` -- "" when it is off, whichever way the
        #: caller spelled that. Read everywhere below instead of the config field.
        self.opp_token_mode = opp_token_mode(e.opp_token)
        if e.opp_future_model not in OPP_FUTURE_MODELS:
            raise ValueError(f"opp_future_model {e.opp_future_model!r} is not one of "
                             f"{list(OPP_FUTURE_MODELS)}")
        if self.opp_token_mode and self.M < 2:
            raise ValueError(f"opp_token {e.opp_token!r} with race_size {self.M}: the block "
                             f"describes the other cars of a race, and with one car per race it "
                             f"would be a constant zero input that still widens every checkpoint.")
        #: Width of the privileged opponent block, 0 when it is off.
        self.opp_token_dim = opp_token_dim(e.opp_token)
        #: The per-car table, parsed and checked, or None when this env is not in slots mode.
        self.slots = None
        if e.opponent_slots is not None:
            from .opponent_slots import parse_slots, validate_slots
            if e.opponent != "slots":
                raise ValueError(
                    f"opponent_slots with opponent={e.opponent!r}: the table already names who "
                    f"drives each car, so two answers to that question would be in the config at "
                    f"once. Set opponent='slots'.")
            self.slots = parse_slots(e.opponent_slots)
            # The files are not opened here -- `learn.opponent_pool` does that and says what is
            # wrong with a checkpoint far better than an existence test can.
            validate_slots(self.slots, self.M, require_files=False)
        elif e.opponent == "slots":
            raise ValueError("opponent 'slots' with no opponent_slots: the table *is* the "
                             "configuration, so an absent one is not 'the default opponent'. Pass "
                             "opponent_slots, or use another mode.")
        self.pool_names = pool_entries(e.opp_pool)
        if self.slots is not None:
            if e.opp_pool:
                raise ValueError("opponent_slots with opp_pool: a slot names its own checkpoint, so "
                                 "a pool drawn per race would be a second, contradicting answer.")
            # A checkpoint named by more than one slot is loaded once and drives both cars: the
            # entry is stateless apart from a runtime whose rows are the whole batch anyway.
            paths, seen = [], set()
            for sl in self.slots:
                if sl.checkpoint and sl.checkpoint not in seen:
                    seen.add(sl.checkpoint); paths.append(sl.checkpoint)
            self.pool_names = tuple(paths)
        if e.opponent == "pool":
            if self.M < 2:
                raise ValueError("opponent 'pool' needs race_size > 1: with one car per race there "
                                 "is no other car for the population to drive")
            if not self.pool_names:
                raise ValueError("opponent 'pool' with an empty opp_pool: the population is what "
                                 "drives the other car, so an empty one is not 'the default "
                                 "opponent', it is no opponent at all. Name entries (a checkpoint "
                                 f"path, {POOL_SELF!r} or {POOL_TEACHER!r}) or use another mode.")
        # Which cars the *policy* drives. Never the opponents in "teacher" mode; in "mixed" and
        # "pool" mode it changes at every race reset, so the PPO buffers (whose width is fixed by
        # `learner`) hold every car and the loss weights each sample by `on_policy` instead. A pool
        # without a `self` entry can never put a policy-driven car in an opponent slot, so there the
        # buffers stay narrow -- half of them would otherwise be masked out of every update.
        self.pool_can_self = e.opponent == "pool" and POOL_SELF in self.pool_names
        self.learner = torch.ones(self.B, dtype=torch.bool, device=self.device)
        if self.M > 1 and (e.opponent == "teacher" or (e.opponent == "pool" and not self.pool_can_self)):
            self.learner[self.slot > 0] = False
        if self.slots is not None:
            # Fixed for the life of the env, unlike "mixed" / "pool": which car a slot drives is the
            # table's answer, not a draw. So the PPO buffers are narrow exactly when no slot is the
            # learner's own weights, and no reset ever moves a row between the two.
            self_slot = torch.tensor([True] + [sl.policy_driven for sl in self.slots],
                                     dtype=torch.bool, device=self.device)
            self.learner = self_slot[self.slot].clone()
        self.on_policy = self.learner.clone()
        self.learner_ids = torch.nonzero(self.learner).flatten()
        self.teacher = None                                    # set_teacher() for opponent == "teacher"
        #: Teacher kinds a slot table named that are not the raceline teacher, their per-car masks
        #: and (after `set_teacher`) the objects themselves. Empty everywhere else, which is what
        #: keeps `_opponent_actions` one teacher call on every path that existed before slots.
        self.alt_teacher_kinds: tuple = ()
        self.alt_teacher_mask: Dict[str, torch.Tensor] = {}
        self.alt_teachers: list = []
        #: Whether any car of this configuration can be teacher-driven, and so whether a teacher is
        #: required at all. A pool of checkpoints alone needs none.
        self.teacher_any = (e.opponent in ("teacher", "mixed")
                            or (e.opponent == "pool" and POOL_TEACHER in self.pool_names)
                            or (self.slots is not None
                                and any(sl.teacher_driven for sl in self.slots)))
        # Opponent behaviour. Allocated whenever there are teacher-driven cars so a viewer or
        # logger can read `info["opp_event"]` unconditionally; inert (and drawing nothing from the
        # generator) until `opp_events` names an event with a positive rate or probability.
        self.events = (OpponentEvents(self.B, self.device, e, self.sim.control_dt, self.sim.gen,
                                      slots=self.slots,
                                      slot_of=(None if self.slots is None else self.slot))
                       if self.M > 1 and self.teacher_any else None)
        #: Per-car driver code (`OPP_DRIVER_*`), only meaningful in pool mode. Slot 0 is always the
        #: policy; an opponent's is redrawn at every full race reset.
        self.opp_driver = torch.zeros(self.B, dtype=torch.long, device=self.device)
        #: Entry index -> driver code. The checkpoint entries take 2, 3, ... in the order named, so
        #: `learn.opponent_pool` loads them in that order and indexes by `driver - OPP_DRIVER_POOL`.
        lut, ck = [], 0
        for name in self.pool_names:
            if name == POOL_SELF:
                lut.append(OPP_DRIVER_POLICY)
            elif name == POOL_TEACHER:
                lut.append(OPP_DRIVER_TEACHER)
            else:
                lut.append(OPP_DRIVER_POOL + ck); ck += 1
        self.pool_driver_lut = torch.tensor(lut or [OPP_DRIVER_POLICY], dtype=torch.long, device=self.device)
        #: The checkpoint paths of the pool, in driver-code order.
        self.pool_paths = tuple(n for n in self.pool_names if n not in (POOL_SELF, POOL_TEACHER))
        self.pool = None                                       # set_opponent_pool()
        #: Car-level masks the behaviour gate and the pool action both read. `teacher_race` below is
        #: a *race* flag; these are per car, and slot 0 is in neither.
        self.teacher_driven = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.pool_driven = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if self.M > 1 and e.opponent == "teacher":
            self.teacher_driven[self.slot > 0] = True
        if self.slots is not None:
            self._build_slot_tables()
        #: Which grid each race drew (`SPAWN_ORDER_ID`), so a single car respawning mid-race does
        #: not re-draw its race's order.
        self.spawn_order_race = torch.zeros(self.B, dtype=torch.long, device=self.device)
        #: The observation the policy last saw, which is the one a pool opponent acts on: the pool
        #: is asked for its action at the top of `step()`, before any new observation exists.
        self._last_obs = None
        #: The pose the plan tracker last solved from, kept so that `car_future` can put its
        #: predicted trajectory (`PlanTracker.last_pred`, a body-frame rollout) back into the world.
        #: (B, 3) and written once per step; `None` until the first plan-mode step.
        self._plan_pose = None
        # Obstacle layouts redrawn per env at every reset. Built here so the prop tensors exist
        # before anything compiles against them; the raceline corridor arrives later, with the
        # teacher (`set_teacher`), because that is when the line the opponents drive is known.
        self.procedural = None
        if e.procedural_obstacles > 0:
            from .procedural_obstacles import ProceduralObstacles
            self.procedural = ProceduralObstacles(
                self.sim.track, self.B, self.sim.gen, density=e.procedural_density,
                fraction=e.procedural_obstacles, max_props=e.procedural_max_props,
                raceline_margin=e.procedural_raceline_margin,
                car_half_width=0.5 * self.cfg.vehicle.width)
            self.sim.track.attach_env_props(self.procedural)
            # how far a prop centre can be from the car's and still touch it: the contact tests cull
            # to the slots inside this, and count anything they dropped that was not
            self.sim.prop_reach = float(self.sim.corners.norm(dim=1).max()) + self.procedural.max_radius
        self.opp_scale = torch.ones(self.B, device=self.device)
        #: (B,) the speed-scale draw each car is running, whichever way that car uses it (a teacher
        #: scales its profile, a policy scales its cap). One place to read "how fast was this car
        #: told to be", for the census, the facts strip and a test.
        self.slot_scale = torch.ones(self.B, device=self.device)
        # signed arc to each opponent last step (+ ahead of me, - behind), and whether it is usable.
        # A flag rather than "0 means unset": 0 is a perfectly ordinary gap -- it is the pass itself.
        self.gap_prev = torch.zeros(self.B, max(1, self.M - 1), device=self.device)
        self.gap_valid = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        T_, S_ = self.sim.track.T, int(e.reward_time_sectors)
        self.ideal_lap = torch.zeros(T_, device=self.device)                 # per track, seconds
        self.sector_lim = torch.zeros(T_, S_, device=self.device)            # raceline time per sector
        self.sector_ref = torch.zeros(T_, S_, device=self.device)            # time to beat, 0 = unseen
        self.sector_idx = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.sector_t0 = torch.zeros(self.B, device=self.device)             # episode time at sector entry
        self.sector_valid = torch.zeros(self.B, dtype=torch.bool, device=self.device)  # entered from a boundary
        self.s_prev = torch.zeros(self.B, device=self.device)
        self.hist_len = (e.scan_stack - 1) * e.scan_stride + 1
        self.scan_hist = torch.ones(self.B, self.hist_len, self.n_beams, device=self.device)
        self.act_dim = PLAN_DIM if e.action_mode == "plan" else 2
        self.prev_action = torch.zeros(self.B, self.act_dim, device=self.device)
        self.act_hist = torch.zeros(self.B, self.ecfg.action_history, self.act_dim, device=self.device)
        self.tracker = None; self.prev_steer_norm = torch.zeros(self.B, device=self.device); self.last_cmd = torch.zeros(self.B, 2, device=self.device)
        # External command override (ROS 2 `/drive`, an outside controller): per-car (steer [rad],
        # speed [m/s]) that replaces whatever the policy / tracker produced for the cars whose mask
        # is set. Applied after the action mapping and before the physics, so both action modes,
        # the opponents, the observation history and `last_cmd` see the command that was driven.
        self.ext_cmd = torch.zeros(self.B, 2, device=self.device)
        self.ext_mask = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._ext_ids: set = set()               # host-side copy of the mask: no device sync per command
        #: Optional `(B, 2) -> (B, 2)` shaper for the command on its way to the simulator, installed
        #: by a controller that sits between the policy and the VESC. `learn/traction_arm.py` is the
        #: one that exists. None is the untouched path; there is deliberately room for exactly one,
        #: since two shapers on one command path is not a thing to resolve by installation order.
        self.cmd_shaper = None
        self.row_dim = 1 + 6 + 2 + self.act_dim
        self.hist = torch.zeros(self.B, (e.hist_len - 1) * e.hist_stride + 1, self.row_dim, device=self.device) if e.hist_len > 0 else None
        self._last_feat = torch.zeros(self.B, 9, device=self.device)
        self._math = self._step_math
        if self.device.type == "cuda" and self.cfg.sim.compile_mode == "reduce-overhead":
            compiled = torch.compile(self._step_math, dynamic=False, mode="reduce-overhead")
            def _math(*args, _c=compiled):
                try:
                    return _c(*args)
                except Exception:                                  # inductor/Triton failure at first use: stay eager
                    self._math = self._step_math
                    return self._step_math(*args)
            self._math = _math
        self.tracker_delay = None
        if e.action_mode == "plan":
            self.tracker = PlanTracker(self.B, self.device, self.cfg.vehicle.lf + self.cfg.vehicle.lr, self.cfg.vehicle.s_max,
                                       e.v_max_policy, compile_solver=e.compile_tracker)
            self._calibrate_tracker(torch.arange(self.B, device=self.device))
        self.speed_cap = torch.full((self.B,), float(self.ecfg.speed_cap), device=self.device)
        # Self-play with identical cars produces almost no passing: the front car is exactly as fast
        # as the one behind, so races settle into a constant-gap procession and the overtake and
        # contact terms read ~0 -- measured, overtake fell from 0.016/step against teachers to
        # 0.0006 the moment the opponent became the policy. So in self-play the *front* car of each
        # race drives under a speed cap scaled by opp_speed_range, the way teacher opponents were
        # slowed: the car behind gets a pass to make, the car in front gets a slower car to defend
        # with, and both are the same network learning both sides of the situation.
        self.cap_base = float(self.ecfg.speed_cap)
        self.cap_scale = torch.ones(self.B, device=self.device)
        self.teacher_race = torch.zeros(self.B, dtype=torch.bool, device=self.device)   # mixed: this race's others are teachers
        self.ep_step = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.lap_start_step = torch.zeros(self.B, dtype=torch.long, device=self.device)   # step of the last finish-line crossing
        self.prev_lap = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.ep_return = torch.zeros(self.B, device=self.device)
        self.ep_progress = torch.zeros(self.B, device=self.device)
        self.last_result: Optional[StepResult] = None
        self._empty_long = torch.zeros(0, dtype=torch.long, device=self.device); self._empty_float = torch.zeros(0, device=self.device)
        self._no_plan = torch.zeros(self.B, 1, 4, device=self.device)
        self.single_observation_space, self.single_action_space = self._spaces()

    def _spaces(self):
        try:
            import gymnasium as gym
            spaces = {
                "scan": gym.spaces.Box(0.0, 1.0, (self.ecfg.scan_stack, self.n_beams), np.float32),
                "speed": gym.spaces.Box(-1.0, 2.0, (1,), np.float32),
                "prev_action": gym.spaces.Box(-1.0, 1.0, (self.act_dim * self.ecfg.action_history,), np.float32),
                **({"hist": gym.spaces.Box(-5.0, 5.0, (self.ecfg.hist_len * self.row_dim,), np.float32)} if self.ecfg.hist_len > 0 else {}),
                "speed_cap": gym.spaces.Box(0.0, 1.0, (1,), np.float32)}
            if self.ecfg.obs_imu and self.cfg.imu.enabled:
                spaces["imu"] = gym.spaces.Box(-5.0, 5.0, (6,), np.float32)
                spaces["imu_att"] = gym.spaces.Box(-2.0, 2.0, (2,), np.float32)
            obs = gym.spaces.Dict(spaces)
            act = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
            return obs, act
        except ImportError:
            return None, None

    def _calibrate_tracker(self, ids: torch.Tensor):
        """The tracker is *calibrated* on the real car the way the odometry is (measured once): in sim it
        knows each env's command delay + half the servo lag (+-20 ms residual), the servo offset
        (+-0.01 rad) and gain (+-4 %) and the VESC speed gain (+-3 %); grip stays unknown by decision."""
        P = self.sim.P; n = ids.numel(); g = self.sim.gen
        u = lambda a: (torch.rand(n, device=self.device, generator=g) - 0.5) * 2 * a
        if self.tracker_delay is None:
            self.tracker_delay = torch.zeros(self.B, device=self.device)
            self.tracker_cal = torch.zeros(self.B, 3, device=self.device)          # steer bias, steer gain, speed gain
        self.tracker_delay[ids] = (P["cmd_delay"][ids] + 0.5 * P["servo_tau"][ids] + u(0.02)).clamp(0.0, 0.2)
        self.tracker_cal[ids, 0] = P["steer_bias"][ids] + u(0.01)
        self.tracker_cal[ids, 1] = P["steer_gain"][ids] * (1 + u(0.04))
        self.tracker_cal[ids, 2] = P["speed_gain"][ids] * (1 + u(0.03))

    # ------------------------------------------------------------------ slots mode
    def _build_slot_tables(self):
        """Per-car tables the slot mode reads instead of drawing a mode per race.

        Everything here is fixed for the life of the env: who drives slot i, which checkpoint, which
        grip label, which speed band, which spawn. That is the whole difference from "mixed" and
        "pool", where the same tensors are re-drawn at every race reset -- a slot table is a *grid*,
        not a population.
        """
        from .opponent_slots import SLOT_SPAWN_ID
        from .teacher import LABEL_GRIP_CODE
        dev = self.device
        drv, spawn, lo, hi, cap, cap_set, grip = [OPP_DRIVER_POLICY], [SLOT_SPAWN_LEARNER], [1.0], [1.0], [0.0], [False], [0]
        ck_index = {p: j for j, p in enumerate(self.pool_names)}
        for sl in self.slots:
            if sl.checkpoint:
                drv.append(OPP_DRIVER_POOL + ck_index[sl.checkpoint])
            elif sl.teacher_driven:
                drv.append(OPP_DRIVER_TEACHER)
            else:
                drv.append(OPP_DRIVER_POLICY)
            spawn.append(SLOT_SPAWN_ID.get(sl.spawn, len(SLOT_SPAWN_ID)))   # 3 == "random"
            lo.append(sl.speed_scale[0]); hi.append(sl.speed_scale[1])
            cap.append(0.0 if sl.speed_cap is None else float(sl.speed_cap))
            cap_set.append(sl.speed_cap is not None)
            grip.append(LABEL_GRIP_CODE[sl.label_grip])
        t = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=dev)
        #: (M,) per grid slot; the (B,) forms below are these indexed by `self.slot`.
        self.slot_driver_of = t(drv, torch.long)
        self.slot_spawn_of = t(spawn, torch.long)
        self.opp_driver = self.slot_driver_of[self.slot].clone()
        self.teacher_driven = (self.opp_driver == OPP_DRIVER_TEACHER) & (self.slot > 0)
        self.pool_driven = self.opp_driver >= OPP_DRIVER_POOL
        self.on_policy = self.learner.clone()
        self.slot_scale_lo, self.slot_scale_hi = t(lo)[self.slot], t(hi)[self.slot]
        self.slot_cap = t(cap)[self.slot]
        self.slot_cap_set = t(cap_set, torch.bool)[self.slot]
        self.slot_grip_code = t(grip, torch.long)[self.slot]
        #: (B,) the spawn each car's *race* drew, so a car respawning mid-race rejoins the grid its
        #: race started on. Only the `random` slots move; the rest are their own code for ever.
        self.slot_spawn_code = self.slot_spawn_of[self.slot].clone()
        self._slot_spawn_random = bool((self.slot_spawn_of == len(SLOT_SPAWN_ID)).any())
        #: Per-slot generators for the slots that asked for one, so re-writing slot 2's band leaves
        #: slot 1 replaying the numbers it replayed before.
        self.slot_gens = [None] + [
            (torch.Generator(device=self.device).manual_seed(int(sl.seed))
             if sl.seed is not None else None) for sl in self.slots]
        # Teacher kinds that are not the raceline teacher, and the rows each one drives. `set_teacher`
        # builds the objects; the masks are fixed here because a slot's kind never changes.
        alt, masks = [], {}
        for i, sl in enumerate(self.slots, start=1):
            kind = sl.driver
            if not (kind.teacher and kind.teacher_factory):
                continue
            if kind.name not in masks:
                alt.append(kind.name)
                masks[kind.name] = torch.zeros(self.B, dtype=torch.bool, device=dev)
            masks[kind.name] |= self.slot == i
        self.alt_teacher_kinds = tuple(alt)
        self.alt_teacher_mask = masks

    def _slot_spawn_codes(self, is_full_race: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """(G, M) spawn code per car of every race, re-drawing the `random` slots of a full reset.

        Drawn per race and remembered for the same reason `_spawn_order` remembers its draw: a car
        that crashes rejoins the grid its race was started on, not a new one.
        """
        from .opponent_slots import SLOT_SPAWN_ID
        codes = self.slot_spawn_code.view(-1, self.M)
        if self._slot_spawn_random:
            pick = torch.randint(len(SLOT_SPAWN_ID), (codes.shape[0], self.M),
                                 device=self.device, generator=gen)
            fresh = torch.where(self.slot_spawn_of[None, :] == len(SLOT_SPAWN_ID), pick,
                                self.slot_spawn_of[None, :].expand_as(pick))
            codes.copy_(torch.where(is_full_race[:, None], fresh, codes))
        return codes

    @staticmethod
    def _slot_ranks(codes: torch.Tensor) -> torch.Tensor:
        """(G, M) grid rank, 0 = leader, from the per-car spawn codes.

        Front to back: the cars that start *ahead* of the learner (highest slot index first, so an
        all-`ahead` table reproduces `spawn_order behind` exactly), then the learner together with
        whatever starts alongside it, then the cars that start behind, in slot order. Slot 0 carries
        the sentinel and lands in the middle group by construction, which is where the learner is.
        """
        ahead = codes == SLOT_SPAWN_ID_AHEAD
        behind = codes == SLOT_SPAWN_ID_BEHIND
        n_ahead = ahead.sum(1)
        rank_ahead = ahead.flip(1).cumsum(1).flip(1) - 1          # 0 for the highest ahead slot
        rank_behind = n_ahead[:, None] + behind.cumsum(1)
        mid = n_ahead[:, None].expand_as(codes)
        return torch.where(ahead, rank_ahead, torch.where(behind, rank_behind, mid))

    def _slot_grid(self, ids, full, gen, base, L_race):
        """Where every resetting car of a slot-mode race spawns: (rank, arc, lat, clearance, yaw).

        The same three questions the flag-driven grid answers -- what order, what arc, and whether an
        abreast row fits between the walls -- asked per car instead of per race. Everything is
        computed for the whole batch and then read at `ids`: the rank of a car depends on the codes
        of its race mates, so the natural shape here is (G, M) and not (n,).
        """
        e = self.ecfg
        G, M = full.shape[0], self.M
        codes = self._slot_spawn_codes(full, gen)                        # (G, M)
        rank = self._slot_ranks(codes)
        gap_rank = e.spawn_gap[0] + (e.spawn_gap[1] - e.spawn_gap[0]) * torch.rand(
            G, M, device=self.device, generator=gen)
        gap_rank[:, 0] = 0.0                                             # the leader has nobody ahead
        stagger = e.spawn_alongside_gap[0] + (e.spawn_alongside_gap[1] - e.spawn_alongside_gap[0]) \
            * torch.rand(G, M, device=self.device, generator=gen)
        s_rank = base[:, None] * L_race[:, None] - gap_rank.cumsum(1)    # (G, M) arc of each rank
        # The abreast group: the learner plus whatever starts level with it. One car is not a row,
        # so a table with no `alongside` slot never touches this path at all.
        slot_index = torch.arange(M, device=self.device)
        group = (codes == SLOT_SPAWN_ID_ALONGSIDE)
        group[:, 0] = True
        g_size = group.sum(1)
        want = torch.zeros(G, M, device=self.device)
        abreast = torch.zeros(G, M, dtype=torch.bool, device=self.device)
        psi = 3.0 * min(e.spawn_yaw_std, e.spawn_alongside_yaw)
        extent = (self.cfg.vehicle.width * math.cos(psi) + self.cfg.vehicle.length * math.sin(psi))
        if bool((g_size > 1).any()):
            j = group.cumsum(1) - 1                                      # index within the group
            want = torch.where(group, (j - (g_size[:, None] - 1) / 2.0) * (extent + e.spawn_alongside_sep),
                               torch.zeros_like(want))
            s_group = s_rank.gather(1, rank.clamp_min(0)) - stagger      # each group member's own arc
            tid_row = self.sim.tid.view(-1, M)
            xy0, yaw0 = self.sim.track.pose_at_s(s_group.reshape(-1), tid_row.reshape(-1))
            room = (self.sim.track.sample_edt(xy0, tid_row.reshape(-1)).view(G, M)
                    - 0.5 * extent - e.opp_event_margin)
            headroom = torch.where(group, room - want.abs() - 0.03,
                                   torch.full_like(room, math.inf))
            if getattr(self.sim.track, "has_props", False):
                nrm = torch.stack([-torch.sin(yaw0), torch.cos(yaw0)], 1)
                pose_ab = torch.cat([xy0 + nrm * want.reshape(-1)[:, None], yaw0[:, None]], 1)
                blocked = self.sim._spawn_in_prop(pose_ab, tid_row.reshape(-1),
                                                  torch.arange(self.B, device=self.device)).view(G, M)
                headroom = torch.where(group & blocked, torch.full_like(headroom, -math.inf), headroom)
            # Per race: a row with one car abreast and another staggered is exactly the grid that
            # collides, because the stagger puts that car back on the arc the abreast one holds.
            fits = (headroom.amin(1) >= 0.0) & (g_size > 1) & full
            # Where it does not fit, the alongside cars fall back to starting ahead -- the stagger
            # that path takes has no lateral offset to lose, so it spawns safely.
            codes = torch.where(fits[:, None], codes,
                                torch.where(group & (slot_index[None, :] > 0),
                                            torch.full_like(codes, SLOT_SPAWN_ID_AHEAD), codes))
            rank = self._slot_ranks(codes)
            abreast = fits[:, None] & group
            want = torch.where(abreast, want, torch.zeros_like(want))
        s_cars = s_rank.gather(1, rank.clamp_min(0))
        s_cars = torch.where(abreast, s_cars - stagger, s_cars)
        lat_all = torch.where(abreast, want,
                              torch.randn(G, M, device=self.device, generator=gen) * e.spawn_lateral_std)
        min_all = torch.where(abreast, torch.full_like(lat_all, 0.5 * extent + e.opp_event_margin),
                              torch.full_like(lat_all, e.spawn_min_clearance))
        yaw_all = torch.where(abreast, torch.full_like(lat_all, min(e.spawn_yaw_std, e.spawn_alongside_yaw)),
                              torch.full_like(lat_all, e.spawn_yaw_std))
        flat = lambda t_: t_.reshape(-1)[ids]
        return (flat(rank).float(), flat(s_cars), flat(lat_all), flat(min_all), flat(yaw_all))

    def _slot_speeds(self, ids, slot, is_full, full, gen):
        """Draw each resetting car's speed scale from *its own* slot's band, and set its cap.

        Two different things are called a speed scale here, and which one a car gets is decided by
        who drives it. A teacher has a speed *profile* to scale, and the scale goes on the teacher's
        own per-car tensor, so the braking points move with it. A checkpoint or a `self` car drives
        at whatever pace its network drives; the only handle on it is the cap, which is exactly what
        `--opp-speed` already did to a pool car. Exactly one of the two carries the scale for a given
        car, so nothing is ever scaled twice; `slot_scale` records the draw either way.
        """
        e = self.ecfg
        G, M = full.shape[0], self.M
        u = torch.rand(G, M, device=self.device, generator=gen)
        for j, g_ in enumerate(self.slot_gens):
            if g_ is not None:
                u[:, j] = torch.rand(G, device=self.device, generator=g_)
        lo = self.slot_scale_lo.view(-1, M)[0][None, :]
        hi = self.slot_scale_hi.view(-1, M)[0][None, :]
        scale_all = lo + (hi - lo) * u
        scale = scale_all.reshape(-1)[ids]
        self.slot_scale[ids] = torch.where(is_full, scale, self.slot_scale[ids])
        scale = self.slot_scale[ids]
        teach = self.teacher_driven[ids]
        # The teacher's own per-car scale; `opp_scale` stays 1 there so `_opponent_actions` does not
        # apply it a second time on the way out.
        if self.teacher is not None and torch.is_tensor(getattr(self.teacher, "speed_scale", None)):
            self.teacher.speed_scale[ids] = torch.where(teach, scale, torch.ones_like(scale))
        self.opp_scale[ids] = torch.where(teach, torch.ones_like(scale), scale)
        cap = torch.full_like(scale, self.cap_base)
        if e.selfplay_front_cap:
            # The pace a policy actually drives, not the curriculum cap: at cap 9.0 the policy runs
            # corner-limited at ~4.5 m/s, so a scale against the cap would bind nothing. The viewer
            # turns this off, and there a `self` car races the learner as an equal.
            paced = (scale * e.selfplay_pace_ref).clamp(max=self.cap_base)
            cap = torch.where(teach | (slot == 0), cap, paced)
        # An explicit per-slot cap wins over both, and is the one handle that means the same thing
        # for every kind of driver. Stored as a ratio so `set_speed_cap` keeps its meaning.
        cap = torch.where(self.slot_cap_set[ids], self.slot_cap[ids].clamp(max=e.v_max_policy), cap)
        self.cap_scale[ids] = torch.where(is_full, cap / max(self.cap_base, 1e-6), self.cap_scale[ids])
        self.speed_cap[ids] = self.cap_base * self.cap_scale[ids]

    def set_teacher(self, teacher):
        """Raceline teacher that drives the opponent cars (opponent == "teacher")."""
        self.teacher = teacher
        if self.slots is not None:
            # One teacher object drives every teacher-driven car, so the two things a slot can say
            # about it -- how fast its profile is and which friction that profile was planned for --
            # become per-car tensors on it. `speed_scale` starts at 1 and is written at every reset;
            # `label_grip_codes` never changes, because the label is the slot's, not the race's.
            teacher.speed_scale = torch.ones(self.B, device=self.device)
            teacher.label_grip_codes = self.slot_grip_code
            # ... and one object per non-raceline teacher kind the table named, each built *from*
            # this teacher, so the per-car tensors above are the ones it plans with.
            from .opponent_slots import build_teacher, kind_of
            self.alt_teachers = [build_teacher(kind_of(name), teacher, self)
                                 for name in self.alt_teacher_kinds]
        if self.procedural is not None:
            # The teacher is pure pursuit on a raceline built from the occupancy grid, and the
            # procedural props are not in the grid: it cannot see them and will not steer round
            # them. So the layouts are laid *outside* the band the raceline occupies, which is the
            # contract's first option ("place patterns only where the raceline is not"). The other
            # option -- teaching `clamp_offset` about props -- bounds an offset away from the line
            # and does nothing about a crate standing on it.
            self.procedural.set_raceline(teacher)
        if self.events is not None and self.events.enabled:
            # How far off the line each raceline point can be driven without putting a car in the
            # wall, measured once from the track's distance field. The teacher clamps against it at
            # the car's own raceline index, so a shift through a narrow section shrinks instead of
            # crashing. Nothing is computed, and nothing changes, when no event moves a car sideways.
            teacher.offset_limit = raceline_offset_limit(teacher, self.sim.track,
                                                         0.5 * self.cfg.vehicle.width,
                                                         self.ecfg.opp_event_margin)
        if self.events is not None and self.events.needs_corners:
            # Where the corners are on each raceline, for the `line` behaviour. Built once here for
            # the same reason the offset budget is: it is a function of the track set, not of a step.
            self.events.corners = raceline_corners(teacher, self.ecfg.opp_corner_kappa,
                                                   self.ecfg.opp_corner_min_arc,
                                                   self.ecfg.opp_corner_smooth)

    def set_opponent_pool(self, pool):
        """Population that drives the opponent cars (opponent == "pool").

        `pool` is a `learn.opponent_pool.OpponentPool` -- injected rather than built here for the
        same reason the teacher is: loading a checkpoint means `learn.model`, and the env is below
        `learn` in the import order. The env owns which car each entry drives; the pool owns how an
        entry is asked for an action.
        """
        if self.ecfg.opponent not in ("pool", "slots"):
            raise RuntimeError(f"set_opponent_pool on an env with opponent={self.ecfg.opponent!r}: "
                               f"the pool only drives cars in 'pool' mode, or as the checkpoint "
                               f"entries of a slot table")
        if len(pool) != len(self.pool_paths):
            raise ValueError(f"the pool holds {len(pool)} checkpoint(s) but opp_pool names "
                             f"{len(self.pool_paths)}: {list(self.pool_paths)}")
        self.pool = pool

    def learner_view(self, state: Optional[torch.Tensor] = None) -> LearnerView:
        """Where the policy-driven cars are, from every car's own frame, right now.

        The reactive behaviours react to this and `learn.opponent_census` counts situations out of
        it, deliberately from one implementation: "alongside" meaning one thing to the opponent
        that yields and another to the census that reports how long the learner spent there is
        exactly the failure the census exists to rule out.
        """
        st = self.sim.state if state is None else state
        o = self.sim.other_idx
        if o is None:
            raise RuntimeError("learner_view needs a race (race_size > 1)")
        d = st[o][:, :, :2] - st[:, None, :2]                              # (B,C,2) me -> them
        c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        lon = d[..., 0] * c[:, None] + d[..., 1] * sn[:, None]
        lat = -d[..., 0] * sn[:, None] + d[..., 1] * c[:, None]
        dist = d.norm(dim=2).clamp_min(1e-6)
        vme = torch.stack([st[:, 3] * c - st[:, 4] * sn, st[:, 3] * sn + st[:, 4] * c], 1)
        co, so = torch.cos(st[o][:, :, 2]), torch.sin(st[o][:, :, 2])
        vth = torch.stack([st[o][:, :, 3] * co - st[o][:, :, 4] * so,
                           st[o][:, :, 3] * so + st[o][:, :, 4] * co], 2)
        closing = -((vth - vme[:, None, :]) * d).sum(2) / dist
        rl_idx = None
        if self.events is not None and self.events.needs_corners and self.teacher is not None:
            # One extra raceline projection per step, paid only by the `line` behaviour: the
            # teacher's own projection is of the latency-compensated pose and is not returned.
            rl_idx, _ = self.teacher.project(st[:, :2], self.sim.tid)
        return LearnerView(gap=self.signed_gaps(self.sim.s, self.sim.tid), lon=lon, lat=lat,
                           closing=closing, learner=self.on_policy[o], rl_idx=rl_idx,
                           tid=self.sim.tid)

    # ------------------------------------------------------------------ helpers
    def _norm_scan(self, scan: torch.Tensor) -> torch.Tensor:
        # `learn.obs.norm_scan`, not a second copy of it: the ROS policy node runs the same
        # function on the same numbers, and this is the boundary a divergence would hide behind.
        return norm_scan(scan[:, :: self.ecfg.scan_subsample], self.range_max)

    def _obs(self, r: StepResult) -> Dict[str, torch.Tensor]:
        speed = norm_speed(r.odom[:, 3], self.ecfg.v_max_policy)[:, None]
        obs = {"scan": self.scan_hist[:, ::self.ecfg.scan_stride].clone(), "speed": speed, "prev_action": self.act_hist.reshape(self.B, -1).clone(),
               "speed_cap": norm_speed(self.speed_cap, self.ecfg.v_max_policy)[:, None]}
        if self.ecfg.obs_imu and r.imu is not None and r.imu.shape[1] > 0:
            m = r.imu.mean(1)
            obs["imu"] = norm_imu(m, self.ecfg.imu_gyro_scale, self.ecfg.imu_accel_scale)
            # VESC roll/pitch estimate (yaw drifts: excluded)
            obs["imu_att"] = norm_att(r.imu_att[:, :2], ATT_SCALE)
        if self.hist is not None:
            self._last_feat = torch.cat([speed, obs.get("imu", torch.zeros(self.B, 6, device=self.device)), obs.get("imu_att", torch.zeros(self.B, 2, device=self.device))], 1)
            obs["hist"] = self.hist[:, ::self.ecfg.hist_stride].reshape(self.B, -1)
        if self.opp_token_mode:
            # LAST, after every key the observation already had, so that the columns a checkpoint
            # was trained without keep their index and a warm start is a copy plus zeros.
            obs["opp_token"] = self.opp_token(r.state)
        return obs

    def message_inputs(self, i: int, r: Optional[StepResult] = None) -> MessageInputs:
        """One car's control step, stated as the topics the ROS graph would have carried.

        The simulator's bridges (`f1sim_ros/bridge_node.py`, `viewer/ros_link.py`) publish exactly
        these numbers: `r.scan[i]` as `LaserScan.ranges` with `cfg.lidar.range_max`, `r.odom[i, 3]`
        as `Odometry.twist.twist.linear.x`, `r.imu[i]` as the `Imu` samples of this step in SI, and
        `r.imu_att[i, :2]` as the roll and pitch of `Imu.orientation`. Nothing privileged is in it:
        no pose, no ground-truth speed, no friction.

        It exists so the claim "the batched env and the node build the same observation" can be
        run: `ObsBuilder.build_message` takes this object, and `tests/test_obs_identity.py`
        compares the result against `_obs`'s row `i`. A field that this method has to invent is a
        field the node could not have -- which is the sim-to-real gap the observation contract is
        supposed to make impossible.
        """
        r = self.last_result if r is None else r
        if r is None:
            raise RuntimeError("no step to describe: reset the env first")
        if r.imu is None or r.imu.shape[1] == 0:
            raise RuntimeError("this env publishes no IMU (EnvConfig.obs_imu / cfg.imu.enabled), so "
                               "there is no message-equivalent step to state")
        return MessageInputs(ranges=r.scan[i].detach().cpu().numpy().astype(np.float32),
                             range_max=float(self.range_max),
                             speed=float(r.odom[i, 3]),
                             imu=r.imu[i].detach().cpu().numpy().astype(np.float32),
                             att=(float(r.imu_att[i, 0]), float(r.imu_att[i, 1])),
                             speed_cap=float(self.speed_cap[i]))

    def _priv(self, r: StepResult) -> torch.Tensor:
        st = r.state
        xy_c, yaw_c = self.sim.track.pose_at_s(r.s, self.sim.tid)
        herr = torch.remainder(st[:, 2] - yaw_c + np.pi, 2 * np.pi) - np.pi
        pv = torch.stack([st[:, 3], st[:, 4], st[:, 5], r.lateral, herr, r.s / self.sim.track.length[self.sim.tid],
                          r.progress / self.sim.control_dt, r.wall_dist], 1)
        if self.M > 1:
            # Nearest opponent, four columns, every one of them metres (or m/s) divided by
            # PRIV_OPP_DIST_SCALE: [ahead offset, side offset, longitudinal speed difference,
            # distance]. `dv` is other.vx - ego.vx, a difference of two body-frame longitudinal
            # speeds -- it is NOT the line-of-sight closing speed; car_proximity_penalty computes
            # that one separately. Anything comparing these columns against a distance has to
            # multiply by PRIV_OPP_DIST_SCALE first.
            o = self.sim.other_idx                             # (B,C)
            d = st[o][:, :, :2] - st[:, None, :2]
            dist = d.norm(dim=2); j = dist.argmin(1); ar = torch.arange(self.B, device=self.device)
            dn = d[ar, j]; c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
            dx, dy = dn[:, 0] * c + dn[:, 1] * sn, -dn[:, 0] * sn + dn[:, 1] * c
            dv = st[o[ar, j], 3] - st[:, 3]
            s_ = PRIV_OPP_DIST_SCALE
            pv = torch.cat([pv, torch.stack([dx / s_, dy / s_, dv / s_, dist[ar, j] / s_], 1)], 1)
        return pv

    def _reset_envs(self, ids: torch.Tensor):
        if ids.numel() == 0:
            return
        e = self.ecfg
        gen = self.sim.gen
        n = ids.numel()
        lat, min_clear, yaw_std = None, e.spawn_min_clearance, e.spawn_yaw_std
        if self.M == 1:
            if e.resample_track_on_reset and self.sim.track.T > 1:
                self.sim.tid[ids] = torch.randint(self.sim.track.T, (n,), device=self.device, generator=gen)
            self._redraw_layouts(ids, n)
            s = None
        else:
            # a race resets as a whole (new track, cars staggered along the lane) when all its cars are in
            # ids; a single car (crashed) respawns behind its race mates on the race's current track
            in_ids = torch.zeros(self.B, dtype=torch.bool, device=self.device); in_ids[ids] = True
            full = in_ids.view(-1, self.M).all(1)                             # (G,)
            race = self.race[ids]; slot = self.slot[ids]; is_full = full[race]
            if e.resample_track_on_reset and self.sim.track.T > 1:
                new_tid = torch.randint(self.sim.track.T, (full.shape[0],), device=self.device, generator=gen)
                self.sim.tid[ids] = torch.where(is_full, new_tid[race], self.sim.tid[ids])
            self._redraw_layouts(ids, n, full=full)
            L = self.sim.track.length[self.sim.tid[ids]]
            gap = e.spawn_gap[0] + (e.spawn_gap[1] - e.spawn_gap[0]) * torch.rand(n, device=self.device, generator=gen)
            base = torch.rand(full.shape[0], device=self.device, generator=gen)   # leader position, fraction of a lap
            # Slot 0 is the learner when the opponents are teachers, and the learner belongs at the
            # *back* of the grid: an overtake is a car behind catching and passing a car ahead, and a
            # learner spawned as the leader meets that situation only if a faster teacher passes it
            # first -- the opposite lesson (being overtaken pays negative), and at 0.6x the teacher
            # never even arrives. Self-play (every car a learner) keeps the leader-first stagger.
            if e.opponent == "mixed":
                teach_race = torch.rand(full.shape[0], device=self.device, generator=gen) < e.mixed_teacher_frac
                self.teacher_race[ids] = torch.where(is_full, teach_race[race], self.teacher_race[ids])
                self.on_policy[ids] = ~(self.teacher_race[ids] & (slot > 0))
            elif e.opponent == "pool":
                # One entry per race, drawn from the simulator's own generator so a seed reproduces
                # which opponent the learner met in which race. Only a *full* race reset re-draws:
                # a single car respawning after a crash rejoins the race it was already in, against
                # the same opponent, which is what "per race" has to mean for a partial reset.
                pick = torch.randint(len(self.pool_names), (full.shape[0],), device=self.device, generator=gen)
                drv = self.pool_driver_lut[pick]                              # (G,)
                self.opp_driver[ids] = torch.where(is_full & (slot > 0), drv[race], self.opp_driver[ids])
                self.opp_driver[ids] = torch.where(slot == 0, torch.zeros_like(slot), self.opp_driver[ids])
                self.teacher_race[ids] = torch.where(is_full, drv[race] == OPP_DRIVER_TEACHER,
                                                     self.teacher_race[ids])
                self.on_policy[ids] = self.opp_driver[ids] == OPP_DRIVER_POLICY
            if self.slots is None:
                teacher_race_row = (self.teacher_race[ids] if e.opponent in ("mixed", "pool")
                                    else torch.full_like(slot, e.opponent == "teacher", dtype=torch.bool))
                self.teacher_driven[ids] = teacher_race_row & (slot > 0)
                self.pool_driven[ids] = self.opp_driver[ids] >= OPP_DRIVER_POOL
            if self.slots is not None:
                # Per-car grid. The rank of a car is decided by where *its own* slot says it starts,
                # not by one order for the whole race, so this replaces the block below whole rather
                # than patching it: there is no "the order" left to patch, and no "the opponent" to
                # ask whether the race is scripted.
                L_race = self.sim.track.length[self.sim.tid].view(-1, self.M)[:, 0]
                rank, s_full, lat, min_clear, yaw_std = self._slot_grid(ids, full, gen, base, L_race)
            else:
                # Whether this race's *other* cars are something other than the policy under
                # training. It decides both halves of the grid: only such a race has a "the learner"
                # to place ahead of or behind the rest, and only a self-play race gets the front-car
                # cap.
                if e.opponent == "pool":
                    scripted_row = self.opp_driver.view(-1, self.M)[:, 1:].ne(OPP_DRIVER_POLICY).any(1)[race]
                elif e.opponent == "mixed":
                    scripted_row = teacher_race_row
                else:
                    scripted_row = torch.full_like(slot, e.opponent == "teacher", dtype=torch.bool)
                order = self._spawn_order(ids, race, is_full, full.shape[0], gen)
                # Grid position, 0 = leader. The learner is slot 0, and in a race it does not drive both
                # sides of, where it starts is the whole difference between "a pass to make" and "a
                # place to defend": `behind` (every race trained so far) puts it last, `ahead` first,
                # `alongside` level with the field. A self-play race keeps the leader-first stagger it
                # has always had -- relabelling which learner leads changes nothing about the race.
                rank = torch.where(scripted_row,
                                   torch.where(order == SPAWN_ORDER_ID["ahead"], slot.float(),
                                               (self.M - 1 - slot).float()),
                                   slot.float())
                rank = torch.where(order == SPAWN_ORDER_ID["alongside"], torch.zeros_like(rank), rank)
                # Cumulative, not `rank * gap`: the gap is drawn per car, so multiplying it by the rank
                # scrambles a grid of three or more (rank 1 drawing 6 m and rank 2 drawing 2.5 m puts
                # the third car a metre *ahead* of the second, and close draws put them in contact --
                # measured, 14 of 528 three-car spawns). Each car sits its own gap behind the car in
                # front of it, which is what the flag says it does. For two cars the two expressions are
                # the same number, so nothing about a trained race changes.
                gmat = torch.zeros(self.B // self.M, self.M, device=self.device)
                rk = rank.long()
                gmat[race, rk] = torch.where(rk > 0, gap, torch.zeros_like(gap))
                s_full = base[race] * L - gmat.cumsum(1)[race, rk]                # front car at base, the rest behind
                if e.spawn_order != "behind":
                    # An alongside grid is the one spawn that has to consult the lane: two cars put side
                    # by side where there is no room for two cars are in contact on step 1. So the
                    # lateral offset is a fixed half-separation and the *lane* decides whether that car
                    # takes it: `room` is measured at the car's own spawn point (not the leader's -- the
                    # grid spans up to half a metre of arc and the lane narrows inside that), and where
                    # it is short that car falls back to a stagger. A car abreast and a car staggered is
                    # a perfectly good grid; what is not is two cars sharing a pose.
                    #
                    # Because the distance field is 1-Lipschitz, `edt(centre) - |lat|` is a true lower
                    # bound on the clearance at the offset point, so `room >= |lat|` is exactly the
                    # condition under which the offset survives the pull-back below.
                    #
                    # Cars lined up on a grid are aligned with the track, so an abreast row spawns with
                    # the jitter cut to `spawn_alongside_yaw`: keeping the full 0.2 rad would need 0.73 m
                    # of centre separation at 3 sigma and no lane here has it to spare.
                    psi = 3.0 * min(e.spawn_yaw_std, e.spawn_alongside_yaw)
                    extent = (self.cfg.vehicle.width * math.cos(psi)
                              + self.cfg.vehicle.length * math.sin(psi))
                    al = e.spawn_alongside_gap[0] + (e.spawn_alongside_gap[1] - e.spawn_alongside_gap[0]) \
                        * torch.rand(n, device=self.device, generator=gen)
                    want = (slot.float() - (self.M - 1) / 2.0) * (extent + e.spawn_alongside_sep)
                    # Independent per car, not `slot * al`: an accumulating stagger spreads a three-car
                    # grid over a metre of arc, and on a 2 m radius the centerline normal turns enough
                    # over that distance that "0.56 m apart along the normal" stops meaning 0.56 m
                    # apart. Inside 0.2 m the frames agree to a couple of centimetres.
                    s_slot = torch.remainder(base[race] * L - al, L)
                    xy0, yaw0 = self.sim.track.pose_at_s(s_slot, self.sim.tid[ids])
                    room = self.sim.track.sample_edt(xy0, self.sim.tid[ids]) - 0.5 * extent - e.opp_event_margin
                    # Per *race*, not per car: a race with one car abreast and another staggered is
                    # exactly the grid that collides, because the stagger puts the last slot back on the
                    # leader's own arc position -- where the abreast car already is. Measured with the
                    # decision taken per car: 232 of 528 three-car spawns in contact.
                    headroom = room - want.abs() - 0.03
                    if getattr(self.sim.track, "has_props", False):
                        # Props are not in the distance field, so `room` cannot see them. A crate
                        # standing where the grid wants to be would send the abreast cars into
                        # `sample_spawn`'s prop rejection, which replaces a blocked pose with a
                        # *centerline* one -- and two cars pulled onto one line are two cars in contact.
                        # So the grid is tested against the layout as well (drawn already, see
                        # `_redraw_layouts`), and a race the crates block starts staggered instead: that
                        # path goes through the rejection safely, because it has no lateral offset to
                        # lose. `-inf` rather than a flag so the per-race reduction below covers it.
                        nrm = torch.stack([-torch.sin(yaw0), torch.cos(yaw0)], 1)
                        pose_ab = torch.cat([xy0 + nrm * want[:, None], yaw0[:, None]], 1)
                        blocked = self.sim._spawn_in_prop(pose_ab, self.sim.tid[ids], ids)
                        headroom = torch.where(blocked, torch.full_like(headroom, -math.inf), headroom)
                    worst = torch.full((self.B // self.M,), math.inf, device=self.device)
                    worst.scatter_reduce_(0, race, headroom, reduce="amin", include_self=False)
                    abreast = (order == SPAWN_ORDER_ID["alongside"]) & is_full & (worst[race] >= 0.0)
                    rank = torch.where((order == SPAWN_ORDER_ID["alongside"]) & ~abreast,
                                       (self.M - 1 - slot).float(), rank)
                    rk = rank.long()
                    gmat = torch.zeros(self.B // self.M, self.M, device=self.device)
                    gmat[race, rk] = torch.where(rk > 0, gap, torch.zeros_like(gap))
                    s_full = torch.where(abreast, s_slot, base[race] * L - gmat.cumsum(1)[race, rk])
                    lat = torch.where(abreast, want,
                                      torch.randn(n, device=self.device, generator=gen) * e.spawn_lateral_std)
                    min_clear = torch.where(abreast, torch.full_like(lat, 0.5 * extent + e.opp_event_margin),
                                            torch.full_like(lat, e.spawn_min_clearance))
                    yaw_std = torch.where(abreast, torch.full_like(lat, min(e.spawn_yaw_std, e.spawn_alongside_yaw)),
                                          torch.full_like(lat, e.spawn_yaw_std))
            other = self.sim.other_idx[ids, 0]
            s_part = self.sim.s[other] - gap                                  # behind the next car of the race
            s = torch.remainder(torch.where(is_full, s_full, s_part), L)
            if self.slots is not None:
                # Each car draws from its own band, and what the draw scales -- a teacher's profile
                # or a policy's cap -- follows from who drives it.
                self._slot_speeds(ids, slot, is_full, full, gen)
            else:
                scale = e.opp_speed_range[0] + (e.opp_speed_range[1] - e.opp_speed_range[0]) * torch.rand(full.shape[0], device=self.device, generator=gen)
                self.opp_scale[ids] = torch.where(is_full, scale[race], self.opp_scale[ids])
                if e.opponent in ("policy", "mixed", "pool") and e.selfplay_front_cap:   # heterogeneous self-play
                    # scaled against the pace the policy actually drives (selfplay_pace_ref), not the
                    # curriculum cap: at cap 9.0 the policy runs corner-limited at ~4.5 m/s, so 0.5-1.0x
                    # of 9.0 bound almost nothing and the races stayed processions (contact and
                    # proximity terms ~0, same as with identical cars). Teacher opponents were scaled
                    # against their raceline profile (~4.8 m/s), which is the analogue.
                    front_cap = (self.opp_scale[ids] * e.selfplay_pace_ref).clamp(max=self.cap_base)
                    capped = (rank == 0) & ~scripted_row & (order != SPAWN_ORDER_ID["alongside"])
                    if e.opponent == "pool":
                        # A checkpoint drives at its own pace, and the only handle `--opp-speed` has on
                        # it is the cap -- the same handle self-play uses on its front car. A scale
                        # above 1.0 clamps to the curriculum cap, i.e. leaves it alone: being overtaken
                        # by a pool car is a question of the grid, not of a cap.
                        capped = capped | self.pool_driven[ids]
                    self.cap_scale[ids] = torch.where(capped, front_cap / max(self.cap_base, 1e-6),
                                                      torch.ones_like(self.opp_scale[ids]))
                    self.speed_cap[ids] = self.cap_base * self.cap_scale[ids]
        poses = self.sim.sample_spawn(n, e.spawn_lateral_std, yaw_std, s=s, tid=self.sim.tid[ids],
                                      min_clearance=min_clear, lat=lat, eid=ids)
        speed = torch.rand(n, device=self.device, generator=gen) * e.spawn_speed_max
        self.sim.reset(ids, poses, speed)
        self.sim.odom.state[ids, 3] = speed
        self.prev_action[ids] = 0.0
        if self.act_dim == 2:
            self.prev_action[ids, 1] = speed / e.v_max_policy * 2 - 1
        else:
            self.prev_action[ids, -2:] = (speed / e.v_max_policy * 2 - 1)[:, None]
            self.tracker.reset(ids); self._calibrate_tracker(ids)
        self.prev_steer_norm[ids] = 0.0; self.last_cmd[ids] = 0.0; self.last_cmd[ids, 1] = speed
        if self.events is not None:
            self.events.reset(ids)                                  # a respawned car starts with no event
        if self.pool is not None:
            # A pool checkpoint may carry memory: a hidden state kept across a respawn is a policy
            # remembering a track its car is no longer on.
            self.pool.reset(ids)
        self.gap_prev[ids] = 0.0; self.gap_valid[ids] = False       # no gain scored on the first step
        self.act_hist[ids] = self.prev_action[ids][:, None, :]
        self.ep_step[ids] = 0; self.ep_return[ids] = 0.0; self.ep_progress[ids] = 0.0
        self.lap_start_step[ids] = 0; self.prev_lap[ids] = 0
        if self.ecfg.reward_lap_time > 0:
            # a car spawns mid-sector, so its first boundary is not a sector it drove: start the clock
            # but withhold the reward until it has crossed one boundary and entered the next cleanly
            s_new = self.sim.s[ids]
            L_ = self.sim.track.length[self.sim.tid[ids]].clamp_min(1e-6)
            self.sector_idx[ids] = (s_new / L_ * self.sector_lim.shape[1]).long().clamp_(0, self.sector_lim.shape[1] - 1)
            self.sector_t0[ids] = 0.0; self.sector_valid[ids] = False; self.s_prev[ids] = s_new
        # fill the scan history with a fresh scan from the new pose. Scanned for the whole batch on the
        # compiled path (fixed shapes -> one CUDA graph); a per-reset partial batch would run eager kernels
        cars = self.sim._car_boxes(self.sim.state) if self.M > 1 else None
        scan, scan_true, scan_type = self.sim.lidar.scan(self.sim.state[:, :3], None, self.sim.P, motion_distortion=False, tid=self.sim.tid, cars=cars, compiled=True, eid=self.sim.eid)
        self.scan_hist[ids] = self._norm_scan(scan[ids])[:, None, :]
        if self.hist is not None:
            feat = torch.zeros(ids.numel(), 9, device=self.device); feat[:, 0] = speed / e.v_max_policy
            self.hist[ids] = torch.cat([feat, torch.zeros(ids.numel(), self.act_dim, device=self.device)], 1)[:, None, :]

        return scan, scan_true, scan_type

    def _redraw_layouts(self, ids: torch.Tensor, n: int, full: Optional[torch.Tensor] = None) -> None:
        """Draw a fresh obstacle layout for the envs that are resetting. No-op when off.

        Called *before* the grid is laid out rather than just before the spawn, for two reasons.
        `sample_spawn` rejects a pose that lands inside a prop and walks the lap until one is clear,
        so it has to test the layout the car is about to drive rather than the one it just crashed
        out of. And an abreast grid (`spawn_order`) has to be tested against those same crates,
        which the lane's own distance field cannot see -- so the layout has to exist by the time
        `_reset_envs` decides whether the grid fits.
        """
        if self.procedural is None:
            return
        if self.M == 1:
            self.procedural.redraw(self.sim.tid[ids], ids, torch.arange(n, device=self.device))
            return
        # One layout per race, not per car. The cars of a race share a track and see each other;
        # giving them different crates would have car 0 collide with a box car 1 cannot see. A race
        # redraws only when it resets as a whole -- a single car respawning behind its mates keeps
        # the layout the race is running, exactly as it keeps the race's track.
        races = torch.nonzero(full).flatten()
        if races.numel():
            rows = (races[:, None] * self.M + torch.arange(self.M, device=self.device)[None]).reshape(-1)
            src = torch.arange(races.numel(), device=self.device).repeat_interleave(self.M)
            self.procedural.redraw(self.sim.tid[races * self.M], rows, src)

    def _spawn_order(self, ids: torch.Tensor, race: torch.Tensor, is_full: torch.Tensor,
                     n_races: int, gen: torch.Generator) -> torch.Tensor:
        """(n,) grid each resetting car's race is on, from `SPAWN_ORDER_ID`.

        Drawn per race and remembered, so a car respawning mid-race rejoins the grid its race was
        started on. `behind` -- every race trained before this -- draws nothing and returns zeros,
        which is what keeps an unflagged run bit-identical.
        """
        if self.ecfg.spawn_order == "behind":
            return torch.zeros(ids.numel(), dtype=torch.long, device=self.device)
        if self.ecfg.spawn_order == "random":
            pick = torch.randint(3, (n_races,), device=self.device, generator=gen)
        else:
            pick = torch.full((n_races,), SPAWN_ORDER_ID[self.ecfg.spawn_order],
                              dtype=torch.long, device=self.device)
        self.spawn_order_race[ids] = torch.where(is_full, pick[race], self.spawn_order_race[ids])
        return self.spawn_order_race[ids]

    def _reset_result(self, r: StepResult, ids: torch.Tensor, scans: tuple[torch.Tensor, ...]) -> StepResult:
        """Refresh reset rows without advancing any car or mutating transition info."""
        current = replace(r, **{f.name: value.clone() for f in fields(r)
                               if isinstance(value := getattr(r, f.name), torch.Tensor)})
        current.state[ids] = self.sim.state[ids]
        current.odom[ids] = self.sim.odom.state[ids]
        current.s[ids] = self.sim.s[ids]
        _, lateral, _ = self.sim.track.project(self.sim.state[:, :2], self.sim.tid)
        current.lateral[ids] = lateral[ids]
        current.wall_dist[ids] = self.sim._footprint_clearance(self.sim.state)[ids]
        for target, scan in zip((current.scan, current.scan_true, current.scan_type), scans):
            target[ids] = scan[ids]
        for target in (current.attitude, current.imu, current.imu_att, current.collision,
                       current.progress, current.lap, current.car_collision):
            if target is not None:
                target[ids] = 0
        return current

    def _opponent_actions(self, action: torch.Tensor) -> torch.Tensor:
        if self.M == 1 or self.ecfg.opponent not in ("teacher", "mixed", "pool", "slots"):
            return action
        ev_speed, ev_off = None, None
        if self.events is not None:
            # The gate is read fresh rather than cached: in "mixed" and "pool" mode which cars are
            # teacher-driven is redrawn at every race reset, and a stale gate would script a car the
            # policy is driving.
            self.events.set_gate(self.teacher_driven)
            self.events.step()
            if self.events.react_on:
                # The reactive half, after the timed one and before either is read: what it needs is
                # where the learner is *now*, which is the state this step's command will act on.
                self.events.step_reactive(self.learner_view())
            ev_speed, ev_off = self.events.speed_scale(), self.events.lateral_offset()
        out = action
        if self.pool is not None:
            obs = self._opponent_obs()
            if obs is not None:
                # No gradient, and on the observation the policy itself last saw: a pool opponent
                # is a checkpoint driving the same car in the same race, not a privileged
                # controller.
                with torch.no_grad():
                    out = torch.where(self.pool_driven[:, None],
                                      self.pool.act(obs, self.opp_driver), out)
        if not self.teacher_any:
            return out
        if self.teacher is None:
            raise RuntimeError(f"opponent == {self.ecfg.opponent!r} needs env.set_teacher(RacelineTeacher)")
        follow, v_cap = self.follow_cap(self.sim.state)
        if self.events is not None:
            blind = self.events.oblivious_mask()
            if blind is not None:
                # The one behaviour that makes an opponent *more* dangerous rather than less. It is
                # subtracted from the follow mask rather than added to the command, so the "an event
                # never lifts an opponent over its follow cap" invariant is untouched: this car
                # never had a cap to be lifted over.
                follow = follow & ~blind
        an = self._teacher_normalized(self.teacher, ev_off, ev_speed, follow, v_cap)
        for kind, alt in zip(self.alt_teacher_kinds, self.alt_teachers):
            # A teacher kind that is not the raceline teacher drives its own slots. Asked for the
            # whole batch and selected, like the pool is, for the same reason: compacting to the
            # rows it owns would be a device-to-host sync every step. The mask is fixed for the life
            # of the env, so this loop is empty unless a slot actually named such a kind -- and it
            # is what stops a kind the tree has from being silently driven by the raceline teacher
            # the moment its module appears.
            an = torch.where(self.alt_teacher_mask[kind][:, None],
                             self._teacher_normalized(alt, ev_off, ev_speed, follow, v_cap), an)
        return torch.where(self.teacher_driven[:, None], an, out)

    def _teacher_normalized(self, teacher, ev_off, ev_speed, follow, v_cap):
        """One teacher's command for the whole batch, as a normalized action.

        Split out of `_opponent_actions` so that a second teacher kind is a second call rather than
        a second copy; the instructions are the ones that were inline, in order.
        """
        if self.act_dim == 2:
            cmd = teacher(self.sim.state, self.sim.P, self.sim.tid, offset=ev_off)
            v = cmd[:, 1] * self.opp_scale
            # Before the follow cap, never after: an event can only slow a car (speed_scale <= 1), so
            # taking the minimum of the two leaves an opponent that is already braking for the car
            # ahead braking. Applied the other way round, a "resume" would drive it into that car.
            v = v if ev_speed is None else v * ev_speed
            v = torch.where(follow, torch.minimum(v, v_cap), v)
            return self.teacher_action_to_normalized(torch.stack([cmd[:, 0], v], 1))
        an = teacher.plan_action(self.sim.state, self.sim.P, self.sim.tid, self.ecfg.v_max_policy,
                                 self.tracker.spec, offset=ev_off)
        an = an.clone(); an[:, -2:] = ((an[:, -2:] + 1) * self.opp_scale[:, None] - 1).clamp(-1, 1)   # speed scale
        if ev_speed is not None:                             # same scaling in the normalized plan speeds
            an[:, -2:] = (ev_speed[:, None] * (an[:, -2:] + 1) - 1).clamp(-1, 1)
        cap_n = (v_cap / self.ecfg.v_max_policy * 2 - 1)[:, None]
        an[:, -2:] = torch.where(follow[:, None], torch.minimum(an[:, -2:], cap_n), an[:, -2:])
        return an

    def _opponent_obs(self):
        """The observation a pool opponent acts on, or None before one exists.

        After `reset()` this is always the observation the caller was last handed, which is the one
        the learner's own policy saw -- the pool is asked for its action at the top of `step()`,
        before any new observation exists. None happens on exactly two kinds of step, both thrown
        away by construction: `sim.warmup()`'s throw-away steps, and the one
        `learn.graph_runtime.prepare_graph_runtime` takes to record the solver's arguments, neither
        of which has run a reset yet. The pool sits those out rather than being handed a zeroed
        observation, which would be a policy driving on a scan that says "no returns anywhere".
        """
        if self._last_obs is not None:
            return self._last_obs
        return None if self.last_result is None else self._obs(self.last_result)

    def follow_cap(self, state: torch.Tensor):
        """(is there a car within opp_follow_gap ahead of each car, the speed to hold behind it)."""
        e = self.ecfg
        d = self.signed_gaps(self.sim.s, self.sim.tid)                                  # (B, M-1)
        ahead = torch.where(d > 0, d, torch.full_like(d, 1e9))
        gap, j = ahead.min(1)
        v_ahead = state[self.sim.other_idx.gather(1, j[:, None])[:, 0], 3]
        closing = (state[:, 3] - v_ahead).clamp_min(0.0)
        trigger = e.opp_follow_gap + closing * closing / (2.0 * e.opp_follow_decel)     # room to brake in
        return gap < trigger, (v_ahead * e.opp_follow_ratio).clamp_min(0.5)

    def car_contact_charge(self, s: torch.Tensor, tid: torch.Tensor, car_hit: torch.Tensor) -> torch.Tensor:
        """The contact penalty falls on the car behind. The one in front was driven into; charging it
        would teach that completing a pass is dangerous, which is the opposite of the lesson."""
        d = self.signed_gaps(s, tid)
        nearest = d.abs().argmin(1, keepdim=True)
        dn = d.gather(1, nearest)[:, 0]
        # exempt only a car that is clearly *behind* me (more than a car length). At |gap| below a car
        # length the two are alongside -- the contact is the pass itself, and the sign of a near-zero
        # arc gap says nothing about who moved into whom; measured, 16 of 43 contacts were "from
        # behind" by sign and nearly all of them were the learner cutting across during a pass.
        responsible = dn > -0.6
        return car_hit * responsible.to(car_hit.dtype)

    # ------------------------------------------------------------------ API
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.sim.gen.manual_seed(seed); torch.manual_seed(seed)
        self._reset_envs(torch.arange(self.B, device=self.device))
        # one zero-action step so that odom/scan come from the simulator path
        r = self.sim.step(torch.stack([torch.zeros(self.B, device=self.device), self.sim.state[:, 3]], 1))
        self.scan_hist = torch.roll(self.scan_hist, 1, 1); self.scan_hist[:, 0] = self._norm_scan(r.scan)
        self.last_result = r
        obs = self._obs(r)
        if self.hist is not None:                                  # history starts from the first real observation (as the car's builder does)
            self.hist[:] = torch.cat([self._last_feat, torch.zeros(self.B, self.act_dim, device=self.device)], 1)[:, None, :]
            obs = self._obs(r)
        self._last_obs = obs
        return obs, {"priv": self._priv(r)}

    def step(self, action: torch.Tensor):
        a = self._opponent_actions(action.to(self.device).clamp(-1.0, 1.0))
        if self.act_dim == 2:
            v_cmd = torch.minimum((a[:, 1] + 1) * 0.5 * self.ecfg.v_max_policy, self.speed_cap)
            cmd = torch.stack([a[:, 0] * self.s_max, v_cmd], 1)
        else:                                                  # plan -> (steer, speed) through the tracker
            lr = self.last_result
            v_meas = lr.odom[:, 3] if lr is not None else self.sim.state[:, 3]
            yaw_rate = lr.imu[:, :, 2].mean(1) if (lr is not None and lr.imu is not None and lr.imu.shape[1] > 0) else None
            # Before the solve, because `last_pred` comes back in the body frame of exactly this
            # pose and `car_future` has to put it back in the world.
            self._plan_pose = self.sim.state[:, :3].clone()
            raw = self.tracker(a, v_meas, self.speed_cap, yaw_rate, delay=self.tracker_delay)
            self.last_cmd_raw = raw                                # what the tracker asked for (before calibration)
            cal = self.tracker_cal
            cmd = torch.stack([((raw[:, 0] - cal[:, 0]) / cal[:, 1]).clamp(-self.s_max, self.s_max), raw[:, 1] / cal[:, 2]], 1)
        if self.cmd_shaper is not None:
            # A controller sitting between the policy and the VESC: `learn/traction_arm.py` is the
            # one that exists, and it shapes the speed the way `policy_node` shapes `/drive` on the
            # car. Before the external override on purpose -- a teleoperated or ROS-driven car is
            # the mux output, which the node does not shape either.
            cmd = self.cmd_shaper(cmd)
        if self._ext_ids:
            cmd = torch.where(self.ext_mask[:, None], self.ext_cmd, cmd)
        steer_norm = cmd[:, 0] / self.s_max
        r = self.sim.step(cmd)
        self.last_cmd = cmd
        e = self.ecfg
        self.ep_step += 1
        plan_ref = self.tracker.last_ref if self.tracker is not None and e.reward_plan_clearance > 0 else self._no_plan
        car_hit = (r.car_collision.float() if r.car_collision is not None
                   else torch.zeros_like(self.ep_return))
        out = self._math(r.scan, r.wall_dist, r.s, r.state, r.progress, r.collision, r.lap, a, steer_norm, self.prev_steer_norm,
                         self.scan_hist, self.act_hist, self.ep_step, self.sim.tid, self.ep_return, self.ep_progress, self.prev_lap,
                         plan_ref, self.lap_start_step, car_hit, self.gap_prev, self.gap_valid)
        (self.scan_hist, steer_rate, reward, reward_components, self.act_hist, terminated, truncated,
         crossed, done, self.ep_return, self.ep_progress, flags, self.gap_prev,
         self.gap_valid) = (t.clone() for t in out)
        self.prev_steer_norm = steer_norm; self.prev_action = a
        if self.hist is not None:
            self.hist = torch.cat([torch.cat([self._last_feat, a], 1)[:, None, :], self.hist[:, :-1]], 1)
        obs = self._obs(r)
        any_done, any_lap = flags.tolist()                          # the one host sync of the step
        # true lap times: time between consecutive finish-line crossings (a spawn mid-track does not count)
        if any_lap:
            lap_ids = torch.nonzero(crossed & (self.prev_lap > 0)).flatten()
            lap_times = (self.ep_step[lap_ids] - self.lap_start_step[lap_ids]).float() * self.sim.control_dt
            # same physical bound as the lap bonus, so a wobble on the line cannot be reported as the
            # best lap of the run (it was, repeatedly: 0.05 s)
            real = lap_times >= self.sim.track.length[self.sim.tid[lap_ids]] / e.v_max_policy
            lap_ids, lap_times = lap_ids[real], lap_times[real]
            self.lap_start_step[crossed] = self.ep_step[crossed]
        else:
            lap_ids = self._empty_long; lap_times = self._empty_float
        if e.reward_lap_time > 0:
            self._sector_time_reward(r.s, reward, reward_components)
        self.prev_lap = r.lap.clone()
        info = {"priv": self._priv(r), "progress": r.progress, "lap": r.lap, "wall_dist": r.wall_dist,
                # a snapshot, like track_id below: _reset_envs re-draws the roles in place further
                # down this same step, and the trainer reads this after step() has returned
                "on_policy": self.on_policy.clone(),
                "scan_true": r.scan_true, "track_id": self.sim.tid.clone(), "lap_times": lap_times, "lap_ids": lap_ids,
                "learner": self.learner,
                "reward_components": dict(zip(REWARD_COMPONENT_KEYS, reward_components.unbind(1)))}
        if r.car_collision is not None:
            info["car_collision"] = r.car_collision
        if self.events is not None:
            # What each opponent is doing right now, for a viewer or a logger: event id (0 = none,
            # f1sim.opponent_events.EVENT_ID), seconds left, and the lateral offset it is holding.
            info["opp_event"] = {k: v.clone() for k, v in self.events.info().items()}
        if self.tracker is not None:
            info["plan"] = self.tracker.last_ref                 # (B, N+1, 4) body-frame x, y, heading, speed
        if any_done:
            ids = torch.nonzero(done).flatten()
            info["final"] = {"ids": ids, "return": self.ep_return[ids].clone(), "progress": self.ep_progress[ids].clone(),
                             "steps": self.ep_step[ids].clone(), "collided": terminated[ids].clone(),
                             "lap_time": self.ep_step[ids].float() * self.sim.control_dt}
            # final observation of the ended episodes (before auto-reset)
            info["final_obs"] = {k: v[ids].clone() for k, v in obs.items()}
            info["final_priv"] = self.privileged(r)[ids].clone()
            scans = self._reset_envs(ids)
            r = self._reset_result(r, ids, scans)
            obs = self._obs(r)
        self.last_result = r
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def _step_math(self, scan, wall_dist, s, state, progress, collision, lap, a, steer_norm, prev_steer_norm, scan_hist, act_hist, ep_step, tid,
                   ep_return, ep_progress, prev_lap, plan_ref, lap_start_step, car_hit, gap_prev, gap_valid):
        """Reward, histories and episode flags as pure tensor math (compiled into one CUDA graph when
        the sim runs in reduce-overhead mode: next to a training job every small kernel waits its turn)."""
        e = self.ecfg
        scan_hist = torch.cat([self._norm_scan(scan)[:, None, :], scan_hist[:, :-1]], 1)
        steer_rate = (steer_norm - prev_steer_norm).abs()
        proximity = (e.safe_dist - wall_dist).clamp(min=0.0) / e.safe_dist if e.reward_proximity > 0 else torch.zeros_like(wall_dist)
        if e.proximity_speed_ref > 0:
            proximity = proximity * (1.0 + state[:, 3].abs() / e.proximity_speed_ref)
        wrong_way = torch.zeros_like(wall_dist)
        if e.reward_wrong_way > 0 and self.sim.track.cl is not None:
            _, yaw_c = self.sim.track.pose_at_s(s, tid)
            wrong_way = (torch.cos(state[:, 2] - yaw_c) < 0.0).float()      # more than 90 deg off the lane direction
        crash = collision.float()
        plan_penalty = torch.zeros_like(wall_dist)
        if e.reward_plan_clearance > 0 and plan_ref.shape[1] > 1:
            cosine, sine = torch.cos(state[:, 2])[:, None], torch.sin(state[:, 2])[:, None]
            plan_x = state[:, 0:1] + plan_ref[:, :, 0] * cosine - plan_ref[:, :, 1] * sine
            plan_y = state[:, 1:2] + plan_ref[:, :, 0] * sine + plan_ref[:, :, 1] * cosine
            clearance = self.sim.track.sample_edt(torch.stack([plan_x, plan_y], -1), tid[:, None]).min(1).values
            clearance = clearance - 0.5 * self.cfg.vehicle.width
            plan_penalty = (e.plan_margin - clearance).clamp(min=0.0) / e.plan_margin
        # per metre *driven*, not per metre of centerline progress: with progress the penalty vanishes
        # for a car that stops next to a wall (progress -> 0) and is charged to one that reverses away
        # from it, so "park against the boundary" was the cheapest way to satisfy it.
        travelled = state[:, 3].abs() * self.sim.control_dt
        lap_bonus = torch.zeros_like(progress)
        if e.reward_lap > 0:
            # only a lap between two finish-line crossings has a defined time; the first crossing
            # after a mid-track spawn does not (prev_lap == 0), matching the lap-time bookkeeping
            full_lap = (lap > prev_lap) & (prev_lap > 0)
            lap_dt = (ep_step - lap_start_step).to(progress.dtype) * self.sim.control_dt
            L_lap = self.sim.track.length[tid]
            avg_speed = L_lap / lap_dt.clamp_min(1e-3)
            # A crossing is not a lap unless the time since the last one could physically hold one.
            # The lap counter is symmetric: it counts down when the car slips back over the line and
            # up again when it returns, so a wobble on the start line reads as a lap driven in two
            # control steps. Those show up as 0.05 s "best laps" in the metrics, and here they would
            # be paid an average speed of hundreds of m/s.
            lap_bonus = torch.where(full_lap & (lap_dt >= L_lap / e.v_max_policy),
                                    e.reward_lap * avg_speed, lap_bonus)
        # Overtaking: arc length taken out of the field this step, averaged over the opponents.
        # Positive while closing or passing, negative while being passed. Rewarding the *change*
        # rather than the position keeps it dense -- a pass is one step in a long approach, and a
        # sparse bonus at the moment of passing would be nearly unlearnable.
        #
        # The gap is a *signed* arc wrapped to (-L/2, L/2], not the forward arc to the other car.
        # The forward arc jumps from ~0 to ~L at the instant a pass completes, and the reward reads
        # that jump as a full lap lost: measured, the step that completes a pass scored -2.00 (the
        # clamp) where it should have scored +0.25, and a whole pass paid +5.75 instead of +8.00.
        # The single most informative step of the manoeuvre carried its largest penalty.
        #
        # Averaged over every opponent rather than read off one slot: other_idx is a fixed roster,
        # not "the car ahead", so with three cars the old form tracked whichever car happened to sit
        # in slot 0. Summed over the field there is no target to switch between, and passing anyone
        # pays.
        gap_gain = torch.zeros_like(progress)
        if e.reward_overtake > 0 and self.sim.other_idx is not None:
            gap_gain, _ = self.overtake_gain(s, tid, gap_prev, gap_valid)
        car_prox = self.car_proximity(state) if (e.reward_car_proximity > 0 and self.sim.other_idx is not None) \
            else torch.zeros_like(progress)
        reward_components = torch.stack([
            e.reward_progress * progress,
            e.reward_collision * crash,
            -e.reward_collision_speed * crash * state[:, 3].abs(),
            -e.reward_steer_rate * steer_rate,
            -e.reward_proximity * proximity * travelled,
            -e.reward_plan_clearance * plan_penalty * self.sim.control_dt,
            -e.reward_wrong_way * wrong_way,
            lap_bonus,
            torch.full_like(progress, e.reward_alive),
            -e.reward_car_contact * (self.car_contact_charge(s, tid, car_hit) if self.sim.other_idx is not None else car_hit),
            e.reward_overtake * gap_gain,
            torch.zeros_like(progress),          # lap_time: filled in on the crossing, in step()
            -e.reward_car_proximity * car_prox * travelled,
            -e.reward_sideslip * (torch.atan2(state[:, 4].abs(), state[:, 3].abs().clamp_min(0.5)) - e.sideslip_free).clamp_min(0.0) * travelled,
        ], 1)
        reward = reward_components.sum(1)
        act_hist = torch.cat([a[:, None, :], act_hist[:, :-1]], 1)
        terminated = collision.clone()
        truncated = (~terminated) & ((ep_step >= e.max_steps) | (lap >= e.laps))
        if self.M > 1:                                         # the leader's time limit ends the whole race
            truncated = truncated | (truncated & (self.slot == 0)).view(-1, self.M)[:, 0].repeat_interleave(self.M)
            truncated = truncated & ~terminated
        crossed = lap > prev_lap
        done = terminated | truncated
        flags = torch.stack([done.any(), crossed.any()])
        gap_now = torch.zeros_like(gap_prev)
        if self.sim.other_idx is not None:
            gap_now = self.signed_gaps(s, tid)
        return (scan_hist, steer_rate, reward, reward_components, act_hist, terminated, truncated, crossed,
                done, ep_return + reward, ep_progress + progress, flags, gap_now,
                torch.ones_like(gap_valid))

    def signed_gaps(self, s: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """Arc to each opponent, signed and wrapped to (-L/2, L/2]: + is ahead of me, - is behind.

        Signed, because the forward arc jumps from ~0 to ~L the instant a pass completes and any
        reward reading its change sees a whole lap lost at the one moment that mattered.
        """
        L_ = self.sim.track.length[tid][:, None]
        return (s[self.sim.other_idx] - s[:, None] + L_ / 2) % L_ - L_ / 2

    def car_proximity(self, state: torch.Tensor) -> torch.Tensor:
        """(B,) closeness to the most threatening opponent: 0 outside car_safe_gap of its body, 1 at
        contact, scaled up by the speed the gap is closing at. Euclidean, not arc: the overtake term
        reads arc so a pass can go through zero arc gap, and this reads the body distance so the pass
        has to go *around* the other car rather than through it."""
        e = self.ecfg
        o = self.sim.other_idx                                             # (B,C)
        d = state[o][:, :, :2] - state[:, None, :2]                        # (B,C,2) me -> them
        dist = d.norm(dim=2).clamp_min(1e-6)
        c, sn = torch.cos(state[:, 2]), torch.sin(state[:, 2])
        vme = torch.stack([state[:, 3] * c - state[:, 4] * sn, state[:, 3] * sn + state[:, 4] * c], 1)
        co, so = torch.cos(state[o][:, :, 2]), torch.sin(state[o][:, :, 2])
        vth = torch.stack([state[o][:, :, 3] * co - state[o][:, :, 4] * so,
                           state[o][:, :, 3] * so + state[o][:, :, 4] * co], 2)
        closing = -((vth - vme[:, None, :]) * d).sum(2) / dist             # >0 when the gap is shrinking
        # Body-to-body clearance from an oriented box, not centre distance minus a constant. Cars are
        # 0.5 long and 0.3 wide: alongside, the bodies touch at 0.3 m between centres, so a
        # centre-distance term with a 0.45 m "body" was already saturated at full penalty with 0.3 m
        # of daylight between the cars, and could not tell a rub from a clean pass. Measured on the
        # race leg: 100 % of the learner's contacts were side contacts, and 4M steps of training
        # against that saturated term moved the contact count by nothing.
        lon = (d[..., 0] * c[:, None] + d[..., 1] * sn[:, None]).abs()
        lat = (-d[..., 0] * sn[:, None] + d[..., 1] * c[:, None]).abs()
        gap = torch.sqrt((lon - e.car_len).clamp_min(0.0) ** 2 + (lat - e.car_wid).clamp_min(0.0) ** 2)
        closeness = (1.0 - gap / e.car_safe_gap).clamp(0.0, 1.0)
        scale = 1.0 + closing.clamp_min(0.0) / e.car_prox_speed_ref
        return (closeness * scale).max(1).values

    def overtake_gain(self, s: torch.Tensor, tid: torch.Tensor, gap_prev: torch.Tensor,
                      gap_valid: torch.Tensor):
        """(arc taken out of the field this step, the new gaps). Averaged over every opponent rather
        than read off one slot -- other_idx is a fixed roster, not "the car ahead"."""
        d_now = self.signed_gaps(s, tid)
        near = (d_now.abs() < self.ecfg.overtake_range) & (gap_prev.abs() < self.ecfg.overtake_range)
        gain = torch.where(gap_valid[:, None] & near, (gap_prev - d_now).clamp(-2.0, 2.0),
                           torch.zeros_like(d_now)).mean(1)
        return gain, d_now

    PRIV_PARAMS = ("mu", "mu_f_scale", "cmd_delay", "servo_tau", "motor_tau", "roll_per_g", "steer_bias", "speed_gain")

    @property
    def priv_mu_index(self) -> int:
        """Column of the true friction in privileged(): after the 8 dynamic-state values and, in a
        race, the 4 nearest-opponent values."""
        return 8 + (4 if self.M > 1 else 0)

    def privileged(self, r: StepResult) -> torch.Tensor:
        """Critic-only vector: true dynamic state, track-relative pose, wall clearance, randomized params."""
        pv = self._priv(r)
        params = torch.stack([self.sim.P[k] for k in self.PRIV_PARAMS], 1)
        return torch.cat([pv, params, (self.speed_cap / self.ecfg.v_max_policy)[:, None]], 1)

    def floor_labels(self, r: Optional[StepResult] = None) -> torch.Tensor:
        """(B, N) int8 per-beam label for the auxiliary floor head: 1 floor, 0 solid, -1 no return.

        A *label*, not an observation: it is `scan_type` (`f1sim.lidar.HIT_*`), which only the
        simulator has, and nothing the policy sees is built from it. The beams that carry no return
        are marked -1 rather than 0 because there is nothing there to be solid -- classifying a
        dropout as "not floor" would train the head to call the sensor's own gaps obstacles, and a
        grazing floor beam is exactly what `lidar.floor_dropout` removes, so those gaps are *more*
        likely floor than average.

        Beam resolution is the observation's, not the sensor's: `scan_subsample` is applied here as
        `_norm_scan` applies it, so label and scan column mean the same beam.
        """
        r = r if r is not None else self.last_result
        typ = r.scan_type[:, ::self.ecfg.scan_subsample]
        scan = self.scan_hist[:, 0]
        lab = torch.where(typ == 3, torch.ones_like(typ, dtype=torch.int8),
                          torch.zeros_like(typ, dtype=torch.int8))
        gap = (typ == 0) | (scan >= 1.0 - 1e-4)
        return torch.where(gap, torch.full_like(lab, -1), lab)

    def future_labels(self, r: Optional[StepResult] = None) -> torch.Tensor:
        """(B, FUTURE_LABEL_DIM) privileged snapshot of THIS instant, in `FUTURE_LABEL_KEYS` order.

        This is a *label*, not an observation: nothing the policy sees is built from it. The
        auxiliary future head is scored against the row belonging to the state K control steps
        later (`f1sim.learn.future.align_future_targets` does the alignment), and
        `f1sim.learn.probe_hidden` regresses the same row out of frozen hidden states, so the head
        and the probe measure the same quantity by construction.

        Columns, all O(1):

        * `opp_lon`, `opp_lat` -- the nearest opponent's position in the ego body frame, divided by
          PRIV_OPP_DIST_SCALE, the same scale `privileged()` puts its present-opponent offsets on.
        * `opp_vlon`, `opp_vlat` -- that opponent's velocity *relative to the ego*, rotated into the
          ego body frame, on the same scale. NOTE this is not `privileged()[10]`: that column is
          `other.vx - ego.vx`, a difference of two body-frame longitudinal speeds taken in two
          different frames, which is a fine present-tense cue and a poor prediction target. Here
          both velocities are taken to the world and the difference is rotated into one frame, so
          `opp_lon + dt * opp_vlon` is (to first order) where the car will be.
        * `ego_speed`, `ego_yaw_rate` -- the ego's own longitudinal speed and yaw rate, on the
          observation's own normalisers (`v_max_policy`, `imu_gyro_scale`).
        * `opp_present` -- 1 where the nearest opponent is inside `overtake_range`, else 0. The four
          opponent columns are ZEROED where it is 0: there is no relative position to a car that is
          not there, and a far-away car's offsets leaking through an unmasked path is exactly the
          "trained on an empty road" failure `AUX_OPP_RANGE_M` was written to stop.

        Solo (`race_size 1`) is legal and gives `opp_present = 0` everywhere: the two ego columns are
        still real targets and the presence logit still has a (constant) label.

        `r` defaults to `last_result`, which is the state the next action will be taken from -- the
        same convention `privileged(env.last_result)` is read under in the trainer's rollout.
        """
        res = self.last_result if r is None else r
        if res is None:
            raise RuntimeError("future_labels needs a stepped env: call reset() first")
        st = res.state
        e = self.ecfg
        out = torch.zeros(self.B, FUTURE_LABEL_DIM, device=self.device, dtype=st.dtype)
        c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        if self.M > 1:
            o = self.sim.other_idx                                  # (B, C)
            d = st[o][:, :, :2] - st[:, None, :2]
            dist = d.norm(dim=2)
            j = dist.argmin(1); ar = torch.arange(self.B, device=self.device)
            dn = d[ar, j]; kj = o[ar, j]
            co, so = torch.cos(st[kj, 2]), torch.sin(st[kj, 2])
            # world velocity of each car, then their difference back in the ego frame
            vwx = (st[kj, 3] * co - st[kj, 4] * so) - (st[:, 3] * c - st[:, 4] * sn)
            vwy = (st[kj, 3] * so + st[kj, 4] * co) - (st[:, 3] * sn + st[:, 4] * c)
            s_ = PRIV_OPP_DIST_SCALE
            present = (dist[ar, j] < e.overtake_range).to(st.dtype)
            out[:, 0] = present * (dn[:, 0] * c + dn[:, 1] * sn) / s_
            out[:, 1] = present * (-dn[:, 0] * sn + dn[:, 1] * c) / s_
            out[:, 2] = present * (vwx * c + vwy * sn) / s_
            out[:, 3] = present * (-vwx * sn + vwy * c) / s_
            out[:, FUTURE_PRESENT_INDEX] = present
        out[:, 4] = st[:, 3] / e.v_max_policy
        out[:, 5] = st[:, 5] / e.imu_gyro_scale
        return out

    def opponent_beam_mask(self, r: Optional[StepResult] = None) -> torch.Tensor:
        """(B, N) 1 where this beam of the NEWEST scan came back off another car, else 0.

        Privileged and train-time only: the label for the beam-mask auxiliary (`learn.motion`). It
        is not a new quantity -- `lidar.Lidar.scan` has classified every beam since cars were cast
        as meshes (`HIT_CAR`), and `StepResult.scan_type` has carried it -- so nothing in the
        simulator changes to produce it.

        Read from the newest scan, which is frame 0 of the observation's stack and the frame the
        aligned residual is computed against; a mask from any other frame would be a label for a
        different picture. `r` defaults to `last_result`, the state the next action is taken from,
        the same convention `privileged()` and `future_labels()` are read under.

        Solo (`race_size 1`) gives all zeros, honestly: there is no other car to hit.
        """
        from .lidar import HIT_CAR
        res = self.last_result if r is None else r
        if res is None:
            raise RuntimeError("opponent_beam_mask needs a stepped env: call reset() first")
        return (res.scan_type == HIT_CAR).to(res.scan.dtype)

    @torch.no_grad()
    def car_future(self, times, model: Optional[str] = None,
                   state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, K, 2) world position every car of the batch is predicted to be at, at each of
        `times` [s] from now.

        This is a **prediction, not a peek**: a batched simulator cannot step ahead and come back,
        so what is carried forward is each car's own controller, which the env already knows this
        step.

        * a **teacher-driven** car is walked along its raceline from where it is now. Its speed
          target at each grid point is that point's own profile speed times the race's speed scale
          times the scheduled event's speed multiplier -- held for exactly as long as that event has
          left to run and 1.0 after it expires -- capped by the follow-gap speed if it is behind
          another car, and walked from the car's measured speed under acceleration bounds. Its
          lateral offset is the one `OpponentEvents` is holding right now (a `shift`'s ramp, a
          `defend`'s block), held constant and clamped against the lane at every walked point. The
          part that is genuinely unknowable is the reactive layer's *future*: `defend` and `yield`
          read where the learner will be, and the learner has not decided yet.
        * a **policy-driven** car (a pool entry, or the learner itself) is read off the plan
          tracker's own predicted trajectory (`PlanTracker.last_pred`), put back into the world
          through the pose it was solved from, and continued straight on at its last predicted
          heading and speed past the tracker's 0.6 s horizon.
        * everything else -- and every row, under `model="constv"` -- is world-frame constant
          velocity, which is also the floor the other two have to beat
          (`python -m f1sim.learn.opp_future_check`).

        `model` defaults to `EnvConfig.opp_future_model`.
        """
        st = self.sim.state if state is None else state
        mode = self.ecfg.opp_future_model if model is None else model
        if mode not in OPP_FUTURE_MODELS:
            raise ValueError(f"opp future model {mode!r} is not one of {list(OPP_FUTURE_MODELS)}")
        t = torch.as_tensor(times, dtype=st.dtype, device=self.device).reshape(-1)
        if t.numel() == 0:
            raise ValueError("car_future needs at least one horizon")
        if float(t.min()) < 0:
            raise ValueError(f"car_future horizons must be >= 0, got {t.tolist()}")
        c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        vw = torch.stack([st[:, 3] * c - st[:, 4] * sn, st[:, 3] * sn + st[:, 4] * c], 1)
        out = st[:, None, :2] + vw[:, None, :] * t[None, :, None]          # constant velocity
        if mode == "constv" or self.M == 1:
            return out
        pred = self._tracker_future(st, t)
        if mode == "pred":
            return out if pred is None else pred
        if mode == "hybrid":
            seam = self._tracker_seam()
            if pred is None or seam is None or self.teacher is None:
                return out if pred is None else pred
            # Both models evaluated at the requested times AND at the seam, so the tail can be
            # attached to the tracker's own endpoint rather than to the walk's -- otherwise the
            # prediction would jump by the walk's error at the join, which is the error the tracker
            # was brought in to avoid.
            tt = torch.cat([t, t.new_full((1,), seam)])
            pa, wa = self._tracker_future(st, tt), self._raceline_future(st, tt)
            late = (t > seam)[None, :, None]
            return torch.where(late, pa[:, -1:] + (wa[:, :-1] - wa[:, -1:]), pa[:, :-1])
        if self.teacher is not None and bool(self.teacher_driven.any()):
            walk = self._raceline_future(st, t)
            out = torch.where(self.teacher_driven[:, None, None], walk, out)
        if pred is not None:
            out = torch.where((~self.teacher_driven)[:, None, None], pred, out)
        return out

    def _tracker_seam(self) -> Optional[float]:
        """[s from now] the last instant the plan tracker's own rollout reaches, or None without one.

        Past this the rollout is a straight line at its final heading and speed, which on a curving
        track is where the raceline walk starts being the better answer.
        """
        if self.tracker is None or self.tracker.last_pred is None:
            return None
        dly = float(self.tracker.spec.delay) if self.tracker_delay is None else float(self.tracker_delay.mean())
        return dly - self.sim.control_dt + (self.tracker.last_pred.shape[1] - 1) * self.tracker.spec.dt

    def _raceline_future(self, st: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The teacher-driven walk of `car_future`, for every row (the caller selects)."""
        T = self.teacher
        tid = self.sim.tid
        dtw = OPP_FUTURE_WALK_DT
        n = max(2, int(math.ceil(float(t.max()) / dtw)) + 1)
        idx0, _ = T.project(st[:, :2], tid)
        ds = T.ds[tid]
        gb = T.grip_bin(self.sim.P, self.B, self.device)
        scale = torch.ones(self.B, device=self.device)
        left = torch.zeros(self.B, device=self.device)
        off = torch.zeros(self.B, device=self.device)
        if self.events is not None and self.events.enabled:
            sc = self.events.speed_scale()
            if sc is not None:
                scale = sc
                left = torch.where(self.events.gate & (self.events.kind != NO_EVENT),
                                   (self.events.dur - self.events.t).clamp_min(0.0), left)
            o = self.events.lateral_offset()
            if o is not None:
                off = o
        follow, v_cap = self.follow_cap(st)
        if self.events is not None:
            blind = self.events.oblivious_mask()
            if blind is not None:
                follow = follow & ~blind
        v_lim = torch.where(follow, v_cap, torch.full_like(v_cap, float("inf")))
        # What a plan tracker actually delivers when a plan asks it to brake is 3.1 m/s^2 against
        # its nominal 5.0 bound (`EnvConfig.opp_follow_decel`); the nominal one would predict a
        # braking opponent stopping sooner than it does.
        a_brk = float(self.ecfg.opp_follow_decel)
        a_acc = float(self.tracker.spec.a_max if self.tracker is not None else PlanSpec().a_max)
        v = st[:, 3].clamp_min(0.0)
        s_w = torch.zeros(self.B, device=self.device)
        pts, arc = [], []
        for k in range(n):
            j = (idx0 + (s_w / ds).round().long()) % T.N
            tn = T.tan[tid, j]
            o_k = T.clamp_offset(off, tid, j)
            pts.append(T.xy[tid, j] + o_k[:, None] * torch.stack([-tn[:, 1], tn[:, 0]], 1))
            arc.append(s_w)
            if k == n - 1:
                break
            on = torch.full_like(left, k * dtw) < left
            v_t = T.speed_at(tid, j, gb) * self.opp_scale * torch.where(on, scale, torch.ones_like(scale))
            v_t = torch.minimum(v_t, v_lim)
            v_n = torch.minimum(torch.maximum(v_t, v - a_brk * dtw), v + a_acc * dtw).clamp_min(0.0)
            s_w = s_w + 0.5 * (v + v_n) * dtw
            v = v_n
        walk = torch.stack(pts, 1)                                          # (B, n, 2)
        # ---- anchored at the car, in position AND heading.
        #
        # The walk is where the raceline goes; the car is not on it and is not pointing along it.
        # Both mismatches have to be carried and then closed, because a pure-pursuit tracker closes
        # both -- over its lookahead, which is a DISTANCE (`OPP_FUTURE_REJOIN_M`), not a time: a car
        # that has just been told to stop does not slide sideways onto the line while standing
        # still. Smoothstep rather than a ramp, so the label does not open with a lateral velocity
        # the car does not have.
        #
        #     p_k = x_now + sum_i R(dpsi * keep_i) (walk_i - walk_{i-1})  -  e_0 (1 - keep_k)
        #
        # which is exactly x_now at k = 0 (whatever the errors) and exactly the walk once keep
        # reaches 0 (whatever the car was doing). Heading matters more than it looks: at 0.1 s a
        # car has travelled 0.38 m, and 0.2 rad of heading error is 7.6 cm of it -- the size of the
        # whole error at that horizon, and larger than the lateral motion being predicted.
        x = (torch.stack(arc, 1) / OPP_FUTURE_REJOIN_M).clamp(0.0, 1.0)      # (B, n)
        keep = 1.0 - x * x * (3.0 - 2.0 * x)
        tan0 = T.tan[tid, idx0]
        dpsi = torch.remainder(st[:, 2] - torch.atan2(tan0[:, 1], tan0[:, 0]) + math.pi,
                               2 * math.pi) - math.pi
        ang = dpsi[:, None] * keep[:, :-1]                                   # (B, n-1)
        c_, s_ = torch.cos(ang), torch.sin(ang)
        d = walk[:, 1:] - walk[:, :-1]
        rot = torch.stack([d[..., 0] * c_ - d[..., 1] * s_, d[..., 0] * s_ + d[..., 1] * c_], 2)
        walk = (st[:, None, :2] + torch.cat([torch.zeros_like(rot[:, :1]), rot.cumsum(1)], 1)
                - (st[:, :2] - walk[:, 0])[:, None, :] * (1.0 - keep)[..., None])
        grid = torch.arange(n, device=self.device, dtype=t.dtype) * dtw
        return _interp_path(walk, grid, t)

    def _tracker_future(self, st: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        """`PlanTracker.last_pred` in the world, sampled at `t`, or None before a plan-mode step."""
        if self.tracker is None or self.tracker.last_pred is None or self._plan_pose is None:
            return None
        z = self.tracker.last_pred                                          # (B, N+1, 4) body frame
        pp = self._plan_pose
        c, sn = torch.cos(pp[:, 2]), torch.sin(pp[:, 2])
        x = pp[:, :1] + z[:, :, 0] * c[:, None] - z[:, :, 1] * sn[:, None]
        y = pp[:, 1:2] + z[:, :, 0] * sn[:, None] + z[:, :, 1] * c[:, None]
        xy = torch.stack([x, y], 2)                                         # (B, N+1, 2)
        # When the samples are, relative to now: the solve predicted forward over the command
        # latency, and one control step has been driven since it was taken.
        dly = float(self.tracker.spec.delay) if self.tracker_delay is None else float(self.tracker_delay.mean())
        grid = dly - self.sim.control_dt + torch.arange(z.shape[1], device=self.device, dtype=t.dtype) * self.tracker.spec.dt
        psi_e = pp[:, 2] + z[:, -1, 2]
        v_e = z[:, -1, 3]
        over = (t[None, :] - grid[-1]).clamp_min(0.0)                       # (1, K) past the horizon
        tail = torch.stack([torch.cos(psi_e), torch.sin(psi_e)], 1)[:, None, :] * (v_e[:, None, None] * over[..., None])
        # Rigidly anchored at the car: the solve's own start pose is a control step stale and
        # latency-compensated, so the shape is right and the origin is not.
        shift = (st[:, :2] - xy[:, 0])[:, None, :]
        return _interp_path(xy, grid, t) + tail + shift

    @torch.no_grad()
    def opponent_future(self, times, model: Optional[str] = None,
                        state: Optional[torch.Tensor] = None):
        """(positions (B, C, K, 2), present (B, C)) for the C other cars of each race.

        `present` is the same rule `future_labels` uses: the car is inside `overtake_range` right
        now. A car further away than that is not being raced, and a cost term that reaches for it
        would move the plan for something that is not there.
        """
        o = self.sim.other_idx
        if o is None:
            raise RuntimeError("opponent_future needs a race (race_size > 1)")
        st = self.sim.state if state is None else state
        fut = self.car_future(times, model=model, state=st)                 # (B, K, 2)
        d = (st[o][:, :, :2] - st[:, None, :2]).norm(dim=2)                 # (B, C)
        return fut[o], d < self.ecfg.overtake_range

    @torch.no_grad()
    def opp_token(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, OPP_TOKEN_CARS * OPP_TOKEN_COLS[mode]) privileged opponent block, or (B, 0) when off.

        An oracle input, not an observation the car could build. Per opponent, nearest first, in the
        ego body frame and on `PRIV_OPP_DIST_SCALE` (the scale `privileged()` and `future_labels`
        already put opponent offsets on):

            Dx, Dy, presence                 ("pos")
            + Dv_x, Dv_y                     ("posvel")
            + the opponent's position at each of OPP_FUTURE_TIMES, relative to the ego's pose NOW
                                             ("future")

        Every column of an absent slot -- fewer than `OPP_TOKEN_CARS` other cars, or one outside
        `overtake_range` -- is exactly 0, including its presence flag. That gating is the whole
        reason presence is a column: "no car" and "a car at the origin" must not be the same input.
        """
        mode = self.opp_token_mode
        cols = OPP_TOKEN_COLS[mode]
        if not cols:
            return torch.zeros(self.B, 0, device=self.device)
        st = self.sim.state if state is None else state
        o = self.sim.other_idx
        C = o.shape[1]
        n = OPP_TOKEN_CARS
        c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        d = st[o][:, :, :2] - st[:, None, :2]                               # (B, C, 2) me -> them
        dist = d.norm(dim=2)
        # nearest first, and a slot that does not exist is pushed past every real one
        big = torch.full_like(dist, float("inf"))
        order = torch.where(dist < self.ecfg.overtake_range, dist, big).argsort(1)[:, :n]
        pad = max(0, n - C)
        if pad:
            order = torch.cat([order, torch.zeros(self.B, pad, dtype=order.dtype, device=self.device)], 1)
        ar = torch.arange(self.B, device=self.device)[:, None]
        kj = o.gather(1, order.clamp(max=C - 1))                            # (B, n) global row of each slot
        sel = dist.gather(1, order.clamp(max=C - 1))
        present = (sel < self.ecfg.overtake_range).to(st.dtype)
        if pad:
            present[:, C:] = 0.0
        s_ = PRIV_OPP_DIST_SCALE
        dn = d[ar, order.clamp(max=C - 1)]                                  # (B, n, 2)
        lon = (dn[..., 0] * c[:, None] + dn[..., 1] * sn[:, None]) / s_
        lat = (-dn[..., 0] * sn[:, None] + dn[..., 1] * c[:, None]) / s_
        parts = [lon, lat, present]
        if cols > 3:
            co, so = torch.cos(st[kj, 2]), torch.sin(st[kj, 2])
            vwx = (st[kj, 3] * co - st[kj, 4] * so) - (st[:, 3] * c - st[:, 4] * sn)[:, None]
            vwy = (st[kj, 3] * so + st[kj, 4] * co) - (st[:, 3] * sn + st[:, 4] * c)[:, None]
            parts += [(vwx * c[:, None] + vwy * sn[:, None]) / s_,
                      (-vwx * sn[:, None] + vwy * c[:, None]) / s_]
        if cols > 5:
            fut = self.car_future(OPP_FUTURE_TIMES, state=st)[kj]           # (B, n, K, 2)
            fd = fut - st[:, None, None, :2]
            fl = (fd[..., 0] * c[:, None, None] + fd[..., 1] * sn[:, None, None]) / s_
            ft = (-fd[..., 0] * sn[:, None, None] + fd[..., 1] * c[:, None, None]) / s_
            for i in range(len(OPP_FUTURE_TIMES)):
                parts += [fl[:, :, i], ft[:, :, i]]
        block = torch.stack(parts, 2) * present[:, :, None]                 # (B, n, cols), gated
        block[:, :, 2] = present                                            # ... except the flag itself
        return block.reshape(self.B, n * cols)

    def race_boundary(self, done: torch.Tensor) -> torch.Tensor:
        """(B,) bool: did ANY car sharing this row's race end its episode on this step?

        A future label read across a reset is a label from a different situation, and in a race the
        reset does not have to be this car's. An opponent that crashes is respawned behind the field
        *in place*, without ending the learner's episode, so its position half a second later is not
        the continuation of the motion the head was asked to extrapolate. The rows are dealt into
        races of M contiguous slots (`self.race = arange(B) // M`), which is what makes this a view
        and a reduction rather than a scatter.
        """
        d = done.to(torch.bool).reshape(-1, self.M)
        return d.any(1, keepdim=True).expand_as(d).reshape(-1)

    def set_external_command(self, idx: int, steer: float, speed: float) -> None:
        """Drive car `idx` with an outside (steer [rad], speed [m/s]) from the next step on, in place
        of the policy's action. Steering is clamped to the vehicle's lock; speed is passed through
        as an external controller commanded it (the bridge does the same), so a speed the actuator
        model cannot reach is the actuator model's problem, not silently capped here."""
        s_max = float(self.s_max)
        steer = max(-s_max, min(s_max, float(steer)))
        self.ext_cmd[idx] = torch.tensor([steer, float(speed)], device=self.device)
        if idx not in self._ext_ids:
            self.ext_mask[idx] = True
            self._ext_ids.add(int(idx))

    def clear_external_command(self, idx: Optional[int] = None) -> None:
        """Hand car `idx` (or every car) back to the policy."""
        if idx is None:
            self.ext_mask.zero_(); self._ext_ids.clear()
        else:
            self.ext_mask[idx] = False; self._ext_ids.discard(int(idx))

    def set_speed_cap(self, v: float):
        self.cap_base = float(min(v, self.ecfg.v_max_policy))
        self.speed_cap = self.cap_base * self.cap_scale

    # convenience for the teacher / IL
    @property
    def state(self) -> torch.Tensor:
        return self.sim.state

    def teacher_action_to_normalized(self, cmd: torch.Tensor) -> torch.Tensor:
        """(steer [rad], speed [m/s]) -> policy action in [-1, 1] (direct action space)."""
        return torch.stack([cmd[:, 0] / self.s_max, cmd[:, 1] / self.ecfg.v_max_policy * 2 - 1], 1).clamp(-1, 1)

    def teacher_label(self, teacher) -> torch.Tensor:
        """The teacher's action in this env's action space: (steer, speed) or its local plan."""
        if self.act_dim == 2:
            return self.teacher_action_to_normalized(teacher(self.sim.state, self.sim.P, self.sim.tid))
        return teacher.plan_action(self.sim.state, self.sim.P, self.sim.tid, self.ecfg.v_max_policy, self.tracker.spec)

    def set_ideal_lap(self, racelines) -> None:
        """Fastest plausible time per track and per sector, from the raceline speed profile: sum(ds / v).

        This is what the time reward measures headroom against, so that a hundredth found near a
        sector's limit outweighs a tenth found while still far off it. The raceline is built by
        moving each resampled centerline point along its own normal, so raceline point j sits at
        centerline fraction j/N and its sector follows from the index alone -- no projection, and
        none of the fold-back ambiguity a projection would have to survive.

        A point-mass optimum is not a hard bound: a real lap can beat it in places. reward_time_floor
        keeps the headroom positive where that happens rather than pretending it cannot.
        """
        S = self.sector_lim.shape[1]
        for i, rl in enumerate(racelines[:self.sim.track.T]):
            ds = np.linalg.norm(np.roll(rl.xy, -1, 0) - rl.xy, axis=1)
            dt = ds / np.maximum(rl.v, 0.3)
            sec = np.minimum((np.arange(len(dt)) * S) // len(dt), S - 1)
            self.ideal_lap[i] = float(dt.sum())
            self.sector_lim[i] = torch.as_tensor(np.bincount(sec, weights=dt, minlength=S)[:S],
                                                 dtype=self.sector_lim.dtype, device=self.device)

    def _sector_time_reward(self, s: torch.Tensor, reward: torch.Tensor,
                            reward_components: torch.Tensor) -> None:
        """Pay for beating this track's own recent time through the sector just completed.

        Fully dense over the batch: a sector boundary is crossed several times per second by a
        thousand envs, so anything that had to ask the host which of them fired would cost a CUDA
        sync per step. Nothing here reads a value back.

        The crossing instant is interpolated from arc position rather than rounded to the control
        step. A tenth spread over twelve sectors is under a hundredth each, well below the 25 ms
        step, so integer step counts would quantise away the very signal this term exists to give.
        """
        e = self.ecfg
        S = self.sector_lim.shape[1]
        dt = self.sim.control_dt
        L = self.sim.track.length[self.sim.tid].clamp_min(1e-6)
        sec = (s / L * S).long().clamp(0, S - 1)
        moved = sec != self.sector_idx
        # only a sector left through its own exit boundary was actually driven end to end: a skip
        # (two boundaries in one step) or a slide backwards has no time worth scoring
        adv = ((sec - self.sector_idx) % S) == 1
        ds = (s - self.s_prev) % L
        past = (s - sec.to(s.dtype) * (L / S)) % L
        t_now = self.ep_step.to(s.dtype) * dt - (past / ds.clamp_min(1e-6)).clamp(0.0, 1.0) * dt
        t_sec = t_now - self.sector_t0

        flat = self.sim.tid * S + self.sector_idx              # the sector being left
        lim = self.sector_lim.view(-1)[flat]
        ref = self.sector_ref.view(-1)[flat]
        h_min = (e.reward_time_floor * lim).clamp_min(1e-3)
        # a sector cannot be driven faster than its arc length at the speed limit, so anything quicker
        # is a wobble across the boundary and not a sector at all -- and a wobble is cheap to repeat,
        # which would make the cap on the gain a per-oscillation payout rather than a safety limit.
        # 0.7 of the bound because the racing line cuts inside the centerline the sectors are cut on.
        t_min = 0.7 * (L / S) / e.v_max_policy
        scored = adv & self.sector_valid & (t_sec >= t_min)
        gain = torch.where(scored & (ref > 0),
                           torch.log((ref - lim).clamp_min(h_min) / (t_sec - lim).clamp_min(h_min)),
                           torch.zeros_like(t_sec)).clamp(0.0, e.reward_time_max)
        reward += e.reward_lap_time * gain
        reward_components[:, REWARD_COMPONENT_KEYS.index("lap_time")] = e.reward_lap_time * gain

        # EMA per (track, sector). Many envs finish the same sector on the same step, so their times
        # are averaged and the decay applied once -- stepping it per env would move a busy sector
        # much further than a quiet one for no reason but how many cars happened to be on it.
        w = scored.to(t_sec.dtype)
        flat_ref = self.sector_ref.view(-1)
        total = torch.zeros_like(flat_ref).index_add_(0, flat, t_sec * w)
        count = torch.zeros_like(flat_ref).index_add_(0, flat, w)
        m = e.reward_time_momentum
        mean = total / count.clamp_min(1.0)
        flat_lim = self.sector_lim.view(-1)
        floor = flat_lim + (e.reward_time_floor * flat_lim).clamp_min(1e-3)
        # a sector's first clean time has no reference yet: seed it rather than decaying from zero
        upd = torch.where(flat_ref > 0, m * flat_ref + (1 - m) * mean, mean).clamp_min(floor)
        self.sector_ref = torch.where(count > 0, upd, flat_ref).view_as(self.sector_ref)

        self.sector_t0 = torch.where(moved, t_now, self.sector_t0)
        self.sector_valid |= moved
        self.sector_idx = sec
        self.s_prev = s.clone()

    def plan_world(self, i: int, predicted: bool = False):
        """Env i's current plan (or the tracker's predicted motion) as world (K,3) x, y, speed for viewers."""
        if self.tracker is None or self.tracker.last_ref is None:
            return None
        src = self.tracker.last_pred if predicted else self.tracker.last_ref
        if src is None:
            return None
        ref = src[i].cpu().numpy()
        x, y, yaw = self.sim.state[i, :3].tolist()
        c, s_ = math.cos(yaw), math.sin(yaw)
        return np.stack([x + ref[:, 0] * c - ref[:, 1] * s_, y + ref[:, 0] * s_ + ref[:, 1] * c, ref[:, 3]], 1)
