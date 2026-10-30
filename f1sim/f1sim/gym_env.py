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
from typing import Any, Dict, Optional

import math

import numpy as np
import torch

from .mpc import ACT_DIM as PLAN_DIM, PlanSpec, PlanTracker, act_dim as plan_act_dim, encode as plan_encode, decode as plan_decode
from .mpc import plan_layout as _plan_layout
from .opponent_events import (NO_EVENT, LearnerView, OpponentEvents, raceline_corners,
                               raceline_offset_limit_sided, split_events)
from .learn.obs import ATT_SCALE, MessageInputs, norm_att, norm_imu, norm_scan, norm_speed
from .learn.obs import (EGO_ACTIONS_DEFAULT, EGO_SENSOR_KEYS, OBS_VERSIONS, V2_MAX_FRAMES, assemble_ego_v2,
                        norm_gyro_z, v2_ego_dim, v2_ring_len)
from .opp_token import (OPP_TOKEN_CARS, OPP_TOKEN_HORIZONS_S, OPP_TOKEN_PLAN_SOURCE,
                        OPP_TOKEN_SCALE, opp_token_car_dim, opp_token_dim, sample_body_traj,
                        straight_traj, to_ego, to_world, validate_opp_token)
from .params import Config
from .sim import Simulator, StepResult
from .track import Track

REWARD_COMPONENT_KEYS = ("progress", "collision", "collision_speed", "steer_rate", "proximity",
                         "plan_clearance", "wrong_way", "lap", "alive", "car_contact", "overtake",
                         "lap_time", "car_proximity", "sideslip", "ttc", "overtake_hold",
                         "grip_budget", "rear_end", "merge")

#: [s] the time to contact reported for a pair of cars that is not closing. Finite rather than an
#: infinity so the penalty stays NaN-free under `torch.compile`; far past any `ttc_safe` anyone
#: would set, so it is the same thing as "no contact coming".
TTC_FAR = 1e3

#: The nearest-opponent columns of privileged() are stored as metres (or m/s) divided by this, to
#: keep them O(1) for the critic. Anything comparing them against a real distance must multiply.
PRIV_OPP_DIST_SCALE = 5.0

# ---- contract v2's critic vector (`F1VecEnv.privileged_v2`) -----------------------------------------
#: The redesign's B9 as corrected by the 10-14 gates (architecture-redesign appendix 2): ego 8, three
#: opponents x 7 and the explicit geometry the information-gap gate measured (+0.049 AUC and R over
#: the same critic without it, CI excluding 0) -- 33 columns. Every column is divided by a fixed
#: scale so all of them are O(1) -- v1's progress rate reached 56 unnormalised. The dial is NOT here:
#: the trainer owns it (`conditioning.DialDraw`) and hands the critic the same `c` column the actor
#: reads.
#:
#: No speed-cap column (removed 2026-10-14). It was there for the learner's unobserved random cap
#: (critique B5), a guess no measurement backed, which is gone; the cap left is the env's own -- the
#: fixed `--speed-cap` in DAgger, the curriculum's in PPO, one number for every learner row of a
#: batch -- so the column was a constant. (A self-play front car's cap, `selfplay_front_cap`, and a
#: slot table's cap on a `self` seat still differ per row; the critic does not see them.)
#:
#: v1's `privileged()` is untouched (21 wide in a race, 22 with the dial): every v1 critic keeps it.
PRIV_V2_EGO_KEYS = ("vx", "vy", "yaw_rate", "lateral", "heading_err", "lap_frac", "progress_rate",
                    "wall_dist")
#: Per opponent, nearest first by centre distance, absent seats all zero -- the layout the G4 gate
#: collected and trained on (`present, dx/5, dy/5, dist/5, dvx/5, dvy/5, dyaw`).
PRIV_V2_OPP_KEYS = ("present", "dx", "dy", "dist", "dvx", "dvy", "dyaw")
PRIV_V2_OPP_SLOTS = 3
#: The align gate's 'explicit geometry': the nearest present opponent's centre distance, its closing
#: speed and time to contact, and the minimum time to contact over present opponents.
PRIV_V2_GEO_KEYS = ("near_dist", "near_closing", "near_ttc", "min_ttc")
PRIV_V2_KEYS = (PRIV_V2_EGO_KEYS
                + tuple(f"opp{j}_{k}" for j in range(PRIV_V2_OPP_SLOTS) for k in PRIV_V2_OPP_KEYS)
                + PRIV_V2_GEO_KEYS)
PRIV_V2_DIM = len(PRIV_V2_KEYS)
#: `EnvConfig.priv_props` (rl-audit 2026-10-28): the props the car is heading into, for the critic only. The nearest
#: live props not behind the car (body-frame dx > -1 m) within `PRIV_V2_PROP_RANGE`, nearest first, absent slots all
#: zero: present, dx/5, dy/5, dist/5 -- the opponent block's units. Appended after `PRIV_V2_KEYS`; off, the vector
#: is exactly `PRIV_V2_DIM` wide as before.
PRIV_V2_PROP_KEYS = ("present", "dx", "dy", "dist")
PRIV_V2_PROP_SLOTS = 3
PRIV_V2_PROP_DIM = PRIV_V2_PROP_SLOTS * len(PRIV_V2_PROP_KEYS)
PRIV_V2_PROP_RANGE = 12.0
#: `EnvConfig.priv_time` (2026-10-28): the share of this episode's time still to run, (limit - step) / max_steps --
#: what a critic needs to value the rest of a fixed-length race (`ppo --race-end-terminal`). Appended last.
PRIV_V2_TIME_DIM = 1
#: [s] TTC reported when nothing closes faster than `PRIV_V2_TTC_MIN_CLOSING`, and its cap. The
#: align gate's numbers; with no opponent at all: distance 30 m, closing 0, TTC 10, min TTC 10.
PRIV_V2_TTC_CAP = 10.0
PRIV_V2_TTC_MIN_CLOSING = 0.05
PRIV_V2_NO_OPP_DIST = 30.0
#: Each column's sign under a left-right mirror of the world (for the training unit's augmentation).
PRIV_V2_MIRROR = tuple(
    {"vy": -1.0, "yaw_rate": -1.0, "lateral": -1.0, "heading_err": -1.0}.get(k, 1.0)
    for k in PRIV_V2_EGO_KEYS) + tuple(
    {"dy": -1.0, "dvy": -1.0, "dyaw": -1.0}.get(k, 1.0)
    for _j in range(PRIV_V2_OPP_SLOTS) for k in PRIV_V2_OPP_KEYS) + (1.0,) * len(PRIV_V2_GEO_KEYS)

# ---- per-seat visibility for the v2 opponent-slot auxiliary (B8) -----------------------------------
#: [m] a seat is a label candidate within this distance of the LiDAR (requirement P2: the positions of
#: up to three cars within 6 m).
OPP_LABEL_RANGE_M = 6.0
#: A seat is SEEN on a step when at least this many of the newest scan's car beams land on it -- the
#: rule G1, align and the perception gate all used (`opponent_visible` uses it for the nearest car).
OPP_SEEN_BEAMS = 3
#: [steps] a sighting stays a label this long (3 s at 40 Hz): occlusions last p50 0.68 s, p90 1.71 s,
#: behind the car p90 2.96 s (requirement M1).
OPP_MEMORY_STEPS = 120
#: `age` of a seat that has not been seen since its track began (episode start or respawn).
OPP_NEVER_SEEN = 1_000_000
#: [m] a car beam is attributed to the seat whose body + rear detection box (the rectangle the LiDAR
#: draws for it, `Simulator._car_boxes`) is nearest its true hit point, if it is within this. The
#: scan is motion-distorted over 25 ms (0.22 m at 9 m/s), which is what the tolerance absorbs.
OPP_ATTRIB_TOL_M = 0.30
#: A return is VALID -- one the seat attribution counts (`_attribute_car_beams`) -- when it is finite,
#: short of range_max, and the noisy range is within this of the true one (the perception gate's
#: definition).
BEAM_VALID_TOL_M = 0.30

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
    #: [m/s] the slowest speed a LEARNER may command in direct mode, <= 0 (2026-10-30): a = -1 maps here instead of to 0,
    #: so a negative value lets the policy back out of a wedge itself instead of an outside recovery rule; opponents keep
    #: the [0, v_max_policy] map. 0 = as before.
    v_min_policy: float = 0.0
    #: Reverse BELOW the usual range (2026-10-30), for a learner in direct mode: its speed action may go down to
    #: -1 - reverse_ext, and a in [-1 - reverse_ext, -1) commands -reverse_speed * (-(a + 1) / reverse_ext); a in [-1, 1]
    #: keeps exactly the [0, v_max_policy] map, so a policy trained without it drives the same and can learn to back out
    #: of a wedge itself (the outside recovery rule re-wedged cars on the same prop). 0 = as before.
    reverse_ext: float = 0.0
    reverse_speed: float = 2.0
    max_steps: int = 1600            # 40 s at 40 Hz
    # Give each race's FIRST episode after `reset()` a random length in [1, max_steps] instead of
    # max_steps; every later episode is full length. Under `collision_mode="soft"` nothing
    # terminates, so without this every car in the batch spawned on the same step and hit the time
    # limit on the same step, forever: spec_korea_contact_s911 reset all 256 at once every 50
    # updates, and the 10 updates after each one were rollouts of nothing but starts (reward -0.047
    # per step against +0.15 for the other 40). Under `terminate` the crashes spread the resets out
    # by themselves. Training only: an evaluation that starts every car at once wants full episodes.
    stagger_first_episode: bool = False
    #: Probability that a whole-race reset starts from a scene recorded `replay_lag_s` before one of
    #: the learner's car contacts, instead of the grid (two-car races only). GT Sophy's "mistake
    #: learning": the moments the policy gets wrong are rare in a rollout, and restarting just before
    #: them is the cheapest way to make them common. 0 builds nothing: every run before 2026-09-27.
    #: The two ways the learner makes most of its car contacts (docs/research, 2026-09-27: 25 % rear-ending
    #: a car its own plan runs through, 34 % cutting back in right after a pass, mostly onto a car in the
    #: LiDAR's rear blind zone). `reward_rear_end` charges the onset of a contact with a car ahead by the
    #: squared relative speed (GT Sophy's rear-end term); `reward_merge` charges, per metre, the learner's
    #: own lateral motion toward a car that is level with it or just behind while the two are within a
    #: metre sideways. Both read the simulator's truth; the policy has to learn to satisfy them from what
    #: it sees and remembers. 0 = off: every run before 2026-09-27.
    reward_rear_end: float = 0.0
    reward_merge: float = 0.0
    #: Multiplicative safety on the progress reward (robotics practice; the F1TENTH overtaking RL of 2025
    #: scales progress by (1 - proximity) the same way): progress x g, g = 1 - (1 - gate_min) x danger,
    #: danger rising linearly from 0 at a body gap of gate_gap + closing x gate_tau to 1 at contact,
    #: closing measured along the line of sight -- so running up on a car ahead and sliding into one
    #: alongside both shrink it. Summed penalties let the policy trade one failure for another (merge
    #: penalty x3: cut-backs 25 -> 16, rear-ends 10 -> 17); a product leaves nothing to trade.
    #: 0 = off: every run before 2026-09-27.
    reward_progress_gate: float = 0.0          # 1.0 = on (a flag with room for a strength)
    gate_min: float = 0.3
    gate_gap: float = 0.3
    gate_tau: float = 0.5
    replay_contacts: float = 0.0
    replay_lag_s: float = 0.75
    replay_bank: int = 1024
    # How a `kind_mix` slot picks its driver. "redraw": uniformly at every race's reset, so one race
    # meets every kind over time -- what the console shows. "partition": once, by race number
    # (race g drives `kind_mix[g % len]`), fixed for the life of the env -- what training wants: the
    # batch holds every kind in exact proportion at every update, and each kind then owns a fixed
    # quarter of the rows, so a teacher that can plan a subset (`plan_rows`) is asked for those
    # rows only. s911's interactive opponent planned the whole batch to drive an eighth of it.
    kind_mix_assign: str = "redraw"
    laps: int = 100                  # truncate after this many laps (racing: keep going)
    reward_progress: float = 1.0     # per meter
    reward_collision: float = -10.0
    reward_steer_rate: float = 0.05  # per unit of normalized steer change
    reward_proximity: float = 0.5    # per metre driven, at zero wall gap
    proximity_speed_ref: float = 0.0 # [m/s] >0: the proximity penalty is scaled by (1 + v / ref): fast past a wall costs more than creeping
    reward_wrong_way: float = 0.2    # per step while facing backwards along the lane (progress is signed anyway; this makes it explicit)
    reward_collision_speed: float = 0.0   # extra collision penalty per m/s of speed at impact (a fast crash costs more than a nudge)
    # What a collision *is*. "terminate" ends the episode, which is every run before this existed
    # and is what every benchmark number on record was measured under.
    #
    # "soft" does not: the simulator already resolves the contact (`Simulator._resolve_wall_contact`
    # -- push out of penetration, kill the into-surface velocity with `collision_restitution` and
    # `wall_friction`, damp the yaw rate), the episode continues, and the crash is charged as a cost
    # instead of an ending. The whole machinery was there and unreachable, because this class forced
    # `terminate_on_collision = True` whatever the config said.
    #
    # Why it matters: with "terminate" the policy can only ever learn to *avoid*. It never sees what
    # happens after a touch, so it never learns to steer out of one, to back off and go again, or --
    # the thing that actually decides whether a recovery is worth anything -- not to end up driving
    # the wrong way (`reward_wrong_way`). A real race has contact in it.
    #
    # It is an option and not the default on purpose: it changes what "collisions per km" counts,
    # so a run under it is not comparable with one under "terminate" without saying so.
    collision_mode: str = "terminate"
    # What the `collision` reward component (and so `--lagrange wall=`) counts under "soft". False: walls and
    # props only, as `--lagrange` documents. True: also every car contact, which is how every run before
    # 2026-10-13 was trained -- the simulator ORs the car flag into its collision flag, so a car contact was
    # charged at the wall's multiplier on top of the car's (the 10-13 audit: 27-52 % of the training "wall"
    # rate was cars). Turning the double charge off halves a car contact's price unless the car multiplier
    # takes over the wall's share.
    collision_counts_cars: bool = False
    # Whether a car contact's charge is decided once, when it begins. The rule (`car_contact_charge`) exempts
    # the car that was clearly in front; read every step, cars that pass through each other (soft mode has no
    # car-car response) swap the order within 1-19 steps and the exempt car is charged anyway (96/96 exempt
    # at the first step, then charged). True holds the first step's verdict for the whole contact.
    car_charge_latch: bool = True
    # [m/s] The impact severity a contact onset needs to be charged (`f1sim.contact_severity`): the speed of
    # the two surfaces into each other along the contact normal, at the contact point. 0 charges every
    # onset, which is every run before this existed. Above 0 an onset slower than this is free -- for the
    # `collision` component (and `collision_speed`, which rides on the same onset), so for `--lagrange
    # wall=`, and for the car-contact charge, so for `--lagrange car=` -- and one at or above it is charged
    # exactly as before. The goal it serves (2026-10-15): fast with the odd graze is good, hard hits are
    # not. Soft contact only: under terminate every contact ends the episode whatever it cost.
    contact_severity_min: float = 0.0
    # Whether a struck obstacle is shoved or is a wall with a crate's shape. Off is every run
    # before this existed. On, each prop carries the mass of what it is
    # (`procedural_obstacles.STYLE_MASS`: a cardboard box 1.2 kg, a wooden crate 10, a steel drum
    # 18, against a 3.74 kg car) and a contact exchanges momentum both ways -- the car keeps
    # m_p/(m_c+m_p) of the bounce and the prop slides off with the rest until the floor stops it.
    #
    # Needs `collision_mode = "soft"` to mean anything: under "terminate" the episode ends at the
    # touch and nothing after it is ever seen.
    movable_obstacles: bool = False
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
    reward_grip_budget: float = 0.0   # per second, per (m/s^2)^2 of lateral acceleration beyond the budget the car was
                                      # GIVEN (`set_grip_budget`, a friction in mu units -- the policy's grip dial).
                                      # Without it a dial is only a hint under RL: the floor has at least that much
                                      # grip, so the return-maximising policy learns to drive faster than it was told.
                                      # With it the dial is a command, and what is optimised is the lap inside it.
    grip_budget_rho: float = 0.95     # share of the front axle's capacity at that friction the budget allows
    reward_alive: float = 0.0
    # Time to contact with the nearest car, per step. The reward audit
    # (`docs/research/reward-audit-2026-09-13.md`) found that nothing prices a *margin* to another
    # car until the margin becomes a contact: `car_proximity` is charged per metre driven and scaled
    # by closing speed, so a car standing still next to another pays nothing, and a car closing at
    # 3 m/s from 2 m away -- 0.7 s from contact -- pays nothing either, because at 2 m the gap is
    # outside `car_safe_gap`. TTC is the quantity that sees both: it is small exactly when the
    # geometry and the closing speed together say a contact is imminent, whatever the current gap.
    # The attribution note measured the failure this is aimed at: contacts are side-by-side at
    # +1.5 m/s of closing speed, i.e. the learner drives level with a car and keeps closing.
    reward_ttc: float = 0.0           # per step, at zero time to contact
    ttc_safe: float = 1.0             # [s] time to contact above which the term is 0; it ramps
                                      # linearly to `reward_ttc` at TTC = 0
    # Paid once, when a lead has been *held*. `reward_overtake` pays for arc taken out of the field
    # every step, which pays the approach and pays the crossing instant; measured at +0.03/s at
    # coefficient 1.0 it is 1 % of progress, and what it buys at 5.0 is a faster approach rather
    # than a completed pass. This term pays nothing for closing and nothing at the crossing: it
    # fires when the learner has been `overtake_hold_dist` metres ahead of an opponent along the
    # lane, continuously, for `overtake_hold_time` seconds -- which is a pass that stuck.
    reward_overtake_hold: float = 0.0 # paid once per held lead, per opponent
    overtake_hold_dist: float = 1.5   # [m] signed arc the learner must be ahead by. Two car lengths
                                      # and a bit: inside that the cars are still side by side and
                                      # the "pass" can be undone by the other car simply continuing.
    overtake_hold_time: float = 1.0   # [s] how long it must hold, continuously
    # Privileged opponent tokens in the observation (`f1sim.opp_token`). An ORACLE: "off" allocates
    # nothing and is bit-identical to the env without it; anything else appends a proprio block the
    # real car cannot produce, and the checkpoint it trains is refused by the exporter and the ROS
    # node. Its whole purpose is to answer whether the planner, *handed* the opponent's state, races
    # any better than it does with LiDAR alone.
    opp_token: str = "off"
    opp_token_ablate: bool = False    # emit the block as ZEROS while keeping its width. A diagnostic
                                      # and never a deployment path: it measures how much a policy
                                      # trained on the block actually depends on it, which is the
                                      # difference between "the planner ignored the oracle" and "the
                                      # planner used it and it did not pay" -- two readings of the
                                      # same flat result that call for opposite next steps. Refused
                                      # unless `opp_token` is on, because zeroing a block that does
                                      # not exist is not an ablation, it is the control arm.
    spawn_lateral_std: float = 0.3
    spawn_yaw_std: float = 0.2
    spawn_min_clearance: float = 0.5  # [m] spawn poses closer to a wall are pulled back to the centerline
    # [m] of clear road every spawning car is guaranteed ahead of it -- no prop, no wall inside it.
    # 0 is off, which is every run before this existed. A car is placed at up to `spawn_speed_max`
    # and cannot stop in less than v^2/2a plus its own length, so a crate inside that distance is a
    # collision the policy was never given the chance to avoid; it lands in the collision rate all
    # the same and makes the metric read worse the more interesting the layout is.
    # `procedural_raceline_corridor = "off"` requires a positive value, because that is the setting
    # that puts something worth avoiding directly in front of a spawn.
    spawn_runway: float = 0.0
    spawn_speed_max: float = 3.0     # random initial speed in [0, this]
    resample_track_on_reset: bool = True   # multi-track sets: pick a random track for each new episode
    scan_stride: int = 1             # frames between stacked scans (3 x stride 3 = 225 ms of history: velocity cues)
    hist_len: int = 0                # >0: proprioceptive history rows (speed, imu, roll/pitch, action) in the observation
    hist_stride: int = 2             # steps between rows (20 x 2 = the last second): implicit identification of grip / lag
    #: Observation contract (`learn.obs.OBS_VERSIONS`). 1 is every run before 2026-10-14 and the
    #: default leaves it bit-identical (`tests/test_v1_bit_identity.py`). 2 is `learn.obs.ObsSpecV2`:
    #: {"scan": the newest `scan_stack` (1..3) frames, "ego": [speed, gyro z] now + the action issued
    #: two steps back + `hist_len` rows of [speed, gyro z, the issued action] `hist_stride` steps
    #: apart} -- no speed cap, attitude, accelerometer, gyro x/y or prev_action key. Plan mode only;
    #: the IMU must be on.
    obs_version: int = 1
    #: Contract v2 only (`learn.obs.ObsSpecV2.ego_actions`): whether the ego vector carries the
    #: actions the policy issued (a_{t-2} and every history row's plan). False is the copycat study's
    #: arm A -- [speed, gyro z] now and the history's [speed, gyro z], 42 columns -- because the issued
    #: actions were measured to be what v2 DAgger's student copied (docs/research/copycat-2026-10-14.md).
    #: The history ring records the actions either way; only the assembly reads less.
    ego_actions: bool = EGO_ACTIONS_DEFAULT
    #: Per-opponent LiDAR visibility for the v2 opponent-slot auxiliary (B8) and the nearest-visible-
    #: car head: each step, the car beams of the newest scan attributed to each seat and the steps
    #: since each seat was last seen (>= 3 beams within 6 m of the LiDAR). Off computes nothing.
    opp_visibility: bool = False
    action_mode: str = "direct"      # "direct": (steer, speed); "plan": short local trajectory tracked by an MPC (f1sim.mpc)
    speed_mode: str = "linear"       # plan only, `mpc.SPEED_MODES`: what the plan's speed dimensions mean. "linear" is
                                     # every existing checkpoint's; "envelope" / "knots" are opt-in experiments
    #: Plan only: the format the cars this env's learner does NOT drive plan in -- teacher-driven seats
    #: (the raceline teacher and every OpponentPlanner kind) and league checkpoints -- when it is not
    #: `speed_mode` (output-format design 6, migration step 6). None: every car plans in `speed_mode`,
    #: as in every env before 2026-10-14. "linear" beside speed_mode "wp4" is the migration's race: the
    #: scripted cars keep the plan they were built for and the league keeps the format it was trained
    #: in, the learner's rows (and the labels a teacher writes for them) are wp4, and one tracker reads
    #: each row in its own format with one iLQR solve (`PlanTracker.row_format`). Without it the wp4
    #: label gates raced wp4 opponents, so a wp4-against-linear comparison was one world against
    #: another rather than one label against another (docs/research/wp4-label-gates-2026-10-14/).
    opponent_speed_mode: Optional[str] = None
    plan_a_brake: float = 4.0        # [m/s^2] "envelope" only: braking its backward pass plans with (match the teacher's)
    #: What a curvature knot of +-1 means (`mpc.PlanSpec.kappa_mode`): "absolute" is +-1.6 1/m at any
    #: speed, "feasible" is the tightest arc the tyres can hold at the speed the plan starts from.
    #: A checkpoint trained under one and driven under the other is a different policy -- the console
    #: and the evaluation read the mode the run recorded.
    plan_kappa_mode: str = "absolute"
    #: Speed feedforward between the plan tracker and the VESC. The VESC loop is first order
    #: (`actuators.vesc_accel`: accel = (target - speed) / motor_tau), and the tracker commands the
    #: speed it predicts 0.15 s ahead as if it were reached -- so the car accelerates in proportion
    #: to how far that prediction leads: measured, 3 m/s^2 per 1 m/s of lead, and the command led by
    #: under 0.5 m/s most of the time against the 2 m/s a 6.5 m/s^2 profile needs. On: the command
    #: is the loop's inverse, predicted speed + calibrated motor time constant x planned acceleration. Off (the
    #: default) is every run before 2026-09-24, bit for bit. Measured with the line teacher on ICCAS
    #: (friction pinned): lap 7.34 s either way. The command follows the tracker's plan, and the
    #: tracker itself plans only ~2.5 m/s^2 while its reference leads the car, so the lead it adds is
    #: small; see docs/research/layout-line-teacher-2026-09-24.md.
    tracker_speed_ff: bool = False
    #: Who sets the speed command (2026-09-28). "tracker": the plan tracker's predicted speed, as
    #: every run before. "profile": the speed the plan's own profile asks for, sent straight to the
    #: VESC -- the reference speed where the command lands plus the calibrated motor time constant
    #: times the reference's acceleration (the inverse of the VESC's first-order loop). The tracker
    #: still steers along a reference walked at that profile. Measured on the refitted car: the
    #: tracker plans only ~2.5 m/s^2 and holds even the line teacher at ~75 % of its profile; with
    #: the profile commanded directly the teacher laps ICCAS 4 % quicker. Meant for the profile
    #: speed modes ("envelope", "knots"), whose profile is what the policy actually decides.
    speed_command: str = "tracker"
    #: [s] "profile" only: first-order low-pass on the profile's speed command. A profile is built
    #: afresh every 25 ms from the plan's own curvature, and v = sqrt(a / kappa) turns a small wobble
    #: in a student's curvature into a large one in speed; the tracker used to absorb it. Measured on
    #: the first envelope/profile DAgger student (ICCAS): step-to-step command change median 0.16 m/s
    #: (p90 0.77) against 0.06 (0.17) through the tracker. 0 is no filter.
    speed_command_tau: float = 0.0
    #: The plan tracker's speed-tracking weight and drive bound, when not None (2026-09-28). The
    #: defaults (0.4, 6.0 m/s^2) plan ~2.5 m/s^2 while the reference leads the car; the line teacher
    #: on ICCAS goes 8.618 -> 8.400 s with (2.0, 7.0) under the 'budget' output. None: unchanged.
    tracker_speed_weight: Optional[float] = None
    tracker_a_max: Optional[float] = None
    #: `PlanSpec.profile_from_prediction` (profile speed modes): the profile ramps from the speed the
    #: tracker predicts, not the measured one. Off: the executor every checkpoint before 2026-09-30 had.
    tracker_profile_from_prediction: bool = False
    plan_kappa_a_lat: float = 8.0    # [m/s^2] "feasible" only: the lateral budget it scales by
    compile_tracker: bool = True
    #: Who follows the policy's plan (2026-10-19, the user: "우리도 그냥 map 쓰자"). "mpc": the iLQR plan
    #: tracker, every run before. "map": ForzaETH's MAP controller on the plan's own waypoints
    #: (`map_controller.MAPPlanTracker`), for the policy's cars only -- the traffic keeps the iLQR.
    plan_follower: str = "mpc"
    #: `map_controller.MAPParams` fields for "map", over `map_controller.POLICY_MAP` (the paper's
    #: formulation; delay_comp -1 compensates each car's calibrated delay, the one the iLQR is given).
    plan_follower_map: Optional[Dict[str, Any]] = None
    #: The learner's cars decide every this many control steps (2026-10-27, RL research). One 25 ms decision
    #: moves the car by about sigma * dt (0.8 m/s * 25 ms = 2 cm), so its advantage drowns: measured with the
    #: actor frozen, the PPO gradient for "drive faster" had an SNR of 0.1-0.3 an update and no GAE lambda
    #: lifted it (`tools/research/rl_speed`). Between decisions the last plan is HELD ON THE TRACK -- MAP
    #: follows the waypoints of the plan as it was issued, in the frame of the pose it was issued from, as a
    #: 10 Hz planner and a 40 Hz controller on a real stack -- while the scan, the IMU and the policy's memory
    #: still run every step. The traffic keeps deciding every step. 1: every step (every run before). MAP only.
    decision_every: int = 1
    #: The venue's speed knob (2026-10-20, the user: the floor's friction is the console's to set and the
    #: controller's to answer with a speed-profile scale, like race_stack's per-track speed scaling): the
    #: speeds of the policy's plans are multiplied by this before any follower (iLQR or MAP) sees them. The
    #: policy's own action -- its history, its log-prob -- is untouched; only what is tracked is slower.
    #: Same grip fraction on a floor of k x the friction at about sqrt(k). 'linear' plans only.
    plan_speed_scale: float = 1.0
    #: Append the props ahead to the v2 critic's privileged vector (`PRIV_V2_PROP_KEYS`). Critic-only, never an
    #: observation. Why (rl-audit 2026-10-28): the critic saw every opponent but no prop except through the scan and
    #: the current clearance, and pre-priced ~30 % of a hard wall/prop hit before it (cars ~45 %); the rest of the
    #: price lands on the onset step, where GAE cannot carry it back to the line choice 1-2 s earlier. Off = as before.
    priv_props: bool = False
    #: Append the episode time still to run to the v2 critic's privileged vector (`PRIV_V2_TIME_DIM`). Critic-only.
    #: Why (2026-10-28, the user: a race is 1 vs 1 over ~10 laps and is won by finishing them sooner): with the
    #: episode = one race and its end a true end (`ppo --race-end-terminal`), the value of a state depends on how
    #: much race is left, which nothing else in the vector says. Off = as before.
    priv_time: bool = False
    #: The speed limit for a car ahead (2026-10-25, `opponent_speed_limit.py`): the learner's speed command is held to
    #: the speed from which it can still slow to the car's speed before the gap closes, for a car in its lane now or
    #: moving into it within 1 s -- read off the policy's own nearest-car output, which its runner writes into
    #: `F1VecEnv.opponent_reading` every step (`learn/common.student_policy`, the console's worker). Plan mode only.
    opponent_speed_limit: bool = False
    #: The speed limit for an obstacle on the plan's own path (2026-10-26, `obstacle_speed_limit.py`): the learner's speed
    #: command is held to the speed from which it can still stop before the first small isolated cluster of its last
    #: scan that lies on the path its plan draws -- a box, a pole; walls are long and left out, so no map is needed.
    #: Returns at the nearest-car reading are left to `opponent_speed_limit`, so this one needs the policy's nearest-car
    #: output too (`F1VecEnv.opponent_reading`, written by the same runners), with or without the car limit on: without
    #: it a car on the path would be held to as if it stood still (2026-10-26). Plan mode only.
    obstacle_speed_limit: bool = False
    #: [s] from a held speed command to the brakes biting, as both speed limits assume it (2026-10-26). 0.2; with the
    #: competition car's slower speed loop simulated (`ActuatorParams.motor_delay` 0.1), 0.3 did no better over 63 km.
    speed_limit_lag: float = 0.2
    #: The label teachers' wall check on the raceline covers the plan's own path, not the preview's run-on past its end
    #: (`InteractiveTeacher.walls_within_plan`): the guard `teacher_label` puts over a raceline teacher (x1.0 / 0.8 /
    #: 0.55 / 0.3 of its plan's speeds) and DAgger's / evaluate's interactive teacher. 2026-10-26: that run-on raised
    #: false wall alarms at full pace and cost the Korea teacher 0.4 s a lap under MAP with no contact avoided. Props,
    #: cars, offset lines and evasions keep the whole preview. False: what every result before 2026-10-26 used. The
    #: interactive OPPONENT (`opponent_slots`) keeps the old check either way: it is a benchmark car.
    teacher_walls_within_plan: bool = True
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
    #: [m] body-to-body lateral gap an abreast row is laid out with, where the lane has room for it.
    #: Where two bodies plus this gap do not fit, `spawn_alongside_sep_min` is tried, and where that
    #: does not fit either the race spawns staggered, because two cars spawned in contact terminate
    #: on step 1 for ever. Before 2026-10-14 the gap was 0.15 m, and in the boot-camp config an
    #: opponent steering for its line from there reached the learner in 0.1-0.5 s (22 of G4's 122
    #: early contacts, all charged to the learner, whose own lateral speed was under 0.11 m/s).
    #: 0.15 (with the minimum at 0.15 or below): the grid every run before 2026-10-14 had.
    spawn_alongside_sep: float = 0.30
    #: [m] the narrowest body-to-body gap an abreast row may take: used only for a race whose lane
    #: has no room for `spawn_alongside_sep`, so a narrow section still gets a side-by-side start
    #: instead of a stagger. At or above `spawn_alongside_sep`: no fallback, one gap only.
    spawn_alongside_sep_min: float = 0.15
    # A rolling start for teacher-driven opponents placed abreast -- hold the lane position they were
    # placed at for 1 s, then ease onto their line over 1 s -- was built, fixed and measured on
    # 2026-10-14 and removed (docs/research/rolling-start-2026-10-14.md; the code is in 95b069f).
    # Pre-registered test, boot-camp config with every seat abreast, 176 learner episodes an arm, same
    # seeds: contacts with the abreast partner in the first 2 s went 32 -> 25 raw (30 -> 22 charged),
    # against the >= 50 % drop it had to show. The hold held -- motion toward the learner at 1 s
    # median +0.01 m against +0.23 -- and the merge happened in the ramp instead (1-2 s: 17 against
    # 18): it delayed the merge, it did not remove it.
    #: Lay a race out as one: a rejected car walks the whole race along the lap with it, an abreast
    #: row is never pulled onto one line, empty seats neither block nor get tested, a car rejoining
    #: a race keeps `spawn_gap[0]` of arc from every race mate (see `Simulator.sample_spawn`).
    #: False: the per-car placement before 2026-10-14, kept only to reproduce old tables -- in the
    #: boot-camp config 74 learner pairs in 800 races overlapped at placement (about 9 % of learner
    #: episodes), and in a 10-s drive 12 of 160 learner episodes began with a contact on step 0, all
    #: charged to the learner (0 of 158 laid out as one).
    spawn_as_race: bool = True
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
    # Whether that corridor is enforced at all. "on" is every run before this field existed, and
    # what a raceline-driven opponent needs: it is pure pursuit on a line built from the grid, and
    # the props are not in the grid, so a crate on the line is a crash it cannot see coming.
    #
    # "off" lets a layout stand anywhere, including across the racing line -- which is the only way
    # the policy ever meets an obstacle there, and it does not today. The env refuses "off" unless
    # every teacher-driven car is a kind that can see props (`opponent_slots.DriverKind.prop_aware`),
    # because the alternative is a baseline whose collisions are the layout's fault and a result
    # nobody can read.
    procedural_raceline_corridor: str = "on"
    # One layout for the whole batch instead of one per env (or per race). The console runs this:
    # it draws one track, and every car on it has to be driving through the crates that are drawn.
    # The layout is redrawn only when every env resets together -- the console's 리셋, a new
    # session -- and a car that resets alone keeps it, as a car respawning mid-race keeps its race's.
    procedural_shared: bool = False


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


class _ContactReplay:
    """Records two-car scenes shortly before the learner's car contacts and hands them back as
    spawn poses (see `EnvConfig.replay_contacts`). Poses and speeds only: a replayed race starts
    with fresh planner memory, a fresh opponent speed draw and the layout of its own reset, so what
    it repeats is the geometry of the moment, not the moment."""

    def __init__(self, B: int, e, dt: float, device):
        self.lag = max(1, int(round(e.replay_lag_s / dt)))
        self.D = self.lag + 1
        self.hist = torch.zeros(self.D, B, 4, device=device)          # x, y, yaw, speed per step
        self.ptr = 0
        self.filled = torch.zeros(B, dtype=torch.long, device=device)  # steps recorded since the car's reset
        self.prev = torch.zeros(B, dtype=torch.bool, device=device)
        C = int(e.replay_bank)
        self.bank = torch.zeros(C, 10, device=device)                 # tid, kind, learner x y yaw v, opp x y yaw v
        self.n = 0; self.head = 0; self.C = C
        self.p = float(e.replay_contacts)
        self.recorded = 0; self.replayed = 0

    def record(self, env, car_hit: torch.Tensor) -> None:
        st = env.sim.state
        self.hist[self.ptr] = torch.stack([st[:, 0], st[:, 1], st[:, 2], st[:, 3]], 1)
        self.ptr = (self.ptr + 1) % self.D
        fresh = env.ep_step <= 1
        self.filled = torch.where(fresh, torch.zeros_like(self.filled), self.filled + 1)
        onset = car_hit & ~self.prev & env.learner & (self.filled > self.lag)
        self.prev = car_hit.clone()
        rows = onset.nonzero().flatten()
        if rows.numel() == 0:                                         # host sync only on the rare step
            return
        old = self.hist[(self.ptr - 1 - self.lag) % self.D]           # the scene `lag` steps back
        j = env.sim.other_idx[rows, 0]
        kinds = getattr(env, "slot_kind_code", None)                  # slot mode only; one kind otherwise
        kind = kinds[j, None].float() if kinds is not None else torch.zeros_like(rows[:, None], dtype=torch.float)
        e = torch.cat([env.sim.tid[rows, None].float(), kind, old[rows], old[j]], 1)
        for k in range(e.shape[0]):
            self.bank[self.head] = e[k]; self.head = (self.head + 1) % self.C; self.n = min(self.n + 1, self.C)
        before = self.recorded
        self.recorded += int(e.shape[0])
        if self.recorded // 50 != before // 50:              # a trace that the arm is doing anything at all
            print(f"[replay] {self.recorded} scenes recorded, {self.replayed} races started from one", flush=True)

    def apply(self, env, ids, poses, speed, gen):
        """Replace the grid poses of whole-race resets, with probability p per race.

        Returns (poses, speed, replayed): `replayed` (n,) marks the rows given a recorded scene, which
        are not the grid they were laid out as -- in particular not an abreast row (`spawn_abreast`):
        their poses are the recorded moment before a contact."""
        none = torch.zeros(ids.numel(), dtype=torch.bool, device=env.device)
        if self.n == 0:
            return poses, speed, none
        in_ids = torch.zeros(env.B, dtype=torch.bool, device=env.device); in_ids[ids] = True
        full = in_ids.view(-1, 2).all(1)
        learner_pos = ((env.slot[ids] == 0) & full[env.race[ids]]).nonzero().flatten()
        if learner_pos.numel() == 0:
            return poses, speed, none
        pick = torch.rand(learner_pos.numel(), device=env.device, generator=gen) < self.p
        learner_pos = learner_pos[pick]
        if learner_pos.numel() == 0:
            return poses, speed, none
        poses, speed = poses.clone(), speed.clone()
        replayed = none.clone()
        bank = self.bank[:self.n]
        for q in learner_pos.tolist():
            row = int(ids[q]); j = int(env.sim.other_idx[row, 0])
            qo = int((ids == j).nonzero().flatten()[0])
            kinds = getattr(env, "slot_kind_code", None)
            ok = (bank[:, 0] == float(env.sim.tid[row])) & (bank[:, 1] == (float(kinds[j]) if kinds is not None else 0.0))
            cand = ok.nonzero().flatten()
            if cand.numel() == 0:
                continue
            b = bank[cand[torch.randint(cand.numel(), (1,), device=env.device, generator=gen)]][0]
            pl, po = b[2:5][None], b[6:9][None]
            tid = env.sim.tid[[row, j]]
            if bool(env.sim._spawn_in_prop(torch.cat([pl, po]), tid, torch.tensor([row, j], device=env.device)).any()):
                continue                                              # the new layout stands where the scene was
            poses[q], poses[qo] = b[2:5], b[6:9]
            speed[q], speed[qo] = b[5], b[9]
            replayed[q] = replayed[qo] = True
            self.replayed += 1
        return poses, speed, replayed


class F1VecEnv:
    # The spawn state `__init__` sets, as a class default that means "off": an env put together
    # without `__init__` (a test shell, `F1VecEnv.__new__` plus the attributes the code under test
    # reads) reaches `signed_gaps`, and died there with an AttributeError when this existed only as an
    # instance attribute.
    #: whether `signed_gaps` masks empty seats (any seat with `present` < 1)
    _present_mask_gaps = False

    def __init__(self, track, cfg: Optional[Config] = None, env_cfg: Optional[EnvConfig] = None,
                 num_envs: int = 1024, device: Optional[str] = None):
        """track: a Track or a list of Tracks (envs are spread over them, re-drawn on reset)."""
        self.cfg = cfg or Config()
        self.ecfg = env_cfg or EnvConfig()
        if self.ecfg.collision_mode not in ("terminate", "soft"):
            raise ValueError(f"collision_mode must be 'terminate' or 'soft', not "
                             f"{self.ecfg.collision_mode!r}")
        # This used to be an unconditional True, which made `Simulator`'s soft-contact path -- and
        # `SimParams.collision_restitution` / `wall_friction`, both of which document themselves as
        # "when not terminating" -- dead code that nothing could reach.
        self.cfg.sim.terminate_on_collision = self.ecfg.collision_mode == "terminate"
        if self.ecfg.movable_obstacles and self.ecfg.collision_mode != "soft":
            raise ValueError(
                "movable_obstacles needs collision_mode='soft': under 'terminate' the episode ends "
                "at the touch, so a crate that was shoved aside is a crate nothing ever saw move.")
        sev_min = float(self.ecfg.contact_severity_min)
        if not (sev_min >= 0.0 and math.isfinite(sev_min)):
            raise ValueError(f"contact_severity_min is a speed in m/s, >= 0; got {self.ecfg.contact_severity_min!r}")
        if sev_min > 0.0 and self.ecfg.collision_mode != "soft":
            raise ValueError("contact_severity_min needs collision_mode='soft': under 'terminate' every contact "
                             "ends the episode, so a free graze would still be the end of it.")
        if sev_min > 0.0 and self.ecfg.collision_counts_cars:
            raise ValueError("contact_severity_min with collision_counts_cars: the legacy double charge reads a car "
                             "contact through the wall flag; charge by severity on the split accounting only.")
        self.sim = Simulator(track, self.cfg, num_envs, device, race_size=self.ecfg.race_size)
        if sev_min > 0.0:
            # the car contacts' closing speed is measured only when somebody reads it
            self.sim.measure_impacts = True
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
        #: prioritized level replay (`learn.level_replay.LevelReplay`, one seat per race); None = off
        self.level_replay = None
        #: (B,) the level seed each row's race is playing (-1: not drawn through a level replay)
        self.level_seed = torch.full((self.B,), -1, dtype=torch.long, device=self.device)
        if e.opponent not in OPPONENT_MODES:
            raise ValueError(f"opponent {e.opponent!r} is not one of {list(OPPONENT_MODES)}")
        if e.spawn_order not in SPAWN_ORDERS:
            raise ValueError(f"spawn_order {e.spawn_order!r} is not one of {list(SPAWN_ORDERS)}")
        if e.opp_future_model not in OPP_FUTURE_MODELS:
            raise ValueError(f"opp_future_model {e.opp_future_model!r} is not one of "
                             f"{list(OPP_FUTURE_MODELS)}")
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
            #: (M,) probability each seat is filled per race (seat 0, the learner, always); None when every
            #: slot is always there -- then nothing is drawn and a seed reproduces every earlier run
            pres = [1.0] + [float(sl.present) for sl in self.slots]
            self._present_p = (None if all(p_ >= 1.0 for p_ in pres)
                               else torch.tensor(pres, device=self.device))
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
                for ck in ((sl.checkpoint,) if sl.checkpoint else ()) + tuple(sl.ckpt_mix):
                    if ck not in seen:
                        seen.add(ck); paths.append(ck)
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
        self.external_driven = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        #: Every car's command replaced by an external driver (a Python bool, set with the drivers):
        #: then the plan tracker's answer is read by no one and `step` does not solve it.
        self._all_external = False
        self.external_slot_rows: Dict[int, torch.Tensor] = {}
        self.external_drivers: Dict[int, object] = {}        # set_external_drivers()
        if self.M > 1 and e.opponent == "teacher":
            self.teacher_driven[self.slot > 0] = True
        if self.slots is not None:
            self._build_slot_tables()
        #: Which grid each race drew (`SPAWN_ORDER_ID`), so a single car respawning mid-race does
        #: not re-draw its race's order.
        self.spawn_order_race = torch.zeros(self.B, dtype=torch.long, device=self.device)
        #: (B,) whether each car was placed in an abreast row at its last reset (`_note_spawn_formation`).
        self.spawn_abreast = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._present_mask_gaps = getattr(self, "_present_p", None) is not None
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
        self.sim.spawn_runway = float(e.spawn_runway)
        self.procedural = None
        self._check_prop_aware_drivers()
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
            self.sim.movable_obstacles = bool(e.movable_obstacles)
            # One prop set per race (`SimParams.props_per_race`): every change is the race's total, given
            # to each of its cars' copies, so they all see the crate where it was pushed to.
            if self.M > 1 and bool(getattr(self.cfg.sim, "props_per_race", False)):
                self.procedural.share_m = int(self.M)
        self.opp_scale = torch.ones(self.B, device=self.device)
        #: (B,) the speed-scale draw each car is running, whichever way that car uses it (a teacher
        #: scales its profile, a policy scales its cap). One place to read "how fast was this car
        #: told to be", for the census, the facts strip and a test.
        self.slot_scale = torch.ones(self.B, device=self.device)
        # signed arc to each opponent last step (+ ahead of me, - behind), and whether it is usable.
        # A flag rather than "0 means unset": 0 is a perfectly ordinary gap -- it is the pass itself.
        self.gap_prev = torch.zeros(self.B, max(1, self.M - 1), device=self.device)
        self.gap_valid = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        # Sustained-lead bookkeeping, per (car, opponent): how many consecutive steps this row has
        # been `overtake_hold_dist` ahead of that opponent, and whether the bonus for the current
        # run of them has already been paid. Both live outside `_step_math` and are passed through
        # it, because that function is compiled and has to stay a pure map from tensors to tensors.
        self.lead_steps = torch.zeros(self.B, max(1, self.M - 1), device=self.device)
        #: 1 = "this lead may not be paid" (see `overtake_hold`). Starts blocked, so a learner that
        #: spawned ahead is not paid for its grid position.
        self.lead_paid = torch.ones(self.B, max(1, self.M - 1), device=self.device)
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
        self.act_dim = plan_act_dim(e.speed_mode) if e.action_mode == "plan" else 2
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
        #: Observation contract v2 (`EnvConfig.obs_version`, `learn.obs.ObsSpecV2`). It keeps its own
        #: history ring -- rows of [speed, gyro z, issued action] -- and none of v1's: `self.hist` is
        #: None under v2, so every v1 history path below is skipped rather than computed for nothing.
        #: Under v1 (the default) these stay None and nothing else in this class looks at them.
        self.obs_version = int(e.obs_version)
        self.hist2 = None
        self._last_feat2 = None
        self._fresh2 = None
        if self.obs_version not in OBS_VERSIONS:
            raise ValueError(f"obs_version must be one of {OBS_VERSIONS}, got {e.obs_version}")
        if self.obs_version == 2:
            self._check_obs_v2()
            self.hist = None
            # one row per step back, enough for the oldest history row and for a_{t-2}
            # (`learn.obs.v2_ring_len`: 39 rows for the contract's 20 x 2, as before a_{t-2})
            self.hist2 = torch.zeros(self.B, v2_ring_len(e.hist_len, e.hist_stride),
                                     len(EGO_SENSOR_KEYS) + self.act_dim, device=self.device)
            self._last_feat2 = torch.zeros(self.B, len(EGO_SENSOR_KEYS), device=self.device)
            #: rows whose next observation is their episode's first: `_obs_v2` fills their history
            #: from it (`ObsBuilderV2`'s rule) and clears the flag
            self._fresh2 = torch.ones(self.B, dtype=torch.bool, device=self.device)
        #: Per-seat LiDAR visibility (`EnvConfig.opp_visibility`): car beams of the newest scan on
        #: each seat of `sim.other_idx`, and steps since that seat was last seen. None when off.
        self._opp_beams = None
        self._opp_age = None
        if e.opp_visibility:
            C = max(0, self.M - 1)
            self._opp_beams = torch.zeros(self.B, C, dtype=torch.long, device=self.device)
            self._opp_age = torch.full((self.B, C), OPP_NEVER_SEEN, dtype=torch.long, device=self.device)
        #: "no prop anywhere near the plan", as a tensor so the compiled reward takes one shape.
        self._no_plan_prop = torch.full((self.B,), float("inf"), device=self.device)
        #: `EnvConfig.opponent_speed_limit` / `.obstacle_speed_limit`: each row's nearest-car reading (the policy's own
        #: output, written in place by its runner before `step`; one buffer for the session so a captured step reads the
        #: live values) -- the car limit reads the car from it, the obstacle limit leaves that car's returns to the car
        #: limit -- and where the car limit held the speed command this step. None when both are off.
        self.opponent_reading = None
        self.opponent_limited = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        #: rows whose speed limit applied last step (`speed_limit(held=...)`): their car stays held to `lane_exit`
        self.opponent_held = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        #: `EnvConfig.obstacle_speed_limit`: the rows whose speed command the obstacle limit held this step
        self.obstacle_limited = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if e.opponent_speed_limit or e.obstacle_speed_limit:
            from .opponent_speed_limit import no_car
            self.opponent_reading = no_car(self.B, device=self.device)
        #: Whether each car was in contact last step, so a soft collision is charged on its onset.
        self._was_touching = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._was_touching_wall = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        #: Car contacts: in contact last step, and whether the contact under way is charged (`car_charge_latch`).
        self._car_prev = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._car_resp = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        #: `contact_severity_min`: the wall/prop impact carried from the physical contact's start to its
        #: onset (`contact_severity.carry_wall_impact`), and per car, in contact last step and whether the
        #: contact under way was hard enough to be charged (decided at its first step).
        self._wall_pending = torch.zeros(self.B, device=self.device)
        self._car_sev_prev = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._car_sev_ok = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._replay = None
        if self.ecfg.replay_contacts > 0:
            if self.M != 2:
                raise ValueError("replay_contacts replays a two-car scene; race_size must be 2")
            self._replay = _ContactReplay(self.B, self.ecfg, 1.0 / self.cfg.sim.control_rate, self.device)
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
        self.grip_budget = None                                # see `set_grip_budget`
        if e.action_mode == "plan":
            # `None` for the default mode, so the tracker builds the PlanSpec it always built
            if e.speed_command not in ("tracker", "profile"):
                raise ValueError(f"speed_command must be 'tracker' or 'profile', not {e.speed_command!r}")
            _trk = (e.tracker_speed_weight is not None or e.tracker_a_max is not None
                    or bool(e.tracker_profile_from_prediction))
            pspec = (None if e.speed_mode == "linear" and e.plan_kappa_mode == "absolute" and not _trk
                     else PlanSpec(speed_mode=e.speed_mode, a_brake_profile=float(e.plan_a_brake),
                                   kappa_mode=e.plan_kappa_mode, kappa_a_lat=float(e.plan_kappa_a_lat)))
            if _trk:
                import dataclasses as _dc
                _kw = {}
                if e.tracker_a_max is not None:
                    _kw["a_max"] = float(e.tracker_a_max)
                if e.tracker_speed_weight is not None:
                    _kw["q"] = pspec.q[:3] + (float(e.tracker_speed_weight),)
                    _kw["qf"] = pspec.qf[:3] + (float(e.tracker_speed_weight),)
                if e.tracker_profile_from_prediction:
                    _kw["profile_from_prediction"] = True
                pspec = _dc.replace(pspec, **_kw)
            self.tracker = PlanTracker(self.B, self.device, self.cfg.vehicle.lf + self.cfg.vehicle.lr, self.cfg.vehicle.s_max,
                                       e.v_max_policy, spec=pspec, compile_solver=e.compile_tracker)
            self.tracker.control_dt = float(self.sim.control_dt)
            self._calibrate_tracker(torch.arange(self.B, device=self.device))
            if e.plan_follower == "map":
                from .map_controller import POLICY_MAP, MAPParams, MAPPlanTracker, nominal_lut
                self.tracker = MAPPlanTracker(self.tracker, nominal_lut(self.sim),
                                              MAPParams.from_dict(dict(POLICY_MAP, **(e.plan_follower_map or {}))),
                                              self.sim.params.nominal)
            elif e.plan_follower != "mpc":
                raise ValueError(f"plan_follower must be 'mpc' or 'map', not {e.plan_follower!r}")
        if int(e.decision_every) < 1 or (int(e.decision_every) > 1 and (e.action_mode != "plan" or e.plan_follower != "map")):
            raise ValueError(f"decision_every {e.decision_every}: >= 1, and above 1 only for plans followed by MAP "
                             f"(the held plan is MAP's waypoints in the world)")
        #: control steps since each car's last decision (`EnvConfig.decision_every`); 0 at an episode start
        self._dec_cnt = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._held_action: Optional[torch.Tensor] = None
        #: The spec the cars the learner does not drive plan in (`EnvConfig.opponent_speed_mode`): the
        #: tracker's own, the very object, unless that names another format -- then the tracker's with
        #: that speed mode, and the tracker reads every row in its own (`_sync_row_formats`).
        self.opp_plan_spec = None if self.tracker is None else self.tracker.spec
        self._row_codes = None
        #: Per league entry, the format its rows are read in (`_set_entry_formats`): a uint8 table
        #: indexed by driver code -- the learner's, the opponents', then checkpoint entry j's own at
        #: `OPP_DRIVER_POOL + j`. None while every entry plans in the opponents' format.
        self._pool_codes = None
        if e.opponent_speed_mode is not None and e.opponent_speed_mode != e.speed_mode:
            from .mpc import ROW_FORMATS
            if self.tracker is None:
                raise ValueError("opponent_speed_mode is a plan format: it needs action_mode 'plan'")
            if e.speed_mode not in ROW_FORMATS or e.opponent_speed_mode not in ROW_FORMATS:
                raise ValueError(f"one tracker reads the plan formats {ROW_FORMATS} side by side; speed_mode "
                                 f"{e.speed_mode!r} with opponent_speed_mode {e.opponent_speed_mode!r} is not a pair of them")
            import dataclasses as _dc
            self.opp_plan_spec = _dc.replace(self.tracker.spec, speed_mode=e.opponent_speed_mode)
            #: (code of the learner's format, code of the others'), as 0-d tensors so the per-step
            #: selection in `_sync_row_formats` allocates and synchronises nothing
            self._row_codes = tuple(torch.tensor(ROW_FORMATS.index(m), dtype=torch.uint8, device=self.device)
                                    for m in (e.speed_mode, e.opponent_speed_mode))
            self.tracker.set_row_formats(torch.full((self.B,), ROW_FORMATS.index(e.speed_mode),
                                                    dtype=torch.uint8, device=self.device))
            self._sync_row_formats()
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
        #: A staggered first episode's own time limit, per row; `NO_LIMIT` otherwise. Only ever
        #: shortens `ecfg.max_steps`, which is still read live -- callers change it after
        #: construction (the episode-boundary tests do), and a copy taken at reset would miss that.
        self.ep_limit = torch.full((self.B,), self.NO_LIMIT, dtype=torch.long, device=self.device)
        #: Soft contacts begun this episode (onsets, as charged). Under `terminate` an episode holds
        #: at most one and it ends it, so `final["collided"]` counted them; under `soft` nothing
        #: ends and `collided` was always 0, which is what the training log printed as "0.0/km".
        self.ep_contacts = torch.zeros(self.B, device=self.device)
        self.lap_start_step = torch.zeros(self.B, dtype=torch.long, device=self.device)   # step of the last finish-line crossing
        self.prev_lap = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.ep_return = torch.zeros(self.B, device=self.device)
        self.ep_progress = torch.zeros(self.B, device=self.device)
        self.last_result: Optional[StepResult] = None
        self._empty_long = torch.zeros(0, dtype=torch.long, device=self.device); self._empty_float = torch.zeros(0, device=self.device)
        self._no_plan = torch.zeros(self.B, 1, 4, device=self.device)
        # ---- privileged opponent tokens (f1sim.opp_token). An oracle; "off" allocates nothing.
        self.opp_token = validate_opp_token(e.opp_token)
        self.opp_token_dim = opp_token_dim(self.opp_token)
        self.opp_token_ablate = bool(e.opp_token_ablate)
        if self.opp_token_ablate and self.opp_token == "off":
            raise ValueError("opp_token_ablate with opp_token 'off': zeroing a block that does not "
                             "exist is not an ablation, it is the control arm. Score A0 instead.")
        if self.opp_token != "off":
            if self.M < 2:
                raise ValueError(
                    f"opp_token {self.opp_token!r} needs race_size > 1: with one car per race the "
                    f"whole block is the zeros an absent opponent already produces, so the arm "
                    f"would differ from the control by {self.opp_token_dim} dead input columns and "
                    f"nothing else.")
            if e.action_mode != "plan":
                # The `future` columns are the opponent's own tracked plan, and `direct` mode has no
                # tracker to read one from. Refused for every mode, not just `future`, so that the
                # three arms of a comparison are the same object in every respect but their width.
                raise ValueError(
                    f"opp_token {self.opp_token!r} needs action_mode 'plan': the future columns are "
                    f"the opponent's own plan-tracker reference, and in 'direct' mode no plan "
                    f"exists to read.")
            #: This step's plan for every car, in that car's own body frame, on the tracker's
            #: uniform time grid. `OPP_TOKEN_PLAN_SOURCE` says which of the tracker's two
            #: trajectories it is; `_plan_shift` is the offset of its clock from the anchor pose
            #: (zero for `ref`, the car's calibrated command latency for `pred`, whose rollout
            #: starts at the latency-compensated pose).
            self._plan_traj = torch.zeros(self.B, self.tracker.spec.N + 1, 4, device=self.device)
            #: Pose (x, y, yaw) the row of `_plan_traj` above is anchored at -- the pose the tracker
            #: was called with, i.e. the state one control step BEFORE the observation the token
            #: goes into. `_opp_token_times` carries that offset.
            self._plan_pose = torch.zeros(self.B, 3, device=self.device)
            #: Per row, whether `_plan_traj` is this episode's own plan rather than the
            #: constant-velocity stand-in a just-respawned car gets. Reported, not used: it is how
            #: often the label is a fallback, which a reader of the validation table needs.
            self._plan_real = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            self._token_horizons = torch.tensor(OPP_TOKEN_HORIZONS_S, device=self.device)
            #: Off in training: keeping the tracker's OTHER trajectory beside the one the label uses
            #: costs a second (B, N+1, 4) copy per step and nothing reads it.
            #: `work/oracle-planner/work/label_validation.py` turns it on to score both against the
            #: realised future, which is the evidence `OPP_TOKEN_PLAN_SOURCE` was chosen on.
            self.opp_token_keep_alt = False
            self._plan_alt = torch.zeros(self.B, self.tracker.spec.N + 1, 4, device=self.device)
        self.single_observation_space, self.single_action_space = self._spaces()

    def _check_obs_v2(self) -> None:
        """Refuse a v2 env that could not produce the v2 observation honestly."""
        e = self.ecfg
        if e.action_mode != "plan":
            raise ValueError("obs_version 2 is the plan policy's contract (its history holds plan "
                             "actions): set action_mode 'plan'")
        if not (e.obs_imu and self.cfg.imu.enabled):
            raise ValueError("obs_version 2 reads the gyro's z axis; the IMU is off "
                             "(EnvConfig.obs_imu / cfg.imu.enabled)")
        if validate_opp_token(e.opp_token) != "off":
            raise ValueError("opp_token is an oracle block of the v1 proprio vector; contract v2 has "
                             "no place for it")
        if not 1 <= int(e.scan_stack) <= V2_MAX_FRAMES:
            raise ValueError(f"obs_version 2 takes the newest 1..{V2_MAX_FRAMES} frames; "
                             f"scan_stack is {e.scan_stack}")
        if not isinstance(e.ego_actions, (bool, np.bool_)):
            raise ValueError(f"ego_actions is a bool (the v2 ego with or without the issued actions), "
                             f"got {e.ego_actions!r}")
        if int(e.hist_len) < 1:
            raise ValueError("obs_version 2's ego vector is its history: hist_len must be >= 1 "
                             "(the contract's is 20 rows, hist_stride 2)")

    def _spaces(self):
        try:
            import gymnasium as gym
            if self.obs_version == 2:
                ego_dim = v2_ego_dim(self.ecfg.hist_len, self.ecfg.hist_stride, self.act_dim,
                                     bool(self.ecfg.ego_actions))
                obs = gym.spaces.Dict({
                    "scan": gym.spaces.Box(0.0, 1.0, (self.ecfg.scan_stack, self.n_beams), np.float32),
                    "ego": gym.spaces.Box(-5.0, 5.0, (ego_dim,), np.float32)})
                return obs, gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
            spaces = {
                "scan": gym.spaces.Box(0.0, 1.0, (self.ecfg.scan_stack, self.n_beams), np.float32),
                "speed": gym.spaces.Box(-1.0, 2.0, (1,), np.float32),
                "prev_action": gym.spaces.Box(-1.0, 1.0, (self.act_dim * self.ecfg.action_history,), np.float32),
                **({"hist": gym.spaces.Box(-5.0, 5.0, (self.ecfg.hist_len * self.row_dim,), np.float32)} if self.ecfg.hist_len > 0 else {}),
                "speed_cap": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
                **({"opp_token": gym.spaces.Box(-np.inf, np.inf, (opp_token_dim(self.ecfg.opp_token),), np.float32)}
                   if validate_opp_token(self.ecfg.opp_token) != "off" else {})}
            if self.ecfg.obs_imu and self.cfg.imu.enabled:
                spaces["imu"] = gym.spaces.Box(-5.0, 5.0, (6,), np.float32)
                spaces["imu_att"] = gym.spaces.Box(-2.0, 2.0, (2,), np.float32)
            obs = gym.spaces.Dict(spaces)
            act = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
            return obs, act
        except ImportError:
            return None, None

    def _calibrate_external(self, ids: torch.Tensor):
        """The car calibration an external driver's commands go through when there is no tracker to
        carry one (direct actions): the same measurement `_calibrate_tracker` gives the tracker -- servo
        offset to +-0.01 rad, gain to +-4 %, VESC speed gain to +-3 % -- drawn from a generator of its
        own, so a direct-action run's spawns and randomisation are the ones they always were."""
        if getattr(self, "_ext_cal", None) is None:
            self._ext_gen = torch.Generator(device=self.device)
            self._ext_gen.manual_seed(int(self.sim.gen.initial_seed()) + 7919)
            self._ext_cal = torch.zeros(self.B, 3, device=self.device)
            self._ext_cal[:, 1:] = 1.0
        P = self.sim.P; n = ids.numel(); g = self._ext_gen
        u = lambda a: (torch.rand(n, device=self.device, generator=g) - 0.5) * 2 * a
        self._ext_cal[ids, 0] = P["steer_bias"][ids] + u(0.01)
        self._ext_cal[ids, 1] = P["steer_gain"][ids] * (1 + u(0.04))
        self._ext_cal[ids, 2] = P["speed_gain"][ids] * (1 + u(0.03))

    def _calibrate_tracker(self, ids: torch.Tensor):
        """The tracker is *calibrated* on the real car the way the odometry is (measured once): in sim it
        knows each env's command delay + half the servo lag (+-20 ms residual), the servo offset
        (+-0.01 rad) and gain (+-4 %) and the VESC speed gain (+-3 %); grip stays unknown by decision."""
        P = self.sim.P; n = ids.numel(); g = self.sim.gen
        u = lambda a: (torch.rand(n, device=self.device, generator=g) - 0.5) * 2 * a
        if self.tracker_delay is None:
            self.tracker_delay = torch.zeros(self.B, device=self.device)
            self.tracker_cal = torch.zeros(self.B, 3, device=self.device)          # steer bias, steer gain, speed gain
            self.tracker_motor_tau = torch.zeros(self.B, device=self.device)
        self.tracker_delay[ids] = (P["cmd_delay"][ids] + 0.5 * P["servo_tau"][ids] + u(0.02)).clamp(0.0, 0.2)
        self.tracker_cal[ids, 0] = P["steer_bias"][ids] + u(0.01)
        self.tracker_cal[ids, 1] = P["steer_gain"][ids] * (1 + u(0.04))
        self.tracker_cal[ids, 2] = P["speed_gain"][ids] * (1 + u(0.03))
        # measured once like the rest, to +-10 %: `tracker_speed_ff` reads it. Drawn only when it is
        # on -- a draw here moves the generator, and every spawn after it, for runs that never read it.
        if self.ecfg.tracker_speed_ff or self.ecfg.speed_command == "profile":
            self.tracker_motor_tau[ids] = P["motor_tau"][ids] * (1 + u(0.10))

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
        #: Slots driven by a published baseline from the car's own sensors (`DriverKind.external`):
        #: grid slot -> its rows. The drivers themselves come from `set_external_drivers`.
        self.external_slot_rows = {i: torch.nonzero(self.slot == i).flatten()
                                   for i, sl in enumerate(self.slots, start=1) if sl.external_driven}
        self.external_driven = torch.zeros(self.B, dtype=torch.bool, device=dev)
        for rows in self.external_slot_rows.values():
            self.external_driven[rows] = True
        self.on_policy = self.learner.clone()
        self.slot_scale_lo, self.slot_scale_hi = t(lo)[self.slot], t(hi)[self.slot]
        self.slot_cap = t(cap)[self.slot]
        self.slot_cap_set = t(cap_set, torch.bool)[self.slot]
        self.slot_grip_code = t(grip, torch.long)[self.slot]
        #: League slots (`OpponentSlot.ckpt_mix`): per grid slot, the probability a race draws a
        #: checkpoint for it and the driver codes it draws from. Empty when no slot has a league,
        #: and then `_redraw_slot_league` draws nothing.
        self._league = [(i, float(sl.ckpt_p), t([OPP_DRIVER_POOL + ck_index[c] for c in sl.ckpt_mix], torch.long))
                        for i, sl in enumerate(self.slots, start=1) if sl.ckpt_mix and sl.ckpt_p > 0]
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
        # Every kind any slot can be driven by, not only the one it starts as: a `kind_mix` slot
        # changes driver at every race reset, and the objects are built once for all of them.
        alt, masks = [], {}
        from .opponent_slots import kind_of as _kind_of
        self._slot_kinds = [tuple(sl.kind_mix) or (sl.kind,) for sl in self.slots]
        self._slot_mixed = any(len(k) > 1 for k in self._slot_kinds)
        for i, names in enumerate(self._slot_kinds, start=1):
            for name in names:
                kind = _kind_of(name)
                if not (kind.teacher and kind.teacher_factory):
                    continue
                if kind.name not in masks:
                    alt.append(kind.name)
                    masks[kind.name] = torch.zeros(self.B, dtype=torch.bool, device=dev)
                if len(names) == 1:
                    masks[kind.name] |= self.slot == i
        self.alt_teacher_kinds = tuple(alt)
        self.alt_teacher_mask = masks
        #: Which of `alt_teacher_kinds` drives each row right now, or -1 for "the raceline teacher".
        #: Only mixed slots move; a fixed slot's mask above is already final and this stays -1 for
        #: it, which is why an unmixed table costs nothing and draws nothing.
        self.slot_kind_code = torch.full((self.B,), -1, dtype=torch.long, device=dev)
        #: Rows whose driver `_apply_slot_kinds` owns. A fixed slot's mask bit was set once above
        #: and must survive every redraw, so the rewrite is scoped to exactly these rows.
        self._slot_mixed_rows = torch.zeros(self.B, dtype=torch.bool, device=dev)
        for i, names in enumerate(self._slot_kinds, start=1):
            if len(names) > 1:
                self._slot_mixed_rows |= self.slot == i
        self._slot_kind_ids = [
            torch.tensor([alt.index(n) if n in alt else -1 for n in names],
                         dtype=torch.long, device=dev) if len(names) > 1 else None
            for names in self._slot_kinds]
        if self.ecfg.kind_mix_assign not in ("redraw", "partition"):
            raise ValueError(f"kind_mix_assign must be 'redraw' or 'partition', not {self.ecfg.kind_mix_assign!r}")
        if self._slot_mixed and self.ecfg.kind_mix_assign == "partition":
            # Race g's mixed slot drives kind_mix[g % len]: balanced, and fixed from here on.
            G = self.B // self.M
            code = self.slot_kind_code.view(G, self.M)
            race = torch.arange(G, device=dev)
            for i, ids in enumerate(self._slot_kind_ids, start=1):
                if ids is not None:
                    code[:, i] = ids[race % ids.numel()]
        if self._slot_mixed:
            self._apply_slot_kinds()
        self._alt_rows = self._rows_each_kind_can_drive()

    def _rows_each_kind_can_drive(self) -> dict:
        """The rows each alt kind can ever drive, as a static index, read back once.

        A teacher that can plan a subset (`plan_rows`) is asked for these rows only instead of the
        whole batch -- the learners' rows and the other kinds' rows never need it. Under
        `kind_mix_assign="redraw"` that is its fixed slots and every mixed slot listing it, since a
        redraw can hand it any of those; under "partition" the masks are already final and it is
        exactly the rows it drives.
        """
        from .opponent_slots import kind_of as _kind_of
        out = {}
        slot_host = self.slot.cpu()
        for k in self.alt_teacher_kinds:
            if self.ecfg.kind_mix_assign == "partition" or not self._slot_mixed:
                hit = self.alt_teacher_mask[k].cpu()
            else:
                own = [i for i, names in enumerate(self._slot_kinds, start=1)
                       if any(_kind_of(n).name == k for n in names)]
                hit = torch.zeros_like(slot_host, dtype=torch.bool)
                for i in own:
                    hit |= slot_host == i
            out[k] = torch.nonzero(hit).flatten().to(self.device)
        return out

    def _widen_clearance_to_props(self, r) -> None:
        """Make `wall_dist` mean what its consumers think it means: room to the nearest *thing*.

        It is the occupancy distance field, and the procedural obstacles are props -- never
        rasterised into that grid. Two consumers read it and both were being misled:

        * `reward_proximity`, which pays per metre driven inside `safe_dist` of a wall, paid
          nothing at all for driving straight at a crate;
        * the privileged observation's clearance column, which told the policy it had room while
          the car sat beside one.

        So the only thing the policy ever learned about an obstacle was the terminal -60 at the
        moment of contact: no gradient for coming close, none for steering early. That is the
        sparse-terminal regime, and it produces a policy that avoids one sometimes and drives into
        the next -- which is what both the numbers and the user's own eye said after 6.3 M steps
        with layouts on the racing line (2026-09-21).

        The collision test is deliberately NOT touched. `hit` was computed inside the physics roll
        from the wall-only clearance and the props have their own contact SAT; folding them in here
        would count one contact twice and turn a prop hit into a wall hit in every report.
        """
        if self.procedural is None:
            return
        pts = self.sim._footprint_corners(r.state)                     # (B, 4, 2)
        B, K, _ = pts.shape
        eid = torch.arange(B, device=pts.device)[:, None].expand(B, K).reshape(-1)
        prop = self.procedural.clearance(pts.reshape(-1, 2), eid).view(B, K).min(1).values
        r.wall_dist = torch.minimum(r.wall_dist, prop.to(r.wall_dist.dtype))

    def _check_prop_aware_drivers(self) -> None:
        """Refuse a layout on the racing line when somebody driving cannot see one.

        `procedural_raceline_corridor = "off"` is what lets an obstacle stand *on* the line, and it
        is the only way the policy ever meets one there. The price is that every car driven by a
        teacher now has to steer round a prop by itself, and most cannot: the raceline teacher is
        pure pursuit on a line built from the occupancy grid, and the props were never rasterised
        into it. Left unchecked, the layout would simply end the opponents' races, and a collision
        rate measured against that is the layout's number rather than the policy's.

        So it is refused here, before anything is drawn, with the fix in the message.
        """
        e = self.ecfg
        if e.procedural_raceline_corridor not in ("on", "off"):
            raise ValueError(f"procedural_raceline_corridor must be 'on' or 'off', not "
                             f"{e.procedural_raceline_corridor!r}")
        if e.procedural_raceline_corridor == "on" or e.procedural_obstacles <= 0:
            return
        if e.spawn_runway <= 0.0:
            raise ValueError(
                "procedural_raceline_corridor='off' puts obstacles on the racing line, which is "
                "also where cars spawn -- so without spawn_runway every reset can begin a metre "
                "from a crate at up to spawn_speed_max, and the collision rate that comes out is "
                "the layout's rather than the policy's. Set spawn_runway to what the car needs to "
                f"stop from {e.spawn_speed_max:g} m/s plus its own length (about 3 m), or leave "
                "the corridor on.")
        if self.M < 2:
            return
        from .opponent_slots import KIND_BY_NAME
        if self.slots is None:
            if e.opponent in ("teacher", "mixed"):
                raise ValueError(
                    "procedural_raceline_corridor='off' puts obstacles on the racing line, and "
                    f"opponent={e.opponent!r} drives the other cars with the raceline teacher, "
                    "which cannot see a prop (they are not in the occupancy grid) and has no "
                    "lateral freedom to use if it could. Name prop-aware drivers with --opp-slots "
                    f"({', '.join(k.name for k in KIND_BY_NAME.values() if k.prop_aware)}), or "
                    "leave the corridor on.")
            return
        # Every kind a slot can be *drawn as*, not only the one it starts as: a `kind_mix` that
        # lists `raceline` puts a prop-blind driver on the line at some reset, and the run would
        # look fine until it did.
        blind = sorted({name for sl in self.slots for name in (sl.kind_mix or (sl.kind,))
                        if KIND_BY_NAME[name].teacher and not KIND_BY_NAME[name].prop_aware})
        if blind:
            raise ValueError(
                f"procedural_raceline_corridor='off' puts obstacles on the racing line, and slot "
                f"kind(s) {blind} cannot see a prop -- their races would end on the layout and "
                f"every collision number measured against them would be the layout's. Prop-aware "
                f"kinds: {', '.join(k.name for k in KIND_BY_NAME.values() if k.prop_aware)}.")

    def _redraw_slot_kinds(self, full: torch.Tensor, gen: torch.Generator) -> None:
        """Redraw which driver moves each `kind_mix` car, once per race that fully reset.

        Per race and not per step, for the reason every other slot draw is: a car that crashes and
        rejoins is in the same race it started, and a driver that changed mid-race would be a
        different opponent behind the same car. Uniform over the names listed; a name written twice
        is drawn twice as often, which is the whole weighting vocabulary and is enough.
        """
        if not self._slot_mixed or self.ecfg.kind_mix_assign == "partition":
            return                            # partitioned: each race keeps the kind it was given
        G = full.shape[0]
        code = self.slot_kind_code.view(G, self.M)
        for i, ids in enumerate(self._slot_kind_ids, start=1):
            if ids is None:
                continue
            pick = ids[torch.randint(ids.numel(), (G,), device=self.device, generator=gen)]
            code[:, i] = torch.where(full, pick, code[:, i])
        self._apply_slot_kinds()

    def _redraw_slot_league(self, full: torch.Tensor, gen: torch.Generator) -> None:
        """Per race that fully reset: does a league slot's car get a checkpoint or its teacher?

        Before `_slot_speeds`, which reads `teacher_driven` to decide whether the drawn speed scale
        goes on a teacher profile or on a network's cap. Learner rows never move: a league slot is a
        teacher slot, and a checkpoint car is an opponent like any other.
        """
        if not self._league:
            return
        G = full.shape[0]
        drv = self.opp_driver.view(G, self.M)
        for i, p, codes in self._league:
            use = torch.rand(G, device=self.device, generator=gen) < p
            pick = codes[torch.randint(codes.numel(), (G,), device=self.device, generator=gen)]
            new = torch.where(use, pick, torch.full_like(pick, OPP_DRIVER_TEACHER))
            drv[:, i] = torch.where(full, new, drv[:, i])
        self.teacher_driven.copy_((self.opp_driver == OPP_DRIVER_TEACHER) & (self.slot > 0))
        self.pool_driven.copy_(self.opp_driver >= OPP_DRIVER_POOL)

    def _apply_slot_kinds(self) -> None:
        """Rebuild the alt-teacher masks from `slot_kind_code`, in place.

        In place because `_opponent_actions` holds these tensors, and because a fixed slot's bit was
        set once at construction and must survive: only the rows of mixed slots are rewritten here.
        """
        for j, name in enumerate(self.alt_teacher_kinds):
            mine = self.slot_kind_code == j
            m = self.alt_teacher_mask[name]
            m &= ~self._slot_mixed_rows            # clear only what this function owns
            m |= mine

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

    def _slot_grid(self, ids, full, gen, base, L_race, present=None):
        """Where every resetting car of a slot-mode race spawns: (rank, arc, lat, clearance, yaw,
        abreast).

        The same three questions the flag-driven grid answers -- what order, what arc, and whether an
        abreast row fits between the walls -- asked per car instead of per race. Everything is
        computed for the whole batch and then read at `ids`: the rank of a car depends on the codes
        of its race mates, so the natural shape here is (G, M) and not (n,).

        present (G, M): which seats this race fills. An empty seat is not part of an abreast row --
        a car that is not there neither needs room beside the learner nor pushes it off-centre.
        """
        e = self.ecfg
        G, M = full.shape[0], self.M
        codes = self._slot_spawn_codes(full, gen)                        # (G, M)
        self._redraw_slot_kinds(full, gen)
        self._redraw_slot_league(full, gen)
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
        if present is not None:
            group = group & present
        group[:, 0] = True
        g_size = group.sum(1)
        want = torch.zeros(G, M, device=self.device)
        abreast = torch.zeros(G, M, dtype=torch.bool, device=self.device)
        psi = 3.0 * min(e.spawn_yaw_std, e.spawn_alongside_yaw)
        extent = (self.cfg.vehicle.width * math.cos(psi) + self.cfg.vehicle.length * math.sin(psi))
        if bool((g_size > 1).any()):
            j = group.cumsum(1) - 1                                      # index within the group
            s_group = s_rank.gather(1, rank.clamp_min(0)) - stagger      # each group member's own arc
            tid_row = self.sim.tid.view(-1, M)
            xy0, yaw0 = self.sim.track.pose_at_s(s_group.reshape(-1), tid_row.reshape(-1))
            room = (self.sim.track.sample_edt(xy0, tid_row.reshape(-1)).view(G, M)
                    - 0.5 * extent - e.opp_event_margin)

            def fit(sep):
                w = torch.where(group, (j - (g_size[:, None] - 1) / 2.0) * (extent + sep), torch.zeros_like(want))
                headroom = torch.where(group, room - w.abs() - 0.03, torch.full_like(room, math.inf))
                if getattr(self.sim.track, "has_props", False):
                    nrm = torch.stack([-torch.sin(yaw0), torch.cos(yaw0)], 1)
                    pose_ab = torch.cat([xy0 + nrm * w.reshape(-1)[:, None], yaw0[:, None]], 1)
                    blocked = self.sim._spawn_in_prop(pose_ab, tid_row.reshape(-1),
                                                      torch.arange(self.B, device=self.device)).view(G, M)
                    headroom = torch.where(group & blocked, torch.full_like(headroom, -math.inf), headroom)
                # Per race: a row with one car abreast and another staggered is exactly the grid that
                # collides, because the stagger puts that car back on the arc the abreast one holds.
                return w, (headroom.amin(1) >= 0.0) & (g_size > 1) & full

            # The preferred separation where the lane has room for it, `spawn_alongside_sep_min` where
            # only that fits; the order of the draws above is untouched by either.
            want, fits = fit(e.spawn_alongside_sep)
            if e.spawn_alongside_sep_min < e.spawn_alongside_sep:
                want_n, fits_n = fit(e.spawn_alongside_sep_min)
                want = torch.where(fits[:, None], want, want_n)
                fits = fits | fits_n
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
        return (flat(rank).float(), flat(s_cars), flat(lat_all), flat(min_all), flat(yaw_all), flat(abreast))

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
        if self.procedural is not None and self.ecfg.procedural_raceline_corridor != "off":
            # The teacher is pure pursuit on a raceline built from the occupancy grid, and the
            # procedural props are not in the grid: it cannot see them and will not steer round
            # them. So the layouts are laid *outside* the band the raceline occupies, which is the
            # contract's first option ("place patterns only where the raceline is not"). The other
            # option -- teaching `clamp_offset` about props -- bounds an offset away from the line
            # and does nothing about a crate standing on it.
            #
            # `procedural_raceline_corridor = "off"` skips this, and then a layout may stand on the
            # line. `_check_prop_aware_drivers` has already refused that unless every teacher-driven
            # car is a kind that asks `ProceduralObstacles.clearance` and steers round one.
            self.procedural.set_raceline(teacher)
        if self.events is not None and self.events.enabled:
            # How far off the line each raceline point can be driven without putting a car in the
            # wall, measured once from the track's distance field. The teacher clamps against it at
            # the car's own raceline index, so a shift through a narrow section shrinks instead of
            # crashing. Nothing is computed, and nothing changes, when no event moves a car sideways.
            teacher.offset_limit = raceline_offset_limit_sided(teacher, self.sim.track,
                                                               0.5 * self.cfg.vehicle.width,
                                                               self.ecfg.opp_event_margin)
        if self.events is not None and self.events.needs_corners:
            # Where the corners are on each raceline, for the `line` behaviour. Built once here for
            # the same reason the offset budget is: it is a function of the track set, not of a step.
            self.events.corners = raceline_corners(teacher, self.ecfg.opp_corner_kappa,
                                                   self.ecfg.opp_corner_min_arc,
                                                   self.ecfg.opp_corner_smooth)

    def set_external_drivers(self, drivers: Dict[int, object]) -> None:
        """The published baselines that drive the `external` slots: grid slot -> a
        `learn.baselines.BaselineDriver`, already bound to this car's scanner. Injected, like the pool,
        because building one imports `learn`, which sits above the env."""
        missing = sorted(set(self.external_slot_rows) - set(drivers))
        if missing:
            raise ValueError(f"external slot(s) {missing} have no driver")
        self.external_drivers = dict(drivers)
        self._all_external = bool(self.external_driven.all())

    def set_ego_driver(self, driver) -> None:
        """A published driver for the ego car of every race (slot 0) in place of the caller's action
        (the console's "내 차", 2026-10-17). The same override as an external opponent: its command
        replaces the row's after the tracker, and the other cars predict it at constant velocity. A
        planner (`learn.common.PlannerEgoDriver`) is asked through `command_env(env, rows)`; a sensor
        baseline through its scan and speed."""
        rows = torch.nonzero(self.slot == 0).flatten()
        self.external_slot_rows[0] = rows
        self.external_drivers[0] = driver
        self.external_driven[rows] = True
        self._all_external = bool(self.external_driven.all())
        if hasattr(driver, "rows"):
            driver.rows = rows

    def _external_commands(self, cmd: torch.Tensor) -> torch.Tensor:
        """Each external slot's (steer, speed) from its cars' own raw scans and odometry speed, in
        place of whatever the tracker or the caller's action made of those rows. The scan is the
        one the policy's observation was built from this step (`last_result`), saturated at the
        scanner's range as on the car; steering is clamped to the lock and the speed to the car's cap.
        One host round trip per step: these drivers are numpy / CPU models, exactly as on the car."""
        lr = self.last_result
        if lr is None:
            return cmd
        rmax = float(self.cfg.lidar.range_max)
        # The car is calibrated whoever drives it (2026-10-19). The tracker our policy drives through has
        # always been handed the car's measured servo offset / gain and speed gain (`_calibrate_tracker`),
        # and these drivers command a nominal car (a geometric steering angle, MAP's lookup of the nominal
        # car); they used to reach the actuators raw, on a servo trimmed up to +-0.03 rad off. A team sets
        # the trim and gains once whatever runs on the car, so every driver gets the same correction.
        cal = getattr(self, "tracker_cal", None)
        cal = cal if cal is not None else getattr(self, "_ext_cal", None)
        if cal is None:
            self._calibrate_external(torch.arange(self.B, device=self.device))
            cal = self._ext_cal

        def calibrated(c, rows):
            return torch.stack([(c[:, 0] - cal[rows, 0].to(c.dtype)) / cal[rows, 1].to(c.dtype),
                                c[:, 1] / cal[rows, 2].to(c.dtype)], 1)
        for i, rows in self.external_slot_rows.items():
            drv = self.external_drivers[i]
            if hasattr(drv, "command_env"):                  # a planner: privileged state, not a scan
                c = calibrated(drv.command_env(self, rows).to(cmd.dtype), rows)
                steer = c[:, 0].clamp(-float(self.s_max), float(self.s_max))
                speed = torch.minimum(c[:, 1], self.speed_cap[rows].to(cmd.dtype))
                cmd = cmd.index_copy(0, rows, torch.stack([steer, speed], 1))
                continue
            rng = torch.nan_to_num(lr.scan[rows].float(), nan=rmax, posinf=rmax, neginf=0.0).clamp(0.0, rmax)
            v = lr.odom[rows, 3].float() if drv.needs_speed else None
            c = drv.command(drv.adapt(rng.cpu().numpy()), None if v is None else v.cpu().numpy())
            c = calibrated(torch.as_tensor(c, dtype=cmd.dtype, device=cmd.device), rows)
            steer = c[:, 0].clamp(-float(self.s_max), float(self.s_max))
            speed = torch.minimum(c[:, 1], self.speed_cap[rows].to(cmd.dtype))
            cmd = cmd.index_copy(0, rows, torch.stack([steer, speed], 1))
        return cmd

    def set_opponent_pool(self, pool):
        """Population that drives the opponent cars (opponent == "pool").

        `pool` is a `learn.opponent_pool.OpponentPool` -- injected rather than built here for the
        same reason the teacher is: loading a checkpoint means `learn.model`, and the env is below
        `learn` in the import order. The env owns which car each entry drives; the pool owns how an
        entry is asked for an action.

        Called again whenever an entry is swapped (`OpponentPool.replace`): each entry plans in its
        own format (`entry_formats`, `_set_entry_formats`) and may read its own observation contract
        (`seat_builders`, fed by this env's hooks), and both can change with a swap.
        """
        if self.ecfg.opponent not in ("pool", "slots"):
            raise RuntimeError(f"set_opponent_pool on an env with opponent={self.ecfg.opponent!r}: "
                               f"the pool only drives cars in 'pool' mode, or as the checkpoint "
                               f"entries of a slot table")
        if len(pool) != len(self.pool_paths):
            raise ValueError(f"the pool holds {len(pool)} checkpoint(s) but opp_pool names "
                             f"{len(self.pool_paths)}: {list(self.pool_paths)}")
        formats = getattr(pool, "entry_formats", None)
        if formats is not None:
            self._set_entry_formats(list(formats))
        self.pool = pool

    def _seat_builders(self) -> tuple:
        """The pool's per-seat observation builders (`learn.seat_obs`): one per observation contract an
        entry reads that is not this env's own. Empty -- every v1 pool in a v1 env, every path before
        2026-10-14 -- and then no hook below does anything. Plus `extra_seat_builders`: builders a
        consumer other than the pool attached (`learn.policy_teacher`, a v1 DAgger teacher's own view)."""
        pool = getattr(self.pool, "seat_builders", ()) if self.pool is not None else ()
        extra = getattr(self, "extra_seat_builders", ())
        return tuple(pool) + tuple(extra) if extra else pool

    def _set_entry_formats(self, formats: list) -> None:
        """Read each league entry's rows in the entry's own plan format (output-format design 6).

        `formats[j]` is checkpoint entry j's speed mode (`OpponentPool.entry_formats`, from its
        contract). While every entry plans in the opponents' format (`opp_plan_spec`) nothing changes:
        the env reads every non-learner row in that format, as before. A v2 learner's snapshot joining
        a linear league -- or a linear league seat beside wp4 scripted cars -- needs per-row formats:
        the tracker's codes are written per driver (`_row_format_codes`), with the one iLQR solve the
        mixed tracker already runs (`mpc.solve(row_format=...)`, bit for bit the single-format solve
        per row)."""
        if self.tracker is None:
            return                                    # direct actions: no plan format to read
        from .mpc import ROW_FORMATS
        opp_mode = self.opp_plan_spec.speed_mode
        if self._pool_codes is None and all(f == opp_mode for f in formats):
            return
        modes = [self.tracker.spec.speed_mode, opp_mode] + list(formats)
        bad = [m for m in modes if m not in ROW_FORMATS]
        if bad:
            raise ValueError(f"league entries plan in {sorted(set(formats))} beside the learner's "
                             f"{self.tracker.spec.speed_mode!r} and the opponents' {opp_mode!r}; one tracker reads "
                             f"only {ROW_FORMATS} side by side")
        if self._row_codes is None:
            if self.tracker._solver is not None:
                # a solver graph recorded without codes would be handed one more argument than it took
                raise ValueError("a league entry plans in another format than every car of this env, and the "
                                 "tracker's solver was already captured for one format: give the env its "
                                 "opponents' format (EnvConfig.opponent_speed_mode) from the start")
            ego = ROW_FORMATS.index(self.tracker.spec.speed_mode)
            self._row_codes = tuple(torch.tensor(c, dtype=torch.uint8, device=self.device)
                                    for c in (ego, ROW_FORMATS.index(opp_mode)))
            self.tracker.set_row_formats(torch.full((self.B,), ego, dtype=torch.uint8, device=self.device))
        lut = [ROW_FORMATS.index(self.tracker.spec.speed_mode), ROW_FORMATS.index(opp_mode)]
        assert len(lut) == OPP_DRIVER_POOL
        lut += [ROW_FORMATS.index(f) for f in formats]
        self._pool_codes = torch.tensor(lut, dtype=torch.uint8, device=self.device)
        self._sync_row_formats()

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
        if self.obs_version == 2:
            return self._obs_v2(r)
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
        if self.opp_token != "off":
            # Last, so the block is appended after every key the policy has ever had -- which is
            # what makes the extra input columns of an oracle arm the trailing ones, and a warm
            # start from a checkpoint without it a column-append rather than a re-layout.
            obs["opp_token"] = self.opp_tokens(r)
        return obs

    def _obs_v2(self, r: StepResult) -> Dict[str, torch.Tensor]:
        """Contract v2 (`learn.obs.ObsSpecV2`): {"scan": the newest frames, "ego": (B, ego_dim)}.

        The ego vector is [speed, gyro z] of THIS observation, the action issued two steps back and
        the history ring's rows, every `hist_stride`-th (without the issued actions when
        `EnvConfig.ego_actions` is off) -- `learn.obs.ego_columns` is the table,
        `learn.obs.assemble_ego_v2` the one assembly the car's builder shares. A row whose episode just started
        is filled here, from its own first observation with zero actions, which is the rule
        `ObsBuilderV2` applies on the car: one rule on both sides, so they agree from the first
        step on rather than only after the history has rolled over. `torch.where` rather than a
        branch on the flag: no host sync per step.
        """
        e = self.ecfg
        if r.imu is None or r.imu.shape[1] == 0:
            raise RuntimeError("obs_version 2 reads the gyro's z axis and this step delivered no IMU "
                               "sample; the node inhibits on such a step, and so must the env")
        speed = norm_speed(r.odom[:, 3], e.v_max_policy)[:, None]
        gz = norm_gyro_z(r.imu.mean(1)[:, 2:3], e.imu_gyro_scale)
        feat = torch.cat([speed, gz], 1)
        fill = torch.cat([feat, torch.zeros(self.B, self.act_dim, device=feat.device, dtype=feat.dtype)], 1)
        self.hist2 = torch.where(self._fresh2[:, None, None], fill[:, None, :].expand_as(self.hist2), self.hist2)
        self._fresh2 = torch.zeros_like(self._fresh2)
        self._last_feat2 = feat
        ego = assemble_ego_v2(feat, self.hist2, e.hist_len, e.hist_stride, bool(e.ego_actions))
        return {"scan": self.scan_hist[:, ::e.scan_stride].clone(), "ego": ego}

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
            if self.sim.absent_any:
                # an empty seat's car is parked 1 km away: read as "no car near" at a bounded 30 m, not
                # as a 200-unit input to the critic
                far = self.sim.absent[o[ar, j]]
                dx = torch.where(far, torch.full_like(dx, 30.0), dx); dy = torch.where(far, torch.zeros_like(dy), dy)
                dv = torch.where(far, torch.zeros_like(dv), dv)
                dd = torch.where(far, torch.full_like(dx, 30.0), dist[ar, j])
            else:
                dd = dist[ar, j]
            pv = torch.cat([pv, torch.stack([dx / s_, dy / s_, dv / s_, dd / s_], 1)], 1)
        return pv

    def _reset_envs(self, ids: torch.Tensor):
        """Reset these rows. With a `level_replay` (prioritized level replay, `learn.level_replay`) every
        race that resets whole is reset on its own, with the simulator's generator set to the level seed
        the replay chooses for that race seat -- which makes the seed name the whole level: venue variant,
        layout, opponents, grid, physics. The main stream is restored after each, so the rest of the
        batch draws exactly as it would have. Off (None): exactly the reset it always was."""
        if getattr(self, "level_replay", None) is None or ids.numel() == 0:
            return self._reset_envs_core(ids)
        M = self.M
        in_ids = torch.zeros(self.B, dtype=torch.bool, device=self.device); in_ids[ids] = True
        full = in_ids.view(-1, M).all(1)
        whole = full[self.race] & in_ids
        part = ids[~whole[ids]]
        if part.numel():
            self._reset_envs_core(part, fill=False)          # a car rejoining its race: the main stream
        gen = self.sim.gen
        for r in torch.nonzero(full).flatten().tolist():
            rows = torch.arange(r * M, (r + 1) * M, device=self.device)
            seed = int(self.level_replay.choose(r))
            st = gen.get_state(); gen.manual_seed(seed)
            try:
                self._reset_envs_core(rows, fill=False)
            finally:
                gen.set_state(st)
            self.level_seed[rows] = seed
        return self._fill_scans(ids, self.sim.odom.state[ids, 3])

    def _reset_envs_core(self, ids: torch.Tensor, fill: bool = True):
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
            abreast_rows = torch.zeros(n, dtype=torch.bool, device=self.device)
            empty_g = None
            if e.spawn_as_race and getattr(self, "_present_p", None) is not None:
                # Who is in the race is drawn before the grid is laid out, not after: an empty seat
                # must neither hold the race's placement up nor be what an abreast row makes room for.
                empty_g = torch.rand(full.shape[0], self.M, device=self.device, generator=gen) >= self._present_p[None, :]
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
                rank, s_full, lat, min_clear, yaw_std, abreast_rows = self._slot_grid(
                    ids, full, gen, base, L_race, present=None if empty_g is None else ~empty_g)
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
                    # Independent per car, not `slot * al`: an accumulating stagger spreads a three-car
                    # grid over a metre of arc, and on a 2 m radius the centerline normal turns enough
                    # over that distance that "0.56 m apart along the normal" stops meaning 0.56 m
                    # apart. Inside 0.2 m the frames agree to a couple of centimetres.
                    s_slot = torch.remainder(base[race] * L - al, L)
                    xy0, yaw0 = self.sim.track.pose_at_s(s_slot, self.sim.tid[ids])
                    room = self.sim.track.sample_edt(xy0, self.sim.tid[ids]) - 0.5 * extent - e.opp_event_margin

                    def fit(sep):
                        w = (slot.float() - (self.M - 1) / 2.0) * (extent + sep)
                        # Per *race*, not per car: a race with one car abreast and another staggered
                        # is exactly the grid that collides, because the stagger puts the last slot
                        # back on the leader's own arc position -- where the abreast car already is.
                        # Measured with the decision taken per car: 232 of 528 three-car spawns in
                        # contact.
                        headroom = room - w.abs() - 0.03
                        if getattr(self.sim.track, "has_props", False):
                            # Props are not in the distance field, so `room` cannot see them. A crate
                            # standing where the grid wants to be would send the abreast row into
                            # `sample_spawn`'s rejection -- which now walks the whole race rather than
                            # pulling one car onto its partner's line, but a row the crates block
                            # where it was drawn is still better started staggered here than moved
                            # half a lap. `-inf` rather than a flag so the per-race reduction covers it.
                            nrm = torch.stack([-torch.sin(yaw0), torch.cos(yaw0)], 1)
                            pose_ab = torch.cat([xy0 + nrm * w[:, None], yaw0[:, None]], 1)
                            blocked = self.sim._spawn_in_prop(pose_ab, self.sim.tid[ids], ids)
                            headroom = torch.where(blocked, torch.full_like(headroom, -math.inf), headroom)
                        worst_ = torch.full((self.B // self.M,), math.inf, device=self.device)
                        worst_.scatter_reduce_(0, race, headroom, reduce="amin", include_self=False)
                        return w, worst_

                    want, worst = fit(e.spawn_alongside_sep)
                    if e.spawn_alongside_sep_min < e.spawn_alongside_sep:
                        # `spawn_alongside_sep_min` for a race whose lane has no room for the preferred gap
                        want_n, worst_n = fit(e.spawn_alongside_sep_min)
                        want = torch.where(worst[race] >= 0.0, want, want_n)
                        worst = torch.maximum(worst, worst_n)
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
                    abreast_rows = abreast
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
        race_kw = {}
        if self.M > 1 and e.spawn_as_race:
            # The race is laid out as one (`Simulator.sample_spawn`, `group`): a race resetting
            # whole walks together, a car rejoining one is its own group and keeps `spawn_gap[0]`
            # of arc from every race mate that is in the race.
            G_ = full.shape[0]
            group = torch.where(is_full, race, G_ + torch.arange(n, device=self.device))
            empty_rows = (torch.zeros_like(is_full) if empty_g is None else empty_g[race, slot]) & is_full
            active = ~empty_rows & ~(~is_full & self.sim.absent[ids])
            race_kw = dict(group=group, active=active, lat_fixed=abreast_rows, clear_gap=float(e.spawn_gap[0]))
            if bool((~is_full).any()):
                o_ = self.sim.other_idx[ids]
                mate_s = torch.where(self.sim.absent[o_] | is_full[:, None],
                                     torch.full_like(self.sim.s[o_], float("nan")), self.sim.s[o_])
                race_kw["avoid_s"] = mate_s
        poses = self.sim.sample_spawn(n, e.spawn_lateral_std, yaw_std, s=s, tid=self.sim.tid[ids],
                                      min_clearance=min_clear, lat=lat, eid=ids, **race_kw)
        speed = torch.rand(n, device=self.device, generator=gen) * e.spawn_speed_max
        if self._replay is not None and self.M == 2:
            poses, speed, replayed = self._replay.apply(self, ids, poses, speed, gen)
            abreast_rows = abreast_rows & ~replayed          # a recorded scene, not a grid
        self.sim.reset(ids, poses, speed)
        if self.M > 1 and getattr(self, "_present_p", None) is not None:
            # 0-3 opponents: each seat of a race that resets whole is filled with its slot's `present`
            # probability; an empty seat's car is parked off the map until its race resets again
            if empty_g is None:                   # `spawn_as_race` off: drawn here, as it always was
                empty_g = torch.rand(full.shape[0], self.M, device=self.device, generator=gen) >= self._present_p[None, :]
            empty = empty_g[race, slot]
            self.sim.unpark(ids[is_full & ~empty])
            self.sim.park(ids[is_full & empty])
            speed = torch.where(is_full & empty, torch.zeros_like(speed), speed)
        if self.M > 1:
            self._note_spawn_formation(ids, abreast_rows)
        self.sim.odom.state[ids, 3] = speed
        self.prev_action[ids] = 0.0
        if self.act_dim == 2:
            v_lo = self._v_lo()[ids]
            self.prev_action[ids, 1] = (speed - v_lo) / (e.v_max_policy - v_lo) * 2 - 1
            self._calibrate_external(ids)
        else:
            self.prev_action[ids, self.plan_layout.speed] = (speed / e.v_max_policy * 2 - 1)[:, None]
            if self._row_codes is not None and self._pool_codes is None:
                # A car that plans in the opponents' format starts from "hold this speed" in THAT
                # format: a league checkpoint reads its own previous action back as an observation
                held = torch.zeros_like(self.prev_action[ids])
                held[:, _plan_layout(self.opp_plan_spec).speed] = (speed / e.v_max_policy * 2 - 1)[:, None]
                self.prev_action[ids] = torch.where(self._opp_format_rows()[ids][:, None], held, self.prev_action[ids])
            elif self._row_codes is not None:
                # ... and with league entries in formats of their own, each car in its own: the codes
                # are read after this reset's driver draw (`_redraw_slot_league` above), so a seat
                # handed from a linear checkpoint to a wp4 one holds its speed in wp4's layout
                from .mpc import ROW_FORMATS
                codes = self._row_format_codes()[ids]
                v_n = (speed / e.v_max_policy * 2 - 1)[:, None]
                out = self.prev_action[ids]
                for c, mode in enumerate(ROW_FORMATS):
                    held = torch.zeros_like(out)
                    held[:, _plan_layout(mode).speed] = v_n
                    out = torch.where((codes == c)[:, None], held, out)
                self.prev_action[ids] = out
            self.tracker.reset(ids); self._calibrate_tracker(ids)
        self.prev_steer_norm[ids] = 0.0; self.last_cmd[ids] = 0.0; self.last_cmd[ids, 1] = speed
        self._dec_cnt[ids] = 0                                     # a respawned car decides on its first step
        if self.events is not None:
            self.events.reset(ids)                                  # a respawned car starts with no event
        for alt in self.alt_teachers:
            # A planner keeps state across steps on purpose -- which side a pass is committed to,
            # which lane is held. A respawned car is in a different race and must not inherit it.
            reset_rows = getattr(alt, "reset_rows", None)
            if reset_rows is not None:
                reset_rows(ids)
        if self.pool is not None:
            # A pool checkpoint may carry memory: a hidden state kept across a respawn is a policy
            # remembering a track its car is no longer on.
            self.pool.reset(ids)
        for i, drv in self.external_drivers.items():
            # End2Race is a GRU: the same reasoning, per external slot's own rows.
            rows = self.external_slot_rows[i]
            hit = torch.isin(rows, ids)
            if bool(hit.any()):
                drv.reset(hit.cpu().numpy())
        self._was_touching[ids] = False                             # a fresh car is touching nothing
        self._was_touching_wall[ids] = False
        self._car_prev[ids] = False; self._car_resp[ids] = False
        if self.ecfg.contact_severity_min > 0:
            self._wall_pending[ids] = 0.0; self._car_sev_prev[ids] = False; self._car_sev_ok[ids] = False
        self.gap_prev[ids] = 0.0; self.gap_valid[ids] = False       # no gain scored on the first step
        self.act_hist[ids] = self.prev_action[ids][:, None, :]
        for seat in self._seat_builders():
            seat.reset_actions(ids, self.prev_action)
        self.ep_step[ids] = 0; self.ep_return[ids] = 0.0; self.ep_progress[ids] = 0.0
        self.ep_limit[ids] = self.NO_LIMIT; self.ep_contacts[ids] = 0.0
        self.opponent_held[ids] = False                            # a fresh car holds no car ahead
        self.lap_start_step[ids] = 0; self.prev_lap[ids] = 0
        if self.ecfg.reward_lap_time > 0:
            # a car spawns mid-sector, so its first boundary is not a sector it drove: start the clock
            # but withhold the reward until it has crossed one boundary and entered the next cleanly
            s_new = self.sim.s[ids]
            L_ = self.sim.track.length[self.sim.tid[ids]].clamp_min(1e-6)
            self.sector_idx[ids] = (s_new / L_ * self.sector_lim.shape[1]).long().clamp_(0, self.sector_lim.shape[1] - 1)
            self.sector_t0[ids] = 0.0; self.sector_valid[ids] = False; self.s_prev[ids] = s_new
        if not fill:
            return None
        return self._fill_scans(ids, speed)

    def _note_spawn_formation(self, ids: torch.Tensor, abreast: torch.Tensor) -> None:
        """Record which resetting cars were placed abreast (a parked seat is in no row)."""
        self.spawn_abreast[ids] = abreast & ~self.sim.absent[ids]

    def _fill_scans(self, ids: torch.Tensor, speed: torch.Tensor):
        """Fill the scan history of reset rows with a fresh scan from the new pose. Scanned for the whole
        batch on the compiled path (fixed shapes -> one CUDA graph); a per-reset partial batch would run
        eager kernels."""
        e = self.ecfg
        if self.M > 1:
            # With other cars in the scan the whole-batch call never reached the compiled path (`Lidar.scan` takes it
            # only without cars), so it bought nothing and scanned every row to fill a few: 508 ms a reset at 1024
            # envs, 87 % of the reset (measured 2026-10-27). Only the reset rows are scanned now, eager, and returned
            # in whole-batch tensors -- every reader takes `[ids]` (`_reset_result`, the seats, the history below).
            B = self.B
            cut = lambda t: t[ids] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == B else t
            cars = [tuple(cut(x) for x in entry) for entry in self.sim._car_boxes(self.sim.state)]
            P = {k: cut(v) for k, v in self.sim.P.items()}
            sub = self.sim.lidar.scan(self.sim.state[ids, :3], None, P, motion_distortion=False, tid=self.sim.tid[ids],
                                      cars=cars, compiled=False, eid=self.sim.eid[ids])
            scan, scan_true, scan_type = (torch.zeros((B,) + t.shape[1:], device=t.device, dtype=t.dtype).index_copy_(0, ids, t)
                                          for t in sub)
        else:
            scan, scan_true, scan_type = self.sim.lidar.scan(self.sim.state[:, :3], None, self.sim.P, motion_distortion=False, tid=self.sim.tid, cars=None, compiled=True, eid=self.sim.eid)
        self.scan_hist[ids] = self._norm_scan(scan[ids])[:, None, :]
        if self.hist is not None:
            feat = torch.zeros(ids.numel(), 9, device=self.device); feat[:, 0] = speed / e.v_max_policy
            self.hist[ids] = torch.cat([feat, torch.zeros(ids.numel(), self.act_dim, device=self.device)], 1)[:, None, :]
        if self.hist2 is not None:
            self._fresh2[ids] = True                     # filled from the row's first observation
        for seat in self._seat_builders():
            seat.fill(ids, scan, speed)                  # a league seat's own contract, the same fresh scan

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
        if self.ecfg.procedural_shared:
            if n == self.B:
                self.procedural.redraw(self.sim.tid[ids[:1]], torch.arange(self.B, device=self.device),
                                       torch.zeros(self.B, dtype=torch.long, device=self.device))
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
                       current.progress, current.lap, current.car_collision, current.wall_collision,
                       current.wall_impact, current.car_impact):
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
                # A parked car (an empty seat, `Simulator.park`) drives nothing -- the sim zeroes its command -- so its
                # entry is not run for it (`OpponentPool.act` runs each entry on its rows only); a race reset unparks it
                # and clears its state (`pool.reset`) before it drives again.
                drv = (torch.where(self.sim.absent, torch.full_like(self.opp_driver, -1), self.opp_driver)
                       if getattr(self.sim, "absent_any", False) else self.opp_driver)
                with torch.no_grad():
                    out = torch.where(self.pool_driven[:, None], self.pool.act(obs, drv), out)
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
        # The raceline teacher's command, unless every teacher-driven car is driven by another kind:
        # then each of its rows is overwritten below and it would be computed only to be thrown away
        # (with the console's interactive preset that is half of every step).
        an = (self._teacher_normalized(self.teacher, ev_off, ev_speed, follow, v_cap)
              if self._raceline_teacher_needed() else out)
        for kind, alt in zip(self.alt_teacher_kinds, self.alt_teachers):
            rows = self._alt_rows.get(kind) if (self.act_dim != 2 and hasattr(alt, "plan_rows")) else None
            if rows is not None and rows.numel() == 0:
                continue                      # partitioned onto no race at all: nothing to drive
            if rows is not None:
                # Planned for the rows this kind can drive and nothing else (`_alt_rows`); the
                # mask then picks, within them, the rows it drives this race.
                sub = self._teacher_normalized(alt, ev_off, ev_speed, follow, v_cap, rows)
                an = an.index_copy(0, rows, torch.where(self.alt_teacher_mask[kind][rows][:, None], sub, an[rows]))
                continue
            # A teacher kind that is not the raceline teacher drives its own slots. Asked for the
            # whole batch and selected, like the pool is, for the same reason: compacting to the
            # rows it owns would be a device-to-host sync every step. The mask is fixed for the life
            # of the env, so this loop is empty unless a slot actually named such a kind -- and it
            # is what stops a kind the tree has from being silently driven by the raceline teacher
            # the moment its module appears.
            an = torch.where(self.alt_teacher_mask[kind][:, None],
                             self._teacher_normalized(alt, ev_off, ev_speed, follow, v_cap), an)
        return torch.where(self.teacher_driven[:, None], an, out)

    def _note_imu_yaw_rate(self, r) -> None:
        """The last step's gyro yaw rate, (B,), in a tensor whose shape never changes.

        `r.imu` is (B, k, 6) with k the samples that landed in the step -- 1 or 2, alternating,
        because the IMU runs at 50 Hz and the control at 40. Anything that reads it directly reads a
        tensor whose shape changes every other step, and a CUDA graph cannot follow one: the
        interactive teacher's MPC preview did, and its graph went back to eager on the first step
        after capture. Written in place, so the tensor a capture follows is the one being updated.
        None without an IMU, which is what readers already test for.
        """
        if r is None or r.imu is None or r.imu.shape[1] == 0:
            self.imu_yaw_rate = None
            return
        y = r.imu[:, :, 2].mean(1)
        held = getattr(self, "imu_yaw_rate", None)
        if held is None or held.shape != y.shape or held.dtype != y.dtype or held.device != y.device:
            self.imu_yaw_rate = y.clone()
        else:
            held.copy_(y)

    def _raceline_teacher_needed(self) -> bool:
        """Does any teacher-driven car take the raceline teacher's command?

        Only a slot table can say no, and there both masks are fixed for the life of the env
        (`teacher_driven` is redrawn at resets only without a table, `alt_teacher_mask` never), so
        the answer is read back once and kept -- per step it would be a host sync.
        """
        if self.slots is None or not self.alt_teacher_kinds:
            return True
        if self._slot_mixed and any("raceline" in kinds for kinds in self._slot_kinds):
            # A mix that can draw `raceline` can draw it for any of its cars at any reset, and the
            # answer is cached on tensor identity -- which does not change, because the masks are
            # rewritten in place. Rather than invalidate a cache from four places, say yes: the cost
            # is one teacher call on the resets where no car happened to draw it.
            return True
        # A mix WITHOUT `raceline` always hands each of its cars to one of its own kinds, so those
        # rows are covered by an alt mask at every moment and the static check below is exact. It
        # used to say yes for every mix, which in s911's recipe (four planners, none of them the
        # raceline teacher) computed the raceline teacher -- 16 % of an env step, 63 ms of 385 at 256
        # envs -- for a result every row then overwrote.
        key = (id(self.teacher_driven), tuple(id(self.alt_teacher_mask[k]) for k in self.alt_teacher_kinds))
        cached = getattr(self, "_raceline_needed_cache", None)
        if cached is None or cached[0] != key:
            covered = torch.zeros_like(self.teacher_driven)
            for k in self.alt_teacher_kinds:
                covered |= self.alt_teacher_mask[k]
            cached = self._raceline_needed_cache = (key, bool((self.teacher_driven & ~covered).any()))
        return cached[1]

    def _teacher_normalized(self, teacher, ev_off, ev_speed, follow, v_cap, rows=None):
        """One teacher's command for the whole batch, as a normalized action -- or, with `rows`,
        for those rows only, (R, ACT_DIM), from a teacher that can plan a subset (`plan_rows`).

        Split out of `_opponent_actions` so that a second teacher kind is a second call rather than
        a second copy; the instructions are the ones that were inline, in order.
        """
        if rows is not None:
            cut = lambda t: t if t is None or not torch.is_tensor(t) or t.dim() == 0 else t[rows]
            # the opponents' format, which is the tracker's own unless `opponent_speed_mode` says otherwise
            spec, v_max = self.opp_plan_spec, self.ecfg.v_max_policy
            sp = _plan_layout(spec).speed
            an = teacher.plan_rows(rows, self.sim.state, self.sim.P, self.sim.tid, v_max, spec,
                                   offset=ev_off, plan_speed=self._tracker_plan_speed())
            an = an.clone(); an[:, sp] = ((an[:, sp] + 1) * cut(self.opp_scale)[:, None] - 1).clamp(-1, 1)
            if ev_speed is not None:
                an[:, sp] = (cut(ev_speed)[:, None] * (an[:, sp] + 1) - 1).clamp(-1, 1)
            cap_n = (cut(v_cap) / v_max * 2 - 1)[:, None]
            an[:, sp] = torch.where(cut(follow)[:, None], torch.minimum(an[:, sp], cap_n), an[:, sp])
            return teacher.project_rows(an, rows, self.sim.state, self.sim.P, v_max, spec,
                                        plan_speed=self._tracker_plan_speed())
        if self.act_dim == 2:
            cmd = teacher(self.sim.state, self.sim.P, self.sim.tid, offset=ev_off)
            v = cmd[:, 1] * self.opp_scale
            # Before the follow cap, never after: an event can only slow a car (speed_scale <= 1), so
            # taking the minimum of the two leaves an opponent that is already braking for the car
            # ahead braking. Applied the other way round, a "resume" would drive it into that car.
            v = v if ev_speed is None else v * ev_speed
            v = torch.where(follow, torch.minimum(v, v_cap), v)
            return self.teacher_action_to_normalized(torch.stack([cmd[:, 0], v], 1))
        an = self.teacher_label(teacher, offset=ev_off, spec=self.opp_plan_spec)
        sp = _plan_layout(self.opp_plan_spec).speed
        an = an.clone(); an[:, sp] = ((an[:, sp] + 1) * self.opp_scale[:, None] - 1).clamp(-1, 1)   # speed scale
        if ev_speed is not None:                             # same scaling in the normalized plan speeds
            an[:, sp] = (ev_speed[:, None] * (an[:, sp] + 1) - 1).clamp(-1, 1)
        cap_n = (v_cap / self.ecfg.v_max_policy * 2 - 1)[:, None]
        an[:, sp] = torch.where(follow[:, None], torch.minimum(an[:, sp], cap_n), an[:, sp])
        # Scripted pace/events can replace the certified endpoints (including
        # speed multipliers above one). Certify the final teacher plan again;
        # policy-driven rows and their controller are unchanged.
        projector = getattr(getattr(teacher, 'base', teacher), 'project_plan_action', None)
        if projector is not None:
            an = projector(an, self.sim.state, self.sim.P, self.ecfg.v_max_policy,
                           self.opp_plan_spec, plan_speed=self._tracker_plan_speed())
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

        The privileged opponent block is REMOVED. A pool entry is a deployable LiDAR-only
        checkpoint driving the other car -- that is the whole point of the population -- and it was
        trained on the proprio vector without the block. Handing it those columns would not just be
        a shape error: it would make every opponent in the race an oracle too, and the arm would no
        longer be "the learner is told where the other cars are", it would be "everyone is". The
        learner's own rows are unaffected; a `self` entry is the learner's own weights driving off
        the caller's action, which is computed from the full observation for every row.
        """
        obs = self._last_obs
        if obs is None:
            obs = None if self.last_result is None else self._obs(self.last_result)
        if obs is None or self.opp_token == "off":
            return obs
        return {k: v for k, v in obs.items() if k != "opp_token"}

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

    def progress_gate(self, state: torch.Tensor) -> torch.Tensor:
        """(B,) in [gate_min, 1]: 1 when every other car is beyond its closing-speed-dependent safe
        gap, falling linearly to gate_min at body contact. See `EnvConfig.reward_progress_gate`."""
        e = self.ecfg
        gap, closing = self.car_gap_closing(state)                        # (B, C) body gap, closing speed
        safe = e.gate_gap + closing.clamp_min(0.0) * e.gate_tau
        danger = ((safe - gap) / safe.clamp_min(1e-3)).clamp(0.0, 1.0)
        # a car clearly behind (more than a car length) closing on me is its business, not mine
        behind = self.signed_gaps(self.sim.s, self.sim.tid) < -0.6
        danger = torch.where(behind, torch.zeros_like(danger), danger).amax(1)
        return 1.0 - (1.0 - e.gate_min) * danger

    def _interaction_penalties(self, state: torch.Tensor, car_hit: torch.Tensor):
        """(rear-end penalty, merge penalty), (B,) each, <= 0. See `EnvConfig.reward_rear_end`."""
        e = self.ecfg
        j = self.sim.other_idx[:, 0]
        c, s_ = torch.cos(state[:, 2]), torch.sin(state[:, 2])
        vw = torch.stack([state[:, 3] * c - state[:, 4] * s_, state[:, 3] * s_ + state[:, 4] * c], 1)
        pen_r = torch.zeros_like(state[:, 0]); pen_m = torch.zeros_like(state[:, 0])
        if e.reward_rear_end > 0:
            prev = getattr(self, "_rear_prev", None)
            if prev is None:
                prev = self._rear_prev = torch.zeros_like(car_hit)
            gap = self.signed_gaps(self.sim.s, self.sim.tid)[:, 0]
            onset = car_hit & ~prev & (gap > 0.3)                        # the other car is ahead of me
            dv2 = (vw - vw[j]).square().sum(1)
            pen_r = torch.where(onset, -e.reward_rear_end * dv2, pen_r)
            self._rear_prev = car_hit.clone()
        if e.reward_merge > 0:
            co, so = torch.cos(state[j, 2]), torch.sin(state[j, 2])
            d = state[:, :2] - state[j, :2]                              # me relative to it
            along = d[:, 0] * co + d[:, 1] * so                          # + : I am ahead of it
            side = -d[:, 0] * so + d[:, 1] * co                          # + : I am on its left
            v_side = -vw[:, 0] * so + vw[:, 1] * co                      # my own lateral velocity in its frame
            toward = (-torch.sign(side) * v_side).clamp_min(0.0)
            near = (along > -0.3) & (along < 1.2) & (side.abs() < 1.0)
            pen_m = torch.where(near & self.learner, -e.reward_merge * toward * self.sim.control_dt, pen_m)
        return pen_r, pen_m

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

    def _severity_gate(self, r: StepResult, crash: torch.Tensor) -> dict:
        """`contact_severity_min`: which of this step's contact onsets are hard enough to charge.

        Walls and props: `crash` (the onsets of `wall_collision`) at or above the threshold, each with the
        impact carried from its physical contact's start (`contact_severity.carry_wall_impact`; the flag
        can rise steps after the hit). Cars: a contact is charged for its whole length if its first step's
        closing speed (`car_impact`) reaches the threshold, decided once like `car_charge_latch` decides
        who pays -- `car_basis` is the per-step contact flag with the soft ones taken out, which is what
        `_car_charge` then charges. Returns the onsets both ways for the log."""
        from .contact_severity import after_onset, carry_wall_impact
        v_min = float(self.ecfg.contact_severity_min)
        if r.contact_onsets is not None:
            # onsets per kind, each with its own step's impact (the flag rises on the step of the hit)
            on, imp = r.contact_onsets, r.contact_impacts
            hard = (on[:, 0] & (imp[:, 0] >= v_min)) | (on[:, 1] & (imp[:, 1] >= v_min))
            out = {"wall_all": crash, "wall_hard": crash & hard}
        else:
            carry = carry_wall_impact(self._wall_pending, r.wall_impact, r.wall_collision)
            self._wall_pending = after_onset(carry, crash)
            out = {"wall_all": crash, "wall_hard": crash & (carry >= v_min)}
        if r.car_collision is not None:
            hit = r.car_collision
            onset = hit & ~self._car_sev_prev
            hard = onset & (r.car_impact >= v_min)
            self._car_sev_ok = torch.where(onset, hard, self._car_sev_ok) & hit
            self._car_sev_prev = hit.clone()
            out.update(car_all=onset, car_hard=hard, car_basis=self._car_sev_ok.to(self.ep_return.dtype))
        return out

    def _car_charge(self, s: torch.Tensor, tid: torch.Tensor, car_hit: torch.Tensor) -> torch.Tensor:
        """Per car, 1 while a contact it is charged for lasts. `car_contact_charge` decides who is charged; with
        `car_charge_latch` it decides once, at the step the contact begins, and the verdict holds until the
        cars separate."""
        if self.sim.other_idx is None:
            return car_hit
        if not self.ecfg.car_charge_latch:
            return self.car_contact_charge(s, tid, car_hit)
        hit = car_hit > 0
        onset = hit & ~self._car_prev
        verdict = self.car_contact_charge(s, tid, torch.ones_like(car_hit)) > 0
        self._car_resp = torch.where(onset, verdict, self._car_resp) & hit
        self._car_prev = hit
        return self._car_resp.to(car_hit.dtype)

    # ------------------------------------------------------------------ API
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.sim.gen.manual_seed(seed); torch.manual_seed(seed)
        self._reset_envs(torch.arange(self.B, device=self.device))
        if self.ecfg.stagger_first_episode:
            # One draw per race, not per car: the leader's limit ends the whole race (see the
            # truncation in `_step_math`), so the cars of a race have to share it.
            n = int(self.ecfg.max_steps)
            per_race = torch.randint(1, n + 1, (self.B // self.M,), generator=self.sim.gen, device=self.device)
            self.ep_limit.copy_(per_race.repeat_interleave(self.M))
        # one zero-action step so that odom/scan come from the simulator path
        r = self.sim.step(torch.stack([torch.zeros(self.B, device=self.device), self.sim.state[:, 3]], 1))
        self.scan_hist = torch.roll(self.scan_hist, 1, 1); self.scan_hist[:, 0] = self._norm_scan(r.scan)
        self.last_result = r
        self._note_imu_yaw_rate(r)
        self._plan_fallback(torch.arange(self.B, device=self.device), r.state)
        self.lead_steps.zero_(); self.lead_paid.fill_(1.0)
        obs = self._obs(r)
        if self.hist is not None:                                  # history starts from the first real observation (as the car's builder does)
            self.hist[:] = torch.cat([self._last_feat, torch.zeros(self.B, self.act_dim, device=self.device)], 1)[:, None, :]
            obs = self._obs(r)
        for seat in self._seat_builders():
            seat.begin(r)                                          # the same start, in each league contract
        if self._opp_age is not None:
            self._update_opp_visibility(r, torch.ones(self.B, dtype=torch.bool, device=self.device))
        self._last_obs = obs
        return obs, {"priv": self._priv(r)}

    def step(self, action: torch.Tensor):
        ext = float(self.ecfg.reverse_ext)
        if ext > 0 and self.act_dim == 2:
            # the learners' speed may reach into the reverse extension; everything else stays in [-1, 1]
            act_in = action.to(self.device)
            lo = torch.where(self.learner, torch.full((self.B,), -1.0 - ext, device=self.device), torch.full((self.B,), -1.0, device=self.device))
            a = torch.stack([act_in[:, 0].clamp(-1.0, 1.0), torch.maximum(torch.minimum(act_in[:, 1], torch.ones_like(lo)), lo)], 1)
            a = self._opponent_actions(a)
        else:
            a = self._opponent_actions(action.to(self.device).clamp(-1.0, 1.0))
        if self.act_dim == 2:
            v_lo = self._v_lo()
            v_cmd = torch.minimum(v_lo + (a[:, 1] + 1) * 0.5 * (self.ecfg.v_max_policy - v_lo), self.speed_cap)
            if ext > 0:
                v_rev = -float(self.ecfg.reverse_speed) * (-(a[:, 1] + 1.0) / ext)
                v_cmd = torch.where(a[:, 1] < -1.0, v_rev, v_cmd)
            cmd = torch.stack([a[:, 0] * self.s_max, v_cmd], 1)
        elif self._all_external:
            # Every car is driven by an external driver whose command replaces this one below, so
            # nothing reads the tracker's answer and it is not solved: a baseline alone in the console
            # paid ~5 ms a step for a plan nobody drove (2026-10-19, "simtime이 왜이리 느리지").
            cmd = torch.zeros(self.B, 2, device=self.device)
            self.last_cmd_raw = cmd
        else:                                                  # plan -> (steer, speed) through the tracker
            lr = self.last_result
            v_meas = self._tracker_plan_speed()
            yaw_rate = lr.imu[:, :, 2].mean(1) if (lr is not None and lr.imu is not None and lr.imu.shape[1] > 0) else None
            # Before the solve, because `last_pred` comes back in the body frame of exactly this
            # pose and `car_future` has to put it back in the world.
            self._plan_pose = self.sim.state[:, :3].clone()
            self._sync_row_formats()
            if self.ecfg.plan_follower == "map":
                # MAP reads the car's grip level and the IMU's longitudinal acceleration (race_stack's
                # `acc_now`), and follows the plans of the policy's own cars
                self.tracker.P = self.sim.P
                self.tracker.map_rows = self.learner
                if lr is not None and lr.imu is not None and lr.imu.shape[1] > 0:
                    self.tracker.push_imu(lr.imu[:, :, 3])
            K = int(self.ecfg.decision_every)
            if K > 1:
                # Between decisions the learner's cars drive the plan they last issued, held on the track
                # (`EnvConfig.decision_every`): its action stands for every reader below, and MAP follows its
                # waypoints from the pose it was issued at. The traffic decides every step.
                hold = self.learner & ((self._dec_cnt % K) != 0)
                if self._held_action is None:
                    self._held_action = a.detach().clone()
                a = torch.where(hold[:, None], self._held_action, a)
                self._held_action = a.detach().clone()
                self.tracker.hold_rows = hold
                self.tracker.pose_now = self.sim.state[:, :3]
                self._dec_cnt += 1                                 # a reset below sets its rows back to 0
            a_trk = a if self.ecfg.plan_speed_scale == 1.0 else self._scale_plan_speeds(a)
            raw = self.tracker(a_trk, v_meas, self.speed_cap, yaw_rate, delay=self.tracker_delay)
            # Before the physics: `sim.state` is still the pose the plan was issued from, which is
            # the frame `last_ref` lives in.
            self._capture_plan(self.sim.state)
            if self.ecfg.tracker_speed_ff:
                # The inverse of the VESC's first-order loop, accel = (cmd - v) / tau: to get the
                # acceleration the tracker planned, command the speed it will be at when the command
                # lands (`last_pred` starts one latency ahead) plus tau times that acceleration.
                from .mpc import feedforward_speed_command
                v_ff = feedforward_speed_command(self.tracker.last_pred[:, 0, 3], self.tracker.u_seq[:, 0, 1],
                                                 self.tracker_motor_tau)
                raw = torch.stack([raw[:, 0], torch.where(raw[:, 1] < 0, raw[:, 1],
                                   torch.minimum(v_ff, self.speed_cap))], 1)
            if self.ecfg.speed_command == "profile":
                # The reference is time-indexed from now at the tracker's dt, starting at the
                # measured speed and walked towards the plan's profile within the tracker's own
                # drive / brake bounds. Read it where the command lands (the VESC's delay plus the
                # zero-order hold of one control step) and add tau x its slope, so the first-order
                # loop reaches the profile instead of trailing it by tau.
                from .mpc import profile_speed_command
                t_land = (self.sim.P["cmd_delay"] + self.sim.control_dt) / self.tracker.spec.dt
                v_pr = profile_speed_command(self.tracker.last_ref[:, :, 3], self.tracker.spec.dt, t_land,
                                             self.tracker_motor_tau)
                if self.ecfg.speed_command_tau > 0:
                    k_f = self.sim.control_dt / (self.ecfg.speed_command_tau + self.sim.control_dt)
                    prev = getattr(self, "_v_pr_prev", None)
                    if prev is None or prev.shape != v_pr.shape:
                        prev = v_meas.clone()
                    # a car that has just been reset starts its filter from where it is
                    prev = torch.where(self.ep_step <= 1, v_meas, prev)
                    v_pr = prev + k_f * (v_pr - prev)
                    self._v_pr_prev = v_pr.detach()
                raw = torch.stack([raw[:, 0], torch.where(raw[:, 1] < 0, raw[:, 1],
                                   torch.minimum(v_pr, self.speed_cap))], 1)
            if self.ecfg.opponent_speed_limit and self.opponent_reading is not None:
                # the speed limit for a car ahead, from the policy's own reading of it (`opponent_speed_limit.py`); the
                # LiDAR's nominal mount, as the car knows it
                from dataclasses import replace as _replace
                from .opponent_speed_limit import DEFAULT as _OPP, speed_limit
                lim = speed_limit(self.opponent_reading, self.sim.state[:, 3], float(self.cfg.lidar.mount_x),
                                  float(self.cfg.lidar.mount_y), held=self.opponent_held,
                                  p=_replace(_OPP, lag=float(self.ecfg.speed_limit_lag)))
                self.opponent_held = torch.isfinite(lim)
                v = raw[:, 1]
                held = self.learner & (v > lim.to(v.dtype))
                self.opponent_limited = held
                raw = torch.stack([raw[:, 0], torch.where(held, lim.to(v.dtype), v)], 1)
            if self.ecfg.obstacle_speed_limit and lr is not None and lr.scan is not None:
                # the speed limit for an obstacle on the plan's own path (`obstacle_speed_limit.py`), on the scan the
                # policy planned from, the plan decoded as the tracker decodes it, the LiDAR's nominal mount
                from dataclasses import replace as _replace
                from .obstacle_speed_limit import DEFAULT as _OBS, speed_limit as obstacle_limit
                from . import mpc as _mpc
                spec_t = self.tracker.spec
                dec = _mpc.decode_wp4 if getattr(spec_t, "speed_mode", "linear") == "wp4" else _mpc.decode
                k_o, L_o = dec(a_trk.float(), v_meas.float(), float(self.tracker.v_max), self.speed_cap.float(), spec_t)[:2]
                ang_o = self.sim.lidar.angles
                lim_o = obstacle_limit(lr.scan, ang_o, k_o, L_o, self.sim.state[:, 3],
                                       float(self.cfg.lidar.mount_x), float(self.cfg.lidar.mount_y),
                                       car_reading=self.opponent_reading, range_max=float(self.cfg.lidar.range_max),
                                       p=_replace(_OBS, lag=float(self.ecfg.speed_limit_lag)))
                v = raw[:, 1]
                held_o = self.learner & (v > lim_o.to(v.dtype))
                self.obstacle_limited = held_o
                raw = torch.stack([raw[:, 0], torch.where(held_o, lim_o.to(v.dtype), v)], 1)
            self.last_cmd_raw = raw                                # what the tracker asked for (before calibration)
            cal = self.tracker_cal
            cmd = torch.stack([((raw[:, 0] - cal[:, 0]) / cal[:, 1]).clamp(-self.s_max, self.s_max), raw[:, 1] / cal[:, 2]], 1)
        if self.cmd_shaper is not None:
            # A controller sitting between the policy and the VESC: `learn/traction_arm.py` is the
            # one that exists, and it shapes the speed the way `policy_node` shapes `/drive` on the
            # car. Before the external override on purpose -- a teleoperated or ROS-driven car is
            # the mux output, which the node does not shape either.
            cmd = self.cmd_shaper(cmd)
        if self.external_drivers:
            cmd = self._external_commands(cmd)
        if self._ext_ids:
            cmd = torch.where(self.ext_mask[:, None], self.ext_cmd, cmd)
        steer_norm = cmd[:, 0] / self.s_max
        r = self.sim.step(cmd)
        self.last_cmd = cmd
        e = self.ecfg
        self.ep_step += 1
        if self.tracker is not None:
            # What the tracker needs to know for `reverse_cmd_gate`: a car asking to stop while it
            # is already stopped is only asking to back out if it is stuck on something. Taken
            # before the edge-detect below, because that rewrites `r.collision` to the onset and
            # this wants the whole contact -- the car is wedged for the duration, not for a step.
            self.tracker.contact = (r.collision if e.collision_mode == "soft"
                                    else torch.zeros_like(r.collision))
        if e.collision_mode == "soft":
            # Charge the crash once, when it starts. `collided` is now per step rather than
            # latched, but a car scraping a hose is in contact for the measured 40 ms -- sixteen
            # steps of it at 40 Hz -- and a penalty designed as a one-off ending would be paid
            # sixteen times. The rising edge is the impact; the rest is the same impact.
            # With the simulator's onsets per kind (`SimParams.wall_latch`) a contact of one kind that
            # starts during another -- a crate hit while scraping a hose -- is an onset too.
            on = r.contact_onsets
            edge = (r.collision & ~self._was_touching) if on is None else on.any(1)
            self._was_touching = r.collision.clone()
            r.collision = edge
            self.ep_contacts += edge.to(self.ep_contacts.dtype)
            if e.collision_counts_cars or r.wall_collision is None:
                crash = edge
            else:
                crash = (r.wall_collision & ~self._was_touching_wall) if on is None else (on[:, 0] | on[:, 1])
                self._was_touching_wall = r.wall_collision.clone()
        else:
            crash = r.collision
        # `contact_severity_min`: an onset slower than it is not charged (`_severity_gate`). 0 skips all of it.
        gate = self._severity_gate(r, crash) if e.contact_severity_min > 0 else None
        if gate is not None:
            crash = gate["wall_hard"]
        self._widen_clearance_to_props(r)
        if self.procedural is not None and e.movable_obstacles and not self.sim._props_move_in_roll():
            # Whatever the contacts shoved this step now slides, and the floor takes it back down.
            # After the roll, so a prop moves between steps and not inside one: the LiDAR and the
            # contact test in this step both saw it where it was when the step began. Only without
            # `SimParams.prop_relative`, which moves the props inside the roll instead.
            self.procedural.advance(self.sim.control_dt)
        plan_ref = self.tracker.last_ref if self.tracker is not None and e.reward_plan_clearance > 0 else self._no_plan
        car_hit = (r.car_collision.float() if r.car_collision is not None
                   else torch.zeros_like(self.ep_return))
        if self._replay is not None:
            self._replay.record(self, car_hit > 0)
        # Under `contact_severity_min` only the hard contacts reach the charge; the replay above and the
        # rear-end / merge terms below still see every contact.
        car_charge = self._car_charge(r.s, self.sim.tid, car_hit if gate is None else gate.get("car_basis", car_hit))
        out = self._math(r.scan, r.wall_dist, r.s, r.state, r.progress, crash, r.lap, a, steer_norm, self.prev_steer_norm,
                         self.scan_hist, self.act_hist, self.ep_step, self.sim.tid, self.ep_return, self.ep_progress, self.prev_lap,
                         plan_ref, self.lap_start_step, car_charge, self.gap_prev, self.gap_valid,
                         self.lead_steps, self.lead_paid,
                         self._plan_clearance(r.state, plan_ref), self.ep_limit)
        (self.scan_hist, steer_rate, reward, reward_components, self.act_hist, terminated, truncated,
         crossed, done, self.ep_return, self.ep_progress, flags, self.gap_prev,
         self.gap_valid, self.lead_steps, self.lead_paid) = (t.clone() for t in out)
        self.prev_steer_norm = steer_norm; self.prev_action = a
        if self.hist is not None:
            self.hist = torch.cat([torch.cat([self._last_feat, a], 1)[:, None, :], self.hist[:, :-1]], 1)
        if self.hist2 is not None:
            # the row the observation `a` was decided from, and `a` itself: what the car felt, what it was told
            self.hist2 = torch.cat([torch.cat([self._last_feat2, a], 1)[:, None, :], self.hist2[:, :-1]], 1)
        obs = self._obs(r)
        for seat in self._seat_builders():
            seat.advance(r, a)                                     # each league contract's own histories
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
        if e.reward_grip_budget > 0 and self.grip_budget is not None:
            # |v r|: the lateral acceleration the path itself demands, from the state (the emulated
            # accelerometer carries 3 m/s^2 rms of vibration and would make this a noise penalty).
            allowed = e.grip_budget_rho * self.grip_budget * self.cfg.vehicle.mu_f_scale * 9.81
            over = ((r.state[:, 3] * r.state[:, 5]).abs() - allowed).clamp_min(0.0)
            pen = -e.reward_grip_budget * over * over * self.sim.control_dt
            reward += pen
            reward_components[:, REWARD_COMPONENT_KEYS.index("grip_budget")] = pen
        if e.reward_progress_gate > 0 and self.sim.other_idx is not None:
            g = self.progress_gate(r.state)
            ip = REWARD_COMPONENT_KEYS.index("progress")
            pc = reward_components[:, ip]
            # only positive progress is gated: a car going backwards keeps its full cost
            gated = torch.where(pc > 0, pc * g, pc)
            reward += gated - pc
            reward_components[:, ip] = gated
        if (e.reward_rear_end > 0 or e.reward_merge > 0) and self.sim.other_idx is not None:
            pen_r, pen_m = self._interaction_penalties(r.state, car_hit > 0)
            reward += pen_r + pen_m
            reward_components[:, REWARD_COMPONENT_KEYS.index("rear_end")] = pen_r
            reward_components[:, REWARD_COMPONENT_KEYS.index("merge")] = pen_m
        self.prev_lap = r.lap.clone()
        info = {"priv": self._priv(r), "progress": r.progress, "lap": r.lap, "wall_dist": r.wall_dist,
                # a snapshot, like track_id below: _reset_envs re-draws the roles in place further
                # down this same step, and the trainer reads this after step() has returned
                "on_policy": self.on_policy.clone(),
                "scan_true": r.scan_true, "track_id": self.sim.tid.clone(), "lap_times": lap_times, "lap_ids": lap_ids,
                "learner": self.learner,
                # A collision this step, counted once: the onset of a soft contact, or the crash
                # that ends a `terminate` episode. What to count per step, in either mode --
                # `terminated` is never set under soft, and an evaluation that counted it (or the
                # episodes' `collided`) read every soft run as collision-free.
                "contact_onset": r.collision.clone(),
                # what the `collision` component charged this step: wall/prop onsets (or, with
                # `collision_counts_cars`, every onset)
                "wall_onset": crash.clone(),
                "reward_components": dict(zip(REWARD_COMPONENT_KEYS, reward_components.unbind(1)))}
        if r.car_collision is not None:
            info["car_collision"] = r.car_collision
            info["car_charge"] = car_charge
        if gate is not None:
            # the onsets both ways, for the trainer's log: every one, and those at or above the threshold
            # (`wall_onset` above is the charged, hard one)
            info["wall_onset_all"] = gate["wall_all"]
            if "car_all" in gate:
                info["car_onset_all"] = gate["car_all"]
                info["car_onset_hard"] = gate["car_hard"]
        if self.obs_version == 2:
            # Critique B5: the v2 policy does not observe the speed cap (the env's own: DAgger's fixed
            # --speed-cap, PPO's curriculum), and the plan decode clips every speed at it -- on a
            # clipped step the speed dimensions get no reward gradient and can drift. Which steps
            # those were, for the trainer to count.
            lay = self.plan_layout
            v_plan = (a[:, lay.speed] + 1.0) * 0.5 * e.v_max_policy
            info["speed_capped"] = (v_plan > self.speed_cap[:, None]).any(1)
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
                             "lap_time": self.ep_step[ids].float() * self.sim.control_dt,
                             # what the episode ran into: its soft contacts, or the one that ended it
                             "contacts": (self.ep_contacts[ids].clone() if e.collision_mode == "soft"
                                          else terminated[ids].to(self.ep_contacts.dtype)),
                             # cut short on purpose (`stagger_first_episode`): not a sample of how
                             # far an episode gets, so a trainer leaves it out of its episode stats
                             "staggered": self.ep_limit[ids] < int(e.max_steps)}
            # final observation of the ended episodes (before auto-reset)
            info["final_obs"] = {k: v[ids].clone() for k, v in obs.items()}
            info["final_priv"] = self.privileged(r)[ids].clone()
            if self.obs_version == 2:
                # the v2 critic's vector of the same terminal observation, for the truncation bootstrap
                info["final_priv_v2"] = self.privileged_v2(r)[ids].clone()
            scans = self._reset_envs(ids)
            r = self._reset_result(r, ids, scans)
            self._plan_fallback(ids, r.state)
            obs = self._obs(r)
            for seat in self._seat_builders():
                seat.refresh(r)                                    # the reset rows' first observation
        if self._opp_age is not None:
            # after the resets: the age describes the observation the next action is taken from
            self._update_opp_visibility(r, done if any_done else None)
        self.last_result = r
        self._note_imu_yaw_rate(r)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    #: [m] the plan-clearance term probes at least this far along the path, and samples it at most
    #: this far apart. Both numbers exist because the tracker's reference is *time* indexed -- N*dt
    #: = 0.6 s of it, whatever the speed -- and that is the wrong axis for "is there something in
    #: the way":
    #:
    #: * at 9 m/s the 13 reference points are 0.45 m apart and a crate is 0.24-0.42 m wide, so the
    #:   path could straddle one and score clear. Measured: 8.4 % of the cases where a crate was
    #:   really inside the margin.
    #: * at 1.3 m/s the whole reference is 0.78 m long -- less than a car -- so a crate a metre
    #:   ahead was outside it, the footprint was outside `safe_dist`, and the only live term was
    #:   progress. That is the creep this whole diagnosis started from, and the earlier fix did not
    #:   close it: making the terms *see* props does nothing if neither is looking that far.
    #:
    #: 3 m is the distance a car doing 9 m/s covers in a third of a second and one it can stop in
    #: from 4.9 m/s; 0.15 m is half a body width, the same rule the spawn runway samples on.
    PLAN_PROBE_M = 3.0
    PLAN_SAMPLE_M = 0.15

    def _plan_clearance(self, state, plan_ref):
        """(B,) room the committed path has, walls and props, or None when the term is off.

        Replaces the distance-field sampling that used to live in `_step_math`, for three reasons
        at once: it has to see the props (which are not in that field), it has to sample finely
        enough to not straddle a crate, and it has to look a fixed distance rather than a fixed
        time. Computed here and passed in because `_step_math` is compiled under
        `reduce-overhead`, and a topk over layout slots inside an inductor trace is a graph break
        at best.

        The car's own width is *not* subtracted here -- the caller does that, as it always did.
        """
        e = self.ecfg
        if e.reward_plan_clearance <= 0 or plan_ref is None or plan_ref.shape[1] <= 1:
            return self._no_plan_prop
        ref = plan_ref[:, :, :2]
        B, P, _ = ref.shape
        # Densify and extend, in the body frame, before going to world: the reference is a polyline
        # and its own last heading is the direction the plan was still going in.
        span = ref[:, -1].norm(dim=1).clamp_min(1e-3)                        # (B,) how far it reaches
        n_s = max(P, int(math.ceil(float(span.max().item()) / self.PLAN_SAMPLE_M)) + 1)
        u = torch.linspace(0, P - 1, n_s, device=ref.device)
        lo = u.floor().long().clamp(max=P - 1); hi = u.ceil().long().clamp(max=P - 1)
        w = (u - lo).view(1, -1, 1)
        pts = ref[:, lo] * (1 - w) + ref[:, hi] * w                          # (B, n_s, 2)
        # ... and carry on in the plan's final direction until `PLAN_PROBE_M` is covered, so a
        # crawling car is not blind to what it is crawling towards.
        tail = ref[:, -1] - ref[:, -2]
        tail = tail / tail.norm(dim=1, keepdim=True).clamp_min(1e-6)
        extra = (self.PLAN_PROBE_M - span).clamp_min(0.0)                    # (B,)
        n_e = int(math.ceil(self.PLAN_PROBE_M / self.PLAN_SAMPLE_M))
        frac = torch.linspace(0, 1, n_e, device=ref.device).view(1, -1, 1)
        ext = ref[:, -1:, :] + tail[:, None, :] * (extra[:, None, None] * frac)
        pts = torch.cat([pts, ext], 1)
        c, sn = torch.cos(state[:, 2])[:, None], torch.sin(state[:, 2])[:, None]
        px = state[:, 0:1] + pts[:, :, 0] * c - pts[:, :, 1] * sn
        py = state[:, 1:2] + pts[:, :, 0] * sn + pts[:, :, 1] * c
        xy = torch.stack([px, py], -1)
        n = xy.shape[1]
        out = self.sim.track.sample_edt(xy, self.sim.tid[:, None]).min(1).values
        if self.procedural is not None:
            eid = torch.arange(B, device=xy.device)[:, None].expand(B, n).reshape(-1)
            prop = self.procedural.clearance(xy.reshape(-1, 2), eid).view(B, n).min(1).values
            out = torch.minimum(out, prop.to(out.dtype))
        return out

    def _step_math(self, scan, wall_dist, s, state, progress, collision, lap, a, steer_norm, prev_steer_norm, scan_hist, act_hist, ep_step, tid,
                   ep_return, ep_progress, prev_lap, plan_ref, lap_start_step, car_hit, gap_prev, gap_valid,
                   lead_steps, lead_paid, plan_clear, ep_limit):
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
            # `plan_clear` is `_plan_clearance`: the same question this used to ask of the distance
            # field alone, asked of the props as well, sampled at half a body width and probed a
            # fixed distance rather than a fixed 0.6 s. The car's own width is subtracted here, as
            # it always was -- the path is a centreline and the body is 0.31 m wide.
            clearance = plan_clear - 0.5 * self.cfg.vehicle.width
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
        # Time to contact: a per-step toll that rises linearly from 0 at `ttc_safe` to `reward_ttc`
        # at the contact itself. Per step and not per metre driven, unlike the two proximity terms:
        # what is being priced is the time left, and standing still beside a car that is closing on
        # you is exactly as close to a contact as driving past it is.
        ttc_pen = torch.zeros_like(progress)
        if e.reward_ttc > 0 and self.sim.other_idx is not None:
            ttc_pen = ((e.ttc_safe - self.car_ttc(state)) / max(e.ttc_safe, 1e-6)).clamp(0.0, 1.0)
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
            -e.reward_car_contact * car_hit,     # already the charge: `_car_charge` in step()
            e.reward_overtake * gap_gain,
            torch.zeros_like(progress),          # lap_time: filled in on the crossing, in step()
            -e.reward_car_proximity * car_prox * travelled,
            -e.reward_sideslip * (torch.atan2(state[:, 4].abs(), state[:, 3].abs().clamp_min(0.5)) - e.sideslip_free).clamp_min(0.0) * travelled,
            -e.reward_ttc * ttc_pen,
            torch.zeros_like(progress),          # overtake_hold: filled in below, once the new gaps
                                                 # and the episode boundaries of this step are known
            torch.zeros_like(progress),          # grip_budget: filled in by step(), from a tensor the trainer owns
            torch.zeros_like(progress),          # rear_end: filled in by step() (`_interaction_penalties`)
            torch.zeros_like(progress),          # merge: likewise
        ], 1)
        act_hist = torch.cat([a[:, None, :], act_hist[:, :-1]], 1)
        # Under "soft" a contact is a cost, not an ending: the simulator has already pushed the car
        # out of it and taken the into-surface velocity, and the episode carries on so the policy
        # can be taught what to do next. `crash` above still charges for it, and still scales with
        # the speed at impact, so a nudge and a shunt are not the same price.
        terminated = collision.clone() if e.collision_mode == "terminate" else torch.zeros_like(collision)
        truncated = (~terminated) & ((ep_step >= ep_limit.clamp(max=e.max_steps)) | (lap >= e.laps))
        if self.M > 1:                                         # the leader's time limit ends the whole race
            truncated = truncated | (truncated & (self.slot == 0)).view(-1, self.M)[:, 0].repeat_interleave(self.M)
            truncated = truncated & ~terminated
        crossed = lap > prev_lap
        done = terminated | truncated
        flags = torch.stack([done.any(), crossed.any()])
        gap_now = torch.zeros_like(gap_prev)
        if self.sim.other_idx is not None:
            gap_now = self.signed_gaps(s, tid)
        # Sustained lead. It needs the gaps AFTER this step and the episode boundaries OF this step,
        # so it is the one reward term that cannot be filled in with the others above.
        if e.reward_overtake_hold > 0 and self.sim.other_idx is not None:
            hold, lead_steps, lead_paid = self.overtake_hold(gap_now, gap_valid,
                                                            self.race_boundary(done), lead_steps, lead_paid)
            reward_components[:, REWARD_COMPONENT_KEYS.index("overtake_hold")] = e.reward_overtake_hold * hold
        reward = reward_components.sum(1)
        return (scan_hist, steer_rate, reward, reward_components, act_hist, terminated, truncated, crossed,
                done, ep_return + reward, ep_progress + progress, flags, gap_now,
                torch.ones_like(gap_valid), lead_steps, lead_paid)

    #: `ep_limit` of a row that is not in a staggered first episode: never reached.
    NO_LIMIT = 2 ** 62

    def signed_gaps(self, s: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """Arc to each opponent, signed and wrapped to (-L/2, L/2]: + is ahead of me, - is behind.

        Signed, because the forward arc jumps from ~0 to ~L the instant a pass completes and any
        reward reading its change sees a whole lap lost at the one moment that mattered.
        """
        L_ = self.sim.track.length[tid][:, None]
        d = (s[self.sim.other_idx] - s[:, None] + L_ / 2) % L_ - L_ / 2
        if self._present_mask_gaps:
            # An empty seat is parked 1 km off the map, but the lane projection only searches 3 m of
            # arc around its last index, so its `s` stays on the lap -- at its grid slot, then
            # drifting to a local minimum. Measured (2026-10-14, boot-camp config): the parked seat
            # was the nearest car by arc in 1,711 of 3,941 live row-steps of races with an empty
            # seat, and the "car" a teacher held 0.5 m/s behind in 616 of 1,190 follow-gap
            # triggers. Half a lap is as far as a signed arc goes; nothing reads it as near.
            # Decided at construction (any seat with `present` < 1), not on `absent_any`: a Python
            # flag that flips at resets would be a guard the compiled step math recompiles on.
            d = torch.where(self.sim.absent[self.sim.other_idx], L_.expand_as(d) / 2, d)
        return d

    def car_gap_closing(self, state: torch.Tensor):
        """(body-to-body gap [m], line-of-sight closing speed [m/s]) to every opponent, (B, C) each.

        Body-to-body clearance from an oriented box, not centre distance minus a constant. Cars are
        0.5 long and 0.3 wide: alongside, the bodies touch at 0.3 m between centres, so a
        centre-distance term with a 0.45 m "body" was already saturated at full penalty with 0.3 m
        of daylight between the cars, and could not tell a rub from a clean pass. Measured on the
        race leg: 100 % of the learner's contacts were side contacts, and 4M steps of training
        against that saturated term moved the contact count by nothing.

        `closing` is the rate the *line of sight* is shortening: the relative world velocity
        projected onto the unit vector from this car to that one, positive while the gap shrinks.
        Both the proximity penalty and the time-to-contact penalty are functions of exactly this
        pair, so they are computed once here rather than twice with two chances to drift apart.
        """
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
        lon = (d[..., 0] * c[:, None] + d[..., 1] * sn[:, None]).abs()
        lat = (-d[..., 0] * sn[:, None] + d[..., 1] * c[:, None]).abs()
        gap = torch.sqrt((lon - e.car_len).clamp_min(0.0) ** 2 + (lat - e.car_wid).clamp_min(0.0) ** 2)
        return gap, closing

    def car_proximity(self, state: torch.Tensor) -> torch.Tensor:
        """(B,) closeness to the most threatening opponent: 0 outside car_safe_gap of its body, 1 at
        contact, scaled up by the speed the gap is closing at. Euclidean, not arc: the overtake term
        reads arc so a pass can go through zero arc gap, and this reads the body distance so the pass
        has to go *around* the other car rather than through it."""
        e = self.ecfg
        gap, closing = self.car_gap_closing(state)
        closeness = (1.0 - gap / e.car_safe_gap).clamp(0.0, 1.0)
        scale = 1.0 + closing.clamp_min(0.0) / e.car_prox_speed_ref
        return (closeness * scale).max(1).values

    def car_ttc(self, state: torch.Tensor) -> torch.Tensor:
        """(B,) seconds until the bodies touch at the current closing speed; `TTC_FAR` if nothing closes.

        The quantity the reward audit says is missing. `car_proximity` prices a gap that is already
        small and charges it per metre driven; a collision prices the contact itself, once, at 0.2 %
        of steps. Neither sees the situation the traffic attribution actually found -- side by side
        at +1.5 m/s of closing speed with half a metre of daylight, which is 0.3 s from a contact and
        costs nothing until it happens. TTC is gap over closing speed, so it is small exactly when
        geometry and closing speed together say the contact is coming, at any gap.

        Divergent and parallel pairs (`closing <= 0`) get `TTC_FAR`, not a negative or an infinity: a
        car moving away has no time to contact, and a finite stand-in keeps the term differentiable-
        free of NaN under `torch.compile`.
        """
        gap, closing = self.car_gap_closing(state)
        ttc = torch.where(closing > 1e-3, gap / closing.clamp_min(1e-3),
                          torch.full_like(gap, TTC_FAR))
        return ttc.min(1).values

    def overtake_hold(self, gap_now: torch.Tensor, gap_valid: torch.Tensor, boundary: torch.Tensor,
                      lead_steps: torch.Tensor, lead_paid: torch.Tensor):
        """(bonus this step (B,), new lead_steps, new lead_paid) for the sustained-lead term.

        `gap_now` is `signed_gaps`: + is that opponent ahead of me, so this car leads it by
        `-gap_now`. A lead counts while it is at least `overtake_hold_dist` metres and the opponent
        is still inside `overtake_range` -- a car half a lap away is not being led, it is being
        lapped or lapping, and the signed arc flips sign out there by construction.

        `lead_paid` carries two things at once, which is why it starts at **1** and not at 0:

        * 1 = this lead may not be paid. It is 1 at the start of every race and after every reset in
          it, so a learner that **spawned** ahead -- one race in three under `--spawn-order random`
          -- is not paid for a grid position, and a crashed opponent respawning behind the field
          does not hand the learner a bonus for the other car's mistake. It is also 1 immediately
          after a lead has been paid.
        * 0 = armed. It is cleared on the first step where the gap is valid and this car is **not**
          leading by the distance, which is the only evidence available that the lead, when it
          comes, was taken rather than given.

        The counter is consecutive steps of such a lead; the bonus is paid on the step the counter
        first reaches `overtake_hold_time`, once. Being re-passed disarms and re-arms, so passing
        again pays again -- which is the behaviour the term is for.

        `boundary` is `race_boundary(done)`: any car of this race resetting clears the counter and
        re-blocks the payment.
        """
        e = self.ecfg
        lead = (-gap_now >= e.overtake_hold_dist) & (gap_now.abs() < e.overtake_range) & gap_valid[:, None]
        steps = torch.where(lead, lead_steps + 1.0, torch.zeros_like(lead_steps))
        need = max(1.0, round(e.overtake_hold_time / self.sim.control_dt))
        fires = (steps >= need) & (lead_paid <= 0.0)
        # Armed only by a valid gap in which this car is not leading. Gated on `gap_valid` because
        # the step after a reset has no gap to read, and reading its absence as "not leading" would
        # arm the very spawn this flag exists to exclude.
        blocked = lead | ~gap_valid[:, None]
        paid = torch.where(blocked, torch.maximum(lead_paid, fires.to(lead_paid.dtype)),
                           torch.zeros_like(lead_paid))
        keep = (~boundary).to(steps.dtype)[:, None]
        return fires.to(steps.dtype).sum(1), steps * keep, paid * keep + (1.0 - keep)

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

    def opponent_visible(self, r: Optional[StepResult] = None, beams: int = 3, deg: float = 12.0) -> torch.Tensor:
        """(B,) bool: the nearest opponent (the one `privileged()` columns 8-11 describe) is seen by the newest
        scan -- at least `beams` returns off a car within `deg` of its bearing from the LiDAR.

        Privileged, train-time only. What it is for: an auxiliary target about the other car is only
        learnable where the scan holds it. The 90 deg behind the sensor are blind, and a car hidden
        behind another car or a wall is not in the returns; scoring those rows asks the network to
        guess (measured 2026-09-30: 27 % of rows in 3-car races, and they held the held-out R^2 of a
        supervised position read-out at -0.19-0.35 against 0.76-0.81 with the mask).
        """
        res = self.last_result if r is None else r
        if self.M <= 1:
            return torch.zeros(self.B, dtype=torch.bool, device=self.device)
        pv = self._priv(res)
        dx = pv[:, 8] * PRIV_OPP_DIST_SCALE - self.cfg.lidar.mount_x
        dy = pv[:, 9] * PRIV_OPP_DIST_SCALE - self.cfg.lidar.mount_y
        bearing = torch.atan2(dy, dx)
        fov = self.cfg.lidar.fov
        ang = torch.linspace(-0.5 * fov, 0.5 * fov, int(self.n_beams), device=self.device, dtype=pv.dtype)
        near = (ang[None] - bearing[:, None]).abs() < math.radians(deg)
        car = self.opponent_beam_mask(res) > 0.5
        return (car & near).sum(1) >= beams

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

    # ------------------------------------------------------------------ contract v2: critic and labels
    def _opp_rel(self, st: torch.Tensor):
        """Every seat of `sim.other_idx` relative to the ego, in the ego body frame (x forward, origin
        at the ego pose): dx, dy [m], dyaw [rad], dvx, dvy [m/s] (world velocities differenced, then
        rotated into the ego frame, so dx + dt dvx is to first order where the car will be), centre
        distance [m] and whether the seat's car is in the race at all (not parked). All (B, C).
        The G4 gate's `opp_block` arithmetic, which the critic results were measured on."""
        o = self.sim.other_idx
        me, ot = st[:, None, :], st[o]
        d = ot[..., :2] - me[..., :2]
        c, s = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
        dx, dy = d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c
        co, so = torch.cos(ot[..., 2]), torch.sin(ot[..., 2])
        vwx = (ot[..., 3] * co - ot[..., 4] * so) - (me[..., 3] * c - me[..., 4] * s)
        vwy = (ot[..., 3] * so + ot[..., 4] * co) - (me[..., 3] * s + me[..., 4] * c)
        dvx, dvy = vwx * c + vwy * s, -vwx * s + vwy * c
        dyaw = torch.remainder(ot[..., 2] - me[..., 2] + math.pi, 2 * math.pi) - math.pi
        present = (~self.sim.absent[o]) if self.sim.absent_any else torch.ones_like(dx, dtype=torch.bool)
        return dx, dy, dyaw, dvx, dvy, d.norm(dim=-1), present

    def privileged_v2(self, r: Optional[StepResult] = None) -> torch.Tensor:
        """(B, PRIV_V2_DIM) the v2 critic's privileged vector, in `PRIV_V2_KEYS` order.

        Critic-only and never an observation. v1's `privileged()` is untouched, so a v1 critic keeps
        its 21 (+ dial) columns. The scales are fixed, not learned or measured per run, so the same
        number means the same thing in every checkpoint: speeds over v_max (10), the side slip over
        2 m/s, yaw rate over the gyro scale (5 rad/s), lateral offset in metres, heading error over
        pi, the progress rate over v_max clamped to +-2 (a lap wrap is a one-step spike), the wall
        distance over 2 m clamped to [-1, 3], opponents as the G4 gate stored them, the geometry's
        distances and speeds over 5 and its times over the 10 s cap.
        """
        res = self.last_result if r is None else r
        if res is None:
            raise RuntimeError("privileged_v2 needs a stepped env: call reset() first")
        e, st = self.ecfg, res.state
        xy_c, yaw_c = self.sim.track.pose_at_s(res.s, self.sim.tid)
        herr = torch.remainder(st[:, 2] - yaw_c + math.pi, 2 * math.pi) - math.pi
        vm = float(e.v_max_policy)
        ego = torch.stack([st[:, 3] / vm, st[:, 4] / 2.0, st[:, 5] / float(e.imu_gyro_scale), res.lateral,
                           herr / math.pi, torch.remainder(res.s / self.sim.track.length[self.sim.tid], 1.0),
                           (res.progress / self.sim.control_dt / vm).clamp(-2.0, 2.0),
                           (res.wall_dist / 2.0).clamp(-1.0, 3.0)], 1)
        K, S = PRIV_V2_OPP_SLOTS, PRIV_OPP_DIST_SCALE
        opp = torch.zeros(self.B, K, len(PRIV_V2_OPP_KEYS), device=st.device, dtype=st.dtype)
        geo = torch.tensor([PRIV_V2_NO_OPP_DIST / S, 0.0, 1.0, 1.0], device=st.device,
                           dtype=st.dtype).expand(self.B, -1).clone()
        if self.M > 1:
            dx, dy, dyaw, dvx, dvy, dist, present = self._opp_rel(st)
            blk = torch.stack([present.to(st.dtype), dx / S, dy / S, dist / S, dvx / S, dvy / S, dyaw], 2)
            blk = blk * present[..., None].to(st.dtype)
            key = torch.where(present, dist, torch.full_like(dist, 1e6))
            order = key.argsort(1)                                   # nearest present seat first
            blk = blk.gather(1, order[..., None].expand(-1, -1, blk.shape[2]))
            n = min(K, blk.shape[1])
            opp[:, :n] = blk[:, :n]
            dist_c = torch.where(present, dist, torch.full_like(dist, PRIV_V2_NO_OPP_DIST))
            clos = torch.where(present, -(dx * dvx + dy * dvy) / dist_c.clamp_min(1e-3), torch.zeros_like(dist))
            ttc = torch.where(present & (clos > PRIV_V2_TTC_MIN_CLOSING),
                              (dist_c / clos.clamp_min(1e-3)).clamp(0.0, PRIV_V2_TTC_CAP),
                              torch.full_like(dist, PRIV_V2_TTC_CAP))
            j = key.argmin(1, keepdim=True)
            geo = torch.cat([dist_c.gather(1, j) / S, clos.gather(1, j) / S,
                             ttc.gather(1, j) / PRIV_V2_TTC_CAP,
                             ttc.min(1, keepdim=True).values / PRIV_V2_TTC_CAP], 1)
        out = torch.cat([ego, opp.reshape(self.B, -1), geo], 1)
        if self.ecfg.priv_props:
            out = torch.cat([out, self._prop_block(st)], 1)
        if self.ecfg.priv_time:
            n = float(self.ecfg.max_steps)
            lim = self.ep_limit.clamp(max=int(self.ecfg.max_steps))
            out = torch.cat([out, ((lim - self.ep_step).clamp(min=0).to(st.dtype) / n)[:, None]], 1)
        return out

    def _prop_block(self, st: torch.Tensor) -> torch.Tensor:
        """(B, PRIV_V2_PROP_DIM) the props the car is heading into (`EnvConfig.priv_props`): per slot present, dx/5,
        dy/5, dist/5 in the ego body frame, nearest first among live props with dx > -1 m within PRIV_V2_PROP_RANGE."""
        K, S = PRIV_V2_PROP_SLOTS, PRIV_OPP_DIST_SCALE
        blk = torch.zeros(self.B, K, len(PRIV_V2_PROP_KEYS), device=st.device, dtype=st.dtype)
        po = self.procedural
        if po is None or po.C == 0:
            return blk.reshape(self.B, -1)
        d = po.p_poses[:, :, :2].to(st.dtype) - st[:, None, :2]
        c, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
        dx, dy = d[..., 0] * c + d[..., 1] * s_, -d[..., 0] * s_ + d[..., 1] * c
        dist = d.norm(dim=-1)
        live = (po.p_sid >= 0) & (dx > -1.0) & (dist < PRIV_V2_PROP_RANGE)
        key = torch.where(live, dist, torch.full_like(dist, 1e6))
        n = min(K, key.shape[1])
        j = key.topk(n, dim=1, largest=False).indices
        g = lambda t: t.gather(1, j)
        pres = g(live).to(st.dtype)
        blk[:, :n] = torch.stack([pres, g(dx) / S, g(dy) / S, g(dist) / S], 2) * pres[..., None]
        return blk.reshape(self.B, -1)

    def _lidar_pose(self):
        """(mount x, mount y, beam angles (B, N_sensor)) of each car's LiDAR in its body frame: the
        per-env truth the scan was cast with (`Lidar._rays` reads the same `P` entries)."""
        P, cl = self.sim.P, self.cfg.lidar
        mx = P["mount_x"] if "mount_x" in P else torch.full((self.B,), float(cl.mount_x), device=self.device)
        my = P["mount_y"] if "mount_y" in P else torch.full((self.B,), float(cl.mount_y), device=self.device)
        ang = self.sim.lidar.angles[None, :]
        if "mount_yaw" in P:
            ang = ang + P["mount_yaw"][:, None]
        return mx, my, ang

    def _beam_valid(self, res: StepResult) -> torch.Tensor:
        """(B, N_sensor) bool: a return the seat attribution may count (`BEAM_VALID_TOL_M`)."""
        rng, tru = res.scan, res.scan_true
        return (torch.isfinite(rng) & (rng < self.range_max - 1e-3)
                & ((rng - tru).abs() < BEAM_VALID_TOL_M))

    def _attribute_car_beams(self, res: StepResult) -> torch.Tensor:
        """(B, C) long: valid car returns of the newest scan, attributed seat by seat.

        The LiDAR marks a beam HIT_CAR without saying which car (`lidar.Lidar.scan`), so each car
        beam goes to the seat whose body + rear detection box -- the rectangle the scan draws for it
        -- is nearest the beam's TRUE hit point, if within `OPP_ATTRIB_TOL_M`. A beam stopped by one
        car is not counted for the car behind it: occlusion comes out of the scan itself.
        """
        from .lidar import HIT_CAR
        C = self.M - 1
        if C <= 0:
            return torch.zeros(self.B, 0, dtype=torch.long, device=self.device)
        car = (res.scan_type == HIT_CAR) & self._beam_valid(res)
        mx, my, ang = self._lidar_pose()
        rt = torch.where(torch.isfinite(res.scan_true), res.scan_true, torch.zeros_like(res.scan_true))
        px = mx[:, None] + rt * torch.cos(ang)                       # (B, N) hit points, ego frame
        py = my[:, None] + rt * torch.sin(ang)
        dx, dy, dyaw, _dvx, _dvy, _dist, present = self._opp_rel(res.state)
        c, s = torch.cos(dyaw)[..., None], torch.sin(dyaw)[..., None]
        ex, ey = px[:, None, :] - dx[..., None], py[:, None, :] - dy[..., None]
        qx, qy = ex * c + ey * s, -ex * s + ey * c                   # (B, C, N) in each seat's frame
        o = self.sim.other_idx
        L, W = self.sim.car_dims[o][..., 0], self.sim.car_dims[o][..., 1]
        rd, rw = self.sim.car_rear[o][..., 0], self.sim.car_rear[o][..., 1]
        x_lo, x_hi = (-(0.5 * L + rd))[..., None], (0.5 * L)[..., None]
        hw = (0.5 * torch.maximum(W, rw))[..., None]
        out_x = (x_lo - qx).clamp_min(0.0) + (qx - x_hi).clamp_min(0.0)
        out_y = (qy.abs() - hw).clamp_min(0.0)
        dbox = torch.sqrt(out_x * out_x + out_y * out_y)
        dbox = torch.where(present[..., None], dbox, torch.full_like(dbox, float("inf")))
        dmin, j = dbox.min(1)                                        # (B, N)
        hit = car & (dmin < OPP_ATTRIB_TOL_M)
        counts = torch.zeros(self.B, C, dtype=torch.long, device=self.device)
        return counts.scatter_add_(1, j, hit.long())

    def _update_opp_visibility(self, res: StepResult, reset_rows: Optional[torch.Tensor]) -> None:
        """Advance every seat's `age` (steps since it was last seen) by one observation.

        A seat's track ends when either car of the pair starts a new episode: the ego's own reset
        clears every track it held, and a seat whose car respawned is a different car in a different
        place (the perception gate's rule: identity broken at respawn and episode start)."""
        if self._opp_age is None or self.M <= 1:
            return
        age = (self._opp_age + 1).clamp_max(OPP_NEVER_SEEN)
        if reset_rows is not None:
            o = self.sim.other_idx
            gone = reset_rows[:, None] | reset_rows[o]
            age = torch.where(gone, torch.full_like(age, OPP_NEVER_SEEN), age)
        beams = self._attribute_car_beams(res)
        mx, my, _ang = self._lidar_pose()
        dx, dy, _dyaw, _dvx, _dvy, _dist, present = self._opp_rel(res.state)
        dist_l = torch.hypot(dx - mx[:, None], dy - my[:, None])
        seen = present & (beams >= OPP_SEEN_BEAMS) & (dist_l < OPP_LABEL_RANGE_M)
        self._opp_age = torch.where(seen, torch.zeros_like(age), age)
        self._opp_beams = beams

    def opponent_tracks_v2(self, r: Optional[StepResult] = None) -> Dict[str, torch.Tensor]:
        """Per seat, the labels of the v2 opponent-slot auxiliary (B8), for THIS observation.

        Train-time only, privileged. Keys, all (B, C) unless noted, C = race_size - 1:
          present  the seat's car is in the race (not parked)
          cand     present and within OPP_LABEL_RANGE_M of the LiDAR: a car the head should report
          lab      (B, C, 3) [dx / 5, dy / 2, v_lat / 2], LiDAR frame (x forward from the LiDAR,
                   y left), v_lat the relative lateral velocity in the ego frame -- the align and
                   perception gates' targets
          dist     distance from the LiDAR [m]
          beams    valid car returns of the newest scan on this seat (`_attribute_car_beams`)
          age      steps since the seat was last seen (>= OPP_SEEN_BEAMS beams within range);
                   0 = seen now, OPP_NEVER_SEEN = not since its track began
        Which candidates are KNOWABLE inside a training window -- seen within OPP_MEMORY_STEPS and
        that sighting inside the chunk or its burn-in -- is the trainer's to decide, because only it
        knows the window: `learn.aux_v2.b8_know_mask(cand, age, steps_into_window)`.
        Needs `EnvConfig.opp_visibility`: the age is a running count, not a function of one step.
        """
        if self._opp_age is None:
            raise RuntimeError("opponent_tracks_v2 needs EnvConfig.opp_visibility: the age since a "
                               "seat was last seen is counted every step, not recomputed on demand")
        res = self.last_result if r is None else r
        C = max(0, self.M - 1)
        dev, dt_ = self.device, res.state.dtype
        if C == 0:
            z = torch.zeros(self.B, 0, device=dev)
            return {"present": z.bool(), "cand": z.bool(), "lab": torch.zeros(self.B, 0, 3, device=dev),
                    "dist": z, "beams": z.long(), "age": z.long()}
        mx, my, _ang = self._lidar_pose()
        dx, dy, _dyaw, _dvx, dvy, _dist, present = self._opp_rel(res.state)
        lx, ly = dx - mx[:, None], dy - my[:, None]
        dist_l = torch.hypot(lx, ly)
        pf = present.to(dt_)
        lab = torch.stack([lx / 5.0, ly / 2.0, dvy / 2.0], 2) * pf[..., None]
        return {"present": present, "cand": present & (dist_l < OPP_LABEL_RANGE_M), "lab": lab,
                "dist": torch.where(present, dist_l, torch.full_like(dist_l, float("inf"))),
                "beams": self._opp_beams.clone(), "age": self._opp_age.clone()}

    def nearest_visible_car_v2(self, r: Optional[StepResult] = None) -> torch.Tensor:
        """(B, 5) the nearest car the newest scan SEES -- >= OPP_SEEN_BEAMS of its returns, within
        OPP_LABEL_RANGE_M of the LiDAR: [exists, dx / 5, dy / 2, closing / 5, v_lat / 2], LiDAR
        frame, closing = -(unit line of sight . relative velocity). Zeros where none is seen.

        The target of the perception gate's auxiliary B (the align gate found the true distance to
        the nearest visible car alone gives contact-risk AUC 0.902). Train-time only; needs
        `EnvConfig.opp_visibility` for this step's attributed beams."""
        if self._opp_beams is None:
            raise RuntimeError("nearest_visible_car_v2 needs EnvConfig.opp_visibility")
        res = self.last_result if r is None else r
        out = torch.zeros(self.B, 5, device=self.device, dtype=res.state.dtype)
        if self.M <= 1:
            return out
        mx, my, _ang = self._lidar_pose()
        dx, dy, _dyaw, dvx, dvy, _dist, present = self._opp_rel(res.state)
        lx, ly = dx - mx[:, None], dy - my[:, None]
        dist_l = torch.hypot(lx, ly)
        seen = present & (self._opp_beams >= OPP_SEEN_BEAMS) & (dist_l < OPP_LABEL_RANGE_M)
        key = torch.where(seen, dist_l, torch.full_like(dist_l, float("inf")))
        j = key.argmin(1, keepdim=True)
        ex = seen.any(1).to(out.dtype)
        g = lambda t: t.gather(1, j)[:, 0]
        dl = g(dist_l).clamp_min(1e-6)
        closing = -(g(lx) * g(dvx) + g(ly) * g(dvy)) / dl
        out[:, 0] = ex
        out[:, 1] = ex * g(lx) / 5.0
        out[:, 2] = ex * g(ly) / 2.0
        out[:, 3] = ex * closing / 5.0
        out[:, 4] = ex * g(dvy) / 2.0
        return out

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
        # Checked on the host copy when there is one. A device tensor is left unchecked: reading it
        # back is a sync per call, and the one caller that passes one (the interactive teacher,
        # every step) builds it from `arange * dt`.
        if not (torch.is_tensor(times) and times.device.type != "cpu") and float(t.min()) < 0:
            raise ValueError(f"car_future horizons must be >= 0, got {t.tolist()}")
        c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        vw = torch.stack([st[:, 3] * c - st[:, 4] * sn, st[:, 3] * sn + st[:, 4] * c], 1)
        out = st[:, None, :2] + vw[:, None, :] * t[None, :, None]          # constant velocity
        if mode == "constv" or self.M == 1:
            return out
        cv = out
        # A car an external baseline drives (`external_driven`) has a tracker row too, solving the
        # caller's placeholder action -- a phantom plan nothing executes. Its honest prediction is the
        # constant-velocity floor, which is what every other model falls back to for a car it cannot read.
        ext = self.external_driven if bool(self.external_drivers) else None
        keep = (lambda res: res if ext is None else torch.where(ext[:, None, None], cv, res))
        pred = self._tracker_future(st, t)
        if mode == "pred":
            return keep(out if pred is None else pred)
        if mode == "hybrid":
            seam = self._tracker_seam()
            if pred is None or seam is None or self.teacher is None:
                return keep(out if pred is None else pred)
            # Both models evaluated at the requested times AND at the seam, so the tail can be
            # attached to the tracker's own endpoint rather than to the walk's -- otherwise the
            # prediction would jump by the walk's error at the join, which is the error the tracker
            # was brought in to avoid.
            tt = torch.cat([t, t.new_full((1,), seam)])
            pa, wa = self._tracker_future(st, tt), self._raceline_future(st, tt)
            late = (t > seam)[None, :, None]
            return keep(torch.where(late, pa[:, -1:] + (wa[:, :-1] - wa[:, -1:]), pa[:, :-1]))
        if self.teacher is not None and bool(self.teacher_driven.any()):
            walk = self._raceline_future(st, t)
            out = torch.where(self.teacher_driven[:, None, None], walk, out)
        if pred is not None:
            out = torch.where((~self.teacher_driven)[:, None, None], pred, out)
        return keep(out)

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

    def _pred_grid_start(self):
        """`delay - control_dt`, the time of `last_pred[:, 0]` relative to now.

        With per-car delays this stays on the device: it used to be `float(tracker_delay.mean())`,
        a host sync on every call, and the teacher calls it twice a step. The double subtraction
        and the single cast to float32 are what the Python-float version did implicitly, so the
        grid is bit-identical.
        """
        if self.tracker_delay is None:
            return float(self.tracker.spec.delay) - self.sim.control_dt
        return (self.tracker_delay.mean().double() - self.sim.control_dt).float()

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
        grid = self._pred_grid_start() + torch.arange(z.shape[1], device=self.device, dtype=t.dtype) * self.tracker.spec.dt
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
                        state: Optional[torch.Tensor] = None, *, return_yaw: bool = False):
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
        # tracker.reset clears controls but last_pred can still belong to the
        # previous episode (especially with opponent tokens disabled). Until a
        # new control step has produced a prediction, use the measured velocity.
        steps = getattr(self, "ep_step", None)
        fresh = torch.zeros(st.shape[0], dtype=torch.bool, device=st.device) if steps is None else steps == 0
        t = torch.as_tensor(times, dtype=st.dtype, device=self.device).reshape(-1)
        c, sn = st[:, 2].cos(), st[:, 2].sin()
        velocity = torch.stack((st[:, 3] * c - st[:, 4] * sn, st[:, 3] * sn + st[:, 4] * c), -1)
        fallback = st[:, None, :2] + velocity[:, None] * t[None, :, None]
        fut = torch.where(fresh[:, None, None], fallback, fut)
        d = (st[o][:, :, :2] - st[:, None, :2]).norm(dim=2)                 # (B, C)
        present = d < self.ecfg.overtake_range
        if not return_yaw:
            return fut[o], present
        # Body orientation, not atan2(0, 0) for a stopped car. The raceline walk has
        # position only: estimate its moving tangent and retain the measured slip
        # angle while rejoining. Tracker branches use their actual yaw samples on
        # the same latency clock as _tracker_future.
        t = torch.as_tensor(times, dtype=st.dtype, device=self.device).reshape(-1)
        mode = self.ecfg.opp_future_model if model is None else model
        yaw = st[:, 2:3].expand(-1, t.numel()).clone()
        if mode != "constv" and t.numel() > 1:
            step = fut[:, 1:] - fut[:, :-1]
            step = torch.cat((step[:, :1], step), 1)
            slip = torch.atan2(st[:, 4], st[:, 3].clamp_min(1e-6))
            tangent = torch.atan2(step[..., 1], step[..., 0]) - slip[:, None]
            moving = (step.norm(dim=-1) > 1e-5) & (t > 0)[None]
            # A car that stops later retains its last predicted body heading,
            # rather than snapping back to its heading at the query origin.
            samples = torch.cat((st[:, 2:3], tangent), 1)
            indices = torch.arange(1, t.numel() + 1, device=self.device)[None].expand_as(yaw)
            last_moving = torch.where(moving, indices, torch.zeros_like(indices)).cummax(1).values
            yaw = samples.gather(1, last_moving)
        if mode != "constv" and self.tracker is not None and self.tracker.last_pred is not None and self._plan_pose is not None:
            z = self.tracker.last_pred
            # Interpolate sin/cos to avoid wrapping across +/- pi.
            angle = self._plan_pose[:, 2:3] + z[:, :, 2]
            grid = self._pred_grid_start() + torch.arange(z.shape[1], device=self.device, dtype=t.dtype) * self.tracker.spec.dt
            unit = _interp_path(torch.stack((angle.cos(), angle.sin()), -1), grid, t)
            predicted = torch.atan2(unit[..., 1], unit[..., 0])
            use = (~self.teacher_driven)[:, None].expand_as(yaw)
            if mode == "pred":
                use = torch.ones_like(use)
            elif mode == "hybrid":
                use = (torch.ones_like(use) if self.teacher is None else
                       (t <= grid[-1])[None].expand_as(yaw))
            yaw = torch.where(use, predicted, yaw)
        yaw = torch.where((t == 0)[None] | fresh[:, None], st[:, 2:3], yaw)
        return fut[o], present, yaw[o]

    # ------------------------------------------------------------------ privileged opponent tokens
    def _capture_plan(self, pose_before: torch.Tensor) -> None:
        """Keep this step's plan and the pose it is anchored at, for `opp_tokens`.

        Called from `step()` immediately after the tracker has run, with the state the tracker was
        called on. `tracker.last_ref` is (B, N+1, 4) x/y/heading/speed in that pose's body frame on
        the grid 0, dt, ... -- one row per car, opponents included, because every car of a race is
        driven through the same tracker. For a teacher-driven opponent that row is its raceline
        target with its scheduled event (brake / stop / shift / defend / yield / line) and its
        follow cap already applied, since all of those act on the action before the tracker sees it;
        for a pool opponent it is that checkpoint's own plan. This is what makes "the opponent's own
        tracked plan" one object rather than one per kind of opponent.
        """
        if self.opp_token == "off":
            return
        main, alt = ((self.tracker.last_pred, self.tracker.last_ref)
                     if OPP_TOKEN_PLAN_SOURCE == "pred" else
                     (self.tracker.last_ref, self.tracker.last_pred))
        self._plan_traj.copy_(main)
        self._plan_pose.copy_(pose_before[:, :3])
        self._plan_real.fill_(True)
        if self.opp_token_keep_alt:
            self._plan_alt.copy_(alt)

    def _plan_fallback(self, ids: torch.Tensor, state: torch.Tensor) -> None:
        """Give `ids` a constant-velocity stand-in plan anchored at their current pose.

        Two rows need it: every row at `reset()`, where no action has been taken yet, and a row that
        respawned inside `step()`, whose stored plan belongs to the episode that just ended and to a
        pose on the other side of the track. Predicting "it keeps going the way it is going" is the
        only thing available without a plan, and it is marked so the validation can report how often
        the label is one (a respawn is ~0.2 % of steps).
        """
        if self.opp_token == "off" or ids.numel() == 0:
            return
        n = self.tracker.spec.N
        dt = self.tracker.spec.dt
        self._plan_traj[ids] = straight_traj(state[ids], n, dt, self._plan_clock_shift("main", ids))
        self._plan_pose[ids] = state[ids, :3]
        self._plan_real[ids] = False
        if self.opp_token_keep_alt:
            self._plan_alt[ids] = straight_traj(state[ids], n, dt, self._plan_clock_shift("alt", ids))

    def _plan_clock_shift(self, which: str, ids: Optional[torch.Tensor] = None):
        """How far ahead of the anchor pose a stored trajectory's t = 0 sits, per row.

        `ref` starts at the car (0). `pred` starts at the pose the tracker predicts over its own
        command latency, so its clock is `tracker_delay` ahead and the sampler subtracts that.
        """
        src = OPP_TOKEN_PLAN_SOURCE if which == "main" else ("ref" if OPP_TOKEN_PLAN_SOURCE == "pred" else "pred")
        if src != "pred":
            return 0.0
        d = self.tracker_delay
        return d if ids is None else d[ids]

    def _opp_token_times(self) -> torch.Tensor:
        """(B, H) the times on each row's plan clock that the token's horizons land on.

        The plan's t = 0 is the instant the tracker was called, which is one control step before the
        state the observation describes (the action is issued, then the simulator advances by
        `control_dt`, then the observation is built). So the horizon h after the observation is
        h + control_dt on the plan's own clock. Getting this wrong is a systematic along-track bias
        of one step -- about 0.1 m at 4 m/s -- in the same direction at every horizon, which is
        exactly the kind of error a validation R^2 hides and a residual plot does not.
        """
        return (self._token_horizons + self.sim.control_dt)[None].expand(self.B, -1)

    def opp_token_slots(self, st: torch.Tensor):
        """(rows (B, n), present (B, n)) -- which car fills each token slot, and whether it counts.

        `rows` are global env rows, ordered by body-centre distance, nearest first; `present` is 1
        only inside `overtake_range`. One definition, read by `opp_tokens`, by the future columns
        and by the label validation, so all three are talking about the same car.
        """
        o = self.sim.other_idx                                   # (B, C)
        d = st[o][:, :, :2] - st[:, None, :2]
        dist = d.norm(dim=2)
        n = min(OPP_TOKEN_CARS, o.shape[1])
        order = dist.argsort(dim=1)[:, :n]
        rows = torch.gather(o, 1, order)
        present = (torch.gather(dist, 1, order) < self.ecfg.overtake_range).to(st.dtype)
        return rows, present

    def opp_future_columns(self, st: torch.Tensor, rows: torch.Tensor, source: Optional[str] = None):
        """(B, n, H, 2) where each slot's car intends to be, in the ego's current body frame.

        `source` picks which of the tracker's two trajectories is read. `ref` is the reference the
        plan asks for -- the raceline target with the opponent's scheduled event applied, or a pool
        policy's own emitted plan. `pred` is the iLQR's forward rollout of its own kinematic model
        under the commands it chose. `OPP_TOKEN_PLAN_SOURCE` names the one the token uses (`pred`,
        on the measurement in `f1sim/opp_token.py`); the other is available only when
        `opp_token_keep_alt` is set, which is what
        `work/oracle-planner/work/label_validation.py` does to score both against the realised
        future.

        `pred` starts at the delay-compensated pose rather than at the car, because that is where
        the tracker's model starts its rollout, so its clock is shifted by that car's calibrated
        command latency and the shift is undone here.
        """
        t = self._opp_token_times()                              # (B, H)
        source = OPP_TOKEN_PLAN_SOURCE if source is None else source
        if source not in ("ref", "pred"):
            raise ValueError(f"plan source must be 'ref' or 'pred', got {source!r}")
        if source == OPP_TOKEN_PLAN_SOURCE:
            traj, which = self._plan_traj, "main"
        else:
            if not self.opp_token_keep_alt:
                raise RuntimeError(f"opp_future_columns(source={source!r}) is not this build's "
                                   f"label ({OPP_TOKEN_PLAN_SOURCE}); set env.opp_token_keep_alt")
            traj, which = self._plan_alt, "alt"
        shift = self._plan_clock_shift(which)
        shift = None if not torch.is_tensor(shift) else shift
        outs = []
        for slot in range(rows.shape[1]):
            kj = rows[:, slot]
            ts = t if shift is None else (t - shift[kj][:, None]).clamp_min(0.0)
            body = sample_body_traj(traj[kj], self.tracker.spec.dt, ts)
            outs.append(to_ego(to_world(body[:, :, :2], self._plan_pose[kj]), st[:, :3]))
        return torch.stack(outs, 1)

    def opp_tokens(self, r: Optional[StepResult] = None) -> torch.Tensor:
        """(B, opp_token_dim) the privileged opponent block, in `opp_token.opp_token_keys` order.

        **An oracle.** The columns are the simulator's ground truth about the other cars, in the
        ego's current body frame, normalised by `OPP_TOKEN_SCALE` -- the same scale `future_labels`
        and the critic's privileged opponent columns use. `f1sim/opp_token.py` documents the layout;
        the short version is `OPP_TOKEN_CARS` slots ordered by body distance, each

            dx, dy [, dvx, dvy [, (fx, fy) at each of OPP_TOKEN_HORIZONS_S ]], present

        with `present` = 1 only where that slot holds a car inside `overtake_range`, and every other
        column of the slot multiplied by it, so an absent, out-of-range or not-yet-existing opponent
        contributes exactly zeros rather than a stale position.

        `dvx, dvy` is the opponent's velocity relative to the ego, taken to the world and rotated
        into the ego frame -- the same convention `future_labels` uses and deliberately NOT
        `privileged()`'s `other.vx - ego.vx`, which is a difference of two body-frame speeds in two
        different frames.

        `fx, fy` is where the opponent's own controller intends to be at that horizon: its plan
        tracker's reference trajectory, sampled at the horizon plus one control step (see
        `_opp_token_times`), mapped to the world through the pose the plan was anchored at, and then
        into the ego's *current* frame. The ego's own future is its own choice and is not subtracted:
        the column answers "where will that car be", not "where will it be relative to where I will
        be". The label is validated against the realised future per horizon in
        `work/oracle-planner/work/label_validation.py`.
        """
        if self.opp_token == "off":
            raise RuntimeError("opp_tokens() on an env built with opp_token 'off'")
        res = self.last_result if r is None else r
        if res is None:
            raise RuntimeError("opp_tokens needs a stepped env: call reset() first")
        st = res.state
        e = self.ecfg
        per = opp_token_car_dim(self.opp_token)
        out = torch.zeros(self.B, self.opp_token_dim, device=self.device, dtype=st.dtype)
        if self.opp_token_ablate:
            # The width the actor's first layer expects, carrying nothing. Zeros and not noise: the
            # columns were zero at initialisation, so zero is the one value this policy has
            # certainly seen and the one that cannot be mistaken for an opponent somewhere.
            return out
        rows, present_all = self.opp_token_slots(st)
        n = rows.shape[1]
        fut = self.opp_future_columns(st, rows) if self.opp_token == "future" else None
        c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        for slot in range(n):
            kj = rows[:, slot]                                   # (B,) global row of that opponent
            dn = st[kj, :2] - st[:, :2]
            present = present_all[:, slot]
            col = slot * per
            out[:, col + 0] = present * (dn[:, 0] * c + dn[:, 1] * sn) / OPP_TOKEN_SCALE
            out[:, col + 1] = present * (-dn[:, 0] * sn + dn[:, 1] * c) / OPP_TOKEN_SCALE
            k = col + 2
            if self.opp_token in ("posvel", "future"):
                co, so = torch.cos(st[kj, 2]), torch.sin(st[kj, 2])
                vwx = (st[kj, 3] * co - st[kj, 4] * so) - (st[:, 3] * c - st[:, 4] * sn)
                vwy = (st[kj, 3] * so + st[kj, 4] * co) - (st[:, 3] * sn + st[:, 4] * c)
                out[:, k + 0] = present * (vwx * c + vwy * sn) / OPP_TOKEN_SCALE
                out[:, k + 1] = present * (-vwx * sn + vwy * c) / OPP_TOKEN_SCALE
                k += 2
            if fut is not None:
                rel = fut[:, slot] * present[:, None, None] / OPP_TOKEN_SCALE
                out[:, k:k + 2 * rel.shape[1]] = rel.reshape(self.B, -1)
                k += 2 * rel.shape[1]
            out[:, col + per - 1] = present
        return out

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
        # Two scalar fills, not `torch.tensor([...], device=...)`: that is a pageable host copy,
        # which waits for everything already queued on the GPU. This runs every control step in
        # the console's /drive mode, so it was a full CPU-GPU sync per step and the two stopped
        # overlapping -- the step became their sum instead of their maximum.
        self.ext_cmd[idx, 0].fill_(steer)
        self.ext_cmd[idx, 1].fill_(float(speed))
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

    def _v_lo(self) -> torch.Tensor:
        """(B,) the speed a = -1 commands: `v_min_policy` for learners, 0 for everyone else."""
        v_min = float(self.ecfg.v_min_policy)
        if v_min == 0.0:
            return torch.zeros(self.B, device=self.device)
        return torch.where(self.learner, torch.full((self.B,), v_min, device=self.device), torch.zeros(self.B, device=self.device))

    def teacher_action_to_normalized(self, cmd: torch.Tensor) -> torch.Tensor:
        """(steer [rad], speed [m/s]) -> policy action in [-1, 1] (direct action space)."""
        v_lo = self._v_lo()[: cmd.shape[0]] if cmd.shape[0] == self.B else torch.zeros(cmd.shape[0], device=cmd.device)
        return torch.stack([cmd[:, 0] / self.s_max, (cmd[:, 1] - v_lo) / (self.ecfg.v_max_policy - v_lo) * 2 - 1], 1).clamp(-1, 1)

    @property
    def plan_layout(self):
        """Which columns of this env's plans are path and which speed: `mpc.plan_layout` of the spec
        its tracker decodes with. Read by every place that scales or sets a plan's speed -- "linear"
        and "wp4" are both 8 wide, so a positional slice would scale the wrong numbers without any
        shape complaining. Plan mode only. The learner's plans; the opponents' are `opp_plan_spec`'s
        (`EnvConfig.opponent_speed_mode`)."""
        return _plan_layout(self.tracker.spec)

    def _opp_format_rows(self) -> torch.Tensor:
        """(B,) the cars that plan in `opp_plan_spec`'s format: driven by a teacher or by a league
        checkpoint (unless that entry plans in a format of its own, `_row_format_codes`). A `self` pool
        entry is the learner's own weights, so its car plans like the learner."""
        return self.teacher_driven | self.pool_driven

    def _row_format_codes(self) -> torch.Tensor:
        """(B,) uint8 code into `mpc.ROW_FORMATS` of the format each car plans in: the learner's, the
        opponents' for a teacher or league car -- and a league car's own entry's format once the
        entries differ (`_set_entry_formats`)."""
        ego, opp = self._row_codes
        codes = torch.where(self._opp_format_rows(), opp, ego)
        if self._pool_codes is not None:
            codes = torch.where(self.pool_driven, self._pool_codes[self.opp_driver], codes)
        return codes

    def _sync_row_formats(self) -> None:
        """Point each row of the tracker at its car's plan format (`EnvConfig.opponent_speed_mode`, and
        each league entry's own).

        From the driver masks at every step rather than kept in step with them: "mixed", "pool" and
        the league slots redraw those masks at race resets, and the rollout teacher's shadow env has
        them copied in wholesale. In place, so a captured solver graph follows it. Nothing at all
        when every car plans in one format."""
        if self._row_codes is None:
            return
        self.tracker.row_format.copy_(self._row_format_codes())

    def _tracker_plan_speed(self) -> torch.Tensor:
        """The incoming SI odometry sample shared by teacher geometry and the plan decoder."""
        return self.last_result.odom[:, 3] if self.last_result is not None else self.sim.state[:, 3]

    def _scale_plan_speeds(self, a: torch.Tensor) -> torch.Tensor:
        """The plans the follower tracks, the learner's speeds x `EnvConfig.plan_speed_scale` (the
        venue's speed knob); the traffic's rows and every path dimension are left as they are."""
        from .mpc import scale_plan_speeds
        scaled = scale_plan_speeds(a, self.ecfg.speed_mode, self.ecfg.plan_speed_scale)
        return torch.where(self.learner[:, None], scaled, a)

    def set_grip_budget(self, mu: Optional[torch.Tensor]) -> None:
        """(B,) friction each car is allowed to use from now on, or None for no budget. The trainer
        owns it (it is the policy's dial, redrawn per episode); the env only charges for exceeding it."""
        self.grip_budget = None if mu is None else mu.to(self.device).reshape(-1)

    def decision_mask(self) -> torch.Tensor:
        """(B,) bool: the cars whose NEXT step is a decision (`EnvConfig.decision_every`) -- every car at 1. A
        learner row that is not deciding drives its held plan whatever action `step` is given for it."""
        K = int(self.ecfg.decision_every)
        if K <= 1:
            return torch.ones(self.B, dtype=torch.bool, device=self.device)
        return (self._dec_cnt % K) == 0

    def teacher_label(self, teacher, offset: Optional[torch.Tensor] = None,
                      mu: Optional[torch.Tensor] = None, spec=None,
                      rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The teacher's action in this env's action space: (steer, speed) or its local plan.

        mu: (B,) friction the teacher should *drive for* instead of the env's own -- the label of a
        dial-conditioned student, whose input says how much grip to use rather than how much there
        is. Everything else the teacher is privileged to (latency, calibration) stays the truth.

        spec: the plan format to write it in -- the tracker's own (the learner's) by default; the
        opponents' commands pass `opp_plan_spec`.

        rows: the env rows to label, (R,) long -- the result is then (R, ACT_DIM), and the teacher's
        `last_label_valid` is (R,) too. The guard over the raceline teacher and the interactive
        teacher plan those rows only (`InteractiveTeacher.plan_rows`; the opponents are still read
        from the whole batch, `row_planning`), each row getting the command the whole-batch call
        gives it; any other teacher is asked for the whole batch and cut. Every argument stays the
        whole batch's. Why: in a race of four the guard labelled every car while DAgger stores the
        learner's quarter -- 0.118 s of every 0.56 s collection step at 512 cars (v2smoke_b, 21 % of
        collection), for rows nothing reads."""
        P = self.sim.P if mu is None else {**self.sim.P, "mu": mu.to(self.sim.P["mu"].dtype)}
        if self.act_dim == 2:
            kwargs = {} if offset is None else {"offset": offset}
            a = self.teacher_action_to_normalized(teacher(self.sim.state, P, self.sim.tid, **kwargs))
            return a if rows is None else a[rows]
        spec = self.tracker.spec if spec is None else spec
        v_max, v_meas = self.ecfg.v_max_policy, self._tracker_plan_speed()
        from .teacher import RacelineTeacher
        if isinstance(teacher, RacelineTeacher):
            # A global raceline alone does not certify what this tracker will
            # execute. Keep its line, but check achievable pace/braking choices
            # through the same preview and geometry as the interactive teacher.
            # Walls on the plan's own path only (`teacher_walls_within_plan`).
            from .interactive_teacher import InteractiveTeacher
            guards = getattr(self, '_raceline_teacher_guards', None)
            if guards is None:
                guards = self._raceline_teacher_guards = {}
            guard = guards.get(teacher)
            if guard is None:
                guard = guards[teacher] = InteractiveTeacher(teacher, self, offsets=(0.0,), cand_iters=6, lane_clamp=True,
                                                             walls_within_plan=self.ecfg.teacher_walls_within_plan)
            if rows is None:
                action = guard.plan_action(self.sim.state, P, self.sim.tid, v_max, spec, offset=offset,
                                           plan_speed=v_meas)
            else:
                action = guard.plan_rows(rows, self.sim.state, P, self.sim.tid, v_max, spec, offset=offset,
                                         plan_speed=v_meas)
            teacher.last_label_valid = guard.last_label_valid
            return action
        from .interactive_teacher import InteractiveTeacher
        if rows is not None and isinstance(teacher, InteractiveTeacher):
            # its subset call scores and validates exactly those rows (`last_label_valid` (R,)); the
            # layout teacher's `plan_rows` keeps a per-env validity it does not refresh, so it is not
            # asked for a subset here
            return teacher.plan_rows(rows, self.sim.state, P, self.sim.tid, v_max, spec, offset=offset,
                                     plan_speed=v_meas)
        a = teacher.plan_action(self.sim.state, P, self.sim.tid, v_max, spec, offset=offset, plan_speed=v_meas)
        if rows is None:
            return a
        valid = getattr(teacher, "last_label_valid", None)
        if torch.is_tensor(valid) and valid.shape[:1] == a.shape[:1]:
            teacher.last_label_valid = valid[rows]
        return a[rows]

    def action_driven(self) -> torch.Tensor:
        """(B,) bool: the cars whose command is `step`'s `action` argument -- every row
        `_opponent_actions` does not overwrite with a pool checkpoint's or a teacher's command. The
        learners (slot 0) always; in self-play, mixed or pool races the opponent seats the policy
        drives as well. A label written for any other row is thrown away by `step`, so a trainer
        that labels only these rows (`learn.dagger_v2.label_rows`) drives exactly as before."""
        if self.M == 1 or self.ecfg.opponent not in ("teacher", "mixed", "pool", "slots"):
            return torch.ones(self.B, dtype=torch.bool, device=self.device)
        mine = torch.ones(self.B, dtype=torch.bool, device=self.device)
        if self.pool is not None:
            mine = mine & ~self.pool_driven
        if self.teacher_any:
            mine = mine & ~self.teacher_driven
        return mine

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
            dt = 2.0 * ds / np.maximum(rl.v + np.roll(rl.v, -1), 1e-9)
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
        # The time to beat is the POLICY's own recent pace (the design above): only the learner's cars update it. In a race
        # the other cars -- scripted planners at 0.7-1.05 of the line, league seats, parked seats -- used to be averaged in
        # too, so the bar a learner had to beat moved with the opponents' mix (2026-10-27).
        w = (scored & self.learner).to(t_sec.dtype)
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
