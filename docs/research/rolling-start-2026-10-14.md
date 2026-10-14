# Rolling start for abreast opponents: the fix, and whether it works (2026-10-14)

Branch `fix/rolling-start` (from `main` 4db62bc). Background: commit 4db62bc added a rolling start for
teacher-driven opponents placed abreast of the learner (`EnvConfig.spawn_alongside_hold` /
`spawn_alongside_ramp`) and measured that it did **not** reduce the abreast opponent's merge contacts
(every seat abreast, 80 six-second episodes per arm: 23 / 19 raw / charged with it, 20 / 18 without,
first contact at 0.45-0.55 s either way). Two causes were diagnosed on 228 abreast opponents: the held
raceline offset was computed at the pure-pursuit lookahead but cut by the lane clamp at the car's own
raceline point (the 83 cut by more than 5 cm moved 0.21-0.26 m toward the learner within 1 s, median;
p90 0.59), and the planner kinds added their own lateral choice on top of the held offset.

## The fix (what the code does now)

`teacher.LaneHold`, installed on the one `RacelineTeacher` every teacher-driven car plans through
(`F1VecEnv.set_teacher`, only where the rolling start can happen: `_roll_cfg`).

* **Held point per station, along the centreline.** A held car is steered toward
  `w * held point + (1 - w) * (raceline offset point it was asked for)` at every point the teacher
  plans through (the six plan points, the pure-pursuit target, the error its slowdown reads, the
  initial curvature knots). The held point is the one at the placed centreline offset, the same arc
  ahead along the held line itself, from the car's own centreline station. Nothing is summed with
  it: the events' offset and a planner kind's own choice are inside the second term, so they are
  scaled by `1 - w` without any planner knowing (cause 2). `w` is 1 through the hold and ramps to 0
  (`F1VecEnv._hold_weight`); at 0 the plan is the planner's own, bit for bit.
* **Clamped where it is applied.** The held offset is cut to the lane's room at each station
  (`room`: swept along the centreline normal against the distance field for half the body plus
  `opp_event_margin` -- the test the abreast row was placed with), never at another point's
  raceline position (cause 1).
* Found on the way, and fixed because the hold could not hold without them:
  * **The raceline is not a usable frame.** A raceline point's foot on the centreline jumps across
    a sharp corner, so the held line is sampled from the car's centreline station
    (`Simulator.cl_idx`), and by its own arc length: on the inside of a bend it is shorter than the
    centreline by `1 - c kappa`, and points placed by centreline arc left a plan fit 0.5 m off them
    (0.13 m of drift in 0.5 s).
  * **A lane position is held at the speed it allows.** The raceline profile is for the raceline's
    radius; held 0.36 m inside a 0.8 m-radius bend at the profile's speed a car slid 0.64 m out in
    0.6 s at full lock. The held car's speed is capped by the curvature of its held line ahead
    (`a_lat` x grip, braking at `a_brake`, fading with `sqrt(w)`). Where the held line is tighter
    than the car can steer at all (inside an ICCAS hairpin) it is driven at full lock and still
    goes wide: the hold delays a merge, it cannot repeal the geometry.
  * **A held car starts lined up with its lane** (no heading jitter; the draw is still made, so the
    random stream is the same on and off): 0.075 rad at 3 m/s carried a held car 6.9 cm toward the
    learner in 0.33 s.
  * **A crate on the held line releases the hold** for that car (`_hold_blocked`): going round it is
    its planner's job, and the placement keeps 3 m clear, not the 3-5 m a held car covers in 1 s.
* CUDA graphs: the per-car tensors (`roll_w`, `roll_cl0`, `roll_ci`) are env tensors written in place
  and installed for the life of the env, and `TeacherGraph` guards the hold's presence, so a captured
  teacher call follows them; the offset channel is still the events' own tensor, never None <-> tensor.

## PREREG

Written before any contact measurement of the code above. The thresholds are the ones the task set.
Disclosed: during development two short CPU diagnostics were run on this code's predecessors
(`m/H_*`, `m/H2_*`: every seat abreast, 1.2-s episodes, 100 learner episodes per arm) to see whether
held cars hold; they also counted 6 / 3 (on) against 12 / 12 (off) raw / charged contacts with the
abreast partner. Nothing below was chosen after them.

**Arms.** ON: `spawn_alongside_hold = 1.0`, `spawn_alongside_ramp = 1.0` (the defaults). OFF: both 0
(`_roll_cfg` False: no lane hold, no yaw alignment, the planners drive from t = 0). Same commit,
same seeds, same placement draws (nothing in the hold draws from a generator).

**Configuration.** The boot-camp `EnvConfig` exactly as `ppo.py` builds it from `ppo_boot.sh`'s flags
(round 2's capture, `ecfg_boot.pkl`): ICCAS and its 16 warps, race of 4, three slots of the
forzaeth / forzaeth_pred / lane_switch / interactive mix (partitioned per race), presence 0.5,
events brake / stop / shift / weave at 0.5 per 10 s, reactive defend 0.3 / line 0.2 / oblivious 0.1,
procedural props (density 1.5, at most 18, movable, on the raceline), soft contacts, 3 m spawn
runway. Learner `checkpoints/iccas_bootcamp_safest.pt`, sampled, with the training dial draw (margin
0.30, exact 0.30), speed cap 9. Harness `scratchpad/roll/m/drive4.py` on GPU index 1 (RTX 5060,
it had > 6 GB free), 64 envs (16 races) per run, eager. Episodes truncated at their length
(`stagger_first_episode` off), so the reset times are the same in both arms.

**Measurement 1 (primary): every seat abreast** (`spawn = "alongside"` for all three slots, presence
0.5 kept), 6-s episodes, 2640 steps x 64 envs = 176 learner episodes per arm, seed 701 in both.
Metric: raw car-contact onsets of the learner (rising edge of `info["car_collision"]` on the learner
row) whose partner -- the car in box overlap after the step, else the nearest present race mate -- is
the opponent laid abreast with the learner at that episode's reset, with the onset in the first 2 s
(step < 80). **PASS-1: ON <= 0.5 x OFF** (point estimate). Reported with it: the charged count, the
exact one-sided conditional binomial p of ON given ON + OFF at the two arms' learner-km.

**Measurement 2 (guard): the boot-camp configuration itself** (spawn grid "random"), 10-s episodes,
4400 steps x 64 envs = 176 learner episodes per arm, seed 702 in both. Metric: all raw car-contact
onsets of the learner in the 10 s, any partner. **PASS-2 (not worse):** the ON rate per learner-km is
not significantly above OFF's -- the one-sided exact conditional binomial test of ON given ON + OFF,
with p0 the ON arm's share of the learner-km, does not reject at alpha 0.05.

**Verdict.** PASS iff PASS-1 and PASS-2. On FAIL the rolling start is removed: its fields default to 0
and its code is deleted, and this document says so.

Also reported, not part of the verdict: raw / charged contacts by time bin (0-1, 1-2, 2-4, 4-6, 6-10
s), the abreast opponents' centreline motion toward the learner at 0.5 / 1 / 1.5 / 2 s (median / p90 /
max, opponents not yet in contact), and the merge timing (first body gap under 0.10 m) and contact
timing with the abreast partner.
