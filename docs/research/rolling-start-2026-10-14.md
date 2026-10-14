# Rolling start for abreast opponents: the fix, and whether it works (2026-10-14)

**Outcome: FAIL by the pre-registered test, removed.** The fixed hold holds -- the abreast
opponent's motion toward the learner at 1 s is a median +0.01 m against +0.23 m without it -- but in
the primary test (every seat abreast) contacts with the abreast partner in the first 2 s went only
32 -> 25 raw (30 -> 22 charged), against the >= 50 % drop it had to show: there the merge came in
the ramp instead. The rolling start (its fields, flags and code) is deleted on this branch; the
measured code is commit 95b069f. **Not part of the verdict, and the reason to read on:** in the
boot-camp configuration itself (the guard measurement, which it passed) the same code cut those
contacts 17 -> 7 and all the learner's car contacts 29 -> 20 raw (28 -> 18 charged) per 176
ten-second episodes. Broken down by how many cars the abreast row had, the hold halves the merges of
a two-car row in both measurements (16 -> 7 each) and does nothing for a row of three or four
(16 -> 18), which is most of what an every-seat-abreast grid lays out.

**Since then:** the hold is back for rows of two only, under a second test pre-registered for it
(see "Restored for rows of two" at the end).

Branch `fix/rolling-start` (from `main` 4db62bc). Background: commit 4db62bc added a rolling start for
teacher-driven opponents placed abreast of the learner (`EnvConfig.spawn_alongside_hold` /
`spawn_alongside_ramp`) and measured that it did **not** reduce the abreast opponent's merge contacts
(every seat abreast, 80 six-second episodes per arm: 23 / 19 raw / charged with it, 20 / 18 without,
first contact at 0.45-0.55 s either way). Two causes were diagnosed on 228 abreast opponents: the held
raceline offset was computed at the pure-pursuit lookahead but cut by the lane clamp at the car's own
raceline point (the 83 cut by more than 5 cm moved 0.21-0.26 m toward the learner within 1 s, median;
p90 0.59), and the planner kinds added their own lateral choice on top of the held offset.

## The fix that was measured (commit 95b069f; removed since)

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

## Result (code 95b069f, measured 14:05-14:56 on the RTX 5060)

Harness `drive4.py`, tables `ana4.py`, raw pickles `P1_*.pkl` / `P2_*.pkl`, all in
`/tmp/claude-1000/-home-shchon11-F1tenth-E2E/d98b5699-f98e-4674-9a12-ab5a57bf35f1/scratchpad/roll/m/`
(the development diagnostics `H*` and the CPU harness `drive3.py` beside them). Both arms of each
measurement ran side by side with the same seed: 216 of 220 (measurement 1) and 79 of 80
(measurement 2) abreast opponents started at the same track station and lateral offset (within 6 mm)
in both arms. The rest differ, which fits the one thing placement does differently with the hold on
(not checked further): a held car spawns with no heading jitter, and the spawn-runway check reads
the heading.
Contacts are raw / charged onsets of the learner; "abreast" is the partner laid abreast with it at
that episode's reset; motion toward the learner is the abreast opponent's centreline offset change,
signed toward the learner, median / p90 / max over opponents not yet in contact with it.

### Measurement 1: every seat abreast, 6-s episodes (primary)

| arm | episodes | learner-km | 0-1 s | 1-2 s | 2-4 s | 4-6 s | total | abreast partner, first 2 s |
|---|---|---|---|---|---|---|---|---|
| ON | 176 | 3.68 | 8 / 7 | 17 / 15 | 17 / 16 | 7 / 2 | 49 / 40 | **25 / 22** |
| OFF | 176 | 3.58 | 14 / 14 | 18 / 16 | 16 / 15 | 5 / 4 | 53 / 49 | **32 / 30** |

(Every early contact here was with the abreast partner: 0-2 s all-partner and abreast counts are the
same; 2-4 s too. In 23 of ON's 25 and 30 of OFF's 32 the partner was the car moving in, side by
side; by kind, ON / OFF: forzaeth 8 / 8, forzaeth_pred 8 / 14, lane_switch 8 / 10, interactive 1 / 0.)

Motion toward the learner (m, median / p90 / max), 220 abreast opponents an arm:

| arm | 0.5 s | 1.0 s | 1.5 s | 2.0 s | max over 0-1 s: p90, share < 0.05 m |
|---|---|---|---|---|---|
| ON | +0.00 / +0.13 / +0.71 | +0.01 / +0.18 / +0.81 | +0.04 / +0.19 / +0.78 | +0.12 / +0.54 / +1.40 | +0.19, 64 % |
| OFF | +0.06 / +0.28 / +0.83 | +0.23 / +0.78 / +1.24 | +0.33 / +1.12 / +1.77 | +0.30 / +1.20 / +1.85 | +0.76, 30 % |

Merge timing with the abreast partner (body gap under 0.10 m / in contact), of 220:

| arm | merged by 1 / 2 / 2.5 s | merge time median (p10-p90) | contact by 1 / 2 / 2.5 s | contact time median (p10-p90) |
|---|---|---|---|---|
| ON | 17 / 41 / 51 | 1.20 s (0.70-2.23) | 7 / 25 / 35 | 1.35 s (0.82-2.17) |
| OFF | 25 / 45 / 54 | 1.01 s (0.47-2.19) | 14 / 28 / 34 | 1.12 s (0.57-2.17) |

26 of ON's 220 held cars were released early by a crate on their held line.

**PASS-1: FAIL.** 25 against 32 is a ratio of 0.78 (needed <= 0.50); one-sided exact conditional
binomial p = 0.21 for ON this low at equal rates (exposure 1.040 / 1.033 learner-km in the first 2 s).

### Measurement 2: the boot-camp configuration, 10-s episodes (guard)

| arm | episodes | learner-km | 0-1 s | 1-2 s | 2-4 s | 4-6 s | 6-10 s | total | abreast partner, first 2 s |
|---|---|---|---|---|---|---|---|---|---|
| ON | 176 | 6.26 | 2 / 1 | 7 / 7 | 5 / 5 | 4 / 3 | 2 / 2 | **20 / 18** | 7 / 6 |
| OFF | 176 | 6.24 | 6 / 6 | 13 / 13 | 3 / 3 | 4 / 3 | 3 / 3 | **29 / 28** | 17 / 17 |

Motion toward the learner, 80 abreast opponents an arm: at 1 s ON -0.00 / +0.06 / +0.41 against OFF
+0.13 / +0.64 / +1.00; at 2 s (after the ramp) ON +0.13 / +0.47 / +0.64 against +0.36 / +1.10 /
+1.39. Merged by 1 / 2 / 2.5 s: ON 3 / 12 / 13, OFF 19 / 30 / 32; in contact: ON 2 / 8 / 8, OFF 9 / 17 /
19. 5 held cars were released by a crate.

**PASS-2: PASS.** 20 against 29 contacts (3.2 against 4.7 per learner-km); one-sided p = 0.93 for ON
this high at equal rates. **VERDICT: FAIL** (PASS-1 failed).

### Not pre-registered: by the size of the abreast row

The two measurements disagree, and the grid explains it. Abreast-partner contacts in the first 2 s
(raw), split by how many cars the learner's abreast row had (the learner included):

| measurement | arm | episodes with a row of 2 / 3 / 4 | contacts in rows of 2 / 3 / 4 |
|---|---|---|---|
| 1 (every seat abreast) | ON | 73 / 48 / 17 | 7 / 15 / 3 |
| 1 | OFF | 73 / 48 / 17 | 16 / 13 / 3 |
| 2 (boot camp) | ON | 60 / 10 / 0 | 7 / 0 / 0 |
| 2 | OFF | 60 / 10 / 0 | 16 / 1 / 0 |

A held pair halves its merges in both (16 -> 7, and 16 -> 7); a row of three or four does not (16 ->
18 together), and the every-seat-abreast grid is mostly such rows. A post-hoc split, so it does not
change the verdict; it says where a rolling start would be worth re-testing (pairs only) -- see
below.

## What it says

* **The hold works; on a crowded row the merge does not care.** Through the hold the held car
  stays put (1 s: median +0.01 m, p90 0.18 m, against +0.23 / 0.78), and contacts in the first
  second halve (14 -> 8). On the every-seat-abreast grid the opponents' own planners -- the spliner
  kinds and the lane switcher drive toward their line or round the learner -- then close the gap as
  soon as the side is handed back: 1-2 s 17 against 18, merge 0.2 s later, contact 0.2 s later,
  and over 2.5 s as many abreast opponents end in contact (35 against 34). Three or four cars abreast
  on an ICCAS lane cannot all keep their lanes through the next bend; a pair can, and there the
  merge mostly does not come (measurement 2: contact by 2.5 s 8 against 19).
* **What is left in the hold is geometry.** On ICCAS 87 of 118 abreast starts (development
  diagnostic `m/H2_*`) are within 3 m of a bend whose held line is tighter than 0.7 1/m -- most of
  them past the car's full lock (a centreline radius under 0.75 m inside the hairpins). There a held
  car slides out whatever it is told, and the p90 / max above are those.
* What would be worth a new pre-registered test, not done here: the same hold for abreast pairs
  only (rows of three or more start staggered or unheld), judged on the boot-camp grid; or an
  opponent that does not drive into a car beside it (a yield for an abreast learner in the planner
  kinds -- an opponent-behaviour change). The placement half of 4db62bc (no step-0 contacts, no
  crowded grid, empty seats off the lap) is unaffected and stays.

## Removed

`EnvConfig.spawn_alongside_hold` / `spawn_alongside_ramp`, `--spawn-alongside-hold` /
`--spawn-alongside-ramp` (console schema regenerated), the env's hold state and `_rolling_start`,
the rollout shadow's copy of its host count, and their tests. `teacher.py`, `interactive_teacher.py`,
`row_planning.py`, `viewer/graph_fastpath.py` are back to 4db62bc. Kept: `spawn_abreast` (which cars
a reset laid abreast; a replayed contact scene is not one). With the rolling start off, the code of
95b069f drove the boot-camp configuration bit for bit like 4db62bc (201 steps x 16 cars, every seat
abreast and the random grid, resets included), so the removal changes nothing for runs that had it
off; runs on 4db62bc's defaults had the first (non-holding) version on.

## Tests of the fixed hold (95b069f / 788da6f, before removal)

* The hold held for every planner kind (raceline, forzaeth, forzaeth_pred, lane_switch,
  interactive): under 5 cm toward the learner through the hold on every start whose held line is
  drivable (no bend tighter than 0.3 1/m over the 4 m it covers, blackbox2022_1, plan action space);
  all five kinds gave identical drifts, i.e. the planner's choice was fully overridden. A start in a
  0.4 1/m bend at 3.5 m/s still went 6.9 cm out and 12 cm back in 0.3-1 s (zero steer at spawn).
* At weight 1 a planner's own offset (+0.45 m or -0.45 m) did not move the plan; at weight 0 the plan
  was bit for bit the planner's own with no hold installed.
* An idle hold changed no trajectory bit for bit; the captured teacher graph replayed the hold bit
  for bit (CUDA, every kind, through hold, ramp and after).

## Restored for rows of two

The post-hoc split above is the reason for a second test with its own pre-registration. The hold of
95b069f / 788da6f is back -- `EnvConfig.spawn_alongside_hold` / `spawn_alongside_ramp` (defaults 1.0
/ 1.0), `--spawn-alongside-hold` / `--spawn-alongside-ramp` (console schema regenerated), the env's
hold state, `teacher.LaneHold` and its users, the rollout shadow's copy of the host count, and their
tests -- with one change: only an abreast row of exactly two cars, the learner and one
teacher-driven opponent, is held and lined up at the spawn (`F1VecEnv._abreast_pair`, read from the
row actually placed, empty seats excluded). A car of a row of three or four starts as with the
rolling start off, bit for bit (`test_only_a_row_of_two_is_held`: race of 4 on the random grid, every
race without a pair identical to the off arm over 2.5 s). On the boot-camp grid nearly every row is
a pair (measurement 2: 60 of the 70 learner episodes that had a row). If the new test fails, this
goes again, back to 45d1113.
