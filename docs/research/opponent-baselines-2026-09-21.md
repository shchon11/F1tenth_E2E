# Somebody else's driver to race against (2026-09-21)

Every opponent this project could put on the track was one of ours. The raceline teacher has never
heard of the other cars — in a race it is kept off them by the env's follow cap, which only ever
slows it down. The interactive teacher is our own best-response search. A checkpoint is our own
policy. So "our policy overtakes" has, until now, only ever been a statement about us.

This note is three published-planner opponents, what each one is and is not, and the first
measurement of a policy of ours against them. Everything below is simulation.

Companion to [`baselines-2026-09-15.md`](baselines-2026-09-15.md), which is a different axis: that
one races published *end-to-end policies* as the learner. This one puts a published *classical
planner* in the other car.

## 1. What is reproduced, and what is not

| kind | module | what it is |
| --- | --- | --- |
| `forzaeth` | `f1sim/spliner_teacher.py` | ForzaETH race stack's `spliner` local planner, from its own source and paper (arXiv:2403.11784). Seven spline control points around the other car, the side with room, the side-switch lock, Racing/Trailing/Overtaking. |
| `forzaeth_pred` | same module, one default | the same planner aimed where the other car will be in 0.5 s — the stack's `predictive spliner`. |
| `lane_switch` | `f1sim/lane_teacher.py` | fixed lanes offset from the racing line, pick the cheapest clear one, hold it with hysteresis, move at a bounded rate. **Not a reproduction of anything** — see §1.2. |

### 1.1 The ForzaETH pair

Reproduced from the published stack: the seven control points at arc offsets
`-4.0, -3.0, -1.5, 0, +2.0, +3.0, +4.0` m, `evasion_dist` 0.65 m, `spline_bound_mindist` 0.2 m,
`obs_traj_tresh` 0.3 m, `lookahead` 10 m, the 0.25 m side-switch lock, the inside-overtake speed
scale 0.9, and the distance scaling `clip(1 + v/v_max, 1.0, 1.5)`. Each is a named constant in the
module so a reader can check it against the source rather than trust a docstring.

**Not** reproduced, and deliberately: the ROS graph, Cartographer localisation, the opponent
detector and the EKF. This simulator does those exactly and the planner reads its state from it, so
reproducing their *error* would make the baseline worse than the one the authors measured, not more
faithful. The MAP controller is likewise absent, because `RacelineTeacher` already tracks a line
with this simulator's own tyre model — a better tracker of these cars than a LUT fitted to theirs.
What is reproduced is the part that is the contribution: **where the line goes when there is a car
in the way.**

The two planners differ in exactly one number, so they are one class and a changed default
(`fixed_pred_time`), not two modules. A test asserts the identity: the predictive planner facing a
car `g` ahead doing `v` draws the same line the plain one draws facing a *stopped* car
`g + 0.5 v` ahead. Anything else would mean the prediction had leaked into a second decision.

One caveat on `forzaeth_pred`, which matters for reading its numbers: upstream reaches the
opponent's future position through a learned model of its speed around the lap. Here the opponent's
speed is known exactly. This is therefore the **optimistic end** of that planner — what it would do
with a perfect predictor — and not a like-for-like port.

### 1.2 The lane switcher, and why it is not called UNICORN

The ask was to reproduce UNIST's UNICORN alongside ForzaETH. That cannot be done at the same
fidelity and the difference is worth stating rather than glossing. ForzaETH is open source. What is
public about UNICORN is a competition page naming components — Cartographer, a rule-based state
machine, a Lane Change Planner, Frenet-frame tracking, Advanced Pure Pursuit / L1. No parameters,
no source, nothing to check an implementation against.

So `lane_switch` implements what that composition *is*, and which several teams race: a small fixed
set of lanes, the cheapest clear one, hysteresis, a bounded transition. It is a genuinely different
comparison point from the spline family — discrete choice rather than a continuous curve — which is
why it is worth having even though it is nobody's code. **Its numbers are ours and are marked as
ours in the module**; `spliner_teacher`'s are marked as upstream's. A reader of a result has to be
able to tell those apart, so a test asserts the registry entry still says
*"재현이 아니라 같은 계열의 구현"*.

## 2. How they are wired

One entry each in `opponent_slots.KINDS`, which is what that registry promised when it was built.
The shared half — the Frenet view of a race, the clearance probe, how an offset reaches the
reference teacher, the wiring that keeps a slot's speed band and grip label on it — is
`opponent_planner.FrenetOpponentPlanner`; a baseline is the `decide` on top.

Two properties of that base are load-bearing:

* **The reference teacher is never reimplemented.** A baseline returns a lateral offset; the line
  that comes out is `RacelineTeacher`'s line through a displaced reference. A baseline that drew
  its own line would be measured against a different driver than the solo control.
* **Nothing in the path is a host sync.** These run inside the env's per-step opponent command for
  every row, in training as well as evaluation.

They reach a run through `--opp-slots '[{"kind":"forzaeth"}]'` — in training, in the console's slot
table, and now in `learn.evaluate`, which previously accepted only `policy` and `teacher` and so
could not race any of them.

## 3. Three bugs that stood between the baselines and a number

All three were only reachable on a raceline cache miss, which is why they had never fired: training
reuses a prebuilt line, and the evaluation afterwards is where a new map first gets built. The same
shape as the `Raceline.build(objective=)` regression of 2026-09-19.

1. **`learn.evaluate` is `@torch.no_grad()` end to end** and the minimum-time raceline solver needs
   reverse-mode autograd. The *first* evaluation on any new map has always died with
   `element 0 of tensors does not require grad`. `enable_grad` now sits on `MinimumTime.restore`,
   where the requirement is.
2. **Restoring `objective` put it in `build_cached`'s key**, and `objective=None` is not a choice —
   `build` resolves it into `optimize_lap_time` before doing anything. Every raceline ever cached
   (737 files on this machine) was missing and rebuilding into a file identical to its neighbour.
3. **`evaluate` never called `torch.cuda.set_device`**, so an evaluation on the second card got a
   Triton kernel compiled for the first: `no kernel image is available for execution on the
   device`. Same cause as `8313f1f` fixed in the viewer, a second place.

## 4. The first measurement

`spec_korea_traffic_s906` at 6.3M steps, `real:korea_2025_iccas`, 16 races × 2000 steps, one
opponent at `speed_scale [0.7, 0.9]`, seed 41, eager. The teacher limits and raceline objective are
the run's own (`7 / 6.5 / 4`, `min_time`, dial 0.30).

| | `raceline` | `forzaeth` | `forzaeth_pred` | `lane_switch` |
| --- | ---: | ---: | ---: | ---: |
| lap [s] | 8.03 | 8.27 | 8.00 | 7.92 |
| collisions / km | 2.14 | **18.11** | 13.29 | 11.98 |
| passes | 27 | **13** | 23 | 21 |
| passes / learner-min | 2.02 | **0.97** | 1.72 | 1.57 |
| pace vs opponent | 1.19 | 1.16 | 1.25 | 1.22 |
| following fraction | 0.33 | **0.68** | 0.37 | — |
| defending fraction | 0.40 | **0.18** | 0.37 | — |
| car contacts | 5 | **44** | 43 | — |
| wall collisions | 3 | **21** | 5 | 7 |

The pace advantage is about the same against all four (1.16–1.25). What changes is everything else:
against a car that **holds its line** the policy passes 27 times and touches it 5 times; against a
car that **moves sideways to evade** it passes 13 times and touches it 44. It spends twice as long
behind (0.68 vs 0.33) and half as long ahead (0.18 vs 0.40).

The reading that fits: this policy was trained entirely against opponents that hold a line, and has
never seen a car change line. That is a statement about our training mix, not about the ForzaETH
planner — which is the whole reason for having a driver somebody else designed in the other car.

**What these numbers do not say.** `car_contacts` counts contacts involving the learner and does
not attribute fault; the baselines are themselves attempting passes (measured over a 600-step race:
`forzaeth` 7.0 % overtaking / 26.4 % trailing, `forzaeth_pred` 7.7 % / 30.3 %, `lane_switch` 17.7 %
changing / 14.5 % trailing), so some share of those 44 is the other car arriving. And this is one
seed. A second is running; until it lands, treat `forzaeth` 21 walls against `forzaeth_pred` 5 as
suggestive and not as a difference between the two planners.

## 5. What this suggests next

Put `forzaeth` in the training opponent mix. The gap above is not a pace gap and not a collision
budget gap — it is that the policy has no answer to a car that changes line, and the cheapest way
to give it one is to let it race against a driver that does, for the same reason the traffic run was
started against obstacles at all.

## 6. Files

* `f1sim/f1sim/opponent_planner.py` — the shared Frenet view.
* `f1sim/f1sim/spliner_teacher.py` — `SplinerTeacher`, `PredictiveSplinerTeacher`.
* `f1sim/f1sim/lane_teacher.py` — `LaneSwitchTeacher`.
* `f1sim/tests/test_spliner_teacher.py` — 17 tests, 47 s.
* `f1sim/f1sim/opponent_slots.py` — the three registry entries.
* `f1sim/f1sim/learn/evaluate.py` — `--opp-slots`, and the device fix.
