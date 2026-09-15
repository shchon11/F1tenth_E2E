# Checkpoint benchmark

A development leaderboard for comparing trained checkpoints on fixed scenarios: driving, stability,
per-surface retention, obstacle avoidance and overtaking. It ships inside the installed package, so
it runs from a normal `pip install -e f1sim` with no extra path setup.

```bash
python3 -m f1sim.learn.benchmark --help
```

A measured run of this suite is published as a research note:
[Checkpoint benchmark v1 — three-system subset](research/benchmark-v1-2026-09-12.md) (3 systems, 102 cells, 816 trials). It is a subset chosen for time, not the 17-system benchmark.

The measured runs are published as a rendered leaderboard: [Leaderboard](leaderboard/README.md) —
six metrics per cohort with exact counts, paired pace against a reference and the evidence behind
each row. `python3 -m f1sim.learn.leaderboard` builds it from existing result files, validating each
cohort through `report.validate_results` first; it scores nothing and needs no weights or GPU.
`leaderboard/index.html` is the same report as a self-contained offline page.

**Three suites, three questions.** **v1 is in-distribution**: its three maps are reused development
tracks -- `gen:control:1400` is an SGR training map and `real:korea_2026_competition` is trained
through twenty obstacle variants -- so a v1 number says how well the training distribution was
fitted and is not evidence of generalisation to an unseen venue. **v2 is held out**: every map in
it is outside `common.TRAIN_TRACKS` in every variant, including two real floors this car drove on
that the simulator did not previously contain. **v2.1 is v2 plus traffic**: the same 64 cells,
unchanged, plus an 80-cell **T** family that measures what a policy does with other cars around it
on those same held-out floors -- including the two where a completed pass could not be demonstrated
and O was therefore dropped. **Adoption decisions use v2.1** (see
[Suite v2.1](#suite-v21-the-t-traffic-family)). No suite is an on-car claim. Scope is restated in
every generated report, which reads it from the suite's own provenance field.

Map names on this page are written in the **loader** grammar (`real:korea_2026_competition`,
`gen:competition:9200+pinch9200`), and stay that way: they are the frozen suites' own cell ids
(`S:gen:control:1400:0.73423:4401`), so every result file already published is keyed by them.
[Tracks](tracks.md) has the short id for each (`real/korea26`, `gen/comp-9200#pinch:9200`) and the
table that converts between the two.

## The pipeline

| stage | loads a checkpoint? | needs a GPU? | what it does |
| --- | --- | --- | --- |
| `plan` | no | no | matrix, trial counts, cost projection, roster **file-pin** verification |
| `geometry` | no | no | proves obstacle placements (or **inherits** a predecessor's), optionally times a smoke run, then **freezes** the suite |
| `gate` | no | **no — runs on CPU** | proves the opponent does not react to the candidate's controller |
| `feasibility` | no | no | scripted-expert check that each scenario is achievable at all |
| `run` | **yes** | yes, with `--lease` | scores one pinned system against the frozen suite |
| `report` | no | no | validates raw results and renders the leaderboard |

`plan` and `geometry` never load an evaluated checkpoint. That ordering is deliberate: scenario
choice cannot be influenced by how a candidate happens to score on it. `run` refuses to score against
a suite that is not frozen.

## The suite (v1, in-distribution)

This section describes v1. The held-out set is [v2](#held-out-suite-v2); it shares every protocol
field below and differs only in which maps they are applied to.

`plan` prints the matrix without touching a checkpoint:

```console
$ python3 -m f1sim.learn.benchmark plan
suite v1  freeze (not frozen)
cells 34   trials/system 272   S=144  A=64  O=64
projected cost: unavailable (pass --map-lengths for the pinned geometry file)
```

Three scenario families, resolved from the suite definition:

| family | maps | µ | seeds | cells | trials |
| --- | --- | --- | --- | ---: | ---: |
| **S** solo | `gen:control:1400`, `real:korea_2026_competition`, `gen:control:9100` | 0.73423, 0.94401, 1.15379 | 4401, 4402 | 18 | 144 |
| **A** avoidance | `gen:control:1400`, `gen:control:9100` | 0.73423, 0.94401 | 4401, 4402 | 8 | 64 |
| **O** overtaking | `gen:control:1400`, `gen:control:9100` | 0.73423, 0.94401 | 4401, 4402 | 8 | 64 |

**34 cells, 272 trials per system.** S runs 8 environments per cell; O runs 8 learners against
`race_size` 2, so 16 cars. A cell id is `suite:map:mu:seed`, e.g.
`S:gen:control:1400:0.73423:4401`.

Friction is **static per episode**: µ is set per cell and constant for the whole run. Only a reset
may resample it. High µ appears in S only — A and O use low and mid, where the interesting failures
are.

Protocol fields that change outcomes live in the suite and are hashed with it: `speed_cap` 9.0 m/s,
`budget_laps` 3.0, `sensor_noise` on, opponent `teacher` at `opp_speed_range` (0.6, 0.8),
`hold_seconds` 1.0, `obstacle_window_m` 4.0. A benchmark whose speed cap drifted silently is not the
same benchmark, so these are part of the freeze rather than command-line options.

### Freezing

```bash
python3 -m f1sim.learn.benchmark geometry --suite suite-v1.json --measure --freeze
```

This proves each obstacle placement actually blocks the racing line while leaving a traversable
corridor, and writes the suite with a `freeze_sha256`. **`--measure` is what times a teacher-driven
cell per map**; without it no timing is taken and the suite's `calibration.measured` is `false`.
`suite.load()`
recomputes that hash and **refuses a file edited after freezing**. Placement geometry and the
measured calibration are recorded in the suite; the calibration is explicitly excluded from the
freeze hash, because re-measuring a rate is a fact about the machine rather than about the scenario.

Run `geometry` before any system is scored, and do not re-run it to "fix" a scenario after seeing
results.

## Held-out suite v2

v1 answers "how well was the training distribution fitted". That is worth measuring and it is not
what an adoption decision needs, and while v1 was the only suite there was no number for the other
question at all: of its three maps, `gen:control:1400` is in `TRAIN_TRACKS`, the geometry of
`real:korea_2026_competition` is trained through twenty obstacle variants (`common.KOREA26_TRAIN`),
and only `gen:control:9100` was unseen.

**v2 changes the maps and nothing else.** Friction levels, seeds, envs, race size, speed cap, lap
budget, sensor noise, opponent and every other protocol field are identical to v1, so a v1 row and
a v2 row differ in the scenario and in nothing that could explain a difference away.

```bash
python3 -m f1sim.learn.benchmark geometry --version v2 \
    --suite suite-v2.json --s-obs 10 --measure --freeze
```

| family | maps | µ | seeds | cells | trials |
| --- | --- | --- | --- | ---: | ---: |
| **S** solo | `real:map16x07`, `real:map12x16`, `real:korea_2025_iccas`, `real:blackbox2022_3`, `rt:Monza`, `gen:competition:0`, `gen:control:9100`, `gen:competition:9200+pinch9200` | 0.73423, 0.94401, 1.15379 | 4401, 4402 | 48 | 384 |
| **A** avoidance | `real:map16x07`, `real:map12x16`, `gen:control:9100` | 0.73423, 0.94401 | 4401, 4402 | 12 | 96 |
| **O** overtaking | `gen:control:9100` | 0.73423, 0.94401 | 4401, 4402 | 4 | 32 |

**64 cells, 512 trials per system**, against v1's 34 and 272. The frozen definition ships as
`f1sim/learn/benchmark/suite-v2.example.json`, freeze hash
`89805514350d932a36cbfe28eed7ca379ec92d71806ec5202736aa61805394f9`.

### What "held out" means here

The solo family is every base map in `common.HELDOUT_TRACKS` that has a raceline the car fits
through -- all eight, measured. `HELDOUT_TRACKS` also carries `~rev` for each real floor; the suite
takes the base map once, because a benchmark cell is a scenario rather than a direction.

The two entries that make the claim worth anything are `real:map16x07` (15.5 × 7.0 m, the
pre-competition hairpin loop) and `real:map12x16` (12.3 × 16.4 m), extracted from the team's own
recordings with `scripts/extract_bag_map.py`. They are registered with a `duct` boundary; the
evidence is in the held-out suite report, and the short version is that the extracted grids are
binary with no unknown class and so carry no wall-thickness information at all, while both free
regions are closed bands around free-standing islands -- a track laid out on open floor, not a
room -- on the same rig and from the same recordings as `korea_2026_competition`.

`real:korea_2026_competition` is **not** in v2. It stays in `TRAIN_TRACKS`, and a score on it, with
or without an obstacle seed it has not seen, measures a memorised circuit. Putting it in an eval
list was the specific defect v2 exists to fix.

Leakage is a test, not a convention. `common.heldout_leakage(train, heldout)` compares *base maps*,
with `~rev`, `~mir`, `+obs`, `+rlobs`, `+pinch` and `+props` stripped through the catalog's own
splitters, so `real:map16x07+obs5~mir` in a training list is caught even though the string differs
from anything in the held-out list. `tests/test_heldout_split.py` fails if it ever returns anything.

### The paired families are smaller than the solo family, deliberately

A paired cell has to be demonstrated before it is frozen. An avoidance cell needs a proven blocking
obstacle — one that really obstructs the racing line while leaving a car-wide corridor connected to
the lane either side, eroded by the car's own footprint — and an overtaking cell needs a pass a
scripted reference driver can actually complete. Where that cannot be shown, the map leaves that
family. The proofs are never trimmed to admit a map.

**A** runs on the two real floors plus `gen:control:9100`: all three placements pass on the first
position tried, and the avoidance expert clears every one at both friction levels.

**O** runs on `gen:control:9100` alone. Both real floors were declared, measured and dropped:
`feasibility` shows **0/4 at both friction levels** on each, the failures being collisions with the
track rather than contact with the car being passed. Reversing the reference driver's side and
cutting its lateral offset from 0.40 m to 0.25 m recovered nothing, so this is not a
side-of-the-track accident. It is also not proof that the floors are too narrow — two cars abreast
fit everywhere on both (narrowest lane 1.000 m and 1.082 m against 0.620 m of car). What has not
been shown is a pass by *this* reference driver, whose 0.40 m offset already exceeds the lane over
2.0 % of `map16x07` and whose 7 m engage window is a fifth of that map's 33 m lap. A stronger
reference driver may restore these cells; until one shows a pass, they stay out.

The obstacle proofs, at `--s-obs 10`:

| map | lane | half-lane at s | obstacle | corridor (left/right) | required |
| --- | ---: | ---: | --- | --- | ---: |
| `real:map16x07` | 33.22 m | 0.667 m | 0.29 × 0.724 m, 79 cells | 0.60 / 0.05 m | 0.51 m |
| `real:map12x16` | 36.06 m | 0.962 m | 0.29 × 1.058 m, 123 cells | 0.85 / 0.00 m | 0.51 m |
| `gen:control:9100` | 56.89 m | 1.282 m | 0.29 × 1.410 m, 163 cells | 1.15 / 0.05 m | 0.51 m |

`real:map16x07` is the tightest scenario in either suite: a 0.667 m half-lane against 1.282 m on
`gen:control:9100`, and a 0.60 m corridor against a 0.51 m requirement. That is the point of it.

### Which suite to quote

| question | suite |
| --- | --- |
| did this checkpoint learn the training distribution at all | v1 |
| does it hold up on geometry it has never seen | **v2** |
| does it handle traffic on that geometry | **v2.1** |
| should we adopt it | **v2.1** |

A v1 improvement with no v2 improvement is a fit to the training maps, and the two suites are
reported separately for that reason. Rows are never pooled across suites: the freeze hash differs,
and `report.validate_results` refuses a file that mixes them.

## Suite v2.1: the T (traffic) family

v2.1 is **v2 plus one family**. Every S, A and O cell in it is v2's cell — same map, same friction,
same seed, same obstacle box, byte for byte — so a v2 avoidance row and a v2.1 avoidance row are
the same measurement, and only T is new. The obstacle placements are *inherited* from v2's frozen
file rather than re-derived, because the placement search depends on `--s-obs` and v2 was frozen at
10 while the flag defaults to 20; a re-derived box is a different scenario wearing the same name.

```bash
python3 -m f1sim.learn.benchmark geometry --version v2.1 \
    --suite suite-v2.1.json \
    --inherit-placements f1sim/learn/benchmark/suite-v2.example.json --freeze
```

The frozen definition ships as `f1sim/learn/benchmark/suite-v2.1.example.json`, freeze hash
`ec7a5bbfdd81d646eabd8bdc1a8de1af2ef757e4351048e3655ca210ae9b7a41`. **144 cells, 1152 trials per
system**, against v2's 64 and 512.

### Why T is not more O

O asks **"was a pass completed and held?"** and answers with a binary. That is the right question
where a pass is something a driver has been shown to do — and on the two unseen real floors it is
not. The held-out worker declared O there, measured **0/4 at both friction levels on each**, tried
the other side and a smaller offset, recovered nothing, and dropped the cells rather than admit them
undemonstrated. v2's overtaking number is therefore one generated map and 32 trials, which cannot
tell whether a recipe improved overtaking.

The narrowness is real: `real:map16x07` has a 0.702 m median half-width against 0.620 m of two cars
abreast. So the fix is not a bigger denominator on the same binary. It is a metric whose floor is
not zero.

**T's trial outcome is a clean contested run**: the learner came through the traffic stint with no
wall collision and no contact with another car. That is achievable on a 0.70 m half-lane — it is
what a car in traffic has to do before anything else counts — and it is what the feasibility check
demonstrates. A completed pass is reported *next to* it as a count that is allowed to be zero, never
as the gate.

| column | direction | what it is |
| --- | --- | --- |
| **clean** | ↑ | the trial outcome: finished the stint, no wall, no car contact |
| **passes/race** | ↑ | completed, held passes — counted over the stint, not latched at one |
| **pace vs opponent** | ↑ | the learner's arc ÷ the opponents' arc over the same steps |
| **attacking s/race** | ↑ | seconds spent within `attack_range_m` (3 m) behind a car |
| **following s/race** | — | seconds spent within `contention_range_m` (12 m) behind a car |
| **car contact** | ↓ | trials that ended by touching another car |
| **wall collisions** | ↓ | trials that ended in the track |
| **leads lost** | ↓ | a clear lead taken and then given back |

**No composite**, as everywhere else here. The panel is the metric, and the columns are designed to
refuse each other's blind spots: "clean" on its own is gameable — a car that hangs back and never
tries never crashes — and **pace vs opponent** is the column that catches it, scoring that car well
under 1.0 while a car that actually races scores at or above it. Neither is worth reading alone.

**Two windows, because one saturates.** `contention_range_m` is the env's own `overtake_range`,
12 m, which is the distance beyond which the reward already stops treating two cars as racing. On a
33.2 m lap — `map16x07` — a gap of 12 m or less is three quarters of every gap the two cars can be
at, so the wide window reads ~1.0 for everybody and separates nobody. `attack_range_m` is 3.0 m,
about five car lengths, close enough that a pass is actually on, and it does not saturate on any map
in the family. Both are frozen in the suite and both are reported.

### The scenarios

Scenarios, not counts. Four of them, each varying one axis against a comparable baseline:

| variant | opponent | cars | events | what it isolates |
| --- | --- | ---: | --- | --- |
| `slow` | 0.5–0.7× profile | 2 | — | a clearly slower car: the pass is available, the question is whether it is taken and taken cleanly |
| `pace` | 0.8–0.95× profile | 2 | — | a car at nearly the learner's own pace: passes are rare by construction, so the cell is carried by pace and by staying clean through a whole stint in close company |
| `event` | 0.6–0.8× profile | 2 | `brake,stop,shift` @ 3.0 /10 s | the scripted behaviours of `f1sim.opponent_events`. Nothing else in the benchmark evaluates against them |
| `pair` | 0.6–0.8× profile | 3 | — | two opponents: the lane beside the car ahead is no longer assumed empty |

`event` and `pair` hold the speed range at v2's own `(0.6, 0.8)`, so each differs from a comparable
baseline in exactly one thing. `weave` is deliberately absent: it is a small continuous oscillation
rather than a decision to react to.

**Maps, friction, seeds.** The five held-out maps `real:map16x07`, `real:map12x16`,
`gen:control:9100`, `real:korea_2025_iccas`, `gen:competition:0`, at µ 0.73423 and 0.94401, seeds
4401 and 4402. **4 × 5 × 2 × 2 = 80 cells, 640 trials per system**, on top of v2's 64 cells and 512
trials: **144 cells, 1152 trials** for v2.1 as a whole.

A T cell id carries its scenario — `T:slow:real:map16x07:0.94401:4401` — because four T cells share a
map, a friction and a seed and differ only in what the opponent is doing, which is the point of them.
Without the variant in the identity every duplicate check in the report would read them as one cell
measured four times.

### The leakage guard

Every T map is a base map of `common.HELDOUT_TRACKS`, checked by `suite.assert_heldout_maps` before
the suite is built and again from outside by `tests/test_heldout_split.py` — once against the code
that declares the maps and once against the **frozen file**, because a suite is shipped as a file and
reproduced from one. The comparison is by base map through `common.base_map`, so a `~rev`, `~mir`,
`+obs`, `+rlobs`, `+pinch` or `+props` variant of a training venue cannot enter under a different
string. The guard raises rather than warns, and raises if it cannot import the split at all: a
leakage guard that quietly does not run is worse than none, because the suite then carries a claim
nobody checked.

### Feasibility: what has to be shown, and what deliberately does not

A T cell is admitted when the scripted, checkpoint-free reference driver shows three things, on the
scoring path:

1. **at least one clean trial** — the stint is survivable;
2. **at least one trial that met traffic** inside the contention window — the cell measures traffic
   rather than a lonely lap;
3. on `event`, **an event that landed inside that window** — measured, not assumed. An opponent that
   brakes half a lap away is a schedule entry, not something the learner had to react to, and a cell
   whose defining feature only shows up sometimes is two cells sharing a name.

**A completed pass is not one of the conditions.** Making it one would drop the same four scenarios
O lost and leave the traffic number a generated-map number again. Passes by the reference driver are
reported alongside as evidence about headroom, and gate nothing.

The reference driver is `experts.TrafficExpert`, and it is the answer to the follow-up the held-out
worker left. `PassExpert` takes a fixed 0.40 m off the centreline whenever a car is within a fixed
7 m — on a 33 m lap that is a fifth of the way round held at an offset that already exceeds the lane
over 2 % of it. `TrafficExpert` differs in three things:

* the offset is bounded per centreline point by the track's own distance field (clearance − the
  car's half-width − a margin), the same construction `opponent_events.raceline_offset_limit` uses
  for a scripted lane change;
* **a pass that does not fit is not attempted** — where the bound falls below what a car needs to get
  past another, it holds station behind instead. "The reference driver crashed" is not evidence about
  the scenario;
* it can actually stop, because a 0.5 m/s creep into a car a `stop` event has parked is a contact.

```bash
python3 -m f1sim.learn.benchmark feasibility --suite suite-v2.1.json --envs 4 --device cpu
python3 -m f1sim.learn.benchmark feasibility --suite suite-v2.1.json --envs 4 --only T:event
```

Measured before the freeze, CPU, seed 4401, 4 trials per cell — **40/40 cells feasible, none
dropped**:

| scenario | clean | contended | passes | pace vs opponent | event landed in window |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slow` | 39/40 | 40/40 | 10 | 1.16 | — |
| `pace` | 37/40 | 40/40 | 0 | 1.00 | — |
| `event` | 27/40 | 40/40 | 6 | 1.17 | **40/40** (99 % of event seconds) |
| `pair` | 35/40 | 40/40 | 0 | 1.03 | — |

The panel separates the scenarios the way the design predicts, which is the first evidence that it
measures anything: `pace` is where ground cannot be gained (1.00, zero passes, by construction) and
is also nearly the easiest to survive — the pair of numbers that "clean alone is gameable"
describes; `event` is the hardest to survive *and* one of the two where ground is gained, because a
car that brakes is both a hazard and an opportunity; `slow` is where the passes are and `pair` is
not, because a second car removes the opening the same speed advantage would otherwise give.

**The event rate is measured, not assumed.** At 3.0 events per opponent per 10 s an event lands
inside the learner's contention window in 40 of 40 event-scenario trials, and 99 % of all opponent
event seconds are spent inside it. A scenario whose defining feature only showed up sometimes would
be two scenarios sharing a name.

**Zero passes on `real:map16x07` in every scenario is the finding, not a gap.** The room on the side
the reference driver uses, taken as the minimum over the next 3 m of arc, is a median 0.25 m there
against 0.80 m on `real:korea_2025_iccas`, so a driver that will not commit below 0.70 m never
pulls out — and stays clean 15/16 times instead. This is the same geometry that took the O family
off that floor; the difference is that it is now a number the cell reports rather than a reason the
cell cannot exist.

### Checking traffic without the suite

`evaluate` takes the same opponent flags as training and reports the same metrics, from the same
implementation (`benchmark.overtake.TrafficMeter` wraps `TrafficTrace` and the same pass detector):

```bash
python3 -m f1sim.learn.evaluate "$CKPT" --per-track --envs 64 \
    --race-size 2 --opponent teacher --opp-events brake,stop,shift --opp-event-rate 3
```

It loads no roster, freezes nothing, and **its numbers are not benchmark scores**. It also runs a
different protocol — auto-resetting envs, where there is no such thing as a trial — so it reports
rates over the rollout rather than per-trial outcomes, and a `traffic` block from it is not
comparable with a T table.

## The roster

The roster is a JSON object with a `systems` list pinning exactly which weights are being compared —
**not a bare array**. Every entry is verified before anything is scored.

```json
{
  "systems": [
    {
      "system_id": "sgr_aux_anchor_s701",
      "path": "/abs/path/cl_sgr_aux_anchor_s701/ppo_final.pt",
      "checkpoint_sha256": "…64 hex…",
      "controller_arm": "estimated",
      "estimator_path": "/abs/path/estimator_seed401.pt",
      "estimator_sha256": "a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf",
      "note": "SGR screen final, seed 701"
    },
    {
      "system_id": "frozen_original_estimated",
      "path": "/abs/path/baseline/ppo_latest_frozen.pt",
      "checkpoint_sha256": "…64 hex…",
      "controller_arm": "estimated",
      "estimator_path": "/abs/path/estimator_seed401.pt",
      "estimator_sha256": "a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf",
      "cross_runtime": true,
      "note": "legacy-trained weights evaluated under the estimated arm — declared deliberately"
    }
  ]
}
```

| field | meaning |
| --- | --- |
| `system_id` | unique label; duplicates are refused |
| `path` | the real file. Not a symlink, and no `latest` path segment |
| `checkpoint_sha256` | re-hashed at verification; a mismatch refuses |
| `controller_arm` | a `grip_runtime.ARMS` name: a tracker base (`legacy`, `fixed_low`, `estimated`; `oracle` is privileged and not pinnable) optionally wearing the composable layers `+clearance` and `+tcs`, in that order — e.g. `fixed_low+clearance` |
| `estimator_path` / `estimator_sha256` | required **iff** the arm is `estimated`; both, not one |
| `cross_runtime` | declares a checkpoint evaluated under an arm it was not trained under |
| `note` | free text, carried into the report |

### What verification refuses

- **Moving pointers.** `ppo_latest.pt`, `latest.pt` and `last.pt` are rewritten in place by training.
  A roster may not point at one — pin an immutable copy. Note this is a check on the *basename*, so a
  legitimately frozen file such as `ppo_latest_frozen.pt` is fine.
- **Symlinks**, and any unresolved `latest` directory segment.
- **A drifted file** — the sha is recomputed, not trusted.
- **A bare estimator sha with no resolvable file.** A sha the runtime cannot load is not a pin, and
  it must fail here rather than hours later on the GPU.
- **Arm/estimator disagreement** in either direction: an `estimated` entry without an estimator, or a
  non-`estimated` entry carrying one.
- **An arm that contradicts the checkpoint's own record** — see below.
- **Duplicate identity** — two entries with the same (weights, arm, estimator) triple, even under
  different `system_id`s.

### External systems: the published baselines

A roster entry may pin a **published baseline** instead of one of our checkpoints. Those models
(`f1sim.learn.baselines`) emit a steering angle and a speed and nothing else, so they run in the
`direct` action mode with no plan tracker at all — which means they have no controller arm, and the
entry has to say so:

```json
{
  "system_id": "end2race@none",
  "kind": "end2race",
  "weights": "/abs/path/pretrained/end2race.pth",
  "checkpoint_sha256": "…64 hex…",
  "controller_arm": "none",
  "options": {"scan_fill": 30.0},
  "note": "arXiv 2509.16894, pretrained"
}
```

| field | meaning |
| --- | --- |
| `kind` | `tinylidarnet` or `end2race`. Its presence is what makes the entry external |
| `weights` | the model file. An accepted spelling of `path`; every pin check (symlink, moving pointer, unresolved `latest`, sha) applies to it unchanged |
| `controller_arm` | must be **`"none"`**. Not `legacy` — that means "our plan tracker, with nothing installed on it", which is a different system |
| `options` | the driver's own options, and **part of the row identity** |

`options` being part of the identity is the same rule as "the same weights under two arms are two
systems": End2Race with the quarter of its scan this car cannot see filled at 30 m, and the same
weights filled at 0 m, are two evaluated systems, and a roster that declared them with one
`system_id` each but the same options is refused as a duplicate identity.

What is refused, and why each one is a category error rather than a typo:

- an external entry naming **any** real arm — there is no plan for a tracker to follow and no solver
  for an arm to wrap;
- `controller_arm: "none"` on an entry with no `kind` — only a baseline can declare the absence of a
  plan tracker;
- an **estimator pin** on an external entry — no arm, so no friction estimator either;
- `cross_runtime` on an external entry — it declares a checkpoint evaluated under an arm it did not
  train under, and there is no arm;
- an unknown `kind`, and any roster field this loader would otherwise drop (a dropped field is a pin
  nobody is checking).

The cell such an entry builds differs from a checkpoint's in exactly three ways, all recorded in the
row's `effective` block: `action_mode` is `direct`, there is no controller and no
[routed tracker](#independence-of-the-opponent), and `external` carries what the driver actually was
— its backend, beam count, clipping, speed mapping and, for End2Race, what filled the bearings this
car cannot see.

The routed tracker's absence is not a gap in the protocol. It exists because the plan path pushes
every car's plan through one tracker object, so the candidate's arm would otherwise move the
opponent. In `direct` mode there is no such object: the teacher opponents' commands are built from
the raceline in `gym_env._opponent_actions:977-985` and touch nothing the candidate owns, so an
external baseline cannot move its opponent even in principle.

Scoring one is the ordinary command; nothing about the CLI changes:

```bash
python3 -m f1sim.learn.benchmark run --suite suite-v2.1.json --roster roster.json \
    --system end2race@none --out cells --device cpu --lease --estimator /abs/estimator.pt
```

`effective.device` is on every row, and two rows are only comparable when it agrees: CPU and CUDA
float arithmetic are not bit-identical and a rollout is chaotic enough for that to change an
outcome.

### Cross-runtime entries

A checkpoint records the arm it trained under, and `cross_runtime` covers exactly one case:

| checkpoint recorded as | declared arm | result |
| --- | --- | --- |
| `legacy` | `legacy` | fine |
| `legacy` | `estimated` / `fixed_low` | allowed **only** with `"cross_runtime": true` |
| `estimated` | `estimated` | fine |
| `estimated` | anything else | **refused — `cross_runtime` does not override this** |

So `cross_runtime` is not a general escape hatch. A non-legacy checkpoint may only run under its own
arm, because a policy trained against one controller is not comparable under another; only
legacy-trained weights may be lifted into a non-legacy runtime, and then only when declared, so it
cannot arise from a typo. That case is how the frozen original enters a leaderboard whose other
systems trained under `estimated`.

The same weights may appear twice under different arms; those are two systems, and the report keeps
them distinct by runtime.

Verify a roster without scoring anything:

```bash
python3 -m f1sim.learn.benchmark plan --suite suite-v1.json --roster roster.json \
    --map-lengths map-geometry-pins.json
```

This prints each system with its short shas and arm, the unique-weight count, total trials and a
projected GPU-hour figure. The projection is a measured per-cell rate scaled by raceline length with
an inferred race multiplier — rough in both directions, and no substitute for the geometry smoke.

**`plan` checks file pins only** — paths, symlinks, moving pointers, shas, and arm/estimator
consistency *within the entry*. It does **not** open the checkpoint to compare the declared arm
against the arm the checkpoint recorded. That check runs in `run`, and it runs even without
`--lease`, so this is the cheapest way to catch a mis-declared arm before booking a GPU:

```bash
python3 -m f1sim.learn.benchmark run --suite suite-v1.json --roster roster.json \
    --system <id> --estimator /abs/path/estimator_seed401.pt          # no --lease
```

It verifies the pin, checks the recorded arm, reports how many cells are ready, and then refuses to
score (exit `3`).

## Independence of the opponent

In the O family the opponent must not react to the candidate's *controller*. That is not obvious: the
teacher's plans and the candidate's are pushed through one tracker, and the teacher opponents
legitimately react to the candidate's arc position and speed, so simply running both and comparing
proves nothing.

```bash
python3 -m f1sim.learn.benchmark gate --estimator /abs/path/estimator_seed401.pt --steps 60
```

The gate restores identical state and probes it under each arm, then checks two things:

- **routed** — opponent commands must be *identical* across arms. Any difference means the opponent
  moved with the candidate.
- **unrouted control** — a deliberately perturbed run that must *differ*. Without it a gate that
  compares nothing would pass, which is a vacuous pass, and the CLI reports that case separately.

It also requires both **cold and warm** estimator history to be covered: `--steps` must exceed the
estimator's warm-frame count or warm coverage cannot be claimed. Exit codes: `1` opponent moved,
`2` control did not diverge (vacuous), `3` cold+warm not covered.

`--estimator` is required — the `estimated` arm has no default. There is **no device option**: the
gate builds its environments on CPU, so it can be run before any GPU is booked.

## Scoring

```bash
python3 -m f1sim.learn.benchmark run \
    --suite suite-v1.json \
    --roster roster.json \
    --system sgr_aux_anchor_s701 \
    --estimator /abs/path/estimator_seed401.pt \
    --device cuda --lease \
    --out results/
```

Without `--lease`, `run` verifies everything and then refuses (exit `3`), printing what it would have
scored.

**`--lease` is a caller acknowledgement, not a lock.** It takes no OS or GPU lock and reserves
nothing; it is you asserting that exclusive access was arranged. Nothing stops a second process from
starting on the same card, so the coordination is yours to do.

Before any cell runs, `run` requires all of: a frozen suite, a verified pin whose recorded arm
matches, proven placements, and a passing independence gate. Each is reported as itself rather than
surfacing as a confusing failure later.

Results are written per cell as they complete. Re-running resumes: existing cells are matched on full
identity — suite freeze, protocol, checkpoint sha, estimator sha and arm — not on cell id alone, and
a duplicate cell id or a corrupt line is refused rather than silently accepted.

**Concurrency.** Budget with *resident* VRAM, not with torch's own counters: a scoring process also
pays a fixed CUDA context and cuBLAS/cuDNN workspace cost that `max_memory_allocated()` cannot see,
and that cost is per process. Two processes is a reasonable default on a single 8 GiB card; verify on
yours before assuming it.

## Reading the report

`run` writes one file per system, `results/<system_id>.cells.jsonl`. **`report --results` takes a
single file, not a directory** — either one `.jsonl` of cell records, or a `.json` payload with a
`cells` list. So concatenate first:

```bash
cat results/*.cells.jsonl > benchmark-cells.jsonl

python3 -m f1sim.learn.benchmark report --suite suite-v1.json --roster roster.json \
    --results benchmark-cells.jsonl --out docs/benchmarks/leaderboard.md
```

Write the combined file to a **separate name**, not into `results/` — a `.cells.jsonl` inside that
directory would be picked up by the next glob and double-count every row.

A payload carrying only a precomputed `summary` is refused: everything rendered has to be derived
from raw per-cell records, or nothing ties the table to a measurement.

Validation happens **before** aggregation: rows are checked against the declared cells and roster.
Comparisons also require a single observation spec — if the rows span more than one,
`assert_single_obs_spec` **refuses** and names the split rather than reporting groups side by side.
Systems built on different input layouts therefore need their own roster, their own results file and
their own report; they are separate benchmarks, not two rows of one table.

### Metrics

| category | columns | units |
| --- | --- | --- |
| **Driving** | completion ↑, progress mean ↑, progress p10 ↑, progress p90 ↑, lap time ↓ | fraction, fraction of route, seconds |
| **Stability** | collisions/km ↓, distance km, spins ↓, large-slip s/km ↓, max yaw rate | per km, km, count, s/km, rad/s |
| **Surface** | completion ↑, progress mean ↑, completion vs midpoint ↑, progress vs midpoint ↑, large-slip s/km ↓, centreline offset RMS | fraction, fraction, pp, fraction, s/km, metres |
| **Avoidance** | cleared/pre-validated ↑, encountered, approach failures ↓ | fraction, count, count |
| **Overtaking** | passes held/race ↑, hold interruptions, contact ↓ | per race, count, count |
| **Traffic** | clean ↑, passes/race ↑, pace vs opponent ↑, attacking s/race ↑, following s/race, car contact ↓, wall collisions ↓, leads lost ↓, event-in-window trials | fraction, per race, ratio, seconds, seconds, count, count, count, count |

- **Progress** is a *signed* route fraction, so driving backwards does not accumulate credit.
- **Lap time** is over the system's **own completions only**, and is therefore not comparable across
  systems with different completion rates — a system that fails the hard cells can look fast.
- **Completion vs midpoint** compares against µ 0.94401, which is the *midpoint of the tested range*.
  It is not the vehicle's nominal friction — that is 1.0489 — so "mid" should not be read as the
  design point. The generated report repeats this next to the table.
- **"vs midpoint" columns are rendered as a difference of fractions, not percentage points.** A
  value of `-0.05` means five percentage points worse; multiply by 100 to state it in pp.
- **Max yaw rate** is labelled diagnostic: it is a single extreme value, not a rate.
- **Surface** reports per µ, which is the point of the category — a system may hold mid and high while
  losing low.
- **Avoidance** — `cleared/pre-validated` divides by **all pre-validated spawned trials**, which
  includes trials that failed on the approach and never reached the obstacle. It is not "cleared ÷
  obstacles encountered": a system that crashes before arriving is counted as not having cleared it,
  which is the intended reading. `encountered` is reported alongside as its own column, and approach
  collisions and timeouts are broken out as `approach failures`.
- **Overtaking** requires a pass to be *held*, not merely achieved: a pass counts once the candidate
  is clear ahead and stays clear for `hold_seconds`. Clearance comes from the declared vehicle
  footprint rather than a hand-tuned distance.
- **Traffic** is the v2.1-only family and is a different question from Overtaking, not a bigger
  version of it — see [Why T is not more O](#why-t-is-not-more-o). Its `clean` rate is the trial
  outcome and its `passes/race` is a count that may legitimately be 0; **read them together with
  `pace vs opponent`**, which is what separates a car that came through traffic by racing from one
  that came through it by hanging back. One T row is printed per scenario, plus a pooled
  `all scenarios` row, because the four scenarios are four different questions.

### N/A is not zero

An unmeasured value renders as `N/A (reason)` and never as `0`. Rendering an absent measurement as
zero would assert "no spins observed" where the truth is "spins were not measured", which is how a
system with missing instrumentation comes to look like the most stable one. A category with no
measurement at all prints `N/A (category not evaluated)`.

## Provenance in every result

Each cell records a **start fingerprint** taken immediately after the final seeded reset, before the
controller initialises and before any action: three separate digests — physical state, actor input,
and calibration — plus tensor counts, `sim_t`, IMU phase, observation-spec hash, total cars and race
size.

Three digests rather than one combined hash, so a disagreement names the layer it happened in: a
physical mismatch is a different starting state, an actor-input mismatch is a different observation,
and a calibration mismatch is a different vehicle. Digests are taken over the original tensor bytes
including each field's name, dtype and shape, so a reshape or a dtype change cannot pass as identical
and a 1e-10 difference is not rounded away.

Comparisons refuse unless the starts match. Every row is validated before grouping or comparison,
including a lone row — a single row is where "paired" is least earned. The recorded cars/race-size
must also match the declared cell, so contradictory batch metadata cannot reach a paired comparison.

Each result additionally carries a **source digest**: the sha256 of every runtime module a
measurement depends on plus the benchmark's own code. A result measured against changed source is
identifiable as such.

## Adding a checkpoint later

The suite is frozen; the roster is not. To add a newly completed checkpoint:

1. Copy the checkpoint to an immutable path — not `ppo_latest.pt`, which training rewrites.
2. Append an entry with its sha, arm, and estimator pin if the arm is `estimated`.
3. `plan --roster` to verify the pin resolves and the arm matches the checkpoint's record.
4. `run` that one `--system`. **Do not re-run existing systems**: they were scored against this same
   frozen suite, and their results stay valid.
5. `report` over the whole results directory.

Adding a system does not invalidate earlier rows, which is the reason the suite freeze and the
per-row source digest exist. What *would* invalidate them is editing the suite — `load()` refuses
that outright — or changing a runtime module, which the source digest makes visible.

If a new checkpoint expects a different observation layout, it is still admissible: the report groups
by observation spec and reports the groups separately rather than pooling them.

## Reproducing an existing table

**Start from the packaged suite, not from `geometry`.** Re-running `geometry` regenerates placements
from the current defaults (`--s-obs 20.0`), which produces a *different* suite with a different
freeze hash — scenarios that were never the ones scored. All three suites ship inside the package (`suite-v1.example.json`, `suite-v2.example.json`,
`suite-v2.1.example.json`; v2 was frozen with `--s-obs 10`, and v2.1 inherits those placements
rather than re-deriving them):

```bash
python3 -c "
from importlib import resources
src = resources.files('f1sim.learn.benchmark').joinpath('suite-v1.example.json')
open('suite-v1.json','w').write(src.read_text())
"
python3 -m f1sim.learn.benchmark plan --suite suite-v1.json     # confirm the freeze hash
```

Then:

```bash
# 1. verify the roster resolves and every pin still matches
python3 -m f1sim.learn.benchmark plan --suite suite-v1.json --roster roster.json

# 2. prove the opponent is independent of the candidate controller (CPU)
python3 -m f1sim.learn.benchmark gate --estimator /abs/path/estimator_seed401.pt --steps 60

# 3. score each system (GPU, one at a time or two concurrent)
python3 -m f1sim.learn.benchmark run --suite suite-v1.json --roster roster.json \
    --system <id> --estimator /abs/path/estimator_seed401.pt --device cuda --lease --out results/

# 4. combine the per-system files, then render
cat results/*.cells.jsonl > benchmark-cells.jsonl
python3 -m f1sim.learn.benchmark report --suite suite-v1.json --roster roster.json \
    --results benchmark-cells.jsonl --out docs/benchmarks/leaderboard.md
```

A reproduction is only meaningful if the **suite freeze hash matches** the published one and step 1
passes unchanged. If the hash differs, the scenarios differ and the numbers are not comparable —
which is precisely why the suite is copied rather than regenerated.

## Checking it without a GPU

```bash
python3 -m f1sim.learn.benchmark feasibility --suite suite-v1.json --envs 4 --device cpu
python3 -m f1sim.learn.benchmark feasibility --suite suite-v2.json --envs 4 --device cpu
python3 -m f1sim.learn.benchmark feasibility --suite suite-v2.1.json --envs 4 --device cpu
```

This drives the scenarios with a scripted expert on synthetic metadata to confirm each one is
achievable at all. It loads **no candidate weights and is not a benchmark score** — it answers "is
this scenario possible?", not "how good is this policy?". A scenario the expert cannot complete
should be fixed or dropped before the freeze, never after seeing candidate results.
