# Per-opponent slots: one configuration per car, and what stayed the same (2026-09-15)

Branch `feat/opponent-slots`, base `df60b44`. Worker 21.

## What was asked, and why the old vocabulary could not say it

> 대상차 설정을 좀 다양하게 할 수 있었으면 좋겠네. 대상차를 티쳐/정책 선택여부 뿐만 아니라 각 대상차에서
> 적용할 속도 프로파일링 및 체크포인트 등의 설정이 가능했으면 해.

Everything about the other cars was a property of the **race**. `--opponent teacher` made every
opponent a teacher; `--opp-speed 0.6 1.0` gave all of them the same band; `--opp-events brake`
scripted all of them; `--spawn-order behind` placed the whole grid; `--opp-pool` drew *one* entry per
race and gave it to every opponent slot of that race. With two cars that is the same thing as
configuring the opponent. With three it stops being: **"a slow car ahead and a defending car
alongside" is not a sentence this vocabulary contains**, and it is what a race is.

The console had the same shape: one combo, `상대차 주행 방식`, with two entries.

## What it is now

`f1sim/opponent_slots.py` holds an `OpponentSlot` — one specification per car of a race — and the
**driver-kind registry** the rest of the system asks. `EnvConfig.opponent_slots` is a table of
`race_size - 1` of them, and slot *i* of every race in the batch is built from spec *i*. What a spec
fixes is deterministic per slot; what it leaves as a range is still drawn per reset from the
simulator's own generator, so a seed still reproduces the race.

| what a slot carries | where it lands |
| --- | --- |
| `kind` | the driver code (`OPP_DRIVER_*`), fixed for the life of the env rather than re-drawn per race |
| `checkpoint` + `controller` | an `OpponentPool` entry, loaded with the arm the slot declares and refused when the file records another |
| `speed_scale` | `RacelineTeacher.speed_scale` as a **(B,) tensor** for a teacher car; the speed cap for a `policy` / `self` car |
| `label_grip` | `RacelineTeacher.label_grip_codes`, a **(B,) tensor** of mode codes: `grip_bin` computes all three answers and selects |
| `speed_cap` | that car's `speed_cap` / `cap_scale` directly |
| `events`, `event_rate` | per-car rate and per-car lookup rows in `OpponentEvents` |
| `reactive` | a (B, 4) probability tensor in `OpponentEvents.reset` |
| `spawn` | a per-car grid rank, replacing the per-race `spawn_order` |
| `seed` | that slot's own draw stream for its per-reset ranges |

### The two design decisions worth recording

**A registry, not an if-chain.** `KINDS` is a table; adding a kind is one entry in it. Worker 17's
`InteractiveTeacher` is not on main, so `interactive` sits in the registry with `available = False`
and a sentence saying why. The console lists it greyed (`interactive 티처 [병합 후 활성]`) with that
sentence in its tooltip; the flag refuses it with the same one.

The entry also carries `teacher_factory = "f1sim.interactive_teacher:InteractiveTeacher"`, and that
field is what makes this a registry rather than a list of names. Without it, the day the branch
merges the kind would become *available* and be driven by the raceline teacher — the silent
substitution the `available` machinery exists to prevent, arriving by the back door. With it, the env
builds one object per non-raceline teacher kind a table named (`Class(raceline_teacher, env=env)`,
which is `InteractiveTeacher`'s own constructor), asks it for the whole batch and selects the rows it
owns, exactly as the checkpoint pool is asked. The per-car tensors stay on the raceline teacher,
which every such teacher keeps as its reference, so a slot's speed band and grip label reach it
without the env knowing what it is. `tests/test_opponent_slots.py` exercises that path with a stub
kind, because the real one is not on this branch.

**`spawn` is the mirror of `--spawn-order`, and says so.** `--spawn-order` names where the *learner*
starts relative to everyone; a per-car table reads "this car starts ahead of me", so that is what the
field means. `ahead` is the default and reproduces `--spawn-order behind` — the learner at the back
with a pass to make. The rank rule generalises the old one exactly: cars that start ahead take ranks
0…k-1 in **descending** slot order, the learner and everything alongside it share rank k, and cars
that start behind take k+1… in ascending slot order. An all-`ahead` table gives `M - 1 - slot`, which
is what `spawn_order behind` computed; an all-`behind` table gives `slot`, which is `ahead`; an
all-`alongside` table gives 0 everywhere, which is `alongside`. Mixed grids are the new part, and the
abreast group (the learner plus its `alongside` slots) is tested against the lane and the props per
**race**, falling back to a stagger where it does not fit — the same rule and the same reason as
before: two cars put side by side where the lane has room for one terminate on step 1 for ever.

### What a slot's `controller` does, precisely

It is the plan-controller arm the *checkpoint* records, and it is **checked, not installed**. The env
has one `PlanTracker` shared by every car, so a slot cannot run its own arm; what it can honestly do
is refuse a checkpoint whose plans were fitted to a friction-limited tracker when this session is not
running one. `OpponentPool.load(arms=...)` compares the slot's answer with `controller_arm_of(file)`
and raises when they differ, and passes `allow_controller=True` only when the slot names the arm the
file records. Per-car tracker arms would be a separate piece of work (`learn/grip_runtime.py`
installs one solver hook for the whole batch), and are not claimed here.

## Off is off: the byte-identity result

The claim that matters for every existing checkpoint and benchmark number is that the feature
switched off leaves the env the env it was. Measured rather than asserted: the same rollout script
was run against this tree and against a clean extract of the base commit `df60b44`, hashing, per step
for 80 steps, `obs["scan"]`, `obs["speed"]`, the reward, `terminated`, `truncated`, `info["priv"]`,
`sim.state`, `last_cmd`, `speed_cap` and `opp_scale`, plus the simulator generator's state at the
end.

| configuration | `df60b44` | `feat/opponent-slots` |
| --- | --- | --- |
| `--opponent teacher` | `b418ec9e7dd763be` | `b418ec9e7dd763be` |
| `--opponent policy` | `e3a874b1005c7965` | `e3a874b1005c7965` |
| `--opponent mixed` | `e7d02cfc98463e12` | `e7d02cfc98463e12` |
| teacher + all four timed events @ 1.0/10 s | `c6ce250e7d7a9fd8` | `c6ce250e7d7a9fd8` |
| teacher + all four reactive behaviours | `caf5caac4096b28e` | `caf5caac4096b28e` |
| `--spawn-order alongside` | `0a1b676436343ab0` | `0a1b676436343ab0` |
| `--spawn-order random` | `7abf37858c5f3343` | `7abf37858c5f3343` |
| `--race-size 3 --spawn-order ahead` | `165f84c1f6ee124e` | `165f84c1f6ee124e` |
| solo (`--race-size 1`) | `8d7c2938982eba18` | `8d7c2938982eba18` |

Identical in all nine, generator state included — so nothing new consumes randomness on the old path
either, which is the failure that would otherwise desync every *future* draw of a reproduced run.

The hashes come from `tests/` code paths only in the sense that they use the same env builder; the
script itself is a one-off (`work/` scratch), because a test cannot check out another commit. What
the suite pins instead is that the old path still runs the old instructions
(`tests/test_opponent_diversity.py::test_everything_new_off_leaves_the_rollout_untouched`, unchanged
and passing) and that the scalar teacher path is still scalar
(`tests/test_opponent_slots.py::test_the_default_teacher_path_is_untouched_by_the_per_car_tensors`).

## What the console does with it

One widget, `viewer/console/opponent_table.py`, on both pages. The 주행 page's 고급 설정 shows it in
place of the old combo and puts the table into `SessionConfig.opponent_slots`; the 학습 page shows the
same widget behind a `차량별 상대차 설정` switch and emits one `--opp-slots` JSON, dropping the flags
that table replaces. One widget and not two because the two pages must not be able to disagree about
what a slot is: a table that emitted something `learn/opponent_config.py` could not parse would be a
second configuration language with no way to diff it against the first, and
`tests/test_console_opponent_table.py::test_the_training_page_emits_a_flag_the_trainer_parses` runs
the widget's own output through the trainer's own parser.

The sidebar grows from 330 px to 640 px while the table has rows, and back when it does not: ten
fixed-width columns at 330 px is three columns and a scrollbar, which is not a table anyone can read.
It only grows while the window can still give the 3D view 600 px, and a splitter the user has dragged
is left alone.

## Numbers this note does *not* contain

No training run, no benchmark. What a mixed grid does to a policy is a GPU question and the card is
queued (`../_rules.md`); this is the mechanism and the proof that switching it off changes nothing.
The census (`python -m f1sim.learn.opponent_census --opp-slots …`) now reports a row per slot —
learner-seconds in contention with that car, seconds in its attack window, events it fired, seconds
each disposition acted, its own wall and contact terminations — which is the instrument the next
person needs to answer that question without taking a number on trust.

## Tests

* `tests/test_opponent_slots.py` — 53 tests: off-path identity and generator quiet, kind → driver
  code, per-slot speed scale on the teacher tensors, per-slot grip label moving the profile bin,
  per-slot cap binding one car, `spawn` placing each car (including a mixed ahead/alongside grid and
  `random` being remembered across a partial reset), per-slot events and dispositions, checkpoints
  (shared file loaded once, arm mismatch refused, oracle refused), every refusal the flag and the env
  owe, JSON round trip, `SessionConfig` round trip, the 4-car CPU smoke, and the census rows.
* `tests/test_console_opponent_table.py` — 20 tests under xvfb: widget → spec → widget round trip,
  every preset, the unavailable kind listed and disabled, the cells a kind cannot carry switched off,
  a checkpoint's arm and memory kind, the oracle refusal, both pages' output, and the jobs card.
* Unchanged and re-run: `tests/test_opponent_diversity.py`, `tests/test_opponent_events.py`,
  `tests/test_ppo_memory.py`, `tests/test_console_session.py`, `tests/test_console_training.py`,
  `tests/test_console_layout.py`.
