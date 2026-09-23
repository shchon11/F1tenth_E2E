# The console's obstacle option, and three tests that had stopped meaning anything — 2026-09-23

Branch `main`, base `f48ec29`. CPU only for the measurements below: the 4070 was held by a training
run for the whole session, and the 5060 is the user's own console. Nothing here needed a GPU, which
is itself part of the finding — the console's logic can be exercised through `SimWorker` on the CPU.

Code touched: `f1sim/viewer/sim_worker.py`. Tests: `tests/test_procedural_obstacles.py`,
`tests/test_console_train_obstacles.py`, `tests/test_graph_fastpath.py`.

Companion to [`procedural-obstacles-2026-09-13.md`](procedural-obstacles-2026-09-13.md), which is
where the generator and the digest test below came from.

## Summary

* **장애물 "학습과 같음" could not be driven with more than one car.** Raising 차 대수 above 1 did
  not run badly — it refused to start, with the environment's English sentence. The console sends
  raceline teachers for the other cars, the training settings put crates on the racing line, and
  `F1VecEnv` refuses that combination on purpose. Training itself never hit it because the runs
  since s911 drove their opponents with prop-aware planners. Fixed by giving the option the
  opponents it is named after.
* **The "byte-identical off" digest had failed since before it was last looked at, for two
  independent reasons**, and neither was a bug in the feature. The pinned number never reproduced
  on this machine at all, and the off path's physics has legitimately moved since the merge base.
  The claim survives in a form that is immune to both: the *generator state*, which is identical to
  the merge base's across 295 commits.
* **`test_graph_fastpath.py`'s order dependence is one line.** A test calls
  `torch.cuda.set_device` and never puts it back, so every later test that says `device="cuda"`
  silently moves to the other card. On this machine that is the 8 GB laptop GPU the user's console
  is on, rather than the 12 GB one the test would have picked alone.
* **Nine more console tests had gone stale**, under two redesigns the tests were not moved with.
  Eight are fixed. The one thing that turned out to be a real defect is small and was found by
  refusing to edit an expectation until the behaviour behind it had been measured: the training
  page restored a mode's values without the name of the recipe they came from.

## 1. 학습과 같음 with more than one car did not start

Measured through the console's own path — `SimWorker.build_session` with
`SessionConfig(procedural=True, cars_per_race=3)`, which is what the window builds when the user
raises 차 대수 — on four maps. All four failed identically, before any stepping:

```
StartConfigError: 학습과 같은 장애물: procedural_raceline_corridor='off' puts obstacles on the
racing line, and opponent='teacher' drives the other cars with the raceline teacher, which cannot
see a prop (they are not in the occupancy grid) and has no lateral freedom to use if it could.
```

The chain is exact and each link is deliberate:

| link | where | why it is there |
|---|---|---|
| the console's other cars are raceline teachers | `window.py: opponent="teacher"` / the 상대차 table's default | what the viewer has always run |
| 학습과 같음 copies the run's obstacle settings | `sim_worker.procedural_settings` | s915's `args.json`: `procedural_raceline_corridor='off'` |
| crates on the line + a prop-blind teacher is refused | `gym_env._check_procedural_opponents` | a race that ends on the layout measures the layout, not the policy |

The missing link is the fourth: **the run's own opponents were never copied.** s915 trained with
`opponent='slots'` and one slot of
`kind_mix=('forzaeth', 'forzaeth_pred', 'lane_switch', 'interactive'), speed_scale=(0.7, 1.0)` —
every one of them prop-aware. So the option called "학습과 같음" was reproducing the run's
obstacles and not the run's traffic, and the two are not independent: the obstacle setting is what
makes the traffic setting mandatory.

`procedural_settings` reads the run's `args.json`, but that file cannot supply the table. `ppo.main`
writes `vars(a)` with `default=str`, so `opp_slots` is stored as the *repr* of the dataclasses
(`"OpponentSlot(kind='forzaeth', kind_mix=(...), ...)"`), not as JSON. The JSON form
(`opp_cfg.slots_config(a)`) goes only to the W&B config, which an offline run does not write. Left
as is — see §5.

**Fix** (`sim_worker.procedural_opponents`): on this obstacle choice with a grid of more than one,
any opponent row that could be *drawn as* a teacher that cannot see a prop becomes the training mix;
rows that can see one are left exactly as the user set them. The criterion is read from
`opponent_slots.KIND_BY_NAME` rather than copied, so it cannot drift from the env's own, and
`kind_mix` is read as the set of kinds a row can become — a mix listing `raceline` counts as blind,
which is the same reading the env takes and the reason a session would otherwise look fine until
the draw came up raceline. The console says in its log which cars it changed and to what.

## 2. What else the console option does, checked

Through `SimWorker` on the CPU, `s915`, 학습과 같음, `real/iccas25`:

| | 1 car | 3 cars |
|---|---|---|
| session builds | 1.4 s | 4.5 s (before the fix: refused) |
| pieces standing | 16 | 12, 12, 12 — the same layout, `procedural_shared` |
| catalogue to the console | 25 shapes, 18 slots | same |
| frame's `props_dyn` rows = live pieces | yes | yes |
| a full 리셋 draws a new layout | yes | yes |
| one car resetting alone keeps it | n/a (B = 1 *is* the batch) | yes |

The single-car reset is only observable with more than one car on the grid, which is one more thing
that had never been exercised: with `cars_per_race=1` every reset is a full-batch reset, so the
"a car that rejoins keeps its race's layout" rule was untested until the session above could start.

One thing the strip did not say: `procedural_obstacles` is a *share of resets*, and below 1 some
resets put nothing on the track. Every run since s911 trained at 1.0, so this is invisible today,
but a run at 0.5 would give the console an empty lap half the time with no explanation. The strip
now says so when the share is below 1.

## 3. The digest that could not have passed

`test_off_is_byte_identical_to_the_merge_base` hashed a fixed 40-step rollout's observations,
rewards, flags and generator state with the feature off, and compared it against a number recorded
from the merge base (`4209ec2`) on 2026-09-13. Two separate things are wrong with it, both measured
today by checking old commits out into a worktree and running the same rollout:

1. **It does not reproduce on this machine even at the commit that recorded it.** At `7877b18` —
   the commit that introduced both the test and the constant — the rollout produces
   `b67b98b5...`, not the `7a0f4024...` written down beside it. The same value comes out of the
   merge base itself. Run twice, it is stable, so this is not nondeterminism within a run: the
   pinned number carries the float/library environment it was recorded in as much as it carries the
   code.
2. **The off path's physics really has moved.** Bisecting `4209ec2..HEAD` on the observation digest
   (295 commits) gives a first change at **`9035c44`** (2026-09-21, "A parked car's LiDAR no longer
   looks at the floor"): the road-tilt OU state and the parked-car LiDAR, touching `params.py`,
   `raceline.py` and `sim.py`. Deliberate sensor work, nothing to do with obstacles. It is not the
   only one — `9035c44` gives `dc582ba...` and HEAD gives `0a10828...` — so at least two
   independent changes have moved it since.

An observation pin therefore cannot state this feature's claim: it fails for every unrelated change
to the simulator, and it has to be re-recorded per machine. What it was *for* is "off consumes
nothing, draws nothing, and leaves the run exactly as it was", and that is the **generator state**,
which is a function of the sequence of draws alone.

Measured:

| build | generator state after the same rollout, feature off |
|---|---|
| merge base `4209ec2` | `2b7fdc0a639590451d2254e88b57d21a2d2f3cb67d13518d3351785b2db60563` |
| HEAD (295 commits later) | **the same** |
| HEAD, feature on | `e6d100a6...` (different, so the guard has teeth) |

So the original claim holds and is now pinned in the form that states it
(`test_off_draws_exactly_what_the_merge_base_drew`). The observable side is kept as a live
comparison in the same build (off vs on) rather than against a number, which is what it could
always have been.

## 4. The other two

**`test_no_piece_stands_on_the_raceline`** asked for `scene:scene_0912_2344`, one of the user's own
recordings. Scenes live outside the repository (`~/f1sim_scenes`, or `$F1SIM_SCENES`) and this
machine has only `real_iccas25` and `real_iccas25_2`, so the test raised `FileNotFoundError` from
`scene.load` on any checkout but the one it was written on. It now drops `scene:` tracks whose
directory is not there and runs on what remains — `gen:control:1400` builds everywhere — and skips
only if nothing is left. Keeping the catalogue track under test is the point: a plain skip would
have removed the check entirely on every other machine.

**`test_evaluation_defaults_are_off`** asserted `"procedural" not in inspect.getsource(evaluate)`.
`evaluate` grew `--procedural-*` on purpose — without it `spec_korea_contact_s911`, trained on
crates, could only be scored on an empty track — so the assertion broke the day the flags landed
and stayed broken. What it *meant* is that the frozen suites do not move unless someone asks, which
is now checked by running `evaluate.main()` with the evaluation itself replaced by a spy: with
default arguments nothing at all reaches `EnvConfig` (`opp_extra is None`), and with the flags given
exactly those keys do. The guard is on the behaviour, not on the presence of a word in a file.

## 5. Nine console tests that had gone stale under the console's own redesigns

Not part of the brief, found by running the console suite: 9 failures, all from before today and
all of the same kind as §3 and §4 — a test pinned to what the code used to do. Two redesigns, five
days apart, and in both cases the tests were left where they were.

**Fixed (`test_console_map_card.py`).** The 장애물 control became a placement control over modelled
props on 2026-09-18 (`tracks.ASSET_OBSTACLES`, `asset_scenario`), and the tests were last touched
on 09-15. Three consequences, each pinned in a test:

* a family now carries its catalogue, so the spec gained `!assets=mixed:1`;
* the list is `(없음, 기본, 학습과 같음, 랜덤·낮음, 랜덤·중간, 랜덤·높음)`, and `props`/`pinch` are
  gone as separate entries;
* `line`'s label is 랜덤 · 중간, not 주행선 위.

One of the five asserted a capability — "racetracks take +props / +hard and no other family" —
so it was checked against the loader instead of edited to match: `maps.load(tracks.resolve(...))`
builds `rt:Monza+rlobs44!assets=mixed:1`, `+hard3` and `+obs7`. A racetrack could never carry the
grid-rasterised `+obs`/`+pinch`, and there is no such thing on this control any more, so offering
all three is right and the pinned list was the stale part.

**The other four, which looked like a regression and were not.** All four are on the training-launch
page, and all four have one cause. `test_the_training_page_leaves_the_recipes_alone_with_the_switch
_off` expects the default recipe's `--opponent mixed` and gets `policy`, while `RECIPES[0]` in
`viewer/console/training.py` still says `opponent="mixed"` — which reads exactly like the form no
longer emitting the recipe it names, and that is how it was first reported.

It is not. Building a `RecipeForm` and printing it:

| state | mode | recipe | `--opponent` | `--obstacle-draws` | `--controller` | `--race-size` |
|---|---|---|---:|---:|---|---:|
| as constructed | **dagger** | custom | policy | absent | absent | 1 |
| PPO + 기본 레이스 레시피 | ppo | origrecipe | **mixed** | **8** | **legacy** | **2** |

`RECIPES[0]` reaches the command in full, `--estimator` included by its absence. What changed is
where the page *opens*: `d372589` (2026-09-20, "the training page hid its own pipeline") added the
training stages and made the form open on step ① — DAgger — and the recipe combo is filled per
mode, so a freshly built form holds one entry (사용자 정의). The four tests build a bare form and
ask it about the PPO recipes, which is now a different page of the same form. They were last
touched the same day, by a different commit, and were not moved with it.

So they are fixed the way the first five were: the claim is kept and asked where it holds (PPO mode,
기본 레이스 레시피 selected). Nothing about the recipes changed.

**One real thing came out of it.** Leaving a mode and coming back restores its values from
`_cache`, but the recipe's *name* was not restored with them: `_mode_changed` only reset the combo
for a mode it had never built, and the form builds the PPO page once at construction before opening
on step ①, so PPO was always "already built". The page then read 사용자 정의 over another recipe's
numbers — the same class of problem as a 장애물 label that does not describe what is placed. The
recipe key is now cached beside the values and restored with them, and a loaded config file gets
사용자 정의, which is the honest name for values that came from a file.

## 6. Not done

* **`args.json` cannot rebuild a run's opponent table.** `ppo.main` writes it through `str()`, so
  `opp_slots` is a repr. Writing `opp_cfg.slots_config(a)` into `args.json` the way it already goes
  into the W&B config would let 학습과 같음 copy each run's own traffic instead of the s911 mix.
  Not done here: a training run held the GPU and `ppo.py` is its code.
* **The console's step time with the training obstacles, on any GPU.** 45.5 ms a step for three
  cars was measured on the 4070 under load (2026-09-22) and 25 ms is realtime, but that was with
  the old configuration — the session that measured it could only have been the solo one, since a
  grid of three refused to start until today. The three cars now on the grid are ForzaETH planners
  and an interactive teacher, which are not free, so the number has to be taken again. Nothing
  here measures it: the 4070 was training all session and the 5060 is the user's own console.
  The CPU figures above (69 ms a step solo, 291 ms with three cars, eager) say nothing about it.
* **`test_graph_fastpath.py`'s CUDA tests** are fixed but not re-run, for the same reason — its 8
  CUDA tests skip without a GPU and its 26 CPU ones pass. The fix is the device restore and the
  sessions the file was leaving built; the console path itself was already clean, because
  `SimWorker._release_session` collects and empties the cache.
