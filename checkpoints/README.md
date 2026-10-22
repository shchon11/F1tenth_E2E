# Archived checkpoints

The two policies the 2026-09-19/20 work left behind, kept in the repository so a result can be
re-measured without the run directory it came from. How they were made, and every number not
repeated here, is in [the research note](../docs/research/mintime-teacher-speed-head-2026-09-19.md).

| file | what it is | sha256 (first 16) |
| --- | --- | --- |
| `dial_student_s901.pt` | the generalist: DAgger over the 149-scenario `train` split against the min-time 7 / 6.5 / 4 teacher, with the grip dial as an input | `a27fd18f685d1712` |
| `map12x16_specialist_s903.pt` | the same student specialised on one map for 6.3 M steps — the per-venue recipe of record | `df51e197ed4d466a` |

Both are 9.1 MB. The specialist's 17.4 MB of Adam moments were dropped: they exist only to *resume*
training, and the run that would be resumed is still in `~/f1sim_runs/spec_map12_fastsafe_budget/`.
Everything else — actor, critic, `meta`, the experiment block — is what the run wrote, and both
reproduce their published numbers exactly (re-measured after stripping).

## What they score

Held-out, 896 first attempts (the held-out split minus `rt:Monza` and `real:map16x07`, which the
user ruled out, and minus `real:blackbox2022_3`, ruled an unfair map on 2026-09-13):

| | dial | completed | collisions / km |
| --- | --- | ---: | ---: |
| `dial_student_s901` | exact | 93.4 % | 1.48 |
| `dial_student_s901` | −0.15 | **97.3 %** | **0.55** |

On `real:map12x16` (256 cars × 60 s, randomisation on, true line-to-line laps):

| | median lap | a crash every |
| --- | ---: | ---: |
| privileged teacher | 8.57 s | 89 laps |
| `dial_student_s901` | 8.48 s | 26 laps |
| `map12x16_specialist_s903`, dial −0.15 | **7.62 s** | **105 laps** |

## How to run them

Both are **conditional** checkpoints: they take a grip dial — one number saying how much friction to
use — so the deployment loaders refuse them unless the caller supplies it. `evaluate` does:

```bash
python3 -m f1sim.learn.evaluate checkpoints/dial_student_s901.pt --per-track --protocol trials \
    --tracks 'real:map12x16' --envs 128 --budget-laps 3 --dial-offset -0.15
```

`--dial-offset` is relative to the floor's true friction, which only a simulator knows. On a car the
dial is a number an operator sets from the surface, and **0.15 under what you measure is the setting
these scores were taken at**: it costs 1.6 % of lap time and halves the collisions. Setting it
*above* the real friction is out of distribution and measured to be dangerous.

The specialist is specialised: it is quick on `real:map12x16` and is not the policy to take to
another venue. Make a new one per venue from `dial_student_s901`, which is the point of it
(`~/f1sim_runs/_eval/mintime-teacher-2026-09-19/scripts/specialize.sh`, ~45 min on one GPU):

```
--lr 2e-4 --lr-end 5e-5 --kl-coef 0.05 --kl-decay 4e6 --gamma 0.997 --collision-penalty 60
--steer-penalty 0.02 --lap-time-bonus 6 --dial-margin 0.30 --dial-exact 0.30 --grip-budget-penalty 2.0
```

## What is not here

The negative results, which are in the note and in `~/f1sim_runs/`: the full-scale generalist PPO
runs (`ppo_dial_s901`, with and without the grip budget) were 1–2 % quicker than the student they
started from and no safer, so the student is the generalist of record; and `spec_map12_revised` /
`spec_map12_fastsafe_nobudget` are the same specialisation with the grip budget off — quicker again
(6.97 / 7.15 s) at four times the collision rate.

## ICCAS specialists (added 2026-09-24)

The two best policies of the `spec_korea_contact` line on `real:korea_2025_iccas`, trained against
procedural obstacles (up to 18 props) and slot opponents (`forzaeth`, `forzaeth_pred`, `lane_switch`,
`interactive`, speed 0.7–1.0 of their own profile). Same run, two updates; Adam moments dropped as
above, and both re-measured after stripping to the same lap, collision rate and mean speed as the
run's own file.

| file | what it is | sha256 (first 16) |
| --- | --- | --- |
| `iccas_specialist_s915_u768.pt` | `spec_korea_contact_s915`, update 768 — the reference policy of the 09-23/24 work | `1b129e4020245bf8` |
| `iccas_specialist_s915_u1152.pt` | the same run, update 1152 — statistically tied with u768 | `906dd38ce5fd98f8` |

Static contacts (walls and props) per km, pooled over five obstacle scenarios and two evaluation
seeds (77, 78), from `~/f1sim_runs/_eval/mintime-teacher-2026-09-19/eval_center{,78}/`:

| | contacts / km | per seed | lap, props | lap, empty |
| --- | ---: | ---: | ---: | ---: |
| `iccas_specialist_s915_u768` | 5.32 (275 / 52 km) | 5.86 / 4.80 | 8.42 s | 7.98 s |
| `iccas_specialist_s915_u1152` | 5.18 (266 / 51 km) | 5.02 / 5.33 | 8.39 s | 7.88 s |
| every other checkpoint measured the same way (s912–s915, 7 of them) | 6.20–8.25 | | | |

Read these with two caveats. They were taken with the grip dial **0.30 above** the true friction,
which is outside the range the policy was trained on (`dial = mu - U(0, 0.30)`); at the reference
setting, dial 0.0, u768 laps the empty track in 7.86 / 7.96 s (seeds 77 / 78) and, under the
2026-09-24 spawn rules, touches props 5.03 times per km. And the car-contact column those tables
also carry is left out on purpose: before 2026-09-24 the traffic meter counted every *step* of a
soft contact rather than its onset, so it measured contact time, not contacts.

Both are dial-conditioned like the others:

```bash
python3 -m f1sim.learn.evaluate checkpoints/iccas_specialist_s915_u768.pt --action-mode plan \
    --tracks real:korea_2025_iccas --race-size 2 --opp-slots '[{"kind":"forzaeth"}]' \
    --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time \
    --collision-mode soft --dial-offset 0.0
```

## `iccas_league_c_s949.pt` (2026-09-29): the ICCAS specialist on the refitted simulator, trained with a league

Trained on the simulator refitted to the car's recordings on 2026-09-28 (tyres, servo, drive limit,
contacts, LiDAR porosity, tracker understeer k_us 0.02; `docs/research/sim-realism-2026-09-28.md`), so
every earlier checkpoint here is an old-simulator policy and this is the first one for the current
car. Recipe: s915u768 + GRU + opponent auxiliaries, rear-end and merge penalties, and a league -- in
40 % of races the opponent is driven by one of three earlier policies instead of a planner
(`OpponentSlot.ckpt_mix`). Linear plan output, tracker speed command (what the car's controller node runs).

ICCAS, 4 seeds x 4 planner opponents (`docs/research/retrain-plan-2026-09-28.md`):

| | car contacts | per km | passes | solo lap | props / km |
| --- | ---: | ---: | ---: | ---: | ---: |
| same recipe without the league (2 training seeds) | 172 / 185 | 2.55 | 256 / 277 | 8.13 / 7.73 s | 3.04 / 2.11 |
| **this file** | **126** | **1.82** | 246 | 8.18 s | **1.39** |
| its league replicate (seed 951) | 170 | 2.40 | 245 | 7.90 s | 1.82 |

Against learned opponents it never met (B s948, gate s944): 3.07 and 1.84 contacts / km, against
8.40 and 7.33 for the recipe without the league. Re-measured after stripping: identical to the source run.

## `iccas_constrained_v2R.pt` (2026-10-02) — best ICCAS race policy so far

Linear plan output. Lineage: DAgger with the line teacher on every training track → ICCAS solo PPO
(feedforward) → race PPO with a GRU, KL leash and a league (`ffR_lin`) → **12.6 M steps of race PPO
under the constrained reward** (`--lagrange wall=3.0,car=1.5`: reward is progress + lap + lap time,
wall/prop and at-fault car contacts are Lagrangian constraints; GAE λ 0.97; league of 5 learned
policies + 4 planners at p 0.5). Optimizer moments dropped; re-measured identical after stripping.

| ICCAS, 4 seeds × 4 planners, dial 0 | v2R | ffR_lin (its start) | C (`iccas_league_c_s949`) |
| --- | ---: | ---: | ---: |
| solo lap | **7.66 s** | 7.86 s | 8.18 s |
| car contacts | 160 | 227 | **126** |
| passes | 235 | 270 | 246 |
| prop contacts / km | 1.25 | **0.89** | 1.39 |
| unseen learned opponents B / gate, contacts (2 seeds) | **18 / 17** | 53 / 39 | — |

## `iccas_constrained_safe_s976.pt` (2026-10-04) — the safest ICCAS race policy so far

The v2R recipe (above) on training seed 976, picked from five seeds (973 = v2R, 976-979). Optimizer
moments dropped; re-measured identical after stripping.

| ICCAS, 4 seeds × 4 planners, dial 0 | s976 | v2R (fastest) | C |
| --- | ---: | ---: | ---: |
| solo lap | 7.91 s | **7.66 s** | 8.18 s |
| car contacts | **119** | 160 | 126 |
| passes | 236 | 235 | **246** |
| prop contacts / km | **1.10** | 1.25 | 1.39 |
| unseen learned opponents B / gate, contacts / passes (2 seeds) | 19/37, 23/29 | 18/14, 17/12 | — |

The five seeds of the same recipe ranged 119-204 car contacts and 7.66-8.25 s solo laps: one seed
is not a measurement of a recipe.

## `iccas_bootcamp_safe.pt` (2026-10-05) — fewest car and prop contacts so far

Lineage: the champion's feedforward ICCAS solo stage (`ffS_lin`) → **12.6 M steps of "boot camp" race PPO**
(`ppo_boot.sh`, commits 91f5510 / e303c49): ICCAS plus 16 rebuilt variants of it (`+warp`: size ±5 %, lane
±0.10 m, wall sections set in, corners cut back), a race of 4 whose 3 opponent seats are each filled with
p 0.5 per race (0-3 opponents), the four planners with scripted defending / line holding and brake / stop /
shift / weave events or (p 0.5) a learned policy from a league that the learner's own snapshots join every
64 updates, prioritized level replay (p 0.5), the constrained reward of v2R, KL 0.05 to the solo stage.
Optimizer moments dropped; re-measured identical after stripping (seed 77 vs interactive).

| ICCAS, 4 seeds × 4 planners, dial 0 | boot camp | s976 (prev. safest) | v2R (fastest) |
| --- | ---: | ---: | ---: |
| solo lap | 7.89 s | 7.91 s | **7.66 s** |
| car contacts | **102** | 119 | 160 |
| passes | 218 | **236** | 235 |
| prop contacts / km | **0.45** | 1.10 | 1.25 |
| unseen learned opponents B / gate, contacts / passes (2 seeds) | 26/22, 17/22 | 19/37, 23/29 | 18/14, 17/12 |

Its 30 % checkpoint scored 185 contacts / 0.87 props/km: the gains came over the whole run. The same boot camp
with KL 0 from v2R drifted (kl_ref 24, solo 9.8 s at 26 %) and was stopped.

## `iccas_bootcamp_safest.pt` (2026-10-08) — fewest car contacts so far (59), chosen from the run's last 20 checkpoints

The same run as `iccas_bootcamp_safe.pt` (bootA2R), **update 5856 of 6144** instead of the final. The run's late
checkpoints swing (33/65/45/33 contacts on a quick test of the last four), so the last 20 (u5536-u6144) were scored on
a quick test with **selection seeds 81/83** — not the table's seeds 77-80 — against `forzaeth_pred` and
`lane_switch`: u5888 17, u5856 23, u5824 34, u6016 35, …, the final 41, the worst 82. The best two went through the
full table below. Optimizer moments dropped; re-measured identical after stripping (seed 77, all six scenarios, same GPU; on another GPU
the float path differs and single contacts move by one).

| ICCAS, 4 seeds × 4 planners, dial 0 | **u5856 (this)** | boot camp final | s976 | v2R (fastest) |
| --- | ---: | ---: | ---: | ---: |
| solo lap | 8.62 s | 7.89 s | 7.91 s | **7.66 s** |
| car contacts | **59** | 102 | 119 | 160 |
| passes | 198 | 218 | **236** | 235 |
| contacts per pass | **0.30** | 0.47 | 0.50 | 0.68 |
| prop contacts / km | 0.70 | **0.45** | 1.10 | 1.25 |
| unseen learned opponents B / gate, contacts / passes (2 seeds) | 15/10, 14/8 | 26/22, 17/22 | 19/37, 23/29 | 18/14, 17/12 |

The price is pace and boldness: 0.73 s slower alone and far fewer passes against unseen learned cars. Use it where
contact matters more than lap time; v2R stays the fastest. u5888 scored 61 contacts / 168 passes on 3 seeds.

## `iccas_bootcamp_balanced.pt` (2026-10-09) — near-safest contacts, 0.4 s faster, a fifth of the prop contacts

`iccas_bootcamp_safest.pt` (bootA2R u5856) **continued for 3072 updates** in the same boot camp (bootA13R: lr
1e-5 → 5e-6, constraint multipliers starting at 300 where A2 ended, KL 0, seed 1062), **update 2784 of 3072**,
picked as before: its last 20 checkpoints scored on `forzaeth_pred` + `lane_switch` with selection seeds 81/83
(u2784 21, u2688 28, …, final ~60s), then the full table on the independent seeds 77-80. Its final checkpoint drifted
back to the A2-final level (99 contacts); the selected point did not. Optimizer moments dropped; re-measured
identical after stripping (seed 77, all six scenarios, same GPU).

| ICCAS, 4 seeds × 4 planners, dial 0 | **balanced (this)** | safest | boot camp final | v2R (fastest) |
| --- | ---: | ---: | ---: | ---: |
| solo lap | 8.20 s | 8.62 s | 7.89 s | **7.66 s** |
| car contacts | 66 | **59** | 102 | 160 |
| passes | 207 | 198 | 218 | **235** |
| contacts per pass | 0.32 | **0.30** | 0.47 | 0.68 |
| prop contacts / km | **0.15** | 0.70 | 0.45 | 1.25 |
| unseen learned opponents B / gate, contacts / passes (2 seeds) | **6/14, 9/14** | 15/10, 14/8 | 26/22, 17/22 | 18/14, 17/12 |

The best all-round point so far: seven more car contacts than the safest over 16 races, but 0.42 s a lap faster,
4.7× fewer prop contacts, and the fewest contacts against unseen learned opponents (15 over 4 races vs 29 / 43).

## Fixed-physics ICCAS policies (added 2026-10-16)

`iccas_bootcamp_safe.pt` trained on in the fixed contact physics (main 0cc3151: car-car impulses, shared props, every
wall contact counted). Only impacts at >= 1.5 m/s were charged, with annealed lagrange targets, no KL, and the contK
recipe otherwise (`pipeline/v1fix.sh`). The checkpoints are full: actor, critic, `meta` and `extra`.

| file | what it is | sha256 (first 16) |
| --- | --- | --- |
| `iccas_fixphys_s1102_u384.pt` | `v1fix_s1102`, update 384: the balanced one | `09ac138b10b77589` |
| `iccas_fixphys_s1101_u448.pt` | `v1fix_s1101` u384 + 64 more updates: the fastest one | `69d1272633ae7763` |

**8-seed noise protocol** (seeds 81-88 x 4 planners, race of 2, props on the line, ICCAS, 126-134 km each). "Hard" means
an impact at >= 1.5 m/s normal speed.

| policy | race speed | hard walls+props /km | hard cars /km | ICCAS solo lap (128 trials) |
| --- | ---: | ---: | ---: | ---: |
| `iccas_bootcamp_safe` | 4.10 m/s | 0.93 | 1.21 | 8.59 s |
| `iccas_fixphys_s1102_u384` | 4.23 m/s | 0.75 | 0.88 | 8.46 s |
| `iccas_fixphys_s1101_u448` | 4.37 m/s | 1.45 | 1.72 | 8.31 s |

s1102 u384 is faster than safe and hits hard less often. s1101 u448 is the fastest, at more hard hits.

## Korea 2026 policy (added 2026-10-17)

The same recipe (`v1fix`) run on a second map, `real:korea_2026_competition` + 16 warps, from `iccas_fixphys_s1102_u384`
(which had never driven Korea 2026). Two seeds, 512 updates each; the pick is `map2_s1201` update 384. Full checkpoint:
actor, critic, `meta` and `extra`.

| file | what it is | sha256 (first 16) |
| --- | --- | --- |
| `korea26_fixphys_s1201_u384.pt` | `map2_s1201`, update 384: the Korea 2026 pick | `d2bf8a5bffa6a979` |

**8-seed noise protocol on Korea 2026** (seeds 81-88 x 4 planners, race of 2, props on the line). "Hard" = impact >= 1.5 m/s.

| policy | race speed | hard walls+props /km | hard cars /km | Korea 2026 solo lap (128 trials) |
| --- | ---: | ---: | ---: | ---: |
| `iccas_fixphys_s1102_u384` (the start) | 3.57 m/s | 0.77 | 0.24 | 10.78 s |
| `korea26_fixphys_s1201_u384` | 4.02 m/s | 1.02 | 0.62 | 9.92 s |
| `map2_s1202` u256 (not archived) | 4.05 m/s | 0.76 | 1.17 | 10.02 s |
| `map2_s1202` u512 (not archived) | 3.96 m/s | 0.65 | 1.17 | 9.90 s |

On 2 seeds, `iccas_bootcamp_safe` drove Korea 2026 at 3.44 m/s with 1.63 / 0.42 hard hits per km and a 10.93 s solo lap.
The map-specific training made the car 0.86 s a lap faster and 13 % faster in races. Hard hits rose from 1.0 to 1.6 per
km, mostly with cars; that is the same level as the ICCAS picks above (1.6-2.1). The pick is nearly the fastest of the
four and has the fewest hard hits of the trained ones.

## Representative-floor picks (added 2026-10-21)

The user's direction of 2026-10-20: train on one representative floor (friction x1.0 = 1.0489, rear 0.92), with no
friction randomisation. The venue's floor is answered by a speed knob on the plans (`plan_speed_scale`), not by the
policy. ForzaETH's MAP follows the plans in the car's controller node and in the console. Optimizer state dropped
(`extra.opt`); actor, critic, `meta` and `extra` (the contract: MAP follower and its 20 settings, dial held at 1.0489)
are kept.

| file | what it is | sha256 (first 16) |
| --- | --- | --- |
| `korea26_rep_s2001_u256.pt` | `k26rep_s2001` update 256: Korea 2026 retrained on the representative floor with MAP in the loop, from `korea26_fixphys_s1201_u384` | `1853f2bf44a96546` |
| `icra2022_map3R2_s1321_u384.pt` | `map3R2_s1321` update 384: the ICRA 2022 pick (solo stage then race stage on ICRA) | `2c349e21e49e90d0` |

**Round-6b finals** (representative floor for every car; fresh seeds: race 91-92 x 4 planners, race of 2, props; solo 128
trials). "Hard" = impact >= 1.5 m/s, walls+props+cars.

| map | policy (MAP follows) | solo lap (done /128) | race speed | hard /km |
| --- | --- | ---: | ---: | ---: |
| Korea 2026 | `korea26_rep_s2001_u256` | 9.32 s (128) | 4.40 m/s | 1.92 |
| Korea 2026 | `korea26_fixphys_s1201_u384` (iLQR) | 9.68 s (127) | 4.14 m/s | 1.54 |
| ICCAS | `iccas_fixphys_s1102_u384` | 7.73 s (128) | 4.53 m/s | 2.15 |
| ICRA 2022 | `icra2022_map3R2_s1321_u384` | 14.61 s (127) | 3.61 m/s | 2.88 |

The same retraining did not win on ICCAS or ICRA. On the selection seeds, the retrained checkpoints were slower solo
(+0.29 s and +0.18 s) and in races, with fewer hard hits, so the earlier picks stay there.

## Korea 2026 entry: the v2 network (added 2026-10-23)

`korea26_v2_distill_s2362_it7.pt` is the first entry in the output format the user asked for on 2026-10-13: a
trajectory of four waypoints, each with its own speed (wp4). It replaces `korea26_rep_s2001_u256` as the Korea pick.

How it was made:
- The v2 network reached v1 solo pace on Korea (S stage) but raced by following: 3.4 m/s and 0.5-1.3 passes/km, against
  the v1 pick's 4.5 m/s and 2.5-3.2.
- PPO variants made it safe or fast, never both.
- The v1 Korea pick taught it by DAgger (`--wp4-label policy`, `learn/policy_teacher.py`): 6 iterations
  (v2PT_s2361), then 8 more (v2PT_s2362).
- Recorded in `extra`: the follower it was selected with (MAP, the Korea round-5 controller) and the held dial (1.0489).
  DAgger itself does not record them.

**Finals** (fresh seeds: race 91-92 x 4 planners, race of 2, props; solo 128 trials; representative floor; MAP follows):

| policy | solo lap (done /128) | race speed | hard /km | passes /km |
| --- | ---: | ---: | ---: | ---: |
| `korea26_v2_distill_s2362_it7` | 9.28 s (128) | 4.51 m/s | 2.68 | 2.94 |
| `korea26_rep_s2001_u256` (the previous pick) | 9.32 s (128) | 4.43 m/s | 2.35 | 3.17 |

The replacement rule was fixed before the run: solo within +0.05 s, done within -1, race speed at least v1's, hard hits at
most 1.15x v1's. It passes on every term. The hard-hit term is the tightest: 2.68 against a limit of 2.70.
