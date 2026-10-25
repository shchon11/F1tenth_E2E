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

### Correction (2026-10-23, later the same night): the Korea entry is `korea26_rep_s2001_u256` again

The distilled v2 policy uses the wp4 format but not its freedom. Over 13,184 executed plans (Korea solo, MAP), its four
waypoint speeds sit 0.002 m/s (mean) off a straight line through the first and last, and 0 % of them dip in the middle.
That is v1's two-endpoint linear profile, which it was taught to copy (the labels are v1 plans refitted to wp4). The v2
policies trained by PPO alone use the freedom: 0.64-0.74 m/s off the line, 44-50 % with a dip in the middle. 256 PPO
updates of the v1 R recipe after the distillation moved it to 0.027 m/s. A copy of v1 in v2's format gains nothing over
v1, and v1 hits less hard (2.35 vs 2.68 /km), so v1 is the entry again. The v2 file stays archived as the start of PPO runs
that may leave the linear profile.

## Korea 2026 entry (2026-10-23 morning): `korea26_v2_ppo_s2421_u512`, a v2 policy trained past the copy

The distilled v2 above was used only as a start (DAgger = init). PPO followed, with the S stage's update granularity
(horizon 32, minibatch 1024, lr 2e-5 -> 1e-5), MAP in the loop, the representative floor and the dial held. The lagrange
multipliers were held at 80: at 40 or adaptive, hard hits stayed at 3.0-3.9 /km. 512 updates, run `v2PH_s2421`. Optimizer
state dropped; extra records MAP + its 20 settings and the dial 1.0489 (trained so).

| policy (MAP) | solo lap (done /128) | race speed | hard /km | waypoint speeds off a line / dips |
| --- | ---: | ---: | ---: | ---: |
| `korea26_v2_ppo_s2421_u512` | 9.01 s (128) | 4.59 m/s | 2.32 | 0.143 m/s / 0 % |
| `korea26_rep_s2001_u256` (v1, the previous entry) | 9.32 s (128) | 4.43 m/s | 2.35 | linear by format |
| `korea26_v2_distill_s2362_it7` (the start) | 9.28 s (128) | 4.51 m/s | 2.68 | 0.002 m/s / 0 % |

Finals: fresh seeds, the representative floor. The replacement rule fixed beforehand passes on every term: solo -0.31 s,
race +3.6 %, hard hits equal, all 128 done.

The profile is curved, not a copy of v1's line, but no plan brakes and accelerates again inside it. PPO runs that did use
dips (lambda adaptive: 39 % of plans) hit 25-36 % harder and were seed-unstable.

On slipperier floors, without the venue's speed knob, it keeps its speed where v1 slows:

| floor | v2 entry | v1 |
| --- | --- | --- |
| x0.85 | 9.28 s (128), 4.50 m/s, 2.66 hard /km | 9.54 s (128), 4.28 m/s, 2.01 |
| x0.75 | 9.57 s (127), 4.41 m/s, 3.60 | 9.80 s (128), 4.17 m/s, 2.68 |

At a slippery venue, set the knob (`plan_speed_scale`). Console path, Korea, 30 s: solo 5.34 m/s, race of 2 5.22 m/s,
0 contact steps (v1: 5.02 / 4.57).

## Korea 2026 entry with a retrained nearest-car output (2026-10-25): `korea26_v2_ppo_s2421_u512_nearcar`

The same policy as `korea26_v2_ppo_s2421_u512`: only the four tensors of `actor.near_car` differ (verified), so it drives
identically. That output -- the policy's own estimate of the nearest car it sees (exists, dx, dy, closing speed, sideways
speed) -- does not feed the action. It was retrained on the policy's frozen features from 15 race cells of the entry
(seeds 85-86), with more weight on the closing and sideways speeds (`docs/research/passing-risk-2026-10-24`, section 6).
Use this file with the speed limit for a car ahead switched on: the console's 「앞차 충돌 방지 감속」, the controller node's
`opponent_speed_limit:=true`, `evaluate --opponent-speed-limit`.

| finals race cells, fresh seeds 91-94 (~70 km a row) | hard car /km | hard all /km | race speed | passes /min |
| --- | ---: | ---: | ---: | ---: |
| props on the line, limit off | 1.36 | 2.34 | 4.59 m/s | 0.91 |
| props on the line, limit on | 0.90 (-34 %) | 2.04 (-13 %) | 4.53 m/s | 0.96 |
| props off the line, limit off | 1.21 | 1.75 | 4.67 m/s | 0.77 |
| props off the line, limit on | 0.58 (-52 %) | 1.35 (-23 %) | 4.58 m/s | 0.75 |

Solo laps unchanged (the limit never engaged without another car). For races with the ROS graph launches, also pass
`speed_cap:=9.0`: their default cap of 4 m/s leaves no car to catch. 30 % of the limits are ones the true state would not set; most extra wall hits
(+0.1-0.2 /km) follow those.

## The speed limit reads near-stopped cars (2026-10-25): `korea26_v2_ppo_s2421_u512_nearcar_slow`

The same policy again (only `actor.near_car` differs from `korea26_v2_ppo_s2421_u512`, verified; it drives identically),
with the nearest-car output retrained once more, now with near-stopped cars in its data. Use it instead of
`korea26_v2_ppo_s2421_u512_nearcar` with the speed limit for a car ahead switched on.

Why: a car crawling on the racing line (a raceline car at x0.05 of the line's speed) is rear-ended at 2.5 hard /km, and
the first limit barely helped (2.19). Its output had never seen a slow car: it missed 37 % of them and read their closing
speed 1.2 m/s low -- and on the real venue bags it read the fixed objects on the track as cars doing 1.5 m/s. With this
output, and with the limit now holding a car it already holds out to 0.5 m sideways (`OpponentSpeedLimit.lane_exit`; the
output's lateral offset drifts outward in the last 0.2 s before a hit), the limit does most of what the true state would
(0.94). Details: `docs/research/passing-risk-2026-10-24`, section 11.

| fresh seeds 91-94, same card per row, limit on | hard car /km | hard all /km | race speed |
| --- | ---: | ---: | ---: |
| near-stopped car x0.05, `_nearcar` | 2.74 | 2.90 | 4.75 m/s |
| near-stopped car x0.05, `_nearcar_slow` + latch | 1.33 (-51 %) | 1.55 | 4.69 m/s |
| finals cells, props on the line, `_nearcar` | 1.01 | 2.25 | 4.53 m/s |
| finals cells, props on the line, `_nearcar_slow` + latch | 0.85 | 1.95 (-13 %) | 4.51 m/s |
| finals cells, props off the line, `_nearcar` | 0.58 | 1.35 | 4.58 m/s |
| finals cells, props off the line, `_nearcar_slow` + latch | 0.58 | 1.21 (-10 %) | 4.58 m/s |

Alone on a track with props on the line the limit now holds the speed on ~1 % of the steps (props read as a car; 0.25 %
with `_nearcar`), at -0.4 % speed.

### With the speed limit for an obstacle on the plan's path (2026-10-26)

For races with boxes on the track (the venue had them), also switch on the obstacle limit: the console's 「장애물 충돌 방지
감속」, the controller node's `obstacle_speed_limit:=true`, `evaluate --obstacle-speed-limit`. It holds the speed command to
the speed from which the car can still stop before the first small isolated cluster of the scan on the plan's own path (a
box, a pole; walls are long and left out -- no map). Returns at the policy's nearest-car reading are left to the car limit,
so it needs this file's output too (a v2 checkpoint with `near_car`; refused without one). Details:
`docs/research/passing-risk-2026-10-24`, section 13.

| finals race cells, props on the line, `_nearcar_slow` + latch (car limit on in both rows) | hard all /km | hard prop /km | race speed |
| --- | ---: | ---: | ---: |
| seeds 81-84, car limit only | 2.19 | 1.24 | 4.46 m/s |
| seeds 81-84, + obstacle limit | 1.74 (-21 %) | 0.72 | 4.41 m/s |
| fresh 91-94, car limit only | 1.95 | 0.91 | 4.51 m/s |
| fresh 91-94, + obstacle limit | 1.70 (-13 %) | 0.75 | 4.44 m/s |

These are the rule since 300e051 (2026-10-26). The first numbers published here (-24..-39 % at -4..-5 % speed) came from a rule
that read a missed return as background: walls seen at a grazing angle set half of its holds and slowed the car at walls.
Boxes standing against a wall are read as part of it and not held for.

Alone with props on the line (the earlier rule): 4.84 -> 4.81 m/s, no more hits. One control cycle of it costs 0.6 ms on one core of the lab
PC (Ryzen 7 260); measure it on the car's AGX Orin before relying on the margin.

## ICCAS and ICRA picks record their follower (2026-10-23)

`iccas_fixphys_s1102_u384` and `icra2022_map3R2_s1321_u384` were selected and scored with MAP (each map's round-5
controller, chosen on tracking RMSE) at dial 1.0489, but trained with the iLQR, and their files said so: the car's
controller node and a plain `evaluate` read the contract and ran them on the iLQR. Their `extra` now carries
`plan_follower: map` and that map's MAP numbers at the top level (`contract_of` reads it before the experiment block, which
still says how they were trained), plus an `archive_note`. Weights and meta unchanged. Checked on both paths: the console
drives them with `MAPPlanTracker` (ICCAS solo 5.75 m/s, ICRA 4.26 m/s over 30 s, 0 contact steps), and the controller node
starts with `plan follower map` and each map's numbers. They were trained on drawn dials and record none, so on the car the
dial is graph.yaml's (or `dial:=`).
