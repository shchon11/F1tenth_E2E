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
