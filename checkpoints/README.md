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
