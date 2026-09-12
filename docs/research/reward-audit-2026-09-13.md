# Reward and loss audit (2026-09-13)

User: "보상체계 싹 정리 … 근거가 있는 항목만 넣어. 뇌피셜 말고." This note separates the two
objects people call "the objective", lists every term the code actually applies with its measured
weight, and marks what evidence exists for each. Nothing here is a design opinion; where there is
no evidence it says so.

## 1. Reward vs loss

* **Reward** is paid by the environment every control step (`gym_env.py`, `REWARD_COMPONENT_KEYS`).
  The policy's goal is the discounted sum of it, and the critic learns to predict that sum. It defines
  *what behaviour is good*.
* **Loss** is what the optimiser minimises per update (`ppo.py`): the PPO clipped surrogate
  (raise the probability of actions whose advantage — reward-to-go minus the critic's estimate —
  was positive), the value regression, an entropy bonus (0 here), a **KL leash** to the frozen
  original actor, and two auxiliary supervised heads. It defines *how the network moves* toward the
  reward, and the leash/aux terms are the only things in the loss that are not derived from reward.

## 2. Reward terms as trained (recipe A = the original's final phase, `ppo_race_0910` W&B config)

Measured: 64 envs × 800 steps (51 200 learner-steps, ~100 episodes) on six training-style tracks
(real, obstacle, hard, user scene, generated) with race size 2, mixed opponents, events on, policy
sampling stochastic. Per-second figures = per-step × 40. A701 and the frozen original give the same
picture (both JSON files beside this note); A701 shown.

| term (flag) | value | what the code charges | measured /s | steps active | evidence for the term itself |
|---|---|---|---|---|---|
| progress (`reward_progress`) | 1.0 /m | signed centreline progress | **+4.00** | 100 % | the objective; the only term every note measures against |
| collision (`--collision-penalty`) | −10 | once, on wall/obstacle hit, ends the episode | −0.79 | 0.2 % | none per term; the original converged with it. It is the *only* signal an obstacle sends |
| collision_speed (`--collision-speed-penalty`) | 0.5 /(m/s) | extra at impact speed | −0.13 | 0.2 % | none |
| steer_rate (`--steer-penalty`) | 0.05 | per unit of normalised steer change | −0.25 | 100 % | none; 6 % of progress, always on |
| proximity (`--proximity-penalty`, `--safe-dist`) | 0.5 /m, 0.30 m | per metre driven with the body within 0.3 m of a wall | −0.07 | 12 % | none |
| plan_clearance (`--plan-clearance-penalty`) | 0 in A; 1.0 in `cl_free_s701` | plan points within margin of occupied cells | 0 (A) | 0 % | tested once (free run, confounded with the leash): no held-out change |
| wrong_way (`--wrong-way-penalty`) | 0.2 /step | facing backwards | **0.00** | 0 % | never fires under these policies |
| lap (`--lap-bonus`) | 5 × avg speed | once per completed lap | +0.11 | 0.0 % | none |
| alive (`reward_alive`) | 0 | — | 0 | — | off |
| car_contact (`--car-contact-penalty`) | 5 | touching another car (ends episode) | −0.15 | 0.1 % | none per term |
| overtake (`--overtake-bonus`) | 1.0 /m | arc gained on opponents, signed, dense | +0.03 | 70 % | the *formulation* was measured (code comment: signed gap fixed a −2.00 step on pass completion); the value is not |
| lap_time (`--lap-time-bonus`) | 2 | per sector beating the track's own reference | +0.15 | 0.7 % | none |
| car_proximity (`--car-proximity-penalty`, `--car-safe-gap`) | 0.8 /m, 0.9 m | per metre within the gap of another car | −0.14 | 8 % | none |
| sideslip (`--sideslip-penalty`, `sideslip_free`) | 0.5, 0.06 rad | per metre per radian of drift beyond 3.5° | **−0.02** | 12 % | none; negligible weight |

Total measured reward: +0.069/step (+2.75/s). Progress is 4.0/s; everything else together is −1.3/s,
of which collision (−0.79) and steer rate (−0.25) are most of it.

## 3. Loss terms

| term (flag) | value in A | evidence |
|---|---|---|
| PPO clip (`--clip`) | 0.2 | standard; untested here |
| value loss (`--vf`) | 0.5 | standard; untested here |
| entropy (`--ent`) | 0 | off |
| KL leash to the original (`--kl-coef`, `--kl-decay`) | 0.05, 40 M | **tested 2026-09-13**: 0.05 / 0.01 / 0 land in the same held-out band (`free-run-selection-2026-09-13.md`, `cl_kl0_s701` u8 116/30) — no effect on held-out |
| aux grip head (`--aux-grip`) | 1.0 (original final phase: 0.1) | **SGR screen** (`static-grip-retention-2026-09-12.md`): no aux × anchor recipe met the +2 pp gate; seeds disagree in sign — no evidence of benefit |
| aux opponent head (`--aux-opp`) | 1.0 | none |
| critic warm-up (`--critic-warmup`) | 10 | none |

## 4. What the evidence supports, and what it does not

* **Evidence for the set, not for the terms.** The original policy was trained with exactly this reward
  and it drives; every finetune that changed the training distribution but kept the reward
  (A701, hard+events, free, kl0) scored the same on held-out. That is evidence the reward is not the
  lever for the failures we measure, and it is *not* evidence for or against any single term.
* **Three terms do nothing measurable** under the current policies: `wrong_way` (never fires),
  `sideslip` (−0.02/s, 0.4 % of progress), `alive` (off). Removing them cannot change the objective
  the policy sees; keeping them costs nothing. That is a magnitude argument, not an ablation.
* **The failures we care about are rare events under this reward.** A collision is 0.2 % of steps and
  carries −10; between collisions the obstacle is invisible to the reward. Progress (+4/s) is
  continuous. Whether a denser obstacle signal helps is untested: `plan_clearance` was tried once,
  confounded, and moved nothing.
* **No per-term ablation has ever been run.** Every value above was set once on 2026-09-10 and
  inherited since.

## 5. The cleaned reward (proposal, to be tested — not adopted by this note)

Keep what has measured weight and a stated purpose; drop what fires never or negligibly; test the
rest. Terms: `progress`, `collision`, `collision_speed`, `steer_rate`, `proximity`, `car_contact`,
`car_proximity`, `overtake`, `lap`/`lap_time`. Drop: `wrong_way`, `sideslip`, `alive`. Untested
knobs to ablate (one run each, same selector proxy, ≤ 1 M steps): steer_rate 0.05 → 0.01,
collision_speed 0.5 → 0, lap/lap_time → 0, aux heads → 0. Given §4, expect flat held-out numbers;
the value of the ablation is to *know* which terms can go, not to gain points.
