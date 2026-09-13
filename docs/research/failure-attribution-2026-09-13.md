# Where the policy fails, and why training did not move it (2026-09-13)

Four finetunes today (A701, hard+events, free leash, memory policy) scored the same on a held-out
proxy. This note is the wide-angle look the user asked for ("어디가 병목인지 넓은 시야를 가지고").
Everything below is measured on the same eight held-out proxy tracks (32 first-attempt trials each,
random μ, 3-lap budget): `scene:scene_0912_2355` (the user's editor scene), its `+hard1`,
`real:map16x07+hard2`, `real:map12x16+hard3`, `gen:control:9100+hard4`, `real:korea_2025_iccas+hard5`,
`real:map16x07`, `real:map12x16`. Script: `failure-attribution-2026-09-13-crash_attribution.py`;
per-checkpoint JSON beside it.

## 1. The ceiling: the tracks are feasible

The privileged raceline teacher (true μ, the map's racing line, the same iLQR tracker the policy
uses) on the tracks whose raceline fits the car:

| track | teacher | speed |
|---|---|---|
| `gen:control:9100+hard4` | 30/32 | 4.5 m/s |
| `real:korea_2025_iccas+hard5` | 30/32 | 4.3 |
| `real:map16x07+hard2` | 30/32 | 3.3 |
| `real:map16x07` | 31/32 | 3.4 |
| `real:map12x16` | 32/32 | 3.5 |
| `scene:scene_0912_2355` | 0/32 | — (the raceline ignores the scene's props; a teacher limit, not the map's) |

The policies complete 7–24 of 32 on the same tracks under the deployment arm (`fixed_low`: frozen
102/256, memory u8 136/256) and 88 / 137 under `legacy`. The gap to a 30/32 ceiling is the whole
problem; it is not a proxy artefact and not the tracker.

## 2. What the policy's own plan looked like at the moment of collision

`tracker.last_ref` is the policy's decoded plan in the body frame. At every collision: did that plan
pass through an occupied cell, and how close did it come (body-edge margin = EDT − 0.14 m)?

| checkpoint | collisions | plan through occupied | plan margin < 0.10 m | < 0.25 m | median margin | car clearance 25 ms before impact | impact speed |
|---|---|---|---|---|---|---|---|
| frozen original | 223 | 49 % | **85 %** | 89 % | 0.00 m | 0.21 m | 3.7 m/s |
| A701 | 218 | 46 % | 82 % | 85 % | 0.00 | 0.21 | 3.7 |
| free u88 (trained with plan-clearance 1.0 @ 0.25 m) | 219 | 43 % | 83 % | 91 % | 0.01 | 0.20 | 3.5 |
| memory u8 | 180 | 39 % | 77 % | 85 % | 0.01 | 0.22 | 3.6 |

Per track (frozen): on the narrow real floors without any obstacle the policy hits walls at
2.6–2.9 m/s with μ 0.93 — slow and grippy — with a plan margin median 0.01 m. On the obstacle
tracks the plan runs through boxes (`map12x16+hard3`: 17 of 32 collisions into boxes, median margin
−0.14 m). Low-μ (< 0.85) episodes are 31 % of collisions, not the majority.

## 3. Reading

* The policy is not blind and the car is not sliding: it **plans lines that graze or cross what it
  sees**, and the tracker executes them faithfully. That is why memory and scan channels changed
  the plan-margin distribution by a few points only, and why loosening the KL leash did nothing.
* Nothing in the reward prices a margin until it becomes a collision (`reward-audit-2026-09-13.md`:
  proximity −0.07/s, plan-clearance 0). On fixed layouts a zero-margin line is free most of the time
  and pays in progress. The plan-clearance penalty at coefficient 1.0 (free run) left the margin
  distribution unchanged — 83 % under 0.10 m after 88 updates — so at that weight it was noise
  against +4/s of progress.
* The one thing that moved held-out numbers today was a runtime layer that needs no training
  (`fixed_low`, +14 for the frozen policy on this proxy). A margin-keeping layer is the geometric
  twin of it.

## 4. What follows from it

1. `cl_margin_s701`: recipe A with the plan-clearance penalty at **8.0 @ 0.25 m** and nothing else
   changed; judged by the held-out proxy **and** by this attribution — the margin distribution must
   move off zero, or the penalty as implemented is not reaching the policy.
2. A `clearance` controller arm (worker, `feat/clearance-arm`): a LiDAR-local occupancy and distance
   field on the runtime path, the plan shifted or slowed to keep ≥ 0.20 m, composable with
   `fixed_low`, proven with the same script before/after.
3. Per-reset procedural obstacle layouts (worker, `feat/procedural-obstacles`) so that a
   zero-margin line stops being free during training.
