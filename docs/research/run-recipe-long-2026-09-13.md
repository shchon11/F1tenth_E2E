# Run `cl_long_s701` (2026-09-13 20:10): the long run on everything merged today

User: "상황이 다양하다 보니 학습을 오래해야할 수도 … 로스도 좀 크게크게 … 끝나고 정리되고 긴 런".
A first launch at 12:40 from `15f05bd` was stopped after ten minutes (it predated three merges). This
one runs from main `282fdfe`: memory policy, wheel model, hard obstacles, per-reset procedural
layouts, opponent population + reactive behaviours + spawn diversity, clearance arm (runtime only).

| ingredient | value | why |
|---|---|---|
| policy | memory GRU 128 + scan channels `memory,edges`, warm-started from the frozen original | the only architecture whose early checkpoints reached 137/256 on the proxy (`memory-policy-2026-09-13.md`) |
| plant | wheel model on (default) | measured from the bags |
| tracks | 264 = 204 (train + gapped `+hard` + user scenes 2334/2344) + 60 `gen:recipe` random maps (50 clean, 10 `+hard`) | map diversity at zero engineering cost; no held-out leakage (guard) |
| opponents | **race size 3**, `--opponent pool` = teacher, self, frozen original, A701, mem_u8; `--spawn-order random`; opponent speed 0.6–1.15×; reactive `defend 0.3 / yield 0.2 / line 0.3 / oblivious 0.1` + timed brake/stop/shift at 1.0 | the census showed these situations were absent before (`opponent-diversity-2026-09-13.md`) |
| obstacles | `--procedural-obstacles 1.0`: a fresh prop layout per environment at every reset, on top of the rasterised `+hard` variants | fixed layouts let speed be memorised (`failure-attribution-2026-09-13.md`) |
| steps | **8 388 608** (1024 updates), lr 1e-4 → 3e-5, KL 0.01 | long, larger steps, loose leash |
| `--car-safe-gap` | **0.45 m** (was 0.9) | 0.9 m is wider than the lateral room in a 1.4 m lane: every pass there was a penalty band, which taught fast side-swipes (`failure-attribution-2026-09-13.md` §5) |
| `--overtake-bonus` | **5.0** (was 1.0) | measured +0.03/s at 1.0 — 1 % of progress, effectively absent (`reward-audit-2026-09-13.md`) |
| `--plan-clearance-penalty` | **4.0 @ 0.25 m** (was 0; 8.0 in `cl_margin_s701`) | the failure is zero-margin plans; 8.0 held the proxy at 122–123 for two checkpoints; 4.0 is the compromise with pace |
| checkpoints | every 32 updates; every 64th scored on the **128-trial** proxy under `fixed_low` (`work/bigproxy-20260913/bigproxy.sh`) | the 32-trial proxy (±10) misled the day; ±5 now |

Selection: best (obstacle + traffic completions, ties by collisions/km) → `~/f1sim_runs/cl_long_s701/ppo_best.pt`,
table in `work/learning-next/long-20260913/select/selection.md`. Not in the training loop: the clearance arm (runtime layer; deploy as `fixed_low+clearance`, the
grip-clamp lesson applies). Validation of the opponent configuration: a CPU census before launch
(`work/learning-next/long-20260913/`).
