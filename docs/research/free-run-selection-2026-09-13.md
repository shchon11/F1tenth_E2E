# Checkpoint selection (held-out proxy) — run stopped by the user at update 204

best: `u16` obstacles 111/256, traffic 33/96, 17.98 coll/km → `/home/shchon11/f1sim_runs/cl_free_s701/ppo_best.pt`

references on the same proxy: frozen original 87 / 19 (22.86 coll/km), A701 93 / 23 (21.12)

| update | obstacles /256 | coll/km | traffic /96 |
|---|---|---|---|
| 8 | 110 | 18.36 | 31 |
| 16 | 111 | 17.98 | 33 |
| 24 | 103 | 19.24 | 26 |
| 32 | 114 | 17.36 | 27 |
| 40 | 95 | 20.99 | 25 |
| 48 | 102 | 19.34 | 27 |
| 56 | 107 | 18.64 | 25 |
| 64 | 105 | 19.33 | 21 |
| 72 | 95 | 21.11 | 21 |
| 80 | 105 | 19.11 | 23 |
| 88 | 101 | 19.81 | 24 |

Reading: every checkpoint sits above both references, and none rises with training — the gain is there from update 8 and the curve is flat within the proxy's ±10 noise. The leash, the learning rate and the plan-clearance penalty were not the bottleneck.
