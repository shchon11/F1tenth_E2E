# Controller arms on suite v1: fixed_low is the deployment default; reactive is not adoptable

**2026-09-12, late.** The frozen original policy scored under two more plan-controller arms on the
same frozen suite v1 (freeze `6f710412bacf5a09`, 34 cells / 272 trials, estimated arm rows from the
published table). **Suite v1 is in-distribution** (two of its three maps are training maps; see the
held-out suite v2 work); these numbers rank controllers on the same policy, they do not measure
generalisation.

| arm | S /144 | low-µ /48 | A /64 | O /64 | coll/km ↓ | large-slip s/km ↓ | paired lap Δ vs estimated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 110 | 16 | 18 | 44 | 12.07 | 7.16 | −1.25 s |
| estimated | 134 | 41 | 41 | 42 | 6.07 | 3.42 | 0 |
| **fixed_low** (µ = 0.73423 constant) | **136** | **43** | **43** | 42 | 5.58 | **2.65** | +0.22 s |
| reactive (slip-triggered, worktree `feat/reactive-grip` @ fe4e3c7) | 134 | 41 | **33** | 42 | 7.12 | 4.33 | **−0.66 s** |

`fixed_low` needs no estimator, is the safest arm on every stability column, and costs 0.22 s per
lap (~2 %) against `estimated`. It is the deployment default from here.

`reactive` (start at µ 0.95, cut on a yaw-rate deficit, recover slowly) delivers the speed it was
built for and loses avoidance: obstacle contacts happen head-on before any lateral slip exists, so
the detector never fires and the car arrives at the obstacle with a nominal-friction speed
envelope. The worker's CPU comparison (solo cells only) could not show this. The branch is kept as a
record and not merged. A longitudinal slip (wheel-spin / lock-up) rule remains a real-car-only
addition: the simulator has no wheel state and its odometry is ground speed plus noise, while the
real recordings show both signatures clearly (see the 2026-09-12 bag scan in the orchestration notes).

Files: `controller-arms-2026-09-12-{fixed_low,reactive}-cells.jsonl` (34 rows each, as produced),
`controller-arms-2026-09-12-roster.json`. Scored from the `feat/reactive-grip` worktree on the GPU
while training ran alongside; runtime digests differ from the published rows in `grip_*.py` only.
