# Test specification

Unit: preserve low tactical target; straight acceleration unaffected by distant curve; high/low mu corner limits; current-state reachability/overspeed; two-axle combined force/power/braking/drag/slew feasibility; historical solver parity; per-stage bounds; no GT leakage; cold/evidence/hold/fault/reset; independent recurrent-reference hidden state; exact frozen rows after optimizer; checkpoint roundtrip.

Integration: GRU+auto+2/3car PPO smoke and resume; correct policy/estimator normalizers and source hashes; actor and controller CUDA graph parity; ordinary viewer selects one auto path and reports three speed profiles/limits coherently.

Empirical: policy-domain estimator calibration and ordered-stream response; frozen-D3 controller ablation; speed-only then bounded full PPO; independent development selection and untouched final paired solo/obstacle/traffic matrix at low/high grip and2eval seeds. Check speed/progress jointly with collisions/km, contact/held passes and slip exposure. Do not promote from training reward or shuffled-window accuracy alone.

Numerical empirical thresholds and exact caseIDs will be sealed after acceptance analysis and before the first training rollout. Prior failed V2A/V2B outcomes remain immutable.
