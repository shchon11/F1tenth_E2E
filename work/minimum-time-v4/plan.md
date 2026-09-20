# Correction: explicit minimum-time optimization and obstacle geometry

- Replace the bounded nested-profile heuristic with a joint periodic path/squared-speed NLP using the existing SciPy/Torch dependencies. Objective is traversal time, without curvature penalties, hidden 90% grip reserve or arbitrary 1 m offset bounds. Keep explicitly requested track margin.
- Use spline derivatives in the fixed reference coordinate. Enforce front/rear combined grip, longitudinal load transfer, current/power/braking, true steering angle/rate and occupied-map clearance for the oriented vehicle footprint.
- Require solver convergence and independent dense physical/geometry checks; preserve optimizer metadata through cache; never cache nonconvergence as an optimized path. This is a reduced quasi-steady model, not a global optimum or a full Pacejka/wheel transient OCP.
- Project analytic props into planner collision geometry without changing their simulation representation. Regenerate caches from this projected geometry.
- Verify analytic toy optimum, gradients, full corridor use, mesh checks, failure/cache behavior, real TRAIN maps and authored obstacles. Do not reuse fixed prior qualification numbers as evidence for the changed source.
- Audit actual teacher modes, per-race opponent behavior, and past experiment settings; answer labeling diversity separately from opponent diversity.
