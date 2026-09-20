# Quarantine the empirically rejected local controller

Evidence: final policy walls 31 vs old auto 3; unchanged D3 with local controller walls 14 vs old auto 3. Do not alter training/evaluation artifacts or frozen source.

1. Runtime: default ordinary auto to historical-global-v1 with existing nominal 4WD fields. Add an internal explicit auto_profile selection, independently of estimator approval. Resolve saved profile before runtime construction and reject explicit conflicts. Keep old saved local runtime identifiers valid and distinguish new historical metadata.
2. PPO: select local-v2 explicitly for adaptive speed/full runs; do not widen research_estimator permission. Existing saved runtime pairs remain binding.
3. Latest selection: skip unqualified local-v2 checkpoints only during automatic selection. Require controller-level qualification.approved is exactly true. Manual selection still uses normal structural/estimator validation and the saved profile.
4. CPU tests: default profile and nominal math parity, fresh adaptive intent, bound saved local reload, profile/version mismatch, research capability independent of observer approval, latest quarantine and explicit manual access. Run existing runtime/latest/loading coverage. No GPU or source-snapshot edits.
