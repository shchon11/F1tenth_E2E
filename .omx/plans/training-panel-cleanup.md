# Final scoped training-panel cleanup

Scope: new training_schema, training_schema_export, training_setup, checkpoint_metadata modules; current changes in training.py, opponent_table.py, and the pure opponent event contract. Runtime learner algorithms and the immutable scientific source are outside this cleanup.

Behavior lock: 162 console/schema/startup/layout tests, independent 88-test review, real CPU panel launches for all eight modes, same-stage resume with zero added updates, custom report-path readback, and replay fitting.

Order: inspect dead imports and remnants of the old form; inspect duplicate configuration ownership; inspect process lifecycle and optional-value boundaries; retain only necessary compatibility aliases. Delete only clearly unused code. Do not rewrite working field editors, checkpoint checks or solver code for style.

Result: old form removed in favor of one schema-driven editor; stale training.py imports removed; pure event names/parsing extracted once with runtime re-exports. No further safe, meaningful deletion was found in the final pass. Python static checks are clean in scoped modules. There are no new application dependencies. The build verification used the already-declared setuptools requirement in an isolated scratch environment.

Remaining limits: imported legacy opponent policy metadata can still perform its existing synchronous CPU inspection on file selection. Initial training-panel construction and new adaptive checkpoint inspection are verified Torch-free in the UI process. Scientific model acceptance remains separate from panel completion.
