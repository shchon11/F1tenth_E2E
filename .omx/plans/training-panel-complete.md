# Training panel coverage and delivery plan

User update: all supported training, teacher, map/obstacle and PPO reward settings must be available without typing CLI flags. The current PPO-only recipe form and freeform extra-arguments box do not satisfy this.

1. Keep ongoing adaptive-racing experiments on their immutable source snapshot. UI work changes the repository, not the frozen cohort.
2. Derive a checked-in, Torch-free settings schema from the real argument parsers for PPO, DAgger, policy-domain grip collection/fitting/evaluation and the controlled grip pilot. Test complete parser coverage, types, choices and defaults. No duplicated handwritten trainer parameter list.
3. Add a typed, searchable setup form organized by task: experiment/checkpoints, maps and obstacles, teacher/opponents, model/controller learning, optimizer/curriculum, PPO rewards and outputs. Use existing TrackPicker and OpponentSlotTable. All options reachable as actual controls; no requirement to type extra CLI arguments.
4. Preserve named recipes as starting points and user-edited values per algorithm. Add configuration save/load, file/folder selectors, launch validation, launch and resume. Keep experimental controller arms in advanced reproduction settings, not the normal driving picker.
5. Reuse existing JobManager, stop controls, logs and checkpoint browsing. Observer jobs must use their real output directories and show epoch/loss/progress; DAgger iterations and PPO updates retain their own charts. Launching is argv-only with no shell.
6. Verify parser roundtrips, configuration persistence, invalid cross-field combinations, map/obstacle/opponent selections, resume commands, all algorithm launch paths using fake subprocesses, and real short CPU smoke jobs. Perform offscreen Qt rendering at normal and compact window sizes and inspect screenshots. Existing relevant console tests remain valid or are updated only for deliberate UI contract changes.

Ownership: backend agent owns new schema/export/validation files and tests. UI agent owns training.py and new setup UI file and its tests. Parent owns integration decisions, experiment progression and final review. No runtime learner changes are required for schema introspection.

Boundary cleanup: the opponent editor currently imports Torch through pure event-name parsing. First run existing event/slot/table regressions, then move only canonical constants and parse/split functions to a Torch-free module and re-export them from the runtime module. Keep all event IDs, bitmasks and parsing behavior unchanged. Parent owns the pure-module extraction and opponent_slots import; UI agent owns its table import. Re-run the same regressions and the panel cold-import check.
