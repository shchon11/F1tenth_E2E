# Adaptive racing scoped cleanup plan

## Scope

Review only current-task Python sections in:

- `f1sim/f1sim/learn/policy_adaptation.py`
- `f1sim/f1sim/learn/policy_grip_data.py`
- `f1sim/f1sim/learn/grip_runtime.py`
- `f1sim/f1sim/learn/grip_control.py`
- `f1sim/f1sim/lidar.py`

The intake source is `/home/shchon11/Documents/Codex/2026-09-17/new-chat/work/adaptive-racing-v3/intake-source/f1sim`.
No other files may be edited.

## Behavior lock and order

1. Run the narrow existing CPU regression set covering adaptive grip, policy grip data,
   runtime/checkpoint contracts, local speed/control physics, grip overlays, and car mesh
   LiDAR behavior with the requested single-thread CPU environment.
2. Review the scoped diff and new files for dead code, duplicate logic, needless wrappers,
   and naming/error-handling noise. Preserve all formulas, tensor shapes/devices/dtypes,
   checkpoint metadata and digest semantics, historical compatibility paths, and LiDAR
   sentinel/chunk behavior.
3. Apply only a clearly behavior-preserving simplification, one smell category at a time;
   skip files where no meaningful simplification is demonstrated.
4. Re-run the narrow regression set and syntax/compile diagnostics for every edited file.

## Review result

- Dead code: removed the unused `dataclasses.asdict` import from `policy_grip_data.py`.
- Duplication, needless abstraction, and boundary changes: no unambiguous behavior-preserving
  cleanup identified in the five files; intentionally left unchanged because their repeated
  branches and wrappers carry checkpoint, graph, tensor, or historical compatibility contracts.

## Acceptance criteria

- No behavior, physics, checkpoint integrity, model, safety, or historical-parity changes.
- No new dependencies, tests, abstractions, or style-only churn.
- Diff remains confined to the five listed files plus this plan.
- If no meaningful simplification is found, leave the five files unchanged and record the
  reviewed evidence in the final report.
