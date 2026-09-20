# Implementation plan

## Phase1: design review

Read autopilot-spec.md and context. Critic reviews executable interfaces, scope and acceptance; resolve substantive gaps before source edits. Save final preregistration before training. Analyst criteria will be incorporated without user interruption.

## Phase2: independent source lanes

1. Controller executor owns mpc.py, grip_control.py, new controller-specific tests. Add explicit reachable profiles/local budgets and optional per-step iLQR bounds; keep old arms numerically unchanged. Pin nominal physical spec. Validate independently on short synthetic paths and CUDA parity after CPU unit checks.
2. Training executor owns ppo.py/model.py/memory.py and recurrent-training tests. Add fresh-optimizer speed-row-only mode, exact frozen parameter invariance, independent recurrent-reference rollout targets/curvature retention, and validate then permit memory+auto+multi-car. Do not launch real training before integration approval.
3. Estimator executor owns adaptive_grip.py/adaptive_grip_train.py plus estimator tests. Add a reusable on-policy collection route, policy/controller/episode cohort manifests, calibration/held-out source separation and diagnosable belief consumption. Investigate double conservatism and self-trapping via development experiments, not final threshold relaxation. Preserve V2A/V2B artifacts.
4. Parent owns runtime wiring, checkpoint/evaluate/viewer compatibility, frozen source snapshot, resumable experiment launcher, metrics/plots and evaluation. All worker interfaces agreed before overlapping runtime changes. No UI option proliferation.

Mandatory seam details: controller lane owns PlanTracker auto command conversion and state-dependent bound projection (including warm start). Training lane owns all3PPO init/resume loaders plus preserved recurrent-reference identity. Parent owns evaluate/viewer loading and runtime metadata validation shared helper. Auto can be evaluated/trained with an unapproved estimator only through an explicit research capability in the experiment harness, never by mutating approval metadata or silently disabling the production loader gate.

## Phase3: integrate and test

- Unit and contract checks, historical-arm parity, simulator observation invariance.
- Short real GPU training smoke with GRU+auto+traffic and checkpoint resume, plus device/parity checks.
- Freeze a separate runtime worktree/snapshot including the approved dirty source; no code mutation in an active training/evaluation tree. Keep datasets/checkpoints/reports outside source snapshot.
- Measure throughput/memory before committing staged update budgets.

## Phase4: learning and judgment

- Paired frozen-D3 controller diagnostics identify controller effect separately from estimator.
- Collect policy-domain estimator data with train/cal/dev/test episode separation; fit/calibrate frozen observer and validate ordered streams.
- Speed-only stage256updates with fresh optimizer; select only from preregistered development checkpoints. If approved, limited full-policy stage512updates with independent recurrent-reference curvature retention.
- Freeze candidate, run final representative solo/obstacle/traffic suite on two eval seeds alongside current-source controls. Produce raw-cell report and gate decision. No final-test tuning.
- Code approved by architecture, security and quality reviewers. Model promotion additionally requires empirical gates. Any new iteration after failure is separately preregistered, never an altered verdict for a prior run.

## Completion

Deliver runnable training command, resume artifacts, source/model/estimator/runtime pins, quantitative comparison and remaining limits. Retain best previous model unless new pair passes. Update single automatic viewer and diagnostics only for the approved pair. Preserve user processes; terminate only owned runs. Clear autopilot mode state on completion.
