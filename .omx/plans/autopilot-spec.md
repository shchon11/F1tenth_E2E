# Adaptive racing: technical specification

Context: ../context/adaptive-racing-control-20260917.md. User explicitly authorizes design, implementation, training and autonomous validation. Preserve prior dirty edits and weights.

## Requirements

1. One automatic onboard controller, no operator arm selection. D3's 8D action remains six curvature knots and two desired speeds; do not reinterpret existing yielding speeds as aggression fractions.
2. Desired speeds can increase through learning. Physical limits alone cannot decide when an existing low desired speed is tactical, so no blind runtime multiplier.
3. Separate desired/feasible/reachable profiles. Reachable profile begins at measured/delay-predicted state; overspeed is explicit recovery rather than a falsified feasible initial state.
4. Per-spatial-sample/per-MPC-stage budgets, including nominal drivetrain, axle grip, load transfer, motor power, braking, drag and steering slew. Eliminate the whole-path minimum budget from the new auto path; historical arms retain their original math.
5. Causal friction belief uses measured sensor/issued-command histories only. Before informative saturation use nominal; after evidence update/retain belief. Sensor faults are distinct. Do not claim increasing-friction identification from an arbitrary timer reset.
6. Fit and calibrate on actual policy/controller trajectories as well as scripted excitation. Separate episode cohorts before windowing; final evaluation is untouched during selection.
7. Train speed rows first with frozen D3 recurrent geometry representation; enable limited full-policy learning only with a genuine frozen recurrent reference. Preserve initial checkpoint behavior and fresh optimizer semantics.
8. Viewer, evaluator, checkpoint resume and training agree on model/runtime/estimator versions and normalizers. Preserve single-control UI and G-force truth/estimate/applied visualization.

## Chosen architecture and alternatives

Choice A: retain existing action contract, improve local physical planning, finetune desired-speed head then (if justified) full policy. Smallest compatible design, preserves D3 traffic intent. Cost: frozen representation may limit early speed gains.

Rejected for this iteration: reinterpret speeds as physical-envelope fractions. Existing checkpoints do not encode that meaning and would lose yielding behavior. A separately trained pace/yield representation remains a future experiment if fixed-contract adaptation demonstrably saturates.

Deferred: full dynamic tire NMPC. Substantial solver/identification/latency cost; not required to remove current global-budget bottleneck or train speed decisions.

## Source contracts

- `f1sim/f1sim/mpc.py`: decode/reference/ilqr/solve/PlanTracker. Optional new profile and B,N,2,2 bounds preserve old call behavior.
- `learn/grip_control.py`: new auto-only open-path feasible/reachable planning; existing GripSpec modes unchanged for history.
- `learn/grip_runtime.py`: causal history/reset, estimator input normalization, versioned auto installation and physical telemetry.
- `learn/adaptive_grip.py`, `adaptive_grip_train.py`: belief net/consumption, on-policy collection and held-out calibration. Truth is labels/scoring only.
- `learn/ppo.py`, `learn/model.py`, `learn/memory.py`: speed-only row freeze, actual recurrent-reference KL, validated memory+auto+traffic support.
- `learn/evaluate.py`, checkpoint readers and viewer: load the exact runtime pair; no silent estimator substitution or random initial weights.

## Acceptance

Contract tests: tactical slow target preserved, higher friction raises feasible corner speed, future corner does not throttle acceleration everywhere, per-stage tire/actuator bounds, overspeed recovery, eager/graph parity, row-frozen optimizer invariance, recurrent reference uses independent hidden/reset state, no privileged input leakage, checkpoint parity, 2+ cars.

Research success requires measured speed/progress improvement without losing traffic safety/interaction. Development selection and final paired suite are distinct; old417/512 is not a current baseline. Predeclare source/checkpoint/estimator hashes and row sets before learning/evaluation.

Initial training budget: controller/estimator qualification; speed-only PPO256 updates; development evaluation; at most512 full-policy updates with curvature-retention reference if speed-only demonstrates a coherent gain. Adjust wall-clock scheduling based on measured throughput, not metric-driven threshold changes. Every subsequent scientific hypothesis gets a new preregistered cohort; preserve failures.

Final representative validation includes solo/obstacle/traffic, high/low grip, two evaluation seeds, cold starts, sustained response and friction transitions. Threshold matrix to be finalized from acceptance analyst before execution and sealed in experiment manifest. Training-seed scope remains explicit.

## Nonclaims

No guarantee of globally optimal driving, exact plant tire model, unseen full benchmark or real-car validation. Code quality approval is not model promotion. Promote only a model/runtime pair that passes declared held-out comparison and measured latency.

## Critic amendments (sealed before implementation)

1. Local constraints use delay-predicted and actual iLQR rollout speed/steer, not just the slower desired profile. Clamp initial warm controls and each forward rollout control; enforce slew from previous measured/predicted steer. Re-evaluate longitudinal residual grip at the stage's selected steering. If entering a corner above its feasible speed, preserve initial speed and expose recovery/infeasibility; do not claim the state already satisfies the envelope.
2. Auto installs an explicit acceleration-to-command conversion in PlanTracker, while historical conversion stays unchanged. Map bounded net acceleration plus modeled rolling/aero resistance through nominal motor_tau and measured wheel-feedback speed; calibration remains applied once by the existing outer command path. A tactical speed cap is a trajectory target, not an instantaneous final command clamp during overspeed recovery. Limit the nominal motor request and physical actuator target; flag situations where target-range saturation makes the requested recovery itself infeasible. Test solver acceleration AND reconstructed nominal VESC request after final conversion, at low/high speed and overspeed. This is nominal-command feasibility, not a guarantee under randomized actuator error.
3. Training init/resume explicitly covers load_for_memory, load_for_conditioning and load_checkpoint. Evaluate and viewer must validate controller runtime version/spec + estimator SHA/embedded weights before allowing a nonlegacy policy; strict actor shapes remain mandatory. Save the original D3 reference path/hash/metadata and keep it across resumed stages; never replace it with the resumed candidate. Recurrent reference owns hidden/reset state and rollout mean/std targets independent of learner hidden state.
