# Adaptive racing control context

Task: The user authorizes a proper redesign, implementation and training without repeatedly asking permission. Prior objective: retain D3 opponent response, drive faster by exploiting available friction, remove operator controller presets, visualize GT/estimated/applied limits.

Desired outcome: one usable sensor-based runtime in which path/tactical intent, a reachable speed plan, learned friction belief and feedback tracking cooperate. Speed may rise when feasible; unknown friction follows a nominal profile until informative saturation evidence. Preserve genuine slowing for traffic/obstacles.

Current evidence:
- D3 checkpoint: /home/shchon11/f1sim_runs/cl_it_lidar_ppo_s701/ppo_u500.pt (legacy-trained GRU128, six curvature knots plus two speed endpoints). Historical suite417/512 is a prior source cohort, not a current-head score.
- auto currently uses old estimator_seed401 + corrected nominal4WD torque share; it only clips policy speeds and uses the worst whole-path longitudinal bound everywhere.
- adaptive_grip_v2 GRU64 has causal40x11 sensor/previous-command features, ordered quantiles, learned saturation-confidence, nominal-before-evidence, hold-after-evidence. V2A failed offline; V2B data scaling passed offline6/6 (muMAE.07085, q10 exceed6.29%, AUROC.9523).
- V2B failed actual D3-loop qualification: all4 comparisons slower, worst -10.49% progress at high-muICRA, zero collisions. Appliedmu collapsed to~.49/.52 onICRA despite true .734/1.154. Candidate unapproved. Artifacts under /home/shchon11/Documents/Codex/2026-09-17/new-chat/work/adaptive-grip/.
- Existing PPO refuses GRU+nonlegacy and nonlegacy+multi-car, due prior unvalidated combination; user now explicitly authorizes design and validation of these combinations.
- Viewer/lidar/teacher performance work and UI/raceline changes from prior turn are uncommitted and must be preserved.399 tests passed. Candidate model not installed.
- GPU RTX4060Ti8GB currently mostly free; use bounded runs, preserve old weights, stop only owned processes. No full training process active at intake.

Constraints:
- No simulator mu/state/map truth in deployed actor/estimator/controller inputs. Truth permitted only teacher labels, offline supervision and scoring.
- Do not equate elastic slip with saturation. Distinguish actuator lag, motor cap, sensor bias/tilt and limited friction.
- Fix architecture/observation/training mismatch rather than post-hoc threshold relaxation.
- New experiment cohorts and preregistration before outcomes; freeze test criteria and never tune on final held-out cases.
- Preserve previous dirty edits, historical arms and source/checkpoint/estimator pins; avoid new user knobs/dependencies.
- User grants training and native agents; simple work may use Luna xhigh. No need to ask routine permission.

Unknowns:
- Does V2B overconservatism arise mainly from lower-quantile consumption, false event admission, policy-state distribution shift, or globally-minimized MPC budgets?
- Smallest speed-increasing design that preserves traffic intent: learned speed-head finetuning vs separate feasible speed planner with explicit tactical cap.
- Required on-policy coverage and calibration for estimator; how to learn recovery/increased friction without forgetting real low-grip evidence.

Touchpoints: learn/grip_control.py, grip_runtime.py, adaptive_grip.py/train.py, ppo.py, model.py, mpc.py, clearance.py, graph_runtime.py, evaluate.py, viewer/sim_worker.py and unified console/visualization. Existing references: ForzaETH/TUM/UNICORN speed planning and friction-identifiability sources in outputs/f1tenth-update.md of current chat.
