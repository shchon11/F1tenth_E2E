"""A v2 checkpoint for the driving-path tests (`test_v2_graph_parity`, `test_v2_console`).

Torch and f1sim only -- no ROS -- so a console test can build one on a machine without rclpy.
"""
from __future__ import annotations

from typing import Optional

import torch

#: The cap the generated checkpoint records (`extra["cap"]`, the console's default speed cap).
CAP = 6.0


def generated_v2_checkpoint(path: str, ego_actions: Optional[bool] = None) -> str:
    """A v2 actor-critic at the deployed 1081 beams, seeded inside a forked RNG so the suite's global
    generator is left as it was. The action head's last layer is put back to an untrained network's
    scale (B6 builds it x 0.1, which leaves every plan near zero) so the plans visibly move.
    `ego_actions` None: the contract's default (`learn.obs.EGO_ACTIONS_DEFAULT` -- without the issued
    actions since 2026-10-15, 42 ego columns); True / False names the contract explicitly: with the
    issued actions (210 columns, every v2 checkpoint before 2026-10-15) or without them (the copycat
    study's arm A)."""
    from f1sim.learn.model import save_checkpoint
    from f1sim.learn.model_v2 import ActorCriticV2
    from f1sim.learn.obs import ObsSpecV2
    spec = ObsSpecV2() if ego_actions is None else ObsSpecV2(ego_actions=bool(ego_actions))
    with torch.random.fork_rng():
        torch.manual_seed(20261014)
        m = ActorCriticV2(n_beams=1081, ego_dim=spec.ego_dim, obs=dict(vars(spec)))
        with torch.no_grad():
            m.actor.mu.weight.mul_(10.0)
    save_checkpoint(path, m, {"spec": dict(vars(spec)), "phase": "dagger", "arch": "v2", "run": "tiny_v2",
                              "iter": 0, "iters": 1, "speed_mode": "wp4", "speed_command": "tracker", "cap": CAP})
    return path
