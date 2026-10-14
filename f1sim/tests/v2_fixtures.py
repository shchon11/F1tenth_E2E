"""A v2 checkpoint for the driving-path tests (`test_v2_graph_parity`, `test_v2_console`).

Torch and f1sim only -- no ROS -- so a console test can build one on a machine without rclpy.
"""
from __future__ import annotations

import torch

#: The cap the generated checkpoint records (`extra["cap"]`, the console's default speed cap).
CAP = 6.0


def generated_v2_checkpoint(path: str) -> str:
    """A v2 actor-critic at the deployed 1081 beams, seeded inside a forked RNG so the suite's global
    generator is left as it was. The action head's last layer is put back to an untrained network's
    scale (B6 builds it x 0.1, which leaves every plan near zero) so the plans visibly move."""
    from f1sim.learn.model import save_checkpoint
    from f1sim.learn.model_v2 import ActorCriticV2
    from f1sim.learn.obs import ObsSpecV2
    with torch.random.fork_rng():
        torch.manual_seed(20261014)
        m = ActorCriticV2(n_beams=1081)
        with torch.no_grad():
            m.actor.mu.weight.mul_(10.0)
    save_checkpoint(path, m, {"spec": dict(vars(ObsSpecV2())), "phase": "dagger", "arch": "v2", "run": "tiny_v2",
                              "iter": 0, "iters": 1, "speed_mode": "wp4", "speed_command": "tracker", "cap": CAP})
    return path
