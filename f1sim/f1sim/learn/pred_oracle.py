"""(11-25) plan_rl --pred-clear-oracle: the ORACLE gate (agreed with the user).

The question: probes showed that the policy's own input (polar scan + GRU), trained supervised, predicts the time-aligned
clearance between our plan point at t_k and the other car's body at t_k to 6.7 cm median (96 % right on overlap vs clear below
0.5 m; `out/analysis/lidar_repr/`, the `car_*` results), yet race33b's plans overlap a 0.6x car's real future ~13 % of the time
(u975-u1075, exam4). The information is perceivable; RL does not turn it into plans. The gate: GIVEN near-true time-aligned
clearances as the trunk's --pred-clear input, does PPO learn to plan around where the car WILL be? Pass: the 0.6x overlap
share clearly below ~13 % with passes >= ~1.5. Fail: a learned clearance head is not worth building.

Near-true: the other car is a frozen self-copy (`plan_rl.SelfOpponents`) that drives its own plan exactly, so the plan it drew
this decision is where it will be (0.07 m off at 0.5 s, 0.27 m at 1 s against the realised future). A copy at pace f plans as
if at its own pace and drives the plan's speeds x f: t s on it is at the plan's pose at f t (exam_record.py's
F1SIM_EXAM_PRED_ORACLE does the same). Everything else is `Actor2.clear_in`'s: our car where the PREVIOUS plan puts it
(`PlanObs2.prev_poses`), the plan-car cost's 3 discs (`plan_car_cost.car_cover` / `clearance`), 'there' = within --pred-range
and hit by a LiDAR beam within the last 1 s (the head's target rule) as P = 1 / 0, `PC_FAR` when not there or without a
previous plan, clipped and / PC_SCALE. Used by the trainer and by the exam (F1SIM_EXAM_PRED_CLEAR_ORACLE), the same code."""
from typing import Dict, Tuple

import torch

from .. import plan_steps as PS
from . import plan_car_cost as PCC
from .plan_policy import PC_N, PC_FAR, PC_LO, PC_HI, PC_SCALE, PC_PMIN, plan_world


def copy_plan_future(p: Dict[str, torch.Tensor], st_plan: torch.Tensor, pace: torch.Tensor, n: int = PC_N,
                     sub: int = PS.SUB) -> Tuple[torch.Tensor, torch.Tensor]:
    """where copies that drew plans `p` (`plan_steps.roll`: dense x, y, psi (m, N SUB + 1) in the frame of `st_plan` (m, >= 3),
    index 0 = now) will be at t_k = 0.1 k s, k = 1..n, driving them at pace `pace` (m,): the plan's pose at f t_k, linear
    between the dense poses -> (world xy (m, n, 2), world heading (m, n))"""
    xw, yw, pw = plan_world({k: p[k].float() for k in ("x", "y", "psi")}, st_plan.float())
    kk = torch.arange(1, n + 1, device=xw.device, dtype=torch.float32) * float(sub)
    tau = pace.float()[:, None] * kk[None, :]
    i0 = tau.floor().long().clamp(0, xw.shape[1] - 2)
    w = (tau - i0.float()).clamp(0.0, 1.0)
    lerp = lambda a: a.gather(1, i0) * (1.0 - w) + a.gather(1, i0 + 1) * w
    return torch.stack([lerp(xw), lerp(yw)], -1), lerp(pw)


def to_frame(xy: torch.Tensor, psi: torch.Tensor, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """world points (m, n, 2) and headings (m, n) -> the frame of `pose` (m, >= 3: x, y, yaw) -- our car at the decision, the
    frame the head's prediction and the previous plan's poses are in"""
    pose = pose.float()
    dx, dy = xy[..., 0] - pose[:, 0:1], xy[..., 1] - pose[:, 1:2]
    c, s = torch.cos(pose[:, 2:3]), torch.sin(pose[:, 2:3])
    return torch.stack([dx * c + dy * s, -dx * s + dy * c], -1), psi - pose[:, 2:3]


def oracle_clear(xy: torch.Tensor, psi: torch.Tensor, there: torch.Tensor, prev, geom) -> torch.Tensor:
    """`Actor2.clear_in` with the other car's path GIVEN: xy (m, PC_N, 2) / psi (m, PC_N) in our frame at t_1..t_PC_N,
    there (m,) bool (P = 1 / 0), prev = (the previous plan's poses (m, PC_N, 3), there is one (m,) bool), geom = (car length,
    width, margin) [m] -> (m, PC_N), the trunk's scale (/ PC_SCALE)"""
    poses, ok = prev
    L, W, mg = (float(x) for x in geom)
    elem = PCC.car_cover(torch, xy.float()[:, None], psi.float()[:, None], L, W, conf=there.float()[:, None])
    c = PCC.clearance(torch, poses[..., :2].float(), poses[..., 2].float(), *elem, L, W, mg, far=PC_FAR, lo=PC_LO, hi=PC_HI,
                      w_min=PC_PMIN)
    c = torch.where(ok[:, None], c, torch.full_like(c, PC_FAR))
    return c / PC_SCALE


def copy_oracle(opp, oi: torch.Tensor, st_me: torch.Tensor, there: torch.Tensor, absent: torch.Tensor, prev, geom):
    """the --pred-clear-oracle input of learner rows whose other car is row `oi` (n,) of `SelfOpponents` `opp`, its `plan`
    drawn this decision (before ours: a copy does not read our plan). st_me (n, >= 3) our cars now; there (n,) bool the 'there'
    rule; absent (n,) bool the other car out of the race. -> (values (n, PC_N), use (n,) bool): use where a copy drives the
    other car (its plan is the truth) or there is no other car (far); elsewhere (a scripted driver at the wheel: no plan to
    read) the caller keeps the head's own clearances"""
    sub = {k: opp.p[k][oi] for k in ("x", "y", "psi")}
    xy_w, psi_w = copy_plan_future(sub, opp.st_plan[oi], opp.pace[oi])
    xy, psi = to_frame(xy_w, psi_w, st_me)
    live = opp.live[oi] if getattr(opp, "live", None) is not None else opp.on[oi]
    return oracle_clear(xy, psi, there & live & ~absent, prev, geom), live | absent
