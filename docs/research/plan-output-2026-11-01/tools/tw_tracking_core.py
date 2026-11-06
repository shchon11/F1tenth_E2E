"""tw_tracking.py의 기준 궤적 -> tw4 plan 함수 (tw_authority.py가 같이 씀)"""
import torch
from f1sim import plan_tw


def plan_from_ref(ref, prog, step, pose, dev):
    """tw4 action from the car's pose along the reference from its progress point `prog` (B,) on: the reference points
    DT, .., K DT later joined by arcs -- following the path at the reference's speed, no catching up on time lost"""
    T = ref.shape[1]
    x, y, p = pose[:, 0].clone(), pose[:, 1].clone(), pose[:, 2].clone()
    ar_ = torch.arange(ref.shape[0], device=dev)
    ks, vs = [], []
    for k in range(1, plan_tw.TW_K + 1):
        i_t = (prog + int(round(k * plan_tw.TW_DT / step))).clamp(max=T - 1)
        i_m = (prog + int(round((k - 0.5) * plan_tw.TW_DT / step))).clamp(max=T - 1)
        tx, ty = ref[ar_, i_t, 0], ref[ar_, i_t, 1]
        dirn = torch.sign(ref[ar_, i_m, 3])
        dx, dy = tx - x, ty - y
        c_, s_ = torch.cos(p), torch.sin(p)
        rx, ry = c_ * dx + s_ * dy, -s_ * dx + c_ * dy
        d2 = (rx * rx + ry * ry).clamp_min(1e-9)
        kap = (2 * ry / d2).clamp(-plan_tw.KAPPA_MAX, plan_tw.KAPPA_MAX)
        ch = d2.sqrt()
        arg = (kap * ch / 2).clamp(-1, 1)
        sabs = torch.where(kap.abs() * ch < 1e-4, ch, 2 * torch.asin(arg.abs()) / kap.abs().clamp_min(1e-9))
        sabs = torch.where(d2 < 1e-6, torch.zeros_like(sabs), sabs)
        # the reference's own speed (not distance / DT: with 0.1 s segments a car 0.5 m off the reference got a 5-7 m/s
        # first segment to catch up -- the test measured that planner, not the follower); the curvature points at the
        # reference point, the next plan corrects the rest
        v = (dirn * ref[ar_, i_m, 3].abs()).clamp(-plan_tw.V_REV, plan_tw.V_FWD)
        ks.append(kap); vs.append(v)
        x, y, p = plan_tw._arc(x, y, p, kap, v * plan_tw.TW_DT)
    return plan_tw.encode(torch.stack(ks, 1), torch.stack(vs, 1))


