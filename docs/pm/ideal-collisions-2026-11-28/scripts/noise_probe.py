"""학습 때와 같은 탐색 잡음으로 몰면서(학습 없음) 결정마다 잡음을 남김 -> 한 결정의 조향 잡음이 상자 통과 여유를 얼마나 바꾸나 (H1).
exam_record.py를 수정 없이 runpy로 부름: F1SIM_EXAM_SAMPLE=1, F1SIM_EXAM_NOISE_RHO=0 (매 결정 새 잡음)으로 부르고, 여기서 plan_v2에 들어오는
잡음을 학습(plan_rl, noise_corr_box 1.5,4.0 / noise_corr 0.9)처럼 바꿈: 상자가 차 앞 1.5-4 m, 옆 1 m 안이면 첫 단계 조향 잡음을 AR(1) 0.9로
이어 감, 그 밖은 새로 뽑음. 속도 잡음은 늘 새로. (학습의 판정은 트랙을 따라 잰 상자 칸이고 여기서는 차 틀의 직선 거리: 근사.)
남김(NP_OUT npz, (T, B_l)): e0 = 첫 단계 조향 잡음(표준화), e10 = 속도 잡음, sd0 / sd10 = 그 표준편차(pre-tanh), m0 / m10 = 평균 행동(pre-tanh),
zone = 이어 간 결정. 인자는 exam_record.py 그대로."""
import os, sys, runpy, atexit
TOOLS = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools"
sys.path.insert(0, TOOLS)
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
import numpy as np
import torch
import f1sim.learn.plan_policy as PP
from f1sim import plan_steps as PS

OUT = os.environ["NP_OUT"]
REC = {k: [] for k in ("e0", "e10", "sd0", "sd10", "m0", "m10", "zone")}
_orig = PP.plan_v2
_S = {}


def plan_v2(actor, obs, lim, sample=False, eps=None, cap=None, intent_fix=None):
    if sample and eps is not None:
        env, lid = obs.env, obs.lid
        po = env.procedural
        st = env.sim.state[lid].float()
        zone = torch.zeros(lid.numel(), dtype=torch.bool, device=st.device)
        if po is not None:
            pos = po.p_poses[lid][..., :2].float(); live = po.p_sid[lid] >= 0
            dx, dy = pos[..., 0] - st[:, 0:1], pos[..., 1] - st[:, 1:2]
            c, s = torch.cos(st[:, 2:3]), torch.sin(st[:, 2:3])
            x, y = dx * c + dy * s, -dx * s + dy * c
            zone = (live & (x >= 1.5) & (x <= 4.0) & (y.abs() <= 1.0)).any(1)
        prev = _S.get("e0")
        e0 = eps[:, 0].clone()
        if prev is not None and prev.shape == e0.shape:
            e0 = torch.where(zone, 0.9 * prev + (1 - 0.81) ** 0.5 * eps[:, 0], eps[:, 0])
        eps = eps.clone(); eps[:, 0] = e0
        _S["e0"] = e0
    out = _orig(actor, obs, lim, sample, eps, cap, intent_fix)
    if sample and eps is not None:
        u, mean, log_std, v_hat, p = out
        sd = (u - mean) / torch.where(eps.abs() > 1e-6, eps, torch.ones_like(eps))
        REC["e0"].append(eps[:, 0].cpu()); REC["e10"].append(eps[:, PS.N_STEP].cpu())
        REC["sd0"].append(sd[:, 0].cpu()); REC["sd10"].append(sd[:, PS.N_STEP].cpu())
        REC["m0"].append(mean[:, 0].cpu()); REC["m10"].append(mean[:, PS.N_STEP].cpu()); REC["zone"].append(zone.float().cpu())
    return out


PP.plan_v2 = plan_v2


@atexit.register
def _save():
    if REC["e0"]:
        np.savez_compressed(OUT, **{k: torch.stack(v).numpy().astype(np.float32) for k, v in REC.items()})
        print(f"[noise] saved {OUT} ({len(REC['e0'])} decisions)", flush=True)


sys.argv = [os.path.join(TOOLS, "exam_record.py")] + sys.argv[1:]
runpy.run_path(os.path.join(TOOLS, "exam_record.py"), run_name="__main__")
