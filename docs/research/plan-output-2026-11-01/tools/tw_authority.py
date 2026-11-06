"""tw4 plan의 8개 숫자가 각각 실제 명령을 얼마나 움직이나 (11-06 22:00; 'MAP kills the wp4 profile' 10-23: MAP으로 바꾸자 wp4의 v2..v4·κ4가
명령을 전혀 안 움직였음 -> 출력/추종기를 바꾸면 다시 잼). tw_tracking.py의 A/B처럼 중앙선을 3 / 5 m/s로 따라가는 plan을 0.05 s마다 내며 달리다가,
결정마다 그 plan의 숫자 하나씩을 +0.1 움직였을 때 그 순간 MAP/후진 추종이 내는 조향[deg]·속도[m/s] 명령이 얼마나 바뀌는지 (MAP 상태는
복사해서 씀, 차는 원래 plan으로 계속 달림). 인자: 체크포인트(학습 인자만) 차수"""
import os, sys, math, copy, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(__file__))
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim import plan_tw
from f1sim.map_controller import PLAN_WP_SPACING, PLAN_WP_EXTRA_M, grip_index

torch.set_grad_enabled(False)
path, n_env = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
ck = torch.load(path, map_location="cpu", weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, 61, "cuda"
a.plan_format, a.plan_reverse_ext, a.plan_relabel = "tw", 0.0, 0.0
a.prop_fraction, a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop, a.stage = 0.0, 0.0, 0, 0.0, 0.0, "props"
a.episode_s = 3600.0
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None)
lid = env.learner_ids; B = int(lid.numel()); dt = env.sim.control_dt; R = int(a.repeat); tr = env.sim.track
trk = env.tracker
n_tw = int(math.ceil((plan_tw.TW_K * plan_tw.TW_DT * plan_tw.V_FWD + PLAN_WP_EXTRA_M) / PLAN_WP_SPACING)) + 1


def commands(act, mp):
    """(steer, speed) of the learner rows for plan `act` from where they stand, through a copy of MAP `mp`"""
    k_, v_ = plan_tw.decode(act)
    pth, pv, pk, d_, tot = plan_tw.run_path(k_, v_, PLAN_WP_SPACING, n_tw)
    z = torch.zeros(B, device=dev)
    vm = env.sim.state[lid, 3].abs()
    st, _ = mp(torch.zeros(B, 2, device=dev), z, vm, pth, pv, pk, torch.full((B,), PLAN_WP_SPACING, device=dev), z,
               grip_index(env.sim.P, trk.nominal)[lid])
    sp = torch.where(v_[:, 0].abs() < plan_tw.V_STOP, torch.zeros_like(v_[:, 0]), v_[:, 0])
    return st, sp


def ref_center(v, dur, s0, tid):
    t = torch.arange(int(round(dur / dt)) + 1, device=dev, dtype=torch.float32) * dt
    s = torch.remainder(s0[:, None] + v * t[None, :], tr.length[tid][:, None])
    xy, yaw = tr.pose_at_s(s.reshape(-1), tid.repeat_interleave(t.numel()))
    return torch.cat([xy.reshape(B, -1, 2), yaw.reshape(B, -1, 1), torch.full((B, t.numel(), 1), v, device=dev)], -1)


import tw_tracking_core as core
for v_ref in (3.0, 5.0):
    env.reset(seed=61)
    tid = env.sim.tid[lid]
    s0, _, _ = tr.project(env.sim.state[lid, :2], tid)
    xy0, yaw0 = tr.pose_at_s(s0, tid)
    st = env.sim.state.clone(); st[lid, 0], st[lid, 1], st[lid, 2] = xy0[:, 0], xy0[:, 1], yaw0; st[lid, 3:] = 0.0
    env.sim.state.copy_(st)
    ref = ref_center(v_ref, 8.0, s0, tid)
    T = ref.shape[1]
    act_full = torch.zeros(env.B, env.act_dim, device=dev)
    prog = torch.zeros(B, dtype=torch.long, device=dev); win = int(round(0.3 / dt)); ar_ = torch.arange(B, device=dev)
    dst, dsp = [], []
    for k in range(int(5.0 / dt)):
        if k % R == 0:
            idx = (prog[:, None] + torch.arange(win, device=dev)[None, :]).clamp(max=T - 1)
            dist = (ref[ar_[:, None], idx, :2] - env.sim.state[lid, None, :2]).norm(dim=-1)
            prog = idx.gather(1, (dist - 1e-4 * torch.arange(win, device=dev)[None, :]).argmin(1, keepdim=True))[:, 0]  # ties (a stop in the reference) go to the latest
            act = core.plan_from_ref(ref, prog, dt, env.sim.state[lid, :3], dev)
            if k * dt >= 1.5:                                  # up to speed
                st0, sp0 = commands(act, copy.deepcopy(trk.map))
                row_st, row_sp = [], []
                for i in range(act.shape[1]):
                    ai = act.clone(); ai[:, i] = (ai[:, i] + 0.1).clamp(-1, 1)
                    st1, sp1 = commands(ai, copy.deepcopy(trk.map))
                    row_st.append((st1 - st0).abs()); row_sp.append((sp1 - sp0).abs())
                dst.append(torch.stack(row_st, 1)); dsp.append(torch.stack(row_sp, 1))
            act_full[lid] = act
        env.step(act_full)
    ds = torch.cat(dst, 0).mean(0) * 180 / math.pi / 0.1
    dv = torch.cat(dsp, 0).mean(0) / 0.1
    names = [f"곡률{i + 1}" for i in range(plan_tw.TW_K)] + [f"속도{i + 1}" for i in range(plan_tw.TW_K)]
    print(f"중앙선 {v_ref:.0f} m/s, 숫자 1만큼 바꿀 때 명령 변화 (평균): " + " | ".join(
        f"{n} 조향 {s:.2f} deg·속도 {v:.2f} m/s" for n, s, v in zip(names, ds.tolist(), dv.tolist())), flush=True)
