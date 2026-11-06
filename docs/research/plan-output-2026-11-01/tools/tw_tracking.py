"""시간별 waypoint plan(tw4)을 제어기가 얼마나 정확히 따라가나 (11-06 22:00, 사용자: "이대로 ㄱㄱ" -- 학습 전에 전진·후진·후진->전진
경로의 거리 오차부터). 학습 env(학습 인자, plan 형식 tw4, 상자·상대 없음, 차 1대)에서 정해 둔 기준 궤적(시간별 자세)을 짜 놓고,
0.05 s마다 차의 지금 자세에서 다음 1.5 s의 기준점 4개(0.375 s 간격)를 원호 4개로 이어 tw4 plan을 만들어 넘김 -- policy가 낼 plan과
같은 형식, 같은 주기. 잣대: 차 위치에서 기준 경로까지 가장 가까운 거리(시작 0.5 s 뒤부터) RMSE / 최대, 끝 위치 오차, 벽 접촉.
시나리오:
  A 중앙선 3 m/s 8 s      B 중앙선 5 m/s 6 s
  C 후진 1 m(1 m/s) -> 왼쪽으로 크게 꺾어 전진 2 m(1.5 m/s)       (상자 앞에서 빠져나오기)
  D 오른쪽으로 꺾으며 후진 1.2 m(1.5 m/s) -> 왼쪽으로 꺾어 전진 2.5 m(2 m/s)
인자: 체크포인트(학습 인자만 씀) 시나리오별 차 수"""
import os, sys, math, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim import plan_tw
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

torch.set_grad_enabled(False)
path, n_env = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
ck = torch.load(path, map_location="cpu", weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, 51, "cuda"
a.plan_format, a.plan_reverse_ext, a.plan_relabel = "tw", 0.0, 0.0
a.prop_fraction, a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop, a.stage = 0.0, 0.0, 0, 0.0, 0.0, "props"
a.episode_s = 3600.0
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None)
assert env.ecfg.learner_plan == "tw", env.ecfg.learner_plan
lid = env.learner_ids; B = int(lid.numel()); dt = env.sim.control_dt; R = int(a.repeat)
i_w = CC.index("hard_wall") if "hard_wall" in CC else None
tr = env.sim.track
ONSET_COLS = [CC.index("onsets_wall"), CC.index("onsets_prop"), CC.index("onsets_car")]


def ref_pieces(pieces, x0, y0, p0, step):
    """world (T, 4) x, y, heading, signed speed every `step` s of a piecewise (duration, v, kappa) drive from (x0, y0, p0)"""
    out = []
    x, y, p = x0.clone(), y0.clone(), p0.clone()
    for dur, v, k in pieces:
        for _ in range(int(round(dur / step))):
            out.append(torch.stack([x, y, p, torch.full_like(x, v)], 1))
            x, y, p = plan_tw._arc(x, y, p, torch.full_like(x, k), torch.full_like(x, v * step))
    out.append(torch.stack([x, y, p, torch.zeros_like(x)], 1))
    return torch.stack(out, 1)                                  # (B, T, 4)


def ref_center(v, dur, s0, tid, step):
    t = torch.arange(int(round(dur / step)) + 1, device=dev, dtype=torch.float32) * step
    s = torch.remainder(s0[:, None] + v * t[None, :], tr.length[tid][:, None])
    xy, yaw = tr.pose_at_s(s.reshape(-1), tid.repeat_interleave(t.numel()))
    xy = xy.reshape(s0.shape[0], -1, 2); yaw = yaw.reshape(s0.shape[0], -1)
    return torch.cat([xy, yaw[..., None], torch.full_like(yaw, v)[..., None]], -1)


def plan_from_ref(ref, prog, step, pose):
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


def open_start(tid, n_try=256):
    """per car, the pose with the most room among n_try random spots of its track (lane heading)"""
    L = tr.length[tid]
    best_xy = torch.zeros(B, 2, device=dev); best_yaw = torch.zeros(B, device=dev); best_e = torch.full((B,), -1.0, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(7)
    for _ in range(n_try):
        s = torch.rand(B, device=dev, generator=g) * L
        xy, yaw = tr.pose_at_s(s, tid)
        lat = (torch.rand(B, device=dev, generator=g) - 0.5) * 2.0
        xy = xy + torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1) * lat[:, None]
        e = tr.sample_edt(xy, tid)
        m = e > best_e
        best_xy = torch.where(m[:, None], xy, best_xy); best_yaw = torch.where(m, yaw, best_yaw); best_e = torch.where(m, e, best_e)
    return best_xy, best_yaw, best_e


def run(name, make_ref, dur, open_space=False, quiet=False):
    env.reset(seed=51)
    st0 = env.sim.state[lid].clone()
    tid = env.sim.tid[lid]
    s0, _, _ = tr.project(st0[:, :2], tid)
    if open_space:
        xy0, yaw0, room = open_start(tid)
    else:
        xy0, yaw0 = tr.pose_at_s(s0, tid)                     # start on the centreline, along it, standing
        room = tr.sample_edt(xy0, tid)
    st = env.sim.state.clone()
    st[lid, 0], st[lid, 1], st[lid, 2] = xy0[:, 0], xy0[:, 1], yaw0
    st[lid, 3:] = 0.0
    env.sim.state.copy_(st)
    ref = make_ref(xy0[:, 0], xy0[:, 1], yaw0, s0, tid)
    T = ref.shape[1]
    act_full = torch.zeros(env.B, env.act_dim, device=dev)
    pos, wall = [], torch.zeros(B, dtype=torch.bool, device=dev)
    cf0 = env.sim.contact_frame()[lid].float().clone()
    prog = torch.zeros(B, dtype=torch.long, device=dev)
    win = int(round(0.3 / dt))
    ar_ = torch.arange(B, device=dev)
    act = None
    for k in range(int(dur / dt)):
        if k % R == 0:
            idx = (prog[:, None] + torch.arange(win, device=dev)[None, :]).clamp(max=T - 1)
            dist = (ref[ar_[:, None], idx, :2] - env.sim.state[lid, None, :2]).norm(dim=-1)
            prog = idx.gather(1, (dist - 1e-4 * torch.arange(win, device=dev)[None, :]).argmin(1, keepdim=True))[:, 0]  # ties (a stop in the reference) go to the latest
            act = plan_from_ref(ref, prog, dt, env.sim.state[lid, :3])
            act_full[lid] = act
        env.step(act_full)
        pos.append(env.sim.state[lid, :2].clone())
        if os.environ.get("TW_DEBUG") and k % 20 == 0:
            kk_, vv_ = plan_tw.decode(act)
            print(f"  {name[:1]} t={k * dt:.2f} 속도 {float(env.sim.state[lid, 3].mean()):.2f} 계획 v1 {float(vv_[:, 0].mean()):.2f} "
                  f"명령 {float(env.last_cmd_raw[lid, 1].mean()):.2f} 조향 {float(env.last_cmd_raw[lid, 0].mean()):.3f} "
                  f"진행 {float(prog.float().mean() * dt):.2f} s", flush=True)
        cf = env.sim.contact_frame()[lid].float()
        wall |= ((cf - cf0)[:, ONSET_COLS] > 0).any(1)
    P = torch.stack(pos, 1)                                    # (B, T, 2)
    dist_all = (P[:, :, None, :] - ref[:, None, :, :2]).norm(dim=-1)
    d, inear = dist_all.min(-1)                                           # cross-track: nearest reference point
    back_ph = ref[torch.arange(B, device=dev)[:, None], inear, 3] < 0      # the nearest reference point is a backing one
    k0 = int(0.5 / dt)
    ok = ~wall
    dd = d[ok][:, k0:]
    done = (prog.float() * dt)[ok]
    bp = back_ph[ok][:, k0:]
    out = dict(n_wall=int(wall.sum()), done=float(done.median()) if done.numel() else 0.0,
               rmse=float(dd.square().mean().sqrt()) if dd.numel() else float("nan"),
               rmse_back=float(dd[bp].square().mean().sqrt()) if bool(bp.any()) else float("nan"),
               rmse_fwd=float(dd[~bp].square().mean().sqrt()) if bool((~bp).any()) else float("nan"))
    if quiet:
        return out
    if bool(bp.any()) and bool((~bp).any()):
        print(f"  {name}: 후진 구간 RMSE {float(dd[bp].square().mean().sqrt()) * 100:.1f} cm (최대 {float(dd[bp].max()) * 100:.1f}), "
              f"전진 구간 RMSE {float(dd[~bp].square().mean().sqrt()) * 100:.1f} cm (최대 {float(dd[~bp].max()) * 100:.1f})", flush=True)
    if dd.numel() == 0:
        print(f"{name}: 모든 차가 벽에 닿음", flush=True); return out
    print(f"{name}: 차 {int(ok.sum())}/{B}대 (벽 접촉 {int(wall.sum())}대 뺌, 출발 자리 여유 중앙 {float(room.median()):.2f} m) | "
          f"기준 경로와 거리 RMSE {float(dd.square().mean().sqrt()) * 100:.1f} cm, p95 {float(dd.flatten().quantile(0.95)) * 100:.1f} cm, "
          f"최대 {float(dd.max()) * 100:.1f} cm | {dur:.1f} s 동안 기준 궤적 {float(done.median()):.2f} s 분량 진행", flush=True)
    return out


step = dt
SCENARIOS = [
    ("A 중앙선 3 m/s", lambda x, y, p, s, t: ref_center(3.0, 10.0, s, t, step), 7.0, False),
    ("B 중앙선 4 m/s", lambda x, y, p, s, t: ref_center(4.0, 8.0, s, t, step), 5.0, False),
    ("C 후진 1 m -> 왼쪽 꺾어 전진", lambda x, y, p, s, t: ref_pieces([(1.0, -1.0, 0.0), (0.2, 0.0, 0.0), (1.4, 1.5, 1.2), (1.5, 0.0, 0.0)], x, y, p, step), 5.0, True),
    ("D 오른쪽 꺾어 후진 -> 왼쪽 꺾어 전진", lambda x, y, p, s, t: ref_pieces([(0.8, -1.5, -0.8), (0.2, 0.0, 0.0), (1.25, 2.0, 1.0), (1.5, 0.0, 0.0)], x, y, p, step), 5.0, True),
]
if __name__ == "__main__":
    for nm, mk, du, op in SCENARIOS:
        run(nm, mk, du, op)
