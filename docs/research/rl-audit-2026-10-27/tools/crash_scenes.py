"""충돌 장면을 위에서 본 그림으로 (rl-audit 2026-10-29, 사용자: "사고 나는 장면들을 보면서 하면 원인분석이 더 쉽지 않을까?").

참 상태 정책(state_rl 체크포인트)의 평균 행동으로 단계의 세계를 몰고, 학습 행에 강한 충돌(>= 1.5 m/s: 벽, 상자, 차)이
날 때마다 직전 2초를 한 장에 그린다: 벽(검정), 상자(주황), 중심선(회색 점선), 내 차 궤적(속도 색), 충돌 순간의 차(빨강),
상대 차와 그 궤적(파랑), 제목에 종류·충격 속도·속도·직전 0.5초의 조향/속도 명령. 그림은 OUT/scene_NNN.png, 목록은
OUT/events.json. 사용: python crash_scenes.py CKPT --stage props [--envs 64] [--steps 4000] [--max-images 30] [--out DIR]"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from f1sim.learn import state_rl  # noqa: E402
from f1sim.learn.state_rl import ActorCritic, RunningNorm, StateObs, Unwedge  # noqa: E402

H = 80          # 2 s of history at 40 Hz


def draw(path, ev, track, props_in, opp_hist, title, plan_xy=None, my_paths=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK JP"        # 한글이 들어 있는 CJK 글꼴
    plt.rcParams["axes.unicode_minus"] = False
    x0, y0 = ev["xy"]
    W = 8.0
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    occ = track.occupancy; res = track.resolution; ox, oy = track.origin
    c0, c1 = int((x0 - W - ox) / res), int((x0 + W - ox) / res)
    r0, r1 = int((y0 - W - oy) / res), int((y0 + W - oy) / res)
    c0, r0 = max(c0, 0), max(r0, 0); c1, r1 = min(c1, occ.shape[1]), min(r1, occ.shape[0])
    sub = occ[r0:r1, c0:c1]
    ax.imshow(sub, origin="lower", cmap="Greys", extent=(ox + c0 * res, ox + c1 * res, oy + r0 * res, oy + r1 * res),
              vmin=0, vmax=1.5, interpolation="nearest")
    if track.centerline is not None:
        cl = track.centerline
        m = (np.abs(cl[:, 0] - x0) < W) & (np.abs(cl[:, 1] - y0) < W)
        ax.plot(cl[m, 0], cl[m, 1], ".", color="0.6", ms=1.5)
    if props_in is not None and len(props_in):
        ax.scatter(props_in[:, 0], props_in[:, 1], s=4, c="darkorange", marker="s", linewidths=0)
    tr = np.array(ev["traj"])
    sc = ax.scatter(tr[:, 0], tr[:, 1], c=tr[:, 3], cmap="viridis", s=8, vmin=0, vmax=9)
    plt.colorbar(sc, ax=ax, fraction=0.035, label="속도 m/s")
    # the car at the hit: 0.51 x 0.27 m
    yaw = ev["yaw"]; c, s = math.cos(yaw), math.sin(yaw)
    L, Wc = 0.51 / 2, 0.27 / 2
    corners = np.array([[L, Wc], [L, -Wc], [-L, -Wc], [-L, Wc], [L, Wc]])
    R = np.array([[c, -s], [s, c]])
    poly = corners @ R.T + np.array([x0, y0])
    ax.plot(poly[:, 0], poly[:, 1], "r-", lw=2)
    if plan_xy is not None:
        # the layout-line planner's minimum-time line for THIS layout (privileged, re-planned around the props)
        m = (np.abs(plan_xy[:, 0] - x0) < W) & (np.abs(plan_xy[:, 1] - y0) < W)
        ax.plot(plan_xy[m, 0], plan_xy[m, 1], ".", color="magenta", ms=2.5, label="계획기 선(이 배치)")
        ax.legend(loc="lower right", fontsize=7)
    if my_paths is not None:
        # the teacher's own planned path (learn.path_plan) 1 s / 0.5 s before the hit and at it
        for pth, col, lab in zip(my_paths, ("lightskyblue", "deepskyblue", "navy"), ("내 계획 1 s 전", "0.5 s 전", "충돌 때")):
            ax.plot(pth[:, 0], pth[:, 1], "-", color=col, lw=1.6, label=lab)
        ax.legend(loc="lower right", fontsize=7)
    for oh in opp_hist:
        oh = np.array(oh)
        ax.plot(oh[:, 0], oh[:, 1], "-", color="royalblue", lw=1.2)
        ax.plot(oh[-1, 0], oh[-1, 1], "s", color="royalblue", ms=6)
    ax.set_xlim(x0 - W, x0 + W); ax.set_ylim(y0 - W, y0 + W); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def prop_points(po, row):
    """points inside each live prop of env row `row` (a grid around its pose, kept where every half-plane holds)."""
    pts = []
    live = (po.p_sid[row] >= 0).nonzero().flatten().tolist()
    g = torch.linspace(-0.7, 0.7, 29, device=po.p_poses.device)
    gx, gy = torch.meshgrid(g, g, indexing="ij")
    for j in live:
        cx, cy, cyaw = (float(v) for v in po.p_poses[row, j, :3])
        P = torch.stack([gx.reshape(-1) + cx, gy.reshape(-1) + cy], 1)
        # the half-planes are in the prop's own frame (`prop_math._to_world`)
        c, s_ = math.cos(cyaw), math.sin(cyaw)
        dxy = P - torch.tensor([cx, cy], device=P.device)
        loc = torch.stack([dxy[:, 0] * c + dxy[:, 1] * s_, -dxy[:, 0] * s_ + dxy[:, 1] * c], 1)
        n, d = po.p_n[row, j], po.p_d[row, j]
        inside = ((loc @ n.T) - d[None, :] <= 0).all(1)
        if inside.any():
            pts.append(P[inside].cpu().numpy())
    return np.concatenate(pts) if pts else np.zeros((0, 2))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--stage", default="props", choices=("solo", "props", "race"))
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--warps", type=int, default=0)
    ap.add_argument("--beams", type=int, default=36)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=30)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gap-rule", type=float, default=0.0,
                    help="falsifier for 'the teacher is near its optimum on cars': cap its speed command to keep this time "
                         "gap [s] (+0.3 m) to a car ahead in its lane; compare the J80 rate with the rule off")
    ap.add_argument("--all-car", type=int, default=0, help="1: count every hard car contact, charged or not")
    ap.add_argument("--unwedge", type=int, default=0, help="1: the wedge recovery (as state_student evalx / training)")
    ap.add_argument("--follow-rule", type=float, default=0.0,
                    help="ORACLE diagnostic: state_student.follow_cap at this deceleration [m/s^2] on every step; 0 = off")
    ap.add_argument("--steps-total", type=int, default=0, help="unused")
    ap.add_argument("--overlay-layout", type=int, default=0,
                    help="1: draw the layout-line planner's line for the hit row's layout (one-track envs: --like / --warps 0)")
    ap.add_argument("--like", default="", help="an eval_ckpt.sh race JSON: drive in exactly that world (state_student.make_env_like)")
    ap.add_argument("--student", default="", help="a state_student checkpoint: the STUDENT drives (its own sensors); the "
                    "teacher CKPT labels every decision, so each hit shows what the teacher would have done instead")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "scenes_" + os.path.basename(a.ckpt).replace(".pt", ""))
    os.makedirs(out, exist_ok=True)
    device = torch.device(a.device)
    ck0 = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sck = torch.load(a.student, map_location=device, weights_only=False) if a.student else None
    a.v_min = float(ck0["args"].get("v_min", 0.0) or 0.0)       # build_env reads it: a policy that may reverse
    a.reverse_ext = float(ck0["args"].get("reverse_ext", 0.0) or 0.0)
    if a.like:
        from f1sim.learn.state_student import make_env_like
        fr_ = int(sck["frames"]) if sck is not None else 1
        sd_ = int(sck["args"].get("stride", 3)) if sck is not None else 1
        if sck is not None:
            a.beams = int(sck["beams"])
        # priv flags on in both modes: the teacher's StateObs labels / probes every decision
        vmin_ = float(ck0["args"].get("v_min", 0.0) or 0.0); rext_ = float(ck0["args"].get("reverse_ext", 0.0) or 0.0)
        ov_ = dict(**({"v_min_policy": vmin_} if vmin_ else {}), **({"reverse_ext": rext_} if rext_ else {}))
        env, _cfg, _ecfg, _md = make_env_like(a.like, device, int(a.beams), fr_, sd_, True, envs=int(a.envs),
                                              override=(ov_ or None))
    elif sck is not None:
        # the student's sensors (its beams, the 3-frame scan stack of the v1 obs dict); the teacher's scan is
        # interpolated to ITS beams over the same field of view (as state_student does)
        a.beams = int(sck["beams"])
        env = state_rl.build_env(a, device, extra=dict(obs_version=1, scan_stack=int(sck["frames"]), scan_stride=int(sck["args"].get("stride", 3))))
    else:
        env = state_rl.build_env(a, device)
    from f1sim.learn.graph_runtime import prepare_graph_runtime
    if not a.like:
        prepare_graph_runtime(env, log=lambda _t: None)        # make_env_like already did
    lt = None
    if int(a.overlay_layout):
        from f1sim.learn import common as _cm
        from f1sim.layout_line import LayoutLineTeacher
        _lim = _cm.teacher_limits(7.0, 6.5, 4.0)
        _trs, _rls = _cm.load_tracks([state_rl.KOREA], racelines=True, **dict(_lim, objective="min_time"))
        lt = LayoutLineTeacher(_cm.make_teacher(_rls, env, **_lim), env)
    t_scan = bool(ck0["args"].get("scan_obs", 0)); t_beams = int(ck0["args"].get("beams", 36))
    # the teacher's observation (its scan interpolated to its beams when the student's env casts more; its prop corridor
    # if it was trained with one) -- state_rl.StateObs builds it in the training order
    obs_fn = StateObs(env, scan=t_scan, corridor=bool(ck0["args"].get("corridor_obs", 0)), corridor_cars=bool(ck0["args"].get("corridor_cars", 0)), car_channel=bool(ck0["args"].get("car_channel", 0)), bev=bool(ck0["args"].get("bev", 0)), path_plan=int(ck0["args"].get("path_plan", 0)), plan_ref=ck0["args"].get("plan_ref"), t_beams=t_beams)
    obs_d, _ = env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    if sck is not None:
        from f1sim.learn.lidar_grid import StudentRunner
        student = StudentRunner(env, lid, sck, device)              # the 1-D scan student or the grid student
        student.observe()
        ego_keys = sck.get("ego_keys") or sorted(k for k in obs_d if k != "scan")
        egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in ego_keys], 1)
    dif_hist = torch.zeros(40, B, 2, device=device)        # (student - teacher) action per decision, last 2 s
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    last_a = torch.zeros(B, 2, device=device)
    o = obs_fn(last_a)
    model = ActorCritic(o.shape[1], bev=ck.get("bev")).to(device); model.load_state_dict(ck["model"], strict=False)
    norm = RunningNorm(o.shape[1], device); norm.load_state_dict(ck["norm"])
    # `--base` checkpoints (10-31): the gated mixture drives -- the frozen solo teacher where nothing is near
    mix = state_rl.SoloMix(env, ck0["args"]["base"], device, float(ck0["args"].get("base_reach", state_rl.BASE_REACH)),
                           int(ck0["args"].get("base_hold", state_rl.BASE_HOLD))) if ck0["args"].get("base") else None
    ghist = torch.ones(40, B, device=device)                # 1 = the race policy drove that decision
    last_exec = last_a
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    hist = torch.zeros(H, env.B, 6, device=device)     # x, y, yaw, vx, steer cmd, speed cmd -- every car
    HD = 40                                              # 2 s of decisions (20 Hz)
    ohist = torch.zeros(HD, B, o.shape[1], device=device)
    # the observation's blocks (state_rl.StateObs): privileged_v2 = ego 8, opponents 21, geometry 4, props 12, time 1;
    # then the track ahead 13 x 4, the steering angle, the last action
    groups = {"내 차": list(range(0, 8)), "상대": list(range(8, 29)), "상대 기하": list(range(29, 33)),
              "상자": list(range(33, 45)), "남은 시간": [45], "앞 트랙": list(range(46, 98)), "조향각": [98],
              "직전 행동": [99, 100]}
    D_all = int(o.shape[1]); n_sc = t_beams if t_scan else 0
    if n_sc:
        groups["라이다"] = list(range(101, 101 + n_sc))
    n_car = 91 if bool(ck0["args"].get("car_channel", 0)) else 0       # the car corridor block, appended last
    corr_cols = list(range(101 + n_sc, D_all - n_car)) if bool(ck0["args"].get("corridor_obs", 0)) else []
    if corr_cols:
        groups["상자 통로"] = corr_cols
    if n_car:
        groups["차 통로"] = list(range(D_all - n_car, D_all))
    slot_cols = list(range(33, 45))                       # what "props erased" zeroes besides the corridor
    is_bev = bool(ck0["args"].get("bev", 0))
    if is_bev:
        # state_rl.StateObs(bev=True): own state 8, time left, track ahead 13 x 4, steer, last action, then the grid
        from f1sim.learn.state_rl import BEV_C, BEV_H, BEV_W
        HW = BEV_H * BEV_W; G0 = D_all - BEV_C * HW
        groups = {"내 차": list(range(0, 8)), "남은 시간": [8], "앞 트랙": list(range(9, 61)), "조향각": [61],
                  "직전 행동": [62, 63], "격자 벽": list(range(G0, G0 + HW)), "격자 상자": list(range(G0 + HW, G0 + 2 * HW)),
                  "격자 차": list(range(G0 + 2 * HW, D_all))}
        slot_cols = list(range(G0 + HW, G0 + 2 * HW))      # the grid's prop channel
        corr_cols = []
    probe = []                                           # per hit: the policy's response 0.5 s before it

    def respond(x_raw):
        """action, value and |input x gradient| per group at one raw observation (1, D)"""
        with torch.enable_grad():
            x = norm(x_raw).detach().requires_grad_(True)
            a_ = model.pi(x)
            g_s = torch.autograd.grad(a_[0, 0], x, retain_graph=True)[0][0]
            g_v = torch.autograd.grad(a_[0, 1], x)[0][0]
        xn = x.detach()[0]
        sal = {k: (float((g_s[idx] * xn[idx]).abs().sum()), float((g_v[idx] * xn[idx]).abs().sum())) for k, idx in groups.items()}
        v = float(model.v(xn[None]).squeeze()) * float(ret_rms.var.sqrt() + 1e-6)
        # the hit head (P(a hard hit within 20 decisions)); untrained (no --hit-head) checkpoints give noise here
        ph = float(torch.sigmoid(model.value_and_hit(xn[None])[1]).squeeze())
        return state_rl.clamp_act(a_.detach()[0], ck0["args"].get("reverse_ext", 0.0)), v, sal, ph

    ret_rms = RunningNorm(1, device); ret_rms.load_state_dict(ck["ret_rms"])
    touching = torch.zeros(B, dtype=torch.bool, device=device)
    dif_all = []; dif_hit = []                           # |student - teacher| per decision, overall and before hits
    gap_on = []                                          # share of decisions the gap rule limited
    slot_log = []                                        # per prop hit: was the prop in the obs's 3 slots 1 s / 0.5 s before
    prog_m = 0.0; n_dec_tot = 0
    events, n_img = [], 0
    km = 0.0
    po = env.procedural
    enc_seen = None; enc_hit = None
    if po is not None and po.C > 0:
        ext_all = torch.where(torch.isfinite(po.p_d), po.p_d, torch.full_like(po.p_d, -1e9)).amax(-1)   # (B, C) extent
        enc_seen = torch.zeros(env.B, po.C, dtype=torch.bool, device=device)
        enc_hit = torch.zeros(env.B, po.C, dtype=torch.bool, device=device)
        enc_log = []                     # (extent, speed at closest, hit) per finished encounter
        enc_vmax = torch.zeros(env.B, po.C, device=device)
        enc_gmin = torch.full((env.B, po.C), 9.0, device=device)     # the closest the car's centre came to the edge
        # was the prop shoved before the car reached it (by anyone: in a race the props are shared by the race's cars)?
        p_moved = torch.zeros(env.B, po.C, dtype=torch.bool, device=device)
        enc_moved = torch.zeros(env.B, po.C, dtype=torch.bool, device=device)
        enc_vel = torch.zeros(env.B, po.C, device=device)             # the prop's own speed while the car was near
        enc_od = torch.full((env.B, po.C), 99.0, device=device)       # the nearest opponent's distance during it
    tracks = env.sim.track.tracks
    uw = Unwedge(env, lid, mode=int(a.unwedge)) if int(a.unwedge) else None
    env.sim.measure_impacts = True                     # car impacts too (as ContactSeverityMeter turns it on for evaluate)
    car_prev_all = torch.zeros(B, dtype=torch.bool, device=device)
    my_pl = getattr(obs_fn, "planner", None)               # the teacher's own path planner (state_rl --path-plan)
    if my_pl is not None:
        pl_ds = torch.arange(0.0, 15.01, 0.4, device=device)
        planhist = torch.zeros(HD, B, pl_ds.numel(), 2, device=device)
    for t in range(int(a.steps) // int(a.repeat)):
        ohist = torch.roll(ohist, -1, 0); ohist[-1] = o
        if my_pl is not None:
            planhist = torch.roll(planhist, -1, 0)
            planhist[-1] = my_pl.path_xy(env.last_result.s[lid][:, None] + pl_ds[None], env.sim.tid[lid])[0]
        act = state_rl.clamp_act(model.pi(norm(o)), ck0["args"].get("reverse_ext", 0.0))
        if mix is not None:
            act = mix(act, last_exec)
            ghist = torch.roll(ghist, -1, 0); ghist[-1] = mix.gate.float()
        if float(a.gap_rule) > 0 and env.M > 1:
            st0 = env.sim.state
            dx_, dy_, dyaw_, dvx_, dvy_, dist_, pr_ = env._opp_rel(st0)
            dx_, dy_, dvx_, pr_ = dx_[lid], dy_[lid], dvx_[lid], pr_[lid]
            ahead = pr_ & (dx_ > 0.0) & (dy_.abs() < 0.6)
            key = torch.where(ahead, dx_, torch.full_like(dx_, 1e6))
            k_ = key.argmin(1, keepdim=True)
            anyq = ahead.gather(1, k_)[:, 0]
            gap = dx_.gather(1, k_)[:, 0] - 0.55
            clos = -dvx_.gather(1, k_)[:, 0]
            v_e = st0[lid, 3]
            g_star = 0.3 + float(a.gap_rule) * v_e
            v_cap = (v_e - clos - 1.5 * (g_star - gap)).clamp(0.0, float(env.ecfg.v_max_policy))
            a_cap = 2.0 * v_cap / float(env.ecfg.v_max_policy) - 1.0
            limit = anyq & (gap < g_star)
            act = act.clone(); act[:, 1] = torch.where(limit, torch.minimum(act[:, 1], a_cap), act[:, 1])
            gap_on.append(float(limit.float().mean()))
        if sck is not None:
            with torch.no_grad():
                s_act = state_rl.clamp_act(student(obs_d, egov(obs_d)), sck["args"].get("reverse_ext", 0.0))
            dif_hist = torch.roll(dif_hist, -1, 0); dif_hist[-1] = s_act - act
            dif_all.append((s_act - act).abs().mean(0).cpu())
            act = s_act
        act_full[lid] = act
        for r in range(int(a.repeat)):
            if float(a.follow_rule) > 0:
                from f1sim.learn.state_student import follow_cap
                act_full[lid, 1] = torch.minimum(act[:, 1], follow_cap(env, lid, float(a.follow_rule)))
            if uw is not None:
                uw.before(act_full)
            obs_d, r_, te, trc, info = env.step(act_full)
            if sck is not None:
                student.observe()                                   # the pose this step left (grid student odometry)
            if uw is not None:
                uw.after()
            st = env.sim.state
            cmd = torch.zeros(env.B, 2, device=device); cmd[lid] = act
            hist = torch.roll(hist, -1, 0)
            hist[-1] = torch.cat([st[:, :4], cmd[:, :1] * float(env.s_max), a.v_min + (cmd[:, 1:2] + 1) / 2 * (float(env.ecfg.v_max_policy) - a.v_min)], 1)
            res = env.last_result
            comp = info["reward_components"]
            car = comp["car_contact"][lid].float(); tch = car < 0; c_on = tch & ~touching
            d_ = (te | trc)[lid]; touching = tch & ~d_
            km += float(info["progress"][lid].float().clamp_min(0).sum()) / 1000.0
            prog_m += float(info["progress"][lid].float().sum())
            on, imp = res.contact_onsets[lid], res.contact_impacts[lid]
            hard_w = on[:, 0] & (imp[:, 0] >= 1.5); hard_p = on[:, 1] & (imp[:, 1] >= 1.5)
            if enc_seen is not None:
                # encounters: a live prop whose edge comes within 0.6 m of the car's centre; the hard prop onset of a row
                # is credited to its nearest prop; an encounter ends when the prop falls 2 m behind or the episode resets
                ext_all = torch.where(torch.isfinite(po.p_d), po.p_d, torch.full_like(po.p_d, -1e9)).amax(-1)
                dxy = po.p_poses[:, :, :2] - st[:, None, :2]
                gap = dxy.norm(dim=-1) - ext_all
                live = po.p_sid >= 0
                near = live & (gap < 0.6)
                lrow = torch.zeros(env.B, dtype=torch.bool, device=device); lrow[lid] = True
                near = near & lrow[:, None]
                pv_ = po.p_vel.norm(dim=-1)
                new_enc = near & ~enc_seen
                enc_moved |= new_enc & p_moved                                 # shoved before this encounter began
                p_moved |= live & (pv_ > 0.02)
                enc_vel = torch.where(near, torch.maximum(enc_vel, pv_), enc_vel)
                if env.M > 1:
                    oth = torch.full((env.B,), 99.0, device=device)
                    rr = torch.arange(env.B, device=device)
                    for k_ in range(env.M):
                        q_ = (rr // env.M) * env.M + k_
                        dq = (st[q_, :2] - st[:, :2]).norm(dim=1)
                        oth = torch.where(q_ != rr, torch.minimum(oth, dq), oth)
                    enc_od = torch.where(near, torch.minimum(enc_od, oth[:, None].expand_as(enc_od)), enc_od)
                enc_seen |= near
                enc_vmax = torch.where(near, torch.maximum(enc_vmax, st[:, 3:4].expand_as(enc_vmax)), enc_vmax)
                enc_gmin = torch.where(near, torch.minimum(enc_gmin, gap), enc_gmin)
                hp = torch.zeros(env.B, dtype=torch.bool, device=device); hp[lid] = hard_p
                jn = torch.where(live, gap, torch.full_like(gap, 1e9)).argmin(1)
                hitmask = torch.zeros_like(enc_hit); hitmask[torch.arange(env.B, device=device), jn] = hp
                enc_hit |= hitmask & enc_seen
                c_, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
                behind = (dxy[..., 0] * c_ + dxy[..., 1] * s_) < -2.0
                dn = torch.zeros(env.B, dtype=torch.bool, device=device); dn[lid] = d_
                fin = enc_seen & (behind | dn[:, None] | ~live)
                if fin.any():
                    ii = fin.nonzero()
                    # where the prop sits across the track: |lateral| / half width at its arc position (1 = at the wall)
                    trk_ = env.sim.track; tid_f = env.sim.tid[ii[:, 0]]
                    pc_f = po.p_poses[ii[:, 0], ii[:, 1], :2]
                    s_f, lat_f, _ = trk_.project(pc_f, tid_f)
                    xy_f, _ = trk_.pose_at_s(s_f, tid_f)
                    half_f = trk_.sample_edt(xy_f, tid_f).clamp_min(0.05)
                    side_f = (lat_f.abs() / half_f).tolist()
                    for k_, (b_, c_i) in enumerate(ii.tolist()):
                        enc_log.append((float(ext_all[b_, c_i]), float(enc_vmax[b_, c_i]), bool(enc_hit[b_, c_i]),
                                        float(side_f[k_]), float(enc_gmin[b_, c_i]), bool(enc_moved[b_, c_i]),
                                        float(enc_vel[b_, c_i]), float(enc_od[b_, c_i])))
                    enc_seen &= ~fin; enc_hit &= ~fin; enc_vmax = torch.where(fin, torch.zeros_like(enc_vmax), enc_vmax)
                    enc_gmin = torch.where(fin, torch.full_like(enc_gmin, 9.0), enc_gmin)
                    enc_moved &= ~fin; enc_vel = torch.where(fin, torch.zeros_like(enc_vel), enc_vel)
                    enc_od = torch.where(fin, torch.full_like(enc_od, 99.0), enc_od)
                p_moved &= ~dn[:, None]
            hard_c = c_on & (res.car_impact[lid] >= 1.5) if res.car_impact is not None else c_on
            if int(a.all_car) and res.car_collision is not None:
                # every hard car contact onset (as learn.evaluate's ContactSeverityMeter counts), not only the ones the
                # reward charges to this car (the car behind / alongside)
                cc_now = res.car_collision[lid]
                hard_c = cc_now & ~car_prev_all & (res.car_impact[lid] >= 1.5)
                car_prev_all = cc_now.clone() & ~d_
            for kind, mask, impv in (("벽", hard_w, imp[:, 0]), ("상자", hard_p, imp[:, 1]),
                                     ("차", hard_c, res.car_impact[lid] if res.car_impact is not None else imp[:, 2])):
                for i in mask.nonzero().flatten().tolist():
                    row = int(lid[i])
                    n_ep = int(min(H, int(env.ep_step[row]) + 1))          # only this episode's own steps
                    h = hist[H - n_ep:, row].cpu().numpy()
                    od = float("nan"); geo = {}
                    if env.M > 1:
                        race = row // env.M
                        others = [q for q in range(race * env.M, race * env.M + env.M) if q != row]
                        od = float(min((st[q, :2] - st[row, :2]).norm() for q in others))
                        if kind == "차":
                            # who ran into whom: the nearest opponent's position in the ego frame and the closing speed
                            # along the ego heading; and 0.5 s earlier, who was ahead along the track
                            q = min(others, key=lambda q_: float((st[q_, :2] - st[row, :2]).norm()))
                            ce, se = math.cos(float(st[row, 2])), math.sin(float(st[row, 2]))
                            dxq, dyq = float(st[q, 0] - st[row, 0]), float(st[q, 1] - st[row, 1])
                            fx, fy = dxq * ce + dyq * se, -dxq * se + dyq * ce
                            vq = float(st[q, 3]); dyaw_q = float(st[q, 2] - st[row, 2])
                            v_rel = float(st[row, 3]) - vq * math.cos(dyaw_q)          # > 0: the ego closes from behind
                            h_q = hist[-min(20, n_ep):, q].cpu().numpy(); h_e = hist[-min(20, n_ep):, row].cpu().numpy()
                            s_e0, _, _ = env.sim.track.project(torch.tensor(h_e[:1, :2], device=device), env.sim.tid[row:row + 1])
                            s_q0, _, _ = env.sim.track.project(torch.tensor(h_q[:1, :2], device=device), env.sim.tid[row:row + 1])
                            L_ = float(env.sim.track.length[env.sim.tid[row]]) if hasattr(env.sim.track, "length") else 1e9
                            ds0 = float(s_q0[0] - s_e0[0]); ds0 = (ds0 + L_ / 2) % L_ - L_ / 2 if L_ < 1e8 else ds0
                            # the opponent's lateral motion over the last 0.5 s (in the ego's frame at the hit)
                            lat_q = [(-(x_ - float(st[row, 0])) * se + (y_ - float(st[row, 1])) * ce) for x_, y_ in h_q[:, :2]]
                            geo = {"opp_fx": fx, "opp_fy": fy, "closing": v_rel, "opp_speed": vq,
                                   "opp_ahead_0.5s": ds0, "opp_lat_move_0.5s": float(lat_q[-1] - lat_q[0]) if len(lat_q) > 1 else 0.0}
                    ev = {"step": t * int(a.repeat) + r, "row": row, "kind": kind, "impact": float(impv[i]),
                          "ep_step": int(env.ep_step[row]), "opp_dist": od, **geo,
                          "speed": float(st[row, 3]), "xy": [float(st[row, 0]), float(st[row, 1])], "yaw": float(st[row, 2]),
                          "traj": h[:, :4].tolist(), "steer_cmd_0.5s": h[-min(20, len(h)):, 4].tolist(), "speed_cmd_0.5s": h[-min(20, len(h)):, 5].tolist(),
                          "race_policy_1s": float(ghist[-20:, i].mean()), "race_policy_2s": float(ghist[:, i].mean()),
                          "race_policy_now": float(ghist[-1, i])}
                    if mix is not None:
                        # the handover: the speed the car had when the race policy last took it over from the solo teacher
                        g_ = ghist[:, i].cpu().numpy(); k_ = np.nonzero(g_ < 0.5)[0]
                        ev["takeover_s_before"] = (len(g_) - 1 - int(k_[-1])) / 20.0 if len(k_) else None
                        if len(k_) and int(k_[-1]) + 1 < len(g_):
                            j_ = H - (len(g_) - 1 - int(k_[-1])) * int(a.repeat)
                            ev["takeover_speed"] = float(hist[max(j_, 0), row, 3])
                    # the policy's response 0.5 s before the hit (10 decisions back), with the props as seen and erased
                    n_dec = int(min(HD, int(env.ep_step[row]) // int(a.repeat)))
                    if n_dec >= 11:
                        x0 = ohist[-11, i][None].clone()
                        a_full, v_full, sal, _ = respond(x0)
                        x_np = x0.clone(); x_np[0, slot_cols] = 0.0
                        if corr_cols:
                            x_np[0, corr_cols] = 0.0          # the prop corridor is props too
                        a_np, v_np, _, _ = respond(x_np)
                        # which prop input drives it: the 3 nearest-prop slots alone / the prop corridor alone erased
                        x_ns = x0.clone(); x_ns[0, slot_cols] = 0.0
                        a_ns, _, _, _ = respond(x_ns)
                        x_nc = x0.clone()
                        if corr_cols:
                            x_nc[0, corr_cols] = 0.0
                        a_nc, _, _, _ = respond(x_nc)
                        rr = [respond(ohist[-k, i][None].clone()) for k in (min(n_dec, 40), 21, 11, 1)]
                        vals = [r_[1] for r_ in rr]; phs = [r_[3] for r_ in rr]
                        ev.update(resp={"steer_deg": float(a_full[0]) * math.degrees(float(env.s_max)),
                                        "speed_ms": a.v_min + (float(a_full[1]) + 1) / 2 * (float(env.ecfg.v_max_policy) - a.v_min),
                                        "steer_deg_noprops": float(a_np[0]) * math.degrees(float(env.s_max)),
                                        "steer_deg_noslots": float(a_ns[0]) * math.degrees(float(env.s_max)),
                                        "steer_deg_nocorr": float(a_nc[0]) * math.degrees(float(env.s_max)),
                                        "speed_ms_noprops": a.v_min + (float(a_np[1]) + 1) / 2 * (float(env.ecfg.v_max_policy) - a.v_min),
                                        "V_2s_1s_05s_0s": vals, "Phit_2s_1s_05s_0s": phs, "sal": sal,
                                        "props_seen": int(float(x0[0, slot_cols].max() if is_bev else x0[0, 33]) > 0.5)})
                        probe.append(ev["resp"])
                    # the nearest live prop and the room beside it: track width at its arc position (2 x the centerline's
                    # distance to the wall) minus the prop's own width across the track
                    po = env.procedural
                    if po is not None and kind == "상자" and not is_bev:
                        live = (po.p_sid[row] >= 0)
                        if live.any():
                            pc = po.p_poses[row, :, :2]
                            dd = (pc - st[row, :2]).norm(dim=1); dd = torch.where(live, dd, torch.full_like(dd, 1e9))
                            j = int(dd.argmin())
                            # the obs slots back then: ohist holds the raw teacher obs of the last 40 decisions (cols 33:45 =
                            # 3 x (present, dx/5, dy/5, dist/5)); hist the poses per step (2 steps per decision)
                            slot_info = {}
                            for lag_d, nm in ((20, "1s"), (10, "0.5s")):
                                if int(env.ep_step[row]) // int(a.repeat) < lag_d + 1:
                                    continue
                                ob = ohist[-lag_d - 1, i]
                                hx, hy, hyaw = [float(v) for v in hist[-(2 * lag_d) - 1, row, :3]]
                                ddx, ddy = float(pc[j, 0]) - hx, float(pc[j, 1]) - hy
                                ex = ddx * math.cos(hyaw) + ddy * math.sin(hyaw); ey = -ddx * math.sin(hyaw) + ddy * math.cos(hyaw)
                                found = False
                                for q in range(3):
                                    if float(ob[33 + 4 * q]) > 0.5:
                                        sx, sy = float(ob[34 + 4 * q]) * 5.0, float(ob[35 + 4 * q]) * 5.0
                                        if abs(sx - ex) < 0.3 and abs(sy - ey) < 0.3:
                                            found = True
                                # its rank among live props ahead (dx > -1 m, < 12 m) by distance, as the env sorts them
                                pp = po.p_poses[row, :, :2].cpu().numpy(); lv = live.cpu().numpy()
                                rx = (pp[:, 0] - hx) * math.cos(hyaw) + (pp[:, 1] - hy) * math.sin(hyaw)
                                rd = np.hypot(pp[:, 0] - hx, pp[:, 1] - hy)
                                ok_ = lv & (rx > -1.0) & (rd < 12.0)
                                rank = int((rd[ok_] < math.hypot(ex, ey) - 1e-6).sum()) + 1 if math.hypot(ex, ey) < 12.0 and ex > -1.0 else 99
                                slot_info[nm] = {"in_slots": found, "rank": rank, "dist": math.hypot(ex, ey)}
                            ev["slot_info"] = slot_info
                            slot_log.append(slot_info)
                            ext = float(po.p_d[row, j][torch.isfinite(po.p_d[row, j])].max())   # farthest face from the pose
                            trk = env.sim.track
                            tid_t = env.sim.tid[row:row + 1]
                            s_p, lat_p, _ = trk.project(pc[j:j + 1], tid_t)
                            xy_c, _ = trk.pose_at_s(s_p, tid_t)
                            half = float(trk.sample_edt(xy_c, tid_t)[0])
                            gap = 2 * half - 2 * ext
                            ev.update(prop_lat=float(lat_p[0]), track_half_width=half, prop_extent=ext,
                                      gap_beside=max(half - (float(lat_p[0]) + ext), half + (float(lat_p[0]) - ext)),
                                      gap_total=gap)
                    if sck is not None:
                        dd = dif_hist[-10:, i].cpu().numpy()                  # last 0.5 s
                        ev["stud_minus_teach_05s"] = {"steer_deg": float(np.mean(dd[:, 0])) * math.degrees(float(env.s_max)),
                                                      "speed_ms": float(np.mean(dd[:, 1])) / 2 * (float(env.ecfg.v_max_policy) - a.v_min)}
                        dif_hit.append((kind, ev["stud_minus_teach_05s"]["steer_deg"], ev["stud_minus_teach_05s"]["speed_ms"]))
                    events.append({k: v for k, v in ev.items() if k != "traj"})
                    if n_img < int(a.max_images):
                        tid = int(env.sim.tid[row])
                        props = prop_points(env.procedural, row) if env.procedural is not None else None
                        opp = []
                        if env.M > 1:
                            race = row // env.M
                            for q in range(race * env.M, race * env.M + env.M):
                                if q != row:
                                    opp.append(hist[-40:, q, :2].cpu().numpy())
                        sc = np.array(ev["speed_cmd_0.5s"]); stc = np.degrees(np.array(ev["steer_cmd_0.5s"]))
                        title = (f"#{n_img} {kind} 충돌, 충격 {ev['impact']:.2f} m/s, 속도 {ev['speed']:.2f} m/s, 출발 후 {ev['ep_step'] / 40:.1f} s\n"
                                 f"직전 0.5초 속도 명령 {sc[0]:.1f} -> {sc[-1]:.1f} m/s, 조향 {stc[0]:+.0f} -> {stc[-1]:+.0f} 도")
                        if mix is not None:
                            title += (f" | 몬 정책: 지금 {'경주' if ev['race_policy_now'] > 0.5 else '혼자'}, "
                                      f"직전 2 s 경주 {ev['race_policy_2s'] * 100:.0f} %")
                            if ev.get("takeover_s_before") is not None:
                                title += f"\n경주 정책이 넘겨받은 때: {ev['takeover_s_before']:.2f} s 전, 그때 속도 {ev.get('takeover_speed', float('nan')):.1f} m/s"
                        if "resp" in ev:
                            rp = ev["resp"]
                            title += (f"\n0.5초 전: 상자 보임 {'예' if rp['props_seen'] else '아니오'} | 조향 {rp['steer_deg']:+.0f}도 / 상자 지우면 "
                                      f"{rp['steer_deg_noprops']:+.0f}도, 속도 {rp['speed_ms']:.1f} / {rp['speed_ms_noprops']:.1f} m/s"
                                      f"\n가치 2s/1s/0.5s/0s 전: " + " / ".join(f"{x:.0f}" for x in rp["V_2s_1s_05s_0s"])
                                      + " | 충돌 예측: " + " / ".join(f"{x:.2f}" for x in rp["Phit_2s_1s_05s_0s"]))
                        if "stud_minus_teach_05s" in ev:
                            title += (f"\n학생-선생(직전 0.5초): 조향 {ev['stud_minus_teach_05s']['steer_deg']:+.1f}도, "
                                      f"속도 {ev['stud_minus_teach_05s']['speed_ms']:+.2f} m/s")
                        plan_xy = None
                        if lt is not None:
                            lt.refresh([row])
                            plan_xy = lt.lines.xy[row].cpu().numpy()
                        my_paths = None
                        if my_pl is not None:
                            my_paths = [planhist[-21, i].cpu().numpy(), planhist[-11, i].cpu().numpy(), planhist[-1, i].cpu().numpy()]
                        draw(os.path.join(out, f"scene_{n_img:03d}.png"), ev, tracks[tid], props, opp, title, plan_xy, my_paths)
                        n_img += 1
        o = obs_fn(act)
        last_exec = act
    for nm in ("1s", "0.5s"):
        q = [x[nm] for x in slot_log if nm in x]
        if q:
            ins = np.array([x["in_slots"] for x in q]); rk = np.array([x["rank"] for x in q]); dist = np.array([x["dist"] for x in q])
            print(f"[관측 칸] 상자 충돌 {len(q)}건, {nm} 전: 박은 상자가 관측 3칸 안에 있었음 {ins.mean() * 100:.0f} % | 그 상자의 거리 순위 "
                  f"1위 {(rk == 1).mean() * 100:.0f} %, 2~3위 {((rk >= 2) & (rk <= 3)).mean() * 100:.0f} %, 4위 이하 {((rk >= 4) & (rk < 99)).mean() * 100:.0f} %, "
                  f"12 m 밖·뒤 {(rk == 99).mean() * 100:.0f} % | 거리 중앙 {np.median(dist):.1f} m", flush=True)
    secs = (int(a.steps) // int(a.repeat)) * int(a.repeat) * B * env.sim.control_dt
    n_hard = len(events)
    if events:
        from collections import Counter
        per_row = Counter(e["row"] for e in events)
        top = per_row.most_common(3)
        print(f"[차별] 충돌 {n_hard}회가 {len(per_row)}대에서: 가장 많은 차 {top} | 끼임 복구 {uw.count if uw is not None else '-'}회", flush=True)
    print(f"[J80] 진행 {prog_m / secs:.3f} m/s, 강한 충돌 {n_hard / max(km, 1e-9):.2f}/km, J80 = (진행 - 80 x 충돌)/시간 = "
          f"{(prog_m - 80.0 * n_hard) / secs:.4f} m/s" + (f" | 간격 규칙이 막은 결정 {np.mean(gap_on) * 100:.1f} %" if gap_on else ""), flush=True)
    if sck is not None and dif_all:
        da = torch.stack(dif_all).mean(0).numpy()
        print(f"[학생] 평소 |학생-선생| 조향 {da[0] * math.degrees(float(env.s_max)):.1f}도, 속도 {da[1] / 2 * (float(env.ecfg.v_max_policy) - a.v_min):.2f} m/s", flush=True)
        for kind in ("벽", "상자", "차"):
            q = [(x, y) for k_, x, y in dif_hit if k_ == kind]
            if q:
                q = np.array(q)
                print(f"[학생] {kind} 충돌 {len(q)}건, 직전 0.5초 학생-선생: 조향 평균 {np.mean(q[:, 0]):+.1f}도 (|중앙| {np.median(np.abs(q[:, 0])):.1f}), "
                      f"속도 평균 {np.mean(q[:, 1]):+.2f} m/s (|중앙| {np.median(np.abs(q[:, 1])):.2f}); 학생이 0.3 m/s 넘게 더 빨랐던 비율 "
                      f"{(q[:, 1] > 0.3).mean() * 100:.0f} %", flush=True)
    if probe:
        ds = np.array([abs(p_["steer_deg"] - p_["steer_deg_noprops"]) for p_ in probe])
        dv = np.array([p_["speed_ms_noprops"] - p_["speed_ms"] for p_ in probe])
        seen = np.array([p_["props_seen"] for p_ in probe])
        V = np.array([p_["V_2s_1s_05s_0s"] for p_ in probe])
        print(f"[반응] 충돌 0.5초 전 {len(probe)}건: 상자가 관측에 보임 {seen.mean() * 100:.0f} %; 상자를 지웠을 때 조향 변화 중앙값 "
              f"{np.median(ds):.1f}도 (90 % {np.percentile(ds, 90):.1f}), 속도 변화 중앙값 {np.median(dv):+.2f} m/s "
              f"(상자가 보일 때 {np.median(dv[seen > 0]) if (seen > 0).any() else float('nan'):+.2f})", flush=True)
        print(f"[반응] 비평가 가치 2초/1초/0.5초/충돌 시점: " + " / ".join(f"{x:.1f}" for x in V.mean(0)), flush=True)
        PH = np.array([p_["Phit_2s_1s_05s_0s"] for p_ in probe])
        print(f"[반응] 충돌 예측 머리 P(20결정 안 충돌) 2초/1초/0.5초/충돌 시점 중앙값: " + " / ".join(f"{x:.2f}" for x in np.median(PH, 0)), flush=True)
        tot_s = {k: np.mean([p_["sal"][k][0] for p_ in probe]) for k in groups}
        tot_v = {k: np.mean([p_["sal"][k][1] for p_ in probe]) for k in groups}
        ss, sv = sum(tot_s.values()) + 1e-9, sum(tot_v.values()) + 1e-9
        print("[반응] 입력 x 기울기 몫(조향 / 속도): " + ", ".join(f"{k} {tot_s[k] / ss * 100:.0f}/{tot_v[k] / sv * 100:.0f} %" for k in groups),
              flush=True)
    gp = np.array([e["gap_beside"] for e in events if "gap_beside" in e])
    if gp.size:
        print(f"[폭] 상자 충돌 {gp.size}건: 상자 옆 남은 폭(넓은 쪽) 중앙값 {np.median(gp):.2f} m, 차 폭 0.27 m보다 좁은 경우 "
              f"{(gp < 0.27).mean() * 100:.0f} %, 0.27~0.5 m {((gp >= 0.27) & (gp < 0.5)).mean() * 100:.0f} %, 0.5 m 이상 "
              f"{(gp >= 0.5).mean() * 100:.0f} %", flush=True)
    if enc_seen is not None and enc_log:
        E_ = np.array(enc_log)
        print(f"[만남] 상자 {len(E_)}번 가까이 지나감, 그중 강한 충돌 {E_[:, 2].mean() * 100:.1f} %", flush=True)
        for lo, hi in ((0, 0.12), (0.12, 0.2), (0.2, 0.3), (0.3, 9)):
            m = (E_[:, 0] >= lo) & (E_[:, 0] < hi)
            if m.sum():
                print(f"   상자 크기(중심에서 가장 먼 면) {lo:.2f}~{hi:.2f} m: {m.sum()}번, 충돌률 {E_[m, 2].mean() * 100:.1f} %, "
                      f"지날 때 최고 속도 중앙 {np.median(E_[m, 1]):.1f} m/s", flush=True)
        for lo, hi in ((0, 3), (3, 5), (5, 7), (7, 20)):
            m = (E_[:, 1] >= lo) & (E_[:, 1] < hi)
            if m.sum():
                print(f"   지날 때 속도 {lo}~{hi} m/s: {m.sum()}번, 충돌률 {E_[m, 2].mean() * 100:.1f} %", flush=True)
        for lo, hi, nm in ((0, 0.4, "가운데"), (0.4, 0.7, "중간"), (0.7, 9, "벽 쪽")):
            m = (E_[:, 3] >= lo) & (E_[:, 3] < hi)
            if m.sum():
                mi = m & (E_[:, 2] < 0.5)
                print(f"   상자 위치 {nm}(옆 위치/반폭 {lo}~{hi}): {m.sum()}번 ({m.mean() * 100:.0f} %), 충돌률 {E_[m, 2].mean() * 100:.1f} %, "
                      f"안 박은 만남의 최소 간격 중앙 {np.median(E_[mi, 4]) if mi.sum() else float('nan'):.2f} m", flush=True)
        mv = E_[:, 5] > 0.5; mvn = E_[:, 6] > 0.05
        print(f"   이미 밀린 상자: {mv.sum()}번 ({mv.mean() * 100:.0f} %), 충돌률 {E_[mv, 2].mean() * 100 if mv.sum() else float('nan'):.1f} % | "
              f"안 밀린 상자: 충돌률 {E_[~mv, 2].mean() * 100:.1f} % | 지날 때 상자가 움직이는 중(>0.05 m/s): {mvn.sum()}번, "
              f"충돌률 {E_[mvn, 2].mean() * 100 if mvn.sum() else float('nan'):.1f} %", flush=True)
        for lo, hi in ((0, 1.5), (1.5, 4), (4, 999)):
            m = (E_[:, 7] >= lo) & (E_[:, 7] < hi)
            if m.sum() and (E_[:, 7] < 99).any():
                print(f"   지날 때 상대와 거리 {lo}~{hi} m: {m.sum()}번, 충돌률 {E_[m, 2].mean() * 100:.1f} %", flush=True)
        fast = E_[:, 1] >= 7
        for lo, hi, nm in ((0, 0.7, "가운데~중간"), (0.7, 9, "벽 쪽")):
            m = fast & (E_[:, 3] >= lo) & (E_[:, 3] < hi)
            if m.sum():
                print(f"   7 m/s 이상 + {nm}: {m.sum()}번, 충돌률 {E_[m, 2].mean() * 100:.1f} %", flush=True)
    od = np.array([e["opp_dist"] for e in events if e.get("opp_dist") == e.get("opp_dist")])
    if od.size:
        for kind in ("벽", "상자", "차"):
            k_ = np.array([e["opp_dist"] for e in events if e["kind"] == kind and e.get("opp_dist") == e.get("opp_dist")])
            if k_.size:
                print(f"[상대 거리] {kind} 충돌 {k_.size}건: 상대가 1 m 안 {(k_ < 1).mean() * 100:.0f} %, 1~3 m {((k_ >= 1) & (k_ < 3)).mean() * 100:.0f} %, "
                      f"3 m 밖 {(k_ >= 3).mean() * 100:.0f} %", flush=True)
        cc = [e for e in events if e["kind"] == "차" and "opp_fx" in e]
        if cc:
            fx = np.array([e["opp_fx"] for e in cc]); fy = np.array([e["opp_fy"] for e in cc])
            cl = np.array([e["closing"] for e in cc]); ah = np.array([e["opp_ahead_0.5s"] for e in cc])
            lm = np.array([e["opp_lat_move_0.5s"] for e in cc]); vq = np.array([e["opp_speed"] for e in cc])
            front = (fx > 0.15) & (np.abs(fy) < np.abs(fx) * 1.5)
            rear = (fx < -0.15) & (np.abs(fy) < np.abs(fx) * 1.5)
            side = ~front & ~rear
            print(f"[차 충돌 {len(cc)}건] 상대가 앞 {front.mean() * 100:.0f} % / 옆 {side.mean() * 100:.0f} % / 뒤 {rear.mean() * 100:.0f} % | "
                  f"내가 다가가는 속도 중앙 {np.median(cl):.2f} m/s | 0.5초 전 상대가 앞(트랙 따라) {(ah > 0).mean() * 100:.0f} % | "
                  f"상대 속도 중앙 {np.median(vq):.1f} m/s, 0.5초간 상대 옆 이동 |중앙| {np.median(np.abs(lm)):.2f} m", flush=True)
            for nm, m in (("앞", front), ("옆", side), ("뒤", rear)):
                if m.sum():
                    print(f"   상대 {nm} {m.sum()}건: 다가가는 속도 중앙 {np.median(cl[m]):.2f} m/s, 상대 속도 중앙 {np.median(vq[m]):.1f} m/s, "
                          f"0.5초간 상대 옆 이동 |중앙| {np.median(np.abs(lm[m])):.2f} m, 0.5초 전 상대가 앞 {(ah[m] > 0).mean() * 100:.0f} %", flush=True)
        t_ = np.array([e["ep_step"] / 40 for e in events])
        print(f"[출발 후] 1초 안 {(t_ < 1).mean() * 100:.0f} %, 1~3초 {((t_ >= 1) & (t_ < 3)).mean() * 100:.0f} %, 3초 뒤 {(t_ >= 3).mean() * 100:.0f} %", flush=True)
    by = {}
    for e in events:
        by[e["kind"]] = by.get(e["kind"], 0) + 1
    json.dump({"km": km, "counts": by, "events": events}, open(os.path.join(out, "events.json"), "w"))
    print(f"[장면] {os.path.basename(a.ckpt)} {a.stage}: {km:.1f} km에서 강한 충돌 {len(events)}회 "
          f"({', '.join(f'{k} {v}' for k, v in by.items())}) = {len(events) / max(km, 1e-3):.2f}/km, 그림 {n_img}장 -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
