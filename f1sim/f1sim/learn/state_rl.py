"""Privileged-state RL (rl-audit 2026-10-29): learn to race from the simulator's TRUE state, from scratch.

Why, in one line: every PPO run so far learned from a LiDAR scan through a plan that MAP mostly ignores, with 25 ms
decisions whose effect (2 cm) drowns in the return's noise -- the policies only traded speed for hits on one front
(J ~0.10). The fundamental change: give the learner what the outcome depends on, and let it act on the car directly.

* Observation (no scan): the v2 privileged vector (ego dynamics, offset and heading to the centerline, 3 nearest
  opponents with relative position / velocity, the nearest car's closing speed and time to contact, the 3 nearest props
  ahead, the race time left) + the track AHEAD: the centerline at fixed arc distances, in the car's frame, with the half
  width there + the steering angle + the last action. Running mean / std normalised.
* Action: the car's own command, steer and speed (`EnvConfig.action_mode "direct"`), held REPEAT control steps.
* Reward per decision: progress + steer-rate + wrong-way - PRICE x charged hard (>= --severity-min, 1.5 m/s) onsets (walls, props, cars):
  the objective every table reports.
* PPO, MLP actor-critic, many envs (no LiDAR to cast for the learner: --beams small).

Stages (--stage): solo (no props, no cars) -> props (props on the racing line, movable) -> race (+ planner opponents).
This policy is never deployed: the LiDAR policy learns it afterwards (teacher-student), so no localization on the car.
usage: python -m f1sim.learn.state_rl --name NAME [--stage solo] [--envs 4096] [--updates 600] [--init CKPT]"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .. import Config, maps
from ..gym_env import PRIV_V2_EGO_KEYS, REWARD_COMPONENT_KEYS, EnvConfig
from . import common

#: arc distances ahead [m] where the centerline is shown to the policy
AHEAD = (1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0)
KOREA = "real:korea_2026_competition"


def build_env(a, device, extra=None):
    # `--tracks` (11-01): the base maps, each with its --warps variants -- Korea alone left the plan teacher unable to
    # drive ICRA 2022 (laps 20-22 s, 26 wall touches a minute) or ICCAS (19 a minute) in the console
    bases = [t.strip() for t in str(getattr(a, "tracks", "") or KOREA).split(",") if t.strip()]
    tracks = [x for b in bases for x in [b] + ([f"{b}+warp{k}" for k in range(int(a.warps))] if a.warps else [])]
    # the racelines the planner opponents follow: the training world's (the cache key is every Raceline.build argument)
    limits = common.teacher_limits(7.0, 6.5, 4.0)
    trs, rls = common.load_tracks(tracks, racelines=(a.stage == "race" or bool(getattr(a, "path_plan", 0))),
                                  **dict(limits, objective="min_time"))
    cfg = Config()
    cfg.sim.compile = False
    cfg.lidar.n_beams = int(a.beams)
    cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0)
    cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
    e = dict(action_mode="direct", max_steps=int(a.episode_s * 40), speed_cap=10.0, v_max_policy=10.0,
             v_min_policy=float(getattr(a, "v_min", 0.0) or 0.0), reverse_ext=float(getattr(a, "reverse_ext", 0.0) or 0.0),
             collision_mode="soft", contact_severity_min=float(getattr(a, "severity_min", 1.5)), stagger_first_episode=True,
             reward_progress=1.0, reward_collision=-1.0, reward_car_contact=1.0, reward_steer_rate=0.02,
             reward_wrong_way=0.4, reward_proximity=0.0, reward_lap=0.0, reward_lap_time=0.0,
             spawn_runway=3.0, priv_props=True, priv_time=True, compile_tracker=False)
    if a.stage in ("props", "race"):
        e.update(procedural_obstacles=float(getattr(a, "prop_fraction", 1.0)), procedural_density=float(getattr(a, "prop_density", 1.5)),
                 procedural_max_props=int(getattr(a, "prop_max", 18)), procedural_count=str(getattr(a, "prop_count", "fixed") or "fixed"),
                 procedural_raceline_corridor="off", movable_obstacles=bool(int(getattr(a, "prop_movable", 1))),
                 procedural_line_share=float(getattr(a, "prop_line", 0.0) or 0.0) if a.stage == "props" else 0.0,
                 procedural_place=str(getattr(a, "prop_place", "pattern") or "pattern"),
                 procedural_bank=str(getattr(a, "prop_bank", "") or ""),
                 procedural_shapes=str(getattr(a, "prop_shapes", "all") or "all"),
                 spawn_behind_prop=float(getattr(a, "spawn_behind_prop", 0.0) or 0.0))
    if a.stage == "race":
        # the competition's format: 1 vs 1, the four planner kinds at 0.7-1.05 of their pace, reacting and with events
        # (the training world's slot without the league: league entries read a full scan, this env casts --beams)
        slot = {"kind_mix": ["forzaeth", "forzaeth_pred", "lane_switch", "interactive"], "speed_scale": [0.7, 1.05],
                "present": float(getattr(a, "opp_present", 1.0)), "spawn": str(getattr(a, "opp_spawn", "random")),
                "reactive": {"defend": 0.3, "line": 0.2, "oblivious": 0.1},
                "events": ["brake", "stop", "shift", "weave"], "event_rate": 0.5}
        slot.update(json.loads(getattr(a, "opp_slot", "") or "{}"))          # `--opp-slot`: a stage's opponents
        from .. import opponent_slots as opp_sl
        e.update(race_size=2, opponent="slots", opponent_slots=opp_sl.parse_slots(json.dumps([slot])))
    e.update(plan_env_fields(a))
    e.update(extra or {})
    env = common.make_env(trs, int(a.envs), device, EnvConfig(**e), cfg=cfg, seed=int(a.seed), rls=rls,
                          teacher_limits=limits)
    apply_plan_spec(env, a)
    env._track_names = list(tracks); env._rls = rls          # `--path-plan`: the planner's reference lines
    if env.procedural is not None and env.procedural.line_share > 0:
        # `--prop-line`: the min-time raceline's offset at every centerline point of every track, then fresh layouts
        tr = env.sim.track
        T = int(tr.cl_s.shape[0])
        tid = torch.arange(T, device=device)[:, None].expand_as(tr.cl_s)
        env.procedural.set_line(_line_planner(env).rl_offset(tr.cl_s, tid))
        env.reset()
    return env


#: `--corridor-obs`: 7 lateral bins across the track (left positive), each 2 samples wide, over 13 arc segments ahead
CORRIDOR_U = tuple(float(x) for x in np.linspace(-0.93, 0.93, 14))
CORRIDOR_BINS = 7
#: dense arc offsets [m] the segments are swept with: every 0.4 m to 26 m (10-29 check: at 1 m a small prop fell between
#: two rows)
CORRIDOR_S = tuple(float(x) for x in np.arange(0.4, 26.01, 0.4))


def corridor_block(env, lid, s, tid, inflate=0.15, cars=False, props=True, visible=None):
    """(B, 13 * 7) in {0, 1}: for each arc segment between consecutive track-ahead points (AHEAD; the first from the car
    to 1 m), which lateral bins across the track a live prop (inflated by the car's half width) covers anywhere along the
    segment -- the free corridor the props leave, in the frame of the centerline points the policy steers by (10-29:
    the hit prop was outside the 3 nearest-prop slots 1 s before 24 % of the teacher's prop hits; in the scenes the
    policy kept its line through apex props). Swept every 0.4 m, so no prop falls between rows. `visible` (B, C) bool:
    only those props (`visible_props`, 11-03)."""
    po = env.procedural
    tr = env.sim.track
    B = s.shape[0]
    n_seg = len(AHEAD)
    if ((po is None or po.C == 0) or not props) and not (cars and env.M > 1):
        return torch.zeros(B, n_seg * CORRIDOR_BINS, device=s.device)
    offs = torch.tensor(CORRIDOR_S, device=s.device)
    seg = torch.bucketize(offs, torch.tensor(AHEAD, device=s.device))     # (R,) segment of each dense row
    R = offs.numel()
    xy, yaw = tr.pose_at_s((s[:, None] + offs[None, :]).reshape(-1), tid.repeat_interleave(R))
    xy = xy.reshape(B, R, 2); yaw = yaw.reshape(B, R)
    w = tr.sample_edt(xy, tid[:, None].expand(-1, R))                      # (B, R) half width
    U = torch.tensor(CORRIDOR_U, device=s.device)
    nx, ny = -torch.sin(yaw), torch.cos(yaw)
    off = U[None, None, :] * w[..., None]                                 # (B, R, M)
    px = xy[..., 0:1] + off * nx[..., None]; py = xy[..., 1:2] + off * ny[..., None]
    blocked = torch.zeros(B, R, U.numel(), dtype=torch.bool, device=s.device)
    n_slots = po.C if (po is not None and props) else 0
    if n_slots:
        pp_all = po.p_poses[lid]; live_all = po.p_sid[lid] >= 0
        if visible is not None:
            live_all = live_all & visible
        nrm_all, dd_all = po.p_n[lid], po.p_d[lid]
    for c in range(n_slots):                                                 # one prop slot at a time: bounded memory
        live = live_all[:, c]
        if not bool(live.any()):
            continue
        pp = pp_all[:, c]
        dx = px - pp[:, None, None, 0]; dy = py - pp[:, None, None, 1]
        cc, ss = torch.cos(pp[:, 2])[:, None, None], torch.sin(pp[:, 2])[:, None, None]
        lx, ly = dx * cc + dy * ss, -dx * ss + dy * cc
        nrm, dd = nrm_all[:, c], dd_all[:, c]                             # (B, k, 2), (B, k)
        fin = torch.isfinite(dd)
        sd = (nrm[:, None, None, :, 0] * lx[..., None] + nrm[:, None, None, :, 1] * ly[..., None]
              - torch.where(fin, dd, torch.zeros_like(dd))[:, None, None, :])
        sd = torch.where(fin[:, None, None, :], sd, torch.full_like(sd, -1e9)).amax(-1)        # (B, R, M)
        blocked |= (sd < inflate) & live[:, None, None]
    if cars and env.M > 1:
        # the other cars in the same corridor (`--corridor-cars`, 10-29): a disc of 0.30 m (+ our half width) at each car's
        # pose now and where its velocity takes it in 0.5 s and 1 s -- the rear-ends were of slower / stopped cars straight
        # ahead while the policy steers by this channel and barely by the opponent block
        st_all = env.last_result.state
        o = env.sim.other_idx[lid]                                         # (B, M-1)
        so = st_all[o]
        co, si_ = torch.cos(so[..., 2]), torch.sin(so[..., 2])
        vx = so[..., 3] * co - so[..., 4] * si_; vy = so[..., 3] * si_ + so[..., 4] * co
        pres = (~env.sim.absent[o]) if env.sim.absent_any else torch.ones_like(vx, dtype=torch.bool)
        for tau in (0.0, 0.5, 1.0):
            cx = so[..., 0] + vx * tau; cy = so[..., 1] + vy * tau          # (B, M-1)
            d2 = (px[:, None] - cx[..., None, None]) ** 2 + (py[:, None] - cy[..., None, None]) ** 2
            blocked |= ((d2 < (0.30 + inflate) ** 2) & pres[..., None, None]).any(1)
    bins = blocked.reshape(B, R, CORRIDOR_BINS, -1).any(-1)                # (B, R, 7)
    out = torch.zeros(B, n_seg, CORRIDOR_BINS, dtype=torch.bool, device=s.device)
    for k in range(n_seg):
        m = seg == k
        if bool(m.any()):
            out[:, k] = bins[:, m].any(1)
    return out.float().reshape(B, -1)


def _line_speed(env):
    """the min-time raceline's speed profile per track on a 0.1 m s-grid (`PathPlanner.rl_speed`), built once per env"""
    lp = getattr(env, "_line_speed_pl", None)
    if lp is None:
        from .path_plan import PathPlanner
        rls = getattr(env, "_rls", None)
        if rls is None:
            names = getattr(env, "_track_names", None) or [KOREA]
            _, rls = common.load_tracks(names, racelines=True, **dict(common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
        lp = PathPlanner(env, env.learner_ids, rls)
        env._line_speed_pl = lp
    return lp.rl_speed


def _line_planner(env):
    """the PathPlanner `_line_speed` builds (the min-time raceline per track: its speed and its offset from the centerline)"""
    _line_speed(env)
    return env._line_speed_pl


def _arc_time(T_ext, x, g, v_first, v_last):
    """the time [s] a car at arc gap g needs to reach arc x (B, R'): `T_ext` (B, R + 1) its times at 0, 0.4, ..., 0.4 R;
    before the grid at the first row's speed (0 for an x it is already past), beyond it at the last row's"""
    R1 = T_ext.shape[1]
    pos = x / 0.4
    j0 = pos.floor().clamp(0, R1 - 2).long()
    w_ = (pos - j0.float()).clamp(0.0, 1.0)
    T = T_ext.gather(1, j0) * (1 - w_) + T_ext.gather(1, j0 + 1) * w_
    T = torch.where(x < 0, (x - g).clamp_min(0.0) / v_first, T)
    return torch.where(x > 0.4 * (R1 - 1), T_ext[:, -1:] + (x - 0.4 * (R1 - 1)) / v_last, T)


def car_spacetime_block(env, lid, s, tid, a_acc=5.0, v_floor=1.0, half_len=0.55, half_wid=0.35, grow=0.10, line=False,
                        visible=None):
    """(B, 13 * 7) in {0, 1} (`--car-st`, 10-31): the other cars in the corridor grid at the time WE would get there if we
    drove our fast line: row d (every 0.4 m ahead) is reached at tau(d) = sum of 0.4 m / v_plan, v_plan = min(the raceline's
    speed there, sqrt(v^2 + 2 a_acc d)) (from our speed now, at most `a_acc`), at least `v_floor`; a car with arc gap g
    (signed, + ahead), along-track
    speed v_o and lateral offset l blocks the row's lateral samples within `half_wid` of l when |g + v_o tau - d| <
    `half_len` + `grow` tau (its prediction, along the track, gets less sure with time). A car pulling away blocks nothing;
    a slower car blocks the rows where we would reach it, the lanes beside it stay free -- the same picture a prop gives.
    (10-31: the static car channel -- the car now / +0.5 s / +1 s -- and the car in the scan made the race policy brake to
    0 m/s for a car 5.6 m ahead closing at 0.8 m/s, 18 % of the time 4-10 m behind a car, and never close in to pass.)

    `line` (`--car-st-line`, 11-02): the car drives the raceline at its present fraction of the line's speed and keeps its
    present offset from the line, instead of holding its speed and its offset from the centerline. 11-02: a props policy
    reading this block in its corridor passed a 0.6x car 20 races in 20 but hit it hard 15 times, every one at a hairpin --
    it took the lane beside the car, and the car, on the raceline to the metre (0.01-0.07 m off it), followed the line
    0.9 m across into that lane within 2 m of arc while braking to half the line's speed."""
    B = s.shape[0]
    n_seg = len(AHEAD)
    out = torch.zeros(B, n_seg, CORRIDOR_BINS, device=s.device)
    if env.M <= 1:
        return out.reshape(B, -1)
    tr = env.sim.track
    st_all = env.last_result.state
    o = env.sim.other_idx[lid]                                             # (B, M-1)
    pres = (~env.sim.absent[o]) if env.sim.absent_any else torch.ones(o.shape, dtype=torch.bool, device=s.device)
    if visible is not None:                                                # `--vis-actor`: the cars the car can see
        pres = pres & visible
    g = env.signed_gaps(env.last_result.s, env.sim.tid)[lid]               # (B, M-1), + = ahead of us
    so = st_all[o]
    s_o = env.last_result.s[o]
    xy_c, yaw_c = tr.pose_at_s(s_o.reshape(-1), env.sim.tid[o].reshape(-1))
    xy_c = xy_c.reshape(*o.shape, 2); yaw_c = yaw_c.reshape(o.shape)
    l_o = -(so[..., 0] - xy_c[..., 0]) * torch.sin(yaw_c) + (so[..., 1] - xy_c[..., 1]) * torch.cos(yaw_c)   # left +
    v_o = so[..., 3] * torch.cos(torch.remainder(so[..., 2] - yaw_c + math.pi, 2 * math.pi) - math.pi)
    offs = torch.tensor(CORRIDOR_S, device=s.device)
    seg = torch.bucketize(offs, torch.tensor(AHEAD, device=s.device))
    R = offs.numel()
    xy, yaw = tr.pose_at_s((s[:, None] + offs[None, :]).reshape(-1), tid.repeat_interleave(R))
    w = tr.sample_edt(xy.reshape(B, R, 2), tid[:, None].expand(-1, R))      # (B, R) half width
    U = torch.tensor(CORRIDOR_U, device=s.device)
    lat = U[None, None, :] * w[..., None]                                  # (B, R, 14) lateral of each sample, left +
    v0 = st_all[lid, 3].clamp_min(0.0)[:, None]
    v_line = _line_speed(env)(s[:, None] + offs[None, :], tid[:, None])  # (B, R)
    v_plan = torch.minimum(v_line, torch.sqrt(v0 ** 2 + 2 * a_acc * offs[None, :])).clamp_min(v_floor)
    tau = torch.cumsum(0.4 / v_plan, 1)                                   # (B, R) [s] to reach each row
    blocked = torch.zeros(B, R, U.numel(), dtype=torch.bool, device=s.device)
    if line:
        pl = _line_planner(env)
        tid_o = env.sim.tid[o]
        r_o = (v_o / pl.rl_speed(s_o, tid_o).clamp_min(0.5)).clamp(0.0, 1.3)          # (B, M-1) its pace on the line
        dl_o = l_o - pl.rl_offset(s_o, tid_o)                                          # (B, M-1) its offset from it
        l_line = pl.rl_offset(s[:, None] + offs[None, :], tid[:, None].expand(-1, R))  # (B, R) the line at each row
    for q in range(o.shape[1]):
        if line:
            gq = g[:, q:q + 1]
            v_pred = (v_line * r_o[:, q:q + 1]).clamp_min(0.05)                         # (B, R) its speed at each row
            ell = (offs[None, :] - torch.maximum(offs[None, :] - 0.4, gq)).clamp_min(0.0)   # its stretch of each cell
            T0 = (-gq).clamp_min(0.0) / v_pred[:, :1]                                    # behind us: up to our arc
            T_ext = torch.cat([T0, T0 + torch.cumsum(ell / v_pred, 1)], 1)               # (B, R + 1) at 0, 0.4, ...
            h = half_len + grow * tau
            t_in = _arc_time(T_ext, offs[None, :] - h, gq, v_pred[:, :1], v_pred[:, -1:])
            t_out = _arc_time(T_ext, offs[None, :] + h, gq, v_pred[:, :1], v_pred[:, -1:])
            along = (t_in <= tau) & (tau <= t_out)                                       # its centre within h of the row
            side = (lat - (l_line + dl_o[:, q:q + 1])[..., None]).abs() < half_wid
        else:
            along = (g[:, q:q + 1] + v_o[:, q:q + 1] * tau - offs[None, :]).abs() < half_len + grow * tau      # (B, R)
            side = (lat - l_o[:, q, None, None]).abs() < half_wid                                              # (B, R, 14)
        blocked |= along[..., None] & side & pres[:, q, None, None]
    bins = blocked.reshape(B, R, CORRIDOR_BINS, -1).any(-1)
    for k in range(n_seg):
        m = seg == k
        if bool(m.any()):
            out[:, k] = bins[:, m].any(1).float()
    return out.reshape(B, -1)


#: `--bev` (10-30): the car-centred grid -- forward x in [BEV_X0, BEV_X0 + BEV_H * BEV_RES), lateral y (left +) in
#: [-BEV_W * BEV_RES / 2, +BEV_W * BEV_RES / 2), cells of BEV_RES m, each the mean of 2 x 2 samples
BEV_X0, BEV_RES, BEV_H, BEV_W = -2.0, 0.25, 56, 28
#: channels: walls, props, cars, the cars' velocity relative to ours along our x and y [/5 m/s]
BEV_C = 5


def bev_grid(env, lid, margin=0.125):
    """(B, BEV_C, BEV_H, BEV_W): what is around the car, in its own frame, from the TRUE geometry -- one representation
    for walls, props and cars instead of centerline-frame features (10-30: the prop corridor, built on the centerline,
    told a car standing beside the line to steer into the prop it was wedged on). Occupancy is soft: 1 inside, falling
    to 0 at `margin` outside, so a thin wall or a small prop between two samples still shows."""
    res = env.last_result
    st = res.state[lid]
    B = st.shape[0]
    dev = st.device
    k = 2
    fx = BEV_X0 + (torch.arange(BEV_H * k, device=dev) + 0.5) * (BEV_RES / k)
    fy = (torch.arange(BEV_W * k, device=dev) + 0.5) * (BEV_RES / k) - BEV_W * BEV_RES / 2
    gx, gy = torch.meshgrid(fx, fy, indexing="ij")                     # (Hk, Wk) car frame
    gx = gx.reshape(-1); gy = gy.reshape(-1); P = gx.numel()
    c, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
    px = st[:, 0:1] + gx[None] * c - gy[None] * s_                       # (B, P) world
    py = st[:, 1:2] + gx[None] * s_ + gy[None] * c
    tid = env.sim.tid[lid]
    edt = env.sim.track.sample_edt(torch.stack([px, py], -1), tid[:, None].expand(-1, P))
    walls = (1.0 - edt / margin).clamp(0.0, 1.0)
    props = torch.zeros(B, P, device=dev)
    po = env.procedural
    if po is not None and po.C > 0:
        pp_all = po.p_poses[lid]; live_all = po.p_sid[lid] >= 0
        nrm_all, dd_all = po.p_n[lid], po.p_d[lid]
        for ci in range(po.C):
            live = live_all[:, ci]
            if not bool(live.any()):
                continue
            pp = pp_all[:, ci]
            dx = px - pp[:, None, 0]; dy = py - pp[:, None, 1]
            cc, ss = torch.cos(pp[:, 2])[:, None], torch.sin(pp[:, 2])[:, None]
            lx, ly = dx * cc + dy * ss, -dx * ss + dy * cc
            nrm, dd = nrm_all[:, ci], dd_all[:, ci]                         # (B, k, 2), (B, k)
            fin = torch.isfinite(dd)
            sd = nrm[:, None, :, 0] * lx[..., None] + nrm[:, None, :, 1] * ly[..., None] - torch.where(fin, dd, torch.zeros_like(dd))[:, None, :]
            sd = torch.where(fin[:, None, :], sd, torch.full_like(sd, -1e9)).amax(-1)              # (B, P)
            props = torch.maximum(props, (1.0 - sd.clamp_min(0.0) / margin).clamp(0.0, 1.0) * live[:, None].float())
    cars = torch.zeros(B, P, device=dev); cvx = torch.zeros(B, P, device=dev); cvy = torch.zeros(B, P, device=dev)
    if env.M > 1 and env.sim.other_idx is not None:
        so_all = res.state[env.sim.other_idx[lid]]                         # (B, M-1, 7+)
        pres = (~env.sim.absent[env.sim.other_idx[lid]]) if env.sim.absent_any else torch.ones(so_all.shape[:2], dtype=torch.bool, device=dev)
        hl, hw = 0.5 * float(env.cfg.vehicle.length), 0.5 * float(env.cfg.vehicle.width)
        # our own velocity in the world frame, to make the others' relative
        vxe = st[:, 3] * c[:, 0] - st[:, 4] * s_[:, 0]; vye = st[:, 3] * s_[:, 0] + st[:, 4] * c[:, 0]
        for q in range(so_all.shape[1]):
            so = so_all[:, q]
            dx = px - so[:, None, 0]; dy = py - so[:, None, 1]
            cq, sq = torch.cos(so[:, 2])[:, None], torch.sin(so[:, 2])[:, None]
            lx, ly = dx * cq + dy * sq, -dx * sq + dy * cq
            out_ = torch.maximum(lx.abs() - hl, ly.abs() - hw).clamp_min(0.0)
            occ = (1.0 - out_ / margin).clamp(0.0, 1.0) * pres[:, q, None].float()
            vxw = so[:, 3] * cq[:, 0] - so[:, 4] * sq[:, 0]; vyw = so[:, 3] * sq[:, 0] + so[:, 4] * cq[:, 0]
            rvx, rvy = vxw - vxe, vyw - vye
            rx = (rvx * c[:, 0] + rvy * s_[:, 0]) / 5.0; ry = (-rvx * s_[:, 0] + rvy * c[:, 0]) / 5.0   # in our frame
            new = occ > cars
            cvx = torch.where(new, occ * rx[:, None], cvx); cvy = torch.where(new, occ * ry[:, None], cvy)
            cars = torch.maximum(cars, occ)
    g = torch.stack([walls, props, cars, cvx, cvy], 1).reshape(B, BEV_C, BEV_H, k, BEV_W, k).mean((3, 5))
    return g


class RacePosition:
    """`--pos-bonus` (11-02): the race's order in the reward, zero-sum -- +1 the step this car takes a place from a car
    within `rng` m (from more than `hyst` m behind it to more than `hyst` m ahead), -1 the step it loses one. A side is only
    known while the other car is within `rng` m: out of range it is forgotten (no change is paid), and it is set without
    pay on entering range or after a reset -- so a spawn, a lap of the track and a car that falls out of range pay nothing,
    and handing a place back to take it again pays nothing either. 11-02: with progress and the hit price alone a pass of
    a 0.6x car was worth ~22 (10 s horizon) against ~24 of expected hits at 0.3 hard hits per pass, and every race-stage
    run (raceb, racec, racew) erased passing between update 50 and 100 while it kept following safely."""

    def __init__(self, env, lid, hyst=1.0, rng=8.0, void_s=0.0):
        self.env, self.lid, self.hyst, self.rng, self.void_s = env, lid, float(hyst), float(rng), float(void_s)
        self.side = torch.zeros(int(lid.numel()), max(1, env.M - 1), dtype=torch.long, device=lid.device)
        self.t_hit = torch.full((int(lid.numel()),), 1e9, device=lid.device)

    def step(self, hit=None):
        """(gained places that pay (B,), lost (B,), gained (B,)) this step. `hit` (B,): this car's hard car contact this
        step -- with `void_s` > 0 a place taken within `void_s` s of one does not pay (`--pos-void`, 11-02: at +80 a pass
        and -80 a hard hit, shoving a same-pace car off at a hairpin exit and driving past it came out even plus the time
        it lost -- racep u100 passed a 1.0x car in 13 races of 20 with 12 hard hits; a race gives no place for a contact)"""
        env = self.env
        z = torch.zeros(self.side.shape[0], device=self.side.device)
        if hit is not None:
            self.t_hit = torch.where(hit, torch.zeros_like(self.t_hit), self.t_hit + env.sim.control_dt)
        if env.M <= 1:
            return z, z, z
        g = env.signed_gaps(env.last_result.s, env.sim.tid)[self.lid]              # (B, M-1), + = that car ahead
        o = env.sim.other_idx[self.lid]
        pres = (~env.sim.absent[o]) if env.sim.absent_any else torch.ones_like(g, dtype=torch.bool)
        lead = -g
        new = torch.where(lead > self.hyst, torch.ones_like(self.side),
                          torch.where(lead < -self.hyst, -torch.ones_like(self.side), self.side))
        new = torch.where((g.abs() < self.rng) & pres, new, torch.zeros_like(new))
        gained = ((self.side == -1) & (new == 1)).sum(1).float()
        lost = ((self.side == 1) & (new == -1)).sum(1).float()
        self.side = new
        paid = gained * (self.t_hit > self.void_s).float() if self.void_s > 0 else gained
        return paid, lost, gained

    def reset(self, d):
        self.side[d] = 0
        self.t_hit[d] = 1e9


#: [m] the LiDAR student's grid (`lidar_grid`, `BEV_*`): what a prop must be inside to be something the car can see
VIS_X0, VIS_X1, VIS_Y = BEV_X0, BEV_X0 + BEV_H * BEV_RES, BEV_W * BEV_RES / 2


def visible_props(env, lid, tol=0.20):
    """(B, C) bool (`--vis-actor`, 11-03): the live props the car itself can see now -- inside the LiDAR student's grid
    (x in [VIS_X0, VIS_X1], |y| <= VIS_Y in the car frame), within the LiDAR's range, and in line of sight: the noise-free
    scan at the bearing of the prop's centre or of one of four points at its extent (its largest half-plane offset)
    reaches that point (range >= distance - `tol`); a wall or another prop in front cuts the beam short. The privileged
    teacher's prop inputs read only these, so it reacts to nothing the student cannot see (11-03, the user: no
    privileged input far from what the student has -- the corridor reached 26 m along the track, round hairpins and
    behind walls)."""
    po = env.procedural
    if po is None or po.C == 0:
        return None
    res = env.last_result
    st = res.state[lid]
    B, C = st.shape[0], po.C
    pp = po.p_poses[lid]                                                  # (B, C, 3)
    dd = po.p_d[lid]
    ext = torch.where(torch.isfinite(dd), dd, torch.zeros_like(dd)).amax(-1).clamp_min(0.05)   # (B, C)
    cp, sp = torch.cos(pp[..., 2]), torch.sin(pp[..., 2])
    off = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=st.device)   # (5, 2)
    ox = off[None, None, :, 0] * ext[..., None]; oy = off[None, None, :, 1] * ext[..., None]
    px = pp[..., 0:1] + ox * cp[..., None] - oy * sp[..., None]           # (B, C, 5) world
    py = pp[..., 1:2] + ox * sp[..., None] + oy * cp[..., None]
    c, s_ = torch.cos(st[:, 2])[:, None, None], torch.sin(st[:, 2])[:, None, None]
    dx, dy = px - st[:, 0, None, None], py - st[:, 1, None, None]
    bx, by = dx * c + dy * s_, -dx * s_ + dy * c                          # car frame
    inwin = (bx >= VIS_X0) & (bx <= VIS_X1) & (by.abs() <= VIS_Y)
    mx, my, ang = env._lidar_pose()
    ang = ang.expand(env.B, -1)
    mx, my, ang = mx[lid], my[lid], ang[lid].float()                      # (B,), (B,), (B, N)
    lx, ly = bx - mx[:, None, None], by - my[:, None, None]
    d = torch.hypot(lx, ly); th = torch.atan2(ly, lx)
    N = ang.shape[1]
    a0, a1 = ang[:, :1, None], ang[:, -1:, None]
    pos = (th - a0) / (a1 - a0) * (N - 1)                                 # (B, C, 5) fractional beam index
    infov = (pos >= 0) & (pos <= N - 1)
    i0 = pos.floor().clamp(0, N - 2).long(); w = (pos - i0.float()).clamp(0, 1)
    sc = res.scan_true[lid].float()                                       # (B, N) [m]
    r0 = sc.gather(1, i0.reshape(B, -1)).reshape(B, C, 5); r1 = sc.gather(1, (i0 + 1).reshape(B, -1)).reshape(B, C, 5)
    r = r0 * (1 - w) + r1 * w
    rmax = float(env.range_max)
    seen = infov & inwin & (d < rmax) & (r >= d - tol)
    return (po.p_sid[lid] >= 0) & seen.any(-1)


def visible_cars(env, lid, tol=0.20):
    """(B, M-1) bool (`--vis-actor`, 11-03): the other cars (in `sim.other_idx` order) the car itself can see now -- as
    `visible_props`: a point of the car (its centre or a corner of its footprint) inside the LiDAR student's grid, within
    the LiDAR's range, and the noise-free scan at its bearing reaches it (the scan sees cars; a wall or a prop in front
    cuts it short). None in a solo env."""
    if env.M <= 1:
        return None
    res = env.last_result
    st_all = res.state
    st = st_all[lid]
    o = env.sim.other_idx[lid]                                             # (B, M-1)
    so = st_all[o]
    B, Q = o.shape
    hl, hw = 0.5 * float(env.cfg.vehicle.length), 0.5 * float(env.cfg.vehicle.width)
    off = torch.tensor([[0.0, 0.0], [hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]], device=st.device)   # (5, 2)
    co, si = torch.cos(so[..., 2])[..., None], torch.sin(so[..., 2])[..., None]
    px = so[..., 0:1] + off[None, None, :, 0] * co - off[None, None, :, 1] * si     # (B, Q, 5) world
    py = so[..., 1:2] + off[None, None, :, 0] * si + off[None, None, :, 1] * co
    c, s_ = torch.cos(st[:, 2])[:, None, None], torch.sin(st[:, 2])[:, None, None]
    dx, dy = px - st[:, 0, None, None], py - st[:, 1, None, None]
    bx, by = dx * c + dy * s_, -dx * s_ + dy * c
    inwin = (bx >= VIS_X0) & (bx <= VIS_X1) & (by.abs() <= VIS_Y)
    mx, my, ang = env._lidar_pose()
    ang = ang.expand(env.B, -1)
    mx, my, ang = mx[lid], my[lid], ang[lid].float()
    lx, ly = bx - mx[:, None, None], by - my[:, None, None]
    d = torch.hypot(lx, ly); th = torch.atan2(ly, lx)
    N = ang.shape[1]
    a0, a1 = ang[:, :1, None], ang[:, -1:, None]
    pos = (th - a0) / (a1 - a0) * (N - 1)
    infov = (pos >= 0) & (pos <= N - 1)
    i0 = pos.floor().clamp(0, N - 2).long(); w = (pos - i0.float()).clamp(0, 1)
    sc = res.scan_true[lid].float()
    r0 = sc.gather(1, i0.reshape(B, -1)).reshape(B, Q, 5); r1 = sc.gather(1, (i0 + 1).reshape(B, -1)).reshape(B, Q, 5)
    r = r0 * (1 - w) + r1 * w
    seen = infov & inwin & (d < float(env.range_max)) & (r >= d - tol)
    pres = (~env.sim.absent[o]) if env.sim.absent_any else torch.ones(o.shape, dtype=torch.bool, device=st.device)
    return pres & seen.any(-1)


def scan_point_masks(env, lid, sc, typ, pad=0.05):
    """(prop (B, N), car (B, N)) bool (`--scan-points`, 11-06): which beams of the noise-free scan `sc` (B, N) [m] ended on a
    prop / on another car. Cars from the scan's own hit type (`lidar.HIT_CAR`); props report as HIT_TALL like a tall wall
    (`Lidar._merge_props`), so a HIT_TALL beam is a prop's when its end point lies within a live prop's bounding circle
    (its largest half-plane offset) + `pad`. The beams in the order of `env.sim.lidar.angles` (the env's own cast)."""
    from ..lidar import HIT_CAR, HIT_TALL
    car = typ == HIT_CAR
    prop = torch.zeros_like(car)
    po = env.procedural
    if po is None or po.C == 0:
        return prop, car
    st = env.last_result.state[lid]
    mx, my, ang = env._lidar_pose()
    ang = ang.expand(env.B, -1)[lid].float()
    ex = mx[lid, None] + sc * torch.cos(ang); ey = my[lid, None] + sc * torch.sin(ang)          # end points, car frame
    c, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
    wx = st[:, 0:1] + ex * c - ey * s_; wy = st[:, 1:2] + ex * s_ + ey * c                     # world
    pp = po.p_poses[lid]; live = po.p_sid[lid] >= 0
    dd = po.p_d[lid]
    rad = torch.where(torch.isfinite(dd), dd, torch.zeros_like(dd)).amax(-1).clamp_min(0.05) + pad   # (B, C)
    tall = typ == HIT_TALL
    for j in range(po.C):                                                  # one prop slot at a time: bounded memory
        lj = live[:, j]
        if not bool(lj.any()):
            continue
        d2 = (wx - pp[:, j, 0:1]).square() + (wy - pp[:, j, 1:2]).square()
        prop |= tall & lj[:, None] & (d2 <= rad[:, j:j + 1].square())
    return prop, car


def grow_to_scan_points(sd, nm, scan0, b0, b1, vis0, D, device):
    """An --init without `--scan-points` moved to this run's columns (11-06), so update 1 drives (nearly) as the parent:
    the first layers' (pi.0, v.0) columns before the scan as they were; each of the b1 new scan beams takes the weight of
    the parent's nearest beam over the number of new beams sharing it (the parent's response to the scan, smoothed over
    its beam's sector) and its input statistics; the prop / car point rows start with zero weights (statistics: the
    range cap 1.0, var 0.05); the columns between the scan and the visible block as they were; the parent's three visible
    prop slots (`visible_prop_slots`, PRIV_V2_PROP_DIM columns at its vis0) dropped; the rest of the visible block as it was."""
    from ..gym_env import PRIV_V2_PROP_DIM
    mid = vis0 - (scan0 + 3 * b1)                                          # corridor (+ car channel) between scan and vis
    vis0_old = scan0 + b0 + mid
    tail = D - vis0
    d0 = int(sd["pi.0.weight"].shape[1])
    if d0 != vis0_old + PRIV_V2_PROP_DIM + tail:
        raise SystemExit(f"--init: {d0} columns, not the expected parent layout ({vis0_old} + {PRIV_V2_PROP_DIM} + {tail})")
    near = torch.round(torch.arange(b1, device=device, dtype=torch.float32) * (b0 - 1) / max(b1 - 1, 1)).long()
    share = torch.bincount(near, minlength=b0).float()
    src = torch.cat([torch.arange(scan0, device=device), scan0 + near])     # old column of each new pre-scan / scan column
    def move(x, new_fill, scale=None):
        # x (..., d0) -> (..., D)
        pre = x[..., src]
        if scale is not None:
            pre = torch.cat([pre[..., :scan0], pre[..., scan0:] / scale], -1)
        pts = new_fill.expand(*x.shape[:-1], 2 * b1).to(x.dtype)
        midc = x[..., scan0 + b0: vis0_old]
        rest = x[..., vis0_old + PRIV_V2_PROP_DIM:]
        return torch.cat([pre, pts, midc, rest], -1)
    for k in ("pi.0.weight", "v.0.weight"):
        sd[k] = move(sd[k].to(device), torch.zeros(1, device=device), scale=share[near])
    nm = {"mean": move(nm["mean"].to(device), torch.ones(1, device=device)),
          "var": move(nm["var"].to(device), torch.full((1,), 0.05, device=device)), "n": nm["n"]}
    return sd, nm


def visible_prop_slots(env, lid, visible):
    """(B, PRIV_V2_PROP_DIM): the privileged prop slots (`GymEnv._prop_block`: present, dx/5, dy/5, dist/5, nearest first,
    dx > -1 m, within PRIV_V2_PROP_RANGE) over the `visible` props only"""
    from ..gym_env import PRIV_V2_PROP_KEYS, PRIV_V2_PROP_RANGE, PRIV_V2_PROP_SLOTS, PRIV_OPP_DIST_SCALE
    st = env.last_result.state[lid]
    B = st.shape[0]
    K, S = PRIV_V2_PROP_SLOTS, PRIV_OPP_DIST_SCALE
    blk = torch.zeros(B, K, len(PRIV_V2_PROP_KEYS), device=st.device, dtype=st.dtype)
    po = env.procedural
    if po is None or po.C == 0 or visible is None:
        return blk.reshape(B, -1)
    d = po.p_poses[lid][:, :, :2].to(st.dtype) - st[:, None, :2]
    c, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
    dx, dy = d[..., 0] * c + d[..., 1] * s_, -d[..., 0] * s_ + d[..., 1] * c
    dist = d.norm(dim=-1)
    live = visible & (dx > -1.0) & (dist < PRIV_V2_PROP_RANGE)
    key = torch.where(live, dist, torch.full_like(dist, 1e6))
    n = min(K, key.shape[1])
    j = key.topk(n, dim=1, largest=False).indices
    g = lambda t: t.gather(1, j)
    pres = g(live).to(st.dtype)
    blk[:, :n] = torch.stack([pres, g(dx) / S, g(dy) / S, g(dist) / S], 2) * pres[..., None]
    return blk.reshape(B, -1)


def visible_car_slots(env, lid, visible, K=3):
    """(B, K * 6) `--car-slots` (11-03): the other cars the car itself can see (`visible_cars`: inside the LiDAR student's
    grid, which reaches 2 m behind and 3.5 m to each side, and in the scan's line of sight), nearest first --
    present, dx/5, dy/5, dist/5 and their velocity relative to ours, dvx/5, dvy/5, all in our frame. The corridor's car
    channel starts 0.4 m ahead along the track, so a car beside and just behind us (the one we cut in front of after a
    pass: 3 of 8 hard hits of srp25h3gwt_race u50, `car_cf.py`) was only in the raw scan; the student's grid has it."""
    from ..gym_env import PRIV_OPP_DIST_SCALE
    st_all = env.last_result.state
    st = st_all[lid]
    B = st.shape[0]
    blk = torch.zeros(B, K, 6, device=st.device, dtype=st.dtype)
    if env.M <= 1 or visible is None:
        return blk.reshape(B, -1)
    o = env.sim.other_idx[lid]                                             # (B, M-1)
    so = st_all[o]
    c, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
    d = so[..., :2] - st[:, None, :2]
    dx, dy = d[..., 0] * c + d[..., 1] * s_, -d[..., 0] * s_ + d[..., 1] * c
    vel = lambda q: torch.stack([q[..., 3] * torch.cos(q[..., 2]) - q[..., 4] * torch.sin(q[..., 2]),
                                 q[..., 3] * torch.sin(q[..., 2]) + q[..., 4] * torch.cos(q[..., 2])], -1)
    dv = vel(so) - vel(st)[:, None, :]
    dvx, dvy = dv[..., 0] * c + dv[..., 1] * s_, -dv[..., 0] * s_ + dv[..., 1] * c
    dist = d.norm(dim=-1)
    key = torch.where(visible, dist, torch.full_like(dist, 1e6))
    n = min(K, key.shape[1])
    j = key.topk(n, dim=1, largest=False).indices
    g = lambda t: t.gather(1, j)
    pres = g(visible).to(st.dtype)
    S = PRIV_OPP_DIST_SCALE
    blk[:, :n] = torch.stack([pres, g(dx) / S, g(dy) / S, g(dist) / S, g(dvx) / S, g(dvy) / S], 2) * pres[..., None]
    return blk.reshape(B, -1)


class StateObs:
    """The observation, from the env's state after a step. scan: append the env's NOISE-FREE scan (`scan_true`, the
    beams `--beams` casts) / range: the true shape of every prop, wall and car in view (10-29: three prop centres
    without their size let the car clip a prop's edge in every narrowing -- crash_scenes.py)."""

    def __init__(self, env, scan: bool = False, scan_range: float = 10.0, corridor: bool = False, t_beams: int = 0,
                 corridor_cars: bool = False, car_channel: bool = False, bev: bool = False, path_plan: bool = False,
                 plan_ref=None, car_st: bool = False, scan_nocars: bool = False, corridor_car_st: bool = False,
                 blank_opp: bool = False, car_st_line: bool = False, vis_actor: bool = False, car_slots: bool = False,
                 scan_points: bool = False):
        self.env = env
        #: `--scan-points` (11-06, the user: "장애물이나 상대차의 라이다 포인트를 주면 그만 아님?", "라이다 1081 그대로 안 주는 이유가
        #: 뭔데"): the scan as three rows of the same beams -- every range, the ranges that ended on a prop (others at the
        #: range cap), the ranges that ended on a car (`scan_point_masks`) -- and no prop slots / prop corridor: the props
        #: reach the actor as their points, size and shape included (the slots held centres only: 20 of 22 hard hits of
        #: srp25owp u450 were on crates smaller than the layout median, each in view 1.4 s before, `plan_vs_hit.py`).
        #: Meant with --beams 1081, the car's own LiDAR (108 beams put a 0.22 m crate at 3 m on 1-2 beams).
        self.scan_points = bool(scan_points)
        #: `--vis-actor` (11-03): append the prop slots and the prop corridor over the props the car can SEE
        #: (`visible_props`) after everything else, and list in `actor_hide` the columns the actor must not read -- the
        #: full-truth prop slots and corridor and the time left; the critic keeps all of them (asymmetric actor-critic)
        self.vis_actor = bool(vis_actor)
        #: `--car-slots` (11-03): the visible other cars' positions and relative velocities, appended last (`visible_car_slots`)
        self.car_slots = bool(car_slots)
        self.actor_hide = [ACTOR_NO_LAP]
        self._hide_full = False
        self.vis0 = None
        self.ahead = torch.tensor(AHEAD, device=env.device)
        self.dim = None
        self.scan, self.scan_range = bool(scan), float(scan_range)
        self.corridor = bool(corridor)
        self.corridor_cars = bool(corridor_cars)
        #: `--corridor-car-st` (11-02): the other cars' `car_spacetime_block` OR'ed INTO the prop corridor -- a slower car
        #: is then a prop where we would reach it, so a props policy's swerve applies to it from the first decision (the
        #: race stage's own car block starts at zero weights, and every race lineage learned to hang back, never to pass:
        #: 0-1 of 20 races past a 0.6x car). srl19's cars-in-the-corridor were the cars NOW, a prop that moves away.
        #: `--blank-opp`: the privileged opponent slots as on the empty track (`blank_opponents`), the props policy's own
        self.corridor_car_st, self.blank_opp = bool(corridor_car_st), bool(blank_opp)
        #: `--car-st-line`: the cars predicted along the raceline (`car_spacetime_block(line=True)`)
        self.car_st_line = bool(car_st_line)
        #: `--car-channel` (10-30): the other cars (now, +0.5 s, +1 s) in the same 13 x 7 corridor grid but as their OWN block
        #: appended last -- the prop corridor keeps its meaning (srl19 put cars into it and the parent's prop avoidance broke),
        #: and an --init parent starts with zero weights on it, so update 1 drives as the parent did
        self.car_channel = bool(car_channel)
        #: `--car-st` (10-31): the car channel is `car_spacetime_block` (the cars where they will be when we get there)
        #: instead of the cars now / +0.5 s / +1 s; `--scan-nocars`: the scan is cast without the other cars (walls and
        #: props only) -- cars reach the policy as moving things only, never as a static obstacle
        self.car_st, self.scan_nocars = bool(car_st), bool(scan_nocars)
        #: `--bev` (10-30): the observation is [own state, time left, track ahead, steer, last action] + the car-centred
        #: grid (`bev_grid`, flattened, LAST) -- no prop / opponent slots, no corridor, no scan
        self.bev = bool(bev)
        self.grid_dim = BEV_C * BEV_H * BEV_W if self.bev else 0
        #: `--path-plan` (10-30): the track-ahead block is the PLANNED path (`learn.path_plan`: the raceline moved round
        #: props and cars by a sampling planner on the true geometry) instead of the centerline, plus the free distance
        #: along the plan; the policy drives the given path, the planner does the geometry
        self.planner = None
        #: 1: the block is the planned path itself (raceline + plan offset). 2 (10-30): the CENTERLINE moved sideways by the
        #: plan's offset -- on the open track the plan is the raceline (offset 0) and the block is exactly the solo teacher's,
        #: so a fast solo teacher keeps its line; near a prop the line it follows moves by what the planner moved (mode 1
        #: from scratch drove the empty track in 9.0 s against the solo teachers' 7.5-7.8)
        self.plan_mode = int(path_plan)
        if path_plan:
            from .path_plan import PathPlanner
            rls = getattr(env, "_rls", None)
            if rls is None:
                names = getattr(env, "_track_names", None) or [KOREA]
                _, rls = common.load_tracks(names, racelines=True, **dict(common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
            self.planner = PathPlanner(env, env.learner_ids, rls)
            if plan_ref:
                # the reference the plan moves: a recorded line of the policy itself (path_plan.record_line), not the raceline
                self.planner.set_reference(np.load(os.path.expanduser(plan_ref)))
        # t_beams: the teacher's own beam count when this env casts more (a student's env): its noise-free scan is
        # interpolated to those beams over the same field of view, as the teacher saw it in training
        self.interp = None
        if self.scan and t_beams:
            ang_s = env.sim.lidar.angles.float()
            if ang_s.numel() != int(t_beams):
                ang_t = torch.linspace(float(ang_s[0]), float(ang_s[-1]), int(t_beams), device=env.device)
                pos = (ang_t - ang_s[0]) / (ang_s[-1] - ang_s[0]) * (ang_s.numel() - 1)
                i0 = pos.floor().long().clamp(0, ang_s.numel() - 2)
                self.interp = (i0, (pos - i0.float())[None, :])

    def __call__(self, last_action):
        env = self.env
        lid = env.learner_ids
        if getattr(env, "_tw_carry", None) is not None:
            # a carried plan ("tw5c"): the action is its change, the plan itself is what the policy reads
            last_action = env._tw_carry.obs()[lid]
        pv = env.privileged_v2()[lid]
        if self.blank_opp:
            pv = blank_opponents(pv)
        res = env.last_result
        st = res.state[lid]
        tid = env.sim.tid[lid]
        s = res.s[lid]
        tr = env.sim.track
        n = self.ahead.numel()
        s_k = s[:, None] + self.ahead[None, :]
        if self.planner is not None and self.plan_mode == 1:
            self.planner.update()
            xy_k, _ = self.planner.path_xy(s_k, tid)
            xy_n, _ = self.planner.path_xy(s_k + 0.2, tid)
            yaw_k = torch.atan2(xy_n[..., 1] - xy_k[..., 1], xy_n[..., 0] - xy_k[..., 0])
        elif self.planner is not None:
            self.planner.update()
            xy_k, yaw_k = tr.pose_at_s(s_k.reshape(-1), tid.repeat_interleave(n))
            xy_k = xy_k.reshape(-1, n, 2); yaw_k = yaw_k.reshape(-1, n)
            w_cl = tr.sample_edt(xy_k, tid[:, None].expand(-1, n))         # the centerline's half width, as the solo teacher saw
            # the shift actually applied: the (in-track) plan offset less the reference line it moves
            tk_ = tid[:, None].expand(-1, n)
            e_k = self.planner.offset(s_k, tid) - self.planner.rl_offset(s_k, tk_)
            xy_k = torch.stack([xy_k[..., 0] - e_k * torch.sin(yaw_k), xy_k[..., 1] + e_k * torch.cos(yaw_k)], -1)
        else:
            xy_k, yaw_k = tr.pose_at_s(s_k.reshape(-1), tid.repeat_interleave(n))
            xy_k = xy_k.reshape(-1, n, 2); yaw_k = yaw_k.reshape(-1, n)
        d = xy_k - st[:, None, :2]
        c, si = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
        dx, dy = d[..., 0] * c + d[..., 1] * si, -d[..., 0] * si + d[..., 1] * c
        dyaw = torch.remainder(yaw_k - st[:, 2:3] + math.pi, 2 * math.pi) - math.pi
        w = tr.sample_edt(xy_k, tid[:, None].expand(-1, n)) if self.plan_mode != 2 else w_cl   # mode 1: clearance at the plan
        ahead = torch.stack([dx / 10.0, dy / 5.0, dyaw / math.pi, w / 2.0], 2).reshape(st.shape[0], -1)
        steer = (st[:, 6] / float(env.s_max))[:, None]
        if self.bev:
            # own state (privileged_v2's first 8) and the time left (its last); everything else is in the grid
            out = torch.cat([pv[:, :8], pv[:, -1:], ahead, steer, last_action, bev_grid(env, lid).reshape(st.shape[0], -1)], 1).float()
            self.dim = out.shape[1]
            return out
        parts = [pv, ahead, steer, last_action]
        self.scan0 = sum(int(p_.shape[1]) for p_ in parts)                     # the first scan column
        if self.scan:
            if self.scan_nocars and env.M > 1:
                sc = env.sim.lidar.scan(env.sim.state[:, :3], None, env.sim.P, motion_distortion=False, tid=env.sim.tid,
                                        cars=None, compiled=True, eid=env.sim.eid)[1][lid].float()
            else:
                sc = res.scan_true[lid].float()
            if self.scan_points:
                pm, cm = scan_point_masks(env, lid, sc, res.scan_type[lid])
            if self.interp is not None:
                i0, w_ = self.interp
                sc = sc[:, i0] * (1 - w_) + sc[:, i0 + 1] * w_
                if self.scan_points:
                    near = (i0 + (w_[0] >= 0.5).long())                     # labels: the nearer cast beam
                    pm, cm = pm[:, near], cm[:, near]
            parts.append((sc / self.scan_range).clamp(0.0, 1.0))
            if self.scan_points:
                cap = torch.full_like(sc, self.scan_range)
                parts.append((torch.where(pm, sc, cap) / self.scan_range).clamp(0.0, 1.0))
                parts.append((torch.where(cm, sc, cap) / self.scan_range).clamp(0.0, 1.0))
        if self.planner is not None:
            parts.append((self.planner.free / 15.0)[:, None])             # how far the plan is free [/15 m]
        if self.corridor:
            cb = corridor_block(env, lid, s, tid, cars=self.corridor_cars)
            if self.corridor_car_st:
                cb = torch.maximum(cb, car_spacetime_block(env, lid, s, tid, line=self.car_st_line))
            parts.append(cb)
        if self.car_channel:
            parts.append(car_spacetime_block(env, lid, s, tid, line=self.car_st_line) if self.car_st
                         else corridor_block(env, lid, s, tid, cars=True, props=False))
        if self.vis_actor:
            from ..gym_env import PRIV_V2_OPP_KEYS, PRIV_V2_OPP_SLOTS, PRIV_V2_PROP_DIM
            n0 = sum(int(p_.shape[1]) for p_ in parts)
            #: from here on (visible prop slots, visible corridor + cars, visible car slots) every input is 0 while
            #: nothing is in view (`ResidualPi`)
            self.vis0 = n0
            vis = visible_props(env, lid)
            if not self.scan_points:
                parts.append(visible_prop_slots(env, lid, vis))
            if self.corridor:
                if self.scan_points:                                       # the cars' part of the corridor only
                    cb_v = corridor_block(env, lid, s, tid, cars=self.corridor_cars, props=False)
                elif vis is not None:
                    cb_v = corridor_block(env, lid, s, tid, cars=self.corridor_cars, visible=vis)
                else:
                    cb_v = torch.zeros(st.shape[0], len(AHEAD) * CORRIDOR_BINS, device=st.device)
                if self.corridor_car_st:
                    cb_v = torch.maximum(cb_v, car_spacetime_block(env, lid, s, tid, line=self.car_st_line,
                                                                   visible=visible_cars(env, lid)))
                parts.append(cb_v)
            if self.car_slots:
                parts.append(visible_car_slots(env, lid, visible_cars(env, lid)))
            if not self._hide_full:
                # privileged_v2: ego 8, opponents, nearest-car geometry 4, the prop slots, [time left]
                p0 = 8 + PRIV_V2_OPP_SLOTS * len(PRIV_V2_OPP_KEYS) + 4
                # the opponent slots and the nearest-car geometry (full truth: cars behind walls, beyond the grid) and the
                # full-truth prop slots
                hide = list(range(8, p0)) + list(range(p0, p0 + PRIV_V2_PROP_DIM))
                if bool(env.ecfg.priv_time):
                    hide.append(int(pv.shape[1]) - 1)
                if self.corridor:
                    c0 = n0 - len(AHEAD) * CORRIDOR_BINS * (1 + int(self.car_channel))
                    hide += list(range(c0, c0 + len(AHEAD) * CORRIDOR_BINS))
                self.actor_hide = [ACTOR_NO_LAP] + hide
                self._hide_full = True
        out = torch.cat(parts, 1).float()
        self.dim = out.shape[1]
        return out


#: `--base` (10-31): race training slowed the empty-track lap in every lineage (srl14 7.72 -> 8.05 s over 250 updates,
#: srl50 8.52 -> 8.68 s by u300) and none of the causes tested was it (prop share srl51, hit-head shaping, price srl52,
#: the noise floor srl53) -- so the solo optimum is kept by construction: a gated mixture of two drivers
BASE_REACH = 10.0          # [m] what counts as near: the grid student sees 12 m ahead, so its labels stay learnable
BASE_HOLD = 10             # decisions (0.5 s) the race policy keeps the car after the last thing near has gone


def blank_opponents(o):
    """`o` (B, D) starting with `privileged_v2`, its opponent slots and nearest-car geometry as on a track with no other
    car -- what a policy trained without opponents (solo, props) saw there"""
    from ..gym_env import PRIV_OPP_DIST_SCALE, PRIV_V2_NO_OPP_DIST, PRIV_V2_OPP_KEYS, PRIV_V2_OPP_SLOTS
    ko = PRIV_V2_OPP_SLOTS * len(PRIV_V2_OPP_KEYS)
    o = o.clone()
    o[:, 8:8 + ko] = 0.0
    o[:, 8 + ko:12 + ko] = torch.tensor([PRIV_V2_NO_OPP_DIST / PRIV_OPP_DIST_SCALE, 0.0, 1.0, 1.0], device=o.device,
                                        dtype=o.dtype)
    return o


class CalmGate:
    """`--calm-std` (11-04): SoloMix's 'nothing near' test without a solo policy -- (B,) bool `near` per decision (a visible prop
    in the corridor within `reach` m, a car within `reach` m no more than 2 m behind, or below 1 m/s, held `hold` decisions)"""

    def __init__(self, env, reach=10.0, hold=10, visible=True):
        self.env, self.lid, self.reach, self.hold, self.visible = env, env.learner_ids, float(reach), int(hold), bool(visible)
        self.n_seg = sum(1 for x in AHEAD if x <= self.reach + 1e-6)
        self.left = torch.zeros(int(self.lid.numel()), dtype=torch.long, device=env.device)

    def step(self):
        env, lid = self.env, self.lid
        st = env.last_result.state
        s, tid = env.last_result.s[lid], env.sim.tid[lid]
        B = s.shape[0]
        cb = corridor_block(env, lid, s, tid, visible=visible_props(env, lid) if self.visible else None
                            ).reshape(B, len(AHEAD), CORRIDOR_BINS)[:, :self.n_seg]
        near = cb.flatten(1).any(1) | (st[lid, 3] < 1.0)
        if env.M > 1:
            dx, dy, _, _, _, dist, pres = env._opp_rel(st)
            nc = (pres & (dx > -2.0) & (dist < self.reach))[lid]
            if self.visible:
                nc = nc & visible_cars(env, lid)
            near |= nc.any(1)
        self.left = torch.where(near, torch.full_like(self.left, self.hold), (self.left - 1).clamp_min(0))
        return self.left > 0


class SoloMix:
    """The frozen SOLO teacher (`--base`, e.g. srl1: empty track 7.53 s) drives while nothing is near; the race policy only
    when a live prop is on the track within `reach` m ahead, another car is within `reach` m (no more than 2 m behind),
    or the car is below 1 m/s (starts, wedges: only the race policy can reverse), and `hold` decisions after. The solo
    teacher sees the world without props and cars -- its prop / opponent slots as on the empty track it was trained on."""

    def __init__(self, env, path, device, reach=BASE_REACH, hold=BASE_HOLD, visible=False):
        from ..gym_env import PRIV_V2_OPP_KEYS, PRIV_V2_OPP_SLOTS, PRIV_V2_PROP_DIM
        ckb = torch.load(os.path.expanduser(path), map_location=device, weights_only=False)
        ar = ckb["args"]
        if ar.get("scan_obs") or ar.get("corridor_obs") or ar.get("car_channel") or ar.get("bev") or ar.get("path_plan"):
            raise ValueError("--base must be a plain state teacher (privileged vector + track ahead), e.g. srl1_solo")
        # 11-02: a plan solo teacher (`--action plan`, e.g. srp25h2_solo2) mixes into a plan run -- the same trajectory
        # format through the same follower, so its plan and the race policy's mean the same thing
        env_plan = str(getattr(env.ecfg, "action_mode", "direct")) == "plan"
        if plan_of(ar) != env_plan:
            raise ValueError(f"--base {path}: a {'plan' if plan_of(ar) else 'direct'} teacher cannot drive this "
                             f"{'plan' if env_plan else 'direct'} run")
        if env_plan:
            inner = env.tracker._inner if hasattr(env.tracker, "_inner") else env.tracker
            if abs(float(ar.get("plan_horizon", 0.6) or 0.6) - float(inner.spec.wp_horizon_s)) > 1e-9 \
                    or int(ar.get("repeat", 2)) != int(env.ecfg.decision_every):
                raise ValueError(f"--base {path}: plan length / decision rate differ from this run's")
        self.env, self.lid = env, env.learner_ids
        B = int(self.lid.numel())
        self.sobs = StateObs(env)
        o = self.sobs(torch.zeros(B, act_dim_of(ar), device=device))
        self.net = ActorCritic(o.shape[1], act_dim=act_dim_of(ar), hidden=int(ar.get("hidden", 256) or 256)).to(device)
        self.net.load_state_dict(ckb["model"], strict=False); self.net.eval()
        self.nrm = RunningNorm(o.shape[1], device); self.nrm.load_state_dict(ckb["norm"])
        self.reach, self.hold = float(reach), int(hold)
        #: `--vis-actor`: "near" means a prop the car can SEE (`visible_props`), as the actor's own inputs
        self.visible = bool(visible)
        ko = PRIV_V2_OPP_SLOTS * len(PRIV_V2_OPP_KEYS)
        self.prop_cols = slice(12 + ko, 12 + ko + PRIV_V2_PROP_DIM)
        self.n_seg = sum(1 for x in AHEAD if x <= self.reach + 1e-6)
        self.left = torch.zeros(B, dtype=torch.long, device=device)
        self.gate = torch.ones(B, dtype=torch.bool, device=device)

    def near(self):
        """(B,) bool: something the race policy must handle"""
        env, lid = self.env, self.lid
        st = env.last_result.state
        s, tid = env.last_result.s[lid], env.sim.tid[lid]
        B = s.shape[0]
        cb = corridor_block(env, lid, s, tid, visible=visible_props(env, lid) if self.visible else None
                            ).reshape(B, len(AHEAD), CORRIDOR_BINS)[:, :self.n_seg]
        out = cb.flatten(1).any(1) | (st[lid, 3] < 1.0)
        if env.M > 1:
            dx, dy, _, _, _, dist, pres = env._opp_rel(st)
            nc = (pres & (dx > -2.0) & (dist < self.reach))[lid]
            if self.visible:
                nc = nc & visible_cars(env, lid)
            out |= nc.any(1)
        return out

    def solo(self, last):
        """(the solo teacher's mean action, the gate) -- `__call__` without the mixing (`--anchor-solo`)"""
        self.left = torch.where(self.near(), torch.full_like(self.left, self.hold), (self.left - 1).clamp_min(0))
        self.gate = self.left > 0
        o = blank_opponents(self.sobs(last.clamp(-1, 1)))
        o[:, self.prop_cols] = 0.0
        return self.net.pi(self.nrm(o)).clamp(-1, 1), self.gate

    def __call__(self, a_pol, last):
        """the executed action: the race policy's `a_pol` where gated, the solo teacher's mean action elsewhere"""
        self.left = torch.where(self.near(), torch.full_like(self.left, self.hold), (self.left - 1).clamp_min(0))
        self.gate = self.left > 0
        o = blank_opponents(self.sobs(last.clamp(-1, 1)))
        o[:, self.prop_cols] = 0.0
        a_b = self.net.pi(self.nrm(o)).clamp(-1, 1)
        return torch.where(self.gate[:, None], a_pol, a_b)


class TeacherPolicy:
    """A state_rl checkpoint as a driver -- for the tools and the student's labels: its observation, input norm and MEAN
    action in its action range; with `--base` the gated mixture (`SoloMix`). `self.o` is the last observation."""

    def __init__(self, env, ck, device, t_beams=0):
        ar = ck["args"]
        self.env = env
        self.ck, self.ext = ck, float(ar.get("reverse_ext", 0.0) or 0.0)
        self.ar, self.A = ar, act_dim_of(ar)
        self.sobs = StateObs(env, scan=bool(ar.get("scan_obs", 0)), corridor=bool(ar.get("corridor_obs", 0)),
                             corridor_cars=bool(ar.get("corridor_cars", 0)), car_channel=bool(ar.get("car_channel", 0)),
                             bev=bool(ar.get("bev", 0)), path_plan=int(ar.get("path_plan", 0) or 0),
                             plan_ref=ar.get("plan_ref"), t_beams=int(t_beams), car_st=bool(ar.get("car_st", 0)),
                             scan_nocars=bool(ar.get("scan_nocars", 0)),
                             corridor_car_st=bool(ar.get("corridor_car_st", 0)), blank_opp=bool(ar.get("blank_opp", 0)),
                             car_st_line=bool(ar.get("car_st_line", 0)), vis_actor=bool(ar.get("vis_actor", 0)),
                             car_slots=bool(ar.get("car_slots", 0)), scan_points=bool(ar.get("scan_points", 0)))
        B = int(env.learner_ids.numel())
        self.o = self.sobs(torch.zeros(B, self.A, device=device))
        self.net = ActorCritic(self.o.shape[1], act_dim=self.A, bev=ck.get("bev"), hidden=int(ar.get("hidden", 256) or 256),
                               residual=residual_of(ar)).to(device)
        self.net.load_state_dict(ck["model"], strict=False); self.net.eval()
        self.nrm = RunningNorm(self.o.shape[1], device); self.nrm.load_state_dict(ck["norm"])
        self.mix = SoloMix(env, ar["base"], device, float(ar.get("base_reach", BASE_REACH)),
                           int(ar.get("base_hold", BASE_HOLD)), visible=bool(ar.get("vis_actor", 0))) if ar.get("base") else None

    def policy(self, o):
        """the race policy alone at observation `o`. F1SIM_POLICY_NOISE=k (a diagnostic, 11-06): sample with k x the
        policy's own exploration std instead of the mean -- does the training noise carry the car out of stops the mean
        action stays in?"""
        k_ = float(os.environ.get("F1SIM_POLICY_NOISE", "0") or 0)
        if k_ > 0:
            nn_ = self.nrm(o)
            d_ = self.net.dist(nn_)
            se_ = os.environ.get("F1SIM_SLOW_EXPLORE")
            if se_:
                # F1SIM_SLOW_EXPLORE=V0,STEER,SPEED (a diagnostic, 11-06): the distribution `--slow-explore` trained with --
                # does a stop the mean holds after a hard hit go when the slow-speed floor's noise is on?
                v0_, fk_, fs_ = (float(x) for x in se_.split(","))
                s_ = (v0_ / self.env.sim.state[self.env.learner_ids, 3].float().abs().clamp_min(1e-3)).square().clamp(max=1.0)
                kd_ = kappa_dims_of(self.ar)
                fl_ = torch.cat([fk_ * s_[:, None].expand(-1, kd_), fs_ * s_[:, None].expand(-1, self.A - kd_)], 1)
                d_ = torch.distributions.Normal(d_.mean, torch.maximum(d_.stddev, fl_))
            # F1SIM_POLICY_NOISE_CORR=c: AR(1) noise held across decisions, as --noise-corr trains with
            c_ = float(os.environ.get("F1SIM_POLICY_NOISE_CORR", "0") or 0)
            e_ = torch.randn_like(d_.mean)
            if c_ > 0:
                prev = getattr(self, "_eps_c", None)
                cs_ = os.environ.get("F1SIM_CORR_SLOW")
                if cs_:
                    # the hold only below V0 (x (V0 / v)^2 above), as `--corr-slow` trains with
                    v0c_ = float(cs_)
                    s2_ = (v0c_ / self.env.sim.state[self.env.learner_ids, 3].float().abs().clamp_min(1e-3)).square().clamp(max=1.0)
                    cc_ = (c_ * s2_)[:, None]
                else:
                    cc_ = c_
                if prev is not None and prev.shape == e_.shape:
                    e_ = cc_ * prev + torch.sqrt(1.0 - torch.as_tensor(cc_) ** 2) * e_
                self._eps_c = e_
            return clamp_any(d_.mean + k_ * d_.stddev * e_, self.ar)
        return clamp_any(self.net.pi(self.nrm(o)), self.ar)

    def __call__(self, last):
        self.o = self.sobs(last)
        a = self.policy(self.o)
        return self.mix(a, last) if self.mix is not None else a


def geo_risk(env, lid, horizon=1.0, n=10, margin=0.30, radius=0.20):
    """(B,) in [0, 1]: how deep the car's next `horizon` seconds -- a constant-curvature arc from its speed and yaw rate --
    run into a live prop, from the props' true shapes (their local half-planes at their poses). 0 while the arc keeps
    `margin` of clearance beyond the car's `radius`, 1 at contact. A state function: as a potential (Ng et al. 1999) it
    leaves the optimum alone and only moves the price of a hit to the moment the path first crosses the prop (10-29: the
    learned hit head rises only 0.5 s before, median P 0.05 at 1 s, though the hits are avoidable and worth avoiding)."""
    po = env.procedural
    st = env.last_result.state[lid]
    B = st.shape[0]
    if po is None or po.C == 0:
        return torch.zeros(B, device=st.device)
    x, y, yaw, v, r = st[:, 0], st[:, 1], st[:, 2], st[:, 3].clamp_min(0.0), st[:, 5]
    kap = r / v.clamp_min(0.5)
    t = torch.linspace(horizon / n, horizon, n, device=st.device)
    sa = v[:, None] * t[None, :]                                          # (B, n) arc length
    k_ = kap[:, None]
    small = k_.abs() < 1e-3
    th = yaw[:, None] + k_ * sa
    ks = torch.where(small, torch.ones_like(k_), k_)
    px = torch.where(small, x[:, None] + sa * torch.cos(yaw)[:, None], x[:, None] + (torch.sin(th) - torch.sin(yaw)[:, None]) / ks)
    py = torch.where(small, y[:, None] + sa * torch.sin(yaw)[:, None], y[:, None] - (torch.cos(th) - torch.cos(yaw)[:, None]) / ks)
    pp = po.p_poses[lid]                                                  # (B, C, 3)
    live = po.p_sid[lid] >= 0                                             # (B, C)
    dx = px[:, None, :] - pp[:, :, None, 0]; dy = py[:, None, :] - pp[:, :, None, 1]   # (B, C, n)
    c, s_ = torch.cos(pp[..., 2])[..., None], torch.sin(pp[..., 2])[..., None]
    lx, ly = dx * c + dy * s_, -dx * s_ + dy * c                          # the arc in each prop's own frame
    nrm, off = po.p_n[lid], po.p_d[lid]                                   # (B, C, k, 2), (B, C, k)
    fin = torch.isfinite(off)
    sd = nrm[..., None, :, 0] * lx[..., None] + nrm[..., None, :, 1] * ly[..., None] - torch.where(fin, off, torch.zeros_like(off))[..., None, :]
    sd = torch.where(fin[..., None, :], sd, torch.full_like(sd, -1e9)).amax(-1)        # (B, C, n): > 0 outside
    sd = torch.where(live[..., None], sd, torch.full_like(sd, 1e9))
    clear = sd.amin(-1).amin(-1) - radius
    return ((margin - clear) / margin).clamp(0.0, 1.0)


class PropRemoval:
    """The competition's sessions (10-29, user: "혼자 달리는 세션이 있고 1대1로 달리는 세션이 있는데, 둘 다 장애물을 고정된 위치에
    두고 있다가 시간이 지나면 치우는 식으로 해서 장애물 치우면 또 원래 가장 빠른 라인으로 달려야 함"): per race, with probability `p`
    the layout vanishes at a time drawn in [lo, hi] s after the episode start (fixed: lo = hi), the same moment for every car of
    the race. A removed prop is a dead slot like any unused one (`p_sid` -1, z_hi < z_lo, no mass): the beams, the contact test,
    the privileged obs, the corridor and the prop-aware opponents all read those. Call `step()` after every env step."""

    def __init__(self, env, p: float, lo: float, hi: float, seed: int = 0):
        self.env, self.p, self.lo, self.hi = env, float(p), float(lo), float(hi)
        self.R = env.B // env.M
        dev = env.device
        self.g = torch.Generator(device=dev); self.g.manual_seed(int(seed))
        self.t_rm = torch.full((self.R,), float("inf"), device=dev)
        self.removed = torch.zeros(self.R, dtype=torch.bool, device=dev)
        self.prev = torch.full((self.R,), 10 ** 9, dtype=torch.long, device=dev)
        self.first = env.learner_ids if env.M > 1 else torch.arange(env.B, device=dev)

    def step(self):
        env, po = self.env, self.env.procedural
        if po is None or self.p <= 0:
            return
        es = env.ep_step[self.first].long()
        new = es < self.prev                                              # this race's episode (and layout) began anew
        if bool(new.any()):
            n = int(new.sum())
            draw = torch.rand(n, device=es.device, generator=self.g) < self.p
            t = self.lo + (self.hi - self.lo) * torch.rand(n, device=es.device, generator=self.g)
            self.t_rm[new] = torch.where(draw, t, torch.full_like(t, float("inf")))
            self.removed[new] = False
        self.prev = es
        due = (~self.removed) & (es.float() * float(env.sim.control_dt) >= self.t_rm)
        if bool(due.any()):
            races = due.nonzero().flatten()
            rows = (races[:, None] * env.M + torch.arange(env.M, device=races.device)[None, :]).flatten()
            po.p_sid[rows] = -1
            po.p_zhi[rows] = po.p_zlo[rows] - 1.0
            po.p_mass[rows] = 0.0
            po.p_vel[rows] = 0.0
            self.removed |= due


class Unwedge:
    """The old pipeline's wedge recovery (`mpc.PlanSpec.wedge_s`, which its plan tracker runs on the car), for a direct
    command: touching something below 0.4 m/s for 1 s -> back out at 0.8 m/s for 0.75 s with the steering mirrored.
    10-29: without it a direct-command car that hit a wall stayed pinned for the rest of the episode -- the speed is
    0..10 m/s, no reverse (solo session of srl13: cars wedged in 1-8 s and stood 25 s, one at full throttle into the
    wall). It drives through the env's external-command override, vectorised; the policy's own action is what is
    recorded. Call `before(act_full)` before `env.step` and `after()` right after it."""

    def __init__(self, env, rows, wedge_s=1.0, backout_s=0.75, v_gate=0.4, v_reverse=0.8, mode=1):
        self.env, self.rows = env, rows
        #: mode 1: the old tracker's mirrored steering. mode 2 (10-30): the back-out steering turns the nose toward the
        #: freer side of the car's own front scan, latched for the back-out -- the mirror assumes the policy was steering
        #: away from what it hit; in the srl21 loop it steered INTO a prop at the inner wall, the mirrored back-out turned
        #: the nose further toward it, and the car re-hit the same prop 15 times in 37 s
        self.mode = int(mode)
        self.rev_steer = torch.zeros(env.B, device=env.device)
        dev = env.device
        self.mask = torch.zeros(env.B, dtype=torch.bool, device=dev); self.mask[rows] = True
        self.wedged_t = torch.zeros(env.B, device=dev); self.backout_t = torch.zeros(env.B, device=dev)
        self.wedge_s, self.backout_s, self.v_gate, self.v_rev = float(wedge_s), float(backout_s), float(v_gate), float(v_reverse)
        if not env._ext_ids:
            env._ext_ids.add(-1)                         # keep the env's override path on; the mask decides per row
        self.count = 0

    def before(self, act_full):
        env = self.env
        on = (self.backout_t > 1e-6) & self.mask
        env.ext_mask.copy_(on)
        if self.mode == 2:
            env.ext_cmd[:, 0] = self.rev_steer
        else:
            env.ext_cmd[:, 0] = -act_full[:, 0] * float(env.s_max)
        env.ext_cmd[:, 1] = -self.v_rev

    def after(self):
        env = self.env
        dt = float(env.sim.control_dt)
        touching = env._was_touching & self.mask & (self.backout_t <= 0)
        v = env.sim.state[:, 3].abs()
        step = torch.where(v < self.v_gate, torch.full_like(self.wedged_t, dt), torch.full_like(self.wedged_t, -dt))
        self.wedged_t = torch.where(touching, (self.wedged_t + step).clamp_min(0.0), torch.zeros_like(self.wedged_t))
        start = self.wedged_t >= self.wedge_s - 1e-6
        self.count += int(start.sum()) if bool(start.any()) else 0
        if self.mode == 2 and bool(start.any()):
            # reversing, a left steer swings the nose right: steer toward the MORE blocked front side
            sc = env.last_result.scan.float()
            ang = env.sim.lidar.angles
            left = (ang > math.radians(10)) & (ang < math.radians(80))
            right = (ang < -math.radians(10)) & (ang > -math.radians(80))
            cl = torch.quantile(sc[:, left], 0.2, dim=1); cr = torch.quantile(sc[:, right], 0.2, dim=1)
            st = torch.where(cl < cr, torch.full_like(cl, 0.7), torch.full_like(cl, -0.7)) * float(env.s_max)
            self.rev_steer = torch.where(start, st, self.rev_steer)
        self.backout_t = torch.where(start, torch.full_like(self.backout_t, self.backout_s), (self.backout_t - dt).clamp_min(0.0))
        self.wedged_t = self.wedged_t.masked_fill(start, 0.0)


class RunningNorm:
    def __init__(self, dim, device, clip=8.0, raw_tail=0):
        self.mean = torch.zeros(dim, device=device); self.var = torch.ones(dim, device=device); self.n = 1e-4
        self.clip = clip
        #: `--residual-solo`: the statistics stay as loaded (the frozen solo policy reads its own, `vis_empty` stays exact)
        self.frozen = False
        #: the last `raw_tail` dims pass unnormalised (the --bev grid: already in [-2, 1], and a cell that is almost
        #: always empty would be blown up to the clip by its tiny variance)
        self.raw_tail = int(raw_tail)

    def update(self, x):
        if self.frozen:
            return
        x = x.reshape(-1, x.shape[-1])
        bm, bv, bn = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean; tot = self.n + bn
        self.mean = self.mean + d * bn / tot
        self.var = (self.var * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x):
        y = ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)
        if self.raw_tail:
            y = torch.cat([y[..., :-self.raw_tail], x[..., -self.raw_tail:]], -1)
        return y

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "n": self.n, "raw_tail": self.raw_tail}

    def load_state_dict(self, d):
        self.mean, self.var, self.n = d["mean"], d["var"], float(d["n"])
        self.raw_tail = int(d.get("raw_tail", 0))


def clamp_act(a, ext=0.0):
    """the action range: steer in [-1, 1]; speed in [-1 - ext, 1] -- `EnvConfig.reverse_ext` (below -1 = reverse)"""
    ext = float(ext or 0.0)
    if ext <= 0:
        return a.clamp(-1, 1)
    return torch.stack([a[..., 0].clamp(-1, 1), a[..., 1].clamp(-1 - ext, 1)], -1)


#: `--action plan` (11-01; the user: "누가 제어기까지 시뮬에서 쳐 먹이래? sim2real 갭을 늘리는 짓거리를 왜하고있는건데?",
#: "trajectory 로 e2e 하라니까"): the policy outputs a TRAJECTORY -- the v2 plan, 4 waypoints x (curvature, speed) in its
#: own frame (`mpc` speed mode "wp4") -- and ForzaETH's MAP follows it, as the car's controller node does; the plan is
#: decided every --repeat control steps and held on the track in between (`EnvConfig.decision_every`). Every direct
#: (steer, speed) policy before learned the simulator's tyres and steering itself (sst12: |side slip| > 10 deg 11-15 % of
#: the time, p99 40-47 deg). The follower is the v2 entry's tuned Korea MAP (`korea26_v2_ppo_s2421_u512`'s contract),
#: its speed read --plan-lookahead s ahead (10-24: the drivetrain lag 0.235 s gives v2 its authority), the plan
#: --plan-horizon s of the measured speed long (`docs/research/plan-output-2026-11-01`).
PLAN_MAP = {"t_clip_min": 0.9573609450243445, "t_clip_max": 4.779545745206895, "m_l1": 0.4731895571264694,
            "q_l1": -0.4957209879277062, "speed_lookahead": 0.0, "lat_err_coeff": 0.0, "acc_scaler_for_steer": 0.8857957332213734,
            "dec_scaler_for_steer": 0.8092827304787531, "start_scale_speed": 8.700352304982314,
            "downscale_factor": 0.24481196266259284, "speed_lookahead_for_steer": 0.060053547789199635,
            "v_steer_gain_max": 1.2810367013484487, "l1_speed": "measured", "l1_floor": True, "l1_point": "arc",
            "steer_speed": "planned", "lut_interp": "code", "delay_comp": 0.021416617709494996, "lat_slow": 0.0,
            "end_scale_speed": 9.580093136639107}
PLAN_KAPPA_DIMS = 4                 # wp4: kappa_1..4 first, then v_1..4


def _ar(x) -> dict:
    return x if isinstance(x, dict) else vars(x)


def plan_of(ar) -> bool:
    """a run (its argparse namespace or a checkpoint's args) that outputs plans"""
    return str((_ar(ar) or {}).get("action", "direct") or "direct") == "plan"


def tw_of(ar) -> bool:
    """a time-waypoint plan (`--plan-format tw`, `plan_tw`)"""
    return plan_of(ar) and str(_ar(ar).get("plan_format", "wp4") or "wp4") in ("tw", "tw5", "tw5c", "tw5g")


def kappa_dims_of(ar) -> int:
    """how many of a plan's numbers are curvatures (they come first): wp4's 4, a time-waypoint plan's TW_K"""
    from .. import plan_tw
    return plan_tw.N_KNOT if tw_of(ar) else PLAN_KAPPA_DIMS


def act_dim_of(ar) -> int:
    from ..mpc import act_dim as _act_dim
    from .. import plan_tw
    if tw_of(ar):
        return 2 * plan_tw.N_KNOT
    return _act_dim("wp4") if plan_of(ar) else 2


def clamp_any(a, ar):
    """this run's executable action: a plan's numbers in [-1, 1] (its first speed down to -1 - `--plan-reverse-ext`);
    (steer, speed) as `clamp_act`"""
    if plan_of(ar):
        pext = float(_ar(ar).get("plan_reverse_ext", 0.0) or 0.0)
        if pext <= 0:
            return a.clamp(-1.0, 1.0)
        lo = torch.full((a.shape[-1],), -1.0, device=a.device, dtype=a.dtype)
        lo[PLAN_KAPPA_DIMS] = -1.0 - pext
        return torch.maximum(torch.minimum(a, torch.ones_like(lo)), lo)
    return clamp_act(a, float(_ar(ar).get("reverse_ext", 0.0) or 0.0))


def mean_bounds(ar, A: int):
    """(lo, hi, margin) of each action dimension, for `--soft-bound` / `--bound-loss` / the dead-decision count"""
    if plan_of(ar):
        lo = [-1.0] * A
        lo[PLAN_KAPPA_DIMS] -= float(_ar(ar).get("plan_reverse_ext", 0.0) or 0.0)
        kd = kappa_dims_of(ar)
        return lo, [1.0] * A, [0.1] * kd + [0.2] * (A - kd)
    ext = float(_ar(ar).get("reverse_ext", 0.0) or 0.0)
    return [-1.0, -1.0 - ext], [1.0, 1.0], [0.1, 0.2]


def steer_speed_dims(ar, A: int):
    """(steering dims, speed dims): the plan's curvatures and speeds, or (steer,) and (speed,)"""
    if plan_of(ar):
        kd = kappa_dims_of(ar)
        return list(range(kd)), list(range(kd, A))
    return [0], [1]


def std_floor_of(ar, A: int):
    """`--std-floor` (steer, speed) per action dimension: a plan's curvatures take the steer floor, its speeds the speed's"""
    f = _ar(ar).get("std_floor")
    if not f:
        return None
    f = [float(x) for x in f]
    if plan_of(ar) and len(f) == 2:
        kd = kappa_dims_of(ar)
        return [f[0]] * kd + [f[1]] * (A - kd)
    return f


def apply_plan_spec(env, ar) -> None:
    """the plan's length (`--plan-horizon`, `mpc.PlanSpec.wp_horizon_s`) on the env's follower -- every env a plan
    checkpoint drives in (training, the tools, the console) calls this after building it"""
    if plan_of(ar) and getattr(env, "tracker", None) is not None:
        import dataclasses
        inner = env.tracker._inner if hasattr(env.tracker, "_inner") else env.tracker
        inner.spec = dataclasses.replace(inner.spec, wp_horizon_s=float(_ar(ar).get("plan_horizon", 0.6) or 0.6))


#: `--plan-relabel`: the plan is compared with the driven path at RELABEL_PTS points per plan length L, over the
#: RELABEL_SPAN_S of planned travel the console draws (the tracker's 0.6 s reference walk): [0, 2 L] for a 0.3 s plan,
#: whose last stretch MAP aims into (L1 = 1.2-1.4 L at 4-9 m/s), [0, L] for a 0.6 s one; and at its WP_K waypoints'
#: speeds. The driven path is read over RELABEL_WINDOW_S, twice the span, so a braking car still covers it
RELABEL_PTS = 8
RELABEL_SPAN_S = 0.6
RELABEL_WINDOW_S = 1.2


def relabel_ext(spec) -> int:
    """plan lengths the relabel compares over: the console's 0.6 s walk in units of this spec's plan"""
    return max(1, int(round(RELABEL_SPAN_S / float(spec.wp_horizon_s))))


def plan_relabel_targets(env, pose_s, d_s, vm, acts, R, window_s=RELABEL_WINDOW_S):
    """`--plan-relabel` (11-01): for each decision, the path the car then DROVE, in the body frame of the pose its plan
    was issued from, at arc lengths j L / RELABEL_PTS (j = 1 .. relabel_ext RELABEL_PTS), and its speed where it passed
    the plan's waypoints (k L / WP_K). `relabel_loss` pulls the mean plan onto them.

    Why: MAP steers by one point of the plan and reads its speed at one point, so the rest of the plan is free, and the
    PPO plans drifted into a steering lever (11-01, srp1 / srp25 / srp26x: the plan's own [0, L] ends 0.4-0.6 m off the
    path the car then drove, the console's drawing 1.2-1.8 m, hooks through walls). Pulled onto what was driven, the
    plan is the trajectory the car follows. Compared in position, not as fitted curvatures: kappa_4 moves [0, L] by
    millimetres, so a fit to it turns the path's sampling error into 0.1 1/m (measured on a sampled circle).

    pose_s (T R + 1, B, 4): x, y, yaw, v before every control step and after the last; d_s (T R, B): an episode ended
    in that step; vm (T, B): the odometry speed each plan was decoded at; acts (T, B, A): the sampled actions.
    Returns points (T, B, n, 2), speeds (T, B, WP_K), a (T, B) mask -- the next `window_s` inside this rollout and one
    episode, and the car covered the compared span in it -- and, over the masked rows, the executed plans' farthest
    point from the driven path on [0, L] and on the whole span [m]. (11-02: a 0.8 s window and [0, 2 L] under a 0.6 s
    plan left 0-6 % of the decisions valid -- 1.2 s of travel never fits in 0.8 s.)"""
    from .. import mpc
    inner = env.tracker._inner if hasattr(env.tracker, "_inner") else env.tracker
    spec, v_max = inner.spec, float(inner.v_max)
    S1, B, _ = pose_s.shape
    T = (S1 - 1) // R
    Wn = int(round(window_s / float(env.sim.control_dt)))
    dev, dt_ = pose_s.device, pose_s.dtype
    ext = relabel_ext(spec)
    n = ext * RELABEL_PTS
    pts = torch.zeros(T, B, n, 2, device=dev); vks = torch.zeros(T, B, mpc.WP_K, device=dev)
    msk = torch.zeros(T, B, device=dev)
    cap = torch.full((B,), float(env.ecfg.speed_cap), device=dev, dtype=dt_)
    gi = torch.arange(1, n + 1, device=dev) * (spec.wp_grid // RELABEL_PTS)
    d1, d2 = [], []
    for t in range(T):
        i0 = t * R
        if i0 + Wn > S1 - 1:
            break
        win = pose_s[i0:i0 + Wn + 1]                                           # (Wn + 1, B, 4)
        ok = ~d_s[i0:i0 + Wn].any(0)
        c0, s0 = torch.cos(win[0, :, 2]), torch.sin(win[0, :, 2])
        dx, dy = win[:, :, 0] - win[0, :, 0], win[:, :, 1] - win[0, :, 1]
        bx, by = (c0 * dx + s0 * dy).T.contiguous(), (-s0 * dx + c0 * dy).T.contiguous()   # (B, Wn + 1)
        arc = torch.cat([torch.zeros(B, 1, device=dev, dtype=dt_),
                         torch.cumsum(torch.hypot(bx.diff(dim=1), by.diff(dim=1)), 1)], 1).contiguous()
        L = mpc.plan_length(vm[t], spec)
        ok &= arc[:, -1] >= ext * L

        def at(q, val):                                                        # val (B, Wn + 1) at arc lengths q (B, m)
            idx = torch.searchsorted(arc, q.contiguous()).clamp(1, Wn)
            a0, a1 = arc.gather(1, idx - 1), arc.gather(1, idx)
            w = ((q - a0) / (a1 - a0).clamp_min(1e-6)).clamp(0.0, 1.0)
            return val.gather(1, idx - 1) * (1 - w) + val.gather(1, idx) * w

        q = torch.arange(1, n + 1, device=dev, dtype=dt_)[None] / RELABEL_PTS * L[:, None]
        tx, ty = at(q, bx), at(q, by)
        pts[t] = torch.stack([tx, ty], -1)
        fk = torch.arange(1, mpc.WP_K + 1, device=dev, dtype=dt_)[None] / mpc.WP_K * L[:, None]
        vks[t] = at(fk, win[:, :, 3].T.contiguous()).clamp_min(0.0)
        msk[t] = ok.float()
        k_exe, _, _ = mpc.decode_wp4(acts[t], vm[t], v_max, cap, spec)
        px, py, _, _, _ = mpc.wp4_path(k_exe, L, spec, extend=float(ext))
        e = torch.hypot(px[:, gi] - tx, py[:, gi] - ty)
        d1.append(e[:, :RELABEL_PTS].amax(1)[ok]); d2.append(e.amax(1)[ok])
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0, device=dev)
    return pts, vks, msk, cat(d1), cat(d2)


def relabel_loss(env, mean, vm, pts, vks, w_speed=0.0):
    """(N,) the mean plan's squared distance to the driven path at the relabel points (mean over points) +
    `w_speed` x its waypoint speeds' squared error to the driven ones -- differentiable in `mean` (`--plan-relabel`)"""
    from .. import mpc
    inner = env.tracker._inner if hasattr(env.tracker, "_inner") else env.tracker
    spec, v_max = inner.spec, float(inner.v_max)
    cap = torch.full_like(vm, float(env.ecfg.speed_cap))
    k, L, v = mpc.decode_wp4(mean, vm, v_max, cap, spec)
    px, py, _, _, _ = mpc.wp4_path(k, L, spec, extend=float(pts.shape[1] // RELABEL_PTS))
    gi = torch.arange(1, pts.shape[1] + 1, device=mean.device) * (spec.wp_grid // RELABEL_PTS)
    e_pos = (px[:, gi] - pts[..., 0]).square() + (py[:, gi] - pts[..., 1]).square()
    return e_pos.mean(1) + float(w_speed) * (v - vks).square().mean(1)


def plan_env_fields(ar) -> dict:
    """EnvConfig fields of a plan run (`--action plan`); {} for a direct one"""
    if not plan_of(ar):
        return {}
    a = _ar(ar)
    tw = str(a.get("plan_format", "wp4") or "wp4") in ("tw", "tw5", "tw5c", "tw5g")
    # tw: MAP commands the speed of the segment the car is in. A speed lookahead reads the waypoint array 0.235 s of the
    # measured speed ahead -- past a slow first segment (0.1 m/s x 0.375 s is 4 cm) into the next one's speed: the
    # tracking test drove at 5-6 m/s on plans that said 0.1 (11-06)
    # tw's lookahead, tuned by the distance to given reference paths (`tools/tw_tune.py`, 11-06, the user: "ld가 먼저"): L1 =
    # max(0.35, 0.47 v) -- centreline 3 / 4 m/s 6.2 / 12.4 cm (7.1 / 14.6 at wp4's max(0.96, 0.47 v - 0.50)), the turn after a
    # change of direction 23.5 / 33.8 cm (25.9 / 36.7), wall contacts 6 (15) of 128
    mp = dict(PLAN_MAP, speed_lookahead=0.0, t_clip_min=0.35, m_l1=0.47, q_l1=0.0) if tw else \
        dict(PLAN_MAP, speed_lookahead=float(a.get("plan_lookahead", 0.235)))
    return dict(action_mode="plan", speed_mode="wp4", speed_command="tracker", plan_follower="map",
                plan_follower_map=mp,
                decision_every=int(a.get("repeat", 2) or 2), reverse_ext=0.0, v_min_policy=0.0,
                plan_reverse_ext=float(a.get("plan_reverse_ext", 0.0) or 0.0),
                learner_plan=str(a.get("plan_format")) if tw else "")


def soft_bound(z, lo, hi, m):
    """the identity on [lo + m, hi - m]; beyond, lo / hi approached smoothly (tanh, slope 1 at the knee)"""
    up, dn = hi - m, lo + m
    return torch.where(z > up, up + m * torch.tanh((z - up) / m), torch.where(z < dn, dn + m * torch.tanh((z - dn) / m), z))


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


def grad_conflict(model, fo, fa, fadv, fcase, fm, n_max=16384):
    """`--grad-diag` (11-04, the user: "gradient conflict ?"): the actor's policy gradient (PPO's at ratio 1: -E[adv log pi])
    of each world (`world_case`: empty / car / props / both) on this update's batch, and of two random halves of each world.
    Logs |g| per world, the cosine between every pair, and each world's half-vs-half cosine (does its gradient have a
    direction at all, or is it noise?). Conflict = a world with a consistent direction (halves cos > 0.3) whose cosine to
    the others is < 0; no signal = halves cos ~ 0 (then gradient surgery has nothing to keep)."""
    params = [p for p in model.pi.parameters() if p.requires_grad] + [model.log_std]
    def g_of(idx):
        if idx.numel() < 64:
            return None
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:n_max]]
        d = model.dist(fo[idx].float())
        lp = d.log_prob(fa[idx]).sum(1)
        w = fm[idx]
        loss = -(lp * fadv[idx] * w).sum() / w.sum().clamp_min(1.0)
        gs = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])
    cos = lambda x, y: float((x @ y) / (x.norm() * y.norm() + 1e-12)) if x is not None and y is not None else None
    full, half = [], []
    for c in range(4):
        idx = torch.nonzero(fcase == c).squeeze(1)
        full.append(g_of(idx))
        h = idx[torch.randperm(idx.numel(), device=idx.device)]
        k = h.numel() // 2
        half.append(cos(g_of(h[:k]), g_of(h[k:])))
    return {"norm": [float(g.norm()) if g is not None else None for g in full],
            "cos": [[cos(full[i], full[j]) for j in range(4)] for i in range(4)],
            "half_cos": half}


def world_case(env, lid):
    """(B,) long: 2 x (a live prop on the track) + (an opponent in the race) -- empty 0, car 1, props 2, both 3"""
    po = env.procedural
    props = (po.p_sid[lid] >= 0).any(1) if po is not None and po.C > 0 else torch.zeros(lid.numel(), dtype=torch.bool, device=lid.device)
    opp = torch.zeros_like(props)
    if env.M > 1:
        opp = env._opp_rel(env.last_result.state)[-1][lid].any(1)
    return 2 * props.long() + opp.long()


def residual_of(ar):
    """`--residual-solo`: (the frozen solo policy's obs width, its hidden width), or None"""
    ar = _ar(ar)
    return (int(ar["residual_d0"]), int(ar.get("residual_h0") or 256)) if ar.get("residual_solo") else None


class ResidualPi(nn.Module):
    """`--residual-solo` (11-04): the actor mean = a FROZEN solo policy + a learned change that is exactly 0 while no prop
    and no car is in view. One network in one world (empty track / props / cars, srp25ows, srp26ows) slowed the empty-track
    lap 6.50 -> 6.71 -> 6.90 s at u50 / u100 (2026 7.40 -> 7.65 s at u50): most episodes pay for speed near props and cars
    and the net slows everywhere (10-31: the same in every race lineage, `--base`). change = R(x) - R(x with the visible
    prop / car inputs (`StateObs.vis0` on) at their empty value): on an empty track both are the same input. Unlike the
    `--base` mixture there is no hand-over: whatever is in view, this one actor drives. The solo policy's own clamp plus a
    soft-bounded change: the empty-track action is the solo one bit for bit, and the change never sits in a clamped zone."""

    def __init__(self, obs_dim, act_dim, d0, h0, hidden):
        super().__init__()
        self.d0 = int(d0)
        self.base = mlp(self.d0, act_dim, h0).requires_grad_(False)
        self.res = mlp(obs_dim, act_dim, hidden)
        nn.init.zeros_(self.res[-1].bias); self.res[-1].weight.data.mul_(0.01)
        self.register_buffer("vis_mask", torch.zeros(obs_dim, dtype=torch.bool))
        self.register_buffer("vis_empty", torch.zeros(obs_dim))
        self.register_buffer("lo", torch.full((act_dim,), -1e6))
        self.register_buffer("hi", torch.full((act_dim,), 1e6))
        self.register_buffer("m", torch.ones(act_dim))

    def forward(self, x):
        with torch.no_grad():
            c = torch.maximum(torch.minimum(self.base(x[..., :self.d0]), self.hi), self.lo)
        d = self.res(x) - self.res(torch.where(self.vis_mask, self.vis_empty, x))
        return c + soft_bound(c + d, self.lo, self.hi, self.m) - soft_bound(c, self.lo, self.hi, self.m)


class BevNet(nn.Module):
    """obs = [vector, grid (C * H * W) last] -> a small conv over the grid, joined with the vector, -> the MLP"""

    def __init__(self, obs_dim, bev, out, hidden=256, feat=128):
        super().__init__()
        C, H, W = (int(x) for x in bev)
        self.bev = (C, H, W); self.vec = obs_dim - C * H * W
        self.conv = nn.Sequential(nn.Conv2d(C, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten())
        n = int(self.conv(torch.zeros(1, C, H, W)).shape[1])
        self.proj = nn.Sequential(nn.Linear(n, feat), nn.Tanh())
        self.mlp = mlp(self.vec + feat, out, hidden)

    def trunk(self, o):
        C, H, W = self.bev
        g = o[..., self.vec:].reshape(-1, C, H, W)
        z = torch.cat([o[..., :self.vec].reshape(-1, self.vec), self.proj(self.conv(g))], -1)
        return self.mlp[:-1](z).reshape(*o.shape[:-1], -1)

    def forward(self, o):
        return self.mlp[-1](self.trunk(o))


def widen_mlp_state(sd, h0, h):
    """`--hidden` (11-03): a 2-hidden-layer actor / critic MLP state (`mlp`) widened h0 -> h units per layer, the function
    unchanged (Net2Net-style, Chen et al. 2016): the new units read every input with fresh random weights, nothing reads
    them yet (zero weights from them into the old second layer, the output and the hit head). The new units' outgoing
    weights get gradient from the first update. 11-03 question: is the 2 x 256 teacher too small to hold the solo line and
    the prop behaviour at once (each props continuation traded pace for fewer hits)?"""
    sd = dict(sd)
    for net in ("pi", "v"):
        w0, b0 = sd[f"{net}.0.weight"], sd[f"{net}.0.bias"]
        w2, b2 = sd[f"{net}.2.weight"], sd[f"{net}.2.bias"]
        w4 = sd[f"{net}.4.weight"]
        l0, l2 = nn.Linear(w0.shape[1], h - h0), nn.Linear(h, h - h0)
        dev, dt = w0.device, w0.dtype
        sd[f"{net}.0.weight"] = torch.cat([w0, l0.weight.data.to(dev, dt)], 0)
        sd[f"{net}.0.bias"] = torch.cat([b0, l0.bias.data.to(dev, dt)], 0)
        top = torch.cat([w2, torch.zeros(h0, h - h0, device=dev, dtype=dt)], 1)
        sd[f"{net}.2.weight"] = torch.cat([top, l2.weight.data.to(dev, dt)], 0)
        sd[f"{net}.2.bias"] = torch.cat([b2, l2.bias.data.to(dev, dt)], 0)
        sd[f"{net}.4.weight"] = torch.cat([w4, torch.zeros(w4.shape[0], h - h0, device=dev, dtype=dt)], 1)
    if "hit.weight" in sd:
        w = sd["hit.weight"]
        sd["hit.weight"] = torch.cat([w, torch.zeros(w.shape[0], h - h0, device=w.device, dtype=w.dtype)], 1)
    return sd


#: The actor never reads the lap position (11-03, user: "랩 위치를 주지 마 ... 숏컷러닝"): `privileged_v2`'s lap_frac is a
#: global place index the LiDAR student has no counterpart for (its inputs: scan frames, speed, gyro, past actions). Every
#: StateObs layout starts with privileged_v2's ego keys, so it is this column everywhere; the critic keeps it.
ACTOR_NO_LAP = PRIV_V2_EGO_KEYS.index("lap_frac")


def _hide_lap(m, _incompatible):
    """load_state_dict post-hook: a checkpoint saved before 11-03 carries actor_keep 1 at the lap column"""
    m.actor_keep[ACTOR_NO_LAP] = 0.0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=2, log_std=-0.5, hidden=256, std_floor=None, bev=None, shared_enc=False, residual=None):
        super().__init__()
        self.bev = tuple(bev) if bev else None
        self.residual = residual is not None
        #: `--std-floor` (10-29): the race teachers cut their speed noise 0.66 -> 0.20 m/s, the cheapest way to fewer hits,
        #: after which no cautious maneuver (3-5 m/s less behind a car) is ever tried; the floor keeps it explored
        self.register_buffer("std_floor", torch.tensor(std_floor if std_floor is not None else [0.0] * act_dim, dtype=torch.float32))
        if self.bev:
            self.pi = BevNet(obs_dim, self.bev, act_dim, hidden)
            self.v = BevNet(obs_dim, self.bev, 1, hidden)
            if shared_enc:
                # `--shared-enc` (10-30): ONE grid encoder for actor and critic -- the value / hit-head losses are supervised and
                # build prop / car features fast, the policy gradient alone did not (srl31 100 updates: the critic's first-layer
                # prop weights 0 -> 0.106, the actor's 0 -> 0.022; the actor's steering ignored the prop channel)
                self.v.conv = self.pi.conv; self.v.proj = self.pi.proj
            last_pi = self.pi.mlp[-1]
        elif self.residual:
            self.pi = ResidualPi(obs_dim, act_dim, *residual, hidden)
            self.v = mlp(obs_dim, 1, hidden)
            last_pi = None
        else:
            self.pi = mlp(obs_dim, act_dim, hidden)
            self.v = mlp(obs_dim, 1, hidden)
            last_pi = self.pi[-1]
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))
        if last_pi is not None:
            nn.init.zeros_(last_pi.bias); last_pi.weight.data.mul_(0.01)
        #: `--soft-bound` (11-01): the policy MEAN is squashed into the executable range near its ends -- the identity on
        #: [lo + m, hi - m], smoothly saturating to lo / hi beyond (`soft_bound`). A clamped Gaussian's mean drifts far past
        #: the bound where the bound is good (measured: mix0's race policy 34 % of its decisions speed-dead, 31 % steer-dead,
        #: 47 % speed-dead in the second before a hard hit); there every sample executes the bound and PPO cannot learn that
        #: dimension (the command stays 10 m/s into a slow car). Buffers, so every tool that loads the checkpoint drives the
        #: same mean (a forward hook on `pi`); a checkpoint without them loads +-1e6 = the identity. `raw_mean`: the last
        #: pre-squash output (`--bound-loss`)
        self.register_buffer("mean_lo", torch.full((act_dim,), -1e6))
        self.register_buffer("mean_hi", torch.full((act_dim,), 1e6))
        self.register_buffer("mean_m", torch.ones(act_dim))
        self.raw_mean = None
        self.pi.register_forward_hook(self._soft_bound)
        #: `--vis-actor` (11-03): which normalised inputs the ACTOR reads (1) -- 0 on the full-truth prop slots and corridor
        #: and the time left, which only the critic keeps (`StateObs.actor_hide`). A buffer, so every tool that loads the
        #: checkpoint drives the actor the student is taught by; a checkpoint without it loads all ones (as before)
        self.register_buffer("actor_keep", torch.ones(obs_dim))
        self.actor_keep[ACTOR_NO_LAP] = 0.0
        self.register_load_state_dict_post_hook(_hide_lap)
        self.pi.register_forward_pre_hook(lambda _m, inp: (inp[0] * self.actor_keep,) + tuple(inp[1:]))
        #: P(a hard hit within the next HIT_HORIZON decisions | s), a head on the critic's trunk (`--hit-head`): supervised
        #: on every step, so the critic's features learn what danger looks like before the price of a hit reaches them
        self.hit = nn.Linear(hidden, 1)

    def set_mean_bounds(self, lo, hi, m):
        dev = self.mean_lo.device
        self.mean_lo.copy_(torch.tensor(lo, device=dev)); self.mean_hi.copy_(torch.tensor(hi, device=dev))
        self.mean_m.copy_(torch.tensor(m, device=dev))
        if self.residual:
            self.pi.lo.copy_(self.mean_lo); self.pi.hi.copy_(self.mean_hi); self.pi.m.copy_(self.mean_m)

    def _soft_bound(self, _mod, _inp, z):
        self.raw_mean = z
        return z if self.residual else soft_bound(z, self.mean_lo, self.mean_hi, self.mean_m)

    def critic_trunk(self, o):
        return self.v.trunk(o) if self.bev else self.v[:-1](o)

    def value_and_hit(self, o):
        z = self.critic_trunk(o)
        head = self.v.mlp[-1] if self.bev else self.v[-1]
        return head(z).squeeze(-1), self.hit(z).squeeze(-1)

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), torch.maximum(self.log_std.exp(), self.std_floor))


def train(a):
    device = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    env = build_env(a, device)
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    obs_fn = StateObs(env, scan=bool(a.scan_obs), corridor=bool(a.corridor_obs), corridor_cars=bool(a.corridor_cars),
                      car_channel=bool(a.car_channel), bev=bool(a.bev), path_plan=int(a.path_plan), plan_ref=a.plan_ref,
                      car_st=bool(a.car_st), scan_nocars=bool(a.scan_nocars),
                      corridor_car_st=bool(a.corridor_car_st), blank_opp=bool(a.blank_opp), car_st_line=bool(a.car_st_line),
                      vis_actor=bool(a.vis_actor), car_slots=bool(a.car_slots), scan_points=bool(a.scan_points))
    if a.scan_points and not a.scan_obs:
        raise SystemExit("--scan-points needs --scan-obs 1")
    if a.car_slots and not a.vis_actor:
        raise SystemExit("--car-slots needs --vis-actor (the slots are the cars the car itself can see)")
    if float(a.level_replay) > 0:
        # prioritized level replay (Jiang et al. 2021, `learn.level_replay`): resets replay the layouts / spawns whose
        # advantages were largest -- the situations the policy gets most wrong come back more often
        from .level_replay import LevelReplay
        env.level_replay = LevelReplay(env.B // env.M, p_replay=float(a.level_replay), size=int(a.level_replay_size),
                                       seed=int(a.seed))
    env.reset(seed=int(a.seed))
    lid = env.learner_ids
    B = int(lid.numel())
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    A = act_dim_of(a)
    if plan_of(a) and (int(a.unwedge) or float(a.reverse_ext)):
        raise SystemExit("--action plan: no --unwedge or --reverse-ext (MAP backs a stalled car out); --base must be a plan teacher")
    if int(env.act_dim) != A:
        raise SystemExit(f"the env's action is {env.act_dim}-wide, this run's {A}")
    if float(a.plan_relabel) > 0 and not plan_of(a):
        raise SystemExit("--plan-relabel relabels wp4 plans: it needs --action plan")
    if tw_of(a) and (float(a.plan_relabel) > 0 or float(a.plan_reverse_ext) > 0):
        raise SystemExit("--plan-format tw: no --plan-relabel (it relabels wp4 geometry) and no --plan-reverse-ext (time "
                         "waypoint plans reverse in their own speeds)")
    last_a = torch.zeros(B, A, device=device)
    o = obs_fn(last_a)
    D = obs_fn.dim
    bev = (BEV_C, BEV_H, BEV_W) if a.bev else None
    ck_solo = None
    if a.residual_solo:
        if a.bev or a.base or not a.vis_actor or obs_fn.vis0 is None:
            raise SystemExit("--residual-solo: plain MLP with --vis-actor, no --base")
        ck_solo = torch.load(a.residual_solo, map_location=device, weights_only=False)
        if ck_solo["args"].get("stage") != "solo" or ck_solo["args"].get("bev"):
            raise SystemExit("--residual-solo takes a plain solo-stage checkpoint")
        a.residual_d0, a.residual_h0 = int(ck_solo["obs_dim"]), int(ck_solo["args"].get("hidden") or 256)
    model = ActorCritic(D, act_dim=A, log_std=a.log_std, std_floor=std_floor_of(a, A),
                        bev=bev, shared_enc=bool(a.shared_enc), hidden=int(a.hidden), residual=residual_of(a)).to(device)
    norm = RunningNorm(D, device, raw_tail=obs_fn.grid_dim)
    ret_rms = RunningNorm(1, device)
    upd0 = 0
    if a.init:
        ck = torch.load(a.init, map_location=device, weights_only=False)
        d0 = int(ck.get("obs_dim", D))
        sd = ck["model"]
        if a.scan_points and not ck["args"].get("scan_points"):
            sd, ck["norm"] = grow_to_scan_points(sd, ck["norm"], obs_fn.scan0, int(ck["args"].get("beams", 108) or 108),
                                                 int(a.beams), obs_fn.vis0, D, device)
            sd.pop("actor_keep", None)
            print(f"[state_rl] --init 관측 {d0} -> {D}차원: scan {int(ck['args'].get('beams', 108) or 108)} -> {int(a.beams)}빔(가까운 옛 빔의 "
                  f"가중치를 나눠 씀), 상자·차 점 줄은 가중치 0, 보이는 상자 slot은 뺌", flush=True)
            d0 = D
        if d0 < D:
            # the new inputs (the scan) enter with zero weights: the run starts as the --init policy and learns to use them
            for k in ("pi.0.weight", "v.0.weight"):
                w = sd[k]; sd[k] = torch.cat([w, torch.zeros(w.shape[0], D - d0, device=w.device, dtype=w.dtype)], 1)
            nm = ck["norm"]
            ck["norm"] = {"mean": torch.cat([nm["mean"], torch.zeros(D - d0, device=device)]),
                          "var": torch.cat([nm["var"], torch.ones(D - d0, device=device)]), "n": nm["n"]}
            sd.pop("actor_keep", None)             # rebuilt from this run's StateObs.actor_hide below, at the new width
            print(f"[state_rl] --init 관측 {d0} -> {D}차원: 새 입력은 가중치 0으로 시작", flush=True)
        h0 = int(ck["args"].get("hidden", 256) or 256)
        if h0 < int(a.hidden):
            if a.bev:
                raise SystemExit("--hidden: widening an --init is for the plain MLP teachers only")
            sd = widen_mlp_state(sd, h0, int(a.hidden))
            print(f"[state_rl] --init 은닉 {h0} -> {int(a.hidden)}: 새 단위는 나가는 가중치 0 (처음엔 부모와 같은 출력)", flush=True)
        elif h0 > int(a.hidden):
            raise SystemExit(f"--hidden {a.hidden} is narrower than the --init's {h0}")
        if a.bev and a.shared_enc and not ck["args"].get("shared_enc"):
            # the shared encoder starts as the ACTOR's (its head keeps the features it was trained on; the critic re-fits)
            for k in [k for k in sd if k.startswith("pi.conv.") or k.startswith("pi.proj.")]:
                sd["v." + k[3:]] = sd[k].clone()
            print("[state_rl] --init: 공유 격자 인코더를 부모 정책의 것에서 시작", flush=True)
        if a.bev and ck["args"].get("stage") == "solo" and a.stage != "solo":
            # a solo parent never saw a prop or a car: its first conv layer's weights on those grid channels (1..) got no
            # gradient and are still the random init -- zero them, so the first update drives as the parent did
            for k in ("pi.conv.0.weight", "v.conv.0.weight"):
                sd[k] = sd[k].clone(); sd[k][:, 1:] = 0.0
            print("[state_rl] --init 혼자 부모: 격자 상자·차 채널의 첫 합성곱 가중치를 0에서 시작", flush=True)
        if a.car_st and a.car_channel and a.corridor_obs and not ck["args"].get("car_st") and d0 == D:
            # the space-time car channel starts as the parent's PROP corridor (same 13 x 7 layout, same meaning: these bins
            # will be occupied when we get there): its weights and input statistics copied over the old car channel's
            n_cb = len(AHEAD) * CORRIDOR_BINS
            cc, pc = slice(D - n_cb, D), slice(D - 2 * n_cb, D - n_cb)
            for k in ("pi.0.weight", "v.0.weight"):
                sd[k] = sd[k].clone(); sd[k][:, cc] = sd[k][:, pc]
            nm = ck["norm"]
            nm["mean"] = nm["mean"].clone(); nm["var"] = nm["var"].clone()
            nm["mean"][cc] = nm["mean"][pc]; nm["var"][cc] = nm["var"][pc]
            print("[state_rl] --init: 시공간 차 칸을 부모의 상자 통로 가중치·통계에서 시작", flush=True)
        missing = [k for k in model.load_state_dict(sd, strict=False).missing_keys
                   if k not in ("std_floor", "mean_lo", "mean_hi", "mean_m", "actor_keep")]
        if a.std_floor:                                    # this run's floor, not the one an --init checkpoint carried
            model.std_floor.copy_(torch.tensor(std_floor_of(a, A), device=device))
        if missing:
            print(f"[state_rl] --init에 없는 매개변수(새로 시작): {missing}", flush=True)
        norm.load_state_dict(ck["norm"]); ret_rms.load_state_dict(ck["ret_rms"])
        if float(a.log_std_shift):
            # `--log-std-shift` (11-03): PPO optimises the noisy policy, so the margins it keeps scale with the noise (g 8.21 s
            # with correlated noise, g2 7.88 s after white noise); log_std barely moves on its own (10-27 audit)
            with torch.no_grad():
                model.log_std.add_(float(a.log_std_shift))
            print(f"[state_rl] --init log_std {float(a.log_std_shift):+.3f} -> 표준편차 {[round(float(x), 3) for x in model.log_std.exp()]}", flush=True)
    if obs_fn.actor_hide:
        model.actor_keep.fill_(1.0); model.actor_keep[torch.tensor(obs_fn.actor_hide, device=device)] = 0.0
        print(f"[state_rl] 배우는 쪽이 못 보는 {len(obs_fn.actor_hide)}칸(랩 위치{', 전체 정답 상자 칸·통로, 남은 시간' if a.vis_actor else ''}) -- 비평가만 봄", flush=True)
    if int(a.soft_bound):
        model.set_mean_bounds(*mean_bounds(a, A))
    elif bool((model.mean_hi < 1e5).any()):
        print("[state_rl] --init의 정책 평균 부드러운 범위를 이어받음", flush=True)
    if ck_solo is not None:
        # the frozen solo policy, its input statistics in the first d0 columns (the obs starts with the solo obs), all
        # statistics frozen; the visible-prop / car columns' empty value as the actor reads it (normalised, actor_keep)
        d0 = model.pi.d0
        model.pi.base.load_state_dict({k[3:]: v for k, v in ck_solo["model"].items() if k.startswith("pi.")})
        sn = ck_solo["norm"]
        norm.mean = norm.mean.clone(); norm.var = norm.var.clone()
        norm.mean[:d0] = sn["mean"].to(device); norm.var[:d0] = sn["var"].to(device)
        norm.frozen = True
        with torch.no_grad():
            model.pi.vis_mask.zero_(); model.pi.vis_mask[obs_fn.vis0:] = True
            model.pi.vis_empty.copy_(norm(torch.zeros(1, D, device=device))[0] * model.actor_keep)
        if model.pi.lo[0] < -1e5:
            # the solo policy ran under the env's clamp; the change is bounded the same way
            model.set_mean_bounds(*mean_bounds(a, A))
        if int(a.residual_init):
            # `--residual-init` (11-04 14:45): R starts as the --init checkpoint's ACTOR, so the change starts as how that
            # policy changes its action for what it sees (a zero-started R learned no avoiding in 100 updates: srp25owr u100
            # hard 272 on the 30 console layouts, the solo line straight into the crates). Its own input statistics are
            # folded into R's first layer: x_init = x * s_f / s_i + (m_f - m_i) / s_i on the columns the actor reads
            if not a.init or ck["model"].get("pi.0.weight") is None or ck["model"]["pi.0.weight"].shape != model.pi.res[0].weight.shape:
                raise SystemExit("--residual-init needs an --init plain actor of this obs width and hidden size")
            ni = ck["norm"]
            s_i, s_f = ni["var"].to(device).sqrt() + 1e-6, norm.var.sqrt() + 1e-6
            sc = s_f / s_i
            sh = (norm.mean - ni["mean"].to(device)) / s_i * model.actor_keep
            r_sd = {k[3:]: v.clone() for k, v in ck["model"].items() if k.startswith("pi.") and k[3:4].isdigit()}
            W = r_sd["0.weight"]
            r_sd["0.weight"] = W * sc[None, :]; r_sd["0.bias"] = r_sd["0.bias"] + W @ sh
            model.pi.res.load_state_dict(r_sd)
            print(f"[state_rl] --residual-init: 보정 R을 {a.init}의 actor에서 시작(입력 통계 첫 층에 합침)", flush=True)
        print(f"[state_rl] --residual-solo {a.residual_solo}: 고정된 solo policy(관측 {d0}) + 보이는 obstacle·차 입력 "
              f"{D - obs_fn.vis0}칸이 있을 때만 0이 아닌 보정, 입력 통계 고정", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    w_col, w_car = abs(env.ecfg.reward_collision), abs(env.ecfg.reward_car_contact)
    ci = {k: REWARD_COMPONENT_KEYS.index(k) for k in REWARD_COMPONENT_KEYS}
    out = os.path.expanduser(f"~/f1sim_runs/{a.name}"); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    log = open(os.path.join(out, "progress.jsonl"), "a")
    T, R = int(a.horizon), int(a.repeat)
    touching = torch.zeros(B, dtype=torch.bool, device=device)
    eps_n = torch.randn(B, A, device=device)                       # the correlated exploration state (--noise-corr)
    lo_b, hi_b, _m = mean_bounds(a, A)
    lo_b, hi_b = torch.tensor(lo_b, device=device), torch.tensor(hi_b, device=device)
    d_steer, d_speed = steer_speed_dims(a, A)
    remover = PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=int(a.seed)) \
        if float(a.removal_p) > 0 else None
    unwedge = Unwedge(env, lid, mode=int(a.unwedge)) if int(a.unwedge) else None
    # the gated mixture (`--base`): the solo teacher drives where nothing is near; the policy gradient only on the
    # decisions this policy drove (elsewhere its sampled action was not executed), the critic on all
    mix = SoloMix(env, a.base, device, float(a.base_reach), int(a.base_hold), visible=bool(a.vis_actor)) if a.base else None
    #: `--anchor-solo` (11-04): ONE network, but where nothing is near (SoloMix's own test: no visible prop in the corridor
    #: within reach, no car within reach, for `hold` decisions) its mean action is pulled onto the solo policy's -- every PPO
    #: variant in the mixed world lost empty-track pace (6.50 -> 6.70 s by u50: settings, hit-shaping, per-world advantage
    #: normalisation, a speed-convex reward, a supervised start), while a supervised fit of one network kept 6.50 s: PPO has
    #: no consistent 'faster' gradient on the empty track (per-world gradient cosines, `--grad-diag`), so nothing pulls back
    anchor = SoloMix(env, a.anchor_solo, device, float(a.base_reach), int(a.base_hold), visible=bool(a.vis_actor)) \
        if a.anchor_solo else None
    #: `--calm-std` (11-04): the exploration std x this where nothing is near (`CalmGate`). On the empty track the training
    #: noise's rare hard wall hits (price 80) set the reward's spread: 10 s windows 58.7 +- 13.7 (6.50 s policy) vs 54.6 +- 14.9
    #: (6.90 s), the pace gap 0.29 sd, 1.3 se over one update (`tools/reward_snr.py`) -- the time trial needs precision, the
    #: props / cars need exploration; one std for both serves the second only
    calm = CalmGate(env, float(a.base_reach), int(a.base_hold), visible=bool(a.vis_actor)) if float(a.calm_std) != 1.0 else None
    if anchor is not None and mix is not None:
        raise SystemExit("--anchor-solo is for one network: no --base")
    ptp = float(getattr(a, "prop_touch_price", 0.0) or 0.0)
    ptv = float(getattr(a, "prop_touch_vmax", 0.0) or 0.0)
    # --stall-mode progress: each learner's progress over the last --stall-window s (ring buffer of per-step progress)
    W_st = max(1, int(round(float(a.stall_window) / env.sim.control_dt)))
    win_buf = torch.zeros(W_st, B, device=device); win_sum = torch.zeros(B, device=device)
    win_n = torch.zeros(B, dtype=torch.long, device=device); win_ptr = 0
    posb = float(getattr(a, "pos_bonus", 0.0) or 0.0)
    rpos = RacePosition(env, lid, void_s=float(getattr(a, "pos_void", 0.0) or 0.0)) if posb > 0 and env.M > 1 else None
    print(f"[state_rl] {a.name}: 단계 {a.stage}, 환경 {env.B} (학습 행 {B}), 관측 {D}차원, 결정 {40 // R} Hz, "
          f"롤아웃 {T}결정 = {T * R} 스텝, 가격 {a.price:g}", flush=True)
    t0 = time.time()
    for u in range(upd0 + 1, upd0 + int(a.updates) + 1):
        frac = (u - upd0 - 1) / max(1, int(a.updates))
        for g in opt.param_groups:
            g["lr"] = a.lr + (a.lr_end - a.lr) * frac
        # the hit price ramps in (--price-hold, --price-ramp): from scratch, a full price at once teaches the car to stand
        # still (measured 10-29: 2.0 -> 0.2 m/s in 10 updates); progress first, then the price, then the full objective
        k = (u - upd0 - 1 - int(a.price_hold)) / max(1, int(a.price_ramp))
        price = a.price_start + (a.price - a.price_start) * min(1.0, max(0.0, k))
        # the --bev observation is ~7.9k wide: held in half precision (occupancy in [0, 1], velocities /5), 2 GB at 2048 envs
        bo = torch.zeros(T, B, D, device=device, dtype=torch.float16 if a.bev else torch.float32)
        ba = torch.zeros(T, B, A, device=device)
        blp = torch.zeros(T, B, device=device); bv = torch.zeros(T + 1, B, device=device)
        br = torch.zeros(T, B, device=device); bd = torch.zeros(T, B, device=device)
        bh = torch.zeros(T, B, device=device); bp = torch.zeros(T + 1, B, device=device)
        bg = torch.zeros(T + 1, B, device=device)
        bm = torch.ones(T, B, device=device)                     # 1 = this policy's action was executed (`--base`)
        bcase = torch.zeros(T, B, dtype=torch.long, device=device)  # `--adv-norm-case`: 2 x props on track + an opponent in the race
        bsolo = torch.zeros(T, B, A, device=device) if anchor is not None else None    # `--anchor-solo`: its action
        banch = torch.zeros(T, B, device=device)                                    # 1 = nothing near: anchored
        bscale = torch.ones(T, B, device=device)                                    # `--calm-std`: the std factor used
        bfl = torch.zeros(T, B, A, device=device) if a.slow_explore else None        # `--slow-explore`: the std floor used
        blv =torch.full((T, B), -1, dtype=torch.long, device=device) if env.level_replay is not None else None
        prog = hits_w = hits_c = prop_s = push_s = 0.0; laps = []; dev_sum = dev_n = 0.0; n_gain = n_lost = n_void = 0.0
        relabel = float(getattr(a, "plan_relabel", 0.0) or 0.0) > 0
        if relabel:                                              # `--plan-relabel`: the path each plan was followed by
            pose_s = torch.zeros(T * R + 1, B, 4, device=device); d_s = torch.zeros(T * R, B, dtype=torch.bool, device=device)
            vm_t = torch.zeros(T, B, device=device)
        dead_n = torch.zeros(3, device=device)                   # decisions this policy drove; of them speed / steer dead
        tr0 = time.time()
        with torch.no_grad():
            for t in range(T):
                norm.update(o)
                on = norm(o)
                dist = model.dist(on)
                if calm is not None:
                    bscale[t] = torch.where(calm.step(), torch.ones(B, device=device), torch.full((B,), float(a.calm_std), device=device))
                    dist = torch.distributions.Normal(dist.mean, dist.stddev * bscale[t][:, None])
                if bfl is not None:
                    # `--slow-explore V0 STEER SPEED` (11-06): a std floor that lives where the car is slow. A flat floor of 0.2
                    # (srp25exp) is 0.32 1/m of curvature at any speed -- 8 m/s^2 of lateral noise at 5 m/s -- and took the
                    # training from ~5 to 80 hard hits per km; a stopped car needs the whole range to find a way round a box
                    # (about 0.4 of steer held ~0.5 s, ~10 sigma of the collapsed 0.04). The floor scales as (V0 / v)^2, a
                    # fixed lateral acceleration like `kappa_mode feasible`: full below V0, a quarter at 2 V0
                    v0_, fk_, fs_ = (float(x) for x in a.slow_explore)
                    s_ = (v0_ / env.sim.state[lid, 3].float().abs().clamp_min(1e-3)).square().clamp(max=1.0)
                    bfl[t, :, :kappa_dims_of(a)] = fk_ * s_[:, None]
                    bfl[t, :, kappa_dims_of(a):] = fs_ * s_[:, None]
                    dist = torch.distributions.Normal(dist.mean, torch.maximum(dist.stddev, bfl[t]))
                if float(a.noise_corr) > 0:
                    # temporally correlated exploration (AR(1), marginal N(0, 1)): white noise at 20 Hz averages out within
                    # a second, so a sampled deviation never becomes a maneuver (brake 0.3 s earlier, hold a gap, move the
                    # line for a second) whose consequence the advantage could see; held noise does. PPO's per-decision
                    # ratio uses the Gaussian marginal, as gSDE does (Raffin et al. 2021)
                    al = float(a.noise_corr)
                    if float(getattr(a, "corr_slow", 0.0) or 0.0) > 0:
                        # `--corr-slow V0` (11-07): the hold only where the car is slow, x (V0 / v)^2 above -- held at speed, a
                        # fresh network's curvature noise (~0.45 of 1.6 1/m for 0.5 s) is ~18 m/s^2 sideways at 5 m/s: srp25tw2's
                        # training hit walls at speed 26 times a km and the policy slowed to a stop as the price rose; the same
                        # checkpoint with the hold below 1.5 m/s only hit 14.7 a km in all (38.2)
                        s2_ = (float(a.corr_slow) / env.sim.state[lid, 3].float().abs().clamp_min(1e-3)).square().clamp(max=1.0)
                        al_t = (al * s2_)[:, None]
                        eps_n = al_t * eps_n + torch.sqrt(1.0 - al_t * al_t) * torch.randn_like(eps_n)
                    else:
                        eps_n = al * eps_n + math.sqrt(1.0 - al * al) * torch.randn_like(eps_n)
                    act = dist.mean + dist.stddev * eps_n
                else:
                    act = dist.sample()
                bo[t] = on; ba[t] = act; blp[t] = dist.log_prob(act).sum(1)
                bcase[t] = world_case(env, lid)
                if blv is not None:
                    blv[t] = env.level_seed[lid]
                v_t, hl_t = model.value_and_hit(on)
                bv[t] = v_t * (ret_rms.var.sqrt() + 1e-6); bp[t] = torch.sigmoid(hl_t)
                if float(a.geo_shaping) > 0:
                    bg[t] = geo_risk(env, lid)
                rew = torch.zeros(B, device=device); done = torch.zeros(B, dtype=torch.bool, device=device)
                hit_now = torch.zeros(B, dtype=torch.bool, device=device)
                exe = clamp_any(act, a)
                if anchor is not None:
                    a_s, g_s = anchor.solo(last_a)
                    if a.anchor_gate == "seen":
                        # `--anchor-gate seen` (11-05, the user: "장애물이나 상대차가 ego 라이다 포인트에 찍히는 순간부터는 정책
                        # supervised를 안쓰게" + "벽 안쪽만 해당이여"): anchored only while no prop and no car is in the LiDAR's view
                        # (`visible_props` / `visible_cars`: in the student's grid, in range, in line of sight -- a wall in front hides
                        # it). The 10 m corridor gate also anchored states with props in view beside the corridor, so the solo target
                        # taught 'ignore what you see' (srp25owa: 8-layout hard 9 -> 33 -> 18, 30-layout 41 -> 82 by u200)
                        seen = torch.zeros(B, dtype=torch.bool, device=device)
                        vp = visible_props(env, lid)
                        if vp is not None:
                            seen |= vp.any(1)
                        vc = visible_cars(env, lid)
                        if vc is not None:
                            seen |= vc.any(1)
                        g_s = seen
                    bsolo[t] = a_s; banch[t] = (~g_s).float()
                if mix is not None:
                    exe = mix(exe, last_a); bm[t] = mix.gate.float()
                # dead decisions (11-01): the mean beyond the action bound by more than 2 sigma -- every sample executes the
                # bound, so the advantage says nothing about that dimension there (`--bound-loss`)
                mu_, sd_ = model.raw_mean, dist.stddev                   # the network's own output, before any squash
                dd_ = (mu_ > hi_b + 2 * sd_) | (mu_ < lo_b - 2 * sd_)    # (B, A): this dimension dead here
                dead_n += torch.stack([bm[t].sum(), (bm[t] * dd_[:, d_speed].any(1).float()).sum(),
                                       (bm[t] * dd_[:, d_steer].any(1).float()).sum()])
                act_full[lid] = exe
                if relabel:
                    lr_ = env.last_result
                    vm_t[t] = (lr_.odom[lid, 3] if lr_ is not None else env.sim.state[lid, 3]).float()
                for k in range(R):
                    if relabel:
                        pose_s[t * R + k] = env.sim.state[lid, :4].float()
                    if unwedge is not None:
                        unwedge.before(act_full)
                    _, r_, te, trc, info = env.step(act_full)
                    if unwedge is not None:
                        unwedge.after()
                    if remover is not None:
                        remover.step()
                    comp = info["reward_components"]
                    col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
                    tch = car < 0
                    onset = tch & ~touching
                    touching = tch
                    onw = (-col / max(w_col, 1e-9)).clamp_min(0)
                    dense = r_[lid].float() - col - car
                    if float(a.speed_pow) != 1.0:
                        # `--speed-pow` (11-04, the user: "한 랩이 15->10초보다 7->6초로 가는게 훨씬 더 보상을 키워버리는거지"): the
                        # metre at speed v pays (v / --speed-ref)^(p - 1) -- progress alone pays a lap 15 -> 10 s (x1.5) more than
                        # 7 -> 6 s (x1.17); at p 3 the near-limit gain is worth more, and the empty track's small 'faster' share of
                        # the gradient grows p-fold against the props / cars' 'slower' one (audit: 6.50 -> 6.71 s by u50 in the mixed world)
                        ds_ = info["progress"][lid].float().clamp_min(0)
                        v_ = env.sim.state[lid, 3].float().clamp_min(0)
                        dense = dense + float(env.ecfg.reward_progress) * ds_ * ((v_ / float(a.speed_ref)) ** (float(a.speed_pow) - 1.0) - 1.0)
                    # --car-price: a hard car contact can end a 1 vs 1 race; at the shared price the teacher's tailgating
                    # into braking zones is break-even against a 0.5 s gap rule (10-29 gap test), so cars get their own
                    price_c = price * (float(a.car_price) / float(a.price)) if float(a.car_price) > 0 else price
                    step_r = dense - price * onw - price_c * onset.float()
                    # --prop-touch-price (11-02): every step in contact with a prop costs, at any speed. Only onsets at or
                    # above contact_severity_min were charged, so braking under it and shoving a crate off the line was free
                    # -- the props teachers did exactly that on 2-3 of 30 console layouts (stuck 4-11 s, 7-18 touches)
                    ptch = env.sim.kind_prev[lid, 1].float()
                    # --prop-touch-vmax: only while the car itself is slower than this -- a push, not a graze in passing (11-02:
                    # charged at any speed, the teacher kept a margin for its own exploration noise and lost 7-8 % of pace)
                    pch = ptch * (env.sim.state[lid, 3].abs() < ptv).float() if ptv > 0 else ptch
                    if ptp > 0:
                        step_r = step_r - ptp * (price / max(float(a.price), 1e-9)) * env.sim.control_dt * pch
                    if float(a.stall_price) > 0 and a.stall_mode == "progress":
                        # --stall-mode progress (11-06, the user on rocking back and forth to dodge a speed rule: "이거는 어떻게
                        # 방지하려나"): the rate is the progress along the lane over the last --stall-window s (signed), so standing,
                        # rocking and going the wrong way all pay; a quick back-out pays only while it lasts
                        p_now = info["progress"][lid].float()
                        win_sum = win_sum + p_now - win_buf[win_ptr]
                        win_buf[win_ptr] = p_now
                        win_ptr = (win_ptr + 1) % win_buf.shape[0]
                        win_n = (win_n + 1).clamp(max=win_buf.shape[0])
                        rate_ = win_sum / (win_n.float() * env.sim.control_dt)
                        short_ = ((float(a.stall_vmin) - rate_) / max(float(a.stall_vmin), 1e-6)).clamp(0.0, 1.0)
                        step_r = step_r - float(a.stall_price) * (price / max(float(a.price), 1e-9)) * env.sim.control_dt * short_
                        d_w = (te | trc)[lid]
                        if bool(d_w.any()):                                   # a new episode starts its own window
                            win_buf[:, d_w] = 0.0; win_sum = torch.where(d_w, torch.zeros_like(win_sum), win_sum)
                            win_n = torch.where(d_w, torch.zeros_like(win_n), win_n)
                    elif float(a.stall_price) > 0:
                        # --stall-price (11-06, the user: "실격이 아니라 지속적으로 벌점을 계속 주는건?"): every second below
                        # --stall-vmin costs the shortfall's share of the price (speed magnitude, so backing out is not charged). In
                        # the box layouts the policy stood still 75-82 % of the time: standing was free and a retry risked -80
                        sv_ = env.sim.state[lid, 3].float().abs()
                        short_ = ((float(a.stall_vmin) - sv_) / max(float(a.stall_vmin), 1e-6)).clamp(0.0, 1.0)
                        step_r = step_r - float(a.stall_price) * (price / max(float(a.price), 1e-9)) * env.sim.control_dt * short_
                    if obs_fn.planner is not None:
                        # drive the plan: leaving it by more than --track-dead costs --track-k per m and step
                        dev_ = obs_fn.planner.deviation().abs()
                        w_ = torch.ones_like(dev_)
                        if a.track_where == "shifted":
                            # only where the plan leaves the raceline (to get round something): the open track is the
                            # policy's own (10-30: tracking everywhere held srl40 at 3.7-3.8 m/s for 50 updates)
                            pl_ = obs_fn.planner
                            s_l, t_l = env.last_result.s[lid], env.sim.tid[lid]
                            e_ = (pl_.offset(s_l[:, None], t_l)[:, 0] - pl_.rl_offset(s_l, t_l)).abs()
                            w_ = (e_ / 0.2).clamp(0.0, 1.0)
                        step_r = step_r - float(a.track_k) * w_ * (dev_ - float(a.track_dead)).clamp_min(0.0)
                        dev_sum += float((dev_ * (~done).float()).sum()); dev_n += float((~done).float().sum())
                    alive = (~done).float()
                    if rpos is not None:
                        paid_, lost_, gain_ = rpos.step(onset)
                        step_r = step_r + posb * (paid_ - lost_)
                        n_gain += float((gain_ * alive).sum()); n_lost += float((lost_ * alive).sum())
                        n_void += float(((gain_ - paid_) * alive).sum())
                    rew += step_r * alive
                    prog += float((info["progress"][lid].float().clamp_min(0) * alive).sum())
                    prop_s += float((ptch * alive).sum()) * env.sim.control_dt
                    push_s += float((ptch * (env.sim.state[lid, 3].abs() < 1.5).float() * alive).sum()) * env.sim.control_dt
                    hits_w += float((onw * alive).sum()); hits_c += float((onset.float() * alive).sum())
                    hit_now |= ((onw > 0) | onset) & ~done
                    d_ = (te | trc)[lid]
                    if relabel:
                        d_s[t * R + k] = d_
                    touching = touching & ~d_
                    if rpos is not None:
                        rpos.reset(d_)
                    done |= d_
                    if info["lap_times"].numel():
                        laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                br[t] = rew; bd[t] = done.float(); bh[t] = hit_now.float()
                last_a = torch.where(done[:, None], torch.zeros_like(act), exe)
                if float(a.noise_corr) > 0:
                    eps_n = torch.where(done[:, None], torch.randn_like(eps_n), eps_n)
                o = obs_fn(last_a)
            v_T, hl_T = model.value_and_hit(norm(o))
            bv[T] = v_T * (ret_rms.var.sqrt() + 1e-6); bp[T] = torch.sigmoid(hl_T)
            if relabel:
                pose_s[T * R] = env.sim.state[lid, :4].float()
                rl_pts, rl_vk, rl_msk, rl_d1, rl_d2 = plan_relabel_targets(env, pose_s, d_s, vm_t, ba, R)
            if float(a.geo_shaping) > 0:
                bg[T] = geo_risk(env, lid)
            # the hit head's labels: a hard hit within the next K decisions of the same episode (incomplete windows masked)
            K = int(a.hit_horizon)
            y_hit = torch.zeros(T, B, device=device); v_hit = torch.zeros(T, B, device=device)
            run_h = torch.zeros(B, device=device); run_end = torch.zeros(B, device=device); since = torch.full((B,), 1e9, device=device)
            for t in reversed(range(T)):
                # since[t] = decisions until the next hit (inf if none in this episode's remaining rollout)
                since = torch.where(bh[t] > 0, torch.zeros_like(since), torch.where(bd[t] > 0, torch.full_like(since, 1e9), since + 1))
                y_hit[t] = (since < K).float()
                ended = torch.zeros(B, dtype=torch.bool, device=device)
                complete = torch.full((B,), t + K <= T, dtype=torch.bool, device=device)
                for k in range(t, min(T, t + K)):
                    ended |= bd[k] > 0
                v_hit[t] = (complete | ended | (y_hit[t] > 0)).float()
            # potential-based shaping (Ng et al. 1999, the optimum unchanged): Phi(s) = -price x P(hard hit within K | s)
            shaping = torch.zeros(T, B, device=device)
            if a.hit_shaping > 0 or float(a.geo_shaping) > 0:
                phi = -float(a.hit_shaping) * price * bp - float(a.geo_shaping) * price * bg
                shaping = a.gamma * phi[1:] * (1.0 - bd) - phi[:-1]
            adv = torch.zeros(T, B, device=device); last = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                nd = 1.0 - bd[t]
                delta = br[t] + shaping[t] + a.gamma * bv[t + 1] * nd - bv[t]
                last = delta + a.gamma * a.lam * nd * last
                adv[t] = last
            ret = adv + bv[:T]
            if blv is not None:
                seat = env.race[lid].cpu().numpy()
                lv = blv.cpu().numpy(); aa = adv.abs().float().cpu().numpy()
                seat_tb = np.broadcast_to(seat[None, :], lv.shape)
                okl = lv >= 0
                keys = np.stack([seat_tb[okl], lv[okl]], 1)
                if keys.shape[0]:
                    uk, inv = np.unique(keys, axis=0, return_inverse=True)
                    sums = np.bincount(inv.ravel(), weights=aa[okl]); cnts = np.bincount(inv.ravel())
                    for (st_, sd_), sm_, cn_ in zip(uk, sums, cnts):
                        env.level_replay.accumulate(int(st_), int(sd_), float(sm_), int(cn_))
            ret_rms.update(ret.reshape(-1, 1))
            ret_n = ret / (ret_rms.var.sqrt() + 1e-6)
            if mix is not None and bool((bm > 0).any()):
                am = adv[bm > 0]
                adv_n = (adv - am.mean()) / (am.std() + 1e-8)
            elif int(a.adv_norm_case):
                # `--adv-norm-case` (11-04): each of the four worlds (empty / props / car / both) normalised on its own --
                # one network in the mixed world lost empty-track pace 6.50 -> 6.75 s by u100 even with the solo run's settings
                # (audit B), 6.55 s on the empty track alone (audit A): the price-80 hits of the prop / car worlds set the
                # advantage scale, and an empty-track decision's pace advantage (~0.02 m a decision) came out near 0. Per task,
                # as multi-task RL does for tasks of different reward scales (PopArt, Hessel et al. 2019)
                adv_n = torch.zeros_like(adv)
                for g in range(4):
                    mg = bcase == g
                    if int(mg.sum()) > 1:
                        ag = adv[mg]
                        adv_n[mg] = (ag - ag.mean()) / (ag.std() + 1e-8)
            else:
                adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        t_roll = time.time() - tr0
        N = T * B
        fo, fa, flp, fadv, fret = bo.reshape(N, D), ba.reshape(N, A), blp.reshape(N), adv_n.reshape(N), ret_n.reshape(N)
        fyh, fvh = y_hit.reshape(N), v_hit.reshape(N)
        fm = bm.reshape(N)
        if relabel:
            f_rp, f_rv = rl_pts.reshape(N, -1, 2), rl_vk.reshape(N, -1)
            f_vm, f_rm = vm_t.reshape(N), (rl_msk * bm).reshape(N)
        if anchor is not None:
            f_solo, f_anch = bsolo.reshape(N, A), banch.reshape(N)
        f_scale = bscale.reshape(N)
        f_floor = bfl.reshape(N, A) if bfl is not None else None
        anch_mse = []
        mb = N // int(a.minibatches)
        kls, cfs = [], []
        gdiag = grad_conflict(model, fo, fa, fadv, bcase.reshape(N), fm) if int(a.grad_diag) else None
        for ep in range(int(a.epochs)):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, mb):
                j = perm[i:i + mb]
                fo_j = fo[j].float()
                dist = model.dist(fo_j)
                if calm is not None:
                    dist = torch.distributions.Normal(dist.mean, dist.stddev * f_scale[j][:, None])
                if bfl is not None:
                    dist = torch.distributions.Normal(dist.mean, torch.maximum(dist.stddev, f_floor[j]))
                lp = dist.log_prob(fa[j]).sum(1)
                ratio = (lp - flp[j]).exp()
                m_j = fm[j]
                pg = -(torch.min(ratio * fadv[j], ratio.clamp(1 - a.clip, 1 + a.clip) * fadv[j]) * m_j).sum() / m_j.sum().clamp_min(1.0)
                v_j, hl_j = model.value_and_hit(fo_j)
                vf = 0.5 * (v_j - fret[j]).pow(2).mean()
                loss = pg + a.vf * vf
                if float(a.bound_loss) > 0:
                    # `--bound-loss` (11-01): the network's raw output pulled back into the executable range (see
                    # `--soft-bound`: there the squash keeps the executed mean, this only restores the gradient through it).
                    # Alone and at weight 1 it wrecked the policy (srl57: KL 5.8 on update 1, 4.1 -> 1.7 m/s and 16 wall /
                    # prop hits per km by update 5 -- pulling the far-out outputs moved the in-range ones with them)
                    mu_j = model.raw_mean
                    over = (mu_j - hi_b).clamp_min(0) ** 2 + (lo_b - mu_j).clamp_min(0) ** 2
                    loss = loss + float(a.bound_loss) * over.sum(1).mean()
                if relabel:
                    # `--plan-relabel`: the mean plan pulled towards the plan of what the car then drove (plan_relabel_targets)
                    e_j = relabel_loss(env, dist.mean, f_vm[j], f_rp[j], f_rv[j], w_speed=float(a.plan_relabel_speed))
                    loss = loss + float(a.plan_relabel) * (e_j * f_rm[j]).sum() / f_rm[j].sum().clamp_min(1.0)
                if anchor is not None and (int(a.anchor_updates) <= 0 or u - upd0 <= int(a.anchor_updates)):
                    # `--anchor-updates N` (11-05): the solo target only for the first N updates -- it brought the empty-track pace
                    # back within 50 (srp25owa 7.8 -> 6.55 s, srp26owa 9.4 -> 7.45 s) but, left on, it pulled every decision with
                    # nothing near onto the solo line and the avoiding never came back (the user: "그냥 냅다 때려박으려고 하니까")
                    w_a = f_anch[j]
                    e_a = (dist.mean - f_solo[j]).pow(2).sum(1)
                    loss = loss + float(a.anchor_w) * (e_a * w_a).sum() / w_a.sum().clamp_min(1.0)
                    anch_mse.append(float((e_a * w_a).sum() / w_a.sum().clamp_min(1.0)))
                if a.hit_head > 0:
                    bce = nn.functional.binary_cross_entropy_with_logits(hl_j, fyh[j], reduction="none")
                    loss = loss + float(a.hit_head) * (bce * fvh[j]).sum() / fvh[j].sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if u - upd0 <= int(a.critic_warmup):
                    # `--critic-warmup` (11-05): the first N updates train the critic / hit head only; the actor's grads are
                    # dropped (grad None: Adam leaves those weights and their moments alone). A supervised single-network start
                    # (capacity_test.py: 6.50 s, 8-layout hard 14) carries the gated run's critic, which never valued the empty
                    # track it now drives -- every PPO start from it collapsed (7.65 / 8.25 / 8.85 s by u50, noise x1 / x1 / x0.5)
                    for p_ in list(model.pi.parameters()) + [model.log_std]:
                        p_.grad = None
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    n_m = m_j.sum().clamp_min(1.0)
                    kls.append(float((((ratio - 1) - (lp - flp[j])) * m_j).sum() / n_m)); cfs.append(float((((ratio - 1).abs() > a.clip).float() * m_j).sum() / n_m))
        km = prog / 1000.0
        secs = T * R * B * env.sim.control_dt
        stats = {"update": u, "progress_ms": prog / secs, "hits_wall_prop_per_km": hits_w / max(km, 1e-3),
                 "prop_touch_s_per_km": prop_s / max(km, 1e-3), "prop_push_s_per_km": push_s / max(km, 1e-3),
                 "hits_car_per_km": hits_c / max(km, 1e-3),
                 "passes_per_km": n_gain / max(km, 1e-3), "passed_per_km": n_lost / max(km, 1e-3),
                 "passes_void_per_km": n_void / max(km, 1e-3), "J_per_step": float(br.sum() / (T * R * B)),
                 "laps": len(laps), "lap_mean_s": float(np.mean(laps)) if laps else None,
                 "lap_best_s": float(np.min(laps)) if laps else None, "std": torch.maximum(model.log_std.exp(), model.std_floor).tolist(),
                 "kl": float(np.mean(kls)), "clipfrac": float(np.mean(cfs)), "t_roll": t_roll,
                 "t_upd": time.time() - tr0 - t_roll, "min": (time.time() - t0) / 60, "price": price,
                 "J80_per_step": float((prog - 80.0 * (hits_w + hits_c)) / (T * R * B)),
                 "plan_dev_m": dev_sum / max(dev_n, 1.0),
                 "hit_p_pos": float((bp[:T] * y_hit * v_hit).sum() / (y_hit * v_hit).sum().clamp_min(1.0)),
                 "hit_p_neg": float((bp[:T] * (1 - y_hit) * v_hit).sum() / ((1 - y_hit) * v_hit).sum().clamp_min(1.0)),
                 "shaping_abs": float(shaping.abs().mean()), "geo_risk_mean": float(bg[:T].mean()),
                 "geo_risk_before_hit": float((bg[:T] * y_hit).sum() / y_hit.sum().clamp_min(1.0)),
                 "policy_share": float(bm.mean()),
                 "case_share": [float((bcase == g).float().mean()) for g in range(4)],
                 "grad_diag": gdiag,
                 "anchor_share": float(banch.mean()) if anchor is not None else None,
                 "calm_share": float((bscale != 1.0).float().mean()) if calm is not None else None,
                 "anchor_mse": float(np.mean(anch_mse)) if anch_mse else None,
                 "dead_speed": float(dead_n[1] / dead_n[0].clamp_min(1.0)), "dead_steer": float(dead_n[2] / dead_n[0].clamp_min(1.0))}
        if relabel:
            stats["plan_vs_driven_m"] = float(rl_d1.median()) if rl_d1.numel() else None
            stats["plan_vs_driven_p90_m"] = float(rl_d1.quantile(0.9)) if rl_d1.numel() else None
            stats["plan2L_vs_driven_m"] = float(rl_d2.median()) if rl_d2.numel() else None
            stats["relabel_valid"] = float(rl_msk.mean())
        log.write(json.dumps(stats) + "\n"); log.flush()
        if u % 5 == 0 or u == upd0 + 1:
            lp_ = f"{stats['lap_mean_s']:.2f}" if laps else "-"
            print(f"[{time.strftime('%H:%M')}] 업데이트 {u} | 진행 {stats['progress_ms']:.2f} m/s | 강한 충돌 벽+상자 "
                  f"{stats['hits_wall_prop_per_km']:.2f} 차 {stats['hits_car_per_km']:.2f} /km | 추월 {stats['passes_per_km']:.2f} (부딪혀 무효 {stats['passes_void_per_km']:.2f}) 추월당함 {stats['passed_per_km']:.2f} /km | 상자 닿음 {stats['prop_touch_s_per_km']:.1f} (1.5 m/s 미만 {stats['prop_push_s_per_km']:.1f}) s/km | "
                  f"랩 {len(laps)}개 평균 {lp_} s | "
                  f"J(가격 {price:.0f}) {stats['J_per_step']:.4f} J80 {stats['J80_per_step']:.4f} | 길 벗어남 {stats['plan_dev_m']:.2f} m | 충돌 예측 {stats['hit_p_pos']:.2f}/{stats['hit_p_neg']:.3f} | 경주 정책 몫 {stats['policy_share']:.2f} | 죽은 결정 속도 {stats['dead_speed'] * 100:.0f} % 조향 {stats['dead_steer'] * 100:.0f} % | 표준편차 {[round(x, 3) for x in stats['std']]} | KL {stats['kl']:.4f} | "
                  f"{t_roll:.1f}+{stats['t_upd']:.1f} s"
                  + (f" | 계획-실제 길 어긋남 [0,L] 중앙 {stats['plan_vs_driven_m']:.2f} p90 {stats['plan_vs_driven_p90_m']:.2f} m, "
                     f"그리는 길이 전체 중앙 {stats['plan2L_vs_driven_m']:.2f} m (유효 {stats['relabel_valid'] * 100:.0f} %)"
                     if relabel and stats.get("plan_vs_driven_m") is not None else ""),
                  flush=True)
        if u % int(a.save_every) == 0 or u == upd0 + int(a.updates):
            torch.save({"model": model.state_dict(), "norm": norm.state_dict(), "ret_rms": ret_rms.state_dict(),
                        "obs_dim": D, "args": vars(a), "update": u, "ahead": AHEAD, "bev": bev},
                       os.path.join(out, f"state_u{u}.pt"))
    print(f"[state_rl] {a.name} 끝", flush=True)


@torch.no_grad()
def evaluate(a):
    """The checkpoint's MEAN action (no exploration noise; `--eval-noise 1`: with it) in its stage's world: lap times,
    progress, hard hits per km."""
    device = torch.device(a.device)
    ck = torch.load(a.eval, map_location=device, weights_only=False)
    if plan_of(ck["args"]):
        # a plan checkpoint drives through the follower it was trained with, whatever this command line says
        for k in ("action", "plan_lookahead", "plan_horizon", "repeat"):
            setattr(a, k, ck["args"].get(k, getattr(a, k)))
    a.tracks = a.tracks or str(ck["args"].get("tracks", "") or "")       # the map it was trained on, unless --tracks says
    env = build_env(a, device)
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    tp = TeacherPolicy(env, ck, device)                   # its observation and, with --base, the gated mixture
    act = torch.zeros(B, tp.A, device=device)
    w_col = abs(env.ecfg.reward_collision)
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    touching = torch.zeros(B, dtype=torch.bool, device=device)
    prog = hw = hc = prop_s = push_s = 0.0; laps = []; steps = 0
    R = int(a.repeat)
    eps = torch.zeros(B, tp.A, device=device)
    for t in range(int(a.eval_steps) // R):
        if a.eval_noise:
            # the training's exploration (AR(1) noise at the policy's std) on the race policy: the J that PPO's gradient
            # follows, beside the mean action's J that the evaluations judge (11-01)
            tp.o = tp.sobs(act)
            dist = tp.net.dist(tp.nrm(tp.o))
            al = float(a.noise_corr)
            eps = al * eps + math.sqrt(1.0 - al * al) * torch.randn_like(eps)
            a_pol = clamp_any(dist.mean + dist.stddev * eps, tp.ar)
            act = tp.mix(a_pol, act) if tp.mix is not None else a_pol
        else:
            act = tp(act)
        act_full[lid] = act
        for k in range(R):
            _, r_, te, trc, info = env.step(act_full)
            comp = info["reward_components"]
            col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
            tch = car < 0; onset = tch & ~touching; d_ = (te | trc)[lid]; touching = tch & ~d_
            hw += float((-col / max(w_col, 1e-9)).clamp_min(0).sum()); hc += float(onset.float().sum())
            prog += float(info["progress"][lid].float().clamp_min(0).sum()); steps += 1
            prop_s += float(env.sim.kind_prev[lid, 1].float().sum()) * env.sim.control_dt
            push_s += float((env.sim.kind_prev[lid, 1] & (env.sim.state[lid, 3].abs() < 1.5)).float().sum()) * env.sim.control_dt
            if info["lap_times"].numel():
                laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
    km = prog / 1000.0
    secs = steps * B * env.sim.control_dt
    q = np.percentile(laps, [10, 50, 90]) if laps else [float("nan")] * 3
    print(f"[평가] {os.path.basename(a.eval)} 단계 {a.stage}, {'탐색 잡음 행동' if a.eval_noise else '평균 행동'}, {B}대 x {steps * env.sim.control_dt:.0f} s = {km:.1f} km | "
          f"진행 {prog / secs:.2f} m/s | 강한 충돌 벽+상자 {hw / max(km, 1e-3):.2f}, 차 {hc / max(km, 1e-3):.2f} /km | "
          f"상자 닿음 {prop_s / max(km, 1e-3):.1f} (1.5 m/s 미만 {push_s / max(km, 1e-3):.1f}) s/km | 랩 {len(laps)}개, 중앙값 {q[1]:.2f} s (p10 {q[0]:.2f}, p90 {q[2]:.2f}) | J80/스텝 {(prog - 80 * (hw + hc)) / (steps * B):.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="", help="evaluate this checkpoint's mean action instead of training")
    ap.add_argument("--eval-steps", type=int, default=2400)
    ap.add_argument("--eval-noise", type=int, default=0, help="--eval: 1 = with the training's exploration noise on the race policy")
    ap.add_argument("--name", required=True)
    ap.add_argument("--stage", default="solo", choices=("solo", "props", "race"))
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--warps", type=int, default=16)
    ap.add_argument("--beams", type=int, default=36)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2, help="control steps per decision (40 Hz / X); with --action plan the plan "
                    "is held on the track for these steps (EnvConfig.decision_every)")
    ap.add_argument("--tracks", default="", help="comma-separated base maps (each with its --warps variants); default Korea 2026")
    ap.add_argument("--action", default="direct", choices=("direct", "plan"),
                    help="plan (11-01): a trajectory, 4 waypoints x (curvature, speed), followed by MAP; direct: (steer, speed)")
    ap.add_argument("--plan-lookahead", type=float, default=0.235, help="--action plan: MAP reads the plan's speed this far ahead [s]")
    ap.add_argument("--plan-horizon", type=float, default=0.6, help="--action plan: the plan is this many s of the measured speed long")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--log-std", type=float, default=-0.5)
    ap.add_argument("--hidden", type=int, default=256,
                    help="width of the actor / critic MLP layers; an --init of a narrower net is widened function-preservingly (11-03)")
    ap.add_argument("--log-std-shift", type=float, default=0.0, help="added to the --init checkpoint's log_std (-0.693 = half the noise)")
    ap.add_argument("--std-floor", type=float, nargs=2, default=None, help="floor of the exploration std (steer, speed) in action units")
    ap.add_argument("--price", type=float, default=80.0)
    ap.add_argument("--price-start", type=float, default=0.0)
    ap.add_argument("--price-hold", type=int, default=60, help="updates at --price-start before the ramp")
    ap.add_argument("--price-ramp", type=int, default=200, help="updates of the linear ramp to --price")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--init", default="")
    ap.add_argument("--unwedge", type=int, default=0, help="1: back a wedged car out (the old tracker's recovery, mirrored steering); 2: back-out steering from the front scan (10-30)")
    ap.add_argument("--opp-present", type=float, default=1.0, help="race stage: share of races with the opponent on track")
    ap.add_argument("--opp-slot", default="", help="race stage: JSON merged into the opponent slot (speed_scale, reactive, events, ...)")
    ap.add_argument("--opp-spawn", default="random", help="race stage: where the opponent starts ('ahead': we start behind it)")
    ap.add_argument("--prop-fraction", type=float, default=1.0, help="share of resets that get a layout (the rest: empty track)")
    ap.add_argument("--removal-p", type=float, default=0.0,
                    help="competition sessions: share of episodes whose props are removed mid-episode")
    ap.add_argument("--removal-s", type=float, nargs=2, default=(10.0, 30.0), help="the removal time window [s]")
    ap.add_argument("--car-st", type=int, default=0, help="1: the car channel = the cars where they will be when we get there (10-31)")
    ap.add_argument("--scan-nocars", type=int, default=0, help="1: the scan obs is cast without the other cars (10-31)")
    ap.add_argument("--corridor-car-st", type=int, default=0,
                    help="1: the cars where they will be when we get there, INTO the prop corridor (11-02)")
    ap.add_argument("--blank-opp", type=int, default=0, help="1: the privileged opponent slots as on the empty track (11-02)")
    ap.add_argument("--car-st-line", type=int, default=0,
                    help="1: the car space-time block predicts the cars along the raceline (its pace on it, its offset from it)")
    ap.add_argument("--base", default="", help="the gated mixture (10-31): this frozen SOLO teacher drives while nothing is near")
    ap.add_argument("--residual-solo", default="", help="ResidualPi (11-04): this frozen SOLO policy + a change that is 0 "
                    "while no prop / car is in view (--vis-actor); input statistics frozen")
    ap.add_argument("--residual-init", type=int, default=0, help="--residual-solo: R starts as the --init checkpoint's actor "
                    "(the change = how it reacts to what it sees), its input statistics folded into R's first layer")
    ap.add_argument("--base-reach", type=float, default=BASE_REACH, help="--base: near = a prop on the track / a car within this [m]")
    ap.add_argument("--base-hold", type=int, default=BASE_HOLD, help="--base: decisions the policy keeps the car after")
    ap.add_argument("--corridor-cars", type=int, default=0, help="1: mark the other cars (now, +0.5 s, +1 s) in the corridor")
    ap.add_argument("--car-channel", type=int, default=0, help="1: the other cars in their own corridor block (appended)")
    ap.add_argument("--bev", type=int, default=0, help="1: the car-centred grid of walls / props / cars + a conv (10-30)")
    ap.add_argument("--v-min", type=float, default=0.0, help="[m/s] the slowest command, < 0 lets the policy reverse")
    ap.add_argument("--shared-enc", type=int, default=0, help="1: actor and critic share the --bev grid encoder")
    ap.add_argument("--reverse-ext", type=float, default=0.0, help="speed action down to -1 - X = reverse (EnvConfig.reverse_ext)")
    ap.add_argument("--scan-points", type=int, default=0,
                    help="the actor's scan as three rows (every range / prop points / car points), no prop slots or prop corridor "
                         "(StateObs.scan_points, 11-06); with --beams 1081")
    ap.add_argument("--plan-reverse-ext", type=float, default=0.0,
                    help="--action plan: the first plan speed down to -1 - X = the car backs along the plan's curve; MAP's own "
                         "back-out rules off (EnvConfig.plan_reverse_ext, 11-05)")
    ap.add_argument("--path-plan", type=int, default=0, help="1: drive a planned path (learn.path_plan) given as the track-ahead block")
    ap.add_argument("--track-k", type=float, default=0.5, help="--path-plan: price per m and step of leaving the plan beyond --track-dead")
    ap.add_argument("--track-dead", type=float, default=0.15, help="[m] --path-plan: free lateral room around the plan")
    ap.add_argument("--plan-ref", default="", help="--path-plan: .npy of the line the plan moves (path_plan.record_line)")
    ap.add_argument("--track-where", default="all", choices=("all", "shifted"),
                    help="--path-plan: price leaving the plan everywhere, or only where the plan is off the raceline")
    ap.add_argument("--corridor-obs", type=int, default=0,
                    help="1: append which lateral positions props block at each track-ahead point (13 x 7, appended last)")
    ap.add_argument("--car-slots", type=int, default=0,
                    help="1: append the visible other cars' positions and relative velocities (visible_car_slots, 11-03); "
                         "an --init without them starts these inputs at zero weight")
    ap.add_argument("--prop-line", type=float, default=0.0,
                    help="props stage: share of the layout's pattern slots that become one piece on the racing line "
                         "(ProceduralObstacles.line_share, 11-03: the wall-measured patterns never reach a mid-lane line)")
    ap.add_argument("--prop-bank", default="",
                    help="file or dir of uniform layouts the per-layout line generator judged drivable (tools/layout_bank.py, "
                         "11-03 night); each reset takes one of its own track's and keeps 1 .. all of its pieces")
    ap.add_argument("--prop-place", default="pattern", choices=("pattern", "uniform"),
                    help="where props stand (ProceduralObstacles place, 11-03 night): wall-measured patterns, or uniform over the "
                         "lane area keeping a 0.7 m way past every piece")
    ap.add_argument("--geo-shaping", type=float, default=0.0,
                    help="potential -X x price x geo_risk (the 1 s arc's overlap with a prop); 0 = off")
    ap.add_argument("--car-price", type=float, default=0.0,
                    help="final price of a hard car contact (0 = --price); ramps with the price schedule")
    ap.add_argument("--severity-min", type=float, default=1.5,
                    help="[m/s] the impact a contact needs to be charged (EnvConfig.contact_severity_min). 1.0 = what the console "
                         "and every report call hard (contact_severity.HARD_MPS); 1.5 left 1.0-1.5 m/s hits free (11-02)")
    ap.add_argument("--vis-actor", type=int, default=0,
                    help="the actor reads props only as the car can SEE them (in the student's grid, in line of sight) and "
                         "not the time left; the critic keeps the full truth (11-03)")
    ap.add_argument("--pos-void", type=float, default=0.0,
                    help="--pos-bonus: a place taken within this many seconds of a hard car contact does not pay (11-02)")
    ap.add_argument("--pos-bonus", type=float, default=0.0,
                    help="race stage: + this when taking a place from a car within 8 m (1 m hysteresis), - when losing one (11-02)")
    ap.add_argument("--prop-touch-vmax", type=float, default=0.0,
                    help="--prop-touch-price only while the car is slower than this [m/s] (a push, not a graze); 0 = any speed")
    ap.add_argument("--prop-touch-price", type=float, default=0.0,
                    help="price per second in contact with a prop at any speed (ramps with --price); 0 = off (11-02: soft "
                         "pushes were free, so the props teachers shoved crates off the line)")
    ap.add_argument("--noise-corr", type=float, default=0.0,
                    help="AR(1) correlation of the exploration noise per decision (0 = white; 0.9 = ~0.5 s at 20 Hz)")
    ap.add_argument("--level-replay", type=float, default=0.0, help="prioritized level replay probability (0 = off)")
    ap.add_argument("--level-replay-size", type=int, default=24)
    ap.add_argument("--prop-density", type=float, default=1.5, help="props / stages: procedural_density (training world)")
    ap.add_argument("--prop-count", default="fixed", choices=("fixed", "uniform"),
                    help="patterns per layout: all of the lap's (fixed) or 1 .. all, uniform (11-03: fixed = always dense)")
    ap.add_argument("--prop-max", type=int, default=18, help="props / stages: procedural_max_props")
    ap.add_argument("--stall-price", type=float, default=0.0,
                    help="price per second standing still, linear in the shortfall below --stall-vmin (ramps with --price); 0 = off (11-06)")
    ap.add_argument("--stall-vmin", type=float, default=1.0, help="[m/s] --stall-price charges below this speed magnitude")
    ap.add_argument("--stall-mode", default="speed", choices=("speed", "progress"),
                    help="--stall-price on the speed magnitude now, or on the progress rate over the last --stall-window s (11-06)")
    ap.add_argument("--stall-window", type=float, default=2.0, help="[s] --stall-mode progress: the progress window")
    ap.add_argument("--spawn-behind-prop", type=float, default=0.0,
                    help="share of the learner's resets that start 0.65-1.6 m behind an obstacle of its layout (11-06)")
    ap.add_argument("--prop-shapes", default="all", choices=("all", "boxes"),
                    help="the obstacle pieces: all catalogue pieces, or box-sized ones only (11-06; a bank keeps what it was made with)")
    ap.add_argument("--scan-obs", type=int, default=0, help="1: append the noise-free scan (--beams beams) to the observation")
    ap.add_argument("--soft-bound", type=int, default=0, help="1: the policy mean squashed into the action range near its ends "
                    "(margins steer 0.1, speed 0.2; `soft_bound`) -- exploration reaches inside the range everywhere")
    ap.add_argument("--plan-relabel", type=float, default=0.0,
                    help="weight of the loss pulling the mean plan towards the wp4 plan of the path the car then drove "
                         "(--action plan; plan_relabel_targets)")
    ap.add_argument("--plan-relabel-speed", type=float, default=0.0,
                    help="--plan-relabel's weight on the waypoint speeds [per (m/s)^2]. 0 (11-01): pulled onto the speeds "
                         "the car then had, the plan stops asking for more than it gets and the car stops accelerating "
                         "(srp25r_test: top speed 9.2 -> 6.1 m/s, laps 6.5 -> 7.8 s) -- a planner's speeds are a reference the "
                         "car lags, its path is not")
    ap.add_argument("--bound-loss", type=float, default=0.0, help="weight of the penalty on the policy mean outside the "
                    "executable action range (11-01: clamped-Gaussian dead zone); 0 = off")
    ap.add_argument("--hit-head", type=float, default=0.0, help="weight of the critic's hit-within-K head (BCE); 0 = off")
    ap.add_argument("--hit-horizon", type=int, default=20, help="K decisions (20 = 1 s at 20 Hz)")
    ap.add_argument("--critic-warmup", type=int, default=0, help="the first N updates train the critic only (actor grads dropped, 11-05)")
    ap.add_argument("--corr-slow", type=float, default=0.0,
                    help="hold the --noise-corr noise only below this speed [m/s] (x (V0/v)^2 above); 0 = at every speed")
    ap.add_argument("--plan-format", default="wp4", choices=("wp4", "tw", "tw5", "tw5c", "tw5g"),
                    help="wp4 = 4 curvature + 4 speed knots over 0.6 s of the measured speed (8 numbers); tw = time waypoints "
                         "(plan_tw): TW_K segments of TW_DT s, each a curvature and a signed speed, reverse and forward in one plan")
    ap.add_argument("--slow-explore", type=float, nargs=3, default=None, metavar=("V0", "STEER", "SPEED"),
                    help="exploration std floor for the plan's curvatures / speeds, full below V0 m/s and x (V0/v)^2 above")
    ap.add_argument("--calm-std", type=float, default=1.0, help="exploration std x this where nothing is near (CalmGate: the --base "
                    "test without a solo policy); 1 = off (11-04)")
    ap.add_argument("--prop-movable", type=int, default=1, help="1: props can be shoved (their STYLE_MASS); 0: every prop as fixed as "
                    "a wall (11-05, the user: '장애물이 거의 밀리지 않은 상자로' -- a shove still made progress, so PPO pushed crates "
                    "aside instead of driving round them)")
    ap.add_argument("--anchor-solo", default="", help="ONE network: where nothing is near (the --base test) its mean action is "
                    "pulled onto this frozen solo policy's (11-04)")
    ap.add_argument("--anchor-gate", default="near", choices=["near", "seen"], help="--anchor-solo: anchored where nothing is near "
                    "(SoloMix's 10 m corridor test) or where no prop / car is in the LiDAR's view at all ('seen', 11-05)")
    ap.add_argument("--anchor-updates", type=int, default=0, help="--anchor-solo only for the first N updates; 0 = always (11-05)")
    ap.add_argument("--anchor-w", type=float, default=1.0, help="--anchor-solo: weight of the squared action distance")
    ap.add_argument("--speed-pow", type=float, default=1.0, help="progress paid per metre x (v / --speed-ref)^(p - 1); 1 = plain progress (11-04)")
    ap.add_argument("--speed-ref", type=float, default=5.0, help="[m/s] --speed-pow: the speed at which a metre pays 1")
    ap.add_argument("--grad-diag", type=int, default=0, help="1: log the per-world policy gradients' norms and cosines each update "
                    "(`grad_conflict`, 11-04)")
    ap.add_argument("--adv-norm-case", type=int, default=0, help="1: normalise the advantages within each world (empty / props / "
                    "car / both, `world_case`) instead of over the whole batch (11-04)")
    ap.add_argument("--hit-shaping", type=float, default=0.0,
                    help="potential-based shaping Phi = -X x price x P(hit within K); 0 = off (needs --hit-head)")
    ap.add_argument("--seed", type=int, default=5001)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.eval:
        evaluate(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
