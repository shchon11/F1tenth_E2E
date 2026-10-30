"""The LiDAR student of a privileged-state RL policy (rl-audit 2026-10-29): the policy the car can run.

The teacher (`state_rl.py`) drives from the simulator's true state and is never deployed. The student sees what the car
sees -- the newest LiDAR frames (3, 225 ms apart: motion shows as change) and the v2 ego history (speed, IMU, the
issued commands) -- and outputs the same command, steer + speed. DAgger: every iteration the cars are driven by the
teacher with probability beta and by the student otherwise (per car, per decision), every learner step is labelled with
the teacher's mean action at that state, and the student regresses on a FIFO of recent labels. Beta falls to 0, so
the student is trained where it drives itself.
usage: python -m f1sim.learn.state_student --name NAME --teacher CKPT [--stage solo] [--envs 1024] [--iters 40]"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from . import state_rl
from .state_rl import ActorCritic, PropRemoval, RunningNorm, StateObs, Unwedge


class Student(nn.Module):
    """scan frames (B, F, N) + ego (B, E) -> (steer, speed) in [-1, 1] (tanh)."""

    def __init__(self, frames, beams, ego_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(frames, 32, 7, stride=3), nn.ReLU(), nn.Conv1d(32, 64, 5, stride=2), nn.ReLU(),
                                  nn.Conv1d(64, 64, 5, stride=2), nn.ReLU(), nn.AdaptiveAvgPool1d(32), nn.Flatten(),
                                  nn.Linear(64 * 32, 256), nn.ReLU())
        self.ego = nn.Sequential(nn.Linear(ego_dim, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 2), nn.Tanh())

    def forward(self, scan, ego):
        return self.head(torch.cat([self.conv(scan), self.ego(ego)], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--stage", default="solo", choices=("solo", "props", "race"))
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--warps", type=int, default=16)
    ap.add_argument("--beams", type=int, default=1081)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--steps", type=int, default=256, help="control steps collected per iteration")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--buffer", type=int, default=400000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--beta-iters", type=int, default=10, help="iterations over which beta falls to 0")
    ap.add_argument("--seed", type=int, default=6001)
    ap.add_argument("--frames", type=int, default=3, help="scan frames in the stack")
    ap.add_argument("--stride", type=int, default=3, help="control steps between stacked frames")
    ap.add_argument("--danger-weight", type=float, default=0.0, help="sample weight 1 + X * teacher P(hit within 1 s)")
    ap.add_argument("--buffer-cpu", type=int, default=0, help="keep the scan buffer in pinned host memory")
    # the competition's session world (state_rl's options, 10-29): opponent on track for a share of races, some resets with
    # an empty track, props removed mid-episode; and the wedge recovery on the learner's car
    ap.add_argument("--opp-present", type=float, default=1.0)
    ap.add_argument("--prop-fraction", type=float, default=1.0)
    ap.add_argument("--removal-p", type=float, default=0.0)
    ap.add_argument("--removal-s", type=float, nargs=2, default=(10.0, 30.0))
    ap.add_argument("--unwedge", type=int, default=0)
    ap.add_argument("--grid", type=int, default=0, help="1: the student reads a car-centred grid built from its LiDAR stack "
                                                       "and odometry (learn.lidar_grid), the teacher's --bev cells (10-30)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.price = 80.0
    device = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    # the teacher's world with the car's sensors: a full scan, 3 frames 3 steps apart, the v2 ego history
    # (obs_version 2 is the plan policy's contract; with a direct command the v1 dict carries the same: the scan
    # stack, speed, the issued commands, the proprio history)
    ck = torch.load(a.teacher, map_location=device, weights_only=False)
    a.v_min = float(ck["args"].get("v_min", 0.0) or 0.0)       # the teacher's command range (it may reverse), saved in args
    env = state_rl.build_env(a, device, extra=dict(obs_version=1, scan_stack=int(a.frames), scan_stride=int(a.stride)))
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    t_scan = bool(ck["args"].get("scan_obs", 0)); t_beams = int(ck["args"].get("beams", 36))
    # the teacher's observation: the state block, its noise-free scan at ITS beams (interpolated from this env's full
    # one over the same field of view), its prop corridor if it was trained with one
    obs_fn = StateObs(env, scan=t_scan, corridor=bool(ck["args"].get("corridor_obs", 0)), corridor_cars=bool(ck["args"].get("corridor_cars", 0)), car_channel=bool(ck["args"].get("car_channel", 0)), bev=bool(ck["args"].get("bev", 0)), path_plan=bool(ck["args"].get("path_plan", 0)), t_beams=t_beams)
    obs, _ = env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    remover = PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=int(a.seed)) \
        if float(a.removal_p) > 0 else None
    unwedge = Unwedge(env, lid, mode=int(a.unwedge)) if int(a.unwedge) else None

    last_a = torch.zeros(B, 2, device=device)
    o = obs_fn(last_a)
    teacher = ActorCritic(o.shape[1], bev=ck.get("bev")).to(device); teacher.load_state_dict(ck["model"], strict=False); teacher.eval()
    tnorm = RunningNorm(o.shape[1], device); tnorm.load_state_dict(ck["norm"])
    ego_keys = sorted(k for k in obs if k != "scan")
    egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in ego_keys], 1)
    F, N = obs["scan"].shape[1], obs["scan"].shape[2]; E = egov(obs).shape[1]
    from .lidar_grid import OdomHist, make_student
    student = make_student(None, env, F, N, E, bool(a.grid), device)
    oh = OdomHist(env, lid, F, int(a.stride)) if a.grid else None
    if oh is not None:
        oh.update()
    stud = (lambda sc, eg, rl: student(sc, eg, rl)) if a.grid else (lambda sc, eg, rl: student(sc, eg))
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    cap = int(a.buffer)
    # the scan buffer can live in host memory (--buffer-cpu): 400k samples x 4 frames x 1081 beams fp16 = 3.5 GB, which
    # does not fit beside a teacher run on one GPU; minibatches are copied over per step
    bdev = torch.device("cpu") if a.buffer_cpu else device
    b_scan = torch.zeros(cap, F, N, dtype=torch.float16, device=bdev, pin_memory=bool(a.buffer_cpu))
    b_ego = torch.zeros(cap, E, device=device); b_y = torch.zeros(cap, 2, device=device)
    b_rel = torch.zeros(cap, F, 3, device=device)                     # the grid student's frame poses (odometry)
    b_w = torch.ones(cap, device=device)
    n_buf = 0; ptr = 0
    out = os.path.expanduser(f"~/f1sim_runs/{a.name}"); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    log = open(os.path.join(out, "progress.jsonl"), "a")
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    w_col = abs(env.ecfg.reward_collision)
    R = int(a.repeat)
    print(f"[state_student] {a.name}: 선생 {os.path.basename(a.teacher)}, 단계 {a.stage}, 환경 {env.B} (학습 행 {B}), "
          f"라이다 {F}장 x {N}빔, 자아 {E}차원", flush=True)
    t0 = time.time()
    for it in range(1, int(a.iters) + 1):
        beta = max(0.0, float(a.beta0) * (1.0 - (it - 1) / max(1, int(a.beta_iters))))
        prog = hw = hc = 0.0; laps = []; agree = []
        touching = torch.zeros(B, dtype=torch.bool, device=device)
        student.eval()
        with torch.no_grad():
            for t in range(int(a.steps) // R):
                y = teacher.pi(tnorm(o)).clamp(-1, 1)                      # the label: the teacher's mean action
                # danger weight (--danger-weight): the student's errors cost hits where the teacher's own hit head sees
                # one coming (10-29: before its car contacts the student was 0.68 m/s faster than the label in 65 % of
                # cases); uniform MSE spends the fit on cruising
                w_s = 1.0 + float(a.danger_weight) * torch.sigmoid(teacher.value_and_hit(tnorm(o))[1]) if float(a.danger_weight) > 0 \
                    else torch.ones(B, device=device)
                ego_now = egov(obs)
                rel_now = oh.rel() if oh is not None else torch.zeros(B, F, 3, device=device)
                s_act = stud(obs["scan"][lid].float(), ego_now, rel_now)
                idx = torch.arange(ptr, ptr + B, device=device) % cap
                b_scan[idx.to(bdev)] = obs["scan"][lid].half().to(bdev); b_ego[idx] = ego_now; b_y[idx] = y; b_w[idx] = w_s
                b_rel[idx] = rel_now
                ptr = (ptr + B) % cap; n_buf = min(cap, n_buf + B)
                use_t = torch.rand(B, device=device) < beta
                act = torch.where(use_t[:, None], y, s_act)
                agree.append(float((s_act - y).abs().mean()))
                act_full[lid] = act
                for r in range(R):
                    if unwedge is not None:
                        unwedge.before(act_full)
                    obs, r_, te, trc, info = env.step(act_full)
                    if oh is not None:
                        oh.update()
                    if unwedge is not None:
                        unwedge.after()
                    if remover is not None:
                        remover.step()
                    comp = info["reward_components"]
                    col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
                    tch = car < 0; onset = tch & ~touching; d_ = (te | trc)[lid]; touching = tch & ~d_
                    hw += float((-col / max(w_col, 1e-9)).clamp_min(0).sum()); hc += float(onset.float().sum())
                    prog += float(info["progress"][lid].float().clamp_min(0).sum())
                    if info["lap_times"].numel():
                        laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                last_a = act
                o = obs_fn(last_a)
        student.train()
        losses = []
        for ep in range(int(a.epochs)):
            perm = torch.randperm(n_buf, device=device)
            for i in range(0, n_buf, int(a.batch)):
                j = perm[i:i + int(a.batch)]
                pred = stud(b_scan[j.to(bdev)].to(device, non_blocking=True).float(), b_ego[j], b_rel[j])
                per = ((pred - b_y[j]) ** 2 * torch.tensor([2.0, 1.0], device=device)).mean(1)
                loss = (per * b_w[j]).sum() / b_w[j].sum()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                losses.append(float(loss))
        km = prog / 1000.0
        secs = int(a.steps) * B * env.sim.control_dt
        st = {"iter": it, "beta": beta, "progress_ms": prog / secs, "hits_wall_prop_per_km": hw / max(km, 1e-3),
              "hits_car_per_km": hc / max(km, 1e-3), "laps": len(laps), "lap_median_s": float(np.median(laps)) if laps else None,
              "student_vs_teacher_abs": float(np.mean(agree)), "loss": float(np.mean(losses)), "buffer": n_buf,
              "min": (time.time() - t0) / 60}
        log.write(json.dumps(st) + "\n"); log.flush()
        lm = f"{st['lap_median_s']:.2f}" if laps else "-"
        print(f"[{time.strftime('%H:%M')}] 반복 {it} | 선생 비율 {beta:.2f} | 진행 {st['progress_ms']:.2f} m/s | 강한 충돌 벽+상자 "
              f"{st['hits_wall_prop_per_km']:.2f} 차 {st['hits_car_per_km']:.2f} /km | 랩 {len(laps)}개 중앙값 {lm} s | "
              f"학생-선생 차이 {st['student_vs_teacher_abs']:.3f} | 손실 {st['loss']:.4f}", flush=True)
        if it % 5 == 0 or it == int(a.iters):
            torch.save({"student": student.state_dict(), "frames": F, "beams": N, "ego_dim": E, "ego_keys": ego_keys,
                        "args": vars(a), "iter": it, "grid": bool(a.grid)},
                       os.path.join(out, f"student_i{it}.pt"))
    print(f"[state_student] {a.name} 끝", flush=True)


@torch.no_grad()
def evaluate_student():
    """A student on the line-eval conditions (eval_ckpt.sh): Korea solo (no props) laps; 1 vs 1 races against each planner
    kind at 0.7-1.0 of its pace, props on the racing line (density 1.5, at most 10, movable), seeds 81/82. The student
    drives alone (its own command); reported like summarize.py: lap, race speed, hard hits per km by kind."""
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(sys.argv[2:])
    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    # a state_rl teacher checkpoint ("model") drives on the same worlds with its own privileged observation: the
    # teacher's ceiling next to its student's number separates "the teacher can't" from "the student didn't learn it"
    is_teacher = "model" in ck
    n_beams = int(ck["args"].get("beams", 36)) if is_teacher else int(ck["beams"])
    frames = 1 if is_teacher else int(ck["frames"])
    from .. import Config, maps
    from ..gym_env import EnvConfig
    from .. import opponent_slots as opp_sl
    from . import common
    limits = common.teacher_limits(7.0, 6.5, 4.0)
    rows = []

    def run(race_kind, seed, envs):
        trs, rls = common.load_tracks([state_rl.KOREA], racelines=race_kind is not None, **dict(limits, objective="min_time"))
        cfg = Config(); cfg.sim.compile = False; cfg.lidar.n_beams = n_beams
        cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0); cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
        e = dict(action_mode="direct", max_steps=int(a.steps), speed_cap=10.0, v_max_policy=10.0, collision_mode="soft",
                 contact_severity_min=1.5, stagger_first_episode=False, reward_collision=-1.0, reward_car_contact=1.0,
                 spawn_runway=3.0, compile_tracker=False, obs_version=1, scan_stack=frames,
                 scan_stride=int(ck["args"].get("stride", 3)) if not is_teacher else 3)
        if is_teacher:
            e.update(priv_props=True, priv_time=True)
        if race_kind is not None:
            e.update(procedural_obstacles=1.0, procedural_density=1.5, procedural_max_props=10,
                     procedural_raceline_corridor="off", movable_obstacles=True, race_size=2, opponent="slots",
                     opponent_slots=opp_sl.parse_slots(json.dumps([{"kind": race_kind, "speed_scale": [0.7, 1.0]}])))
        env = common.make_env(trs, envs, device, EnvConfig(**e), cfg=cfg, seed=seed, rls=rls, teacher_limits=limits)
        from .graph_runtime import prepare_graph_runtime
        prepare_graph_runtime(env, log=lambda _t: None)
        obs, _ = env.reset(seed=seed)
        lid = env.learner_ids; B = int(lid.numel())
        keys = ck.get("ego_keys") or sorted(k for k in obs if k != "scan")
        egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in keys], 1)
        if is_teacher:
            sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)), corridor=bool(ck["args"].get("corridor_obs", 0)), corridor_cars=bool(ck["args"].get("corridor_cars", 0)), car_channel=bool(ck["args"].get("car_channel", 0)), bev=bool(ck["args"].get("bev", 0)), path_plan=bool(ck["args"].get("path_plan", 0)))
            last = torch.zeros(B, 2, device=device)
            o_ = sobs(last)
            tp = ActorCritic(o_.shape[1], bev=ck.get("bev")).to(device); tp.load_state_dict(ck["model"], strict=False); tp.eval()
            tn = RunningNorm(o_.shape[1], device); tn.load_state_dict(ck["norm"])
        else:
            st = Student(int(ck["frames"]), int(ck["beams"]), int(ck["ego_dim"])).to(device); st.load_state_dict(ck["student"]); st.eval()
        act_full = torch.zeros(env.B, env.act_dim, device=device)
        touching = torch.zeros(B, dtype=torch.bool, device=device)
        prog = 0.0; hits = {"벽": 0.0, "상자": 0.0, "차": 0.0}; laps = []
        R = int(ck["args"].get("repeat", 2))
        # passing (1 vs 1): the along-track gap to the opponent, opponent ahead > 0; a pass = the gap crossing from
        # (0, 4] to [-4, 0) with neither car reset; following = the gap in (0, 3]
        pas = {"pass": 0, "passed": 0, "follow": 0, "n": 0}
        opp_row = None
        if env.M == 2:
            opp_row = env.race[lid] * 2 + (1 - env.slot[lid])
        trk = env.sim.track
        gap_prev = None
        for t in range(int(a.steps) // R):
            if is_teacher:
                last = tp.pi(tn(sobs(last))).clamp(-1, 1)
                act_full[lid] = last
            else:
                act_full[lid] = st(obs["scan"][lid].float(), egov(obs))
            for r in range(R):
                obs, r_, te, trc, info = env.step(act_full)
                res = env.last_result
                on, imp = res.contact_onsets[lid], res.contact_impacts[lid]
                hits["벽"] += float((on[:, 0] & (imp[:, 0] >= 1.5)).sum()); hits["상자"] += float((on[:, 1] & (imp[:, 1] >= 1.5)).sum())
                car = info["reward_components"]["car_contact"][lid].float(); tch = car < 0; onset = tch & ~touching
                touching = tch & ~(te | trc)[lid]; hits["차"] += float(onset.sum())
                prog += float(info["progress"][lid].float().clamp_min(0).sum())
                if info["lap_times"].numel():
                    laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                if opp_row is not None:
                    L_ = trk.length[env.sim.tid[lid]]
                    g = torch.remainder(res.s[opp_row] - res.s[lid] + L_ / 2, L_) - L_ / 2
                    ok = ~((te | trc)[lid] | (te | trc)[opp_row])
                    if gap_prev is not None:
                        okp = ok & (gap_prev.abs() < 4) & (g.abs() < 4)
                        pas["pass"] += int((okp & (gap_prev > 0) & (g < 0)).sum())
                        pas["passed"] += int((okp & (gap_prev < 0) & (g > 0)).sum())
                    pas["follow"] += int((ok & (g > 0) & (g <= 3.0)).sum()); pas["n"] += int(ok.sum())
                    gap_prev = torch.where(ok, g, torch.full_like(g, 99.0))
        km = prog / 1000.0; secs = (int(a.steps) // R) * R * B * env.sim.control_dt
        return km, prog / secs, hits, laps, pas

    km, v, h, laps, _ = run(None, 81, 16)
    solo_lap = float(np.median(laps)) if laps else float("nan")
    tot = {"km": 0.0, "prog": 0.0, "벽": 0.0, "상자": 0.0, "차": 0.0, "pass": 0, "passed": 0, "follow": 0, "n": 0}
    for seed in (81, 82):
        for kind in ("forzaeth", "forzaeth_pred", "lane_switch", "interactive"):
            km_r, v_r, h_r, _, p_r = run(kind, seed, int(a.envs))
            tot["km"] += km_r; tot["prog"] += v_r * km_r
            for k in ("벽", "상자", "차"):
                tot[k] += h_r[k]
            for k in ("pass", "passed", "follow", "n"):
                tot[k] += p_r[k]
            print(f"   {kind} 시드 {seed}: {km_r:.1f} km, {v_r:.2f} m/s, 벽 {h_r['벽'] / max(km_r, 1e-9):.2f} 상자 "
                  f"{h_r['상자'] / max(km_r, 1e-9):.2f} 차 {h_r['차'] / max(km_r, 1e-9):.2f} /km, 추월 {p_r['pass'] / max(km_r, 1e-9):.2f}/km, "
                  f"추월당함 {p_r['passed'] / max(km_r, 1e-9):.2f}/km, 3 m 안 뒤따름 {p_r['follow'] / max(p_r['n'], 1) * 100:.0f} %", flush=True)
    vr = tot["prog"] / max(tot["km"], 1e-9)
    hk = {k: tot[k] / max(tot["km"], 1e-9) for k in ("벽", "상자", "차")}
    who = "선생 평가" if is_teacher else "학생 평가"
    print(f"[{who}] {os.path.basename(a.ckpt)} | 혼자 랩 중앙값 {solo_lap:.2f} s ({len(laps)}개), 혼자 강한 충돌 "
          f"{sum(h.values()) / max(km, 1e-9):.2f}/km | 경주 {tot['km']:.1f} km, 속도 {vr:.2f} m/s, 강한 충돌 {sum(hk.values()):.2f}/km "
          f"(벽 {hk['벽']:.2f} / 상자 {hk['상자']:.2f} / 차 {hk['차']:.2f}) | 추월 {tot['pass'] / max(tot['km'], 1e-9):.2f}/km, "
          f"추월당함 {tot['passed'] / max(tot['km'], 1e-9):.2f}/km, 3 m 안 뒤따름 {tot['follow'] / max(tot['n'], 1) * 100:.0f} %", flush=True)


def make_env_like(ref, device, n_beams, frames, stride, is_teacher, envs=None, override=None):
    """The env of one old eval_ckpt.sh run (its JSON's `metadata.env_config`), with only the action / observation contract
    of a direct-command policy changed: (env, cfg, ecfg, metadata)."""
    import dataclasses
    from .. import Config
    from ..gym_env import EnvConfig
    from .. import opponent_slots as opp_sl
    from . import common
    fields = {f.name for f in dataclasses.fields(EnvConfig)}
    md = json.load(open(ref))["nominal"]["metadata"]
    ec = {k: v for k, v in md["env_config"].items() if k in fields}
    if ec.get("opponent_slots"):
        ec["opponent_slots"] = opp_sl.parse_slots(json.dumps(ec["opponent_slots"]))
    for k in list(ec):
        if isinstance(ec[k], list) and k != "opponent_slots":
            ec[k] = tuple(ec[k])
    # this policy's contract: a direct command at 20 Hz from the v1 observation (students) or the privileged state
    # (teachers); the plan / MAP / wp4 fields of the reference only shape a plan policy's output and are reset
    base = EnvConfig()
    for k in [k for k in ec if k.startswith("plan_") or k in ("speed_mode", "hist_len", "hist_stride", "ego_actions",
                                                               "opp_token", "dial", "speed_command")]:
        ec[k] = getattr(base, k)
    ec.update(action_mode="direct", obs_version=1, scan_stack=frames, scan_stride=stride, v_max_policy=10.0)
    if is_teacher:
        ec.update(priv_props=True, priv_time=True)
    if override:
        ec.update(override)
    ecfg = EnvConfig(**ec)
    limits = md.get("teacher_limits") or common.teacher_limits(7.0, 6.5, 4.0)
    race = int(ecfg.race_size) > 1
    trs, rls = common.load_tracks(list(md["tracks"]), racelines=race,
                                  **dict(limits, objective=md.get("raceline_objective", "min_time")))
    cfg = Config(); cfg.sim.compile = False; cfg.lidar.n_beams = n_beams
    cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0); cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
    env = common.make_env(trs, int(envs or md["envs"]), device, ecfg, cfg=cfg, seed=int(md["seed"]), rls=rls,
                          teacher_limits=limits)
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    env.sim.warmup()
    return env, cfg, ecfg, md


@torch.no_grad()
def follow_cap(env, lid, decel, gap0=0.25, lane=0.55, reach=10.0):
    """(B,) the normalized speed command cap of the `--follow-rule` ORACLE (10-29 diagnostic: would braking in time have
    avoided the car contacts?): for the nearest other car ahead of us in our lane (body frame: 0 < dx < reach, |dy| <
    lane), the speed from which we can still slow to its speed along our heading at `decel` before the bumper gap
    (dx - one car length) closes to gap0; +1 (no cap) without such a car. Reads the TRUE states -- an oracle, not a policy."""
    if env.sim.other_idx is None:                                          # a race of one
        return torch.ones(lid.numel(), device=lid.device)
    st = env.last_result.state
    me = st[lid]
    o = env.sim.other_idx[lid]                                             # (B, M-1)
    so = st[o]
    c, s_ = torch.cos(me[:, 2])[:, None], torch.sin(me[:, 2])[:, None]
    ddx, ddy = so[..., 0] - me[:, None, 0], so[..., 1] - me[:, None, 1]
    dx, dy = ddx * c + ddy * s_, -ddx * s_ + ddy * c
    co, so_ = torch.cos(so[..., 2]), torch.sin(so[..., 2])
    vxw = so[..., 3] * co - so[..., 4] * so_; vyw = so[..., 3] * so_ + so[..., 4] * co
    v_along = (vxw * c + vyw * s_).clamp_min(0.0)                          # its speed along our heading
    pres = (~env.sim.absent[o]) if env.sim.absent_any else torch.ones_like(dx, dtype=torch.bool)
    ahead = pres & (dx > 0) & (dx < reach) & (dy.abs() < lane)
    gap = (dx - float(env.cfg.vehicle.length) - gap0).clamp_min(0.0)
    v_lim = v_along + torch.sqrt(2.0 * decel * gap)
    v_lim = torch.where(ahead, v_lim, torch.full_like(v_lim, 1e9)).amin(1)
    return (v_lim / float(env.ecfg.v_max_policy) * 2 - 1).clamp(-1.0, 1.0)


def evaluate_like():
    """One policy (a state_student or a state_rl checkpoint) on EXACTLY the worlds and metric code of an old eval_ckpt.sh
    directory (2026-10-29, the user: "이전 데이터들 너 믿지 말고 너가 직접 확인해"): every race's EnvConfig is read back from the
    reference's JSON (`metadata.env_config`; only the action / observation contract this policy needs is changed), and the
    metrics are `learn.evaluate`'s own -- `common.rollout_metrics` inside `TrafficMeter`, `EncounterMeter`,
    `ContactSeverityMeter`, `WrongWayMeter` -- written as race_s<seed>_<kind>.json so `summarize.py` reads both alike.
    The solo run (race of one, no props) is written as solo_x.json (summarize.py's solo column needs teacher_speed_chain's
    bins and stays empty)."""
    import dataclasses
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--like", required=True, help="an eval_ckpt.sh output directory (its race_s*_*.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="", help="'solo' = the solo run only")
    ap.add_argument("--unwedge", type=int, default=1,
                    help="the wedge recovery rlc6's plan tracker ran (state_rl.Unwedge); 0 = as before 10-29 13:20")
    ap.add_argument("--follow-rule", type=float, default=0.0,
                    help="ORACLE diagnostic (true opponent state, never a policy): cap the speed command to the speed that "
                         "can still slow to a car ahead in our lane at this deceleration [m/s^2]; 0 = off")
    a = ap.parse_args(sys.argv[2:])
    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    is_teacher = "model" in ck
    n_beams = int(ck["args"].get("beams", 36)) if is_teacher else int(ck["beams"])
    frames = 1 if is_teacher else int(ck["frames"])
    stride = 1 if is_teacher else int(ck["args"].get("stride", 3))
    R = int(ck["args"].get("repeat", 2))
    from .. import Config
    from ..gym_env import EnvConfig
    from .. import opponent_slots as opp_sl
    from . import common
    from .benchmark.overtake import TrafficMeter
    from .encounter import EncounterMeter
    from .severity_meter import ContactSeverityMeter
    from .wrong_way_meter import WrongWayMeter
    os.makedirs(a.out, exist_ok=True)
    fields = {f.name for f in dataclasses.fields(EnvConfig)}
    refs = sorted(glob.glob(os.path.join(a.like, "race_s*_*.json")))
    solo_ref = os.path.join(a.like, "solo.json")
    jobs = ([("solo", solo_ref)] if os.path.exists(solo_ref) else []) + [(os.path.basename(f)[:-5], f) for f in refs]
    if a.only == "solo":
        jobs = jobs[:1]
    for tag, ref in jobs:
        v_min = float(ck["args"].get("v_min", 0.0) or 0.0)       # a policy that may reverse (state_rl --v-min)
        env, cfg, ecfg, md = make_env_like(ref, device, n_beams, frames, stride, is_teacher,
                                           override=(dict(v_min_policy=v_min) if v_min else None))
        seed = int(md["seed"]); envs = int(md["envs"])
        lid = env.learner_ids; B = int(lid.numel())

        class Pol:
            """decides every R steps (the 20 Hz the policies were trained at), holds in between"""
            def __init__(self):
                self.t = 0; self.act = torch.zeros(env.B, env.act_dim, device=device); self.last = torch.zeros(B, 2, device=device)
                self.bound = 0; self.n_rule = 0
                self.stuck = 0; self.n_live = 0                    # learner steps below 0.3 m/s from 3 s into an episode
                self.uw = Unwedge(env, lid, mode=int(a.unwedge)) if int(a.unwedge) else None
                if is_teacher:
                    self.sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)), corridor=bool(ck["args"].get("corridor_obs", 0)), corridor_cars=bool(ck["args"].get("corridor_cars", 0)), car_channel=bool(ck["args"].get("car_channel", 0)), bev=bool(ck["args"].get("bev", 0)), path_plan=bool(ck["args"].get("path_plan", 0)))
                    self.net = None
                else:
                    from .lidar_grid import StudentRunner
                    self.st = StudentRunner(env, lid, ck, device)
                    self.keys = ck.get("ego_keys")

            def __call__(self, obs):
                if not is_teacher:
                    self.st.observe()                     # the pose the last step left (a grid student's odometry)
                if self.uw is not None and self.t > 0:
                    self.uw.after()                       # the clocks advance on the step just taken
                if self.t > 0:
                    live = env.ep_step[lid] > int(3.0 / float(env.sim.control_dt))
                    self.stuck += int((live & (env.sim.state[lid, 3].abs() < 0.3)).sum()); self.n_live += int(live.sum())
                if self.t % R == 0:
                    if is_teacher:
                        o_ = self.sobs(self.last)
                        if self.net is None:
                            self.net = ActorCritic(o_.shape[1], bev=ck.get("bev")).to(device); self.net.load_state_dict(ck["model"], strict=False)
                            self.net.eval(); self.nrm = RunningNorm(o_.shape[1], device); self.nrm.load_state_dict(ck["norm"])
                        a_ = self.net.pi(self.nrm(o_)).clamp(-1, 1)
                    else:
                        keys = self.keys or sorted(k for k in obs if k != "scan")
                        ego = torch.cat([obs[k][lid].float().reshape(B, -1) for k in keys], 1)
                        a_ = self.st(obs, ego)
                    self.act[lid] = a_; self.last = a_
                out = self.act
                if float(a.follow_rule) > 0:
                    out = self.act.clone(); out[lid, 1] = torch.minimum(out[lid, 1], follow_cap(env, lid, float(a.follow_rule)))
                    self.bound += int((out[lid, 1] < self.act[lid, 1] - 1e-6).sum()); self.n_rule += B
                if self.uw is not None:
                    self.uw.before(out)
                self.t += 1
                return out

            def reset(self, done=None):
                if done is None:
                    self.t = 0; self.last.zero_()
                else:
                    self.last[done[lid]] = 0.0

        pol = Pol()
        meter = TrafficMeter(env, contention_range_m=12.0, attack_range_m=3.0, vehicle_length=cfg.vehicle.length)
        enc = EncounterMeter(env); sev = ContactSeverityMeter(env); ww = WrongWayMeter(env)
        with meter, enc, sev, ww:
            result = common.rollout_metrics(env, pol, int(a.steps), float(ecfg.speed_cap))
        for k_, v_ in list(result.items()):
            if isinstance(v_, float) and not math.isfinite(v_):
                result[k_] = None
        result.update(meter.report()); result.update(enc.report()); result.update(sev.report()); result.update(ww.report())
        km_ = float((result.get("contact_severity") or {}).get("learner_distance_m", 0)) / 1000
        result["stuck"] = {"stuck_frac": pol.stuck / max(pol.n_live, 1),
                           "unwedge_per_km": (pol.uw.count / max(km_, 1e-6)) if pol.uw is not None else None}
        print(f"[같은 평가] {tag}: 멈춰 있는 시간 {pol.stuck / max(pol.n_live, 1) * 100:.1f} %, 끼임 복구 "
              f"{(pol.uw.count / max(km_, 1e-6)) if pol.uw is not None else float('nan'):.2f}/km", flush=True)
        if float(a.follow_rule) > 0:
            result["follow_rule"] = {"decel": float(a.follow_rule), "bound_frac": pol.bound / max(pol.n_rule, 1)}
            print(f"[같은 평가] {tag}: 따라가기 규칙(참 상태, 진단) 감속 {a.follow_rule:g} m/s^2, 묶인 비율 "
                  f"{pol.bound / max(pol.n_rule, 1) * 100:.1f} %", flush=True)
        result["metadata"] = {"checkpoint": a.ckpt, "like": ref, "env_config_overrides": "action/obs contract only",
                              "seed": seed, "envs": envs, "steps": int(a.steps), "env_config": dataclasses.asdict(ecfg)}
        name = "solo_x.json" if tag == "solo" else tag + ".json"
        json.dump({"nominal": result}, open(os.path.join(a.out, name), "w"), indent=1, default=str)
        cs = result.get("contact_severity") or {}
        km = float(cs.get("learner_distance_m", 0)) / 1000
        hard = {k: sum(((cs.get(k) or {}).get("fine_hist") or [])[15:]) for k in ("wall", "prop", "car")}
        print(f"[같은 평가] {tag}: 진행 {result.get('progress_rate_mps', float('nan')):.2f} m/s, 평균 속력 {result.get('mean_speed', float('nan')):.2f}, "
              f"랩 {result.get('lap_time_s')}, {km:.1f} km, 강한/km 벽 {hard['wall'] / max(km, 1e-6):.2f} 상자 {hard['prop'] / max(km, 1e-6):.2f} "
              f"차 {hard['car'] / max(km, 1e-6):.2f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_student()
    elif len(sys.argv) > 1 and sys.argv[1] == "evalx":
        evaluate_like()
    else:
        main()
