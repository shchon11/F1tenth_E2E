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
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from . import state_rl
from .state_rl import ActorCritic, RunningNorm, StateObs


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
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.price = 80.0
    device = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    # the teacher's world with the car's sensors: a full scan, 3 frames 3 steps apart, the v2 ego history
    # (obs_version 2 is the plan policy's contract; with a direct command the v1 dict carries the same: the scan
    # stack, speed, the issued commands, the proprio history)
    env = state_rl.build_env(a, device, extra=dict(obs_version=1, scan_stack=3, scan_stride=3))
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    obs_fn = StateObs(env)
    obs, _ = env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    ck = torch.load(a.teacher, map_location=device, weights_only=False)
    last_a = torch.zeros(B, 2, device=device)
    o = obs_fn(last_a)
    teacher = ActorCritic(o.shape[1]).to(device); teacher.load_state_dict(ck["model"], strict=False); teacher.eval()
    tnorm = RunningNorm(o.shape[1], device); tnorm.load_state_dict(ck["norm"])
    ego_keys = sorted(k for k in obs if k != "scan")
    egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in ego_keys], 1)
    F, N = obs["scan"].shape[1], obs["scan"].shape[2]; E = egov(obs).shape[1]
    student = Student(F, N, E).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    cap = int(a.buffer)
    b_scan = torch.zeros(cap, F, N, dtype=torch.float16, device=device)
    b_ego = torch.zeros(cap, E, device=device); b_y = torch.zeros(cap, 2, device=device)
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
                ego_now = egov(obs)
                s_act = student(obs["scan"][lid].float(), ego_now)
                idx = torch.arange(ptr, ptr + B, device=device) % cap
                b_scan[idx] = obs["scan"][lid].half(); b_ego[idx] = ego_now; b_y[idx] = y
                ptr = (ptr + B) % cap; n_buf = min(cap, n_buf + B)
                use_t = torch.rand(B, device=device) < beta
                act = torch.where(use_t[:, None], y, s_act)
                agree.append(float((s_act - y).abs().mean()))
                act_full[lid] = act
                for r in range(R):
                    obs, r_, te, trc, info = env.step(act_full)
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
                pred = student(b_scan[j].float(), b_ego[j])
                loss = ((pred - b_y[j]) ** 2 * torch.tensor([2.0, 1.0], device=device)).mean()
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
                        "args": vars(a), "iter": it},
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
    from .. import Config, maps
    from ..gym_env import EnvConfig
    from .. import opponent_slots as opp_sl
    from . import common
    limits = common.teacher_limits(7.0, 6.5, 4.0)
    rows = []

    def run(race_kind, seed, envs):
        trs, rls = common.load_tracks([state_rl.KOREA], racelines=race_kind is not None, **dict(limits, objective="min_time"))
        cfg = Config(); cfg.sim.compile = False; cfg.lidar.n_beams = int(ck["beams"])
        cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0); cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
        e = dict(action_mode="direct", max_steps=int(a.steps), speed_cap=10.0, v_max_policy=10.0, collision_mode="soft",
                 contact_severity_min=1.5, stagger_first_episode=False, reward_collision=-1.0, reward_car_contact=1.0,
                 spawn_runway=3.0, compile_tracker=False, obs_version=1, scan_stack=int(ck["frames"]), scan_stride=3)
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
        st = Student(int(ck["frames"]), int(ck["beams"]), int(ck["ego_dim"])).to(device); st.load_state_dict(ck["student"]); st.eval()
        act_full = torch.zeros(env.B, env.act_dim, device=device)
        touching = torch.zeros(B, dtype=torch.bool, device=device)
        prog = 0.0; hits = {"벽": 0.0, "상자": 0.0, "차": 0.0}; laps = []
        R = int(ck["args"].get("repeat", 2))
        for t in range(int(a.steps) // R):
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
        km = prog / 1000.0; secs = (int(a.steps) // R) * R * B * env.sim.control_dt
        return km, prog / secs, hits, laps

    km, v, h, laps = run(None, 81, 16)
    solo_lap = float(np.median(laps)) if laps else float("nan")
    tot = {"km": 0.0, "prog": 0.0, "벽": 0.0, "상자": 0.0, "차": 0.0}
    for seed in (81, 82):
        for kind in ("forzaeth", "forzaeth_pred", "lane_switch", "interactive"):
            km_r, v_r, h_r, _ = run(kind, seed, int(a.envs))
            tot["km"] += km_r; tot["prog"] += v_r * km_r
            for k in ("벽", "상자", "차"):
                tot[k] += h_r[k]
    vr = tot["prog"] / max(tot["km"], 1e-9)
    hk = {k: tot[k] / max(tot["km"], 1e-9) for k in ("벽", "상자", "차")}
    print(f"[학생 평가] {os.path.basename(a.ckpt)} | 혼자 랩 중앙값 {solo_lap:.2f} s ({len(laps)}개), 혼자 강한 충돌 "
          f"{sum(h.values()) / max(km, 1e-9):.2f}/km | 경주 {tot['km']:.1f} km, 속도 {vr:.2f} m/s, 강한 충돌 {sum(hk.values()):.2f}/km "
          f"(벽 {hk['벽']:.2f} / 상자 {hk['상자']:.2f} / 차 {hk['차']:.2f})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_student()
    else:
        main()
