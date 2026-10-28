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
    ap.add_argument("--frames", type=int, default=3, help="scan frames in the stack")
    ap.add_argument("--stride", type=int, default=3, help="control steps between stacked frames")
    ap.add_argument("--danger-weight", type=float, default=0.0, help="sample weight 1 + X * teacher P(hit within 1 s)")
    ap.add_argument("--buffer-cpu", type=int, default=0, help="keep the scan buffer in pinned host memory")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.price = 80.0
    device = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    # the teacher's world with the car's sensors: a full scan, 3 frames 3 steps apart, the v2 ego history
    # (obs_version 2 is the plan policy's contract; with a direct command the v1 dict carries the same: the scan
    # stack, speed, the issued commands, the proprio history)
    env = state_rl.build_env(a, device, extra=dict(obs_version=1, scan_stack=int(a.frames), scan_stride=int(a.stride)))
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    ck = torch.load(a.teacher, map_location=device, weights_only=False)
    t_scan = bool(ck["args"].get("scan_obs", 0)); t_beams = int(ck["args"].get("beams", 36))
    base_obs = StateObs(env)
    obs, _ = env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    ang_s = env.sim.lidar.angles.float()                                   # the student's beams
    ang_t = torch.linspace(float(ang_s[0]), float(ang_s[-1]), t_beams, device=device)   # the teacher's (same field of view)

    def obs_fn(last):
        """the teacher's observation: the state block, plus its noise-free scan at ITS beams, interpolated from this
        env's full one (the teacher was trained with `t_beams` beams over the same field of view)"""
        o_ = base_obs(last)
        if not t_scan:
            return o_
        sc = env.last_result.scan_true[lid].float()
        pos = (ang_t - ang_s[0]) / (ang_s[-1] - ang_s[0]) * (ang_s.numel() - 1)
        i0 = pos.floor().long().clamp(0, ang_s.numel() - 2); w = (pos - i0.float())[None, :]
        st_ = sc[:, i0] * (1 - w) + sc[:, i0 + 1] * w
        return torch.cat([o_, (st_ / 10.0).clamp(0.0, 1.0)], 1)

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
    # the scan buffer can live in host memory (--buffer-cpu): 400k samples x 4 frames x 1081 beams fp16 = 3.5 GB, which
    # does not fit beside a teacher run on one GPU; minibatches are copied over per step
    bdev = torch.device("cpu") if a.buffer_cpu else device
    b_scan = torch.zeros(cap, F, N, dtype=torch.float16, device=bdev, pin_memory=bool(a.buffer_cpu))
    b_ego = torch.zeros(cap, E, device=device); b_y = torch.zeros(cap, 2, device=device)
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
                s_act = student(obs["scan"][lid].float(), ego_now)
                idx = torch.arange(ptr, ptr + B, device=device) % cap
                b_scan[idx.to(bdev)] = obs["scan"][lid].half().to(bdev); b_ego[idx] = ego_now; b_y[idx] = y; b_w[idx] = w_s
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
                pred = student(b_scan[j.to(bdev)].to(device, non_blocking=True).float(), b_ego[j])
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
            sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)))
            last = torch.zeros(B, 2, device=device)
            o_ = sobs(last)
            tp = ActorCritic(o_.shape[1]).to(device); tp.load_state_dict(ck["model"], strict=False); tp.eval()
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_student()
    else:
        main()
