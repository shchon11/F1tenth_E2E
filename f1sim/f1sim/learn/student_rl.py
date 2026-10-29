"""학생 강화학습: DAgger로 시작한 라이다 학생을 자기 관측 위에서 PPO로 (2026-10-29).

왜 (직접 잰 것만): 같은 세계·같은 집계 코드(`state_student evalx --like rlc6`)에서 DAgger 학생 `sst4`는 차 충돌이 `rlc6`의 8배 넘게
많음. 선생은 상대의 정확한 상대 속도·TTC로 바짝 붙었다가 제때 감속하지만, 학생의 라이다로는 상대 속도가 보이지 않음
(`tools/motion_probe.py`: 라이다 4장 = 1장, 상자가 있으면 상대 옆 위치 R2 0.72 -> 0.47) -> 학생은 볼 수 없는 것에 기대는 선생의
행동을 따라 할 수 없음(정보 비대칭). 학생 자신의 관측 위에서 강화학습하면 자기가 볼 수 있는 것으로 피하는 운전을 배움.

구성 (비대칭 행위자-비평가, Pinto et al. 2018):
  행위자  `state_student.Student` (DAgger 체크포인트에서): 라이다 F장 + 자아 -> 조향·속도 평균, 탐색은 가우시안(--noise-corr AR 상관)
  비평가  참 상태 선생의 `state_rl.ActorCritic` (가치 + 충돌 예측 머리, 선생 체크포인트에서; 선생 관측 = StateObs + 선생 빔 수로
          보간한 잡음 없는 라이다) — 배포되지 않으므로 무엇을 봐도 됨
  보상    state_rl과 같은 식, 가격 고정(벽·상자 --price, 차 --car-price), 충돌 예측 머리의 잠재 조형(Ng et al. 1999)
처음 --critic-warmup 회는 비평가만 학습(선생의 가치는 선생의 정책 것이라 학생 정책에 맞출 시간).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import state_rl
from .state_rl import ActorCritic, RunningNorm, StateObs
from .state_student import Student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--student", required=True, help="state_student checkpoint (the actor)")
    ap.add_argument("--teacher", required=True, help="state_rl checkpoint (the critic and its hit head)")
    ap.add_argument("--stage", default="race", choices=("solo", "props", "race"))
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--warps", type=int, default=16)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=64, help="decisions per rollout")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="actor")
    ap.add_argument("--lr-critic", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--log-std", type=float, default=math.log(0.06))
    ap.add_argument("--noise-corr", type=float, default=0.9)
    ap.add_argument("--price", type=float, default=80.0)
    ap.add_argument("--car-price", type=float, default=200.0)
    ap.add_argument("--hit-head", type=float, default=1.0)
    ap.add_argument("--hit-shaping", type=float, default=1.0)
    ap.add_argument("--hit-horizon", type=int, default=20)
    ap.add_argument("--critic-warmup", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    sck = torch.load(a.student, map_location=device, weights_only=False)
    tck = torch.load(a.teacher, map_location=device, weights_only=False)
    F_, N_ = int(sck["frames"]), int(sck["beams"])
    stride = int(sck["args"].get("stride", 3))
    a.beams = N_
    env = state_rl.build_env(a, device, extra=dict(obs_version=1, scan_stack=F_, scan_stride=stride))
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    obs, _ = env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    # the critic's observation: the teacher's, its scan interpolated from this env's full one (as state_student labels)
    t_scan = bool(tck["args"].get("scan_obs", 0)); t_beams = int(tck["args"].get("beams", 36))
    tobs = StateObs(env, scan=t_scan, corridor=bool(tck["args"].get("corridor_obs", 0)), t_beams=t_beams)

    last_a = torch.zeros(B, 2, device=device)
    to = tobs(last_a); Dt = to.shape[1]
    critic = ActorCritic(Dt).to(device); critic.load_state_dict(tck["model"], strict=False)
    tnorm = RunningNorm(Dt, device); tnorm.load_state_dict(tck["norm"])
    ret_rms = RunningNorm(1, device); ret_rms.load_state_dict(tck["ret_rms"])
    keys = sck.get("ego_keys") or sorted(k for k in obs if k != "scan")
    egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in keys], 1)
    E_ = egov(obs).shape[1]
    actor = Student(F_, N_, int(sck["ego_dim"])).to(device); actor.load_state_dict(sck["student"])
    log_std = nn.Parameter(torch.full((2,), float(a.log_std), device=device))
    crit_params = [p for n, p in critic.named_parameters() if not n.startswith("pi.") and n != "log_std"]
    opt_a = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=a.lr, eps=1e-5)
    opt_c = torch.optim.Adam(crit_params, lr=a.lr_critic, eps=1e-5)
    out = os.path.expanduser(f"~/f1sim_runs/{a.name}"); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    log = open(os.path.join(out, "progress.jsonl"), "a")
    T, R = int(a.horizon), int(a.repeat)
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    touching = torch.zeros(B, dtype=torch.bool, device=device)
    eps_n = torch.randn(B, 2, device=device)
    w_col = abs(env.ecfg.reward_collision)
    print(f"[student_rl] {a.name}: 학생 {os.path.basename(a.student)} (라이다 {F_}장 x {N_}빔, 간격 {stride}), 비평가 "
          f"{os.path.basename(a.teacher)} ({Dt}차원), 환경 {env.B} (학습 행 {B}), 가격 벽·상자 {a.price:g} 차 {a.car_price:g}, "
          f"탐색 AR {a.noise_corr}", flush=True)
    t0 = time.time()
    for u in range(1, int(a.updates) + 1):
        b_scan = torch.zeros(T, B, F_, N_, dtype=torch.float16, device=device)
        b_ego = torch.zeros(T, B, E_, device=device); b_to = torch.zeros(T, B, Dt, device=device)
        ba = torch.zeros(T, B, 2, device=device); blp = torch.zeros(T, B, device=device)
        bv = torch.zeros(T + 1, B, device=device); bp = torch.zeros(T + 1, B, device=device)
        br = torch.zeros(T, B, device=device); bd = torch.zeros(T, B, device=device); bh = torch.zeros(T, B, device=device)
        prog = hits_w = hits_c = 0.0; laps = []
        tr0 = time.time()
        with torch.no_grad():
            for t in range(T):
                tnorm.update(to)
                tn = tnorm(to)
                sc_ = obs["scan"][lid]; eg_ = egov(obs)
                mean = actor(sc_.float(), eg_)
                std = log_std.exp()
                al = float(a.noise_corr)
                eps_n = al * eps_n + math.sqrt(max(0.0, 1.0 - al * al)) * torch.randn_like(eps_n) if al > 0 else torch.randn_like(eps_n)
                act = mean + std * eps_n
                b_scan[t] = sc_.half(); b_ego[t] = eg_; b_to[t] = tn; ba[t] = act
                blp[t] = torch.distributions.Normal(mean, std).log_prob(act).sum(1)
                v_t, hl_t = critic.value_and_hit(tn)
                bv[t] = v_t * (ret_rms.var.sqrt() + 1e-6); bp[t] = torch.sigmoid(hl_t)
                rew = torch.zeros(B, device=device); done = torch.zeros(B, dtype=torch.bool, device=device)
                hit_now = torch.zeros(B, dtype=torch.bool, device=device)
                act_full[lid] = act.clamp(-1, 1)
                for k in range(R):
                    obs, r_, te, trc, info = env.step(act_full)
                    comp = info["reward_components"]
                    col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
                    tch = car < 0
                    onset = tch & ~touching
                    touching = tch
                    onw = (-col / max(w_col, 1e-9)).clamp_min(0)
                    dense = r_[lid].float() - col - car
                    step_r = dense - float(a.price) * onw - float(a.car_price) * onset.float()
                    alive = (~done).float()
                    rew += step_r * alive
                    prog += float((info["progress"][lid].float().clamp_min(0) * alive).sum())
                    hits_w += float((onw * alive).sum()); hits_c += float((onset.float() * alive).sum())
                    hit_now |= ((onw > 0) | onset) & ~done
                    d_ = (te | trc)[lid]
                    touching = touching & ~d_
                    done |= d_
                    if info["lap_times"].numel():
                        laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                br[t] = rew; bd[t] = done.float(); bh[t] = hit_now.float()
                last_a = torch.where(done[:, None], torch.zeros_like(act), act.clamp(-1, 1))
                eps_n = torch.where(done[:, None], torch.randn_like(eps_n), eps_n)
                to = tobs(last_a)
            v_T, hl_T = critic.value_and_hit(tnorm(to))
            bv[T] = v_T * (ret_rms.var.sqrt() + 1e-6); bp[T] = torch.sigmoid(hl_T)
            K = int(a.hit_horizon)
            y_hit = torch.zeros(T, B, device=device); v_hit = torch.zeros(T, B, device=device)
            since = torch.full((B,), 1e9, device=device)
            for t in reversed(range(T)):
                since = torch.where(bh[t] > 0, torch.zeros_like(since), torch.where(bd[t] > 0, torch.full_like(since, 1e9), since + 1))
                y_hit[t] = (since < K).float()
                ended = torch.zeros(B, dtype=torch.bool, device=device)
                complete = torch.full((B,), t + K <= T, dtype=torch.bool, device=device)
                for k in range(t, min(T, t + K)):
                    ended |= bd[k] > 0
                v_hit[t] = (complete | ended | (y_hit[t] > 0)).float()
            shaping = torch.zeros(T, B, device=device)
            if a.hit_shaping > 0:
                phi = -float(a.hit_shaping) * float(a.price) * bp
                shaping = a.gamma * phi[1:] * (1.0 - bd) - phi[:-1]
            adv = torch.zeros(T, B, device=device); last = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                nd = 1.0 - bd[t]
                delta = br[t] + shaping[t] + a.gamma * bv[t + 1] * nd - bv[t]
                last = delta + a.gamma * a.lam * nd * last
                adv[t] = last
            ret = adv + bv[:T]
            ret_rms.update(ret.reshape(-1, 1))
            ret_n = ret / (ret_rms.var.sqrt() + 1e-6)
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        t_roll = time.time() - tr0
        N = T * B
        fs, fe, ft = b_scan.reshape(N, F_, N_), b_ego.reshape(N, E_), b_to.reshape(N, Dt)
        fa, flp, fadv, fret = ba.reshape(N, 2), blp.reshape(N), adv_n.reshape(N), ret_n.reshape(N)
        fyh, fvh = y_hit.reshape(N), v_hit.reshape(N)
        mb = N // int(a.minibatches)
        kls, cfs = [], []
        actor_on = u > int(a.critic_warmup)
        for ep in range(int(a.epochs)):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, mb):
                j = perm[i:i + mb]
                v_j, hl_j = critic.value_and_hit(ft[j])
                vf = 0.5 * (v_j - fret[j]).pow(2).mean()
                lc = a.vf * vf
                if a.hit_head > 0:
                    bce = nn.functional.binary_cross_entropy_with_logits(hl_j, fyh[j], reduction="none")
                    lc = lc + float(a.hit_head) * (bce * fvh[j]).sum() / fvh[j].sum().clamp_min(1.0)
                opt_c.zero_grad(set_to_none=True); lc.backward()
                nn.utils.clip_grad_norm_(crit_params, 1.0); opt_c.step()
                if actor_on:
                    mean = actor(fs[j].float(), fe[j])
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    lp = dist.log_prob(fa[j]).sum(1)
                    ratio = (lp - flp[j]).exp()
                    pg = -torch.min(ratio * fadv[j], ratio.clamp(1 - a.clip, 1 + a.clip) * fadv[j]).mean()
                    opt_a.zero_grad(set_to_none=True); pg.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()) + [log_std], 1.0); opt_a.step()
                    with torch.no_grad():
                        kls.append(float(((ratio - 1) - (lp - flp[j])).mean()))
                        cfs.append(float(((ratio - 1).abs() > a.clip).float().mean()))
        km = prog / 1000.0
        secs = T * R * B * env.sim.control_dt
        stats = {"update": u, "progress_ms": prog / secs, "hits_wall_prop_per_km": hits_w / max(km, 1e-3),
                 "hits_car_per_km": hits_c / max(km, 1e-3), "J_per_step": float(br.sum() / (T * R * B)),
                 "J80_per_step": float((prog - 80.0 * (hits_w + hits_c)) / (T * R * B)),
                 "laps": len(laps), "lap_mean_s": float(np.mean(laps)) if laps else None,
                 "std": log_std.exp().tolist(), "kl": float(np.mean(kls)) if kls else 0.0,
                 "clipfrac": float(np.mean(cfs)) if cfs else 0.0, "actor_on": actor_on,
                 "hit_p_pos": float((bp[:T] * y_hit * v_hit).sum() / (y_hit * v_hit).sum().clamp_min(1.0)),
                 "hit_p_neg": float((bp[:T] * (1 - y_hit) * v_hit).sum() / ((1 - y_hit) * v_hit).sum().clamp_min(1.0)),
                 "t_roll": t_roll, "t_upd": time.time() - tr0 - t_roll, "min": (time.time() - t0) / 60}
        log.write(json.dumps(stats) + "\n"); log.flush()
        if u % 5 == 0 or u == 1:
            lp_ = f"{stats['lap_mean_s']:.2f}" if laps else "-"
            print(f"[{time.strftime('%H:%M')}] 업데이트 {u}{'' if actor_on else ' (비평가만)'} | 진행 {stats['progress_ms']:.2f} m/s | "
                  f"강한 충돌 벽+상자 {stats['hits_wall_prop_per_km']:.2f} 차 {stats['hits_car_per_km']:.2f} /km | 랩 {len(laps)}개 평균 {lp_} s | "
                  f"J80 {stats['J80_per_step']:.4f} | 충돌 예측 {stats['hit_p_pos']:.2f}/{stats['hit_p_neg']:.3f} | 표준편차 "
                  f"{[round(x, 3) for x in stats['std']]} | KL {stats['kl']:.4f} | {t_roll:.1f}+{stats['t_upd']:.1f} s", flush=True)
        if u % int(a.save_every) == 0 or u == int(a.updates):
            sargs = dict(sck["args"]); sargs.update(stride=stride, repeat=R, rl=vars(a))
            torch.save({"student": actor.state_dict(), "frames": F_, "beams": N_, "ego_dim": int(sck["ego_dim"]),
                        "ego_keys": keys, "args": sargs, "iter": u, "log_std": log_std.detach(),
                        "critic": critic.state_dict(), "norm": tnorm.state_dict(), "ret_rms": ret_rms.state_dict()},
                       os.path.join(out, f"student_u{u}.pt"))
    print(f"[student_rl] {a.name} 끝", flush=True)


if __name__ == "__main__":
    main()
