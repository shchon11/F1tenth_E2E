"""라이다 쌓기에 상대 차의 움직임이 보이나 -- 지금 학생 설정에서 직접 잰다 (2026-10-29).

사용자: "이전 데이터들 너 믿지 말고 너가 직접 확인해." 10-07 기록("쌓아도 옆 접근 속도 설명력 0.28~0.38")을 근거로 쓰지 않고, 지금의
학생 관측(v1 관측 사전: 잡음 있는 1081빔 라이다를 F장, stride 스텝 간격으로 쌓음 + 자아 벡터)으로 다시 잰다.

선생(state_rl 체크포인트)의 평균 행동으로 경주 세계를 몰고, 앞쪽에 상대가 보이는 결정(0.5 < dx < 8 m, |dy| < 2.5 m)마다 학생 관측과
가장 가까운 상대의 참값(내 차 좌표계 dx, dy, 상대 속도 dvx, dvy, 시선 방향 접근 속도)을 모은 뒤, 학생과 같은 합성곱 몸통으로 회귀한다:
  1장   가장 새 라이다 한 장 + 자아
  F장   쌓은 F장 + 자아
  F장-자아  쌓은 F장만
다른 차 20 %로 시험한 R2와 RMSE. 위치(dx, dy)는 한 장으로도 잘 맞아야 함(도구 점검). 속도 R2가 F장에서 1장보다 크게 오르면 움직임
정보가 라이다에 있음 -> 학생의 늦은 감속은 학습 문제; 안 오르면 이 센서로는 상대 속도를 못 봄.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl  # noqa: E402
from f1sim.learn.state_rl import ActorCritic, RunningNorm, StateObs  # noqa: E402

NAMES = ("dx", "dy", "dvx", "dvy", "접근속도")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--warps", type=int, default=16)
    ap.add_argument("--decisions", type=int, default=2000)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-props", type=int, default=0, help="1: the race world without props (is the car confused with a prop?)")
    a = ap.parse_args()
    a.stage = "race"; a.beams = 1081
    device = torch.device(a.device)
    ck = torch.load(a.teacher, map_location=device, weights_only=False)
    t_beams = int(ck["args"].get("beams", 36)); t_scan = bool(ck["args"].get("scan_obs", 0))
    extra = dict(obs_version=1, scan_stack=int(a.frames), scan_stride=int(a.stride))
    if int(a.no_props):
        extra.update(procedural_obstacles=0.0)
    env = state_rl.build_env(a, device, extra=extra)
    from f1sim.learn.graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    base = StateObs(env)
    obs, _ = env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    ang_s = env.sim.lidar.angles.float()
    ang_t = torch.linspace(float(ang_s[0]), float(ang_s[-1]), t_beams, device=device)
    pos = (ang_t - ang_s[0]) / (ang_s[-1] - ang_s[0]) * (ang_s.numel() - 1)
    i0 = pos.floor().long().clamp(0, ang_s.numel() - 2); wgt = (pos - i0.float())[None, :]

    def tobs(last):
        o_ = base(last)
        if not t_scan:
            return o_
        sc = env.last_result.scan_true[lid].float()
        return torch.cat([o_, ((sc[:, i0] * (1 - wgt) + sc[:, i0 + 1] * wgt) / 10.0).clamp(0.0, 1.0)], 1)

    last = torch.zeros(B, 2, device=device)
    o = tobs(last)
    model = ActorCritic(o.shape[1]).to(device); model.load_state_dict(ck["model"], strict=False); model.eval()
    norm = RunningNorm(o.shape[1], device); norm.load_state_dict(ck["norm"])
    ego_keys = sorted(k for k in obs if k != "scan")
    egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in ego_keys], 1)
    F, N = obs["scan"].shape[1], obs["scan"].shape[2]
    print(f"[움직임] 학생 관측: 라이다 {F}장 x {N}빔 (간격 {a.stride} 스텝 = {a.stride * 25} ms), 자아 {egov(obs).shape[1]}차원, "
          f"자아 키 {ego_keys}", flush=True)
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    Ss, Es, Ys, Rs = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for t in range(int(a.decisions)):
            st = env.last_result.state
            dx, dy, dyaw, dvx, dvy, dist, pres = env._opp_rel(st)
            k = torch.where(pres, dist, torch.full_like(dist, 1e6)).argmin(1, keepdim=True)
            g = lambda x: x.gather(1, k)[:, 0][lid]
            dx_, dy_, dvx_, dvy_, d_ = g(dx), g(dy), g(dvx), g(dvy), g(dist)
            clos = -(dx_ * dvx_ + dy_ * dvy_) / d_.clamp_min(1e-3)
            keep = g(pres.float()) > 0
            keep &= (dx_ > 0.5) & (dx_ < 8.0) & (dy_.abs() < 2.5)
            if keep.any():
                ii = keep.nonzero().flatten()
                Ss.append(obs["scan"][lid][ii].half()); Es.append(egov(obs)[ii])
                Ys.append(torch.stack([dx_, dy_, dvx_, dvy_, clos], 1)[ii]); Rs.append(ii)
            act = model.pi(norm(o)).clamp(-1, 1)
            act_full[lid] = act
            dn = torch.zeros(B, dtype=torch.bool, device=device)
            for r in range(2):
                obs, r_, te, trc, info = env.step(act_full)
                dn |= (te | trc)[lid]
            last = torch.where(dn[:, None], torch.zeros_like(act), act)
            o = tobs(last)
            if t % 500 == 0:
                print(f"[움직임] 굴림 {t}/{a.decisions}, 표본 {sum(x.shape[0] for x in Ss)}, {time.time() - t0:.0f} s", flush=True)
    S = torch.cat(Ss); E = torch.cat(Es); Y = torch.cat(Ys); R = torch.cat(Rs)
    # geometric check, nothing learned: the newest frame's beam toward the opponent's centre should read about its distance
    # (minus the half body it hides behind); a mismatch means the scan does not show the car where the state says it is
    ang = env.sim.lidar.angles.float().to(device)
    th = torch.atan2(Y[:, 1], Y[:, 0])
    jb = (ang[None, :] - th[:, None]).abs().argmin(1)
    meas = S[torch.arange(S.shape[0], device=device), 0, jb].float() * float(env.range_max)
    dist_c = (Y[:, 0] ** 2 + Y[:, 1] ** 2).sqrt()
    err = (meas - dist_c).float()
    print(f"[움직임] 기하 점검(학습 없음): 상대 중심 쪽 빔의 거리 - 중심 거리, 중앙 {float(err.median()):+.2f} m, "
          f"|오차| < 0.5 m 비율 {float((err.abs() < 0.5).float().mean()) * 100:.0f} %, 빔이 상대보다 1 m 넘게 멀리 봄 "
          f"{float((err > 1.0).float().mean()) * 100:.0f} %", flush=True)
    test = (R % 5) == 0
    print(f"[움직임] 표본 {S.shape[0]} (시험 {int(test.sum())}), 목표 표준편차 " +
          ", ".join(f"{n} {float(Y[:, j].std()):.2f}" for j, n in enumerate(NAMES)), flush=True)
    # raw sanity: how much do the frames differ at all (the stack is not duplicates)? index 0 = newest
    fd = (S[:, 0].float() - S[:, -1].float()).abs().mean()
    print(f"[움직임] 가장 오래된 장과 새 장의 평균 차이 {float(fd):.4f} (관측 단위), 장 평균 {float(S.float().abs().mean()):.4f}", flush=True)
    mu_y, sd_y = Y[~test].mean(0), Y[~test].std(0) + 1e-6
    mu_e, sd_e = E[~test].mean(0), E[~test].std(0) + 1e-3

    class Net(nn.Module):
        """trunk "학생": the student's (state_student.Student: pooled to 32 bins of ~8.4 deg); "고해상도": the same convs
        with no pooling (~89 positions of ~3 deg) and a channel per frame difference, so a sideways step of a car at a few
        metres is not averaged away"""
        def __init__(self, f_in, e_in, trunk="학생"):
            super().__init__()
            self.trunk = trunk
            if trunk == "학생":
                self.conv = nn.Sequential(nn.Conv1d(f_in, 32, 7, stride=3), nn.ReLU(), nn.Conv1d(32, 64, 5, stride=2), nn.ReLU(),
                                          nn.Conv1d(64, 64, 5, stride=2), nn.ReLU(), nn.AdaptiveAvgPool1d(32), nn.Flatten(),
                                          nn.Linear(64 * 32, 256), nn.ReLU())
            else:
                c_in = f_in + max(f_in - 1, 0)
                self.conv = nn.Sequential(nn.Conv1d(c_in, 32, 5, stride=2, padding=2), nn.ReLU(),
                                          nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ReLU(),
                                          nn.Conv1d(32, 16, 3, stride=3, padding=1), nn.ReLU(), nn.Flatten(),
                                          nn.LazyLinear(256), nn.ReLU())
            self.e_in = e_in
            self.ego = nn.Sequential(nn.Linear(max(e_in, 1), 128), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, len(NAMES)))

        def forward(self, s, e):
            ze = self.ego(e) if self.e_in > 0 else torch.zeros(s.shape[0], 128, device=s.device)
            if self.trunk != "학생" and s.shape[1] > 1:
                s = torch.cat([s, (s[:, :1] - s[:, 1:]) * 5.0], 1)         # newest minus each older frame
            return self.head(torch.cat([self.conv(s), ze], 1))

    class EgoOnly(nn.Module):
        def __init__(self, e_in):
            super().__init__()
            self.m = nn.Sequential(nn.Linear(e_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, len(NAMES)))

        def forward(self, s, e):
            return self.m(e)

    def fit(name, frames_idx, use_ego, trunk="학생"):
        torch.manual_seed(0)
        net = (EgoOnly(E.shape[1]) if trunk == "자아만" else Net(len(frames_idx), E.shape[1] if use_ego else 0, trunk)).to(device)
        with torch.no_grad():
            net(S[:2][:, torch.tensor(frames_idx, device=device)].float(), ((E[:2] - mu_e) / sd_e).clamp(-8, 8))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        tr = (~test).nonzero().flatten(); te = test.nonzero().flatten()
        fi = torch.tensor(frames_idx, device=device)
        for it in range(int(a.iters)):
            j = tr[torch.randint(0, tr.numel(), (2048,), device=device)]
            s = S[j][:, fi].float(); e = ((E[j] - mu_e) / sd_e).clamp(-8, 8)
            loss = ((net(s, e) - (Y[j] - mu_y) / sd_y) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, te.numel(), 8192):
                j = te[i:i + 8192]
                out.append(net(S[j][:, fi].float(), ((E[j] - mu_e) / sd_e).clamp(-8, 8)) * sd_y + mu_y)
        P = torch.cat(out); T_ = Y[te]
        r2 = 1 - ((P - T_) ** 2).mean(0) / T_.var(0)
        rm = ((P - T_) ** 2).mean(0).sqrt()
        print(f"[움직임] {name:14s} | 시험 R2 " + ", ".join(f"{n} {float(r2[j]):.3f}" for j, n in enumerate(NAMES))
              + " | RMSE " + ", ".join(f"{n} {float(rm[j]):.2f}" for j, n in enumerate(NAMES)), flush=True)

    # the env's stack is newest first (gym_env: scan_hist[:, 0] = this step)
    fit("자아만(라이다 없음)", [0], True, "자아만")
    fit("학생 1장+자아", [0], True)
    fit(f"학생 {F}장+자아", list(range(F)), True)
    fit("고해상 1장+자아", [0], True, "고해상도")
    fit(f"고해상 {F}장+자아", list(range(F)), True, "고해상도")
    fit(f"고해상 {F}장", list(range(F)), False, "고해상도")


if __name__ == "__main__":
    main()
